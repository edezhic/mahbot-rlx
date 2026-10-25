// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Gather(table, Add(idx, c))` folded into one gather step, with `c` applied
//! in-kernel.
//!
//! The code predictor adds a constant offset to its index before every
//! embedding gather, which is a full dispatch per frame over a one-element
//! tensor. The fold suppresses that `Add` and lets the gather kernel add the
//! constant to the f32 index before truncating it, in the same order the graph
//! did.
//!
//! What this file pins:
//!
//! **The step disappears on both gather routes.** The whole-arena
//! `Step::Gather` and the split-binding `Step::GatherSplit` each leave no
//! `binary` step, and both return the CPU backend's answer. The split route is
//! forced with `RLX_WGPU_GATHER_SPLIT=1` on the small fixture table.
//!
//! **The switch restores the `Add`.** `RLX_WGPU_NO_GATHER_INDEX_BIAS=1` is read
//! at compile time, so one process compiles both arms and the pin cannot be
//! satisfied by a stale config snapshot.
//!
//! **An exposed `Add` keeps its step.** Folding it would suppress the only
//! write to that output slot.
//!
//! The index is in range before and after the offset (`2 + 4 = 6` of 8 rows),
//! so `+4` selects a different row from `2` and a dropped bias cannot pass the
//! parity bar. The process-global env override is why each test takes
//! [`route_lock`].
mod common;

use std::collections::BTreeMap;

use common::{ramp, route_lock};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// The switch that stands the fold down. Read live by the lowering.
const NO_FOLD: &str = "RLX_WGPU_NO_GATHER_INDEX_BIAS";
/// The switch that forces the split-binding gather on a small table.
const FORCE_SPLIT: &str = "RLX_WGPU_GATHER_SPLIT";
const VOCAB: usize = 8;
const DIM: usize = 16;
/// The raw index and the constant the Add applies.
const IDX: f32 = 2.0;
const OFFSET: f32 = 4.0;

/// The fixture table: a deterministic ±1 ramp over `[VOCAB, DIM]`.
fn table() -> Vec<f32> {
    ramp(VOCAB, DIM, 0.37)
}

/// `out = Gather(table[VOCAB, DIM], Add(idx, OFFSET))`, with the `Add` also a
/// graph output when asked.
fn graph(expose_the_add: bool) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gather_index_bias");
    let table = g.param("table", Shape::new(&[VOCAB, DIM], f));
    let idx = g.input("idx", Shape::new(&[1], f));
    let offset = g.add_node(
        Op::Constant {
            data: OFFSET.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], f),
    );
    let biased = g.add(idx, offset);
    let out = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, biased],
        Shape::new(&[1, DIM], f),
    );
    g.set_outputs(if expose_the_add {
        vec![out, biased]
    } else {
        vec![out]
    });
    g
}

/// One device arm: its step kinds, and every output. The two switches are
/// installed before `compile` — the lowering reads them then — and both are
/// stated explicitly, so an ambient value cannot make an arm be mislabelled.
fn run_gpu(
    g: Graph,
    no_fold: bool,
    force_split: bool,
) -> (BTreeMap<&'static str, usize>, Vec<Vec<f32>>) {
    let _switches = switches(no_fold, force_split);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("table", &table());
    let kinds = exe.step_kind_counts();
    let outs = exe.run(&[("idx", &[IDX])]);
    (kinds, outs)
}

/// The two lower-time switches, installed for the guard's lifetime.
fn switches(no_fold: bool, force_split: bool) -> rlx_ir::env::RuntimeOverrides {
    let on = |b: bool| if b { "1" } else { "0" };
    rlx_ir::env::RuntimeOverrides::install([(NO_FOLD, on(no_fold)), (FORCE_SPLIT, on(force_split))])
}

/// The same graph on the CPU backend, as this file's reference.
fn run_cpu(g: Graph) -> Vec<Vec<f32>> {
    let mut s = Session::new(Device::Cpu).compile(g);
    s.set_param("table", &table());
    s.run(&[("idx", &[IDX])])
}

/// Every output of `got` is bit-identical to the CPU backend's: a gather
/// copies whole rows, so both backends must produce the same bytes.
fn assert_matches_cpu(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: output count");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a, b, "{what}: output {i} is not the CPU backend's");
    }
}

/// The product's own compile path — `Session` → the compile pipeline's fusion
/// passes → wgpu — keeps the fold. The raw-IR arms above compile the lowering
/// directly; this one proves the pipeline does not turn the pair into another
/// step the detector cannot see.
#[test]
fn the_pipeline_keeps_the_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();
    let _switches = switches(false, false);

    let mut s = Session::new(Device::Gpu).compile(graph(false));
    s.set_param("table", &table());
    let names = s.schedule_step_names().expect("wgpu reports its schedule");
    let got = s.run(&[("idx", &[IDX])]);

    assert_eq!(
        names.iter().filter(|n| **n == "gather").count(),
        1,
        "the gather must stay one step: {names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "binary").count(),
        0,
        "the Add belongs to the gather's read after the pipeline: {names:?}"
    );
    assert_matches_cpu(&got, &run_cpu(graph(false)), "pipeline gather");
}

/// The Add's step is gone on both gather routes: one gather step, no `binary`,
/// and the numbers are the CPU backend's.
#[test]
fn the_index_offset_folds_into_one_gather_step_on_both_routes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();
    let want = run_cpu(graph(false));

    let (whole, got) = run_gpu(graph(false), false, false);
    assert_eq!(
        whole.get("gather").copied().unwrap_or(0),
        1,
        "the gather must be one step; schedule = {whole:?}"
    );
    assert_eq!(
        whole.get("binary").copied().unwrap_or(0),
        0,
        "the `Add(idx, 4)` belongs to the gather's read; schedule = {whole:?}"
    );
    assert_matches_cpu(&got, &want, "folded gather");

    let (split, got) = run_gpu(graph(false), false, true);
    assert_eq!(
        split.get("gather_split").copied().unwrap_or(0),
        1,
        "the forced split route is one step; schedule = {split:?}"
    );
    assert_eq!(
        split.get("binary").copied().unwrap_or(0),
        0,
        "the split gather applies the offset in-kernel too; schedule = {split:?}"
    );
    assert_matches_cpu(&got, &want, "folded gather, split route");
}

/// Under the switch the Add keeps its own step, and the numbers still match.
#[test]
fn the_switch_brings_the_add_step_back() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();
    let want = run_cpu(graph(false));

    let (kinds, got) = run_gpu(graph(false), true, false);
    assert_eq!(
        kinds.get("gather").copied().unwrap_or(0),
        1,
        "the gather is still one step; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        1,
        "`{NO_FOLD}=1` must keep the `Add`'s step; schedule = {kinds:?}"
    );
    assert_matches_cpu(&got, &want, "unfolded gather");
}

/// An `Add` that is also a graph output keeps its step: folding it would
/// suppress the only write to that slot. Both outputs are still the CPU
/// backend's.
#[test]
fn an_add_that_is_an_output_keeps_its_step() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();
    let want = run_cpu(graph(true));

    let (kinds, got) = run_gpu(graph(true), false, false);
    assert_eq!(
        kinds.get("gather").copied().unwrap_or(0),
        1,
        "the gather is still one step; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        1,
        "an exposed `Add` keeps its own step; schedule = {kinds:?}"
    );
    assert_matches_cpu(&got, &want, "exposed add");
}
