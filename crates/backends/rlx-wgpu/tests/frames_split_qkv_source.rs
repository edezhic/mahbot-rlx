// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A framing whose source is a tensor the *split-QKV* fold rewrites refuses the
//! mel framing fold.
//!
//! The split-QKV fold elides a `FusedMatMulBiasAct`'s step, replacing it with
//! `MatmulQkv`, which writes Q/K/V into the *narrow* slots and never the parent's
//! own slot. Three last-axis narrows of one parent, re-joined by a concat on axis 0,
//! are exactly the framing fold's pattern — so a framing folded over such a source
//! would read a slot nothing writes (`rlx-lightbeam`'s GRU gate projection is that
//! shape in the wild). `detect_frames_pattern` refuses it; this file pins that.
//!
//! Its own test binary on purpose, as `frames_concat_host_knob.rs` is: `RLX_ARENA_NO_REUSE`
//! is a code override (`rlx_ir::env::set`), which is process-global. This fixture's
//! class makes the isolation necessary rather than convenient — the source's step is
//! replaced, so the planner frees the source at its last reader and hands those very
//! bytes to the destination, which puts the graph in the *live-range* clause's case
//! as well (see `frames_concat_fusion.rs`). With one slot per tensor no other clause
//! can be the one that decides, so this file reads this clause and nothing else.

mod common;

use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn a_source_the_split_qkv_fold_rewrites_refuses_the_framing() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    common::neutralise_device_route_env();
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "0");
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
    let (hidden, gate) = (24usize, 8usize);
    let f = DType::F32;
    let mut g = Graph::new("frames_over_split_qkv");
    let x = g.input("x", Shape::new(&[1, hidden], f));
    let w = g.param("w", Shape::new(&[hidden, hidden], f));
    let b = g.param("b", Shape::new(&[hidden], f));
    let parent = g.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![x, w, b],
        Shape::new(&[1, hidden], f),
    );
    // Bare `narrow` rows: a reshape between the narrow and the concat would only
    // give this file's plan mirror something to rewire.
    let rows: Vec<NodeId> = (0..3)
        .map(|i| g.narrow_(parent, 1, i * gate, gate))
        .collect();
    let frames = g.concat_(rows, 0);
    g.set_outputs(vec![frames]);

    let census = WgpuExecutable::compile(g.clone()).step_kind_counts();
    assert_eq!(
        census.get("matmul_qkv"),
        Some(&1),
        "premise: this graph is the split-QKV class — the parent's step must be replaced, or the \
         source below is materialized and the pin is about nothing; census {census:?}"
    );
    let slots = common::planned_slots(&g, &[parent, frames]);
    let (s, d) = (
        slots[0].expect("premise: the parent has a slot"),
        slots[1].expect("premise: the destination has a slot"),
    );
    assert!(
        d.0 + d.1 <= s.0 || s.0 + s.1 <= d.0,
        "premise: with one slot per tensor the destination cannot cover the source's bytes, so \
         the live-range clause is not the one refusing here; src {}..{}, destination {}..{}",
        s.0,
        s.0 + s.1,
        d.0,
        d.0 + d.1
    );
    assert_eq!(
        census.get("frames"),
        None,
        "a source whose own step was replaced by a fold that writes elsewhere must not fold; \
         census {census:?}"
    );

    let sig = common::framing_signal(hidden);
    let wv = common::framing_signal(hidden * hidden);
    let bv = common::framing_signal(hidden);
    let want = {
        let mut exe = Session::new(Device::Cpu).compile(g.clone());
        exe.set_param("w", &wv);
        exe.set_param("b", &bv);
        exe.run(&[("x", &sig)]).remove(0)
    };
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &wv);
    exe.set_param("b", &bv);
    let got = exe.run(&[("x", &sig)]).remove(0);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let dd = common::max_abs_diff(&got, &want);
    assert!(
        scale > 0.0 && dd.is_finite() && dd / scale < 1e-4,
        "the refused framing must be the graph's own maths against the CPU device; max|Δ| = \
         {dd:.3e} against values up to {scale:.3e}",
    );
    assert!(
        got.iter().any(|v| *v != 0.0),
        "the refused framing read all zeros — the signature of a slot nothing wrote"
    );
}
