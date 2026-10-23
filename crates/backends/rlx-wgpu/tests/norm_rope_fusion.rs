// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The per-head norm and its rotation cost one dispatch, and the numbers do
//! not move.**
//!
//! `narrow → RmsNorm(head) → RoPE` is what every transformer block walks per
//! head group; decomposed it is four dispatches over a one-row activation. The
//! folded step reads the norm's operand and writes the rope's slot, so this
//! asserts both halves:
//!
//! * the fold fires — one `norm_rope` step for a two-part (q, k) chain, and none
//!   where the norm's output has a second reader, which is the admission rule —
//!   and
//! * wgpu still matches the CPU reference elementwise, at one position and at
//!   several, so the head-row decode (`pos = row / heads`) and both parts of a
//!   merged dispatch are exercised.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

mod common;

const F: DType = DType::F32;
const HD: usize = 128;
const Q_HEADS: usize = 4;
const K_HEADS: usize = 2;
const EPS: f32 = 1e-6;

fn tables(rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HD / 2;
    let mut cos = vec![0f32; rows * half];
    let mut sin = vec![0f32; rows * half];
    for p in 0..rows {
        for i in 0..half {
            let theta = p as f32 / 10_000f32.powf(2.0 * i as f32 / HD as f32);
            cos[p * half + i] = theta.cos();
            sin[p * half + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 23) as f32 - 11.0) * 0.05)
        .collect()
}

/// One head group: `[1, rows, heads·hd] → [rows·heads, hd] → RmsNorm → reshape →
/// RoPE`.
fn group(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    rows: usize,
    heads: usize,
    gamma: rlx_ir::NodeId,
    beta: rlx_ir::NodeId,
    cos: rlx_ir::NodeId,
    sin: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let r = g.reshape_(x, vec![(rows * heads) as i64, HD as i64]);
    let r = g.rms_norm(r, gamma, beta, EPS);
    let y = g.reshape_(r, vec![1, rows as i64, (heads * HD) as i64]);
    g.rope(y, cos, sin, HD)
}

fn chain(rows: usize) -> Graph {
    let mut g = Graph::new("norm_rope");
    let qx = g.input("qx", Shape::new(&[1, rows, Q_HEADS * HD], F));
    let kx = g.input("kx", Shape::new(&[1, rows, K_HEADS * HD], F));
    let gamma_q = g.param("gamma_q", Shape::new(&[HD], F));
    let gamma_k = g.param("gamma_k", Shape::new(&[HD], F));
    let beta = g.param("beta", Shape::new(&[HD], F));
    let cos = g.param("cos", Shape::new(&[rows, HD / 2], F));
    let sin = g.param("sin", Shape::new(&[rows, HD / 2], F));
    let q = group(&mut g, qx, rows, Q_HEADS, gamma_q, beta, cos, sin);
    let k = group(&mut g, kx, rows, K_HEADS, gamma_k, beta, cos, sin);
    g.set_outputs(vec![q, k]);
    g
}

fn params(rows: usize) -> Vec<(&'static str, Vec<f32>)> {
    let (cos, sin) = tables(rows);
    vec![
        ("gamma_q", fill(HD, 1)),
        ("gamma_k", fill(HD, 2)),
        ("beta", fill(HD, 7)),
        ("cos", cos),
        ("sin", sin),
    ]
}

fn compile(device: Device, graph: Graph, rows: usize) -> CompiledGraph {
    let mut c = Session::new(device).compile(graph);
    for (name, values) in params(rows) {
        c.set_param(name, &values);
    }
    c.finalize_params();
    c
}

fn steps(c: &CompiledGraph) -> Vec<&'static str> {
    c.schedule_step_names()
        .expect("the wgpu backend reports its schedule")
}

#[test]
fn the_norm_and_rope_fold_into_one_step_and_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    for rows in [1usize, 3] {
        let q = fill(rows * Q_HEADS * HD, 3);
        let k = fill(rows * K_HEADS * HD, 5);
        let inputs = [("qx", q.as_slice()), ("kx", k.as_slice())];
        let want = compile(Device::Cpu, chain(rows), rows).run(&inputs);
        let mut gpu = compile(Device::Gpu, chain(rows), rows);
        let names = steps(&gpu);
        let got = gpu.run(&inputs);
        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "rows={rows}: the folded norm+rope chain drifts from the CPU by {worst:e}"
        );
        assert_eq!(
            names.iter().filter(|n| **n == "norm_rope").count(),
            1,
            "rows={rows}: the q and k norm+rope pairs are one dispatch ({names:?})"
        );
        assert_eq!(
            names.iter().filter(|n| **n == "rope").count(),
            0,
            "rows={rows}: no rope step survives the fold ({names:?})"
        );
        assert_eq!(
            names.iter().filter(|n| **n == "layer_norm").count(),
            0,
            "rows={rows}: no norm step survives the fold ({names:?})"
        );
    }
}

/// A fused projection's row cut into its parts: one dispatch for the pieces
/// where the decompose emits one `narrow` each, and the same bytes either way.
#[test]
fn the_split_that_feeds_a_pair_rides_in_its_dispatch() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // The shape the predictor's block makes: one `[1, q + k + v]` row, its q and
    // k head-normalized and rotated, its v read as it stands.
    let rows = 1;
    let (q, k, v) = (Q_HEADS * HD, K_HEADS * HD, 3 * HD);
    let n = q + k + v;
    let mut g = Graph::new("qkv_norm_rope");
    let x = g.input("x", Shape::new(&[1, n], F));
    let gamma_q = g.param("gamma_q", Shape::new(&[HD], F));
    let gamma_k = g.param("gamma_k", Shape::new(&[HD], F));
    let beta = g.param("beta", Shape::new(&[HD], F));
    let cos = g.param("cos", Shape::new(&[rows, HD / 2], F));
    let sin = g.param("sin", Shape::new(&[rows, HD / 2], F));
    let narrow = |g: &mut Graph, start, len| g.narrow_(x, 1, start, len);
    // The model's node order: the split's three pieces first, then the pair's
    // norms and rotations (the split detector requires the pieces adjacent).
    let qp = narrow(&mut g, 0, q);
    let kp = narrow(&mut g, q, k);
    let vy = narrow(&mut g, q + k, v);
    let rot = |g: &mut Graph, p, len: usize, gamma| {
        let r = g.reshape_(p, vec![(rows * len / HD) as i64, HD as i64]);
        let r = g.rms_norm(r, gamma, beta, EPS);
        let y = g.reshape_(r, vec![1, rows as i64, len as i64]);
        g.rope(y, cos, sin, HD)
    };
    let qy = rot(&mut g, qp, q, gamma_q);
    let ky = rot(&mut g, kp, k, gamma_k);
    g.set_outputs(vec![qy, ky, vy]);
    let data = fill(n, 6);
    let want = compile(Device::Cpu, g.clone(), rows).run(&[("x", &data)]);
    let mut gpu = compile(Device::Gpu, g, rows);
    let names = steps(&gpu);
    let got = gpu.run(&[("x", &data)]);
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        1,
        "the split and the pair are one dispatch ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "narrow_split").count(),
        0,
        "the split the pair absorbed keeps no step ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "narrow").count(),
        0,
        "no piece keeps a copy of its own ({names:?})"
    );
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.len(), b.len(), "output {i} length");
        let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-5, "output {i} drifts from the CPU by {worst:e}");
    }
    // The copied piece is the row's own bytes, not a re-computation.
    assert_eq!(got[2], &data[q + k..], "the leftover piece is a copy");
}

#[test]
fn the_pieces_of_one_row_are_cut_in_one_dispatch() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // The shape the predictor's fused qkv makes: one `[1, 2048 + 512 + 512]` row
    // read as q, k and v. `n` is not a multiple of the workgroup.
    let (q, k, v) = (2048usize, 512, 508);
    let n = q + k + v;
    let mut g = Graph::new("narrow_split");
    let x = g.input("x", Shape::new(&[1, n], F));
    let a = g.narrow_(x, 1, 0, q);
    let b = g.narrow_(x, 1, q, k);
    let c = g.narrow_(x, 1, q + k, v);
    g.set_outputs(vec![a, b, c]);
    let data = fill(n, 2);
    let want = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &data)]);
    let mut gpu = Session::new(Device::Gpu).compile(g);
    let names = steps(&gpu);
    let got = gpu.run(&[("x", &data)]);
    assert_eq!(
        names.iter().filter(|n| **n == "narrow_split").count(),
        1,
        "the three pieces are one dispatch ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "narrow").count(),
        0,
        "no piece keeps a dispatch of its own ({names:?})"
    );
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a, b, "piece {i} is not the row's own bytes");
    }
}

/// A chain of two layout views between the norm and the rotation is still the
/// chain the admission admits; reading it in the wrong direction would decline it.
#[test]
fn a_chain_of_views_still_folds() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let rows = 2;
    let mut g = Graph::new("norm_rope_views");
    let x = g.input("x", Shape::new(&[1, rows, Q_HEADS * HD], F));
    let gamma = g.param("gamma_q", Shape::new(&[HD], F));
    let beta = g.param("beta", Shape::new(&[HD], F));
    let cos = g.param("cos", Shape::new(&[rows, HD / 2], F));
    let sin = g.param("sin", Shape::new(&[rows, HD / 2], F));
    let r = g.reshape_(x, vec![(rows * Q_HEADS) as i64, HD as i64]);
    let r = g.rms_norm(r, gamma, beta, EPS);
    let a = g.reshape_(r, vec![1, rows as i64, (Q_HEADS * HD) as i64]);
    let b = g.reshape_(a, vec![1, rows as i64, (Q_HEADS * HD) as i64]);
    let rot = g.rope(b, cos, sin, HD);
    g.set_outputs(vec![rot]);
    let data = fill(rows * Q_HEADS * HD, 4);
    let want = compile(Device::Cpu, g.clone(), rows).run(&[("x", &data)]);
    let mut gpu = compile(Device::Gpu, g, rows);
    let names = steps(&gpu);
    let got = gpu.run(&[("x", &data)]);
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        1,
        "a two-view chain is still one norm+rope dispatch ({names:?})"
    );
    let worst = got[0]
        .iter()
        .zip(&want[0])
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-5, "the two-view chain drifts by {worst:e}");
}

/// A view of the norm's output that is also a graph output reads the slot the fold
/// would leave unwritten, so the fold must decline and the norm keep its step.
#[test]
fn a_view_of_the_norm_that_is_an_output_is_not_folded() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let rows = 1;
    let mut g = Graph::new("norm_rope_output_view");
    let x = g.input("x", Shape::new(&[1, rows, Q_HEADS * HD], F));
    let gamma = g.param("gamma", Shape::new(&[HD], F));
    let beta = g.param("beta", Shape::new(&[HD], F));
    let cos = g.param("cos", Shape::new(&[rows, HD / 2], F));
    let sin = g.param("sin", Shape::new(&[rows, HD / 2], F));
    let r = g.reshape_(x, vec![(rows * Q_HEADS) as i64, HD as i64]);
    let r = g.rms_norm(r, gamma, beta, EPS);
    let y = g.reshape_(r, vec![1, rows as i64, (Q_HEADS * HD) as i64]);
    let rot = g.rope(y, cos, sin, HD);
    // The norm's own layout is an output, so `y`'s bytes must be written.
    g.set_outputs(vec![rot, y]);
    let names = steps(&Session::new(Device::Gpu).compile(g));
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        0,
        "a norm whose view is a graph output is not folded ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "layer_norm").count(),
        1,
        "the norm keeps the step that writes the output ({names:?})"
    );
}

#[test]
fn a_tapped_norm_keeps_its_own_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let rows = 1;
    let mut g = Graph::new("norm_rope_tapped");
    let x = g.input("x", Shape::new(&[1, rows, Q_HEADS * HD], F));
    let gamma = g.param("gamma", Shape::new(&[HD], F));
    let beta = g.param("beta", Shape::new(&[HD], F));
    let c = g.param("cos", Shape::new(&[rows, HD / 2], F));
    let s = g.param("sin", Shape::new(&[rows, HD / 2], F));
    let r = g.reshape_(x, vec![(rows * Q_HEADS) as i64, HD as i64]);
    let r = g.rms_norm(r, gamma, beta, EPS);
    let y = g.reshape_(r, vec![1, rows as i64, (Q_HEADS * HD) as i64]);
    let rot = g.rope(y, c, s, HD);
    let tap = g.reshape_(r, vec![1, (rows * Q_HEADS * HD) as i64]);
    g.set_outputs(vec![rot, tap]);
    let names = steps(&Session::new(Device::Gpu).compile(g));
    assert_eq!(
        names.iter().filter(|n| **n == "rope").count(),
        1,
        "a tapped norm keeps the rope's own step ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        0,
        "a tapped norm is not folded ({names:?})"
    );
}

/// The switch that keys the per-head norms to the host is a routing decision, not
/// just a performance one, so the fold must stand down with it: the fold suppresses
/// the norm's own step, and a hosted norm keeps it.
///
/// The lever is process-global, so the child gets it alone — the same re-exec the
/// crate's other switch pins use.
const KNOB_MARKER: &str = "RLX_NORM_ROPE_KNOB_FIXTURE";
const KNOB_FIXTURE: &str = "host_norm";

#[test]
fn a_hosted_norm_stands_the_fold_down() {
    if std::env::var(KNOB_MARKER).ok().as_deref() != Some(KNOB_FIXTURE) {
        let (ok, stderr) = common::rerun_self(
            "a_hosted_norm_stands_the_fold_down",
            KNOB_MARKER,
            KNOB_FIXTURE,
            &common::CLEARED_SWITCHES,
            // `RLX_WGPU_GPU_NORM` takes precedence inside `should_pack_host_op`,
            // so the host decision is only this case's when both are pinned.
            &[("RLX_WGPU_HOST_NORM", "1"), ("RLX_WGPU_GPU_NORM", "0")],
        );
        assert!(ok, "the hosted-norm child failed:\n{stderr}");
        return;
    }
    eprintln!("{KNOB_MARKER} {KNOB_FIXTURE}");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let names = steps(&Session::new(Device::Gpu).compile(chain(1)));
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        0,
        "RLX_WGPU_HOST_NORM=1 must keep the norms' own steps ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "host_op").count(),
        2,
        "the q and k norms are the two hosted steps ({names:?})"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "rope").count(),
        2,
        "the rotations stay on the device ({names:?})"
    );
}
