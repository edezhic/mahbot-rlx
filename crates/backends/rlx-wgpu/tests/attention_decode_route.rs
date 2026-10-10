// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The decode-shaped attention route on wgpu: what it costs per context
//! position, and (in `decode_route_follows_the_extent_patched_down`) that it is
//! chosen from the length the run actually walks.
//!
//! The measured shape is a single query row over a cached key/value sequence —
//! a decode step — with `LAYERS` attention ops chained through one query, so the
//! timing is a whole model body's attention and reaches the serial walk's
//! per-layer cost rather than the submission floor.
//!
//! Several tests here bracket the process-global route counters, so they take
//! [`route_lock`] instead of living in one single-test binary. The timing line is
//! printed rather than asserted — the property it reports is adapter-dependent —
//! while the route counts, which is what a regression moves first, are asserted.
mod common;

use common::max_abs_diff;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::kernels::{
    ATTENTION_DECODE_MAX_HEAD_DIM, ATTENTION_DECODE_MIN_KEYS, AttentionRoutes,
    attention_decode_eligible, attention_route_counts, reset_attention_route_counts,
};
use std::sync::{Mutex, MutexGuard};

/// The route counters are process-global, so tests in this binary that bracket
/// one (`reset` … `run` … `read`) must not run next to each other.
static ROUTE_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn route_lock() -> MutexGuard<'static, ()> {
    ROUTE_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

/// Head count / width of the 1.7B-class decode body the route targets.
const NH: usize = 16;
const HD: usize = 128;
/// A body's layer count: the chain is measured whole and divided by it.
const LAYERS: usize = 28;
/// A long context and a short one, plus a prefill-shaped row count for the
/// no-regression band.
const DECODE_LK: usize = 512;
const DECODE_LK_SHORT: usize = 104;
const PREFILL_ROWS: usize = 64;
/// Best-of-`REPS` timing loop; the counts asserted in
/// `cost_per_position_per_layer` are `REPS * LAYERS`.
const REPS: usize = 5;

/// `layers` attention ops chained through the query, all reading one K/V.
///
/// `rows == 1` is a decode step (the shape the decode route serves);
/// `rows > 1` is the prefill-shaped path, whose cost must not move.
fn chain(
    rows: usize,
    lk: usize,
    layers: usize,
    nh: usize,
    hd: usize,
    mask_kind: MaskKind,
) -> Graph {
    let dim = nh * hd;
    let f = DType::F32;
    let mut g = Graph::new("attn_chain");
    let mut q = g.input("q", Shape::new(&[1, rows, dim], f));
    let k = g.input("k", Shape::new(&[1, lk, dim], f));
    let v = g.input("v", Shape::new(&[1, lk, dim], f));
    for _ in 0..layers {
        q = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, rows, dim], f),
        );
    }
    g.set_outputs(vec![q]);
    g
}

/// As [`chain`], with a mask input whose shape is the one the CPU reference
/// reads: `[1, lk]` binary key padding for [`MaskKind::Custom`], `[1, nh, rows,
/// lk]` additive per-head scores for [`MaskKind::Bias`].
fn masked_chain(
    rows: usize,
    lk: usize,
    layers: usize,
    nh: usize,
    hd: usize,
    kind: MaskKind,
) -> Graph {
    let dim = nh * hd;
    let f = DType::F32;
    let mut g = Graph::new("attn_chain_masked");
    let mut q = g.input("q", Shape::new(&[1, rows, dim], f));
    let k = g.input("k", Shape::new(&[1, lk, dim], f));
    let v = g.input("v", Shape::new(&[1, lk, dim], f));
    let mask = g.input(
        "mask",
        match kind {
            MaskKind::Bias => Shape::new(&[1, nh, rows, lk], f),
            _ => Shape::new(&[1, lk], f),
        },
    );
    for _ in 0..layers {
        q = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: kind,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, mask],
            Shape::new(&[1, rows, dim], f),
        );
    }
    g.set_outputs(vec![q]);
    g
}

/// The mask values for [`masked_chain`], at the length its shape implies.
fn mask_data(kind: MaskKind, rows: usize, lk: usize, nh: usize) -> Vec<f32> {
    match kind {
        // ~14% of positions padding-masked out.
        MaskKind::Custom => (0..lk)
            .map(|i| if i % 7 == 0 { 0.0 } else { 1.0 })
            .collect(),
        // Additive position bias, plus a per-head term so a kernel that
        // collapsed the head axis would be caught.
        _ => (0..nh * rows * lk)
            .map(|i| {
                let s = (i % lk) as f32;
                let h = (i / lk) % nh;
                (s * 0.02 + h as f32 * 0.05).sin() * 0.5
            })
            .collect(),
    }
}

fn data(n: usize, step: f32, cos: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 * step;
            if cos { x.cos() } else { x.sin() }
        })
        .collect()
}

/// Best-of-`REPS` whole-chain timing in µs per context position per layer, plus
/// the submission and wait counts of one such run and the attention route that
/// run took (both counters reset before the timed loop).
fn cost(rows: usize, lk: usize) -> (f64, u64, u64, AttentionRoutes) {
    const WARMUP: usize = 3;
    let dim = NH * HD;
    let g = chain(rows, lk, LAYERS, NH, HD, MaskKind::Causal);
    let mut c = Session::new(Device::Gpu).compile(g);
    let q = data(rows * dim, 0.011, false);
    let k = data(lk * dim, 0.013, true);
    let v = data(lk * dim, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    for _ in 0..WARMUP {
        let _ = c.run(&inputs).remove(0);
    }
    rlx_wgpu::reset_device_counts();
    reset_attention_route_counts();
    let mut best_us = f64::MAX;
    for _ in 0..REPS {
        let start = rlx_runtime::Tick::now();
        let _ = c.run(&inputs).remove(0);
        best_us = best_us.min(rlx_runtime::Tick::now().elapsed_us(start));
    }
    let submits = rlx_wgpu::device_submit_count();
    let waits = rlx_wgpu::device_wait_count();
    let routes = attention_route_counts();
    (best_us / (LAYERS * lk) as f64, submits, waits, routes)
}

/// The multi-row path's numbers, which the route switch must not move: this is
/// the shape `cost_per_position_per_layer` times as `prefill`, and nothing else
/// in the crate checks a multi-row MHA chain at `head_dim` 128.
fn prefill_chain_matches_cpu(rows: usize, lk: usize, layers: usize) {
    let dim = NH * HD;
    let g = chain(rows, lk, layers, NH, HD, MaskKind::Causal);
    let q = data(rows * dim, 0.011, false);
    let k = data(lk * dim, 0.013, true);
    let v = data(lk * dim, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    let cpu = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&inputs)
        .remove(0);
    let gpu = Session::new(Device::Gpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!("[rlx-attn-decode] rows={rows} lk={lk} layers={layers} vs CPU: max_abs={d:.6e}");
    // Measured on this shape family: 7.5e-8 at rows=64 — a max-abs bar, unlike a
    // relative one, does not blow up when the diffuse softmax leaves the output
    // small (|out| ~ 7e-3 at lk=512).
    assert!(d < 1e-6, "attention chain rows={rows} lk={lk}: max_abs {d}");
}

#[test]
fn prefill_path_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // The decode shapes' parity lives in `attention_long_seq_precision` (Causal)
    // and in the mask sweep below (all five kinds); this is the timed multi-row
    // reference the cost rows are judged against.
    let _guard = route_lock();
    prefill_chain_matches_cpu(PREFILL_ROWS, 128, 2);
}

/// Rank-4 `[B, S, H, D]` decode (the EEG-DINO layout): a one-row rank-4 step
/// reads the same per-tensor strides the serial kernel does, so it must take the
/// same route and reach the CPU reference. Covered here because the layout, not
/// the query-row count, is what rank-4 changes.
#[test]
fn decode_route_rank4_bshd_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const B: usize = 2;
    const LK: usize = 186;
    const NH4: usize = 4;
    const HD4: usize = 64;
    let f = DType::F32;
    // MHA on the causal branch, and grouped KV heads without a mask.
    for (nkv, kind) in [(NH4, MaskKind::Causal), (2, MaskKind::None)] {
        let mut g = Graph::new("attn_bshd_decode");
        let q = g.input("q", Shape::new(&[B, 1, NH4, HD4], f));
        let k = g.input("k", Shape::new(&[B, LK, nkv, HD4], f));
        let v = g.input("v", Shape::new(&[B, LK, nkv, HD4], f));
        let y = g.add_node(
            Op::Attention {
                num_heads: NH4,
                head_dim: HD4,
                v_head_dim: None,
                mask_kind: kind,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[B, 1, NH4, HD4], f),
        );
        g.set_outputs(vec![y]);
        let qd = data(B * NH4 * HD4, 0.011, false);
        let kd = data(B * LK * nkv * HD4, 0.013, true);
        let vd = data(B * LK * nkv * HD4, 0.017, false);
        let inputs: Vec<(&str, &[f32])> = vec![("q", &qd), ("k", &kd), ("v", &vd)];
        reset_attention_route_counts();
        let gpu = Session::new(Device::Gpu)
            .compile(g.clone())
            .run(&inputs)
            .remove(0);
        let routes = attention_route_counts();
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-bshd] batch={B} nh={NH4} nkv={nkv} head_dim={HD4} lk={LK} {kind:?} \
             decode routes={routes:?} vs CPU: max_abs={d:.6e}"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 1,
                serial: 0,
            },
            "rank-4 decode nkv={nkv} {kind:?}: the decode kernel must serve the step"
        );
        assert!(
            d < 1e-6,
            "rank-4 decode nkv={nkv} {kind:?}: max_abs {d} vs CPU"
        );
    }
}

#[test]
fn cost_per_position_per_layer() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    // `serial-1row` is the pre-change baseline this build can still produce: one
    // query row at the largest key length that is not decode-shaped, so the same
    // harness measures the serial walk on a single row. The serial form's cost is
    // flat in key length per position (the `prefill` row below walks 512 keys for
    // the same figure), so this is the number the decode rows are compared to.
    for (label, rows, lk) in [
        ("serial-1row", 1usize, ATTENTION_DECODE_MIN_KEYS as usize),
        ("decode", 1, DECODE_LK_SHORT),
        ("decode", 1, DECODE_LK),
        ("prefill", PREFILL_ROWS, DECODE_LK),
    ] {
        let (us, submits, waits, routes) = cost(rows, lk);
        eprintln!(
            "[rlx-attn-cost] {label} rows={rows} heads={NH} head_dim={HD} lk={lk} \
             layers={LAYERS} mask=causal: {us:.3} us/position/layer \
             (submits={submits} waits={waits}) routes={routes:?}"
        );
        if label == "decode" {
            assert_eq!(
                routes,
                AttentionRoutes {
                    decode: REPS as u64 * LAYERS as u64,
                    serial: 0,
                },
                "{label} rows={rows} lk={lk}: every layer must take the decode kernel"
            );
        } else {
            assert_eq!(
                routes,
                AttentionRoutes {
                    decode: 0,
                    serial: REPS as u64 * LAYERS as u64,
                },
                "{label} rows={rows} lk={lk}: the serial path must keep its kernel"
            );
        }
    }
}

/// Compiled at `UPPER` keys, run at a shorter extent by the same hint the
/// bucketed compile cache sets (`set_active_extent`). The route must follow the
/// patch: 512 keys is above the decode threshold and takes the decode form, 48
/// keys is below it and must fall back to the serial kernel even though the
/// graph was lowered for 1024. Both results are checked against the CPU on the
/// same graph, where the mask makes the padded tail inert.
#[test]
fn decode_route_follows_the_extent_patched_down() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const UPPER: usize = 1024;
    const LAYERS_PATCHED: usize = 2;
    let dim = NH * HD;
    for (actual, want) in [
        (
            512usize,
            AttentionRoutes {
                decode: LAYERS_PATCHED as u64,
                serial: 0,
            },
        ),
        (
            48,
            AttentionRoutes {
                decode: 0,
                serial: LAYERS_PATCHED as u64,
            },
        ),
    ] {
        let g = masked_chain(1, UPPER, LAYERS_PATCHED, NH, HD, MaskKind::Custom);
        let q = data(dim, 0.011, false);
        let k = data(UPPER * dim, 0.013, true);
        let v = data(UPPER * dim, 0.017, false);
        let mask: Vec<f32> = (0..UPPER)
            .map(|i| if i < actual { 1.0 } else { 0.0 })
            .collect();
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v), ("mask", &mask)];
        let mut c = Session::new(Device::Gpu).compile(g.clone());
        // Warm at the declared extent so the timed run is not the first one.
        let _ = c.run(&inputs).remove(0);
        reset_attention_route_counts();
        c.set_active_extent(Some((actual, UPPER)));
        let gpu = c.run(&inputs).remove(0);
        c.set_active_extent(None);
        let routes = attention_route_counts();
        assert_eq!(
            routes, want,
            "patched to {actual} of {UPPER} keys: the route must come from the \
             extent the run walks, not the one the graph was lowered at"
        );
        // No hint on the CPU: it walks all `UPPER` keys and the mask drops the
        // tail, so this checks "patched walk == full walk with the tail masked
        // out" — the property the scaling machinery promises.
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-patch] actual={actual} upper={UPPER} routes={routes:?} \
             vs CPU: max_abs={d:.6e}"
        );
        assert!(
            d < 1e-5,
            "patched attention walk actual={actual} of {UPPER}: max_abs {d} vs CPU"
        );
    }
}

/// Every mask kind the serial kernel implements, on the decode route against the
/// CPU: the decode form is a re-shaping of the same rule, so each kind has to
/// reach the reference by the same numbers, and none of them may fall off the
/// route.
#[test]
fn decode_route_mask_kinds_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let dim = NH * HD;
    let q = data(dim, 0.011, false);
    let k = data(DECODE_LK * dim, 0.013, true);
    let v = data(DECODE_LK * dim, 0.017, false);
    for kind in [
        MaskKind::None,
        MaskKind::Causal,
        // Only keys [q_abs - w, q_abs] attend, and a decode row sits at
        // q_abs = lk - 1, so this reads the absolute query position too.
        MaskKind::SlidingWindow(2),
        MaskKind::Custom,
        MaskKind::Bias,
    ] {
        let carries_mask = matches!(kind, MaskKind::Custom | MaskKind::Bias);
        let g = if carries_mask {
            masked_chain(1, DECODE_LK, 1, NH, HD, kind)
        } else {
            chain(1, DECODE_LK, 1, NH, HD, kind)
        };
        let mask = mask_data(kind, 1, DECODE_LK, NH);
        let mut inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        if carries_mask {
            inputs.push(("mask", &mask));
        }
        reset_attention_route_counts();
        let gpu = Session::new(Device::Gpu)
            .compile(g.clone())
            .run(&inputs)
            .remove(0);
        let routes = attention_route_counts();
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-mask] {kind:?} decode rows=1 heads={NH} head_dim={HD} lk={DECODE_LK} \
             routes={routes:?} vs CPU: max_abs={d:.6e}"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 1,
                serial: 0,
            },
            "{kind:?} on the decode shape must take the decode kernel"
        );
        assert!(d < 1e-5, "{kind:?} on the decode shape: max_abs {d} vs CPU");
    }
}

/// Shapes the decode form cannot serve keep the serial kernel, and keep its
/// numbers: a key length at (not above) the decode threshold, and a head width
/// above the width the lane-parallel form caps at.
#[test]
fn route_falls_back_outside_the_decode_form() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    // The constant-derived width, not a bare 256: it is `MAX + 128` that is the
    // interesting side of the boundary here.
    let wide_hd = ATTENTION_DECODE_MAX_HEAD_DIM as usize + 128;
    for (rows, nh, hd, lk) in [
        (1usize, NH, HD, ATTENTION_DECODE_MIN_KEYS as usize),
        (1, 4, wide_hd, 186),
    ] {
        let dim = nh * hd;
        let g = chain(rows, lk, 1, nh, hd, MaskKind::Causal);
        let q = data(rows * dim, 0.011, false);
        let k = data(lk * dim, 0.013, true);
        let v = data(lk * dim, 0.017, false);
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        let mut c = Session::new(Device::Gpu).compile(g.clone());
        let _ = c.run(&inputs).remove(0);
        reset_attention_route_counts();
        let gpu = c.run(&inputs).remove(0);
        let routes = attention_route_counts();
        // The host routes on the mirror of this guard; the observed route must
        // agree with it.
        assert!(
            !attention_decode_eligible(rows as u32, lk as u32, hd as u32, 0),
            "rows={rows} nh={nh} hd={hd} lk={lk} is not decode-shaped"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 0,
                serial: 1,
            },
            "rows={rows} nh={nh} hd={hd} lk={lk}: outside the decode form the \
             serial kernel must serve the step"
        );
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-fallback] rows={rows} heads={nh} head_dim={hd} lk={lk} \
             routes={routes:?} vs CPU: max_abs={d:.6e}"
        );
        assert!(
            d < 1e-5,
            "serial-kernel fallback rows={rows} nh={nh} hd={hd} lk={lk}: max_abs {d} vs CPU"
        );
    }
}
