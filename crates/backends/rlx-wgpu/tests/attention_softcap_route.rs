// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Attention`'s `attn_logit_softcap` on the wgpu route: the logit cap
//! `c·tanh(s/c)` over the *scaled* scores, ahead of the mask/bias and the
//! softmax, on each of the three arrangements the route serves — the fused
//! serial kernel, the decode-shaped one and the many-row dense one. Four things
//! can go wrong along that path, and each has its own gate rather than a share
//! of the parity assertion.
//!
//! A cap *dropped* on the way to the kernels would fail the parity assertion
//! below — this file's own reference applies the term, unlike the CPU backend
//! the other attention tests measure against — but that assertion cannot say how
//! much of the answer the term carries. Every arrangement is therefore also run
//! a second time on the same data with no cap, and the pair has to differ by
//! more than a hundred times its own parity bar. A case that would stay green
//! with the term inert has no business being one.
//!
//! A cap landing *after* the mask/bias rather than before it is what the
//! operation's definition rules out: an additive term inside the `tanh` is not
//! the same function of the score as one outside it, and an exclusion sentinel
//! read through the cap stops being an exclusion. The additive
//! [`MaskKind::Bias`] kind is where the two orders are compared with no sentinel
//! in the way. A cap at or below zero, like an absent one, is instead the
//! *disabled* spelling and has to keep today's answer bit for bit.
//!
//! A head width past the fused kernel's per-thread arrays can lower to a step
//! that returns without writing its output slot — a blank result rather than a
//! failure. `backend::attention_route`'s plan refuses that width, pinned here on
//! both entry points, the backend's own `compile` and the runtime's `Session`,
//! as a panic naming the operator and the width.
//!
//! The oracle is the operation written out at the end of this file, not the CPU
//! backend: its accumulation widths are named on the spot — the dot product in
//! f32, the width the device contracts in, and everything after it in f64 — so a
//! mismatch points at the *order* of the cap, the mask and the softmax rather
//! than at a second kernel's rounding. Its data is a unit-range LCG, small
//! enough that the f32-versus-f64 gap stays under every bar here; the one
//! contrast that needs larger scores lifts it ([`DENSE_CAP50`]).
//!
//! Several tests bracket the process-global route counters, so they take
//! [`route_lock`] instead of living in one single-test binary.
mod common;

use common::{assert_bit_identical, max_abs_diff, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

/// One of the three arrangements the route serves, at the geometry the cap is
/// pinned on. The extents are the shape's own answer to
/// `kernels::attention_decode_eligible` and `backend::ATTENTION_DECOMPOSE_MIN_Q`,
/// so `routes` is what the arrangement this case *is* must count, and `bar` is the
/// max-abs against the reference that arrangement is held to: the three reduce over
/// 128 (or 48) keys and over different lane counts, so one bar would be slack for
/// the fused form or tight for the dense one. `scale` lifts the unit-range data at
/// the call site, and is 1.0 everywhere but [`DENSE_CAP50`].
struct Case {
    label: &'static str,
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    hd: usize,
    routes: AttentionRoutes,
    bar: f32,
    scale: f32,
}

/// The fused serial kernel's shape: one query row over 48 keys, below both the
/// many-row route's query-row floor and the decode form's own floor. The causal
/// branch puts that row at the cache's end, so it still walks all 48 keys.
const FUSED: Case = Case {
    label: "fused",
    b: 1,
    rows: 1,
    lk: 48,
    nh: 4,
    hd: 64,
    routes: AttentionRoutes {
        decode: 0,
        decomposed: 0,
        serial: 1,
    },
    bar: 1e-6,
    scale: 1.0,
};

/// A single query row over more keys than the decode form's floor: the
/// lane-parallel decode kernel.
const DECODE: Case = Case {
    label: "decode",
    b: 1,
    rows: 1,
    lk: 128,
    nh: 4,
    hd: 64,
    routes: AttentionRoutes {
        decode: 1,
        decomposed: 0,
        serial: 0,
    },
    bar: 1e-5,
    scale: 1.0,
};

/// Above the many-row route's query-row floor, where every key length is served:
/// the dense arrangement, whose masked scale is the shared `attention_mask`
/// entry point.
const DENSE: Case = Case {
    label: "dense",
    b: 1,
    rows: 48,
    lk: 128,
    nh: 4,
    hd: 64,
    routes: AttentionRoutes {
        decode: 0,
        decomposed: 1,
        serial: 0,
    },
    bar: 1e-6,
    scale: 1.0,
};

/// The band's own rows: the dense arrangement at the two query rows the fork's
/// cached decode body walks, where the cap still meets the mask on more than one row.
const DENSE_2ROWS: Case = Case { rows: 2, ..DENSE };

/// The dense arrangement at the lift its `cap = 50.0` contrast needs, which the
/// `cap = 1.0` cases run without: 1.35× the unit-range data (scores ×1.8).
///
/// A cap of 50 shifts a scaled score `x` by the cubic term of `50·tanh(x/50)`,
/// `x³/7500`: on the unit-range data's largest scores that is a measured 1.14e-5
/// of output, a tenth of the 1e-4 the non-vacuity gate asks for. The lift takes
/// the answer's move to 2.12e-4, and the parity error — which grows with the
/// score rather than with its cube — only to 3.87e-7 against its 1e-6.
const DENSE_CAP50: Case = Case {
    scale: 1.35,
    ..DENSE
};

/// A deterministic LCG in `[-1, 1)`: the magnitude that keeps the reference's
/// f32-versus-f64 gap — and the device's own dot-product rounding — well under
/// the bars, rather than what those bars measure.
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// [`data`] at `scale`: the generator stays unit-range and the one case that
/// needs larger scores ([`DENSE_CAP50`]) lifts them here.
fn lift(mut v: Vec<f32>, scale: f32) -> Vec<f32> {
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

/// The additive per-head bias for [`MaskKind::Bias`], in the `[B, heads, rows,
/// lk]` layout the reference indexes: a positional term and a per-head one, so a
/// kernel that collapsed either axis lands outside the bar. Its magnitude stays
/// small for the same reason the data's does — and [`lift`] leaves it alone: the
/// additive mask is a caller's own input rather than part of this file's score
/// scale.
fn bias_data(b: usize, rows: usize, lk: usize, nh: usize) -> Vec<f32> {
    (0..b * nh * rows * lk)
        .map(|i| {
            let s = (i % lk) as f32;
            let h = (i / (rows * lk)) % nh;
            ((s * 0.05).sin() + h as f32 * 0.11) * 0.5
        })
        .collect()
}

/// Rank-3 `[B, rows, heads·head_dim]` MHA over `lk` keys, with `cap` as the
/// node's `attn_logit_softcap`. [`MaskKind::Bias`] carries a `[B, heads, rows,
/// lk]` additive mask input — the shape this file's reference indexes — and the
/// other kinds carry none.
fn chain(case: &Case, kind: MaskKind, cap: Option<f32>) -> Graph {
    let dim = case.nh * case.hd;
    let f = DType::F32;
    let mut g = Graph::new("attn_softcap");
    let q = g.input("q", Shape::new(&[case.b, case.rows, dim], f));
    let k = g.input("k", Shape::new(&[case.b, case.lk, dim], f));
    let v = g.input("v", Shape::new(&[case.b, case.lk, dim], f));
    let mut inputs = vec![q, k, v];
    if matches!(kind, MaskKind::Bias) {
        let mask = g.input(
            "mask",
            Shape::new(&[case.b, case.nh, case.rows, case.lk], f),
        );
        inputs.push(mask);
    }
    let y = g.add_node(
        Op::Attention {
            num_heads: case.nh,
            head_dim: case.hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: cap,
        },
        inputs,
        Shape::new(&[case.b, case.rows, dim], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// One GPU run of `g` at `inputs`: warmed so the pipelines are compiled before
/// the count is bracketed, then one counted run — its output and the
/// arrangement it took.
fn run(g: Graph, inputs: &[(&str, &[f32])]) -> (Vec<f32>, AttentionRoutes) {
    let mut c = Session::new(Device::Gpu).compile(g);
    let _ = c.run(inputs).remove(0);
    reset_attention_route_counts();
    let out = c.run(inputs).remove(0);
    (out, attention_route_counts())
}

/// One case's GPU run at `(kind, cap)`: the output, the arrangement it took and
/// the reference it is measured against. The data is a function of the case and
/// the kind alone, so the capped and uncapped runs of a case differ in nothing
/// else.
fn case_run(
    case: &Case,
    kind: MaskKind,
    cap: Option<f32>,
) -> (Vec<f32>, AttentionRoutes, Vec<f32>) {
    let dim = case.nh * case.hd;
    let g = chain(case, kind, cap);
    let q = lift(data(case.b * case.rows * dim, 1), case.scale);
    let k = lift(data(case.b * case.lk * dim, 2), case.scale);
    let v = lift(data(case.b * case.lk * dim, 3), case.scale);
    let bias = bias_data(case.b, case.rows, case.lk, case.nh);
    let mut inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    if matches!(kind, MaskKind::Bias) {
        inputs.push(("mask", &bias));
    }
    let (gpu, routes) = run(g, &inputs);
    let want = reference(
        &q,
        &k,
        &v,
        matches!(kind, MaskKind::Bias).then_some(bias.as_slice()),
        case.b,
        case.rows,
        case.lk,
        case.nh,
        case.hd,
        kind,
        cap.unwrap_or(0.0),
    );
    (gpu, routes, want)
}

/// One case's run, assertions and printed line: the step takes the arrangement
/// this case *is*, and its output reaches the reference. The output comes back
/// because the non-vacuity gate needs the capped run to compare against.
fn assert_case(case: &Case, kind: MaskKind, cap: Option<f32>) -> Vec<f32> {
    let (gpu, routes, want) = case_run(case, kind, cap);
    let d = max_abs_diff(&gpu, &want);
    eprintln!(
        "[rlx-attn-softcap-{}] kind={kind:?} cap={cap:?} scale={} batch={} rows={} lk={} heads={} \
         head_dim={} routes={routes:?} vs reference: max_abs={d:.6e} (bar {:.0e})",
        case.label, case.scale, case.b, case.rows, case.lk, case.nh, case.hd, case.bar
    );
    assert_eq!(
        routes, case.routes,
        "{}: rows={} lk={} {kind:?} must take the arrangement this case is",
        case.label, case.rows, case.lk
    );
    assert!(
        d < case.bar,
        "{}: {kind:?} cap={cap:?} max_abs {d} against the reference (bar {})",
        case.label,
        case.bar
    );
    gpu
}

/// The gate that makes a case non-vacuous: the same graph at the same shape and
/// data, with no cap at all, has to answer by more than a hundred times that
/// case's parity bar. A later change that drops the cap on the way to the
/// kernels leaves the capped run equal to the uncapped one, and this is the
/// assertion that then says so most directly — the parity one above fails too,
/// but as a shape-sized surprise against a reference that still applies the
/// term.
fn assert_cap_changes_the_answer(case: &Case, capped: &[f32], cap: f32) {
    let (uncapped, routes, _) = case_run(case, MaskKind::Causal, None);
    let d = max_abs_diff(capped, &uncapped);
    let bar = 100.0 * case.bar;
    eprintln!(
        "[rlx-attn-softcap-{}] kind=Causal cap={cap:?} scale={} against none, same data: \
         max_abs={d:.6e} (bar {bar:.0e})",
        case.label, case.scale
    );
    assert_eq!(
        routes, case.routes,
        "{}: the uncapped run of the gate must take the same arrangement as the capped one",
        case.label
    );
    assert!(
        d > bar,
        "{}: cap {cap} moves the answer by {d} against no cap, under the {bar} this gate needs",
        case.label
    );
}

/// The fused serial arrangement — one query row over 48 keys, below both the
/// many-row route's query-row floor and the decode form's own key floor — under the
/// causal kind and under the additive `Bias` that separates the two orders of cap
/// and mask. The causal branch puts that row at the cache's end, so it still walks
/// all 48 keys.
#[test]
fn the_fused_arrangement_caps_ahead_of_the_mask() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let causal = assert_case(&FUSED, MaskKind::Causal, Some(1.0));
    assert_case(&FUSED, MaskKind::Bias, Some(1.0));
    assert_cap_changes_the_answer(&FUSED, &causal, 1.0);
}

/// The decode-shaped arrangement — one query row over more keys than that form's
/// floor — under the same two kinds, at its own bar.
#[test]
fn the_decode_arrangement_caps_ahead_of_the_mask() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let causal = assert_case(&DECODE, MaskKind::Causal, Some(1.0));
    assert_case(&DECODE, MaskKind::Bias, Some(1.0));
    assert_cap_changes_the_answer(&DECODE, &causal, 1.0);
}

/// The two caps the Gemma-2 family runs: `1.0`, the compressing one, and `50.0`,
/// the near-identity one the family actually carries over scores of this size.
/// Both are gated for non-vacuity, and only this arrangement is run at both:
/// the other two are pinned at the cap that moves the answer.
///
/// A cap of 50 is near-identity, not invisible: it leaves a cubically small but
/// real shift, so the gate is asserted here on the data that shows it
/// ([`DENSE_CAP50`]) rather than assumed away. A kernel that dropped the cap
/// would leave this case answering exactly like its uncapped run, which the gate
/// reads as a failure.
#[test]
fn the_dense_arrangement_caps_ahead_of_the_mask() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let causal = assert_case(&DENSE, MaskKind::Causal, Some(1.0));
    assert_case(&DENSE, MaskKind::Bias, Some(1.0));
    assert_case(&DENSE_2ROWS, MaskKind::Causal, Some(1.0));
    assert_case(&DENSE_2ROWS, MaskKind::Bias, Some(1.0));
    let causal_50 = assert_case(&DENSE_CAP50, MaskKind::Causal, Some(50.0));
    assert_case(&DENSE_CAP50, MaskKind::Bias, Some(50.0));
    assert_cap_changes_the_answer(&DENSE, &causal, 1.0);
    assert_cap_changes_the_answer(&DENSE_CAP50, &causal_50, 50.0);
}

/// Zero and negative caps are the disabled spelling, and an absent cap has to
/// keep today's answer bit for bit: all three spellings reach the kernels as a
/// `cap` the `cap > 0` test rejects, so the work is the same work. Pinned on
/// every arrangement and both mask kinds, since the disabled test is the one
/// place the whole path is compared into itself.
#[test]
fn a_zero_or_negative_cap_is_the_absent_cap_bit_for_bit() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    for case in [&FUSED, &DECODE, &DENSE] {
        for kind in [MaskKind::Causal, MaskKind::Bias] {
            let absent = case_run(case, kind, None).0;
            for cap in [0.0f32, -1.0] {
                let disabled = case_run(case, kind, Some(cap)).0;
                assert_bit_identical(
                    &disabled,
                    &absent,
                    &format!("{}: {kind:?} at cap {cap} against no cap", case.label),
                );
            }
            eprintln!(
                "[rlx-attn-softcap-{}] kind={kind:?} cap=0 / cap=-1 against none, same data: \
                 bit-identical",
                case.label
            );
        }
    }
}

/// Rank-3 `[1, rows, heads·head_dim]` attention with an explicit `v_head_dim` —
/// the widths the plan's ceiling is read against — and no mask, because the
/// refusal is about the geometry alone.
fn wide_graph(rows: usize, lk: usize, nh: usize, hd: usize, v_hd: Option<usize>) -> Graph {
    let vd = v_hd.unwrap_or(hd);
    let f = DType::F32;
    let mut g = Graph::new("attn_softcap_ceiling");
    let q = g.input("q", Shape::new(&[1, rows, nh * hd], f));
    let k = g.input("k", Shape::new(&[1, lk, nh * hd], f));
    let v = g.input("v", Shape::new(&[1, lk, nh * vd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: v_hd,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, rows, nh * vd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// `work` with its panic caught: `None` when it returned, and the panic's own
/// message when it did not. The lowering panics with a formatted `String`, and
/// a `&str` payload is read too, because that is what an `assert!` in its path
/// would carry.
fn caught(work: impl FnOnce()) -> Option<String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        ),
    }
}

/// A head width above the 512 the fused kernel's per-thread arrays and the
/// route's decision both hold is refused at plan time rather than lowered to an
/// unwritten output slot, on both the backend's `compile` and the runtime's
/// `Session` — the same refusal, since the runtime's path is the same lowering.
/// The width *at* the ceiling is the control: it is served, so what the pins
/// above catch is the edge and not a blanket refusal of wide heads.
#[test]
fn a_head_width_past_the_ceiling_is_refused_loudly() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const ROWS: usize = 4;
    const LK: usize = 8;
    const NH: usize = 1;
    for (label, hd, v_hd) in [
        ("head_dim", 1024usize, None),
        ("v_head_dim", 512usize, Some(1024usize)),
    ] {
        let msg = caught(|| {
            let _ = WgpuExecutable::compile(wide_graph(ROWS, LK, NH, hd, v_hd));
        })
        .unwrap_or_else(|| panic!("WgpuExecutable::compile accepted {label} above the ceiling"));
        eprintln!("[rlx-attn-softcap-ceiling] {label} above the ceiling: {msg}");
        assert!(
            msg.contains("rlx-wgpu Attention"),
            "the refusal must name the operator so a caller can act on it; got: {msg}"
        );
        assert!(
            msg.contains("1024"),
            "the refusal must name the offending width; got: {msg}"
        );
        // A warm-up reaches the same lowering through the runtime, so the
        // refusal has to survive that entry point rather than being caught into
        // an error a caller could mistake for a transient.
        let msg = caught(|| {
            let _ = Session::new(Device::Gpu).compile(wide_graph(ROWS, LK, NH, hd, v_hd));
        })
        .unwrap_or_else(|| panic!("Session::compile accepted {label} above the ceiling"));
        assert!(
            msg.contains("rlx-wgpu Attention") && msg.contains("1024"),
            "{label} through Session::compile: {msg}"
        );
    }
    assert!(
        caught(|| {
            let _ = WgpuExecutable::compile(wide_graph(ROWS, LK, NH, 512, None));
        })
        .is_none(),
        "head_dim 512 is the ceiling this route serves, not above it"
    );
}

/// The operation's definition, on the host: scores = (Q·K)·scale, then the logit
/// cap over those scaled scores, then the mask/bias, then a row softmax, then ·V.
/// Rank-3 [B, S, H·D] Q/K/V (the dense layouts, so there is no stride to mirror);
/// Q's `rows` queries sit at the END of the `lk` keys (absolute position
/// `qi + lk - rows` when `lk >= rows`).
///
/// The dot product is accumulated in **f32** — the width the device contracts
/// in, which keeps this comparison about the order of the cap, the mask and the
/// softmax rather than about another kernel's rounding — and everything after it
/// in **f64**.
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    hd: usize,
    kind: MaskKind,
    cap: f32,
) -> Vec<f32> {
    let dim = nh * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let q_off = lk.saturating_sub(rows);
    let mut out = vec![0.0f32; b * rows * dim];
    let mut scores = vec![0.0f64; lk];
    for bi in 0..b {
        for h in 0..nh {
            for qi in 0..rows {
                let q_abs = q_off + qi;
                for s in 0..lk {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[(bi * rows + qi) * dim + h * hd + d]
                            * k[(bi * lk + s) * dim + h * hd + d];
                    }
                    let mut x = (dot * scale) as f64;
                    if cap > 0.0 {
                        x = cap as f64 * (x / cap as f64).tanh();
                    }
                    let excluded = match kind {
                        MaskKind::None | MaskKind::Bias => false,
                        MaskKind::Causal => s > q_abs,
                        MaskKind::SlidingWindow(w) => s > q_abs || q_abs - s > w,
                        MaskKind::Custom => mask.expect("a custom mask")[bi * lk + s] < 0.5,
                    };
                    if excluded {
                        // The kernels' own spelling of an exclusion is a large
                        // negative finite number; the definition is that the
                        // position contributes nothing, so this is where the
                        // row's softmax is told so.
                        x = f64::NEG_INFINITY;
                    } else if matches!(kind, MaskKind::Bias) {
                        x +=
                            mask.expect("a bias mask")[((bi * nh + h) * rows + qi) * lk + s] as f64;
                    }
                    scores[s] = x;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|&x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for d in 0..hd {
                    let mut acc = 0.0f64;
                    for s in 0..lk {
                        acc += exps[s] / sum * v[(bi * lk + s) * dim + h * hd + d] as f64;
                    }
                    out[(bi * rows + qi) * dim + h * hd + d] = acc as f32;
                }
            }
        }
    }
    out
}
