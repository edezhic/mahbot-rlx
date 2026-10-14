// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixtures and measurement scaffolding shared by this crate's tests: the packed
//! GGUF weight, the activation ramp, and the median together with the two timing
//! loops ([`time_rounds`], one cell alone, and [`time_round_robin`], round-robin
//! across a cell set) that the three harnesses using them — `gguf_one_row_stream_rate`,
//! `gguf_gemm_grouped_weight_reuse` and `gguf_gemm_grouped_split_rate` — time their
//! calls with, plus the parity helpers, the conv-transpose-2d route levers together
//! with the pair bound its pins derive their bar from, the pins' own refusal and the
//! route-counter lock at the end of the file. `tests/common/` is not a test target of
//! its own; each test binary pulls it in with `mod common;`.

// Each test binary declaring `mod common;` uses only the helpers it needs, so
// the rest are dead code in that binary and `just lint` runs with `-D warnings`.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use rlx_wgpu::backend::ATTENTION_DECOMPOSE_MIN_Q;
use rlx_wgpu::kernels::{ATTENTION_DECODE_MIN_KEYS, AttentionRoutes};

/// A deterministic `[m, k]` row-major activation, at `step`'s ramp rate. `step`
/// stays the caller's because it sets the result's magnitude, which a parity
/// tolerance has to be read against.
pub fn ramp(m: usize, k: usize, step: f32) -> Vec<f32> {
    (0..m * k).map(|i| ((i as f32) * step).sin()).collect()
}

/// `v`'s median: the element at `len / 2` of a sorted **copy**, so an even-length
/// series gives its upper-middle element rather than the mean of the pair, an empty
/// one panics, and `v` keeps the order the caller prints it in.
pub fn median(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    sorted[sorted.len() / 2]
}

/// Time `calls_per_round` back-to-back calls, `rounds` times over: each round's
/// mean per call, in milliseconds. `Tick`'s cycle counter is the clock, since a
/// call at these shapes is well under a millisecond on some cells.
pub fn time_rounds(rounds: usize, calls_per_round: usize, mut one_call: impl FnMut()) -> Vec<f64> {
    (0..rounds)
        .map(|_| {
            let start = rlx_runtime::Tick::now();
            for _ in 0..calls_per_round {
                one_call();
            }
            rlx_runtime::Tick::now().elapsed_ns(start) as f64 / calls_per_round as f64 / 1e6
        })
        .collect()
}

/// [`time_rounds`] for a cell set: each round gives every cell its own
/// `calls_per_round` calls back to back — **round-robin across the cells**, not one
/// cell's whole timing block at a time, so that the process's warm-up and the
/// machine's drift land on every cell alike rather than on whichever cells come
/// first. Each cell's own per-round figures, in milliseconds per call.
pub fn time_round_robin<T>(
    cells: &mut [T],
    rounds: usize,
    calls_per_round: usize,
    mut one_call: impl FnMut(&mut T),
) -> Vec<Vec<f64>> {
    let mut per_call_ms = vec![Vec::with_capacity(rounds); cells.len()];
    for _ in 0..rounds {
        for (i, cell) in cells.iter_mut().enumerate() {
            let start = rlx_runtime::Tick::now();
            for _ in 0..calls_per_round {
                one_call(cell);
            }
            per_call_ms[i].push(
                rlx_runtime::Tick::now().elapsed_ns(start) as f64 / calls_per_round as f64 / 1e6,
            );
        }
    }
    per_call_ms
}

/// Deterministic packed GGUF bytes for a `[k, n]` weight — a fixed f32 ramp
/// quantized into `ty`, the construction `gguf_batch_join`, `buffer_copy_batch_join`
/// and `run_report` all build their weights with. A fixture and not a value:
/// nothing reading it looks at the bytes.
pub fn packed_gguf_bytes(ty: rlx_gguf::GgmlType, k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w, ty).expect("quantize")
}

/// One packed GGUF weight of `[n, k]` at `quant`'s own block geometry, filled
/// deterministically.
///
/// The bytes are laid out by hand rather than quantized from an f32 tensor: the
/// f32 form is larger than the packed one at these shapes and is part of the arm's
/// own cost, while nothing in the measurement reads the byte values. Small on
/// purpose: the high halves of the f16 scale fields get a fixed exponent so `d`
/// and `dmin` land in [2^-13, 2^-12) and the dequantized weights stay ~1e-2 —
/// finite, and small enough that a grouped result can be compared with a per-row
/// one under the op's absolute 1e-2 tolerance rather than measuring the fixture's
/// own weight scale (large random weights cancel catastrophically, and the
/// round-off of that cancellation would be what such a bound caught).
pub fn packed_gguf_weight(quant: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> Vec<u8> {
    use rlx_ir::quant::QuantScheme;
    let (block_elems, block_bytes) = (
        quant.gguf_block_size() as usize,
        quant.gguf_block_bytes() as usize,
    );
    // Byte offsets within a block whose high half is an f16 exponent: Q4_K's `d`
    // and `dmin` at its head, Q6_K's `d` at its tail, Q8_0's `d` at its head.
    let f16_high: &[usize] = match quant {
        QuantScheme::GgufQ4K => &[1, 3],
        QuantScheme::GgufQ6K => &[209],
        QuantScheme::GgufQ8_0 => &[1],
        other => panic!("{other:?} has no row-grouped arm"),
    };
    let mut s = 7u32;
    (0..(k / block_elems) * block_bytes * n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 24) as u8;
            if f16_high.contains(&(i % block_bytes)) {
                // High half of an f16: sign 0, exponent 2, low mantissa bits from
                // the generator.
                0x08 | (b & 0x03)
            } else {
                b & 0x3F
            }
        })
        .collect()
}

/// Largest absolute difference between a computed and a reference output, after
/// a finiteness census.
///
/// The census runs FIRST and is asserted clean, because `f32::max` returns the
/// non-NaN operand: an all-NaN or infinity-poisoned output would otherwise fold
/// to a small number and pass any tolerance, so a broken kernel could satisfy a
/// parity bar. NaN, +inf and −inf are counted separately so an infinity cannot
/// hide behind a NaN.
pub fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "parity needs equal lengths");
    let (mut nan, mut pos_inf, mut neg_inf) = (0usize, 0usize, 0usize);
    for &v in got.iter().chain(want) {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos_inf += 1;
        } else if v == f32::NEG_INFINITY {
            neg_inf += 1;
        }
    }
    assert_eq!(
        (nan, pos_inf, neg_inf),
        (0, 0, 0),
        "non-finite in the parity pair: NaN={nan} +inf={pos_inf} -inf={neg_inf}"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Whole-vector bit-pattern equality — the tree's idiom for "the same work in a
/// different arrangement" (a joined vs an unjoined pass, a striped vs a whole
/// arena). `what` is the site's own claim, which names the two arrangements the
/// pair came from.
pub fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: {}-element output read as {}-element",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{what}: element {i} reads {x} against {y}"
        );
    }
}

/// The 8-bit GGUF scheme's bisect lever, spelled in one place: it takes that
/// scheme's multi-row multiplies off the grouped device arm, so the pins whose
/// routes, joined intervals or counted pairs it would move refuse on it.
pub const Q8_0_BISECT: &str = "RLX_WGPU_Q8_0_GEMM_DISABLE";

/// Refuse when the 8-bit scheme's bisect lever is set: the lever is read where that
/// call is encoded, so under it a call this tree routes on the grouped device arm
/// answers the slab instead — another routing mode, not a defect in the pin.
pub fn refuse_under_q8_0_bisect() {
    assert!(
        !rlx_ir::env::flag(Q8_0_BISECT),
        "RLX_WGPU_Q8_0_GEMM_DISABLE moves the 8-bit scheme's multiplies off the grouped route; \
         unset it to run this pin"
    );
}

/// How many schedule steps are the materialised broadcast family
/// (`expand` / `expand_host`) — the lowering leaves one wherever a broadcast
/// operand has to arrive as a real tensor.
pub fn expand_steps(names: &[&str]) -> usize {
    names
        .iter()
        .filter(|n| **n == "expand" || **n == "expand_host")
        .count()
}

/// The step names this backend dispatches for `g` as the runtime path reaches it:
/// the legalising pass first (what the shim runs ahead of fusion —
/// `rlx-runtime/src/backend/wgpu_backend.rs`), then the lowering.
///
/// Legalising first is what makes a *missing* `expand` step evidence that the
/// peel fired, rather than evidence that the pass never ran and inserted none.
pub fn legalised_step_names(g: &rlx_ir::Graph) -> Vec<&'static str> {
    use rlx_opt::pass::Pass as _;
    let legalised = rlx_opt::LegalizeBroadcast.run(g.clone());
    rlx_wgpu::backend::WgpuExecutable::compile(legalised).test_step_names()
}

/// The run report's switch, spelled in one place: a pin whose figures the report's
/// own fence would move names it by this, and this is what the pins refuse on.
pub const RUN_REPORT_SWITCH: &str = "RLX_WGPU_DBG_COUNTERS";

/// Refuse when the run report's switch is set: a reporting run adds a fence, a
/// resolve submit and a map of its own, so the submission/wait counts and the
/// rates a pin measures under it are not that pin's.
pub fn refuse_under_run_report() {
    assert!(
        !rlx_ir::env::flag(RUN_REPORT_SWITCH),
        "RLX_WGPU_DBG_COUNTERS changes the submission/wait shape of a forward pass; \
         unset it to run this pin"
    );
}

/// The arrangement every layer of a many-layer attention chain is expected to
/// take: one query row over more than `ATTENTION_DECODE_MIN_KEYS` keys is the
/// decode-shaped form, below the many-row route's floor the fused kernel, and
/// from that floor up the dense arrangement. `layers` attention steps per run,
/// over `reps` runs, so the counters it predicts are the whole bracket.
///
/// The shape decides which arrangement the route is *asked for*; the arm's
/// window term (`attention_window_declines`, whose count tells that decline
/// apart from the shape's own answer) can still keep the fused kernel, and a
/// harness on an adapter where it does fails this expectation loudly rather
/// than skipping. The head widths a harness uses must clear
/// `attention_decode_eligible`'s caps for the decode arm to apply; the two
/// harnesses here run 8 heads × 64 and 16 × 128, both inside the cap.
pub fn expected_attention_routes(
    rows: usize,
    lk: usize,
    layers: usize,
    reps: usize,
) -> AttentionRoutes {
    let per_run = if rows == 1 && lk > ATTENTION_DECODE_MIN_KEYS as usize {
        AttentionRoutes {
            decode: layers as u64,
            decomposed: 0,
            serial: 0,
        }
    } else if rows >= ATTENTION_DECOMPOSE_MIN_Q as usize {
        AttentionRoutes {
            decode: 0,
            decomposed: layers as u64,
            serial: 0,
        }
    } else {
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: layers as u64,
        }
    };
    AttentionRoutes {
        decode: per_run.decode * reps as u64,
        decomposed: per_run.decomposed * reps as u64,
        serial: per_run.serial * reps as u64,
    }
}

/// The levers that put `Op::ConvTranspose2d` on its device gather step, spelled in
/// one place because every pin that has to reach that arm needs the same routers out
/// of the way: `RLX_WGPU_FORCE_HOST`, `RLX_WGPU_DISCRETE_HOST`, `RLX_WGPU_ALLOW_SHARD`
/// and `RLX_WGPU_ALL_PARAMS_WEIGHT` zeroed, the shard-cap lever neutralised to a value
/// the adapter clamps back to its own limit, and `RLX_WGPU_SHARD_GPU=1` — which turns
/// the whole structure-host preference off, because on a Vulkan/DX12 adapter its
/// discrete term would otherwise host this op on any adapter kind and no other lever
/// clears it. The graphs these pins build carry no structural ops, so the opt-out's own
/// caveat (it is unsafe for Kitten NSF) does not reach them.
///
/// The parked-weight lever belongs in that list too, as the other decision about this
/// op's operands a pin should make for itself: the one pin that *wants* a parked weight
/// sets it back to `1` after this call.
///
/// The skip stays the caller's own: ask
/// `rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available())` after
/// this, per file.
pub fn conv_transpose2d_device_route_env() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
}

/// One conv-transpose-2d geometry: the operand shapes and the attribute set the bound
/// walk below reads. The harnesses that derive a bar from their operands keep their
/// case tables in this shape rather than in eleven loose numbers, so the walk takes it
/// and there is one vocabulary for "which geometry" across them.
pub struct ConvTranspose2dGeometry {
    pub n: usize,
    pub c_in: usize,
    /// The weight's second dim — the output carries `c_out · groups` channels.
    pub c_out: usize,
    pub h: usize,
    pub w: usize,
    pub k: [usize; 2],
    pub s: [usize; 2],
    pub p: [usize; 2],
    pub d: [usize; 2],
    pub op: [usize; 2],
    pub groups: usize,
}

impl ConvTranspose2dGeometry {
    /// The output's `(h, w)`, from the tree's own shape rule rather than a locally
    /// restated formula.
    pub fn out(&self) -> (usize, usize) {
        let axis = |size: usize, k: usize, s: usize, p: usize, d: usize, op: usize| {
            rlx_ir::shape::conv_transpose2d_spatial_output(size, k, s, p, d, op)
        };
        (
            axis(
                self.h, self.k[0], self.s[0], self.p[0], self.d[0], self.op[0],
            ),
            axis(
                self.w, self.k[1], self.s[1], self.p[1], self.d[1], self.op[1],
            ),
        )
    }

    /// The output's channel count, `c_out · groups`.
    pub fn c_out_total(&self) -> usize {
        self.c_out * self.groups
    }
}

/// The bound on how far `Op::ConvTranspose2d`'s device gather can sit from the scalar
/// oracle `rlx_cpu::kernels::conv_transpose2d_nchw`, derived from `geo`'s operands
/// rather than from an absolute bar. Each route accumulates the same `terms` f32
/// products per output element in its own order, so each can differ from the exact sum
/// by no more than `γ_n · Σ|terms|` (`γ_n = n·u / (1 − n·u)`, `u = 2^-24` the f32 unit
/// roundoff) and the two can differ from each other by at most twice that. Returns
/// `(bound, worst terms, worst Σ|terms|)`.
///
/// The tap walk is input-centric and mirrors the oracle's own, so the `terms` it counts
/// are the products the reference itself would have summed — every tap is counted,
/// including the ones a zero operand makes vanish, which only loosens the bound and
/// never the assertion. Padding, dilation, groups and output padding are all in it, so
/// the decoder's unit-height geometry (one tap on the height axis, `p = 0`) is just a
/// case of the same walk rather than a second derivation of it.
///
/// The grouping is the one thing it does not model, and it checks for it: `groups` must
/// be positive and divide `c_in`, because the degenerate groupings are refusals on both
/// routes rather than value cases, and the walk divides by `c_in / groups`.
pub fn conv_transpose2d_pair_error_bound(
    geo: &ConvTranspose2dGeometry,
    x: &[f32],
    w: &[f32],
) -> (f64, usize, f64) {
    /// The f32 unit roundoff, `2^-24`.
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    // The walk groups channels on `c_in / groups`, so a grouping that does not
    // partition `c_in` is outside what it covers — and both non-divisible groupings
    // are refusals on either route rather than value cases, so there is no bound to
    // derive for them. State the assumption instead of dividing by a zero.
    assert!(
        geo.groups >= 1 && geo.c_in.is_multiple_of(geo.groups),
        "conv_transpose2d_pair_error_bound: c_in={} groups={} is a degenerate grouping, which \
         the device route refuses and this walk does not cover",
        geo.c_in,
        geo.groups
    );
    let (h_out, w_out) = geo.out();
    let c_out_total = geo.c_out_total();
    let c_in_per_g = geo.c_in / geo.groups;
    let mut abs_sum = vec![0f64; geo.n * c_out_total * h_out * w_out];
    let mut terms = vec![0usize; abs_sum.len()];
    for ni in 0..geo.n {
        for ic in 0..geo.c_in {
            let g = ic / c_in_per_g;
            for iy in 0..geo.h {
                for ix in 0..geo.w {
                    let v = x[((ni * geo.c_in + ic) * geo.h + iy) * geo.w + ix] as f64;
                    for ky in 0..geo.k[0] {
                        // `oy` is the padded output coordinate the oracle's own `oy`
                        // lives in; a tap outside `[p, p + out)` lands outside the
                        // output and the oracle skips exactly the same ones.
                        let oy = iy * geo.s[0] + ky * geo.d[0];
                        if oy < geo.p[0] || oy - geo.p[0] >= h_out {
                            continue;
                        }
                        let oy = oy - geo.p[0];
                        for kx in 0..geo.k[1] {
                            let ox = ix * geo.s[1] + kx * geo.d[1];
                            if ox < geo.p[1] || ox - geo.p[1] >= w_out {
                                continue;
                            }
                            let ox = ox - geo.p[1];
                            for oc_off in 0..geo.c_out {
                                let oc = g * geo.c_out + oc_off;
                                let wt = w
                                    [((ic * geo.c_out + oc_off) * geo.k[0] + ky) * geo.k[1] + kx]
                                    as f64;
                                let at = ((ni * c_out_total + oc) * h_out + oy) * w_out + ox;
                                abs_sum[at] += (v * wt).abs();
                                terms[at] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut worst = (0f64, 0usize, 0f64);
    for i in 0..abs_sum.len() {
        let nu = terms[i] as f64 * U;
        // Two accumulation orders of the same sum, so twice one order's bound.
        let bound = 2.0 * nu / (1.0 - nu) * abs_sum[i];
        if bound > worst.0 {
            worst = (bound, terms[i], abs_sum[i]);
        }
    }
    worst
}

/// The route counters are process-global, so tests in one binary that bracket
/// one (`reset` … `run` … `read`) must not run next to each other.
static ROUTE_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// [`ROUTE_TEST_MUTEX`], poisoned-tolerant: an earlier failing assertion must not
/// turn every later test in the binary into a lock error.
pub fn route_lock() -> MutexGuard<'static, ()> {
    ROUTE_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}
