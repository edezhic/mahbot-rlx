// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixtures and measurement scaffolding shared by this crate's tests: the packed
//! GGUF weight, the activation ramp, and the median together with the two timing
//! loops ([`time_rounds`], one cell alone, and [`time_round_robin`], round-robin
//! across a cell set) that the three harnesses using them — `gguf_one_row_stream_rate`,
//! `gguf_gemm_grouped_weight_reuse` and `gguf_gemm_grouped_split_rate` — time their
//! calls with, plus the parity helpers, the pins' own refusal and the route-counter
//! lock at the end of the file. `tests/common/` is not a test target of its own;
//! each test binary pulls it in with `mod common;`.

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

/// The route counters are process-global, so tests in one binary that bracket
/// one (`reset` … `run` … `read`) must not run next to each other.
static ROUTE_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// [`ROUTE_TEST_MUTEX`], poisoned-tolerant: an earlier failing assertion must not
/// turn every later test in the binary into a lock error.
pub fn route_lock() -> MutexGuard<'static, ()> {
    ROUTE_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}
