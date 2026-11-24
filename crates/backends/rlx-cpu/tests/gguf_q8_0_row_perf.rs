// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The single-row Q8_0 multiply, f32 dequantisation against the integer dot: the
//! paired reading behind `gguf_q8_0`'s route, on the shapes the synthesis code
//! predictor actually multiplies. The `f32 unpack` arm is the previous tree's route —
//! `gguf_matmul_bt_m1_rows_generic`, which this one replaced — taken in the same
//! session, so the ratio is the change itself rather than two builds' numbers.
//!
//! The route is aarch64-only, and so is the comparison: every shape is checked against the
//! route's own gates below, so a target or a host where it is not taken prints that and
//! returns rather than comparing two f32 runs under two labels.
//!
//! `#[ignore]`d — it prints, it does not assert. Gate it on `just throttle` first,
//! then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --test gguf_q8_0_row_perf \
//!     --features backend-tests -- --ignored --nocapture
//! ```
//!
//! One `#[test]`: the two arms must alternate inside one session for the comparison
//! to mean anything. Its switches are scoped to the measuring thread.
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// The five distinct packed shapes one 16-level predictor frame emits, with how many of
/// them the frame dispatches: `qkv` and `gate_up` are joined at build time, the
/// `attn_out` count includes the frame's sixteen `proj_in` multiplies, and the head is
/// stepped once per predicted codebook. 348 of the 351 emitted sites are dispatched; the
/// three dead ones are `blk.4`'s `attn_out`, `gate_up` and `ffn_down` at level 0, whose
/// row nothing reads (`crates/mahbot-inference/tests/loop_shape.rs`'s census reconciles
/// them).
const SHAPES: &[(usize, usize, usize)] = &[
    (1024, 4096, 80),
    (2048, 1024, 95),
    (1024, 6144, 79),
    (3072, 1024, 79),
    (1024, 2048, 15),
];
const ROUNDS: usize = 5;

#[test]
#[ignore = "manual kernel timing"]
fn q8_0_row_route_perf() {
    // The product's arrangement: the processor switches the dequant cache off, so a
    // single row reaches the fused arm rather than a cached f32 slab. Scoped to this
    // thread and held for the whole measurement: a process-wide `set` would outlive the
    // harness, and this is the only reader. (The eight-bit arm does not consult the
    // cache either way; the exact arm does.)
    let _no_cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
    // Masked, then checked per shape below: a shell's `RLX_REPRODUCIBLE` or a one-worker
    // pool would run the f32 route under both labels.
    let _reproducible = rlx_ir::env::scoped("RLX_REPRODUCIBLE", "0");
    // The gate itself, read by the per-shape check below: a shell exporting it would skip
    // every shape with the wrong reason. Each arm scopes its own value inside the round.
    let _exact = rlx_ir::env::scoped("RLX_Q8_0_EXACT_GEMV", "0");
    let mut frame = [0f64; 2];
    let mut streamed = 0f64;
    for &(k, n, per_frame) in SHAPES {
        let x = activation(k);
        let w = packed(GgmlType::Q8_0, k, n, 1);
        let bytes = w.len();
        if !rlx_cpu::routing::rounds_to_q8k_at(k, n, bytes, QuantScheme::GgufQ8_0) {
            eprintln!("m=1 k={k} n={n}: the integer route is not taken on this host — skipped");
            continue;
        }
        // Small shapes are a fraction of a millisecond a call, so each arm repeats its
        // call enough times to be timed rather than guessed at.
        let reps = (24 << 20) / bytes.max(1);
        let mut f32_route = vec![0f32; n];
        let mut int8_route = vec![0f32; n];
        let times = common::paired(ROUNDS, 2, |arm| {
            // Both arms scope their own value on this thread and restore it when the
            // round ends: a shell exporting `=1` cannot put the f32 route under the
            // int8 label, and the arm's override cannot outlive the harness.
            let _arm = rlx_ir::env::scoped("RLX_Q8_0_EXACT_GEMV", if arm == 0 { "1" } else { "0" });
            let out = if arm == 0 {
                &mut f32_route
            } else {
                &mut int8_route
            };
            let t = std::time::Instant::now();
            for _ in 0..reps {
                gguf_matmul_bt_dispatch_at(&x, &w, out, 1, k, n, QuantScheme::GgufQ8_0, 0);
            }
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        });
        let (a, b) = (times[0], times[1]);
        let mib = bytes as f64 / (1 << 20) as f64;
        eprintln!("m=1 k={k} n={n} ({mib:.2} MiB of packed weight, x{per_frame} per frame):");
        eprintln!(
            "    f32 unpack        {a:8.3} ms  {:5.1} GB/s",
            bytes as f64 / a / 1e6
        );
        // The gap is the two arms' answers to each other, not a kernel error: the integer
        // route rounds the activation, so on this spike-heavy fixture it is the rounding
        // (~1e0). The route's own oracle is `tests/gguf_single_row_cache_pressure.rs`,
        // which holds it to the Q8_K-quantised activation at 1e-4.
        eprintln!(
            "    int8 dot          {b:8.3} ms  {:5.1} GB/s   {:.2}x  \
             (activation rounding: {:.1e} from the f32 arm)",
            bytes as f64 / b / 1e6,
            a / b,
            max_rel_err(&int8_route, &f32_route)
        );
        frame[0] += a * per_frame as f64;
        frame[1] += b * per_frame as f64;
        streamed += bytes as f64 * per_frame as f64 / 1e9;
    }
    if streamed == 0.0 {
        eprintln!("no shape reaches the integer route on this host — nothing to compare");
        return;
    }
    eprintln!(
        "one 16-level frame step: {streamed:.2} GB of packed weight — f32 unpack \
         {:.0} ms, int8 dot {:.0} ms, {:.2}x",
        frame[0],
        frame[1],
        frame[0] / frame[1]
    );
}
