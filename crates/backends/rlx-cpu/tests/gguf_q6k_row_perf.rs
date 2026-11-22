// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The single-row Q6_K multiply, f32 dequantisation against the integer dot: the
//! paired reading behind `gguf_q6k`'s route.
//!
//! The route is aarch64-only. On any other target both arms run the f32 route and
//! `RLX_Q6K_EXACT_GEMV` is a no-op, so the harness says so and returns rather than
//! printing a comparison that is not one.
//!
//! `#[ignore]`d — it prints, it does not assert. Gate it on `just throttle` first,
//! then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --test gguf_q6k_row_perf \
//!     --features backend-tests -- --ignored --nocapture
//! ```
//!
//! One `#[test]`: the two arms must alternate inside one session for the comparison
//! to mean anything. Both of its switches are scoped to the measuring thread.
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// The two Q6_K weights every generated token reads: the output head (a width the
/// harness can pack — the shipped one is 151936 rows of the same shape) and an
/// `ffn_down`.
const SHAPES: &[(usize, usize)] = &[(2048, 16384), (6144, 2048)];
const ROUNDS: usize = 5;

#[test]
#[ignore = "manual kernel timing"]
fn q6k_row_route_perf() {
    if !rlx_cpu::routing::SIX_BIT_ROUTE_COMPILED {
        // The route is aarch64-only and `RLX_Q6K_EXACT_GEMV` is a no-op elsewhere,
        // so both arms would run the same f32 route under two labels.
        eprintln!("six-bit single-row route: not compiled on this target — nothing to compare");
        return;
    }
    // The product's arrangement: the processor switches the dequant cache off, so
    // a single row reaches the fused arm rather than a cached f32 slab. Scoped to
    // this thread and held for the whole measurement: a process-wide `set` would
    // outlive the harness, and this is the only reader.
    let _no_cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
    for &(k, n) in SHAPES {
        let x = activation(k);
        let w = packed(GgmlType::Q6K, k, n, 1);
        let bytes = n
            * (k / QuantScheme::GgufQ6K.gguf_block_size() as usize)
            * QuantScheme::GgufQ6K.gguf_block_bytes() as usize;
        let mut f32_route = vec![0f32; n];
        let mut int8_route = vec![0f32; n];
        let times = common::paired(ROUNDS, 2, |arm| {
            // Both arms scope their own value on this thread and restore it when the
            // round ends: a shell exporting `=1` cannot put the f32 route under the
            // int8 label, and the arm's override cannot outlive the harness.
            let _arm = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", if arm == 0 { "1" } else { "0" });
            let out = if arm == 0 {
                &mut f32_route
            } else {
                &mut int8_route
            };
            let t = std::time::Instant::now();
            gguf_matmul_bt_dispatch_at(&x, &w, out, 1, k, n, QuantScheme::GgufQ6K, 0);
            t.elapsed().as_secs_f64() * 1e3
        });
        let (a, b) = (times[0], times[1]);
        eprintln!("m=1 k={k} n={n} ({} MiB of packed weight):", bytes >> 20);
        eprintln!(
            "    f32 dequant       {a:8.3} ms  {:5.1} GB/s",
            bytes as f64 / a / 1e6
        );
        eprintln!(
            "    int8 dot          {b:8.3} ms  {:5.1} GB/s   {:.2}x  err {:.1e}",
            bytes as f64 / b / 1e6,
            a / b,
            max_rel_err(&int8_route, &f32_route)
        );
    }
}
