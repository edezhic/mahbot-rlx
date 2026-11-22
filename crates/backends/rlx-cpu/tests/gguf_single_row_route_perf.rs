// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Natalya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Before/after measurement for the single-row packed-multiply routing: per
//! scheme, what a decode pass costs under a dequant-cache budget far below the
//! working set, against the packed arm a cache-off decode takes and against the
//! fused arm the thrash diversion selects.
//!
//! `#[ignore]`d. Gate it on `just throttle` first, then:
//!
//! ```text
//! cargo test -p rlx-cpu --release --test gguf_single_row_route_perf --features backend-tests \
//!     -- --ignored --nocapture
//! ```
//!
//! Its own binary, one `#[test]`: the cache budget and the thrash latch are
//! process-global and the latch never resets, so a second test in the same
//! process would perturb it. (The correctness test lives in
//! `gguf_single_row_cache_pressure.rs` for the same reason.)
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::dequant_cache::clear_dequant_cache;
use rlx_cpu::gguf_matmul::{gguf_matmul_bt, gguf_matmul_bt_dispatch_at};
use rlx_cpu::routing::quantizes_activation;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// An FFN-sized shape: `k*n*4` is 64 MiB per tensor, against the 1 MiB budget.
const K: usize = 2048;
const N: usize = 8192;
/// Distinct weights, so a budget below one slab forces evictions — a single
/// tensor would simply be served from the cache and never thrash.
const TENSORS: usize = 3;
const BUDGET: usize = 1 << 20;
/// Big enough that `cache_thrashing()` stays false even after the earlier rows
/// have accumulated evictions, so the "working set fits" row really hits.
const UNBOUNDED: u64 = 1 << 40;
const SCHEMES: [(QuantScheme, GgmlType); 4] = [
    (QuantScheme::GgufQ4K, GgmlType::Q4K),
    (QuantScheme::GgufQ5K, GgmlType::Q5K),
    (QuantScheme::GgufQ6K, GgmlType::Q6K),
    (QuantScheme::GgufQ8_0, GgmlType::Q8_0),
];

#[test]
#[ignore = "manual kernel timing"]
fn single_row_route_perf() {
    let x = activation(K);
    eprintln!(
        "m=1 k={K} n={N} x{TENSORS}: {} MiB of f32 per scheme, budget {BUDGET} B",
        (K * N * 4 * TENSORS) >> 20
    );

    for (scheme, ggml) in SCHEMES {
        let blobs: Vec<Vec<u8>> = (0..TENSORS).map(|t| packed(ggml, K, N, t)).collect();
        let want: Vec<Vec<f32>> = blobs
            .iter()
            .map(|p| {
                let dq = rlx_gguf::dequant_typed(ggml, p, K * N, "bench weight")
                    .expect("dequant weight");
                (0..N)
                    .map(|j| {
                        dq[j * K..(j + 1) * K]
                            .iter()
                            .zip(&x)
                            .map(|(w, a)| f64::from(*w) * f64::from(*a))
                            .sum::<f64>() as f32
                    })
                    .collect()
            })
            .collect();

        let bench = |label: &str, f: &dyn Fn(usize, &mut [f32])| {
            clear_dequant_cache();
            let mut out = vec![0f32; N];
            for _ in 0..2 {
                for t in 0..TENSORS {
                    f(t, &mut out);
                }
            }
            let iters = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for t in 0..TENSORS {
                    f(t, &mut out);
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / (iters * TENSORS) as f64;
            let err = (0..TENSORS)
                .map(|t| {
                    f(t, &mut out);
                    max_rel_err(&out, &want[t])
                })
                .fold(0f32, f32::max);
            eprintln!("    {label:<40} {ms:7.2} ms/row   max rel err {err:.2e}");
        };
        let dispatch = |t: usize, out: &mut [f32]| {
            gguf_matmul_bt_dispatch_at(&x, &blobs[t], out, 1, K, N, scheme, t * 4096)
        };
        let fused = |t: usize, out: &mut [f32]| gguf_matmul_bt(&x, &blobs[t], out, 1, K, N, scheme);
        // The packed arm both the cache-off dispatch and the direct fused call
        // reach. For a scheme whose arm rounds the activation it is the integer
        // dot, so the row says so and its `max rel err` is that arm's.
        let rounds = quantizes_activation(scheme);
        let packed = if rounds {
            "packed, int8 activations"
        } else {
            "packed, accurate"
        };

        eprintln!("{scheme:?}");
        // Each arm carries its own switches, scoped to it: a process-wide `set` would
        // outlive the arm that asked for it and hand the next one a route it never
        // configured.
        // The working-set-fits case, unchanged by the routing change.
        {
            let _cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "1");
            let _budget = rlx_ir::env::scoped("RLX_DEQUANT_CACHE_MAX_BYTES", UNBOUNDED.to_string());
            bench("accurate, working set fits", &dispatch);
        }
        // The harness's case.
        {
            let _cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "1");
            let _budget = rlx_ir::env::scoped("RLX_DEQUANT_CACHE_MAX_BYTES", BUDGET.to_string());
            bench("accurate, budget < working set (after)", &dispatch);
        }
        // What the harness saw with the budget off.
        {
            let _cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
            bench(&format!("{packed}, RLX_DEQUANT_CACHE=0"), &dispatch);
        }
        // The arm the thrash diversion selects where its numerics are unchanged,
        // and what a caller of `gguf_matmul_bt` gets directly.
        {
            let _cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "1");
            let _budget = rlx_ir::env::scoped("RLX_DEQUANT_CACHE_MAX_BYTES", BUDGET.to_string());
            bench(
                &format!("{packed}, fused (thrash-diversion target)"),
                &fused,
            );
        }
        // A rounding arm has an exact counterpart behind an env switch — the row
        // shows what the rounding buys. (The six-bit switch is read by the aarch64
        // kernel only, so an x86-64 `int8-gemv` host repeats the row above.)
        let switch = match scheme {
            QuantScheme::GgufQ4K => Some("RLX_Q4K_EXACT_GEMV"),
            QuantScheme::GgufQ6K => Some("RLX_Q6K_EXACT_GEMV"),
            _ => None,
        };
        if let (true, Some(switch)) = (rounds, switch) {
            let _exact = rlx_ir::env::scoped(switch, "1");
            bench(&format!("fused, exact activations ({switch})"), &fused);
        }
    }

    clear_dequant_cache();
}
