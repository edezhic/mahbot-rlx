// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Natalya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A single-row packed (GGUF) multiply must keep the engine's accurate 32-bit
//! numerics when the dequant cache cannot hold the model — one row per call over
//! packed weights whose f32 form exceeds the budget, the decode shape of a
//! quantized transformer harness.
//!
//! The regression this pins: cumulative evictions used to re-route the Q4_K
//! decode GEMV to the Q8_K activation arm (~1e-2 off the f32 route) and never
//! reverted, so the first pass of a reused, once-compiled graph was accurate and
//! every later pass was not. The test runs the same multiply three times under a
//! budget far below the working set and requires every pass to match the engine's
//! own dequant-to-f32 computation.
//!
//! One `#[test]`: the cache budget, the thrash latch and the cache itself are
//! process-global, and the latch never resets, so a second test in this process
//! would perturb this one. The routing measurement is a separate binary,
//! `gguf_single_row_route_perf.rs` — hence a separate process.
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::dequant_cache::{clear_dequant_cache, gguf_weight_f32};
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// `k` is a whole number of 256-element K-quant blocks (and of Q8_0's 32).
const K: usize = 1024;
const N: usize = 64;
/// Far below one f32 slab of a single weight (k*n*4 = 256 KiB), so every insert
/// evicts and the thrash signal trips after the first couple of lookups.
const BUDGET: usize = 64 * 1024;
/// The accuracy reference is `dequant → f32 dot`; the Q8_K arm is a ~1e-2-class
/// downgrade (0.12–0.40 on this data), so 1e-4 separates "the same 32-bit
/// computation" from "activation quantized".
const TOL: f32 = 1e-4;

/// The packed schemes the harness multiplies with: the 4/5/6-bit K-quants of the
/// model bodies, and both 8-bit schemes — the report identifies the audio and
/// predictor weights only as "8-bit".
const SCHEMES: [(QuantScheme, GgmlType); 5] = [
    (QuantScheme::GgufQ4K, GgmlType::Q4K),
    (QuantScheme::GgufQ5K, GgmlType::Q5K),
    (QuantScheme::GgufQ6K, GgmlType::Q6K),
    (QuantScheme::GgufQ8_0, GgmlType::Q8_0),
    (QuantScheme::GgufQ8K, GgmlType::Q8K),
];

/// `x[m,k] @ dequant(w)[n,k]^T` in f64, narrowed to f32 — an independent oracle.
fn reference(x: &[f32], w: &[u8], ggml: GgmlType, m: usize) -> Vec<f32> {
    let dq = rlx_gguf::dequant_typed(ggml, w, K * N, "test weight").expect("test weight dequant");
    let mut out = vec![0f32; m * N];
    for row in 0..m {
        for j in 0..N {
            let mut acc = 0f64;
            for p in 0..K {
                acc += f64::from(dq[j * K + p]) * f64::from(x[row * K + p]);
            }
            out[row * N + j] = acc as f32;
        }
    }
    out
}

/// Three distinct tensors per scheme, addressed at distinct offsets as the engine
/// sees the harness's file-backed weights re-bound into arena slots.
fn tensors(ggml: GgmlType) -> Vec<(usize, Vec<u8>)> {
    (0..3).map(|t| (t * 4096, packed(ggml, K, N, t))).collect()
}

#[test]
fn single_row_packed_multiply_is_cache_pressure_independent() {
    // Mask the ambient knobs that could select a different route, so the test
    // asserts the default CPU behaviour whatever the developer's shell holds.
    // (`rlx_ir::env` overrides win over the process environment.)
    rlx_ir::env::set("RLX_DEQUANT_CACHE", "1");
    rlx_ir::env::set("RLX_GGUF_MATMUL_LEGACY", "0");
    rlx_ir::env::set("RLX_Q4K_FUSED_MIN_N", usize::MAX.to_string());
    rlx_ir::env::set("RLX_Q4K_EXACT_GEMV", "0");

    let blobs: Vec<Vec<(usize, Vec<u8>)>> = SCHEMES.iter().map(|(_, g)| tensors(*g)).collect();
    let x = activation(K);

    // Phase 1 — a budget that fits the working set still serves the cache, so the
    // fitting case keeps its fast route. Done first, while no eviction has tripped
    // the (monotone) thrash latch.
    rlx_ir::env::set("RLX_DEQUANT_CACHE_MAX_BYTES", (4 << 20).to_string());
    clear_dequant_cache();
    let (w_off, bytes) = &blobs[0][0];
    let a = gguf_weight_f32(*w_off, bytes, K, N, SCHEMES[0].0);
    let b = gguf_weight_f32(*w_off, bytes, K, N, SCHEMES[0].0);
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "a budget that fits must still serve a repeated lookup from the cache"
    );

    // Phase 2 — budget far below the working set. Every pass of the same dispatch
    // must match the engine's own 32-bit computation.
    //
    // The harness reaches this state with the default budget (a quarter of RAM)
    // and a model whose f32 form exceeds it; forcing a tiny budget reaches the
    // same latch sooner and from any host, since the latch only asks whether more
    // than a quarter of the budget has been evicted.
    rlx_ir::env::set("RLX_DEQUANT_CACHE_MAX_BYTES", BUDGET.to_string());
    clear_dequant_cache();
    for ((scheme, ggml), blobs) in SCHEMES.iter().zip(&blobs) {
        let wants: Vec<Vec<f32>> = blobs
            .iter()
            .map(|(_, w)| reference(&x, w, *ggml, 1))
            .collect();
        let mut first: Option<Vec<Vec<f32>>> = None;
        for pass in 0..3 {
            let got: Vec<Vec<f32>> = blobs
                .iter()
                .map(|(w_off, bytes)| {
                    let mut out = vec![0f32; N];
                    gguf_matmul_bt_dispatch_at(&x, bytes, &mut out, 1, K, N, *scheme, *w_off);
                    out
                })
                .collect();
            for (t, out) in got.iter().enumerate() {
                let err = max_rel_err(out, &wants[t]);
                assert!(
                    err < TOL,
                    "{scheme:?} tensor {t} pass {pass}: max rel err {err} vs the engine's 32-bit \
                     dequant route — the single-row multiply must not fall back to activation \
                     quantization under cache pressure"
                );
            }
            match &first {
                // Two accurate 32-bit routes may sum in different orders
                // (~1e-6); what must not appear is a ~1e-2 class jump.
                Some(prev) => {
                    for (t, out) in got.iter().enumerate() {
                        let drift = max_rel_err(out, &prev[t]);
                        assert!(
                            drift < 1e-5,
                            "{scheme:?} tensor {t}: pass {pass} drifted from pass 0 by {drift} — \
                             the result depends on cache state"
                        );
                    }
                }
                None => first = Some(got),
            }
        }
    }

    // Phase 3 — the multi-row path shares the cache and must stay accurate while
    // thrashing (its fused GEMM keeps the activation in f32).
    let (scheme, ggml) = SCHEMES[0];
    let m = 2;
    let xm: Vec<f32> = (0..m * K).map(|i| 0.001 * i as f32 - 0.5).collect();
    for (w_off, bytes) in &blobs[0] {
        let want = reference(&xm, bytes, ggml, m);
        let mut out = vec![0f32; m * N];
        gguf_matmul_bt_dispatch_at(&xm, bytes, &mut out, m, K, N, scheme, *w_off);
        let err = max_rel_err(&out, &want);
        assert!(err < TOL, "m={m} w_off={w_off}: max rel err {err}");
    }

    rlx_ir::env::unset("RLX_DEQUANT_CACHE_MAX_BYTES");
    rlx_ir::env::unset("RLX_DEQUANT_CACHE");
    rlx_ir::env::unset("RLX_GGUF_MATMUL_LEGACY");
    rlx_ir::env::unset("RLX_Q4K_FUSED_MIN_N");
    rlx_ir::env::unset("RLX_Q4K_EXACT_GEMV");
    clear_dequant_cache();
}
