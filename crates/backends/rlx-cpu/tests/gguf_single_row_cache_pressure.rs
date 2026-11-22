// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Natalya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A single-row packed (GGUF) multiply must not change its *route* when the
//! dequant cache cannot hold the model — one row per call over packed weights
//! whose f32 form exceeds the budget, the decode shape of a quantized transformer
//! harness.
//!
//! Two configurations are pinned, and they are the two the routing has to be
//! stable across.
//!
//! *Under a cache that thrashes* (phases 2 and 4, after phase 1's control that a
//! budget which fits still serves the cache): cumulative evictions used to
//! re-route a decode GEMV to an activation-rounding arm (~1e-2 off the f32
//! route) and never revert, so the first pass of a reused, once-compiled graph was
//! accurate and every later pass was not. The test runs the same multiply three
//! times under a budget far below the working set and requires every pass to agree
//! with the first. The schemes whose arms round
//! ([`rlx_cpu::routing::quantizes_activation`]) therefore keep the
//! cached f32 route whatever the budget does; the rest do the same total work on
//! either route. (A build with no vendor BLAS has no cached slab to keep, so there
//! the packed arm runs either way and the oracle is that arm's.)
//!
//! *With the cache switched off* (phase 3), which is the product's own form: the
//! single-row multiplies are the packed routes, and each is held to [`TOL`]
//! against an f64 dot of the dequantised weight — with the activation rounded the
//! way that route rounds it, which pins the route rather than only the size of its
//! rounding. The rounding this harness can build is Q8_K, which every K-quant arm
//! uses — `int8-gemv`'s three included — so that kernel's 32-element arms (Q8_0,
//! Q4_0, Q4_1; x86-64 only, default off), which round to int8 blocks instead, are
//! named and skipped rather than measured against an oracle they do not compute.
//!
//! One `#[test]`: the cache budget, the thrash latch and the cache itself are
//! process-global, and the latch never resets, so a second test in this process
//! would perturb this one. The routing measurement is a separate binary,
//! `gguf_single_row_route_perf.rs` — hence a separate process.
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::dequant_cache::{clear_dequant_cache, gguf_weight_f32};
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at;
use rlx_cpu::routing::{quantizes_activation, rounds_to_q8k};
use rlx_gguf::{GgmlType, Q8K_BLOCK_BYTES, QK_K};
use rlx_ir::quant::QuantScheme;

/// `k` is a whole number of 256-element K-quant blocks (and of Q8_0's 32).
const K: usize = 1024;
const N: usize = 64;
/// Far below one f32 slab of a single weight (k*n*4 = 256 KiB), so every insert
/// evicts and the thrash signal trips after the first couple of lookups.
const BUDGET: usize = 64 * 1024;
/// The accuracy reference is `dequant → f32 dot`; an activation-quantising arm is
/// a ~1e-2-class downgrade of the *activation*, so 1e-4 separates "the same 32-bit
/// computation" from "activation quantized" — and each case is measured against an
/// oracle that quantizes exactly where its route does, so the same 1e-4 bounds it.
const TOL: f32 = 1e-4;

/// The packed schemes the harness multiplies with: the 4/5/6-bit K-quants of the
/// model bodies, and both 8-bit schemes — the report identifies the audio and
/// predictor weights only as "8-bit".
///
/// The order is load-bearing: the thrash latch trips inside the first scheme's own
/// passes (its second lookup evicts the first), so the scheme listed first is the only
/// one this test ever measures with the latch untripped — the only position from which
/// a scheme is seen on both sides of the trip, which is what the route-change
/// assertion needs. Q6_K leads because it is the scheme whose single-row arm rounds on
/// the reference host (aarch64's integer dot): the class whose route must not change at
/// all.
const SCHEMES: [(QuantScheme, GgmlType); 5] = [
    (QuantScheme::GgufQ6K, GgmlType::Q6K),
    (QuantScheme::GgufQ4K, GgmlType::Q4K),
    (QuantScheme::GgufQ5K, GgmlType::Q5K),
    (QuantScheme::GgufQ8_0, GgmlType::Q8_0),
    (QuantScheme::GgufQ8K, GgmlType::Q8K),
];

/// One activation row rounded to Q8_K and read back: the activation an
/// activation-quantising arm multiplies with.
fn quantised_activation_row(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0u8; K / QK_K * Q8K_BLOCK_BYTES];
    rlx_gguf::quantize_q8_k_row(x, &mut y);
    let mut out = vec![0f32; K];
    for b in 0..K / QK_K {
        let mut blk = [0f32; QK_K];
        rlx_gguf::dequant_q8_k_block(&y[b * Q8K_BLOCK_BYTES..], &mut blk);
        out[b * QK_K..(b + 1) * QK_K].copy_from_slice(&blk);
    }
    out
}

/// `x[m,k] @ dequant(w)[n,k]^T` in f64, narrowed to f32 — an independent oracle.
/// `quantised` rounds each activation row to Q8_K first, which is the activation an
/// activation-quantising arm sees; every other route multiplies the f32 row.
fn reference(x: &[f32], w: &[u8], ggml: GgmlType, m: usize, quantised: bool) -> Vec<f32> {
    use std::borrow::Cow;
    let dq = rlx_gguf::dequant_typed(ggml, w, K * N, "test weight").expect("test weight dequant");
    let mut out = vec![0f32; m * N];
    for row in 0..m {
        let act = &x[row * K..(row + 1) * K];
        let act = if quantised {
            Cow::Owned(quantised_activation_row(act))
        } else {
            Cow::Borrowed(act)
        };
        for j in 0..N {
            let mut acc = 0f64;
            for p in 0..K {
                acc += f64::from(dq[j * K + p]) * f64::from(act[p]);
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
    // asserts the default CPU behaviour whatever the developer's shell holds. Scoped,
    // not process-wide: each is read on this thread, and the sibling harnesses in this
    // crate's other binaries are the same idiom. (`rlx_ir::env` overrides win over the
    // process environment.) The cache switch is the first push and stays for the whole
    // test; phase 3 shadows it with a nested guard rather than ending it.
    let _cache_on = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "1");
    let _legacy = rlx_ir::env::scoped("RLX_GGUF_MATMUL_LEGACY", "0");
    let _fused_min_n = rlx_ir::env::scoped("RLX_Q4K_FUSED_MIN_N", usize::MAX.to_string());
    let _q4k_exact = rlx_ir::env::scoped("RLX_Q4K_EXACT_GEMV", "0");
    let _q6k_exact = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "0");

    let blobs: Vec<Vec<(usize, Vec<u8>)>> = SCHEMES.iter().map(|(_, g)| tensors(*g)).collect();
    let x = activation(K);

    // Phase 1 — a budget that fits the working set still serves the cache, so the
    // fitting case keeps its fast route. Done first, while no eviction has tripped
    // the (monotone) thrash latch.
    let fits = rlx_ir::env::scoped("RLX_DEQUANT_CACHE_MAX_BYTES", (4 << 20).to_string());
    clear_dequant_cache();
    let (w_off, bytes) = &blobs[0][0];
    let a = gguf_weight_f32(*w_off, bytes, K, N, SCHEMES[0].0);
    let b = gguf_weight_f32(*w_off, bytes, K, N, SCHEMES[0].0);
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "a budget that fits must still serve a repeated lookup from the cache"
    );
    drop(fits);

    // Phase 2 — budget far below the working set. Every pass of the same dispatch
    // must match the engine's own 32-bit computation.
    //
    // The harness reaches this state with the default budget (a quarter of RAM)
    // and a model whose f32 form exceeds it; forcing a tiny budget reaches the
    // same latch sooner and from any host, since the latch only asks whether more
    // than a quarter of the budget has been evicted.
    //
    // No scheme's answer may move when the budget shrinks — that is the contract, and
    // the oracle is the f32 dequant route's. A scheme whose arm rounds keeps that route
    // by design; one whose arm does not diverts to the fused kernel, which is the same
    // f32 arithmetic over the same `k` in another summation order and lands inside the
    // same bound.
    //
    // A build with no vendor BLAS, and a portable run of one that has it, have no
    // cached f32 route for this oracle to be: the first sends the multiply to the fused
    // kernel, with Q4_K's single row to its exact packed kernel, and the second sends
    // every scheme to the fused kernel — where Q4_K takes the Q8_K arm, since this test
    // masks `RLX_Q4K_EXACT_GEMV=0` above. The passes are still held to each other below
    // (the drift is the test's subject), and phase 3 holds those packed routes to their
    // own oracles.
    let cached = cfg!(rlx_cpu_blas) && !rlx_cpu::config::portable();
    let _budget = rlx_ir::env::scoped("RLX_DEQUANT_CACHE_MAX_BYTES", BUDGET.to_string());
    clear_dequant_cache();
    for ((scheme, ggml), blobs) in SCHEMES.iter().zip(&blobs) {
        let wants: Vec<Vec<f32>> = if cached {
            blobs
                .iter()
                .map(|(_, w)| reference(&x, w, *ggml, 1, false))
                .collect()
        } else {
            Vec::new()
        };
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
            if cached {
                for (t, out) in got.iter().enumerate() {
                    let err = max_rel_err(out, &wants[t]);
                    assert!(
                        err < TOL,
                        "{scheme:?} tensor {t} pass {pass}: max rel err {err} vs the engine's \
                         32-bit dequant route — the single-row multiply must not change its route \
                         under cache pressure"
                    );
                }
            }
            match &first {
                // Two classes of scheme here, and the bound differs because the design
                // does. One whose arm rounds keeps the cached route on both sides of the
                // latch (`prefer_cached_blas`), so its passes are one route's arithmetic
                // and the bound only has to separate that from a route that moved under
                // it — the ~1e-2 class of an activation rounding. One whose arm does not
                // round diverts to the fused kernel, the same dequant work in another
                // summation order, and the same bound admits that order.
                Some(prev) => {
                    let bound = if quantizes_activation(*scheme) {
                        1e-5
                    } else {
                        TOL
                    };
                    for (t, out) in got.iter().enumerate() {
                        let drift = max_rel_err(out, &prev[t]);
                        assert!(
                            drift < bound,
                            "{scheme:?} tensor {t}: pass {pass} drifted from pass 0 by {drift} — \
                             the result depends on cache state"
                        );
                    }
                }
                None => first = Some(got),
            }
        }
    }

    // Phase 3 — the product's own form: the dequant cache switched off. The
    // single-row multiplies are the packed routes, so each is held to the oracle for
    // the activation its own arm multiplies with. A build whose arm rounds to
    // 32-element int8 blocks has no oracle here and says so rather than passing.
    //
    // A nested guard shadows the phase-1/2 one for these passes without ending it:
    // dropping the outer guard would truncate the whole thread-local stack and un-mask
    // every knob pushed after it.
    {
        let _cache_off = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
        for ((scheme, ggml), blobs) in SCHEMES.iter().zip(&blobs) {
            if quantizes_activation(*scheme) && !rounds_to_q8k(*scheme) {
                println!(
                    "{scheme:?}: this build's single-row arm rounds to 32-element int8 blocks, \
                     which this harness has no oracle for — not measured here"
                );
                continue;
            }
            let quantised = rounds_to_q8k(*scheme);
            for (t, (w_off, bytes)) in blobs.iter().enumerate() {
                let want = reference(&x, bytes, *ggml, 1, quantised);
                let mut out = vec![0f32; N];
                gguf_matmul_bt_dispatch_at(&x, bytes, &mut out, 1, K, N, *scheme, *w_off);
                let err = max_rel_err(&out, &want);
                assert!(
                    err < TOL,
                    "{scheme:?} tensor {t} with the cache off: max rel err {err}"
                );
            }
        }
    }

    // Phase 4 — the multi-row path shares the cache and must stay accurate while
    // thrashing: the outer `_cache_on` guard's value is back now that phase 3's shadow has
    // ended, and phase 2's budget is still in force, so the latch stays tripped. Its route
    // is f32 either way — the tiled BLAS slab where `admits` takes the shape, the fused
    // kernel where the host cannot stripe it — so the activation is never rounded and the
    // oracle is the plain f32 dequant. Both six-bit
    // and four-bit bodies, which are the two the models carry — named here rather than
    // sliced out of `SCHEMES`, whose order serves the latch above.
    let m = 2;
    let xm: Vec<f32> = (0..m * K).map(|i| 0.001 * i as f32 - 0.5).collect();
    for (scheme, ggml) in [
        (QuantScheme::GgufQ6K, GgmlType::Q6K),
        (QuantScheme::GgufQ4K, GgmlType::Q4K),
    ] {
        for (w_off, bytes) in tensors(ggml) {
            let want = reference(&xm, &bytes, ggml, m, false);
            let mut out = vec![0f32; m * N];
            gguf_matmul_bt_dispatch_at(&xm, &bytes, &mut out, m, K, N, scheme, w_off);
            let err = max_rel_err(&out, &want);
            assert!(
                err < TOL,
                "{scheme:?} m={m} w_off={w_off}: max rel err {err}"
            );
        }
    }

    clear_dequant_cache();
}
