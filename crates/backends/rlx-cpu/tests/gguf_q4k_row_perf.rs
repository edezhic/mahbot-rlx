// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The floor the four-bit single-row multiply is held to: at the product's decode shapes
//! its per-row time must be no worse than the six-bit arm's on the same shape in the same
//! session.
//!
//! Both arms go through the dispatch the product uses with the dequant cache off, as the
//! processor device path leaves it — which is what puts the four-bit arm on the integer
//! Q8_K kernel rather than a cached f32 slab. Rounds alternate the arms so a machine that
//! drifts cannot favour one, and the ratio, not either absolute, is the figure: the same
//! arm reads 13.7-22.7 GB/s under load against 27-37 quiet.
//!
//! The floor is the aarch64 arm's, so the harness refuses elsewhere rather than printing a
//! ratio between two other kernels.
//!
//! `#[ignore]`d — it prints, it does not assert. Gate it on `just throttle` first, then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --test gguf_q4k_row_perf \
//!     --features backend-tests -- --ignored --nocapture
//! ```
mod common;

use common::{activation, packed, paired};
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// The product's decode shapes: the attention projections, `gate`+`up`, `ffn_down` and
/// the output head.
const SHAPES: &[(usize, usize)] = &[
    (2048, 1024),
    (2048, 2048),
    (2048, 6144),
    (6144, 2048),
    (2048, 151936),
];
const ROUNDS: usize = 5;

/// One packed row repeated `n` times: a decode pass reads every row exactly once, so the
/// rows need only be the right size — and the head's `k*n` f32 fixture would be 1.2 GiB.
fn streamed(ggml: GgmlType, k: usize, n: usize) -> Vec<u8> {
    let row = packed(ggml, k, 1, 1);
    row.iter().copied().cycle().take(n * row.len()).collect()
}

#[test]
#[ignore = "manual kernel timing"]
fn q4k_row_floor() {
    if !cfg!(target_arch = "aarch64") {
        // The four-bit row is another kernel here, so the ratio would be a comparison the
        // floor is not about.
        eprintln!("four-bit row floor: an aarch64 measurement — nothing to compare here");
        return;
    }
    // The product's run form, plus the levers that would otherwise put a different kernel
    // under a label: `prefer_cached_blas` reads `RLX_REPRODUCIBLE` before it reads the
    // cache, and each scheme's exact arm is not the integer one the floor compares.
    // Scoped to this thread, which is where the dispatch reads them.
    let _no_cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
    let _reproducible = rlx_ir::env::scoped("RLX_REPRODUCIBLE", "0");
    let _q4k_exact = rlx_ir::env::scoped("RLX_Q4K_EXACT_GEMV", "0");
    let _q6k_exact = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "0");
    let schemes = [
        (QuantScheme::GgufQ4K, GgmlType::Q4K),
        (QuantScheme::GgufQ6K, GgmlType::Q6K),
    ];
    eprintln!(
        "the ratio is the figure; a cell's GB/s is its own working set's — only the head \
         streams, the small shapes are re-read from cache across the repeats"
    );
    for &(k, n) in SHAPES {
        let x = activation(k);
        let blobs = schemes.map(|(_, ggml)| streamed(ggml, k, n));
        let mut out = vec![0f32; n];
        // Small shapes are a fraction of a millisecond a call, so each arm repeats its call
        // enough times to be timed rather than guessed at. Both arms repeat equally and the
        // ratio is what is read, so a warm weight favours neither.
        let reps = ((24 << 20) / blobs[0].len()).max(1);
        let times = paired(ROUNDS, 2, |arm| {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                gguf_matmul_bt_dispatch_at(&x, &blobs[arm], &mut out, 1, k, n, schemes[arm].0, 0);
            }
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        });
        eprintln!(
            "m=1 k={k} n={n} ({} MiB four-bit, {} MiB six-bit):",
            blobs[0].len() >> 20,
            blobs[1].len() >> 20
        );
        for (label, ms, bytes) in [
            ("four-bit", times[0], blobs[0].len()),
            ("six-bit", times[1], blobs[1].len()),
        ] {
            eprintln!(
                "    {label:<9} {ms:8.3} ms  {:6.1} ns/row  {:5.1} GB/s",
                ms * 1e6 / n as f64,
                bytes as f64 / ms / 1e6
            );
        }
        eprintln!(
            "    four-bit / six-bit per row: {:.2}x",
            times[0] / times[1]
        );
    }
}
