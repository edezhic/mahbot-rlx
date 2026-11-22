// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where a packed weight's multi-row multiply should run, on the shapes the
//! product actually asks for: the fused kernel as it stands, one vendor call over
//! a whole f32 slab, or the tiled route striped across the pool.
//!
//! `#[ignore]`d — it prints, it does not assert. Gate it on `just throttle` first,
//! then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --test gguf_multi_row_route_perf \
//!     --features backend-tests -- --ignored --nocapture
//! ```
//!
//! One `#[test]`: every arm is timed against the whole pool, so a second test in
//! the process would take workers from the arm it is timing. Each round runs every
//! arrangement on every shape, alternating their order so a drifting machine
//! cannot favour one; the reported figure is the median of the rounds.
mod common;

use common::{activation, max_rel_err, packed};
use rlx_cpu::blas::sgemm_bt;
use rlx_cpu::gguf_matmul::gguf_matmul_bt;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// One multiply the product runs: the encoder's 24 blocks at 64 rows a window,
/// the body's prompt prefill at 79, and a wide head at a width the test can pack.
const SHAPES: &[Shape] = &[
    Shape {
        name: "enc qkv  m64 k1024 n1024",
        scheme: QuantScheme::GgufQ8_0,
        ggml: GgmlType::Q8_0,
        m: 64,
        k: 1024,
        n: 1024,
    },
    Shape {
        name: "enc up   m64 k1024 n4096",
        scheme: QuantScheme::GgufQ8_0,
        ggml: GgmlType::Q8_0,
        m: 64,
        k: 1024,
        n: 4096,
    },
    Shape {
        name: "enc down m64 k4096 n1024",
        scheme: QuantScheme::GgufQ8_0,
        ggml: GgmlType::Q8_0,
        m: 64,
        k: 4096,
        n: 1024,
    },
    Shape {
        name: "body qkv m79 k2048 n2048",
        scheme: QuantScheme::GgufQ4K,
        ggml: GgmlType::Q4K,
        m: 79,
        k: 2048,
        n: 2048,
    },
    Shape {
        name: "body up  m79 k2048 n6144",
        scheme: QuantScheme::GgufQ4K,
        ggml: GgmlType::Q4K,
        m: 79,
        k: 2048,
        n: 6144,
    },
    Shape {
        name: "body dn  m79 k6144 n2048",
        scheme: QuantScheme::GgufQ6K,
        ggml: GgmlType::Q6K,
        m: 79,
        k: 6144,
        n: 2048,
    },
    Shape {
        name: "head     m79 k2048 n8192",
        scheme: QuantScheme::GgufQ6K,
        ggml: GgmlType::Q6K,
        m: 79,
        k: 2048,
        n: 8192,
    },
];

/// The row counts a short prompt spans. The sweep is where a crossover would show,
/// and it shows the opposite: the fused route is already 1.5x behind at two rows.
const ROW_SWEEP: &[usize] = &[2, 4, 8, 16, 32, 79];

struct Shape {
    name: &'static str,
    scheme: QuantScheme,
    ggml: GgmlType,
    m: usize,
    k: usize,
    n: usize,
}

const ROUNDS: usize = 5;

/// The whole-tensor f32 image, filled a row at a time in parallel — the "slab
/// built in tiles" arrangement's own half of the job.
fn fill_slab(scheme: QuantScheme, ggml: GgmlType, w: &[u8], slab: &mut [f32], k: usize) {
    use rayon::prelude::*;
    let row_bytes = k / scheme.gguf_block_size() as usize * scheme.gguf_block_bytes() as usize;
    slab.par_chunks_mut(k).enumerate().for_each(|(j, dst)| {
        let dq = rlx_gguf::dequant_typed(ggml, &w[j * row_bytes..(j + 1) * row_bytes], k, "row")
            .expect("dequant row");
        dst.copy_from_slice(&dq);
    });
}

#[test]
#[ignore = "manual kernel timing"]
fn multi_row_route_perf() {
    eprintln!("{} workers", rlx_cpu::pool::num_threads());
    for shape in SHAPES {
        let x: Vec<f32> = (0..shape.m).flat_map(|_| activation(shape.k)).collect();
        let w = packed(shape.ggml, shape.k, shape.n, 1);
        // The whole-tensor f32 slab, dequantised inside the timed region: that is
        // the arrangement's own cost, and leaving it out would price the dequant at
        // nothing.
        let mut slab = vec![0f32; shape.k * shape.n];
        let mut fused = vec![0f32; shape.m * shape.n];
        let mut one_call = vec![0f32; shape.m * shape.n];
        let mut tiled = vec![0f32; shape.m * shape.n];

        // Warm every arm once: the first call pays the BLAS's own first-touch. The route
        // switch belongs to the arm, not to the multiply: scoped before the timer, so no
        // arm is charged for configuring itself and the next arm does not inherit it.
        // (The slab arm never asks — it calls `sgemm_bt` itself — so the value is
        // immaterial to it.)
        let mut run = |arm: usize, out: &mut [f32]| match arm {
            1 => {
                fill_slab(shape.scheme, shape.ggml, &w, &mut slab, shape.k);
                sgemm_bt(&x, &slab, out, shape.m, shape.k, shape.n, 1.0);
            }
            _ => gguf_matmul_bt(&x, &w, out, shape.m, shape.k, shape.n, shape.scheme),
        };
        for (arm, out) in [(0, &mut fused), (1, &mut one_call), (2, &mut tiled)] {
            let _route =
                rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", if arm == 0 { "0" } else { "1" });
            run(arm, out);
        }

        let times = common::paired(ROUNDS, 3, |arm| {
            let _route =
                rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", if arm == 0 { "0" } else { "1" });
            let out = match arm {
                0 => &mut fused,
                1 => &mut one_call,
                _ => &mut tiled,
            };
            let t = std::time::Instant::now();
            run(arm, out);
            t.elapsed().as_secs_f64() * 1e3
        });
        let flops = 2.0 * (shape.m * shape.k * shape.n) as f64;
        // The third arm asks for the tiled route; whether it runs is the admission's call,
        // taken here under the same switch value the arm ran under, so the label names the
        // route this host actually took rather than the one the shell asked for.
        let tiled_why = {
            let _tiled = rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", "1");
            rlx_cpu::routing::tiled_refusal(shape.m, shape.k, shape.n, shape.scheme)
        };
        eprintln!("{}:", shape.name);
        for (arm, label) in [
            "fused",
            "slab built in tiles, one call",
            match tiled_why {
                None => "tiled, striped across the pool",
                Some(_) => "not tiled",
            },
        ]
        .iter()
        .enumerate()
        {
            let ms = times[arm];
            eprintln!(
                "    {label:<28} {ms:8.3} ms  {:6.1} GFLOP/s  err {:.1e}",
                flops / ms / 1e6,
                max_rel_err(
                    match arm {
                        0 => &fused,
                        1 => &one_call,
                        _ => &tiled,
                    },
                    &one_call
                ),
            );
        }
        if let Some(why) = tiled_why {
            eprintln!("    (the tiled route refuses this shape on this host: {why})");
        }
    }

    // A batched decode at the kernel: four tokens in one four-row pass against
    // four one-row passes. The one-row arm is the int8 dot the decode runs; the
    // four-row arm asks for the tiled route, and the row sweep below is where the
    // other multi-row route, the block-fused GEMM, is its opponent.
    let (k, n) = (2048usize, 6144usize);
    let w = packed(GgmlType::Q4K, k, n, 1);
    let lanes = 4;
    let x: Vec<f32> = (0..lanes).flat_map(|_| activation(k)).collect();
    let mut four_rows = vec![0f32; lanes * n];
    let mut one_row = vec![0f32; lanes * n];
    {
        // The decode's own form: no cache, and the tiled route asked for explicitly.
        let _no_cache = rlx_ir::env::scoped("RLX_DEQUANT_CACHE", "0");
        let _tiled = rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", "1");
        let times = common::paired(ROUNDS, 2, |arm| {
            let t = std::time::Instant::now();
            if arm == 0 {
                for lane in 0..lanes {
                    rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at(
                        &x[lane * k..(lane + 1) * k],
                        &w,
                        &mut one_row[lane * n..(lane + 1) * n],
                        1,
                        k,
                        n,
                        QuantScheme::GgufQ4K,
                        0,
                    );
                }
            } else {
                gguf_matmul_bt(&x, &w, &mut four_rows, lanes, k, n, QuantScheme::GgufQ4K);
            }
            t.elapsed().as_secs_f64() * 1e3
        });
        let (a, b) = (times[0], times[1]);
        // Same question as the shape table, and inside the block that holds the arm's own
        // switch value.
        let tiled_why = rlx_cpu::routing::tiled_refusal(lanes, k, n, QuantScheme::GgufQ4K);
        let label = match tiled_why {
            None => "tiled",
            Some(_) => "not tiled",
        };
        eprintln!(
            "{lanes} lanes, Q4_K k={k} n={n}, per token: one-row int8 dot {:.3} ms, four-row {label} {:.3} ms ({:.2}x)",
            a / lanes as f64,
            b / lanes as f64,
            a / b
        );
        if let Some(why) = tiled_why {
            eprintln!("    (the tiled route refuses this shape on this host: {why})");
        }
    }

    // The row floor: the same shape at the row counts a short prompt takes.
    let (k, n, scheme, ggml) = (2048usize, 6144usize, QuantScheme::GgufQ4K, GgmlType::Q4K);
    let w = packed(ggml, k, n, 1);
    eprintln!("row sweep, Q4_K k={k} n={n}:");
    for &m in ROW_SWEEP {
        let x: Vec<f32> = (0..m).flat_map(|_| activation(k)).collect();
        let mut fused = vec![0f32; m * n];
        let mut tiled = vec![0f32; m * n];
        let times = common::paired(ROUNDS, 2, |arm| {
            let _route =
                rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", if arm == 0 { "0" } else { "1" });
            let t = std::time::Instant::now();
            if arm == 0 {
                gguf_matmul_bt(&x, &w, &mut fused, m, k, n, scheme);
            } else {
                gguf_matmul_bt(&x, &w, &mut tiled, m, k, n, scheme);
            }
            t.elapsed().as_secs_f64() * 1e3
        });
        let (a, b) = (times[0], times[1]);
        let tiled_why = {
            let _tiled = rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", "1");
            rlx_cpu::routing::tiled_refusal(m, k, n, scheme)
        };
        eprintln!(
            "    m={m:<4} fused {a:8.3} ms   {} {b:8.3} ms   {:.2}x  err {:.1e}{}",
            match tiled_why {
                None => "tiled",
                Some(_) => "not tiled",
            },
            a / b,
            max_rel_err(&tiled, &fused),
            match tiled_why {
                None => String::new(),
                Some(why) => format!("  (tiled refused here: {why})"),
            }
        );
    }
}
