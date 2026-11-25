// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tiled dequantise-then-BLAS: a packed GGUF weight against several activation
//! rows, which is what a prompt prefill and an audio encoder run.
//!
//! The fused `m > 1` arm ([`crate::gguf_matmul::gguf_matmul_bt`], which for the
//! K-quants lands in `gguf_gemm_specialized`'s `gemm_q4k` / `gemm_q6k`)
//! dequantises one 256-element super-block into a stack array and dots it
//! against every row with `dot256`, a plain f32 loop — four flops a cycle —
//! while the machine's matrix units sit idle. This module instead dequantises a
//! tile of weight *rows* into an f32 slab and hands that slab to the vendor
//! BLAS, which is the route that reaches those units without materialising the
//! whole weight.
//!
//! A tile is bounded in bytes rather than rows, so the largest weight in the
//! model (the 2048 x 151936 output head) is never materialised whole. Nothing here
//! outlives a call: the scratch is one [`TILE_BYTES`] tile per stripe plus its own
//! output rows, and [`MAX_SCRATCH_BYTES`] is the cap on the whole of it.
//!
//! The stand's memory columns do not follow that scratch. With this route off the
//! same binary peaks *higher* on the largest row (1.40 GiB against 1.36 / 1.38 GiB
//! on), so the on-route takes' 1.36 / 1.38 / 1.55 / 1.56 GiB is the column's own
//! wander rather than this route's. The 30 s row's *resting* figure is higher with
//! the route on (527-537 MiB) than off (264 MiB on the first call) because the
//! seven-times-longer fused call leaves the allocator idle time to give pages back:
//! twelve calls on this route plateau at 523.8, 532.6, 533.8, 534.0, 534.1, 536.9,
//! 538.1, 538.1, 538.1, 538.2, 538.2, 538.2 MiB (flat from the sixth) where three
//! off it settle by the second (264.9, 489.8, 490.8), the fused route's own second
//! call reaching the level this route's first call is already at — and neither a
//! `malloc_zone_pressure_relief` over every zone nor `release_process_memory`
//! (109 MiB off, the same on both) finds anything this route keeps. The cap is read
//! on the peak, which moves 13.6 MiB.
//!
//! Where the time goes, and why this is the route. `MAHBOT_PHASE_SPLIT=1` on the
//! processor stand, taken before this route existed (`STAND-CPU.md` is rewritten
//! whole by every take, so the baseline's figures live here and nowhere else),
//! puts the multi-row multiply at 83-85% of every recognition row: the prompt
//! prefill is 17.6 s of the 10 s row's 25.6 s warm, 60.4 of 90.2 at 30 s and
//! 572.9 of 874.4 at 318 s, and the audio encoder's own 24 blocks are most of the
//! 4.1 / 16.2 / 152.9 s "encoder front" beside it. The tagger is 0.3 / 1.8 /
//! 20.0 s and the single-row decode 3.5 / 11.5 / 127.7 s.
//!
//! The three arrangements on those shapes, median of five alternating rounds
//! (`tests/gguf_multi_row_route_perf.rs`, five workers, 2 MiB tiles, ms per
//! multiply, every arm paying its own dequantisation):
//!
//! | shape | fused | slab in tiles, one call | tiled, striped across the pool |
//! |---|---|---|---|
//! | enc qkv   m64 k1024 n1024 | 6.26 | **0.259** | 0.276 |
//! | enc up    m64 k1024 n4096 | 21.85 | 1.165 | **0.857** |
//! | enc down  m64 k4096 n1024 | 27.41 | 1.328 | **0.871** |
//! | body qkv  m79 k2048 n2048 | 27.56 | 2.104 | **1.287** |
//! | body up   m79 k2048 n6144 | 78.93 | 5.500 | **3.330** |
//! | body down m79 k6144 n2048 | 77.93 | 4.506 | **3.200** |
//! | head      m79 k2048 n8192 | 106.19 | 5.888 | **3.270** |
//!
//! So the striped split wins everywhere but the smallest tile, where the two are
//! within noise of each other (the recorded take has a single vendor call over the
//! whole slab 6% ahead there, a later one 5% behind) — and that arrangement needs
//! the whole tensor in memory, which is the arrangement this route exists to
//! avoid. Rows are what the win scales with: on `k2048 n6144` the fused route costs
//! 3.39 / 5.07 / 8.35 / 15.44 / 30.74 / 76.06 ms at 2 / 4 / 8 / 16 / 32 / 79 rows
//! against 2.31 / 2.37 / 2.70 / 2.16 / 2.49 / 3.21, so the route takes every
//! `m > 1` a host that can stripe it asks for.
//!
//! The arithmetic is the vendor BLAS's over the whole `k`, so what moves against
//! the fused route is summation order (~1e-4 relative on the harness's
//! heavy-tailed data), not precision: the activation is never quantised here.
//! The stand's answer digest moves with it, and with the six-bit route beside it.
//! On the 10 s recognition row, switching this route off moves it (`965df68a` to
//! `c206d5b3`) while switching the six-bit route off leaves it, so that row's move
//! is this route's summation order. The synthesis rows move with either: both on
//! `148462cd` / `dd3592a9` against the baseline's `5450ad6d` / `ef2a770e`, this one
//! off `9a52a507` / `f2e3ca23`, the six-bit one off `ccdb09d0` / `3d93543a`,
//! neither restoring the baseline — on a 23-92 frame sample the digest
//! fingerprints the whole summation order, so no one route owns the move. The 30 s
//! and 318 s rows move with the same two routes — they are the same graphs at more
//! chunks.
//!
//! It is a *prompt* route, not a decode one. Four tokens through one four-row
//! pass cost 4.8x four one-row int8 passes (2.0x on a later, busier take) — the
//! same slab that pays at 79 rows loses at four — so batching a decode through here
//! would lose, and a batched decode would have to bring a multi-row *integer*
//! kernel with it instead.
//!
//! One reach to know about: [`crate::gguf_matmul::gguf_matmul_bt`] is also where
//! `rlx-gpu-host` stages a `DequantMatMul` on the host for CUDA/ROCm, where
//! `rlx-mlx` lowers one, and where `rlx-wgpu`'s own host fallback lands through
//! `gguf_host`, so this route runs there too and moves those backends' summation
//! order as well.

use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;

/// The f32 image one worker materialises at a time.
///
/// Two mebibytes is 512 rows of a 1024-wide weight and 85 rows of a 6144-wide
/// one: wide enough that the vendor call inside a tile amortises its own
/// dispatch, small enough that the whole of it stays a fraction of
/// [`MAX_SCRATCH_BYTES`]. Across the product's shapes the stripes hold at most
/// 10.6 MiB of tile plus their output tiles (five stripes at k=1024, n=4096); the
/// narrowest, the encoder's k=1024 n=1024, has two tiles and holds 4.3 MiB.
const TILE_BYTES: usize = 2 << 20;

/// The most scratch a striped call may hold at once, across every worker: the 96 MiB
/// this route is allowed to add, and an absolute cap rather than a floor of one stripe
/// — [`admits`] refuses a shape whose single stripe would exceed it. [`stripes_for`]
/// clamps the worker count to what fits. What the stand's memory columns do with that
/// is the module doc's account.
const MAX_SCRATCH_BYTES: usize = 96 << 20;

/// Bytes one stripe holds: its weight tile plus its `m x rows` output tile.
fn per_stripe_bytes(m: usize, k: usize, n: usize) -> usize {
    tile_rows(k, n) * (k + m) * 4
}

/// Rows of weight one tile holds at width `k` for a weight of `n` rows: bounded
/// in bytes, never fewer than one — a `k` wider than a whole tile still gets a row —
/// and never more than the `n` rows there are. Asked only about a shape [`admits`]
/// takes, so nothing here guards a zero.
fn tile_rows(k: usize, n: usize) -> usize {
    (TILE_BYTES / (k * 4)).max(1).min(n)
}

/// How many workers a striped call runs: the pool width, clamped to the tiles there
/// are and to the stripes [`MAX_SCRATCH_BYTES`] pays for. Zero means not one stripe
/// fits, which [`admits`] refuses; [`matmul_bt_striped`] then runs the minimum of one.
fn stripes_for(m: usize, k: usize, n: usize) -> usize {
    crate::pool::num_threads()
        .min(n.div_ceil(tile_rows(k, n)))
        .min(MAX_SCRATCH_BYTES / per_stripe_bytes(m, k, n))
}

/// Whether the tiled route takes `(m, k, n)` in `scheme`: [`refusal`]'s conditions, and this
/// is their `is_none`.
///
/// It answers for the route, not for the caller — a serial call never asks, and runs the
/// single-threaded paths in [`crate::gguf_matmul::gguf_matmul_bt_serial`] instead.
pub(crate) fn admits(m: usize, k: usize, n: usize, scheme: QuantScheme) -> bool {
    refusal(m, k, n, scheme).is_none()
}

/// Why the tiled route does not take `(m, k, n)` in `scheme`, or `None` when it does:
/// the one list of conditions behind the admission, so a caller that reports a skip can
/// name the cause that actually held instead of the first one that comes to mind. The
/// shape's own conditions come before the host's, so a refusal is the shape's wherever
/// the host cannot tell the difference.
pub(crate) fn refusal(m: usize, k: usize, n: usize, scheme: QuantScheme) -> Option<&'static str> {
    if m <= 1 || k == 0 || n == 0 {
        Some("the shape is not a multi-row multiply")
    } else if scheme.gguf_block_bytes() == 0 {
        Some("the scheme has no block layout")
    } else if !k.is_multiple_of(scheme.gguf_block_size() as usize) {
        Some("k is not a whole number of blocks")
    } else if per_stripe_bytes(m, k, n) > MAX_SCRATCH_BYTES {
        Some("one stripe of this shape is over the scratch cap")
    } else if crate::pool::num_threads() <= 1 {
        Some("the pool is one worker, which cannot stripe")
    } else if !cfg!(rlx_cpu_blas) {
        Some("this build has no vendor BLAS")
    } else if crate::config::portable() {
        Some("this is a portable run")
    } else if !rlx_ir::env::flag_or("RLX_GGUF_TILED_BLAS", true) {
        Some("RLX_GGUF_TILED_BLAS=0")
    } else {
        None
    }
}

/// `C[m,n] = A[m,k] @ B^T` with `B` packed `[n,k]`, dequantised a tile at a time
/// and multiplied by the vendor BLAS. The shape is one [`admits`] takes.
pub(crate) fn matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    matmul_bt_striped(x, w_bytes, out, m, k, n, scheme, stripes_for(m, k, n));
}

/// [`matmul_bt`] at an explicit stripe count: `stripes` workers run at once, each
/// taking every `stripes`-th tile, and one stripe is a single sequential pass.
///
/// The arithmetic is BLAS's over the whole `k`, so the answer does not depend on
/// `stripes` — a tile splits the output columns, never the reduction — and no rows, no
/// columns or no reduction is the zero product. A `k` that is not a whole number of
/// blocks is not the zero product and not this function's to refuse: `blocks_per_row`
/// truncates and the bytes it reads are then the wrong ones, so [`admits`] is what
/// keeps that shape away.
fn matmul_bt_striped(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    stripes: usize,
) {
    use rayon::prelude::*;

    assert_eq!(out.len(), m * n, "the destination is the whole product");
    if m == 0 || k == 0 || n == 0 {
        out.fill(0.0);
        return;
    }
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let rows = tile_rows(k, n);
    let tiles = n.div_ceil(rows);
    let stripes = stripes.clamp(1, tiles);

    // One allocation each for the weight tiles and the output tiles, so a
    // striped call does not pay a heap trip per tile.
    let mut weights = vec![0f32; stripes * rows * k];
    let mut cols = vec![0f32; stripes * m * rows];
    // One arm for the whole call: the switch is process-global, so reading it once
    // here rather than per row in a worker keeps every stripe on the same arm.
    let direct = crate::gguf_matmul::direct_decode();
    // Each stripe writes a disjoint band of `out`'s columns, so the shared
    // destination is sound even though the closure cannot see that. The bands are
    // `tile * rows` for `tile ≡ stripe (mod stripes)`, which are pairwise disjoint
    // and each exactly `jn` wide, and `out` is `m * n` long as asserted above — so
    // every `r * n + j0` the scatter below forms is a whole row of a band.
    let dst = out.as_mut_ptr() as usize;
    weights
        .par_chunks_mut(rows * k)
        .zip(cols.par_chunks_mut(m * rows))
        .enumerate()
        .for_each(|(stripe, (wbuf, obuf))| {
            let mut blk = [0f32; QK_K];
            let mut widened = 0usize;
            let mut tile = stripe;
            while tile < tiles {
                let j0 = tile * rows;
                let jn = (j0 + rows).min(n) - j0;
                for j in 0..jn {
                    let start = (j0 + j) * row_bytes;
                    let Some(row) = w_bytes.get(start..start + row_bytes) else {
                        // Short weight: leave the tile zero, as the fused route's
                        // per-row `get` does.
                        wbuf[j * k..(j + 1) * k].fill(0.0);
                        continue;
                    };
                    crate::gguf_matmul::dequant_row(
                        scheme,
                        row,
                        &mut wbuf[j * k..(j + 1) * k],
                        block_bytes,
                        block_elems,
                        &mut blk,
                        direct,
                    );
                    widened += 1;
                }
                let ot = &mut obuf[..m * jn];
                crate::blas::sgemm_bt(x, &wbuf[..jn * k], ot, m, k, jn, 1.0);
                // `out` is `[m, n]` row-major and a tile is a band of columns, so
                // the tile lands strided. BLAS cannot write it in place: its `ldc`
                // is `n`, which no BLAS entry point here takes.
                for r in 0..m {
                    // SAFETY: this stripe owns columns `j0..j0 + jn` of every row,
                    // and no other stripe touches them.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            ot[r * jn..].as_ptr(),
                            (dst as *mut f32).add(r * n + j0),
                            jn,
                        );
                    }
                }
                tile += stripes;
            }
            crate::gguf_matmul::note_rows(
                crate::gguf_matmul::direct_arm(direct, block_elems),
                widened,
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seven tiles at the narrowest width the models ship (`k = 1024` puts 512 rows
    /// in a tile, so `n = 3200` is six whole tiles and a 128-row last one), which is
    /// what a one-tile shape cannot reach: a three-stripe call takes two to three
    /// tiles per worker, so the band arithmetic, the strided scatter behind the
    /// unsafe write and the partial last tile the 151936-row output head really hits
    /// are all exercised. Every stripe count must land the same columns — the tiles
    /// split the output, never the reduction.
    #[test]
    fn every_stripe_count_lands_the_same_columns() {
        let (m, k, n) = (3usize, 1024usize, 3200usize);
        assert!(
            n % tile_rows(k, n) != 0,
            "the last tile must be partial for this test to be about it"
        );
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).sin()).collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q8_0).expect("pack Q8_0");
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.007).cos()).collect();
        let dq = rlx_gguf::dequant_typed(rlx_gguf::GgmlType::Q8_0, &packed, k * n, "tiled")
            .expect("dequant the weight");
        let mut want = vec![0f32; m * n];
        let mut mag = vec![0f32; m * n];
        for r in 0..m {
            for j in 0..n {
                let (mut acc, mut abs) = (0f64, 0f64);
                for p in 0..k {
                    let term = f64::from(dq[j * k + p]) * f64::from(x[r * k + p]);
                    acc += term;
                    abs += term.abs();
                }
                want[r * n + j] = acc as f32;
                mag[r * n + j] = abs as f32;
            }
        }
        // NaN, so a column no stripe visited fails below rather than reading as a
        // zero the oracle might agree with.
        let run = |stripes: usize| {
            let mut got = vec![f32::NAN; m * n];
            matmul_bt_striped(
                &x,
                &packed,
                &mut got,
                m,
                k,
                n,
                QuantScheme::GgufQ8_0,
                stripes,
            );
            got
        };
        let (one, three) = (run(1), run(3));
        for i in 0..m * n {
            // The bands split the output columns, never the reduction, so the two
            // counts must agree to a few ulp...
            assert!(
                (one[i] - three[i]).abs() <= 1e-6 * one[i].abs().max(1.0),
                "i={i}: one stripe {} vs three {}",
                one[i],
                three[i]
            );
            // ...and both must land the columns the f64 dot names. The bound is the
            // f32 accumulation's own size — `k` roundings of the terms it sums — so a
            // band written to the wrong columns, off by a whole output, cannot hide
            // inside it.
            let err = (one[i] - want[i]).abs() / mag[i].max(1.0);
            assert!(err < 1e-5, "i={i}: {} vs {} ({err})", one[i], want[i]);
        }
    }

    /// The zero shapes are the zero product rather than a panic or a divide: `admits`
    /// keeps them out of the route, and this holds the arithmetic to the answer it owes
    /// them anyway.
    #[test]
    fn a_zero_shape_is_the_zero_product() {
        let packed = rlx_gguf::quantize(&[0.5f32; 32], rlx_gguf::GgmlType::Q8_0).expect("pack");
        for (m, k, n) in [(0usize, 1024usize, 4usize), (2, 0, 4), (2, 1024, 0)] {
            let x = vec![1f32; m * k];
            let mut out = vec![f32::NAN; m * n];
            matmul_bt_striped(&x, &packed, &mut out, m, k, n, QuantScheme::GgufQ8_0, 1);
            assert!(
                out.iter().all(|value| *value == 0.0),
                "m={m} k={k} n={n}: {out:?}"
            );
        }
    }

    /// The cap binds on the stripe count: a tall `m` gets fewer workers rather than
    /// more memory, no shape gets more stripes than it has tiles, and what a stripe
    /// count does buy stays inside [`MAX_SCRATCH_BYTES`] — absolutely, with no shape
    /// allowed the one stripe that would not fit.
    #[test]
    fn the_stripe_count_stays_inside_the_scratch_cap() {
        for (m, k, n) in [
            (79usize, 2048usize, 6144usize),
            (64, 1024, 4096),
            (768, 2048, 6144),
            (20_000, 1024, 8192),
        ] {
            let per_stripe = per_stripe_bytes(m, k, n);
            let stripes = stripes_for(m, k, n);
            assert!(stripes >= 1, "m={m} k={k} n={n}: no stripe at all");
            assert!(stripes <= n.div_ceil(tile_rows(k, n)), "m={m} k={k} n={n}");
            assert!(
                stripes * per_stripe <= MAX_SCRATCH_BYTES,
                "m={m} k={k} n={n}: {stripes} stripes of {per_stripe} B"
            );
        }
        // The last shape is the one where the cap, not the tile count or the pool,
        // decides: sixteen tiles' worth of work against a per-stripe figure the cap
        // pays for two of, so the stripe count above is the cap's.
        let (m, k, n) = (20_000usize, 1024usize, 8192usize);
        assert!(
            MAX_SCRATCH_BYTES / per_stripe_bytes(m, k, n) < n.div_ceil(tile_rows(k, n)),
            "the shape must be tile-rich"
        );
        // And a shape whose *single* stripe is over the cap is refused by the cap
        // itself, on any host: that is what makes the cap absolute rather than a floor
        // of one.
        let (m, k, n) = (100_000usize, 1024usize, 8192usize);
        assert!(
            per_stripe_bytes(m, k, n) > MAX_SCRATCH_BYTES,
            "the shape must be over the cap"
        );
        assert_eq!(stripes_for(m, k, n), 0, "the cap must leave no stripe");
        assert_eq!(
            refusal(m, k, n, QuantScheme::GgufQ8_0),
            Some("one stripe of this shape is over the scratch cap"),
        );
    }
}
