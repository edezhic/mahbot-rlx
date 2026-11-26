// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fused GGUF K-quant dequant + matmul without materializing full F32
//! weights (Tier C.11).
//!
//! Computes `C[m,n] = A[m,k] @ B^T` where `B` is `[n,k]` row-major in
//! packed GGUF layout. For Q4_K decode GEMVs (`m==1`, `k` multiple of 256)
//! activations are quantized once to Q8_K and dotted with int8 NEON kernels
//! (llama.cpp-style); other schemes still fold one 256-element super-block
//! at a time into stack storage.

use std::sync::atomic::{AtomicU64, Ordering};

use rlx_gguf::QK_K;
use rlx_gguf::q2_dequant::{Q2_0_BLOCK_BYTES, Q8_0_G128_BYTES, QK2_0};
use rlx_ir::quant::QuantScheme;

pub(crate) fn dequant_block(scheme: QuantScheme, block: &[u8], out: &mut [f32; QK_K]) {
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k_block(block, out),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k_block(block, out),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k_block(block, out),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k_block(block, out),
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k_block(block, out),
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k_block(block, out),
        QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0_block(block, &mut out[..rlx_gguf::QK4_0]),
        QuantScheme::GgufQ4_1 => {
            rlx_gguf::dequant_q4_1_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ5_0 => {
            rlx_gguf::dequant_q5_0_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ5_1 => {
            rlx_gguf::dequant_q5_1_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0_block(block, &mut out[..rlx_gguf::QK8_0]),
        // Block-level fast paths for the new schemes that share QK_K.
        QuantScheme::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0_block(block, out),
        QuantScheme::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0_block(block, out),
        // 128-element 1-bit block (PrismML Bonsai-27B); caller slices `out`.
        QuantScheme::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0_block(
            block,
            (&mut out[..rlx_gguf::q1_dequant::QK1_0])
                .try_into()
                .unwrap(),
        ),
        QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0_block(
            block,
            (&mut out[..rlx_gguf::q2_dequant::QK2_0])
                .try_into()
                .unwrap(),
        ),
        // Fermion five-value ternary (Neutrino): 256-element blocks == QK_K,
        // so `out` maps 1:1. FV5 = transformer linears, FV5B = int8 embed/lm_head.
        QuantScheme::GgufFV5 => rlx_gguf::fv5_dequant::dequant_fv5_block(block, out),
        QuantScheme::GgufFV5B => rlx_gguf::fv5_dequant::dequant_fv5b_block(block, out),
        // 32-element exact-ternary block (Doses AI Pestle embed / lm_head).
        QuantScheme::GgufG8_0 => rlx_gguf::g8_dequant::dequant_g8_0_block(
            block,
            (&mut out[..rlx_gguf::g8_dequant::QKG8_0])
                .try_into()
                .unwrap(),
        ),
        // 128-element base-3 ternary block (PrismML Ternary Bonsai 2).
        QuantScheme::GgufPtq1_0 => rlx_gguf::ptq1_dequant::dequant_ptq1_0_block(
            block,
            (&mut out[..rlx_gguf::ptq1_dequant::QK_PTQ1_0])
                .try_into()
                .unwrap(),
        ),
        // 32-element blocks: caller slices `out` to the correct length.
        QuantScheme::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4_block(
            block,
            (&mut out[..rlx_gguf::mx_dequant::QK_MXFP4])
                .try_into()
                .unwrap(),
        ),
        QuantScheme::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4_block(
            block,
            (&mut out[..rlx_gguf::mx_dequant::QK_NVFP4])
                .try_into()
                .unwrap(),
        ),
        // IQ-family: no dedicated block-level helper, but the
        // whole-tensor dequant works fine on a single QK_K block —
        // pass the 256-elem slice straight through. ~Same cost as
        // dedicated block functions, just slightly less inlinable.
        QuantScheme::GgufIQ4XS => {
            let v = rlx_gguf::iq_dequant::dequant_iq4_xs(block, QK_K).expect("IQ4_XS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2XXS => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_xxs(block, QK_K).expect("IQ2_XXS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2XS => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_xs(block, QK_K).expect("IQ2_XS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2S => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_s(block, QK_K).expect("IQ2_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ3XXS => {
            let v = rlx_gguf::iq_dequant::dequant_iq3_xxs(block, QK_K).expect("IQ3_XXS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ3S => {
            let v = rlx_gguf::iq_dequant::dequant_iq3_s(block, QK_K).expect("IQ3_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ1S => {
            let v = rlx_gguf::iq_dequant::dequant_iq1_s(block, QK_K).expect("IQ1_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ1M => {
            let v = rlx_gguf::iq_dequant::dequant_iq1_m(block, QK_K).expect("IQ1_M block");
            out.copy_from_slice(&v);
        }
        // 32-element block schemes that go through dequant_block need
        // the caller to slice `out`; IQ4_NL is the only one not handled
        // above. Mirrors the Q4_0/Q8_0 idiom.
        QuantScheme::GgufIQ4NL => {
            rlx_gguf::iq_dequant::dequant_iq4_nl(block, rlx_gguf::iq_dequant::QK4_NL)
                .map(|v| out[..rlx_gguf::iq_dequant::QK4_NL].copy_from_slice(&v))
                .expect("IQ4_NL block")
        }
        // One bf16 per "block": the upper 16 bits of the f32.
        QuantScheme::GgufBF16 => {
            out[0] = f32::from_bits(u32::from(u16::from_le_bytes([block[0], block[1]])) << 16);
        }
        // 128-element FP8 block: f32 scale + 128 codes; caller slices `out`.
        QuantScheme::Fp8E4m3B128 | QuantScheme::Fp8E5m2B128 => {
            let fmt = rlx_ir::lowp_codec::fp8_b128::format(scheme).unwrap();
            rlx_ir::lowp_codec::fp8_b128::dequant_block(fmt, block, out)
        }
        other => panic!(
            "gguf_matmul: scheme {other:?} has no block-level dequant — use load-time dequant_cache"
        ),
    }
}

/// Fused dequant + `sgemm_bt` — `out` is zeroed then accumulated.
///
/// Block-fused reference kernel (no full-weight materialization); default dispatch uses
/// [`gguf_matmul_bt_dispatch`], and [`gguf_matmul_use_legacy`] is the switch that lands
/// here. A multi-row call leaves for `crate::gguf_tiled` wherever that route takes it:
/// `crate::gguf_tiled::refusal` is the only list of when it does not.
pub fn gguf_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_ex(
        x, w_bytes, out, m, k, n, scheme, /*allow_parallel=*/ true,
    );
}

/// Like [`gguf_matmul_bt`] but never spawns Rayon inside the `m==1` kernel.
///
/// Use this from an outer Rayon parallel region (e.g. MoE experts) so block-level
/// parallelism does not nest and oversubscribe the pool.
pub fn gguf_matmul_bt_serial(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_ex(
        x, w_bytes, out, m, k, n, scheme, /*allow_parallel=*/ false,
    );
}

fn gguf_matmul_bt_ex(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    allow_parallel: bool,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    out.fill(0.0);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let total_elems = k * n;
    debug_assert!(
        total_elems.is_multiple_of(block_elems),
        "k*n={total_elems} not aligned to GGUF block {block_elems}"
    );
    // Some backends (notably rlx-mlx, see lower.rs:1554) recover (m, k, n)
    // from inferred MLX shapes that can round one dim up by a single block
    // when an intermediate node carries padding; the caller's k*n then
    // implies one more Q4K block than the GGUF actually stored. Clamp to
    // the bytes we actually have rather than panic in release mode.
    let num_blocks = packed_blocks(k, n, w_bytes.len(), scheme);
    debug_assert_eq!(
        w_bytes.len() / block_bytes,
        total_elems / block_elems,
        "Q4K matmul: caller (k={k}, n={n}) implies {} blocks but w_bytes holds {}",
        total_elems / block_elems,
        w_bytes.len() / block_bytes,
    );

    let mut block_f32 = [0f32; QK_K];

    if m == 1 {
        let x_row = x;
        if allow_parallel && m1_parallel_takes(num_blocks) {
            gguf_matmul_bt_m1_parallel(
                x_row,
                w_bytes,
                out,
                k,
                n,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
            );
        } else {
            gguf_matmul_bt_m1_sequential(
                x_row,
                w_bytes,
                out,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
                k,
            );
        }
        return;
    }

    if allow_parallel
        && k.is_multiple_of(block_elems)
        && block_elems <= QK_K
        && crate::pool::num_threads() > 1
    {
        // The crowded multiply: a tile of weight rows dequantised into f32 and
        // handed to the vendor BLAS, which is the only route here that reaches the
        // matrix units. `gguf_tiled`'s doc comment carries the reading; the fused
        // GEMM below keeps the job for every shape and host
        // `crate::gguf_tiled::refusal` names.
        if crate::gguf_tiled::admits(m, k, n, scheme) {
            crate::gguf_tiled::matmul_bt(x, w_bytes, out, m, k, n, scheme);
            return;
        }
        gguf_matmul_bt_rows_generic_gemm(
            x,
            w_bytes,
            out,
            m,
            k,
            n,
            scheme,
            block_bytes,
            block_elems,
        );
        return;
    }
    // Rows of whole blocks (every GGUF projection): widen each weight row once,
    // then one contiguous dot per activation row. The element loop below costs
    // a division and a strided store per weight per row — 5-10x slower where
    // nothing else runs (wasm32, single-threaded).
    if k.is_multiple_of(block_elems) && num_blocks == k / block_elems * n {
        gguf_matmul_bt_rows_dot(x, w_bytes, out, m, k, n, scheme, block_bytes, block_elems);
        return;
    }
    for bi in 0..num_blocks {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            let j = idx / k;
            let p = idx % k;
            let w_val = block_f32[t];
            for mi in 0..m {
                out[mi * n + j] += x[mi * k + p] * w_val;
            }
        }
    }
}

/// The arm [`dequant_row`] takes: the direct arrangement only when the switch is on *and*
/// the block is the whole super-block. The branch and the row accounting share this one
/// predicate, so a scheme whose block is not the super-block cannot be counted on the arm
/// it did not take.
#[inline]
pub(crate) fn direct_arm(direct: bool, block_elems: usize) -> bool {
    direct && block_elems == QK_K
}

/// Widen one packed weight row into `dst`, one block at a time, for both routes that widen a
/// whole row. `direct` picks the arrangement through [`direct_arm`]. Direct: a scheme whose
/// block is the whole super-block decodes straight into `dst` (the staging copy below wrote
/// every weight twice) and the four-bit decode takes [`rlx_gguf::q4k_fast::block`] instead.
/// Staged: the reference decode through `blk`, which is the caller's scratch, so a row loop
/// allocates nothing per row.
pub(crate) fn dequant_row(
    scheme: QuantScheme,
    row: &[u8],
    dst: &mut [f32],
    block_bytes: usize,
    block_elems: usize,
    blk: &mut [f32; QK_K],
    direct: bool,
) {
    if direct_arm(direct, block_elems) {
        let four_bit = scheme == QuantScheme::GgufQ4K;
        for (b, out) in dst.chunks_exact_mut(QK_K).enumerate() {
            let out: &mut [f32; QK_K] = out.try_into().expect("one super-block");
            let block = &row[b * block_bytes..][..block_bytes];
            if four_bit {
                rlx_gguf::q4k_fast::block(block.try_into().expect("one Q4_K block"), out);
            } else {
                dequant_block(scheme, block, out);
            }
        }
    } else {
        for (b, out) in dst.chunks_exact_mut(block_elems).enumerate() {
            dequant_block(scheme, &row[b * block_bytes..][..block_bytes], blk);
            out.copy_from_slice(&blk[..block_elems]);
        }
    }
}

/// Whether a packed weight row takes [`dequant_row`]'s direct arrangement:
/// `RLX_GGUF_ROW_DECODE` (default on), and never under the portable arithmetic,
/// where the staged reference decode is the arm being validated.
#[inline]
pub(crate) fn direct_decode() -> bool {
    !crate::config::portable() && rlx_ir::env::flag_or("RLX_GGUF_ROW_DECODE", true)
}

static DIRECT_ROWS: AtomicU64 = AtomicU64::new(0);
static STAGED_ROWS: AtomicU64 = AtomicU64::new(0);

/// The weight rows this process has widened each way — `(direct, staged)` — for a harness
/// that has to prove which arm of `RLX_GGUF_ROW_DECODE` a call took: the two arms write the
/// same values by construction, so a take cannot tell them apart from its answer. The count
/// is of rows actually decoded, on the arm the row decode actually took. One relaxed add
/// per route call (per stripe in the tile route), so a weight-row loop pays it nothing; the
/// same instrument `rlx-wgpu`'s `route_counts` gives its packed multiply.
pub fn row_decodes() -> (u64, u64) {
    (
        DIRECT_ROWS.load(Ordering::Relaxed),
        STAGED_ROWS.load(Ordering::Relaxed),
    )
}

/// Record `rows` weight rows decoded by the arm `direct` names.
pub(crate) fn note_rows(direct: bool, rows: usize) {
    let counter = if direct { &DIRECT_ROWS } else { &STAGED_ROWS };
    counter.fetch_add(rows as u64, Ordering::Relaxed);
}

/// `out[mi, j] = x[mi, :] · W[j, :]` with `W` `[n, k]` in GGUF blocks: each
/// weight row widened once into a buffer, then dotted with every activation
/// row. Single-threaded; the dot is written so it vectorizes (wasm simd128,
/// NEON, SSE) without fast-math.
#[allow(clippy::too_many_arguments)]
fn gguf_matmul_bt_rows_dot(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    block_bytes: usize,
    block_elems: usize,
) {
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let mut wrow = vec![0f32; k];
    let mut blk = [0f32; QK_K];
    let direct = direct_decode();
    for j in 0..n {
        let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
        dequant_row(
            scheme,
            row,
            &mut wrow,
            block_bytes,
            block_elems,
            &mut blk,
            direct,
        );
        for mi in 0..m {
            out[mi * n + j] = dot_f32(&x[mi * k..(mi + 1) * k], &wrow);
        }
    }
    note_rows(direct_arm(direct, block_elems), n);
}

/// A dot product in eight independent lanes, so it vectorizes without
/// reassociating a single sum.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `true` when `RLX_GGUF_MATMUL_LEGACY=1` — skip the cached-f32-BLAS arm.
///
/// It does not decide the tiled route: `crate::gguf_tiled::refusal` names what does.
#[inline]
pub fn gguf_matmul_use_legacy() -> bool {
    matches!(
        rlx_ir::env::var("RLX_GGUF_MATMUL_LEGACY").as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

/// Minimum `k*n` for cached dequant + BLAS (tiny tiles stay on fused path).
const CACHED_BLAS_MIN_WEIGHT_ELEMS: usize = 32 * 32;

/// Crossover `n` above which the parallel int8 Q4_K decode GEMV beats
/// cached-f32-BLAS (Accelerate/AMX). Measured on Apple: BLAS ahead at n≈1k,
/// tie at n≈3k, fused ~3× ahead at n≈128k → `n≥4096` captures the wide matmuls
/// (LM head, large FFN) for a ~15–25% CPU decode speedup.
///
/// **Opt-in** (default disabled): the fused path quantizes the *activation* to
/// Q8_K (llama.cpp-style), which is not bit-identical to the f32 cached-BLAS
/// path and flips occasional near-tie greedy tokens — rlx keeps decode on f32
/// for fidelity (and decode↔prefill parity) by default. Set
/// `RLX_Q4K_FUSED_MIN_N=4096` (or another crossover) to trade that for speed.
/// Route Q4_K decode GEMVs through the exact f32-activation kernel
/// ([`q4k_exact_gemv_rows`]) instead of the Q8_K one (`RLX_Q4K_EXACT_GEMV=1`).
/// Costs ALU: ~0.6x the cached-f32-BLAS route, ~2–2.8x the cache-off one, but
/// keeps decode numerics independent of whether the f32 dequant cache happened
/// to fit.
///
/// Its callers are the rayon arm of `gguf_matmul_bt_m1_parallel` and the activation
/// resolution in [`crate::gguf_activation`].
pub(crate) fn q4k_exact_gemv() -> bool {
    rlx_ir::env::flag("RLX_Q4K_EXACT_GEMV")
}

fn q4k_fused_decode_min_n() -> usize {
    rlx_ir::env::var("RLX_Q4K_FUSED_MIN_N")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

/// The fewest blocks of weight [`gguf_matmul_bt_ex`] takes its parallel single-row arm
/// with.
const M1_PARALLEL_MIN_BLOCKS: usize = 32;

/// The blocks of weight a call of this shape actually holds — clamped to the bytes stored,
/// as [`gguf_matmul_bt_ex`] needs, so a caller's `k`/`n` that imply more blocks than the
/// GGUF carries is counted at what is there.
#[inline]
pub(crate) fn packed_blocks(k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> usize {
    (k * n / scheme.gguf_block_size() as usize).min(w_len / scheme.gguf_block_bytes() as usize)
}

/// Whether [`gguf_matmul_bt_ex`]'s parallel single-row arm takes a call with this many
/// blocks of weight: a pool of more than one worker and at least
/// [`M1_PARALLEL_MIN_BLOCKS`]. The single-row integer routes are reached through it rather
/// than through a copy of its conditions, and a caller that allows no nesting
/// ([`gguf_matmul_bt_serial`]) never reaches the arm at all.
#[inline]
pub(crate) fn m1_parallel_takes(blocks: usize) -> bool {
    crate::pool::num_threads() > 1 && blocks >= M1_PARALLEL_MIN_BLOCKS
}

/// Whether an eight-bit single-row call of this shape is taken whatever the dequant cache
/// holds, so its answer cannot depend on cache state and a harness's oracle is that arm's
/// activation rather than the cached f32 route's.
///
/// True only for the calls [`crate::gguf_q8_0::takes`] names:
/// [`gguf_matmul_bt_dispatch_at`] sends those to the fused kernel before the cache is
/// consulted. Q2_0's arm there is the same arrangement but is inline and has a shape gate
/// of its own, so what it does with a row it refuses is still the cache's to decide.
#[inline]
pub(crate) fn q8_0_arm_ignores_cache(
    k: usize,
    n: usize,
    w_len: usize,
    scheme: QuantScheme,
) -> bool {
    scheme == QuantScheme::GgufQ8_0 && crate::gguf_q8_0::takes(k, n, w_len)
}

#[inline]
fn prefer_cached_blas(k: usize, n: usize, m: usize, scheme: QuantScheme) -> bool {
    if gguf_matmul_use_legacy() {
        return false;
    }
    // Reproducible runs keep one path whatever the memory budget: thrashing
    // is the host's, and the fused kernel sums in another order.
    if rlx_ir::reproducible::reproducible() {
        return true;
    }
    // A cache that is switched off cannot serve any of this: the "cached" route dequantises
    // the weight into an f32 slab and hands it to BLAS, and with nothing caching that slab the
    // slab is thrown away and paid for again on the next call. The fused kernel does the same
    // dequant work without it. That includes Q4_K's single-row GEMV, whose fused arm quantises
    // the activation — the precision the exception below preserves is worth a cache hit, not a
    // fresh 42 MB dequant per step. It is also the route the answer depends on: the cached
    // route sums in another order, so a cache whose thrash latch trips mid-run makes one call
    // of a process answer differently from the next (`crate::dequant_cache`).
    if !crate::dequant_cache::cache_enabled() {
        return false;
    }
    // Cache pressure skips the cache for every route whose fused arm leaves the numerics
    // alone: `m > 1`, and every single-row call whose arm keeps f32 activations. There
    // the two routes do the same total dequant work and the fused one skips the
    // bookkeeping. The schemes whose arm rounds the activation
    // ([`crate::gguf_activation::fused_m1_quantizes_activation`]) keep the cached route
    // whatever the budget does: `cache_thrashing()` is monotone process state that never
    // reverts, so diverting them would flip one process's answer part-way through — the
    // defect `tests/gguf_single_row_cache_pressure.rs` pins. The eight-bit arm's taken
    // calls never get here: the dispatch diverts them before the cache is consulted.
    if crate::dequant_cache::cache_thrashing()
        && (m > 1 || !crate::gguf_activation::fused_m1_quantizes_activation(scheme))
    {
        return false;
    }
    m > 1 || k.saturating_mul(n) >= CACHED_BLAS_MIN_WEIGHT_ELEMS
}

/// Dequant once (cached by weight bytes) + Accelerate/OpenBLAS `sgemm_bt`.
///
/// `C[m,n] = A[m,k] @ B^T` with GGUF `B` stored `[n,k]` row-major. Mirrors the MLX
/// dequant-cache path; repeated decode matmuls on the same static param reuse f32 weights.
#[cfg(rlx_cpu_blas)]
pub fn gguf_matmul_bt_cached(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    w_off: usize,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    // Portable runs take the BLAS-less build's path, the fused blocks.
    if crate::config::portable() {
        return gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
    }
    let w_f32 = crate::dequant_cache::gguf_weight_f32(w_off, w_bytes, k, n, scheme);
    if m == 1 {
        out.fill(0.0);
        crate::blas::sgemv_nn(w_f32.as_ref(), x, out, n, k, 1.0, 0.0);
    } else {
        crate::blas::sgemm_bt(x, w_f32.as_ref(), out, m, k, n, 1.0);
    }
}

#[cfg(not(rlx_cpu_blas))]
pub fn gguf_matmul_bt_cached(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    _w_off: usize,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    // No linked BLAS: the portable gemm is slower than the fused kernel, so the
    // fused path is the right default here — and the same schemes are admitted to
    // it as in the BLAS build. Q4_K is the one single-row scheme that needs a
    // substitute, and it gets the exact packed kernel rather than a materialized
    // copy: no f32 slab, no dequant cache, no Q8_K activation. (Q6_K's quantizing
    // arm is already packed-only and needs no BLAS, so it needs none.) Shape
    // decides that, never the budget.
    if m == 1 && scheme == QuantScheme::GgufQ4K && k.is_multiple_of(QK_K) {
        q4k_exact_gemv_rows(x, w_bytes, out, k);
        return;
    }
    gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
}

/// Default GGUF matmul entry: cached BLAS when available, else legacy fused blocks.
pub fn gguf_matmul_bt_dispatch(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_dispatch_at(x, w_bytes, out, m, k, n, scheme, 0);
}

/// Like [`gguf_matmul_bt_dispatch`] but keys the dequant cache by arena `w_off`.
pub fn gguf_matmul_bt_dispatch_at(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    w_off: usize,
) {
    // Q2_0 decode GEMV: int8 VNNI/NEON dot (llama.cpp-style) beats
    // dequant→f32→BLAS and stays on the int path even for large `k*n`, so it
    // bypasses `prefer_cached_blas`. Default-on, runtime-detected.
    if m == 1 && scheme == QuantScheme::GgufQ2_0 && k.is_multiple_of(QK2_0) {
        q2_0_gemv_parallel(x, w_bytes, out, k);
        return;
    }
    // Large Q4_K decode GEMV: the parallel int8 (Q8_K-dot) kernel beats
    // cached-f32-BLAS (single-thread AMX) once `n` is big — measured ~3× at
    // n≈128k, ~tie at n≈3k, BLAS ahead below. So route wide matmuls (LM head,
    // large FFN gate/up) to the fused path; narrow ones stay on AMX. Tune the
    // crossover with `RLX_Q4K_FUSED_MIN_N` (set huge to disable).
    if m == 1
        && scheme == QuantScheme::GgufQ4K
        && k.is_multiple_of(QK_K)
        && n >= q4k_fused_decode_min_n()
        && crate::pool::num_threads() > 1
        && !rlx_ir::reproducible::reproducible()
    {
        gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
        return;
    }
    // Q8_0 decode GEMV: the integer dot beats the f32 unpack on every shape the code
    // predictor has, and it rounds the activation, so it is taken whatever the cache
    // holds — a route that moved with the cache would answer one call differently from
    // the next in the same process. Default-on, runtime-detected, aarch64 only;
    // `RLX_Q8_0_EXACT_GEMV` is the bisect handle that puts the f32 route back. (Q2_0's
    // arm above is the same arrangement and has already returned.)
    if m == 1 && q8_0_arm_ignores_cache(k, n, w_bytes.len(), scheme) {
        gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
        return;
    }
    if prefer_cached_blas(k, n, m, scheme) {
        gguf_matmul_bt_cached(x, w_bytes, out, m, k, n, scheme, w_off);
    } else {
        gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
    }
}

/// Fused GGUF dequant + grouped matmul for MoE expert stacks.
///
/// `w_bytes` holds `num_experts` contiguous packed slabs; expert `e` occupies
/// `[e * slab_bytes .. (e+1) * slab_bytes)` with the same GGML layout as a
/// standalone 2-D K-quant matrix of shape `[n, k]`.
pub fn gguf_grouped_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) {
    gguf_grouped_matmul_bt_ex(
        x,
        w_bytes,
        expert_idx,
        out,
        m,
        k,
        n,
        num_experts,
        scheme,
        /*cache=*/ true,
    );
}

/// Like [`gguf_grouped_matmul_bt`] but never materializes expert slabs into the
/// process-wide F32 dequant cache (required for large MoE packs).
///
/// Unique expert groups run in parallel; each group uses the serial fused
/// kernel so Rayon is not nested. Tokens that share an expert are batched
/// (`m = count`) inside that group.
pub fn gguf_grouped_matmul_bt_fused(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) {
    gguf_grouped_matmul_bt_ex(
        x,
        w_bytes,
        expert_idx,
        out,
        m,
        k,
        n,
        num_experts,
        scheme,
        /*cache=*/ false,
    );
}

fn gguf_grouped_matmul_bt_ex(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
    cache: bool,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(expert_idx.len(), m);
    assert_eq!(out.len(), m * n);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(w_bytes.len(), num_experts * slab_bytes);

    let (packed_in, original_pos, offsets) =
        grouped_moe_sort_plan(x, expert_idx, m, k, num_experts);

    let jobs: Vec<(usize, usize, usize)> = (0..num_experts)
        .filter_map(|e| {
            let count = offsets[e + 1] - offsets[e];
            (count > 0).then_some((e, offsets[e], count))
        })
        .collect();

    let mut packed_out = vec![0f32; m * n];

    if cache {
        for &(e, start, count) in &jobs {
            let in_slice = &packed_in[start * k..(start + count) * k];
            let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let out_slice = &mut packed_out[start * n..(start + count) * n];
            gguf_matmul_bt_dispatch(in_slice, w_slice, out_slice, count, k, n, scheme);
        }
    } else if jobs.len() > 1 {
        // Parallelize over unique experts; serial fused GEMM per group so Rayon
        // is not nested. Batches tokens that share an expert (`count > 1`).
        {
            use rayon::prelude::*;
            let chunks: Vec<(usize, Vec<f32>)> = jobs
                .par_iter()
                .map(|&(e, start, count)| {
                    let in_slice = &packed_in[start * k..(start + count) * k];
                    let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
                    let mut local = vec![0f32; count * n];
                    gguf_matmul_bt_serial(in_slice, w_slice, &mut local, count, k, n, scheme);
                    (start, local)
                })
                .collect();
            for (start, local) in chunks {
                let count = local.len() / n;
                packed_out[start * n..(start + count) * n].copy_from_slice(&local);
            }
        }
    } else {
        // Single expert group: full-width parallel fused GEMM.
        for &(e, start, count) in &jobs {
            let in_slice = &packed_in[start * k..(start + count) * k];
            let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let out_slice = &mut packed_out[start * n..(start + count) * n];
            gguf_matmul_bt(in_slice, w_slice, out_slice, count, k, n, scheme);
        }
    }

    grouped_moe_unpermute_out(&packed_out, &original_pos, out, m, n);
}

/// Dequant an MoE expert stack `[E, K, N]` into GroupedMatMul layout (row-major
/// `[k, n]` slabs per expert). Used by `Op::DequantMoEWeights` and autodiff.
pub fn dequant_moe_weights_to_grouped_f32(
    packed: &[u8],
    out: &mut [f32],
    num_experts: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(packed.len(), num_experts * slab_bytes);
    assert_eq!(out.len(), num_experts * k * n);
    for e in 0..num_experts {
        let slab = &packed[e * slab_bytes..(e + 1) * slab_bytes];
        let deq = match scheme {
            QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(slab, k * n),
            QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(slab, k * n),
            QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(slab, k * n),
            QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(slab, k * n),
            QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k(slab, k * n),
            QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k(slab, k * n),
            QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0(slab, k * n),
            QuantScheme::GgufQ4_1 => rlx_gguf::dequant_q4_1(slab, k * n),
            QuantScheme::GgufQ5_0 => rlx_gguf::dequant_q5_0(slab, k * n),
            QuantScheme::GgufQ5_1 => rlx_gguf::dequant_q5_1(slab, k * n),
            QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0(slab, k * n),
            QuantScheme::GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(slab, k * n),
            QuantScheme::GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(slab, k * n),
            QuantScheme::GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(slab, k * n),
            QuantScheme::GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(slab, k * n),
            QuantScheme::GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(slab, k * n),
            QuantScheme::GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(slab, k * n),
            QuantScheme::GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(slab, k * n),
            QuantScheme::GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(slab, k * n),
            QuantScheme::GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(slab, k * n),
            QuantScheme::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(slab, k * n),
            QuantScheme::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(slab, k * n),
            QuantScheme::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(slab, k * n),
            QuantScheme::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(slab, k * n),
            QuantScheme::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(slab, k * n),
            QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(slab, k * n),
            QuantScheme::GgufG8_0 => rlx_gguf::g8_dequant::dequant_g8_0(slab, k * n),
            QuantScheme::GgufPtq1_0 => rlx_gguf::ptq1_dequant::dequant_ptq1_0(slab, k * n),
            QuantScheme::Fp8E4m3B128 | QuantScheme::Fp8E5m2B128 => Ok(
                rlx_ir::lowp_codec::fp8_b128::dequant_scheme(scheme, slab, k * n)
                    .expect("dequant_moe_weights: slab dequant failed"),
            ),
            other => panic!("dequant_moe_weights: unsupported scheme {other:?}"),
        }
        .expect("dequant_moe_weights: slab dequant failed");
        let base = e * k * n;
        for i in 0..k {
            for j in 0..n {
                out[base + i * n + j] = deq[j * k + i];
            }
        }
    }
}

/// Counting-sort tokens by expert (shared by host and GPU prep paths).
pub fn grouped_moe_sort_plan(
    x: &[f32],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    num_experts: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut counts = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        debug_assert!(e < num_experts);
        counts[e] += 1;
    }
    let mut offsets = vec![0usize; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut packed_in = vec![0f32; m * k];
    let mut original_pos = vec![0usize; m];
    let mut write_idx = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        let dst_row = offsets[e] + write_idx[e];
        packed_in[dst_row * k..(dst_row + 1) * k].copy_from_slice(&x[i * k..(i + 1) * k]);
        original_pos[dst_row] = i;
        write_idx[e] += 1;
    }
    (packed_in, original_pos, offsets)
}

pub fn grouped_moe_unpermute_out(
    packed_out: &[f32],
    original_pos: &[usize],
    out: &mut [f32],
    m: usize,
    n: usize,
) {
    for packed_idx in 0..m {
        let i = original_pos[packed_idx];
        out[i * n..(i + 1) * n].copy_from_slice(&packed_out[packed_idx * n..(packed_idx + 1) * n]);
    }
}

/// Parallel fused matmul — delegates to [`gguf_matmul_bt_dispatch`].
pub fn gguf_matmul_bt_parallel(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_dispatch(x, w_bytes, out, m, k, n, scheme);
}

/// Decode GEMV (`m == 1`): single-threaded block fold.
fn gguf_matmul_bt_m1_sequential(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    scheme: QuantScheme,
    num_blocks: usize,
    block_bytes: usize,
    block_elems: usize,
    k: usize,
) {
    if scheme == QuantScheme::GgufQ4K && k.is_multiple_of(QK_K) && block_elems == QK_K {
        q4k_gemv_rowmajor(x_row, w_bytes, out, k);
        return;
    }
    let mut block_f32 = [0f32; QK_K];
    for bi in 0..num_blocks {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            let j = idx / k;
            let p = idx % k;
            out[j] += x_row[p] * block_f32[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Per-precision GEMV / GEMM kernels, generated.
//
// The dynamic form calls `dequant_block(scheme, ..)` once per 256-element block,
// which is a runtime `match` on the scheme in the innermost loop: the compiler
// cannot inline the actual dequant, so it cannot fuse or vectorize across the
// dequant→dot boundary. Generating one monomorphic kernel per precision makes
// the dequant a direct call, and lets the dot use four independent accumulators
// (a single `acc` serializes on FP latency, ~4 cycles per element).
//
// Adding a precision is one line in `gguf_qk_kernels!`. Every scheme here uses
// the 256-element super-block layout and the `(&[u8], &mut [f32; QK_K])` block
// signature, so the macro body is genuinely shared rather than copy-pasted.
// ---------------------------------------------------------------------------

/// 256-element dot with 4 independent accumulators.
///
/// Both callers are the rayon-parallel `gguf_qk_kernels!` GEMV/GEMM.
#[inline(always)]
fn dot256(xs: &[f32], blk: &[f32; QK_K]) -> f32 {
    let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
    let mut t = 0;
    while t < QK_K {
        a0 += xs[t] * blk[t];
        a1 += xs[t + 1] * blk[t + 1];
        a2 += xs[t + 2] * blk[t + 2];
        a3 += xs[t + 3] * blk[t + 3];
        t += 4;
    }
    (a0 + a1) + (a2 + a3)
}

macro_rules! gguf_qk_kernels {
    ($( $scheme:path => ($gemv:ident, $gemm:ident, $dq:path) ),+ $(,)?) => {
        $(
            /// Row-parallel GEMV (`m == 1`) specialized to one precision.
            fn $gemv(x_row: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize, block_bytes: usize) {
                use rayon::prelude::*;
                let blocks_per_row = k / QK_K;
                let row_bytes = blocks_per_row * block_bytes;
                out.par_iter_mut().enumerate().for_each(|(j, slot)| {
                    let start = j * row_bytes;
                    let Some(row) = w_bytes.get(start..start + row_bytes) else {
                        *slot = 0.0;
                        return;
                    };
                    let mut blk = [0f32; QK_K];
                    let mut acc = 0f32;
                    for b in 0..blocks_per_row {
                        $dq(&row[b * block_bytes..(b + 1) * block_bytes], &mut blk);
                        acc += dot256(&x_row[b * QK_K..(b + 1) * QK_K], &blk);
                    }
                    *slot = acc;
                });
            }

            /// Row-parallel GEMM (`m > 1`) specialized to one precision. Each block
            /// is dequantized ONCE and reused across all `m` rows.
            #[allow(clippy::too_many_arguments)]
            fn $gemm(
                x: &[f32], w_bytes: &[u8], out: &mut [f32],
                m: usize, k: usize, n: usize, block_bytes: usize,
            ) {
                use rayon::prelude::*;
                let blocks_per_row = k / QK_K;
                let row_bytes = blocks_per_row * block_bytes;
                let cols: Vec<Vec<f32>> = (0..n).into_par_iter().map(|j| {
                    let mut acc = vec![0f32; m];
                    let start = j * row_bytes;
                    let Some(row) = w_bytes.get(start..start + row_bytes) else { return acc };
                    let mut blk = [0f32; QK_K];
                    for b in 0..blocks_per_row {
                        $dq(&row[b * block_bytes..(b + 1) * block_bytes], &mut blk);
                        let off = b * QK_K;
                        for (mi, a) in acc.iter_mut().enumerate() {
                            *a += dot256(&x[mi * k + off..mi * k + off + QK_K], &blk);
                        }
                    }
                    acc
                }).collect();
                for (j, acc) in cols.iter().enumerate() {
                    for (mi, a) in acc.iter().enumerate() {
                        out[mi * n + j] = *a;
                    }
                }
            }
        )+

        /// Route to a specialized GEMV when one exists. `false` ⇒ caller falls back.
        fn gguf_gemv_specialized(
            scheme: QuantScheme, x_row: &[f32], w_bytes: &[u8], out: &mut [f32],
            k: usize, block_bytes: usize,
        ) -> bool {
            match scheme {
                $( $scheme => { $gemv(x_row, w_bytes, out, k, block_bytes); true } )+
                _ => false,
            }
        }

        /// Route to a specialized GEMM when one exists. `false` ⇒ caller falls back.
        #[allow(clippy::too_many_arguments)]
        fn gguf_gemm_specialized(
            scheme: QuantScheme, x: &[f32], w_bytes: &[u8], out: &mut [f32],
            m: usize, k: usize, n: usize, block_bytes: usize,
        ) -> bool {
            match scheme {
                $( $scheme => { $gemm(x, w_bytes, out, m, k, n, block_bytes); true } )+
                _ => false,
            }
        }
    };
}

// NOTE: `GgufQ8K` is deliberately absent — it is the ACTIVATION quantization
// format, never a weight dtype in these checkpoints, so a weight-GEMV
// specialization for it would add reassociation risk for no gain.
gguf_qk_kernels! {
    QuantScheme::GgufQ2K => (gemv_q2k, gemm_q2k, rlx_gguf::dequant_q2_k_block),
    QuantScheme::GgufQ3K => (gemv_q3k, gemm_q3k, rlx_gguf::dequant_q3_k_block),
    QuantScheme::GgufQ4K => (gemv_q4k, gemm_q4k, rlx_gguf::dequant_q4_k_block),
    QuantScheme::GgufQ5K => (gemv_q5k, gemm_q5k, rlx_gguf::dequant_q5_k_block),
    QuantScheme::GgufQ6K => (gemv_q6k, gemm_q6k, rlx_gguf::dequant_q6_k_block),
}

/// Generic decode GEMV (`m == 1`) for ANY GGUF scheme with a row-aligned `k`.
///
/// The old generic path was doubly pathological: it walked `k*n` elements
/// SERIALLY and recovered the output row / input position with `idx / k` and
/// `idx % k` — two integer divisions PER ELEMENT. Only `Q4_K` (and `Q2_0`) had a
/// fast row-dot, so every other scheme took it, including `Q5_K`, which is what
/// unsloth ships the Muse-Glimmer LM head as: `[6656, 202048]` = 1.345B params
/// on a serial scalar loop.
///
/// When `k` is a whole number of blocks, block `b` of row `j` covers input
/// positions `[b*block_elems, (b+1)*block_elems)` — the row index is structural,
/// so no division is needed at all. Rows are independent, so this parallelizes
/// over `n` exactly like the Q4_K path, and the inner loop is a contiguous
/// f32 dot the autovectorizer can handle.
fn gguf_matmul_bt_m1_rows_generic(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    k: usize,
    scheme: QuantScheme,
    block_bytes: usize,
    block_elems: usize,
) {
    use rayon::prelude::*;
    if block_elems == QK_K && gguf_gemv_specialized(scheme, x_row, w_bytes, out, k, block_bytes) {
        return;
    }
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let start = j * row_bytes;
        let Some(row) = w_bytes.get(start..start + row_bytes) else {
            *slot = 0.0;
            return;
        };
        let mut blk = [0f32; QK_K];
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            dequant_block(
                scheme,
                &row[b * block_bytes..(b + 1) * block_bytes],
                &mut blk,
            );
            let xs = &x_row[b * block_elems..(b + 1) * block_elems];
            for t in 0..block_elems {
                acc += xs[t] * blk[t];
            }
        }
        *slot = acc;
    });
}

/// Generic GEMM (`m > 1`) for any GGUF scheme with a row-aligned `k`.
///
/// The stock `m > 1` fallback walks `k*n` elements serially and recovers indices
/// with `idx / k` + `idx % k` — two integer divisions per element — which is why
/// fully-fused prefill measured 111 s against 61 s for dequant+BLAS. This does
/// the same arithmetic without either problem: parallel over output columns,
/// each block dequantized ONCE and reused across all `m` rows (so total dequant
/// work matches the cached path), and contiguous inner dots.
///
/// The point is to beat cached BLAS *when the dequant cache is thrashing*, where
/// "cached" degenerates into re-dequantizing the whole model per call plus
/// allocator churn. With a healthy cache, BLAS still wins and callers keep it.
fn gguf_matmul_bt_rows_generic_gemm(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    block_bytes: usize,
    block_elems: usize,
) {
    use rayon::prelude::*;
    if block_elems == QK_K && gguf_gemm_specialized(scheme, x, w_bytes, out, m, k, n, block_bytes) {
        return;
    }
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    // One task per output column; each writes `out[mi * n + j]` for every mi, so
    // scatter into a per-column buffer then transpose in.
    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut acc = vec![0f32; m];
            let start = j * row_bytes;
            let Some(row) = w_bytes.get(start..start + row_bytes) else {
                return acc;
            };
            let mut blk = [0f32; QK_K];
            for b in 0..blocks_per_row {
                dequant_block(
                    scheme,
                    &row[b * block_bytes..(b + 1) * block_bytes],
                    &mut blk,
                );
                let off = b * block_elems;
                for (mi, a) in acc.iter_mut().enumerate() {
                    let xs = &x[mi * k + off..mi * k + off + block_elems];
                    let mut s = 0f32;
                    for t in 0..block_elems {
                        s += xs[t] * blk[t];
                    }
                    *a += s;
                }
            }
            acc
        })
        .collect();
    for (j, acc) in cols.iter().enumerate() {
        for (mi, a) in acc.iter().enumerate() {
            out[mi * n + j] = *a;
        }
    }
}

/// Decode GEMV (`m == 1`): fold/reduce over GGUF super-blocks across Rayon workers.
fn gguf_matmul_bt_m1_parallel(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    k: usize,
    n: usize,
    scheme: QuantScheme,
    num_blocks: usize,
    block_bytes: usize,
    block_elems: usize,
) {
    // Q4_K included, unless the exact f32-activation arm below is asked for.
    #[cfg(all(feature = "int8-gemv", rlx_cpu_x86))]
    if out.len() == n
        && !(scheme == QuantScheme::GgufQ4K && q4k_exact_gemv())
        && crate::int8_gemv::gemv(scheme, x_row, w_bytes, out, k)
    {
        return;
    }
    if scheme == QuantScheme::GgufQ4K && k.is_multiple_of(QK_K) && block_elems == QK_K {
        // Independent output rows — no n-wide partial reduction.
        {
            use rayon::prelude::*;
            let blocks_per_row = k / QK_K;
            let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
            debug_assert_eq!(n * row_bytes, w_bytes.len().min(n * row_bytes));
            // A Q4_K decode GEMV reaches this branch only on a tile wide enough
            // and a pool threaded enough for `gguf_matmul_bt_ex` to choose it, and
            // only when the fused path is taken on purpose: `RLX_Q4K_FUSED_MIN_N`,
            // `RLX_GGUF_MATMUL_LEGACY`, or a caller that calls `gguf_matmul_bt`
            // directly instead of going through the dispatch. There
            // `RLX_Q4K_EXACT_GEMV` picks the exact kernel over the Q8_K one.
            if q4k_exact_gemv() {
                q4k_exact_gemv_rows(x_row, w_bytes, out, k);
                return;
            }
            let mut x_q8 = vec![0u8; blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES];
            rlx_gguf::quantize_q8_k_row(&x_row[..k], &mut x_q8);
            out.par_iter_mut().enumerate().for_each(|(j, slot)| {
                let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
                *slot = q4k_dot_row_q8(row, &x_q8, blocks_per_row);
            });
            return;
        }
    }
    // Q6_K: the output head and every `ffn_down`, dotted as integers against one
    // Q8_K-quantised activation row instead of being unpacked to f32 in scalar
    // code. aarch64 only; x86-64 reaches the same kernel through `int8-gemv`.
    if scheme == QuantScheme::GgufQ6K
        && k.is_multiple_of(QK_K)
        && block_elems == QK_K
        && out.len() == n
        && crate::gguf_q6k::gemv(x_row, w_bytes, out, k)
    {
        return;
    }
    // Q8_0: the code predictor's every multiply, dotted as integers against one
    // Q8_K-quantised activation row instead of 32 elements at a time through a f32
    // unpack. aarch64 only; x86-64 reaches the same width through `int8-gemv`.
    if scheme == QuantScheme::GgufQ8_0
        && out.len() == n
        && crate::gguf_q8_0::gemv(x_row, w_bytes, out, k)
    {
        return;
    }
    // Every other scheme (Q5_K / Q2_K / Q3_K / IQ*): row-parallel,
    // division-free generic GEMV whenever `k` is a whole number of blocks.
    if k.is_multiple_of(block_elems) && block_elems <= QK_K && out.len() == n {
        gguf_matmul_bt_m1_rows_generic(x_row, w_bytes, out, k, scheme, block_bytes, block_elems);
        return;
    }
    {
        use rayon::prelude::*;

        let partial = (0..num_blocks)
            .into_par_iter()
            .fold(
                || vec![0f32; n],
                |mut local, bi| {
                    let off = bi * block_bytes;
                    let mut block_f32 = [0f32; QK_K];
                    dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
                    let idx0 = bi * block_elems;
                    for t in 0..block_elems {
                        let idx = idx0 + t;
                        let j = idx / k;
                        let p = idx % k;
                        local[j] += x_row[p] * block_f32[t];
                    }
                    local
                },
            )
            .reduce(
                || vec![0f32; n],
                |mut acc, chunk| {
                    for (a, b) in acc.iter_mut().zip(chunk) {
                        *a += b;
                    }
                    acc
                },
            );
        out.copy_from_slice(&partial);
    }
}

/// One Q4_K row of an exact decode GEMV: each super-block dequantized on the fly
/// and dotted against the *unquantized* activation.
#[inline]
fn q4k_exact_row_dot(x_row: &[f32], row: &[u8], blocks_per_row: usize) -> f32 {
    let bb = rlx_gguf::Q4K_BLOCK_BYTES;
    let mut acc = 0f32;
    for b in 0..blocks_per_row {
        acc += rlx_gguf::q4_k_dot_f32(&row[b * bb..(b + 1) * bb], &x_row[b * QK_K..(b + 1) * QK_K]);
    }
    acc
}

/// Exact Q4_K decode GEMV (`m == 1`, `k` a multiple of [`QK_K`]) — row-parallel,
/// packed bytes read in place, activation kept in f32.
///
/// Fills the gap between the two pre-existing options: cached-f32-BLAS is exact
/// but materializes the whole matrix as f32 (8.6x the checkpoint on
/// LFM2.5-2.6B), while the Q8_K path reads packed bytes in place but perturbs the
/// activation. This one trades ALU for both.
///
/// Measured on LFM2 (arms alternated in one process, min-of-4), against the two
/// exact options that already existed:
///
/// | exact path | 350M | 2.6B |
/// |---|---|---|
/// | cached-f32-BLAS      | 36.0 tok/s | 9.0 tok/s |
/// | `RLX_DEQUANT_CACHE=0`|  12.0      | 1.8       |
/// | this kernel          |  23.8      | 5.1       |
///
/// So it does not beat the f32 cache when the cache fits — it is 2.8x the
/// throughput of the only other exact option when it does not. `decode_parity_live`
/// passes on 350M and 2.6B with this arm enabled, and fails with the Q8_K arm.
///
/// NOT established: its effect on RSS. Routing is adaptive (`cache_thrashing()` is
/// accumulated runtime state), so per-arm peak-RSS measurements are path-dependent
/// and did not hold still across flag combinations — including one standing oddity
/// that reproduces without this kernel, where lowering `RLX_Q4K_FUSED_MIN_N` from
/// 2048 to 1 *raised* 2.6B RSS from 5.17 GB to 14.46 GB when it should have
/// lowered it.
fn q4k_exact_gemv_rows(x_row: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
    debug_assert!(w_bytes.len() >= out.len() * row_bytes);
    let row = |j: usize| &w_bytes[j * row_bytes..(j + 1) * row_bytes];
    {
        use rayon::prelude::*;
        out.par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| *slot = q4k_exact_row_dot(x_row, row(j), blocks_per_row));
    }
}

/// `C[n] = x[k] @ W[n,k]^T` for Q4_K with `k` a multiple of [`QK_K`].
///
/// Activations are quantized once to Q8_K; each output row then uses the
/// int8 Q4_K×Q8_K vec-dot (llama.cpp-style). This is the hot path for MoE
/// expert GEMVs.
fn q4k_gemv_rowmajor(x: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
    debug_assert!(x.len() >= k);
    debug_assert!(w_bytes.len() >= n * row_bytes);

    let mut x_q8 = vec![0u8; blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES];
    rlx_gguf::quantize_q8_k_row(&x[..k], &mut x_q8);

    for j in 0..n {
        let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
        out[j] = q4k_dot_row_q8(row, &x_q8, blocks_per_row);
    }
}

/// Like [`q4k_gemv_rowmajor`] but `x` is already packed Q8_K (for parallel rows).
#[allow(dead_code)] // kept for callers that pre-quantize activations
fn q4k_gemv_rowmajor_q8(x_q8: &[u8], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
    debug_assert_eq!(x_q8.len(), blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES);
    for j in 0..n {
        let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
        out[j] = q4k_dot_row_q8(row, x_q8, blocks_per_row);
    }
}

/// One weight row of `blocks_per_row` Q4_K super-blocks dotted against the Q8_K
/// activation row, in super-block order.
///
/// aarch64 picks its arm once per row rather than eight times per block: the two arms are
/// the same arithmetic spelled with and without `dotprod`, and the probe behind the choice
/// is an atomic load that has no business in the row's hot loop.
#[inline]
fn q4k_dot_row_q8(row_bytes: &[u8], x_q8: &[u8], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // The arms read both slices with raw pointers, so the bound they rely on is stated
        // here rather than re-checked per block.
        debug_assert!(row_bytes.len() >= blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES);
        debug_assert!(x_q8.len() >= blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES);
        // SAFETY: both slices hold whole blocks, as just stated, and the probe picks an arm
        // the host has.
        unsafe {
            if aarch64_has_dotprod() {
                q4k_dot_row_q8_dotprod(row_bytes, x_q8, blocks_per_row)
            } else {
                q4k_dot_row_q8_wide(row_bytes, x_q8, blocks_per_row)
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let bb = rlx_gguf::Q4K_BLOCK_BYTES;
        let qb = rlx_gguf::Q8K_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let q4 = &row_bytes[b * bb..(b + 1) * bb];
            let q8 = &x_q8[b * qb..(b + 1) * qb];
            acc += q4k_dot_q8_block(q4, q8);
        }
        acc
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn q4k_dot_q8_block(q4: &[u8], q8: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: gated on runtime AVX2 detection; slabs are full blocks.
        return unsafe { q4k_dot_q8_block_avx2(q4, q8) };
    }
    rlx_gguf::q4_k_dot_q8_k(q4, q8)
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_K block dot, AVX2 (x86-64).
//
// aarch64 had a hand-written NEON dot while x86 fell through to the pure-scalar
// `rlx_gguf::q4_k_dot_q8_k` — a nibble-at-a-time loop over every weight. Since
// Q4_K is ~95% of a K-quant checkpoint's parameters, that scalar loop WAS the
// x86 decode cost. It keeps the NEON arm's group order and its i32 sums, so the
// two are bit-comparable.
//
// The inner shape maps cleanly onto AVX2: Q4 nibbles are unsigned 0..15 and Q8
// activations are signed i8, which is precisely `_mm256_maddubs_epi16`
// (u8 x i8 -> i16 pairwise sums), then `_mm256_madd_epi16` against ones widens
// to i32 without overflow (max |sum| per pair = 15*127*2 << i16::MAX).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn q4k_scale_min_x86(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Horizontal sum of 8 x i32 in a `__m256i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    // No `unsafe` block: every intrinsic here is register-only, and those are
    // safe fns once the enabling `#[target_feature]` is on the signature. The
    // block was this crate's only warning.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let sh = _mm_shuffle_epi32(s, 0b01_00_11_10);
    let s = _mm_add_epi32(s, sh);
    let sh2 = _mm_shuffle_epi32(s, 0b10_11_00_01);
    _mm_cvtsi128_si32(_mm_add_epi32(s, sh2))
}

/// BOTH nibble halves of one 32-byte Q4 group, from a SINGLE load.
///
/// Q4_K stores the low nibbles of group `j` and the high nibbles of group
/// `j+1` in the same bytes, and their two Q8 segments are adjacent, so one load
/// feeds both. Returns the `maddubs` i16 pair sums `(lo, hi)` (max 15·127·2, so
/// the caller's `madd` by a 6-bit scale cannot overflow).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4k_group32_q8_maddubs_avx2(
    qs: *const u8,
    q8_lo: *const i8,
) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    unsafe {
        let qbytes = _mm256_loadu_si256(qs as *const __m256i);
        let mask = _mm256_set1_epi8(0x0F);
        let nib_lo = _mm256_and_si256(qbytes, mask);
        let nib_hi = _mm256_and_si256(_mm256_srli_epi16(qbytes, 4), mask);
        let y_lo = _mm256_loadu_si256(q8_lo as *const __m256i);
        let y_hi = _mm256_loadu_si256(q8_lo.add(32) as *const __m256i);
        (
            _mm256_maddubs_epi16(nib_lo, y_lo),
            _mm256_maddubs_epi16(nib_hi, y_hi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_dot_q8_block_avx2(q4: &[u8], q8: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let d = half::f16::from_le_bytes([q4[0], q4[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([q4[2], q4[3]]).to_f32();
        let scales = &q4[4..16];
        let qs = q4.as_ptr().add(16);
        let yd = f32::from_le_bytes([q8[0], q8[1], q8[2], q8[3]]);
        let q8s = q8.as_ptr().add(4) as *const i8;
        let bsums = q8.as_ptr().add(4 + QK_K);

        let mut sumi_min = 0i32;
        for j in 0..16 {
            let (_, m) = q4k_scale_min_x86(j / 2, scales);
            let bs = i16::from_le_bytes([*bsums.add(j * 2), *bsums.add(j * 2 + 1)]) as i32;
            sumi_min += m as i32 * bs;
        }

        // Scale each group in-register (`madd` against the broadcast 6-bit
        // scale) and reduce once per block. Integer sums are exact, so this is
        // bit-identical to scaling per-group hsums — it just drops 7 of the 8
        // horizontal reductions.
        let mut acc = _mm256_setzero_si256();
        let mut is = 0usize;
        let mut yi = 0usize;
        for j in (0..8).step_by(2) {
            let (sc0, _) = q4k_scale_min_x86(j, scales);
            let (sc1, _) = q4k_scale_min_x86(j + 1, scales);
            let (p0, p1) = q4k_group32_q8_maddubs_avx2(qs.add(is), q8s.add(yi));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p0, _mm256_set1_epi16(sc0 as i16)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p1, _mm256_set1_epi16(sc1 as i16)));
            yi += 64;
            is += 32;
        }
        let sumi = hsum_i32_avx2(acc);
        d * yd * sumi as f32 - dmin * yd * sumi_min as f32
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_K, aarch64.
//
// A four-bit super-block is asymmetric — each of its eight 32-element groups carries a
// six-bit minimum as well as a six-bit scale — so the block is
// `d*yd*Σ scale·(group dot) − dmin*yd*Σ min·(16-element activation sum)`. Both folds ran
// in scalar code and between them were more than half the block; 16.4 to 6.9 ns per
// 144-byte block, 2.4x.
//
// Exactness is structural: every term is an integer, so re-associating it cannot move a
// bit, and the block still closes on `d * yd * sumi as f32 - dmin * yd * sumi_min as f32`
// on the same i32 sums, with no fused multiply-add and nothing widened or narrowed
// (`q4k_block_dot_is_bit_exact` holds both arms to `rlx_gguf::q4_k_dot_q8_k`).
// ---------------------------------------------------------------------------

/// The two arms of the block dot differ only in their 16-element dot, so the super-block
/// arithmetic is written once here. `$dot` accumulates into a lane set the caller holds.
#[cfg(target_arch = "aarch64")]
macro_rules! q4k_row_q8 {
    ($(#[$attr:meta])* fn $name:ident, $dot:path) => {
        $(#[$attr])*
        /// One weight row dotted against a whole Q8_K activation row.
        ///
        /// SAFETY: both slices hold whole super-blocks, and the dot this arm is named for
        /// is available on the host.
        unsafe fn $name(row_bytes: &[u8], x_q8: &[u8], blocks_per_row: usize) -> f32 {
            use std::arch::aarch64::*;
            // SAFETY: whole blocks, as the caller checked; every load below stays inside
            // them.
            unsafe {
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let q4 = row_bytes.as_ptr().add(b * rlx_gguf::Q4K_BLOCK_BYTES);
                    let q8 = x_q8.as_ptr().add(b * rlx_gguf::Q8K_BLOCK_BYTES);
                    let d = half::f16::from_le_bytes([*q4, *q4.add(1)]).to_f32();
                    let dmin = half::f16::from_le_bytes([*q4.add(2), *q4.add(3)]).to_f32();
                    let yd = f32::from_le_bytes([*q8, *q8.add(1), *q8.add(2), *q8.add(3)]);

                    // The eight six-bit scales and the eight six-bit minimums share the
                    // twelve bytes at `q4[4..16]`: the first four hold scales 0..3, the next
                    // four minimums 0..3, and the last four a low nibble per scale and a high
                    // nibble per minimum for 4..7, whose top two bits sit above the first
                    // eight. One 16-byte load covers all of it — the four bytes it reads past
                    // the scales are the head of `qs`, which no lane below uses.
                    let w = vld1q_u8(q4.add(4));
                    let tops = vshlq_n_u8::<4>(vshrq_n_u8::<6>(w));
                    let six = vandq_u8(w, vdupq_n_u8(0x3F));
                    let scale_hi =
                        vorrq_u8(vandq_u8(vextq_u8::<8>(w, w), vdupq_n_u8(0x0F)), tops);
                    let min_hi = vorrq_u8(vshrq_n_u8::<4>(vextq_u8::<4>(w, w)), tops);
                    let scale8 = vzip1_u32(
                        vget_low_u32(vreinterpretq_u32_u8(six)),
                        vget_low_u32(vreinterpretq_u32_u8(scale_hi)),
                    );
                    let min8 = vzip2_u32(
                        vget_low_u32(vreinterpretq_u32_u8(six)),
                        vget_low_u32(vreinterpretq_u32_u8(min_hi)),
                    );
                    let scale16 = vmovl_u8(vreinterpret_u8_u32(scale8));
                    let scales_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scale16)));
                    let scales_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scale16)));
                    let mins = vmovl_u8(vreinterpret_u8_u32(min8));

                    // The minimum correction: one product per group, over the block's own
                    // 16-element activation sums taken pairwise, so sixteen terms fold to
                    // eight with nothing narrower than i32 in between.
                    let bsums = q8.add(4 + QK_K) as *const i16;
                    let bs_lo = vpadalq_s16(vdupq_n_s32(0), vld1q_s16(bsums));
                    let bs_hi = vpadalq_s16(vdupq_n_s32(0), vld1q_s16(bsums.add(8)));
                    let sumi_min = vaddvq_s32(vmlaq_s32(
                        vmulq_s32(
                            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mins))),
                            bs_lo,
                        ),
                        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mins))),
                        bs_hi,
                    ));

                    // The product: two groups per 32-byte weight run — the low nibbles are
                    // the even group's, the high nibbles the odd one's — each scaled by its
                    // own lane of the unpacked scales and folded into one of two
                    // accumulators so the chain is four deep rather than eight.
                    let qs = q4.add(16);
                    let ys = q8.add(4) as *const i8;
                    let mut sum_lo = vdupq_n_s32(0);
                    let mut sum_hi = vdupq_n_s32(0);
                    q4k_group_pair!($dot, qs, ys, scales_lo, sum_lo, 0);
                    q4k_group_pair!($dot, qs, ys, scales_lo, sum_lo, 1);
                    q4k_group_pair!($dot, qs, ys, scales_hi, sum_hi, 2);
                    q4k_group_pair!($dot, qs, ys, scales_hi, sum_hi, 3);
                    let sumi = vaddvq_s32(vaddq_s32(sum_lo, sum_hi));

                    acc += d * yd * sumi as f32 - dmin * yd * sumi_min as f32;
                }
                acc
            }
        }
    };
}

/// One 32-byte weight run — two adjacent Q4_K groups — against the two 32-element
/// activation runs they pair with, each scaled by its own lane of `$scales` and added to
/// `$acc`. `$pair` is the run's index in the block: run `p` starts at weight byte `32p`,
/// and its two groups take lanes `2(p % 2)` and `2(p % 2) + 1` of the scale set that
/// covers them.
#[cfg(target_arch = "aarch64")]
macro_rules! q4k_group_pair {
    ($dot:path, $qs:ident, $ys:ident, $scales:ident, $acc:ident, $pair:literal) => {{
        const K: usize = 32 * $pair;
        const LO: i32 = 2 * ($pair % 2);
        const HI: i32 = LO + 1;
        let mask = vdupq_n_u8(0x0F);
        let run0 = vld1q_u8($qs.add(K));
        let run1 = vld1q_u8($qs.add(K + 16));
        let even = $dot(
            $dot(
                vdupq_n_s32(0),
                vreinterpretq_s8_u8(vandq_u8(run0, mask)),
                vld1q_s8($ys.add(K * 2)),
            ),
            vreinterpretq_s8_u8(vandq_u8(run1, mask)),
            vld1q_s8($ys.add(K * 2 + 16)),
        );
        let odd = $dot(
            $dot(
                vdupq_n_s32(0),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(run0)),
                vld1q_s8($ys.add(K * 2 + 32)),
            ),
            vreinterpretq_s8_u8(vshrq_n_u8::<4>(run1)),
            vld1q_s8($ys.add(K * 2 + 48)),
        );
        $acc = vmlaq_laneq_s32($acc, even, $scales, LO);
        $acc = vmlaq_laneq_s32($acc, odd, $scales, HI);
    }};
}

#[cfg(target_arch = "aarch64")]
q4k_row_q8!(
    #[target_feature(enable = "dotprod")]
    fn q4k_dot_row_q8_dotprod,
    crate::intrinsics::int8::dot16_into_dotprod
);

#[cfg(target_arch = "aarch64")]
q4k_row_q8!(
    fn q4k_dot_row_q8_wide,
    crate::intrinsics::int8::dot16_into_wide
);

/// Cached `dotprod` (ARMv8.2-A) availability — Cortex-A76 (Raspberry Pi 5) /
/// Apple Silicon / Graviton2+ / most ARM servers have it; Cortex-A72 (Pi 4)
/// does not. Detected once; the result is a hot-loop branch predictor's dream.
///
/// The one feature check for the whole crate: Q4_K's row dot, the six-bit and eight-bit
/// row dots, and the choice between each route's `dotprod` and baseline arms all ask it,
/// and `RLX_Q4K_NO_DOTPROD=1` pins them all to the baseline.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn aarch64_has_dotprod() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    // `RLX_Q4K_NO_DOTPROD=1` forces the baseline path (A/B benchmarking, or a
    // hedge against a mis-detected feature on an exotic core).
    *OK.get_or_init(|| {
        rlx_ir::env::var_os("RLX_Q4K_NO_DOTPROD").is_none()
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

// ---------------------------------------------------------------------------
// Q2_0 int8 dot GEMV (llama.cpp-style): activations quantized once to int8 per
// 128-group, then dotted directly against the packed 2-bit codes — VNNI on
// x86-64, NEON on aarch64, scalar elsewhere. See `intrinsics::vnni` and
// `rlx_gguf::q2_dequant`.
// ---------------------------------------------------------------------------

/// One `Q2_0` weight block (34 B) dotted with one packed int8 activation block.
#[cfg(target_arch = "x86_64")]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    crate::intrinsics::vnni::q2_0_dot_q8_g128(w, a)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    q2_0_dot_q8_block_neon(w, a)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    rlx_gguf::q2_dequant::q2_0_dot_q8_g128(w, a)
}

#[inline]
fn q2_0_dot_row_q8(row: &[u8], x_q8: &[u8], blocks_per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let w = &row[b * Q2_0_BLOCK_BYTES..(b + 1) * Q2_0_BLOCK_BYTES];
        let a = &x_q8[b * Q8_0_G128_BYTES..(b + 1) * Q8_0_G128_BYTES];
        acc += q2_0_dot_q8_block(w, a);
    }
    acc
}

/// `C[n] = x[k] @ W[n,k]^T` for `Q2_0` with `k` a multiple of [`QK2_0`].
///
/// Activations are quantized once to int8 per 128-group; each output row then
/// uses the int8 dot (VNNI/NEON/scalar). Rows are independent — no n-wide
/// partial reduction.
fn q2_0_gemv_parallel(x: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK2_0;
    let row_bytes = blocks_per_row * Q2_0_BLOCK_BYTES;
    debug_assert!(x.len() >= k);
    debug_assert!(w_bytes.len() >= n * row_bytes);

    let mut x_q8 = vec![0u8; blocks_per_row * Q8_0_G128_BYTES];
    rlx_gguf::q2_dequant::quantize_q8_0_g128_row(&x[..k], &mut x_q8);

    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(j, slot)| {
            let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
            *slot = q2_0_dot_row_q8(row, &x_q8, blocks_per_row);
        });
    }
}

/// NEON `Q2_0 × int8` block dot (aarch64). Uses signed weights `(q−1)` so no
/// `xsum` term; widening `vmull`/`vpadal` (no `dotprod` requirement).
#[cfg(target_arch = "aarch64")]
#[inline]
fn q2_0_dot_q8_block_neon(w: &[u8], a: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: fixed-size Q2_0 (34 B) / Q8_0_G128 blocks; NEON baseline on aarch64.
    unsafe {
        let d = half::f16::from_le_bytes([w[0], w[1]]).to_f32();
        let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
        let qs = w.as_ptr().add(2);
        let acts = a.as_ptr().add(4) as *const i8;
        let mask = vdupq_n_u8(0x03);
        let one = vdupq_n_s8(1);
        let mut acc = vdupq_n_s32(0);
        // 128 codes = 32 packed bytes; 16 packed bytes (64 codes) per iter.
        for blk in 0..2 {
            let pk = vld1q_u8(qs.add(blk * 16)); // 16 bytes → 64 codes
            // Code (q−1) at natural position 4i+r for r = 0..3.
            let w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk, mask)), one);
            let w1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 2), mask)), one);
            let w2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 4), mask)), one);
            let w3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 6), mask)), one);
            // vld4 deinterleaves: av.i[j] = acts[4j+i] — matches wi ordering.
            let av = vld4q_s8(acts.add(blk * 64));
            for (wr, ar) in [(w0, av.0), (w1, av.1), (w2, av.2), (w3, av.3)] {
                let lo = vmull_s8(vget_low_s8(wr), vget_low_s8(ar));
                let hi = vmull_s8(vget_high_s8(wr), vget_high_s8(ar));
                acc = vpadalq_s16(acc, lo);
                acc = vpadalq_s16(acc, hi);
            }
        }
        d * dx * vaddvq_s32(acc) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_0_gemv_matches_f32_reference() {
        let k = 256; // 2 groups of 128
        let n = 48;
        let w: Vec<f32> = (0..k * n)
            .map(|i| (((i * 3) % 3) as i32 - 1) as f32 * ((i % 5) as f32 * 0.1 + 0.1))
            .collect();
        let packed = rlx_gguf::q2_dequant::quantize_q2_0(&w).unwrap();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.017).sin() * 2.0).collect();

        // int8 dot GEMV (dispatch entry).
        let mut got = vec![0f32; n];
        gguf_matmul_bt_dispatch(&x, &packed, &mut got, 1, k, n, QuantScheme::GgufQ2_0);

        // Reference: dequant weight to f32, plain dot.
        let wd = rlx_gguf::q2_dequant::dequant_q2_0(&packed, k * n).unwrap();
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[p] * wd[j * k + p];
            }
            // int8 activation quant → small relative tolerance.
            assert!(
                (got[j] - acc).abs() < 2e-2 * (1.0 + acc.abs()),
                "row {j}: {} vs {acc}",
                got[j]
            );
        }
    }

    /// The exact arm (`RLX_Q4K_EXACT_GEMV`) must land far closer to full-f32
    /// cached BLAS than the Q8_K arm does — that is its whole reason to exist.
    /// Asserting it is *tighter than* the Q8_K arm, rather than a fixed epsilon,
    /// keeps the test meaningful if either kernel is retuned.
    ///
    /// `RLX_Q4K_EXACT_GEMV` is scoped to the arm that reads it: this binary runs its
    /// siblings at once, and `cached_blas_matches_fused_q4k_decode` reads the same switch.
    #[test]
    fn exact_q4k_decode_gemv_beats_the_q8k_arm_on_accuracy() {
        use crate::dequant_cache::clear_dequant_cache;
        let k = 512;
        let n = 96;
        // Heavy-tailed weights and activations: Q8_K's per-256-block absmax
        // scale is set by the outlier, so this is where quantizing the
        // activation actually costs something.
        let w: Vec<f32> = (0..k * n)
            .map(|i| {
                let t = (i as f32 * 0.017).sin();
                if i % 137 == 0 { 9.0 * t } else { t }
            })
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k)
            .map(|i| {
                let t = (i as f32 * 0.031).cos();
                if i % 61 == 0 { 12.0 * t } else { t }
            })
            .collect();

        let run = |exact: bool| {
            let _exact = rlx_ir::env::scoped("RLX_Q4K_EXACT_GEMV", if exact { "1" } else { "0" });
            clear_dequant_cache();
            let mut out = vec![0f32; n];
            gguf_matmul_bt(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
            out
        };
        let q8 = run(false);
        let exact = run(true);

        // Reference: dequantize the packed weights to f32 and dot them in f64 —
        // an independent oracle, and the same one in BLAS and no-BLAS builds.
        // Deliberately NOT `gguf_matmul_bt_cached`: it shares its dequant with
        // the exact arm, so it cannot catch a dequant error, and it carries no
        // accuracy claim of its own — which is the whole question here.
        clear_dequant_cache();
        let w_f32 = crate::dequant_cache::gguf_weight_f32(0, &packed, k, n, QuantScheme::GgufQ4K);
        let reference: Vec<f32> = (0..n)
            .map(|j| {
                let row = &w_f32[j * k..(j + 1) * k];
                row.iter()
                    .zip(&x)
                    .map(|(w, a)| f64::from(*w) * f64::from(*a))
                    .sum::<f64>() as f32
            })
            .collect();

        let err = |v: &[f32]| -> f32 {
            v.iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        let (e_exact, e_q8) = (err(&exact), err(&q8));
        assert!(
            e_exact < e_q8,
            "exact arm should be closer to the f64 oracle: exact={e_exact} q8={e_q8}"
        );
        // It shares the weight dequant with the oracle and differs only in
        // summation order, so it should agree to near f32 round-off on the
        // row scale.
        let scale = reference.iter().fold(0f32, |a, b| a.max(b.abs())).max(1.0);
        assert!(
            e_exact < 1e-4 * scale,
            "exact arm drifted from the f64 oracle: {e_exact} (scale {scale})"
        );
    }

    #[test]
    fn cached_blas_matches_fused_q4k_decode() {
        use crate::dequant_cache::clear_dequant_cache;
        clear_dequant_cache();
        let k = 256;
        let n = 64;
        let m = 1;
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.02 * i as f32).collect();
        let mut legacy = vec![0f32; m * n];
        let mut cached = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut legacy, m, k, n, QuantScheme::GgufQ4K);
        gguf_matmul_bt_cached(&x, &packed, &mut cached, m, k, n, QuantScheme::GgufQ4K, 0);
        for i in 0..legacy.len() {
            // Q4_K fused decode quantizes activations to Q8_K (llama.cpp-style);
            // allow small relative error vs full-f32 cached BLAS.
            let scale = cached[i].abs().max(1.0);
            assert!(
                (legacy[i] - cached[i]).abs() < 1e-2 * scale,
                "i={i}: legacy={} cached={}",
                legacy[i],
                cached[i]
            );
        }
    }

    #[test]
    fn q4k_rowmajor_serial_matches_generic_laguna_shape() {
        // Laguna-XS expert gate/up: n=512, k=2048 (both Q4_K aligned).
        let k = 2048;
        let n = 512;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
        let mut fast = vec![0f32; n];
        let mut slow = vec![0f32; n];
        gguf_matmul_bt_serial(&x, &packed, &mut fast, 1, k, n, QuantScheme::GgufQ4K);
        // Reference: same Q8_K activations + scalar Q4×Q8 block dots.
        let bpr = k / QK_K;
        let mut x_q8 = vec![0u8; bpr * rlx_gguf::Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut x_q8);
        let bb = rlx_gguf::Q4K_BLOCK_BYTES;
        let qb = rlx_gguf::Q8K_BLOCK_BYTES;
        for j in 0..n {
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let q4 = &packed[(j * bpr + b) * bb..(j * bpr + b + 1) * bb];
                let q8 = &x_q8[b * qb..(b + 1) * qb];
                acc += rlx_gguf::q4_k_dot_q8_k(q4, q8);
            }
            slow[j] = acc;
        }
        let mut max_err = 0.0f32;
        for i in 0..n {
            max_err = max_err.max((fast[i] - slow[i]).abs());
        }
        assert!(max_err < 1e-3, "max_err={max_err}");
    }

    /// The AVX2 Q4_K × Q8_K block dot is all-integer up to the final scaling,
    /// so it must equal the scalar reference EXACTLY, whatever order it
    /// reduces in. Raw random bytes (not quantizer output) reach every 6-bit
    /// scale/min pattern.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k_avx2_block_dot_is_bit_exact() {
        if !avx2_available() {
            return;
        }
        let mut seed = 11u64;
        let mut byte = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u8
        };
        for _ in 0..200 {
            let mut q4: Vec<u8> = (0..rlx_gguf::Q4K_BLOCK_BYTES).map(|_| byte()).collect();
            q4[..2].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes());
            q4[2..4].copy_from_slice(&half::f16::from_f32(0.0045).to_le_bytes());
            let x: Vec<f32> = (0..QK_K).map(|_| byte() as f32 / 128.0 - 1.0).collect();
            let mut q8 = vec![0u8; rlx_gguf::Q8K_BLOCK_BYTES];
            rlx_gguf::quantize_q8_k_row(&x, &mut q8);
            // SAFETY: AVX2 checked above; full blocks.
            let fast = unsafe { q4k_dot_q8_block_avx2(&q4, &q8) };
            let slow = rlx_gguf::q4_k_dot_q8_k(&q4, &q8);
            assert_eq!(fast.to_bits(), slow.to_bits(), "{fast} vs {slow}");
        }
    }

    #[test]
    #[ignore = "manual kernel timing"]
    fn q4k_gemv_microbench() {
        let k = 2048usize;
        let n = 512usize;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
        let mut out = vec![0f32; n];
        for _ in 0..5 {
            gguf_matmul_bt_serial(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let iters = 40;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gguf_matmul_bt_serial(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            gguf_matmul_bt(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let ms_par = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!("q4k gemv n={n} k={k}: serial={ms:.3}ms  parallel={ms_par:.3}ms");
        assert!(ms < 50.0, "serial unexpectedly slow: {ms}ms");
    }

    /// The decisive decode question: does the int8 Q4_K GEMV (the SDOT-
    /// optimizable path) actually BEAT cached-f32-BLAS (Accelerate = AMX), which
    /// is what `prefer_cached_blas` routes real LLM Q4_K matmuls to? If int8
    /// already wins, an SDOT speedup is a real decode lever; if Accelerate wins,
    /// the framework's routing is already optimal and SDOT is moot.
    ///   cargo test -p rlx-cpu q4k_int8_vs_cached_blas -- --ignored --nocapture
    #[test]
    #[ignore = "manual kernel timing"]
    fn q4k_int8_vs_cached_blas() {
        use crate::dequant_cache::clear_dequant_cache;
        use std::time::Instant;
        let time = |label: &str, f: &mut dyn FnMut()| {
            for _ in 0..3 {
                f();
            }
            let it = 30;
            let t = Instant::now();
            for _ in 0..it {
                f();
            }
            eprintln!(
                "    {label}: {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3 / it as f64
            );
        };
        for (k, n) in [(1024usize, 1024usize), (1024, 3072), (1024, 151936)] {
            let w: Vec<f32> = (0..k * n)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
            let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
            let mut out = vec![0f32; n];
            eprintln!("decode GEMV k={k} n={n}:");
            // int8 Q4_K GEMV (serial int8 Q8_K dots — the path SDOT would speed up).
            time("int8 q4k gemv (serial)", &mut || {
                super::q4k_gemv_rowmajor(&x, &packed, &mut out, k)
            });
            // int8 Q4_K GEMV (parallel, rayon over rows) — the actual decode path
            // when routed to the fused kernel; the real competitor to single-AMX.
            time("int8 q4k gemv (parallel)", &mut || {
                gguf_matmul_bt(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K)
            });
            // cached-f32-BLAS = what the model actually runs (Accelerate/AMX).
            clear_dequant_cache();
            time("cached f32 BLAS (Accel)", &mut || {
                gguf_matmul_bt_cached(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K, 0)
            });
        }
    }

    /// The tiled route against the block-fused kernel it replaced, for every scheme
    /// the model bodies carry: both are f32 over the whole `k`, so they may differ
    /// only in summation order.
    #[test]
    fn the_tiled_route_agrees_with_the_fused_one() {
        let (k, n, m) = (1024usize, 8usize, 2usize);
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        for (scheme, ggml) in [
            (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
            (QuantScheme::GgufQ5K, rlx_gguf::GgmlType::Q5K),
            (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
            (QuantScheme::GgufQ8K, rlx_gguf::GgmlType::Q8K),
            (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0),
        ] {
            let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).sin()).collect();
            let packed = rlx_gguf::quantize(&w, ggml).expect("pack the weight");
            let mut fused = vec![0f32; m * n];
            let mut tiled = vec![0f32; m * n];
            // Scoped, not process-wide: this binary runs the rest of the crate's tests
            // at once, and a sibling reaching the multi-row path would take the fused
            // route while the override is held. `refusal` reads the switch on this
            // thread, before any worker is handed work.
            let _fused_only = rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", "0");
            gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, scheme);
            // A nested guard shadows the `"0"` rather than dropping it: `"1"`, not unset,
            // so a shell holding `RLX_GGUF_TILED_BLAS=0` cannot compare the fused route
            // with itself and pass vacuously.
            let _tiled_only = rlx_ir::env::scoped("RLX_GGUF_TILED_BLAS", "1");
            if let Some(why) = crate::gguf_tiled::refusal(m, k, n, scheme) {
                println!("{scheme:?}: {why} — the fused kernel is the only route here");
                continue;
            }
            gguf_matmul_bt(&x, &packed, &mut tiled, m, k, n, scheme);
            for i in 0..fused.len() {
                // Both sides are f32 sums over the same `k`; the bound is the
                // accumulation's own size, and a route that landed a different
                // product is orders of magnitude past it.
                assert!(
                    (fused[i] - tiled[i]).abs() <= 1e-4 * fused[i].abs().max(1.0),
                    "{scheme:?} i={i}: fused {} vs tiled {}",
                    fused[i],
                    tiled[i]
                );
            }
        }
    }

    #[test]
    fn fused_q1_0_matches_full_dequant() {
        // Custom 1-bit Q1_0 (PrismML Bonsai-27B): 128-element blocks,
        // f16 group scale + 128 sign bits. Exercises both the m=1 and
        // the m>1 accumulation paths of gguf_matmul_bt.
        use rlx_gguf::q1_dequant::QK1_0;
        let k = 256usize; // 2 blocks per row
        let n = 8usize;
        let blocks_per_row = k / QK1_0;

        let mut packed = Vec::new();
        let mut w_ref = vec![0f32; n * k]; // row-major [n, k]
        for row in 0..n {
            for b in 0..blocks_per_row {
                // f16-roundtrip the scale so w_ref == packed-dequant exactly.
                let d =
                    half::f16::from_f32(0.1 + 0.05 * (row * blocks_per_row + b) as f32).to_f32();
                packed.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut bits = [0u8; QK1_0 / 8];
                for j in 0..QK1_0 {
                    let elem = b * QK1_0 + j;
                    let positive = !(row + elem).is_multiple_of(3);
                    if positive {
                        bits[j / 8] |= 1 << (j % 8);
                    }
                    w_ref[row * k + elem] = if positive { d } else { -d };
                }
                packed.extend_from_slice(&bits);
            }
        }
        // Whole-tensor dequant agrees with the hand-built reference.
        let deq = rlx_gguf::q1_dequant::dequant_q1_0(&packed, n * k).unwrap();
        assert_eq!(deq, w_ref);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32 - 0.3).collect();
            let mut reference = vec![0f32; m * n];
            for mi in 0..m {
                for row in 0..n {
                    let mut acc = 0f32;
                    for p in 0..k {
                        acc += x[mi * k + p] * w_ref[row * k + p];
                    }
                    reference[mi * n + row] = acc;
                }
            }
            let mut fused = vec![0f32; m * n];
            gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ1_0);
            for i in 0..reference.len() {
                assert!(
                    (reference[i] - fused[i]).abs() < 1e-3,
                    "m={m} i={i}: ref={} fused={}",
                    reference[i],
                    fused[i]
                );
            }
        }
    }

    #[test]
    fn fused_ptq1_0_matches_full_dequant() {
        // PrismML Ternary Bonsai 2: base-3 ternary, 128-element blocks.
        // Same 128-wide group as Q1_0, so this also covers the "block
        // smaller than QK_K" path through gguf_matmul_bt at m=1 and m>1.
        use rlx_gguf::ptq1_dequant::QK_PTQ1_0;
        let k = 256usize; // 2 blocks per row
        let n = 8usize;

        // Ternary-at-group-128 input round-trips exactly, so the packed
        // bytes and the f32 reference agree bit-for-bit.
        let mut w_ref = vec![0f32; n * k];
        for row in 0..n {
            for b in 0..(k / QK_PTQ1_0) {
                let d = half::f16::from_f32(0.1 + 0.05 * (row * 2 + b) as f32).to_f32();
                for j in 0..QK_PTQ1_0 {
                    let elem = b * QK_PTQ1_0 + j;
                    let t = ((row + elem) % 3) as f32 - 1.0;
                    w_ref[row * k + elem] = t * d;
                }
            }
        }
        let packed = rlx_gguf::ptq1_dequant::quantize_ptq1_0(&w_ref).unwrap();
        assert_eq!(
            rlx_gguf::ptq1_dequant::dequant_ptq1_0(&packed, n * k).unwrap(),
            w_ref
        );

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32 - 0.3).collect();
            let mut reference = vec![0f32; m * n];
            for mi in 0..m {
                for row in 0..n {
                    let mut acc = 0f32;
                    for p in 0..k {
                        acc += x[mi * k + p] * w_ref[row * k + p];
                    }
                    reference[mi * n + row] = acc;
                }
            }
            let mut fused = vec![0f32; m * n];
            gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufPtq1_0);
            for i in 0..reference.len() {
                assert!(
                    (reference[i] - fused[i]).abs() < 1e-3,
                    "m={m} i={i}: ref={} fused={}",
                    reference[i],
                    fused[i]
                );
            }
        }
    }

    #[test]
    fn a_packed_q8k_multiply_matches_a_full_dequant() {
        let k = 256;
        let n = 4;
        let m = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let w_ref = rlx_gguf::dequant_q8_k(&packed, k * n).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let mut fused = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ8K);
        let mut expected = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[r * k + i] * w_ref[c * k + i];
                }
                expected[r * n + c] = acc;
            }
        }
        for i in 0..fused.len() {
            // Relative to the magnitude: the two routes sum in different orders, and
            // the largest difference seen is 1.5e-6 of the result (0.01 at 6745),
            // which is f32 accumulation over 256 terms — while an unpacking error is
            // ~1e-2 relative. 1e-5 separates them with margin at both ends.
            assert!(
                (fused[i] - expected[i]).abs() <= 1e-5 * expected[i].abs().max(1.0),
                "i={i}: {} vs {}",
                fused[i],
                expected[i]
            );
        }
    }

    #[test]
    fn parallel_m1_matches_sequential_q8k() {
        let k = 512;
        let n = 128;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            for _ in 0..(k / QK_K) {
                packed.extend_from_slice(&scale.to_le_bytes());
                for i in 0..QK_K {
                    let q = (i as i32 - 128).clamp(-128, 127) as i8;
                    packed.push(q as u8);
                }
                for _ in 0..(QK_K / 16) {
                    packed.extend_from_slice(&0i16.to_le_bytes());
                }
            }
        }
        let x: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let scheme = QuantScheme::GgufQ8K;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let num_blocks = packed_blocks(k, n, packed.len(), scheme);
        let mut seq = vec![0f32; n];
        let mut par = vec![0f32; n];
        gguf_matmul_bt_m1_sequential(
            &x,
            &packed,
            &mut seq,
            scheme,
            num_blocks,
            block_bytes,
            block_elems,
            k,
        );
        if m1_parallel_takes(num_blocks) {
            gguf_matmul_bt_m1_parallel(
                &x,
                &packed,
                &mut par,
                k,
                n,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
            );
        } else {
            par.copy_from_slice(&seq);
        }
        for i in 0..n {
            let tol = seq[i].abs().max(1.0) * 1e-5 + 1e-3;
            assert!(
                (seq[i] - par[i]).abs() <= tol,
                "parallel mismatch at {i}: {} vs {} (tol {tol})",
                seq[i],
                par[i]
            );
        }
    }

    #[test]
    fn grouped_q8k_matches_per_expert_reference() {
        let k = 256;
        let n = 4;
        let m = 3;
        let num_experts = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..(num_experts * n) {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let expert_idx = vec![0f32, 1.0, 0.0];
        let mut grouped = vec![0f32; m * n];
        gguf_grouped_matmul_bt(
            &x,
            &packed,
            &expert_idx,
            &mut grouped,
            m,
            k,
            n,
            num_experts,
            QuantScheme::GgufQ8K,
        );
        let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            let e = expert_idx[row] as usize;
            let w_ref = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[row * k + i] * w_ref[col * k + i];
                }
                expected[row * n + col] = acc;
            }
        }
        for i in 0..grouped.len() {
            // f32 GEMM (OpenBLAS/Accelerate) block-accumulates in a different
            // order than the naive f32 reference loop, so the absolute error
            // scales with the output magnitude — in the thousands for these
            // inputs. Compare with a relative tolerance; the fixed 1e-2 was
            // too tight and only passed on backends whose accumulation order
            // happened to stay under it (SIMD, Accelerate — not OpenBLAS).
            let tol = 1e-2 + 1e-4 * expected[i].abs();
            assert!(
                (grouped[i] - expected[i]).abs() <= tol,
                "i={i}: {} vs {} (tol {tol})",
                grouped[i],
                expected[i]
            );
        }
    }

    /// Both aarch64 arms of the Q4_K × Q8_K block dot must be BYTE-IDENTICAL to the
    /// scalar reference, not merely close: the whole block is integer arithmetic up to its
    /// final scale, so every re-association in the vectorised form is exact by
    /// construction and a difference here is a defect.
    ///
    /// Raw random bytes rather than quantizer output, so the six-bit scale and minimum
    /// fields reach every pattern — including the ones a real weight would not produce.
    /// The two f16 header scales are pinned: random bytes there can be a NaN, and a NaN
    /// payload is not what this test is about. `d` and `dmin` are distinct so a swapped
    /// pair cannot cancel.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_block_dot_is_bit_exact() {
        let mut seed = 0x1234_5678u32;
        let mut byte = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 24) as u8
        };
        // The dotprod arm is the one the reference machine ships; the widening arm is the
        // fallback an ARMv8.0 host takes and must render the same numbers.
        let mut arms: Vec<(&str, unsafe fn(&[u8], &[u8], usize) -> f32)> =
            vec![("wide", super::q4k_dot_row_q8_wide)];
        if super::aarch64_has_dotprod() {
            arms.insert(0, ("dotprod", super::q4k_dot_row_q8_dotprod));
        }
        let blocks = 3;
        for round in 0..1024 {
            let mut q4: Vec<u8> = (0..blocks * rlx_gguf::Q4K_BLOCK_BYTES)
                .map(|_| byte())
                .collect();
            for b in 0..blocks {
                let at = b * rlx_gguf::Q4K_BLOCK_BYTES;
                q4[at..at + 2].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes());
                q4[at + 2..at + 4].copy_from_slice(&half::f16::from_f32(0.0045).to_le_bytes());
            }
            // The activation side as the route produces it: a real Q8_K quantisation, so
            // the 16-element sums are the block's own and inside their i16 bound.
            let x: Vec<f32> = (0..blocks * QK_K)
                .map(|_| byte() as f32 / 128.0 - 1.0)
                .collect();
            let mut q8 = vec![0u8; blocks * rlx_gguf::Q8K_BLOCK_BYTES];
            rlx_gguf::quantize_q8_k_row(&x, &mut q8);
            let want: f32 = (0..blocks)
                .map(|b| {
                    rlx_gguf::q4_k_dot_q8_k(
                        &q4[b * rlx_gguf::Q4K_BLOCK_BYTES..],
                        &q8[b * rlx_gguf::Q8K_BLOCK_BYTES..],
                    )
                })
                .fold(0.0f32, |acc, v| acc + v);
            for &(label, arm) in &arms {
                // SAFETY: whole blocks, and the arm was selected by the host's features.
                let got = unsafe { arm(&q4, &q8, blocks) };
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "round {round}: {label} arm {got} vs reference {want}"
                );
            }
        }
    }
}

#[cfg(test)]
mod rows_dot_tests {
    use super::*;

    /// The rows-of-blocks kernel against the plain dequantize-then-multiply.
    #[test]
    fn rows_dot_matches_dequant_then_matmul_q8_0() {
        let (m, k, n) = (5, 96, 7);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 9.0)
            .collect();
        let wq = rlx_gguf::quantize::quantize_q8_0(&w).unwrap();
        let scheme = QuantScheme::GgufQ8_0;
        let (bb, be) = (
            scheme.gguf_block_bytes() as usize,
            scheme.gguf_block_size() as usize,
        );
        // reference: widen every block, then x · Wᵀ
        let mut wf = vec![0f32; n * k];
        let mut blk = [0f32; QK_K];
        for (bi, dst) in wf.chunks_exact_mut(be).enumerate() {
            dequant_block(scheme, &wq[bi * bb..(bi + 1) * bb], &mut blk);
            dst.copy_from_slice(&blk[..be]);
        }
        let mut want = vec![0f32; m * n];
        for mi in 0..m {
            for j in 0..n {
                want[mi * n + j] = (0..k).map(|p| x[mi * k + p] * wf[j * k + p]).sum();
            }
        }
        let mut got = vec![0f32; m * n];
        gguf_matmul_bt_rows_dot(&x, &wq, &mut got, m, k, n, scheme, bb, be);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() <= 1e-4 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}

#[cfg(test)]
mod dequant_row_tests {
    use super::*;

    /// [`dequant_row`]'s direct arrangement against the staged one it replaces, on the
    /// two whole-super-block schemes the product's model files carry. The values must be
    /// the same bits rather than the same to a tolerance: the direct form drops a copy,
    /// not a rounding, and takes the four-bit decode's fixed-size arrangement with it.
    #[test]
    fn the_direct_row_decode_is_the_staged_one() {
        for (ggml, scheme) in [
            (rlx_gguf::GgmlType::Q4K, QuantScheme::GgufQ4K),
            (rlx_gguf::GgmlType::Q6K, QuantScheme::GgufQ6K),
        ] {
            let (k, n) = (1024usize, 8usize);
            let w: Vec<f32> = (0..n * k)
                .map(|i| {
                    let v = ((i as f32 + 3.0) * 0.011).sin();
                    if i % 89 == 0 { v * 7.0 } else { v }
                })
                .collect();
            let packed = rlx_gguf::quantize(&w, ggml).expect("encode");
            let (bb, be) = (
                scheme.gguf_block_bytes() as usize,
                scheme.gguf_block_size() as usize,
            );
            let row_bytes = k / be * bb;
            let mut direct = vec![0f32; k];
            let mut staged = vec![0f32; k];
            let mut blk = [0f32; QK_K];
            for j in 0..n {
                let row = &packed[j * row_bytes..(j + 1) * row_bytes];
                dequant_row(scheme, row, &mut direct, bb, be, &mut blk, true);
                dequant_row(scheme, row, &mut staged, bb, be, &mut blk, false);
                for (a, b) in direct.iter().zip(&staged) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{ggml:?} row {j}");
                }
            }
        }
    }
}
