// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The skeleton the aarch64 single-row integer routes share: one Q8_K quantisation of
//! the activation row, then every packed weight row dotted in place by that route's own
//! `dot`, across the pool. `gguf_q6k` and `gguf_q8_0` differ only in their block layout
//! and in that function.
//!
//! `dot` is one indirect call per row rather than one per 16-element group, and it is
//! named for the feature it needs — which is also what the instruction-set scan reads:
//! the `sdot` stays inside the arm named after it.
//!
//! Both routes are reached from the parallel single-row arm of
//! [`crate::gguf_matmul::gguf_matmul_bt`]; the gate is
//! [`crate::gguf_matmul::m1_parallel_takes`]'s.

#[cfg(target_arch = "aarch64")]
use rlx_gguf::{Q8K_BLOCK_BYTES, QK_K};

/// Whether a single-row integer route is compiled in — the one place that fact is written
/// down. [`crate::gguf_activation::activation`] reads it to decide what a decode's arm
/// multiplies its activation with.
pub(crate) const COMPILED: bool = cfg!(target_arch = "aarch64");

/// A packed block's f16 scale.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn f16_at(block: &[u8], off: usize) -> f32 {
    half::f16::from_le_bytes([block[off], block[off + 1]]).to_f32()
}

/// A Q8_K block's activation scale.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn q8k_d(block: &[u8]) -> f32 {
    f32::from_le_bytes([block[0], block[1], block[2], block[3]])
}

/// `out[n] = W[n,k] . x[k]` for a packed weight of `row_bytes` per row, `k` a whole
/// number of Q8_K super-blocks: `dot` reduces one row against the whole quantised
/// activation. `false` means "not this route" and the caller keeps its own.
#[cfg(target_arch = "aarch64")]
pub(crate) fn gemv(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    k: usize,
    row_bytes: usize,
    dot: unsafe fn(&[u8], &[u8]) -> f32,
) -> bool {
    use rayon::prelude::*;
    if x.len() < k || w.len() < out.len() * row_bytes {
        return false;
    }
    let mut y = vec![0u8; k / QK_K * Q8K_BLOCK_BYTES];
    rlx_gguf::quantize_q8_k_row(&x[..k], &mut y);
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        // SAFETY: `dot` was chosen by the runtime feature check, and both arms read
        // whole blocks out of `w` and `y`.
        *slot = unsafe { dot(&w[j * row_bytes..(j + 1) * row_bytes], &y) };
    });
    true
}

/// The fixtures both single-row routes' tests build: one packed weight, one activation,
/// one error metric and one reference dot, so a Q6_K test and a Q8_0 test cannot drift
/// apart.
///
/// It repeats `tests/common/mod.rs`'s weight and activation — a unit test cannot reach an
/// integration test's module — reduced to the single tensor these pack (no `seed`); a
/// change to one belongs in both.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) mod fixture {
    use rlx_gguf::{GgmlType, Q8K_BLOCK_BYTES, QK_K};

    /// One packed weight of `n` rows, heavy-tailed so a wrong scale, index or lane shows
    /// up as a large error rather than a rounding difference.
    pub(crate) fn packed(ggml: GgmlType, k: usize, n: usize) -> Vec<u8> {
        let w: Vec<f32> = (0..k * n)
            .map(|i| {
                let s = (i as f32 * 0.017).sin();
                if i % 137 == 0 { 9.0 * s } else { s }
            })
            .collect();
        rlx_gguf::quantize(&w, ggml).expect("pack the weight")
    }

    pub(crate) fn activation(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let s = (i as f32 * 0.031).cos();
                if i % 61 == 0 { 12.0 * s } else { s }
            })
            .collect()
    }

    pub(crate) fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    /// One packed weight row dotted against a Q8_K activation in f64: the oracle a route's
    /// test holds its kernel to, exact by construction so a lane, scale or index in the
    /// wrong place is the only thing that can move it.
    pub(crate) fn reference_dot(ggml: GgmlType, w_row: &[u8], y: &[u8], k: usize) -> f32 {
        let w = rlx_gguf::dequant_typed(ggml, w_row, k, "reference row").expect("dequant row");
        let mut acc = 0f64;
        for (s, yb) in y.as_chunks::<Q8K_BLOCK_BYTES>().0.iter().enumerate() {
            let mut xb = [0f32; QK_K];
            rlx_gguf::dequant_q8_k_block(yb, &mut xb);
            for (t, x) in xb.iter().enumerate() {
                acc += f64::from(w[s * QK_K + t]) * f64::from(*x);
            }
        }
        acc as f32
    }
}
