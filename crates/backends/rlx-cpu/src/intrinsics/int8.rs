// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! 16 signed int8 products into four int32 lanes — the primitive every packed
//! single-row integer arm (`gguf_q6k`, `gguf_q8_0`) accumulates with.
//!
//! Two spellings of the same arithmetic: `dot16_dotprod` is one `SDOT`, `dot16_wide` two
//! widening multiplies, two pairwise accumulates and one pairwise add. The second is
//! written to land in the same four lanes — four consecutive products each — so the two
//! are interchangeable to the last bit whatever fold the arm applies afterwards, and a
//! host without the feature renders the same numbers as one with it. Which one runs is a
//! run-time feature probe (`gguf_matmul::aarch64_has_dotprod`) rather than a build flag.
//!
//! `SDOT` is written as `asm!` rather than through `vdotq_s32`: the intrinsic only
//! reached stable in 1.98 and this crate's floor is 1.89, the same reason the Q4_K arm
//! in `gguf_matmul` spells its own this way.
//!
//! Worth 1.43x the widening form at a Q6_K head's width — 23.8 against 16.6 GB/s of
//! packed weight per thread on a 4096-row weight (k = 2048, n = 4096), a shape that is
//! arithmetic-bound rather than stream-bound, which is where the dot wins. The extra
//! `vpaddq` that gives the widening form `SDOT`'s grouping costs it ~7% there; at pool
//! width the route does not see that (a forced-widening frame step reads 18-22 ms against
//! 15-18 with the dot). `gguf_q6k`'s `q6k_dot_width` is the harness.

use std::arch::aarch64::*;

/// 16 signed products into four int32 lanes, as one `SDOT`. The caller checked the
/// `dotprod` feature at run time.
#[target_feature(enable = "dotprod")]
#[inline]
pub(crate) unsafe fn dot16_dotprod(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = vdupq_n_s32(0);
    // SAFETY: `SDOT` is what the `dotprod` target feature above gates; both operands
    // are whole 16-byte vectors the caller loaded.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(nomem, nostack, preserves_flags),
        );
    }
    acc
}

/// The same 16 products without `dotprod`, grouped exactly as `SDOT` groups them — two
/// pairwise accumulates per half, then one pairwise add across the halves — so a host
/// without the feature multiplies to the same numbers, lane for lane.
#[inline(always)]
pub(crate) unsafe fn dot16_wide(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: baseline NEON over whole 16-byte vectors the caller loaded.
    unsafe {
        let lo = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(a), vget_low_s8(b)));
        let hi = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_high_s8(a), vget_high_s8(b)));
        vpaddq_s32(lo, hi)
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// The two forms must be the same four lanes, not merely the same total: an arm that
    /// folds the lanes in f32 (`gguf_q8_0`) sees them individually, so a grouping that
    /// differed would make its numbers depend on whether the host has the instruction.
    /// This is what lets one oracle cover both arms exactly.
    #[test]
    fn the_two_forms_agree_lane_for_lane() {
        if !crate::gguf_matmul::aarch64_has_dotprod() {
            return;
        }
        // Signed values over the whole range, including the pairs that overflow a i16
        // widening multiply's neighbours.
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as i8
        };
        for _ in 0..256 {
            let a: [i8; 16] = std::array::from_fn(|_| next());
            let b: [i8; 16] = std::array::from_fn(|_| next());
            // SAFETY: both are whole 16-byte vectors and the machine has `dotprod`.
            let (wide, narrow) = unsafe {
                let mut wide = [0i32; 4];
                let mut narrow = [0i32; 4];
                vst1q_s32(
                    wide.as_mut_ptr(),
                    dot16_wide(vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr())),
                );
                vst1q_s32(
                    narrow.as_mut_ptr(),
                    dot16_dotprod(vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr())),
                );
                (wide, narrow)
            };
            let want: i32 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| i32::from(*x) * i32::from(*y))
                .sum();
            assert_eq!(wide, narrow, "{a:?} · {b:?}");
            assert_eq!(wide.iter().sum::<i32>(), want, "{a:?} · {b:?}");
        }
    }
}
