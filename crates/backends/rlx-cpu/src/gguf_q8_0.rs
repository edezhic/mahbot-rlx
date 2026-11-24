// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The single-row (`m == 1`) eight-bit GEMV, dotted as integers instead of being
//! dequantised to f32 and multiplied in f32.
//!
//! Q8_0 is the synthesis code predictor: one 16-level frame step emits 351
//! packed multiplies (348 dispatched — three of the emitted sites are dead), 1.31 G
//! multiply-accumulates, and streams 1.39 GB of packed weight out of a set of about
//! 82 MiB that is therefore read sixteen times per frame. Every one of them is `m == 1`,
//! and the route they took was `gguf_matmul_bt_m1_rows_generic` — each 34-byte block
//! unpacked in scalar code into 32 f32 and then folded with a f32 dot. That streams at
//! 5.6-6.2 GB/s of packed weight and barely moves with the machine's load; the integer dot
//! below is the bandwidth-bound arm, so its absolute rate is the quiet session's.
//! `tests/gguf_q8_0_row_perf.rs` measures the ratio, as the median of five alternating
//! rounds: 12.8-18.4x on each of the five shapes and 15.8x over a whole frame step. The
//! best case behind it is 77-114 GB/s of packed weight, 235 ms down to 15; a session
//! sharing the machine's memory bandwidth reads 26-69 GB/s and 7-13x a shape (309 ms down
//! to 26 and 304 down to 27 across the loaded sessions measured since). The ratio is the
//! claim; the absolute figure is the best case.
//!
//! Unlike the neighbouring widths, [`crate::gguf_matmul::gguf_matmul_bt_dispatch_at`]
//! takes this arm whatever the dequant cache holds, for exactly the calls [`takes`]
//! names: it rounds the activation, so a route that moved with the cache would answer
//! one call differently from the next in the same process.
//!
//! This is the arrangement the neighbouring Q4_K and Q6_K arms use, and llama.cpp's for
//! the K-quants: quantise the activation row once to Q8_K, then dot each weight row's
//! packed bytes in place with an int8 vec-dot. llama.cpp pairs Q8_0 weights with a Q8_0
//! activation instead — 32-element blocks, a scale each; here the shared skeleton's Q8_K
//! form is reused, so eight Q8_0 blocks share one super-block and its scale, and each
//! block's own f16 scale is folded in with it.
//!
//! **This route rounds the activation to int8**, which the f32 route it replaces did
//! not. It is the same trade the Q4_K and Q6_K arms already make; what it costs is
//! measured, not asserted, by `tests/gguf_single_row_cache_pressure.rs`, which holds
//! the route to its Q8_K-activation bound rather than to the f32 oracle. It moves the
//! waveform — the render's frames and digest both move, so the harness that prints them
//! prints a different pair — and the words are the witness: the product's own round trip
//! still hears the request back (`crates/mahbot-inference/tests/processor_checks.rs`,
//! `the_processor_renders_the_request_and_hears_it_back`).
//!
//! aarch64 only. On x86-64 the same width already has a kernel behind the default-off
//! `int8-gemv` feature ([`crate::int8_gemv`]), which the single-row arm in
//! [`crate::gguf_matmul`] consults first, so nothing is duplicated and no other host
//! changes behaviour.

#[cfg(target_arch = "aarch64")]
use crate::gguf_m1::{f16_at, q8k_d};
#[cfg(target_arch = "aarch64")]
use rlx_gguf::{Q8K_BLOCK_BYTES, QK_K};
#[cfg(target_arch = "aarch64")]
use rlx_ir::quant::QuantScheme;

/// `RLX_Q8_0_EXACT_GEMV=1` — keep the f32-activation route, the way
/// `RLX_Q4K_EXACT_GEMV` and `RLX_Q6K_EXACT_GEMV` do for the neighbouring widths; it is
/// the arm `tests/gguf_q8_0_row_perf.rs` compares against. [`gemv`], [`takes`] and
/// [`crate::gguf_activation`] consult it, so it is a no-op where that route is not
/// compiled: on x86-64 `int8-gemv` takes the width first.
pub(crate) fn exact_asked() -> bool {
    rlx_ir::env::flag("RLX_Q8_0_EXACT_GEMV")
}

/// A Q8_0 block's packed size and element count, taken from the scheme that defines
/// them (34 bytes: an f16 scale, then 32 signed bytes).
#[cfg(target_arch = "aarch64")]
const Q8_0_BLOCK_BYTES: usize = QuantScheme::GgufQ8_0.gguf_block_bytes() as usize;
#[cfg(target_arch = "aarch64")]
const QK8_0: usize = QuantScheme::GgufQ8_0.gguf_block_size() as usize;
/// The packed bytes of one weight row of `k` elements — the stride
/// [`crate::gguf_m1::gemv`] walks and the length [`takes`] requires, written once.
#[cfg(target_arch = "aarch64")]
const fn row_bytes(k: usize) -> usize {
    k / QK8_0 * Q8_0_BLOCK_BYTES
}
/// The weight bytes one Q8_K super-block's activation covers: eight Q8_0 blocks.
#[cfg(target_arch = "aarch64")]
const Q8_0_SUPER_BYTES: usize = row_bytes(QK_K);

/// The gates that ask for the f32 route instead of this one, or that the activation
/// quantiser cannot meet: [`exact_asked`], reproducible mode, and a `k` that is not a
/// whole number of Q8_K super-blocks. [`takes`] and [`gemv`] both apply them, so they are
/// written once.
#[cfg(target_arch = "aarch64")]
#[inline]
fn f32_route_asked(k: usize) -> bool {
    exact_asked() || rlx_ir::reproducible::reproducible() || !k.is_multiple_of(QK_K)
}

/// Whether a single-row Q8_0 multiply of this shape reaches [`gemv`] on this host — the
/// gates that decide whether the dispatch may pre-empt the dequant cache for it, in one
/// place, so it cannot bypass the cache for a call the arm then refuses (a refused call
/// would fall to the row-parallel generic instead of the cached f32 route it had) and a
/// harness's oracle is the activation the call actually multiplies with.
///
/// The gates are [`f32_route_asked`], whole rows of weight — which makes the block count
/// the shape's own — and the parallel single-row arm's own
/// ([`crate::gguf_matmul::m1_parallel_takes`]). [`gemv`] re-checks the first of those
/// because it is reachable directly.
#[cfg(target_arch = "aarch64")]
pub(crate) fn takes(k: usize, n: usize, w_len: usize) -> bool {
    !f32_route_asked(k)
        && w_len >= n * row_bytes(k)
        && crate::gguf_matmul::m1_parallel_takes(crate::gguf_matmul::packed_blocks(
            k,
            n,
            w_len,
            QuantScheme::GgufQ8_0,
        ))
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn takes(_k: usize, _n: usize, _w_len: usize) -> bool {
    false
}

/// `out[n] = W[n,k] . x[k]` for Q8_0 with `k` a whole number of Q8_K super-blocks (the
/// activation quantiser's own requirement, and the row length every predictor multiply
/// has). `false` means "not this route" and the caller keeps its own — including when
/// [`f32_route_asked`] asks for it, which is what `tests/gguf_q8_0_row_perf.rs` measures
/// this one against.
///
/// The gate is checked here as well as in [`takes`]: `takes` keeps the dispatch from
/// pre-empting the cache for a call this would refuse, and this keeps the route off where
/// there is no cached slab to fall back to (a portable or BLAS-less build).
#[cfg(target_arch = "aarch64")]
pub(crate) fn gemv(x: &[f32], w: &[u8], out: &mut [f32], k: usize) -> bool {
    if f32_route_asked(k) {
        return false;
    }
    let dot = if crate::gguf_matmul::aarch64_has_dotprod() {
        dot_row_dotprod
    } else {
        dot_row_vmull
    };
    crate::gguf_m1::gemv(x, w, out, k, row_bytes(k), dot)
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemv(_x: &[f32], _w: &[u8], _out: &mut [f32], _k: usize) -> bool {
    false
}

/// One weight row: each 32-element block's integer dot times its own f16 scale times
/// the scale of the Q8_K super-block that holds it.
///
/// The four int32 lanes carry the row's whole 32-element dot before they are widened,
/// and the f32 accumulator keeps its four lanes independent across blocks: a block's
/// sum is bounded by 127·128·32 = 5.2e5, and the scales are applied in f32, so nothing
/// overflows and nothing needs a correction term.
#[cfg(target_arch = "aarch64")]
macro_rules! q8_0_row {
    ($(#[$attr:meta])* fn $name:ident, $dot:path) => {
        $(#[$attr])*
        unsafe fn $name(row: &[u8], y: &[u8]) -> f32 {
            use std::arch::aarch64::*;
            // SAFETY: the caller checked both slices hold whole blocks and the loads
            // below stay inside them.
            unsafe {
                let mut total = vdupq_n_f32(0.0);
                for (w, yb) in row
                    .as_chunks::<Q8_0_SUPER_BYTES>()
                    .0
                    .iter()
                    .zip(y.as_chunks::<Q8K_BLOCK_BYTES>().0.iter())
                {
                    // One super-block's activation scale covers the eight blocks below.
                    let da = q8k_d(yb);
                    let qs = &yb[4..4 + QK_K];
                    for (b, wb) in w.as_chunks::<Q8_0_BLOCK_BYTES>().0.iter().enumerate() {
                        let q = &wb[2..2 + QK8_0];
                        let a = &qs[b * QK8_0..(b + 1) * QK8_0];
                        let lo = $dot(
                            vld1q_s8(q.as_ptr().cast::<i8>()),
                            vld1q_s8(a.as_ptr().cast::<i8>()),
                        );
                        let hi = $dot(
                            vld1q_s8(q[16..].as_ptr().cast::<i8>()),
                            vld1q_s8(a[16..].as_ptr().cast::<i8>()),
                        );
                        total = vfmaq_f32(
                            total,
                            vcvtq_f32_s32(vaddq_s32(lo, hi)),
                            vdupq_n_f32(f16_at(wb, 0) * da),
                        );
                    }
                }
                vaddvq_f32(total)
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
q8_0_row!(
    #[target_feature(enable = "dotprod")]
    fn dot_row_dotprod,
    crate::intrinsics::int8::dot16_dotprod
);

#[cfg(target_arch = "aarch64")]
q8_0_row!(fn dot_row_vmull, crate::intrinsics::int8::dot16_wide);

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::gguf_m1::fixture::{activation, max_rel_err, packed, reference_dot};

    /// One packed Q8_0 weight of `n` rows.
    fn packed_q8_0(k: usize, n: usize) -> Vec<u8> {
        packed(rlx_gguf::GgmlType::Q8_0, k, n)
    }

    /// The integer dot must be exactly the dot of the *quantised* activation: every
    /// byte, scale and lane in the packed block has one place it can go, and a single
    /// wrong one moves this far past 1e-5. Then the same call with the exact switch
    /// on, which must hand the route back to the f32 kernel.
    ///
    /// One test, because the two arms have to be observed in one process; each arm
    /// scopes its own value of the switch on this thread, so neither is visible to a
    /// sibling test.
    #[test]
    fn the_integer_dot_is_exact_for_the_activation_it_is_given() {
        // Mask the ambient knobs: a shell exporting either would hand the route back to
        // the f32 kernel and fail the gate below for a reason that is not the dot. Held
        // to the end of the test; the exact arm below shadows the mask.
        let _reproducible = rlx_ir::env::scoped("RLX_REPRODUCIBLE", "0");
        let _exact_masked = rlx_ir::env::scoped("RLX_Q8_0_EXACT_GEMV", "0");
        let (k, n) = (1024usize, 32usize);
        let w = packed_q8_0(k, n);
        let x = activation(k);
        let mut got = vec![0f32; n];
        assert!(gemv(&x, &w, &mut got, k), "the route must take this shape");

        let mut y = vec![0u8; k / QK_K * Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut y);
        let stride = row_bytes(k);
        let want: Vec<f32> = (0..n)
            .map(|j| {
                reference_dot(
                    rlx_gguf::GgmlType::Q8_0,
                    &w[j * stride..(j + 1) * stride],
                    &y,
                    k,
                )
            })
            .collect();
        let err = max_rel_err(&got, &want);
        assert!(
            err < 1e-5,
            "max rel err {err} against the quantised activation"
        );

        // A nested guard shadows the masked one for the rest of the test: dropping the
        // outer one would truncate the whole thread-local stack.
        let _exact_on = rlx_ir::env::scoped("RLX_Q8_0_EXACT_GEMV", "1");
        let took = gemv(&x, &w, &mut got, k);
        assert!(
            !took,
            "the exact arm must leave the route to the f32 kernel"
        );
    }
}
