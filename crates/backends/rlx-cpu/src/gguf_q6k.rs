// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The single-row (`m == 1`) six-bit GEMV, dotted as integers instead of being
//! dequantised to f32 and multiplied in f32.
//!
//! Q6_K is 404 MiB of the recognition model — the 151936-way output head and
//! every other layer's `ffn_down` — so every generated token reads it, and the
//! route it took was `gemv_q6k` from `gguf_gemv_specialized` (reached through
//! `gguf_matmul_bt_m1_rows_generic`): each 210-byte block unpacked in scalar code
//! into 256 f32 and then folded with a f32 dot. That ran at 7.8-9.0 GB/s of packed
//! weight; the integer dot below runs at 66-71, which is 7.9x on the head's shape
//! and 8.5x on an `ffn_down` (`tests/gguf_q6k_row_perf.rs`, median of five
//! alternating rounds on the processor).
//!
//! It is reached from the parallel single-row arm of
//! [`crate::gguf_matmul::gguf_matmul_bt`] — a pool of more than one worker, a weight
//! of at least 32 blocks, and a caller that allows parallelism
//! ([`crate::gguf_matmul::gguf_matmul_bt_serial`] never takes it) — so a one-worker
//! or serial host keeps the sequential block loop.
//!
//! This is the same arrangement the neighbouring Q4_K arm in
//! [`crate::gguf_matmul`] uses, and the one llama.cpp uses for both: quantise the
//! activation row once to Q8_K, then dot each weight row's packed bytes in
//! place with an int8 vec-dot. The 6-bit values are unsigned in the block, so the
//! zero point is removed by subtracting 32 from each byte before the dot rather
//! than through the `bsums` correction the AVX2 arm uses — one vector subtract
//! per 16 elements, and no correction term to get wrong.
//!
//! **This route rounds the activation to int8**, which the f32 route it replaces
//! did not. It is the same trade the Q4_K arm already makes on every decode step
//! in this product; what it costs is measured, not asserted, by
//! `tests/gguf_single_row_cache_pressure.rs` (which holds the route to its
//! activation-precision bound rather than to the f32 oracle) and by the stand's
//! answer digest, which is compared across the change.
//!
//! aarch64 only. On x86-64 the same kernel already exists behind the default-off
//! `int8-gemv` feature (`crate::int8_gemv`), which the single-row arm in
//! [`crate::gguf_matmul`] consults before this one, so nothing is duplicated and
//! no other host changes behaviour.

#[cfg(target_arch = "aarch64")]
use rlx_gguf::{Q8K_BLOCK_BYTES, QK_K};
#[cfg(target_arch = "aarch64")]
use rlx_ir::quant::QuantScheme;

/// Whether [`gemv`] is compiled in — the one place that fact is written down.
/// [`crate::gguf_activation::activation`] reads it to decide what a decode's arm
/// multiplies its activation with.
pub(crate) const COMPILED: bool = cfg!(target_arch = "aarch64");

/// `RLX_Q6K_EXACT_GEMV=1` — keep the f32-activation route, the way
/// `RLX_Q4K_EXACT_GEMV` does for the neighbouring width; it is the arm
/// `tests/gguf_q6k_row_perf.rs` compares against. Only [`gemv`] and
/// [`crate::gguf_activation`] consult it, so it is a no-op where that route is not
/// compiled: on x86-64 `int8-gemv` takes the width first.
pub(crate) fn exact_asked() -> bool {
    rlx_ir::env::flag("RLX_Q6K_EXACT_GEMV")
}

/// A Q6_K block's packed size, taken from the scheme that defines it.
#[cfg(target_arch = "aarch64")]
const Q6K_BLOCK_BYTES: usize = QuantScheme::GgufQ6K.gguf_block_bytes() as usize;

/// `out[n] = W[n,k] . x[k]` for Q6_K with `k` a whole number of super-blocks.
/// `false` means "not this route" and the caller keeps its own — including when
/// [`exact_asked`] asks for the f32-activation arm, which is what
/// `tests/gguf_q6k_row_perf.rs` measures this one against.
#[cfg(target_arch = "aarch64")]
pub(crate) fn gemv(x: &[f32], w: &[u8], out: &mut [f32], k: usize) -> bool {
    use rayon::prelude::*;
    if exact_asked() || !k.is_multiple_of(QK_K) || x.len() < k {
        return false;
    }
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * Q6K_BLOCK_BYTES;
    if w.len() < out.len() * row_bytes {
        return false;
    }
    let mut y = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    rlx_gguf::quantize_q8_k_row(&x[..k], &mut y);
    // One indirect call per row rather than one per 16-element group, and the
    // pointer is named for the feature it needs — which is also what the
    // instruction-set scan reads: the `sdot` stays inside `dot_row_dotprod`.
    let dot: unsafe fn(&[u8], &[u8]) -> f32 = if crate::gguf_matmul::aarch64_has_dotprod() {
        dot_row_dotprod
    } else {
        dot_row_vmull
    };
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        // SAFETY: the pointer above was chosen by the runtime feature check, and
        // both arms read whole blocks out of `w` and `y`.
        *slot = unsafe { dot(&w[j * row_bytes..(j + 1) * row_bytes], &y) };
    });
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gemv(_x: &[f32], _w: &[u8], _out: &mut [f32], _k: usize) -> bool {
    false
}

/// One weight row: every super-block's 16 scale groups, each a 16-element
/// integer dot times its own scale.
///
/// The group's 6-bit values are unsigned in the block, so 32 is subtracted from
/// each byte before the dot. The int32 lanes carry a whole block before they are
/// collapsed: a group's sum is bounded by 32·127·16 = 65024, times a scale of at
/// most 127, over sixteen groups is 1.3e8 — well inside `i32`.
#[cfg(target_arch = "aarch64")]
macro_rules! q6k_row {
    ($(#[$attr:meta])* fn $name:ident, $dot:ident) => {
        $(#[$attr])*
        unsafe fn $name(row: &[u8], y: &[u8]) -> f32 {
            use std::arch::aarch64::*;
            // SAFETY: the caller checked both slices hold whole blocks and the
            // loads below stay inside them.
            unsafe {
                let mut total = 0f32;
                for (w, yb) in row
                    .as_chunks::<Q6K_BLOCK_BYTES>()
                    .0
                    .iter()
                    .zip(y.as_chunks::<Q8K_BLOCK_BYTES>().0.iter())
                {
                    let d = f16_at(w, 208) * q8k_d(yb);
                    let qs = &yb[4..4 + QK_K];
                    let mut acc = vdupq_n_s32(0);
                    for h in 0..2 {
                        let ql = &w[h * 64..h * 64 + 64];
                        let qh = &w[128 + h * 32..128 + h * 32 + 32];
                        let sc = &w[192 + h * 8..192 + h * 8 + 8];
                        for c in 0..2 {
                            // `l = 16c .. 16c + 15` is one 16-value group per
                            // 4-value column, and `c` is that group's scale index.
                            let l = c * 16;
                            let qla = vld1q_u8(ql[l..].as_ptr());
                            let qlb = vld1q_u8(ql[l + 32..].as_ptr());
                            let qhv = vld1q_u8(qh[l..].as_ptr());
                            let lo_a = vandq_u8(qla, vdupq_n_u8(0x0F));
                            let lo_b = vandq_u8(qlb, vdupq_n_u8(0x0F));
                            let hi_a = vshrq_n_u8::<4>(qla);
                            let hi_b = vshrq_n_u8::<4>(qlb);
                            let groups = [
                                (
                                    vorrq_u8(
                                        lo_a,
                                        vshlq_n_u8::<4>(vandq_u8(qhv, vdupq_n_u8(0x03))),
                                    ),
                                    sc[c],
                                    0,
                                ),
                                (
                                    vorrq_u8(
                                        lo_b,
                                        vshlq_n_u8::<2>(vandq_u8(qhv, vdupq_n_u8(0x0C))),
                                    ),
                                    sc[c + 2],
                                    32,
                                ),
                                (
                                    vorrq_u8(hi_a, vandq_u8(qhv, vdupq_n_u8(0x30))),
                                    sc[c + 4],
                                    64,
                                ),
                                (
                                    vorrq_u8(
                                        hi_b,
                                        vshrq_n_u8::<2>(vandq_u8(qhv, vdupq_n_u8(0xC0))),
                                    ),
                                    sc[c + 6],
                                    96,
                                ),
                            ];
                            for (q, scale, off) in groups {
                                let yv = vld1q_s8(qs[h * 128 + l + off..].as_ptr().cast::<i8>());
                                let qz =
                                    vsubq_s8(vreinterpretq_s8_u8(q), vdupq_n_s8(32));
                                acc = vmlaq_s32(
                                    acc,
                                    $dot(qz, yv),
                                    vdupq_n_s32(i32::from(scale as i8)),
                                );
                            }
                        }
                    }
                    total += d * vaddvq_s32(acc) as f32;
                }
                total
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
q6k_row!(#[target_feature(enable = "dotprod")] fn dot_row_dotprod, dot_prod);

#[cfg(target_arch = "aarch64")]
q6k_row!(fn dot_row_vmull, dot_wide);

/// 16 signed products into four int32 lanes.
///
/// `SDOT` is emitted directly rather than through `vdotq_s32`: the intrinsic only
/// reached stable in 1.98 and this crate's floor is 1.89, the same reason the
/// Q4_K arm in [`crate::gguf_matmul`] spells it this way. The caller checked the
/// `dotprod` feature at run time.
///
/// Worth 1.33x the widening form below at the head's width — 23.6 against 17.7 GB/s
/// of packed weight per thread on a 4096-row weight (k = 2048, n = 4096), a shape
/// that is arithmetic-bound rather than stream-bound, which is where the dot wins.
/// `tests`' `q6k_dot_width` is the harness.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_prod(
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
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

/// The same 16 products without `dotprod`: two widening multiplies and two
/// pairwise adds.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_wide(
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b)));
        vpadalq_s16(acc, vmull_s8(vget_high_s8(a), vget_high_s8(b)))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f16_at(block: &[u8], off: usize) -> f32 {
    half::f16::from_le_bytes([block[off], block[off + 1]]).to_f32()
}

/// A Q8_K block's activation scale.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn q8k_d(block: &[u8]) -> f32 {
    f32::from_le_bytes([block[0], block[1], block[2], block[3]])
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// One packed Q6_K weight of `n` rows, heavy-tailed so a wrong scale index or
    /// nibble shows up as a large error rather than a rounding difference.
    fn packed_q6k(k: usize, n: usize) -> Vec<u8> {
        let w: Vec<f32> = (0..k * n)
            .map(|i| {
                let s = (i as f32 * 0.017).sin();
                if i % 137 == 0 { 9.0 * s } else { s }
            })
            .collect();
        rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q6K).expect("pack Q6_K")
    }

    fn activation(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let s = (i as f32 * 0.031).cos();
                if i % 61 == 0 { 12.0 * s } else { s }
            })
            .collect()
    }

    fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    /// The integer dot must be exactly the dot of the *quantised* activation: every
    /// nibble, high-bit and scale index in the packed block has one place it can
    /// go, and a single wrong one moves this far past 1e-5. Then the same call with
    /// the exact switch on, which must hand the route back to the f32 kernel.
    ///
    /// One test, because the two arms have to be observed in one process; each arm
    /// scopes its own value of the switch on this thread, so neither is visible to a
    /// sibling test.
    #[test]
    fn the_integer_dot_is_exact_for_the_activation_it_is_given() {
        // Mask the ambient knob: a shell exporting it would hand the route back to
        // the f32 kernel and fail the gate below for a reason that is not the dot. Held
        // to the end of the test; the exact arm below shadows it.
        let _exact_masked = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "0");
        let (k, n) = (1024usize, 32usize);
        let w = packed_q6k(k, n);
        let x = activation(k);
        let mut got = vec![0f32; n];
        assert!(gemv(&x, &w, &mut got, k), "the route must take this shape");

        let mut y = vec![0u8; k / QK_K * Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut y);
        let row_bytes = k / QK_K * Q6K_BLOCK_BYTES;
        let mut want = vec![0f32; n];
        for j in 0..n {
            let mut acc = 0f64;
            for b in 0..k / QK_K {
                let mut wb = [0f32; QK_K];
                let mut xb = [0f32; QK_K];
                rlx_gguf::dequant_q6_k_block(&w[j * row_bytes + b * Q6K_BLOCK_BYTES..], &mut wb);
                rlx_gguf::dequant_q8_k_block(&y[b * Q8K_BLOCK_BYTES..], &mut xb);
                for t in 0..QK_K {
                    acc += f64::from(wb[t]) * f64::from(xb[t]);
                }
            }
            want[j] = acc as f32;
        }
        let err = max_rel_err(&got, &want);
        assert!(
            err < 1e-5,
            "max rel err {err} against the quantised activation"
        );

        // A nested guard shadows the masked one for the rest of the test: dropping the
        // outer one would truncate the whole thread-local stack.
        let _exact_on = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "1");
        let took = gemv(&x, &w, &mut got, k);
        assert!(
            !took,
            "the exact arm must leave the route to the f32 kernel"
        );
    }

    /// The widening fallback must agree with the `dotprod` arm bit for bit: both
    /// are integer dots of the same bytes, so anything else is a lane or a scale
    /// in the wrong place. Skips where the machine has no `dotprod` (there the
    /// fallback is the only arm and the test above already covers it).
    #[test]
    fn the_widening_arm_agrees_with_the_dotprod_arm() {
        if !crate::gguf_matmul::aarch64_has_dotprod() {
            return;
        }
        let (k, n) = (512usize, 16usize);
        let w = packed_q6k(k, n);
        let x = activation(k);
        let mut y = vec![0u8; k / QK_K * Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut y);
        let row_bytes = k / QK_K * Q6K_BLOCK_BYTES;
        for j in 0..n {
            let row = &w[j * row_bytes..(j + 1) * row_bytes];
            let wide = unsafe { dot_row_vmull(row, &y) };
            let narrow = unsafe { dot_row_dotprod(row, &y) };
            assert_eq!(wide, narrow, "row {j}: {wide} vs {narrow}");
        }
    }

    /// The `dotprod` dot against the widening one, at the head's width on a weight
    /// this harness can pack (k = 2048, n = 4096 — the shipped head is 151936 rows of
    /// the same width), alternating the arms within every round so a drifting machine
    /// cannot favour one; the figure is the median of the rounds. `#[ignore]`d: it
    /// prints, and it needs a quiet machine to mean anything.
    ///
    /// ```text
    /// cargo test -p mahbot-rlx-cpu --release --lib -- --ignored q6k_dot_width
    /// ```
    #[test]
    #[ignore = "manual kernel timing"]
    fn q6k_dot_width() {
        const ROUNDS: usize = 5;
        let (k, n) = (2048usize, 4096usize);
        let w = packed_q6k(k, n);
        let x = activation(k);
        let mut y = vec![0u8; k / QK_K * Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut y);
        let row_bytes = k / QK_K * Q6K_BLOCK_BYTES;
        let mut out = vec![0f32; n];
        let arms = [
            ("dotprod", dot_row_dotprod as unsafe fn(&[u8], &[u8]) -> f32),
            ("vmull", dot_row_vmull as unsafe fn(&[u8], &[u8]) -> f32),
        ];
        let mut times = [(0f64, 0f64); ROUNDS];
        for (round, row) in times.iter_mut().enumerate() {
            for arm in if round % 2 == 0 { [0usize, 1] } else { [1, 0] } {
                let dot = arms[arm].1;
                let t = std::time::Instant::now();
                for (j, slot) in out.iter_mut().enumerate() {
                    *slot = unsafe { dot(&w[j * row_bytes..(j + 1) * row_bytes], &y) };
                }
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if arm == 0 { row.0 = ms } else { row.1 = ms }
            }
        }
        for (arm, (label, _)) in arms.iter().enumerate() {
            let ms = median(times.iter().map(|r| if arm == 0 { r.0 } else { r.1 }));
            eprintln!(
                "    {label:<8} {ms:7.3} ms over {n} rows  {:.1} GB/s",
                (n * row_bytes) as f64 / ms / 1e6
            );
        }
    }

    /// The median of the rounds' timings — the paired-reading figure. The harness
    /// modules share theirs through `tests/common`; a lib unit test cannot reach it.
    fn median(values: impl IntoIterator<Item = f64>) -> f64 {
        let mut v: Vec<f64> = values.into_iter().collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
}
