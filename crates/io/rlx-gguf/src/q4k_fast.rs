// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The four-bit super-block widened into an f32 tile, arranged for the one caller that
//! pays for the decode by the weight: `rlx-cpu`'s `dequant_row`, which both of that
//! crate's row-widening routes run over every packed weight they multiply against several
//! activation rows.
//!
//! [`crate::dequant_q4_k_block`] takes a `&[u8]` and an `&mut [f32; QK_K]`, so each of
//! the 128 `qs` bytes it reads pays a range check, and the sub-block scale and minimum
//! come out of `get_scale_min_k4` — a call the compiler will not inline across the loop
//! bound. (Its 256 stores are bounded by the array's own length, so those it can
//! elide.) [`block`] is the same arithmetic over fixed-size slices, with the unpacking
//! inlined and each 32-element sub-block written four lanes at a time. Every value is
//! the reference's, bit for bit: each element is the same `scale * q - min` in the same
//! order, with no fused multiply-add and no reassociation, so the four-lane store
//! cannot move one. The arrangement is a cost, not an arithmetic. `RLX_GGUF_ROW_DECODE`
//! prices the two against each other: `rlx-cpu`'s `direct_decode()` reads it once per
//! call and passes the answer into `dequant_row`.
//!
//! The readings, paired on one machine, over the product's own weights:
//!
//! - The decode alone, on the recognition body's `blk.9.ffn_up.weight` (2048 x 6144,
//!   49k blocks): the reference costs 0.331 ns per weight and [`block`] 0.122, so 2.72x.
//! - Every value, over both model files: 5.9M packed four-bit blocks, 1.51G weights,
//!   against [`crate::dequant_q4_k_block`] — zero differing.
//! - One prompt prefill pass (79 rows) over the recognition pair's 1.409G four- and
//!   six-bit linear weights, the route with the switch on and off: 349.1 ms against
//!   268.0 ms, a marginal 81 ms — 79 to 86 over five takes. The four-bit 1.204G of that
//!   weight is 79 ms of it; the six-bit arm's own margin is a few percent of its route,
//!   inside the run-to-run spread.
//! - The row, as the record stood when this landed: the 318-second recognition call runs 51
//!   prefill passes — its own phase line's rungs — so 4.0-4.4 s of its 136.43 s warm column
//!   (3.0-3.2 %). The paired whole-call take on the processor could not hold that: three
//!   takes put its prefill phase's margin at 8.4, 4.4 and 0.6 s, because the arms' spreads
//!   overlap on a box carrying a load of 8-11. The row figure is therefore the per-pass
//!   figure times the row's own pass count, and the route take above is what holds the
//!   change.
//! - The single-worker arm (`rlx-cpu`'s `gguf_matmul_bt_serial`, the route a one-worker
//!   build takes): four activation rows over the widest four-bit weight, 8.8 ms against
//!   5.9; over the widest six-bit, 156.4 against 144.2. Bit-identical either way.
//!
//! The six- and eight-bit decoders are untouched: four-bit weight is 85% of what the row
//! unpacks and is what this route is for. The six-bit blocks still gain the direct write
//! — a few percent of their route — and the eight-bit block is not the super-block, so
//! the caller's `block_elems != QK_K` guard leaves it on the staging path.
//!
//! [`crate::dequant_q4_k`], the whole-tensor decoder, and [`crate::q4_k_dot_f32`], the
//! one-row multiply's per-block dot, keep their own copies of this arithmetic and are
//! deliberately not unified with it: the first would have to allocate a `Vec` per row to
//! reach a fixed-size slice, and the second is separate work in a different kernel.

use crate::{K_SCALE_SIZE, Q4K_BLOCK_BYTES, QK_K, read_f16_le};

/// One Q4_K super-block (`block`, 144 bytes) widened into `out` (256 values).
///
/// Public in a shared crate, so the portability rule travels with the item: no caller may
/// take this under `RLX_CPU_PORTABLE`, where the staged reference decode is the arm the
/// switch exists to validate against. `rlx-cpu`'s `direct_decode()` is the one caller and
/// enforces it; a second caller inherits the rule here rather than from that crate.
pub fn block(block: &[u8; Q4K_BLOCK_BYTES], out: &mut [f32; QK_K]) {
    let d = read_f16_le(&block[..2]);
    let dmin = read_f16_le(&block[2..4]);
    let (scale, min) = scale_min(block);
    let qs: &[u8; QK_K / 2] = block[4 + K_SCALE_SIZE..]
        .try_into()
        .expect("a Q4_K block holds QK_K / 2 nibble bytes");
    let (groups, _) = qs.as_chunks::<32>();
    // Four 32-byte nibble groups, two sub-blocks each: the low nibbles are the
    // even sub-block of the pair, the high nibbles the odd one.
    for (g, (q, x)) in groups.iter().zip(out.chunks_exact_mut(64)).enumerate() {
        let (lo, hi) = x.split_at_mut(32);
        widen::<false>(
            lo,
            d * f32::from(scale[2 * g]),
            dmin * f32::from(min[2 * g]),
            q,
        );
        widen::<true>(
            hi,
            d * f32::from(scale[2 * g + 1]),
            dmin * f32::from(min[2 * g + 1]),
            q,
        );
    }
}

/// 32 weights of one sub-block: `out[i] = scale * q - min`, from `q`'s low
/// nibbles when `HIGH` is false and from its high nibbles when it is true.
#[inline(always)]
fn widen<const HIGH: bool>(out: &mut [f32], scale: f32, min: f32, q: &[u8; 32]) {
    let (lanes, _) = out.as_chunks_mut::<4>();
    for (qq, lane) in q.as_chunks::<4>().0.iter().zip(lanes) {
        *lane = [
            scale * nibble::<HIGH>(qq[0]) - min,
            scale * nibble::<HIGH>(qq[1]) - min,
            scale * nibble::<HIGH>(qq[2]) - min,
            scale * nibble::<HIGH>(qq[3]) - min,
        ];
    }
}

#[inline(always)]
fn nibble<const HIGH: bool>(b: u8) -> f32 {
    f32::from(if HIGH { b >> 4 } else { b & 0x0F })
}

/// The block's eight sub-block scales and minima out of the shared 12-byte
/// region: the first eight bytes carry the low six bits of each, and the last
/// four carry the missing top two bits of both halves — scale in the low nibble
/// and minimum in the high one, per the reference's `get_scale_min_k4` bit
/// interleaving.
fn scale_min(block: &[u8; Q4K_BLOCK_BYTES]) -> ([u8; 8], [u8; 8]) {
    let q: &[u8; K_SCALE_SIZE] = block[4..4 + K_SCALE_SIZE]
        .try_into()
        .expect("a Q4_K block holds a K_SCALE_SIZE scale region");
    let (halves, _) = q.as_chunks::<4>();
    let mut scale = [0u8; 8];
    let mut min = [0u8; 8];
    for (i, ((&lo, &mid), &hi)) in halves[0].iter().zip(&halves[1]).zip(&halves[2]).enumerate() {
        scale[i] = lo & 63;
        scale[i + 4] = (hi & 0x0F) | ((lo >> 6) << 4);
        min[i] = mid & 63;
        min[i + 4] = (hi >> 4) | ((mid >> 6) << 4);
    }
    (scale, min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GgmlType, dequant_q4_k_block, get_scale_min_k4, quantize::quantize};

    /// A weight the quantiser turns into the awkward blocks: a sine with a heavy
    /// tail, the fixture the route harnesses use, so the packed blocks carry the
    /// scale and minimum bit patterns a real model does and a few it does not.
    fn weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as f32 + 7.0) * 0.017).sin();
                if i % 137 == 0 { v * 9.0 } else { v }
            })
            .collect()
    }

    /// The whole of the contract: every value `block` writes is the reference's,
    /// bit for bit — no tolerance, and no reassociation hidden behind the
    /// four-lane store.
    #[test]
    fn every_value_is_the_reference_value() {
        let packed = quantize(&weights(QK_K * 64), GgmlType::Q4K).expect("Q4_K encode");
        let mut want = [0f32; QK_K];
        let mut got = [0f32; QK_K];
        for (i, b) in packed.chunks_exact(Q4K_BLOCK_BYTES).enumerate() {
            dequant_q4_k_block(b, &mut want);
            block(b.try_into().expect("one block"), &mut got);
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "block {i} element {k}");
            }
        }
    }

    /// The patterns a quantised file does not contain but a decoder must still
    /// get right: an empty nibble half, a saturated one, a zero scale region and
    /// one whose top bits are set.
    #[test]
    fn saturated_and_empty_blocks_match_the_reference() {
        let mut want = [0f32; QK_K];
        let mut got = [0f32; QK_K];
        for (d, fill) in [
            (0x3C00u16, 0x00u8),
            (0x3C00, 0xFF),
            (0x0000, 0xFF),
            (0x7BFF, 0x55),
        ] {
            let mut packed = [0u8; Q4K_BLOCK_BYTES];
            packed[..2].copy_from_slice(&d.to_le_bytes());
            packed[2..4].copy_from_slice(&d.to_le_bytes());
            packed[4..].fill(fill);
            dequant_q4_k_block(&packed, &mut want);
            block(&packed, &mut got);
            assert_eq!(
                got.map(f32::to_bits),
                want.map(f32::to_bits),
                "d={d:#06x} fill={fill:#04x}"
            );
        }
    }

    /// The unpacking is the reference's `get_scale_min_k4`, over the whole of the
    /// scale region rather than the patterns the quantiser emits.
    #[test]
    fn the_unpacked_scales_are_the_reference_helpers() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for round in 0..256 {
            let mut packed = [0u8; Q4K_BLOCK_BYTES];
            for b in &mut packed[4..4 + K_SCALE_SIZE] {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *b = (state >> 33) as u8;
            }
            let (scale, min) = scale_min(&packed);
            let scales = &packed[4..4 + K_SCALE_SIZE];
            for j in 0..8 {
                assert_eq!(
                    (scale[j], min[j]),
                    get_scale_min_k4(j, scales),
                    "round {round} sub-block {j}"
                );
            }
        }
    }
}
