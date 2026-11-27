// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The width the host keeps a file's half-precision tensors at.
//!
//! A tensor the file stores as f16 or bf16 is declared F32 in every graph this crate
//! builds, and the declaration is what picks the kernel — it does not change here. What
//! changes is where the widening happens. Instead of an f32 image in the process, the
//! compile reads the file's own bytes through the upload's window and widens them a piece
//! at a time as it writes the device ([`pieces`]). The image a row used to carry — 45.8 MiB
//! for the recognition projector's four tensors, 165.5 MiB for the synthesis projector's
//! sixty-seven — is then never made, and what the read leaves behind is clean file-backed
//! pages, which the footprint does not count: what a row used to charge itself for good, this
//! platform never giving a freed anonymous allocation back, is not allocated at all.
//!
//! The conversion is the loaders' own (`rlx_gguf`'s per-scheme decode, `half`'s widening),
//! so a widened parameter is bit for bit what the f32 image held: this moves where the bytes
//! are made, not what they are.
//!
//! `MAHBOT_HOST_F16=0` puts the f32 image back — the arm this route replaced — so the two can
//! be measured against each other in one session. A run that names the variable is screened
//! and writes no record, the stand's and the wake bench's alike, because a take carries the
//! shipped arrangement's numbers only.

use std::cell::Cell;
use std::ffi::OsString;

use anyhow::{Result, bail};
use half::slice::HalfFloatSliceExt;
use rlx_gguf::GgmlType;

thread_local! {
    /// The width this build resolved: the environment is read at the first half-precision
    /// tensor that asks, not once per tensor, and the answer stands for the build
    /// ([`emit::clear_origins`](crate::emit::clear_origins) ends it).
    static KEEP: Cell<Option<bool>> = const { Cell::new(None) };
}

/// The order a parameter's values are written in.
#[derive(Clone, Copy)]
pub(crate) enum Layout {
    /// The file's own element order — what a convolution kernel, a bias or a table declares.
    FileOrder,
    /// `Op::MatMul`'s `[k, n]` with `n` fastest, out of a file holding `[k, n]` with `k`
    /// fastest: the one place a weight is transposed, done while it is widened rather than as
    /// a second pass over a materialised image.
    MatMul { k: usize, n: usize },
}

/// The most f32 elements one piece holds: 2 MiB — small enough that the saving survives it,
/// against the widest image it replaces (42 MiB, the DAC entry kernel). The matmul layout
/// holds one piece and its column, so at most two of these.
const PIECE_ELEMS: usize = 512 << 10;

/// The host-width arm as this process resolved it: whether the file's width is kept, and
/// whether `MAHBOT_HOST_F16` was named at all — which screens a take, because a record
/// carries the shipped arrangement's numbers only.
///
/// # Errors
/// `MAHBOT_HOST_F16` is set and is not a boolean.
pub(crate) fn resolved() -> Result<(bool, bool)> {
    resolved_of(std::env::var_os("MAHBOT_HOST_F16"))
}

/// [`resolved`] over an explicit value, so the parse is testable without touching the
/// process's environment: unset is on and unnamed, and a value that is not a boolean is
/// refused by name rather than defaulted, like every other `MAHBOT_*` switch.
pub(crate) fn resolved_of(value: Option<OsString>) -> Result<(bool, bool)> {
    let Some(value) = value else {
        return Ok((true, false));
    };
    let enabled = match value.to_string_lossy().trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => bail!("MAHBOT_HOST_F16 `{other}` is not a boolean (0/1, off/on)"),
    };
    Ok((enabled, true))
}

/// Whether the build being made keeps the file's width — [`resolved`] once per build rather
/// than once per tensor, like every other variable a build reads.
///
/// # Errors
/// `MAHBOT_HOST_F16` is set and is not a boolean.
pub(crate) fn keep() -> Result<bool> {
    if let Some(keep) = KEEP.with(Cell::get) {
        return Ok(keep);
    }
    let (keep, _) = resolved()?;
    KEEP.with(|cell| cell.set(Some(keep)));
    Ok(keep)
}

/// End the build's resolution, so the next build reads the environment again.
pub(crate) fn forget() {
    KEEP.with(|cell| cell.set(None));
}

/// Widen `src` — half-precision values in `layout` — to f32 one piece at a time:
/// `write(elem_offset, values)` per piece, in ascending order, covering every value once.
///
/// Nothing larger than one piece — and, for the matmul layout, its column — is allocated, so a
/// caller streaming the result to a device holds no f32 image.
pub(crate) fn pieces(
    dtype: GgmlType,
    src: &[u8],
    layout: Layout,
    write: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<()> {
    with_piece(PIECE_ELEMS, dtype, src, layout, write)
}

/// [`pieces`] at an explicit piece size, so the walk's geometry — whole rows per piece, a
/// short last piece — is testable without a 2 MiB buffer.
fn with_piece(
    piece_elems: usize,
    dtype: GgmlType,
    src: &[u8],
    layout: Layout,
    mut write: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<()> {
    if !src.len().is_multiple_of(2) {
        bail!("{} bytes are not whole half-precision values", src.len());
    }
    let elems = src.len() / 2;
    // A piece is whole output rows of the matmul layout: a device write is contiguous only
    // over them, and a row is never split.
    let piece = match layout {
        Layout::FileOrder => piece_elems.min(elems),
        Layout::MatMul { k, n } => {
            if k == 0 || n == 0 || k * n != elems {
                bail!("a {k}×{n} matmul layout for {elems} values");
            }
            // A row wider than a piece cannot be staged whole, and the saving must survive
            // the staging: no weight in these files comes near it.
            if n > piece_elems {
                bail!("a {k}×{n} matmul layout wider than one {piece_elems}-value piece");
            }
            (piece_elems / n).min(k) * n
        }
    };
    let mut stage = vec![0f32; piece];
    // The matmul layout's column, `piece / n` values at most: grown on first use, so the
    // file's own order allocates nothing for it.
    let mut column: Vec<f32> = Vec::new();
    let mut at = 0;
    while at < elems {
        let take = piece.min(elems - at);
        let dst = &mut stage[..take];
        match layout {
            Layout::FileOrder => widen(dtype, &src[2 * at..2 * (at + take)], dst)?,
            // The file holds `[k, n]` with `k` fastest, so output row `i` takes one value out
            // of each of the file's `n` runs, `k` apart. Widening a whole run — contiguous in
            // the file — and scattering it down the piece's column keeps the source read
            // forward and vectorised, and the piece's rows in cache while the sweep fills
            // them.
            Layout::MatMul { k, n } => {
                let piece_rows = take / n;
                let row = at / n;
                column.resize(piece_rows, 0.0);
                for j in 0..n {
                    let start = 2 * (j * k + row);
                    widen(dtype, &src[start..start + 2 * piece_rows], &mut column)?;
                    for (i, &v) in column.iter().enumerate() {
                        dst[i * n + j] = v;
                    }
                }
            }
        }
        write(at, dst)?;
        at += take;
    }
    Ok(())
}

/// `dst[i] = f32(src[i])` for the 2-byte `dtype`, exactly as `rlx_gguf`'s decode widens it.
///
/// `src` holds `dst.len()` whole values — every caller slices both from the same count — so the
/// address is the one thing left to check: `half`'s `Pod` cast is defined over 2-byte-aligned
/// values, and GGUF's 32-byte tensor alignment is what makes it even in practice. A file that
/// does not hold to it is refused rather than taking the process down inside the cast.
fn widen(dtype: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    if !(src.as_ptr() as usize).is_multiple_of(2) {
        bail!(
            "the tensor's {} half-precision values start at an odd address",
            dst.len()
        );
    }
    match dtype {
        // Both conversions are exact widenings — an f16/bf16 value is an f32.
        GgmlType::F16 => bytemuck::cast_slice::<u8, half::f16>(src).convert_to_f32_slice(dst),
        GgmlType::BF16 => bytemuck::cast_slice::<u8, half::bf16>(src).convert_to_f32_slice(dst),
        other => bail!("`{}` is not a half-precision type", other.name()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` f16 values in the file's order: `v(i) = i / 8`, exactly representable.
    fn halves(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|i| half::f16::from_f32(i as f32 / 8.0).to_le_bytes())
            .collect()
    }

    /// Every piece `pieces` writes, in order: `(element offset, values)`.
    fn walk(layout: Layout, elems: usize, piece: usize) -> Vec<(usize, Vec<f32>)> {
        let mut out = Vec::new();
        with_piece(piece, GgmlType::F16, &halves(elems), layout, |at, v| {
            out.push((at, v.to_vec()));
            Ok(())
        })
        .expect("a well-formed walk");
        out
    }

    #[test]
    fn the_switch_is_on_unless_it_is_told_not_to_be() {
        assert_eq!(resolved_of(None).expect("unset is on"), (true, false));
        for off in ["0", "off", "false", "no", " OFF "] {
            assert_eq!(
                resolved_of(Some(off.into())).expect(off),
                (false, true),
                "{off} is off"
            );
        }
        for on in ["1", "on", "true", "yes"] {
            assert_eq!(
                resolved_of(Some(on.into())).expect(on),
                (true, true),
                "{on} is on"
            );
        }
        assert!(resolved_of(Some("f16".into())).is_err(), "not a boolean");
    }

    #[test]
    fn a_file_order_walk_covers_every_value_once_in_order() {
        for piece in [1, 3, 7, 64] {
            let pieces = walk(Layout::FileOrder, 7, piece);
            let offsets: Vec<usize> = pieces.iter().map(|(at, _)| *at).collect();
            let want: Vec<usize> = (0..7).step_by(piece).collect();
            assert_eq!(offsets, want, "piece {piece} ascends without a gap");
            let all: Vec<f32> = pieces.into_iter().flat_map(|(_, v)| v).collect();
            let want: Vec<f32> = (0..7).map(|i| i as f32 / 8.0).collect();
            assert_eq!(all, want);
        }
    }

    #[test]
    fn a_matmul_walk_is_the_transposed_file_and_breaks_only_on_rows() {
        // The file holds [k, n] with k fastest; the parameter is [k, n] with n fastest.
        let (k, n) = (5usize, 3usize);
        for piece in [3, 6, 15, 4096] {
            let pieces = walk(Layout::MatMul { k, n }, k * n, piece);
            let rows = (piece / n).max(1).min(k);
            let offsets: Vec<usize> = pieces.iter().map(|(at, _)| *at).collect();
            let want: Vec<usize> = (0..k).step_by(rows).map(|r| r * n).collect();
            assert_eq!(offsets, want, "piece {piece}: whole rows, ascending");
            let all: Vec<f32> = pieces.into_iter().flat_map(|(_, v)| v).collect();
            let file: Vec<f32> = (0..k * n).map(|i| i as f32 / 8.0).collect();
            let mut want = vec![0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    want[i * n + j] = file[j * k + i];
                }
            }
            assert_eq!(all, want, "piece {piece}");
        }
    }

    #[test]
    fn an_odd_length_and_a_bad_layout_are_refused() {
        let bytes = halves(6);
        let write = |_: usize, _: &[f32]| Ok(());
        assert!(
            with_piece(4, GgmlType::F16, &bytes[..11], Layout::FileOrder, write).is_err(),
            "11 bytes are not whole half-precision values"
        );
        let matmul = Layout::MatMul { k: 4, n: 2 };
        assert!(
            with_piece(4, GgmlType::F16, &bytes, matmul, write).is_err(),
            "8 values declared as 6"
        );
        let wide = Layout::MatMul { k: 1, n: 6 };
        assert!(
            with_piece(4, GgmlType::F16, &bytes, wide, write).is_err(),
            "a row wider than the piece"
        );
        assert!(
            with_piece(4, GgmlType::F32, &bytes, Layout::FileOrder, write).is_err(),
            "f32 is not a half-precision type"
        );
    }
}
