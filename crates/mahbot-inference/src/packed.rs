// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `mahbot.gather_packed` on the processor: rows out of a packed GGUF Q8_0 table,
//! dequantised as they are read.
//!
//! The op is the product's own, so its kernel lives with the product. `rlx-cpu` is
//! upstream's crate and this fork keeps its diff to it minimal; a kernel for a name
//! only this crate emits has no business in a crate that does not know the name. The
//! registration is therefore not a `Once` in the backend's start-up but
//! [`register`], called by [`crate::graphs::Runner::compile`] immediately before the
//! compile that lowers the op — the processor's `compile_custom` refuses a name with
//! no kernel registered, and the frame graphs that carry this op are compiled by a
//! call that would otherwise reach it unregistered.
//!
//! The contract is the graphics backend's, exactly
//! (`rlx-wgpu`'s `backend/compile/lower.rs` arm and `kernels/gather_packed.wgsl`), so
//! the two backends answer the same numbers for the same table:
//!
//! - `attrs` is 12 bytes, three little-endian `u32`: `[cols, rows, bias]`.
//! - Input 0 is the table, `DType::U8`, `rows * (cols / 32) * 34` bytes of Q8_0 rows —
//!   a 2-byte f16 scale then 32 `i8` quants per 34-byte block.
//! - Input 1 is the indices, `DType::F32`, flat, each an integer held in an f32.
//! - The output is `DType::F32`, `n_idx * cols` elements.
//!
//! Per index `i` the row is `clamp(max(indices[i] + bias, 0), 0, rows - 1)` — the
//! bias is the level's first row (`level * vocab`), and the saturation is the
//! shader's `u32(max(idx_f, 0.0))` — and the row's blocks are dequantised straight
//! into `out[i * cols ..][..cols]`, so no f32 image of the table is ever built.
//!
//! What this does not do: it does not check the indices' dtype beyond what the
//! typed view needs, and it does not know about any scheme but Q8_0 — the route
//! serves no other, and another scheme's block geometry would give a plausible wrong
//! row stride, i.e. a silently misread table. Everything it can check it checks and
//! answers `Err` naming the fault rather than panicking on an index: the registry
//! turns an `Err` into a panic that names the op, which is the contract, while a raw
//! index panic would name neither the op nor the tensor.

use std::sync::{Arc, OnceLock};

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

/// The op name, as `emit::gather_rows` emits it and `rlx-wgpu` lowers it.
const NAME: &str = "mahbot.gather_packed";

/// Register the kernel if it is not registered yet. Idempotent: the registry
/// replaces and warns on a second registration, so the call is guarded rather than
/// repeated.
pub(crate) fn register() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        rlx_cpu::op_registry::register_cpu_kernel(Arc::new(GatherPacked));
    });
}

struct GatherPacked;

impl CpuKernel for GatherPacked {
    fn name(&self) -> &str {
        NAME
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if attrs.len() != 12 {
            return Err(format!(
                "{NAME}: the op takes exactly 12 attribute bytes [cols, rows, bias], got {}",
                attrs.len()
            ));
        }
        // Byte-wise rather than `try_into`: the length is checked above, and this
        // keeps the read total.
        let word = |i: usize| {
            let b = &attrs[i * 4..i * 4 + 4];
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        };
        let (cols, rows, bias) = (word(0), word(1), word(2));
        // Q8_0's block geometry, taken from the scheme the graphics route names
        // rather than spelled out here: a second copy of the numbers is a second
        // thing to keep right.
        let scheme = rlx_ir::QuantScheme::GgufQ8_0;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        if cols == 0 || rows == 0 {
            return Err(format!(
                "{NAME}: the table declares rows={rows}, cols={cols}; a zero row leaves an \
                 index nothing to clamp to, and a zero width writes no element"
            ));
        }
        if !(cols as usize).is_multiple_of(block_elems) {
            return Err(format!(
                "{NAME}: a row of {cols} elements is not a whole number of Q8_0 blocks \
                 ({block_elems}), so the row has no packed stride"
            ));
        }
        let blocks_per_row = cols as usize / block_elems;
        let row_bytes = blocks_per_row * block_bytes;
        let table = inputs
            .first()
            .ok_or_else(|| format!("{NAME}: no table input"))?
            .expect_u8("gather_packed table")?;
        let want_table = rows as usize * row_bytes;
        if table.len() != want_table {
            return Err(format!(
                "{NAME}: the table holds {} bytes, but the attrs' geometry (rows={rows}, \
                 cols={cols}, row_bytes={row_bytes}) is {want_table} bytes — the packed row \
                 stride does not match the tensor",
                table.len()
            ));
        }
        let indices = inputs
            .get(1)
            .ok_or_else(|| format!("{NAME}: no indices input"))?
            .expect_f32("gather_packed indices")?;
        let out = output.expect_f32_mut("gather_packed output")?;
        let want_out = indices.len() * cols as usize;
        if out.len() != want_out {
            return Err(format!(
                "{NAME}: the output holds {} elements, but {} indices of {cols} columns are \
                 {want_out}",
                out.len(),
                indices.len()
            ));
        }
        // `rows - 1` would wrap at zero; the guard above refuses such a table, and
        // this does not depend on it — the shader's own `max(rows, 1) - 1`.
        let last = (rows.max(1) - 1) as usize;
        for (i, &idx_f) in indices.iter().enumerate() {
            // Saturating like the shader: a negative index lands on row 0 rather than
            // wrapping to the last row, and NaN lands on 0 (`as` saturates).
            let idx = ((idx_f + bias as f32).max(0.0) as usize).min(last);
            let src = idx * row_bytes;
            let dst = i * cols as usize;
            for blk in 0..blocks_per_row {
                rlx_gguf::dequant_q8_0_block(
                    &table[src + blk * block_bytes..][..block_bytes],
                    &mut out[dst + blk * block_elems..][..block_elems],
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    /// The fixture table's geometry: three rows of two Q8_0 blocks each, so a row
    /// is 68 bytes and the block stride inside a row is exercised rather than
    /// collapsing to one block.
    const ROWS: usize = 3;
    const COLS: usize = 64;

    /// Added to every index before the clamp, as the level's first row is: the
    /// negative indices below saturate to row 0 through it, and a lost bias is
    /// caught by the plain `0` index landing on row 1 rather than on row 0.
    const BIAS: u32 = 1;

    /// `[cols, rows, bias]`, little-endian u32 each — the wire contract.
    fn attrs(cols: usize, rows: usize, bias: u32) -> Vec<u8> {
        [cols as u32, rows as u32, bias]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// A Q8_0 table laid out by hand: `ROWS` rows of `COLS` values, each row with
    /// its own scale and its own quant run, so a row read at the wrong stride
    /// cannot pass as the right one.
    ///
    /// The scales are exact f16 (0.5, 0.75, 1.0) and the quants are small integers,
    /// so `d · q` is exact in f32: the comparison against
    /// [`rlx_gguf::dequant_q8_0`] below is bit-for-bit, not a rounding budget.
    fn table() -> Vec<u8> {
        let block_elems = rlx_ir::QuantScheme::GgufQ8_0.gguf_block_size() as usize;
        let mut bytes = Vec::new();
        for row in 0..ROWS {
            for block in 0..COLS / block_elems {
                let d = half::f16::from_f32(0.5 * (row + 1) as f32);
                bytes.extend_from_slice(&d.to_le_bytes());
                for j in 0..block_elems {
                    let q = ((row * 37 + block * 11 + j) % 121) as i32 - 60;
                    bytes.push(q as i8 as u8);
                }
            }
        }
        assert_eq!(bytes.len(), ROWS * (COLS / block_elems) * 34);
        bytes
    }

    /// Run the kernel over typed views of the operands — the shortest way to
    /// `execute` without a graph or an arena.
    fn run(
        table: &[u8],
        indices: &[f32],
        out_len: usize,
        attrs: &[u8],
    ) -> Result<Vec<f32>, String> {
        let table_shape = Shape::new(&[table.len()], DType::U8);
        let index_shape = Shape::new(&[indices.len()], DType::F32);
        let out_shape = Shape::new(&[out_len], DType::F32);
        // NaN, so an element the kernel never writes is a failure rather than a
        // plausible zero.
        let mut out = vec![f32::NAN; out_len];
        GatherPacked.execute(
            &[
                CpuTensorRef::U8 {
                    data: table,
                    shape: &table_shape,
                },
                CpuTensorRef::F32 {
                    data: indices,
                    shape: &index_shape,
                },
            ],
            CpuTensorMut::F32 {
                data: &mut out,
                shape: &out_shape,
            },
            attrs,
        )?;
        Ok(out)
    }

    /// **The rows the indices select, dequantised exactly.** A plain index, one a
    /// negative reaches only through the bias, one past the last row, and a repeat
    /// — each compared with `rlx_gguf`'s own dequant of the row it must land on, so
    /// a wrong stride, a lost bias or a clamp on the wrong end is a difference here
    /// rather than a plausible row of numbers.
    #[test]
    fn a_gathered_row_is_the_dequantized_row_the_index_selects() {
        let table = table();
        // `idx_u = min(max(idx + bias, 0), rows - 1)`, derived here rather than read
        // back from the kernel: -1 and -9 → 0 (a negative saturates at row 0, the bias
        // folded in first), 0 → 1 and 1 → 2 (a lost bias would answer 0 and 1), 5 → 2
        // (past the table), 1 → 2 again (the repeat).
        let indices: [f32; 6] = [-1.0, 0.0, 1.0, 5.0, 1.0, -9.0];
        let selected = [0usize, 1, 2, 2, 2, 0];
        let got = run(
            &table,
            &indices,
            indices.len() * COLS,
            &attrs(COLS, ROWS, BIAS),
        )
        .expect("a well-formed gather");

        let dequant = rlx_gguf::dequant_q8_0(&table, ROWS * COLS).expect("the fixture dequantizes");
        for (i, &row) in selected.iter().enumerate() {
            let want = &dequant[row * COLS..(row + 1) * COLS];
            assert_eq!(
                &got[i * COLS..(i + 1) * COLS],
                want,
                "index {} (row {row}) is not the row it selects",
                indices[i]
            );
        }
    }

    /// The two refusals about the table itself: an attribute block that does not
    /// carry `[cols, rows, bias]`, and a table whose bytes are not the geometry the
    /// attributes declare — a misread stride being the silently wrong answer this
    /// op must never give.
    #[test]
    fn a_table_that_does_not_match_its_attributes_is_refused() {
        let table = table();
        let indices = [0.0f32];
        let err = |table: &[u8], attrs: &[u8]| {
            run(table, &indices, COLS, attrs).expect_err("a table the kernel cannot read")
        };

        let short = err(&table, &attrs(COLS, ROWS, BIAS)[..11]);
        assert!(
            short.contains("exactly 12 attribute bytes"),
            "a short attribute block: {short}"
        );

        let empty = err(&table, &attrs(COLS, 0, BIAS));
        assert!(
            empty.contains("a zero row leaves an index nothing to clamp to"),
            "a zero-row table: {empty}"
        );

        let ragged = err(&table, &attrs(COLS + 1, ROWS, BIAS));
        assert!(
            ragged.contains("is not a whole number of Q8_0 blocks"),
            "a row that is not whole blocks: {ragged}"
        );

        let short_table = err(&table[..table.len() - 1], &attrs(COLS, ROWS, BIAS));
        assert!(
            short_table.contains("the packed row stride does not match the tensor"),
            "a table one byte short: {short_table}"
        );
    }

    /// The refusals about the operands beside the table: a missing or mistyped
    /// index list, and an output the indices cannot fill — the write that would
    /// otherwise run off its own buffer.
    #[test]
    fn an_index_list_or_an_output_the_op_cannot_fill_is_refused() {
        let table = table();
        let table_shape = Shape::new(&[table.len()], DType::U8);
        let index_shape = Shape::new(&[1], DType::F32);
        let out_shape = Shape::new(&[COLS], DType::F32);
        let attrs = attrs(COLS, ROWS, BIAS);
        let index = [0.0f32];
        let mut out = vec![0f32; COLS];
        let mut bytes = [0u8; COLS];

        let no_table = GatherPacked
            .execute(
                &[],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &out_shape,
                },
                &attrs,
            )
            .expect_err("a call with no table");
        assert!(no_table.contains("no table input"), "{no_table}");

        let no_indices = GatherPacked
            .execute(
                &[CpuTensorRef::U8 {
                    data: &table,
                    shape: &table_shape,
                }],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &out_shape,
                },
                &attrs,
            )
            .expect_err("a call with no indices");
        assert!(no_indices.contains("no indices input"), "{no_indices}");

        let wrong_indices = GatherPacked
            .execute(
                &[
                    CpuTensorRef::U8 {
                        data: &table,
                        shape: &table_shape,
                    },
                    CpuTensorRef::U8 {
                        data: &[0u8],
                        shape: &index_shape,
                    },
                ],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &out_shape,
                },
                &attrs,
            )
            .expect_err("an index list that is not f32");
        assert!(
            wrong_indices.contains("gather_packed indices: expected F32"),
            "{wrong_indices}"
        );

        let short_out = GatherPacked
            .execute(
                &[
                    CpuTensorRef::U8 {
                        data: &table,
                        shape: &table_shape,
                    },
                    CpuTensorRef::F32 {
                        data: &index,
                        shape: &index_shape,
                    },
                ],
                CpuTensorMut::F32 {
                    data: &mut out[..COLS - 1],
                    shape: &out_shape,
                },
                &attrs,
            )
            .expect_err("an output one element short");
        assert!(
            short_out.contains("the output holds"),
            "an output one element short: {short_out}"
        );

        let wrong_out = GatherPacked
            .execute(
                &[
                    CpuTensorRef::U8 {
                        data: &table,
                        shape: &table_shape,
                    },
                    CpuTensorRef::F32 {
                        data: &index,
                        shape: &index_shape,
                    },
                ],
                CpuTensorMut::U8 {
                    data: &mut bytes,
                    shape: &out_shape,
                },
                &attrs,
            )
            .expect_err("an output that is not f32");
        assert!(
            wrong_out.contains("gather_packed output: expected F32"),
            "{wrong_out}"
        );
    }
}
