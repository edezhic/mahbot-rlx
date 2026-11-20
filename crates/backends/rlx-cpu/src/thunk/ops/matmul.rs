// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_fused_mat_mul_bias_act(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedMatMulBiasAct { activation } = &node.op else {
        unreachable!()
    };
    {
        let shape = &node.shape;
        let n = shape.dim(shape.rank() - 1).unwrap_static();
        let total = shape.num_elements().unwrap();
        let m = total / n;
        let a_len = get_len(graph, node.inputs[0]);
        let k = a_len / m;
        Thunk::FusedMmBiasAct {
            a: node_offset(arena, node.inputs[0]),
            w: node_offset(arena, node.inputs[1]),
            bias: node_offset(arena, node.inputs[2]),
            c: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            act: *activation,
            // Every operand's layout, from the one rule that answers it: the epilogue reads
            // f32 lanes, so a packed half operand is widened (or dequantised on the fly, for
            // the weight) before the multiply and rounded back afterwards.
            a_kind: super::elementwise::half_slot(graph, arena, node.inputs[0]),
            w_kind: super::elementwise::half_slot(graph, arena, node.inputs[1]),
            c_kind: super::elementwise::half_slot(graph, arena, node.id),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::MatMul = &node.op else { unreachable!() };
    {
        let shape = &node.shape;
        let a_shape = &graph.node(node.inputs[0]).shape;
        let b_shape = &graph.node(node.inputs[1]).shape;
        // Prefer inferred matmul shape from operands — ONNX bundle
        // meta often over-ranks outputs (e.g. [seq, seq, H]).
        let eff = rlx_ir::shape::matmul_shape(a_shape, b_shape).unwrap_or_else(|_| shape.clone());
        let rank = eff.rank().max(2);
        let n = eff.dim(rank - 1).unwrap_static();
        let k_dim = a_shape.dim(a_shape.rank().max(2) - 1).unwrap_static();
        if shape.dtype() == rlx_ir::DType::C64 {
            // Complex GEMM (interleaved re/im). Handles 2D and
            // 3D×2D (flatten M); both-operand batched C64 is not
            // yet wired.
            let both = a_shape.rank() >= 3 && b_shape.rank() >= 3;
            assert!(!both, "batched (both-operand) C64 matmul not yet supported");
            let m: usize = if a_shape.rank() >= 3 {
                (0..a_shape.rank() - 1)
                    .map(|d| a_shape.dim(d).unwrap_static())
                    .product()
            } else {
                a_shape.dim(a_shape.rank() - 2).unwrap_static()
            };
            Thunk::CgemmC64 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                c: node_offset(arena, node.id),
                m: m as u32,
                k: k_dim as u32,
                n: n as u32,
            }
        } else {
            // Batched GEMM when both operands carry batch dimensions.
            // 3D×2D (activations × shared weight) must flatten to one Sgemm.
            let both_batched = a_shape.rank() >= 3 && b_shape.rank() >= 3;
            // ...and when only `b` is batched. `matmul_shape` broadcasts a rank-2
            // lhs across the rhs's batch, so the output IS batched — but the
            // 2-D flatten below cannot express that. It would emit a single
            // Sgemm against `b`'s first matrix and leave every later output
            // batch holding whatever was already in the arena: a silent wrong
            // answer, not a crash. `BatchedSgemm` handles it via `a_bcast`.
            //
            // This is the mirror of the case the `BatchedSgemm` doc comment
            // already warns about; only the `a`-side was ever wired up.
            let lhs_bcast_batched = a_shape.rank() < 3 && b_shape.rank() >= 3;
            let batched_3d = rank >= 3
                && (both_batched && a_shape.rank() + b_shape.rank() > 4 || lhs_bcast_batched);
            // Per-operand batch counts: an operand whose batch product is 1
            // (while the output batch is > 1) is BROADCAST across the batch.
            let a_batch: usize = (0..a_shape.rank().saturating_sub(2))
                .map(|d| a_shape.dim(d).unwrap_static())
                .product();
            let b_batch: usize = (0..b_shape.rank().saturating_sub(2))
                .map(|d| b_shape.dim(d).unwrap_static())
                .product();
            if batched_3d && shape.dtype() == rlx_ir::DType::F64 {
                let mut batch_prod = 1usize;
                for d in 0..rank - 2 {
                    batch_prod *= eff.dim(d).unwrap_static();
                }
                let m_dim = eff.dim(rank - 2).unwrap_static();
                Thunk::BatchedDgemmF64 {
                    a: node_offset(arena, node.inputs[0]),
                    b: node_offset(arena, node.inputs[1]),
                    c: node_offset(arena, node.id),
                    batch: batch_prod as u32,
                    m: m_dim as u32,
                    k: k_dim as u32,
                    n: n as u32,
                    a_bcast: a_batch == 1 && batch_prod > 1,
                    b_bcast: b_batch == 1 && batch_prod > 1,
                }
            } else if batched_3d && shape.dtype() == rlx_ir::DType::F32 {
                let mut batch_prod = 1usize;
                for d in 0..rank - 2 {
                    batch_prod *= eff.dim(d).unwrap_static();
                }
                let m_dim = eff.dim(rank - 2).unwrap_static();
                Thunk::BatchedSgemm {
                    a: node_offset(arena, node.inputs[0]),
                    b: node_offset(arena, node.inputs[1]),
                    c: node_offset(arena, node.id),
                    batch: batch_prod as u32,
                    m: m_dim as u32,
                    k: k_dim as u32,
                    n: n as u32,
                    a_bcast: a_batch == 1 && batch_prod > 1,
                    b_bcast: b_batch == 1 && batch_prod > 1,
                }
            } else {
                let m = if a_shape.rank() >= 3 && b_shape.rank() <= 2 {
                    let mut m_prod = 1usize;
                    for d in 0..a_shape.rank() - 1 {
                        m_prod *= a_shape.dim(d).unwrap_static();
                    }
                    m_prod
                } else if a_shape.rank() >= 2 {
                    a_shape.dim(a_shape.rank() - 2).unwrap_static()
                } else {
                    eff.num_elements().unwrap_or(1) / n.max(1)
                };
                // The dequant-on-the-fly GEMM is correct for exactly one layout: `A` and `C`
                // f32-wide, `B` packed half. Each operand's layout is `half_slot`'s one answer,
                // never `Param`-ness or the declared dtype — reading a widened slot as packed
                // halves (or the reverse) is silent garbage, and under bf16 autocast the RHS of
                // a matmul is routinely a cast activation rather than a weight.
                let a_kind = super::elementwise::half_slot(graph, arena, node.inputs[0]);
                let b_kind = super::elementwise::half_slot(graph, arena, node.inputs[1]);
                let c_kind = super::elementwise::half_slot(graph, arena, node.id);
                let use_packed_rhs = b_kind.packed() && !a_kind.packed() && !c_kind.packed();
                // BF16 weight (rhs) with f32 output → dequant-on-the-fly GEMM
                // (bf16-resident LM head: half the weight memory traffic).
                if use_packed_rhs && b_shape.dtype() == rlx_ir::DType::BF16 {
                    return Thunk::SgemmBf16 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        m: m as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    };
                }
                // F16 weight (rhs) with f32 output → dequant-on-the-fly GEMM,
                // IEEE-half twin of the BF16 path (else the F16 bytes are read as
                // f32 garbage — the qwen35 vision F16-weight regression).
                if use_packed_rhs && b_shape.dtype() == rlx_ir::DType::F16 {
                    return Thunk::SgemmF16 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        m: m as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    };
                }
                // Any remaining packed-half operand (a half ACTIVATION the
                // planner left at 2 B, a packed LHS, or a packed destination)
                // goes through the general widen→f32-GEMM→round path. The f32
                // `Sgemm` below would read those 2-byte halves as f32.
                if a_kind.packed() || b_kind.packed() || c_kind.packed() {
                    return Thunk::SgemmHalf {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        m: m as u32,
                        k: k_dim as u32,
                        n: n as u32,
                        a_kind,
                        b_kind,
                        c_kind,
                    };
                }
                match shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::Dgemm {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        m: m as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    },
                    _ => {
                        if let Some(&(asrc, ta, bsrc, tb)) = matmul_fold.get(&node.id) {
                            // Folded Transpose→MatMul: read pre-transpose
                            // operands, do the transpose via cblas flags.
                            Thunk::SgemmT {
                                a: node_offset(arena, asrc),
                                b: node_offset(arena, bsrc),
                                c: node_offset(arena, node.id),
                                m: m as u32,
                                k: k_dim as u32,
                                n: n as u32,
                                ta,
                                tb,
                            }
                        } else {
                            Thunk::Sgemm {
                                a: node_offset(arena, node.inputs[0]),
                                b: node_offset(arena, node.inputs[1]),
                                c: node_offset(arena, node.id),
                                m: m as u32,
                                k: k_dim as u32,
                                n: n as u32,
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_lora_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LoraMatMul { scale } = &node.op else {
        unreachable!()
    };
    {
        // x [m, k], w [k, n], a [k, r], b [r, n].
        let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let m = total / n.max(1);
        let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let k = x_total / m.max(1);
        let a_total = graph.node(node.inputs[2]).shape.num_elements().unwrap();
        let r = a_total / k.max(1);
        Thunk::LoraMatMul {
            x: node_offset(arena, node.inputs[0]),
            w: node_offset(arena, node.inputs[1]),
            a: node_offset(arena, node.inputs[2]),
            b: node_offset(arena, node.inputs[3]),
            dst: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            r: r as u32,
            scale: *scale,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_grouped_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GroupedMatMul = &node.op else {
        unreachable!()
    };
    {
        // Inputs: [input(M, K), weight(E, K, N), expert_idx(M)]. The dims are
        // checked against the declared output rather than trusted: `n` comes
        // off the bank, `dst` is sized from `node.shape`, and a bank still in
        // `[E, N, K]` order makes those two disagree — under- or over-writing
        // the output slot with no other symptom. See `grouped_matmul_dims`.
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let dims = rlx_ir::shape::grouped_matmul_dims(in_shape, w_shape, Some(&node.shape))
            .unwrap_or_else(|e| panic!("rlx-cpu: node {:?}: {e}", node.id));
        let (m, k_dim, n, num_experts) = (dims.m, dims.k, dims.n, dims.num_experts);
        // A folded `Transpose(bank, [0, 2, 1])`: read the bank in its `[E, N, K]`
        // form and let the GEMM transpose B, instead of copying the whole bank.
        // `dims` still comes from the graph's declared (post-transpose) shapes,
        // so only the buffer and the stride order change.
        let (weight, w_transposed) = match matmul_fold.get(&node.id) {
            Some(&(_, _, bsrc, true)) => (node_offset(arena, bsrc), true),
            _ => (node_offset(arena, node.inputs[1]), false),
        };
        Thunk::GroupedMatMul {
            input: node_offset(arena, node.inputs[0]),
            weight,
            expert_idx: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            m: m as u32,
            k_dim: k_dim as u32,
            n: n as u32,
            num_experts: num_experts as u32,
            w_transposed,
        }
    }
}

/// `c = act(a @ w + bias)` for any layout of the three operands. A packed half operand is
/// widened before the multiply (or, for the weight, dequantised on the fly) and a packed
/// destination is rounded after the epilogue, so no operand is ever read as lanes it does not
/// hold — the rule `compile_mat_mul` applies to its own three, asked here through the same
/// [`super::elementwise::half_slot`].
#[inline(always)]
pub(crate) fn exec_fused_mm_bias_act(t: &Thunk, base: *mut u8) {
    let Thunk::FusedMmBiasAct {
        a,
        w,
        bias,
        c,
        m,
        k,
        n,
        act,
        a_kind,
        w_kind,
        c_kind,
    } = t
    else {
        unreachable!()
    };
    let (m, k, n) = (*m as usize, *k as usize, *n as usize);
    unsafe {
        let bias_v = sl(*bias, base, n);
        let a_buf;
        let a_sl: &[f32] = if a_kind.packed() {
            a_buf = crate::blas::widen_half_vec(
                sl_typed::<u16>(*a, base, m * k),
                *a_kind == HalfKind::Bf16,
            );
            &a_buf
        } else {
            sl(*a, base, m * k)
        };
        let mut out_buf;
        let out: &mut [f32] = if c_kind.packed() {
            out_buf = vec![0f32; m * n];
            &mut out_buf
        } else {
            sl_mut(*c, base, m * n)
        };
        match w_kind {
            HalfKind::F32 => {
                // Match torch `F.linear` / `addmm(bias, A, W)` association order.
                for row in 0..m {
                    out[row * n..(row + 1) * n].copy_from_slice(bias_v);
                }
                crate::blas::sgemm_accumulate_auto(a_sl, sl(*w, base, k * n), out, m, k, n);
            }
            kind => {
                // A packed half weight: the dequant-on-the-fly GEMM writes `C` rather than
                // accumulating into it, so the bias is added after. That is ~1 ULP from the
                // f32 arm's association order, which is the price of reading 2 B/elem
                // weights at all.
                let b16 = sl_typed::<u16>(*w, base, k * n);
                if *kind == HalfKind::F16 {
                    crate::blas::sgemm_f16_rhs(a_sl, b16, out, m, k, n);
                } else {
                    crate::blas::sgemm_bf16_rhs(a_sl, b16, out, m, k, n);
                }
                crate::blas::bias_add(out, bias_v, m, n);
            }
        }
        if let Some(act) = act {
            apply_activation_inplace(out, *act);
        }
        if c_kind.packed() {
            crate::blas::round_half_into(
                out,
                sl_mut_typed::<u16>(*c, base, m * n),
                *c_kind == HalfKind::Bf16,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_bias_add(t: &Thunk, base: *mut u8) {
    let Thunk::BiasAdd {
        src,
        bias,
        dst,
        m,
        n,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, n) = (*m as usize, *n as usize);
        let len = m * n;
        unsafe {
            let out = sl_mut(*dst, base, len);
            if *src != *dst {
                let src_ptr = base.add(*src) as *const f32;
                let dst_ptr = base.add(*dst) as *mut f32;
                if src_ptr != dst_ptr {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
                }
            }
            crate::blas::bias_add(out, sl(*bias, base, n), m, n);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_lora_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::LoraMatMul {
        x,
        w,
        a,
        b,
        dst,
        m,
        k,
        n,
        r,
        scale,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n, r) = (*m as usize, *k as usize, *n as usize, *r as usize);
        unsafe {
            let xs = sl(*x, base, m * k);
            let ws = sl(*w, base, k * n);
            let a_s = sl(*a, base, k * r);
            let bs = sl(*b, base, r * n);
            let out = sl_mut(*dst, base, m * n);
            crate::blas::sgemm(xs, ws, out, m, k, n);
            let mut tmp = vec![0f32; m * r];
            crate::blas::sgemm(xs, a_s, &mut tmp, m, k, r);
            if *scale != 1.0 {
                for v in tmp.iter_mut() {
                    *v *= *scale;
                }
            }
            crate::blas::sgemm_accumulate(&tmp, bs, out, m, r, n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::bytes::AlignedBytes;

    /// The fused multiply's operands as one arena: `a` f32, the weight's bytes,
    /// `bias` f32 and the `c` slot, each at its own 8-byte-aligned offset.
    ///
    /// The backing store is [`AlignedBytes`] rather than `Vec<u8>` because the
    /// kernels reinterpret these offsets as `&[f32]` and `&[u16]`: an unaligned base
    /// would make those borrows unsound, not merely slow.
    fn exec(
        a: &[f32],
        w: &[u8],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
        w_kind: HalfKind,
        act: Option<Activation>,
    ) -> Vec<f32> {
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let (got, _) = exec_layout(
            &a_bytes,
            w,
            bias,
            m,
            k,
            n,
            HalfKind::F32,
            w_kind,
            HalfKind::F32,
            act,
        );
        got
    }

    /// [`exec`] with each operand's layout spelled out, handing back the destination's bytes as
    /// they lie in the arena: a packed `a` or `c` is 2 B/element there.
    #[allow(clippy::too_many_arguments)]
    fn exec_layout(
        a: &[u8],
        w: &[u8],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
        a_kind: HalfKind,
        w_kind: HalfKind,
        c_kind: HalfKind,
        act: Option<Activation>,
    ) -> (Vec<f32>, Vec<u8>) {
        let align = |at: usize| at.div_ceil(8) * 8;
        let a_off = 0;
        let w_off = align(a_off + a.len());
        let bias_off = align(w_off + w.len());
        let c_off = align(bias_off + bias.len() * 4);
        let c_bytes = if c_kind.packed() {
            m * n * 2
        } else {
            m * n * 4
        };
        let mut arena = AlignedBytes::zeroed(align(c_off + c_bytes));
        {
            let buf = arena.as_mut_slice();
            buf[a_off..a_off + a.len()].copy_from_slice(a);
            buf[w_off..w_off + w.len()].copy_from_slice(w);
            for (i, &v) in bias.iter().enumerate() {
                buf[bias_off + i * 4..bias_off + i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let thunk = Thunk::FusedMmBiasAct {
            a: a_off,
            w: w_off,
            bias: bias_off,
            c: c_off,
            m: m as u32,
            k: k as u32,
            n: n as u32,
            act,
            a_kind,
            w_kind,
            c_kind,
        };
        exec_fused_mm_bias_act(&thunk, arena.as_mut_slice().as_mut_ptr());
        let bytes = arena.as_mut_slice()[c_off..c_off + c_bytes].to_vec();
        let got = if c_kind.packed() {
            bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
                .collect()
        } else {
            arena.as_slice_of::<f32>()[c_off / 4..c_off / 4 + m * n].to_vec()
        };
        (got, bytes)
    }

    /// `a · w + bias`, in f32, with the activation the thunk applies — the
    /// association order the f32 arm uses (bias first, then the accumulation).
    fn reference(
        a: &[f32],
        w: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
        act: Option<Activation>,
    ) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = bias[j];
                for p in 0..k {
                    acc += a[i * k + p] * w[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        if let Some(act) = act {
            apply_activation_inplace(&mut c, act);
        }
        c
    }

    /// The weight as f16 bytes, which is how a packed half param lies in the arena.
    fn halves(w: &[f32]) -> Vec<u8> {
        w.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_ne_bytes())
            .collect()
    }

    /// The comparison, relative and absolute: both arms of the epilogue reach the
    /// same product in a different association order, so the bound is the
    /// reassociation's, not the kernel's.
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "{what} at {i}: {g} against the reference's {w}"
            );
        }
    }

    /// **A packed half weight takes the dequant-on-the-fly GEMM, not an f32 read of
    /// its bytes.** `w_kind` is `HalfKind::F16` and the slot holds 2 B/element, as a
    /// half param does; reading those bytes as f32 lanes gives numbers unrelated to
    /// the product, which is how the CED tagger's 527-way head saturated at exactly
    /// 1.0 and withheld most of the speech from the recogniser. The f32 arm is the
    /// same comparison against its own layout, so the two are held to one reference.
    #[test]
    fn a_packed_half_weight_is_multiplied_as_the_halves_it_holds() {
        // `m` at the widen threshold, `k` a whole number of 8-lane rows, `n` odd so
        // the tail of the weight row is not a padded one.
        let (m, k, n) = (4usize, 8usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.5 * j as f32 - 1.0).collect();

        // The f16 weight is the rounded one both sides see: the reference reads the
        // halves back, so the comparison is of the GEMM and not of a rounding.
        let w_half: Vec<f32> = w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();

        for act in [None, Some(Activation::Gelu)] {
            let want = reference(&a, &w_half, &bias, m, k, n, act);
            let packed = exec(&a, &halves(&w), &bias, m, k, n, HalfKind::F16, act);
            assert_close(&packed, &want, &format!("the packed arm with {act:?}"));

            let f32_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let full = exec(&a, &f32_bytes, &bias, m, k, n, HalfKind::F32, act);
            assert_close(&full, &want, &format!("the f32 arm with {act:?}"));
        }
    }

    /// **A packed activation and a packed destination are read and written as the halves they
    /// hold, too.** The slot rule is one question asked of all three operands, so a graph whose
    /// lhs or output the planner left at 2 B/element — a half activation under autocast — is
    /// multiplied rather than reinterpreted. Only the weight was ever guarded, and the other two
    /// were the same defect waiting for a graph that produced one.
    #[test]
    fn a_packed_activation_and_destination_are_read_and_written_as_halves() {
        let (m, k, n) = (3usize, 8usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.5 * j as f32 - 1.0).collect();

        // The rounded halves both sides see, so the comparison is of the multiply and not of a
        // rounding: a packed operand holds `half::f16` bits, and the reference reads them back.
        let a_half: Vec<f32> = a.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let w_half: Vec<f32> = w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();

        for act in [None, Some(Activation::Gelu)] {
            let want = reference(&a_half, &w_half, &bias, m, k, n, act);
            for (a_kind, c_kind) in [
                (HalfKind::F16, HalfKind::F16),
                (HalfKind::F16, HalfKind::F32),
                (HalfKind::F32, HalfKind::F16),
            ] {
                let a_bytes = if a_kind.packed() {
                    halves(&a)
                } else {
                    a.iter().flat_map(|v| v.to_ne_bytes()).collect()
                };
                let (got, _) = exec_layout(
                    &a_bytes,
                    &halves(&w),
                    &bias,
                    m,
                    k,
                    n,
                    a_kind,
                    HalfKind::F16,
                    c_kind,
                    act,
                );
                assert_close(
                    &got,
                    &want,
                    &format!("a={a_kind:?} c={c_kind:?} with {act:?}"),
                );
            }
        }
    }
}
