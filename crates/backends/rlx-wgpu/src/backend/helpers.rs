// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulCompute {
    F32,
    F16,
    Coop16,
    /// Cooperative-matrix on Apple's `simdgroup_float8x8` — same hardware
    /// GEMM unit as Coop16 but with f32 operands and f32 accumulator.
    /// No precision loss vs F32 baseline; no f16 overflow risk in deep
    /// FFN sums. Used when alignment + features allow but the IR is f32.
    CoopF32,
    /// Vulkan/NVIDIA 16×16 f16 tensor-core matmul with K-slab f32
    /// reduction (avoids Naga mixed f16/f32 coop_mat bugs).
    CoopF16Vk,
    /// `matmul_coop_f16_f32acc`: f16 operands, f32 cooperative accumulator, 64×64
    /// tile per 64-thread workgroup (two 32-wide subgroups over 64×32 halves), A
    /// downcast from the f32 arena into a ping-pong workgroup stage at the load,
    /// B read as f16 from the weight buffer. The packed-weight GEMM the embedding
    /// encoder's text tower wants — see the shader header for the tile and the
    /// load convention, and `crate::coop_probe` for the per-device verdict this
    /// is gated on.
    CoopF16F32Acc,
    /// The same kernel and geometry as [`MatmulCompute::CoopF16F32Acc`], with B
    /// bound from the attention route's own f16 operand buffer rather than from
    /// the weight mirror. The dense attention route has no weight operand for B to
    /// live in, so it materialises Kᵀ and V′ as f16 once per step
    /// (`Step::CastF32ToF16` into that buffer) and reads them here; A is still the
    /// f32 arena, downcast at the load as in `CoopF16F32Acc`.
    CoopF16F32AccScratch,
    /// Packed-BF16 weight kept 2 B/elem in the `bf16_weight_buffer` side
    /// buffer and unpacked in-shader (`matmul_bf16w`). Reads half the B
    /// bytes; bit-exact to a bf16-rounded f32 matmul (f32 accumulator).
    /// Used only by the plain tiled matmul path for a BF16 Param rhs.
    Bf16Packed,
}

/// Split-write QKV matmul kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulQkvKind {
    F32,
    CoopF32,
    CoopF16Vk,
}

pub(crate) fn fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-wgpu Op::Fft: unsupported dtype {other:?}"),
    }
}

pub(crate) fn fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-wgpu Op::Fft: bad dtype tag {other}"),
    }
}

pub(crate) fn binary_op_id(op: BinaryOp) -> u32 {
    op.opcode()
}

pub(crate) fn compare_op_id(op: CmpOp) -> u32 {
    op.opcode()
}

pub(crate) fn reduce_op_id(op: ReduceOp) -> u32 {
    op.opcode()
}

pub(crate) fn activation_op_id(act: Activation) -> u32 {
    act.opcode_relu_first()
}

/// True when a matmul reads its weight `B` from the separate f16 shadow buffer
/// (so `B` is NOT bound through the arena binding). For these precisions the
/// arena window must cover only the activation + output, never the weight.
pub(crate) fn matmul_b_from_f16(precision: MatmulCompute, b_is_param: bool) -> bool {
    // The dense attention route's form reads B from its own f16 operand buffer —
    // a side buffer, so it is never the arena's to cover — whether or not a weight
    // operand is involved (it has none).
    precision == MatmulCompute::CoopF16F32AccScratch
        || (b_is_param
            && matches!(
                precision,
                MatmulCompute::F16
                    | MatmulCompute::Coop16
                    | MatmulCompute::CoopF16Vk
                    | MatmulCompute::CoopF16F32Acc
            ))
}

/// True when a matmul reads its weight `B` from the packed-BF16 side buffer
/// (`bf16_weight_buffer`) instead of the arena — so the arena window must
/// cover only the activation + output, never the weight.
pub(crate) fn matmul_b_from_packed_bf16(precision: MatmulCompute, b_is_param: bool) -> bool {
    b_is_param && precision == MatmulCompute::Bf16Packed
}

/// The geometry `matmul_coop_f16_f32acc` admits, `(m, n, k)`: `m % 64 == 0`,
/// `n % 64 == 0` and `k % 32 == 0`. Two refusals, both geometric rather than
/// numeric:
///
/// * a cooperative store cannot be masked, so a partial 64×64 output tile would
///   write past the tensor — `m` and `n` must be whole tiles;
/// * a partial K stage would need a bounds test per staged element on every tile
///   of every call, so `k` must be whole K stages.
///
/// A multiply off that grid keeps the kernel the cascade would have given it: it
/// is not rounded, fused or written through a scalar path here. This answers for
/// the *kernel*; whether the device can build it and whether it computes `a·b`
/// are separate (`matmul_coop_f16_f32acc_kernel` and `crate::coop_probe`).
pub fn coop_f16_f32acc_admits(m: u64, n: u64, k: u64) -> bool {
    m.is_multiple_of(64) && n.is_multiple_of(64) && k.is_multiple_of(32)
}

/// Whether the crate's absolute f32 pin (`RLX_WGPU_MATMUL_F32_ONLY`) is in
/// force: every 16-bit matrix route stands down, including the cooperative
/// f16-operand form. Read live, as the lowering reads it (`lower.rs`'s
/// `coop_ok`), not from a config snapshot — a pin set after the snapshot must
/// still stand the route down. Exported for callers that *report* which route
/// a multiply takes: a witness that says "16-bit" while this is true is false.
#[must_use]
pub fn f32_matmul_pinned() -> bool {
    rlx_ir::env::flag("RLX_WGPU_MATMUL_F32_ONLY")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predicate is a live read of the switch the lowering reads — not of a
    /// config snapshot — so a pin set after the snapshot still stands the route
    /// down. The override guard restores whatever the process had when it drops.
    #[test]
    fn f32_matmul_pinned_tracks_the_live_switch() {
        let _pin = rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_MATMUL_F32_ONLY", "1")));
        assert!(f32_matmul_pinned());
        drop(_pin);
        let _off = rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_MATMUL_F32_ONLY", "0")));
        assert!(!f32_matmul_pinned());
    }
}
