// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OpKinds this backend claims for legalization (`Backend::supported_ops`).
//!
//! Source of truth for the coverage matrix in `docs/op-coverage.md`.
//! Kept in the backend crate so adding an op is a local edit, not a change
//! to `rlx-runtime`'s mega-`backend.rs`.

use rlx_ir::OpKind;

pub const SUPPORTED_OPS: &[OpKind] = &[
    OpKind::Input,
    OpKind::Param,
    OpKind::Constant,
    OpKind::Activation,
    OpKind::Cast,
    OpKind::StopGradient,
    OpKind::Binary,
    OpKind::Compare,
    OpKind::Where,
    OpKind::Fma,
    OpKind::ElementwiseRegion,
    OpKind::TransformRegion,
    OpKind::BatchElementwiseRegion,
    OpKind::MatMul,
    OpKind::DotGeneral,
    OpKind::LayerNorm,
    OpKind::LayerNorm2d,
    OpKind::GroupNorm,
    OpKind::ResizeNearest2x,
    OpKind::Interpolate3d,
    OpKind::RmsNorm,
    OpKind::Attention,
    OpKind::AttentionBackward,
    OpKind::RmsNormBackwardInput,
    OpKind::RmsNormBackwardGamma,
    OpKind::RmsNormBackwardBeta,
    // LayerNorm backward family:
    //   * Input  — single workgroup-per-row fused kernel.
    //   * Gamma  — two-dispatch (partial + reduce) that uses a tail
    //              scratch zone in the arena to hold per-chunk
    //              partial sums; the reduce kernel sums them.
    // Both beat the autodiff-decomposed primitive chain.
    OpKind::LayerNormBackwardInput,
    OpKind::LayerNormBackwardGamma,
    OpKind::RopeBackward,
    OpKind::CumsumBackward,
    OpKind::GatherBackward,
    // Native host conv backward (D2H→CPU→H2D). Kept out of the autodiff
    // decompose set: the static im2col-gather decomposition corrupts the
    // grad on wgpu when the conv input is a runtime input (arena aliasing);
    // the host kernels read the operands from the arena once and are correct.
    // Weight-grad + input-grad together = full multi-conv CNN training.
    OpKind::Conv2dBackwardWeight,
    OpKind::Conv2dBackwardInput,
    OpKind::Conv3dBackwardWeight,
    OpKind::Conv3dBackwardInput,
    OpKind::MaxPool3dBackward,
    OpKind::Rope,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Narrow,
    OpKind::Concat,
    OpKind::KvAppend,
    OpKind::Expand,
    OpKind::Gather,
    OpKind::Reverse,
    OpKind::Reduce,
    OpKind::Softmax,
    OpKind::SoftmaxCrossEntropy,
    OpKind::ArgMax,
    OpKind::ArgMin,
    OpKind::Cumsum,
    OpKind::CumProd,
    OpKind::CumMax,
    OpKind::TopK,
    OpKind::Sample,
    OpKind::Conv,
    OpKind::Im2Col,
    OpKind::Pool,
    OpKind::GroupedMatMul,
    OpKind::ScaledGroupedMatMul,
    OpKind::DequantGroupedMatMul,
    OpKind::DequantGroupedMatMulMlx,
    OpKind::DequantMoEWeights,
    OpKind::ScatterAdd,
    OpKind::ScatterNd,
    OpKind::ScatterElements,
    OpKind::GatherNd,
    OpKind::GatherElements,
    OpKind::SelectiveScan,
    OpKind::Lstm,
    OpKind::Gru,
    OpKind::Rnn,
    OpKind::Mamba2,
    // Keep GDN fused: unrolling into Expand/MatMul at legalize explodes the
    // arena (~100 GiB for Bonsai-27B @ max_seq=96). Host-eval via gdn_host,
    // matching Metal/CUDA/MLX claim-then-host pattern.
    OpKind::GatedDeltaNet,
    // Transposed conv (vision U-Net decoder) — a native device step either way:
    // the phase-decomposed tiled GEMM for the unit-height, unit-batch, ungrouped
    // `k ≤ 2·s` family the decoder's forms are all in, the gather for every other
    // shape (see `conv_transpose2d_tiled.wgsl`), with the CPU-kernel host route
    // kept behind `RLX_WGPU_CONV_HOST` / the structure-host preference.
    OpKind::ConvTranspose2d,
    // 3-D convs (volumetric UNETR-style decoders) — CPU NCDHW kernels.
    OpKind::Conv3d,
    OpKind::ConvTranspose3d,
    OpKind::DequantMatMul,
    OpKind::FusedMatMulBiasAct,
    OpKind::FusedResidualLN,
    OpKind::FusedResidualRmsNorm,
    OpKind::AdaLayerNorm,
    OpKind::GatedResidual,
    OpKind::AdaLayerNormBackward,
    OpKind::GatedResidualBackward,
    OpKind::FusedSwiGLU,
    OpKind::FusedAttentionBlock,
    OpKind::FusedTransformerLayer,
    // Native FFT (WGSL radix-2): f32 only, power-of-2 N ≤ 1024.
    // Anything outside that envelope panics at lowering with a
    // "pin to Device::Cpu" hint. No host fallback — WGPU has no
    // unified memory, so silent CPU round-trip would be a hidden
    // performance cliff.
    OpKind::Fft,
    // Fixed-point `Op::FftQ`, host fallback. The arena stores integers as f32
    // values, so the adapter converts at the boundary; exact to 2^24 and loud
    // past it (see rlx_gpu_host::run_fft1d_q_valued).
    OpKind::FftQ,
    // Op::Scan (arbitrary-body recurrence) via readback host fallback —
    // compile the body once, loop it on the CPU against an arena readback.
    // Enables IIR (`biquad`/`sosfilt`) on wgpu.
    OpKind::Scan,
    OpKind::ScanBackward,
    OpKind::ScanBackwardXs,
    OpKind::LogMel,
    OpKind::LogMelBackward,
    OpKind::WelchPeaks,
    // 3D Gaussian splat: native Metal / CPU reference per backend.
    OpKind::GaussianSplatRender,
    OpKind::GaussianSplatRenderBackward,
    OpKind::GaussianSplatPrepare,
    OpKind::GaussianSplatRasterize,
    OpKind::Custom,
    OpKind::RngNormal,
    OpKind::RngUniform,
    // Core Riemannian / SPD-manifold ops — no WGSL eigen kernel, so they
    // host-fallback to `rlx_cpu::spd` (F64) via an arena readback. See
    // `rlx_wgpu::spd_host`.
    OpKind::BiMap,
    OpKind::ReEig,
    OpKind::LogEig,
    OpKind::SpdBatchNorm,
    OpKind::SpdKarcherMean,
    OpKind::SpdKarcherMeanWeighted,
    OpKind::SpdLogMap,
    OpKind::SpdExpMap,
    OpKind::SpdParallelTransport,
    OpKind::SpdMatrixFnBatch,
    OpKind::ReEigBackward,
    OpKind::LogEigBackward,
    OpKind::SpdBatchNormBackwardX,
    OpKind::SpdBatchNormBackwardG,
    OpKind::SpdLogMapBackward,
    OpKind::SpdExpMapBackward,
    OpKind::SpdParallelTransportBackward,
    OpKind::SpdMatrixFnBatchBackward,
    OpKind::Eigh,
    OpKind::EighBackward,
    OpKind::EighBatch,
    OpKind::EighBatchBackward,
    // Training / vision (native WGSL where listed; else host / HostOp).
    OpKind::AxialRope2d,
    OpKind::MaxPool2dBackward,
    OpKind::SoftmaxCrossEntropyWithLogits,
    OpKind::SoftmaxCrossEntropyBackward,
    OpKind::GroupNormBackwardInput,
    OpKind::GroupNormBackwardGamma,
    OpKind::GroupNormBackwardBeta,
    OpKind::BatchNormInference,
    OpKind::BatchNormInferenceBackwardInput,
    OpKind::BatchNormInferenceBackwardGamma,
    OpKind::BatchNormInferenceBackwardBeta,
    // Native activation_backward.wgsl (Fixed kinds, op id 0–16).
    OpKind::ReluBackward,
    OpKind::ActivationBackward,
    // Scaled FP8 (Metal-parity host).
    OpKind::ScaledMatMul,
    OpKind::ScaledQuantize,
    OpKind::ScaledQuantScale,
    OpKind::ScaledDequantize,
    // Native FakeQuantize Fixed/PerBatch; EMA + LSQ/Backward stay HostOp.
    OpKind::FakeQuantize,
    OpKind::FakeQuantizeBackward,
    OpKind::FakeQuantizeLSQ,
    OpKind::FakeQuantizeLSQBackwardX,
    OpKind::FakeQuantizeLSQBackwardScale,
    OpKind::Quantize,
    OpKind::Dequantize,
    OpKind::QMatMul,
    OpKind::QConv2d,
    OpKind::DenseSolve,
    OpKind::BatchedDenseSolve,
    // Host-staged to CPU LAPACK (potrf / trsm / getrf) via the `Step::HostOp`
    // catch-all in `compile/lower.rs`, same as DenseSolve.
    OpKind::Cholesky,
    OpKind::TriangularSolve,
    OpKind::Det,
    OpKind::LogDet,
    // Sort / ArgSort host-stage to CPU (stable strided sort) via the
    // `Step::HostOp` catch-all in `compile/lower.rs`, same as Det / LogDet.
    OpKind::Sort,
    OpKind::Svd,
    OpKind::Qr,
    OpKind::ArgSort,
    // Native WGSL C64 Wirtinger surface (`complex_wirtinger.wgsl`).
    OpKind::ComplexNormSq,
    OpKind::ComplexNormSqBackward,
    OpKind::Conjugate,
    // Native WGSL ternary-pruned FFT butterfly (`fft_butterfly_stage.wgsl`).
    OpKind::FftButterflyStage,
    // Remaining QAT / INT8 / fuse forms (HostOp → CPU).
    OpKind::LoraMatMul,
    OpKind::FusedConvBiasAct,
    // NOT PartitionedConv: the HostOp path returned zeros (cpu=0.4 vs gpu=0),
    // and declining the claim lets the shared unfuse decompose it into the
    // rfft → complex-GEMM → irfft primitives it already has WGSL kernels for —
    // which runs on the GPU instead of staging back to the host.
    OpKind::CustomFn,
    // Session + WgpuExecutable compile run LowerControlFlow first.
    OpKind::If,
    OpKind::While,
];

/// Does this op get handed to rlx-cpu through the generic `Step::HostOp`
/// fallback instead of a WGSL kernel?
///
/// Exposed so the routing can be checked from outside the crate **without a
/// GPU**, by `rlx-runtime/tests/host_fallback_never_nops.rs`. The composition
/// it guards is invisible to any single-crate test: [`SUPPORTED_OPS`] claims an
/// op (so nothing upstream expands it), no kernel lowers it, this returns true,
/// and rlx-cpu has no thunk arm for it either — the result is a `Thunk::Nop`
/// over a zeroed slot, with no panic and no unsupported-op error.
///
/// wgpu has already paid for this once: the `PartitionedConv` below is still
/// routed here, and the note next to it in `SUPPORTED_OPS` records what that
/// cost (`cpu=0.4 vs gpu=0`). Dropping the claim is what fixed it, so the route
/// is harmless *today* — and one line re-adding the claim would bring the zeros
/// straight back. That line is what this predicate exists to catch.
///
/// **Scope.** This is the *generic* host route only. The op-specific
/// `Step::*Host` steps (`ConcatHost`, `FftHost`, `GruHost`, …) each name their
/// own rlx-cpu entry point, so a missing arm there is a compile error rather
/// than a silent Nop, and none of them carry a `NO_THUNK_ARM` kind.
pub fn routes_to_cpu_host(op: &rlx_ir::Op) -> bool {
    use rlx_ir::Op;
    matches!(
        op,
        Op::BatchNormInference { .. }
            | Op::BatchNormInferenceBackwardInput { .. }
            | Op::BatchNormInferenceBackwardGamma { .. }
            | Op::BatchNormInferenceBackwardBeta
            | Op::FakeQuantizeBackward { .. }
            | Op::FakeQuantizeLSQ { .. }
            | Op::FakeQuantizeLSQBackwardX { .. }
            | Op::FakeQuantizeLSQBackwardScale { .. }
            // The native low-precision / INT8 family (`scaled_lowp.wgsl`,
            // `quant_i8.wgsl`, `q_matmul.wgsl`, `q_conv2d.wgsl`) keeps this
            // route as a RESIDUAL, and listing it is load-bearing rather than
            // conservative: those kernels address the arena with absolute
            // offsets, so their lowering arms are guarded by
            // `arena_binds_whole_at_zero`, and a failed guard falls through to
            // the catch-all below. Drop them from this list and that
            // fall-through becomes a no-kernel panic.
            //
            // Quantize / Dequantize additionally fall back when `chan_dim`
            // exceeds `QUANT_I8_MAX_CHAN` (the affine table stops fitting the
            // uniform).
            | Op::ScaledMatMul { .. }
            | Op::ScaledQuantize { .. }
            | Op::ScaledQuantScale { .. }
            | Op::ScaledDequantize { .. }
            | Op::QMatMul { .. }
            | Op::QConv2d { .. }
            | Op::Quantize { .. }
            | Op::Dequantize { .. }
            | Op::DenseSolve
            | Op::BatchedDenseSolve
            | Op::Cholesky
            | Op::TriangularSolve { .. }
            | Op::Det
            | Op::LogDet
            | Op::Sort { .. }
            | Op::Svd { .. }
            | Op::Qr { .. }
            | Op::ArgSort { .. }
            | Op::LoraMatMul { .. }
            | Op::PartitionedConv { .. }
            | Op::CustomFn { .. }
    )
}

/// OpKinds whose GPU lowering hands the activation arena to its kernel by
/// ABSOLUTE offset — the logical byte offset the planner assigned, with no
/// rebase against the bound window.
///
/// # The rule
/// `lower::should_host_at_ceiling` sends any op named here to the engine's host
/// route (`Step::HostOp`) whenever the memory it addresses cannot be bound whole at
/// base zero, and the host route reads each operand by absolute byte range
/// (resolving weight-buffer tags) and writes the output, shard-aware — so the answer
/// is right and only these ops pay. The one exception is a carry-bearing
/// `Lstm`/`Gru`/`Rnn`: the host step carries a single output offset and cannot write
/// state back into an input, so the narrow-kernel form of it is refused instead of
/// hosted, while the wider form keeps its own whole-arena host step (see
/// [`carry_lowering`]).
/// The condition is the coarse one, decided at plan time; the rule's own doc states
/// the two geometries it reads and why it never inspects an op's window base.
///
/// # Closed by default
/// The guarantee that a *new* absolute-offset arm cannot silently reintroduce the bug
/// does not rest on this list staying complete. Every such arm that binds the arena
/// binds through `bind_op_window`, the one bind path that does not hand back the
/// window base (an arm that could rebase would use the base-returning helpers), and
/// that path refuses anything but the whole arena at base zero; the host-step arms
/// (the wide `Lstm`/`Gru`/`Rnn` and `Op::Fft`'s fallback) bind nothing and are
/// refused in-arm instead (see [`carry_lowering`]). An unlisted arm is therefore
/// refused loudly at plan time instead of dispatching through a re-based window;
/// listing it is what turns that refusal into the faster host route. The same
/// decision point refuses a listed op that [`has_generic_cpu_route`] rejects — the
/// CPU's own claim list plus its thunk table, i.e. what that route can compute.
///
/// # Inventory
/// Measured wrong before this rule existed (arena over the per-binding ceiling):
/// `Op::Cumsum` last-axis (uniforms `in_off`/`out_off`, `Step::Cumsum`) returned
/// all-zero; `Op::Sample` greedy (`Step::Argmax`) returned 0 where the logits
/// held the max. The rest are the same shape — an absolute `arena.offset(id)`
/// or `/ 4` word offset in the uniform, bound through an output window:
///   `AttentionBackward` (q/k/v/dy/output + mask),
///   `Pool` (1d/2d/3d), `Cumsum`/`CumProd`/`CumMax` last-axis, `WelchPeaks`
///   (GPU branch), `SelectiveScan`, `Mamba2`, `Gru`, `Rnn`, `Lstm` — keyed on op
///   kind, so it also captures the wider (`hidden > 256`) branches that lower to
///   `Step::GruHost`/`RnnHost`/`Lstm`; those wider, carry-bearing forms keep that
///   host step, which the rule refuses when its geometry carries a parked operand
///   or a byte offset past 32 bits (see [`carry_lowering`]),
///   `GroupedMatMul`, `ScaledGroupedMatMul`, `TopK`, `ScatterAdd`,
///   `RmsNormBackwardInput/Gamma/Beta`, `LayerNormBackwardInput`/`Gamma`,
///   `RopeBackward`, `CumsumBackward`, `GatherBackward`.
///
/// The low-precision family listed after them is here for the rule's parked-operand
/// half only: those arms already ask whether the *arena* binds whole, so naming them
/// changes nothing there. But they pass `arena.offset(w) as u32` to a kernel bound to
/// the act arena, and for a packed weight that offset is tagged — the tag vanishes in
/// the cast and the kernel reads a plausible activation instead. `from_plan_split`
/// parks every packed U8/I8 param whenever it runs, which is what makes the parked
/// half reach them.
///
/// Exempt, because they ask the rule's own question inside the arm and route to
/// the host path when it fails: the GPU sub-paths of `Op::DequantMatMul` (guarded
/// in-arm, because the GGUF/Int8 sub-paths of that same arm bind the weight
/// buffer directly and are correct with a parked weight — its remaining quantized
/// sub-path routes to the host path only for an operand that route can stage and
/// refuses a packed weight loudly instead), and `Op::Fft` (its GPU
/// step binds the entire buffer, so its arm asks the same whole-arena-at-zero
/// question the rule asks, and refuses a parked operand, which neither its GPU
/// nor its host step can address).
///
/// `OpKind::Conv` is absent from this list, and no conv arm needs the parked
/// half of the rule: the conv routes this crate enables by default all switch on
/// the parked-weight test and keep the previous route — the tiled or direct conv
/// kernel — which stages that weight exactly as it did before those routes
/// existed, and the im2col sub-path asks the *arena* half in-arm
/// (`arena_whole_arena_bind`) like the rest. Staging it *into the GEMM route* is
/// not the remedy: that route addresses its weight by a raw
/// `arena.offset(w) / 4` word offset, whose 32-bit cast drops `WEIGHT_BUF_TAG`
/// (the kernel would read plausible activations), and its `col` panel occupies
/// the scratch a staged copy would need. Same for the other two conv routes this
/// crate now enables by default (the mirrored 1-D tiled lift and the 1×1
/// pointwise GEMM, both switching on the same parked-weight test): they too keep
/// the previous route.
///
/// `Op::FftQ` is the same class from the host
/// side: it has no GPU step, its only step (`FftQHost`) carries 32-bit byte
/// offsets and resolves no parked tag, so its arm refuses a parked operand and an
/// arena past that byte range instead of routing the geometry to the rule. The
/// `Lstm`/`Gru`/`Rnn` host steps and `Op::Fft`'s host step are the reason the
/// bind-time net above covers the *windowed* arms: a host step binds nothing and
/// carries its own offsets, so it is guarded in-arm instead.
///
/// `Op::Custom` is absent too, and for a different reason: a custom WGSL kernel
/// binds the window relative to the op's operands by contract (see
/// `crate::wgpu_gpu_custom`), so it re-bases like any other windowed kernel.
/// The one absolute path under it — the built-in `umap.knn` GPU step — carries
/// its own guard in the lowering arm.
pub const ABSOLUTE_OFFSET_KINDS: &[OpKind] = &[
    OpKind::AttentionBackward,
    OpKind::Sample,
    OpKind::Pool,
    OpKind::Cumsum,
    OpKind::CumProd,
    OpKind::CumMax,
    OpKind::WelchPeaks,
    OpKind::SelectiveScan,
    OpKind::Mamba2,
    OpKind::Gru,
    OpKind::Rnn,
    OpKind::Lstm,
    OpKind::GroupedMatMul,
    OpKind::ScaledGroupedMatMul,
    OpKind::TopK,
    OpKind::ScatterAdd,
    OpKind::RmsNormBackwardInput,
    OpKind::RmsNormBackwardGamma,
    OpKind::RmsNormBackwardBeta,
    OpKind::LayerNormBackwardInput,
    OpKind::LayerNormBackwardGamma,
    OpKind::RopeBackward,
    OpKind::CumsumBackward,
    OpKind::GatherBackward,
    // Parked-parameter half only; see the note above the list.
    OpKind::BatchNormInference,
    OpKind::BatchNormInferenceBackwardInput,
    OpKind::BatchNormInferenceBackwardGamma,
    OpKind::BatchNormInferenceBackwardBeta,
    OpKind::QMatMul,
    OpKind::QConv2d,
    OpKind::ScaledMatMul,
    OpKind::ScaledQuantize,
    OpKind::ScaledQuantScale,
    OpKind::ScaledDequantize,
    OpKind::Quantize,
    OpKind::Dequantize,
];

/// Is `op`'s lowering one of the absolute-offset family
/// ([`ABSOLUTE_OFFSET_KINDS`])?
pub fn addresses_arena_by_absolute_offset(op: &rlx_ir::Op) -> bool {
    ABSOLUTE_OFFSET_KINDS.contains(&op.kind())
}

/// Can the engine's generic host route (`Step::HostOp` → rlx-cpu) run this op?
///
/// The authority is the CPU backend itself: rlx-cpu **claims** the kind and has a
/// real **thunk** for it (`rlx_cpu::SUPPORTED_OPS` minus `rlx_cpu::NO_THUNK_ARM`).
/// That is what closes the route by default — an op the CPU cannot run is false here
/// even if it was added to [`ABSOLUTE_OFFSET_KINDS`], and the rule refuses it loudly
/// instead of dispatching it to a silent `Thunk::Nop` (which is what the catch-all
/// thunk arm produces for a claimed-but-unthunked kind, the failure mode
/// `rlx_cpu::NO_THUNK_ARM` names).
///
/// Derived rather than restated as its own list: a hand-written copy of
/// [`ABSOLUTE_OFFSET_KINDS`] cannot come out the other way, so it is not a closure.
/// The union this predicate computes is checked against the family in
/// `host_route_closure_covers_the_family`.
pub fn has_generic_cpu_route(kind: OpKind) -> bool {
    rlx_cpu::SUPPORTED_OPS.contains(&kind) && !rlx_cpu::NO_THUNK_ARM.contains(&kind)
}

/// `hidden_size` at or below which `Lstm`/`Gru`/`Rnn` lower to the narrow
/// workgroup kernel; wider goes to the op's own whole-arena host step. Shared
/// with those three lowering arms so [`carry_lowering`] cannot drift from them.
pub const NARROW_RNN_HIDDEN_MAX: u32 = 256;

/// How a carry-bearing `Lstm`/`Gru`/`Rnn` stands with respect to the ceiling rule.
#[derive(Debug, PartialEq, Eq)]
pub enum CarryLowering {
    /// Not a carry-bearing recurrent op: nothing is written back into an input.
    NotCarrying,
    /// `hidden_size <= NARROW_RNN_HIDDEN_MAX`. The narrow kernel writes the final
    /// state into a state input *through the windowed bind*, and the generic host
    /// route cannot take over: `HostOpDesc` carries one `out_byte_off` and no field
    /// can express an input mutation, so hosting returns a plausible `y` while the
    /// state silently never advances. `should_host_at_ceiling` refuses it.
    WindowedKernel,
    /// Wider than that bound. The op lowers to its OWN whole-arena host step
    /// (`Step::Lstm`/`GruHost`/`RnnHost` → `vision_host`), which mirrors the
    /// activation arena, does write the state back, and handles a striped arena.
    /// It does **not** carry a parked operand or a byte offset past 32 bits: its
    /// step fields are 32-bit, so a parameter parked in the separate weight buffer
    /// (whose `WEIGHT_BUF_TAG` vanishes in the cast) or an arena past
    /// `u32::MAX` bytes is read from the wrong place silently. The rule refuses
    /// those two geometries loudly rather than leave this step to them.
    WholeArenaHostStep,
}

/// Classify a carry-bearing recurrent op for the ceiling rule.
pub fn carry_lowering(op: &rlx_ir::Op) -> CarryLowering {
    use rlx_ir::Op;
    let (hidden_size, carry) = match op {
        Op::Lstm {
            hidden_size, carry, ..
        }
        | Op::Gru {
            hidden_size, carry, ..
        }
        | Op::Rnn {
            hidden_size, carry, ..
        } => (*hidden_size, *carry),
        _ => return CarryLowering::NotCarrying,
    };
    if !carry {
        return CarryLowering::NotCarrying;
    }
    if hidden_size as u32 <= NARROW_RNN_HIDDEN_MAX {
        CarryLowering::WindowedKernel
    } else {
        CarryLowering::WholeArenaHostStep
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Op;

    /// Pure and device-free: `cargo test -p rlx-wgpu --lib`.
    #[test]
    fn carry_lowering_splits_on_the_narrow_kernel_bound() {
        let rnn = |hidden_size: usize, carry: bool| {
            [
                Op::Lstm {
                    hidden_size,
                    num_layers: 1,
                    bidirectional: false,
                    carry,
                },
                Op::Gru {
                    hidden_size,
                    num_layers: 1,
                    bidirectional: false,
                    carry,
                },
                Op::Rnn {
                    hidden_size,
                    num_layers: 1,
                    bidirectional: false,
                    carry,
                    relu: false,
                },
            ]
        };
        // Carry on the narrow kernel: nothing here can write the state back.
        for op in rnn(NARROW_RNN_HIDDEN_MAX as usize, true) {
            assert_eq!(carry_lowering(&op), CarryLowering::WindowedKernel, "{op:?}");
        }
        // Wider carry: the op's own whole-arena host step takes it.
        for op in rnn(NARROW_RNN_HIDDEN_MAX as usize + 1, true) {
            assert_eq!(
                carry_lowering(&op),
                CarryLowering::WholeArenaHostStep,
                "{op:?}"
            );
        }
        // No carry at all, at either width.
        for op in rnn(NARROW_RNN_HIDDEN_MAX as usize, false)
            .into_iter()
            .chain(rnn(1024, false))
        {
            assert_eq!(carry_lowering(&op), CarryLowering::NotCarrying, "{op:?}");
        }
        // Other listed kinds are not recurrent at all.
        for op in [
            Op::Cumsum {
                axis: -1,
                exclusive: false,
            },
            Op::Sample {
                top_k: 0,
                top_p: 1.0,
                temperature: 1.0,
                seed: 0,
            },
            Op::GroupedMatMul,
        ] {
            assert_eq!(carry_lowering(&op), CarryLowering::NotCarrying, "{op:?}");
        }
    }

    /// The closure that decides the rule: every kind it may hand to the generic host
    /// route must be one that route is actually known to run — the CPU's own claim list
    /// together with a real thunk for each, which is exactly what
    /// [`has_generic_cpu_route`] asks, so these two assertions pin the whole predicate.
    ///
    /// Without it, adding a kind to [`ABSOLUTE_OFFSET_KINDS`] alone makes the rule
    /// refuse that kind loudly at plan time rather than route it — which is only
    /// survivable because that refusal is what a caller sees, instead of a
    /// `Step::HostOp` that writes a zeroed output slot for a kind rlx-cpu has no thunk
    /// for.
    #[test]
    fn host_route_closure_covers_the_family() {
        for kind in ABSOLUTE_OFFSET_KINDS {
            assert!(
                rlx_cpu::SUPPORTED_OPS.contains(kind),
                "{kind:?} is in the absolute-offset family but rlx-cpu does not claim it — \
                 the rule would refuse it loudly; either the CPU route gains the kind, or the \
                 op is handled explicitly instead of being handed to this route"
            );
            assert!(
                !rlx_cpu::NO_THUNK_ARM.contains(kind),
                "{kind:?} is in the absolute-offset family and rlx-cpu claims it, but has no \
                 thunk arm for it (it expands it away) — Step::HostOp would write a zeroed \
                 output slot"
            );
        }
    }
}
