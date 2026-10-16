// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph lowering match + schedule emission.

#![allow(unused_imports)]
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `compile` — extracted from the `backend` module for navigability (see `mod.rs`).

use super::super::arena_binds_whole_at_zero;
use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ActivationBackwardParams, AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams,
    AttentionBwdParams, AttentionMaskParams, AttentionParams, AxialRope2dParams,
    BatchElementwiseRegionParams, BatchNormInferenceParams, BinaryC64Params, BinaryParams,
    CastParams, ComplexCastParams, ComplexWirtingerParams, Conv1dParams, Conv2dParams,
    Conv3dBwdInputParams, Conv3dBwdWeightParams, Conv3dParams, CopyParams, CumScanParams,
    CumsumBwdParams, CumsumParams, DequantMatmulMlxParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FakeQuantizeParams, FftButterflyStageParams, FmaParams,
    FramesParams, FusedConvBiasActParams, FusedResidualLnParams, FusedResidualLnTeeParams,
    FusedResidualRmsNormParams, FusedSwiGLUParams, GatedDeltaNetParams,
    GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupNormBwdParams, GroupNormParams, GroupedMatmulParams, GruParams,
    Im2Col2dParams, IndexingNdParams, Kernel, LayerNorm2dParams, LayerNormBwdParams,
    LayerNormParams, LstmParams, Mamba2Params, MatmulParams, MatmulQkvParams, MaxPool2dBwdParams,
    MaxPool3dBwdParams, NarrowConcatParams, Pool1dParams, Pool2dParams, Pool3dParams,
    QConv2dParams, QMatMulParams, QUANT_I8_MAX_CHAN, QuantI8Params, ReduceParams, ReverseParams,
    RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams, SampleParams,
    ScaledGroupedMatmulParams, ScaledLowpParams, ScatterAddParams, SceBwdParams, SceParams,
    SelectiveScanParams, SoftmaxParams, TopKParams, TransposeParams, UmapKnnParams, UnaryParams,
    WelchPeaksGpuParams, WhereParams, activation_backward_kernel, ada_layer_norm_backward_kernel,
    ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel, attention_kernel,
    attention_mask_kernel, axial_rope2d_kernel, batch_elementwise_region_kernel,
    batch_norm_inference_bwd_beta_kernel, batch_norm_inference_bwd_gamma_kernel,
    batch_norm_inference_bwd_input_kernel, batch_norm_inference_kernel, binary_c64_kernel,
    binary_kernel, cast_f32_to_f16_kernel, cast_kernel, compare_kernel, complex_cast_kernel,
    complex_norm_sq_backward_kernel, complex_norm_sq_kernel, concat_kernel, conjugate_c64_kernel,
    conv_transpose2d_kernel, conv_transpose3d_kernel, conv1d_kernel, conv1d_tiled_kernel,
    conv2d_kernel, conv3d_backward_input_kernel, conv3d_backward_weight_kernel, conv3d_kernel,
    copy_kernel, copy_sanitize_kernel, cum_scan_kernel, cumsum_backward_kernel, cumsum_kernel,
    dequant_matmul_kernel, dequant_matmul_mlx_kernel, dequantize_i8_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel,
    fake_quantize_fixed_kernel, fake_quantize_perbatch_kernel, fft_butterfly_stage_kernel,
    fma_kernel, frames_kernel, fused_conv_bias_act_kernel, fused_residual_ln_kernel,
    fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel, fused_swiglu_kernel,
    gated_delta_net_kernel, gated_residual_backward_kernel, gated_residual_kernel,
    gather_axis_kernel, gather_backward_acc_kernel, gather_backward_zero_kernel,
    gather_elements_kernel, gather_kernel, gather_nd_kernel, gather_split_kernel,
    group_norm_backward_beta_kernel, group_norm_backward_gamma_kernel,
    group_norm_backward_input_kernel, group_norm_kernel, grouped_matmul_kernel, gru_kernel,
    im2col2d_kernel, layer_norm_backward_gamma_partial_kernel,
    layer_norm_backward_gamma_reduce_kernel, layer_norm_backward_input_kernel, layer_norm2d_kernel,
    layernorm_kernel, lead_pack_uniform, lstm_kernel, mamba2_kernel, matmul_bf16w_kernel,
    matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, maxpool2d_backward_kernel,
    maxpool3d_backward_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel,
    q_conv2d_kernel, q_matmul_kernel, quantize_i8_kernel, reduce_kernel, reverse_kernel,
    rms_norm_backward_kernel, rms_norm_backward_param_kernel, rnn_kernel, rope_backward_kernel,
    rope_kernel, sample_kernel, scaled_dequantize_kernel, scaled_grouped_matmul_decode_kernel,
    scaled_matmul_decode_kernel, scaled_quant_scale_kernel, scaled_quantize_kernel,
    scatter_add_kernel, scatter_elements_kernel, scatter_nd_reduce_kernel, selective_scan_kernel,
    softmax_cross_entropy_backward_kernel, softmax_cross_entropy_kernel,
    softmax_cross_entropy_with_logits_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel, welch_peaks_gpu_kernel, where_kernel,
};
use rlx_gpu_dispatch::indexing::{KernelKind as IndexKind, Reduction as IndexReduction};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, AdaNormKind, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op, ada_modulation_launch, ada_modulation_lead_pack};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::super::*;
use crate::backend::WgpuExecutable;

/// f32 clamp bounds `[lo, hi]` for a float→int Cast destination (`Op::Cast`
/// truncate+saturate on the f32-uniform arena). Endpoints are the exact int
/// range; `i32::MAX`/`u32::MAX`/i64 bounds are not f32-representable and round
/// to the nearest f32 (an inherent arena limit at the extreme int values).
fn int_range_f32(dt: rlx_ir::DType) -> (f32, f32) {
    match dt {
        rlx_ir::DType::I8 => (i8::MIN as f32, i8::MAX as f32),
        rlx_ir::DType::I16 => (i16::MIN as f32, i16::MAX as f32),
        rlx_ir::DType::I32 => (i32::MIN as f32, i32::MAX as f32),
        rlx_ir::DType::I64 => (i64::MIN as f32, i64::MAX as f32),
        rlx_ir::DType::U8 => (0.0, u8::MAX as f32),
        rlx_ir::DType::U32 => (0.0, u32::MAX as f32),
        // Non-int dst never reaches here (guarded by is_int at the call site).
        _ => (f32::NEG_INFINITY, f32::INFINITY),
    }
}

/// Cast target-class code for a fused `ChainStep::Cast`, packed into the chain
/// step's `sub` word (op_kind==1) and decoded by `apply_cast` in
/// `elementwise_region.wgsl`. The fused chain carries only the TARGET dtype, so
/// the source is assumed float for int targets (the common case; an int→int
/// narrow inside a region would saturate rather than wrap — unrepresentable in
/// the f32 arena either way). Float targets (incl. F16/BF16 kept as f32) map to
/// identity; F64/C64 targets are rejected before this is reached.
fn cast_region_sub(dt: rlx_ir::DType) -> u32 {
    match dt {
        rlx_ir::DType::Bool => 1,
        rlx_ir::DType::I8 => 2,
        rlx_ir::DType::I16 => 3,
        rlx_ir::DType::I32 => 4,
        rlx_ir::DType::I64 => 5,
        rlx_ir::DType::U8 => 6,
        rlx_ir::DType::U32 => 7,
        _ => 0,
    }
}

/// Overwrite each fused `Cast` step's `sub` word with its target-class code
/// (see [`cast_region_sub`]). The shared `rlx_ir::encode_chain_steps` drops the
/// Cast dtype (encodes `sub=0` = identity); wgpu re-injects it so the region
/// kernel performs the real float→int / →Bool conversion.
fn patch_region_cast_subs(chain: &[rlx_ir::op::ChainStep], chain_enc: &mut [u32]) {
    for (k, step) in chain.iter().enumerate() {
        if let rlx_ir::op::ChainStep::Cast(to, _) = step {
            // F64 has no f32-arena storage — a fused F64 cast can't run here.
            if matches!(to, rlx_ir::DType::F64) {
                panic!(
                    "rlx-wgpu fused Cast->{to:?}: the f32-uniform arena has no \
                     F64 storage — run this cast on CPU/Metal"
                );
            }
            // Complex casts must never reach a fused region: `mark_elementwise`
            // excludes complex ops (the f32-scalar region kernel can't re-pair
            // `[re, im]` lanes), so they stay standalone and route through the
            // `complex_cast` pass. Reaching here means the exclusion regressed —
            // panic loudly rather than emit a silently-wrong lane-move.
            if to.is_complex() {
                panic!(
                    "rlx-wgpu fused Cast->{to:?}: complex casts must be excluded \
                     from region fusion (mark_elementwise) and lowered as a \
                     standalone complex_cast pass — this is a fusion-exclusion bug"
                );
            }
            let idx = k * 4 + 1;
            if idx < chain_enc.len() {
                chain_enc[idx] = cast_region_sub(*to);
            }
        }
    }
}

/// True when SPIR-V paths should host-fallback.
///
/// Virtually-sharded arenas (NVIDIA 2 GiB bind) force host for most ops.
/// Discrete Vulkan/DX12 no longer blankets every op through host (that was
/// ~20 s/chunk on Kitten 32 k wave graphs); instead:
/// - Elementwise Binary/Where/Fma/Activation → host on discrete (SPIR-V still
///   collapses Kitten NSF; peak ~0.05 if left on GPU) — see
///   [`should_pack_host_op`].
/// - Structural Expand/Concat/Transpose/Narrow/Conv → host on discrete — see
///   [`wgpu_prefer_structure_host`] / [`wgpu_prefer_conv_host`].
/// Metal keeps GPU kernels. Opt out of hosting with `RLX_WGPU_SHARD_GPU=1`.
/// Force full discrete hosting with `RLX_WGPU_DISCRETE_HOST=1` (via
/// [`wgpu_prefer_host_fallback`] for norms/sharded paths). `LayerNorm2d` and
/// `Reverse` do not consult the adapter-kind term at all — see
/// [`wgpu_prefer_host_fallback_agnostic`].
fn wgpu_prefer_host_fallback(arena: &crate::buffer::Arena) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_FORCE_HOST") {
        return true;
    }
    if rlx_ir::env::flag("RLX_WGPU_SHARD_GPU") {
        return false;
    }
    if arena.is_sharded() {
        return true;
    }
    rlx_ir::env::flag("RLX_WGPU_DISCRETE_HOST") && crate::device::coop_discrete_backend()
}

/// [`wgpu_prefer_host_fallback`] without its adapter-kind term.
///
/// The `RLX_WGPU_DISCRETE_HOST`-on-a-discrete-adapter test exists for
/// operations whose SPIR-V path is numerically wrong there. `LayerNorm2d` and
/// `Reverse` are not in that set — their device path is exact on every adapter,
/// so consulting the adapter kind would put them back on the host on exactly
/// the adapters this backend is measured on. `RLX_WGPU_FORCE_HOST`, the sharded
/// arena and the `RLX_WGPU_SHARD_GPU` opt-out still apply.
///
/// Written out rather than derived from [`wgpu_prefer_host_fallback`]: there
/// `RLX_WGPU_FORCE_HOST` is asked before `RLX_WGPU_SHARD_GPU`, and that
/// precedence is observable (`FORCE_HOST=1 SHARD_GPU=1` hosts), so the two
/// cannot share one expression without changing an existing route.
fn wgpu_prefer_host_fallback_agnostic(arena: &crate::buffer::Arena) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_FORCE_HOST") {
        return true;
    }
    if rlx_ir::env::flag("RLX_WGPU_SHARD_GPU") {
        return false;
    }
    arena.is_sharded()
}

/// Expand / Concat / Transpose / Narrow: host on discrete Vulkan/DX12 even when
/// the arena is unsharded (GPU paths collapse Kitten NSF). Also when virtually
/// sharded / `RLX_WGPU_FORCE_HOST`.
fn wgpu_prefer_structure_host(arena: &crate::buffer::Arena) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_SHARD_GPU") {
        return false;
    }
    wgpu_prefer_host_fallback(arena) || crate::device::coop_discrete_backend()
}

/// Ops routed to packed [`Step::HostOp`] when hosting is required.
/// Expand / Conv / Transpose / Narrow / Concat / DequantMatMul / Custom keep
/// dedicated host paths. View-like Gather stays on GPU GatherSplit / axis
/// kernels unless separately hosted.
/// True when this op's operands and output do not all lie in one arena stripe.
///
/// Wraps [`crate::buffer::Arena::straddles_shards`], which is the general form
/// of the older matmul-only `arena_span_crosses_shard`. Reports once per process
/// so a striped run says why an op left the GPU instead of silently getting
/// slower.
fn op_straddles_shards(arena: &crate::buffer::Arena, node: &rlx_ir::Node, graph: &Graph) -> bool {
    let span = |id: rlx_ir::NodeId| -> Option<(usize, usize)> {
        if !arena.has(id) {
            return None;
        }
        let n = graph.node(id);
        let bytes = n
            .shape
            .num_elements()
            .unwrap_or(0)
            .saturating_mul(n.shape.dtype().size_bytes());
        Some((arena.offset(id), bytes))
    };
    let mut spans: Vec<(usize, usize)> = node.inputs.iter().filter_map(|&i| span(i)).collect();
    if let Some(o) = span(node.id) {
        spans.push(o);
    }
    let Some(bad) = arena.straddles_shards(&spans) else {
        return false;
    };
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if rlx_ir::env::flag("RLX_WGPU_STRIPE_VERBOSE") {
        eprintln!(
            "[rlx-wgpu] straddles: {:?} node {} ({} spans)",
            node.op.kind(),
            node.id.0,
            bad.len()
        );
    }
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[rlx-wgpu] note: hosting {:?} — its operands span more than one arena \
             stripe ({} spans, e.g. {:?}), which a single-stripe binding would read \
             as zeros. Reported once per process.",
            node.op.kind(),
            bad.len(),
            &bad[..bad.len().min(3)]
        );
    }
    true
}

fn should_pack_host_op(
    arena: &crate::buffer::Arena,
    op: &Op,
    node: &rlx_ir::Node,
    graph: &Graph,
) -> bool {
    // COMPLEX NEVER HOSTS.
    //
    // The host fallback packs the arena as plain f32 and replays the op through
    // rlx-cpu's real-valued kernel. For a C64 tensor that arena is interleaved
    // `[re, im]` lanes, so a complex multiply came back as the LANE-WISE product
    // `(ar*br, ai*bi)` instead of `(ar*br - ai*bi, ar*bi + ai*br)`.
    //
    // It hid well: complex ADD is lane-wise addition, so `c64_add_sub_bit_exact`
    // passed and only mul/div failed. And it only fires where this function
    // actually hosts elementwise ops — `coop_discrete_backend()`, i.e. discrete
    // NVIDIA Vulkan — so Metal-backed wgpu was always correct and the whole thing
    // read as a "Vulkan-only complex bug" in the kernel rather than a routing
    // decision made before any kernel was chosen.
    //
    // Complex ops have dedicated kernels (`binary_c64`, `complex_cast`,
    // `complex_wirtinger`) precisely because a scalar-per-lane model cannot reach
    // the partner lane. Sending them to a lane-blind path is never right.
    //
    // Verified on discrete NVIDIA Vulkan (`complex_parity`): without this
    // guard `c64_mul_div_close` and `c64_scalar_vector_broadcast_mul` both fail;
    // with it they pass. `expand_complex_materialized` fails either way — a
    // separate, pre-existing complex-Expand bug on that backend, not a routing
    // decision this can reach.
    if node.shape.dtype().is_complex()
        || node
            .inputs
            .iter()
            .any(|&i| graph.node(i).shape.dtype().is_complex())
    {
        return false;
    }
    // STRIPE-STRADDLING OPS ALWAYS HOST, even under `RLX_WGPU_SHARD_GPU`.
    //
    // A kernel binds ONE stripe. Slot placement guarantees no single tensor
    // straddles a boundary, which is not the same as guaranteeing an op's
    // *operands* share one — nothing in the planner knows which tensors are used
    // together. When they do not, the operand outside the bound window reads as
    // **zero**: no error, no warning, and a plausible-looking result. Measured:
    // a 3-op f32 chain with a 4 MiB shard cap returned 0.0 for 786432/786432
    // outputs instead of 18.0 (`shard_straddle_guard`).
    //
    // `RLX_WGPU_SHARD_GPU` asks to keep work on the GPU when the arena is
    // striped; it is a throughput preference, and it cannot be allowed to mean
    // "return zeros". So the straddle test runs BEFORE that opt-out and hosting
    // wins — the host path reads whole tensors and is stripe-agnostic, so the
    // answer is right and only this one op pays.
    //
    // The complementary case — an op whose operands and result all share one
    // stripe, but whose kernel addresses the arena by ABSOLUTE offset — is not a
    // straddle and is not caught here: `should_host_at_ceiling` runs before this
    // whole function and hosts it whenever the arena cannot be bound whole at
    // base zero, also ahead of the `RLX_WGPU_SHARD_GPU` opt-out.
    if arena.is_sharded() && op_straddles_shards(arena, node, graph) {
        return true;
    }
    if rlx_ir::env::flag("RLX_WGPU_SHARD_GPU") {
        return false;
    }
    if matches!(op, Op::MatMul) {
        return rlx_ir::env::flag("RLX_WGPU_HOST_MATMUL");
    }
    if matches!(
        op,
        Op::Softmax { .. }
            | Op::LayerNorm { .. }
            | Op::RmsNorm { .. }
            | Op::FusedResidualLN { .. }
            | Op::FusedResidualRmsNorm { .. }
    ) {
        if rlx_ir::env::flag("RLX_WGPU_GPU_NORM") {
            return false;
        }
        // Norms: host on discrete NVIDIA (unsharded Kitten still needs this)
        // and whenever the arena is virtually sharded.
        return wgpu_prefer_host_fallback(arena)
            || crate::device::coop_discrete_backend()
            || rlx_ir::env::flag("RLX_WGPU_HOST_NORM");
    }
    if matches!(
        op,
        Op::Binary(_) | Op::Where | Op::Fma | Op::Activation(_) | Op::GatedResidual
    ) {
        // Elementwise SPIR-V still collapses Kitten NSF on discrete NVIDIA
        // (peak ~0.05) even with MatMul→Binary pass flushes and Expand hosted.
        // Host on discrete / sharded; Metal keeps GPU kernels. Force GPU
        // elementwise (unsafe for Kitten) with RLX_WGPU_SHARD_GPU=1.
        return wgpu_prefer_host_fallback(arena) || crate::device::coop_discrete_backend();
    }
    false
}

/// Refuse a region carrying an activation the wgsl region kernel has no arm for
/// (`rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS`). Called from the region
/// arm that can still reach it, on purpose *after* the stripe-straddling rule
/// (`should_pack_host_op`) — the only route that carries a region off the kernel onto the
/// host step, whose region interpreter is exhaustive over the activation set, so a plain
/// region diverted there is computed correctly and must not be refused.
///
/// A *well-formed* region cannot reach here: `crate::unfuse::expand_unimplemented_regions`
/// takes every one of them first (plain and prologue `ElementwiseRegion` through the
/// unfuse, `BatchElementwiseRegion` through `DecomposeFusionRegions`; `TransformRegion`
/// carries no activation). What is left is a region that machinery declines — a
/// `BatchElementwiseRegion` whose input count contradicts its own `num_batch_inputs`,
/// which the decomposer copies through untouched. The verifier rejects such a node, but it
/// only runs with `debug_assertions` on the HIR path, so a hand-built graph compiled
/// directly can still carry one. It refuses rather than runs because the alternative is a
/// plausible wrong value.
fn assert_region_activations_implemented(node: &rlx_ir::Node, chain: &[rlx_ir::op::ChainStep]) {
    let Some(act) = rlx_ir::region_encode::first_unimplemented_activation(
        rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS,
        chain,
    ) else {
        return;
    };
    panic!(
        "rlx-wgpu: node {id} is an elementwise region whose chain carries `{act:?}` (region \
         activation code {sub}), and the region kernel this device runs has no arm for that \
         code — it falls through to `return x`, so the activation would be computed as its \
         own input. `crate::unfuse::expand_unimplemented_regions` is the expansion that \
         covers this, so reaching this arm means it could not; there is no switch on this \
         backend that makes such a region run — the graph has to be built without it.",
        id = node.id,
        sub = rlx_ir::region_encode::activation_sub(act),
    );
}

/// The per-binding-ceiling rule: one rule, decided at plan time.
///
/// An op whose kernel carries absolute arena offsets — the family (and the two
/// measured failures) in `supported_ops::addresses_arena_by_absolute_offset` — is
/// computed by the engine's host route instead whenever the memory it addresses
/// cannot be bound whole at base zero. The host route addresses memory by absolute
/// byte range (resolving weight-buffer tags) and is window-agnostic, so the answer is
/// right and only these ops pay — except a carry-bearing `Lstm`/`Gru`/`Rnn`, which
/// `supported_ops::carry_lowering` splits in two: the narrow kernel's variant is refused
/// (the host step carries a single output offset and cannot write state back into an
/// input), and the wider one keeps its own whole-arena host step, which the rule
/// leaves alone only while no operand is parked and the arena stays inside that
/// step's 32-bit byte offsets.
///
/// This runs BEFORE every other routing rule, so `RLX_WGPU_SHARD_GPU` and the
/// discrete/structural host preferences cannot undo it, and an op that is not on that
/// list is untouched. An op that IS on the list but has no host route — or is a
/// carry-bearing `Lstm`/`Gru`/`Rnn` whose geometry its own host step cannot take — is
/// refused loudly, naming the op.
///
/// Two geometries decide it, both read off the arena rather than off any op's
/// window base. The activation arena must bind whole at base zero
/// (`Arena::binds_whole_at_base_zero`). Separately, an operand parked in the
/// weight buffer settles the case on its own: `Arena::offset` reports such a
/// parameter as a `WEIGHT_BUF_TAG`-tagged offset, and these arms pass it into a
/// `u32` uniform of a kernel bound to the ACT arena, where the tag vanishes in
/// the cast and the kernel reads a plausible activation instead of the
/// parameter. Resolving a parked parameter needs `Arena::resolve_w`, staging
/// (`arena_off_in_window_or_stage`) or a separate binding, and no arm on this
/// list uses any of them — `arena_window_for_nodes` skips tagged operands for
/// exactly that reason. A *parameter's* operand the host route cannot stage — one whose
/// read is not one f32 lane per element, in what the upload stored or in the slot its plan
/// gave it ([`crate::buffer::unstageable_reason`]) — is what
/// [`assert_host_route_stageable`] refuses, wherever that parameter lives; an operand that
/// is not a parameter's storage, or a view of one, is not judged at all.
///
/// The price of that second geometry: on an adapter that parks every parameter
/// (`from_plan_split` under a small `max_buffer_size`, or
/// `RLX_WGPU_ALL_PARAMS_WEIGHT`), a listed op that reads a parameter hosts on
/// every run, ceiling or not — except for a parameter the route cannot stage,
/// which [`assert_host_route_stageable`] refuses instead.
fn should_host_at_ceiling(
    dev: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    node: &rlx_ir::Node,
) -> bool {
    let limit = dev.limits().max_storage_buffer_binding_size;
    let parked_operand = any_parked_operand(arena, node);
    if !parked_operand && arena.binds_whole_at_base_zero(limit) {
        return false;
    }
    // What the messages below name as unbindable: the parked region, or the arena.
    let unbindable = || {
        if parked_operand {
            parked_operand_report(arena)
        } else {
            geometry_report(arena, limit)
        }
    };
    let op = &node.op;
    if !crate::supported_ops::addresses_arena_by_absolute_offset(op) {
        return false;
    }
    match crate::supported_ops::carry_lowering(op) {
        crate::supported_ops::CarryLowering::WindowedKernel => panic!(
            "rlx-wgpu: REFUSED (carry state): {:?} writes its final state back into a state input \
             buffer, but the host route carries only one output offset, so the state would silently \
             never advance; this op's own kernel addresses the arena by absolute offset, so it cannot \
             be computed correctly here ({}). Avoid striping this graph, or move it to a backend that \
             can.",
            op.kind(),
            unbindable()
        ),
        // Wider than the narrow kernel's bound: the op keeps its own whole-arena host
        // step. That step mirrors the ACTIVATION arena only and carries 32-bit byte
        // offsets, so a parked operand (whose weight-buffer tag is truncated away in
        // the cast, leaving it read from the mirror) and an arena past `u32::MAX`
        // bytes (which truncates every offset) both compute the recurrence silently
        // wrong — and the generic host route cannot take over, because it cannot
        // write the state back into an input. Refuse rather than return a wrong state.
        crate::supported_ops::CarryLowering::WholeArenaHostStep => {
            assert!(
                !parked_operand && arena.size <= u32::MAX as usize,
                "rlx-wgpu: REFUSED (wide carry RNN): {:?} keeps its own whole-arena host step, \
                 which mirrors the activation arena with 32-bit byte offsets — neither a \
                 parameter parked in the weight buffer nor an arena over 2^32 bytes can be \
                 computed through it, and the generic host route cannot write its state back \
                 into an input ({}). Avoid parking this operand or shrinking the arena, or move \
                 the graph to a backend that can.",
                op.kind(),
                unbindable()
            );
            return false;
        }
        crate::supported_ops::CarryLowering::NotCarrying => {}
    }
    if crate::supported_ops::has_generic_cpu_route(op.kind()) {
        assert_host_route_stageable(graph, arena, node);
        if rlx_ir::env::flag("RLX_WGPU_STRIPE_VERBOSE") {
            eprintln!(
                "[rlx-wgpu] ceiling: hosting {:?} ({})",
                op.kind(),
                unbindable()
            );
        }
        // Reported once per process, like the stripe-straddle note: this is the
        // safe route, but a caller comparing timings must be able to see that
        // some of the graph left the GPU because of the ceiling rather than
        // guess it. `WgpuExecutable::ceiling_hosted_steps()` counts every case.
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[rlx-wgpu] note: hosting {:?} — the memory it addresses cannot be bound whole \
                 at base zero, or lives in the parked weight buffer ({}), so a kernel whose \
                 offsets are ABSOLUTE would read the wrong slots; it is computed on the host \
                 instead. Reported once per process; `ceiling_hosted_steps()` counts every case.",
                op.kind(),
                unbindable()
            );
        }
        return true;
    }
    panic!(
        "rlx-wgpu: REFUSED (per-binding ceiling): {:?} addresses memory by absolute offset that \
         cannot be bound whole at base zero ({}), and no host route is known to compute it — it \
         cannot be computed correctly here. Avoid striping this graph, or move it to a backend \
         that can.",
        op.kind(),
        unbindable()
    );
}

/// Does any operand of `node` live in the separate weight buffer?
fn any_parked_operand(arena: &Arena, node: &rlx_ir::Node) -> bool {
    node.inputs
        .iter()
        .any(|&i| crate::buffer::is_weight_off(arena.offset(i)))
}

/// Is this *parameter's* stored form one f32 lane per element — what both of
/// this op's routes read?
///
/// Exhaustive on purpose, and mirroring `set_param_typed`'s arms in
/// `crates/core/rlx-runtime/src/backend/wgpu_backend.rs` (this crate sits below
/// it): a new encoding has to be decided here rather than default to a verdict
/// nobody checked. [`crate::buffer::unstageable_reason`] decides the same set
/// for the *generic* host route, which reads no more than the four bytes per
/// element it is given, so its `F64`/`I64` verdict is the opposite of this
/// one's — a new encoding belongs in both matches, not in either. `slot_bytes` is what the arena's record says the parameter's
/// storage holds — `rlx_compile`'s `node_slot_bytes` sizes a non-F32 `Param` at
/// its native width, while the split plan's parameters get four bytes per
/// element, and the widened upload has to fit whichever it got.
fn param_lane_readable(dt: rlx_ir::DType, elems: usize, slot_bytes: usize) -> bool {
    use rlx_ir::DType::*;
    match dt {
        // `set_param_typed` widens these to one f32 lane per element.
        F32 | F16 | F64 | I32 | I64 | U32 | Bool => {}
        // Stored raw at the encoding's own one or two bytes per element, so the
        // lanes a device step would read are not the values the model wrote.
        U8 | I8 | BF16 => return false,
        // No arm in the typed upload at all: the parameter never reaches the arena.
        I16 | C64 | C128 => return false,
    }
    slot_bytes >= 4 * elems
}

/// Can every parameter operand of `node` be read as one f32 lane per element?
///
/// Activations always are — the planner widens the sub-four-byte encodings at the
/// boundary and `compile_static_inner` records the payload as `elems * 4` bytes
/// for anything that is not complex — so only an operand the graph says is a
/// parameter's storage is judged; an arena with no record for it is refused
/// ([`param_lane_readable`] never sees a slot).
///
/// [`crate::buffer::unstageable_reason`] answers a neighbouring question for the
/// *generic* host route, whose staged read is wider than a lane for a declared
/// `F64`/`I64`; its verdict on those is deliberately not mirrored here, because
/// they really are one f32 lane per element in an eight-byte slot. Complex keeps
/// its own host route and is refused here.
fn param_operands_are_lane_readable(graph: &Graph, arena: &Arena, node: &rlx_ir::Node) -> bool {
    node.inputs.iter().all(|&id| {
        if !crate::buffer::is_parameter_storage(graph, id) {
            return true;
        }
        let shape = &graph.node(id).shape;
        arena
            .param_storage_at(arena.offset(id))
            .is_some_and(|storage| {
                param_lane_readable(
                    shape.dtype(),
                    shape.num_elements().unwrap_or(0),
                    storage.bytes,
                )
            })
    })
}

/// Refuse a parameter operand neither route can read, naming the geometry.
fn assert_param_operands_are_lane_readable(graph: &Graph, arena: &Arena, node: &rlx_ir::Node) {
    // Only ever read on the refusal path, where the predicate found a parameter
    // operand; the `unwrap_or` keeps the message from naming a subexpression.
    let offender = node
        .inputs
        .iter()
        .copied()
        .find(|&id| crate::buffer::is_parameter_storage(graph, id))
        .unwrap_or(node.inputs[0]);
    assert!(
        param_operands_are_lane_readable(graph, arena, node),
        "rlx-wgpu: REFUSED (parameter not readable as f32 lanes): {:?} reads {offender:?}, a \
         parameter whose stored form is not one f32 lane per element (a packed U8/I8/BF16 upload, \
         or a non-F32 parameter the flat plan sized at its native width, whose widened upload does \
         not fit its own slot; or a parameter the arena holds no storage record for). Both this \
         op's routes read lanes, so neither can compute it. \
         Widen the parameter, or move the op to a backend that honours its declared width.",
        node.op.kind(),
    );
}

/// Refuse a listed op whose *parameter* operand the generic host route cannot stage.
///
/// That route reads every operand as one f32 lane per element, so a parameter is
/// stageable only where its storage and the slot it was given really are that
/// ([`crate::buffer::unstageable_reason`] is the condition,
/// [`crate::buffer::refuse_unstageable_operand`] the message).
/// [`crate::buffer::first_unstageable_parameter_operand`] is that walk, shared with
/// `crate::scan_host::assert_host_reads_fit`, which asks the same question of every
/// operand the host route stages. Only an operand that is a parameter's storage — or a
/// view of one, by the graph ([`crate::buffer::is_parameter_storage`]) — is judged at
/// all: the record is keyed by byte offset alone, so an operand that is not is not judged,
/// whatever the record holds at its offset.
///
/// The graph is what makes this site stronger than that walk: an operand that *is* a
/// parameter's storage (`crate::buffer::is_parameter_storage`) with no record for its
/// offset is refused rather than read as an activation — the arena was built without
/// `Arena::record_param_storage`, and an unanswerable decision takes the safe default.
fn assert_host_route_stageable(graph: &Graph, arena: &Arena, node: &rlx_ir::Node) {
    let operands = node.inputs.iter().map(|&id| {
        let shape = &graph.node(id).shape;
        (
            id,
            arena.offset(id),
            shape.dtype(),
            shape.num_elements().unwrap_or(0),
        )
    });
    if let Some((id, off, dtype, reason)) =
        crate::buffer::first_unstageable_parameter_operand(graph, &arena.param_slots, operands)
    {
        crate::buffer::refuse_unstageable_operand(
            node.op.kind(),
            &format!("{id:?} {}", crate::buffer::operand_location(off)),
            Some(dtype),
            reason,
        );
    }
    // A parameter operand the arena holds no record for cannot be judged at all — its
    // silence reads as "activation". Refusing is the safe default for a decision the
    // arena cannot answer: the graph is here and the arena's record is not, which is the
    // one combination `Arena::param_storage_at` documents as an obligation of the builder.
    let operands: Vec<_> = node
        .inputs
        .iter()
        .map(|&id| (id, arena.offset(id)))
        .collect();
    if let Some((id, off)) =
        crate::buffer::first_unrecorded_parameter(graph, &arena.param_slots, &operands)
    {
        crate::buffer::refuse_unstageable_operand(
            node.op.kind(),
            &format!("{id:?} {}", crate::buffer::operand_location(off)),
            Some(graph.node(id).shape.dtype()),
            crate::buffer::Unstageable::NoRecord,
        );
    }
}

/// The guards a host step addressed by 32-bit arena byte offsets needs, whichever
/// op keeps one.
///
/// `Step::LayerNorm2dHost`, `Step::ReverseHost` and `Step::ConvTranspose2dHost` are
/// addressed by 32-bit byte offsets into the activation arena and read their
/// operands as one f32 lane per element — the transposed conv even in its compact
/// form, which stages only the tensors it touches. A parameter parked in the
/// separate weight buffer loses `WEIGHT_BUF_TAG` in that cast, and an arena past
/// `u32::MAX` bytes truncates every offset, so either geometry would read and
/// write the wrong bytes silently. Refuse loudly instead; `escape_hint` names the
/// step that can serve either geometry, because both are ones this route cannot
/// address and the device route it competes with can.
///
/// For the norm and the flip that route is the device step they take by default,
/// which is what keeps a plan that parks every parameter working: the device step
/// stages such an operand, and the generic packed host route (`Step::HostOp`)
/// stages it too — but an unsharded arena with the norm forced onto this route
/// reaches neither, and this refusal is then the honest answer rather than a wrong
/// one. `Op::ConvTranspose2d` takes a device step by default too and falls back to
/// this host route behind `RLX_WGPU_CONV_HOST` (or the structure-host preference,
/// which covers `RLX_WGPU_FORCE_HOST`, the discrete-adapter term and a sharded
/// arena), so a parked operand or an arena past `u32::MAX` reaches this refusal
/// exactly when that route is the one forced. `Op::ConvTranspose3d`'s host step
/// carries the same 32-bit offsets but is not wired here.
fn assert_byte_offset_host_route(
    graph: &Graph,
    arena: &Arena,
    node: &rlx_ir::Node,
    escape_hint: &str,
) {
    assert_param_operands_are_lane_readable(graph, arena, node);
    assert!(
        !any_parked_operand(arena, node),
        "rlx-wgpu: REFUSED (parked {:?} operand): the host step addresses its operands by 32-bit \
         byte offsets into the activation arena, which drop the weight-buffer tag ({}). Avoid \
         parking this operand, or {escape_hint}.",
        node.op.kind(),
        parked_operand_report(arena),
    );
    assert!(
        arena.size <= u32::MAX as usize,
        "rlx-wgpu: REFUSED ({:?} arena too large): the host step addresses its operands by 32-bit \
         byte offsets into the activation arena and this arena is {} bytes, so its offsets would \
         truncate. Shrink the arena, or {escape_hint}.",
        node.op.kind(),
        arena.size,
    );
}

/// The refusal `arena_off_in_window_or_stage` makes about an operand of
/// `Op::ConvTranspose2d`'s device step: one above the reserve the step would have
/// to copy it into. The shared helper asks the same question, but its message can
/// only name the cap — only a step that has a route back to the host can name it —
/// so the op asks first, where it can.
///
/// It asks with the shared window predicates (`arena_whole_arena_bind`,
/// `arena_tensor_in_window`) and [`arena_stage_ceiling`], which sits beside the caps
/// `arena_off_in_window_or_stage` applies so the two read together — but the ceiling
/// is a parallel derivation of those thresholds rather than the same expression, and
/// has to be kept in step with them. The guard is deliberately the weaker of the two:
/// an operand it lets through still reaches the shared refusal, which keeps the last
/// word.
///
/// The shared helper's other refusal here — a graph parameter above the adapter's
/// binding limit — is deliberately not announced, because the host route cannot
/// serve such an operand either (it stages through the same reserves), so naming it
/// would be advice that does not work.
fn assert_conv_transpose_operands_addressable(
    arena: &Arena,
    device: &wgpu::Device,
    id: rlx_ir::NodeId,
    base: u64,
    size: u64,
) {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    // The window `arena_off_in_bind_window` will judge the operand against: the
    // whole arena when that one binds (it is judged against that window, not the
    // caller's), the caller's window otherwise.
    let (base, size) = arena_whole_arena_bind(arena, max_binding).unwrap_or((base, size));
    // False for a parked operand whatever the window is, so a hit here covers
    // both reasons the shared path would fall through to staging — and a miss is
    // the shared path's own answer that there is nothing to stage.
    if arena_tensor_in_window(arena, id, base, size) {
        return;
    }
    let len = arena.len_of(id) as u64;
    let cap = arena_stage_ceiling(arena, id, base);
    // A parked operand is refused by the host route too (it stages through the same
    // reserves), so sending it there would be advice that does not work.
    let escape = if crate::buffer::is_weight_off(arena.offset(id)) {
        "Keep the operand in the activation arena — parking is what RLX_WGPU_ALL_PARAMS_WEIGHT \
         forces and what a split plan does when the arena cannot hold the weights. \
         RLX_WGPU_SHARD_STAGE_MIB only lowers the reserve a parked operand has to fit, so it is \
         not a lever for making room, and the host route cannot take a parked operand either."
    } else {
        "Run this op on the host route (RLX_WGPU_CONV_HOST=1, or RLX_WGPU_FORCE_HOST=1), which \
         reads its operands straight out of the arena instead of staging them."
    };
    assert!(
        len <= cap,
        "rlx-wgpu: REFUSED (Op::ConvTranspose2d operand too large to stage): node {id:?} is {len} \
         bytes and the device step must copy it into a {cap}-byte staging reserve. {escape}"
    );
}

/// Whether `conv_transpose2d_tiled.wgsl` is proven for `p` — the 1-D-lifted
/// shapes it was measured on, and nothing wider.
///
/// The 2-D arm pins the depth axis before calling (`d = d_out = kd = sd = 1`,
/// `pd = 0`, `dd = 1`), so the family is stated over the rest. The kernel
/// flattens input and output to one row and reads one `kw`-long weight row per
/// (ci, co), so batch, height and the height's single tap must all be one:
/// `h = h_out = kh = 1`. `ph = 0` is the load-bearing one of that group — a form
/// with height padding is one where the gather finds its single height tap
/// outside the input and drops it, while this kernel, which does not know about
/// the height axis at all, sums the width walk anyway. `dw = 1` because the
/// phase's `kx = r + s·m` carries no dilation. `s ≤ k ≤ 2·s` is the family the
/// tile's `TAPS` taps and each phase's tap bound are written against — one or two
/// taps a phase, never none. And the grid must be one the device can hold.
///
/// Everything else keeps [`Step::ConvTranspose2d`], which serves the whole
/// operator — no shape loses its route, and none reaches this form without being
/// proven for it.
fn conv_transpose2d_tiled_fits(p: &Conv3dParams) -> bool {
    if p.sw < 1
        || p.n != 1
        || p.h != 1
        || p.h_out != 1
        || p.kh != 1
        || p.ph != 0
        || p.dw != 1
        || p.groups != 1
    {
        return false;
    }
    // Checked before the doubling below: the grid bounds `sw`.
    let grid = crate::kernels::conv_transpose2d_tiled_grid(p.w_out, p.sw, p.c_out);
    if grid.iter().any(|g| *g > MAX_WORKGROUPS_PER_DIM) {
        return false;
    }
    p.sw <= p.kw && p.kw <= 2 * p.sw
}

/// Resolve `ids` into the bind window a device step will read them from, and
/// return each one's f32-lane offset inside it.
///
/// The two windowed steps added for `Op::LayerNorm2d` and `Op::Reverse` share
/// this prologue. A single binding may not cover every operand, so
/// `arena_multi_op_window` anchors on what fits and stages the rest into the
/// arena's scratch tail (emitting `Step::BufferCopy`); because staging an
/// operand relocates or widens `base`, every offset is then re-resolved against
/// the final window rather than trusted from before the staging.
///
/// `win` is the operand list the window is built over, in the order that
/// builder prefers; `ids` are the operands whose offsets the caller wants, in
/// the order it wants them resolved (each call can stage, so that order is
/// observable).
fn window_offsets(
    device: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    schedule: &mut Vec<Step>,
    win: &[NodeId],
    ids: &[NodeId],
) -> (u64, u64, Vec<u32>) {
    let fits = arena_span_bytes(arena, win) <= device.limits().max_storage_buffer_binding_size;
    let mut scratch = arena.scratch_off as u64;
    let (mut base, mut size, param_anchor) = arena_multi_op_window(
        device,
        arena,
        graph,
        param_offsets,
        schedule,
        &mut scratch,
        win,
    );
    if !fits && !param_anchor {
        base = arena_bind_window_covering_scratch_if_needed(arena, base, size, scratch);
    }
    let mut offsets = Vec::with_capacity(ids.len());
    for &id in ids {
        offsets.push(arena_off_in_bind_window(
            graph,
            param_offsets,
            device,
            arena,
            schedule,
            &mut scratch,
            id,
            &mut base,
            &mut size,
        ));
    }
    for (&id, off) in ids.iter().zip(offsets.iter_mut()) {
        if arena_tensor_in_window(arena, id, base, size) {
            *off = arena_local_off_f32(arena, id, base);
        }
    }
    (base, size, offsets)
}

/// The addressed geometry, for the ceiling rule's messages: which arena could not
/// be bound whole, and why.
fn geometry_report(arena: &Arena, limit: u64) -> String {
    if arena.is_sharded() {
        format!(
            "activation arena {} B is striped, per-binding limit {limit} B",
            arena.size
        )
    } else {
        format!(
            "activation arena {} B vs storage binding limit {limit} B",
            arena.size
        )
    }
}

/// [`geometry_report`]'s other half, for the arms that refuse because a *parameter
/// operand* is parked: the separate region is what they cannot address, whatever the
/// activation arena's own geometry.
fn parked_operand_report(arena: &Arena) -> String {
    format!(
        "a parameter operand is parked in the weight buffer (activation arena {} B)",
        arena.size
    )
}

/// Whether `Op::GroupNorm` must fall back to the whole-arena host step.
///
/// Mirrors the LayerNorm/RmsNorm policy in [`wgpu_prefer_host_fallback`]:
/// virtually-sharded arenas keep the host path (the native kernel binds a
/// single arena window). `RLX_WGPU_HOST_NORM` forces it; `RLX_WGPU_GPU_NORM`
/// forces the kernel. `Op::LayerNorm2d` uses [`wgpu_prefer_host_norm2d`]
/// instead, which drops the adapter-kind term.
fn wgpu_prefer_host_norm(arena: &crate::buffer::Arena) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_GPU_NORM") {
        return false;
    }
    wgpu_prefer_host_fallback(arena) || rlx_ir::env::flag("RLX_WGPU_HOST_NORM")
}

/// Whether `Op::LayerNorm2d` must fall back to the whole-arena host step.
///
/// Same documented switch pair as [`wgpu_prefer_host_norm`] —
/// `RLX_WGPU_HOST_NORM` forces the host step, `RLX_WGPU_GPU_NORM` forces the
/// device kernel — but the adapter-kind term is deliberately absent, so the
/// "force norms onto the GPU" switch the registry documents also covers this
/// normalisation on a discrete adapter.
fn wgpu_prefer_host_norm2d(arena: &crate::buffer::Arena) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_GPU_NORM") {
        return false;
    }
    wgpu_prefer_host_fallback_agnostic(arena) || rlx_ir::env::flag("RLX_WGPU_HOST_NORM")
}

/// Conv2d SPIR-V is still wrong on discrete NVIDIA even for unsharded arenas
/// (Kitten vocoder 1×L "2d" convs → NSF mush). Keep hosting independently of
/// [`wgpu_prefer_host_fallback`]. Both transposed convs are gated by this too —
/// `Op::ConvTranspose3d`'s device step sits behind it as well, and
/// `Op::ConvTranspose2d` follows the sibling under the same gate.
fn wgpu_prefer_conv_host(arena: &crate::buffer::Arena) -> bool {
    wgpu_prefer_structure_host(arena) || rlx_ir::env::flag("RLX_WGPU_CONV_HOST")
}

/// A `Step::HostOp` that also carries which of `node`'s operands are a *parameter's*
/// storage, by the graph — the predicate the plan-time check uses, so the run-time net
/// (`scan_host::assert_host_reads_fit`) and it cannot disagree.
fn host_op_step(graph: &Graph, node: &rlx_ir::Node, arena: &Arena) -> Step {
    Step::HostOp {
        desc: rlx_cpu::rlx_host_op_desc!(graph, node, |id| arena.offset(id)),
        param_operands: node
            .inputs
            .iter()
            .map(|&id| crate::buffer::is_parameter_storage(graph, id))
            .collect(),
    }
}

/// One `Op::Concat{axis: 0}` recognised as the mel front-end's framing: its rows
/// are `narrow`(+`reshape`) frames sliced at a uniform `hop` out of one source,
/// which is one strided read of that source and needs no index table to say so.
///
/// `Graph::stft` / `Graph::spectrogram` build exactly this, and so does any
/// front-end that rolls its own framing out of the public builders — both are
/// folded here without touching either's code.
#[derive(Debug, Clone, Copy)]
struct FramesPlan {
    src: NodeId,
    /// The source's last-axis length, the stride between one frame and the next
    /// inside a leading-dims slice.
    t: u32,
    frame_len: u32,
    hop: u32,
    n_frames: u32,
    /// Product of the source's leading dims — the destination's middle axis.
    outer: u32,
}

/// Recognise per-frame framing (`narrow` per frame + one `concat`), returning the
/// narrow/reshape nodes to elide and, per `concat`, the geometry the single
/// `Step::Frames` dispatch needs.
///
/// The fold is worth it because the per-frame form is one step per frame on both
/// sides of the copy (`Step::Narrow` and one `Step::Concat` input each): 200
/// dispatches for 100 frames of 400 samples, where one strided read of the same
/// source does it in one. It is deliberately NOT the shared IR builders' default
/// — the batched `Op::Gather` spelling of the same matrix is ~9× slower on the
/// CPU device, which is why this is a wgpu-local lowering fold rather than a
/// change to `Graph::stft`.
///
/// Every clause below is a refusal, not a correction: a graph this cannot prove
/// safe keeps the per-frame steps (and the values) it came with. This is the one
/// place the clause list is recorded — the crate README and the pins name classes
/// only, so a clause cannot be recorded in one place and updated in another.
///
/// 1. *the arena*: a sharded arena, one that cannot be bound whole at base zero
///    (the kernel addresses the arena by absolute word offset), and the two
///    structure/route knobs that move this arm's concat off its direct window path
///    (`wgpu_prefer_structure_host`, `RLX_WGPU_CONCAT_HOST`).
/// 2. *the rows*: every row a `narrow` (+ `reshape` view) of one source, at the
///    source's last axis, of one length, each read by nothing but its own
///    successor in the chain, none of them a graph output, and starting at
///    `f · hop` with `hop` the same for all — the op is `Concat { axis: 0 }`.
/// 3. *the source*: no rows of its own (`outer == 1`) — the mode this step scales
///    trims a *row* axis, and the only rows the step presents are the frames, so a
///    source that carries rows keeps the per-frame steps, which trim those rows
///    (this also covers a zero-length leading dim); a shape known at build time
///    (the geometry is baked); in range of the rows
///    (`(n_frames-1)·hop + frame_len ≤ t`); a live slot (`arena.has`); a step that
///    still *writes* it (`split_qkv_parents` — the split-QKV fold replaces that
///    step's write with one into the narrow slots); a home in the activation arena
///    (`is_weight_off` — a parameter parked in the weight buffer is a different
///    binding, and today's planner only parks parameters, whose framing the
///    destination clause below refuses first, so this is the belt to that braces);
///    and a non-complex dtype (one word per element here, 2 or 4 lanes there).
/// 4. *the destination*: a live slot (`arena.has`) and not a tensor the concat arm
///    bakes as a static weight (`is_static_weight_tensor` — strictly wider than
///    that arm's own `once` / `slot_is_exclusive` conditions).
/// 5. *the source's live range*: nothing the planner placed on the source's bytes
///    between the source's last reader (`last_use`) and this concat.
///
/// `node_pos` is the emission order the caller already keeps, which clause 5
/// needs — the memory plan was built before this fold was chosen, and says
/// nothing about the read this fold moves.
///
/// `static_weight_memo` is the caller's own memo for `is_static_weight_tensor`, so
/// the classification clause 4 reads here is the same computation `Op::Concat`
/// below reads rather than a second one.
fn detect_frames_pattern(
    graph: &Graph,
    arena: &Arena,
    dev: &crate::device::WgpuDevice,
    node_pos: &HashMap<NodeId, usize>,
    last_use: &HashMap<NodeId, usize>,
    split_qkv_parents: &HashSet<NodeId>,
    static_weight_memo: &mut HashMap<NodeId, bool>,
) -> (HashSet<NodeId>, HashMap<NodeId, FramesPlan>) {
    let mut skip: HashSet<NodeId> = HashSet::new();
    let mut plans: HashMap<NodeId, FramesPlan> = HashMap::new();

    // Who reads each tensor. A row may only be elided if nothing but its own
    // successor in the chain reads it — and the consumer clauses below are
    // checked per link (narrow → [reshape →] concat), so a frame sliced for a
    // second purpose keeps its step.
    let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            users.entry(input).or_default().push(node.id);
        }
    }
    let sole_user = |id: NodeId, user: NodeId| -> bool {
        matches!(users.get(&id).map(Vec::as_slice), Some([only]) if *only == user)
    };

    // The kernel addresses the activation arena by ABSOLUTE word offset, so the
    // whole arena has to be bound at base zero (`bind_op_output_window` refuses
    // anything else outright) and the source has to live in that arena: a source
    // parked in the weight buffer is a different binding, and the tag is dropped
    // on the way into the uniform.
    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
    if arena.is_sharded()
        || !arena.binds_whole_at_base_zero(max_binding)
        || wgpu_prefer_structure_host(arena)
        // The framed concat is a concat, and `RLX_WGPU_CONCAT_HOST` is the
        // concat arm's own bisect knob for taking those off their direct window
        // path. Folding here would route around the lever, so honour it.
        || rlx_ir::env::flag("RLX_WGPU_CONCAT_HOST")
    {
        return (skip, plans);
    }

    'concat: for node in graph.nodes() {
        let Op::Concat { axis: 0 } = &node.op else {
            continue;
        };
        if node.inputs.len() < 2 {
            continue;
        }
        let mut rows: Vec<u32> = Vec::with_capacity(node.inputs.len());
        // Only merged into `skip` once every clause below has held: an elided
        // narrow with a step still emitting its consumer is a frame the concat
        // reads from a slot nothing writes.
        let mut elide: Vec<NodeId> = Vec::with_capacity(node.inputs.len());
        let mut src: Option<NodeId> = None;
        let mut frame_len: Option<u32> = None;
        for &row in &node.inputs {
            // A row is a frame: the narrow itself, or a view of it.
            let reshaped = match &graph.node(row).op {
                Op::Narrow { .. } => None,
                Op::Reshape { .. } if graph.node(row).inputs.len() == 1 => Some(row),
                _ => continue 'concat,
            };
            let narrow = match reshaped {
                Some(r) => graph.node(r).inputs[0],
                None => row,
            };
            let Op::Narrow { axis, start, len } = &graph.node(narrow).op else {
                continue 'concat;
            };
            let in_id = graph.node(narrow).inputs[0];
            // A frame that is itself a graph output has to be materialized where
            // it stands, which the elided narrow no longer does.
            if graph.outputs.contains(&narrow)
                || reshaped.is_some_and(|r| graph.outputs.contains(&r))
            {
                continue 'concat;
            }
            // The narrow must slice the source's LAST axis: that is what makes a
            // frame's words one contiguous run of the source, which is the whole
            // reason this is a strided read rather than a gather.
            if *axis + 1 != graph.node(in_id).shape.rank() {
                continue 'concat;
            }
            // A reshape between the narrow and the concat has to be a view of it
            // whose last dim is the frame length and whose element count is the
            // narrow's, so the row's word order is the narrow's and the frame
            // length the destination is laid out in is the one the narrow sliced.
            // No leading-dim clause is needed: the source-rows rule below runs
            // before this fold is chosen, and with `outer == 1` every leading dim is
            // a unit, so a permuting view cannot be built from them.
            if let Some(r) = reshaped {
                let reshaped_shape = &graph.node(r).shape;
                let last = reshaped_shape
                    .dims()
                    .last()
                    .filter(|d| d.is_static())
                    .map(|d| d.unwrap_static());
                if last != Some(*len)
                    || reshaped_shape.num_elements() != graph.node(narrow).shape.num_elements()
                {
                    continue 'concat;
                }
                if !sole_user(narrow, r) || !sole_user(r, node.id) {
                    continue 'concat;
                }
            } else if !sole_user(narrow, node.id) {
                continue 'concat;
            }
            // All rows are frames of the same source, of the same length, or the
            // concat is not one frame matrix.
            if src.is_some_and(|s| s != in_id) || frame_len.is_some_and(|fl| fl != *len as u32) {
                continue 'concat;
            }
            src = Some(in_id);
            frame_len = Some(*len as u32);
            rows.push(*start as u32);
            elide.push(narrow);
            if let Some(r) = reshaped {
                elide.push(r);
            }
        }
        let (Some(src), Some(frame_len)) = (src, frame_len) else {
            continue;
        };
        // Uniform hop: `starts[f] == f * hop`, so `start == 0` and the hop is the
        // one the rows already agree on.
        if rows[0] != 0 {
            continue;
        }
        let hop = rows[1] - rows[0];
        if hop == 0 || rows.iter().enumerate().any(|(f, &s)| s != f as u32 * hop) {
            continue;
        }
        // The source's own step may not exist any more. The split-QKV fold elides
        // an `FusedMatMulBiasAct`'s step (replacing it with `MatmulQkv`) and writes
        // Q/K/V into the *narrow* slots, never the parent's own slot — and three
        // last-axis narrows of a parent, re-joined by a concat, are exactly this
        // fold's pattern. Reading such a parent here would read a slot nothing
        // writes (measured: the whole matrix zeros against a non-zero reference).
        // `rlx-lightbeam`'s GRU gate projection is that shape in the wild.
        if split_qkv_parents.contains(&src) {
            continue;
        }
        // In range, in the activation arena, and at a shape that is known at
        // build time: the geometry below is baked into the kernel's parameters,
        // so a source whose dims are symbols has no address to bake.
        if !graph.node(src).shape.is_static() {
            continue;
        }
        let src_dims = graph.node(src).shape.dims();
        let t = src_dims[src_dims.len() - 1].unwrap_static() as u32;
        let n_frames = node.inputs.len() as u32;
        if (n_frames - 1) * hop + frame_len > t
            || !arena.has(src)
            || !arena.has(node.id)
            || crate::buffer::is_weight_off(arena.offset(src))
        {
            continue;
        }
        // `frames.wgsl` moves one arena word per element, which is what the
        // narrow/concat steps it replaces do for every non-complex dtype — and is
        // wrong for a complex one, whose elements are 2 or 4 contiguous lanes.
        // Complex framing would be half-written rather than refused, so refuse.
        if graph.node(src).shape.dtype().is_complex() || node.shape.dtype().is_complex() {
            continue;
        }
        // A framing the concat arm bakes as a static weight tensor (`once` /
        // `slot_is_exclusive`, at `Op::Concat` below) is dispatched once ever
        // there; the folded step has no such bake, so it would dispatch that
        // matrix once per call instead. Leaving those concats to the arm that
        // knows how to bake them needs both of its conditions, so this refusal
        // stands on the destination's provenance alone and is strictly wider than
        // that arm's decision — conservative on purpose.
        if is_static_weight_tensor(graph, node.id, static_weight_memo) {
            continue;
        }
        // The fold reads the source at the concat's step, while the plan's
        // liveness for it ends at its last reader (`last_use`, above). Anything
        // the planner placed on the source's bytes after that was free for the
        // per-frame copy this fold removes — the destination itself, when the
        // planner hands it the source's slot — so the step would read that value
        // back as the framing. Reached by a source of one row whose last-axis
        // narrows are copies — an inner-axis narrow never aliases its source,
        // whatever the hop; only axis-0 narrows are views — so the source dies at
        // its last narrow while the fold still reads it. The hop being wider than a
        // frame is what makes the source wider than the destination, and that is
        // what lets the planner place the destination on its head; the placement
        // itself is read off the plan rather than argued, and the pin asserts it.
        let last_read = last_use.get(&src).copied().unwrap_or(0);
        let concat_pos = node_pos[&node.id];
        if graph.nodes().iter().any(|n| {
            node_pos[&n.id] > last_read
                && node_pos[&n.id] <= concat_pos
                && !elide.contains(&n.id)
                && arena.has(n.id)
                && arena_tensors_overlap(arena, src, n.id)
        }) {
            continue;
        }

        // The mode's trim is a row axis, and the only rows this step presents are
        // the frames — so the fold is taken only where the source has no row axis
        // of its own to compete with them. With a leading dim of its own (`outer`
        // > 1) the per-frame steps this fold replaces trimmed the *source's* rows,
        // so a caller bucketing those rows must keep them: the fold stands down and
        // leaves those steps — and the mode, which scales them correctly — in
        // place. A zero-length leading dim falls out of the same clause, and would
        // otherwise dispatch over a destination with no elements while the kernel's
        // bound counts frames (reading and writing the arena's head).
        let outer = src_dims[..src_dims.len() - 1]
            .iter()
            .map(|d| d.unwrap_static() as u32)
            .product::<u32>();
        if outer != 1 {
            continue;
        }
        skip.extend(elide);
        plans.insert(
            node.id,
            FramesPlan {
                src,
                t,
                frame_len,
                hop,
                n_frames,
                outer,
            },
        );
    }

    (skip, plans)
}

pub(crate) fn compile_static_inner(
    graph: Graph,
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
) -> crate::backend::WgpuExecutable {
    let dev = wgpu_device().expect("rlx-wgpu: no compatible adapter found");

    // Decompose composed/fused ops (FusedMatMulBiasAct, LoraMatMul,
    // FusedAttentionBlock, FusedTransformerLayer, ...) into primitive
    // sequences before memory planning so every intermediate gets a
    // regular arena slot. CPU/Metal/MLX lower the fused variants
    // directly with bespoke kernels; we choose simplicity over peak
    // throughput here.
    let graph = crate::unfuse::unfuse(graph);
    // Regions this kernel's `apply_activation` would silently compute as their own
    // input, back on primitives before anything lowers them (see the hook's docs).
    let graph = crate::unfuse::expand_unimplemented_regions(graph);
    // Elide the redundant reshapes unfuse/matmul-lowering emit — they alias
    // parents and block the planner's slot reuse (see collapse_reshapes).
    let graph = crate::unfuse::collapse_reshapes(graph);
    // The runtime path legalises `Op::Binary` broadcasts before fusion; peel the
    // copies the f32 kernel reads itself, so they are not a step or a slot.
    let graph = crate::unfuse::elide_binary_broadcast_expands(graph);

    if rlx_ir::env::flag("RLX_WGPU_DEBUG_OP_HIST") {
        let mut hist: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for node in graph.nodes() {
            *hist.entry(format!("{:?}", node.op.kind())).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = hist.into_iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.1));
        eprintln!("rlx-wgpu op histogram: {sorted:?}");
    }

    // f32-uniform slots + liveness reuse (pairwise `[n,n]` graphs).
    let mut plan = plan_f32_uniform(&graph, 16);
    // The shared planner only allocates a slot when its byte size is non-zero
    // (`rlx_compile::memory`), so a tensor with a zero-length dimension gets no entry at all —
    // LuxTTS's flow decoder builds an `Expand [0,1,512]`. Every arena lookup in this backend
    // (`Arena::offset`, bind-window checks, dispatch lowering) then panicked on a node the
    // planner had deliberately skipped. Give each one an explicit empty slot: zero bytes to
    // read or write, so any offset is as correct as any other, and the lookups all resolve.
    // `arena_span_bytes` skips zero-length ids so these never anchor a bind window.
    for node in graph.nodes() {
        if node.shape.num_elements() == Some(0) {
            plan.assignments
                .entry(node.id)
                .or_insert(rlx_compile::memory::BufferSlot { offset: 0, size: 0 });
        }
    }
    if rlx_ir::env::flag("RLX_WGPU_HAZARD_REPORT") {
        report_slot_hazards(&graph, &plan);
    }
    // `RLX_WGPU_ARENA_REPORT=1` says where the arena went. Striping is refused
    // outright, so when a graph is too big the only useful question is which
    // tensors made it that way.
    if rlx_ir::env::flag("RLX_WGPU_ARENA_REPORT") {
        let mut by_op: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
        let mut slots: Vec<(usize, String)> = Vec::new();
        for (id, slot) in &plan.assignments {
            let name = graph
                .nodes()
                .iter()
                .find(|n| n.id == *id)
                .map(|n| format!("{:?}", n.op.kind()))
                .unwrap_or_else(|| "?".into());
            let e = by_op.entry(name.clone()).or_insert((0, 0));
            e.0 += 1;
            e.1 += slot.size;
            slots.push((slot.size, name));
        }
        slots.sort_by_key(|s| std::cmp::Reverse(s.0));
        let mut agg: Vec<_> = by_op.into_iter().collect();
        agg.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        // Not printed here: the pooled output slab is a device buffer `plan` does
        // not own, allocated lazily by the first batched packed multiply.
        // `gguf_gpu::pooled_out_buf_bytes()` reports it, read after a run, so a
        // memory figure taken then cannot understate device usage.
        eprintln!(
            "[rlx-wgpu] arena {:.2} GiB over {} slots; distinct-slot sum {:.2} GiB",
            gib(plan.arena_size),
            plan.assignments.len(),
            gib(slots.iter().map(|s| s.0).sum::<usize>()),
        );
        for (op, (n, bytes)) in agg.iter().take(8) {
            eprintln!("    {op:<22} {n:>4} slots  {:.2} GiB", gib(*bytes));
        }
    }
    let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
    let dequant_scratch_off = if dequant_scratch > 0 {
        let aligned = plan.arena_size.div_ceil(16) * 16;
        let new_size = aligned + dequant_scratch.div_ceil(16) * 16;
        if (new_size as u64) <= dev.device.limits().max_buffer_size {
            plan.arena_size = new_size;
            aligned
        } else {
            0
        }
    } else {
        0
    };
    // Ephemeral GatedDeltaNet state (`carry_state=false`) — shared across
    // layers like Metal/CUDA. Host fallback allocates on the CPU heap.
    let gdn_scratch = gdn_ephemeral_state_bytes(&graph);
    let gdn_scratch_off = if gdn_scratch > 0 {
        let aligned = plan.arena_size.div_ceil(16) * 16;
        let new_size = aligned + gdn_scratch.div_ceil(16) * 16;
        if (new_size as u64) <= dev.device.limits().max_buffer_size {
            plan.arena_size = new_size;
            aligned
        } else {
            0
        }
    } else {
        0
    };
    // Ping-pong scratch (pair) for native multi-layer GRU / Elman RNN
    // intermediate layer outputs; shared across ops like gdn_scratch.
    let rnn_gru_scratch = rnn_gru_scratch_bytes(&graph);
    let rnn_gru_scratch_off = if rnn_gru_scratch > 0 {
        let aligned = plan.arena_size.div_ceil(16) * 16;
        let new_size = aligned + rnn_gru_scratch.div_ceil(16) * 16;
        if (new_size as u64) <= dev.device.limits().max_buffer_size {
            plan.arena_size = new_size;
            aligned
        } else {
            0
        }
    } else {
        0
    };
    // Pre-walk to compute the max scratch any single op needs — see
    // `compute_scratch_bytes` for the ops that claim a slice of the tail.
    let base_scratch_bytes = compute_scratch_bytes(&graph);
    // Reserve tail scratch for the im2col `col` matrix only when the accelerated
    // conv routes are enabled (they are on by default; `RLX_WGPU_CONV_IM2COL=0`
    // restores the lean arena — the tiled/direct convs need no extra scratch).
    let conv_col_scratch = if crate::backend::conv_routes_enabled() {
        conv_im2col_scratch_bytes(
            &graph,
            plan.arena_size,
            dev.device.limits().max_storage_buffer_binding_size,
        )
    } else {
        0
    };
    let scratch_bytes = {
        let max_bind = dev.device.limits().max_storage_buffer_binding_size;
        let max_buf = dev.device.limits().max_buffer_size;
        // Large unsharded arenas (> storage bind cap) stage far operands into
        // a dedicated tail so `scratch_off` is not stuck at `arena.size`.
        // On ~2 GiB max_buffer adapters (VirtIO/DX12), a full 768 MiB stage
        // on top of a nearly-full act arena + weight buffer OOMs — keep a
        // modest tail; `RLX_WGPU_SHARD_STAGE_MIB` still overrides.
        let stage = if (plan.arena_size as u64) > max_bind {
            let full = crate::buffer::shard_stage_reserve();
            if max_buf <= (2u64 << 30) && rlx_ir::env::var("RLX_WGPU_SHARD_STAGE_MIB").is_none() {
                full.min(64 * 1024 * 1024)
            } else {
                full
            }
        } else {
            0
        };
        base_scratch_bytes.max(conv_col_scratch).max(stage)
    };
    // When the single-buffer arena would exceed the storage bind limit (or
    // max_buffer_size), split params into a separate weight buffer so the act
    // arena stays compact. Needed for NVIDIA ~2 GiB binds (Qwen3 F32 GGUF /
    // safetensors) and 27B-class packed GGUF. Only fused GGUF GEMV binds the
    // weight buffer directly; other ops stage params into the act window.
    let max_buf = dev.device.limits().max_buffer_size;
    let max_bind = dev.device.limits().max_storage_buffer_binding_size;
    // Use the split/view-alias planner when the arena exceeds one storage bind
    // window (not only when it exceeds max_buffer_size). On Metal we raise
    // max_buffer_size to ~40 GiB so F5 fits in one buffer, but Param→Reshape
    // views still need the split pass or they stay zeroed and DiT collapses.
    // `RLX_WGPU_ALL_PARAMS_WEIGHT` is what makes that planner park EVERY parameter,
    // as its name says; without the flag reaching this decision, the parked-operand
    // half of `should_host_at_ceiling` would only be reachable on adapters whose plan
    // happens to exceed the bind window.
    let mut arena = if rlx_ir::env::flag("RLX_WGPU_ALL_PARAMS_WEIGHT")
        || (plan.arena_size + scratch_bytes) as u64 > max_bind
    {
        // This constructor has the graph, so it records the parameter-storage bounds
        // itself, from the compact plan it re-plans and places the parameters under.
        Arena::from_plan_split(&dev.device, &plan, &graph, scratch_bytes)
    } else {
        // This one never sees a graph, so its record is taken here: one recording per
        // arena, both plans covered.
        let mut flat = Arena::from_plan_with_scratch(&dev.device, &plan, scratch_bytes);
        flat.record_param_storage(&graph);
        flat
    };
    let _ = max_buf;
    // Override slot lengths with the actual elem*4 byte counts so
    // readback returns the right element count (slots may be
    // padded for alignment).
    for node in graph.nodes() {
        let elems = node.shape.num_elements().unwrap_or(0);
        // Actual byte length = (# of f32 lanes) × 4. Real/int/bool tensors are
        // widened to exactly ONE f32 lane per element (→ `elems * 4`). Complex
        // tensors genuinely occupy MULTIPLE f32 lanes — C64 = 2 lanes (8 B),
        // C128 = 4 lanes (16 B) — so `elems * 4` would truncate the slot and
        // drop the imaginary (and df64 lo) lanes on readback / aliasing copies.
        let dt = node.shape.dtype();
        let actual = if dt.is_complex() {
            elems * dt.size_bytes()
        } else {
            elems * 4
        };
        arena.set_actual_len(node.id, actual);
    }

    // Initialize Constants directly into the arena. The wgpu arena is f32-
    // uniform (4 bytes/elem; integer inputs are widened on upload), so integer
    // and bool constants must be converted to f32 here too — writing their raw
    // bytes would corrupt them (e.g. the VITS sequence-mask `arange` i64 const,
    // 8 bytes/elem, read back as f32 garbage → all-zero mask → dead encoder).
    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op
            && arena.has(node.id)
            && !data.is_empty()
        {
            let widened: Option<Vec<u8>> = match node.shape.dtype() {
                rlx_ir::DType::I64 => Some(
                    data.chunks_exact(8)
                        .flat_map(|c| {
                            (i64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                        })
                        .collect(),
                ),
                rlx_ir::DType::I32 => Some(
                    data.chunks_exact(4)
                        .flat_map(|c| {
                            (i32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                        })
                        .collect(),
                ),
                rlx_ir::DType::U32 => Some(
                    data.chunks_exact(4)
                        .flat_map(|c| {
                            (u32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                        })
                        .collect(),
                ),
                rlx_ir::DType::Bool | rlx_ir::DType::U8 => Some(
                    data.iter()
                        .flat_map(|&b| (b as f32).to_le_bytes())
                        .collect(),
                ),
                rlx_ir::DType::I8 => Some(
                    data.iter()
                        .flat_map(|&b| ((b as i8) as f32).to_le_bytes())
                        .collect(),
                ),
                _ => None,
            };
            let bytes: &[u8] = widened.as_deref().unwrap_or(data);
            let bytes_to_write = bytes.len().min(arena.len_of(node.id));
            // Chunked + shard-safe — a single `queue.write_buffer` of a
            // large F5 weight tile can truncate on Metal/wgpu and leave
            // the tail as zeros (teacher-forced peaks then sit ~5% low).
            arena.write_bytes_range(&dev.queue, arena.offset(node.id), &bytes[..bytes_to_write]);
        }
    }

    let mut input_offsets = HashMap::new();
    let mut param_offsets = HashMap::new();
    for node in graph.nodes() {
        match &node.op {
            Op::Input { name } => {
                input_offsets.insert(name.clone(), node.id);
            }
            Op::Param { name } => {
                param_offsets.insert(name.clone(), node.id);
            }
            _ => {}
        }
    }

    let mm_k = matmul_kernel(&dev.device);
    let mm_w = matmul_wide_kernel(&dev.device);
    let _mm_w_active = matmul_wide_active_kernel(&dev.device);
    let mm_f16w = matmul_f16w_kernel(&dev.device);
    let mm_bf16w = matmul_bf16w_kernel(&dev.device);
    let mm_f16c = matmul_f16_compute_kernel(&dev.device);
    let mm_coop = matmul_coop16_kernel(&dev.device);
    let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
    let mm_cast = cast_f32_to_f16_kernel(&dev.device);
    let bk = binary_kernel(&dev.device);
    let uk = unary_kernel(&dev.device);
    let ck = compare_kernel(&dev.device);
    let wk = where_kernel(&dev.device);
    let fk = fma_kernel(&dev.device);
    let abk = activation_backward_kernel(&dev.device);

    let mut schedule = Vec::new();
    let mut uniforms = Vec::new();
    let mut bind_groups = Vec::new();
    let mut fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources> = Vec::new();
    let mut meta_buffers: Vec<wgpu::Buffer> = Vec::new();
    let mut coop_f16_b_param: HashMap<u32, String> = HashMap::new();
    let mut coop_f16_vk_wide_bind_groups: HashMap<usize, wgpu::BindGroup> = HashMap::new();
    let mm_w_active_compile = matmul_wide_active_kernel(&dev.device);

    let coop_f16_vk_mirror_acts = collect_coop_f16_vk_mirror_activations(&graph, &dev.device);

    // Detect (FusedMatMulBiasAct → Narrow×3) split-QKV pattern. Returns
    // a map parent_node_id → (q_narrow_id, k_narrow_id, v_narrow_id).
    // The matmul_qkv kernel collapses the matmul + 3 narrows into one
    // dispatch by routing each output column to the right Q/K/V sink.
    //
    // CRITICAL: only mark a pattern site for elision when the parent
    // FMB will actually take the MatmulQkv path (which only fires
    // for F32 compute precision). For Coop16/CoopF32-eligible FMBs,
    // those kernels write to the FMB's *own* output slot, NOT the
    // 3 narrow slots — skipping the narrows would leave Q/K/V
    // uninitialized and attention would read garbage. Predict the
    // compute precision the FMB will receive; only skip when F32.
    // Emission order, and the last step that reads each node — together these
    // say whether a tensor is still live at some other node's step, which is
    // what the fused-write guards below need and the memory plan cannot tell
    // them (the plan is built before these folds are chosen).
    let node_pos: HashMap<NodeId, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();
    let last_use: HashMap<NodeId, usize> = {
        let mut m: HashMap<NodeId, usize> = HashMap::new();
        for (i, n) in graph.nodes().iter().enumerate() {
            for &inp in &n.inputs {
                let e = m.entry(inp).or_insert(0);
                *e = (*e).max(i);
            }
        }
        // Graph outputs are read after the whole schedule.
        for &o in &graph.outputs {
            m.insert(o, graph.nodes().len());
        }
        m
    };
    let mut qkv_split: HashMap<NodeId, (NodeId, NodeId, NodeId)> = HashMap::new();
    for (parent_id, qkv) in detect_split_qkv_pattern(&graph) {
        let parent = graph.node(parent_id);
        // Mirror the lowering's precision derivation. FMB inputs:
        // [a, w, bias]; we need (m, k, n) to query.
        let a_id = parent.inputs[0];
        let b_id = parent.inputs[1];
        let a_dims = graph.node(a_id).shape.dims();
        let b_dims = graph.node(b_id).shape.dims();
        let out_dims = parent.shape.dims();
        let (m, k, n) = if a_dims.len() >= 2 && b_dims.len() == 2 && out_dims.len() == a_dims.len()
        {
            let leading: usize = a_dims[..a_dims.len() - 2]
                .iter()
                .map(|d| d.unwrap_static())
                .product();
            let m_inner = a_dims[a_dims.len() - 2].unwrap_static();
            let k_inner = a_dims[a_dims.len() - 1].unwrap_static();
            let n_inner = b_dims[1].unwrap_static();
            ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
        } else if a_dims.len() == 2 && b_dims.len() == 2 {
            (
                a_dims[0].unwrap_static() as u32,
                a_dims[1].unwrap_static() as u32,
                b_dims[1].unwrap_static() as u32,
            )
        } else {
            continue; // unusual shape — let the regular FMB path handle
        };
        let cp = derive_matmul_compute(
            &dev.device,
            &graph,
            &coop_f16_vk_mirror_acts,
            a_id,
            b_id,
            m,
            k,
            n,
        );
        // F32 → matmul_qkv. CoopF32 → matmul_qkv_coop_f32. Both write
        // Q/K/V into the narrow output slots, so the narrows can be
        // elided. Coop16 still falls back to FMB+narrows (kernel
        // would need an f16-acc variant; deferred).
        if cp == MatmulCompute::F32 || cp == MatmulCompute::CoopF32 {
            // Slot-safety: the fused kernel writes Q/K/V into the narrow slots
            // *during the same dispatch that reads the matmul inputs* (A activation,
            // weight, bias). The memory planner, however, sees the narrows as born
            // at their (later) narrow step — after A's last read at this matmul —
            // so under liveness-aware reuse it may place a Q/K/V slot on top of a
            // still-being-read input (commonly Q reusing A's just-freed slot). The
            // fused write then clobbers A mid-dispatch and the K/V lanes that still
            // need A read corrupted input → nondeterministic drift on reused arenas
            // (bit-exact only under RLX_ARENA_NO_REUSE). Only elide the narrows when
            // no Q/K/V output aliases any matmul input; otherwise fall back to the
            // standard FMB (writes its own slot) + real Narrow steps, which read the
            // materialized parent after A is dead — always safe.
            let (q, k, v) = qkv;
            // The parent's own inputs are the common case, but they are not the
            // only tensor that can be sitting in a Q/K/V slot at the parent's
            // step. The planner believes Q/K/V are BORN at their (later) narrow
            // steps, so it is free to park ANY tensor whose live range spans the
            // parent there — and the fused write destroys it. `rlx-lightbeam` is
            // the repro: its GRU gate projection is shaped exactly like a QKV
            // split (one FMB → three axis-1 narrows of 24), and an unrelated
            // live activation shared a gate's slot, so wgpu alone drifted to
            // cos=0.958 while CPU/Metal/MLX/CoreML held 1e-8. Check every node
            // live across the parent's step, which subsumes the input check.
            let live_across = |t: NodeId| -> bool {
                graph.nodes().iter().any(|n| {
                    if n.id == t || n.id == parent_id || n.id == q || n.id == k || n.id == v {
                        return false;
                    }
                    let Some(&npos) = node_pos.get(&n.id) else {
                        return false;
                    };
                    let Some(&ppos) = node_pos.get(&parent_id) else {
                        return false;
                    };
                    if npos >= ppos || last_use.get(&n.id).copied().unwrap_or(0) < ppos {
                        return false;
                    }
                    arena_tensors_overlap(&arena, n.id, t)
                })
            };
            let unsafe_slot = live_across(q) || live_across(k) || live_across(v);
            if !unsafe_slot {
                qkv_split.insert(parent_id, qkv);
            }
        }
    }
    let qkv_skip_narrows: HashSet<NodeId> = qkv_split
        .values()
        .flat_map(|&(q, k, v)| [q, k, v])
        .collect();

    // Mel front-end framing: one `narrow`(+`reshape`) per frame, stacked by one
    // `concat`. On device that is two steps per frame for a single strided read
    // of the source, so the whole matrix is folded into one `Step::Frames`.
    // The split-QKV parents go with it: their step is replaced by `MatmulQkv`,
    // which writes the narrow slots rather than the parent's own.
    let split_qkv_parents: HashSet<NodeId> = qkv_split.keys().copied().collect();
    let mut static_weight_memo: HashMap<NodeId, bool> = HashMap::new();
    let (frames_skip, frames_plans) = detect_frames_pattern(
        &graph,
        &arena,
        dev,
        &node_pos,
        &last_use,
        &split_qkv_parents,
        &mut static_weight_memo,
    );

    // EEG-DINO / packed QKV: FMB → [B,S,3,H,D] → Narrow×3 (axis 2) → Attention.
    // Match CPU `compile_thunks` fused_strided_attn: read Q/K/V from the
    // packed parent with seq stride 3·H·D instead of materializing narrows.
    // (`attention_route::plan_attention` derives the same parent per node; the
    // set below is what the Narrow arm needs to elide the three narrows.)
    let mut packed_bshd_skip_narrows: HashSet<NodeId> = HashSet::new();
    if !rlx_ir::env::flag("RLX_WGPU_NO_PACKED_BSHD_ATTN") {
        for node in graph.nodes() {
            let Op::Attention { .. } = &node.op else {
                continue;
            };
            if node.inputs.len() < 3 {
                continue;
            }
            if let Some((_parent, _head_width, narrows)) = rlx_ir::detect_packed_bshd_qkv_attention(
                &graph,
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
            ) {
                for narrow in narrows {
                    if rlx_ir::packed_bshd_narrow_elidable(&graph, narrow, node.id) {
                        packed_bshd_skip_narrows.insert(narrow);
                    }
                }
            }
        }
    }

    // Detect (Add → LayerNorm) where Add has multi-consumer downstream.
    // The standard `FuseResidualLN` pass declines to fuse these (its
    // single-consumer guard forces materializing the sum); we collapse
    // them here at the wgpu lowering level via `Step::FusedResidualLnTee`.
    // Returns:
    //   ln_to_tee: ln_id  → (h, delta, gamma, beta, sum_arena_id)
    //   skip_adds: { add_id }  — these Add nodes are computed by the
    //                            tee step; their normal Step emission
    //                            is suppressed.
    let (mut ln_to_tee, mut skip_adds) = detect_residual_ln_tee_pattern(&graph);

    // ── Arena-safety veto for the tee ────────────────────────────────
    // The tee moves the Add's work to the NORM's step, so `h` and `delta` are
    // read at `ln_pos` rather than at `add_pos`. The memory planner never sees
    // that edge — this fold happens here, after planning, so as far as liveness
    // is concerned both operands died at the Add. Anything emitted in between
    // may therefore have been given their bytes, including the norm's OWN
    // output slot, and the tee then adds whatever overwrote them.
    //
    // `rlx-eegdm` is the repro: LayerNorm node 76's output was placed at the
    // same offset (526336) as `h` (node 68), so the norm read its own partly
    // written output as the residual. Every magnitude survived — same max, same
    // per-row |sum| — while the row means shifted, which is why it presented as
    // a plausible cos=0.994 rather than as garbage. Correct on CPU/Metal/MLX/
    // CoreML (no such fold) and correct here under RLX_ARENA_NO_REUSE=1 or
    // RLX_PIN_OUTPUT_ANCESTORS=1, both of which stop the slot being reused.
    //
    // Veto rather than re-plan: in a real transformer `h` feeds the *next*
    // residual add, so it is still live across the norm and no writer can take
    // its slot — the fold this exists for is kept. It is only dropped where it
    // was never sound.
    {
        let order: HashMap<NodeId, usize> = graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();
        let span = |id: NodeId| -> Option<(usize, usize)> {
            arena
                .has(id)
                .then(|| (arena.offset(id), arena.offset(id) + arena.len_of(id)))
        };
        let unsafe_tees: Vec<NodeId> = ln_to_tee
            .iter()
            .filter(|(ln_id, (h_id, delta_id, _, _, sum_id))| {
                let (Some(ln_pos), Some(add_pos)) =
                    (order.get(ln_id).copied(), order.get(sum_id).copied())
                else {
                    return true;
                };
                let operands: Vec<(usize, usize)> =
                    [*h_id, *delta_id].iter().filter_map(|&o| span(o)).collect();
                if operands.len() != 2 {
                    return true;
                }
                graph.nodes().iter().any(|n| {
                    let pos = order.get(&n.id).copied().unwrap_or(usize::MAX);
                    if pos <= add_pos || pos > ln_pos {
                        return false;
                    }
                    if rlx_compile::memory::is_pure_view(&graph, n) {
                        return false;
                    }
                    let Some((ws, we)) = span(n.id) else {
                        return false;
                    };
                    // The tee is one workgroup per row, and every pass in it
                    // strides over the same lanes (element index ≡ lane mod 64),
                    // so an arena element is only read and written by the lane
                    // that owns it — the reduction is the one cross-lane exchange
                    // and it goes through barrier-ordered workgroup memory. The
                    // norm aliasing an operand EXACTLY (same offset, same length)
                    // is therefore safe: row r's writes land on the lanes those
                    // same threads have already read. Only a partial or shifted
                    // overlap is fatal, and it only ever comes from a *different*
                    // dispatch, which clobbers the bytes before the tee runs.
                    let hit = operands.iter().any(|&(os, oe)| {
                        let overlaps = ws < oe && we > os;
                        let exact_self_alias = n.id == **ln_id && ws == os && we == oe;
                        overlaps && !exact_self_alias
                    });
                    if hit && rlx_ir::env::flag("RLX_WGPU_TEE_VETO_LOG") {
                        eprintln!(
                            "[tee-veto]   ln={:?} add={:?} h={:?}{:?} delta={:?}{:?} clobbered by {:?} {:?} [{},{})",
                            ln_id, sum_id, h_id, span(*h_id), delta_id, span(*delta_id),
                            n.id, n.op, ws, we
                        );
                    }
                    hit
                })
            })
            .map(|(ln_id, _)| *ln_id)
            .collect();
        if rlx_ir::env::flag("RLX_WGPU_TEE_VETO_LOG") {
            eprintln!(
                "[tee-veto] {} tee(s) detected, {} vetoed as arena-unsafe",
                ln_to_tee.len(),
                unsafe_tees.len()
            );
        }
        for ln_id in unsafe_tees {
            if let Some((_, _, _, _, sum_id)) = ln_to_tee.remove(&ln_id) {
                // The Add must emit its own step again now that nothing else
                // computes it.
                skip_adds.remove(&sum_id);
            }
        }
    }

    let mut coop_f16_host_activations: Vec<(NodeId, Activation, String)> = Vec::new();
    let mut static_once_steps: HashSet<usize> = HashSet::new();

    // Slot-safety for `static_once`: a static-weight pack may only skip recompute
    // on a graph re-run if it OWNS its arena slot exclusively. `plan_memory`'s
    // `extend_static_weight_pack_liveness` pins such packs to graph-end so they
    // get a private slot — but any pack the planner failed to pin shares a
    // liveness-reused slot, and a later node clobbers it, so run 2+ (bucketed LM
    // decode reuses one graph per token) reads stale data → zeroed lm_head
    // logits. Count materialized owners per arena offset; only offset-exclusive
    // (count==1) packs are marked run-once. Trusts the pin where it worked, falls
    // back to recompute where it didn't. Repro/verify: rlx-models backend_sweep.
    let mut offset_owner_count: HashMap<usize, usize> = HashMap::new();
    for n in graph.nodes() {
        if rlx_compile::memory::is_pure_view(&graph, n) || !arena.has(n.id) {
            continue;
        }
        *offset_owner_count.entry(arena.offset(n.id)).or_insert(0) += 1;
    }
    let slot_is_exclusive = |arena: &crate::buffer::Arena, id: NodeId| -> bool {
        arena.has(id) && offset_owner_count.get(&arena.offset(id)).copied() == Some(1)
    };

    let emit_uniform = |size: usize| -> wgpu::Buffer {
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu uniform"),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    // One permutation step, shared by the `Op::Transpose` arm and the many-row
    // attention route's operand compactions: `out_dims` are the output extents
    // and `in_strides` the source's stride for each output axis, so the
    // kernel's dense write lands where the caller asked. The axis metadata goes
    // in a storage buffer of its own (uniform arrays of u32 would pad every
    // element to 16 bytes). `arena` is a parameter rather than a capture so the
    // loop can still take it mutably (`register_bf16_packed`).
    let emit_transpose = |schedule: &mut Vec<Step>,
                          uniforms: &mut Vec<wgpu::Buffer>,
                          bind_groups: &mut Vec<wgpu::BindGroup>,
                          meta_buffers: &mut Vec<wgpu::Buffer>,
                          arena: &Arena,
                          label: &str,
                          base: u64,
                          size: u64,
                          out_dims: &[u32],
                          in_strides: &[u32],
                          bucket_outermost: u32,
                          in_off: u32,
                          out_off: u32| {
        let rank = out_dims.len();
        let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
        meta_data.extend_from_slice(out_dims);
        meta_data.extend_from_slice(in_strides);
        let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (meta_data.len() * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue
            .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
        let meta_idx = meta_buffers.len();
        meta_buffers.push(meta_buf);
        let p = TransposeParams {
            rank: rank as u32,
            out_total: out_dims.iter().product(),
            in_off,
            out_off,
            bucket_outermost,
            out_dim_0: out_dims[0],
            _p2: 0,
            _p3: 0,
        };
        schedule.push(Step::Transpose {
            params: p,
            meta_idx,
        });
        let tk = transpose_kernel(&dev.device);
        let u = emit_uniform(std::mem::size_of::<TransposeParams>());
        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu transpose bg"),
            layout: &tk.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: arena_bind_buf(arena, base).0,
                        offset: arena_bind_buf(arena, base).1,
                        size: {
                            let (b, l) = arena_bind_buf(arena, base);
                            aligned_bind_size(size, l, b.size())
                        },
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buffers[meta_idx].as_entire_binding(),
                },
            ],
        });
        uniforms.push(u);
        bind_groups.push(bg);
    };

    // Which node each emitted step belongs to. Recorded at the *top* of the next
    // iteration so the many `continue`s in the match below cannot skip it.
    let mut step_nodes: Vec<NodeId> = Vec::new();
    // Steps the per-binding-ceiling rule routed to the host route (see
    // `should_host_at_ceiling`); surfaced on `WgpuExecutable` so the route is
    // observable rather than inferred.
    let mut ceiling_hosted: usize = 0;
    // Convs whose shape qualified for im2col but which kept the direct/tiled route.
    // The shape half is what sizes the col reservation, so a qualifying shape that
    // still does not route is kept off it for a reason the shapes alone do not
    // settle — counted at that decision and reported (see the conv arm and the
    // schedule report).
    let mut im2col_fallbacks: usize = 0;
    // Many-row attention steps kept on the fused kernel for want of a window over
    // the route's regions — the one term of that route decision the graph's shapes
    // do not settle (see the attention arm and `attention_route`).
    let mut attention_window_declines: usize = 0;
    let mut prev_node: Option<NodeId> = None;
    for node in graph.nodes() {
        if let Some(prev) = prev_node {
            while step_nodes.len() < schedule.len() {
                step_nodes.push(prev);
            }
        }
        prev_node = Some(node.id);
        // ── The per-binding-ceiling rule (one rule, closed by default) ────────
        // Before every other routing rule, for the reason its own doc gives; see
        // `should_host_at_ceiling` and `supported_ops::addresses_arena_by_absolute_offset`.
        if should_host_at_ceiling(&dev.device, &arena, &graph, node) {
            ceiling_hosted += 1;
            schedule.push(host_op_step(&graph, node, &arena));
            continue;
        }
        // Helpers — capture device + arena into closures isn't
        // ergonomic in the loop, so inline the bind-group build
        // when each step is emitted below.
        let elems = node.shape.num_elements().unwrap_or(0) as u32;
        match &node.op {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
            // Virtually-sharded arenas: SPIR-V windowed elementwise / matmul /
            // norm paths collapse NSF (peak ~0.05). Host those ops with packed
            // per-tensor staging. Dedicated paths below still own Expand / Conv /
            // DequantMatMul / Custom / indexing / ConvTranspose.
            // Unsharded discrete: Binary/Activation stay on GPU; Expand/Conv/…
            // still host via wgpu_prefer_structure_host / wgpu_prefer_conv_host.
            op if should_pack_host_op(&arena, op, node, &graph) => {
                schedule.push(host_op_step(&graph, node, &arena));
                continue;
            }
            Op::MatMul => {
                let a_id = node.inputs[0];
                let b_id = node.inputs[1];
                let a_shape = graph.node(a_id).shape.dims();
                let b_shape = graph.node(b_id).shape.dims();
                let out_shape = node.shape.dims();
                // Three patterns:
                //   • 2D×2D                              → batch=1
                //   • [..,M,K] × [K,N]  (broadcast rhs)  → batch=1, flatten leading into M
                //   • [..,M,K] × [..,K,N] (matched batch)→ batch=prod(leading), per-batch strides
                let (m, k, n, batch, a_bs, b_bs, c_bs) = if a_shape.len() == 2
                    && b_shape.len() == 2
                    && out_shape.len() == 2
                {
                    (
                        a_shape[0].unwrap_static() as u32,
                        a_shape[1].unwrap_static() as u32,
                        b_shape[1].unwrap_static() as u32,
                        1u32,
                        0u32,
                        0u32,
                        0u32,
                    )
                } else if a_shape.len() >= 2
                    && b_shape.len() == 2
                    && out_shape.len() == a_shape.len()
                {
                    let leading: usize = a_shape[..a_shape.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                    let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                    let n_inner = b_shape[1].unwrap_static();
                    (
                        (leading * m_inner) as u32,
                        k_inner as u32,
                        n_inner as u32,
                        1u32,
                        0u32,
                        0u32,
                        0u32,
                    )
                } else if (a_shape.len() >= 3 || b_shape.len() >= 3)
                    && a_shape.len() >= 2
                    && b_shape.len() >= 2
                    && out_shape.len() == a_shape.len().max(b_shape.len())
                {
                    // Batched: leading PRODUCTS match the output, OR are 1 and
                    // BROADCAST across the batch (per-matrix stride 0 — reuse
                    // matrix 0). Covers a=[1,H,M,K]×b=[H,K,N] (rank mismatch)
                    // AND a=[1,M,K]×b=[B,K,N] (same-rank batch-1 broadcast).
                    let lead_a: usize = a_shape[..a_shape.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let lead_b: usize = b_shape[..b_shape.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let lead_out: usize = out_shape[..out_shape.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let ok = (lead_a == lead_out || lead_a == 1)
                        && (lead_b == lead_out || lead_b == 1)
                        && lead_out == lead_a.max(lead_b);
                    if !ok {
                        panic!(
                            "rlx-wgpu MatMul: batched leading-product mismatch \
                                a={a_shape:?} b={b_shape:?} out={out_shape:?}"
                        );
                    }
                    let b_count: usize = lead_out;
                    let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                    let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                    let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
                    let a_str = if lead_a == 1 && lead_out > 1 {
                        0
                    } else {
                        m_inner * k_inner
                    };
                    let b_str = if lead_b == 1 && lead_out > 1 {
                        0
                    } else {
                        k_inner * n_inner
                    };
                    (
                        m_inner as u32,
                        k_inner as u32,
                        n_inner as u32,
                        b_count as u32,
                        a_str as u32,
                        b_str as u32,
                        (m_inner * n_inner) as u32,
                    )
                } else if a_shape.len() == 2
                    && b_shape.len() == 1
                    && out_shape.len() == 2
                    && b_shape[0].unwrap_static()
                        == a_shape[1].unwrap_static() * out_shape[1].unwrap_static()
                {
                    // `b` is a FLATTENED `[K,N]` weight (shape `[K*N]`, row-major) —
                    // the conv2d-backward decomposition emits its GEMMs this way, and
                    // CPU/Metal/CUDA all accept it. The buffer layout is identical to a
                    // 2D `[K,N]`, so treat it exactly as the `[M,K]×[K,N]` case.
                    (
                        a_shape[0].unwrap_static() as u32,   // M
                        a_shape[1].unwrap_static() as u32,   // K
                        out_shape[1].unwrap_static() as u32, // N
                        1u32,
                        0u32,
                        0u32,
                        0u32,
                    )
                } else {
                    panic!(
                        "rlx-wgpu MatMul: unsupported shapes a={a_shape:?} b={b_shape:?} \
                            out={out_shape:?} (supported: 2D×2D, [..,M,K]×[K,N], [..,M,K]×[..,K,N], [M,K]×[K*N])"
                    );
                };
                let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                let b_bytes = arena.len_of(b_id) as u64;
                let mut compute_precision = derive_matmul_compute(
                    &dev.device,
                    &graph,
                    &coop_f16_vk_mirror_acts,
                    a_id,
                    b_id,
                    m,
                    k,
                    n,
                );
                if b_is_param
                    && b_bytes > ARENA_STAGE_CAP
                    && arena.param_fits_f16_mirror(b_id)
                    && !rlx_ir::env::flag("RLX_WGPU_NO_F16_MIRROR")
                    // A genuine BF16 weight is stored f32 in the arena and must
                    // stay bit-exact — reading it from the f16 shadow would drop
                    // bf16's exponent range. Keep it on the F32 path.
                    && !matches!(graph.node(b_id).shape.dtype(), rlx_ir::DType::BF16)
                {
                    compute_precision = MatmulCompute::F16;
                }
                // PACKED-BF16 weight path (plain tiled matmul only): a BF16
                // Param rhs stays PACKED (2 B/elem) in the bf16 side buffer and
                // is unpacked in-shader (matmul_bf16w) — reading HALF the B
                // bytes with no host bf16→f32 widen and no ne*4 arena upload,
                // bit-exact to a bf16-rounded f32 matmul. Only overrides the
                // plain F32 case (leaves any forced CoopF32 alone); the fused
                // bias/act matmul sites keep the widened-f32 path.
                if b_is_param
                    && compute_precision == MatmulCompute::F32
                    && matches!(graph.node(b_id).shape.dtype(), rlx_ir::DType::BF16)
                    && arena.bf16_packed_eligible(&dev.device, b_id)
                {
                    arena.register_bf16_packed(&dev.device, b_id);
                    compute_precision = MatmulCompute::Bf16Packed;
                }
                let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param)
                    && !matmul_b_from_packed_bf16(compute_precision, b_is_param);
                let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    node.id,
                    a_id,
                    b_id,
                    b_in_arena,
                );
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                // Only grow to cover B when B is actually bound through the arena.
                let expand_ids: &[NodeId] = if b_in_arena {
                    &[node.id, a_id, b_id]
                } else {
                    &[node.id, a_id]
                };
                arena_expand_bind_window(&arena, expand_ids, &mut base, &mut size, max_binding);
                let mut scratch = arena.scratch_off as u64;
                if param_anchor {
                    arena_ensure_scratch_for_window(&arena, &mut scratch, base, size);
                }
                if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                    // Large act-arena params stage into the bind window via
                    // `arena_off_in_bind_window` below. Weight-buffer params
                    // are tagged and skipped here. Do not assert in-window —
                    // VirtIO/DX12 ~2 GiB caps often leave lm_head outside the
                    // initial window; staging handles that.
                    let _b_off = arena.offset(b_id);
                    let _ = _b_off;
                }
                let a_off_f32 = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    a_id,
                    &mut base,
                    &mut size,
                );
                let b_off_f32 = if !b_in_arena {
                    // B is read from the f16 shadow buffer (separate binding);
                    // never bind/stage it through the arena window. Use the
                    // global arena word index — build_matmul_bind_group rebases
                    // it into the f16 weight window.
                    (arena.offset(b_id) / 4) as u32
                } else if b_is_param
                    && b_bytes > ARENA_STAGE_CAP
                    && arena_tensor_in_window(&arena, b_id, base, size)
                {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        b_id,
                        &mut base,
                        &mut size,
                    )
                };
                // Staging B can widen `base` after A was recorded.
                let a_off_f32 = if arena_tensor_in_window(&arena, a_id, base, size) {
                    arena_local_off_f32(&arena, a_id, base)
                } else {
                    a_off_f32
                };
                let b_off_f32 = if b_in_arena && arena_tensor_in_window(&arena, b_id, base, size) {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    b_off_f32
                };
                maybe_push_coop_f16_vk_casts(
                    &graph,
                    a_id,
                    b_id,
                    &coop_f16_vk_mirror_acts,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut uniforms,
                    &mut bind_groups,
                    &mm_cast,
                    compute_precision,
                    a_off_f32,
                    m,
                    k,
                    batch,
                    b_off_f32,
                    n,
                );
                schedule.push(Step::Matmul {
                    m,
                    k,
                    n,
                    batch,
                    a_batch_stride: a_bs,
                    b_batch_stride: b_bs,
                    c_batch_stride: c_bs,
                    a_off_f32,
                    b_off_f32,
                    c_off_f32: arena_local_off_f32(&arena, node.id, base),
                    has_bias: 0,
                    bias_off_f32: 0,
                    act_id: 0xFFFF,
                    b_is_param,
                    from_conv1x1: false,
                    compute_precision,
                });
                let b_off_global = (arena.offset(b_id) / 4) as u32;
                let b_off_bind = if b_is_param
                    && matches!(
                        compute_precision,
                        MatmulCompute::Coop16
                            | MatmulCompute::CoopF16Vk
                            | MatmulCompute::F16
                            | MatmulCompute::Bf16Packed
                    ) {
                    b_off_global
                } else {
                    b_off_f32
                };
                register_coop_f16_vk_b_param(
                    &mut coop_f16_b_param,
                    &param_offsets,
                    b_id,
                    b_off_bind,
                    compute_precision,
                );
                let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                let (bg, b_off_adj) = build_matmul_bind_group(
                    &dev.device,
                    mm_k,
                    mm_w,
                    &mm_f16w,
                    mm_bf16w,
                    &mm_f16c,
                    &mm_coop,
                    &mm_coop_f32,
                    &arena,
                    base,
                    size,
                    &u,
                    b_is_param,
                    compute_precision,
                    k,
                    n,
                    batch,
                    b_off_bind,
                    b_bs,
                );
                if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                    *b_off_f32 = b_off_adj;
                }
                uniforms.push(u);
                bind_groups.push(bg);
                if compute_precision == MatmulCompute::CoopF16Vk {
                    coop_f16_vk_wide_bind_groups.insert(
                        schedule.len() - 1,
                        bind_arena_window(
                            &dev.device,
                            mm_w_active_compile,
                            &arena,
                            base,
                            size,
                            &uniforms[uniforms.len() - 1],
                        ),
                    );
                }
            }
            Op::Binary(bop) => {
                // Skip emit when this Add is consumed by a downstream
                // FRLTee — the tee step writes the sum to this node's
                // arena slot directly. Subsequent consumers read the
                // same slot and find correct data.
                if skip_adds.contains(&node.id) {
                    continue;
                }
                // Complex binary: C64 add/sub/mul/div reads BOTH `[re, im]`
                // lanes per element, so it cannot ride the fused scalar-per-
                // thread path — lower to a standalone `binary_c64` dispatch
                // (excluded from region fusion). C128 arithmetic is out of
                // scope (CPU has none either) → reject. Broadcast is carried
                // by per-operand element counts (`k % n_x`), matching CPU.
                let out_dt = node.shape.dtype();
                if out_dt.is_complex() {
                    if out_dt == rlx_ir::DType::C128 {
                        panic!(
                            "rlx-wgpu Binary on C128: complex-f64 arithmetic is \
                             unsupported (rlx-cpu has none either) — only C64 \
                             add/sub/mul/div are wired"
                        );
                    }
                    let op_code = binary_op_id(*bop);
                    if op_code > 3 {
                        panic!(
                            "rlx-wgpu C64 Binary: {bop:?} is undefined for complex \
                             (only Add/Sub/Mul/Div); matches rlx-cpu rejection"
                        );
                    }
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let win_ids = [node.id, a_id, b_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let a_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        b_id,
                        &mut base,
                        &mut size,
                    );
                    let c_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let a_off = if arena_tensor_in_window(&arena, a_id, base, size) {
                        arena_local_off_f32(&arena, a_id, base)
                    } else {
                        a_off
                    };
                    let b_off = if arena_tensor_in_window(&arena, b_id, base, size) {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        b_off
                    };
                    let c_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                        arena_local_off_f32(&arena, node.id, base)
                    } else {
                        c_off
                    };
                    let n_a = graph.node(a_id).shape.num_elements().unwrap_or(0) as u32;
                    let n_b = graph.node(b_id).shape.num_elements().unwrap_or(0) as u32;
                    let p = BinaryC64Params {
                        n: elems,
                        a_off,
                        b_off,
                        c_off,
                        op: op_code,
                        n_a: n_a.max(1),
                        n_b: n_b.max(1),
                        _p0: 0,
                    };
                    schedule.push(Step::BinaryC64 { params: p });
                    // If the output was staged into scratch, copy it back.
                    let out_src = arena.offset(node.id) as u64;
                    if !arena_tensor_in_window(&arena, node.id, base, size) {
                        let staged_dst = base + (c_off as u64) * 4;
                        schedule.push(Step::BufferCopy {
                            src_byte_off: staged_dst,
                            dst_byte_off: out_src,
                            bytes: arena.len_of(node.id) as u32,
                        });
                    }
                    let u = emit_uniform(std::mem::size_of::<BinaryC64Params>());
                    let bg = bind_arena_window(
                        &dev.device,
                        binary_c64_kernel(&dev.device),
                        &arena,
                        base,
                        size,
                        &u,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }
                // Binary now broadcasts in-kernel; anything `simple_broadcast`
                // cannot describe still arrives with an Expand prologue, so the
                // only shapes left to reject are the ones neither path handles.
                for &id in &node.inputs {
                    let src = &graph.node(id).shape;
                    assert!(
                        src.num_elements() == node.shape.num_elements()
                            || rlx_unfuse::simple_broadcast(&node.shape, src).is_some(),
                        "rlx-wgpu Binary: operand {:?} broadcasts to {:?} in a way \
                         `(i/rep)%len` cannot express and no Expand prologue was \
                         inserted",
                        src.dims(),
                        node.shape.dims(),
                    );
                }
                let a_id = node.inputs[0];
                let b_id = node.inputs[1];
                let win_ids = [node.id, a_id, b_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let a_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    a_id,
                    &mut base,
                    &mut size,
                );
                let b_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    b_id,
                    &mut base,
                    &mut size,
                );
                let c_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                // Staging B/C can widen `base` after A was recorded.
                let a_off = if arena_tensor_in_window(&arena, a_id, base, size) {
                    arena_local_off_f32(&arena, a_id, base)
                } else {
                    a_off
                };
                let b_off = if arena_tensor_in_window(&arena, b_id, base, size) {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    b_off
                };
                let c_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    c_off
                };
                // A broadcast operand is read through `(i / rep) % len`
                // instead of being materialised: `rlx_unfuse` leaves the mismatch
                // in place when `simple_broadcast` can describe it, and
                // `elide_binary_broadcast_expands` peels the copy the runtime
                // path's `LegalizeBroadcast` inserted for the same shapes.
                // Anything `simple_broadcast` could not describe is still an
                // Expand and lands here dense.
                let bcast = |id: rlx_ir::NodeId| -> (u32, u32) {
                    let src = &graph.node(id).shape;
                    if src.num_elements() == node.shape.num_elements() {
                        return (1, 0); // dense
                    }
                    rlx_unfuse::simple_broadcast(&node.shape, src).unwrap_or((1, 0))
                };
                let (a_rep, a_len) = bcast(a_id);
                let (b_rep, b_len) = bcast(b_id);
                let p = BinaryParams {
                    n: elems,
                    a_off,
                    b_off,
                    c_off,
                    op: binary_op_id(*bop),
                    a_rep,
                    a_len,
                    b_rep,
                    b_len,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Binary { params: p });
                // If the output was staged into scratch, copy it back.
                let out_src = arena.offset(node.id) as u64;
                if !arena_tensor_in_window(&arena, node.id, base, size) {
                    // c_off is scratch-relative; recover absolute scratch byte off.
                    let staged_dst = base + (c_off as u64) * 4;
                    schedule.push(Step::BufferCopy {
                        src_byte_off: staged_dst,
                        dst_byte_off: out_src,
                        bytes: arena.len_of(node.id) as u32,
                    });
                }
                let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                let bg = bind_arena_window(&dev.device, bk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Compare(cop) => {
                require_equal_shapes(&graph, &node.inputs, "Compare");
                let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                let a_id = node.inputs[0];
                let b_id = node.inputs[1];
                let a_src = arena.offset(a_id) as u64;
                let b_src = arena.offset(b_id) as u64;
                let a_len = arena.len_of(a_id) as u64;
                let b_len = arena.len_of(b_id) as u64;
                let a_in = a_src >= base && a_src + a_len <= base + size;
                let b_in = b_src >= base && b_src + b_len <= base + size;
                let a_dst = arena.scratch_off as u64;
                let a_aligned = a_len.div_ceil(256) * 256;
                let b_dst = a_dst + a_aligned;
                if a_dst < base || b_dst + b_len > base + size {
                    base = (arena.size as u64).saturating_sub(size);
                    base = (base / 256) * 256;
                }
                let a_off = if a_in {
                    arena_local_off_f32(&arena, a_id, base)
                } else {
                    if a_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Compare staging operand A too large ({a_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: a_src,
                        dst_byte_off: a_dst,
                        bytes: a_len as u32,
                    });
                    ((a_dst.saturating_sub(base)) / 4) as u32
                };
                let b_off = if b_in {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    if b_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Compare staging operand B too large ({b_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: b_src,
                        dst_byte_off: b_dst,
                        bytes: b_len as u32,
                    });
                    ((b_dst.saturating_sub(base)) / 4) as u32
                };
                let p = BinaryParams {
                    n: elems,
                    a_off,
                    b_off,
                    c_off: arena_local_off_f32(&arena, node.id, base),
                    op: compare_op_id(*cop),
                    // Compare keeps the Expand prologue — the opt-out in
                    // `needs_broadcast_prologue` is scoped to `Op::Binary`.
                    a_rep: 1,
                    a_len: 0,
                    b_rep: 1,
                    b_len: 0,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Compare { params: p });
                let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Activation(act) => {
                if coop_f16_vk_mirror_acts.contains(&node.id) {
                    let src_name = tensor_host_name(&input_offsets, &param_offsets, node.inputs[0]);
                    coop_f16_host_activations.push((node.id, *act, src_name));
                    continue;
                }
                let in_id = node.inputs[0];
                let win_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let p = UnaryParams {
                    n: elems,
                    in_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    op: activation_op_id(*act),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::Unary {
                    params: p,
                    f16_mirror: false,
                });
                let u = emit_uniform(std::mem::size_of::<UnaryParams>());
                let bg = bind_arena_window(&dev.device, uk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Where => {
                let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                let cond_id = node.inputs[0];
                let x_id = node.inputs[1];
                let y_id = node.inputs[2];
                let cond_src = arena.offset(cond_id) as u64;
                let x_src = arena.offset(x_id) as u64;
                let y_src = arena.offset(y_id) as u64;
                let cond_len = arena.len_of(cond_id) as u64;
                let x_len = arena.len_of(x_id) as u64;
                let y_len = arena.len_of(y_id) as u64;
                let cond_in = cond_src >= base && cond_src + cond_len <= base + size;
                let x_in = x_src >= base && x_src + x_len <= base + size;
                let y_in = y_src >= base && y_src + y_len <= base + size;
                let cond_dst = arena.scratch_off as u64;
                let cond_aligned = cond_len.div_ceil(256) * 256;
                let x_dst = cond_dst + cond_aligned;
                let x_aligned = x_len.div_ceil(256) * 256;
                let y_dst = x_dst + x_aligned;
                if cond_dst < base || y_dst + y_len > base + size {
                    base = (arena.size as u64).saturating_sub(size);
                    base = (base / 256) * 256;
                }
                let cond_off = if cond_in {
                    arena_local_off_f32(&arena, cond_id, base)
                } else {
                    if cond_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Where staging cond too large ({cond_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: cond_src,
                        dst_byte_off: cond_dst,
                        bytes: cond_len as u32,
                    });
                    ((cond_dst.saturating_sub(base)) / 4) as u32
                };
                let x_off = if x_in {
                    arena_local_off_f32(&arena, x_id, base)
                } else {
                    if x_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Where staging x too large ({x_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: x_src,
                        dst_byte_off: x_dst,
                        bytes: x_len as u32,
                    });
                    ((x_dst.saturating_sub(base)) / 4) as u32
                };
                let y_off = if y_in {
                    arena_local_off_f32(&arena, y_id, base)
                } else {
                    if y_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Where staging y too large ({y_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: y_src,
                        dst_byte_off: y_dst,
                        bytes: y_len as u32,
                    });
                    ((y_dst.saturating_sub(base)) / 4) as u32
                };
                let p = WhereParams {
                    n: elems,
                    cond_off,
                    x_off,
                    y_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Where { params: p });
                let u = emit_uniform(std::mem::size_of::<WhereParams>());
                let bg = bind_arena_window(&dev.device, wk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Fma => {
                let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                let a_id = node.inputs[0];
                let b_id = node.inputs[1];
                let c_id = node.inputs[2];
                let a_src = arena.offset(a_id) as u64;
                let b_src = arena.offset(b_id) as u64;
                let c_src = arena.offset(c_id) as u64;
                let a_len = arena.len_of(a_id) as u64;
                let b_len = arena.len_of(b_id) as u64;
                let c_len = arena.len_of(c_id) as u64;
                let a_in = a_src >= base && a_src + a_len <= base + size;
                let b_in = b_src >= base && b_src + b_len <= base + size;
                let c_in = c_src >= base && c_src + c_len <= base + size;
                let a_dst = arena.scratch_off as u64;
                let a_aligned = a_len.div_ceil(256) * 256;
                let b_dst = a_dst + a_aligned;
                let b_aligned = b_len.div_ceil(256) * 256;
                let c_dst = b_dst + b_aligned;
                if a_dst < base || c_dst + c_len > base + size {
                    base = (arena.size as u64).saturating_sub(size);
                    base = (base / 256) * 256;
                }
                let a_off = if a_in {
                    arena_local_off_f32(&arena, a_id, base)
                } else {
                    if a_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Fma staging a too large ({a_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: a_src,
                        dst_byte_off: a_dst,
                        bytes: a_len as u32,
                    });
                    ((a_dst.saturating_sub(base)) / 4) as u32
                };
                let b_off = if b_in {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    if b_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Fma staging b too large ({b_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: b_src,
                        dst_byte_off: b_dst,
                        bytes: b_len as u32,
                    });
                    ((b_dst.saturating_sub(base)) / 4) as u32
                };
                let c_off = if c_in {
                    arena_local_off_f32(&arena, c_id, base)
                } else {
                    if c_len > 64 * 1024 * 1024 {
                        panic!("rlx-wgpu: Fma staging c too large ({c_len} bytes)");
                    }
                    schedule.push(Step::BufferCopy {
                        src_byte_off: c_src,
                        dst_byte_off: c_dst,
                        bytes: c_len as u32,
                    });
                    ((c_dst.saturating_sub(base)) / 4) as u32
                };
                let p = FmaParams {
                    n: elems,
                    a_off,
                    b_off,
                    c_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Fma { params: p });
                let u = emit_uniform(std::mem::size_of::<FmaParams>());
                let bg = bind_arena_window(&dev.device, fk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::ReluBackward => {
                let x_id = node.inputs[0];
                let dy_id = node.inputs[1];
                let win_ids = [node.id, x_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let p = ActivationBackwardParams {
                    n: elems,
                    x_off,
                    dy_off,
                    dx_off: arena_local_off_f32(&arena, node.id, base),
                    op: 0, // relu
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::ReluBackward { params: p });
                let u = emit_uniform(std::mem::size_of::<ActivationBackwardParams>());
                let bg = bind_arena_window(&dev.device, abk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::ActivationBackward { kind } => {
                let x_id = node.inputs[0];
                let dy_id = node.inputs[1];
                let win_ids = [node.id, x_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let p = ActivationBackwardParams {
                    n: elems,
                    x_off,
                    dy_off,
                    dx_off: arena_local_off_f32(&arena, node.id, base),
                    op: activation_op_id(*kind),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::ActivationBackward { params: p });
                let u = emit_uniform(std::mem::size_of::<ActivationBackwardParams>());
                let bg = bind_arena_window(&dev.device, abk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => {
                assert_region_activations_implemented(node, chain);
                let n = *num_batch_inputs as usize;
                if n == 0 || chain.len() > 32 {
                    panic!(
                        "rlx-wgpu BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                        chain.len()
                    );
                }
                let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                    .expect("batch region static shape");
                let mut win_ids: Vec<NodeId> = vec![node.id];
                win_ids.extend(node.inputs.iter().copied());
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let mut chain_enc = rlx_ir::encode_chain_steps(chain);
                patch_region_cast_subs(chain, &mut chain_enc);
                let tail = rlx_ir::encode_prologue_tail(*prologue, &slice_shape, *prologue_input);
                let base_dst = arena_local_off_f32(&arena, node.id, base);
                let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                if use_single {
                    let mut batch_input_offs = [0u32; 64];
                    for i in 0..n {
                        batch_input_offs[i] = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            node.inputs[i],
                            &mut base,
                            &mut size,
                        );
                    }
                    let p = BatchElementwiseRegionParams {
                        slice_len: slice_elems,
                        num_batch: n as u32,
                        num_steps: chain.len() as u32,
                        base_dst_off: base_dst,
                        slice_elems,
                        batch_input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                    };
                    schedule.push(Step::BatchElementwiseRegion { params: p });
                    let ek = batch_elementwise_region_kernel(&dev.device);
                    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu batch region params"),
                        size: std::mem::size_of::<BatchElementwiseRegionParams>() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg = bind_arena_window(&dev.device, ek, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                } else {
                    let spatial = tail[0] == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW;
                    let ek = if spatial {
                        elementwise_region_spatial_kernel(&dev.device)
                    } else {
                        elementwise_region_kernel(&dev.device)
                    };
                    for i in 0..n {
                        let mut input_offs = [0u32; 16];
                        input_offs[0] = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            node.inputs[i],
                            &mut base,
                            &mut size,
                        );
                        let p = ElementwiseRegionParams {
                            len: slice_elems,
                            num_inputs: 1,
                            num_steps: chain.len() as u32,
                            dst_off: rlx_ir::batch_region_slice_dst_off_f32(
                                base_dst,
                                slice_elems,
                                i,
                            ),
                            input_offs,
                            chain: chain_enc,
                            scalar_input_mask: *scalar_input_mask,
                            prologue: tail[0],
                            out_n: tail[1],
                            out_c: tail[2],
                            out_h: tail[3],
                            out_w: tail[4],
                            prologue_input: tail[5],
                            input_modulus: *input_modulus,
                        };
                        schedule.push(Step::ElementwiseRegion { params: p });
                        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("rlx-wgpu batch region params"),
                            size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let bg = bind_arena_window(&dev.device, ek, &arena, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }
            }
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => {
                // PLAN L2 native lowering. Encode the chain into a
                // fixed-size u32 buffer; one uniform per region.
                let n = *num_inputs as usize;
                if n > 16 || chain.len() > 32 {
                    panic!(
                        "rlx-wgpu ElementwiseRegion: chain too large \
                            (inputs={n}, steps={}). Caps: 16 / 32. \
                            Use UnfuseElementwiseRegions to fall back.",
                        chain.len()
                    );
                }
                let mut win_ids: Vec<NodeId> = vec![node.id];
                win_ids.extend(node.inputs.iter().copied());
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let mut input_offs = [0u32; 16];
                for (i, &id) in node.inputs.iter().enumerate() {
                    input_offs[i] = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        id,
                        &mut base,
                        &mut size,
                    );
                }
                let mut chain_enc = rlx_ir::encode_chain_steps(chain);
                patch_region_cast_subs(chain, &mut chain_enc);
                let tail = rlx_ir::encode_prologue_tail(*prologue, &node.shape, *prologue_input);
                let p = ElementwiseRegionParams {
                    len: elems,
                    num_inputs: *num_inputs,
                    num_steps: chain.len() as u32,
                    dst_off: arena_local_off_f32(&arena, node.id, base),
                    input_offs,
                    chain: chain_enc,
                    scalar_input_mask: *scalar_input_mask,
                    prologue: tail[0],
                    out_n: tail[1],
                    out_c: tail[2],
                    out_h: tail[3],
                    out_w: tail[4],
                    prologue_input: tail[5],
                    input_modulus: *input_modulus,
                };
                schedule.push(Step::ElementwiseRegion { params: p });
                let ek = if p.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW {
                    elementwise_region_spatial_kernel(&dev.device)
                } else {
                    elementwise_region_kernel(&dev.device)
                };
                // STORAGE (not UNIFORM) — the WGSL params struct
                // contains `array<u32, N>` arrays whose 4-byte
                // stride violates uniform's 16-byte stride rule.
                let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu region params"),
                    size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = bind_arena_window(&dev.device, ek, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Reduce {
                op: rop,
                axes,
                keep_dim: _,
            } => {
                // Single-axis reduce OR contiguous multi-axis reduce.
                // The kernel walks the input as `[outer, reduce_dim,
                // inner]` — for contiguous axes [k..k+m], we set
                // `reduce_dim = product(dims[k..k+m])`.
                // Non-contiguous reductions are not yet wired (no
                // model has hit them); transposing into contiguous
                // form first is the future fix.
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                let contiguous = sorted.windows(2).all(|w| w[1] == w[0] + 1);
                if !contiguous {
                    // Non-contiguous axes: packed HostOp (CPU thunk).
                    schedule.push(host_op_step(&graph, node, &arena));
                    continue;
                }
                let ax_first = sorted[0];
                let ax_last = *sorted.last().unwrap();
                let dims_u32: Vec<u32> =
                    in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                let outer: u32 = dims_u32[..ax_first].iter().product();
                let reduce_dim: u32 = dims_u32[ax_first..=ax_last].iter().product();
                let inner: u32 = dims_u32[ax_last + 1..].iter().product();
                let red_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let red_fits = arena_span_bytes(&arena, &red_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &red_ids,
                );
                if !red_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let p = ReduceParams {
                    outer,
                    reduce_dim,
                    inner,
                    in_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    op: reduce_op_id(*rop),
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::Reduce { params: p });
                let rk = reduce_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ReduceParams>());
                let bg = bind_arena_window(&dev.device, rk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Softmax { axis } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let last = (in_shape.len() - 1) as i32;
                if *axis != -1 && *axis != last {
                    // Non-last-axis Softmax: packed HostOp (CPU thunk).
                    schedule.push(host_op_step(&graph, node, &arena));
                    continue;
                }
                let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let sm_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let sm_fits = arena_span_bytes(&arena, &sm_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &sm_ids,
                );
                if !sm_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                // Staging either side can widen `base`; recompute locals.
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_off
                };
                let p = SoftmaxParams {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::Softmax { params: p });
                let sk = softmax_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<SoftmaxParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::SoftmaxCrossEntropy => {
                // Dense / soft-label cross-entropy: logits [N,C], targets
                // [N,C] → loss [N]. One thread per row (outer=N, inner=C).
                // Window must cover both inputs + the [N] output slot;
                // mirrors the LayerNorm multi-input arena-window dance.
                let logits_id = node.inputs[0];
                let targets_id = node.inputs[1];
                let logits_shape = graph.node(logits_id).shape.dims();
                let inner = logits_shape[logits_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = logits_shape
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product();
                let outer = total / inner.max(1);

                let sce_win = vec![node.id, logits_id, targets_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let sce_fits = arena_span_bytes(&arena, &sce_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &sce_win,
                );
                if !sce_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let logits_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    logits_id,
                    &mut base,
                    &mut size,
                );
                let targets_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    targets_id,
                    &mut base,
                    &mut size,
                );
                let p = SceParams {
                    outer,
                    inner,
                    logits_off,
                    targets_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::SoftmaxCrossEntropy { params: p });
                let sk = softmax_cross_entropy_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<SceParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::SoftmaxCrossEntropyWithLogits => {
                // Integer-label CE: logits [N,C], labels [N] → loss [N].
                let logits_id = node.inputs[0];
                let labels_id = node.inputs[1];
                let logits_shape = graph.node(logits_id).shape.dims();
                let inner = logits_shape[logits_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = logits_shape
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product();
                let outer = total / inner.max(1);

                let sce_win = vec![node.id, logits_id, labels_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let sce_fits = arena_span_bytes(&arena, &sce_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &sce_win,
                );
                if !sce_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let logits_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    logits_id,
                    &mut base,
                    &mut size,
                );
                let labels_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    labels_id,
                    &mut base,
                    &mut size,
                );
                let p = SceParams {
                    outer,
                    inner,
                    logits_off,
                    targets_off: labels_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::SoftmaxCrossEntropyWithLogits { params: p });
                let sk = softmax_cross_entropy_with_logits_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<SceParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::SoftmaxCrossEntropyBackward => {
                // Integer-label CE bwd: [logits, labels, d_loss] → dlogits.
                let logits_id = node.inputs[0];
                let labels_id = node.inputs[1];
                let d_loss_id = node.inputs[2];
                let logits_shape = graph.node(logits_id).shape.dims();
                let inner = logits_shape[logits_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = logits_shape
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product();
                let outer = total / inner.max(1);

                let sce_win = vec![node.id, logits_id, labels_id, d_loss_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let sce_fits = arena_span_bytes(&arena, &sce_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &sce_win,
                );
                if !sce_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let logits_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    logits_id,
                    &mut base,
                    &mut size,
                );
                let labels_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    labels_id,
                    &mut base,
                    &mut size,
                );
                let d_loss_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    d_loss_id,
                    &mut base,
                    &mut size,
                );
                let p = SceBwdParams {
                    outer,
                    inner,
                    logits_off,
                    labels_off,
                    d_loss_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::SoftmaxCrossEntropyBackward { params: p });
                let sk = softmax_cross_entropy_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<SceBwdParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let is_layer_norm = matches!(&node.op, Op::LayerNorm { .. });

                // FRLTee fast path: if this LN is the head of a
                // (multi-consumer Add → LN) pattern, emit one
                // `Step::FusedResidualLnTee` that writes the sum to
                // the eliminated Add's arena slot AND the LN result
                // to this LN's slot. The Add itself is skipped
                // upstream (`skip_adds`).
                if let Some(&(h_id, delta_id, gamma_id, beta_id, sum_id)) = ln_to_tee.get(&node.id)
                {
                    let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                    let gamma_bytes = arena.len_of(gamma_id) as u64;
                    let frlt_win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                        vec![gamma_id, node.id, h_id, delta_id, beta_id, sum_id]
                    } else {
                        vec![node.id, h_id, delta_id, gamma_id, beta_id, sum_id]
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &frlt_win,
                    );
                    if !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        h_id,
                        &mut base,
                        &mut size,
                    );
                    let residual_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        delta_id,
                        &mut base,
                        &mut size,
                    );
                    let sum_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        sum_id,
                        &mut base,
                        &mut size,
                    );
                    let gamma_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        gamma_id,
                        &mut base,
                        &mut size,
                    );
                    let beta_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        beta_id,
                        &mut base,
                        &mut size,
                    );
                    // Same stale-offset trap as FusedResidualLN: staging
                    // gamma/beta can widen `base`, leaving earlier locals
                    // pointing at the wrong stripe (zeros / wrong tensors).
                    let in_off = if arena_tensor_in_window(&arena, h_id, base, size) {
                        arena_local_off_f32(&arena, h_id, base)
                    } else {
                        in_off
                    };
                    let residual_off = if arena_tensor_in_window(&arena, delta_id, base, size) {
                        arena_local_off_f32(&arena, delta_id, base)
                    } else {
                        residual_off
                    };
                    let sum_off = if arena_tensor_in_window(&arena, sum_id, base, size) {
                        arena_local_off_f32(&arena, sum_id, base)
                    } else {
                        sum_off
                    };
                    let gamma_off = if arena_tensor_in_window(&arena, gamma_id, base, size) {
                        arena_local_off_f32(&arena, gamma_id, base)
                    } else {
                        gamma_off
                    };
                    let beta_off = if arena_tensor_in_window(&arena, beta_id, base, size) {
                        arena_local_off_f32(&arena, beta_id, base)
                    } else {
                        beta_off
                    };
                    let p = FusedResidualLnTeeParams {
                        outer,
                        inner,
                        in_off,
                        residual_off,
                        bias_off: 0, // FRLTee currently no-bias only
                        gamma_off,
                        beta_off,
                        sum_off,
                        ln_out_off: arena_local_off_f32(&arena, node.id, base),
                        eps_bits: eps.to_bits(),
                        has_bias: 0,
                        // The detector accepts both norms; the kernel picks the
                        // scaling. RmsNorm is the Llama-class case and was the
                        // whole point of widening this path.
                        is_rms: u32::from(!is_layer_norm),
                    };
                    schedule.push(Step::FusedResidualLnTee { params: p });
                    let frtk = fused_residual_ln_tee_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FusedResidualLnTeeParams>());
                    let bg = bind_arena_window(&dev.device, frtk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }

                let gamma_id = node.inputs[1];
                // beta is the third input for BOTH LayerNorm and RmsNorm — the
                // RmsNorm kernel branch now adds it, matching the CPU oracle.
                let beta_id = if node.inputs.len() >= 3 {
                    node.inputs[2]
                } else {
                    // No beta input (shouldn't happen for RmsNorm/LayerNorm);
                    // gamma's offset is a benign placeholder.
                    gamma_id
                };
                let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                let gamma_bytes = arena.len_of(gamma_id) as u64;
                let ln_win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                    vec![gamma_id, node.id, in_id]
                } else {
                    let mut v = vec![node.id, in_id];
                    if gamma_is_param {
                        v.push(gamma_id);
                    }
                    if is_layer_norm {
                        v.push(beta_id);
                    }
                    v
                };
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let ln_fits = arena_span_bytes(&arena, &ln_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &ln_win,
                );
                if !ln_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let gamma_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    gamma_id,
                    &mut base,
                    &mut size,
                );
                let beta_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    beta_id,
                    &mut base,
                    &mut size,
                );
                // Same stale-offset trap as FusedResidualLN/FusedResidualLnTee:
                // staging gamma/beta after `in_off` was resolved can widen or
                // relocate `base`, leaving `in_off` pointing at the wrong
                // stripe (reads zeros/garbage — this is the plain, non-tee
                // LayerNorm/RmsNorm path used by every DiT block).
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let gamma_off = if arena_tensor_in_window(&arena, gamma_id, base, size) {
                    arena_local_off_f32(&arena, gamma_id, base)
                } else {
                    gamma_off
                };
                let beta_off = if arena_tensor_in_window(&arena, beta_id, base, size) {
                    arena_local_off_f32(&arena, beta_id, base)
                } else {
                    beta_off
                };
                let p = LayerNormParams {
                    outer,
                    inner,
                    in_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    gamma_off,
                    beta_off,
                    eps_bits: eps.to_bits(),
                    op: if is_layer_norm { 0 } else { 1 },
                };
                schedule.push(Step::LayerNorm { params: p });
                let lk = layernorm_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<LayerNormParams>());
                let bg = bind_arena_window(&dev.device, lk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::FusedSwiGLU { gate_first, .. } => {
                // Native SwiGLU: one kernel over the [.., 2*n_half] input instead
                // of the Narrow+Silu+Mul decompose (3 arena intermediates/FFN).
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let n2 = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let n_half = n2 / 2;
                let out_total: u32 = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product();
                let outer = out_total / n_half.max(1);
                // Bind a window covering both the input and output (both
                // activations); resolve each offset relative to that window.
                let sg_win = vec![node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let sg_fits = arena_span_bytes(&arena, &sg_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &sg_win,
                );
                if !sg_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                // Re-resolve if staging relocated `base` (stale-offset trap).
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_off
                };
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let p = FusedSwiGLUParams {
                    n_half,
                    outer,
                    gate_first: *gate_first as u32,
                    in_off,
                    out_off,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::FusedSwiGLU { params: p });
                let k = fused_swiglu_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<FusedSwiGLUParams>());
                let bg = bind_arena_window(&dev.device, k, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Reshape { .. } => {
                // A row of the mel front-end's framing: a view of a frame narrow
                // whose value `Step::Frames` writes into the concat's slot
                // directly, so the row's own slot is never read. Skipped before
                // the aliasing check below rather than through it — the narrow it
                // views is skipped too, and a `BufferCopy` here would be a step
                // for a tensor nothing reads.
                if frames_skip.contains(&node.id) {
                    continue;
                }
                // Normally a view-alias (planner / from_plan_split). If the
                // output lands on a distinct slot, copy like Cast — otherwise
                // the slot stays zero and Expand/Add of Param biases die.
                let in_id = node.inputs[0];
                let src = arena.offset(in_id);
                let dst = arena.offset(node.id);
                if src != dst {
                    let bytes = arena.len_of(in_id).min(arena.len_of(node.id));
                    schedule.push(Step::BufferCopy {
                        src_byte_off: src as u64,
                        dst_byte_off: dst as u64,
                        bytes: bytes as u32,
                    });
                }
            }

            Op::Cast { to } => {
                // The wgpu arena is f32-uniform (every value stored as f32), so
                // a Cast is a per-element re-encode of the already-f32 value.
                // Three regimes, matching rlx-cpu:
                //   * identity (mode 0): int→float / float→float / same-kind /
                //     Bool→num — the stored f32 already holds the right value, so
                //     the proven `BufferCopy` value-move suffices.
                //   * float→int (mode 1): must truncate toward zero and SATURATE
                //     to the destination int range (Rust `x as iN` semantics);
                //     a plain copy WRONGLY kept 3.7 as 3.7. Needs a compute pass.
                //   * →Bool (mode 2): store `value != 0` (1.0/0.0); a plain copy
                //     WRONGLY kept the source magnitude.
                let in_id = node.inputs[0];
                let src_dt = graph.node(in_id).shape.dtype();
                let dst_dt = *to;
                // F64 has no f32-arena storage AND this backend performs no
                // upstream f64→f32 demotion (unlike rlx-vulkan/rlx-oneapi) —
                // reject rather than silently corrupt. (df64 C128 IS carried,
                // 4 f32 lanes, so complex casts fall through to the pass below.)
                if matches!(src_dt, rlx_ir::DType::F64) || matches!(dst_dt, rlx_ir::DType::F64) {
                    panic!(
                        "rlx-wgpu Op::Cast {src_dt:?}->{dst_dt:?}: the f32-uniform \
                         arena has no F64 storage and this backend does not \
                         demote it to f32 — run this cast on CPU/Metal"
                    );
                }
                // Complex casts route to the standalone `complex_cast` pass
                // (real↔C64, real↔C128, C64↔C128 — pure lane moves, no df64
                // compensated math since f32 real sources have lo=0). Excluded
                // from region fusion (the f32-scalar region kernel can't re-pair
                // complex lanes). Same-dtype complex is Identity (BufferCopy —
                // lane-correct now the slot spans every lane).
                if src_dt.is_complex() || dst_dt.is_complex() {
                    if src_dt == dst_dt {
                        let src = arena.offset(in_id);
                        let dst = arena.offset(node.id);
                        if src != dst {
                            let bytes = arena.len_of(in_id).min(arena.len_of(node.id));
                            schedule.push(Step::BufferCopy {
                                src_byte_off: src as u64,
                                dst_byte_off: dst as u64,
                                bytes: bytes as u32,
                            });
                        }
                        continue;
                    }
                    let mode: u32 = match (src_dt, dst_dt) {
                        (s, rlx_ir::DType::C64) if !s.is_complex() => 0, // real → C64
                        (rlx_ir::DType::C64, d) if !d.is_complex() => 1, // C64 → real
                        (s, rlx_ir::DType::C128) if !s.is_complex() => 2, // real → C128
                        (rlx_ir::DType::C128, d) if !d.is_complex() => 3, // C128 → real
                        (rlx_ir::DType::C64, rlx_ir::DType::C128) => 4,
                        (rlx_ir::DType::C128, rlx_ir::DType::C64) => 5,
                        _ => unreachable!("complex cast {src_dt:?}->{dst_dt:?} unclassified"),
                    };
                    // Bind a window covering both operands (staged into scratch
                    // when they don't fit), mirroring the numeric cast path.
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = ComplexCastParams {
                        n: elems,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        mode,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::ComplexCast { params: p });
                    let u = emit_uniform(std::mem::size_of::<ComplexCastParams>());
                    let bg = bind_arena_window(
                        &dev.device,
                        complex_cast_kernel(&dev.device),
                        &arena,
                        base,
                        size,
                        &u,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }
                let (mode, lo, hi) = if dst_dt == rlx_ir::DType::Bool {
                    (2u32, 0.0f32, 0.0f32)
                } else if dst_dt.is_int() && src_dt.is_float() {
                    let (lo, hi) = int_range_f32(dst_dt);
                    (1u32, lo, hi)
                } else {
                    // int→float / float→float(incl. F16/BF16, kept full f32 in
                    // this arena) / int→int (widening; narrowing-wrap isn't
                    // representable here) / Bool→num — all value-preserving.
                    (0u32, 0.0f32, 0.0f32)
                };
                let src = arena.offset(in_id);
                let dst = arena.offset(node.id);
                if mode == 0 {
                    // Value-preserving: same-slot alias is a no-op; distinct
                    // slots copy (handles arbitrary offsets / >4 GiB tensors).
                    if src != dst {
                        let bytes = arena.len_of(in_id).min(arena.len_of(node.id));
                        schedule.push(Step::BufferCopy {
                            src_byte_off: src as u64,
                            dst_byte_off: dst as u64,
                            bytes: bytes as u32,
                        });
                    }
                } else {
                    // Numeric conversion → compute pass. Bind a window covering
                    // both operands (staged into scratch if they don't fit),
                    // mirroring Op::Activation. In-place (src==dst) is fine: each
                    // thread reads then writes only its own element.
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = CastParams {
                        n: elems,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        mode,
                        lo_bits: lo.to_bits(),
                        hi_bits: hi.to_bits(),
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::Cast { params: p });
                    let u = emit_uniform(std::mem::size_of::<CastParams>());
                    let bg = bind_arena_window(
                        &dev.device,
                        cast_kernel(&dev.device),
                        &arena,
                        base,
                        size,
                        &u,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }

            Op::Transpose { perm } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let out_shape = node.shape.dims();
                let base_rank = perm.len();
                if base_rank != in_shape.len() || base_rank != out_shape.len() {
                    panic!("rlx-wgpu Transpose: rank mismatch");
                }
                let mut in_dims: Vec<u32> =
                    in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                let mut out_dims: Vec<u32> =
                    out_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                // Complex tensors pack `lanes` contiguous f32 per element (C64=2
                // [re,im], C128=4 df64). The permute kernel copies one f32 per
                // "element" via the perm-indexed strides, so append an INNERMOST
                // lane axis that maps to ITSELF (never permuted): each thread
                // then copies a whole complex element's lanes as a contiguous
                // group instead of shattering the [re,im] pairing. lanes=1 for
                // real/int ⇒ strict no-op (axis is elided).
                let lanes: u32 = match node.shape.dtype() {
                    rlx_ir::DType::C64 => 2,
                    rlx_ir::DType::C128 => 4,
                    _ => 1,
                };
                let mut perm = perm.clone();
                let rank = if lanes > 1 {
                    in_dims.push(lanes);
                    out_dims.push(lanes);
                    perm.push(base_rank); // lane axis maps to itself (innermost)
                    base_rank + 1
                } else {
                    base_rank
                };
                // Input cumulative strides (row-major) over the extended rank.
                let mut in_strides = vec![1u32; rank];
                for i in (0..rank.saturating_sub(1)).rev() {
                    in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
                }
                // For each *output* axis i, the corresponding input
                // axis is perm[i] — its stride is in_strides[perm[i]].
                let strides_for_out: Vec<u32> = (0..rank).map(|i| in_strides[perm[i]]).collect();
                if wgpu_prefer_structure_host(&arena)
                    || rlx_ir::env::flag("RLX_WGPU_TRANSPOSE_HOST")
                {
                    schedule.push(Step::TransposeHost {
                        in_byte_off: arena.offset(in_id),
                        out_byte_off: arena.offset(node.id),
                        in_dims,
                        out_dims,
                        in_strides: strides_for_out
                            .into_iter()
                            .map(|stride| stride as usize)
                            .collect(),
                        param_operand: crate::buffer::is_parameter_storage(&graph, in_id),
                    });
                    continue;
                }

                // PLAN L1: precompute "bucket axis stays at out
                // axis 0" flag from perm. When `perm[0] == 0`,
                // active-extent scaling of `out_total` is safe.
                let bucket_outermost = if perm[0] == 0 { 1u32 } else { 0u32 };
                let tr_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                // Prefer a window that covers the *output*. Anchoring on a
                // Param (weight buffer / far stripe) and then "staging" the
                // act output only copies zeros *into* scratch — the kernel
                // writes the transpose there and never writebacks to the
                // real act slot. F5 AdaLN `Transpose(linear.weight)` then
                // leaves MatMul B all-zero → Gemm == bias.
                //
                // On sharded arenas always prefer the output stripe and stage
                // the weight in. Unsharded: cover both when the span fits.
                let (mut base, mut size) = if arena.is_sharded() {
                    if arena_span_bytes(&arena, &tr_ids) <= max_binding
                        && !arena_span_crosses_shard(&arena, &tr_ids)
                    {
                        arena_window_for_nodes(&dev.device, &arena, &tr_ids)
                    } else {
                        arena_window_for_nodes(&dev.device, &arena, &[node.id])
                    }
                } else if arena_span_bytes(&arena, &tr_ids) <= max_binding {
                    arena_window_for_nodes(&dev.device, &arena, &tr_ids)
                } else {
                    arena_window_for_nodes(&dev.device, &arena, &[node.id])
                };
                arena_expand_bind_window(&arena, &tr_ids, &mut base, &mut size, max_binding);
                let mut scratch = arena.scratch_off as u64;
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                if rlx_ir::env::flag("RLX_WGPU_DBG_TRANSPOSE") {
                    eprintln!(
                        "[wgpu-transpose] node {} perm {:?} in_dims {:?} out_dims {:?} \
                         slot_in {} slot_out {} | base {base} size {size} in_off {in_off} \
                         out_off {out_off} out_total {} scratch {scratch}",
                        node.id.0,
                        perm,
                        in_dims,
                        out_dims,
                        arena.offset(in_id),
                        arena.offset(node.id),
                        elems * lanes,
                    );
                    let (bb, bo) = arena_bind_buf(&arena, base);
                    eprintln!(
                        "                 sharded={} max_bind={} bind_buf_size={} bind_buf_off={} \
                         out_abs_byte={}",
                        arena.is_sharded(),
                        dev.device.limits().max_storage_buffer_binding_size,
                        bb.size(),
                        bo,
                        (out_off as u64) * 4 + bo,
                    );
                }
                emit_transpose(
                    &mut schedule,
                    &mut uniforms,
                    &mut bind_groups,
                    &mut meta_buffers,
                    &arena,
                    "rlx-wgpu transpose meta",
                    base,
                    size,
                    &out_dims,
                    &strides_for_out,
                    bucket_outermost,
                    in_off,
                    out_off,
                );
            }

            Op::Narrow { axis, start, len } => {
                // Part of a split-QKV pattern: the parent FMB has been
                // (or will be) replaced by Step::MatmulQkv that writes
                // directly into this narrow's arena slot. Skip the
                // narrow's own dispatch.
                //
                // Or a frame of the mel front-end's framing, which
                // `Step::Frames` at the concat below reads straight out of this
                // narrow's source — so this slot is written by nobody and read by
                // nobody (its single consumer is that concat, proven in
                // `detect_frames_pattern`).
                if qkv_skip_narrows.contains(&node.id)
                    || packed_bshd_skip_narrows.contains(&node.id)
                    || frames_skip.contains(&node.id)
                {
                    continue;
                }
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let outer: u32 = in_shape[..*axis]
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let inner: u32 = in_shape[*axis + 1..]
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let axis_in = in_shape[*axis].unwrap_static() as u32;
                if wgpu_prefer_structure_host(&arena) {
                    let lanes = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    schedule.push(Step::NarrowHost {
                        in_byte_off: arena.offset(in_id),
                        out_byte_off: arena.offset(node.id),
                        outer,
                        inner: inner * lanes,
                        axis_in_size: axis_in,
                        start: *start as u32,
                        axis_out_size: *len as u32,
                        param_operand: crate::buffer::is_parameter_storage(&graph, in_id),
                    });
                    continue;
                }
                let win_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_off
                };
                // Complex packs `lanes` contiguous f32 per element (C64=2,
                // C128=4). Narrow copies one f32 per "element"; the narrow axis
                // is an ELEMENT axis and the lane axis is innermost, so only the
                // per-copy inner-count and the output total scale by lanes —
                // axis/start/len stay in element units. lanes=1 ⇒ strict no-op.
                let lanes: u32 = match node.shape.dtype() {
                    rlx_ir::DType::C64 => 2,
                    rlx_ir::DType::C128 => 4,
                    _ => 1,
                };
                let p = NarrowConcatParams {
                    total: elems * lanes,
                    outer,
                    inner: inner * lanes,
                    axis_in_size: axis_in,
                    axis_out_size: *len as u32,
                    start: *start as u32,
                    in_off,
                    out_off,
                };
                schedule.push(Step::Narrow { params: p });
                let nk = narrow_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                let bg = bind_arena_window(&dev.device, nk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            // In-place KV append. The output aliases the cache (the planner
            // gives it input 0's slot via `pure_view_offset`), so the whole op
            // is one contiguous copy of the new row to `pos` along `axis`.
            //
            // `outer == 1` is guaranteed here, not assumed:
            // `rewrite_for_backend` lowers any `KvAppend` whose `[..pos+1]`
            // prefix is not contiguous — anything with a dim > 1 before `axis`
            // — back to narrow + concat on every backend. That is what makes a
            // single `BufferCopy` sufficient instead of a strided kernel.
            Op::KvAppend { axis, pos } => {
                let cache = graph.node(node.inputs[0]);
                let inner_elems: usize = (*axis + 1..cache.shape.rank())
                    .map(|i| cache.shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                // The arena is f32-UNIFORM: every real/int/bool tensor occupies
                // one f32 lane per element whatever its dtype says, and only
                // complex genuinely spans several (C64 = 2 lanes, C128 = 4).
                // `dtype().size_bytes()` is therefore the wrong stride here —
                // it strides an F16 cache by 2 bytes through a buffer laid out
                // 4 bytes per element, so the write covers half the row and
                // leaves the rest of the token stale. `Step::BufferCopy` also
                // moves whole f32 words on every one of its paths, so a lane
                // count is the only unit that works end to end.
                let elem_bytes = if cache.shape.dtype().is_complex() {
                    cache.shape.dtype().size_bytes()
                } else {
                    4
                };
                let inner_bytes = inner_elems * elem_bytes;
                schedule.push(Step::BufferCopy {
                    src_byte_off: arena.offset(node.inputs[1]) as u64,
                    dst_byte_off: (arena.offset(node.id) + pos * inner_bytes) as u64,
                    bytes: inner_bytes as u32,
                });
            }
            Op::Concat { axis } => {
                // Mel front-end framing: every input is a frame narrow (or a view
                // of one) sliced out of one source at a uniform hop, so the whole
                // concat is one strided read of that source. One dispatch instead
                // of one per frame, and no index table (see
                // `detect_frames_pattern` for the clauses this stands on).
                if let Some(plan) = frames_plans.get(&node.id) {
                    let p = FramesParams {
                        src_off: (arena.offset(plan.src) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        t: plan.t,
                        outer: plan.outer,
                        frame_len: plan.frame_len,
                        hop: plan.hop,
                        total: plan.outer * plan.n_frames * plan.frame_len,
                        _p0: 0,
                    };
                    schedule.push(Step::Frames { params: p });
                    let fk = frames_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FramesParams>());
                    // Whole-arena bind: `detect_frames_pattern` refuses a sharded
                    // arena or one past the binding ceiling precisely so these
                    // absolute word offsets address the arena they were taken in.
                    let bg = bind_op_output_window(&dev.device, fk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }
                let out_shape = node.shape.dims();
                let outer: u32 = out_shape[..*axis]
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let inner: u32 = out_shape[*axis + 1..]
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let axis_out = out_shape[*axis].unwrap_static() as u32;
                // Complex packs `lanes` contiguous f32 per element (C64=2,
                // C128=4). Concat copies one f32 per "element"; the concat axis
                // is an ELEMENT axis (lane axis innermost), so only the per-slice
                // inner-count and the input-element total scale by lanes —
                // axis_out/start stay element units. lanes=1 ⇒ strict no-op. The
                // host fallbacks (ConcatHost/Pieces) keep the element-unit `inner`.
                let lanes: u32 = match node.shape.dtype() {
                    rlx_ir::DType::C64 => 2,
                    rlx_ir::DType::C128 => 4,
                    _ => 1,
                };
                let inner_lanes = inner * lanes;
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let once = is_static_weight_tensor(&graph, node.id, &mut static_weight_memo);
                let sched_before = schedule.len();

                // Prefer GPU Concat. Cross-shard tiles are staged into the
                // output stripe's reserve (one piece at a time) — F5 packs
                // ~22 weight tiles on shard 0 + one stray on another shard;
                // a full ConcatHost of the 0.5 GiB result every NFE was both
                // slow and host-heavy vs ORT I/O binding.
                let pair_ok = |in_id: NodeId| {
                    // Discrete Vulkan/DX12 + virtually-sharded: GPU Concat
                    // collapses Kitten NSF — host every piece.
                    if wgpu_prefer_structure_host(&arena)
                        || rlx_ir::env::flag("RLX_WGPU_CONCAT_HOST")
                    {
                        return false;
                    }
                    // Weight-buffer inputs cannot share an act bind window —
                    // force the stage/host path instead of a bogus local off.
                    if crate::buffer::is_weight_off(arena.offset(in_id)) {
                        return false;
                    }
                    let pair = [node.id, in_id];
                    arena_span_bytes(&arena, &pair) <= max_binding
                        && !arena_span_crosses_shard(&arena, &pair)
                };
                let out_byte = arena.offset(node.id) as u64;
                let mut scratch = if arena.is_sharded() {
                    arena.shard_stage_off(out_byte as usize) as u64
                } else {
                    arena.scratch_off as u64
                };
                let stage_end = if arena.is_sharded() {
                    scratch + crate::buffer::shard_stage_reserve() as u64
                } else {
                    (arena.size as u64).saturating_add(1)
                };
                let mut start_pos: u32 = 0;
                let mut host_inputs: Vec<(usize, u32, u32)> = Vec::new();
                let mut host_starts: Vec<u32> = Vec::new();
                let mut host_param_operands: Vec<bool> = Vec::new();
                for &in_id in &node.inputs {
                    let in_shape = graph.node(in_id).shape.dims();
                    let axis_in = in_shape[*axis].unwrap_static() as u32;
                    let in_total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let numel = graph.node(in_id).shape.num_elements().unwrap_or(0) as u32;
                    let in_bytes = arena.len_of(in_id) as u64;
                    if pair_ok(in_id) {
                        let (base, size) =
                            arena_window_for_nodes(&dev.device, &arena, &[node.id, in_id]);
                        let in_off = arena_local_off_f32(&arena, in_id, base);
                        let out_off = arena_local_off_f32(&arena, node.id, base);
                        let p = NarrowConcatParams {
                            total: in_total * lanes,
                            outer,
                            inner: inner_lanes,
                            axis_in_size: axis_in,
                            axis_out_size: axis_out,
                            start: start_pos,
                            in_off,
                            out_off,
                        };
                        schedule.push(Step::Concat { params: p });
                        let cck = concat_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                        let bg = bind_arena_window(&dev.device, cck, &arena, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else if !wgpu_prefer_structure_host(&arena)
                        && in_bytes > 0
                        && in_bytes <= crate::buffer::shard_stage_reserve() as u64
                        && scratch.saturating_add(in_bytes.div_ceil(256) * 256) <= stage_end
                    {
                        // GPU Concat with scratch staging — skipped when
                        // discrete / sharded structure host is preferred.
                        let aligned = in_bytes.div_ceil(256) * 256;
                        let dst = scratch;
                        scratch = scratch.saturating_add(aligned);
                        schedule.push(Step::BufferCopy {
                            src_byte_off: arena.offset(in_id) as u64,
                            dst_byte_off: dst,
                            bytes: in_bytes as u32,
                        });
                        let (base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                        let in_off = ((dst.saturating_sub(base)) / 4) as u32;
                        let out_off = arena_local_off_f32(&arena, node.id, base);
                        let p = NarrowConcatParams {
                            total: in_total * lanes,
                            outer,
                            inner: inner_lanes,
                            axis_in_size: axis_in,
                            axis_out_size: axis_out,
                            start: start_pos,
                            in_off,
                            out_off,
                        };
                        schedule.push(Step::Concat { params: p });
                        let cck = concat_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                        let bg = bind_arena_window(&dev.device, cck, &arena, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        // Complex: the sharded host fallback copies f32 LANES, so
                        // each input's element count scales by `lanes` (C64=2/
                        // C128=4). The per-row `inner` span uses `inner_lanes`
                        // below; axis positions (`axis_in`/`start_pos`) stay
                        // element units (the lane axis is innermost).
                        host_inputs.push((arena.offset(in_id), axis_in, numel * lanes));
                        host_starts.push(start_pos);
                        host_param_operands
                            .push(crate::buffer::is_parameter_storage(&graph, in_id));
                    }
                    start_pos += axis_in;
                }
                if !host_inputs.is_empty() {
                    if host_inputs.len() == node.inputs.len() {
                        schedule.push(Step::ConcatHost {
                            dst_byte_off: arena.offset(node.id),
                            outer,
                            inner: inner_lanes,
                            total_axis: axis_out,
                            inputs: host_inputs,
                            param_operands: host_param_operands,
                        });
                    } else {
                        schedule.push(Step::ConcatHostPieces {
                            dst_byte_off: arena.offset(node.id),
                            outer,
                            inner: inner_lanes,
                            total_axis: axis_out,
                            inputs: host_inputs,
                            starts: host_starts,
                            param_operands: host_param_operands,
                        });
                    }
                }
                if once && slot_is_exclusive(&arena, node.id) {
                    for i in sched_before..schedule.len() {
                        static_once_steps.insert(i);
                    }
                }
            }

            Op::Attention {
                score_scale,
                attn_logit_softcap,
                ..
            } => {
                let q_id = node.inputs[0];
                let k_id = node.inputs[1];
                let v_id = node.inputs[2];
                // One derivation of this step's geometry, shared with the
                // scratch reserve (`backend::compute_scratch_bytes`). A shape it
                // cannot size is one the fused path below cannot index either —
                // except the over-ceiling head width, which that path sizes and
                // its kernel would then return from unwritten — and the refusal
                // names the operator and the dim that was wrong.
                let plan = match attention_route::plan_attention(&graph, node) {
                    Ok(plan) => plan,
                    Err(refusal) => panic!("{refusal}"),
                };
                let attention_route::AttentionPlan {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim: hd,
                    v_head_dim: hd_v,
                    nkv,
                    packed_parent,
                    q_b,
                    q_h,
                    q_s,
                    k_b,
                    k_h,
                    k_s,
                    v_b,
                    v_h,
                    v_s,
                    o_b,
                    o_h,
                    o_s,
                    mask_kind_id,
                    window,
                    m_b,
                    m_h,
                    m_q,
                    m_k,
                    q_dense,
                    v_dense,
                    out_dense,
                    decompose,
                    ..
                } = plan;
                let scale = score_scale.unwrap_or(1.0_f32 / (hd as f32).sqrt());
                // `attn_logit_softcap`, bitcast for the uniform: `c·tanh(s/c)`
                // over the scaled score, before the mask and the softmax. `None`
                // and any non-positive value are disabled — in the shader, for
                // both arrangements below (`kernels/attention.wgsl`).
                let cap_bits = attn_logit_softcap.unwrap_or(0.0_f32).to_bits();

                // ── The bind window both forms below read through ────────────
                //
                // One resolution of the operand window and of the five operand
                // offsets, so the many-row route and the fused path cannot
                // disagree about where an operand is. Staging an operand that
                // falls outside the window widens it, so every offset is read
                // back out of the window the resolve leaves behind — and the
                // same read-back runs wherever a later step (the fused path's
                // alias staging) can widen it under offsets already recorded.
                let mask_id = if mask_kind_id == 2 || mask_kind_id == 4 {
                    Some(node.inputs[3])
                } else {
                    None
                };
                let mut attn_ids = if let Some((parent, _)) = packed_parent {
                    vec![node.id, parent]
                } else {
                    vec![node.id, q_id, k_id, v_id]
                };
                if let Some(mid) = mask_id {
                    attn_ids.push(mid);
                }
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &attn_ids,
                );
                if !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                // An operand whose own slot the window reaches is addressed
                // there; one it does not reach keeps the offset the resolve left
                // — the copy it staged into the window.
                let rebase = |base: u64, size: u64, prev: (u32, u32, u32, u32, u32)| {
                    let read = |id: NodeId, off: u32| {
                        if arena_tensor_in_window(&arena, id, base, size) {
                            arena_local_off_f32(&arena, id, base)
                        } else {
                            off
                        }
                    };
                    let (q, k, v, out, mask) = prev;
                    let (q, k, v) = if let Some((parent, head_width)) = packed_parent {
                        // One parent slot holds all three: Q/K/V are its own
                        // axis-2 narrows, so they keep the parent's rebase.
                        let p = if arena_tensor_in_window(&arena, parent, base, size) {
                            arena_local_off_f32(&arena, parent, base)
                        } else {
                            q
                        };
                        (
                            p,
                            p.saturating_add(head_width),
                            p.saturating_add(head_width * 2),
                        )
                    } else {
                        (read(q_id, q), read(k_id, k), read(v_id, v))
                    };
                    let out = read(node.id, out);
                    let mask = match mask_id {
                        Some(mid) => read(mid, mask),
                        None => mask,
                    };
                    (q, k, v, out, mask)
                };
                let resolve_operands = |base: &mut u64,
                                        size: &mut u64,
                                        scratch: &mut u64,
                                        schedule: &mut Vec<Step>|
                 -> (u32, u32, u32, u32, u32) {
                    let id_off = |id: NodeId,
                                  schedule: &mut Vec<Step>,
                                  scratch: &mut u64,
                                  base: &mut u64,
                                  size: &mut u64| {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            schedule,
                            scratch,
                            id,
                            base,
                            size,
                        )
                    };
                    let (q, k, v) = if let Some((parent, head_width)) = packed_parent {
                        let p = id_off(parent, schedule, scratch, base, size);
                        (
                            p,
                            p.saturating_add(head_width),
                            p.saturating_add(head_width * 2),
                        )
                    } else {
                        (
                            id_off(q_id, schedule, scratch, base, size),
                            id_off(k_id, schedule, scratch, base, size),
                            id_off(v_id, schedule, scratch, base, size),
                        )
                    };
                    let out = id_off(node.id, schedule, scratch, base, size);
                    let mask = match mask_id {
                        Some(mid) => id_off(mid, schedule, scratch, base, size),
                        None => 0,
                    };
                    rebase(*base, *size, (q, k, v, out, mask))
                };
                let out_byte = arena.offset(node.id) as u64;
                let out_len = arena.len_of(node.id) as u64;
                let (mut q_off, mut k_off, mut v_off, mut out_off, mut mask_off) =
                    resolve_operands(&mut base, &mut size, &mut scratch, &mut schedule);
                // An output slot the fused path below overlaps with any operand:
                // its threads read all of K and V while writing the output, so it
                // writes through scratch and copies back (see below).
                let out_aliases_qkv = arena_tensors_overlap(&arena, node.id, q_id)
                    || arena_tensors_overlap(&arena, node.id, k_id)
                    || arena_tensors_overlap(&arena, node.id, v_id)
                    || packed_parent
                        .is_some_and(|(parent, _)| arena_tensors_overlap(&arena, node.id, parent));
                // The route reads its operands in steps of its own, so only V can be
                // in its way: the second multiply writes the output while it reads
                // V, so V is compacted through the region the plan books for exactly
                // this (`v_spare`) instead of being read where it lies. Q and K are
                // read before anything is written, so an output over those — or over
                // a packed QKV parent, whose V is compacted anyway — needs no such
                // step.
                let v_dense = v_dense && !arena_tensors_overlap(&arena, node.id, v_id);

                // ── Many-row attention: the dense arrangement ────────────────
                //
                // Above the measured query-row floor the fused kernel below — one
                // thread per (batch, head, query row), each walking the whole key
                // sequence on its own — is an order of magnitude behind what the
                // same work reaches as a dense multiply of the scores, a masked
                // scale over them and a dense multiply by V (`attention_route`
                // carries the measurements). That arrangement is emitted here as
                // steps rather than as a graph decomposition because on a discrete
                // adapter the corresponding *ops* would be hosted; as steps, every
                // part of it stays on the device.
                //
                // This branch is additive: below the floor, and for every step it
                // does not serve, the fused path below runs exactly as it did.
                //
                // The route's own regions begin at the scratch cursor the operand
                // resolve above left, and the plan sizes every one of them as a
                // 256-byte multiple (the arena's own window granularity) plus one
                // slack for the cursor's own alignment, so the bump below keeps
                // each region aligned and stays inside what the plan reserved.
                // `v_compacted` is this arm's answer — an output slot on V's bytes
                // cannot be written by the multiply that reads V — and the plan
                // books that compaction as `v_spare`, so the offsets cover it.
                let offsets = plan.scratch_offsets(!v_dense);
                let route_start = scratch.div_ceil(256) * 256;
                let route_end = route_start + offsets.end;
                // The window the operands were resolved against has to reach the
                // regions too, since the route's steps bind their operands and
                // their scratch as one range. True whenever the arena binds
                // whole, which is the case the route is measured in, and
                // otherwise only where that window already reaches the tail. A
                // striped arena keeps the fused path instead: its per-stripe
                // stage reserve is a different zone from the tail the plan
                // reserves, and the fused path stages each operand into whatever
                // window it has. A shape that does not fit falls back rather than
                // failing at compile time.
                let route_fits = !arena.is_sharded()
                    && route_start >= base
                    && route_end <= base.saturating_add(size);
                // Every other term of the decision is the graph's own
                // (`decompose`, of which the two dispatch extents are part); the
                // two arena terms below are the arm's, so they are what the count
                // and the print exist for.
                if decompose && !route_fits {
                    attention_window_declines += 1;
                }
                if rlx_ir::env::flag("RLX_WGPU_DEBUG_ATTN_MASK") {
                    let route = if !decompose {
                        "fused"
                    } else if arena.is_sharded() {
                        "fused (the arena is addressed in stripes)"
                    } else if !route_fits {
                        "fused (the bind window does not reach the plan's regions)"
                    } else {
                        "decomposed"
                    };
                    eprintln!(
                        "rlx-wgpu Attention node={:?} mask_kind_id={mask_kind_id} \
                         q={:?}@{} k={:?}@{} v={:?}@{} route={route}",
                        node.id,
                        q_id,
                        arena.offset(q_id),
                        k_id,
                        arena.offset(k_id),
                        v_id,
                        arena.offset(v_id),
                    );
                }
                if decompose && route_fits {
                    // The regions the route bumps through, in the plan's fixed
                    // order (Kᵀ, the scores, then the compactions the operand
                    // layouts force). Each is already a 256-byte multiple, so
                    // summing them keeps every offset aligned.
                    let window_f32 = |byte: u64| ((byte.saturating_sub(base)) / 4) as u32;
                    let kt_off = window_f32(route_start + offsets.kt);
                    let scores_off = window_f32(route_start + offsets.scores);
                    let q_tmp_off = window_f32(route_start + offsets.q);
                    let v_tmp_off = window_f32(route_start + offsets.v);
                    let out_tmp_off = window_f32(route_start + offsets.out);
                    // GQA/MQA: the group of KV heads each KV head serves, as the
                    // fused kernel derives it (`heads / kv_heads`).
                    let group = (heads / nkv.max(1)).max(1);
                    // One dense multiply over the two arena operands the step
                    // reads, `batch = B·H` tiles. Deliberately F32: the score and
                    // value multiplies are f32 and this route is not the place for
                    // `derive_matmul_compute`'s device-dependent variants (coop /
                    // f16 weights), which are keyed on a weight operand this step
                    // has none of. The run-time arm's `m ≥ 32 && n ≥ 64` branch
                    // picks the wide f32 kernel for these shapes — the machinery
                    // the route's measurements were made with.
                    let emit_matmul = |schedule: &mut Vec<Step>,
                                       uniforms: &mut Vec<wgpu::Buffer>,
                                       bind_groups: &mut Vec<wgpu::BindGroup>,
                                       m: u32,
                                       k: u32,
                                       n: u32,
                                       a_off: u32,
                                       b_off: u32,
                                       c_off: u32| {
                        let batch_bh = batch * heads;
                        schedule.push(Step::Matmul {
                            m,
                            k,
                            n,
                            batch: batch_bh,
                            a_batch_stride: m * k,
                            b_batch_stride: k * n,
                            c_batch_stride: m * n,
                            a_off_f32: a_off,
                            b_off_f32: b_off,
                            c_off_f32: c_off,
                            has_bias: 0,
                            bias_off_f32: 0,
                            act_id: 0xFFFF,
                            b_is_param: false,
                            from_conv1x1: false,
                            compute_precision: MatmulCompute::F32,
                        });
                        let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                        let (bg, _) = build_matmul_bind_group(
                            &dev.device,
                            mm_k,
                            mm_w,
                            &mm_f16w,
                            mm_bf16w,
                            &mm_f16c,
                            &mm_coop,
                            &mm_coop_f32,
                            &arena,
                            base,
                            size,
                            &u,
                            false,
                            MatmulCompute::F32,
                            k,
                            n,
                            batch_bh,
                            b_off,
                            k * n,
                        );
                        uniforms.push(u);
                        bind_groups.push(bg);
                    };

                    // 1. Kᵀ: the score multiply needs K as `[k=D, n=S_k]`, and
                    //    this is also where a grouped KV head becomes one tile per
                    //    query head — the head axis splits into (kv head, group)
                    //    with the group's stride 0, which is the only form of the
                    //    fused kernel's `kv_h = h / group` a stride-driven permute
                    //    can express.
                    emit_transpose(
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        &mut meta_buffers,
                        &arena,
                        "rlx-wgpu attention route transpose meta",
                        base,
                        size,
                        &[batch, nkv, group, hd, seq_k],
                        &[k_b, k_h, 0, 1, k_s],
                        // Never the active-extent bucket (`0`): the route's own
                        // steps walk the query rows, not the batch, so a folded
                        // extent could not leave exactly the rows the caller asked
                        // for — this flag is what the step's own predicate reads.
                        // (Such a graph is off that mode anyway: the route's masked
                        // scale declines it.)
                        0,
                        k_off,
                        kt_off,
                    );
                    // 2. Q′: only when Q's tile is not already a dense `[S_q, D]`.
                    if !q_dense {
                        emit_transpose(
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mut meta_buffers,
                            &arena,
                            "rlx-wgpu attention route transpose meta",
                            base,
                            size,
                            &[batch, heads, seq_q, hd],
                            &[q_b, q_h, q_s, 1],
                            0,
                            q_off,
                            q_tmp_off,
                        );
                    }
                    // 3. scores = Q′·Kᵀ, dense `[B, H, S_q, S_k]` in scratch.
                    emit_matmul(
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        seq_q,
                        hd,
                        seq_k,
                        if q_dense { q_off } else { q_tmp_off },
                        kt_off,
                        scores_off,
                    );
                    // 4. The operation's own normalisation, in place over those
                    //    scores: the scale, the cap and the mask as one step, then
                    //    the ordinary dense softmax over the last axis.
                    let ap = AttentionMaskParams {
                        rows: batch * heads * seq_q,
                        cols: seq_k,
                        seq_q,
                        heads,
                        scale_bits: scale.to_bits(),
                        mask_kind: mask_kind_id,
                        window,
                        scores_off,
                        mask_off,
                        mask_batch_stride: m_b,
                        mask_head_stride: m_h,
                        seq_q_stride: m_q,
                        seq_k_stride: m_k,
                        logit_softcap_bits: cap_bits,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::AttentionMask { params: ap });
                    let u = emit_uniform(std::mem::size_of::<AttentionMaskParams>());
                    let bg = bind_arena_window(
                        &dev.device,
                        attention_mask_kernel(&dev.device),
                        &arena,
                        base,
                        size,
                        &u,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                    // …then the standard dense softmax, in place over the same
                    // matrix. Its last pass is elementwise, so `in_off == out_off`
                    // is safe; two dispatches replace a second copy of the
                    // reduction against the mask pass.
                    let sp = SoftmaxParams {
                        outer: batch * heads * seq_q,
                        inner: seq_k,
                        in_off: scores_off,
                        out_off: scores_off,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Softmax { params: sp });
                    let u = emit_uniform(std::mem::size_of::<SoftmaxParams>());
                    let bg = bind_arena_window(
                        &dev.device,
                        softmax_kernel(&dev.device),
                        &arena,
                        base,
                        size,
                        &u,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                    // 5. V′: only when V's tile is not already a dense `[S_k, D_v]`
                    //    (`v_dense` also covers the grouped-KV replication).
                    if !v_dense {
                        emit_transpose(
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mut meta_buffers,
                            &arena,
                            "rlx-wgpu attention route transpose meta",
                            base,
                            size,
                            &[batch, nkv, group, seq_k, hd_v],
                            &[v_b, v_h, 0, v_s, 1],
                            0,
                            v_off,
                            v_tmp_off,
                        );
                    }
                    // 6. out = P·V′, straight into the output slot when that slot's
                    //    own per-(b, h) tiles are dense, else into a scratch region.
                    emit_matmul(
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        seq_q,
                        seq_k,
                        hd_v,
                        scores_off,
                        if v_dense { v_off } else { v_tmp_off },
                        if out_dense { out_off } else { out_tmp_off },
                    );
                    // 7. …and when it is not: the same permute step returns out′
                    //    to the slot. Output axes `(b, s, h, d)` with dims
                    //    `(B, S_q, H, D_v)` and input strides
                    //    `(H·S_q·D_v, D_v, S_q·D_v, 1)` land the kernel's dense
                    //    write exactly on `out[b, s, h·D_v + d]` — the `[B, S, H·D_v]`
                    //    block's own dense memory — so no scatter kernel is needed.
                    if !out_dense {
                        let (h_sq_dv, sq_dv) = (heads * seq_q * hd_v, seq_q * hd_v);
                        emit_transpose(
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mut meta_buffers,
                            &arena,
                            "rlx-wgpu attention route transpose meta",
                            base,
                            size,
                            &[batch, seq_q, heads, hd_v],
                            &[h_sq_dv, hd_v, sq_dv, 1],
                            0,
                            out_tmp_off,
                            out_off,
                        );
                    }
                    continue;
                }
                if out_aliases_qkv && rlx_ir::env::flag("RLX_WGPU_DEBUG_ATTN_ALIAS") {
                    eprintln!(
                        "rlx-wgpu Attention alias: out={:?}@{}+{} q={:?}@{} k={:?}@{} v={:?}@{}",
                        node.id,
                        out_byte,
                        out_len,
                        q_id,
                        arena.offset(q_id),
                        k_id,
                        arena.offset(k_id),
                        v_id,
                        arena.offset(v_id),
                    );
                }
                let mut attn_scratch_copy: Option<(u64, u32)> = None;
                if out_aliases_qkv {
                    // In-place QKV/out: write to scratch then copy back.
                    // On sharded arenas continue past any Q/K/V already
                    // staged into this stripe's reserve — rewinding to
                    // `shard_stage_off` would overlap those copies and
                    // corrupt the attention inputs (seen at max_seq=272).
                    if arena.is_sharded() {
                        let stage = arena.shard_stage_off(base as usize) as u64;
                        if scratch < stage {
                            scratch = stage;
                        }
                    }
                    let tmp_byte = scratch;
                    let tmp_aligned = out_len.div_ceil(256) * 256;
                    scratch = scratch.saturating_add(tmp_aligned);
                    if arena.is_sharded() {
                        let stage_end = arena.shard_stage_off(base as usize) as u64
                            + crate::buffer::shard_stage_reserve() as u64;
                        assert!(
                            tmp_byte >= base
                                && scratch <= stage_end
                                && tmp_byte.saturating_add(out_len) <= base.saturating_add(size),
                            "rlx-wgpu Attention: alias scratch overflow \
                             tmp={tmp_byte} scratch={scratch} stage_end={stage_end} \
                             base={base} size={size}"
                        );
                    } else if param_anchor {
                        arena_ensure_scratch_for_window(&arena, &mut scratch, base, size);
                    } else {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    attn_scratch_copy = Some((tmp_byte, out_len as u32));
                }
                // The alias scratch path above can widen `base` after Q/K/V and
                // the output were recorded, so the offsets are read back out of
                // the window it left.
                (q_off, k_off, v_off, out_off, mask_off) =
                    rebase(base, size, (q_off, k_off, v_off, out_off, mask_off));
                // `out_off` was rebased into the window before the route
                // decision and nothing after it moved `base` unless the alias
                // path above did — which is exactly when the write goes through
                // its own scratch copy instead.
                let kernel_out_off = match attn_scratch_copy {
                    Some((tmp_byte, _)) => ((tmp_byte.saturating_sub(base)) / 4) as u32,
                    None => out_off,
                };
                let p = AttentionParams {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim: hd,
                    q_off,
                    k_off,
                    v_off,
                    out_off: kernel_out_off,
                    mask_off,
                    mask_kind: mask_kind_id,
                    scale_bits: scale.to_bits(),
                    window,
                    // Mask strides — derive from the mask's IR shape:
                    //   [B, S]:           (mb=S,        mh=0,    mq=0,   mk=1)
                    //   [B, S_q, S_k]:    (mb=S_q·S_k,  mh=0,    mq=S_k, mk=1)
                    //   [B, H, S_q, S_k]: (mb=H·S_q·S_k mh=S_q·S_k mq=S_k mk=1)
                    // Stride 0 means the kernel broadcasts across that
                    // axis (reads the same element for every value of
                    // the index). Lets us skip the Expand pre-pass that
                    // unfuse used to emit per attention block.
                    seq_q_stride: m_q,
                    seq_k_stride: m_k,
                    mask_batch_stride: m_b,
                    mask_head_stride: m_h,
                    kv_heads: nkv,
                    v_head_dim: hd_v,
                    logit_softcap_bits: cap_bits,
                    q_batch_stride: q_b,
                    q_head_stride: q_h,
                    q_seq_stride: q_s,
                    _pad_q: 0,
                    k_batch_stride: k_b,
                    k_head_stride: k_h,
                    k_seq_stride: k_s,
                    _pad_k: 0,
                    v_batch_stride: v_b,
                    v_head_stride: v_h,
                    v_seq_stride: v_s,
                    _pad_v: 0,
                    o_batch_stride: o_b,
                    o_head_stride: o_h,
                    o_seq_stride: o_s,
                    _pad_o: 0,
                };
                schedule.push(Step::Attention {
                    params: p,
                    mask_buf: None,
                });
                if let Some((tmp_byte, bytes)) = attn_scratch_copy {
                    schedule.push(Step::BufferCopy {
                        src_byte_off: tmp_byte,
                        dst_byte_off: out_byte,
                        bytes,
                    });
                }
                let ak = attention_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<AttentionParams>());
                let bg = bind_arena_window(&dev.device, ak, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => {
                use rlx_ir::op::AttentionBwdWrt;
                let q_id = node.inputs[0];
                let k_id = node.inputs[1];
                let v_id = node.inputs[2];
                let dy_id = node.inputs[3];
                let q_shape = graph.node(q_id).shape.dims();
                let k_shape = graph.node(k_id).shape.dims();
                let hd = *head_dim as u32;
                let (batch, heads, seq_q, seq_k, bhsd) = match q_shape.len() {
                    4 => {
                        // The forward arm's own reading of a rank-4 operand — one
                        // derivation, so the route cannot read a graph two ways.
                        let (batch, seq_q, seq_k, bhsd) = attention_route::rank4_geometry(
                            &graph,
                            q_id,
                            k_id,
                            *num_heads as u32,
                            hd,
                        )
                        .unwrap_or_else(|refusal| panic!("{refusal}"));
                        (batch, *num_heads as u32, seq_q, seq_k, bhsd)
                    }
                    3 => {
                        // The forward plan's own rank-3 reading — one derivation, so
                        // the route cannot read a rank-3 graph two ways either.
                        let (batch, seq_q, seq_k) = attention_route::rank3_geometry(
                            &graph,
                            q_id,
                            k_id,
                            *num_heads as u32,
                            hd,
                        )
                        .unwrap_or_else(|refusal| panic!("{refusal}"));
                        (batch, *num_heads as u32, seq_q, seq_k, false)
                    }
                    other => {
                        panic!("rlx-wgpu AttentionBackward: only rank-3/4 Q,K,V (got rank {other})")
                    }
                };
                let scale = 1.0_f32 / (hd as f32).sqrt();
                let (mask_kind_id, mask_off, mask_buf, window) = match mask_kind {
                    MaskKind::None => (0u32, 0u32, None, 0u32),
                    MaskKind::Causal => (1u32, 0u32, None, 0u32),
                    MaskKind::Custom => {
                        (2u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32)
                    }
                    MaskKind::Bias => (4u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32),
                    MaskKind::SlidingWindow(w) => (3u32, 0u32, None, *w as u32),
                };
                struct MStrides {
                    b: u32,
                    h: u32,
                    q: u32,
                    k: u32,
                }
                let mask_strides = if mask_kind_id == 2 || mask_kind_id == 4 {
                    let m_dims = graph.node(node.inputs[4]).shape.dims();
                    let dim = |i: usize| m_dims[i].unwrap_static() as u32;
                    match m_dims.len() {
                        2 => MStrides {
                            b: dim(1),
                            h: 0,
                            q: 0,
                            k: 1,
                        },
                        3 => MStrides {
                            b: dim(1) * dim(2),
                            h: 0,
                            q: dim(2),
                            k: 1,
                        },
                        4 => MStrides {
                            b: dim(1) * dim(2) * dim(3),
                            h: dim(2) * dim(3),
                            q: dim(3),
                            k: 1,
                        },
                        _ => MStrides {
                            b: heads * seq_q * seq_k,
                            h: seq_q * seq_k,
                            q: seq_k,
                            k: 1,
                        },
                    }
                } else {
                    MStrides {
                        b: heads * seq_q * seq_k,
                        h: seq_q * seq_k,
                        q: seq_k,
                        k: 1,
                    }
                };
                let stride = |shape: &[rlx_ir::shape::Dim], seq_extent: u32| {
                    rlx_ir::strides_for_shape(shape, heads, hd, seq_extent, bhsd)
                };
                let (q_b, q_h, q_s) = stride(q_shape, seq_q);
                let (k_b, k_h, k_s) = stride(k_shape, seq_k);
                let v_shape = graph.node(v_id).shape.dims();
                let (v_b, v_h, v_s) = stride(v_shape, seq_k);
                let out_shape = node.shape.dims();
                let out_seq = match wrt {
                    AttentionBwdWrt::Query => seq_q,
                    AttentionBwdWrt::Key | AttentionBwdWrt::Value => seq_k,
                };
                let (o_b, o_h, o_s) = stride(out_shape, out_seq);
                let wrt_id = match wrt {
                    AttentionBwdWrt::Query => 0u32,
                    AttentionBwdWrt::Key => 1u32,
                    AttentionBwdWrt::Value => 2u32,
                };
                let p = AttentionBwdParams {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim: hd,
                    q_off: (arena.offset(q_id) / 4) as u32,
                    k_off: (arena.offset(k_id) / 4) as u32,
                    v_off: (arena.offset(v_id) / 4) as u32,
                    dy_off: (arena.offset(dy_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    mask_off,
                    mask_kind: mask_kind_id,
                    scale_bits: scale.to_bits(),
                    window,
                    wrt: wrt_id,
                    seq_q_stride: mask_strides.q,
                    seq_k_stride: mask_strides.k,
                    mask_batch_stride: mask_strides.b,
                    mask_head_stride: mask_strides.h,
                    _pad_mask_0: 0,
                    _pad_mask_1: 0,
                    _pad_mask_2: 0,
                    q_batch_stride: q_b,
                    q_head_stride: q_h,
                    q_seq_stride: q_s,
                    _pad_q: 0,
                    k_batch_stride: k_b,
                    k_head_stride: k_h,
                    k_seq_stride: k_s,
                    _pad_k: 0,
                    v_batch_stride: v_b,
                    v_head_stride: v_h,
                    v_seq_stride: v_s,
                    _pad_v: 0,
                    o_batch_stride: o_b,
                    o_head_stride: o_h,
                    o_seq_stride: o_s,
                    _pad_o: 0,
                };
                schedule.push(Step::AttentionBackward {
                    params: p,
                    mask_buf,
                });
                let ak = attention_bwd_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<AttentionBwdParams>());
                let bg = bind_op_output_window(&dev.device, ak, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => {
                let x_id = node.inputs[0];
                let cos_id = node.inputs[1];
                let sin_id = node.inputs[2];
                let x_shape = graph.node(x_id).shape.dims();
                let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                if !last.is_multiple_of(*head_dim) {
                    panic!(
                        "rlx-wgpu Rope: last_dim ({last}) must be a multiple \
                            of head_dim ({head_dim})"
                    );
                }
                if head_dim % 2 != 0 {
                    panic!("rlx-wgpu Rope: head_dim must be even");
                }
                let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                // PLAN L1: derive batch from total / seq / last_dim
                // (= product of leading dims). `seq_stride` stays at
                // full seq for buffer offset math; `seq` becomes the
                // runtime-scaled loop bound.
                let batch = total / (seq * last as u32).max(1);
                let cos_is_param = tensor_is_graph_param(&graph, &param_offsets, cos_id);
                let cos_bytes = arena.len_of(cos_id) as u64;
                let rope_win: Vec<NodeId> = if cos_is_param && cos_bytes > ARENA_STAGE_CAP {
                    vec![cos_id, sin_id, node.id, x_id]
                } else {
                    vec![node.id, x_id, cos_id, sin_id]
                };
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &rope_win,
                );
                if !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let cos_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    cos_id,
                    &mut base,
                    &mut size,
                );
                let sin_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    sin_id,
                    &mut base,
                    &mut size,
                );
                let p = RopeParams {
                    n_total: total,
                    seq,
                    head_dim: *head_dim as u32,
                    half: (*head_dim / 2) as u32,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    last_dim: last as u32,
                    batch,
                    seq_stride: seq,
                    style: match style {
                        rlx_ir::op::RopeStyle::NeoX => 0,
                        rlx_ir::op::RopeStyle::GptJ => 1,
                    },
                    // Partial rotary: rotate only n_rot dims (Gemma 4 global
                    // layers use n_rot < head_dim). Equals half for full rope.
                    rot_half: (*n_rot / 2) as u32,
                    // The table's own row width, which is `head_dim / 2` for a
                    // table allocated at full width and only partly rotated.
                    cos_row_stride: rlx_ir::shape::rope_table_stride(
                        &graph.node(cos_id).shape,
                        *n_rot,
                    ) as u32,
                };
                schedule.push(Step::Rope { params: p });
                let rk = rope_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<RopeParams>());
                let bg = bind_arena_window(&dev.device, rk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Expand { target_shape } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let in_rank = in_shape.len();
                let rank = target_shape.len();
                if in_rank > rank {
                    panic!(
                        "rlx-wgpu Expand: rank mismatch \
                            (in_rank={in_rank}, target_rank={rank})"
                    );
                }
                // Implicit leading 1s when input rank < target rank (e.g.
                // scalar → vector from `LegalizeBroadcast`).
                let pad = rank.saturating_sub(in_rank);
                let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                let in_dims: Vec<u32> = (0..rank)
                    .map(|i| {
                        if i < pad {
                            1
                        } else {
                            in_shape[i - pad].unwrap_static() as u32
                        }
                    })
                    .collect();
                // Prefer GPU Expand with shard-local staging over ExpandHost on
                // Metal (unsharded): F5 DiT matched CPU at cos=1.0 that way,
                // while ExpandHost left a ~5% teacher-forced peak under-shoot.
                //
                // Discrete Vulkan/DX12 and virtually-sharded arenas: GPU Expand
                // collapses Kitten NSF (peak ~0.05). Host there — and whenever
                // `RLX_WGPU_EXPAND_HOST=1`.
                //
                // ...but COMPLEX NEVER HOSTS, here too. `ExpandHost` sits in
                // front of the complex-lanes handling below and is equally
                // lane-blind: it repeats one complex element and zeroes the
                // tail. Same defect as the `should_pack_host_op` routing above,
                // in the one op that has its own shortcut past it — so the guard
                // has to be repeated rather than inherited.
                let complex_expand =
                    node.shape.dtype().is_complex() || graph.node(in_id).shape.dtype().is_complex();
                let force_expand_host = !complex_expand
                    && (wgpu_prefer_structure_host(&arena)
                        || rlx_ir::env::flag("RLX_WGPU_EXPAND_HOST"));
                if force_expand_host {
                    schedule.push(Step::ExpandHost {
                        in_byte_off: arena.offset(in_id),
                        out_byte_off: arena.offset(node.id),
                        in_dims,
                        out_dims,
                        param_operand: crate::buffer::is_parameter_storage(&graph, in_id),
                    });
                    continue;
                }
                // Complex tensors pack `lanes` contiguous f32 per element (C64=2
                // [re,im], C128=4 df64). The expand kernel copies one f32 per
                // "element", so append an innermost lane axis (in==out, never a
                // broadcast) — each thread then copies a whole complex element's
                // lanes as a contiguous group instead of shattering the [re,im]
                // pairing. lanes=1 for real/int ⇒ strict no-op (axis is elided).
                let lanes: u32 = match node.shape.dtype() {
                    rlx_ir::DType::C64 => 2,
                    rlx_ir::DType::C128 => 4,
                    _ => 1,
                };
                let (rank, out_dims, in_dims) = if lanes > 1 {
                    let mut od = out_dims;
                    let mut idm = in_dims;
                    od.push(lanes);
                    idm.push(lanes);
                    (rank + 1, od, idm)
                } else {
                    (rank, out_dims, in_dims)
                };
                // Cumulative input strides (row-major). When the
                // input dim is 1 but target dim > 1, that axis
                // broadcasts → stride = 0.
                let mut in_strides_row = vec![1u32; rank];
                for i in (0..rank.saturating_sub(1)).rev() {
                    in_strides_row[i] = in_strides_row[i + 1] * in_dims[i + 1];
                }
                let strides_for_out: Vec<u32> = (0..rank)
                    .map(|i| {
                        if in_dims[i] == 1 && out_dims[i] != 1 {
                            0
                        } else {
                            in_strides_row[i]
                        }
                    })
                    .collect();
                let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                meta_data.extend_from_slice(&out_dims);
                meta_data.extend_from_slice(&strides_for_out);
                let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu expand meta"),
                    size: (meta_data.len() * 4).max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                dev.queue
                    .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                let meta_idx = meta_buffers.len();
                meta_buffers.push(meta_buf);

                // PLAN L1: bucket axis stays at out axis 0 iff the
                // expand at axis 0 isn't a broadcast (in_dims[0]
                // matches out_dims[0]). When broadcast at axis 0
                // (in_dims[0]==1, out_dims[0]>1), the bucket-axis
                // contract doesn't apply — fall back to full extent.
                let bucket_outermost = if in_dims[0] == out_dims[0] {
                    1u32
                } else {
                    0u32
                };
                let exp_ids = [node.id, in_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let exp_fits = arena_span_bytes(&arena, &exp_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &exp_ids,
                );
                if !exp_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                // Staging `in` across shards can widen `base` to the whole
                // stripe; recompute locals against the final window.
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_off
                };
                let p = ExpandParams {
                    rank: rank as u32,
                    out_total: elems * lanes,
                    in_off,
                    out_off,
                    bucket_outermost,
                    out_dim_0: out_dims[0],
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::Expand {
                    params: p,
                    meta_idx,
                });
                // Do NOT mark Expand as run-once: planner lifetimes often
                // reuse Expand outputs as activation scratch. Skipping on
                // later NFE steps left stale/corrupt biases → inf by step 4.
                let ek = expand_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ExpandParams>());
                let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rlx-wgpu expand bg"),
                    layout: &ek.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: arena_bind_buf(&arena, base).0,
                                offset: arena_bind_buf(&arena, base).1,
                                size: {
                                    let (b, l) = arena_bind_buf(&arena, base);
                                    aligned_bind_size(size, l, b.size())
                                },
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: u.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: meta_buffers[meta_idx].as_entire_binding(),
                        },
                    ],
                });
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Gather { axis } => {
                let table_id = node.inputs[0];
                let idx_id = node.inputs[1];
                let table_is_param = tensor_is_graph_param(&graph, &param_offsets, table_id);
                let table_bytes = arena.len_of(table_id) as u64;
                // Split-binding path: on >4 GiB arenas the embedding output and
                // its index can lie more than one ≤4 GiB binding window away
                // from the (multi-GiB) table. A single arena binding then can't
                // cover both — the kernel writes the output OUTSIDE the bound
                // window (the write is dropped/clamped) and idx-staging would
                // even overwrite part of the table. Route axis-0 embedding
                // gathers through a host segment with separate table / idx
                // windows and a dedicated output buffer (copied back).
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                // `RLX_WGPU_GATHER_SPLIT` forces the >4 GiB split-binding path on a
                // small table so it's testable (mirrors `RLX_WGPU_CONCAT_HOST`).
                // Also force on virtually-sharded arenas (NVIDIA 2 GiB bind).
                let gather_needs_split = *axis == 0
                    && (arena.is_sharded()
                        || rlx_ir::env::flag("RLX_WGPU_GATHER_SPLIT")
                        || (arena_whole_arena_bind(&arena, max_binding).is_none()
                            && arena_span_bytes(&arena, &[table_id, idx_id, node.id])
                                > max_binding));
                if gather_needs_split {
                    let table_shape = graph.node(table_id).shape.dims();
                    let idx_shape = graph.node(idx_id).shape.dims();
                    let vocab = table_shape[0].unwrap_static() as u32;
                    let dim: u32 = table_shape[1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let n_idx: u32 = idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let table_w = arena.len_of(table_id) as u64;
                    assert!(
                        table_w <= max_binding,
                        "rlx-wgpu gather_split: embedding table {table_w} bytes exceeds \
                         max_storage_buffer_binding_size {max_binding}"
                    );
                    // Complex: each embedding element is `lanes` contiguous f32
                    // (C64=2/C128=4), so the per-row copy width `dim` and the
                    // output total scale by lanes; `vocab`/`n_idx`/indices stay
                    // element-indexed (1 f32 lane per I64 index).
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    schedule.push(Step::GatherSplit {
                        n_out: elems * lanes,
                        n_idx,
                        dim: dim * lanes,
                        vocab,
                        table_byte_off: arena.offset(table_id) as u64,
                        idx_byte_off: arena.offset(idx_id) as u64,
                        out_byte_off: arena.offset(node.id) as u64,
                    });
                    // Host-classified step: it builds its own bind group at exec
                    // time, so do NOT push a uniform/bind group lane (those are
                    // indexed by GPU-step count).
                    continue;
                }
                // Complex packs `lanes` contiguous f32 per element (C64=2,
                // C128=4). Gather copies one f32 per "element"; each gathered
                // element is `lanes` contiguous f32, so scale the inner per-row
                // copy width (`dim`/`trailing`) and the output total by lanes.
                // Indices select ELEMENTS (leading axis / `vocab`), so they stay
                // in element units. lanes=1 (real/int) ⇒ strict no-op.
                let lanes: u32 = match node.shape.dtype() {
                    rlx_ir::DType::C64 => 2,
                    rlx_ir::DType::C128 => 4,
                    _ => 1,
                };
                let gather_win: Vec<NodeId> = if table_is_param && table_bytes > ARENA_STAGE_CAP {
                    vec![table_id, node.id, idx_id]
                } else {
                    vec![node.id, idx_id, table_id]
                };
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, table_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &gather_win,
                );
                if !table_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = if table_anchor && arena_tensor_in_window(&arena, table_id, base, size)
                {
                    arena_local_off_f32(&arena, table_id, base)
                } else {
                    arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        table_id,
                        &mut base,
                        &mut size,
                    )
                };
                let idx_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    idx_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                if *axis == 0 {
                    let table_shape = graph.node(table_id).shape.dims();
                    let idx_shape = graph.node(idx_id).shape.dims();
                    let vocab = table_shape[0].unwrap_static() as u32;
                    let dim: u32 = table_shape[1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let n_idx: u32 = idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let p = GatherParams {
                        n_out: elems * lanes,
                        n_idx,
                        dim: dim * lanes,
                        vocab,
                        in_off,
                        idx_off,
                        out_off,
                        _p0: 0,
                    };
                    schedule.push(Step::Gather { params: p });
                    let gk = gather_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GatherParams>());
                    let bg = bind_arena_window(&dev.device, gk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                } else {
                    let table_shape = graph.node(table_id).shape.dims();
                    let idx_shape = graph.node(idx_id).shape.dims();
                    let outer: u32 = table_shape[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let trailing: u32 = table_shape[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_dim = table_shape[*axis].unwrap_static() as u32;
                    let num_idx: u32 = idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let total = outer * num_idx * trailing;
                    let p = GatherAxisParams {
                        total: total * lanes,
                        outer,
                        axis_dim,
                        num_idx,
                        trailing: trailing * lanes,
                        table_off: in_off,
                        idx_off,
                        out_off,
                    };
                    schedule.push(Step::GatherAxis { params: p });
                    let gk = gather_axis_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GatherAxisParams>());
                    let bg = bind_arena_window(&dev.device, gk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }

            Op::FusedMatMulBiasAct { activation } => {
                // Inputs: [x, w, bias]. We require 2D × 2D or
                // [..,M,K] × [K,N] (broadcast bias). Bias is shape [N].
                let a_id = node.inputs[0];
                let b_id = node.inputs[1];
                let bias_id = node.inputs[2];
                let a_shape = graph.node(a_id).shape.dims();
                let b_shape = graph.node(b_id).shape.dims();
                let out_shape = node.shape.dims();
                let (m, k, n) = if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2
                {
                    (
                        a_shape[0].unwrap_static() as u32,
                        a_shape[1].unwrap_static() as u32,
                        b_shape[1].unwrap_static() as u32,
                    )
                } else if a_shape.len() >= 2
                    && b_shape.len() == 2
                    && out_shape.len() == a_shape.len()
                {
                    let leading: usize = a_shape[..a_shape.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                    let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                    let n_inner = b_shape[1].unwrap_static();
                    ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
                } else {
                    panic!(
                        "rlx-wgpu FusedMatMulBiasAct: unsupported shapes \
                            a={a_shape:?} b={b_shape:?}"
                    );
                };
                let act_id = match activation {
                    None => 0xFFFFu32,
                    Some(a) => activation_op_id(*a),
                };
                let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                let b_bytes = arena.len_of(b_id) as u64;
                let mut compute_precision = derive_matmul_compute(
                    &dev.device,
                    &graph,
                    &coop_f16_vk_mirror_acts,
                    a_id,
                    b_id,
                    m,
                    k,
                    n,
                );
                if b_is_param
                    && b_bytes > ARENA_STAGE_CAP
                    && arena.param_fits_f16_mirror(b_id)
                    && !rlx_ir::env::flag("RLX_WGPU_NO_F16_MIRROR")
                    // A genuine BF16 weight is stored f32 in the arena and must
                    // stay bit-exact — reading it from the f16 shadow would drop
                    // bf16's exponent range. Keep it on the F32 path.
                    && !matches!(graph.node(b_id).shape.dtype(), rlx_ir::DType::BF16)
                {
                    compute_precision = MatmulCompute::F16;
                }

                // Split-QKV pattern: matmul writes Q/K/V directly into
                // 3 separate output buffers, eliminating the 3 Narrow
                // dispatches that would otherwise follow.
                let mqk_eligible = act_id == 0xFFFFu32
                    && matches!(
                        compute_precision,
                        MatmulCompute::F32 | MatmulCompute::CoopF32 | MatmulCompute::CoopF16Vk
                    );
                if mqk_eligible && let Some(&(q_id, k_id_n, v_id)) = qkv_split.get(&node.id) {
                    let head_width = n / 3;
                    let qkv_kind = match compute_precision {
                        MatmulCompute::CoopF16Vk => MatmulQkvKind::CoopF16Vk,
                        MatmulCompute::CoopF32 => MatmulQkvKind::CoopF32,
                        _ => MatmulQkvKind::F32,
                    };
                    let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param);
                    let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        q_id,
                        a_id,
                        b_id,
                        b_in_arena,
                    );
                    let mut scratch = arena.scratch_off as u64;
                    if param_anchor {
                        arena_ensure_scratch_for_window(&arena, &mut scratch, base, size);
                    }
                    if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                        // Staging via arena_off_in_bind_window — no in-window assert
                        // (VirtIO/DX12 ~2 GiB caps leave large F32 params outside).
                        let _ = arena.offset(b_id);
                    }
                    let a_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let q_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        q_id,
                        &mut base,
                        &mut size,
                    );
                    let k_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        k_id_n,
                        &mut base,
                        &mut size,
                    );
                    let v_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        v_id,
                        &mut base,
                        &mut size,
                    );
                    let bias_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        bias_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off_f32 = if !b_in_arena {
                        (arena.offset(b_id) / 4) as u32
                    } else if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena_tensor_in_window(&arena, b_id, base, size)
                    {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            b_id,
                            &mut base,
                            &mut size,
                        )
                    };
                    let b_off_global = (arena.offset(b_id) / 4) as u32;
                    maybe_push_coop_f16_vk_casts(
                        &graph,
                        a_id,
                        b_id,
                        &coop_f16_vk_mirror_acts,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        &mm_cast,
                        compute_precision,
                        a_off,
                        m,
                        k,
                        1,
                        if qkv_kind == MatmulQkvKind::CoopF16Vk {
                            b_off_global
                        } else {
                            b_off_f32
                        },
                        n,
                    );
                    let p = MatmulQkvParams {
                        m,
                        k,
                        n,
                        a_off,
                        b_off: if qkv_kind == MatmulQkvKind::CoopF16Vk {
                            b_off_global
                        } else {
                            b_off_f32
                        },
                        q_off,
                        k_off,
                        v_off,
                        head_width,
                        has_bias: 1,
                        bias_off,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                        _p4: 0,
                    };
                    schedule.push(Step::MatmulQkv {
                        params: p,
                        kind: qkv_kind,
                    });
                    register_coop_f16_vk_b_param(
                        &mut coop_f16_b_param,
                        &param_offsets,
                        b_id,
                        p.b_off,
                        match qkv_kind {
                            MatmulQkvKind::CoopF16Vk => MatmulCompute::CoopF16Vk,
                            MatmulQkvKind::CoopF32 => MatmulCompute::CoopF32,
                            MatmulQkvKind::F32 => MatmulCompute::F32,
                        },
                    );
                    let u = emit_uniform(std::mem::size_of::<MatmulQkvParams>());
                    let bg = match qkv_kind {
                        MatmulQkvKind::CoopF16Vk => {
                            let mqk = matmul_qkv_coop_f16_vk_kernel(&dev.device).expect(
                                "coop f16 matmul_qkv kernel: feature was checked but missing",
                            );
                            let (bg, b_off_adj) = build_matmul_qkv_coop_f16_vk_bind_group(
                                &dev.device,
                                mqk,
                                &arena,
                                base,
                                size,
                                &u,
                                k,
                                n,
                                p.b_off,
                            );
                            if let Some(Step::MatmulQkv { params, .. }) = schedule.last_mut() {
                                params.b_off = b_off_adj;
                            }
                            bg
                        }
                        MatmulQkvKind::CoopF32 => bind_arena_window(
                            &dev.device,
                            matmul_qkv_coop_f32_kernel(&dev.device).expect(
                                "coop matmul_qkv kernel: hardware feature was checked but kernel missing",
                            ),
                            &arena,
                            base,
                            size,
                            &u,
                        ),
                        MatmulQkvKind::F32 => bind_arena_window(
                            &dev.device,
                            matmul_qkv_kernel(&dev.device),
                            &arena,
                            base,
                            size,
                            &u,
                        ),
                    };
                    uniforms.push(u);
                    bind_groups.push(bg);
                    if qkv_kind == MatmulQkvKind::CoopF16Vk {
                        coop_f16_vk_wide_bind_groups.insert(
                            schedule.len() - 1,
                            bind_arena_window(
                                &dev.device,
                                matmul_qkv_kernel(&dev.device),
                                &arena,
                                base,
                                size,
                                &uniforms[uniforms.len() - 1],
                            ),
                        );
                    }
                } else {
                    let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param);
                    let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        node.id,
                        a_id,
                        b_id,
                        b_in_arena,
                    );
                    let mut scratch = arena.scratch_off as u64;
                    if param_anchor {
                        arena_ensure_scratch_for_window(&arena, &mut scratch, base, size);
                    }
                    if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                        // Staging via arena_off_in_bind_window — no in-window assert
                        // (VirtIO/DX12 ~2 GiB caps leave large F32 params outside).
                        let _ = arena.offset(b_id);
                    }
                    let a_off_f32 = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off_f32 = if !b_in_arena {
                        (arena.offset(b_id) / 4) as u32
                    } else if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena_tensor_in_window(&arena, b_id, base, size)
                    {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            b_id,
                            &mut base,
                            &mut size,
                        )
                    };
                    let bias_off_f32 = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        bias_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off_global = (arena.offset(b_id) / 4) as u32;
                    let b_off_bind = if b_is_param
                        && matches!(
                            compute_precision,
                            MatmulCompute::Coop16 | MatmulCompute::CoopF16Vk | MatmulCompute::F16
                        ) {
                        b_off_global
                    } else {
                        b_off_f32
                    };
                    maybe_push_coop_f16_vk_casts(
                        &graph,
                        a_id,
                        b_id,
                        &coop_f16_vk_mirror_acts,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        &mm_cast,
                        compute_precision,
                        a_off_f32,
                        m,
                        k,
                        1,
                        b_off_bind,
                        n,
                    );
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch: 1,
                        a_batch_stride: 0,
                        b_batch_stride: 0,
                        c_batch_stride: 0,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32: arena_local_off_f32(&arena, node.id, base),
                        has_bias: 1,
                        bias_off_f32,
                        act_id,
                        b_is_param,
                        from_conv1x1: false,
                        compute_precision,
                    });
                    register_coop_f16_vk_b_param(
                        &mut coop_f16_b_param,
                        &param_offsets,
                        b_id,
                        b_off_bind,
                        compute_precision,
                    );
                    let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                    let (bg, b_off_adj) = build_matmul_bind_group(
                        &dev.device,
                        mm_k,
                        mm_w,
                        &mm_f16w,
                        mm_bf16w,
                        &mm_f16c,
                        &mm_coop,
                        &mm_coop_f32,
                        &arena,
                        base,
                        size,
                        &u,
                        b_is_param,
                        compute_precision,
                        k,
                        n,
                        1,
                        b_off_bind,
                        0,
                    );
                    if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                        *b_off_f32 = b_off_adj;
                    }
                    uniforms.push(u);
                    bind_groups.push(bg);
                    if compute_precision == MatmulCompute::CoopF16Vk {
                        coop_f16_vk_wide_bind_groups.insert(
                            schedule.len() - 1,
                            bind_arena_window(
                                &dev.device,
                                mm_w_active_compile,
                                &arena,
                                base,
                                size,
                                &uniforms[uniforms.len() - 1],
                            ),
                        );
                    }
                }
            }

            Op::DotGeneral { .. } => {
                // Should be unreachable: DotGeneral is decomposed into
                // MatMul + Transpose + Reshape by the unfusion pass
                // before memory planning. If we hit this arm, the
                // unfusion pass has a gap.
                panic!(
                    "rlx-wgpu DotGeneral: leaked past unfusion pass — \
                        check unfuse.rs::expand_dot_general for missing patterns"
                );
            }

            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                // Greedy: the project's rule is `temperature <= 0` or `top_k == 1`.
                // Those lower to a deterministic argmax (ties → lowest index);
                // temperature == 1.0 is a real draw, NOT greedy.
                let is_greedy = *temperature <= 0.0 || *top_k == 1;
                // Reached only when the whole arena binds at base zero: this arm
                // indexes the arena with absolute offsets (logits, scratch, output)
                // but binds only its output window, so `should_host_at_ceiling` has
                // already routed the op to the host route otherwise.
                if is_greedy {
                    let p = ArgmaxParams {
                        outer,
                        inner,
                        in_off: arena_local_off_f32(&arena, in_id, 0),
                        out_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Argmax { params: p });
                    let amk = argmax_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ArgmaxParams>());
                    let bg = bind_op_output_window(&dev.device, amk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                } else {
                    let p = SampleParams {
                        outer,
                        inner,
                        in_off: arena_local_off_f32(&arena, in_id, 0),
                        out_off: (arena.offset(node.id) / 4) as u32,
                        top_k: *top_k as u32,
                        top_p_bits: top_p.to_bits(),
                        temp_bits: temperature.to_bits(),
                        seed_lo: *seed as u32,
                        seed_hi: (*seed >> 32) as u32,
                        // Per-row working area in the arena tail; sized by `compute_scratch_bytes`.
                        scratch_off: (arena.scratch_off / 4) as u32,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Sample { params: p });
                    let sk = sample_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SampleParams>());
                    let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }

            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                let in_shape = graph.node(node.inputs[0]).shape.dims();
                let out_shape = node.shape.dims();
                let op_id: u32 = match kind {
                    ReduceOp::Sum => 0,
                    ReduceOp::Mean => 1,
                    ReduceOp::Max => 2,
                    ReduceOp::Min => 3,
                    ReduceOp::Prod => 4,
                };
                match (kernel_size.len(), in_shape.len(), out_shape.len()) {
                    (1, 3, 3) => {
                        let p = Pool1dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c: in_shape[1].unwrap_static() as u32,
                            l: in_shape[2].unwrap_static() as u32,
                            l_out: out_shape[2].unwrap_static() as u32,
                            kl: kernel_size[0] as u32,
                            sl: stride.first().copied().unwrap_or(1) as u32,
                            pl: padding.first().copied().unwrap_or(0) as u32,
                            op: op_id,
                            in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                            _p5: 0,
                        };
                        schedule.push(Step::Pool1d { params: p });
                        let pk = pool1d_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<Pool1dParams>());
                        let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                    (2, 4, 4) => {
                        let p = Pool2dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c: in_shape[1].unwrap_static() as u32,
                            h: in_shape[2].unwrap_static() as u32,
                            w: in_shape[3].unwrap_static() as u32,
                            h_out: out_shape[2].unwrap_static() as u32,
                            w_out: out_shape[3].unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            op: op_id,
                            in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::Pool2d { params: p });
                        let pk = pool2d_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<Pool2dParams>());
                        let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                    (3, 5, 5) => {
                        let p = Pool3dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c: in_shape[1].unwrap_static() as u32,
                            d: in_shape[2].unwrap_static() as u32,
                            h: in_shape[3].unwrap_static() as u32,
                            w: in_shape[4].unwrap_static() as u32,
                            d_out: out_shape[2].unwrap_static() as u32,
                            h_out: out_shape[3].unwrap_static() as u32,
                            w_out: out_shape[4].unwrap_static() as u32,
                            kd: kernel_size[0] as u32,
                            kh: kernel_size[1] as u32,
                            kw: kernel_size[2] as u32,
                            sd: stride.first().copied().unwrap_or(1) as u32,
                            sh: stride.get(1).copied().unwrap_or(1) as u32,
                            sw: stride.get(2).copied().unwrap_or(1) as u32,
                            pd: padding.first().copied().unwrap_or(0) as u32,
                            ph: padding.get(1).copied().unwrap_or(0) as u32,
                            pw: padding.get(2).copied().unwrap_or(0) as u32,
                            op: op_id,
                            in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                        };
                        schedule.push(Step::Pool3d { params: p });
                        let pk = pool3d_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<Pool3dParams>());
                        let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                    (k, n, m) => panic!(
                        "rlx-wgpu Pool: kernel-rank {k} with input rank {n} / \
                         output rank {m} not supported (use 1D/2D/3D NCHW)"
                    ),
                }
            }

            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let win_ids = [node.id, in_id, w_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                arena_expand_bind_window(&arena, &win_ids, &mut base, &mut size, max_binding);
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let w_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    w_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let in_shape = graph.node(in_id).shape.dims();
                let w_shape = graph.node(w_id).shape.dims();
                let out_shape = node.shape.dims();
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as u32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as u32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as u32;
                match (
                    kernel_size.len(),
                    in_shape.len(),
                    w_shape.len(),
                    out_shape.len(),
                ) {
                    (1, 3, 3, 3) => {
                        let p1 = Conv1dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c_in: in_shape[1].unwrap_static() as u32,
                            c_out: out_shape[1].unwrap_static() as u32,
                            l: in_shape[2].unwrap_static() as u32,
                            l_out: out_shape[2].unwrap_static() as u32,
                            kl: kernel_size[0] as u32,
                            sl: s(0),
                            pl: p(0),
                            dl: d(0),
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::Conv1d { params: p1 });
                        let ck = conv1d_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<Conv1dParams>());
                        let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                    (2, 4, 4, 4) => {
                        let h_in = in_shape[2].unwrap_static() as u32;
                        let w_in = in_shape[3].unwrap_static() as u32;
                        // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis
                        // and the length in W (`[N,C,1,L]`, kernel `[k,1]`). The 2D
                        // conv kernel would run the k-tap kernel over the singleton
                        // H axis. `[N,C,1,L]` and `[N,C,L,1]` share row-major layout,
                        // so relabel the length onto H (no data copy) — matching the
                        // CPU/MLX 1D paths and onnxruntime.
                        let one_d = h_in == 1
                            && w_in > 1
                            && kernel_size[0] > 1
                            && kernel_size.get(1).copied().unwrap_or(1) == 1;
                        // The mirrored lift: the taps-in-slot-1 shape a decoder body
                        // presents. The relabel is value-preserving only for a genuine
                        // unit-height lift — any height extent or height padding means
                        // H is a real axis, which the relabel would move and silently
                        // resize the output — and `conv1d_tiled`'s own `w_out == 1`
                        // criterion stays as it is: the relabel is what makes it hold.
                        let mirror_eligible = h_in == 1
                            && w_in > 1
                            && out_shape[2].unwrap_static() == 1
                            && kernel_size[0] == 1
                            && kernel_size.get(1).copied().unwrap_or(1) > 1
                            && p(0) == 0
                            && *groups == 1
                            && in_shape[0].unwrap_static() == 1;
                        let (h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw) = if one_d {
                            (
                                w_in,
                                1,
                                out_shape[3].unwrap_static() as u32,
                                1,
                                kernel_size[0] as u32,
                                1,
                                s(0),
                                1,
                                p(0),
                                0,
                                d(0),
                                1,
                            )
                        } else {
                            (
                                h_in,
                                w_in,
                                out_shape[2].unwrap_static() as u32,
                                out_shape[3].unwrap_static() as u32,
                                kernel_size[0] as u32,
                                kernel_size[1] as u32,
                                s(0),
                                s(1),
                                p(0),
                                p(1),
                                d(0),
                                d(1),
                            )
                        };
                        // Every route receives these unless the mirrored relabel
                        // below rewrites them in place; the host branch returns first.
                        let mut p2 = Conv2dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c_in: in_shape[1].unwrap_static() as u32,
                            c_out: out_shape[1].unwrap_static() as u32,
                            h,
                            w,
                            h_out,
                            w_out,
                            kh,
                            kw,
                            sh,
                            sw,
                            ph,
                            pw,
                            dh,
                            dw,
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                        };
                        // Host NCHW conv on discrete Vulkan/DX12 (NVIDIA SPIR-V
                        // conv mush even unsharded) and on virtually-sharded
                        // arenas. Opt-in anywhere with `RLX_WGPU_CONV_HOST=1`.
                        if wgpu_prefer_conv_host(&arena) {
                            schedule.push(Step::Conv2dHost {
                                n: p2.n,
                                c_in: p2.c_in,
                                c_out: p2.c_out,
                                h: p2.h,
                                w: p2.w,
                                h_out: p2.h_out,
                                w_out: p2.w_out,
                                kh: p2.kh,
                                kw: p2.kw,
                                sh: p2.sh,
                                sw: p2.sw,
                                ph: p2.ph,
                                pw: p2.pw,
                                dh: p2.dh,
                                dw: p2.dw,
                                groups: p2.groups,
                                in_byte_off: arena.offset(in_id) as u64,
                                w_byte_off: arena.offset(w_id) as u64,
                                out_byte_off: arena.offset(node.id) as u64,
                            });
                        } else {
                            // Three accelerated conv paths, all under the one
                            // switch (`RLX_WGPU_CONV_IM2COL`, on by default):
                            //   • im2col + GEMM: a GEMM-friendly shape (the floors
                            //     below) materialized into the reserved scratch;
                            //     needs the whole arena bound.
                            //   • 1×1 pointwise GEMM: a `Step::Matmul` straight off
                            //     the activation, no gather needed.
                            //   • conv1d_tiled (the `one_d` shape's default,
                            //     independent of the switch): register-blocked
                            //     direct conv, drop-in for the direct one. The
                            //     `mirror` relabel below is what lets it serve a
                            //     unit-height lift whose taps sit in slot 1.
                            // Everything else falls back to the direct conv.
                            let accel = crate::backend::conv_routes_enabled();
                            // A weight parked in the separate weight buffer
                            // (`WEIGHT_BUF_TAG`) is not addressable by the two GEMM
                            // routes: the raw word offset they address it with drops
                            // the tag, and the `col` panel occupies the scratch a
                            // staged copy would need. It keeps the previous
                            // tiled/direct route, exactly what ran before.
                            let parked = crate::buffer::is_weight_off(arena.offset(w_id));
                            let spatial = (p2.h_out as u64) * (p2.w_out as u64);
                            let k_total = (p2.c_in as u64) * (p2.kh as u64) * (p2.kw as u64);
                            let col_bytes = k_total.saturating_mul(spatial).saturating_mul(4);
                            let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                            let whole = arena_whole_arena_bind(&arena, max_binding);
                            // What a conv has to be for the route, all of it decided
                            // without reading the adapter: one group and a unit batch, real
                            // spatial taps, the shape floors `conv_im2col_col_elems` mirrors,
                            // the hard col cap `conv_im2col_scratch_bytes` applies, and the
                            // route switch — so the col reservation and the lowering agree on
                            // which convs the route could take, on this adapter or any other.
                            let im2col_candidate = accel
                                && p2.groups == 1
                                && p2.n == 1
                                && (p2.kh as u64) * (p2.kw as u64) >= 2
                                && spatial >= im2col_min_spatial()
                                && k_total >= im2col_min_k()
                                && (p2.c_out as u64) >= im2col_min_cout()
                                && col_bytes <= CONV_IM2COL_MAX_COL_BYTES;
                            // …and the three terms the shapes alone do not settle, which keep a
                            // candidate conv on the direct/tiled route: a parked weight the
                            // route's raw word offset cannot address, a col panel past the tail
                            // reserved for it (the planner drops a panel that would push the
                            // arena over the binding limit), and an arena — tail included —
                            // that does not bind in one storage binding. Counted here, where
                            // the decision is made, so none of them is silent and no term added
                            // to the candidate test above can leave the count behind.
                            let use_im2col = im2col_candidate
                                && !parked
                                && col_bytes <= arena.scratch_bytes as u64
                                && whole.is_some();
                            if im2col_candidate && !use_im2col {
                                im2col_fallbacks += 1;
                            }
                            // 1×1 conv: `W[c_out,c_in] @ in[c_in,h*w]`, both operands
                            // already in the layout a GEMM wants, so no gather — a
                            // routing change onto the matmul machinery, not a new
                            // kernel. Admitted at its own frozen bar
                            // (`pointwise_admitted` — a 1×1 kernel's col *is* the
                            // activation) and the shared channel floor, capped so the
                            // step's own dispatch grid stays inside the per-dimension
                            // workgroup limit. That bar is deliberately NOT the im2col
                            // route's floors: the spatial floor below has to move for
                            // the non-pointwise shapes, and this one must not. The
                            // three routes are mutually exclusive by shape — this one
                            // needs 1×1 taps, `mirror_eligible` a >1 tap in slot 1,
                            // `one_d` a >1 tap in slot 0 — so each predicate tests
                            // only what it decides.
                            let pointwise = accel
                                && !parked
                                && p2.kh == 1
                                && p2.kw == 1
                                && p2.sh == 1
                                && p2.sw == 1
                                && p2.ph == 0
                                && p2.pw == 0
                                && p2.dh == 1
                                && p2.dw == 1
                                && p2.groups == 1
                                && p2.n == 1
                                && (p2.c_out as u64) >= im2col_min_cout()
                                && pointwise_admitted(k_total, spatial)
                                && spatial <= POINTWISE_MAX_SPATIAL;
                            // The relabel exists only to unlock the tiled route for a
                            // unit-height lift the im2col route does not take: it
                            // leaves `spatial` and `k_total` alone, so no other
                            // route's criterion moves.
                            let mirror = accel
                                && !parked
                                && !use_im2col
                                && mirror_eligible
                                && spatial >= conv_tiled_min_spatial()
                                && !rlx_ir::env::flag("RLX_WGPU_NO_TILED_CONV");
                            if mirror {
                                // The relabel, in place: the length axis onto H and
                                // the taps/stride/padding/dilation from slot 1.
                                (p2.h, p2.w, p2.h_out, p2.w_out) =
                                    (w_in, 1, out_shape[3].unwrap_static() as u32, 1);
                                (p2.kh, p2.kw) = (kernel_size[1] as u32, 1);
                                (p2.sh, p2.sw) = (s(1), 1);
                                (p2.ph, p2.pw) = (p(1), 0);
                                (p2.dh, p2.dw) = (d(1), 1);
                            }
                            // `mirror` implies every criterion below (the relabel is
                            // what makes `kw == w == w_out == 1` hold), so only the
                            // `one_d` branch has to state them.
                            let use_tiled = mirror
                                || (!use_im2col
                                    && one_d
                                    && p2.kw == 1
                                    && p2.w == 1
                                    && p2.w_out == 1
                                    && p2.groups == 1
                                    && p2.n == 1
                                    && spatial >= conv_tiled_min_spatial()
                                    && !rlx_ir::env::flag("RLX_WGPU_NO_TILED_CONV"));
                            if pointwise {
                                // GEMM: weight[c_out, c_in] @ in[c_in, spatial]
                                //   → out[c_out, spatial]  (NCHW, N==1). No gather:
                                // the activation already sits in the layout a GEMM
                                // wants. The operands are addressed through the same
                                // `arena_off_in_bind_window` offsets the direct conv
                                // uses, not a raw `arena.offset(id)` cast. `m` is the
                                // output-channel axis here, so `from_conv1x1: true`
                                // keeps active-extent dispatch from scaling it.
                                let (compute_precision, mm) = conv_gemm(
                                    &dev.device,
                                    mm_k,
                                    mm_coop_f32,
                                    p2.c_out,
                                    p2.c_in,
                                    spatial as u32,
                                );
                                schedule.push(Step::Matmul {
                                    m: p2.c_out,
                                    k: p2.c_in,
                                    n: spatial as u32,
                                    batch: 1,
                                    a_batch_stride: p2.c_out.saturating_mul(p2.c_in),
                                    b_batch_stride: p2.c_in.saturating_mul(spatial as u32),
                                    c_batch_stride: p2.c_out.saturating_mul(spatial as u32),
                                    a_off_f32: w_off,
                                    b_off_f32: in_off,
                                    c_off_f32: out_off,
                                    has_bias: 0,
                                    bias_off_f32: 0,
                                    act_id: 0xFFFF,
                                    b_is_param: false,
                                    compute_precision,
                                    from_conv1x1: true,
                                });
                                let u_mm = emit_uniform(std::mem::size_of::<MatmulParams>());
                                let bg_mm =
                                    bind_arena_window(&dev.device, mm, &arena, base, size, &u_mm);
                                uniforms.push(u_mm);
                                bind_groups.push(bg_mm);
                            } else if use_im2col {
                                let (base_w, size_w) = whole.expect("whole-arena bind");
                                // col lives at the reserved scratch tail (free of
                                // live data); consumed immediately by the GEMM.
                                let col_word_off = (arena.scratch_off / 4) as u32;
                                let im_params = Im2Col2dParams {
                                    c_in: p2.c_in,
                                    h: p2.h,
                                    w: p2.w,
                                    h_out: p2.h_out,
                                    w_out: p2.w_out,
                                    kh: p2.kh,
                                    kw: p2.kw,
                                    sh: p2.sh,
                                    sw: p2.sw,
                                    ph: p2.ph,
                                    pw: p2.pw,
                                    dh: p2.dh,
                                    dw: p2.dw,
                                    in_off: (arena.offset(in_id) / 4) as u32,
                                    col_off: col_word_off,
                                    k_total: k_total as u32,
                                    spatial: spatial as u32,
                                    _p0: 0,
                                    _p1: 0,
                                    _p2: 0,
                                };
                                schedule.push(Step::Im2ColGpu { params: im_params });
                                let imk = im2col2d_kernel(&dev.device);
                                let u_im = emit_uniform(std::mem::size_of::<Im2Col2dParams>());
                                // Static params — write once at compile (the
                                // active-extent rewrite pass is a no-op here).
                                dev.queue
                                    .write_buffer(&u_im, 0, bytemuck::bytes_of(&im_params));
                                let bg_im = bind_arena_window(
                                    &dev.device,
                                    imk,
                                    &arena,
                                    base_w,
                                    size_w,
                                    &u_im,
                                );
                                uniforms.push(u_im);
                                bind_groups.push(bg_im);
                                // GEMM: weight[c_out, K] @ col[K, spatial]
                                //   → out[c_out, spatial]  (NCHW, N==1).
                                let m = p2.c_out;
                                let kk = k_total as u32;
                                let nn2 = spatial as u32;
                                let (compute_precision, mm) =
                                    conv_gemm(&dev.device, mm_k, mm_coop_f32, m, kk, nn2);
                                schedule.push(Step::Matmul {
                                    m,
                                    k: kk,
                                    n: nn2,
                                    batch: 1,
                                    a_batch_stride: m.saturating_mul(kk),
                                    b_batch_stride: kk.saturating_mul(nn2),
                                    c_batch_stride: m.saturating_mul(nn2),
                                    a_off_f32: (arena.offset(w_id) / 4) as u32,
                                    b_off_f32: col_word_off,
                                    c_off_f32: (arena.offset(node.id) / 4) as u32,
                                    has_bias: 0,
                                    bias_off_f32: 0,
                                    act_id: 0xFFFF,
                                    b_is_param: false,
                                    from_conv1x1: false,
                                    compute_precision,
                                });
                                let u_mm = emit_uniform(std::mem::size_of::<MatmulParams>());
                                let bg_mm = bind_arena_window(
                                    &dev.device,
                                    mm,
                                    &arena,
                                    base_w,
                                    size_w,
                                    &u_mm,
                                );
                                uniforms.push(u_mm);
                                bind_groups.push(bg_mm);
                            } else {
                                schedule.push(if use_tiled {
                                    Step::Conv2dTiled { params: p2 }
                                } else {
                                    Step::Conv2d { params: p2 }
                                });
                                let ck = if use_tiled {
                                    conv1d_tiled_kernel(&dev.device)
                                } else {
                                    conv2d_kernel(&dev.device)
                                };
                                let u = emit_uniform(std::mem::size_of::<Conv2dParams>());
                                let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                                uniforms.push(u);
                                bind_groups.push(bg);
                            }
                        }
                    }
                    (3, 5, 5, 5) => {
                        // A third input would be the per-output-channel bias,
                        // folded into the conv's store. `Op::Conv` is 2-input by
                        // contract, so this is defensive, not a live path — see
                        // the note on `bias_off` below.
                        let (bias_off, has_bias) = match node.inputs.get(2) {
                            Some(&b) => (
                                arena_off_in_bind_window(
                                    &graph,
                                    &param_offsets,
                                    &dev.device,
                                    &arena,
                                    &mut schedule,
                                    &mut scratch,
                                    b,
                                    &mut base,
                                    &mut size,
                                ),
                                1u32,
                            ),
                            None => (0u32, 0u32),
                        };
                        let p3 = Conv3dParams {
                            n: in_shape[0].unwrap_static() as u32,
                            c_in: in_shape[1].unwrap_static() as u32,
                            c_out: out_shape[1].unwrap_static() as u32,
                            d: in_shape[2].unwrap_static() as u32,
                            h: in_shape[3].unwrap_static() as u32,
                            w: in_shape[4].unwrap_static() as u32,
                            d_out: out_shape[2].unwrap_static() as u32,
                            h_out: out_shape[3].unwrap_static() as u32,
                            w_out: out_shape[4].unwrap_static() as u32,
                            kd: kernel_size[0] as u32,
                            kh: kernel_size[1] as u32,
                            kw: kernel_size[2] as u32,
                            sd: s(0),
                            sh: s(1),
                            sw: s(2),
                            pd: p(0),
                            ph: p(1),
                            pw: p(2),
                            dd: d(0),
                            dh: d(1),
                            dw: d(2),
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                            // These were hard-coded to 0 while `bias_off` /
                            // `has_bias` were computed just above and dropped,
                            // so the shader would have skipped its
                            // `acc + arena[bias_off + co]` store. Unreachable
                            // today — `Op::Conv` is 2-input by contract
                            // (`Op::num_inputs`, enforced by `verify`), so
                            // `inputs.get(2)` is always `None` and a 3-D conv
                            // with a bias arrives as `Op::Conv3d` via
                            // `fold_conv_bias_into_conv3d` instead. Kept
                            // correct rather than deleted so the branch cannot
                            // silently drop a bias if that contract changes.
                            bias_off,
                            has_bias,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::Conv3d { params: p3 });
                        let ck = conv3d_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                        let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                    (k, ni, wi, mi) => panic!(
                        "rlx-wgpu Conv: rank kernel={k} in={ni} weight={wi} out={mi} \
                         not supported (use 1D/2D/3D NCHW)"
                    ),
                }
            }

            Op::FusedConvBiasAct {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                activation,
                has_residual,
            } => {
                // Native 2D conv fused with the bias + optional-residual + act
                // epilogue: y = act(conv(x,w) + bias[c] + residual). Replaces the
                // CPU host round-trip; conv body matches conv2d.wgsl bit-for-bit.
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let bias_id = node.inputs[2];
                let res_id = if *has_residual {
                    Some(node.inputs[3])
                } else {
                    None
                };
                assert_eq!(
                    kernel_size.len(),
                    2,
                    "rlx-wgpu FusedConvBiasAct: only 2D NCHW convs are fused"
                );

                let mut win_ids = vec![node.id, in_id, w_id, bias_id];
                if let Some(r) = res_id {
                    win_ids.push(r);
                }
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                arena_expand_bind_window(&arena, &win_ids, &mut base, &mut size, max_binding);
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let off = |id: rlx_ir::NodeId,
                           base: &mut u64,
                           size: &mut u64,
                           schedule: &mut Vec<Step>,
                           scratch: &mut u64| {
                    arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        schedule,
                        scratch,
                        id,
                        base,
                        size,
                    )
                };
                let in_off = off(in_id, &mut base, &mut size, &mut schedule, &mut scratch);
                let w_off = off(w_id, &mut base, &mut size, &mut schedule, &mut scratch);
                let bias_off = off(bias_id, &mut base, &mut size, &mut schedule, &mut scratch);
                let residual_off = match res_id {
                    Some(r) => off(r, &mut base, &mut size, &mut schedule, &mut scratch),
                    None => 0,
                };
                let out_off = off(node.id, &mut base, &mut size, &mut schedule, &mut scratch);

                let in_shape = graph.node(in_id).shape.dims();
                let out_shape = node.shape.dims();
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as u32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as u32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as u32;
                let act_id = match activation {
                    Some(a) => activation_op_id(*a),
                    None => 0xFFFF,
                };
                let params = FusedConvBiasActParams {
                    n: in_shape[0].unwrap_static() as u32,
                    c_in: in_shape[1].unwrap_static() as u32,
                    c_out: out_shape[1].unwrap_static() as u32,
                    h: in_shape[2].unwrap_static() as u32,
                    w: in_shape[3].unwrap_static() as u32,
                    h_out: out_shape[2].unwrap_static() as u32,
                    w_out: out_shape[3].unwrap_static() as u32,
                    kh: kernel_size[0] as u32,
                    kw: kernel_size[1] as u32,
                    sh: s(0),
                    sw: s(1),
                    ph: p(0),
                    pw: p(1),
                    dh: d(0),
                    dw: d(1),
                    groups: *groups as u32,
                    in_off,
                    w_off,
                    out_off,
                    has_bias: 1,
                    bias_off,
                    act_id,
                    has_residual: *has_residual as u32,
                    residual_off,
                };
                schedule.push(Step::FusedConvBiasAct { params });
                let k = fused_conv_bias_act_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<FusedConvBiasActParams>());
                let bg = bind_arena_window(&dev.device, k, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::Im2Col {
                kernel_size,
                stride,
                padding,
                dilation,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                if kernel_size.len() != 2 || x_shape.rank() != 4 {
                    panic!("rlx-wgpu Im2Col: 2D NCHW only");
                }
                let n = match x_shape.dim(0) {
                    rlx_ir::shape::Dim::Static(v) => v as u32,
                    _ => 0,
                };
                let c_in = x_shape.dim(1).unwrap_static() as u32;
                let h = x_shape.dim(2).unwrap_static() as u32;
                let w = x_shape.dim(3).unwrap_static() as u32;
                let kh = kernel_size[0] as u32;
                let kw = kernel_size[1] as u32;
                let sh = stride.first().copied().unwrap_or(1) as u32;
                let sw = stride.get(1).copied().unwrap_or(1) as u32;
                let ph = padding.first().copied().unwrap_or(0) as u32;
                let pw = padding.get(1).copied().unwrap_or(0) as u32;
                let dh = dilation.first().copied().unwrap_or(1) as u32;
                let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
                let h_out = rlx_ir::shape::conv2d_spatial_output(
                    h as usize,
                    kh as usize,
                    sh as usize,
                    ph as usize,
                    dh as usize,
                ) as u32;
                let w_out = rlx_ir::shape::conv2d_spatial_output(
                    w as usize,
                    kw as usize,
                    sw as usize,
                    pw as usize,
                    dw_dil as usize,
                ) as u32;
                schedule.push(Step::Im2ColHost {
                    x_byte_off: arena.offset(node.inputs[0]) as u32,
                    col_byte_off: arena.offset(node.id) as u32,
                    n,
                    c_in,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                });
            }

            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let dy_shape = &graph.node(node.inputs[1]).shape;
                schedule.push(Step::Conv2dBackwardWeightHost {
                    x_byte_off: arena.offset(node.inputs[0]) as u32,
                    dy_byte_off: arena.offset(node.inputs[1]) as u32,
                    dw_byte_off: arena.offset(node.id) as u32,
                    n: x_shape.dim(0).unwrap_static() as u32,
                    c_in: x_shape.dim(1).unwrap_static() as u32,
                    h: x_shape.dim(2).unwrap_static() as u32,
                    w: x_shape.dim(3).unwrap_static() as u32,
                    c_out: dy_shape.dim(1).unwrap_static() as u32,
                    h_out: dy_shape.dim(2).unwrap_static() as u32,
                    w_out: dy_shape.dim(3).unwrap_static() as u32,
                    kh: kernel_size[0] as u32,
                    kw: kernel_size[1] as u32,
                    sh: stride.first().copied().unwrap_or(1) as u32,
                    sw: stride.get(1).copied().unwrap_or(1) as u32,
                    ph: padding.first().copied().unwrap_or(0) as u32,
                    pw: padding.get(1).copied().unwrap_or(0) as u32,
                    dh: dilation.first().copied().unwrap_or(1) as u32,
                    dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                    groups: (*groups).max(1) as u32,
                });
            }

            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // inputs [dy, w]; output dx [n, c_in, h, w_in].
                let dy_shape = &graph.node(node.inputs[0]).shape;
                let dx_shape = &node.shape;
                schedule.push(Step::Conv2dBackwardInputHost {
                    dy_byte_off: arena.offset(node.inputs[0]) as u32,
                    w_byte_off: arena.offset(node.inputs[1]) as u32,
                    dx_byte_off: arena.offset(node.id) as u32,
                    n: dx_shape.dim(0).unwrap_static() as u32,
                    c_in: dx_shape.dim(1).unwrap_static() as u32,
                    h: dx_shape.dim(2).unwrap_static() as u32,
                    w_in: dx_shape.dim(3).unwrap_static() as u32,
                    c_out: dy_shape.dim(1).unwrap_static() as u32,
                    h_out: dy_shape.dim(2).unwrap_static() as u32,
                    w_out: dy_shape.dim(3).unwrap_static() as u32,
                    kh: kernel_size[0] as u32,
                    kw: kernel_size[1] as u32,
                    sh: stride.first().copied().unwrap_or(1) as u32,
                    sw: stride.get(1).copied().unwrap_or(1) as u32,
                    ph: padding.first().copied().unwrap_or(0) as u32,
                    pw: padding.get(1).copied().unwrap_or(0) as u32,
                    dh: dilation.first().copied().unwrap_or(1) as u32,
                    dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                    groups: (*groups).max(1) as u32,
                });
            }

            Op::Interpolate3d { .. } => {
                schedule.push(host_op_step(&graph, node, &arena));
            }

            Op::Cumsum { axis, exclusive } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let last = (in_shape.len() - 1) as i32;
                if *axis != -1 && *axis != last {
                    // Non-last-axis Cumsum: packed HostOp (CPU thunk).
                    schedule.push(host_op_step(&graph, node, &arena));
                    continue;
                }
                let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let p = CumsumParams {
                    outer,
                    inner,
                    in_off: (arena.offset(in_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    exclusive: if *exclusive { 1 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Cumsum { params: p });
                let ck2 = cumsum_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<CumsumParams>());
                let bg = bind_op_output_window(&dev.device, ck2, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::CumProd { axis, exclusive } | Op::CumMax { axis, exclusive } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.dims();
                let last = (in_shape.len() - 1) as i32;
                if *axis != -1 && *axis != last {
                    // Non-last-axis: packed HostOp (CPU thunk).
                    schedule.push(host_op_step(&graph, node, &arena));
                    continue;
                }
                let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let p = CumScanParams {
                    outer,
                    inner,
                    in_off: (arena.offset(in_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    exclusive: if *exclusive { 1 } else { 0 },
                    is_max: matches!(node.op, Op::CumMax { .. }) as u32,
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::CumScan { params: p });
                let ck2 = cum_scan_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<CumScanParams>());
                let bg = bind_op_output_window(&dev.device, ck2, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Fft { inverse, norm } => {
                let in_id = node.inputs[0];
                let in_shape = graph.node(in_id).shape.clone();
                let meta = rlx_ir::fft::fft_meta(&in_shape);
                let dtype = in_shape.dtype();
                // The wgpu arena is f32-uniform (4 bytes/elem — see the Constant
                // upload path). f64/c64 tensors get half-sized slots and are
                // silently truncated to garbage; the FFT-host fallback can't
                // read them back either. Reject cleanly instead (like MLX/ANE,
                // which don't do f64 on the GPU). Use f32 for GPU FFT.
                assert!(
                    matches!(dtype, rlx_ir::DType::F32),
                    "rlx-wgpu Op::Fft: only f32 is supported on the wgpu backend \
                     (the arena is f32-uniform); got {dtype:?}. Run f64/c64 FFT on CPU."
                );
                // The GPU step binds the whole physical buffer and carries absolute
                // WORD offsets (`offset/4`), so it is correct only when the whole
                // arena binds at base zero — the same question the
                // per-binding-ceiling rule asks. Anything else keeps its own host
                // step below, which stages the span by byte range and stitches
                // across stripes. Decided first, because which step runs decides
                // what the arena size has to satisfy.
                let use_gpu = rlx_ir::fft::gpu_fft_native_eligible(dtype, meta.n_complex)
                    && meta.n_complex >= 2
                    && arena.binds_whole_at_base_zero(
                        dev.device.limits().max_storage_buffer_binding_size,
                    );
                // A parameter operand parked in the separate weight buffer cannot
                // be addressed here at all: `Step::FftGpu` and `Step::FftHost` both
                // carry offsets that drop the `WEIGHT_BUF_TAG`, so the transform
                // reads the activation arena instead. No in-tree graph feeds `Fft`
                // a parameter — this is structural — so refuse loudly rather than
                // leave a route that would read the wrong buffer silently.
                assert!(
                    !any_parked_operand(&arena, node),
                    "rlx-wgpu: REFUSED (parked FFT operand): Op::Fft reads {in_id:?} parked in the \
                     weight buffer ({}); both its device step and its host step carry offsets that \
                     drop the parked tag. Avoid parking this operand, or move the graph to a \
                     backend that can address it.",
                    parked_operand_report(&arena)
                );
                // `FftHost` carries absolute BYTE offsets in `u32`, so the host route
                // — the one this falls back to whenever the device step is not
                // eligible — cannot address an arena past that range. The device
                // step needs no such bound: its word offsets fit whenever it is
                // eligible at all (`binds_whole_at_base_zero` already requires
                // `size / 4` to fit a `u32`).
                assert!(
                    use_gpu || arena.size <= u32::MAX as usize,
                    "rlx-wgpu: REFUSED (FFT arena too large): Op::Fft's host step carries 32-bit \
                     byte offsets and this arena is {} bytes, so its offsets would truncate. \
                     Shrink the arena, or run the transform on a backend that can address it.",
                    arena.size
                );
                let scale = norm.output_scale(meta.n_complex, *inverse) as f32;
                if use_gpu {
                    schedule.push(Step::FftGpu {
                        src_off: (arena.offset(in_id) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        outer: meta.outer as u32,
                        n: meta.n_complex as u32,
                        inverse: if *inverse { 1 } else { 0 },
                        norm_scale: scale,
                    });
                    fft_gpu_steps.push(crate::fft_dispatch::FftGpuResources::new(
                        &dev.device,
                        &arena.buffer,
                    ));
                } else {
                    schedule.push(Step::FftHost {
                        src_byte_off: arena.offset(in_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_complex: meta.n_complex as u32,
                        inverse: *inverse,
                        norm_tag: norm.tag(),
                        dtype_tag: fft_dtype_tag(dtype),
                    });
                }
            }
            Op::FftQ {
                inverse,
                norm,
                scale,
            } => {
                // Host fallback. The arena is f32-*valued*, so the adapter
                // converts rather than reinterpreting; reading these slots as
                // raw i32 yields float bit patterns. The conversion is exact
                // to 2^24 and panics past it rather than dropping low bits.
                //
                // `Step::FftQHost` carries those byte offsets in `u32` and resolves
                // no weight-buffer tag, so it is the same class of geometry the
                // `Op::Fft` arm above refuses: a parameter parked in the weight
                // buffer (its tag dropped by the cast) or an arena past the 32-bit
                // byte range (every offset truncated) reads and writes the wrong
                // slots. There is no other route for `FftQ`, so refuse loudly.
                let in_id = node.inputs[0];
                assert!(
                    !any_parked_operand(&arena, node) && arena.size <= u32::MAX as usize,
                    "rlx-wgpu: REFUSED (FftQ geometry): Op::FftQ's only route carries 32-bit byte \
                     offsets and resolves no parked tag, so a parameter parked in the weight \
                     buffer or an arena over 2^32 bytes (arena {} B, parked={}) would read the \
                     wrong slots. Avoid parking this operand or shrink the arena.",
                    arena.size,
                    any_parked_operand(&arena, node)
                );
                let in_shape = graph.node(in_id).shape.clone();
                let meta = rlx_ir::fft::fft_meta(&in_shape);
                schedule.push(Step::FftQHost {
                    src_byte_off: arena.offset(in_id) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    outer: meta.outer as u32,
                    n_complex: meta.n_complex as u32,
                    inverse: *inverse,
                    norm_tag: norm.tag(),
                    scale_tag: scale.tag(),
                });
            }
            Op::WelchPeaks { k, n_segments } => {
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                    .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                let use_gpu =
                    rlx_ir::audio::welch_peaks_gpu_native_eligible(&spec_shape, *k, *n_segments)
                        .unwrap_or(false);
                if use_gpu {
                    let p = WelchPeaksGpuParams {
                        spec_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        welch_batch: meta.welch_batch as u32,
                        n_fft: meta.n_fft as u32,
                        n_segments: meta.n_segments as u32,
                        k: meta.k as u32,
                        n_bins: meta.n_bins as u32,
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::WelchPeaksGpu { params: p });
                    let wk = welch_peaks_gpu_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<WelchPeaksGpuParams>());
                    let bg = bind_op_output_window(&dev.device, wk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                } else {
                    schedule.push(Step::WelchPeaksHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        welch_batch: meta.welch_batch as u32,
                        n_fft: meta.n_fft as u32,
                        n_segments: meta.n_segments as u32,
                        k: meta.k as u32,
                    });
                }
            }
            Op::LogMel => {
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let filt_shape = graph.node(node.inputs[1]).shape.clone();
                let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                    .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                schedule.push(Step::LogMelHost {
                    spec_byte_off: arena.offset(node.inputs[0]) as u32,
                    filt_byte_off: arena.offset(node.inputs[1]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    outer: meta.outer as u32,
                    n_fft: meta.n_fft as u32,
                    n_bins: meta.n_bins as u32,
                    n_mels: meta.n_mels as u32,
                });
            }
            Op::LogMelBackward => {
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let filt_shape = graph.node(node.inputs[1]).shape.clone();
                let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                    .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                schedule.push(Step::LogMelBackwardHost {
                    spec_byte_off: arena.offset(node.inputs[0]) as u32,
                    filt_byte_off: arena.offset(node.inputs[1]) as u32,
                    dy_byte_off: arena.offset(node.inputs[2]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    outer: meta.outer as u32,
                    n_fft: meta.n_fft as u32,
                    n_bins: meta.n_bins as u32,
                    n_mels: meta.n_mels as u32,
                });
            }
            Op::SelectiveScan { state_size } => {
                if *state_size > 256 {
                    panic!(
                        "rlx-wgpu SelectiveScan: state_size {} exceeds compile-time \
                            cap of 256 (kernel uses fixed-size private array)",
                        state_size
                    );
                }
                let x_id = node.inputs[0];
                let dt_id = node.inputs[1];
                let a_id = node.inputs[2];
                let b_id = node.inputs[3];
                let c_id = node.inputs[4];
                let in_dims = graph.node(x_id).shape.dims();
                let seq = in_dims[1].unwrap_static() as u32;
                let p = SelectiveScanParams {
                    batch: in_dims[0].unwrap_static() as u32,
                    seq,
                    hidden: in_dims[2].unwrap_static() as u32,
                    state_size: *state_size as u32,
                    x_off: (arena.offset(x_id) / 4) as u32,
                    delta_off: (arena.offset(dt_id) / 4) as u32,
                    a_off: (arena.offset(a_id) / 4) as u32,
                    b_off: (arena.offset(b_id) / 4) as u32,
                    c_off: (arena.offset(c_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    // PLAN L1: full-extent stride; safe under
                    // active-extent scaling of params.seq.
                    seq_stride: seq,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                    _p4: 0,
                    _p5: 0,
                };
                schedule.push(Step::SelectiveScan { params: p });
                let ssk = selective_scan_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<SelectiveScanParams>());
                let bg = bind_op_output_window(&dev.device, ssk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Mamba2 {
                head_dim,
                state_size,
            } => {
                if *state_size > 256 {
                    panic!(
                        "rlx-wgpu Mamba2: state_size {} exceeds compile-time cap of 256",
                        state_size
                    );
                }
                let x_id = node.inputs[0];
                let in_dims = graph.node(x_id).shape.dims(); // [B,S,H,P]
                let seq = in_dims[1].unwrap_static() as u32;
                let p = Mamba2Params {
                    batch: in_dims[0].unwrap_static() as u32,
                    seq,
                    heads: in_dims[2].unwrap_static() as u32,
                    head_dim: *head_dim as u32,
                    state_size: *state_size as u32,
                    x_off: (arena.offset(x_id) / 4) as u32,
                    dt_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    a_off: (arena.offset(node.inputs[2]) / 4) as u32,
                    b_off: (arena.offset(node.inputs[3]) / 4) as u32,
                    c_off: (arena.offset(node.inputs[4]) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    seq_stride: seq,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                    _p4: 0,
                };
                schedule.push(Step::Mamba2 { params: p });
                let mk = mamba2_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<Mamba2Params>());
                let bg = bind_op_output_window(&dev.device, mk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let x_id = node.inputs[0];
                let in_dims = graph.node(x_id).shape.dims(); // [B,S,In]
                let batch = in_dims[0].unwrap_static() as u32;
                let seq = in_dims[1].unwrap_static() as u32;
                let input_size = in_dims[2].unwrap_static() as u32;
                let hidden = *hidden_size as u32;
                if hidden <= crate::supported_ops::NARROW_RNN_HIDDEN_MAX {
                    // One dispatch per (layer, direction); intermediate layer
                    // outputs ping-pong through the in-arena scratch pair. Bit-
                    // for-bit mirror of `execute_gru_f32` (weight packing).
                    let dirs = if *bidirectional { 2u32 } else { 1u32 };
                    let three_h = 3 * hidden;
                    let out_width = dirs * hidden;
                    let layer_elems = batch * seq * out_width;
                    let scratch_w = (rnn_gru_scratch_off / 4) as u32;
                    let dst_w = (arena.offset(node.id) / 4) as u32;
                    let wih_w = (arena.offset(node.inputs[1]) / 4) as u32;
                    let whh_w = (arena.offset(node.inputs[2]) / 4) as u32;
                    let bih_w = (arena.offset(node.inputs[3]) / 4) as u32;
                    let bhh_w = (arena.offset(node.inputs[4]) / 4) as u32;
                    let h0_w = if *carry {
                        (arena.offset(node.inputs[5]) / 4) as u32
                    } else {
                        0
                    };

                    let mut in_l = input_size;
                    let mut wih_cursor = 0u32;
                    let mut in_off = (arena.offset(x_id) / 4) as u32;
                    for l in 0..*num_layers as u32 {
                        let last = l + 1 == *num_layers as u32;
                        let out_off = if last {
                            dst_w
                        } else {
                            scratch_w + (l % 2) * layer_elems
                        };
                        let wih_block = three_h * in_l;
                        for dir in 0..dirs {
                            let ld = l * dirs + dir;
                            let p = GruParams {
                                batch,
                                seq,
                                input_size: in_l,
                                hidden,
                                x_off: in_off,
                                wih_off: wih_w + wih_cursor + dir * wih_block,
                                whh_off: whh_w + ld * three_h * hidden,
                                bih_off: bih_w + ld * three_h,
                                bhh_off: bhh_w + ld * three_h,
                                out_off,
                                seq_stride: seq,
                                h0_off: if *carry {
                                    h0_w + ld * batch * hidden
                                } else {
                                    0
                                },
                                out_width,
                                dir_off: dir * hidden,
                                reverse: u32::from(dir == 1),
                                _p5: 0,
                            };
                            schedule.push(Step::Gru { params: p });
                            let gk = gru_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<GruParams>());
                            let bg = bind_op_output_window(&dev.device, gk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        wih_cursor += dirs * wih_block;
                        in_l = out_width;
                        in_off = scratch_w + (l % 2) * layer_elems;
                    }
                } else {
                    let h0 = if *carry {
                        arena.offset(node.inputs[5]) as u32
                    } else {
                        0
                    };
                    schedule.push(Step::GruHost {
                        x: arena.offset(x_id) as u32,
                        w_ih: arena.offset(node.inputs[1]) as u32,
                        w_hh: arena.offset(node.inputs[2]) as u32,
                        b_ih: arena.offset(node.inputs[3]) as u32,
                        b_hh: arena.offset(node.inputs[4]) as u32,
                        h0,
                        dst: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        input_size,
                        hidden,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    });
                }
            }
            Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
                relu,
            } => {
                let x_id = node.inputs[0];
                let in_dims = graph.node(x_id).shape.dims();
                let batch = in_dims[0].unwrap_static() as u32;
                let seq = in_dims[1].unwrap_static() as u32;
                let input_size = in_dims[2].unwrap_static() as u32;
                let hidden = *hidden_size as u32;
                if hidden <= crate::supported_ops::NARROW_RNN_HIDDEN_MAX {
                    // One dispatch per (layer, direction); intermediate layer
                    // outputs ping-pong through the in-arena scratch pair. Bit-
                    // for-bit mirror of `execute_rnn_f32` (single merged bias).
                    let dirs = if *bidirectional { 2u32 } else { 1u32 };
                    let out_width = dirs * hidden;
                    let layer_elems = batch * seq * out_width;
                    let scratch_w = (rnn_gru_scratch_off / 4) as u32;
                    let dst_w = (arena.offset(node.id) / 4) as u32;
                    let wih_w = (arena.offset(node.inputs[1]) / 4) as u32;
                    let whh_w = (arena.offset(node.inputs[2]) / 4) as u32;
                    let bias_w = (arena.offset(node.inputs[3]) / 4) as u32;
                    let h0_w = if *carry {
                        (arena.offset(node.inputs[4]) / 4) as u32
                    } else {
                        0
                    };

                    let mut in_l = input_size;
                    let mut wih_cursor = 0u32;
                    let mut in_off = (arena.offset(x_id) / 4) as u32;
                    for l in 0..*num_layers as u32 {
                        let last = l + 1 == *num_layers as u32;
                        let out_off = if last {
                            dst_w
                        } else {
                            scratch_w + (l % 2) * layer_elems
                        };
                        let wih_block = hidden * in_l;
                        for dir in 0..dirs {
                            let ld = l * dirs + dir;
                            let p = RnnParams {
                                batch,
                                seq,
                                input_size: in_l,
                                hidden,
                                x_off: in_off,
                                wih_off: wih_w + wih_cursor + dir * wih_block,
                                whh_off: whh_w + ld * hidden * hidden,
                                bias_off: bias_w + ld * hidden,
                                out_off,
                                seq_stride: seq,
                                relu: u32::from(*relu),
                                h0_off: if *carry {
                                    h0_w + ld * batch * hidden
                                } else {
                                    0
                                },
                                out_width,
                                dir_off: dir * hidden,
                                reverse: u32::from(dir == 1),
                            };
                            schedule.push(Step::Rnn { params: p });
                            let rk = rnn_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<RnnParams>());
                            let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        wih_cursor += dirs * wih_block;
                        in_l = out_width;
                        in_off = scratch_w + (l % 2) * layer_elems;
                    }
                } else {
                    let h0 = if *carry {
                        arena.offset(node.inputs[4]) as u32
                    } else {
                        0
                    };
                    schedule.push(Step::RnnHost {
                        x: arena.offset(x_id) as u32,
                        w_ih: arena.offset(node.inputs[1]) as u32,
                        w_hh: arena.offset(node.inputs[2]) as u32,
                        bias: arena.offset(node.inputs[3]) as u32,
                        h0,
                        dst: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        input_size,
                        hidden,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                        relu: *relu,
                    });
                }
            }
            Op::GatedDeltaNet {
                state_size,
                carry_state,
                gate_per_channel,
            } => {
                if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                    panic!(
                        "rlx-wgpu GatedDeltaNet: state_size {state_size} > {}",
                        rlx_cpu::gdn::GDN_MAX_STATE
                    );
                }
                let q_id = node.inputs[0];
                let k_id = node.inputs[1];
                let v_id = node.inputs[2];
                let g_id = node.inputs[3];
                let beta_id = node.inputs[4];
                let q_shape = &graph.node(q_id).shape;
                let batch = q_shape.dim(0).unwrap_static() as u32;
                let seq = q_shape.dim(1).unwrap_static() as u32;
                let heads = q_shape.dim(2).unwrap_static() as u32;
                let n = *state_size as u32;
                let q_byte = arena.offset(q_id) as u64;
                let k_byte = arena.offset(k_id) as u64;
                let v_byte = arena.offset(v_id) as u64;
                let g_byte = arena.offset(g_id) as u64;
                let beta_byte = arena.offset(beta_id) as u64;
                let dst_byte = arena.offset(node.id) as u64;
                let q_len = arena.len_of(q_id) as u64;
                let k_len = arena.len_of(k_id) as u64;
                let v_len = arena.len_of(v_id) as u64;
                let g_len = arena.len_of(g_id) as u64;
                let beta_len = arena.len_of(beta_id) as u64;
                let dst_len = arena.len_of(node.id) as u64;
                let state_len = (batch as u64) * (heads as u64) * (n as u64) * (n as u64) * 4;
                let mut gdn_ids = vec![q_id, k_id, v_id, g_id, beta_id, node.id];
                let carry_id = if *carry_state {
                    gdn_ids.push(node.inputs[5]);
                    Some(node.inputs[5])
                } else {
                    None
                };
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let colocated = arena_span_bytes(&arena, &gdn_ids) <= max_binding
                    && !arena_span_crosses_shard(&arena, &gdn_ids);
                // Pack working set into the output stripe's stage reserve when
                // Q/K/V/… are split across shards (common on >4 GiB arenas).
                let pack_bytes = q_len
                    + k_len
                    + v_len
                    + g_len
                    + beta_len
                    + dst_len
                    + if *carry_state {
                        arena.len_of(node.inputs[5]) as u64
                    } else {
                        state_len
                    };
                let can_pack = arena.is_sharded()
                    && !colocated
                    && pack_bytes <= crate::buffer::shard_stage_reserve() as u64;
                let use_gpu = !rlx_ir::env::flag("RLX_WGPU_GDN_HOST")
                    && (colocated || can_pack)
                    && (*carry_state || gdn_scratch_off > 0 || arena.is_sharded() || can_pack);

                let (
                    bind_base,
                    bind_size,
                    q_off_w,
                    k_off_w,
                    v_off_w,
                    g_off_w,
                    beta_off_w,
                    state_off_w,
                    out_off_w,
                    packed_out_byte,
                    packed_state_byte,
                ) = if use_gpu && colocated {
                    let (base, size) = arena_window_for_nodes(&dev.device, &arena, &gdn_ids);
                    let reb = base / 4;
                    let state_byte = if let Some(sid) = carry_id {
                        arena.offset(sid) as u64
                    } else if arena.is_sharded() {
                        arena.shard_stage_off(arena.offset(q_id)) as u64
                    } else {
                        gdn_scratch_off as u64
                    };
                    (
                        base,
                        size,
                        (q_byte / 4) - reb,
                        (k_byte / 4) - reb,
                        (v_byte / 4) - reb,
                        (g_byte / 4) - reb,
                        (beta_byte / 4) - reb,
                        (state_byte / 4) - reb,
                        (dst_byte / 4) - reb,
                        None,
                        None,
                    )
                } else if use_gpu && can_pack {
                    // Contiguous pack in the output shard's stage reserve.
                    let stage = arena.shard_stage_off(arena.offset(node.id)) as u64;
                    let shard_base = (arena.offset(node.id) as u64 / arena.shard_size as u64)
                        * arena.shard_size as u64;
                    let (bind_base, bind_size) =
                        arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let mut cur = stage;
                    let align = |c: u64| c.div_ceil(256) * 256;
                    let mut place = |cur: &mut u64, src: u64, len: u64| {
                        *cur = align(*cur);
                        let dst = *cur;
                        schedule.push(Step::BufferCopy {
                            src_byte_off: src,
                            dst_byte_off: dst,
                            bytes: len as u32,
                        });
                        *cur = dst + len;
                        dst
                    };
                    let q_s = place(&mut cur, q_byte, q_len);
                    let k_s = place(&mut cur, k_byte, k_len);
                    let v_s = place(&mut cur, v_byte, v_len);
                    let g_s = place(&mut cur, g_byte, g_len);
                    let beta_s = place(&mut cur, beta_byte, beta_len);
                    let out_s = {
                        cur = align(cur);
                        let d = cur;
                        cur = d + dst_len;
                        d
                    };
                    let (state_s, packed_state) = if let Some(sid) = carry_id {
                        let s = place(&mut cur, arena.offset(sid) as u64, arena.len_of(sid) as u64);
                        (s, Some(s))
                    } else {
                        cur = align(cur);
                        let d = cur;
                        cur = d + state_len;
                        (d, None)
                    };
                    assert!(
                        cur <= stage + crate::buffer::shard_stage_reserve() as u64,
                        "rlx-wgpu GDN pack overflowed shard stage reserve"
                    );
                    let _ = shard_base;
                    let reb = bind_base / 4;
                    (
                        bind_base,
                        bind_size,
                        (q_s / 4) - reb,
                        (k_s / 4) - reb,
                        (v_s / 4) - reb,
                        (g_s / 4) - reb,
                        (beta_s / 4) - reb,
                        (state_s / 4) - reb,
                        (out_s / 4) - reb,
                        Some(out_s),
                        packed_state,
                    )
                } else {
                    (0u64, 0u64, 0, 0, 0, 0, 0, 0, 0, None, None)
                };

                let p = GatedDeltaNetParams {
                    batch,
                    seq,
                    heads,
                    state_size: n,
                    q_off: q_off_w as u32,
                    k_off: k_off_w as u32,
                    v_off: v_off_w as u32,
                    g_off: g_off_w as u32,
                    beta_off: beta_off_w as u32,
                    state_off: state_off_w as u32,
                    out_off: out_off_w as u32,
                    use_carry: if *carry_state { 1 } else { 0 },
                    seq_stride: seq,
                    gate_per_channel: if *gate_per_channel { 1 } else { 0 },
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::GatedDeltaNet {
                    params: p,
                    q_byte_off: q_byte,
                    k_byte_off: k_byte,
                    v_byte_off: v_byte,
                    g_byte_off: g_byte,
                    beta_byte_off: beta_byte,
                    state_byte_off: if *carry_state {
                        arena.offset(node.inputs[5]) as u64
                    } else {
                        0
                    },
                    dst_byte_off: dst_byte,
                    use_gpu,
                });
                if use_gpu {
                    let gk = gated_delta_net_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GatedDeltaNetParams>());
                    let bg = bind_arena_window(&dev.device, gk, &arena, bind_base, bind_size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                    if let Some(packed_out) = packed_out_byte {
                        schedule.push(Step::BufferCopy {
                            src_byte_off: packed_out,
                            dst_byte_off: dst_byte,
                            bytes: dst_len as u32,
                        });
                    }
                    // Pack path updates recurrent state in the stage
                    // reserve; write it back to the persistent carry slot
                    // (decode graphs export this across steps).
                    if let (Some(sid), Some(state_src)) = (carry_id, packed_state_byte) {
                        schedule.push(Step::BufferCopy {
                            src_byte_off: state_src,
                            dst_byte_off: arena.offset(sid) as u64,
                            bytes: arena.len_of(sid) as u32,
                        });
                    }
                }
                // Host path: schedule-only (no uniform/bind_group), same as
                // LSTM — otherwise gpu_ui/gpu_bi drift and later steps land
                // on wrong-sized uniforms ("Copy 76 bytes into a 32-byte
                // buffer" on Bonsai-27B hybrid).
            }
            Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let batch = x_shape.dim(0).unwrap_static() as u32;
                let seq = x_shape.dim(1).unwrap_static() as u32;
                let input_size = x_shape.dim(2).unwrap_static() as u32;
                let hidden = *hidden_size as u32;
                // `hidden <= 256` is the workgroup-size bound; wider goes to the
                // host `Step::Lstm` below.
                //
                // This was briefly clamped to 32 on Apple adapters: Metal's fast
                // `exp`/`tanh` are not accurate enough for this recurrence, and
                // past hidden 32 the error compounded through the cell state
                // until units collapsed to exactly 0.0. The kernel now uses its
                // own range-reduced `rlx_exp_precise` (see `lstm.wgsl`) instead
                // of relying on the backend's fast transcendentals, so the bound
                // is back to the real limit on every adapter. Checked against an
                // independent f64 reference in
                // `rlx-runtime/tests/lstm_three_way.rs`.
                if hidden <= crate::supported_ops::NARROW_RNN_HIDDEN_MAX {
                    // One dispatch per (layer, direction); intermediate layer
                    // outputs ping-pong through the in-arena scratch pair. Bit-
                    // for-bit mirror of `execute_lstm_f32` (gate order i,f,g,o;
                    // single merged bias).
                    let dirs = if *bidirectional { 2u32 } else { 1u32 };
                    let four_h = 4 * hidden;
                    let out_width = dirs * hidden;
                    let layer_elems = batch * seq * out_width;
                    let scratch_w = (rnn_gru_scratch_off / 4) as u32;
                    let dst_w = (arena.offset(node.id) / 4) as u32;
                    let wih_w = (arena.offset(node.inputs[1]) / 4) as u32;
                    let whh_w = (arena.offset(node.inputs[2]) / 4) as u32;
                    let bias_w = (arena.offset(node.inputs[3]) / 4) as u32;
                    let (h0_w, c0_w) = if *carry {
                        (
                            (arena.offset(node.inputs[4]) / 4) as u32,
                            (arena.offset(node.inputs[5]) / 4) as u32,
                        )
                    } else {
                        (0, 0)
                    };

                    let mut in_l = input_size;
                    let mut wih_cursor = 0u32;
                    let mut in_off = (arena.offset(node.inputs[0]) / 4) as u32;
                    for l in 0..*num_layers as u32 {
                        let last = l + 1 == *num_layers as u32;
                        let out_off = if last {
                            dst_w
                        } else {
                            scratch_w + (l % 2) * layer_elems
                        };
                        let wih_block = four_h * in_l;
                        for dir in 0..dirs {
                            let ld = l * dirs + dir;
                            let p = LstmParams {
                                batch,
                                seq,
                                input_size: in_l,
                                hidden,
                                x_off: in_off,
                                wih_off: wih_w + wih_cursor + dir * wih_block,
                                whh_off: whh_w + ld * four_h * hidden,
                                bias_off: bias_w + ld * four_h,
                                out_off,
                                seq_stride: seq,
                                h0_off: if *carry {
                                    h0_w + ld * batch * hidden
                                } else {
                                    0
                                },
                                c0_off: if *carry {
                                    c0_w + ld * batch * hidden
                                } else {
                                    0
                                },
                                out_width,
                                dir_off: dir * hidden,
                                reverse: u32::from(dir == 1),
                                carry: u32::from(*carry),
                            };
                            schedule.push(Step::LstmGpu { params: p });
                            let lk = lstm_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<LstmParams>());
                            let bg = bind_op_output_window(&dev.device, lk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        wih_cursor += dirs * wih_block;
                        in_l = out_width;
                        in_off = scratch_w + (l % 2) * layer_elems;
                    }
                } else {
                    let (h0, c0) = if *carry {
                        (
                            arena.offset(node.inputs[4]) as u32,
                            arena.offset(node.inputs[5]) as u32,
                        )
                    } else {
                        (0u32, 0u32)
                    };
                    schedule.push(Step::Lstm {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                        w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                        bias_byte_off: arena.offset(node.inputs[3]) as u32,
                        h0_byte_off: h0,
                        c0_byte_off: c0,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        input_size,
                        hidden,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    });
                    // `Step::Lstm` is a host step: BOTH run loops skip it (the
                    // uniform pre-pass `continue`s, the compute pass `break`s), so
                    // it must push NEITHER a uniform NOR a bind_group — otherwise
                    // every later GPU step's `gpu_ui`/`gpu_bi` index is off by one
                    // and a downstream matmul reads a wrong-sized uniform.
                }
            }
            Op::ConvTranspose2d {
                kernel_size,
                stride,
                padding,
                dilation,
                output_padding: _,
                groups,
            } => {
                let in_shape = &graph.node(node.inputs[0]).shape;
                let out_shape = &node.shape;
                if wgpu_prefer_conv_host(&arena) {
                    // Host route: the compact BLAS-backed staging, forced here by the lever
                    // or the structure-host preference — and the one route whose 32-bit arena
                    // byte offsets cannot serve a parked parameter or an arena past u32::MAX,
                    // which is what the refusal below is about. The device step this op takes
                    // otherwise stages a parked operand instead.
                    assert_byte_offset_host_route(
                        &graph,
                        &arena,
                        node,
                        "let the op take its device step instead — drop RLX_WGPU_CONV_HOST and the \
                         structure-host preference (RLX_WGPU_FORCE_HOST, a discrete adapter, a \
                         sharded arena), which reads its operands through the bind window rather \
                         than as 32-bit arena offsets",
                    );
                    schedule.push(Step::ConvTranspose2dHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        weight_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w_in: in_shape.dim(3).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride[0] as u32,
                        sw: stride[1] as u32,
                        ph: padding[0] as u32,
                        pw: padding[1] as u32,
                        dh: dilation[0] as u32,
                        dw: dilation[1] as u32,
                        groups: *groups as u32,
                    });
                    // Host step: schedule-only, like `MsDeformAttnHost`/`UmapKnnHost`.
                    // The uniform/bind_group lanes are indexed by GPU-step count
                    // (`gpu_ui`), so host steps must NOT push to them.
                } else {
                    // Device route: two forms. The tiled GEMM takes the shapes
                    // `conv_transpose2d_tiled_fits` admits (unit height and batch, one group,
                    // no dilation, `k ≤ 2·s`); the rest take the sibling 3-D gather with the
                    // depth axis pinned to one, so the 2-D geometry is the 3-D one at
                    // d = d_out = kd = sd = 1, pd = 0, dd = 1.
                    //
                    // The device step reads its operands as one f32 lane per element off
                    // the arena, so a parameter the arena does not store that way (packed
                    // U8, resized BF16) is refused rather than misread. The host route is
                    // guarded the same way, inside `assert_byte_offset_host_route`.
                    assert_param_operands_are_lane_readable(&graph, &arena, node);
                    //
                    // The gather groups on `c_in / groups` and divides the output channel by
                    // `c_out / groups`, so a group count that does not partition the input
                    // channels is refused rather than run as a different operator. The two
                    // non-divisible cases are different failures with different ways out: with
                    // fewer channels than groups no group is ever complete and the gather would
                    // produce the zeroed output the replaced host route produced, while with
                    // more channels it would accumulate the ones inside the whole groups it can
                    // form and silently drop the rest — a geometry the host route cannot take
                    // either, since its own scatter maps those surplus channels to group indices
                    // past the last one and traps on the output write.
                    let c_in = in_shape.dim(1).unwrap_static() as u32;
                    assert!(
                        *groups >= 1,
                        "rlx-wgpu: REFUSED (degenerate Op::ConvTranspose2d grouping): groups=0 \
                         would make the device gather — the step every grouped form keeps — divide \
                         by a zero group size. There is no route to fall back to here — the host \
                         step divides by it too — so use a positive group count."
                    );
                    if !c_in.is_multiple_of(*groups as u32) {
                        if (c_in as usize) < *groups {
                            panic!(
                                "rlx-wgpu: REFUSED (degenerate Op::ConvTranspose2d grouping): \
                                 groups={} exceeds the {c_in} input channels, so no group is ever \
                                 complete and the gather would leave its output zero. The host route \
                                 zeroes it the same way (RLX_WGPU_CONV_HOST=1, or \
                                 RLX_WGPU_FORCE_HOST=1) — that is where this geometry's previous \
                                 behaviour lives — or use a group count that divides the input \
                                 channel count.",
                                groups
                            );
                        }
                        panic!(
                            "rlx-wgpu: REFUSED (degenerate Op::ConvTranspose2d grouping): groups={} \
                             does not divide {c_in} input channels, so the gather would accumulate \
                             only the channels inside the whole groups it can form and silently \
                             drop the rest. No route computes this geometry: the host route's own \
                             scatter walks its channel indices past its output and traps there. \
                             Use a group count that divides the input channel count.",
                            groups
                        );
                    }
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    // The IR validates the weight's channel dims against the input's but not
                    // its taps, and the gather indexes the weight with *this op's*
                    // `kernel_size` — so a graph whose attribute disagrees with the weight
                    // would read across the operand inside the binding instead of failing.
                    // The IR's own shape rule says the weight is `[C_in, C_out/g, kH, kW]`,
                    // so the two are the same number by contract.
                    //
                    // Only this arm is asked. The host route sizes its staging from the same
                    // declared attribute (`run_conv2d_backward_input` copies
                    // `C_out·C_in/g·kH·kW` f32 starting at the weight), so an over-declared
                    // tap count makes it read neighbouring arena bytes just as silently —
                    // a pre-existing behaviour of that route, reported with this one rather
                    // than changed by it.
                    let w_shape = &graph.node(w_id).shape;
                    let (wkh, wkw) = (
                        w_shape.dim(2).unwrap_static(),
                        w_shape.dim(3).unwrap_static(),
                    );
                    assert!(
                        kernel_size[0] == wkh && kernel_size[1] == wkw,
                        "rlx-wgpu: REFUSED (Op::ConvTranspose2d kernel size): the op declares \
                         {}x{} taps and the weight is {wkh}x{wkw} per channel, so the device route \
                         would read across its operand. Declare the kernel size the weight really \
                         has.",
                        kernel_size[0],
                        kernel_size[1]
                    );
                    let ct_win = vec![node.id, in_id, w_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let ct_fits = arena_span_bytes(&arena, &ct_win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &ct_win,
                    );
                    if !ct_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    assert_conv_transpose_operands_addressable(
                        &arena,
                        &dev.device,
                        in_id,
                        base,
                        size,
                    );
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    assert_conv_transpose_operands_addressable(
                        &arena,
                        &dev.device,
                        w_id,
                        base,
                        size,
                    );
                    let w_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        w_id,
                        &mut base,
                        &mut size,
                    );
                    let p2 = Conv3dParams {
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        // The depth axis is pinned to one; the gather reads and writes the
                        // same slice it would for a 2-D NCHW tensor.
                        d: 1,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        d_out: 1,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kd: 1,
                        // Taps come from the op's attribute, as they do on the host step —
                        // not from the weight shape, which the 3-D form reads.
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sd: 1,
                        sh: stride[0] as u32,
                        sw: stride[1] as u32,
                        pd: 0,
                        ph: padding[0] as u32,
                        pw: padding[1] as u32,
                        dd: 1,
                        dh: dilation[0] as u32,
                        dw: dilation[1] as u32,
                        groups: *groups as u32,
                        in_off,
                        w_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        bias_off: 0,
                        has_bias: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    // The tiled GEMM serves the family it is proven for (see
                    // `conv_transpose2d_tiled_fits`); the gather serves the rest.
                    let tiled = conv_transpose2d_tiled_fits(&p2);
                    schedule.push(if tiled {
                        Step::ConvTranspose2dTiled { params: p2 }
                    } else {
                        Step::ConvTranspose2d { params: p2 }
                    });
                    let ck = if tiled {
                        crate::kernels::conv_transpose2d_tiled_kernel(&dev.device)
                    } else {
                        conv_transpose2d_kernel(&dev.device)
                    };
                    let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                    let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }
            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let win_ids = [node.id, in_id, w_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                arena_expand_bind_window(&arena, &win_ids, &mut base, &mut size, max_binding);
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let w_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    w_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                let in_shape = graph.node(in_id).shape.dims();
                let w_shape = graph.node(w_id).shape.dims();
                let out_shape = node.shape.dims();
                // A third input is the per-output-channel bias, folded into
                // the conv's store — see `fold_conv_bias_into_conv3d`.
                let (bias_off, has_bias) = match node.inputs.get(2) {
                    Some(&b) => (
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            b,
                            &mut base,
                            &mut size,
                        ),
                        1u32,
                    ),
                    None => (0u32, 0u32),
                };
                let p3 = Conv3dParams {
                    n: in_shape[0].unwrap_static() as u32,
                    c_in: in_shape[1].unwrap_static() as u32,
                    c_out: out_shape[1].unwrap_static() as u32,
                    d: in_shape[2].unwrap_static() as u32,
                    h: in_shape[3].unwrap_static() as u32,
                    w: in_shape[4].unwrap_static() as u32,
                    d_out: out_shape[2].unwrap_static() as u32,
                    h_out: out_shape[3].unwrap_static() as u32,
                    w_out: out_shape[4].unwrap_static() as u32,
                    kd: w_shape[2].unwrap_static() as u32,
                    kh: w_shape[3].unwrap_static() as u32,
                    kw: w_shape[4].unwrap_static() as u32,
                    sd: stride[0] as u32,
                    sh: stride[1] as u32,
                    sw: stride[2] as u32,
                    pd: padding[0] as u32,
                    ph: padding[1] as u32,
                    pw: padding[2] as u32,
                    dd: dilation[0] as u32,
                    dh: dilation[1] as u32,
                    dw: dilation[2] as u32,
                    groups: (*groups).max(1) as u32,
                    in_off,
                    w_off,
                    out_off,
                    bias_off,
                    has_bias,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Conv3d { params: p3 });
                let ck = conv3d_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                let bg = bind_arena_window(&dev.device, ck, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                output_padding: _,
                groups,
            } => {
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let in_shape = &graph.node(in_id).shape;
                let w_shape = &graph.node(w_id).shape;
                let out_shape = &node.shape;
                let n = in_shape.dim(0).unwrap_static() as u32;
                let c_in = in_shape.dim(1).unwrap_static() as u32;
                let d = in_shape.dim(2).unwrap_static() as u32;
                let h = in_shape.dim(3).unwrap_static() as u32;
                let w_in = in_shape.dim(4).unwrap_static() as u32;
                let c_out = out_shape.dim(1).unwrap_static() as u32;
                let d_out = out_shape.dim(2).unwrap_static() as u32;
                let h_out = out_shape.dim(3).unwrap_static() as u32;
                let w_out = out_shape.dim(4).unwrap_static() as u32;
                let kd = w_shape.dim(2).unwrap_static() as u32;
                let kh = w_shape.dim(3).unwrap_static() as u32;
                let kw = w_shape.dim(4).unwrap_static() as u32;
                let sd = stride[0] as u32;
                let sh = stride[1] as u32;
                let sw = stride[2] as u32;
                let pd = padding[0] as u32;
                let ph = padding[1] as u32;
                let pw = padding[2] as u32;
                let dd = dilation[0] as u32;
                let dh = dilation[1] as u32;
                let dw = dilation[2] as u32;
                let groups_u = (*groups).max(1) as u32;

                if wgpu_prefer_conv_host(&arena) {
                    schedule.push(Step::ConvTranspose3dHost {
                        src_byte_off: arena.offset(in_id) as u32,
                        weight_byte_off: arena.offset(w_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n,
                        c_in,
                        d,
                        h,
                        w_in,
                        c_out,
                        d_out,
                        h_out,
                        w_out,
                        kd,
                        kh,
                        kw,
                        sd,
                        sh,
                        sw,
                        pd,
                        ph,
                        pw,
                        dd,
                        dh,
                        dw,
                        groups: groups_u,
                    });
                } else {
                    let ct_win = vec![node.id, in_id, w_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let ct_fits = arena_span_bytes(&arena, &ct_win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &ct_win,
                    );
                    if !ct_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let w_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        w_id,
                        &mut base,
                        &mut size,
                    );
                    let p3 = Conv3dParams {
                        n,
                        c_in,
                        c_out,
                        d,
                        h,
                        w: w_in,
                        d_out,
                        h_out,
                        w_out,
                        kd,
                        kh,
                        kw,
                        sd,
                        sh,
                        sw,
                        pd,
                        ph,
                        pw,
                        dd,
                        dh,
                        dw,
                        groups: groups_u,
                        in_off,
                        w_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        bias_off: 0,
                        has_bias: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::ConvTranspose3d { params: p3 });
                    let sk = conv_transpose3d_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                    let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }
            Op::GroupNorm { num_groups, eps } if !wgpu_prefer_host_norm(&arena) => {
                // Native WGSL kernel: one workgroup per (batch, group). The
                // host step below mirrors the WHOLE arena per norm, which is
                // ruinous on any graph with more than a couple of them.
                let in_id = node.inputs[0];
                let gamma_id = node.inputs[1];
                let beta_id = node.inputs[2];
                let dims = graph.node(in_id).shape.dims();
                let n = dims[0].unwrap_static() as u32;
                let c = dims[1].unwrap_static() as u32;
                let h = dims[2].unwrap_static() as u32;
                let w = dims[3].unwrap_static() as u32;
                let g = (*num_groups).max(1) as u32;

                let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                let gamma_bytes = arena.len_of(gamma_id) as u64;
                let gn_win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                    vec![gamma_id, node.id, in_id]
                } else {
                    let mut v = vec![node.id, in_id];
                    if gamma_is_param {
                        v.push(gamma_id);
                        v.push(beta_id);
                    }
                    v
                };
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let gn_fits = arena_span_bytes(&arena, &gn_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &gn_win,
                );
                if !gn_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let gamma_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    gamma_id,
                    &mut base,
                    &mut size,
                );
                let beta_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    beta_id,
                    &mut base,
                    &mut size,
                );
                // Same stale-offset trap as LayerNorm: staging gamma/beta can
                // relocate or widen `base` after `in_off` was resolved, leaving
                // it pointing into the wrong stripe.
                let in_off = if arena_tensor_in_window(&arena, in_id, base, size) {
                    arena_local_off_f32(&arena, in_id, base)
                } else {
                    in_off
                };
                let gamma_off = if arena_tensor_in_window(&arena, gamma_id, base, size) {
                    arena_local_off_f32(&arena, gamma_id, base)
                } else {
                    gamma_off
                };
                let beta_off = if arena_tensor_in_window(&arena, beta_id, base, size) {
                    arena_local_off_f32(&arena, beta_id, base)
                } else {
                    beta_off
                };
                let p = GroupNormParams {
                    groups_total: n * g,
                    group_elems: (c / g) * h * w,
                    in_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    gamma_off,
                    beta_off,
                    eps_bits: eps.to_bits(),
                    num_groups: g,
                    c,
                    hw: h * w,
                    _pad0: 0,
                    _pad1: 0,
                };
                schedule.push(Step::GroupNorm { params: p });
                let gk = group_norm_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<GroupNormParams>());
                let bg = bind_arena_window(&dev.device, gk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::GroupNorm { num_groups, eps } => {
                // NCHW: x [n,c,h,w], gamma/beta [c]. Host step (schedule-only).
                let in_shape = &graph.node(node.inputs[0]).shape;
                schedule.push(Step::GroupNormHost {
                    src_byte_off: arena.offset(node.inputs[0]) as u32,
                    gamma_byte_off: arena.offset(node.inputs[1]) as u32,
                    beta_byte_off: arena.offset(node.inputs[2]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    n: in_shape.dim(0).unwrap_static() as u32,
                    c: in_shape.dim(1).unwrap_static() as u32,
                    h: in_shape.dim(2).unwrap_static() as u32,
                    w: in_shape.dim(3).unwrap_static() as u32,
                    num_groups: *num_groups as u32,
                    eps: *eps,
                });
            }
            Op::LayerNorm2d { eps } => {
                let in_id = node.inputs[0];
                let gamma_id = node.inputs[1];
                let beta_id = node.inputs[2];
                let in_shape = &graph.node(in_id).shape;
                // Eligibility first, loudest first: this op is NCHW, and both of
                // this backend's routes read `h`/`w` from axes 2 and 3, so any
                // other rank would be normalised over the wrong axes — silently,
                // for the ranks a shape can still be read at. `rlx-cpu` also
                // takes rank 3 as `[N, C, 1, W]`; neither step here implements
                // that form, so it is refused too.
                assert!(
                    matches!(in_shape.rank(), 4),
                    "rlx-wgpu: REFUSED (LayerNorm2d rank): Op::LayerNorm2d is NCHW-shaped and both \
                     of this backend's routes read h/w from axes 2 and 3, so rank {} would \
                     normalise over the wrong axes. Reshape to rank 4, or run the norm on a \
                     backend that accepts this rank.",
                    in_shape.rank()
                );
                // Complex packs several lanes per element (C64 = 2, C128 = 4) while
                // both routes read an operand as one f32 lane per element — the
                // whole-arena host step would lose half of every C64 element. It
                // is `should_pack_host_op`'s "COMPLEX NEVER HOSTS" verdict.
                for &id in &[in_id, gamma_id, beta_id, node.id] {
                    let dt = graph.node(id).shape.dtype();
                    assert!(
                        !dt.is_complex(),
                        "rlx-wgpu: REFUSED (LayerNorm2d encoding): {id:?} is {dt:?}, which packs \
                         several lanes per element, while both of this op's routes read an operand \
                         as one f32 lane per element. Cast the operand to a real dtype, or run \
                         the norm on a backend that reads its declared encoding."
                    );
                }
                // Native kernel: one workgroup per (batch, spatial) position,
                // tree reduction over the channel axis. `RLX_WGPU_HOST_NORM`
                // takes the host step and `RLX_WGPU_GPU_NORM` this one; the
                // adapter kind is not consulted (see `wgpu_prefer_host_norm2d`).
                if !wgpu_prefer_host_norm2d(&arena)
                    && param_operands_are_lane_readable(&graph, &arena, node)
                {
                    // gamma/beta can sit far from the activations, or be parked
                    // in the weight buffer; the same window + staging machinery
                    // GroupNorm uses resolves them next to `in` — including
                    // into the scratch tail via `Step::BufferCopy` when a single
                    // binding window cannot cover both.
                    let n = in_shape.dim(0).unwrap_static() as u32;
                    let c = in_shape.dim(1).unwrap_static() as u32;
                    let h = in_shape.dim(2).unwrap_static() as u32;
                    let w = in_shape.dim(3).unwrap_static() as u32;

                    let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                    let gamma_bytes = arena.len_of(gamma_id) as u64;
                    let win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                        vec![gamma_id, node.id, in_id]
                    } else {
                        let mut v = vec![node.id, in_id];
                        if gamma_is_param {
                            v.push(gamma_id);
                            v.push(beta_id);
                        }
                        v
                    };
                    let (base, size, offsets) = window_offsets(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &win,
                        &[in_id, gamma_id, beta_id],
                    );
                    let [in_off, gamma_off, beta_off] = offsets[..] else {
                        unreachable!("one offset per operand")
                    };
                    let p = LayerNorm2dParams {
                        positions: n * h * w,
                        c,
                        hw: h * w,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        gamma_off,
                        beta_off,
                        eps_bits: eps.to_bits(),
                    };
                    schedule.push(Step::LayerNorm2d { params: p });
                    let lk = layer_norm2d_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNorm2dParams>());
                    let bg = bind_arena_window(&dev.device, lk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }
                // Host route: a whole-arena step whose byte offsets are `u32`.
                assert_byte_offset_host_route(
                    &graph,
                    &arena,
                    node,
                    "run the norm on the device route",
                );
                let in_shape = &graph.node(in_id).shape;
                schedule.push(Step::LayerNorm2dHost {
                    src_byte_off: arena.offset(in_id) as u32,
                    gamma_byte_off: arena.offset(gamma_id) as u32,
                    beta_byte_off: arena.offset(beta_id) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    n: in_shape.dim(0).unwrap_static() as u32,
                    c: in_shape.dim(1).unwrap_static() as u32,
                    h: in_shape.dim(2).unwrap_static() as u32,
                    w: in_shape.dim(3).unwrap_static() as u32,
                    eps: *eps,
                });
            }
            Op::ResizeNearest2x => {
                // NCHW nearest 2× upsample. Host step (schedule-only).
                let in_shape = &graph.node(node.inputs[0]).shape;
                schedule.push(Step::ResizeNearest2xHost {
                    src_byte_off: arena.offset(node.inputs[0]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    n: in_shape.dim(0).unwrap_static() as u32,
                    c: in_shape.dim(1).unwrap_static() as u32,
                    h: in_shape.dim(2).unwrap_static() as u32,
                    w: in_shape.dim(3).unwrap_static() as u32,
                });
            }
            Op::Reverse { axes } => {
                let in_id = node.inputs[0];
                let in_shape = &graph.node(in_id).shape;
                let rank = in_shape.rank();
                let dims: Vec<u32> = (0..rank)
                    .map(|i| in_shape.dim(i).unwrap_static() as u32)
                    .collect();
                let mut rev_mask = vec![false; rank];
                for &a in axes {
                    if a < rank {
                        rev_mask[a] = true;
                    }
                }
                let total: u64 = dims.iter().map(|&d| d as u64).product();
                // Device path: a per-element lane permutation. NOT gated on
                // `coop_discrete_backend()` — this flip is exact on every
                // adapter, and putting it back on the host there is the cost
                // this route exists to remove. A sharded arena still hosts (a
                // single binding cannot cover it), `RLX_WGPU_FORCE_HOST` still
                // hosts, and an operand the arena does not hold as one f32 lane
                // per element keeps the host route.
                // Complex packs several lanes per element, so it keeps the host
                // route (which indexes what the arena stores). The flip is a pure
                // permutation, so `Op::Reverse` carries no parameter of its own —
                // `param_operands_are_lane_readable` still guards a graph that
                // feeds it one.
                if !in_shape.dtype().is_complex()
                    && !node.shape.dtype().is_complex()
                    && total > 0
                    && total <= u32::MAX as u64
                    && !wgpu_prefer_host_fallback_agnostic(&arena)
                    && param_operands_are_lane_readable(&graph, &arena, node)
                {
                    // Source strides over the full shape (the output has the
                    // same dims). A reversed axis keeps its own stride; only
                    // the index within it flips.
                    let mut strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }
                    // meta = dims (rank) ++ strides (rank) ++ rev flags (rank).
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 3);
                    meta_data.extend_from_slice(&dims);
                    meta_data.extend_from_slice(&strides);
                    meta_data.extend(rev_mask.iter().map(|&r| r as u32));
                    let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu reverse meta"),
                        size: (meta_data.len() * 4).max(4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue
                        .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta_buf);
                    // PLAN L1: the bucket axis is the leading dim. Scaling
                    // `total` is safe only when this flip does not touch it —
                    // the same per-instance decision `Step::Expand` makes with
                    // `bucket_outermost`.
                    let scale_leading = if rev_mask.first().copied().unwrap_or(false) {
                        0u32
                    } else {
                        1u32
                    };
                    let rev_ids = [node.id, in_id];
                    let (base, size, offsets) = window_offsets(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &rev_ids,
                        &[in_id, node.id],
                    );
                    let [in_off, out_off] = offsets[..] else {
                        unreachable!("one offset per operand")
                    };
                    let p = ReverseParams {
                        rank: rank as u32,
                        total: total as u32,
                        in_off,
                        out_off,
                        scale_leading,
                        dim_0: dims.first().copied().unwrap_or(1),
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::Reverse {
                        params: p,
                        meta_idx,
                    });
                    let rk = reverse_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ReverseParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu reverse bg"),
                        layout: &rk.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: arena_bind_buf(&arena, base).0,
                                    offset: arena_bind_buf(&arena, base).1,
                                    size: {
                                        let (b, l) = arena_bind_buf(&arena, base);
                                        aligned_bind_size(size, l, b.size())
                                    },
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                    continue;
                }
                // Host route. `Step::ReverseHost` derives its element width from
                // the declared dtype, so it is correct only for a tensor the arena
                // really stores at that width: complex (its slot is
                // `elems * size_bytes` of interleaved lanes) and the four-byte
                // reals keep this route; every other width is refused loudly.
                let host_dtype = in_shape.dtype();
                assert_byte_offset_host_route(
                    &graph,
                    &arena,
                    node,
                    "move the flip to the device route",
                );
                assert!(
                    host_dtype.is_complex() || host_dtype.size_bytes() == 4,
                    "rlx-wgpu: REFUSED (Reverse element width): the whole-arena host step moves \
                     `size_bytes` per element of a {host_dtype:?} tensor, but the arena stores \
                     real dtypes as one f32 lane per element, so the flip would move a different \
                     number of bytes than it permutes. Either widen the operand or move the flip \
                     to the device route."
                );
                schedule.push(Step::ReverseHost {
                    src_byte_off: arena.offset(in_id) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    dims,
                    rev_mask,
                    elem_bytes: host_dtype.size_bytes() as u32,
                });
            }
            Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                // ArgMax/ArgMin. Host step (schedule-only).
                let in_shape = &graph.node(node.inputs[0]).shape;
                let rank = in_shape.rank();
                let outer: usize = (0..*axis)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let reduced = in_shape.dim(*axis).unwrap_static();
                let inner: usize = (*axis + 1..rank)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                schedule.push(Step::ArgReduceHost {
                    src_byte_off: arena.offset(node.inputs[0]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    outer: outer as u32,
                    reduced: reduced as u32,
                    inner: inner as u32,
                    is_max: matches!(node.op, Op::ArgMax { .. }),
                });
            }
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => {
                let in_id = node.inputs[0];
                let in_shape = &graph.node(in_id).shape;
                let batch = in_shape.dim(0).unwrap_static() as u32;
                let seq = in_shape.dim(1).unwrap_static() as u32;
                let hidden = in_shape.dim(2).unwrap_static() as u32;
                if wgpu_prefer_structure_host(&arena) {
                    schedule.push(Step::AxialRope2dHost {
                        src_byte_off: arena.offset(in_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        hidden,
                        end_x: *end_x as u32,
                        end_y: *end_y as u32,
                        head_dim: *head_dim as u32,
                        num_heads: *num_heads as u32,
                        theta: *theta,
                        repeat_factor: *repeat_factor as u32,
                    });
                } else {
                    let win = vec![node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = AxialRope2dParams {
                        batch,
                        seq,
                        hidden,
                        end_x: *end_x as u32,
                        end_y: *end_y as u32,
                        head_dim: *head_dim as u32,
                        num_heads: *num_heads as u32,
                        repeat_factor: *repeat_factor as u32,
                        theta: *theta,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        n_total: batch * seq * hidden,
                    };
                    schedule.push(Step::AxialRope2d { params: p });
                    let sk = axial_rope2d_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AxialRope2dParams>());
                    let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let x_id = node.inputs[0];
                let dy_id = node.inputs[1];
                let x_shape = &graph.node(x_id).shape;
                let dy_shape = &graph.node(dy_id).shape;
                let mp_win = vec![node.id, x_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let mp_fits = arena_span_bytes(&arena, &mp_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &mp_win,
                );
                if !mp_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let p = MaxPool2dBwdParams {
                    n: x_shape.dim(0).unwrap_static() as u32,
                    c: x_shape.dim(1).unwrap_static() as u32,
                    h: x_shape.dim(2).unwrap_static() as u32,
                    w: x_shape.dim(3).unwrap_static() as u32,
                    h_out: dy_shape.dim(2).unwrap_static() as u32,
                    w_out: dy_shape.dim(3).unwrap_static() as u32,
                    kh: kernel_size[0] as u32,
                    kw: kernel_size[1] as u32,
                    sh: stride.first().copied().unwrap_or(1) as u32,
                    sw: stride.get(1).copied().unwrap_or(1) as u32,
                    ph: padding.first().copied().unwrap_or(0) as u32,
                    pw: padding.get(1).copied().unwrap_or(0) as u32,
                    x_off,
                    dy_off,
                    dx_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                };
                schedule.push(Step::MaxPool2dBackward { params: p });
                let sk = maxpool2d_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<MaxPool2dBwdParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::MaxPool3dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let x_id = node.inputs[0];
                let dy_id = node.inputs[1];
                let x_shape = &graph.node(x_id).shape;
                let dy_shape = &graph.node(dy_id).shape;
                let mp_win = vec![node.id, x_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let mp_fits = arena_span_bytes(&arena, &mp_win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &mp_win,
                );
                if !mp_fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let p = MaxPool3dBwdParams {
                    n: x_shape.dim(0).unwrap_static() as u32,
                    c: x_shape.dim(1).unwrap_static() as u32,
                    d: x_shape.dim(2).unwrap_static() as u32,
                    h: x_shape.dim(3).unwrap_static() as u32,
                    w: x_shape.dim(4).unwrap_static() as u32,
                    d_out: dy_shape.dim(2).unwrap_static() as u32,
                    h_out: dy_shape.dim(3).unwrap_static() as u32,
                    w_out: dy_shape.dim(4).unwrap_static() as u32,
                    kd: kernel_size[0] as u32,
                    kh: kernel_size[1] as u32,
                    kw: kernel_size[2] as u32,
                    sd: stride[0] as u32,
                    sh: stride[1] as u32,
                    sw: stride[2] as u32,
                    pd: padding[0] as u32,
                    ph: padding[1] as u32,
                    pw: padding[2] as u32,
                    x_off,
                    dy_off,
                    dx_off: arena_local_off_f32(&arena, node.id, base),
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };
                schedule.push(Step::MaxPool3dBackward { params: p });
                let sk = maxpool3d_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<MaxPool3dBwdParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Conv3dBackwardInput {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let dy_id = node.inputs[0];
                let w_id = node.inputs[1];
                let dy_shape = &graph.node(dy_id).shape;
                let w_shape = &graph.node(w_id).shape;
                let dx_shape = &node.shape;
                let win = vec![node.id, dy_id, w_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let w_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    w_id,
                    &mut base,
                    &mut size,
                );
                let p = Conv3dBwdInputParams {
                    n: dx_shape.dim(0).unwrap_static() as u32,
                    c_in: dx_shape.dim(1).unwrap_static() as u32,
                    d: dx_shape.dim(2).unwrap_static() as u32,
                    h: dx_shape.dim(3).unwrap_static() as u32,
                    w: dx_shape.dim(4).unwrap_static() as u32,
                    c_out: dy_shape.dim(1).unwrap_static() as u32,
                    d_out: dy_shape.dim(2).unwrap_static() as u32,
                    h_out: dy_shape.dim(3).unwrap_static() as u32,
                    w_out: dy_shape.dim(4).unwrap_static() as u32,
                    kd: w_shape.dim(2).unwrap_static() as u32,
                    kh: w_shape.dim(3).unwrap_static() as u32,
                    kw: w_shape.dim(4).unwrap_static() as u32,
                    sd: stride[0] as u32,
                    sh: stride[1] as u32,
                    sw: stride[2] as u32,
                    pd: padding[0] as u32,
                    ph: padding[1] as u32,
                    pw: padding[2] as u32,
                    dd: dilation[0] as u32,
                    dh: dilation[1] as u32,
                    dw: dilation[2] as u32,
                    groups: (*groups).max(1) as u32,
                    dy_off,
                    w_off,
                    dx_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Conv3dBackwardInput { params: p });
                let sk = conv3d_backward_input_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<Conv3dBwdInputParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Conv3dBackwardWeight {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let x_id = node.inputs[0];
                let dy_id = node.inputs[1];
                let x_shape = &graph.node(x_id).shape;
                let dy_shape = &graph.node(dy_id).shape;
                let dw_shape = &node.shape;
                let win = vec![node.id, x_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let p = Conv3dBwdWeightParams {
                    n: x_shape.dim(0).unwrap_static() as u32,
                    c_in: x_shape.dim(1).unwrap_static() as u32,
                    d: x_shape.dim(2).unwrap_static() as u32,
                    h: x_shape.dim(3).unwrap_static() as u32,
                    w: x_shape.dim(4).unwrap_static() as u32,
                    c_out: dy_shape.dim(1).unwrap_static() as u32,
                    d_out: dy_shape.dim(2).unwrap_static() as u32,
                    h_out: dy_shape.dim(3).unwrap_static() as u32,
                    w_out: dy_shape.dim(4).unwrap_static() as u32,
                    kd: dw_shape.dim(2).unwrap_static() as u32,
                    kh: dw_shape.dim(3).unwrap_static() as u32,
                    kw: dw_shape.dim(4).unwrap_static() as u32,
                    sd: stride[0] as u32,
                    sh: stride[1] as u32,
                    sw: stride[2] as u32,
                    pd: padding[0] as u32,
                    ph: padding[1] as u32,
                    pw: padding[2] as u32,
                    dd: dilation[0] as u32,
                    dh: dilation[1] as u32,
                    dw: dilation[2] as u32,
                    groups: (*groups).max(1) as u32,
                    x_off,
                    dy_off,
                    dw_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::Conv3dBackwardWeight { params: p });
                let sk = conv3d_backward_weight_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<Conv3dBwdWeightParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::GroupNormBackwardInput { num_groups, eps }
            | Op::GroupNormBackwardGamma { num_groups, eps }
            | Op::GroupNormBackwardBeta { num_groups, eps } => {
                let x_id = node.inputs[0];
                let x_shape = &graph.node(x_id).shape;
                let n = x_shape.dim(0).unwrap_static() as u32;
                let c = x_shape.dim(1).unwrap_static() as u32;
                let h = x_shape.dim(2).unwrap_static() as u32;
                let w = x_shape.dim(3).unwrap_static() as u32;
                let is_input = matches!(node.op, Op::GroupNormBackwardInput { .. });
                let is_gamma = matches!(node.op, Op::GroupNormBackwardGamma { .. });
                let win: Vec<NodeId> = if is_input {
                    vec![
                        node.id,
                        x_id,
                        node.inputs[1],
                        node.inputs[2],
                        node.inputs[3],
                    ]
                } else {
                    // gamma/beta: [x, dy]
                    vec![node.id, x_id, node.inputs[1]]
                };
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let (gamma_off, dy_off) = if is_input {
                    let g_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.inputs[1],
                        &mut base,
                        &mut size,
                    );
                    let _beta_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.inputs[2],
                        &mut base,
                        &mut size,
                    );
                    let dy = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.inputs[3],
                        &mut base,
                        &mut size,
                    );
                    (g_off, dy)
                } else {
                    let dy = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.inputs[1],
                        &mut base,
                        &mut size,
                    );
                    (0u32, dy)
                };
                let p = GroupNormBwdParams {
                    n,
                    c,
                    h,
                    w,
                    num_groups: (*num_groups).max(1) as u32,
                    eps_bits: eps.to_bits(),
                    x_off,
                    gamma_off,
                    dy_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _p0: 0,
                    _p1: 0,
                };
                let (sk, step) = if is_input {
                    (
                        group_norm_backward_input_kernel(&dev.device),
                        Step::GroupNormBackwardInput { params: p },
                    )
                } else if is_gamma {
                    (
                        group_norm_backward_gamma_kernel(&dev.device),
                        Step::GroupNormBackwardGamma { params: p },
                    )
                } else {
                    (
                        group_norm_backward_beta_kernel(&dev.device),
                        Step::GroupNormBackwardBeta { params: p },
                    )
                };
                schedule.push(step);
                let u = emit_uniform(std::mem::size_of::<GroupNormBwdParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            // ── Native BatchNormInference (`batch_norm_inference.wgsl`) ──
            //
            // wgpu was the only f32-uniform arena backend host-routing these,
            // which costs a readback per BN layer. Operand orders match the
            // Vulkan lowering; the arithmetic matches rlx-cpu (correctly-rounded
            // `1/sqrt`, not `inverseSqrt`).
            Op::BatchNormInference { eps } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let x = node.inputs[0];
                let xd = node.shape.dims();
                let channels = xd.last().map_or(1, |d| d.unwrap_static() as u32);
                let n: u32 = xd.iter().map(|d| d.unwrap_static() as u32).product();
                let p = BatchNormInferenceParams {
                    n,
                    count: n / channels.max(1),
                    channels,
                    eps: *eps,
                    src_off: (arena.offset(x) / 4) as u32,
                    gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    beta_off: (arena.offset(node.inputs[2]) / 4) as u32,
                    mean_off: (arena.offset(node.inputs[3]) / 4) as u32,
                    var_off: (arena.offset(node.inputs[4]) / 4) as u32,
                    dy_off: 0,
                    dst_off: (arena.offset(node.id) / 4) as u32,
                    _pad0: 0,
                };
                schedule.push(Step::BatchNormInference {
                    params: p,
                    kernel: BatchNormKernel::Forward,
                    threads: n,
                });
                let sk = batch_norm_inference_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<BatchNormInferenceParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::BatchNormInferenceBackwardInput { eps }
                if arena_binds_whole_at_zero(&dev.device, &arena) =>
            {
                let dy = node.inputs[4];
                let xd = graph.node(dy).shape.dims();
                let channels = xd.last().map_or(1, |d| d.unwrap_static() as u32);
                let n: u32 = xd.iter().map(|d| d.unwrap_static() as u32).product();
                let p = BatchNormInferenceParams {
                    n,
                    count: n / channels.max(1),
                    channels,
                    eps: *eps,
                    src_off: 0,
                    gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    beta_off: 0,
                    mean_off: 0,
                    var_off: (arena.offset(node.inputs[3]) / 4) as u32,
                    dy_off: (arena.offset(dy) / 4) as u32,
                    dst_off: (arena.offset(node.id) / 4) as u32,
                    _pad0: 0,
                };
                schedule.push(Step::BatchNormInference {
                    params: p,
                    kernel: BatchNormKernel::BwdInput,
                    threads: n,
                });
                let sk = batch_norm_inference_bwd_input_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<BatchNormInferenceParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::BatchNormInferenceBackwardGamma { eps }
                if arena_binds_whole_at_zero(&dev.device, &arena) =>
            {
                let x = node.inputs[0];
                let xd = graph.node(x).shape.dims();
                let channels = xd.last().map_or(1, |d| d.unwrap_static() as u32);
                let n: u32 = xd.iter().map(|d| d.unwrap_static() as u32).product();
                let p = BatchNormInferenceParams {
                    n,
                    count: n / channels.max(1),
                    channels,
                    eps: *eps,
                    src_off: (arena.offset(x) / 4) as u32,
                    gamma_off: 0,
                    beta_off: 0,
                    mean_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    var_off: (arena.offset(node.inputs[2]) / 4) as u32,
                    dy_off: (arena.offset(node.inputs[3]) / 4) as u32,
                    dst_off: (arena.offset(node.id) / 4) as u32,
                    _pad0: 0,
                };
                schedule.push(Step::BatchNormInference {
                    params: p,
                    kernel: BatchNormKernel::BwdGamma,
                    threads: channels,
                });
                let sk = batch_norm_inference_bwd_gamma_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<BatchNormInferenceParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::BatchNormInferenceBackwardBeta
                if arena_binds_whole_at_zero(&dev.device, &arena) =>
            {
                let dy = node.inputs[0];
                let xd = graph.node(dy).shape.dims();
                let channels = xd.last().map_or(1, |d| d.unwrap_static() as u32);
                let n: u32 = xd.iter().map(|d| d.unwrap_static() as u32).product();
                let p = BatchNormInferenceParams {
                    n,
                    count: n / channels.max(1),
                    channels,
                    eps: 0.0,
                    src_off: 0,
                    gamma_off: 0,
                    beta_off: 0,
                    mean_off: 0,
                    var_off: 0,
                    dy_off: (arena.offset(dy) / 4) as u32,
                    dst_off: (arena.offset(node.id) / 4) as u32,
                    _pad0: 0,
                };
                schedule.push(Step::BatchNormInference {
                    params: p,
                    kernel: BatchNormKernel::BwdBeta,
                    threads: channels,
                });
                let sk = batch_norm_inference_bwd_beta_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<BatchNormInferenceParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            // ── Native INT8 QAT path (`quant_i8.wgsl` / `q_matmul.wgsl`) ──
            //
            // Ported from the Vulkan kernels, with their two bugs already fixed:
            // the affine table is `vec4`-strided (a scalar array in a WGSL
            // uniform has a 16-byte stride) and the packed-i8 writers own a
            // whole word (WGSL has no byte store).
            Op::Quantize {
                axis,
                scales,
                zero_points,
            }
            | Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let is_quant = matches!(node.op, Op::Quantize { .. });
                let xd = node.shape.dims();
                let dims_usize: Vec<usize> = xd.iter().map(|d| d.unwrap_static()).collect();
                let n = dims_usize.iter().product::<usize>();
                let (chan_dim, inner) = match axis {
                    None => (1usize, n.max(1)),
                    Some(d) => (
                        dims_usize[*d],
                        dims_usize[*d + 1..].iter().product::<usize>().max(1),
                    ),
                };
                // Two reasons to keep the host route: an affine table too wide
                // for the uniform, and an arena that cannot be bound whole at
                // base 0 (these kernels carry ABSOLUTE offsets, which a rebased
                // window would silently misdirect).
                if chan_dim > QUANT_I8_MAX_CHAN || !arena_binds_whole_at_zero(&dev.device, &arena) {
                    schedule.push(host_op_step(&graph, node, &arena));
                } else {
                    let x_id = node.inputs[0];
                    let mut affine = [0u32; 24];
                    for c in 0..chan_dim.min(QUANT_I8_MAX_CHAN) {
                        affine[2 * c] = scales[c].to_bits();
                        affine[2 * c + 1] = zero_points[c] as u32;
                    }
                    // Codes are byte offsets; f32 tensors are element offsets.
                    let (a_off, out_off) = if is_quant {
                        (
                            (arena.offset(x_id) / 4) as u32,
                            arena.offset(node.id) as u32,
                        )
                    } else {
                        (
                            arena.offset(x_id) as u32,
                            (arena.offset(node.id) / 4) as u32,
                        )
                    };
                    let p = QuantI8Params {
                        n: n as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        a_off,
                        out_off,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        affine,
                    };
                    // Quantize writes packed bytes: one thread per output WORD.
                    let (kernel, threads) = if is_quant {
                        (QuantI8Kernel::Quantize, (n as u32).div_ceil(4))
                    } else {
                        (QuantI8Kernel::Dequantize, n as u32)
                    };
                    schedule.push(Step::QuantI8 {
                        params: p,
                        kernel,
                        threads,
                    });
                    let sk = if is_quant {
                        quantize_i8_kernel(&dev.device)
                    } else {
                        dequantize_i8_kernel(&dev.device)
                    };
                    let u = emit_uniform(std::mem::size_of::<QuantI8Params>());
                    let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
            }
            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let x_id = node.inputs[0];
                let w_id = node.inputs[1];
                let bias_id = node.inputs[2];
                let xd = graph.node(x_id).shape.dims();
                let od = node.shape.dims();
                let m = xd[0].unwrap_static() as u32;
                let k = xd[1].unwrap_static() as u32;
                let n = od[od.len() - 1].unwrap_static() as u32;
                let p = QMatMulParams {
                    m,
                    k,
                    n,
                    x_off: arena.offset(x_id) as u32,
                    w_off: arena.offset(w_id) as u32,
                    out_off: arena.offset(node.id) as u32,
                    // I32 keeps a full f32-sized slot on this arena and holds
                    // the integer as an f32 value, so this is an element offset.
                    bias_off: (arena.offset(bias_id) / 4) as u32,
                    x_zp: *x_zp,
                    w_zp: *w_zp,
                    out_zp: *out_zp,
                    mult: *mult,
                    _pad0: 0,
                };
                schedule.push(Step::QMatMul {
                    params: p,
                    // One thread per output WORD (four packed i8 results).
                    threads: (m * n).div_ceil(4),
                });
                let sk = q_matmul_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<QMatMulParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            Op::QConv2d {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let x_id = node.inputs[0];
                let w_id = node.inputs[1];
                let bias_id = node.inputs[2];
                let ind = graph.node(x_id).shape.dims();
                let outd = node.shape.dims();
                let d = |s: &[rlx_ir::Dim], i: usize| s[i].unwrap_static() as u32;
                let total = d(outd, 0) * d(outd, 1) * d(outd, 2) * d(outd, 3);
                let p = QConv2dParams {
                    batch: d(ind, 0),
                    c_in: d(ind, 1),
                    c_out: d(outd, 1),
                    h: d(ind, 2),
                    w: d(ind, 3),
                    h_out: d(outd, 2),
                    w_out: d(outd, 3),
                    kh: kernel_size[0] as u32,
                    kw: kernel_size[1] as u32,
                    sh: stride.first().copied().unwrap_or(1) as u32,
                    sw: stride.get(1).copied().unwrap_or(1) as u32,
                    ph: padding.first().copied().unwrap_or(0) as u32,
                    pw: padding.get(1).copied().unwrap_or(0) as u32,
                    dh: dilation.first().copied().unwrap_or(1) as u32,
                    dw: dilation.get(1).copied().unwrap_or(1) as u32,
                    groups: *groups as u32,
                    x_off: arena.offset(x_id) as u32,
                    w_off: arena.offset(w_id) as u32,
                    out_off: arena.offset(node.id) as u32,
                    bias_off: (arena.offset(bias_id) / 4) as u32,
                    x_zp: *x_zp,
                    w_zp: *w_zp,
                    out_zp: *out_zp,
                    mult: *mult,
                };
                schedule.push(Step::QConv2d {
                    params: p,
                    // One thread per output WORD (four packed i8 results).
                    threads: total.div_ceil(4),
                });
                let sk = q_conv2d_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<QConv2dParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            // ── Native general low-precision path (`scaled_lowp.wgsl`) ──
            //
            // These four used to fall through to the generic host route below,
            // even though the *harder* grouped (MoE) variant already had a
            // native decode kernel. That variant is MXFP4-only so none of it was
            // reusable — the dense ops span every `ScaledFormat`, which is why
            // this is a full codec port rather than a rewire.
            //
            // Absolute arena offsets + `bind_op_output_window`, matching the
            // ScaledGroupedMatMul arm below rather than the multi-operand
            // windowing the elementwise ops use.
            Op::ScaledQuantScale {
                format,
                scale_layout,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let x_id = node.inputs[0];
                let xs = graph.node(x_id).shape.dims();
                let cols = xs[xs.len() - 1].unwrap_static() as u32;
                let rows = graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                let (scale_mode, block) = scale_layout.mode_block();
                let nblk = if scale_mode == 0 {
                    1
                } else {
                    cols.div_ceil(block.max(1))
                };
                // Per-tensor writes one f32 from thread 0; block modes write one
                // scale BYTE per block, four to a thread (WGSL has no byte store).
                let threads = if scale_mode == 0 {
                    1
                } else {
                    (rows * nblk).div_ceil(4)
                };
                let p = ScaledLowpParams {
                    a_byte_off: (arena.offset(x_id) / 4) as u32,
                    b_byte_off: 0,
                    a_scale_byte_off: arena.offset(node.id) as u32,
                    b_scale_byte_off: 0,
                    out_off: 0,
                    bias_off: 0,
                    m: 0,
                    k: 0,
                    n: 0,
                    rows,
                    cols,
                    a_fmt: format.kernel_id(),
                    b_fmt: 0,
                    scale_mode,
                    block,
                    has_bias: 0,
                };
                schedule.push(Step::ScaledLowp {
                    params: p,
                    kernel: ScaledLowpKernel::QuantScale,
                    threads,
                });
                let sk = scaled_quant_scale_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ScaledLowpParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ScaledQuantize {
                format,
                scale_layout,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let x_id = node.inputs[0];
                let scale_id = node.inputs[1];
                let xs = graph.node(x_id).shape.dims();
                let cols = xs[xs.len() - 1].unwrap_static() as u32;
                let rows = graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                let (scale_mode, block) = scale_layout.mode_block();
                let p = ScaledLowpParams {
                    a_byte_off: (arena.offset(x_id) / 4) as u32,
                    b_byte_off: 0,
                    a_scale_byte_off: arena.offset(scale_id) as u32,
                    b_scale_byte_off: 0,
                    out_off: arena.offset(node.id) as u32,
                    bias_off: 0,
                    m: 0,
                    k: 0,
                    n: 0,
                    rows,
                    cols,
                    a_fmt: format.kernel_id(),
                    b_fmt: 0,
                    scale_mode,
                    block,
                    has_bias: 0,
                };
                schedule.push(Step::ScaledLowp {
                    params: p,
                    kernel: ScaledLowpKernel::Quantize,
                    // One thread per output WORD (four codes).
                    threads: (rows * cols).div_ceil(4),
                });
                let sk = scaled_quantize_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ScaledLowpParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ScaledDequantize {
                format,
                scale_layout,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                let codes_id = node.inputs[0];
                let scale_id = node.inputs[1];
                let cs = graph.node(codes_id).shape.dims();
                let cols = cs[cs.len() - 1].unwrap_static() as u32;
                let rows = graph.node(codes_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                let (scale_mode, block) = scale_layout.mode_block();
                let p = ScaledLowpParams {
                    a_byte_off: arena.offset(codes_id) as u32,
                    b_byte_off: 0,
                    a_scale_byte_off: arena.offset(scale_id) as u32,
                    b_scale_byte_off: 0,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    bias_off: 0,
                    m: 0,
                    k: 0,
                    n: 0,
                    rows,
                    cols,
                    a_fmt: format.kernel_id(),
                    b_fmt: 0,
                    scale_mode,
                    block,
                    has_bias: 0,
                };
                schedule.push(Step::ScaledLowp {
                    params: p,
                    kernel: ScaledLowpKernel::Dequantize,
                    threads: rows * cols,
                });
                let sk = scaled_dequantize_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ScaledLowpParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ScaledMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                has_bias,
            } if arena_binds_whole_at_zero(&dev.device, &arena) => {
                // TN: lhs [m,k], rhs [n,k] (K-last on both), out [m,n].
                let lhs_id = node.inputs[0];
                let rhs_id = node.inputs[1];
                let ls_id = node.inputs[2];
                let rs_id = node.inputs[3];
                let ld = graph.node(lhs_id).shape.dims();
                let rd = graph.node(rhs_id).shape.dims();
                let m = ld[ld.len() - 2].unwrap_static() as u32;
                let k = ld[ld.len() - 1].unwrap_static() as u32;
                let n = rd[rd.len() - 2].unwrap_static() as u32;
                let (scale_mode, block) = scale_layout.mode_block();
                let bias_off = if *has_bias {
                    (arena.offset(node.inputs[4]) / 4) as u32
                } else {
                    0
                };
                let p = ScaledLowpParams {
                    a_byte_off: arena.offset(lhs_id) as u32,
                    b_byte_off: arena.offset(rhs_id) as u32,
                    a_scale_byte_off: arena.offset(ls_id) as u32,
                    b_scale_byte_off: arena.offset(rs_id) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    bias_off,
                    m,
                    k,
                    n,
                    rows: 0,
                    cols: 0,
                    a_fmt: lhs_format.kernel_id(),
                    b_fmt: rhs_format.kernel_id(),
                    scale_mode,
                    block,
                    has_bias: u32::from(*has_bias),
                };
                schedule.push(Step::ScaledLowp {
                    params: p,
                    kernel: ScaledLowpKernel::MatmulDecode,
                    // 2-D 16x16 tiles; `threads` is unused for this variant.
                    threads: 0,
                });
                let sk = scaled_matmul_decode_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ScaledLowpParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }

            // The op list lives in `crate::supported_ops::routes_to_cpu_host`
            // so the claim and the route can be compared from outside this
            // crate without a GPU — see that function, and
            // `rlx-runtime/tests/host_fallback_never_nops.rs`.
            other if crate::supported_ops::routes_to_cpu_host(other) => {
                schedule.push(host_op_step(&graph, node, &arena));
            }
            Op::FftButterflyStage { stage, n_fft } => {
                let state = node.inputs[0];
                let state_shape = &graph.node(state).shape;
                assert_eq!(
                    state_shape.dtype(),
                    rlx_ir::DType::F32,
                    "rlx-wgpu Op::FftButterflyStage requires F32 state"
                );
                let gate = node.inputs[1];
                let rev = node.inputs[2];
                let tw_re = node.inputs[3];
                let tw_im = node.inputs[4];
                let win_ids = [node.id, state, gate, rev, tw_re, tw_im];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let state_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    state,
                    &mut base,
                    &mut size,
                );
                let gate_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    gate,
                    &mut base,
                    &mut size,
                );
                let rev_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    rev,
                    &mut base,
                    &mut size,
                );
                let tw_re_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    tw_re,
                    &mut base,
                    &mut size,
                );
                let tw_im_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    tw_im,
                    &mut base,
                    &mut size,
                );
                let out_raw = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let state_off = if arena_tensor_in_window(&arena, state, base, size) {
                    arena_local_off_f32(&arena, state, base)
                } else {
                    state_raw
                };
                let gate_off = if arena_tensor_in_window(&arena, gate, base, size) {
                    arena_local_off_f32(&arena, gate, base)
                } else {
                    gate_raw
                };
                let rev_off = if arena_tensor_in_window(&arena, rev, base, size) {
                    arena_local_off_f32(&arena, rev, base)
                } else {
                    rev_raw
                };
                let tw_re_off = if arena_tensor_in_window(&arena, tw_re, base, size) {
                    arena_local_off_f32(&arena, tw_re, base)
                } else {
                    tw_re_raw
                };
                let tw_im_off = if arena_tensor_in_window(&arena, tw_im, base, size) {
                    arena_local_off_f32(&arena, tw_im, base)
                } else {
                    tw_im_raw
                };
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_raw
                };
                let batch = state_shape.dim(0).unwrap_static() as u32;
                let half = *n_fft / 2;
                let p = FftButterflyStageParams {
                    batch,
                    n_fft: *n_fft,
                    stage: *stage,
                    half,
                    state_off,
                    out_off,
                    gate_off,
                    rev_off,
                    tw_re_off,
                    tw_im_off,
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::FftButterflyStage { params: p });
                let u = emit_uniform(std::mem::size_of::<FftButterflyStageParams>());
                let bg = bind_arena_window(
                    &dev.device,
                    fft_butterfly_stage_kernel(&dev.device),
                    &arena,
                    base,
                    size,
                    &u,
                );
                uniforms.push(u);
                bind_groups.push(bg);
            }
            // C64 Wirtinger surface — native WGSL (mirrors CUDA
            // `complex_wirtinger.cu`). Interleaved [re, im] pairs; `n` is the
            // complex-element count (ComplexNormSq output is real F32).
            Op::ComplexNormSq => {
                let src = node.inputs[0];
                if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                    panic!(
                        "rlx-wgpu ComplexNormSq: expected C64 input, got {:?}",
                        graph.node(src).shape.dtype()
                    );
                }
                let win_ids = [node.id, src];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let src_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    src,
                    &mut base,
                    &mut size,
                );
                let dst_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let src_off = if arena_tensor_in_window(&arena, src, base, size) {
                    arena_local_off_f32(&arena, src, base)
                } else {
                    src_off
                };
                let dst_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    dst_off
                };
                let p = ComplexWirtingerParams {
                    n: elems,
                    a_off: src_off,
                    b_off: 0,
                    c_off: dst_off,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::ComplexNormSq { params: p });
                let u = emit_uniform(std::mem::size_of::<ComplexWirtingerParams>());
                let bg = bind_arena_window(
                    &dev.device,
                    complex_norm_sq_kernel(&dev.device),
                    &arena,
                    base,
                    size,
                    &u,
                );
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ComplexNormSqBackward => {
                let z = node.inputs[0];
                let g = node.inputs[1];
                if graph.node(z).shape.dtype() != rlx_ir::DType::C64 {
                    panic!(
                        "rlx-wgpu ComplexNormSqBackward: expected C64 z, got {:?}",
                        graph.node(z).shape.dtype()
                    );
                }
                let win_ids = [node.id, z, g];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let z_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    z,
                    &mut base,
                    &mut size,
                );
                let g_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    g,
                    &mut base,
                    &mut size,
                );
                let dz_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let z_off = if arena_tensor_in_window(&arena, z, base, size) {
                    arena_local_off_f32(&arena, z, base)
                } else {
                    z_off
                };
                let g_off = if arena_tensor_in_window(&arena, g, base, size) {
                    arena_local_off_f32(&arena, g, base)
                } else {
                    g_off
                };
                let dz_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    dz_off
                };
                let p = ComplexWirtingerParams {
                    n: elems,
                    a_off: z_off,
                    b_off: g_off,
                    c_off: dz_off,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::ComplexNormSqBackward { params: p });
                let u = emit_uniform(std::mem::size_of::<ComplexWirtingerParams>());
                let bg = bind_arena_window(
                    &dev.device,
                    complex_norm_sq_backward_kernel(&dev.device),
                    &arena,
                    base,
                    size,
                    &u,
                );
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::Conjugate => {
                let src = node.inputs[0];
                if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                    panic!(
                        "rlx-wgpu Conjugate: expected C64 input, got {:?}",
                        graph.node(src).shape.dtype()
                    );
                }
                let win_ids = [node.id, src];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let src_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    src,
                    &mut base,
                    &mut size,
                );
                let dst_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    node.id,
                    &mut base,
                    &mut size,
                );
                let src_off = if arena_tensor_in_window(&arena, src, base, size) {
                    arena_local_off_f32(&arena, src, base)
                } else {
                    src_off
                };
                let dst_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    dst_off
                };
                let p = ComplexWirtingerParams {
                    n: elems,
                    a_off: src_off,
                    b_off: 0,
                    c_off: dst_off,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                schedule.push(Step::ConjugateC64 { params: p });
                let u = emit_uniform(std::mem::size_of::<ComplexWirtingerParams>());
                let bg = bind_arena_window(
                    &dev.device,
                    conjugate_c64_kernel(&dev.device),
                    &arena,
                    base,
                    size,
                    &u,
                );
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::FakeQuantize {
                bits,
                axis,
                ste: _,
                scale_mode,
            } => {
                use rlx_ir::op::ScaleMode;
                let q_max = match *bits {
                    8 => 127.0f32,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("rlx-wgpu FakeQuantize: unsupported bits {n}"),
                };
                // EMA needs mutable running-scale state — keep HostOp.
                if matches!(scale_mode, ScaleMode::EMA { .. }) {
                    schedule.push(host_op_step(&graph, node, &arena));
                    continue;
                }
                let in_id = node.inputs[0];
                // Mirror `rlx_cpu::thunk::ops::quant::quant_layout`.
                let (chan_dim, inner) = match *axis {
                    None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                    Some(d) => {
                        let chan_dim = node.shape.dim(d).unwrap_static();
                        let inner: usize = (d + 1..node.shape.rank())
                            .map(|i| node.shape.dim(i).unwrap_static())
                            .product::<usize>()
                            .max(1);
                        (chan_dim, inner)
                    }
                };
                let n = node.shape.num_elements().unwrap() as u32;
                let chan_dim = chan_dim as u32;
                let inner = inner as u32;

                let mut win = vec![node.id, in_id];
                if matches!(scale_mode, ScaleMode::Fixed) {
                    win.push(node.inputs[1]);
                }
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    in_id,
                    &mut base,
                    &mut size,
                );
                let scale_off = match scale_mode {
                    ScaleMode::Fixed => arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.inputs[1],
                        &mut base,
                        &mut size,
                    ),
                    ScaleMode::PerBatch | ScaleMode::EMA { .. } => 0,
                };
                let p = FakeQuantizeParams {
                    n,
                    chan_dim,
                    inner,
                    q_max,
                    in_off,
                    scale_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    _pad: 0,
                };
                let (sk, step) = match scale_mode {
                    ScaleMode::Fixed => (
                        fake_quantize_fixed_kernel(&dev.device),
                        Step::FakeQuantizeFixed { params: p },
                    ),
                    ScaleMode::PerBatch => (
                        fake_quantize_perbatch_kernel(&dev.device),
                        Step::FakeQuantizePerBatch { params: p },
                    ),
                    ScaleMode::EMA { .. } => unreachable!(),
                };
                schedule.push(step);
                let u = emit_uniform(std::mem::size_of::<FakeQuantizeParams>());
                let bg = bind_arena_window(&dev.device, sk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            // Core Riemannian / SPD-manifold ops — no WGSL eigen kernel, so
            // they host-fallback to `rlx_cpu::spd` (F64) via a readback of
            // each operand's arena span. See `crate::spd_host`.
            _ if crate::spd_host::is_spd_host(&node.op) => {
                let inputs: Vec<crate::spd_host::SpdInput> = node
                    .inputs
                    .iter()
                    .map(|&id| crate::spd_host::SpdInput {
                        shape: graph.node(id).shape.clone(),
                        byte_off: arena.offset(id),
                    })
                    .collect();
                schedule.push(Step::SpdHost {
                    op: node.op.clone(),
                    inputs,
                    out_shape: node.shape.clone(),
                    out_byte_off: arena.offset(node.id),
                });
            }
            Op::Scan { .. } => {
                // Host fallback (no unified memory): compile the body once,
                // loop it on the CPU against a readback of the arena span.
                schedule.push(Step::ScanHost {
                    desc: rlx_cpu::rlx_scan_host_desc!(graph, node, |id| arena.offset(id)),
                });
            }
            Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => {
                schedule.push(host_op_step(&graph, node, &arena));
            }
            Op::ScatterNd { .. }
            | Op::ScatterElements { .. }
            | Op::GatherNd { .. }
            | Op::GatherElements { .. } => {
                // wgpu arena is f32-uniform: I64 index constants are widened
                // to f32 on upload. Force the f32 index reader — `indices_i64=1`
                // would re-read float bits as i64 (same class of bug as the
                // pre-fix MLX host ScatterNd path that drifted F5 DiT).
                let thunk = rlx_cpu::rlx_indexing_thunk!(graph, node, |id| arena.offset(id))
                    .force_indices_f32();

                let is_scatter =
                    matches!(node.op, Op::ScatterNd { .. } | Op::ScatterElements { .. });
                let data_id = node.inputs[0];
                let idx_id = node.inputs[1];
                // Gathers have no `updates`; reuse `data` so the bind window
                // and the unused offset both stay in range.
                let upd_id = if is_scatter { node.inputs[2] } else { data_id };

                let planned = if rlx_ir::env::flag("RLX_WGPU_INDEXING_HOST") {
                    None
                } else {
                    rlx_gpu_host::indexing_plan::plan_indexing(&thunk)
                };
                // Core WGSL has no f32 atomics, so the WGSL scatters are
                // overwrite-only; an accumulating scatter keeps the host route.
                // CUDA and ROCm take it on-device with `atomicAdd`.
                let planned = planned.filter(|l| {
                    !matches!(
                        l.kind,
                        IndexKind::ScatterElements {
                            reduction: IndexReduction::Add,
                            ..
                        } | IndexKind::ScatterNd {
                            reduction: IndexReduction::Add,
                            ..
                        }
                    )
                });

                let Some(l) = planned else {
                    schedule.push(Step::CpuIndexing { thunk });
                    continue;
                };

                // Bind window over every operand: an absolute offset with an
                // output-only binding is how FusedResidualLN wrote all zeros on
                // F5-scale arenas. All four must land inside one window.
                let win_ids = [node.id, data_id, idx_id, upd_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let mut local_off = |id, base: &mut u64, size: &mut u64| -> u32 {
                    let off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        id,
                        base,
                        size,
                    );
                    // Staging an operand can widen `base` to the whole stripe;
                    // recompute against the final window (as Expand does).
                    if arena_tensor_in_window(&arena, id, *base, *size) {
                        arena_local_off_f32(&arena, id, *base)
                    } else {
                        off
                    }
                };
                let data_off = local_off(data_id, &mut base, &mut size);
                let idx_off = local_off(idx_id, &mut base, &mut size);
                let upd_off = local_off(upd_id, &mut base, &mut size);
                let dst_off = local_off(node.id, &mut base, &mut size);
                // The window can move under a later operand's staging, so the
                // earlier locals are only final once every operand is placed.
                let data_off = if arena_tensor_in_window(&arena, data_id, base, size) {
                    arena_local_off_f32(&arena, data_id, base)
                } else {
                    data_off
                };
                let idx_off = if arena_tensor_in_window(&arena, idx_id, base, size) {
                    arena_local_off_f32(&arena, idx_id, base)
                } else {
                    idx_off
                };
                let upd_off = if arena_tensor_in_window(&arena, upd_id, base, size) {
                    arena_local_off_f32(&arena, upd_id, base)
                } else {
                    upd_off
                };

                let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu indexing_nd meta"),
                    size: (l.meta.len() * 4).max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                dev.queue
                    .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&l.meta));
                let meta_idx = meta_buffers.len();
                meta_buffers.push(meta_buf);

                let base_params = IndexingNdParams {
                    n: l.n,
                    data_off,
                    idx_off,
                    upd_off,
                    dst_off,
                    dst_len: l.dst_len,
                    rank: 0,
                    axis: 0,
                    axis_dim: 0,
                    k: 0,
                    slice: 0,
                    tuples_per_batch: 0,
                    batch_stride: 0,
                    reduction: 0,
                    src_len: 0,
                    flags: 0,
                };

                // One dispatch per step, so a scatter emits the prologue first.
                let mut emit = |params: IndexingNdParams, which: IndexingNdKernel| {
                    let k = match which {
                        IndexingNdKernel::GatherNd => gather_nd_kernel(&dev.device),
                        IndexingNdKernel::GatherElements => gather_elements_kernel(&dev.device),
                        IndexingNdKernel::ScatterElements => scatter_elements_kernel(&dev.device),
                        IndexingNdKernel::ScatterNd => scatter_nd_reduce_kernel(&dev.device),
                        IndexingNdKernel::CopySanitize => copy_sanitize_kernel(&dev.device),
                    };
                    let u = emit_uniform(std::mem::size_of::<IndexingNdParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu indexing_nd bg"),
                        layout: &k.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: arena_bind_buf(&arena, base).0,
                                    offset: arena_bind_buf(&arena, base).1,
                                    size: {
                                        let (b, lg) = arena_bind_buf(&arena, base);
                                        aligned_bind_size(size, lg, b.size())
                                    },
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    schedule.push(Step::IndexingNd {
                        params,
                        kernel: which,
                        meta_idx,
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                };

                if let Some(pro) = l.prologue
                    && l.dst_len > 0
                {
                    let mut p = base_params;
                    p.n = l.dst_len;
                    p.src_len = pro.src_len;
                    p.flags = (pro.do_copy & 1) | ((pro.do_sanitize & 1) << 1);
                    emit(p, IndexingNdKernel::CopySanitize);
                }

                let mut p = base_params;
                let which = match l.kind {
                    IndexKind::GatherNd {
                        k,
                        slice,
                        tuples_per_batch,
                        batch_stride,
                    } => {
                        p.k = k;
                        p.slice = slice;
                        p.tuples_per_batch = tuples_per_batch;
                        p.batch_stride = batch_stride;
                        IndexingNdKernel::GatherNd
                    }
                    IndexKind::GatherElements {
                        rank,
                        axis,
                        axis_dim,
                        data_len,
                    } => {
                        p.rank = rank;
                        p.axis = axis;
                        p.axis_dim = axis_dim;
                        // This kernel's bounds guard is against `data`, not `dst`.
                        p.dst_len = data_len;
                        IndexingNdKernel::GatherElements
                    }
                    IndexKind::ScatterElements { rank, axis, .. } => {
                        p.rank = rank;
                        p.axis = axis;
                        IndexingNdKernel::ScatterElements
                    }
                    IndexKind::ScatterNd { k, slice, .. } => {
                        p.k = k;
                        p.slice = slice;
                        IndexingNdKernel::ScatterNd
                    }
                };
                emit(p, which);
            }
            Op::Custom { name, attrs, .. } => match name.as_str() {
                "llada2.group_limited_gate" => {
                    let sig_id = node.inputs[0];
                    let route_id = node.inputs[1];
                    let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                    let mut attr_buf = [0u8; 20];
                    let n = attrs.len().min(20);
                    attr_buf[..n].copy_from_slice(&attrs[..n]);
                    schedule.push(Step::Llada2GroupLimitedGate {
                        sig_byte_off: arena.offset(sig_id) as u32,
                        route_byte_off: arena.offset(route_id) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                        n_elems,
                        attrs: attr_buf,
                    });
                }
                "umap.knn" => {
                    let pw_id = node.inputs[0];
                    let pw_shape = graph.node(pw_id).shape.dims();
                    let n = pw_shape[0].unwrap_static() as u32;
                    let k = if attrs.len() >= 4 {
                        u32::from_le_bytes(attrs[..4].try_into().unwrap())
                    } else {
                        panic!("rlx-wgpu: umap.knn attrs missing k");
                    };
                    let pw_off = arena.offset(pw_id) as u32;
                    let out_off = arena.offset(node.id) as u32;
                    // `pw_off` is `arena.offset(pw_id) as u32`, so a pairwise matrix
                    // supplied as a Param and parked in the separate weight buffer
                    // loses the `WEIGHT_BUF_TAG` in the cast and is read as a
                    // plausible activation. `Step::UmapKnn` reads it by ABSOLUTE
                    // word offset and `Step::UmapKnnHost` by a bare byte offset, so
                    // neither sub-path can express a parked operand — refuse, like
                    // `Op::DequantMatMul`'s in-arm guard.
                    if crate::buffer::is_weight_off(arena.offset(pw_id)) {
                        panic!(
                            "rlx-wgpu: REFUSED (umap.knn parked operand): the pairwise matrix is \
                             parked in the weight buffer ({}), and both sub-paths address it by a \
                             bare offset that drops the weight tag — it would be read as a \
                             plausible activation. Avoid parking this operand, or move the graph to \
                             a backend that can.",
                            parked_operand_report(&arena)
                        );
                    }
                    // `Step::UmapKnn` reads `pw`/`out` by ABSOLUTE word offset
                    // through the output window, so it is only correct when the
                    // whole arena binds at base zero. The host step addresses the
                    // arena by byte offset and is window-agnostic — the same
                    // coarse condition the per-binding-ceiling rule asks, applied
                    // here because this is a `Op::Custom` sub-path the rule's
                    // family list does not cover.
                    let whole_at_zero = arena.binds_whole_at_base_zero(
                        dev.device.limits().max_storage_buffer_binding_size,
                    );
                    if n as usize >= crate::umap_knn_host::UMAP_KNN_GPU_MIN_N && whole_at_zero {
                        let p = UmapKnnParams {
                            n,
                            k,
                            pw_off: pw_off / 4,
                            out_off: out_off / 4,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::UmapKnn { params: p });
                        let uk = umap_knn_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<UmapKnnParams>());
                        let bg = bind_op_output_window(&dev.device, uk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        schedule.push(Step::UmapKnnHost {
                            pairwise_byte_off: pw_off,
                            out_byte_off: out_off,
                            n,
                            k,
                        });
                    }
                }
                "gdino.ms_deform_attn" => {
                    let in_offs: Vec<(u32, u32)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let bytes = graph.node(id).shape.num_elements().unwrap() * 4;
                            (arena.offset(id) as u32, bytes as u32)
                        })
                        .collect();
                    let out_bytes = (node.shape.num_elements().unwrap() * 4) as u32;
                    schedule.push(Step::MsDeformAttnHost {
                        in_offs,
                        out_byte_off: arena.offset(node.id) as u32,
                        out_bytes,
                        attrs: attrs.clone(),
                    });
                }
                n if crate::collective_host::COLLECTIVE_OPS.contains(&n) => {
                    // Host-delegate collective (all_reduce / all_gather /
                    // reduce_scatter / f / g): stage off-GPU and run the
                    // registered rlx-cpu collective kernel.
                    let in_id = node.inputs[0];
                    let in_bytes = (graph.node(in_id).shape.num_elements().unwrap() * 4) as u32;
                    let out_bytes = (node.shape.num_elements().unwrap() * 4) as u32;
                    schedule.push(Step::CollectiveHost {
                        name: n.to_string(),
                        in_byte_off: arena.offset(in_id) as u32,
                        in_bytes,
                        out_byte_off: arena.offset(node.id) as u32,
                        out_bytes,
                        attrs: attrs.clone(),
                    });
                }
                other
                    if crate::wgpu_gpu_custom::has_gpu_kernel(other)
                        && node.inputs.len() <= 8
                        && !std::iter::once(node.id)
                            .chain(node.inputs.iter().copied())
                            .any(|id| crate::buffer::is_weight_off(arena.offset(id))) =>
                {
                    // Raw-GPU custom op: dispatch a real WGSL kernel straight
                    // against the arena buffer (no host roundtrip). Bind a window
                    // covering all operands + a storage params buffer of
                    // window-relative element offsets, mirroring the umap.knn
                    // GPU precedent. Guarded to activation operands / ≤8 inputs;
                    // anything else falls through to the host arm below.
                    let gk = crate::wgpu_gpu_custom::lookup(other).unwrap();
                    let mut ids = vec![node.id];
                    ids.extend(node.inputs.iter().copied());
                    let (base, size) = arena_window_for_nodes(&dev.device, &arena, &ids);
                    let rel = |id: NodeId| ((arena.offset(id) as u64 - base) / 4) as u32;
                    let out_elems = node.shape.num_elements().unwrap() as u32;
                    // params = [out_off, out_len, n_inputs, _pad, (in_off,in_len)*]
                    let mut params: Vec<u32> =
                        vec![rel(node.id), out_elems, node.inputs.len() as u32, 0];
                    for &in_id in &node.inputs {
                        params.push(rel(in_id));
                        params.push(graph.node(in_id).shape.num_elements().unwrap() as u32);
                    }
                    let kernel = crate::wgpu_gpu_custom::get_or_build_pipeline(&dev.device, &*gk);
                    let pbuf = crate::wgpu_gpu_custom::make_params_buffer(&dev.device, &params);
                    let bg =
                        bind_arena_window(&dev.device, kernel.as_ref(), &arena, base, size, &pbuf);
                    let workgroups = gk.workgroups(out_elems);
                    schedule.push(Step::WgpuGpuKernel {
                        name: other.to_string(),
                        workgroups,
                    });
                    uniforms.push(pbuf);
                    bind_groups.push(bg);
                }
                other if crate::custom_host::has_host_kernel(other) => {
                    // Generic host-delegate: any onnx.* op with a registered
                    // rlx-cpu reference kernel but no wgpu shader. Stage the
                    // operands off-GPU (dtype-aware) and run the CPU reference.
                    let in_specs: Vec<(u32, rlx_ir::Shape)> = node
                        .inputs
                        .iter()
                        .map(|&id| (arena.offset(id) as u32, graph.node(id).shape.clone()))
                        .collect();
                    schedule.push(Step::CustomHost {
                        name: other.to_string(),
                        in_specs,
                        out_byte_off: arena.offset(node.id) as u32,
                        out_shape: node.shape.clone(),
                        attrs: attrs.clone(),
                    });
                }
                other => panic!("rlx-wgpu: unsupported Op::Custom('{other}')"),
            },
            Op::GroupedMatMul => {
                // Inputs: input [M, K], weight [E, K, N], expert_idx [M]
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let idx_id = node.inputs[2];
                // Checked, not trusted: `n` comes off the expert bank while the
                // output slot is sized from `node.shape`, so a bank still in
                // `[E, N, K]` order silently under- or over-writes it.
                let gd = rlx_ir::shape::grouped_matmul_dims(
                    &graph.node(in_id).shape,
                    &graph.node(w_id).shape,
                    Some(&node.shape),
                )
                .unwrap_or_else(|e| panic!("rlx-wgpu: node {:?}: {e}", node.id));
                let (m, k, n, ne) = (gd.m as u32, gd.k as u32, gd.n as u32, gd.num_experts as u32);
                let p = GroupedMatmulParams {
                    m,
                    k,
                    n,
                    num_experts: ne,
                    in_off: (arena.offset(in_id) / 4) as u32,
                    w_off: (arena.offset(w_id) / 4) as u32,
                    idx_off: (arena.offset(idx_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                };
                schedule.push(Step::GroupedMatmul { params: p });
                let gk = grouped_matmul_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<GroupedMatmulParams>());
                let bg = bind_op_output_window(&dev.device, gk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ScaledGroupedMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                has_bias,
            } => {
                // Native MXFP4 (E2M1) grouped decode-GEMM. The WGSL kernel
                // specializes the FP4 E2M1 codec (the MX default); other
                // ScaledFormats aren't lowered natively on wgpu.
                assert!(
                    *lhs_format == rlx_ir::ScaledFormat::F4E2M1
                        && *rhs_format == rlx_ir::ScaledFormat::F4E2M1,
                    "rlx-wgpu: ScaledGroupedMatMul native path is MXFP4 (F4E2M1) only, got \
                     {lhs_format:?}×{rhs_format:?}"
                );
                // input codes [M,K], weight codes [E,N,K], input_s, weight_s,
                // expert_idx [M], (bias [E,N]).
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let is_id = node.inputs[2];
                let ws_id = node.inputs[3];
                let idx_id = node.inputs[4];
                let in_dims = graph.node(in_id).shape.dims();
                let w_dims = graph.node(w_id).shape.dims();
                let m = in_dims[0].unwrap_static() as u32;
                let k = in_dims[1].unwrap_static() as u32;
                let ne = w_dims[0].unwrap_static() as u32;
                let n = w_dims[w_dims.len() - 2].unwrap_static() as u32;
                let (scale_mode, block) = scale_layout.mode_block();
                let bias_off = if *has_bias {
                    (arena.offset(node.inputs[5]) / 4) as u32
                } else {
                    0
                };
                let p = ScaledGroupedMatmulParams {
                    m,
                    k,
                    n,
                    num_experts: ne,
                    input_byte_off: arena.offset(in_id) as u32,
                    weight_byte_off: arena.offset(w_id) as u32,
                    input_scale_byte_off: arena.offset(is_id) as u32,
                    weight_scale_byte_off: arena.offset(ws_id) as u32,
                    idx_off: (arena.offset(idx_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    bias_off,
                    scale_mode,
                    block,
                    has_bias: u32::from(*has_bias),
                    _pad0: 0,
                    _pad1: 0,
                };
                schedule.push(Step::ScaledGroupedMatmul { params: p });
                let sk = scaled_grouped_matmul_decode_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<ScaledGroupedMatmulParams>());
                let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::DequantGroupedMatMul { scheme } => {
                let in_id = node.inputs[0];
                let w_id = node.inputs[1];
                let idx_id = node.inputs[2];
                let in_dims = graph.node(in_id).shape.dims();
                let out_dims = node.shape.dims();
                let m = in_dims[0].unwrap_static() as u32;
                let k = in_dims[1].unwrap_static() as u32;
                let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                let block_elems = scheme.gguf_block_size() as usize;
                let block_bytes = scheme.gguf_block_bytes() as usize;
                let slab_bytes = (k as usize * n as usize) / block_elems * block_bytes;
                let total_bytes = graph.node(w_id).shape.num_elements().unwrap();
                let ne = (total_bytes / slab_bytes.max(1)) as u32;
                schedule.push(Step::DequantGroupedMatmulGguf {
                    m,
                    k,
                    n,
                    num_experts: ne,
                    scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                    x_byte_off: arena.offset(in_id) as u64,
                    w_byte_off: arena.offset(w_id) as u64,
                    idx_byte_off: arena.offset(idx_id) as u64,
                    out_byte_off: arena.offset(node.id) as u64,
                });
                // Host step (builds its own bind groups): schedule-only, like
                // `GroupNorm`. The uniform/bind_group lanes are indexed by
                // GPU-step count (host steps skipped), so pushing here would
                // shift every later GPU op's lane. Do NOT push.
            }
            Op::DequantGroupedMatMulMlx { scheme } => {
                // 5 inputs: input, w_q, scales, biases, expert_idx.
                let in_id = node.inputs[0];
                let in_dims = graph.node(in_id).shape.dims();
                let out_dims = node.shape.dims();
                let m = in_dims[in_dims.len() - 2].unwrap_static() as u32;
                let k = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                let ne = graph.node(node.inputs[2]).shape.dims()[0].unwrap_static() as u32;
                schedule.push(Step::DequantGroupedMatmulMlxHost {
                    m,
                    k,
                    n,
                    num_experts: ne,
                    scheme: *scheme,
                    x_byte_off: arena.offset(in_id) as u64,
                    w_byte_off: arena.offset(node.inputs[1]) as u64,
                    scale_byte_off: arena.offset(node.inputs[2]) as u64,
                    zp_byte_off: arena.offset(node.inputs[3]) as u64,
                    idx_byte_off: arena.offset(node.inputs[4]) as u64,
                    out_byte_off: arena.offset(node.id) as u64,
                    scale_bf16: graph.node(node.inputs[2]).shape.dtype() == rlx_ir::DType::BF16,
                });
                // Host step: schedule-only (see the GGUF grouped note above).
            }
            Op::TopK { k } => {
                let in_id = node.inputs[0];
                let in_dims = graph.node(in_id).shape.dims();
                let inner = in_dims.last().unwrap().unwrap_static() as u32;
                let outer: u32 = in_dims[..in_dims.len() - 1]
                    .iter()
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let p = TopKParams {
                    outer,
                    inner,
                    k: *k as u32,
                    in_off: (arena.offset(in_id) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::TopK { params: p });
                let tk = topk_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<TopKParams>());
                let bg = bind_op_output_window(&dev.device, tk, &arena, node.id, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::ScatterAdd { axis } => {
                // Every backend kernel implements the axis-0 form only;
                // `rlx_fusion::LowerScatterAddAxis` rewrites any other axis to
                // transpose/scatter/transpose before lowering. Reaching here with
                // axis != 0 means that pass did not run, and scattering along axis
                // 0 anyway would silently produce the wrong tensor.
                assert_eq!(
                    *axis, 0,
                    "rlx-wgpu: ScatterAdd axis {{axis}} reached the backend; \
                 LowerScatterAddAxis must run first"
                );
                // Inputs: updates [num_updates, trailing], indices [num_updates].
                // Output: [out_dim, trailing]. Implemented as two phases:
                //   1. Zero `out_dim * trailing` slots.
                //   2. CAS-loop atomic-accumulate `num_updates * trailing` updates.
                let upd_id = node.inputs[0];
                let idx_id = node.inputs[1];
                let upd_dims = graph.node(upd_id).shape.dims();
                let out_dims = node.shape.dims();
                let num_updates = upd_dims[0].unwrap_static() as u32;
                let trailing: u32 = upd_dims
                    .iter()
                    .skip(1)
                    .map(|d| d.unwrap_static() as u32)
                    .product::<u32>()
                    .max(1);
                let out_dim = out_dims[0].unwrap_static() as u32;
                let out_total = out_dim * trailing;

                let common = ScatterAddParams {
                    op: 0,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    upd_off: (arena.offset(upd_id) / 4) as u32,
                    idx_off: (arena.offset(idx_id) / 4) as u32,
                    out_total,
                    num_updates,
                    trailing,
                    out_dim,
                };
                let sk = scatter_add_kernel(&dev.device);

                // Phase 0: zero.
                schedule.push(Step::ScatterAdd { params: common });
                let u0 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                let bg0 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u0);
                uniforms.push(u0);
                bind_groups.push(bg0);

                // Phase 1: accumulate.
                let mut acc = common;
                acc.op = 1;
                schedule.push(Step::ScatterAdd { params: acc });
                let u1 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                let bg1 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u1);
                uniforms.push(u1);
                bind_groups.push(bg1);
            }
            Op::FusedResidualLN { has_bias, eps } => {
                // Inputs: [x, residual, [bias], gamma, beta].
                let x_id = node.inputs[0];
                let r_id = node.inputs[1];
                let (bias_id, g_id, b_id) = if *has_bias {
                    (node.inputs[2], node.inputs[3], node.inputs[4])
                } else {
                    (x_id, node.inputs[2], node.inputs[3]) // bias unused
                };
                let in_dims = node.shape.dims();
                let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                // Same windowing as FusedResidualRmsNorm: absolute offsets +
                // output-only bind left x/residual/gamma/beta outside the
                // bound shard on F5-scale arenas → LN wrote all zeros and
                // the DiT residual collapsed to near-identity.
                let win_ids = [node.id, x_id, r_id, bias_id, g_id, b_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let residual_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    r_id,
                    &mut base,
                    &mut size,
                );
                let bias_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    bias_id,
                    &mut base,
                    &mut size,
                );
                let gamma_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    g_id,
                    &mut base,
                    &mut size,
                );
                let beta_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    b_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                // Recompute locals against the final bind window — staging a
                // cross-shard gamma/beta can widen `base` to the whole stripe
                // and leave earlier offs stale (LN then reads/writes zeros).
                let in_off = if arena_tensor_in_window(&arena, x_id, base, size) {
                    arena_local_off_f32(&arena, x_id, base)
                } else {
                    in_off
                };
                let residual_off = if arena_tensor_in_window(&arena, r_id, base, size) {
                    arena_local_off_f32(&arena, r_id, base)
                } else {
                    residual_off
                };
                let bias_off = if arena_tensor_in_window(&arena, bias_id, base, size) {
                    arena_local_off_f32(&arena, bias_id, base)
                } else {
                    bias_off
                };
                let gamma_off = if arena_tensor_in_window(&arena, g_id, base, size) {
                    arena_local_off_f32(&arena, g_id, base)
                } else {
                    gamma_off
                };
                let beta_off = if arena_tensor_in_window(&arena, b_id, base, size) {
                    arena_local_off_f32(&arena, b_id, base)
                } else {
                    beta_off
                };
                let out_off = if arena_tensor_in_window(&arena, node.id, base, size) {
                    arena_local_off_f32(&arena, node.id, base)
                } else {
                    out_off
                };
                let p = FusedResidualLnParams {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off,
                    eps_bits: eps.to_bits(),
                    has_bias: if *has_bias { 1 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::FusedResidualLn { params: p });
                let frk = fused_residual_ln_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<FusedResidualLnParams>());
                let bg = bind_arena_window(&dev.device, frk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                let x_id = node.inputs[0];
                let r_id = node.inputs[1];
                let (bias_id, g_id, b_id) = if *has_bias {
                    (node.inputs[2], node.inputs[3], node.inputs[4])
                } else {
                    (x_id, node.inputs[2], node.inputs[3])
                };
                let in_dims = node.shape.dims();
                let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                // This op reads 5 scattered arena slots (input, residual, bias,
                // gamma, beta) and writes the output. Under arena windowing
                // (arena > max_storage_buffer_binding_size) an output-only
                // window with absolute offsets leaves the inputs — and the
                // param gamma/beta near the arena tail — gigabytes outside the
                // bound window, so the shader reads zeros and the norm collapses
                // to a constant. Window over ALL operands with window-local
                // offsets (staging far/param operands into scratch), exactly
                // like the Binary/Fma paths.
                let win_ids = [node.id, x_id, r_id, bias_id, g_id, b_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let residual_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    r_id,
                    &mut base,
                    &mut size,
                );
                let bias_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    bias_id,
                    &mut base,
                    &mut size,
                );
                let gamma_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    g_id,
                    &mut base,
                    &mut size,
                );
                let beta_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    b_id,
                    &mut base,
                    &mut size,
                );
                let p = FusedResidualRmsNormParams {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off: arena_local_off_f32(&arena, node.id, base),
                    eps_bits: eps.to_bits(),
                    has_bias: if *has_bias { 1 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                };
                schedule.push(Step::FusedResidualRmsNorm { params: p });
                let frk = fused_residual_rms_norm_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<FusedResidualRmsNormParams>());
                let bg = bind_arena_window(&dev.device, frk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::AdaLayerNorm { norm, eps } => {
                let x_id = node.inputs[0];
                let scale_id = node.inputs[1];
                let shift_id = node.inputs[2];
                let in_dims = node.shape.dims();
                let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let x_dims: Vec<usize> = in_dims.iter().map(|d| d.unwrap_static()).collect();
                let mod_dims: Vec<usize> = graph
                    .node(scale_id)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let lead_pack = ada_modulation_lead_pack(&x_dims, &mod_dims);
                let win_ids = [node.id, x_id, scale_id, shift_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let in_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let scale_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    scale_id,
                    &mut base,
                    &mut size,
                );
                let shift_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    shift_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                let in_off = if arena_tensor_in_window(&arena, x_id, base, size) {
                    arena_local_off_f32(&arena, x_id, base)
                } else {
                    in_off
                };
                let scale_off = if arena_tensor_in_window(&arena, scale_id, base, size) {
                    arena_local_off_f32(&arena, scale_id, base)
                } else {
                    scale_off
                };
                let shift_off = if arena_tensor_in_window(&arena, shift_id, base, size) {
                    arena_local_off_f32(&arena, shift_id, base)
                } else {
                    shift_off
                };
                let p = AdaLayerNormParams {
                    outer,
                    inner,
                    in_off,
                    scale_off,
                    shift_off,
                    out_off,
                    eps_bits: eps.to_bits(),
                    layer_norm: u32::from(matches!(norm, AdaNormKind::LayerNorm)),
                    lead_pack: lead_pack_uniform(lead_pack),
                };
                schedule.push(Step::AdaLayerNorm { params: p });
                let ak = ada_layer_norm_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<AdaLayerNormParams>());
                let bg = bind_arena_window(&dev.device, ak, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::GatedResidual => {
                let x_id = node.inputs[0];
                let y_id = node.inputs[1];
                let gate_id = node.inputs[2];
                let in_dims = node.shape.dims();
                let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                let outer = total / inner.max(1);
                let x_dims: Vec<usize> = in_dims.iter().map(|d| d.unwrap_static()).collect();
                let gate_dims: Vec<usize> = graph
                    .node(gate_id)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let lead_pack = ada_modulation_lead_pack(&x_dims, &gate_dims);
                let win_ids = [node.id, x_id, y_id, gate_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let y_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    y_id,
                    &mut base,
                    &mut size,
                );
                let gate_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    gate_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                let x_off = if arena_tensor_in_window(&arena, x_id, base, size) {
                    arena_local_off_f32(&arena, x_id, base)
                } else {
                    x_off
                };
                let y_off = if arena_tensor_in_window(&arena, y_id, base, size) {
                    arena_local_off_f32(&arena, y_id, base)
                } else {
                    y_off
                };
                let gate_off = if arena_tensor_in_window(&arena, gate_id, base, size) {
                    arena_local_off_f32(&arena, gate_id, base)
                } else {
                    gate_off
                };
                let p = GatedResidualParams {
                    outer,
                    inner,
                    x_off,
                    y_off,
                    gate_off,
                    out_off,
                    _pre0: 0,
                    _pre1: 0,
                    lead_pack: lead_pack_uniform(lead_pack),
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };
                schedule.push(Step::GatedResidual { params: p });
                let gk = gated_residual_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<GatedResidualParams>());
                let bg = bind_arena_window(&dev.device, gk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::AdaLayerNormBackward { norm, eps } => {
                let x_id = node.inputs[0];
                let scale_id = node.inputs[1];
                let shift_id = node.inputs[2];
                let dy_id = node.inputs[3];
                let in_dims = graph.node(x_id).shape.dims();
                let inner = in_dims.last().unwrap().unwrap_static() as u32;
                let x_dims: Vec<usize> = in_dims.iter().map(|d| d.unwrap_static()).collect();
                let mod_dims: Vec<usize> = graph
                    .node(scale_id)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &mod_dims);
                let win_ids = [node.id, x_id, scale_id, shift_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let x_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    x_id,
                    &mut base,
                    &mut size,
                );
                let scale_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    scale_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                let x_off = if arena_tensor_in_window(&arena, x_id, base, size) {
                    arena_local_off_f32(&arena, x_id, base)
                } else {
                    x_off
                };
                let scale_off = if arena_tensor_in_window(&arena, scale_id, base, size) {
                    arena_local_off_f32(&arena, scale_id, base)
                } else {
                    scale_off
                };
                let dy_off = if arena_tensor_in_window(&arena, dy_id, base, size) {
                    arena_local_off_f32(&arena, dy_id, base)
                } else {
                    dy_off
                };
                let p = AdaLayerNormBackwardParams {
                    mod_rows,
                    seq_per_mod,
                    inner,
                    x_off,
                    scale_off,
                    dy_off,
                    out_off,
                    eps_bits: eps.to_bits(),
                    layer_norm: u32::from(matches!(norm, AdaNormKind::LayerNorm)),
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                schedule.push(Step::AdaLayerNormBackward { params: p });
                let ak = ada_layer_norm_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<AdaLayerNormBackwardParams>());
                let bg = bind_arena_window(&dev.device, ak, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::GatedResidualBackward => {
                let x_id = node.inputs[0];
                let y_id = node.inputs[1];
                let gate_id = node.inputs[2];
                let dy_id = node.inputs[3];
                let in_dims = graph.node(x_id).shape.dims();
                let inner = in_dims.last().unwrap().unwrap_static() as u32;
                let x_dims: Vec<usize> = in_dims.iter().map(|d| d.unwrap_static()).collect();
                let gate_dims: Vec<usize> = graph
                    .node(gate_id)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &gate_dims);
                let win_ids = [node.id, x_id, y_id, gate_id, dy_id];
                let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                let mut scratch = arena.scratch_off as u64;
                let (mut base, mut size, param_anchor) = arena_multi_op_window(
                    &dev.device,
                    &arena,
                    &graph,
                    &param_offsets,
                    &mut schedule,
                    &mut scratch,
                    &win_ids,
                );
                if !fits && !param_anchor {
                    base =
                        arena_bind_window_covering_scratch_if_needed(&arena, base, size, scratch);
                }
                let y_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    y_id,
                    &mut base,
                    &mut size,
                );
                let gate_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    gate_id,
                    &mut base,
                    &mut size,
                );
                let dy_off = arena_off_in_bind_window(
                    &graph,
                    &param_offsets,
                    &dev.device,
                    &arena,
                    &mut schedule,
                    &mut scratch,
                    dy_id,
                    &mut base,
                    &mut size,
                );
                let out_off = arena_local_off_f32(&arena, node.id, base);
                let y_off = if arena_tensor_in_window(&arena, y_id, base, size) {
                    arena_local_off_f32(&arena, y_id, base)
                } else {
                    y_off
                };
                let gate_off = if arena_tensor_in_window(&arena, gate_id, base, size) {
                    arena_local_off_f32(&arena, gate_id, base)
                } else {
                    gate_off
                };
                let dy_off = if arena_tensor_in_window(&arena, dy_id, base, size) {
                    arena_local_off_f32(&arena, dy_id, base)
                } else {
                    dy_off
                };
                let p = GatedResidualBackwardParams {
                    mod_rows,
                    seq_per_mod,
                    inner,
                    y_off,
                    gate_off,
                    dy_off,
                    out_off,
                    _p0: 0,
                };
                schedule.push(Step::GatedResidualBackward { params: p });
                let gk = gated_residual_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<GatedResidualBackwardParams>());
                let bg = bind_arena_window(&dev.device, gk, &arena, base, size, &u);
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::DequantMatMul { scheme } => {
                use rlx_ir::QuantScheme;
                let x_id = node.inputs[0];
                let w_id = node.inputs[1];
                // Rank-agnostic GEMM dims: `n` = last output dim, `m` = the
                // product of the leading output dims (e.g. batch·seq), `k` =
                // last input dim. A 2D-only `out_dims[1]` read collapses a 3D
                // decode output `[1, 1, hidden]` to `n = 1` (the seq axis),
                // which then mis-sizes the weight window. Mirrors
                // `gguf_gpu::dequant_gguf_scratch_bytes`.
                let out_total = node.shape.num_elements().unwrap_or(0) as u32;
                let n = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u32;
                let m = out_total / n.max(1);
                let x_total = graph.node(x_id).shape.num_elements().unwrap_or(0) as u32;
                let k = x_total / m.max(1);
                if scheme.is_gguf() {
                    schedule.push(Step::DequantMatmulGguf {
                        m,
                        k,
                        n,
                        scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                        x_byte_off: arena.offset(x_id) as u64,
                        w_byte_off: arena.offset(w_id) as u64,
                        out_byte_off: arena.offset(node.id) as u64,
                    });
                    // Host-classified step (`run_dequant_matmul_gguf_gpu` builds
                    // its own bind groups, and a one-row call of the tiled one-row
                    // arm is encoded into the enclosing batch at run time):
                    // schedule-only, like `GroupNorm`. The uniform/bind_group lanes
                    // are indexed by GPU-step count (`gpu_ui`/`gpu_bi`, host steps
                    // skipped), so pushing here would shift every later GPU op's
                    // lane — fatal in the packed decode graph where GGUF ops
                    // interleave with RmsNorm/RoPE/Attention/SwiGLU. Do NOT push.
                } else if matches!(
                    scheme,
                    QuantScheme::Int8Block { .. } | QuantScheme::Int8BlockAsym { .. }
                ) && crate::device::coop_discrete_backend()
                {
                    // Discrete Vulkan/DX12: SPIR-V Int8 dequant is wrong on
                    // NVIDIA (Kitten NSF mush). Host like rlx-vulkan non-GGUF.
                    // Metal wgpu keeps the GPU kernel below.
                    let (block_size, is_asymmetric) = match scheme {
                        QuantScheme::Int8Block { block_size } => (*block_size, false),
                        QuantScheme::Int8BlockAsym { block_size } => (*block_size, true),
                        _ => unreachable!(),
                    };
                    let scale_id = node.inputs[2];
                    let zp_id = node.inputs[3];
                    schedule.push(Step::DequantMatmulInt8Host {
                        m,
                        k,
                        n,
                        block_size,
                        is_asymmetric,
                        x_byte_off: arena.offset(x_id) as u64,
                        w_byte_off: arena.offset(w_id) as u64,
                        scale_byte_off: arena.offset(scale_id) as u64,
                        zp_byte_off: arena.offset(zp_id) as u64,
                        out_byte_off: arena.offset(node.id) as u64,
                    });
                } else if let Some((kind, bits, group_size)) = scheme.mlx_gpu_launch() {
                    let scale_id = node.inputs[2];
                    let zp_id = node.inputs[3];
                    // The GPU WGSL kernel binds the packed weight as a storage
                    // buffer, so a weight past the binding limit (discrete
                    // Vulkan/DX12, where max_bind is 2 GiB) must host. A weight
                    // parked in the separate `weight_buffer` used to be its own
                    // disjunct here — a driver-level defect reads those wrong at
                    // large offsets (zeros past a fuzzy ~2-3 GiB cliff, exhaustively
                    // ruled non-binding in the llama-lm-head investigation) — and is
                    // now covered by `parked_operand` below, which asks every
                    // operand rather than the weight alone. Small in-arena models
                    // keep the GPU path.
                    let max_bind = dev.device.limits().max_storage_buffer_binding_size as usize;
                    let w_off_tagged = arena.offset(w_id);
                    let weight_host = crate::buffer::raw_weight_off(w_off_tagged) >= max_bind;
                    // `DequantMatmulMlxParams` carries absolute word offsets and
                    // the kernel applies them to an output window, so the GPU
                    // branch is correct only below the per-binding ceiling and only
                    // while every operand is addressed from the act arena — the
                    // same two geometry questions `should_host_at_ceiling` asks,
                    // applied here because `Op::DequantMatMul` is not on that
                    // rule's list (its GGUF and Int8 sub-paths are host steps and
                    // must keep that route). `any_parked_operand` covers `scale`
                    // and `zp` as well as the weight: their offsets are derived the
                    // same way and lose their parked provenance in the same `u32`
                    // cast, so the weight-only `weight_host` test alone would read
                    // them from the activation arena silently.
                    let whole_at_zero = arena.binds_whole_at_base_zero(max_bind as u64);
                    let parked_operand = any_parked_operand(&arena, node);
                    if rlx_gpu_host::mlx_dequant_gpu_disabled()
                        || weight_host
                        || parked_operand
                        || !whole_at_zero
                    {
                        schedule.push(Step::DequantMatmulMlxHost {
                            m,
                            k,
                            n,
                            scheme: *scheme,
                            x_byte_off: arena.offset(x_id) as u64,
                            w_byte_off: arena.offset(w_id) as u64,
                            scale_byte_off: arena.offset(scale_id) as u64,
                            zp_byte_off: arena.offset(zp_id) as u64,
                            out_byte_off: arena.offset(node.id) as u64,
                        });
                    } else {
                        let p = DequantMatmulMlxParams {
                            m,
                            k,
                            n,
                            kind,
                            bits,
                            group_size,
                            x_byte_off: arena.offset(x_id) as u32,
                            w_byte_off: arena.offset(w_id) as u32,
                            scale_byte_off: arena.offset(scale_id) as u32,
                            zp_byte_off: arena.offset(zp_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            _pad: 0,
                        };
                        schedule.push(Step::DequantMatmulMlx { params: p });
                        let dk = dequant_matmul_mlx_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<DequantMatmulMlxParams>());
                        let bg = bind_op_output_window(&dev.device, dk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                } else {
                    let (block_size, scheme_id) = match scheme {
                        QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                        QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                        QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                        QuantScheme::Fp8E4m3 => (1, 3u32),
                        QuantScheme::Fp8E5m2 => (1, 4u32),
                        QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                        other => panic!("rlx-wgpu DequantMatMul: unsupported scheme {other:?}"),
                    };
                    let scale_id = node.inputs[2];
                    let zp_id = node.inputs[3];
                    // `DequantMatmulParams` carries absolute word offsets the
                    // kernel applies to an output window, so the GPU step needs
                    // the same two geometry questions the MLX arm above asks —
                    // asked here because `Op::DequantMatMul` is not on the ceiling
                    // rule's list (its GGUF and Int8 sub-paths are host steps and
                    // keep that route). `any_parked_operand` covers `scale` and
                    // `zp` as well as the weight: their offsets lose the parked
                    // tag in the same `u32` cast, so a weight-only test would read
                    // them from the activation arena silently.
                    if !arena.binds_whole_at_base_zero(
                        dev.device.limits().max_storage_buffer_binding_size,
                    ) || any_parked_operand(&arena, node)
                    {
                        // A parked *packed* weight (the usual case for this arm)
                        // cannot be staged by the generic host route, which sizes
                        // every operand as f32; `assert_host_route_stageable`
                        // refuses it loudly instead of reading four times its
                        // length, past the operand and possibly past the weight
                        // buffer.
                        assert_host_route_stageable(&graph, &arena, node);
                        schedule.push(host_op_step(&graph, node, &arena));
                    } else {
                        let p = DequantMatmulParams {
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            w_off: (arena.offset(w_id) / 4) as u32,
                            scale_off: (arena.offset(scale_id) / 4) as u32,
                            zp_off: (arena.offset(zp_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                        };
                        schedule.push(Step::DequantMatmul { params: p });
                        let dk = dequant_matmul_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<DequantMatmulParams>());
                        let bg = bind_op_output_window(&dev.device, dk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }
            }
            Op::RmsNormBackwardInput { eps, .. }
            | Op::RmsNormBackwardGamma { eps, .. }
            | Op::RmsNormBackwardBeta { eps, .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                let foff = |i: usize| (arena.offset(node.inputs[i]) / 4) as u32;
                let wrt = match &node.op {
                    Op::RmsNormBackwardInput { .. } => 0u32,
                    Op::RmsNormBackwardGamma { .. } => 1u32,
                    Op::RmsNormBackwardBeta { .. } => 2u32,
                    _ => unreachable!(),
                };
                let p = RmsNormBwdParams {
                    outer: rows,
                    inner: h,
                    x_off: foff(0),
                    gamma_off: foff(1),
                    beta_off: foff(2),
                    dy_off: foff(3),
                    out_off: (arena.offset(node.id) / 4) as u32,
                    eps_bits: eps.to_bits(),
                    wrt,
                };
                let rk = if wrt == 0 {
                    rms_norm_backward_kernel(&dev.device)
                } else {
                    rms_norm_backward_param_kernel(&dev.device)
                };
                let u = emit_uniform(std::mem::size_of::<RmsNormBwdParams>());
                let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                match &node.op {
                    Op::RmsNormBackwardInput { .. } => {
                        schedule.push(Step::RmsNormBackwardInput { params: p });
                    }
                    Op::RmsNormBackwardGamma { .. } => {
                        schedule.push(Step::RmsNormBackwardGamma { params: p });
                    }
                    Op::RmsNormBackwardBeta { .. } => {
                        schedule.push(Step::RmsNormBackwardBeta { params: p });
                    }
                    _ => unreachable!(),
                }
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::LayerNormBackwardInput { eps, .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                let p = LayerNormBwdParams {
                    outer: rows,
                    inner: h,
                    x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                    gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    eps_bits: eps.to_bits(),
                    scratch_off: 0,
                };
                let rk = layer_norm_backward_input_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                schedule.push(Step::LayerNormBackwardInput { params: p });
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::LayerNormBackwardGamma { eps, .. } => {
                // Inputs: [x, dy] — gamma_off is unused for this op.
                // Emit two steps: a multi-workgroup partial that
                // writes per-chunk dgamma to the tail scratch zone,
                // and a single-workgroup reduce that sums chunks
                // into the final dgamma slot.
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                const ROWS_PER_WG: u32 = 16;
                let num_workgroups = rows.div_ceil(ROWS_PER_WG.max(1));
                let scratch_off_words = (arena.scratch_off / 4) as u32;
                let partial_params = LayerNormBwdParams {
                    outer: rows,
                    inner: h,
                    x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                    gamma_off: 0,
                    dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    out_off: 0, // unused by the partial kernel
                    eps_bits: eps.to_bits(),
                    scratch_off: scratch_off_words,
                };
                let reduce_params = LayerNormBwdParams {
                    // `outer` for the reduce kernel carries the
                    // number of partial chunks we just emitted.
                    outer: num_workgroups,
                    inner: h,
                    x_off: 0,
                    gamma_off: 0,
                    dy_off: 0,
                    out_off: (arena.offset(node.id) / 4) as u32,
                    eps_bits: eps.to_bits(),
                    scratch_off: scratch_off_words,
                };
                let p_k = layer_norm_backward_gamma_partial_kernel(&dev.device);
                let r_k = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                let p_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                let r_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                let p_bg = bind_op_output_window(&dev.device, p_k, &arena, node.id, &p_u);
                let r_bg = bind_op_output_window(&dev.device, r_k, &arena, node.id, &r_u);
                schedule.push(Step::LayerNormBackwardGammaPartial {
                    params: partial_params,
                    num_workgroups,
                });
                schedule.push(Step::LayerNormBackwardGammaReduce {
                    params: reduce_params,
                });
                uniforms.push(p_u);
                uniforms.push(r_u);
                bind_groups.push(p_bg);
                bind_groups.push(r_bg);
            }
            Op::RopeBackward {
                head_dim,
                n_rot,
                style,
            } => {
                let dy_shape = &graph.node(node.inputs[0]).shape;
                let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                    // Fold leading axes into `batch` — same defect as the
                    // forward path, on the gradient. See `Op::Rope` above.
                    let rank = dy_shape.rank();
                    (
                        (0..rank - 2)
                            .map(|i| dy_shape.dim(i).unwrap_static())
                            .product::<usize>() as u32,
                        dy_shape.dim(rank - 2).unwrap_static() as u32,
                        dy_shape.dim(rank - 1).unwrap_static() as u32,
                    )
                } else {
                    (
                        1,
                        dy_shape.dim(0).unwrap_static() as u32,
                        dy_shape.dim(1).unwrap_static() as u32,
                    )
                };
                let cos_shape = &graph.node(node.inputs[1]).shape;
                let cos_len = cos_shape.num_elements().unwrap() as u32;
                // Row width off the table — see the CUDA site.
                // Shared with every other backend so the rule cannot drift
                // again; a rank-1 table's rows are `n_rot/2`, not its length.
                let cos_row_stride = rlx_ir::shape::rope_table_stride(cos_shape, *n_rot) as u32;
                let p = RopeBwdParams {
                    batch,
                    seq,
                    hidden,
                    head_dim: *head_dim as u32,
                    n_rot: *n_rot as u32,
                    dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                    cos_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    sin_off: (arena.offset(node.inputs[2]) / 4) as u32,
                    dx_off: (arena.offset(node.id) / 4) as u32,
                    cos_len,
                    cos_row_stride,
                    // Must match the forward `Rope` this is the adjoint of;
                    // same mapping as the `Op::Rope` arm above.
                    interleaved: match style {
                        rlx_ir::op::RopeStyle::NeoX => 0,
                        rlx_ir::op::RopeStyle::GptJ => 1,
                    },
                };
                let rk = rope_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<RopeBwdParams>());
                let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                schedule.push(Step::RopeBackward { params: p });
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::CumsumBackward { exclusive, .. } => {
                let dy_shape = &graph.node(node.inputs[0]).shape;
                let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                let p = CumsumBwdParams {
                    outer: rows,
                    inner: cols,
                    dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                    dx_off: (arena.offset(node.id) / 4) as u32,
                    exclusive: if *exclusive { 1 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };
                let ck = cumsum_backward_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<CumsumBwdParams>());
                let bg = bind_op_output_window(&dev.device, ck, &arena, node.id, &u);
                schedule.push(Step::CumsumBackward { params: p });
                uniforms.push(u);
                bind_groups.push(bg);
            }
            Op::GatherBackward { .. } => {
                let dy_shape = &graph.node(node.inputs[0]).shape;
                let idx_shape = &graph.node(node.inputs[1]).shape;
                let out_shape = &node.shape;
                let rank = out_shape.rank();
                let axis = match &node.op {
                    Op::GatherBackward { axis } => *axis,
                    _ => 0,
                };
                let axis_u = if axis < 0 {
                    (rank as i32 + axis) as usize
                } else {
                    axis as usize
                };
                let outer: usize = (0..axis_u)
                    .map(|i| dy_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let num_idx = idx_shape.gather_index_count(axis_u);
                let trailing: usize = (axis_u + 1..dy_shape.rank())
                    .map(|i| dy_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let axis_dim = out_shape.dim(axis_u).unwrap_static();
                let p = GatherBwdParams {
                    outer: outer as u32,
                    axis_dim: axis_dim as u32,
                    num_idx: num_idx as u32,
                    trailing: trailing as u32,
                    dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                    idx_off: (arena.offset(node.inputs[1]) / 4) as u32,
                    dst_off: (arena.offset(node.id) / 4) as u32,
                    _p0: 0,
                };
                let zk = gather_backward_zero_kernel(&dev.device);
                let u = emit_uniform(std::mem::size_of::<GatherBwdParams>());
                let bg = bind_op_output_window(&dev.device, zk, &arena, node.id, &u);
                schedule.push(Step::GatherBackward { params: p });
                uniforms.push(u);
                bind_groups.push(bg);
            }
            #[cfg(feature = "splat")]
            Op::GaussianSplatRender {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => {
                let elem_len =
                    |id: NodeId| -> u32 { graph.node(id).shape.num_elements().unwrap_or(0) as u32 };
                schedule.push(Step::GaussianSplatRender {
                    positions_byte_off: arena.offset(node.inputs[0]) as u32,
                    positions_len: elem_len(node.inputs[0]),
                    scales_byte_off: arena.offset(node.inputs[1]) as u32,
                    scales_len: elem_len(node.inputs[1]),
                    rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                    rotations_len: elem_len(node.inputs[2]),
                    opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                    opacities_len: elem_len(node.inputs[3]),
                    colors_byte_off: arena.offset(node.inputs[4]) as u32,
                    colors_len: elem_len(node.inputs[4]),
                    sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                    sh_coeffs_len: elem_len(node.inputs[5]),
                    meta_byte_off: arena.offset(node.inputs[6]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                    width: *width,
                    height: *height,
                    tile_size: *tile_size,
                    radius_scale: *radius_scale,
                    alpha_cutoff: *alpha_cutoff,
                    max_splat_steps: *max_splat_steps,
                    transmittance_threshold: *transmittance_threshold,
                    max_list_entries: *max_list_entries,
                });
            }

            #[cfg(feature = "splat")]
            Op::GaussianSplatRenderBackward {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
                loss_grad_clip,
                sh_band,
                max_anisotropy,
            } => {
                let elem_len =
                    |id: NodeId| -> u32 { graph.node(id).shape.num_elements().unwrap_or(0) as u32 };
                schedule.push(Step::GaussianSplatRenderBackward {
                    positions_byte_off: arena.offset(node.inputs[0]) as u32,
                    positions_len: elem_len(node.inputs[0]),
                    scales_byte_off: arena.offset(node.inputs[1]) as u32,
                    scales_len: elem_len(node.inputs[1]),
                    rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                    rotations_len: elem_len(node.inputs[2]),
                    opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                    opacities_len: elem_len(node.inputs[3]),
                    colors_byte_off: arena.offset(node.inputs[4]) as u32,
                    colors_len: elem_len(node.inputs[4]),
                    sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                    sh_coeffs_len: elem_len(node.inputs[5]),
                    meta_byte_off: arena.offset(node.inputs[6]) as u32,
                    d_loss_byte_off: arena.offset(node.inputs[7]) as u32,
                    d_loss_len: elem_len(node.inputs[7]),
                    packed_byte_off: arena.offset(node.id) as u32,
                    packed_len: node.shape.num_elements().unwrap_or(0) as u32,
                    width: *width,
                    height: *height,
                    tile_size: *tile_size,
                    radius_scale: *radius_scale,
                    alpha_cutoff: *alpha_cutoff,
                    max_splat_steps: *max_splat_steps,
                    transmittance_threshold: *transmittance_threshold,
                    max_list_entries: *max_list_entries,
                    loss_grad_clip: *loss_grad_clip,
                    sh_band: *sh_band,
                    max_anisotropy: *max_anisotropy,
                });
            }

            #[cfg(feature = "splat")]
            Op::GaussianSplatPrepare {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => {
                let elem_len =
                    |id: NodeId| -> u32 { graph.node(id).shape.num_elements().unwrap_or(0) as u32 };
                schedule.push(Step::GaussianSplatPrepare {
                    positions_byte_off: arena.offset(node.inputs[0]) as u32,
                    positions_len: elem_len(node.inputs[0]),
                    scales_byte_off: arena.offset(node.inputs[1]) as u32,
                    scales_len: elem_len(node.inputs[1]),
                    rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                    rotations_len: elem_len(node.inputs[2]),
                    opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                    opacities_len: elem_len(node.inputs[3]),
                    colors_byte_off: arena.offset(node.inputs[4]) as u32,
                    colors_len: elem_len(node.inputs[4]),
                    sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                    sh_coeffs_len: elem_len(node.inputs[5]),
                    meta_byte_off: arena.offset(node.inputs[6]) as u32,
                    meta_len: elem_len(node.inputs[6]),
                    prep_byte_off: arena.offset(node.id) as u32,
                    prep_len: node.shape.num_elements().unwrap_or(0) as u32,
                    width: *width,
                    height: *height,
                    tile_size: *tile_size,
                    radius_scale: *radius_scale,
                    alpha_cutoff: *alpha_cutoff,
                    max_splat_steps: *max_splat_steps,
                    transmittance_threshold: *transmittance_threshold,
                    max_list_entries: *max_list_entries,
                });
            }

            #[cfg(feature = "splat")]
            Op::GaussianSplatRasterize {
                width,
                height,
                tile_size,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => {
                let elem_len =
                    |id: NodeId| -> u32 { graph.node(id).shape.num_elements().unwrap_or(0) as u32 };
                let prep_id = node.inputs[0];
                let count = match &graph.node(prep_id).op {
                    rlx_ir::Op::GaussianSplatPrepare { .. } => {
                        elem_len(graph.node(prep_id).inputs[0]) / 3
                    }
                    _ => 1,
                };
                schedule.push(Step::GaussianSplatRasterize {
                    prep_byte_off: arena.offset(prep_id) as u32,
                    prep_len: elem_len(prep_id),
                    meta_byte_off: arena.offset(node.inputs[1]) as u32,
                    meta_len: elem_len(node.inputs[1]),
                    dst_byte_off: arena.offset(node.id) as u32,
                    dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                    count,
                    width: *width,
                    height: *height,
                    tile_size: *tile_size,
                    alpha_cutoff: *alpha_cutoff,
                    max_splat_steps: *max_splat_steps,
                    transmittance_threshold: *transmittance_threshold,
                    max_list_entries: *max_list_entries,
                });
            }

            Op::If { .. } | Op::While { .. } => {
                // Should be unreachable: unfuse.rs inlines both branches
                // (If) or unrolls max_iterations (While) into the parent
                // graph using primitive ops + Where for the gating. If
                // we hit this arm, the unfusion pass has a gap.
                panic!(
                    "rlx-wgpu: Op::If/While leaked past unfusion pass — \
                        check unfuse.rs::expand_if / expand_while"
                );
            }
            Op::RngNormal {
                mean,
                scale,
                key,
                op_seed,
            } => {
                let len = node.shape.num_elements().unwrap_or(0) as u32;
                schedule.push(Step::RngNormalHost {
                    dst_byte_off: arena.offset(node.id) as u32,
                    len,
                    mean: *mean,
                    scale: *scale,
                    key: *key,
                    op_seed: *op_seed,
                });
            }
            Op::RngUniform {
                low,
                high,
                key,
                op_seed,
            } => {
                let len = node.shape.num_elements().unwrap_or(0) as u32;
                schedule.push(Step::RngUniformHost {
                    dst_byte_off: arena.offset(node.id) as u32,
                    len,
                    low: *low,
                    high: *high,
                    key: *key,
                    op_seed: *op_seed,
                });
            }
            // Standalone nearest 2× upsample: the region-marking pass wraps
            // a bare `Op::ResizeNearest2x` into a single-step TransformRegion.
            // Route that to the host resize helper (mirrors the bare arm).
            Op::TransformRegion { steps, .. }
                if steps.len() == 1
                    && matches!(
                        steps[0],
                        rlx_ir::op::TransformStep::ResizeNearest2x(
                            rlx_ir::op::ChainOperand::Input(0)
                        )
                    ) =>
            {
                let in_shape = &graph.node(node.inputs[0]).shape;
                schedule.push(Step::ResizeNearest2xHost {
                    src_byte_off: arena.offset(node.inputs[0]) as u32,
                    dst_byte_off: arena.offset(node.id) as u32,
                    n: in_shape.dim(0).unwrap_static() as u32,
                    c: in_shape.dim(1).unwrap_static() as u32,
                    h: in_shape.dim(2).unwrap_static() as u32,
                    w: in_shape.dim(3).unwrap_static() as u32,
                });
            }
            other => panic!(
                "rlx-wgpu: op {other:?} not yet lowered (v2 covers Matmul, \
                 Binary, Compare, Activation, Where — fall back to CPU/Metal/MLX)"
            ),
        }
    }

    if rlx_ir::env::flag("RLX_WGPU_SCHEDULE") || rlx_ir::env::flag("RLX_DISPATCH_REPORT") {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        let mut fft_gpu = 0usize;
        let mut fft_host = 0usize;
        for s in &schedule {
            *counts.entry(step_name(s)).or_insert(0) += 1;
            match s {
                Step::FftGpu { .. } => fft_gpu += 1,
                Step::FftHost { .. } => fft_host += 1,
                _ => {}
            }
        }
        let arena_mb = arena.size as f64 / (1u64 << 20) as f64;
        eprintln!(
            "[rlx-wgpu] schedule: {} steps, arena={arena_mb:.1} MiB, fft_gpu={fft_gpu}, \
             fft_host={fft_host}, ceiling_hosted={ceiling_hosted}",
            schedule.len()
        );
        if ceiling_hosted > 0 {
            eprintln!(
                "    per-binding ceiling: {ceiling_hosted} step(s) computed on the host — the \
                 memory they address could not be bound whole at base zero (an arena over the \
                 device's max storage-buffer binding, a striped arena, or a parameter parked in \
                 the weight buffer), so absolute-addressing ops were hosted instead of reading \
                 the wrong slots. Keep the arena under the per-binding limit and its parameters \
                 in it (do not set RLX_WGPU_ALLOW_SHARD)."
            );
        }
        if im2col_fallbacks > 0 {
            eprintln!(
                "    im2col: {im2col_fallbacks} conv(s) cleared the route's shape gates but \
                 kept the direct/tiled route, for a reason the shapes alone do not settle: \
                 a weight parked in the weight buffer, a col panel over the {} B tail \
                 reserved for it, or an arena that with its panel does not fit one storage \
                 binding (max_storage_buffer_binding_size = {} B)",
                arena.scratch_bytes,
                dev.device.limits().max_storage_buffer_binding_size
            );
        }
        if attention_window_declines > 0 {
            eprintln!(
                "    attention: {attention_window_declines} many-row step(s) cleared the \
                 route's shape gates but kept the fused kernel, for a reason the shapes do \
                 not settle: an arena addressed in stripes rather than whole (a binding \
                 limit below it, or RLX_WGPU_ALLOW_SHARD), or one whose bind window does \
                 not reach the plan's regions. RLX_WGPU_DEBUG_ATTN_MASK names the term per \
                 step."
            );
        }
        for (n, c) in &counts {
            eprintln!("    {c:>4} × {n}");
        }
    }

    if let Some(prev) = prev_node {
        while step_nodes.len() < schedule.len() {
            step_nodes.push(prev);
        }
    }
    let coop_f16_vk = schedule_uses_coop_f16_vk(&schedule);

    crate::backend::WgpuExecutable {
        graph,
        arena,
        dequant_scratch_off,
        gdn_scratch_off,
        schedule,
        step_nodes,
        in_snapshot: false,
        snapshot_stop: None,
        input_offsets,
        param_offsets,
        uniforms,
        bind_groups,
        meta_buffers,
        unresolved: None,
        last_binding: None,
        pending_params: HashMap::new(),
        pending_param_bytes: HashMap::new(),
        active_extent: None,
        uniforms_active_extent: None,
        coop_f16_vk,
        coop_f16_b_param,
        coop_f16_vk_wide_b: HashSet::new(),
        coop_f16_vk_wide_bind_groups,
        coop_f16_host_activations,
        stashed_params: HashMap::new(),
        readback_staging: None,
        tiny_readback: None,
        dispatch_only: false,
        fft_gpu_steps,
        gpu_handles: HashMap::new(),
        gpu_handle_feeds: HashMap::new(),
        gpu_handle_resident: HashSet::new(),
        kv_row_feeds: HashMap::new(),
        kv_feed_staging: None,
        pending_read_indices: None,
        rng,
        static_once_steps,
        static_once_done: false,
        ceiling_hosted,
        im2col_fallbacks,
        attention_window_declines,
    }
}

/// Report every write-after-read hazard the *executed* schedule can hit.
///
/// The memory planner and `plan_check` both reason about liveness on the model
/// graph. wgpu plans the **lowered** graph — after fusion, unfusing and op
/// expansion — so a slot reuse that is safe upstream can be unsafe here, and
/// neither of those checks would show it. This walks the actual schedule: for
/// every consumer `C` reading input `I`, it flags any node written between `I`'s
/// production and `C`'s execution whose slot overlaps `I`'s bytes.
///
/// Views legitimately overlap their producer, so a hazard whose writer is a
/// view-family op feeding the same chain is annotated rather than hidden — the
/// caller decides. `RLX_WGPU_HAZARD_REPORT=1`.
fn report_slot_hazards(graph: &Graph, plan: &rlx_opt::memory::MemoryPlan) {
    let pos: HashMap<NodeId, usize> = plan
        .schedule
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let span = |id: NodeId| {
        plan.assignments
            .get(&id)
            .map(|s| (s.offset, s.offset + s.size))
    };
    let kind = |id: NodeId| {
        graph
            .nodes()
            .iter()
            .find(|n| n.id == id)
            .map(|n| format!("{:?}", n.op.kind()))
            .unwrap_or_else(|| "?".into())
    };
    // Only a *pure* view aliases its parent and compiles to a Nop. A Transpose,
    // or a Narrow on any axis but 0, genuinely writes its own slot — matching
    // those by op kind labelled real hazards `[view]` and hid them.
    let is_view = |id: NodeId| {
        graph
            .nodes()
            .iter()
            .find(|n| n.id == id)
            .is_some_and(|n| rlx_compile::memory::is_pure_view(graph, n))
    };
    let mut found = 0usize;
    let mut real_hazards = 0usize;
    for (&consumer, &cpos) in &pos {
        let Some(node) = graph.nodes().iter().find(|n| n.id == consumer) else {
            continue;
        };
        for &input in &node.inputs {
            let (Some((lo, hi)), Some(&ipos)) = (span(input), pos.get(&input)) else {
                continue;
            };
            for (&writer, &wpos) in &pos {
                if writer == input || writer == consumer || wpos <= ipos || wpos >= cpos {
                    continue;
                }
                let Some((wlo, whi)) = span(writer) else {
                    continue;
                };
                if wlo < hi && lo < whi {
                    found += 1;
                    if !is_view(writer) && !is_view(input) {
                        real_hazards += 1;
                    }
                    if found <= 20 {
                        eprintln!(
                            "[rlx-wgpu] HAZARD step {cpos} {} reads {} (step {ipos}, bytes \
                             [{lo},{hi})) but step {wpos} {} writes [{wlo},{whi}){}",
                            kind(consumer),
                            kind(input),
                            kind(writer),
                            if is_view(writer) || is_view(input) {
                                "  [view]"
                            } else {
                                ""
                            }
                        );
                    }
                }
            }
        }
    }
    // Second class: a node whose OUTPUT slot overlaps one of its own INPUTS.
    // The loop above only looks for a *third-party* writer landing between the
    // producer and the consumer, so it is blind to in-place aliasing — and an
    // in-place kernel that does not read its whole input before writing (or
    // whose lanes are not row-private) corrupts silently. Views are excluded:
    // aliasing a parent is what a view IS.
    let mut inplace = 0usize;
    for (&consumer, &cpos) in &pos {
        let Some(node) = graph.nodes().iter().find(|n| n.id == consumer) else {
            continue;
        };
        if is_view(consumer) {
            continue;
        }
        let Some((olo, ohi)) = span(consumer) else {
            continue;
        };
        for &input in &node.inputs {
            if input == consumer {
                continue;
            }
            let Some((ilo, ihi)) = span(input) else {
                continue;
            };
            if olo < ihi && ilo < ohi {
                inplace += 1;
                if inplace <= 20 {
                    eprintln!(
                        "[rlx-wgpu] INPLACE step {cpos} {} output [{olo},{ohi}) overlaps its                          input {} [{ilo},{ihi}){}",
                        kind(consumer),
                        kind(input),
                        if (olo, ohi) == (ilo, ihi) {
                            "  [exact]"
                        } else {
                            "  [partial]"
                        }
                    );
                }
            }
        }
    }
    eprintln!(
        "[rlx-wgpu] hazard scan: {found} write-after-read hazard(s) ({real_hazards} not view-explained), {inplace} in-place          output/input overlap(s) over {} scheduled nodes",
        plan.schedule.len()
    );
}
