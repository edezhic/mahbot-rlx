// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `dispatch` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams, AttentionBwdParams,
    AttentionParams, BatchElementwiseRegionParams, BinaryParams, Conv1dParams, Conv2dParams,
    Conv3dParams, CumsumBwdParams, CumsumParams, DequantMatmulParams, ElementwiseRegionParams,
    ExpandParams, FmaParams, FusedResidualLnParams, FusedResidualLnTeeParams,
    FusedResidualRmsNormParams, GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams,
    GatherBwdParams, GatherParams, GroupedMatmulParams, GruParams, Kernel, LayerNormBwdParams,
    LayerNormParams, Mamba2Params, MatmulParams, MatmulQkvParams, NarrowConcatParams, Pool1dParams,
    Pool2dParams, Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams,
    RopeParams, SampleParams, ScatterAddParams, SceParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
    argmax_kernel, attention_bwd_kernel, attention_kernel, batch_elementwise_region_kernel,
    binary_kernel, cast_f32_to_f16_kernel, compare_kernel, concat_kernel, conv1d_kernel,
    conv2d_kernel, conv3d_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel, fma_kernel,
    fused_residual_ln_kernel, fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel,
    gather_axis_kernel, gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel,
    gather_split_kernel, grouped_matmul_kernel, gru_kernel,
    layer_norm_backward_gamma_partial_kernel, layer_norm_backward_gamma_reduce_kernel,
    layer_norm_backward_input_kernel, layernorm_kernel, mamba2_kernel,
    matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel,
    pool3d_kernel, reduce_kernel, rms_norm_backward_kernel, rms_norm_backward_param_kernel,
    rnn_kernel, rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_kernel,
    selective_scan_kernel, softmax_cross_entropy_kernel, softmax_kernel, topk_kernel,
    transpose_kernel, umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel,
    welch_peaks_gpu_kernel, where_kernel,
};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    pub(crate) fn dispatch_arena_copy_bytes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
        nbytes: usize,
    ) {
        let src = self.arena.offset(src_id) as u64;
        let dst = self.arena.offset(dst_id) as u64;
        let nbytes = nbytes
            .min(self.arena.len_of(src_id))
            .min(self.arena.len_of(dst_id)) as u64;
        // A request of zero bytes, or one the clamp above shrinks to nothing
        // because a slot is empty: a copy of zero bytes moves nothing.
        if nbytes == 0 {
            return;
        }
        let elems = (nbytes / 4) as u32;
        let (lo, hi) = copy_span(src, dst, nbytes);
        let max_binding = dev.device.limits().max_storage_buffer_binding_size;
        let Some((base, size)) = copy_window(lo, hi, self.arena.size as u64, max_binding) else {
            panic!(
                "rlx-wgpu arena copy spans {lo}..{hi} between offsets {src} and {dst} and no \
                 storage-binding window covers it: the window would exceed this adapter's \
                 {max_binding} B binding limit, or could not reach {hi}. The copy would have to \
                 be split."
            );
        };
        if rlx_ir::env::flag("RLX_WGPU_FEED_TRACE") {
            eprintln!(
                "[copy] src={src} dst={dst} nbytes={nbytes} elems={elems} base={base} size={size} \
                 in_off={} out_off={} src_end_rel={} dst_end_rel={} window_elems={} arena={}",
                src.saturating_sub(base) / 4,
                dst.saturating_sub(base) / 4,
                (src + nbytes).saturating_sub(base),
                (dst + nbytes).saturating_sub(base),
                size / 4,
                self.arena.size,
            );
        }
        super::encode_arena_copy(
            dev,
            &self.arena,
            enc,
            src,
            dst,
            nbytes,
            base,
            size,
            "rlx-wgpu kv_feed_copy",
        );
    }

    #[allow(dead_code)]
    pub(crate) fn dispatch_arena_copy_between_nodes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
    ) {
        let nbytes = self.arena.len_of(src_id).min(self.arena.len_of(dst_id));
        self.dispatch_arena_copy_bytes(dev, enc, src_id, dst_id, nbytes);
    }
}
