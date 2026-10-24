// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WGSL kernel sources + per-kernel pipeline cache.
//!
//! Pipelines are content-addressed: same WGSL source + same entry
//! point yields the same pipeline. We hold them in `OnceLock`s so a
//! single device dispatches every (graph, op) pair against a cached
//! compilation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use bytemuck::{Pod, Zeroable};
pub const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
pub const MATMUL_BF16W_WGSL: &str = include_str!("matmul_bf16w.wgsl");
pub const MATMUL_WIDE_WGSL: &str = include_str!("matmul_wide.wgsl");
pub const MATMUL_WIDE_NV_WGSL: &str = include_str!("matmul_wide_nv.wgsl");
pub const MATMUL_WIDE_VEC4_WGSL: &str = include_str!("matmul_wide_vec4.wgsl");
pub const MATMUL_F16W_WGSL: &str = include_str!("matmul_f16w.wgsl");
pub const MATMUL_F16_COMPUTE_WGSL: &str = include_str!("matmul_f16_compute.wgsl");
pub const MATMUL_COOP16_WGSL: &str = include_str!("matmul_coop16.wgsl");
pub const MATMUL_COOP_F16_F32ACC_WGSL: &str = include_str!("matmul_coop_f16_f32acc.wgsl");
pub const MATMUL_COOP_F32_WGSL: &str = include_str!("matmul_coop_f32.wgsl");
pub const MATMUL_COOP_F32_DIRECT_WGSL: &str = include_str!("matmul_coop_f32_direct.wgsl");
pub const MATMUL_COOP_F32_PORTABLE_WGSL: &str = include_str!("matmul_coop_f32_portable.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WGSL: &str = include_str!("matmul_coop_f16_vulkan.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WIDEN_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_widen.wgsl");
pub const MATMUL_COOP_F16_VULKAN_F32ACC_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_f32acc.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_widen_f32acc.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WGSL: &str = include_str!("matmul_qkv_coop_f16_vk.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WIDEN_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_widen.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_F32ACC_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_f32acc.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_widen_f32acc.wgsl");
pub const CAST_F32_TO_F16_WGSL: &str = include_str!("cast_f32_to_f16.wgsl");
pub const RNG_PHILOX_WGSL: &str = include_str!("rng_philox.wgsl");
pub const FUSED_SWIGLU_WGSL: &str = include_str!("fused_swiglu.wgsl");
pub const FUSED_CONV_BIAS_ACT_WGSL: &str = include_str!("fused_conv_bias_act.wgsl");
pub const BINARY_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/binary.wgsl"));
// Assembled at build time: `rlx_activation_apply` (generated from the shared
// rlxsl manifest) prepended to `unary_main.wgsl`. See build.rs.
pub const UNARY_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/unary.wgsl"));
pub const UNARY_F16_MIRROR_WGSL: &str = include_str!("unary_f16_mirror.wgsl");
pub const COMPARE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/compare.wgsl"));
pub const WHERE_WGSL: &str = include_str!("where.wgsl");
pub const FMA_WGSL: &str = include_str!("fma.wgsl");
// Assembled at build time: `rlx_activation_backward` (auto-differentiated from
// the forward manifest via rlxsl) prepended to `activation_backward_main.wgsl`.
pub const ACTIVATION_BACKWARD_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/activation_backward.wgsl"));
pub const REDUCE_WGSL: &str = include_str!("reduce.wgsl");
pub const SOFTMAX_WGSL: &str = include_str!("softmax.wgsl");
pub const SOFTMAX_CROSS_ENTROPY_WGSL: &str = include_str!("softmax_cross_entropy.wgsl");
pub const SOFTMAX_CROSS_ENTROPY_BWD_WGSL: &str = include_str!("softmax_cross_entropy_bwd.wgsl");
pub const MAXPOOL2D_BWD_WGSL: &str = include_str!("maxpool2d_backward.wgsl");
pub const MAXPOOL3D_BWD_WGSL: &str = include_str!("maxpool3d_backward.wgsl");
pub const CONV3D_BWD_INPUT_WGSL: &str = include_str!("conv3d_backward_input.wgsl");
pub const CONV3D_BWD_WEIGHT_WGSL: &str = include_str!("conv3d_backward_weight.wgsl");
pub const LAYERNORM_WGSL: &str = include_str!("layernorm.wgsl");
pub const LAYERNORM_VEC4_WGSL: &str = include_str!("layernorm_vec4.wgsl");
pub const NORM_ROPE_WGSL: &str = include_str!("norm_rope.wgsl");
pub const NARROW_SPLIT_WGSL: &str = include_str!("narrow_split.wgsl");
pub const CONCAT_MANY_WGSL: &str = include_str!("concat_many.wgsl");
pub const GROUP_NORM_WGSL: &str = include_str!("group_norm.wgsl");
pub const LAYER_NORM2D_WGSL: &str = include_str!("layer_norm2d.wgsl");
pub const RMS_NORM_BWD_WGSL: &str = include_str!("rms_norm_backward.wgsl");
pub const LAYER_NORM_BWD_WGSL: &str = include_str!("layer_norm_backward.wgsl");
pub const CUMSUM_BWD_WGSL: &str = include_str!("cumsum_backward.wgsl");
pub const ROPE_BWD_WGSL: &str = include_str!("rope_backward.wgsl");
pub const GATHER_BWD_WGSL: &str = include_str!("gather_backward.wgsl");
pub const CUMSUM_WGSL: &str = include_str!("cumsum.wgsl");
pub const CUM_SCAN_WGSL: &str = include_str!("cum_scan.wgsl");
pub const FFT_GPU_WGSL: &str = include_str!("fft_gpu.wgsl");
/// native-gpu-fft: 32 KB on-chip radix-2/4/8 kernels (n<=4096) in a separate
/// module — only instantiated on devices with >=32 KB workgroup storage.
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_BIG_WGSL: &str = include_str!("fft_gpu_big.wgsl");
/// native-gpu-fft: portable 16 KB radix-4 kernel for n<=2048 (the default
/// on-chip path on wgpu — higher occupancy than the 32 KB module).
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_R4_16K_WGSL: &str = include_str!("fft_gpu_r4_16k.wgsl");
/// native-gpu-fft: multi-row on-chip FFT for small n (packs rows/workgroup).
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_MULTIROW_WGSL: &str = include_str!("fft_gpu_multirow.wgsl");
pub const COPY_WGSL: &str = include_str!("copy.wgsl");
/// Vector sibling of `copy.wgsl` — see the shader's own header for when the
/// lowering/encode paths take it and why it is a separate module.
pub const COPY_VEC4_WGSL: &str = include_str!("copy_vec4.wgsl");
/// Slab → arena move for the joined GGUF packed multiply: the two ends are
/// different buffers, so this is a dispatch rather than `copy.wgsl`'s move.
pub const COPY_FROM_SLAB_WGSL: &str = include_str!("copy_from_slab.wgsl");
/// Vector sibling of `copy_from_slab.wgsl` — see that kernel's own header.
pub const COPY_FROM_SLAB_VEC4_WGSL: &str = include_str!("copy_from_slab_vec4.wgsl");
/// Read-only stream probe for `bandwidth::stream_ceiling` — see the shader's
/// header for its geometry and for why it is not a copy.
pub const STREAM_READ_WGSL: &str = include_str!("stream_read.wgsl");
pub const CAST_WGSL: &str = include_str!("cast.wgsl");
pub const COMPLEX_CAST_WGSL: &str = include_str!("complex_cast.wgsl");
pub const BINARY_C64_WGSL: &str = include_str!("binary_c64.wgsl");
pub const COMPLEX_WIRINGER_WGSL: &str = include_str!("complex_wirtinger.wgsl");
pub const FFT_BUTTERFLY_STAGE_WGSL: &str = include_str!("fft_butterfly_stage.wgsl");
pub const ELEMENTWISE_REGION_WGSL: &str = include_str!("elementwise_region.wgsl");
pub const TRANSPOSE_WGSL: &str = include_str!("transpose.wgsl");
/// Vector sibling of `transpose.wgsl` — see the shader's own header for its
/// three modes and when the lowering takes each.
pub const TRANSPOSE_VEC4_WGSL: &str = include_str!("transpose_vec4.wgsl");
pub const NARROW_WGSL: &str = include_str!("narrow.wgsl");
/// Vector sibling of `narrow.wgsl` — see the shader's own header.
pub const NARROW_VEC4_WGSL: &str = include_str!("narrow_vec4.wgsl");
pub const CONCAT_WGSL: &str = include_str!("concat.wgsl");
/// Vector sibling of `concat.wgsl` — see the shader's own header.
pub const CONCAT_VEC4_WGSL: &str = include_str!("concat_vec4.wgsl");
/// The fused spelling of `narrow`-per-frame + `concat` — see the shader's own
/// header, and `backend::compile::lower`'s `detect_frames_pattern`.
pub const FRAMES_WGSL: &str = include_str!("frames.wgsl");

/// The invocation count per workgroup `frames.wgsl` declares and `run.rs`
/// dispatches with — one number, in two languages, pinned by
/// `frames_shader_matches_its_dispatch_geometry`.
pub const FRAMES_WORKGROUP: u32 = 64;
pub const GATHER_WGSL: &str = include_str!("gather.wgsl");
pub const GATHER_SPLIT_WGSL: &str = include_str!("gather_split.wgsl");
pub const GATHER_AXIS_WGSL: &str = include_str!("gather_axis.wgsl");
pub const ATTENTION_WGSL: &str = include_str!("attention.wgsl");
/// Banded (windowed) attention — the body [`attention_band_kernel`] prefixes
/// with the tile constants [`attention_band_form_at`] derives for the adapter's
/// thread count. Not a standalone module: it names `BM`, `C`, `DPER`, `THREADS`
/// and `BK`, which the host declares, and writes its staged tile as `%STAGE%`,
/// which the host substitutes for `vec4<f16>` (16-bit operands, 32-bit
/// accumulation) or `vec4<f32>` (the fallback on an adapter without
/// `Features::SHADER_F16` and the `RLX_WGPU_ATTENTION_BAND_F32` baseline). One
/// body, so the two forms cannot drift — they differ in that token and in whether
/// `enable f16;` leads the module.
pub const ATTENTION_BAND_WGSL: &str = include_str!("attention_band.wgsl");
pub const ATTENTION_BWD_WGSL: &str = include_str!("attention_bwd.wgsl");
pub const ROPE_WGSL: &str = include_str!("rope.wgsl");
pub const EXPAND_WGSL: &str = include_str!("expand.wgsl");
pub const REVERSE_WGSL: &str = include_str!("reverse.wgsl");
/// On-device ONNX ND indexing (GatherND / GatherElements / ScatterElements /
/// ScatterND + the scatter prologue). Five entry points over one 3-binding
/// layout: arena, `Params`, and the `meta` shape/stride buffer.
pub const INDEXING_ND_WGSL: &str = include_str!("indexing_nd.wgsl");
/// General (all-format) low-precision quantize / dequantize / decode-GEMM for
/// `Op::ScaledMatMul` and friends. Four entry points over the standard
/// arena + `Params` layout.
pub const SCALED_LOWP_WGSL: &str = include_str!("scaled_lowp.wgsl");
/// INT8 `Op::Quantize` / `Op::Dequantize` with an affine table.
pub const QUANT_I8_WGSL: &str = include_str!("quant_i8.wgsl");
/// Real INT8 `Op::QMatMul` (i32 accumulate + float requantize).
pub const Q_MATMUL_WGSL: &str = include_str!("q_matmul.wgsl");
/// Real INT8 `Op::QConv2d` (i32 accumulate + float requantize).
pub const Q_CONV2D_WGSL: &str = include_str!("q_conv2d.wgsl");
/// `Op::BatchNormInference` and its three backwards (channels-last).
pub const BATCH_NORM_INFERENCE_WGSL: &str = include_str!("batch_norm_inference.wgsl");
pub const ARGMAX_WGSL: &str = include_str!("argmax.wgsl");
pub const POOL2D_WGSL: &str = include_str!("pool2d.wgsl");
pub const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
pub const CONV1D_TILED_WGSL: &str = include_str!("conv1d_tiled.wgsl");
pub const IM2COL2D_WGSL: &str = include_str!("im2col2d.wgsl");
pub const CONV1D_IMPLICIT_WGSL: &str = include_str!("conv1d_implicit.wgsl");
pub const POOL1D_WGSL: &str = include_str!("pool1d.wgsl");
pub const POOL3D_WGSL: &str = include_str!("pool3d.wgsl");
pub const CONV1D_WGSL: &str = include_str!("conv1d.wgsl");
pub const CONV3D_WGSL: &str = include_str!("conv3d.wgsl");
pub const CONV3D_CTILE_WGSL: &str = include_str!("conv3d_ctile.wgsl");
pub const CONV_TRANSPOSE3D_WGSL: &str = include_str!("conv_transpose3d.wgsl");
pub const CONV_TRANSPOSE2D_TILED_WGSL: &str = include_str!("conv_transpose2d_tiled.wgsl");

/// The output-position and output-channel tile of `conv_transpose2d_tiled.wgsl`
/// (its `TILE_U`/`TILE_CO`, which WGSL cannot read from here), tied to the
/// shader's own declarations by the test at the end of this module.
/// [`conv_transpose2d_tiled_grid`] is the one place they become a dispatch.
const TILED_TILE_U: u32 = 64;
const TILED_TILE_CO: u32 = 64;

/// The dispatch grid `conv_transpose2d_tiled.wgsl` is launched with: one
/// workgroup per `TILED_TILE_U` output positions of one phase, one per
/// `TILED_TILE_CO` output channels, and one per phase (`sw` of them, the third
/// dimension). `lower` admits a form by asking whether this grid fits the
/// device's dimensions and `run` launches exactly it — the grid is the only
/// bound on which output elements a dispatch covers, so both call it once.
pub(crate) fn conv_transpose2d_tiled_grid(w_out: u32, sw: u32, c_out: u32) -> [u32; 3] {
    [
        w_out.div_ceil(sw).div_ceil(TILED_TILE_U),
        c_out.div_ceil(TILED_TILE_CO),
        sw,
    ]
}
pub const GROUP_NORM_BWD_WGSL: &str = include_str!("group_norm_backward.wgsl");
pub const AXIAL_ROPE2D_WGSL: &str = include_str!("axial_rope2d.wgsl");
pub const FAKE_QUANTIZE_WGSL: &str = include_str!("fake_quantize.wgsl");
pub const SCATTER_ADD_WGSL: &str = include_str!("scatter_add.wgsl");
pub const TOPK_WGSL: &str = include_str!("topk.wgsl");
pub const WELCH_PEAKS_GPU_WGSL: &str = include_str!("welch_peaks_gpu.wgsl");
pub const UMAP_KNN_WGSL: &str = include_str!("umap_knn.wgsl");
pub const GROUPED_MATMUL_WGSL: &str = include_str!("grouped_matmul.wgsl");
pub const SCALED_GROUPED_MATMUL_DECODE_WGSL: &str =
    include_str!("scaled_grouped_matmul_decode.wgsl");
/// The sample kernel's workgroup width. `sample.wgsl` spreads a one-row draw
/// across it and the dispatch is computed with it, so the two must be the same
/// number — a shader wider than its dispatch would leave part of the reduction
/// unset. The unit test below holds the shader's `LANES` to this.
pub const SAMPLE_LANES: u32 = 64;
/// The widest one-row draw `sample.wgsl` stages in workgroup storage (8 KiB, of
/// the 16 KiB every WebGPU device guarantees). A wider row keeps the
/// global-memory path; the unit test below holds the shader's `STAGE` to this.
pub const SAMPLE_STAGE: u32 = 2048;
pub const SAMPLE_WGSL: &str = include_str!("sample.wgsl");
pub const SELECTIVE_SCAN_WGSL: &str = include_str!("selective_scan.wgsl");
pub const GATED_DELTA_NET_WGSL: &str = include_str!("gated_delta_net.wgsl");
pub const MAMBA2_WGSL: &str = include_str!("mamba2.wgsl");
pub const GRU_WGSL: &str = include_str!("gru.wgsl");
pub const RNN_WGSL: &str = include_str!("rnn.wgsl");
pub const LSTM_WGSL: &str = include_str!("lstm.wgsl");
pub const DEQUANT_MATMUL_WGSL: &str = include_str!("dequant_matmul.wgsl");
pub const DEQUANT_MATMUL_MLX_WGSL: &str = include_str!("dequant_matmul_mlx.wgsl");
pub const DEQUANT_GGUF_WGSL: &str = include_str!("dequant_gguf.wgsl");
/// First byte of `dequant_gemv_gguf.wgsl`'s subgroup section (the marker line is
/// in the shader, spelled exactly once).
const GEMV_WIDE_MARKER: &str =
    "// >>> dequant_gemv_wide: subgroup-reduction arm (module split here)";

/// `needle`'s byte offset in `haystack`, at compile time: a `&str` constant
/// cannot be cut at a marker on stable Rust (`str::split_at`/`find` are not
/// `const`), and this file needs a cut rather than a second copy of the shader.
const fn marker_at(haystack: &[u8], needle: &[u8]) -> usize {
    let mut i = 0;
    while i + needle.len() <= haystack.len() {
        let mut j = 0;
        while j < needle.len() && haystack[i + j] == needle[j] {
            j += 1;
        }
        if j == needle.len() {
            return i;
        }
        i += 1;
    }
    panic!("dequant_gemv_gguf.wgsl: the `dequant_gemv_wide` marker is missing");
}

/// `dequant_gemv_gguf.wgsl` **up to** its subgroup section: the two shared-memory
/// one-row entry points (`dequant_gemv`, `dequant_gemv_tiled`).
///
/// The cut is not cosmetic. wgpu refuses a shader module that mentions a subgroup
/// built-in on a device without `Features::SUBGROUP`, and those two entry points
/// are created on every device — so they are built from a module that does not
/// contain the wide arm at all. [`DEQUANT_GEMV_GGUF_WIDE_WGSL`] is the whole
/// file, and is only built once `gguf_gpu::gemv_subgroup_probe` has found a width
/// on a device that carries the feature.
pub const DEQUANT_GEMV_GGUF_WGSL: &str = {
    const FULL: &str = include_str!("dequant_gemv_gguf.wgsl");
    const AT: usize = marker_at(FULL.as_bytes(), GEMV_WIDE_MARKER.as_bytes());
    // The marker is the first byte of an ASCII line in a UTF-8 source, so the
    // prefix is valid UTF-8 — proved rather than asserted, so no `unsafe` is
    // needed for a cut whose failure mode would be a `str` the reader cannot
    // trust.
    match std::str::from_utf8(FULL.as_bytes().split_at(AT).0) {
        Ok(shared) => shared,
        Err(_) => panic!("dequant_gemv_gguf.wgsl: the split is not on a char boundary"),
    }
};
/// The whole `dequant_gemv_gguf.wgsl`: the two shared-memory entry points and the
/// subgroup arm below the marker (see [`DEQUANT_GEMV_GGUF_WGSL`]).
pub const DEQUANT_GEMV_GGUF_WIDE_WGSL: &str = include_str!("dequant_gemv_gguf.wgsl");
/// The shader that reads the adapter's real subgroup width and how many of them
/// one 128-invocation workgroup (the wide arm's width) holds.
///
/// `AdapterInfo::subgroup_min_size`/`max_size` are buckets (Metal reports 4..64),
/// and the wide one-row arm maps columns onto whole subgroups, so it needs the
/// number itself. Two built-in values give it: `subgroup_size` and
/// `num_subgroups`. Dispatched once per process, mapped back by
/// `gguf_gpu::gemv_subgroup_probe`; only built on a device that carries
/// `Features::SUBGROUP`, so naming the built-ins in this module is safe.
///
/// The workgroup is 128 invocations — the wide arm's own width — because
/// `num_subgroups` counts the subgroups *in a workgroup*: at one invocation it
/// would read 1 on every device and measure nothing.
pub const GGUF_SUBGROUP_PROBE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> probe: array<u32>;

@compute @workgroup_size(128)
fn gemv_subgroup_probe(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg: u32,
    @builtin(num_subgroups) n: u32,
) {
    if (lid.x == 0u) {
        probe[0] = sg;
        probe[1] = n;
    }
}
"#;
pub const DEQUANT_GEMM_Q1_0_WGSL: &str = include_str!("dequant_gemm_q1_0.wgsl");
pub const DEQUANT_GEMM_GGUF_WGSL: &str = include_str!("dequant_gemm_gguf.wgsl");
pub const FUSED_RESIDUAL_LN_WGSL: &str = include_str!("fused_residual_ln.wgsl");
pub const FUSED_RESIDUAL_LN_TEE_VEC4_WGSL: &str = include_str!("fused_residual_ln_tee_vec4.wgsl");
pub const FUSED_RESIDUAL_LN_TEE_WGSL: &str = include_str!("fused_residual_ln_tee.wgsl");
pub const FUSED_RESIDUAL_RMS_NORM_WGSL: &str = include_str!("fused_residual_rms_norm.wgsl");
pub const ADA_LAYER_NORM_WGSL: &str = include_str!("ada_layer_norm.wgsl");
pub const GATED_RESIDUAL_WGSL: &str = include_str!("gated_residual.wgsl");
pub const ADA_LAYER_NORM_BACKWARD_WGSL: &str = include_str!("ada_layer_norm_backward.wgsl");
pub const GATED_RESIDUAL_BACKWARD_WGSL: &str = include_str!("gated_residual_backward.wgsl");
pub const MATMUL_QKV_WGSL: &str = include_str!("matmul_qkv.wgsl");
pub const MATMUL_QKV_COOP_F32_WGSL: &str = include_str!("matmul_qkv_coop_f32.wgsl");
pub const SNAKE_BETA_WGSL: &str = include_str!("snake_beta.wgsl");
/// Columns one `snake_beta.wgsl` workgroup covers, and so the width the run's
/// dispatch divides a row by. It must be the shader's own `@workgroup_size`; the
/// unit test below holds the two to each other.
pub const SNAKE_BETA_WG: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub batch: u32,
    pub a_batch_stride: u32,
    pub b_batch_stride: u32,
    pub c_batch_stride: u32,
    pub has_bias: u32,
    pub bias_off: u32,
    pub act_id: u32, // 0xFFFF = no activation
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// One fused DAC snake, `y = x + beta·sin²(alpha·x)` — see `snake_beta.wgsl`.
/// `has_bias` is a fold flag, not a shape: the lowering passes 0 for a step it
/// did not fold a bias into. Eight words, so the uniform struct is 16-byte
/// aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SnakeBetaParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub alpha_off: u32,
    pub beta_off: u32,
    pub bias_off: u32,
    pub out_off: u32,
    pub has_bias: u32,
}

/// Shared layout for binary, compare. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub op: u32,
    /// Per-operand broadcast — output element `i` reads `(i / rep) % len`.
    /// `len == 0` is dense. See `binary_main.wgsl`.
    pub a_rep: u32,
    pub a_len: u32,
    pub b_rep: u32,
    pub b_len: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for unary kernel. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UnaryParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for where (3-input select). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhereParams {
    pub n: u32,
    pub cond_off: u32,
    pub x_off: u32,
    pub y_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for fma (3-input fused multiply-add). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FmaParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for ReluBackward / ActivationBackward. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ActivationBackwardParams {
    pub n: u32,
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for reductions. 32 bytes.
///
/// Supports arbitrary-axis reductions. The reduce kernel walks the
/// input as a 3D tensor `[outer, reduce_dim, inner]` where:
///   * `outer` = product of dims BEFORE the reduce axis
///   * `reduce_dim` = the reduce axis itself
///   * `inner` = product of dims AFTER the reduce axis (=1 for the
///     last-axis case, which is what the v3 dispatcher emitted).
/// Output shape is `[outer, inner]` (or with the reduce axis kept as 1
/// when `keep_dim`; the dispatcher handles the shape arithmetic).
#[repr(C)]
pub struct ReduceParams {
    pub outer: u32,
    pub reduce_dim: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
}

// Manual impls to avoid issues with structural derives if any field
// arrangement subtly trips bytemuck.
unsafe impl Pod for ReduceParams {}
unsafe impl Zeroable for ReduceParams {}
impl Copy for ReduceParams {}
impl Clone for ReduceParams {
    fn clone(&self) -> Self {
        *self
    }
}
impl std::fmt::Debug for ReduceParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReduceParams {{ outer: {}, reduce_dim: {}, inner: {}, op: {} }}",
            self.outer, self.reduce_dim, self.inner, self.op
        )
    }
}

/// Layout for softmax. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SoftmaxParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for `attention_mask` — the many-row attention route's masked scale.
/// 64 bytes: the 14 fields the pass reads, padded to 16 `u32`.
///
/// Its own struct rather than the fused entry's `Params`: that one is 36 `u32`
/// of Q/K/V/output offsets and per-tensor strides this pass reads none of, and
/// it has no field for the score matrix the pass scales in place. What the two
/// do share is the bind-group shape (arena + one uniform), which is why
/// `attention.wgsl` carries a binding per entry point.
///
/// `rows × cols` is the whole score matrix: rows are `(batch, head, query row)`
/// major, which is the order the dense multiply wrote them in, and `seq_q` /
/// `heads` decompose a row index back into the mask address the fused kernel
/// derives for the same `(b, h, qi)` triple. `scale_bits` is `1/sqrt(D)` bitcast,
/// applied before the cap and the mask exactly as the fused kernel applies it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttentionMaskParams {
    pub rows: u32,
    /// The key sequence — the last axis the mask pass walks, also the fused
    /// kernel's `seq_k`.
    pub cols: u32,
    pub seq_q: u32,
    pub heads: u32,
    pub scale_bits: u32,
    /// 0=None, 1=Causal, 2=Custom(binary), 3=SlidingWindow, 4=Bias(additive),
    /// 5=two-sided band (`MaskKind::Window`, and the id a dense square step takes
    /// when `plan_attention` routes it through the band kernel with `window` at
    /// the key extent). 5 is never read by this pass.
    pub mask_kind: u32,
    /// SlidingWindow width (only read when `mask_kind == 3`).
    pub window: u32,
    /// The score matrix's own base, read and written in place.
    pub scores_off: u32,
    pub mask_off: u32,
    pub mask_batch_stride: u32,
    pub mask_head_stride: u32,
    pub seq_q_stride: u32,
    pub seq_k_stride: u32,
    /// `attn_logit_softcap` bitcast: `c·tanh(s/c)` over the scaled score,
    /// applied before the mask. 0 or below disables.
    pub logit_softcap_bits: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for the fused dense softmax cross-entropy. 32 bytes.
/// Also used by integer-label `SoftmaxCrossEntropyWithLogits` (labels in
/// `targets_off`, one f32 per row).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceParams {
    pub outer: u32,
    pub inner: u32,
    pub logits_off: u32,
    pub targets_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for integer-label softmax cross-entropy backward. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub logits_off: u32,
    pub labels_off: u32,
    pub d_loss_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for MaxPool2d backward. 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaxPool2dBwdParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub _p0: u32,
}

/// Layout for MaxPool3d backward. 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaxPool3dBwdParams {
    pub n: u32,
    pub c: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Layout for Conv3d BackwardInput. 112 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dBwdInputParams {
    pub n: u32,
    pub c_in: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub c_out: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub dd: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub dy_off: u32,
    pub w_off: u32,
    pub dx_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for Conv3d BackwardWeight. 112 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dBwdWeightParams {
    pub n: u32,
    pub c_in: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub c_out: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub dd: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub x_off: u32,
    pub dy_off: u32,
    pub dw_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GroupNorm (NCHW) backward. 48 bytes. Shared by dx/dgamma/dbeta
/// entry points (`group_norm_bwd_{input,gamma,beta}`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroupNormBwdParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub num_groups: u32,
    pub eps_bits: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for AxialRope2d. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AxialRope2dParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub end_x: u32,
    pub end_y: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub repeat_factor: u32,
    pub theta: f32,
    pub in_off: u32,
    pub out_off: u32,
    pub n_total: u32,
}

/// Layout for FakeQuantize (Fixed / PerBatch). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FakeQuantizeParams {
    pub n: u32,
    pub chan_dim: u32,
    pub inner: u32,
    pub q_max: f32,
    pub in_off: u32,
    pub scale_off: u32,
    pub out_off: u32,
    pub _pad: u32,
}

/// Layout for the native NCHW `GroupNorm` kernel. 48 bytes.
///
/// One workgroup per `(batch, group)`; `gamma`/`beta` are indexed per channel,
/// so the kernel needs `c` and `hw` to recover the channel from a flat index.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroupNormParams {
    pub groups_total: u32,
    pub group_elems: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub eps_bits: u32,
    pub num_groups: u32,
    pub c: u32,
    pub hw: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Layout for the native NCHW `LayerNorm2d` kernel. Mirrors the WGSL `Params`;
/// `positions = n * hw` (one workgroup per position). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayerNorm2dParams {
    pub positions: u32,
    pub c: u32,
    pub hw: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub eps_bits: u32,
}

/// Layout for LayerNorm / RmsNorm.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayerNormParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub eps_bits: u32, // bitcast::<u32>(eps)
    pub op: u32,       // 0=LayerNorm, 1=RmsNorm
    /// 1 = the `layernorm_vec4` module (256-lane workgroup, 16 B accesses).
    pub vec4: u32,
}

/// Layout for the fused RmsNorm + RoPE step: two head-row parts that share one
/// position table, normalized and rotated in one dispatch, plus the split's
/// leftover piece copied verbatim when the step also owns the cut that fed it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NormRopeParams {
    /// Positions the run walks (`m`); the active extent scales it.
    pub seq: u32,
    /// 0 ⇒ that part is absent.
    pub heads_a: u32,
    pub heads_b: u32,
    /// Head width (even); the rotated pair is `(j, j + inner / 2)`.
    pub inner: u32,
    pub cos_row_stride: u32,
    pub eps_bits: u32,
    /// The split's other piece, copied at `copy_len` elements per row; 0 ⇒ the
    /// step took no split with it.
    pub copy_in: u32,
    pub copy_out: u32,
    pub copy_len: u32,
    pub in_a: u32,
    pub out_a: u32,
    pub gamma_a: u32,
    pub beta_a: u32,
    pub in_b: u32,
    pub out_b: u32,
    pub gamma_b: u32,
    pub beta_b: u32,
    pub cos_off: u32,
    pub sin_off: u32,
}

/// LayerNorm backward kernel params (f32 element offsets). Shared by
/// the three entry points; the dispatcher picks `layer_norm_bwd_input`,
/// `layer_norm_bwd_gamma_partial`, or `layer_norm_bwd_gamma_reduce`
/// based on which Step variant fired. dbeta isn't a dedicated op — it's
/// a plain `Reduce::Sum` over the batch dim of `dy`, handled by the
/// general reduce kernel.
///
/// `scratch_off` is the f32-element offset of the tail scratch zone
/// (only used by the gamma partial/reduce kernels). For the reduce
/// kernel `outer` carries the number of partial chunks emitted by the
/// partial kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayerNormBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub scratch_off: u32,
}

/// RMSNorm backward kernel params (f32 element offsets). `wrt`: 0=dx, 1=dgamma, 2=dbeta.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RmsNormBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub wrt: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumsumBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub exclusive: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RopeBwdParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub head_dim: u32,
    pub n_rot: u32,
    pub dy_off: u32,
    pub cos_off: u32,
    pub sin_off: u32,
    pub dx_off: u32,
    pub cos_len: u32,
    /// The cos/sin table's own last dimension — see the WGSL `tab_off` note.
    pub cos_row_stride: u32,
    /// GptJ pairing (adjacent lanes) rather than NeoX rotate-half; 0 or 1.
    /// Must match the forward `Rope` this is the adjoint of.
    pub interleaved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherBwdParams {
    pub outer: u32,
    pub axis_dim: u32,
    pub num_idx: u32,
    pub trailing: u32,
    pub dy_off: u32,
    pub idx_off: u32,
    pub dst_off: u32,
    pub _p0: u32,
}

/// Layout for cumsum. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumsumParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub exclusive: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for cum_scan (cumprod / cummax). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumScanParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub exclusive: u32,
    pub is_max: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for FFT. 32 bytes. Matches `fft.wgsl::Params`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftParams {
    pub src_off: u32,
    pub dst_off: u32,
    pub n: u32,
    pub log2n: u32,
    pub inverse: u32,
    pub norm_scale: f32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Uniform block for multi-kernel FFT (`fft_gpu.wgsl::Params`). 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftGpuParams {
    pub off: u32,
    pub dst_off: u32,
    pub n: u32,
    pub log2n: u32,
    pub inverse: u32,
    pub norm_scale: f32,
    pub outer: u32,
    pub tile: u32,
    pub inner_stages: u32,
    pub q_or_hs: u32,
}

/// PLAN L2 — interpreted N-ary element-wise region. Chain encoded
/// as 4 u32s per step (op_kind, op_sub, lhs_enc, rhs_enc). Operand
/// encoding: bit 31 = src kind (0=Input, 1=Step), bits 0..30 = index.
/// `scalar_input_mask` is the per-input scalar fast-path bitfield;
/// `input_modulus[i]` is the per-input element count for trailing-
/// shape broadcast (`0` ⇒ no broadcast, kernel reads gid; `>0` ⇒
/// kernel reads `gid % input_modulus[i]`). Fixed cap at 32 steps +
/// 16 inputs (ample for chains rlx produces). 12 padding bytes
/// after `scalar_input_mask` align the next array on WGSL's
/// 16-byte uniform alignment boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ElementwiseRegionParams {
    pub len: u32,
    pub num_inputs: u32,
    pub num_steps: u32,
    pub dst_off: u32,
    pub input_offs: [u32; 16],
    pub chain: [u32; 128], // 32 steps * 4 u32s
    pub scalar_input_mask: u32,
    pub prologue: u32,
    pub out_n: u32,
    pub out_c: u32,
    pub out_h: u32,
    pub out_w: u32,
    pub prologue_input: u32,
    pub input_modulus: [u32; 16],
}

/// FKL batch region: `batch_input_offs[slice]` + shared chain (no prologue).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BatchElementwiseRegionParams {
    pub slice_len: u32,
    pub num_batch: u32,
    pub num_steps: u32,
    pub base_dst_off: u32,
    pub slice_elems: u32,
    pub batch_input_offs: [u32; 64],
    pub chain: [u32; 128],
    pub scalar_input_mask: u32,
    pub input_modulus: [u32; 16],
}

/// Layout for a numeric `Op::Cast` (`cast.wgsl`). 32 bytes. `mode`:
/// 0 identity, 1 float→int (trunc + saturate to `[lo_bits, hi_bits]`, NaN→0),
/// 2 →Bool (`value != 0`). `lo_bits`/`hi_bits` are f32-as-u32 clamp bounds.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CastParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub mode: u32,
    pub lo_bits: u32,
    pub hi_bits: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Standalone complex `Op::Cast` (`complex_cast.wgsl`). 32 bytes.
/// `mode` selects one of the six lane-move directions (real↔C64, real↔C128,
/// C64↔C128); `n` is the complex-element count; offsets are f32-element.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ComplexCastParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub mode: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// C64 element-wise binary (`binary_c64.wgsl`). 32 bytes. `n` is the output
/// complex-element count; `n_a`/`n_b` are the operands' complex-element counts
/// (broadcast via `k % n_x`); offsets are f32-element (lane `2*m + j`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryC64Params {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub op: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub _p0: u32,
}

/// C64 Wirtinger surface (`complex_wirtinger.wgsl`). 32 bytes.
/// Shared by ComplexNormSq / ComplexNormSqBackward / Conjugate:
/// - NormSq / Conjugate: `a_off`=src, `c_off`=dst (`b_off` unused)
/// - NormSqBackward: `a_off`=z, `b_off`=g, `c_off`=dz
/// `n` is the complex-element count; offsets are f32-element.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ComplexWirtingerParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Ternary-pruned radix-2 butterfly (`fft_butterfly_stage.wgsl`). 48 bytes.
/// Offsets are f32-element; `half = n_fft / 2`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftButterflyStageParams {
    pub batch: u32,
    pub n_fft: u32,
    pub stage: u32,
    pub half: u32,
    pub state_off: u32,
    pub out_off: u32,
    pub gate_off: u32,
    pub rev_off: u32,
    pub tw_re_off: u32,
    pub tw_im_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout shared by Reshape / same-dtype Cast / generic full copy. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CopyParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Uniform layout for the slab → arena move (`copy_from_slab.wgsl`). 16 bytes.
/// The offsets are f32 words from each bound window's 256 B-aligned base.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CopyFromSlabParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
}

/// Uniform layout for the on-device Philox RNG kernels (`rng_philox.wgsl`).
/// `a`/`b` are `mean`/`scale` (normal) or `low`/`high` (uniform); unused by
/// `rng_fill_zero`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RngParams {
    pub n: u32,
    pub out_off: u32,
    pub a: f32,
    pub b: f32,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Uniform layout for the native `fused_swiglu.wgsl` kernel. `out = up *
/// silu(gate)` from a `[.., 2*n_half]` input; `outer` = rows (total output
/// = `outer*n_half`). Offsets are relative to the bound arena window.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedSwiGLUParams {
    pub n_half: u32,
    pub outer: u32,
    pub gate_first: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Uniform layout for `fused_conv_bias_act.wgsl` — the Conv2d params plus the
/// fused bias/activation/residual epilogue. Offsets are window-relative.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedConvBiasActParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    pub has_bias: u32,
    pub bias_off: u32,
    pub act_id: u32,
    pub has_residual: u32,
    pub residual_off: u32,
}

/// Layout for transpose (uses the 3-binding bind layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TransposeParams {
    pub rank: u32,
    pub out_total: u32,
    pub in_off: u32,
    pub out_off: u32,
    /// PLAN L1 — precomputed at compile time. `1` when `perm[0] == 0`
    /// (= bucket axis stays at output axis 0). Active-extent path
    /// scales `out_total` proportionally only when this is `1`.
    pub bucket_outermost: u32,
    /// PLAN L1 — `out_dims[0]` for active-extent scaling math.
    pub out_dim_0: u32,
    /// `0` keeps the scalar `transpose.wgsl` kernel; `1`/`2`/`3` select
    /// `transpose_vec4_kernel`'s general-last-axis / `[0,2,1,3]` /
    /// `[0,2,1]` modes. See `transpose_vec4.wgsl`.
    pub mode: u32,
    /// Vector modes: the y workgroup count. Mode 1 also linearizes its flat
    /// outer index with it (`gid.y + gid.z * grid_y`).
    pub grid_y: u32,
    /// Vector modes: the z workgroup count.
    pub grid_z: u32,
    /// Vector modes: the contiguous innermost extent in elements (a multiple
    /// of 4). The x grid is `ceil(inner / 4 / 64)`.
    pub inner: u32,
}

/// Layout for narrow / concat (the same struct serves both).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NarrowConcatParams {
    pub total: u32, // total elements (output for narrow, input for concat)
    pub outer: u32,
    pub inner: u32,
    pub axis_in_size: u32,
    pub axis_out_size: u32,
    pub start: u32,
    pub in_off: u32,
    pub out_off: u32,
    /// `1` selects the `*_vec4` kernel (one 16 B vector per thread). Set only
    /// when `inner % 4 == 0` and both offsets are 16 B aligned — see
    /// `concat_vec4.wgsl` / `narrow_vec4.wgsl` and `vec4_move_ok`.
    pub vec4: u32,
}

/// Layout for the split copy (`narrow_split.wgsl`): one row cut into `count`
/// contiguous last-axis pieces, each into its own slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NarrowSplitParams {
    pub src: u32,
    /// Elements in the source row — the sum of the pieces.
    pub total: u32,
    /// End of piece 0 and of piece 1 (piece 2 runs to `total`).
    pub end0: u32,
    pub end1: u32,
    pub dst0: u32,
    pub dst1: u32,
    pub dst2: u32,
    /// 2 or 3.
    pub count: u32,
}

/// Most inputs one `Step::ConcatMany` folds (the loop's largest is 15).
pub const CONCAT_MANY_MAX: usize = 16;

/// Layout for the whole-concat copy (`concat_many.wgsl`): every input of one
/// `Op::Concat`, in output order, copied by one dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ConcatManyParams {
    /// Product of the trailing dims — the row width below the concat axis.
    pub inner: u32,
    /// The output's first word, as a bind-window offset.
    pub out_off: u32,
    /// Inputs, in output order.
    pub count: u32,
    /// Pads `ins` to its `vec4` stride; the struct is 272 B, the size the shader
    /// declares.
    pub _pad: u32,
    /// `(in_off, axis_len, start)` per input, in output order. `start` is the
    /// piece's planned offset along the axis, which a walked run keeps so the
    /// pieces do not compact; `.w` is the `vec4` stride's spare lane.
    pub ins: [[u32; 4]; CONCAT_MANY_MAX],
}

/// Frame-matrix dispatch parameters: `narrow`-per-frame + `concat` folded into
/// one dispatch (`frames.wgsl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FramesParams {
    pub src_off: u32,
    pub dst_off: u32,
    pub t: u32,
    /// The source's leading-dims product, which the destination keeps as its middle
    /// axis — and `1` in every schedule this fold emits: a source with rows of its
    /// own keeps its per-frame steps instead (`detect_frames_pattern`), because the
    /// mode the step scales trims a row axis.
    pub outer: u32,
    pub frame_len: u32,
    pub hop: u32,
    /// Elements to cover: `outer · n_frames · frame_len`, the row-major prefix of
    /// the destination in (frame, leading dim, position) order — and the one word
    /// the active-extent mode scales (see `safe_for_active_extent` in `step.rs`).
    pub total: u32,
    /// Uniform stride padding, as in this crate's other param structs.
    pub _p0: u32,
}

/// Layout for gather.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherParams {
    pub n_out: u32,
    pub n_idx: u32,
    pub dim: u32,
    pub vocab: u32,
    pub in_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
    pub _p0: u32,
}

/// Layout for gather along a non-zero axis.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherAxisParams {
    pub total: u32,
    pub outer: u32,
    pub axis_dim: u32,
    pub num_idx: u32,
    pub trailing: u32,
    pub table_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
}

/// Layout for fused SDPA.
///
/// Per-tensor (Q, K, V, output) strides are passed explicitly so the
/// kernel can read either canonical [B, H, S, D] or transposed
/// [B, S, H, D] without inserting upstream Transpose dispatches. The
/// layout-elimination saves ~24 transpose dispatches per BERT-L6
/// forward (one per Q/K/V/output × layers), each ~50µs at small batch.
///
/// The `seq_q_stride` / `seq_k_stride` fields are retained because
/// they describe the MASK layout `[B, H, S_q, S_k]` (separate from
/// Q/K/V layout), used by `MaskKind::Custom`.
///
/// 144 bytes (36 u32s); WebGPU uniform-buffer 16-byte alignment OK.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttentionParams {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub out_off: u32,
    pub mask_off: u32,
    pub mask_kind: u32,
    pub scale_bits: u32,
    pub window: u32,
    /// MASK address strides. Mask address math (per-element):
    ///   addr = mask_off
    ///        + b  * mask_batch_stride
    ///        + h  * mask_head_stride
    ///        + qi * seq_q_stride         (per-query stride)
    ///        + s  * seq_k_stride         (per-key   stride)
    /// Setting some strides to 0 lets the kernel read a *broadcast*
    /// mask without materializing the broadcast. e.g. BERT padding mask
    /// `[B, S]`: mask_batch_stride=S, mask_head_stride=0, seq_q_stride=0,
    /// seq_k_stride=1. Saves the Expand pre-pass that unfuse used to
    /// emit per attention block.
    pub seq_q_stride: u32,
    pub seq_k_stride: u32,
    pub mask_batch_stride: u32,
    pub mask_head_stride: u32,
    /// GQA/MQA: number of key/value heads that the query heads share. Equals
    /// `heads` for plain MHA; 0 means unset and the shader falls back to MHA.
    pub kv_heads: u32,
    /// Asymmetric SDPA (DeepSeek/Kimi MLA): V rows + output are `v_head_dim`
    /// wide while the Q·K score contracts over `head_dim`. Equals `head_dim`
    /// for ordinary attention; 0 makes the shader fall back to `head_dim`.
    pub v_head_dim: u32,
    /// `attn_logit_softcap` bitcast: `c·tanh(s/c)` over the scaled score,
    /// applied before the mask. 0 or below disables.
    pub logit_softcap_bits: u32,

    // Q stride triple (in f32 elements). For [B, H, S, D]:
    //   q_batch_stride = H·S·D, q_head_stride = S·D, q_seq_stride = D
    // For [B, S, H, D]:
    //   q_batch_stride = S·H·D, q_head_stride = D,   q_seq_stride = H·D
    pub q_batch_stride: u32,
    pub q_head_stride: u32,
    pub q_seq_stride: u32,
    /// Key-range splits this step runs as: 1 (the default) is one workgroup per
    /// (batch, head), `n` is `n` workgroups over disjoint key ranges plus one
    /// merge dispatch. Never 0 — the lowerer writes
    /// `attention_route::attention_decode_key_splits`'s return, which is 1 or
    /// more — and read only by the decode entry and its merge.
    pub splits: u32,

    pub k_batch_stride: u32,
    pub k_head_stride: u32,
    pub k_seq_stride: u32,
    pub _pad_k: u32,

    pub v_batch_stride: u32,
    pub v_head_stride: u32,
    pub v_seq_stride: u32,
    pub _pad_v: u32,

    pub o_batch_stride: u32,
    pub o_head_stride: u32,
    pub o_seq_stride: u32,
    pub _pad_o: u32,
}

/// Layout for `attention_bwd.wgsl` — forward strides + `dy_off` + `wrt`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttentionBwdParams {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub mask_off: u32,
    pub mask_kind: u32,
    pub scale_bits: u32,
    pub window: u32,
    pub wrt: u32,
    pub seq_q_stride: u32,
    pub seq_k_stride: u32,
    pub mask_batch_stride: u32,
    pub mask_head_stride: u32,
    pub _pad_mask_0: u32,
    pub _pad_mask_1: u32,
    pub _pad_mask_2: u32,
    pub q_batch_stride: u32,
    pub q_head_stride: u32,
    pub q_seq_stride: u32,
    pub _pad_q: u32,
    pub k_batch_stride: u32,
    pub k_head_stride: u32,
    pub k_seq_stride: u32,
    pub _pad_k: u32,
    pub v_batch_stride: u32,
    pub v_head_stride: u32,
    pub v_seq_stride: u32,
    pub _pad_v: u32,
    pub o_batch_stride: u32,
    pub o_head_stride: u32,
    pub o_seq_stride: u32,
    pub _pad_o: u32,
}

/// Layout for Rope.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RopeParams {
    pub n_total: u32,
    pub seq: u32,
    pub head_dim: u32,
    pub half: u32,
    pub in_off: u32,
    pub cos_off: u32,
    pub sin_off: u32,
    pub out_off: u32,
    pub last_dim: u32,
    /// PLAN L1 — set at compile time. Together with `seq_stride`,
    /// lets the WGSL kernel decompose iteration index into
    /// `(bi, si, d)` while indexing into the underlying full-extent
    /// buffer. `n_total` is the runtime-scaled iteration bound;
    /// `seq_stride` is the compile-time-fixed full seq for stride.
    pub batch: u32,
    pub seq_stride: u32,
    /// RoPE pairing flavor: `0` = NeoX rotate-half `(i, i+half)`, `1` = GPT-J /
    /// llama.cpp-NORM interleaved adjacent pairs `(2i, 2i+1)`. GGUF Llama weights
    /// are permuted for the GPT-J layout, so GGUF-backed decode needs `style=1`.
    pub style: u32,
    /// Partial rotary: half of `n_rot` (the rotated width). Dims `[n_rot,
    /// head_dim)` are copied through unchanged. Equals `half` for full rotation
    /// (n_rot == head_dim); smaller for p-RoPE (Gemma 4 global layers).
    pub rot_half: u32,
    /// Elements per row of the cos/sin tables — the table's OWN last dimension
    /// (`rlx_ir::shape::rope_table_stride`), which is not `rot_half`.
    ///
    /// A table allocated at `head_dim / 2` and only partly used under partial
    /// rotation has a wider row than `n_rot / 2`; indexing it by `rot_half`
    /// reads the wrong row for every position past the first, while still
    /// producing a correctly-normed rotation, so it looks right. The CPU kernel
    /// carries the same field for the same reason; the shader used to assume
    /// `rot_half` and silently disagreed with CPU whenever `n_rot < head_dim`.
    pub cos_row_stride: u32,
}

/// Layout for Expand. Mirrors TransposeParams (rank, total, offsets);
/// per-axis dims/strides ride in the meta storage buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ExpandParams {
    pub rank: u32,
    pub out_total: u32,
    pub in_off: u32,
    pub out_off: u32,
    /// PLAN L1 — precomputed at compile time. `1` when the bucket
    /// axis stays at output axis 0 after the expand mapping.
    pub bucket_outermost: u32,
    /// PLAN L1 — `out_dims[0]` for active-extent scaling math.
    pub out_dim_0: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for Reverse. Mirrors the WGSL `Params`; per-axis dims/strides/flags
/// ride in the meta storage buffer. `scale_leading` is the PLAN L1 flag. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ReverseParams {
    pub rank: u32,
    pub total: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub scale_leading: u32,
    pub dim_0: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for the two `quant_i8.wgsl` entry points.
///
/// `affine` is `vec4`-strided on purpose: a scalar array in a WGSL *uniform*
/// buffer has a 16-byte stride, so a tightly packed host-side table would be
/// read from the wrong offsets. The Vulkan twin shipped exactly that bug.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QuantI8Params {
    pub n: u32,
    pub chan_dim: u32,
    pub inner: u32,
    /// `x` f32-element offset (quantize) or code byte offset (dequantize).
    pub a_off: u32,
    /// Code byte offset (quantize) or `out` f32-element offset (dequantize).
    pub out_off: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// `[scale_bits, zero_point]` per channel, two channels per `vec4`.
    pub affine: [u32; 24],
}

/// Channels whose affine pair fits [`QuantI8Params`]. Wider tensors keep the
/// CPU host route, matching `QUANTIZE_I8_MAX_CHAN` on Vulkan.
pub const QUANT_I8_MAX_CHAN: usize = 12;

/// Layout for `q_matmul.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QMatMulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Byte offsets for the packed i8 operands.
    pub x_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    /// f32-element offset.
    pub bias_off: u32,
    pub x_zp: i32,
    pub w_zp: i32,
    pub out_zp: i32,
    pub mult: f32,
    pub _pad0: u32,
}

/// Layout for the four `batch_norm_inference.wgsl` entry points.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BatchNormInferenceParams {
    /// Total elements (forward / bwd_input) — `count * channels`.
    pub n: u32,
    /// Rows (bwd_gamma / bwd_beta).
    pub count: u32,
    pub channels: u32,
    pub eps: f32,
    pub src_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub mean_off: u32,
    pub var_off: u32,
    pub dy_off: u32,
    pub dst_off: u32,
    pub _pad0: u32,
}

/// Layout for `q_conv2d.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QConv2dParams {
    pub batch: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    /// Byte offsets for the packed i8 operands.
    pub x_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    /// f32-element offset.
    pub bias_off: u32,
    pub x_zp: i32,
    pub w_zp: i32,
    pub out_zp: i32,
    pub mult: f32,
}

/// Layout for the four `scaled_lowp.wgsl` entry points.
///
/// One struct for all of them: they share a bind-group layout, and a single
/// layout is what keeps the WGSL in step with `scaled_lowp_general.cu`'s
/// argument lists. Fields a given entry point does not use are zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScaledLowpParams {
    /// lhs codes (byte off) | `x` (f32 elem off) | codes (byte off).
    pub a_byte_off: u32,
    /// rhs codes (byte off).
    pub b_byte_off: u32,
    pub a_scale_byte_off: u32,
    pub b_scale_byte_off: u32,
    /// f32 element offset, or a byte offset for `scaled_quantize`.
    pub out_off: u32,
    pub bias_off: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub rows: u32,
    pub cols: u32,
    /// `ScaledFormat::kernel_id()` of the lhs / quantized tensor.
    pub a_fmt: u32,
    /// `ScaledFormat::kernel_id()` of the rhs.
    pub b_fmt: u32,
    /// 0 per-tensor f32, 1 block E8M0, 2 NVFP4 E4M3.
    pub scale_mode: u32,
    pub block: u32,
    pub has_bias: u32,
}

/// Layout for the five `indexing_nd.wgsl` entry points.
///
/// One struct for all of them rather than five: they share a bind-group layout
/// and a dispatch shape, and a single layout is also what keeps the WGSL in
/// step with `indexing_nd.cu`'s argument list. Fields not used by a given entry
/// point are zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndexingNdParams {
    /// Thread count.
    pub n: u32,
    pub data_off: u32,
    pub idx_off: u32,
    /// Scatter only.
    pub upd_off: u32,
    pub dst_off: u32,
    /// Scatter bounds guard; carries `data_len` for GatherElements.
    pub dst_len: u32,
    /// GatherElements / ScatterElements.
    pub rank: u32,
    /// GatherElements / ScatterElements.
    pub axis: u32,
    /// GatherElements wrap/clamp bound.
    pub axis_dim: i32,
    /// GatherND / ScatterND tuple width.
    pub k: u32,
    /// GatherND / ScatterND contiguous run per tuple.
    pub slice: u32,
    /// GatherND.
    pub tuples_per_batch: u32,
    /// GatherND.
    pub batch_stride: u32,
    /// Always 0 on wgpu — the WGSL scatters are overwrite-only because core
    /// WGSL has no f32 atomics. Present so this layout matches the shared
    /// launch plan (and the CUDA argument list) rather than silently diverging.
    pub reduction: u32,
    /// `copy_sanitize` only.
    pub src_len: u32,
    /// `copy_sanitize` only: bit0 = do_copy, bit1 = do_sanitize.
    pub flags: u32,
}

/// Layout for argmax (matches Reduce shape).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ArgmaxParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for Pool2D NCHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool2dParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for Conv2D NCHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv2dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
}

/// Layout for GPU im2col (NCHW, N==1, groups==1). 80 bytes (20 u32).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Im2Col2dParams {
    pub c_in: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub in_off: u32,
    pub col_off: u32,
    pub k_total: u32,
    pub spatial: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for Pool1D NCL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool1dParams {
    pub n: u32,
    pub c: u32,
    pub l: u32,
    pub l_out: u32,
    pub kl: u32,
    pub sl: u32,
    pub pl: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
}

/// Layout for Pool3D NCDHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool3dParams {
    pub n: u32,
    pub c: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for Conv1D NCL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv1dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub l: u32,
    pub l_out: u32,
    pub kl: u32,
    pub sl: u32,
    pub pl: u32,
    pub dl: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for dequant_gguf. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGgufParams {
    pub w_byte_off: u32,
    pub dst_f32_off: u32,
    pub scheme_id: u32,
    pub num_blocks: u32,
}

/// Layout for the fused GGUF K-quant GEMV (`dequant_gemv_gguf.wgsl`). 32 bytes.
/// Offsets are relative to each kernel binding's windowed base.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Hash)]
pub struct DequantGemvGgufParams {
    pub k: u32,
    pub n: u32,
    pub scheme_id: u32,
    pub x_f32_off: u32,
    pub w_byte_off: u32,
    pub out_f32_off: u32,
    /// Padding to the 32 bytes a WGSL uniform struct of eight words occupies
    /// (its alignment is 16), so the record written and the record read agree.
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for fused Q1_0 GEMM (`dequant_gemm_q1_0.wgsl`). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGemmQ10Params {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub x_f32_off: u32,
    pub w_byte_off: u32,
    pub out_f32_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for the fused multi-row K-quant GEMM (`dequant_gemm_gguf.wgsl`).
/// 32 bytes. `m` is the full row count (the out slab is `[m, ncols]`);
/// `ncols` is only this dispatch's column chunk, and the weight window starts
/// at column 0 of that chunk. `split` is 1 when the encoder put the Q4_K entry
/// point on its split geometry (see `gguf_gpu.rs`'s
/// `GROUPED_SPLIT_MAX_BASE_WORKGROUPS`), 0 for the base one; the Q6_K and Q8_0
/// entries ignore it. The last two words are padding: a uniform binding size
/// must be a multiple of 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGemmGgufParams {
    pub m: u32,
    pub k: u32,
    pub ncols: u32,
    pub x_vec4_off: u32,
    pub w_byte_off: u32,
    pub split: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for DequantMatMul. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantMatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub block_size: u32,
    pub scheme_id: u32,
    pub x_off: u32,
    pub w_off: u32,
    pub scale_off: u32,
    pub zp_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for MLX DequantMatMul. 48 bytes (12 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantMatmulMlxParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub kind: u32,
    pub bits: u32,
    pub group_size: u32,
    pub x_byte_off: u32,
    pub w_byte_off: u32,
    pub scale_byte_off: u32,
    pub zp_byte_off: u32,
    pub out_byte_off: u32,
    pub _pad: u32,
}

/// Layout for FusedResidualLN-Tee. 52 bytes (13 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedResidualLnTeeParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub residual_off: u32,
    pub bias_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub sum_off: u32,
    pub ln_out_off: u32,
    pub eps_bits: u32,
    pub has_bias: u32,
    /// 1 = RMS norm (scale by 1/rms, no mean subtraction), 0 = layer norm.
    /// Uses what used to be pad: the two norms share this kernel because they
    /// share the (multi-consumer Add -> norm) shape, and only the final scaling
    /// differs.
    pub is_rms: u32,
    /// 1 = the `fused_residual_ln_tee_vec4` module (256 lanes, 16 B accesses).
    pub vec4: u32,
}

/// Layout for matmul_qkv (split-write QKV matmul).
/// 64 bytes (16 u32s); WebGPU uniform-buffer 16-byte alignment OK.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MatmulQkvParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub head_width: u32,
    pub has_bias: u32,
    pub bias_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Layout for FusedResidualRmsNorm (same bind layout as FusedResidualLN).
pub type FusedResidualRmsNormParams = FusedResidualLnParams;

/// Layout for AdaLayerNorm. 112 bytes (28 u32s).
/// `lead_pack` is 20 u32s (5×vec4 in WGSL); first 17 from IR are used.
/// Prefixed by 8 scalar u32s (32 bytes) so the vec4 array is naturally
/// 16-byte aligned — no implicit WGSL padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AdaLayerNormParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub scale_off: u32,
    pub shift_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub layer_norm: u32,
    pub lead_pack: [u32; 20],
}

/// Layout for GatedResidual. 128 bytes (32 u32s).
/// Six scalar fields (24 B) + 8 B explicit pad so `lead_pack` (vec4 array)
/// starts at offset 32, matching WGSL uniform layout; trailing pad to 128.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedResidualParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub y_off: u32,
    pub gate_off: u32,
    pub out_off: u32,
    pub _pre0: u32,
    pub _pre1: u32,
    pub lead_pack: [u32; 20],
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Expand the 17-u32 IR lead pack into the 20-u32 WGSL uniform slot.
#[inline]
pub fn lead_pack_uniform(src: [u32; 17]) -> [u32; 20] {
    let mut out = [0u32; 20];
    out[..17].copy_from_slice(&src);
    out
}

/// Layout for AdaLayerNormBackward. 48 bytes (12 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AdaLayerNormBackwardParams {
    pub mod_rows: u32,
    pub seq_per_mod: u32,
    pub inner: u32,
    pub x_off: u32,
    pub scale_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub layer_norm: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GatedResidualBackward. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedResidualBackwardParams {
    pub mod_rows: u32,
    pub seq_per_mod: u32,
    pub inner: u32,
    pub y_off: u32,
    pub gate_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub _p0: u32,
}

/// Layout for FusedResidualLN. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedResidualLnParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub residual_off: u32,
    pub bias_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub has_bias: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for Mamba2 (SSD scan). 68→72 bytes padded to 16.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Mamba2Params {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    pub x_off: u32,
    pub dt_off: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Layout for GRU (native WGSL). 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GruParams {
    pub batch: u32,
    pub seq: u32,
    pub input_size: u32,
    pub hidden: u32,
    pub x_off: u32,
    pub wih_off: u32,
    pub whh_off: u32,
    pub bih_off: u32,
    pub bhh_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    /// h0 (carry) word offset; 0 = no carry (h0 = 0).
    pub h0_off: u32,
    /// dirs·hidden — width of an output row (this dir owns `[dir_off, dir_off+h)`).
    pub out_width: u32,
    pub dir_off: u32,
    /// 1 → walk the sequence backwards (reverse direction of a bidir layer).
    pub reverse: u32,
    pub _p5: u32,
}

/// Layout for Elman RNN (native WGSL). 68→padded. `relu` selects activation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RnnParams {
    pub batch: u32,
    pub seq: u32,
    pub input_size: u32,
    pub hidden: u32,
    pub x_off: u32,
    pub wih_off: u32,
    pub whh_off: u32,
    pub bias_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    pub relu: u32,
    /// h0 (carry) word offset; 0 = no carry (h0 = 0).
    pub h0_off: u32,
    /// dirs·hidden — width of an output row (this dir owns `[dir_off, dir_off+h)`).
    pub out_width: u32,
    pub dir_off: u32,
    /// 1 → walk the sequence backwards (reverse direction of a bidir layer).
    pub reverse: u32,
}

/// Layout for LSTM (native WGSL). 64 bytes. Gate order i, f, g, o; single
/// merged bias. One dispatch per (layer, direction).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LstmParams {
    pub batch: u32,
    pub seq: u32,
    pub input_size: u32,
    pub hidden: u32,
    pub x_off: u32,
    pub wih_off: u32,
    pub whh_off: u32,
    pub bias_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    /// h0 (carry) word offset; 0 = no carry (h0 = 0).
    pub h0_off: u32,
    /// c0 (carry) word offset; 0 = no carry (c0 = 0).
    pub c0_off: u32,
    /// dirs·hidden — width of an output row (this dir owns `[dir_off, dir_off+h)`).
    pub out_width: u32,
    pub dir_off: u32,
    /// 1 → walk the sequence backwards (reverse direction of a bidir layer).
    pub reverse: u32,
    /// 1 → write the final `hn`/`cn` back into `h0_off`/`c0_off` in place.
    /// Explicit rather than inferred from `h0_off != 0`, because arena offset 0
    /// is a legal placement for the state tensor.
    pub carry: u32,
}

/// Layout for SelectiveScan. 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SelectiveScanParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub state_size: u32,
    pub x_off: u32,
    pub delta_off: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    /// PLAN L1 — full-extent seq stride for per-batch offset math.
    /// Stays at compile-time `seq` even when runtime `seq` is scaled,
    /// so per-batch arena offsets stay correct under active-extent.
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
}

/// Layout for GatedDeltaNet. 64 bytes (16 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedDeltaNetParams {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub state_size: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub g_off: u32,
    pub beta_off: u32,
    pub state_off: u32,
    pub out_off: u32,
    pub use_carry: u32,
    /// PLAN L1 — full-extent seq stride for per-batch offset math.
    pub seq_stride: u32,
    /// 1 = per-channel gate (`g` is `[b,s,h,n]`, Kimi-K3 KDA); 0 = per-head scalar.
    pub gate_per_channel: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for Sample. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SampleParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub top_k: u32,
    pub top_p_bits: u32,
    pub temp_bits: u32,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub scratch_off: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GroupedMatMul. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroupedMatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub num_experts: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
}

/// Layout for ScaledGroupedMatMul (native MXFP4 grouped decode-GEMM). 64 bytes.
/// Code/scale offsets are BYTE offsets (packed U8); idx/out/bias are f32
/// element offsets.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScaledGroupedMatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub num_experts: u32,
    pub input_byte_off: u32,
    pub weight_byte_off: u32,
    pub input_scale_byte_off: u32,
    pub weight_scale_byte_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
    pub bias_off: u32,
    pub scale_mode: u32,
    pub block: u32,
    pub has_bias: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Layout for TopK. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TopKParams {
    pub outer: u32,
    pub inner: u32,
    pub k: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Native GPU WelchPeaks dispatch parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WelchPeaksGpuParams {
    pub spec_off: u32,
    pub dst_off: u32,
    pub welch_batch: u32,
    pub n_fft: u32,
    pub n_segments: u32,
    pub k: u32,
    pub n_bins: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for UMAP k-NN on a pairwise `[n, n]` matrix. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UmapKnnParams {
    pub n: u32,
    pub k: u32,
    pub pw_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for ScatterAdd. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScatterAddParams {
    pub op: u32, // 0 = zero phase, 1 = accumulate phase
    pub out_off: u32,
    pub upd_off: u32,
    pub idx_off: u32,
    pub out_total: u32,
    pub num_updates: u32,
    pub trailing: u32,
    pub out_dim: u32,
}

/// Layout for Conv3D NCDHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub dd: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    /// Per-output-channel bias offset, valid when `has_bias != 0`. Folding it
    /// into the conv's store removes both the full-size `Expand` of the bias
    /// and the full-size `Add` that consumed it.
    pub bias_off: u32,
    pub has_bias: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Lazy-init container for a compute pipeline + its bind-group layout.
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl Kernel {
    pub fn bind_two(
        &self,
        device: &wgpu::Device,
        arena: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu fft gpu bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    }
}

/// Build a 4-binding compute kernel: storage(rw) / uniform / storage(ro)
/// / storage(rw). `transpose_vec4.wgsl` is its user: binding 0 is the `vec4`
/// arena (modes 1/2), binding 1 params, binding 2 the axis metadata, and
/// binding 3 a scalar alias of the same window for mode 3's non-adjacent
/// stores — the alias is a separate binding because a component store through
/// the `vec4` array is a whole-group read-modify-write on some backends, which
/// two threads writing different lanes of one group would race on.
fn build_kernel_4(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

fn build_kernel_3(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    build_kernel_3_with(device, label, wgsl, entry_point, true)
}

/// [`build_kernel_3`] with binding 2's access mode chosen by the caller: the
/// attention entries write their split partials through it, the other users of
/// this shape only read it.
fn build_kernel_3_with(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
    parts_read_only: bool,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: parts_read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// Binding layout: storage + uniform + storage + storage — the GGUF GEMV's x
/// (arena window) + params + weight (arena window) + out (window), each arena
/// window bound **once**.
///
/// `in_place_arena` selects the one-row GEMV's access modes. That arm can bind
/// x, the weight and the output out of one buffer, so all three arena bindings
/// (0, 2 and 3) are declared read_write even though the entry points only read
/// through 0 and 2. The windows are disjoint — the planner keeps the input live
/// until the store, and `gguf_gpu::gemv_in_place_out_window` refuses an output
/// window whose align-down base would reach into the X row or the packed weight —
/// so no two writable bindings share a range, which is what WebGPU requires.
/// Without `in_place_arena` the two arena windows are read-only and the only
/// read-write binding is a distinct buffer.
fn build_kernel_ro_u_ro_rw(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
    in_place_arena: bool,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    // With `in_place_arena`, every window this layout carries is declared
    // read_write — see this function's doc for why that is the point of the flag.
    let arena_read_only = !in_place_arena;
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(arena_read_only),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(arena_read_only),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(false),
            count: None,
        },
    ];
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// Read-only stream (`stream_read.wgsl`): binding 0 is the read-only source,
/// binding 1 params, binding 2 the one-f32-per-workgroup sink that keeps the
/// loads live. Binding 0 has to be `read`: the figure the probe takes is a
/// read stream, and the shader must not be able to write it.
fn build_kernel_ro_u_rw(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// f16 shadow (rw) + uniform + f32 arena (rw) — `cast_f32_to_f16` only.
/// Separate from `build_kernel_3`: cast reads f32 written by a prior unary in
/// the same arena; other 3-binding kernels keep binding 2 read-only.
fn build_kernel_cast_f32_to_f16(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// f32 arena (rw) + uniform + f16 shadow (rw) — unary with CoopF16Vk mirror.
fn build_kernel_f32_rw_uniform_f16_rw(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// f16 shadow (read) + f32 arena (rw) + uniform — Vulkan coop f16 matmul.
fn build_kernel_coop_f16_vk(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

fn try_build_kernel_coop_f16_vk(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Option<Kernel> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_kernel_coop_f16_vk(device, label, wgsl, entry_point)
    }))
    .ok()
}

fn build_kernel(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

static MATMUL: OnceLock<Kernel> = OnceLock::new();
static MATMUL_WIDE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_WIDE_NV: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16W: OnceLock<Kernel> = OnceLock::new();
static MATMUL_BF16W: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16_COMPUTE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP16: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static MATMUL_COOP_F32: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F32_DIRECT: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F32_PORTABLE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_WIDEN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static CAST_F32_TO_F16: OnceLock<Kernel> = OnceLock::new();
static BINARY: OnceLock<Kernel> = OnceLock::new();
static SNAKE_BETA: OnceLock<Kernel> = OnceLock::new();
static BINARY_C64: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_CAST: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_NORM_SQ: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_NORM_SQ_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static CONJUGATE_C64: OnceLock<Kernel> = OnceLock::new();
static FFT_BUTTERFLY_STAGE: OnceLock<Kernel> = OnceLock::new();
static UNARY: OnceLock<Kernel> = OnceLock::new();
static UNARY_F16_MIRROR: OnceLock<Kernel> = OnceLock::new();
static COMPARE: OnceLock<Kernel> = OnceLock::new();
static WHEREK: OnceLock<Kernel> = OnceLock::new();
static FMAK: OnceLock<Kernel> = OnceLock::new();
static ACTIVATION_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static REDUCE: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY_WITH_LOGITS: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY_BWD: OnceLock<Kernel> = OnceLock::new();
static MAXPOOL2D_BWD: OnceLock<Kernel> = OnceLock::new();
static MAXPOOL3D_BWD: OnceLock<Kernel> = OnceLock::new();
static CONV3D_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static CONV3D_BWD_WEIGHT: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_GAMMA: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_BETA: OnceLock<Kernel> = OnceLock::new();
static AXIAL_ROPE2D: OnceLock<Kernel> = OnceLock::new();
static FAKE_QUANTIZE_FIXED: OnceLock<Kernel> = OnceLock::new();
static FAKE_QUANTIZE_PERBATCH: OnceLock<Kernel> = OnceLock::new();
static LAYERNORM: OnceLock<Kernel> = OnceLock::new();
static LAYERNORM_VEC4: OnceLock<Kernel> = OnceLock::new();
static NORM_ROPE: OnceLock<Kernel> = OnceLock::new();
static NARROW_SPLIT: OnceLock<Kernel> = OnceLock::new();
static CONCAT_MANY: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM2D: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD_PARAM: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_GAMMA: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_GAMMA_REDUCE: OnceLock<Kernel> = OnceLock::new();
static CUMSUM_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE_BWD: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ZERO: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ACC: OnceLock<Kernel> = OnceLock::new();
static CUMSUM: OnceLock<Kernel> = OnceLock::new();
static CUM_SCAN: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_RADIX2: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_RADIX2_BIG: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R2: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R4: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R8: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_R4_16K: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_MULTIROW: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_BITREV: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_INNER: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_OUTER_R4: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_OUTER_R2: OnceLock<Kernel> = OnceLock::new();
static COPY: OnceLock<Kernel> = OnceLock::new();
static COPY_VEC4: OnceLock<Kernel> = OnceLock::new();
static COPY_FROM_SLAB: OnceLock<Kernel> = OnceLock::new();
static COPY_FROM_SLAB_VEC4: OnceLock<Kernel> = OnceLock::new();
static STREAM_READ: OnceLock<Kernel> = OnceLock::new();
static CAST: OnceLock<Kernel> = OnceLock::new();
static ELEMENTWISE_REGION: OnceLock<Kernel> = OnceLock::new();
static ELEMENTWISE_REGION_SPATIAL: OnceLock<Kernel> = OnceLock::new();
static TRANSPOSE: OnceLock<Kernel> = OnceLock::new();
static TRANSPOSE_VEC4: OnceLock<Kernel> = OnceLock::new();
static NARROW: OnceLock<Kernel> = OnceLock::new();
static NARROW_VEC4: OnceLock<Kernel> = OnceLock::new();
static CONCAT: OnceLock<Kernel> = OnceLock::new();
static CONCAT_VEC4: OnceLock<Kernel> = OnceLock::new();
static FRAMES: OnceLock<Kernel> = OnceLock::new();
static GATHER: OnceLock<Kernel> = OnceLock::new();
static GATHER_SPLIT: OnceLock<Kernel> = OnceLock::new();
static GATHER_AXIS: OnceLock<Kernel> = OnceLock::new();
static ATTENTION: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_DECODE_MERGE: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_DECODE: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_MASK: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE: OnceLock<Kernel> = OnceLock::new();
static EXPAND: OnceLock<Kernel> = OnceLock::new();
static REVERSE: OnceLock<Kernel> = OnceLock::new();
static QUANTIZE_I8: OnceLock<Kernel> = OnceLock::new();
static DEQUANTIZE_I8: OnceLock<Kernel> = OnceLock::new();
static Q_MATMUL: OnceLock<Kernel> = OnceLock::new();
static Q_CONV2D: OnceLock<Kernel> = OnceLock::new();
static BN_INFER: OnceLock<Kernel> = OnceLock::new();
static BN_INFER_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static BN_INFER_BWD_GAMMA: OnceLock<Kernel> = OnceLock::new();
static BN_INFER_BWD_BETA: OnceLock<Kernel> = OnceLock::new();
static SCALED_QUANT_SCALE: OnceLock<Kernel> = OnceLock::new();
static SCALED_QUANTIZE: OnceLock<Kernel> = OnceLock::new();
static SCALED_DEQUANTIZE: OnceLock<Kernel> = OnceLock::new();
static SCALED_MATMUL_DECODE: OnceLock<Kernel> = OnceLock::new();
static GATHER_ND: OnceLock<Kernel> = OnceLock::new();
static GATHER_ELEMENTS: OnceLock<Kernel> = OnceLock::new();
static SCATTER_ELEMENTS: OnceLock<Kernel> = OnceLock::new();
static SCATTER_ND_REDUCE: OnceLock<Kernel> = OnceLock::new();
static COPY_SANITIZE: OnceLock<Kernel> = OnceLock::new();
static ARGMAX: OnceLock<Kernel> = OnceLock::new();
static POOL2D: OnceLock<Kernel> = OnceLock::new();
static CONV2D: OnceLock<Kernel> = OnceLock::new();
static CONV1D_TILED: OnceLock<Kernel> = OnceLock::new();
static IM2COL2D: OnceLock<Kernel> = OnceLock::new();
static CONV1D_IMPLICIT: OnceLock<Kernel> = OnceLock::new();
static POOL1D: OnceLock<Kernel> = OnceLock::new();
static POOL3D: OnceLock<Kernel> = OnceLock::new();
static CONV1D: OnceLock<Kernel> = OnceLock::new();
static CONV3D: OnceLock<Kernel> = OnceLock::new();
static CONV3D_CTILE: OnceLock<Kernel> = OnceLock::new();
static CONV_TRANSPOSE3D: OnceLock<Kernel> = OnceLock::new();
static CONV_TRANSPOSE2D: OnceLock<Kernel> = OnceLock::new();
static CONV_TRANSPOSE2D_TILED: OnceLock<Kernel> = OnceLock::new();
static SCATTER_ADD: OnceLock<Kernel> = OnceLock::new();
static TOPK: OnceLock<Kernel> = OnceLock::new();
static WELCH_PEAKS_GPU: OnceLock<Kernel> = OnceLock::new();
static UMAP_KNN: OnceLock<Kernel> = OnceLock::new();
static GROUPED_MATMUL: OnceLock<Kernel> = OnceLock::new();
static SCALED_GROUPED_MATMUL_DECODE: OnceLock<Kernel> = OnceLock::new();
static SAMPLE: OnceLock<Kernel> = OnceLock::new();
static SELECTIVE_SCAN: OnceLock<Kernel> = OnceLock::new();
static GATED_DELTA_NET: OnceLock<Kernel> = OnceLock::new();
static MAMBA2: OnceLock<Kernel> = OnceLock::new();
static GRU: OnceLock<Kernel> = OnceLock::new();
static RNN: OnceLock<Kernel> = OnceLock::new();
static LSTM: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_MATMUL: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_MATMUL_MLX: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GGUF: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMV_GGUF: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMV_GGUF_TILED: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMV_GGUF_WIDE: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMV_GGUF_WIDE_ILP: OnceLock<Kernel> = OnceLock::new();
static GGUF_SUBGROUP_PROBE: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMM_Q1_0: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMM_Q4K: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMM_Q6K: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMM_Q8_0: OnceLock<Kernel> = OnceLock::new();
static MATMUL_BT: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN_TEE: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN_TEE_VEC4: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_RMS_NORM: OnceLock<Kernel> = OnceLock::new();
static ADA_LAYER_NORM: OnceLock<Kernel> = OnceLock::new();
static GATED_RESIDUAL: OnceLock<Kernel> = OnceLock::new();
static ADA_LAYER_NORM_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static GATED_RESIDUAL_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F32: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_WIDEN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();

pub fn matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL.get_or_init(|| build_kernel(device, "rlx-wgpu matmul", MATMUL_WGSL, "matmul"))
}
pub fn matmul_wide_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_WIDE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide",
            MATMUL_WIDE_WGSL,
            "matmul_wide",
        )
    })
}
/// 64×64 / 256-thread variant for discrete GPUs (Vulkan path).
/// Vectorised 64×64 wide tile — see `matmul_wide_vec4.wgsl`. Measured 2.8×
/// `matmul_wide` on an M4 Pro and bit-identical to it.
pub fn matmul_wide_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    static MATMUL_WIDE_VEC4: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
    MATMUL_WIDE_VEC4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide_vec4",
            MATMUL_WIDE_VEC4_WGSL,
            "matmul_wide_vec4",
        )
    })
}

/// The remainder pipeline of the wide path: the same source as
/// `matmul_wide_vec4_kernel` at half the row tile (`matmul_wide_vec4_tail`,
/// 16x8 threads / 32 rows), so the rows the full-tile grid rounds up are
/// computed once instead of as a whole padded tile (see `wide_grid`).
pub fn matmul_wide_vec4_tail_kernel(device: &wgpu::Device) -> &'static Kernel {
    static MATMUL_WIDE_VEC4_TAIL: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
    MATMUL_WIDE_VEC4_TAIL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide_vec4_tail",
            MATMUL_WIDE_VEC4_WGSL,
            "matmul_wide_vec4_tail",
        )
    })
}

pub fn matmul_wide_nv_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_WIDE_NV.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide_nv",
            MATMUL_WIDE_NV_WGSL,
            "matmul_wide_nv",
        )
    })
}
/// f16-weight matmul (f32 compute). Returns Some only when the device
/// exposes the `SHADER_F16` feature. EXPERIMENTAL: currently slower
/// than the f32 baseline on Apple Silicon — kept as foundation; see
/// `matmul_f16w.wgsl` for the empirical analysis.
pub fn matmul_f16w_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(MATMUL_F16W.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_f16w",
            MATMUL_F16W_WGSL,
            "matmul_f16w",
        )
    }))
}
/// Packed-BF16-weight matmul (f32 compute). Reads B as `array<u32>`
/// (two bf16 per word) and unpacks `bitcast<f32>(bits << 16)` in-shader,
/// so it reads HALF the B bytes vs the widened-f32 arena path and stays
/// bit-exact to a bf16-rounded f32 matmul. Pure WGSL u32 ops — available
/// on every device (no `SHADER_F16` needed).
pub fn matmul_bf16w_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_BF16W.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_bf16w",
            MATMUL_BF16W_WGSL,
            "matmul_bf16w",
        )
    })
}
/// f16-compute matmul: f16 operands, f16 multiply, f32 accumulator.
/// Targets the 2× f16 ALU throughput on Apple Silicon. Returns Some
/// only when the device exposes `SHADER_F16`.
pub fn matmul_f16_compute_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(MATMUL_F16_COMPUTE.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_f16_compute",
            MATMUL_F16_COMPUTE_WGSL,
            "matmul_f16_compute",
        )
    }))
}
/// Cooperative-matrix matmul (8×8 tiles, hardware GEMM units).
/// Lowers to MSL `simdgroup_matrix` on Metal and SPIR-V's
/// `OpCooperativeMatrixMulAddKHR` on Vulkan. Returns Some only when
/// the device exposes both `SHADER_F16` and
/// `EXPERIMENTAL_COOPERATIVE_MATRIX`.
pub fn matmul_coop16_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::SHADER_F16)
        || !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return None;
    }
    Some(MATMUL_COOP16.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_coop16",
            MATMUL_COOP16_WGSL,
            "matmul_coop16",
        )
    }))
}
/// f16 operands with an f32 cooperative accumulator, 64×64 output tile per
/// 64-thread workgroup (two 32-wide subgroups, each owning a 64×32 half). A is
/// downcast from the f32 arena into a ping-pong workgroup stage at the load; B
/// is read as f16 from the weight buffer; the accumulator never leaves f32.
///
/// Three refusals, each named rather than left to a later panic: the device
/// needs `SHADER_F16` and `EXPERIMENTAL_COOPERATIVE_MATRIX` for the mixed
/// f16-operand/f32-accumulator form to exist at all, and it needs the workgroup
/// storage the shader actually declares — two 4 KiB ping-pong A stages plus the
/// 256 B of zero lanes the accumulators are seeded from,
/// [`COOP_F16_F32ACC_STORAGE`] (8 448 B) — below that `create_compute_pipeline`
/// would fail, so the kernel is reported absent instead. There is no B stage (B
/// is read as f16 straight from the weight buffer) and no f32 output tile in
/// workgroup memory (the 64×64 tile lives in registers); both omissions are what
/// the shader header measures against the forms that had them. Whether the built
/// kernel computes `a·b` is a separate question, answered per device by
/// `crate::coop_probe`.
pub fn matmul_coop_f16_f32acc_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::SHADER_F16)
        || !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        || device.limits().max_compute_workgroup_storage_size < COOP_F16_F32ACC_STORAGE
    {
        return None;
    }
    MATMUL_COOP_F16_F32ACC
        .get_or_init(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                build_kernel_3(
                    device,
                    "rlx-wgpu matmul_coop_f16_f32acc",
                    MATMUL_COOP_F16_F32ACC_WGSL,
                    "matmul_coop_f16_f32acc",
                )
            }))
            .ok()
        })
        .as_ref()
}

/// The workgroup storage `matmul_coop_f16_f32acc.wgsl` declares: two 2048-f16
/// ping-pong staging buffers (4 KiB each) plus the 64 f32 zero lanes the
/// accumulators are seeded from. The shader keeps its output tile out of
/// workgroup memory for exactly this reason — see its header. Shared with nothing
/// else, but named here because the kernel getter refuses a device below it.
pub const COOP_F16_F32ACC_STORAGE: u32 = 4 * 1024 + 4 * 1024 + 256;

/// The output-tile extent [`matmul_coop_f16_f32acc_kernel`] covers per workgroup,
/// as the dispatch grid needs it: `(n / 64, m / 64)` — whole tiles only, which is
/// what the admission requires.
pub fn coop_f16_f32acc_grid(m: u32, n: u32) -> (u32, u32) {
    (n.div_ceil(64), m.div_ceil(64))
}

/// Pure-f32 cooperative-matrix matmul. No SHADER_F16 needed — uses
/// `coop_mat8x8<f32>` throughout (lowers to `simdgroup_float8x8` on
/// Apple). Returns None if the cooperative-matrix feature is missing
/// OR if the device's WGSL→backend lowering can't compile it (some
/// implementations only expose half-precision coop matrices).
pub fn matmul_coop_f32_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return None;
    }
    Some(MATMUL_COOP_F32.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_coop_f32",
            MATMUL_COOP_F32_WGSL,
            "matmul_coop_f32",
        )
    }))
}
/// The convolution GEMM routes' direct form: the same 32×32 tile as
/// [`matmul_coop_f32_kernel`], with no workgroup staging and no epilogue pass
/// (see the shader header). Metal only — the load/store conventions this entry
/// stands on are the MSL ones, and the portable Vulkan form is a different
/// kernel. `None` when the device cannot build it, in which case every call keeps
/// the general kernel.
pub fn matmul_coop_f32_direct_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        || crate::device::wgpu_device().map(|d| d.backend) != Some(wgpu::Backend::Metal)
    {
        return None;
    }
    Some(MATMUL_COOP_F32_DIRECT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_coop_f32_direct",
            MATMUL_COOP_F32_DIRECT_WGSL,
            "matmul_coop_f32_direct",
        )
    }))
}
/// Portable 8×8 coop f32 matmul (`coopLoad`, 8×8 workgroups) — the discrete
/// Vulkan form; DX12 exposes no cooperative matrices.
pub fn matmul_coop_f32_portable_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        || !crate::device::coop_f32_8x8_supported()
    {
        return None;
    }
    Some(MATMUL_COOP_F32_PORTABLE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_coop_f32_portable",
            MATMUL_COOP_F32_PORTABLE_WGSL,
            "matmul_coop_f32_portable",
        )
    }))
}
fn coop_f16_vk_device_ready(device: &wgpu::Device) -> bool {
    // Cooperative-matrix f16 matmul is OFF by default — see
    // `coop_f16_vk_eligible` in `backend.rs` for the rationale. Opt in
    // with `RLX_WGPU_COOP_F16_VK_ENABLE=1`. Legacy
    // `RLX_WGPU_COOP_F16_VK_DISABLE=1` also fully disables.
    if rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_DISABLE")
        || !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_ENABLE")
    {
        return false;
    }
    device.features().contains(wgpu::Features::SHADER_F16)
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && crate::device::coop_f16_16x16_supported()
        && crate::device::coop_discrete_backend()
}

fn coop_f16_vk_f32acc_device_ready(device: &wgpu::Device) -> bool {
    coop_f16_vk_device_ready(device) && crate::device::coop_f16_16x16_f32_acc_supported()
}

pub fn matmul_coop_f16_vulkan_f32acc_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_COOP_F16_VULKAN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_coop_f16_vulkan_f32acc",
                MATMUL_COOP_F16_VULKAN_F32ACC_WGSL,
                "matmul_coop_f16_vulkan_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_coop_f16_vulkan_widen_f32acc_kernel(
    device: &wgpu::Device,
) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_coop_f16_vulkan_widen_f32acc",
                MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC_WGSL,
                "matmul_coop_f16_vulkan_widen_f32acc",
            )
        })
        .as_ref()
}

pub(crate) fn coop_f16_vk_use_f32acc(device: &wgpu::Device) -> bool {
    !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_NO_F32ACC")
        && matmul_coop_f16_vulkan_f32acc_kernel(device).is_some()
}

fn pick_coop_f16_vk_matmul(
    device: &wgpu::Device,
    n: u32,
    loadt: fn(&wgpu::Device) -> Option<&'static Kernel>,
    loadt_f32acc: fn(&wgpu::Device) -> Option<&'static Kernel>,
    widen: fn(&wgpu::Device) -> Option<&'static Kernel>,
    widen_f32acc: fn(&wgpu::Device) -> Option<&'static Kernel>,
) -> Option<&'static Kernel> {
    let f32acc = coop_f16_vk_use_f32acc(device);
    // WHICH B-LOAD IS CORRECT IS MEASURED, NOT GUESSED.
    //
    // `matmul_coop_f16_vulkan.wgsl` and `..._widen.wgsl` differ by exactly one
    // token — `coopLoadT` vs `coopLoad` on B — against the same `b_ptr` and the
    // same `params.n` stride. This used to pick between them on `n > 768`. A
    // memory layout is a property of the bytes, not of N: the same buffer cannot
    // be row-major at n=768 and column-major at n=769, so at most one of the two
    // is right and the threshold was serving a transposed product to one half of
    // the N range. `f16_vk_widen_choice` runs a non-commuting probe on the actual
    // device (memoized) and returns the variant that computes `a·b`, keeping the
    // old size-based preference only as a tie-break between two correct kernels.
    //
    // `None` means neither variant is correct here. `derive_matmul_compute`
    // gates on the same probe so this path should already be ineligible, but
    // refuse anyway rather than hand back a kernel known to produce wrong
    // numbers.
    let widen_b = crate::coop_probe::f16_vk_widen_choice(n, f32acc)?;
    if f32acc {
        if widen_b {
            return widen_f32acc(device).or_else(|| loadt_f32acc(device));
        }
        return loadt_f32acc(device);
    }
    if widen_b {
        widen(device).or_else(|| loadt(device))
    } else {
        loadt(device)
    }
}

/// Matmul CoopF16Vk kernel for column count `n`.
pub fn matmul_coop_f16_vulkan_active_kernel(
    device: &wgpu::Device,
    n: u32,
) -> Option<&'static Kernel> {
    pick_coop_f16_vk_matmul(
        device,
        n,
        matmul_coop_f16_vulkan_kernel,
        matmul_coop_f16_vulkan_f32acc_kernel,
        matmul_coop_f16_vulkan_widen_kernel,
        matmul_coop_f16_vulkan_widen_f32acc_kernel,
    )
}

pub fn matmul_coop_f16_vulkan_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_COOP_F16_VULKAN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_coop_f16_vulkan",
            MATMUL_COOP_F16_VULKAN_WGSL,
            "matmul_coop_f16_vulkan",
        )
    }))
}
/// N above which coop may use the row-major B-load variant (`RLX_WGPU_COOP_F16_VK_LARGE_N`).
pub const COOP_F16_VK_WIDEN_N: u32 = 768;

/// Use `coopLoad` on B instead of `coopLoadT` when N > 768 and `RLX_WGPU_COOP_F16_VK_LOAD_T` is unset.
pub fn coop_f16_vk_widen_b_load(n: u32) -> bool {
    n > COOP_F16_VK_WIDEN_N && !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_LOAD_T")
}

pub fn matmul_coop_f16_vulkan_widen_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_COOP_F16_VULKAN_WIDEN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_coop_f16_vulkan_widen",
            MATMUL_COOP_F16_VULKAN_WIDEN_WGSL,
            "matmul_coop_f16_vulkan_widen",
        )
    }))
}
pub fn coop_f16_vk_f32acc_available(device: &wgpu::Device) -> bool {
    matmul_coop_f16_vulkan_f32acc_kernel(device).is_some()
}
/// CoopF32 kernel for the active wgpu backend (Metal `simdgroup_float8x8` vs the
/// portable 8×8 form, which only a discrete Vulkan adapter can actually build —
/// the DX12 arm names the same kernel and gets `None` from it).
pub fn matmul_coop_f32_active_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    match crate::device::wgpu_device().map(|d| d.backend) {
        Some(wgpu::Backend::Metal) => matmul_coop_f32_kernel(device),
        Some(wgpu::Backend::Vulkan) | Some(wgpu::Backend::Dx12) => {
            matmul_coop_f32_portable_kernel(device)
        }
        _ => None,
    }
}
/// Wide f32 matmul kernel for the active backend.
/// Which wide-tile matmul kernel this adapter uses, and its output tile height.
///
/// The kernel and the dispatch grid MUST be derived from this one function.
/// They were previously chosen by two separate `matches!(backend, ...)`
/// expressions, which is the same shape of bug as the cooperative-matrix
/// base/widen split: two places deciding the same thing, free to disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideVariant {
    /// `matmul_wide` — 32×64 tile, 64 threads. Scalar; ~2.5% of peak.
    /// Keeps the single rounded-up dispatch: no remainder pipeline.
    Std,
    /// `matmul_wide_nv` — 64×64 tile, 256 threads. Scalar.
    /// Keeps the single rounded-up dispatch: no remainder pipeline.
    Nv,
    /// `matmul_wide_vec4` — 64×64 tile, 256 threads, `vec4` tiles. Default:
    /// measured 3.0–4.4× `Std` on an M4 Pro and bit-identical to it
    /// (`tests/matmul_vec4_parity.rs`). Also ships the half-height remainder
    /// configuration (`matmul_wide_vec4_tail`, `WIDE_TAIL_M` rows).
    Vec4,
}

/// The row tile of the wide path's remainder pipeline: `matmul_wide_vec4_tail`,
/// 16×8 threads / 32 rows. Mirrors `TAIL_M` in `matmul_wide_vec4.wgsl`, and the
/// geometry pin in this module's tests holds the two together.
pub const WIDE_TAIL_M: u32 = 32;

impl WideVariant {
    /// Output rows per workgroup — the `m` divisor for the dispatch grid. All
    /// three variants are 64 columns wide.
    pub fn tile_m(self) -> u32 {
        match self {
            WideVariant::Std => 32,
            WideVariant::Nv | WideVariant::Vec4 => 64,
        }
    }

    /// The row count of this variant's remainder pipeline and the entry point
    /// that runs it, for the variants that ship one; `None` keeps the single
    /// rounded-up dispatch, so pinning the variant reproduces the pre-remainder
    /// geometry exactly. One match decides both halves, so the grid
    /// [`wide_grid`] builds and the pipeline [`matmul_wide_tail_kernel`]
    /// records cannot disagree.
    fn remainder(self) -> Option<(u32, fn(&wgpu::Device) -> &'static Kernel)> {
        match self {
            WideVariant::Vec4 => Some((WIDE_TAIL_M, matmul_wide_vec4_tail_kernel)),
            WideVariant::Std | WideVariant::Nv => None,
        }
    }
}

/// Escape hatches: `RLX_WGPU_WIDE_STD` / `RLX_WGPU_WIDE_NV` pin the older
/// scalar kernels, for A/B measurement or if a driver mishandles the vectorised
/// one.
pub fn wide_variant() -> WideVariant {
    if rlx_ir::env::flag("RLX_WGPU_WIDE_STD") {
        return WideVariant::Std;
    }
    if rlx_ir::env::flag("RLX_WGPU_WIDE_NV") {
        return WideVariant::Nv;
    }
    WideVariant::Vec4
}

pub fn matmul_wide_active_kernel(device: &wgpu::Device) -> &'static Kernel {
    match wide_variant() {
        WideVariant::Std => matmul_wide_kernel(device),
        WideVariant::Nv => matmul_wide_nv_kernel(device),
        WideVariant::Vec4 => matmul_wide_vec4_kernel(device),
    }
}

/// The active variant's remainder pipeline, `Some` exactly when [`wide_grid`]
/// can ask for a remainder workgroup — both read `WideVariant::remainder`.
pub fn matmul_wide_tail_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    wide_variant().remainder().map(|(_, build)| build(device))
}

/// One wide-path matmul's row-axis split: `full` workgroups of
/// `variant.tile_m()` rows, then `tail` workgroups of the variant's remainder
/// tile (`WIDE_TAIL_M` rows), all over `gx` column tiles of 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideGrid {
    pub gx: u32,
    pub full: u32,
    pub tail: u32,
}

/// The dispatch grid of the wide path, derived from the variant alone so the
/// kernel pick and the grid cannot disagree (see [`WideVariant`]'s note).
///
/// The rows `full` and `tail` cover add up to at least `m_s` and never to more
/// than the single rounded-up grid `m_s.div_ceil(tile_m) * tile_m`, so the split
/// removes padded rows and can never add one. Two kinds of row count keep the
/// single rounded-up grid, both with `tail = 0`:
///
/// - below one whole tile: the remainder tile is anchored after the last full
///   one, so it would start at row 0 and a 33..63-row node would lose every row
///   above 32 — the hole, not the padding, is why this case stays whole;
/// - a remainder wider than the remainder tile: two remainders would compute the
///   same 64 rows through two workgroups and two passes over the B panel.
///
/// A variant that ships no remainder pipeline (`WideVariant::remainder`) has
/// no remainder tile to record and always takes the single grid.
pub fn wide_grid(variant: WideVariant, m_s: u32, n: u32) -> WideGrid {
    let tile_m = variant.tile_m();
    let gx = n.div_ceil(64);
    match variant.remainder() {
        Some((tail_m, _))
            if m_s >= tile_m && !m_s.is_multiple_of(tile_m) && m_s % tile_m <= tail_m =>
        {
            WideGrid {
                gx,
                full: m_s / tile_m,
                tail: 1,
            }
        }
        _ => WideGrid {
            gx,
            full: m_s.div_ceil(tile_m),
            tail: 0,
        },
    }
}

/// Mirrors a region of the f32 arena into the f16 shadow buffer.
/// Used before `matmul_coop16` for the matmul's activation operand
/// (intermediate activations don't go through `set_param` /
/// `write_f32`, so they aren't in the f16 buffer otherwise).
pub fn cast_f32_to_f16_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(CAST_F32_TO_F16.get_or_init(|| {
        build_kernel_cast_f32_to_f16(
            device,
            "rlx-wgpu cast_f32_to_f16",
            CAST_F32_TO_F16_WGSL,
            "cast_f32_to_f16",
        )
    }))
}
pub fn binary_kernel(device: &wgpu::Device) -> &'static Kernel {
    BINARY.get_or_init(|| build_kernel(device, "rlx-wgpu binary", BINARY_WGSL, "binary"))
}
pub fn snake_beta_kernel(device: &wgpu::Device) -> &'static Kernel {
    SNAKE_BETA
        .get_or_init(|| build_kernel(device, "rlx-wgpu snake_beta", SNAKE_BETA_WGSL, "snake_beta"))
}
pub fn binary_c64_kernel(device: &wgpu::Device) -> &'static Kernel {
    BINARY_C64.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu binary_c64",
            BINARY_C64_WGSL,
            "binary_c64_main",
        )
    })
}
pub fn complex_cast_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_CAST.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_cast",
            COMPLEX_CAST_WGSL,
            "complex_cast_main",
        )
    })
}
pub fn complex_norm_sq_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_NORM_SQ.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_norm_sq",
            COMPLEX_WIRINGER_WGSL,
            "complex_norm_sq",
        )
    })
}
pub fn complex_norm_sq_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_NORM_SQ_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_norm_sq_backward",
            COMPLEX_WIRINGER_WGSL,
            "complex_norm_sq_backward",
        )
    })
}
pub fn conjugate_c64_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONJUGATE_C64.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conjugate_c64",
            COMPLEX_WIRINGER_WGSL,
            "conjugate_c64",
        )
    })
}
pub fn fft_butterfly_stage_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_BUTTERFLY_STAGE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_butterfly_stage",
            FFT_BUTTERFLY_STAGE_WGSL,
            "fft_butterfly_stage",
        )
    })
}
pub fn unary_kernel(device: &wgpu::Device) -> &'static Kernel {
    UNARY.get_or_init(|| build_kernel(device, "rlx-wgpu unary", UNARY_WGSL, "unary"))
}
pub fn unary_f16_mirror_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(UNARY_F16_MIRROR.get_or_init(|| {
        build_kernel_f32_rw_uniform_f16_rw(
            device,
            "rlx-wgpu unary_f16_mirror",
            UNARY_F16_MIRROR_WGSL,
            "unary_f16_mirror",
        )
    }))
}
pub fn compare_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPARE.get_or_init(|| build_kernel(device, "rlx-wgpu compare", COMPARE_WGSL, "compare"))
}
pub fn where_kernel(device: &wgpu::Device) -> &'static Kernel {
    WHEREK.get_or_init(|| build_kernel(device, "rlx-wgpu where", WHERE_WGSL, "where_select"))
}
pub fn fma_kernel(device: &wgpu::Device) -> &'static Kernel {
    FMAK.get_or_init(|| build_kernel(device, "rlx-wgpu fma", FMA_WGSL, "fma_main"))
}
pub fn activation_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ACTIVATION_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu activation_backward",
            ACTIVATION_BACKWARD_WGSL,
            "activation_backward",
        )
    })
}
pub fn reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    REDUCE.get_or_init(|| build_kernel(device, "rlx-wgpu reduce", REDUCE_WGSL, "reduce"))
}
pub fn softmax_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX.get_or_init(|| build_kernel(device, "rlx-wgpu softmax", SOFTMAX_WGSL, "softmax"))
}
pub fn softmax_cross_entropy_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy",
            SOFTMAX_CROSS_ENTROPY_WGSL,
            "softmax_cross_entropy",
        )
    })
}
pub fn softmax_cross_entropy_with_logits_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy_with_logits",
            SOFTMAX_CROSS_ENTROPY_WGSL,
            "softmax_cross_entropy_with_logits",
        )
    })
}
pub fn softmax_cross_entropy_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy_backward",
            SOFTMAX_CROSS_ENTROPY_BWD_WGSL,
            "softmax_cross_entropy_backward",
        )
    })
}
pub fn maxpool2d_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAXPOOL2D_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu maxpool2d_backward",
            MAXPOOL2D_BWD_WGSL,
            "maxpool2d_backward",
        )
    })
}
pub fn maxpool3d_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAXPOOL3D_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu maxpool3d_backward",
            MAXPOOL3D_BWD_WGSL,
            "maxpool3d_backward",
        )
    })
}
pub fn conv3d_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv3d_backward_input",
            CONV3D_BWD_INPUT_WGSL,
            "conv3d_backward_input",
        )
    })
}
pub fn conv3d_backward_weight_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D_BWD_WEIGHT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv3d_backward_weight",
            CONV3D_BWD_WEIGHT_WGSL,
            "conv3d_backward_weight",
        )
    })
}
pub fn group_norm_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_input",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_input",
        )
    })
}
pub fn group_norm_backward_gamma_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_GAMMA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_gamma",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_gamma",
        )
    })
}
pub fn group_norm_backward_beta_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_BETA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_beta",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_beta",
        )
    })
}
pub fn axial_rope2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    AXIAL_ROPE2D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu axial_rope2d",
            AXIAL_ROPE2D_WGSL,
            "axial_rope2d",
        )
    })
}
pub fn fake_quantize_fixed_kernel(device: &wgpu::Device) -> &'static Kernel {
    FAKE_QUANTIZE_FIXED.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fake_quantize_fixed",
            FAKE_QUANTIZE_WGSL,
            "fake_quantize_fixed",
        )
    })
}
pub fn fake_quantize_perbatch_kernel(device: &wgpu::Device) -> &'static Kernel {
    FAKE_QUANTIZE_PERBATCH.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fake_quantize_perbatch",
            FAKE_QUANTIZE_WGSL,
            "fake_quantize_perbatch",
        )
    })
}
pub fn layernorm_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYERNORM.get_or_init(|| build_kernel(device, "rlx-wgpu layernorm", LAYERNORM_WGSL, "norm"))
}
/// Lanes one `*_vec4` norm workgroup runs, and so the width `run.rs` dispatches
/// a row's rows-of-workgroups against. Both modules declare it as their
/// `@workgroup_size` and their `WG`; the unit test below holds the three
/// together.
pub const NORM_VEC4_LANES: u32 = 256;

pub fn layernorm_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYERNORM_VEC4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layernorm vec4",
            LAYERNORM_VEC4_WGSL,
            "norm",
        )
    })
}
pub fn narrow_split_kernel(device: &wgpu::Device) -> &'static Kernel {
    NARROW_SPLIT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu narrow_split",
            NARROW_SPLIT_WGSL,
            "narrow_split",
        )
    })
}
pub fn concat_many_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONCAT_MANY.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu concat_many",
            CONCAT_MANY_WGSL,
            "concat_many",
        )
    })
}
pub fn norm_rope_kernel(device: &wgpu::Device) -> &'static Kernel {
    NORM_ROPE
        .get_or_init(|| build_kernel(device, "rlx-wgpu norm_rope", NORM_ROPE_WGSL, "norm_rope"))
}
pub fn group_norm_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM
        .get_or_init(|| build_kernel(device, "rlx-wgpu group_norm", GROUP_NORM_WGSL, "group_norm"))
}
pub fn layer_norm2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM2D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm2d",
            LAYER_NORM2D_WGSL,
            "layer_norm2d",
        )
    })
}
pub fn rms_norm_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    RMS_NORM_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rms_norm_bwd",
            RMS_NORM_BWD_WGSL,
            "rms_norm_bwd",
        )
    })
}
pub fn rms_norm_backward_param_kernel(device: &wgpu::Device) -> &'static Kernel {
    RMS_NORM_BWD_PARAM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rms_norm_bwd_param",
            RMS_NORM_BWD_WGSL,
            "rms_norm_bwd_param",
        )
    })
}
pub fn layer_norm_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_input",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_input",
        )
    })
}
pub fn layer_norm_backward_gamma_partial_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_GAMMA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_gamma_partial",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_gamma_partial",
        )
    })
}

pub fn layer_norm_backward_gamma_reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_GAMMA_REDUCE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_gamma_reduce",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_gamma_reduce",
        )
    })
}
pub fn cumsum_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM_BWD
        .get_or_init(|| build_kernel(device, "rlx-wgpu cumsum_bwd", CUMSUM_BWD_WGSL, "cumsum_bwd"))
}
pub fn rope_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ROPE_BWD.get_or_init(|| build_kernel(device, "rlx-wgpu rope_bwd", ROPE_BWD_WGSL, "rope_bwd"))
}
pub fn gather_backward_zero_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ZERO.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_bwd_zero",
            GATHER_BWD_WGSL,
            "gather_bwd_zero",
        )
    })
}
pub fn gather_backward_acc_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ACC.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_bwd_acc",
            GATHER_BWD_WGSL,
            "gather_bwd_acc",
        )
    })
}
pub fn cumsum_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM.get_or_init(|| build_kernel(device, "rlx-wgpu cumsum", CUMSUM_WGSL, "cumsum"))
}
pub fn cum_scan_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUM_SCAN.get_or_init(|| build_kernel(device, "rlx-wgpu cum_scan", CUM_SCAN_WGSL, "cum_scan"))
}
pub fn fft_gpu_radix2_full_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_RADIX2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_full",
            FFT_GPU_WGSL,
            "fft_radix2_full",
        )
    })
}
/// native-gpu-fft: single-kernel on-chip FFT for n in (1024, 2048] (16 KB).
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_radix2_full_big_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_RADIX2_BIG.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_full_big",
            FFT_GPU_WGSL,
            "fft_radix2_full_big",
        )
    })
}
/// native-gpu-fft: 32 KB on-chip kernels (n<=4096). Only call when the device
/// reports >=32 KB workgroup storage — pipeline creation otherwise exceeds the
/// limit.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r2_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix2_big",
        )
    })
}
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r4_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix4_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix4_big",
        )
    })
}
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r8_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R8.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix8_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix8_big",
        )
    })
}
/// native-gpu-fft: portable 16 KB radix-4 (n<=2048); no device-limit gate.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_r4_16k_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_R4_16K.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix4_16k",
            FFT_GPU_R4_16K_WGSL,
            "fft_radix4_16k",
        )
    })
}
/// native-gpu-fft: multi-row small-n FFT (16 KB); no device-limit gate.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_multirow_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_MULTIROW.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_multirow",
            FFT_GPU_MULTIROW_WGSL,
            "fft_multirow",
        )
    })
}
pub fn fft_gpu_bit_reverse_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BITREV.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_bit_reverse",
            FFT_GPU_WGSL,
            "fft_bit_reverse",
        )
    })
}
pub fn fft_gpu_inner_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_INNER
        .get_or_init(|| build_kernel(device, "rlx-wgpu fft_inner", FFT_GPU_WGSL, "fft_inner"))
}
pub fn fft_gpu_outer_r4_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_OUTER_R4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_outer_r4",
            FFT_GPU_WGSL,
            "fft_outer_r4",
        )
    })
}
pub fn fft_gpu_outer_r2_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_OUTER_R2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_outer_r2",
            FFT_GPU_WGSL,
            "fft_outer_r2",
        )
    })
}
pub fn copy_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY.get_or_init(|| build_kernel(device, "rlx-wgpu copy", COPY_WGSL, "copy"))
}
/// Vector sibling of [`copy_kernel`]; same binding layout, `array<vec4<f32>>`
/// arena. See `copy_vec4.wgsl`.
pub fn copy_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY_VEC4
        .get_or_init(|| build_kernel(device, "rlx-wgpu copy vec4", COPY_VEC4_WGSL, "copy_vec4"))
}
/// Read-only streaming probe for `bandwidth::stream_ceiling`: binding 0 is the
/// `vec4<f32>` source (ro), binding 1 its params, binding 2 one f32 per
/// workgroup (rw). See `stream_read.wgsl`.
pub fn stream_read_kernel(device: &wgpu::Device) -> &'static Kernel {
    STREAM_READ.get_or_init(|| {
        build_kernel_ro_u_rw(
            device,
            "rlx-wgpu stream read",
            STREAM_READ_WGSL,
            "stream_read",
        )
    })
}
/// Slab → arena move for the joined GGUF packed multiply. Bindings are
/// 0 = arena window (rw), 1 = uniform, 2 = slab window (ro) — the two ends are
/// deliberately different buffers, so no usage-scope exclusion applies.
pub fn copy_from_slab_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY_FROM_SLAB.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu copy from slab",
            COPY_FROM_SLAB_WGSL,
            "copy_from_slab",
        )
    })
}
/// Vector sibling of [`copy_from_slab_kernel`]; same layout, `vec4` element
/// type on both ends.
pub fn copy_from_slab_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY_FROM_SLAB_VEC4.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu copy from slab vec4",
            COPY_FROM_SLAB_VEC4_WGSL,
            "copy_from_slab_vec4",
        )
    })
}

static RNG_NORMAL_PHILOX: OnceLock<Kernel> = OnceLock::new();
static RNG_UNIFORM_PHILOX: OnceLock<Kernel> = OnceLock::new();
static RNG_FILL_ZERO: OnceLock<Kernel> = OnceLock::new();
pub fn rng_normal_philox_kernel(device: &wgpu::Device) -> &'static Kernel {
    RNG_NORMAL_PHILOX.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rng_normal_philox",
            RNG_PHILOX_WGSL,
            "rng_normal_philox",
        )
    })
}
pub fn rng_uniform_philox_kernel(device: &wgpu::Device) -> &'static Kernel {
    RNG_UNIFORM_PHILOX.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rng_uniform_philox",
            RNG_PHILOX_WGSL,
            "rng_uniform_philox",
        )
    })
}
pub fn rng_fill_zero_kernel(device: &wgpu::Device) -> &'static Kernel {
    RNG_FILL_ZERO.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rng_fill_zero",
            RNG_PHILOX_WGSL,
            "rng_fill_zero",
        )
    })
}

static FUSED_SWIGLU: OnceLock<Kernel> = OnceLock::new();
pub fn fused_swiglu_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_SWIGLU.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_swiglu",
            FUSED_SWIGLU_WGSL,
            "fused_swiglu",
        )
    })
}

static FUSED_CONV_BIAS_ACT: OnceLock<Kernel> = OnceLock::new();
pub fn fused_conv_bias_act_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_CONV_BIAS_ACT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_conv_bias_act",
            FUSED_CONV_BIAS_ACT_WGSL,
            "fused_conv_bias_act",
        )
    })
}
pub fn cast_kernel(device: &wgpu::Device) -> &'static Kernel {
    CAST.get_or_init(|| build_kernel(device, "rlx-wgpu cast", CAST_WGSL, "cast_main"))
}
pub fn elementwise_region_kernel(device: &wgpu::Device) -> &'static Kernel {
    // Region params bind as a STORAGE buffer (not uniform) — WGSL's
    // uniform-storage spec requires 16-byte stride for `array<T, N>`,
    // which our packed `array<u32, N>` chain layout doesn't satisfy.
    // Storage allows arbitrary stride.
    ELEMENTWISE_REGION.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu elementwise_region",
            ELEMENTWISE_REGION_WGSL,
            "elementwise_region",
        )
    })
}

pub fn elementwise_region_spatial_kernel(device: &wgpu::Device) -> &'static Kernel {
    ELEMENTWISE_REGION_SPATIAL.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu elementwise_region_spatial",
            ELEMENTWISE_REGION_WGSL,
            "elementwise_region_spatial",
        )
    })
}

static BATCH_ELEMENTWISE_REGION: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();

pub fn batch_elementwise_region_kernel(device: &wgpu::Device) -> &'static Kernel {
    BATCH_ELEMENTWISE_REGION.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu batch_elementwise_region",
            ELEMENTWISE_REGION_WGSL,
            "batch_elementwise_region",
        )
    })
}

fn build_kernel_region(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // Region params: read-only storage (vs uniform).
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}
pub fn transpose_kernel(device: &wgpu::Device) -> &'static Kernel {
    TRANSPOSE
        .get_or_init(|| build_kernel_3(device, "rlx-wgpu transpose", TRANSPOSE_WGSL, "transpose"))
}
/// Vector sibling of [`transpose_kernel`]; same 3-binding layout,
/// `array<vec4<f32>>` arena. See `transpose_vec4.wgsl`.
pub fn transpose_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    TRANSPOSE_VEC4.get_or_init(|| {
        build_kernel_4(
            device,
            "rlx-wgpu transpose vec4",
            TRANSPOSE_VEC4_WGSL,
            "transpose_vec4",
        )
    })
}
pub fn narrow_kernel(device: &wgpu::Device) -> &'static Kernel {
    NARROW.get_or_init(|| build_kernel(device, "rlx-wgpu narrow", NARROW_WGSL, "narrow"))
}
/// Vector sibling of [`narrow_kernel`]; see `narrow_vec4.wgsl`.
pub fn narrow_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    NARROW_VEC4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu narrow vec4",
            NARROW_VEC4_WGSL,
            "narrow_vec4",
        )
    })
}
pub fn concat_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONCAT.get_or_init(|| build_kernel(device, "rlx-wgpu concat", CONCAT_WGSL, "concat"))
}
/// Vector sibling of [`concat_kernel`]; see `concat_vec4.wgsl`.
pub fn concat_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONCAT_VEC4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu concat vec4",
            CONCAT_VEC4_WGSL,
            "concat_vec4",
        )
    })
}
pub fn frames_kernel(device: &wgpu::Device) -> &'static Kernel {
    FRAMES.get_or_init(|| build_kernel(device, "rlx-wgpu frames", FRAMES_WGSL, "frames"))
}
pub fn gather_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER.get_or_init(|| build_kernel(device, "rlx-wgpu gather", GATHER_WGSL, "gather"))
}
/// Split-binding gather: table (ro) + uniform + idx (ro) + out (rw, separate
/// buffer). For >4 GiB arenas where the embedding output lies outside the
/// table's bind window. See `build_kernel_ro_u_ro_rw`.
pub fn gather_split_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_SPLIT.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu gather_split",
            GATHER_SPLIT_WGSL,
            "gather",
            false,
        )
    })
}
pub fn gather_axis_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_AXIS.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_axis",
            GATHER_AXIS_WGSL,
            "gather_axis",
        )
    })
}
pub fn attention_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION.get_or_init(|| {
        build_kernel_3_with(
            device,
            "rlx-wgpu attention",
            ATTENTION_WGSL,
            "attention",
            false,
        )
    })
}
/// Decode-shaped attention: one workgroup per (batch, head) and key range, the
/// sequence walked across the workgroup's lanes. Same WGSL module and same
/// three-entry bind-group layout as [`attention_kernel`], so the bind group the
/// lowerer built from that layout drives either pipeline.
pub fn attention_decode_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION_DECODE.get_or_init(|| {
        build_kernel_3_with(
            device,
            "rlx-wgpu attention_decode",
            ATTENTION_WGSL,
            "attention_decode",
            false,
        )
    })
}
/// The key-range split's reduction: one workgroup per (batch, head) merges the
/// partials the split form of [`attention_decode_kernel`] wrote. Same WGSL
/// module and same bind-group layout as that kernel — the parts buffer rides
/// binding 2 there and here.
pub fn attention_decode_merge_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION_DECODE_MERGE.get_or_init(|| {
        build_kernel_3_with(
            device,
            "rlx-wgpu attention_decode_merge",
            ATTENTION_WGSL,
            "attention_decode_merge",
            false,
        )
    })
}
/// The many-row attention route's masked scale: the score multiply's own
/// output, scaled, capped and masked in place, in front of the ordinary dense
/// softmax (`softmax_kernel`). Two-entry bind-group layout: it reads and writes
/// the arena and nothing else, so it needs neither the parts buffer the decode
/// entries carry nor their pipeline layout.
pub fn attention_mask_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION_MASK.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu attention_mask",
            ATTENTION_WGSL,
            "attention_mask",
        )
    })
}

/// One workgroup shape of [`attention_band_kernel`]: the chunk threads a row of
/// `head_dim` is split across (`C`) and the query rows one workgroup covers
/// (`BM`), so every accepted form has `rows · chunks == threads` by construction.
/// That is what lets the module's workgroup be declared from the pair rather than
/// from a table of widths.
///
/// Both figures are stored rather than re-derived from the `head_dim` the form
/// came from: a caller holds the pair the form was built with, so a mismatched
/// width is unrepresentable rather than a value the caller has to remember not to
/// re-pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBandForm {
    /// Threads a row is split across: `head_dim / ATTENTION_BAND_DIMS_PER_THREAD`,
    /// the chunks the reduce sums.
    pub chunks: u32,
    /// Query rows one workgroup covers: `threads / chunks`.
    pub rows: u32,
}

/// The form [`attention_band_kernel`] is built with at `head_dim` on an adapter
/// whose [`attention_band_threads`] is `threads`, or `None` where no form of the
/// kernel splits that width into whole rows.
///
/// The admitted *width set* does not depend on `threads`: chunks are
/// `head_dim / ATTENTION_BAND_DIMS_PER_THREAD`, so the widths the plan holds give
/// 2, 4, 8 and 16 chunks, and both thread counts are whole multiples of each. Only a
/// workgroup's row count moves — `rows = threads / chunks`, 128/64/32/16 at the
/// floor and 256/128/64/32 at the wide form for head widths 64/128/256/512.
/// That is what lets the device-free plan answer at the floor
/// ([`attention_band_form`]) without knowing which form the adapter will build.
///
/// Both halves are measured, and the table in `tests/attention_band_cost.rs`
/// carries every cell (best of seven rounds after three warm-ups, this rig):
///
/// | threads | dims/thread | BM at d=512 | global (d=512) | sliding (d=256) |
/// |---------|-------------|-------------|----------------|-----------------|
/// | 256     | 32          | 16          | 566 ms         | 46.4 ms         |
/// | 256     | 64          | 32          | 757 ms         | 58.4 ms         |
/// | 256     | 128         | 64          | 1 498 ms       | 118.4 ms        |
/// | 512     | 32          | 32          | 535 ms         | 41.9 ms         |
/// | 512     | 64          | 64          | 763 ms         | 67.9 ms         |
/// | 512     | 128         | 128         | 1 560 ms       | 133.9 ms        |
/// | 1024    | 32          | 64          | 1 385 ms       | 106 ms          |
///
/// The 256 and 512 rows are the comparison the thread choice rests on, and the
/// only two the kernel builds: 512 is the add-on (taken on an adapter that
/// reports the room, this rig's Metal) and 256 the floor every conforming adapter
/// launches, so the pair differs in nothing but `BM`.
///
/// The bytes a call re-reads are `blocks · heads · keys walked · head_dim · 4 ·
/// 2` with `blocks = ceil(seq / BM)`, so every step down that row — and every row
/// with a bigger `BM` — halves the traffic. The wall clock does not follow: 256/64
/// and 512/32 have the same `BM` (32 at d = 512) and read the same 34.4 GB, and
/// the first is 41% *slower* because a thread's Q and O doubled to 128 f32, which
/// halves how many workgroups a core holds. 512/32 doubles `BM` with a thread's
/// registers untouched, and that is the only arrangement here that pays for its
/// bytes: 5% at the global cell and 10% at the sliding one — the figures the
/// wide form is worth on this rig, and the reason the floor is a fallback rather
/// than the shape. 1024/32 quarters the bytes again and loses 2.4× — one
/// workgroup a core and the key block squeezed to 4. So this walk is bound by
/// what each thread issues and by how many workgroups a core holds, not by the
/// bytes it streams; the note in the kernel header carries that reading forward.
///
/// `chunks < 2` is never admitted: the reduce's four-accumulator form reads
/// `part_s[(r·C + cc + 1)…]`, which at `C == 1` is the next row's partials (and
/// past the array on the last row), so a width that would build a one-chunk row
/// is refused by `backend::attention_route::band_attention` instead.
pub fn attention_band_form_at(head_dim: u32, threads: u32) -> Option<AttentionBandForm> {
    if head_dim == 0 || !head_dim.is_multiple_of(ATTENTION_BAND_DIMS_PER_THREAD) {
        return None;
    }
    let chunks = head_dim / ATTENTION_BAND_DIMS_PER_THREAD;
    (chunks >= 2 && threads.is_multiple_of(chunks)).then_some(AttentionBandForm {
        chunks,
        rows: threads / chunks,
    })
}

/// [`attention_band_form_at`] at the guaranteed floor thread count,
/// [`ATTENTION_BAND_MIN_THREADS`]: the device-free answer the plan
/// (`backend::attention_route::band_attention`) is decided on and the widths a
/// test holds the tile arithmetic to. Valid for a pipeline built at either thread
/// count, because the admitted widths are the same at both — the rows a device
/// actually runs come from [`attention_band_rows_for`] at its own
/// [`attention_band_threads`].
pub fn attention_band_form(head_dim: u32) -> Option<AttentionBandForm> {
    attention_band_form_at(head_dim, ATTENTION_BAND_MIN_THREADS)
}

/// The module [`attention_band_kernel`] builds: the tile the adapter's workgroup
/// storage affords ([`attention_band_key_block_within`]), the constants the body
/// names, the body with its `// @unroll` blocks expanded and its staged-tile type
/// bound, and `enable f16` ahead of them where the tile is 16-bit (a global
/// directive must lead the module). Split out so the text a device is handed is
/// the text a device-free test can hand to naga **at the same storage budget** —
/// this shader is assembled, not read from a file, so nothing else would catch a
/// WGSL mistake before an adapter does.
fn attention_band_module(head_dim: u32, threads: u32, budget: u32, f16_stage: bool) -> String {
    let block =
        attention_band_key_block_within(head_dim, if f16_stage { 2 } else { 4 }, budget, threads);
    attention_band_source(head_dim, threads, block, f16_stage)
}

fn attention_band_source(head_dim: u32, threads: u32, block: u32, f16_stage: bool) -> String {
    let form = attention_band_form_at(head_dim, threads).unwrap_or_else(|| {
        panic!(
            "rlx-wgpu: the band kernel has no form for head_dim {head_dim} at \
             {threads} threads"
        )
    });
    let body = bind_band_stage(
        ATTENTION_BAND_WGSL,
        if f16_stage { "vec4<f16>" } else { "vec4<f32>" },
    );
    let body = expand_band_loops(&body, block, ATTENTION_BAND_DIMS_PER_THREAD / 4);
    let constants = format!(
        "const BM: u32 = {rows}u;\nconst C: u32 = {chunks}u;\n\
         const DPER: u32 = {dims}u;\nconst THREADS: u32 = {threads}u;\n\
         const BK: u32 = {block}u;\n",
        rows = form.rows,
        chunks = form.chunks,
        dims = ATTENTION_BAND_DIMS_PER_THREAD,
    );
    if f16_stage {
        format!("enable f16;\n{constants}{body}")
    } else {
        format!("{constants}{body}")
    }
}

/// The two-sided band kernel for `head_dim` — `Step::Attention`'s pipeline for a
/// `MaskKind::Window` step, and for a dense `MaskKind::None` step whose extents
/// are square (the plan routes it here with `window = seq_k`, which the body
/// reads as "no band"). No other entry point serves either: the fused and the
/// many-row forms read kind 5 as "no mask" and run a dense pass.
///
/// The module is formatted per (head width, workgroup width) because the tile
/// constants are compile-time inside it: `C = head_dim / DPER` is what makes a
/// thread's Q/O slice a fixed `ATTENTION_BAND_DIMS_PER_THREAD` f32 it can hold in
/// registers (the
/// fused entry's `MAX_HEAD_DIM`-wide arrays behind a runtime index are the spill
/// this form exists to avoid), and `@workgroup_size` is `THREADS`, which comes
/// from [`attention_band_threads`] — the widest the adapter admits, and the floor
/// where it does not. One process (one device) therefore holds one module per
/// width it meets — the four head widths the plan admits, 64 to
/// [`ATTENTION_BAND_MAX_HEAD_DIM`] — and the cache is keyed by `(width, threads,
/// staging)`, so at most sixteen.
///
/// The body's hot loops are then expanded (`expand_band_loops`) because WGSL
/// has no way to ask for an unrolled loop: a local array behind a loop index is
/// thread-local memory in the module wgpu builds, not a register array.
///
/// The staged tile follows the device: `Features::SHADER_F16` selects a
/// `vec4<f16>` tile (16-bit operands with f32 accumulation), and an adapter
/// without it takes `vec4<f32>` — the same body over an f32 tile, substituted
/// into the one [`ATTENTION_BAND_WGSL`] by `bind_band_stage`.
/// [`RLX_WGPU_ATTENTION_BAND_F32`](self) forces the f32 tile on a device that
/// has the feature, so the two readings are re-takeable on one machine. Neither
/// is a dense fallback: the plan admits the shape before either is asked for.
pub fn attention_band_kernel(device: &wgpu::Device, head_dim: u32) -> &'static Kernel {
    static BAND: OnceLock<Mutex<HashMap<(u32, u32, bool), &'static Kernel>>> = OnceLock::new();
    let f16_stage = !rlx_ir::env::flag("RLX_WGPU_ATTENTION_BAND_F32")
        && device.features().contains(wgpu::Features::SHADER_F16);
    let threads = attention_band_threads(device);
    let key = (head_dim, threads, f16_stage);
    let mut cache = BAND
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        // Poison-tolerant: a build that panicked must report its own message on
        // the next call rather than a lock error.
        .unwrap_or_else(|e| e.into_inner());
    if let Some(kernel) = cache.get(&key) {
        return kernel;
    }
    // The thread count comes from the adapter's limit
    // ([`attention_band_threads`]) and the rows are that count over the chunks;
    // the dispatch grid takes the same figure from the same limit
    // ([`attention_band_rows_for`]), so the two agree by construction and there
    // is no limit left to assert against here. The key block is the largest one
    // whose tile fits this adapter's workgroup storage, which is the one thing
    // about the module that is the device's own.
    let source = attention_band_module(
        head_dim,
        threads,
        device.limits().max_compute_workgroup_storage_size,
        f16_stage,
    );
    let label = if f16_stage {
        "rlx-wgpu attention_band"
    } else {
        "rlx-wgpu attention_band_f32"
    };
    // One leak per (width, thread count, staging) triple, the bounded set the
    // cache holds: the returned reference outlives any graph that asked for it.
    let kernel: &'static Kernel = Box::leak(Box::new(build_kernel_3_with(
        device,
        label,
        &source,
        "attention_band",
        false,
    )));
    cache.insert(key, kernel);
    kernel
}

/// Query rows one workgroup of [`attention_band_kernel`] takes (`BM` in the
/// shader), for a head of `head_dim`, at an explicit workgroup width: a thread
/// owns [`ATTENTION_BAND_DIMS_PER_THREAD`] dims and a row is `head_dim` wide, so
/// the rows follow `threads / chunks`. Infallible for a width the plan admitted:
/// a width with no form is one the plan refused. The run loop builds its grid
/// from this at the adapter's own [`attention_band_threads`], so the grid and the
/// module's `BM` are one number by construction.
pub fn attention_band_rows_for(head_dim: u32, threads: u32) -> u32 {
    attention_band_form_at(head_dim, threads)
        .unwrap_or_else(|| {
            panic!(
                "rlx-wgpu: the band kernel has no form for head_dim {head_dim} at \
                 {threads} threads"
            )
        })
        .rows
}

/// The plan-side row count: [`attention_band_rows_for`] at the floor thread count
/// ([`ATTENTION_BAND_MIN_THREADS`]), because the plan that admits a band step is
/// device-free. 32 at head width 256 (the model's sliding layers) and 16 at 512
/// (its global ones).
///
/// This is what `backend::attention_route::band_attention` reads to bound a grid,
/// and what a device-free test can hold the tile arithmetic to. The rows a run
/// actually dispatches are the same call at the adapter's own
/// [`attention_band_threads`] — twice this figure where the wide form is taken.
pub fn attention_band_rows(head_dim: u32) -> u32 {
    attention_band_rows_for(head_dim, ATTENTION_BAND_MIN_THREADS)
}

/// Threads one workgroup of [`attention_band_kernel`] runs where the adapter
/// admits them — the add-on, not the requirement.
///
/// 512, not the 256 this file shipped with: a workgroup's rows are
/// `threads / chunks`, so doubling the thread count doubles `BM` — and halves the
/// bytes the walk re-reads — while a thread keeps the same 32 dims of Q and O. It
/// is the widest form that pays: 1024 threads quarters the bytes again and loses
/// 2.4× (one workgroup a core, and the key block squeezed to 4). 512 is also
/// above WebGPU's guaranteed `maxComputeInvocationsPerWorkgroup` of 256, so it is
/// taken only where the adapter reports the room ([`attention_band_threads`]),
/// and the route falls back to [`ATTENTION_BAND_MIN_THREADS`] below that; 1024 is
/// what this rig's Metal reports and what `fft_gpu.wgsl` already takes.
pub const ATTENTION_BAND_MAX_THREADS: u32 = 512;

/// Threads one workgroup of [`attention_band_kernel`] runs where the adapter
/// cannot launch [`ATTENTION_BAND_MAX_THREADS`]: the width every conforming
/// WebGPU adapter can dispatch, because the spec guarantees
/// `maxComputeInvocationsPerWorkgroup >= 256`.
///
/// It is also the width the device-free answers are taken at
/// ([`attention_band_form`], [`attention_band_rows`]) — the plan decides a band
/// step without a device, so it has to answer at the floor every device meets.
pub const ATTENTION_BAND_MIN_THREADS: u32 = 256;

/// The workgroup width to build [`attention_band_kernel`] with on `device`:
/// [`ATTENTION_BAND_MAX_THREADS`] where its
/// `maxComputeInvocationsPerWorkgroup` holds that many invocations, else
/// [`ATTENTION_BAND_MIN_THREADS`].
///
/// This is the one term that turns the wide form into an add-on: the plan that
/// chooses the band route is device-free, and the admitted head widths are the
/// same at either count (only `BM` moves — see [`attention_band_form_at`]), so an
/// adapter that cannot launch 512 loses rows per workgroup, not the route. Below
/// the floor there is nothing to fall back to: no conforming adapter reports
/// fewer than 256, so an adapter that does is refused here, naming the limit,
/// rather than building a module its HAL would fail to launch.
pub fn attention_band_threads(device: &wgpu::Device) -> u32 {
    let invocations = device.limits().max_compute_invocations_per_workgroup;
    if invocations >= ATTENTION_BAND_MAX_THREADS {
        ATTENTION_BAND_MAX_THREADS
    } else if invocations >= ATTENTION_BAND_MIN_THREADS {
        ATTENTION_BAND_MIN_THREADS
    } else {
        panic!(
            "rlx-wgpu: the band kernel needs a workgroup of at least \
             {ATTENTION_BAND_MIN_THREADS} invocations, and this adapter reports \
             maxComputeInvocationsPerWorkgroup = {invocations}; no conforming WebGPU \
             adapter is below that floor"
        )
    }
}

/// Dims one thread of [`attention_band_kernel`] owns (`DPER` in the shader).
/// Held as `vec4` slots: Q, O and the V rows it accumulates are `DPER` f32 each,
/// all unrolled into registers, so `head_dim / this` is the chunk count — 16 at
/// head width 512, 8 at 256, 4 at 128, 2 at 64.
///
/// 32 at every width, at every thread count: the wider chunks halve the traffic
/// and lose 1.3–2.6×, because a second 32-f32 slice of Q and O a thread is what
/// halves the workgroups a core holds. [`attention_band_form_at`] carries the table.
pub const ATTENTION_BAND_DIMS_PER_THREAD: u32 = 32;

/// Head width above which no band step is served — the fused kernel's own
/// `MAX_HEAD_DIM`, and the widest head the form's chunk widths split into whole
/// rows. A width that is not a multiple of [`ATTENTION_BAND_DIMS_PER_THREAD`] is
/// refused with it: the shader's chunk is a constant, so a width it was not
/// built for would read the wrong dims rather than merely run slowly.
pub const ATTENTION_BAND_MAX_HEAD_DIM: u32 = 512;

/// Head width below which no band step is served: the narrowest width whose chunk
/// count is at least two. A row split across a single chunk has nothing for the
/// reduce to sum, and the pair form it would take reads
/// `part_s[(r·C + cc + 1)·BK + kk]` — the next query row's partials, and past the
/// array on the last row. A 32-wide head is the width that would build that form,
/// so it is refused by name (see `backend::attention_route::band_attention`)
/// rather than answered from the wrong row.
pub const ATTENTION_BAND_MIN_HEAD_DIM: u32 = 64;

/// Keys one block of [`attention_band_kernel`] stages (`BK` in the shader) for
/// one configuration: the largest power of two in `{32, 16, 8, 4, 2}` whose
/// per-key cost — the K slot in the staged tile (reused for V), the key's share of
/// the score partials and of the block's scores — fits `budget`, less a sixteenth
/// left unspent for driver padding.
///
/// `budget` is the adapter's `maxComputeWorkgroupStorageSize`, `stage_bytes` the
/// staged tile's width a dim (2 for the f16 module, 4 for the f32 one) and
/// `threads` the workgroup the module is built with
/// ([`attention_band_threads`]). The kernel asks at its own adapter's figures;
/// the plan-side and device-free tests ask at WebGPU's *guaranteed* floor —
/// `16 384 B` of workgroup storage and [`ATTENTION_BAND_MIN_THREADS`] threads —
/// which is also what an adapter below this rig's 32 KiB degrades to.
///
/// At the floor the partials are `threads · BK` f32 whatever the head width, so
/// they dominate but no longer fix every width at one block: 8 keys a block for
/// the f16 tile at head widths 64, 128 and 256 (13.0, 12.0 and 13.0 KiB) and 4 at
/// 512 (8.25 KiB), where 8 at 512 would be 16.5 KiB and over the budget. The
/// kernel's own figure on this rig's 32 768 B and 512 threads is 8 keys a block at
/// every width. The degradation is of the walk, never of the answer: a smaller
/// block only means more barriers. The f32 tile is twice as wide a dim, so on a
/// 16 KiB adapter it stages 8, 8 and 4 keys at head widths 64, 128 and 256 and 4
/// at 512 (12.25 KiB at the widest), and the wide form — whose partials are
/// `threads · BK` f32, twice the floor's — takes fewer keys than the floor does at
/// one budget: 4 against 8 at head width 512 on the 16 KiB floor, and 8 against
/// 16 at head widths 64–256 on this rig's 32 KiB.
///
/// The floor is 2 keys rather than 4: on the 512-thread form the f32 tile's
/// partials are `512 · BK` f32 on their own, so a 4-key block at head width 512
/// needs 16.9 KiB and would not build on the 16 KiB floor. 2 keys is a legal
/// block everywhere — the reduce covers keys `c, c + C, …` and the host expands
/// the body for whatever block this returns.
pub fn attention_band_key_block_within(
    head_dim: u32,
    stage_bytes: u32,
    budget: u32,
    threads: u32,
) -> u32 {
    let form = attention_band_form_at(head_dim, threads).unwrap_or_else(|| {
        panic!(
            "rlx-wgpu: the band kernel has no form for head_dim {head_dim} at \
             {threads} threads"
        )
    });
    let chunks = form.chunks;
    let rows = form.rows;
    // A staged key costs its K slot plus its own partials and its score entry.
    let per_key = head_dim * stage_bytes + rows * chunks * 4 + rows * 4;
    // A sixteenth of the budget is left unspent: the figures above are the three
    // arrays' payload, and a driver is free to pad each declaration.
    let limit = budget - budget / 16;
    let mut block = 32;
    while block > 2 && block * per_key > limit {
        block /= 2;
    }
    block
}

/// Bind the band body's staged-tile type: every `%STAGE%` becomes `stage`
/// (`vec4<f16>` or `vec4<f32>`).
///
/// This is the whole difference between the 16-bit-staging module and the f32
/// one — the tile's declaration and the store into it — so the two are one
/// source file and one substitution rather than two bodies that can drift. The
/// read side needs no marker: `vec4<f32>(stage[...])` widens an f16 slot and is
/// an identity constructor over an f32 one. A marker left in place reaches naga
/// as an unknown identifier, so a missed substitution fails the pipeline build.
pub(crate) fn bind_band_stage(body: &str, stage: &str) -> String {
    let out = body.replace("%STAGE%", stage);
    assert!(
        !out.contains("%STAGE%"),
        "a band stage marker survived its substitution"
    );
    out
}

/// Repeat the band body's `// @unroll <name>` … `// @end` blocks with the index
/// as a literal: `keys` times for `kk`, then `slots` times for `j`.
///
/// This exists because WGSL has no way to ask for an unrolled loop and the
/// module wgpu builds is the one naga emits at dispatch: every `for` and `loop`
/// leaves naga with a termination guard (`while (true) { if (all(loop_bound ==
/// 0u)) break; … }`) that no backend can see through. A local array behind a
/// loop index is therefore *thread-local memory* — every `q4[j]`, `o4[j]` and
/// `score[kk]` becomes a load and a store. Measured on the global cell
/// (`tests/attention_band_cost.rs`, this rig) with nothing else changed: the
/// slot loops alone 2 223 ms as loops against 1 196 ms expanded, and 656 ms with
/// the key loops expanded too. Same code, same shapes: this is how a WGSL loop
/// reaches the driver, not a property of one GPU family.
///
/// Deliberately not a general preprocessor — one marker form, nesting by block
/// depth, and the only substitution is `%<name>%` → `<n>` (the source keeps the
/// `u` suffix). A marker left in
/// place reaches naga as an invalid literal, so a missed expansion fails the
/// pipeline build rather than computing something else.
pub(crate) fn expand_band_loops(body: &str, keys: u32, slots: u32) -> String {
    let keys = expand_band_block(body, "kk", keys);
    expand_band_block(&keys, "j", slots)
}

/// One [`expand_band_loops`] pass: expand the blocks marked `name` and leave the
/// others in place, markers intact, for their own pass.
fn expand_band_block(src: &str, name: &str, count: u32) -> String {
    // `(indent, marker)` a depth of nesting, `(lines)` its collected body.
    let mut open: Vec<(String, Vec<String>)> = Vec::new();
    let mut out = String::with_capacity(src.len() * 2);
    // A rendered block belongs to the enclosing `// @unroll` block when there is
    // one — nesting is indentation-free, so the parent is wherever we are.
    fn emit(open: &mut [(String, Vec<String>)], out: &mut String, text: &str) {
        match open.last_mut() {
            Some((_, block)) => block.push(text.to_string()),
            None => {
                out.push_str(text);
                out.push('\n');
            }
        }
    }
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(inner) = trimmed.strip_prefix("// @unroll ") {
            let indent = &line[..line.len() - trimmed.len()];
            open.push((format!("{indent}|{}", inner.trim()), Vec::new()));
        } else if trimmed == "// @end" {
            let (stamp, block) = open.pop().expect("unbalanced // @end in a band kernel");
            let (indent, inner) = stamp.split_once('|').expect("block stamp");
            if inner == name {
                let find = format!("%{name}%");
                for n in 0..count {
                    for body_line in &block {
                        emit(
                            &mut open,
                            &mut out,
                            &body_line.replace(&find, &format!("{n}")),
                        );
                    }
                }
            } else {
                // Another pass's block: hand it back on, markers intact, so the
                // enclosing pass can substitute into it and its own pass can
                // expand it.
                emit(&mut open, &mut out, &format!("{indent}// @unroll {inner}"));
                for body_line in &block {
                    emit(&mut open, &mut out, body_line);
                }
                emit(&mut open, &mut out, &format!("{indent}// @end"));
            }
        } else {
            emit(&mut open, &mut out, line);
        }
    }
    assert!(
        open.is_empty(),
        "unbalanced // @unroll {name} in a band kernel: {} block(s) left open",
        open.len()
    );
    assert!(
        !out.contains(&format!("%{name}%")),
        "a %{name}% marker survived its own expansion pass"
    );
    out
}

/// Head width the lane-parallel decode form serves, mirroring the native
/// backend's decode kernel. That cap is the native kernel's own — 32 lanes of
/// partial O in one threadgroup array. Here the per-lane O array is the serial
/// kernel's and the merge scratch is dim-chunked, so this is coverage parity
/// with the native route, not a limit of this kernel. Wider heads stay on
/// `attention`.
pub const ATTENTION_DECODE_MAX_HEAD_DIM: u32 = 128;

/// Value dims per pass of the decode entry's merge — the shader's
/// `DECODE_DIM_CHUNK`, and the unit the key-range split's partials are padded
/// to (`2 + pad` f32 a partial). The host sizes that buffer from this number and
/// the kernel writes it from the shader's own; `the_split_partials_pad_is_the_
/// shaders_own` pins the two together.
pub const ATTENTION_PARTS_DIM_CHUNK: u32 = 32;

/// Which attention kernel `Step::Attention` takes, from the extents the run
/// actually walks (`seq_q`/`seq_k` after active-extent scaling) rather than the
/// lowered ones: every one-row step whose head widths this form holds. What
/// keeps the serial kernel is a query row count above one (the dense route
/// takes two and up), a step with no keys, and a head wider than the cap above;
/// which mask kinds exist is decided inside the WGSL entry point, which mirrors
/// the serial one's. The two widths are only each compared against the cap, so
/// their order is immaterial.
///
/// There is deliberately **no key floor**. The one this route used to carry (64
/// keys, the figure the native backend records for its own decode kernel) was a
/// shape assumption these measurements do not support: the serial form walks the
/// keys on one thread per query row, with `MAX_HEAD_DIM`-wide private Q/O arrays
/// behind a runtime index, so its per-layer cost grows with the key count rather
/// than disappearing at short lengths — 498 µs a layer at 32 keys and 942 at 64
/// on the stand (Apple M3 Pro, wgpu over Metal, the chain
/// `tests/attention_decode_route.rs` walks), where those two are the pre-change
/// build's own readings and cannot be re-taken now that a one-row step at those
/// widths takes this form. The fork's speech cells live in the band the floor
/// kept — the code predictor's attention is one row over 1..=16 keys (one cache
/// row per level) at 16 heads of width 128 — and routing it here took that frame
/// from 56.8 to 40.8–42.0 ms and the synthesis cell from 1.053 s per second of
/// speech, with the waveform digest unchanged; `MEASUREMENTS.md` carries the rate
/// it reached, which moves with the machine between runs. The same frame's
/// attention dispatches read 307 µs and 154.7 µs before and after in the
/// one-op-per-pass ablation, which charges each dispatch a pass open and close of
/// its own and so prices small kinds far above their device cost: read those two
/// as the ablation's, indicative of the direction, not as per-dispatch device
/// time (the cell rate is the measured one).
pub fn attention_decode_eligible(seq_q: u32, seq_k: u32, head_dim: u32, v_head_dim: u32) -> bool {
    seq_q == 1
        // Not a floor: a step with no keys has nothing for this form to walk.
        && seq_k > 0
        && head_dim <= ATTENTION_DECODE_MAX_HEAD_DIM
        // 0 means "unset": the kernel then reads `head_dim` for V, already
        // bounded by the conjunct above.
        && v_head_dim <= ATTENTION_DECODE_MAX_HEAD_DIM
}

/// Which kernel each `Step::Attention` dispatch took, counted process-wide, plus
/// the many-row route (`AttentionRoutes::decomposed`), which is not a
/// `Step::Attention` dispatch at all: it is counted where the route's own
/// normalise runs, one per attention step it served. `serial` covers both the
/// multi-row path and every shape the decode form cannot serve. `band` is the
/// two-sided band kernel ([`attention_band_kernel`]) — a `Step::Attention`
/// dispatch like `decode`/`serial`, and disjoint from both: a `MaskKind::Window`
/// step takes it whatever its extents, since neither of the other two computes
/// kind 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionRoutes {
    pub decode: u64,
    pub decomposed: u64,
    pub serial: u64,
    pub band: u64,
}

static ATTENTION_DECODE_STEPS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_SERIAL_STEPS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_DECOMPOSED_STEPS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_BAND_STEPS: AtomicU64 = AtomicU64::new(0);

/// How many attention dispatches this process has routed each way. The engine's
/// dispatch report answers per `OpKind`, not per kernel, so this is the
/// wgpu-local answer for *which* attention kernel a step took.
pub fn attention_route_counts() -> AttentionRoutes {
    AttentionRoutes {
        decode: ATTENTION_DECODE_STEPS.load(Ordering::Relaxed),
        decomposed: ATTENTION_DECOMPOSED_STEPS.load(Ordering::Relaxed),
        serial: ATTENTION_SERIAL_STEPS.load(Ordering::Relaxed),
        band: ATTENTION_BAND_STEPS.load(Ordering::Relaxed),
    }
}

/// Zero [`attention_route_counts`] (tests bracketing one run).
pub fn reset_attention_route_counts() {
    ATTENTION_DECODE_STEPS.store(0, Ordering::Relaxed);
    ATTENTION_SERIAL_STEPS.store(0, Ordering::Relaxed);
    ATTENTION_DECOMPOSED_STEPS.store(0, Ordering::Relaxed);
    ATTENTION_BAND_STEPS.store(0, Ordering::Relaxed);
}

pub(crate) fn note_attention_route(decode: bool) {
    let counter = if decode {
        &ATTENTION_DECODE_STEPS
    } else {
        &ATTENTION_SERIAL_STEPS
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// One `Step::Attention` dispatch served by the band kernel — counted where the
/// pipeline is chosen, so it never joins `serial`'s tally.
pub(crate) fn note_attention_band_route() {
    ATTENTION_BAND_STEPS.fetch_add(1, Ordering::Relaxed);
}

/// One attention step served by the many-row route — counted at its normalise,
/// which every such step emits exactly one of.
pub(crate) fn note_attention_decomposed_route() {
    ATTENTION_DECOMPOSED_STEPS.fetch_add(1, Ordering::Relaxed);
}

pub fn attention_bwd_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu attention_bwd",
            ATTENTION_BWD_WGSL,
            "attention_bwd",
        )
    })
}
pub fn rope_kernel(device: &wgpu::Device) -> &'static Kernel {
    ROPE.get_or_init(|| build_kernel(device, "rlx-wgpu rope", ROPE_WGSL, "rope"))
}
pub fn expand_kernel(device: &wgpu::Device) -> &'static Kernel {
    EXPAND.get_or_init(|| build_kernel_3(device, "rlx-wgpu expand", EXPAND_WGSL, "expand"))
}
pub fn reverse_kernel(device: &wgpu::Device) -> &'static Kernel {
    REVERSE.get_or_init(|| build_kernel_3(device, "rlx-wgpu reverse", REVERSE_WGSL, "reverse"))
}
pub fn quantize_i8_kernel(device: &wgpu::Device) -> &'static Kernel {
    QUANTIZE_I8
        .get_or_init(|| build_kernel(device, "rlx-wgpu quantize_i8", QUANT_I8_WGSL, "quantize_i8"))
}
pub fn dequantize_i8_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANTIZE_I8.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu dequantize_i8",
            QUANT_I8_WGSL,
            "dequantize_i8",
        )
    })
}
pub fn q_matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    Q_MATMUL.get_or_init(|| build_kernel(device, "rlx-wgpu q_matmul", Q_MATMUL_WGSL, "q_matmul"))
}
pub fn q_conv2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    Q_CONV2D.get_or_init(|| build_kernel(device, "rlx-wgpu q_conv2d", Q_CONV2D_WGSL, "q_conv2d"))
}
pub fn batch_norm_inference_kernel(device: &wgpu::Device) -> &'static Kernel {
    BN_INFER.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu batch_norm_inference",
            BATCH_NORM_INFERENCE_WGSL,
            "batch_norm_inference",
        )
    })
}
pub fn batch_norm_inference_bwd_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    BN_INFER_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu batch_norm_inference_bwd_input",
            BATCH_NORM_INFERENCE_WGSL,
            "batch_norm_inference_bwd_input",
        )
    })
}
pub fn batch_norm_inference_bwd_gamma_kernel(device: &wgpu::Device) -> &'static Kernel {
    BN_INFER_BWD_GAMMA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu batch_norm_inference_bwd_gamma",
            BATCH_NORM_INFERENCE_WGSL,
            "batch_norm_inference_bwd_gamma",
        )
    })
}
pub fn batch_norm_inference_bwd_beta_kernel(device: &wgpu::Device) -> &'static Kernel {
    BN_INFER_BWD_BETA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu batch_norm_inference_bwd_beta",
            BATCH_NORM_INFERENCE_WGSL,
            "batch_norm_inference_bwd_beta",
        )
    })
}
pub fn scaled_quant_scale_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCALED_QUANT_SCALE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scaled_quant_scale",
            SCALED_LOWP_WGSL,
            "scaled_quant_scale",
        )
    })
}
pub fn scaled_quantize_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCALED_QUANTIZE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scaled_quantize",
            SCALED_LOWP_WGSL,
            "scaled_quantize",
        )
    })
}
pub fn scaled_dequantize_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCALED_DEQUANTIZE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scaled_dequantize",
            SCALED_LOWP_WGSL,
            "scaled_dequantize",
        )
    })
}
pub fn scaled_matmul_decode_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCALED_MATMUL_DECODE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scaled_matmul_decode",
            SCALED_LOWP_WGSL,
            "scaled_matmul_decode",
        )
    })
}
pub fn gather_nd_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_ND
        .get_or_init(|| build_kernel_3(device, "rlx-wgpu gather_nd", INDEXING_ND_WGSL, "gather_nd"))
}
pub fn gather_elements_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_ELEMENTS.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu gather_elements",
            INDEXING_ND_WGSL,
            "gather_elements",
        )
    })
}
pub fn scatter_elements_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCATTER_ELEMENTS.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu scatter_elements",
            INDEXING_ND_WGSL,
            "scatter_elements",
        )
    })
}
pub fn scatter_nd_reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCATTER_ND_REDUCE.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu scatter_nd_reduce",
            INDEXING_ND_WGSL,
            "scatter_nd_reduce",
        )
    })
}
pub fn copy_sanitize_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY_SANITIZE.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu copy_sanitize",
            INDEXING_ND_WGSL,
            "copy_sanitize",
        )
    })
}
pub fn argmax_kernel(device: &wgpu::Device) -> &'static Kernel {
    ARGMAX.get_or_init(|| build_kernel(device, "rlx-wgpu argmax", ARGMAX_WGSL, "argmax"))
}
pub fn pool2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL2D.get_or_init(|| build_kernel(device, "rlx-wgpu pool2d", POOL2D_WGSL, "pool2d"))
}
pub fn conv2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV2D.get_or_init(|| build_kernel(device, "rlx-wgpu conv2d", CONV2D_WGSL, "conv2d"))
}
pub fn im2col2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    IM2COL2D.get_or_init(|| build_kernel(device, "rlx-wgpu im2col2d", IM2COL2D_WGSL, "im2col2d"))
}
/// Layout for the cooperative implicit-GEMM convolution. 64 bytes (16 u32).
///
/// The offsets are absolute arena words, not window locals: the entry reads the
/// weight, the 1-D input and its own output straight out of one whole-arena
/// binding at base zero, exactly as the im2col panel multiply does (see the
/// `Op::Conv` arm in `backend/compile/lower.rs`, which admits this route only
/// when the arena binds whole).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv1dImplicitParams {
    pub c_in: u32,
    pub c_out: u32,
    /// `c_in · kw` — the weight's row stride and the contraction.
    pub k_total: u32,
    /// The input's own length.
    pub w: u32,
    /// The output's length — the multiply's `n`.
    pub out_len: u32,
    pub kw: u32,
    pub dilation: u32,
    pub pad: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}
/// Whether this adapter can run the cooperative implicit-GEMM conv at all —
/// Metal, plus cooperative matrices. Split from the constructor so the compile
/// arm and the panel reservation can ask it without building the pipeline.
pub fn conv1d_implicit_supported(device: &wgpu::Device) -> bool {
    device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && crate::device::wgpu_device().map(|d| d.backend) == Some(wgpu::Backend::Metal)
}

/// The convolution GEMM routes' implicit form: the direct entry's 32×32 f32
/// cooperative tile with B gathered through the conv's taps into workgroup memory
/// instead of materialised as an im2col panel (see the shader header). Metal only,
/// for the same reason `matmul_coop_f32_direct_kernel` is: the load/store
/// conventions this entry stands on are the MSL ones. `None` when the device
/// cannot build it, in which case the conv keeps the im2col/tiled/direct routes.
pub fn conv1d_implicit_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !conv1d_implicit_supported(device) {
        return None;
    }
    Some(CONV1D_IMPLICIT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv1d_implicit",
            CONV1D_IMPLICIT_WGSL,
            "conv1d_implicit",
        )
    }))
}
pub fn conv1d_tiled_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV1D_TILED.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv1d_tiled",
            CONV1D_TILED_WGSL,
            "conv1d_tiled",
        )
    })
}
pub fn pool1d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL1D.get_or_init(|| build_kernel(device, "rlx-wgpu pool1d", POOL1D_WGSL, "pool1d"))
}
pub fn pool3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL3D.get_or_init(|| build_kernel(device, "rlx-wgpu pool3d", POOL3D_WGSL, "pool3d"))
}
pub fn conv1d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV1D.get_or_init(|| build_kernel(device, "rlx-wgpu conv1d", CONV1D_WGSL, "conv1d"))
}
pub fn conv3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D.get_or_init(|| build_kernel(device, "rlx-wgpu conv3d", CONV3D_WGSL, "conv3d"))
}
/// Four-output-channel register tiling; ungrouped convolution only.
pub fn conv3d_ctile_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D_CTILE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv3d_ctile",
            CONV3D_CTILE_WGSL,
            "conv3d_ctile",
        )
    })
}
pub fn conv_transpose3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV_TRANSPOSE3D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv_transpose3d",
            CONV_TRANSPOSE3D_WGSL,
            "conv_transpose3d",
        )
    })
}
/// The 2-D entry point of the same module (see `conv_transpose3d.wgsl`).
pub fn conv_transpose2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV_TRANSPOSE2D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv_transpose2d",
            CONV_TRANSPOSE3D_WGSL,
            "conv_transpose2d",
        )
    })
}
/// `Op::ConvTranspose2d`'s phase-decomposed tiled GEMM, the fast path over the
/// gather above (see `conv_transpose2d_tiled.wgsl` for both the design and the
/// family it is proven for).
pub fn conv_transpose2d_tiled_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV_TRANSPOSE2D_TILED.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv_transpose2d_tiled",
            CONV_TRANSPOSE2D_TILED_WGSL,
            "conv_transpose2d_tiled",
        )
    })
}
pub fn scatter_add_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCATTER_ADD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scatter_add",
            SCATTER_ADD_WGSL,
            "scatter_add",
        )
    })
}
pub fn topk_kernel(device: &wgpu::Device) -> &'static Kernel {
    TOPK.get_or_init(|| build_kernel(device, "rlx-wgpu topk", TOPK_WGSL, "topk"))
}
pub fn welch_peaks_gpu_kernel(device: &wgpu::Device) -> &'static Kernel {
    WELCH_PEAKS_GPU.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu welch_peaks_gpu",
            WELCH_PEAKS_GPU_WGSL,
            "welch_peaks_gpu",
        )
    })
}
pub fn umap_knn_kernel(device: &wgpu::Device) -> &'static Kernel {
    UMAP_KNN.get_or_init(|| build_kernel(device, "rlx-wgpu umap_knn", UMAP_KNN_WGSL, "umap_knn"))
}
pub fn grouped_matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUPED_MATMUL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu grouped_matmul",
            GROUPED_MATMUL_WGSL,
            "grouped_matmul",
        )
    })
}
pub fn scaled_grouped_matmul_decode_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCALED_GROUPED_MATMUL_DECODE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scaled_grouped_matmul_decode",
            SCALED_GROUPED_MATMUL_DECODE_WGSL,
            "scaled_grouped_matmul_decode",
        )
    })
}
pub fn sample_kernel(device: &wgpu::Device) -> &'static Kernel {
    SAMPLE.get_or_init(|| build_kernel(device, "rlx-wgpu sample", SAMPLE_WGSL, "sample"))
}
pub fn selective_scan_kernel(device: &wgpu::Device) -> &'static Kernel {
    SELECTIVE_SCAN.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu selective_scan",
            SELECTIVE_SCAN_WGSL,
            "selective_scan",
        )
    })
}
pub fn gated_delta_net_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_DELTA_NET.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_delta_net",
            GATED_DELTA_NET_WGSL,
            "gated_delta_net",
        )
    })
}
pub fn mamba2_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAMBA2.get_or_init(|| build_kernel(device, "rlx-wgpu mamba2", MAMBA2_WGSL, "mamba2"))
}
pub fn gru_kernel(device: &wgpu::Device) -> &'static Kernel {
    GRU.get_or_init(|| build_kernel(device, "rlx-wgpu gru", GRU_WGSL, "gru"))
}
pub fn rnn_kernel(device: &wgpu::Device) -> &'static Kernel {
    RNN.get_or_init(|| build_kernel(device, "rlx-wgpu rnn", RNN_WGSL, "rnn"))
}
pub fn lstm_kernel(device: &wgpu::Device) -> &'static Kernel {
    LSTM.get_or_init(|| build_kernel(device, "rlx-wgpu lstm", LSTM_WGSL, "lstm"))
}
pub fn dequant_matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_MATMUL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu dequant_matmul",
            DEQUANT_MATMUL_WGSL,
            "dequant_matmul",
        )
    })
}
pub fn dequant_matmul_mlx_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_MATMUL_MLX.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu dequant_matmul_mlx",
            DEQUANT_MATMUL_MLX_WGSL,
            "dequant_matmul_mlx",
        )
    })
}
pub fn dequant_gguf_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GGUF.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu dequant_gguf",
            DEQUANT_GGUF_WGSL,
            "dequant_gguf",
        )
    })
}
pub fn matmul_bt_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_BT.get_or_init(|| build_kernel(device, "rlx-wgpu matmul_bt", MATMUL_WGSL, "matmul_bt"))
}
pub fn dequant_gemv_gguf_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMV_GGUF.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemv_gguf",
            DEQUANT_GEMV_GGUF_WGSL,
            "dequant_gemv",
            true,
        )
    })
}
/// The one-row Q4_K / Q6_K / Q8_0 entry of `dequant_gemv_gguf.wgsl`: one
/// workgroup per two output columns, the workgroup's threads splitting each
/// column's contraction.
pub fn dequant_gemv_gguf_tiled_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMV_GGUF_TILED.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemv_gguf_tiled",
            DEQUANT_GEMV_GGUF_WGSL,
            "dequant_gemv_tiled",
            true,
        )
    })
}
/// The subgroup-reduction entry of `dequant_gemv_gguf.wgsl`: one subgroup per
/// output column, `GEMV_WIDE_THREADS` invocations per workgroup, the lane
/// partials combined by `subgroupAdd`.
///
/// Built from [`DEQUANT_GEMV_GGUF_WIDE_WGSL`] (the whole file) rather than
/// [`DEQUANT_GEMV_GGUF_WGSL`] (the file up to this arm's marker): only the wide
/// module names a subgroup built-in, and wgpu refuses such a module without
/// `Features::SUBGROUP`. The caller reaches this constructor only after
/// `gguf_gpu::gemv_subgroup_probe` has read a width on a device carrying the
/// feature.
pub fn dequant_gemv_gguf_wide_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMV_GGUF_WIDE.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemv_gguf_wide",
            DEQUANT_GEMV_GGUF_WIDE_WGSL,
            "dequant_gemv_wide",
            true,
        )
    })
}
/// The subgroup arm's ILP entry: `WIDE_ILP` output columns a subgroup, an
/// independent accumulator each and one X read shared between them, so a lane
/// keeps two rows' loads in flight where the entry above keeps one. Same bind
/// layout and same module as [`dequant_gemv_gguf_wide_kernel`]; the host
/// dispatches it only for a chunk whose columns are a whole number of
/// workgroups.
pub fn dequant_gemv_gguf_wide_ilp_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMV_GGUF_WIDE_ILP.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemv_gguf_wide_ilp",
            DEQUANT_GEMV_GGUF_WIDE_WGSL,
            "dequant_gemv_wide_ilp",
            true,
        )
    })
}
/// The 128-invocation shader that reads the adapter's real subgroup width and
/// subgroup count into a two-word storage buffer. One binding; the caller
/// (`gguf_gpu::gemv_subgroup_probe`) allocates the buffer and maps it back.
pub fn gguf_subgroup_probe_kernel(device: &wgpu::Device) -> &'static Kernel {
    GGUF_SUBGROUP_PROBE.get_or_init(|| {
        let label = "rlx-wgpu gguf subgroup probe";
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(GGUF_SUBGROUP_PROBE_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("gemv_subgroup_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        Kernel { pipeline, bgl }
    })
}
pub fn dequant_gemm_q1_0_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMM_Q1_0.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemm_q1_0",
            DEQUANT_GEMM_Q1_0_WGSL,
            "dequant_gemm_q1_0",
            false,
        )
    })
}
/// Multi-row Q4_K GEMM (`m > 1`): one thread per column covers 8 rows, or —
/// when the encoder puts the dispatch on the split geometry, because it would
/// otherwise be too small to fill the GPU — half a column's contraction each,
/// two threads to a column.
pub fn dequant_gemm_q4k_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMM_Q4K.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemm_q4k",
            DEQUANT_GEMM_GGUF_WGSL,
            "dequant_gemm_q4k",
            false,
        )
    })
}
/// Multi-row Q6_K GEMM (`m > 1`): one thread per column covers 8 rows.
pub fn dequant_gemm_q6k_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMM_Q6K.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemm_q6k",
            DEQUANT_GEMM_GGUF_WGSL,
            "dequant_gemm_q6k",
            false,
        )
    })
}
/// Multi-row Q8_0 GEMM (`m > 1`): one thread per column covers 8 rows. Same
/// binding set as the K-quant entries; its 34-byte blocks are read without the
/// whole-word alignment they have, through the shader's reassembling reader.
pub fn dequant_gemm_q8_0_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMM_Q8_0.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemm_q8_0",
            DEQUANT_GEMM_GGUF_WGSL,
            "dequant_gemm_q8_0",
            false,
        )
    })
}
pub fn fused_residual_ln_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_LN.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_ln",
            FUSED_RESIDUAL_LN_WGSL,
            "fused_residual_ln",
        )
    })
}
pub fn fused_residual_ln_tee_vec4_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_LN_TEE_VEC4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_ln_tee vec4",
            FUSED_RESIDUAL_LN_TEE_VEC4_WGSL,
            "fused_residual_ln_tee",
        )
    })
}
pub fn fused_residual_ln_tee_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_LN_TEE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_ln_tee",
            FUSED_RESIDUAL_LN_TEE_WGSL,
            "fused_residual_ln_tee",
        )
    })
}
pub fn fused_residual_rms_norm_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_RMS_NORM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_rms_norm",
            FUSED_RESIDUAL_RMS_NORM_WGSL,
            "fused_residual_rms_norm",
        )
    })
}
pub fn ada_layer_norm_kernel(device: &wgpu::Device) -> &'static Kernel {
    ADA_LAYER_NORM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu ada_layer_norm",
            ADA_LAYER_NORM_WGSL,
            "ada_layer_norm",
        )
    })
}
pub fn gated_residual_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_RESIDUAL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_residual",
            GATED_RESIDUAL_WGSL,
            "gated_residual",
        )
    })
}
pub fn ada_layer_norm_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ADA_LAYER_NORM_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu ada_layer_norm_backward",
            ADA_LAYER_NORM_BACKWARD_WGSL,
            "ada_layer_norm_backward",
        )
    })
}
pub fn gated_residual_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_RESIDUAL_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_residual_backward",
            GATED_RESIDUAL_BACKWARD_WGSL,
            "gated_residual_backward",
        )
    })
}
pub fn matmul_qkv_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_QKV
        .get_or_init(|| build_kernel(device, "rlx-wgpu matmul_qkv", MATMUL_QKV_WGSL, "matmul_qkv"))
}
pub fn matmul_qkv_coop_f32_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return None;
    }
    Some(MATMUL_QKV_COOP_F32.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_qkv_coop_f32",
            MATMUL_QKV_COOP_F32_WGSL,
            "matmul_qkv_coop_f32",
        )
    }))
}
pub fn matmul_qkv_coop_f16_vk_f32acc_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_QKV_COOP_F16_VK_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_qkv_coop_f16_vk_f32acc",
                MATMUL_QKV_COOP_F16_VK_F32ACC_WGSL,
                "matmul_qkv_coop_f16_vk_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_qkv_coop_f16_vk_widen_f32acc_kernel(
    device: &wgpu::Device,
) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_qkv_coop_f16_vk_widen_f32acc",
                MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC_WGSL,
                "matmul_qkv_coop_f16_vk_widen_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_qkv_coop_f16_vk_active_kernel(
    device: &wgpu::Device,
    n: u32,
) -> Option<&'static Kernel> {
    pick_coop_f16_vk_matmul(
        device,
        n,
        matmul_qkv_coop_f16_vk_kernel,
        matmul_qkv_coop_f16_vk_f32acc_kernel,
        matmul_qkv_coop_f16_vk_widen_kernel,
        matmul_qkv_coop_f16_vk_widen_f32acc_kernel,
    )
}

pub fn matmul_qkv_coop_f16_vk_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_QKV_COOP_F16_VK.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_qkv_coop_f16_vk",
            MATMUL_QKV_COOP_F16_VK_WGSL,
            "matmul_qkv_coop_f16_vk",
        )
    }))
}
pub fn matmul_qkv_coop_f16_vk_widen_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_QKV_COOP_F16_VK_WIDEN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_qkv_coop_f16_vk_widen",
            MATMUL_QKV_COOP_F16_VK_WIDEN_WGSL,
            "matmul_qkv_coop_f16_vk_widen",
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_DECODE_MAX_HEAD_DIM, ATTENTION_WGSL, CONV_TRANSPOSE2D_TILED_WGSL,
        ELEMENTWISE_REGION_WGSL, FRAMES_WGSL, FRAMES_WORKGROUP, FUSED_RESIDUAL_LN_TEE_VEC4_WGSL,
        LAYERNORM_VEC4_WGSL, MATMUL_WIDE_VEC4_WGSL, NORM_VEC4_LANES, SAMPLE_LANES, SAMPLE_STAGE,
        SAMPLE_WGSL, SNAKE_BETA_WG, SNAKE_BETA_WGSL, TILED_TILE_CO, TILED_TILE_U, WIDE_TAIL_M,
        WideGrid, WideVariant, conv_transpose2d_tiled_grid, wide_grid,
    };

    /// The widths the band route admits are the same at both workgroup widths the
    /// kernel builds: 256 (the floor every conforming adapter launches) and 512
    /// (the add-on). This is what keeps the device-free plan — which answers at
    /// the floor — valid for a pipeline the adapter resizes. Device-free: the
    /// form is arithmetic, not a pipeline, so this runs in the plain
    /// `cargo test -p rlx-wgpu --lib` gate.
    ///
    /// The refusals are the widths that would build a one-chunk row (`32`, whose
    /// `chunks == 1` breaks the reduce) or a fraction of a row
    /// (`96, 160, 192, 224, 288, 320, 352, 384, 416, 448, 480`, whose chunk count
    /// does not divide 256; `544` is past
    /// [`ATTENTION_BAND_MAX_HEAD_DIM`](super::ATTENTION_BAND_MAX_HEAD_DIM)).
    #[test]
    fn the_band_admits_the_same_widths_at_the_floor_and_the_wide_form() {
        use super::{
            ATTENTION_BAND_MAX_THREADS, ATTENTION_BAND_MIN_THREADS, attention_band_form_at,
        };
        for threads in [ATTENTION_BAND_MIN_THREADS, ATTENTION_BAND_MAX_THREADS] {
            for hd in [64u32, 128, 256, 512] {
                let form = attention_band_form_at(hd, threads)
                    .unwrap_or_else(|| panic!("hd={hd} has no form at {threads} threads"));
                assert_eq!(
                    form.chunks * form.rows,
                    threads,
                    "hd={hd} at {threads} threads must split into whole rows"
                );
            }
            for hd in [
                32u32, 96, 160, 192, 224, 288, 320, 352, 384, 416, 448, 480, 544,
            ] {
                assert!(
                    attention_band_form_at(hd, threads).is_none(),
                    "hd={hd} must have no form at {threads} threads"
                );
            }
        }
    }

    /// `frames.wgsl` parses and validates under the WebGPU baseline, and the
    /// workgroup its dispatch covers elements with is the one it declares.
    /// Device-free (naga only) and on the same reasoning as the tile pin below: a
    /// WGSL mistake would surface as a panic at `create_shader_module` on every
    /// machine whose graph takes this route rather than on whichever one happened
    /// to run it. The fold can be stood down — `RLX_WGPU_CONCAT_HOST` does, pinned
    /// in `tests/frames_concat_host_knob.rs` — but that lever only takes the concat
    /// off its direct window path rather than back to the pre-fold spelling (the
    /// crate README records the route it does take), so the pre-fold *device*
    /// spelling is what this tree cannot re-take.
    #[test]
    fn frames_shader_matches_its_dispatch_geometry() {
        let module = naga::front::wgsl::parse_str(FRAMES_WGSL).expect("wgsl parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("wgsl validate without any optional capability");
        assert_eq!(
            module
                .entry_points
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            vec!["frames"],
            "one compute entry, and it the one `run.rs` sets as the pipeline"
        );
        assert_eq!(
            module.entry_points[0].workgroup_size,
            [FRAMES_WORKGROUP, 1, 1],
            "the shader's own workgroup size must be the Rust constant `run.rs` dispatches with"
        );
        assert!(
            FRAMES_WGSL.contains(&format!("const WG: u32 = {FRAMES_WORKGROUP}u;")),
            "the shader must declare its workgroup size from `FRAMES_WORKGROUP`: it also strides \
             its invocations by it, so two copies of that number could be two different strides"
        );
    }

    /// The sampler's cooperative one-row path splits a row across `SAMPLE_LANES`
    /// and dispatches one workgroup per row with that width; a shader whose
    /// `LANES` disagreed would dispatch a partial workgroup into a reduction that
    /// assumes all of them. `STAGE` is the same kind of shared literal: it sizes
    /// the workgroup array the staged path indexes, so a shader that declared a
    /// larger `STAGE` than this would write past the storage the device grants and
    /// one that declared a smaller one would read a row it did not stage. The parse
    /// and validate is the same device-free gate the sibling shaders carry: this
    /// module's workgroup storage (the staged row, the selection tree's lists) is
    /// a layout naga can reject without a device, and a rejection would otherwise
    /// surface as a panic at `create_shader_module` on the first machine whose
    /// graph reaches a one-row draw.
    #[test]
    fn sample_shader_matches_its_dispatch_geometry() {
        let module = naga::front::wgsl::parse_str(SAMPLE_WGSL).expect("wgsl parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("sample.wgsl validates under the WebGPU baseline");
        assert!(
            SAMPLE_WGSL.contains(&format!("const LANES: u32 = {SAMPLE_LANES}u;")),
            "sample.wgsl must declare `LANES` from `SAMPLE_LANES` ({SAMPLE_LANES})"
        );
        assert!(
            SAMPLE_WGSL.contains(&format!("const STAGE: u32 = {SAMPLE_STAGE}u;")),
            "sample.wgsl must declare `STAGE` from `SAMPLE_STAGE` ({SAMPLE_STAGE})"
        );
    }

    /// `snake_beta.wgsl` parses and validates under the WebGPU baseline, and the
    /// column width its dispatch is built from is the one it declares. Device-free
    /// (naga only), on the same reasoning as the tile pins below: a WGSL mistake
    /// would otherwise surface as a panic at `create_shader_module` on the first
    /// machine whose graph reaches the folded step.
    #[test]
    fn snake_beta_shader_matches_its_dispatch_geometry() {
        let module = naga::front::wgsl::parse_str(SNAKE_BETA_WGSL).expect("wgsl parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("snake_beta.wgsl validates under the WebGPU baseline");
        assert!(
            SNAKE_BETA_WGSL.contains(&format!("@workgroup_size({SNAKE_BETA_WG})")),
            "snake_beta.wgsl must declare `@workgroup_size({SNAKE_BETA_WG})`: the run \
             divides a row by `SNAKE_BETA_WG`"
        );
    }

    /// `layernorm_vec4.wgsl` and `fused_residual_ln_tee_vec4.wgsl` parse and
    /// validate under the WebGPU baseline, and each declares the lane count its
    /// dispatch is built from. Device-free (naga only), on the same reasoning as
    /// the snake's pin: a WGSL mistake would otherwise surface as a panic at
    /// `create_shader_module` on the first machine whose graph reaches the vector
    /// route, which on this rig is the one adapter that always takes it.
    #[test]
    fn vec4_norm_shaders_match_their_dispatch_geometry() {
        for (name, src) in [
            ("layernorm_vec4.wgsl", LAYERNORM_VEC4_WGSL),
            (
                "fused_residual_ln_tee_vec4.wgsl",
                FUSED_RESIDUAL_LN_TEE_VEC4_WGSL,
            ),
        ] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
                panic!("{name}: {e:?}");
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate under the WebGPU baseline: {e:?}"));
            assert!(
                src.contains(&format!("@workgroup_size({NORM_VEC4_LANES})")),
                "{name} must declare `@workgroup_size({NORM_VEC4_LANES})`: the run \
                 dispatches `NORM_VEC4_LANES` lanes per row"
            );
            assert!(
                src.contains(&format!("const WG: u32 = {NORM_VEC4_LANES}u;")),
                "{name} must stride its invocations by `NORM_VEC4_LANES` too: the two \
                 numbers live in one file and a drift between them would leave a row's \
                 tail unread"
            );
        }
    }

    /// `matmul_coop_f16_f32acc.wgsl` parses and validates, and declares the entry
    /// point and workgroup the lowering's arm sets. Device-free (naga only): WGSL
    /// is not validated at build time, and this kernel is reached only where an
    /// adapter exposes cooperative matrices — so without this a WGSL mistake would
    /// surface as a panic at `create_shader_module` on the machines that take the
    /// route and stay invisible on the others. The capability set is exactly what
    /// the kernel needs: an arm reaching for a third optional capability cannot
    /// compile here and fail only on a device that provides less.
    #[test]
    fn coop_f16_f32acc_shader_parses_and_validates() {
        let src = super::MATMUL_COOP_F16_F32ACC_WGSL;
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("matmul_coop_f16_f32acc.wgsl: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::COOPERATIVE_MATRIX,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("must validate with f16 and cooperative matrices: {e:?}"));
        assert_eq!(
            module
                .entry_points
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            vec!["matmul_coop_f16_f32acc"],
            "one compute entry, and it the one the lowering sets as the pipeline"
        );
        assert!(
            src.contains("@workgroup_size(64)"),
            "the tile is assembled by two 32-wide subgroups from one 64-thread \
             workgroup: `coop_f16_f32acc_admits`' 64x64x32 grid and the probe's \
             single 64-thread launch both assume it"
        );
    }

    /// The band module [`attention_band_kernel`] assembles parses and validates at
    /// both staging widths, and its `@workgroup_size` is the `THREADS` its grid is
    /// built from. Device-free (naga only) on the same reasoning as the kernels
    /// above, and because this shader is not a file a reader can check on its own:
    /// it is valid only after the tile constants, the staged-tile type and the
    /// expanded loops are substituted in. Driven through
    /// [`super::attention_band_module`] at the storage budgets adapters report —
    /// WebGPU's guaranteed 16 KiB and this rig's 32 KiB — so the key block under
    /// test is the block those devices get, not one the test chose.
    #[test]
    fn band_shader_parses_and_validates_at_both_stagings() {
        for (head_dim, threads) in [(256u32, super::ATTENTION_BAND_MIN_THREADS), (512, 512)] {
            for budget in [16 * 1024, 32 * 1024] {
                for f16_stage in [true, false] {
                    let src = super::attention_band_module(head_dim, threads, budget, f16_stage);
                    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                        panic!(
                            "attention_band d={head_dim} threads={threads} budget={budget} \
                             f16={f16_stage}: {e:?}"
                        )
                    });
                    let caps = if f16_stage {
                        naga::valid::Capabilities::SHADER_FLOAT16
                    } else {
                        naga::valid::Capabilities::empty()
                    };
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
                        .validate(&module)
                        .unwrap_or_else(|e| {
                            panic!(
                                "attention_band d={head_dim} threads={threads} budget={budget} \
                                 f16={f16_stage} must validate: {e:?}"
                            )
                        });
                    assert!(
                        src.contains("@workgroup_size(THREADS)")
                            && src.contains(&format!("const THREADS: u32 = {threads}u;")),
                        "d={head_dim}: the module's workgroup must be the count the grid is \
                         built from"
                    );
                }
            }
        }
    }

    /// `conv_transpose2d_tiled.wgsl` parses and validates under the WebGPU
    /// baseline, and the tile its dispatch grid is built from is the tile it
    /// declares. Device-free (naga only): without this a WGSL mistake would
    /// surface as a panic at `create_shader_module` on the first graph that takes
    /// the route — and, since the route is `lower.rs`'s family predicate rather
    /// than an option, on every machine at once rather than on whichever one
    /// happened to run it. `Capabilities::empty()` is the strictest set, so an arm
    /// reaching for an optional capability cannot compile here and fail only on a
    /// device that provides less.
    #[test]
    fn conv_transpose2d_tiled_shader_matches_its_dispatch_geometry() {
        let module = naga::front::wgsl::parse_str(CONV_TRANSPOSE2D_TILED_WGSL).expect("wgsl parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("wgsl validate without any optional capability");

        assert_eq!(
            module
                .entry_points
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            vec!["conv_transpose2d_tiled"],
            "one compute entry, and it the one `run.rs` sets as the pipeline"
        );

        for (name, value) in [("TILE_U", TILED_TILE_U), ("TILE_CO", TILED_TILE_CO)] {
            let decl = format!("const {name}: u32 = {value}u;");
            assert!(
                CONV_TRANSPOSE2D_TILED_WGSL.contains(&decl),
                "the shader must declare `{decl}`: the dispatch grid is built from the Rust \
                 constant, and that grid is the only bound on which output elements a dispatch \
                 covers"
            );
        }

        // The grid's rounding, at a width that is not a multiple of the tile: a
        // floor division here would leave a phase's last positions unserved, which
        // no shape in the pinned family would show.
        assert_eq!(
            conv_transpose2d_tiled_grid(3 * TILED_TILE_U + 1, 1, 2 * TILED_TILE_CO + 1),
            [4, 3, 1],
        );
    }

    /// `matmul_wide_vec4.wgsl` holds both wide-path geometries: the full 64-row
    /// tile `run` sets as the pipeline, and the half-height remainder entry
    /// `wide_grid` records for the rows the full-tile grid rounds up. Same
    /// device-free argument as the conv pin above, with one extra stake — the two
    /// entry points are one body at two geometries, so a row count they disagree
    /// with `wide_grid` about is a coverage hole (the remainder tile reaching too
    /// few rows) or a padded one (reaching rows the full grid already stored).
    #[test]
    fn wide_matmul_shader_matches_its_dispatch_geometry() {
        let module = naga::front::wgsl::parse_str(MATMUL_WIDE_VEC4_WGSL).expect("wgsl parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("wgsl validate without any optional capability");

        // Both entry points, each a call to the SAME function with no loop of its
        // own — name, workgroup size, one shared callee. A second copy of the hot
        // loop would be free to drift from the first, and the drift would show up
        // as a wrong number in one of the two, not as a compile error.
        let mut entries = Vec::new();
        for e in &module.entry_points {
            let stmts = &*e.function.body;
            let calls = stmts
                .iter()
                .filter_map(|s| match s {
                    naga::Statement::Call { function, .. } => Some(*function),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                calls.len(),
                1,
                "`{}` must call exactly one function, its tile: {:?}",
                e.name,
                e.function.body
            );
            assert!(
                !stmts
                    .iter()
                    .any(|s| matches!(s, naga::Statement::Loop { .. })),
                "`{}` holds a loop of its own: the two geometries must be one body \
                 at two tiles, not a copy of the body each: {:?}",
                e.name,
                e.function.body
            );
            entries.push((e.name.as_str(), e.workgroup_size, calls[0]));
        }
        assert_eq!(
            entries
                .iter()
                .map(|(name, size, _)| (*name, *size))
                .collect::<Vec<_>>(),
            vec![
                ("matmul_wide_vec4", [16, 16, 1]),
                ("matmul_wide_vec4_tail", [16, 8, 1]),
            ],
            "the entry point `run` sets as the pipeline, then the one it records \
             for the remainder; either renamed or re-geometried is a dispatch the \
             wide path cannot make"
        );
        assert_eq!(
            entries[0].2, entries[1].2,
            "the two geometries must call one shared tile function"
        );
        assert_eq!(
            module.functions[entries[0].2].name.as_deref(),
            Some("wide_vec4_tile"),
        );

        for (name, value) in [
            ("FULL_M", WideVariant::Vec4.tile_m()),
            ("TAIL_M", WIDE_TAIL_M),
        ] {
            let decl = format!("const {name}: u32 = {value}u;");
            assert!(
                MATMUL_WIDE_VEC4_WGSL.contains(&decl),
                "the shader must declare `{decl}`: the geometry of each entry \
                 point is a Rust constant the dispatch and `wide_grid` are both \
                 built from, and WGSL cannot read it from here"
            );
        }

        // The row counts the rule was written for, per variant: the two pins keep
        // the single rounded-up grid they shipped (no remainder pipeline, so no
        // split even where a remainder would fit), and the default variant splits
        // 96 = 64 + 32 — the pin that the split is actually taken, which the
        // bounds in the sweep below cannot tell from a rule that never splits.
        assert_eq!(
            wide_grid(WideVariant::Std, 100, 64),
            WideGrid {
                gx: 1,
                full: 4,
                tail: 0,
            },
        );
        assert_eq!(
            wide_grid(WideVariant::Nv, 100, 64),
            WideGrid {
                gx: 1,
                full: 2,
                tail: 0,
            },
        );
        assert_eq!(
            wide_grid(WideVariant::Vec4, 96, 48000),
            WideGrid {
                gx: 750,
                full: 1,
                tail: 1,
            },
        );
    }

    /// `wide_grid` over every row count a matmul dispatch can see, for each
    /// variant. The failure this guards is a silently MISSING row — a grid that
    /// covers less than the caller asked for — which reads back as whatever the
    /// arena slot held before and cannot be attributed to this rule by any
    /// numeric test.
    #[test]
    fn wide_grid_covers_every_row_and_pads_no_more_than_the_single_grid() {
        for variant in [WideVariant::Std, WideVariant::Nv, WideVariant::Vec4] {
            let tile_m = variant.tile_m();
            for n in [64, 130, 48000] {
                for m_s in 1..=600 {
                    let g = wide_grid(variant, m_s, n);
                    let rows = g.full * tile_m + g.tail * WIDE_TAIL_M;
                    assert_eq!(g.gx, n.div_ceil(64), "column tiles at n={n}");
                    assert!(
                        rows >= m_s,
                        "{variant:?} m={m_s} n={n}: the grid covers {rows} rows, \
                         so {} row(s) would keep their previous value",
                        m_s - rows.min(m_s)
                    );
                    assert!(
                        rows <= m_s.div_ceil(tile_m) * tile_m,
                        "{variant:?} m={m_s} n={n}: the split must never pad more \
                         than the single rounded-up grid it replaces"
                    );
                    assert!(g.full >= 1, "{variant:?} m={m_s}: no full tile");
                    assert!(g.tail <= 1, "{variant:?} m={m_s}: at most one tail");
                }
            }
        }
    }

    /// The activation codes `apply_activation` actually has an arm for, read out of
    /// the kernel's own source. The scan is scoped to that one function: the same file's
    /// `apply_binary` / `apply_cast` / `apply_compare` arms share the `sub == Nu` shape,
    /// and a file-wide scan would conflate their code spaces.
    fn kernel_activation_subs(src: &str) -> Vec<u32> {
        let body = src
            .split_once("fn apply_activation(")
            .expect("apply_activation in the region kernel")
            .1;
        let body = body.split("\nfn ").next().unwrap();
        let mut subs: Vec<u32> = body
            .split("sub == ")
            .skip(1)
            .filter_map(|rest| rest.split('u').next()?.trim().parse::<u32>().ok())
            .collect();
        subs.sort_unstable();
        subs.dedup();
        subs
    }

    /// Claimed implies implemented, and the exclusion covers every code the kernel
    /// lacks: the declared set and the kernel's arms must agree exactly. Reading the
    /// kernel source is what makes this a pin rather than an assertion that a list
    /// exists — a code added to the declared set without an arm in the kernel fails
    /// here, which is exactly the silent-identity defect this set exists to prevent.
    #[test]
    fn declared_region_activation_codes_match_the_kernel_arms() {
        let mut declared = rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS.to_vec();
        declared.sort_unstable();
        assert_eq!(
            kernel_activation_subs(ELEMENTWISE_REGION_WGSL),
            declared,
            "the declared region-kernel activation set and `apply_activation`'s arms \
             disagree: a declared code without an arm is computed as its own input \
             silently, and an arm left out of the set keeps a region the kernel could \
             have run off it"
        );
    }

    /// The decode entry's register ceiling is written twice — the shader's
    /// `DECODE_MAX_DIM` and this module's [`ATTENTION_DECODE_MAX_HEAD_DIM`] — and
    /// nothing in the route ties them. If the shader's constant is the smaller, a
    /// width the route admits reaches the entry and its early return writes no
    /// output at all, silently; if it is the larger, the per-thread arrays grow
    /// past what the Metal compiler keeps in registers. Read the source and pin
    /// the two together.
    #[test]
    fn the_decode_register_ceiling_is_the_shaders_own() {
        assert!(
            ATTENTION_WGSL.contains(&format!(
                "const DECODE_MAX_DIM: u32 = {ATTENTION_DECODE_MAX_HEAD_DIM}u;"
            )),
            "the route's decode head ceiling and the shader's `DECODE_MAX_DIM` are one \
             number: a width past the shader's leaves the entry's early return with no \
             output, so the step's slot is written by nobody"
        );
    }

    /// The key-range split's partials are sized on the host
    /// ([`super::ATTENTION_PARTS_DIM_CHUNK`]) and addressed in the shader
    /// (`DECODE_DIM_CHUNK`). A pad the host is smaller than the shader's makes
    /// every split step write past its buffer; a pad the host is larger than the
    /// shader's leaves the merge reading a layout the decode entry did not fill.
    #[test]
    fn the_split_partials_pad_is_the_shaders_own() {
        assert!(
            ATTENTION_WGSL.contains(&format!(
                "const DECODE_DIM_CHUNK: u32 = {}u;",
                super::ATTENTION_PARTS_DIM_CHUNK
            )),
            "the split's partial stride is `2 + pad` on both sides, so the host's pad \
             and the shader's `DECODE_DIM_CHUNK` are one number"
        );
    }
}
