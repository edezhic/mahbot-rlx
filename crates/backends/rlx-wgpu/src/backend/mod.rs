// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WgpuExecutable` — compiles an rlx-ir Graph into a sequence of
//! kernel dispatches against a pre-allocated arena buffer.
//!
//! v2 op coverage: MatMul + element-wise families (Binary 7, Unary 12,
//! Compare 6, Where) + leaves. Anything else panics at compile time.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use rlx_ir::dynamic::{bind_graph, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::Activation;
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};

use crate::buffer::{Arena, ReadbackStaging, TinyReadbackStaging};
use crate::device::wgpu_device;
use crate::kernels::{GatherParams, Kernel, gather_split_kernel, matmul_coop_f16_vulkan_kernel};
/// Compute the maximum tail-scratch bytes any single op needs across
/// the graph: `Op::LayerNormBackwardGamma`'s partial sums, norm staging,
/// and `Op::Sample`'s per-row working area (it reads its logits and must
/// not overwrite them).
fn compute_scratch_bytes(graph: &rlx_ir::Graph) -> usize {
    const ROWS_PER_WG: u32 = 16;
    let mut max_bytes = 0usize;
    for node in graph.nodes() {
        // Norm staging: when params live far from activations in the arena,
        // wgpu's `max_storage_buffer_binding_size` can prevent binding a
        // single window that covers both. We reserve a small scratch tail
        // zone so the schedule can move gamma/beta next to activations via
        // a copy step and keep shader bindings local.
        if matches!(
            &node.op,
            rlx_ir::Op::LayerNorm { .. } | rlx_ir::Op::RmsNorm { .. }
        ) {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let h_dim = x_shape.dim(x_shape.rank() - 1);
            if h_dim.is_static() {
                let h = h_dim.unwrap_static();
                // gamma + beta, 256B-aligned for binding offsets.
                let bytes = ((h * 4).div_ceil(256) * 256) * 2;
                if bytes > max_bytes {
                    max_bytes = bytes;
                }
            }
        }
        if let rlx_ir::Op::LayerNormBackwardGamma { .. } = &node.op {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let Some(elems) = x_shape.num_elements() else {
                continue;
            };
            let h_dim = x_shape.dim(x_shape.rank() - 1);
            if !h_dim.is_static() {
                continue;
            }
            let h = h_dim.unwrap_static();
            if h == 0 {
                continue;
            }
            let rows = (elems / h) as u32;
            let num_workgroups = rows.div_ceil(ROWS_PER_WG.max(1));
            let bytes = (num_workgroups as usize) * h * 4;
            if bytes > max_bytes {
                max_bytes = bytes;
            }
        }
        // The sampling kernel needs a per-row working area and must not
        // write its logits, so its intermediates live in the scratch zone.
        if let rlx_ir::Op::Sample { .. } = &node.op {
            if let Some(elems) = graph.node(node.inputs[0]).shape.num_elements()
                && elems != 0
            {
                let bytes = elems * 4;
                if bytes > max_bytes {
                    max_bytes = bytes;
                }
            }
        }
        // The many-row attention route's Kᵀ, score matrix and operand
        // compaction: the route's own plan of this node. Every other node — and
        // every attention shape the route does not serve — answers zero
        // (`attention_route`). An admitted plan fits the reserve without
        // depending on anything but this branch.
        let attn = match attention_route::plan_attention(graph, node) {
            Ok(plan) if plan.decompose || plan.dense_coop => plan.scratch_bytes(),
            _ => 0,
        };
        if attn > max_bytes {
            max_bytes = attn;
        }
    }
    // Reserve extra scratch for staging small far-apart operands when the
    // arena exceeds wgpu's binding window. This keeps compile-time simple
    // and avoids per-op scratch sizing plumbing.
    //
    // The floor is the minimum every graph is guaranteed, not the route's cap:
    // an admitted plan raises `max_bytes` to its own `scratch_bytes()` above.
    // Raising the floor instead would make every graph reserve (and, under a
    // small shard cap, stripe) the cap-sized tail it never writes, and
    // `sharded_arena`'s forced-small-shard case then refuses on the stripe bound.
    // The measurements behind the cap live on
    // `attention_route::ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES`.
    max_bytes.max(64 * 1024 * 1024)
}

/// Shared ephemeral state for `Op::GatedDeltaNet` with `carry_state=false`
/// (one slab reused across layers — same sizing as Metal/CUDA).
fn gdn_ephemeral_state_bytes(graph: &rlx_ir::Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::GatedDeltaNet {
            carry_state,
            state_size,
            ..
        } = &node.op
            && !*carry_state
        {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let elems = q_shape.dim(0).unwrap_static()
                * q_shape.dim(2).unwrap_static()
                * state_size
                * state_size;
            max = max.max(elems * std::mem::size_of::<f32>());
        }
    }
    max
}

/// Scratch bytes for native multi-layer GRU / Elman RNN / LSTM: a ping-pong pair
/// of `[batch, seq, dirs·hidden]` f32 buffers holding intermediate layer outputs
/// (the LSTM cell state stays in registers, so no extra scratch). Only
/// `num_layers > 1` needs it (single-layer writes straight to its output slot;
/// both directions own disjoint output slices). Shared/reused across ops
/// (sequential execution) → the max over nodes.
fn rnn_gru_scratch_bytes(graph: &rlx_ir::Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        let (hidden, num_layers, bidirectional) = match &node.op {
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            }
            | Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            }
            | Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            } => (*hidden_size, *num_layers, *bidirectional),
            _ => continue,
        };
        if num_layers <= 1 {
            continue;
        }
        let x_shape = &graph.node(node.inputs[0]).shape;
        let batch = x_shape.dim(0).unwrap_static();
        let seq = x_shape.dim(1).unwrap_static();
        let dirs = if bidirectional { 2 } else { 1 };
        let layer_elems = batch * seq * dirs * hidden;
        max = max.max(2 * layer_elems * std::mem::size_of::<f32>());
    }
    max
}

/// Hard cap on the col scratch matrix (bytes). Convs whose col would exceed
/// this fall back to the direct conv kernel (keeps the arena well under the
/// 4 GiB storage-binding limit). The host transposed-conv route reuses the same
/// cap for its own `col` panel.
pub(crate) const CONV_IM2COL_MAX_COL_BYTES: u64 = 1024 * 1024 * 1024;

/// Largest spatial extent the 1×1 pointwise GEMM may take. The route's `n` is
/// the spatial extent and the matmul dispatch emits one workgroup per 32 (or 64)
/// output columns, so [`MAX_WORKGROUPS_PER_DIM`] is reached at
/// `MAX_WORKGROUPS_PER_DIM * 32`. This cap is what keeps the route's own grid
/// under the ceiling — the f32 cooperative kernel included, its x grid being the
/// same `n / 32`; [`conv_gemm`]'s ceiling check is the general guard over both
/// routes and cannot fire under this cap. Past it the shape keeps the previous
/// direct/tiled route.
pub(crate) const POINTWISE_MAX_SPATIAL: u64 = MAX_WORKGROUPS_PER_DIM as u64 * 32;

/// Admission bar for the 1×1 pointwise GEMM, in `c_in × spatial` elements. The
/// im2col spatial and K floors used to be multiplied together for this test; the
/// number that product then stood for (`256 × 2048`) is kept here as a frozen
/// literal of its own, so the spatial floor can move under the non-pointwise shapes
/// without the 1×1 admission moving with it — nothing recomputes this bar from
/// either floor, and the 1×1 route keeps exactly the admission it had. The channel
/// floor ([`im2col_min_cout`]) is what both routes still share.
pub(crate) const POINTWISE_MIN_COL_ELEMS: u64 = 524_288;

/// Whether a 1×1 conv of this shape takes the pointwise GEMM route: its own bar
/// above, deliberately not a product of the im2col floors.
pub(crate) fn pointwise_admitted(k_total: u64, spatial: u64) -> bool {
    k_total.saturating_mul(spatial) >= POINTWISE_MIN_COL_ELEMS
}

// Defaults live in [`crate::config::WgpuRuntimeConfig::from_env`].
fn im2col_min_spatial() -> u64 {
    crate::runtime_config().im2col_min_spatial as u64
}
fn im2col_min_k() -> u64 {
    crate::runtime_config().im2col_min_k as u64
}
fn im2col_min_cout() -> u64 {
    crate::runtime_config().im2col_min_cout as u64
}
/// The single switch the accelerated conv routes sit under: im2col+GEMM, the
/// mirrored 1-D tiled lift, the 1×1 pointwise GEMM and the cooperative implicit
/// GEMM. Default **on**; `RLX_WGPU_CONV_IM2COL=0` restores the plain routing and
/// the lean arena exactly, which is what keeps a bisect of that flip possible. The
/// same predicate gates the col-scratch reservation, which is sized from a conv's
/// *shape* and the implicit route's own two terms (its switch and the adapter's
/// entry) without reading the graph's schedule (see [`conv_im2col_col_elems`]): a
/// graph whose convs qualify for neither reserves nothing, and one whose conv does
/// pays that reservation even where the route then does not run — a parked weight,
/// a conv its adapter routes to the host, a sharded arena.
pub(crate) fn conv_routes_enabled() -> bool {
    rlx_ir::env::flag_or("RLX_WGPU_CONV_IM2COL", true)
}

/// The cooperative implicit-GEMM convolution (`conv1d_implicit.wgsl`), live:
/// `RLX_WGPU_CONV_IMPLICIT=0` puts every conv back on the im2col/tiled/direct
/// routing for a bisect. Read at the point of decision rather than frozen in
/// `WgpuRuntimeConfig`, like the other route switches.
pub(crate) fn conv_implicit_enabled() -> bool {
    rlx_ir::env::flag_or("RLX_WGPU_CONV_IMPLICIT", true)
}

/// The shortest output extent the implicit route is admitted at.
///
/// The gather is one strided read per k step and column block, so the route's cost
/// per output element barely moves with the extent; the floor separates a conv with
/// enough columns for that gather to dominate its own dispatch from one small enough
/// that the dispatch overhead does. It is the measured crossover between the two
/// routes, and the recognition front's convolutions, a few hundred positions, sit
/// under it. Overridable via `RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL` (the parity cell
/// drives the kernel at its own small extents with it).
///
/// The floor is a fallback-cost switch as well as a win threshold: the panel
/// reservation subtracts this class without reading the schedule, so a class-shaped
/// conv the lowering then declines — a parked weight, an arena that does not bind in
/// one storage binding — has no panel to fall back on and runs the direct kernel.
const CONV_IMPLICIT_MIN_SPATIAL: u64 = 4_800;
pub(crate) fn conv_implicit_min_spatial() -> u64 {
    rlx_ir::env::parse_or(
        "RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL",
        CONV_IMPLICIT_MIN_SPATIAL,
    )
}

/// The most output rows the implicit entry's accumulator sets cover — its
/// `ROW_TILES` sets of `TILE_M` rows, pinned against the shader source by
/// `the_implicit_class_cap_is_the_entrys_row_tiles`. A wider `c_out` would have
/// rows no set covers, so the class refuses it on the host.
pub(crate) const CONV_IMPLICIT_MAX_C_OUT: u64 = 96;

/// The narrowest output the implicit entry takes. The entry multiplies a whole
/// 32-row tile whatever `c_out` is, so below a tile's rows it multiplies output the
/// conv does not have and the panel route — which reads back only the rows it wrote
/// — overtakes it. It is the measured crossover between the two routes at the
/// block's own extent and taps; `conv_implicit_vs_direct_by_output_channels` is the
/// cell that takes it. A narrower `c_out` keeps the direct routing.
pub(crate) const CONV_IMPLICIT_MIN_C_OUT: u64 = 4;

/// The shape class the cooperative implicit-GEMM conv is admitted for, in the
/// terms the `Op::Conv` compile arm's geometry states it: a unit height with every
/// tap on W and no height padding, one group and one batch, at most 96 input
/// channels and between [`CONV_IMPLICIT_MIN_C_OUT`] and 96 of output, at or above
/// the route's own extent floor, and column blocks inside the per-dimension
/// dispatch cap. The cap on `c_out` is the kernel's own: one workgroup covers every
/// output row tile, and the accumulator sets it has are three — 96 rows is the
/// widest output they cover.
///
/// The lowering adds the terms the shape does not settle — the route's switches,
/// the adapter, the weight's placement, the arena's binding — and
/// [`conv_im2col_col_elems`] subtracts this class, so a conv the route takes
/// reserves no panel and a conv it does not take still gets one.
pub(crate) struct ConvImplicitShape {
    pub c_in: u64,
    pub c_out: u64,
    pub kh: u64,
    pub kw: u64,
    pub h: u64,
    pub h_out: u64,
    pub sw: u64,
    pub ph: u64,
    pub groups: u64,
    pub n: u64,
    pub spatial: u64,
}

impl ConvImplicitShape {
    fn admits(&self) -> bool {
        self.kh == 1
            && self.kw > 1
            && self.sw == 1
            && self.h == 1
            && self.h_out == 1
            && self.ph == 0
            && self.groups == 1
            && self.n == 1
            && self.c_in <= CONV_IMPLICIT_MAX_C_OUT
            && self.c_out <= CONV_IMPLICIT_MAX_C_OUT
            && self.c_out >= CONV_IMPLICIT_MIN_C_OUT
            && self.spatial >= conv_implicit_min_spatial()
            && self.spatial.div_ceil(32) <= MAX_WORKGROUPS_PER_DIM as u64
    }
}

/// Minimum output length for routing a `one_d` conv through the register-blocked
/// `conv1d_tiled` kernel. Short convs stay on the direct kernel (the tile setup
/// isn't worth it). Overridable via `RLX_WGPU_TILED_MIN_SPATIAL`.
const CONV_TILED_MIN_SPATIAL: u64 = 256;
fn conv_tiled_min_spatial() -> u64 {
    rlx_ir::env::parse_or("RLX_WGPU_TILED_MIN_SPATIAL", CONV_TILED_MIN_SPATIAL)
}

/// Element count of the im2col `col[c_in*kh*kw, spatial]` matrix if this conv
/// node qualifies for the im2col+GEMM path (2D / 1D-as-2D NCHW, `N==1`,
/// `groups==1`, real spatial kernel, spatial ≥ the im2col spatial floor).
/// Returns `None` otherwise, and `None` for a conv the cooperative implicit route
/// takes instead (`implicit_live`, see [`ConvImplicitShape`]). Mirrors the shape
/// math in the `Op::Conv` compile arm so scratch sizing and lowering agree on
/// which convs are routed.
fn conv_im2col_col_elems(
    op: &Op,
    in_shape: &rlx_ir::Shape,
    w_shape: &rlx_ir::Shape,
    out_shape: &rlx_ir::Shape,
    implicit_live: bool,
) -> Option<u64> {
    let Op::Conv {
        kernel_size,
        stride,
        padding,
        groups,
        ..
    } = op
    else {
        return None;
    };
    if *groups != 1 {
        return None;
    }
    if !(kernel_size.len() == 2
        && in_shape.rank() == 4
        && w_shape.rank() == 4
        && out_shape.rank() == 4)
    {
        return None;
    }
    for d in [
        in_shape.dim(0),
        in_shape.dim(1),
        in_shape.dim(2),
        in_shape.dim(3),
    ] {
        if !d.is_static() {
            return None;
        }
    }
    if !out_shape.dim(2).is_static() || !out_shape.dim(3).is_static() {
        return None;
    }
    if !out_shape.dim(1).is_static() {
        return None;
    }
    if in_shape.dim(0).unwrap_static() != 1 {
        return None;
    }
    let c_in = in_shape.dim(1).unwrap_static() as u64;
    let c_out = out_shape.dim(1).unwrap_static() as u64;
    let h_in = in_shape.dim(2).unwrap_static() as u32;
    let w_in = in_shape.dim(3).unwrap_static() as u32;
    let one_d = h_in == 1
        && w_in > 1
        && kernel_size[0] > 1
        && kernel_size.get(1).copied().unwrap_or(1) == 1;
    let (kh, kw, spatial) = if one_d {
        (
            kernel_size[0] as u64,
            1u64,
            out_shape.dim(3).unwrap_static() as u64,
        )
    } else {
        (
            kernel_size[0] as u64,
            kernel_size.get(1).copied().unwrap_or(1) as u64,
            out_shape.dim(2).unwrap_static() as u64 * out_shape.dim(3).unwrap_static() as u64,
        )
    };
    let k_total = c_in * kh * kw;
    if kh * kw < 2 {
        return None;
    }
    if spatial < im2col_min_spatial() || k_total < im2col_min_k() || c_out < im2col_min_cout() {
        return None;
    }
    // The fields the class test reads, mapped exactly as the compile arm's
    // pre-relabel `p2` maps them: a `one_d` lift puts the length on H (so its
    // `h_out` is the output's `dim(3)`), which `kh == 1` rejects before any of the
    // rest is read — the mirror is stated rather than left to that short-circuit.
    let (h, h_out) = if one_d {
        (w_in as u64, out_shape.dim(3).unwrap_static() as u64)
    } else {
        (h_in as u64, out_shape.dim(2).unwrap_static() as u64)
    };
    let implicit_shape = ConvImplicitShape {
        c_in,
        c_out,
        kh,
        kw,
        h,
        h_out,
        // The compile arm's own accessors default a short stride/padding vector
        // rather than indexing it, so a 2-D op missing them cannot panic here.
        sw: stride.get(1).copied().unwrap_or(1) as u64,
        ph: padding.first().copied().unwrap_or(0) as u64,
        groups: *groups as u64,
        n: in_shape.dim(0).unwrap_static() as u64,
        spatial,
    };
    if implicit_live && implicit_shape.admits() {
        return None;
    }
    Some(k_total * spatial)
}

/// Extra arena scratch (bytes, 256-aligned) needed to hold the largest conv
/// col matrix that will be routed through im2col+GEMM. Capped so the whole
/// arena still binds in a single storage-binding window. Returns 0 when no
/// conv qualifies or none fits.
///
/// `implicit_live` is the cooperative implicit route's own switch and adapter
/// capability ([`conv_implicit_enabled`] and `kernels::conv1d_implicit_supported`):
/// a conv that class takes never writes a panel, so it reserves none. The two
/// terms the shapes still do not settle — a weight parked in the weight buffer and
/// an arena that with its panel does not bind — are not known here, and a conv
/// refused for one of them keeps the direct/tiled route instead of a panel-less
/// im2col (the count is `im2col_fallbacks`, and the lowering refuses the im2col
/// route for want of reserved scratch).
fn conv_im2col_scratch_bytes(
    graph: &Graph,
    planned_arena_size: usize,
    max_binding: u64,
    implicit_live: bool,
) -> usize {
    let mut max_col_bytes: u64 = 0;
    for node in graph.nodes() {
        if node.inputs.len() < 2 {
            continue;
        }
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let Some(elems) =
            conv_im2col_col_elems(&node.op, in_shape, w_shape, &node.shape, implicit_live)
        else {
            continue;
        };
        let col_bytes = elems.saturating_mul(4);
        if col_bytes > CONV_IM2COL_MAX_COL_BYTES {
            continue;
        }
        if (planned_arena_size as u64).saturating_add(col_bytes) > max_binding {
            continue;
        }
        max_col_bytes = max_col_bytes.max(col_bytes);
    }
    (max_col_bytes.div_ceil(256) * 256) as usize
}

/// Inner-FMA precision for matmul.
///   F32    — full f32 path (matmul.wgsl / matmul_wide.wgsl).
///   F16    — f16 multiply, f32 acc (matmul_f16_compute.wgsl).
///   Coop16 — cooperative-matrix 8×8 hardware GEMM
///            (matmul_coop16.wgsl, simdgroup_multiply_accumulate on
///             Apple, OpCooperativeMatrixMulAddKHR on Vulkan).
///            Requires M/N/K multiples of 8, b is a Param, and
///            both SHADER_F16 + EXPERIMENTAL_COOPERATIVE_MATRIX.
///            Caller must ensure A is mirrored to arena_f16 first
///            (the lowering inserts a `Step::CastF32ToF16` pre-pass).
pub struct WgpuExecutable {
    graph: Graph,
    arena: Arena,
    /// Byte offset of GGUF dequant scratch slab (0 when host fallback).
    dequant_scratch_off: usize,
    /// Byte offset of ephemeral GatedDeltaNet state (`carry_state=false`).
    gdn_scratch_off: usize,
    schedule: Vec<Step>,
    /// Which graph node produced each schedule step, parallel to `schedule`.
    ///
    /// A node with no entry here was never written by any step — it was elided,
    /// folded into a consumer, or is a pure view. That is invisible in a
    /// post-hoc arena read, where an unwritten slot and a slot a kernel wrongly
    /// filled with zeros look identical; conflating the two produced three
    /// separate wrong diagnoses of the `rlx-vieeg` divergence.
    step_nodes: Vec<NodeId>,
    /// Guard so the snapshot replay does not re-enter itself.
    in_snapshot: bool,
    /// Stop the schedule after this many steps (snapshot replay only).
    snapshot_stop: Option<usize>,
    input_offsets: HashMap<String, NodeId>,
    param_offsets: HashMap<String, NodeId>,
    /// One uniform buffer + bind group per dispatch step. Pre-allocated
    /// so run() just writes new bytes per step.
    uniforms: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    /// Per-step metadata storage buffers, for the steps that need per-axis
    /// geometry their uniform cannot carry: `Transpose`, `Expand`, `IndexingNd`
    /// and `Reverse`. Indexed by each step's `meta_idx`.
    meta_buffers: Vec<wgpu::Buffer>,

    // ── Lazy dynamic-shape state ─────────────────────────────────
    /// The originally-supplied graph (pre-resolution). Only set when
    /// the input graph contained `Dim::Dynamic` entries — otherwise
    /// `None` and the compiled fields above are authoritative. On each
    /// `run()` we infer a `DimBinding` from the live input data, and
    /// if it differs from `last_binding` we re-resolve + recompile.
    unresolved: Option<Graph>,
    last_binding: Option<DimBinding>,
    /// Buffered params written via `set_param` / `set_param_bytes`
    /// before the first `run()`. Replayed against the freshly compiled
    /// arena once shapes resolve, and kept as the record every later
    /// rebind replays from.
    pending_params: HashMap<String, Vec<f32>>,
    pending_param_bytes: HashMap<String, Vec<u8>>,
    /// A param was written since the pending set was last replayed into the
    /// arena this executable holds. `set_param` only buffers while a graph is
    /// unresolved, so a write at an unchanged binding would never reach the
    /// device without this mark.
    pending_params_dirty: bool,
    /// Active-extent hint (PLAN L1). When set + every Step in the
    /// safe set, both the uniform write and the dispatch workgroup
    /// count are scaled by `actual / upper`. Otherwise full-extent.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Skip-redundant-uniform-writes guard. Each `run()` would
    /// otherwise re-`queue.write_buffer` ~115 per-step uniforms (one
    /// per dispatched op in BERT) even when their bytes are identical
    /// to the previous call's. At small batches, that fixed write +
    /// staging-copy overhead is the dominant cost. We track the last
    /// active-extent value the uniforms were written for; subsequent
    /// `run()`s with the same `active_extent` (and `recompile`-clean
    /// schedule) skip the entire uniform-write loop. `None` ⇒ never
    /// written; `Some(x)` ⇒ uniforms hold params for active_extent=x.
    uniforms_active_extent: Option<Option<(usize, usize)>>,
    /// True when the schedule contains CoopF16Vk matmul.
    coop_f16_vk: bool,
    /// CoopF16Vk Param B offsets (f32 arena / 4) → param name for wide routing.
    coop_f16_b_param: HashMap<u32, String>,
    /// Param names flagged by the oscillation probe for wide f32 fallback.
    coop_f16_vk_wide_b: HashSet<String>,
    /// Wide f32 bind groups for CoopF16Vk steps (schedule index → bg).
    coop_f16_vk_wide_bind_groups: HashMap<usize, wgpu::BindGroup>,
    /// CoopF16Vk activation operands mirrored on the host each `run()` (f32+f16).
    coop_f16_host_activations: Vec<(NodeId, Activation, String)>,
    /// Last `set_param` f32 payload per name (for host activation mirrors).
    stashed_params: HashMap<String, Vec<f32>>,
    /// Reused output readback staging (avoids per-run buffer alloc).
    readback_staging: Option<ReadbackStaging>,
    /// Persistent tiny readback buffer for single scalar outputs.
    tiny_readback: Option<TinyReadbackStaging>,
    /// When set, `run_inner` dispatches + submits all compute but skips the
    /// blocking output readback (results stay in the arena). Read by the
    /// snapshot replay and by `RLX_BENCH_DISPATCH_ONLY`; a normal run leaves it
    /// false.
    dispatch_only: bool,
    /// Per-`FftGpu` step: isolated uniform buffers + bind groups (one vec entry per op).
    fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources>,
    /// Persistent KV inputs (host staging uploaded each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Arena input slots authoritative — skip host KV mirror each decode step.
    gpu_handle_resident: HashSet<String>,
    /// Resident-KV row feeds: handle name -> graph output index. See
    /// `register_kv_row_feed`.
    kv_row_feeds: HashMap<String, usize>,
    /// Bounce buffer for `feed_kv_row` — WebGPU refuses same-buffer copies, so
    /// the arena row goes arena -> here -> arena. Cached and grown on demand
    /// rather than allocated per feed (28 layers x every token otherwise).
    kv_feed_staging: Option<wgpu::Buffer>,
    pending_read_indices: Option<Vec<usize>>,
    /// Runtime-mutable RNG policy for [`Step::RngNormalHost`] / [`Step::RngUniformHost`].
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    /// Schedule indices that only pack Param/Constant subgraphs (F5 weight
    /// Concat, static Expand, …). Executed on the first `run()`, then skipped
    /// so the NFE loop stays device-resident like ORT I/O binding. Safe because
    /// the memory planner pins those packed tensors through graph end (see
    /// `extend_static_weight_pack_liveness`).
    static_once_steps: HashSet<usize>,
    /// The one-row packed multiply's per-dispatch setup, kept across frames: the
    /// same step rebuilds the same uniform and bind group every frame otherwise
    /// (~5.7 µs an op over ~516 ops a frame).
    gemv_setup: crate::gguf_gpu::GemvSetupCache,
    static_once_done: bool,
    /// Steps the per-binding-ceiling rule routed to the host route. Non-zero
    /// means the graph's arena exceeded the device's per-binding limit and at
    /// least one absolute-addressing op was computed on the host instead of
    /// through a re-based window. See `lower::should_host_at_ceiling`.
    ceiling_hosted: usize,
    /// Convs whose shape qualified for im2col but which took the direct/tiled route
    /// for a reason the shapes alone do not settle — counted where that route is
    /// decided, in the `Op::Conv` arm of `crate::backend::compile`, so a term added
    /// there cannot leave the count behind.
    im2col_fallbacks: usize,
    /// Convs whose shape is the cooperative implicit-GEMM route's class
    /// (`ConvImplicitShape`) but which kept the im2col/tiled/direct routing, for
    /// one of the five terms the shape does not settle: the umbrella switch
    /// `RLX_WGPU_CONV_IM2COL`, the route's own `RLX_WGPU_CONV_IMPLICIT=0`, an
    /// adapter without the entry, a weight parked in the weight buffer, an arena
    /// that does not bind whole. Counted where the route is decided, in the
    /// `Op::Conv` arm of `crate::backend::compile`, so no route that was not taken
    /// has to be inferred from a timing.
    conv1d_implicit_declines: usize,
    /// Many-row attention steps this compile sent to the fused kernel for want of
    /// a window over the route's regions — the one term of the route decision the
    /// graph's shapes do not settle (an arena addressed in stripes rather than
    /// whole, or one whose bind window does not reach the plan's regions). Reaching
    /// it takes an arena larger than what the adapter binds at once — 4 GiB on the
    /// adapter this crate is measured on, so nothing in its suite counts one, and a
    /// striped arena hosts a straddling attention step before the decision. Counted
    /// where the step is lowered, so a graph that is declined says so instead of
    /// leaving the route to be inferred from a timing: the schedule report prints
    /// the count, and the arm's attention debug print names which term it was.
    attention_window_declines: usize,
    /// Dense square attention steps this compile kept on the fused walk because
    /// the band kernel's admission refused their shapes (a head width with no
    /// whole-row form at either workgroup width, a KV-head count that does not
    /// divide the head count, or a dispatch grid over the per-dimension cap) where
    /// the cooperative form was not booked for them. Every term is the graph's
    /// own, so the count is a count of geometry rather than of an adapter; the
    /// schedule report prints it and the arm's attention debug print names the
    /// conjunct per step, so the dense-through-band fallback is visible instead of
    /// silent.
    attention_band_declines: usize,
    /// Dense square attention steps the cooperative form's admission took but this
    /// compile could not lower onto it: an adapter that does not build
    /// `matmul_coop_f16_f32acc`, a device probe that has not shown it computes
    /// `a·b`, a bind window that does not reach the plan's regions, an output slot
    /// on a dense Q's bytes (which the route's later chunks would read after the
    /// first chunk wrote them), or one of the two switches that decline the form
    /// (`RLX_WGPU_NO_DENSE_COOP`, `RLX_WGPU_MATMUL_F32_ONLY`). Those are device,
    /// memory, aliasing and switch terms rather than shapes, so the count says so;
    /// the schedule report prints it and the arm's attention debug print names the
    /// route.
    attention_dense_coop_declines: usize,
}

mod attention_route;
mod compile;
mod copy_route;
mod dispatch;
mod helpers;
mod host_plan;
mod run;
mod run_report;
mod set;
/// Static-string label for each Step variant — used by the Perfetto
/// trace layer (PLAN L3) to mark per-step events without allocating.
mod step;
mod test;

pub(crate) use copy_route::*;
/// The coop-GEMM admission and the absolute f32 pin, public because
/// `mahbot-inference` asks the backend which multiplies this kernel serves
/// instead of restating 64/64/32, and must agree with the pin when it reports a
/// route.
pub use helpers::coop_f16_f32acc_admits;
pub use helpers::f32_matmul_pinned;
pub(crate) use helpers::*;
pub(crate) use host_plan::plan as host_plan;
pub use host_plan::{HostBatch, HostGroup, HostIo, HostPlan, Region};
pub(crate) use run_report::*;
pub(crate) use step::*;

/// The many-row attention route's query-row floor and the decode form's
/// key-range split predicate: public because the crate's integration tests are
/// external and read them to place their shapes on either side.
pub use attention_route::{
    ATTENTION_DECOMPOSE_MIN_Q, DENSE_COOP_CHUNK_ROWS, DENSE_COOP_MIN_Q, attention_decode_key_splits,
};

/// The run report's marks handle, public only because the `gguf_gpu` entry
/// points that take it are: the crate's integration tests are external and
/// cannot name a crate-private type.
pub use run_report::PassMarks;

/// Process-wide count of packed-BF16 (`matmul_bf16w`) matmul dispatches.
/// Test/telemetry hook proving the packed path was actually taken.
pub(crate) static BF16_PACKED_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Increment the packed-BF16 dispatch counter (called from the run loop).
#[inline]
pub(crate) fn bf16_packed_dispatch_inc() {
    BF16_PACKED_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total packed-BF16 matmul dispatches since process start.
pub fn bf16_packed_dispatch_count() -> u64 {
    BF16_PACKED_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-wide count of dispatches of the ACTIVE f32 cooperative kernel —
/// `matmul_coop_f32` on Metal, `matmul_coop_f32_portable` on a Vulkan/DX12
/// adapter that selects it — across every call site: an ordinary multiply, and
/// the two convolution GEMM routes' weight-in-A multiplies. The same kind of
/// hook as [`bf16_packed_dispatch_count`]: the schedule's step names carry no
/// precision, so this is how "that node took the cooperative kernel" is observed
/// without reading the schedule.
pub(crate) static COOP_F32_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Increment the f32 cooperative dispatch counter (called from the run loop).
#[inline]
pub(crate) fn coop_f32_dispatch_inc() {
    COOP_F32_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total f32 cooperative-matrix matmul dispatches since process start.
pub fn coop_f32_dispatch_count() -> u64 {
    COOP_F32_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The subset of [`coop_f32_dispatch_count`] that took the epilogue-free direct
/// entry (`matmul_coop_f32_direct`): the form the convolution GEMM routes use.
pub(crate) static COOP_F32_DIRECT_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn coop_f32_direct_dispatch_inc() {
    COOP_F32_DIRECT_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total dispatches of the direct f32 cooperative entry since process start.
pub fn coop_f32_direct_dispatch_count() -> u64 {
    COOP_F32_DIRECT_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-wide count of `matmul_coop_f16_f32acc` dispatches. Same hook as
/// [`coop_f32_dispatch_count`], for the same reason: the schedule's step names
/// carry no precision, so a parity test that wants to show the shape it measured
/// took this kernel — rather than the f16-accumulator one or the f32 arena form
/// it outranks — counts dispatches under a forced arm.
pub(crate) static COOP_F16_F32ACC_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn coop_f16_f32acc_dispatch_inc() {
    COOP_F16_F32ACC_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total `matmul_coop_f16_f32acc` matmul dispatches since process start.
pub fn coop_f16_f32acc_dispatch_count() -> u64 {
    COOP_F16_F32ACC_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-wide count of `queue.submit` calls `src/` makes. Together with
/// [`DEVICE_WAITS`] it is the test/telemetry hook for whether a forward pass pays
/// its own submissions: a call whose work is encoded into the enclosing batch
/// adds none.
///
/// Every `queue.submit` and every `device.poll` in `src/` is annotated with
/// [`note_device_submit`] / [`note_device_wait`] — including the shared readback
/// helpers a host route stages through, which no call-site-only counter inside the
/// run loop would see. `tests/` and `examples/` drive their own brackets and are
/// not annotated on purpose. A new site in `src/` must be annotated too, or a pin
/// that brackets a forward pass reads a count that silently missed it — that
/// completeness is a handoff statement, not a gate.
pub(crate) static DEVICE_SUBMITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Process-wide count of places this crate blocks on device progress: every
/// `Device::poll` (indefinite or submission-indexed — the latter is what a
/// mapped readback waits on).
pub(crate) static DEVICE_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one `queue.submit`.
#[inline]
pub(crate) fn note_device_submit() {
    DEVICE_SUBMITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count one blocking `Device::poll`.
#[inline]
pub(crate) fn note_device_wait() {
    DEVICE_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Queue submissions this crate has made since process start (or since
/// [`reset_device_counts`]).
pub fn device_submit_count() -> u64 {
    DEVICE_SUBMITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Places this crate has blocked on device progress since process start (or
/// since [`reset_device_counts`]).
pub fn device_wait_count() -> u64 {
    DEVICE_WAITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Zero both device counters — for a test bracketing one forward pass.
pub fn reset_device_counts() {
    DEVICE_SUBMITS.store(0, std::sync::atomic::Ordering::Relaxed);
    DEVICE_WAITS.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide count of movement dispatches that took a `vec4` kernel.
///
/// The scalar arm produces the same bytes, so an output-only parity test cannot
/// tell which one ran; this counter is what ties such a test to the vector path
/// it claims to cover.
pub(crate) static VEC4_MOVES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one `vec4` movement dispatch.
#[inline]
pub(crate) fn note_vec4_move() {
    VEC4_MOVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// `vec4` movement dispatches since process start ([`reset_vec4_move_count`]).
pub fn vec4_move_count() -> u64 {
    VEC4_MOVES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Zero [`vec4_move_count`] — for a test bracketing one forward pass.
pub fn reset_vec4_move_count() {
    VEC4_MOVES.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide count of norm dispatches that took a `*_vec4` module.
///
/// The scalar and vector modules compute the same norm, so only a counter can
/// tell which one ran: without it a silent un-routing of `LayerNorm` /
/// `FusedResidualLnTee` back to the 64-lane module would leave every parity test
/// green.
pub(crate) static NORM_VEC4: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one vectorised norm dispatch.
#[inline]
pub(crate) fn note_norm_vec4() {
    NORM_VEC4.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Vectorised norm dispatches since process start ([`reset_norm_vec4_count`]).
pub fn norm_vec4_count() -> u64 {
    NORM_VEC4.load(std::sync::atomic::Ordering::Relaxed)
}

/// Zero [`norm_vec4_count`] — for a test bracketing one forward pass.
pub fn reset_norm_vec4_count() {
    NORM_VEC4.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Where a named handle's row is, as `WgpuExecutable::locate_handle_row` answers
/// it: the host mirror that still carries the rows, or the byte range of the
/// arena slot they live in. One ladder for both readback paths, so a row resolved
/// on its own and one resolved inside a step's batch cannot answer differently.
enum LocatedRow<'a> {
    /// The rows this executable holds are the host mirror's and no arena slot
    /// serves the name: the answer itself, and no device work.
    Mirror(&'a [f32]),
    /// `len` bytes of the arena at `byte_off` — weight-tagged if the slot is in
    /// the parked weight buffer.
    Arena { byte_off: usize, len: usize },
}

impl WgpuExecutable {
    /// Resolve the deferred graph against bindings inferred from
    /// `inputs`, recompile the inner state if the bindings changed
    /// since the last call, and replay any pending params.
    pub(crate) fn lazy_compile_for_inputs(&mut self, inputs: &[(&str, &[f32])]) {
        let unresolved = self
            .unresolved
            .as_ref()
            .expect("lazy_compile_for_inputs called without an unresolved graph");
        let binding = infer_bindings_from_f32_inputs(unresolved, inputs)
            .expect("rlx-wgpu lazy compile: could not infer DimBinding from inputs");

        // No-op if shapes haven't changed since the last compile — apart from a
        // param written since the last replay, which is buffered rather than
        // written and would otherwise never reach this arena.
        if let Some(prev) = &self.last_binding
            && same_binding(prev, &binding)
        {
            if self.pending_params_dirty {
                self.replay_pending_params();
            }
            return;
        }

        // Resolve and recompile.
        let resolved = bind_graph(unresolved, &binding);
        let original = self.unresolved.take();

        let fresh = Self::compile_static_inner(resolved, self.rng.clone());

        // Move the freshly-compiled fields into self, preserve the
        // unresolved+binding state for the next round.
        self.graph = fresh.graph;
        self.arena = fresh.arena;
        self.dequant_scratch_off = fresh.dequant_scratch_off;
        self.gdn_scratch_off = fresh.gdn_scratch_off;
        self.schedule = fresh.schedule;
        self.ceiling_hosted = fresh.ceiling_hosted;
        self.im2col_fallbacks = fresh.im2col_fallbacks;
        self.conv1d_implicit_declines = fresh.conv1d_implicit_declines;
        self.attention_window_declines = fresh.attention_window_declines;
        self.attention_band_declines = fresh.attention_band_declines;
        self.attention_dense_coop_declines = fresh.attention_dense_coop_declines;
        self.input_offsets = fresh.input_offsets;
        // The arena is this compile's, and a resident handle's host mirror was
        // emptied when its rows became resident — so this executable no longer
        // holds those rows, and must not answer as if it did. The marks go with
        // the arena, and so do the emptied mirrors: what is left is the mirrors
        // that still carry rows.
        self.gpu_handle_resident.clear();
        self.gpu_handles.retain(|_, rows| !rows.is_empty());
        self.param_offsets = fresh.param_offsets;
        self.uniforms = fresh.uniforms;
        self.bind_groups = fresh.bind_groups;
        self.meta_buffers = fresh.meta_buffers;
        self.last_binding = Some(binding);
        // Recompiled — uniforms are now empty buffers; force re-write
        // on next run().
        self.uniforms_active_extent = None;
        self.coop_f16_vk = fresh.coop_f16_vk;
        self.coop_f16_b_param = fresh.coop_f16_b_param;
        self.coop_f16_vk_wide_bind_groups = fresh.coop_f16_vk_wide_bind_groups;
        self.coop_f16_host_activations = fresh.coop_f16_host_activations;
        // The one-row packed multiply's setup cache holds bind groups over this
        // executable's previous arena, weight buffer and slab; its key carries no
        // buffer identity, so an entry would dispatch against a dead arena and keep
        // it alive. The fresh compile's cache is empty.
        self.gemv_setup = fresh.gemv_setup;

        self.replay_pending_params();
        // Restore the unresolved graph so a later binding recompiles again.
        self.unresolved = original;
    }

    /// Write the buffered param uploads into the arena this executable holds.
    ///
    /// While a graph is unresolved `set_param` / `set_param_bytes` only buffer,
    /// so this is the one place a deferred executable's params reach the device.
    /// The buffered set is the record every later rebind (and `clone_for_cache`)
    /// replays from, so it is kept rather than consumed.
    fn replay_pending_params(&mut self) {
        let original = self.unresolved.take();
        let params = std::mem::take(&mut self.pending_params);
        let bytes = std::mem::take(&mut self.pending_param_bytes);
        for (name, data) in &params {
            self.set_param(name, data);
        }
        for (name, data) in &bytes {
            self.set_param_bytes(name, data);
        }
        self.pending_params = params;
        self.pending_param_bytes = bytes;
        self.pending_params_dirty = false;
        self.unresolved = original;
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }

    /// Compile placeholder for a graph with `Dim::Dynamic` entries.
    /// The real compile happens on the first `run()` once input data
    /// reveals the symbol → size bindings. Buffered params (set via
    /// `set_param` / `set_param_bytes` before run) are replayed.
    pub(crate) fn deferred(
        graph: Graph,
        rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    ) -> Self {
        let dev = wgpu_device().expect("rlx-wgpu: no compatible adapter found");
        // Minimal valid arena buffer. Replaced on first run().
        let placeholder = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu deferred placeholder"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let arena = Arena {
            buffer: placeholder,
            extra_shards: Vec::new(),
            shard_size: 0,
            f16_buffer: None,
            bf16_weight_buffer: None,
            bf16_packed_nodes: std::collections::HashSet::new(),
            offsets: HashMap::new(),
            lens: HashMap::new(),
            size: 0,
            scratch_off: 0,
            scratch_bytes: 0,
            weight_buffer: None,
            weight_extra_shards: Vec::new(),
            weight_shard_size: 0,
            weight_offsets: HashMap::new(),
            weight_share: None,
            weight_written: std::sync::Mutex::new(HashSet::new()),
            plan_slots: HashMap::new(),
            param_slots: HashMap::new(),
        };
        Self {
            graph: graph.clone(),
            arena,
            dequant_scratch_off: 0,
            gdn_scratch_off: 0,
            schedule: Vec::new(),
            step_nodes: Vec::new(),
            in_snapshot: false,
            snapshot_stop: None,
            input_offsets: HashMap::new(),
            param_offsets: HashMap::new(),
            uniforms: Vec::new(),
            bind_groups: Vec::new(),
            meta_buffers: Vec::new(),
            unresolved: Some(graph),
            last_binding: None,
            pending_params: HashMap::new(),
            pending_param_bytes: HashMap::new(),
            pending_params_dirty: false,
            active_extent: None,
            uniforms_active_extent: None,
            coop_f16_vk: false,
            coop_f16_b_param: HashMap::new(),
            coop_f16_vk_wide_b: HashSet::new(),
            coop_f16_vk_wide_bind_groups: HashMap::new(),
            coop_f16_host_activations: Vec::new(),
            stashed_params: HashMap::new(),
            readback_staging: None,
            tiny_readback: None,
            dispatch_only: false,
            fft_gpu_steps: Vec::new(),
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            kv_row_feeds: HashMap::new(),
            kv_feed_staging: None,
            pending_read_indices: None,
            rng,
            static_once_steps: HashSet::new(),
            gemv_setup: crate::gguf_gpu::GemvSetupCache::default(),
            static_once_done: false,
            ceiling_hosted: 0,
            im2col_fallbacks: 0,
            conv1d_implicit_declines: 0,
            attention_window_declines: 0,
            attention_band_declines: 0,
            attention_dense_coop_declines: 0,
        }
    }

    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    /// The active extent (PLAN L1) actually in force: the requested one, or none
    /// at all when any step in the schedule cannot honour it (the mode is
    /// all-or-nothing). Both readers — the run loop and the fed-handle copy — must
    /// agree on it: a copy sized from the *requested* extent while the run wrote the
    /// *full* one would leave a stale tail.
    pub(crate) fn active_extent_in_effect(&self) -> Option<(usize, usize)> {
        self.active_extent.filter(|_| self.all_safe_for_active())
    }

    /// Rewrite the draw seeds of this executable's `Sample` steps, in schedule
    /// order — the one per-run knob a sampling graph needs, since `Op::Sample`'s
    /// seed is a uniform word and not a pipeline constant.
    ///
    /// Answers `false` (writing nothing) when the schedule's `Sample` count is
    /// not `seeds.len()` or the device is gone, so a graph whose sampler lowered
    /// to `Argmax` — greedy draws, `temperature <= 0` or `top_k == 1` — is asked
    /// with an empty slice rather than silently keeping a stale seed. The seeds
    /// reach the device here — the two words of each `Sample`'s uniform, not the
    /// whole struct — because `run` skips its whole uniform loop when the active
    /// extent is unchanged, so a seed change would otherwise never arrive.
    pub fn set_sample_seeds(&mut self, seeds: &[u64]) -> bool {
        let count = self
            .schedule
            .iter()
            .filter(|s| matches!(s, Step::Sample { .. }))
            .count();
        if count != seeds.len() {
            return false;
        }
        let Some(dev) = wgpu_device() else {
            return false;
        };
        let mut i = 0;
        let mut gpu_ui = 0usize;
        for step in &mut self.schedule {
            // The run loop's own cursor rule: a step that owns no uniform slot is
            // skipped, and every step that owns one advances the cursor.
            if !step_owns_slot(step) {
                continue;
            }
            if let Step::Sample { params } = step {
                let seed = seeds[i];
                params.seed_lo = seed as u32;
                params.seed_hi = (seed >> 32) as u32;
                // Only the seed's two words: the rest of the uniform — `outer`
                // above all, which the run loop writes scaled to the active
                // extent — holds what the last run left there, and rewriting the
                // whole struct here would restore the full extent it trimmed.
                let at = std::mem::offset_of!(crate::kernels::SampleParams, seed_lo) as u64;
                dev.queue.write_buffer(
                    &self.uniforms[gpu_ui],
                    at,
                    bytemuck::cast_slice(&[params.seed_lo, params.seed_hi]),
                );
                i += 1;
            }
            gpu_ui += 1;
        }
        true
    }

    /// Steps the per-binding-ceiling rule routed to the host route. Non-zero means
    /// the graph's arena exceeded the device's per-binding limit (or a parameter
    /// was parked in the weight buffer) and at least one absolute-addressing op
    /// was computed on the host instead of through a re-based window. Exposed so a
    /// caller (and the regression test) can observe the route rather than infer it
    /// from timing.
    pub fn ceiling_hosted_steps(&self) -> usize {
        self.ceiling_hosted
    }

    /// Convs whose shape qualified for the im2col route but which took the
    /// direct/tiled route for a reason the shapes alone do not settle (see the
    /// `Op::Conv` arm of `crate::backend::compile`, where the count is taken).
    /// Exposed so the fallback is observable — the schedule report prints it, and the
    /// route tests assert both arms of it, zero on a graph that binds and parks nothing
    /// and the qualifying conv's own count where it does — instead of the route that
    /// was *not* taken having to be inferred from a timing.
    pub fn im2col_fallbacks(&self) -> usize {
        self.im2col_fallbacks
    }

    /// Many-row attention steps this compile kept on the fused kernel for want of
    /// a window over the route's regions (see the field of the same name). Every
    /// other term of that decision is the graph's own, so a non-zero count is the
    /// arena and not the geometry; the schedule report prints it, and the lowering's
    /// attention debug print says which term it was, for the step's own addresses.
    pub fn attention_window_declines(&self) -> usize {
        self.attention_window_declines
    }

    /// Dense square attention steps this compile kept on the fused walk because
    /// the band kernel's admission refused their shapes and the cooperative form
    /// was not booked for them (see the field of the same name). A non-zero count
    /// is geometry the dense-through-band route could not hold, never an adapter
    /// term; the schedule report prints it.
    pub fn attention_band_declines(&self) -> usize {
        self.attention_band_declines
    }

    /// Dense square attention steps whose shapes the cooperative form admits but
    /// which this compile kept on another arrangement (see the field of the same
    /// name). Device, arena and switch terms only, never geometry; the schedule
    /// report prints it.
    pub fn attention_dense_coop_declines(&self) -> usize {
        self.attention_dense_coop_declines
    }

    /// Debug helper: run forward, then read every node slot back and
    /// report the first node whose output contains a NaN, plus a
    /// summary of the *previous* finite node's value range so the
    /// caller can see the input that broke. Slow — diagnosis only.
    pub fn debug_first_nan_node(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Option<(usize, String, String)> {
        let _ = self.run(inputs);
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        let mut prev_summary = String::from("(none)");
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if !self.arena.has(node.id) {
                continue;
            }
            let elems = node.shape.num_elements().unwrap_or(0);
            if elems == 0 {
                continue;
            }
            let data = self.arena.read_f32(&dev.device, &dev.queue, node.id);
            let nan_count = data.iter().filter(|v| v.is_nan()).count();
            let inf_count = data.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                return Some((i, format!("{:?}", node.op), prev_summary));
            }
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let abs_max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            prev_summary = format!(
                "node #{i} {:?} shape={:?}  min={min:.6e} max={max:.6e} |max|={abs_max:.6e}",
                node.op,
                node.shape
                    .dims()
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect::<Vec<_>>()
            );
        }
        None
    }

    /// Declared output dtypes (one per graph output). Used by the
    /// runtime wrapper's `run_typed` to narrow F32 results back to
    /// F16/BF16 etc. on the way out.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

    pub(crate) fn dump_node_stats_if_requested(&self, dev: &crate::device::WgpuDevice) {
        if !rlx_ir::env::flag("RLX_WGPU_DUMP_NODES") {
            return;
        }
        let flat_probe = rlx_ir::env::parse_or::<usize>("RLX_WGPU_DUMP_FLAT", usize::MAX);
        let limit = rlx_ir::env::parse_or("RLX_WGPU_DUMP_NODES_LIMIT", 40usize);
        let from_end = rlx_ir::env::flag("RLX_WGPU_DUMP_TAIL");
        eprintln!("[rlx-wgpu-dump] per-node max |x| (limit={limit}, from_end={from_end})");
        let mut candidates: Vec<(usize, &rlx_ir::Node)> = self
            .graph
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                if !self.arena.has(node.id) {
                    return false;
                }
                if let Some(spec) = rlx_ir::env::var("RLX_WGPU_DUMP_IDS") {
                    let want: std::collections::HashSet<u32> = spec
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    return want.contains(&node.id.0);
                }
                !matches!(
                    node.op,
                    rlx_ir::Op::Input { .. }
                        | rlx_ir::Op::Param { .. }
                        | rlx_ir::Op::Constant { .. }
                        | rlx_ir::Op::Cast { .. }
                )
            })
            .collect();
        if from_end {
            candidates.reverse();
        }
        for (shown, (i, node)) in candidates.into_iter().take(limit).enumerate() {
            let _ = shown;
            let data = self.arena.read_f32(&dev.device, &dev.queue, node.id);
            let max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let nz = data.iter().filter(|&&v| v != 0.0).count();
            let flat_s = if flat_probe < data.len() {
                format!(" flat[{flat_probe}]={:.6}", data[flat_probe])
            } else {
                String::new()
            };
            // A node no step writes has no value of its own — it was elided,
            // folded into a consumer, or is a pure view. Saying `max=0` for it
            // is how a post-hoc dump invents divergences that are not there.
            let written = self.step_nodes.contains(&node.id);
            eprintln!(
                "  [{i:>3}] {:?} id={:?} off={} max={max:.6} nonzero={}/{}{flat_s}{}",
                node.op,
                node.id,
                self.arena.offset(node.id),
                nz,
                data.len(),
                if written {
                    ""
                } else {
                    "  [no writer: elided/fused/view]"
                }
            );
            if rlx_ir::env::flag("RLX_WGPU_DUMP_INPUTS") {
                for (j, &inp) in node.inputs.iter().enumerate() {
                    if self.arena.has(inp) {
                        let d = self.arena.read_f32(&dev.device, &dev.queue, inp);
                        let mx = d.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                        eprintln!(
                            "       in[{j}] {:?} op={:?} off={} max={mx:.6} n={}",
                            inp,
                            self.graph.node(inp).op,
                            self.arena.offset(inp),
                            d.len()
                        );
                    }
                }
            }
        }
    }

    /// The debug scans that read the whole arena back once a run's device work is
    /// complete: the node-stats dump and the per-node NaN/Inf localization. Every
    /// tail that reads outputs back runs this, not only the post-loop one — a batch
    /// that carries its work to the end returns through the fused readback (or the
    /// tiny / deferred-tail readback), so a scan wired only to the run loop's exit
    /// left those runs unscanned. The zero-filled return (`skip_readback` /
    /// `defer_tail`) reads nothing back and does not come here.
    ///
    /// Both knobs are debug-only, so this costs a normal run nothing; the two
    /// tiny-readback tails deliberately gain the node-stats dump as well, since a
    /// tail that reads outputs back is exactly where the dump belongs.
    pub(crate) fn debug_scans_after_run(&self, dev: &crate::device::WgpuDevice) {
        self.dump_node_stats_if_requested(dev);
        // NaN/Inf localization (`RLX_DEBUG_NANS`; `RLX_WGPU_NAN_TRACE` kept as an
        // alias). wgpu can read every node's buffer back from the arena after the
        // run, so this is a true per-node scan in topological order — the first bad
        // node is the origin, and the shared scanner classifies it as culprit vs
        // propagator with provenance + a fix hint.
        let scanner = {
            let s = rlx_ir::numeric_check::DebugScanner::from_env("wgpu");
            if s.enabled() {
                Some(s)
            } else if rlx_ir::env::flag("RLX_WGPU_NAN_TRACE") {
                Some(rlx_ir::numeric_check::DebugScanner::with_mode(
                    rlx_ir::numeric_check::DebugMode::Warn,
                    "wgpu",
                ))
            } else {
                None
            }
        };
        if let Some(scanner) = scanner {
            let mut found = false;
            for node in self.graph.nodes() {
                if !self.arena.has(node.id)
                    || matches!(
                        node.op,
                        rlx_ir::Op::Input { .. }
                            | rlx_ir::Op::Param { .. }
                            | rlx_ir::Op::Constant { .. }
                    )
                {
                    continue;
                }
                let out = self.arena.read_f32(&dev.device, &dev.queue, node.id);
                // Gather f32 operand buffers for culprit-vs-propagator (skip
                // packed U8/I8 quant weights, which aren't f32).
                let mut owned: Vec<(rlx_ir::NodeId, Vec<f32>)> = Vec::new();
                for &inp in &node.inputs {
                    let ish = self.graph.node(inp).shape.dtype();
                    if self.arena.has(inp) && !matches!(ish, rlx_ir::DType::U8 | rlx_ir::DType::I8)
                    {
                        owned.push((inp, self.arena.read_f32(&dev.device, &dev.queue, inp)));
                    }
                }
                let inrefs: Vec<(rlx_ir::NodeId, &[f32])> =
                    owned.iter().map(|(id, v)| (*id, v.as_slice())).collect();
                if scanner.check(&self.graph, node.id, &out, &inrefs).is_some() {
                    found = true;
                    break; // first bad in topo order is the origin
                }
            }
            if !found {
                eprintln!("rlx nan-check [wgpu]: clean run — no NaN/Inf");
            }
        }
    }

    /// Whether this executable can bind `name` as a GPU handle — the one question
    /// [`Self::bind_gpu_handle`], the two feed registrations and
    /// [`Self::prepare_resident_gpu_handle`] all ask.
    fn can_bind_gpu_handle(&self, name: &str) -> bool {
        self.input_offsets.contains_key(name)
    }

    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.can_bind_gpu_handle(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    /// Whether this executable holds `name`'s rows as a GPU handle at all: a host
    /// copy of them, or — once they are resident — the mark left where the mirror
    /// was emptied because the arena's rows became the authoritative ones. A
    /// different question from `can_bind_gpu_handle`, which asks whether `name` is
    /// a graph input here.
    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    /// The whole handle on the host, or `None` — never an empty tensor — for a
    /// handle whose rows this executable no longer holds. The rule the read
    /// family follows is stated in [`Self::read_gpu_handle_rows`].
    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(id) = self.resolve_handle_slot(name) {
            let dev = wgpu_device()?;
            return Some(self.arena.read_f32(&dev.device, &dev.queue, id));
        }
        // No slot: the host mirror is the answer, and an empty one is a refusal,
        // not a zero-element tensor. The row reads bound a mirror by the declared
        // extent; an emptied one declares nothing.
        self.gpu_handles
            .get(name)
            .filter(|rows| !rows.is_empty())
            .cloned()
    }

    /// The arena slot a named handle's rows are in right now, or `None` when only
    /// the host mirror is left. Both reads ask this one ladder — fed output slot,
    /// else resident input slot — so their answers cannot drift apart.
    fn resolve_handle_slot(&self, name: &str) -> Option<NodeId> {
        if let Some(id) = self
            .gpu_handle_feeds
            .get(name)
            .and_then(|&out_idx| self.graph.outputs.get(out_idx).copied())
            .filter(|id| self.arena.has(*id))
        {
            return Some(id);
        }
        if self.gpu_handle_resident.contains(name) {
            return self
                .input_offsets
                .get(name)
                .copied()
                .filter(|id| self.arena.has(*id));
        }
        None
    }

    /// Where a named handle's row is. Both readback paths ask this one ladder, so
    /// a row resolved on its own and one resolved inside a step's batch cannot be
    /// answered differently.
    fn locate_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<LocatedRow<'_>> {
        let Some(id) = self.resolve_handle_slot(name) else {
            // No slot here: the host mirror, bounded by what the name declares —
            // it may be a longer binding than the slot it was uploaded into.
            let (start, end) = Self::row_bounds(row, row_inner)?;
            let mirror = self.gpu_handles.get(name)?;
            let declared = self.declared_extent_elems(*self.input_offsets.get(name)?, mirror.len());
            if end > declared {
                return None;
            }
            return Some(LocatedRow::Mirror(&mirror[start..end]));
        };
        let (byte_off, len) = self.arena_row_range(id, row, row_inner)?;
        Some(LocatedRow::Arena { byte_off, len })
    }

    /// The checked `(start, end)` element bounds of one row of a row-major
    /// carrier. `None` for a zero-length row, and for an index that would wrap
    /// rather than land inside — these are release products, so they are checked.
    fn row_bounds(row: usize, row_inner: usize) -> Option<(usize, usize)> {
        if row_inner == 0 {
            return None;
        }
        let start = row.checked_mul(row_inner)?;
        Some((start, start.checked_add(row_inner)?))
    }

    /// The `(byte_off, byte_len)` of one arena row of the slot `id`, bounded by
    /// the extent it declares — the one place a row's arena address is computed,
    /// so the KV feed's row and the two read paths cannot disagree on which
    /// bytes a row is. A slot this executable does not hold declares no rows, so
    /// every row of it is out of extent.
    fn arena_row_range(&self, id: NodeId, row: usize, row_inner: usize) -> Option<(usize, usize)> {
        let (start, end) = Self::row_bounds(row, row_inner)?;
        if end > self.extent_elems(id) {
            return None;
        }
        let f32_bytes = std::mem::size_of::<f32>();
        Some((
            self.arena.offset(id) + start * f32_bytes,
            row_inner * f32_bytes,
        ))
    }

    /// Read one row (`row_inner` f32 elements at `row`) of a named handle,
    /// resolving the slot exactly as [`Self::read_gpu_handle`] does — fed output
    /// slot, else resident arena slot, else host mirror — but copying only that
    /// row to the host, so the read costs the row and not the cache's capacity.
    ///
    /// `None` when the row is outside the extent the handle declares, and for a
    /// zero-length row: a shortened answer could not be told apart from a
    /// truncated one. A slot this executable no longer holds does not fall back
    /// to bytes somewhere else — the answer is nothing.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        self.read_gpu_handle_rows(&[(name, row, row_inner)])
            .pop()
            .flatten()
    }

    /// Read many rows in **one** device round trip, answering in request order:
    /// everything a decode step consumes, at once. A step's rows span one handle
    /// per layer per K/V, so a read that covers a single handle pays a submission
    /// and a wait per row however small the rows are, while here the copies are
    /// encoded together and answered by one map — the cost follows the rows asked
    /// for, not how many handles they are spread across, nor how long the caches
    /// are.
    ///
    /// Every answer holds exactly what [`Self::read_gpu_handle_row`] answers for
    /// that request, and a `None` is that request's alone: a row that is not there
    /// withholds nothing its neighbours asked for. A name whose rows are only in
    /// this executable's host mirror is answered from it and asks nothing of the
    /// device, so a call that asks for no arena row submits nothing at all.
    pub fn read_gpu_handle_rows(&self, rows: &[(&str, usize, usize)]) -> Vec<Option<Vec<f32>>> {
        let mut answers: Vec<Option<Vec<f32>>> = Vec::with_capacity(rows.len());
        // The arena-backed requests in the order asked, each with the answer slot
        // it fills.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        for &(name, row, row_inner) in rows {
            match self.locate_handle_row(name, row, row_inner) {
                None => answers.push(None),
                Some(LocatedRow::Mirror(mirror)) => answers.push(Some(mirror.to_vec())),
                Some(LocatedRow::Arena { byte_off, len }) => {
                    slots.push(answers.len());
                    answers.push(None);
                    ranges.push((byte_off, len));
                }
            }
        }
        if !ranges.is_empty() {
            // A device torn down between calls answers nothing, like a row it
            // refuses, and the caller falls back.
            let Some(dev) = wgpu_device() else {
                return answers;
            };
            let read = self.arena.read_f32_ranges(&dev.device, &dev.queue, &ranges);
            // One answer per range, or the slots left `None` below would read as
            // requests the batch refused.
            assert_eq!(
                read.len(),
                ranges.len(),
                "rlx-wgpu: a row readback answered {} of {} ranges",
                read.len(),
                ranges.len()
            );
            for (&slot, row) in slots.iter().zip(read) {
                answers[slot] = Some(row);
            }
        }
        answers
    }

    /// Read one row of a row-major graph output without a whole-tensor D2H —
    /// the read half of the `kv_resident` contract, next to the row feed.
    ///
    /// Bounded by the extent the output declares, and `None` — not a short
    /// answer — outside it or for a zero-length row; a slot this executable no
    /// longer holds declares no rows, so every row of it is out of extent. One
    /// submission and one map, the same staging path
    /// [`Self::read_gpu_handle_rows`] uses.
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        let id = *self.graph.outputs.get(out_idx)?;
        let (byte_off, len) = self.arena_row_range(id, row, row_inner)?;
        let dev = wgpu_device()?;
        self.arena
            .read_f32_ranges(&dev.device, &dev.queue, &[(byte_off, len)])
            .pop()
    }

    /// The extent the node behind `id` declares, in f32 elements, never more than
    /// `held` — the slot or mirror that carries its rows: an unresolved shape
    /// leaves the carrier itself as the extent. The row feed and both read paths
    /// bound their row by this, so a row that does not exist is neither written
    /// beside the cache nor read out of it.
    fn declared_extent_elems(&self, id: NodeId, held: usize) -> usize {
        self.graph
            .node(id)
            .shape
            .num_elements()
            .unwrap_or(held)
            .min(held)
    }

    /// [`Self::declared_extent_elems`] capped by the node's arena slot — the
    /// extent of any node whose rows this executable holds, cache input or feed
    /// output alike.
    fn extent_elems(&self, id: NodeId) -> usize {
        let slot = if self.arena.has(id) {
            self.arena.len_of(id) / std::mem::size_of::<f32>()
        } else {
            0
        };
        self.declared_extent_elems(id, slot)
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let graph = self
            .unresolved
            .clone()
            .unwrap_or_else(|| self.graph.clone());
        let mut exe = Self::compile_rng(graph, self.rng());
        for (k, v) in &self.stashed_params {
            exe.set_param(k, v);
        }
        for (k, v) in &self.pending_params {
            exe.set_param(k, v);
        }
        for (k, v) in &self.pending_param_bytes {
            exe.set_param_bytes(k, v);
        }
        for (k, v) in &self.gpu_handles {
            // An emptied mirror is not a handle: its rows became resident and the
            // arena stayed with the parent, so a clone must not report it as held.
            if v.is_empty() {
                continue;
            }
            exe.bind_gpu_handle(k, v);
        }
        // Copied, not re-registered: the feed map is a plain name→output record,
        // and a clone whose arena is still deferred has no input set to ask about
        // — asking would drop the registration its parent had, and a lost prefix
        // feed stops updating the cache with nothing reporting it.
        exe.gpu_handle_feeds.clone_from(&self.gpu_handle_feeds);
        exe.set_active_extent(self.active_extent);
        exe.set_rng(self.rng());
        exe
    }

    pub(crate) fn readback_plan(&self) -> Vec<usize> {
        let n = self.graph.outputs.len();
        if self.pending_read_indices.is_none() && self.gpu_handle_feeds.is_empty() {
            return (0..n).collect();
        }
        if let Some(ref want) = self.pending_read_indices {
            let mut v: Vec<_> = want.to_vec();
            v.sort_unstable();
            return v;
        }
        (0..n).collect()
    }

    pub(crate) fn propagate_gpu_handle_feeds_on_gpu(
        &mut self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
    ) {
        let extent = self.active_extent_in_effect();
        let feeds: Vec<(String, usize)> = self
            .gpu_handle_feeds
            .iter()
            .map(|(n, &i)| (n.clone(), i))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_bytes = self.arena.len_of(out_id);
                let copy_bytes = match extent {
                    Some((actual, upper)) if upper > 0 => {
                        let stride = (out_bytes / (upper + 1)).max(4);
                        (actual * stride).min(out_bytes)
                    }
                    _ => out_bytes,
                };
                if rlx_ir::env::flag("RLX_WGPU_FEED_TRACE") {
                    eprintln!(
                        "[feed] {name}: out#{} id={:?} -> in id={:?}  out_bytes={out_bytes} copy={copy_bytes} extent={extent:?}                          out_len={} in_len={}",
                        out_idx,
                        out_id,
                        in_id,
                        self.arena.len_of(out_id),
                        self.arena.len_of(in_id)
                    );
                }
                self.dispatch_arena_copy_bytes(dev, enc, out_id, in_id, copy_bytes);
            } else if rlx_ir::env::flag("RLX_WGPU_FEED_TRACE") {
                eprintln!("[feed] {name}: in_id == out_id ({in_id:?}) — aliased, no copy");
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    pub(crate) fn stage_gpu_handle_inputs(
        &mut self,
        dev: &crate::device::WgpuDevice,
        inputs: &[(&str, &[f32])],
    ) {
        if rlx_ir::env::flag("RLX_WGPU_FEED_TRACE") {
            let mut names: Vec<String> = self.gpu_handle_feeds.keys().cloned().collect();
            // Also the per-step scalars: lr, the Adam bias corrections and the
            // per-parameter gate. These change every step and are fed as inputs.
            for k in self.input_offsets.keys() {
                if k.starts_with("__") || k.ends_with("__scale") {
                    names.push(k.clone());
                }
            }
            for name in names {
                if let Some(&id) = self.input_offsets.get(name.as_str())
                    && self.arena.has(id)
                {
                    let v = self.arena.read_f32(&dev.device, &dev.queue, id);
                    eprintln!(
                        "[stage] {name}: in id={id:?} first={:?} resident={} host_len={}",
                        v.first(),
                        self.gpu_handle_resident.contains(&name),
                        self.gpu_handles.get(&name).map(|h| h.len()).unwrap_or(0)
                    );
                }
            }
        }
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                self.arena.write_f32(&dev.queue, id, data);
            }
        }
    }

    pub(crate) fn pack_readback_outputs(
        &mut self,
        plan: &[usize],
        partial: Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        if self.pending_read_indices.is_none() {
            for (pos, &out_i) in plan.iter().enumerate() {
                if let Some(data) = partial.get(pos) {
                    for (name, &feed_i) in &self.gpu_handle_feeds {
                        if feed_i == out_i {
                            self.gpu_handles.insert(name.clone(), data.clone());
                        }
                    }
                }
            }
        }
        if self.pending_read_indices.is_none() && plan.len() == self.graph.outputs.len() {
            return partial;
        }
        let want = self.pending_read_indices.as_deref().unwrap_or(plan);
        let mut by_idx = std::collections::HashMap::new();
        for (pos, &i) in plan.iter().enumerate() {
            if let Some(d) = partial.get(pos) {
                by_idx.insert(i, d.clone());
            }
        }
        want.iter()
            .map(|&i| {
                by_idx
                    .get(&i)
                    .cloned()
                    .expect("readback plan missing output")
            })
            .collect()
    }
}

/// WebGPU's spec cap on a single `dispatch_workgroups` dimension, which
/// [`dispatch_dims`] folds a wider workload across the dispatch's own dimensions
/// against, [`POINTWISE_MAX_SPATIAL`] scales by the pointwise kernel's columns,
/// and `attention_route`'s plan caps the extents the many-row attention
/// multiplies dispatch on. A few older sites in this crate still spell the number
/// out; they are left as they are.
pub(crate) const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// Compute a (X, Y, 1) workgroup grid for a 1-D workload.
///
/// WebGPU caps `dispatch_workgroups` per-dimension at
/// [`MAX_WORKGROUPS_PER_DIM`]. For workloads beyond
/// `MAX_WORKGROUPS_PER_DIM × workgroup_size_x` threads we split into
/// a 2-D grid; kernels recover the linear thread index via
/// `gid.x + gid.y * num_workgroups.x * 64u`.
fn dispatch_prologue_nchw(w: u32, h: u32, nc: u32) -> (u32, u32, u32) {
    (w.div_ceil(8).max(1), h.div_ceil(8).max(1), nc.max(1))
}

pub(crate) fn dispatch_dims(threads_total: u32, workgroup_size: u32) -> (u32, u32, u32) {
    let groups = threads_total.div_ceil(workgroup_size);
    if groups <= MAX_WORKGROUPS_PER_DIM {
        (groups, 1, 1)
    } else {
        let gx = MAX_WORKGROUPS_PER_DIM;
        let gy = groups.div_ceil(gx);
        (gx, gy, 1)
    }
}

/// The grid a kernel already counted in workgroups needs, folded across `y`
/// past one dimension's cap.
pub(crate) fn workgroup_grid(groups: u32) -> (u32, u32, u32) {
    dispatch_dims(groups, 1)
}

/// Threads in one workgroup of the movement kernels (`copy`/`narrow`/`concat`/
/// `transpose` and their `vec4` siblings), whose `@workgroup_size` is this. Every
/// grid those kernels are dispatched on divides by it, so the width is written
/// once here rather than as a literal at each call site.
pub(crate) const MOVE_WORKGROUP: u32 = 64;

/// Workgroups one flat `vec4` move dispatch of `elems` words needs.
pub(crate) fn move_workgroups(elems: u32) -> u32 {
    (elems / 4).div_ceil(MOVE_WORKGROUP)
}

/// One `vec4<f32>` move per thread is a pure 16 B lane copy, so it is only
/// legal when the contiguous span is whole vectors and both ends are 16 B
/// aligned. The offsets here are f32-word offsets from a bind-window base that
/// is 256 B aligned, so "word offset divisible by 4" is exactly "16 B aligned".
/// Anything else — an odd `inner`, a staged piece at a non-vector offset —
/// keeps the scalar kernel rather than re-deriving the shape.
///
/// The one eligibility predicate: every caller that picks a vector movement
/// kernel reads this, and a caller that must also fit a 3-D grid layers
/// `vec4_axes_fit` on top rather than restating the alignment terms.
pub(crate) fn vec4_move_ok(span: u32, in_off: u32, out_off: u32) -> bool {
    span.is_multiple_of(4) && in_off.is_multiple_of(4) && out_off.is_multiple_of(4)
}

/// Whether a norm row may run the `*_vec4` module: the row is whole `vec4`
/// groups (`inner % 4 == 0`) and every offset it touches is 16 B aligned. The
/// gate is the same alignment rule as [`vec4_move_ok`]; a row that fails it
/// keeps the scalar 64-lane module, which computes the identical function.
pub(crate) fn norm_vec4_ok(inner: u32, offs: &[u32]) -> bool {
    inner.is_multiple_of(4) && inner >= 4 && offs.iter().all(|o| o.is_multiple_of(4))
}

/// One flat `vec4` dispatch covers `elems`: `elems / 4` vectors at
/// [`MOVE_WORKGROUP`] threads each, inside the adapter's x dimension. `false`
/// means the span needs the scalar kernel, whose grid [`dispatch_dims`] splits:
/// the in-arena `copy_vec4` form reads only `gid.x` and has no second dimension
/// to fall back on.
pub(crate) fn vec4_flat_grid_ok(elems: u32, max_wg: u32) -> bool {
    move_workgroups(elems) <= max_wg
}

/// Encode one in-arena `src → dst` move of `nbytes` inside the storage window
/// `[base, base + size)`, as a `copy_vec4` dispatch when the span and both ends
/// are whole 16 B vectors and the x grid fits the adapter's cap, else the
/// scalar `copy` kernel.
///
/// Both callers that move bytes device-side share this — the KV-row feed
/// (`dispatch::dispatch_arena_copy_bytes`) and a lowering `Step::BufferCopy`
/// (`run::encode_arena_copy_dispatch`) — so the eligibility rule, the params
/// and the grid are written once. The pass is the caller's: its own pass was
/// what used to provide the barrier between this write and every later read of
/// the arena, and a caller that dispatches into a wider batch gets that
/// barrier from the batch's ordering instead.
pub(crate) fn encode_arena_copy(
    dev: &crate::device::WgpuDevice,
    arena: &Arena,
    enc: &mut wgpu::CommandEncoder,
    src: u64,
    dst: u64,
    nbytes: u64,
    base: u64,
    size: u64,
    label: &str,
) {
    let (ck, bg, vec4, elems) = prepare_arena_copy(dev, arena, src, dst, nbytes, base, size, label);
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        ..Default::default()
    });
    encode_arena_copy_into_pass(ck, &bg, vec4, elems, &mut pass);
}

/// Dispatch a move prepared by [`encode_arena_copy`]'s rules into an already
/// open pass. The eligibility, the params and the kernel are picked the same
/// way; only the pass is the caller's, so a run that batches several moves into
/// one pass pays one pass boundary instead of one per move.
pub(crate) fn encode_arena_copy_into_pass(
    ck: &Kernel,
    bg: &wgpu::BindGroup,
    vec4: bool,
    elems: u32,
    pass: &mut wgpu::ComputePass<'_>,
) {
    pass.set_pipeline(&ck.pipeline);
    pass.set_bind_group(0, bg, &[]);
    if vec4 {
        note_vec4_move();
        pass.dispatch_workgroups(move_workgroups(elems), 1, 1);
    } else {
        let (gx, gy, gz) = dispatch_dims(elems, MOVE_WORKGROUP);
        pass.dispatch_workgroups(gx, gy, gz);
    }
}

/// The pipeline, bind group and grid decision for an in-arena move, without a
/// pass: a caller that already holds one dispatches through
/// [`encode_arena_copy_into_pass`].
pub(crate) fn prepare_arena_copy(
    dev: &crate::device::WgpuDevice,
    arena: &Arena,
    src: u64,
    dst: u64,
    nbytes: u64,
    base: u64,
    size: u64,
    label: &str,
) -> (&'static Kernel, wgpu::BindGroup, bool, u32) {
    let elems = (nbytes / 4) as u32;
    let in_off = (src.saturating_sub(base) / 4) as u32;
    let out_off = (dst.saturating_sub(base) / 4) as u32;
    let vec4 = vec4_move_ok(elems, in_off, out_off)
        && vec4_flat_grid_ok(
            elems,
            dev.device.limits().max_compute_workgroups_per_dimension,
        );
    let p = crate::kernels::CopyParams {
        n: elems,
        in_off,
        out_off,
        _p0: 0,
        _p1: 0,
        _p2: 0,
        _p3: 0,
        _p4: 0,
    };
    let ck = if vec4 {
        crate::kernels::copy_vec4_kernel(&dev.device)
    } else {
        crate::kernels::copy_kernel(&dev.device)
    };
    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<crate::kernels::CopyParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = bind_arena_window(&dev.device, ck, arena, base, size, &u);
    (ck, bg, vec4, elems)
}

#[cfg(test)]
mod dispatch_dims_tests {
    use super::*;

    /// The two-dimensional branch, at the dimension boundary. The frames pins'
    /// shapes do not reach it (76 × 400 is 30 400 elements) — one pin does, and
    /// executes it (`frames_concat_fusion.rs`'s
    /// `a_framing_past_one_dispatch_dimension_reads_its_own_matrix`) — so what is
    /// pinned here, device-free, is the grid this branch builds and the shader's own
    /// `lin = gid.x + gid.y · nwg.x · WG`, which is what makes a 2-D grid a
    /// linearization rather than a hole.
    #[test]
    fn the_grid_splits_into_two_dimensions_at_the_dimension_cap() {
        let wg = crate::kernels::FRAMES_WORKGROUP;
        let one_row = MAX_WORKGROUPS_PER_DIM * wg;
        assert_eq!(
            dispatch_dims(one_row, wg),
            (MAX_WORKGROUPS_PER_DIM, 1, 1),
            "exactly one dimension's worth stays one-dimensional"
        );
        let (gx, gy, gz) = dispatch_dims(one_row + 1, wg);
        assert_eq!((gx, gz), (MAX_WORKGROUPS_PER_DIM, 1));
        assert_eq!(
            gy, 2,
            "one workgroup past the first dimension needs a second"
        );
        let covered = gx as u64 * gy as u64 * wg as u64;
        let wanted = (one_row + 1) as u64;
        assert!(
            covered >= wanted && covered - wanted < gx as u64 * wg as u64,
            "the grid must cover the workload in whole workgroup rows and no more: covered \
             {covered}"
        );
        let linearization: String = crate::kernels::FRAMES_WGSL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            linearization.contains("gid.x + gid.y * nwg.x * WG"),
            "the shader's linearization must be the one this grid assumes"
        );
    }
}

/// Shape/feature gate for CoopF16Vk (no operand tracing — avoids circular
/// dependency with compile-time f16 mirror planning).
///
/// **Default OFF.** The 16-bit cooperative-matrix matmul path — Vulkan only,
/// since this engine exposes no cooperative matrices on DX12 at all — silently
/// produces wrong output on BERT-family attention chains on at
/// least NVIDIA GPU (verified empirically against Bio_ClinicalBERT:
/// encoder cosine collapses from ≈1.0 on the wide-F32 fallback to ≈0.09
/// when the coop path runs, regardless of whether the kernel uses
/// F16-acc or F32-acc accumulators). The root cause is upstream — likely
/// in how wgpu's `coopLoadT` / `coopMultiplyAdd` interact with strided
/// arena buffers on non-Apple drivers — and needs a focused
/// reproducer before it can be fixed in `rlx-wgpu`. Until then the
/// correctness-first default is to route Vulkan matmuls through the
/// wide-F32 path, even though it's substantially slower (~80× on this
/// shape).
///
/// Opt back in (at the user's risk) with `RLX_WGPU_COOP_F16_VK_ENABLE=1`
/// — useful for measuring the perf headroom or for non-BERT models
/// where the precision loss may be acceptable. Legacy
/// `RLX_WGPU_NO_COOP_F16_VK=1` and explicit
/// `RLX_WGPU_COOP_F16_VK_DISABLE=1` are honored for completeness.
fn coop_f16_vk_eligible(dev: &wgpu::Device, m: u32, k: u32, n: u32) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_NO_COOP_F16_VK")
        || rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_DISABLE")
    {
        return false;
    }
    if !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_ENABLE") {
        return false;
    }
    m.is_multiple_of(16)
        && k.is_multiple_of(16)
        && n.is_multiple_of(16)
        && dev
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && dev.features().contains(wgpu::Features::SHADER_F16)
        && crate::device::coop_discrete_backend()
        && crate::device::coop_f16_16x16_supported()
}

fn dispatch_wide_f32_matmul(
    pass: &mut wgpu::ComputePass<'_>,
    mm_w_active: &Kernel,
    mm_k: &Kernel,
    m_s: u32,
    n: u32,
    batch: u32,
) {
    // Tile-size selection differs by GPU backend.
    //
    // **Vulkan / DX12** (small M takes the 32×32-tile kernel): when `m_s < 64`
    // the bottom rows of every workgroup's M-axis tile contain padded zeros that
    // the kernel still computes and writes back — pure wasted work on small-M
    // shapes like BERT-base prefill (m=32). The regular 32×32-tile kernel
    // sidesteps the M-axis padding and is ~8% faster end-to-end on NVIDIA GPU
    // (verified on Bio_ClinicalBERT: encoder forward 58.9 ms → 54.1 ms at cosine
    // 0.9999995 vs HF).
    //
    // **Metal / other**: the wider tile wins even on small M — Apple GPUs
    // prefer the larger workgroup and amortize the M-padding well. Forcing the
    // 32×32 kernel here regresses Mac WGPU encoder time (26.6 → 29.1 ms
    // verified).
    let backend = wgpu_device()
        .map(|d| d.backend)
        .unwrap_or(wgpu::Backend::Noop);
    let is_vulkan_dx12 = matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12);
    let prefer_small_for_m = is_vulkan_dx12 && m_s < 64;
    let use_wide = !prefer_small_for_m && m_s >= 32 && n >= 64;
    if use_wide {
        pass.set_pipeline(&mm_w_active.pipeline);
        let (gx, gy) = if is_vulkan_dx12 {
            (n.div_ceil(64), m_s.div_ceil(64))
        } else {
            (n.div_ceil(64), m_s.div_ceil(32))
        };
        pass.dispatch_workgroups(gx, gy, batch);
    } else {
        pass.set_pipeline(&mm_k.pipeline);
        pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), batch);
    }
}

fn coop_f16_vk_bind_group(exe: &WgpuExecutable, gpu_bi: usize, use_wide: bool) -> &wgpu::BindGroup {
    if use_wide {
        exe.coop_f16_vk_wide_bind_groups
            .get(&gpu_bi)
            .unwrap_or(&exe.bind_groups[gpu_bi])
    } else {
        &exe.bind_groups[gpu_bi]
    }
}

fn require_equal_shapes(graph: &Graph, ids: &[NodeId], op_name: &str) {
    let s0 = graph.node(ids[0]).shape.num_elements().unwrap_or(0);
    for &id in &ids[1..] {
        let si = graph.node(id).shape.num_elements().unwrap_or(0);
        if si != s0 {
            panic!(
                "rlx-wgpu {op_name}: broadcasting not yet implemented; \
                    inputs must have the same element count (got {s0} vs {si})"
            );
        }
    }
}

/// Can a kernel that addresses the arena with ABSOLUTE offsets be bound?
///
/// The shared coarse condition behind the per-binding-ceiling routing rule
/// ([`crate::supported_ops::addresses_arena_by_absolute_offset`] +
/// `lower::should_host_at_ceiling`): it is a thin device-limit wrapper over
/// [`Arena::binds_whole_at_base_zero`](crate::buffer::Arena::binds_whole_at_base_zero),
/// which also rejects an arena whose word offsets would not fit the `u32` a
/// uniform declares. The rule hosts on its own for an op with a parked parameter
/// operand, which this predicate does not see (see `lower::should_host_at_ceiling`).
///
/// `bind_op_output_window` rebases the binding onto a window around the op's
/// operands when the arena is sharded or larger than one storage binding, and an
/// absolute offset is meaningless against a rebased window — it would read and
/// write the wrong slots, silently. Kernels that carry absolute offsets (the
/// `scaled_lowp` / `quant_i8` / `q_matmul` / `q_conv2d` family, matching the
/// `ScaledGroupedMatMul` convention) ask this first and keep the CPU host route
/// when it says no.
///
/// **Exercised.** A dedicated test drives the striping branch through a small
/// `RLX_WGPU_SHARD_CAP_MIB` with `RLX_WGPU_ALLOW_SHARD=1`, and the same predicate
/// is asked with a caller-supplied (hence synthetic) device limit, so both
/// answers are reachable without a multi-gigabyte allocation.
pub(crate) fn arena_binds_whole_at_zero(dev: &wgpu::Device, arena: &Arena) -> bool {
    arena.binds_whole_at_base_zero(dev.limits().max_storage_buffer_binding_size)
}

fn arena_whole_arena_bind(arena: &Arena, max_binding: u64) -> Option<(u64, u64)> {
    if arena.is_sharded() {
        return None;
    }
    let need = arena.size as u64;
    if need > max_binding {
        return None;
    }
    // Bind size must not exceed the allocated buffer (planner may leave a small tail gap).
    let buf_bytes = arena.buffer.size();
    let size = need.min(buf_bytes).max(256);
    Some((0, size))
}

/// Clamp `[base, base+size)` so it never runs past the physical buffer.
/// `aligned_bind_size` already truncates silently — without this, lower.rs
/// keeps computing locals against the *requested* size and the shader sees
/// zeros past the real bind (F5 DiT on a single >4 GiB Metal buffer).
fn arena_clamp_bind_window(arena: &Arena, base: &mut u64, size: &mut u64) {
    let buf = if arena.is_sharded() {
        arena_bind_buf(arena, *base).0.size()
    } else {
        arena.buffer.size()
    };
    let local_base = if arena.is_sharded() {
        arena_bind_buf(arena, *base).1
    } else {
        *base
    };
    let cap = buf.saturating_sub(local_base).max(256);
    if *size > cap {
        // On a SHARDED arena this truncation is the silent-wrong-answer path.
        //
        // A kernel binds one stripe. Clamping the window here means the op asked
        // for more than the stripe holds, so whatever lay past the boundary is
        // simply not bound — and reads of it come back as **zero**. No error, no
        // warning, and a result that is entirely plausible: a convolution in that
        // state returns its bias for every voxel, which looks like a real feature
        // map. `Arena::straddles_shards` exists to catch this at compile time,
        // before any wrong number exists; this is the last line of defence for
        // whatever it does not cover.
        //
        // Unsharded arenas clamp benignly (the caller over-requested against the
        // single buffer's tail), so the report is gated on `is_sharded`.
        if arena.is_sharded() {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[rlx-wgpu] WARNING: bind window clamped on a sharded arena \
                     (base={base} requested={size} available={cap}). Operands of \
                     this op do not share a stripe; the part outside the bound \
                     window reads as ZERO, silently. Host the op \
                     (`RLX_WGPU_FORCE_HOST=1`) or raise the shard size. \
                     Reported once per process."
                );
            }
        }
        *size = cap;
    }
}

/// Map a logical rebase (window start in arena address space) to the physical
/// buffer + local bind offset.
fn arena_bind_buf(arena: &Arena, rebase: u64) -> (&wgpu::Buffer, u64) {
    if arena.is_sharded() {
        let (buf, local) = arena.resolve_act(rebase as usize);
        (buf, local as u64)
    } else {
        (&arena.buffer, rebase)
    }
}

/// Bind arena storage at a logical rebase window (shard-aware).
fn bind_arena_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    mut rebase: u64,
    mut size: u64,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    arena_clamp_bind_window(arena, &mut rebase, &mut size);
    let (buf, local) = arena_bind_buf(arena, rebase);
    bind_two_buf0_window(device, kernel, buf, local, size, params, None)
}

/// True when activation nodes do not all lie in one shard stripe.
fn arena_span_crosses_shard(arena: &Arena, ids: &[NodeId]) -> bool {
    if !arena.is_sharded() {
        return false;
    }
    let s = arena.shard_size as u64;
    let mut shard: Option<u64> = None;
    for &id in ids {
        let off = arena.offset(id);
        if crate::buffer::is_weight_off(off) {
            continue;
        }
        let o = off as u64;
        let end = o.saturating_add(arena.len_of(id) as u64).saturating_sub(1);
        let s0 = o / s;
        let s1 = end / s;
        if s0 != s1 {
            return true;
        }
        match shard {
            None => shard = Some(s0),
            Some(prev) if prev != s0 => return true,
            _ => {}
        }
    }
    false
}

/// Pick a single-shard subset for windowing when `ids` span multiple shards.
/// Prefers the shard of the **first** activation id (callers put the op output
/// first) so GPU writes land in the real output slot; inputs are staged in.
fn arena_ids_for_shard_window(arena: &Arena, ids: &[NodeId]) -> Vec<NodeId> {
    if !arena_span_crosses_shard(arena, ids) {
        let acts: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|&id| !crate::buffer::is_weight_off(arena.offset(id)))
            .collect();
        return acts;
    }
    let s = arena.shard_size as u64;
    let prefer_shard = ids
        .iter()
        .find(|&&id| !crate::buffer::is_weight_off(arena.offset(id)))
        .map(|&id| (arena.offset(id) as u64) / s);
    let Some(want) = prefer_shard else {
        return Vec::new();
    };
    ids.iter()
        .copied()
        .filter(|&id| {
            let off = arena.offset(id);
            !crate::buffer::is_weight_off(off) && (off as u64) / s == want
        })
        .collect()
}

fn arena_window_for_nodes(dev: &wgpu::Device, arena: &Arena, ids: &[NodeId]) -> (u64, u64) {
    if arena.is_sharded() {
        let local_ids = arena_ids_for_shard_window(arena, ids);
        let spec = arena.bind_spec_for_nodes(dev, &local_ids);
        return (spec.rebase, spec.size);
    }
    // wgpu requires storage buffer binding offsets aligned to 256 bytes.
    const ALIGN: u64 = 256;
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    if let Some(w) = arena_whole_arena_bind(arena, max_binding) {
        return w;
    }
    let mut lo: u64 = u64::MAX;
    let mut hi: u64 = 0;
    for &id in ids {
        let off = arena.offset(id);
        if crate::buffer::is_weight_off(off) {
            continue;
        }
        let o = off as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(o);
        hi = hi.max(o.saturating_add(len));
    }
    if lo == u64::MAX {
        return (0, max_binding.max(256));
    }
    let span = hi.saturating_sub(lo).max(1);
    if span > max_binding {
        let mut details = String::new();
        for &id in ids.iter().take(6) {
            let off = arena.offset(id);
            let len = arena.len_of(id);
            details.push_str(&format!(" id={id:?}@{off}+{len};"));
        }
        panic!(
            "rlx-wgpu: op needs {} bytes of arena span (>{});{}",
            span, max_binding, details
        );
    }
    let mut base = (lo / ALIGN) * ALIGN;
    // Bind only the byte span the op needs (not the full 4 GiB cap) so we
    // don't slide the window to the arena tail and drop low-offset tensors.
    let mut size = span.div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(max_binding);
    if base.saturating_add(size) > arena.size as u64 {
        base = (arena.size as u64).saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    if base > lo || base.saturating_add(size) < hi {
        base = (lo / ALIGN) * ALIGN;
        size = hi.saturating_sub(base).div_ceil(ALIGN) * ALIGN;
        size = size.max(256).min(max_binding);
        if base.saturating_add(size) > arena.size as u64 {
            base = hi.saturating_sub(size);
            base = (base / ALIGN) * ALIGN;
        }
    }
    arena_clamp_bind_window(arena, &mut base, &mut size);
    (base, size)
}

fn arena_local_off_f32(arena: &Arena, id: NodeId, base: u64) -> u32 {
    let off = arena.offset(id);
    // A weight-buffer offset carries `WEIGHT_BUF_TAG` (bit 62) and indexes a
    // *different buffer*. Rebasing it against an act-arena window is not just
    // wrong, it is SILENTLY wrong: `(tagged - base) / 4` truncated to u32 comes
    // out as a small, perfectly plausible index (bit 62 vanishes in the cast),
    // so the kernel reads whatever activation happens to sit there — usually
    // zeros, this early in a graph. That is how Qwen3-0.6B F32 on a >4 GiB
    // arena produced an all-zero token embedding and emitted token id 0
    // forever. Callers must stage weight params into the act window
    // (`arena_off_in_window_or_stage`) or bind the weight buffer separately
    // (`run_gather_split`, fused GGUF GEMV); there is no correct rebase here.
    assert!(
        !crate::buffer::is_weight_off(off),
        "rlx-wgpu: arena_local_off_f32 on weight-buffer node {id:?} (off={off:#x}) - stage it into the act window or bind the weight buffer separately"
    );
    (((off as u64).saturating_sub(base)) / 4) as u32
}

/// Split-binding embedding gather for >4 GiB arenas (see `Step::GatherSplit`).
///
/// The table and the idx/output slots can be more than one ≤4 GiB binding
/// window apart, so they're bound as separate read-only windows of the arena;
/// the output goes to a dedicated read-write buffer that is copied back into
/// the arena afterwards (the arena cannot also be bound read-write in the same
/// dispatch — wgpu treats STORAGE_READ_WRITE as exclusive per buffer). Mirrors
/// [`crate::gguf_gpu::run_dequant_matmul_gguf_gemv`].
#[allow(clippy::too_many_arguments)]
fn run_gather_split(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n_out: u32,
    n_idx: u32,
    dim: u32,
    vocab: u32,
    idx_bias: f32,
    table_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
) {
    const ALIGN: u64 = 256;
    let max_bind = device.limits().max_storage_buffer_binding_size;

    // Table / idx may live in the weight buffer (tagged offsets) or the act
    // arena — always `resolve_w` so bit-62 tags are stripped correctly.
    let t_bytes = (vocab as u64) * (dim as u64) * 4;
    let (t_buf, t_raw) = arena.resolve_w(table_byte_off);
    let t_base = (t_raw as u64 / ALIGN) * ALIGN;
    let t_local = t_base;
    let t_size = ((t_raw as u64 + t_bytes - t_base).div_ceil(16) * 16)
        .min(t_buf.size().saturating_sub(t_local));

    // Index window: cover [idx_byte_off, + n_idx*4).
    let i_bytes = ((n_idx as u64) * 4).max(4);
    let (i_buf, i_raw) = arena.resolve_w(idx_byte_off);
    let i_base = (i_raw as u64 / ALIGN) * ALIGN;
    let i_local = i_base;
    let i_size = ((i_raw as u64 + i_bytes - i_base).div_ceil(16) * 16)
        .min(i_buf.size().saturating_sub(i_local));

    assert!(
        t_size <= max_bind && i_size <= max_bind,
        "rlx-wgpu gather_split: window too large (table={t_size}, idx={i_size}, max={max_bind})"
    );

    // Separate output buffer (rw) — copied into the arena after the dispatch.
    let out_bytes = ((n_out as u64) * 4).max(4);
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gather_split out"),
        size: out_bytes.div_ceil(16) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let p = GatherParams {
        n_out,
        n_idx,
        dim,
        vocab,
        in_off: ((t_raw as u64 - t_base) / 4) as u32,
        idx_off: ((i_raw as u64 - i_base) / 4) as u32,
        out_off: 0,
        idx_bias,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gather_split uniform"),
        size: std::mem::size_of::<GatherParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));

    let gk = gather_split_kernel(device);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gather_split bg"),
        layout: &gk.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: t_buf,
                    offset: t_local,
                    size: wgpu::BufferSize::new(t_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: i_buf,
                    offset: i_local,
                    size: wgpu::BufferSize::new(i_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gather_split"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gather_split pass"),
            ..Default::default()
        });
        pass.set_pipeline(&gk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(n_out, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    // Copy the embedding back into the arena (distinct buffers → legal).
    let (dst, dst_off) = arena.resolve_w(out_byte_off);
    enc.copy_buffer_to_buffer(&out_buf, 0, dst, dst_off as u64, out_bytes);
    crate::backend::note_device_submit();
    queue.submit(std::iter::once(enc.finish()));
}

fn arena_tensor_in_window(arena: &Arena, id: NodeId, base: u64, size: u64) -> bool {
    let src_tagged = arena.offset(id);
    // Weight-buffer params are never inside the act bind window; callers stage
    // them (or bind the weight buffer separately for packed GGUF).
    if crate::buffer::is_weight_off(src_tagged) {
        return false;
    }
    let src = src_tagged as u64;
    let len = arena.len_of(id) as u64;
    src >= base && src.saturating_add(len) <= base.saturating_add(size)
}

/// True when two planned arena slots share any byte (memory planner reuse).
fn arena_tensors_overlap(arena: &Arena, a: NodeId, b: NodeId) -> bool {
    if a == b {
        return true;
    }
    let (a0, al) = (arena.offset(a) as u64, arena.len_of(a) as u64);
    let (b0, bl) = (arena.offset(b) as u64, arena.len_of(b) as u64);
    if al == 0 || bl == 0 {
        return false;
    }
    let a1 = a0.saturating_add(al);
    let b1 = b0.saturating_add(bl);
    a0 < b1 && b0 < a1
}

/// Arena bind window for matmul: when the weight alone fits the bind limit but
/// activations + weight do not, anchor on the param tensor (e.g. tied `LmHead`).
fn arena_matmul_bind_window(
    device: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    out_id: NodeId,
    a_id: NodeId,
    b_id: NodeId,
    b_in_arena: bool,
) -> (u64, u64, bool) {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if let Some((base, size)) = arena_whole_arena_bind(arena, max_binding) {
        return (base, size, false);
    }
    if !b_in_arena {
        // B is read from the separate f16 shadow buffer (F16 / Coop16 /
        // CoopF16Vk), so the arena binding only carries the activation A and
        // the output C. Anchor on [out, a] — NOT on B. Anchoring on a >4 GiB
        // param B (e.g. the tied F32 lm_head/embedding) would push the window
        // onto the weight and leave C outside it, dropping the output write
        // (logits stayed mostly zero → argmax = STOP).
        let (base, size) = arena_window_for_nodes(device, arena, &[out_id, a_id]);
        return (base, size, false);
    }
    let ids = [out_id, a_id, b_id];
    let all_fits =
        arena_span_bytes(arena, &ids) <= max_binding && !arena_span_crosses_shard(arena, &ids);
    let b_bytes = arena.len_of(b_id) as u64;
    let b_is_param = tensor_is_graph_param(graph, param_offsets, b_id);
    // Param-anchor is unsafe on sharded arenas: the weight stripe rarely holds
    // the matmul output, so C writes land at bogus local offsets.
    let param_anchor = !arena.is_sharded()
        && b_is_param
        && b_bytes <= max_binding
        && (!all_fits || b_bytes > ARENA_STAGE_CAP);
    let (mut base, mut size) = if param_anchor {
        arena_window_for_nodes(device, arena, &[b_id])
    } else if all_fits {
        arena_window_for_nodes(device, arena, &ids)
    } else if arena_span_bytes(arena, &[out_id, b_id]) <= max_binding
        && !arena_span_crosses_shard(arena, &[out_id, b_id])
    {
        // Prefer covering activation+weight over staging a huge B (F5 packs a
        // 0.5 GiB Concat weight at off=0; staging it into a far output window
        // exceeds ARENA_STAGE_CAP).
        arena_window_for_nodes(device, arena, &[out_id, b_id])
    } else if arena_span_bytes(arena, &[out_id, a_id]) <= max_binding
        && !arena_span_crosses_shard(arena, &[out_id, a_id])
    {
        arena_window_for_nodes(device, arena, &[out_id, a_id])
    } else {
        // Prefer the matmul output stripe on sharded arenas.
        arena_window_for_nodes(device, arena, &[out_id])
    };
    let param_anchor = param_anchor
        || (!arena.is_sharded()
            && b_is_param
            && b_bytes <= max_binding
            && !arena_tensor_in_window(arena, b_id, base, size));
    if param_anchor && !arena_tensor_in_window(arena, b_id, base, size) {
        (base, size) = arena_window_for_nodes(device, arena, &[b_id]);
    }
    (base, size, param_anchor)
}

/// Grow `[base, base+size)` to cover all listed tensors when the span still
/// fits `max_storage_buffer_binding_size` (avoids spurious staging copies).
fn arena_expand_bind_window(
    arena: &Arena,
    ids: &[NodeId],
    base: &mut u64,
    size: &mut u64,
    max_binding: u64,
) {
    const ALIGN: u64 = 256;
    let mut lo = *base;
    let mut hi = base.saturating_add(*size);
    for &id in ids {
        let off = arena.offset(id);
        if crate::buffer::is_weight_off(off) {
            continue;
        }
        let o = off as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(o);
        hi = hi.max(o.saturating_add(len));
    }
    let span = hi.saturating_sub(lo).max(1);
    if span > max_binding {
        return;
    }
    if arena.is_sharded() {
        let s = arena.shard_size as u64;
        if lo / s != (hi.saturating_sub(1)) / s {
            return; // would cross shard — keep existing window
        }
        // Keep the whole stripe bound so the per-shard stage reserve stays
        // addressable. Tight sub-windows forced huge BufferCopies / panics and
        // left staged bytes outside the bind range.
        let shard_base = (lo / s) * s;
        *base = shard_base;
        *size = s.min(max_binding).max(256);
        return;
    }
    *base = (lo / ALIGN) * ALIGN;
    *size = span.div_ceil(ALIGN) * ALIGN;
    *size = (*size).max(256).min(max_binding);
    if (*base).saturating_add(*size) > arena.size as u64 {
        *base = (arena.size as u64).saturating_sub(*size);
        *base = (*base / ALIGN) * ALIGN;
    }
}

fn arena_off_in_bind_window(
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    device: &wgpu::Device,
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    id: NodeId,
    base: &mut u64,
    size: &mut u64,
) -> u32 {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if let Some((b, s)) = arena_whole_arena_bind(arena, max_binding) {
        *base = b;
        *size = s;
        // …but only activations live in that window. A param parked in the
        // separate weight buffer has to be staged in even when the whole act
        // arena binds — which is precisely the case this used to miss, because
        // "arena fits one binding" and "params are in another buffer" are
        // independent conditions. `from_plan_split` parks every param once
        // act + params + scratch exceed the bind cap, so a model whose
        // activations alone stay small (Qwen3-0.6B F32) hit the shortcut for
        // every weight it read.
        if !crate::buffer::is_weight_off(arena.offset(id)) {
            return arena_local_off_f32(arena, id, b);
        }
    }
    if arena_tensor_in_window(arena, id, *base, *size) {
        arena_local_off_f32(arena, id, *base)
    } else {
        let len = arena.len_of(id) as u64;
        if tensor_is_graph_param(graph, param_offsets, id) && len > max_binding {
            panic!(
                "rlx-wgpu: param node {:?} ({} bytes) exceeds max_storage_buffer_binding_size \
                 ({max_binding}); split weights or use f16 shadow binds",
                id, len
            );
        }
        if len > ARENA_STAGE_CAP {
            let op = &graph.node(id).op;
            panic!(
                "rlx-wgpu: bind_window would stage {} bytes for {:?} op={op:?} \
                 (off={}, base={}, bind_size={})",
                len,
                id,
                arena.offset(id),
                *base,
                *size,
            );
        }
        arena_off_in_window_or_stage(arena, schedule, scratch, base, size, max_binding, id)
    }
}

/// Bind window for ops that read/write multiple arena tensors (conv, concat, …).
/// Returns `(base, size)` and rebased f32 offsets; stages operands that fall outside
/// the window when the full span exceeds `max_storage_buffer_binding_size`.
fn arena_multi_op_window(
    dev: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    _schedule: &mut Vec<Step>,
    scratch: &mut u64,
    ids: &[NodeId],
) -> (u64, u64, bool) {
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    if let Some((base, size)) = arena_whole_arena_bind(arena, max_binding) {
        *scratch = arena.scratch_off as u64;
        return (base, size, false);
    }
    let param_anchor = if arena.is_sharded() {
        // On striped arenas, never anchor on a far param — that leaves the
        // op output outside the bound shard and `arena_local_off_f32` writes
        // corrupt the wrong stripe. Stage params into the output shard instead.
        None
    } else if arena_span_bytes(arena, ids) > max_binding || arena_span_crosses_shard(arena, ids) {
        ids.iter()
            .find(|&&id| {
                let nbytes = arena.len_of(id) as u64;
                tensor_is_graph_param(graph, param_offsets, id) && nbytes <= max_binding
            })
            .copied()
    } else {
        None
    };
    let mut param_anchored = param_anchor.is_some();
    let (mut base, mut size) =
        if arena_span_bytes(arena, ids) <= max_binding && !arena_span_crosses_shard(arena, ids) {
            arena_window_for_nodes(dev, arena, ids)
        } else if let Some(id) = param_anchor {
            arena_window_for_nodes(dev, arena, &[id])
        } else {
            // Span exceeds one storage bind (unsharded >4 GiB arenas) or crosses
            // shards: anchor on the op output (first non-param) and stage far
            // inputs into that window.
            let outish = ids
                .iter()
                .copied()
                .find(|&id| !tensor_is_graph_param(graph, param_offsets, id))
                .unwrap_or(ids[0]);
            arena_window_for_nodes(dev, arena, &[outish])
        };
    if let Some(id) = param_anchor {
        if !arena_tensor_in_window(arena, id, base, size) {
            (base, size) = arena_window_for_nodes(dev, arena, &[id]);
        }
        param_anchored = true;
    } else if !arena.is_sharded() {
        for &id in ids {
            let nbytes = arena.len_of(id) as u64;
            if tensor_is_graph_param(graph, param_offsets, id)
                && nbytes <= max_binding
                && !arena_tensor_in_window(arena, id, base, size)
            {
                (base, size) = arena_window_for_nodes(dev, arena, &[id]);
                param_anchored = true;
                break;
            }
        }
    }
    *scratch = arena.scratch_off as u64;
    // Staging into scratch must land inside the bound window.
    if param_anchored
        || arena_span_crosses_shard(arena, ids)
        || arena_span_bytes(arena, ids) > max_binding
    {
        arena_ensure_scratch_for_window(arena, scratch, base, size);
    }
    (base, size, param_anchored)
}

fn arena_bind_window_covering_scratch_if_needed(
    arena: &Arena,
    base: u64,
    size: u64,
    scratch: u64,
) -> u64 {
    // Planner places scratch at the arena tail; do not relocate the bind
    // window until this op has actually started staging into scratch.
    if scratch <= arena.scratch_off as u64 {
        return base;
    }
    if scratch >= base && scratch.saturating_add(ARENA_STAGE_CAP) <= base.saturating_add(size) {
        return base;
    }
    arena_window_covering_scratch(arena, base, size)
}

/// Keep staging writes inside `[base, base+size)` when the bind window is anchored on a
/// param far from the arena tail scratch zone.
fn arena_ensure_scratch_in_window(scratch: &mut u64, base: u64, size: u64) {
    let cap = ARENA_STAGE_CAP.min(size);
    let end = base.saturating_add(size);
    if *scratch < base || scratch.saturating_add(cap) > end {
        *scratch = end.saturating_sub(cap);
        *scratch = (*scratch / 256) * 256;
    }
}

/// Prefer the dedicated stage reserve (never overlaps live tensors) when the
/// arena is striped **or** unsharded-but-oversized with a compile-time scratch
/// tail. Falling back to “scratch at bind-window end” on a single >4 GiB
/// buffer parked staging on top of F5 params at the arena tail and zeroed the
/// whole DiT (`RLX_WGPU_LARGE_BUFFERS=1`).
fn arena_ensure_scratch_for_window(arena: &Arena, scratch: &mut u64, base: u64, size: u64) {
    if arena.is_sharded() {
        let stage = arena.shard_stage_off(base as usize) as u64;
        let reserve = crate::buffer::shard_stage_reserve() as u64;
        let stage_end = stage.saturating_add(reserve);
        // Keep an already-bumping cursor inside this stripe's reserve. Resetting
        // to `stage` on every call made the second staged tensor (e.g. LN beta)
        // overwrite the first (gamma) — F5 FusedResidualLN then read gamma=0
        // and wrote an all-zero normalized row.
        if *scratch >= stage && *scratch < stage_end {
            return;
        }
        if stage >= base
            && stage.saturating_add(ARENA_STAGE_CAP.min(size)) <= base.saturating_add(size)
        {
            *scratch = stage;
            return;
        }
        // Window doesn't cover the reserve (sub-window bind) — slide scratch to
        // the reserve and rely on callers binding the whole shard when staging.
        *scratch = stage;
        return;
    }
    if arena.scratch_bytes > 0 {
        let stage = arena.scratch_off as u64;
        let reserve = arena.scratch_bytes as u64;
        let stage_end = stage.saturating_add(reserve);
        if *scratch >= stage && *scratch < stage_end {
            return;
        }
        if stage >= base && stage < base.saturating_add(size) {
            *scratch = stage;
            return;
        }
        // Point at the safe tail even when the current window doesn't cover it;
        // `arena_off_in_window_or_stage` widens the bind to include `dst`.
        let _ = stage_end;
        *scratch = stage;
        return;
    }
    arena_ensure_scratch_in_window(scratch, base, size);
}

#[allow(dead_code)]
fn arena_off_for_window(
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    id: NodeId,
    _window_ids: &[NodeId],
    mut base: u64,
    mut size: u64,
    max_binding: u64,
    _fits_in_one_binding: bool,
) -> u32 {
    let src = arena.offset(id) as u64;
    let len = arena.len_of(id) as u64;
    if src >= base && src.saturating_add(len) <= base.saturating_add(size) {
        arena_local_off_f32(arena, id, base)
    } else {
        arena_off_in_window_or_stage(
            arena,
            schedule,
            scratch,
            &mut base,
            &mut size,
            max_binding,
            id,
        )
    }
}

/// f16 shadow buffer window matching an f32 arena bind `[arena_base, arena_base+arena_size)`.
fn f16_shadow_bind_range(arena_base: u64, arena_size: u64, f16_buf_bytes: u64) -> (u64, u64) {
    const ALIGN: u64 = 256;
    let mut base = (arena_base / 2 / ALIGN) * ALIGN;
    let mut size = (arena_size / 2).div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(f16_buf_bytes);
    if base.saturating_add(size) > f16_buf_bytes {
        base = f16_buf_bytes.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    (base, size)
}

/// Window into `f16_buffer` for matmul weight reads (`params.b_off` is in
/// f16-element indices, matching the f32 arena word index).
fn f16_weight_bind_range(
    dev: &wgpu::Device,
    f16_buf_bytes: u64,
    b_off: u32,
    k: u32,
    n: u32,
    batch: u32,
    b_batch_stride: u32,
) -> (u64, u64, u32) {
    const ALIGN: u64 = 256;
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    let b0 = b_off as u64;
    let span = (k as u64).saturating_mul(n as u64);
    let batch_n = batch.max(1) as u64;
    let stride = if batch_n > 1 {
        b_batch_stride as u64
    } else {
        span
    };
    let hi_elems = b0
        .saturating_add((batch_n - 1).saturating_mul(stride))
        .saturating_add(span);
    let lo_byte = b0.saturating_mul(2);
    let hi_byte = hi_elems.saturating_mul(2).saturating_add(8);
    let need = hi_byte.saturating_sub(lo_byte).max(1);
    if need > max_binding {
        panic!(
            "rlx-wgpu: f16 weight region needs {need} bytes (> {max_binding}); \
             matmul k={k} n={n} batch={batch}"
        );
    }
    let mut base = (lo_byte / ALIGN) * ALIGN;
    // The window has to cover `[base, hi_byte)`. Sizing it from `lo_byte` (or from
    // `need`, which is measured from there) under-counts by the alignment slack —
    // `base` is rounded DOWN and can sit up to `ALIGN - 1` bytes below it — so the
    // window ended a few elements short of the region it describes whenever
    // `b_off` was not already 256-aligned. A cooperative load of an element past
    // the binding is not a fault: WebGPU returns zero, so the last rows of a
    // weight silently became zeros and the product was off by ~1e-3 relative. That
    // is what `tests/matmul_f16_f32acc_parity.rs` caught at `n = 64` (a weight
    // window that landed on an unaligned offset); every shape whose `b_off` is
    // already aligned was exact, which is why the fault looked shape-specific.
    let mut size = hi_byte.saturating_sub(base).div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(max_binding).min(f16_buf_bytes);
    if base.saturating_add(size) < hi_byte {
        base = hi_byte.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    if base.saturating_add(size) > f16_buf_bytes {
        base = f16_buf_bytes.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    let rebased = b_off.saturating_sub((base / 2) as u32);
    (base, size, rebased)
}

const ARENA_STAGE_CAP: u64 = crate::buffer::SHARD_STAGE_RESERVE as u64;

/// Byte ceiling the staging path applies to an operand of `id` that it has to copy
/// into the bind window — because the operand is parked in the weight buffer or
/// lies outside the window: the shard reserve on a striped arena, the arena's own
/// scratch tail for a parked parameter, the dedicated reserve otherwise.
///
/// [`arena_off_in_window_or_stage`] refuses above this. A windowed step that has a
/// host route asks here *before* it, so its refusal can name that route — the
/// shared panic can only name the cap.
///
/// This is a **parallel derivation** of the thresholds that function applies at its
/// own panic sites, kept beside `ARENA_STAGE_CAP` so the two read together; it is
/// not the same expression, so any change to those thresholds has to be mirrored
/// here. A divergence is a wrong decision in either direction, not just a worse
/// message: a ceiling below what that function allows refuses a geometry the shared
/// path would have served, and one above it lets the operand through to the shared
/// panic, which cannot name this caller's route.
fn arena_stage_ceiling(arena: &Arena, id: NodeId, base: u64) -> u64 {
    let parked = crate::buffer::is_weight_off(arena.offset(id));
    if arena.is_sharded() {
        let s = arena.shard_size as u64;
        let src = arena.offset(id) as u64;
        let len = arena.len_of(id) as u64;
        // An activation inside the window's own stripe is served by opening the
        // whole stripe instead of staging (a parked operand never is — it lives in
        // the weight buffer), and `arena_off_in_window_or_stage` checks the
        // dedicated reserve before it gets that far, so that is all it is bounded
        // by. Without this the asker would refuse a stripe the shared path serves.
        if !parked
            && s > 0
            && src / s == base / s
            && src.saturating_add(len).saturating_sub(1) / s == base / s
        {
            return ARENA_STAGE_CAP;
        }
        return crate::buffer::shard_stage_reserve() as u64;
    }
    // A parked parameter is judged against the dedicated reserve *and* the arena's
    // own scratch tail, whichever is smaller.
    let cap = if parked {
        arena.scratch_bytes as u64
    } else {
        ARENA_STAGE_CAP
    };
    cap.min(ARENA_STAGE_CAP)
}

/// Output spatial positions computed per thread by `conv2d.wgsl` (register
/// tiling for weight reuse). MUST equal `TILE` in that kernel.
const CONV2D_TILE: u32 = 4;

/// Return a window-local f32 offset, staging into scratch when the tensor lies
/// outside the bind window (by emitting a copy step).
fn arena_off_in_window_or_stage(
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    base: &mut u64,
    size: &mut u64,
    max_binding: u64,
    id: NodeId,
) -> u32 {
    let src_tagged = arena.offset(id);
    let len = arena.len_of(id) as u64;
    if crate::buffer::is_weight_off(src_tagged) {
        // Params live in a separate buffer — always stage into the act window.
        if len > ARENA_STAGE_CAP {
            panic!(
                "rlx-wgpu: cannot stage {} bytes for weight node {:?} (cap {ARENA_STAGE_CAP})",
                len, id
            );
        }
        if arena.is_sharded() {
            let s = arena.shard_size as u64;
            let win_shard = *base / s;
            let stage_cap = crate::buffer::shard_stage_reserve() as u64;
            if len > stage_cap {
                panic!(
                    "rlx-wgpu: cannot stage {} bytes for weight node {:?} \
                     (shard reserve {stage_cap})",
                    len, id
                );
            }
            *base = win_shard * s;
            *size = s.min(max_binding).max(256);
            arena_ensure_scratch_for_window(arena, scratch, *base, *size);
            let stage_begin = arena.shard_stage_off(*base as usize) as u64;
            let stage_end = stage_begin.saturating_add(stage_cap);
            let aligned = len.div_ceil(256) * 256;
            // Bump-allocate inside the reserve; wrap if this op's stages fill it.
            if scratch.saturating_add(aligned) > stage_end {
                *scratch = stage_begin;
            }
            let dst = *scratch;
            *scratch = scratch.saturating_add(aligned);
            schedule.push(Step::BufferCopy {
                src_byte_off: src_tagged as u64,
                dst_byte_off: dst,
                bytes: len as u32,
            });
            return ((dst.saturating_sub(*base)) / 4) as u32;
        }
        // Unsharded act arena: bump-allocate inside the dedicated scratch
        // tail and wrap when an op stages more than fits in one pass.
        // Without wrapping, parking many F32 params in the weight buffer
        // walked `scratch` past `arena.size` and matmul/gather read zeros.
        let aligned = len.div_ceil(256) * 256;
        let stage_begin = arena.scratch_off as u64;
        let stage_end = stage_begin.saturating_add(arena.scratch_bytes.max(1) as u64);
        if (arena.scratch_bytes as u64) < len {
            panic!(
                "rlx-wgpu: cannot stage {} bytes for weight node {:?} \
                 (scratch reserve {} bytes)",
                len, id, arena.scratch_bytes
            );
        }
        if *scratch < stage_begin || scratch.saturating_add(aligned) > stage_end {
            *scratch = stage_begin;
        }
        let dst = *scratch;
        *scratch = scratch.saturating_add(aligned);
        schedule.push(Step::BufferCopy {
            src_byte_off: src_tagged as u64,
            dst_byte_off: dst,
            bytes: len as u32,
        });
        // Widen the act bind window so the shader can see the staged copy
        // (same as the non-weight staging path below).
        let lo = (*base).min(dst);
        let hi = (*base)
            .saturating_add(*size)
            .max(dst.saturating_add(aligned));
        let span = hi.saturating_sub(lo).max(1);
        if span <= max_binding {
            const ALIGN: u64 = 256;
            *base = (lo / ALIGN) * ALIGN;
            *size = span.div_ceil(ALIGN) * ALIGN;
            *size = (*size).max(256).min(max_binding);
            if (*base).saturating_add(*size) > arena.size as u64 {
                *base = (arena.size as u64).saturating_sub(*size);
                *base = (*base / ALIGN) * ALIGN;
            }
            arena_clamp_bind_window(arena, base, size);
        }
        return ((dst.saturating_sub(*base)) / 4) as u32;
    }
    let src = src_tagged as u64;
    if src >= *base && src.saturating_add(len) <= (*base).saturating_add(*size) {
        return arena_local_off_f32(arena, id, *base);
    }
    if len > ARENA_STAGE_CAP {
        panic!(
            "rlx-wgpu: cannot stage {} bytes for node {:?} (cap {ARENA_STAGE_CAP})",
            len, id
        );
    }
    if arena.is_sharded() {
        let s = arena.shard_size as u64;
        let win_shard = *base / s;
        let src_shard = src / s;
        let src_end_shard = src.saturating_add(len).saturating_sub(1) / s;
        // Same stripe: open the whole shard instead of copying into the reserve.
        // Requires the WHOLE tensor (not just its start byte) to sit inside
        // `win_shard` — a tensor straddling into the next stripe (e.g. a
        // large per-block param tail-appended near a shard boundary) would
        // otherwise get a local offset whose tail reads past this shard's
        // physical buffer (silently clamped/zeroed by wgpu), corrupting
        // exactly the columns/rows that spilled over — this produced a
        // small-but-correlated wrong result (e.g. AdaLN modulation Gemm)
        // instead of a loud failure.
        if src_shard == win_shard && src_end_shard == win_shard {
            *base = win_shard * s;
            *size = s.min(max_binding).max(256);
            arena_clamp_bind_window(arena, base, size);
            return arena_local_off_f32(arena, id, *base);
        }
        let stage_cap = crate::buffer::shard_stage_reserve() as u64;
        if len > stage_cap {
            // Too large to stage: open a whole-stripe window on the tensor's
            // own shard. Callers that need both this tensor and the previous
            // window's contents must have already packed or used a host path.
            // Prefer keeping the write target (current window) and erroring
            // only when we cannot address the tensor at all.
            panic!(
                "rlx-wgpu: cannot stage {} bytes for node {:?} across shards \
                 (shard reserve {stage_cap}; src_shard={src_shard} win_shard={win_shard}). \
                 Shrink max_seq or raise SHARD_STAGE_RESERVE.",
                len, id
            );
        }
        // Bind the whole destination stripe so the stage reserve is in-range.
        *base = win_shard * s;
        *size = s.min(max_binding).max(256);
        arena_clamp_bind_window(arena, base, size);
        // Keep staging inside this stripe's reserved tail — never expand the
        // bind window across a shard boundary (that used to clobber live slots).
        arena_ensure_scratch_for_window(arena, scratch, *base, *size);
        let stage_cap = crate::buffer::shard_stage_reserve() as u64;
        let stage_begin = arena.shard_stage_off(*base as usize) as u64;
        let stage_end = stage_begin.saturating_add(stage_cap);
        let aligned = len.div_ceil(256) * 256;
        if scratch.saturating_add(aligned) > stage_end {
            *scratch = stage_begin;
        }
        let dst = *scratch;
        *scratch = scratch.saturating_add(aligned);
        schedule.push(Step::BufferCopy {
            src_byte_off: src,
            dst_byte_off: dst,
            bytes: len as u32,
        });
        return ((dst.saturating_sub(*base)) / 4) as u32;
    }
    let aligned = len.div_ceil(256) * 256;
    // Prefer the dedicated tail reserve when present (oversized unsharded
    // arenas). Window-end parking used to clobber live params on F5 DiT.
    arena_ensure_scratch_for_window(arena, scratch, *base, *size);
    if arena.scratch_bytes > 0 {
        let stage_begin = arena.scratch_off as u64;
        let stage_end = stage_begin.saturating_add(arena.scratch_bytes as u64);
        if scratch.saturating_add(aligned) > stage_end {
            *scratch = stage_begin;
        }
    } else if scratch.saturating_add(aligned) > (*base).saturating_add(*size) {
        arena_ensure_scratch_in_window(scratch, *base, *size);
    }
    let dst = *scratch;
    *scratch = scratch.saturating_add(aligned);
    schedule.push(Step::BufferCopy {
        src_byte_off: src,
        dst_byte_off: dst,
        bytes: len as u32,
    });
    let lo = (*base).min(dst);
    let hi = (*base)
        .saturating_add(*size)
        .max(dst.saturating_add(aligned));
    let span = hi.saturating_sub(lo).max(1);
    if span <= max_binding {
        const ALIGN: u64 = 256;
        *base = (lo / ALIGN) * ALIGN;
        *size = span.div_ceil(ALIGN) * ALIGN;
        *size = (*size).max(256).min(max_binding);
        if (*base).saturating_add(*size) > arena.size as u64 {
            *base = (arena.size as u64).saturating_sub(*size);
            *base = (*base / ALIGN) * ALIGN;
        }
        arena_clamp_bind_window(arena, base, size);
    }
    if arena_tensor_in_window(arena, id, *base, *size) {
        arena_local_off_f32(arena, id, *base)
    } else {
        ((dst.saturating_sub(*base)) / 4) as u32
    }
}

/// If scratch does not fall inside `[base, base+size)`, slide the window to the tail.
fn arena_window_covering_scratch(arena: &Arena, base: u64, size: u64) -> u64 {
    let scratch = arena.scratch_off as u64;
    if scratch >= base && scratch.saturating_add(ARENA_STAGE_CAP) <= base.saturating_add(size) {
        return base;
    }
    if arena.is_sharded() {
        // Sliding to the logical arena tail would move the bind onto a
        // different stripe, invalidating every previously rebased offset
        // (Attention/QKV, norms, …) and writing outputs into the wrong
        // shard. Keep the current stripe; staging uses `shard_stage_off`.
        return base;
    }
    let new_base = (arena.size as u64).saturating_sub(size);
    (new_base / 256) * 256
}

fn arena_span_bytes(arena: &Arena, ids: &[NodeId]) -> u64 {
    let mut lo: u64 = u64::MAX;
    let mut hi: u64 = 0;
    for &id in ids {
        // Zero-byte tensors (a zero-length dimension, e.g. LuxTTS's `Expand [0,1,512]`) carry
        // an explicit empty slot at offset 0 — see the plan fixup in `compile_static_inner`.
        // They occupy nothing, so they must not anchor the window: including offset 0 here
        // would stretch it back to the head of the arena.
        if !arena.has(id) || arena.len_of(id) == 0 {
            continue;
        }
        let off = arena.offset(id);
        if crate::buffer::is_weight_off(off) {
            // Weight-buffer tensors don't enlarge the activation bind span;
            // callers stage them into the act window separately.
            continue;
        }
        let o = off as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(o);
        hi = hi.max(o.saturating_add(len));
    }
    if lo == u64::MAX {
        0
    } else {
        hi.saturating_sub(lo)
    }
}

#[allow(dead_code)]
fn bind_two(
    device: &wgpu::Device,
    kernel: &Kernel,
    buf0: &wgpu::Buffer,
    buf1: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if buf0.size() > max_binding {
        panic!(
            "rlx-wgpu: bind_two buffer {} bytes exceeds max_storage_buffer_binding_size {}; \
             use bind_two_buf0_window or bind_op_output_window",
            buf0.size(),
            max_binding
        );
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf1.as_entire_binding(),
            },
        ],
    })
}

/// Windowed arena bind around the op's output. The window base is **not** returned,
/// so no caller can rebase its offsets against it: every caller builds uniforms from
/// absolute `arena.offset(id)` and is correct only while the whole arena is bound at
/// base zero — and only for operands that live in this arena at all, since
/// `arena_window_for_nodes` drops weight-tagged operands from the window. This is the
/// net under `lower::should_host_at_ceiling`: an absolute-offset arm that is not
/// classified there (or whose guard was dropped) is refused loudly by `bind_op_window`
/// rather than producing a plausible wrong answer. A caller that genuinely needs a
/// re-based window uses `arena_window_for_nodes` + `bind_arena_window`, which hand it
/// the base.
fn bind_op_output_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    out_id: NodeId,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    bind_op_window(device, kernel, arena, &[out_id], params)
}

fn bind_op_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    ids: &[NodeId],
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    let (base, size) = if arena_span_bytes(arena, ids) <= max_binding {
        arena_window_for_nodes(device, arena, ids)
    } else {
        arena_window_for_nodes(device, arena, &[ids[0]])
    };
    // This path derives the window from the op's own nodes and does **not**
    // return its base, so no caller can rebase offsets against it: every caller
    // builds uniforms from absolute `arena.offset(id)`. The net under the routing
    // rule (see `bind_op_output_window`). It asks the *whole* coarse condition,
    // not just `base == 0`: a window can start at zero and still cover only one
    // stripe, or only this op's own span, and an unlisted absolute-offset arm
    // reading a higher offset would then be neither refused nor correct. When the
    // condition holds, `arena_window_for_nodes` returns the whole arena (its
    // `arena_whole_arena_bind` branch ignores the ids), so every arm that
    // legitimately binds here already satisfies it — including the guarded ones,
    // which host whenever it is false. WGSL makes an out-of-range access a dynamic
    // error whose permitted outcomes (return 0 or another in-bounds value, drop the
    // store, discard the dispatch) never include telling the caller, so there is no
    // version of this that fails safe on its own.
    assert!(
        base == 0 && arena.binds_whole_at_base_zero(max_binding),
        "rlx-wgpu: REFUSED (arena not bindable whole): the window for {ids:?} starts at {base} \
         (size {size}) of an arena of {} bytes, but this bind path cannot rebase a kernel's \
         ABSOLUTE offsets — only the whole arena bound at base zero is correct. Classify the op \
         that owns {ids:?} in `supported_ops::addresses_arena_by_absolute_offset` so \
         `lower::should_host_at_ceiling` routes it, or bind through the base-returning window \
         helpers.",
        arena.size,
    );
    bind_arena_window(device, kernel, arena, base, size, params)
}

/// Storage-buffer binding size for an arena window. wgpu 30 validates that
/// storage-buffer binding sizes are a multiple of 4; arena windows can end on
/// a non-4 byte (u8-packed GGUF / f16 buffers), so round up — clamped to the
/// buffer's tail so the binding never runs past the buffer end.
pub(crate) fn aligned_bind_size(size: u64, base: u64, buffer_size: u64) -> Option<NonZeroU64> {
    // Round the window up to a multiple of 4, clamped to the buffer's 4-aligned
    // capacity (the arena buffer may itself end on a non-4 byte, so `& !3`).
    let cap = (buffer_size & !3).saturating_sub(base);
    NonZeroU64::new(size.next_multiple_of(4).min(cap))
}

fn bind_two_buf0_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    buf0: &wgpu::Buffer,
    buf0_base: u64,
    buf0_size: u64,
    buf1: &wgpu::Buffer,
    parts: Option<&wgpu::Buffer>,
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf0,
                offset: buf0_base,
                size: aligned_bind_size(buf0_size, buf0_base, buf0.size()),
            }),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: buf1.as_entire_binding(),
        },
    ];
    if let Some(parts) = parts {
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: parts.as_entire_binding(),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu bg window"),
        layout: &kernel.bgl,
        entries: &entries,
    })
}

/// [`bind_arena_window`] with the attention kernels' binding 2 bound too: the
/// key-range split's partials, which the single-workgroup form never writes.
fn bind_arena_window_parts(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    mut rebase: u64,
    mut size: u64,
    params: &wgpu::Buffer,
    parts: &wgpu::Buffer,
) -> wgpu::BindGroup {
    arena_clamp_bind_window(arena, &mut rebase, &mut size);
    let (buf, local) = arena_bind_buf(arena, rebase);
    bind_two_buf0_window(device, kernel, buf, local, size, params, Some(parts))
}

/// Compute precision selector: derive from IR dtypes of A and B and
/// the device features.
///
/// Priority (the cascade at the end of this function, in order):
///   1. Coop16 — if EXPERIMENTAL_COOPERATIVE_MATRIX + SHADER_F16 +
///      F16 IR tag + b traces to a Param + M/K/N are 32/8/32 aligned.
///      Unlocks Apple's `simdgroup_matrix` / Vulkan's KHR_cooperative
///      hardware GEMM units (~18× faster than f32 ALU on Apple M-series).
///   2. CoopF16Vk — the 16-bit cooperative path on a discrete Vulkan
///      adapter: opt-in (`RLX_WGPU_COOP_F16_VK_ENABLE=1`), gated by the
///      runtime operand-order probe, and off for large N unless
///      `RLX_WGPU_COOP_F16_VK_LARGE_N=1`. See `coop_f16_vk_eligible`.
///   3. CoopF32 — `simdgroup_float8x8` on Metal, the portable 8×8 form on
///      Vulkan, when cooperative matrices are available, B is a Param and
///      the shapes align. A shape outside this cascade that keeps the weight in
///      the first operand — the two convolution GEMM routes — reads the same
///      admission with the roles swapped; see [`conv_gemm`].
///   4. F32 — every other case, *including* when AutoMixedPrecision
///      tagged the matmul as F16 but it failed Coop16's alignment
///      check. The non-coop F16 path (`matmul_f16_compute.wgsl`) was
///      empirically measured 4-5× SLOWER than the f32 baseline on
///      Apple via wgpu/naga 29 — the WGSL→MSL emit doesn't unlock
///      Apple's f16 ALU through portable WGSL ALU. So at small /
///      unaligned shapes we lose nothing by ignoring the IR's f16
///      tag and using f32 — precision improves AND speed wins.
///
/// (The F16 variant of `MatmulCompute` and `matmul_f16_compute.wgsl`
/// remain for future use — e.g. when naga gains a portable subgroup-
/// matrix surface that lowers efficiently without needing the full
/// coop-matrix dance, or when bf16 hardware lands. Today no path
/// dispatches them.)
///
/// Three layers, in increasing authority: this cascade (the compile-time
/// default), then a **measured override** from the shared dispatch table, then
/// the `RLX_WGPU_*` env pins. The override is accepted only for a path this
/// same function judged *eligible* — see [`derive_matmul_compute`]. Almost
/// every condition below is eligibility rather than preference (dtype tags,
/// device features, alignment the kernel assumes, and two known-bad-kernel
/// guards), which is exactly why the table may not reach into them.
fn derive_matmul_compute_default(
    dev: &wgpu::Device,
    graph: &Graph,
    mirror_acts: &HashSet<NodeId>,
    a_id: NodeId,
    b_id: NodeId,
    m: u32,
    k: u32,
    n: u32,
) -> (MatmulCompute, MatmulEligibility) {
    if rlx_ir::env::flag("RLX_WGPU_MATMUL_F32_ONLY") {
        // An explicit pin: F32 is the only admissible path, so it is also the
        // only eligible one — the table must not be able to route around it.
        return (MatmulCompute::F32, MatmulEligibility::only_f32());
    }
    use rlx_ir::DType;
    let a_dt = graph.node(a_id).shape.dtype();
    let b_dt = graph.node(b_id).shape.dtype();
    let any_low =
        matches!(a_dt, DType::F16 | DType::BF16) || matches!(b_dt, DType::F16 | DType::BF16);
    // A genuine BF16 weight (e.g. a bf16-resident LM head fed via
    // `set_param_typed(name, bytes, DType::BF16)`) is already widened to
    // f32 in the f32-uniform arena — the plain F32 kernel reads correct
    // values and is bit-exact. It must NOT be routed to the f16-shadow
    // coop path: (a) the f16 shadow downcast (f32→f16) discards bf16's
    // wider exponent range, and (b) the `matmul_coop16` kernel produced
    // orthogonal garbage on Metal for aligned bf16-tagged weights. Keep
    // bf16 in `any_low` (so it never falls into the `!any_low`
    // CoopF16Vk branch) but exclude it from Coop16 below → it lands on
    // F32 (or CoopF32, which reads the correct f32 arena weight).
    let any_bf16 = matches!(a_dt, DType::BF16) || matches!(b_dt, DType::BF16);
    // CoopF32 (`simdgroup_float8x8`) needs K and N aligned to 8 and 32
    // (one micro-tile per K-iter, one 32-col workgroup per N-tile).
    // M can be arbitrary — the kernel pads to the next multiple of 32
    // and bounds-checks the output writes so out-of-range rows stay
    // untouched. (The Coop16 / matmul_qkv paths still require m%32==0;
    // their kernels don't have the same bounds check.)
    //
    // Vulkan uses `matmul_coop_f32_portable` (8×8 tiles, coopLoad) which
    // only requires k%8 and n%8.
    //
    // The two convolution GEMM routes ([`conv_gemm`]) ask the same kernel
    // with that looser alignment on Metal too, and for their own reason:
    // their `n` is a conv's spatial extent, whose padded columns the kernel
    // drops where it drops the padded rows.
    let coop16_aligned = m.is_multiple_of(32) && k.is_multiple_of(8) && n.is_multiple_of(32);
    let has_coop = dev
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let b_is_param = traces_to_param(graph, b_id);

    // ── Eligibility, computed ONCE ──────────────────────────────────────────
    //
    // Nearly every condition here is a correctness gate rather than a
    // preference — dtype tags, device features, the alignment each kernel
    // assumes, and two known-bad-kernel guards. Computing them once and having
    // both the cascade below and the dispatch-table override read the SAME
    // values is the point: restating a predicate for the override check is how
    // a tuning table starts admitting a kernel that produces garbage.

    // Coop16 has an f16 accumulator (Naga 29 can't compile the mixed
    // f32-acc / f16-operand form). Sums of 3072 BERT-FFN activations
    // overflow f16, so we only enter on F16/BF16 IR tags — AutoMixed
    // users have already opted into the precision tradeoff.
    let coop16_ok = any_low
        && !any_bf16
        && has_coop
        && dev.features().contains(wgpu::Features::SHADER_F16)
        && b_is_param
        && coop16_aligned
        && crate::coop_probe::coop16_path_ok();

    // CoopF16F32Acc — `matmul_coop_f16_f32acc`, the 64×64 form with an f32
    // accumulator. Same IR term as Coop16 (a 16-bit-tagged weight, never a bf16
    // one, which must stay bit-exact and is read from the f32 arena), and it
    // outranks Coop16 wherever both are eligible because it costs nothing to: the
    // two forms sit within 1–8% of each other per shape on this rig (2.24–2.56
    // against 2.26–2.61 TFLOP/s at the embedding encoder's m = 8192 geometries, one
    // dispatch with no readback, `tests/encoder_gemm_cost.rs`), so the f32
    // accumulator — which keeps the deep-FFN sum that overflows Coop16's f16
    // accumulator out of range — is a precision gain at the same rate.
    //
    // The geometry term is the shader's, and it is a refusal rather than a
    // rounding: a cooperative store cannot be masked, so the output tile is
    // written whole — `m` and `n` must be whole tiles (64) and `k` whole K stages
    // (32), which is exactly [`coop_f16_f32acc_admits`]. A multiply off that grid
    // keeps the kernel this cascade would have given it, and never takes this
    // arm. Every projection of the text tower is whole on all three axes; a tail
    // falls back rather than being padded, fused or written through a scalar
    // path. What else it needs is a device that can build it (the 8.25 KiB of
    // workgroup storage is the term that can go missing) and the operand-order
    // probe's verdict, which is the only thing standing between this shader and a
    // wrong product.
    let coop_f16_f32acc_aligned = coop_f16_f32acc_admits(m as u64, n as u64, k as u64);
    let coop_f16_f32acc_ok = any_low
        && !any_bf16
        && b_is_param
        && coop_f16_f32acc_aligned
        && crate::kernels::matmul_coop_f16_f32acc_kernel(dev).is_some()
        && crate::coop_probe::f16_f32acc_path_ok();

    let coop_f16_vk_ok = !any_low
        && coop_f16_vk_eligible(dev, m, k, n)
        && b_is_param
        && !mirror_acts.contains(&a_id)
        && !mirror_acts.contains(&b_id)
        && crate::coop_probe::f16_vk_path_ok(n);

    // CoopF32 for a multiply whose weight is operand B: `b_is_param` is the
    // cascade's term for that role — it keeps the f16-shadow, packed-bf16 and
    // cooperative-f16 families, each of which reads a weight-side buffer the B
    // role builds, in front of the f32 cooperative kernel. A BF16 Param rhs
    // keeps the PACKED path (`matmul_bf16w`), which reads 2 B per element
    // instead of 4 and is installed by the caller only when this function
    // answers `F32`; letting default-on CoopF32 outrank it silently disabled
    // the packed path for every aligned bf16 shape — correct numbers, but
    // twice the weight bandwidth, and
    // `rlx-runtime/tests/wgpu_bf16_weight_matmul.rs` asserts the packed
    // dispatch actually happens. `RLX_WGPU_FORCE_COOP_F32` still overrides.
    //
    // The two convolution GEMM routes read the same admission with the weight
    // in the FIRST operand, no packed form to keep, and the portable
    // output-column alignment rather than the Metal form's own; see
    // [`conv_gemm`].
    let coop_f32_ok =
        b_is_param && coop_f32_admitted(dev, k, n, any_bf16 && b_is_param, CoopF32Alignment::Metal);

    let eligible = MatmulEligibility {
        coop_f16_f32acc: coop_f16_f32acc_ok,
        coop16: coop16_ok,
        coop_f16_vk: coop_f16_vk_ok,
        coop_f32: coop_f32_ok,
        // The plain tiled f32 kernel has no alignment or feature requirement —
        // it is the path everything else falls back to, so it is always legal.
        f32: true,
    };

    // ── The cascade: preference order among the eligible ────────────────────
    let default = if coop_f16_f32acc_ok {
        MatmulCompute::CoopF16F32Acc
    } else if coop16_ok {
        MatmulCompute::Coop16
    } else if coop_f16_vk_ok {
        MatmulCompute::CoopF16Vk
    } else if coop_f32_ok {
        MatmulCompute::CoopF32
    } else {
        MatmulCompute::F32
    };
    (default, eligible)
}

/// Which output-column alignment a caller asks the f32 cooperative kernel for:
/// the Metal form's own 32 columns (the ordinary multiply, whose `n` is a model's
/// hidden width) or the portable 8 (the two convolution GEMM routes, whose `n` is
/// a conv's spatial extent). See [`coop_f32_admitted`].
#[derive(Clone, Copy)]
enum CoopF32Alignment {
    Metal,
    Portable,
}

/// Whether the f32 cooperative-matrix kernel may serve a multiply with this
/// contraction (`k`) and output-column (`n`) extent on this device.
///
/// Shared by the ordinary matmul cascade and the two convolution GEMM routes
/// ([`conv_gemm`]), so both read one admission rather than restating it:
/// `RLX_WGPU_NO_COOP_F32` opts out for both, `RLX_WGPU_MATMUL_F32_ONLY` pins the
/// whole matmul machinery to F32 for both, and both are behind the same
/// per-device operand-order probe. `RLX_WGPU_FORCE_COOP_F32` widens admission to
/// adapters that report no 8×8 f32 cooperative matrices; its geometry term is the
/// alignment the conv routes already ask for, so on them it has nothing left to
/// widen.
///
/// Two terms are the callers' own: `bf16_weight_in_b` is the ordinary multiply's
/// (the conv callers pass `false`), and `alignment` is the form the caller asks
/// for — the cascade asks the Metal form's own 32 columns, the conv routes the
/// portable 8. The backend term is the other way round: the conv routes reach this
/// from inside a Metal check of their own, narrower than what this function admits
/// on a Vulkan/DX12 adapter. See [`conv_gemm`].
///
/// CoopF32 (`simdgroup_float8x8` on Apple) is the f32 hardware-GEMM path. It is
/// on by default on Metal; the Vulkan portable 8×8 path is selected
/// automatically too, when the adapter reports 8×8 f32 cooperative matrices.
/// This was opt-in because Metal `simdgroup_float8x8` CoopF32 produced
/// ORTHOGONAL GARBAGE (not mere imprecision) on Gemma-4 vision/audio f32
/// param-weight matmuls — scaled[1,64,768] @ input_proj[768,768], all axes
/// aligned, gave cos 0.016 vs CPU. GGUF text models dodged it via the
/// DequantMatMul path.
///
/// ROOT CAUSE, and why "orthogonal" was the exact right word: the kernel
/// computed `b·a` instead of `a·b`. For two unrelated matrices those are
/// near-orthogonal, hence cos≈0 rather than a merely noisy cos≈1. Every
/// `coopLoad` from row-major memory yields the transpose under this path's
/// column-major fragment convention, so `ã·b̃ = aᵀbᵀ = (b·a)ᵀ`, and the
/// equally-transposed `coopStore` writes that back as `b·a`. Loading each
/// operand into the OTHER fragment role fixes it; see the note in
/// `kernels/matmul_coop_f32.wgsl`. The same bug was in
/// `matmul_qkv_coop_f32.wgsl` and `matmul_coop16.wgsl`.
///
/// Measured after the fix, M4 Pro, f32 square GEMM against `matmul_wide`:
///   512³   88 → 403 GF/s   (4.6×)
///   1024³  96 → 745 GF/s   (7.8×)
///   2048³ 100 → 990 GF/s   (9.9×)
/// and the Gemma-4 shape above is now BIT-EXACT vs CPU (cos 1.000000,
/// max|Δ| 0.0). `RLX_WGPU_NO_COOP_F32=1` opts back out.
fn coop_f32_admitted(
    dev: &wgpu::Device,
    k: u32,
    n: u32,
    bf16_weight_in_b: bool,
    alignment: CoopF32Alignment,
) -> bool {
    // `RLX_WGPU_MATMUL_F32_ONLY` is an explicit pin: F32 is the only admissible
    // path for the whole matmul machinery. Read in two places, and both are
    // load-bearing: the cascade's own early return is what makes the pin absolute
    // for every ordinary multiply (it clears all four eligibility flags), while
    // this read covers the two convolution GEMM routes, which ask this function
    // directly — a pin they could route around would not be the pin the registry
    // documents. If that early return is ever relaxed, this read has to become
    // the whole gate there rather than just the f32 cooperative flag.
    if rlx_ir::env::flag("RLX_WGPU_MATMUL_F32_ONLY") {
        return false;
    }
    let disabled = rlx_ir::env::flag("RLX_WGPU_NO_COOP_F32");
    let forced = rlx_ir::env::flag("RLX_WGPU_FORCE_COOP_F32");
    let has_coop = dev
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let backend = crate::device::wgpu_device().map(|d| d.backend);
    // The Metal 32×32 form's own alignment, and the portable 8×8 form's; M is
    // free on the Metal kernel (see the kernel header for the padded A read it
    // drops). Nothing wider is asked of either.
    //
    // `alignment` swaps the first for the second on the convolution GEMM routes.
    // Their `n` is a conv's spatial extent — `16 × 13 = 208` on the recognition
    // front — so it is a multiple of 8 far more often than of 32, and the kernel
    // lays such an extent out anyway: the x grid rounds up exactly as the y grid
    // does, the padded columns are gathered and accumulated and then dropped by the
    // store guard's `global_col >= params.n` term, the same reliance the padded rows
    // already stand on (see `kernels/matmul_coop_f32.wgsl`). What that extent is
    // worth on the front's own geometry, which other extents move with it and why
    // the widening is scoped to these two routes rather than asked of every shape is
    // in `README.md`. The ordinary multiply keeps its own 32: its `n` is a model's
    // hidden width, where a widening would move every such shape in the tree at
    // once.
    let metal_aligned = k.is_multiple_of(8) && n.is_multiple_of(32);
    let portable_aligned = k.is_multiple_of(8) && n.is_multiple_of(8);
    let aligned = match alignment {
        CoopF32Alignment::Metal => metal_aligned,
        CoopF32Alignment::Portable => portable_aligned,
    };
    let metal_coop = !disabled
        && has_coop
        && aligned
        && (forced || (matches!(backend, Some(wgpu::Backend::Metal)) && !bf16_weight_in_b));
    let vulkan_coop = !disabled
        && has_coop
        && portable_aligned
        && crate::device::coop_discrete_backend()
        && crate::device::coop_f32_8x8_supported();
    (metal_coop
        || vulkan_coop
        || (forced && has_coop && (metal_aligned || portable_aligned)))
        // Final gate: the kernel must have been shown to compute `a·b` ON THIS
        // DEVICE. `coopLoad`/`coopLoadT` mean different things on Metal and
        // Vulkan (naga's one `row_major` bool becomes Metal's
        // `transpose_matrix` but SPIR-V's `RowMajorKHR`/`ColumnMajorKHR`), so
        // no compile-time choice can be right everywhere. Memoized; see
        // `crate::coop_probe`.
        && crate::coop_probe::f32_path_ok(backend)
}

/// The workgroup grid the f32 cooperative kernel expects, by form: Metal's
/// 32×32 tiles over (n, m), the portable 8×8 form over (m, n) — the axes are
/// swapped between them. One definition for the two places that need it: the
/// dispatch, and [`conv_gemm`]'s check that the grid is one the device can
/// launch.
pub(crate) fn coop_f32_grid(m: u32, n: u32) -> (u32, u32) {
    if crate::device::wgpu_device().map(|d| d.backend) == Some(wgpu::Backend::Metal) {
        (n.div_ceil(32), m.div_ceil(32))
    } else {
        (m.div_ceil(8), n.div_ceil(8))
    }
}

/// Whether the direct f32 cooperative entry (`matmul_coop_f32_direct`) may serve
/// this multiply: a whole 32×32 tile with no epilogue, on one of the two
/// convolution GEMM routes.
///
/// The route term is `!b_is_param`, and it is exact rather than a proxy: the
/// matmul cascade names `CoopF32` only for a multiply whose second operand is a
/// Param (`MatmulEligibility`'s own admission), while both conv GEMM routes hold
/// the weight in the FIRST operand — so a `CoopF32` step with `b_is_param` false
/// is one of those routes and nothing else. That is what keeps this entry off the
/// ordinary multiply, whose form is `matmul_coop_f32`'s and whose numbers are
/// measured there; the sound half's convolutions are `m = c_out` and `n = a
/// spatial extent`, both whole tiles at these geometries.
///
/// The epilogue terms are the kernel's: it stores its fragments straight to C with
/// no scratch tile and no scalar pass, so a bias or an activation has nowhere to
/// go. `act_id`'s no-activation sentinel is `0xFFFF` (`Step::Matmul`'s own). `k`'s
/// term is the kernel's too: its k loop walks whole `TILE_K = 8` legs
/// (`(k + 7) / 8` of them), so a contraction off that multiple folds the next
/// row's values into the last leg.
///
/// Shared by the dispatch arm and the routes' witness (`conv_gemm_coop_f32`), so
/// the two cannot spell the shape terms differently.
pub fn coop_f32_direct_admitted(
    has_bias: u32,
    act_id: u32,
    m: u32,
    k: u32,
    n: u32,
    b_is_param: bool,
) -> bool {
    !b_is_param
        && has_bias == 0
        && act_id == 0xFFFF
        && m.is_multiple_of(32)
        && k.is_multiple_of(8)
        && n.is_multiple_of(32)
}

/// Which dense-matmul compute paths are legal for one specific call.
///
/// `Bf16Packed` and `F16` are absent on purpose: neither is chosen by
/// `derive_matmul_compute` — `Bf16Packed` is installed later by the caller when
/// it sees a BF16 `Param` rhs, and no path dispatches `F16` today. A tuning
/// table must not be able to conjure either one here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatmulEligibility {
    pub coop_f16_f32acc: bool,
    pub coop16: bool,
    pub coop_f16_vk: bool,
    pub coop_f32: bool,
    pub f32: bool,
}

impl MatmulEligibility {
    /// Under `RLX_WGPU_MATMUL_F32_ONLY` the pin is absolute.
    fn only_f32() -> Self {
        Self {
            coop_f16_f32acc: false,
            coop16: false,
            coop_f16_vk: false,
            coop_f32: false,
            f32: true,
        }
    }

    fn allows(&self, v: MatmulCompute) -> bool {
        match v {
            MatmulCompute::CoopF16F32Acc => self.coop_f16_f32acc,
            MatmulCompute::Coop16 => self.coop16,
            MatmulCompute::CoopF16Vk => self.coop_f16_vk,
            MatmulCompute::CoopF32 => self.coop_f32,
            MatmulCompute::F32 => self.f32,
            // Never selectable from the table — see the struct doc.
            MatmulCompute::F16
            | MatmulCompute::Bf16Packed
            | MatmulCompute::CoopF16F32AccScratch => false,
        }
    }
}

/// Pick the dense-matmul compute path for one `MatMul` node: the cascade's
/// eligibility, then a measured override from the shared dispatch table.
#[allow(clippy::too_many_arguments)]
fn derive_matmul_compute(
    dev: &wgpu::Device,
    graph: &Graph,
    mirror_acts: &HashSet<NodeId>,
    a_id: NodeId,
    b_id: NodeId,
    m: u32,
    k: u32,
    n: u32,
) -> MatmulCompute {
    let (default, eligible) =
        derive_matmul_compute_default(dev, graph, mirror_acts, a_id, b_id, m, k, n);
    apply_matmul_tuning(m, k, n, default, eligible)
}

/// The compute path for the two convolution GEMM routes' multiplies — the im2col
/// panel multiply and the 1×1 pointwise one. Returns the path **and** the kernel
/// to bind it with, because a bind group carries the pipeline's own layout.
///
/// Both routes put the weight in the **FIRST** operand, the reverse of every
/// other call site, and that role alone rules out every other family: the
/// f16-shadow, packed-bf16 and cooperative-f16 paths each read a weight-side
/// buffer this role never builds. So the choice is binary, and it asks the same
/// admission an ordinary multiply of the same shape asks ([`coop_f32_admitted`],
/// with no packed form to keep and the portable output-column alignment instead
/// of the Metal form's own) and reads the same shared dispatch table — which
/// can move it back to `F32`, never onto a family that cannot serve the layout.
///
/// The extent tolerance is the conv routes' one widening; [`coop_f32_admitted`]
/// explains it, and `README.md` records what it measures and which other extents
/// move with it.
///
/// Three readings of "cannot be launched with the chosen kernel", and in each of
/// them the node keeps today's kernel: this device's f32 cooperative pipeline
/// must exist (`coop_kernel`), the grid it would be dispatched on must fit the
/// per-dimension ceiling, and the run loop's dispatch guard must let it through.
/// The dispatch guard is the one the conv sites sit inside rather than satisfy:
/// its `b_is_param` term is the weight-in-B eligibility restated, and this kernel
/// reads both operands straight out of the arena, so weight-in-A belongs inside
/// that guard — every family that DOES need a weight-side buffer keeps its own
/// eligibility (see the `Step::Matmul` arm in `run.rs`). Of the other two, the
/// pipeline exists exactly when the admission's device feature does (the getter
/// returns `None` for that reason), and the ceilings are out of reach at the
/// shipped floors — the col cap bounds the im2col route's spatial and the
/// pointwise route's cap is the ceiling itself. Both are checked anyway: they are
/// the readings this call owns, and the floors are tunable.
///
/// The Metal term is scope and extent, not routing. The scope half is what keeps
/// the portable 8×8 sibling, Vulkan's, out of these two classes: although these
/// routes are host-staged on a Vulkan/DX12 adapter (`wgpu_prefer_conv_host` folds
/// in `coop_discrete_backend`), `RLX_WGPU_SHARD_GPU` opts them back onto the
/// device, and the kernel `matmul_coop_f32_active_kernel` would then return there
/// is the portable one.
fn conv_gemm(
    dev: &wgpu::Device,
    mm_k: &'static Kernel,
    coop_kernel: Option<&'static Kernel>,
    m: u32,
    k: u32,
    n: u32,
) -> (MatmulCompute, &'static Kernel) {
    let coop_f32 = match crate::device::wgpu_device().map(|d| d.backend) {
        Some(wgpu::Backend::Metal) => {
            let (gx, gy) = coop_f32_grid(m, n);
            coop_kernel.is_some()
                && gx <= MAX_WORKGROUPS_PER_DIM
                && gy <= MAX_WORKGROUPS_PER_DIM
                && coop_f32_admitted(dev, k, n, false, CoopF32Alignment::Portable)
        }
        _ => false,
    };
    let eligible = MatmulEligibility {
        coop_f16_f32acc: false,
        coop16: false,
        coop_f16_vk: false,
        coop_f32,
        f32: true,
    };
    let picked = apply_matmul_tuning(
        m,
        k,
        n,
        if coop_f32 {
            MatmulCompute::CoopF32
        } else {
            MatmulCompute::F32
        },
        eligible,
    );
    // `allows` admits CoopF32 only when `coop_f32` holds, which is the same
    // condition that makes `coop_kernel` present.
    match (picked, coop_kernel) {
        (MatmulCompute::CoopF32, Some(kernel)) => (picked, kernel),
        _ => (picked, mm_k),
    }
}

/// Let the shared dispatch table retune `default` **within** what is legal here.
///
/// The table proposes; eligibility disposes. An override naming a path this
/// call cannot run — a stale cache, or one copied from another adapter — is
/// discarded in favour of the default rather than honoured.
fn apply_matmul_tuning(
    m: u32,
    k: u32,
    n: u32,
    default: MatmulCompute,
    eligible: MatmulEligibility,
) -> MatmulCompute {
    let Some(want) = rlx_gpu_dispatch::dispatch::resolve_matmul(
        crate::tuning::gpu_arch(),
        m as usize,
        k as usize,
        n as usize,
    )
    .as_wgpu_matmul()
    .map(matmul_compute_from_dispatch) else {
        return default;
    };
    if want == default {
        return default;
    }
    if eligible.allows(want) {
        want
    } else {
        if rlx_ir::env::flag("RLX_VERBOSE") {
            eprintln!(
                "rlx-wgpu: tuning cache picks {want:?} for m={m} k={k} n={n} \
                 but it is not eligible there — using {default:?}"
            );
        }
        default
    }
}

/// Bridge to the shared table's backend-agnostic variant enum. A test pins the
/// mapping in both directions so a new variant cannot silently become another.
///
/// Only the `from_` direction is on the hot path today (the table proposes, this
/// crate consumes). `to_` exists so the mapping is stated as a bijection and can
/// be checked as one — and it is what a tuner writing measured winners back into
/// the table will call.
///
/// One deliberate exception to the bijection: `CoopF16F32Acc` has no entry of its
/// own in the shared enum, so it maps onto `D::Coop16`, the 16-bit cooperative
/// family the table already names — adding a variant there would touch every GPU
/// backend's table for one wgpu shader. The consequence is exact and confined to
/// the table: it can name that family (which then means whichever of the two the
/// caller's eligibility admits, `CoopF16F32Acc` first), and
/// `matmul_compute_from_dispatch` never returns `CoopF16F32Acc`, so no cached
/// entry can select it. The round-trip test still holds because it walks the
/// shared enum, not this one.
///
/// `CoopF16F32AccScratch` — the same shader with B bound from the attention
/// route's own f16 operand buffer instead of the weight mirror — appears here for
/// the same reason and one more: it is a *wgpu dispatch* distinction (which
/// buffer binding 2 is), not a device fact the shared table has any term for, so
/// it maps onto the same family and the table can never select it. It is stated
/// rather than omitted so that the match stays exhaustive over this crate's enum:
/// a new variant has to say what it means to the table in the same place the
/// others do.
#[allow(dead_code)]
pub(crate) fn matmul_compute_to_dispatch(
    v: MatmulCompute,
) -> rlx_gpu_dispatch::dispatch::WgpuMatmul {
    use rlx_gpu_dispatch::dispatch::WgpuMatmul as D;
    match v {
        MatmulCompute::F32 => D::F32,
        MatmulCompute::F16 => D::F16,
        // The scratch form reuses the probe's convention: same kernel, same
        // family, only B's buffer differs.
        MatmulCompute::Coop16
        | MatmulCompute::CoopF16F32Acc
        | MatmulCompute::CoopF16F32AccScratch => D::Coop16,
        MatmulCompute::CoopF32 => D::CoopF32,
        MatmulCompute::CoopF16Vk => D::CoopF16Vk,
        MatmulCompute::Bf16Packed => D::Bf16Packed,
    }
}

pub(crate) fn matmul_compute_from_dispatch(
    v: rlx_gpu_dispatch::dispatch::WgpuMatmul,
) -> MatmulCompute {
    use rlx_gpu_dispatch::dispatch::WgpuMatmul as D;
    match v {
        D::F32 => MatmulCompute::F32,
        D::F16 => MatmulCompute::F16,
        D::Coop16 => MatmulCompute::Coop16,
        D::CoopF32 => MatmulCompute::CoopF32,
        D::CoopF16Vk => MatmulCompute::CoopF16Vk,
        D::Bf16Packed => MatmulCompute::Bf16Packed,
    }
}

/// Detects the BERT-style fused-QKV-then-narrow-then-attention
/// pattern. When all three of an attention's Q/K/V inputs are
/// `Op::Narrow` of a single source tensor on the last axis with
/// sequential offsets `(0, H·D, 2·H·D)` and equal lengths `H·D`,
/// returns `Some((qkv_source_node, h_d))` — naming the source
/// tensor and per-slice width.
///
/// EMPIRICAL FINDING: the obvious "skip the narrow + read attention
/// directly from QKV with stride 3·H·D" optimization REGRESSED end-
/// to-end perf 7-15× on Apple M4 Pro. The narrow's apparent overhead
/// (~3 dispatches per attention block, ~150µs at small batch) is
/// dwarfed by the cost of strided attention reads — stepping by
/// 3·H·D = 4.6 KB between sequence positions defeats the hardware
/// prefetcher (prefetch distance maxes around 1-2 KB on M-series).
/// Cosine stayed 0.9999+ (output is correct, just slow).
///
/// Kept as a helper for future smarter fusions — e.g. a coop kernel
/// that reads Q/K/V cooperatively from QKV in a single pass over
/// the sequence dim, avoiding the random-access stride pattern.
#[allow(dead_code)]
fn detect_qkv_narrow_pattern(
    graph: &Graph,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
) -> Option<(NodeId, u32)> {
    let unwrap_narrow = |id: NodeId| -> Option<(NodeId, usize, usize, usize)> {
        let node = graph.node(id);
        match &node.op {
            Op::Narrow { axis, start, len } => Some((node.inputs[0], *axis, *start, *len)),
            _ => None,
        }
    };
    let (q_src, q_axis, q_start, q_len) = unwrap_narrow(q_id)?;
    let (k_src, k_axis, k_start, k_len) = unwrap_narrow(k_id)?;
    let (v_src, v_axis, v_start, v_len) = unwrap_narrow(v_id)?;
    // Same source tensor.
    if q_src != k_src || k_src != v_src {
        return None;
    }
    // Equal slice widths (= H · D).
    if q_len != k_len || k_len != v_len {
        return None;
    }
    // Sequential offsets 0, H·D, 2·H·D.
    if q_start != 0 || k_start != q_len || v_start != q_len * 2 {
        return None;
    }
    // All on the LAST axis of the source.
    let src_rank = graph.node(q_src).shape.dims().len();
    if q_axis + 1 != src_rank || k_axis + 1 != src_rank || v_axis + 1 != src_rank {
        return None;
    }
    Some((q_src, q_len as u32))
}

/// Detects the (FusedMatMulBiasAct → Narrow×3) split-QKV pattern that
/// shows up at the start of every BERT-style attention block. Returns
/// a map `parent_fmb_id → (q_narrow_id, k_narrow_id, v_narrow_id)`
/// for every site where the pattern can be replaced by one
/// `Step::MatmulQkv` dispatch.
///
/// Pattern requirements:
///   - Parent is `Op::FusedMatMulBiasAct { activation: None }` with
///     output shape `[..., 3·head_width]`.
///   - The parent's *only* consumers are exactly 3 `Op::Narrow` nodes,
///     all on the last axis, with offsets `(0, head_width, 2·head_width)`
///     and equal `len = head_width`.
///
/// The win is purely structural: same FMA work, but the 3 narrow
/// dispatches (and their full-tensor read+write of the QKV intermediate)
/// disappear. Different from the reverted "skip narrow + read attention
/// strided" approach because reads from each Q/K/V buffer remain
/// sequential — the prefetcher stays happy.
/// Detects (`Op::Binary(Add) → Op::LayerNorm | Op::RmsNorm`) where the Add has
/// more than one consumer in the graph — the case `FuseResidualLN` /
/// `FuseResidualRmsNorm` decline because their single-consumer guard would
/// force materializing the sum.
///
/// That guard makes the fusion miss EVERY transformer layer, because the
/// residual sum feeds both the norm and the next residual
/// (`h += attn; n = norm(h); h += ffn(n)`). On an 8-block probe wgpu emitted 8
/// standalone Binary + 8 standalone norms where this collapses them to 8.
///
/// Returns:
///   - `ln_to_tee`: `ln_id → (h, delta, gamma, beta, sum_id)` so the
///     wgpu LayerNorm lowering can emit `Step::FusedResidualLnTee`
///     using the existing arena slot for the sum (= the Add's slot).
///   - `skip_adds`: the set of Add `NodeId`s whose normal Step emission
///     should be suppressed; their output value is written by the tee
///     step instead.
fn detect_residual_ln_tee_pattern(
    graph: &Graph,
) -> (
    HashMap<NodeId, (NodeId, NodeId, NodeId, NodeId, NodeId)>,
    HashSet<NodeId>,
) {
    use rlx_ir::op::BinaryOp;
    // Consumer counts (output references count once each).
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    // …and the consuming nodes themselves, so we can check *when* they run.
    let mut consumer_nodes: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            *consumers.entry(input).or_insert(0) += 1;
            consumer_nodes.entry(input).or_default().push(node.id);
        }
    }
    for &out in &graph.outputs {
        *consumers.entry(out).or_insert(0) += 1;
    }
    // Emission order: steps are pushed while walking `graph.nodes()`.
    let order: HashMap<NodeId, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    let mut ln_to_tee = HashMap::new();
    let mut skip_adds = HashSet::new();
    for node in graph.nodes() {
        // BOTH norms. They share this shape — the residual sum feeds the norm
        // and the next residual — and share the wgpu lowering arm; only the
        // final scaling differs, which the kernel selects on `is_rms`. RmsNorm
        // is the one that matters for Llama-class models, and it was excluded
        // here purely because the pattern was written for a LayerNorm vision
        // transformer.
        if !matches!(&node.op, Op::LayerNorm { .. } | Op::RmsNorm { .. }) {
            continue;
        }
        if node.inputs.len() < 3 {
            continue;
        } // need [in, gamma, beta]
        let in_id = node.inputs[0];
        let in_node = graph.node(in_id);
        if !matches!(in_node.op, Op::Binary(BinaryOp::Add)) {
            continue;
        }
        // Only fire when Add has >= 2 consumers (otherwise `FuseResidualLN`
        // already collapses it into Op::FusedResidualLN upstream).
        if consumers.get(&in_id).copied().unwrap_or(0) < 2 {
            continue;
        }
        // Add must be plain — both operands shape-equal to LN's input
        // and to each other.
        if in_node.inputs.len() != 2 {
            continue;
        }
        let h_id = in_node.inputs[0];
        let delta_id = in_node.inputs[1];
        if graph.node(h_id).shape.dims() != node.shape.dims() {
            continue;
        }
        if graph.node(delta_id).shape.dims() != node.shape.dims() {
            continue;
        }
        // The tee writes the sum as a side effect of the *norm's* step, and the
        // Add's own step is suppressed. That is only sound if every other reader
        // of the Add runs after the norm — otherwise it reads a slot nothing has
        // written yet. `rlx-vieeg` is exactly that shape: the residual also
        // feeds a mean-pool head five steps earlier, so on wgpu the head read
        // zeros and every hierarchy level came out wrong, but only when the
        // graph had two or more distinct outputs (with one, the head is dead
        // code and the early reader disappears). Graph outputs are read after
        // the whole schedule, so they do not constrain this.
        let ln_pos = order.get(&node.id).copied().unwrap_or(usize::MAX);
        let early_reader = consumer_nodes.get(&in_id).is_some_and(|cs| {
            cs.iter()
                .any(|c| *c != node.id && order.get(c).copied().unwrap_or(usize::MAX) < ln_pos)
        });
        if early_reader {
            continue;
        }
        let gamma_id = node.inputs[1];
        let beta_id = node.inputs[2];
        ln_to_tee.insert(node.id, (h_id, delta_id, gamma_id, beta_id, in_id));
        skip_adds.insert(in_id);
    }
    (ln_to_tee, skip_adds)
}

/// One matched DAC snake for [`Step::SnakeBeta`].
pub(super) struct SnakeBetaPlan {
    /// The addend the snake scales and adds back to: `x` itself, or the
    /// convolution's output when the snake's input `+ bias` was folded in.
    pub x: NodeId,
    /// The per-channel `[1, C, 1, 1]` operands, resolved to their own arena
    /// offsets by the lowering like any other operand.
    pub alpha: NodeId,
    pub beta: NodeId,
    /// The per-channel bias the snake's own input `Add` applied, when that Add
    /// had no reader but this snake and its ranges allowed the fold.
    pub bias: Option<NodeId>,
}

/// The graph-level facts every bias-fold detector asks: who reads what, which
/// nodes are graph outputs, and the order steps are pushed in.
pub(crate) struct FoldGraph<'a> {
    graph: &'a Graph,
    consumers: HashMap<NodeId, Vec<NodeId>>,
    outputs: HashSet<NodeId>,
    order: HashMap<NodeId, usize>,
}

impl<'a> FoldGraph<'a> {
    pub(crate) fn new(graph: &'a Graph) -> Self {
        let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            for &input in &node.inputs {
                consumers.entry(input).or_default().push(node.id);
            }
        }
        Self {
            graph,
            consumers,
            outputs: graph.outputs.iter().copied().collect(),
            // Steps are pushed while walking `graph.nodes()`.
            order: graph
                .nodes()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.id, i))
                .collect(),
        }
    }

    /// The one node reading `id`, if that is the whole of its use and it is not
    /// also a graph output (a node reading `s` twice counts once).
    fn sole_reader(&self, id: NodeId) -> Option<NodeId> {
        if self.outputs.contains(&id) {
            return None;
        }
        let cs = self.consumers.get(&id)?;
        let first = *cs.first()?;
        cs.iter().all(|c| *c == first).then_some(first)
    }

    fn readers(&self, id: NodeId) -> &[NodeId] {
        self.consumers.get(&id).map_or(&[], Vec::as_slice)
    }

    fn is_output(&self, id: NodeId) -> bool {
        self.outputs.contains(&id)
    }

    /// `[C]` or `[1, C, 1, 1]` — the two shapes whose element `c` is word `c`,
    /// from a parameter's pinned bytes. A fold reads them at its own position
    /// while the plan ends their liveness early, which is sound only for pinned
    /// bytes: anything computed would be a read of a reused slot.
    fn per_channel(&self, id: NodeId) -> Option<usize> {
        if !crate::buffer::is_parameter_storage(self.graph, id) {
            return None;
        }
        let n = self.graph.node(id);
        if n.shape.dtype() != rlx_ir::DType::F32 {
            return None;
        }
        let d = all_static_dims(&n.shape)?;
        match d.as_slice() {
            [c] | [1, c, 1, 1] if *c > 0 => Some(*c),
            _ => None,
        }
    }

    /// `ids` are a binary node's two operands: `(per_channel_id, channel_count,
    /// other)`.
    fn split_by_channel(&self, ids: &[NodeId]) -> Option<(NodeId, usize, NodeId)> {
        ids.iter()
            .enumerate()
            .find_map(|(i, &a)| Some((a, self.per_channel(a)?, ids[1 - i])))
    }
}

/// Detects the DAC snake, `out = x + beta·sin²(alpha·x)`, written as the five
/// nodes the sound half's `snake_beta` helper emits:
///
/// ```text
/// arg   = Mul(alpha, x)      alpha  [1, C, 1, 1]
/// s     = Activation(Sin, arg)
/// s2    = Mul(s, s)
/// scale = Mul(beta, s2)      beta   [1, C, 1, 1]
/// out   = Add(x, scale)      x, out [1, C, 1, L]
/// ```
///
/// The elementwise REGION pass cannot carry this chain: the per-channel operands
/// walk the channel with the time axis on the way out of a `[1, C, 1, L]`
/// output, which is the mid-axis singleton `region_input_modulus` refuses, so
/// the chain stays five dispatches over the same tensor today. A DAC block has
/// seven of them.
///
/// The fold is structural and takes no measurement: one dispatch reading `x`,
/// `alpha[c]`, `beta[c]` and writing `out` replaces five that each read and
/// write the whole tensor. It changes no arithmetic — `snake_beta.wgsl` groups
/// the product as `beta·(s·s)` and adds in the chain's order.
///
/// Returns:
///   - `out_id → SnakeBetaPlan` — the Add whose step becomes `Step::SnakeBeta`;
///   - the ids whose own steps are suppressed: the four chain intermediates,
///     plus the input `Add` when its bias was folded.
///
/// ELIGIBILITY. `x`/`out` must be a rank-4 static `[1, C, 1, L]` f32, which is
/// what makes the flattened address `base + c·inner + l` the tensor's own; the
/// per-channel operands must be `[C]` or `[1, C, 1, 1]`, so element `c` is word
/// `c`, and a parameter's storage — the step reads them at `out`'s position,
/// after the plan ends their liveness at `arg`/`scale`. Every node the fold
/// consumes must have no reader but the next one in the chain (the same node
/// twice counting once — that is `s2 = s·s`) and none may be a graph output; the
/// input `Add` counts among them, since folding its bias is what suppresses its
/// own step. Each of them must also be one `on_device` answers true for — the
/// lowering's own routing predicate, so a route that would carry a binary or an
/// activation to the host keeps the chain's steps. The dispatch's grid must fit
/// [`MAX_WORKGROUPS_PER_DIM`] on both axes — one workgroup per channel, and the
/// row split into `SNAKE_BETA_WG`-wide workgroups — because a clipped grid would
/// leave rows unwritten. Every one of those rules declines the whole pattern with
/// `continue`; the lowering's arm only lowers.
///
/// The bias fold has one ownership rule of its own, and it declines the bias fold
/// alone — the chain still folds and `bias` is left unset: the `Add` it replaces may
/// not be another fold's `out`, because the write the fold suppresses would be that
/// snake's. A `conv` that is another snake's `out` is read deliberately: that step
/// is emitted, so the bytes are there.
///
/// THE BIAS FOLD, AND WHY IT NEEDS A RANGE GUARD. `x = Add(conv, bias)` becomes
/// the step's own `v = conv[i] + bias[c]` when that Add's only readers are `arg`
/// and `out`. Two things then differ from the plain fold, and both are about the
/// arena plan rather than the arithmetic:
///
///   * `conv` dies at the Add's position while this step reads it at `out`'s, so
///     every node strictly between the two positions must be one the fold
///     consumes — those emit no step, so nothing of their own can write there.
///     Params, Inputs and Constants are exempt: they are pinned for the whole
///     execution and write nothing.
///   * a freed range is reused at a *shifted* offset, not only wholesale. The
///     DAC front graph is the repro: `out` took `[58 982 400, 66 846 720)` while
///     `conv` sat at `[58 989 312, 66 853 632)` — the same freed region, 6 912
///     bytes apart. Reading `conv` while writing `out` then races its own step
///     against itself, and the sound half's waveform moved between runs of one
///     binary (digest `006256…` against `bb2eeb…` of the same 300-token cell,
///     peak |diff| 395 of 552 counts). The fold is therefore taken only when the
///     two byte ranges are **disjoint**, which is what makes one dispatch over
///     `conv` and `out` an identity mapping rather than a race. The ranges are
///     [`Arena::storage_span`]s — a view's slot is bounded by the storage it
///     reads, not by its own size — and the fold is declined outright where
///     either operand is parked in the weight buffer, whose tagged offsets are a
///     different byte space and would make the comparison vacuous. `x` itself
///     needs no such guard: `out` reads it, so the planner keeps its slot alive
///     to `out`'s own position, and the two ranges abut rather than overlap.
fn detect_snake_beta_pattern(
    fold: &FoldGraph<'_>,
    arena: &Arena,
    on_device: &dyn Fn(NodeId) -> bool,
) -> (HashMap<NodeId, SnakeBetaPlan>, HashSet<NodeId>) {
    use rlx_ir::op::{Activation, BinaryOp};

    let graph = fold.graph;

    let mut out_to_snake = HashMap::new();
    let mut skip: HashSet<NodeId> = HashSet::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::Binary(BinaryOp::Add)) || node.inputs.len() != 2 {
            continue;
        }
        if node.shape.dtype() != rlx_ir::DType::F32 {
            continue;
        }
        let Some(out_dims) = all_static_dims(&node.shape) else {
            continue;
        };
        // `[1, C, 1, L]`, the contiguous layout the kernel's row/column
        // addressing is the flattened form of.
        if out_dims.len() != 4 || out_dims[0] != 1 || out_dims[2] != 1 {
            continue;
        }
        let (channels, inner) = (out_dims[1], out_dims[3]);
        if channels == 0 || inner == 0 {
            continue;
        }
        // The dispatch is one workgroup per channel and `SNAKE_BETA_WG` columns
        // per workgroup, in two dimensions; a shape whose grid would not fit a
        // per-dimension cap would dispatch a clipped grid and leave rows
        // unwritten, so it declines like every other rule here.
        if channels > MAX_WORKGROUPS_PER_DIM as usize
            || inner.div_ceil(crate::kernels::SNAKE_BETA_WG as usize)
                > MAX_WORKGROUPS_PER_DIM as usize
        {
            continue;
        }
        for (xi, si) in [(0usize, 1usize), (1, 0)] {
            let (x, scale) = (node.inputs[xi], node.inputs[si]);
            if graph.node(x).shape != node.shape {
                continue;
            }
            // scale = Mul(beta, s2)
            let scale_node = graph.node(scale);
            if !matches!(scale_node.op, Op::Binary(BinaryOp::Mul)) || scale_node.inputs.len() != 2 {
                continue;
            }
            if fold.sole_reader(scale) != Some(node.id) {
                continue;
            }
            let Some((beta, beta_len, s2)) = fold.split_by_channel(&scale_node.inputs) else {
                continue;
            };
            if beta_len != channels || graph.node(s2).shape != node.shape {
                continue;
            }
            // s2 = Mul(s, s)
            let s2_node = graph.node(s2);
            if !matches!(s2_node.op, Op::Binary(BinaryOp::Mul))
                || s2_node.inputs.len() != 2
                || s2_node.inputs[0] != s2_node.inputs[1]
            {
                continue;
            }
            let s = s2_node.inputs[0];
            if fold.sole_reader(s2) != Some(scale) || graph.node(s).shape != node.shape {
                continue;
            }
            if fold.sole_reader(s) != Some(s2) {
                continue;
            }
            // s = Activation(Sin, arg)
            let s_node = graph.node(s);
            let Op::Activation(Activation::Sin) = s_node.op else {
                continue;
            };
            if s_node.inputs.len() != 1 {
                continue;
            }
            let arg = s_node.inputs[0];
            if fold.sole_reader(arg) != Some(s) || graph.node(arg).shape != node.shape {
                continue;
            }
            // arg = Mul(alpha, x)
            let arg_node = graph.node(arg);
            if !matches!(arg_node.op, Op::Binary(BinaryOp::Mul)) || arg_node.inputs.len() != 2 {
                continue;
            }
            let Some((alpha, alpha_len, arg_x)) = fold.split_by_channel(&arg_node.inputs) else {
                continue;
            };
            if alpha_len != channels || arg_x != x {
                continue;
            }
            if ![arg, s, s2, scale].into_iter().all(on_device) {
                continue;
            }

            // Bias fold: `x = Add(conv, bias)`, read by this snake and nothing
            // else. `conv` takes over as the addend.
            let mut x = x;
            let mut bias = None;
            let x_node = graph.node(x);
            let chain = [arg, s, s2, scale];
            if let Op::Binary(BinaryOp::Add) = x_node.op
                && x_node.inputs.len() == 2
                && x_node.shape == node.shape
                // Its step is what the fold replaces, so it cannot also be an
                // output: nothing else would write that slot.
                && !fold.is_output(x)
                && on_device(x)
                // Its step is the one the fold suppresses, so it cannot be
                // another fold's `out`: that write would go with it.
                && !out_to_snake.contains_key(&x)
                && let readers = fold.readers(x)
                && !readers.is_empty()
                && readers.iter().all(|c| *c == arg || *c == node.id)
                && let Some((b, b_len, conv)) = fold.split_by_channel(&x_node.inputs)
                && b_len == channels
                && graph.node(conv).shape == node.shape
                && bias_fold_is_sound(fold, arena, &chain, x, conv, node.id)
            {
                x = conv;
                bias = Some(b);
            }

            skip.extend(chain);
            if bias.is_some() {
                // The folded Add's own step is what the fold replaces.
                skip.insert(x_node.id);
            }
            out_to_snake.insert(
                node.id,
                SnakeBetaPlan {
                    x,
                    alpha,
                    beta,
                    bias,
                },
            );
            break;
        }
    }
    (out_to_snake, skip)
}

/// One matched `Gather(table, Add(idx, c))` for the in-kernel index bias.
struct GatherIndexBias {
    /// The Add's other operand, read in place of the Add's output.
    pub idx: NodeId,
    /// The constant the Add applied, added in-kernel before truncation.
    pub bias: f32,
}

/// Detects `out = Gather(table, Add(idx, c))` — a gather whose index carries a
/// constant offset — so the gather reads `idx` with `c` added in-kernel and the
/// `Add` emits no step of its own.
///
/// The fold changes no arithmetic: the kernel adds the f32 constant to the f32
/// index before truncating it, exactly as the `Add` did. It needs an axis-0
/// gather over a single-element index, an f32 `Add` over a single-element
/// operand and a 4-byte `Constant`, and the Add's only reader to be this
/// gather. Both the Add and the gather must be ones `on_device` answers true
/// for: a straddle or host route emits no step for the gather, so the Add's
/// write would go with nothing to consume it.
/// [`bias_fold_is_sound`] then proves that
/// reading `idx` at the gather's position is safe — nothing between the two can
/// write its bytes, and the gather's output does not alias them. Every rule
/// declines the whole pattern.
///
/// Returns `gather id → plan` and the Add ids whose steps are suppressed.
fn detect_gather_index_bias(
    fold: &FoldGraph<'_>,
    arena: &Arena,
    on_device: &dyn Fn(NodeId) -> bool,
) -> (HashMap<NodeId, GatherIndexBias>, HashSet<NodeId>) {
    use rlx_ir::op::BinaryOp;

    let graph = fold.graph;
    // A scalar or `[1]` — shapes whose single element is the whole tensor.
    let one_element = |id: NodeId| -> bool {
        all_static_dims(&graph.node(id).shape).is_some_and(|d| d.iter().product::<usize>() == 1)
    };

    let mut plans: HashMap<NodeId, GatherIndexBias> = HashMap::new();
    let mut skip: HashSet<NodeId> = HashSet::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::Gather { axis: 0 }) || node.inputs.len() != 2 {
            continue;
        }
        let gather_id = node.id;
        if !one_element(node.inputs[1]) {
            continue;
        }
        let add = node.inputs[1];
        let add_node = graph.node(add);
        if !matches!(add_node.op, Op::Binary(BinaryOp::Add))
            || add_node.inputs.len() != 2
            || add_node.shape.dtype() != rlx_ir::DType::F32
        {
            continue;
        }
        // One operand is the 4-byte constant; the other is read in its place.
        let constant = add_node
            .inputs
            .iter()
            .enumerate()
            .find_map(|(i, &operand)| {
                let Op::Constant { data } = &graph.node(operand).op else {
                    return None;
                };
                (data.len() == 4).then(|| {
                    (
                        f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                        add_node.inputs[1 - i],
                    )
                })
            });
        let Some((bias, idx)) = constant else {
            continue;
        };
        // Both ends have to be ones `on_device` answers true for: a straddle or
        // host-routed gather emits no step of its own, so suppressing the Add
        // would leave an unwritten slot behind.
        if !on_device(add)
            || !on_device(gather_id)
            || fold.sole_reader(add) != Some(gather_id)
            || !bias_fold_is_sound(fold, arena, &[], add, idx, gather_id)
        {
            continue;
        }
        plans.insert(gather_id, GatherIndexBias { idx, bias });
        skip.insert(add);
    }
    (plans, skip)
}

/// Whether a bias fold may read `x` at `out`'s position — the two conditions
/// every bias-fold detector's doc spells out, in the order they are cheap:
/// nothing but the fold's own consumed nodes (or a pinned
/// Param/Input/Constant) may sit between the folded Add and `out`, and the
/// bytes of `x` and `out` must be disjoint, since the one dispatch reads the
/// first while writing the second.
fn bias_fold_is_sound(
    fold: &FoldGraph<'_>,
    arena: &Arena,
    consumed: &[NodeId],
    folded_add: NodeId,
    x: NodeId,
    out: NodeId,
) -> bool {
    let (Some(&add_pos), Some(&out_pos)) = (fold.order.get(&folded_add), fold.order.get(&out))
    else {
        return false;
    };
    if add_pos >= out_pos {
        return false;
    }
    let window_is_quiet = fold.graph.nodes()[add_pos + 1..out_pos].iter().all(|n| {
        consumed.contains(&n.id)
            || matches!(
                n.op,
                Op::Param { .. } | Op::Input { .. } | Op::Constant { .. }
            )
    });
    if !window_is_quiet {
        return false;
    }
    // The step reads `x` while writing `out` from the same dispatch, so the two
    // have to be disjoint in the bytes each one *reaches* — the storage span,
    // not the view's own slot: a pure view covers only its own range while the
    // storage underneath it runs on, and a parked parameter's tagged offset is a
    // different byte space whose comparison would answer nothing.
    let (Some((cs, ce)), Some((os, oe))) = (
        arena.storage_span(fold.graph, x),
        arena.storage_span(fold.graph, out),
    ) else {
        return false;
    };
    ce <= os || oe <= cs
}

fn detect_split_qkv_pattern(graph: &Graph) -> HashMap<NodeId, (NodeId, NodeId, NodeId)> {
    // consumers[parent] = list of node ids that read parent
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(node.id);
        }
    }
    // Output nodes also count as consumers — would prevent QKV elision
    // if the matmul output is ever read externally.
    for &out_id in &graph.outputs {
        consumers.entry(out_id).or_default().push(NodeId(u32::MAX));
    }

    let mut result = HashMap::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::FusedMatMulBiasAct { activation: None }) {
            continue;
        }
        let cs = match consumers.get(&node.id) {
            Some(c) if c.len() == 3 => c,
            _ => continue,
        };
        let dims = node.shape.dims();
        if dims.is_empty() {
            continue;
        }
        let last_axis = dims.len() - 1;
        let n = dims[last_axis].unwrap_static();
        if n % 3 != 0 {
            continue;
        }
        let head_width = n / 3;

        // Each consumer must be a Narrow on the last axis, len = head_width.
        let mut narrows: Vec<(usize, NodeId)> = Vec::with_capacity(3);
        let mut all_match = true;
        for &c in cs {
            let cn = graph.node(c);
            match cn.op {
                Op::Narrow { axis, start, len }
                    if axis == last_axis && len == head_width && cn.inputs[0] == node.id =>
                {
                    narrows.push((start, c));
                }
                _ => {
                    all_match = false;
                    break;
                }
            }
        }
        if !all_match {
            continue;
        }
        narrows.sort_by_key(|&(start, _)| start);
        if narrows[0].0 != 0 || narrows[1].0 != head_width || narrows[2].0 != 2 * head_width {
            continue;
        }
        result.insert(node.id, (narrows[0].1, narrows[1].1, narrows[2].1));
    }
    result
}

/// One `RmsNorm → Rope` pair of the loop's per-layer chain: a per-head norm over
/// rows of `inner` elements, then the NeoX rotation of those same rows.
#[derive(Clone)]
pub(crate) struct NormRopeMember {
    pub norm: NodeId,
    pub rope: NodeId,
    /// The norm's operand — what the fused step reads instead of the norm's
    /// intermediate slot.
    pub input: NodeId,
    pub gamma: NodeId,
    pub beta: NodeId,
    pub heads: u32,
}

/// The members one `Step::NormRope` folds, and the geometry they share. A
/// block's q and k are two members of one group, so they cost one dispatch.
pub(crate) struct NormRopeGroup {
    pub members: Vec<NormRopeMember>,
    pub seq: u32,
    pub inner: u32,
    pub cos: NodeId,
    pub sin: NodeId,
    pub cos_row_stride: u32,
    pub eps_bits: u32,
    /// Set when the pair's operands are pieces of one split, whose step this one
    /// replaces (see [`NormRopeSplit`]).
    pub split: Option<NormRopeSplit>,
}

/// Two byte spans that cannot overlap; `None` is a span the arena cannot place
/// (a parked parameter lives in the weight buffer, a different byte space) and
/// carries no hazard.
fn spans_disjoint(a: Option<(u64, u64)>, b: Option<(u64, u64)>) -> bool {
    match (a, b) {
        (Some((as_, ae)), Some((bs, be))) => ae <= bs || be <= as_,
        _ => true,
    }
}

/// The split a norm+rope pair absorbed: the pair's pieces are slices of one row,
/// so the step reads that row directly and copies the piece the pair does not
/// read, instead of the split writing all three first.
pub(crate) struct NormRopeSplit {
    /// The row the members' pieces are slices of, in elements.
    pub src: NodeId,
    /// Each member's piece start in that row, in member order.
    pub starts: [u32; 2],
    /// The piece the pair does not read: `(node, start, len)`, in elements.
    pub copy: Option<(NodeId, u32, u32)>,
}

/// The extents of a shape, or `None` where the graph has not pinned them.
fn all_static_dims(shape: &rlx_ir::Shape) -> Option<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| d.is_static().then(|| d.unwrap_static()))
        .collect()
}

/// `RmsNorm → RoPE` as one `Step::NormRope`, keyed by the node the step is
/// emitted at (the last member's rope), plus the nodes whose own steps the fold
/// replaces.
///
/// The admission is `bias_fold_is_sound`'s rule, for the same reason: the step
/// reads each norm's operand and writes each rope's slot from one dispatch one
/// or more node positions after the operand's last planned reader, so every node
/// strictly between must be consumed by the fold or write nothing at all, and
/// the bytes read must be disjoint from the bytes written.
///
/// The geometry it will only fold:
/// * full NeoX rotation (`n_rot == head_dim`, even, no partial pass-through),
/// * `batch == 1` with the operand's last extent the head width,
/// * the norm's outer extent exactly `seq × heads`,
/// * the norm's output read by nothing but this rope, with no node whose step the
///   fold suppresses also a graph output, and
/// * a cos/sin table one row per position, `head_dim / 2` wide.
///
/// Two members group only when they are adjacent in node order up to pure views
/// and share the position table.
///
/// The fold suppresses both the norm's and the rope's own steps, so it is taken
/// only where they would have stayed on the device: `on_device` is the walk's own
/// routing answer (`should_pack_host_op`), so a configuration that hosts the
/// per-head norms — `RLX_WGPU_HOST_NORM`, a discrete-Vulkan norm, a sharded arena
/// — keeps the pair's steps, and a change to that rule moves this gate with it.
///
/// A pair whose two operands are pieces of one [`NarrowSplitGroup`] absorbs that
/// split: the step reads the split's source row at each piece's start and copies
/// the piece the pair does not read, so the block's fused qkv row and its two
/// head norms cost one dispatch instead of two. The absorbed group is removed
/// from `splits` and all of its pieces are added to `split_skip`, so no piece
/// keeps a step of its own; the admission is the same window rule (extended back
/// to the split, whose source is now read at the pair's position) plus strict
/// disjointness between that source and every written span. Runs after
/// [`detect_narrow_split_pattern`] and may consume the groups it returns.
pub(crate) fn detect_norm_rope_pattern(
    graph: &Graph,
    arena: &Arena,
    on_device: &dyn Fn(NodeId) -> bool,
    splits: &mut HashMap<NodeId, NarrowSplitGroup>,
    split_skip: &mut HashSet<NodeId>,
) -> (HashMap<NodeId, NormRopeGroup>, HashSet<NodeId>) {
    use rlx_ir::op::RopeStyle;

    let ctx = NormRopeCtx::new(graph, arena);

    struct Candidate {
        member: NormRopeMember,
        seq: u32,
        inner: u32,
        cos: NodeId,
        sin: NodeId,
        cos_row_stride: u32,
        eps_bits: u32,
        /// The norm's own position — the far end of the window the fold must own.
        norm_pos: usize,
        rope_pos: usize,
        /// The norm and the views between it and the rope.
        chain: Vec<NodeId>,
    }

    let mut cands: Vec<Candidate> = Vec::new();
    for node in graph.nodes() {
        let Op::Rope {
            head_dim,
            n_rot,
            style,
        } = node.op
        else {
            continue;
        };
        if style != RopeStyle::NeoX || n_rot != head_dim || head_dim == 0 || head_dim % 2 != 0 {
            continue;
        }
        // The rope's operand must be this one norm, through layout-only views.
        let mut cur = node.inputs[0];
        let mut chain: Vec<NodeId> = Vec::new();
        let norm = loop {
            let n = graph.node(cur);
            if matches!(n.op, Op::Reshape { .. })
                && !n.inputs.is_empty()
                && rlx_compile::memory::is_pure_view(graph, n)
            {
                chain.push(cur);
                cur = n.inputs[0];
                continue;
            }
            break cur;
        };
        let Op::RmsNorm { axis, eps } = graph.node(norm).op else {
            continue;
        };
        let norm_node = graph.node(norm);
        if norm_node.inputs.len() != 3 {
            continue;
        }
        // Both suppressed steps must be ones the walk would have run on the
        // device; a hosted norm or rope keeps its own step.
        if !on_device(norm) || !on_device(node.id) {
            continue;
        }
        // Single consumer all the way down the view chain, ending at this rope.
        // `chain` is built rope-first, so the norm's consumer is its last entry
        // and each view reads the one before it.
        let head = chain.last().copied().unwrap_or(node.id);
        if !ctx.sole_user(norm, head) {
            continue;
        }
        let mut single = true;
        for (i, &v) in chain.iter().enumerate() {
            let want = if i == 0 { node.id } else { chain[i - 1] };
            if !ctx.sole_user(v, want) {
                single = false;
                break;
            }
        }
        if !single {
            continue;
        }
        let (input, gamma, beta) = (
            norm_node.inputs[0],
            norm_node.inputs[1],
            norm_node.inputs[2],
        );
        let (Some(in_dims), Some(x_dims)) = (
            all_static_dims(&graph.node(input).shape),
            all_static_dims(&node.shape),
        ) else {
            continue;
        };
        if in_dims.len() != 2 || x_dims.len() < 2 {
            continue;
        }
        let inner = in_dims[1];
        if inner != head_dim || !inner.is_multiple_of(2) {
            continue;
        }
        // `-1` and the explicit last axis are the same axis spelled two ways.
        if axis != -1 && axis != (in_dims.len() as i32 - 1) {
            continue;
        }
        let seq = x_dims[x_dims.len() - 2];
        let batch: usize = x_dims[..x_dims.len() - 2].iter().product();
        if batch != 1 || seq == 0 || head_dim == 0 || x_dims[x_dims.len() - 1] % head_dim != 0 {
            continue;
        }
        let heads = x_dims[x_dims.len() - 1] / head_dim;
        if in_dims.iter().product::<usize>() / inner != seq * heads {
            continue;
        }
        let (cos, sin) = (node.inputs[1], node.inputs[2]);
        let Some(cos_dims) = all_static_dims(&graph.node(cos).shape) else {
            continue;
        };
        let cos_row_stride = rlx_ir::shape::rope_table_stride(&graph.node(cos).shape, n_rot);
        if cos_dims.len() != 2
            || cos_dims[0] != seq
            || cos_row_stride != head_dim / 2
            || !graph
                .node(sin)
                .shape
                .dims()
                .eq(graph.node(cos).shape.dims())
        {
            continue;
        }
        cands.push(Candidate {
            member: NormRopeMember {
                norm,
                rope: node.id,
                input,
                gamma,
                beta,
                heads: heads as u32,
            },
            seq: seq as u32,
            inner: inner as u32,
            cos,
            sin,
            cos_row_stride: cos_row_stride as u32,
            eps_bits: eps.to_bits(),
            norm_pos: ctx.order[&norm],
            rope_pos: ctx.order[&node.id],
            chain,
        });
    }
    cands.sort_by_key(|c| c.rope_pos);

    // An exact alias is safe for a read against a write of the same member: with
    // an exact alias every thread reads its own pair before writing it, and every
    // norm pass reads the whole row before the write pass starts — the rule the
    // residual+norm tee rests on too. A shifted overlap is the fatal case, and two
    // members of one dispatch are ordered by nothing, so a cross-member pair has to
    // be strictly disjoint.
    let disjoint_or_same =
        |a: Option<(u64, u64)>, b: Option<(u64, u64)>| a == b || spans_disjoint(a, b);

    let mut groups: HashMap<NodeId, NormRopeGroup> = HashMap::new();
    let mut skip: HashSet<NodeId> = HashSet::new();
    let index = PieceIndex::of(splits);
    let mut i = 0;
    while i < cands.len() {
        let first = &cands[i];
        // The nodes one fold owns: its norm, its rope and the views between them.
        let owned = |c: &Candidate| -> HashSet<NodeId> {
            c.chain
                .iter()
                .copied()
                .chain([c.member.norm, c.member.rope])
                .collect()
        };
        // The window starts at the earlier of the members' norms: an operand's
        // liveness ends at its own norm, whichever member that is.
        let mut members = vec![first.member.clone()];
        let mut own = owned(first);
        let mut last = i;
        if let Some(next) = cands.get(i + 1) {
            let same = next.seq == first.seq
                && next.inner == first.inner
                && next.cos == first.cos
                && next.sin == first.sin
                && next.cos_row_stride == first.cos_row_stride
                && next.eps_bits == first.eps_bits;
            let merged: HashSet<NodeId> = own.iter().copied().chain(owned(next)).collect();
            if same && ctx.quiet(first.norm_pos.min(next.norm_pos), next.rope_pos, &merged) {
                members.push(next.member.clone());
                own = merged;
                last = i + 1;
            }
        }
        if last == i && !ctx.quiet(first.norm_pos, first.rope_pos, &own) {
            i += 1;
            continue;
        }
        // One dispatch reads every operand while writing every rope's slot: each
        // written span must be disjoint (or exactly aliased) with its own member's
        // reads and strictly disjoint from every other member's, and strictly
        // disjoint from the other written spans.
        let writes: Vec<_> = members
            .iter()
            .map(|m| arena.storage_span(graph, m.rope))
            .collect();
        let reads: Vec<Vec<_>> = members
            .iter()
            .map(|m| {
                [m.input, m.gamma, m.beta]
                    .into_iter()
                    .chain([first.cos, first.sin])
                    .map(|id| arena.storage_span(graph, id))
                    .collect()
            })
            .collect();
        let sound = writes.iter().enumerate().all(|(j, &w)| {
            reads.iter().enumerate().all(|(k, rs)| {
                rs.iter().all(|&r| {
                    if k == j {
                        disjoint_or_same(r, w)
                    } else {
                        spans_disjoint(r, w)
                    }
                })
            }) && writes
                .iter()
                .enumerate()
                .all(|(k, &o)| k == j || spans_disjoint(o, w))
        });
        if sound {
            // The split the pair's operands are pieces of, if one dispatch can own
            // both: the members read the source row at their pieces' starts and
            // the leftover piece is copied, so the split keeps no step.
            let absorbed = match ctx.absorbed_split(&members, first.seq, first.inner, &index) {
                Some((split, emitter)) => {
                    // The group must still be there: an earlier pair may have
                    // claimed it, and no step writes a claimed piece.
                    let Some(gone) = splits.remove(&emitter) else {
                        i = last + 1;
                        continue;
                    };
                    split_skip.extend(gone.pieces.iter().copied());
                    Some(split)
                }
                None => None,
            };
            let emitter = members.last().expect("a group has a member").rope;
            for m in &members {
                skip.insert(m.norm);
                if m.rope != emitter {
                    skip.insert(m.rope);
                }
            }
            groups.insert(
                emitter,
                NormRopeGroup {
                    members,
                    seq: first.seq,
                    inner: first.inner,
                    cos: first.cos,
                    sin: first.sin,
                    cos_row_stride: first.cos_row_stride,
                    eps_bits: first.eps_bits,
                    split: absorbed,
                },
            );
        }
        i = last + 1;
    }
    (groups, skip)
}

/// The graph-level facts the norm+rope admission reads: node order, uses, graph
/// outputs, and the quiet-window rule that makes reading an operand past its
/// planned last reader sound.
struct NormRopeCtx<'a> {
    graph: &'a Graph,
    arena: &'a Arena,
    order: HashMap<NodeId, usize>,
    consumers: HashMap<NodeId, Vec<NodeId>>,
    outputs: HashSet<NodeId>,
}

/// The `detect_narrow_split_pattern` groups, by piece and by emitter, cloned so a
/// pair can find the split that fed it while the loop consumes groups out of
/// `splits`.
struct PieceIndex {
    owner: HashMap<NodeId, NodeId>,
    group: HashMap<NodeId, NarrowSplitGroup>,
}

impl PieceIndex {
    fn of(splits: &HashMap<NodeId, NarrowSplitGroup>) -> Self {
        Self {
            owner: splits
                .iter()
                .flat_map(|(e, g)| g.pieces.iter().map(move |p| (*p, *e)))
                .collect(),
            group: splits.iter().map(|(e, g)| (*e, g.clone())).collect(),
        }
    }
}

impl<'a> NormRopeCtx<'a> {
    fn new(graph: &'a Graph, arena: &'a Arena) -> Self {
        let order = graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();
        let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            for &input in &node.inputs {
                consumers.entry(input).or_default().push(node.id);
            }
        }
        Self {
            graph,
            arena,
            order,
            consumers,
            outputs: graph.outputs.iter().copied().collect(),
        }
    }

    /// The one node reading `id`, if that is the whole of its use and `id` is not
    /// a graph output — an output's slot must be written by a step, and this fold
    /// suppresses the step that would write it.
    fn sole_user(&self, id: NodeId, user: NodeId) -> bool {
        !self.outputs.contains(&id)
            && matches!(self.consumers.get(&id).map(Vec::as_slice), Some([only]) if *only == user)
    }

    /// Everything strictly between two positions must be the fold's own or write
    /// nothing, so no other step can write an operand between its planned death
    /// and the one dispatch that reads it.
    fn quiet(&self, from: usize, to: usize, owned: &HashSet<NodeId>) -> bool {
        self.graph.nodes()[from + 1..to].iter().all(|n| {
            owned.contains(&n.id)
                || rlx_compile::memory::is_pure_view(self.graph, n)
                || matches!(
                    n.op,
                    Op::Param { .. } | Op::Input { .. } | Op::Constant { .. }
                )
        })
    }

    /// The split that fed a norm+rope pair, when one dispatch can own both: each
    /// member's operand is one of the split's pieces (through its own line of
    /// pure views), and the piece the pair does not read is copied. The window the
    /// fold must own starts at the split's own emitter — the row's last planned
    /// reader — because the step now reads that row at the pair's position, and
    /// the source must be strictly disjoint from every write.
    ///
    /// `index` is the piece-keyed snapshot of the split groups, so the answer does
    /// not depend on which groups an earlier pair already consumed; the caller
    /// removes the group it absorbed.
    fn absorbed_split(
        &self,
        members: &[NormRopeMember],
        seq: u32,
        inner: u32,
        index: &PieceIndex,
    ) -> Option<(NormRopeSplit, NodeId)> {
        if members.len() != 2 {
            return None;
        }
        let mut pieces: Vec<(NodeId, u32, u32)> = Vec::with_capacity(2);
        for m in members {
            let mut cur = m.input;
            let mut prev = m.norm;
            let piece = loop {
                if !self.sole_user(cur, prev) {
                    return None;
                }
                let n = self.graph.node(cur);
                if rlx_compile::memory::is_pure_view(self.graph, n) && !n.inputs.is_empty() {
                    prev = cur;
                    cur = n.inputs[0];
                    continue;
                }
                let Op::Narrow { start, len, .. } = n.op else {
                    return None;
                };
                if (len as u64) != (seq as u64) * (m.heads as u64) * (inner as u64) {
                    return None;
                }
                break (cur, start as u32, len as u32);
            };
            pieces.push(piece);
        }
        let emitter = *index.owner.get(&pieces[0].0)?;
        if emitter != *index.owner.get(&pieces[1].0)? || pieces[0].0 == pieces[1].0 {
            return None;
        }
        let cut = index.group.get(&emitter)?;
        let mut leftover = cut
            .pieces
            .iter()
            .copied()
            .filter(|p| *p != pieces[0].0 && *p != pieces[1].0);
        let copy = match (leftover.next(), leftover.next()) {
            (None, _) => None,
            (Some(p), None) => match self.graph.node(p).op {
                Op::Narrow { start, len, .. } => Some((p, start as u32, len as u32)),
                _ => return None,
            },
            (Some(_), Some(_)) => return None,
        };
        if self.order[&emitter] >= self.order[&members[1].rope] {
            return None;
        }
        let mut owned: HashSet<NodeId> = cut.pieces.iter().copied().collect();
        for m in members {
            owned.extend([m.norm, m.rope]);
        }
        if !self.quiet(self.order[&emitter], self.order[&members[1].rope], &owned) {
            return None;
        }
        let src = self.arena.storage_span(self.graph, cut.src);
        let written: Vec<_> = members
            .iter()
            .map(|m| m.rope)
            .chain(copy.map(|(p, _, _)| p))
            .collect();
        if !written
            .iter()
            .all(|&w| spans_disjoint(src, self.arena.storage_span(self.graph, w)))
        {
            return None;
        }
        Some((
            NormRopeSplit {
                src: cut.src,
                starts: [pieces[0].1, pieces[1].1],
                copy,
            },
            emitter,
        ))
    }
}

/// The pieces one `Step::NarrowSplit` cuts from a single row, and the node it is
/// emitted at (the last piece, so every destination slot is already live and the
/// source is read at its planned last reader).
#[derive(Clone)]
pub(crate) struct NarrowSplitGroup {
    pub pieces: Vec<NodeId>,
    pub src: NodeId,
    pub total: u32,
}

/// A fused projection writes one `[1, q + k + v]` row and its three consumers
/// read it as three tensors; the decomposed form copies each piece with its own
/// `narrow` dispatch. This finds the 2–3 pieces of one row that cover its last
/// axis contiguously and turns them into one dispatch.
///
/// Admission is what makes one flat walk describe the cut: `outer == 1` (a single
/// row — with rows the pieces are strided), pieces contiguous from 0 and covering
/// the axis, the narrows adjacent in node order, every destination span disjoint
/// from the source's (an exact alias is allowed; each thread reads its own element
/// before writing it), and a real dtype — the kernel moves one f32 word per
/// element, so a complex row's packs need the plain `narrow`'s lane handling.
/// `on_device` is the walk's own routing answer, so a piece the plain form would
/// have staged on the host keeps its own step.
pub(crate) fn detect_narrow_split_pattern(
    graph: &Graph,
    arena: &Arena,
    on_device: bool,
) -> (HashMap<NodeId, NarrowSplitGroup>, HashSet<NodeId>) {
    // One compile-global answer for every piece: a walk that stages structure
    // work on the host keeps all of them.
    if !on_device {
        return (HashMap::new(), HashSet::new());
    }
    let order: HashMap<NodeId, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(node.id);
        }
    }
    let span = |id: NodeId| arena.storage_span(graph, id);

    let mut groups: HashMap<NodeId, NarrowSplitGroup> = HashMap::new();
    let mut skip: HashSet<NodeId> = HashSet::new();
    for node in graph.nodes() {
        let Some(cs) = consumers.get(&node.id) else {
            continue;
        };
        if cs.len() < 2 || cs.len() > 3 || node.shape.dtype().is_complex() {
            continue;
        }
        let Some(dims) = all_static_dims(&node.shape) else {
            continue;
        };
        if dims.is_empty() {
            continue;
        }
        let axis = dims.len() - 1;
        if dims[..axis].iter().product::<usize>() != 1 {
            continue;
        }
        let mut pieces: Vec<(usize, usize, NodeId)> = Vec::with_capacity(cs.len());
        let mut all = true;
        for &c in cs {
            let cn = graph.node(c);
            match cn.op {
                Op::Narrow {
                    axis: a,
                    start,
                    len,
                } if a == axis => pieces.push((start, len, c)),
                _ => {
                    all = false;
                    break;
                }
            }
        }
        if !all {
            continue;
        }
        pieces.sort_by_key(|p| p.0);
        let mut at = 0usize;
        for (start, len, _) in &pieces {
            if *start != at {
                all = false;
                break;
            }
            at += len;
        }
        if !all || at != dims[axis] {
            continue;
        }
        // Adjacent in node order: nothing else can write a destination between
        // its own slot's birth and the one dispatch that fills it.
        let mut adjacent = true;
        for w in pieces.windows(2) {
            if order[&w[1].2] != order[&w[0].2] + 1 {
                adjacent = false;
            }
        }
        if !adjacent {
            continue;
        }
        let src = span(node.id);
        let dsts: Vec<_> = pieces.iter().map(|(_, _, id)| span(*id)).collect();
        let sound = dsts.iter().enumerate().all(|(i, &d)| {
            (src == d || spans_disjoint(src, d))
                && dsts
                    .iter()
                    .enumerate()
                    .all(|(j, &o)| i == j || spans_disjoint(o, d))
        });
        if !sound {
            continue;
        }
        let emitter = pieces.last().expect("two pieces at least").2;
        for (_, _, id) in &pieces {
            if *id != emitter {
                skip.insert(*id);
            }
        }
        groups.insert(
            emitter,
            NarrowSplitGroup {
                pieces: pieces.iter().map(|(_, _, id)| *id).collect(),
                src: node.id,
                total: dims[axis] as u32,
            },
        );
    }
    (groups, skip)
}

/// Walk through Cast/Reshape nodes (which alias the underlying arena
/// slot, per `plan_f32_uniform`) to find whether `id` ultimately
/// refers to an `Op::Param`. AutoMixedPrecision wraps params in
/// Cast(F32→F16) nodes, so a literal `matches!(node.op, Op::Param)`
/// check on the matmul's `b_id` would miss the Cast(Param) case.
fn node_is_arena_param(param_offsets: &HashMap<String, NodeId>, id: NodeId) -> bool {
    param_offsets.values().any(|&nid| nid == id)
}

fn traces_to_param(graph: &Graph, mut id: NodeId) -> bool {
    loop {
        let node = graph.node(id);
        match &node.op {
            Op::Param { .. } => return true,
            Op::Cast { .. } | Op::Reshape { .. } | Op::Transpose { .. } => {
                if node.inputs.is_empty() {
                    return false;
                }
                id = node.inputs[0];
            }
            _ => return false,
        }
    }
}

/// Re-export of the planner's predicate. `lower.rs` calls it unqualified; the
/// definition lives in `rlx_compile::memory` so it cannot drift from the one
/// that decides which packs get pinned.
pub(crate) use rlx_compile::memory::is_static_weight_tensor;

fn tensor_is_graph_param(
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    id: NodeId,
) -> bool {
    node_is_arena_param(param_offsets, id) || traces_to_param(graph, id)
}

fn traces_to_input(graph: &Graph, mut id: NodeId) -> bool {
    loop {
        let node = graph.node(id);
        match &node.op {
            Op::Input { .. } => return true,
            Op::Cast { .. } | Op::Reshape { .. } => {
                if node.inputs.is_empty() {
                    return false;
                }
                id = node.inputs[0];
            }
            _ => return false,
        }
    }
}

/// Mirror A/B into the f16 shadow buffer before CoopF16Vk when the operand
/// is not already mirrored (Inputs/Params are written via `write_f32`).
fn schedule_uses_coop_f16_vk(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Matmul {
                compute_precision: MatmulCompute::CoopF16Vk,
                ..
            } | Step::MatmulQkv {
                kind: MatmulQkvKind::CoopF16Vk,
                ..
            }
        )
    })
}

fn register_coop_f16_vk_b_param(
    map: &mut HashMap<u32, String>,
    param_offsets: &HashMap<String, NodeId>,
    b_id: NodeId,
    b_off_f32: u32,
    compute: MatmulCompute,
) {
    if compute != MatmulCompute::CoopF16Vk {
        return;
    }
    for (name, &id) in param_offsets {
        if id == b_id {
            map.insert(b_off_f32, name.clone());
            return;
        }
    }
}

fn tensor_host_name(
    input_offsets: &HashMap<String, NodeId>,
    param_offsets: &HashMap<String, NodeId>,
    id: NodeId,
) -> String {
    for (name, &nid) in input_offsets {
        if nid == id {
            return name.clone();
        }
    }
    for (name, &nid) in param_offsets {
        if nid == id {
            return name.clone();
        }
    }
    panic!("rlx-wgpu: CoopF16Vk host activation source {id} is not an input or param");
}

fn host_tensor_f32<'a>(
    name: &str,
    inputs: &'a [(&str, &[f32])],
    stashed_params: &'a HashMap<String, Vec<f32>>,
) -> Option<&'a [f32]> {
    inputs
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, d)| *d)
        .or_else(|| stashed_params.get(name).map(|v| v.as_slice()))
}

fn apply_activation_host(act: Activation, data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&x| match act {
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Exp => x.exp(),
            Activation::Log => x.ln(),
            Activation::Sqrt => x.sqrt(),
            Activation::Rsqrt => 1.0 / x.sqrt(),
            Activation::Neg => -x,
            Activation::Abs => x.abs(),
            Activation::Gelu | Activation::GeluApprox => {
                let c = 0.797_884_6_f32;
                let x3 = x * x * x;
                let inner = (c * (x + 0.044_715 * x3)).clamp(-15.0, 15.0);
                0.5 * x * (1.0 + inner.tanh())
            }
            Activation::Silu => {
                let nx = (-x).clamp(-88.0, 88.0);
                x / (1.0 + nx.exp())
            }
            Activation::Round => x.round(),
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            Activation::Tan => x.tan(),
            Activation::Atan => x.atan(),
            Activation::Recip => 1.0 / x,
            Activation::Floor => x.floor(),
            Activation::Ceil => x.ceil(),
            Activation::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Activation::Softplus => x.max(0.0) + (-(x.abs())).exp().ln_1p(),
            Activation::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::Erf => {
                let s = x.signum();
                let ax = x.abs();
                let t = 1.0 / (1.0 + 0.327_591_1 * ax);
                s * (1.0
                    - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_74)
                        * t
                        + 0.254_829_6)
                        * t)
                        * (-ax * ax).exp())
            }
            Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            Activation::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
            Activation::Mish => x * (x.max(0.0) + (-(x.abs())).exp().ln_1p()).tanh(),
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::LogSigmoid => x.min(0.0) - (-(x.abs())).exp().ln_1p(),
        })
        .collect()
}

/// Activation node ids consumed as CoopF16Vk matmul A/B operands.
fn collect_coop_f16_vk_mirror_activations(graph: &Graph, dev: &wgpu::Device) -> HashSet<NodeId> {
    let mut acts = HashSet::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        let a_id = node.inputs[0];
        let b_id = node.inputs[1];
        let a_shape = graph.node(a_id).shape.dims();
        let b_shape = graph.node(b_id).shape.dims();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            continue;
        }
        let m = a_shape[0].unwrap_static() as u32;
        let k = a_shape[1].unwrap_static() as u32;
        let n = b_shape[1].unwrap_static() as u32;
        if !coop_f16_vk_eligible(dev, m, k, n) || !traces_to_param(graph, b_id) {
            continue;
        }
        if matches!(graph.node(a_id).op, Op::Activation(_)) {
            acts.insert(a_id);
        }
        if matches!(graph.node(b_id).op, Op::Activation(_)) {
            acts.insert(b_id);
        }
    }
    acts
}

/// When A/B are computed (not Input/Param), mirror f32 arena into f16 shadow
/// via `cast_f32_to_f16` before CoopF16Vk matmul (non-activation intermediates).
fn maybe_push_coop_f16_vk_casts(
    graph: &Graph,
    a_id: NodeId,
    b_id: NodeId,
    mirror_acts: &HashSet<NodeId>,
    device: &wgpu::Device,
    arena: &Arena,
    schedule: &mut Vec<Step>,
    uniforms: &mut Vec<wgpu::Buffer>,
    bind_groups: &mut Vec<wgpu::BindGroup>,
    mm_cast: &Option<&'static Kernel>,
    compute_precision: MatmulCompute,
    a_off_f32: u32,
    m: u32,
    k: u32,
    batch: u32,
    b_off_f32: u32,
    n: u32,
) {
    if compute_precision != MatmulCompute::CoopF16Vk {
        return;
    }
    let batch_n = batch.max(1);
    if !traces_to_input(graph, a_id)
        && !traces_to_param(graph, a_id)
        && !mirror_acts.contains(&a_id)
    {
        let a_elems = m.saturating_mul(k).saturating_mul(batch_n);
        let (base, size) = arena_window_for_nodes(device, arena, &[a_id]);
        push_cast_f32_to_f16_step(
            device,
            arena,
            base,
            size,
            schedule,
            uniforms,
            bind_groups,
            mm_cast,
            None,
            0,
            a_off_f32,
            a_elems,
        );
    }
    if !traces_to_input(graph, b_id)
        && !traces_to_param(graph, b_id)
        && !mirror_acts.contains(&b_id)
    {
        let b_elems = k.saturating_mul(n).saturating_mul(batch_n);
        let (base, size) = arena_window_for_nodes(device, arena, &[b_id]);
        push_cast_f32_to_f16_step(
            device,
            arena,
            base,
            size,
            schedule,
            uniforms,
            bind_groups,
            mm_cast,
            None,
            0,
            b_off_f32,
            b_elems,
        );
    }
}

fn build_matmul_qkv_coop_f16_vk_bind_group(
    device: &wgpu::Device,
    mqk: &Kernel,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    params: &wgpu::Buffer,
    k: u32,
    n: u32,
    b_off: u32,
) -> (wgpu::BindGroup, u32) {
    let f16_buf = arena
        .f16_buffer
        .as_ref()
        .expect("CoopF16Vk QKV requires SHADER_F16 f16 shadow arena");
    let (f16_res, rebased_b) = {
        let (base, size, rebased) =
            f16_weight_bind_range(device, f16_buf.size(), b_off, k, n, 1, 0);
        (
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: f16_buf,
                offset: base,
                size: NonZeroU64::new(size),
            }),
            rebased,
        )
    };
    (
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_qkv_coop_f16_vk bg"),
            layout: &mqk.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: f16_res,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: {
                        let (b, l) = arena_bind_buf(arena, arena_base);
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b,
                            offset: l,
                            size: NonZeroU64::new(arena_size),
                        })
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        }),
        rebased_b,
    )
}
/// Append a CastF32ToF16 pre-pass: mirrors `arena[off..off+len]` (f32) into an
/// f16 buffer so coop matmul kernels can read operands as f16.
///
/// `dst` names the buffer and the f16-element offset to write at; `None` is the
/// weight mirror (`arena.f16_buffer`), which is indexed by the same logical
/// element as the arena window, so the offset is the source's own. The dense
/// attention route passes its own operand buffer, whose regions are packed from
/// element 0 (see the route's arm in `compile::lower`).
#[allow(clippy::too_many_arguments)]
fn push_cast_f32_to_f16_step(
    device: &wgpu::Device,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    schedule: &mut Vec<Step>,
    uniforms: &mut Vec<wgpu::Buffer>,
    bind_groups: &mut Vec<wgpu::BindGroup>,
    mm_cast: &Option<&'static Kernel>,
    dst: Option<(&wgpu::Buffer, u32)>,
    scale_bits: u32,
    src_off: u32,
    len: u32,
) {
    let kernel = match mm_cast {
        Some(k) => *k,
        None => return, // device lacks SHADER_F16; fall through, dispatch will skip
    };
    let mirror = arena.f16_buffer.as_ref();
    let (f16_buf, f16_base, f16_size) = match dst {
        Some((b, _)) => (b, 0, b.size()),
        None => {
            let Some(b) = mirror else { return };
            let (base, size) = f16_shadow_bind_range(arena_base, arena_size, b.size());
            (b, base, size)
        }
    };
    let src_window = match dst {
        // The mirror is indexed by the arena's own logical element (see
        // `f16_shadow_bind_range`), so the source index is rebased against the
        // arena window's base.
        None => src_off.saturating_sub((arena_base / 4) as u32),
        // The route's side buffer binds its own window, so the source index is the
        // arena window's own.
        Some(_) => src_off,
    };
    let p = CastF32ToF16Params {
        src_off: src_window,
        len,
        dst_off: dst.map_or(src_window, |(_, off)| off),
        scale_bits,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu cast_f32_to_f16 uniform"),
        size: std::mem::size_of::<CastF32ToF16Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Write params at compile (kernel doesn't depend on active extent).
    let dev = wgpu_device().expect("rlx-wgpu: device gone");
    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu cast_f32_to_f16 bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: f16_buf,
                    offset: f16_base,
                    size: NonZeroU64::new(f16_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: arena_bind_buf(arena, arena_base).0,
                    offset: arena_bind_buf(arena, arena_base).1,
                    size: NonZeroU64::new(arena_size),
                }),
            },
        ],
    });
    schedule.push(Step::CastF32ToF16 { params: p });
    uniforms.push(u);
    bind_groups.push(bg);
}

/// Per-Matmul-step bind group builder. Returns `(bind_group, rebased_b_off)`;
/// `rebased_b_off` adjusts `MatmulParams.b_off` when the f16 weight buffer is
/// window-bound.
///
/// `scratch_b16` is the dense attention route's own f16 operand buffer, bound
/// whole at offset 0 for [`MatmulCompute::CoopF16F32AccScratch`]'s B — the one
/// caller that has no weight operand to bind instead. Every other caller passes
/// `None`.
#[allow(clippy::too_many_arguments)]
fn build_matmul_bind_group(
    device: &wgpu::Device,
    mm_k: &Kernel,
    _mm_w: &Kernel,
    mm_f16w: &Option<&'static Kernel>,
    mm_bf16w: &'static Kernel,
    mm_f16c: &Option<&'static Kernel>,
    mm_coop: &Option<&'static Kernel>,
    mm_coop_f32: &Option<&'static Kernel>,
    scratch_b16: Option<&wgpu::Buffer>,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    params: &wgpu::Buffer,
    b_is_param: bool,
    compute_precision: MatmulCompute,
    k: u32,
    n: u32,
    batch: u32,
    b_off: u32,
    b_batch_stride: u32,
) -> (wgpu::BindGroup, u32) {
    // PACKED-BF16 weight: 3-binding layout (arena rw for A/C, params, packed
    // bf16 weight buffer bound WHOLE at offset 0 so `b_off` stays the global
    // bf16-element index). No f16 shadow / rebase.
    if b_is_param
        && compute_precision == MatmulCompute::Bf16Packed
        && let Some(bf16_buf) = &arena.bf16_weight_buffer
    {
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_bf16w bg"),
                layout: &mm_bf16w.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bf16_buf.as_entire_binding(),
                    },
                ],
            }),
            b_off,
        );
    }
    let f16_bind = |b_off: u32| -> (wgpu::BindingResource<'_>, u32) {
        let f16_buf = arena
            .f16_buffer
            .as_ref()
            .expect("f16 weight bind without f16_buffer");
        let (base, size, rebased) =
            f16_weight_bind_range(device, f16_buf.size(), b_off, k, n, batch, b_batch_stride);
        (
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: f16_buf,
                offset: base,
                size: NonZeroU64::new(size),
            }),
            rebased,
        )
    };
    if compute_precision == MatmulCompute::CoopF16Vk
        && let (Some(coop_vk), Some(_f16_buf)) =
            (matmul_coop_f16_vulkan_kernel(device), &arena.f16_buffer)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f16_vulkan bg"),
                layout: &coop_vk.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: f16_res,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            }),
            rebased_b,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::CoopF32
        && let Some(coop_f32) = mm_coop_f32
    {
        // 2-binding layout — both A and B come from the f32 arena
        // (no f16 shadow buffer needed for the pure-f32 path). The `b_is_param`
        // term above is this path's operand order, NOT a requirement of the
        // kernel: the convolution GEMM routes reach CoopF32 weight-in-A and bind
        // through `bind_arena_window`, so this branch deliberately does not
        // mirror `run.rs`'s role-free CoopF32 dispatch guard.
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f32 bg"),
                layout: &coop_f32.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            }),
            b_off,
        );
    }
    if compute_precision == MatmulCompute::CoopF16F32AccScratch
        && let (Some(b16), Some(f16acc)) = (
            scratch_b16,
            crate::kernels::matmul_coop_f16_f32acc_kernel(device),
        )
    {
        // The sibling's 3-binding layout with B bound from the route's own f16
        // operand buffer, whole at offset 0 so `b_off` stays that buffer's element
        // index. A is read from the f32 arena and downcast inside the kernel, as
        // in `CoopF16F32Acc`.
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f16_f32acc scratch bg"),
                layout: &f16acc.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b16.as_entire_binding(),
                    },
                ],
            }),
            b_off,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::CoopF16F32Acc
        && let (Some(_f16_buf), Some(f16acc)) = (
            &arena.f16_buffer,
            crate::kernels::matmul_coop_f16_f32acc_kernel(device),
        )
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        // The same 3-binding layout as `matmul_coop16`: A is read from the f32
        // arena and downcast inside the kernel, so it has no binding of its own.
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f16_f32acc bg"),
                layout: &f16acc.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    }, // weights
                ],
            }),
            rebased_b,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::Coop16
        && let (Some(_f16_buf), Some(coop)) = (&arena.f16_buffer, mm_coop)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        // 3-binding layout — A is staged from arena (f32) through
        // workgroup-shared memory inside the kernel, no separate
        // f16 binding for A.
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop16 bg"),
                layout: &coop.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    }, // weights
                ],
            }),
            rebased_b,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::F16
        && let (Some(_f16_buf), Some(f16c)) = (&arena.f16_buffer, mm_f16c)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_f16_compute bg"),
                layout: &f16c.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    },
                ],
            }),
            rebased_b,
        );
    }
    let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
    if b_is_param
        && f16w_opt_in
        && let (Some(_f16_buf), Some(f16w)) = (&arena.f16_buffer, mm_f16w)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_f16w bg"),
                layout: &f16w.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena_bind_buf(arena, arena_base).0,
                            offset: arena_bind_buf(arena, arena_base).1,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    },
                ],
            }),
            rebased_b,
        );
    }
    (
        bind_arena_window(device, mm_k, arena, arena_base, arena_size, params),
        b_off,
    )
}

/// The two conv-route decisions, pinned without a device: the 1×1 route's frozen
/// bar and the shape/arena split of the im2col decision, plus the col reservation
/// the planner sizes from that split. The route *tests* observe which route a
/// shape took on an adapter; these pin the arithmetic under it, and run in the
/// plain `cargo test` gate rather than only under `--features backend-tests`.
/// The rows are sized from the *effective* floors (`im2col_min_*()`, the env the
/// lowering reads), so they hold at the shipped values and across the lever's measured
/// range (`RLX_WGPU_IM2COL_MIN_SPATIAL` 100..48000); a floor far outside that range is
/// outside this claim. A row that deliberately pins a shipped constant says so in place.
#[cfg(test)]
mod conv_route_decision_tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    /// The 1×1 admission is its own frozen bar: it must not follow the im2col
    /// spatial floor, which moved under the non-pointwise shapes of the same graphs.
    #[test]
    fn the_pointwise_bar_is_frozen_against_the_im2col_floors() {
        // 128 × 3000 = 384 000 — under this bar, but over what the two floors admit
        // once the spatial floor is 100 (256 × 100 = 25 600), so an admission test
        // re-coupled to them would take this shape.
        assert!(!pointwise_admitted(128, 3000));
        assert!(pointwise_admitted(256, 2048), "the bar itself");
        assert!(!pointwise_admitted(256, 2047));
        // The decoder body's 1×1 form (`[1,768,1,800]`) was over the bar and stays.
        assert!(pointwise_admitted(768, 800));
    }

    /// The col reservation is sized from the shapes the floor admits and dropped by
    /// the two caps the planner applies, over a tail floored at 64 MiB. The binding
    /// half is driven with a caller-supplied limit, so it is reachable without an
    /// adapter under the arena.
    #[test]
    fn the_col_reservation_follows_the_shape_and_the_binding_window() {
        let (band, col_elems) = band_graph();
        let col_elems = col_elems.expect("the band shape clears the effective floors");
        let col_bytes = col_elems * 4;
        // The planner hands the panel back 256-byte aligned, so the row is compared
        // against the aligned figure: a K or spatial floor that makes the col a
        // non-multiple of the alignment would otherwise fail a rule that is not in doubt.
        let aligned = col_bytes.div_ceil(256) * 256;
        let planned = 1u64 << 20;
        assert_eq!(
            conv_im2col_scratch_bytes(&band, planned as usize, planned + aligned, false),
            aligned as usize,
            "a window with room for the panel reserves exactly it, 256-byte aligned"
        );
        assert_eq!(
            conv_im2col_scratch_bytes(&band, planned as usize, planned + col_bytes - 1, false),
            0,
            "a window one byte short of the panel drops it"
        );
        assert_eq!(
            conv_im2col_scratch_bytes(&band, planned as usize, u64::MAX, false),
            aligned as usize,
            "an unlimited window reserves it whatever the plan costs"
        );
        // The hard cap drops a col past it even where the window would take it. The map is
        // `(2 × the spatial floor).max(64)` wide — 200 at the shipped 100 — with enough
        // channels to clear the K floor and to push the panel past the cap at that extent.
        let w = (im2col_min_spatial() * 2).max(64) as usize;
        let c_in = (im2col_min_k().div_ceil(25) as usize)
            .max((CONV_IM2COL_MAX_COL_BYTES as usize).div_ceil(25 * w * 4));
        let (huge, huge_elems) =
            conv_graph(c_in, im2col_min_cout() as usize, [1, w], [5, 5], [2, 2]);
        assert!(huge_elems.expect("floor-clearing shape") * 4 > CONV_IM2COL_MAX_COL_BYTES);
        assert_eq!(conv_im2col_scratch_bytes(&huge, 0, u64::MAX, false), 0);
        // …and the tail the panel is reserved in is floored at 64 MiB, the base every
        // recorded `expected_col_delta` in the route harness is measured against.
        assert_eq!(compute_scratch_bytes(&band), 64 * 1024 * 1024);
    }

    /// The implicit class's channel bounds are the entry's own: a ceiling because it
    /// carries three accumulator sets (an output one row past three tiles would have
    /// rows no set covers) and a floor because below a tile's rows it multiplies
    /// output the conv does not have. The kernel guards neither, so the refusal has
    /// to happen here.
    #[test]
    fn a_conv_outside_the_implicit_class_channel_bounds_is_refused() {
        let shape = |c_in, c_out| ConvImplicitShape {
            c_in,
            c_out,
            kh: 1,
            kw: 7,
            h: 1,
            h_out: 1,
            sw: 1,
            ph: 0,
            groups: 1,
            n: 1,
            spatial: conv_implicit_min_spatial(),
        };
        assert!(
            shape(96, 96).admits(),
            "three whole row tiles is the ceiling"
        );
        assert!(
            shape(96, 4).admits(),
            "four rows is the narrowest output the entry is admitted for"
        );
        for (c_in, c_out) in [(97, 96), (96, 97), (96, 104), (96, 3), (96, 1)] {
            assert!(
                !shape(c_in, c_out).admits(),
                "{c_in}/{c_out} is outside the channel bounds the entry keeps"
            );
        }
    }

    /// The class's cap restates what the entry covers, so the numbers cannot drift:
    /// `ROW_TILES` accumulator sets of `TILE_M` rows are exactly the cap, and the
    /// epilogue stores one row tile per set. Both are read out of the shader source,
    /// because the kernel guards neither and the host cap is all that keeps a wider
    /// output away from it.
    #[test]
    fn the_implicit_class_cap_is_the_entrys_row_tiles() {
        let src = crate::kernels::CONV1D_IMPLICIT_WGSL;
        assert!(src.contains("const ROW_TILES: u32 = 3u;"), "three sets");
        assert!(src.contains("const TILE_M: u32 = 32u;"), "32 rows a set");
        let stores = src
            .lines()
            .filter(|l| l.trim_start().starts_with("copy_out("))
            .count();
        assert_eq!(
            stores, 3,
            "the epilogue stores one row tile per accumulator set"
        );
        assert_eq!(3 * 32, CONV_IMPLICIT_MAX_C_OUT, "the cap is that coverage");
    }

    /// The spatial floor's rule, at its own boundary: the quantity is one conv
    /// node's own output spatial extent (`h_out * w_out`, the output length for the
    /// unit-height lift) and the comparison is a bare `u64` `>=`, so the extent equal
    /// to the floor is admitted and one shorter is refused — no multiplier, margin or
    /// rounding. Sized from the effective floors, so the documented
    /// `RLX_WGPU_IM2COL_MIN_SPATIAL` A/B lever moves both rows with it.
    #[test]
    fn the_spatial_floor_admits_exactly_its_own_extent() {
        let floor = im2col_min_spatial() as usize;
        if floor < 2 {
            // A floor of 0 or 1 admits every extent there is, so there is no two-sided
            // boundary to state, and `floor - 1` would be a degenerate width.
            return;
        }
        let c_in = (im2col_min_k().div_ceil(7) as usize).max(64);
        let c_out = im2col_min_cout() as usize;
        let (_, on_the_floor) = conv_graph(c_in, c_out, [1, floor], [1, 7], [0, 3]);
        assert_eq!(
            on_the_floor,
            Some(c_in as u64 * 7 * floor as u64),
            "out {floor} is the floor itself"
        );
        let (_, one_short) = conv_graph(c_in, c_out, [1, floor - 1], [1, 7], [0, 3]);
        assert_eq!(one_short, None, "one output shorter is under the floor");
    }

    /// The reservation and the 1×1 route are two decisions, which is what lets the
    /// spatial floor move without moving the pointwise shapes: the same 1×1 form is
    /// admitted by its own bar and reserves no col panel at all.
    #[test]
    fn a_one_by_one_conv_reserves_no_col_panel() {
        let mut g = Graph::new("pointwise_col");
        let x = g.input("x", Shape::new(&[1, 768, 1, 800], DType::F32));
        let w = g.param("w", Shape::new(&[768, 768, 1, 1], DType::F32));
        let y = g.conv2d(x, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
        let node = g.node(y);
        assert_eq!(
            conv_im2col_col_elems(
                &node.op,
                &g.node(node.inputs[0]).shape,
                &g.node(node.inputs[1]).shape,
                &node.shape,
                false,
            ),
            None,
            "1×1 taps are not the im2col route's shape"
        );
        assert_eq!(conv_im2col_scratch_bytes(&g, 1 << 20, u64::MAX, false), 0);
    }

    /// A conv the cooperative implicit route takes reserves no panel: its own
    /// `[1, 96, 1, l]` dilated form clears every im2col floor, so the reservation
    /// would book `k_total · spatial · 4` for a matrix the route never writes —
    /// and books nothing for it once the route is live.
    #[test]
    fn a_conv_the_implicit_route_takes_reserves_no_panel() {
        let (g, elems) = conv_graph(96, 96, [1, 140_000], [1, 7], [0, 3]);
        let panel = elems.expect("the class's shape clears the im2col floors") * 4;
        assert_eq!(
            conv_im2col_scratch_bytes(&g, 0, u64::MAX, false),
            panel as usize,
            "with the implicit route off the panel is booked, as it was"
        );
        assert_eq!(
            conv_im2col_scratch_bytes(&g, 0, u64::MAX, true),
            0,
            "and a route that gathers through the taps books none"
        );
    }

    /// The direct cooperative entry's admission, term by term: the route term keeps
    /// it off the ordinary multiply, the epilogue terms off a multiply with a bias or
    /// an activation, and the shape terms off a tile the kernel's unguarded 32×32
    /// store or its whole-`TILE_K` legs would write or read past.
    #[test]
    fn the_direct_coop_f32_entry_is_admitted_by_its_own_terms() {
        assert!(
            coop_f32_direct_admitted(0, 0xFFFF, 64, 128, 64, false),
            "the bar"
        );
        // A weight in the second operand is the ordinary multiply's cascaded form,
        // which keeps `matmul_coop_f32`'s staging and scalar epilogue.
        assert!(!coop_f32_direct_admitted(0, 0xFFFF, 64, 128, 64, true));
        assert!(!coop_f32_direct_admitted(1, 0xFFFF, 64, 128, 64, false));
        assert!(!coop_f32_direct_admitted(0, 1, 64, 128, 64, false));
        assert!(!coop_f32_direct_admitted(0, 0xFFFF, 63, 128, 64, false));
        assert!(!coop_f32_direct_admitted(0, 0xFFFF, 64, 127, 64, false));
        assert!(!coop_f32_direct_admitted(0, 0xFFFF, 64, 128, 33, false));
    }

    /// A conv graph, and the col element count `conv_im2col_col_elems` gives its
    /// shape (`None` where the shape clears no floor) — the planner's own function,
    /// so these tests and the planner cannot drift apart.
    fn conv_graph(
        c_in: usize,
        c_out: usize,
        hw: [usize; 2],
        k: [usize; 2],
        p: [usize; 2],
    ) -> (Graph, Option<u64>) {
        let mut g = Graph::new("conv_col_reservation");
        let x = g.input("x", Shape::new(&[1, c_in, hw[0], hw[1]], DType::F32));
        let w = g.param("w", Shape::new(&[c_out, c_in, k[0], k[1]], DType::F32));
        let y = g.conv2d(x, w, k, [1, 1], p, [1, 1], 1);
        let node = g.node(y);
        let elems = conv_im2col_col_elems(
            &node.op,
            &g.node(node.inputs[0]).shape,
            &g.node(node.inputs[1]).shape,
            &node.shape,
            false,
        );
        (g, elems)
    }

    /// A 2-D map that clears the effective floors with the smallest side there is —
    /// `ceil(sqrt(spatial floor))`, so it sits just inside the im2col band — and at
    /// least 64 channels, above the K floor. Derived rather than taken from the route
    /// harness's fixture tables: the reservation it sizes follows the map's area and
    /// the kernel's taps, not the fixture's geometry.
    fn band_graph() -> (Graph, Option<u64>) {
        let side = (im2col_min_spatial() as f64).sqrt().ceil() as usize;
        let c_in = (im2col_min_k().div_ceil(9) as usize).max(64);
        conv_graph(
            c_in,
            im2col_min_cout() as usize,
            [side, side],
            [3, 3],
            [1, 1],
        )
    }
}
