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
        // (`attention_route`). The 64 MiB floor below already covers the route's
        // own cap, so what this branch raises the reserve by is the plan's
        // ≤ 256-byte alignment slack — which is exactly what lets a route shape
        // fit without depending on the bump cursor's own alignment.
        let attn = match attention_route::plan_attention(graph, node) {
            Ok(plan) if plan.decompose => plan.scratch_bytes(),
            _ => 0,
        };
        if attn > max_bytes {
            max_bytes = attn;
        }
    }
    // Reserve extra scratch for staging small far-apart operands when the
    // arena exceeds wgpu's binding window. This keeps compile-time simple
    // and avoids per-op scratch sizing plumbing.
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
/// mirrored 1-D tiled lift, and the 1×1 pointwise GEMM. Default **on**;
/// `RLX_WGPU_CONV_IM2COL=0` restores the plain routing and the lean arena
/// exactly, which is what keeps a bisect of that flip possible. The same
/// predicate gates the col-scratch reservation, which is decided from a conv's
/// *shape* alone (see [`conv_im2col_col_elems`]): a graph whose convs do not
/// qualify by shape reserves nothing, while one whose conv qualifies pays that
/// reservation even where the route then does not run — a parked weight, a conv
/// its adapter routes to the host, a sharded arena.
pub(crate) fn conv_routes_enabled() -> bool {
    rlx_ir::env::flag_or("RLX_WGPU_CONV_IM2COL", true)
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
/// Returns `None` otherwise. Mirrors the shape math in the `Op::Conv` compile
/// arm so scratch sizing and lowering agree on which convs are routed.
fn conv_im2col_col_elems(
    op: &Op,
    in_shape: &rlx_ir::Shape,
    w_shape: &rlx_ir::Shape,
    out_shape: &rlx_ir::Shape,
) -> Option<u64> {
    let Op::Conv {
        kernel_size,
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
    Some(k_total * spatial)
}

/// Extra arena scratch (bytes, 256-aligned) needed to hold the largest conv
/// col matrix that will be routed through im2col+GEMM. Capped so the whole
/// arena still binds in a single storage-binding window. Returns 0 when no
/// conv qualifies or none fits.
fn conv_im2col_scratch_bytes(graph: &Graph, planned_arena_size: usize, max_binding: u64) -> usize {
    let mut max_col_bytes: u64 = 0;
    for node in graph.nodes() {
        if node.inputs.len() < 2 {
            continue;
        }
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let Some(elems) = conv_im2col_col_elems(&node.op, in_shape, w_shape, &node.shape) else {
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
    /// arena once shapes resolve.
    pending_params: HashMap<String, Vec<f32>>,
    pending_param_bytes: HashMap<String, Vec<u8>>,
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
    /// blocking output readback (results stay in the arena). Used by the wasm
    /// `run_async` path, which then reads outputs back asynchronously — the
    /// browser event loop cannot be blocked. Always false on native.
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
    static_once_done: bool,
    /// Steps the per-binding-ceiling rule routed to the host route. Non-zero
    /// means the graph's arena exceeded the device's per-binding limit and at
    /// least one absolute-addressing op was computed on the host instead of
    /// through a re-based window. See `lower::should_host_at_ceiling`.
    ceiling_hosted: usize,
    /// Convs whose shape qualified for im2col but which took the direct/tiled route
    /// for a reason the shapes alone do not settle — counted where that route is
    /// decided, in the `Op::Conv` arm of [`crate::backend::compile`], so a term added
    /// there cannot leave the count behind.
    im2col_fallbacks: usize,
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
}

mod attention_route;
mod compile;
mod copy_route;
mod dispatch;
mod helpers;
mod run;
mod run_report;
mod set;
/// Static-string label for each Step variant — used by the Perfetto
/// trace layer (PLAN L3) to mark per-step events without allocating.
mod step;
mod test;

pub(crate) use copy_route::*;
pub(crate) use helpers::*;
pub(crate) use run_report::*;
pub(crate) use step::*;

/// The many-row attention route's query-row floor: public because the crate's
/// integration tests are external and read it to place their shapes on either
/// side of it.
pub use attention_route::ATTENTION_DECOMPOSE_MIN_Q;

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

        // No-op if shapes haven't changed since the last compile.
        if let Some(prev) = &self.last_binding
            && same_binding(prev, &binding)
        {
            return;
        }

        // Resolve and recompile.
        let resolved = bind_graph(unresolved, &binding);
        let original = self.unresolved.take();
        let pending_params = std::mem::take(&mut self.pending_params);
        let pending_bytes = std::mem::take(&mut self.pending_param_bytes);

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
        self.attention_window_declines = fresh.attention_window_declines;
        self.input_offsets = fresh.input_offsets;
        // The arena is this compile's, and a resident handle's host mirror was
        // emptied when its rows became resident — so this executable no longer
        // holds those rows, and must not answer as if it did.
        self.gpu_handle_resident.clear();
        self.param_offsets = fresh.param_offsets;
        self.uniforms = fresh.uniforms;
        self.bind_groups = fresh.bind_groups;
        self.meta_buffers = fresh.meta_buffers;
        self.unresolved = original;
        self.last_binding = Some(binding);
        // Recompiled — uniforms are now empty buffers; force re-write
        // on next run().
        self.uniforms_active_extent = None;
        self.coop_f16_vk = fresh.coop_f16_vk;
        self.coop_f16_b_param = fresh.coop_f16_b_param;
        self.coop_f16_vk_wide_bind_groups = fresh.coop_f16_vk_wide_bind_groups;
        self.coop_f16_host_activations = fresh.coop_f16_host_activations;

        // Replay pending param uploads against the new arena.
        for (name, data) in pending_params {
            self.set_param(&name, &data);
        }
        for (name, data) in pending_bytes {
            self.set_param_bytes(&name, &data);
        }
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
            static_once_done: false,
            ceiling_hosted: 0,
            im2col_fallbacks: 0,
            attention_window_declines: 0,
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
    /// `Op::Conv` arm of [`crate::backend::compile`], where the count is taken).
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

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(id) = self.resolve_handle_slot(name) {
            let dev = wgpu_device().expect("rlx-wgpu: device gone");
            return Some(self.arena.read_f32(&dev.device, &dev.queue, id));
        }
        self.gpu_handles.get(name).cloned()
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
        if row_inner == 0 {
            return None;
        }
        // Checked, because this is release code: a row index near `usize::MAX`
        // must be refused rather than wrap into a range inside the slot.
        let start = row.checked_mul(row_inner)?;
        let end = start.checked_add(row_inner)?;
        let Some(id) = self.resolve_handle_slot(name) else {
            // No slot here: the host mirror, bounded by what the name declares —
            // it may be a longer binding than the slot it was uploaded into.
            let mirror = self.gpu_handles.get(name)?;
            let declared = self.declared_extent_elems(*self.input_offsets.get(name)?, mirror.len());
            if end > declared {
                return None;
            }
            return Some(LocatedRow::Mirror(&mirror[start..end]));
        };
        if end > self.extent_elems(id) {
            return None;
        }
        Some(LocatedRow::Arena {
            byte_off: self.arena.offset(id) + start * std::mem::size_of::<f32>(),
            len: row_inner * std::mem::size_of::<f32>(),
        })
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
            let dev = wgpu_device().expect("rlx-wgpu: device gone");
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

fn dispatch_dims(threads_total: u32, workgroup_size: u32) -> (u32, u32, u32) {
    let groups = threads_total.div_ceil(workgroup_size);
    if groups <= MAX_WORKGROUPS_PER_DIM {
        (groups, 1, 1)
    } else {
        let gx = MAX_WORKGROUPS_PER_DIM;
        let gy = groups.div_ceil(gx);
        (gx, gy, 1)
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
    bind_two_buf0_window(device, kernel, buf, local, size, params)
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
        _p0: 0,
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
    let mut size = need.div_ceil(ALIGN) * ALIGN;
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
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu bg window"),
        layout: &kernel.bgl,
        entries: &[
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
        ],
    })
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
    // in the FIRST operand, and no packed form to keep; see
    // [`conv_gemm`].
    let coop_f32_ok = b_is_param && coop_f32_admitted(dev, k, n, any_bf16 && b_is_param);

    let eligible = MatmulEligibility {
        coop16: coop16_ok,
        coop_f16_vk: coop_f16_vk_ok,
        coop_f32: coop_f32_ok,
        // The plain tiled f32 kernel has no alignment or feature requirement —
        // it is the path everything else falls back to, so it is always legal.
        f32: true,
    };

    // ── The cascade: preference order among the eligible ────────────────────
    let default = if coop16_ok {
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

/// Whether the f32 cooperative-matrix kernel may serve a multiply with this
/// contraction (`k`) and output-column (`n`) extent on this device.
///
/// Shared by the ordinary matmul cascade and the two convolution GEMM routes
/// ([`conv_gemm`]), so every lever governs both: `RLX_WGPU_NO_COOP_F32` opts
/// out, `RLX_WGPU_FORCE_COOP_F32` widens admission to adapters that report no
/// 8×8 f32 cooperative matrices, `RLX_WGPU_MATMUL_F32_ONLY` pins the whole
/// matmul machinery to F32 (read here as well as in the cascade, so the conv
/// routes obey it too), and the last term is the per-device operand-order probe.
/// `bf16_weight_in_b` is the one term that is the ordinary multiply's alone — the
/// conv callers pass `false` for it. The backend term is the other way round: the
/// conv callers ask this from inside their own Metal scope, which is narrower
/// than what this function admits on a Vulkan/DX12 adapter, so every lever
/// governs both but the scope does not — see [`conv_gemm`].
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
fn coop_f32_admitted(dev: &wgpu::Device, k: u32, n: u32, bf16_weight_in_b: bool) -> bool {
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
    let metal_aligned = k.is_multiple_of(8) && n.is_multiple_of(32);
    let portable_aligned = k.is_multiple_of(8) && n.is_multiple_of(8);
    let metal_coop = !disabled
        && has_coop
        && metal_aligned
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

/// Which dense-matmul compute paths are legal for one specific call.
///
/// `Bf16Packed` and `F16` are absent on purpose: neither is chosen by
/// `derive_matmul_compute` — `Bf16Packed` is installed later by the caller when
/// it sees a BF16 `Param` rhs, and no path dispatches `F16` today. A tuning
/// table must not be able to conjure either one here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatmulEligibility {
    pub coop16: bool,
    pub coop_f16_vk: bool,
    pub coop_f32: bool,
    pub f32: bool,
}

impl MatmulEligibility {
    /// Under `RLX_WGPU_MATMUL_F32_ONLY` the pin is absolute.
    fn only_f32() -> Self {
        Self {
            coop16: false,
            coop_f16_vk: false,
            coop_f32: false,
            f32: true,
        }
    }

    fn allows(&self, v: MatmulCompute) -> bool {
        match v {
            MatmulCompute::Coop16 => self.coop16,
            MatmulCompute::CoopF16Vk => self.coop_f16_vk,
            MatmulCompute::CoopF32 => self.coop_f32,
            MatmulCompute::F32 => self.f32,
            // Never selectable from the table — see the struct doc.
            MatmulCompute::F16 | MatmulCompute::Bf16Packed => false,
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
/// with no packed form to keep) and reads the same shared dispatch table — which
/// can move it back to `F32`, never onto a family that cannot serve the layout.
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
/// The Metal term is scope, not alignment. It is what keeps the portable 8×8
/// sibling, Vulkan's, out of these two classes: although these routes are
/// host-staged on a Vulkan/DX12 adapter (`wgpu_prefer_conv_host` folds in
/// `coop_discrete_backend`), `RLX_WGPU_SHARD_GPU` opts them back onto the device,
/// and the kernel `matmul_coop_f32_active_kernel` would then return there is the
/// portable one.
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
                && coop_f32_admitted(dev, k, n, false)
        }
        _ => false,
    };
    let eligible = MatmulEligibility {
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
#[allow(dead_code)]
pub(crate) fn matmul_compute_to_dispatch(
    v: MatmulCompute,
) -> rlx_gpu_dispatch::dispatch::WgpuMatmul {
    use rlx_gpu_dispatch::dispatch::WgpuMatmul as D;
    match v {
        MatmulCompute::F32 => D::F32,
        MatmulCompute::F16 => D::F16,
        MatmulCompute::Coop16 => D::Coop16,
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
/// Append a CastF32ToF16 pre-pass: mirrors `arena[off..off+len]` (f32) into
/// `arena_f16[off..off+len]` (f16) so coop matmul kernels can read operands
/// as f16. Used before CoopF16Vk when A/B are computed activations.
fn push_cast_f32_to_f16_step(
    device: &wgpu::Device,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    schedule: &mut Vec<Step>,
    uniforms: &mut Vec<wgpu::Buffer>,
    bind_groups: &mut Vec<wgpu::BindGroup>,
    mm_cast: &Option<&'static Kernel>,
    src_off: u32,
    len: u32,
) {
    let kernel = match mm_cast {
        Some(k) => *k,
        None => return, // device lacks SHADER_F16; fall through, dispatch will skip
    };
    let f16_buf = match &arena.f16_buffer {
        Some(b) => b,
        None => return,
    };
    let p = CastF32ToF16Params {
        src_off: src_off.saturating_sub((arena_base / 4) as u32),
        len,
        _p0: 0,
        _p1: 0,
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
    let (f16_base, f16_size) = f16_shadow_bind_range(arena_base, arena_size, f16_buf.size());
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
fn build_matmul_bind_group(
    device: &wgpu::Device,
    mm_k: &Kernel,
    _mm_w: &Kernel,
    mm_f16w: &Option<&'static Kernel>,
    mm_bf16w: &'static Kernel,
    mm_f16c: &Option<&'static Kernel>,
    mm_coop: &Option<&'static Kernel>,
    mm_coop_f32: &Option<&'static Kernel>,
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
            conv_im2col_scratch_bytes(&band, planned as usize, planned + aligned),
            aligned as usize,
            "a window with room for the panel reserves exactly it, 256-byte aligned"
        );
        assert_eq!(
            conv_im2col_scratch_bytes(&band, planned as usize, planned + col_bytes - 1),
            0,
            "a window one byte short of the panel drops it"
        );
        assert_eq!(
            conv_im2col_scratch_bytes(&band, planned as usize, u64::MAX),
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
        assert_eq!(conv_im2col_scratch_bytes(&huge, 0, u64::MAX), 0);
        // …and the tail the panel is reserved in is floored at 64 MiB, the base every
        // recorded `expected_col_delta` in the route harness is measured against.
        assert_eq!(compute_scratch_bytes(&band), 64 * 1024 * 1024);
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
            ),
            None,
            "1×1 taps are not the im2col route's shape"
        );
        assert_eq!(conv_im2col_scratch_bytes(&g, 1 << 20, u64::MAX), 0);
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
