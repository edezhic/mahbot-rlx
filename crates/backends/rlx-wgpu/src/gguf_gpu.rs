// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU GGUF dequant + wgpu matmul for `Op::DequantMatMul`.
//!
//! Preferred path: scratch-free windowed kernels for Q4_K / Q6_K / Q1_0
//! (decode `m=1` and prefill `m>1`). At `m == 1` Q4_K / Q6_K take the tiled
//! one-row arm (`dequant_gemv_tiled`: one workgroup per output column, its
//! threads splitting the contraction, X read four elements at a time); Q1_0
//! keeps the original one-thread-per-column arm on the scalar X view. For
//! `m > 1` Q4_K / Q6_K uses the row-grouped GEMM
//! (`encode_dequant_matmul_gguf_gemm_grouped`): one pass over each packed weight
//! tile per group of 8 rows, instead of the per-row GEMV's one full pass over
//! the weight per row. Older scratch+`matmul_bt` path still exists for schemes
//! without a fused kernel.
//!
//! The one-row arm is picked by *scheme*, not by row count, so it also serves
//! the per-row fallback the grouped GEMM declines (rows spread over shards, an
//! X window over the binding limit) — one row at a time, with each row's weight
//! reuse unchanged.
//!
//! **Host fallback:** [`crate::gguf_host`] when the scheme is unsupported on GPU
//! and no dequant scratch was reserved, and for a call whose X window the tiled
//! one-row arm cannot address ([`gemv_one_row_device_ok`]).
//!
//! **Grouped MoE:** [`run_dequant_grouped_matmul_gguf_gpu`] when scratch fits;
//! otherwise [`crate::gguf_host::run_dequant_grouped_matmul_gguf`].
//!
//! **Which route a call took:** [`route_counts`]. **What shape the one-row
//! route took:** [`gemv_one_row_shape`].
//!
//! **Limits:** arena byte offsets are u32; IQ-family branches mirror Metal/CUDA
//! but are not covered by dedicated WGPU parity tests yet.
//!
//! Full matrix: [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use std::sync::atomic::{AtomicU64, Ordering};

use rlx_ir::{Graph, Op};

use crate::buffer::Arena;
use crate::gguf_host::scheme_from_id;
use crate::kernels::{
    DequantGemmGgufParams, DequantGemmQ10Params, DequantGemvGgufParams, DequantGgufParams, Kernel,
    MatmulParams, dequant_gemm_q1_0_kernel, dequant_gemm_q4k_kernel, dequant_gemm_q6k_kernel,
    dequant_gemv_gguf_kernel, dequant_gemv_gguf_tiled_kernel, dequant_gguf_kernel,
    matmul_bt_kernel,
};

/// Schemes the fused decode GEMV ([`run_dequant_matmul_gguf_gemv`]) handles
/// on-GPU without f32 scratch — the union of its two one-row arms: the tiled
/// arm's K-quants ([`gemv_tiled_supports_scheme`]) and Q1_0, which keeps the
/// original shape. Q4_K (0) + Q6_K (2) cover Llama Q4_K_M GGUFs (q/k/o/gate/up
/// are Q4_K; v/down/embed are Q6_K). Q1_0 (24) is the prism-ml Bonsai-27B 1-bit
/// scheme — scratch-free here avoids re-materializing the whole 27B to f32 every
/// token (~200s/tok → fused) AND avoids the multi-GiB dequant scratch that blew
/// wgpu's storage-buffer binding limit.
pub fn gemv_supports_scheme(scheme_id: u32) -> bool {
    gemv_tiled_supports_scheme(scheme_id) || scheme_id == 24
}

/// The reshaped one-row arm of that GEMV (`dequant_gemv_tiled`): the K-quant
/// schemes whose 256-element blocks split into 32-element sub-blocks, which is
/// the element/byte mapping that arm is written against. The other arm is what
/// is left of [`gemv_supports_scheme`] once these are taken out — Q1_0, whose
/// 128-element blocks with 8-element sign bytes do not fit the tiled arm's
/// sub-blocks and so keeps one thread per output column. Each arm's shader
/// branches are tied to its routing by `tests/gguf_kernel_scheme_coverage.rs`.
pub fn gemv_tiled_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0 | 2)
}

/// Whether the one-row route can run this call on the device.
///
/// The tiled arm reads X as `vec4` (8 vector loads per lane instead of 32 scalar
/// ones), so the window's own offset and the row stride (`k` elements) must both
/// line up with a four-element group. True at the shapes these schemes are used
/// at — the arena aligns every tensor to 16 bytes and `k` is a multiple of 256 —
/// and false for a view the route then hands to the host rather than reading X
/// shifted. The grouped GEMM checks the same two things for its own `vec4` X
/// reads. Pinned device-free by `tests/gguf_kernel_scheme_coverage.rs`.
pub fn gemv_one_row_device_ok(scheme_id: u32, x_byte_off: usize, row_bytes: usize) -> bool {
    !gemv_tiled_supports_scheme(scheme_id)
        || (x_byte_off.is_multiple_of(16) && row_bytes.is_multiple_of(16))
}

/// Route a batched packed GGUF multiply can take.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DequantMatmulRoute {
    /// Row-grouped fused GEMM: each packed weight tile is read once per group
    /// of 8 rows (Q4_K / Q6_K, `m > 1`).
    Grouped,
    /// One fused GEMV dispatch per output row (decode `m == 1`, or a fallback).
    PerRow,
    /// Tiled Q1_0 GEMM (one-bit scheme, `m > 1`).
    Q1_0Gemm,
    /// Device dequant into f32 scratch, then `matmul_bt`.
    Scratch,
    /// The whole multiply on the host (CPU dequant).
    Host,
}

/// Per-route tallies, one field per route.
#[derive(Debug, Clone, Copy)]
pub struct DequantMatmulRoutes {
    pub grouped: u64,
    pub per_row: u64,
    pub q1_0_gemm: u64,
    pub scratch: u64,
    pub host: u64,
}

/// One counter per route. Named rather than an array indexed by the enum's
/// discriminant: `counter` below is an exhaustive `match`, so a new variant has to
/// name its counter — forgetting one is a compile error, not an index past the
/// last element.
static ROUTE_GROUPED: AtomicU64 = AtomicU64::new(0);
static ROUTE_PER_ROW: AtomicU64 = AtomicU64::new(0);
static ROUTE_Q1_0_GEMM: AtomicU64 = AtomicU64::new(0);
static ROUTE_SCRATCH: AtomicU64 = AtomicU64::new(0);
static ROUTE_HOST: AtomicU64 = AtomicU64::new(0);

impl DequantMatmulRoute {
    /// Every variant, for the union below and for [`reset_route_counts`] — the one
    /// list the compiler does not check (`counter`/`defers_host_read` are exhaustive).
    const ALL: [Self; 5] = [
        Self::Grouped,
        Self::PerRow,
        Self::Q1_0Gemm,
        Self::Scratch,
        Self::Host,
    ];

    fn counter(self) -> &'static AtomicU64 {
        match self {
            DequantMatmulRoute::Grouped => &ROUTE_GROUPED,
            DequantMatmulRoute::PerRow => &ROUTE_PER_ROW,
            DequantMatmulRoute::Q1_0Gemm => &ROUTE_Q1_0_GEMM,
            DequantMatmulRoute::Scratch => &ROUTE_SCRATCH,
            DequantMatmulRoute::Host => &ROUTE_HOST,
        }
    }

    /// Whether this route defers its operand read past its own step.
    fn defers_host_read(self) -> bool {
        match self {
            // Device-resident: the operand is read inside the route's own step.
            DequantMatmulRoute::Grouped
            | DequantMatmulRoute::PerRow
            | DequantMatmulRoute::Q1_0Gemm
            | DequantMatmulRoute::Scratch => false,
            // The `gguf_host` fallback of `Step::DequantMatmulGguf`, a
            // `step_runs_on_host` step: it drains, reads, computes and writes
            // back inside its own step. Nothing is pending at the next step.
            DequantMatmulRoute::Host => false,
        }
    }
}

/// Whether **any** packed multiply this backend can run reads an operand after its
/// own step, i.e. the hazard pinned by
/// [`rlx_compile::MemoryPlanOptions::dequant_host_fallback`].
///
/// Derived from the per-route classification over `DequantMatmulRoute::ALL`, so a
/// new route answers next to the route table. Packed steps outside that table are
/// covered by the `no_packed_multiply_step_is_deferred_to_the_host_tail` test in
/// `backend::step`; Metal needs the pin, since `flush_deferred_host` defers its
/// dequant flush.
pub fn dequant_route_defers_host_read() -> bool {
    DequantMatmulRoute::ALL.iter().any(|r| r.defers_host_read())
}

/// How many packed GGUF multiplies this process has routed each way.
///
/// The engine's dispatch report is claim-based: a backend that lists
/// `OpKind::DequantMatMul` in `SUPPORTED_OPS` is reported as native however the
/// op actually ran, including when it is dequantized on the host. This is the
/// wgpu-local answer — read it after a run (or reset it first to bracket a
/// single call).
pub fn route_counts() -> DequantMatmulRoutes {
    let n = |r: DequantMatmulRoute| r.counter().load(Ordering::Relaxed);
    DequantMatmulRoutes {
        grouped: n(DequantMatmulRoute::Grouped),
        per_row: n(DequantMatmulRoute::PerRow),
        q1_0_gemm: n(DequantMatmulRoute::Q1_0Gemm),
        scratch: n(DequantMatmulRoute::Scratch),
        host: n(DequantMatmulRoute::Host),
    }
}

/// Zero [`route_counts`] (tests bracketing a single call).
pub fn reset_route_counts() {
    for r in DequantMatmulRoute::ALL {
        r.counter().store(0, Ordering::Relaxed);
    }
}

pub(crate) fn note_route(route: DequantMatmulRoute) {
    route.counter().fetch_add(1, Ordering::Relaxed);
}

/// Threads one workgroup of a one-row packed multiply runs, per arm: the tiled
/// arm splits a column's contraction over 32 lanes (one lane per 32-element
/// sub-block), the legacy arm gives each of its 64 threads one output column.
///
/// The shader's declared `@workgroup_size` and the shape the route records must
/// both be these numbers: `tests/gguf_kernel_scheme_coverage.rs` pins the first
/// (as literals, the constants being private), `tests/gguf_matmul_route.rs` the
/// second.
const GEMV_TILED_THREADS: u32 = 32;
const GEMV_LEGACY_THREADS: u32 = 64;

/// Most workgroups one dispatch may have along a dimension — the WebGPU floor
/// (`max_compute_workgroups_per_dimension`), which every backend honours. Only
/// the tiled one-row arm can reach it at one workgroup per output column.
const GEMV_MAX_WORKGROUPS: usize = 65535;

/// The shape the one-row packed multiply actually issued, recorded per call.
///
/// The one-row route (`DequantMatmulRoute::PerRow`) only says a call took it;
/// this says what shape it took — output columns per workgroup, threads per
/// workgroup, workgroups in the dispatch — which together say how one row's work
/// was arranged. Always recorded, in-process, with no environment variable
/// behind it, so a change that quietly returns the model-width shapes (1024…6144
/// columns, one row) to one unit of work per output column reading the weight
/// byte-wise, or to a tile with discarded rows, is visible in
/// `tests/gguf_matmul_route.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvOneRowShape {
    /// Output columns one workgroup serves. `1` on the tiled arm; `64` while the
    /// arm is one thread per output column.
    pub cols_per_workgroup: u32,
    /// Threads one workgroup runs. On the tiled arm they split that column's
    /// contraction; on the legacy arm each thread serves its own column.
    pub threads_per_workgroup: u32,
    /// Workgroups in the recorded dispatch — the last chunk's, when the call was
    /// split.
    pub workgroups: u32,
}

static ONE_ROW_COLS_PER_WG: AtomicU64 = AtomicU64::new(0);
static ONE_ROW_THREADS: AtomicU64 = AtomicU64::new(0);
static ONE_ROW_WORKGROUPS: AtomicU64 = AtomicU64::new(0);

/// Shape of the last one-row packed multiply this process issued, `None` before
/// the first one. Read after a run (or [`reset_gemv_one_row_shape`] first to
/// bracket a single call).
pub fn gemv_one_row_shape() -> Option<GemvOneRowShape> {
    let threads = ONE_ROW_THREADS.load(Ordering::Relaxed);
    if threads == 0 {
        return None;
    }
    Some(GemvOneRowShape {
        cols_per_workgroup: ONE_ROW_COLS_PER_WG.load(Ordering::Relaxed) as u32,
        threads_per_workgroup: threads as u32,
        workgroups: ONE_ROW_WORKGROUPS.load(Ordering::Relaxed) as u32,
    })
}

/// Forget [`gemv_one_row_shape`] (tests bracketing a single call).
pub fn reset_gemv_one_row_shape() {
    ONE_ROW_COLS_PER_WG.store(0, Ordering::Relaxed);
    ONE_ROW_THREADS.store(0, Ordering::Relaxed);
    ONE_ROW_WORKGROUPS.store(0, Ordering::Relaxed);
}

/// Record what one one-row dispatch was shaped like. `cols_per_workgroup` is how
/// many output columns a workgroup covers, so on the tiled arm
/// `workgroups * cols_per_workgroup` is exactly the output width dispatched — a
/// tile with rows nothing reads would make it larger. (The legacy arm's last
/// workgroup rounds up to 64 columns, so there the product can exceed the width
/// by under 64; that is rounding, not a discarded tile.)
fn note_gemv_one_row_shape(cols_per_workgroup: u32, threads_per_workgroup: u32, workgroups: u32) {
    ONE_ROW_COLS_PER_WG.store(cols_per_workgroup as u64, Ordering::Relaxed);
    ONE_ROW_THREADS.store(threads_per_workgroup as u64, Ordering::Relaxed);
    ONE_ROW_WORKGROUPS.store(workgroups as u64, Ordering::Relaxed);
}

/// Scheme ids the WGSL `dequant_gguf` kernel actually has a branch for.
///
/// The kernel is an `if (scheme_id == N) { … return; }` chain with **no
/// default arm**: an id that reaches it without a matching branch writes
/// nothing, so the weights read as zeros and the model degrades silently
/// instead of failing. Any scheme outside this set is therefore routed to
/// the host dequant path ([`crate::gguf_host`]) — slower, but correct.
///
/// Every registered scheme is branched today, so this currently never
/// diverts — it is the seam that keeps the *next* unbranched scheme correct.
/// Kept in sync with the shader by
/// `tests/gguf_kernel_scheme_coverage.rs`, which parses the branches out of
/// `DEQUANT_GGUF_WGSL` and requires the two to agree in both directions — so
/// adding a scheme id without a branch (or a branch without listing it here)
/// fails the tests rather than shipping a silent mis-decode.
pub fn kernel_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0..=29)
}

/// Max f32 scratch for dequantized weights `[n, k]` across all GGUF ops.
pub fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.is_gguf()
        {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            // Q4_K/Q6_K/Q1_0 GGUF matmuls run scratch-free: Q4_K/Q6_K through
            // the windowed kernels in this module (one GEMV per row at m=1,
            // row-grouped GEMM above that), Q1_0 through its tiled GEMM.
            // Skipping them here keeps the arena from reserving the (multi-GiB)
            // dequant scratch — the LM-head slab (~0.6 GiB for a 0.6B model) is
            // what pushed the arena over wgpu's 2 GiB storage-buffer binding
            // limit and made the prefill dequant bind fail (Validation Error:
            // binding range > limit).
            if gemv_supports_scheme(crate::gguf_host::gguf_scheme_id(*scheme)) {
                continue;
            }
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { scheme: _ } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
        }
    }
    max
}

fn slab_bytes_for(scheme: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> usize {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    (k * n) / block_elems * block_bytes
}

fn launch_dequant_gguf(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w_byte_off: usize,
    scratch_byte_off: usize,
    scheme_id: u32,
    num_blocks: usize,
) {
    let dk = dequant_gguf_kernel(device);
    let lut = crate::iq_grid::wgpu_iq_grid_buffer(device, queue);
    let p = DequantGgufParams {
        w_byte_off: w_byte_off as u32,
        dst_f32_off: (scratch_byte_off / 4) as u32,
        scheme_id,
        num_blocks: num_blocks as u32,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant_gguf uniform"),
        size: std::mem::size_of::<DequantGgufParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = bind_dequant_gguf(device, dk, &arena.buffer, &u, &lut);

    let block = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(block as usize) as u32;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gguf"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu dequant_gguf pass"),
            ..Default::default()
        });
        pass.set_pipeline(&dk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

fn bind_dequant_gguf(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &wgpu::Buffer,
    uniform: &wgpu::Buffer,
    lut: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu dequant_gguf bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lut.as_entire_binding(),
            },
        ],
    })
}

fn dispatch_matmul_bt(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    arena: &wgpu::Buffer,
    m: u32,
    k: u32,
    n: u32,
    x_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) {
    let mm = matmul_bt_kernel(device);
    let p = MatmulParams {
        m,
        k,
        n,
        a_off: x_off_f32,
        b_off: w_off_f32,
        c_off: out_off_f32,
        batch: 1,
        a_batch_stride: m * k,
        b_batch_stride: 0,
        c_batch_stride: m * n,
        has_bias: 0,
        bias_off: 0,
        act_id: 0xFFFF,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt uniform"),
        size: std::mem::size_of::<MatmulParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt bg"),
        layout: &mm.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gguf matmul_bt pass"),
            ..Default::default()
        });
        pass.set_pipeline(&mm.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(32), m.div_ceil(32), 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via matmul_bt.
pub fn run_dequant_matmul_gguf_gpu(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    // Windowed GEMV covers the sharded / split-weight case; this scratch+matmul_bt
    // path still assumes one contiguous arena bind.
    if arena.is_sharded() || crate::buffer::is_weight_off(w_byte_off) {
        note_route(DequantMatmulRoute::Host);
        crate::gguf_host::run_dequant_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }
    // Q4_K/Q6_K/Q1_0: scratch-free windowed kernels (batched rows → one submit),
    // row-grouped for m>1 Q4_K/Q6_K.
    if gemv_supports_scheme(scheme_id) {
        run_dequant_matmul_gguf_gemv_rows(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }

    // No WGSL branch for this scheme → host dequant. Without this the kernel
    // would run, match nothing, and leave the scratch untouched (= zeros).
    if !kernel_supports_scheme(scheme_id) {
        note_route(DequantMatmulRoute::Host);
        crate::gguf_host::run_dequant_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }

    note_route(DequantMatmulRoute::Scratch);
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let num_blocks = (k * n) / block_elems.max(1);

    launch_dequant_gguf(
        arena,
        device,
        queue,
        w_byte_off,
        scratch_byte_off,
        scheme_id,
        num_blocks,
    );

    dispatch_matmul_bt(
        device,
        queue,
        &arena.buffer,
        m as u32,
        k as u32,
        n as u32,
        (x_byte_off / 4) as u32,
        (scratch_byte_off / 4) as u32,
        (out_byte_off / 4) as u32,
    );
}

const STORAGE_ALIGN: u64 = 256;

/// Reused GEMM/GEMV output slabs — creating a fresh buffer per matmul was a
/// large fraction of Bonsai-27B prefill time (hundreds of allocs/frame).
/// Safe across dispatches in one encoder because each dispatch is followed by
/// a `copy_buffer_to_buffer` into the arena before the next write.
/// Pooled output buffer + its capacity. Named (vs a tuple) so it can assert
/// `Send` on the browser WebGPU backend where `wgpu::Buffer` is `!Send`; wasm is
/// single-threaded, so the process-global pool is never sent cross-thread.
struct OutPool(wgpu::Buffer, u64);
#[cfg(target_arch = "wasm32")]
unsafe impl Send for OutPool {}

/// Bytes held by the pooled output slab above — the one device buffer the
/// windowed GGUF kernels use that the arena plan does not know about, so a
/// memory report counts it rather than understating real usage.
static OUT_POOL_BYTES: AtomicU64 = AtomicU64::new(0);

/// Device bytes resident in the pooled GEMM/GEMV output slab (0 until the first
/// burst of bundled multiplies). Grows to the largest slab requested; never
/// released, like the rest of the backend's process-global scratch.
///
/// This is the accurate surface for that figure: the slab is allocated lazily by
/// the first batched packed multiply, so the compile-time
/// `RLX_WGPU_ARENA_REPORT` line has nothing to print for it.
pub fn pooled_out_buf_bytes() -> u64 {
    OUT_POOL_BYTES.load(Ordering::Relaxed)
}

fn with_pooled_out_buf<R>(
    device: &wgpu::Device,
    need_bytes: u64,
    f: impl FnOnce(&wgpu::Buffer) -> R,
) -> R {
    use std::sync::Mutex;
    static POOL: Mutex<Option<OutPool>> = Mutex::new(None);
    let need = need_bytes.max(16).div_ceil(16) * 16;
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let recreate = match slot.as_ref() {
        Some(OutPool(_, cap)) => *cap < need,
        None => true,
    };
    if recreate {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu dequant_gemm/gemv out pool"),
            size: need,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        *slot = Some(OutPool(buf, need));
        OUT_POOL_BYTES.store(need, Ordering::Relaxed);
    }
    let OutPool(buf, _) = slot.as_ref().expect("out pool");
    f(buf)
}

/// Encode fused GGUF DequantMatMul into `enc` (no `queue.submit`), leaving the
/// submission to the caller. Scheme and shape pick the route: the Q1_0 and
/// row-grouped K-quant GEMMs first, then the per-row GEMV loop. Uniforms are
/// unique per dispatch (`queue.write_buffer` is not ordered inside an encoder
/// the way copies are).
///
/// The per-row loop's tiled arm reads X as `vec4`, so X's window must be
/// four-element addressable. [`run_dequant_matmul_gguf_gemv_rows`] — the route
/// the executable takes — checks that with [`gemv_one_row_device_ok`] before
/// calling here; a direct caller has to make sure of it itself.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_matmul_gguf_into(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    if m == 0 {
        return;
    }
    if scheme_id == 24
        && m > 1
        && !rlx_ir::env::flag("RLX_WGPU_Q1_0_GEMM_DISABLE")
        && encode_dequant_matmul_gguf_gemm_q1_0(
            arena,
            device,
            queue,
            enc,
            m,
            k,
            n,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        )
    {
        note_route(DequantMatmulRoute::Q1_0Gemm);
        return;
    }
    // Q4_K / Q6_K batched: one row-grouped GEMM per weight tile, instead of the
    // per-row loop below re-streaming the whole packed weight per row. Which
    // schemes it has an entry point for is decided inside, before any size math.
    // Falling back is driven by what can be *encoded* (X window), never by the
    // weight window merely exceeding the binding limit — that case is N-chunked
    // inside.
    if m > 1
        && encode_dequant_matmul_gguf_gemm_grouped(
            arena,
            device,
            queue,
            enc,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        )
    {
        note_route(DequantMatmulRoute::Grouped);
        return;
    }
    for row in 0..m {
        encode_dequant_matmul_gguf_gemv_one(
            arena,
            device,
            queue,
            enc,
            k,
            n,
            scheme_id,
            x_byte_off + row * k * 4,
            w_byte_off,
            out_byte_off + row * n * 4,
        );
    }
    note_route(DequantMatmulRoute::PerRow);
}

/// Fused decode/prefill path: `Y[m,n] = X[m,k] @ W^T` with packed GGUF `W`.
///
/// * Q4_K / Q6_K + `m > 1` → row-grouped tiled GEMM — one pass over each packed
///   weight tile per group of 8 rows (unless X spans shards).
/// * Q1_0 + `m > 1` → tiled GEMM (Metal `q1_0_mm_f32` parity) — one dispatch,
///   unless `RLX_WGPU_Q1_0_GEMM_DISABLE=1` or X spans shards.
/// * Otherwise → per-row GEMV encoded into one command buffer (one submit).
///
/// An X window the tiled arm's `vec4` read cannot address sends the whole call
/// to [`crate::gguf_host`] — this is the route that reaches that arm, so the
/// check belongs here rather than at any one of its callers.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_gguf_gemv_rows(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    if !gemv_one_row_device_ok(scheme_id, x_byte_off, k * 4) {
        note_route(DequantMatmulRoute::Host);
        crate::gguf_host::run_dequant_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gguf rows"),
    });
    encode_dequant_matmul_gguf_into(
        arena,
        device,
        queue,
        &mut enc,
        m,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
    );
    queue.submit(std::iter::once(enc.finish()));
}

/// Tiled Q1_0 GEMM into `enc`. Returns `false` when X/W cannot fit a shard window.
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_gguf_gemm_q1_0(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    m: usize,
    k: usize,
    n: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) -> bool {
    debug_assert!(k.is_multiple_of(128), "Q1_0 GEMM requires k % 128 == 0");
    let block_elems = 128usize;
    let block_bytes = 18usize;
    let w_total_bytes = (k * n) / block_elems * block_bytes;

    let x0 = x_byte_off as u64;
    let x_bytes = (m * k * 4) as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_need = x0 + x_bytes - x_base;
    let x_size = (x_need.div_ceil(16) * 16).min(x_buf.size().saturating_sub(x_local));
    if x_size < x_need {
        return false; // X spans shards — GEMV-per-row is safe
    }

    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w_buf_size = w_buf.size();
    let w0 = w_raw as u64;
    let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let w_need = w0 + w_total_bytes as u64 - w_base;
    let w_size = (w_need.div_ceil(16) * 16).min(w_buf_size.saturating_sub(w_base));
    if w_size < w_need {
        return false;
    }

    let max_bind = device.limits().max_storage_buffer_binding_size;
    if x_size > max_bind || w_size > max_bind {
        return false;
    }

    let out_elems = m * n;
    let out_bytes = (out_elems * 4) as u64;
    with_pooled_out_buf(device, out_bytes, |out_buf| {
        let p = DequantGemmQ10Params {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            x_f32_off: ((x0 - x_base) / 4) as u32,
            w_byte_off: (w0 - w_base) as u32,
            out_f32_off: 0,
            _p0: 0,
            _p1: 0,
        };
        let dk = dequant_gemm_q1_0_kernel(device);
        let u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu dequant_gemm_q1_0 uni"),
            size: std::mem::size_of::<DequantGemmQ10Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu dequant_gemm_q1_0 bg"),
            layout: &dk.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: x_buf,
                        offset: x_local,
                        size: wgpu::BufferSize::new(x_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: w_buf,
                        offset: w_base,
                        size: wgpu::BufferSize::new(w_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rlx-wgpu dequant_gemm_q1_0 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&dk.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            const TM: u32 = 8;
            pass.dispatch_workgroups((n as u32).div_ceil(64), (m as u32).div_ceil(TM), 1);
        }

        let (dst0, dst0_off) = arena.resolve_act(out_byte_off);
        let row_bytes = (n * 4) as u64;
        let contiguous = (dst0_off as u64) + out_bytes <= dst0.size();
        if contiguous {
            enc.copy_buffer_to_buffer(out_buf, 0, dst0, dst0_off as u64, out_bytes);
        } else {
            for r in 0..m {
                let (dst, dst_off) = arena.resolve_act(out_byte_off + r * n * 4);
                enc.copy_buffer_to_buffer(
                    out_buf,
                    (r as u64) * row_bytes,
                    dst,
                    dst_off as u64,
                    row_bytes,
                );
            }
        }
    });
    true
}

/// Binding-size budget for the two windowed paths that chunk around it — the
/// per-row GEMV encoder and the multi-row GEMM: the device's storage-binding
/// limit, lowered to `RLX_WGPU_MAX_BIND_MB` when set — which artificially caps
/// it to exercise the N-chunked window paths on GPUs whose real limit is large
/// (validation/testing). Only a lowering: a value above the device limit must
/// not produce a binding the device rejects outright. The Q1_0 tiled GEMM reads
/// the device limit directly and ignores this cap.
fn gguf_max_bind(device: &wgpu::Device) -> u64 {
    let device_max = device.limits().max_storage_buffer_binding_size;
    rlx_ir::env::var("RLX_WGPU_MAX_BIND_MB")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|mb| (mb * 1024 * 1024).min(device_max))
        .unwrap_or(device_max)
}

/// Packed-weight bytes that fit under `max_bind`: leaves room for the window's
/// align-down base slack (< [`STORAGE_ALIGN`]) and its 16-byte round-up. Both
/// the row-grouped GEMM's column chunking and the per-row GEMV's row chunking
/// are sized from this, so they cannot drift apart.
fn gguf_window_budget(max_bind: u64) -> u64 {
    max_bind.saturating_sub(STORAGE_ALIGN + 16)
}

/// Largest output slab the grouped path will ask the pool for.
///
/// The slab exists only to make one dispatch's output bindable, so the binding
/// limit already bounds it — but the pool is process-global and never released,
/// and `m * ncols * 4` at a few thousand rows against a 200k-wide head is
/// gigabytes of scratch the per-row path it replaces never held (that path writes
/// row by row straight into the arena). So cap the slab and chunk columns
/// further: a larger `m` then costs more chunks, not more memory. 256 MiB keeps
/// the widest served shape (276 rows x 202048 columns = 213 MiB) in one chunk.
const OUT_SLAB_CAP: u64 = 256 << 20;

/// Fused multi-row GGUF K-quant GEMM for Q4_K / Q6_K into `enc` (no submit).
///
/// One pass over each packed weight tile per group of `TM = 8` rows, instead of
/// the per-row GEMV's one full pass over the whole weight per row. `n` is split
/// into column chunks that fit `max_storage_buffer_binding_size` (and the
/// weight buffer's shard), so a weight parked in [`Arena::weight_buffer`] — or
/// one whose window is larger than the binding limit — still takes this path.
///
/// The output slab is `[m, ncols]` bound whole, so `ncols` is bounded by the
/// binding limit *and* by [`OUT_SLAB_CAP`] — what keeps the wide-`n`,
/// `m`-in-the-hundreds LM-head shape bindable and its scratch bounded. The slab
/// comes from the same pooled buffer the Q1_0 GEMM uses
/// ([`with_pooled_out_buf`]), so it is allocated once for the widest shape seen
/// rather than per call, and it is reported by
/// [`pooled_out_buf_bytes`].
///
/// Returns `false` when the tiled kernel cannot be encoded for this call: the
/// scheme has no entry point, X spans shards, its window exceeds the binding
/// limit, or `k` is not a whole number of blocks — all decided before anything is
/// encoded, including the per-chunk window fit (only the first chunk can decline,
/// every later one reads less and ends inside the weight region). The caller then
/// uses the per-row GEMV, which rewrites the same `m*n*4` output bytes row by row,
/// so the fallback is correct.
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_gguf_gemm_grouped(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) -> bool {
    debug_assert!(
        m > 1,
        "m == 1 goes through encode_dequant_matmul_gguf_gemv_one"
    );
    // The schemes this kernel set covers (Q4_K 0, Q6_K 2), decided before any
    // size math — so a third K-quant is one arm here, and no other scheme reaches
    // the block-size check below. The `_` arm declines the call rather than waving
    // the scheme through: an id that fell back to the Q6_K entry point would decode
    // wrong values, not fail.
    let dk = match scheme_id {
        0 => dequant_gemm_q4k_kernel(device),
        2 => dequant_gemm_q6k_kernel(device),
        _ => return false,
    };
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize; // 256
    let block_bytes = scheme.gguf_block_bytes() as usize; // 144 (Q4_K) | 210 (Q6_K)
    // The entries above hardcode this geometry; if a scheme's block layout ever
    // stops matching it, fall back rather than stride the kernel wrongly.
    if block_elems != 256 || !(block_bytes == 144 || block_bytes == 210) {
        return false;
    }
    if !k.is_multiple_of(block_elems) || n == 0 {
        return false;
    }
    let row_bytes = (k / block_elems) * block_bytes; // packed weight bytes per column

    // X: all m rows are read through one window, so they must share a shard.
    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_need = x0 + (m * k * 4) as u64 - x_base;
    let x_size = (x_need.div_ceil(16) * 16).min(x_buf.size().saturating_sub(x_local));
    if x_size < x_need {
        return false;
    }

    let max_bind = gguf_max_bind(device);
    if x_size > max_bind {
        return false;
    }
    // The kernel reads X as `vec4`: with `x_base` = `x0` rounded down to
    // [`STORAGE_ALIGN`] (256, a multiple of 16), `x0 - x_base` is a multiple of
    // 16 exactly when `x0` is. That is this encoder's precondition — enforced on
    // the route by [`gemv_one_row_device_ok`], and documented on the public entry
    // [`encode_dequant_matmul_gguf_into`] for a direct caller.
    if !(x0 - x_base).is_multiple_of(16) {
        return false;
    }
    let x_vec4_off = ((x0 - x_base) / 16) as u32;

    // Columns per chunk, from both budgets: the bound weight window and the
    // whole-bound output slab (m * ncols * 4). Bounding the slab rather than
    // only the weight is what keeps a wide-n, large-m head inside the limit.
    if (m as u64) * 4 > max_bind {
        return false; // no column chunk's slab could be bound
    }
    let by_bind = (gguf_window_budget(max_bind) / row_bytes.max(1) as u64).max(1);
    let by_out = (OUT_SLAB_CAP.min(max_bind).saturating_sub(16) / (m as u64 * 4)).max(1);
    let n_chunk_max = (n as u64).min(by_bind).min(by_out).max(1) as usize;

    // Slab bytes for one chunk: `m * ncols * 4` out of the pooled output buffer,
    // which is process-global and only grows. It is scratch the arena plan does
    // not account for (`dequant_gguf_scratch_bytes` reports 0 for these schemes);
    // `pooled_out_buf_bytes` is the accurate report of it.
    let out_slab = (m * n_chunk_max * 4) as u64;

    let mut ok = true;
    with_pooled_out_buf(device, out_slab, |out_buf| {
        let mut n0 = 0usize;
        while n0 < n {
            let (w_buf, w_raw) = arena.resolve_w(w_byte_off + n0 * row_bytes);
            let w_buf_size = w_buf.size();
            let w0 = w_raw as u64;
            let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
            let n_chunk = (n - n0).min(n_chunk_max).max(1);

            // The window the chunk reads — every packed weight byte it touches.
            // That one quantity is compared against both limits, rather than the
            // 4-byte-rounded binding size below: both limits are multiples of 4
            // (the binding limit is; so is the weight buffer's end, whose size is
            // rounded up, and so is every packed weight's own byte count, since
            // `write_buffer` rejects an unaligned copy), so the two comparisons
            // agree — and a row size that is not a multiple of 4 (Q6_K rows are
            // 210·B bytes) can never make a window that fits look like one that
            // does not. `by_bind` sizes `n_chunk` inside the same budget, so only
            // a first chunk whose row is wider than the whole budget — which no
            // chunking here or in the per-row GEMV can fix — declines here, before
            // anything is encoded; every later chunk reads less than the first and
            // ends inside the weight region, so none of them can bail. (A parked
            // weight never spans buffers: `from_plan_split` spills any single
            // weight that does not fit one weight buffer into the act arena, so
            // `resolve_w` yields the whole weight region.)
            let need = (w0 - w_base) + (n_chunk * row_bytes) as u64;
            if need > max_bind.min(w_buf_size.saturating_sub(w_base)) {
                ok = false;
                return;
            }
            let w_size = need.div_ceil(4) * 4;

            let p = DequantGemmGgufParams {
                m: m as u32,
                k: k as u32,
                ncols: n_chunk as u32,
                x_vec4_off,
                w_byte_off: (w0 - w_base) as u32,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            };
            let u = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu dequant_gemm_gguf uniform"),
                size: std::mem::size_of::<DequantGemmGgufParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu dequant_gemm_gguf bg"),
                layout: &dk.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: x_buf,
                            offset: x_local,
                            size: wgpu::BufferSize::new(x_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: w_buf,
                            offset: w_base,
                            size: wgpu::BufferSize::new(w_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        // Only this chunk's slab, not the whole pooled buffer:
                        // the pool is process-global and may be larger than the
                        // binding limit of the device in play.
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: out_buf,
                            offset: 0,
                            size: wgpu::BufferSize::new(out_slab),
                        }),
                    },
                ],
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rlx-wgpu dequant_gemm_gguf pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n_chunk as u32).div_ceil(64), (m as u32).div_ceil(8), 1);
            }

            // Slab rows are [m, n_chunk]; this chunk owns arena columns
            // [n0, n0 + n_chunk).
            let row_out = (n_chunk * 4) as u64;
            let (dst0, dst0_off) = arena.resolve_act(out_byte_off);
            if n_chunk == n && dst0_off as u64 + (m * n * 4) as u64 <= dst0.size() {
                enc.copy_buffer_to_buffer(out_buf, 0, dst0, dst0_off as u64, (m * n * 4) as u64);
            } else {
                for r in 0..m {
                    let (dst, dst_off) = arena.resolve_act(out_byte_off + r * n * 4 + n0 * 4);
                    enc.copy_buffer_to_buffer(
                        out_buf,
                        (r * n_chunk * 4) as u64,
                        dst,
                        dst_off as u64,
                        row_out,
                    );
                }
            }
            n0 += n_chunk;
        }
    });
    ok
}

/// Fused decode GEMV: `y[1,n] = x[1,k] @ W^T` with `W` GGUF-packed `[n,k]`,
/// dequantizing each weight block on the fly (no f32 scratch).
///
/// `x` and `weight` are bound as separate **read-only windows** of the arena
/// (each < 4 GiB; the whole-arena `as_entire_binding` overruns the binding
/// limit on multi-GiB models). `y` goes to a small **separate** output buffer —
/// the arena cannot also be bound read-write in the same dispatch (wgpu treats
/// STORAGE_READ_WRITE as exclusive) — which is then copied back into the arena.
///
/// Caller guarantees [`gemv_supports_scheme`]. This is the `m == 1` form of
/// [`run_dequant_matmul_gguf_gemv_rows`] — the entry the executable calls — and
/// delegates to it rather than encoding the row itself, so the route's X-window
/// guard covers both.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_gguf_gemv(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    run_dequant_matmul_gguf_gemv_rows(
        arena,
        device,
        queue,
        1,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
    );
}

fn encode_dequant_matmul_gguf_gemv_one(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let w_total_bytes = (k * n) / block_elems.max(1) * block_bytes;

    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_size = ((x0 + (k * 4) as u64 - x_base).div_ceil(16) * 16)
        .min(x_buf.size().saturating_sub(x_local));

    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w_buf_size = w_buf.size();
    let w0_all = w_raw as u64;

    // Packed-weight bytes per output row. Output rows are independent (each is a
    // separate dot product over k), so when the full weight window would exceed
    // `max_storage_buffer_binding_size` (128 MiB minimum on wgpu — hit by llvmpipe /
    // min-spec adapters, or large models on any GPU) we split the GEMV along N and
    // bind only each chunk's rows. `w_total_bytes / n` is the exact per-row stride
    // (matches the shader's `j * nbk * blk` indexing).
    let row_bytes = (w_total_bytes / n.max(1)) as u64;

    let max_bind = gguf_max_bind(device);
    assert!(
        x_size <= max_bind,
        "rlx-wgpu gguf gemv: x window {x_size} > max_bind {max_bind}"
    );
    // Two arms, one route: the tiled arm gives each output column a whole
    // workgroup (`cols_per_workgroup == 1`); the original gives one thread per
    // column, 64 columns to a workgroup. Which one ran is recorded — see
    // [`gemv_one_row_shape`].
    let tiled = gemv_tiled_supports_scheme(scheme_id);
    // Rows per chunk from the window budget; at least one, so a single row
    // wider than the whole budget still makes progress. The tiled arm also
    // spends one workgroup per output column, and a dispatch dimension is capped
    // (65535 on every backend, WebGPU's guaranteed floor), so an output wider
    // than that is split too — the same arm, one more chunk.
    let budget = gguf_window_budget(max_bind).max(row_bytes);
    let mut rows_per_chunk = ((budget / row_bytes.max(1)) as usize).max(1);
    if tiled {
        rows_per_chunk = rows_per_chunk.min(GEMV_MAX_WORKGROUPS);
    }

    let out_bytes = ((n * 4).max(4) as u64).div_ceil(16) * 16;
    with_pooled_out_buf(device, out_bytes, |out_buf| {
        let dk = if tiled {
            dequant_gemv_gguf_tiled_kernel(device)
        } else {
            dequant_gemv_gguf_kernel(device)
        };
        let mut n0 = 0usize;
        while n0 < n {
            let n_chunk = (n - n0).min(rows_per_chunk);
            let w0 = w0_all + n0 as u64 * row_bytes;
            let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
            let chunk_bytes = n_chunk as u64 * row_bytes;
            let w_size = ((w0 + chunk_bytes - w_base).div_ceil(16) * 16).min(w_buf_size - w_base);
            debug_assert!(
                w_size <= max_bind,
                "rlx-wgpu gguf gemv chunk window {w_size} > max_bind {max_bind}"
            );

            let p = DequantGemvGgufParams {
                k: k as u32,
                n: n_chunk as u32,
                scheme_id,
                x_f32_off: ((x0 - x_base) / 4) as u32,
                w_byte_off: (w0 - w_base) as u32,
                out_f32_off: n0 as u32,
                _p0: 0,
                _p1: 0,
            };

            let u = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu dequant_gemv_gguf uniform"),
                size: std::mem::size_of::<DequantGemvGgufParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu dequant_gemv_gguf bg"),
                layout: &dk.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: x_buf,
                            offset: x_local,
                            size: wgpu::BufferSize::new(x_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: w_buf,
                            offset: w_base,
                            size: wgpu::BufferSize::new(w_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                    // The same X window as binding 0, read as `vec4` by the tiled
                    // arm — see that shader's binding declaration.
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: x_buf,
                            offset: x_local,
                            size: wgpu::BufferSize::new(x_size),
                        }),
                    },
                ],
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rlx-wgpu dequant_gemv_gguf pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                if tiled {
                    // One workgroup per output column of the chunk.
                    note_gemv_one_row_shape(1, GEMV_TILED_THREADS, n_chunk as u32);
                    pass.dispatch_workgroups(n_chunk as u32, 1, 1);
                } else {
                    // One thread per output column, 64 columns per workgroup.
                    let workgroups = (n_chunk as u32).div_ceil(GEMV_LEGACY_THREADS);
                    note_gemv_one_row_shape(GEMV_LEGACY_THREADS, GEMV_LEGACY_THREADS, workgroups);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            n0 += n_chunk;
        }
        let (dst, dst_off) = arena.resolve_act(out_byte_off);
        enc.copy_buffer_to_buffer(out_buf, 0, dst, dst_off as u64, (n * 4) as u64);
    });
}

/// GPU dequant + grouped matmul for MoE packed expert stacks.
///
/// Scratch layout at `scratch_byte_off` (f32 bytes):
///   `[0 .. k*n)`: dequantized expert slab
///   `[k*n .. k*n+m*k)`: sorted token inputs
///   `[k*n+m*k .. k*n+m*k+m*n)`: sorted outputs before unpermute
pub fn run_dequant_grouped_matmul_gguf_gpu(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    idx_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    if arena.is_sharded() {
        crate::gguf_host::run_dequant_grouped_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            num_experts,
            scheme_id,
            x_byte_off,
            w_byte_off,
            idx_byte_off,
            out_byte_off,
        );
        return;
    }
    let scheme = scheme_from_id(scheme_id);
    let slab_bytes = slab_bytes_for(scheme, k, n);
    let num_blocks = (k * n) / scheme.gguf_block_size() as usize;

    let x_bytes = arena.read_bytes_range(device, queue, x_byte_off, m * k * 4);
    let x_host: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let idx_bytes = arena.read_bytes_range(device, queue, idx_byte_off, m * 4);
    let idx_host: Vec<f32> = idx_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let (packed_in, original_pos, offsets) =
        rlx_cpu::gguf_matmul::grouped_moe_sort_plan(&x_host, &idx_host, m, k, num_experts);

    let dequant_off = scratch_byte_off;
    let pack_in_off = scratch_byte_off + k * n * 4;
    let pack_out_off = scratch_byte_off + (k * n + m * k) * 4;

    let pack_in_bytes: Vec<u8> = packed_in.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, pack_in_off, &pack_in_bytes);

    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let w_off = w_byte_off + e * slab_bytes;
        launch_dequant_gguf(
            arena,
            device,
            queue,
            w_off,
            dequant_off,
            scheme_id,
            num_blocks,
        );
        let in_start = offsets[e];
        dispatch_matmul_bt(
            device,
            queue,
            &arena.buffer,
            count as u32,
            k as u32,
            n as u32,
            (pack_in_off / 4 + in_start * k) as u32,
            (dequant_off / 4) as u32,
            (pack_out_off / 4 + in_start * n) as u32,
        );
    }

    let pack_out_bytes = arena.read_bytes_range(device, queue, pack_out_off, m * n * 4);
    let packed_out: Vec<f32> = pack_out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
        &packed_out,
        &original_pos,
        &mut out_host,
        m,
        n,
    );

    let out_bytes: Vec<u8> = out_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, out_byte_off, &out_bytes);
}
