// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU GGUF dequant + wgpu matmul for `Op::DequantMatMul`.
//!
//! Preferred path: scratch-free windowed kernels for Q4_K / Q6_K / Q1_0
//! (decode `m=1` and prefill `m>1`) and for Q8_0. At `m == 1`
//! Q4_K / Q6_K / Q8_0 take the tiled one-row arm (`dequant_gemv_tiled`: one
//! workgroup per `GEMV_TILED_COLS_PER_WG` output columns, its threads splitting
//! each column's contraction over 32-element units of the packed weight, X read
//! four elements at a time and shared across those columns; for Q8_0 that unit
//! is its own whole 32-element block); Q1_0 keeps the original one-thread-per-column
//! arm on the scalar X view. For `m > 1` Q4_K / Q6_K / Q8_0 use the row-grouped GEMM
//! (`encode_dequant_matmul_gguf_gemm_grouped`): one pass over each packed
//! weight tile per group of 8 rows, instead of the per-row GEMV's one full pass
//! over the weight per row. Its `k` gate is the scheme's own packing block
//! ([`grouped_k_ok`]): the K-quants' block is the staged tile itself, while Q8_0's
//! 32-element one lets a row end inside a tile — the `dequant_gemm_q8_0` entry
//! point's staged-tile loop stops at the last whole block of such a row, so a
//! contraction of several tiles plus a block, or a sub-tile one, takes this route
//! rather than the slab. On a Q4_K chunk whose base dispatch would starve the
//! GPU, that entry point covers half the columns per workgroup with two threads to
//! each — the same quant bytes over up to twice the warps, every block's header
//! read twice (`GROUPED_SPLIT_MAX_BASE_WORKGROUPS` carries the rule and its
//! worth).
//! Older scratch+`matmul_bt` path still exists for schemes without a fused
//! kernel, and for every shape those kernels refuse.
//!
//! The per-row fallback is picked by *scheme*, not by row count, so it also
//! serves the rows the grouped GEMM declines (rows spread over shards, an
//! X window over the binding limit) — one row at a time, with each row's weight
//! reuse unchanged. The per-row fallback carries the fused entry point's own
//! schemes alone, so a multi-row Q8_0 call the grouped GEMM declines falls to the
//! scratch slab below, like the single rows its own arm's gates keep out.
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

use crate::backend::PassMarks;
use crate::buffer::Arena;
use crate::gguf_host::scheme_from_id;
use crate::kernels::{
    DequantGemmGgufParams, DequantGemmQ10Params, DequantGemvGgufParams, DequantGgufParams, Kernel,
    MatmulParams, dequant_gemm_q1_0_kernel, dequant_gemm_q4k_kernel, dequant_gemm_q6k_kernel,
    dequant_gemm_q8_0_kernel, dequant_gemv_gguf_kernel, dequant_gemv_gguf_tiled_kernel,
    dequant_gguf_kernel, matmul_bt_kernel,
};

/// Schemes the fused packed-multiply entry point
/// ([`run_dequant_matmul_gguf_gemv_rows`]) runs at **every** row count: the
/// K-quants, whose multi-row arm is the row-grouped GEMM, and Q1_0, which keeps
/// the original one-thread-per-column shape at every row. Q4_K (0) + Q6_K (2)
/// cover Llama Q4_K_M GGUFs (q/k/o/gate/up are Q4_K; v/down/embed are Q6_K).
/// Q1_0 (24) is the prism-ml Bonsai-27B 1-bit scheme — scratch-free here avoids
/// re-materializing the whole 27B to f32 every token (~200s/tok → fused) AND
/// avoids the multi-GiB dequant scratch that blew wgpu's storage-buffer binding
/// limit.
///
/// Q8_0 is deliberately not here although the tiled kernel has a branch for its
/// one row: this set is also the slab booking's skip
/// ([`dequant_gguf_scratch_bytes`]).
pub fn gemv_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0 | 2 | 24)
}

/// Whether the tiled one-row kernel (`dequant_gemv_tiled`) has a branch for this
/// scheme: the K-quants, whose 256-element blocks this arm reads through
/// 32-element units (what a unit *is* differs per scheme — see its shader
/// header), and Q8_0, whose own 32-element block IS that unit — so its
/// contraction's granularity is the block, not the K-quants' 256-element tile.
/// Each arm's shader branches are tied to its routing by
/// `tests/gguf_kernel_scheme_coverage.rs`.
pub fn gemv_tiled_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0 | 2 | 20)
}

/// Whether this call's single row is the tiled arm's — its scheme's branch over
/// its own whole blocks. [`select_path`] admits the call on it and
/// [`encode_dequant_matmul_gguf_into`] refuses one that fails it, so no caller
/// can encode a contraction the kernel would floor to whole blocks.
fn one_row_arm_ok(scheme_id: u32, m: usize, k: usize) -> bool {
    m == 1
        && gemv_tiled_supports_scheme(scheme_id)
        && k.is_multiple_of(scheme_from_id(scheme_id).gguf_block_size() as usize)
}

/// Whether a device route this call is admitted to can read its X window.
///
/// The tiled arm reads X as `vec4` (8 vector loads per unit instead of 32 scalar
/// ones), so the window's own offset and the input row stride (`x_row_bytes`, the
/// row's `k` elements — not `gguf_row_bytes`, the packed weight's stride) must
/// both line up with a four-element group. True at the shapes these schemes are
/// used at — the arena aligns every tensor to 16 bytes and `k` is a multiple of
/// 4 wherever this arm is admitted at all (the K-quants' 256, Q8_0's 32) — and
/// false for a view the route then hands to the host rather than reading X
/// shifted. The grouped GEMM checks the same two things for its own
/// `vec4` X reads, and [`select_path`] composes this one guard for the multi-row
/// admission as well. Pinned device-free by `tests/gguf_kernel_scheme_coverage.rs`.
pub fn gemv_one_row_device_ok(scheme_id: u32, x_byte_off: usize, x_row_bytes: usize) -> bool {
    !gemv_tiled_supports_scheme(scheme_id)
        || (x_byte_off.is_multiple_of(16) && x_row_bytes.is_multiple_of(16))
}

/// Which entry point the run path calls for one packed multiply — the ladder's
/// single answer for the call, taken before anything is encoded. The step arm
/// dispatches on it and the batch boundary joins on it, so entry point and
/// boundary cannot disagree.
///
/// **Batch join:** the run encodes the call's dispatches into its own enclosing
/// batch instead of letting the call submit on its own exactly for the answers
/// [`Self::is_provably_device_only`] admits.
///
/// A variant names the entry point, NOT the route `route_counts` ends up
/// recording: the fused and scratch arms may still hand the call to
/// [`crate::gguf_host`] when their own arm refuses it, and
/// [`DequantMatmulPath::ScratchArm`] delegates to the fused arm when the scheme is
/// one [`gemv_supports_scheme`] carries. Read the route off `route_counts`, or ask
/// `DequantMatmulPath::is_provably_device_only`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantMatmulPath {
    /// The one-row tiled arm ([`dequant_gemv_gguf_tiled_kernel`]) with an
    /// X window it can address: fused windowed kernels, no host read, and one of
    /// the two routes whose dispatches a caller can encode itself
    /// ([`encode_dequant_matmul_gguf_into`]) instead of letting the call submit
    /// on its own. Its schemes are [`gemv_tiled_supports_scheme`]'s.
    OneRowDevice,
    /// The row-grouped GEMM at `m > 1` — the multi-row half of
    /// [`Self::OneRowDevice`]'s route for the K-quants, and Q8_0's own
    /// multi-row arm, both encoded the same way
    /// ([`encode_dequant_matmul_gguf_into`]) instead of letting the call submit on
    /// its own. For the tiled schemes the `k` gate is inherited from the `fused`
    /// term of [`select_path`], which for them *is* their own packing block — so an
    /// admitted call cannot fall to the per-row loop, whose tiled arm floors `k` to
    /// whole blocks. For Q8_0, whose own 32-element block is one of the eight the
    /// staged tile holds, [`grouped_k_ok`] is the gate ([`select_path`]'s own
    /// term, and the arm's scheme set is folded into it).
    ///
    /// The arm's own gates can still decline the call, and what that means differs
    /// by scheme: for one the fused entry point carries it is a fall-through, read
    /// through the per-row loop exactly as an unjoined call of that shape is, while
    /// for Q8_0 it is a refusal — that loop carries its one row alone, this arm being
    /// above it — and the caller has to route the call to the entry point that
    /// recomputes it. These gates read arena and adapter
    /// facts this pure ladder cannot see, and what they decline is a window that does
    /// not fit (X's own, an N-chunked weight window, the binding budget, an adapter
    /// limit, the bisect lever) — never a striped arena as such: the question is
    /// whether a window fits, and a call laid out inside one stripe has windows that
    /// do. Where today's scratch entry point sends such a call to
    /// [`crate::gguf_host`] instead (`is_sharded`, `is_weight_off`), the admission
    /// moves it onto this arm: the parked-weight and the virtual bind-stripe
    /// regimes, both measured by `tests/gguf_gemm_parked_weight.rs` against the crate's
    /// 1e-2 cross-route bar and not as bit-identity. A physically striped arena is the
    /// same movement whenever its windows fit their stripe, and is unmeasured — the
    /// shard levers are off by default and no measured cell takes them. See
    /// [`encode_dequant_matmul_gguf_into`] for the refusal itself.
    GroupedDevice,
    /// The fused windowed entry point ([`run_dequant_matmul_gguf_gemv_rows`]):
    /// the per-row fallback for a stranded call, the Q1_0 arm, and a call whose X
    /// window the tiled arm refuses — which that entry point hands to
    /// [`crate::gguf_host`] and records as the host route.
    FusedArm,
    /// The dequant-scratch entry point ([`run_dequant_matmul_gguf_gpu`]) with a
    /// slab booked — the answer for the calls the ladder's gates keep out of
    /// [`DequantMatmulPath::GroupedDevice`]: a contraction that is not a whole
    /// number of the scheme's own packing blocks, an X window its `vec4` reads
    /// cannot address, a single row of a scheme no one-row arm carries, or a
    /// multi-row call of a scheme neither the grouped GEMM nor the fused entry point
    /// serves (both of the latter two only when the slab is booked; with none they
    /// answer [`DequantMatmulPath::Host`]).
    /// That entry point's own arms decide next — dequant scratch + `matmul_bt`, the
    /// row-grouped GEMM, the fused arm's per-row loop, or [`crate::gguf_host`] when
    /// it refuses the call — and its grouped and fused arms apply the very gates
    /// this ladder did, so the shapes above land on the slab. The slab is that entry
    /// point's fallback, not a statement of the route.
    ScratchArm,
    /// Straight to [`crate::gguf_host::run_dequant_matmul_gguf`].
    Host,
}

impl DequantMatmulPath {
    /// Whether this call is known to compute entirely on the device — true for
    /// [`DequantMatmulPath::OneRowDevice`] and
    /// [`DequantMatmulPath::GroupedDevice`], the two answers whose work a caller can
    /// encode into its own submission. This is the batch boundary's own predicate, so
    /// the join set is this selection and not a second rule; `pub` so
    /// [`select_path`]'s callers and the device-free pin in
    /// `tests/gguf_kernel_scheme_coverage.rs` read the real predicate rather than
    /// restating it. An encode can still refuse such a call, and the other variants
    /// reach [`crate::gguf_host`] without an encode to answer for it, so nothing may be
    /// scheduled on the assumption that they are device-only — see
    /// [`DequantMatmulPath::GroupedDevice`] for where a refusal lands.
    pub fn is_provably_device_only(self) -> bool {
        matches!(self, Self::OneRowDevice | Self::GroupedDevice)
    }
}

/// The ladder's selection for one call, callable from anywhere the caller has the
/// step's shape and the plan's dequant slab offset. `m` is the ROW COUNT IN USE
/// (the step arm's active-extent-scaled one), `k` the contraction width, and
/// `scratch_off` the slab the plan booked (0 when it booked none).
///
/// Deliberately pure, and deliberately blind: it reads no device, no adapter
/// limit, no arena and no lever beyond the inputs above, so a caller may ask it
/// what the route will be without perturbing [`route_counts`] — and so its
/// answer over-approximates. The call's own encode is where a shape or a device
/// this selection could not rule out is refused.
pub fn select_path(
    m: usize,
    k: usize,
    scheme_id: u32,
    x_byte_off: usize,
    scratch_off: usize,
) -> DequantMatmulPath {
    // The step arm's own selector, verbatim: the fused entry point serves the
    // K-quants and Q1_0 — the former at one row, the latter at any row — and
    // itself only when the contraction is a whole number of packing blocks, plus
    // the tiled arm's own schemes' single row ([`one_row_arm_ok`]).
    let fused = (gemv_supports_scheme(scheme_id)
        && (m == 1
            || scheme_id == 24
            || k.is_multiple_of(scheme_from_id(scheme_id).gguf_block_size() as usize)))
        || one_row_arm_ok(scheme_id, m, k);
    if !fused {
        // The scratch entry point's own ladder, split: a multi-row call of a scheme
        // the row-grouped GEMM serves, with the slab the plan booked behind it, over
        // whole packing blocks of that scheme ([`grouped_k_ok`], which carries the
        // scheme term too) and an X window its `vec4` reads can address, is
        // device-only work whose arm a caller can encode itself. Every other call
        // keeps today's answer: a single-row form off its arm's own shapes, a
        // contraction off the scheme's blocks or a multi-row call of a scheme neither
        // the row-grouped GEMM nor the fused entry point serves (the slab), no slab
        // at all (the host).
        if scratch_off > 0 && m > 1 && grouped_k_ok(scheme_id, k) && x_window_vec4_ok(x_byte_off) {
            return DequantMatmulPath::GroupedDevice;
        }
        return if scratch_off > 0 {
            DequantMatmulPath::ScratchArm
        } else {
            DequantMatmulPath::Host
        };
    }
    // The tiled arm's schemes with an X window its `vec4` reads can address — the
    // same guard the grouped GEMM applies to its own `vec4` X reads, and the one
    // the fused entry point re-checks before handing the call to the host. One row
    // is the arm a caller encodes itself; above one row the same entry point
    // reaches the row-grouped GEMM, which a caller encodes the same way. The `k`
    // that arm needs is `fused`'s own term above rather than a rule stated here —
    // see [`DequantMatmulPath::GroupedDevice`].
    if gemv_tiled_supports_scheme(scheme_id) && gemv_one_row_device_ok(scheme_id, x_byte_off, k * 4)
    {
        return match m {
            // No call at all keeps the fused arm it has today.
            0 => DequantMatmulPath::FusedArm,
            1 => DequantMatmulPath::OneRowDevice,
            _ => DequantMatmulPath::GroupedDevice,
        };
    }
    DequantMatmulPath::FusedArm
}

/// Route a batched packed GGUF multiply can take.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DequantMatmulRoute {
    /// Row-grouped fused GEMM: each packed weight tile is read once per group
    /// of 8 rows (Q4_K / Q6_K / Q8_0, `m > 1`), no f32 scratch — except that a
    /// chunk the Q4_K entry point splits reads each block's 16-byte header twice,
    /// its quant bytes still once.
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
/// arm splits a column's contraction over 32 lanes, one lane per 32-element unit
/// of the row (a Q4_K sub-block, or half a Q6_K `ql` span read at both of its
/// nibble sets — see the shader header); the legacy arm gives each of its 64
/// threads one output column.
///
/// The shader's declared `@workgroup_size` and the shape the route records must
/// both be these numbers: the device-free `tiled_gemv_geometry_matches_the_shader`
/// below derives the tiled arm's `@workgroup_size` (and its `LANES`/`COLS_PER_WG`)
/// from these constants, `tests/gguf_kernel_scheme_coverage.rs` pins the legacy
/// arm's size and the portable floor, and `tests/gguf_matmul_route.rs` pins the
/// recorded shape.
const GEMV_TILED_THREADS: u32 = 32;
const GEMV_LEGACY_THREADS: u32 = 64;

/// Output columns one workgroup of the tiled arm covers. Two, so a workgroup
/// covers two output columns and splits each one's contraction over its threads,
/// reusing one X read for both; the dispatch is then `ceil(columns / this)`
/// workgroups wide, which is the form both the dispatch cap in
/// [`gemv_one_row_cols_per_chunk`] and the recorded shape below are written in.
const GEMV_TILED_COLS_PER_WG: u32 = 2;

/// Most workgroups one dispatch may have along a dimension — the WebGPU floor
/// (`max_compute_workgroups_per_dimension`), which every backend honours. Both
/// one-row arms are chunked to it: at `GEMV_TILED_COLS_PER_WG` output columns per
/// workgroup on the tiled arm, and at `GEMV_LEGACY_THREADS` columns per workgroup
/// on the one-thread-per-column arm, so the column cap is this times the arm's own
/// workgroup width.
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
    /// Output columns one workgroup serves. `GEMV_TILED_COLS_PER_WG` (two) on
    /// the tiled arm; `64` while the arm is one thread per output column.
    pub cols_per_workgroup: u32,
    /// Threads one workgroup runs. On the tiled arm they split each of those
    /// columns' contractions; on the legacy arm each thread serves its own
    /// column.
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
/// `workgroups * cols_per_workgroup` is the output width the dispatch covered
/// rounded up to a whole workgroup — a *ceiling*, not an equality: a workgroup
/// whose second column is off the end of the chunk computes nothing for it (its
/// second accumulator stays zero) and skips that store, so no row or column is
/// ever discarded. (The legacy arm's last workgroup rounds up to 64 columns the
/// same way.)
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
            //
            // Q8_0 keeps its booking: the slab is this scheme's route for the
            // one-row calls the tiled arm's own block gate keeps out and for every
            // shape its grouped arm declines, and `select_path` reads the reservation
            // as the plan's willingness to reserve memory — the term that also admits
            // a multi-row call of this scheme to the grouped arm and to the batch
            // boundary's join. Dropping it would send those calls to the host
            // instead, which is a regression, not a saving.
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

/// The schemes the row-grouped GEMM serves, each with the kernel that carries
/// it. The one place that set is named: [`grouped_gemm_kernel`] is this list read
/// with a device and [`grouped_scheme_supported`] is it read without one, so the
/// ladder that admits a multi-row call of a scheme and the encoder that dispatches
/// it cannot drift into admitting a scheme no kernel would run — a drift that
/// would stride the wrong entry point's format rather than fail.
const GROUPED_SCHEMES: [(u32, fn(&wgpu::Device) -> &'static Kernel); 3] = [
    (0, dequant_gemm_q4k_kernel),
    (2, dequant_gemm_q6k_kernel),
    (20, dequant_gemm_q8_0_kernel),
];

/// Whether the row-grouped GEMM has an arm for `scheme_id` — the `GROUPED_SCHEMES`
/// list without a device, which is the only form of it a device-free caller can
/// read. [`grouped_k_ok`]'s scheme term is where the ladder reads it, and
/// `tests/gguf_kernel_scheme_coverage.rs` pins the list itself, over every
/// registered id.
pub fn grouped_scheme_supported(scheme_id: u32) -> bool {
    GROUPED_SCHEMES.iter().any(|(id, _)| *id == scheme_id)
}

/// Whether the row-grouped GEMM covers a contraction `k` of `scheme_id`: the arm
/// has a kernel for that scheme ([`grouped_scheme_supported`]) and `k` is a whole
/// number of its **own packing blocks**, the granularity its entry point's weight
/// loop reads. The arm's `k` gate, asked by [`select_path`] of the shape it is
/// handed and by [`encode_dequant_matmul_gguf_gemm_grouped`] of the one it is about
/// to dispatch, so a call the ladder admits cannot then be declined for its
/// contraction. The scheme term is folded in rather than left to the caller, so the
/// answer is defined for any id: [`scheme_from_id`] panics on one no table carries.
///
/// For the K-quants that block *is* [`GROUPED_STAGE_ELEMS`] — their entry points
/// read whole staged tiles and nothing else — so this is the gate they have always
/// had. Q8_0's block is one of the eight a tile holds, so its rows may end inside
/// one: the `dequant_gemm_q8_0` entry point's staged-tile loop is written to stop at
/// the last whole block of such a tail tile (`BLK_ELEMS` there is the tile, not the
/// unit), which is what lets a row of several tiles plus a block — and the sub-tile
/// rows this allows — take this route instead of the slab. The surplus the tile
/// stages past `k` is never read: the block loop stops before it.
pub fn grouped_k_ok(scheme_id: u32, k: usize) -> bool {
    grouped_scheme_supported(scheme_id)
        && k.is_multiple_of(scheme_from_id(scheme_id).gguf_block_size() as usize)
}

/// The row-grouped GEMM's entry point for `scheme_id`, or `None` for a scheme it
/// has no arm for.
fn grouped_gemm_kernel(scheme_id: u32, device: &wgpu::Device) -> Option<&'static Kernel> {
    GROUPED_SCHEMES
        .iter()
        .find(|(id, _)| *id == scheme_id)
        .map(|(_, kernel)| kernel(device))
}

/// Whether a row-grouped GEMM call's X window begins on a four-element group,
/// which is what that kernel's `vec4` X reads need. The encoder rounds the window
/// offset down to a [`STORAGE_ALIGN`]-aligned base — 256, a multiple of 16 — so
/// `x0 - x_base` is a multiple of 16 exactly when `x0` is; this is that gate's
/// one statement, applied by the encoder to the window it resolved and by
/// [`select_path`] to the byte offset it is handed, so a call the ladder admits
/// cannot then be declined for its X window.
fn x_window_vec4_ok(x_byte_off: usize) -> bool {
    x_byte_off.is_multiple_of(16)
}

fn launch_dequant_gguf(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w_byte_off: usize,
    scratch_byte_off: usize,
    scheme_id: u32,
    num_blocks: usize,
    marks: &mut PassMarks<'_>,
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
        let mut pass = marks.begin(&mut enc, "rlx-wgpu dequant_gguf pass");
        pass.set_pipeline(&dk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid, 1, 1);
    }
    crate::backend::note_device_submit();
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
    marks: &mut PassMarks<'_>,
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
        let mut pass = marks.begin(&mut enc, "rlx-wgpu gguf matmul_bt pass");
        pass.set_pipeline(&mm.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(32), m.div_ceil(32), 1);
    }
    crate::backend::note_device_submit();
    queue.submit(std::iter::once(enc.finish()));
}

/// Launch the fused ladder for a scheme that has an arm for this call — the
/// windowed GEMV for a one-row K-quant, the row-grouped GEMM for a multi-row
/// K-quant or Q8_0 — and otherwise `dequant_gguf` into arena scratch, then
/// `C = X @ W^T` via matmul_bt.
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
    marks: &mut PassMarks<'_>,
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
    // Q4_K/Q6_K/Q1_0: scratch-free windowed kernels — one submit from this fused
    // entry point for batched rows, row-grouped for m>1 Q4_K/Q6_K. The K-quant
    // windowed arm with an addressable X window — one row, and the row-grouped
    // GEMM above one — is instead encoded into the caller's enclosing batch at run
    // time ([`DequantMatmulPath::OneRowDevice`] / [`DequantMatmulPath::GroupedDevice`]).
    if gemv_supports_scheme(scheme_id) {
        // A scheme in [`gemv_supports_scheme`] is always run by that entry point:
        // its own per-row loop is the arm its grouped GEMM declines into (see
        // `encode_dequant_matmul_gguf_into`), and a scheme outside that set is the one
        // condition under which the ladder can answer `false` — so there is nothing
        // left to route here.
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
            marks,
        );
        return;
    }
    // A scheme outside the fused entry point's own set — Q8_0 — has no per-row
    // fallback there either, so its multi-row calls take the row-grouped GEMM of the
    // same ladder the fused entry point uses, and this is the entry point that owns
    // the recompute of one the batch boundary joined and that entry point's encode
    // refused. A call the arm declines (a `k` off the scheme's own blocks, an X window
    // over the binding limit or not four-element addressable, an adapter limit, its
    // bisect lever) falls through to the scratch slab below — this scheme's route for
    // the single rows its tiled arm's own block and window gates keep out, and at
    // every shape its grouped arm refuses. The two short-circuits above are the facts
    // this pure ladder cannot see — `is_sharded` (a striped arena, virtual bind
    // stripes or physical shards) and `is_weight_off` (a weight parked in the weight
    // buffer) put those classes on the host route *here*, while an admitted call of
    // either class reaches the device arm, where its windows fit (see
    // [`DequantMatmulPath::GroupedDevice`]). The gate is [`GROUPED_SCHEMES`], the set
    // the ladder admits from, so the two cannot disagree.
    if m > 1 && grouped_gemm_kernel(scheme_id, device).is_some() {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu dequant_gguf grouped"),
        });
        if encode_dequant_matmul_gguf_into(
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
            marks,
        ) {
            crate::backend::note_device_submit();
            queue.submit(std::iter::once(enc.finish()));
            return;
        }
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
        marks,
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
        marks,
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
/// row-grouped GEMMs first, then the per-row GEMV loop — through which a multi-row
/// call the grouped GEMM's own gates decline is read ([`DequantMatmulPath::GroupedDevice`]
/// lists them), the same route an unjoined call of that shape takes. For a multi-row
/// Q8_0 call, whose per-row loop carries its one row alone, those same declines are
/// the *refusal* below instead, its bisect lever `RLX_WGPU_Q8_0_GEMM_DISABLE`
/// included. Uniforms are unique per
/// dispatch (`queue.write_buffer` is not ordered inside an encoder the way copies
/// are).
///
/// Returns whether this ladder has an arm for the call: `false` when it does not —
/// a scheme outside [`gemv_supports_scheme`] above one row, whose row-grouped arm
/// declined, for which the per-row loop has no branch (its leg dispatch would match
/// nothing and read the weights as zeros). Nothing is encoded in that case and the
/// caller owns the fallback, [`run_dequant_matmul_gguf_gpu`] for such a scheme — so
/// this answer cannot be discarded, by either caller: the run's join branch reads it
/// from its own call here, and [`run_dequant_matmul_gguf_gemv_rows`] from its.
///
/// The per-row loop's tiled arm reads X as `vec4`, so X's window must be
/// four-element addressable. [`run_dequant_matmul_gguf_gemv_rows`] checks that
/// with [`gemv_one_row_device_ok`] for the routes it owns, and the run's join
/// decision asks [`select_path`] — the same ladder, whose device-only answers
/// compose that guard — before calling here directly. This gate carries the
/// 8-bit scheme's own block term ([`one_row_arm_ok`]), so a direct caller has to
/// make sure of the window alone.
///
/// `marks` is the run's pass-marks handle: every pass this ladder opens carries
/// it, so the report's device view covers a packed multiply's passes as it covers
/// the batch's.
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
    marks: &mut PassMarks<'_>,
) -> bool {
    if m == 0 {
        return true;
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
            marks,
        )
    {
        note_route(DequantMatmulRoute::Q1_0Gemm);
        return true;
    }
    // Batched, any scheme this kernel set covers: one row-grouped GEMM per weight
    // tile, instead of the per-row loop below re-streaming the whole packed weight
    // per row. Which schemes it has an entry point for is decided inside, before
    // any size math. Falling back is driven by what can be *encoded* (X window),
    // never by the weight window merely exceeding the binding limit — that case is
    // N-chunked inside.
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
            marks,
        )
    {
        note_route(DequantMatmulRoute::Grouped);
        return true;
    }
    // No arm for this scheme in this ladder above one row: the loop below carries the
    // fused arms' schemes, and one outside them is the caller's fallback (Q8_0's is
    // the scratch slab), not a per-row pass of a kernel that has no branch for it —
    // except the tiled kernel's own single-row case.
    if !gemv_supports_scheme(scheme_id) && !one_row_arm_ok(scheme_id, m, k) {
        return false;
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
            marks,
        );
    }
    note_route(DequantMatmulRoute::PerRow);
    true
}

/// Fused decode/prefill path: `Y[m,n] = X[m,k] @ W^T` with packed GGUF `W`.
///
/// * Q4_K / Q6_K / Q8_0 + `m > 1` → row-grouped tiled GEMM — one pass over each
///   packed weight tile per group of 8 rows (unless X spans shards); on a split
///   chunk the Q4_K block headers are read twice, the quant bytes still once.
/// * Q1_0 + `m > 1` → tiled GEMM (Metal `q1_0_mm_f32` parity) — one dispatch,
///   unless `RLX_WGPU_Q1_0_GEMM_DISABLE=1` or X spans shards.
/// * Otherwise → per-row GEMV, encoded into one command buffer and submitted
///   here — except the calls [`DequantMatmulPath::is_provably_device_only`] admits
///   (one row of a tiled scheme, and the row-grouped GEMM above it for a tiled or a
///   Q8_0 one), which the run may encode into the caller's enclosing batch instead of
///   letting this entry point submit them.
///
/// An X window the tiled arm's `vec4` read cannot address sends the whole call
/// to [`crate::gguf_host`]. Every caller of the tiled arm has to hold that
/// condition: this entry point checks it for the routes it owns, and the run's
/// join decision asks [`select_path`] — the same guard, composed into the ladder —
/// before calling [`encode_dequant_matmul_gguf_into`] directly.
///
/// Returns whether this entry point ran the call — `true` for dispatches it encoded
/// and submitted here, and for a call it handed to [`crate::gguf_host`]; `false` for
/// one the ladder has no arm for, which [`encode_dequant_matmul_gguf_into`] declined
/// without encoding anything. Nothing is submitted then, and the caller owns the
/// fallback ([`run_dequant_matmul_gguf_gpu`], the recompute owner for such a scheme):
/// submitting the empty encoder instead leaves the output slot unwritten with no
/// counter and no error. The run's step arm is the caller that has to read it; the
/// batch boundary asks [`encode_dequant_matmul_gguf_into`] directly and reads the
/// same answer there.
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
    marks: &mut PassMarks<'_>,
) -> bool {
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
        return true;
    }
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gguf rows"),
    });
    if !encode_dequant_matmul_gguf_into(
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
        marks,
    ) {
        // Nothing was encoded, so there is nothing to submit. Hand the call back
        // rather than submit an empty encoder: see the doc.
        return false;
    }
    crate::backend::note_device_submit();
    queue.submit(std::iter::once(enc.finish()));
    true
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
    marks: &mut PassMarks<'_>,
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
            let mut pass = marks.begin(enc, "rlx-wgpu dequant_gemm_q1_0 pass");
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
/// align-down base slack (under [`STORAGE_ALIGN`]) and its 16-byte round-up. Both
/// the row-grouped GEMM's column chunking and the per-row GEMV's own column
/// chunking are sized from this, so they cannot drift apart.
fn gguf_window_budget(max_bind: u64) -> u64 {
    max_bind.saturating_sub(STORAGE_ALIGN + 16)
}

/// Output columns one *dispatch* of a one-row GEMV covers: what fits the weight
/// window ([`gguf_window_budget`]) and the dispatch-dimension cap, which counts
/// workgroups and so reaches `GEMV_MAX_WORKGROUPS * GEMV_TILED_COLS_PER_WG`
/// columns on the tiled arm and a whole `GEMV_LEGACY_THREADS` per workgroup —
/// one thread per column — on the arm that keeps the original shape.
///
/// The encoder chunks its N loop by this and
/// [`gemv_one_row_dispatches`] reports it, so the rule has one spelling.
fn gemv_one_row_cols_per_chunk(row_bytes: u64, max_bind: u64, tiled: bool) -> usize {
    let cols_per_wg = if tiled {
        GEMV_TILED_COLS_PER_WG as usize
    } else {
        GEMV_LEGACY_THREADS as usize
    };
    // The `.max(1)` is what makes a row wider than the whole budget one column per
    // chunk, i.e. still progress; `row_bytes.max(1)` only keeps the division defined.
    ((gguf_window_budget(max_bind) / row_bytes.max(1)) as usize)
        .max(1)
        .min(GEMV_MAX_WORKGROUPS * cols_per_wg)
}

/// Packed-weight bytes one output row (one column of the weight) occupies at
/// this shape — the per-row stride the encoders chunk by and the shaders index by
/// (`col * nbk * block_bytes`), taken from the scheme's block geometry rather
/// than restated.
///
/// Public for the measurement cells, which quote a tensor's packed bytes and
/// compare a chunking budget against them ([`grouped_dispatch_cols`]) —
/// `gemv_one_row_dispatches`'s reason. Nothing in the encoder calls it from
/// outside this module.
pub fn gguf_row_bytes(scheme_id: u32, k: usize) -> u64 {
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    ((k / block_elems.max(1)) * block_bytes) as u64
}

/// How many dispatches one step (one weight region, one input row) of the
/// one-row packed multiply takes at this shape — the chunking rule above, from
/// the binding budget and the dispatch cap, which is where the count comes from.
///
/// Reported by the stream-rate cell rather than re-derived there, so a change to
/// the rule or to the arm's own dispatch geometry moves the reported count with
/// it. Public for that cell (an integration test cannot reach `pub(crate)`
/// surface); nothing else calls it. `None` for a scheme with no one-row arm.
pub fn gemv_one_row_dispatches(
    device: &wgpu::Device,
    scheme_id: u32,
    k: usize,
    n: usize,
) -> Option<usize> {
    // Either one-row arm's scheme: the tiled kernel's own set, and the fused
    // ladder's, whose Q1_0 is the legacy arm's.
    if !gemv_tiled_supports_scheme(scheme_id) && !gemv_supports_scheme(scheme_id) {
        return None;
    }
    let tiled = gemv_tiled_supports_scheme(scheme_id);
    Some(n.div_ceil(gemv_one_row_cols_per_chunk(
        gguf_row_bytes(scheme_id, k),
        gguf_max_bind(device),
        tiled,
    )))
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

/// Threads in one workgroup of the row-grouped GEMM — the shader's `TG`, from
/// which the dispatch geometry below is sized: a workgroup covers this many
/// output columns, one thread each, or [`GROUPED_SPLIT_COLS`] columns with two
/// threads to each in the split geometry [`GROUPED_SPLIT_MAX_BASE_WORKGROUPS`]
/// describes.
const GROUPED_THREADS: u32 = 64;

/// Rows one workgroup of the row-grouped GEMM covers — the shader's `TM`, the
/// tile over which one pass over the packed weight is reused. Both are tied to
/// the shader they describe by the `grouped_geometry_matches_the_shader` test
/// below, which is why they stay private here.
///
/// The tile is a measurement and not a free knob: rows per thread is the lever
/// this const sizes — the row tile, one row group — and the larger-row arms
/// measured slower on both shapes at both sizes (the owner's re-take on the stand —
/// an Apple M3 Pro, 14-core GPU, 18 GiB, macOS, wgpu over Metal; sides alternated
/// in adjacent rounds, the median of three). A **step** in that re-take is a model
/// body's whole step, every packed multiply the body runs rather than one multiply
/// — not the sense this file's one-row-path docs give the word — which is why these
/// figures sit far above this crate's per-multiply ones and are not comparable with
/// them. The arms were not the shipped kernel with only that constant moved, so what
/// was measured is recorded against the arms as measured rather than as a law about
/// tile size, and no mechanism is claimed for the loss:
///
/// | step | 8-row tile | 16 rows | 32 rows |
/// | --- | --- | --- | --- |
/// | the recognition body's, m = 104 | 450 ms | 1.18 s | 3.51 s |
/// | the synthesis body's, m = 24 | 132 ms | 325 ms | 1.05 s |
///
/// Nor are the larger rows a free knob in a second sense: on the 32-row arms
/// 89.6 % of the output elements differed from the baseline, max |delta| = 1.87e-4
/// against a max |.| of about 28 — summation-reordering round-off, not a wrong
/// answer. The 16-row arms' output is not recorded either way. The 104-row cell
/// also appears in the shader header's duplicate-read probe, which is a separate
/// experiment and not this one. The walk and cost facts this reading sits beside —
/// what a step does and does not pay for, and the per-group price of one further
/// pass over the weight — are in that header's re-take paragraphs, and the two move
/// together.
const GROUPED_ROWS: u32 = 8;

/// Elements of `k` one staged activation tile covers — the shader's `BLK_ELEMS`,
/// which sizes its `xs` tile and the barrier pair the weight loop is written
/// around. Every arm of this entry set reads whole packing blocks out of it: the
/// tile is one 256-element Q4_K/Q6_K block, and eight of Q8_0's 32-element ones —
/// so for the K-quants this is also their `k` gate, while Q8_0's is its own block
/// ([`grouped_k_ok`]), whose tail tile the `dequant_gemm_q8_0` entry stops early
/// on. Pinned against the shader's own const by
/// `grouped_geometry_matches_the_shader`.
pub const GROUPED_STAGE_ELEMS: usize = 256;

/// Columns one workgroup of the row-grouped GEMM covers with the split geometry
/// on — the shader's `COLS`, half of [`GROUPED_THREADS`] because the two halves
/// are its two k-lane groups (the shader's column and lane indexing assumes that
/// pair, and `grouped_geometry_matches_the_shader` pins it). It sizes that
/// geometry's x grid, and is tied to the shader's own const like the rest of this
/// geometry.
const GROUPED_SPLIT_COLS: u32 = 32;

/// Workgroups in the **base** geometry's dispatch at or below which the Q4_K
/// entry point runs the split geometry instead.
///
/// A dispatch of this kernel costs about the same whether it has 16 workgroups or
/// 500, so a starved one leaves the device idle. What the split is worth there is
/// the author's own **out-of-tree** raw-kernel A/B on one adapter (wgpu over Metal,
/// M3 Pro) — one process, one set of buffers, the two geometries interleaved
/// round-robin over the same packed bytes, three processes x two sweeps of 28
/// shapes, the min of each arm's medians — and is to be re-taken on the adapter at
/// hand rather than trusted: **-11% .. -27%** at or below this ceiling (m = 8 .. 24,
/// 13 .. 32 base workgroups), growing with `k` (m = 13, n = 1024 is -12% at k = 1024
/// and -27% at k = 6144); it still wins at 33 .. 52 base workgroups (-4% .. -21%),
/// is a wash at 60 .. 64, and from 72 up loses (+2% .. +16%), where the base
/// geometry already fills the GPU and the second warp per column gives nothing back
/// while its second read of every block's 16-byte header, its second pass at the
/// staged activations and its fold stay paid. The in-tree
/// `tests/gguf_gemm_grouped_split_rate.rs` is not that A/B: it measures the geometry
/// the encoder itself picks for each shape — one geometry per shape, printing that
/// geometry, the bytes it reads and the per-shape times — and no split-versus-base
/// pair.
///
/// It is a ceiling on the product of the grid's two dimensions rather than on
/// either alone because a narrow `n` starves a dispatch exactly as a small `m` does
/// (m = 100, n = 64 is 13 workgroups); [`grouped_chunk_geometry`] combines them.
/// wgpu exposes no compute-thread-capacity limit to derive the cut from (the
/// transition measured here sits between 52 and 72 workgroups, about three per core
/// on an 18-core adapter), so this ceiling is a measurement rather than a
/// derivation: a GPU that saturates earlier loses where this wins, a larger one
/// could raise this const.
///
/// This arm's presence costs the base geometry **+0.9% .. +4.5%** over this module's
/// pre-split kernel (same 28 shapes, six samples each, interleaved round-robin,
/// against a positional control — the same two pipelines timed from other positions
/// in the round-robin, averaging 0.05%). The same protocol reads +0.2% .. +3.5% on
/// the untouched Q6_K entry point of this module, so this tree cannot separate that
/// residue from the protocol's own spread: it is recorded as an exposure, not
/// claimed to be zero.
///
/// The reach is per **chunk**, not per call: the geometry follows the chunk width,
/// which follows the weight window — the adapter's binding budget and
/// `RLX_WGPU_MAX_BIND_MB` — against the scheme's own row stride, so a smaller budget
/// shortens the chunks and can move a form onto the split that the full-width
/// dispatch leaves on the base geometry, and whether it does depends on the stride
/// as well. One call can mix the two geometries between its chunks (its trailing
/// chunk is the usual candidate). At `RLX_WGPU_MAX_BIND_MB=1` a Q4_K weight at
/// k = 1024 is chunked 1819 columns to a dispatch, so the 13-row, n = 6144 form is
/// three 1819-column chunks on the base geometry (a 29 x 2 = 58 base grid, above
/// this ceiling) and a trailing 687-column one on the split (a 22 x 2 grid over the
/// same rows); `grouped_split_reach_at_the_body_shapes` pins both widths' verdicts.
///
/// Chunking alone is value-neutral on this route (a chunk boundary decides which
/// dispatch reads which columns, not the arithmetic), so a chunk that crosses the
/// ceiling is where the route's last bits start to depend on the host and the
/// environment — the chunk width, and so the adapter's binding budget and
/// `RLX_WGPU_MAX_BIND_MB`: the fold adds a column's two half-`k` partials in a fixed
/// order the base geometry has no equivalent of, so a split chunk and a base chunk of
/// the same weight can round differently in the last bits (2.4414e-4 against 2.2888e-4
/// for the two geometries, from the same out-of-tree A/B as the band and to be
/// re-taken with it — no committed cell prints a split-versus-base pair — both far
/// inside the route's tolerance) — the same graph can differ between two machines.
/// There is no switch for it; the geometry is the encoder's own answer for the chunk
/// width.
///
/// At the widths a model body's own forms are chunked at on this adapter (n = 1024
/// and 6144, one chunk each) it reaches the 13-row body at n = 1024 — 16 x 2 = 32
/// base workgroups — and neither that body's n = 6144 node (96 x 2 = 192) nor any
/// of the 24-, 48-, 100- or 104-row ones, whose base grids are 48 workgroups and up.
/// A **chunk** narrow enough to starve those grids reaches it at any row count: 640
/// columns at 24 rows, 320 at 48, 128 at 100 and 104, and 2048 at 8 — each with a
/// width past it pinned by `grouped_split_reach_at_the_body_shapes`. Moving this
/// const moves which shapes take each geometry; re-taking the **band** needs the
/// A/B the band came from, which compares the split arm against the pre-split kernel
/// and so needs the arm compiled out — the in-tree file above re-measures the
/// encoder's own choice at the new geometry instead.
const GROUPED_SPLIT_MAX_BASE_WORKGROUPS: u32 = 32;

/// The grid one dispatch of the row-grouped GEMM covers in its **base** geometry
/// — `(x, y)` workgroups over a chunk of `n_chunk` columns and `m` rows. This is
/// what the Q6_K and Q8_0 entry points dispatch at every shape, and what the
/// Q4_K one dispatches wherever [`grouped_chunk_geometry`] does not put it on the
/// split; that function is this plus the split decision, so the arithmetic has
/// one spelling.
pub fn grouped_base_geometry(m: usize, n_chunk: usize) -> (u32, u32) {
    (
        n_chunk.div_ceil(GROUPED_THREADS as usize) as u32,
        m.div_ceil(GROUPED_ROWS as usize) as u32,
    )
}

/// The row-grouped GEMM's dispatch for one chunk: `(x, y)` workgroups and whether
/// the Q4_K entry point runs its `GROUPED_SPLIT_COLS` geometry over them. One
/// function, so the grid the encoder dispatches and the flag it writes cannot
/// drift apart, and so a test can read the encoder's own answer rather than
/// restating the rule.
pub fn grouped_chunk_geometry(scheme_id: u32, m: usize, n_chunk: usize) -> (u32, u32, bool) {
    let (bx, by) = grouped_base_geometry(m, n_chunk);
    // Q4_K only: the Q6_K and Q8_0 entry points have no split arm.
    let split = scheme_id == 0 && bx as u64 * by as u64 <= GROUPED_SPLIT_MAX_BASE_WORKGROUPS as u64;
    if split {
        // The split's workgroup covers half the columns, so its x is
        // `ceil(n_chunk / COLS)` where the base one is `ceil(n_chunk / TG)`.
        let x = n_chunk.div_ceil(GROUPED_SPLIT_COLS as usize) as u32;
        (x, by, true)
    } else {
        (bx, by, false)
    }
}

/// Output columns one *dispatch* of the row-grouped GEMM covers from the two
/// budgets that bind it: the bound weight window ([`gguf_window_budget`]) and the
/// whole-bound output slab (`OUT_SLAB_CAP` against the device limit) — `m * ncols * 4`.
/// The encoder chunks its column loop by this; [`grouped_dispatch_cols`] reports it.
/// Three floors, for three different reasons: the trailing `.max(1)` keeps a row
/// wider than the whole budget at one column per chunk, i.e. still progress;
/// `m.max(1)` below keeps the division defined for [`grouped_dispatch_cols`], which
/// is public and can be handed no rows even though the encoder declines those
/// before it dispatches; and `row_bytes.max(1)` only keeps the other division
/// defined, as in [`gemv_one_row_cols_per_chunk`].
fn grouped_cols_per_chunk(row_bytes: u64, max_bind: u64, m: usize) -> usize {
    let by_bind = gguf_window_budget(max_bind) / row_bytes.max(1);
    let by_out = OUT_SLAB_CAP.min(max_bind).saturating_sub(16) / (m.max(1) as u64 * 4);
    by_bind.min(by_out).max(1) as usize
}

/// The columns one *dispatch* of the row-grouped GEMM covers at this shape, from
/// the device's own binding budget: the weight window and the output slab
/// together (`grouped_cols_per_chunk` above), `m` and the per-row stride in.
///
/// Public for the split-rate measurement, which has to know whether its `n` is
/// one chunk before it prints a whole-tensor geometry the encoder would not
/// dispatch — [`gemv_one_row_dispatches`] is the same kind of reporting hook.
/// `n` is not an argument: the caller compares this against it.
pub fn grouped_dispatch_cols(device: &wgpu::Device, scheme_id: u32, m: usize, k: usize) -> usize {
    grouped_cols_per_chunk(gguf_row_bytes(scheme_id, k), gguf_max_bind(device), m)
}

/// Fused multi-row GGUF packed-weight GEMM for Q4_K / Q6_K / Q8_0 into `enc` (no
/// submit).
///
/// One pass over each packed weight tile per group of `TM = 8` rows, instead of
/// the per-row GEMV's one full pass over the whole weight per row — its Q4_K entry
/// point's split chunk excepted, where the quant bytes still pass once but each
/// block's 16-byte header passes twice (the paragraph below). `n` is split
/// into column chunks that fit `max_storage_buffer_binding_size` (and the
/// weight buffer's shard), so a weight parked in [`Arena::weight_buffer`] — or
/// one whose window is larger than the binding limit — still takes this path.
///
/// A chunk whose base dispatch would leave the GPU idle — at most
/// [`GROUPED_SPLIT_MAX_BASE_WORKGROUPS`] workgroups of it, [`grouped_chunk_geometry`]
/// being the rule — puts the Q4_K scheme on the split geometry instead: one
/// workgroup covers [`GROUPED_SPLIT_COLS`] columns with two half-`k` threads each
/// (its x grid is `ceil(ncols / GROUPED_SPLIT_COLS)`, its y untouched), so the
/// dispatch has up to twice the warps over the same packed quant bytes, with a
/// column's two half-`k` partial sums added in workgroup memory (one fixed order,
/// no atomics) before the store. The flag rides in the params; the Q6_K and Q8_0
/// entry points have no split arm and run the base geometry whatever it says.
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
/// limit, `k` is not a whole number of the scheme's own packing blocks
/// ([`grouped_k_ok`]), or X's own
/// window offset is not a multiple of 16 (the `vec4` reads) — all decided before
/// anything is encoded, including the per-chunk window fit and, for Q4_K, the
/// weight window's 4-alignment that its word-granular reads need (only the first
/// chunk can decline, every later one reads less and ends inside the weight
/// region). The caller then uses the per-row GEMV — which rewrites the same
/// `m*n*4` output bytes row by row — or, for a scheme with no per-row arm, the
/// scratch slab, so the fallback is correct either way.
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
    marks: &mut PassMarks<'_>,
) -> bool {
    debug_assert!(
        m > 1,
        "m == 1 goes through encode_dequant_matmul_gguf_gemv_one"
    );
    // The schemes [`GROUPED_SCHEMES`] covers, decided before any size math — so a
    // fourth is one entry in that list, and no other scheme reaches the block-size
    // check below. The `None` arm declines the call rather than waving the scheme
    // through: an id that fell back to the Q6_K entry point would decode wrong
    // values, not fail.
    let Some(dk) = grouped_gemm_kernel(scheme_id, device) else {
        return false;
    };
    // Bisect escape hatch, a lowering only: it puts a Q8_0 call back on the
    // scratch slab, which is this scheme's route before the arm at all and still
    // its route for every shape the arm declines. The default build takes the arm
    // below — obtaining it never needs this switch. Mirrors the Q1_0 GEMM's
    // (`RLX_WGPU_Q1_0_GEMM_DISABLE`) and Metal's Q8_0 GEMM's.
    if scheme_id == 20 && rlx_ir::env::flag("RLX_WGPU_Q8_0_GEMM_DISABLE") {
        return false;
    }
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize; // 256 (Q4_K/Q6_K) | 32 (Q8_0)
    let block_bytes = scheme.gguf_block_bytes() as usize; // 144 | 210 | 34
    // The entries above hardcode this geometry; if a scheme's block layout ever
    // stops matching the pair its entry point is written against, fall back
    // rather than stride the kernel wrongly.
    let geometry_ok = match scheme_id {
        20 => block_elems == 32 && block_bytes == 34,
        _ => block_elems == 256 && (block_bytes == 144 || block_bytes == 210),
    };
    if !geometry_ok {
        return false;
    }
    // A whole number of this scheme's own packing blocks: the granularity the
    // weight loop reads, and — for every arm here — what the staged tile's barrier
    // pair is written around. Q8_0's 32-element blocks let a row end inside a
    // staged tile, which that entry point's own loop handles; the K-quants' block
    // is the tile, so this is the whole-tile gate they have always had. Asked
    // through the ladder's own predicate rather than restated, so a call it admits
    // is never declined here for its contraction (`block_elems` above is this
    // scheme's copy of what [`grouped_k_ok`] divides by).
    if !grouped_k_ok(scheme_id, k) || n == 0 {
        return false;
    }
    let row_bytes = gguf_row_bytes(scheme_id, k); // packed weight bytes per column

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
    // The kernel reads X as `vec4`, through a window that begins where this
    // call's does — [`x_window_vec4_ok`] is that gate, the same one the ladder's
    // multi-row admission applies, and it is this encoder's precondition too;
    // documented on the public entry [`encode_dequant_matmul_gguf_into`] for a
    // direct caller.
    if !x_window_vec4_ok(x0 as usize) {
        return false;
    }
    let x_vec4_off = ((x0 - x_base) / 16) as u32;

    // Columns per chunk, from both budgets: the bound weight window and the
    // whole-bound output slab (m * ncols * 4). Bounding the slab rather than
    // only the weight is what keeps a wide-n, large-m head inside the limit. The
    // rule is [`grouped_cols_per_chunk`]'s, which the split-rate measurement also
    // reads through [`grouped_dispatch_cols`], so a shape this encoder dispatches
    // whole is one that cell prints as one chunk.
    if (m as u64) * 4 > max_bind {
        return false; // no column chunk's slab could be bound
    }
    let n_chunk_max = grouped_cols_per_chunk(row_bytes, max_bind, m).min(n);

    // Slab bytes for one chunk: `m * ncols * 4` out of the pooled output buffer,
    // which is process-global and only grows. It is scratch the arena plan does
    // not account for — nothing reserves it, `dequant_gguf_scratch_bytes` included
    // — so `pooled_out_buf_bytes` is what reports it.
    let out_slab = (m * n_chunk_max * 4) as u64;

    let mut ok = true;
    with_pooled_out_buf(device, out_slab, |out_buf| {
        let mut n0 = 0usize;
        while n0 < n {
            let (w_buf, w_raw) = arena.resolve_w(w_byte_off + (n0 as u64 * row_bytes) as usize);
            let w_buf_size = w_buf.size();
            let w0 = w_raw as u64;
            let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
            let n_chunk = (n - n0).min(n_chunk_max).max(1);

            // Q4_K reads the weight as whole words (`read_u32`), so this window's
            // offset has to be a multiple of 4; Q6_K's reader reassembles across a
            // 2-byte boundary and its 210-byte rows put odd chunks on a 2 anyway,
            // as do Q8_0's 34-byte blocks for its quants. Mirrors the X-side check
            // above and declines nothing that occurs today: it turns a silent
            // shifted read into the per-row path.
            if scheme_id == 0 && !(w0 - w_base).is_multiple_of(4) {
                ok = false;
                return;
            }

            // The window the chunk reads — every packed weight byte it touches.
            // That one quantity is compared against both limits, rather than the
            // 4-byte-rounded binding size below: both limits are multiples of 4
            // (the binding limit is; so is the weight buffer's end, whose size is
            // rounded up, and so is every packed weight's own byte count, since
            // `write_buffer` rejects an unaligned copy), so the two comparisons
            // agree — and a row size that is not a multiple of 4 (Q6_K rows are
            // 210·B bytes) can never make a window that fits look like one that
            // does not. `grouped_cols_per_chunk` sizes `n_chunk` inside that same
            // budget, so only a first chunk whose row is wider than the whole budget
            // — which no chunking here or in the per-row GEMV can fix — declines
            // here, before anything is encoded; every later chunk reads less than
            // the first and ends inside the weight region, so none of them can bail.
            // (A parked weight never spans buffers: `from_plan_split` spills any
            // single weight that does not fit one weight buffer into the act arena,
            // so `resolve_w` yields the whole weight region.)
            let need = (w0 - w_base) + n_chunk as u64 * row_bytes;
            if need > max_bind.min(w_buf_size.saturating_sub(w_base)) {
                ok = false;
                return;
            }
            let w_size = need.div_ceil(4) * 4;

            // The grid this chunk dispatches, and whether the Q4_K entry point
            // runs the split geometry over it — the encoder's one answer to both
            // ([`grouped_chunk_geometry`]).
            let (x_groups, y_groups, split) = grouped_chunk_geometry(scheme_id, m, n_chunk);

            let p = DequantGemmGgufParams {
                m: m as u32,
                k: k as u32,
                ncols: n_chunk as u32,
                x_vec4_off,
                w_byte_off: (w0 - w_base) as u32,
                split: split as u32,
                _p0: 0,
                _p1: 0,
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
                let mut pass = marks.begin(enc, "rlx-wgpu dequant_gemm_gguf pass");
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
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
/// Caller guarantees a scheme this ladder carries (`debug_assert!` below). This is
/// the `m == 1` form of [`run_dequant_matmul_gguf_gemv_rows`] — the entry the
/// executable calls — and delegates to it rather than encoding the row itself, so
/// the route's X-window guard covers both.
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
    marks: &mut PassMarks<'_>,
) {
    let handled = run_dequant_matmul_gguf_gemv_rows(
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
        marks,
    );
    debug_assert!(
        handled,
        "the caller guarantees a scheme this ladder carries"
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
    marks: &mut PassMarks<'_>,
) {
    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_size = ((x0 + (k * 4) as u64 - x_base).div_ceil(16) * 16)
        .min(x_buf.size().saturating_sub(x_local));

    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w_buf_size = w_buf.size();
    let w0_all = w_raw as u64;

    // Packed-weight bytes per output row ([`gguf_row_bytes`]). Output rows are
    // independent (each is a separate dot product over k), so when the full weight
    // window would exceed `max_storage_buffer_binding_size` (128 MiB minimum on
    // wgpu — hit by llvmpipe / min-spec adapters, or large models on any GPU) we
    // split the GEMV along N and bind only each chunk's rows.
    let row_bytes = gguf_row_bytes(scheme_id, k);

    let max_bind = gguf_max_bind(device);
    assert!(
        x_size <= max_bind,
        "rlx-wgpu gguf gemv: x window {x_size} > max_bind {max_bind}"
    );
    // Two arms, one route: the tiled arm gives each workgroup
    // `GEMV_TILED_COLS_PER_WG` output columns and splits each column's
    // contraction over its threads; the original gives one thread per column, 64
    // columns to a workgroup. Which one ran is recorded — see
    // [`gemv_one_row_shape`].
    let tiled = gemv_tiled_supports_scheme(scheme_id);
    // Output columns per chunk from the window budget; at least one, so a single
    // row wider than the whole budget still makes progress. The tiled arm also
    // spends workgroups on those columns, and a dispatch dimension is capped
    // (65535 on every backend, WebGPU's guaranteed floor), so an output wider
    // than the columns that cap covers is split too — the same arm, one more
    // chunk. Both halves of that rule live in `gemv_one_row_cols_per_chunk`.
    let cols_per_chunk = gemv_one_row_cols_per_chunk(row_bytes, max_bind, tiled);

    let out_bytes = ((n * 4).max(4) as u64).div_ceil(16) * 16;
    with_pooled_out_buf(device, out_bytes, |out_buf| {
        let dk = if tiled {
            dequant_gemv_gguf_tiled_kernel(device)
        } else {
            dequant_gemv_gguf_kernel(device)
        };
        let mut n0 = 0usize;
        while n0 < n {
            let n_chunk = (n - n0).min(cols_per_chunk);
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
                let mut pass = marks.begin(enc, "rlx-wgpu dequant_gemv_gguf pass");
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                if tiled {
                    // `GEMV_TILED_COLS_PER_WG` output columns per workgroup.
                    let workgroups = (n_chunk as u32).div_ceil(GEMV_TILED_COLS_PER_WG);
                    note_gemv_one_row_shape(GEMV_TILED_COLS_PER_WG, GEMV_TILED_THREADS, workgroups);
                    pass.dispatch_workgroups(workgroups, 1, 1);
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
    marks: &mut PassMarks<'_>,
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
            marks,
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
            marks,
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

#[cfg(test)]
mod tests {
    use super::{
        GEMV_TILED_COLS_PER_WG, GEMV_TILED_THREADS, GROUPED_ROWS, GROUPED_SPLIT_COLS,
        GROUPED_STAGE_ELEMS, GROUPED_THREADS, gguf_row_bytes, grouped_chunk_geometry,
        grouped_cols_per_chunk,
    };

    /// The row-grouped GEMM's geometry is written twice — `GROUPED_THREADS` /
    /// `GROUPED_ROWS` size the dispatch, `GROUPED_STAGE_ELEMS` the staged tile,
    /// `GROUPED_SPLIT_COLS` the split one — and the shader declares its own
    /// `@workgroup_size` plus its `TG` / `TM` / `BLK_ELEMS` / `COLS` consts, so
    /// read the shader here and tie the two. A lone edit of any one of them would
    /// otherwise describe a kernel that no longer runs: `TG` drives
    /// `col = wid.x * TG + tid` and the staging stride, `TM` the rows a pass over
    /// the weight covers, `BLK_ELEMS` the tile the barriers fall at, `COLS` the
    /// columns a split workgroup covers.
    #[test]
    fn grouped_geometry_matches_the_shader() {
        let wgsl = crate::kernels::DEQUANT_GEMM_GGUF_WGSL;
        let module = naga::front::wgsl::parse_str(wgsl).expect("wgsl parse");
        for entry in &module.entry_points {
            assert_eq!(
                entry.workgroup_size,
                [GROUPED_THREADS, 1, 1],
                "{}: workgroup must be GROUPED_THREADS wide",
                entry.name
            );
        }
        for want in [
            format!("const TG: u32 = {GROUPED_THREADS}u;"),
            format!("const TM: u32 = {GROUPED_ROWS}u;"),
            format!("const BLK_ELEMS: u32 = {GROUPED_STAGE_ELEMS}u;"),
            format!("const COLS: u32 = {GROUPED_SPLIT_COLS}u;"),
        ] {
            assert!(wgsl.contains(&want), "expected `{want}` in the shader");
        }
        // The split entry point's indexing reads exactly two k-lane groups out of
        // a workgroup (`sl = tid / COLS` into two halves whose slices are
        // `sl * 128` and `sl * 64` of a block), so `COLS` being half of `TG` is
        // what makes `sl` the set {0, 1}. Moving either const on its own — and both
        // spellings with it — would let `sl` reach 2 and read a column's slice out
        // of another column's half, silently, which no other assertion here sees.
        assert_eq!(
            GROUPED_SPLIT_COLS * 2,
            GROUPED_THREADS,
            "the split geometry is two k-lane groups of a workgroup, so COLS must be TG / 2"
        );
        // The split's fold writes its partials into the staged tile's own first
        // `COLS * (TM + 1)` slots (`xs`'s first `TM * (BLK_ELEMS / 4)`) rather than
        // into an array of its own, so the tile has to be at least that big: an `xs`
        // index past its length is clamped by WGSL, which would drop a column's
        // partial sum silently instead of failing, and the assert above makes a
        // coordinated move of COLS and TG the natural way to reach that.
        let xs_slots = GROUPED_ROWS * (GROUPED_STAGE_ELEMS as u32 / 4);
        assert!(
            GROUPED_SPLIT_COLS * (GROUPED_ROWS + 1) <= xs_slots,
            "the split's fold needs COLS * (TM + 1) = {} of the staged tile's {xs_slots} slots",
            GROUPED_SPLIT_COLS * (GROUPED_ROWS + 1)
        );
    }

    /// Which shapes the Q4_K entry point's split geometry reaches: the rule is the
    /// product of the base grid's two dimensions against
    /// [`GROUPED_SPLIT_MAX_BASE_WORKGROUPS`], so a narrow `n` reaches it exactly as
    /// a small `m` does, and the row counts a model body's forms have reach it at
    /// very different widths — n <= 1024 at 13 rows, n <= 640 at 24, n <= 128 at
    /// 100. Those verdicts are pinned here; a change to the ceiling, to `TG` or to
    /// `TM` has to move them with it.
    #[test]
    fn grouped_split_reach_at_the_body_shapes() {
        // (scheme_id, m, n, split): 0 Q4_K, 2 Q6_K, 20 Q8_0.
        for &(scheme_id, m, n, split) in &[
            // The 13-row body at the two widths its forms are chunked at: 16 x 2 =
            // 32 workgroups at n = 1024, 96 x 2 = 192 at n = 6144.
            (0u32, 13usize, 1024usize, true),
            (0, 13, 1920, false),
            (0, 13, 6144, false),
            // The 24-row body: 10 x 3 = 30 at n = 640, 16 x 3 = 48 at n = 1024.
            (0, 24, 640, true),
            (0, 24, 1024, false),
            // Wider bodies are reached by a narrow chunk alone: 10 x 6 = 60 at 48
            // rows / 640, 2 x 13 = 26 at 100 rows / 128, 1 x 13 = 13 at 100 rows /
            // 64, 4 x 13 = 52 at 100 rows / 256, 32 x 1 = 32 at 8 rows / 2048. The
            // chunk widths the const's doc names, and the far side of each of their
            // boundaries.
            (0, 48, 640, false),
            (0, 48, 320, true),
            (0, 48, 384, false),
            (0, 8, 2048, true),
            (0, 8, 2112, false),
            (0, 100, 64, true),
            (0, 100, 128, true),
            (0, 100, 256, false),
            (0, 100, 1024, false),
            (0, 104, 1024, false),
            (0, 104, 128, true),
            // Q6_K and Q8_0 have no split arm: their entry points run the base
            // geometry whatever this says.
            (2, 13, 1024, false),
            (20, 13, 1024, false),
        ] {
            let (x, y, split_taken) = grouped_chunk_geometry(scheme_id, m, n);
            assert_eq!(
                split_taken, split,
                "scheme {scheme_id} m={m} n={n} dispatched as {x} x {y} workgroups"
            );
        }
        // Two grids written out rather than re-derived from the helper: under the
        // split a workgroup covers `GROUPED_SPLIT_COLS` columns, so its x is twice
        // the base geometry's, or one workgroup short of twice where the chunk's
        // last thread group is half-full or less — n = 65 is 2 column groups base
        // and 3 split, not 4.
        assert_eq!(grouped_chunk_geometry(0, 13, 1024), (32, 2, true));
        assert_eq!(grouped_chunk_geometry(0, 13, 6144), (96, 2, false));
        assert_eq!(grouped_chunk_geometry(0, 8, 65), (3, 1, true));
        // The mixed call the ceiling const's doc records, derived here through the
        // encoder's own rule rather than restated: the 1819 columns one dispatch of
        // a Q4_K weight at k = 1024 covers under a 1 MiB binding cap
        // (`grouped_cols_per_chunk` at that budget against this scheme's 576-byte
        // row stride), then the verdicts for the 13-row, n = 6144 form's three
        // chunks of it and its trailing 687-column chunk.
        let chunk = grouped_cols_per_chunk(gguf_row_bytes(0, 1024), 1024 * 1024, 13);
        assert_eq!(chunk, 1819, "the 1 MiB cap's chunk width at k = 1024");
        assert_eq!(grouped_chunk_geometry(0, 13, chunk), (29, 2, false));
        assert_eq!(
            grouped_chunk_geometry(0, 13, 6144 - 3 * chunk),
            (22, 2, true)
        );
        // The split arm's committed numeric coverage, named because lowering this
        // ceiling is what would take a cell out of it: `gguf_dequant_matmul_prefill_parity.rs`'s
        // Gemma cell (m = 5, n = 1536, k = 12288 — 48 blocks of 256) is the one
        // prefill-parity cell that runs the split across many blocks, its
        // m = 5 / n = 100 cell covers the narrowest partial column tile (4 real
        // columns of its 32; the m = 4 / n = 8 cell above already covers 8 of 32),
        // its m = 104 / n = 128 cell is the one that folds 13 row groups, and
        // `dequant_plan_relaxed.rs`'s two projections cover both block-count cases
        // (n = 256 over k = 512 is the two-block one; the n = 512 projection
        // contracts over k = 256, a single block). The `48` asserted below is that
        // first cell's x grid — 1536 / COLS — not its block count.
        assert_eq!(grouped_chunk_geometry(0, 5, 1536), (48, 1, true));
        assert_eq!(grouped_chunk_geometry(0, 5, 100), (4, 1, true));
        assert_eq!(grouped_chunk_geometry(0, 104, 128), (4, 13, true));
        assert_eq!(grouped_chunk_geometry(0, 8, 256), (8, 1, true));
        assert_eq!(grouped_chunk_geometry(0, 8, 512), (16, 1, true));
    }

    /// The tiled one-row arm's geometry is written twice the same way: the
    /// encoder sizes its dispatch from `GEMV_TILED_THREADS` /
    /// `GEMV_TILED_COLS_PER_WG`, and the shader declares its own
    /// `@workgroup_size` plus the two constants that size `red`, span the entry's
    /// columns and place the second column's store. Read the shader here and tie
    /// the two, so this arm is not invisible to the fast, device-free gate: the
    /// reduction array's declaration is one slot per lane per output column
    /// (`LANES * COLS_PER_WG`) and is pinned here rather than restated in
    /// `tests/gguf_kernel_scheme_coverage.rs`; the entry's column span and its
    /// second column's slot base are those constants, and the second column is
    /// stored one column over — a workgroup widened past what the Rust constants
    /// declare would otherwise accumulate into slots the array does not have.
    ///
    /// Half of this reads the shader as text, so a reflow of those hot-loop lines
    /// fails it without any geometry having moved: that is the accepted cost of
    /// tying the two spellings at all (the grouped route's pin does the same), and
    /// the fix is to re-spell the line, not to read the failure as geometry drift.
    #[test]
    fn tiled_gemv_geometry_matches_the_shader() {
        let wgsl = crate::kernels::DEQUANT_GEMV_GGUF_WGSL;
        let module = naga::front::wgsl::parse_str(wgsl).expect("wgsl parse");
        let tiled = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "dequant_gemv_tiled")
            .expect("the tiled one-row entry point");
        assert_eq!(
            tiled.workgroup_size,
            [GEMV_TILED_THREADS, 1, 1],
            "dequant_gemv_tiled: workgroup must be GEMV_TILED_THREADS wide"
        );
        for want in [
            format!("const LANES: u32 = {GEMV_TILED_THREADS}u;"),
            format!("const COLS_PER_WG: u32 = {GEMV_TILED_COLS_PER_WG}u;"),
            "var<workgroup> red: array<f32, LANES * COLS_PER_WG>;".to_string(),
            // The entry's column span, and the second column's reduction slots and
            // store — the places that column's geometry is spelled out.
            "let j0 = wid.x * COLS_PER_WG;".to_string(),
            "red[LANES + lane] = acc1;".to_string(),
            "outarr[params.out_f32_off + j0 + 1u]".to_string(),
        ] {
            assert!(wgsl.contains(&want), "expected `{want}` in the shader");
        }
    }
}
