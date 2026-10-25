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
//! is its own whole 32-element block) — or, on a device whose subgroup width the
//! probe reads, the subgroup arm (`dequant_gemv_wide`: one subgroup per output
//! column, the lane partials combined by `subgroupAdd`), which is the default:
//! `RLX_WGPU_GEMV_WIDE=0`, or `=shared`, restores the tiled arm, and the shader
//! header carries what the two measured; Q1_0 keeps the original one-thread-per-column
//! arm on the scalar X view. For `m > 1` Q4_K / Q6_K / Q8_0 use the row-grouped GEMM
//! (`encode_dequant_matmul_gguf_gemm_grouped`): one pass over each packed
//! weight tile per group of 8 rows, instead of the per-row GEMV's one full pass
//! over the weight per row. Its `k` gate is the scheme's own packing block
//! ([`grouped_k_ok`]): the K-quants' block is the staged tile itself, while Q8_0's
//! 32-element one lets a row end inside a tile — the `dequant_gemm_q8_0` entry
//! point's staged-tile loop stops at the last whole block of such a row and stages
//! the tile through a fetch bounded by the row's own length (`stage_x_bounded`), so
//! a contraction of several tiles plus a block, or a sub-tile one, takes this route
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

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rlx_ir::{Graph, Op};

use crate::backend::PassMarks;
use crate::buffer::Arena;
use crate::gguf_host::scheme_from_id;
use crate::kernels::{
    CopyFromSlabParams, DequantGemmGgufParams, DequantGemmQ10Params, DequantGemvGgufParams,
    DequantGgufParams, Kernel, MatmulParams, copy_from_slab_kernel, copy_from_slab_vec4_kernel,
    dequant_gemm_q1_0_kernel, dequant_gemm_q4k_kernel, dequant_gemm_q6k_kernel,
    dequant_gemm_q8_0_kernel, dequant_gemv_gguf_kernel, dequant_gemv_gguf_tiled_kernel,
    dequant_gemv_gguf_wide_ilp_kernel, dequant_gemv_gguf_wide_kernel, dequant_gguf_kernel,
    gguf_subgroup_probe_kernel, matmul_bt_kernel,
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

/// `RLX_WGPU_GEMV_WIDE`: admit the subgroup arm (`dequant_gemv_wide`) for a
/// one-row call the tiled arm would take.
///
/// `0`, `false`, `off`, `no` and `shared` keep the shared-memory arm —
/// `shared` is the spelling that says *why*, since the arm it keeps is
/// `dequant_gemv_tiled`'s workgroup-memory reduction. Anything else admits the
/// wide arm, subject to the probe below. Read at the point of decision rather
/// than frozen into a snapshot: the one-row encoder asks once per dispatch, and
/// a test that flips the knob between two calls must see it move.
fn gemv_wide_enabled() -> bool {
    // `flag_or` carries the crate's truthiness (`0`/`false`/`off`/`no`/empty off),
    // so only the arm's own spelling of *which* arm to keep is added here.
    match rlx_ir::env::var("RLX_WGPU_GEMV_WIDE") {
        Some(v) if v.trim().eq_ignore_ascii_case("shared") => false,
        _ => rlx_ir::env::flag_or("RLX_WGPU_GEMV_WIDE", GEMV_WIDE_DEFAULT),
    }
}

/// The adapter's real `(subgroup_size, num_subgroups)`, probed once per process
/// and cached.
///
/// `AdapterInfo` only reports bucketed sizes (`subgroup_min_size`/
/// `subgroup_max_size`; Metal says 4..64), and the wide one-row arm maps output
/// columns onto whole subgroups, so the number itself is needed: it is the
/// divisor that says how many columns a 128-invocation workgroup covers and the
/// stride a lane walks a column's units by. The probe is one invocation of
/// `GGUF_SUBGROUP_PROBE_WGSL`, dispatched and mapped back on the spot; see
/// `src/coop_probe.rs` for why this crate asks the device rather than trusting a
/// compile-time constant.
///
/// `None` when the device does not carry [`wgpu::Features::SUBGROUP`] (the probe
/// shader names subgroup built-ins, so it is never created there). A probed read
/// that does not come back is refused on stderr rather than left as a silent
/// `None`: the arm is then never taken for the life of the process, and an
/// undiagnosed refusal would read as "this device has no subgroups".
/// The answer is the process-global device's: the probe caches into one `OnceLock`
/// and submits on the process-global queue, so a second adapter in the same
/// process would read the first one's width. This crate has one device per process
/// (`device.rs`), which is why the signature takes the device at all.
pub fn gemv_subgroup_probe(device: &wgpu::Device) -> Option<(u32, u32)> {
    if !device.features().contains(wgpu::Features::SUBGROUP) {
        return None;
    }
    *GEMV_SUBGROUP.get_or_init(|| match run_subgroup_probe(device) {
        Some(probed) => Some(probed),
        None => {
            eprintln!(
                "rlx-wgpu: the subgroup probe on {:?} did not complete — the wide one-row \
                 arm stays off and the shared-memory arm runs",
                crate::device::adapter_name()
            );
            None
        }
    })
}

/// [`gemv_subgroup_probe`]'s width alone, when it is one the wide arm can use: a
/// width that does not divide the arm's 128 threads would leave a workgroup's
/// last subgroup spanning past the group, so such a width refuses the arm and
/// the shared-memory entry runs instead.
pub fn gemv_subgroup_width(device: &wgpu::Device) -> Option<u32> {
    let (size, _count) = gemv_subgroup_probe(device)?;
    (size > 0 && GEMV_WIDE_THREADS.is_multiple_of(size)).then_some(size)
}

/// Read the adapter's subgroup width now rather than on the first one-row
/// admission: the probe is a dispatch and a fence of its own, and compiling a
/// graph that carries any packed multiply is outside every run's submission
/// bracket. The predicate is deliberately loose (any `Op::DequantMatMul`, before
/// this backend's own unfuse), so on a graph whose multiplies all end up on the
/// grouped arm this probes a width no call will use — one dispatch, once per
/// process, in exchange for no probe inside a measured run.
pub(crate) fn prime_gemv_subgroup_probe(device: &wgpu::Device) {
    if gemv_wide_enabled() {
        let _ = gemv_subgroup_probe(device);
    }
}

/// Output columns one workgroup of the wide arm covers, when this device admits
/// it: `GEMV_WIDE_THREADS / subgroup_size`, one whole subgroup per column.
pub fn gemv_wide_cols_per_wg(device: &wgpu::Device) -> Option<u32> {
    if !gemv_wide_enabled() {
        return None;
    }
    gemv_subgroup_width(device).map(|sg| GEMV_WIDE_THREADS / sg)
}

static GEMV_SUBGROUP: OnceLock<Option<(u32, u32)>> = OnceLock::new();

/// [`gemv_subgroup_probe`]'s one dispatch: run the probe, copy its two words into
/// a map-readable buffer, wait once, and read them.
fn run_subgroup_probe(device: &wgpu::Device) -> Option<(u32, u32)> {
    let queue = crate::device::device_queue()?;
    let kernel = gguf_subgroup_probe_kernel(device);
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gguf subgroup probe out"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // MAP_READ cannot be combined with COPY_SRC, so the value lands here through
    // one copy rather than being mapped in place.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gguf subgroup probe readback"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gguf subgroup probe bg"),
        layout: &kernel.bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out.as_entire_binding(),
        }],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gguf subgroup probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gguf subgroup probe"),
            ..Default::default()
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &readback, 0, 8);
    // Counted like every other one-shot probe's submission and fence
    // (`coop_probe`'s): this one fires on the first admission decision inside a run
    // that may bracket its submission counts, so an uncounted dispatch here would
    // read one low.
    crate::backend::note_device_submit();
    queue.submit([enc.finish()]);
    // One fence covers both the copy and the map: `map_async` queues the read and
    // the mapping completes under the same wait, so a poll before it would be a
    // second fence for one read.
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    crate::backend::note_device_wait();
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let data = readback.slice(..).get_mapped_range().ok()?;
    let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"));
    let probed = (word(0), word(4));
    drop(data);
    readback.unmap();
    Some(probed)
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
/// Both one-row arms read X through the buffer's `vec4` view: the tiled arm as 8
/// vector loads per 32-element unit, the legacy Q1_0 arm component-wise
/// (`xarr4[i >> 2u][i & 3u]`). So the window's own offset must line up with a
/// four-element group, and for the tiled arm so must the input row stride
/// (`x_row_bytes`, the row's `k` elements — not `gguf_row_bytes`, the packed
/// weight's stride). True at the shapes these schemes are used at — the arena
/// aligns every tensor to 16 bytes and `k` is a multiple of 4 wherever an arm is
/// admitted at all (the K-quants' 256, Q8_0's 32, Q1_0's 128) — and false for a
/// view the route then hands to the host rather than reading X shifted. The
/// grouped GEMM checks the same two things for its own `vec4` X reads, and
/// [`select_path`] composes this one guard for the multi-row admission as well.
/// Pinned device-free by `tests/gguf_kernel_scheme_coverage.rs`.
pub fn gemv_one_row_device_ok(scheme_id: u32, x_byte_off: usize, x_row_bytes: usize) -> bool {
    x_byte_off.is_multiple_of(16)
        && (!gemv_tiled_supports_scheme(scheme_id) || x_row_bytes.is_multiple_of(16))
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
    // the tiled arm's own schemes' single row (`one_row_arm_ok`).
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
/// arm splits a column's contraction over `GEMV_TILED_THREADS` lanes, one lane per
/// 32-element unit of the row (a Q4_K sub-block, or half a Q6_K `ql` span read at
/// both of its nibble sets — see the shader header); the legacy arm gives each of
/// its 64 threads one output column.
///
/// The shader's declared `@workgroup_size` and the shape the route records must
/// both be these numbers: the device-free `tiled_gemv_geometry_matches_the_shader`
/// below derives the tiled arm's `@workgroup_size` (and its `LANES`/`COLS_PER_WG`)
/// from these constants, `tests/gguf_kernel_scheme_coverage.rs` pins the legacy
/// arm's size and the portable floor, and `tests/gguf_matmul_route.rs` pins the
/// recorded shape. `tests/gguf_one_row_stream_rate.rs` carries what moving the
/// tiled arm's two constants does to the rate.
const GEMV_TILED_THREADS: u32 = 32;
const GEMV_LEGACY_THREADS: u32 = 64;

/// Threads one workgroup of the one-row **subgroup** arm runs: the shader's
/// `dequant_gemv_wide` `@workgroup_size` and its `WIDE_THREADS`. The workgroup
/// covers `GEMV_WIDE_THREADS / subgroup_width` output columns — one whole
/// subgroup each — so at this machine's probed width of 32 it covers four, twice
/// the tiled arm's two. `tests/gguf_one_row_stream_rate.rs` carries what the arm
/// reads at the model's geometries against the tiled arm's figure.
const GEMV_WIDE_THREADS: u32 = 128;

/// Output columns one **subgroup** of the ILP form takes: the shader's
/// `WIDE_ILP`, and the multiplier of the columns a workgroup covers. One column
/// per lane left the GEMV latency-bound (39 GB/s at Q8_0 `k = 1024` against a
/// 130 GB/s probe), so the form holds several columns' independent accumulators
/// per subgroup with one X read shared across them.
///
/// Two rather than four, measured on both real frame graphs: halving the columns
/// per subgroup doubles the workgroups a chunk is cut into, and the body frame's
/// `n = 1 024` blocks issue only 64 of them at four. No switch moves this
/// constant, so a pair of widths is rebuild-to-rebuild and the figures in
/// `MEASUREMENTS.md` are not re-takeable from the shipped binary. Public with
/// [`gemv_wide_cols_per_wg`] so the parity and measurement cells derive the arm's
/// width instead of restating it.
///
/// Re-taken on the code predictor's own widths: the variants were built from this
/// one shader source with `WIDE_ILP` and `WIDE_THREADS` substituted at
/// construction, and four columns a subgroup reads ahead on the harness's
/// `k = 1 024` forms while the frame's own shapes keep two — the same cause this
/// constant was set on, since four columns a subgroup halves the workgroups a
/// chunk is cut into and a frame's smaller dispatches cannot absorb that where a
/// 32-step synthetic call can. The width stays two; a change of width is
/// rebuild-to-rebuild.
pub const GEMV_WIDE_ILP: u32 = 2;

/// Columns a **chunk** needs before the ILP entry is worth its longer per-thread
/// walk. The floor was set on the four-column form the entry first shipped with:
/// paired on `tests/gguf_one_row_stream_rate.rs`'s forms, rebuilt with the floor at
/// 4096 and at 1024, the predictor's Q4_K `k = 2048, n = 2048` block read
/// 63.4 → 64.7 GB/s, the Q8_0 `n = 1024` head 52.1 → 54.3 and its `n = 2048`
/// 69.5 → 70.9 — the ILP form's one input read shared across its column group,
/// against one each. One form paid for it: Q6_K `k = 6144, n = 2048` read
/// 79.9 → 74.8 GB/s, its per-column walk already long enough that a quarter of
/// the threads starve the device.
///
/// The shipped entry is two columns a subgroup rather than the four this floor
/// was set on, and the floor's boundary width is where that entry is weakest: on
/// the harness's `Q8_0 k = 1024, n = 1 024` form the one-column entry reads ahead,
/// so the floor keeps the ILP entry on a width where the narrow one is ahead of
/// it. What it buys is further up: against the four-column form two columns is
/// behind at 1 024 and ahead from 2 048 up, which is where the predictor's every
/// width sits. The body's are not all up there — 74 MB of its 851 MB a frame are
/// `k = 2 048, n = 1 024` and land on the boundary, where the two entries are
/// closest and the floor decides against the narrow one for the reason above,
/// except on a narrow Q6_K row, which [`GEMV_ILP_Q6K_MIN_COLS`] carries. No
/// switch moves this constant, so any pair of floors is rebuild-to-rebuild;
/// `MEASUREMENTS.md` carries the per-form table. Public with [`GEMV_WIDE_ILP`] so
/// the cells state the floor they assert.
pub const GEMV_ILP_MIN_COLS: usize = 1024;

/// Columns up to which a **narrow** Q6_K chunk keeps the one-column-per-subgroup
/// entry — a row whose `k / 32` is two 32-element units a lane, the shape the
/// body's `k = 2 048, n = 1 024` attention projections have. A Q6_K block is
/// 210 bytes of four quarters, so on that row the pair entry's second column costs
/// more than its one shared input read returns; from this width up it returns more.
/// Twice [`GEMV_ILP_MIN_COLS`], which is where that row's pair entry pulls ahead.
pub const GEMV_ILP_Q6K_MIN_COLS: usize = 2048;

/// Default of `RLX_WGPU_GEMV_WIDE`: the subgroup arm.
///
/// Paired on the synthesis cell it puts the generation loop 2.3 % ahead of the
/// shared-memory arm (20 020.8 against 20 502.1 ms, the predictor 38.98 against
/// 40.39 ms a level), and the session in the shader's header table reads it ahead
/// on every `Q8_0` form this model's levels issue; `RLX_WGPU_GEMV_WIDE=shared`
/// restores the workgroup-memory arm.
const GEMV_WIDE_DEFAULT: bool = true;

/// Output columns one workgroup of the tiled arm covers. Two, so a workgroup
/// covers two output columns and splits each one's contraction over its threads,
/// reusing one X read for both; the dispatch is then `ceil(columns / this)`
/// workgroups wide, which is the form both the dispatch cap in
/// [`gemv_one_row_cols_per_chunk`] and the recorded shape below are written in.
const GEMV_TILED_COLS_PER_WG: u32 = 2;

/// Most workgroups one dispatch may have along a dimension — the WebGPU floor
/// (`max_compute_workgroups_per_dimension`), which every backend honours. Every
/// one-row arm is chunked to it at its own width: `GEMV_TILED_COLS_PER_WG` output
/// columns per workgroup on the tiled arm, `GEMV_WIDE_THREADS / subgroup_width`
/// on the subgroup arm, and `GEMV_LEGACY_THREADS` on the one-thread-per-column
/// arm — so the column cap is this times whatever width the arm records.
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
    /// Output columns one workgroup serves — the width of the entry that ran.
    /// `GEMV_TILED_COLS_PER_WG` (two) on the tiled arm; on the subgroup arm
    /// `GEMV_WIDE_THREADS / subgroup_width` (four at this machine's width of 32)
    /// for the one-column entry and `GEMV_WIDE_ILP` times that for the ILP entry,
    /// which serves the chunks at or above [`GEMV_ILP_MIN_COLS`]; `64` while the
    /// arm is one thread per output column.
    pub cols_per_workgroup: u32,
    /// Threads one workgroup runs. On the tiled and subgroup arms they split each
    /// of those columns' contractions; on the legacy arm each thread serves its
    /// own column.
    pub threads_per_workgroup: u32,
    /// Workgroups in the recorded dispatch — the last chunk's, when the call was
    /// split.
    pub workgroups: u32,
    /// Which entry of the one-row arm ran, as the id the encoder's arm table
    /// assigns it: `1` the ILP entry (`dequant_gemv_wide_ilp`), `2` the
    /// one-column-per-subgroup entry (`dequant_gemv_wide`), `3` the tiled arm,
    /// `4` the legacy one-thread-per-column arm. `1` and `2` are the subgroup
    /// arm's: both need a usable probed width. `0` is never recorded.
    pub arm: u32,
}

static ONE_ROW_COLS_PER_WG: AtomicU64 = AtomicU64::new(0);
static ONE_ROW_THREADS: AtomicU64 = AtomicU64::new(0);
static ONE_ROW_WORKGROUPS: AtomicU64 = AtomicU64::new(0);
static ONE_ROW_ARM: AtomicU64 = AtomicU64::new(0);

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
        arm: ONE_ROW_ARM.load(Ordering::Relaxed) as u32,
    })
}

/// Forget [`gemv_one_row_shape`] (tests bracketing a single call).
pub fn reset_gemv_one_row_shape() {
    ONE_ROW_COLS_PER_WG.store(0, Ordering::Relaxed);
    ONE_ROW_THREADS.store(0, Ordering::Relaxed);
    ONE_ROW_WORKGROUPS.store(0, Ordering::Relaxed);
    ONE_ROW_ARM.store(0, Ordering::Relaxed);
}

/// Record what one one-row dispatch was shaped like. `cols_per_workgroup` is how
/// many output columns a workgroup covers, so on the tiled and subgroup arms
/// `workgroups * cols_per_workgroup` is the output width the dispatch covered
/// rounded up to a whole workgroup — a *ceiling*, not an equality: a workgroup
/// whose last column is off the end of the chunk computes nothing for it (that
/// column's accumulator — or its whole subgroup — stays zero) and skips that
/// store, so no row or column is ever discarded. (The legacy arm's last
/// workgroup rounds up to 64 columns the same way.) `arm` names which entry ran —
/// the encoder's own id for it, so the record and the setup cache's key cannot
/// disagree about which reduction is being described.
fn note_gemv_one_row_shape(
    cols_per_workgroup: u32,
    threads_per_workgroup: u32,
    workgroups: u32,
    arm: u32,
) {
    ONE_ROW_COLS_PER_WG.store(cols_per_workgroup as u64, Ordering::Relaxed);
    ONE_ROW_THREADS.store(threads_per_workgroup as u64, Ordering::Relaxed);
    ONE_ROW_WORKGROUPS.store(workgroups as u64, Ordering::Relaxed);
    ONE_ROW_ARM.store(u64::from(arm), Ordering::Relaxed);
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
/// handed and by `encode_dequant_matmul_gguf_gemm_grouped` of the one it is about
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
/// rows this allows — take this route instead of the slab. The tile's surplus, what
/// lies past `k`, is neither fetched nor read: that entry stages it through the
/// row-bounded `stage_x_bounded` and its block loop stops at the row's last whole
/// block, so this route does not depend on what a read outside the window it binds
/// would return.
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
/// Safe across dispatches in one encoder because each multiply is followed by
/// the slab → arena move (`encode_out_slab_copy`) before the next write, and
/// dispatches execute in recording order.
/// Pooled output buffer and its capacity in bytes.
struct OutPool(wgpu::Buffer, u64);

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

/// Record one `out_buf → arena` move into `pass`.
///
/// The output slab and the arena are different buffers, so this cannot be the
/// arena-only `copy.wgsl` move, and — because it must ride the pass a packed
/// multiply was joined into — it cannot be a `copy_buffer_to_buffer` blit
/// either: a blit is not a compute dispatch, so recording one ends the pass.
/// It is a storage-to-storage dispatch instead, bindings 0 = the arena window
/// (read-write), 1 = params, 2 = the slab window (read-only); the usage scopes
/// are per dispatch, so the same slab being read here and written by the
/// multiply's own dispatch before it is not a conflict.
///
/// Both windows start at a 256 B-aligned base and carry their intra-window
/// offset in the uniform, the same shape every other windowed GGUF kernel
/// uses. `elems` is the f32 count; the vector form is taken only when the span
/// is whole vectors and both offsets are 16 B aligned.
///
/// `pass` executes its dispatches in recording order, so a later multiply that
/// reuses the pooled slab cannot overwrite it before this copy has read it.
fn encode_out_slab_copy(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::ComputePass<'_>,
    out_buf: &wgpu::Buffer,
    src_byte_off: u64,
    dst_byte_off: usize,
    elems: u32,
) {
    if elems == 0 {
        return;
    }
    let dst0 = dst_byte_off as u64;
    let dst_base = (dst0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (dst_buf, dst_local) = arena.resolve_act(dst_base as usize);
    let dst_need = dst0 + elems as u64 * 4 - dst_base;
    let dst_size =
        (dst_need.div_ceil(16) * 16).min(dst_buf.size().saturating_sub(dst_local as u64));
    // The old blit this replaces failed validation when the destination slot
    // straddled a shard; keep that refusal explicit rather than binding a
    // short window and writing half the output. A truncated window is any case
    // where the buffer ends inside the move — a shard boundary is the common one.
    assert!(
        dst_size >= dst_need,
        "rlx-wgpu: packed multiply's output copy runs past the end of its buffer \
         (out_byte_off {dst_byte_off}, {elems} f32, window {dst_size} of {dst_need} B)"
    );

    let src_base = (src_byte_off / STORAGE_ALIGN) * STORAGE_ALIGN;
    let src_need = src_byte_off + elems as u64 * 4 - src_base;
    let src_size = src_need.div_ceil(16) * 16;

    let in_off = ((src_byte_off - src_base) / 4) as u32;
    let out_off = ((dst0 - dst_base) / 4) as u32;
    let vec4 = crate::backend::vec4_move_ok(elems, in_off, out_off);
    let p = CopyFromSlabParams {
        n: elems,
        in_off,
        out_off,
        _p0: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu copy_from_slab uniform"),
        size: std::mem::size_of::<CopyFromSlabParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let ck = if vec4 {
        copy_from_slab_vec4_kernel(device)
    } else {
        copy_from_slab_kernel(device)
    };
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu copy_from_slab bg"),
        layout: &ck.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: dst_buf,
                    offset: dst_local as u64,
                    size: wgpu::BufferSize::new(dst_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: out_buf,
                    offset: src_base,
                    size: wgpu::BufferSize::new(src_size),
                }),
            },
        ],
    });
    // Vectors are one 16 B lane each; scalars one f32 each. A span wider than
    // the grid cap splits over `gid.y` (the shaders' own linearization), which
    // keeps a full-vocabulary head one dispatch.
    let items = if vec4 { elems.div_ceil(4) } else { elems };
    let (gx, gy, gz) = crate::backend::dispatch_dims(items, crate::backend::MOVE_WORKGROUP);
    pass.set_pipeline(&ck.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    if vec4 {
        crate::backend::note_vec4_move();
    }
    pass.dispatch_workgroups(gx, gy, gz);
}

/// Encode the fused GGUF DequantMatMul ladder into `pass`, the caller's already
/// open compute pass. Scheme and shape pick the route: the Q1_0 and row-grouped
/// GEMMs first, then the per-row GEMV loop — through which a multi-row call the
/// grouped GEMM's own gates decline is read ([`DequantMatmulPath::GroupedDevice`]
/// lists them), the same route an unjoined call of that shape takes. For a
/// multi-row Q8_0 call, whose per-row loop carries its one row alone, those same
/// declines are the *refusal* below instead, its bisect lever
/// `RLX_WGPU_Q8_0_GEMM_DISABLE` included. Uniforms are unique per dispatch
/// (`queue.write_buffer` is not ordered inside an encoder the way a dispatch is).
///
/// This is the entry the run's batch join uses: every dispatch the call issues is
/// recorded into the pass the run already has open, so a packed multiply no longer
/// ends that pass. Where an arm still lands its output through a pooled slab, that
/// move is a dispatch (`encode_out_slab_copy`) rather than a
/// `copy_buffer_to_buffer` blit; the one-row arm stores straight into the arena
/// and issues no move at all. Dispatches in one pass execute in recording
/// order, which is what makes a shared pooled slab safe: a later multiply can
/// only overwrite it after this call's copy has read it.
///
/// The caller's pass is the one the run report already marked, so no
/// [`PassMarks`] handle is taken here.
///
/// Returns whether this ladder has an arm for the call: `false` when it does not —
/// a scheme outside [`gemv_supports_scheme`] above one row, whose row-grouped arm
/// declined, for which the per-row loop has no branch (its leg dispatch would match
/// nothing and read the weights as zeros). Nothing is encoded in that case — the
/// refusals are all decided before the first dispatch — so the caller can end its
/// pass and hand the call to the fallback, [`run_dequant_matmul_gguf_gpu`] for such
/// a scheme. This answer cannot be discarded, by either caller: the run's in-pass
/// join reads it from its own call here, and [`run_dequant_matmul_gguf_gemv_rows`]
/// from the encoder wrapper's.
///
/// The per-row loop's tiled arm reads X as `vec4`, so X's window must be
/// four-element addressable. [`run_dequant_matmul_gguf_gemv_rows`] checks that
/// with [`gemv_one_row_device_ok`] for the routes it owns, and the run's join
/// decision asks [`select_path`] — the same ladder, whose device-only answers
/// compose that guard — before calling here directly. This gate carries the
/// 8-bit scheme's own block term (`one_row_arm_ok`), so a direct caller has to
/// make sure of the window alone.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_matmul_gguf_into_pass(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::ComputePass<'_>,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
    cache: Option<&GemvSetupCache>,
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
            pass,
            m,
            k,
            n,
            x_byte_off,
            w_byte_off,
            out_byte_off,
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
            pass,
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
            pass,
            k,
            n,
            scheme_id,
            x_byte_off + row * k * 4,
            w_byte_off,
            out_byte_off + row * n * 4,
            cache,
        );
    }
    note_route(DequantMatmulRoute::PerRow);
    true
}

/// Encode the fused GGUF DequantMatMul into `enc` (no `queue.submit`), leaving the
/// submission to the caller. The whole ladder goes into one pass this call opens
/// through `marks`, so a caller outside the run path — [`run_dequant_matmul_gguf_gemv_rows`]
/// and the crate's tests — keeps the shape it had. The run's own joined calls use
/// [`encode_dequant_matmul_gguf_into_pass`] instead, recording into the batch pass
/// already open; the route ladder and its answer are the same function's.
///
/// `marks` is the run's pass-marks handle: the one pass this wrapper opens carries
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
    let mut pass = marks.begin(enc, "rlx-wgpu dequant_matmul_gguf pass");
    encode_dequant_matmul_gguf_into_pass(
        arena,
        device,
        queue,
        &mut pass,
        m,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
        None,
    )
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
/// An X window either one-row arm's `vec4` read cannot address sends the whole
/// call to [`crate::gguf_host`]. Every caller of the tiled arm has to hold that
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
    pass: &mut wgpu::ComputePass<'_>,
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
            pass.set_pipeline(&dk.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            const TM: u32 = 8;
            pass.dispatch_workgroups((n as u32).div_ceil(64), (m as u32).div_ceil(TM), 1);
        }

        let (dst0, dst0_off) = arena.resolve_act(out_byte_off);
        let row_bytes = (n * 4) as u64;
        let contiguous = (dst0_off as u64) + out_bytes <= dst0.size();
        if contiguous {
            encode_out_slab_copy(
                arena,
                device,
                queue,
                pass,
                out_buf,
                0,
                out_byte_off,
                (m * n) as u32,
            );
        } else {
            for r in 0..m {
                encode_out_slab_copy(
                    arena,
                    device,
                    queue,
                    pass,
                    out_buf,
                    (r as u64) * row_bytes,
                    out_byte_off + r * n * 4,
                    n as u32,
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
/// workgroups and so reaches `GEMV_MAX_WORKGROUPS * cols_per_wg` columns.
/// `cols_per_wg` is the arm's own width — `GEMV_TILED_COLS_PER_WG` on the tiled
/// arm, `GEMV_WIDE_THREADS / subgroup_width` on the subgroup arm, and
/// `GEMV_LEGACY_THREADS` (one thread per column) on the arm that keeps the
/// original shape.
///
/// The encoder chunks its N loop by this and
/// [`gemv_one_row_dispatches`] reports it, so the rule has one spelling.
fn gemv_one_row_cols_per_chunk(row_bytes: u64, max_bind: u64, cols_per_wg: usize) -> usize {
    // The `.max(1)` is what makes a row wider than the whole budget one column per
    // chunk, i.e. still progress; `row_bytes.max(1)` only keeps the division defined.
    ((gguf_window_budget(max_bind) / row_bytes.max(1)) as usize)
        .max(1)
        .min(GEMV_MAX_WORKGROUPS * cols_per_wg)
}

/// The ILP entry's columns per workgroup on `device`, when the arm admits it:
/// `GEMV_WIDE_ILP` times a subgroup's width, behind `RLX_WGPU_GEMV_ILP`. One
/// spelling for the encoder, [`gemv_one_row_dispatches`] and the cells, so the
/// chunk cut and the entry it is cut for are sized from the same number.
pub fn gemv_ilp_cols_per_wg(device: &wgpu::Device, scheme_id: u32) -> Option<u32> {
    gemv_tiled_supports_scheme(scheme_id)
        .then(|| gemv_wide_cols_per_wg(device))
        .flatten()
        .filter(|_| rlx_ir::env::flag_or("RLX_WGPU_GEMV_ILP", true))
        .map(|c| c * GEMV_WIDE_ILP)
}

/// Whether the ILP entry serves a chunk of `n_chunk` columns, and at what width:
/// the chunk's floor, a whole number of the entry's workgroups, and neither of
/// those against a narrow Q6_K row (a row whose `k / 32` is two 32-element units
/// a lane), which keeps the one-column entry up to [`GEMV_ILP_Q6K_MIN_COLS`].
/// `ilp_cols` is that entry's width ([`gemv_ilp_cols_per_wg`], `None` where the
/// arm or its switch stands it down) and `sg` the arm's probed subgroup width
/// ([`gemv_subgroup_width`]); the caller reads both once per dispatch, so this is
/// a pure function of the chunk's shape. `scheme_from_id` must know `scheme_id`,
/// which every one-row call's does.
/// Public so the route cell derives the entry instead of restating the rule.
pub fn gemv_ilp_serves_chunk(
    ilp_cols: Option<u32>,
    sg: Option<u32>,
    scheme_id: u32,
    k: usize,
    n_chunk: usize,
) -> Option<u32> {
    let (ilp, sg) = (ilp_cols?, sg?);
    let narrow_q6k = matches!(
        scheme_from_id(scheme_id),
        rlx_ir::quant::QuantScheme::GgufQ6K
    ) && k / 32 == 2 * sg as usize;
    let floor = if narrow_q6k {
        GEMV_ILP_Q6K_MIN_COLS
    } else {
        GEMV_ILP_MIN_COLS
    };
    (n_chunk >= floor && n_chunk.is_multiple_of(ilp as usize)).then_some(ilp)
}

/// Output columns per chunk the per-row encoder cuts its N loop at: the budget
/// and grid cap above, rounded **down** to a whole number of ILP workgroups when
/// that entry is admissible, so the widest multiplies reach it on a chunking
/// adapter. The columns the cut leaves over are the narrow entry's. Public so the
/// route cell pins the chunk boundary it asserts instead of re-deriving it.
pub fn gemv_one_row_chunk_cols(device: &wgpu::Device, scheme_id: u32, k: usize) -> usize {
    chunk_cols(
        gemv_one_row_cols_per_chunk(
            gguf_row_bytes(scheme_id, k),
            gguf_max_bind(device),
            gemv_one_row_cols_per_wg(device, scheme_id) as usize,
        ),
        gemv_ilp_cols_per_wg(device, scheme_id),
    )
}

/// [`gemv_one_row_chunk_cols`]'s rounding, device-free: the largest whole number
/// of ILP workgroups inside `budget`, or `budget` itself when one does not fit —
/// which is the case the narrow entry then serves.
fn chunk_cols(budget: usize, ilp_cols: Option<u32>) -> usize {
    match ilp_cols {
        Some(ic) if budget >= ic as usize => budget / ic as usize * ic as usize,
        _ => budget,
    }
}

/// Output columns one workgroup covers on the **narrowest** entry of the arm this
/// scheme's one-row call takes on `device`: one subgroup per column on the
/// subgroup arm, else the tiled arm's two, else the legacy arm's one thread per
/// column. The ILP entry serves [`GEMV_WIDE_ILP`] times that on a chunk wide
/// enough to pay, so the grid cap below is written against this number — the
/// wider grid — and [`chunk_cols`] rounds the chunk it produces down to whole
/// ILP workgroups.
fn gemv_one_row_cols_per_wg(device: &wgpu::Device, scheme_id: u32) -> u32 {
    if !gemv_tiled_supports_scheme(scheme_id) {
        return GEMV_LEGACY_THREADS;
    }
    gemv_wide_cols_per_wg(device).unwrap_or(GEMV_TILED_COLS_PER_WG)
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
    Some(n.div_ceil(gemv_one_row_chunk_cols(device, scheme_id, k)))
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
/// tile size:
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
///
/// The shipped kernel was then taken to a taller tile with the accumulator footprint
/// paid down first, and lost at every height this form was taken to — 10, 12 and 15
/// rows — so the tile stays at 8. That header's sweep, its declared-storage probe and
/// the narrow accumulator the kernel now carries are where those readings live, and
/// the next height is not free even where the bytes fit — a 16-row tile is exactly
/// the 16 384 B WebGPU workgroup-storage baseline this const's tile is held under by
/// `grouped_geometry_matches_the_shader`, with no headroom.
const GROUPED_ROWS: u32 = 8;

/// Elements of `k` one staged activation tile covers — the shader's `BLK_ELEMS`,
/// which sizes its `xs` tile and the barrier pair the weight loop is written
/// around. Every arm of this entry set reads whole packing blocks out of it: the
/// tile is one 256-element Q4_K/Q6_K block, and eight of Q8_0's 32-element ones —
/// so for the K-quants this is also their `k` gate, while Q8_0's is its own block
/// ([`grouped_k_ok`]), whose tail tile that entry stops early on and stages only as
/// far as the row reaches. Pinned against the shader's own const by
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
///
/// `x` is the column axis and `y` the row-group axis, so a launch walks every
/// column chunk of row group 0 before row group 1 and the `m / 8` groups that
/// share a weight slice never read it from the same cache line. Swapping the two
/// axes is arithmetic-free and was tried: the recognition prefill measured
/// 577.4 → 576.8 ms warm (three round-robin pairs, inside the spread), so this
/// route is not limited by the weight re-reads the swap would have served but by
/// the staged-activation traffic and the dequant arithmetic. Do not re-take it as
/// a bandwidth win without a new reading.
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
    pass: &mut wgpu::ComputePass<'_>,
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
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            // Slab rows are [m, n_chunk]; this chunk owns arena columns
            // [n0, n0 + n_chunk).
            let (dst0, dst0_off) = arena.resolve_act(out_byte_off);
            if n_chunk == n && dst0_off as u64 + (m * n * 4) as u64 <= dst0.size() {
                encode_out_slab_copy(
                    arena,
                    device,
                    queue,
                    pass,
                    out_buf,
                    0,
                    out_byte_off,
                    (m * n) as u32,
                );
            } else {
                for r in 0..m {
                    encode_out_slab_copy(
                        arena,
                        device,
                        queue,
                        pass,
                        out_buf,
                        (r * n_chunk * 4) as u64,
                        out_byte_off + r * n * 4 + n0 * 4,
                        n_chunk as u32,
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
/// `x`, `weight` and `y` are 256 B-aligned **windows of the buffer each lives in**
/// — `x` and `y` of the arena, `weight` of the arena or of the separate weight
/// buffer when the plan parks it there — all declared read_write for the dispatch:
/// the one-row arm stores the row straight into the output node's own window
/// (`gemv_in_place_out_window`). WebGPU forbids overlapping writable ranges in one
/// dispatch whatever the access mode, and wgpu rejects a read-only binding on a
/// buffer another binding writes, so matching read_write declarations are what let
/// the three windows sit on one buffer — safe because they are disjoint and the
/// planner keeps `x` live until the store. The windowed bindings are
/// what keep each under the 4 GiB binding limit the whole-arena `as_entire_binding`
/// overruns on multi-GiB models. A pooled slab plus `encode_out_slab_copy` remains
/// for a sharded arena, a row crossing a shard boundary, or a window that cannot
/// hold the whole row.
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

/// The arena window a one-row multiply can store its whole output row through,
/// when the row resolves to one window: the buffer holding the row's
/// [`STORAGE_ALIGN`]-aligned base, that base's byte offset in it, the window's
/// size, and the row's f32 index within that window.
#[derive(Clone, Copy)]
struct GemvOutWindow<'a> {
    buf: &'a wgpu::Buffer,
    off: u64,
    size: u64,
    f32_base: u32,
}

/// Whether a row's byte offset can be stored through an arena window at all: f32
/// aligned, so an f32 index exists for it. The window's *range* is checked
/// separately by [`gemv_in_place_out_window`], against the other windows the same
/// dispatch binds.
fn gemv_in_place_out_ok(out_byte_off: usize) -> bool {
    out_byte_off.is_multiple_of(4)
}

/// A byte count as `BufferBinding` covers it: rounded up to 16. Every window the
/// one-row arm binds goes through this one spelling, so the window the kernel
/// writes and the window the alias guard compares cannot disagree — a row whose
/// stride is not a multiple of 16 (Q6_K's 210 bytes) would otherwise reach into a
/// window the comparison called clear.
fn round16(bytes: u64) -> u64 {
    bytes.div_ceil(16) * 16
}

/// A window inside one buffer, as the binding covers it: from `local` to the
/// 16-byte-rounded end, clamped to the buffer. `raw_end` is where the operand
/// itself ends; the step to the bound end is [`round16`]'s, and the alias guard
/// must measure the bound window, not the raw one.
fn bound_window(local: u64, raw_end: u64, buf_size: u64) -> (u64, u64) {
    (local, round16(raw_end).min(buf_size))
}

/// The weight binding's size for one chunk of the one-row multiply's rows, relative
/// to the window's own base: cut on a **16-byte boundary**, which is what the kernel
/// reads — a block is decoded as 16-byte legs (`read_w16`), and a leg off a group
/// boundary is assembled from the two groups it straddles, so the last leg's second
/// group lies one group past it. `round16` of the last row's end therefore *covers*
/// that fetch (no access leaves it, at any scheme's stride), and a window clamped
/// below the round-up would leave the group outside the binding, where a robust
/// access zeroes it and the tail column silently reads a zero scale.
///
/// So the buffer under it has to end on a 16-byte multiple: the weight buffer and
/// its shards are sized that way, and the activation arena's own round-up is
/// `Arena::from_plan`'s (a packed weight keeps its native width under
/// `plan_f32_uniform`, so it can be the slot the arena ends on). This asserts the
/// invariant where the window is cut rather than clamping to whatever the buffer
/// has, because a clamp is the silent wrong tail column.
fn gemv_weight_window(w0: u64, chunk_bytes: u64, w_base: u64, w_buf_size: u64) -> u64 {
    let need = round16(w0 + chunk_bytes - w_base);
    let avail = w_buf_size.saturating_sub(w_base);
    assert!(
        need <= avail,
        "rlx-wgpu gguf gemv: the weight window's 16-byte round-up needs {need} B from \
         base {w_base}, but the {w_buf_size}-byte weight buffer has {avail} — the \
         one-row multiply reads its packed blocks as 16-byte legs, so that buffer (and \
         each of its shards) must end on a 16-byte multiple"
    );
    need
}

/// Whether two byte ranges of one buffer overlap — the term WebGPU's writable-alias
/// rule turns on.
fn windows_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The window one output row's slot resolves to inside its buffer: the row's byte
/// offset rounded **down** to [`STORAGE_ALIGN`] (a storage binding's offset must be
/// a multiple of it) and the 16-byte-rounded length that covers the row from there.
/// Shared with [`gemv_in_place_out_window`], and pinned by
/// `a_row_inside_a_storage_block_brings_the_slot_below_it_into_the_window`, because
/// a dropped round-up or a base that did not round down would silently shrink the
/// window the alias check below measures.
fn in_place_out_span(out_byte_off: usize, n: usize) -> (u64, u64) {
    let out0 = out_byte_off as u64;
    let base = (out0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    (base, round16(out0 + n as u64 * 4 - base))
}

/// Whether an output window keeps clear of every other window this dispatch binds
/// writable. `None` is an operand that lives in another buffer and so cannot
/// overlap; `Some(range)` one that shares this buffer.
fn in_place_window_clear(out: (u64, u64), x: Option<(u64, u64)>, w: Option<(u64, u64)>) -> bool {
    // Every pair: the output against both operands, and the two operands against
    // each other — all three bindings are writable, so each pair must be disjoint.
    [x, w]
        .into_iter()
        .flatten()
        .all(|other| !windows_overlap(out, other))
        && match (x, w) {
            (Some(a), Some(b)) => !windows_overlap(a, b),
            _ => true,
        }
}

/// The window `out_byte_off .. out_byte_off + n * 4` resolves to, or `None` for the
/// pooled slab and its copying dispatch, because the row does not resolve to one
/// buffer/window that holds all `n` outputs: a sharded arena splits the logical
/// arena across physical buffers, a row crossing a shard boundary (or a truncated
/// window) would not hold every output, and a row whose byte offset is not
/// f32-aligned has no f32 index for it.
///
/// The window's base rounds **down** to [`STORAGE_ALIGN`] (a storage binding's
/// offset must be a multiple of it) while the plan aligns slots to 16 bytes, so a
/// row that starts inside a 256-byte block brings the tail of the slot below it
/// into the binding. The row is stored in place only when that slack — and the
/// window's own 16-byte round-up — reaches neither the X row nor the packed
/// weight, both of which this dispatch also binds writable:
/// [`windows_overlap`] against those two, so the three windows are disjoint in
/// fact and not only by construction. `RLX_WGPU_GEMV_IN_PLACE=0` stands the store
/// down to the pooled slab, which is how its saving is re-taken.
fn gemv_in_place_out_window(
    arena: &Arena,
    out_byte_off: usize,
    n: usize,
    x_byte_off: usize,
    k: usize,
    w_byte_off: usize,
    w_row_bytes: u64,
) -> Option<GemvOutWindow<'_>> {
    if arena.is_sharded()
        || !gemv_in_place_out_ok(out_byte_off)
        || !rlx_ir::env::flag_or("RLX_WGPU_GEMV_IN_PLACE", true)
    {
        return None;
    }
    let out0 = out_byte_off as u64;
    let (base, need) = in_place_out_span(out_byte_off, n);
    let (buf, local) = arena.resolve_act(base as usize);
    let local = local as u64;
    if need > buf.size().saturating_sub(local) {
        return None;
    }
    let out_range = (local, local + need);
    // X: the row the multiply reads, from its own align-down base to the end of
    // its last element (both windows of one buffer only when the arena is one),
    // as the binding covers it.
    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_range = std::ptr::eq(x_buf, buf).then(|| {
        bound_window(
            x_local as u64,
            x_local as u64 + (x0 + (k * 4) as u64 - x_base),
            x_buf.size(),
        )
    });
    // The packed weight: every chunk's rows this call reads, from their own
    // align-down base to the last row's end, as those bindings cover it.
    let (w_buf, w_local) = arena.resolve_w(w_byte_off);
    let w_base = (w_local as u64 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let w_range = std::ptr::eq(w_buf, buf).then(|| {
        bound_window(
            w_base,
            w_local as u64 + n as u64 * w_row_bytes,
            w_buf.size(),
        )
    });
    if !in_place_window_clear(out_range, x_range, w_range) {
        return None;
    }
    Some(GemvOutWindow {
        buf,
        off: local,
        size: need,
        f32_base: ((out0 - base) / 4) as u32,
    })
}

/// Encode the one-row packed multiply's chunks into `out`, at
/// `out.f32_base + n0` for chunk column `n0` — the window the caller resolved
/// (the arena slot, or the pooled slab).
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_gguf_gemv_one_into_out(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::ComputePass<'_>,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out: GemvOutWindow<'_>,
    cache: Option<&GemvSetupCache>,
) {
    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_size = round16(x0 + (k * 4) as u64 - x_base).min(x_buf.size().saturating_sub(x_local));

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
    // Three arms, one route: the subgroup arm gives each whole subgroup one
    // output column (four columns to a workgroup at this machine's width) and
    // reduces inside it — or `WIDE_ILP` columns a subgroup, one accumulator each;
    // the tiled arm gives each workgroup `GEMV_TILED_COLS_PER_WG` output
    // columns and splits each column's contraction over its threads; the original
    // gives one thread per column, 64 columns to a workgroup. Which one ran is
    // recorded — see [`gemv_one_row_shape`] — and `RLX_WGPU_GEMV_WIDE=0|shared`
    // refuses the subgroup arm so the workgroup-memory one is exercised where
    // subgroups are present.
    let tiled = gemv_tiled_supports_scheme(scheme_id);
    // The kernel is picked from `wide_cols` and the chunk cut from the same two
    // sources — the probe and the scheme — so the width the chunking rule divides
    // by cannot describe a different arm than the kernel that runs.
    let wide_cols = tiled.then(|| gemv_wide_cols_per_wg(device)).flatten();
    let ilp_cols = gemv_ilp_cols_per_wg(device, scheme_id);
    let sg = gemv_subgroup_width(device);
    let cols_per_chunk = gemv_one_row_chunk_cols(device, scheme_id, k);

    let mut n0 = 0usize;
    while n0 < n {
        let n_chunk = (n - n0).min(cols_per_chunk);
        // The floors and the divisibility are the *chunk's* terms: the entry's
        // longer per-thread walk only pays on a chunk wide enough to hide it, and
        // the form stores `GEMV_WIDE_ILP` columns per subgroup with no tail, so a
        // chunk that is not a whole number of its workgroups keeps the
        // one-column-per-subgroup entry. `RLX_WGPU_GEMV_ILP=0` stands the entry
        // down, which is how its saving is re-taken.
        let ilp = gemv_ilp_serves_chunk(ilp_cols, sg, scheme_id, k, n_chunk);
        // One table for all of them: the arm's kernel, the geometry it runs and the
        // arm's identity — which keys the setup cache and is what
        // [`gemv_one_row_shape`] records — are chosen together, so a further arm
        // cannot desync the dispatch grid, the entry it dispatches or a bind group
        // reused from the other arm's layout.
        let (dk, cols_wg, threads_wg, workgroups, arm) = match (ilp, wide_cols) {
            // The form's columns a subgroup, so `ic` columns a workgroup; the chunk
            // is a whole number of them by construction.
            (Some(ic), _) => (
                dequant_gemv_gguf_wide_ilp_kernel(device),
                ic,
                GEMV_WIDE_THREADS,
                (n_chunk / ic as usize) as u32,
                1,
            ),
            // One whole subgroup per output column; the last workgroup rounds up
            // and its off-the-end subgroup returns before `subgroupAdd`.
            (None, Some(cols)) => (
                dequant_gemv_gguf_wide_kernel(device),
                cols,
                GEMV_WIDE_THREADS,
                (n_chunk as u32).div_ceil(cols),
                2,
            ),
            // `GEMV_TILED_COLS_PER_WG` output columns per workgroup.
            (None, None) if tiled => (
                dequant_gemv_gguf_tiled_kernel(device),
                GEMV_TILED_COLS_PER_WG,
                GEMV_TILED_THREADS,
                (n_chunk as u32).div_ceil(GEMV_TILED_COLS_PER_WG),
                3,
            ),
            // One thread per output column, 64 columns per workgroup.
            (None, None) => (
                dequant_gemv_gguf_kernel(device),
                GEMV_LEGACY_THREADS,
                GEMV_LEGACY_THREADS,
                (n_chunk as u32).div_ceil(GEMV_LEGACY_THREADS),
                4,
            ),
        };
        let w0 = w0_all + n0 as u64 * row_bytes;
        let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
        let chunk_bytes = n_chunk as u64 * row_bytes;
        let w_size = gemv_weight_window(w0, chunk_bytes, w_base, w_buf_size);
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
            out_f32_off: out.f32_base + n0 as u32,
            _p0: 0,
            _p1: 0,
        };

        // This step's setup is a function of the schedule, so a frame after the
        // first reuses the uniform and the bind group instead of rebuilding them.
        let key = gemv_setup_key(
            p,
            arm,
            (x_local, x_size),
            (w_base, w_size),
            (out.off, out.size),
        );
        let pooled = cache.filter(|_| rlx_ir::env::flag_or("RLX_WGPU_GEMV_POOL", true));
        if let Some(cache) = pooled {
            if cache.dispatch(&key, dk, pass, cols_wg, threads_wg, workgroups) {
                n0 += n_chunk;
                continue;
            }
        }
        let u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu dequant_gemv_gguf uniform"),
            size: std::mem::size_of::<DequantGemvGgufParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A queued write, not a creation mapping: `small_step_dispatch_cost` times
        // this driver's create-and-map-and-write at 6.7 µs against 5.8 for
        // create-then-`write_buffer`, so the mapping is the slower of the two.
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: out.buf,
                        offset: out.off,
                        size: wgpu::BufferSize::new(out.size),
                    }),
                },
            ],
        });

        {
            pass.set_pipeline(&dk.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            note_gemv_one_row_shape(cols_wg, threads_wg, workgroups, arm);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        if let Some(cache) = pooled {
            cache.insert(key, u, bg);
        }
        n0 += n_chunk;
    }
}

/// Per-executable cache of a one-row multiply's per-dispatch setup.
///
/// The uniform record and the bind group are a function of the compiled schedule:
/// the same step reads and writes the same windows of the same arena every frame,
/// so after the first frame neither has to be created, written or grouped again.
/// Measured, that setup is ~5.7 µs an op (`small_step_dispatch_cost`) and a frame
/// issues ~516 one-row multiplies — ~2.9 ms of a ~63 ms frame, of which the
/// encode interval exposes ~2 ms.
///
/// The key carries the uniform record itself plus the output window's extent, so
/// a pooled slab the allocator had to grow cannot be served from a stale entry.
/// What a cached entry must match: the uniform record (which carries the
/// contraction, the columns, the scheme and the *relative* offsets) plus the
/// windows the bind group was built from — their buffer-local bases and sizes, and
/// the output window's. Two calls with one record can still resolve different
/// windows when their align-down bases differ, so the record alone is not a key.
///
/// The **arm** is a key element because which entry runs is a live decision
/// (`RLX_WGPU_GEMV_WIDE`, `_ILP`): the tiled, wide and ILP entries have
/// different bind-group layouts, so a switch flipped between two runs of one
/// compiled executable would otherwise serve a bind group against a foreign
/// pipeline.
///
/// Buffer *identity* is not a key element: the cache lives on one compile's arena
/// and weight buffers, and a lazy recompile replaces the arena — the rebind takes
/// the fresh, empty cache with it — while the one buffer that can change inside
/// one compile, the pooled output slab, is keyed by its size.
pub type GemvSetupKey = (DequantGemvGgufParams, u32, u64, u64, u64, u64, u64, u64);

/// The key of one one-row dispatch: the record, the arm, then X's, the weight's
/// and the output's window.
fn gemv_setup_key(
    p: DequantGemvGgufParams,
    arm: u32,
    x: (u64, u64),
    w: (u64, u64),
    out: (u64, u64),
) -> GemvSetupKey {
    (p, arm, x.0, x.1, w.0, w.1, out.0, out.1)
}

#[derive(Default)]
pub struct GemvSetupCache {
    slots:
        std::sync::Mutex<std::collections::HashMap<GemvSetupKey, (wgpu::Buffer, wgpu::BindGroup)>>,
}

impl GemvSetupCache {
    /// Dispatch from the cache when this exact call was encoded before. The arm
    /// is the key's own (`key.1`), so the record the setup was built for and the
    /// one the shape reports cannot disagree.
    fn dispatch(
        &self,
        key: &GemvSetupKey,
        kernel: &Kernel,
        pass: &mut wgpu::ComputePass<'_>,
        cols_wg: u32,
        threads_wg: u32,
        workgroups: u32,
    ) -> bool {
        let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let Some((_, bg)) = slots.get(key) else {
            return false;
        };
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        note_gemv_one_row_shape(cols_wg, threads_wg, workgroups, key.1);
        true
    }

    fn insert(&self, key: GemvSetupKey, uniform: wgpu::Buffer, bg: wgpu::BindGroup) {
        // A poisoned lock is a fault elsewhere, not a reason to take the process
        // down on every frame after it: the map itself is still sound.
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, (uniform, bg));
    }
}

/// Encode a one-row packed multiply for `[1, k] x [n, k]^T`, storing the output row
/// straight into the arena window `gemv_in_place_out_window` resolves — the
/// one-row arm's whole result, so no second dispatch moves it out of a slab. A row
/// that does not resolve to one arena window (see that helper) keeps the pooled
/// slab and the `encode_out_slab_copy` dispatch that follows it.
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_gguf_gemv_one(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::ComputePass<'_>,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
    cache: Option<&GemvSetupCache>,
) {
    match gemv_in_place_out_window(
        arena,
        out_byte_off,
        n,
        x_byte_off,
        k,
        w_byte_off,
        gguf_row_bytes(scheme_id, k),
    ) {
        Some(out) => encode_dequant_matmul_gguf_gemv_one_into_out(
            arena, device, queue, pass, k, n, scheme_id, x_byte_off, w_byte_off, out, cache,
        ),
        None => {
            let out_bytes = ((n * 4).max(4) as u64).div_ceil(16) * 16;
            with_pooled_out_buf(device, out_bytes, |out_buf| {
                encode_dequant_matmul_gguf_gemv_one_into_out(
                    arena,
                    device,
                    queue,
                    pass,
                    k,
                    n,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    GemvOutWindow {
                        buf: out_buf,
                        off: 0,
                        size: out_buf.size(),
                        f32_base: 0,
                    },
                    cache,
                );
                encode_out_slab_copy(
                    arena,
                    device,
                    queue,
                    pass,
                    out_buf,
                    0,
                    out_byte_off,
                    n as u32,
                );
            });
        }
    }
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let idx_bytes = arena.read_bytes_range(device, queue, idx_byte_off, m * 4);
    let idx_host: Vec<f32> = idx_bytes
        .as_chunks::<4>()
        .0
        .iter()
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
        .as_chunks::<4>()
        .0
        .iter()
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
        DequantGemvGgufParams, GEMV_ILP_MIN_COLS, GEMV_TILED_COLS_PER_WG, GEMV_TILED_THREADS,
        GEMV_WIDE_ILP, GEMV_WIDE_THREADS, GROUPED_ROWS, GROUPED_SPLIT_COLS, GROUPED_STAGE_ELEMS,
        GROUPED_THREADS, bound_window, chunk_cols, gemv_in_place_out_ok,
        gemv_one_row_cols_per_chunk, gemv_setup_key, gemv_weight_window, gemv_wide_enabled,
        gguf_row_bytes, grouped_chunk_geometry, grouped_cols_per_chunk, in_place_out_span,
        in_place_window_clear, round16, windows_overlap,
    };

    /// The setup cache is served on the key alone — there is no second check at the
    /// dispatch — so every term the bind group depends on has to be in the key:
    /// the arm (the entries' layouts differ), both operands' windows, and the
    /// output's, whose size is how a pooled slab the allocator grew stops matching
    /// the entry built against the smaller one. A term dropped here is a stale bind
    /// group, which this driver accepts silently.
    #[test]
    fn the_setup_key_carries_the_arm_and_every_window() {
        let p = DequantGemvGgufParams {
            k: 1024,
            n: 1024,
            scheme_id: 20,
            x_f32_off: 0,
            w_byte_off: 0,
            out_f32_off: 0,
            _p0: 0,
            _p1: 0,
        };
        let (x, w, out) = ((0, 4112), (0, 1_114_112), (0, 4096));
        let base = gemv_setup_key(p, 2, x, w, out);
        for (what, other) in [
            ("the arm", gemv_setup_key(p, 1, x, w, out)),
            ("the record", {
                let mut q = p;
                q.n = 2048;
                gemv_setup_key(q, 2, x, w, out)
            }),
            ("x's base", gemv_setup_key(p, 2, (256, 4112), w, out)),
            ("x's size", gemv_setup_key(p, 2, (0, 4128), w, out)),
            (
                "the weight's base",
                gemv_setup_key(p, 2, x, (256, 1_114_112), out),
            ),
            (
                "the weight's size",
                gemv_setup_key(p, 2, x, (0, 1_114_096), out),
            ),
            ("the output's base", gemv_setup_key(p, 2, x, w, (16, 4096))),
            ("the output's size", gemv_setup_key(p, 2, x, w, (0, 8192))),
        ] {
            assert_ne!(base, other, "{what} must distinguish two dispatches");
        }
    }

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
        // The tile is this entry set's whole workgroup storage, so it is what a
        // device's 16 384 B WebGPU baseline has to hold — a form that needs more is
        // a form a baseline device cannot run, and this fork takes the portable one.
        // Strictly under, not at: a 16-row tile is exactly the baseline, which is the
        // "no headroom" case `GROUPED_ROWS` refuses rather than a form this const may
        // move to.
        const BASELINE_WORKGROUP_STORAGE: u32 = 16_384;
        const VEC4_BYTES: u32 = 16; // `vec4<f32>`, the shader's `xs` element
        let tile_bytes = xs_slots * VEC4_BYTES;
        assert!(
            tile_bytes < BASELINE_WORKGROUP_STORAGE,
            "the staged tile is {tile_bytes} B, not under the {BASELINE_WORKGROUP_STORAGE} B \
             WebGPU baseline"
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
            // The reduction's own shape: a halving tree from `LANES / 2`, so the
            // number of partial sums a store reads is the width the dispatch declares.
            "var step = LANES / 2u;".to_string(),
            // The entry's column span, and the second column's reduction slots and
            // store — the places that column's geometry is spelled out.
            "let j0 = wid.x * COLS_PER_WG;".to_string(),
            "red[LANES + lane] = acc1;".to_string(),
            "outarr[params.out_f32_off + j0 + 1u]".to_string(),
        ] {
            assert!(wgsl.contains(&want), "expected `{want}` in the shader");
        }
    }

    /// The subgroup arm's geometry is written twice the same way: the encoder
    /// sizes its dispatch from `GEMV_WIDE_THREADS / subgroup_width`, and the
    /// shader declares its own `@workgroup_size` plus the three lines that map an
    /// invocation onto a column, name the reduction and place the store. Read
    /// both spellings here so the fast, device-free gate sees them.
    ///
    /// It also pins the module split the arm forced: the shared-memory entry
    /// points are created on every device, so the text they are built from must
    /// not contain the arm's subgroup built-ins — a module that names one is
    /// refused without `wgpu::Features::SUBGROUP` (`Capabilities::SUBGROUP` to
    /// naga), and the fallback for such a device *is* those entry points.
    #[test]
    fn wide_gemv_geometry_matches_the_shader() {
        let shared = crate::kernels::DEQUANT_GEMV_GGUF_WGSL;
        let full = crate::kernels::DEQUANT_GEMV_GGUF_WIDE_WGSL;
        assert!(
            shared.contains("fn dequant_gemv_tiled(") && shared.contains("fn dequant_gemv("),
            "the shared-memory module must keep both entry points"
        );
        assert!(
            !shared.contains("fn dequant_gemv_wide"),
            "the shared-memory module must stop before the subgroup arm: it is built \
             on every device, including ones without Features::SUBGROUP"
        );
        let module = naga::front::wgsl::parse_str(shared).expect("wgsl parse");
        // The substantive half of the split: no subgroup capability at all, so a
        // device that does not carry the feature can still create both entry
        // points. A subgroup built-in left above the marker fails right here.
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("the shared-memory arms must validate without the SUBGROUP capability");

        let module = naga::front::wgsl::parse_str(full).expect("wgsl parse");
        // The wide arm's own validity, which `parse_str` alone does not establish:
        // a module that names a subgroup built-in also has to validate with the
        // capability a device carrying `Features::SUBGROUP` grants. A WGSL mistake
        // here would otherwise surface only as a `create_shader_module` panic on
        // the machine that admits the arm.
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SUBGROUP,
        )
        .validate(&module)
        .expect("the subgroup arm must validate with the SUBGROUP capability");
        let wide = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "dequant_gemv_wide")
            .expect("the subgroup one-row entry point");
        assert_eq!(
            wide.workgroup_size,
            [GEMV_WIDE_THREADS, 1, 1],
            "dequant_gemv_wide: workgroup must be GEMV_WIDE_THREADS wide"
        );
        for want in [
            format!("const WIDE_THREADS: u32 = {GEMV_WIDE_THREADS}u;"),
            "let cols = WIDE_THREADS / sg;".to_string(),
            // The column comes from `subgroup_id`, the builtin that is defined to
            // be the subgroup's index, not from `local_invocation_id / sg`, which
            // WGSL leaves unrelated to it.
            "@builtin(subgroup_id) sgid: u32,".to_string(),
            "let j = wid.x * cols + sgid;".to_string(),
            // And the decode is the tiled arm's — one function per scheme — so the
            // two arms cannot drift apart in block addressing or scale decode.
            "for (var u = sid; u < nu; u = u + sg)".to_string(),
            "acc = acc + dq_q4k_cover(j, sid, sg, nbk, xb4, false).x;".to_string(),
            "acc = acc + dq_q6k_unit(j, u, nbk, xb4, false).x;".to_string(),
            "acc = acc + dq_q8_0_unit(j, u, q8blk, xb4, false).x;".to_string(),
            "let total = subgroupAdd(acc);".to_string(),
            "if (sid == 0u) { outarr[params.out_f32_off + j] = total; }".to_string(),
        ] {
            assert!(full.contains(&want), "expected `{want}` in the shader");
        }

        // The ILP entry: the host sizes its grid from `GEMV_WIDE_ILP` and the
        // shader takes that many columns a subgroup, so the two spellings must be
        // one number — a mismatch writes columns twice or never.
        let ilp = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "dequant_gemv_wide_ilp")
            .expect("the ILP subgroup entry point");
        assert_eq!(
            ilp.workgroup_size,
            [GEMV_WIDE_THREADS, 1, 1],
            "dequant_gemv_wide_ilp: workgroup must be GEMV_WIDE_THREADS wide"
        );
        for want in [
            format!("const WIDE_ILP: u32 = {GEMV_WIDE_ILP}u;"),
            "let j = wid.x * (cols * WIDE_ILP) + sgid * WIDE_ILP;".to_string(),
            // The tail guard: the host cuts chunks to whole ILP workgroups, but the
            // grid is sized from the probed width while this entry reads the
            // dispatch-time one, so a disagreement must skip a column rather than
            // store past the chunk's window.
            "if (j + WIDE_ILP > params.n) { return; }".to_string(),
            // Every scheme the ILP entry carries, decoding both of its columns:
            // each loop's decode request and accumulate, not just Q8_0's, so a
            // scheme left on the one-column `second = false` path goes red here.
            "acc = acc + dq_q4k_cover(j, sid, sg, nbk, xb4, true);".to_string(),
            "acc = acc + dq_q6k_unit(j, u, nbk, xb4, true);".to_string(),
            "acc = acc + dq_q8_0_unit(j, u, q8blk, xb4, true);".to_string(),
            // Two columns a subgroup, stored by the first invocation: the pair the
            // form's accumulators cover, and no third store.
            "outarr[params.out_f32_off + j + 1u] = total.y;".to_string(),
        ] {
            assert!(full.contains(&want), "expected `{want}` in the shader");
        }
        // Q4_K's group-or-sub-block choice is one body both entries call, so a
        // second copy of the gate is the drift this test exists to catch.
        assert_eq!(
            full.matches("if (ng >= 2u * sg) {").count(),
            1,
            "the Q4_K cover's gate belongs in `dq_q4k_cover`, once"
        );
        // `total` is the `vec2` pair and nothing beyond it: a third column would
        // need a third component the entry does not have.
        assert!(
            !full.contains("j + 2u") && full.contains("var acc = vec2<f32>(0.0);"),
            "the ILP entry covers WIDE_ILP = {GEMV_WIDE_ILP} columns only"
        );
    }

    /// The in-place window's admission rules, device-free: the row must be f32
    /// aligned, and its window must reach no other window the same dispatch binds
    /// writable. The base rounds down to a `STORAGE_ALIGN` boundary while the plan
    /// aligns slots to 16 bytes, so a row inside a block brings the slot below it —
    /// the X row's among them — into the binding; `gguf_gpu`'s resolver refuses
    /// exactly those.
    #[test]
    fn the_in_place_window_admits_only_a_row_whose_slack_reaches_no_other_window() {
        assert!(gemv_in_place_out_ok(0) && gemv_in_place_out_ok(16 * 3));
        assert!(
            !gemv_in_place_out_ok(2),
            "a row two bytes in has no f32 index"
        );

        // A row's own window, and the two neighbours it could reach into.
        let out = (1024u64, 1024 + 256);
        assert!(!windows_overlap(out, (1536, 2048)), "the slot above it");
        assert!(
            !windows_overlap(out, (0, 1024)),
            "the slot below ends where this one begins"
        );
        assert!(
            windows_overlap(out, (1024, 1040)),
            "a row inside this block brings its slot's tail into the binding"
        );
    }

    /// The chunk the per-row encoder cuts its N loop at: the binding budget and
    /// the dispatch cap, rounded down to whole ILP workgroups when that entry is
    /// admissible. Without the rounding the widest multiplies — the ones the entry
    /// exists for — would be served entirely by the narrow entry on any adapter
    /// that chunks, because a budget-derived boundary is almost never a whole
    /// number of ILP workgroups.
    #[test]
    fn the_chunk_is_a_whole_number_of_ilp_workgroups_when_that_entry_is_admissible() {
        // Q8_0 at k = 1024 packs a row into 1088 bytes; at the 128 MiB floor the
        // budget alone allows ~123k columns, far below the grid cap.
        let (row, max_bind) = (1088u64, 128u64 << 20);
        let budget = gemv_one_row_cols_per_chunk(row, max_bind, 4);
        assert!(
            budget >= GEMV_ILP_MIN_COLS,
            "the fixture must reach the ILP entry's floor: {budget}"
        );
        assert_eq!(
            chunk_cols(budget, None),
            budget,
            "no ILP entry, no rounding"
        );
        for ilp in [8u32, 16, 128] {
            let cut = chunk_cols(budget, Some(ilp));
            assert!(
                cut.is_multiple_of(ilp as usize) && cut <= budget,
                "a cut of {cut} columns must be whole {ilp}-column workgroups inside {budget}"
            );
            assert!(
                cut > budget - ilp as usize,
                "the cut must keep the budget's columns: {cut} of {budget}"
            );
        }
        // A budget too small to hold one workgroup keeps its own value, and the
        // narrow entry serves it — the last-chunk case of a call this wide.
        let tiny = gemv_one_row_cols_per_chunk(row, 4096, 4);
        assert!(
            tiny < 8,
            "a 4 KiB budget holds {tiny} rows, not a workgroup"
        );
        assert_eq!(chunk_cols(tiny, Some(8)), tiny);
    }

    /// The refusal the resolver makes, at the geometry its own doc names: Q8_0 at
    /// `k = 64` packs a row into 68 bytes, so an output row whose slot starts 16
    /// bytes into a 256-byte block opens a window that spans the X row's own.
    #[test]
    fn a_row_inside_a_storage_block_brings_the_slot_below_it_into_the_window() {
        // The window opens at the block's base, not at the row: 16 B in, four
        // outputs, so 32 bytes from the base once the 16-byte round-up is applied.
        assert_eq!(in_place_out_span(16, 4), (0, 32));
        assert_eq!(
            in_place_out_span(1024, 4),
            (1024, 16),
            "a row on the block's own boundary opens there, at its own length"
        );

        // X begins at the arena's base: the window above reaches it, and the
        // resolver keeps the slab.
        assert!(!in_place_window_clear((0, 32), Some((0, 256)), None));
        // Two rows further into the block, clear of X's 68-byte row.
        assert!(in_place_window_clear((256, 288), Some((0, 68)), None));
        // The packed weight's window is measured the same way.
        assert!(!in_place_window_clear((0, 32), None, Some((0, 1088))));
        assert!(
            in_place_window_clear((0, 32), Some((32, 68)), Some((512, 1600))),
            "both operands in this buffer, both clear of the row's window"
        );
        assert!(
            in_place_window_clear((0, 32), None, None),
            "an operand in another buffer cannot overlap"
        );

        // The window the guard measures is the 16-byte-rounded one the binding
        // covers, which is where Q6_K's 210-byte rows differ from the raw end: the
        // guard and the kernel must be comparing one window, not two.
        assert_eq!(round16(7 * 210), 1472);
        assert_eq!(bound_window(0, 7 * 210, 4096), (0, 1472));
        assert_eq!(
            bound_window(0, 7 * 210, 1000),
            (0, 1000),
            "a window is clamped to the bytes its buffer has"
        );
        // And the two operands are compared with each other: all three bindings are
        // writable, so a window shared by X and the weight is refused even when the
        // output row is clear of both.
        assert!(!in_place_window_clear(
            (0, 32),
            Some((1024, 2048)),
            Some((1536, 1536 + 1088))
        ));
    }

    /// The one-row multiply's weight window ends on the 16-byte boundary above the
    /// last row's packed bytes, because the kernel's last leg fetch reaches one group
    /// past them; a buffer that cannot give that group is refused instead of bound.
    /// Q6_K's 210-byte rows are the case that makes it visible — six of them are 1 260
    /// bytes, which the weight buffer's old four-byte rounding would have ended at
    /// exactly, 12 bytes short of the group.
    #[test]
    fn the_weight_window_ends_on_a_16_byte_boundary() {
        // The window is the round-up, from the base, whatever the base is.
        assert_eq!(gemv_weight_window(0, 6 * 210, 0, 1_264), 1_264);
        // A weight parked past a 256-byte block base: the length covers its own
        // rows from the base, so the base's slack is included and the end is still
        // on a group boundary.
        assert_eq!(gemv_weight_window(2_500, 6 * 210, 2_304, 4_096), 1_456);
        // A weight that fits with whole groups to spare adds nothing: the window is
        // the rows, not the buffer.
        assert_eq!(gemv_weight_window(0, 6 * 210, 0, 4_096), 1_264);
    }

    /// The refusal: a weight buffer that ends off a 16-byte boundary — what the
    /// buffer's four-byte rounding used to produce — takes the last leg's second
    /// group outside the binding, where a robust access answers zeros and the tail
    /// column reads a zero scale. The window cut says so rather than clamping.
    #[test]
    #[should_panic(expected = "16-byte multiple")]
    fn a_weight_buffer_that_ends_off_16_is_refused() {
        // 1 260 bytes of Q6_K rows in a buffer rounded to four bytes: no group for
        // the last leg.
        gemv_weight_window(0, 6 * 210, 0, 1_260);
    }

    /// The knob's spelling: `1`/`true` (and unset, the measured default) admit the
    /// subgroup arm; `0`, `false`, `off`, `no` and `shared` keep the
    /// workgroup-memory one. Brackets the process-global
    /// environment through `RuntimeOverrides`, which restores whatever the run was
    /// started with, so a measurement session's own `RLX_WGPU_GEMV_WIDE` is not
    /// disturbed.
    #[test]
    fn the_wide_knob_refuses_on_zero_and_shared() {
        for (set, want) in [
            (None, true),
            (Some("1"), true),
            (Some("true"), true),
            (Some("yes"), true),
            (Some("0"), false),
            (Some("false"), false),
            (Some("off"), false),
            (Some("no"), false),
            (Some("shared"), false),
        ] {
            let _guard =
                rlx_ir::env::RuntimeOverrides::install(set.map(|v| ("RLX_WGPU_GEMV_WIDE", v)));
            assert_eq!(gemv_wide_enabled(), want, "RLX_WGPU_GEMV_WIDE={set:?}");
        }
    }

    /// The row-grouped 8-bit entry stages no activation past the call's own window.
    ///
    /// The class's value cells cannot witness this: the surplus past a row is never
    /// consumed, so they pass with and without the bound. What is pinned here is the
    /// shader's own guard over the staged fetch — `p0 + c4 < params.k`, whose left
    /// operand is the very expression the fetch's address is built from
    /// (`((p0 + c4) >> 2u)`, the `vec4` it loads), with the fetch itself inside the
    /// guard. A guard that only masks the value (a `select`), one moved onto the
    /// store with the load left outside it, and one written against a tile's
    /// remainder rather than the contraction's own extent all fail this. Reading the
    /// shader as text costs what the geometry pin above says it costs: a reflow of
    /// these lines fails it without any rule having moved, and the fix is to
    /// re-spell them.
    ///
    /// The module carries the other half, which no text pin reaches: the bounded
    /// routine's staged store sits under a conditional and the unbounded routine's
    /// under none, so the two 256-element-block schemes' staging carries no compare
    /// — the class's bound is the 8-bit entry's alone, and their staged element set
    /// and trip counts are what they were. The guard tests the contraction's own
    /// extent, not a tile's remainder, so its reach is a property of the guard and
    /// not of a shape: the class's off-tile rows — a block past a tile (288),
    /// several tiles plus a block (896), the sub-tile ends (128, 32) — are the route
    /// census's cells, whose route, join and values `gguf_matmul_route.rs` asserts.
    #[test]
    fn the_q8_0_tail_tile_fetches_nothing_past_the_call_window() {
        let wgsl = crate::kernels::DEQUANT_GEMM_GGUF_WGSL;
        // Device-free validity: a WGSL mistake in the staging routines would
        // otherwise surface only where an adapter builds the module.
        let module = naga::front::wgsl::parse_str(wgsl).expect("wgsl parse");

        // The guard and the fetch under it. The fetch must be inside the guard's own
        // block — a guard the block closes before is a guard on the store alone,
        // which leaves the address free to leave the window.
        let guard = "if (p0 + c4 < params.k) {";
        let fetch = "xs[i] = xarr[xb + min(row0 + r, last_row) * k4 + ((p0 + c4) >> 2u)];";
        let guard_at = wgsl
            .find(guard)
            .unwrap_or_else(|| panic!("the staged fetch must be bounded: no `{guard}`"));
        let fetch_at = guard_at
            + wgsl[guard_at..]
                .find(fetch)
                .unwrap_or_else(|| panic!("the guarded statement must be the fetch `{fetch}`"));
        assert!(
            !wgsl[guard_at..fetch_at].contains('}'),
            "`{guard}` must wrap the fetch itself, not close over it"
        );
        // The two staging routines differ by that guard alone: a fetch respelled in
        // one of them would be a second address to keep right, silently.
        assert_eq!(
            wgsl.matches(fetch).count(),
            2,
            "both staging routines must fetch the tile through the same address"
        );

        // One bound in the shader, and it on the staging the 8-bit entry alone uses:
        // the two K-quant entries call the unbounded `stage_x`, so nothing bounds
        // their staged set or their trip counts.
        assert_eq!(
            wgsl.matches("< params.k)").count(),
            1,
            "the staged fetch's bound is the one comparison against the contraction's extent"
        );
        assert_eq!(
            wgsl.matches("stage_x(tid, row0, p0, k4, xb, last_row);")
                .count(),
            2,
            "the two K-quant entries must stage through the unbounded routine"
        );
        assert_eq!(
            wgsl.matches("stage_x_bounded(tid, row0, b0, k4, xb, last_row);")
                .count(),
            1,
            "the 8-bit entry must stage through the bounded routine"
        );

        // The same two facts off the parsed module, where a text pin cannot reach:
        // the conditional is on the bounded routine's own staged store, and the
        // unbounded routine has none.
        assert!(
            staged_store_is_guarded(&module, "stage_x_bounded"),
            "the bounded routine must store the staged tile under its guard"
        );
        assert!(
            !staged_store_is_guarded(&module, "stage_x"),
            "the unbounded routine must stage through no conditional at all"
        );

        /// Whether `routine` carries the store into `xs` under a conditional, read
        /// off the emitted module rather than the text: a `Statement::If` (the loop's
        /// own included) above that store is a guard over it.
        fn staged_store_is_guarded(module: &naga::Module, routine: &str) -> bool {
            let (_, f) = module
                .functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(routine))
                .unwrap_or_else(|| panic!("no `{routine}` in the shader"));
            let (xs, _) = module
                .global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some("xs"))
                .expect("the staged tile `xs` in the shader");
            fn walk(
                exprs: &naga::Arena<naga::Expression>,
                xs: naga::Handle<naga::GlobalVariable>,
                stmts: &[naga::Statement],
                guarded: bool,
                found: &mut Option<bool>,
            ) {
                for s in stmts {
                    match s {
                        naga::Statement::If { accept, reject, .. } => {
                            walk(exprs, xs, accept, true, found);
                            walk(exprs, xs, reject, true, found);
                        }
                        naga::Statement::Block(b) => walk(exprs, xs, b, guarded, found),
                        naga::Statement::Loop {
                            body, continuing, ..
                        } => {
                            walk(exprs, xs, body, guarded, found);
                            walk(exprs, xs, continuing, guarded, found);
                        }
                        naga::Statement::Store { pointer, .. } => {
                            let naga::Expression::Access { base, .. } = &exprs[*pointer] else {
                                continue;
                            };
                            if matches!(&exprs[*base], naga::Expression::GlobalVariable(h) if *h == xs)
                            {
                                *found = Some(guarded);
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut found = None;
            walk(&f.expressions, xs, &f.body, false, &mut found);
            found.expect("the routine stores the staged tile")
        }
    }
}
