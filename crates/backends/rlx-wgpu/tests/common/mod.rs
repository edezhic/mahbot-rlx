// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixtures and measurement scaffolding shared by this crate's tests: the packed
//! GGUF weight, the activation ramp, the band module's own geometry helpers
//! ([`band_device`], [`band_rows`], [`band_key_block`]), and the median together
//! with the three timing loops — [`time_rounds`] (one cell alone),
//! [`time_round_robin`] (round-robin across a cell set) and [`paired_round_robin`]
//! (two arms with the order reversed each round) — plus the rate harnesses' shared
//! skeleton ([`RateCell`], [`RateSide`], [`rate_cell_fixture`], [`rate_build_side`] and
//! the two reporters, which the prefill and short-last-tile pairs are both built from),
//! the parity helpers, the conv-transpose-2d route levers
//! together with the pair bound its pins derive their bar from and the six real
//! decoder geometries those pins and the route harness both read, the bounded child
//! re-exec harness ([`rerun_self`]) that a case which has to run in its own process
//! waits on with a deadline, the switch set those children clear
//! ([`cleared_switches`]), the pins' own refusal and the route-counter lock at the end
//! of the file.
//! `tests/common/` is not a test target of its own; each test binary pulls it in with
//! `mod common;`.

// Each test binary declaring `mod common;` uses only the helpers it needs, so
// the rest are dead code in that binary and `just lint` runs with `-D warnings`.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::backend::ATTENTION_DECOMPOSE_MIN_Q;
use rlx_wgpu::gguf_gpu::{
    DequantMatmulRoutes, coop_tail_dispatches, gguf_row_bytes, grouped_chunk_geometry,
    grouped_dispatch_cols, reset_coop_tail_dispatches, reset_route_counts, route_counts,
};
use rlx_wgpu::gguf_host::gguf_scheme_id;
use rlx_wgpu::kernels::{AttentionRoutes, coop_gemm_gguf_grid};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// A deterministic `[m, k]` row-major activation, at `step`'s ramp rate. `step`
/// stays the caller's because it sets the result's magnitude, which a parity
/// tolerance has to be read against.
pub fn ramp(m: usize, k: usize, step: f32) -> Vec<f32> {
    (0..m * k).map(|i| ((i as f32) * step).sin()).collect()
}

/// The deterministic sweep the attention route tests feed their operands with: `n`
/// values of a `cos` (or `sin`) ramp at `step`'s rate.
pub fn data(n: usize, step: f32, cos: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 * step;
            if cos { x.cos() } else { x.sin() }
        })
        .collect()
}

/// The bar a dense-arrangement attention parity is held to against the CPU backend
/// at these shapes: a max-abs, never a relative error, so a diffuse softmax leaving
/// the output small cannot loosen it.
pub const ATTENTION_BAR: f32 = 1e-6;

/// The bar the cooperative-matrix packed GEMM tile is held to against the CPU
/// reference. Its operands are f16 — the activations are downcast as they land in
/// the workgroup stage and the packed weight is decoded to f16 — so it computes
/// the f32 product with both operands rounded to an 11-bit mantissa.
/// `matmul_f16_f32acc_parity` decomposes exactly that error for the tree's other
/// f16-operand kernel and holds the total against the true f32 product to
/// `TOTAL_BAR` = 5e-2; this is the same absolute bar, paired with
/// [`COOP_REL_BAR`], and every cell prints the band it was actually measured at.
pub const COOP_BAR: f32 = 5e-2;
/// ...and the relative floor it is also held to, which is the one that binds on a
/// cell whose output is not of order one. f16 operand rounding is a *relative*
/// effect, so an absolute bar alone would call a correct result wrong at a
/// magnitude the cell itself prints: measured over the cooperative cells the
/// relative error lands at 3e-4 .. 1.2e-3, so this is 4x the worst reading.
pub const COOP_REL_BAR: f32 = 5e-3;
/// The bar a packed multiply that did **not** take the cooperative tile is held to:
/// what every route in this crate used before that tile existed.
pub const PACKED_BAR: f32 = 1e-2;

/// The scale a bar is stated against: the largest magnitude in `reference`.
pub fn magnitude(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |a, x| a.max(x.abs()))
}

/// [`COOP_BAR`] floored by [`COOP_REL_BAR`] at a cell's own output scale.
pub fn coop_bar_at(reference_scale: f32) -> f32 {
    COOP_BAR.max(reference_scale * COOP_REL_BAR)
}

/// The bar a packed-multiply cell is held to, from the census of what ran: the
/// cooperative tile's two bars at `reference_scale` where that tile served the call,
/// [`PACKED_BAR`] on every other route.
pub fn packed_bar_at(took_coop: bool, reference_scale: f32) -> f32 {
    if took_coop {
        coop_bar_at(reference_scale)
    } else {
        PACKED_BAR
    }
}

/// The wgpu adapter this process runs on, for the figures that are the band
/// module's own: the workgroup width it is built with and the storage its key
/// block is chosen for. Every caller is behind a `skip_unless_device` prologue, so
/// a missing device here is a harness error rather than a skip.
pub fn band_device() -> &'static rlx_wgpu::device::WgpuDevice {
    rlx_wgpu::wgpu_device().expect("a wgpu adapter passed this file's device prologue")
}

/// Whether the band module this process builds stages in f16 — the same two terms
/// `kernels::attention_band_kernel` asks: the tile is f32 where the adapter lacks
/// `Features::SHADER_F16` *and* where `RLX_WGPU_ATTENTION_BAND_F32` forces the
/// twin, so the switch is part of the question rather than a separate one.
pub fn band_stages_f16() -> bool {
    !rlx_ir::env::flag("RLX_WGPU_ATTENTION_BAND_F32") && rlx_wgpu::shader_f16_supported()
}

/// The rows one workgroup of the band module this process builds runs — the `BM`
/// the kernel is declared with, and the block its dispatch grid is built from: the
/// adapter's own thread width over the head's chunk count, not the plan-side floor
/// figure.
pub fn band_rows(head_dim: u32) -> usize {
    let dev = band_device();
    rlx_wgpu::kernels::attention_band_rows_for(
        head_dim,
        rlx_wgpu::kernels::attention_band_threads(&dev.device),
    ) as usize
}

/// The key block the band module this process builds walks (`BK`): the largest one
/// the adapter's own workgroup storage holds at the width and staging that module
/// uses.
pub fn band_key_block(head_dim: u32) -> usize {
    let dev = band_device();
    rlx_wgpu::kernels::attention_band_key_block_within(
        head_dim,
        if band_stages_f16() { 2 } else { 4 },
        dev.device.limits().max_compute_workgroup_storage_size,
        rlx_wgpu::kernels::attention_band_threads(&dev.device),
    ) as usize
}

/// `v`'s smallest figure, for the rate harnesses' "best of" readings.
pub fn least(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MAX, f64::min)
}

/// `v`'s largest figure: `least`'s mirror, seeded so an all-negative arm reports its
/// own bound rather than zero.
pub fn greatest(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MIN, f64::max)
}

/// One rate harness's cell graph: `steps` identical packed multiplies over one
/// input matrix, each over its own parameter — the shape a prefill's own graph
/// runs them in rather than `steps` submissions of one, which is what makes a
/// call's steps ride one batch join.
pub fn rate_cell_graph(
    name: &str,
    scheme: QuantScheme,
    m: usize,
    k: usize,
    n: usize,
    packed_bytes: usize,
    steps: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut outs = Vec::with_capacity(steps);
    for s in 0..steps {
        let w = g.param(format!("w{s}"), Shape::new(&[packed_bytes], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[m, n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    g
}

/// The matrix every step of one rate call reads, at the ramp rate those harnesses
/// share.
pub fn rate_input_matrix(m: usize, k: usize) -> Vec<f32> {
    ramp(m, k, 0.017)
}

// ── the rate harnesses' shared skeleton ─────────────────────────────────────
//
// `gguf_gemm_prefill_rate.rs` and `gguf_gemm_coop_tail_rate.rs` measure one packed
// multiply's dispatch as a pair of sides over the same cells, and the build, the
// census and the printed fields are the same in both. What stays in each file is the
// lever its pair toggles, its cells, and what its ratio means.

/// One rate harness's cell: one packed multiply at one shape, run `steps` times over
/// one call. Shared so a cell means the same thing in every harness that reads one.
pub struct RateCell {
    pub label: &'static str,
    pub scheme: QuantScheme,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// One side of one rate cell: built, warmed, censused and asserted, held alive beside
/// the other side of its pair while that cell's rounds run.
pub struct RateSide {
    /// The entry this was built from: alive in the harness's `CELLS` for the whole run.
    pub cell: &'static RateCell,
    /// Which side of the pair this is, as the harness prints it and as its timed
    /// closure reads it to decide which lever to install.
    pub side: &'static str,
    /// The packed bytes one step reads, and the denominator of the printed
    /// `GBps_at_the_weight`: the fixture's own length.
    pub packed_bytes_per_step: usize,
    /// The dispatch this side's route makes: column tiles by row groups, from the
    /// module that dispatches it.
    pub cols: u32,
    pub tiles: u32,
    /// The short last row group the route's own geometry allows — `None` where it has
    /// no such form, so the row-grouped arm's line claims no tail it cannot have.
    pub tail: Option<bool>,
    /// Short last row groups the census call dispatched: one per step where this
    /// side's grid has one, zero otherwise.
    pub tail_dispatches: u64,
    /// The one route the call's steps took, as [`rate_single_route`] names it.
    pub route: &'static str,
    pub routes: DequantMatmulRoutes,
    pub submits: u64,
    pub waits: u64,
    pub steps: usize,
    pub x: Vec<f32>,
    pub compiled: CompiledGraph,
    /// This side's mean ms per call, one entry per round.
    pub per_call_ms: Vec<f64>,
}

/// One rate cell's fixture — the packed bytes a step reads, the weight itself and the
/// activation — after the two checks every such cell has to pass: the weight is one
/// column chunk of the encoder's own making, and the fixture's length is the encoder's
/// per-row stride times `n` (the fixture's own shape stated twice, not a check that the
/// stride has not moved).
///
/// `None`: the cell is **named as not one chunk and skipped** rather than measured with
/// a whole-tensor geometry the encoder does not dispatch — for both of its sides at
/// once, since the cap is a property of the shape and the device rather than of the
/// tile. That skip can silently empty a run, so a printed `pairs=0` is no evidence.
pub fn rate_cell_fixture(
    prefix: &str,
    cell: &RateCell,
    dev: &rlx_wgpu::device::WgpuDevice,
) -> Option<(Vec<u8>, Vec<f32>)> {
    let row_bytes = gguf_row_bytes(gguf_scheme_id(cell.scheme), cell.k);
    let packed_bytes_per_step = cell.n * row_bytes as usize;

    // [`grouped_dispatch_cols`] is the encoder's own column cap, both of its budgets at
    // once (the bound weight window and the whole-bound output slab), and the device
    // limit it reads is the one `RLX_WGPU_MAX_BIND_MB` lowers.
    let chunk_cols =
        grouped_dispatch_cols(&dev.device, gguf_scheme_id(cell.scheme), cell.m, cell.k);
    if chunk_cols < cell.n {
        eprintln!(
            "{prefix} cell=\"{}\" is not one chunk: the encoder covers {chunk_cols} of its {} \
             columns per dispatch — skipped rather than measured as a shape whose printed \
             geometry it is not",
            cell.label, cell.n
        );
        return None;
    }

    let packed = packed_gguf_weight(cell.scheme, cell.k, cell.n);
    assert_eq!(
        packed.len(),
        packed_bytes_per_step,
        "{}: the fixture must reproduce the cell's packed bytes",
        cell.label
    );
    let x = rate_input_matrix(cell.m, cell.k);
    Some((packed, x))
}

/// The one route a call's steps took, or a panic naming the counts. Which route it is
/// stays the encoder's answer — a harness reads it out rather than pinning it — but a
/// call whose steps *disagree* is a routing defect, and a figure averaged over two
/// routes is not a measurement of either.
pub fn rate_single_route(
    cell: &RateCell,
    side: &str,
    routes: &DequantMatmulRoutes,
    steps: usize,
) -> &'static str {
    let counts = [
        ("grouped", routes.grouped),
        ("coop", routes.coop),
        ("per_row", routes.per_row),
        ("q1_0_gemm", routes.q1_0_gemm),
        ("scratch", routes.scratch),
        ("host", routes.host),
    ];
    let taken: Vec<(&str, u64)> = counts.iter().copied().filter(|(_, n)| *n > 0).collect();
    assert!(
        taken.len() == 1 && taken[0].1 == steps as u64,
        "{} (side={side}): the {steps} steps of one call did not take one route: {taken:?} — a \
         call whose steps disagree is a routing defect, not a measurement",
        cell.label
    );
    taken[0].0
}

/// One side of one rate cell: compile, parameterise, warm, assert, census. The caller
/// installs its lever around this, so it covers the compile as well as the runs — the
/// encoder reads its levers where the multiply is encoded, not where the graph is
/// lowered, so nothing rests on where the decision is taken.
///
/// The grid comes from the module that dispatches it — the cooperative tile's own
/// [`coop_gemm_gguf_grid`] for a `coop` call, the row-grouped arm's own
/// [`grouped_chunk_geometry`] for a `grouped` one — never re-derived here, which is how
/// the two tiles' grids were once conflated. [`rate_single_route`] has established that
/// every step took this one route; a route with no such grid cannot be one of these
/// cells, and saying so is better than printing a geometry that did not run.
///
/// Returns the side and the warm-up's per-step outputs: a pair that compares the two
/// sides' answers holds them until it has.
pub fn rate_build_side(
    cell: &'static RateCell,
    side: &'static str,
    graph_name: &str,
    packed: &[u8],
    x: &[f32],
    steps: usize,
) -> (RateSide, Vec<Vec<f32>>) {
    let mut compiled = Session::new(Device::Gpu).compile(rate_cell_graph(
        graph_name,
        cell.scheme,
        cell.m,
        cell.k,
        cell.n,
        packed.len(),
        steps,
    ));
    for s in 0..steps {
        compiled.set_param_typed(&format!("w{s}"), packed, DType::U8);
    }

    let warm = compiled.run(&[("x", x)]);
    assert_eq!(warm.len(), steps, "{}: one output per step", cell.label);
    for (s, out) in warm.iter().enumerate() {
        assert_eq!(
            out.len(),
            cell.m * cell.n,
            "{} (side={side}): step {s} output shape",
            cell.label
        );
        let non_finite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "{} (side={side}): step {s} produced {non_finite} non-finite values — the fixture's \
             scale slots are not the finite patterns it writes",
            cell.label
        );
    }

    // The census: which route, how many short last row groups, how many submissions and
    // waits — one call's own counts, taken outside the timing, and taken while this side
    // is the only one running.
    reset_route_counts();
    reset_coop_tail_dispatches();
    reset_device_counts();
    let _ = compiled.run(&[("x", x)]);
    let routes = route_counts();
    let tail_dispatches = coop_tail_dispatches();
    let (submits, waits) = (device_submit_count(), device_wait_count());
    let route = rate_single_route(cell, side, &routes, steps);
    let (cols, tiles, tail) = match route {
        "coop" => {
            let g = coop_gemm_gguf_grid(cell.m as u32, cell.n as u32);
            (g.cols, g.tiles, Some(g.tail))
        }
        "grouped" => {
            let (cols, tiles, _) =
                grouped_chunk_geometry(gguf_scheme_id(cell.scheme), cell.m, cell.n);
            (cols, tiles, None)
        }
        other => panic!(
            "{} (side={side}): route {other} dispatches no single grid at these shapes — every \
             rate cell is a multi-row call over whole packing blocks, which the ladder answers \
             with one of the two fused tiles",
            cell.label
        ),
    };

    (
        RateSide {
            cell,
            side,
            packed_bytes_per_step: packed.len(),
            cols,
            tiles,
            tail,
            tail_dispatches,
            route,
            routes,
            submits,
            waits,
            steps,
            x: x.to_vec(),
            compiled,
            per_call_ms: Vec::new(),
        },
        warm,
    )
}

/// One rate cell's record header: the geometry its two sides share.
pub fn report_rate_cell(prefix: &str, r: &RateSide) {
    let cell = r.cell;
    eprintln!(
        "{prefix} cell=\"{}\" scheme={:?} m={} k={} n={} steps={} packed_bytes_per_step={}",
        cell.label, cell.scheme, cell.m, cell.k, cell.n, r.steps, r.packed_bytes_per_step,
    );
}

/// One side's printed fields: its grid and census, then its per-round figures and its
/// two rates. `TFLOPs` is `2 * m * k * n * steps / call_seconds`, the rate a prefill is
/// quoted at, and `GBps_at_the_weight` the tensor read once over the same call; neither
/// figure is a DRAM counter.
pub fn report_rate_side(prefix: &str, r: &RateSide) {
    let cell = r.cell;
    let workgroups = r.cols * r.tiles;
    let per_call_ms = &r.per_call_ms;
    let min_ms = least(per_call_ms);
    let med_ms = median(per_call_ms);
    let max_ms = greatest(per_call_ms);
    let flops = 2.0 * cell.m as f64 * cell.k as f64 * cell.n as f64 * r.steps as f64;
    let tflops = |call_ms: f64| flops / (call_ms * 1e-3) / 1e12;
    let gbps =
        |call_ms: f64| r.packed_bytes_per_step as f64 * r.steps as f64 / (call_ms * 1e-3) / 1e9;
    let (side, route, routes, submits, waits, tails) = (
        r.side,
        r.route,
        &r.routes,
        r.submits,
        r.waits,
        r.tail_dispatches,
    );

    let tail = match r.tail {
        Some(t) => t.to_string(),
        None => "none".to_string(),
    };
    eprintln!(
        "{prefix} cell=\"{}\" side={side} route={route} routes={routes:?} \
         grid={}x{}={workgroups} workgroups grid_tail={tail} tail_dispatches={tails} \
         submits={submits} waits={waits}",
        cell.label, r.cols, r.tiles,
    );
    let per_call = per_call_ms
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    let spread = (max_ms - min_ms) / med_ms * 100.0;
    eprintln!(
        "{prefix} cell=\"{}\" side={side} per_call_ms=[{per_call}] min_ms={min_ms:.3} \
         median_ms={med_ms:.3} max_ms={max_ms:.3} spread_pct={spread:.1} \
         min_TFLOPs={:.2} median_TFLOPs={:.2} min_GBps_at_the_weight={:.1} \
         median_GBps_at_the_weight={:.1}",
        cell.label,
        tflops(min_ms),
        tflops(med_ms),
        gbps(min_ms),
        gbps(med_ms),
    );
}

/// One cell's whole record as far as both rate harnesses share it: the cell's header, each
/// side's line, and the pair's two ratios in one line ending with `meaning`. `label` names
/// the pair in the printed fields (`median_full_over_tail`), and the ratios are
/// `first / second`, so above one means the second-named side was the faster of the two.
pub fn report_rate_pair(
    prefix: &str,
    first: &RateSide,
    second: &RateSide,
    label: &str,
    meaning: &str,
) {
    report_rate_cell(prefix, first);
    report_rate_side(prefix, first);
    report_rate_side(prefix, second);
    eprintln!(
        "{prefix} cell=\"{}\" pair= median_{label}={:.2}x min_{label}={:.2}x {meaning}",
        first.cell.label,
        median(&first.per_call_ms) / median(&second.per_call_ms),
        least(&first.per_call_ms) / least(&second.per_call_ms),
    );
}

/// `v`'s median: the element at `len / 2` of a sorted **copy**, so an even-length
/// series gives its upper-middle element rather than the mean of the pair, an empty
/// one panics, and `v` keeps the order the caller prints it in.
pub fn median(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    sorted[sorted.len() / 2]
}

/// Time `calls_per_round` back-to-back calls, `rounds` times over: each round's
/// mean per call, in milliseconds. `Tick`'s cycle counter is the clock, since a
/// call at these shapes is well under a millisecond on some cells.
pub fn time_rounds(rounds: usize, calls_per_round: usize, mut one_call: impl FnMut()) -> Vec<f64> {
    (0..rounds)
        .map(|_| {
            let start = rlx_runtime::Tick::now();
            for _ in 0..calls_per_round {
                one_call();
            }
            rlx_runtime::Tick::now().elapsed_ns(start) as f64 / calls_per_round as f64 / 1e6
        })
        .collect()
}

/// [`time_rounds`] for a cell set: each round gives every cell its own
/// `calls_per_round` calls back to back — **round-robin across the cells**, not one
/// cell's whole timing block at a time, so that the process's warm-up and the
/// machine's drift land on every cell alike rather than on whichever cells come
/// first. Each cell's own per-round figures, in milliseconds per call.
pub fn time_round_robin<T>(
    cells: &mut [T],
    rounds: usize,
    calls_per_round: usize,
    mut one_call: impl FnMut(&mut T),
) -> Vec<Vec<f64>> {
    let mut per_call_ms = vec![Vec::with_capacity(rounds); cells.len()];
    for _ in 0..rounds {
        for (i, cell) in cells.iter_mut().enumerate() {
            let start = rlx_runtime::Tick::now();
            for _ in 0..calls_per_round {
                one_call(cell);
            }
            per_call_ms[i].push(
                rlx_runtime::Tick::now().elapsed_ns(start) as f64 / calls_per_round as f64 / 1e6,
            );
        }
    }
    per_call_ms
}

/// [`time_round_robin`] with the order reversed each round, and the per-round paired
/// difference beside the two arms' figures: `rounds` rounds of `calls_per_round` calls
/// per arm, `first` running first on even rounds and second on odd ones, so a
/// within-round drift lands on both arms rather than on their difference. Returns each
/// arm's own per-round figures and `first − second` per round — the statistic a paired
/// cell reads as the median of.
pub fn paired_round_robin<T>(
    first: &mut T,
    second: &mut T,
    rounds: usize,
    calls_per_round: usize,
    mut one_call: impl FnMut(&mut T),
) -> ([Vec<f64>; 2], Vec<f64>) {
    let mut per_call = [Vec::with_capacity(rounds), Vec::with_capacity(rounds)];
    let mut deltas = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let mut this = [0.0f64; 2];
        for i in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let arm = if i == 0 { &mut *first } else { &mut *second };
            let start = rlx_runtime::Tick::now();
            for _ in 0..calls_per_round {
                one_call(arm);
            }
            this[i] =
                rlx_runtime::Tick::now().elapsed_ns(start) as f64 / calls_per_round as f64 / 1e6;
        }
        per_call[0].push(this[0]);
        per_call[1].push(this[1]);
        deltas.push(this[0] - this[1]);
    }
    (per_call, deltas)
}

/// Deterministic packed GGUF bytes for a `[k, n]` weight — a fixed f32 ramp
/// quantized into `ty`, the construction `gguf_batch_join`, `buffer_copy_batch_join`
/// and `run_report` all build their weights with. A fixture and not a value:
/// nothing reading it looks at the bytes.
pub fn packed_gguf_bytes(ty: rlx_gguf::GgmlType, k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w, ty).expect("quantize")
}

/// One packed GGUF weight of `[n, k]` at `quant`'s own block geometry, filled
/// deterministically.
///
/// The bytes are laid out by hand rather than quantized from an f32 tensor: the
/// f32 form is larger than the packed one at these shapes and is part of the arm's
/// own cost, while nothing in the measurement reads the byte values. Small on
/// purpose: the high halves of the f16 scale fields get a fixed exponent so `d`
/// and `dmin` land in [2^-13, 2^-12) and the dequantized weights stay ~1e-2 —
/// finite, and small enough that a grouped result can be compared with a per-row
/// one under the op's absolute 1e-2 tolerance rather than measuring the fixture's
/// own weight scale (large random weights cancel catastrophically, and the
/// round-off of that cancellation would be what such a bound caught).
pub fn packed_gguf_weight(quant: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> Vec<u8> {
    use rlx_ir::quant::QuantScheme;
    let (block_elems, block_bytes) = (
        quant.gguf_block_size() as usize,
        quant.gguf_block_bytes() as usize,
    );
    // Byte offsets within a block whose high half is an f16 exponent: Q4_K's `d`
    // and `dmin` at its head, Q6_K's `d` at its tail, Q8_0's `d` at its head.
    let f16_high: &[usize] = match quant {
        QuantScheme::GgufQ4K => &[1, 3],
        QuantScheme::GgufQ6K => &[209],
        QuantScheme::GgufQ8_0 => &[1],
        other => panic!("{other:?} has no row-grouped arm"),
    };
    let mut s = 7u32;
    (0..(k / block_elems) * block_bytes * n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 24) as u8;
            if f16_high.contains(&(i % block_bytes)) {
                // High half of an f16: sign 0, exponent 2, low mantissa bits from
                // the generator.
                0x08 | (b & 0x03)
            } else {
                b & 0x3F
            }
        })
        .collect()
}

/// Largest absolute difference between a computed and a reference output, after
/// a finiteness census.
///
/// The census runs FIRST and is asserted clean, because `f32::max` returns the
/// non-NaN operand: an all-NaN or infinity-poisoned output would otherwise fold
/// to a small number and pass any tolerance, so a broken kernel could satisfy a
/// parity bar. NaN, +inf and −inf are counted separately so an infinity cannot
/// hide behind a NaN.
pub fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "parity needs equal lengths");
    let (mut nan, mut pos_inf, mut neg_inf) = (0usize, 0usize, 0usize);
    for &v in got.iter().chain(want) {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos_inf += 1;
        } else if v == f32::NEG_INFINITY {
            neg_inf += 1;
        }
    }
    assert_eq!(
        (nan, pos_inf, neg_inf),
        (0, 0, 0),
        "non-finite in the parity pair: NaN={nan} +inf={pos_inf} -inf={neg_inf}"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Whole-vector bit-pattern equality — the tree's idiom for "the same work in a
/// different arrangement" (a joined vs an unjoined pass, a striped vs a whole
/// arena). `what` is the site's own claim, which names the two arrangements the
/// pair came from.
pub fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: {}-element output read as {}-element",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{what}: element {i} reads {x} against {y}"
        );
    }
}

/// A deterministic fill for the convolution-route harnesses' activations and
/// weights: `(i * mul + add) % modu`, scaled and biased. The two callers differ
/// only in the tabulation, so the shape of the fill is one definition.
pub fn fill(n: usize, mul: usize, add: usize, modu: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul + add) % modu) as f32 * scale + bias)
        .collect()
}

/// The conv the convolution-route harnesses measure — `conv_gemm_coop_f32.rs`,
/// `conv_routes_parked_weight.rs` and `conv1d_implicit.rs`: one unit-height 1-D
/// conv, input `[1, c_in, 1, l]`, weight `[c_out, c_in, 1, k]`, kernel `[1, k]`,
/// padding `[0, p]`, stride `[1, 1]`, dilation `[1, dilation]`, `groups = 1`,
/// output `[1, c_out, 1, l]`.
pub fn unit_conv_graph(
    name: &str,
    c_in: usize,
    c_out: usize,
    l: usize,
    k: usize,
    p: usize,
    dilation: usize,
) -> rlx_ir::Graph {
    let mut g = rlx_ir::Graph::new(name);
    let x = g.input(
        "x",
        rlx_ir::Shape::new(&[1, c_in, 1, l], rlx_ir::DType::F32),
    );
    let w = g.param(
        "w",
        rlx_ir::Shape::new(&[c_out, c_in, 1, k], rlx_ir::DType::F32),
    );
    let y = g.conv2d(x, w, [1, k], [1, 1], [0, p], [1, dilation], 1);
    g.set_outputs(vec![y]);
    g
}

/// That conv's output from the crate's own CPU conv rather than a hand-written
/// loop, so the reference and the device arm cannot share a misreading of the
/// layout. The weight param is `w`; the only input is `x`.
pub fn unit_conv_cpu_reference(g: rlx_ir::Graph, xs: &[f32], ws: &[f32]) -> Vec<f32> {
    let mut cpu = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    cpu.set_param("w", ws);
    cpu.run(&[("x", xs)]).remove(0)
}

/// `got` against `want` under the convolution routes' own `1e-5` budget, the bar
/// both harnesses hold every arm to; returns the difference it measured, so the
/// caller can print or compare it. `label` and `arm` name the case and which
/// compile it was. The budget is the claim, so the bar leaves room for a
/// different accumulation order — on a Metal adapter an arm may run one of these
/// convs on the f32 cooperative kernel (`conv_gemm`), whose order this tree does
/// not fix. The fold is [`max_abs_diff`], whose finiteness census runs first, so
/// a poisoned output cannot fold to a small number and pass.
pub fn assert_close_under(label: &str, arm: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {arm} length {} vs {}",
        got.len(),
        want.len()
    );
    let max = max_abs_diff(got, want);
    assert!(max < 1e-5, "{label}: {arm} max|Δ| {max:e} > 1e-5");
    max
}

/// The 8-bit GGUF scheme's bisect lever, spelled in one place: it takes that
/// scheme's multi-row multiplies off the grouped device arm, so the pins whose
/// routes, joined intervals or counted pairs it would move refuse on it.
pub const Q8_0_BISECT: &str = "RLX_WGPU_Q8_0_GEMM_DISABLE";

/// Refuse when the 8-bit scheme's bisect lever is set: the lever is read where that
/// call is encoded, so under it a call this tree routes on the grouped device arm
/// answers the slab instead — another routing mode, not a defect in the pin.
pub fn refuse_under_q8_0_bisect() {
    assert!(
        !rlx_ir::env::flag(Q8_0_BISECT),
        "RLX_WGPU_Q8_0_GEMM_DISABLE moves the 8-bit scheme's multiplies off the grouped route; \
         unset it to run this pin"
    );
}

/// The cooperative tile's bisect lever, spelled in one place: it puts every multi-row
/// multiply back on the row-grouped tile, so the pins whose routes or admitted shapes it
/// would move refuse on it.
pub const COOP_BISECT: &str = "RLX_WGPU_GEMM_COOP_DISABLE";

/// Refuse when the cooperative tile's bisect lever is set: the lever is read where the
/// call is encoded, so under it a shape this tree admits to the tile answers the
/// row-grouped one instead — another routing mode, not a defect in the pin.
pub fn refuse_under_coop_bisect() {
    assert!(
        !rlx_ir::env::flag(COOP_BISECT),
        "RLX_WGPU_GEMM_COOP_DISABLE puts every multi-row multiply back on the row-grouped \
         tile; unset it to run this pin"
    );
}

/// How many schedule steps are the materialised broadcast family
/// (`expand` / `expand_host`) — the lowering leaves one wherever a broadcast
/// operand has to arrive as a real tensor.
pub fn expand_steps(names: &[&str]) -> usize {
    names
        .iter()
        .filter(|n| **n == "expand" || **n == "expand_host")
        .count()
}

/// The step names this backend dispatches for `g` as the runtime path reaches it:
/// the legalising pass first (what the shim runs ahead of fusion —
/// `rlx-runtime/src/backend/wgpu_backend.rs`), then the lowering.
///
/// Legalising first is what makes a *missing* `expand` step evidence that the
/// peel fired, rather than evidence that the pass never ran and inserted none.
pub fn legalised_step_names(g: &rlx_ir::Graph) -> Vec<&'static str> {
    legalised_executable(g).test_step_names()
}

/// [`legalised_step_names`]'s compile, for a test that reads another of the
/// executable's own reports.
pub fn legalised_executable(g: &rlx_ir::Graph) -> rlx_wgpu::backend::WgpuExecutable {
    use rlx_opt::pass::Pass as _;
    let legalised = rlx_opt::LegalizeBroadcast.run(g.clone());
    rlx_wgpu::backend::WgpuExecutable::compile(legalised)
}

/// The run report's switch, spelled in one place: a pin whose figures the report's
/// own fence would move names it by this, and this is what the pins refuse on.
pub const RUN_REPORT_SWITCH: &str = "RLX_WGPU_DBG_COUNTERS";

/// Refuse when the run report's switch is set: a reporting run adds a fence, a
/// resolve submit and a map of its own, so the submission/wait counts and the
/// rates a pin measures under it are not that pin's.
pub fn refuse_under_run_report() {
    assert!(
        !rlx_ir::env::flag(RUN_REPORT_SWITCH),
        "RLX_WGPU_DBG_COUNTERS changes the submission/wait shape of a forward pass; \
         unset it to run this pin"
    );
}

/// The switches a re-exec'd child must **inherit** rather than clear: the device layer
/// decides which device it runs on, and `RLX_REQUIRE_DEVICE` decides whether a missing
/// one is a failure. Clearing either would let a case that asked for the GPU quietly
/// answer on the CPU and pass. The device layer is read from the registry rather than
/// listed, so a device-layer switch added upstream is inherited without this file being
/// touched; `RLX_REQUIRE_DEVICE` is registered in a group of its own and is the one name
/// here that has to be spelled.
fn inherited(entry: &rlx_ir::env_registry::EnvVarEntry) -> bool {
    matches!(entry.layer, rlx_ir::env_registry::EnvLayer::Device)
        || entry.name == "RLX_REQUIRE_DEVICE"
}

/// Every registered switch except [`inherited`] — which of a fixture's steps runs
/// where, whether a step is a device step at all, whether a packed multiply joins the
/// batch, which fusion is expanded at the lowering, whether a masked key's reads are
/// stood down, how the arena is planned, which dispatch tuning row is read, and whether
/// the run reports its own figures — cleared on every re-exec'd child before the case's
/// own switches are applied, so what a child reads is the case's environment and never
/// the operator's. A case that is *about* one of them sets it again by name.
///
/// Read from [`rlx_ir::env_registry`] rather than kept as a list: every switch this
/// crate reads is registered there (`just check-rlx-env-vars` enforces it), so a route
/// that grows one is cleared without this file being touched. Debug and dump switches
/// are cleared with the rest, which costs nothing and cannot pin a child to an
/// operator's tracing. Spelled once for every harness here that re-execs a child
/// (`run_report.rs`, `hosted_audio_step_placement.rs`, `gather_packed_parity.rs`,
/// `kv_append_in_place.rs`, `norm_rope_fusion.rs`). Pins that run **in-process** do
/// not clear anything — a child's environment is not theirs to rewrite — so they
/// refuse over [`guarded_switches`] and redirect the one switch whose unset default
/// is a real file ([`pin_tuning_cache_off`]).
pub fn cleared_switches() -> Vec<&'static str> {
    rlx_ir::env_registry::REGISTRY
        .iter()
        .filter(|entry| !inherited(entry))
        .flat_map(|entry| std::iter::once(entry.name).chain(entry.aliases.iter().copied()))
        .collect()
}

/// Switches in this backend's own group that only report: a dump, a trace, a log, a
/// schedule print, a plan report. They cannot move what a pin reads, so a pin that
/// refuses under an operator's environment does not refuse under these.
///
/// The `DUMP_` family is a prefix; the rest are named. Nothing with a `DBG_` or
/// `DEBUG_` prefix is excluded by rule — `RLX_WGPU_DBG_COUNTERS` is the run report,
/// which a pin *does* refuse under, and the rest are read one by one.
const REPORTING: &[&str] = &[
    "RLX_WGPU_ACTIVE_TRACE",
    "RLX_WGPU_ARENA_REPORT",
    "RLX_WGPU_COOP_PROBE_VERBOSE",
    "RLX_WGPU_DEBUG_OP_HIST",
    "RLX_WGPU_DBG_HOST_OP",
    "RLX_WGPU_DBG_TRANSPOSE",
    "RLX_WGPU_FEED_TRACE",
    "RLX_WGPU_HAZARD_REPORT",
    "RLX_WGPU_PRINT_LIMITS",
    "RLX_WGPU_QUIET_SHARD",
    "RLX_WGPU_ROW_TABS",
    "RLX_WGPU_SCHEDULE",
    "RLX_WGPU_SHARD_LOG",
    "RLX_WGPU_SNAPSHOT_WATCH",
    "RLX_WGPU_STRIPE_VERBOSE",
    "RLX_WGPU_TEE_VETO_LOG",
];

/// Switches outside this backend's group that still move a schedule: the fusion
/// passes, the arena plan, whether a packed multiply joins the batch, whether a
/// nan scan runs, and what a submission means. Kept short because it is the only
/// part of [`guarded_switches`] that is written down rather than derived.
///
/// `RLX_GPU_TUNING_CACHE` is deliberately absent: refusing under it only catches
/// an operator who set it explicitly, while the ambient default
/// (`~/.cache/rlx-wgpu/dispatch-tuning.tsv`) is read either way. A pin is safe
/// only once the path is pointed somewhere that cannot exist, which is
/// [`pin_tuning_cache_off`]'s job.
const CROSS_CUTTING: &[&str] = &[
    "RLX_ARENA_NO_REUSE",
    "RLX_BENCH_DISPATCH_ONLY",
    "RLX_DEBUG_NANS",
    "RLX_DISPATCH_REPORT",
    "RLX_PIN_OUTPUT_ANCESTORS",
];

/// The dump family's prefix: a family of switches, not one this file reads.
const DUMP_PREFIX: &str = "RLX_WGPU_DUMP_"; // env-lint: ignore

/// The switches a pin that runs **in-process** refuses under, rather than clearing
/// them the way [`rerun_self`] clears a child's: everything this backend registers
/// (`group = "wgpu"`, which is where a route switch lands), plus [`CROSS_CUTTING`],
/// minus [`REPORTING`] and minus the pin's own `own`.
///
/// Derived from the registry so a route that adds a switch is guarded without this
/// file being touched. The exclusions are the hand-kept part, and they fail in the
/// safe direction: a reporting switch left out makes a pin refuse something benign,
/// where a routing switch left out would let it run under a switch that moved its
/// reading.
pub fn guarded_switches(own: &[&str]) -> Vec<&'static str> {
    let derived = rlx_ir::env_registry::REGISTRY
        .iter()
        .filter(|entry| entry.group == "wgpu")
        .flat_map(|entry| std::iter::once(entry.name).chain(entry.aliases.iter().copied()))
        .filter(|name| {
            !own.contains(name) && !REPORTING.contains(name) && !name.starts_with(DUMP_PREFIX)
        });
    derived.chain(CROSS_CUTTING.iter().copied()).collect()
}

/// Point the persisted dispatch tuning table at a name that cannot exist, so an
/// in-process pin reads the shipped rows and the compile-time cascade rather than
/// an operator's overrides. [`rerun_self`] does the same for a child through the
/// environment; a pin that runs in this process has to do it itself.
///
/// The pin then *performs* the read rather than assuming it got there first:
/// [`rlx_wgpu::tuning::ensure_tuning_cache_loaded`] runs at most once and blocks
/// while another thread is inside it, so this either loads under the redirect or
/// waits for a sibling's load and sees its final count. What it asserts is the
/// property that matters — that nothing of the operator's was applied
/// ([`rlx_wgpu::tuning::tuning_overrides_applied`]) — which a machine with no cache
/// file satisfies however the cases interleave.
pub fn pin_tuning_cache_off(tag: &str) {
    // Read before the redirect: this is the file the process would have loaded, and
    // it is what the failure message has to name.
    let operator = rlx_wgpu::tuning::tuning_cache_path();
    rlx_ir::env::set(
        TUNING_CACHE_SWITCH,
        absent_tuning_cache(tag).to_string_lossy(),
    );
    rlx_wgpu::tuning::ensure_tuning_cache_loaded();
    assert_eq!(
        rlx_wgpu::tuning::tuning_overrides_applied(),
        0,
        "pin_tuning_cache_off({tag:?}): this process already applied dispatch tuning \
         overrides from {}, so the pin cannot protect its readings — move that file \
         aside and re-run",
        operator.map_or_else(|| "<no path>".to_string(), |p| p.display().to_string())
    );
}

/// The arrangement every layer of a many-layer attention chain is expected to
/// take: a single query row is the decode-shaped form whatever its key count
/// and whatever the caller's harness geometry, and from the route's query-row
/// floor up it takes the dense arrangement (below that floor, and for a head
/// wider than the decode form holds, the fused/serial kernel). `layers`
/// attention steps per run, over `reps` runs, so the counters it predicts are
/// the whole bracket.
///
/// The shape decides which arrangement the route is *asked for*; the arm's
/// window term (`attention_window_declines`, whose count tells that decline
/// apart from the shape's own answer) can still keep the fused kernel, and a
/// harness on an adapter where it does fails this expectation loudly rather
/// than skipping. This is the route's one spelling: the decode term is
/// [`rlx_wgpu::kernels::attention_decode_eligible`] itself, called with the
/// harness's own extents, so a caller cannot re-derive that half and pass a bare
/// `true` after the geometry outgrew the cap.
pub fn expected_attention_routes(
    rows: usize,
    lk: usize,
    head_dim: usize,
    v_head_dim: usize,
    layers: usize,
    reps: usize,
) -> AttentionRoutes {
    let per_run = if rlx_wgpu::kernels::attention_decode_eligible(
        rows as u32,
        lk as u32,
        head_dim as u32,
        v_head_dim as u32,
    ) {
        AttentionRoutes {
            decode: layers as u64,
            decomposed: 0,
            serial: 0,
            band: 0,
        }
    } else if rows >= ATTENTION_DECOMPOSE_MIN_Q as usize {
        AttentionRoutes {
            decode: 0,
            decomposed: layers as u64,
            serial: 0,
            band: 0,
        }
    } else {
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: layers as u64,
            band: 0,
        }
    };
    AttentionRoutes {
        decode: per_run.decode * reps as u64,
        decomposed: per_run.decomposed * reps as u64,
        serial: per_run.serial * reps as u64,
        band: 0,
    }
}

/// The deterministic signal a framing is read over.
pub fn framing_signal(t: usize) -> Vec<f32> {
    (0..t).map(|i| (i as f32 * 0.017).sin() * 0.5).collect()
}

/// The starts a uniform hop implies: `f · hop`.
pub fn framing_starts(n_frames: usize, hop: usize) -> Vec<usize> {
    (0..n_frames).map(|f| f * hop).collect()
}

/// The frame matrix in plain Rust: `out[f, o, j] = sig[o·t + starts[f] + j]` for a
/// source of `b` rows of `t`. What a correct framing is, independent of every
/// backend — the rows are copies, so this is what they must be bit for bit. A
/// complex source is read in lane units, where `t` is the source's lane count and
/// a frame is `2 · frame_len` of them.
pub fn framing_reference(
    sig: &[f32],
    b: usize,
    t: usize,
    starts: &[usize],
    frame_len: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; starts.len() * b * frame_len];
    for (f, &s) in starts.iter().enumerate() {
        for o in 0..b {
            for j in 0..frame_len {
                out[(f * b + o) * frame_len + j] = sig[o * t + s + j];
            }
        }
    }
    out
}

/// One `narrow_(x, axis, start, frame_len)` per start, each reshaped to
/// `row_shape`, stacked by one `concat_(rows, 0)` — the framing
/// `detect_frames_pattern` recognises, built row by row.
pub fn framing(
    g: &mut rlx_ir::Graph,
    x: rlx_ir::NodeId,
    axis: usize,
    starts: &[usize],
    frame_len: usize,
    row_shape: &[i64],
) -> rlx_ir::NodeId {
    use rlx_ir::GraphExt as _;
    let mut rows = Vec::with_capacity(starts.len());
    for &s in starts {
        let frame = g.narrow_(x, axis, s, frame_len);
        rows.push(g.reshape_(frame, row_shape.to_vec()));
    }
    g.concat_(rows, 0)
}

/// `[T] → [n_frames, frame_len]` over the graph's own input: one `narrow_` +
/// `reshape_` per frame, stacked by one `concat_`. This is the framing
/// `rlx_wgpu`'s lowering recognises (`detect_frames_pattern`), and the shape a
/// front-end that slices one frame at a time builds for itself.
pub fn framed_over_input(n_frames: usize, frame_len: usize, hop: usize) -> rlx_ir::Graph {
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = rlx_ir::Graph::new("frames_per_frame");
    let x = g.input("x", rlx_ir::Shape::new(&[t], rlx_ir::DType::F32));
    let y = framing(
        &mut g,
        x,
        0,
        &framing_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    g.set_outputs(vec![y]);
    g
}

/// The signal length one such framing needs: `frame_len + (n_frames - 1) · hop`.
pub const fn frames_signal_len(n_frames: usize, frame_len: usize, hop: usize) -> usize {
    frame_len + (n_frames - 1) * hop
}

/// The slots the compiler's plan gave these nodes — `(byte offset, byte size)` each,
/// `None` where a node has none — read from a compiled executable, which is the
/// artifact the lowering's placement clauses were decided against. Nothing here
/// re-runs the compiler's pass sequence, so a premise cannot describe a plan other
/// than the one the refusal was taken on. The offsets are raw: a weight parked in
/// the weight buffer shows up through `buffer::is_weight_off`.
pub fn planned_slots(g: &rlx_ir::Graph, ids: &[rlx_ir::NodeId]) -> Vec<Option<(usize, usize)>> {
    let exe = rlx_wgpu::backend::WgpuExecutable::compile(g.clone());
    ids.iter().map(|&id| exe.test_slot_bytes(id)).collect()
}

/// The knobs that move a graph off this backend's device route — the structural
/// ones (`wgpu_prefer_structure_host`, the shard striping) and the levers for the
/// ops whose routes they decide — set to their "device, unsharded, params
/// unparked" values: `RLX_WGPU_FORCE_HOST`, `RLX_WGPU_DISCRETE_HOST`,
/// `RLX_WGPU_ALL_PARAMS_WEIGHT` and `RLX_WGPU_ALLOW_SHARD` zeroed, the shard-cap
/// lever neutralised to a value the adapter clamps back to its own limit, and
/// `RLX_WGPU_SHARD_GPU=1` — which turns the whole structure-host preference off,
/// because on a Vulkan/DX12 adapter its discrete term would otherwise host any op
/// on any adapter kind and no other lever clears it.
///
/// A lever that belongs to *one* op's arm (`RLX_WGPU_CONCAT_HOST` for concat,
/// `RLX_WGPU_CONV_HOST` for conv-transpose) is deliberately not in this set: the
/// pins of that op add their own, so a developer bisecting another route keeps
/// that lever. A pin that neutralises one variable is green in a default
/// environment and red under a knob the crate documents, which is not a pin;
/// `buffer_copy_batch_join.rs` guards the same set by assertion, the conv-transpose
/// pins spell their own superset where a lever is theirs
/// (`conv_route_cost.rs`, `conv_routes_parked_weight.rs`), and this is what the
/// pins that read a device route use instead of an operator's environment.
pub fn neutralise_device_route_env() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
}

/// One conv-transpose-2d geometry: the operand shapes and the attribute set the bound
/// walk below reads. The harnesses that derive a bar from their operands keep their
/// case tables in this shape rather than in eleven loose numbers, so the walk takes it
/// and there is one vocabulary for "which geometry" across them. `Copy` because the
/// real forms below are a `const` table read out more than once.
#[derive(Clone, Copy)]
pub struct ConvTranspose2dGeometry {
    pub n: usize,
    pub c_in: usize,
    /// The weight's second dim — the output carries `c_out · groups` channels.
    pub c_out: usize,
    pub h: usize,
    pub w: usize,
    pub k: [usize; 2],
    pub s: [usize; 2],
    pub p: [usize; 2],
    pub d: [usize; 2],
    pub op: [usize; 2],
    pub groups: usize,
}

impl ConvTranspose2dGeometry {
    /// The output's `(h, w)`, from the tree's own shape rule rather than a locally
    /// restated formula.
    pub fn out(&self) -> (usize, usize) {
        let axis = |size: usize, k: usize, s: usize, p: usize, d: usize, op: usize| {
            rlx_ir::shape::conv_transpose2d_spatial_output(size, k, s, p, d, op)
        };
        (
            axis(
                self.h, self.k[0], self.s[0], self.p[0], self.d[0], self.op[0],
            ),
            axis(
                self.w, self.k[1], self.s[1], self.p[1], self.d[1], self.op[1],
            ),
        )
    }

    /// The output's channel count, `c_out · groups`.
    pub fn c_out_total(&self) -> usize {
        self.c_out * self.groups
    }
}

/// One of the stand's `dac:25` decoder-cell transposed convolutions: `(label,
/// geometry)`, single batch, unit height, one group, no dilation, and the width
/// padding / output padding the decoder's generator derives from the taps and
/// stride (`padding = (k − s + 1)/2`, `output_padding = (k − s) % 2`) — derived
/// here once rather than restated per caller, because these six shapes are the
/// operation's own acceptance and every copy of them is a shape that can drift.
pub const fn decoder_form(
    label: &'static str,
    c_in: usize,
    c_out: usize,
    w_in: usize,
    k: usize,
    s: usize,
) -> (&'static str, ConvTranspose2dGeometry) {
    (
        label,
        ConvTranspose2dGeometry {
            n: 1,
            c_in,
            c_out,
            h: 1,
            w: w_in,
            k: [1, k],
            s: [1, s],
            p: [0, (k - s).div_ceil(2)],
            d: [1, 1],
            op: [0, (k - s) % 2],
            groups: 1,
        },
    )
}

/// The six real forms, in graph order (`up.blk.0`, `up.blk.1`, `blk.0`…`blk.3`),
/// with the channel counts and input widths the decoder really carries. The
/// cross-crate copy of these numbers is
/// `crates/core/rlx-runtime/tests/conv_transpose2d_parity.rs`'s `decoder_cfgs()`,
/// which is where a change to them has to land too.
pub const DECODER_FORMS: &[(&str, ConvTranspose2dGeometry)] = &[
    decoder_form("up.blk.0", 1024, 1024, 25, 2, 2),
    decoder_form("up.blk.1", 1024, 1024, 50, 2, 2),
    decoder_form("blk.0", 1536, 768, 100, 16, 8),
    decoder_form("blk.1", 768, 384, 800, 10, 5),
    decoder_form("blk.2", 384, 192, 4000, 8, 4),
    decoder_form("blk.3", 192, 96, 16000, 6, 3),
];

/// The bound on how far `Op::ConvTranspose2d`'s device steps can sit from the scalar
/// oracle `rlx_cpu::kernels::conv_transpose2d_nchw`, derived from `geo`'s operands
/// rather than from an absolute bar. Each route accumulates the same `terms` f32
/// products per output element in its own order, so each can differ from the exact sum
/// by no more than `γ_n · Σ|terms|` (`γ_n = n·u / (1 − n·u)`, `u = 2^-24` the f32 unit
/// roundoff) and the two can differ from each other by at most twice that. Returns
/// `(bound, worst terms, worst Σ|terms|)`.
///
/// The tap walk is input-centric and mirrors the oracle's own, so the `terms` it counts
/// are the products the reference itself would have summed — every tap is counted,
/// including the ones a zero operand makes vanish, which only loosens the bound and
/// never the assertion. Padding, dilation, groups and output padding are all in it, so
/// the decoder's unit-height geometry (one tap on the height axis, `p = 0`) is just a
/// case of the same walk rather than a second derivation of it.
///
/// The grouping is the one thing it does not model, and it checks for it: `groups` must
/// be positive and divide `c_in`, because the degenerate groupings are refusals on both
/// routes rather than value cases, and the walk divides by `c_in / groups`.
pub fn conv_transpose2d_pair_error_bound(
    geo: &ConvTranspose2dGeometry,
    x: &[f32],
    w: &[f32],
) -> (f64, usize, f64) {
    /// The f32 unit roundoff, `2^-24`.
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    // The walk groups channels on `c_in / groups`, so a grouping that does not
    // partition `c_in` is outside what it covers — and both non-divisible groupings
    // are refusals on either route rather than value cases, so there is no bound to
    // derive for them. State the assumption instead of dividing by a zero.
    assert!(
        geo.groups >= 1 && geo.c_in.is_multiple_of(geo.groups),
        "conv_transpose2d_pair_error_bound: c_in={} groups={} is a degenerate grouping, which \
         the device route refuses and this walk does not cover",
        geo.c_in,
        geo.groups
    );
    let (h_out, w_out) = geo.out();
    let c_out_total = geo.c_out_total();
    let c_in_per_g = geo.c_in / geo.groups;
    let mut abs_sum = vec![0f64; geo.n * c_out_total * h_out * w_out];
    let mut terms = vec![0usize; abs_sum.len()];
    for ni in 0..geo.n {
        for ic in 0..geo.c_in {
            let g = ic / c_in_per_g;
            for iy in 0..geo.h {
                for ix in 0..geo.w {
                    let v = x[((ni * geo.c_in + ic) * geo.h + iy) * geo.w + ix] as f64;
                    for ky in 0..geo.k[0] {
                        // `oy` is the padded output coordinate the oracle's own `oy`
                        // lives in; a tap outside `[p, p + out)` lands outside the
                        // output and the oracle skips exactly the same ones.
                        let oy = iy * geo.s[0] + ky * geo.d[0];
                        if oy < geo.p[0] || oy - geo.p[0] >= h_out {
                            continue;
                        }
                        let oy = oy - geo.p[0];
                        for kx in 0..geo.k[1] {
                            let ox = ix * geo.s[1] + kx * geo.d[1];
                            if ox < geo.p[1] || ox - geo.p[1] >= w_out {
                                continue;
                            }
                            let ox = ox - geo.p[1];
                            for oc_off in 0..geo.c_out {
                                let oc = g * geo.c_out + oc_off;
                                let wt = w
                                    [((ic * geo.c_out + oc_off) * geo.k[0] + ky) * geo.k[1] + kx]
                                    as f64;
                                let at = ((ni * c_out_total + oc) * h_out + oy) * w_out + ox;
                                abs_sum[at] += (v * wt).abs();
                                terms[at] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut worst = (0f64, 0usize, 0f64);
    for i in 0..abs_sum.len() {
        let nu = terms[i] as f64 * U;
        // Two accumulation orders of the same sum, so twice one order's bound.
        let bound = 2.0 * nu / (1.0 - nu) * abs_sum[i];
        if bound > worst.0 {
            worst = (bound, terms[i], abs_sum[i]);
        }
    }
    worst
}

/// The route counters are process-global, so tests in one binary that bracket
/// one (`reset` … `run` … `read`) must not run next to each other.
static ROUTE_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// [`ROUTE_TEST_MUTEX`], poisoned-tolerant: an earlier failing assertion must not
/// turn every later test in the binary into a lock error.
pub fn route_lock() -> MutexGuard<'static, ()> {
    ROUTE_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long a fixture child may run before it is killed and its case reported. The
/// crate's re-exec harnesses used to wait on `Command::output()` with no bound, so a
/// regression that makes a run spin hung the gate instead of failing it.
pub const CHILD_BUDGET: std::time::Duration = std::time::Duration::from_secs(120);

/// The one cleared switch whose *unset* default is a real file on this machine:
/// removing it does not hand the child "no tuning", it hands it
/// `$XDG_CACHE_HOME/rlx-wgpu/dispatch-tuning.tsv` — whatever the operator last
/// measured — and a route pin then reads an arrangement it never asked for. Pointed at
/// a name that cannot exist instead, so the child loads no overrides at all.
const TUNING_CACHE_SWITCH: &str = "RLX_GPU_TUNING_CACHE";

/// The value that switch takes when the table is to be read from nowhere: a name
/// under the temp directory that no writer creates. Shared by [`rerun_self`]'s
/// child and [`pin_tuning_cache_off`]'s in-process pin so the two cannot drift
/// into redirecting the table in two different ways.
fn absent_tuning_cache(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rlx-wgpu-{tag}-no-tuning-cache.tsv"))
}

/// Run this test binary again as a child running the test `filter`, for the fixture
/// `fixture_var` = `fixture`, `env` set and `cleared` removed, and wait for it **with a
/// deadline** ([`CHILD_BUDGET`]) — a child that outlives it is killed and the case panics
/// with the stderr it managed to print, so a run that cannot advance reports instead of
/// spinning. Its stderr is captured through a file in the temp directory rather than a
/// pipe, so the child cannot deadlock on a full pipe. Returns `(exited cleanly, stderr)`.
///
/// [`TUNING_CACHE_SWITCH`] is redirected rather than merely removed; everything else in
/// `cleared` is removed, and `env` is applied last so a case that is *about* a switch
/// still wins.
///
/// The child prints `"{fixture_var} {fixture}"` as soon as it takes its fixture branch (see
/// each harness's own child path), and this waits for that line in the captured stderr: a
/// child whose `filter` matches no test exits 0 having run nothing, which must report
/// rather than pass as a case that pinned something.
pub fn rerun_self(
    filter: &str,
    fixture_var: &str,
    fixture: &str,
    cleared: &[&str],
    env: &[(&str, &str)],
) -> (bool, String) {
    let path = std::env::temp_dir().join(format!(
        "rlx-wgpu-child-{}-{fixture}.stderr",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("child stderr file");
    let mut cmd = std::process::Command::new(std::env::current_exe().expect("current exe"));
    cmd.args(["--exact", filter, "--nocapture"])
        .env(fixture_var, fixture)
        .stdout(std::process::Stdio::null())
        .stderr(file);
    for key in cleared {
        cmd.env_remove(key);
    }
    cmd.env(
        TUNING_CACHE_SWITCH,
        absent_tuning_cache(&format!("child-{}", std::process::id())),
    );
    for (key, value) in env {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn().expect("re-run this test in a child process");
    let deadline = std::time::Instant::now() + CHILD_BUDGET;
    let ok = loop {
        match child.try_wait().expect("wait on the child") {
            Some(status) => break status.success(),
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                let stderr = std::fs::read_to_string(&path).unwrap_or_default();
                // The file is left behind on purpose: a child that could not finish is
                // the one whose output is worth keeping, and the temp directory is the
                // right place for it.
                panic!(
                    "the `{fixture}` child did not finish within {}s — a run that cannot \
                     advance must report, not spin; its stderr so far:\n{stderr}",
                    CHILD_BUDGET.as_secs()
                );
            }
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    };
    let stderr = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    assert!(
        stderr.contains(&format!("{fixture_var} {fixture}")),
        "the `{fixture}` child never reached its fixture — it ran no test of `{filter}`:\n{stderr}"
    );
    (ok, stderr)
}
