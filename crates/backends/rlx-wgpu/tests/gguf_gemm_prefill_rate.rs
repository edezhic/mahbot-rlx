// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The packed-weight multiply at the row counts the mahbot product's **prompt
//! prefill** actually runs, and at the shapes its four body projections and two
//! encoder projections use — measured as a **pair**, the two tiles the encoder
//! chooses between, in one window: what one step reads, the dispatch each side's
//! route takes, what a call costs, the two rates that cost is quoted at, and the
//! pair's own ratio.
//!
//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test gguf_gemm_prefill_rate -- --ignored --nocapture
//! ```
//!
//! # The pair, and why it is one window
//!
//! Every cell is built **twice**, from the same graph and the same packed fixture,
//! and both builds are held alive for the whole run and timed together:
//!
//! * `side=grouped` — compiled and run with `RLX_WGPU_GEMM_COOP_DISABLE=1`
//!   installed, the bisect lever that puts every call back on the row-grouped tile
//!   (`dequant_gemm_gguf.wgsl`: 8-row tiles, f32 scalars);
//! * `side=taken` — compiled and run with no lever, so the encoder answers: the
//!   cooperative-matrix tile (`dequant_gemm_coop_gguf.wgsl`: 32 rows x 128 columns,
//!   the weight decoded into a workgroup f16 stage and the arithmetic on the
//!   hardware matrix units) where `gguf_gpu::coop_gemm_admits` admits the shape, the
//!   row-grouped one otherwise.
//!
//! A cell's two sides are timed in **adjacent rounds of the same process**, not in
//! two runs with the lever toggled between them. This box's per-round spread on
//! these cells runs to 95 % on the shortest rung, so a run-to-run difference of the
//! size the pair is here to report — the cooperative tile is 1.26-1.67x the
//! row-grouped one on these cells, `dequant_gemm_coop_gguf.wgsl`'s header table — is
//! not resolvable across two runs at all, while within one window it is a ratio of
//! two figures taken under the same load, in the same round-robin, on the same graph.
//!
//! The lever is **read per call, by the encoder, at run time**: `coop_gemm_admits`
//! consults it where the multiply is encoded, not where the graph is lowered. So it
//! is installed around every run of each side — its warm-up, its census call and each
//! timed call — and around its compile as well, so nothing here rests on the decision
//! staying at run time. Both sides install it, `1` on the `grouped` side and `0` on
//! the `taken` one, so neither pays for a guard the other does not and an ambient
//! `RLX_WGPU_GEMM_COOP_DISABLE` in the measuring shell cannot turn the `taken` side
//! into a second `grouped` one. `RuntimeOverrides::install` restores the environment
//! when it drops. The census is what makes the lever's effect a fact rather than an
//! assumption: the `grouped` side asserts `grouped == STEPS` and `coop == 0`.
//!
//! # What the ratio means, and what it does not
//!
//! The pair line prints the cell's `median` and its `min`, grouped over taken. Where
//! the `taken` side really did take the cooperative tile (`route=coop` on that side)
//! the ratio is the two tiles' rate on that shape, in one window, which is the number
//! the pair exists for.
//!
//! Where it did not, the line says so in words and the ratio is a **repeatability
//! witness only**. The census cannot say which of three reasons applies — the lever
//! is off, or the adapter has no cooperative matrices, or the shape is refused by
//! `coop_gemm_admits`, and all three leave a `grouped` call and nothing else — so all
//! three are named as the possibilities rather than one guessed. Two builds of the
//! same graph on the same tile then differ by whatever this window's noise is, which
//! is worth reading as the file's own floor.
//!
//! Neither figure is a claim about the rest of a prefill's graph, and no time is
//! asserted anywhere.
//!
//! # Where the shapes come from
//!
//! A recognition pass prefills one body graph over its prompt, and the prompt's
//! length is the piece's own: `mahbot-inference`'s `listen::prompt_len` is three
//! head ids + six text ids + the piece's audio rows + six tail ids — `tokens + 15`
//! — and the pieces a transcription cuts are `{16, 32, 48, 64, 80, 96}` audio rows,
//! so the six rung row counts a prefill runs at are `m = {31, 47, 63, 79, 95, 111}`.
//! The four body cells below are taken at `m = 95` (80 audio rows, the fourth of
//! the six) and the sweep then walks all six row counts on the shape that carries
//! half of the four projections' arithmetic — the fused gate+up, whose `n = 12288`
//! is twice the 6144-wide MLP.
//!
//! The body's projections are the four a Q4_K_M body runs per layer: the fused
//! `qkv` (`k = 2048` → `n = 4096`), the output projection (`k = 2048` →
//! `n = 2048`), the fused `gate_up` (`k = 2048` → `n = 12288`) and the down
//! projection, which is Q6_K in this file's model and so is the one cell whose
//! contraction is the 6144-wide MLP (`k = 6144` → `n = 2048`). The encoder is the
//! mmproj's Q8_0 pair: its three stride-2 convs halve the frame axis, so a
//! 768-frame piece is 96 rows (`mahbot-inference`'s `graphs.rs`: "the token count
//! is frames/8"), and its projections run between the `dim = 1024` it is wide and
//! the `ffn = 4096` its blocks are: the cells below take `k = 1024` → `n = 4096`
//! and the `ffn_down`'s own `k = 4096` → `n = 1024`.
//!
//! # Steps, calls and the batch join
//!
//! A step, a call and the batch join are `gguf_gemm_grouped_split_rate.rs`'s
//! definitions, and this file measures the same eight-step call shape: one run of a
//! graph holding [`STEPS`] `Op::DequantMatMul`s over one packed region and one
//! `m x k` activation, which is how a prefill's own graph runs them. What is this
//! file's own is that **each side's route is printed from the backend's counter**,
//! because a rate averaged over two routes is a measurement of neither.
//!
//! # Interleaving
//!
//! The round-robin protocol is [`common::time_round_robin`]'s, and the pair is why
//! it is used: a cell's two sides land in **adjacent turns of the same round**, so
//! the ratio is two figures taken under the same load in one window rather than two
//! runs' difference — this box's per-round spread on these cells runs to 95 % on the
//! shortest rung, so the 1.26-1.67x the pair reports is not resolvable across two
//! runs at all. Holding both sides alive costs `STEPS x packed_bytes` of device
//! memory each, summed, which the file prints before the timed rounds.
//!
//! The fixture, the ramp and the timing protocol are the crate's shared ones
//! ([`common::packed_gguf_weight`], [`common::ramp`], [`common::time_round_robin`]);
//! what stays here is the cell set.
//!
//! # The rates
//!
//! `packed_bytes_per_step` is the tensor's own packed bytes — the encoder's own
//! per-row stride ([`gguf_row_bytes`]) times its column count — and the two printed
//! rates come from it, as in the split file: `GBps_at_the_weight` is the rate a
//! plain read of the same tensor would be quoted at, and `TFLOPs` =
//! `2 * m * k * n * STEPS / call_seconds` is the arithmetic rate a prefill is
//! quoted at. Neither is a DRAM counter. The whole weight is read once per row
//! *tile* — 8 rows on the row-grouped tile, 32 on the cooperative one — so a call's
//! weight demand is not a comparison across routes; the pair's ratio is a ratio of
//! times for that reason, the two sides' packed bytes being the same number.
//!
//! # What is asserted, and what is printed
//!
//! Asserted, never a time: **all [`STEPS`] steps of the untimed call take the same
//! route, on each side of the pair** — and on the `grouped` side that route is the
//! row-grouped tile's, `grouped == STEPS` and `coop == 0`, so a run where the lever
//! stopped reaching the encoder fails here instead of printing a pair of two
//! cooperative figures. The warm-up's outputs are finite and each is `m * n` long on
//! both sides, and the fixture's packed length is `n` times the encoder's per-row
//! stride ([`gguf_row_bytes`]) — the fixture's own shape stated twice, not a check
//! that the encoder's stride has not moved.
//!
//! Skipped rather than asserted: a cell whose `n` exceeds the encoder's own column
//! cap ([`grouped_dispatch_cols`]) is named and dropped — for both sides at once,
//! since the cap is a property of the shape and the device rather than of the tile.
//! That skip can silently empty the run, so a printed `pairs=0` is no evidence.
//!
//! Printed: per cell the geometry, then one line per side with its route, its
//! census, **the grid that side's route dispatches** — [`coop_gemm_gguf_grid`] for a
//! cooperative call, [`grouped_chunk_geometry`] for the row-grouped one, read from
//! the module that dispatches rather than re-derived here — its submissions and
//! waits, its per-round times and both rates; then the pair's ratio line, which says
//! in words when the tile was not taken. Not in this file's sight: the dequant
//! arithmetic (the parity suites cover values), the per-chunk host work, the later
//! chunks of a chunked shape, the rest of a prefill's graph around these
//! projections, and any adapter but the one printed.
//!
//! # Cells
//!
//! Each row of [`CELLS`] is a call of [`STEPS`] = 8 multiplies, measured twice — once
//! per side of its pair — and the cell's own comment says why it is there. What the
//! file prints per cell is its whole record — the geometry, the two sides' routes
//! and grids, the bytes a step reads, the per-call times and the pair's ratio — so
//! the table is not repeated here.

mod common;

use rlx_ir::DType;
use rlx_ir::env::RuntimeOverrides;
use rlx_ir::quant::QuantScheme;
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::gguf_gpu::{
    DequantMatmulRoutes, gguf_row_bytes, grouped_chunk_geometry, grouped_dispatch_cols,
    reset_route_counts, route_counts,
};
use rlx_wgpu::gguf_host::gguf_scheme_id;
use rlx_wgpu::kernels::coop_gemm_gguf_grid;
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts, wgpu_device};

/// One measured cell; see the module doc's "Cells".
struct Cell {
    label: &'static str,
    scheme: QuantScheme,
    m: usize,
    k: usize,
    n: usize,
}

/// Multiplies per call, for every cell: see the module doc's "Steps, calls and the
/// batch join".
const STEPS: usize = 8;

/// The two sides of a pair, as printed and as the timed closure reads them.
const GROUPED: &str = "grouped";
const TAKEN: &str = "taken";

/// The bisect lever that puts every packed multiply back on the row-grouped tile —
/// `gguf_gpu::coop_gemm_admits`'s own last term, read per call at run time. Installed
/// for the `grouped` side and for nothing else.
const COOP_LEVER: &str = "RLX_WGPU_GEMM_COOP_DISABLE";

/// [`COOP_LEVER`] installed for one side, `1` on the `grouped` side and `0` on the
/// `taken` one. Both sides pay the same install, so the pair is not one side paying
/// for a guard the other does not — and stating `0` rather than leaving it unset
/// means an ambient `RLX_WGPU_GEMM_COOP_DISABLE` in the measuring shell cannot
/// quietly turn the `taken` side into a second `grouped` one.
fn lever(side: &str) -> RuntimeOverrides {
    RuntimeOverrides::install(Some((COOP_LEVER, if side == GROUPED { "1" } else { "0" })))
}

/// The cells. A `static` rather than a `const` so a cell built up front can hold
/// the entry it was built from for the whole run, the registry outliving every
/// borrow of it.
static CELLS: [Cell; 12] = [
    // The fused `qkv` of a body layer at the 80-audio-row rung (m = 95): the
    // projection every layer runs first, over the 2048-wide hidden.
    Cell {
        label: "q4k qkv m95 k2048 n4096",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 4096,
    },
    // The attention output projection at the same rows: the same contraction,
    // half the columns.
    Cell {
        label: "q4k o m95 k2048 n2048",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 2048,
    },
    // The fused `gate_up`: twice the 6144-wide MLP, and the widest `n` a body
    // dispatch covers at these rows — half of the four projections' arithmetic.
    Cell {
        label: "q4k gate/up m95 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 12288,
    },
    // The down projection, the one body cell on Q6_K and the one whose contraction
    // is the MLP's own 6144 rather than the hidden's 2048.
    Cell {
        label: "q6k down m95 k6144 n2048",
        scheme: QuantScheme::GgufQ6K,
        m: 95,
        k: 6144,
        n: 2048,
    },
    // The encoder's `pw1` at the 96 rows a 768-frame piece makes (frames/8), on the
    // mmproj's Q8_0: the narrowest contraction here against the widest column
    // count, which is the shape the encoder's own block geometry is read at.
    Cell {
        label: "q8_0 enc m96 k1024 n4096",
        scheme: QuantScheme::GgufQ8_0,
        m: 96,
        k: 1024,
        n: 4096,
    },
    // ... and `pw2`, the same tensor transposed: four times the contraction into a
    // quarter of the columns.
    Cell {
        label: "q8_0 enc-down m96 k4096 n1024",
        scheme: QuantScheme::GgufQ8_0,
        m: 96,
        k: 4096,
        n: 1024,
    },
    // The row-count sweep on the dominant shape: the six rung row counts a
    // prefill runs at, `tokens + 15` for `tokens` in {16, 32, 48, 64, 80, 96}. The
    // grid's `y` is `ceil(m / 8)`, so these six are the row-group counts 4, 6, 8,
    // 10, 12 and 14 — the `m` axis of the weight re-reads a prefill pays. The
    // `m = 95` rung is the third cell above as well, built twice: two independent
    // cells of one shape, which read as that shape's repeatability witness.
    Cell {
        label: "q4k gate/up m31 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 31,
        k: 2048,
        n: 12288,
    },
    Cell {
        label: "q4k gate/up m47 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 47,
        k: 2048,
        n: 12288,
    },
    Cell {
        label: "q4k gate/up m63 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 63,
        k: 2048,
        n: 12288,
    },
    Cell {
        label: "q4k gate/up m79 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 79,
        k: 2048,
        n: 12288,
    },
    Cell {
        label: "q4k gate/up m95 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 12288,
    },
    Cell {
        label: "q4k gate/up m111 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 111,
        k: 2048,
        n: 12288,
    },
];

/// Timed rounds, and calls per round per side: the per-round figure is that
/// round's own mean per call, and the spread across rounds is reported as it
/// stands rather than smoothed away, so run-to-run spread is information about
/// the shape. The rounds are the round-robin the module doc describes: one round
/// is every side's own [`CALLS_PER_ROUND`] calls, not one side's.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 3;

/// The one route a call's steps took, or a panic naming the counts: the module
/// doc's "What is asserted". Which route it is stays the encoder's answer — this
/// file reads it out rather than pinning it — but a call whose steps *disagree* is
/// a routing defect, and a figure averaged over two routes is not a measurement of
/// either.
fn single_route(cell: &Cell, side: &str, routes: &DequantMatmulRoutes) -> &'static str {
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
        taken.len() == 1 && taken[0].1 == STEPS as u64,
        "{} (side={side}): the {STEPS} steps of one call did not take one route: {taken:?} — a \
         call whose steps disagree is a routing defect, not a measurement",
        cell.label
    );
    taken[0].0
}

/// The grid the route that ran dispatches, from the module that dispatches it:
/// the cooperative tile's own [`coop_gemm_gguf_grid`] for a `coop` call, the
/// row-grouped arm's own [`grouped_chunk_geometry`] for a `grouped` one — never a
/// grid re-derived here, which is how the two tiles' grids were once conflated.
/// [`single_route`] has established that every step of the call took this one
/// route; a route with no such grid cannot be one of these cells, and saying so is
/// better than printing a geometry that did not run.
fn route_grid(cell: &Cell, route: &str) -> (u32, u32) {
    match route {
        "coop" => coop_gemm_gguf_grid(cell.m as u32, cell.n as u32),
        "grouped" => {
            let (x, y, _) = grouped_chunk_geometry(gguf_scheme_id(cell.scheme), cell.m, cell.n);
            (x, y)
        }
        other => panic!(
            "{}: route {other} dispatches no single grid at these shapes — every cell here is a \
             multi-row call over whole packing blocks, which the ladder answers with one of the \
             two fused tiles",
            cell.label
        ),
    }
}

/// One side of one cell, built, warmed, censused and asserted — held alive for the
/// whole run rather than built and dropped on its own, so the rounds can interleave
/// it with every other side (the module doc's "Interleaving, and the aggregate
/// footprint").
struct Ready {
    /// The entry this was built from: alive in [`CELLS`] for the whole run.
    cell: &'static Cell,
    /// [`GROUPED`] or [`TAKEN`]: which of the pair this is, and the one thing the
    /// timed closure reads to decide whether the lever goes on.
    side: &'static str,
    packed_bytes_per_step: usize,
    /// The grid **this side's route** dispatches, from [`route_grid`].
    grid: (u32, u32),
    /// The one route the call's [`STEPS`] steps took, as [`single_route`] names it.
    route: &'static str,
    routes: DequantMatmulRoutes,
    submits: u64,
    waits: u64,
    x: Vec<f32>,
    compiled: CompiledGraph,
    /// This side's mean ms per call, one entry per round.
    per_call_ms: Vec<f64>,
}

/// One cell's pair: the same graph compiled and measured twice, once with the lever
/// installed and once with the encoder's own answer. Built, warmed and censused
/// together so both sides of a cell are alive for the same rounds.
struct Pair {
    grouped: Ready,
    taken: Ready,
}

/// One cell's fixture, then both sides' compile, warm-up, finiteness census and
/// structural assertions — everything but the timing, which interleaves the sides so
/// it cannot live here. `None`: the cell was **named as not one chunk and skipped**
/// rather than measured, both of its sides with it, the later cells still being
/// measured.
fn build(cell: &'static Cell, dev: &rlx_wgpu::device::WgpuDevice) -> Option<Pair> {
    // The tensor's own packed bytes — the module doc's numerator — from the
    // encoder's own per-row stride rather than the scheme's block geometry
    // restated here.
    let row_bytes = gguf_row_bytes(gguf_scheme_id(cell.scheme), cell.k);
    let packed_bytes_per_step = cell.n * row_bytes as usize;

    // The whole packed weight is one chunk only if the encoder dispatches it as
    // one: [`grouped_dispatch_cols`] is the encoder's own column cap, both of its
    // budgets at once (the bound weight window and the whole-bound output slab),
    // and the device limit it reads is the one `RLX_WGPU_MAX_BIND_MB` lowers. A
    // shape the encoder would chunk is named here rather than measured with a
    // whole-tensor geometry it does not dispatch. The cap is a property of the shape
    // and the device, not of the tile, so it skips both sides at once.
    let chunk_cols =
        grouped_dispatch_cols(&dev.device, gguf_scheme_id(cell.scheme), cell.m, cell.k);
    if chunk_cols < cell.n {
        eprintln!(
            "[gguf prefill rate] cell=\"{}\" is not one chunk: the encoder covers {chunk_cols} of \
             its {} columns per dispatch — skipped rather than measured as a shape whose printed \
             geometry it is not",
            cell.label, cell.n
        );
        return None;
    }

    // The fixture lays out `n` rows of the scheme's own block geometry, and
    // [`gguf_row_bytes`] is that geometry's own arithmetic on the other side, so
    // this equality is the fixture stated twice rather than a check that the
    // encoder's stride has not moved.
    let packed = common::packed_gguf_weight(cell.scheme, cell.k, cell.n);
    assert_eq!(
        packed.len(),
        packed_bytes_per_step,
        "{}: the fixture must reproduce the cell's packed bytes",
        cell.label
    );

    let x = common::rate_input_matrix(cell.m, cell.k);
    let grouped = build_side(cell, GROUPED, packed_bytes_per_step, &packed, &x);
    let taken = build_side(cell, TAKEN, packed_bytes_per_step, &packed, &x);
    Some(Pair { grouped, taken })
}

/// One side of a cell: compile, parameterise, warm, assert, census. The `grouped`
/// side has [`COOP_LEVER`] installed for all of it, because the encoder reads that
/// lever per call at run time rather than at the lowering; the guard covers the
/// compile as well, so it does not rest on where the decision is taken.
fn build_side(
    cell: &'static Cell,
    side: &'static str,
    packed_bytes_per_step: usize,
    packed: &[u8],
    x: &[f32],
) -> Ready {
    let _lever = lever(side);

    let mut compiled = Session::new(Device::Gpu).compile(common::rate_cell_graph(
        "wgpu_gguf_gemm_prefill_rate",
        cell.scheme,
        cell.m,
        cell.k,
        cell.n,
        packed.len(),
        STEPS,
    ));
    for s in 0..STEPS {
        compiled.set_param_typed(&format!("w{s}"), packed, DType::U8);
    }

    let warm = compiled.run(&[("x", x)]);
    assert_eq!(warm.len(), STEPS, "{}: one output per step", cell.label);
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

    // The census: which route, how many submissions and waits — one call's own
    // counts, taken outside the timing, and taken while this side is the only one
    // running.
    reset_route_counts();
    reset_device_counts();
    let _ = compiled.run(&[("x", x)]);
    let routes = route_counts();
    let (submits, waits) = (device_submit_count(), device_wait_count());
    let route = single_route(cell, side, &routes);

    // The lever's effect as a fact rather than an assumption: on the `grouped` side
    // every step is the row-grouped tile and none is cooperative, so a lever that
    // stopped reaching the encoder fails here instead of printing a pair of two
    // cooperative figures.
    if side == GROUPED {
        assert_eq!(
            (routes.grouped, routes.coop),
            (STEPS as u64, 0),
            "{}: the {COOP_LEVER} side is not the row-grouped tile — {routes:?}; the lever is \
             read by the encoder where the multiply is encoded, so a census that disagrees means \
             the guard no longer reaches it",
            cell.label
        );
    }

    Ready {
        cell,
        side,
        packed_bytes_per_step,
        grid: route_grid(cell, route),
        route,
        routes,
        submits,
        waits,
        x: x.to_vec(),
        compiled,
        per_call_ms: Vec::with_capacity(ROUNDS),
    }
}

/// The timed rounds, round-robin across the sides the module doc describes: one
/// round gives every side its own [`CALLS_PER_ROUND`] calls back to back before the
/// next round starts, so a cell's two sides land in adjacent turns and each side's
/// figures are its own rather than its position's.
fn run_rounds(sides: &mut [Ready]) {
    let per_side = common::time_round_robin(sides, ROUNDS, CALLS_PER_ROUND, |r| {
        // The lever, per call: the encoder reads it where the multiply is encoded,
        // so it has to state this side's answer on every run. The guard costs a few
        // allocations against a call of milliseconds, it is installed for both sides
        // alike, and it is the price of measuring both tiles in one window.
        let _lever = lever(r.side);
        let _ = r.compiled.run(&[("x", r.x.as_slice())]);
    });
    for (r, times) in sides.iter_mut().zip(per_side) {
        r.per_call_ms = times;
    }
}

/// One side's printed fields, both rates and its per-round figures.
fn report_side(r: &Ready) {
    let cell = r.cell;
    let (x_groups, y_groups) = r.grid;
    let per_call_ms = &r.per_call_ms;
    let min_ms = common::least(per_call_ms);
    let med_ms = common::median(per_call_ms);
    let max_ms = per_call_ms.iter().cloned().fold(0.0f64, f64::max);
    // The rates: the tensor read once (`at_the_weight`) and the arithmetic the
    // call does over it (`tflops`) — see the module doc for both.
    let at_the_weight =
        |call_ms: f64| r.packed_bytes_per_step as f64 * STEPS as f64 / (call_ms * 1e-3) / 1e9;
    let flops = 2.0 * cell.m as f64 * cell.k as f64 * cell.n as f64 * STEPS as f64;
    let tflops = |call_ms: f64| flops / (call_ms * 1e-3) / 1e12;
    let (side, route, routes, submits, waits) = (r.side, r.route, &r.routes, r.submits, r.waits);

    eprintln!(
        "[gguf prefill rate] cell=\"{}\" side={side} route={route} routes={routes:?} \
         grid={x_groups}x{y_groups}={} workgroups submits={submits} waits={waits}",
        cell.label,
        x_groups * y_groups
    );
    eprintln!(
        "[gguf prefill rate] cell=\"{}\" side={side} per_call_ms=[{}] min_ms={min_ms:.3} \
         median_ms={med_ms:.3} max_ms={max_ms:.3} spread_pct={:.1} min_TFLOPs={:.2} \
         median_TFLOPs={:.2} min_GBps_at_the_weight={:.1} \
         median_GBps_at_the_weight={:.1}",
        cell.label,
        per_call_ms
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        (max_ms - min_ms) / med_ms * 100.0,
        tflops(min_ms),
        tflops(med_ms),
        at_the_weight(min_ms),
        at_the_weight(med_ms),
    );
}

/// One cell's whole record: the geometry it shares, each side's own fields, and the
/// ratio the pair exists for — with the reason it is not a tile comparison when the
/// `taken` side did not take the tile.
fn report_pair(grouped: &Ready, taken: &Ready) {
    let cell = grouped.cell;
    eprintln!(
        "[gguf prefill rate] cell=\"{}\" scheme={:?} m={} k={} n={} steps={} \
         packed_bytes_per_step={}",
        cell.label, cell.scheme, cell.m, cell.k, cell.n, STEPS, grouped.packed_bytes_per_step,
    );
    report_side(grouped);
    report_side(taken);

    let grouped_median = common::median(&grouped.per_call_ms);
    let taken_median = common::median(&taken.per_call_ms);
    let grouped_min = common::least(&grouped.per_call_ms);
    let taken_min = common::least(&taken.per_call_ms);
    let median_ratio = grouped_median / taken_median;
    let min_ratio = grouped_min / taken_min;
    if taken.route == "coop" {
        eprintln!(
            "[gguf prefill rate] cell=\"{}\" pair= median_grouped_over_taken={median_ratio:.2}x \
             min_grouped_over_taken={min_ratio:.2}x — the cooperative tile was taken, so this is \
             the two tiles' rate on this shape in one window",
            cell.label
        );
    } else {
        eprintln!(
            "[gguf prefill rate] cell=\"{}\" pair= median_grouped_over_taken={median_ratio:.2}x \
             min_grouped_over_taken={min_ratio:.2}x the tile was not taken: the lever is off, or \
             the adapter has no cooperative matrices, or the shape is refused — the \"taken\" side \
             ran the row-grouped tile too, so this ratio is a repeatability witness only, not a \
             tile comparison",
            cell.label
        );
    }
}

/// One test fn on purpose: what it reads is process-global (the route census and
/// the device submit/wait counters), and the round-robin needs every side's device
/// memory alive at once rather than one side's gone before the next is built.
#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn gguf_prefill_rate_at_the_product_shapes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    common::refuse_under_run_report();
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    eprintln!(
        "[gguf prefill rate] adapter={:?} — this cell's figures hold for this adapter and \
         machine load only",
        rlx_wgpu::device::adapter_name()
    );

    // Every pair built, warmed and censused up front and held for the whole run:
    // that is the interleaving's cost, and the aggregate below is the number the
    // module doc names.
    let pairs: Vec<Pair> = CELLS.iter().filter_map(|cell| build(cell, dev)).collect();
    let mut sides: Vec<Ready> = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        // The flatten is `grouped` then `taken` per cell, in [`CELLS`] order — the
        // order `chunks_exact(2)` below reads back. Both sides of a pair come from
        // one `build` call on one cell, so the pair's own association is the
        // constructor's and nothing here re-establishes it.
        sides.push(pair.grouped);
        sides.push(pair.taken);
    }
    let aggregate_packed_bytes: u64 = sides
        .iter()
        .map(|r| r.packed_bytes_per_step as u64 * STEPS as u64)
        .sum();
    eprintln!(
        "[gguf prefill rate] pairs={} sides={} aggregate_packed_bytes={aggregate_packed_bytes} — \
         every side held alive at once for the round-robin, so this is twice the single-tile \
         figure; a cell skipped above is named there",
        sides.len() / 2,
        sides.len()
    );

    run_rounds(&mut sides);

    for pair in sides.chunks_exact(2) {
        report_pair(&pair[0], &pair[1]);
    }
}
