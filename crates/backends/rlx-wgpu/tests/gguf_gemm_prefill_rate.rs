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
//! * `side=taken` — compiled and run with `RLX_WGPU_GEMM_COOP_DISABLE=0` and
//!   `RLX_WGPU_GEMM_COOP_TAIL_DISABLE=0` installed, so the encoder answers with both
//!   bisect levers at their shipped values: the cooperative-matrix tile
//!   (`dequant_gemm_coop_gguf.wgsl`: 32 rows x 128 columns, the weight decoded into a
//!   workgroup f16 stage and the arithmetic on the hardware matrix units) where
//!   `gguf_gpu::coop_gemm_admits` admits the shape, the row-grouped one otherwise — with
//!   the short last tile its shipped form.
//!
//! A cell's two sides are timed in **adjacent rounds of the same process**, not in
//! two runs with the lever toggled between them. This box's per-round spread on
//! these cells runs to 95 % on the shortest rung, so a run-to-run difference of the
//! size the pair is here to report (per `dequant_gemm_coop_gguf.wgsl`'s header table)
//! is not resolvable across two runs at all, while within one window it is a ratio of
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
//! shortest rung, so the ratio the pair reports is not resolvable across two
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
//! per-row stride times its column count — and the two printed rates come from it, as
//! in the split file: `GBps_at_the_weight` is the rate a
//! plain read of the same tensor would be quoted at, and `TFLOPs` =
//! `2 * m * k * n * STEPS / call_seconds` is the arithmetic rate a prefill is
//! quoted at. Neither is a DRAM counter. The whole weight is read once per row
//! *group* — 8 rows on the row-grouped tile, 32 on the cooperative one, whose last
//! group may be the 16-row short form — so a call's weight demand is not a comparison
//! across routes; the pair's ratio is a ratio of times for that reason, the two sides'
//! packed bytes being the same number.
//!
//! # What is asserted, and what is printed
//!
//! Asserted, never a time: **all [`STEPS`] steps of the untimed call take the same
//! route, on each side of the pair** — and on the `grouped` side that route is the
//! row-grouped tile's, `grouped == STEPS` and `coop == 0`, so a run where the lever
//! stopped reaching the encoder fails here instead of printing a pair of two
//! cooperative figures. The warm-up's outputs are finite and each is `m * n` long on
//! both sides, and the fixture's packed length is `n` times the encoder's per-row
//! stride — the fixture's own shape stated twice, not a check that the encoder's
//! stride has not moved.
//!
//! Skipped rather than asserted: a cell whose `n` exceeds the encoder's own column
//! cap is named and dropped — for both sides at once, since the cap is a property of
//! the shape and the device rather than of the tile. That skip can silently empty the
//! run, so a printed `pairs=0` is no evidence.
//!
//! Printed: per cell the geometry, then one line per side with its route, its
//! census, **the grid that side's route dispatches** — `coop_gemm_gguf_grid` for a
//! cooperative call, `grouped_chunk_geometry` for the row-grouped one, both read in
//! `common` from the modules that dispatch them rather than re-derived here — its
//! submissions and waits, its per-round times and both rates; then the pair's ratio
//! line, which says in words when the tile was not taken. Not in this file's sight: the dequant
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

use rlx_ir::env::RuntimeOverrides;
use rlx_ir::quant::QuantScheme;
use rlx_runtime::Device;
use rlx_wgpu::wgpu_device;

/// One measured cell; see the module doc's "Cells".
type Cell = common::RateCell;

/// Multiplies per call, for every cell: see the module doc's "Steps, calls and the
/// batch join".
const STEPS: usize = 8;

/// Every line this file prints starts here.
const PREFIX: &str = "[gguf prefill rate]";

/// The two sides of a pair, as printed and as the timed closure reads them.
const GROUPED: &str = "grouped";
const TAKEN: &str = "taken";

/// The bisect lever that puts every packed multiply back on the row-grouped tile —
/// `gguf_gpu::coop_gemm_admits`'s own last term, read per call at run time. Installed
/// for the `grouped` side and for nothing else.
const COOP_LEVER: &str = "RLX_WGPU_GEMM_COOP_DISABLE";

/// The tile's own bisect lever, which restores its full 32-row last row group. Pinned
/// off on both sides: this file measures the tile as it ships, and an ambient setting
/// in the measuring shell would silently put the documented table back on the dispatch
/// the short last tile replaced.
const TAIL_LEVER: &str = "RLX_WGPU_GEMM_COOP_TAIL_DISABLE";

/// Both levers installed for one side, `1` on the `grouped` side and `0` on the
/// `taken` one. Both sides pay the same install, so the pair is not one side paying
/// for a guard the other does not — and stating `0` rather than leaving it unset
/// means an ambient setting in the measuring shell cannot quietly turn the `taken`
/// side into a second `grouped` one, or the tile into its pre-tail dispatch.
fn lever(side: &str) -> RuntimeOverrides {
    RuntimeOverrides::install([
        (COOP_LEVER, if side == GROUPED { "1" } else { "0" }),
        (TAIL_LEVER, "0"),
    ])
}

/// The cells. A `static` rather than a `const` so a cell built up front can hold
/// the entry it was built from for the whole run, the registry outliving every
/// borrow of it.
static CELLS: [Cell; 16] = [
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
    // The shortest row count the product's prefill reaches — a single 32-row
    // tile — at the three column counts its projections land on. These are the
    // cells the cooperative tile's workgroup floor is read at: 8, 16 and 32
    // workgroups, one per `n / 128`.
    Cell {
        label: "q4k o m31 k2048 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 31,
        k: 2048,
        n: 1024,
    },
    Cell {
        label: "q4k o m31 k2048 n2048",
        scheme: QuantScheme::GgufQ4K,
        m: 31,
        k: 2048,
        n: 2048,
    },
    Cell {
        label: "q4k o m31 k2048 n4096",
        scheme: QuantScheme::GgufQ4K,
        m: 31,
        k: 2048,
        n: 4096,
    },
    // 24 workgroups: three 32-row tiles over 1 024 columns, the encoder's
    // `ffn_down` shape in the body's scheme.
    Cell {
        label: "q4k o m95 k2048 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 1024,
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

/// One cell's fixture, then both sides' compile, warm-up, finiteness census and
/// structural assertions — everything but the timing, which interleaves the sides so
/// it cannot live here. The two sides come back in the order the caller names them,
/// `grouped` then `taken`. `None`: the cell was **named as not one chunk and skipped**
/// rather than measured, both of its sides with it, the later cells still being
/// measured.
fn build(cell: &'static Cell, dev: &rlx_wgpu::device::WgpuDevice) -> Option<[common::RateSide; 2]> {
    let (packed, x) = common::rate_cell_fixture(PREFIX, cell, dev)?;

    let (grouped, _) = {
        let _lever = lever(GROUPED);
        let (side, warm) = common::rate_build_side(
            cell,
            GROUPED,
            "wgpu_gguf_gemm_prefill_rate",
            &packed,
            &x,
            STEPS,
        );
        // The lever's effect as a fact rather than an assumption: on the `grouped` side
        // every step is the row-grouped tile and none is cooperative, so a lever that
        // stopped reaching the encoder fails here instead of printing a pair of two
        // cooperative figures.
        assert_eq!(
            (side.routes.grouped, side.routes.coop),
            (STEPS as u64, 0),
            "{}: the {COOP_LEVER} side is not the row-grouped tile — {:?}; the lever is read by \
             the encoder where the multiply is encoded, so a census that disagrees means the \
             guard no longer reaches it",
            cell.label,
            side.routes
        );
        (side, warm)
    };
    let (taken, _) = {
        let _lever = lever(TAKEN);
        common::rate_build_side(
            cell,
            TAKEN,
            "wgpu_gguf_gemm_prefill_rate",
            &packed,
            &x,
            STEPS,
        )
    };
    Some([grouped, taken])
}

/// The timed rounds, round-robin across the sides the module doc describes: one
/// round gives every side its own [`CALLS_PER_ROUND`] calls back to back before the
/// next round starts, so a cell's two sides land in adjacent turns and each side's
/// figures are its own rather than its position's.
fn run_rounds(sides: &mut [common::RateSide]) {
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

/// One cell's whole record: the geometry it shares, each side's own fields, and the
/// ratio the pair exists for — with the reason it is not a tile comparison when the
/// `taken` side did not take the tile.
fn report_pair(grouped: &common::RateSide, taken: &common::RateSide) {
    let meaning = if taken.route == "coop" {
        "— the cooperative tile was taken, so this is the two tiles' rate on this shape in one \
         window"
            .to_string()
    } else {
        "the tile was not taken: the lever is off, or the adapter has no cooperative matrices, or \
         the shape is refused — the \"taken\" side ran the row-grouped tile too, so this ratio is \
         a repeatability witness only, not a tile comparison"
            .to_string()
    };
    common::report_rate_pair(PREFIX, grouped, taken, "grouped_over_taken", &meaning);
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
        "{PREFIX} adapter={:?} — this cell's figures hold for this adapter and machine load \
         only",
        rlx_wgpu::device::adapter_name()
    );

    // Every pair built, warmed and censused up front and held for the whole run:
    // that is the interleaving's cost, and the aggregate below is the number the
    // module doc names. The flatten is `grouped` then `taken` per cell, in [`CELLS`]
    // order — the order `chunks_exact(2)` below reads back, both sides of a pair
    // coming from one `build` call on one cell.
    let mut sides: Vec<common::RateSide> = CELLS
        .iter()
        .filter_map(|cell| build(cell, dev))
        .flatten()
        .collect();
    let aggregate_packed_bytes: u64 = sides
        .iter()
        .map(|r| r.packed_bytes_per_step as u64 * STEPS as u64)
        .sum();
    eprintln!(
        "{PREFIX} pairs={} sides={} aggregate_packed_bytes={aggregate_packed_bytes} — \
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
