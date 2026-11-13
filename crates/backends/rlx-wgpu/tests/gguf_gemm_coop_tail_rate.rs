// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cooperative-matrix packed multiply's **short last tile** — the 16-row form of
//! the last row group, which replaces the half-empty 32-row tile when a prompt leaves
//! sixteen rows or fewer after the whole ones — measured as a **pair**, the tail against
//! the full last tile it replaces, in one window: the grid each side dispatches, how
//! many short row groups each encodes, what a call costs, and the pair's own ratio.
//!
//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test gguf_gemm_coop_tail_rate -- --ignored --nocapture
//! ```
//!
//! # The pair, and why it is one window
//!
//! Every cell is built **twice**, from the same graph and the same packed fixture, and
//! both builds are held alive together while that cell is timed:
//!
//! * `side=tail` — compiled and run with `RLX_WGPU_GEMM_COOP_TAIL_DISABLE=0`, the
//!   encoder's own answer: `kernels::coop_gemm_gguf_grid` keeps the whole 32-row
//!   tiles before the last and makes the last row group the 16-row form anchored at the
//!   prompt's end, which the same entry point runs under `params.tail_row`;
//! * `side=full` — the same call with `RLX_WGPU_GEMM_COOP_TAIL_DISABLE=1`, the
//!   A/B lever that restores the full 32-row last row group — the dispatch before the
//!   tail existed, and the one every rung took until now.
//!
//! A cell's two sides are timed in **adjacent rounds of the same process**, not in two
//! runs with the lever toggled between them, and the side that runs first alternates
//! round by round (`common::paired_round_robin`), so a drift inside a round lands on both
//! sides rather than on their difference. The per-round spread on these cells
//! reaches tens of percent across takes — `gguf_gemm_prefill_rate.rs` records 95 % on
//! its shortest rung — while the effect sought is sixteen rows: half of one row group,
//! a tenth to a quarter of the dispatch a cell makes. A difference that size is not
//! resolvable across two runs at all; within one window it is a ratio of two figures
//! taken under the same load, in the same round-robin, on the same graph.
//!
//! The lever is **read per call, by the encoder, at run time**: the encode arm applies it
//! to `kernels::CoopGrid`'s `tail` where the multiply is encoded, not where the graph is
//! lowered. So it is installed around every run of each side — its
//! warm-up, its census call and each timed call — and around its compile as well, so
//! nothing here rests on the decision staying at run time. Both sides install it, `0`
//! on the `tail` side and `1` on the `full` one, so neither pays for a guard the other
//! does not and an ambient `RLX_WGPU_GEMM_COOP_TAIL_DISABLE` in the measuring shell
//! cannot turn the `tail` side into a second `full` one. `RuntimeOverrides::install`
//! restores the environment when it drops.
//!
//! # What the tail is, and what it is not
//!
//! The tail is a **second form of the same entry point**, not a route: which calls
//! take the cooperative tile is `gguf_gpu::coop_gemm_admits`'s answer and the tail
//! does not move it. Both sides of every pair are therefore `coop`, which is asserted
//! rather than assumed — a cell the adapter does not serve is named and skipped, since
//! there is no short form to measure there — and the short form is read from
//! `gguf_gpu::coop_tail_dispatches`, a counter beside the route census, because it is
//! one row group of a `DequantMatmulRoute::Coop` call rather than a route of its own.
//! The counter is what makes the lever's effect a fact: the grid is the shape's
//! geometry and reads no lever, so a `full` side that still encodes a short tile is
//! visible in the census and nowhere else.
//!
//! The tail replaces the last whole tile rather than extending the grid, so both sides
//! dispatch `ceil(n / 128)` columns by `ceil(m / 32)` row groups — the count
//! `COOP_MIN_WORKGROUPS` is read against does not move with the tail, and the short-form
//! census is what witnesses which form the last group ran. At a
//! remainder of exactly 16 (48, 80 rows) the tail's rows are the ones no whole tile
//! wrote; at 15 (47, 79, 111 rows) the tail is anchored at `m - 16` and so re-writes
//! one row the last whole tile already wrote, with the same inputs and the same value;
//! at 1 (65 rows) it re-writes fifteen. The three classes are measured because they are
//! three different dispatches over the same number of row groups, and the last is the
//! largest overlap the rule can produce. A remainder between 2 and 14 runs the same
//! body with a different anchor offset, so it adds no class the three do not carry.
//!
//! # What the ratio means, and what it does not
//!
//! The pair line prints the cell's `median` and its `min`, full over tail, and the
//! per-round paired difference (tail minus full) with the rounds the short tile won:
//! above 1 and a negative delta the short tile is the faster of the two. Where the tail
//! dispatched (the counter says [`STEPS`] short tiles on the `tail` side and none on the
//! `full` one) the pair is what the sixteen rows buy on that shape, in one window.
//!
//! Where it did not — the `m = 95` cell, whose remainder of 31 is above the short
//! tile's 16, so both sides dispatch the same grid — the line says so in words and the
//! pair is a **repeatability witness only**: two builds of the same graph on the same
//! tile differ by whatever this window's noise is, which is worth reading as the
//! file's own floor before reading any of the other cells.
//!
//! Neither figure is a claim about the rest of a prefill's graph, and no time is
//! asserted anywhere.
//!
//! # Where the shapes come from
//!
//! A recognition pass prefills one body graph over its prompt, and the prompt's length
//! is the piece's own: `mahbot-inference`'s `listen::prompt_len` is three head ids +
//! six text ids + the piece's audio rows + six tail ids — `tokens + 15` — and the
//! pieces a transcription cuts are `{16, 32, 48, 64, 80, 96}` audio rows, so the six
//! rung row counts a prefill runs at are `m = {31, 47, 63, 79, 95, 111}`. The cells
//! below take the three rungs whose remainder is 15 (47, 79, 111), the two whose
//! remainder is exactly 16 (48, 80), and the remainder of 1 a 65-row prompt lands on —
//! the latter two reached by a single-piece input, which runs at whatever frame count
//! it has rather than at one of the six rungs, and 65 being the smallest remainder-one
//! row count the admission lets through, the rounding term refusing 33
//! (`kernels::CoopGrid`'s `tail` has that arithmetic). The tail rule is a rule about
//! `m % 32`, not about the six rungs, so all three classes are measured.
//!
//! The body's projections are the four a Q4_K_M body runs per layer: the fused `qkv`
//! (`k = 2048` → `n = 4096`), the output projection (`k = 2048` → `n = 2048`), the
//! fused `gate_up` (`k = 2048` → `n = 12288`, half of the four projections' arithmetic
//! and so the cell the widest dispatch is read at) and the down projection, which is
//! Q6_K in this file's model and so is the one cell whose contraction is the 6144-wide
//! MLP (`k = 6144` → `n = 2048`). The encoder is the mmproj's Q8_0 pair: its three
//! stride-2 convs halve the frame axis, so a 768-frame piece is 96 rows, and its
//! projections run between the `dim = 1024` it is wide and the `ffn = 4096` its blocks
//! are — the cells below take `k = 1024` → `n = 4096` and `ffn_down`'s own
//! `k = 4096` → `n = 1024`.
//!
//! # Steps, calls and the batch join
//!
//! A step, a call and the batch join are `gguf_gemm_grouped_split_rate.rs`'s
//! definitions, and this file measures the same eight-step call shape: one run of a
//! graph holding [`STEPS`] `Op::DequantMatMul`s over one packed region and one
//! `m x k` activation, which is how a prefill's own graph runs them.
//!
//! # What is asserted, and what is printed
//!
//! Asserted, never a time: on **each side** of every pair, the [`STEPS`] steps of the
//! untimed call take one route and that route is `coop` — a call that left the tile is
//! not this file's measurement; the short-dispatch census counts [`STEPS`] short row
//! groups on the `tail` side and none on the `full` one; the warm-up's outputs are finite
//! and each is `m * n` long on both sides; **the two sides' outputs are bit-identical**,
//! step for step; and the fixture's packed length is `n` times the encoder's per-row
//! stride — the fixture's own shape stated twice, not a check that the encoder's stride
//! has not moved.
//!
//! The bit comparison is this file's evidence that the product's answers do not move.
//! The short form computes its rows from the same operands in the same k order over the
//! same f16 stage, so there is nothing for a rounding to differ on; a moved bit would be
//! the tail's arithmetic and nothing else. The parity suite bounds a value against the
//! CPU at 1e-2 and the stand's digests are witnesses, so neither would see a reordering
//! inside the tile.
//!
//! Skipped rather than asserted: a cell whose `n` exceeds the encoder's own column cap,
//! or whose adapter does not serve the cooperative tile, is named and dropped — for both
//! sides at once, since both are properties of the shape, the device and the adapter
//! rather than of the tile. That skip can silently empty the run, so a printed `pairs=0`
//! is no evidence.
//!
//! Printed: per cell the geometry, then one line per side with its route, its census,
//! **the grid that side's route dispatches** — `coop_gemm_gguf_grid`, read from the
//! module that dispatches rather than re-derived here, with its `cols`, `tiles` and
//! `tail` fields — its submissions and waits, its per-round times and both rates; then
//! the pair's ratio line, which carries the per-round paired difference and says in
//! words when no short row group ran. Not in this file's sight: the dequant arithmetic
//! (the parity suites cover values), the per-chunk host work, the rest of a prefill's
//! graph around these projections, the decode, and any adapter but the one printed.
//!
//! # Cells
//!
//! Each row of [`CELLS`] is a call of [`STEPS`] = 8 multiplies, measured twice — once
//! per side of its pair — and the cell's own comment says why it is there. What the
//! file prints per cell is its whole record, so the table is not repeated here.

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
const PREFIX: &str = "[gguf tail rate]";

/// The two sides of a pair, as printed and as the timed closure reads them.
const TAIL: &str = "tail";
const FULL: &str = "full";

/// The A/B lever that restores the full 32-row last tile — applied to
/// `kernels::CoopGrid`'s `tail` where the multiply is encoded, read per call at run time.
/// Installed for both sides, `0` on the `tail` one.
const TAIL_LEVER: &str = "RLX_WGPU_GEMM_COOP_TAIL_DISABLE";

/// [`TAIL_LEVER`] installed for one side, `0` on the `tail` side and `1` on the `full`
/// one. Both sides pay the same install, so the pair is not one side paying for a
/// guard the other does not — and stating `0` rather than leaving it unset means an
/// ambient `RLX_WGPU_GEMM_COOP_TAIL_DISABLE` in the measuring shell cannot quietly
/// turn the `tail` side into a second `full` one.
fn lever(side: &str) -> RuntimeOverrides {
    RuntimeOverrides::install(Some((TAIL_LEVER, if side == FULL { "1" } else { "0" })))
}

/// The cells. A `static` rather than a `const` so a cell built up front can hold the
/// entry it was built from for the whole run, the registry outliving every borrow of
/// it.
static CELLS: [Cell; 15] = [
    // The widest body projection at the longest rung: 111 rows leave 15 past three
    // whole tiles, so the tail replaces a 32-row tile that is 17 rows empty.
    Cell {
        label: "q4k gate/up m111 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 111,
        k: 2048,
        n: 12288,
    },
    // The same shape two rungs down: two whole tiles, then a tail over the 15 rows
    // they leave.
    Cell {
        label: "q4k gate/up m79 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 79,
        k: 2048,
        n: 12288,
    },
    // The shortest rung that leaves a remainder, on the widest dispatch: one whole
    // tile, then a tail over 15 rows — the largest share of a cell the tail removes
    // (48 rows of dispatch against 64).
    Cell {
        label: "q4k gate/up m47 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 47,
        k: 2048,
        n: 12288,
    },
    // A remainder of exactly 16: the tail's rows are the ones no whole tile wrote, so
    // 48 rows of dispatch for 48 valid ones and no overlap to re-write.
    Cell {
        label: "q4k gate/up m48 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 48,
        k: 2048,
        n: 12288,
    },
    // ... and the same at 80 rows: two whole tiles, then 16 rows with no overlap.
    Cell {
        label: "q4k gate/up m80 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 80,
        k: 2048,
        n: 12288,
    },
    // A remainder of one: 65 rows leave a single row past two whole tiles, the largest
    // overlap the rule can produce and the one place a wrong anchor shows as more than a
    // single row. A single-piece input runs at whatever frame count it has, and 15 prompt
    // tokens plus 50 audio rows is a 400-frame clip. It is also the **smallest**
    // remainder-one shape the admission lets through (`kernels::CoopGrid`'s `tail` works
    // out why).
    Cell {
        label: "q4k gate/up m65 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 65,
        k: 2048,
        n: 12288,
    },
    // The other two rungs that only duplicate one row — 31 is a single tile, 63 two —
    // where the short form must leave the dispatch exactly as it was. Both sides run the
    // same grid and the ratio is this file's repeatability floor, which is what makes
    // "the rungs the tail does not serve are not worse" a reading rather than a claim.
    Cell {
        label: "q4k gate/up m31 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 31,
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
    // The null-result witness at the product's typical rung: 95 rows leave 31 past two
    // whole tiles, above the short tile's 16, so neither side dispatches a tail and the
    // pair's ratio is this file's own repeatability floor.
    Cell {
        label: "q4k gate/up m95 k2048 n12288",
        scheme: QuantScheme::GgufQ4K,
        m: 95,
        k: 2048,
        n: 12288,
    },
    // Eight column tiles by four row groups — 32 workgroups, one row group above the
    // tile's floor of 24 — where the short row group is a larger share of the grid than
    // on the 96-column shape above.
    Cell {
        label: "q4k o m111 k2048 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 111,
        k: 2048,
        n: 1024,
    },
    // The fused `qkv`'s own width at a rung the tail takes: 16 x 3 = 48 workgroups.
    Cell {
        label: "q4k qkv m79 k2048 n2048",
        scheme: QuantScheme::GgufQ4K,
        m: 79,
        k: 2048,
        n: 2048,
    },
    // 16 x 2 = 32 workgroups at the rung the tail takes a quarter of the rows from.
    Cell {
        label: "q4k o m47 k2048 n2048",
        scheme: QuantScheme::GgufQ4K,
        m: 47,
        k: 2048,
        n: 2048,
    },
    // The encoder's `pw1` on the mmproj's Q8_0 at a remainder of exactly 16: the other
    // scheme's short form, at the widest column count here.
    Cell {
        label: "q8_0 enc m48 k1024 n4096",
        scheme: QuantScheme::GgufQ8_0,
        m: 48,
        k: 1024,
        n: 4096,
    },
    // The encoder's `ffn_down` at 80 rows: 8 x 3 = 24 workgroups, the tile's floor
    // itself, its last row group short.
    Cell {
        label: "q8_0 enc m80 k4096 n1024",
        scheme: QuantScheme::GgufQ8_0,
        m: 80,
        k: 4096,
        n: 1024,
    },
    // The Q6_K short form at a product shape: the one cell whose contraction is the
    // MLP's own 6144 rather than the hidden's 2048.
    Cell {
        label: "q6k down m111 k6144 n2048",
        scheme: QuantScheme::GgufQ6K,
        m: 111,
        k: 6144,
        n: 2048,
    },
];

/// Timed rounds, and calls per round per side: the per-round figure is that round's
/// own mean per call, and the spread across rounds is reported as it stands rather
/// than smoothed away. One round gives each side of a pair its own
/// [`CALLS_PER_ROUND`] calls back to back, the side that goes first alternating.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 3;

/// One cell's fixture, then both sides' compile, warm-up, finiteness census and
/// structural assertions — everything but the timing, which alternates the two sides and
/// so cannot live here. The two sides come back in the order the caller names them,
/// `tail` then `full`. `None`: the cell is **named and skipped** rather than measured —
/// it is not one chunk, or the adapter does not serve the cooperative tile — both of its
/// sides with it, the later cells still being measured.
fn build(cell: &'static Cell, dev: &rlx_wgpu::device::WgpuDevice) -> Option<[common::RateSide; 2]> {
    let (packed, x) = common::rate_cell_fixture(PREFIX, cell, dev)?;
    let graph = "wgpu_gguf_gemm_coop_tail_rate";

    let (tail, tail_warm) = {
        let _lever = lever(TAIL);
        common::rate_build_side(cell, TAIL, graph, &packed, &x, STEPS)
    };
    // The tail is a form inside a cooperative call, not a route. An adapter that cannot
    // serve the tile — no cooperative matrices, the probe declining the operand
    // convention, a refused shape — has nothing here to measure, so the cell is named
    // and skipped rather than failing the fourteen that follow it.
    if tail.route != "coop" {
        eprintln!(
            "{PREFIX} cell=\"{}\" did not take the cooperative tile (route={}) — no cooperative \
             matrices on this adapter, the probe declined the operand convention, or the shape is \
             refused; skipped, there is no short form here to measure: {:?}",
            cell.label, tail.route, tail.routes
        );
        return None;
    }
    let (full, full_warm) = {
        let _lever = lever(FULL);
        common::rate_build_side(cell, FULL, graph, &packed, &x, STEPS)
    };
    assert_eq!(
        full.route, "coop",
        "{} (side={FULL}): the steps took the {} route — the lever changes the last row group, \
         not which calls take the tile: {:?}",
        cell.label, full.route, full.routes
    );

    // The census is what witnesses which form the last row group ran: both sides' grid is
    // the same pure geometry, so comparing it would assert nothing.
    let short = u64::from(tail.tail.unwrap_or(false));
    assert_eq!(
        (tail.tail_dispatches, full.tail_dispatches),
        (STEPS as u64 * short, 0),
        "{}: the census must count {STEPS} short row groups on the {TAIL} side and none on the \
         {FULL} one — tail={:?} full={:?}",
        cell.label,
        (tail.cols, tail.tiles, tail.tail),
        (full.cols, full.tiles, full.tail)
    );

    // The same operands in the same k order over the same f16 stage: the two forms must
    // agree to the bit, so a moved bit is the short form's arithmetic and nothing else.
    for (s, (a, b)) in tail_warm.iter().zip(&full_warm).enumerate() {
        common::assert_bit_identical(
            a,
            b,
            &format!(
                "{}: step {s} differs between the two forms of the last row group — the short \
                 form computes the same rows from the same operands, so this is its arithmetic",
                cell.label
            ),
        );
    }
    Some([tail, full])
}

/// One cell's timed rounds: the two sides alternate, the one that runs first changing
/// every round, so a drift inside a round lands on both sides rather than on their
/// difference. Returns the per-round paired difference, tail minus full, in milliseconds
/// — negative when the short last tile is the faster of the two.
fn run_rounds(tail: &mut common::RateSide, full: &mut common::RateSide) -> Vec<f64> {
    let (per_call, deltas) = common::paired_round_robin(tail, full, ROUNDS, CALLS_PER_ROUND, |r| {
        // The lever, per call: the encoder reads it where the multiply is encoded,
        // so it has to state this side's answer on every run. The guard costs a few
        // allocations against a call of milliseconds, it is installed for both
        // sides alike, and it is the price of measuring both forms in one window.
        let _lever = lever(r.side);
        let _ = r.compiled.run(&[("x", r.x.as_slice())]);
    });
    let [tail_ms, full_ms] = per_call;
    tail.per_call_ms = tail_ms;
    full.per_call_ms = full_ms;
    deltas
}

/// One cell's whole record: the geometry it shares, each side's own fields, and the paired
/// difference the pair exists for — with the reason it is a repeatability witness when no
/// tail dispatched.
fn report_pair(tail: &common::RateSide, full: &common::RateSide, deltas: &[f64]) {
    // `deltas` are tail minus full per round: negative when the short tile won that
    // round, and the count is what says how often it did.
    let median_delta = common::median(deltas);
    let tail_won = deltas.iter().filter(|d| **d < 0.0).count();
    let meaning = if tail.tail == Some(true) {
        format!(
            "median_delta_ms={median_delta:+.3} tail_faster={tail_won}/{} — above 1 and a \
             negative delta mean the short last tile is the faster of the two: {} row groups \
             over {} column tiles, the last row group the short form on the tail side and a \
             whole 32-row tile on the full one",
            deltas.len(),
            full.tiles,
            full.cols
        )
    } else {
        format!(
            "median_delta_ms={median_delta:+.3} tail_faster={tail_won}/{} the short form ran on \
             neither side — the last row group is not one — so both sides ran the same grid and \
             this ratio is the file's own repeatability floor, not a measurement of the tail",
            deltas.len()
        )
    };
    common::report_rate_pair(PREFIX, full, tail, "full_over_tail", &meaning);
}

/// One test fn on purpose: what it reads is process-global (the route and short-tile
/// censuses and the device submit/wait counters), so two of these in one process would
/// see each other's counts.
#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn gguf_coop_tail_rate_against_the_full_last_tile() {
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
        "{PREFIX} adapter={:?} — these figures hold for this adapter and machine load \
         only",
        rlx_wgpu::device::adapter_name()
    );

    // One cell at a time: both of its sides alive together, timed in alternation, then
    // dropped before the next cell is built. A cell skipped by `build` is named there.
    let mut pairs = 0;
    for cell in CELLS.iter() {
        let Some([mut tail, mut full]) = build(cell, dev) else {
            continue;
        };
        let deltas = run_rounds(&mut tail, &mut full);
        report_pair(&tail, &full, &deltas);
        pairs += 1;
    }
    eprintln!(
        "{PREFIX} pairs={pairs} of {} cells measured — a printed `pairs=0` is no \
         evidence",
        CELLS.len()
    );
}
