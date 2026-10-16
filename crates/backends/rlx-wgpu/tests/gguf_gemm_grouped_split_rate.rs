// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The row-grouped GGUF GEMM at the row counts a model body's own forms have:
//! what one step reads, how many times it reads it, the dispatch the encoder
//! chooses for it, and what a call costs.
//!
//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test gguf_gemm_grouped_split_rate -- --ignored --nocapture
//! ```
//!
//! # Steps, calls and the batch join
//!
//! **A step** is one `Op::DequantMatMul` over one packed weight region and one
//! `m x k` activation; **a call** is one run of a graph holding [`STEPS`] of them
//! — eight identical multiplies in one graph. Every cell here is a multi-row call
//! over whole packing blocks of its scheme — the staged tile itself for Q4_K/Q6_K,
//! eight of them for Q8_0 — which the ladder answers as the row-grouped arm, and
//! that answer is **in** the batch boundary's join set: a call's steps ride the
//! enclosing batch's submission instead of each opening one of its own, so what a
//! call costs in submissions and waits is printed per cell rather than asserted
//! against a fixed shape.
//!
//! # Interleaving, and the aggregate footprint
//!
//! Every cell is built, warmed and censused up front and **held alive for the
//! whole run**, and the timed rounds then run **round-robin across cells**: one
//! round gives every cell its own [`CALLS_PER_ROUND`] calls back to back before
//! the next round starts, so no cell's whole timing block runs before another's
//! starts. A fixed-order loop — one cell's whole block at a time — puts the
//! process's warm-up and the machine's drift on whichever cells come first.
//! Interleaving is what makes a cell's figure its own rather than its position's.
//!
//! All of them alive at once costs `STEPS x packed_bytes` of device memory per
//! cell, summed: the figure this file prints before the timed rounds. A cell the
//! encoder would **not** dispatch as one chunk is named and skipped, with every
//! cell that does fit still measured and still interleaved.
//!
//! The fixture, the ramp and the round-robin timing protocol are the crate's
//! shared ones ([`common::packed_gguf_weight`], [`common::ramp`],
//! [`common::time_round_robin`]); what stays here is the cell set.
//!
//! # The numerator, and the two rates
//!
//! `packed_bytes_per_step` is the tensor's own packed bytes — the encoder's own
//! per-row stride ([`gguf_row_bytes`]) times its column count — and both printed
//! rates come from it. **Neither is a DRAM counter**: no counter in this tree
//! reports bytes read.
//!
//! * `GBps_at_the_weight` = `packed_bytes_per_step * STEPS / call_seconds` — the
//!   rate a plain read of the same tensor would be quoted at. It is the
//!   denominator a cache does not help (the bytes have to reach the kernel once
//!   either way).
//! * `GBps_over_reads` = `row_groups x GBps_at_the_weight` — the
//!   **address-space** figure: every row group is charged for the packed bytes it
//!   requests, whether or not the adapter serves it from a cache. This is what the
//!   route costs in `m`, not what the memory system pays.
//!
//! Both rates are figures of the **packed bytes alone**, so they are the same
//! quantity at both geometries. `weight_bytes_read_per_step` is not: it adds the
//! split geometry's own extra, whose two half-`k` threads read disjoint halves of a
//! block's quant bytes but each read the block's header — one more header per
//! block per column than a base step reads, which is the difference between the
//! byte figure and `row_groups x packed_bytes_per_step`.
//!
//! # What is asserted, and what is printed
//!
//! Asserted, never a time: every step of the untimed call takes the row-grouped
//! route (the process-global census, reset and read around that one call); the
//! warm-up's outputs are finite and each is `m * n` long; and the fixture's packed
//! length equals `n` times the encoder's per-row stride ([`gguf_row_bytes`]) — an
//! identity between two spellings of the same block geometry, since
//! `common::packed_gguf_weight` lays out the same `k / block_elems x block_bytes`
//! per row, so it checks the fixture's own shape and **not** that the encoder's
//! stride has not moved.
//!
//! Skipped rather than asserted: a cell whose `n` exceeds the encoder's own column
//! cap ([`grouped_dispatch_cols`], both of its budgets) is named and dropped — this
//! file will not print a whole-tensor geometry the encoder does not dispatch. That
//! skip can silently empty the run, taking every cell and the process-global
//! census's only witness with it, so a printed `cells=0` is no evidence at all.
//!
//! Printed: the geometry, the flag, and the base grid — [`grouped_base_geometry`],
//! the same arithmetic the dispatch comes from, so the comparison is never
//! arithmetic this file did itself; the row-group multiplicity and the byte volume
//! it means; the submission and wait counts one call costs; and the per-round
//! times with both rates.
//!
//! Not in this cell's sight, and not to be read as checked: whether the row
//! groups' re-reads hit an adapter cache or DRAM (no in-tree counter separates
//! them); the dequant arithmetic — the parity suites cover values; the per-chunk
//! host work (one uniform write and one bind group per chunk per step); the
//! chunked shapes' later chunks; and any adapter but the one printed.
//!
//! # The band, and where it is recorded
//!
//! The ceiling that decides the flag is a measured one, and its band — what the
//! split wins below it, what it still wins above it, and where it starts to lose —
//! is recorded on `GROUPED_SPLIT_MAX_BASE_WORKGROUPS` in `gguf_gpu.rs`. That band is
//! the author's own **out-of-tree** raw-kernel A/B of the two geometries over the
//! same packed bytes, run on one adapter, and is to be re-taken on the adapter at
//! hand rather than trusted. **This file is not that A/B**: it measures the geometry
//! the encoder itself picks for each shape — one geometry per shape — and prints no
//! split-versus-base pair, so it reads out what the encoder chose at whatever
//! `gguf_gpu.rs` currently says rather than confirming the band. It does name the
//! adapter it ran on ([`rlx_wgpu::device::adapter_name`] below; `--nocapture` is what
//! makes any of it visible).
//!
//! # Cells
//!
//! Each row of [`CELLS`] is a call of [`STEPS`] = 8 multiplies, and the cell's own
//! comment says why it is there: the bodies' row counts against the widths their
//! forms are chunked at, the two sides of the split ceiling, and the two schemes
//! whose entry points have no split arm. What the file prints per cell is its
//! whole record — the geometry the encoder picked, the bytes a step reads and the
//! per-call times — so the table is not repeated here.

mod common;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::gguf_gpu::{
    DequantMatmulRoutes, gguf_row_bytes, grouped_base_geometry, grouped_chunk_geometry,
    grouped_dispatch_cols, reset_route_counts, route_counts,
};
use rlx_wgpu::gguf_host::gguf_scheme_id;
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

/// The cells. A `static` rather than a `const` so a cell built up front can hold
/// the entry it was built from for the whole run, the registry outliving every
/// borrow of it.
static CELLS: [Cell; 15] = [
    // The bodies' own H = 1024 projection at the smallest body row count — its
    // base grid sits exactly on the split ceiling (16 x 2 = 32 workgroups of 64
    // columns).
    Cell {
        label: "q4k body m13 k1024 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 13,
        k: 1024,
        n: 1024,
    },
    // The same rows against I = 6144 columns: six times the x grid, past the
    // ceiling on columns alone.
    Cell {
        label: "q4k ffn-out m13 k1024 n6144",
        scheme: QuantScheme::GgufQ4K,
        m: 13,
        k: 1024,
        n: 6144,
    },
    // The same 1024 columns with six times the contraction: the grid is a
    // function of m and n only, so this is the split cell at the wider k.
    Cell {
        label: "q4k ffn-in m13 k6144 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 13,
        k: 6144,
        n: 1024,
    },
    // One row group more than the 13-row cell: 48 workgroups, the base geometry.
    Cell {
        label: "q4k body m24 k1024 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 24,
        k: 1024,
        n: 1024,
    },
    Cell {
        label: "q4k ffn-in m24 k6144 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 24,
        k: 6144,
        n: 1024,
    },
    Cell {
        label: "q4k body m48 k1024 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 48,
        k: 1024,
        n: 1024,
    },
    Cell {
        label: "q4k ffn-in m48 k6144 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 48,
        k: 6144,
        n: 1024,
    },
    Cell {
        label: "q4k body m100 k1024 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 100,
        k: 1024,
        n: 1024,
    },
    Cell {
        label: "q4k ffn-in m100 k6144 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 100,
        k: 6144,
        n: 1024,
    },
    // Thirteen row groups exactly (104 = 13 x 8): the same group count as the
    // 100-row cell, so the two differ only in how full the last group is.
    Cell {
        label: "q4k body m104 k1024 n1024",
        scheme: QuantScheme::GgufQ4K,
        m: 104,
        k: 1024,
        n: 1024,
    },
    // The two sides of the ceiling, narrower than the bodies' H: the base grid is
    // 10 x 2 = 20 workgroups (the split geometry's 20 x 2 = 40), below the ceiling.
    Cell {
        label: "q4k band below m13 k1024 n640",
        scheme: QuantScheme::GgufQ4K,
        m: 13,
        k: 1024,
        n: 640,
    },
    // ... and the base grid is 30 x 2 = 60 workgroups of 64 columns, above it.
    Cell {
        label: "q4k band above m13 k1024 n1920",
        scheme: QuantScheme::GgufQ4K,
        m: 13,
        k: 1024,
        n: 1920,
    },
    // Q6_K's entry point has no split arm; the same shape as the Q4_K body cell
    // so the two schemes' geometries are read at one shape.
    Cell {
        label: "q6k body m13 k1024 n1024",
        scheme: QuantScheme::GgufQ6K,
        m: 13,
        k: 1024,
        n: 1024,
    },
    // Q8_0 likewise, and at the Q6_K cells' shape.
    Cell {
        label: "q8_0 body m13 k1024 n1024",
        scheme: QuantScheme::GgufQ8_0,
        m: 13,
        k: 1024,
        n: 1024,
    },
    // The third scheme at 72 rows and a 3k contraction against H = 1024 columns
    // (the `wave72` the label names): nine row groups, the most groups any Q8_0
    // form here takes.
    Cell {
        label: "q8_0 wave72 m72 k3072 n1024",
        scheme: QuantScheme::GgufQ8_0,
        m: 72,
        k: 3072,
        n: 1024,
    },
];

/// Timed rounds, and calls per round per cell: the per-round figure is that
/// round's own mean per call, and the spread across rounds is reported as it
/// stands rather than smoothed away, so run-to-run spread is information about
/// the shape. The rounds are the round-robin the module doc describes: one round
/// is every cell's own [`CALLS_PER_ROUND`] calls, not one cell's.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 3;

/// The cell's graph: [`STEPS`] grouped multiplies, each over its own parameter and
/// all of them over one input matrix.
fn cell_graph(cell: &Cell, packed_bytes: usize) -> Graph {
    let mut g = Graph::new("wgpu_gguf_gemm_grouped_split_rate");
    let x = g.input("x", Shape::new(&[cell.m, cell.k], DType::F32));
    let mut outs = Vec::with_capacity(STEPS);
    for s in 0..STEPS {
        let w = g.param(format!("w{s}"), Shape::new(&[packed_bytes], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul {
                scheme: cell.scheme,
            },
            vec![x, w],
            Shape::new(&[cell.m, cell.n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    g
}

/// One call's steps share this matrix — see the module doc.
fn input_matrix(m: usize, k: usize) -> Vec<f32> {
    common::ramp(m, k, 0.017)
}

/// One cell built, warmed, censused and asserted — held alive for the whole run
/// rather than built and dropped on its own, so the rounds can interleave it with
/// every other cell (the module doc's "Interleaving, and the aggregate
/// footprint").
struct Ready {
    /// The entry this was built from: alive in [`CELLS`] for the whole run.
    cell: &'static Cell,
    packed_bytes_per_step: usize,
    row_groups: u64,
    weight_bytes_read_per_step: u64,
    /// The encoder's own answer for the cell: `(x, y)` workgroups and the flag.
    grid: (u32, u32, bool),
    /// The base geometry's own grid for the same chunk — what the Q6_K and Q8_0
    /// entries dispatch, and the comparison the split grid is read against.
    base_grid: (u32, u32),
    routes: DequantMatmulRoutes,
    submits: u64,
    waits: u64,
    x: Vec<f32>,
    compiled: CompiledGraph,
    /// This cell's mean ms per call, one entry per round.
    per_call_ms: Vec<f64>,
}

/// One cell's fixture, compile, warm-up, finiteness census and structural
/// assertions — everything but the timing, which interleaves the cells so it
/// cannot live here. `None`: the cell was **named as not one chunk and skipped**
/// rather than measured, the later cells still being measured.
fn build(cell: &'static Cell, dev: &rlx_wgpu::device::WgpuDevice) -> Option<Ready> {
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
    // whole-tensor geometry it does not dispatch.
    let chunk_cols =
        grouped_dispatch_cols(&dev.device, gguf_scheme_id(cell.scheme), cell.m, cell.k);
    if chunk_cols < cell.n {
        eprintln!(
            "[grouped split rate] cell=\"{}\" is not one chunk: the encoder covers {chunk_cols} of \
             its {} columns per dispatch — skipped rather than measured as a shape whose printed \
             geometry it is not",
            cell.label, cell.n
        );
        return None;
    }

    // The encoder's own rule for the grid and the flag, and its own base grid for
    // the comparison — one spelling each, read from the module rather than
    // re-derived here.
    let (x_groups, y_groups, split) =
        grouped_chunk_geometry(gguf_scheme_id(cell.scheme), cell.m, cell.n);
    let (base_x, base_y) = grouped_base_geometry(cell.m, cell.n);
    // The route reads each packed byte once per group of the kernel's `TM` = 8
    // rows, which is exactly the geometry's y grid — the split flag changes the
    // columns one workgroup covers, not the row groups. A split step additionally
    // requests every block's header once per half-`k` lane group, i.e. twice, of
    // which the packed count above charges one: Q4_K packs one nibble per element
    // and its block header is 16 bytes, so a column's header bytes are its row
    // stride less `k / 2` (a `k` that is not a whole number of 256-element blocks
    // is declined by the encoder before this, so the subtraction is defined).
    let row_groups = y_groups as u64;
    let header_extra_per_step = if split {
        cell.n as u64 * row_bytes.saturating_sub(cell.k as u64 / 2)
    } else {
        0
    };
    let weight_bytes_read_per_step =
        row_groups * (packed_bytes_per_step as u64 + header_extra_per_step);

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

    let mut compiled = Session::new(Device::Gpu).compile(cell_graph(cell, packed.len()));
    for s in 0..STEPS {
        compiled.set_param_typed(&format!("w{s}"), &packed, DType::U8);
    }
    let x = input_matrix(cell.m, cell.k);

    let warm = compiled.run(&[("x", &x)]);
    assert_eq!(warm.len(), STEPS, "{}: one output per step", cell.label);
    for (s, out) in warm.iter().enumerate() {
        assert_eq!(
            out.len(),
            cell.m * cell.n,
            "{}: step {s} output shape",
            cell.label
        );
        let non_finite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "{}: step {s} produced {non_finite} non-finite values — the fixture's \
             scale slots are not the finite patterns it writes",
            cell.label
        );
    }

    // The census: which route, how many submissions and waits — one call's own
    // counts, taken outside the timing, and taken while this cell is the only one
    // running.
    reset_route_counts();
    reset_device_counts();
    let _ = compiled.run(&[("x", &x)]);
    let routes = route_counts();
    let (submits, waits) = (device_submit_count(), device_wait_count());
    assert_eq!(
        (routes.grouped, routes.per_row, routes.host),
        (STEPS as u64, 0, 0),
        "{}: every step must take the row-grouped route: {routes:?} — every figure this cell \
         would print is work that route did, so the cell is not measured as this route's",
        cell.label
    );
    Some(Ready {
        cell,
        packed_bytes_per_step,
        row_groups,
        weight_bytes_read_per_step,
        grid: (x_groups, y_groups, split),
        base_grid: (base_x, base_y),
        routes,
        submits,
        waits,
        x,
        compiled,
        per_call_ms: Vec::with_capacity(ROUNDS),
    })
}

/// The timed rounds, round-robin across the cells the module doc describes: one
/// round gives every cell its own [`CALLS_PER_ROUND`] calls back to back before
/// the next round starts, so each cell's figures are its own rather than its
/// position's.
fn run_rounds(cells: &mut [Ready]) {
    let per_cell = common::time_round_robin(cells, ROUNDS, CALLS_PER_ROUND, |r| {
        let _ = r.compiled.run(&[("x", r.x.as_slice())]);
    });
    for (r, times) in cells.iter_mut().zip(per_cell) {
        r.per_call_ms = times;
    }
}

/// One cell's printed fields, both rates and its per-round figures.
fn report(r: &Ready) {
    let cell = r.cell;
    let (x_groups, y_groups, split) = r.grid;
    let (base_x, base_y) = r.base_grid;
    let per_call_ms = &r.per_call_ms;
    let min_ms = per_call_ms.iter().cloned().fold(f64::MAX, f64::min);
    let med_ms = common::median(per_call_ms);
    let max_ms = per_call_ms.iter().cloned().fold(0.0f64, f64::max);
    // The rates: the tensor read once (`at_the_weight`) and every row group
    // charged for it (`over_reads`) — see the module doc for both.
    let at_the_weight =
        |call_ms: f64| r.packed_bytes_per_step as f64 * STEPS as f64 / (call_ms * 1e-3) / 1e9;
    let (routes, submits, waits) = (r.routes, r.submits, r.waits);

    eprintln!(
        "[grouped split rate] cell=\"{}\" scheme={:?} m={} k={} n={} steps={} \
         packed_bytes_per_step={} row_groups={} weight_bytes_read_per_step={}",
        cell.label,
        cell.scheme,
        cell.m,
        cell.k,
        cell.n,
        STEPS,
        r.packed_bytes_per_step,
        r.row_groups,
        r.weight_bytes_read_per_step,
    );
    eprintln!(
        "[grouped split rate] cell=\"{}\" grid={x_groups}x{y_groups} workgroups split={split} \
         base_grid={base_x}x{base_y} workgroups",
        cell.label,
    );
    eprintln!(
        "[grouped split rate] cell=\"{}\" routes={routes:?} submits={submits} waits={waits}",
        cell.label
    );
    eprintln!(
        "[grouped split rate] cell=\"{}\" per_call_ms=[{}] min_ms={min_ms:.3} \
         median_ms={med_ms:.3} max_ms={max_ms:.3} spread_pct={:.1} \
         min_GBps_at_the_weight={:.1} median_GBps_at_the_weight={:.1} \
         min_GBps_over_reads={:.1} median_GBps_over_reads={:.1}",
        cell.label,
        per_call_ms
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        (max_ms - min_ms) / med_ms * 100.0,
        at_the_weight(min_ms),
        at_the_weight(med_ms),
        r.row_groups as f64 * at_the_weight(min_ms),
        r.row_groups as f64 * at_the_weight(med_ms),
    );
}

/// One test fn on purpose: what it reads is process-global (the route census and
/// the device submit/wait counters), and the round-robin needs every cell's
/// device memory alive at once rather than one cell's gone before the next is
/// built.
#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn gguf_grouped_split_rate_at_the_body_shapes() {
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
        "[grouped split rate] adapter={:?} — this cell's figures hold for this adapter and \
         machine load only",
        rlx_wgpu::device::adapter_name()
    );

    // Every cell built, warmed and censused up front and held for the whole run:
    // that is the interleaving's cost, and the aggregate below is the number the
    // module doc names.
    let mut cells: Vec<Ready> = CELLS.iter().filter_map(|cell| build(cell, dev)).collect();
    let aggregate_packed_bytes: u64 = cells
        .iter()
        .map(|r| r.packed_bytes_per_step as u64 * STEPS as u64)
        .sum();
    eprintln!(
        "[grouped split rate] cells={} aggregate_packed_bytes={aggregate_packed_bytes} — every \
         cell held alive at once for the round-robin; a cell skipped above is named there",
        cells.len()
    );

    run_rounds(&mut cells);

    for r in &cells {
        report(r);
    }
}
