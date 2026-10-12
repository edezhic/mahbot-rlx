// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight reuse, measured, for the row-grouped GGUF GEMM at the widest served
//! shape: a ~800 MiB packed Q4_K weight (and the same shape's Q6_K side), wider
//! than one bind window, so the encoder has to re-resolve and re-bind the weight
//! several times per call. The route exists to read the packed weight's quant bytes
//! once per group of 8 rows instead of once per row; a "pass" is the single-row
//! cell's own cost, so every ratio here is denominated in the one-row arm.
//!
//! As recorded before that arm's reshape, one group cost 1.45-1.50 passes at 2-4
//! rows and 1.49-1.54 at 8 (Q6_K: 1.36-1.43), with the 96- and 276-row cells at
//! `ceil(m / 8)` = 12 and 35 such groups (16.5-17.2x and 48-50x) — short of the
//! ~1.2x the kernel was designed for and straddling the ~1.5x this route was
//! asked to reach. After it, with the grouped kernel untouched, the same shape
//! reads **1.72-1.82x at 2-8 rows and 1.80-1.85x on the Q6_K side**, and
//! 20.2x / 58.8x at 96 and 276 rows. Those are the figures of one run, not a
//! guard: later runs have printed 1.75-1.83 and 1.81-1.86, and 20.6x / 60.0x,
//! and every cell is printed by every run.
//!
//! # The ratio is no longer asserted
//!
//! The movement above belongs to this ratio's denominator: the one-row arm's
//! reshape (a workgroup per two output columns sharing one X read, the Q6_K lane
//! unit halved) took the single-row cell from ~12.3 ms to ~9.5 ms at this shape,
//! with the grouped kernel untouched. This file used to fail a call whose row group
//! cost more than `ceil(m / 8) x 1.6` single-row passes, a bound sized to the
//! measured band plus a margin; denominated *in the single-row arm*, it is tightened
//! by that arm getting faster for reasons that have nothing to do with the grouped
//! route, and the m=2 cell's ratio went to 1.77, past it. Rather than re-denominate
//! the bound in a unit that would cancel the improvement, it was **retired**: the
//! band above is still printed every run, nothing asserts it, and the grouped route
//! therefore has **no in-tree regression ceiling**. What stands in its place is the
//! route census this loop still asserts (m=1 per-row, m>1 one grouped dispatch), the
//! CPU-parity check below, and the owner's rig, whose own re-take of the grouped
//! cells is what covers a regression here.
//!
//! # Measurement
//!
//! Both sides of the ratio are the same instrument, in this one process, in a
//! release build, on one uploaded weight: each cell compiles its own graph and
//! uploads the packed bytes once, every cell runs [`WARMUP`] untimed calls, and
//! then the cells are timed in [`ROUNDS`] round-robin rounds — one call each per
//! round — with each cell reporting the minimum and the median of its own calls.
//! Round-robin rather than one cell's whole run at a time because this crate's
//! A/B protocol notes (`rlx-metal/examples/schedule_sgemm_ab.rs`,
//! `rlx-wgpu/examples/wgpu_schedule_matmul_ab.rs`) record a first-touch ordering
//! bias: on Metal a byte-identical control reported 1.24x, and every later arm
//! looked up to 20% faster than it was.
//!
//! The single-row side (`m = 1`, the tiled per-row GEMV the grouped route
//! replaces) is the first cell of the shape and is the denominator of every
//! ratio, so the table states both absolutes: the ratio is the grouped cell's
//! minimum over the single-row cell's minimum, both measured in this session with
//! the same order and the same warm-up.
//!
//! The weight is in the act arena here: the executable only moves a param into the
//! separate packed-weight buffer when the plan exceeds the device binding limit —
//! 4 GiB of packed weights on this hardware — which a single 777 MiB weight does
//! not reach. `gguf_gemm_parked_weight.rs` drives the split arena (harness size and
//! kilobyte size) directly instead.
//!
//! `#[ignore]`d — it materialises ~800 MiB (Q4_K) + ~1.1 GiB (Q6_K) of packed
//! weights on the host, and the interleaved cells hold their uploaded copies at
//! once: ~5.3 GiB of device arena for the seven Q4_K cells and ~4.4 GiB for the
//! four Q6_K ones, above the output slabs. Shrink the shape list on a smaller
//! machine, and run it deliberately:
//!
//! ```text
//! cargo test -p rlx-wgpu --release --test gguf_gemm_grouped_weight_reuse --features backend-tests \
//!     -- --ignored --nocapture
//! ```
//!
//! Its own binary, and one test fn, so the process-global route counters have no
//! concurrent writers.

mod common;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::gguf_gpu::{DequantMatmulRoutes, reset_route_counts, route_counts};

/// Untimed calls before the measured ones, per cell.
const WARMUP: usize = 3;

/// Timed rounds, one call per cell per round. See the measurement section of the
/// module doc for why round-robin rather than a long run per cell.
const ROUNDS: usize = 9;

/// The two K-quant schemes the row-grouped route serves, each with the entry
/// point that decodes its blocks.
struct Scheme {
    quant: QuantScheme,
    entry: &'static str,
}

const Q4K: Scheme = Scheme {
    quant: QuantScheme::GgufQ4K,
    entry: "dequant_gemm_q4k",
};
const Q6K: Scheme = Scheme {
    quant: QuantScheme::GgufQ6K,
    entry: "dequant_gemm_q6k",
};

/// One `DequantMatMul` over `packed` (`[n, k]`) and its row-major `x`.
fn graph(scheme: &Scheme, k: usize, n: usize, m: usize, packed_len: usize) -> (Graph, Vec<f32>) {
    let x: Vec<f32> = common::ramp(m, k, 1e-4);
    let mut g = Graph::new("wgpu_gguf_weight_reuse");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed_len], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: scheme.quant,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, x)
}

/// What one cell measured, and the route its calls took.
struct Measured {
    /// Fastest of the timed calls.
    min_ms: f64,
    median_ms: f64,
    routes: DequantMatmulRoutes,
}

/// One measured cell: compile the `m`-row graph against the shared weight,
/// upload once, then hand back the live executable so the caller can interleave
/// its calls with the other cells'.
fn prepared(
    scheme: &Scheme,
    packed: &[u8],
    k: usize,
    n: usize,
    m: usize,
) -> (CompiledGraph, Vec<f32>) {
    let (g, x) = graph(scheme, k, n, m, packed.len());
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    (c, x)
}

/// Measure every cell round-robin: one timed call of each, [`ROUNDS`] times over.
///
/// The single-row arm and the grouped arm are the two sides of the ratio, and on
/// this backend a sequential A/B is not trustworthy (see the module doc). Round
/// robin gives both sides the same device state, the same warm-up and the same
/// number of calls, and the report is each cell's minimum over the rounds. The
/// loop itself is [`common::time_round_robin`]; what stays here is this cell set's
/// warm-up and its route census.
fn measure_round_robin(cells: &mut [(usize, CompiledGraph, Vec<f32>)]) -> Vec<(usize, Measured)> {
    for (_, c, x) in cells.iter_mut() {
        for _ in 0..WARMUP {
            let _ = c.run(&[("x", x)]);
        }
    }
    let times = common::time_round_robin(cells, ROUNDS, 1, |(_, c, x)| {
        let _ = c.run(&[("x", x)]);
    });
    cells
        .iter_mut()
        .enumerate()
        .map(|(i, (m, c, x))| {
            // The route census of this cell alone: reset, one call, read.
            reset_route_counts();
            let _ = c.run(&[("x", x)]);
            let routes = route_counts();
            (
                *m,
                Measured {
                    min_ms: times[i].iter().cloned().fold(f64::MAX, f64::min),
                    median_ms: common::median(&times[i]),
                    routes,
                },
            )
        })
        .collect()
}

/// Measure one scheme's weight at the widest served shape and assert each cell's
/// route census, then its parity against the per-row path at 4 rows.
fn measure_scheme(scheme: &Scheme, k: usize, n: usize, ms: &[usize]) {
    let packed = common::packed_gguf_weight(scheme.quant, k, n);
    let mut cells: Vec<(usize, CompiledGraph, Vec<f32>)> = ms
        .iter()
        .map(|&m| {
            let (c, x) = prepared(scheme, &packed, k, n, m);
            (m, c, x)
        })
        .collect();
    let measured = measure_round_robin(&mut cells);
    drop(cells);
    // The single-row cell is the denominator of every ratio: it is reported
    // beside each row so a ratio measured against a slowed single-row arm cannot
    // read as an improvement of the grouped kernel.
    let t1 = measured[0].1.min_ms;
    eprintln!(
        "weight reuse ({} entry `{}` k={k} n={n}, {} MiB packed + bind-cap windows, \
         min/median of {ROUNDS} interleaved calls after {WARMUP} warm-up): {}",
        scheme.quant,
        scheme.entry,
        packed.len() / (1024 * 1024),
        measured
            .iter()
            .map(|(m, v)| format!(
                "m={m} {:.1}/{:.1} ms ({:.2}x)",
                v.min_ms,
                v.median_ms,
                v.min_ms / t1
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for &(m, ref v) in &measured {
        let r = &v.routes;
        let want = if m == 1 { (0, 1) } else { (1, 0) };
        assert_eq!(
            (r.grouped, r.per_row),
            want,
            "m={m} must take {}: {r:?}",
            if m == 1 {
                "the per-row GEMV"
            } else {
                "one grouped dispatch"
            }
        );
    }

    // Accuracy against today's path — the same rows computed one at a time (the m=1
    // GEMV) — at the full 202048-column width, where the column chunks and their
    // copy-back are all in play. Same absolute 1e-2 the packed-multiply CPU-parity
    // checks use, through the shared [`common::max_abs_diff`], which censuses both
    // outputs for NaN/inf before folding — `f32::max` returns the non-NaN operand,
    // so a poisoned output would otherwise fold to a small number and pass. The
    // reference's own magnitude is printed so the margin is visible.
    // `gguf_matmul_route.rs` pins the same comparison on a small weight in the
    // default run.
    let (g, x) = graph(scheme, k, n, 4, packed.len());
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let _ = c.run(&[("x", &x)]);
    let grouped = c.run(&[("x", &x)]).remove(0);
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for r in 0..4 {
        let (g1, _) = graph(scheme, k, n, 1, packed.len());
        let mut c1 = Session::new(Device::Gpu).compile(g1);
        c1.set_param_typed("w", &packed, DType::U8);
        let row_x = &x[r * k..(r + 1) * k];
        let _ = c1.run(&[("x", row_x)]);
        let one = c1.run(&[("x", row_x)]).remove(0);
        worst = worst.max(common::max_abs_diff(&grouped[r * n..(r + 1) * n], &one));
        // `one` is finite — `max_abs_diff` above censuses it — so this fold of the
        // reference's own magnitude cannot hide a NaN either.
        scale = one.iter().fold(scale, |m, v| m.max(v.abs()));
    }
    eprintln!(
        "{} (`{}`) grouped m=4 vs per-row m=1 at k={k} n={n}: max_abs={worst:e} \
         (per-row |y|max={scale:e})",
        scheme.quant, scheme.entry
    );
    // The bound below is absolute, so a degenerate reference would satisfy it
    // vacuously — the same guard `gguf_gemm_parked_weight.rs` carries.
    assert!(
        scale > 1.0,
        "{}: per-row reference is degenerate ({scale:e})",
        scheme.quant
    );
    assert!(
        worst <= 1e-2,
        "{} grouped vs per-row max_abs {worst:e} > 1e-2",
        scheme.quant
    );
}

#[test]
#[ignore = "777 MiB + 1.1 GiB of packed weights (~5.3 GiB of device copies for \
            Q4_K, ~4.4 GiB for Q6_K); run with --ignored --nocapture"]
fn a_row_group_reuses_the_packed_weight() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    common::refuse_under_run_report();
    // The widest served shape: 202048 x 7168 — 777 MiB packed as Q4_K, 1.1 GiB as
    // Q6_K. The 256 MiB bind cap splits each weight into its own column chunks,
    // standing in for a device whose binding limit is under the weight size.
    //
    // Both schemes the row-grouped route serves are measured, and the ratio the
    // print reports beside each cell is against the single-row (m=1) cell's own
    // cost in the same session — the denominator the module doc describes. 2 is
    // the smallest batch that has to be served and 276 the widest matrix; the
    // Q6_K weight is half as large again, so its 200-600 ms cells are left to the
    // Q4_K side.
    let (k, n) = (7168usize, 202048usize);
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "256");
    measure_scheme(&Q4K, k, n, &[1, 2, 3, 4, 8, 96, 276]);
    measure_scheme(&Q6K, k, n, &[1, 2, 4, 8]);

    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");

    // The group path's output slab is real device memory the arena plan does not
    // own, so it has to be visible to the backend's memory report — and capped, so a
    // wide head at a few thousand rows cannot allocate to the binding limit. 256 MiB
    // is that cap here (the widest served shape needs 213 MiB of it).
    let pooled = rlx_wgpu::gguf_gpu::pooled_out_buf_bytes();
    assert!(
        pooled > 0 && pooled <= 256 * 1024 * 1024,
        "pooled out slab {pooled} bytes must be reported and inside the 256 MiB cap"
    );
}
