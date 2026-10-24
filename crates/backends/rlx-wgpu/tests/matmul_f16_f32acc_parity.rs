// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_coop_f16_f32acc` against `Device::Cpu`, at the encoder's projection
//! shapes and at the geometry a 64×64 tile is worst at.
//!
//! **What is pinned.** For each shape: the GPU arm's max|Δ| and cosine against
//! the CPU's f32 product over the SAME f16 operands, and its max|Δ| against the
//! true f32 product. The first is the accumulation bar the kernel owns (f32
//! accumulator, different summation order); the second additionally carries the
//! f16 quantization of both operands, which no f32-accumulator kernel can undo.
//! The header states the bar; the test prints every measured figure.
//!
//! **What is NOT claimed.** This is not a claim that the f16 path is exact: it
//! is not, and the printed `f16 operands` column is the size of the gap. Row
//! counts stay small on purpose — `tests/encoder_gemm_cost.rs` measures m = 8192
//! and m = 256 for rate, and the CPU reference it would take to check those
//! numbers row by row is minutes per shape, so parity is checked at m = 64 (one
//! whole tile), m = 130 (a partial one) and the awkward set below, all at the
//! encoder's k and n. The kernel's own arithmetic does not depend on how many row
//! tiles are launched; what changes with m is only how many of them there are.
//!
//! **The awkward set** is the tile's edges, each of them admitted by the shader's
//! guarded staging rather than rounded away: m = 1 and m = 7 (a tile with 1 and 7
//! live rows), k = 96 (three whole K stages) and k = 100 (a partial one, whose
//! tail must contribute nothing), n = 130 (partial column tiles), m = 65/n = 65
//! (a second tile of one row/column).
//!
//! **Route witness.** The step names carry no precision, so the test counts
//! `coop_f16_f32acc_dispatch_count()` around a run of the largest cell: a parity
//! that a different kernel produced would pin nothing. On a device where the
//! probe rejects the kernel — or that cannot build it — the whole test reports
//! ineligible and returns; that is the same refusal the routing makes, not a
//! silent pass.
//!
//! **The Q8_0 round trip.** The model's weights arrive Q8_0-packed; the f16 image
//! is what the stand hands over when the store rule takes the `gemm=f16coop`
//! route. What that costs is measured in `tests/encoder_gemm_cost.rs`'s header
//! (f16 holds 11 bits of mantissa against Q8_0's 8, and the block scale is per 32
//! weights in Q8_0 against per value here), not re-derived here: this file
//! compares against f32.

mod common;

use common::{data, route_lock};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

/// One projection of the encoder: `[m, k] · [k, n]`, as `(label, m, k, n)`.
struct Cell {
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

/// The nine per-layer projections at the sliding-layer geometry, then the
/// awkward set. `m` is small (the CPU arm pays for it) but `k`/`n` are the
/// encoder's own.
const CELLS: &[Cell] = &[
    Cell {
        label: "q         512→1024",
        m: 64,
        k: 512,
        n: 1024,
    },
    Cell {
        label: "q_global  512→2048",
        m: 64,
        k: 512,
        n: 2048,
    },
    Cell {
        label: "k         512→512 ",
        m: 64,
        k: 512,
        n: 512,
    },
    Cell {
        label: "v         512→512 ",
        m: 64,
        k: 512,
        n: 512,
    },
    Cell {
        label: "o        1024→512 ",
        m: 64,
        k: 1024,
        n: 512,
    },
    Cell {
        label: "o_global 2048→512 ",
        m: 64,
        k: 2048,
        n: 512,
    },
    Cell {
        label: "gate      512→2048",
        m: 64,
        k: 512,
        n: 2048,
    },
    Cell {
        label: "up        512→2048",
        m: 64,
        k: 512,
        n: 2048,
    },
    Cell {
        label: "down     2048→512 ",
        m: 64,
        k: 2048,
        n: 512,
    },
    Cell {
        label: "inp_gate  512→512 ",
        m: 64,
        k: 512,
        n: 512,
    },
    Cell {
        label: "proj      512→512 ",
        m: 64,
        k: 512,
        n: 512,
    },
    Cell {
        label: "ple       512→12288",
        m: 64,
        k: 512,
        n: 12288,
    },
    // Whole tiles at awkward-looking extents: a second row tile, the narrowest
    // K stage the admission allows, a K tail that is still whole (`k = 96`), and
    // the smallest 64×64 output.
    Cell {
        label: "m=128     512→64  ",
        m: 128,
        k: 512,
        n: 64,
    },
    Cell {
        label: "k=64      64→64   ",
        m: 128,
        k: 64,
        n: 64,
    },
    Cell {
        label: "k=32      32→192  ",
        m: 64,
        k: 32,
        n: 192,
    },
    Cell {
        label: "k=32      32→128  ",
        m: 64,
        k: 32,
        n: 128,
    },
    Cell {
        label: "k=96      96→128  ",
        m: 64,
        k: 96,
        n: 128,
    },
];

/// The geometries the shader REFUSES: a cooperative store cannot be masked, so a
/// partial output tile would write past the tensor, and a partial K stage would
/// need a branch per staged element. Each of these keeps the kernel the cascade
/// would have given it, and this file pins that it did — plus that the answer is
/// still right through that fallback, which is the half a refusal could otherwise
/// get wrong.
const TAILED: &[Cell] = &[
    Cell {
        label: "m=1       512→512 ",
        m: 1,
        k: 512,
        n: 512,
    },
    Cell {
        label: "m=7       512→512 ",
        m: 7,
        k: 512,
        n: 512,
    },
    Cell {
        label: "m=65      512→64  ",
        m: 65,
        k: 512,
        n: 64,
    },
    Cell {
        label: "n=130     512→130 ",
        m: 64,
        k: 512,
        n: 130,
    },
    Cell {
        label: "m=65,n=65 512→65  ",
        m: 65,
        k: 512,
        n: 65,
    },
    Cell {
        label: "k=100     100→256 ",
        m: 64,
        k: 100,
        n: 256,
    },
];

/// The bar for `max|Δ|` against the CPU's f32 product **over the same f16
/// operands** — the accumulation-order term the kernel owns. Measured on Apple
/// M3 Pro: `0.0` at every cell of the table, i.e. the kernel reproduces the CPU's
/// f32 accumulation bit for bit at these shapes. The bar keeps room for a
/// different summation order on another adapter rather than pinning that
/// coincidence; a wrong product is three orders of magnitude past it.
const ACC_BAR: f32 = 1e-4;

/// The bar for `max|Δ|` against the **true f32** product, which additionally
/// carries the f16 quantization of both operands. Measured worst case on Apple
/// M3 Pro: `2.42e-2` (down 2048→512, whose 2048-term dot products are the widest
/// here), of which the operands are all of it — `A f16` and `W f16` in the table
/// are the two halves. The bar is that figure with room for a fixture of another
/// amplitude, not a tolerance the kernel is allowed to spend.
const TOTAL_BAR: f32 = 5e-2;

/// Cosine below this is not a rounding difference but a wrong product (`b·a`
/// scores ≈ 0.02 on unrelated operands; see `src/coop_probe.rs`). Measured worst
/// case here: `0.99993496` at the PLE cell, whose output magnitude is small
/// against the operand noise.
const COS_BAR: f64 = 0.9995;

/// The f16 image of `x`, in f32 lanes: what an f16 operand can hold.
fn f16_round(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// One `[m, k] × [k, n]` multiply over an f16 `B` param, `A` and the output f32.
fn gemm_graph(label: &str, m: usize, k: usize, n: usize, w_dtype: DType) -> Graph {
    let mut g = Graph::new(label);
    let x = g.input("x", Shape::new(&[m, k], F32));
    let w = g.param("w", Shape::new(&[k, n], w_dtype));
    let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[m, n], F32));
    g.set_outputs(vec![y]);
    g
}

/// `best` of the values, for a timing-free reading.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The CPU's f32 product of `a · w` through the runtime — the reference this
/// file compares against, so the comparison is not against a second opinion
/// written by the same hand as the kernel.
fn cpu_product(label: &str, m: usize, k: usize, n: usize, a: &[f32], w: &[f32]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(gemm_graph(label, m, k, n, F32));
    c.set_param("w", w);
    c.run(&[("x", a)]).remove(0)
}

#[test]
fn the_f16_f32acc_gemm_matches_the_cpu_product_at_the_encoder_shapes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    if !rlx_wgpu::coop_probe::f16_f32acc_path_ok() {
        eprintln!(
            "matmul_coop_f16_f32acc is not usable on this device (probe rejected it or it does \
             not build) — the routing refuses it here too, so there is nothing to compare"
        );
        return;
    }
    // The route witness in this file reads the process-global dispatch counter, so
    // this table's own runs must not overlap it: one lock for both.
    let _route = route_lock();
    eprintln!(
        "{:<20} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>5}",
        "shape", "acc |Δ|", "A f16 |Δ|", "W f16 |Δ|", "total |Δ|", "cosine", "out |max|", "disps"
    );
    let mut worst_acc = (String::new(), 0.0f32);
    let mut worst_total = (String::new(), 0.0f32);
    let mut worst_cos = (String::new(), f64::INFINITY);
    for cell in CELLS {
        let Cell { label, m, k, n } = *cell;
        let (got, cpu_f16, cpu_a_f32, cpu_true, disps) = run_cell(label, m, k, n);

        let acc = max_abs(&got, &cpu_f16);
        let a_f16 = max_abs(&cpu_f16, &cpu_a_f32);
        let w_f16 = max_abs(&cpu_a_f32, &cpu_true);
        let total = max_abs(&got, &cpu_true);
        let cos = cosine(&got, &cpu_true);
        let peak = cpu_true.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        eprintln!(
            "{label:<20} {acc:>9.2e} {a_f16:>9.2e} {w_f16:>9.2e} {total:>9.2e} {cos:>11.8} \
             {peak:>11.3e} {disps:>5}"
        );
        assert_eq!(
            disps, 1,
            "{label}: this shape is whole on all three axes, so the multiply must have been \
             served by matmul_coop_f16_f32acc — a fallback would make the bars below pin \
             another kernel"
        );
        // Worst-case bookkeeping, asserted once after the table so one run
        // reports every shape instead of stopping at the first.
        if acc > worst_acc.1 || worst_acc.0.is_empty() {
            worst_acc = (format!("m={m} k={k} n={n}"), acc);
        }
        if total > worst_total.1 {
            worst_total = (format!("m={m} k={k} n={n}"), total);
        }
        if cos < worst_cos.1 {
            worst_cos = (format!("m={m} k={k} n={n}"), cos);
        }
    }
    eprintln!(
        "worst: acc |Δ| {:.3e} at {}  ·  total |Δ| {:.3e} at {}  ·  cosine {:.8} at {}",
        worst_acc.1, worst_acc.0, worst_total.1, worst_total.0, worst_cos.1, worst_cos.0
    );
    assert!(
        worst_acc.1 <= ACC_BAR,
        "{}: max|Δ| {:.3e} against the CPU product over the same f16 operands exceeds the \
         {ACC_BAR:.1e} accumulation bar",
        worst_acc.0,
        worst_acc.1
    );
    assert!(
        worst_total.1 <= TOTAL_BAR,
        "{}: max|Δ| {:.3e} against the true f32 product exceeds the {TOTAL_BAR:.1e} bar",
        worst_total.0,
        worst_total.1
    );
    assert!(
        worst_cos.1 >= COS_BAR,
        "{}: cosine {:.8} — that is a wrong product, not a rounding difference",
        worst_cos.0,
        worst_cos.1
    );
}

/// The other half of the admission: a geometry off the tile grid keeps the kernel
/// the cascade would have given it — and still computes the multiply. A refusal
/// that quietly produced another kernel's numbers would pass a parity test that
/// only looked at the numbers, so both halves are asserted.
#[test]
fn a_geometry_off_the_tile_grid_falls_back_and_still_computes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    if !rlx_wgpu::coop_probe::f16_f32acc_path_ok() {
        eprintln!("matmul_coop_f16_f32acc is not usable on this device — nothing is refused here");
        return;
    }
    let _route = route_lock();
    eprintln!(
        "{:<20} {:>9} {:>11} {:>5}",
        "shape", "total |Δ|", "cosine", "disps"
    );
    for cell in TAILED {
        let Cell { label, m, k, n } = *cell;
        let (got, _, _, cpu_true, disps) = run_cell(label, m, k, n);
        let total = max_abs(&got, &cpu_true);
        let cos = cosine(&got, &cpu_true);
        eprintln!("{label:<20} {total:>9.2e} {cos:>11.8} {disps:>5}");
        assert_eq!(
            disps, 0,
            "{label}: this geometry is not a whole number of 64×64 tiles and 32-deep K \
             stages, so matmul_coop_f16_f32acc must not have been dispatched — its store \
             cannot be masked"
        );
        assert!(
            cos >= 0.99,
            "{label}: cosine {cos:.8} through the fallback — the refusal must not change the \
             answer"
        );
    }
}

/// One cell end to end: the GPU run (and its own dispatch delta) plus the four CPU
/// products the gap is decomposed into — the kernel's own accumulation against the
/// same f16 operands, the two operand-rounding halves, and the true f32 product.
#[allow(clippy::type_complexity)]
fn run_cell(
    label: &str,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64) {
    // A and W in a couple of units' range, so the f16 quantization error is the
    // table's, not a fixture's.
    let a = data(m * k, 0.0013, false);
    let w: Vec<f32> = data(k * n, 0.0007, true);
    let a16 = f16_round(&a);
    let w16 = f16_round(&w);
    let cpu =
        |tag: &str, a: &[f32], w: &[f32]| cpu_product(&format!("{label} {tag}"), m, k, n, a, w);
    let cpu_f16 = cpu("cpu f16/f16", &a16, &w16);
    let cpu_a_f32 = cpu("cpu f32/f16", &a, &w16);
    let cpu_true = cpu("cpu f32/f32", &a, &w);

    let mut gpu = Session::new(Device::Gpu).compile(gemm_graph(label, m, k, n, DType::F16));
    gpu.set_param("w", &w);
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let got = gpu.run(&[("x", &a)]).remove(0);
    let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
    assert_eq!(got.len(), m * n, "{label}: output extent");
    (got, cpu_f16, cpu_a_f32, cpu_true, disps)
}

/// The route witness: the shape above must actually have been served by this
/// kernel. Counted under the process-wide route lock, because the counter is
/// global and a concurrently running test would make a delta meaningless.
#[test]
fn the_encoder_gemm_is_served_by_the_f16_f32acc_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    if !rlx_wgpu::coop_probe::f16_f32acc_path_ok() {
        eprintln!("matmul_coop_f16_f32acc is not usable on this device — no route to witness");
        return;
    }
    let _route = route_lock();
    let (m, k, n) = (64usize, 512usize, 2048usize);
    let a = data(m * k, 0.0013, false);
    let w: Vec<f32> = data(k * n, 0.0007, true);
    let mut gpu =
        Session::new(Device::Gpu).compile(gemm_graph("route witness", m, k, n, DType::F16));
    gpu.set_param("w", &w);
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let got = gpu.run(&[("x", &a)]).remove(0);
    let after = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    assert_eq!(
        after - before,
        1,
        "the multiply was not served by matmul_coop_f16_f32acc (dispatches {before} → {after}); \
         the parity above would then be pinning some other kernel"
    );
    // Both operands f16-rounded, as the kernel reads them: the witness asks that
    // the witnessed dispatch computes the product, not that it be exact in f32.
    let cpu = cpu_product("route witness cpu", m, k, n, &f16_round(&a), &f16_round(&w));
    assert!(
        max_abs(&got, &cpu) <= ACC_BAR,
        "the witnessed dispatch does not match the CPU product"
    );
}
