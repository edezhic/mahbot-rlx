// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — it prints both routes' wall time and asserts
//! nothing about either.**
//!
//! `Op::ConvTranspose2d`'s wgpu route is the device gather step (`Step::ConvTranspose2d`,
//! printed `conv_transpose2d`), the sibling 3-D gather with the depth axis pinned to
//! one — what the router picks with the structure-host preference off, which this
//! harness forces on any adapter. The route it replaced, still one lever away
//! (`RLX_WGPU_CONV_HOST=1`, or the structure-host preference itself), is a compact
//! BLAS-backed GEMM plus col2im on the host.
//!
//! Moving the op onto the device is **not a pure subtraction**: the gather is
//! output-centric and scalar, so it walks *every* kernel tap per output element and
//! only the taps whose `(o + pad − k·dil) % stride == 0` land inside the input — at
//! stride `s` about `s`× the tap set the operator strictly needs, the taps outside
//! paying the index arithmetic and the modulo rather than a multiply. The host route
//! pays the opposite way: its staged operands and its readback are a fixed cost per
//! step, which a small geometry cannot amortise but a large one can.
//!
//! Each cell compiles the same graph twice — once under
//! `common::conv_transpose2d_device_route_env()`, once with `RLX_WGPU_CONV_HOST=1` —
//! **asserts the route each compile actually took** (`step_kind_counts()`, `conv_transpose2d`
//! against `conv_transpose2d_host`: a measurement that silently timed the wrong route is
//! worthless), warms each arm once untimed, then times `ROUNDS` rounds of
//! `CALLS_PER_ROUND` back-to-back calls per route with `common::time_rounds` and reports
//! the median of the rounds. Nothing asserts a duration: `run()` includes the submission,
//! the device wait and the readback, so a reading is mostly that fixed floor, and it is
//! the machine's, not this file's.
//!
//! Three cells, in the order the run prints them, all group 1, no dilation, no output
//! padding: the decoder geometry the sibling pins use, the tree's large U-Net upsample,
//! and the detection U-Net's own upsample at a small channel count. What the pair reads
//! at them on the fork's own stand (Apple M3 Pro, Metal adapter through wgpu, no thermal
//! warning recorded — `pmset -g therm`):
//!
//! | cell | device/host | arena |
//! |---|---|---|
//! | `decoder-1d-k16s8` — `n=1 c_in=8 c_out=6 h=1 w_in=96 k=[1,16] s=[1,8] p=[0,0]` | 0.5–0.8× | byte-identical |
//! | `up2-k3s2` — `n=1 c_in=8 c_out=4 h=w=32 k=[3,3] s=[2,2] p=[1,1]` | ~0.5× | byte-identical |
//! | `large-up2-k3s2` — `n=1 c_in=64 c_out=32 h=w=32 k=[3,3] s=[2,2] p=[1,1]` | within roughly ±20% of parity | byte-identical |
//!
//! The two small cells are the stable ones: the device gather returns there, because the
//! host route's staged operands and readback are a fixed cost neither can amortise. The
//! U-Net cell is where the two routes are closest and it has no fixed ratio — four runs
//! on an idle stand gave 1.14×, 1.12×, 1.15× and 1.16×, while a loaded one (both
//! absolute times ~2× larger) put it *below* parity at 0.80×. The host route is CPU
//! BLAS, so a loaded machine moves this cell and no other: read it as "around parity,
//! sign load-dependent" rather than as a number. The arena column is not a timing and
//! does not move — the plan's arenas are byte-identical between the two routes at all
//! three cells (the weight is a `param` either way; only the step reading it differs).
//!
//! Run it (it is `#[ignore]`d, so the default device suite compiles it and runs
//! nothing — that is the intended state, not a missing gate):
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests --test conv_transpose2d_route_cost \
//!     -- --ignored --nocapture
//! ```
//!
//! Check the thermal gate first (`just throttle`, or `pmset -g therm` by hand): the
//! fork's rule applies here as everywhere — a thermally throttled machine's numbers
//! are not the numbers, so a run under a reduced `CPU_Speed_Limit` has to be repeated
//! rather than reported (`RLX_ALLOW_THROTTLE=1` accepts one as a one-off).

mod common;

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// Rounds and calls per round, both printed with the results: five rounds of twenty
/// back-to-back calls per route, the median of the five. A call at these shapes is
/// dominated by submission, the device wait and the readback, so twenty back-to-back
/// calls are long enough that one round is not mostly the timer and five rounds are
/// enough to see the machine drift under the median.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 20;

/// One measurement cell: the geometry both routes are compiled and run at, under the
/// label the run prints it by.
struct Cell {
    label: &'static str,
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w_in: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    pad: [usize; 2],
}

impl Cell {
    /// The cell's graph: the weight is a `param` (what the host route stages and what
    /// the device route keeps in the arena), the activation a graph input.
    fn graph(&self) -> Graph {
        let mut g = Graph::new("conv_transpose2d_route_cost");
        let x = g.input(
            "x",
            Shape::new(&[self.n, self.c_in, self.h, self.w_in], DType::F32),
        );
        let w = g.param(
            "w",
            Shape::new(
                &[self.c_in, self.c_out, self.kernel[0], self.kernel[1]],
                DType::F32,
            ),
        );
        let y = g.conv_transpose2d(x, w, self.kernel, self.stride, self.pad, [1, 1], [0, 0], 1);
        g.set_outputs(vec![y]);
        g
    }

    /// The cell's geometry as printed, so a reading can be read against the shape it
    /// came from without the reader counting fields in the label.
    fn geometry(&self) -> String {
        format!(
            "n={} c_in={} c_out={} {}×{} k={}×{} s={}×{} p={}×{}",
            self.n,
            self.c_in,
            self.c_out,
            self.h,
            self.w_in,
            self.kernel[0],
            self.kernel[1],
            self.stride[0],
            self.stride[1],
            self.pad[0],
            self.pad[1]
        )
    }

    /// The cell's operands, filled deterministically the way the neighbouring pins
    /// fill theirs. Nothing in the measurement reads the values.
    fn operands(&self) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..self.n * self.c_in * self.h * self.w_in)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let w: Vec<f32> = (0..self.c_in * self.c_out * self.kernel[0] * self.kernel[1])
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        (x, w)
    }
}

/// The three cells, in the order the run prints them — the decoder's own geometry
/// where the staging dominates, the tree's large upsample shape, and the shipped
/// stride-2 upsample at a small channel count (see the module doc).
const CELLS: &[Cell] = &[
    Cell {
        label: "decoder-1d-k16s8",
        n: 1,
        c_in: 8,
        c_out: 6,
        h: 1,
        w_in: 96,
        kernel: [1, 16],
        stride: [1, 8],
        pad: [0, 0],
    },
    Cell {
        label: "large-up2-k3s2",
        n: 1,
        c_in: 64,
        c_out: 32,
        h: 32,
        w_in: 32,
        kernel: [3, 3],
        stride: [2, 2],
        pad: [1, 1],
    },
    Cell {
        label: "up2-k3s2",
        n: 1,
        c_in: 8,
        c_out: 4,
        h: 32,
        w_in: 32,
        kernel: [3, 3],
        stride: [2, 2],
        pad: [1, 1],
    },
];

/// One route's median ms/call on `exe`: one untimed warm-up run first (the first
/// `run` materialises the pooled output slab and stages the uniforms, and neither is
/// the per-call cost), then [`ROUNDS`] rounds of [`CALLS_PER_ROUND`] calls, each
/// round's mean per call, and the median of the rounds.
fn time_route(exe: &mut WgpuExecutable, x: &[f32]) -> f64 {
    let _ = exe.run(&[("x", x)]);
    let rounds = common::time_rounds(ROUNDS, CALLS_PER_ROUND, || {
        let _ = exe.run(&[("x", x)]);
    });
    common::median(&rounds)
}

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn conv_transpose2d_device_gather_against_the_host_route_it_replaced() {
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    common::refuse_under_run_report();
    let adapter = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!(
        "wgpu conv_transpose2d route cost on {adapter}: {} cells, {ROUNDS} rounds × \
         {CALLS_PER_ROUND} calls per route, median of the rounds after one untimed warm-up run",
        CELLS.len()
    );

    for cell in CELLS {
        // The device route: what the router picks with the structure-host
        // preference off, and every other router that could take the decision away
        // from it zeroed.
        let mut device = WgpuExecutable::compile(cell.graph());
        let device_counts = device.step_kind_counts();
        assert_eq!(
            device_counts.get("conv_transpose2d").copied().unwrap_or(0),
            1,
            "{}: with the structure-host preference off the route must be the device \
             gather; schedule = {device_counts:?}",
            cell.label
        );

        // The host route: the same graph, one lever away, unset again before anything
        // is timed so the device arm's runs cannot see it.
        rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
        let mut host = WgpuExecutable::compile(cell.graph());
        rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
        let host_counts = host.step_kind_counts();
        assert_eq!(
            host_counts
                .get("conv_transpose2d_host")
                .copied()
                .unwrap_or(0),
            1,
            "{}: RLX_WGPU_CONV_HOST=1 must leave the host step this route replaced; \
             schedule = {host_counts:?}",
            cell.label
        );

        let (x, w) = cell.operands();
        device.set_param("w", &w);
        host.set_param("w", &w);
        let device_ms = time_route(&mut device, &x);
        let host_ms = time_route(&mut host, &x);

        // `test_arena_bytes()` is the plan's visible half — the doc-hidden test hook the
        // sibling harnesses read as well.
        eprintln!(
            "{}: {} — device {device_ms:.4} ms/call, host {host_ms:.4} ms/call, device/host \
             {:.2}×, arena device {:.2} MiB / host {:.2} MiB — device census {device_counts:?}, \
             host census {host_counts:?}",
            cell.label,
            cell.geometry(),
            device_ms / host_ms,
            device.test_arena_bytes() as f64 / (1u64 << 20) as f64,
            host.test_arena_bytes() as f64 / (1u64 << 20) as f64
        );
    }
}
