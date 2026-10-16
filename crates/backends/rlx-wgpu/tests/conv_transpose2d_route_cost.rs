// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — it prints both routes' wall time and asserts
//! nothing about either.**
//!
//! `Op::ConvTranspose2d`'s wgpu device route is two steps. The gather
//! (`Step::ConvTranspose2d`, printed `conv_transpose2d`) is the sibling 3-D gather
//! with the depth axis pinned to one, and serves the whole operator. Over it, for
//! the unit-height, unit-batch, ungrouped, `s ≤ k ≤ 2·s` family — which is what a
//! 1-D-lifted transposed conv is, and every one of the decoder's six forms — sits
//! the phase-decomposed tiled GEMM (`Step::ConvTranspose2dTiled`, printed
//! `conv_transpose2d_tiled`; `lower.rs`'s `conv_transpose2d_tiled_fits` is its
//! applicability). Both are what the router picks with the structure-host
//! preference off, which this harness forces on any adapter. The route they
//! replaced, still one lever away (`RLX_WGPU_CONV_HOST=1`, or the structure-host
//! preference itself), is a compact BLAS-backed GEMM plus col2im on the host.
//!
//! Moving the op onto the device was **not a pure subtraction** while the gather
//! was the only device form: it is output-centric and scalar, so it walks *every*
//! kernel tap per output element and only the taps whose
//! `(o + pad − k·dil) % stride == 0` land inside the input — at stride `s` about
//! `s`× the tap set the operator strictly needs, the taps outside paying the index
//! arithmetic and the modulo rather than a multiply. The host route pays the
//! opposite way: its staged operands and its readback are a fixed cost per step,
//! which a small geometry cannot amortise but a large one can.
//!
//! Each cell compiles the same graph twice — once under
//! `common::neutralise_device_route_env()`, once with `RLX_WGPU_CONV_HOST=1` —
//! **asserts the route each compile actually took** (`step_kind_counts()`: a
//! measurement that silently timed the wrong route is worthless), warms each arm
//! once untimed, then times `ROUNDS` rounds of `CALLS_PER_ROUND` back-to-back calls
//! per route with `common::time_rounds` and reports the median of the rounds.
//! Nothing asserts a duration: `run()` includes the submission, the device wait and
//! the readback, so a reading is mostly that fixed floor, and it is the machine's,
//! not this file's.
//!
//! Three cells, in the order the run prints them, all group 1, no dilation, no output
//! padding: the decoder geometry the sibling pins use, the tree's large U-Net upsample,
//! and the detection U-Net's own upsample at a small channel count. What the pair reads
//! at them on the fork's own stand (Apple M3 Pro, Metal adapter through wgpu, no thermal
//! warning recorded — `pmset -g therm`):
//!
//! | cell | device/host | arena |
//! |---|---|---|
//! | `decoder-1d-k16s8` — `n=1 c_in=8 c_out=6 h=1 w_in=96 k=[1,16] s=[1,8] p=[0,0]` | 0.45–0.9× (the tiled step's) | byte-identical |
//! | `up2-k3s2` — `n=1 c_in=8 c_out=4 h=w=32 k=[3,3] s=[2,2] p=[1,1]` | ~0.5× | byte-identical |
//! | `large-up2-k3s2` — `n=1 c_in=64 c_out=32 h=w=32 k=[3,3] s=[2,2] p=[1,1]` | within roughly ±20% of parity | byte-identical |
//!
//! The first cell's ratio is the **tiled** step's — the decoder's own geometry is
//! inside the family — and both of its arms sit on a 0.2–0.5 ms-per-call floor, so
//! the ratio moves with the window (0.47× and 0.71× on two of them): re-reading it
//! with each device form in turn (one flip of the family predicate, nothing else
//! changed) left them at 0.37 ms and 0.39 ms of a 0.6–0.7 ms call, so the two steps
//! are not distinguishable on a cell this small and the reading is submission and
//! readback either way. The two U-Net cells are outside the family, so only their
//! ratios stay the gather's, and it is the six real forms below — the shapes the
//! decoder actually carries — that the move is visible on. The `large-up2-k3s2` cell
//! is where the two routes are closest and it has no fixed ratio — four runs on an
//! idle stand gave 1.14×, 1.12×, 1.15× and 1.16×, while a loaded one (both absolute
//! times ~2× larger) put it *below* parity at 0.80×. The host route is CPU BLAS, so a
//! loaded machine moves this cell and no other: read it as "around parity, sign
//! load-dependent" rather than as a number. The arena column is not a timing and does
//! not move — the plan's arenas are byte-identical between the two routes at all
//! three cells (the weight is a `param` either way; only the step reading it differs).
//!
//! The second test here is the six real transposed convolutions of the stand's
//! `dac:25` decoder cell, and it is the one that decides the op's disposition. On
//! the stand the gather measured 8.6× the host route over those six (9–29× per
//! form on the four heavy ones, 4.3–10.5× on the two upsamples), none of which is
//! a cell above, and the tiled form is what makes the device route pay there. The
//! same instrumented cell reads the host route itself at 57.7–62.1 ms a pass,
//! which is what this file's host arm should reproduce. It times the *step*
//! rather than the call — the input resident, nothing read back, on the form's own
//! graph — because at the two light forms the one-op graph's per-call plumbing is
//! the reading, and it reads the six-form aggregate twice: as the sum of the
//! per-form medians, and as all six forms in one graph — one submission a call, the
//! way the cell's pass runs its 370 steps — the two agreeing to ~1 % (20.0 against
//! 20.1–20.3 ms over the stand's quiet windows). Both harnesses are `#[ignore]`d, so
//! the default device suite compiles them and runs nothing; that is the intended
//! state, not a missing gate.
//!
//! Run them:
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
use rlx_wgpu::device::WgpuDevice;

use common::ConvTranspose2dGeometry;

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

/// One of the six real transposed convolutions of the stand's `dac:25` decoder
/// cell, in graph order, read off `common::DECODER_FORMS` — this crate's one copy of
/// those shapes, and the padding the stand's generator derives for them — under the
/// naming the harness builds and sets the form by. The geometry is field for field
/// `conv_transpose2d_parity.rs`'s `Cfg`, so the places that hold these shapes witness
/// the same ones.
struct RealForm {
    label: &'static str,
    geo: ConvTranspose2dGeometry,
}

impl RealForm {
    /// The label with its dots turned into underscores: the graph's own name and its
    /// operand `param`s cannot carry them.
    fn underscored(&self) -> String {
        self.label.replace('.', "_")
    }
    /// The width-axis pad, read off the shared table's `p[1]` rather than derived over
    /// again from the generator's formula.
    fn pad(&self) -> usize {
        self.geo.p[1]
    }
    fn out_pad(&self) -> usize {
        self.geo.op[1]
    }
    /// The output width, from the tree's own shape rule rather than a locally restated
    /// formula.
    fn w_out(&self) -> usize {
        self.geo.out().1
    }
    /// The operator's multiply–accumulates: every output element sums `c_in`
    /// channels over the taps that land on it, `k/s` of them on average.
    fn macs(&self) -> u64 {
        (self.geo.c_out * self.w_out() * self.geo.c_in * (self.geo.k[1] / self.geo.s[1])) as u64
    }
    fn geometry(&self) -> String {
        format!(
            "{} n=1 c_in={} c_out={} h=1 w_in={} k=[1,{}] s=[1,{}] p=[0,{}] op=[0,{}]",
            self.label,
            self.geo.c_in,
            self.geo.c_out,
            self.geo.w,
            self.geo.k[1],
            self.geo.s[1],
            self.pad(),
            self.out_pad()
        )
    }
    /// The form's graph. Both operands are `param`s, `x` included: this harness
    /// measures what a *step* costs, and in the cell the activations are already
    /// resident in the arena — a `run()` that uploaded `x` every call would be
    /// timing a one-op graph's plumbing instead, and at the two light forms that
    /// plumbing would decide the reading.
    fn graph(&self) -> Graph {
        let mut g = Graph::new(format!("ct2d_{}", self.underscored()));
        let y = self.branch(&mut g);
        g.set_outputs(vec![y]);
        g
    }

    /// This form as one branch of `g`: its two `param`s (`x_{label}`, `w_{label}` you
    /// must set) and the conv between them, returned so the caller can make it an
    /// output. The one-form graphs and the six-in-one graph the harness builds share
    /// this, so both arms read the same geometry.
    fn branch(&self, g: &mut Graph) -> rlx_ir::NodeId {
        let x = g.param(
            format!("x_{}", self.underscored()),
            Shape::new(&[1, self.geo.c_in, 1, self.geo.w], DType::F32),
        );
        let w = g.param(
            format!("w_{}", self.underscored()),
            Shape::new(
                &[self.geo.c_in, self.geo.c_out, 1, self.geo.k[1]],
                DType::F32,
            ),
        );
        g.conv_transpose2d(
            x,
            w,
            [1, self.geo.k[1]],
            [1, self.geo.s[1]],
            [0, self.pad()],
            [1, 1],
            [0, self.out_pad()],
            1,
        )
    }
    fn operands(&self) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..self.geo.c_in * self.geo.w)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let w: Vec<f32> = (0..self.geo.c_in * self.geo.c_out * self.geo.k[1])
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        (x, w)
    }
}

/// The six forms of the `dac:25` cell, in graph order (`up.blk.0`, `up.blk.1`,
/// `blk.0`…`blk.3`). On the fork's own stand the host route pays 1.9–21.0 ms for
/// these, 57.7–62.1 ms a pass.
fn real_forms() -> Vec<RealForm> {
    common::DECODER_FORMS
        .iter()
        .map(|&(label, geo)| RealForm { label, geo })
        .collect()
}

/// Rounds and calls for the real-form harness — as above, but ten calls a round
/// because the heavy four forms cost milliseconds each.
const REAL_ROUNDS: usize = 3;
const REAL_CALLS_PER_ROUND: usize = 10;

/// One route's median ms per **step** on `exe` for the six real forms: the input
/// is already resident and the output is never read back, so what is timed is the
/// submission, the step's own work and the wait — no per-call stage of `x` and no
/// per-call round trip for the output, neither of which the cell pays.
///
/// The readback is dropped by the backend's own dispatch-only lever rather than by
/// an empty read plan: `run_read_outputs(&[], Some(&[]))` reads like it would plan
/// no read, but `run_inner` still lays out and maps a zero-byte staging buffer and
/// the map fails. That lever drops the per-call wait with the readback, so the
/// queue is drained once per round and the round divided by its calls — one wait
/// per ten steps, where the cell's pass waits once for 370 of them. Neither arm
/// gains from this: the host step's own staging waits are inside `run` either way,
/// and the device arm's readback would have cost it more than the host's.
fn time_step(exe: &mut WgpuExecutable, dev: &WgpuDevice) -> f64 {
    rlx_ir::env::set("RLX_BENCH_DISPATCH_ONLY", "1");
    let _ = exe.run(&[]);
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    let mut rounds = Vec::with_capacity(REAL_ROUNDS);
    for _ in 0..REAL_ROUNDS {
        let start = rlx_runtime::Tick::now();
        for _ in 0..REAL_CALLS_PER_ROUND {
            let _ = exe.run(&[]);
        }
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        rounds.push(
            rlx_runtime::Tick::now().elapsed_ns(start) as f64 / REAL_CALLS_PER_ROUND as f64 / 1e6,
        );
    }
    rlx_ir::env::unset("RLX_BENCH_DISPATCH_ONLY");
    common::median(&rounds)
}

/// The six real geometries, the device step against the host route — the line the
/// operation's disposition is decided on.
///
/// The cells below are the shapes a router can be *read* on; these six are the
/// shapes the decoder actually carries, and the gather disagreed with the host
/// route on them by 9–29× where it returns on the cells. The tiled step
/// (`conv_transpose2d_tiled`, and its family is `lower.rs`'s
/// `conv_transpose2d_tiled_fits`) is what closes that, so a reading here must
/// name the step it timed: every form asserts its census on both arms. Nothing
/// is asserted about the durations — they are the machine's, not this file's.
///
/// The aggregate is read two ways: as the per-form medians of the loop below, and
/// as all six forms in one graph, so the cost of a call is read back-to-back in one
/// submission rather than summed off six separate graphs. The two should agree, and
/// the print carries both so that a combined reading they disagree on is visible
/// rather than hidden behind the sum.
#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn conv_transpose2d_tiled_against_the_host_route_on_the_decoders_six_forms() {
    // As in the cell harness above: the `RLX_WGPU_CONV_HOST` windows of the two
    // tests in this binary are process-global.
    let _route = common::route_lock();
    common::neutralise_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    common::refuse_under_run_report();
    let adapter = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    let dev = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    eprintln!(
        "wgpu conv_transpose2d on the decoder's six real forms, {adapter}: input resident, no \
         readback, {REAL_ROUNDS} rounds × {REAL_CALLS_PER_ROUND} back-to-back steps per route, \
         queue drained once a round, median of the rounds after one untimed warm-up step"
    );

    let mut device_total = 0.0f64;
    let mut host_total = 0.0f64;
    let mut macs_total = 0u64;
    let forms = real_forms();
    for form in &forms {
        let mut device = WgpuExecutable::compile(form.graph());
        let device_counts = device.step_kind_counts();
        assert_eq!(
            device_counts
                .get("conv_transpose2d_tiled")
                .copied()
                .unwrap_or(0),
            1,
            "{}: the decoder's own forms are what the tiled step's family is for, so a reading \
             that took any other route is not about it; schedule = {device_counts:?}",
            form.label
        );

        rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
        let mut host = WgpuExecutable::compile(form.graph());
        rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
        let host_counts = host.step_kind_counts();
        assert_eq!(
            host_counts
                .get("conv_transpose2d_host")
                .copied()
                .unwrap_or(0),
            1,
            "{}: RLX_WGPU_CONV_HOST=1 must leave the host step this pair is about; schedule = \
             {host_counts:?}",
            form.label
        );

        let (x, w) = form.operands();
        let (x_name, w_name) = (
            format!("x_{}", form.underscored()),
            format!("w_{}", form.underscored()),
        );
        device.set_param(&x_name, &x);
        device.set_param(&w_name, &w);
        host.set_param(&x_name, &x);
        host.set_param(&w_name, &w);
        let device_ms = time_step(&mut device, dev);
        let host_ms = time_step(&mut host, dev);
        device_total += device_ms;
        host_total += host_ms;
        macs_total += form.macs();

        eprintln!(
            "{}: device {device_ms:.3} ms, host {host_ms:.3} ms, device/host {:.2}×, {:.1} \
             GFLOP/s of useful work on the device step — device census {device_counts:?}",
            form.geometry(),
            device_ms / host_ms,
            2.0 * form.macs() as f64 / (device_ms * 1e-3) / 1e9,
        );
    }
    eprintln!(
        "six forms: device {device_total:.3} ms, host {host_total:.3} ms, device/host {:.2}× — \
         {:.1} GFLOP/s over {} M MAC. The cell pays this once per step.",
        device_total / host_total,
        2.0 * macs_total as f64 / (device_total * 1e-3) / 1e9,
        macs_total / 1_000_000,
    );

    // The same six geometries as one graph, so the aggregate is read in one
    // submission a call rather than as a sum over six separate graphs.
    let one_pass = || {
        let mut g = Graph::new("ct2d_real_forms_one_pass");
        let mut outputs = Vec::with_capacity(forms.len());
        for form in &forms {
            outputs.push(form.branch(&mut g));
        }
        g.set_outputs(outputs);
        g
    };
    let mut device_one_pass = WgpuExecutable::compile(one_pass());
    let device_one_pass_counts = device_one_pass.step_kind_counts();
    assert_eq!(
        device_one_pass_counts
            .get("conv_transpose2d_tiled")
            .copied()
            .unwrap_or(0),
        forms.len(),
        "every branch of the one-pass graph must be the tiled step; schedule = \
         {device_one_pass_counts:?}"
    );
    rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
    let mut host_one_pass = WgpuExecutable::compile(one_pass());
    rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
    let host_one_pass_counts = host_one_pass.step_kind_counts();
    assert_eq!(
        host_one_pass_counts
            .get("conv_transpose2d_host")
            .copied()
            .unwrap_or(0),
        forms.len(),
        "RLX_WGPU_CONV_HOST=1 must leave the host step on every branch; schedule = \
         {host_one_pass_counts:?}"
    );
    for form in &forms {
        let (x, w) = form.operands();
        let (x_name, w_name) = (
            format!("x_{}", form.underscored()),
            format!("w_{}", form.underscored()),
        );
        device_one_pass.set_param(&x_name, &x);
        device_one_pass.set_param(&w_name, &w);
        host_one_pass.set_param(&x_name, &x);
        host_one_pass.set_param(&w_name, &w);
    }
    let one_pass_device_ms = time_step(&mut device_one_pass, dev);
    let one_pass_host_ms = time_step(&mut host_one_pass, dev);
    eprintln!(
        "six forms, one graph: device {one_pass_device_ms:.3} ms, host {one_pass_host_ms:.3} ms, \
         device/host {:.2}× — the per-form medians summed to device {device_total:.3} ms, host \
         {host_total:.3} ms",
        one_pass_device_ms / one_pass_host_ms,
    );
}

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn conv_transpose2d_device_step_against_the_host_route_it_replaced() {
    // The `RLX_WGPU_CONV_HOST` window below is process-global, so this file's two
    // tests must not bracket one another.
    let _route = common::route_lock();
    common::neutralise_device_route_env();
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
        // Either device form is this harness's business — the tiled fast path for
        // the family it is proven for, the gather for the rest — but a compile
        // that took the host route would be timing something else. Which form
        // each cell got is printed with its reading rather than asserted, because
        // a cell is a shape a router can be read on, not a form pin; the real
        // decoder forms below are where the form is pinned.
        let device_steps = device_counts.get("conv_transpose2d").copied().unwrap_or(0)
            + device_counts
                .get("conv_transpose2d_tiled")
                .copied()
                .unwrap_or(0);
        assert_eq!(
            device_steps, 1,
            "{}: with the structure-host preference off the route must be a device step; \
             schedule = {device_counts:?}",
            cell.label
        );
        assert!(
            !device_counts.contains_key("conv_transpose2d_host"),
            "{}: the host step must not also be in the schedule; schedule = {device_counts:?}",
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
