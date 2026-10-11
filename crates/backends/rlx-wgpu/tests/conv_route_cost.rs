// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The accelerated conv routes — im2col+GEMM, the mirrored 1-D tiled relabel and
//! the 1×1 pointwise GEMM — are on by default under one switch
//! (`RLX_WGPU_CONV_IM2COL`; `=0` restores the plain routing). This harness is the
//! marginal-cost evidence for that flip over the decoder-body conv shapes the
//! forward-parity suite pins (`conv2d_groups_parity.rs`), plus one dilated
//! mirrored shape the relabel has to carry and one shape it must refuse: per family
//! (`Conv1x7`, `Conv1x1`) it prints the wall time of one `WgpuExecutable::run` on
//! each arm of the switch, then the same protocol per *single shape* — the
//! family schedule is the union over its shapes, so a shape the route leaves
//! behind hides in that union, while a one-shape graph names the route that
//! shape alone takes. It also prints the compiled schedule (arena MiB +
//! step-name table, which names the routes taken), the cost of the reduced-dispatch
//! (active-extent) mode the routes veto — once per veto on its own graph: the
//! dense body conv's `im2col_gpu` step and the 1×1 pointwise conv's
//! `conv1x1_gemm` step, each probe naming the step that vetoes the mode — and
//! the plan-invariance check — a conv-free graph must plan identically under
//! both arms, arena included — and the col-scratch reservation a shape-qualifying
//! conv pays even where its route does not run (a parked weight).
//!
//! Nothing here asserts a duration. `run()` includes submission, the device wait
//! and the readback, so a single reading is mostly that fixed floor; the table is
//! a direction, not a gate. Only facts that can be stated exactly are asserted: the
//! route each body shape takes on each arm, that the off arm's schedules carry no
//! `im2col_gpu` step, that a graph whose convolutions never qualify keeps its plan
//! *and* its arena under both arms, and that a shape-qualifying conv pays the col
//! reservation even where its route then does not run (a parked weight).
//!
//! Run the thermal gate first (`just throttle`).

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::Tick;
use rlx_wgpu::backend::WgpuExecutable;
use std::collections::BTreeMap;

mod common;

/// One unit-height 1-D conv of the reproduced decoder body: `[1, C_in, 1, L]`
/// input, kernel `k`, stride `s`, padding `p`, per-axis dilation `d`, `groups`.
#[derive(Clone, Copy)]
struct BodyConv {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    l: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    groups: usize,
}

/// The 7-tap (`Conv1x7`) family: the nine dense/sub-threshold shapes with the
/// length taps in slot 1, the two depthwise shapes, the ONNX lift with the taps
/// in slot 0, and one dilated mirrored shape — the shapes the forward-parity
/// suite added plus the dilation axis the mirrored relabel has to carry.
const CONV1X7: [BodyConv; 12] = [
    BodyConv {
        label: "dense-l800",
        c_in: 768,
        c_out: 768,
        l: 800,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "dense-l4000",
        c_in: 384,
        c_out: 384,
        l: 4000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "dense-l16000",
        c_in: 192,
        c_out: 192,
        l: 16000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "dense-l48000",
        c_in: 96,
        c_out: 96,
        l: 48000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "mirror-projection",
        c_in: 96,
        c_out: 1,
        l: 48000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "mirror-strided-s2",
        c_in: 768,
        c_out: 32,
        l: 4000,
        k: [1, 7],
        s: [1, 2],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "entry-l100",
        c_in: 1024,
        c_out: 1536,
        l: 100,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "depthwise-l50",
        c_in: 1024,
        c_out: 1024,
        l: 50,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1024,
    },
    BodyConv {
        label: "depthwise-l100",
        c_in: 1024,
        c_out: 1024,
        l: 100,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1024,
    },
    BodyConv {
        label: "overlap-slot0",
        c_in: 384,
        c_out: 384,
        l: 4000,
        k: [7, 1],
        s: [1, 1],
        p: [3, 0],
        d: [1, 1],
        groups: 1,
    },
    // The only shape here that exercises the dilation axis through the mirrored
    // relabel: the out length is 994, under the 2048 im2col spatial floor — so
    // im2col does not take it — but over the 256 tiled floor, so it lands on the
    // mirrored relabel, which copies the slot-1 dilation into the tiled kernel's
    // `dh` (2 here, the stride-2 case above only covers `sh`).
    BodyConv {
        label: "mirror-dilated-d2",
        c_in: 384,
        c_out: 384,
        l: 1000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 2],
        groups: 1,
    },
    // The mirror must be REFUSED here, and the route table below pins that: the
    // height axis carries padding, so H is a real axis — the relabel would move
    // it onto the length axis and resize the output (3×512 instead of 512). The
    // output spatial (1536) is deliberately *above* the 256 tiled floor and below
    // the 2048 im2col floor, so this row discriminates: were the refusal lost,
    // the shape would relabel into the tiled route and the assertion below would
    // fail instead of the row printing a plausible number (with spatial below the
    // tiled floor both outcomes are `conv2d` and the row is inert).
    BodyConv {
        label: "refuse-height-pad",
        c_in: 64,
        c_out: 64,
        l: 512,
        k: [1, 7],
        s: [1, 1],
        p: [1, 3],
        d: [1, 1],
        groups: 1,
    },
];

/// The pointwise (`Conv1x1`) family.
const CONV1X1: [BodyConv; 4] = [
    BodyConv {
        label: "l800",
        c_in: 768,
        c_out: 768,
        l: 800,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "l4000",
        c_in: 384,
        c_out: 384,
        l: 4000,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "l16000",
        c_in: 192,
        c_out: 192,
        l: 16000,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "l48000",
        c_in: 96,
        c_out: 96,
        l: 48000,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
];

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.2
        })
        .collect()
}

/// A graph carrying every conv of one family as an independent node (its own
/// input and parameter), so one compile covers the whole family.
struct BodyGraph {
    graph: Graph,
    params: Vec<(String, Vec<f32>)>,
    inputs: Vec<(String, Vec<f32>)>,
    labels: Vec<&'static str>,
}

fn build_family(name: &'static str, shapes: &[BodyConv]) -> BodyGraph {
    let mut g = Graph::new(name);
    let mut params = Vec::new();
    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    let mut outs = Vec::new();
    for (i, c) in shapes.iter().enumerate() {
        let xn = format!("x{i}");
        let wn = format!("w{i}");
        let x = g.input(&xn, Shape::new(&[1, c.c_in, 1, c.l], DType::F32));
        let w = g.param(
            &wn,
            Shape::new(&[c.c_out, c.c_in / c.groups, c.k[0], c.k[1]], DType::F32),
        );
        outs.push(g.conv2d(x, w, c.k, c.s, c.p, c.d, c.groups));
        inputs.push((xn, fill(c.c_in * c.l, i as u64)));
        params.push((
            wn,
            fill(
                c.c_out * (c.c_in / c.groups) * c.k[0] * c.k[1],
                i as u64 + 97,
            ),
        ));
        labels.push(c.label);
    }
    g.set_outputs(outs);
    BodyGraph {
        graph: g,
        params,
        inputs,
        labels,
    }
}

/// One compiled arm of the switch: the executable plus the inputs to hand it.
struct Arm {
    label: &'static str,
    exe: WgpuExecutable,
    inputs: Vec<(String, Vec<f32>)>,
}

impl Arm {
    fn new(label: &'static str, bg: &BodyGraph) -> Self {
        let mut exe = WgpuExecutable::compile(bg.graph.clone());
        for (n, d) in &bg.params {
            exe.set_param(n, d);
        }
        Self {
            label,
            exe,
            inputs: bg.inputs.clone(),
        }
    }

    /// One timed `run`, in microseconds. `Tick` is this tree's convention at this
    /// scale — `CNTVCT_EL0` on Apple silicon, `Instant` elsewhere, as in
    /// `fused_residual_norm_cost.rs` — and the shapes are small enough that a
    /// single reading is the measurement.
    fn run_us(&mut self) -> u128 {
        self.run_us_out().0
    }

    /// The same reading, keeping the outputs so the two arms of the switch can
    /// be compared elementwise.
    fn run_us_out(&mut self) -> (u128, Vec<Vec<f32>>) {
        let refs: Vec<(&str, &[f32])> = self
            .inputs
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let start = Tick::now();
        let out = self.exe.run(&refs);
        (u128::from(Tick::now().elapsed_ns(start)) / 1000, out)
    }
}

/// The route a *single-shape* graph's schedule took: exactly one of the conv route
/// names is present, and `conv1x1_gemm` doubles as the route's GEMM step.
fn conv_route(counts: &BTreeMap<&'static str, usize>) -> &'static str {
    for r in [
        "im2col_gpu",
        "conv2d_tiled",
        "conv1x1_gemm",
        "conv2d",
        "conv1d_tiled",
    ] {
        if counts.contains_key(r) {
            return r;
        }
    }
    panic!("no conv route in {counts:?}");
}

/// The route each body shape must take on each arm of the switch, `(off, on)`:
/// a shape whose route silently disappears would otherwise show up only as a
/// 1.00× row in a harness nobody diffs. The two arms are pinned together, so the
/// table also states which shape moved (the `overlap` row: already accelerated
/// today, onto the other accelerated route), which shapes stay on the direct
/// kernel under both arms (`entry`, the two `depthwise`, and the shape the mirror
/// must *refuse*), and that a lost refusal cannot pass unnoticed.
fn expected_routes(family: &str, label: &str) -> (&'static str, &'static str) {
    match (family, label) {
        ("Conv1x1", _) => ("conv2d", "conv1x1_gemm"),
        (_, "dense-l800") => ("conv2d", "conv2d_tiled"),
        (_, "dense-l4000" | "dense-l16000" | "dense-l48000") => ("conv2d", "im2col_gpu"),
        (_, "mirror-projection" | "mirror-strided-s2" | "mirror-dilated-d2") => {
            ("conv2d", "conv2d_tiled")
        }
        (_, "overlap-slot0") => ("conv2d_tiled", "im2col_gpu"),
        (_, "entry-l100" | "depthwise-l50" | "depthwise-l100" | "refuse-height-pad") => {
            ("conv2d", "conv2d")
        }
        other => panic!("no expected route for {other:?} — the body shape list grew"),
    }
}

fn fmt_counts(counts: &BTreeMap<&'static str, usize>) -> String {
    counts
        .iter()
        .map(|(n, c)| format!("{c}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn set_arm(off: bool) {
    if off {
        rlx_ir::env::set("RLX_WGPU_CONV_IM2COL", "0");
    } else {
        rlx_ir::env::unset("RLX_WGPU_CONV_IM2COL");
    }
}

/// One graph measured under both arms of the switch: the medians, each arm's
/// step-name table (which names the routes it took) and the outputs of the last
/// round.
struct Pair {
    off_us: u128,
    on_us: u128,
    off_counts: BTreeMap<&'static str, usize>,
    on_counts: BTreeMap<&'static str, usize>,
    off_out: Vec<Vec<f32>>,
    on_out: Vec<Vec<f32>>,
}

/// The harness protocol, once: compile both arms, print each arm's step-name
/// table, then one warm-up run per arm and 3 rounds with the arms interleaved
/// round-by-round (every round printed) and the median taken. `scope` prefixes
/// every line so a per-shape run's detail sits under its own label.
fn measure_pair(scope: &str, bg: &BodyGraph) -> Pair {
    set_arm(true);
    let mut off = Arm::new("off", bg);
    set_arm(false);
    let mut on = Arm::new("on", bg);
    let off_counts = off.exe.step_kind_counts();
    let on_counts = on.exe.step_kind_counts();
    eprintln!(
        "  {scope}: arm off {} steps [{}]",
        off_counts.values().sum::<usize>(),
        fmt_counts(&off_counts)
    );
    eprintln!(
        "  {scope}: arm on  {} steps [{}]",
        on_counts.values().sum::<usize>(),
        fmt_counts(&on_counts)
    );
    assert!(
        !off_counts.contains_key("im2col_gpu"),
        "{scope}: the off arm must not route a conv through im2col_gpu"
    );
    // Warm-up: first touch of the arena and each pipeline.
    let _ = off.run_us();
    let _ = on.run_us();
    let (mut off_t, mut on_t) = (Vec::new(), Vec::new());
    let (mut off_out, mut on_out) = (Vec::new(), Vec::new());
    for round in 0..3 {
        // Alternate which arm runs first so a drift cannot favour one.
        let ((o, oo), (n, no)) = if round % 2 == 0 {
            (off.run_us_out(), on.run_us_out())
        } else {
            let n = on.run_us_out();
            let o = off.run_us_out();
            (o, n)
        };
        eprintln!("  {scope} round {round}: off {o} µs  on {n} µs");
        off_t.push(o);
        on_t.push(n);
        (off_out, on_out) = (oo, no);
    }
    let (off_us, on_us) = (median(off_t), median(on_t));
    eprintln!(
        "  {scope}: median over 3 rounds — off {off_us} µs, on {on_us} µs, on/off {:.2}×",
        on_us as f64 / off_us.max(1) as f64
    );
    Pair {
        off_us,
        on_us,
        off_counts,
        on_counts,
        off_out,
        on_out,
    }
}

/// The active-extent (reduced-dispatch) protocol, once per graph: on each arm of
/// the switch, compile, print the arm's step-name table and the mode's
/// availability, and — where some step vetoes the mode — measure what a bucketed
/// `(2, 4)` extent costs against the full extent. `scope` names the graph and the
/// step the routes are expected to veto it with, so the two probes (the dense
/// conv's `im2col_gpu`, the 1×1 pointwise conv's `conv1x1_gemm`) read apart.
fn probe_active_extent(scope: &str, bg: &BodyGraph) {
    for off in [true, false] {
        set_arm(off);
        let mut arm = Arm::new(if off { "off" } else { "on" }, bg);
        let counts = arm.exe.step_kind_counts();
        let safe = arm.exe.test_all_steps_safe_for_active_extent();
        eprintln!(
            "  active-extent, {scope}, arm {}: test_all_steps_safe_for_active_extent = {safe} [{}]",
            arm.label,
            fmt_counts(&counts)
        );
        if safe {
            // Nothing vetoes the mode on this arm, so no bucketed run is lost.
            continue;
        }
        // The mode is off unless every step is safe, so a bucketed (actual,
        // upper) extent is ignored and the run pays the full extent — nothing of
        // it could be skipped. The full-extent reading is taken first and warm,
        // so the bucketed run is compared against a reading that paid no
        // bring-up either; `RLX_WGPU_ACTIVE_TRACE` names the step that vetoed the
        // mode, from the run loop, for exactly that run.
        let _ = arm.run_us();
        let full_us = arm.run_us();
        arm.exe.set_active_extent(Some((2, 4)));
        rlx_ir::env::set("RLX_WGPU_ACTIVE_TRACE", "1");
        let us = arm.run_us();
        rlx_ir::env::unset("RLX_WGPU_ACTIVE_TRACE");
        arm.exe.set_active_extent(None);
        eprintln!(
            "  active-extent, {scope}, arm {}: the bucketed (2, 4) run took {us} µs vs {full_us} µs \
             at the full extent ({:.2}×), so 0% of the run could be skipped (this is the cost of \
             the route, not a gate)",
            arm.label,
            us as f64 / full_us.max(1) as f64
        );
    }
}

/// Largest `|off − on|` over every element both arms produced. Both arms sum
/// the same f32 products in a different order, so this is either exactly zero
/// (the routes reassociate nothing here) or the honest size of the
/// reassociation — either way it is stated, not assumed. The fold is the shared
/// `common::max_abs_diff`, whose finiteness census runs first.
fn max_delta(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "both arms must produce the same outputs");
    a.iter()
        .zip(b)
        .map(|(x, y)| common::max_abs_diff(x, y))
        .fold(0f32, f32::max)
}

fn fmt_delta(d: f32) -> String {
    if d == 0.0 {
        "0 (bit-identical)".to_string()
    } else {
        format!("{d:.3e}")
    }
}

/// One row of a per-shape table.
struct ShapeRow {
    label: &'static str,
    off_counts: BTreeMap<&'static str, usize>,
    on_counts: BTreeMap<&'static str, usize>,
    off_us: u128,
    on_us: u128,
    max_delta: f32,
}

/// Print a family's per-shape table: for each single-shape graph, the route its
/// *own* compile took on each arm, the medians, the speedup, and the cross-arm
/// `max|Δ|`.
fn print_shape_table(family: &str, rows: &[ShapeRow]) {
    eprintln!("  {family} per-shape (one graph per shape, same protocol, same session):");
    eprintln!(
        "    {:<16} | {:<40} | {:>8} | {:>8} | {:>8} | {:>16}",
        "label", "route off → route on", "off µs", "on µs", "speedup", "max|Δ|"
    );
    for r in rows {
        eprintln!(
            "    {:<16} | {:<40} | {:>8} | {:>8} | {:>7.2}× | {:>16}",
            r.label,
            format!(
                "{} → {}",
                fmt_counts(&r.off_counts),
                fmt_counts(&r.on_counts)
            ),
            r.off_us,
            r.on_us,
            r.off_us as f64 / r.on_us.max(1) as f64,
            fmt_delta(r.max_delta),
        );
    }
}

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn conv_route_cost_over_the_decoder_body_shapes() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
        "RLX_WGPU_CONV_HOST",
        "RLX_WGPU_NO_TILED_CONV",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let device = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!(
        "wgpu conv-route cost on {device}: {} Conv1x7 + {} Conv1x1 body shapes, 3 interleaved \
         rounds per arm (off = RLX_WGPU_CONV_IM2COL=0, on = unset/default) after one warm-up run",
        CONV1X7.len(),
        CONV1X1.len()
    );
    // The lowering prints `[rlx-wgpu] schedule: N steps, arena=X MiB` plus the
    // `count × step-name` table for every compile below.
    rlx_ir::env::set("RLX_WGPU_SCHEDULE", "1");

    // (1) timing over the two arms of the switch: first the whole family in one
    // compile, then — the same protocol, one graph per single shape — a table
    // whose every row names the route *that shape alone* takes. The family
    // schedule is the union over its shapes, so a shape the route leaves behind
    // (or one it newly takes) is invisible in it.
    for (family, shapes) in [("Conv1x7", &CONV1X7[..]), ("Conv1x1", &CONV1X1[..])] {
        let bg = build_family(family, shapes);
        eprintln!("  {family}: shapes [{}]", bg.labels.join(", "));
        let _ = measure_pair(family, &bg);
        let mut rows = Vec::new();
        for shape in shapes {
            let one = build_family(family, std::slice::from_ref(shape));
            let pair = measure_pair(shape.label, &one);
            let (want_off, want_on) = expected_routes(family, shape.label);
            assert_eq!(
                conv_route(&pair.off_counts),
                want_off,
                "{}: the off arm must take {want_off} (schedule {:?})",
                shape.label,
                pair.off_counts
            );
            assert_eq!(
                conv_route(&pair.on_counts),
                want_on,
                "{}: the on arm must take {want_on} (schedule {:?})",
                shape.label,
                pair.on_counts
            );
            rows.push(ShapeRow {
                label: shape.label,
                off_counts: pair.off_counts,
                on_counts: pair.on_counts,
                off_us: pair.off_us,
                on_us: pair.on_us,
                max_delta: max_delta(&pair.off_out, &pair.on_out),
            });
        }
        print_shape_table(family, &rows);
    }

    // (2) the reduced-dispatch (active-extent) cost of the routes that veto it,
    // one probe per veto on its own graph: the dense body conv routes through
    // `im2col_gpu`, and the 1×1 pointwise conv emits a `conv1x1_gemm` step
    // (`Step::Matmul { from_conv1x1: true }` — its `m` is the output-channel axis,
    // so scaling `m` would leave channels unwritten) on a shape the dense probe
    // never measures.
    let dense = BodyConv {
        label: "dense-l4000",
        c_in: 384,
        c_out: 384,
        l: 4000,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 1],
        groups: 1,
    };
    probe_active_extent(
        "dense [1,384,1,4000] k=[1,7], vetoed by im2col_gpu",
        &build_family("conv_active_extent", std::slice::from_ref(&dense)),
    );
    let pointwise = BodyConv {
        label: "l800",
        c_in: 768,
        c_out: 768,
        l: 800,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    };
    probe_active_extent(
        "pointwise [1,768,1,800] 1×1 768→768, vetoed by conv1x1_gemm",
        &build_family(
            "conv_active_extent_pointwise",
            std::slice::from_ref(&pointwise),
        ),
    );

    // (3) plan invariance. The col-scratch reservation is decided from a conv's
    // *shape* alone, separately from the routing, so it must not leak into graphs
    // whose convolutions never qualify: both arms must plan identically, both for
    // a conv-free graph and for one carrying convolutions that clear no route's
    // floor (a 1×7 below every kernel/spatial floor, and a 1×1 below the `c_out`
    // floor). One compile per arm gives both halves — step counts and the planned
    // arena — so the `arena=… MiB` schedule lines the lowering prints above each
    // compile are asserted here, not merely printed.
    let plan_probe = |with_convs: bool| {
        let mut g = Graph::new(if with_convs {
            "small_convs_plan"
        } else {
            "conv_free_plan"
        });
        let x = g.input("x", Shape::new(&[4, 64], DType::F32));
        let w1 = g.param("w1", Shape::new(&[64, 64], DType::F32));
        let h = g.mm(x, w1);
        let w2 = g.param("w2", Shape::new(&[64, 64], DType::F32));
        let mut outs = vec![g.mm(h, w2)];
        if with_convs {
            for (i, k) in [[1usize, 7], [1, 1]].into_iter().enumerate() {
                let xn = format!("cx{i}");
                let wn = format!("cw{i}");
                let c = g.input(&xn, Shape::new(&[1, 8, 1, 100], DType::F32));
                let w = g.param(&wn, Shape::new(&[8, 8, k[0], k[1]], DType::F32));
                outs.push(g.conv2d(c, w, k, [1, 1], [0, k[1] / 2], [1, 1], 1));
            }
        }
        g.set_outputs(outs);
        g
    };
    for with_convs in [false, true] {
        set_arm(true);
        let off = WgpuExecutable::compile(plan_probe(with_convs));
        set_arm(false);
        let on = WgpuExecutable::compile(plan_probe(with_convs));
        let (off_counts, on_counts) = (off.step_kind_counts(), on.step_kind_counts());
        eprintln!(
            "  plan invariance ({}): off [{}] arena {} B / on [{}] arena {} B — the two arena=… MiB \
             schedule lines above must agree",
            if with_convs {
                "convs, none qualifying by shape"
            } else {
                "no conv"
            },
            fmt_counts(&off_counts),
            off.test_arena_bytes(),
            fmt_counts(&on_counts),
            on.test_arena_bytes()
        );
        assert_eq!(
            off_counts, on_counts,
            "a graph whose convolutions never qualify must plan identically under both arms \
             of RLX_WGPU_CONV_IM2COL"
        );
        assert_eq!(
            off.test_arena_bytes(),
            on.test_arena_bytes(),
            "a graph whose convolutions never qualify must reserve the same arena under both \
             arms of RLX_WGPU_CONV_IM2COL (the arena=… MiB schedule lines)"
        );
    }

    // (4) the shape-decided col reservation, seen where its route does not run: a
    // parked weight keeps the qualifying conv on the previous `conv2d` route, yet the
    // arena still holds the reservation the switch grants by *shape* alone. The tail
    // scratch is `max(existing, col)` — shared, not additive — and
    // `compute_scratch_bytes` floors the existing half at 64 MiB, so the reservation
    // costs arena bytes only above that floor: nothing for a col under it
    // (dense-l4000, 43 MB) and `col − 64 MiB` for the body's largest (dense-l48000,
    // 129 MB) — measured below as the difference between the two arms' printed
    // arena sizes, and the reason the whole Conv1x7 family's arena grows by the
    // largest col alone rather than by the sum of them.
    const SCRATCH_FLOOR: u64 = 64 * 1024 * 1024;
    let big = *CONV1X7
        .iter()
        .find(|c| c.label == "dense-l48000")
        .expect("dense-l48000");
    let parked_pair = |shape: &BodyConv| {
        rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
        let bg = build_family("conv_col_scratch_parked", std::slice::from_ref(shape));
        set_arm(true);
        let off = Arm::new("off", &bg);
        set_arm(false);
        let on = Arm::new("on", &bg);
        rlx_ir::env::unset("RLX_WGPU_ALL_PARAMS_WEIGHT");
        (off, on)
    };
    for shape in [dense, big] {
        let (off, on) = parked_pair(&shape);
        let (off_counts, on_counts) = (off.exe.step_kind_counts(), on.exe.step_kind_counts());
        let col_bytes =
            (shape.c_in as u64 * shape.k[0] as u64 * shape.k[1] as u64 * shape.l as u64 * 4)
                .div_ceil(256)
                * 256;
        let (off_b, on_b) = (off.exe.test_arena_bytes(), on.exe.test_arena_bytes());
        eprintln!(
            "  col reservation, {} weight parked: off [{}] arena {off_b} B / on [{}] arena {on_b} B \
             — on − off = {} B, col {col_bytes} B",
            shape.label,
            fmt_counts(&off_counts),
            fmt_counts(&on_counts),
            on_b - off_b
        );
        assert_eq!(
            conv_route(&on_counts),
            "conv2d",
            "a parked weight must keep the accelerated routes away ({}: {on_counts:?})",
            shape.label
        );
        assert_eq!(
            on_b - off_b,
            col_bytes.saturating_sub(SCRATCH_FLOOR),
            "{}: a shape-qualifying conv pays the col reservation even where its route does \
             not run",
            shape.label
        );
    }
}
