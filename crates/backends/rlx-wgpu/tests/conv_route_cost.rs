// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The accelerated conv routes — im2col+GEMM, the mirrored 1-D tiled relabel and the
//! 1×1 pointwise GEMM — are on by default under one switch (`RLX_WGPU_CONV_IM2COL`;
//! `=0` restores the plain routing). This harness is the marginal-cost evidence for
//! that flip over the decoder-body conv shapes the forward-parity suite pins
//! (`conv2d_groups_parity.rs`), over one dilated mirrored shape the relabel has to
//! carry, over one it must refuse, and over the band the im2col floor newly admits on
//! real 2-D maps (`Conv2d` — unit-height rows reach that band only through the length
//! axis, so the 2-D half of its blast radius is that family's job). Per family, and
//! then per *single shape* (the family schedule is the union over its shapes, where a
//! shape the route leaves behind hides), it prints the route table, the arena MiB, the
//! wall time of one `WgpuExecutable::run` on each arm of the switch, the
//! reduced-dispatch (active-extent) cost the routes veto, the plan-invariance check
//! and the col-scratch reservation a shape-qualifying conv pays even where its route
//! does not run (a parked weight).
//!
//! The im2col spatial floor and the 1×1 route's admission bar are two decisions, not
//! one: the 1×1 bar is the frozen `256 × 2048` product the two once shared
//! (`backend::POINTWISE_MIN_COL_ELEMS`), so the floor's default of 800 moves the
//! non-pointwise sub-2048 shapes (the decoder body's `conv1` forms) onto the
//! materialised route and leaves every 1×1 shape exactly where it was.
//!
//! Nothing here asserts a duration: `run()` includes submission, the device wait and
//! the readback, so a reading is mostly that fixed floor. Only facts that can be
//! stated exactly are asserted — the route each shape takes on each arm (only the
//! scalable direct/tiled ones can honour a bucketed extent), that the off arm carries
//! no `im2col_gpu` step, that a graph whose convolutions never qualify keeps its plan
//! *and* its arena under both arms, every row's arena growth (see `expected_col_delta`
//! for the recorded deltas and `backend::conv_route_decision_tests` for the arithmetic
//! under them), and that a shape-qualifying conv pays its reservation, parked or not.
//!
//! For the rows this change moves that no other accelerated path served — every `Conv2d`
//! row, which the old floor of 2048 left on the direct kernel — the off arm *is* the
//! pre-change routing, so its ratio is the row's own. `dense-l800` is not such a row: that
//! floor left it on the `conv2d_tiled` relabel, so its off column (the direct kernel)
//! overstates the win, and it alone is measured a third time with the old floor put back
//! through the lever the run itself pulls (`RLX_WGPU_IM2COL_MIN_SPATIAL=2048`; see
//! `predecessor_pair`). `lift-slot0-l1000`'s off arm already *is* that relabel, so its off
//! column is its before side.
//!
//! The adapter/allocation half of the same decision is printed here where it is non-zero
//! and asserted where both arms are reachable (`conv_routes_parked_weight.rs` reads
//! `im2col_fallbacks()` as zero on the arm that routes and non-zero on the parked arm);
//! the causes are stated once, in the `Op::Conv` arm of `src/backend/compile/lower.rs`.
//!
//! Run the thermal gate first (`just throttle`).

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::Tick;
use rlx_wgpu::backend::WgpuExecutable;
use std::collections::BTreeMap;

mod common;

/// One conv fixture of an `N = 1` NCHW graph: `[1, C_in, h, l]` input, kernel
/// `k`, stride `s`, padding `p`, per-axis dilation `d`, `groups`. The decoder
/// body rows are unit-height 1-D lifts (`h: 1`); the 2-D band rows below are real
/// maps.
#[derive(Clone, Copy)]
struct BodyConv {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    h: usize,
    l: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    groups: usize,
}

/// The 7-tap (`Conv1x7`) family: the dense shapes with the length taps in slot 1
/// (`dense-l800` sits exactly on the 800 im2col spatial floor and is the
/// boundary pin for it), the two depthwise shapes, the ONNX lift with the taps in
/// slot 0, and one dilated mirrored shape — the shapes the forward-parity suite
/// added plus the dilation axis the mirrored relabel has to carry.
const CONV1X7: [BodyConv; 13] = [
    BodyConv {
        label: "dense-l800",
        c_in: 768,
        c_out: 768,
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
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
        h: 1,
        l: 4000,
        k: [7, 1],
        s: [1, 1],
        p: [3, 0],
        d: [1, 1],
        groups: 1,
    },
    // The same slot-0 lift as `overlap-slot0`, but in the band the floor moved: out
    // 1000 clears the 800 floor, so this row leaves the `one_d` tiled route for the
    // materialised one (`overlap-slot0`'s 4000 was already im2col at the old floor,
    // so it cannot census that move). Col 1 792 000 B — under the 64 MiB base
    // scratch, so it costs the arena nothing.
    BodyConv {
        label: "lift-slot0-l1000",
        c_in: 64,
        c_out: 64,
        h: 1,
        l: 1000,
        k: [7, 1],
        s: [1, 1],
        p: [3, 0],
        d: [1, 1],
        groups: 1,
    },
    // The only shape here that exercises the dilation axis through the mirrored
    // relabel: the out length is 794, just under the 800 im2col spatial floor — so
    // im2col does not take it — and over the 256 tiled floor, so it lands on the
    // mirrored relabel, which copies the slot-1 dilation into the tiled kernel's
    // `dh` (2 here; the stride-2 case above only covers `sh`).
    BodyConv {
        label: "mirror-dilated-d2",
        c_in: 384,
        c_out: 384,
        h: 1,
        l: 800,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        d: [1, 2],
        groups: 1,
    },
    // The mirror must be REFUSED here, and the route table below pins that: the
    // height axis carries padding, so H is a real axis — the relabel would move
    // it onto the length axis and resize the output (3×200 instead of 200). The
    // output spatial (3×200 = 600) is deliberately *above* the 256 tiled floor
    // and *below* the 800 im2col floor, so this row discriminates: were the
    // refusal lost, the shape would relabel into the tiled route and the assertion
    // below would fail instead of the row printing a plausible number (with
    // spatial below the tiled floor both outcomes are `conv2d` and the row is
    // inert).
    BodyConv {
        label: "refuse-height-pad",
        c_in: 64,
        c_out: 64,
        h: 1,
        l: 200,
        k: [1, 7],
        s: [1, 1],
        p: [1, 3],
        d: [1, 1],
        groups: 1,
    },
];

/// The pointwise (`Conv1x1`) family: four shapes above the 1×1 route's own
/// frozen col-matrix bar and one below it.
const CONV1X1: [BodyConv; 5] = [
    BodyConv {
        label: "l800",
        c_in: 768,
        c_out: 768,
        h: 1,
        l: 800,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
    // The frozen 1×1 bar, seen from below: `c_in × spatial` = 384 000 lies inside
    // the band the im2col floor's old coupling would admit once that floor moved
    // (256 × 800) and below the 1×1 route's own bar (256 × 2048), so the row must
    // keep the direct kernel on both arms. Were the 1×1 admission still tied to
    // the im2col floor, it would flip to `conv1x1_gemm` here.
    BodyConv {
        label: "sub-bar-l3000",
        c_in: 128,
        c_out: 128,
        h: 1,
        l: 3000,
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
        h: 1,
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
        h: 1,
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
        h: 1,
        l: 48000,
        k: [1, 1],
        s: [1, 1],
        p: [0, 0],
        d: [1, 1],
        groups: 1,
    },
];

/// The band the 800 im2col spatial floor newly admits — spatial in `[800, 2048)`,
/// the extent the old floor of 2048 refused — on shapes the unit-height families
/// above cannot represent: real 2-D maps. Those rows reach the floor only through
/// the length axis, so the 2-D half of the band's blast radius (an ordinary
/// vision conv on a small map) is this family's job.
///
/// All three rows sit strictly inside the band. `map-3x3-29x29-c32` is the smallest
/// ordinary one: extent 841 just over the floor, `c_in*kh*kw` 288 just over the K
/// floor, `c_out` on its floor. `map-3x3-40x40` is the ordinary case
/// (spatial 1600, col 3.7 MB — under the 64 MiB base scratch, so the shape costs
/// the arena nothing); `map-5x5-30x30-c1024` carries the same band with a col
/// panel past that base (`c_in*kh*kw*spatial*4` = 92 MB), so its reservation does
/// grow the tail, by `col − 64 MiB`, exactly as any qualifying conv above the old
/// floor already did before this change.
const CONV2D: [BodyConv; 3] = [
    BodyConv {
        label: "map-3x3-29x29-c32",
        c_in: 32,
        c_out: 64,
        h: 29,
        l: 29,
        k: [3, 3],
        s: [1, 1],
        p: [1, 1],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "map-3x3-40x40",
        c_in: 64,
        c_out: 64,
        h: 40,
        l: 40,
        k: [3, 3],
        s: [1, 1],
        p: [1, 1],
        d: [1, 1],
        groups: 1,
    },
    BodyConv {
        label: "map-5x5-30x30-c1024",
        c_in: 1024,
        c_out: 64,
        h: 30,
        l: 30,
        k: [5, 5],
        s: [1, 1],
        p: [2, 2],
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
        let x = g.input(&xn, Shape::new(&[1, c.c_in, c.h, c.l], DType::F32));
        let w = g.param(
            &wn,
            Shape::new(&[c.c_out, c.c_in / c.groups, c.k[0], c.k[1]], DType::F32),
        );
        outs.push(g.conv2d(x, w, c.k, c.s, c.p, c.d, c.groups));
        inputs.push((xn, fill(c.c_in * c.h * c.l, i as u64)));
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

/// The table row a probe measures, looked up rather than re-spelled: a probe that
/// repeated a row's geometry would silently measure a different shape once the table
/// is edited.
fn row(table: &[BodyConv], label: &str) -> BodyConv {
    *table
        .iter()
        .find(|c| c.label == label)
        .unwrap_or_else(|| panic!("no {label} row in the harness tables"))
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
/// table also states which shapes moved (`overlap`: already accelerated today, onto
/// the other accelerated route; `dense-l800`: the decoder-body row whose route the
/// 800 im2col floor changes — it sits exactly on it; the `Conv2d` rows: shapes
/// strictly inside the band that floor newly admits), which shapes stay on the
/// direct kernel
/// under both arms (`entry`, the two `depthwise`, the shape the mirror must
/// *refuse*, and the 1×1 shape below the pointwise bar), and that a lost refusal
/// cannot pass unnoticed. The two rows that floor would otherwise swallow —
/// `mirror-dilated-d2` (the relabel's dilation axis) and `refuse-height-pad` (the
/// refusal) — sit just under it by construction.
fn expected_routes(family: &str, label: &str) -> (&'static str, &'static str) {
    match (family, label) {
        ("Conv1x1", "sub-bar-l3000") => ("conv2d", "conv2d"),
        ("Conv1x1", _) => ("conv2d", "conv1x1_gemm"),
        // The newly admitted band on real 2-D maps: all three rows sit strictly
        // inside it (extents 841, 1600 and 900), so all move direct → materialised.
        ("Conv2d", "map-3x3-29x29-c32" | "map-3x3-40x40" | "map-5x5-30x30-c1024") => {
            ("conv2d", "im2col_gpu")
        }
        // `dense-l800` sits exactly on the 800 im2col spatial floor: the boundary
        // row for it (the floor is inclusive).
        (_, "dense-l800" | "dense-l4000" | "dense-l16000" | "dense-l48000") => {
            ("conv2d", "im2col_gpu")
        }
        // `mirror-projection` and `mirror-strided-s2` clear the spatial floor but
        // not the `c_out` floor, so they still carry the mirrored relabel; and
        // `mirror-dilated-d2` carries the relabel's dilation axis from below the
        // 800 floor.
        (_, "mirror-projection" | "mirror-strided-s2" | "mirror-dilated-d2") => {
            ("conv2d", "conv2d_tiled")
        }
        (_, "overlap-slot0" | "lift-slot0-l1000") => ("conv2d_tiled", "im2col_gpu"),
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
/// step-name table (which names the routes it took), the planned arena bytes and
/// the outputs of the last round.
struct Pair {
    off_us: u128,
    on_us: u128,
    off_counts: BTreeMap<&'static str, usize>,
    on_counts: BTreeMap<&'static str, usize>,
    off_arena: u64,
    on_arena: u64,
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
    let (off_counts, on_counts) = (off.exe.step_kind_counts(), on.exe.step_kind_counts());
    let (off_arena, on_arena) = (off.exe.test_arena_bytes(), on.exe.test_arena_bytes());
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
        off_arena,
        on_arena,
        off_out,
        on_out,
    }
}

/// The third arm, for `dense-l800`, whose off column is not its before side: the same
/// protocol with the documented lever's floor put back, so the row's before side is printed
/// by this run instead of living as a reading in prose.
fn predecessor_pair(scope: &str, bg: &BodyGraph) -> Pair {
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_SPATIAL", "2048");
    rlx_wgpu::reload_runtime_config();
    let pair = measure_pair(scope, bg);
    rlx_ir::env::unset("RLX_WGPU_IM2COL_MIN_SPATIAL");
    rlx_wgpu::reload_runtime_config();
    pair
}

/// The active-extent (reduced-dispatch) protocol, once per graph: on each arm of the
/// switch, compile, assert the arm's conv route and whether the mode is available for
/// it (only the scalable `conv2d` / `conv2d_tiled` routes can honour a bucketed
/// extent), print the arm's step-name table, and — where some step vetoes the mode —
/// measure what a bucketed `(2, 4)` extent costs against the full extent. `want_routes`
/// is the `(off, on)` route the probe's graph must take, which also names the step that
/// vetoes the mode.
fn probe_active_extent(scope: &str, bg: &BodyGraph, want_routes: [&'static str; 2]) {
    for (i, off) in [true, false].into_iter().enumerate() {
        set_arm(off);
        let mut arm = Arm::new(if off { "off" } else { "on" }, bg);
        let counts = arm.exe.step_kind_counts();
        let route = conv_route(&counts);
        let safe = arm.exe.test_all_steps_safe_for_active_extent();
        eprintln!(
            "  active-extent, {scope}, arm {}: route {route}, \
             test_all_steps_safe_for_active_extent = {safe} [{}]",
            arm.label,
            fmt_counts(&counts)
        );
        assert_eq!(route, want_routes[i], "{scope}: arm {} route", arm.label);
        assert_eq!(
            safe,
            matches!(route, "conv2d" | "conv2d_tiled"),
            "{scope}: arm {} — only the scalable conv routes can honour a bucketed extent",
            arm.label
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

/// The arena bytes each row's col reservation costs, recorded from a run of this
/// harness rather than recomputed here: the tail scratch is `max(existing, col)`,
/// and `compute_scratch_bytes` floors the existing half at 64 MiB, so a col under
/// that base costs nothing and a larger one costs `col − 64 MiB`. A row that
/// qualifies for no accelerated route by shape reserves nothing. The reservation's
/// arithmetic — the col a shape gives, both caps the planner applies to it and the
/// 64 MiB tail base — is pinned device-free by `backend::conv_route_decision_tests`;
/// what the numbers below add is that arithmetic against these exact fixtures. An
/// unlisted label panics rather than recording a silent zero, so a renamed or added
/// fixture cannot assert nothing.
fn expected_col_delta(label: &str) -> u64 {
    match label {
        // 129 024 000 B col − 64 MiB, the body's largest.
        "dense-l48000" => 61_915_136,
        // 86 016 000 B col − 64 MiB.
        "dense-l16000" => 18_907_136,
        // 92 160 000 B col − 64 MiB, the in-band 2-D map.
        "map-5x5-30x30-c1024" => 25_051_136,
        // Cols that fit the 64 MiB base tail: nothing to reserve.
        "dense-l800" | "dense-l4000" | "map-3x3-29x29-c32" | "map-3x3-40x40"
        | "lift-slot0-l1000" | "mirror-projection" | "mirror-strided-s2" | "mirror-dilated-d2"
        | "refuse-height-pad" | "overlap-slot0" | "entry-l100" | "depthwise-l50"
        | "depthwise-l100" | "l800" | "l4000" | "l16000" | "l48000" | "sub-bar-l3000" => 0,
        other => panic!("no recorded col delta for {other:?} — the body shape list grew"),
    }
}

/// One row of a per-shape table.
struct ShapeRow {
    label: &'static str,
    off_counts: BTreeMap<&'static str, usize>,
    on_counts: BTreeMap<&'static str, usize>,
    off_us: u128,
    on_us: u128,
    off_arena: u64,
    on_arena: u64,
    max_delta: f32,
}

/// Print a family's per-shape table: for each single-shape graph, the route its
/// *own* compile took on each arm, the medians, the speedup, the planned arena on
/// each arm (MiB) and the cross-arm `max|Δ|`.
fn print_shape_table(family: &str, rows: &[ShapeRow]) {
    eprintln!("  {family} per-shape (one graph per shape, same protocol, same session):");
    eprintln!(
        "    {:<16} | {:<40} | {:>8} | {:>8} | {:>8} | {:>16} | {:>13}",
        "label", "route off → route on", "off µs", "on µs", "speedup", "max|Δ|", "arena MiB"
    );
    for r in rows {
        eprintln!(
            "    {:<16} | {:<40} | {:>8} | {:>8} | {:>7.2}× | {:>16} | {:>13}",
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
            format!(
                "{:.1}→{:.1}",
                r.off_arena as f64 / (1u64 << 20) as f64,
                r.on_arena as f64 / (1u64 << 20) as f64
            ),
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
    common::refuse_under_run_report();
    let device = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!(
        "wgpu conv-route cost on {device}: {} Conv1x7 + {} Conv1x1 body shapes + {} in-band 2-D \
         maps, 3 interleaved rounds per arm (off = RLX_WGPU_CONV_IM2COL=0, on = unset/default) \
         after one warm-up run",
        CONV1X7.len(),
        CONV1X1.len(),
        CONV2D.len()
    );
    // The lowering prints `[rlx-wgpu] schedule: N steps, arena=X MiB` plus the
    // `count × step-name` table for every compile below.
    rlx_ir::env::set("RLX_WGPU_SCHEDULE", "1");

    // (1) timing over the two arms of the switch: first the whole family in one
    // compile, then — the same protocol, one graph per single shape — a table
    // whose every row names the route *that shape alone* takes. The family
    // schedule is the union over its shapes, so a shape the route leaves behind
    // (or one it newly takes) is invisible in it.
    for (family, shapes) in [
        ("Conv1x7", &CONV1X7[..]),
        ("Conv1x1", &CONV1X1[..]),
        ("Conv2d", &CONV2D[..]),
    ] {
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
            // The memory half of the switch, pinned on every row against the delta
            // recorded for it in `expected_col_delta` — nothing is recomputed here. A
            // shape-qualifying conv costs the col panel above the 64 MiB base
            // `compute_scratch_bytes` floors the existing tail at; every other row
            // costs nothing, which is why the moved decoder shape's own col leaves the
            // arena alone while an in-band col over that base does grow it. A failure
            // here on an adapter whose `max_storage_buffer_binding_size` cannot take
            // the col says the accelerated routes are off on that adapter — loud, not
            // silent.
            assert_eq!(
                pair.on_arena - pair.off_arena,
                expected_col_delta(shape.label),
                "{}: the arena growth the row's recorded col reservation predicts",
                shape.label
            );
            // `dense-l800`'s off column is not its before side (that floor left it on the
            // relabel, off leaves it on the direct kernel): it alone gets the third arm.
            if shape.label == "dense-l800" {
                // The reading at the old floor, where the `conv2d_tiled` relabel carried it.
                let pre = predecessor_pair(&format!("{} @2048", shape.label), &one);
                assert_eq!(
                    conv_route(&pre.on_counts),
                    "conv2d_tiled",
                    "{}: the route this row ran at the old 2048 floor",
                    shape.label
                );
            }
            rows.push(ShapeRow {
                label: shape.label,
                off_counts: pair.off_counts,
                on_counts: pair.on_counts,
                off_us: pair.off_us,
                on_us: pair.on_us,
                off_arena: pair.off_arena,
                on_arena: pair.on_arena,
                max_delta: max_delta(&pair.off_out, &pair.on_out),
            });
        }
        print_shape_table(family, &rows);
    }

    // (2) the reduced-dispatch (active-extent) cost of the routes that veto it,
    // three probes on their own graphs, each naming the step that vetoes the mode:
    // the dense body conv routes through `im2col_gpu`; the 1×1 pointwise conv emits
    // a `conv1x1_gemm` step (`Step::Matmul { from_conv1x1: true }` — its `m` is the
    // output-channel axis, so scaling `m` would leave channels unwritten) on a shape
    // the dense probe never measures; and (2b) below, the band the moved floor newly
    // admits.
    let dense = row(&CONV1X7, "dense-l4000");
    probe_active_extent(
        "dense [1,384,1,4000] k=[1,7], vetoed by im2col_gpu",
        &build_family("conv_active_extent", std::slice::from_ref(&dense)),
        ["conv2d", "im2col_gpu"],
    );
    let pointwise = row(&CONV1X1, "l800");
    probe_active_extent(
        "pointwise [1,768,1,800] 1×1 768→768, vetoed by conv1x1_gemm",
        &build_family(
            "conv_active_extent_pointwise",
            std::slice::from_ref(&pointwise),
        ),
        ["conv2d", "conv1x1_gemm"],
    );

    // (2b) the band the moved floor newly admits, on a graph whose only conv sits in
    // it — the smallest row there is, its extent just over the floor: the direct kernel
    // the shape used to take was active-extent safe, the `im2col_gpu` step it now takes
    // is not, so this graph loses the mode.
    let near_floor = row(&CONV2D, "map-3x3-29x29-c32");
    probe_active_extent(
        "band [map-3x3-29x29-c32] (the floor's newest band), vetoed by im2col_gpu",
        &build_family("conv_active_extent_band", std::slice::from_ref(&near_floor)),
        ["conv2d", "im2col_gpu"],
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
    // arena still holds the reservation the switch grants by *shape* alone — the same
    // recorded deltas `expected_col_delta` holds (a col under the 64 MiB base scratch
    // costs nothing, a larger one costs `col − 64 MiB`), which is also why the whole
    // Conv1x7 family's arena grows by the largest col alone rather than by the sum of
    // them. This is also the arm where `im2col_fallbacks()` is non-zero; the count is
    // printed here, and `conv_routes_parked_weight.rs` asserts both of its arms.
    let big = row(&CONV1X7, "dense-l48000");
    let moved = row(&CONV1X7, "dense-l800");
    // The band's own parked row: a 2-D map newly admitted by the floor whose col
    // panel is past the 64 MiB base, so the combination "parked, in the band, col
    // over the base" is pinned too rather than only its two halves.
    let in_band = row(&CONV2D, "map-5x5-30x30-c1024");
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
    for shape in [dense, moved, big, in_band] {
        let (off, on) = parked_pair(&shape);
        let (off_counts, on_counts) = (off.exe.step_kind_counts(), on.exe.step_kind_counts());
        let (off_b, on_b) = (off.exe.test_arena_bytes(), on.exe.test_arena_bytes());
        eprintln!(
            "  col reservation, {} weight parked: off [{}] arena {off_b} B / on [{}] arena {on_b} B \
             — on − off = {} B, recorded delta {} B, on-arm im2col fallbacks {}",
            shape.label,
            fmt_counts(&off_counts),
            fmt_counts(&on_counts),
            on_b - off_b,
            expected_col_delta(shape.label),
            on.exe.im2col_fallbacks()
        );
        assert_eq!(
            conv_route(&on_counts),
            "conv2d",
            "a parked weight must keep the accelerated routes away ({}: {on_counts:?})",
            shape.label
        );
        assert_eq!(
            on_b - off_b,
            expected_col_delta(shape.label),
            "{}: a shape-qualifying conv pays the col reservation even where its route does \
             not run",
            shape.label
        );
    }
}
