// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cooperative implicit-GEMM convolution computes the conv it is admitted
//! for, and the switch that turns it off puts the same graph back on the
//! im2col/tiled/direct routing with the same answer.
//!
//! The route is `conv1d_implicit.wgsl`: the direct cooperative entry's 32×32 f32
//! tile with B gathered through the conv's taps into workgroup memory, so no
//! im2col panel is materialised. This file pins three things about it.
//!
//! **The geometry.** The gather decomposes the contraction as `c = k / kw`,
//! `tap = k % kw` and carries the pair forward; a tap that walks off the input's
//! length is a zero, the k-step's legs past `k_total` are zeros, and the epilogue
//! drops the rows and columns past `c_out` and `out_len`. The cases below walk
//! those edges: `96 → 96, kw = 7` at `d ∈ {1, 3, 9}` (the 6-8 block's own taps, on
//! the shapes the route's gate is built for), `32 → 64, kw = 5, d = 2` (a
//! different tap/dilation ratio against the same channel multiple), and a tails
//! case whose `c_out = 40` ends in a partial row tile and whose padded output,
//! `1 033`, ends in a partial column tile. Two of them put every accumulator set
//! and a non-multiple-of-8 contraction in play at once (`20 → 96` in three whole
//! tiles, `20 → 80` with the third partial), which is the combination the entry's
//! three sets can be wrong in. The k-step's own tail is zero-filled
//! (`gk < k_total`) rather than left to a host term, so a shape whose `k_total` is
//! not a multiple of 8 is correct too.
//!
//! **The parity.** Every case is run against the crate's own CPU conv
//! (`common::unit_conv_cpu_reference` — an `rlx_runtime::Session` on
//! `Device::Cpu`), and the same graph is compiled a second time under
//! `RLX_WGPU_CONV_IMPLICIT=0` and held to the same reference. The bar is the
//! crate's `1e-2` for a route whose accumulation order is not the CPU's: these
//! contractions are 96–672 deep and the cooperative fragments sum them in their
//! own order, so nothing here asserts identity. The switch-off arm is the
//! statement that the route is *additive* — a graph it takes and a graph it does
//! not take both keep the answer they had.
//!
//! **The gate, not the kernel, decides admission.** The route is admitted only for
//! the class it wins in (see `ConvImplicitShape` in `backend/mod.rs`), and the parity
//! cases here sit below the class's own extent floor: the parity cell drives the
//! kernel at small extents with `RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL`,
//! which is the floor's own override and does not move anything else. What that
//! leaves unpinned is stated plainly — the floor's own arithmetic is the stand's,
//! not this file's. The cells after the table run the route the way a real
//! schedule does: [`a_conv_between_other_steps`] mid-schedule,
//! [`a_height_padded_unit_tap_conv_is_not_the_route`] at a shape whose `h_out == 1`
//! holds while its own tap reads outside the input, and the `#[ignore]`d one at
//! the block's own geometry.
//!
//! Its own test binary: the route switch is process-global and the floor override
//! is read at compile time, so the cases cannot share a process with any other
//! harness's routing.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

/// One case: the conv the graph is built from and the label its assertions carry.
struct Case {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    /// The input's length. The output's is [`out_len`]'s, not this.
    l: usize,
    kw: usize,
    dilation: usize,
    pad: usize,
}

/// The route's own geometries. `l` is small on purpose: the parity is the claim,
/// and the admitted extents are the measurement cell's
/// ([`conv_implicit_vs_panel_measurement`]), not this table's.
const CASES: &[Case] = &[
    Case {
        label: "96/96 kw=7 d=1",
        c_in: 96,
        c_out: 96,
        l: 1024,
        kw: 7,
        dilation: 1,
        pad: 6,
    },
    Case {
        label: "96/96 kw=7 d=3",
        c_in: 96,
        c_out: 96,
        l: 1024,
        kw: 7,
        dilation: 3,
        pad: 18,
    },
    Case {
        label: "96/96 kw=7 d=9",
        c_in: 96,
        c_out: 96,
        l: 1024,
        kw: 7,
        dilation: 9,
        pad: 54,
    },
    Case {
        label: "32/64 kw=5 d=2",
        c_in: 32,
        c_out: 64,
        l: 1024,
        kw: 5,
        dilation: 2,
        pad: 8,
    },
    Case {
        label: "tails 32/40 kw=3 d=1 l=1031",
        c_in: 32,
        c_out: 40,
        l: 1031,
        kw: 3,
        dilation: 1,
        pad: 2,
    },
    // Neither channel count is a multiple of 8, so the k-step's last leg is a real
    // tail (`k_total = 100`) and the row tile is partial: the kernel's own zero fill
    // and store guard cover both, with no host term narrowing the class.
    Case {
        label: "tails 20/12 kw=5 d=2",
        c_in: 20,
        c_out: 12,
        l: 1024,
        kw: 5,
        dilation: 2,
        pad: 8,
    },
    // Three whole row tiles over the same non-multiple-of-8 contraction: the k-step
    // tail and every tile the entry carries are in play at once.
    Case {
        label: "three tiles 20/96 kw=5 d=2",
        c_in: 20,
        c_out: 96,
        l: 1024,
        kw: 5,
        dilation: 2,
        pad: 8,
    },
    // …and three tiles whose last one is partial, so the store guard's row bound is
    // exercised past the second tile as well as inside it.
    Case {
        label: "three tiles partial 20/80 kw=5 d=2",
        c_in: 20,
        c_out: 80,
        l: 1024,
        kw: 5,
        dilation: 2,
        pad: 8,
    },
];

/// The unit-height 1-D conv these two routes are built for: `[1, c_in, 1, l]`,
/// kernel `[1, kw]`, one group, `N == 1`, and the case's dilation and padding.
/// `common::unit_conv_graph`'s shape, so the shared CPU reference applies
/// unchanged.
fn graph(case: &Case) -> Graph {
    common::unit_conv_graph(
        "conv1d_implicit",
        case.c_in,
        case.c_out,
        case.l,
        case.kw,
        case.pad,
        case.dilation,
    )
}

/// The conv's output length from the tree's own shape rule: stride 1, the case's
/// padding and dilation, and the kernel's `kw` taps.
fn out_len(case: &Case) -> usize {
    case.l + 2 * case.pad - case.dilation * (case.kw - 1)
}

/// Whether this adapter can build the implicit entry at all — the constructor's
/// own refusal, asked without building the pipeline. Without the path the route is
/// never admitted and there is nothing to witness, so the test self-skips: an
/// adapter that is present but cannot run the entry is not a device failure.
fn conv1d_implicit_available() -> bool {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        return false;
    };
    rlx_wgpu::kernels::conv1d_implicit_supported(&dev.device)
}

/// The routing switches every cell in this binary needs: the host/route flips that
/// would move a conv off the device, the parked-weight arm (a tagged weight offset
/// is not what either GEMM route addresses), the umbrella conv switch, the
/// implicit route's own floor moved down to these cells' small extents, and the
/// persisted dispatch table pinned off (the switch-off arm's im2col multiply reads
/// it, and a stored override for one of these shapes would move that arm's kernel
/// under the parity; it is read once per process, on `gpu_arch()`).
fn route_switches() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_NO_TILED_CONV",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    rlx_ir::env::set("RLX_WGPU_CONV_IM2COL", "1");
    rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", "1");
    rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL", "16");
    rlx_ir::env::set(
        "RLX_GPU_TUNING_CACHE",
        std::env::temp_dir()
            .join("rlx-conv1d-implicit-absent.tsv")
            .to_string_lossy()
            .into_owned(),
    );
}

/// The two guards every cell here shares: an adapter at all, and one that can
/// build the route's entry — without the second the class is never admitted by the
/// adapter term and a cell could hold with the property it pins removed.
fn skip_without_the_route() -> bool {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return true;
    }
    if !conv1d_implicit_available() {
        eprintln!("no Metal cooperative-matrix path on this adapter — skipping");
        return true;
    }
    false
}

/// Compile `case` under the switches in force, bind the weight, run `x`, and
/// report the step names the schedule took, the output, the route's decline
/// counter and the panel bytes the route removed. One conv in the graph, so the
/// counter and the bytes are this node's.
fn compile_and_run(
    case: &Case,
    xs: &[f32],
    ws: &[f32],
) -> (Vec<&'static str>, Vec<f32>, usize, u64) {
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    let steps = exe.test_step_names();
    let declines = exe.test_conv1d_implicit_declines();
    let panel = exe.test_conv_panel_bytes();
    (steps, exe.run(&[("x", xs)]).remove(0), declines, panel)
}

#[test]
fn the_implicit_conv_computes_its_class_and_the_switch_restores_the_old_route() {
    // Every cell in this binary sets the route's own process-global switches; the
    // lock keeps their compiles from reading each other's.
    let _routes = common::route_lock();
    route_switches();
    if skip_without_the_route() {
        return;
    }

    for case in CASES {
        let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(case.c_out * case.c_in * case.kw, 5, 3, 11, 0.1, -0.5);
        let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
        // The kernel's contraction and its output-column extent.
        let k_total = (case.c_in * case.kw) as u64;
        let spatial = out_len(case) as u64;

        rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", "1");
        let (steps_on, got_on, declines_on, panel_on) = compile_and_run(case, &xs, &ws);
        rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", "0");
        let (steps_off, got_off, declines_off, panel_off) = compile_and_run(case, &xs, &ws);

        assert!(
            steps_on.contains(&"conv1d_implicit"),
            "{}: the switch on must reach the implicit route (schedule {steps_on:?})",
            case.label
        );
        assert!(
            !steps_off.contains(&"conv1d_implicit"),
            "{}: the switch off must leave the implicit route (schedule {steps_off:?})",
            case.label
        );
        assert_eq!(
            declines_on, 0,
            "{}: nothing declined with the route on",
            case.label
        );
        assert_eq!(
            declines_off, 1,
            "{}: the switched-off route must say it declined the shape",
            case.label
        );
        assert_eq!(
            panel_on,
            k_total * spatial * 4,
            "{}: the panel the route removed is k_total · spatial · 4",
            case.label
        );
        assert_eq!(
            panel_off, 0,
            "{}: no implicit step, no panel bytes to name",
            case.label
        );

        let max_on = common::max_abs_diff(&got_on, &want);
        let max_off = common::max_abs_diff(&got_off, &want);
        assert!(
            max_on < 1e-2,
            "{}: implicit max|Δ| {max_on:e} > 1e-2",
            case.label
        );
        assert!(
            max_off < 1e-2,
            "{}: switch-off max|Δ| {max_off:e} > 1e-2",
            case.label
        );
        println!(
            "{}: implicit max|Δ| {max_on:e}, switch-off max|Δ| {max_off:e}, \
             panel {} B",
            case.label, panel_on
        );
    }
}

/// A unit-height kernel with height padding is not the route's shape: the single
/// H tap lands at row `−ph`, outside the input, so the conv that shape describes is
/// zeros — while the entry reads input row 0 unconditionally. `h_out == 1` does not
/// imply `ph == 0` (`h_out = floor((h + 2·ph − kh)/sh) + 1`), so the gate states it.
#[test]
fn a_height_padded_unit_tap_conv_is_not_the_route() {
    let _routes = common::route_lock();
    route_switches();
    if skip_without_the_route() {
        return;
    }

    const C: usize = 32;
    const L: usize = 1024;
    const KW: usize = 5;
    // `h_out = floor((1 + 2·1 − 1)/3) + 1 = 1`, so the route's `h_out == 1` holds
    // while the tap that shape describes reads outside the input.
    let mut g = Graph::new("conv1d_implicit_height_pad");
    let x = g.input("x", Shape::new(&[1, C, 1, L], DType::F32));
    let w = g.param("w", Shape::new(&[C, C, 1, KW], DType::F32));
    let y = g.conv2d(x, w, [1, KW], [3, 1], [1, 4], [1, 1], 1);
    g.set_outputs(vec![y]);

    let xs = common::fill(C * L, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(C * C * KW, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(g.clone(), &xs, &ws);
    assert!(
        want.iter().all(|v| v.abs() < 1e-6),
        "the shape's only tap is outside the input, so its conv is zeros"
    );

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &ws);
    let steps = exe.test_step_names();
    assert!(
        !steps.contains(&"conv1d_implicit"),
        "a height-padded conv must not take the route ({steps:?})"
    );
    assert_eq!(
        exe.test_conv1d_implicit_declines(),
        0,
        "the shape is outside the class, so there is nothing to decline ({steps:?})"
    );
    let got = exe.run(&[("x", &xs)]).remove(0);
    let max = common::max_abs_diff(&got, &want);
    assert!(max < 1e-6, "max|Δ| {max:e} against the zeros it must be");
    println!("height-padded unit tap: max|Δ| {max:e}, steps {steps:?}");
}

/// The class shape declined at lowering while the route is live: a weight parked in
/// the weight buffer cannot be addressed by the entry (its word offset drops the
/// tag), so the conv keeps the direct route — with no panel to fall back to, because
/// the reservation subtracts this class. The fallback is the guard this pins: a
/// panel past the tail's 64 MiB floor is refused by `col_bytes <= arena.scratch_bytes`
/// and the conv runs direct, to the CPU's answer.
#[test]
fn a_class_shape_with_a_parked_weight_falls_to_the_direct_conv() {
    let _routes = common::route_lock();
    route_switches();
    if skip_without_the_route() {
        return;
    }

    // The class's own tap shape at an extent whose panel (224 × 131 072 × 4 =
    // 117 MB) is past the tail's floor, so im2col cannot take it either.
    const C: usize = 32;
    const KW: usize = 7;
    const L: usize = 131_066;
    let build = || {
        let mut g = Graph::new("conv1d_implicit_parked");
        let x = g.input("x", Shape::new(&[1, C, 1, L], DType::F32));
        let w = g.param("w", Shape::new(&[C, C, 1, KW], DType::F32));
        let y = g.conv2d(x, w, [1, KW], [1, 1], [0, 6], [1, 1], 1);
        g.set_outputs(vec![y]);
        g
    };
    let n_in = C * L;
    let out_len = L + 12 - (KW - 1);
    let xs = common::fill(n_in, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(C * C * KW, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(build(), &xs, &ws);

    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    let mut exe = WgpuExecutable::compile(build());
    exe.set_param("w", &ws);
    let steps = exe.test_step_names();
    let declines = exe.test_conv1d_implicit_declines();
    let panel = exe.test_conv_panel_bytes();
    let got = exe.run(&[("x", &xs)]).remove(0);
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");

    assert!(
        !steps.contains(&"conv1d_implicit"),
        "a parked weight keeps the route away ({steps:?})"
    );
    assert!(
        steps.contains(&"conv2d"),
        "and the direct conv is what runs ({steps:?})"
    );
    assert_eq!(declines, 1, "the refusal is counted ({steps:?})");
    assert_eq!(panel, 0, "no implicit step, no panel to name ({steps:?})");
    let max = common::max_abs_diff(&got, &want);
    assert!(max < 1e-2, "max|Δ| {max:e} > 1e-2");
    println!("parked weight: max|Δ| {max:e}, out {out_len}, steps {steps:?}, declines {declines}");
}

/// The route mid-schedule: two implicit convs with other steps between and after
/// them, so each runs where a real block puts it — its uniform slot and bind group
/// surrounded by other steps', its output consumed by the next step instead of the
/// readback. Same CPU reference, and the switch-off arm runs the same graph the
/// same way.
#[test]
fn a_conv_between_other_steps() {
    // See the parity cell's lock: these switches are process-global and read at
    // compile time.
    let _routes = common::route_lock();
    route_switches();
    if skip_without_the_route() {
        return;
    }

    // `conv → relu → conv → relu`, the second conv reading the first one's output
    // through another step. Both convs are the route's class; unit strides, one
    // group, `N == 1`.
    const C: usize = 32;
    const L: usize = 1024;
    const K1: usize = 5;
    const K2: usize = 3;
    let build = || {
        let mut g = Graph::new("conv1d_implicit_mid_schedule");
        let x = g.input("x", Shape::new(&[1, C, 1, L], DType::F32));
        let w1 = g.param("w1", Shape::new(&[C, C, 1, K1], DType::F32));
        let w2 = g.param("w2", Shape::new(&[C, C, 1, K2], DType::F32));
        let y1 = g.conv2d(x, w1, [1, K1], [1, 1], [0, 4], [1, 2], 1);
        let y2 = g.relu(y1);
        let y3 = g.conv2d(y2, w2, [1, K2], [1, 1], [0, 2], [1, 1], 1);
        let y4 = g.relu(y3);
        g.set_outputs(vec![y4]);
        g
    };
    let xs = common::fill(C * L, 7, 1, 13, 0.1, -0.6);
    let w1 = common::fill(C * C * K1, 5, 3, 11, 0.1, -0.5);
    let w2 = common::fill(C * C * K2, 5, 3, 11, 0.1, -0.5);
    let want = {
        let mut cpu = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(build());
        cpu.set_param("w1", &w1);
        cpu.set_param("w2", &w2);
        cpu.run(&[("x", &xs)]).remove(0)
    };

    for implicit in [true, false] {
        rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", if implicit { "1" } else { "0" });
        let mut exe = WgpuExecutable::compile(build());
        exe.set_param("w1", &w1);
        exe.set_param("w2", &w2);
        let steps = exe.test_step_names();
        let at: Vec<usize> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == "conv1d_implicit")
            .map(|(i, _)| i)
            .collect();
        if implicit {
            assert_eq!(at.len(), 2, "both convs take the route ({steps:?})");
            assert!(
                at[0] + 1 < at[1],
                "another step sits between the two convs ({steps:?})"
            );
            assert!(
                at[1] + 1 < steps.len(),
                "the second conv is not the last step ({steps:?})"
            );
            // Each conv's own panel: `k_total · out_len · 4`, at each output's own
            // length (dilation 2 then 1, so the second is the longer).
            let out1 = L + 2 * 4 - 2 * (K1 - 1);
            let out2 = out1 + 2 * 2 - (K2 - 1);
            let panel = ((C * K1) as u64 * out1 as u64 + (C * K2) as u64 * out2 as u64) * 4;
            assert_eq!(
                exe.test_conv_panel_bytes(),
                panel,
                "both convs' panels are named as removed ({steps:?})"
            );
        } else {
            assert!(at.is_empty(), "the switch off leaves the route ({steps:?})");
            assert_eq!(exe.test_conv_panel_bytes(), 0);
        }
        let max = common::max_abs_diff(&exe.run(&[("x", &xs)]).remove(0), &want);
        assert!(max < 1e-2, "implicit={implicit}: max|Δ| {max:e} > 1e-2");
        println!(
            "mid-schedule implicit={implicit}: max|Δ| {max:e}, steps {}",
            steps.len()
        );
    }
}

/// The class's output-channel floor, at the block's own extent and tap geometry:
/// the entry's 32-row tile against the direct kernel a narrower output falls to.
///
/// The entry computes a whole row tile whatever `c_out` is, so below a tile's 32
/// rows it multiplies rows the conv does not have. This cell is where the cost of
/// that crosses the cost of the panel route's removed traffic — the basis for
/// `CONV_IMPLICIT_MIN_C_OUT`.
///
/// `#[ignore]` for the extent cell's reason: it allocates the block's 101 MiB
/// operand and prints figures rather than asserting them.
#[test]
#[ignore = "measurement cell: the class's output-channel floor, at the block's extent"]
fn conv_implicit_vs_direct_by_output_channels() {
    let _routes = common::route_lock();
    route_switches();
    if skip_without_the_route() {
        return;
    }

    let n = 276_480usize;
    for c_out in [1usize, 2, 4, 8] {
        let case = Case {
            label: "96/c_out kw=7 d=3",
            c_in: 96,
            c_out,
            l: n - (2 * 18 - 3 * 6),
            kw: 7,
            dilation: 3,
            pad: 18,
        };
        let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(case.c_out * case.c_in * case.kw, 5, 3, 11, 0.1, -0.5);

        let mut arms: Vec<(bool, WgpuExecutable)> = Vec::new();
        for implicit in [true, false] {
            rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", if implicit { "1" } else { "0" });
            let mut exe = WgpuExecutable::compile(graph(&case));
            exe.set_param("w", &ws);
            let steps = exe.test_step_names();
            assert!(
                steps.contains(&"conv1d_implicit") == implicit,
                "c_out = {c_out}: arm {implicit} took {steps:?}"
            );
            exe.run(&[("x", &xs)]);
            arms.push((implicit, exe));
        }

        let panel = (case.c_in * case.kw) as u64 * n as u64 * std::mem::size_of::<f32>() as u64;
        let weight = (case.c_out * case.c_in * case.kw * std::mem::size_of::<f32>()) as u64;
        let flops = 2.0 * (case.c_out * case.c_in * case.kw * n) as f64;

        let rounds = 3;
        let calls = 2;
        let per_call = common::time_round_robin(&mut arms, rounds, calls, |(_, exe)| {
            exe.run(&[("x", &xs)]);
        });
        for (i, (implicit, exe)) in arms.iter().enumerate() {
            let ms = common::median(&per_call[i]);
            let bytes = if *implicit {
                // The gather, once, plus the weight re-read per column block.
                panel + weight * (n as u64).div_ceil(32)
            } else {
                panel * (1 + (case.c_out as u64).div_ceil(32)) + weight
            };
            // The rows the entry multiplies against rows the conv has.
            let rows = format!(
                "{} rows",
                if *implicit {
                    (case.c_out as u64).div_ceil(32) * 32
                } else {
                    case.c_out as u64
                }
            );
            println!(
                "{} c_out = {c_out}: {} {ms:.1} ms over {rows}, {:.1} GB/s over {bytes} B, \
                 {:.3} TFLOP/s, arena {} MiB ({rounds} rounds × {calls})",
                case.label,
                if *implicit { "implicit" } else { "panel" },
                bytes as f64 / (ms * 1e-3) / 1e9,
                flops / (ms * 1e-3) / 1e12,
                exe.test_arena_bytes() / (1 << 20),
            );
        }
    }
}

/// The 6-8 block's own conv at both routes, across the extents the block's windows
/// run it at: the cell the route exists for.
///
/// `#[ignore]` because it allocates the block's real extent (96 channels ×
/// 276 480 positions, ~101 MiB an operand) and is a measurement: only each arm's own
/// route and the panel it names are asserted, every figure is printed.
#[test]
#[ignore = "measurement cell: the 6-8 block's conv at both routes, by extent"]
fn conv_implicit_vs_panel_measurement() {
    // See the parity cell's lock: the two arms here move the same process-global
    // route switch the other test compiles under.
    let _routes = common::route_lock();
    route_switches();
    // The sweep exists to show the ratio as the extent shrinks, so the floor is
    // driven to zero here rather than left at the gate's own default.
    rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL", "0");
    if skip_without_the_route() {
        return;
    }

    // 276 480 is a full window's output; 88 320 is the block's short window.
    for n in [
        276_480usize,
        88_320,
        53_760,
        29_440,
        19_200,
        9_600,
        4_800,
        2_400,
    ] {
        let case = Case {
            label: "96/96 kw=7 d=3",
            c_in: 96,
            c_out: 96,
            // Chosen so the padded conv's output is `n`.
            l: n - (2 * 18 - 3 * 6),
            kw: 7,
            dilation: 3,
            pad: 18,
        };
        let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(case.c_out * case.c_in * case.kw, 5, 3, 11, 0.1, -0.5);

        // The two arms, implicit first.
        let mut arms: Vec<(bool, WgpuExecutable)> = Vec::new();
        for implicit in [true, false] {
            rlx_ir::env::set("RLX_WGPU_CONV_IMPLICIT", if implicit { "1" } else { "0" });
            let mut exe = WgpuExecutable::compile(graph(&case));
            exe.set_param("w", &ws);
            let steps = exe.test_step_names();
            assert!(
                steps.contains(&"conv1d_implicit") == implicit,
                "n = {n}: arm {implicit} took {steps:?}"
            );
            exe.run(&[("x", &xs)]);
            arms.push((implicit, exe));
        }

        // The panel, and the weight both arms read once.
        let panel = (case.c_in * case.kw) as u64 * n as u64 * std::mem::size_of::<f32>() as u64;
        let weight = (case.c_out * case.c_in * case.kw * std::mem::size_of::<f32>()) as u64;
        let flops = 2.0 * (case.c_out * case.c_in * case.kw * n) as f64;

        let rounds = 3;
        let calls = 2;
        let per_call = common::time_round_robin(&mut arms, rounds, calls, |(_, exe)| {
            exe.run(&[("x", &xs)]);
        });
        for (i, (implicit, exe)) in arms.iter().enumerate() {
            let ms = common::median(&per_call[i]);
            // Each arm's own traffic. The implicit arm gathers `k_total` floats per
            // output column — the panel's own volume, never materialised, which is
            // what `test_conv_panel_bytes` reports for it — and re-reads the weight
            // once per column block. The panel arm writes the panel once and reads it
            // back once per 32-row output tile.
            let bytes = if *implicit {
                assert_eq!(
                    exe.test_conv_panel_bytes(),
                    panel,
                    "the panel the route removed is the volume it gathers"
                );
                panel + weight * (n as u64).div_ceil(32)
            } else {
                panel * (1 + (case.c_out as u64).div_ceil(32)) + weight
            };
            println!(
                "{} n = {n}: {} {ms:.1} ms, {:.1} GB/s over {bytes} B, {:.3} TFLOP/s, arena {} MiB \
                 ({rounds} rounds × {calls})",
                case.label,
                if *implicit { "implicit" } else { "panel" },
                bytes as f64 / (ms * 1e-3) / 1e9,
                flops / (ms * 1e-3) / 1e12,
                exe.test_arena_bytes() / (1 << 20),
            );
        }
    }
}
