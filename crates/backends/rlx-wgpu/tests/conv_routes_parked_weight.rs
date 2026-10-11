// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Conv` when its weight is parked in the separate weight buffer.
//!
//! `from_plan_split` parks parameters whenever the arena needs the split plan and
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` forces that on any adapter. The accelerated conv
//! routes — im2col+GEMM, the mirrored 1-D tiled relabel, the 1×1 pointwise GEMM,
//! all under `RLX_WGPU_CONV_IM2COL` (on by default) — keep the *previous* route for a
//! conv whose weight is parked, so that configuration compiles and runs exactly as
//! it did before the routes existed. For im2col that fallback is the route's only
//! remedy: it addresses its weight by a raw `arena.offset(w) / 4` word offset, whose
//! 32-bit cast drops `WEIGHT_BUF_TAG` — so the weight would be read from the wrong
//! bytes — and the `col` panel it gathers into occupies the very scratch a staged
//! weight would need. The mirrored lift and the 1×1 pointwise GEMM are gated for the
//! same previous-behaviour-preserved reason; the pointwise route additionally
//! addresses its operands through the windowed `arena_off_in_bind_window` offsets
//! rather than a raw word cast.
//!
//! **How this proves the fallback rather than just asserting numbers.** Each case is
//! compiled and run twice in the same process. First with the routes on and the
//! weight in the activation arena — the positive control — where the shape must
//! really reach the accelerated route it is meant to prove: `im2col_gpu` for
//! `[1,64,1,2048] k=[1,7]`, `conv2d_tiled` for the mirrored `[1,64,1,300] k=[1,7]`,
//! `conv1x1_gemm` for the pointwise `[1,128,1,8192] k=[1,1]`, each clearing its
//! route's shape floor. Then with `RLX_WGPU_ALL_PARAMS_WEIGHT=1` — the weight parked
//! in the separate weight buffer — where every accelerated route must stay away and
//! the shape keeps the previous route: the plain direct `conv2d` kernel here (all
//! three are `kernel_size[0] == 1`, so none is the ONNX 1-D lift that
//! `conv1d_tiled` already served), on the same CPU reference. The
//! positive control is what makes the gate load-bearing: ungated, exactly these
//! geometries reach the accelerated route, whose raw `arena.offset(w) / 4` word
//! offset a parked weight cannot survive — the tag-dropping read described above.
//! `RLX_WGPU_CONV_IM2COL=1` is pinned so the parked weight — never an ambient
//! `=0` — is what keeps the accelerated routes away.
//!
//! Its own test binary: the switch is a process-global override, and the test leaves
//! it parked at the end, so a second test in the same binary would compile parked
//! parameters it never asked for.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

/// One unit-height 1-D conv: `[1, C_in, 1, L]` input, kernel `[1, k]`, padding
/// `[0, p]`, stride `[1, 1]`, `groups = 1`. The output is `[1, C_out, 1, L]`.
struct Case {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    l: usize,
    k: usize,
    p: usize,
    /// The accelerated step this shape must reach with the weight in the
    /// activation arena — the route whose parked-weight gate the case pins.
    accelerated: &'static str,
}

/// The route every accelerated route forks off: what a parked weight must keep.
const PREVIOUS_ROUTE: &str = "conv2d";

/// Every accelerated conv route, none of which a parked weight may reach.
const ACCELERATED_ROUTES: [&str; 3] = ["im2col_gpu", "conv2d_tiled", "conv1x1_gemm"];

/// The three families: one that takes im2col, one that takes the mirrored
/// relabel (a `conv2d_tiled` step), one that takes the 1×1 pointwise GEMM.
///
/// Each shape clears the floor of the route it names — otherwise the positive
/// control below would not be a control: `[1,64,1,2048] k=[1,7]` is a
/// `448 × 2048` col matrix (≥ `256 × 2048`), `[1,64,1,300] k=[1,7]` is 300 ≥ the
/// 256 tiled floor, and the pointwise `[1,128,1,8192] k=[1,1]` is
/// `128 × 8192` — 2× that same col-matrix floor — with `c_out = 64` on the
/// channel floor.
const CASES: [Case; 3] = [
    Case {
        label: "im2col-qualifying [1,64,1,2048] k=[1,7] p=[0,3] 64→64",
        c_in: 64,
        c_out: 64,
        l: 2048,
        k: 7,
        p: 3,
        accelerated: "im2col_gpu",
    },
    Case {
        label: "mirrored-route [1,64,1,300] k=[1,7] p=[0,3] 64→64",
        c_in: 64,
        c_out: 64,
        l: 300,
        k: 7,
        p: 3,
        accelerated: "conv2d_tiled",
    },
    Case {
        label: "pointwise [1,128,1,8192] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 8192,
        k: 1,
        p: 0,
        accelerated: "conv1x1_gemm",
    },
];

fn graph(case: &Case) -> Graph {
    let mut g = Graph::new("conv_routes_parked_weight");
    let x = g.input("x", Shape::new(&[1, case.c_in, 1, case.l], DType::F32));
    let w = g.param(
        "w",
        Shape::new(&[case.c_out, case.c_in, 1, case.k], DType::F32),
    );
    let y = g.conv2d(x, w, [1, case.k], [1, 1], [0, case.p], [1, 1], 1);
    g.set_outputs(vec![y]);
    g
}

fn fill(n: usize, mul: usize, add: usize, modu: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul + add) % modu) as f32 * scale + bias)
        .collect()
}

/// The reference: the same graph compiled for the CPU backend, so the numbers come
/// from the crate's own conv rather than from a hand-written loop.
fn reference(case: &Case, xs: &[f32], ws: &[f32]) -> Vec<f32> {
    let mut cpu = Session::new(Device::Cpu).compile(graph(case));
    cpu.set_param("w", ws);
    cpu.run(&[("x", xs)]).remove(0)
}

/// Compile `case` under the switches currently in force, bind the weight, run `x`,
/// and return the step names the schedule took, whether every step is safe for
/// active-extent (bucketed decode) dispatch, and the output.
fn compile_and_run(case: &Case, xs: &[f32], ws: &[f32]) -> (Vec<&'static str>, bool, Vec<f32>) {
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    let steps = exe.test_step_names();
    let safe = exe.test_all_steps_safe_for_active_extent();
    (steps, safe, exe.run(&[("x", xs)]).remove(0))
}

/// `got` against the CPU reference under a `1e-5` budget — measured 0 on all six
/// checks below, so the bar only leaves room for a different accumulation order.
/// `arm` names which compile it was. The fold is the shared `common::max_abs_diff`,
/// whose finiteness census runs first, so a poisoned output cannot fold to a small
/// number and pass.
fn assert_close(case: &Case, arm: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "{}: {arm} length {} vs {}",
        case.label,
        got.len(),
        want.len()
    );
    let max = common::max_abs_diff(got, want);
    assert!(max < 1e-5, "{}: {arm} max|Δ| {max:e} > 1e-5", case.label);
    max
}

#[test]
fn a_parked_conv_weight_keeps_the_conv_values_right() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // Pin the routes on so the parked weight — not an ambient `=0` — is what keeps
    // the accelerated routes away.
    rlx_ir::env::set("RLX_WGPU_CONV_IM2COL", "1");

    for case in &CASES {
        let xs = fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
        let want = reference(case, &xs, &ws);

        // Weight in the activation arena — the positive control. The shape must
        // really reach the accelerated route this case exists to prove, or the
        // parked arm below would pass while exercising nothing.
        rlx_ir::env::unset("RLX_WGPU_ALL_PARAMS_WEIGHT");
        let (steps, safe, got) = compile_and_run(case, &xs, &ws);
        assert!(
            steps.contains(&case.accelerated),
            "{}: shape does not reach {} (routes taken: {steps:?})",
            case.label,
            case.accelerated
        );
        // Active-extent availability follows the route: `im2col_gpu` cannot honour a
        // bucketed extent, the 1×1 pointwise GEMM's `m` is the output-channel axis
        // (scaling it would leave channels unwritten — why the step is
        // `from_conv1x1: true`), and `conv2d_tiled` can be scaled.
        assert_eq!(
            safe,
            case.accelerated == "conv2d_tiled",
            "{}: active-extent availability must follow the {} route (schedule {steps:?})",
            case.label,
            case.accelerated
        );
        let max = assert_close(case, "arena", &got, &want);
        eprintln!(
            "{}: {} max|Δ| {max:.2e} (n={})",
            case.label,
            case.accelerated,
            got.len()
        );

        // Weight parked in the separate weight buffer — the fallback. Every
        // accelerated route stays away, the shape keeps the previous direct conv,
        // and the values are unchanged.
        rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
        let (steps, _, got) = compile_and_run(case, &xs, &ws);
        for route in ACCELERATED_ROUTES {
            assert!(
                !steps.contains(&route),
                "{}: a parked weight must keep the {route} route away, \
                 but the schedule was {steps:?}",
                case.label
            );
        }
        assert!(
            steps.contains(&PREVIOUS_ROUTE),
            "{}: a parked weight must keep the previous direct {PREVIOUS_ROUTE} route, \
             but the schedule was {steps:?}",
            case.label
        );
        let max = assert_close(case, "parked", &got, &want);
        eprintln!(
            "{}: parked {PREVIOUS_ROUTE} max|Δ| {max:.2e} (n={})",
            case.label,
            got.len()
        );
    }
}
