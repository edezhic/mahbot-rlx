// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Conv` when its weight is parked in the separate weight buffer.
//!
//! `from_plan_split` parks parameters whenever the arena needs the split plan and
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` forces that on any adapter.
//!
//! The parked rule: the accelerated routes — im2col+GEMM, the mirrored 1-D tiled
//! relabel, the 1×1 pointwise GEMM, all under `RLX_WGPU_CONV_IM2COL` (on by default) —
//! keep the previous route for a conv whose weight is parked, so that configuration
//! compiles and runs exactly as it did before the routes existed.
//!
//! Why im2col cannot serve a parked weight, and why the other two are gated with it:
//! the im2col step addresses its weight by a raw `arena.offset(w) / 4` word offset,
//! whose 32-bit cast drops `WEIGHT_BUF_TAG` — so the weight would be read from the
//! wrong bytes — and the `col` panel it gathers into occupies the very scratch a
//! staged weight would need; the pointwise route addresses its operands through the
//! windowed `arena_off_in_bind_window` offsets rather than a raw word cast, and the
//! mirrored lift is gated for the same previous-behaviour-preserved reason.
//!
//! On the arm that routes, each GEMM picks its own kernel: on a Metal adapter the
//! f32 cooperative one where that kernel's alignment holds (both GEMM shapes below
//! are aligned, so what is printed here is that kernel's numbers there), the wide
//! f32 kernel on any other adapter and for any shape the alignment refuses — see
//! `conv_gemm`, and `conv_gemm_coop_f32.rs` for the pin on that choice. Both arms
//! are held to the crate's `1e-5` bar against the CPU conv
//! (`common::assert_close_under`); this rig measures 0 on every check below.
//!
//! **How this proves the fallback rather than just asserting numbers.** Each case is
//! compiled and run twice in the same process. First with the routes on and the
//! weight in the activation arena — the positive control — where the shape must
//! really reach the accelerated route it is meant to prove: `im2col_gpu` for
//! `[1,64,1,2048] k=[1,7]`, `conv2d_tiled` for the mirrored
//! `[1,64,1,300] k=[1,7] 64→32` (the shape reaches the relabel through the im2col
//! route's 64-channel floor, a gate the relabel does not test, now that the spatial
//! floor at 100 no longer keeps it off the materialised route),
//! `conv1x1_gemm` for the pointwise `[1,128,1,8192] k=[1,1]`. Then with
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` — the weight parked in the separate weight buffer —
//! where every accelerated route must stay away and
//! the shape keeps the previous route: the plain direct `conv2d` kernel here (all
//! four are `kernel_size[0] == 1`, so none is the ONNX 1-D lift that
//! `conv1d_tiled` already served), on the same CPU reference. The
//! positive control is what makes the gate load-bearing: ungated, exactly these
//! geometries reach the accelerated route, whose raw `arena.offset(w) / 4` word
//! offset a parked weight cannot survive — the tag-dropping read described above.
//! `RLX_WGPU_CONV_IM2COL=1` is pinned so the parked weight — never an ambient
//! `=0` — is what keeps the accelerated routes away. Both arms of
//! `im2col_fallbacks()` are asserted here — zero on the arm that routes, the case's
//! `parked_fallbacks` on the other — which is what makes the half of the im2col
//! decision the shapes alone do not settle (see the `Op::Conv` arm where the count
//! is taken) observable rather than inferred from a timing; `conv_route_cost.rs`
//! prints the same count in the same parked configuration.
//!
//! Its own test binary: the switch is a process-global override, and the test leaves
//! it parked at the end, so a second test in the same binary would compile parked
//! parameters it never asked for.

use rlx_ir::Graph;
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
    /// The step this shape must reach with the weight in the activation arena: for the
    /// first three cases an accelerated route — the one whose parked-weight gate the case
    /// pins — and for the fourth the direct `conv2d`, which is what the 1×1 route's frozen
    /// bar keeps it on.
    route: &'static str,
    /// The `im2col_fallbacks()` count this case must report when its weight is parked.
    parked_fallbacks: usize,
}

/// The route every accelerated route forks off: what a parked weight must keep.
const PREVIOUS_ROUTE: &str = "conv2d";

/// Every accelerated conv route, none of which a parked weight may reach.
const ACCELERATED_ROUTES: [&str; 3] = ["im2col_gpu", "conv2d_tiled", "conv1x1_gemm"];

/// The four cases: one that takes im2col, one that takes the mirrored relabel (a
/// `conv2d_tiled` step), one that takes the 1×1 pointwise GEMM, and one that
/// deliberately clears *no* accelerated floor.
///
/// Each of the first three shapes reaches the route it names — otherwise the
/// positive control below would not be a control: `[1,64,1,2048] k=[1,7]` sits well
/// inside the im2col floors (spatial 2048 ≥ 100, `k_total` 448 ≥ 256),
/// `[1,64,1,300] k=[1,7] 64→32` fails the im2col route's 64-channel floor — the gate
/// the relabel does not test — and so still reaches the relabel (300 ≥ the 256 tiled
/// floor), and the pointwise `[1,128,1,8192] k=[1,1]` is `128 × 8192` — 2× the 1×1
/// route's own `256 × 2048` bar — with `c_out = 64` on the channel floor. The
/// fourth, `[1,128,1,3000] k=[1,1]`, is `128 × 3000 = 384 000`: under the 1×1
/// route's frozen bar (`256 × 2048` = 524 288) but over the product that bar would
/// be if the 1×1 admission were still coupled to the floors
/// (`256 × 100` = 25 600), so it keeps the direct kernel — and re-coupling the two
/// decisions fails this case on its route assertion.
const CASES: [Case; 4] = [
    Case {
        label: "im2col-qualifying [1,64,1,2048] k=[1,7] p=[0,3] 64→64",
        c_in: 64,
        c_out: 64,
        l: 2048,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        parked_fallbacks: 1,
    },
    Case {
        label: "mirrored-route [1,64,1,300] k=[1,7] p=[0,3] 64→32",
        c_in: 64,
        c_out: 32,
        l: 300,
        k: 7,
        p: 3,
        route: "conv2d_tiled",
        parked_fallbacks: 0,
    },
    Case {
        label: "pointwise [1,128,1,8192] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 8192,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        parked_fallbacks: 0,
    },
    // The two decisions pinned apart. The 1×1 route's admission bar is the frozen
    // `256 × 2048` (`backend::POINTWISE_MIN_COL_ELEMS`), not a product of the im2col
    // floors, so a shape over the coupled product (`256 × 100` = 25 600) but
    // under that bar keeps the direct kernel on both arms — the route assertion is
    // what fails if the two decisions are ever coupled again.
    Case {
        label: "pointwise-below-its-own-bar [1,128,1,3000] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 3000,
        k: 1,
        p: 0,
        route: "conv2d",
        parked_fallbacks: 0,
    },
];

/// This file's own geometry, from the fixtures `common` shares with
/// `conv_gemm_coop_f32.rs` (see `common::unit_conv_graph`).
fn graph(case: &Case) -> Graph {
    common::unit_conv_graph(
        "conv_routes_parked_weight",
        case.c_in,
        case.c_out,
        case.l,
        case.k,
        case.p,
    )
}

/// Compile `case` under the switches currently in force, bind the weight, run `x`,
/// and return the step names the schedule took, whether every step is safe for
/// active-extent (bucketed decode) dispatch, the output, and the number of convs
/// that qualified for im2col by shape but kept the direct/tiled route.
fn compile_and_run(
    case: &Case,
    xs: &[f32],
    ws: &[f32],
) -> (Vec<&'static str>, bool, Vec<f32>, usize) {
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    let steps = exe.test_step_names();
    let safe = exe.test_all_steps_safe_for_active_extent();
    let fallbacks = exe.im2col_fallbacks();
    (steps, safe, exe.run(&[("x", xs)]).remove(0), fallbacks)
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
        let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
        let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);

        // Weight in the activation arena — the positive control. The shape must
        // really reach the route this case exists to prove, or the parked arm below
        // would pass while exercising nothing.
        rlx_ir::env::unset("RLX_WGPU_ALL_PARAMS_WEIGHT");
        let (steps, safe, got, fallbacks) = compile_and_run(case, &xs, &ws);
        assert!(
            steps.contains(&case.route),
            "{}: shape does not reach {} (routes taken: {steps:?})",
            case.label,
            case.route
        );
        // Active-extent availability follows the route: `im2col_gpu` cannot honour a
        // bucketed extent, the 1×1 pointwise GEMM's `m` is the output-channel axis
        // (scaling it would leave channels unwritten — why the step is
        // `from_conv1x1: true`), while the direct `conv2d` kernel and the scaled
        // `conv2d_tiled` one can be.
        assert_eq!(
            safe,
            matches!(case.route, "conv2d_tiled" | "conv2d"),
            "{}: active-extent availability must follow the {} route (schedule {steps:?})",
            case.label,
            case.route
        );
        // The other arm of the fallback count: the shape reaches the route it
        // qualifies for, so no allocation term (a parked weight, a col past the
        // reserved tail, an arena that will not bind whole) keeps it off that route.
        assert_eq!(
            fallbacks, 0,
            "{}: a shape that routes must count no im2col fallback",
            case.label
        );
        let max = common::assert_close_under(case.label, "arena", &got, &want);
        eprintln!(
            "{}: {} max|Δ| {max:.2e} (n={}, im2col fallbacks {fallbacks})",
            case.label,
            case.route,
            got.len()
        );

        // Weight parked in the separate weight buffer — the fallback. Every
        // accelerated route stays away, the shape keeps the previous direct conv,
        // and the values are unchanged.
        rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
        let (steps, _, got, fallbacks) = compile_and_run(case, &xs, &ws);
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
        assert_eq!(
            fallbacks, case.parked_fallbacks,
            "{}: a parked weight must report the fallback count for this shape \
             (schedule {steps:?})",
            case.label
        );
        let max = common::assert_close_under(case.label, "parked", &got, &want);
        eprintln!(
            "{}: parked {PREVIOUS_ROUTE} max|Δ| {max:.2e} (n={}), im2col fallbacks {} (want {})",
            case.label,
            got.len(),
            fallbacks,
            case.parked_fallbacks
        );
    }
}
