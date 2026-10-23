// RLX — versatile ML compiler & runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The convolution GEMM routes' multiplies take the f32 cooperative kernel
//! exactly where the alignment these routes ask of it holds.
//!
//! The routes are the im2col panel multiply and the 1×1 pointwise GEMM. Both put
//! the weight in the FIRST operand — the reverse of every other matmul call site —
//! which is what used to keep them off the cooperative path: it was reserved for
//! a multiply whose SECOND operand is a Param. What changed is that these two sites
//! now ask that path's admission (`conv_gemm`), with the roles swapped, and they
//! ask it for the portable output-column alignment (`n % 8`) instead of the Metal
//! form's own 32 — their one widening, explained once on the admission itself
//! (`coop_f32_admitted` in `backend/mod.rs`). So `RLX_WGPU_NO_COOP_F32`,
//! `RLX_WGPU_FORCE_COOP_F32`, `RLX_WGPU_MATMUL_F32_ONLY` and the per-device
//! operand-order probe govern them as they govern any other multiply, and an extent
//! the portable form admits where the Metal form alone would refuse is part of what
//! this file pins.
//!
//! **Alignment decides, not the call.** Each case below is compiled and run twice
//! in one process, once with the cooperative path available and once under
//! `RLX_WGPU_NO_COOP_F32=1`. The step names carry no precision, so the witness is
//! the route's own per-dispatch counter
//! (`rlx_wgpu::coop_f32_dispatch_count()`), which both arms are read around:
//! an admitted shape must show exactly one cooperative dispatch with the arm on
//! and zero with it off, and a shape the alignment refuses must show zero either
//! way — the `coop` field is computed from the alignment these routes ask the
//! kernel for (the contraction extent a multiple of 8, and the output-column extent
//! a multiple of 8 rather than the Metal form's own 32) and never from what the run
//! did.
//!
//! **The refused shapes' outputs do not move**, and that is asserted exactly:
//! with the arm off they take the same kernel they take with it on, so the two
//! builds' outputs are compared bit-for-bit.
//!
//! **The admitted shapes' outputs are compared, not assumed.** Both kernels are
//! held to the route's existing `1e-5` parity bar against the crate's own CPU
//! conv, and then against each other, with the maximum difference printed
//! (`common::max_abs_diff`, whose finiteness census runs first). Nothing here
//! asserts equality: the tree fixes no accumulation order for the cooperative
//! path, so a kernel that reordered the contraction would move these numbers,
//! and the bar is what has to hold when it does.
//!
//! **Measured** (the rig: Apple M3 Pro, Metal, macOS 26.5, shipped defaults, no
//! tuning cache), on the routes' own contraction extents (448, 336, `512 × 7` =
//! 3584, and 128, 40 — every extent the run prints) rather than on nodes below
//! them: cooperative vs wide `max|Δ|` = `0.0` on every admitted shape, and each
//! arm `0.0` against the CPU conv. That the two kernels' accumulation orders land
//! on the same bits is a fact about this device, not a property of either: no
//! contract fixes an order, reordered f32 sums over these contractions differ in
//! general, and the `1e-5` bar — not the zero — is what has to hold when they do.
//! The `n = 5000` shape — a multiple of 8 and not of 32, so its x grid rounds up —
//! is `0.0` against the CPU conv on the cooperative arm too, in the loop and again
//! under the lever at the end. Every figure in this note, and the front-convolution
//! rows the README takes from this file's trailing block, are runs of this test; the
//! README marks the readings there that no in-tree graph or harness takes.
//!
//! **The ordinary multiply is not the widening's, and not the direct entry's.** A
//! plain matmul keeps the Metal form's own 32 columns: the counter reads zero for an
//! `n` that is a multiple of 8 only and one for the next shape up — the block at the
//! end of the test — so the cascade's own alignment is what this change leaves alone.
//! It also never takes `matmul_coop_f32_direct`: that entry is admitted only where
//! the weight sits in the FIRST operand (`coop_f32_direct_admitted`), which only
//! these two routes do.
//!
//! **The padded-`M` cases are the bounds reliance, exercised.** The Metal kernel
//! gathers 32 A rows per tile without checking `params.m`, so a `c_out = 80` shape
//! — not a multiple of 32 — makes it read 16 rows past the weight's extent: the
//! read `kernels/matmul_coop_f32.wgsl` states the reliance for, whose accumulators
//! must reach no output. Both routes carry such a shape, because their A windows
//! are not the same one — the im2col panel multiply binds the whole arena, the
//! pointwise GEMM the conv node's own window — so the read is checked against a
//! whole-arena bind and against a bounded one. The padded columns are the mirror
//! case on the column axis, admitted on the route's own alignment now (`n = 5000`,
//! a multiple of 8 and not of 32): the CPU parities this case takes in the loop, and
//! again under the lever at the end, are what hold them to no output.
//!
//! The case set is the route's own geometries: `conv_routes_parked_weight.rs`
//! pins `[1,64,1,2048] k=[1,7]` (im2col), `[1,128,1,8192] k=[1,1]` (pointwise)
//! and the mirrored `[1,64,1,300]` lift. Both axes the alignment reads get a
//! refused member per class: the output-column extent (`2048 → 2050` on im2col,
//! `2050 % 8 = 2`, and `8192 → 5004` on pointwise, `5004 % 8 = 4` — the two shapes
//! here that fail the route's own portable alignment on that axis, each the same
//! route with the same admission floors as its admitted sibling, only that extent
//! moved) and the contraction extent (`448 → 420` on im2col, `128 → 100` on
//! pointwise, both `% 8 = 4`, each with an output-column extent the alignment
//! accepts, so the contraction axis alone refuses them). The output-column axis
//! also gets a member the portable form admits and the Metal form's own 32 alone
//! would not (`8192 → 5000`, `5000 % 8 = 0` and `5000 % 32 = 8`), which is the
//! widening this file is shaped around; the rest are the axes the alignment is
//! stated in — a contraction that is a multiple of 8 only, one deepened past the
//! routed default, and an `M` that is not a multiple of 32.
//!
//! **The dispatch-tuning cache is pinned off** (`RLX_GPU_TUNING_CACHE`, a path
//! that does not exist) because these two sites now read the shared per-shape
//! tuning table as any other multiply does: a persisted override for one of these
//! shapes can pin it back to the wide kernel and silently cancel the move, which
//! would read as an alignment failure here when it is a tuning-cache fact.
//!
//! Its own test binary: the arm is a process-global switch and the counter is
//! process-global too.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

/// One case: the conv the graph is built from (`c_in`/`c_out`/`l`/`k`/`p`, see
/// `common::unit_conv_graph`), the route it must reach on both arms, and how the
/// arm must behave on it.
struct Case {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    l: usize,
    k: usize,
    p: usize,
    /// The route this shape's conv must reach, on both arms.
    route: &'static str,
    /// Whether this shape's conv runs a GEMM the alignment these routes ask the
    /// f32 cooperative kernel for admits: `Some(true)` admitted, `Some(false)`
    /// refused, and `None` for a conv that reaches no GEMM route at all — the
    /// mirrored lift, whose place in the set is the witness that the arm switches
    /// neither it nor any other node outside the two classes.
    coop: Option<bool>,
}

/// Nine admitted shapes, four the arm may not switch, and one inert.
///
/// The two classes are the im2col panel multiply and the pointwise GEMM, and each
/// axis the alignment reads gets a refused member per class: the output-column
/// extent (`2050` on im2col, `2050 % 8 = 2`, and `5004` on pointwise, `5004 % 8 =
/// 4` — the same routes with the same admission floors as their admitted
/// siblings, only that extent moved) and the contraction extent (`420`, `100` —
/// both `% 8 = 4`, each with an output-column extent the alignment accepts, so the
/// contraction axis alone is what refuses them). The last member is the mirrored
/// `conv2d_tiled` lift, which is not a GEMM route at all: its place in the set is
/// the witness that the arm reaches neither it nor any other node outside the two
/// classes.
///
/// The admitted set covers the alignment's own terms: `n` a multiple of 32 as well
/// as one that is a multiple of 8 only (`5000`, whose `% 32 = 8` — the portable
/// extent the Metal form alone refuses, carried by the pointwise route with a
/// padded `M` as well, so the two padded axes meet in one case); `k` a multiple of
/// 32 as well as one that is a multiple of 8 only (`336`, `40` — the axis the
/// shader's `REQUIREMENTS` names as `8`, not `32`); a contraction deepened past the
/// routed default; and an `M` that is not a multiple of 32 on **both** routes —
/// because their A windows are not the same one (the im2col panel multiply binds
/// the whole arena, the pointwise GEMM the conv node's own window), so the
/// padded-row read is checked against both a whole-arena and a bounded bind.
///
/// The recognition front's own third convolution is not a case here: its three
/// extents are pinned at the end of the test instead, on its real 2-D geometry,
/// where the kernel and not this fixture's protocol decides.
///
/// The im2col shapes' contraction extent is `c_in × kh × kw` (448, 420, 336 and
/// 3584 for the deep one) and the pointwise ones' is `c_in` (128, 100, 40).
const CASES: [Case; 14] = [
    Case {
        label: "im2col [1,64,1,2048] k=[1,7] p=[0,3] 64→64",
        c_in: 64,
        c_out: 64,
        l: 2048,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(true),
    },
    Case {
        label: "pointwise [1,128,1,8192] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 8192,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(true),
    },
    Case {
        label: "im2col-deep [1,512,1,1024] k=[1,7] p=[0,3] 512→64",
        c_in: 512,
        c_out: 64,
        l: 1024,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(true),
    },
    Case {
        label: "im2col-padded-M [1,64,1,2048] k=[1,7] p=[0,3] 64→80",
        c_in: 64,
        c_out: 80,
        l: 2048,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(true),
    },
    Case {
        label: "im2col-K8 [1,48,1,2048] k=[1,7] p=[0,3] 48→64",
        c_in: 48,
        c_out: 64,
        l: 2048,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(true),
    },
    Case {
        label: "pointwise-K8 [1,40,1,13184] k=[1,1] p=[0,0] 40→64",
        c_in: 40,
        c_out: 64,
        l: 13184,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(true),
    },
    Case {
        label: "pointwise-padded-M [1,128,1,8192] k=[1,1] p=[0,0] 128→80",
        c_in: 128,
        c_out: 80,
        l: 8192,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(true),
    },
    Case {
        label: "pointwise-n8-only [1,128,1,5000] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 5000,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(true),
    },
    Case {
        label: "pointwise-n8-only-padded-M [1,128,1,5000] k=[1,1] p=[0,0] 128→80",
        c_in: 128,
        c_out: 80,
        l: 5000,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(true),
    },
    Case {
        label: "im2col-unaligned [1,64,1,2050] k=[1,7] p=[0,3] 64→64",
        c_in: 64,
        c_out: 64,
        l: 2050,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(false),
    },
    Case {
        label: "pointwise-unaligned [1,128,1,5004] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 5004,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(false),
    },
    Case {
        label: "im2col-unaligned-K [1,60,1,2048] k=[1,7] p=[0,3] 60→64",
        c_in: 60,
        c_out: 64,
        l: 2048,
        k: 7,
        p: 3,
        route: "im2col_gpu",
        coop: Some(false),
    },
    Case {
        label: "pointwise-unaligned-K [1,100,1,8192] k=[1,1] p=[0,0] 100→64",
        c_in: 100,
        c_out: 64,
        l: 8192,
        k: 1,
        p: 0,
        route: "conv1x1_gemm",
        coop: Some(false),
    },
    Case {
        label: "mirrored lift [1,64,1,300] k=[1,7] p=[0,3] 64→32",
        c_in: 64,
        c_out: 32,
        l: 300,
        k: 7,
        p: 3,
        route: "conv2d_tiled",
        coop: None,
    },
];

/// This file's own geometry, from the fixtures `common` shares with
/// `conv_routes_parked_weight.rs` (see `common::unit_conv_graph`).
fn graph(case: &Case) -> Graph {
    common::unit_conv_graph(
        "conv_gemm_coop_f32",
        case.c_in,
        case.c_out,
        case.l,
        case.k,
        case.p,
        1,
    )
}

/// Compile `case` under the switches in force, bind the weight, run `x`, and
/// report the step names the schedule took, the output, and how many f32
/// cooperative dispatches the run made — of which how many took the direct
/// (epilogue-free) entry. One conv in the graph, so each count is this node's.
fn compile_and_run(case: &Case, xs: &[f32], ws: &[f32]) -> (Vec<&'static str>, Vec<f32>, u64, u64) {
    let before = rlx_wgpu::coop_f32_dispatch_count();
    let before_direct = rlx_wgpu::coop_f32_direct_dispatch_count();
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    let steps = exe.test_step_names();
    let got = exe.run(&[("x", xs)]).remove(0);
    let direct = rlx_wgpu::coop_f32_direct_dispatch_count() - before_direct;
    (
        steps,
        got,
        rlx_wgpu::coop_f32_dispatch_count() - before,
        direct,
    )
}

/// The two things a shape the arm may not switch must show: no cooperative
/// dispatch, and the two arms' outputs equal bit for bit — the arm off is the
/// arm on, so there is no kernel to have changed. `why` names the reading that
/// keeps today's kernel (`Some(false)` an alignment refusal, `None` a conv that
/// reaches no GEMM at all).
fn assert_arm_inert(case: &Case, coop_dispatches: u64, got: &[f32], got_base: &[f32], why: &str) {
    assert_eq!(coop_dispatches, 0, "{}: {why}", case.label);
    common::assert_bit_identical(
        got,
        got_base,
        "a conv node the cooperative arm may not switch",
    );
}

/// Whether this adapter can dispatch the f32 cooperative kernel the two
/// convolution GEMM routes can reach — Metal's 32×32 form, plus the
/// operand-order probe the admission ends on. The portable 8×8 form is not it:
/// these routes are host-staged on every Vulkan/DX12 adapter. Without the path
/// an aligned conv keeps the wide kernel and there is nothing to witness, so the
/// test self-skips — the `require_coop_f32_test` precedent in
/// `coop_f32_correctness.rs`, not `env::skip_unless_device`, because an adapter
/// that is present but has no cooperative matrices is not a device failure.
fn coop_f32_metal_available() -> bool {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        return false;
    };
    dev.backend == wgpu::Backend::Metal && rlx_wgpu::coop_probe::f32_path_ok(Some(dev.backend))
}

#[test]
fn alignment_decides_which_conv_gemms_take_the_cooperative_kernel() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
        // A parked weight keeps every accelerated route away, and the tiled
        // route's own switch moves the mirrored lift — both would take a case off
        // the route it is here to prove.
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_NO_TILED_CONV",
        // Both sit above the cooperative path: the f32-only pin rules it out
        // entirely, and FORCE widens past the alignment this pin is about.
        "RLX_WGPU_MATMUL_F32_ONLY",
        "RLX_WGPU_FORCE_COOP_F32",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    // The routes on, so an arm that takes one is the route's doing.
    rlx_ir::env::set("RLX_WGPU_CONV_IM2COL", "1");
    // Pin the shared dispatch table off: a persisted override for one of these
    // shapes would pin it back to the wide kernel and read as an alignment
    // failure below. Read once per process, on the first `gpu_arch()`.
    rlx_ir::env::set(
        "RLX_GPU_TUNING_CACHE",
        std::env::temp_dir()
            .join("rlx-conv-gemm-coop-f32-absent.tsv")
            .to_string_lossy()
            .into_owned(),
    );
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    if !coop_f32_metal_available() {
        eprintln!(
            "no f32 cooperative-matrix path on this adapter — skipping \
             (the portable 8×8 form is not admitted for these routes)"
        );
        return;
    }

    for case in &CASES {
        let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
        let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
        // This family's kernel is `[1, k]`, so the GEMM's contraction extent is
        // `c_in × 1 × k` and its output-column extent is the conv's spatial one.
        let k_total = case.c_in * case.k;

        rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "0");
        let (steps_coop, got_coop, coop_dispatches, direct_dispatches) =
            compile_and_run(case, &xs, &ws);
        rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "1");
        let (steps_base, got_base, base_dispatches, _) = compile_and_run(case, &xs, &ws);

        for (arm, steps) in [("arm", &steps_coop), ("base", &steps_base)] {
            assert!(
                steps.contains(&case.route),
                "{}: the {arm} must still reach {} (schedule {steps:?})",
                case.label,
                case.route
            );
        }
        // The switch's own claim, on every shape: it takes the cooperative arm
        // away, whether or not the shape would have taken it.
        assert_eq!(
            base_dispatches, 0,
            "{}: RLX_WGPU_NO_COOP_F32=1 must leave no cooperative dispatch",
            case.label
        );
        let max_coop = common::assert_close_under(case.label, "cooperative", &got_coop, &want);
        let max_base = common::assert_close_under(case.label, "wide", &got_base, &want);

        match case.coop {
            Some(true) => {
                assert_eq!(
                    coop_dispatches, 1,
                    "{}: the route's alignment admits this shape's GEMM (k = {}, n = {}, \
                     each % 8 = 0), so its one conv must dispatch the cooperative \
                     kernel once",
                    case.label, k_total, case.l
                );
                // Both routes' multiply is epilogue-free, so a whole 32×32 output
                // tile takes the direct entry (`matmul_coop_f32_direct`). The
                // expectation is the dispatch's own admission, asked with this
                // route's operand roles (weight in A, so `b_is_param` is false):
                // `coop_f32_direct_admitted`. A case with a padded m or n keeps
                // the general kernel, whose store guard and scalar epilogue cover
                // the ragged tile.
                let want_direct = rlx_wgpu::backend::coop_f32_direct_admitted(
                    0,
                    0xFFFF,
                    case.c_out as u32,
                    k_total as u32,
                    case.l as u32,
                    false,
                );
                assert_eq!(
                    direct_dispatches,
                    u64::from(want_direct),
                    "{}: the direct entry is exactly the route-scoped, whole-tile, \
                     epilogue-free form `coop_f32_direct_admitted` names",
                    case.label
                );
                let d = common::max_abs_diff(&got_coop, &got_base);
                eprintln!(
                    "{}: cooperative vs wide max|Δ| {d:e}, vs CPU {max_coop:e} / {max_base:e} (n={})",
                    case.label,
                    got_coop.len()
                );
                // Two arms each inside the route's `1e-5` bar against the same
                // reference are at most `2e-5` apart, so that — not `1e-5` — is
                // what the arm-to-arm difference can be held to. Nothing here
                // asserts equality: the cooperative kernel's accumulation order is
                // its own, and this is the bar for when it moves the numbers.
                assert!(
                    d < 2e-5,
                    "{}: cooperative vs wide max|Δ| {d:e} past the 2e-5 two conforming \
                     arms can differ by",
                    case.label
                );
            }
            Some(false) => {
                assert_arm_inert(
                    case,
                    coop_dispatches,
                    &got_coop,
                    &got_base,
                    "the alignment refuses this shape's GEMM, so today's kernel is the \
                     one it keeps",
                );
                eprintln!(
                    "{}: not admitted — arm on/off bit-identical (n={}), vs CPU {max_coop:e}",
                    case.label,
                    got_coop.len()
                );
            }
            None => {
                assert_arm_inert(
                    case,
                    coop_dispatches,
                    &got_coop,
                    &got_base,
                    "this conv reaches no GEMM route, so the arm has nothing to switch",
                );
                eprintln!(
                    "{}: outside the two classes — arm on/off bit-identical (n={}), vs CPU \
                     {max_coop:e}",
                    case.label,
                    got_coop.len()
                );
            }
        }
    }

    // The f32-only pin governs these classes too: it is read by the same
    // admission they ask, not only by the cascade the ordinary multiply goes
    // through — a pin these two sites could route around would not be the pin
    // the registry documents.
    let case = &CASES[0];
    let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
    // The cooperative arm back on, so the pin is the only thing keeping the
    // kernel away — the last case above left it switched off.
    rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "0");
    rlx_ir::env::set("RLX_WGPU_MATMUL_F32_ONLY", "1");
    let (steps, got, dispatches, _) = compile_and_run(case, &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_MATMUL_F32_ONLY", "0");
    assert!(
        steps.contains(&case.route),
        "{}: the route is unchanged by the pin (schedule {steps:?})",
        case.label
    );
    assert_eq!(
        dispatches, 0,
        "{}: RLX_WGPU_MATMUL_F32_ONLY=1 must keep the cooperative kernel away from \
         a shape its alignment admits",
        case.label
    );
    common::assert_close_under(case.label, "f32-only", &got, &want);
    eprintln!(
        "{}: RLX_WGPU_MATMUL_F32_ONLY=1 — no cooperative dispatch, vs CPU {:.2e}",
        case.label,
        common::max_abs_diff(&got, &want)
    );

    // The force lever's geometry term is what these classes ask for already, so on
    // a shape they admit it moves nothing: `RLX_WGPU_FORCE_COOP_F32=1` widens the
    // ordinary multiply's output-column extent from a multiple of 32 to a multiple
    // of 8, and this route asks for that 8 — the Metal scope and the operand-order
    // probe being the terms it does not touch. Read on two shapes, so "the alignment
    // and not the call decides" holds under the lever here as well.
    //
    // The first is admitted on the route's own alignment (`n = 5000`, a multiple of
    // 8 and not of 32), where the lever must change neither the dispatch nor the
    // output: both arms take the cooperative kernel. Its column axis is padded, so
    // the output is held to the route's CPU bar rather than merely counted.
    let case = CASES
        .iter()
        .find(|c| c.label.starts_with("pointwise-n8-only "))
        .expect("the pointwise multiple-of-8-only case");
    let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "1");
    let (steps, got_forced, forced, _) = compile_and_run(case, &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "0");
    let (_, got_off, off, _) = compile_and_run(case, &xs, &ws);
    assert!(
        steps.contains(&case.route),
        "{}: the route is unchanged by the force lever (schedule {steps:?})",
        case.label
    );
    assert_eq!(
        forced, 1,
        "{}: the route's own alignment admits this shape, so the lever — the same \
         widening it asks for — must leave that admission standing",
        case.label
    );
    assert_eq!(
        off, 1,
        "{}: with the lever off the shape keeps the kernel its own alignment admits, \
         which is why the lever moves nothing on this class",
        case.label
    );
    common::assert_close_under(case.label, "forced", &got_forced, &want);
    common::assert_close_under(case.label, "unforced", &got_off, &want);
    eprintln!(
        "{}: RLX_WGPU_FORCE_COOP_F32=1 — cooperative dispatch {forced} with the lever \
         on, {off} off (n={}), vs CPU {:.2e}",
        case.label,
        case.l,
        common::max_abs_diff(&got_forced, &want)
    );

    // The second is a shape neither alignment accepts (`n = 2050`, `n % 8 = 2`),
    // which is where the lever still has to be inert: what it widens is the Metal
    // form's own 32, to the portable 8 these routes already ask for, and `2050` fails
    // even that — no dispatch under it, as none without it (the case loop above took
    // this shape's two arms bit-identical), so there is no kernel for it to have
    // changed.
    let case = CASES
        .iter()
        .find(|c| c.label.starts_with("im2col-unaligned "))
        .expect("the im2col output-column refusal");
    let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "1");
    let (steps, got_forced, forced, _) = compile_and_run(case, &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "0");
    assert!(
        steps.contains(&case.route),
        "{}: the route is unchanged by the force lever (schedule {steps:?})",
        case.label
    );
    assert_eq!(
        forced, 0,
        "{}: RLX_WGPU_FORCE_COOP_F32=1 widens to the portable 8 this shape fails \
         (n % 8 = 2), so it keeps today's kernel",
        case.label
    );
    common::assert_close_under(case.label, "forced", &got_forced, &want);
    eprintln!(
        "{}: RLX_WGPU_FORCE_COOP_F32=1 — still no cooperative dispatch on an n % 8 ≠ 0 \
         shape (n={}), vs CPU {:.2e}",
        case.label,
        case.l,
        common::max_abs_diff(&got_forced, &want)
    );

    // The two routes' newly admitted extents, priced: the front's own third
    // convolution — 3×3 taps, stride 2, pad 1, 480 → 512, so the contraction is
    // 4320 (480 × 3 × 3) and `m` is not trivial — at the three input widths its own
    // lengths reach (19, 25, 31, i.e. the extents 160, 208 and 256), and the 1×1
    // pointwise route at the extent the case loop admits above it (`5000`, a
    // multiple of 8 and not of 32), for which no route-cost cell exists.
    for l in [25usize, 19, 31] {
        let mut g = Graph::new("front_third_conv");
        let x = g.input("x", Shape::new(&[1, 480, 31, l], DType::F32));
        let w = g.param("w", Shape::new(&[512, 480, 3, 3], DType::F32));
        let y = g.conv2d(x, w, [3, 3], [2, 2], [1, 1], [1, 1], 1);
        g.set_outputs(vec![y]);
        let xs = common::fill(480 * 31 * l, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(512 * 480 * 9, 5, 3, 11, 0.1, -0.5);
        time_two_arms(g, &xs, &ws, "front third conv", "im2col_gpu");
    }
    // The other caller of the widened admission, at the extent the case loop admits
    // above it: 1×1 pointwise, 128 → 64, a multiple of 8 and not of 32.
    let mut g = Graph::new("pointwise_n8_only");
    let x = g.input("x", Shape::new(&[1, 128, 1, 5000], DType::F32));
    let w = g.param("w", Shape::new(&[64, 128, 1, 1], DType::F32));
    let y = g.conv2d(x, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
    g.set_outputs(vec![y]);
    let xs = common::fill(128 * 5000, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(64 * 128, 5, 3, 11, 0.1, -0.5);
    time_two_arms(g, &xs, &ws, "pointwise 1×1", "conv1x1_gemm");

    // And the ordinary multiply's own rule, which this change leaves alone: a plain
    // matmul asks the Metal form's own 32, so an `n` that is a multiple of 8 only
    // takes the wide kernel while the next shape up takes the cooperative one. The
    // counter is the witness either way — the two arms' step names carry no
    // precision.
    for (n, want) in [(5000u32, 0u64), (5056, 1)] {
        let mut g = Graph::new("plain_matmul_n");
        let x = g.input("x", Shape::new(&[64, 256], DType::F32));
        let w = g.param("w", Shape::new(&[256, n as usize], DType::F32));
        let y = g.mm(x, w);
        g.set_outputs(vec![y]);
        let xs = common::fill(64 * 256, 7, 1, 13, 0.1, -0.6);
        let ws = common::fill(256 * n as usize, 5, 3, 11, 0.1, -0.5);
        let before = rlx_wgpu::coop_f32_dispatch_count();
        let mut exe = WgpuExecutable::compile(g);
        exe.set_param("w", &ws);
        let _ = exe.run(&[("x", &xs)]);
        let dispatches = rlx_wgpu::coop_f32_dispatch_count() - before;
        assert_eq!(
            dispatches,
            want,
            "a plain multiply with n = {n} (n % 32 = {}) must take the cooperative kernel \
             {want} time(s), as it did before the conv routes' widening",
            n % 32
        );
        eprintln!(
            "plain matmul n = {n} (n % 32 = {}): cooperative dispatches {dispatches}",
            n % 32
        );
    }
    rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "0");
}

/// The spatial extent a conv GEMM node's multiply works over — the product of its
/// output's two trailing dims — off the graph itself, so a caller cannot label the
/// node with an extent its own geometry does not produce.
fn conv_extent(g: &Graph) -> usize {
    g.node(g.outputs[0]).shape.dims()[2..]
        .iter()
        .map(|d| d.unwrap_static())
        .product()
}

/// Compile one node under both arms — `RLX_WGPU_NO_COOP_F32=1` and the shaped
/// admission — hold it to `route`, to exactly one cooperative dispatch, to both
/// arms' place against the crate's own CPU conv and to the same 2e-5 arm-to-arm bar
/// the case loop uses, time the arms interleaved for three rounds, and print the
/// medians, the arm difference, both arms' CPU difference and the arena. The
/// cooperative counter is read around the shaped arm's run alone, so the wide arm's
/// own run cannot enter the count.
fn time_two_arms(g: Graph, xs: &[f32], ws: &[f32], node: &str, route: &str) {
    let extent = conv_extent(&g);
    let label = &format!("{node} at extent {extent}");
    let want = common::unit_conv_cpu_reference(g.clone(), xs, ws);
    rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "1");
    let mut wide = WgpuExecutable::compile(g.clone());
    wide.set_param("w", ws);
    rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "0");
    let mut coop = WgpuExecutable::compile(g);
    coop.set_param("w", ws);
    let steps = coop.test_step_names();
    let arena = coop.test_arena_bytes();
    assert!(
        steps.contains(&route),
        "{label}: must reach {route} (schedule {steps:?})"
    );
    assert_eq!(
        arena,
        wide.test_arena_bytes(),
        "{label}: the plan must be the same on both arms — the cooperative kernel reads \
         both operands out of the f32 arena, so it adds no buffer of its own"
    );
    let before = rlx_wgpu::coop_f32_dispatch_count();
    let got_coop = coop.run(&[("x", xs)]).remove(0);
    let dispatches = rlx_wgpu::coop_f32_dispatch_count() - before;
    let got_wide = wide.run(&[("x", xs)]).remove(0);
    assert_eq!(
        dispatches,
        1,
        "{label}: a multiple-of-8 extent (extent % 8 = {}) must dispatch the \
         cooperative kernel once",
        extent % 8
    );
    let mut arms = vec![wide, coop];
    let per_round = common::time_round_robin(&mut arms, 3, 1, |arm| {
        let _ = arm.run(&[("x", xs)]);
    });
    common::assert_close_under(label, "cooperative", &got_coop, &want);
    common::assert_close_under(label, "wide", &got_wide, &want);
    let d = common::max_abs_diff(&got_coop, &got_wide);
    assert!(
        d < 2e-5,
        "{label}: cooperative vs wide max|Δ| {d:e} past the 2e-5 two conforming arms \
         can differ by"
    );
    eprintln!(
        "{label}: wide {:.3} ms {:?}, cooperative {:.3} ms {:?}, cooperative vs wide {d:e}, \
         vs CPU cooperative {:e} / wide {:e}, arena {arena} B",
        common::median(&per_round[0]),
        per_round[0],
        common::median(&per_round[1]),
        per_round[1],
        common::max_abs_diff(&got_coop, &want),
        common::max_abs_diff(&got_wide, &want),
    );
}
