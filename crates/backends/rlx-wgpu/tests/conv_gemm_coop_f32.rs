// RLX — versatile ML compiler & runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The convolution GEMM routes' multiplies take the f32 cooperative kernel
//! exactly where that kernel's own alignment holds.
//!
//! The routes are the im2col panel multiply and the 1×1 pointwise GEMM. Both put
//! the weight in the FIRST operand — the reverse of every other matmul call site —
//! which is what used to keep them off the cooperative path: it was reserved for
//! a multiply whose SECOND operand is a Param. The admission itself did not
//! change; what changed is that these two sites now ask it (`conv_gemm`) with the
//! roles swapped, so `RLX_WGPU_NO_COOP_F32`, `RLX_WGPU_FORCE_COOP_F32`,
//! `RLX_WGPU_MATMUL_F32_ONLY` and the per-device operand-order probe govern them
//! as they govern any other multiply.
//!
//! **Alignment decides, not the call.** Each case below is compiled and run twice
//! in one process, once with the cooperative path available and once under
//! `RLX_WGPU_NO_COOP_F32=1`. The step names carry no precision, so the witness is
//! the route's own per-dispatch counter
//! (`rlx_wgpu::coop_f32_dispatch_count()`), which both arms are read around:
//! an admitted shape must show exactly one cooperative dispatch with the arm on
//! and zero with it off, and a shape the alignment refuses must show zero either
//! way — the `coop` field is computed from the kernel's requirement (the
//! contraction extent a multiple of 8, the output-column extent a multiple of 32)
//! and never from what the run did.
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
//! The forced block's `n = 5000` run, whose column extent is a multiple of 8 only,
//! is `0.0` against the CPU conv as well.
//!
//! **The padded-`M` cases are the bounds reliance, exercised.** The Metal kernel
//! gathers 32 A rows per tile without checking `params.m`, so a `c_out = 80` shape
//! — not a multiple of 32 — makes it read 16 rows past the weight's extent: the
//! read `kernels/matmul_coop_f32.wgsl` states the reliance for, whose accumulators
//! must reach no output. Both routes carry such a shape, because their A windows
//! are not the same one — the im2col panel multiply binds the whole arena, the
//! pointwise GEMM the conv node's own window — so the read is checked against a
//! whole-arena bind and against a bounded one. The block at the end is the mirror
//! case on the column axis: a forced admission whose `n` is a multiple of 8 only,
//! so the x grid rounds up the same way.
//!
//! The case set is the route's own geometries: `conv_routes_parked_weight.rs`
//! pins `[1,64,1,2048] k=[1,7]` (im2col), `[1,128,1,8192] k=[1,1]` (pointwise)
//! and the mirrored `[1,64,1,300]` lift. Each class has one member the alignment
//! refuses on its output-column extent (`2048 → 2050`, `8192 → 5000` — same route,
//! same admission floors, only that extent moved) and one it refuses on its
//! contraction extent (`448 → 420`, `128 → 100`), so both axes the admission
//! checks are covered; the rest are the axes the kernel's requirement is stated
//! in — a contraction that is a multiple of 8 only, one deepened past the routed
//! default, and an `M` that is not a multiple of 32.
//!
//! **The dispatch-tuning cache is pinned off** (`RLX_GPU_TUNING_CACHE`, a path
//! that does not exist) because these two sites now read the shared per-shape
//! tuning table as any other multiply does: a persisted override for one of these
//! shapes can pin it back to the wide kernel and silently cancel the move, which
//! would read as an alignment failure here when it is a tuning-cache fact.
//!
//! Its own test binary: the arm is a process-global switch and the counter is
//! process-global too.

use rlx_ir::Graph;
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
    /// Whether this shape's conv runs a GEMM the f32 cooperative kernel's
    /// alignment admits: `Some(true)` admitted, `Some(false)` refused, and `None`
    /// for a conv that reaches no GEMM route at all — the mirrored lift, whose
    /// place in the set is the witness that the arm switches neither it nor any
    /// other node outside the two classes.
    coop: Option<bool>,
}

/// Seven admitted shapes, four the arm may not switch, and one inert.
///
/// The two classes are the im2col panel multiply and the pointwise GEMM, and each
/// axis the admission checks gets a refused member per class: the output-column
/// extent (2050, 5000 — both multiple-of-32 failures, both still over their
/// route's own admission floors) and the contraction extent (420, 100 — both
/// multiple-of-8 failures whose output-column extent is aligned, so the
/// contraction axis alone is what refuses them). The last member is the mirrored
/// `conv2d_tiled` lift, which is not a GEMM route at all: its place in the set is
/// the witness that the arm reaches neither it nor any other node outside the two
/// classes.
///
/// The admitted set covers what the kernel's requirement is stated in: `k` a
/// multiple of 32 as well as one that is a multiple of 8 only (`336`, `40` — the
/// axis the shader's `REQUIREMENTS` names as `8`, not `32`), a contraction
/// deepened past the routed default, and an `M` that is not a multiple of 32 on
/// **both** routes — because their A windows are not the same one (the im2col
/// panel multiply binds the whole arena, the pointwise GEMM the conv node's own
/// window), so the padded-row read is checked against both a whole-arena and a
/// bounded bind.
///
/// The im2col shapes' contraction extent is `c_in × kh × kw` (448, 420, 336 and
/// 3584 for the deep one) and the pointwise ones' is `c_in` (128, 100, 40).
const CASES: [Case; 12] = [
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
        label: "pointwise-unaligned [1,128,1,5000] k=[1,1] p=[0,0] 128→64",
        c_in: 128,
        c_out: 64,
        l: 5000,
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
    )
}

/// Compile `case` under the switches in force, bind the weight, run `x`, and
/// report the step names the schedule took, the output, and how many f32
/// cooperative dispatches the run made. One conv in the graph, so the count is
/// this node's.
fn compile_and_run(case: &Case, xs: &[f32], ws: &[f32]) -> (Vec<&'static str>, Vec<f32>, u64) {
    let before = rlx_wgpu::coop_f32_dispatch_count();
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    let steps = exe.test_step_names();
    let got = exe.run(&[("x", xs)]).remove(0);
    (steps, got, rlx_wgpu::coop_f32_dispatch_count() - before)
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
        let (steps_coop, got_coop, coop_dispatches) = compile_and_run(case, &xs, &ws);
        rlx_ir::env::set("RLX_WGPU_NO_COOP_F32", "1");
        let (steps_base, got_base, base_dispatches) = compile_and_run(case, &xs, &ws);

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
                    coop_dispatches,
                    1,
                    "{}: the alignment admits this shape's GEMM (k={k_total} % 8 = {}, \
                     n={} % 32 = {}), so its one conv must dispatch the cooperative \
                     kernel once",
                    case.label,
                    k_total % 8,
                    case.l,
                    case.l % 32
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
    let (steps, got, dispatches) = compile_and_run(case, &xs, &ws);
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

    // The force lever keeps its meaning for these classes too, and what it moves
    // is the admission: on a Metal adapter it widens the alignment to the portable
    // form's `n % 8` — the same widening an ordinary multiply gets from it, the
    // Metal scope and the operand-order probe being the terms it does not touch.
    // Read on one shape with the lever on and off, so "the alignment and not the
    // call decides" holds under the lever as well. The unaligned `n` is the
    // kernel's own bounds-checked case — its grid rounds up and the store guard
    // drops the columns past `params.n` — which is why the output is held to the
    // route's CPU bar here rather than merely counted.
    let case = CASES
        .iter()
        .find(|c| c.label.starts_with("pointwise-unaligned "))
        .expect("the pointwise unaligned case");
    let xs = common::fill(case.c_in * case.l, 7, 1, 13, 0.1, -0.6);
    let ws = common::fill(case.c_out * case.c_in * case.k, 5, 3, 11, 0.1, -0.5);
    let want = common::unit_conv_cpu_reference(graph(case), &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "1");
    let (steps, got_forced, forced) = compile_and_run(case, &xs, &ws);
    rlx_ir::env::set("RLX_WGPU_FORCE_COOP_F32", "0");
    let (_, got_off, off) = compile_and_run(case, &xs, &ws);
    assert!(
        steps.contains(&case.route),
        "{}: the route is unchanged by the force lever (schedule {steps:?})",
        case.label
    );
    assert_eq!(
        forced, 1,
        "{}: RLX_WGPU_FORCE_COOP_F32=1 widens the alignment to n % 8 for these \
         classes as it does for any other multiply",
        case.label
    );
    assert_eq!(
        off, 0,
        "{}: with the lever off the same shape keeps today's kernel, so it is the \
         alignment the lever widened and not the call that decided",
        case.label
    );
    common::assert_close_under(case.label, "forced", &got_forced, &want);
    common::assert_close_under(case.label, "unforced", &got_off, &want);
    eprintln!(
        "{}: RLX_WGPU_FORCE_COOP_F32=1 — cooperative dispatch on an n % 32 ≠ 0 shape \
         (n={}), vs CPU {:.2e}",
        case.label,
        got_forced.len(),
        common::max_abs_diff(&got_forced, &want)
    );
}
