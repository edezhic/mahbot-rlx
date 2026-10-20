// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acceptance checks for the wgpu region kernel's activation coverage, run on the device
//! whose kernel it is.
//!
//! Each check builds the region-forming graph BY HAND — no measured graph in this tree
//! carries a region — and compares it against THE SAME GRAPH WITH THE REGION GONE, ON THE
//! SAME DEVICE: the unfused device path is the oracle, so the device's own kernels are on
//! both sides of the comparison. What makes a comparison falsifiable is the PRE-CHANGE
//! run: there the region is present and the region kernel hands back the activation's own
//! input (see `rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS` for that
//! fallback), so the two sides differ by the activation itself.
//!
//! Which switch dissolves the region, shape by shape — naming it is also what says the
//! oracle graph is the same computation rather than a different one:
//!
//! * the prologue shape (a resize feeding an affected activation) —
//!   `FusionOptions::fuse_region_prologue = false`.
//! * the batch shape (slices that narrow, activate and concatenate) —
//!   `FusionOptions::fuse_batch_preprocess = false`.
//! * the plain chain — `FusionOptions::keep_elementwise_regions = false`, which is the
//!   default: the plain region is never kept in the first place.
//! * the supplied-region shape (a hand-built region compiled through
//!   [`WgpuExecutable::compile`], the direct entry point that bypasses the compiler's
//!   pipeline) — NO switch dissolves it; the primitives are built by hand. That is also
//!   the shape that reaches the expansion decision: nothing formed the region, so nothing
//!   can decline to form it.
//!
//! The device is `Device::Gpu` — the label that resolves to the wgpu backend, whose
//! region kernel this is. `Device::Vulkan` is a different backend (`rlx-vulkan`,
//! registered separately) that decomposes every region, so a check run on it would pass
//! before this change and prove nothing.
//!
//! Tolerance policy, per activation: `Recip` is a division — the device's f32 is specified
//! to a few units in the last place, never bit-exact — so it is compared under a relative
//! bound. `Round` is half-to-even on this device while the tree's references disagree on
//! the halves, so the inputs are kept away from exact half-integers (asserted over the
//! generated slice, not assumed of the table) and it is compared exactly: a bound there
//! would only ever hide the identity these checks exist to catch.
//!
//! Against the pre-change tree every check that carries a region fails, for one reason:
//! the region kernel hands that activation back as its own input. The prologue-operand
//! check is the exception and has to be read as such — before this change such a region
//! reached the kernel, which honours the operand field, so it passes there; it pins the
//! expansion's own correctness, and the exact behaviour it falsifies is the input-0
//! expansion its own doc names.
//!
//! The batch shape is the one check two layers can satisfy, since both the marking pass
//! declining the code and the lowering hook expanding the region end in the same primitives;
//! the batch formation site itself is pinned in-crate, in `rlx-fusion`'s
//! `fk_fusion::tests`.

use rlx_ir::op::{Activation, ChainOperand, ChainStep};
use rlx_ir::{DType, Graph, GraphExt, Op, RegionPrologue, Shape};
use rlx_opt::FusionOptions;
use rlx_opt::rlx_fusion::fk_graphs;
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// Refuse to run under any of the env switches the fusion options these checks set
/// programmatically are keyed to.
///
/// [`FusionOptions::merge_env`] merges the process environment ON TOP of whatever the
/// caller set, so a developer with one of these exported would be measuring a
/// different graph than the one the check names — several of them switch off the very
/// fusion under test, and the check would then pass by comparing two unfused paths.
/// Nothing here sets an override and nothing takes a lock: refusing is the point, and
/// setting the options programmatically is the alternative this tree's convention
/// allows.
fn refuse_under_fusion_env() {
    for key in [
        "RLX_METAL_NO_FUSION",
        "RLX_METAL_UNFUSE_REGIONS",
        "RLX_KEEP_ELEMENTWISE_REGIONS",
        "RLX_DECOMPOSE_FUSION_REGIONS",
        "RLX_NO_FK_FUSION",
        "RLX_FUSE_REGION_PROLOGUE",
        "RLX_FUSE_BATCH_PREPROCESS",
        "RLX_NATIVE_FK_REGIONS",
        "RLX_NO_NATIVE_FK_REGIONS",
    ] {
        assert!(
            rlx_ir::env::is_unset(key),
            "{key} is set and these checks set their fusion options programmatically: \
             `FusionOptions::merge_env` merges the process value on top, so the graph \
             measured would not be the one this file names — unset it to run them"
        );
    }
}

fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

/// Compile `g` through the runtime's device path with `opts` and return its first
/// output.
fn run_session(g: Graph, opts: &CompileOptions, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(Device::Gpu)
        .compile_with(g, opts)
        .run(inputs)
        .into_iter()
        .next()
        .expect("run_session: graph produced no output")
}

/// Compile `g` through the backend's direct entry point and return its first output.
///
/// This is the path that bypasses the compiler's fusion pipeline entirely: whatever
/// region the graph already carries is what lowering sees.
fn run_exec(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    WgpuExecutable::compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .expect("run_exec: graph produced no output")
}

/// Compare a check's output against its oracle's, under the per-activation tolerance
/// policy the module doc states.
///
/// The failure message carries the aggregate `max|Δ|` together with the worst
/// element's two values, because the pre-change value is the whole point of the
/// check: it is the activation's own input.
fn assert_matches(got: &[f32], want: &[f32], act: Activation, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: output length");
    // Before any tolerance comparison: `(NaN - want).abs() > bound` is false and the
    // `f32::max` fold discards it, so a kernel writing NaN would pass below. Counted
    // separately, as the numeric parity suites do.
    assert_finite(got, label);
    let bound = |w: f32| match act {
        // Round's inputs are tie-free and its answer is then exact, so the bound is
        // zero — anything looser would let the identity pass.
        Activation::Round => 0.0,
        // Relative, because a division's f32 result is specified only to a few ulp.
        _ => 1e-5 * (1.0 + w.abs()),
    };
    let outside: Vec<usize> = (0..got.len())
        .filter(|&i| (got[i] - want[i]).abs() > bound(want[i]))
        .collect();
    let worst = (0..got.len())
        .map(|i| (got[i] - want[i]).abs())
        .fold(0.0f32, f32::max);
    let i = outside.first().copied().unwrap_or(0);
    assert!(
        outside.is_empty(),
        "{label}: {}/{} element(s) outside tolerance, max|Δ| = {worst}; first of them is \
         element {i} (got {}, want {})",
        outside.len(),
        got.len(),
        got[i],
        want[i]
    );
}

/// The device-side finiteness rule a tolerance comparison does not carry.
fn assert_finite(got: &[f32], label: &str) {
    let nan = got.iter().filter(|v| v.is_nan()).count();
    let pos_inf = got.iter().filter(|v| **v == f32::INFINITY).count();
    let neg_inf = got.iter().filter(|v| **v == f32::NEG_INFINITY).count();
    assert_eq!(
        (nan, pos_inf, neg_inf),
        (0, 0, 0),
        "{label}: non-finite device output"
    );
}

/// Deterministic inputs for the rounding checks.
///
/// `base` is an exact integer, so each value's fractional part is the table's own
/// three-decimal entry and none of them is anywhere near a half.
fn round_inputs(n: usize) -> Vec<f32> {
    const FRAC: [f32; 8] = [0.123, 0.251, 0.374, 0.626, 0.749, 0.877, 0.001, 0.999];
    (0..n).map(|i| (i / 8) as f32 - 2.0 + FRAC[i % 8]).collect()
}

/// The premise the `Round` checks rest on, checked over the slice that is actually
/// fed to them: no value is within `1e-3` of a half-integer, so nothing they compare
/// is a tie, and a reference that resolved ties the other way would still agree.
fn assert_off_half_integers(xs: &[f32]) {
    for v in xs {
        let frac = v.fract().abs();
        assert!(
            (frac - 0.5).abs() > 1e-3,
            "{v} is within 1e-3 of a half-integer — the rounding checks must stay off \
             ties, which the tree's references resolve differently"
        );
    }
}

/// Deterministic inputs for the reciprocal checks, every one inside `[0.5, 4.0]` so
/// that the relative bound they are compared under means something: a reciprocal has
/// no relative accuracy to speak of near zero.
fn recip_inputs(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + (i % 8) as f32 * 0.43).collect()
}

/// Primitive `resize → Round`, the chain the prologue formation closes into one region
/// when it is left on — `fk_graphs`' fixture at this file's shape.
fn resize_round_primitives() -> Graph {
    fk_graphs::resize_activation_graph("region_round_prologue", Activation::Round, 1, 3, 8, 8)
}

/// Primitive `narrow → Round → concat` over two slices — `fk_graphs`' fixture at this
/// file's shape.
fn batch_narrow_round_primitives() -> Graph {
    fk_graphs::batch_narrow_activation_primitive_graph(
        "region_round_batch",
        Activation::Round,
        2,
        3,
        8,
        8,
    )
}

/// Primitive `exp → Recip`: the shortest chain the marking pass fuses.
fn exp_then_recip_primitives() -> Graph {
    let mut g = Graph::new("region_recip_plain");
    let x = g.input("x", Shape::new(&[64], DType::F32));
    let e = g.exp(x);
    let out = g.activation(Activation::Recip, e, Shape::new(&[64], DType::F32));
    g.set_outputs(vec![out]);
    g
}

/// Primitive `Recip` over one `[64]` input — one step, which the marking pass does
/// not fuse, so it stays the step the supplied region stands in for.
fn recip_primitives() -> Graph {
    let mut g = Graph::new("region_recip_primitives");
    let x = g.input("x", Shape::new(&[64], DType::F32));
    let out = g.activation(Activation::Recip, x, Shape::new(&[64], DType::F32));
    g.set_outputs(vec![out]);
    g
}

/// A region over one input, in the shape every formation site on this device mints
/// — built here rather than left to a pass, so nothing can decline to form it.
fn supplied_region(act: Activation, prologue: RegionPrologue, out_shape: Shape) -> Graph {
    let mut g = Graph::new("region_supplied");
    let in_shape = match prologue {
        RegionPrologue::None => out_shape.clone(),
        RegionPrologue::ResizeNearest2x => nchw(1, 3, 8, 8),
    };
    let x = g.input("x", in_shape);
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![ChainStep::Activation(act, ChainOperand::Input(0))],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue,
            prologue_input: 0,
        },
        vec![x],
        out_shape,
    );
    g.set_outputs(vec![out]);
    g
}

/// The prologue formation is on by default, so the defect run closes the resize and
/// the activation into one prologue region carrying the rounding code; the oracle
/// switches that formation off and leaves the two steps primitive.
#[test]
fn a_resize_feeding_the_rounding_code_matches_the_unfused_device_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let xs = round_inputs(3 * 8 * 8);
    assert_off_half_integers(&xs);
    let inputs: &[(&str, &[f32])] = &[("x", &xs)];

    let got = run_session(resize_round_primitives(), &CompileOptions::new(), inputs);
    let want = run_session(
        resize_round_primitives(),
        &CompileOptions::new().fusion_opts(FusionOptions {
            fuse_region_prologue: false,
            ..Default::default()
        }),
        inputs,
    );

    assert_matches(&got, &want, Activation::Round, "prologue round");
}

/// The batch preprocess fusion merges the per-slice regions into one
/// `BatchElementwiseRegion`, which the batch region kernel then runs with the same
/// `apply_activation`.
///
/// The oracle switches only that merge off. On this code the per-slice regions are not
/// marked in the first place — `rlx_fusion::region_kernel` declines the rounding code —
/// so after this change both sides run the primitive `narrow → Round → concat`, while on
/// the pre-change tree the default contour did form the batch region, and its kernel
/// then computed each slice as its own input.
#[test]
fn slicing_a_batch_on_the_rounding_code_matches_the_unfused_device_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let xs = round_inputs(2 * 3 * 8 * 8);
    assert_off_half_integers(&xs);
    let inputs: &[(&str, &[f32])] = &[("batch", &xs)];

    let got = run_session(
        batch_narrow_round_primitives(),
        &CompileOptions::new(),
        inputs,
    );
    let want = run_session(
        batch_narrow_round_primitives(),
        &CompileOptions::new().fusion_opts(FusionOptions {
            fuse_batch_preprocess: false,
            ..Default::default()
        }),
        inputs,
    );

    assert_matches(&got, &want, Activation::Round, "batch round");
}

/// The plain-region shape: `keep_elementwise_regions` is what keeps the marked chain
/// fused on this target, and the default contour unfuses it.
#[test]
fn keeping_plain_regions_on_the_reciprocal_code_matches_the_unfused_device_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let xs = recip_inputs(64);
    let inputs: &[(&str, &[f32])] = &[("x", &xs)];

    let got = run_session(
        exp_then_recip_primitives(),
        &CompileOptions::new().fusion_opts(FusionOptions {
            keep_elementwise_regions: true,
            ..Default::default()
        }),
        inputs,
    );
    let want = run_session(exp_then_recip_primitives(), &CompileOptions::new(), inputs);

    assert_matches(&got, &want, Activation::Recip, "plain reciprocal");
}

/// The supplied-region shape, and the one that reaches the expansion decision: the
/// region is built here and compiled through the direct entry point, so there is no
/// dissolving switch to name — the primitives are built by hand. Pre-change the region
/// kernel returns the region's own input where `Recip` is wanted; the expansion has to
/// put the region back on the single primitive it stands for.
#[test]
fn a_supplied_plain_region_on_the_reciprocal_code_is_expanded() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let xs = recip_inputs(64);
    let inputs: &[(&str, &[f32])] = &[("x", &xs)];

    let got = run_exec(
        supplied_region(
            Activation::Recip,
            RegionPrologue::None,
            Shape::new(&[64], DType::F32),
        ),
        inputs,
    );
    let want = run_exec(recip_primitives(), inputs);

    assert_matches(&got, &want, Activation::Recip, "supplied reciprocal");
}

/// The same, with the resize prologue: a prologue region is the one kind the default
/// contour deliberately keeps ON the kernel, so the expansion has to decompose this
/// one too — while leaving prologue regions whose codes the kernel honours on the
/// kernel, which is what keeps this from being a blanket unfuse.
#[test]
fn a_supplied_prologue_region_on_the_rounding_code_is_expanded() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let xs = round_inputs(3 * 8 * 8);
    assert_off_half_integers(&xs);
    let inputs: &[(&str, &[f32])] = &[("x", &xs)];

    let got = run_exec(
        supplied_region(
            Activation::Round,
            RegionPrologue::ResizeNearest2x,
            nchw(1, 3, 16, 16),
        ),
        inputs,
    );
    let want = run_exec(resize_round_primitives(), inputs);

    assert_matches(&got, &want, Activation::Round, "supplied prologue round");
}

/// The prologue field reaches the expansion too: with `prologue_input = 1` the region's
/// resize reads its *second* input, so a supplied region of that shape stands for
/// `resize(b) → Round`. Expanding it on the first input instead computes the chain from
/// `b` as it stands — the resize lands on `a` and the chained operand still reads the raw
/// `b` — so the region yields `b`'s own 192 elements where its output is 768, a wrong
/// shape and a wrong value at once.
///
/// The oracle builds the primitives by hand (no switch dissolves a supplied region), and
/// `a` is `b` shifted by 1.0, so a mis-placed resize cannot coincide with the right one.
#[test]
fn a_supplied_prologue_region_resizes_the_input_its_prologue_input_names() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let b: Vec<f32> = round_inputs(3 * 8 * 8);
    let a: Vec<f32> = b.iter().map(|v| v + 1.0).collect();
    assert_off_half_integers(&a);
    assert_off_half_integers(&b);
    let inputs: &[(&str, &[f32])] = &[("a", &a), ("b", &b)];

    let mut g = Graph::new("region_prologue_operand");
    let a_in = g.input("a", nchw(1, 3, 8, 8));
    let b_in = g.input("b", nchw(1, 3, 8, 8));
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![ChainStep::Activation(
                Activation::Round,
                ChainOperand::Input(1),
            )],
            num_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::ResizeNearest2x,
            prologue_input: 1,
        },
        vec![a_in, b_in],
        nchw(1, 3, 16, 16),
    );
    g.set_outputs(vec![out]);

    let mut oracle = Graph::new("region_prologue_operand_primitives");
    let _ = oracle.input("a", nchw(1, 3, 8, 8));
    let oracle_b = oracle.input("b", nchw(1, 3, 8, 8));
    let up = oracle.add_node(Op::ResizeNearest2x, vec![oracle_b], nchw(1, 3, 16, 16));
    let oracle_out = oracle.activation(Activation::Round, up, nchw(1, 3, 16, 16));
    oracle.set_outputs(vec![oracle_out]);

    let got = run_exec(g, inputs);
    let want = run_exec(oracle, inputs);

    assert_matches(&got, &want, Activation::Round, "supplied prologue operand");
}

/// The one region kind the expansion machinery declines, so the only one the lowering's
/// refusal can see: a `BatchElementwiseRegion` whose input count contradicts its own
/// `num_batch_inputs`, which `DecomposeFusionRegions` copies through untouched.
///
/// Fed to the direct entry point, because `Session::compile`'s verifier rejects such a
/// node in a debug build and the device path here is release. Refusing loudly is the
/// point: the alternative is an activation the kernel has no arm for, computed as its
/// own input.
#[test]
fn a_region_the_expansion_cannot_reach_is_refused_rather_than_computed() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let mut g = Graph::new("region_unreachable");
    let x = g.input("x", nchw(1, 3, 8, 8));
    let out = g.add_node(
        Op::BatchElementwiseRegion {
            chain: vec![ChainStep::Activation(
                Activation::Recip,
                ChainOperand::Input(0),
            )],
            // Two slices claimed, one input given.
            num_batch_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![x],
        nchw(2, 3, 8, 8),
    );
    g.set_outputs(vec![out]);

    let raised = std::panic::catch_unwind(move || {
        WgpuExecutable::compile(g);
    })
    .expect_err("a region the expansion cannot reach must fail the compile");
    let message = raised
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| raised.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("Recip") && message.contains("no arm for that code"),
        "the refusal must name the activation and the kernel's missing arm; got: {message}"
    );
    assert!(
        message.contains("no switch on this backend"),
        "the refusal must not offer a remedy this backend does not honour; got: {message}"
    );
}
