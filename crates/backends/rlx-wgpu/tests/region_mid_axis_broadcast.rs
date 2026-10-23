// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The region encoding's broadcast admission rule, pinned on wgpu.
//!
//! The rule and the mis-index it prevents are
//! `rlx_ir::region_encode::region_input_modulus`'s doc, in one place. This file pins
//! its consequences on a device: a mid-axis singleton (`[1, C, 1, 1]` against
//! `[1, C, 1, L]`, whose divisor would be `gid % C` instead of `(gid / L) % C`) keeps
//! its chain as ordinary `Op::Binary`/`Op::Activation` nodes, and a genuine trailing
//! broadcast (`[1, 1, L]` into `[1, C, L]`, index `gid % L`) still fuses.
//!
//! The oracle is the CPU device with `unfuse_elementwise_regions = true`: an
//! independent implementation on the unfused path, not the region kernel compared
//! against itself. The device arm sets `keep_elementwise_regions` programmatically —
//! wgpu's default contour unfuses regions, which would make the fused arm a second
//! copy of the oracle — and refuses to run under the environment switches
//! `FusionOptions::merge_env` would merge on top of it.
//!
//! Tolerance: 1e-6 relative, read as `1e-6 * (1 + |want|)` so a ReLU-zero reference
//! is still compared rather than skipped. `Add` and `Relu` are exact operations, so
//! the bar only admits the device's own f32 form; the failure this file pins is
//! off by O(1), never by an ulp.
//!
//! The batch half is the other side of the same rule: a `BatchElementwiseRegion`
//! binds each of its inputs as the chain's slot-0 operand for that input's own run of
//! the output's batch axis, so it is judged against its own slice of the output. The
//! guard first judged it against `batch_region_slice_shape` (dim 0 forced to 1), which
//! refused every merged slice of two or more rows.
//!
//! Which checks are red on the pre-change tree: the mid-axis test and the
//! supplied-region test fail there (the divider admitted the operand, the region
//! formed, and its kernel walked the channel with time), the batch test fails
//! there (a two-row slice was refused). The trailing-broadcast test is the
//! non-vacuity control and passes on both trees.
//!
//! The rule's equal-element-count arm — an operand holding the output's element count,
//! read through `gid` because both are contiguous row-major — has no check here: a
//! region's shape is *inferred* from its chain, so no marking pass can present an
//! operand whose element order differs from the output's (the arm exists so the rule
//! is total, and the rule's own unit test pins both admitted layouts).

mod common;

use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep};
use rlx_ir::{DType, Graph, Op, RegionPrologue, Shape};
use rlx_opt::pass::Pass as _;
use rlx_opt::{FusionOptions, MarkElementwiseRegions};
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

const CHANNELS: usize = 64;
const LENGTH: usize = 32;

fn output_shape() -> Shape {
    Shape::new(&[1, CHANNELS, 1, LENGTH], DType::F32)
}

fn channel_bias_shape() -> Shape {
    Shape::new(&[1, CHANNELS, 1, 1], DType::F32)
}

/// `x[1,C,1,L] + bias[1,C,1,1]` then `relu`: two steps, long enough for the marking
/// pass to want the chain, and a bias whose broadcast keeps a mid-axis extent.
fn mid_axis_graph() -> Graph {
    let mut g = Graph::new("region_mid_axis");
    let x = g.input("x", output_shape());
    let bias = g.input("bias", channel_bias_shape());
    let sum = g.binary(BinaryOp::Add, x, bias, output_shape());
    let out = g.activation(Activation::Relu, sum, output_shape());
    g.set_outputs(vec![out]);
    g
}

fn trailing_output_shape() -> Shape {
    Shape::new(&[1, CHANNELS, LENGTH], DType::F32)
}

fn trailing_bias_shape() -> Shape {
    Shape::new(&[1, 1, LENGTH], DType::F32)
}

/// The same chain with a genuine trailing broadcast: `[1,1,L]` into `[1,C,L]`, whose
/// index under one divisor is `gid % L` and is correct.
fn trailing_graph() -> Graph {
    let mut g = Graph::new("region_trailing");
    let x = g.input("x", trailing_output_shape());
    let bias = g.input("bias", trailing_bias_shape());
    let sum = g.binary(BinaryOp::Add, x, bias, trailing_output_shape());
    let out = g.activation(Activation::Relu, sum, trailing_output_shape());
    g.set_outputs(vec![out]);
    g
}

/// A region carrying the mid-axis divisor by hand: the shape `MarkElementwiseRegions`
/// must never produce, and the one the direct entry (`WgpuExecutable::compile`, which
/// runs no legalization) has to refuse rather than run.
fn supplied_mid_axis_region() -> Graph {
    let mut modulus = [0u32; 16];
    modulus[1] = CHANNELS as u32;
    let mut g = Graph::new("region_mid_axis_supplied");
    let x = g.input("x", output_shape());
    let bias = g.input("bias", channel_bias_shape());
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![
                ChainStep::Binary(
                    BinaryOp::Add,
                    ChainOperand::Input(0),
                    ChainOperand::Input(1),
                ),
                ChainStep::Activation(Activation::Relu, ChainOperand::Step(0)),
            ],
            num_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: modulus,
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![x, bias],
        output_shape(),
    );
    g.set_outputs(vec![out]);
    g
}

/// Deterministic activation values, non-negative at the scale the channel bias moves
/// them, so a wrong bias element changes the result rather than being ReLU'd away.
fn x_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.017).sin() * 1.5).collect()
}

/// One value per channel, all distinct: `gid % C` and `(gid / L) % C` disagree
/// wherever the two channel indices differ.
fn channel_bias_values() -> Vec<f32> {
    (0..CHANNELS).map(|c| 0.25 + 0.031 * c as f32).collect()
}

fn trailing_bias_values() -> Vec<f32> {
    (0..LENGTH).map(|l| 0.2 + 0.01 * l as f32).collect()
}

fn run(device: Device, g: Graph, opts: &CompileOptions, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(device)
        .compile_with(g, opts)
        .run(inputs)
        .into_iter()
        .next()
        .expect("graph produced no output")
}

/// The unfused oracle: the same graph on CPU, with the marking pass's output
/// explicitly dissolved so the environment's keep switch cannot keep a region there.
fn unfused_opts() -> CompileOptions {
    CompileOptions::new().fusion_opts(FusionOptions {
        unfuse_elementwise_regions: true,
        ..Default::default()
    })
}

/// The device arm's options: keep the marked region through lowering, set here rather
/// than through `RLX_KEEP_ELEMENTWISE_REGIONS`, which is process-global and would leak
/// into every other check in this binary.
fn fused_opts() -> CompileOptions {
    CompileOptions::new().fusion_opts(FusionOptions {
        keep_elementwise_regions: true,
        ..Default::default()
    })
}

/// [`FusionOptions::merge_env`] merges the process environment on top of these
/// options, so a developer with one of these exported would measure a different graph
/// than the one the check names — `RLX_METAL_UNFUSE_REGIONS` in particular would leave
/// the mid-axis check asserting over an unfused graph it never formed.
fn refuse_under_fusion_env() {
    for key in [
        "RLX_METAL_UNFUSE_REGIONS",
        "RLX_KEEP_ELEMENTWISE_REGIONS",
        "RLX_DECOMPOSE_FUSION_REGIONS",
        "RLX_METAL_NO_FUSION",
        "RLX_NO_FK_FUSION",
    ] {
        assert!(
            rlx_ir::env::is_unset(key),
            "{key} is set and these checks set their fusion options programmatically: \
             `FusionOptions::merge_env` merges the process value on top, so the graph \
             measured would not be the one this file names — unset it to run them"
        );
    }
}

fn regions_in(g: &Graph) -> usize {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        .count()
}

fn marked(g: &Graph) -> Graph {
    MarkElementwiseRegions.run(g.clone())
}

fn assert_matches(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: output length");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: non-finite device output"
    );
    let outside: Vec<usize> = (0..got.len())
        .filter(|&i| (got[i] - want[i]).abs() > 1e-6 * (1.0 + want[i].abs()))
        .collect();
    let worst = (0..got.len())
        .map(|i| (got[i] - want[i]).abs())
        .fold(0.0f32, f32::max);
    let i = outside.first().copied().unwrap_or(0);
    assert!(
        outside.is_empty(),
        "{label}: {}/{} element(s) outside 1e-6 relative, max|Δ| = {worst}; first is \
         element {i} (got {}, want {})",
        outside.len(),
        got.len(),
        got[i],
        want[i]
    );
}

#[test]
fn a_mid_axis_broadcast_is_left_unfused_and_matches_the_unfused_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let g = mid_axis_graph();
    let m = marked(&g);
    assert_eq!(
        regions_in(&m),
        0,
        "`[1,C,1,1]` against `[1,C,1,L]` is a mid-axis broadcast: the single divisor the \
         region encoding stores is `gid % C`, not the `(gid / L) % C` the unfused path \
         computes. The chain must stay ordinary Binary/Activation nodes"
    );
    assert!(
        !common::legalised_step_names(&m).contains(&"elementwise_region"),
        "the marking pass left a region in the lowered schedule: {:?}",
        common::legalised_step_names(&m)
    );

    let xs = x_values(CHANNELS * LENGTH);
    let bs = channel_bias_values();
    let inputs: &[(&str, &[f32])] = &[("x", &xs), ("bias", &bs)];
    let got = run(Device::Gpu, g.clone(), &fused_opts(), inputs);
    let want = run(Device::Cpu, g, &unfused_opts(), inputs);
    assert_matches(&got, &want, "mid-axis broadcast");
}

#[test]
fn a_trailing_broadcast_is_still_fused_and_matches_the_unfused_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let g = trailing_graph();
    let m = marked(&g);
    assert_eq!(
        regions_in(&m),
        1,
        "`[1,1,L]` against `[1,C,L]` is a trailing broadcast and its index is `gid % L`: \
         the chain must still be marked, or the mid-axis decline would be vacuous"
    );
    assert!(
        common::legalised_step_names(&m).contains(&"elementwise_region"),
        "the marked region did not reach the device schedule: {:?}",
        common::legalised_step_names(&m)
    );

    let xs = x_values(CHANNELS * LENGTH);
    let bs = trailing_bias_values();
    let inputs: &[(&str, &[f32])] = &[("x", &xs), ("bias", &bs)];
    let got = run(Device::Gpu, g.clone(), &fused_opts(), inputs);
    let want = run(Device::Cpu, g, &unfused_opts(), inputs);
    assert_matches(&got, &want, "trailing broadcast");
}

#[test]
fn a_supplied_mid_axis_region_is_refused_rather_than_computed() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    // The direct entry runs no `LegalizeBroadcast` and no verifier, so the refusal
    // that matters here is the lowering's own.
    let raised = std::panic::catch_unwind(move || {
        WgpuExecutable::compile(supplied_mid_axis_region());
    })
    .expect_err("a region whose divisor cannot express its operand must fail the compile");
    let message = raised
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| raised.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("operand 1") && message.contains("mid-axis"),
        "refusal must name the operand and the reason; got: {message}"
    );
}

/// Two rows per slice: `narrow(0..2) → relu`, `narrow(2..4) → relu`, `concat`, which
/// `FuseBatchPreprocess` merges into one `BatchElementwiseRegion` over a four-row
/// output. NCHW, the shape the models' preprocess chains carry.
fn two_row_slice_graph() -> Graph {
    let shape = |n: usize| Shape::new(&[n, CHANNELS, 8, 8], DType::F32);
    let mut g = Graph::new("region_batch_two_row_slices");
    let batch = g.input("batch", shape(4));
    let mut slices = Vec::new();
    for start in [0usize, 2] {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start,
                len: 2,
            },
            vec![batch],
            shape(2),
        );
        slices.push(g.add_node(Op::Activation(Activation::Relu), vec![sl], shape(2)));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, shape(4));
    g.set_outputs(vec![out]);
    g
}

/// A merged slice of two rows is not a mid-axis broadcast: each batch input is the
/// chain's slot-0 operand for its own run of the output's batch axis, so the guard
/// must judge it against `[2, C, L]` and not against `[1, C, L]`. Judged against the
/// latter it looks mid-axis, and the pre-change guard refused it — through the
/// verifier and through `WgpuExecutable::compile` — on a graph whose unfused form is
/// correct.
#[test]
fn a_batch_slice_of_two_rows_verifies_lowers_and_matches_the_unfused_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_fusion_env();

    let primitive = two_row_slice_graph();
    use rlx_opt::rlx_fusion::{FuseBatchPreprocess, MarkBatchSliceRegions};
    let g = FuseBatchPreprocess.run(MarkBatchSliceRegions.run(primitive.clone()));
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "the pass must merge the two slices, or this check is about nothing"
    );
    assert!(
        rlx_ir::verify::verify(&g).is_empty(),
        "a batch region over two-row slices must verify clean: {:?}",
        rlx_ir::verify::verify(&g)
    );

    let xs: Vec<f32> = x_values(4 * CHANNELS * 8 * 8);
    let inputs: &[(&str, &[f32])] = &[("batch", &xs)];
    // The direct entry is the one that runs no verifier and no `LegalizeBroadcast`:
    // the lowering's own guard has to admit the region.
    let got = WgpuExecutable::compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .expect("the batch region produced no output");
    let want = run(Device::Cpu, primitive, &unfused_opts(), inputs);
    assert_matches(&got, &want, "batch slice of two rows");
}
