// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The sound half's DAC snake, `y = x + beta·sin²(alpha·x)`, folded into one
//! `snake_beta` dispatch.
//!
//! The chain is written as five nodes — `arg = alpha·x`, `s = sin(arg)`,
//! `s2 = s·s`, `scale = beta·s2`, `out = x + scale` — and the elementwise REGION
//! pass declines it, because `alpha`/`beta` are `[1, C, 1, 1]` against a
//! `[1, C, 1, L]` output and a flat `i % C` modulus would walk the channel with
//! the time axis (`region_input_modulus`'s mid-axis singleton). What this file
//! pins, in the order it matters:
//!
//! **The step exists and replaces all five.** `snake_beta` is counted once and
//! `binary`/`unary` not at all on the plain chain, and the five come back under
//! `RLX_WGPU_NO_SNAKE_BETA` — the switch is read at compile time, so one process
//! compiles both arms and the pin cannot be satisfied by a stale config snapshot.
//!
//! **The one dispatch computes the chain.** The fused output is held to the CPU
//! backend's own answer for the same graph at `1e-6` max-abs; the unfused device
//! arm is held to the same bar in the same run, so a fused arm that drifted and a
//! CPU arm that drifted with it cannot both be green. The data is a unit-range
//! ramp scaled into the ±2 band the model's activations sit in, and `alpha`/`beta`
//! cross zero, so the term is not inert on it.
//!
//! **The folds are per-shape.** A snake whose input is `Add(conv, bias)` folds
//! that Add too (no `binary` step is left, and the numbers still match), and a
//! snake whose `alpha` is full-shape is left as five steps — the shape the kernel
//! cannot serve, since its per-channel read is word `c` of a `[C]`-long region.
//! A bias `Add` that is itself a graph output keeps its step — the fold replaces
//! that write, and nothing else would make it — and so does one whose addend is
//! parked in the weight buffer, where the fold's range guard has no arena byte
//! space to compare in. A shape whose dispatch grid would not fit the adapter's
//! per-dimension cap is left as five steps rather than dispatched clipped.
//!
//! Every graph here is compiled through `WgpuExecutable::compile` and run on the
//! device; the oracle is `Session::new(Device::Cpu)` on the same graph, or the
//! chain's closed form where the graph holds one term everywhere. Nothing here
//! measures a time: the stand's cell is where that number lives. The
//! process-global env override is why the tests that set one, or bracket a route
//! counter, take [`route_lock`].
mod common;

use common::{max_abs_diff, ramp, route_lock};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// The switch the five-step arm is compiled under. Read live by the lowering, so
/// it can be flipped between two `compile`s in one process.
const NO_SNAKE_BETA: &str = "RLX_WGPU_NO_SNAKE_BETA";
/// The bar the fused step is held to against the CPU backend, over the ±2 signal
/// [`inputs`] builds. Both arms of this file's parity assert the same one.
const SNAKE_BAR: f32 = 1e-6;
/// The fixture shape: 8 channels of a 16-column `[1, C, 1, L]`.
const C: usize = 8;
const L: usize = 16;

/// The chain's five nodes over `x`, exactly as the sound half's `emit::snake_beta`
/// writes them: `a·x`, `sin`, `s·s`, `beta·s2`, `x + scale`.
fn snake_chain(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    alpha: rlx_ir::NodeId,
    beta: rlx_ir::NodeId,
    full: &Shape,
) -> rlx_ir::NodeId {
    let arg = g.add_node(Op::Binary(BinaryOp::Mul), vec![alpha, x], full.clone());
    let s = g.add_node(Op::Activation(Activation::Sin), vec![arg], full.clone());
    let s2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![s, s], full.clone());
    let scale = g.add_node(Op::Binary(BinaryOp::Mul), vec![beta, s2], full.clone());
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, scale], full.clone())
}

/// The snake exactly as the sound half emits it, over the graph input `x`.
/// `alpha`/`beta` are `[1, C, 1, 1]` unless `full_shape_alpha` asks for the
/// pattern the fold must decline, and `bias` prepends the `Add(conv, bias)` the
/// DAC blocks write before every snake.
fn snake_graph(alpha_full_shape: bool, bias: bool) -> Graph {
    let f = DType::F32;
    let full = Shape::new(&[1, C, 1, L], f);
    let per_channel = Shape::new(&[1, C, 1, 1], f);
    let mut g = Graph::new("snake_beta");
    let src = g.input("x", full.clone());
    let alpha = g.param(
        "alpha",
        if alpha_full_shape {
            full.clone()
        } else {
            per_channel.clone()
        },
    );
    let beta = g.param("beta", per_channel.clone());
    let x = if bias {
        let b = g.param("bias", per_channel);
        g.add(src, b)
    } else {
        src
    };
    let out = snake_chain(&mut g, x, alpha, beta, &full);
    g.set_outputs(vec![out]);
    g
}

/// The same chain with a parameter under the bias `Add` rather than the graph
/// input, so `RLX_WGPU_ALL_PARAMS_WEIGHT=1` can park the addend in the weight
/// buffer — the case the fold's range guard cannot compare a byte space for.
fn snake_graph_with_a_param_addend() -> Graph {
    let f = DType::F32;
    let full = Shape::new(&[1, C, 1, L], f);
    let per_channel = Shape::new(&[1, C, 1, 1], f);
    let mut g = Graph::new("snake_beta_param_addend");
    let conv = g.param("conv", full.clone());
    let alpha = g.param("alpha", per_channel.clone());
    let beta = g.param("beta", per_channel.clone());
    let b = g.param("bias", per_channel);
    let x = g.add(conv, b);
    let out = snake_chain(&mut g, x, alpha, beta, &full);
    g.set_outputs(vec![out]);
    g
}

/// The fixture's operands: `x` a ramp through ±2 (the band the sound half's
/// activations sit in), `alpha` crossing zero so the sine is not a constant, and
/// `beta` small but non-zero, so the scaled term carries the answer's shape.
fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = ramp(1, C * L, 0.7).iter().map(|v| v * 2.0).collect();
    let alpha = (0..C).map(|c| (c as f32 - 4.0) * 0.35).collect();
    let beta = (0..C).map(|c| 0.2 + 0.05 * c as f32).collect();
    let bias = (0..C).map(|c| 0.1 * (c as f32) - 0.35).collect();
    (x, alpha, beta, bias)
}

use std::collections::BTreeMap;

/// The same graph with all of its outputs, in graph order.
fn run_gpu_all(g: Graph, switch: Option<bool>) -> (BTreeMap<&'static str, usize>, Vec<Vec<f32>>) {
    if let Some(on) = switch {
        rlx_ir::env::set(NO_SNAKE_BETA, if on { "1" } else { "0" });
    }
    let has_bias = g.param_id("bias").is_some();
    let has_conv = g.param_id("conv").is_some();
    let (x, alpha, beta, bias) = inputs();
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("alpha", &alpha);
    exe.set_param("beta", &beta);
    if has_bias {
        exe.set_param("bias", &bias);
    }
    if has_conv {
        exe.set_param("conv", &x);
    }
    let kinds = exe.step_kind_counts();
    (kinds, exe.run(&[("x", &x)]))
}

/// The step kinds one arm compiles to, and its first output. `switch` installs
/// the five-step switch first when given; the lowering reads it live, so two
/// calls in one process can disagree, which is what the test below relies on.
fn run_gpu(g: Graph, switch: Option<bool>) -> (BTreeMap<&'static str, usize>, Vec<f32>) {
    let (kinds, mut out) = run_gpu_all(g, switch);
    (kinds, out.remove(0))
}

/// The same graph on the CPU backend, as this file's reference, all outputs.
fn run_cpu_all(g: Graph) -> Vec<Vec<f32>> {
    let has_bias = g.param_id("bias").is_some();
    let has_conv = g.param_id("conv").is_some();
    let (x, alpha, beta, bias) = inputs();
    let mut s = Session::new(Device::Cpu).compile(g);
    s.set_param("alpha", &alpha);
    s.set_param("beta", &beta);
    if has_bias {
        s.set_param("bias", &bias);
    }
    if has_conv {
        s.set_param("conv", &x);
    }
    s.run(&[("x", &x)])
}

/// The same graph on the CPU backend, as this file's reference.
fn run_cpu(g: Graph) -> Vec<f32> {
    run_cpu_all(g).remove(0)
}

/// The five-node chain compiles to one `snake_beta` step, and to the five it is
/// written as under the switch. Both arms are compiled in one process, so the
/// switch is proven to be read at compile time rather than frozen in a snapshot.
#[test]
fn the_chain_is_one_snake_beta_step_and_five_under_the_switch() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let (fused, _) = run_gpu(snake_graph(false, false), Some(false));
    assert_eq!(
        fused.get("snake_beta").copied().unwrap_or(0),
        1,
        "the snake must be one step; schedule = {fused:?}"
    );
    assert_eq!(
        (
            fused.get("binary").copied().unwrap_or(0),
            fused.get("unary").copied().unwrap_or(0)
        ),
        (0, 0),
        "no node of the chain may keep its own dispatch; schedule = {fused:?}"
    );

    let (unfused, _) = run_gpu(snake_graph(false, false), Some(true));
    assert_eq!(
        unfused.get("snake_beta").copied().unwrap_or(0),
        0,
        "`{NO_SNAKE_BETA}=1` must stand the fold down; schedule = {unfused:?}"
    );
    assert_eq!(
        (
            unfused.get("binary").copied().unwrap_or(0),
            unfused.get("unary").copied().unwrap_or(0)
        ),
        (4, 1),
        "the chain is 4 `Mul`/`Add` steps and one `Sin`; schedule = {unfused:?}"
    );

    // The fold consumes five elementwise nodes, so it stands down wherever those
    // would be hosted. `RLX_WGPU_FORCE_HOST` is the term of that rule this rig can
    // reach (the discrete-adapter and striped-arena ones are not); the host arm
    // still computes the chain, so the numbers are the CPU backend's.
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    let (hosted, hosted_out) = run_gpu(snake_graph(false, false), Some(false));
    rlx_ir::env::unset("RLX_WGPU_FORCE_HOST");
    assert_eq!(
        hosted.get("snake_beta").copied().unwrap_or(0),
        0,
        "`RLX_WGPU_FORCE_HOST=1` must keep the chain's own (hosted) steps; \
         schedule = {hosted:?}"
    );
    let want = run_cpu(snake_graph(false, false));
    let hosted_diff = max_abs_diff(&hosted_out, &want);
    assert!(
        hosted_diff <= SNAKE_BAR,
        "the hosted chain vs cpu: max abs diff {hosted_diff:e} > {SNAKE_BAR:e}"
    );

    rlx_ir::env::unset(NO_SNAKE_BETA);
}

/// The fused dispatch computes the chain: both device arms reach the CPU
/// backend's own answer inside [`SNAKE_BAR`], and so reach each other.
///
/// The CPU arm is the reference rather than a second device kernel, so a fold
/// that dropped or reordered a term cannot pass by agreeing with itself. The
/// pair of device arms is asserted too, because the fused arm's order is the
/// one thing the chain's spelling fixes: `beta·(s·s)` and not `(beta·s)·s`.
#[test]
fn the_fused_answer_is_the_chain_answer() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let (_, fused) = run_gpu(snake_graph(false, false), Some(false));
    let (_, unfused) = run_gpu(snake_graph(false, false), Some(true));
    let want = run_cpu(snake_graph(false, false));
    rlx_ir::env::unset(NO_SNAKE_BETA);

    assert_eq!(fused.len(), C * L, "one output over the whole tensor");
    let fused_diff = max_abs_diff(&fused, &want);
    let unfused_diff = max_abs_diff(&unfused, &want);
    eprintln!(
        "snake_beta [1,{C},1,{L}]: fused {fused_diff:.3e}, five-step {unfused_diff:.3e} \
         against the CPU backend"
    );
    assert!(
        fused_diff <= SNAKE_BAR,
        "fused snake vs cpu: max abs diff {fused_diff:e} > {SNAKE_BAR:e}"
    );
    assert!(
        unfused_diff <= SNAKE_BAR,
        "five-step snake vs cpu: max abs diff {unfused_diff:e} > {SNAKE_BAR:e}"
    );
    // Anti-vacuity: the scaled term has to move the answer. Without it the
    // output is `x`, so an inert `sin`/`beta` would land here and pass.
    let x_only_diff = max_abs_diff(&fused, &inputs().0);
    assert!(
        x_only_diff > SNAKE_BAR,
        "the fixture must show the snake's own term; `x` alone is \
         {x_only_diff:e} from the answer"
    );
}

/// The snake's input `Add(conv, bias)` folds into the step as well: the bias is
/// applied inside the fused dispatch, so no `binary` step is left and the numbers
/// are still the CPU backend's.
#[test]
fn a_bias_feeding_the_snake_is_folded_with_it() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let (kinds, got) = run_gpu(snake_graph(false, true), Some(false));
    // The fold reads `conv` at `out`'s position, one position past where the plan
    // ends its life, so `bias_fold_is_sound` admits the pair only when their
    // storage spans are disjoint. That term has no fixture here: the planner
    // decides slot reuse, so a hand-built graph cannot place two same-shape
    // tensors on overlapping bytes. The guard's other two terms — an exposed
    // output and a parked addend — are the tests below.
    let want = run_cpu(snake_graph(false, true));
    rlx_ir::env::unset(NO_SNAKE_BETA);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        1,
        "the snake is still one step; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        0,
        "the input `Add(conv, bias)` belongs to the fused step; schedule = {kinds:?}"
    );
    let diff = max_abs_diff(&got, &want);
    eprintln!("snake_beta with a folded bias: {diff:.3e} against the CPU backend");
    assert!(
        diff <= SNAKE_BAR,
        "folded-bias snake vs cpu: max abs diff {diff:e} > {SNAKE_BAR:e}"
    );
}

/// The same chain with the bias `Add` also exposed as a graph output. Its own
/// step is the only writer of that slot, so the fold has to decline.
fn snake_graph_exposing_the_add() -> Graph {
    let mut g = snake_graph(false, true);
    let out = g.outputs[0];
    let add = g.node(out).inputs[0];
    g.set_outputs(vec![out, add]);
    g
}

/// A bias `Add` that is also a graph output keeps its step: folding it would
/// suppress the only write to that slot, and the graph would return whatever the
/// planner left there. The snake itself still folds, and both outputs are the
/// CPU backend's.
#[test]
fn a_folded_add_that_is_an_output_keeps_its_step() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let (kinds, got) = run_gpu_all(snake_graph_exposing_the_add(), Some(false));
    let want = run_cpu_all(snake_graph_exposing_the_add());
    rlx_ir::env::unset(NO_SNAKE_BETA);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        1,
        "the snake itself is still one step; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        1,
        "the exposed bias `Add` keeps its own step (the four chain nodes are the \
         snake's); schedule = {kinds:?}"
    );
    assert_eq!(got.len(), 2, "both outputs come back");
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = max_abs_diff(a, b);
        assert!(
            d <= SNAKE_BAR,
            "output {i} of the exposed-`Add` graph: {d:e} from the CPU backend \
             (bar {SNAKE_BAR:e}) — an output whose only writer was folded away \
             returns stale bytes"
        );
    }
}

/// A parked addend takes the fold down: `conv` lives in the weight buffer, whose
/// tagged offsets are a different byte space from the arena's, so the range guard
/// cannot answer and refuses. The Add keeps its step, the snake still folds, and
/// the numbers are the CPU backend's.
#[test]
fn a_parked_addend_declines_the_bias_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    // The same graph without the parking, so the arm below is the parking's
    // effect and not a decline for some other reason.
    let (arena_kinds, _) = run_gpu(snake_graph_with_a_param_addend(), Some(false));
    assert_eq!(
        arena_kinds.get("binary").copied().unwrap_or(0),
        0,
        "an in-arena addend must fold the bias `Add`; schedule = {arena_kinds:?}"
    );

    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    let (kinds, got) = run_gpu(snake_graph_with_a_param_addend(), Some(false));
    rlx_ir::env::unset("RLX_WGPU_ALL_PARAMS_WEIGHT");
    let want = run_cpu(snake_graph_with_a_param_addend());

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        1,
        "the snake itself still folds; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        1,
        "a parked addend keeps the bias `Add`'s own step; schedule = {kinds:?}"
    );
    let diff = max_abs_diff(&got, &want);
    eprintln!("snake_beta with a parked addend: {diff:.3e} against the CPU backend");
    assert!(
        diff <= SNAKE_BAR,
        "parked addend vs cpu: max abs diff {diff:e} > {SNAKE_BAR:e}"
    );
}

/// The chain over `channels` channels of a single column, with no residual and no
/// bias — the shape used to push the dispatch grid past its cap.
fn one_column_chain(channels: usize) -> Graph {
    let f = DType::F32;
    let full = Shape::new(&[1, channels, 1, 1], f);
    let mut g = Graph::new("snake_beta_one_column");
    let src = g.input("x", full.clone());
    let alpha = g.param("alpha", Shape::new(&[1, channels, 1, 1], f));
    let beta = g.param("beta", Shape::new(&[1, channels, 1, 1], f));
    let out = snake_chain(&mut g, src, alpha, beta, &full);
    g.set_outputs(vec![out]);
    g
}

/// A channel count past the per-dimension dispatch cap declines the fold instead
/// of dispatching a clipped grid — the rule every other eligibility term uses, and
/// a `continue` in the detector rather than a panic in the lowering. `gid.y` is one
/// workgroup per channel, so 65 536 rows is one row past `MAX_WORKGROUPS_PER_DIM`.
#[test]
fn a_grid_past_the_dispatch_cap_keeps_the_five_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();
    const CHANNELS: usize = 65_536; // MAX_WORKGROUPS_PER_DIM + 1
    let alpha = 0.5f32;
    let beta = 0.3f32;
    let x = 0.25f32;

    let mut exe = WgpuExecutable::compile(one_column_chain(CHANNELS));
    exe.set_param("alpha", &vec![alpha; CHANNELS]);
    exe.set_param("beta", &vec![beta; CHANNELS]);
    let kinds = exe.step_kind_counts();
    let out = exe.run(&[("x", &vec![x; CHANNELS])]).remove(0);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        0,
        "a grid past the per-dimension cap must decline; schedule = {kinds:?}"
    );
    let want = x + beta * (alpha * x).sin().powi(2);
    let d = out.iter().map(|v| (v - want).abs()).fold(0.0f32, f32::max);
    assert!(
        out.len() == CHANNELS && d < 1e-6,
        "the declined chain still computes `x + beta·sin²(alpha·x)`: {d:e} from it \
         over {} elements",
        out.len()
    );
}

/// A full-shape `alpha` is not the pattern: the kernel reads `alpha[c]` as word
/// `c` of a `[C]`-long region, so a `[1, C, 1, L]` operand is refused and the
/// chain keeps its five steps — which still compute the answer.
#[test]
fn a_full_shape_alpha_keeps_the_five_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let (kinds, got) = run_gpu(snake_graph(true, false), Some(false));
    let want = run_cpu(snake_graph(true, false));
    rlx_ir::env::unset(NO_SNAKE_BETA);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        0,
        "a full-shape `alpha` must not take the fused step; schedule = {kinds:?}"
    );
    assert_eq!(
        (
            kinds.get("binary").copied().unwrap_or(0),
            kinds.get("unary").copied().unwrap_or(0)
        ),
        (4, 1),
        "the declined pattern keeps the chain; schedule = {kinds:?}"
    );
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= SNAKE_BAR,
        "declined snake vs cpu: max abs diff {diff:e} > {SNAKE_BAR:e}"
    );
}

/// A *computed* per-channel `alpha` — `[1, C, 1, 1]` bytes of its own, not a
/// parameter's — is not the pattern either. The fused step reads `alpha[c]` at
/// `out`'s position while the plan ends its liveness at `arg`, so anything but a
/// pinned parameter there is a read of a slot already reused by then.
#[test]
fn a_computed_alpha_keeps_the_five_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let f = DType::F32;
    let full = Shape::new(&[1, C, 1, L], f);
    let per_channel = Shape::new(&[1, C, 1, 1], f);
    let mut g = Graph::new("snake_beta_computed_alpha");
    let src = g.input("x", full.clone());
    let alpha = g.param("alpha", per_channel.clone());
    let beta = g.param("beta", per_channel);
    let computed = g.neg(alpha);
    let out = snake_chain(&mut g, src, computed, beta, &full);
    g.set_outputs(vec![out]);

    let (x, alpha, beta, _) = inputs();
    let mut exe = WgpuExecutable::compile(g);
    // `computed = -alpha`, so feeding `-alpha` keeps the fixture's own values.
    exe.set_param("alpha", &alpha.iter().map(|v| -v).collect::<Vec<_>>());
    exe.set_param("beta", &beta);
    let kinds = exe.step_kind_counts();
    let got = exe.run(&[("x", &x)]).remove(0);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        0,
        "a computed `alpha` must not take the fused step; schedule = {kinds:?}"
    );
    let want: Vec<f32> = (0..C * L)
        .map(|i| x[i] + beta[i / L] * (alpha[i / L] * x[i]).sin().powi(2))
        .collect();
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= SNAKE_BAR,
        "declined snake vs the closed form: max abs diff {diff:e} > {SNAKE_BAR:e}"
    );
}

/// Two snakes chained at `inner == 1`, where `[1, C, 1, 1]` is both the full and
/// the per-channel shape: the outer snake's input is the inner snake's own output
/// `Add`, which is a bias-fold candidate whose `conv` is the inner snake's `scale`.
/// Folding it would suppress the write that produces the value the outer chain
/// reads, so the outer fold declines its *bias* and keeps its chain: each snake is
/// one step and neither `Add` keeps a step of its own.
#[test]
fn a_chained_snake_keeps_the_add_its_fold_would_consume() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let graph = || {
        let f = DType::F32;
        let full = Shape::new(&[1, C, 1, 1], f);
        let mut g = Graph::new("snake_beta_chained");
        let x = g.param("x", full.clone());
        let a1 = g.param("alpha_a", full.clone());
        let b1 = g.param("beta_a", full.clone());
        let inner = snake_chain(&mut g, x, a1, b1, &full);
        let a2 = g.param("alpha_b", full.clone());
        let b2 = g.param("beta_b", full.clone());
        let outer = snake_chain(&mut g, inner, a2, b2, &full);
        g.set_outputs(vec![outer]);
        g
    };

    let x: Vec<f32> = ramp(1, C, 0.7).iter().map(|v| v * 2.0).collect();
    let alpha_a: Vec<f32> = (0..C).map(|c| (c as f32 - 4.0) * 0.35).collect();
    let beta_a: Vec<f32> = (0..C).map(|c| 0.2 + 0.05 * c as f32).collect();
    let alpha_b: Vec<f32> = (0..C).map(|c| 0.4 - 0.2 * c as f32).collect();
    let beta_b: Vec<f32> = (0..C).map(|c| 0.3 - 0.02 * c as f32).collect();

    let mut exe = WgpuExecutable::compile(graph());
    for (name, values) in [
        ("x", &x),
        ("alpha_a", &alpha_a),
        ("beta_a", &beta_a),
        ("alpha_b", &alpha_b),
        ("beta_b", &beta_b),
    ] {
        exe.set_param(name, values);
    }
    let kinds = exe.step_kind_counts();
    let got = exe.run(&[("x", &x)]).remove(0);

    assert_eq!(
        kinds.get("snake_beta").copied().unwrap_or(0),
        2,
        "each snake folds its own four chain nodes; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("binary").copied().unwrap_or(0),
        0,
        "neither `Add` keeps a step of its own — each is its snake's output; \
         schedule = {kinds:?}"
    );

    // v1 = x + beta_a·sin²(alpha_a·x), then the same over v1.
    let v1: Vec<f32> = (0..C)
        .map(|c| x[c] + beta_a[c] * (alpha_a[c] * x[c]).sin().powi(2))
        .collect();
    let want: Vec<f32> = (0..C)
        .map(|c| v1[c] + beta_b[c] * (alpha_b[c] * v1[c]).sin().powi(2))
        .collect();
    let diff = max_abs_diff(&got, &want);
    eprintln!("chained snake at inner == 1: {diff:.3e} from the two-stage closed form");
    assert!(
        diff <= SNAKE_BAR,
        "chained snake vs the closed form: max abs diff {diff:e} > {SNAKE_BAR:e}"
    );
}
