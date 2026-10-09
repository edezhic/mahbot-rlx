// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::LayerNorm2d` when the weight and bias are parked in the weight buffer.
//!
//! `from_plan_split` parks parameters into the separate weight buffer whenever
//! the arena needs the split plan, and `RLX_WGPU_ALL_PARAMS_WEIGHT=1` forces
//! that on any adapter — it is how this backend runs on the adapters whose
//! `max_buffer_size` parks every parameter. A parked operand's `Arena::offset`
//! carries `WEIGHT_BUF_TAG`, so the device arm has to stage it into its bind
//! window instead of reading the activation arena, and must refuse rather than
//! read it as f32 lanes when its storage is packed.
//!
//! Its own test binary: the switch is process-global and lowering reads it at
//! the first compile.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

fn ln2d_graph(c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new("ln2d_parked");
    let x = g.input("x", Shape::new(&[1, c, h, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn layer_norm2d_stages_a_parked_weight_and_bias() {
    // The one switch this check is about; the rest are neutralised so the route
    // is decided by parking alone (the shard-cap lever to a value the adapter
    // clamps back to its own limit, so a lowered ambient one cannot stripe).
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_HOST_NORM",
        "RLX_WGPU_GPU_NORM",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let (c, h, w) = (1024usize, 1usize, 144usize);
    let xs: Vec<f32> = (0..c * h * w)
        .map(|i| 1024.0 + ((i % 17) as f32 - 8.0))
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let beta: Vec<f32> = (0..c).map(|i| (i % 5) as f32 * 0.02 - 0.04).collect();

    let mut gpu = WgpuExecutable::compile(ln2d_graph(c, h, w));
    let kinds = gpu.step_kind_counts();
    eprintln!("RLX_WGPU_ALL_PARAMS_WEIGHT=1: {kinds:?}");
    // The premise, not just the outcome: with every parameter parked, the device
    // step has no operand in the activation arena to read until it has staged
    // both of them, which is what the pair of copies is. Without this, a
    // `layer_norm2d` route would pass here whether or not anything was parked.
    assert_eq!(
        kinds.get("buffer_copy").copied().unwrap_or(0),
        2,
        "the weight and the bias must be parked and staged; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("layer_norm2d").copied().unwrap_or(0),
        1,
        "a parked weight/bias must be staged, not routed off the device; schedule = {kinds:?}"
    );
    gpu.set_param("gamma", &gamma);
    gpu.set_param("beta", &beta);
    let got = gpu.run(&[("x", &xs)]).remove(0);

    let mut cpu = Session::new(Device::Cpu).compile(ln2d_graph(c, h, w));
    cpu.set_param("gamma", &gamma);
    cpu.set_param("beta", &beta);
    let want = cpu.run(&[("x", &xs)]).remove(0);

    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= 1e-4,
        "layer_norm2d with a parked weight/bias vs cpu: max abs diff {diff:e} > 1e-4"
    );
    eprintln!("ln2d parked params: max abs diff {diff:.2e}");

    // The other half of the same question: when the norm is *asked* for the host
    // step while its operands are parked, that step cannot address them, so it
    // refuses loudly instead of reading the activation arena at the parked
    // offset's untagged value.
    rlx_ir::env::set("RLX_WGPU_HOST_NORM", "1");
    let caught = std::panic::catch_unwind(|| WgpuExecutable::compile(ln2d_graph(c, h, w)));
    let Err(payload) = caught else {
        panic!("the host step must refuse a parked operand, not read the wrong bytes");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("REFUSED (parked LayerNorm2d operand)"),
        "the refusal must name the parked operand; got {msg:?}"
    );
    eprintln!(
        "parked + RLX_WGPU_HOST_NORM=1: {}",
        msg.lines().next().unwrap_or("")
    );
    rlx_ir::env::set("RLX_WGPU_HOST_NORM", "0");
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
}
