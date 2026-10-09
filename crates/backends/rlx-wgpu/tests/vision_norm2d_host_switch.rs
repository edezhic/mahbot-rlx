// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The documented norm switch pair, on `Op::LayerNorm2d`.
//!
//! `RLX_WGPU_HOST_NORM` advertises "force norms onto the host" and
//! `RLX_WGPU_GPU_NORM` "force norms onto the GPU (opposite of
//! RLX_WGPU_HOST_NORM)". `Op::LayerNorm2d` used to be a host step whatever
//! either switch said, so the force-device switch did not actually cover this
//! normalisation. This pins the pair behaviourally rather than vacuously in the
//! default configuration: with the host switch alone the normalisation is a CPU
//! step (and the CPU step is what the schedule names), and adding the
//! force-device switch — the second state the pair exists for — puts it back on
//! the device.
//!
//! Its own test binary: these are process-global switches that lowering reads
//! before the first compile, and the second state must be compiled after the
//! first.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

fn ln2d_graph(c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new("ln2d_norm_switch");
    let x = g.input("x", Shape::new(&[1, c, h, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn host_switch_selects_the_cpu_step_and_the_force_device_switch_overrides_it() {
    // The switches under test are set explicitly below; these are the other
    // routers that would otherwise decide the route for us. The shard-cap lever
    // is neutralised to a value the adapter clamps back to its own limit, so a
    // lowered ambient one cannot stripe this graph's arena.
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let (c, h, w) = (32usize, 2usize, 3usize);
    let xs: Vec<f32> = (0..c * h * w)
        .map(|i| 64.0 + ((i % 13) as f32 - 6.0))
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i % 5) as f32 * 0.03).collect();
    let beta: Vec<f32> = (0..c).map(|i| (i % 3) as f32 * 0.05 - 0.05).collect();

    // State 1: the host switch alone. The normalisation runs on the CPU, and
    // this backend's own schedule names that step.
    rlx_ir::env::set("RLX_WGPU_GPU_NORM", "0");
    rlx_ir::env::set("RLX_WGPU_HOST_NORM", "1");
    let mut host = WgpuExecutable::compile(ln2d_graph(c, h, w));
    let host_kinds = host.step_kind_counts();
    eprintln!("RLX_WGPU_HOST_NORM=1: {host_kinds:?}");
    assert_eq!(
        host_kinds.get("layer_norm2d_host").copied().unwrap_or(0),
        1,
        "the host switch must select the CPU step; schedule = {host_kinds:?}"
    );
    assert!(
        !host_kinds.contains_key("layer_norm2d"),
        "the host switch must not leave the device step in place; schedule = {host_kinds:?}"
    );

    // State 2: the force-device switch on top of it. Distinguishing the two
    // states is the point — the default configuration would pass either way.
    rlx_ir::env::set("RLX_WGPU_GPU_NORM", "1");
    let mut device = WgpuExecutable::compile(ln2d_graph(c, h, w));
    let device_kinds = device.step_kind_counts();
    eprintln!("RLX_WGPU_HOST_NORM=1 RLX_WGPU_GPU_NORM=1: {device_kinds:?}");
    assert_eq!(
        device_kinds.get("layer_norm2d").copied().unwrap_or(0),
        1,
        "the force-device switch must cover this normalisation; schedule = {device_kinds:?}"
    );
    assert!(
        !device_kinds.contains_key("layer_norm2d_host"),
        "the force-device switch must override the host switch; schedule = {device_kinds:?}"
    );

    // Both routes compute the same normalisation.
    host.set_param("gamma", &gamma);
    host.set_param("beta", &beta);
    let host_out = host.run(&[("x", &xs)]).remove(0);
    device.set_param("gamma", &gamma);
    device.set_param("beta", &beta);
    let device_out = device.run(&[("x", &xs)]).remove(0);

    let mut cpu = Session::new(Device::Cpu).compile(ln2d_graph(c, h, w));
    cpu.set_param("gamma", &gamma);
    cpu.set_param("beta", &beta);
    let want = cpu.run(&[("x", &xs)]).remove(0);

    for (route, got) in [("host step", &host_out), ("device step", &device_out)] {
        assert_eq!(got.len(), want.len(), "{route}: length");
        let diff = max_abs_diff(got, &want);
        assert!(
            diff <= 1e-4,
            "{route} of layer_norm2d vs cpu: max abs diff {diff:e} > 1e-4"
        );
    }

    rlx_ir::env::unset("RLX_WGPU_GPU_NORM");
    rlx_ir::env::unset("RLX_WGPU_HOST_NORM");
}
