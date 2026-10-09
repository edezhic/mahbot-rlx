// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channels-first normalisation and the flip, with this backend's
//! adapter-dependent host policy forced on.
//!
//! `RLX_WGPU_DISCRETE_HOST=1` is the switch that puts a discrete adapter back on
//! the whole-host route for the norm family, and the adapter-kind test behind it
//! (`coop_discrete_backend`) is true on the software Vulkan adapter this
//! project's Linux port is validated on. Neither of these two operations may
//! consult it: their device path is exact on every adapter, so gating them on
//! the adapter kind would return them to the host on exactly the adapters this
//! backend is measured on. Forcing the policy on has to leave both on the device.
//!
//! On an adapter whose kind test is false — Metal, here — this is the same state
//! the default configuration compiles in, so it is the software Vulkan gate that
//! makes the arm bite; the route pins below hold on either kind, and this run also
//! prices the two device steps' parity.
//!
//! Its own test binary: the switch is process-global and lowering reads it at the
//! first compile.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// One graph with both operations, at the rank/dtype the synthesis path presents
/// them: the channels-first normalisation, the rank-4 strip the speech encoder
/// flips, and the rank-1 strip the mel front-end flips.
///
/// The elementwise multiply feeding the normalisation is deliberate: on the
/// adapters this file's switch is about, the policy hosts that family, so the
/// normalisation then reads an input a host step wrote — the cross-route chain
/// the parity arm below is there to price.
fn policy_graph() -> Graph {
    let mut g = Graph::new("adapter_policy");
    let x = g.input("x", Shape::new(&[1, 64, 1, 8], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[64], DType::F32));
    let beta = g.param("beta", Shape::new(&[64], DType::F32));
    let sq = g.mul(x, x);
    let n = g.layer_norm2d(sq, gamma, beta, 1e-6);
    let s4 = g.input("s4", Shape::new(&[1, 64, 1, 3], DType::F32));
    let r4 = g.reverse(s4, vec![3]);
    let s1 = g.input("s1", Shape::new(&[200], DType::F32));
    let r1 = g.reverse(s1, vec![0]);
    g.set_outputs(vec![n, r4, r1]);
    g
}

#[test]
fn the_adapter_host_policy_does_not_reach_either_operation() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_HOST_NORM",
        "RLX_WGPU_GPU_NORM",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_DISCRETE_HOST", "1");
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let mut gpu = WgpuExecutable::compile(policy_graph());
    let kinds = gpu.step_kind_counts();
    eprintln!("RLX_WGPU_DISCRETE_HOST=1: {kinds:?}");
    assert_eq!(
        kinds.get("layer_norm2d").copied().unwrap_or(0),
        1,
        "the adapter-dependent host policy must not reach the channels-first \
         normalisation; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("reverse").copied().unwrap_or(0),
        2,
        "the adapter-dependent host policy must not reach the flip; schedule = {kinds:?}"
    );

    // The forced policy is also the run the two device steps are measured in —
    // here the normalisation reads what the multiply produced, and on an adapter
    // the policy bites on that multiply is a host step.
    let gamma: Vec<f32> = (0..64).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let beta: Vec<f32> = (0..64).map(|i| (i % 5) as f32 * 0.02 - 0.04).collect();
    let xs: Vec<f32> = (0..512).map(|i| 128.0 + ((i % 11) as f32 - 5.0)).collect();
    let s4: Vec<f32> = (0..192).map(|i| i as f32 * 0.5).collect();
    let s1: Vec<f32> = (0..200).map(|i| i as f32 - 100.0).collect();
    gpu.set_param("gamma", &gamma);
    gpu.set_param("beta", &beta);
    let got = gpu.run(&[("x", &xs), ("s4", &s4), ("s1", &s1)]);

    let mut cpu = Session::new(Device::Cpu).compile(policy_graph());
    cpu.set_param("gamma", &gamma);
    cpu.set_param("beta", &beta);
    let want = cpu.run(&[("x", &xs), ("s4", &s4), ("s1", &s1)]);

    let norm_diff = max_abs_diff(&got[0], &want[0]);
    assert!(
        norm_diff <= 1e-4,
        "channels-first normalisation under the forced policy: max abs diff {norm_diff:e} > 1e-4"
    );
    assert_eq!(got[1], want[1], "rank-4 flip under the forced policy");
    assert_eq!(got[2], want[2], "rank-1 flip under the forced policy");
    eprintln!("forced-policy parity: norm {norm_diff:.2e}, flips exact");
}
