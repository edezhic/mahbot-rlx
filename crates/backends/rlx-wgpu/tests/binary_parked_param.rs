// RLX — versatile ML compiler and runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A per-channel **parameter** operand of an `Op::Binary`, parked in the weight
//! buffer.
//!
//! The runtime path legalises binary broadcasts before fusion, and the lowering
//! peels the copy it inserted for a per-channel operand
//! (`rlx_wgpu::unfuse::elide_binary_broadcast_expands`) — so the operand arrives
//! at the elementwise step itself, read through `(i / rep) % len`. When that
//! operand is a parameter, *staging* is what makes it readable: a parked
//! parameter's `Arena::offset` carries `WEIGHT_BUF_TAG` (the weight buffer, not
//! the activation arena), so the arm has to copy it into its bind window
//! (`buffer_copy`) and read it there — an input's own slot is read in place.
//! Both have to give the CPU answer.
//!
//! The step table is read off the **legalised** graph: legalising first is what
//! makes the absence of an `expand` step evidence that the peel fired rather
//! than that this graph was simply never given a copy, and the `buffer_copy` is
//! the staging the device route adds for a parked parameter.
//!
//! Its own test binary: `RLX_WGPU_ALL_PARAMS_WEIGHT` is process-global and the
//! lowering reads it at the first compile.

mod common;

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn parked_per_channel_parameter_is_staged_and_read_channel_wise() {
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

    let (b, c, t) = (2usize, 6usize, 5usize);
    let xs: Vec<f32> = (0..b * c * t)
        .map(|i| ((i * 41 % 173) as f32) / 8.0 - 4.0)
        .collect();
    let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.25 - 0.5).collect();

    let mut g = Graph::new("parked_per_channel");
    let x = g.input("x", Shape::new(&[b, c, t], DType::F32));
    let p = g.param("bias", Shape::new(&[1, c, 1], DType::F32));
    let y = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![x, p],
        Shape::new(&[b, c, t], DType::F32),
    );
    g.set_outputs(vec![y]);

    // The copy the legalising pass inserts for the per-channel operand is peeled
    // (no `expand` step), and the parked parameter is staged for the step that
    // reads it.
    let names = common::legalised_step_names(&g);
    // `should_pack_host_op` routes the whole elementwise family to `Step::HostOp`
    // on any Vulkan/DX12 adapter, and that arm precedes the `Op::Binary` arm — so
    // on a discrete rig (lavapipe included) there is no device step and no
    // staging copy to assert, only the answer.
    if names == ["host_op"] {
        eprintln!(
            "parked per-channel operand: the elementwise family is host-routed on this adapter — \
             no staging step to assert"
        );
    } else {
        assert_eq!(
            names,
            ["buffer_copy", "binary"],
            "parked per-channel operand: expected a staging copy and the elementwise step"
        );
    }

    let mut gpu = Session::new(Device::Gpu).compile(g.clone());
    gpu.set_param("bias", &bias);
    let got = gpu.run(&[("x", &xs)]).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    cpu.set_param("bias", &bias);
    let want = cpu.run(&[("x", &xs)]).remove(0);
    // `max_abs_diff` censuses finiteness before it folds, so a wrong-byte read of
    // the parked weight cannot fold away behind a `f32::max`.
    let max_d = common::max_abs_diff(&got, &want);
    assert_eq!(
        max_d, 0.0,
        "a parked per-channel parameter must reach the kernel channel-wise (max|Δ| = {max_d:.3e})"
    );
}
