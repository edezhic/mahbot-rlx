// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `CompiledGraph::host_execution_summary()` on native wgpu — the one call that
//! says *which* steps of a compiled schedule the CPU runs, and in how many sync
//! points.
//!
//! Two schedules, one compile each: a matmul-only graph, which is device work
//! end to end and answers `None`, and a graph that adds a convolution, whose
//! host route is forced with `RLX_WGPU_CONV_HOST=1` so the answer does not
//! depend on which adapter the test happens to run on. The summary must name the
//! step kind, because "contains host-executed steps" was the answer before the
//! plan existed and it is the one thing a caller cannot act on.
//!
//! Both halves live in one test on purpose: the switch is process-global and
//! read at compile time, so a second test running beside it could compile the
//! conv under the other arm.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// A single matmul: no host step of any kind.
fn matmul_graph() -> Graph {
    let mut g = Graph::new("host_summary_mm");
    let x = g.input("x", Shape::new(&[4, 8], DType::F32));
    let w = g.param("w", Shape::new(&[8, 4], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[4, 4], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// A small NCHW convolution behind the same matmul, so the schedule carries one
/// host step and one device step.
fn conv_graph() -> Graph {
    let mut g = Graph::new("host_summary_conv");
    let x = g.input("x", Shape::new(&[1, 2, 8, 8], DType::F32));
    let k = g.param("k", Shape::new(&[3, 2, 3, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3 * 6 * 6, 4], DType::F32));
    let c = g.conv2d(x, k, [3, 3], [1, 1], [0, 0], [1, 1], 1);
    let flat = g.reshape(
        c,
        vec![1, 3 * 6 * 6],
        Shape::new(&[1, 3 * 6 * 6], DType::F32),
    );
    let y = g.matmul(flat, w, Shape::new(&[1, 4], DType::F32));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn host_execution_summary_names_the_host_steps_on_native_wgpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let device_only = Session::new(Device::Gpu).compile(matmul_graph());
    assert_eq!(
        device_only.host_execution_summary(),
        None,
        "a matmul-only schedule runs entirely on the device"
    );

    rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
    let hosted = Session::new(Device::Gpu).compile(conv_graph());
    rlx_ir::env::unset("RLX_WGPU_CONV_HOST");

    let summary = hosted
        .host_execution_summary()
        .expect("the forced host conv route puts a step on the host");
    eprintln!("host summary: {summary}");
    assert!(
        summary.contains("conv2d_host"),
        "the summary must name the step kind, not just its existence: {summary}"
    );
}
