// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `Step::BufferCopy` must move exactly the requested bytes: a value-preserving
//! `Op::Cast` of an empty tensor lowers to a zero-byte one, and this pins that
//! the step is still emitted and that the graph's own output keeps its bytes
//! (the skip itself is pinned device-free in `copy_route`).

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// A zero-element `Bool` param, cast to `F32` (mode 0, a value-preserving
/// copy), ahead of a real 64×64 computation.
fn graph_with_zero_byte_copy() -> Graph {
    let mut g = Graph::new("zero_byte_copy");
    let x = g.input("x", Shape::new(&[64, 64], DType::F32));
    let w = g.param("w", Shape::new(&[0, 4], DType::Bool));
    let cast = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![w],
        Shape::new(&[0, 4], DType::F32),
    );
    let y = g.binary(BinaryOp::Add, x, x, Shape::new(&[64, 64], DType::F32));
    g.set_outputs(vec![y, cast]);
    g
}

#[test]
fn zero_byte_copy_leaves_every_other_tensor_untouched() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let mut exe = WgpuExecutable::compile(graph_with_zero_byte_copy());
    // Without the zero-byte copy step the assertions below would pass for a
    // graph that no longer exercises the route this file is about.
    let steps = exe.test_step_names();
    assert!(
        steps.contains(&"buffer_copy"),
        "the graph no longer lowers to a zero-byte copy step: {steps:?}"
    );
    let out = exe.run(&[("x", &[1.0; 64 * 64])]);
    assert_eq!(
        out[0],
        vec![2.0; 64 * 64],
        "the zero-byte copy must not write into the arena"
    );
    assert!(out[1].is_empty(), "zero elements read back empty");
}
