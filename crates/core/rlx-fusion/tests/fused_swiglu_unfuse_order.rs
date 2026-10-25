// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `FusedSwiGLU`'s decomposition must read the half the op says is the gate.
//! `gate_first` says the builder emitted gate before up, and the halves are
//! swapped when it is set; reading up from the low half regardless (what this
//! pass used to do) computes `silu(up) · gate`, which is not the operation and
//! not a rounding of it. `rlx-wgpu`'s `tests/fusion_standdowns.rs` holds the
//! expanded node to the fused kernel's own numbers on the device; what this pins
//! is which half the decomposition reads, on the graph alone.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// The width of each half of the input row.
const HALF: usize = 2;

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// One node the expansion emits, as its `Narrow`'s start and length.
fn narrow(g: &Graph, id: NodeId) -> (usize, usize) {
    match g.node(id).op {
        Op::Narrow { start, len, .. } => (start, len),
        ref other => panic!("the expansion's halves are narrows, not {other}"),
    }
}

#[test]
fn the_decomposition_reads_the_half_gate_first_names() {
    for gate_first in [false, true] {
        let mut g = Graph::new("swiglu_unfuse");
        let x = g.input("x", shape(&[1, 2 * HALF]));
        let y = g.add_node(
            Op::FusedSwiGLU {
                cast_to: None,
                gate_first,
            },
            vec![x],
            shape(&[1, HALF]),
        );
        g.set_outputs(vec![y]);
        let out = rlx_fusion::unfuse::unfuse_fused_for_autodiff(g);

        let silu = out
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Activation(Activation::Silu)))
            .expect("the expansion computes one silu");
        // The gate half: the low one when `gate_first`, the high one otherwise.
        let gate_start = if gate_first { 0 } else { HALF };
        assert_eq!(
            narrow(&out, silu.inputs[0]),
            (gate_start, HALF),
            "gate_first={gate_first}: the silu must read the gate half"
        );
        let mul = out
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
            .expect("the expansion multiplies the two halves");
        let up = mul
            .inputs
            .iter()
            .copied()
            .find(|&i| i != silu.id)
            .expect("the multiply pairs the silu with the other half");
        assert_eq!(
            narrow(&out, up),
            (if gate_first { HALF } else { 0 }, HALF),
            "gate_first={gate_first}: the multiply pairs the silu with the other half"
        );
    }
}
