// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared FKL-style benchmark / test graphs.

use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue};
use rlx_ir::{DType, Graph, Op, Shape};

pub fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

/// `resize ? act` primitive chain (unfused baseline), with the code chosen — the shape
/// [`resize_relu_graph`] builds.
pub fn resize_activation_graph(
    name: &str,
    act: Activation,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", nchw(n, c, h, w));
    let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(n, c, h * 2, w * 2));
    let out = g.add_node(Op::Activation(act), vec![up], nchw(n, c, h * 2, w * 2));
    g.set_outputs(vec![out]);
    g
}

/// `resize ? relu` primitive chain (unfused baseline).
pub fn resize_relu_graph(name: &str, n: usize, c: usize, h: usize, w: usize) -> Graph {
    resize_activation_graph(name, Activation::Relu, n, c, h, w)
}

/// Hand-built `ElementwiseRegion` with resize prologue.
pub fn resize_relu_region_graph(name: &str, n: usize, c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", nchw(n, c, h, w));
    let chain = vec![ChainStep::Activation(
        Activation::Relu,
        ChainOperand::Input(0),
    )];
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::ResizeNearest2x,
            prologue_input: 0,
        },
        vec![x],
        nchw(n, c, h * 2, w * 2),
    );
    g.set_outputs(vec![out]);
    g
}

/// Per-slice `ElementwiseRegion` + `Concat` (batch fusion input), with the code chosen —
/// the shape [`batch_narrow_relu_regions_graph`] builds.
pub fn batch_narrow_activation_regions_graph(
    name: &str,
    act: Activation,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let batch = g.input("batch", nchw(batch_n, c, h, w));
    let chain = vec![ChainStep::Activation(act, ChainOperand::Input(0))];
    let mut slices = Vec::with_capacity(batch_n);
    for i in 0..batch_n {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start: i,
                len: 1,
            },
            vec![batch],
            nchw(1, c, h, w),
        );
        slices.push(g.add_node(
            Op::ElementwiseRegion {
                chain: chain.clone(),
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![sl],
            nchw(1, c, h, w),
        ));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, nchw(batch_n, c, h, w));
    g.set_outputs(vec![out]);
    g
}

/// Per-slice `ElementwiseRegion` + `Concat` (batch fusion input).
pub fn batch_narrow_relu_regions_graph(
    name: &str,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    batch_narrow_activation_regions_graph(name, Activation::Relu, batch_n, c, h, w)
}

/// Primitive `narrow ? act ? concat` (needs `MarkBatchSliceRegions`), with the code
/// chosen — the shape [`batch_narrow_relu_primitive_graph`] builds.
pub fn batch_narrow_activation_primitive_graph(
    name: &str,
    act: Activation,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let batch = g.input("batch", nchw(batch_n, c, h, w));
    let mut slices = Vec::with_capacity(batch_n);
    for i in 0..batch_n {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start: i,
                len: 1,
            },
            vec![batch],
            nchw(1, c, h, w),
        );
        slices.push(g.add_node(Op::Activation(act), vec![sl], nchw(1, c, h, w)));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, nchw(batch_n, c, h, w));
    g.set_outputs(vec![out]);
    g
}

/// Primitive `narrow ? relu ? concat` (needs `MarkBatchSliceRegions`).
pub fn batch_narrow_relu_primitive_graph(
    name: &str,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    batch_narrow_activation_primitive_graph(name, Activation::Relu, batch_n, c, h, w)
}
