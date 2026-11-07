// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A segmentation net's pointwise conv into an `ArgMax` is one kernel, and the
//! indices it writes are indices.
//!
//! `unfuse::fold_conv_argmax` replaces the pair with a `rlx_wgpu.conv3d_argmax`
//! node so the class logits are never stored. The defect this pins: the folded
//! node lowers to the conv kernel with an "argmax" store mode, and a store mode
//! the dispatch ladder did not carry left it writing the **logits** where the
//! graph declared indices — a wrong answer with no error, and one that no
//! op-level tolerance catches, because both are plausible floats. So the
//! reference here is computed in plain Rust from the same operands, and the
//! output is required to be whole numbers equal to it.
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// `[1, C_IN, 1, 2, 3]` in, `[1, 1, 2, 3]` out: six voxels of five classes, a
/// 1x1x1 kernel, so the conv is a per-voxel matmul and the reference needs no
/// convolution.
const C_IN: usize = 4;
const C_OUT: usize = 5;
const VOXELS: usize = 6;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 17.0)
        .collect()
}

fn graph() -> Graph {
    let mut g = Graph::new("conv3d_argmax");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, 2, 3], DType::F32));
    let w = g.input("w", Shape::new(&[C_OUT, C_IN, 1, 1, 1], DType::F32));
    let logits = g.add_node(
        Op::Conv3d {
            stride: [1, 1, 1],
            padding: [0, 0, 0],
            dilation: [1, 1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, C_OUT, 1, 2, 3], DType::F32),
    );
    let argmax = g.add_node(
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        vec![logits],
        Shape::new(&[1, 1, 2, 3], DType::F32),
    );
    g.set_outputs(vec![argmax]);
    g
}

/// Each voxel's largest class, from the same operands.
fn reference(x: &[f32], w: &[f32]) -> Vec<f32> {
    (0..VOXELS)
        .map(|v| {
            let best = (0..C_OUT)
                .map(|c| {
                    let dot: f32 = (0..C_IN).map(|i| x[i * VOXELS + v] * w[c * C_IN + i]).sum();
                    (dot, c)
                })
                .fold((f32::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a });
            best.1 as f32
        })
        .collect()
}

#[test]
fn the_folded_conv_argmax_writes_indices_not_logits() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let x = ramp(C_IN * VOXELS, 3);
    let w = ramp(C_OUT * C_IN, 5);
    let want = reference(&x, &w);
    // The fold is only interesting if the classes actually differ per voxel.
    assert!(
        want.iter().any(|&v| v != want[0]),
        "the fixture's argmax is one class everywhere: {want:?}"
    );

    let mut exe = WgpuExecutable::compile(graph());
    let steps = exe.test_step_names();
    assert!(
        !steps.contains(&"argmax"),
        "the conv and its argmax are one node after the fold, so no argmax step: {steps:?}"
    );
    assert_eq!(
        steps.iter().filter(|s| **s == "conv3d").count(),
        1,
        "one conv3d step carries the fold: {steps:?}"
    );

    let got = exe.run(&[("x", &x), ("w", &w)]).remove(0);
    assert_eq!(got.len(), VOXELS, "one index per voxel: {got:?}");
    for (v, (&index, &want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            index, want,
            "voxel {v} answered {index}, which is not the index {want} — the whole output: {got:?}"
        );
    }
    assert!(
        got.iter()
            .all(|v| v.fract() == 0.0 && (0.0..C_OUT as f32).contains(v)),
        "every value must be a class index, not a logit: {got:?}"
    );
}
