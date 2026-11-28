// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The elementwise in-place plan is the same arithmetic: a `conv → bias-add →
//! activation` chain planned with the rule writes its output where it read its
//! input, and every bit of the result must match the two-block plan's.
//!
//! The rule changes an offset, not a kernel. This runs both layouts of one
//! graph on the device and compares the outputs bit for bit, and checks that
//! the arena actually fell — an exactness check that never took the aliased
//! route would prove nothing.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

/// `[1,1,16,16] → conv 3×3/2 → +bias → gelu`: the encoder front's chain.
fn chain() -> (Graph, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("elementwise_in_place");
    let x = g.input("x", Shape::new(&[1, 1, 16, 16], F));
    let w = g.param("w", Shape::new(&[8, 1, 3, 3], F));
    let conv = g.conv2d(x, w, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    let b = g.param("b", Shape::new(&[1, 8, 1, 1], F));
    let add = g.add(conv, b);
    let act = g.gelu(add);
    g.set_outputs(vec![act]);
    let ramp = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    };
    (g, ramp(72, 0.11), ramp(8, 0.53), ramp(256, 0.97))
}

/// One plan arm: compile, fill the parameters, run, and report the arena the
/// graph was built at.
fn arm(no_inplace: bool, w: &[f32], b: &[f32], x: &[f32]) -> (u64, Vec<f32>) {
    rlx_ir::env::set("RLX_WGPU_NO_INPLACE", if no_inplace { "1" } else { "0" });
    let (graph, ..) = chain();
    let mut compiled = Session::new(Device::Gpu).compile(graph);
    compiled.set_param("w", w);
    compiled.set_param("b", b);
    let out = compiled.run(&[("x", x)]).remove(0);
    (
        compiled
            .arena_bytes()
            .expect("the wgpu backend reports its arena"),
        out,
    )
}

#[test]
fn the_two_layouts_compute_the_same_bits_and_the_aliased_one_is_smaller() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (_, w, b, x) = chain();
    let (plain_arena, plain) = arm(true, &w, &b, &x);
    let (in_place_arena, in_place) = arm(false, &w, &b, &x);

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&plain),
        bits(&in_place),
        "writing where it read must be the same operation on the same values"
    );
    assert!(
        in_place_arena < plain_arena,
        "the aliased plan must actually take the rule: {in_place_arena} vs {plain_arena} B"
    );
}
