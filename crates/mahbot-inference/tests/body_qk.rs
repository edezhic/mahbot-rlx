// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One body layer's values, with q and k joined into one wide multiply and with the
//! three projections separate — the recognition file's own `blk.0` weights, through the
//! product's own [`graphs::BodyStateStage`] and [`graphs::Runner::compile`].
//!
//! The two arms differ in nothing but how the attention input is emitted, so this is the
//! value half of the join's evidence: each output column of the joined multiply reads the
//! bytes the separate one read and reduces over the same input width, so no value should
//! move beyond another summation order. The pinned recognition witness is text-level, so a
//! wrong offset in the joined parameter's binding, or a wider kernel cell reading
//! differently, would show up here and nowhere else.
//!
//! The separate arm is also the fallback's end-to-end pass: no shipped body refuses the
//! two-way join, so the arm a refusing file takes is built directly — the emission, the
//! upload and the run are the shipped code path, only the decision that picks the arm is
//! bypassed.
//!
//! Needs the installed model directory and a wgpu adapter; a machine without them skips.
//! Run with `cargo test -p mahbot-inference --release --features hand-run --test body_qk
//! -- --ignored --nocapture`.

use mahbot_inference::graphs;
use mahbot_inference::loader::{Pair, Role};
use rlx_runtime::Device;
use std::sync::Arc;

mod support;

/// Rows in the layer this test builds: two, so the causal attention is not degenerate — over a
/// single key every query reads `v` whatever q and k are.
const ROWS: usize = 2;

/// The bar the arms' rows are held to, **relative** to the row's largest element: another
/// summation order of the same products, never another operation. The joined width is a
/// different kernel cell and at two rows takes a different GEMM route, so the reduction
/// order differs; `multiply_census`'s absolute 1e-4 would be a 1e-6 relative bar on rows
/// whose elements reach ~2e2, tighter than another summation order promises. Measured
/// here: 5.4e-7 of the row's largest element.
const VALUE_BAR: f32 = 1e-5;

/// One arm: the body's first block, built the way the product builds it, with the
/// attention input emitted either as the joined q+k or as three separate projections. `None`
/// when the file cannot emit that form.
fn arm(pair: &Arc<Pair>, fuse: graphs::AttentionFuse) -> Option<Vec<Vec<f32>>> {
    let cfg = pair.body.config().clone();
    let built = support::body_graph(pair, fuse, 1, ROWS, false)?;
    let mut runner = graphs::Runner::compile(Device::Gpu, built).expect("compile the layer");
    let input: Vec<f32> = (0..ROWS * cfg.d)
        .map(|i| ((i as f32) * 0.0007).sin())
        .collect();
    // Position 0 is the identity rotation; a zero `cos` would zero q and k and leave the
    // attention reading nothing but `v`.
    let mut cos = vec![1.0f32; ROWS * cfg.head_dim / 2];
    let mut sin = vec![0.0f32; ROWS * cfg.head_dim / 2];
    cos[cfg.head_dim / 2..].fill(0.5);
    sin[cfg.head_dim / 2..].fill(0.866_025_4);
    Some(runner.run(&[
        ("x".to_string(), input),
        ("cos".to_string(), cos),
        ("sin".to_string(), sin),
    ]))
}

#[test]
#[ignore = "needs the installed model directory and a wgpu adapter; measurement only"]
fn the_body_layer_reads_the_same_values_with_q_and_k_joined_or_separate() {
    let Some(pair) = support::pair_of(Role::Recognition) else {
        return;
    };
    let (Some(fused), Some(separate)) = (
        arm(&pair, graphs::AttentionFuse::Qk),
        arm(&pair, graphs::AttentionFuse::Separate),
    ) else {
        eprintln!("skipped: the recognition file cannot emit both attention forms");
        return;
    };
    assert_eq!(
        fused.len(),
        separate.len(),
        "both arms publish the stage's outputs: the hidden rows and the cache rows"
    );
    for (index, (a, b)) in fused.iter().zip(&separate).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "output {index}: the two arms' widths differ"
        );
        assert!(
            a.iter().chain(b).any(|v| *v != 0.0),
            "output {index} is zero in both arms: the comparison would be vacuous"
        );
        let largest = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "[body qk] output={index} elements={} nonzero={} largest_difference={largest:e} \
             largest_element={scale:e}",
            a.len(),
            a.iter().filter(|v| **v != 0.0).count(),
        );
        assert!(
            largest <= VALUE_BAR * scale,
            "output {index}: joining q and k moved the layer's values (largest difference \
             {largest:e} against a largest element of {scale:e})"
        );
    }
}
