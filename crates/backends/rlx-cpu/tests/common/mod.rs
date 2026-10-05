// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Natalya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixtures shared by the single-row packed-multiply tests. `tests/common/` is not
//! a test target of its own; each test binary pulls it in with `mod common;`.
use rlx_gguf::GgmlType;

/// Max relative error, normalized by `max(1, |want|)` so that an output near zero
/// cannot dominate the figure.
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, f32::max)
}

/// `k*n` packed weights for tensor `seed`, heavy-tailed: the outliers are where a
/// quantized activation's per-block absmax scale actually costs accuracy.
pub fn packed(ggml: GgmlType, k: usize, n: usize, seed: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..k * n)
        .map(|i| {
            let s = ((i + seed * 7) as f32 * 0.017).sin();
            if i % 137 == 0 { 9.0 * s } else { s }
        })
        .collect();
    rlx_gguf::quantize(&w, ggml).expect("pack test weight")
}

/// A heavy-tailed activation row of `k` elements, same rationale as [`packed`].
pub fn activation(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let s = (i as f32 * 0.031).cos();
            if i % 61 == 0 { 12.0 * s } else { s }
        })
        .collect()
}
