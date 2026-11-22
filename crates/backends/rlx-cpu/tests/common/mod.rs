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

/// The fork's paired-reading figure: the median of the rounds' timings.
fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.into_iter().collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A paired reading at kernel scale: `rounds` rounds, each timing every arm once,
/// back to back, in alternating order — so a machine that drifts across the run
/// moves both arms of a round together rather than favouring one. Returns each
/// arm's median, in arm order.
#[allow(dead_code)]
pub fn paired(rounds: usize, arms: usize, mut run: impl FnMut(usize) -> f64) -> Vec<f64> {
    let mut times = vec![Vec::with_capacity(rounds); arms];
    for round in 0..rounds {
        let order = (0..arms).map(|arm| if round % 2 == 0 { arm } else { arms - 1 - arm });
        for arm in order {
            times[arm].push(run(arm));
        }
    }
    times.iter().map(|t| median(t.iter().copied())).collect()
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
