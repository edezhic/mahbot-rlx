// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixtures shared by this crate's parity tests. `tests/common/` is not a test
//! target of its own; each test binary pulls it in with `mod common;`.

// Each test binary declaring `mod common;` uses only the helpers it needs, so
// the rest are dead code in that binary and `just lint` runs with `-D warnings`.
#![allow(dead_code)]

/// Largest absolute difference between a computed and a reference output, after
/// a finiteness census.
///
/// The census runs FIRST and is asserted clean, because `f32::max` returns the
/// non-NaN operand: an all-NaN or infinity-poisoned output would otherwise fold
/// to a small number and pass any tolerance, so a broken kernel could satisfy a
/// parity bar. NaN, +inf and −inf are counted separately so an infinity cannot
/// hide behind a NaN. The Metal prefill parity suite carries the same rule.
pub fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "parity needs equal lengths");
    let (mut nan, mut pos_inf, mut neg_inf) = (0usize, 0usize, 0usize);
    for &v in got.iter().chain(want) {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos_inf += 1;
        } else if v == f32::NEG_INFINITY {
            neg_inf += 1;
        }
    }
    assert_eq!(
        (nan, pos_inf, neg_inf),
        (0, 0, 0),
        "non-finite in the parity pair: NaN={nan} +inf={pos_inf} -inf={neg_inf}"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// How many schedule steps are the materialised broadcast family
/// (`expand` / `expand_host`) — the lowering leaves one wherever a broadcast
/// operand has to arrive as a real tensor.
pub fn expand_steps(names: &[&str]) -> usize {
    names
        .iter()
        .filter(|n| **n == "expand" || **n == "expand_host")
        .count()
}

/// The step names this backend dispatches for `g` as the runtime path reaches it:
/// the legalising pass first (what the shim runs ahead of fusion —
/// `rlx-runtime/src/backend/wgpu_backend.rs`), then the lowering.
///
/// Legalising first is what makes a *missing* `expand` step evidence that the
/// peel fired, rather than evidence that the pass never ran and inserted none.
pub fn legalised_step_names(g: &rlx_ir::Graph) -> Vec<&'static str> {
    use rlx_opt::pass::Pass as _;
    let legalised = rlx_opt::LegalizeBroadcast.run(g.clone());
    rlx_wgpu::backend::WgpuExecutable::compile(legalised).test_step_names()
}
