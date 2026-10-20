// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::Sample` (top-k / top-p / temperature
//! token sampling).
//!
//! CPU is the reference. Metal shares CPU's random-number stream, so with a
//! fixed seed it must return the identical row. wgpu draws from its own
//! (Threefry) stream, where only same-backend determinism is meaningful — an
//! in-graph draw must equal that backend's own standalone draw, and the
//! logits it read must come back unchanged.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn build_sample_graph(b: usize, v: usize, top_k: usize, top_p: f32, temp: f32, seed: u64) -> Graph {
    let mut g = Graph::new("sample");
    let logits = g.input("logits", Shape::new(&[b, v], DType::F32));
    let y = g.sample(
        logits,
        top_k,
        top_p,
        temp,
        seed,
        Shape::new(&[b], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn logits_data(b: usize, v: usize) -> Vec<f32> {
    // Deterministic, non-uniform logits so top-k/top-p actually bite.
    (0..b * v)
        .map(|i| {
            let r = (i % v) as f32;
            ((r * 0.37).sin() * 3.0) + ((i / v) as f32) * 0.1
        })
        .collect()
}

#[allow(dead_code)]
fn run_on(
    device: Device,
    b: usize,
    v: usize,
    top_k: usize,
    top_p: f32,
    temp: f32,
    seed: u64,
) -> Vec<f32> {
    let logits = logits_data(b, v);
    let mut exe = Session::new(device).compile(build_sample_graph(b, v, top_k, top_p, temp, seed));
    exe.run(&[("logits", logits.as_slice())]).pop().unwrap()
}

/// `batch1-draw`: one row over 4096 logits, no truncation, temperature 1.0, seed
/// 11 — the case whose `outer == 1` draw takes the cooperative single-row kernel.
/// Named because the GPU gate below reads the same tuple the parity sweep does.
const BATCH1_DRAW: (&str, usize, usize, usize, f32, f32, u64) =
    ("batch1-draw", 1, 4096, 0, 1.0, 1.0, 11);

/// The truncation the single-row gate's filtered leg sweeps: the plain draw above
/// skips `need_filter` entirely, so the cooperative top-k scans need a shape of
/// their own (`top_k=40`, `top_p=1.0`, `temp=0.9`; the loader's default is
/// `top_k=50` at that temperature).
const FILTERED_DRAW: (usize, f32, f32) = (40, 1.0, 0.9);

#[allow(dead_code)]
fn cases() -> Vec<(&'static str, usize, usize, usize, f32, f32, u64)> {
    vec![
        // (name, batch, vocab, top_k, top_p, temperature, seed)
        ("argmax-temp0", 4, 256, 0, 1.0, 0.0, 0), // temperature 0 → argmax
        ("topk1", 3, 1024, 1, 1.0, 1.0, 42),      // top_k=1 → deterministic argmax
        // The default configuration: no top-k, no top-p, temperature 1.0. This
        // must be a genuine draw, not the old `temperature == 1.0 → argmax`
        // fast path — the seed has to matter here.
        ("uniform-draw", 4, 512, 0, 1.0, 1.0, 7),
        ("topk40", 2, 32000, 40, 1.0, 0.8, 7), // realistic LLM decode
        ("topp", 2, 4096, 0, 0.9, 1.0, 123),   // nucleus
        ("topk-topp", 5, 8192, 50, 0.95, 0.7, 999), // combined
        // One row is the decode shape, and the only one the cooperative
        // single-row kernel draws: a genuine draw (not the argmax fast path), so
        // the branch is compared against this backend's own standalone draw.
        BATCH1_DRAW,
    ]
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// A sample node inside a larger graph must (1) not clobber the logits it read,
/// and (2) draw exactly what the same graph would draw standalone.
///
/// The `shared` graph is built so a second consumer of the logits — `softmax` —
/// is created *after* the sample node, and the raw `Op::Input` node is itself a
/// declared output. That keeps the logits live to the end of the run and is the
/// only way to read the exact tensor the sample consumed.
fn assert_in_graph_draw_keeps_logits(
    device: Device,
    b: usize,
    v: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: u64,
) {
    let logits = logits_data(b, v);
    let shape = Shape::new(&[b, v], DType::F32);

    let mut standalone = Graph::new("standalone");
    let l = standalone.input("logits", shape.clone());
    let y = standalone.sample(
        l,
        top_k,
        top_p,
        temperature,
        seed,
        Shape::new(&[b], DType::F32),
    );
    standalone.set_outputs(vec![y]);

    let mut reference = Graph::new("reference");
    let l = reference.input("logits", shape.clone());
    let sm = reference.softmax(l, 1, shape.clone());
    reference.set_outputs(vec![sm]);

    // Deliberately: input → sample → softmax, so the sample runs first and the
    // second consumer of the logits runs afterwards.
    let mut shared = Graph::new("shared");
    let l = shared.input("logits", shape.clone());
    let y = shared.sample(
        l,
        top_k,
        top_p,
        temperature,
        seed,
        Shape::new(&[b], DType::F32),
    );
    let sm = shared.softmax(l, 1, shape.clone());
    shared.set_outputs(vec![l, sm, y]);

    let mut standalone_exe = Session::new(device).compile(standalone);
    let mut reference_exe = Session::new(device).compile(reference);
    let mut shared_exe = Session::new(device).compile(shared);

    let standalone_out = standalone_exe.run(&[("logits", logits.as_slice())]);
    let reference_out = reference_exe.run(&[("logits", logits.as_slice())]);
    let shared_out = shared_exe.run(&[("logits", logits.as_slice())]);

    // Tolerance, not bit-exact: the corruption this guards against is O(1e-1)
    // (softmax-of-clobbered-probabilities came back with max|Δ| ≈ 0.106), i.e.
    // four orders of magnitude above the tolerance.
    assert!(
        close(&shared_out[0], &logits, 1e-5),
        "{device:?}: sample clobbered the logits it read \
         (max|Δ| = {:e})",
        max_abs_diff(&shared_out[0], &logits),
    );
    assert!(
        close(&shared_out[1], &reference_out[0], 1e-5),
        "{device:?}: the other consumer of the logits saw changed values \
         (max|Δ| = {:e})",
        max_abs_diff(&shared_out[1], &reference_out[0]),
    );
    assert!(
        close(&shared_out[2], &standalone_out[0], 1e-5),
        "{device:?}: in-graph draw {:?} != standalone draw {:?}",
        shared_out[2],
        standalone_out[0],
    );
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn sample_cpu_runs() {
    let _gpu = common::serialize_gpu();
    for (name, b, v, k, p, t, s) in cases() {
        let out = run_on(Device::Cpu, b, v, k, p, t, s);
        assert_eq!(out.len(), b, "{name}: expected {b} sampled indices");
        for &idx in &out {
            assert!(
                idx >= 0.0 && (idx as usize) < v,
                "{name}: index {idx} out of range"
            );
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn sample_metal_matches_cpu() {
    let _gpu = common::serialize_gpu();
    for (name, b, v, k, p, t, s) in cases() {
        let metal = run_on(Device::Metal, b, v, k, p, t, s);
        let cpu = run_on(Device::Cpu, b, v, k, p, t, s);
        assert_eq!(
            metal, cpu,
            "metal {name}: sampled indices diverge {metal:?} vs {cpu:?}"
        );
        eprintln!("metal {name}: {metal:?} == cpu");
    }
}

#[test]
fn sample_in_graph_matches_standalone_and_keeps_logits_cpu() {
    let _gpu = common::serialize_gpu();
    assert_in_graph_draw_keeps_logits(Device::Cpu, 4, 512, 0, 1.0, 1.0, 7);
}

#[test]
#[cfg(feature = "gpu")]
fn sample_in_graph_matches_standalone_and_keeps_logits_wgpu() {
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let _gpu = common::serialize_gpu();
    assert_in_graph_draw_keeps_logits(Device::Gpu, 4, 512, 0, 1.0, 1.0, 7);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn sample_in_graph_matches_standalone_and_keeps_logits_metal() {
    if common::skip_unless_available(Device::Metal, "metal") {
        return;
    }
    let _gpu = common::serialize_gpu();
    assert_in_graph_draw_keeps_logits(Device::Metal, 4, 512, 0, 1.0, 1.0, 7);
}

/// The default configuration (`top_k=0`, `top_p=1.0`, `temperature=1.0`) must
/// draw, not argmax: two seeds have to be able to disagree.
///
/// A single row on peaked logits cannot tell a draw from an argmax — both land
/// on the dominant token — so this uses 8 near-uniform rows over a 256-token
/// vocab: two independent draws agreeing on every row is ~(1/256)^8, which a
/// deterministic argmax would do every time.
///
/// This is the only test that fails if a backend silently re-reads the neutral
/// configuration as an argmax, so it runs wherever the backend does: the
/// in-graph equality test above compares each backend against *itself*, and an
/// argmax satisfies it.
fn assert_default_config_draws(device: Device) {
    let _gpu = common::serialize_gpu();
    let (b, v) = (8, 256);
    let logits = vec![0.0f32; b * v];
    let draw = |seed: u64| -> Vec<f32> {
        let mut exe = Session::new(device).compile(build_sample_graph(b, v, 0, 1.0, 1.0, seed));
        exe.run(&[("logits", logits.as_slice())]).pop().unwrap()
    };
    let first = draw(1);
    let second = draw(2);
    let differing = first.iter().zip(&second).filter(|(a, b)| a != b).count();
    assert!(
        differing > 0,
        "{device:?}: default Sample config must draw, not argmax: two seeds \
         agreed on all {b} rows ({first:?})"
    );
}

#[test]
fn sample_default_config_is_a_genuine_draw_cpu() {
    assert_default_config_draws(Device::Cpu);
}

#[test]
#[cfg(feature = "gpu")]
fn sample_default_config_is_a_genuine_draw_wgpu() {
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    assert_default_config_draws(Device::Gpu);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn sample_default_config_is_a_genuine_draw_metal() {
    if common::skip_unless_available(Device::Metal, "metal") {
        return;
    }
    assert_default_config_draws(Device::Metal);
}

/// The cooperative one-row branch must pick the token the serial per-row path
/// picks for row 0 — on both of the branch's legs: the plain draw, and the
/// top-k filter the product path runs.
///
/// `sample_one_coop` splits a single row across the workgroup; the reductions
/// whose *order* decides the drawn id stay serial on lane 0, and the Threefry
/// counter it feeds is `(0, i)` — exactly what the serial path uses for row 0.
/// So a batch-1 draw and row 0 of a batch-2 draw at the same seed must agree,
/// while the batch-2 draw (`outer == 2`) takes the serial per-row path with
/// counter `(row, i)`. The unfiltered case skips `need_filter` entirely, so the
/// filter's workgroup scans, sentinel marking and lane-0 restore/renormalise are
/// swept on a second shape at its own truncation (`FILTERED_DRAW`). The seeds
/// are swept because the drawn id is the only observable, and a single seed that
/// happens to pick the same token either way cannot see a last-bit difference in
/// the cooperative reduction.
///
/// The two rows of the batch-2 draw differ only by the 0.1 per-row offset the
/// logits fixture adds, which no argmax can see, so a seed where they disagree
/// also rules out an argmax fast path behind either branch.
#[test]
#[cfg(feature = "gpu")]
fn sample_batch1_coop_draw_matches_the_serial_row_wgpu() {
    let (_, _, v, top_k, top_p, temp, case_seed) = BATCH1_DRAW;
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let _gpu = common::serialize_gpu();

    for (top_k, top_p, temp) in [(top_k, top_p, temp), FILTERED_DRAW] {
        let mut drew_apart = false;
        for seed in [case_seed].into_iter().chain(1u64..=8) {
            let coop = run_on(Device::Gpu, 1, v, top_k, top_p, temp, seed);
            let serial = run_on(Device::Gpu, 2, v, top_k, top_p, temp, seed);
            assert_eq!(coop.len(), 1, "seed {seed}: expected one drawn row");
            assert_eq!(serial.len(), 2, "seed {seed}: expected two drawn rows");
            for &idx in coop.iter().chain(&serial) {
                assert!(
                    idx >= 0.0 && (idx as usize) < v,
                    "batch1-draw (top_k={top_k}) seed {seed}: index {idx} out of range"
                );
            }
            assert_eq!(
                coop[0], serial[0],
                "batch1-draw (top_k={top_k}, temp={temp}) seed {seed}: the cooperative \
                 one-row draw {} != serial row 0 {}",
                coop[0], serial[0]
            );
            drew_apart |= serial[0] != serial[1];
            eprintln!(
                "wgpu batch1-draw (top_k={top_k}, temp={temp}) seed {seed}: coop {coop:?}, \
                 serial {serial:?}"
            );
        }
        assert!(
            drew_apart,
            "batch1-draw (top_k={top_k}, temp={temp}): both rows of every batch-2 draw \
             picked the same token — the logits fixture differs by only 0.1 per row, so \
             that is an argmax, not a draw"
        );
    }
}
