// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sampler's one-row draw, both paths of it: `sample_one_staged` (the row in
//! workgroup storage, taken when it fits `SAMPLE_STAGE`) and `sample_one_coop`
//! (the same arithmetic over the arena, which carries a wider row).
//!
//! The two must draw the same id from the same probabilities — they are the same
//! algorithm — so the equivalence below is pinned at the stage boundary: a row
//! that fits, padded with `-inf` logits that cannot be drawn, is a wider row whose
//! extra elements all have probability zero. Every pick, the exponent sum and the
//! Gumbel scores are then the same numbers in both paths, and a divergence is a
//! wrong answer rather than a numerically close one.

mod common;

use rlx_ir::env::RuntimeOverrides;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::kernels::SAMPLE_STAGE;

/// The sampler as a graph of one node: `[1, width]` logits in, one drawn id out.
fn sampler(width: usize, top_k: usize, top_p: f32, temperature: f32, seed: u64) -> WgpuExecutable {
    let mut g = Graph::new("sample-paths");
    let l = g.input("l", Shape::new(&[1, width], DType::F32));
    let id = g.add_node(
        Op::Sample {
            top_k,
            top_p,
            temperature,
            seed,
        },
        vec![l],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![id]);
    WgpuExecutable::compile(g)
}

/// A deterministic row of `width` logits. The four kinds are the shapes a draw
/// has to agree on: smooth, peaked, and two with heavy ties, where the
/// lowest-index tie-break is what decides the id.
fn row(width: usize, kind: usize) -> Vec<f32> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32) / ((1u64 << 53) as f32)
    };
    (0..width)
        .map(|i| match kind {
            0 => next() * 6.0 - 3.0,
            1 => {
                if i == width / 3 {
                    12.0
                } else {
                    next() * 0.5
                }
            }
            2 => ((i % 6) as f32) * 0.25,
            _ => ((i / 7) as f32) * 0.05,
        })
        .collect()
}

fn draw(exe: &mut WgpuExecutable, logits: &[f32]) -> i32 {
    exe.run(&[("l", logits)])[0][0] as i32
}

/// The staged path and the cooperative one, on the same probabilities.
#[test]
fn a_staged_draw_agrees_with_the_cooperative_path() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // One device and a finite box: the cases in this file run one at a time.
    let _guard = common::route_lock();
    // (top_k, top_p, temperature): the models' own draw, the recorded set's
    // capacity exactly (the picks bound), the stand's sampling cell, no filter at
    // all, top-p alone, a pick count above the recorded set — the full-row branch
    // of the restore and the discharge — and a `top_k` above a narrow row's own
    // width, where `k_limit` stays 200 and the picks run out of candidates.
    let cfgs = [
        (50usize, 1.0f32, 0.9f32),
        (64, 1.0, 0.9),
        (50, 0.9, 0.9),
        (0, 1.0, 1.0),
        (0, 0.9, 1.0),
        (200, 0.5, 1.0),
        (200, 1.0, 0.9),
    ];
    // The boundary width, plus the widths the model's own row leaves unexecuted:
    // 2 047 leaves a ragged tail in the unrolled exponent sum, 1 000 a ragged last
    // lane, and 17 and 3 are narrower than the workgroup — the last with fewer
    // elements than a lane's candidate list holds.
    for width in [SAMPLE_STAGE as usize, 2047, 1000, 17, 3] {
        let rows: Vec<Vec<f32>> = (0..4).map(|kind| row(width, kind)).collect();
        for extra in [1usize, 64, width] {
            // `-inf`: probability zero, never scored, contributes exactly 0.0 to
            // the exponent sum — so the wider row's draw must be the narrower
            // one's. Every width past `SAMPLE_STAGE` is the cooperative path.
            let wide = SAMPLE_STAGE as usize + extra;
            for (top_k, top_p, temp) in cfgs {
                for seed in [12_345u64, 7, 0xDEAD_BEEF] {
                    let mut staged = sampler(width, top_k, top_p, temp, seed);
                    let mut coop = sampler(wide, top_k, top_p, temp, seed);
                    for (kind, logits) in rows.iter().enumerate() {
                        let mut padded = logits.clone();
                        padded.resize(wide, f32::NEG_INFINITY);
                        let a = draw(&mut staged, logits);
                        let b = draw(&mut coop, &padded);
                        assert_eq!(
                            a, b,
                            "width {width} padded to {wide} kind {kind} top_k {top_k} top_p \
                             {top_p} seed {seed}: the staged draw took {a}, the cooperative path \
                             {b}"
                        );
                        assert!(
                            (0..width as i32).contains(&a),
                            "drawn id {a} outside the row (kind {kind}, top_k {top_k}, seed \
                             {seed})"
                        );
                        // The draw is a function of the row and the seed, not of
                        // the slot it writes: a second call must agree.
                        assert_eq!(a, draw(&mut staged, logits), "kind {kind} top_k {top_k}");
                    }
                }
            }
        }
    }
}

/// What the draw costs, on both paths, at the boundary width: a printer, not a
/// gate — the ratio belongs to the device, and this crate asserts no timings.
/// `top_k` is the round count, so the arms separate the selection's fixed cost
/// from a round's. Re-take with
/// `cargo test --release -p rlx-wgpu --features backend-tests --test
/// sample_staged_paths -- --ignored --nocapture`.
#[test]
#[ignore = "measurement harness: prints the draw's cost on this device"]
fn the_staged_draw_cost_at_the_stage_boundary() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = common::route_lock();
    let width = SAMPLE_STAGE as usize;
    let logits = row(width, 0);
    let mut padded = logits.clone();
    padded.push(f32::NEG_INFINITY);
    let mut coop = sampler(width + 1, 50, 1.0, 0.9, 12_345);

    // Readback off for the measurement; the guard restores the operator's own value.
    let _only = RuntimeOverrides::install(Some(("RLX_BENCH_DISPATCH_ONLY", "1")));
    let _ = coop.run(&[("l", &padded)]);
    // Each round is drained before its clock stops: without the poll the numbers
    // would be the host's encode rate, not the draw.
    let per_draw = |exe: &mut WgpuExecutable, row: &[f32]| -> f64 {
        let mut rounds = Vec::with_capacity(9);
        for _ in 0..9 {
            let start = rlx_runtime::Tick::now();
            for _ in 0..20 {
                let _ = exe.run(&[("l", row)]);
            }
            if let Some(dev) = rlx_wgpu::device::wgpu_device() {
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
            rounds.push(rlx_runtime::Tick::now().elapsed_ns(start) as f64 / 20.0 / 1e3);
        }
        common::median(&rounds)
    };
    // `top_k` is the round count: 0 draws without the selection, 4 is one round of
    // the selection tree, 50 is the models' own draw (thirteen rounds). The
    // difference is what a round costs.
    let mut arms: Vec<(usize, WgpuExecutable)> = [0usize, 4, 50]
        .into_iter()
        .map(|top_k| (top_k, sampler(width, top_k, 1.0, 0.9, 12_345)))
        .collect();
    for (_, exe) in arms.iter_mut() {
        let _ = exe.run(&[("l", &logits)]);
    }
    // Round-robin, keeping each arm's own minimum: a clock that drifts over the
    // measurement then lands on every arm alike instead of on whichever ran last.
    let mut best = vec![f64::INFINITY; arms.len()];
    for _ in 0..3 {
        for (n, (_, exe)) in arms.iter_mut().enumerate() {
            best[n] = best[n].min(per_draw(exe, &logits));
        }
    }
    let coop_us = per_draw(&mut coop, &padded);
    println!(
        "sample one-row draw at {width}: staged {} us, cooperative {coop_us:.1} us ({:.2}x)",
        arms.iter()
            .zip(&best)
            .map(|((k, _), us)| format!("top_k {k} {us:.1}"))
            .collect::<Vec<_>>()
            .join(", "),
        coop_us / best[2]
    );
}
