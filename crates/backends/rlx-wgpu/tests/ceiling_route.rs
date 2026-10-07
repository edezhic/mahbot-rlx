// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The per-binding-ceiling rule: an op that hands the arena to its kernel by
//! ABSOLUTE offset must not run behind a re-based window.
//!
//! This is the same executed geometry the two measured failures came from — an
//! op whose operands and result all sit in one stripe, bound through a window
//! whose base is not zero, so nothing else catches it. Before the rule: the
//! last-axis prefix sum returned every output exactly `0.0` (262144/262144) and
//! the greedy argmax kernel behind `Op::Sample` returned `0` where the logits
//! held their maximum. `WgpuExecutable::ceiling_hosted_steps()` reports the
//! route, so a pass that never reached the striped geometry fails loudly here
//! rather than passing vacuously.
//!
//! Both cases need the same process-global switches, so they live in one test.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Small enough that a few megabyte-scale tensors cannot share one stripe.
const SHARD_MIB: usize = 4;
/// Without this the per-stripe usable region collapses to the 256-byte
/// alignment floor and every op straddles instead of sharing a stripe.
const STAGE_MIB: usize = 1;

#[test]
fn absolute_addressing_ops_over_a_striped_arena_are_hosted_and_correct() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: single-threaded by construction — one test per file. Both knobs only
    // clamp downward, so a device that cannot shard this small fails the hosted-steps
    // assertion below loudly instead of passing vacuously; `ALLOW_SHARD` is required
    // because striping is refused by default.
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_STAGE_MIB", STAGE_MIB.to_string());
        std::env::set_var("RLX_WGPU_ALLOW_SHARD", "1");
    }

    const COLS: usize = 256;
    const VOCAB: usize = 512;
    /// The argmax of the logits below, far from index 0 so a kernel that reads
    /// the wrong slot cannot land on it by accident.
    const PEAK: usize = 401;
    // 3 MiB: packed first (so it fills stripe 0) and live across both ops under
    // test (so the planner's peak arena exceeds one stripe).
    const F_ROWS: usize = 3 * 1024 * 1024 / 4 / COLS;
    const A_ROWS: usize = 1024 * 1024 / 4 / COLS;

    let mut g = Graph::new("ceiling");
    let f = g.param("f", Shape::new(&[F_ROWS, COLS], F));
    let s1 = g.sum(f, vec![1], false);
    let a = g.input("a", Shape::new(&[A_ROWS, COLS], F));
    let cs = g.cumsum(a, -1, false, Shape::new(&[A_ROWS, COLS], F));
    let logits = g.input("logits", Shape::new(&[1, VOCAB], F));
    // temperature 0 is the project's greedy rule: a deterministic argmax, ties
    // to the lowest index (`lower.rs` Op::Sample).
    let tok = g.sample(logits, 0, 1.0, 0.0, 0, Shape::new(&[1], F));
    let s2 = g.sum(f, vec![0], false);
    g.set_outputs(vec![cs, s1, s2, tok]);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("f", &vec![0.5f32; F_ROWS * COLS]);
    // Exactly representable operands: a running sum of integers below 2^24 is
    // exact, so the only slack needed is accumulation order.
    let av: Vec<f32> = (0..A_ROWS * COLS).map(|i| (i % 32) as f32).collect();
    let mut logits_v = vec![0.0f32; VOCAB];
    logits_v[PEAK] = 1000.0;
    let got = exe.run(&[("a", &av), ("logits", &logits_v)]);

    // The route must have changed for BOTH ops, or this test is not exercising
    // the geometry the failures came from.
    assert!(
        exe.ceiling_hosted_steps() >= 2,
        "the pinned cap did not stripe the arena, or the ceiling rule did not fire for both \
         ops ({} step(s) hosted) — this test would pass vacuously",
        exe.ceiling_hosted_steps()
    );

    let mut want = vec![0.0f32; A_ROWS * COLS];
    for r in 0..A_ROWS {
        let mut acc = 0.0f32;
        for c in 0..COLS {
            acc += av[r * COLS + c];
            want[r * COLS + c] = acc;
        }
    }
    let bad = got[0]
        .iter()
        .zip(&want)
        .filter(|(x, w)| (**x - **w).abs() > 1e-4 * (1.0 + w.abs()))
        .count();
    let zeros = got[0].iter().filter(|v| **v == 0.0).count();
    assert_eq!(
        bad,
        0,
        "{bad} of {} cumsum outputs wrong ({zeros} exactly zero) — absolute arena \
         offsets were read through a re-based window. Expected the running sum, \
         got e.g. {:?}",
        want.len(),
        &got[0][..got[0].len().min(4)]
    );

    // Judged against the maximum over the same logits, not against the engine's
    // own CPU sampling path.
    assert_eq!(
        got[3][0] as usize, PEAK,
        "greedy draw returned token {} but the logits peak at {PEAK} — the argmax \
         kernel read through a re-based window (a wrong window reads as 0, so the \
         draw lands on index 0)",
        got[3][0]
    );
}
