// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and ignored: what two scoring windows cost through the ear
//! path — one batched pass against two one-window passes — and whether the pair's
//! embeddings are the single-window pass's.
//!
//! Needs the recognition pair and the pinned corpus recording; a machine without
//! them skips. Run with `cargo test -p mahbot-inference --release --test
//! wake_pair_cost -- --ignored --nocapture`, after `just throttle`.
//!
//! Both arms run the product's own front ends ([`Listener::embed_pair`] and
//! [`Listener::embed`]) over the same two consecutive windows of the pinned corpus
//! recording — the windows the detector pairs, one scoring stride apart — so the
//! comparison prices the batching and not an arrangement of the arithmetic. Rounds alternate
//! which arm runs first; the medians, the median of the paired differences and the
//! rounds the pair won are what the run prints.
//!
//! The parity line is the other half: each window's embedding from the pair pass
//! against the same window's from a lone pass, under the rule the grouped path is
//! already held to (`a_grouped_pass_matches_the_single_window_pass` in `listen`,
//! cosine > 0.9999). Two windows share the pass's weights, never their rows.
//!
//! The absolute figures move between runs; the paired difference and the rounds
//! that agree on it are the reading. Nothing here asserts a time.
use std::sync::Arc;

use mahbot_inference::host::wav_mono_16k;
use mahbot_inference::listen::{self, Listener};
use mahbot_inference::loader::{Pair, Role};
use mahbot_inference::stand::WAKE_OFFSET;
use mahbot_inference::wake::cosine;
use rlx_ir::Tick;

mod support;

use support::median;

/// Rounds each arm is timed over, alternating the order within a round.
const ROUNDS: usize = 9;

/// The grouped path's parity rule, restated here so this file states what it holds
/// the pair to rather than borrowing a number by accident.
const PARITY: f32 = 0.999_9;

#[test]
#[ignore = "measurement: prices the two-window pass against two one-window passes"]
fn a_pair_pass_against_two_single_window_passes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(recording) = support::recording(mahbot_inference::stand::RECOGNITION_RECORDING) else {
        eprintln!("skipped: the pinned corpus is not installed — the pair pass was not priced");
        return;
    };
    let read = |offset: usize| {
        wav_mono_16k(
            &recording.display().to_string(),
            offset,
            listen::WINDOW_SAMPLES,
        )
        .expect("the pinned corpus recording")
    };
    let first = read(WAKE_OFFSET);
    let second = read(WAKE_OFFSET + listen::SCORE_STRIDE_SAMPLES);
    assert_eq!(first.len(), listen::WINDOW_SAMPLES);
    assert_eq!(second.len(), listen::WINDOW_SAMPLES);

    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => Arc::new(pair),
        Err(error) => {
            eprintln!("skipped: {error:#} — the pair pass was not priced");
            return;
        }
    };
    let mut listener = Listener::new(pair);

    // The parity: each slot of the pair pass against that window's own pass.
    let [paired_a, paired_b] = listener.embed_pair(&first, &second).expect("the pair pass");
    let lone_a = listener.embed(&first).expect("a lone pass");
    let lone_b = listener.embed(&second).expect("a lone pass");
    for (slot, (paired, lone)) in [(paired_a, lone_a), (paired_b, lone_b)]
        .into_iter()
        .enumerate()
    {
        let cosine = cosine(&paired, &lone);
        let delta = paired
            .iter()
            .zip(&lone)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "slot {slot}: pair pass vs lone pass cosine {cosine:.7}, max|Δ| {delta:e} over {} \
             dimensions",
            paired.len()
        );
        assert!(
            cosine > PARITY,
            "the pair pass's slot {slot} matches a lone pass: cosine {cosine}"
        );
    }

    let mut pair_ms = Vec::with_capacity(ROUNDS);
    let mut serial_ms = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let one_pair = |listener: &mut Listener| {
            let t = Tick::now();
            let _ = listener.embed_pair(&first, &second).expect("the pair pass");
            Tick::now().elapsed_ns(t) as f64 / 1e6
        };
        let two_passes = |listener: &mut Listener| {
            let t = Tick::now();
            let _ = listener.embed(&first).expect("a lone pass");
            let _ = listener.embed(&second).expect("a lone pass");
            Tick::now().elapsed_ns(t) as f64 / 1e6
        };
        if round % 2 == 0 {
            pair_ms.push(one_pair(&mut listener));
            serial_ms.push(two_passes(&mut listener));
        } else {
            serial_ms.push(two_passes(&mut listener));
            pair_ms.push(one_pair(&mut listener));
        }
    }

    let (pair, serial) = (median(&pair_ms), median(&serial_ms));
    let deltas: Vec<f64> = pair_ms
        .iter()
        .zip(&serial_ms)
        .map(|(pair, serial)| pair - serial)
        .collect();
    let won = deltas.iter().filter(|delta| **delta < 0.0).count();
    let windows = listen::PAIR as f64;
    let paired = median(&deltas);
    eprintln!(
        "pair pass {pair:.3} ms ({:.3} ms per window) · two passes {serial:.3} ms ({:.3} ms per \
         window)",
        pair / windows,
        serial / windows,
    );
    eprintln!(
        "median paired difference {paired:+.3} ms · per window {:+.3} ms · the pair won {won} of \
         {ROUNDS} rounds · per-window ratio {:.3}",
        paired / windows,
        pair / serial,
    );
}
