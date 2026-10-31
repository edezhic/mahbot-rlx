// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measurement only and ignored: the audio encoder's three attention input
//! projections, as three separate packed passes against one wide pass over the same
//! bytes concatenated (`graphs::block`'s `qkv_fused`).
//!
//! Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… RLX_ALLOW_THROTTLE=1 cargo test
//! -p mahbot-inference --release --test encoder_qkv -- --ignored --nocapture <subject>`.
//! One subject per process (`wake_window`, `tail_cells`): the device keeps what a subject
//! built, and each holds two spellings of the encoder's weights.
//!
//! `wake_window` is the product's own cell — the whole encoder over the wake row's
//! 76-frame window — and `tail_cells` the tail at the row counts the other paths run it
//! at: the recognition ladder's two ends (128 frames is 16 rows, 768 is 96) and the bench's
//! grouped pass (sixteen windows, 160 rows). Each cell is two arms of one graph fed the
//! same input, reading the same weight bytes — the wide one as the three tensors
//! concatenated along their output rows at load time — so the pair prices the arrangement
//! and not the arithmetic. The rounds alternate which arm runs first; the medians, the
//! median of the paired differences and the rounds the wide arm won are what a cell prints.
//!
//! **The three-pass arm is also the fallback**, the shape an unfusable file takes (its
//! q/k/v in different schemes). Each cell therefore prints that path's correctness too: the
//! cosine and the largest element difference of the arms' `[rows, dim]` outputs, and for
//! the product cell of the vector the wake decision actually scores — the encoder's rows
//! through the product's own projector (`graphs::projector_chain`), pooled and normalised by
//! the two helpers `Listener::embed` ends on. The arms' arithmetic is element-for-element
//! the same, so a difference in their outputs is a kernel reading the wider shape differently,
//! not a different model: an arm that parts company here is the reading to chase, not a
//! number to accept. Outputs of different shapes are refused rather than compared, because
//! `cosine` and `largest` both stop at the shorter one and a mismatch would print as a
//! perfect match.
//!
//! That arm is **spelled out** here rather than taken from a builder: `three_pass` and
//! `three_pass_tail` restate `graphs::asr_encoder` / `graphs::asr_encoder_tail`'s chain —
//! the conv front, the tail, the input shape — with `qkv_fused` off. Nothing else would
//! notice the pair drifting apart, so every cell asserts that the two schedules differ in
//! nothing but the projections: `2 · blocks` more packed multiplies on the three-pass arm,
//! `3 · blocks` more narrows on the fused one, and every other step kind identical.
//!
//! Each subject also asserts what the timings cannot show: the shipped arm at
//! `support::PER_BLOCK_FUSED · blocks` packed multiplies — the fused projection, `attn_out`,
//! `ffn_up`, `ffn_down` — and the fallback arm at `support::PER_BLOCK_SPLIT · blocks`, since
//! a silent fallback would otherwise leave no trace beyond a few milliseconds. The same route
//! is a bare `cargo test` gate in `tests/encoder_route.rs`, which needs neither a second arm
//! nor this harness's numbers. Nothing on the pinned `mmproj` refuses: its three Q8_0
//! `[1024, 1024]` tensors share a scheme and in-dim, which that count asserts, so an
//! unfusable file is the fallback arm's shape; the same decision answers false on a real
//! mixed-scheme file on every build of the body, whose `attn_v` is Q6_K.
use std::collections::BTreeMap;
use std::sync::Arc;

use mahbot_inference::emit::LoaderWeights;
use mahbot_inference::graphs::{self, Compiled, EncoderConvs, EncoderTail};
use mahbot_inference::host;
use mahbot_inference::listen::{GROUP, MEL_FRAMES, WINDOW_SAMPLES, l2, mean_pool};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::stand::WAKE_OFFSET;
use mahbot_inference::wake::cosine;
use rlx_ir::{DType, Shape, Tick};
use rlx_runtime::Device;

mod support;

use support::{
    PER_BLOCK_FUSED, PER_BLOCK_SPLIT, median, packed_multiplies, ramped_row, step_census,
};

/// Rounds one arm is timed over; each round times both arms, alternating the order.
const ROUNDS: usize = 9;

/// One timed pass of a compiled graph.
fn run_ms(graph: &mut Compiled, inputs: &[(&str, &[f32])]) -> f64 {
    let t = Tick::now();
    graph.run_slices(inputs);
    Tick::now().elapsed_ns(t) as f64 / 1e6
}

/// The one pass a graph produces, which every caller here feeds whole.
fn pass(graph: &mut Compiled, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut outs = graph.run_slices(inputs);
    outs.remove(0)
}

/// The two arms' cosine and largest element difference, refusing outputs of different
/// shapes: [`cosine`] and [`largest`] both stop at the shorter one, so a length mismatch
/// would print as a perfect match instead of the drift this harness exists to catch.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "the arms must produce the same output shape; a different one is not the same work"
    );
    (cosine(a, b), largest(a, b))
}

/// The largest element difference between two outputs of one shape.
fn largest(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The packed multiplies one arm dispatches, asserted against the route it is supposed to
/// take: [`PER_BLOCK_FUSED`] when q/k/v ride one blob, [`PER_BLOCK_SPLIT`] when they ride
/// three. This is the witness the milliseconds are not — a fallback would otherwise cost
/// only time. A backend that exposes no schedule prints that and asserts nothing.
fn assert_route(graph: &Compiled, blocks: usize, fused: bool) {
    let Some(names) = graph.schedule_step_names() else {
        eprintln!("  census unavailable: this backend exposes no schedule");
        return;
    };
    let per_block = if fused {
        PER_BLOCK_FUSED
    } else {
        PER_BLOCK_SPLIT
    };
    println!(
        "  {} · steps {} · census {}",
        if fused {
            "one fused pass"
        } else {
            "three separate passes"
        },
        names.len(),
        step_census(&names),
    );
    assert_eq!(
        packed_multiplies(&names),
        blocks * per_block,
        "a block dispatches {PER_BLOCK_FUSED} packed multiplies when q/k/v ride one blob (the \
         fused projection, attn_out, ffn_up, ffn_down) and {PER_BLOCK_SPLIT} when they ride \
         three; a moved count means the route moved with it"
    );
}

/// The arms must differ in nothing but the projections: `2 · blocks` more packed multiplies
/// and `3 · blocks` more narrows on the fused side — the split back into q/k/v, which the
/// three-pass form never pays — and every other step kind identical. A step kind that moved
/// any other way means `three_pass`/`three_pass_tail` no longer restate the builder they
/// mirror, so the pair has stopped pricing one change — the drift the header warns of.
fn assert_only_the_projections(wide: &Compiled, split: &Compiled, blocks: usize) {
    let kinds = |graph: &Compiled| -> Option<BTreeMap<&'static str, usize>> {
        let mut kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
        for name in graph.schedule_step_names()? {
            *kinds.entry(name).or_default() += 1;
        }
        Some(kinds)
    };
    let (Some(mut wide), Some(mut split)) = (kinds(wide), kinds(split)) else {
        eprintln!("  schedule comparison unavailable: this backend exposes no schedule");
        return;
    };
    let packed = wide.remove("dequant_matmul_gguf").unwrap_or(0);
    let packed_split = split.remove("dequant_matmul_gguf").unwrap_or(0);
    let narrows = wide.remove("narrow").unwrap_or(0);
    let narrows_split = split.remove("narrow").unwrap_or(0);
    assert_eq!(
        packed_split,
        packed + 2 * blocks,
        "two extra packed multiplies per block is the whole difference between the forms"
    );
    assert_eq!(
        narrows,
        narrows_split + 3 * blocks,
        "the fused projection is split into q/k/v by three narrows per block, and the \
         three-pass form pays none"
    );
    assert_eq!(
        wide, split,
        "every other step kind must be identical in both arms: a moved one means the \
         fallback arm no longer mirrors the builder it restates"
    );
}

/// The encoder the product builds: `qkv_fused` comes from the weight source itself
/// ([`graphs::asr_encoder`]), so the shipped arm is that builder and not a copy of it.
fn shipped(pattern: &Arc<Pair>, frames: usize) -> anyhow::Result<Compiled> {
    Compiled::new(
        Device::Gpu,
        graphs::asr_encoder(pattern.sound.clone(), frames)?,
    )
}

/// The same encoder with `qkv_fused` off — the shape a file whose three projections
/// cannot be concatenated takes.
fn three_pass(pattern: &Arc<Pair>, frames: usize) -> anyhow::Result<Compiled> {
    let sound = pattern.sound.clone();
    let c = sound.clip().clone();
    let mut weights = LoaderWeights::new(sound.clone());
    let flow = rlx_flow::ModelFlow::new(format!("encoder_{frames}_three_pass"))
        .with_profile(graphs::profile())
        .input("mel", Shape::new(&[1, 1, c.mel_bins, frames], DType::F32))
        .layer_stage(EncoderConvs {
            model: sound.clone(),
            frames,
        })
        .layer_stage(EncoderTail {
            model: sound,
            frames,
            batch: 1,
            qkv_fused: false,
        })
        .output("tokens");
    Compiled::new(Device::Gpu, flow.build(&mut weights)?)
}

/// The encoder tail over `batch` windows of `frames`: the batching cell's one weight
/// read, `qkv_fused` off.
fn three_pass_tail(
    model: Arc<loader::Model>,
    frames: usize,
    batch: usize,
) -> anyhow::Result<Compiled> {
    let c = model.clip().clone();
    let mut weights = LoaderWeights::new(model.clone());
    let flow = rlx_flow::ModelFlow::new(format!("tail_{frames}x{batch}_three_pass"))
        .with_profile(graphs::profile())
        .input(
            "flat",
            Shape::new(
                &[
                    batch * graphs::asr_tokens(frames),
                    graphs::flat_width(c.mel_bins),
                ],
                DType::F32,
            ),
        )
        .layer_stage(EncoderTail {
            model,
            frames,
            batch,
            qkv_fused: false,
        })
        .output("tokens");
    Compiled::new(Device::Gpu, flow.build(&mut weights)?)
}

/// One tail cell: `frames` and `batch` — the product's own graph against the same graph
/// with `qkv_fused` off, over one deterministic input of the cell's own width. `rows` is
/// what the two arms share, so the pair prices the arrangement on identical numbers.
fn tail_cell(pair: &Arc<Pair>, frames: usize, batch: usize) {
    let rows_count = batch * graphs::asr_tokens(frames);
    let flat = graphs::flat_width(pair.sound.clip().mel_bins);
    let rows = ramped_row(rows_count * flat);
    let (mut wide, mut split) = (
        Compiled::new(
            Device::Gpu,
            graphs::asr_encoder_tail(pair.sound.clone(), frames, batch).expect("the shipped tail"),
        )
        .expect("compile the shipped tail"),
        three_pass_tail(pair.sound.clone(), frames, batch).expect("the three-pass tail"),
    );
    assert_route(&wide, pair.sound.clip().layers, true);
    assert_route(&split, pair.sound.clip().layers, false);
    assert_only_the_projections(&wide, &split, pair.sound.clip().layers);
    let input: [(&str, &[f32]); 1] = [("flat", &rows)];
    let (a, b) = time_arms(
        &format!("tail {frames} frames × batch {batch} · {rows_count} rows"),
        &mut wide,
        &mut split,
        &input,
    );
    let (cos, diff) = compare(&a, &b);
    println!("  encoder output · cosine {cos:.9} · largest difference {diff:.3e}");
}

/// Time both arms over [`ROUNDS`] paired rounds, alternating which runs first, and
/// print the medians, the median paired difference (wide − three passes) and the rounds
/// the wide arm won. The two arms' outputs are the warm-up pass each, which is the same
/// reading every round of a deterministic graph produces.
fn time_arms(
    label: &str,
    wide: &mut Compiled,
    split: &mut Compiled,
    inputs: &[(&str, &[f32])],
) -> (Vec<f32>, Vec<f32>) {
    // One warm-up pass each: both graphs are already compiled here, so this only keeps
    // the first (lazily bound) run out of the medians.
    let a_out = pass(wide, inputs);
    let b_out = pass(split, inputs);
    let (mut wide_ms, mut split_ms, mut deltas) =
        (Vec::new(), Vec::new(), Vec::with_capacity(ROUNDS));
    let mut wins = 0;
    for round in 0..ROUNDS {
        let (a, b) = if round % 2 == 0 {
            let a = run_ms(wide, inputs);
            let b = run_ms(split, inputs);
            (a, b)
        } else {
            let b = run_ms(split, inputs);
            let a = run_ms(wide, inputs);
            (a, b)
        };
        wide_ms.push(a);
        split_ms.push(b);
        deltas.push(a - b);
        wins += (a < b) as usize;
    }
    println!(
        "{label}: three passes median {:.3} ms · one wide pass median {:.3} ms · \
         paired wide − three {:.3} ms · the wide arm won {wins}/{ROUNDS}",
        median(&split_ms),
        median(&wide_ms),
        median(&deltas),
    );
    (a_out, b_out)
}

/// The pinned recording's window the wake row embeds, as the product reads it.
fn window_slice() -> Option<Vec<f32>> {
    let path = loader::voice_wav().ok();
    let samples = path.and_then(|p| {
        host::wav_mono_16k(&p.display().to_string(), WAKE_OFFSET, WINDOW_SAMPLES).ok()
    });
    match samples {
        Some(s) if s.len() == WINDOW_SAMPLES => Some(s),
        _ => {
            eprintln!("skipped: no wake window under the voice recording — nothing to measure");
            None
        }
    }
}

fn pair_of_role() -> Option<Arc<Pair>> {
    match Pair::of_role(Role::Recognition) {
        Ok(pair) => Some(Arc::new(pair)),
        Err(e) => {
            eprintln!("skipped: {} — nothing to measure", e);
            None
        }
    }
}

#[test]
#[ignore = "measurement: one process, two spellings of the encoder's weights"]
fn wake_window() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(audio) = window_slice() else { return };
    let Some(pair) = pair_of_role() else { return };

    let bins = pair.sound.clip().mel_bins;
    let mut mel = Compiled::new(
        Device::Gpu,
        graphs::mel_frontend(MEL_FRAMES, bins).expect("the mel front end"),
    )
    .expect("compile the mel front end");
    let mel = pass(&mut mel, &[("audio", &audio)]);

    let (mut wide, mut split) = (
        shipped(&pair, MEL_FRAMES).expect("the shipped encoder"),
        three_pass(&pair, MEL_FRAMES).expect("the three-pass encoder"),
    );
    assert_route(&wide, pair.sound.clip().layers, true);
    assert_route(&split, pair.sound.clip().layers, false);
    assert_only_the_projections(&wide, &split, pair.sound.clip().layers);
    let input: [(&str, &[f32]); 1] = [("mel", &mel)];
    let (a, b) = time_arms(
        "wake window · 76 frames, 10 tokens",
        &mut wide,
        &mut split,
        &input,
    );
    let (cos, diff) = compare(&a, &b);
    println!("  encoder output · cosine {cos:.9} · largest difference {diff:.3e}");

    // What the decision scores: the encoder's rows through the product's projector
    // ([`graphs::projector_chain`]), pooled and normalised by the same two helpers
    // `Listener::embed` ends on.
    let tokens = graphs::asr_tokens(MEL_FRAMES);
    let dim = pair.sound.clip().projection_dim;
    let mut projector = Compiled::new(
        Device::Gpu,
        graphs::projector_chain(pair.clone(), tokens).expect("the projector"),
    )
    .expect("compile the projector");
    let embedding = |rows: &[f32], projector: &mut Compiled| -> Vec<f32> {
        let projected = pass(projector, &[("features", rows)]);
        l2(&mean_pool(&projected, tokens, dim).expect("the projector's rows pool"))
    };
    let (ea, eb) = (embedding(&a, &mut projector), embedding(&b, &mut projector));
    let (cos, diff) = compare(&ea, &eb);
    println!(
        "  window embedding · {tokens} rows pooled to {} dims · cosine {cos:.9} · largest \
         difference {diff:.3e}",
        ea.len(),
    );
}

#[test]
#[ignore = "measurement: one process, two spellings of the encoder tail's weights"]
fn tail_cells() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair_of_role() else { return };
    // The recognition ladder's two ends — 128 frames is 16 rows, 768 is 96 — where the
    // multiplies are fed by the row count alone, and the bench's grouped cell, where
    // one pass carries sixteen windows.
    for (frames, batch) in [(128, 1), (768, 1), (MEL_FRAMES, GROUP)] {
        tail_cell(&pair, frames, batch);
    }
}
