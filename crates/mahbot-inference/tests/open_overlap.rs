// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The record behind `overlap`'s module doc: the three opens a recognition call needs and the
//! two a synthesis call needs, serially and overlapped, with the two arms alternating and the
//! median of paired rounds taken.
//!
//! The page cache cannot be emptied from here — the instrument that does it is the crate's own
//! (`src/cold.rs`) — so both arms read warm files. That prices the tagger's open without the
//! disk; the saving is the pair-and-vocabulary chain, which is host work either way.
//!
//! Needs the installed static tree and a wgpu adapter; run by hand with
//! `cargo test --release -p mahbot-inference --features hand-run --test open_overlap -- \
//! --ignored --nocapture`. A missing input prints why and returns.

use std::sync::Arc;

use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::graphs::{self, Compiled, SampleSpec};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::synth::Synthesis;
use mahbot_inference::tokenizer::{Decoder, Encoder};
use mahbot_inference::transcribe::Transcriber;
use rlx_ir::Tick;

mod support;

/// Rounds of each arm: odd, so a median is one round's reading rather than an average of two.
const ROUNDS: usize = 9;

fn ms(start: Tick) -> f64 {
    Tick::now().elapsed_ns(start) as f64 / 1e6
}

/// The three opens in sequence: the pair, the vocabulary built from it, then the tagger.
fn recognition_serial(dir: &std::path::Path) -> (f64, f64, f64) {
    let t = Tick::now();
    let pair = Arc::new(Pair::of_role(Role::Recognition).expect("the recognition pair"));
    let pair_ms = ms(t);
    let t = Tick::now();
    let decoder = Decoder::from_model(&pair.body).expect("the vocabulary");
    let vocab_ms = ms(t);
    let t = Tick::now();
    let tagger = Tagger::open(&dir.join(CED_MODEL)).expect("the tagger");
    let tagger_ms = ms(t);
    drop((decoder, tagger, pair));
    (pair_ms, vocab_ms, tagger_ms)
}

/// The same three through the product's own entry point.
fn recognition_overlapped(ceiling_mib: u64) -> f64 {
    let t = Tick::now();
    let transcriber = Transcriber::open_product(ceiling_mib).expect("the product's open");
    let total = ms(t);
    drop(transcriber);
    total
}

/// The vocabulary, then the predictor frame's build and compile — the order the synthesis open
/// ran them in before the overlap.
fn synthesis_serial(pair: &Arc<Pair>) -> (f64, f64) {
    let t = Tick::now();
    let encoder = Encoder::from_model(&pair.body).expect("the vocabulary");
    let vocab_ms = ms(t);
    let steps = graphs::code_predictor_steps(&pair.sound).expect("the predictor's steps");
    let (top_k, top_p, temperature) = pair.body.sampling();
    let spec = SampleSpec {
        top_k,
        top_p,
        temperature,
        seed: 0,
    };
    let t = Tick::now();
    let built =
        graphs::tts_code_predictor_frame(pair.sound.clone(), steps + 1, spec).expect("the frame");
    let frame = Compiled::new(built).expect("the frame's compile");
    let frame_ms = ms(t);
    drop((frame, encoder));
    (vocab_ms, frame_ms)
}

fn synthesis_overlapped(pair: &Arc<Pair>) -> f64 {
    let t = Tick::now();
    let engine = Synthesis::new(Arc::clone(pair)).expect("the synthesis open");
    let total = ms(t);
    drop(engine);
    total
}

#[test]
#[ignore = "needs the models and a wgpu adapter; run by hand"]
fn the_opens_hide_under_the_tagger_and_the_frame() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    mahbot_inference::device::pin_gpu();
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(error) => {
            println!("skip: {error:#}");
            return;
        }
    };
    // Both arms need both pairs and the tagger's file; a tree missing one is a skip, not a panic.
    if let Err(error) = Pair::of_role(Role::Recognition) {
        println!("skip: {error:#}");
        return;
    }
    if !dir.join(CED_MODEL).is_file() {
        println!("skip: {} is not installed", dir.join(CED_MODEL).display());
        return;
    }
    let Ok(synth_pair) = Pair::of_role(Role::Synthesis) else {
        println!("skip: the model directory names no TTS pair");
        return;
    };
    let synth_pair = Arc::new(synth_pair);
    let ceiling_mib = support::ceiling_mib();

    // Warm the device and the allocator once: every arm afterwards pays its own work only.
    let _ = recognition_serial(&dir);
    let _ = recognition_overlapped(ceiling_mib);
    let _ = synthesis_serial(&synth_pair);
    let _ = synthesis_overlapped(&synth_pair);

    let (mut pair, mut vocab, mut tagger) = (Vec::new(), Vec::new(), Vec::new());
    let (mut serial, mut overlapped) = (Vec::new(), Vec::new());
    let (mut s_vocab, mut s_frame, mut s_overlapped) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        let arms = if round % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for serial_arm in arms {
            if serial_arm {
                let (p, v, t) = recognition_serial(&dir);
                pair.push(p);
                vocab.push(v);
                tagger.push(t);
                serial.push(p + v + t);
            } else {
                overlapped.push(recognition_overlapped(ceiling_mib));
            }
            mahbot_inference::release_process_memory();
        }
        for serial_arm in arms {
            if serial_arm {
                let (v, f) = synthesis_serial(&synth_pair);
                s_vocab.push(v);
                s_frame.push(f);
            } else {
                s_overlapped.push(synthesis_overlapped(&synth_pair));
            }
            mahbot_inference::release_process_memory();
        }
    }

    // Paired per round, so a machine that drifts over the run does not decide the figure.
    let recognition: Vec<f64> = serial
        .iter()
        .zip(&overlapped)
        .map(|(serial, overlapped)| serial - overlapped)
        .collect();
    let synthesis: Vec<f64> = s_vocab
        .iter()
        .zip(&s_frame)
        .zip(&s_overlapped)
        .map(|((vocab, frame), overlapped)| vocab + frame - overlapped)
        .collect();
    println!("rounds: serial {serial:?}");
    println!("rounds: overlapped {overlapped:?}");
    println!("rounds: synthesis overlapped {s_overlapped:?}");
    println!("rounds: recognition saving {recognition:?}");
    println!("rounds: synthesis saving {synthesis:?}");
    println!(
        "recognition: pair {:.1} · vocabulary {:.1} · tagger {:.1} ms · serial {:.1} · \
         overlapped {:.1} · saving {:.1} ms (paired)",
        support::median(&pair),
        support::median(&vocab),
        support::median(&tagger),
        support::median(&serial),
        support::median(&overlapped),
        support::median(&recognition)
    );
    println!(
        "synthesis: vocabulary {:.1} · frame {:.1} ms · serial {:.1} · overlapped {:.1} · \
         saving {:.1} ms (paired)",
        support::median(&s_vocab),
        support::median(&s_frame),
        support::median(&s_vocab) + support::median(&s_frame),
        support::median(&s_overlapped),
        support::median(&synthesis)
    );
}
