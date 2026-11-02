// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and ignored: how much of a text the crate actually says.
//!
//! The stand's synthesis rows are timed, not listened to. This walks the product's
//! synthesis call over texts of three lengths and reads every waveform back through
//! the product's transcription call, region by region, so what the render says can
//! be compared with what was asked for. Nothing here compares a hash: the reading is
//! the words, and the last region is where a cut render shows itself. What the round
//! trip's rules make of each reading is printed, not asserted — the assertion is the
//! product check's, in `product_checks`. A text the talker's context cannot hold is
//! checked too — the one request that is refused rather than rendered.
//!
//! Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… cargo test -p mahbot-inference
//! --release --test speak_readback -- --ignored --nocapture --test-threads 1`.
//! Each subject holds a synthesis pair and then a recognition pair with the tagger,
//! releasing the first before opening the second. The doubled render alone peaks at
//! about 5.5 GiB of physical footprint and the three in one process peak higher than
//! one subject's own, so name a single subject when the ceiling is tight.

mod support;

use std::sync::Arc;

use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::stand::{SHORT_TEXT, SYNTH_SEED};
use mahbot_inference::synth::{SAMPLE_RATE, Synthesis};
use mahbot_inference::transcribe::Transcriber;
use mahbot_inference::{reference, report};

/// Twice [`support::TTS_TEXT`] by word count and not by repeating it: the second
/// half is another passage of the same chapter, so a render that says it whole has
/// carried four hundred words and not one paragraph twice. The audio's tail must be
/// this passage's own last words — the four the round trip demands, "here they all
/// unite".
const DOUBLED_TAIL: &str = "Circumambulate the city of a dreamy Sabbath afternoon. Go from \
    Corlears Hook to Coenties Slip, and from thence, by Whitehall, northward. What do you see? \
    Posted like silent sentinels all around the town, stand thousands upon thousands of mortal \
    men fixed in ocean reveries. Some leaning against the spiles; some seated upon the \
    pier-heads; some looking over the bulwarks of ships from China; some high aloft in the \
    rigging, as if striving to get a little nearer to the heavens. But these are all landsmen; \
    of week days pent up in lath and plaster, tied to counters, nailed to benches, clinched to \
    desks. How then is this? Are the green fields gone? What do they here? But look! here come \
    more crowds, pacing straight for the water, and seemingly bound for a dive. Strange! Nothing \
    will content them but the extremest limit of the land; loitering under the shady lee of \
    yonder warehouses will not suffice. They must get just as nigh the water as they possibly \
    can without falling in. And there they stand, miles of them, leagues. Inlanders all, they \
    come from lanes and alleys, streets and avenues, north, east, south, and west. Yet here they \
    all unite.";

/// The synthesis pair and the reference voice, or `None` with the reason printed.
fn synthesis_pair() -> Option<(Arc<Pair>, std::path::PathBuf)> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return None;
    }
    let voice = match loader::voice_wav() {
        Ok(voice) => voice,
        Err(error) => {
            println!("skip: {error}");
            return None;
        }
    };
    match Pair::of_role(Role::Synthesis) {
        Ok(pair) => Some((Arc::new(pair), voice)),
        Err(error) => {
            println!("skip: {error}");
            None
        }
    }
}

/// One text through the product call, then its audio read back: what the render
/// said, in the recogniser's words.
fn speak_and_read_back(text: &str) {
    let Some((pair, voice)) = synthesis_pair() else {
        return;
    };
    let mut engine = Synthesis::new(pair, &voice).expect("the synthesis path");
    let memory = report::Memory::take();
    let t = report::Stopwatch::start();
    let wave = engine
        .synthesize(text, SYNTH_SEED)
        .expect("the render reached end-of-speech");
    let ms = t.ms();
    let build_ms = engine.build_ms();
    let memory = memory.finish();
    let frames = wave.len() / mahbot_inference::graphs::samples_per_frame();
    println!(
        "speak words={} frames={frames} seconds={:.2} ms={ms:.0} ms_per_frame={:.1} \
         build_ms={build_ms:.0}",
        text.split_whitespace().count(),
        wave.len() as f64 / f64::from(SAMPLE_RATE),
        ms / frames as f64,
    );
    assert_eq!(
        report::non_finite(&wave),
        0,
        "the waveform has non-finite values"
    );
    assert!(
        report::max_abs(&wave) <= 1.0,
        "the waveform leaves the PCM range: max|y|={}",
        report::sci(report::max_abs(&wave))
    );
    println!("  synthesis footprint {}", memory.peak_text());
    // The read-back is a second pair: give the synthesis half's weights back before
    // the recogniser's are asked for.
    drop(engine);
    mahbot_inference::release_process_memory();
    let mut transcriber = match Transcriber::open_product(support::ceiling_mib()) {
        Ok(transcriber) => transcriber,
        Err(error) => {
            println!("skip: the read-back needs the recognition pair and its tagger: {error:#}");
            return;
        }
    };
    let out = transcriber
        .transcribe(&wave, SAMPLE_RATE)
        .expect("the read-back");
    println!("  regions={}", out.regions.len());
    for region in &out.regions {
        println!(
            "  [{:6.2}..{:6.2}] read={} label={:?} text={:?}",
            region.start as f64 / f64::from(SAMPLE_RATE),
            region.end as f64 / f64::from(SAMPLE_RATE),
            region.read,
            region.label,
            region.text
        );
    }
    let heard: String = out
        .regions
        .iter()
        .filter_map(|region| region.text.as_deref())
        .collect::<Vec<_>>()
        .join(" ");
    // What the rules make of this render is printed, not asserted: the assertion is the
    // product check's, and one rule stated in two places drifts apart.
    println!("  round trip: {}", reference::round_trip(&heard, text));
}

/// The stand's short request, read back.
#[test]
#[ignore = "needs both model pairs and the reference voice; run by hand"]
fn the_short_request_read_back() {
    speak_and_read_back(SHORT_TEXT);
}

/// The pinned long paragraph, read back whole.
#[test]
#[ignore = "needs both model pairs and the reference voice; run by hand"]
fn the_long_paragraph_read_back() {
    speak_and_read_back(support::TTS_TEXT);
}

/// Twice that length, likewise read back whole.
#[test]
#[ignore = "needs both model pairs and the reference voice; run by hand"]
fn the_doubled_text_read_back() {
    speak_and_read_back(&format!("{} {DOUBLED_TAIL}", support::TTS_TEXT));
}

/// A text the talker's context cannot hold is refused before this request's own
/// graphs exist, and the refusal names what it ran out of: no waveform comes back as
/// though it were the whole text.
#[test]
#[ignore = "needs the synthesis pair and the reference voice; run by hand"]
fn a_text_longer_than_the_context_is_refused() {
    let Some((pair, voice)) = synthesis_pair() else {
        return;
    };
    let mut engine = Synthesis::new(pair, &voice).expect("the synthesis path");
    // Twenty thousand repetitions are tens of thousands of tokens: past any talker's
    // context, and refused before the request's own graphs are built.
    let text = "the sea ".repeat(20_000);
    let error = engine
        .synthesize(&text, 1)
        .expect_err("a text longer than the context cannot be spoken whole");
    let message = format!("{error:#}");
    println!("refused: {message}");
    assert!(
        message.contains("context"),
        "the refusal does not name the context it ran out of: {message}"
    );
}
