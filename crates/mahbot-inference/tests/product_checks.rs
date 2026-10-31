// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The product checks that are not measurements: run by hand when needed, never
//! by the main stand run, never by the corpus run, never by the routine test
//! command, and with no place in any record file.
//!
//! Four of them, each `#[ignore]`d:
//!
//! * instrumental music yields no transcribed text;
//! * a paste onset at a seam lands within one detector hop, and whole-input
//!   conditioning is time-preserving;
//! * the pinned reference clip's transcript and digest, reported as a witness;
//! * the synthesis round trip — that the recogniser hears words in the rendered
//!   waveform.
//!
//! Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… MAHBOT_ASR_MUSIC=… \
//! cargo test -p mahbot-inference --release --test product_checks -- --ignored
//! --nocapture`. A test whose input is absent prints why and returns.

use std::sync::Arc;

use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::synth::Synthesis;
use mahbot_inference::transcribe::{self, Transcriber};
use mahbot_inference::{host, listen, music, reference, report, tokenizer::Tokenizer, vad};

mod support;

/// The pinned synthesis input: the long text and the seed the waveform pin was
/// taken with.
const SYNTH_SEED: u64 = 12_345;

/// The resident-set bound these checks open the product under: the default the
/// stand and the corpus run use, read through the guard's one parser.
fn ceiling_mib() -> u64 {
    mahbot_inference::guard::max_rss_mib().expect("the resident-set bound")
}

/// One role's pair files through the model directory, or the reason this check
/// cannot happen: a model that is absent is a skip; one that is present and does
/// not open is a failure, the same in every check here. `also` names the further
/// files the check needs beside the pair.
fn pair_files(
    role: Role,
    also: &[&str],
) -> Option<(std::path::PathBuf, &'static str, &'static str)> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return None;
    }
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(error) => {
            println!("skip: {error:#}");
            return None;
        }
    };
    let (body, sound) = loader::pair_files(role).expect("every role is pinned");
    for name in [body, sound].into_iter().chain(also.iter().copied()) {
        let path = dir.join(name);
        if !path.is_file() {
            println!("skip: {} is not on this machine", path.display());
            return None;
        }
    }
    Some((dir, body, sound))
}

/// One role's pair, opened.
fn pair(role: Role) -> Option<Arc<Pair>> {
    let (dir, body, sound) = pair_files(role, &[])?;
    Some(open_pair(&dir, body, sound))
}

/// The recognition pair and the tagger the transcription checks need.
fn open() -> Option<(Arc<Pair>, Tagger)> {
    let (dir, body, sound) = pair_files(Role::Recognition, &[CED_MODEL])?;
    let pair = open_pair(&dir, body, sound);
    let tagger = Tagger::open(&dir.join(CED_MODEL))
        .unwrap_or_else(|error| panic!("{}: {error:#}", dir.join(CED_MODEL).display()));
    Some((pair, tagger))
}

/// The reference voice, or the reason a check that needs it cannot happen.
fn voice() -> Option<std::path::PathBuf> {
    match loader::voice_wav() {
        Ok(voice) => Some(voice),
        Err(error) => {
            println!("skip: {error:#}");
            None
        }
    }
}

/// A pair that is on this machine and does not open is a failure, not a skip.
fn open_pair(dir: &std::path::Path, body: &str, sound: &str) -> Arc<Pair> {
    Arc::new(
        Pair::open(dir.join(body), dir.join(sound))
            .unwrap_or_else(|error| panic!("{}: {error:#}", dir.display())),
    )
}

/// **Instrumental music yields no transcribed text.** The music pool's split is
/// the truth: a transcript over any `instrumental/` clip is words invented over
/// music, and the guards never return a reading they refused.
#[test]
#[ignore = "needs the models and the pinned music pool; run by hand"]
fn instrumental_music_yields_no_transcribed_text() {
    let Some(paths) = support::pool() else {
        return;
    };
    let Some((pair, tagger)) = open() else {
        return;
    };
    let mut transcriber =
        Transcriber::open(pair, tagger, ceiling_mib()).expect("the transcriber opens");
    for (clip, path) in music::CLIPS.iter().zip(&paths) {
        if !clip.path.starts_with("instrumental/") {
            continue;
        }
        let out = transcriber
            .transcribe_file(path)
            .unwrap_or_else(|error| panic!("{}: {error:#}", clip.path));
        let text: String = out
            .regions
            .iter()
            .filter_map(|region| region.text.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            text.is_empty(),
            "{} is instrumental and came back with a transcript: {text:?}",
            clip.path
        );
        println!(
            "music {} — {} · regions={} text=\"\"",
            clip.path,
            clip.source,
            out.regions.len()
        );
    }
}

/// **A paste onset lands within one detector hop, and conditioning is
/// time-preserving.** The excerpt is the pinned clip as it is read, trimmed to its
/// own edges and cut to the detector's hop; four pastes into digital silence,
/// separated by gaps past the merge gap, are the construction, and their offsets
/// are the truth.
#[test]
#[ignore = "needs the models and the reference voice; run by hand"]
fn a_paste_onset_lands_within_one_detector_hop() {
    const TRIM_WINDOW: usize = 160;
    const TRIM_RMS: f32 = 3e-3;
    const PASTE_START: usize = 16_384;
    const PASTE_GAPS_S: [f64; 4] = [1.024, 2.048, 1.536, 3.072];

    let Some((pair, tagger)) = open() else {
        return;
    };
    let Some(voice) = voice() else {
        return;
    };
    let raw = match host::wav_mono_16k(&voice.display().to_string(), 0, listen::PASS_SAMPLES) {
        Ok(audio) => audio,
        Err(error) => {
            println!("skip: {error:#}");
            return;
        }
    };
    let mut transcriber =
        Transcriber::open(pair, tagger, ceiling_mib()).expect("the transcriber opens");

    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let edges: Vec<usize> = raw
        .chunks_exact(TRIM_WINDOW)
        .enumerate()
        .filter(|(_, window)| rms(window) > TRIM_RMS)
        .map(|(i, _)| i)
        .collect();
    let (first, last) = match (edges.first(), edges.last()) {
        (Some(first), Some(last)) => (*first, *last),
        _ => panic!("the pinned clip has no 10 ms window above {TRIM_RMS}: nothing to paste"),
    };
    let mut excerpt = raw[first * TRIM_WINDOW..(last + 1) * TRIM_WINDOW].to_vec();
    excerpt.truncate(excerpt.len() / vad::HOP_LENGTH * vad::HOP_LENGTH);
    let gaps: Vec<usize> = PASTE_GAPS_S
        .iter()
        .map(|seconds| (seconds * f64::from(transcribe::RATE)).round() as usize)
        .collect();
    let mut signal = vec![0f32; PASTE_START + 4 * excerpt.len() + gaps.iter().sum::<usize>()];
    let mut truth: Vec<usize> = Vec::with_capacity(4);
    let mut at = PASTE_START;
    for gap in &gaps {
        truth.push(at);
        signal[at..at + excerpt.len()].copy_from_slice(&excerpt);
        at += excerpt.len() + gap;
    }
    let pasted = transcriber
        .transcribe(&signal, transcribe::RATE)
        .expect("the constructed material");

    let hop_ms = 1000.0 * vad::HOP_LENGTH as f64 / f64::from(transcribe::RATE);
    let ms_per_sample = 1000.0 / f64::from(transcribe::RATE);
    let errors: Vec<f64> = truth
        .iter()
        .map(|&onset| {
            let nearest = pasted
                .regions
                .iter()
                .map(|region| region.start as i64)
                .min_by_key(|start| (start - onset as i64).abs())
                .unwrap_or(0);
            (nearest - onset as i64) as f64 * ms_per_sample
        })
        .collect();
    let max_abs = errors
        .iter()
        .fold(0f64, |most, error| most.max(error.abs()));
    println!(
        "boundary pastes={} regions={} err_ms={} max_abs_ms={max_abs:.1}",
        truth.len(),
        pasted.regions.len(),
        errors
            .iter()
            .map(|error| format!("{error:+.1}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    assert!(
        max_abs <= hop_ms,
        "a paste onset is {max_abs:.1} ms from its own offset, past the detector's one {hop_ms:.0} ms hop"
    );
    let read = pasted
        .regions
        .iter()
        .filter(|region| region.text.is_some())
        .count();
    assert!(
        read >= truth.len(),
        "{read} of the {} pasted excerpts came back with a transcript",
        truth.len()
    );

    // Conditioning is one gain over the whole input and may not move a boundary:
    // the same signal at a level the library leaves alone and at the quiet level it
    // lifts must tile identically.
    let amplitude = signal
        .iter()
        .fold(0f32, |most, value| most.max(value.abs()));
    let untouched = (host::SPEECH_FLOOR + host::SPEECH_TARGET) / 2.0;
    let steady_copy: Vec<f32> = signal
        .iter()
        .map(|value| value * (untouched / amplitude))
        .collect();
    let steady = transcriber
        .transcribe(&steady_copy, transcribe::RATE)
        .expect("the steady copy");
    let bounds = |out: &transcribe::Transcription| {
        out.regions
            .iter()
            .flat_map(|region| [region.start, region.end])
            .collect::<Vec<usize>>()
    };
    let (pasted_bounds, steady_bounds) = (bounds(&pasted), bounds(&steady));
    let moved = pasted_bounds
        .iter()
        .zip(&steady_bounds)
        .filter(|(pasted, steady)| pasted != steady)
        .count()
        + pasted_bounds.len().abs_diff(steady_bounds.len());
    println!(
        "boundary gain={:.4} lifted_gain={:.4} moved={moved}",
        steady.gain, pasted.gain,
    );
    assert_ne!(
        steady.gain, pasted.gain,
        "both copies conditioned at the same gain, so no level was compared"
    );
    assert_eq!(moved, 0, "a gain is time-preserving and moved a boundary");
}

/// **The pinned reference clip's transcript and digest, reported as a witness.**
/// The rule is the sentence, not the hash: a moved digest is printed and the check
/// carries on, as long as the words still read as the pin. The witness pins one
/// clip; a voice that is not it has nothing to be compared with, so the check says
/// why it did not run rather than walking a different input.
#[test]
#[ignore = "needs the models and the reference voice; run by hand"]
fn the_pinned_clip_reads_as_its_witness() {
    if !reference::voice_is_pinned() {
        println!(
            "skip: {} does not name the voice the witness was taken with",
            loader::VOICE_WAV_ENV
        );
        return;
    }
    let Some(asr) = pair(Role::Recognition) else {
        return;
    };
    let Some(voice) = voice() else {
        return;
    };
    let answer = support::listen_answer(&asr, &voice);
    assert!(
        support::witness_listen(&answer),
        "the pinned clip no longer reads as its witness: {:?}",
        answer.text
    );
}

/// **The synthesis round trip:** the rendered waveform, resampled 24 kHz → 16 kHz
/// and read back through the recogniser on its raw samples, still says the pinned
/// words. The main stand run neither performs this nor fails on it.
#[test]
#[ignore = "needs both model pairs and the reference voice; run by hand"]
fn the_recogniser_hears_words_in_the_rendered_waveform() {
    let Some(voice) = voice() else {
        return;
    };
    // The waveform's pin was taken with that voice: another one renders different
    // words, which the pin cannot judge, so the check says why it did not run
    // instead of failing on the substitution.
    if !reference::voice_is_pinned() {
        println!(
            "skip: {} does not name the voice the round-trip pin was taken with",
            loader::VOICE_WAV_ENV
        );
        return;
    }
    let Some(synthesis) = pair(Role::Synthesis) else {
        return;
    };
    let mut engine = Synthesis::new(synthesis, &voice).expect("the synthesis path");
    let wave = engine
        .synthesize(support::TTS_TEXT, SYNTH_SEED)
        .expect("the synthesis call");
    assert_eq!(report::non_finite(&wave), 0, "the waveform is finite");
    assert!(
        report::max_abs(&wave) <= 1.0,
        "the waveform leaves the PCM range"
    );
    // The round trip is a second pair: give the synthesis half's weights back
    // before the recogniser's are asked for.
    drop(engine);
    mahbot_inference::release_process_memory();
    let audio = host::resample(&wave, 24_000, listen::SAMPLE_RATE);
    let Some(asr) = pair(Role::Recognition) else {
        return;
    };
    let tokenizer = Tokenizer::from_model(&asr.body)
        .unwrap_or_else(|error| panic!("the recognition tokenizer: {error:#}"));
    let heard = listen::recognise(&asr, &tokenizer, audio)
        .unwrap_or_else(|error| panic!("the recogniser on the rendered waveform: {error:#}"))
        .text;
    let holds = reference::check_speak_roundtrip(&heard);
    assert!(holds, "the round trip is not the pinned transcript");
}
