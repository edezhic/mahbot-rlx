// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The product checks that are not measurements: run by hand when needed, never
//! by the main stand run, never by the corpus run, never by the routine test
//! command, and with no place in any record file.
//!
//! Five of them, each `#[ignore]`d:
//!
//! * instrumental music yields no transcribed text;
//! * a paste onset at a seam lands within one detector hop, and whole-input
//!   conditioning is time-preserving;
//! * the pinned reference clip's transcript and digest, reported as a witness;
//! * the long recordings are read as what was said, through both reading paths;
//! * the synthesis round trip — that the recogniser hears words in the rendered
//!   waveform.
//!
//! Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… MAHBOT_ASR_MUSIC=… \
//! MAHBOT_FAPH_CORPUS=… cargo test -p mahbot-inference --release --test product_checks
//! -- --ignored --nocapture`. A test whose input is absent prints why and returns.

use std::sync::Arc;

use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::synth::{SAMPLE_RATE, Synthesis};
use mahbot_inference::transcribe::{self, Transcriber};
use mahbot_inference::{
    corpus, host, listen, music, reference, report, stand, tokenizer::Tokenizer, vad,
};

mod support;

/// The seed the round trip renders `support::TTS_TEXT` with.
const SYNTH_SEED: u64 = 12_345;

/// The render the round trip reads back: its first twelve seconds. The render
/// itself is as long as the text, and `speak_readback` is where the rest of it is
/// read.
const ROUND_TRIP_SECONDS: usize = 12;

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
        Transcriber::open(pair, tagger, support::ceiling_mib()).expect("the transcriber opens");
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
        Transcriber::open(pair, tagger, support::ceiling_mib()).expect("the transcriber opens");

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

/// What the engine says about long real audio, and the audio it says it about: a corpus
/// recording, the first `seconds` of it, and the text the one-shot path returned — read once
/// against the recording and confirmed to follow its speech, so a regression fails here
/// because the words changed. The second carries the recogniser's own mishearings, pinned as
/// it said them.
const LONG_READINGS: &[(&str, usize, &str)] = &[
    (
        stand::RECOGNITION_RECORDING,
        30,
        "dedicação e introdução de amor de perdição Esta é uma gravação de Revox. Todas as \
         gravações de Revox estão em domínio público. para mais informações ou para ser um \
         voluntário, por favor visite divulgs gravado por Sandra Luna, Amor Predição de Camilo \
         Castelo Branco, dedicação e introdução ao ilustre simão",
    ),
    (
        "speech/us-gov/speech-us-gov-0130.wav",
        35,
        "District court judge, confined by the law, to order changes to violations that are. \
         So let me just pull the the start of my slide. This is the beginning. Your response is \
         yes. You would want to afford to enjoy the luxury, yes, and no. No, we're of course \
         just. Yeah, I have control now. So do you? I do. Hendrix. Yes, ma'am. Why? Yes, \
         because people have ears on their ears.",
    ),
];

/// **The long recordings are read as what was said, through both reading paths.** The
/// stand's typical thirty seconds and the thirty-five seconds of a second real recording,
/// each through the one-shot path and through the product's, on the same audio.
///
/// The defect this holds against is one pass over the whole input: thirty seconds in one
/// prompt came back as `vitória, emoção, emoção, …`, one phrase twenty-five times and none
/// of the recording's words. So the one-shot reading is the pinned text, and neither path
/// says a phrase more often than that text does — repetition past the recording's own is
/// the invented words this check exists for.
#[test]
#[ignore = "needs the models, the tagger and the pinned corpus; run by hand"]
fn long_recordings_are_read_as_what_was_said() {
    let Some(dir) = corpus_dir() else {
        return;
    };
    let Some((pair, tagger)) = open() else {
        return;
    };
    let tokenizer = Tokenizer::from_model(&pair.body)
        .unwrap_or_else(|error| panic!("the recognition tokenizer: {error:#}"));
    let mut transcriber = Transcriber::open(Arc::clone(&pair), tagger, support::ceiling_mib())
        .expect("the transcriber opens");
    for &(name, seconds, expected) in LONG_READINGS {
        let path = dir.join(name);
        if !path.is_file() {
            println!("skip: {name} is not on this machine");
            continue;
        }
        // A copy that is not the pinned one fails here, in the pin's own words, rather than
        // later as a diff in the expected text.
        corpus::verify_file(&path, name).unwrap_or_else(|error| panic!("{error:#}"));
        let audio = host::wav_mono_16k(
            &path.display().to_string(),
            0,
            seconds * listen::SAMPLE_RATE as usize,
        )
        .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
        let read = listen::recognise(&pair, &tokenizer, audio.clone())
            .unwrap_or_else(|error| panic!("the one-shot reading of {name}: {error:#}"));
        println!(
            "one-shot {name} seconds={:.2} frames={} prompt={} generated={} truncated={} \
             text={:?}",
            audio.len() as f64 / f64::from(listen::SAMPLE_RATE),
            read.frames,
            read.prompt_rows,
            read.generated.len(),
            read.truncated,
            read.text
        );
        let product = transcriber
            .transcribe(&audio, listen::SAMPLE_RATE)
            .unwrap_or_else(|error| panic!("the product's reading of {name}: {error:#}"));
        let heard: String = product
            .regions
            .iter()
            .filter_map(|region| region.text.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "product {name} regions={} text={heard:?}",
            product.regions.len()
        );
        assert!(
            !read.truncated,
            "the one-shot reading of {name} ran out of its budget"
        );
        assert!(
            !reference::words(&heard).is_empty(),
            "the product path read nothing at all of {name}"
        );
        assert_eq!(
            reference::words(&read.text),
            reference::words(expected),
            "the one-shot reading of {name} is no longer what the recording says"
        );
        // A phrase a reading says more often than the recording's own text is invented,
        // whichever path produced it: the pinned text is the count every reading is held to.
        for (path, text) in [
            ("one-shot", read.text.as_str()),
            ("product", heard.as_str()),
        ] {
            let Some((phrase, said, recorded)) = worst_repeat(text, expected) else {
                println!("repeat {name} {path}: no phrase said twice");
                continue;
            };
            println!("repeat {name} {path} {phrase:?} ×{said}, the recording ×{recorded}");
            assert!(
                said <= recorded,
                "the {path} path on {name} says {phrase:?} ×{said}; the recording says it \
                 ×{recorded}"
            );
        }
    }
}

/// The phrase `text` repeats most often beyond `recording`'s own count of it, with both
/// counts: the invented repetition, whichever path produced it. Only a phrase the reading
/// says more than once is a repeat — one it says once is a word the recording spells
/// differently, not a repetition — so a reading with none of those is `None`. Words are
/// compared the way the pin compares them, over the guards' own
/// [`transcribe::REPEAT_GRAM`].
fn worst_repeat(text: &str, recording: &str) -> Option<(String, usize, usize)> {
    let words = reference::words(text);
    let said = reference::words(recording);
    let count = |haystack: &[String], phrase: &[String]| {
        haystack
            .windows(transcribe::REPEAT_GRAM)
            .filter(|other| *other == phrase)
            .count()
    };
    words
        .windows(transcribe::REPEAT_GRAM)
        .map(|phrase| {
            (
                phrase.join(" "),
                count(&words, phrase),
                count(&said, phrase),
            )
        })
        .filter(|&(_, heard, _)| heard > 1)
        .max_by_key(|&(_, heard, spoken)| heard.saturating_sub(spoken))
}

/// The corpus directory, or the reason the check cannot run.
fn corpus_dir() -> Option<std::path::PathBuf> {
    match corpus::dir() {
        Ok(dir) => Some(dir),
        Err(error) => {
            println!("skip: {error:#}");
            None
        }
    }
}

/// **The synthesis round trip:** the rendered waveform's opening, resampled 24 kHz →
/// 16 kHz and read back through the recogniser on its raw samples, says the words the
/// render was asked to speak. The main stand run neither performs this nor fails on
/// it.
///
/// The check reads the render's first [`ROUND_TRIP_SECONDS`] rather than all of it: a
/// render is as long as its text, and what the recogniser makes of the rest of a long
/// render is `speak_readback`'s reading — this check is the waveform still being
/// speech.
///
/// It is the **raw** waveform's reading: the check does not apply the recognition
/// side's one-shot conditioning, so a quiet render is heard as it is rather than at
/// the gain the product path would lift it by (printed beside the peak). The same clip
/// through the prepared path answers differently, so the two are not interchangeable
/// and this reading belongs to the path the gate takes.
#[test]
#[ignore = "needs both model pairs and the reference voice; run by hand"]
fn the_recogniser_hears_words_in_the_rendered_waveform() {
    let Some(voice) = voice() else {
        return;
    };
    // The round trip was judged with that voice: another one renders different words,
    // which the check cannot judge, so it says why it did not run instead of failing
    // on the substitution.
    if !reference::voice_is_pinned() {
        println!(
            "skip: {} does not name the voice the round trip is judged with",
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
    let opening = &wave[..wave.len().min(SAMPLE_RATE as usize * ROUND_TRIP_SECONDS)];
    let audio = host::resample(opening, SAMPLE_RATE, listen::SAMPLE_RATE);
    // What the check reads: the render's length, the opening's own peak, and the gain
    // the product path would condition that opening by — this check reads it raw.
    println!(
        "round trip: samples={} opening_samples={} opening_peak={} product_gain={:.2}",
        wave.len(),
        audio.len(),
        report::sci(report::max_abs(&audio)),
        host::speech_gain(&audio)
    );
    let Some(asr) = pair(Role::Recognition) else {
        return;
    };
    let tokenizer = Tokenizer::from_model(&asr.body)
        .unwrap_or_else(|error| panic!("the recognition tokenizer: {error:#}"));
    let heard = listen::recognise(&asr, &tokenizer, audio)
        .unwrap_or_else(|error| panic!("the recogniser on the rendered waveform: {error:#}"))
        .text;
    let holds = reference::check_speak_roundtrip(&heard, support::TTS_TEXT);
    assert!(
        holds,
        "the recogniser does not hear the text the render was asked to speak: {heard:?}"
    );
}
