// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CED tagger on real music: eight clips, their labels, and the shapes the
//! front end promises.
//!
//! **What it pins.** Not a golden vector — there is none for this model here —
//! but the thing a tagger is *for*, on a corpus whose split is ground truth
//! (`~/.mahbot/asr_audio/sources.md`): every vocal clip's five best labels name
//! a voice, the a cappella solfège recording is not tagged as an instrument,
//! none of the four instrumental clips is tagged as speech, and at least one of
//! them is tagged as music or an instrument. Around those assertions the run
//! prints, for every clip, its mel shape, the middle window's `top1` and top
//! five labels, and what that window cost — timings are printed and never
//! asserted, so a busy machine is not a regression.
//!
//! **One bar is a witness, not the reference's answer.** On the pinned file the
//! a cappella clip's *own* top-1 is `Lullaby`, with `A capella`,
//! `Female singing` and `Vocal music` in the five behind it — so this file
//! asserts that a voice is among the top five and that no instrument is first,
//! rather than requiring one particular voice column to lead. CED's 527 AudioSet
//! columns overlap heavily ("Singing", "Choir", "A capella", "Vocal music",
//! "Lullaby" all describe that recording), and which of them wins is the
//! reference model's own arithmetic, not something this port can decide. What
//! the run does establish is that the whole path — log-mel, patch lift, twelve
//! blocks, the pooled head — is wired in the reference's order and scale: the
//! instrumental half comes out of the same graph as `Music`, `Piano`,
//! `Brass instrument`/`Trombone`/`Trumpet`, `Trance music`/`Techno` and
//! `Synthesizer`, which is what the recordings are.
//!
//! **What it does not establish.** Nothing about parity with the reference
//! engine: no fixture records its 527 columns, so the tolerance here is a
//! label, not a cosine. It says nothing about a device without wgpu (it skips)
//! and nothing about another CED checkpoint — the pins below name one file.
//!
//! **When it runs.** The test ends in `_with_models`. It needs the pinned GGUF
//! under `MAHBOT_MODELS_DIR` and the eight clips under `MAHBOT_ASR_MUSIC`, plus a
//! wgpu adapter; without any of them it prints why and returns, and an input the
//! run was pointed at that is not what it names fails.

use std::path::PathBuf;

use mahbot_inference::ced::{CED_MODEL, CED_MODEL_SOURCE, Tagger};
use mahbot_inference::host;
use mahbot_inference::loader;
use mahbot_inference::music;
use mahbot_inference::report;

mod support;

/// The model directory: `MAHBOT_MODELS_DIR`'s, or the reason this gate cannot run.
/// The crate's tests skip on an absent variable and say so; a variable that is set
/// and names something this gate cannot use is a mis-provisioned run, and fails.
fn models_dir() -> Option<PathBuf> {
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(error) => {
            println!("skip: {error:#}");
            return None;
        }
    };
    let path = dir.join(CED_MODEL);
    if !path.is_file() {
        panic!(
            "{} is set and {} is not on this machine",
            loader::MODELS_DIR_ENV,
            path.display()
        );
    }
    Some(dir)
}

/// Every label of the file, the top `n` of one window by probability.
fn top(probabilities: &[f32], labels: &[String], n: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("probabilities compare"));
    ranked.truncate(n);
    for (index, _) in &ranked {
        assert!(*index < labels.len(), "a probability past the label list");
    }
    ranked
}

/// Whether a label names music or an instrument — the words the AudioSet
/// ontology uses for the instrumental half of the corpus.
fn is_music_like(label: &str) -> bool {
    let label = label.to_lowercase();
    [
        "music",
        "instrument",
        "piano",
        "orchestra",
        "guitar",
        "electronic",
        "synthesizer",
        "keyboard",
        "drum",
        "percussion",
        "ambient",
        "violin",
        "string",
        "brass",
        "wind",
    ]
    .iter()
    .any(|word| label.contains(word))
}

/// Whether a label names a voice — the a cappella clip's own kind.
fn is_voice(label: &str) -> bool {
    let label = label.to_lowercase();
    ["sing", "choir", "vocal", "voice", "chant", "hum"]
        .iter()
        .any(|word| label.contains(word))
}

/// Whether a label names an instrument rather than a kind of music: the class of
/// answer the a cappella clip must not get.
fn is_instrument(label: &str) -> bool {
    let label = label.to_lowercase();
    [
        "instrument",
        "piano",
        "guitar",
        "drum",
        "violin",
        "cello",
        "trumpet",
        "trombone",
        "saxophone",
        "flute",
        "organ",
        "synthesizer",
        "bass",
        "percussion",
        "string",
        "brass",
        "mallet",
    ]
    .iter()
    .any(|word| label.contains(word))
}

#[test]
fn the_eight_music_clips_are_labelled_by_their_split_with_models() {
    let Some(models) = models_dir() else {
        return;
    };
    let Some(pool) = support::pool() else {
        return;
    };
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let open = report::Stopwatch::start();
    let mut tagger = Tagger::open(&models.join(CED_MODEL)).expect("the pinned tagger opens");
    let build_ms = open.ms();
    let window = tagger.window_frames();
    println!(
        "tagger source={}@{} sha256={} build_ms={build_ms:.2} labels={} window={window} hop={}",
        CED_MODEL_SOURCE.0,
        CED_MODEL_SOURCE.1,
        CED_MODEL_SOURCE.2,
        tagger.labels().len(),
        tagger.hop_frames(),
    );

    // The pool's own table, in its own order: one list, so a renamed clip cannot
    // leave the pins and these assertions describing different files.
    let mut seen: Vec<(&str, bool, String, Vec<String>)> = Vec::with_capacity(pool.len());
    for (clip, path) in music::CLIPS.iter().zip(&pool) {
        let name = clip.path;
        let instrumental = clip.path.starts_with("instrumental/");
        let samples = host::wav_mono_16k(&path.to_string_lossy(), 0, usize::MAX)
            .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        let mel_timer = report::Stopwatch::start();
        let mel = tagger
            .log_mel(&samples)
            .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        let mel_ms = mel_timer.ms();

        // ── the front end's own shape ──
        // The hop the region labels are read at is the file's `ced.hop_size`, and
        // the front end's frame count follows it.
        let hop = tagger.samples_per_frame();
        assert_eq!(
            mel.frames,
            1 + samples.len() / hop,
            "{name}: frames = 1 + samples/hop"
        );
        assert_eq!(
            mel.values.len(),
            mel.n_mels * mel.frames,
            "{name}: n_mels · frames"
        );
        assert!(
            mel.values.iter().all(|v| v.is_finite()),
            "{name}: the log-mel has a non-finite value"
        );

        // ── one window from the middle of the file ──
        assert!(
            mel.frames >= window,
            "{name}: {} frames is shorter than one {window}-frame window",
            mel.frames
        );
        let start = (mel.frames - window) / 2;
        let timer = report::Stopwatch::start();
        let probabilities = tagger
            .probabilities(&mel, start)
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let ms = timer.ms();
        assert_eq!(
            probabilities.len(),
            tagger.labels().len(),
            "{name}: one column a label"
        );
        assert_eq!(probabilities.len(), 527, "{name}: the pinned head's width");
        assert!(
            probabilities
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "{name}: a probability is outside [0, 1] or not finite"
        );

        let ranked = top(&probabilities, tagger.labels(), 5);
        let label = tagger.labels()[ranked[0].0].clone();
        let five: Vec<String> = ranked
            .iter()
            .map(|(i, p)| format!("{}={p:.4}", tagger.labels()[*i]))
            .collect();
        println!(
            "{name} samples={} frames={} mel_ms={mel_ms:.2} start={start} top1={label} \
             score={:.4} window_ms={ms:.2} top5=[{}]",
            samples.len(),
            mel.frames,
            ranked[0].1,
            five.join(", ")
        );

        // ── the whole file, through the tag reader ──
        let tags = tagger
            .tags(&mel)
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let whole = tagger.label_of(&tags, mel.frames / 4, 3 * mel.frames / 4);
        println!(
            "    tags={} windows={:?} whole_file={}",
            tags.len(),
            tags.iter()
                .map(|t| (t.start, t.end, tagger.labels()[t.index].as_str()))
                .collect::<Vec<_>>(),
            whole.map_or_else(
                || "none".to_string(),
                |(i, s)| format!("{}={s:.4}", tagger.labels()[i])
            )
        );
        seen.push((
            name,
            instrumental,
            label,
            ranked
                .iter()
                .map(|(i, _)| tagger.labels()[*i].clone())
                .collect(),
        ));
    }

    // ── the split, as the corpus states it ──
    // The a cappella recording's own top-1 is `Lullaby` — a music class, with
    // `A capella`, `Female singing` and `Vocal music` in the five behind it. So
    // the bar here is that the model hears a voice in it and does not call it an
    // instrument, not that one particular voice column comes first: CED's 527
    // columns overlap ("Singing", "Choir", "A capella", "Vocal music", "Lullaby"
    // all describe this recording), and which of them leads is the reference
    // model's own business.
    let (name, _, label, top5) = seen
        .iter()
        .find(|(name, _, _, _)| *name == "vocal/01-acappella-solfege.wav")
        .expect("the solfège clip is one of the pins");
    assert!(
        !is_instrument(label),
        "{name}: `{label}` is an instrument — an a cappella recording tagged as one \
         is the front end or the patch order being wrong"
    );
    assert!(
        top5.iter().any(|l| is_voice(l)),
        "{name}: none of {top5:?} names a voice"
    );
    for (name, instrumental, _, top5) in &seen {
        assert!(
            *instrumental || top5.iter().any(|l| is_voice(l)),
            "{name}: none of the five labels {top5:?} names a voice"
        );
    }

    let instrumental: Vec<&(&str, bool, String, Vec<String>)> = seen
        .iter()
        .filter(|(_, instrumental, _, _)| *instrumental)
        .collect();
    assert!(
        instrumental
            .iter()
            .any(|(_, _, label, _)| is_music_like(label)),
        "no instrumental clip was tagged as music or an instrument: {instrumental:?}"
    );
    for (name, _, label, _) in instrumental {
        assert!(
            !label.to_lowercase().contains("speech"),
            "{name}: `{label}` is speech, and this clip holds none"
        );
    }
    println!("peak_rss_mib={}", report::peak_rss() / (1 << 20));
}
