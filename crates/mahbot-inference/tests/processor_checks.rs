// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The product's own checks on the **processor** — the device the product falls back to when no
//! graphics device comes up, and the one no harness here could reach while every check pinned the
//! graphics device first.
//!
//! One target of its own, because the switch it sets (`MAHBOT_DEVICE=cpu`) is process-global and
//! the product reads it once, at its first compile: set beside the checks that pin the graphics
//! device it would either panic them or measure an already-settled device under a processor
//! heading. A separate binary is the isolation.
//!
//! The three checks are the product's three shapes — what its tagger says, what its recognition
//! returns, what its synthesis renders — and each takes the process alone for its whole body
//! (`support::solo`): the switch, the device it settles and the models' memory are all one per
//! process, so no two of them may measure at once.
//!
//! Each is `#[ignore]`d and needs the installed model tree and the pinned corpus; none is in a CI
//! gate for the same reason, and they are run by hand. `--test-threads=1` is part of the
//! invocation, not cosmetic: `solo` refuses a second subject rather than serialising it, so a
//! parallel run panics instead of quietly reporting one check's device under another's name.
//!
//! ```text
//! cargo test --release -p mahbot-inference --features hand-run --test processor_checks \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::loader::Role;
use mahbot_inference::synth::{SAMPLE_RATE, Synthesis};
use mahbot_inference::transcribe::{self, Transcriber};
use mahbot_inference::{host, listen, reference, report, stand};

mod support;

/// The stand's `recognition.typical` row: the pinned recording's first thirty seconds.
const ROW_SECONDS: usize = 30;

/// How many of the tagger's 527 classes may sit at 0.9999 or above on one window.
///
/// A handful is what a sigmoid head does at the top of a clear class; the defect this
/// check holds against — a half weight read as f32 lanes — put *hundreds* there, every
/// column saturated at exactly 1.0, and the label a region took from them withheld its
/// text from the recogniser. The pinned recording's middle window reads **0 of 527** on
/// this machine's processor, so the bound is not near the reading.
const MAX_SATURATED: usize = 8;

/// **The tagger on the processor.** Every other harness here pins the graphics device, so the
/// product's tagger had only ever been read there — which is how a 527-way head saturating at
/// exactly 1.0 reached a record, withholding most of the speech from the recogniser, with
/// nothing on the processor to compare it against.
///
/// On the thirty seconds the stand's recognition row reads: every window's label is one the
/// product counts as word-bearing (its own [`transcribe::bears_words`], not a copy of the rule),
/// and no more than a handful of the 527 columns are saturated. *Which* class leads is the
/// reference model's arithmetic and is printed rather than asserted — CED's AudioSet columns
/// overlap heavily, the same reason `ced_tagger.rs` asks for a voice among the top five rather
/// than for one particular column. On this machine's processor all five windows read `Speech`,
/// the middle one at 0.7633.
///
/// **What it does not establish.** Nothing about the graphics device — the other checks here are
/// its; and nothing about another CED checkpoint or another recording.
#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_processor_tagger_reads_the_recording_as_speech() {
    let _alone = support::force_processor("the processor's three checks");
    support::assert_tail_group_shipped();
    let Some((dir, ..)) = support::pair_paths(Role::Recognition, &[CED_MODEL]) else {
        return;
    };
    let Some(recording) = support::recording(stand::RECOGNITION_RECORDING) else {
        return;
    };
    let audio = host::wav_mono_16k(
        &recording.display().to_string(),
        0,
        ROW_SECONDS * listen::SAMPLE_RATE as usize,
    )
    .unwrap_or_else(|error| panic!("{}: {error:#}", recording.display()));
    let mut tagger = Tagger::open(&dir.join(CED_MODEL))
        .unwrap_or_else(|error| panic!("{}: {error:#}", dir.join(CED_MODEL).display()));

    let mel = tagger
        .log_mel(&audio)
        .unwrap_or_else(|error| panic!("the tagger's front end: {error:#}"));
    let window = tagger.window_frames();
    assert!(
        mel.frames >= window,
        "{} mel frames is shorter than one {window}-frame window",
        mel.frames
    );
    let tags = tagger
        .tags(&mel)
        .unwrap_or_else(|error| panic!("the tagger's windows: {error:#}"));
    support::assert_settled_on_the_processor();
    let labels: Vec<&str> = tags
        .iter()
        .map(|tag| tagger.labels()[tag.index].as_str())
        .collect();
    let speech = labels
        .iter()
        .filter(|label| transcribe::bears_words(label))
        .count();
    println!(
        "processor tagger windows={} labels={:?} speech_windows={speech}",
        tags.len(),
        labels
    );
    assert!(
        speech > 0,
        "no window of {} says the audio can hold words: {:?}",
        stand::RECOGNITION_RECORDING,
        labels
    );

    // The head's own census on the middle window: a sigmoid gives a few columns at 1.0 and
    // hundreds only when the weight was read as the wrong width.
    let start = (mel.frames - window) / 2;
    let probabilities = tagger
        .probabilities(&mel, start)
        .unwrap_or_else(|error| panic!("the tagger's head: {error:#}"));
    assert_eq!(
        probabilities.len(),
        527,
        "the pinned head's width, one column a label"
    );
    let saturated = probabilities.iter().filter(|p| **p >= 0.9999).count();
    let top = probabilities
        .iter()
        .copied()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |best, (i, p)| {
                if p > best.1 { (i, p) } else { best }
            },
        );
    let best = tagger.labels()[top.0].as_str();
    println!(
        "processor tagger top1={best} score={:.4} saturated={saturated}/527",
        top.1
    );
    assert!(
        transcribe::bears_words(best),
        "the middle window's best label is `{best}` at {:.4}, which the product does not count as \
         word-bearing: the 527-way head is not reading the recording as the speech it is",
        top.1
    );
    assert!(
        saturated <= MAX_SATURATED,
        "{saturated} of 527 classes are at 0.9999 or above; the head is saturated, not reading \
         its weight as the halves it holds"
    );
}

/// **Recognition on the processor.** The product's own transcription call, on the same thirty
/// seconds: the regions tile the input and some of them carry words. It is the reading a user
/// gets, not a layer of it — and the one the tagger's labels gate, which is why the check above
/// comes first in a session that has to explain a short transcript.
///
/// **What it does not establish.** Nothing about the graphics device's answer: the processor's
/// numerics are its own, so the region and sample counts are comparable and the bytes are not.
#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_processor_reads_the_recording() {
    let _alone = support::force_processor("the processor's three checks");
    support::assert_tail_group_shipped();
    let Some((dir, body, sound)) = support::pair_paths(Role::Recognition, &[CED_MODEL]) else {
        return;
    };
    let Some(recording) = support::recording(stand::RECOGNITION_RECORDING) else {
        return;
    };
    let audio = host::wav_mono_16k(
        &recording.display().to_string(),
        0,
        ROW_SECONDS * listen::SAMPLE_RATE as usize,
    )
    .unwrap_or_else(|error| panic!("{}: {error:#}", recording.display()));
    let pair = support::open_pair(&dir, body, sound);
    let tagger = Tagger::open(&dir.join(CED_MODEL))
        .unwrap_or_else(|error| panic!("{}: {error:#}", dir.join(CED_MODEL).display()));
    let mut transcriber = Transcriber::open(pair, tagger, support::ceiling_mib())
        .expect("the transcriber opens on the processor");
    let out = transcriber
        .transcribe(&audio, listen::SAMPLE_RATE)
        .unwrap_or_else(|error| panic!("the processor's reading: {error:#}"));
    support::assert_settled_on_the_processor();
    out.check()
        .unwrap_or_else(|error| panic!("the regions do not tile the input: {error:#}"));
    let read: String = out
        .regions
        .iter()
        .filter_map(|region| region.text.as_deref())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "processor reading samples={} seconds={:.2} gain={:.2} regions={} read={} text={read:?}",
        out.samples,
        out.duration_s(),
        out.gain,
        out.regions.len(),
        out.regions.iter().filter(|region| region.read).count(),
    );
    assert!(
        !reference::words(&read).is_empty(),
        "the processor read nothing at all of {}",
        stand::RECOGNITION_RECORDING
    );
}

/// The checks a render must pass whatever its text: finite, inside the PCM range, whole
/// frames and at least a second of speech. Prints its length and waveform digest under
/// `label`, and returns the digest for a caller that has something to compare it with.
fn rendered(label: &str, wave: &[f32]) -> u64 {
    assert_eq!(
        report::non_finite(wave),
        0,
        "the {label} waveform is finite"
    );
    assert!(
        report::max_abs(wave) <= 1.0,
        "the {label} waveform leaves the PCM range"
    );
    let frame = mahbot_inference::graphs::samples_per_frame();
    assert!(
        wave.len() >= SAMPLE_RATE as usize && wave.len().is_multiple_of(frame),
        "the {label} render is {} samples, not whole {frame}-sample frames and at least a \
         second of speech",
        wave.len()
    );
    let digest = report::digest(wave);
    println!(
        "processor {label} samples={} frames={} seconds={:.2} digest={digest:016x}",
        wave.len(),
        wave.len() / frame,
        wave.len() as f64 / f64::from(SAMPLE_RATE),
    );
    digest
}

/// **Synthesis on the processor, heard back through the same product.** The stand's short request
/// at the stand's seed, rendered twice — the two calls must agree in length and waveform digest,
/// because the first call of a process once answered differently from every later one — and then
/// heard back through the recognition call, which must reach the text's closing words exactly as
/// the round trip demands of a graphics render. The stand's 129-character request is rendered and
/// heard back too: it is the other row the eight-bit route moved, and the short request's words
/// alone would not say whether a longer render still reads as asked.
///
/// Two ambient switches are masked for the whole check, through the process-wide override and
/// restored — not thread-scoped: the schedule runs on a pool worker by default
/// (`RLX_CPU_EXEC_IN_POOL`), so a thread-local override would not be visible where the route
/// reads it. `RLX_Q8_0_EXACT_GEMV` puts the f32 route back, which renders different words,
/// and a diagnostic handle must not be able to redden a gate that prices the shipped route.
/// `RLX_REPRODUCIBLE` stands every single-row integer route down: a supported run mode, not a
/// handle, and its own long render does not close on the text's last four words today (80 frames,
/// heard back as "…before the show"). That is a question about the mode's numerics, not about the
/// route this gate prices; it is recorded here rather than absorbed.
///
/// **What it does not establish.** Nothing about the waveform's agreement with the graphics
/// device's: the processor's numerics are its own.
#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_processor_renders_the_request_and_hears_it_back() {
    let _alone = support::force_processor("the processor's three checks");
    support::assert_tail_group_shipped();
    support::with_env("RLX_REPRODUCIBLE", "0", || {
        support::with_env("RLX_Q8_0_EXACT_GEMV", "0", || {
            let Some((dir, body, sound)) = support::pair_paths(Role::Synthesis, &[]) else {
                return;
            };
            let mut engine =
                Synthesis::new(support::open_pair(&dir, body, sound)).expect("the synthesis path");
            let wave = engine
                .synthesize(stand::SHORT_TEXT, stand::SYNTH_SEED)
                .expect("the processor's synthesis call");
            support::assert_settled_on_the_processor();
            let digest = rendered("render", &wave);
            // The same text and the same seed must give the same waveform, bit for bit.
            let again = engine
                .synthesize(stand::SHORT_TEXT, stand::SYNTH_SEED)
                .expect("the processor's second synthesis call");
            assert_eq!(
                (again.len(), report::digest(&again)),
                (wave.len(), digest),
                "the second call of the process rendered a different waveform from the first"
            );

            // The stand's other synthesis row, rendered once: it is the row the eight-bit route
            // moved most, and the words are its witness.
            let long = engine
                .synthesize(stand::TYPICAL_TEXT, stand::SYNTH_SEED)
                .expect("the processor's long synthesis call");
            // Its digest is printed by the check and has nothing here to be compared with.
            rendered("long render", &long);

            // The render's weights go back before the recogniser's are asked for, as the round trip
            // does in the other direction.
            drop(engine);
            mahbot_inference::release_process_memory();
            let Some((dir, body, sound)) = support::pair_paths(Role::Recognition, &[CED_MODEL])
            else {
                return;
            };
            let pair = support::open_pair(&dir, body, sound);
            let tagger = Tagger::open(&dir.join(CED_MODEL))
                .unwrap_or_else(|error| panic!("{}: {error:#}", dir.join(CED_MODEL).display()));
            let mut transcriber = Transcriber::open(pair, tagger, support::ceiling_mib())
                .expect("the transcriber reopens on the processor");
            let hearing = support::heard(&mut transcriber, "processor render", &wave, SAMPLE_RATE);
            assert!(
                reference::check_speak_roundtrip(&hearing, stand::SHORT_TEXT),
                "the processor's render does not pass the round trip"
            );
            let long_hearing = support::heard(
                &mut transcriber,
                "processor long render",
                &long,
                SAMPLE_RATE,
            );
            assert!(
                reference::check_speak_roundtrip(&long_hearing, stand::TYPICAL_TEXT),
                "the processor's long render does not pass the round trip"
            );
        });
    });
}
