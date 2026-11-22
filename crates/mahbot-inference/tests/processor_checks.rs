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

/// The processor, forced with the product's own switch, alone in this binary.
///
/// # Safety
/// The caller is the only check running — `support::solo` refuses a second subject rather than
/// serialising it, and the invocation names one test thread — and the product has not read the
/// switch yet, because its first compile is below this call.
fn force_processor() -> std::sync::MutexGuard<'static, ()> {
    let alone = support::solo("the processor's three checks");
    // SAFETY: see above; no other check is in the process to read the environment.
    unsafe { std::env::set_var("MAHBOT_DEVICE", "cpu") };
    alone
}

/// The check's subject, asserted rather than assumed: a switch that were ignored, or a pin that
/// had already latched, would otherwise let a check measure the graphics device and pass under a
/// processor heading — the failure the processor path exists to remove. Asked after the first
/// compile, because that is where the product settles its device.
fn assert_settled_on_the_processor() {
    assert_eq!(
        mahbot_inference::device::settled(),
        "cpu",
        "the check's first graph settled on another device"
    );
}

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
    let _alone = force_processor();
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
    assert_settled_on_the_processor();
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
    let _alone = force_processor();
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
    assert_settled_on_the_processor();
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

/// **Synthesis on the processor, heard back through the same product.** The stand's short request
/// at the stand's seed, rendered twice — the two calls must agree in length and waveform digest,
/// because the first call of a process once answered differently from every later one — and then
/// heard back through the recognition call, which must reach the text's closing words exactly as
/// the round trip demands of a graphics render.
///
/// **What it does not establish.** Nothing about the waveform's agreement with the graphics
/// device's: the processor's numerics are its own.
#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_processor_renders_the_request_and_hears_it_back() {
    let _alone = force_processor();
    support::assert_tail_group_shipped();
    let Some((dir, body, sound)) = support::pair_paths(Role::Synthesis, &[]) else {
        return;
    };
    let mut engine =
        Synthesis::new(support::open_pair(&dir, body, sound)).expect("the synthesis path");
    let wave = engine
        .synthesize(stand::SHORT_TEXT, stand::SYNTH_SEED)
        .expect("the processor's synthesis call");
    assert_settled_on_the_processor();
    assert_eq!(report::non_finite(&wave), 0, "the waveform is finite");
    assert!(
        report::max_abs(&wave) <= 1.0,
        "the waveform leaves the PCM range"
    );
    let frame = mahbot_inference::graphs::samples_per_frame();
    assert!(
        wave.len() >= SAMPLE_RATE as usize && wave.len().is_multiple_of(frame),
        "the render is {} samples, not whole {frame}-sample frames and at least a second of \
         speech",
        wave.len()
    );
    let digest = report::digest(&wave);
    println!(
        "processor render samples={} frames={} seconds={:.2} digest={digest:016x}",
        wave.len(),
        wave.len() / frame,
        wave.len() as f64 / f64::from(SAMPLE_RATE),
    );
    // The same text and the same seed must give the same waveform, bit for bit.
    let again = engine
        .synthesize(stand::SHORT_TEXT, stand::SYNTH_SEED)
        .expect("the processor's second synthesis call");
    assert_eq!(
        (again.len(), report::digest(&again)),
        (wave.len(), digest),
        "the second call of the process rendered a different waveform from the first"
    );

    // The render's weights go back before the recogniser's are asked for, as the round trip
    // does in the other direction.
    drop(engine);
    mahbot_inference::release_process_memory();
    let Some((dir, body, sound)) = support::pair_paths(Role::Recognition, &[CED_MODEL]) else {
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
}
