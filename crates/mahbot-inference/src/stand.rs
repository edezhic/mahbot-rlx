// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stand: how the models do now.
//!
//! One table, six rows, each measured in a process of its own:
//!
//! * **recognition** at a short, a typical and a long voice message — audio in, the
//!   regions the product's transcription call returns out;
//! * **synthesis** at a short and a typical text — text in, a 24 kHz waveform out;
//! * **wake word** — one window through the ear path to a score.
//!
//! Every row carries four numbers: cold time (the process starts with no models
//! loaded and the call loads them), warm time (the models already resident),
//! resting memory (after the cold call, with the models loaded and the graphs
//! compiled) and peak memory (the highest the process reaches). Spread is not
//! recorded; diagnostics go to the console and never enter the file.
//!
//! `mahbot-inference` with no arguments is the complete run: it takes the heavy
//! run's lock, spawns one child per row and writes `STAND.md`. `mahbot-inference
//! N…` measures the named rows only and prints them; it writes no record unless it
//! measured the whole table, and neither does a run under a bound larger than the
//! default (a screened run). A row whose input is missing fails the run and names it.
//! A row stopped by a guard makes the run exit [`crate::guard::EXIT_STOPPED`].
//!
//! Inputs: `MAHBOT_MODELS_DIR` (the two pairs and the audio tagger),
//! `MAHBOT_VOICE_WAV` (the wake window and the synthesis reference) and
//! `MAHBOT_FAPH_CORPUS` (the pinned recordings the recognition messages are cut
//! from). The checkout carries no models, no corpus and no audio.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use serde_json::json;

use crate::corpus;
use crate::guard::Bounds;
use crate::listen::{self, Listener};
use crate::loader::{self, Pair, Role};
use crate::measurements;
use crate::reference;
use crate::report::{self, Stopwatch};
use crate::run::{self, Reading};
use crate::synth::Synthesis;
use crate::transcribe::{self, Transcriber};
use crate::wake;

/// Marks a child of the complete run: it measures the one row its argument names.
const ROW_CHILD: &str = "MAHBOT_STAND_ROW";

/// The pinned recording the three recognition messages are cut from: a five-minute
/// speech file of the corpus listing. Named rather than picked out of the listing by
/// position, so a reordering of the bench's table cannot re-point these rows at other
/// material; [`corpus::verify_file`] holds it to its own pin.
const RECOGNITION_RECORDING: &str = "speech/librivox/speech-librivox-0027.wav";
/// The short and typical voice messages, in samples at 16 kHz.
const SHORT_SAMPLES: usize = 10 * 16_000;
const TYPICAL_SAMPLES: usize = 30 * 16_000;

/// The two synthesis requests: a greeting and a sentence about a sentence.
const SHORT_TEXT: &str = "Hello, how are you today?";
const TYPICAL_TEXT: &str = "The weather this morning is quiet and cool, so I will walk down to \
     the harbour and watch the boats come in before the shops open.";
/// The draw seed both requests are synthesised from.
const SYNTH_SEED: u64 = 1;

/// The speech window in the recorded voice: the file is silent up to ~1.16 s, so
/// the ear starts where there is speech. The reference window is a later segment
/// of the same recording, used only as the cosine's other side.
const WAKE_OFFSET: usize = 16_000;
const WAKE_REFERENCE_OFFSET: usize = 60_000;

/// What one row measures, and the witness its output is printed against.
#[derive(Clone, Copy)]
enum Surface {
    /// A prefix of the pinned recording in samples at 16 kHz (`None` is the
    /// whole recording) and the SHA-256 of the answer the row was recorded with.
    Recognition {
        samples: Option<usize>,
        witness: &'static str,
    },
    /// A text, the waveform digest and the sample count the row was recorded with.
    Synthesis {
        text: &'static str,
        digest: u64,
        samples: usize,
    },
    /// The score the row was recorded with.
    Wake { score: f32 },
}

/// One row of the table: its machine name, its label, and the product shape it
/// measures.
struct Row {
    name: &'static str,
    label: &'static str,
    surface: Surface,
}

const ROWS: [Row; 6] = [
    Row {
        name: "recognition.short",
        label: "recognition, short voice message (10 s)",
        surface: Surface::Recognition {
            samples: Some(SHORT_SAMPLES),
            witness: reference::STAND_RECOGNITION_SHORT_SHA256,
        },
    },
    Row {
        name: "recognition.typical",
        label: "recognition, typical voice message (30 s)",
        surface: Surface::Recognition {
            samples: Some(TYPICAL_SAMPLES),
            witness: reference::STAND_RECOGNITION_TYPICAL_SHA256,
        },
    },
    Row {
        name: "recognition.long",
        label: "recognition, long voice message (the whole pinned recording)",
        surface: Surface::Recognition {
            samples: None,
            witness: reference::STAND_RECOGNITION_LONG_SHA256,
        },
    },
    Row {
        name: "synthesis.short",
        label: "synthesis, short text",
        surface: Surface::Synthesis {
            text: SHORT_TEXT,
            digest: reference::STAND_SYNTH_SHORT_DIGEST,
            samples: reference::STAND_SYNTH_SHORT_SAMPLES,
        },
    },
    Row {
        name: "synthesis.typical",
        label: "synthesis, typical text",
        surface: Surface::Synthesis {
            text: TYPICAL_TEXT,
            digest: reference::STAND_SYNTH_TYPICAL_DIGEST,
            samples: reference::STAND_SYNTH_TYPICAL_SAMPLES,
        },
    },
    Row {
        name: "wake",
        label: "wake word, one scoring step",
        surface: Surface::Wake {
            score: reference::STAND_WAKE_SCORE,
        },
    },
];

/// The row a child named, or `None` when the name is not in the table.
fn row(name: &str) -> Option<&'static Row> {
    ROWS.iter().find(|row| row.name == name)
}

/// The stand's entry point: the parent walks the closed row list, a child
/// re-entered with one row index measures that row.
pub fn main_entry() {
    run::record_run(run::Mode {
        label: "stand",
        record: measurements::Record::Stand,
        marker: ROW_CHILD,
        total: ROWS.len(),
        name: |index| ROWS[index].name,
        measure: |index, bounds| measure(&ROWS[index], bounds),
        print: print_rows,
        compose: |readings| {
            Ok((
                format!("{}\n\n{}", table(readings).trim_end(), caption(readings)),
                inputs_note(),
            ))
        },
    });
}

/// Measure one row in this process, under the bounds this run resolved.
fn measure(row: &Row, bounds: Bounds) -> Result<Reading> {
    match row.surface {
        Surface::Recognition { samples, witness } => recognition(row, samples, witness, bounds),
        Surface::Synthesis {
            text,
            digest,
            samples,
        } => synthesis(row, text, digest, samples),
        Surface::Wake { score } => wake(row, score),
    }
}

/// One voice message through the product's transcription call — the recording read
/// and resampled to 16 kHz, then guards, gain, tagger, voice-activity mask, span
/// merge, cut, ladder, text guards and retime — cold and then warm.
fn recognition(
    row: &Row,
    samples: Option<usize>,
    witness: &str,
    bounds: Bounds,
) -> Result<Reading> {
    let recording = corpus::dir()?.join(RECOGNITION_RECORDING);
    corpus::verify_file(&recording, RECOGNITION_RECORDING)?;
    let limit = samples.unwrap_or(usize::MAX);

    let t = Stopwatch::start();
    let mut transcriber = Transcriber::open_product(bounds.max_rss_mib)?;
    let cold = call(&mut transcriber, &recording, limit)?;
    let cold_ms = t.ms();
    let memory = report::Memory::take();

    let t = Stopwatch::start();
    let warm = call(&mut transcriber, &recording, limit)?;
    let warm_ms = t.ms();
    cold_warm(row, cold.regions == warm.regions, "regions");
    let digest = answer_digest(&warm);
    let equal = digest == witness;
    if !equal {
        eprintln!(
            "stand: {}: the answer moved from its witness ({digest} against {witness})",
            row.name
        );
    }
    Ok(Reading {
        row: row.name.to_string(),
        cold_ms,
        warm_ms,
        memory: memory.finish(),
        detail: run::detail([
            ("regions", json!(warm.regions.len())),
            ("samples", json!(warm.samples)),
            ("gain", json!(warm.gain)),
            ("digest", json!(digest)),
            ("witness", json!(witness)),
            ("equal", json!(equal)),
        ]),
    })
}

/// SHA-256 over the call's answer: each region's bounds, its read flag and its
/// transcript or label, so the witness moves only when the output does.
fn answer_digest(out: &transcribe::Transcription) -> String {
    use sha2::{Digest as _, Sha256};
    let mut hasher = Sha256::new();
    for region in &out.regions {
        let text = region
            .text
            .as_deref()
            .or(region.label.as_deref())
            .unwrap_or("");
        hasher.update(format!(
            "{}:{}:{}:{text}\n",
            region.start, region.end, region.read
        ));
    }
    measurements::hex(&hasher.finalize())
}

/// One text through the product's synthesis call, cold and then warm.
fn synthesis(row: &Row, text: &str, witness: u64, witness_samples: usize) -> Result<Reading> {
    let voice = loader::voice_wav()?;

    let t = Stopwatch::start();
    let pair = Arc::new(Pair::of_role(Role::Synthesis)?);
    let mut engine = Synthesis::new(pair, &voice)?;
    let cold = engine.synthesize(text, SYNTH_SEED)?;
    let cold_ms = t.ms();
    let memory = report::Memory::take();

    let t = Stopwatch::start();
    let warm = engine.synthesize(text, SYNTH_SEED)?;
    let warm_ms = t.ms();
    cold_warm(
        row,
        cold.len() == warm.len() && report::digest(&cold) == report::digest(&warm),
        "waveform",
    );
    let peak_abs = report::max_abs(&warm);
    if report::non_finite(&warm) != 0 || !peak_abs.is_finite() || peak_abs > 1.0 {
        bail!(
            "the waveform leaves the PCM range: max|y|={} nonfinite={}",
            report::sci(peak_abs),
            report::non_finite(&warm)
        );
    }
    let digest = report::digest(&warm);
    let equal = digest == witness && warm.len() == witness_samples;
    if !equal {
        eprintln!(
            "stand: {}: the waveform moved from its witness ({digest:016x} · {} samples against \
             {witness:016x} · {witness_samples})",
            row.name,
            warm.len(),
        );
    }
    Ok(Reading {
        row: row.name.to_string(),
        cold_ms,
        warm_ms,
        memory: memory.finish(),
        detail: run::detail([
            (
                "frames",
                json!(warm.len() / crate::graphs::samples_per_frame()),
            ),
            ("samples", json!(warm.len())),
            ("digest", json!(format!("{digest:016x}"))),
            ("witness", json!(format!("{witness:016x}"))),
            ("equal", json!(equal)),
        ]),
    })
}

/// One wake window through the ear path to a score: the window embedded cold and
/// warm, and its cosine against a later window of the same recording.
fn wake(row: &Row, witness: f32) -> Result<Reading> {
    let voice = loader::voice_wav()?;
    let primary = window(&voice, WAKE_OFFSET)?;

    let t = Stopwatch::start();
    let pair = Arc::new(Pair::of_role(Role::Recognition)?);
    let mut listener = Listener::new(pair)?;
    let embedding = listener.embed(&primary)?;
    let cold_ms = t.ms();
    let memory = report::Memory::take();

    let t = Stopwatch::start();
    let again = listener.embed(&primary)?;
    let warm_ms = t.ms();
    if report::non_finite(&embedding) != 0 || report::non_finite(&again) != 0 {
        bail!(
            "the embedding has {} non-finite values — a score against it is no reading",
            report::non_finite(&embedding) + report::non_finite(&again)
        );
    }
    cold_warm(row, embedding == again, "embedding");
    // The score needs a second side: a later window of the same recording, on the
    // same compiled graphs. It is not a second calibration pass and not timed.
    let reference = window(&voice, WAKE_REFERENCE_OFFSET)?;
    let score = wake::cosine(&embedding, &listener.embed(&reference)?);
    let equal = (score - witness).abs() < 5e-5;
    if !equal {
        eprintln!(
            "stand: {}: the score moved from its witness ({score:.4} against {witness:.4})",
            row.name
        );
    }
    Ok(Reading {
        row: row.name.to_string(),
        cold_ms,
        warm_ms,
        memory: memory.finish(),
        detail: run::detail([
            ("dim", json!(embedding.len())),
            ("score", json!(format!("{score:.4}"))),
            ("witness", json!(format!("{witness:.4}"))),
            ("equal", json!(equal)),
        ]),
    })
}

/// Read one recording at 16 kHz mono, `offset` samples in and at most `limit`.
fn read(path: &Path, offset: usize, limit: usize) -> Result<Vec<f32>> {
    crate::host::wav_mono_16k(&path.display().to_string(), offset, limit)
        .with_context(|| format!("reading `{}`", path.display()))
}

/// One product call on a recording: read and resampled to 16 kHz, then
/// transcribed — the surface `transcribe_file` serves and the corpus run times,
/// with the message's own length in place of the file's.
fn call(
    transcriber: &mut Transcriber,
    path: &Path,
    limit: usize,
) -> Result<transcribe::Transcription> {
    let audio = read(path, 0, limit)?;
    transcriber
        .transcribe(&audio, transcribe::RATE)
        .with_context(|| format!("transcribing `{}`", path.display()))
}

/// One 16 kHz window at `offset`, refused when the recording is short: the
/// ear pads a short slice with silence, and a score against a mostly-zero
/// embedding is not a reading of the voice.
fn window(path: &Path, offset: usize) -> Result<Vec<f32>> {
    let samples = read(path, offset, listen::WINDOW_SAMPLES)?;
    if samples.len() < listen::WINDOW_SAMPLES {
        bail!(
            "`{}` holds {} samples at {offset}, short of the {} a window takes",
            path.display(),
            samples.len(),
            listen::WINDOW_SAMPLES
        );
    }
    Ok(samples)
}

/// Print a cold/warm difference as a witness: the two calls may take different
/// cached routes, and the row is still a reading of the product.
fn cold_warm(row: &Row, same: bool, what: &str) {
    if !same {
        eprintln!(
            "stand: {}: the cold and warm calls differ ({what})",
            row.name
        );
    }
}

fn print_rows(readings: &[Reading]) {
    for reading in readings {
        let [cold, warm, resting, peak] = cells(reading);
        println!(
            "{:<58} cold={cold:>8} warm={warm:>8} resting={resting:>9} peak={peak:>9} {}",
            label(reading),
            reading.diagnostics(),
        );
    }
}

/// A row's four figures, as its console line and its table entry both state them.
fn cells(reading: &Reading) -> [String; 4] {
    [
        report::fmt_ms(reading.cold_ms),
        report::fmt_ms(reading.warm_ms),
        reading.memory.now_text(),
        reading.memory.peak_text(),
    ]
}

/// The row's label in the table.
fn label(reading: &Reading) -> &str {
    row(&reading.row).map_or(reading.row.as_str(), |row| row.label)
}

/// The record's caption: what the four figures are, and what the memory figures
/// are measured with — the basis read off the rows themselves.
fn caption(readings: &[Reading]) -> String {
    let basis = report::Basis::of(readings.iter().map(|reading| reading.memory.footprint));
    format!(
        "Each row is a product shape measured in a process of its own. Cold is the process with \
         no models loaded and the call that loads them; warm is the same call with the models \
         already resident. {}",
        basis.caption()
    )
}

/// The record's one table: a row per product shape and the four figures it carries.
fn table(readings: &[Reading]) -> String {
    let mut table = String::from("| row | cold | warm | resting | peak |\n|---|---|---|---|---|\n");
    for reading in readings {
        let [cold, warm, resting, peak] = cells(reading);
        table.push_str(&format!(
            "| {} | {cold} | {warm} | {resting} | {peak} |\n",
            label(reading),
        ));
    }
    table
}

/// The run's inputs, as bullets a reader can reproduce the rows from.
fn inputs_note() -> String {
    let mut note = String::new();
    for (what, path) in [
        ("models", loader::models_dir().ok()),
        ("voice", loader::voice_wav().ok()),
        ("corpus", corpus::dir().ok()),
    ] {
        if let Some(path) = path {
            note.push_str(&format!(
                "- {what}: `{}`\n",
                measurements::short_path(&path)
            ));
        }
    }
    note
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(row: &'static Row, memory: report::Memory) -> Reading {
        Reading {
            row: row.name.to_string(),
            cold_ms: 2_000.0,
            warm_ms: 1_000.0,
            memory,
            detail: run::detail([("regions", json!(3))]),
        }
    }

    /// A recognition row's label states the length of its own cut, so a changed cut
    /// cannot leave the record's label claiming another.
    #[test]
    fn a_recognition_label_states_its_own_cut() {
        for row in &ROWS {
            let Surface::Recognition {
                samples: Some(samples),
                ..
            } = row.surface
            else {
                continue;
            };
            let seconds = samples / transcribe::RATE as usize;
            assert!(
                row.label.contains(&format!("({seconds} s)")),
                "`{}` does not state its {seconds} s cut",
                row.label
            );
        }
    }

    /// The rows read a recording the corpus listing pins: a name that is not in the
    /// listing would be refused at run time, and one picked by position could move.
    #[test]
    fn the_recognition_rows_read_a_pinned_recording() {
        assert!(
            corpus::files()
                .iter()
                .any(|(path, ..)| *path == RECOGNITION_RECORDING),
            "{RECOGNITION_RECORDING} is not one of the pinned recordings"
        );
    }

    /// A name identifies one row: the table is looked up by it, so a duplicate
    /// would silently place a row's figures under another's label.
    #[test]
    fn a_name_identifies_one_row() {
        let mut names: Vec<&str> = ROWS.iter().map(|row| row.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ROWS.len());
        let mut labels: Vec<&str> = ROWS.iter().map(|row| row.label).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), ROWS.len());
    }

    #[test]
    fn a_reading_round_trips() {
        let printed = Reading {
            row: "synthesis.typical".to_string(),
            cold_ms: 12_345.6,
            warm_ms: 6_789.0,
            memory: report::Memory {
                now: Some(2 << 30),
                peak: Some(3 << 30),
                footprint: true,
            },
            detail: run::detail([("frames", json!(330)), ("digest", json!("deadbeef"))]),
        };
        let parsed =
            Reading::parse(&format!("setup line\n{}\n", printed.line())).expect("a reading");
        assert_eq!(parsed.row, printed.row);
        assert_eq!(parsed.cold_ms, printed.cold_ms);
        assert_eq!(parsed.warm_ms, printed.warm_ms);
        assert_eq!(parsed.memory.now, printed.memory.now);
        assert_eq!(parsed.memory.peak, printed.memory.peak);
        assert!(parsed.memory.footprint);
        assert_eq!(parsed.detail, printed.detail);
    }

    #[test]
    fn a_reading_missing_a_figure_is_refused() {
        // No `cold_ms`.
        assert!(
            Reading::parse(
                r#"{"row":"wake","warm_ms":2,"resting_bytes":3,"peak_bytes":4,"memory":"footprint"}"#
            )
            .is_err()
        );
        // A memory basis this binary never prints.
        assert!(
            Reading::parse(
                r#"{"row":"wake","cold_ms":1,"warm_ms":2,"resting_bytes":3,"peak_bytes":4,"memory":"physical"}"#
            )
            .is_err()
        );
        assert!(Reading::parse("nothing here").is_err());
    }

    /// A figure no platform reading supplied is a dash, never a zero; a figure that
    /// was read is stated under its own heading.
    #[test]
    fn the_table_states_every_figure_and_a_missing_one_as_a_dash() {
        let readings = [
            reading(
                &ROWS[0],
                report::Memory {
                    now: Some(2 << 30),
                    peak: Some(3 << 30),
                    footprint: true,
                },
            ),
            reading(
                &ROWS[1],
                report::Memory {
                    now: None,
                    peak: None,
                    footprint: false,
                },
            ),
        ];
        let table = table(&readings);
        let mut lines = table.lines();
        assert_eq!(lines.next(), Some("| row | cold | warm | resting | peak |"));
        assert_eq!(lines.next(), Some("|---|---|---|---|---|"));
        let body: Vec<&str> = lines.collect();
        assert_eq!(
            body[0],
            "| recognition, short voice message (10 s) | 2.00s | 1.00s | 2.00GiB | 3.00GiB |"
        );
        assert_eq!(
            body[1],
            "| recognition, typical voice message (30 s) | 2.00s | 1.00s | - | - |"
        );
    }
}
