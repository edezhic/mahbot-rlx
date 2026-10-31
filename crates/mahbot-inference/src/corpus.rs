// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pinned real-audio corpus — the one material the transcription measurements
//! read — and the corpus run that measures it.
//!
//! The listing and its SHA-256 pins are the wake bench's ([`crate::bench`]); this
//! module only resolves them against the machine's copy and checks a file against
//! its pin before it is read, so a row names a recording whose bytes are the ones
//! the record describes.
//!
//! The corpus run is its own mode, started by hand like the bench:
//! `cargo run --release -p mahbot-inference --features bench --bin
//! transcription-corpus`. One row per pinned recording, each in a process of its
//! own, and only a complete run inside the default bounds writes
//! `TRANSCRIPTION-CORPUS.md`. It takes the same single-instance heavy-run lock, so
//! two heavy runs cannot compete for the device.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use serde_json::json;

use crate::guard::Bounds;
use crate::measurements;
use crate::report::{self, Stopwatch};
use crate::run::{self, Reading};
use crate::transcribe::Transcriber;

/// Where the corpus lives on the machine that supplies it.
pub const ENV: &str = "MAHBOT_FAPH_CORPUS";

/// Marks a child of the corpus run: it measures the one row its argument names.
const ROW_CHILD: &str = "MAHBOT_CORPUS_ROW";

/// The pinned files, in the listing's fixed order: path, SHA-256, byte size.
pub(crate) fn files() -> &'static [(&'static str, &'static str, u64)] {
    crate::bench::REAL_AUDIO_FILES
}

/// The corpus directory, from [`ENV`] — no default: the checkout carries no audio.
pub(crate) fn dir() -> Result<PathBuf> {
    match std::env::var_os(ENV) {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir)),
        _ => bail!("{ENV} is not set — point it at the pinned real-audio corpus"),
    }
}

/// One recording's entry in the listing — its canonical name, SHA-256 and byte size —
/// or the reason the name is not in it. The one lookup over the pinned table, so a
/// name is matched and pinned by the same code.
fn pinned(name: &str) -> Result<&'static (&'static str, &'static str, u64)> {
    files()
        .iter()
        .find(|(path, _, _)| *path == name)
        .with_context(|| format!("`{name}` is not one of the pinned corpus recordings"))
}

/// Check one file against its pin: the size first, then the SHA-256.
pub(crate) fn verify_file(path: &Path, name: &str) -> Result<()> {
    let (_, sha256, bytes) = *pinned(name)?;
    let len = std::fs::metadata(path)
        .with_context(|| format!("cannot read the pinned corpus file {}", path.display()))?
        .len();
    if len != bytes {
        bail!(
            "the pinned corpus file {} is {len} bytes, not the pinned {bytes}",
            path.display()
        );
    }
    let actual = crate::measurements::sha256_file(path)?;
    if actual != sha256 {
        bail!(
            "the pinned corpus file {} hashes {actual}, the pin says {sha256}",
            path.display()
        );
    }
    Ok(())
}

/// The instrumented corpus run: the parent walks the listing, a child measures one
/// recording.
pub fn main_entry() {
    run::record_run(run::Mode {
        label: "transcription-corpus",
        record: measurements::Record::TranscriptionCorpus,
        marker: ROW_CHILD,
        total: files().len(),
        name: |index| files()[index].0,
        measure,
        print: print_rows,
        compose,
    });
}

/// Print one row per recording, refusing a row that lost its audio length.
fn print_rows(rows: &[Reading]) {
    if let Err(error) = print_table(rows) {
        eprintln!("transcription-corpus: {error:#}");
        std::process::exit(1);
    }
}

/// Measure one recording in this process: open the product, transcribe the file
/// cold (the process starts with no models loaded), then again warm.
fn measure(index: usize, bounds: Bounds) -> Result<Reading> {
    let (name, _, _) = files()[index];
    let path = dir()?.join(name);
    verify_file(&path, name)?;

    let t = Stopwatch::start();
    let mut transcriber = Transcriber::open_product(bounds.max_rss_mib)?;
    let cold = transcriber.transcribe_file(&path)?;
    let cold_ms = t.ms();
    let memory = report::Memory::take();

    let t = Stopwatch::start();
    let warm = transcriber.transcribe_file(&path)?;
    let warm_ms = t.ms();
    if cold.regions != warm.regions {
        eprintln!("transcription-corpus: {name}: the cold and warm calls differ (regions)");
    }
    Ok(Reading {
        row: name.to_string(),
        cold_ms,
        warm_ms,
        memory: memory.finish(),
        detail: run::detail([("audio_s", json!(warm.duration_s()))]),
    })
}

/// One recording's figures: its length in seconds, as its row carried it, and the
/// warm call's time per second of audio — the one formula the console and the
/// record both state.
fn figures(row: &Reading) -> Result<(f64, f64)> {
    let seconds = row
        .detail
        .get("audio_s")
        .and_then(serde_json::Value::as_f64)
        .filter(|seconds| *seconds > 0.0)
        .with_context(|| format!("`{}` carries no audio length", row.row))?;
    Ok((seconds, row.warm_ms / 1000.0 / seconds))
}

/// Print one row per recording. A row that lost its audio length fails the run,
/// like [`table`] — a rate against a length nobody measured may not be printed.
fn print_table(rows: &[Reading]) -> Result<()> {
    for row in rows {
        let (seconds, per_second) = figures(row)?;
        println!(
            "{} audio_s={seconds:.3} s_per_audio_s={per_second:.3} cold={} warm={} resting={} \
             peak={}",
            row.row,
            report::fmt_ms(row.cold_ms),
            report::fmt_ms(row.warm_ms),
            row.memory.now_text(),
            row.memory.peak_text(),
        );
    }
    Ok(())
}

/// The record's caption: what one call per recording costs, and what the memory
/// figures are measured with — the basis read off the rows themselves.
fn caption(rows: &[Reading]) -> String {
    let basis = report::Basis::of(rows.iter().map(|row| row.memory.footprint));
    format!(
        "The pinned real-audio corpus, one call per recording, each measured in a process of its \
         own. Time per second of audio is the warm call — the models already resident — divided \
         by the recording's own length. {}",
        basis.caption()
    )
}

/// The record's one table: a row per pinned recording and the figures it carries.
fn table(rows: &[Reading]) -> Result<String> {
    let mut table = String::from(
        "| recording | audio s | s per audio s | resting | peak |\n|---|---|---|---|---|\n",
    );
    for row in rows {
        let (seconds, per_second) = figures(row)?;
        table.push_str(&format!(
            "| `{}` | {seconds:.1} | {per_second:.3} | {} | {} |\n",
            row.row,
            row.memory.now_text(),
            row.memory.peak_text(),
        ));
    }
    Ok(table)
}

/// The record's body and its input bullets: one table, its caption, and the corpus
/// the recordings were read from.
fn compose(rows: &[Reading]) -> Result<(String, String)> {
    let table = table(rows)?;
    let inputs = dir().map_or_else(
        |_| String::new(),
        |dir| format!("- corpus: `{}`\n", measurements::short_path(&dir)),
    );
    Ok((format!("{}\n\n{}", table.trim_end(), caption(rows)), inputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(audio_s: f64, memory: report::Memory) -> Reading {
        Reading {
            row: files()[0].0.to_string(),
            cold_ms: 20_000.0,
            warm_ms: 12_720.0,
            memory,
            detail: run::detail([("audio_s", json!(audio_s))]),
        }
    }

    /// The record's table names the recording, its length, its time per audio
    /// second, and the two memory figures — and a figure nobody measured is a dash.
    #[test]
    fn the_table_states_the_rate_and_both_memory_figures() {
        let rows = [
            reading(
                318.0,
                report::Memory {
                    now: Some(2 << 30),
                    peak: Some(3 << 30),
                    footprint: true,
                },
            ),
            reading(
                318.0,
                report::Memory {
                    now: None,
                    peak: None,
                    footprint: false,
                },
            ),
        ];
        let table = table(&rows).expect("a table");
        let mut lines = table.lines();
        assert_eq!(
            lines.next(),
            Some("| recording | audio s | s per audio s | resting | peak |")
        );
        assert_eq!(lines.next(), Some("|---|---|---|---|---|"));
        let body: Vec<&str> = lines.collect();
        assert!(
            body[0].contains("`speech/librivox/speech-librivox-0027.wav`"),
            "{}",
            body[0]
        );
        assert!(body[0].contains("| 318.0 | 0.040 |"), "{}", body[0]);
        assert!(
            body[0].contains("2.00GiB") && body[0].contains("3.00GiB"),
            "{}",
            body[0]
        );
        assert!(body[1].ends_with("| - | - |"), "{}", body[1]);
    }
}
