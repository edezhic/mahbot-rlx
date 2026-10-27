// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stand: the live model paths from the compressed files to their answer, with
//! the measurement taken on the way.
//!
//! Prerequisites for a full run: `MAHBOT_MODELS_DIR` naming the GGUF pairs (the
//! transcription cell opens the audio tagger beside them), `MAHBOT_VOICE_WAV`
//! naming the 16 kHz voice asset, `MAHBOT_EMBED_TEXT` naming the article the
//! embedding cells read, and for the transcription cell the two corpus inputs —
//! `MAHBOT_FAPH_CORPUS` (the wake bench's own pinned real-audio subset) and
//! `MAHBOT_ASR_MUSIC` (the music pool) — plus the audio tagger beside the pair. A
//! run missing any of them does not fail: the transcription cell is not spawned,
//! the line `witness transcribe skipped=<input> reason=<why>` says which input is
//! absent (`corpus`, `music-pool`, `voice`, `models-dir`, `tagger`,
//! `recognition-pair`, each with its reason) and its row is carried like any other
//! the run did not take. The checkout carries no
//! models, no corpus and no audio, and the record pins their SHA-256s. The article is an input
//! like the voice: the text itself is not committed, and `reference::article`
//! refuses a file whose bytes do not hash to the pinned extract. `speak`'s
//! round-trip gate opens the recognition pair too, and without it the cell runs
//! and prints that the gate did not. The run rewrites `MEASUREMENTS.md`
//! unless `MAHBOT_MEASUREMENTS` names another path, and `MAHBOT_STAND_MAX_SWAP_MIB`
//! raises the swap ceiling the run is held to from its default.
//!
//! One entry point, no flags: an optional list of cell indices — `mahbot-inference
//! 7 6` — names the cells a run measures, and naming them writes their rows and
//! one `code` line. The acceptance walks every cell in its own process: the ear's 12 160-sample wake window (mel with the last frame dropped,
//! 76 frames, the 1.7B 24-block encoder, the projection, mean over ten tokens, L2,
//! cosine); 30 s of listening (1 746 mel frames, the 234-row prompt, at most
//! 2 048 new tokens on a cache that grows in 128-row buckets); 300 text tokens →
//! 330 codec rows (26.4 s) through the talker, predictor and sound half; and the
//! two embedding cells — `embed.256` and `embed.8192` — one pass each over the
//! whole sequence through EmbeddingGemma 2's text tower, pooled and normalised.
//! The transcription cell is the third surface: one pass over the pinned 33-file
//! corpus, the eight-clip music pool, constructed boundary material and the longest
//! pinned track, written as its own section of the record.
//!
//! It is a heavy run of this crate and carries the bench's three rules: one
//! instance at a time (`MAHBOT_LOCK`, defaulting to the temp directory), a
//! wall-clock stop (`MAHBOT_TIMEOUT_SECS`, 3 600 s) and a resident-set ceiling
//! (`MAHBOT_MAX_RSS_MIB`, 6 144 MiB). The parent holds the lock, and a cell that
//! outlives either limit is killed and the stand exits 75; the cell also carries
//! both limits itself, so a parent that dies leaves a bounded cell behind. A
//! cell that cannot build its models exits 1 with the reason on stdout.
//!
//! The unit paths here carry the per-phase instrumentation; everything that must
//! agree with the bench is shared rather than copied — the front end
//! (`listen::Front`), the talker's prompt (`synth::talker_prompt`) and the codec
//! windows and stitch (`graphs::windows`, `synth::codec_slice`,
//! `synth::stitch_window`). What stays here is the measurement around them:
//! the generation and decode loops and the per-window `Outcome`s the bench does
//! not report.
//!
//! A unit builds and compiles each graph by phase, immediately before that phase's
//! runs; there is no separate "build everything first" step (`graphs::Compiled`).
//! `build_ms` is the sum of those builds, `cold_ms` takes the first iteration of
//! each phase cold and `run_ms` sums the warm medians. The parent collects the
//! cells, runs each in its own process three times — once for transcription, which
//! measures a corpus rather than a hot loop — and prints a unit row for each
//! (cold/warm time, peak, map and their spreads) plus one `build_ms` line whose
//! value is the median of the three per-run sums over the cells that ran three
//! times. Everything goes to stdout; the child's `cell`/`phase`/`unit` lines are
//! forwarded with their `at=`/`load=`.
//!
//! A cell still prints its own diagnostic line (`key=value`):
//!
//! ```text
//! cell ear frames=76 rows=10 samples=12160 offset=16000 cosine=0.8755 dim=2048 blocks=24
//!      build_ms=336.4 cold_ms=38.4 run_ms=38.4 compile_ms=.. upload_ms=..
//!      first_ms=37.7 step_ms=37.8 min_ms=37.8 max_ms=38.2 passes=6 phase_passes=mel:6,encoder:6
//!      steps=.. kinds=.. digest=.. values=.. maxabs=.. nonfinite=0
//!      keys=none map=0.33GiB beyond_map=0.31GiB file_over_rss=0B arena=...
//! phase ear encoder iters=1 nodes=1108 first_ms=37.71 step_ms=37.85 … cold_ms=37.71 warm_ms=37.85
//! unit ear build_ms=336.361 run_ms=38.420 peak=564215808 cold_ms=38.444 map=355709344
//!      beyond_map=208506464 file_over_rss=0B submits=36 waits=36
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;

use crate::graphs::{DAC_MARGIN, DAC_WINDOW, FRONT_WINDOW, windows};
use crate::graphs::{Outcome, SampleSpec};
use crate::guard::{
    DEFAULT_MAX_RSS_MIB, DEFAULT_TIMEOUT_SECS, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock,
    Watch,
};
use crate::listen::{self, ASR_MAX_NEW, DECODE_CACHE_BUCKET, Front, l2, mean_pool};
use crate::loader::{Model, Pair, Role};
use crate::report::{
    digest, fmt_ms, human, keys, load_average, max_abs, non_finite, peak_rss, resident_rss, sci,
};
use crate::synth::{
    MAX_CODEC_ROWS as SPEAK_CODEC_ROWS, MAX_TEXT_TOKENS as SPEAK_TEXT_TOKENS, TTS_CODEC_EOS,
    TTS_REPEAT_PENALTY, TTS_TEXT_ROWS, TalkerPrompt, apply_repetition, codec_slice, stitch_window,
    suppress_service, talker_prompt,
};
use crate::tokenizer::Tokenizer;
use crate::{graphs, host, loader, music, report};

/// The ear's product input: 12 160 samples at 16 kHz, the mel front drops the
/// last frame, leaving 76 frames → 10 encoder tokens.
const EAR_FRAMES: usize = 76;
const EAR_SAMPLES: usize = 12_160;
/// The recognition product segment: one recognition pass's own length
/// ([`listen::PASS_SAMPLES`], 30 s at 16 kHz = 480 000 samples); the mel front
/// drops the last frame, leaving 3 000 frames. A shorter recording is trimmed to
/// its speech and measured on that fixed frame count instead.
const SPEECH_SAMPLES: usize = listen::PASS_SAMPLES;

/// The corpus input the wake bench and the transcription cell share: the pinned
/// real-audio subset, named once so the two surfaces cannot read different trees.
pub const CORPUS_ENV: &str = "MAHBOT_FAPH_CORPUS";

/// The synthesis text condition, tokenized by the file's own tokenizer; it is
/// long enough to fill all 300 tokens. `pub` for `tests/predictor_draft.rs`, which
/// reproduces the cell's pinned waveform through the library path.
pub const TTS_TEXT: &str = "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off, then I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.";
/// The wake product artifact: the 1.7B 24-block sound encoder and its body,
/// the same pair as recognition.
const WAKE_BODY: &str = "Qwen3-ASR-1.7B-Q4_K_M.gguf";
const WAKE_SOUND: &str = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf";
/// The live speech window in the recorded asset: the file is silent up to
/// ~1.16 s, so the ear starts where there is speech.
const SPEECH_OFFSET: usize = 16_000;
/// The recognition product segment's own recording: the FLEURS Russian clip
/// (17.46 s), right-padded with zeros to the fixed 30 s segment.
/// The ear's reference window, also speech (3.75 s into the same recording).
const EAR_TARGET_OFFSET: usize = 60_000;

/// A live cell of the standing configuration.
#[derive(Clone, Copy)]
enum Cell {
    Ear,
    Ear075,
    Ear2,
    Ear4,
    Listen,
    Speak,
    Embed256,
    Embed8192,
    Transcribe,
}

fn cells() -> Vec<Cell> {
    vec![
        Cell::Ear,
        Cell::Ear075,
        Cell::Ear2,
        Cell::Ear4,
        Cell::Listen,
        Cell::Speak,
        Cell::Embed256,
        Cell::Embed8192,
        Cell::Transcribe,
    ]
}

/// How many times a row is measured: three repetitions, so the report carries a
/// real spread.
const RUNS: usize = 3;

/// The number of times a cell is run: the transcription cell measures a corpus on
/// the inputs it is given, so one pass is its repetition — three would triple the
/// round for the same aggregates.
fn runs_of(cell: &Cell) -> usize {
    match cell {
        Cell::Transcribe => 1,
        _ => RUNS,
    }
}

/// A run whose transcription cell cannot be given its inputs drops that cell and keeps
/// the rest: the witness names the input that is missing and why, and the cell's row is
/// carried from the run that took it.
fn witness_absent(all: &mut Vec<(usize, Cell)>) {
    let Some((skipped, reason)) = transcribe_absent() else {
        return;
    };
    let index = cells()
        .iter()
        .position(|cell| matches!(cell, Cell::Transcribe))
        .unwrap_or(usize::MAX);
    println!(
        "witness transcribe skipped={skipped} reason={reason}: cell {index} not run, its row \
         carried"
    );
    all.retain(|(_, cell)| !matches!(cell, Cell::Transcribe));
}

/// Why the transcription cell cannot run on this machine, as the witness prints it
/// — `skipped=<the input> reason=<why>`. The cell reads the pinned real-audio
/// corpus, the music pool, the recognition pair, the audio tagger beside it and the
/// voice reference, so a run missing any of them would fail one cell and (with it)
/// the round — the tagger, say, only after every other cell had done its work. The two
/// pools are held to their pins here as well as in the cell, byte size and hash: a pool
/// that is present but not the pinned one is the same failure one step later. Every
/// other input is only decided on its existence.
fn transcribe_absent() -> Option<(&'static str, &'static str)> {
    let Some(corpus) = std::env::var_os(CORPUS_ENV)
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_dir())
    else {
        return Some(("corpus", "not-set"));
    };
    // The same pins the cell holds every file to, checked here: a corpus that is present
    // but does not match them would otherwise fail the cell after every other cell had
    // done its work, and take the round's numbers with it.
    if corpus_pins(&corpus).is_err() {
        return Some(("corpus", "not-the-pinned-files"));
    }
    if music::dir().is_err() {
        return Some(("music-pool", "not-set"));
    }
    // The pool is verified here rather than only read, for the same reason.
    if music::verify().is_err() {
        return Some(("music-pool", "not-the-pinned-files"));
    }
    if !loader::voice_wav().is_ok_and(|path| path.is_file()) {
        return Some(("voice", "not-set"));
    }
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(_) => return Some(("models-dir", "not-set")),
    };
    if !dir.join(crate::ced::CED_MODEL).is_file() {
        return Some(("tagger", "not-in-the-models-dir"));
    }
    match loader::pair_files(loader::Role::Recognition) {
        Some((body, sound)) if dir.join(body).is_file() && dir.join(sound).is_file() => None,
        Some(_) => Some(("recognition-pair", "not-in-the-models-dir")),
        None => Some(("recognition-pair", "not-pinned")),
    }
}

/// Every file the pinned real-audio corpus lists, present at its own byte size and its
/// own hash: one check, so the cell's gate and the witness before it cannot disagree.
fn corpus_pins(dir: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::Context as _;
    for (name, sha256, bytes) in crate::bench::REAL_AUDIO_FILES {
        let path = dir.join(name);
        let meta =
            std::fs::metadata(&path).with_context(|| format!("{} is missing", path.display()))?;
        if meta.len() != *bytes {
            anyhow::bail!(
                "{} is {} bytes, the pin says {bytes}",
                path.display(),
                meta.len()
            );
        }
        let actual = crate::measurements::sha256_file(&path)
            .with_context(|| format!("hashing {}", path.display()))?;
        if actual != *sha256 {
            anyhow::bail!("{} hashes {actual}, the pin says {sha256}", path.display());
        }
    }
    Ok(())
}

/// A cell that cannot continue prints why and ends its own process with a
/// non-zero status: the parent reads that status, so a failed cell cannot be
/// reported as a cell that exited zero.
macro_rules! cell_failed {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        std::process::exit(1);
    }};
}

/// The lock's path: the shared one every heavy run of this crate takes.
fn stand_lock() -> std::path::PathBuf {
    crate::guard::lock_path()
}

/// Set by the parent on a cell's own process: the parent holds the lock and kills
/// the cell at its own stop, so the child takes no lock of its own — but it does
/// carry the stop and the ceiling, in case the parent dies without killing it.
const CELL_CHILD: &str = "MAHBOT_CELL_CHILD";

/// The guards a spawned cell carries: the wall-clock stop and the resident-set
/// ceiling (the parent takes the shared lock itself). The watch is the same
/// [`Watch`] the bench installs, so a blocked device call cannot outlive either
/// limit on any entry point.
struct StandGuard {
    /// Dropped with the guard: the thread wakes within a second and exits.
    _watch: Watch,
    _lock: Option<RunLock>,
}

impl StandGuard {
    /// A cell the parent spawned. The parent holds the lock and kills the cell at
    /// its own stop, but the cell carries both limits itself as well: a parent
    /// that dies abnormally then leaves a cell that still ends at its ceiling or
    /// its stop, instead of a multi-gigabyte process with nothing over it.
    fn child() -> Self {
        let deadline = Deadline::new(limit("MAHBOT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS));
        let ceiling = MemoryCeiling::new(crate::guard::max_rss_mib());
        Self::watched(None, deadline, ceiling)
    }

    fn watched(lock: Option<RunLock>, deadline: Deadline, ceiling: MemoryCeiling) -> Self {
        let watch = match Watch::install(
            "stand",
            deadline,
            ceiling,
            || "stand",
            |abort| {
                eprintln!("stand: {abort} — stopping");
                std::process::exit(EXIT_STOPPED);
            },
        ) {
            Ok(watch) => watch,
            Err(error) => {
                eprintln!("stand: cannot start the guard thread: {error:#}");
                std::process::exit(1);
            }
        };
        Self {
            _watch: watch,
            _lock: lock,
        }
    }
}

/// The lock and the two limits every heavy run of the stand carries — the parent
/// and a cell started on its own alike — or the message the run exits with.
fn heavy_run() -> (RunLock, Deadline, MemoryCeiling) {
    let lock = match RunLock::acquire(&stand_lock()) {
        Ok(lock) => lock,
        Err(error) => {
            eprintln!("stand: {error:#}");
            std::process::exit(1);
        }
    };
    let deadline = Deadline::new(limit("MAHBOT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS));
    let ceiling = MemoryCeiling::new(crate::guard::max_rss_mib());
    (lock, deadline, ceiling)
}

/// A guard limit from the environment, or the crate default.
fn limit(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// One cell's own process, held to the stand's stop: a cell that outlives the
/// wall clock or the resident-set ceiling is killed instead of waited for.
fn run_cell(
    exe: &std::path::Path,
    index: usize,
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> std::process::Output {
    if let Err(error) = deadline
        .check("stand")
        .and_then(|()| ceiling.check("stand"))
    {
        eprintln!("stand: {error:#}");
        std::process::exit(EXIT_STOPPED);
    }
    let mut child = match std::process::Command::new(exe)
        .arg(index.to_string())
        .env(CELL_CHILD, "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            eprintln!("stand: cannot start cell {index}: {error}");
            std::process::exit(1);
        }
    };
    let mut ticks = 0u32;
    loop {
        let status = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                eprintln!("stand: cannot wait for cell {index}: {error}");
                std::process::exit(1);
            }
        };
        match status {
            Some(_) => {
                return match child.wait_with_output() {
                    Ok(out) => out,
                    Err(error) => {
                        eprintln!("stand: cannot read cell {index}'s output: {error}");
                        std::process::exit(1);
                    }
                };
            }
            None if deadline.expired() => {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!("stand: cell {index} outlived the {}s stop", deadline.secs());
                std::process::exit(EXIT_STOPPED);
            }
            None => {
                // The ceiling is a reading of the cell, not of this parent.
                ticks += 1;
                if ticks.is_multiple_of(2) {
                    let rss = child_rss(child.id());
                    if rss > ceiling.rss_bytes() {
                        let _ = child.kill();
                        let _ = child.wait();
                        eprintln!(
                            "stand: cell {index} reached {} MiB, over the {} MiB ceiling",
                            rss / (1024 * 1024),
                            ceiling.rss_bytes() / (1024 * 1024)
                        );
                        std::process::exit(EXIT_STOPPED);
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

/// A process's resident set, in bytes, from `ps` (`0` where it has no reading).
fn child_rss(pid: u32) -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// The stand's entry point: the parent walks the closed cell list, a child
/// re-entered with one cell index runs that cell.
pub fn main_entry() {
    // The parent walks the closed list; the child (one cell) is this same binary
    // with the cell's position and nothing else — the list is the only thing that
    // can be named. Anything else is refused before any work starts: an argument
    // this binary does not understand must not launch the whole acceptance.
    //
    // Named indices may name more than one cell: a run that measures several at
    // once writes each of their rows under one `code` line, which is what makes a
    // figure's provenance the digest of the tree that produced it rather than of a
    // later run that carried it.
    let mut named: Vec<usize> = Vec::new();
    for arg in std::env::args().skip(1) {
        let Ok(i) = arg.parse::<usize>() else {
            eprintln!(
                "stand: `{arg}` is not a cell index — the stand takes cell indices and no flags"
            );
            std::process::exit(2);
        };
        // A repeated index would measure one cell twice and still write one row.
        if named.contains(&i) {
            eprintln!("stand: cell {i} is named twice");
            std::process::exit(2);
        }
        named.push(i);
    }
    // Every named index must exist before anything runs.
    let count = cells().len();
    for i in &named {
        if *i >= count {
            println!("cell {i} — there are {count} cells");
            std::process::exit(2);
        }
    }
    // A child measures exactly one cell and returns: the parent spawns one per
    // index and owns the limits and the record, so two cells in one process would
    // put the first one's residency inside the second's peak.
    if std::env::var_os(CELL_CHILD).is_some() {
        let [index] = named[..] else {
            eprintln!(
                "stand: {CELL_CHILD} takes one cell index and {} are named — run the cells one at \
                 a time, or name them without the marker",
                named.len()
            );
            std::process::exit(2);
        };
        let _guard = StandGuard::child();
        run(cells()[index], index);
        return;
    }

    // The stand is the crate's other multi-gigabyte, tens-of-minutes run, so it
    // carries the bench's three rules: one instance at a time, a wall-clock stop
    // and a resident-set ceiling. The parent holds the lock and watches the run's
    // clock and each cell's resident set; the cell carries the stop and the
    // ceiling itself as well, so it stays bounded even if the parent dies.
    let (lock, deadline, ceiling) = heavy_run();
    // The load the table's figures were taken under: the cells print it per run,
    // and the record keeps the range so a reader can tell a quiet run from one
    // that shared the machine.
    let load_at_start = load_average();
    // The record carries the swap the run actually saw, not only the ceiling.
    let swap_at_start = crate::guard::swap_used_bytes();
    // The machine's streaming ceiling, measured here and now: the reference every
    // per-phase weight-stream rate below is reported against, because a rate on
    // its own says nothing about how far the kernel is from what the memory system
    // delivers on this hardware. Read again after the cells, so the record carries
    // the probe's own reading-to-reading spread and not one session's figure.
    let ceiling_streams = probe_ceiling();
    print_ceiling("ceiling", &ceiling_streams);
    eprintln!(
        "stand: stop {}s · ceiling {} MiB · lock {}",
        deadline.secs(),
        ceiling.rss_bytes() / (1024 * 1024),
        stand_lock().display()
    );
    let _lock = lock;

    // The acceptance units, each in its own process: one per row of the record's
    // table — the ear and its three batch variants, listen, speak, the two
    // embedding shapes and transcription. Each unit is run [`runs_of`] times so the
    // report carries a real spread; the child's own `cell`/`phase`/`unit` lines are
    // forwarded with their `at=`/`load=`.
    // The cell's position in `cells()` and the cell itself: the position is what
    // the child is spawned with, so a run cut down to one cell still names the
    // cell the record and the argument speak of.
    let all: Vec<(usize, Cell)> = if named.is_empty() {
        let mut all: Vec<(usize, Cell)> = cells().into_iter().enumerate().collect();
        // A cell whose inputs are not on this machine is not spawned: the record
        // carries its row and the other cells' figures are not lost with it.
        witness_absent(&mut all);
        all
    } else {
        // In the order named: the record's rows are written in the table's own
        // order whatever this is, so a caller may name them for the run alone. A named
        // cell runs on the caller's assertion that its inputs are there — unless one of
        // them is not even present, which is the one failure that would take every other
        // named cell's figures with it, so the same witness keeps the round whole and
        // says why.
        let positions = cells();
        let mut all: Vec<(usize, Cell)> = named.into_iter().map(|i| (i, positions[i])).collect();
        if all.iter().any(|(_, cell)| matches!(cell, Cell::Transcribe)) {
            witness_absent(&mut all);
        }
        all
    };
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            eprintln!("stand: cannot find this binary: {error}");
            std::process::exit(1);
        }
    };
    struct Row {
        name: &'static str,
        /// `(cold, warm, peak, map, build, loop, sound)` per run. The split is
        /// kept per run so the row reports the split of the same run its warm
        /// figure came from; mixing the medians of two repetitions would print a
        /// loop and a sound figure that do not add up to the row's warm one.
        got: Vec<(f64, f64, u64, u64, f64, f64, f64)>,
        /// The per-phase weight streams, one `(phase, unit, bytes, ms)` per run —
        /// the same run-indexed form as `got`, for the same reason.
        streams: Vec<Vec<(&'static str, &'static str, f64, f64)>>,
        arena: String,
        digest: String,
        speech_s: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    // The transcription surface's rows, straight from that cell's stdout: the
    // section is written from them, after the stand's own, by the run that took
    // them (`measurements::refresh("transcribe", …)`).
    let mut transcribe_rows: Vec<String> = Vec::new();
    for (i, cell) in &all {
        let name = unit_name(cell);
        let runs = runs_of(cell);
        let mut got = Vec::with_capacity(runs);
        let mut streams = Vec::with_capacity(runs);
        let mut arena = String::new();
        let mut digest = String::new();
        let mut speech_s = f64::NAN;
        for _ in 0..runs {
            let out = run_cell(&exe, *i, deadline, ceiling);
            let text = String::from_utf8_lossy(&out.stdout);
            // The transcription cell's rows are its own aggregate lines, not a
            // unit line: they are kept for the section this run writes below.
            if matches!(cell, Cell::Transcribe) {
                transcribe_rows.extend(
                    text.lines()
                        .filter(|line| line.starts_with("transcribe "))
                        .map(str::to_string),
                );
            }
            streams.push(streams_of(name, &text));
            if let Some(value) = text
                .split("arena=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
            {
                arena = value.to_string();
            }
            if let Some(value) = text
                .split(" digest=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
            {
                digest = value.to_string();
            }
            // The row's own line, not the first `speech_s=` in the child's output:
            // the transcription cell prints it on three of them, and the table's
            // `speech s` column is the unit row's.
            if let Some(value) = unit_line(&text, name).and_then(|line| field(line, "speech_s")) {
                speech_s = value;
            }
            // A cell with no synthesis split prints no `loop_ms`/`sound_ms`;
            // absent is `NaN`, which the table renders as `-`.
            let loop_ms = field(&text, "loop_ms").unwrap_or(f64::NAN);
            let sound_ms = field(&text, "sound_ms").unwrap_or(f64::NAN);
            // Keep the child's numbers in the normal run: dropping them is what
            // used to hide the window and the load each number was taken at.
            print!("{text}");
            match out
                .status
                .success()
                .then(|| parse_unit(&text, name))
                .flatten()
            {
                Some(v) => got.push((v.0, v.1, v.2, v.3, v.4, loop_ms, sound_ms)),
                None => {
                    eprint!("{}", String::from_utf8_lossy(&out.stderr));
                    eprintln!(
                        "cell {i} {name} — no unit line; the cell's own process {}",
                        out.status
                    );
                    std::process::exit(1);
                }
            }
        }
        rows.push(Row {
            name,
            got,
            streams,
            arena,
            digest,
            speech_s,
        });
    }
    let mut build_runs = vec![0.0f64; RUNS];
    for row in &rows {
        // A one-pass cell's build is its own and is not a third of a repetition:
        // summing it here would fold a corpus into the median of a hot loop.
        if row.got.len() != RUNS {
            continue;
        }
        for (run, v) in row.got.iter().enumerate() {
            if run < build_runs.len() {
                build_runs[run] += v.4;
            }
        }
    }
    println!(
        "# cold/warm run_ms, seconds per second of speech, the synthesis loop/sound split, \
         peak/map and their spreads"
    );
    let mut lines: Vec<(&'static str, String)> = Vec::new();
    for row in &rows {
        let (name, got, arena, digest) = (row.name, &row.got, &row.arena, &row.digest);
        let warm: Vec<f64> = got.iter().map(|v| v.1).collect();
        let cold: Vec<f64> = got.iter().map(|v| v.0).collect();
        let peak: Vec<f64> = got
            .iter()
            .map(|v| v.2 as f64 / (1u64 << 30) as f64)
            .collect();
        let map = got[0].3 as f64 / (1u64 << 30) as f64;
        let spreads = |v: &[f64]| {
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let (cold_ms, warm_ms, peak_gib) = (
            median(&cold).unwrap_or(0.0),
            median(&warm).unwrap_or(0.0),
            median(&peak).unwrap_or(0.0),
        );
        let build_ms = median(&got.iter().map(|v| v.4).collect::<Vec<_>>()).unwrap_or(0.0);
        // The split is the one of the run whose warm figure is the row's: the two
        // then add up by construction.
        let median_run = median_index(&warm)
            .and_then(|i| got.get(i))
            .map(|v| (v.5, v.6))
            .unwrap_or((f64::NAN, f64::NAN));
        // The cell's own number: warm seconds per second of speech, and the
        // denominator printed beside it so the rate re-derives from the table. The
        // two surfaces are the only rows with a denominator; the others print `-`.
        let secs_of_speech = row
            .speech_s
            .is_finite()
            .then_some(row.speech_s)
            .filter(|s| *s > 0.0);
        let per_s = secs_of_speech.map_or(f64::NAN, |s| warm_ms / 1000.0 / s);
        let split = |v: f64| {
            if v.is_finite() {
                format!("{v:.1}")
            } else {
                "-".to_string()
            }
        };
        let per_s_cell = if per_s.is_finite() {
            format!("{per_s:.3}")
        } else {
            "-".to_string()
        };
        // The denominator itself, in the same number the rate beside it used.
        let speech_cell = secs_of_speech.map_or_else(|| "-".to_string(), |s| format!("{s:.3}"));
        println!(
            "{name} cold={cold_ms:.1} warm={warm_ms:.1} s_per_s={per_s_cell} loop_ms={} sound_ms={} peak={peak_gib:.2} map={map:.2} cold_spread={:.1} warm_spread={:.1} peak_spread={:.3} at={} load={}",
            split(median_run.0),
            split(median_run.1),
            spreads(&cold),
            spreads(&warm),
            spreads(&peak),
            now(),
            load_average(),
        );
        lines.push((
            name,
            format!(
                "| {name} | {cold_ms:.1} | {warm_ms:.1} | {speech_cell} | {per_s_cell} | {} | {} | {build_ms:.1} | {peak_gib:.2} GiB | {map:.2} GiB | {arena} | {digest} |\n",
                split(median_run.0),
                split(median_run.1),
            ),
        ));
    }
    // A run that measures a subset of the cells rewrites only those rows. The
    // rest are carried from the block this file already holds: they were taken by
    // an earlier run against inputs this one does not open, so dropping them would
    // lose the only record of them, and re-taking them here is impossible. The
    // `rows` line under Provenance names which is which.
    let mut carried = carried_block();
    let mut table = String::from(
        "| unit | cold ms | warm ms | speech s | s/speech s | loop ms | sound ms | build ms | peak | map | arena | digest |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut measured: Vec<&'static str> = Vec::new();
    let mut carried_rows = 0usize;
    for name in cell_names() {
        match lines.iter().find(|(unit, _)| *unit == name) {
            Some((unit, line)) => {
                table.push_str(line);
                measured.push(unit);
            }
            None => {
                if let Some(line) = carried.rows.get(name) {
                    table.push_str(line);
                    carried_rows += 1;
                }
            }
        }
    }
    let build_spread = build_runs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - build_runs.iter().copied().fold(f64::INFINITY, f64::min);
    let build_ms = median(&build_runs).unwrap_or(0.0);
    println!(
        "build_ms {build_ms:.1} build_spread={build_spread:.1} at={} load={}",
        now(),
        load_average()
    );

    // The probe again, after the cells: two readings of one run bound the
    // probe's own spread, which a single reading cannot. A diagnostic must not
    // discard a run whose cells and witnesses all succeeded — it allocates ~130 MiB
    // while the process holds ~2.6 GiB — so a panic reads as its own outcome
    // (`panicked`), never as a missing adapter.
    let ceiling_after = probe_ceiling();
    print_ceiling("ceiling_after", &ceiling_after);

    // The repository's own record, refreshed by the run that produced it.
    let mut body = String::new();
    body.push_str("## Per-model measurement (stand)\n\n");
    body.push_str(&format!("_Measured {}._\n\n", crate::measurements::now()));
    body.push_str(&table);
    body.push_str(
        "\nThe cell's number is the **`s/speech s`** column: the **warm** figure \
         (`warm ms`) divided by the `speech s` column — the cell's own seconds of \
         speech, `samples / 16 000` for recognition and `rows · 1 920 / 24 000` for \
         synthesis, printed so the rate re-derives from this table alone. `loop ms` is the \
         synthesis generation loop (speaker, prefill, body, predictor, sampler) and \
         `sound ms` is the sound half; their sum is `warm ms`, so a ×N synthesis \
         claim reads off both. Both are the same phase sums the cell's \
         `loop_ms`/`sound_ms` tokens record; `speak`'s `sound_phase_ms` field is \
         the sound graphs' summed `step_ms` — the same sum the `sound ms` column \
         is taken from, carried as a field so a cell line is self-describing. \
         For `speak` the recorded `digest` is the **waveform** digest, not the \
         `prompt_digest` its own line prints. `-` is a cell with no speech \
         denominator. Each cell line \
         prints its own `warm_spread` (the warm runs' max − min, in ms). `peak` is the \
         cell process's own high-water (`ru_maxrss`, which no later drop can lower) and \
         each cell is its own process, so the figure is the maximum that unit's own \
         process reached — what it held when it peaked, not what is live when the line \
         prints. For `speak` the reading is taken before the round-trip gate loads the \
         recogniser, so the column excludes it (the gate prints its own post-gate \
         reading as `speak roundtrip peak_mb`). Across this machine's records the \
         column moves between runs of one tree, and the summary line carries the spread \
         each run saw as `peak_spread`. \
         `embed.256` and `embed.8192` are one bidirectional pass each over the whole \
         sequence on a Q8_0 stack. Their `digest` is over the L2-normalised \
         768-vector the row names — the cell's answer, not the graph's raw output.\n",
    );
    // The pinned answers themselves, in the record and not only in the run's
    // stdout: the sentences the two generative surfaces are held to, and the fact
    // that the comparison is on the words rather than on a digest that an unrelated
    // drift could leave intact — the listen transcript within
    // `LISTEN_MOVED_WORDS` recorded, its digest a witness, the round trip exact.
    body.push_str(&format!(
        "\nThe pinned answers: recognition is recorded against `{}` — compared word \
         for word, at most {} of them different, its digest and generated-token count \
         printed as witnesses with what moved, and the run carries on; the synthesis \
         round trip — the waveform the `speak` row's digest names, resampled to 16 kHz \
         and read back through the recogniser on its raw samples — must reproduce \
         `{}` exactly.\n",
        crate::reference::LISTEN_TRANSCRIPT,
        crate::reference::LISTEN_MOVED_WORDS,
        crate::reference::SPEAK_TRANSCRIPT,
    ));
    // The per-phase weight streams, next to the machine's own streaming ceiling:
    // the figure the synthesis and recognition targets are stated in, because a
    // rate is only meaningful against what the same machine delivers on a plain
    // stream. Every row is one run's own pair of a byte count and a time — the run
    // whose warm figure is the row's — so the row's arithmetic is that division.
    //
    // Only the rows are carried, never a paragraph of the earlier run's: the
    // ceiling above them is this run's own probe on this run's device, and every
    // percentage is taken against it here — so a carried row and a measured one
    // cannot be denominated by two ceilings, and a second run cannot stack a note
    // on a note.
    let mut streams_body = String::new();
    match &ceiling_streams {
        Ok(c) => {
            // The probe's second reading is a diagnostic, and it must not read as
            // "not re-taken" when it failed: its own outcome — a missing adapter, a
            // device fault, a panic — is printed in the record.
            let after = match &ceiling_after {
                Ok(a) => format!(
                    " The same probe after the cells read {:.1} GB/s ({:.1}–{:.1}), which is \
                     the reading-to-reading spread to hold the figure to.",
                    a.read_gb_s, a.read_min_gb_s, a.read_max_gb_s
                ),
                Err(failure) => format!(
                    " The same probe after the cells produced no figure — {}, so the \
                     before reading stands alone.",
                    failure_reading(failure),
                ),
            };
            streams_body.push_str(&format!(
                "Ceiling, measured on this run's device before the cells: **{:.1} GB/s** on a \
                 streaming read ({:.1}–{:.1} over {} rounds) and {:.1} GB/s of total traffic on a \
                 streaming copy ({:.1}–{:.1}). The read figure is the one the rows below are \
                 reported against, because a weight is read and never written; the copy figure is \
                 the same probe's other arm, recorded beside it. Both come from \
                 `rlx_wgpu::bandwidth::stream_ceiling`: a {:.1} MiB vectorised read and a {:.1} MiB \
                 vectorised copy, on `{}`.{after}\n\n",
                c.read_gb_s,
                c.read_min_gb_s,
                c.read_max_gb_s,
                c.rounds,
                c.gb_s,
                c.min_gb_s,
                c.max_gb_s,
                c.bytes_per_direction as f64 / (1u64 << 20) as f64,
                c.bytes_per_direction as f64 / (1u64 << 20) as f64,
                crate::measurements::device(),
            ));
        }
        Err(failure) => streams_body.push_str(&format!(
            "Ceiling: the probe produced no figure — {}, so the rows below carry no \
             denominator.\n\n",
            failure_reading(failure),
        )),
    }
    streams_body.push_str(
        "| phase | unit | weight bytes/unit | ms/unit | GB/s | of the read ceiling |\n\
         |---|---|---|---|---|---|\n",
    );
    let ceiling_gb_s = ceiling_streams.as_ref().ok().map(|c| c.read_gb_s);
    let mut measured_streams: Vec<StreamRow> = Vec::new();
    for row in &rows {
        let Some(run) = median_index(&row.got.iter().map(|v| v.1).collect::<Vec<_>>())
            .and_then(|i| row.streams.get(i))
        else {
            continue;
        };
        for (phase, unit, bytes, ms) in run {
            // The rate is the division of the two figures printed beside it, so
            // both are rounded to the precision they are written at before the
            // division; the percentage then divides by the ceiling printed above.
            // A reader can reproduce every figure from the table.
            let ms = printed(*ms, 2);
            let rate = printed(bytes / (ms * 1e-3) / 1e9, 1);
            measured_streams.push(StreamRow {
                unit: (*unit).to_string(),
                columns: format!(
                    "| {phase} | {unit} | {} | {ms:.2} | {rate:.1} |",
                    report::grouped(*bytes as u64)
                ),
            });
        }
    }
    // The rows, keyed by the `unit` they are reported for: a run that measures one
    // generative cell replaces that row and keeps the others, in the order the
    // record already holds them in.
    let measured_count = measured_streams.len();
    let table_rows = merge_stream_rows(std::mem::take(&mut carried.streams), measured_streams);
    let carried_any = table_rows.len() > measured_count;
    for row in &table_rows {
        streams_body.push_str(&stream_row(&row.columns, ceiling_gb_s));
    }
    streams_body.push_str(
        "\nEach row's numerator is the graph's own parameter bytes — packed and f32 — \
         which its multiplies and convolutions read once per unit, and only operands that \
         are parameters count; a graph that read one of them twice per unit would move \
         more, and one that read a slice would move less. `talker body` is one frame, `code \
         predictor` one of a frame's sixteen projecting levels (the frame's stream and time \
         divided by the levels it walks), `sound half` the utterance's windows summed with \
         the graph each window actually ran, and `recognition decode` one generated token. \
         The figure in each row is the run whose warm figure is the table's.\n",
    );
    if carried_any && measured_count > 0 {
        streams_body.push_str(
            "\nA row this run did not measure is carried from the run that took it — its `ms` \
             and `GB/s` are that run's — and only its percentage is taken against the ceiling \
             above, which is this run's probe.\n",
        );
    }
    body.push_str("\n### Weight streams, against the machine's streaming ceiling\n\n");
    if measured_count == 0 && !table_rows.is_empty() {
        body.push_str(
            "This run measured no generative cell, so it took no stream reading of its own: \
             the rows below are the run that measured them, restated against the ceiling \
             printed below.\n\n",
        );
    }
    body.push_str(&streams_body);
    body.push_str("\nProvenance:\n\n");
    // A row this run did not take is carried, and the tree that took it is in no
    // line of this file — the `code` line is this run's, rewritten on every run —
    // so the line names which rows are the run's rather than placing the others.
    body.push_str(&provenance(
        &load_at_start,
        swap_at_start,
        &format!(
            "this run measured {}{}",
            measured.join(", "),
            if carried_rows > 0 {
                "; every other row is carried from an earlier run of this file, and its figures \
                 belong to the run that took them — the `code` line above is this run's tree, not \
                 theirs. A carried row keeps its weight-stream row and the artefact lines of the \
                 run that took it"
            } else {
                "; the table is this run's throughout"
            },
        ),
    ));
    // The invocation that took these rows, cell indices and all: a fixed string
    // here would name a run nobody made, and a row's provenance is the command
    // that produced it.
    let mut argv = std::env::args();
    let exe = argv.next().unwrap_or_default();
    let exe = std::path::Path::new(&exe)
        .file_name()
        .map_or(exe.clone(), |name| name.to_string_lossy().into_owned());
    let command = argv.fold(exe, |mut line, arg| {
        line.push(' ');
        line.push_str(&arg);
        line
    });
    body.push_str(&format!(
        "- command: `{command}` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`, article from `MAHBOT_EMBED_TEXT`)\n"
    ));
    // The artefact lines this run can take, by the file name each line names: an
    // artefact this run cannot read keeps the line the earlier run wrote for it,
    // beside the rows it belongs to.
    let mut artefacts: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    if let Ok(dir) = crate::loader::models_dir() {
        for (_, body_name, sound_name) in crate::loader::PAIRS {
            for name in [*body_name, *sound_name] {
                if let Some((bytes, sha)) = crate::measurements::digest(&dir.join(name)) {
                    body.push_str(&format!(
                        "- model: `{name}` · {bytes} bytes · sha256 {sha}\n"
                    ));
                    artefacts.insert(name.to_string());
                }
            }
        }
        // The embeddings role is one file rather than a pair, so `PAIRS` cannot
        // name it: same form, its own line, and the source the numbers belong to
        // when the file's own hash is that source's.
        let name = crate::loader::EMBED_MODEL;
        if let Some((bytes, sha)) = crate::measurements::digest(&dir.join(name)) {
            let (url, revision, pin) = crate::loader::EMBED_MODEL_SOURCE;
            let source = if sha == pin {
                format!("url {url} · revision {revision}")
            } else {
                format!(
                    "not the artefact of record (the record's is url {url} · revision {revision} \
                     · sha256 {pin})"
                )
            };
            body.push_str(&format!(
                "- model: `{name}` · {bytes} bytes · sha256 {sha} · {source}\n"
            ));
            artefacts.insert(name.to_string());
        }
    }
    if let Ok(voice) = crate::loader::voice_wav()
        && let Some((bytes, sha)) = crate::measurements::digest(&voice)
    {
        // By bare file name, like the model files above and the bench's section:
        // one form for the field across both halves of the record.
        let name = voice
            .file_name()
            .unwrap_or(voice.as_os_str())
            .to_string_lossy();
        body.push_str(&format!(
            "- voice: `{name}` · {bytes} bytes · sha256 {sha}\n"
        ));
        artefacts.insert(name.to_string());
    }
    // The embedding cells' text: an input the checkout cannot carry, so the
    // record names it the way it names the voice file and pins what was read —
    // the article, the query that reproduces it, and the two cuts the cells
    // measure. Read through [`crate::reference::article`], the same check the
    // cells themselves use, so the line cannot name a file the cells refused, and
    // built before the carried lines so this run's text replaces an earlier run's
    // rather than joining it: there is one article, whatever it is named.
    let text_line = crate::reference::article().ok().and_then(|text| {
        let path = crate::loader::embed_text().ok()?;
        let name = path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy();
        // A cut that does not reproduce says why: the two cuts are pinned
        // separately from the article, so a drift in one is not a drift in the
        // other and the record must not fold the reason into "not reproducible".
        let cut = |rows: usize| match crate::reference::cell_text(&text, rows) {
            Ok(cut) => format!(
                "{} bytes, sha256 {}",
                cut.len(),
                crate::reference::sha256_hex(cut.as_bytes())
            ),
            Err(reason) => format!("not the pinned cut: {reason}"),
        };
        Some(format!(
            "- text: `{name}` · {} bytes · sha256 {} · {} (revision {}) · fetched with \
             `{}revids={}` · cells: `embed.8192` first {} words ({}) and `embed.256` {} words \
             from `{}` ({}) · the text itself is not committed\n",
            text.len(),
            crate::reference::sha256_hex(text.as_bytes()),
            crate::reference::ARTICLE_URL,
            crate::reference::ARTICLE_REVISION,
            crate::reference::ARTICLE_FETCH,
            crate::reference::ARTICLE_REVISION,
            crate::reference::CELL_8192_WORDS,
            cut(8_192),
            crate::reference::CELL_256_WORDS,
            crate::reference::CELL_256_ANCHOR,
            cut(256),
        ))
    });
    // The artefact lines this run did not emit, deduplicated by the file name in
    // the line: the models directory here holds the embedding tower but not the
    // ASR/TTS pairs, and those lines belong with the rows they took.
    for line in &carried.artefacts {
        if line.starts_with("- text: ") && text_line.is_some() {
            continue;
        }
        let Some(name) = artefact_name(line) else {
            continue;
        };
        if artefacts.insert(name.to_string()) {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(line) = &text_line {
        body.push_str(line);
    }
    if let Err(error) = crate::measurements::refresh("stand", &body) {
        // The numbers are the deliverable, and the file is where they live: a
        // run that cannot record them is a failed run, as the bench's is.
        eprintln!("stand: cannot refresh MEASUREMENTS.md: {error:#}");
        std::process::exit(1);
    }
    // The transcription surface's own section, after the stand's and only when
    // this round actually ran that cell: a round that named other cells carries
    // the section the run that took it wrote. Its provenance is the same run's, so
    // its figures are attributed to the same tree.
    if !transcribe_rows.is_empty() {
        let body = match transcribe_section(
            &transcribe_rows,
            &provenance(
                &load_at_start,
                swap_at_start,
                "every row above is this run's; the model files, the corpus and the music pool are \
                 verified by hash before the first one, and the `model:` lines below are what those \
                 files hashed to here",
            ),
        ) {
            Ok(body) => body,
            // The section is written from the cell's own rows: a row or a field it names
            // and the run did not print is a record that would state a figure nobody
            // measured, which is a failed run rather than a written zero.
            Err(error) => {
                eprintln!("stand: cannot write the transcription section: {error:#}");
                std::process::exit(1);
            }
        };
        if let Err(error) = crate::measurements::refresh("transcribe", &body) {
            eprintln!("stand: cannot refresh MEASUREMENTS.md: {error:#}");
            std::process::exit(1);
        }
    }
}

/// The units of the acceptance, in report order.
fn unit_name(cell: &Cell) -> &'static str {
    match cell {
        Cell::Ear => "ear",
        Cell::Ear075 => "ear.075",
        Cell::Ear2 => "ear.batch2",
        Cell::Ear4 => "ear.batch4",
        Cell::Listen => "listen",
        Cell::Speak => "speak",
        Cell::Embed256 => "embed.256",
        Cell::Embed8192 => "embed.8192",
        Cell::Transcribe => "transcribe",
    }
}

/// The provenance block both surfaces' sections carry: the machine, the device and
/// the tree the figures were taken from, which rows this run took, and the
/// conditions of the run. One function, so the two sections cannot describe
/// themselves differently.
fn provenance(load_at_start: &str, swap_at_start: Option<u64>, rows: &str) -> String {
    let mut body = String::new();
    body.push_str(&format!("- machine: {}\n", crate::measurements::machine()));
    body.push_str(&format!("- device: {}\n", crate::measurements::device()));
    body.push_str(&format!("- driver: {}\n", crate::measurements::driver()));
    body.push_str(&format!("- profile: {}\n", crate::measurements::profile()));
    body.push_str(&format!(
        "- code: {} (the commit and the SHA-256 of the tree's own difference from it — {} — which \
         together are the content that produced these figures; a clean tree's digest is \
         `e3b0c44298fc1c14`, the hash of an empty diff)\n",
        crate::measurements::code_state(),
        crate::measurements::provenance_rule(),
    ));
    body.push_str(&format!("- rows: {rows}\n"));
    body.push_str(&format!(
        "- thermal: {} (`pmset -g therm`, the reading `scripts/check-throttle.sh` gates on)\n",
        crate::report::thermal().unwrap_or_else(|| "no reading".to_string())
    ));
    body.push_str(&format!(
        "- load: {load_at_start} (1/5/15-min) at the start, {} at the table; every cell line carries its own `load=`\n",
        load_average(),
    ));
    let ceiling_mib = crate::guard::max_swap_growth_mib();
    let swap_growth = crate::guard::swap_used_bytes()
        .zip(swap_at_start)
        .map(|(now, start)| now.saturating_sub(start) / (1024 * 1024));
    match (swap_at_start, swap_growth) {
        (Some(start), Some(growth)) => body.push_str(&format!(
            "- swap: {} MiB in use at the start, {growth} MiB grown during the run (ceiling {ceiling_mib} MiB{})\n",
            start / (1024 * 1024),
            if ceiling_mib == crate::guard::MAX_SWAP_GROWTH_MIB {
                String::new()
            } else {
                format!(
                    ", raised by `MAHBOT_STAND_MAX_SWAP_MIB` from its default {}",
                    crate::guard::MAX_SWAP_GROWTH_MIB
                )
            },
        )),
        _ => body.push_str(&format!(
            "- swap: the platform reported no figure; ceiling {ceiling_mib} MiB\n"
        )),
    }
    body
}

/// The table's rows, in the record's own order.
fn cell_names() -> Vec<&'static str> {
    cells().into_iter().map(|cell| unit_name(&cell)).collect()
}

/// What the stand block in the record already holds and this run cannot take
/// again: the `| unit |` rows by name, the weight-stream rows by the `unit` they
/// are reported for (a run that measures one generative cell keeps the others'),
/// and the `- model:` / `- voice:` / `- text:` lines for artefacts this run did
/// not open.
#[derive(Default)]
struct Carried {
    rows: std::collections::BTreeMap<&'static str, String>,
    streams: Vec<StreamRow>,
    artefacts: Vec<String>,
}
/// One weight-stream row: the `unit` it is reported for — the key a later run's
/// row for the same phase replaces — and the five measured columns beside it (see
/// [`stream_row_prefix`]). The percentage is not carried: it divides by the
/// ceiling, and the ceiling is re-probed every run.
struct StreamRow {
    unit: String,
    columns: String,
}

/// Read the block the record already holds. A run that measures one cell
/// replaces that row and leaves the others where they are: the record holds the
/// latest figures, not a log, so a row this run did not take is carried rather
/// than dropped — and it cannot be re-taken here, because its own inputs are not
/// what this run opens. The same holds for the stream rows and artefact lines
/// that belong to those rows.
fn carried_block() -> Carried {
    std::fs::read_to_string(crate::measurements::path())
        .map_or_else(|_| Carried::default(), |text| carried_from(&text))
}

/// [`carried_block`] over a block's text, so the carry's replace-vs-carry rule is
/// drivable without a file on disk.
fn carried_from(text: &str) -> Carried {
    let mut carried = Carried::default();
    let names = cell_names();
    let mut inside = false;
    let mut in_streams = false;
    for line in text.lines() {
        if line.starts_with("<!-- BEGIN stand -->") {
            inside = true;
            continue;
        }
        if line.starts_with("<!-- END stand -->") {
            break;
        }
        if !inside {
            continue;
        }
        if line.starts_with("### Weight streams") {
            in_streams = true;
            continue;
        }
        if in_streams {
            if line.starts_with("Provenance:") || line.starts_with("### ") {
                in_streams = false;
            } else {
                // Rows only: the header names the columns, the ceiling above them
                // is this run's probe, and the prose describes both. Carrying a
                // paragraph would re-emit it on every run that carries it.
                let unit = line
                    .trim_matches('|')
                    .split('|')
                    .nth(1)
                    .map(str::trim)
                    .filter(|unit| !unit.is_empty() && *unit != "unit");
                if let (true, Some(unit), Some(five)) =
                    (line.starts_with("| "), unit, stream_row_prefix(line))
                {
                    carried.streams.push(StreamRow {
                        unit: unit.to_string(),
                        columns: five,
                    });
                }
                continue;
            }
        }
        if line.starts_with("- model: ")
            || line.starts_with("- voice: ")
            || line.starts_with("- text: ")
        {
            // The provenance's artefact lines, and nothing else from it: the
            // block's `code` line is the last run's tree and this run rewrites it,
            // so it describes no carried row.
            carried.artefacts.push(line.to_string());
        } else if line.starts_with("| ") {
            let unit = line
                .trim_start_matches("| ")
                .split(' ')
                .next()
                .unwrap_or("");
            if let Some(name) = names.iter().find(|name| **name == unit) {
                carried.rows.insert(*name, format!("{line}\n"));
            }
        }
    }
    carried
}

/// The five columns a weight-stream row is measured in — `| phase | unit | weight
/// bytes/unit | ms/unit | GB/s |` — which is what a run takes and what a carried
/// row keeps; its percentage is derived at write time, never carried.
fn stream_row_prefix(line: &str) -> Option<String> {
    let cols: Vec<&str> = line.trim_end().trim_matches('|').split('|').collect();
    (cols.len() == 6).then(|| format!("|{}|", cols[..5].join("|")))
}

/// One weight-stream row: the five measured columns with the share their own
/// `GB/s` takes of `ceiling_gb_s` beside them — always the ceiling printed above
/// the table, so a carried row and a measured one are denominated alike. A run
/// whose probe produced no figure writes `-` rather than a share of nothing.
fn stream_row(five: &str, ceiling_gb_s: Option<f64>) -> String {
    let rate = five
        .trim_end_matches('|')
        .rsplit('|')
        .next()
        .and_then(|rate| rate.trim().parse::<f64>().ok());
    let share = match (rate, ceiling_gb_s) {
        (Some(rate), Some(ceiling)) => format!("{:.1} %", rate / printed(ceiling, 1) * 100.0),
        _ => "-".to_string(),
    };
    format!("{five} {share} |\n")
}

/// This run's rows over the ones the record holds, keyed by the `unit` each is
/// reported for: a fresh row replaces its unit in place, a unit this run did not
/// measure keeps the row the record holds — order and all — so measuring one
/// generative cell cannot drop the other three.
fn merge_stream_rows(carried: Vec<StreamRow>, measured: Vec<StreamRow>) -> Vec<StreamRow> {
    let mut rows = carried;
    for row in measured {
        match rows.iter_mut().find(|slot| slot.unit == row.unit) {
            Some(slot) => slot.columns = row.columns,
            None => rows.push(row),
        }
    }
    rows
}

/// The file name an artefact line names, between its first pair of backticks —
/// the key the carried lines are deduplicated by.
fn artefact_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("- model: ")
        .or_else(|| line.strip_prefix("- voice: "))
        .or_else(|| line.strip_prefix("- text: "))?;
    rest.strip_prefix('`')?.split('`').next()
}

/// One `key=value` figure from a child's stdout, by label rather than by
/// position: the first whitespace-delimited token that starts with `{label}=`.
/// A figure the cell does not print (a cell with no synthesis split) is `None`,
/// which the table renders as `-`.
fn field(text: &str, label: &str) -> Option<f64> {
    let key = format!("{label}=");
    text.split_whitespace()
        .find_map(|token| token.strip_prefix(&key))
        .and_then(|value| value.parse().ok())
}

/// The `unit {name} …` line of a child's stdout: the row the table is built from,
/// as opposed to the cell's other lines that carry figures under the same names.
fn unit_line<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("unit {name} ");
    text.lines().find(|line| line.starts_with(&key))
}

/// The per-phase weight streams one cell's text carries, as
/// `(phase, unit, bytes per unit, ms per unit)`. Only the two generative cells
/// print them; a cell that does not is a row the rate table leaves out.
fn streams_of(cell: &str, text: &str) -> Vec<(&'static str, &'static str, f64, f64)> {
    fn pair(text: &str, bytes: &str, ms: &str) -> Option<(f64, f64)> {
        Some((field(text, bytes)?, field(text, ms)?))
    }
    let mut out = Vec::new();
    match cell {
        "listen" => {
            if let Some((bytes, ms)) = pair(text, "decode_stream_bytes", "decode_stream_ms") {
                out.push(("recognition decode", "one token", bytes, ms));
            }
        }
        "speak" => {
            if let Some((bytes, ms)) = pair(text, "body_stream_bytes", "body_stream_ms") {
                out.push(("talker body", "one frame", bytes, ms));
            }
            // One level of the frame's walk, not one of its draws: the walk is
            // `steps + 1` levels (`graphs.rs`'s `for s in 0..=steps`), and the
            // last one hands on logits instead of a code.
            // A missing field drops the row instead of dividing by a stand-in:
            // the level count is the divisor of one of the four phase figures.
            if let (Some(levels), Some((bytes, ms))) = (
                field(text, "predictor_levels").filter(|n| *n > 0.0),
                pair(text, "predictor_stream_bytes", "predictor_stream_ms"),
            ) {
                out.push(("code predictor", "one level", bytes / levels, ms / levels));
            }
            if let Some((bytes, ms)) = pair(text, "sound_stream_bytes", "sound_stream_ms") {
                out.push(("sound half", "one utterance", bytes, ms));
            }
        }
        _ => {}
    }
    out
}

/// One resident-set sample in decimal megabytes, or `unknown` where the platform
/// did not answer — a zero would read as a measurement of a live process.
fn mb(rss: Option<u64>) -> String {
    rss.map_or_else(
        || "unknown".to_string(),
        |b| format!("{:.1}", b as f64 / 1e6),
    )
}

/// One streaming-probe reading as a line: the ceiling every per-phase rate in the
/// record is reported against, or the reading's own name and reason when the
/// probe produced none — `no_adapter`, `device_work` and `panicked` are three
/// different outcomes and the line keeps them apart.
fn print_ceiling(
    label: &str,
    c: &Result<rlx_wgpu::bandwidth::StreamCeiling, rlx_wgpu::bandwidth::CeilingFailure>,
) {
    match c {
        Ok(c) => println!(
            "{label} read_gb_s={:.1} read_min_gb_s={:.1} read_max_gb_s={:.1} \
             copy_gb_s={:.1} copy_min_gb_s={:.1} copy_max_gb_s={:.1} \
             bytes_per_direction={} rounds={} at={} load={}",
            c.read_gb_s,
            c.read_min_gb_s,
            c.read_max_gb_s,
            c.gb_s,
            c.min_gb_s,
            c.max_gb_s,
            c.bytes_per_direction,
            c.rounds,
            now(),
            load_average(),
        ),
        Err(failure) => println!(
            "{label} unavailable={} reason=\"{}\"",
            failure.kind(),
            failure.reason().map_or_else(String::new, one_line),
        ),
    }
}

/// A probe failure as the record reads it: its kind (`no_adapter`, `device_work`,
/// `panicked`) and, where the platform gave one, the reason in parentheses. Both
/// record sites go through it so the two readings cannot drift apart.
fn failure_reading(failure: &rlx_wgpu::bandwidth::CeilingFailure) -> String {
    format!(
        "{}{}",
        failure.kind(),
        failure
            .reason()
            .map_or_else(String::new, |reason| format!(" ({})", one_line(reason)))
    )
}

/// The streaming probe with its panic caught: a diagnostic must not discard a
/// run, and a panic is its own reading rather than a missing adapter.
fn probe_ceiling() -> Result<rlx_wgpu::bandwidth::StreamCeiling, rlx_wgpu::bandwidth::CeilingFailure>
{
    caught(rlx_wgpu::bandwidth::stream_ceiling)
}

/// [`probe_ceiling`]'s panic mapping, with the probe as its argument so a test can
/// supply each of the three outcomes without a device.
fn caught(
    probe: impl FnOnce() -> Result<
        rlx_wgpu::bandwidth::StreamCeiling,
        rlx_wgpu::bandwidth::CeilingFailure,
    > + std::panic::UnwindSafe,
) -> Result<rlx_wgpu::bandwidth::StreamCeiling, rlx_wgpu::bandwidth::CeilingFailure> {
    std::panic::catch_unwind(probe)
        .map_err(|payload| {
            rlx_wgpu::bandwidth::CeilingFailure::Panicked(panic_text(payload.as_ref()))
        })
        .and_then(|reading| reading)
}

/// A panic payload's message, for the `panicked` reading.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// One probe reason on one line: the record's fields are `key=value` and rows are
/// markdown tables, so a newline, a quote or a table separator inside a reason would
/// break the line or the row it belongs to.
fn one_line(text: &str) -> String {
    text.replace(['\n', '\r', '"', '|'], " ")
}

/// The machine-readable line a unit prints after its run:
/// `unit <name> build_ms=<…> run_ms=<…> peak=<bytes> cold_ms=<…> map=<bytes>`.
fn parse_unit(text: &str, name: &str) -> Option<(f64, f64, u64, u64, f64)> {
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("unit ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        if parts.next()? != name {
            continue;
        }
        let build = parts.next()?.strip_prefix("build_ms=")?.parse().ok()?;
        let run = parts.next()?.strip_prefix("run_ms=")?.parse().ok()?;
        let peak = parts.next()?.strip_prefix("peak=")?.parse().ok()?;
        let cold = parts.next()?.strip_prefix("cold_ms=")?.parse().ok()?;
        let map = parts.next()?.strip_prefix("map=")?.parse().ok()?;
        return Some((cold, run, peak, map, build));
    }
    None
}

fn now() -> String {
    std::process::Command::new("date")
        .arg("+%H:%M:%S")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn label(cell: &Cell) -> String {
    match cell {
        Cell::Ear => "ear frames=76".into(),
        Cell::Ear075 => "ear.075 samples=12000".into(),
        Cell::Ear2 => "ear.batch2 samples=24000".into(),
        Cell::Ear4 => "ear.batch4 samples=48000".into(),
        Cell::Listen => "listen frames=3000".into(),
        Cell::Speak => format!("speak text={SPEAK_TEXT_TOKENS} rows={SPEAK_CODEC_ROWS}"),
        Cell::Embed256 => "embed.256 rows=256".into(),
        Cell::Embed8192 => "embed.8192 rows=8192".into(),
        Cell::Transcribe => "transcribe corpus+music+constructed".into(),
    }
}

/// The number of measured warm passes after a graph's first (cold) pass.
const WARM_REPS: usize = 5;

/// A cell's steps: its graphs built and compiled, then the cold first pass and
/// the warm passes. The build is paid once per process; the two run figures are
/// the same rule for every phase and every unit.
#[derive(Clone, Copy)]
struct Steps {
    build_ms: f64,
    cold_ms: f64,
    warm_ms: f64,
    passes: usize,
}

/// One phase of a unit: its graph, how many times the unit runs it, and the
/// cold/warm prices of one run. A loop's first iteration is cold, the rest warm;
/// the phase totals are what the unit sums.
#[derive(Clone)]
struct Phase {
    name: String,
    iters: usize,
    first_ms: f64,
    step_ms: f64,
    min_ms: f64,
    max_ms: f64,
    passes: usize,
    nodes: usize,
    cold_ms: f64,
    warm_ms: f64,
}

impl Phase {
    fn single(name: impl Into<String>, o: &Outcome) -> Self {
        let nodes = o.ops.iter().map(|(_, n)| n).sum();
        Self {
            name: name.into(),
            iters: 1,
            first_ms: o.first_ms,
            step_ms: o.step_ms,
            min_ms: o.min_ms,
            max_ms: o.max_ms,
            passes: o.passes,
            nodes,
            cold_ms: o.first_ms,
            warm_ms: o.step_ms,
        }
    }

    /// A loop's own samples: the first iteration is cold, the rest give the warm
    /// median (and the spread). `times` is in execution order.
    fn loop_times(name: impl Into<String>, times: &[f64], nodes: usize) -> Self {
        let n = times.len().max(1);
        let first = times.first().copied().unwrap_or(0.0);
        let warm = median(&times[1.min(times.len())..]).unwrap_or(first);
        Self {
            name: name.into(),
            iters: n,
            first_ms: first,
            step_ms: warm,
            min_ms: times.iter().copied().fold(first, f64::min),
            max_ms: times.iter().copied().fold(first, f64::max),
            passes: n,
            nodes,
            cold_ms: first + warm * (n - 1) as f64,
            warm_ms: warm * n as f64,
        }
    }

    /// A graph run once per window: the first window pays the cold pass, every
    /// window's warm median is part of the phase's warm total.
    fn windows(name: String, windows: &[(f64, f64, f64, f64, usize)], nodes: usize) -> Self {
        let first = windows.first().copied().unwrap_or((0.0, 0.0, 0.0, 0.0, 0));
        let steps: Vec<f64> = windows.iter().map(|w| w.1).collect();
        let warm_each = median(&steps).unwrap_or(first.1);
        Self {
            name: name.into(),
            iters: windows.len().max(1),
            first_ms: first.0,
            step_ms: warm_each,
            min_ms: windows.iter().map(|w| w.2).fold(f64::INFINITY, f64::min),
            max_ms: windows
                .iter()
                .map(|w| w.3)
                .fold(f64::NEG_INFINITY, f64::max),
            passes: windows.iter().map(|w| w.4).sum(),
            nodes,
            cold_ms: first.0 + windows.iter().skip(1).map(|w| w.1).sum::<f64>(),
            warm_ms: windows.iter().map(|w| w.1).sum(),
        }
    }
}

/// The index of the value [`median`] returns, so a row takes its split from the
/// same repetition rather than the nearest one.
fn median_index(v: &[f64]) -> Option<usize> {
    if v.is_empty() {
        return None;
    }
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    Some(order[v.len() / 2])
}

fn median(v: &[f64]) -> Option<f64> {
    median_index(v).map(|i| v[i])
}

/// `v` at the number of decimals the record prints it with, so a figure derived
/// from it is the printed division rather than a more precise one.
fn printed(v: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    (v * scale).round() / scale
}

/// One cell's measurements in the stand's own line: the run's price, its witnesses
/// (digest, census, counters) and its memory. `arena` is `Outcome::arena_bytes` —
/// rlx-wgpu's own planner over the graph before its unfuse passes, without the
/// scratch tail — not the bytes the arena buffer was allocated with.
fn line(
    cell: &str,
    fields: &[(&str, String)],
    o: &Outcome,
    values: &[f32],
    phases: &[Phase],
    s: Steps,
    peak: u64,
    map_bytes: u64,
    counters: (u64, u64),
    speech_s: f64,
    split_ms: (f64, f64),
) {
    let steps: usize = o.ops.iter().map(|(_, n)| n).sum();
    let kinds: Vec<String> = o.ops.iter().map(|(k, n)| format!("{k}:{n}")).collect();
    let mut text = format!("cell {cell}");
    for (k, v) in fields {
        text.push_str(&format!(" {k}={v}"));
    }
    text.push_str(&format!(
        " build_ms={:.1} cold_ms={:.1} run_ms={:.1} compile_ms={:.1} upload_ms={:.1} first_ms={:.1} step_ms={:.1} min_ms={:.1} max_ms={:.1} passes={} phase_passes={}",
        s.build_ms, s.cold_ms, s.warm_ms, o.compile_ms, o.upload_ms, o.first_ms, o.step_ms, o.min_ms, o.max_ms, s.passes,
        phases.iter().map(|p| format!("{}:{}", p.name, p.passes)).collect::<Vec<_>>().join(",")
    ));
    text.push_str(&format!(" submits={} waits={}", counters.0, counters.1));
    // Printed only by a cell that has a speech denominator: a cell with none says
    // nothing rather than `0`.
    if speech_s > 0.0 {
        text.push_str(&format!(" speech_s={speech_s:.6}"));
    }
    if split_ms.0 > 0.0 || split_ms.1 > 0.0 {
        text.push_str(&format!(
            " loop_ms={:.2} sound_ms={:.2}",
            split_ms.0, split_ms.1
        ));
    }
    text.push_str(&format!(" steps={steps} kinds={}", kinds.join("+")));
    text.push_str(&format!(
        " digest={:016x} values={} maxabs={} nonfinite={} params_f32={} params_packed={} dequant_nodes={}",
        digest(values),
        values.len(),
        sci(max_abs(values)),
        non_finite(values),
        human(o.params_f32 as u64),
        human(o.params_packed as u64),
        o.dequant_nodes
    ));
    text.push_str(&format!(" arena={}", human(o.arena_bytes as u64)));
    text.push_str(&format!(" keys={}", keys()));
    let beyond_map = peak.saturating_sub(map_bytes);
    let file_over_rss = map_bytes.saturating_sub(peak);
    text.push_str(&format!(
        " rss_end={} map={} beyond_map={} file_over_rss={} at={} load={}",
        resident_rss().map_or_else(|| "unknown".to_string(), human),
        human(map_bytes),
        human(beyond_map),
        human(file_over_rss),
        now(),
        load_average()
    ));
    println!("{text}");
    for p in phases {
        println!(
            "phase {cell} {} iters={} nodes={} first_ms={:.2} step_ms={:.2} min_ms={:.2} max_ms={:.2} passes={} cold_ms={:.2} warm_ms={:.2} at={} load={}",
            p.name,
            p.iters,
            p.nodes,
            p.first_ms,
            p.step_ms,
            p.min_ms,
            p.max_ms,
            p.passes,
            p.cold_ms,
            p.warm_ms,
            now(),
            load_average()
        );
    }
    // The unit's own summary: build, the two run figures under one rule, the
    // unit's whole-cell counters and the process peak (`ru_maxrss`, a high-water
    // mark that covers everything this cell allocated and that no drop lowers).
    let beyond_map = peak.saturating_sub(map_bytes);
    let file_over_rss = map_bytes.saturating_sub(peak);
    // A cell with no speech denominator says nothing rather than `0`, the same way its
    // `cell` line does.
    let speech = if speech_s > 0.0 {
        format!(" speech_s={speech_s:.6}")
    } else {
        String::new()
    };
    println!(
        "unit {cell} build_ms={:.3} run_ms={:.3} peak={} cold_ms={:.3} map={} beyond_map={} file_over_rss={} arena={} submits={} waits={} at={} load={}{speech}",
        s.build_ms,
        s.warm_ms,
        peak,
        s.cold_ms,
        map_bytes,
        beyond_map,
        file_over_rss,
        human(o.arena_bytes as u64),
        counters.0,
        counters.1,
        now(),
        load_average()
    );
}

fn run(cell: Cell, i: usize) {
    let name = label(&cell);
    println!("--- {i}: {name}");
    match cell {
        Cell::Ear => ear(),
        Cell::Ear075 => ear_short(12_000, "ear.075"),
        Cell::Ear2 => ear_short(24_000, "ear.batch2"),
        Cell::Ear4 => ear_short(48_000, "ear.batch4"),
        Cell::Listen => listen(),
        Cell::Speak => speak(),
        Cell::Embed256 => embed(256),
        Cell::Embed8192 => embed(8_192),
        Cell::Transcribe => transcribe_cell(),
    }
}

/// The wake path: 12 160 samples → mel (last frame dropped) → the 1.7B 24-block
/// sound encoder → the `ln_post → proj1 → GELU → proj2` projector → mean over
/// the ten projected tokens → L2 → cosine. The target is the same pipeline on a
/// different segment of the recorded 16 kHz asset.
fn ear() {
    let pair = match crate::loader::models_dir()
        .and_then(|dir| Pair::open(dir.join(WAKE_BODY), dir.join(WAKE_SOUND)))
    {
        Ok(p) => Arc::new(p),
        Err(e) => {
            cell_failed!("cell ear — open the 1.7B pair: {e}");
        }
    };
    let sound = pair.sound.clone();
    let frames = EAR_FRAMES;
    let samples = EAR_SAMPLES;
    let path = match crate::loader::voice_wav() {
        Ok(p) => p,
        Err(e) => {
            cell_failed!("cell ear — voice reference: {e}");
        }
    };
    let offset = SPEECH_OFFSET;
    let audio = match host::wav_mono_16k(&path.display().to_string(), offset, samples) {
        Ok(v) if v.len() == samples => v,
        Ok(v) => {
            println!(
                "cell ear — `{}` gave {} of {samples} samples",
                path.display(),
                v.len()
            );
            return;
        }
        Err(e) => {
            cell_failed!("cell ear — audio: {e}");
        }
    };
    let tokens = graphs::asr_tokens(frames);

    let mut front = match Front::build(&pair, frames) {
        Ok(f) => f,
        Err(e) => {
            cell_failed!("cell ear — front build: {e}");
        }
    };
    let build_ms = front.build_ms();
    let c0 = graphs::device_counts();
    let (m, e, p) = front.run(audio, WARM_REPS);
    let dim = sound.clip().projection_dim;
    let blocks = sound.clip().layers;
    let pooled = match mean_pool(&p.outs[0], tokens, dim) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell ear — projection pool: {e}");
        }
    };
    let values = l2(&pooled);
    if let Ok(dir) = std::env::var("STAND_EAR_DUMP") {
        let _ = std::fs::create_dir_all(&dir);
        for (name, buf) in [
            ("mel", &m.outs[0]),
            ("encoder", &e.outs[0]),
            ("projector", &p.outs[0]),
            ("pooled", &values),
        ] {
            let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/{name}.f32"), bytes);
        }
    }
    if let Ok(dir) = std::env::var("STAND_EAR_REF") {
        let read = |name: &str| -> Option<Vec<f32>> {
            let bytes = std::fs::read(format!("{dir}/{name}.f32")).ok()?;
            Some(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        };
        for (name, got) in [
            ("mel", &m.outs[0]),
            ("encoder", &e.outs[0]),
            ("projector", &p.outs[0]),
            ("pooled", &values),
        ] {
            if let Some(reference) = read(name) {
                if reference.len() == got.len() {
                    let dot: f32 = reference.iter().zip(got).map(|(a, b)| a * b).sum();
                    let na: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let nb: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
                    println!(
                        "ear ref {name} cosine={:.8} maxdiff={:.3e}",
                        dot / (na * nb),
                        max_abs(
                            &reference
                                .iter()
                                .zip(got)
                                .map(|(a, b)| a - b)
                                .collect::<Vec<_>>()
                        )
                    );
                }
            }
        }
    }

    // The registered target is built and run **after** the measured path, on a
    // separate trio of graphs: its cold pass cannot spend the measured cell's
    // first pass.
    let ref_path = path.clone();
    let ref_audio =
        match host::wav_mono_16k(&ref_path.display().to_string(), EAR_TARGET_OFFSET, samples) {
            Ok(v) if v.len() == samples => v,
            Ok(v) => {
                cell_failed!("cell ear — reference gave {} of {samples} samples", v.len());
            }
            Err(e) => {
                cell_failed!("cell ear — reference audio: {e}");
            }
        };
    let mut reference = match Front::build(&pair, frames) {
        Ok(f) => f,
        Err(e) => {
            cell_failed!("cell ear — reference front build: {e}");
        }
    };
    let (tm, te, tp) = reference.run(ref_audio, WARM_REPS);
    let pooled_target = match mean_pool(&tp.outs[0], tokens, dim) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell ear — reference projection pool: {e}");
        }
    };
    let target = l2(&pooled_target);

    let phases = vec![
        Phase::single("mel", &m),
        Phase::single("encoder", &e),
        Phase::single("projector", &p),
        Phase::single("target.mel", &tm),
        Phase::single("target.encoder", &te),
        Phase::single("target.projector", &tp),
    ];
    let measured = &phases[..3];
    let cold_ms: f64 = measured.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = measured.iter().map(|p| p.warm_ms).sum();
    let passes = measured.iter().map(|p| p.passes).sum();
    let cosine = values.iter().zip(&target).map(|(a, b)| a * b).sum::<f32>();
    let o = p;
    // The target graphs are the calibration, not the measured cell, and are
    // released before the line's own readings. `peak` below is `ru_maxrss`, the
    // process high-water mark: it covers the whole cell — its build, the measured
    // path and this calibration — and no drop can lower it.
    drop(reference);
    drop(tm);
    drop(te);
    drop(tp);
    let peak = peak_rss();
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    drop(front);
    drop(pair);
    drop(sound);
    report::release_free();
    let c = graphs::device_counts();
    line(
        "ear",
        &[
            ("frames", frames.to_string()),
            ("rows", tokens.to_string()),
            ("samples", samples.to_string()),
            ("offset", offset.to_string()),
            ("cross_cosine", format!("{cosine:.4}")),
            ("dim", dim.to_string()),
            ("blocks", blocks.to_string()),
        ],
        &o,
        &values,
        &phases,
        Steps {
            build_ms,
            cold_ms,
            warm_ms,
            passes,
        },
        peak,
        map_bytes,
        (c.0 - c0.0, c.1 - c0.1),
        0.0,
        (0.0, 0.0),
    );
}

fn file_bytes(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// What one recognition pass leaves behind: the walk itself, the phases the stand's
/// line prints, and the decode step's median — everything else is the walk's own
/// answer, read from it rather than copied field by field.
struct Recognised {
    walk: listen::Walk,
    phases: Vec<Phase>,
    decode_ms: f64,
}

/// One recognition pass over 16 kHz mono audio, on the ASR pair's own path: the
/// mel front end, the encoder, the projector, the prompt's prefill and the greedy
/// decode on per-layer plane caches. The listen cell measures this walk, and
/// the speak cell's round-trip gate transcribes its own waveform through the same
/// function, so the two cells cannot drift: a change to the arithmetic moves both
/// answers together.
///
/// `reps` is the number of warm repetitions each measured phase runs after its
/// cold pass (`WARM_REPS` in the stand). The gate passes 0 — one cold pass is the
/// transcript it needs, and a repetition would be a second of the cell's budget
/// spent on a correctness reading. `cell` names the caller in the
/// `cell_failed!` messages, which end this process: a recognition that cannot
/// run is a failed cell in both callers.
///
/// The audio is conditioned by the caller and never here: the listen cell applies
/// [`prepare_speech`], the gate transcribes the raw waveform the synthesis half
/// produced.
fn recognise(
    pair: &Arc<Pair>,
    tokenizer: &Tokenizer,
    audio: Vec<f32>,
    reps: usize,
    cell: &str,
) -> Recognised {
    let w = match listen::recognise(pair, tokenizer, audio, reps) {
        Ok(w) => w,
        Err(e) => {
            cell_failed!("cell {cell} — {e:#}");
        }
    };
    let mut phases = Vec::with_capacity(5);
    phases.push(Phase::single("mel", &w.mel));
    phases.push(Phase::single("encoder", &w.encoder));
    phases.push(Phase::single("projector", &w.projector));
    let mut prefill = Phase::single("prefill", &w.prefill);
    prefill.first_ms += w.prompt_read_ms;
    prefill.step_ms += w.prompt_read_ms;
    prefill.cold_ms += w.prompt_read_ms;
    prefill.warm_ms += w.prompt_read_ms;
    phases.push(prefill);
    let mut decode = Phase::loop_times("decode", &w.decode_times, w.decode_nodes);
    decode.first_ms = w.decode_times.first().copied().unwrap_or(0.0);
    decode.cold_ms = decode.first_ms + decode.step_ms * (decode.iters.saturating_sub(1)) as f64;
    let decode_ms = decode.step_ms;
    phases.push(decode);
    Recognised {
        walk: w,
        phases,
        decode_ms,
    }
}

/// What a recognition walk answered, for the harnesses that price a decode
/// change: the token stream the listen cell prints as `first_tokens` (all of it),
/// the transcript and the digest it prints as `digest`, and the walk's own price.
pub struct Answer {
    pub generated: Vec<usize>,
    /// The greedy choice that stopped the walk, i.e. the model's choice at
    /// position `generated.len()`.
    pub next: usize,
    pub frames: usize,
    pub prompt_rows: usize,
    pub capacity: usize,
    pub text: String,
    /// The cell's own `digest=` — the walk's value list, hashed here so a harness
    /// compares against the pin rather than restating the list.
    pub digest: u64,
    pub decode_ms: f64,
    pub build_ms: f64,
    pub stream_bytes: u64,
}

impl Answer {
    fn of(r: Recognised) -> Self {
        Self {
            generated: r.walk.generated,
            next: r.walk.next,
            frames: r.walk.frames,
            prompt_rows: r.walk.prompt_rows,
            capacity: r.walk.capacity,
            text: r.walk.text,
            digest: digest(&r.walk.values),
            decode_ms: r.decode_ms,
            build_ms: r.walk.build_ms,
            stream_bytes: r.walk.decode_stream_bytes,
        }
    }
}

/// The listen cell's recording: `STAND_ASR_WAV` when set, else the reference voice.
pub fn listen_path() -> anyhow::Result<std::path::PathBuf> {
    match std::env::var("STAND_ASR_WAV").ok() {
        Some(p) => Ok(std::path::PathBuf::from(p)),
        None => crate::loader::voice_wav(),
    }
}

/// The offset into that recording, `STAND_ASR_OFFSET`, `0` when unset.
pub fn listen_offset() -> usize {
    std::env::var("STAND_ASR_OFFSET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Whether a walk over `path` at `offset` is the recording the pinned witness was
/// taken on: the reference voice's path, at offset 0, and the voice file's content
/// still at its pin. The pins hold only for that clip, so any other input is a
/// different cell and its answer is printed as `skipped=alternate-input` instead of
/// compared — the file the walk actually reads decides, not the switch that named
/// it, so a two-recording run's first clip is still witnessed. Identity is the path,
/// not the bytes: a copy of the pinned voice under another name reads as an
/// alternate, and hashing every input to relabel a skip is not worth it.
fn listen_is_alternate(path: &std::path::Path, offset: usize) -> bool {
    offset != 0
        // The voice WAV is this cell's only audio input, so its *presence* is not
        // an alternate: the pin holds for the clip the witness was recorded with
        // and is skipped only when that file's content differs.
        || !crate::reference::voice_is_pinned()
        || !crate::loader::voice_wav().is_ok_and(|voice| voice == path)
}

/// The listen witness, printed as the cell prints it: the pins compared when this
/// run's input is the recorded clip, `skipped=alternate-input` when it is not. One
/// rule, so a harness that walks the same recording cannot print a different verdict
/// than the cell.
pub fn witness_listen(
    path: &std::path::Path,
    offset: usize,
    frames: usize,
    prompt_rows: usize,
    generated: usize,
    answer_digest: u64,
    transcript: &str,
) {
    if listen_is_alternate(path, offset) {
        println!("witness listen skipped=alternate-input");
        return;
    }
    let _ =
        crate::reference::check_listen(frames, prompt_rows, generated, answer_digest, transcript);
}

/// The listen cell's input, conditioned the way the cell takes it: `path` at
/// `offset`, 16 kHz mono, `SPEECH_SAMPLES` long, trimmed and gained by
/// [`prepare_speech`]. The pair is the caller's, so a harness that walks several
/// recordings opens the model once.
fn listen_input(
    pair: &Pair,
    path: &std::path::Path,
    offset: usize,
) -> anyhow::Result<(Tokenizer, Vec<f32>)> {
    let tokenizer = Tokenizer::from_model(&pair.body).context("the recognition tokenizer")?;
    let mut audio = host::wav_mono_16k(&path.display().to_string(), offset, SPEECH_SAMPLES)
        .context("the recording")?;
    prepare_speech(&mut audio);
    Ok((tokenizer, audio))
}

/// One recognition walk over `path` — the listen cell's own walk, its answer
/// returned rather than printed, one pass per phase (a harness prices a decode
/// change, not a repetition). `None` when the input cannot be read (the reason is
/// printed, so a harness skip is attributable); a walk that has the input but cannot
/// run ends the process exactly as the cell does (`recognise`'s `cell_failed!`).
pub fn listen_answer(pair: &Arc<Pair>, path: &std::path::Path, offset: usize) -> Option<Answer> {
    let (tokenizer, audio) = match listen_input(pair, path, offset) {
        Ok(v) => v,
        Err(e) => {
            println!("listen-answer skipped: {e:#}");
            return None;
        }
    };
    Some(Answer::of(recognise(
        pair,
        &tokenizer,
        audio,
        1,
        "listen-answer",
    )))
}

/// The recognition product segment: the voice clip (17.46 s, 1 746 mel frames)
/// → mel → encoder → the 234-row prompt (3 head + 6 text + 219 audio + 6 tail)
/// → body prefill → the greedy walk, 65 tokens on the recorded clip over a
/// 362-row cache (the prompt plus one 128-row bucket), stopping at EOS or at
/// `ASR_MAX_NEW`. A walk that reaches the budget without EOS is printed as
/// `unfinished`, not as an answer.
fn listen() {
    let path = match listen_path() {
        Ok(p) => p,
        Err(e) => {
            cell_failed!("cell listen — voice reference: {e}");
        }
    };
    let offset = listen_offset();
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            cell_failed!("cell listen — the recognition pair: {e:#}");
        }
    };
    let (tokenizer, audio) = match listen_input(&pair, &path, offset) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell listen — input: {e:#}");
        }
    };

    // The measurement walks the same path the round-trip gate transcribes with —
    // one function, so the two cells' answers move together.
    let Recognised {
        walk,
        phases,
        decode_ms,
    } = recognise(&pair, &tokenizer, audio, WARM_REPS, "listen");
    let (frames, tokens, prompt_rows) = (walk.frames, walk.tokens, walk.prompt_rows);
    let (samples, capacity, kv_bytes) = (walk.samples, walk.capacity, walk.kv_bytes);
    let (generated, written, finished, truncated) =
        (&walk.generated, walk.written, walk.finished, walk.truncated);
    if truncated {
        println!("cell listen — UNFINISHED: the walk used all {ASR_MAX_NEW} tokens without EOS");
    }
    let cold_ms: f64 = phases.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = phases.iter().map(|p| p.warm_ms).sum();
    let passes = phases.iter().map(|p| p.passes).sum();
    let fields = [
        ("frames", frames.to_string()),
        ("rows", tokens.to_string()),
        ("samples", samples.to_string()),
        ("offset", offset.to_string()),
        ("prompt", prompt_rows.to_string()),
        ("capacity", capacity.to_string()),
        ("capacity_bucket", DECODE_CACHE_BUCKET.to_string()),
        ("kv_bytes", kv_bytes.to_string()),
        ("generated", generated.len().to_string()),
        (
            "first_tokens",
            format!("{:?}", &generated[..generated.len().min(8)]),
        ),
        ("eos", finished.to_string()),
        ("truncated", truncated.to_string()),
        (
            "state",
            if truncated { "unfinished" } else { "finished" }.to_string(),
        ),
        ("text", format!("{:?}", walk.text)),
        ("decode_steps", (written - prompt_rows).to_string()),
        // One token's weight stream — the body graph's multiply and conv
        // parameters, which one decode step re-reads — and the step's own seconds.
        // The rate table in `MEASUREMENTS.md` divides the two.
        ("decode_stream_bytes", walk.decode_stream_bytes.to_string()),
        ("decode_stream_ms", format!("{decode_ms:.3}")),
    ];
    let peak = peak_rss();
    // `peak_rss` is `ru_maxrss`, the process's high-water mark: it reports the peak
    // the walk reached even though `recognise` dropped its runners and graphs before
    // returning, so the record attributes it to whoever held it then rather than to
    // what is live now.
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    // The witness is printed before the line: the pins record what this clip
    // answered, and a moved answer is reported with what moved, never a failed
    // run — a round that reorders a summation (`reference.rs` says why the pins
    // are witnesses) carries on and prints its reading. A cell deliberately fed
    // another recording or another generation budget is a different cell, so its
    // answer is not compared against the pinned one.
    witness_listen(
        &path,
        offset,
        frames,
        prompt_rows,
        generated.len(),
        digest(&walk.values),
        &walk.text,
    );
    line(
        "listen",
        &fields,
        &walk.prefill,
        &walk.values,
        &phases,
        Steps {
            build_ms: walk.build_ms,
            cold_ms,
            warm_ms,
            passes,
        },
        peak,
        map_bytes,
        walk.counters,
        samples as f64 / 16_000.0,
        (0.0, 0.0),
    );
}

/// The product feeds exactly 30 s; a shorter recording is right-padded with
/// zeros. The encoder has no feature mask, so a long artificial silence makes
/// the language model stop at `<asr_text>` (the FLEURS 17.46 s case). Keep the
/// speech and a short tail, and bring a very quiet recording up to a usable
/// level; the graph's frame count follows the resulting length instead of a
/// silent 30 s pad.
fn prepare_speech(audio: &mut Vec<f32>) {
    const WIN: usize = 800;
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    // A generated part can open with a loud pre-roll that is not speech — the
    // 26.4 s cell's first 2 s sit about ten times above the rest and make the
    // recognizer answer with the `None` language and stop. A recording whose
    // opening is much louder than its own body is trimmed at that opening; an
    // ordinary recording (including FLEURS' short leading silence) is untouched.
    const HEAD: usize = 2 * 16_000;
    if audio.len() > HEAD + WIN {
        let head_rms = rms(&audio[..HEAD]);
        let mut body: Vec<f32> = audio[HEAD..].chunks_exact(WIN).map(rms).collect();
        body.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let body_rms = body[body.len() / 2];
        if head_rms > 0.01 && head_rms > 4.0 * body_rms {
            eprintln!(
                "prepare_speech: dropping the opening {HEAD} samples (head RMS {head_rms:.4}, body {body_rms:.4})"
            );
            audio.drain(..HEAD);
        }
    }
    // The gain is taken after the pre-roll is gone: normalizing the loud opening
    // first would leave the actual speech at its own, much lower level. It is the
    // crate's one gain rule (`host::speech_gain`), which the transcription path
    // applies over a whole input.
    let gain = host::speech_gain(audio);
    if gain != 1.0 {
        for v in audio.iter_mut() {
            *v *= gain;
        }
    }
    let mut last_speech = 0usize;
    for (i, w) in audio.chunks_exact(WIN).enumerate() {
        if rms(w) > 3e-4 {
            last_speech = (i + 1) * WIN;
        }
    }
    if last_speech > 0 && last_speech + WIN < audio.len() {
        audio.truncate(last_speech + WIN);
    }
}

/// The three product ear windows as separate cells: one 0.75 s window, two of
/// them concatenated, four of them. The graph is built for the resulting fixed
/// frame count; there is no dynamic axis.
fn ear_short(samples: usize, name: &str) {
    let pair = match crate::loader::models_dir()
        .and_then(|dir| Pair::open(dir.join(WAKE_BODY), dir.join(WAKE_SOUND)))
    {
        Ok(p) => Arc::new(p),
        Err(e) => {
            cell_failed!("cell ear — open the 1.7B pair: {e}");
        }
    };
    let sound = pair.sound.clone();
    let frames = samples / graphs::MEL_HOP;
    let offset = SPEECH_OFFSET;
    let voice = match crate::loader::voice_wav() {
        Ok(p) => p,
        Err(e) => {
            cell_failed!("cell {name} — voice reference: {e}");
        }
    };
    let audio = match host::wav_mono_16k(&voice.display().to_string(), offset, samples) {
        Ok(v) if v.len() == samples => v,
        Ok(v) => {
            cell_failed!("cell ear — gave {} of {samples} samples", v.len());
        }
        Err(e) => {
            cell_failed!("cell ear — audio: {e}");
        }
    };
    let tokens = graphs::asr_tokens(frames);
    let mut front = match Front::build(&pair, frames) {
        Ok(f) => f,
        Err(e) => {
            cell_failed!("cell ear — front build: {e}");
        }
    };
    let build_ms = front.build_ms();
    let c0 = graphs::device_counts();
    let (m, e, p) = front.run(audio, WARM_REPS);
    let dim = sound.clip().projection_dim;
    let values = match mean_pool(&p.outs[0], tokens, dim) {
        Ok(v) => l2(&v),
        Err(e) => {
            cell_failed!("cell ear — projection pool: {e}");
        }
    };
    let phases = vec![
        Phase::single("mel", &m),
        Phase::single("encoder", &e),
        Phase::single("projector", &p),
    ];
    let cold_ms: f64 = phases.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = phases.iter().map(|p| p.warm_ms).sum();
    let passes = phases.iter().map(|p| p.passes).sum();
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    let peak = report::peak_rss();
    let c = graphs::device_counts();
    line(
        name,
        &[
            ("frames", frames.to_string()),
            ("rows", tokens.to_string()),
            ("samples", samples.to_string()),
            ("offset", offset.to_string()),
            ("dim", dim.to_string()),
            ("blocks", sound.clip().layers.to_string()),
        ],
        &p,
        &values,
        &phases,
        Steps {
            build_ms,
            cold_ms,
            warm_ms,
            passes,
        },
        peak,
        map_bytes,
        (c.0 - c0.0, c.1 - c0.1),
        0.0,
        (0.0, 0.0),
    );
}

/// The speak cell: the synthesis half, then its two quality gates, then the line.
///
/// The gates run before the line prints: a run whose waveform left the PCM range
/// or no longer says the same thing to our own recogniser is a failed run even if
/// its seconds improved — these two are the stand's hard gate. The round trip needs
/// the recognition pair; without it the cell still runs and the gate prints that
/// it did not.
fn speak() {
    let Some(Synthesis {
        frames,
        fields,
        wave,
        prompt_digest,
        o,
        phases,
        steps,
        peak,
        map_bytes,
        counters,
    }) = speak_synthesis()
    else {
        return;
    };
    let cell = "speak";
    // The quality gate, both halves before the line prints: the stand's hard gate.
    // The waveform's movement is a witness (`reference.rs`): what
    // must hold is the range and finiteness below and the round trip after, even
    // where a digest has moved.
    //
    // The first half is the waveform itself: nothing clipped (|y| <= 1, the range
    // a 24 kHz PCM file carries) and nothing non-finite. The pinned waveform is
    // quiet (max |y| 5.3e-2), so this bar catches a waveform that has run away,
    // not one that moved in its last bits.
    let peak_abs = max_abs(&wave);
    if !peak_abs.is_finite() || peak_abs > 1.0 || non_finite(&wave) != 0 {
        cell_failed!(
            "cell speak — the waveform leaves the PCM range: max|y|={} nonfinite={}",
            sci(peak_abs),
            non_finite(&wave)
        );
    }
    // The voice is an input, not a switch: the pin holds only for the voice the
    // witness was recorded with. Another `MAHBOT_VOICE_WAV` records its own
    // digest rather than failing the pinned one.
    let other_voice = !crate::reference::voice_is_pinned();
    let alternate = std::env::var_os("STAND_SPEAK_ROWS").is_some()
        || std::env::var_os("STAND_SPEAK_TEXT").is_some()
        || std::env::var_os("STAND_SPEAKER_WAV").is_some()
        || other_voice;
    if alternate {
        println!("witness speak skipped=alternate-input");
    } else {
        // The pinned hashes are a witness, not a gate (`reference.rs`): the
        // waveform's movement is recorded, the range above and the round trip
        // below are what decide.
        crate::reference::check_speak(frames, wave.len(), digest(&wave), prompt_digest);
    }

    // The round trip, after every phase: the waveform this cell just produced,
    // back through the recogniser. It reads the raw waveform — `prepare_speech`
    // is the recognition side's conditioning (a loud pre-roll trim and a gain on a
    // quiet recording), and a gate that conditioned its own input would test a
    // signal the product never sends. Only the rate changes, 24 kHz to the 16 kHz
    // the ASR front reads, through the same resampler the listen cell's file goes
    // through.
    //
    // One cold pass costs 5.8 s of the cell's 28.3 s on the recorded models — the
    // recognition graphs' build and the 107 decode steps at ~25 ms each, which a
    // switch could not take away without transcribing something other than the
    // waveform.
    if alternate {
        println!("witness speak-roundtrip skipped=alternate-input");
    } else {
        report::release_free();
        let audio = host::resample(&wave, 24_000, 16_000);
        // The round trip is a check on the synthesis half, not a precondition of
        // measuring it: a host that carries only the synthesis pair still runs the
        // cell and says the gate did not. A pair whose files are there but will not
        // open is a broken pair, not an absent one, and fails the cell.
        match Pair::of_role_optional(Role::Recognition) {
            Ok(Some(pair)) => {
                let tokenizer = match Tokenizer::from_model(&pair.body) {
                    Ok(t) => t,
                    Err(e) => {
                        cell_failed!("cell speak — the round trip's tokenizer: {e}");
                    }
                };
                let heard = recognise(&Arc::new(pair), &tokenizer, audio, 0, "speak")
                    .walk
                    .text;
                if !crate::reference::check_speak_roundtrip(&heard) {
                    cell_failed!("cell speak — the round trip is not the pinned transcript");
                }
                // The gate's pair is the process peak from here on. `peak_rss` is
                // `ru_maxrss`, a high-water mark, so it cannot say what is live now;
                // this reading is what the record's note compares the cell's own peak
                // against.
                println!("speak roundtrip peak_mb={:.1}", peak_rss() as f64 / 1e6);
            }
            Ok(None) => {
                println!("witness speak-roundtrip skipped=no-recognition-pair reason=files-absent");
            }
            Err(e) => {
                cell_failed!("cell speak — the round trip's recognition pair: {e}");
            }
        }
    }

    // The generation loop (body, predictor, sampler, speaker, prefill) against
    // the sound half; their sum is the unit's warm figure by construction.
    let loop_ms: f64 = phases
        .iter()
        .filter(|p| !p.name.starts_with("sound"))
        .map(|p| p.warm_ms)
        .sum();
    let sound_total: f64 = phases
        .iter()
        .filter(|p| p.name.starts_with("sound"))
        .map(|p| p.warm_ms)
        .sum();
    line(
        cell,
        &fields,
        &o,
        &wave,
        &phases,
        steps,
        peak,
        map_bytes,
        counters,
        frames as f64 * graphs::samples_per_frame() as f64 / 24_000.0,
        (loop_ms, sound_total),
    );
}

/// What the speak cell's synthesis half produced, and what its gate and its line
/// read from it.
///
/// Owning it is what puts the drop at `speak_synthesis`'s return: the round trip
/// loads the recogniser's own pair, so the boundary is the function and the
/// process holds one model pair, not two. A graph the loop grows later is inside
/// that boundary by construction rather than in a list to be maintained.
struct Synthesis {
    frames: usize,
    fields: Vec<(&'static str, String)>,
    wave: Vec<f32>,
    prompt_digest: u64,
    o: Outcome,
    phases: Vec<Phase>,
    steps: Steps,
    peak: u64,
    map_bytes: u64,
    counters: (u64, u64),
}

/// The speak cell's synthesis half: 300 text tokens and 330 codec rows (~26 s).
/// Per frame the body step, then the sixteen predictor levels each reading the
/// code just drawn, then the sum of the sixteen code embeddings plus the next
/// text hidden row becomes the next frame's input; then the sound half.
///
/// Every graph it compiled, and both models, are still resident when the process
/// peak is read, and all of them are dropped on return — see [`Synthesis`].
fn speak_synthesis() -> Option<Synthesis> {
    let rows = std::env::var("STAND_SPEAK_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(SPEAK_CODEC_ROWS);
    let pair = match Pair::of_role(Role::Synthesis) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            cell_failed!("cell speak — open the pair: {e}");
        }
    };
    let tokenizer = match Tokenizer::from_model(&pair.body) {
        Ok(t) => t,
        Err(e) => {
            cell_failed!("cell speak — tokenizer: {e}");
        }
    };
    let text_src = std::env::var("STAND_SPEAK_TEXT").unwrap_or_else(|_| TTS_TEXT.to_string());
    let mut text_ids = tokenizer.encode(&text_src);
    text_ids.truncate(SPEAK_TEXT_TOKENS);
    let text_count = text_ids.len();
    if text_ids.is_empty() {
        cell_failed!("cell speak — the tokenizer gave no text tokens");
    }
    // `SPEAK_TEXT_TOKENS` is the model's text capacity, not a fixed input
    // length: the reference feeds exactly the real tokens, then EOS. Padding the
    // tail with `tts_pad` here would make the model read pad rows as content.
    let device = rlx_runtime::Device::Gpu;
    let cfg = pair.body.config().clone();
    let d = cfg.d;
    let body_vocab = match cfg.head {
        loader::Head::Separate { vocab, .. } | loader::Head::Tied { vocab, .. } => vocab,
    };
    let hkv = cfg.kv_heads * cfg.head_dim;
    let steps = graphs::code_predictor_steps(&pair.sound).unwrap_or(0);
    let (top_k, top_p, temperature) = pair.body.sampling();
    let (blocks, hkv_pred, _) = match graphs::code_predictor_cache(&pair.sound) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell speak — predictor cache: {e}");
        }
    };
    let capacity_pred = steps + 1;
    let (d_talker, _) = pair.sound.code_proj_widths().unwrap();
    let head_out = pair
        .sound
        .plan("a.gen.code.head.weight")
        .unwrap()
        .ggml_shape[1];
    let mut build_ms = 0f64;
    let c0 = graphs::device_counts();
    let mut phases = Vec::new();

    // The engine's own `Op::Sample`, built once for each width. The measured
    // path still draws on the host; the first real logits of each width are
    // compared below, so the substitution is proved, not assumed.
    let sampler = SampleSpec {
        top_k,
        top_p,
        temperature,
        seed: 12345,
    };
    let t = report::Stopwatch::start();
    let mut sample_body = match graphs::sample_graph_one(body_vocab, sampler)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — engine sampler build: {e}");
        }
    };
    let mut sample_pred = match graphs::sample_graph_one(head_out, sampler)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — predictor sampler build: {e}");
        }
    };
    build_ms += t.ms();
    let sampler0: String;
    let mut sampler1 = String::from("not-run");

    // The voice encoder: the recorded reference through the mmproj's ECAPA
    // speaker half (24 kHz mel → 2048-d embedding). This is the conditioning
    // the base model uses for the voice; without it the talker has only text.
    let spk_path = match std::env::var("STAND_SPEAKER_WAV") {
        Ok(p) => p,
        Err(_) => match crate::loader::voice_wav() {
            Ok(p) => p.display().to_string(),
            Err(e) => {
                cell_failed!("cell speak — speaker reference: {e}");
            }
        },
    };
    // The reference loads the voice at its own length and resamples it to
    // 24 kHz: no offset and no 5-second cut. `max_samples` is the whole file.
    let spk_audio = match host::wav_mono(&spk_path, 24_000, 0, usize::MAX) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell speak — speaker audio `{spk_path}`: {e}");
        }
    };
    let (spk_mel, spk_frames) = host::mel_24k(&spk_audio);
    let t = report::Stopwatch::start();
    let mut speaker = match graphs::speaker_encoder(pair.sound.clone(), spk_frames, d)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — speaker encoder build: {e}");
        }
    };
    build_ms += t.ms();
    let spk = speaker.run(&[("mel".into(), spk_mel)], WARM_REPS);
    phases.push(Phase::single("speaker", &spk));
    let speaker_embed = spk.outs[0].clone();

    // The prompt, its trailing text and the pad row come from `synth`, where
    // `Synthesis::generate` builds the same form.
    let TalkerPrompt {
        x: prompt_x,
        rows: prompt_rows,
        trailing,
        pad,
        speaker_norm,
        text0_norm,
    } = match talker_prompt(&pair.body, d, &speaker_embed, &text_ids[..text_count]) {
        Ok(p) => p,
        Err(e) => {
            cell_failed!("cell speak — prompt: {e}");
        }
    };

    // Residency is attributed by measurement: one sample with the sampler graphs,
    // the speaker encoder and the talker prompt already built and run, one after
    // each graph the generation loop holds live, one with the
    // sound half's, so the peak's growth is read off the deltas rather than
    // inferred from plans.
    let rss_start = report::resident_rss();
    let t = report::Stopwatch::start();
    let mut prefill =
        match graphs::body_new_rows_head_chain(pair.body.clone(), prompt_rows, 0, true)
            .and_then(|m| graphs::Compiled::new(device, m))
        {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell speak — prompt prefill build: {e}");
            }
        };
    build_ms += t.ms();
    let rss_prefill = report::resident_rss();
    let (cos, sin) = host::rope_tables_at(0, prompt_rows, cfg.head_dim, cfg.rope_base as f64);
    let prompt_digest = digest(&prompt_x);
    let p = prefill.run(
        &[
            ("x".into(), prompt_x),
            ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
            ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
        ],
        WARM_REPS,
    );
    phases.push(Phase::single("prefill", &p));
    let mut code_seen: Vec<usize> = Vec::new();
    let mut draw_index: u64 = 0;
    let mut sample_times: Vec<f64> = Vec::new();
    let mut code0 = {
        let mut logits = p.outs[3][..body_vocab].to_vec();
        suppress_service(&mut logits, body_vocab);
        apply_repetition(&mut logits, &code_seen, TTS_REPEAT_PENALTY);
        let t = report::Stopwatch::start();
        let id = host::draw(&logits, top_k, top_p, temperature, 12345 + draw_index);
        sample_times.push(t.ms());
        let engine = sample_body.run(&[("logits".into(), logits)], 0);
        let got = engine.outs[0].first().copied().unwrap_or(f32::NAN) as usize;
        sampler0 = if got == id {
            "equal".into()
        } else {
            format!("host={id} engine={got}")
        };
        draw_index += 1;
        id
    };
    let mut hidden = p.outs[0][(prompt_rows - 1) * d..prompt_rows * d].to_vec();

    // The talker on the part's fixed-capacity cache: one row per frame. The
    // prompt rows are copied into that cache before the first step.
    let capacity = prompt_rows + rows;
    let t = report::Stopwatch::start();
    let mut body = match graphs::body_capacity_chain_planes(pair.body.clone(), capacity)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — body build: {e}");
        }
    };
    build_ms += t.ms();
    let rss_body = report::resident_rss();
    let graphs::PlaneCache {
        k: mut k_cache,
        v: mut v_cache,
        mut mask,
    } = graphs::seed_plane_cache(
        cfg.layers,
        capacity,
        prompt_rows,
        hkv,
        &p.outs[1],
        &p.outs[2],
    );
    if !body.runner.resident_cache_planes(&k_cache, &v_cache) {
        cell_failed!("cell speak — body resident cache refused");
    }
    let mut kv_bytes: usize = k_cache
        .iter()
        .chain(v_cache.iter())
        .map(|rows| rows.len())
        .sum();
    for rows in k_cache.iter_mut().chain(v_cache.iter_mut()) {
        rows.clear();
    }

    // The code predictor: **one** graph for the whole sixteen-level frame — the
    // head-less body at level 0, then every projecting level with its own head —
    // with each code drawn by an `Op::Sample` on the device and the cache carried
    // between levels in the arena. A frame is one run and one fence instead of
    // sixteen, its dispatches submitted in encode chunks, and the sampled code
    // never comes back to the host between levels.
    let t = report::Stopwatch::start();
    let mut pred_frame =
        match graphs::tts_code_predictor_frame(pair.sound.clone(), capacity_pred, sampler)
            .and_then(|m| graphs::Compiled::new(device, m))
        {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell speak — predictor frame build: {e}");
            }
        };
    build_ms += t.ms();
    // The frame graph carries the row-first `[capacity, blocks, hkv]` cache in
    // its own arena; the figure stays the cache the walk holds.
    kv_bytes += 2 * capacity_pred * blocks * hkv_pred;
    // The predictor's first step is its cold pass; no throw-away run.
    let rss_loop_graphs = report::resident_rss();
    let mut all_codes: Vec<Vec<usize>> = Vec::with_capacity(rows);
    let mut body_times: Vec<f64> = Vec::new();
    let mut pred_times: Vec<f64> = Vec::new();
    let mut written = prompt_rows;
    for f in 0..rows {
        // The reference stops on codec EOS at level 0, but never before two
        // frames (`min_new_tokens = 2`), and falls back to `max_new_tokens`
        // when EOS never appears. EOS itself is not part of the returned
        // sequence: it is neither seen by the repetition penalty nor fed to
        // the sound half.
        if f >= 2 && code0 == TTS_CODEC_EOS {
            break;
        }
        code_seen.push(code0);
        let mut frame_codes = vec![code0];
        let code0_embed =
            match pair
                .body
                .table_rows_f32("token_embd.weight", &[(0, TTS_TEXT_ROWS + code0)], d)
            {
                Ok(mut v) => v.remove(0),
                Err(e) => {
                    cell_failed!("cell speak — reading code0 embedding: {e}");
                }
            };

        // One fence for the whole walk, and one submission per encode chunk: the
        // sixteen levels live in one schedule, each draw is `Op::Sample` on the
        // device, and the cache rows stay in the arena between them. Only the draw
        // seeds cross the boundary — they are `Op::Sample`'s uniform words, so the
        // compiled frame serves every frame.
        let seeds = graphs::predictor_draw_seeds(12345, draw_index, steps, top_k, temperature);
        if !pred_frame.runner.set_sample_seeds(&seeds) {
            cell_failed!("cell speak — predictor frame refused the frame's draw seeds");
        }
        let read: &[usize] = if f == 0 { &[0, 1] } else { &[0] };
        let t = report::Stopwatch::start();
        let out = pred_frame.runner.run_read(
            &[
                ("hidden".into(), hidden.clone()),
                ("code0_embed".into(), code0_embed.clone()),
            ],
            read,
        );
        pred_times.push(t.ms());
        let codes = match out.first() {
            Some(v) if v.len() >= steps => v,
            _ => {
                cell_failed!("cell speak — predictor frame returned no codes at frame {f}");
            }
        };
        for k in 0..steps {
            frame_codes.push(codes[k] as usize);
        }
        draw_index += steps as u64;
        // The one-off arithmetic witness, on the frame graph's own level-1
        // logits: the stand's one-node sampler and the host draw agree.
        if f == 0 {
            let logits = match out.get(1) {
                Some(v) if v.len() >= head_out => v[..head_out].to_vec(),
                _ => {
                    cell_failed!("cell speak — predictor frame returned no level-1 logits");
                }
            };
            let same_seed = host::draw(&logits, top_k, top_p, temperature, 12345);
            let engine = sample_pred.run(&[("logits".into(), logits)], 0);
            let got = engine.outs[0].first().copied().unwrap_or(f32::NAN) as usize;
            sampler1 = if got == same_seed {
                "equal".into()
            } else {
                format!("host={same_seed} engine={got}")
            };
        }
        // The next frame's input: the sixteen code embeddings, summed. The rows
        // are picked out of the file one level at a time — the frame graph's
        // gather reads its own copy in the arena, so the host keeps none.
        let mut sum = vec![0f32; d_talker];
        for (o, v) in sum.iter_mut().zip(&code0_embed) {
            *o += *v;
        }
        let picks: Vec<(usize, usize)> = (1..=steps)
            .map(|level| (level - 1, frame_codes[level]))
            .collect();
        let embeds = match pair
            .sound
            .table_rows_f32("a.gen.code.embd.weight", &picks, d_talker)
        {
            Ok(v) => v,
            Err(e) => {
                cell_failed!("cell speak — reading the code embeddings: {e}");
            }
        };
        for e in &embeds {
            for (o, v) in sum.iter_mut().zip(e) {
                *o += *v;
            }
        }
        if f + 1 < rows {
            let mut x = trailing.get(f).cloned().unwrap_or_else(|| pad.clone());
            for (v, s) in x.iter_mut().zip(&sum) {
                *v += *s;
            }
            let (cos, sin) = host::rope_tables_at(written, 1, cfg.head_dim, cfg.rope_base as f64);
            let inputs = vec![
                ("x".into(), x),
                ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
                ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
                ("mask".into(), mask.clone()),
            ];
            let t = report::Stopwatch::start();
            // Output order: `hidden`, each layer's K row, its V row, logits.
            // The step reads the hidden row and the logits; the cache rows are
            // fed device-side.
            let body_o = match body.runner.run_resident_planes(
                &inputs,
                written,
                hkv,
                &graphs::plane_step_reads(cfg.layers),
            ) {
                Some(o) => o,
                None => {
                    cell_failed!("cell speak — body feed refused at frame {f}");
                }
            };
            body_times.push(t.ms());
            mask[written] = 1.0;
            written += 1;
            hidden = body_o[0].clone();
            let mut logits = body_o[1][..body_vocab].to_vec();
            suppress_service(&mut logits, body_vocab);
            apply_repetition(&mut logits, &code_seen, TTS_REPEAT_PENALTY);
            let t = report::Stopwatch::start();
            code0 = host::draw(&logits, top_k, top_p, temperature, 12345 + draw_index);
            sample_times.push(t.ms());
            draw_index += 1;
        }
        all_codes.push(frame_codes);
    }
    let frames = all_codes.len();
    let body_nodes = body.ops.iter().map(|(_, n)| n).sum();
    let pred_nodes: usize = pred_frame.ops.iter().map(|(_, n)| n).sum();
    phases.push(Phase::loop_times("body", &body_times, body_nodes));
    phases.push(Phase::loop_times("predictor", &pred_times, pred_nodes));
    phases.push(Phase::loop_times("sampler", &sample_times, 0));

    // The sound half: codes → codebook → wave transformer → DAC.
    let levels = pair
        .sound
        .plan("a.gen.wav.quant.rest.codebook.weight")
        .unwrap()
        .ggml_shape[2];
    let mut flat: Vec<f32> = Vec::with_capacity((levels + 1) * frames);
    for level in 0..=levels {
        for f in 0..frames {
            flat.push(all_codes[f][level] as f32);
        }
    }
    let parts = windows(frames, DAC_WINDOW, DAC_MARGIN);
    let groups = graphs::codec_groups();
    let (front_group, dac_groups) = groups.split_first().expect("the sound half has a front");
    let (_wave_d_in, wave_d_tfm) = pair
        .sound
        .plan("a.gen.wav.tfm.in_proj.weight")
        .map(|p| p.in_out())
        .unwrap_or((0, 0));
    let wave_layers = pair
        .sound
        .plans
        .iter()
        .filter(|p| p.name.starts_with("a.gen.wav.tfm.blk.") && p.name.ends_with(".ln1.weight"))
        .count();
    let wave_heads = 16usize;
    let wave_hd = pair
        .sound
        .plan("a.gen.wav.tfm.blk.0.attn_q.weight")
        .map(|p| p.ggml_shape[1] / wave_heads)
        .unwrap_or(0);
    let wave_hkv = wave_heads * wave_hd;
    let wave_k = pair
        .sound
        .plan("a.gen.wav.pre_conv.weight")
        .map(|p| p.ggml_shape[0])
        .unwrap_or(1);
    // The decoder's graphs are this call's: built on first use, reused for a repeat
    // of the same length, and dropped when this returns — before the round trip
    // loads the recogniser's own pair (`graphs::SoundGraphs`).
    let mut sound_cache = graphs::SoundGraphs::default();
    // A window runs its graph over the frames it holds: the part's own length for
    // the first window, a full chunk for every later one (`graphs::FrontGraphs`).
    let t = report::Stopwatch::start();
    let front = match graphs::FrontGraphs::build(
        &mut sound_cache,
        device,
        &pair.sound,
        frames,
        front_group.clone(),
    ) {
        Ok(f) => f,
        Err(e) => {
            cell_failed!("cell speak — sound {front_group:?} build: {e}");
        }
    };
    build_ms += t.ms();
    let prefix = graphs::FRONT_CARRY;
    let mut sound_phase_ms = 0f64;
    // The sound half's own weight stream: every window re-reads the graph it runs,
    // so the numerator is that graph's parameter bytes once per window.
    let mut sound_weight_bytes = 0u64;
    // The wave transformer carries a 72-frame window: 71 K/V frames per layer,
    // the last `pre_conv` frames and the global position. Running the front on
    // the whole part makes every frame depend on the part's total length (the
    // first 72 frames of a 330-row run differed from a 72-row run with the same
    // codes), and starting each window from zero state makes everything after
    // the first window garbage. Carry the state window to window.
    let (front_ch, front_per) = graphs::dac_stage_io(front_group.end, 1);
    let mut wave = vec![0f32; front_ch * frames * front_per];
    let mut front_stats = Vec::new();
    let mut front_nodes = 0usize;
    let mut o: Option<Outcome> = None;
    let mut kv_k = vec![0f32; wave_layers * prefix * wave_hkv];
    let mut kv_v = vec![0f32; wave_layers * prefix * wave_hkv];
    let mut left = vec![0f32; wave_d_tfm * (wave_k - 1)];
    for (a, b) in windows(frames, FRONT_WINDOW, 0)
        .into_iter()
        .map(|(_, _, a, b)| (a, b))
    {
        let core = b - a;
        // The first window runs its own length; every later one a full chunk, zero
        // padded when the part ends inside it (`graphs::FrontGraphs`).
        let (graph, run_len) = front.pick(a);
        let mut chunk = vec![0f32; (levels + 1) * run_len];
        for level in 0..=levels {
            for f in a..b {
                chunk[level * run_len + (f - a)] = flat[level * frames + f];
            }
        }
        let (cos, sin) = host::rope_tables_at(a, run_len, wave_hd, 1e4);
        let mut inputs = vec![
            ("codes".into(), chunk),
            ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
            ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
            ("left".into(), left.clone()),
        ];
        if a > 0 && prefix > 0 {
            inputs.push(("kpre".into(), kv_k.clone()));
            inputs.push(("vpre".into(), kv_v.clone()));
        }
        let run = graph.borrow_mut().run(&inputs, 1);
        if run.outs.len() < 4 {
            println!(
                "cell speak — the front returned {} outputs, expected waveform + k/v/left",
                run.outs.len()
            );
            return None;
        }
        front_nodes = front_nodes.max(run.ops.iter().map(|(_, n)| n).sum());
        let chunk = &run.outs[0];
        if chunk.len() != front_ch * run_len * front_per {
            println!(
                "cell speak — the front ran a {run_len}-frame window and returned {} samples, \
                 expected {} for {front_ch} channels × {run_len} frames × {front_per}",
                chunk.len(),
                front_ch * run_len * front_per
            );
            return None;
        }
        for ch in 0..front_ch {
            let src = ch * run_len * front_per;
            let dst = ch * frames * front_per + a * front_per;
            wave[dst..dst + core * front_per].copy_from_slice(&chunk[src..src + core * front_per]);
        }
        // Side outputs are published in this order: `left`, `knew`, `vnew`.
        // Carry the last 71 rows of a full chunk; a part's first window is one
        // whenever another window follows it, so the carry is filled before the
        // window that reads it.
        let knew = &run.outs[2];
        let vnew = &run.outs[3];
        let rows = knew.len() / (wave_layers * wave_hkv);
        if prefix > 0 && rows > prefix {
            for l in 0..wave_layers {
                let src = (l * rows + (rows - prefix)) * wave_hkv;
                let dst = l * prefix * wave_hkv;
                kv_k[dst..dst + prefix * wave_hkv]
                    .copy_from_slice(&knew[src..src + prefix * wave_hkv]);
                kv_v[dst..dst + prefix * wave_hkv]
                    .copy_from_slice(&vnew[src..src + prefix * wave_hkv]);
            }
        }
        left = run.outs[1].clone();
        front_stats.push((
            run.first_ms,
            run.step_ms,
            run.min_ms,
            run.max_ms,
            run.passes,
        ));
        sound_phase_ms += run.step_ms;
        sound_weight_bytes += graphs::stream_bytes(&graph.borrow().runner.graph);
        if o.is_none() {
            o = Some(run);
        }
    }
    phases.push(Phase::windows(
        "sound.front".to_string(),
        &front_stats,
        front_nodes,
    ));
    for group in dac_groups {
        let (ch_in, per_in) = graphs::dac_stage_io(group.start, 1);
        let expected = ch_in * frames * per_in;
        if wave.len() != expected {
            println!(
                "cell speak — sound {}..{} input is {} values, its stage reads {expected}",
                group.start,
                group.end,
                wave.len()
            );
            return None;
        }
        let (ch_out, per_out) = if group.end == graphs::DAC_STAGES {
            (1, graphs::samples_per_frame())
        } else {
            graphs::dac_stage_io(group.end, 1)
        };
        let t = report::Stopwatch::start();
        let mut lengths: Vec<usize> = parts.iter().map(|&(p0, p1, _, _)| p1 - p0).collect();
        lengths.sort_unstable();
        lengths.dedup();
        let mut graphs_by_len: Vec<(usize, graphs::SoundGraph)> = Vec::with_capacity(lengths.len());
        for &len in &lengths {
            match sound_cache.graph(device, &pair.sound, len, group.clone(), 0) {
                Ok(b) => graphs_by_len.push((len, b)),
                Err(e) => {
                    println!(
                        "cell speak — sound {}..{} at {len} frames build: {e}",
                        group.start, group.end
                    );
                    return None;
                }
            }
        }
        build_ms += t.ms();
        // The group's output is a tensor of its own shape, not a waveform until
        // the group that carries `post`. Stitch it channel-major at `per_out`,
        // exactly as the next group's input expects it.
        let mut stitched = vec![0f32; ch_out * frames * per_out];
        let mut stats = Vec::with_capacity(parts.len());
        let mut nodes = 0usize;
        for &(p0, p1, a, b) in &parts {
            let len = p1 - p0;
            let chunk = codec_slice(&wave, frames, p0, p1, per_in);
            let gi = graphs_by_len
                .iter()
                .position(|(l, _)| *l == len)
                .expect("part length built");
            let run = graphs_by_len[gi]
                .1
                .borrow_mut()
                .run(&[("codec_in".into(), chunk)], WARM_REPS);
            nodes = nodes.max(run.ops.iter().map(|(_, n)| n).sum());
            sound_phase_ms += run.step_ms;
            sound_weight_bytes += graphs::stream_bytes(&graphs_by_len[gi].1.borrow().runner.graph);
            stats.push((
                run.first_ms,
                run.step_ms,
                run.min_ms,
                run.max_ms,
                run.passes,
            ));
            let chunk = run.outs[0].clone();
            if o.as_ref().is_none_or(|p| run.arena_bytes > p.arena_bytes) {
                o = Some(run);
            }
            stitch_window(
                &mut stitched,
                &chunk,
                frames,
                (p0, p1, a, b),
                ch_out,
                per_out,
            );
        }
        phases.push(Phase::windows(
            format!("sound.{}-{}", group.start, group.end),
            &stats,
            nodes,
        ));
        wave = stitched;
    }
    if wave.len() != frames * graphs::samples_per_frame() {
        println!(
            "cell speak — the sound half left {} values, a waveform is {}",
            wave.len(),
            frames * graphs::samples_per_frame()
        );
        return None;
    }
    if let Ok(path) = std::env::var("STAND_SPEAK_WAV") {
        if let Err(e) = host::write_wav16(&path, &wave, 24_000) {
            println!("cell speak — writing `{path}`: {e}");
        }
    }
    let rss_sound_graphs = report::resident_rss();
    let o = o?;
    let body_total = phases
        .iter()
        .find(|p| p.name == "body")
        .map(|p| p.warm_ms)
        .unwrap_or(0.0);
    let pred_total = phases
        .iter()
        .find(|p| p.name == "predictor")
        .map(|p| p.warm_ms)
        .unwrap_or(0.0);
    let sampler_total = phases
        .iter()
        .find(|p| p.name == "sampler")
        .map(|p| p.warm_ms)
        .unwrap_or(0.0);
    let cold_ms: f64 = phases.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = phases.iter().map(|p| p.warm_ms).sum();
    let passes = phases.iter().map(|p| p.passes).sum();
    // The loop's own rate over its compulsory packed traffic is the record's
    // table, not a second figure here: `body_stream_bytes`/`predictor_stream_bytes`
    // are the numerators and their `_ms` the denominators.
    let fields = [
        ("rows", frames.to_string()),
        ("predictor_levels", (steps + 1).to_string()),
        ("sound_graphs", groups.len().to_string()),
        ("cache", "resident".to_string()),
        ("kv_bytes", (kv_bytes * 4).to_string()),
        ("dac", format!("{DAC_WINDOW}/{DAC_MARGIN}x{}", parts.len())),
        ("body_ms", fmt_ms(body_total)),
        ("predictor_ms", fmt_ms(pred_total)),
        ("sampler_ms", fmt_ms(sampler_total)),
        ("sound_phase_ms", fmt_ms(sound_phase_ms)),
        // The per-phase weight streams behind `MEASUREMENTS.md`'s rate table, one
        // unit each: the bytes a unit re-reads and the milliseconds it took, both
        // plain numbers so the record's rate is that division and nothing else.
        // The body is one frame; the predictor's figures are one frame of the walk
        // (all sixteen projecting levels, whose stream the record splits over them);
        // the sound half is one utterance, every window's graph counted.
        (
            "body_stream_bytes",
            graphs::stream_bytes(&body.runner.graph).to_string(),
        ),
        ("body_stream_ms", format!("{:.3}", body_total / rows as f64)),
        (
            "predictor_stream_bytes",
            graphs::stream_bytes(&pred_frame.runner.graph).to_string(),
        ),
        (
            "predictor_stream_ms",
            format!("{:.3}", pred_total / rows as f64),
        ),
        ("sound_stream_bytes", sound_weight_bytes.to_string()),
        ("sound_stream_ms", format!("{sound_phase_ms:.3}")),
        ("speaker_norm", format!("{speaker_norm:.4}")),
        ("text0_norm", format!("{text0_norm:.4}")),
        ("sampler_body", sampler0.to_string()),
        ("sampler_pred", sampler1.to_string()),
        ("prompt_digest", format!("{prompt_digest:016x}")),
        (
            "out_lens",
            format!("{:?}", p.outs.iter().map(|v| v.len()).collect::<Vec<_>>()),
        ),
        ("logits_digest", format!("{:016x}", digest(&p.outs[3]))),
        ("hidden_digest", format!("{:016x}", digest(&p.outs[0]))),
        (
            "first_codes",
            format!("{:?}", &all_codes[0][..all_codes[0].len().min(16)]),
        ),
    ];
    let peak = peak_rss();
    // Every owner above is still live at this point: `peak_rss` is the process's
    // high-water mark, so the record attributes the peak to whoever held it when it
    // was reached instead of summing plans that were never allocated whole, and
    // everything but this record is dropped on return — before the round trip loads
    // the recogniser's own pair.
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    println!(
        "speak residency start_mb={} prefill_mb={} body_mb={} loop_graphs_mb={} \
         sound_graphs_mb={} peak_mb={:.1} map_mb={:.1}",
        mb(rss_start),
        mb(rss_prefill),
        mb(rss_body),
        mb(rss_loop_graphs),
        mb(rss_sound_graphs),
        peak as f64 / 1e6,
        map_bytes as f64 / 1e6,
    );
    let r = rlx_wgpu::gguf_gpu::route_counts();
    // The shape of the *last* one-row dispatch, not of every call: `per_row` counts
    // the calls, `cols_per_workgroup`/`threads_per_workgroup`/`workgroups` describe
    // the one that ran last (see `gemv_one_row_shape`).
    let (cols_per_wg, threads, workgroups) = rlx_wgpu::gguf_gpu::gemv_one_row_shape()
        .map_or((0, 0, 0), |s| {
            (s.cols_per_workgroup, s.threads_per_workgroup, s.workgroups)
        });
    println!(
        "speak packed routes per_row={} host={} scratch={} grouped={} q1_0={} \
         cols_per_workgroup={cols_per_wg} threads_per_workgroup={threads} \
         workgroups={workgroups}",
        r.per_row, r.host, r.scratch, r.grouped, r.q1_0_gemm,
    );
    let c = graphs::device_counts();
    Some(Synthesis {
        frames,
        fields: fields.to_vec(),
        wave,
        prompt_digest,
        o,
        phases,
        steps: Steps {
            build_ms,
            cold_ms,
            warm_ms,
            passes,
        },
        peak,
        map_bytes,
        counters: (c.0 - c0.0, c.1 - c0.1),
    })
}

/// One embedding cell: the model's own tokenizer over the article, BOS and EOS
/// included, every row of the sequence and no padding, one graph run per
/// repetition. The two cells are separate rows of the record — `embed.256` and
/// `embed.8192` — and each is its own process, so neither is a sum of the other.
///
/// The sequence is `rows` long by construction — BOS, `rows - 2` content tokens,
/// EOS — and the cell refuses a text that tokenises to anything else rather
/// than padding it: the model's mean pooling has no padding mask, so a padded
/// row would be pooled as if it were text.
///
/// The graph is the whole encoder over all rows at once ([`graphs::encoder`]):
/// bidirectional attention, the twenty sliding layers banded at ±512, the four
/// global ones over every key. Which arrangement serves a dense square is the
/// plan's (`plan_attention`), not this cell's: the cooperative chunked form above
/// its 2 048-row floor, the band walk with the band switched off where the
/// adapter declines it.
fn embed(rows: usize) {
    let path = match crate::loader::models_dir() {
        Ok(dir) => dir.join(loader::EMBED_MODEL),
        Err(e) => {
            cell_failed!("cell embed.{rows} — model directory: {e}");
        }
    };
    let mut model = match Model::open(&path) {
        Ok(m) => m,
        Err(e) => {
            cell_failed!("cell embed.{rows} — open: {e}");
        }
    };
    // The route, before anything is built: the store rule hands the packed
    // linears over as f16 params only when this device admits the cooperative
    // f16 kernel for every multiply of this row count. The cell line names
    // whichever route ran, so its figures cannot be read as the other's.
    let gemm = model.f16_linears(rows);
    let model = Arc::new(model);
    let tokenizer = match Tokenizer::from_model(&model) {
        Ok(t) => t,
        Err(e) => {
            cell_failed!("cell embed.{rows} — tokenizer: {e}");
        }
    };
    let article = match crate::reference::article() {
        Ok(text) => text,
        Err(e) => {
            cell_failed!("cell embed.{rows} — article: {e}");
        }
    };
    let text = match crate::reference::cell_text(&article, rows) {
        Ok(text) => text,
        Err(e) => {
            cell_failed!("cell embed.{rows} — cell text: {e}");
        }
    };
    let ids = tokenizer.encode_with_specials(&text);
    let content = crate::reference::content_tokens(rows);
    if ids.len() != content + 2 {
        cell_failed!(
            "cell embed.{rows} — the text tokenises to {} ids, the cell is {content} content \
             tokens plus BOS and EOS; no padding is applied",
            ids.len()
        );
    }
    let cfg = model.gemma().clone();
    // The table's used rows only, in sequence order: the graph's input is
    // `[rows, d]` and the whole 262 144-row table never enters the process.
    let picks: Vec<(usize, usize)> = ids.iter().map(|&id| (0, id)).collect();
    let mut x = Vec::with_capacity(rows * cfg.d);
    match model.table_rows_f32("token_embd.weight", &picks, cfg.d) {
        Ok(table) => {
            for row in table {
                x.extend_from_slice(&row);
            }
        }
        Err(e) => {
            cell_failed!("cell embed.{rows} — token table rows: {e}");
        }
    }

    let device = rlx_runtime::Device::Gpu;
    let t = report::Stopwatch::start();
    let mut graph = match graphs::encoder(model.clone(), rows)
        .and_then(|m| graphs::compile_encoder(device, m, gemm))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell embed.{rows} — build: {e}");
        }
    };
    let build_ms = t.ms();
    // The memory ceiling, against this cell's own estimate before it launches:
    // the planner's arena, the params that were just uploaded, and the mapped
    // weight file. A graph that could only run by exhausting the resident-set
    // ceiling is refused here, naming the three figures and the ceiling.
    let ceiling_mib = crate::guard::max_rss_mib();
    let map_bytes = file_bytes(&model.path);
    let params = (graph.params_f32 + graph.params_packed) as u64;
    let estimate = graph.arena_bytes as u64 + params + map_bytes;
    if estimate > ceiling_mib.saturating_mul(1024 * 1024) {
        cell_failed!(
            "cell embed.{rows} — refuses to launch: arena {} + params {} + weights {} = {} is \
             over the {ceiling_mib} MiB ceiling (`MAHBOT_MAX_RSS_MIB`, default \
             {DEFAULT_MAX_RSS_MIB})",
            human(graph.arena_bytes as u64),
            human(params),
            human(map_bytes),
            human(estimate),
        );
    }

    let c0 = graphs::device_counts();
    let o = graph.run(&[("x".into(), x)], WARM_REPS);
    let values = l2(&o.outs[0]);
    println!(
        "cell embed.{rows} tokens={rows} content={content} norm={:.6}",
        values.iter().map(|v| v * v).sum::<f32>().sqrt()
    );
    let phases = vec![Phase::single(format!("embed.{rows}"), &o)];
    let steps = Steps {
        build_ms,
        cold_ms: o.first_ms,
        warm_ms: o.step_ms,
        passes: o.passes,
    };
    let c = graphs::device_counts();
    let peak = peak_rss();
    drop(graph);
    drop(model);
    report::release_free();
    line(
        &format!("embed.{rows}"),
        &[
            ("tokens", rows.to_string()),
            ("dim", cfg.d.to_string()),
            ("out", cfg.out_width.to_string()),
            ("gemm", gemm.token().to_string()),
        ],
        &o,
        &values,
        &phases,
        steps,
        peak,
        map_bytes,
        (c.0 - c0.0, c.1 - c0.1),
        0.0,
        (0.0, 0.0),
    );
}

// ── transcription ───────────────────────────────────────────────────

/// The longest pinned track: the whole multi-minute row, far past the 30 s the
/// single-shot convention reads and past the encoder's own position table.
const LONG_TRACK: &str = "speech/librivox/speech-librivox-0027.wav";
/// The constructed material's edge: a 10 ms window whose RMS is above `3e-3` is
/// the excerpt's own beginning or end, so the trailing silence of the read is cut
/// without cutting a phoneme.
const TRIM_WINDOW: usize = 160;
const TRIM_RMS: f32 = 3e-3;
/// The paste plan: a 1.024 s lead-in, then four copies of the excerpt separated by
/// these gaps. Every one is a multiple of the detector's 256-sample hop, so the
/// true onsets are the paste offsets exactly.
const PASTE_START: usize = 16_384;
const PASTE_GAPS_S: [f64; 4] = [1.024, 2.048, 1.536, 3.072];

/// The transcription cell: one [`crate::transcribe::Transcriber`] over the pinned
/// 33-file corpus, the eight-clip music pool, constructed boundary material and the
/// longest pinned track — the record's third surface.
///
/// What it establishes. The constructed material's truth is its construction: four
/// pastes of the pinned clip into digital silence, separated by gaps past
/// `MERGE_GAP_HOPS`, have to come back as nine regions whose four speech onsets sit
/// within one detector hop of the paste offsets, and the same signal at a different
/// gain has to keep every boundary where it was — conditioning is time-preserving
/// or the plan is wrong. The music pool's truth is its split: nothing in
/// `instrumental/` may come back with a reliable transcript — a transcript the
/// guards rejected is returned marked unreliable, beside its label, and counted
/// rather than failed. Every region of every file is
/// held to the same invariant: a text or a label, never a reliable transcript
/// beside a label, and regions that tile the input.
///
/// What it does not establish. Word correctness — the pinned clip's two parity rows
/// are reported, not gated: they count word positions that moved between two
/// readings of the same speech, which a recogniser produces for one second of
/// different audio around a cut. Nor does it say that a transcript is right: the
/// `invented` row counts regions that answered over the noise pool, not how wrong
/// those answers are.
/// The end-to-end bar the transcription call is measured against: milliseconds of call
/// per second of audio, hand-over to text, and the tolerance the plan allows above it.
/// The record states the measured figure against these, because a figure with no bar
/// beside it cannot be read as fast or slow.
const TRANSCRIBE_BAR_MS_PER_AUDIO_S: f64 = 100.0;
const TRANSCRIBE_BAR_TOLERANCE_MS_PER_AUDIO_S: f64 = 103.0;
/// The quality bar the same page states on its other headline figure: the share of the
/// detector's marked speech that may come back with no transcript.
const TRANSCRIBE_TEXTLESS_BAR_PCT: f64 = 2.0;

fn transcribe_cell() {
    let started = report::Stopwatch::start();
    // The counters as the cell found them: this cell's own submits and waits are
    // the difference, not every call the process made before it.
    let counts_at_start = graphs::device_counts();
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(e) => cell_failed!("cell transcribe — the model directory: {e:#}"),
    };
    let (body, sound) = match loader::pair_files(Role::Recognition) {
        Some(pair) => pair,
        None => cell_failed!("cell transcribe — no recognition pair is pinned"),
    };
    let pair = match Pair::open(dir.join(body), dir.join(sound)) {
        Ok(pair) => Arc::new(pair),
        Err(e) => cell_failed!("cell transcribe — the recognition pair: {e:#}"),
    };
    // The tagger is the caller's, opened from the same directory as the pair: the
    // transcriber never resolves a model path of its own.
    let ced_path = dir.join(crate::ced::CED_MODEL);
    let open = report::Stopwatch::start();
    let tagger = match crate::ced::Tagger::open(&ced_path) {
        Ok(t) => t,
        Err(e) => cell_failed!(
            "cell transcribe — the tagger (`{}` beside the recognition pair): {e:#}",
            crate::ced::CED_MODEL
        ),
    };
    let mut transcriber = match crate::transcribe::Transcriber::open(pair.clone(), tagger) {
        Ok(t) => t,
        Err(e) => cell_failed!("cell transcribe — the transcriber: {e:#}"),
    };
    let open_ms = open.ms();

    // ── the inputs, verified before anything is measured ────────────
    let corpus_dir = match std::env::var_os(CORPUS_ENV) {
        Some(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
        _ => cell_failed!(
            "cell transcribe — {CORPUS_ENV} is not set: point it at the pinned real-audio corpus \
             the wake bench reads"
        ),
    };
    let mut corpus: Vec<(&'static str, std::path::PathBuf)> = Vec::new();
    let mut corpus_s = 0f64;
    if let Err(e) = corpus_pins(&corpus_dir) {
        cell_failed!("cell transcribe — the pinned corpus: {e:#}");
    }
    for (name, ..) in crate::bench::REAL_AUDIO_FILES {
        let path = corpus_dir.join(name);
        // The duration is the reader's own count, not arithmetic on the file's
        // bytes: the reader the measurement uses decides how long the audio is.
        let audio = read_whole(&path, name);
        corpus_s += audio.len() as f64 / f64::from(crate::transcribe::RATE);
        corpus.push((*name, path));
    }
    let music_dir = match music::dir() {
        Ok(dir) => dir,
        Err(e) => cell_failed!("cell transcribe — {e:#}"),
    };
    let clips = match music::verify() {
        Ok(paths) => paths,
        Err(e) => cell_failed!("cell transcribe — the music pool: {e:#}"),
    };
    let voice = match loader::voice_wav() {
        Ok(path) => path,
        Err(e) => cell_failed!("cell transcribe — the voice reference: {e:#}"),
    };
    println!(
        "transcribe inputs corpus={} files={} audio_s={corpus_s:.1} music={} clips={} voice={} \
         open_ms={open_ms:.1}",
        crate::measurements::short_path(&corpus_dir),
        corpus.len(),
        crate::measurements::short_path(&music_dir),
        clips.len(),
        crate::measurements::short_path(&voice),
    );
    // The files this cell binds, hashed here — what is on disk, not what the pins
    // say should be, and a row that differs says which pin it differs from. Only
    // the tagger has a source and revision pinned in the tree, and the revision is
    // printed whole: a prefix is not a revision a file can be fetched from.
    transcribe_artefact(&dir.join(body), None, "-");
    transcribe_artefact(&dir.join(sound), None, "-");
    let (repo, revision, sha256) = crate::ced::CED_MODEL_SOURCE;
    transcribe_artefact(&ced_path, Some(sha256), &format!("{repo}@{revision}"));

    // ── (0) a real 10 s and a real 30 s voice message, cold and warm ─
    // The material is the first pinned speech recording long enough for both cuts; the
    // text each call returns is printed, so its correctness is visible rather than
    // asserted. `cold` empties the device's shared weight buffer, loads the models
    // inside the measurement and takes the call — no weights on the device when it
    // starts — and `warm` is the same call on the transcriber the cold one left behind.
    let message_source = corpus
        .iter()
        .find(|(name, _)| name.starts_with("speech/"))
        .map(|(name, path)| (*name, path.clone()));
    let Some((message_name, message_path)) = message_source else {
        cell_failed!("cell transcribe — the pinned corpus holds no speech recording");
    };
    let message_audio = read_whole(&message_path, message_name);
    for seconds in [10.0f64, 30.0] {
        let samples =
            ((seconds * f64::from(crate::transcribe::RATE)) as usize).min(message_audio.len());
        if (samples as f64 / f64::from(crate::transcribe::RATE)) < seconds - 0.5 {
            continue;
        }
        let audio = &message_audio[..samples];
        // Cold: the models are loaded by the call, with the device's shared weight
        // buffer emptied first so the compilations upload what they need.
        rlx_wgpu::buffer::clear_shared_weights();
        let t = report::Stopwatch::start();
        let mut fresh = or_exit(
            "the models",
            load_transcriber(&pair.body.path, &pair.sound.path, &ced_path),
        );
        let load_ms = t.ms();
        let t = report::Stopwatch::start();
        let out = or_exit(
            "the message",
            fresh.transcribe(audio, crate::transcribe::RATE),
        );
        let call_ms = t.ms();
        check_regions(&out, "the message");
        let text = spoken(&out);
        println!(
            "transcribe message s={seconds:.0} pass=cold load_ms={load_ms:.1} call_ms={call_ms:.1} \
             total_ms={:.1} audio_s={:.3} ms_per_audio_s={:.1} regions={} words={} text=\"{}\"",
            load_ms + call_ms,
            samples as f64 / f64::from(crate::transcribe::RATE),
            (load_ms + call_ms) / seconds,
            out.regions.len(),
            text.split_whitespace().count(),
            prefix(&text),
        );
        let t = report::Stopwatch::start();
        let out = or_exit(
            "the message",
            fresh.transcribe(audio, crate::transcribe::RATE),
        );
        let warm_ms = t.ms();
        check_regions(&out, "the message");
        let text = spoken(&out);
        println!(
            "transcribe message s={seconds:.0} pass=warm call_ms={warm_ms:.1} audio_s={:.3} \
             ms_per_audio_s={:.1} regions={} words={} text=\"{}\"",
            samples as f64 / f64::from(crate::transcribe::RATE),
            warm_ms / seconds,
            out.regions.len(),
            text.split_whitespace().count(),
            prefix(&text),
        );
    }

    // ── (a) the pinned clip's parity with the single-shot answer ─────
    // The clip the listen cell reads: the same file, at the same offset
    // (`STAND_ASR_OFFSET`, 0 unless a run names another recording) and the same
    // length, so the two surfaces' answers describe the same audio.
    let raw = match host::wav_mono_16k(
        &voice.display().to_string(),
        listen_offset(),
        SPEECH_SAMPLES,
    ) {
        Ok(audio) => audio,
        Err(e) => cell_failed!("cell transcribe — the pinned clip: {e:#}"),
    };
    let mut conditioned = raw.clone();
    prepare_speech(&mut conditioned);
    let pinned = or_exit(
        "the pinned clip",
        transcriber.transcribe(&conditioned, crate::transcribe::RATE),
    );
    check_regions(&pinned, "the pinned clip");
    let segmented = spoken(&pinned);
    let (moved, produced, pinned_words) =
        word_difference(&segmented, crate::reference::LISTEN_TRANSCRIPT);
    // The segments the comparison covers, beside it: the segmented call transcribes
    // the speech spans it detects, so its text describes less audio than the
    // single-shot read of the same file, and the row says how much.
    let covered = transcriber.costs().speech_s;
    println!(
        "transcribe parity moved={moved} words={produced} pinned={pinned_words} same_count={} \
         clip_s={:.3} transcribed_s={covered:.3} text=\"{}\"",
        produced == pinned_words,
        pinned.duration_s(),
        prefix(&segmented),
    );
    // The mechanism, on the same audio: the segmented call reads only the span it
    // detected, so the same span through the single-shot walk at its own exact frame
    // count says whether the difference above is the span's price or the segmented
    // input's.
    let compared: Vec<&crate::transcribe::Region> = pinned
        .regions
        .iter()
        .filter(|region| region.text.is_some())
        .collect();
    match compared.first() {
        Some(region) => {
            // The transcriber's own tokenizer: the same file, already opened above,
            // so the parity row cannot read the prompt table through a second parse.
            let single = or_exit(
                "the parity span",
                listen::recognise(
                    &pair,
                    transcriber.tokenizer(),
                    conditioned[region.start..region.end].to_vec(),
                    0,
                ),
            );
            let (moved, produced, single_words) =
                word_difference(region.text.as_deref().unwrap_or_default(), &single.text);
            println!(
                "transcribe parity-span span_s={:.3} moved={moved} words={produced} \
                 single={single_words} same_count={} text=\"{}\"",
                (region.end - region.start) as f64 / f64::from(crate::transcribe::RATE),
                produced == single_words,
                prefix(region.text.as_deref().unwrap_or_default()),
            );
        }
        None => println!("transcribe parity-span span_s=0 moved=0 words=0 single=0"),
    }

    // ── (b) boundary material whose truth is the construction ───────
    // The excerpt is the pinned clip as it is read — before `prepare_speech`
    // conditions it, so its edges are the recording's own — trimmed to those edges
    // and cut to the detector's hop, so every paste offset is on its grid.
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let edges: Vec<usize> = raw
        .chunks_exact(TRIM_WINDOW)
        .enumerate()
        .filter(|(_, window)| rms(window) > TRIM_RMS)
        .map(|(i, _)| i)
        .collect();
    let (first, last) = match (edges.first(), edges.last()) {
        (Some(first), Some(last)) => (*first, *last),
        _ => cell_failed!(
            "cell transcribe — the pinned clip has no 10 ms window above {TRIM_RMS}: nothing to \
             paste"
        ),
    };
    let mut excerpt = raw[first * TRIM_WINDOW..(last + 1) * TRIM_WINDOW].to_vec();
    excerpt.truncate(excerpt.len() / crate::vad::HOP_LENGTH * crate::vad::HOP_LENGTH);
    let gaps: Vec<usize> = PASTE_GAPS_S
        .iter()
        .map(|seconds| (seconds * f64::from(crate::transcribe::RATE)).round() as usize)
        .collect();
    let mut signal = vec![0f32; PASTE_START + 4 * excerpt.len() + gaps.iter().sum::<usize>()];
    let mut truth: Vec<usize> = Vec::with_capacity(4);
    let mut at = PASTE_START;
    for gap in &gaps {
        truth.push(at);
        signal[at..at + excerpt.len()].copy_from_slice(&excerpt);
        at += excerpt.len() + gap;
    }
    let pasted = or_exit(
        "the constructed material",
        transcriber.transcribe(&signal, crate::transcribe::RATE),
    );
    check_regions(&pasted, "the constructed material");
    // The detector's own hop in milliseconds: a paste sits on its grid, so the
    // onset error this material can show is smaller than one of these.
    let hop_ms = 1000.0 * crate::vad::HOP_LENGTH as f64 / f64::from(crate::transcribe::RATE);
    // A sample difference is a time through the rate, not through the hop: the
    // bound below is a duration, so the error beside it has to be one too.
    let ms_per_sample = 1000.0 / f64::from(crate::transcribe::RATE);
    // Each paste's onset against the region that begins nearest it: the construction
    // is the truth, and a cut may put more than one region inside a paste, so the
    // check is per onset rather than per region index.
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
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let max_abs = errors
        .iter()
        .fold(0f64, |most, error| most.max(error.abs()));
    println!(
        "transcribe boundary pastes={} regions={} err_ms={} mean_ms={mean:+.1} max_abs_ms={max_abs:.1}",
        truth.len(),
        pasted.regions.len(),
        errors
            .iter()
            .map(|error| format!("{error:+.1}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    if max_abs > hop_ms {
        cell_failed!(
            "cell transcribe — a paste onset is {max_abs:.1} ms from its own offset, past the \
             detector's one {hop_ms:.0} ms hop: the pastes are on its grid, so a larger error is a \
             defect"
        );
    }
    // Every paste has to be read: a paste that produced no transcript is speech the
    // segmented path lost, and the boundary material is the one input where that cannot
    // be the audio's fault.
    let read = pasted
        .regions
        .iter()
        .filter(|region| region.text.is_some())
        .count();
    if read < truth.len() {
        cell_failed!(
            "cell transcribe — {read} of the {} pasted excerpts came back with a transcript",
            truth.len()
        );
    }
    // Conditioning is one gain over the whole input and may not move a boundary,
    // whatever gain it is: the same signal at a level the library leaves alone (every
    // cell above `host::SPEECH_FLOOR` is taken as it is) and at the quiet level it
    // lifts to `host::SPEECH_TARGET` must tile identically. The two arms condition to
    // different levels by construction, so this compares two levels of one signal
    // rather than one buffer read twice.
    let peak = signal
        .iter()
        .fold(0f32, |most, value| most.max(value.abs()));
    let untouched = (crate::host::SPEECH_FLOOR + crate::host::SPEECH_TARGET) / 2.0;
    let steady_copy: Vec<f32> = signal
        .iter()
        .map(|value| value * (untouched / peak))
        .collect();
    let steady = or_exit(
        "the steady copy",
        transcriber.transcribe(&steady_copy, crate::transcribe::RATE),
    );
    check_regions(&steady, "the steady copy");
    let bounds = |out: &crate::transcribe::Transcription| {
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
        "transcribe boundary gain={:.4} lifted_gain={:.4} moved={moved} identical={}",
        steady.gain,
        pasted.gain,
        moved == 0,
    );
    // The two arms have to have conditioned differently, or nothing was compared:
    // a copy that lands on the same level as the lifted one is the same waveform.
    if steady.gain == pasted.gain {
        cell_failed!(
            "cell transcribe — both copies conditioned at gain {:.4}, so no level was compared: \
             the copy above the floor is taken as it is at {:.2}, and the quiet one is lifted to \
             {:.2}",
            steady.gain,
            untouched,
            crate::host::SPEECH_TARGET,
        );
    }
    if moved > 0 {
        cell_failed!(
            "cell transcribe — the copy at gain {:.4} moved {moved} of {} boundaries away from the \
             one at {:.4}: a gain is time-preserving and may not move a start or an end",
            steady.gain,
            pasted_bounds.len(),
            pasted.gain,
        );
    }

    // ── (c) the music pool and the song rule ────────────────────────
    let (mut music_audio_s, mut music_wall_s) = (0f64, 0f64);
    let mut music_suppressed_s = 0f64;
    let (mut instrumental_any_text, mut vocal_reliable) = (0usize, 0usize);
    for (clip, path) in music::CLIPS.iter().zip(&clips) {
        let name = clip.path;
        let read = report::Stopwatch::start();
        let audio = read_whole(path, name);
        let read_ms = read.ms();
        let audio_s = audio.len() as f64 / f64::from(crate::transcribe::RATE);
        let t = report::Stopwatch::start();
        let out = or_exit(
            name,
            transcriber.transcribe(&audio, crate::transcribe::RATE),
        );
        let wall_ms = t.ms();
        let c = transcriber.costs();
        check_regions(&out, name);
        music_audio_s += audio_s;
        music_wall_s += wall_ms / 1000.0;
        music_suppressed_s += c.suppressed_s;
        let text = spoken(&out);
        let transcribed = !text.is_empty();
        let labels = labels_of(&out);
        println!(
            "transcribe music {name} audio_s={audio_s:.2} regions={} text_chars={} labels={labels} \
             suppressed_s={:.1} read_ms={read_ms:.1} wall_ms={wall_ms:.1} s_per_audio_s={:.4}",
            out.regions.len(),
            text.chars().count(),
            c.suppressed_s,
            wall_ms / 1000.0 / audio_s,
        );
        if name.starts_with("instrumental/") {
            // The defect the pool exists to catch is a transcript the guards
            // *trust* here. A rejected one is the settled contract: the transcript
            // is returned anyway, marked `reliable: false`, beside the region's
            // label, so it is counted and not failed. A region with no text has no
            // label missing — `check_regions` already holds every region to a text or
            // a label.
            let trusted = out
                .regions
                .iter()
                .any(|region| region.text.is_some() && region.reliable);
            instrumental_any_text += usize::from(transcribed);
            if trusted {
                cell_failed!(
                    "cell transcribe — {name} is instrumental and came back with a reliable \
                     transcript: \"{}\"",
                    prefix(&text)
                );
            }
        } else {
            let reliable = out
                .regions
                .iter()
                .filter(|region| region.text.is_some() && region.reliable)
                .count();
            // The aggregate counts clips whose transcript survived the guards: a
            // clip that only produced a loop is the `transcribed=false` row beside
            // it, not a transcript the pool delivered.
            vocal_reliable += usize::from(reliable > 0);
            // No vocal clip may come back empty: `check_regions` above already
            // holds every region to a text or a label, and the row says which way
            // each clip went — a solfège clip with no transcript is not a miss.
            println!(
                "transcribe music-report {name} transcribed={transcribed} \
                 reliable_regions={reliable} suppressed_s={:.1} labels={labels} text=\"{}\"",
                c.suppressed_s,
                prefix(&text)
            );
        }
    }
    // The two pools' counters count different things, because the gates do: an
    // instrumental clip may carry a transcript the guards *rejected* (any text is
    // counted and none may be trusted), a vocal clip is transcribed only where its
    // text survived. `suppressed_s` is what the label rule kept out of the recogniser
    // on this pool: a clip with no transcript and suppressed seconds was never asked,
    // which is what tells the rule's verdict from the recogniser's.
    println!(
        "transcribe music-total clips={} audio_s={music_audio_s:.1} wall_s={music_wall_s:.1} \
         s_per_audio_s={:.4} suppressed_s={music_suppressed_s:.1} \
         instrumental_any_text={instrumental_any_text} vocal_reliable={vocal_reliable}",
        clips.len(),
        music_wall_s / music_audio_s,
    );

    // ── (e) the corpus, with (f) on the same pass and (g) inside it ──
    // The headline: one call per file, hand-over to text, with every step the text
    // cannot exist without inside it — the file read, the gain and resample, the mask,
    // the tagger, the cutting, every compilation the call needs, the recognition
    // passes, the guards and the joins. The first file of the pass is read by a
    // transcriber loaded inside the measurement (the pass's cold figure); the rest run
    // on it warm.
    let mut corpus_audio_s = 0f64;
    let mut corpus_speech_s = 0f64;
    let mut corpus_wall_ms = 0f64;
    let mut corpus_read_ms = 0f64;
    let (mut regions, mut text_regions, mut label_regions) = (0usize, 0usize, 0usize);
    let mut costs = crate::transcribe::Costs::default();
    let (mut corpus_masked_s, mut corpus_uncovered_s) = (0f64, 0f64);
    let (mut speech_pool_speech_s, mut speech_pool_textless_s, mut speech_pool_flagged_s) =
        (0f64, 0f64, 0f64);
    let (mut speech_pool_textless_voiced_s, mut speech_pool_textless_other_s) = (0f64, 0f64);
    let mut speech_pool_suppressed_s = 0f64;
    let mut seams = 0usize;
    let (mut speech_pool_chunks, mut speech_pool_textless_chunks) = (0usize, 0usize);
    let (mut noise_pool_speech_s, mut noise_pool_textless_s) = (0f64, 0f64);
    let (mut noise_pool_marked_s, mut noise_pool_suppressed_s) = (0f64, 0f64);
    let (mut invented_text, mut invented_reliable, mut invented_marked, mut invented_chars) =
        (0usize, 0usize, 0usize, 0usize);
    let mut invented_examples: Vec<String> = Vec::new();
    // What the tagger says the noise files are: the pool the "no speech" label
    // has to be plausible on, counted over every labelled region of it.
    let mut noise_labels: std::collections::BTreeMap<String, usize> = Default::default();
    let mut long: Option<(f64, f64, usize, f64, f64, f64)> = None;
    // The pass's cold figure: the first file, with the models loaded inside the
    // measurement — and with the device's shared weight buffer emptied first, so the
    // compilations really do upload the weights: an entry left by the calls above (the
    // messages, the parity clip, the music pool) would otherwise price the upload at
    // nothing and the figure would not be the cold one it says it is.
    drop(transcriber);
    rlx_wgpu::buffer::clear_shared_weights();
    let (cold_name, cold_path) = match corpus.first() {
        Some((name, path)) => (*name, path.clone()),
        None => cell_failed!("cell transcribe — the pinned corpus is empty"),
    };
    let t = report::Stopwatch::start();
    let mut transcriber = or_exit(
        "the models",
        load_transcriber(&pair.body.path, &pair.sound.path, &ced_path),
    );
    let cold_load_ms = t.ms();
    let t = report::Stopwatch::start();
    let cold = or_exit(cold_name, transcriber.transcribe_file(&cold_path));
    let cold_call_ms = t.ms();
    check_regions(&cold, cold_name);
    let cold_s = cold.duration_s();
    let cold_text = spoken(&cold);
    println!(
        "transcribe corpus-cold file={cold_name} load_ms={cold_load_ms:.1} call_ms={cold_call_ms:.1} \
         audio_s={cold_s:.1} ms_per_audio_s={:.1} words={} text=\"{}\"",
        (cold_load_ms + cold_call_ms) / cold_s,
        cold_text.split_whitespace().count(),
        prefix(&cold_text),
    );
    for (name, path) in &corpus {
        // The cold call above already read this file through the same path; the pass
        // reads it again so every row is the same measurement.
        let t = report::Stopwatch::start();
        let out = or_exit(name, transcriber.transcribe_file(path));
        let wall_ms = t.ms();
        let c = transcriber.costs();
        let seconds = out.duration_s();
        // The tiling gate every file is held to, the multi-minute track included:
        // the first region starts at 0 and the last ends at the input's length.
        check_regions(&out, name);
        corpus_audio_s += seconds;
        corpus_wall_ms += wall_ms;
        corpus_read_ms += c.read_ms;
        corpus_speech_s += c.speech_s;
        corpus_masked_s += c.masked_s;
        corpus_uncovered_s += c.uncovered_s;
        regions += out.regions.len();
        text_regions += out.regions.iter().filter(|r| r.text.is_some()).count();
        label_regions += out.regions.iter().filter(|r| r.label.is_some()).count();
        costs.read_ms += c.read_ms;
        costs.segment_ms += c.segment_ms;
        costs.tag_ms += c.tag_ms;
        costs.build_ms += c.build_ms;
        costs.recognise_ms += c.recognise_ms;
        costs.chunks += c.chunks;
        costs.shapes += c.shapes;
        costs.windows += c.windows;
        if name == &LONG_TRACK {
            long = Some((
                c.read_ms,
                seconds,
                out.regions.len(),
                c.build_ms,
                c.recognise_ms,
                wall_ms / 1000.0,
            ));
        }
        if name.starts_with("speech/") {
            // The pool where there is speech to lose: how much of the detected
            // speech came back with no transcript at all, how much the guards
            // rejected, and — for every textless region — what the tagger says the
            // audio there is, which is what tells lost speech from the detector's
            // own artifact. A file that lost any speech is printed on its own row
            // below, so the aggregate can be attributed rather than taken on trust.
            speech_pool_speech_s += c.speech_s;
            speech_pool_textless_s += c.textless_s;
            speech_pool_textless_voiced_s += c.textless_voiced_s;
            speech_pool_textless_other_s += c.textless_other_s;
            speech_pool_suppressed_s += c.suppressed_s;
            speech_pool_flagged_s += c.flagged_s;
            speech_pool_chunks += c.chunks;
            speech_pool_textless_chunks += c.textless_chunks;
            // A region that still begins with the words the region before it ended
            // with carries one of them twice, which the `seams` row counts.
            seams += seam_pairs(&out);
            if c.textless_s > 0.0 {
                println!(
                    "transcribe textless-file {name} audio_s={seconds:.1} chunks={} \
                     textless_chunks={} textless_s={:.1} labelled_voiced_s={:.1} \
                     labelled_other_s={:.1} e2e_ms={wall_ms:.1} labels={}",
                    c.chunks,
                    c.textless_chunks,
                    c.textless_s,
                    c.textless_voiced_s,
                    c.textless_other_s,
                    textless_labels(&out),
                );
            }
        } else {
            noise_pool_speech_s += c.speech_s;
            noise_pool_textless_s += c.textless_s;
            noise_pool_marked_s += c.masked_s;
            noise_pool_suppressed_s += c.suppressed_s;
            for region in &out.regions {
                let Some(text) = region.text.as_deref() else {
                    continue;
                };
                invented_text += 1;
                if region.reliable {
                    invented_reliable += 1;
                } else {
                    invented_marked += 1;
                }
                invented_chars += text.trim().chars().count();
                if invented_examples.len() < 3 {
                    invented_examples.push(prefix(text));
                }
            }
            for region in &out.regions {
                if let Some(label) = &region.label {
                    *noise_labels.entry(label.clone()).or_default() += 1;
                }
            }
        }
    }
    let minutes = corpus_audio_s / 60.0;
    let speech_files = corpus
        .iter()
        .filter(|(name, _)| name.starts_with("speech/"))
        .count();
    let noise_files = corpus.len() - speech_files;
    println!(
        "transcribe corpus files={} audio_s={corpus_audio_s:.1} speech_s={corpus_speech_s:.1} \
         regions={regions} text_regions={text_regions} label_regions={label_regions} \
         wall_s={:.1}",
        corpus.len(),
        corpus_wall_ms / 1000.0,
    );
    // The headline figure: everything the call does, from hand-over to text, per second
    // of audio. Every other row of this cell is a breakdown for the agents that
    // optimise, and none of them is the recognition number.
    println!(
        "transcribe e2e scope=corpus files={} audio_s={corpus_audio_s:.1} wall_ms={:.1} \
         ms_per_audio_s={:.1} speech_s={corpus_speech_s:.1}",
        corpus.len(),
        corpus_wall_ms,
        corpus_wall_ms / corpus_audio_s,
    );
    // The breakdown, per audio minute, and what it does not attribute: the file read,
    // the mask and plan, the tagger, the compilations, the recognition passes. The
    // remainder — gain and resample, the guards, the joins, the retimed result — is
    // reported as unattributed rather than split by guesswork.
    let unattributed_ms = (corpus_wall_ms
        - (costs.read_ms + costs.segment_ms + costs.tag_ms + costs.build_ms + costs.recognise_ms))
        .max(0.0);
    println!(
        "transcribe breakdown read_ms_per_audio_min={:.1} mask_plan_ms_per_audio_min={:.1} \
         tagger_ms_per_audio_min={:.1} compile_ms_per_audio_min={:.1} \
         passes_ms_per_audio_min={:.1} unattributed_ms_per_audio_min={:.1} \
         read_ms={:.1} chunks={} shapes={} windows={}",
        costs.read_ms / minutes,
        costs.segment_ms / minutes,
        costs.tag_ms / minutes,
        costs.build_ms / minutes,
        costs.recognise_ms / minutes,
        unattributed_ms / minutes,
        corpus_read_ms,
        costs.chunks,
        costs.shapes,
        costs.windows,
    );
    // The guard and segmentation numbers this run was taken with: the direction
    // requirement ("filter more rather than less") is a choice, and a choice that is
    // not printed cannot be reviewed.
    println!(
        "transcribe tuning min_speech_hops={} merge_gap_hops={} hangover_hops={} \
         cut_s={} cut_early_s={:.3} cut_late_s={:.3} cut_slack_pct={} \
         frame_alignment={} min_chunk_s={:.1} repeat_gram={} repeat_limit={} \
         word_repeat_limit={} density_ceiling={:.1} chunk_tokens_per_s={:.1} min_chunk_tokens={}",
        crate::transcribe::MIN_SPEECH_HOPS,
        crate::transcribe::MERGE_GAP_HOPS,
        crate::transcribe::HANGOVER_HOPS,
        crate::transcribe::CUT_SECONDS,
        crate::transcribe::CUT_EARLY_HOPS as f64 * crate::vad::HOP_LENGTH as f64
            / f64::from(crate::transcribe::RATE),
        crate::transcribe::CUT_LATE_HOPS as f64 * crate::vad::HOP_LENGTH as f64
            / f64::from(crate::transcribe::RATE),
        crate::transcribe::CUT_SLACK_PCT,
        crate::transcribe::FRAME_ALIGNMENT,
        crate::transcribe::MIN_CHUNK_SAMPLES as f64 / f64::from(crate::transcribe::RATE),
        crate::transcribe::REPEAT_GRAM,
        crate::transcribe::REPEAT_LIMIT,
        crate::transcribe::WORD_REPEAT_LIMIT,
        crate::transcribe::DENSITY_CEILING,
        crate::transcribe::CHUNK_TOKENS_PER_S,
        crate::transcribe::MIN_CHUNK_TOKENS,
    );
    // The speech pool, where the quality bar has an answer: how much of the detected
    // speech came back with no transcript, and how much came back with text the guards
    // would not vouch for. `textless_s` counts a region's whole seconds, the hangover
    // into silence included, so it is measured against the seconds the passes covered
    // rather than being the detector's own count.
    // `textless_voiced_s` and `textless_other_s` split the textless seconds by what the
    // tagger says the audio is: words could be there, or the detector marked something
    // that is not words. The noise pool's figures are counted apart: there the
    // detector's marks are mostly false positives, so a textless second is the
    // detector's, not lost speech, and `noise_suppressed_s` says how many marked seconds
    // the tagger's labels took out of it.
    let textless_pct = speech_pool_textless_s / speech_pool_speech_s.max(f64::MIN_POSITIVE) * 100.0;
    println!(
        "transcribe speech files={speech_files} speech_s={speech_pool_speech_s:.1} \
         chunks={speech_pool_chunks} textless_s={:.1} textless_pct={textless_pct:.2} \
         textless_bar_pct={TRANSCRIBE_TEXTLESS_BAR_PCT:.2} \
         textless_gap_pct={:.2} \
         textless_chunks={speech_pool_textless_chunks} \
         textless_voiced_s={speech_pool_textless_voiced_s:.1} \
         textless_other_s={speech_pool_textless_other_s:.1} \
         suppressed_s={speech_pool_suppressed_s:.1} flagged_s={:.1} flagged_pct={:.2} \
         noise_files={noise_files} noise_marked_s={noise_pool_marked_s:.1} \
         noise_suppressed_s={noise_pool_suppressed_s:.1} \
         noise_speech_s={noise_pool_speech_s:.1} noise_textless_s={noise_pool_textless_s:.1}",
        speech_pool_textless_s,
        textless_pct - TRANSCRIBE_TEXTLESS_BAR_PCT,
        speech_pool_flagged_s,
        speech_pool_flagged_s / speech_pool_speech_s.max(f64::MIN_POSITIVE) * 100.0,
    );
    // A seam between two adjacent transcripts that shares three or more whole words: a
    // reading carried twice, which is what a lead-in or a piece join could once produce.
    // The plan gives each chunk disjoint audio, so this is a reading to look at rather
    // than a state the path can be in — it is printed beside the pool's figures, not
    // held against the round.
    println!(
        "transcribe seams seams={seams} seam_words={}",
        crate::transcribe::SEAM_WORDS
    );
    // The loss in audio, which no paraphrase can hide: the seconds the detector
    // marked speech that no recognition pass covered.
    println!(
        "transcribe coverage masked_s={corpus_masked_s:.1} covered_s={corpus_speech_s:.1} \
         uncovered_s={corpus_uncovered_s:.1} uncovered_pct={:.2}",
        100.0 * corpus_uncovered_s / corpus_masked_s.max(f64::MIN_POSITIVE),
    );
    println!(
        "transcribe invented files={noise_files} text_regions={invented_text} \
         reliable_regions={invented_reliable} marked_regions={invented_marked} \
         chars={invented_chars}"
    );
    for (i, example) in invented_examples.iter().enumerate() {
        println!("transcribe invented-example {} text=\"{example}\"", i + 1);
    }
    // One entry per label, `;`-separated because several AudioSet names carry a comma
    // of their own and a space is the row's own separator.
    let labels: Vec<String> = noise_labels
        .iter()
        .map(|(label, count)| format!("{}:{count}", label.replace(' ', "_")))
        .collect();
    // The rule that decides whether a merged span is offered to the recogniser at all:
    // the tagger's own names that can carry words, printed so a suppression can be read
    // against the list that made it rather than guessed at.
    println!(
        "transcribe label-rule offered={} labels={}",
        crate::transcribe::WORD_LABELS.len(),
        crate::transcribe::WORD_LABELS.join(";"),
    );
    println!(
        "transcribe noise-labels files={noise_files} labels={}",
        if labels.is_empty() {
            "-".to_string()
        } else {
            labels.join(";")
        }
    );

    // ── (g) one multi-minute track, whole ───────────────────────────
    let (read_ms, long_s, long_regions, long_build_ms, long_recognise_ms, long_wall_s) = match long
    {
        Some(entry) => entry,
        None => cell_failed!("cell transcribe — `{LONG_TRACK}` is not in the pinned corpus"),
    };
    println!(
        "transcribe long file={LONG_TRACK} audio_s={long_s:.1} regions={long_regions} \
         read_ms={read_ms:.1} build_ms={long_build_ms:.1} passes_s={:.1} wall_s={long_wall_s:.1} \
         s_per_audio_s={:.4} peak={}",
        long_recognise_ms / 1000.0,
        long_wall_s / long_s,
        report::human(report::peak_rss()),
    );

    // The unit row the table has for this cell. Its `cold` is the whole cell — the
    // messages, the corpus pass, the music pool, the constructed material, the pinned
    // clip's parity and the longest track — and its `run_ms` is the corpus pass alone:
    // one call per file, hand-over to text, which is also what the `e2e` row beside it
    // reports per second of audio. `map` is the three model files the cell binds, which
    // is what it holds beyond the arenas its calls plan; there is no single graph here,
    // so `arena` and `digest` are `-`. The rest of the row is the field set every other
    // unit prints, taken the same way, so this row is read beside the others rather than
    // as a shape of its own.
    let cell_ms = started.ms();
    let map_bytes = file_bytes(&pair.body.path)
        + file_bytes(&pair.sound.path)
        + file_bytes(&dir.join(crate::ced::CED_MODEL));
    let peak = peak_rss();
    let counts = graphs::device_counts();
    println!(
        "unit transcribe build_ms={:.3} run_ms={corpus_wall_ms:.3} peak={peak} cold_ms={cell_ms:.3} \
         map={map_bytes} beyond_map={} file_over_rss={} arena=- submits={} waits={} at={} load={} \
         digest=- speech_s={corpus_speech_s:.6}",
        costs.build_ms,
        peak.saturating_sub(map_bytes),
        map_bytes.saturating_sub(peak),
        counts.0 - counts_at_start.0,
        counts.1 - counts_at_start.1,
        now(),
        load_average(),
    );
}

/// One model file the transcription cell binds, printed as the record's artefact
/// line: its bare name, its own bytes and hash, the pin when the tree holds one,
/// and the source and revision.
fn transcribe_artefact(path: &std::path::Path, pinned: Option<&str>, source: &str) {
    let (bytes, sha256) = match crate::measurements::digest(path) {
        Some(digest) => digest,
        None => cell_failed!("cell transcribe — {} cannot be read", path.display()),
    };
    println!(
        "transcribe artefact name={} bytes={bytes} sha256={sha256} pinned={} source={source}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("-"),
        pinned.unwrap_or("-"),
    );
}

/// The models a transcription call needs, loaded the way a product loads them on its
/// first request: the recognition pair and the tagger, with no weights on the device
/// and no compiled recognition graph.
fn load_transcriber(
    body: &std::path::Path,
    sound: &std::path::Path,
    ced: &std::path::Path,
) -> anyhow::Result<crate::transcribe::Transcriber> {
    let pair = std::sync::Arc::new(loader::Pair::open(body, sound)?);
    let tagger = crate::ced::Tagger::open(ced)?;
    crate::transcribe::Transcriber::open(pair, tagger)
}

/// A WAV file read whole, or the process ends naming it: the cell cannot report a
/// number for audio it does not have.
fn read_whole(path: &std::path::Path, name: &str) -> Vec<f32> {
    match host::wav_mono_16k(&path.display().to_string(), 0, usize::MAX) {
        Ok(audio) => audio,
        Err(error) => cell_failed!("cell transcribe — {name}: {error:#}"),
    }
}

/// A fallible step of the transcription cell, or the process ends naming what
/// failed: every step here is a `cell_failed!`, not a printed skip.
fn or_exit<T>(what: &str, result: anyhow::Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => cell_failed!("cell transcribe — {what}: {error:#}"),
    }
}

/// The first 60 characters of `text` on one line, so a quote or a newline in a
/// transcript cannot break the `key=value` line it is printed in.
fn prefix(text: &str) -> String {
    one_line(text).chars().take(60).collect()
}

/// The pairwise word difference over the shared prefix — the rule
/// [`crate::reference`]'s own comparison uses — as `(moved, produced, pinned)`.
fn word_difference(produced: &str, pinned: &str) -> (usize, usize, usize) {
    let got: Vec<&str> = produced.split_whitespace().collect();
    let want: Vec<&str> = pinned.split_whitespace().collect();
    let moved = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    (moved, got.len(), want.len())
}

/// One transcription's transcripts, in region order, joined by a space — the
/// caller's own reading of the result. The call returns regions, not a string: a
/// labelled region carries no text and is dropped here, and a caller that needs the
/// labels or the times reads them off the regions. The stand's transcription cell and
/// the music-pool test read the same result, so they share this rather than each
/// writing the join.
pub fn spoken(out: &crate::transcribe::Transcription) -> String {
    out.regions
        .iter()
        .filter_map(|region| region.text.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The labels one transcription's regions carry, deduplicated in region order and
/// joined with `+` (a space inside a label becomes `_`, so the row stays one
/// token).
fn labels_of(out: &crate::transcribe::Transcription) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for label in out
        .regions
        .iter()
        .filter_map(|region| region.label.as_deref())
    {
        if !seen.contains(&label) {
            seen.push(label);
        }
    }
    seen.iter()
        .map(|label| label.replace(' ', "_"))
        .collect::<Vec<_>>()
        .join("+")
}

/// What the tagger says the audio is, over one transcription's textless regions:
/// one entry per label with its region count, `;`-separated because several AudioSet
/// names carry a comma of their own and a space is the row's own separator.
fn textless_labels(out: &crate::transcribe::Transcription) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for label in out
        .regions
        .iter()
        .filter(|region| region.text.is_none())
        .filter_map(|region| region.label.as_deref())
    {
        *counts.entry(label).or_default() += 1;
    }
    if counts.is_empty() {
        return "-".to_string();
    }
    counts
        .iter()
        .map(|(label, count)| format!("{}:{count}", label.replace(' ', "_")))
        .collect::<Vec<_>>()
        .join(";")
}

/// The adjacent pairs of text regions where the later one begins with the words the
/// earlier one ended with: one region carrying a word twice at a seam, which
/// [`crate::transcribe::overlap_words`] reads. Only regions adjacent in the plan are
/// compared, so a labelled region between two transcripts is not mistaken for a seam, and
/// only an overlap of [`crate::transcribe::SEAM_WORDS`] words or more is counted, because
/// two words can coincide by chance.
fn seam_pairs(out: &crate::transcribe::Transcription) -> usize {
    out.regions
        .windows(2)
        .filter(
            |pair| match (pair[0].text.as_deref(), pair[1].text.as_deref()) {
                (Some(previous), Some(next)) => {
                    crate::transcribe::overlap_words(previous, next)
                        >= crate::transcribe::SEAM_WORDS
                }
                _ => false,
            },
        )
        .count()
}

/// The invariant every transcription of this cell is held to, ending the process
/// when it does not hold: the result's own contract ([`crate::transcribe::
/// Transcription::check`] — a region answers with a text or a label, a labelled
/// region never carries a `reliable` transcript, the regions are ordered without a
/// hole or an overlap inside `[0, samples]`, the first starting at 0 and the last
/// ending at `samples`), named by the input it failed on.
fn check_regions(out: &crate::transcribe::Transcription, name: &str) {
    if let Err(error) = out.check() {
        cell_failed!("cell transcribe — {name}: {error:#}");
    }
}

/// What the transcription rows measured, under the table: what the times cost, what
/// the `speech` row's figures are taken over, what the guards were set to, where the
/// constructed material's truth comes from, and how the table's own row is read.
/// Every figure it quotes is named by the row that carries it, because these notes
/// are re-emitted on every refresh and must not contradict the table above them.
const TRANSCRIBE_NOTES: &str =
    "\n**The headline is `e2e`.** One figure is the recognition number, and it is the whole
call from hand-over of the audio to the returned text: reading the file, the gain and
resample, the detector's mask, the tagger's labels, the cut, every compilation the call
needs, the recognition passes, the guards and the joins. `ms_per_audio_s` is that figure
over the pass; `corpus-cold` is the same call with the models loaded inside the
measurement — no weights on the device when it starts — on the first file of the pass,
and `message` gives it absolutely for a real ten-second and a real thirty-second voice
message, `cold` (loaded by the call) and `warm` (the same call on the loaded
transcriber), each printing the text it returned so the result is checkable rather than
asserted. Nothing else in this section is a recognition figure: `breakdown` is the same
call split into what could be attributed — the file read, the mask and plan, the tagger,
the compilations and the recognition passes, per minute of audio — plus
`unattributed_ms_per_audio_min`, which is the remainder (gain and resample, the guards,
the joins, the retimed result) reported rather than guessed at.

`breakdown`'s `compile` is the price of a graph per chunk length: each compile uploads
the body's weights into the shared weight buffer, and a compile for a model whose
weights a previous compile already placed in that buffer skips the upload — the
`set_param_sharing_scope` path this tree declares in `graphs::Runner::compile`. What
remains of a compile is its own lowering plus the host-side read of the weight bytes,
which is why `compile` does not fall to zero. `chunks`, `shapes` and `windows` are the
work it prices: the recognition passes, the distinct frame counts their graphs were
compiled at, and the tagger's windows. The tagger's own compile is `open_ms`, and the
music pool, the constructed material and the parity rows are priced by their own rows.

The `tuning` row is the decision surface this run was taken with — the pause it merges
across, the run it drops, the hangover it reaches into silence, the length every cut aims
at, the caps on how far before and after that aim a cut may search and the share of the
piece it may leave (`cut_slack_pct`: the narrower of the two bounds is the one that
applies, so a piece is between half a share and one and a half), the frame alignment its
graphs are compiled at, and the three text guards — because the product wants to filter
more rather than less and a choice that is not printed cannot be reviewed. A chunk is read
once: the plan cuts at the span the
encoder was trained over, so there is no second presentation to price.

The `speech` row answers how much of the speech the segmented path failed to
transcribe, in audio rather than in words. Its denominator is the speech the passes
covered, not the seconds the detector marked: a chunk's hangover reaches into silence,
so the covered figure is larger than `masked_s` in the coverage row, and both are named
so neither has to be derived from the other. `textless_s` is the covered audio whose
region came back with no transcript, `textless_chunks` the passes that produced it, and
`flagged_s` the speech whose transcript the guards rejected — returned, marked and
labelled, so a caller that filters loses those seconds and a caller that does not still
reads what was produced. A region's whole seconds are counted, the hangover into silence
included. `textless_voiced_s` and `textless_other_s` split the textless seconds by what
the tagger says the audio there is: the first a label that can hold words, the second a
detector mark on audio the tagger calls something else. `suppressed_s` is the third kind:
the marked seconds inside merged spans the tagger calls something that cannot carry a
word, whose audio is their label and which are not presented to the recogniser at all.
Every file that lost any speech has its own `textless-file` row — its `textless_s`, how
that splits by label, what the labels are, and the call's own `e2e_ms` — so the
aggregate is attributed rather than taken on trust. The noise pool is counted apart,
because there the detector's marks are mostly false positives and a textless second is
the detector's, not lost speech: `noise_marked_s` is what the detector marked,
`noise_suppressed_s` the marked seconds the tagger's labels kept out of the recognition
path and `noise_speech_s` what the recogniser was given. A word-level comparison cannot
answer the question: a recogniser paraphrases across a cut, and the parity rows measure
exactly that. `seams` counts the adjacent transcripts that share `seam_words` whole
words or more — a reading carried twice, which is what a lead-in or a piece join could
once produce; the plan gives every chunk disjoint audio, so it is printed beside the
pool rather than held against the round.

`label-rule` is the list that decides whether a merged span is offered to the recogniser
at all: the tagger's own names that can carry words — the speech classes, the singing
ones and the whole music family — matched on the name the file carries. Music and singing
are offered even when no word comes out of them, because a song over an orchestra is
exactly what the song rule must transcribe when its words come out, and an instrumental
clip produces no text at all: offering it costs a pass and invents nothing.
`suppressed_s` is what the rule kept out of the recognition path — on the speech pool, on
the noise pool (`noise_suppressed_s`) and on the music pool, where the `music` and
`music-report` rows carry it so that a clip with no transcript and suppressed seconds was
never asked rather than misread — and `noise-labels` what the labelled regions say. The
tagger's own arithmetic, as this tree runs it: f16 weights read from the file and served
to f32 multiplies over their widened image — not f16 in compute, which the cooperative
f16 tile cannot do at this model's 252-token window without padding the token axis and
changing the arithmetic.

`parity` reads the pinned clip the listen cell reads — the same file at the same
offset and length, conditioned by the same `prepare_speech` — through the segmented
call, whose text describes the speech spans it detected (`transcribed_s` of the
clip's `clip_s`) and whose `moved` counts word positions against the pinned
transcript. `parity-span` hands the same detected span to the single-shot walk at
its own exact frame count: that is the segmented walk's own input, so a move there
is what the segmented path costs and a match is the span's — the segmented walk
reproduces the single-shot reading of the same audio when a chunk's frame count is
rounded to the encoder's stride and no further, so no padded frame becomes an extra
attention key. The `parity` row's remainder is the span's: the recogniser's own cut
sensitivity, which moves most word positions when only the second of non-speech
around the speech is added or removed — a reading difference, not lost speech. Both
`moved` figures compare word by position, so one inserted or missing word marks
every position after it.

`boundary` is the binary check on material whose truth is the construction: a speech
excerpt pasted into digital silence at known offsets. `err_ms` is each kept region's
start against the paste it came from, signed; a boundary that conditioning moved would
show there. The gain row repeats the same material as a copy at a level the library
leaves alone against the lifted one: `moved` counts boundaries that shifted, and
`identical` says the two readings are the same.

`music` and `music-report` are the music pool: instrumental clips must produce no
trusted transcript (a flagged guess is counted, `instrumental_any_text`), and a vocal
clip is transcribed where its words survive the guards. `music-total` is the pool's own
wall time per second of audio — the call, not the baseline of a phase.

The `unit` row is this cell's row in the table above. Its `cold` is the whole cell —
the messages, the corpus pass, the music pool, the constructed material, the pinned
clip's parity and the longest track — and its `run_ms` is the corpus pass alone, which is
what `s/speech s` re-derives from. `build` is the pass's compilations, `map` the model
files the cell binds, and `arena` and `digest` are `-` because the cell has no single
graph and no pinned answer of its own.
";
/// The transcription surface's section of the record, from the rows its cell
/// printed: the rows as one table, what they measured, the files the cell bound
/// with their measured hashes, and the music pool's provenance, so the CC BY 2.5
/// attribution travels with the numbers.
fn transcribe_section(rows: &[String], provenance: &str) -> anyhow::Result<String> {
    // Every figure the prose names must be there: a renamed or missing field is an
    // error, never a zero written into the record as if the run had measured it.
    let number = |key: &str, label: &str| -> anyhow::Result<f64> {
        let row = rows
            .iter()
            .find(|row| row.starts_with(key))
            .with_context(|| format!("this run printed no `{key}` row"))?;
        field(row, label).with_context(|| format!("the `{key}` row carries no `{label}`"))
    };
    let files = number("transcribe corpus ", "files")?;
    let audio_s = number("transcribe corpus ", "audio_s")?;
    let mut body = String::new();
    body.push_str("### Transcription: one call over the pinned corpus and the music pool\n\n");
    body.push_str(&format!("_Measured {}._\n\n", crate::measurements::now()));
    // The headline, from the rows themselves: the whole call per second of audio, and
    // what a ten-second and a thirty-second voice message cost, cold and warm.
    let e2e = number("transcribe e2e ", "ms_per_audio_s")?;
    let mut headline = String::new();
    for seconds in [10, 30] {
        for pass in ["cold", "warm"] {
            let key = format!("transcribe message s={seconds} pass={pass} ");
            // Cold is the call that loads the models and warm the same call on the
            // loaded ones, so the cold figure is the total and the warm one its call.
            let label = if pass == "cold" {
                "total_ms"
            } else {
                "call_ms"
            };
            let ms = number(&key, label)?;
            headline.push_str(&format!(
                "a {seconds}-second message takes {ms:.0} ms ({pass}); "
            ));
        }
    }
    body.push_str(&format!(
        "**The headline: {e2e:.0} ms of call per second of audio** over the corpus pass, and \
         {headline}the text each of those calls returned is in the `message` rows below.\n\n",
    ));
    // The bar, and the gap against it, computed from the rows rather than recalled: the
    // target is 100 ms of call per second of audio, 103 at the tolerance, and the two
    // largest remaining items are the rows of the breakdown beside them.
    let passes = number("transcribe breakdown ", "passes_ms_per_audio_min")? / 60.0;
    let compile = number("transcribe breakdown ", "compile_ms_per_audio_min")? / 60.0;
    let tagger = number("transcribe breakdown ", "tagger_ms_per_audio_min")? / 60.0;
    body.push_str(&format!(
        "The bar for that figure is {bar:.0} ms per second of audio, {tolerance:.0} at the \
         tolerance the plan allows: this run is **{gap:.0} ms/s above it** ({ratio:.2}× the bar). \
         Of the {e2e:.0} ms, the breakdown below attributes {passes:.0} to the recognition passes, \
         {compile:.0} to the compilations, {tagger:.1} to the tagger and the rest to the read, the \
         mask and plan and what it does not attribute — so the passes and the compilations are \
         what remains between this figure and the bar. Every other figure here is a breakdown for \
         the agents that optimise, and none of them is the recognition number.\n\n",
        bar = TRANSCRIBE_BAR_MS_PER_AUDIO_S,
        tolerance = TRANSCRIBE_BAR_TOLERANCE_MS_PER_AUDIO_S,
        gap = e2e - TRANSCRIBE_BAR_MS_PER_AUDIO_S,
        ratio = e2e / TRANSCRIBE_BAR_MS_PER_AUDIO_S,
    ));
    // Warm is the whole pass; cold is the one file of it that pays the model load and the
    // first compilations, because a pass has exactly one first file. Said plainly, so the
    // two figures cannot be read as the same measurement.
    let cold_file = rows
        .iter()
        .find(|row| row.starts_with("transcribe corpus-cold "))
        .context("this run printed no `transcribe corpus-cold` row")?;
    body.push_str(&format!(
        "Warm is the whole pass above; **cold** is `corpus-cold`, that same call with the models \
         loaded inside the measurement (load {:.0} ms + call {:.0} ms) on the pass's first file, \
         `{}` — the one file of a pass that pays the load and the first compilations, since the \
         rest of the pass reuses them.\n\n",
        field(cold_file, "load_ms").context("`corpus-cold` carries no `load_ms`")?,
        field(cold_file, "call_ms").context("`corpus-cold` carries no `call_ms`")?,
        cold_file
            .strip_prefix("transcribe corpus-cold file=")
            .and_then(|rest| rest.split_whitespace().next())
            .unwrap_or("-"),
    ));
    // The other headline figure, with its bar in the same form as the latency one.
    let textless_pct = number("transcribe speech ", "textless_pct")?;
    let textless_bar = number("transcribe speech ", "textless_bar_pct")?;
    body.push_str(&format!(
        "**The quality figure: {textless_pct:.2} % of the covered speech came back with no \
         transcript**, against a bar of {textless_bar:.0} %: this run is **{:.2} points above \
         it**. The `speech` row carries the seconds, the chunks and the split by the tagger's \
         label, and every file that lost any of them has its own `textless-file` row.\n\n",
        textless_pct - textless_bar,
    ));
    body.push_str(&format!(
        "These rows are one pass of `transcribe::Transcriber` over the pinned corpus of {files:.0} \
         real-audio files ({:.1} min of speech and noise) and the eight-clip music pool, plus the \
         constructed boundary material, the longest pinned track and the pinned clip's parity with \
         the single-shot answer; a corpus has no warm repetition, so each row is one measurement of \
         the audio it names. Every transcription is held to the invariant that a region answers \
         with a text or a label, that a labelled region carries no reliable transcript, and that \
         the regions tile the input exactly — the first at 0, the last at its length. The times \
         are the detector's own 16 ms hops and the labels are the tagger's per-window winners, not \
         a transcript.\n\n",
        audio_s / 60.0,
    ));
    body.push_str("| what | numbers |\n|---|---|\n");
    let mut artefacts = String::new();
    for row in rows {
        let rest = row.strip_prefix("transcribe ").unwrap_or(row);
        // An artefact row is a model file, not a measurement: it belongs in the
        // block below the table with the other provenance lines.
        if let Some(rest) = rest.strip_prefix("artefact ") {
            let value = |label: &str| -> &str {
                let key = format!("{label}=");
                rest.split_whitespace()
                    .find_map(|token| token.strip_prefix(&key))
                    .unwrap_or("-")
            };
            let (name, bytes, sha256) = (value("name"), value("bytes"), value("sha256"));
            artefacts.push_str(&format!(
                "- model: `{name}` · {bytes} bytes · sha256 {sha256}"
            ));
            if value("source") != "-" {
                artefacts.push_str(&format!(" · {}", value("source")));
            }
            // A file whose own bytes do not hash to the pin is not the artefact the
            // figures are attributed to; the record says so rather than repeating
            // the pin as if it had been read.
            if value("pinned") != "-" && value("pinned") != sha256 {
                artefacts.push_str(&format!(
                    " · NOT the artefact of record, whose sha256 is {}",
                    value("pinned")
                ));
            }
            artefacts.push('\n');
            continue;
        }
        // A key is the leading tokens with no `=` in them (`music <path>` is one
        // key); everything from the first `key=value` token on is its numbers. A
        // row that is one figure and nothing else is named by that figure's key.
        let mut key = String::new();
        let mut numbers = String::new();
        for token in rest.split_whitespace() {
            if numbers.is_empty() && !token.contains('=') {
                if !key.is_empty() {
                    key.push(' ');
                }
                key.push_str(token);
            } else {
                numbers.push_str(token);
                numbers.push(' ');
            }
        }
        if key.is_empty()
            && let Some((head, tail)) = numbers.trim_end().split_once('=')
        {
            key.push_str(head);
            numbers = tail.to_string();
        }
        // A `|` in a transcript would end the table cell it sits in.
        let key = key.replace('|', "\\|");
        body.push_str(&format!(
            "| {key} | {} |\n",
            numbers.trim_end().replace('|', "\\|")
        ));
    }
    body.push_str(TRANSCRIBE_NOTES);
    body.push_str("\nProvenance:\n\n");
    body.push_str(provenance);
    body.push_str(&artefacts);
    body.push_str(&format!(
        "\n> **The music pool's provenance.** {}\n",
        music::provenance()
    ));
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::{carried_from, caught, merge_stream_rows, stream_row, stream_row_prefix};
    use rlx_wgpu::bandwidth::{CeilingFailure, StreamCeiling};

    /// A block of the shape the record holds: cell rows, a weight-stream
    /// subsection whose note paragraph was written by a run that measured no
    /// generative cell, and the three kinds of artefact line.
    const BLOCK: &str = "\
<!-- BEGIN stand -->
## Per-model measurement (stand)

| unit | cold ms | warm ms | speech s | s/speech s | loop ms | sound ms | build ms | peak | map | arena | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | 1.0 | 2.0 | 3.0 | 4.0 | - | - | 5.0 | 0.1 GiB | 0.2 GiB | 1 MiB | aa |
| embed.8192 | 6.0 | 7.0 | - | - | - | - | 8.0 | 0.3 GiB | 0.4 GiB | 2 MiB | bb |
| no such cell | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |

### Weight streams, against the machine's streaming ceiling

This run measured no generative cell, so it took no stream reading of its own: the rows below are the run that measured them, restated against the ceiling printed below.

Ceiling, measured on this run's device before the cells: **112.9 GB/s** on a streaming read.

| phase | unit | weight bytes/unit | ms/unit | GB/s | of the read ceiling |
|---|---|---|---|---|---|
| recognition decode | one token | 1 100 961 792 | 22.66 | 48.6 | 43.0 % |
| talker body | one frame | 850 870 272 | 19.91 | 42.7 | 37.8 % |

Each row's numerator is the graph's own parameter bytes.

Provenance:

- machine: nowhere
- code: 1654bb26 change-sha256 aa (the commit)
- model: `a.gguf` · 1 bytes · sha256 aa
- voice: `v.wav` · 2 bytes · sha256 bb
- text: `pinned.txt` · 3 bytes · sha256 cc
<!-- END stand -->
";

    /// What the carry takes is rows and lines, never a paragraph: a paragraph it
    /// read back would be written again on every run that carries it, one copy per
    /// run.
    #[test]
    fn a_carried_block_takes_rows_and_lines_but_no_paragraph() {
        let carried = carried_from(BLOCK);
        assert_eq!(
            carried.rows.keys().copied().collect::<Vec<_>>(),
            ["ear", "embed.8192"],
            "the cell rows it names, and no header or stranger's row"
        );
        assert_eq!(
            carried
                .streams
                .iter()
                .map(|row| row.unit.as_str())
                .collect::<Vec<_>>(),
            ["one token", "one frame"],
            "one row per unit"
        );
        assert!(
            !carried
                .streams
                .iter()
                .any(|row| row.columns.contains("no generative cell")),
            "the note is a paragraph, not a row, and re-emitting it stacks it"
        );
        assert_eq!(
            carried
                .artefacts
                .iter()
                .filter_map(|line| super::artefact_name(line))
                .collect::<Vec<_>>(),
            ["a.gguf", "v.wav", "pinned.txt"],
            "the article's identity travels with the embed rows it belongs to"
        );
        assert!(
            !carried
                .artefacts
                .iter()
                .any(|line| line.starts_with("- machine:") || line.starts_with("- code:")),
            "only artefact lines are carried, not the rest of Provenance"
        );
    }

    /// A carried row keeps its own `ms` and `GB/s` — they are the run that took
    /// it — and has only its percentage re-taken, against the ceiling printed
    /// above it: one table, one denominator.
    #[test]
    fn a_carried_row_is_restated_against_this_runs_ceiling() {
        let line = "| recognition decode | one token | 1 100 961 792 | 22.66 | 48.6 | 43.0 % |";
        let five = stream_row_prefix(line).expect("a six-column row");
        assert_eq!(
            stream_row(&five, Some(112.9)),
            "| recognition decode | one token | 1 100 961 792 | 22.66 | 48.6 | 43.0 % |\n",
            "the row's own ceiling reproduces its own share"
        );
        assert_eq!(
            stream_row(&five, Some(97.2)),
            format!("{five} 50.0 % |\n"),
            "another ceiling restates the share, and only the share"
        );
        assert_eq!(
            stream_row(&five, None),
            format!("{five} - |\n"),
            "a probe that produced no figure invents no denominator"
        );
    }

    /// Measuring one generative cell replaces its row and keeps the rest: the
    /// other units' rows are what the record holds, and this run cannot re-take
    /// them.
    #[test]
    fn one_measured_unit_keeps_the_other_stream_rows() {
        let row = |unit: &str, columns: &str| super::StreamRow {
            unit: unit.to_string(),
            columns: columns.to_string(),
        };
        let table = || {
            merge_stream_rows(
                vec![
                    row(
                        "one token",
                        "| recognition decode | one token | 1 | 2.00 | 3.0 |",
                    ),
                    row("one frame", "| talker body | one frame | 4 | 5.00 | 6.0 |"),
                    row(
                        "one level",
                        "| code predictor | one level | 7 | 8.00 | 9.0 |",
                    ),
                ],
                vec![row(
                    "one frame",
                    "| talker body | one frame | 10 | 11.00 | 12.0 |",
                )],
            )
        };
        let rows = table();
        assert_eq!(
            rows.iter().map(|row| row.unit.as_str()).collect::<Vec<_>>(),
            ["one token", "one frame", "one level"],
            "order and the unmeasured units survive"
        );
        assert_eq!(
            rows[1].columns,
            "| talker body | one frame | 10 | 11.00 | 12.0 |"
        );
        assert_eq!(
            merge_stream_rows(table(), table())
                .iter()
                .map(|row| row.columns.as_str())
                .collect::<Vec<_>>(),
            rows.iter()
                .map(|row| row.columns.as_str())
                .collect::<Vec<_>>(),
            "carrying a table over itself changes nothing"
        );
    }

    /// The record keeps three readings apart — no adapter, failed device work, a
    /// panic — and a diagnostic must never fold the last two into the first. This
    /// exercises the mapping both probe sites share.
    #[test]
    fn a_probe_failure_is_three_distinguishable_readings() {
        fn failure(reading: &Result<StreamCeiling, CeilingFailure>) -> &CeilingFailure {
            match reading {
                Ok(_) => panic!("no probe produced a figure"),
                Err(failure) => failure,
            }
        }
        let readings = [
            caught(|| Err(CeilingFailure::NoAdapter)),
            caught(|| Err(CeilingFailure::DeviceWork("the wait poll failed".into()))),
            caught(|| -> Result<StreamCeiling, CeilingFailure> { panic!("probe exploded") }),
        ];
        let kinds: Vec<_> = readings.iter().map(|r| failure(r).kind()).collect();
        assert_eq!(kinds, ["no_adapter", "device_work", "panicked"]);
        let reasons: Vec<_> = readings
            .iter()
            .map(|r| failure(r).reason().map(str::to_string))
            .collect();
        assert_eq!(
            reasons,
            [
                None,
                Some("the wait poll failed".to_string()),
                Some("probe exploded".to_string())
            ],
            "a device fault and a panic must each say what happened"
        );
        for (reading, kind) in readings.iter().zip(kinds) {
            let line = failure(reading).to_string();
            assert!(
                !line.contains("no wgpu adapter") || kind == "no_adapter",
                "{kind} must not read as a missing adapter: {line}"
            );
        }
    }

    /// The rows the section's prose reads: headline, the four message arms, the
    /// breakdown, the speech pool and the pass's first-file cold reading. A section
    /// built without one of them is an error, not a zero.
    fn prose_rows() -> Vec<String> {
        vec![
            "transcribe corpus files=33 audio_s=3838.8 speech_s=3300.1".to_string(),
            "transcribe e2e ms_per_audio_s=167.2".to_string(),
            "transcribe message s=10 pass=cold total_ms=3093.3 words=17".to_string(),
            "transcribe message s=10 pass=warm call_ms=2368.2 words=17".to_string(),
            "transcribe message s=30 pass=cold total_ms=7195.3 words=49".to_string(),
            "transcribe message s=30 pass=warm call_ms=6539.8 words=49".to_string(),
            "transcribe breakdown read_ms_per_audio_min=3.2 passes_ms_per_audio_min=5500.3 \
             compile_ms_per_audio_min=4209.6 tagger_ms_per_audio_min=225.9"
                .to_string(),
            "transcribe speech files=18 speech_s=3174.8 textless_s=90.7 textless_pct=2.86 \
             textless_bar_pct=2.00 textless_gap_pct=0.86"
                .to_string(),
            "transcribe corpus-cold file=speech/librivox/speech-librivox-0027.wav load_ms=401.2 \
             call_ms=61181.5"
                .to_string(),
        ]
    }

    /// The transcription section's table is made from the cell's own rows, and a
    /// row's key is every leading token without a `=`: a music row's path and an
    /// invented example's index are part of what the row is about, while a quoted
    /// transcript's words are part of its numbers.
    #[test]
    fn a_transcription_row_splits_into_its_key_and_its_numbers() {
        let mut rows = prose_rows();
        rows.extend([
            "transcribe corpus files=33 audio_s=3838.8 speech_s=3300.1".to_string(),
            "transcribe music instrumental/01-orchestra.wav audio_s=40.00 regions=1".to_string(),
            "transcribe cost times_ms_per_audio_min=16.3 chunks=435".to_string(),
            "transcribe invented-example 1 text=\"la la la and then it kept going\"".to_string(),
            "transcribe invented-example 2 text=\"a | b\"".to_string(),
        ]);
        let body = super::transcribe_section(&rows, "- machine: test\n").expect("a full section");
        for expected in [
            "| corpus | files=33 audio_s=3838.8 speech_s=3300.1 |",
            "| music instrumental/01-orchestra.wav | audio_s=40.00 regions=1 |",
            "| cost | times_ms_per_audio_min=16.3 chunks=435 |",
            "| invented-example 1 | text=\"la la la and then it kept going\" |",
            "| invented-example 2 | text=\"a \\| b\" |",
            "- machine: test",
        ] {
            assert!(
                body.contains(expected),
                "the table must hold `{expected}`:\n{body}"
            );
        }
        assert!(
            body.contains("33 real-audio files (64.0 min"),
            "the prose is generated from the corpus row's own figures:\n{body}"
        );
        assert!(
            body.contains("creativecommons.org/licenses/by/2.5/"),
            "the music pool's CC BY 2.5 attribution travels with the numbers"
        );
    }

    /// A model file's row is a provenance line, not a table cell, and a file whose
    /// own hash differs from its pin says so instead of repeating the pin.
    #[test]
    fn a_model_row_becomes_a_provenance_line() {
        let mut rows = prose_rows();
        rows.extend([
            "transcribe artefact name=ced-small-f16.gguf bytes=43000000 \
             sha256=c391ed8697a1b0 pinned=c391ed8697a1b0 source=mudler/ced-gguf@b5e9a4aad643"
                .to_string(),
            "transcribe artefact name=Qwen3-ASR-1.7B-Q4_K_M.gguf bytes=1282435552 \
             sha256=other pinned=- source=-"
                .to_string(),
        ]);
        let body = super::transcribe_section(&rows, "").expect("a full section");
        assert!(
            body.contains(
                "- model: `ced-small-f16.gguf` · 43000000 bytes · sha256 c391ed8697a1b0 \
                 · mudler/ced-gguf@b5e9a4aad643"
            ),
            "the pinned file's line carries its source: {body}"
        );
        assert!(
            !body.contains("NOT the artefact of record"),
            "a file that hashes to its pin is the artefact of record: {body}"
        );
        assert!(
            body.contains("- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf`"),
            "a file with no pin still gets its line: {body}"
        );
    }

    /// The table's `speech s` column is the unit line's own figure, taken from there
    /// and nowhere else.
    #[test]
    fn the_speech_denominator_comes_from_the_unit_line() {
        let text = "cell listen frames=3000 speech_s=17.460000 steps=1 kinds=matmul:1\n\
                    unit listen build_ms=1.000 run_ms=2.000 peak=3 cold_ms=4.000 map=5 \
                    beyond_map=0 file_over_rss=0 arena=6 submits=7 waits=8 at=09:00:00 load=0.1 \
                    speech_s=17.460000\n";
        let line = super::unit_line(text, "listen").expect("the unit line");
        assert_eq!(super::field(line, "speech_s"), Some(17.46));
    }

    /// A figure the prose names and the run did not print is an error naming what is
    /// missing: the lookup used to answer `0.0`, which would write "0 ms of call per
    /// second of audio" into the record as if the run had measured it.
    #[test]
    fn a_missing_figure_fails_the_section_instead_of_printing_zero() {
        let mut rows = prose_rows();
        rows.retain(|row| !row.starts_with("transcribe e2e "));
        let error = super::transcribe_section(&rows, "").expect_err("no headline figure");
        assert!(
            format!("{error:#}").contains("no `transcribe e2e ` row"),
            "a missing row is named: {error:#}"
        );

        let mut rows = prose_rows();
        for row in rows.iter_mut() {
            if row.starts_with("transcribe speech ") {
                *row = "transcribe speech files=18 speech_s=3174.8 textless_pct=2.86".to_string();
            }
        }
        let error = super::transcribe_section(&rows, "").expect_err("no bar beside the figure");
        assert!(
            format!("{error:#}").contains("carries no `textless_bar_pct`"),
            "a missing field is named: {error:#}"
        );
    }

    /// And the checked-in record has it, for every row that has a denominator: the
    /// parse once read a line that does not carry the figure, and `listen` and
    /// `speak` came back blank without anything saying so.
    #[test]
    fn the_record_reports_a_denominator_for_every_spoken_row() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("MEASUREMENTS.md");
        let Ok(text) = std::fs::read_to_string(&root) else {
            return;
        };
        for name in ["listen", "speak", "transcribe"] {
            let Some(row) = text
                .lines()
                .find(|line| line.starts_with(&format!("| {name} |")))
            else {
                continue;
            };
            let seconds = row.split('|').nth(4).unwrap_or_default().trim();
            assert!(
                seconds.parse::<f64>().is_ok(),
                "{name}'s speech denominator in {}: {row}",
                root.display()
            );
        }
    }
}
