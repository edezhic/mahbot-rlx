// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stand: the live model paths from the compressed files to their answer, with
//! the measurement taken on the way.
//!
//! One entry point and no flags. The acceptance walks four units, each in its own
//! process: the ear's 12 160-sample wake window (mel with the last frame dropped,
//! 76 frames, the 1.7B 24-block encoder, the projection, mean over ten tokens, L2,
//! cosine); 30 s of listening (3 000 mel frames, the 390-row prompt, at most
//! 2 048 new tokens on a 2 438-row fixed cache); 300 text tokens → 330 codec rows
//! (26.4 s) through the talker, predictor and sound half; and a tokenizer-fed
//! embedding cell over the whole token table. Every other cell stays behind its
//! number in `cells`.
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
//! windows and stitch (`synth::windows`, `synth::codec_slice`,
//! `synth::stitch_window`). What stays here is the measurement around them:
//! the generation and decode loops and the per-window `Outcome`s the bench does
//! not report.
//!
//! A unit builds and compiles each graph by phase, immediately before that phase's
//! runs; there is no separate "build everything first" step (`graphs::Compiled`).
//! `build_ms` is the sum of those builds, `cold_ms` takes the first iteration of
//! each phase cold and `run_ms` sums the warm medians. The parent collects the
//! four units, runs each in its own process three times, and prints four unit rows
//! (cold/warm time, peak, map and their spreads) plus one `build_ms` line whose
//! value is the median of the three per-run sums of the four units. Everything
//! goes to stdout; the child's `cell`/`phase`/`unit` lines are forwarded with
//! their `at=`/`load=`.
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

use crate::graphs::{BodyInput, Outcome, SampleSpec};
use crate::guard::{
    DEFAULT_MAX_RSS_MIB, DEFAULT_TIMEOUT_SECS, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock,
    Watch,
};
use crate::listen::{
    ASR_HEAD_IDS, ASR_MAX_NEW, ASR_PROMPT_ROWS, ASR_TAIL_IDS, ASR_TEXT_IDS, Front, argmax,
    is_asr_eos, l2, mean_pool,
};
use crate::loader::{Model, Pair, Role};
use crate::report::{
    digest, fmt_ms, human, keys, load_average, max_abs, non_finite, peak_rss, pseudo_rand,
    resident_rss, sci,
};
use crate::synth::{
    DAC_MARGIN, DAC_WINDOW, FRONT_WINDOW, MAX_CODEC_ROWS as SPEAK_CODEC_ROWS,
    MAX_TEXT_TOKENS as SPEAK_TEXT_TOKENS, TTS_CODEC_EOS, TTS_REPEAT_PENALTY, TTS_TEXT_ROWS,
    TalkerPrompt, apply_repetition, codec_slice, stitch_window, suppress_service, talker_prompt,
    windows,
};
use crate::tokenizer::Tokenizer;
use crate::{graphs, host, loader, report};

/// The ear's product input: 12 160 samples at 16 kHz, the mel front drops the
/// last frame, leaving 76 frames → 10 encoder tokens.
const EAR_FRAMES: usize = 76;
const EAR_SAMPLES: usize = 12_160;
/// The recognition product segment: 30 s at 16 kHz = 480 000 samples; the mel
/// front drops the last frame, leaving 3 000 frames. A shorter recording is
/// trimmed to its speech and measured on that fixed frame count instead.
const SPEECH_SAMPLES: usize = 480_000;
/// The two embedding cells: an ordinary request/ticket, and the model limit.
const EMBED_CELLS: [usize; 2] = [256, 8_192];
/// The mel front end's own window.
const MEL_FRAMES: usize = 72;
/// The biases are read on a second of speech.
const BIAS_FRAMES: usize = 100;

/// The synthesis text condition, tokenized by the file's own tokenizer; it is
/// long enough to fill all 300 tokens.
const TTS_TEXT: &str = "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off, then I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.";
/// A normal embedding request: the same tokenizer, filled into the 256/8192
/// cells with the model's padding id.
const EMBED_TEXT: &str = "Please summarize this support ticket. The customer cannot sign in after the password reset, and the verification email never arrives. Check the account state, resend the message, and report the result in one short answer.";
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
    Embed,
    Mel,
    Bias,
    Head,
}

fn cells() -> Vec<Cell> {
    vec![
        Cell::Ear,
        Cell::Ear075,
        Cell::Ear2,
        Cell::Ear4,
        Cell::Listen,
        Cell::Speak,
        Cell::Embed,
        Cell::Mel,
        Cell::Bias,
        Cell::Head,
    ]
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

/// The guards one heavy run carries: the shared lock (in the parent and in a
/// cell started on its own), the wall-clock stop and the resident-set ceiling.
/// The watch is the same [`Watch`] the bench installs, so a blocked device call
/// cannot outlive either limit on any entry point.
struct StandGuard {
    /// Dropped with the guard: the thread wakes within a second and exits.
    _watch: Watch,
    _lock: Option<RunLock>,
}

impl StandGuard {
    /// The parent, and a cell started on its own: the lock plus both limits.
    fn install() -> Self {
        let (lock, deadline, ceiling) = heavy_run();
        Self::watched(Some(lock), deadline, ceiling)
    }

    /// A cell the parent spawned. The parent holds the lock and kills the cell at
    /// its own stop, but the cell carries both limits itself as well: a parent
    /// that dies abnormally then leaves a cell that still ends at its ceiling or
    /// its stop, instead of a multi-gigabyte process with nothing over it.
    fn child() -> Self {
        let deadline = Deadline::new(limit("MAHBOT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS));
        let ceiling = MemoryCeiling::new(limit("MAHBOT_MAX_RSS_MIB", DEFAULT_MAX_RSS_MIB));
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
    let ceiling = MemoryCeiling::new(limit("MAHBOT_MAX_RSS_MIB", DEFAULT_MAX_RSS_MIB));
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
    let mut args = std::env::args().skip(1);
    if let Some(first) = args.next() {
        let Ok(i) = first.parse::<usize>() else {
            eprintln!(
                "stand: `{first}` is not a cell index — the stand takes one cell index and no \
                 flags"
            );
            std::process::exit(2);
        };
        if let Some(extra) = args.next() {
            eprintln!(
                "stand: unexpected argument `{extra}` — the stand takes one cell index and no flags"
            );
            std::process::exit(2);
        }
        let all = cells();
        match all.get(i) {
            Some(cell) => {
                // A cell run on its own is a heavy run: it carries the shared
                // lock, the stop and the ceiling. Spawned by the parent, the
                // parent holds the lock and the cell carries both limits, so a
                // parent that dies abnormally leaves a bounded cell behind.
                let _guard = if std::env::var_os(CELL_CHILD).is_some() {
                    StandGuard::child()
                } else {
                    StandGuard::install()
                };
                run(*cell, i);
                return;
            }
            None => {
                println!("cell {i} — there are {} cells", all.len());
                std::process::exit(2);
            }
        }
    }

    // The stand is the crate's other multi-gigabyte, tens-of-minutes run, so it
    // carries the bench's three rules: one instance at a time, a wall-clock stop
    // and a resident-set ceiling. The parent holds the lock and watches the run's
    // clock and each cell's resident set; the cell carries the stop and the
    // ceiling itself as well, so it stays bounded even if the parent dies.
    let (lock, deadline, ceiling) = heavy_run();
    eprintln!(
        "stand: stop {}s · ceiling {} MiB · lock {}",
        deadline.secs(),
        ceiling.rss_bytes() / (1024 * 1024),
        stand_lock().display()
    );
    let _lock = lock;

    // The acceptance units, each in its own process: the four paths whose time
    // and peak the run reports. Every other cell stays behind its number. Each
    // unit is run `RUNS` times so the report carries a real spread; the child's
    // own `cell`/`phase`/`unit` lines are forwarded with their `at=`/`load=`.
    const RUNS: usize = 3;
    let all = cells();
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            eprintln!("stand: cannot find this binary: {error}");
            std::process::exit(1);
        }
    };
    let mut rows: Vec<(&'static str, Vec<(f64, f64, u64, u64, f64)>, String, String)> = Vec::new();
    for (i, cell) in all.iter().enumerate() {
        let Some(name) = unit_name(cell) else {
            continue;
        };
        let mut got = Vec::with_capacity(RUNS);
        let mut arena = String::new();
        let mut digest = String::new();
        for _ in 0..RUNS {
            let out = run_cell(&exe, i, deadline, ceiling);
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(value) = text
                .split("arena=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
            {
                arena = value.to_string();
            }
            if let Some(value) = text
                .split("digest=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
            {
                digest = value.to_string();
            }
            // Keep the child's numbers in the normal run: dropping them is what
            // used to hide the window and the load each number was taken at.
            print!("{text}");
            match out
                .status
                .success()
                .then(|| parse_unit(&text, name))
                .flatten()
            {
                Some(v) => got.push(v),
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
        rows.push((name, got, arena, digest));
    }
    let mut build_runs = vec![0.0f64; RUNS];
    for (_, got, _, _) in &rows {
        for (run, v) in got.iter().enumerate() {
            if run < build_runs.len() {
                build_runs[run] += v.4;
            }
        }
    }
    println!("# eight numbers: cold/warm run_ms, peak (GiB), map (GiB), spreads");
    let mut table = String::new();
    table.push_str("| unit | cold ms | warm ms | build ms | peak | map | arena | digest |\n|---|---|---|---|---|---|---|---|\n");
    for (name, got, arena, digest) in &rows {
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
        println!(
            "{name} cold={cold_ms:.1} warm={warm_ms:.1} peak={peak_gib:.2} map={map:.2} cold_spread={:.1} warm_spread={:.1} peak_spread={:.3} at={} load={}",
            spreads(&cold),
            spreads(&warm),
            spreads(&peak),
            now(),
            load_average(),
        );
        let build_ms = median(&got.iter().map(|v| v.4).collect::<Vec<_>>()).unwrap_or(0.0);
        table.push_str(&format!(
            "| {name} | {cold_ms:.1} | {warm_ms:.1} | {build_ms:.1} | {peak_gib:.2} GiB | {map:.2} GiB | {arena} | {digest} |\n"
        ));
    }
    let build_spread = build_runs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - build_runs.iter().copied().fold(f64::INFINITY, f64::min);
    let build_ms = median(&build_runs).unwrap_or(0.0);
    println!(
        "build_ms {build_ms:.1} build_spread={build_spread:.1} at={} load={}",
        now(),
        load_average()
    );

    // The repository's own record, refreshed by the run that produced it.
    let mut body = String::new();
    body.push_str("## Per-model measurement (stand)\n\n");
    body.push_str(&format!("_Measured {}._\n\n", crate::measurements::now()));
    body.push_str(&table);
    body.push_str("\nProvenance:\n\n");
    body.push_str(&format!("- machine: {}\n", crate::measurements::machine()));
    body.push_str(&format!("- device: {}\n", crate::measurements::device()));
    body.push_str(&format!("- profile: {}\n", crate::measurements::profile()));
    body.push_str(&format!("- code: {}\n", crate::measurements::code_state()));
    body.push_str("- command: `cargo run --release -p mahbot-inference` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`)\n");
    if let Ok(dir) = crate::loader::models_dir() {
        for (_, body_name, sound_name) in crate::loader::PAIRS {
            for name in [*body_name, *sound_name] {
                if let Some((bytes, sha)) = crate::measurements::digest(&dir.join(name)) {
                    body.push_str(&format!(
                        "- model: `{name}` · {bytes} bytes · sha256 {sha}\n"
                    ));
                }
            }
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
    }
    if let Err(error) = crate::measurements::refresh("stand", &body) {
        // The numbers are the deliverable, and the file is where they live: a
        // run that cannot record them is a failed run, as the bench's is.
        eprintln!("stand: cannot refresh MEASUREMENTS.md: {error:#}");
        std::process::exit(1);
    }
}

/// The four units of the acceptance, in report order.
fn unit_name(cell: &Cell) -> Option<&'static str> {
    match cell {
        Cell::Ear => Some("ear"),
        Cell::Ear075 => Some("ear.075"),
        Cell::Ear2 => Some("ear.batch2"),
        Cell::Ear4 => Some("ear.batch4"),
        Cell::Listen => Some("listen"),
        Cell::Speak => Some("speak"),
        Cell::Embed => Some("embed"),
        _ => None,
    }
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
        Cell::Embed => format!("embed cells={EMBED_CELLS:?}"),
        Cell::Mel => "mel frames=72".into(),
        Cell::Bias => format!("bias frames={BIAS_FRAMES}"),
        Cell::Head => "head rows=1".into(),
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

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(s[s.len() / 2])
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
        human(resident_rss()),
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
    // unit's whole-cell counters and the residency read before any drop.
    let beyond_map = peak.saturating_sub(map_bytes);
    let file_over_rss = map_bytes.saturating_sub(peak);
    println!(
        "unit {cell} build_ms={:.3} run_ms={:.3} peak={} cold_ms={:.3} map={} beyond_map={} file_over_rss={} arena={} submits={} waits={} at={} load={}",
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
        Cell::Embed => embed(),
        Cell::Mel => mel(),
        Cell::Bias => bias(),
        Cell::Head => head(),
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
    // first pass. It is dropped before the unit's memory read, so the peak is
    // the measured path alone.
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
    // The target graphs are the calibration, not the measured cell: drop them
    // before the memory read so the ear's peak is its own path, not a second
    // encoder. One memory rule for every unit: RSS while the measured graphs and
    // weights are still resident; the build is not counted.
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
    );
}

fn file_bytes(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// The recognition product segment: 30 s → mel → encoder → the 390-row prompt
/// (3 head + 6 text + 375 audio + 6 tail) → body prefill → at most 2 048 new
/// tokens on the 2 438-row masked cache. A walk that reaches the budget without
/// EOS is printed as `unfinished`, not as an answer.
fn listen() {
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            cell_failed!("cell listen — open the pair: {e}");
        }
    };
    let tokenizer = match Tokenizer::from_model(&pair.body) {
        Ok(t) => t,
        Err(e) => {
            cell_failed!("cell listen — tokenizer: {e}");
        }
    };
    let env_path = std::env::var("STAND_ASR_WAV").ok();
    let path = match env_path {
        Some(p) => std::path::PathBuf::from(p),
        None => match crate::loader::voice_wav() {
            Ok(p) => p,
            Err(e) => {
                cell_failed!("cell listen — voice reference: {e}");
            }
        },
    };
    let offset = std::env::var("STAND_ASR_OFFSET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut audio = match host::wav_mono_16k(&path.display().to_string(), offset, SPEECH_SAMPLES) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell listen — audio: {e}");
        }
    };
    prepare_speech(&mut audio);
    let frames = audio.len() / graphs::MEL_HOP;
    if frames == 0 {
        cell_failed!("cell listen — `{}` gave no speech", path.display());
    }
    audio.truncate(frames * graphs::MEL_HOP);
    let samples = audio.len();
    let tokens = graphs::asr_tokens(frames);
    let device = rlx_runtime::Device::Gpu;
    let cfg = pair.body.config().clone();
    let hkv = cfg.kv_heads * cfg.head_dim;
    let vocab = match cfg.head {
        loader::Head::Separate { vocab, .. } | loader::Head::Tied { vocab, .. } => vocab,
    };
    let prompt_rows = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + tokens + ASR_TAIL_IDS.len();
    let capacity = ASR_PROMPT_ROWS + ASR_MAX_NEW;
    let mut build_ms = 0f64;

    let mut front = match Front::build(&pair, frames) {
        Ok(f) => f,
        Err(e) => {
            cell_failed!("cell listen — front build: {e}");
        }
    };
    build_ms += front.build_ms();
    let c0 = graphs::device_counts();
    let mut phases = Vec::new();

    let (mel_o, enc_o, proj_o) = front.run(audio, WARM_REPS);
    phases.push(Phase::single("mel", &mel_o));
    phases.push(Phase::single("encoder", &enc_o));
    phases.push(Phase::single("projector", &proj_o));
    drop(mel_o);
    drop(enc_o);
    drop(front);
    // Everything the unit built stays resident until the memory read at the end:
    // the peak is taken before any drop, for every unit.

    // The prompt: the tokenizer's real rows, with the thirteen audio rows
    // replaced by the encoder's projected features.
    let mut prompt = vec![0f32; prompt_rows * cfg.d];
    let mut text_ids: Vec<(usize, usize)> = Vec::new();
    for (i, id) in ASR_HEAD_IDS.iter().enumerate() {
        text_ids.push((i, *id));
    }
    for (i, id) in ASR_TEXT_IDS.iter().enumerate() {
        text_ids.push((ASR_HEAD_IDS.len() + i, *id));
    }
    for (i, id) in ASR_TAIL_IDS.iter().enumerate() {
        text_ids.push((ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + tokens + i, *id));
    }
    let picks: Vec<(usize, usize)> = text_ids.iter().map(|(_, id)| (0, *id)).collect();
    let t_rows = report::Stopwatch::start();
    let rows = match pair.body.table_rows_f32("token_embd.weight", &picks, cfg.d) {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell listen — reading prompt embeddings: {e}");
        }
    };
    let prompt_read_ms = t_rows.ms();
    for ((at, _), row) in text_ids.iter().zip(rows) {
        prompt[at * cfg.d..(at + 1) * cfg.d].copy_from_slice(&row);
    }
    for a in 0..tokens {
        let at = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + a;
        prompt[at * cfg.d..(at + 1) * cfg.d]
            .copy_from_slice(&proj_o.outs[0][a * cfg.d..(a + 1) * cfg.d]);
    }
    drop(proj_o);

    // Prompt prefill on a growing cache: all 28 rows, logits on every row.
    let t = report::Stopwatch::start();
    let mut prefill =
        match graphs::body_new_rows_head_chain(pair.body.clone(), prompt_rows, 0, true)
            .and_then(|m| graphs::Compiled::new(device, m))
        {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell listen — prompt build: {e}");
            }
        };
    build_ms += t.ms();
    let (cos, sin) = host::rope_tables_at(0, prompt_rows, cfg.head_dim, cfg.rope_base as f64);
    let inputs = vec![
        ("x".into(), prompt),
        ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
        ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
    ];
    let mut p = prefill.run(&inputs, WARM_REPS);
    let mut prefill_phase = Phase::single("prefill", &p);
    // The prompt is part of the measured path: the table read is priced in the
    // same phase as the graph that consumes the rows.
    prefill_phase.first_ms += prompt_read_ms;
    prefill_phase.step_ms += prompt_read_ms;
    prefill_phase.cold_ms += prompt_read_ms;
    prefill_phase.warm_ms += prompt_read_ms;
    phases.push(prefill_phase);
    // The decode is split into four layer groups: each group's arena sees only
    // its own blocks, while the full body's weights and K/V stay the same.
    let group_len = cfg.layers.div_ceil(4);
    let ranges: Vec<std::ops::Range<usize>> = (0..4)
        .map(|g| (g * group_len).min(cfg.layers)..((g + 1) * group_len).min(cfg.layers))
        .filter(|r| !r.is_empty())
        .collect();
    let mut k_groups: Vec<Vec<f32>> = ranges
        .iter()
        .map(|r| vec![0f32; capacity * r.len() * hkv])
        .collect();
    let mut v_groups: Vec<Vec<f32>> = ranges
        .iter()
        .map(|r| vec![0f32; capacity * r.len() * hkv])
        .collect();
    for (gi, range) in ranges.iter().enumerate() {
        for (local, l) in range.clone().enumerate() {
            for r in 0..prompt_rows {
                let dst = (r * range.len() + local) * hkv;
                let src = (l * prompt_rows + r) * hkv;
                k_groups[gi][dst..dst + hkv].copy_from_slice(&p.outs[1][src..src + hkv]);
                v_groups[gi][dst..dst + hkv].copy_from_slice(&p.outs[2][src..src + hkv]);
            }
        }
    }
    let kv_bytes: usize = k_groups
        .iter()
        .chain(v_groups.iter())
        .map(|c| c.len())
        .sum::<usize>()
        * 4;
    // `mask[0..written]` is the history the decode may attend to;
    // `mask[capacity]` is the one query slot, not a cache row.
    let mut mask = vec![0f32; capacity + 1];
    for r in 0..prompt_rows {
        mask[r] = 1.0;
    }
    mask[capacity] = 1.0;
    let mut next = argmax(&p.outs[3][..vocab]) as usize;
    p.outs.clear();
    drop(prefill);

    let t = report::Stopwatch::start();
    let mut decodes = Vec::with_capacity(ranges.len());
    for (gi, range) in ranges.iter().enumerate() {
        let head = gi + 1 == ranges.len();
        let d = match graphs::body_capacity_layers(pair.body.clone(), capacity, range.clone(), head)
            .and_then(|m| graphs::Compiled::new(device, m))
        {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell listen — decode group {gi} build: {e}");
            }
        };
        decodes.push(d);
    }
    build_ms += t.ms();
    for (gi, graph) in decodes.iter_mut().enumerate() {
        if !graph
            .runner
            .resident_cache(&k_groups[gi], &v_groups[gi], (1, 2))
        {
            cell_failed!("cell listen — resident cache refused for group {gi}");
        }
        k_groups[gi].clear();
        v_groups[gi].clear();
    }
    let nodes: usize = decodes
        .iter()
        .map(|g| g.ops.iter().map(|(_, n)| n).sum::<usize>())
        .sum();
    let mut generated: Vec<usize> = Vec::new();
    let mut written = prompt_rows;
    let mut decode_times: Vec<f64> = Vec::new();
    while !is_asr_eos(next) && generated.len() < ASR_MAX_NEW && written < capacity {
        let t_row = report::Stopwatch::start();
        let mut hidden = match pair
            .body
            .table_rows_f32("token_embd.weight", &[(0, next)], cfg.d)
        {
            Ok(mut v) => v.remove(0),
            Err(e) => {
                cell_failed!("cell listen — reading token {next}: {e}");
            }
        };
        let row_read_ms = t_row.ms();
        let (cos, sin) = host::rope_tables_at(written, 1, cfg.head_dim, cfg.rope_base as f64);
        let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
        let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
        let t = report::Stopwatch::start();
        let mut logits = Vec::new();
        let last = decodes.len().saturating_sub(1);
        for (gi, graph) in decodes.iter_mut().enumerate() {
            let inputs = vec![
                ("x".into(), hidden),
                ("cos".into(), cos.clone()),
                ("sin".into(), sin.clone()),
                ("mask".into(), mask.clone()),
            ];
            let o = match graph
                .runner
                .run_resident(&inputs, written, ranges[gi].len(), hkv)
            {
                Some(o) => o,
                None => {
                    cell_failed!("cell listen — resident feed refused at {written}, group {gi}");
                }
            };
            hidden = o[0].clone();
            if gi == last {
                logits = o[3].clone();
            }
        }
        decode_times.push(t.ms() + row_read_ms);
        generated.push(next);
        next = argmax(&logits[..vocab]) as usize;
        mask[written] = 1.0;
        written += 1;
    }
    // The decode phase: its first iteration is the cold pass, the rest give the
    // warm median.
    let mut decode_phase = Phase::loop_times("decode", &decode_times, nodes);
    decode_phase.first_ms = decode_times.first().copied().unwrap_or(0.0);
    decode_phase.cold_ms = decode_phase.first_ms
        + decode_phase.step_ms * (decode_phase.iters.saturating_sub(1)) as f64;
    phases.push(decode_phase);
    let cold_ms: f64 = phases.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = phases.iter().map(|p| p.warm_ms).sum();
    let passes = phases.iter().map(|p| p.passes).sum();
    let finished = is_asr_eos(next);
    let truncated = !finished && generated.len() >= ASR_MAX_NEW;
    let text = match generated.iter().position(|&id| id == 151704) {
        Some(at) => tokenizer.decode(&generated[at + 1..]),
        None => tokenizer.decode(&generated),
    };
    if truncated {
        println!("cell listen — UNFINISHED: the walk used all {ASR_MAX_NEW} tokens without EOS");
    }
    // The digest covers the answer itself, not just its length: the generated
    // ids, the next token that stopped the walk, and the EOS flag.
    let mut values: Vec<f32> = generated.iter().map(|&v| v as f32).collect();
    values.push(next as f32);
    values.push(if is_asr_eos(next) { 1.0 } else { 0.0 });
    values.push(written as f32);
    let fields = [
        ("frames", frames.to_string()),
        ("rows", tokens.to_string()),
        ("samples", samples.to_string()),
        ("offset", offset.to_string()),
        ("prompt", prompt_rows.to_string()),
        ("capacity", capacity.to_string()),
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
        ("text", format!("{:?}", text)),
        ("decode_steps", (written - prompt_rows).to_string()),
    ];
    let o = p;
    let peak = peak_rss();
    // No drop before this read: the peak is the unit's whole live state.
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    let c = graphs::device_counts();
    line(
        "listen",
        &fields,
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
    // first would leave the actual speech at its own, much lower level.
    let peak = audio.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.0 && peak < 0.1 {
        let gain = 0.7 / peak;
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
    );
}

/// The generation loop: 300 text tokens and 330 codec rows (~26 s).
/// Per frame the body step, then the fifteen predictor steps each reading the
/// code just drawn, then the sum of the sixteen code embeddings plus the next
/// text hidden row becomes the next frame's input.
fn speak() {
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
    let (blocks, hkv_pred, hd_pred) = match graphs::code_predictor_cache(&pair.sound) {
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
    let vocab_pred = pair
        .sound
        .plan("a.gen.code.embd.weight")
        .unwrap()
        .ggml_shape[1];
    let (embd, _) = match pair.sound.f32_file_order("a.gen.code.embd.weight") {
        Ok(v) => v,
        Err(e) => {
            cell_failed!("cell speak — reading `embd`: {e}");
        }
    };
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
    let mut body = match graphs::body_capacity_chain(pair.body.clone(), capacity)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — body build: {e}");
        }
    };
    build_ms += t.ms();
    let mut k_cache = vec![0f32; capacity * cfg.layers * hkv];
    let mut v_cache = vec![0f32; capacity * cfg.layers * hkv];
    for l in 0..cfg.layers {
        for r in 0..prompt_rows {
            let dst = (r * cfg.layers + l) * hkv;
            let src = (l * prompt_rows + r) * hkv;
            k_cache[dst..dst + hkv].copy_from_slice(&p.outs[1][src..src + hkv]);
            v_cache[dst..dst + hkv].copy_from_slice(&p.outs[2][src..src + hkv]);
        }
    }
    let mut mask = vec![0f32; capacity + 1];
    for r in 0..prompt_rows {
        mask[r] = 1.0;
    }
    mask[capacity] = 1.0;
    if !body.runner.resident_cache(&k_cache, &v_cache, (1, 2)) {
        cell_failed!("cell speak — body resident cache refused");
    }
    let mut kv_bytes = k_cache.len() + v_cache.len();
    k_cache.clear();
    v_cache.clear();

    // The code predictor on its own carried cache: one row and all fifteen heads
    // per submission, reset for every frame.
    let t = report::Stopwatch::start();
    let mut walk = match graphs::tts_code_predictor_step_all(pair.sound.clone(), capacity_pred)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — predictor build: {e}");
        }
    };
    build_ms += t.ms();
    let mut k_pred = vec![0f32; capacity_pred * blocks * hkv_pred];
    let mut v_pred = vec![0f32; capacity_pred * blocks * hkv_pred];
    if !walk.runner.resident_cache(&k_pred, &v_pred, (1, 2)) {
        cell_failed!("cell speak — predictor resident cache refused");
    }
    kv_bytes += k_pred.len() + v_pred.len();
    // The predictor's first step is its cold pass; no throw-away run.
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

        // Reset the predictor's cache for this frame and walk its fifteen heads.
        k_pred.fill(0.0);
        v_pred.fill(0.0);
        if !walk.runner.resident_cache(&k_pred, &v_pred, (1, 2)) {
            cell_failed!("cell speak — predictor rebind refused");
        }
        // The predictor is a two-row prefill followed by fourteen generated
        // levels: row 0 is the talker hidden, row 1 is code0, and head `s`
        // reads row `s+1` and predicts level `s+1`.
        for s in 0..=steps {
            let (cos, sin) = host::rope_tables_at(s, 1, hd_pred, 1e6);
            let row = if s == 0 {
                hidden.clone()
            } else if s == 1 {
                code0_embed.clone()
            } else {
                let id = frame_codes[s - 1];
                let at = ((s - 2) * vocab_pred + id) * d_talker;
                embd[at..at + d_talker].to_vec()
            };
            let mut p_mask = vec![0f32; capacity_pred + 1];
            for p in 0..s {
                p_mask[p] = 1.0;
            }
            p_mask[capacity_pred] = 1.0;
            let inputs = vec![
                ("x".into(), row),
                ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
                ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
                ("mask".into(), p_mask),
            ];
            let t = report::Stopwatch::start();
            let out = walk.runner.run_read(&inputs, &[0]);
            // The predictor's cache is row-first now: this step's new rows of
            // every block are one contiguous row, so one feed carries all of
            // them instead of one submission per block.
            if !walk.runner.feed_rows(0, s, blocks * hkv_pred) {
                cell_failed!("cell speak — predictor feed refused at step {s}");
            }
            pred_times.push(t.ms());
            if s == 0 {
                continue;
            }
            let at = (s - 1) * head_out;
            let t = report::Stopwatch::start();
            let logits = out[0][at..at + head_out].to_vec();
            let id = host::draw(&logits, top_k, top_p, temperature, 12345 + draw_index);
            sample_times.push(t.ms());
            if s == 1 {
                let same_seed = host::draw(&logits, top_k, top_p, temperature, 12345);
                let engine = sample_pred.run(&[("logits".into(), logits)], 0);
                let got = engine.outs[0].first().copied().unwrap_or(f32::NAN) as usize;
                sampler1 = if got == same_seed {
                    "equal".into()
                } else {
                    format!("host={same_seed} engine={got}")
                };
            }
            draw_index += 1;
            frame_codes.push(id);
        }
        // The next frame's input: the sixteen code embeddings, summed.
        let mut sum = vec![0f32; d_talker];
        for (o, v) in sum.iter_mut().zip(&code0_embed) {
            *o += *v;
        }
        for level in 1..=steps {
            let id = frame_codes[level];
            let at = ((level - 1) * vocab_pred + id) * d_talker;
            for (o, v) in sum.iter_mut().zip(&embd[at..at + d_talker]) {
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
            let body_o = match body.runner.run_resident(&inputs, written, cfg.layers, hkv) {
                Some(o) => o,
                None => {
                    cell_failed!("cell speak — body feed refused at frame {f}");
                }
            };
            body_times.push(t.ms());
            mask[written] = 1.0;
            written += 1;
            hidden = body_o[0].clone();
            let mut logits = body_o[3][..body_vocab].to_vec();
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
    let pred_nodes = walk.ops.iter().map(|(_, n)| n).sum();
    phases.push(Phase::loop_times("body", &body_times, body_nodes));
    phases.push(Phase::loop_times("predictor", &pred_times, pred_nodes));
    phases.push(Phase::loop_times("sampler", &sample_times, 0));
    // No drop before the peak: `prefill`, `body`, `walk`, the prompt and the
    // predictor tables all stay alive through the sound half.

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
    let prefix = FRONT_WINDOW.saturating_sub(1);
    let t = report::Stopwatch::start();
    let mut front_first = match graphs::synthesis_sound_group(
        pair.sound.clone(),
        FRONT_WINDOW,
        front_group.clone(),
        0,
    )
    .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell speak — sound {front_group:?} build: {e}");
        }
    };
    let mut front_next = if frames > FRONT_WINDOW {
        match graphs::synthesis_sound_group(
            pair.sound.clone(),
            FRONT_WINDOW,
            front_group.clone(),
            prefix,
        )
        .and_then(|m| graphs::Compiled::new(device, m))
        {
            Ok(b) => Some(b),
            Err(e) => {
                cell_failed!("cell speak — sound {front_group:?} carried build: {e}");
            }
        }
    } else {
        None
    };
    build_ms += t.ms();
    let mut sound_graphs: Vec<graphs::Compiled> = Vec::new();
    let mut sound_ms = 0f64;
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
        let len = b - a;
        let mut chunk = Vec::with_capacity((levels + 1) * FRONT_WINDOW);
        for level in 0..=levels {
            for f in a..b {
                chunk.push(flat[level * frames + f]);
            }
            chunk.resize(chunk.len() + (FRONT_WINDOW - len), 0.0);
        }
        let (cos, sin) = host::rope_tables_at(a, FRONT_WINDOW, wave_hd, 1e4);
        let mut inputs = vec![
            ("codes".into(), chunk),
            ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
            ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
            ("left".into(), left.clone()),
        ];
        let graph = if a == 0 {
            &mut front_first
        } else {
            match front_next.as_mut() {
                Some(g) => g,
                None => {
                    cell_failed!("cell speak — a carried front is missing for frame {a}");
                }
            }
        };
        if a > 0 && prefix > 0 {
            inputs.push(("kpre".into(), kv_k.clone()));
            inputs.push(("vpre".into(), kv_v.clone()));
        }
        let run = graph.run(&inputs, 1);
        if run.outs.len() < 4 {
            println!(
                "cell speak — the front returned {} outputs, expected waveform + k/v/left",
                run.outs.len()
            );
            return;
        }
        front_nodes = front_nodes.max(run.ops.iter().map(|(_, n)| n).sum());
        let chunk = run.outs[0].clone();
        for ch in 0..front_ch {
            let src = ch * FRONT_WINDOW * front_per;
            let dst = ch * frames * front_per + a * front_per;
            wave[dst..dst + len * front_per].copy_from_slice(&chunk[src..src + len * front_per]);
        }
        // Side outputs are published in this order: `left`, `knew`, `vnew`.
        // Keep the last 71 rows of this window for the next one.
        let knew = &run.outs[2];
        let vnew = &run.outs[3];
        if prefix > 0 && knew.len() >= wave_layers * FRONT_WINDOW * wave_hkv {
            for l in 0..wave_layers {
                let src = (l * FRONT_WINDOW + (FRONT_WINDOW - prefix)) * wave_hkv;
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
        sound_ms += run.step_ms;
        if o.is_none() {
            o = Some(run);
        }
    }
    phases.push(Phase::windows(
        "sound.front".to_string(),
        &front_stats,
        front_nodes,
    ));
    sound_graphs.push(front_first);
    if let Some(front_next) = front_next {
        sound_graphs.push(front_next);
    }
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
            return;
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
        let mut graphs_by_len: Vec<(usize, graphs::Compiled)> = Vec::with_capacity(lengths.len());
        for &len in &lengths {
            match graphs::synthesis_sound_group(pair.sound.clone(), len, group.clone(), 0)
                .and_then(|m| graphs::Compiled::new(device, m))
            {
                Ok(b) => graphs_by_len.push((len, b)),
                Err(e) => {
                    println!(
                        "cell speak — sound {}..{} at {len} frames build: {e}",
                        group.start, group.end
                    );
                    return;
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
                .run(&[("codec_in".into(), chunk)], WARM_REPS);
            nodes = nodes.max(run.ops.iter().map(|(_, n)| n).sum());
            sound_ms += run.step_ms;
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
        sound_graphs.extend(graphs_by_len.into_iter().map(|(_, g)| g));
        wave = stitched;
    }
    if wave.len() != frames * graphs::samples_per_frame() {
        println!(
            "cell speak — the sound half left {} values, a waveform is {}",
            wave.len(),
            frames * graphs::samples_per_frame()
        );
        return;
    }
    if let Ok(path) = std::env::var("STAND_SPEAK_WAV") {
        if let Err(e) = host::write_wav16(&path, &wave, 24_000) {
            println!("cell speak — writing `{path}`: {e}");
        }
    }
    let o = match o {
        Some(o) => o,
        None => return,
    };
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
    let fields = [
        ("rows", frames.to_string()),
        ("predictor_steps", steps.to_string()),
        ("sound_graphs", groups.len().to_string()),
        ("cache", "resident".to_string()),
        ("kv_bytes", (kv_bytes * 4).to_string()),
        ("dac", format!("{DAC_WINDOW}/{DAC_MARGIN}x{}", parts.len())),
        ("body_ms", fmt_ms(body_total)),
        ("predictor_ms", fmt_ms(pred_total)),
        ("sampler_ms", fmt_ms(sampler_total)),
        ("sound_ms", fmt_ms(sound_ms)),
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
    let cell = "speak";
    let peak = peak_rss();
    // No drop before the peak: body, predictor, prompt, sound graphs and the
    // codec all remain resident.
    let map_bytes = file_bytes(&pair.body.path) + file_bytes(&pair.sound.path);
    let c = graphs::device_counts();
    line(
        cell,
        &fields,
        &o,
        &wave,
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
    );
}

/// The 8 192-token embedding cell walked one row at a time on a fixed-capacity
/// cache. The block form rebuilt a graph per 512-row prefix and kept the whole
/// prefix twice on the host; the one-row form carries the cache as device-resident
/// handles and keeps only the graph's own row. Causal attention is exact — each
/// step sees every row written before it and nothing after.
fn embed_cell(
    model: Arc<Model>,
    cfg: &loader::Config,
    ids: &[usize],
    text_len: usize,
    cell: usize,
    device: rlx_runtime::Device,
) -> anyhow::Result<(Outcome, Vec<f32>)> {
    const BLOCK: usize = 512;
    const GROUPS: usize = 8;
    let hkv = cfg.kv_heads * cfg.head_dim;
    let cell_cache = cell * hkv;
    // Four layer groups: each group's arena sees only its own blocks, while the
    // K/V for the whole body stays one row-first cache per group.
    let group_len = cfg.layers.div_ceil(GROUPS);
    let ranges: Vec<std::ops::Range<usize>> = (0..GROUPS)
        .map(|g| (g * group_len).min(cfg.layers)..((g + 1) * group_len).min(cfg.layers))
        .filter(|r| !r.is_empty())
        .collect();
    let mut k_cache: Vec<Vec<f32>> = ranges
        .iter()
        .map(|r| vec![0f32; cell_cache * r.len()])
        .collect();
    let mut v_cache: Vec<Vec<f32>> = ranges
        .iter()
        .map(|r| vec![0f32; cell_cache * r.len()])
        .collect();
    let mut times = Vec::new();
    let mut pooled = Vec::new();
    let mut graphs = Vec::with_capacity(ranges.len());
    for r in &ranges {
        graphs.push(graphs::Compiled::new(
            device,
            graphs::body_block_causal_layers(model.clone(), BLOCK, cell, r.clone())?,
        )?);
    }
    // The K/V is bound one group at a time, just before its first block, so the
    // host mirror of a group is gone by the time the next group is bound. Each
    // block then feeds its own rows straight into the resident cache.
    let compile_ms: f64 = graphs.iter().map(|g| g.compile_ms).sum();
    let upload_ms: f64 = graphs.iter().map(|g| g.upload_ms).sum();
    let arena_bytes = graphs.iter().map(|g| g.arena_bytes).max().unwrap_or(0);
    let params_f32: usize = graphs.iter().map(|g| g.params_f32).sum();
    let params_packed: usize = graphs.iter().map(|g| g.params_packed).sum();
    let ops = graphs.last().map(|g| g.ops.clone()).unwrap_or_default();
    let mut start = 0usize;
    while start < cell {
        let rows = BLOCK.min(cell - start);
        let picks: Vec<(usize, usize)> =
            ids[start..start + rows].iter().map(|&id| (0, id)).collect();
        let mut table = model.table_rows_f32("token_embd.weight", &picks, cfg.d)?;
        let mut x = Vec::with_capacity(rows * cfg.d);
        for row in &mut table {
            x.append(row);
        }
        let (cos, sin) = host::rope_tables_at(start, rows, cfg.head_dim, cfg.rope_base as f64);
        let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
        let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
        let t = report::Stopwatch::start();
        let last = graphs.len().saturating_sub(1);
        for (gi, graph) in graphs.iter_mut().enumerate() {
            let glen = ranges[gi].len();
            if start == 0 {
                if !graph
                    .runner
                    .resident_cache(&k_cache[gi], &v_cache[gi], (1, 2))
                {
                    anyhow::bail!("embed — resident cache refused for group {gi}");
                }
                k_cache[gi].clear();
                v_cache[gi].clear();
            }
            let inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("cos", &cos), ("sin", &sin)];
            let out = graph.run_slices(&inputs);
            // A packed feed declares the cache as `[block / rows, rows · layers
            // · hkv]`; the block's side output is one row of that width, so all
            // rows of this block go in one device copy. A cell smaller than a
            // block keeps the row-at-a-time path.
            if cell >= BLOCK && cell.is_multiple_of(BLOCK) {
                if !graph.runner.feed_rows(0, start / rows, rows * glen * hkv) {
                    anyhow::bail!("embed — packed feed refused at group {gi}, block {start}");
                }
            } else {
                for r in 0..rows {
                    if !graph.runner.feed_rows(r, start + r, glen * hkv) {
                        anyhow::bail!("embed — resident feed refused at group {gi}, row {r}");
                    }
                }
            }
            if gi == last && pooled.is_empty() && start < text_len && text_len <= start + rows {
                let at = (text_len - 1 - start) * cfg.d;
                pooled = out[0][at..at + cfg.d].to_vec();
            }
            x = out[0].clone();
        }
        times.push(t.ms());
        start += rows;
    }
    drop(graphs);
    embed_finish(
        times,
        compile_ms,
        upload_ms,
        arena_bytes,
        params_f32,
        params_packed,
        ops,
        pooled,
    )
}

fn embed_finish(
    times: Vec<f64>,
    compile_ms: f64,
    upload_ms: f64,
    arena_bytes: usize,
    params_f32: usize,
    params_packed: usize,
    ops: Vec<(String, usize)>,
    pooled: Vec<f32>,
) -> anyhow::Result<(Outcome, Vec<f32>)> {
    let v = l2(&pooled);
    let total: f64 = times.iter().sum();
    let o = Outcome {
        compile_ms,
        upload_ms,
        first_ms: total,
        step_ms: total,
        min_ms: times.iter().copied().fold(f64::INFINITY, f64::min),
        max_ms: times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        outs: vec![pooled],
        arena_bytes,
        dequant_nodes: 0,
        params_f32,
        params_packed,
        ops,
        passes: times.len(),
    };
    Ok((o, v))
}

/// The two embedding cells: the file's own tokenizer encodes the text, the cell
/// is filled to its fixed length with the padding id, the body emits the last
/// non-padding row, and the cell normalizes it to L2. The first cell is a normal
/// request (256 tokens), the second is the model limit (8 192), walked as causal
/// blocks so its attention never becomes the cell's whole square.
fn embed() {
    let path = match crate::loader::models_dir() {
        Ok(dir) => dir.join("Qwen3-Embedding-0.6B-Q4_K_M.gguf"),
        Err(e) => {
            cell_failed!("cell embed — model directory: {e}");
        }
    };
    let model = match Model::open(&path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            cell_failed!("cell embed — open: {e}");
        }
    };
    let tokenizer = match Tokenizer::from_model(&model) {
        Ok(t) => t,
        Err(e) => {
            cell_failed!("cell embed — tokenizer: {e}");
        }
    };
    let cfg = model.config().clone();
    let device = rlx_runtime::Device::Gpu;
    let mut text = tokenizer.encode(EMBED_TEXT);
    text.truncate(*EMBED_CELLS.last().unwrap());
    let pad = model
        .file
        .metadata
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(151643) as usize;
    let text_len = text.len().max(1);
    let mut build_ms = 0f64;
    let c0 = graphs::device_counts();
    let mut phases = Vec::new();
    let mut values = Vec::new();
    let mut last_outcome: Option<Outcome> = None;
    for &cell in &EMBED_CELLS {
        let mut ids = text.clone();
        ids.truncate(cell);
        while ids.len() < cell {
            ids.push(pad);
        }
        let last = ids.len().min(text_len) - 1;
        let t = report::Stopwatch::start();
        let (o, v) = if cell >= 8_192 {
            match embed_cell(model.clone(), &cfg, &ids, text_len, cell, device) {
                Ok(v) => v,
                Err(e) => {
                    cell_failed!("cell embed — serial {cell}: {e}");
                }
            }
        } else {
            let mut graph = match graphs::body_graph(
                model.clone(),
                cell,
                BodyInput::Ids {
                    table_rows: cfg.vocab,
                    last,
                },
            )
            .and_then(|m| graphs::Compiled::new(device, m))
            {
                Ok(b) => b,
                Err(e) => {
                    cell_failed!("cell embed — build {cell}: {e}");
                }
            };
            let o = graph.run(
                &[("x".into(), ids.iter().map(|v| *v as f32).collect())],
                WARM_REPS,
            );
            let v = l2(&o.outs[0]);
            (o, v)
        };
        build_ms += t.ms();
        println!(
            "cell embed.{cell} tokens={cell} text_tokens={} pool_row={last} norm={:.6}",
            text_len,
            v.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        values.extend_from_slice(&v);
        phases.push(Phase::single(format!("embed.{cell}"), &o));
        last_outcome = Some(o);
    }
    let o = match last_outcome {
        Some(o) => o,
        None => return,
    };
    let cold_ms: f64 = phases.iter().map(|p| p.cold_ms).sum();
    let warm_ms: f64 = phases.iter().map(|p| p.warm_ms).sum();
    let passes = phases.iter().map(|p| p.passes).sum();
    let steps = Steps {
        build_ms,
        cold_ms,
        warm_ms,
        passes,
    };
    let c = graphs::device_counts();
    let peak = peak_rss();
    let map_bytes = file_bytes(&model.path);
    drop(model);
    report::release_free();
    line(
        "embed",
        &[
            ("cells", format!("{:?}", EMBED_CELLS)),
            ("text_tokens", text_len.to_string()),
            ("pad", pad.to_string()),
            ("table_rows", cfg.vocab.to_string()),
            ("dim", cfg.d.to_string()),
        ],
        &o,
        &values,
        &phases,
        steps,
        peak,
        map_bytes,
        (c.0 - c0.0, c.1 - c0.1),
    );
}

/// The mel front end alone, on a window of the same size the sound file is read at.
fn mel() {
    let frames = MEL_FRAMES;
    let bins = 128;
    let device = rlx_runtime::Device::Gpu;
    let audio = pseudo_rand(graphs::MEL_HOP * frames, 5);
    let t = report::Stopwatch::start();
    let mut built =
        match graphs::mel_frontend(frames, bins).and_then(|m| graphs::Compiled::new(device, m)) {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell mel — build: {e}");
            }
        };
    let build_ms = t.ms();
    let c0 = graphs::device_counts();
    let o = built.run(&[("audio".into(), audio)], WARM_REPS);
    let values = o.outs[0].clone();
    let phases = vec![Phase::single("mel", &o)];
    let steps = Steps {
        build_ms,
        cold_ms: o.first_ms,
        warm_ms: o.step_ms,
        passes: o.passes,
    };
    let c = graphs::device_counts();
    let peak = peak_rss();
    drop(built);
    report::release_free();
    line(
        "mel",
        &[("frames", frames.to_string()), ("bins", bins.to_string())],
        &o,
        &values,
        &phases,
        steps,
        peak,
        0,
        (c.0 - c0.0, c.1 - c0.1),
    );
}

/// The attention biases on this path: what the fusion pass does with them, read off
/// the graph after the pass the backend itself runs.
fn bias() {
    let Some((_, sound)) = loader::pair_files(Role::Recognition) else {
        cell_failed!("cell bias — no sound file");
    };
    let model = match crate::loader::models_dir().and_then(|dir| Model::open(dir.join(sound))) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            cell_failed!("cell bias — open {sound}: {e}");
        }
    };
    let t = report::Stopwatch::start();
    let built = match graphs::asr_encoder(model.clone(), BIAS_FRAMES) {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell bias — build: {e}");
        }
    };
    let Some(hir) = built.module.into_hir() else {
        cell_failed!("cell bias — flow built a non-HIR module");
    };
    let graph = match rlx_ir::hir_to_graph(hir) {
        Ok(g) => g,
        Err(e) => {
            cell_failed!("cell bias — lowering: {e}");
        }
    };
    let before = histogram(&graph);
    let biases = biases(&graph);
    let options = rlx_runtime::CompileOptions::new().with_fusion_report(true);
    let result = match rlx_runtime::stages::compile_module_stages(
        rlx_runtime::Device::Gpu,
        rlx_ir::GraphModule::from_graph(graph),
        &options,
    ) {
        Ok(r) => r,
        Err(e) => {
            cell_failed!("cell bias — compile: {e}");
        }
    };
    let after = histogram(result.lir.mir.as_graph());
    let build_ms = t.ms();
    let report = format!("{}", result.fusion);
    let merged: Vec<&str> = report.lines().filter(|l| l.contains("nodes=")).collect();
    println!(
        "cell bias frames={BIAS_FRAMES} biases={} on_a_narrow={} build_ms={build_ms:.1} before={} after={} pass={} keys={}",
        biases.0,
        biases.1,
        before,
        after,
        merged.last().map(|s| s.trim()).unwrap_or("none"),
        keys(),
    );
}

/// The recognition head on one row: the file's own packed multiply, and the draw on
/// its logits.
fn head() {
    let Some((body, _)) = loader::pair_files(Role::Recognition) else {
        cell_failed!("cell head — no body file");
    };
    let model = match crate::loader::models_dir().and_then(|dir| Model::open(dir.join(body))) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            cell_failed!("cell head — open {body}: {e}");
        }
    };
    let cfg = model.config().clone();
    let vocab = match cfg.head {
        loader::Head::Separate { vocab, .. } | loader::Head::Tied { vocab, .. } => vocab,
    };
    let device = rlx_runtime::Device::Gpu;
    let sampler = SampleSpec {
        top_k: 50,
        top_p: 0.9,
        temperature: 0.9,
        seed: 12345,
    };
    // The build step: both graphs the cell walks, compiled before either runs.
    let t = report::Stopwatch::start();
    let mut head_graph =
        match graphs::head_graph(model.clone(), 1).and_then(|m| graphs::Compiled::new(device, m)) {
            Ok(b) => b,
            Err(e) => {
                cell_failed!("cell head — build: {e}");
            }
        };
    let mut sampler_graph = match graphs::sample_graph_one(vocab, sampler)
        .and_then(|m| graphs::Compiled::new(device, m))
    {
        Ok(b) => b,
        Err(e) => {
            cell_failed!("cell head.sample — build: {e}");
        }
    };
    let build_ms = t.ms();

    let c0 = graphs::device_counts();
    let o = head_graph.run(&[("x".into(), pseudo_rand(cfg.d, 5))], WARM_REPS);
    let values = o.outs.first().cloned().unwrap_or_default();
    let phases = vec![Phase::single("head", &o)];
    let steps = Steps {
        build_ms,
        cold_ms: o.first_ms,
        warm_ms: o.step_ms,
        passes: o.passes,
    };
    let c = graphs::device_counts();
    line(
        "head",
        &[("rows", "1".to_string()), ("vocab", vocab.to_string())],
        &o,
        &values,
        &phases,
        steps,
        peak_rss(),
        0,
        (c.0 - c0.0, c.1 - c0.1),
    );

    let c0 = graphs::device_counts();
    let o = sampler_graph.run(&[("logits".into(), pseudo_rand(vocab, 7))], WARM_REPS);
    let values = o.outs.first().cloned().unwrap_or_default();
    let phases = vec![Phase::single("head.sample", &o)];
    let steps = Steps {
        build_ms,
        cold_ms: o.first_ms,
        warm_ms: o.step_ms,
        passes: o.passes,
    };
    let c = graphs::device_counts();
    line(
        "head.sample",
        &[("vocab", vocab.to_string())],
        &o,
        &values,
        &phases,
        steps,
        peak_rss(),
        0,
        (c.0 - c0.0, c.1 - c0.1),
    );
}

// ── small readings the cells share ─────────────────────────────────

/// The op histogram of a lowered graph, most frequent first.
fn histogram(g: &rlx_ir::Graph) -> String {
    let mut h: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for n in g.nodes() {
        *h.entry(format!("{:?}", n.op.kind())).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = h.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.iter()
        .map(|(k, n)| format!("{k}:{n}"))
        .collect::<Vec<_>>()
        .join("+")
}

/// `(attention biases added, of which next to a merged multiply's `Narrow`)`.
fn biases(g: &rlx_ir::Graph) -> (usize, usize) {
    let mut total = 0usize;
    let mut on_a_narrow = 0usize;
    for n in g.nodes() {
        if !matches!(n.op, rlx_ir::Op::Binary(rlx_ir::op::BinaryOp::Add)) {
            continue;
        }
        let has_bias = n.inputs.iter().any(|i| {
            matches!(&g.node(*i).op, rlx_ir::Op::Param { name }
                if name.ends_with(".bias") && name.contains(".attn_"))
        });
        if !has_bias {
            continue;
        }
        total += 1;
        if n.inputs
            .iter()
            .any(|x| matches!(g.node(*x).op, rlx_ir::Op::Narrow { .. }))
        {
            on_a_narrow += 1;
        }
    }
    (total, on_a_narrow)
}
