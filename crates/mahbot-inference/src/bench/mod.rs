// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake-word bench: recall, false reactions and false accepts per hour on
//! this crate's own models.
//!
//! One run measures three things:
//!
//! 1. **recall** — 40 bare wake-phrase clips plus 10 wake-phrase-plus-command
//!    clips, fed through one warm continuous stream;
//! 2. **false reactions** — 121 non-phrase clips (confusables, unrelated
//!    speech, silence, noise, phrase-in-passing), through the same kind of
//!    stream;
//! 3. **false accepts per hour** — the pinned 33-file real-audio subset, fed in
//!    order through one continuous stream.
//!
//! The enrolment is the one enrolled voice: ten conditioned renderings, a
//! consistency gate, and a negative pool that shares no clip with either
//! measurement set. Everything the report states was measured on the models the
//! run was given; a run that measured nothing is an error, not a report.
//!
//! # Window groups
//!
//! The real-audio sweep embeds its windows in groups of [`listen::GROUP`]: the
//! mel front end and the conv front run per window, and the encoder tail and the
//! projector run once for the whole group, so the quantised weights are read once
//! per sixteen windows instead of once per window per layer. The decisions are
//! applied in the order the windows arrived: a group that contains a fire drops
//! the windows after it — the stream would not have scored them — and the
//! cooldown starts at the fire's own position in the audio, not at the moment the
//! group was resolved. Anything the stream holds that a segment close would reset
//! is flushed before the reset, so no queued window is ever scored into state it
//! did not see.
//!
//! Recall and false reactions run ungrouped, one window per pass: their two
//! phases cost about sixty seconds together, and the ungrouped path keeps their
//! decisions free of any group boundary. The group facts are a property of the
//! run and the report states them.
//!
//! # Guards
//!
//! A run holds an exclusive lock (`--lock` / `MAHBOT_LOCK`, defaulting to the one
//! shared path every heavy run of this crate takes) so a second launch refuses
//! instead of competing for the same device. It carries a wall-clock stop
//! (3 600 s by default) and a memory guard covering the process's resident set,
//! swap growth since the run started and the system's free-memory percentage.
//!
//! The work items check the deadline at every synthesized clip, detection chunk
//! and corpus file, and the cheap process figure at every clip and corpus file;
//! the two-subprocess system figures run at phase and file boundaries. A guard
//! thread watches the wall clock every second and both figures every five, so a
//! stop fires even while the measuring thread is inside one blocked device call.
//! A report is written once, by whichever thread gets there first — a stop or a
//! fatal error leaves what it measured, names the phase it stopped in, and exits
//! non-zero.
//!
//! # Inputs
//!
//! See the private `config` module: an argument first, the environment second.
//! The report and the lock are inputs too, so a run never writes into the model
//! directory; the synthesized corpus has one fixed, durable location instead
//! (below). There is no crate-relative path and no network access.
//!
//! # The record
//!
//! One run is this surface's benchmark: it scores with the shipped threshold set
//! over all three measurements, reports `complete`, and rewrites this surface's
//! section of the repository's `MEASUREMENTS.md` (see [`crate::measurements`]);
//! `MAHBOT_MEASUREMENTS` points that write at another file, which is how a run
//! can be checked without editing the tree.
//!
//! # Pricing an operating point
//!
//! `--wake-thresholds` overrides the firing-threshold set ([`crate::wake`]) and
//! `--phases` takes a subset of the three measurements — the recall stream alone
//! prices a candidate in half a minute instead of the run's five. Both are
//! measurement aids, and the classification is by value rather than by flag: a
//! run that prices another set, or measures a subset, reports `screened` and
//! leaves `MEASUREMENTS.md` alone even when it measured all three phases, while a
//! run that states the shipped set — naming it or not — over all three phases is
//! the benchmark and writes the record.
//!
//! # The synthesized corpus
//!
//! The corpus is 394 clips. It is generated once into a fixed, durable cache
//! (`$XDG_CACHE_HOME/mahbot/wake-word-corpus`, else `~/.cache/…`), not into a
//! per-run directory: a clip is synthesized only when the cache key — the text,
//! the seed, the sample rate, both TTS files, the voice reference and the code
//! lines of the modules that produce it — has no entry. A run that reused the
//! corpus and a run that had to build it are both stated, with their counts and
//! the time synthesis took; a cache that cannot be written stops the run instead
//! of quietly costing the corpus again.

mod audio;
pub(crate) mod config;
mod detect;
mod real_audio;
mod sets;
// The stand transcribes the same corpus the false-accept phase feeds, so its pin
// is named from `crate::stand` without opening the bench's private modules.
pub(crate) use sets::REAL_AUDIO_FILES;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::listen;
use crate::loader::Pair;
use crate::measurements;
use crate::synth::Synthesis;
use crate::wake::{self, Enrollment};

use crate::guard::{Abort, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock, Watch};
use audio::Synth;
use config::BenchConfig;
use detect::DetectionStream;

/// Run the bench once: measure, write the report atomically.
///
/// # Errors
/// A missing or malformed input; a model, corpus or synthesis failure; an
/// enrolment that fails its consistency gate; the wall-clock stop; or the
/// resident-set ceiling. The report is written for every one of those — a stop
/// or a failure leaves what it measured, and a run whose denominator measured
/// nothing writes its report and then exits non-zero.
pub fn run() -> Result<()> {
    let started = Instant::now();
    let cfg = BenchConfig::from_env()?;
    let _lock = RunLock::acquire(&cfg.lock)?;
    let deadline = Deadline::new(cfg.timeout_secs);
    let ceiling = MemoryCeiling::new(cfg.max_rss_mib);
    let progress = Arc::new(Mutex::new(Progress::default()));
    let report = Arc::new(Report::default());
    let watchdog = match Watch::install(
        "wake-word bench",
        deadline,
        ceiling,
        {
            let progress = Arc::clone(&progress);
            move || state(&progress).current
        },
        {
            let cfg = cfg.clone();
            let progress = Arc::clone(&progress);
            let report = Arc::clone(&report);
            move |abort: Abort| {
                let phase = abort.phase();
                let message = abort.to_string();
                eprintln!("[wake-word] stopped in `{phase}`: {message}");
                match report.write(
                    &cfg,
                    &progress,
                    started,
                    "aborted",
                    Some((phase, message.as_str())),
                ) {
                    Ok(_) => eprintln!("[wake-word] partial report: {}", cfg.report.display()),
                    Err(error) => eprintln!(
                        "[wake-word] cannot write the report {}: {error:#} — stopping without one",
                        cfg.report.display()
                    ),
                }
                std::process::exit(EXIT_STOPPED);
            }
        },
    ) {
        Ok(watchdog) => watchdog,
        Err(error) => {
            // Without the guard thread a single blocked device call could
            // outlive the stop, so the run does not start — and says why.
            eprintln!("[wake-word] cannot start the guard thread: {error:#}");
            let _ = report.write(
                &cfg,
                &progress,
                started,
                "failed",
                Some(("inputs", &format!("{error:#}"))),
            );
            return Err(error);
        }
    };
    eprintln!(
        "[wake-word] models {} · corpus {} · up to {} windows per real-audio pass (the clip \
         sets run one per pass) · stop {}s · ceiling {} MiB · lock {}",
        cfg.models_label(),
        cfg.corpus.display(),
        listen::GROUP,
        deadline.secs(),
        cfg.max_rss_mib,
        cfg.lock.display()
    );

    let measured = measure(&cfg, deadline, ceiling, &progress);
    // A denominator that measured nothing: a rate with no denominator.
    let unmeasured = match &measured {
        Ok(()) => require_measured(&state(&progress), cfg.phases).err(),
        Err(_) => None,
    };
    let (status, stop) = classify(
        &cfg,
        &measured,
        unmeasured.as_ref(),
        state(&progress).current,
    );
    let write = report.write(
        &cfg,
        &progress,
        started,
        status,
        stop.as_ref().map(|(phase, text)| (*phase, text.as_str())),
    );
    watchdog.finish();
    if let Some((phase, message)) = &stop {
        eprintln!("[wake-word] stopped in `{phase}`: {message}");
    }
    match (&write, &stop) {
        (Ok(_), Some(_)) => eprintln!("[wake-word] partial report: {}", cfg.report.display()),
        (Ok(_), None) => eprintln!("[wake-word] report: {}", cfg.report.display()),
        (Err(error), _) => eprintln!(
            "[wake-word] cannot write the report {}: {error:#}",
            cfg.report.display()
        ),
    }
    if let Err(error) = measured {
        // A guard stop at a work-item boundary exits with the code the guard
        // thread uses, so a caller can tell a stop from a fatal error.
        if error.downcast_ref::<Abort>().is_some() {
            std::process::exit(EXIT_STOPPED);
        }
        return Err(error);
    }
    // The report is the deliverable: a measurement whose report could not be
    // written is not a successful run.
    let written = match write {
        Ok(written) => written,
        Err(error) => {
            return Err(error.context(format!(
                "the measurement completed but the report {} could not be written",
                cfg.report.display()
            )));
        }
    };
    if let Some(error) = unmeasured {
        return Err(error);
    }

    let progress = state(&progress);
    if let Some(real) = &progress.real {
        eprintln!(
            "[wake-word] real audio: {} files, {:.2} h audio ({:.2} h speech, {:.2} h VAD-active), \
             {} FA events · {:.4}/h over VAD-active speech (the headline's denominator), \
             {:.4}/h over wall-clock audio · one stream · {:.1}s wall · peak {} MiB",
            real.files_fed,
            real.audio_hours,
            real.speech_hours,
            real.vad_active_hours,
            real.events,
            real.events as f64 / real.vad_active_hours,
            real.events as f64 / real.audio_hours,
            real.wall_secs,
            crate::report::peak_rss() / (1024 * 1024),
        );
    }
    let corpus = progress.synthesis;
    eprintln!(
        "[wake-word] corpus {}: {} of {} clips reused, {} synthesized in {:.1}s",
        cfg.cache.display(),
        corpus.cache_hits,
        corpus.clips,
        corpus.clips - corpus.cache_hits,
        corpus.synth_secs
    );
    // The repository's own record, refreshed from the same JSON this run wrote.
    // It is part of the run, not a courtesy: a change is not complete until the
    // file shows the numbers from after it, so a run that cannot refresh it says
    // so and fails.
    // `Ok(None)` means the guard thread claimed the report: it writes the stop's
    // own file and exits the process, so this thread never reaches the console if
    // the stop had already fired. A measurement whose report another thread owns
    // is a stopped run, not a successful one, so it exits the way the guard does.
    let Some(written) = written else {
        eprintln!(
            "[wake-word] the guard thread wrote the report {} before the measurement returned",
            cfg.report.display()
        );
        std::process::exit(EXIT_STOPPED);
    };
    // Only the benchmark itself is the record: a screened run priced a candidate
    // point and says so instead.
    if status == COMPLETE {
        measurements::refresh("bench", &record_body(&written)).with_context(|| {
            format!(
                "the measurement completed but {} could not be refreshed",
                measurements::path().display()
            )
        })?;
    } else {
        eprintln!(
            "[wake-word] screened run at {} — the record {} is left to the benchmark",
            progress.point(&cfg),
            measurements::path().display()
        );
    }
    Ok(())
}

/// The status of the benchmark itself — the only one whose numbers the record
/// states. One name for the producer and the gate, so they cannot desynchronize.
const COMPLETE: &str = "complete";

/// The one classification that drives the report, the console and the record.
///
/// `complete` is the benchmark itself: the shipped threshold set over all three
/// phases. A run that priced a candidate — another set, or a subset of the
/// phases — is `screened` and never writes the record, even when it measured
/// everything; anything that stopped is named with the phase and the reason.
fn classify(
    cfg: &BenchConfig,
    measured: &Result<()>,
    unmeasured: Option<&anyhow::Error>,
    phase: &'static str,
) -> (&'static str, Option<(&'static str, String)>) {
    let benchmark = cfg.thresholds == wake::Thresholds::DEFAULT && cfg.phases.is_full();
    match (&measured, &unmeasured) {
        (Ok(()), None) if benchmark => (COMPLETE, None),
        (Ok(()), None) => ("screened", None),
        (Ok(()), Some(error)) => ("incomplete", Some((phase, format!("{error:#}")))),
        (Err(error), _) => match error.downcast_ref::<Abort>() {
            Some(abort) => ("aborted", Some((abort.phase(), abort.to_string()))),
            None => ("failed", Some((phase, format!("{error:#}")))),
        },
    }
}

/// What a bench row prints for a figure its report does not carry: the same words
/// every row uses, so no figure is ever printed as a zero the run did not measure.
const UNAVAILABLE: &str = "not in the report";

/// One row of the record's table: the figures the row states, or [`UNAVAILABLE`].
fn row(body: &mut String, label: &str, figures: Option<String>) {
    body.push_str(&format!(
        "| {label} | {} |\n",
        figures.as_deref().unwrap_or(UNAVAILABLE)
    ));
}

/// [`row`] for a row of numbers, each bound by name out of the report. The row is
/// written with its render only when every figure it states is present.
macro_rules! figure_row {
    ($body:expr, $label:expr, [$($name:ident = $value:expr),+ $(,)?] => $render:expr) => {
        row(
            &mut $body,
            $label,
            match ($($value.and_then(Value::as_f64),)+) {
                ($(Some($name),)+) => Some($render),
                _ => None,
            },
        )
    };
}

/// The bench section this run's report states — read out of the report the run
/// itself wrote, so the file and the JSON cannot state the same fact twice or a
/// degenerate rate differently.
fn record_body(report: &Value) -> String {
    let mut body = String::new();
    body.push_str("## Wake-word bench\n\n");
    body.push_str(&format!("_Measured {}._\n\n", measurements::now()));
    body.push_str(&format!(
        "status: `{}`\n\n",
        report
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or(UNAVAILABLE)
    ));
    body.push_str("| metric | value |\n|---|---|\n");
    let recognition = report.get("recognition");
    figure_row!(body, "recall", [
        detected = recognition.and_then(|node| node.get("detected")),
        of = recognition.and_then(|node| node.get("of")),
        rate = recognition.and_then(|node| node.get("rate")),
    ] => format!("{detected}/{of} ({:.1} %)", 100.0 * rate));
    let non_phrase = report
        .get("false_reactions")
        .and_then(|section| section.get("non_phrase_set"));
    figure_row!(body, "false reactions", [
        fired = non_phrase.and_then(|node| node.get("false_reactions")),
        of = non_phrase.and_then(|node| node.get("of")),
        rate = non_phrase.and_then(|node| node.get("rate")),
    ] => format!("{fired}/{of} ({:.1} %)", 100.0 * rate));
    let real = report
        .get("false_reactions")
        .and_then(|section| section.get("real_audio"));
    let fa = real.and_then(|section| section.get("fa"));
    let feed = real.and_then(|section| section.get("feed"));
    let hours = |key: &str| feed.and_then(|node| node.get(key)).and_then(Value::as_f64);
    // Nothing is formatted for a row the report cannot state: without an event
    // count the row names its absence.
    match fa
        .and_then(|node| node.get("events"))
        .and_then(Value::as_u64)
    {
        Some(events) => {
            let cells: Vec<String> = [
                (
                    "fa_per_hour_vad_active",
                    "VAD-active speech",
                    "vad_active_hours_fed",
                ),
                ("fa_per_hour_audio", "wall-clock audio", "audio_hours_fed"),
            ]
            .into_iter()
            .map(|(key, label, hours_key)| {
                match fa.and_then(|node| node.get(key)).and_then(Value::as_f64) {
                    Some(rate) => match hours(hours_key) {
                        Some(hours) => format!("{rate:.4}/h over {label} ({hours:.3} h)"),
                        None => format!("{rate:.4}/h over {label}"),
                    },
                    // A rate the report cannot divide names the denominator
                    // that is missing rather than asserting a cause.
                    None if hours(hours_key).is_some_and(|hours| hours <= 0.0) => {
                        format!("no rate: {label} measured no time")
                    }
                    None => format!("no rate: {label} not in the report"),
                }
            })
            .collect();
            let ladder = hours("vad_active_hours_fed")
                .filter(|hours| *hours > 0.0)
                .map(|hours| format!(" · one event = {:.4}/h", 1.0 / hours))
                .unwrap_or_default();
            row(
                &mut body,
                "false accepts",
                Some(format!("{events} events · {}{ladder}", cells.join(" · "))),
            );
        }
        None => row(&mut body, "false accepts", None),
    }
    // The set the run priced, in the vocabulary `--wake-thresholds` reads, so the
    // point can be replayed from the record by copy-paste.
    row(
        &mut body,
        "thresholds",
        report
            .get("thresholds")
            .and_then(Value::as_str)
            .map(|line| format!("`{line}`")),
    );
    let calibration = report.get("calibration");
    let derived = |key: &str| calibration.and_then(|node| node.get(key));
    figure_row!(body, "derived point", [
        floor = derived("derived_soft_floor"),
        fire = derived("fire_factor"),
        bar = derived("static_match_threshold"),
        reset = derived("reset_soft"),
    ] => format!("floor {floor:.4} · fire factor {fire:.4} · static bar {bar:.4} · reset {reset:.4}"));
    figure_row!(body, "wall clock", [
        secs = report.get("wall_clock_secs"),
    ] => format!("{secs:.1} s"));
    figure_row!(body, "peak resident", [
        bytes = report.get("peak_rss_bytes"),
    ] => format!("{:.0} MiB", bytes / (1024.0 * 1024.0)));
    let corpus = report.get("corpus");
    figure_row!(body, "corpus", [
        reused = corpus.and_then(|node| node.get("reused")),
        clips = corpus.and_then(|node| node.get("clips")),
        synthesized = corpus.and_then(|node| node.get("synthesized")),
        secs = corpus.and_then(|node| node.get("synthesis_secs")),
    ] => format!("{reused} of {clips} clips reused, {synthesized} synthesized in {secs:.1} s"));
    figure_row!(body, "windows per encoder pass", [
        most = report.get("windows_per_pass"),
        capacity = report.get("group_capacity"),
        partial = report.get("partial_passes"),
        passes = report.get("encoder_passes"),
    ] => format!("at most {most} of {capacity} · {partial} partial of {passes} passes"));
    body.push('\n');
    body.push_str("| phase | seconds |\n|---|---|\n");
    if let Some(phases) = report.get("phases").and_then(Value::as_object) {
        for (name, secs) in phases {
            row(
                &mut body,
                name,
                secs.as_f64().map(|secs| format!("{secs:.1}")),
            );
        }
    }
    body.push_str("\nProvenance:\n\n");
    body.push_str(&format!("- machine: {}\n", measurements::machine()));
    body.push_str(&format!("- device: {}\n", measurements::device()));
    body.push_str(&format!("- profile: {}\n", measurements::profile()));
    body.push_str(&format!("- code: {}\n", measurements::code_state()));
    body.push_str(&format!(
        "- command: `{}`\n",
        report
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or(UNAVAILABLE)
    ));
    let models = report.get("models");
    let pin = |section: &str, key: Option<&str>| -> Option<&Value> {
        let node = models?.get(section)?;
        key.map_or(Some(node), |key| node.get(key))
    };
    for (role, value) in [
        ("ASR body", pin("recognition", Some("body"))),
        ("ASR sound", pin("recognition", Some("sound"))),
        ("TTS body", pin("synthesis", Some("body"))),
        ("TTS sound", pin("synthesis", Some("sound"))),
        ("voice", pin("voice_wav", None)),
    ] {
        // One behaviour for an absent pin: the bullet is written and names the
        // absence, whether the report has no pin or a pin without the figure.
        let Some(pin) = value else {
            body.push_str(&format!("- {role}: {UNAVAILABLE}\n"));
            continue;
        };
        let bytes = pin
            .get("bytes")
            .and_then(Value::as_u64)
            .map(|bytes| format!("{bytes} bytes"))
            .unwrap_or_else(|| UNAVAILABLE.to_string());
        body.push_str(&format!(
            "- {role}: `{}` · {bytes} · sha256 {}\n",
            pin.get("file")
                .and_then(Value::as_str)
                .unwrap_or(UNAVAILABLE),
            pin.get("sha256")
                .and_then(Value::as_str)
                .unwrap_or(UNAVAILABLE)
        ));
    }
    body
}

/// What the run has measured so far. The complete and the stopped report are
/// both built from this, so a phase cannot appear in one shape and not the
/// other.
struct Progress {
    /// The phase the run is inside, for a stop or a failure to name.
    current: &'static str,
    inputs: Option<config::Inputs>,
    enrolment: Option<EnrollStats>,
    calibration: Option<wake::Calibration>,
    /// The threshold set the enrolment scores with — the point this run prices.
    thresholds: Option<wake::Thresholds>,
    recall: Option<Recall>,
    non_phrase: Option<NonPhrase>,
    real: Option<real_audio::RealAudio>,
    /// The corpus cache's counters, updated by [`Synth`] through this same lock:
    /// a report written mid-synthesis counts the clips that phase has already
    /// cached instead of the zero it started from.
    synthesis: audio::SynthStats,
    phases: std::collections::BTreeMap<&'static str, f64>,
}

impl Progress {
    /// The point this run prices: the enrolment's own set once it exists, the
    /// requested one before that.
    fn point(&self, cfg: &BenchConfig) -> wake::Thresholds {
        self.thresholds.unwrap_or(cfg.thresholds)
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            current: "inputs",
            inputs: None,
            enrolment: None,
            calibration: None,
            thresholds: None,
            recall: None,
            non_phrase: None,
            real: None,
            synthesis: audio::SynthStats::default(),
            phases: std::collections::BTreeMap::new(),
        }
    }
}

struct Recall {
    detected: usize,
    of: usize,
    first_pass: bool,
    missed: Vec<String>,
}

struct NonPhrase {
    fired: Vec<String>,
    of: usize,
}

/// Run one phase, recording its wall time.
fn phase<T>(
    progress: &Arc<Mutex<Progress>>,
    name: &'static str,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    state(progress).current = name;
    let started = Instant::now();
    let value = work();
    // Recorded even when the phase fails: a partial report that omits the time
    // a failed phase spent under-reports the run.
    state(progress)
        .phases
        .insert(name, started.elapsed().as_secs_f64());
    value
}

/// The measurement, phase by phase. Each phase marks itself current, so a stop
/// names where it happened, and every long phase checks the stop at its own work
/// item — a synthesized clip, a detection chunk, a corpus file — on top of the
/// guard thread's own watch.
fn measure(
    cfg: &BenchConfig,
    deadline: Deadline,
    ceiling: MemoryCeiling,
    progress: &Arc<Mutex<Progress>>,
) -> Result<()> {
    let inputs = phase(progress, "inputs", || {
        deadline.check("inputs")?;
        let inputs = config::pin_inputs(cfg)?;
        // The corpus is verified here, before the synthesis it would otherwise
        // follow: a bad pin stops the run in its first seconds.
        real_audio::verify(&cfg.corpus, sets::REAL_AUDIO_FILES)
            .context("the pinned real-audio corpus")?;
        Ok(inputs)
    })?;
    std::fs::create_dir_all(&cfg.cache).with_context(|| {
        format!(
            "cannot create the corpus cache {} — the synthesized corpus is the run's durable \
             artifact, so the run stops rather than re-synthesizing it silently",
            cfg.cache.display()
        )
    })?;
    audio::clean_cache(&cfg.cache);

    let asr = Arc::new(
        Pair::open(&cfg.asr_body, &cfg.asr_sound).context("cannot open the recognition pair")?,
    );
    let tts = Arc::new(
        Pair::open(&cfg.tts_body, &cfg.tts_sound).context("cannot open the synthesis pair")?,
    );
    let synthesis = Synthesis::new(tts, &cfg.voice).context("cannot build the synthesis path")?;
    let mut synth = Synth::new(
        synthesis,
        cfg.cache.clone(),
        inputs.synthesis_hash(),
        Arc::clone(progress),
        deadline,
        ceiling,
    );
    let mut listener =
        listen::Listener::new(Arc::clone(&asr)).context("cannot build the listening path")?;
    state(progress).inputs = Some(inputs);

    // ── Enrolment: conditioned clips, the consistency gate, and the negative
    // pool that no measurement clip belongs to. ──
    let (enrollment, stats) = phase(progress, "enrolment", || {
        deadline.check("enrolment")?;
        ceiling.check("enrolment")?;
        enroll(&mut synth, &mut listener, cfg.thresholds)
    })?;
    let calibration = enrollment.calibration.clone();
    // The set the enrolment scores with, not the requested one: every figure the
    // report derives is read from this same set.
    let thresholds = enrollment.thresholds;
    let geometry = enrollment.geometry();
    eprintln!(
        "[wake-word] enrolment: {} of {} clips, {} utterances, neg_mean {:.4}, floor {:.4}, \
         fire factor {:.4}, static threshold {:.4}, reset {:.4}",
        stats.configured - stats.skipped.len(),
        stats.configured,
        stats.utterances,
        calibration.neg_mean,
        calibration.soft_floor(&thresholds),
        calibration.fire_soft(&thresholds),
        geometry.static_threshold,
        geometry.reset
    );
    {
        let mut p = state(progress);
        p.enrolment = Some(stats);
        p.calibration = Some(calibration);
        p.thresholds = Some(thresholds);
    }

    // ── Recall: one warm continuous stream over the 50 phrase clips. ──
    let recall = if cfg.phases.recall {
        let clips = phase(progress, "recall synthesis", || {
            audio::recall_clips(&mut synth)
        })?;
        let recall = phase(progress, "recall", || {
            deadline.check("recall")?;
            ceiling.check("recall")?;
            let (detected, first_pass, missed) =
                measure_recognition(&enrollment, &mut listener, &clips, deadline, ceiling)?;
            Ok(Recall {
                detected,
                of: clips.len(),
                first_pass,
                missed,
            })
        })?;
        if !recall.missed.is_empty() {
            eprintln!(
                "[wake-word] missed recall clips: {}",
                recall.missed.join(", ")
            );
        }
        eprintln!("[wake-word] recognition {}/{}", recall.detected, recall.of);
        Some(recall)
    } else {
        None
    };

    // ── False reactions: one warm continuous stream over the 121 clips. ──
    let non_phrase = if cfg.phases.non_phrase {
        let clips = phase(progress, "non-phrase synthesis", || {
            audio::non_phrase_clips(&mut synth)
        })?;
        let non_phrase = phase(progress, "non-phrase", || {
            deadline.check("non-phrase")?;
            ceiling.check("non-phrase")?;
            let fired =
                measure_false_reactions(&enrollment, &mut listener, &clips, deadline, ceiling)?;
            Ok(NonPhrase {
                fired,
                of: clips.len(),
            })
        })?;
        eprintln!(
            "[wake-word] false reactions {}/{}",
            non_phrase.fired.len(),
            non_phrase.of
        );
        Some(non_phrase)
    } else {
        None
    };
    {
        let mut p = state(progress);
        p.recall = recall;
        p.non_phrase = non_phrase;
    }

    // The synthesis side is done: release its mapped pair and graphs before the
    // phase whose memory cliff this run is bounded for. The real-audio stream
    // compiles its own listener.
    drop(listener);
    drop(synth);

    if !cfg.phases.real_audio {
        return Ok(());
    }
    // ── False accepts per hour: the pinned corpus through one stream. ──
    phase(progress, "real audio", || {
        deadline.check("real audio")?;
        ceiling.check("real audio")?;
        let cut = real_audio::run(cfg, &asr, &enrollment, deadline, ceiling);
        if !cut.audio.failures.is_empty() {
            eprintln!("[wake-word] real audio: {}", cut.audio.failures.join("; "));
        }
        // Stored before the error is propagated: a phase the guard stopped still
        // leaves the files, hours and events it measured.
        state(progress).real = Some(cut.audio);
        match cut.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })?;
    Ok(())
}

/// The whole report in one place. A stopped run gets the same shape with only
/// the phases it completed, so an incomplete run cannot borrow a rate from a
/// phase that never ran.
fn report_json(
    p: &Progress,
    status: &str,
    wall_clock_secs: f64,
    cfg: &BenchConfig,
    abort: Option<(&str, &str)>,
) -> Value {
    let mut report = serde_json::Map::new();
    report.insert("benchmark".into(), json!("wake_word"));
    report.insert("status".into(), json!(status));
    report.insert("wake_phrase".into(), json!(sets::WAKE_PHRASE));
    report.insert("timeout_secs".into(), json!(cfg.timeout_secs));
    report.insert("max_rss_mib".into(), json!(cfg.max_rss_mib));
    // The operating point is stated whether or not a phase got far enough to
    // derive anything, in the vocabulary `--wake-thresholds` reads, and so is the
    // invocation that produced it.
    report.insert("thresholds".into(), json!(p.point(cfg).to_string()));
    report.insert("command".into(), json!(cfg.command_line()));
    if let Some((phase, message)) = abort {
        report.insert("aborted_in".into(), json!(phase));
        report.insert("message".into(), json!(message));
    }
    if let Some(inputs) = &p.inputs {
        report.insert("models".into(), inputs.json());
    }
    if let Some(stats) = &p.enrolment {
        report.insert(
            "enrollment".into(),
            json!({
                "configured_clips": stats.configured,
                "utterances": stats.utterances,
                "skipped_clips": stats.skipped,
                "skipped_negatives": stats.skipped_negatives,
                "note": "A conditioned render with no VAD-positive speech is skipped and named; \
                         the consistency gate and the five-utterance minimum decide whether the \
                         remainder is an enrolment. A skipped negative shrinks the calibration \
                         pool, so it is named here too.",
            }),
        );
    }
    if let (Some(calibration), Some(thresholds)) = (&p.calibration, &p.thresholds) {
        let geometry = wake::DetectionGeometry::for_calibration(calibration, thresholds);
        report.insert(
            "calibration".into(),
            json!({
                "neg_mean": calibration.neg_mean,
                "neg_std": calibration.neg_std,
                "neg_p99": calibration.neg_p99,
                "n_negatives": calibration.n_negatives,
                "derived_soft_floor": calibration.soft_floor(thresholds),
                "fire_factor": calibration.fire_soft(thresholds),
                "static_match_threshold": geometry.static_threshold,
                "reset_soft": geometry.reset,
                "clip_seed_bands": {
                    "owner_negatives": {"base": sets::OWNER_NEGATIVE_SEED_BASE, "per_phrase": sets::OWNER_NEGATIVE_SEEDS_PER_PHRASE, "phrases": sets::OWNER_NEGATIVE_PHRASES.len()},
                    "ambient_noise": {"base": sets::CALIBRATION_NOISE_SEED_BASE, "profiles": sets::NOISE_PROFILES.len(), "levels": ["full", "-6 dB"]},
                    "confusable": {"base": sets::CONFUSABLE_SEED_BASE, "per_phrase": sets::CONFUSABLE_SEEDS_PER_PHRASE, "phrases": sets::CONFUSABLE_PHRASES.len()},
                    "unrelated": {"base": sets::UNRELATED_SEED_BASE, "per_phrase": sets::UNRELATED_SEEDS_PER_PHRASE, "phrases": sets::UNRELATED_PHRASES.len()},
                },
                "noise_seed_bands": {
                    "calibration_base": sets::CALIBRATION_NOISE_SEED_BASE,
                    "detection_base": sets::DETECTION_NOISE_SEED_BASE,
                    "note": "A profile name renders at 4242+index for calibration and at 42+index \
                             (the original constants) for the detection set; no waveform is both \
                             calibration material and measurement material.",
                },
                "note": "Derived for the models that ran — thresholds do not transfer between models.",
            }),
        );
    }
    if let Some(recall) = &p.recall
        && recall.of > 0
    {
        report.insert(
            "recognition".into(),
            json!({
                "detected": recall.detected,
                "of": recall.of,
                "rate": recall.detected as f64 / recall.of as f64,
                "single_voice": true,
                "first_pass_detected": recall.first_pass,
                "missed": recall.missed,
                "clip_seed_bands": {
                    "wake_only": {"base": sets::HELD_OUT_SEED_BASE, "clips": sets::HELD_OUT_WAKE_ONLY_CLIPS},
                    "command_plus_speech": {"base": sets::COMMAND_PLUS_SPEECH_SEED_BASE, "clips": sets::COMMAND_PLUS_SPEECH_PHRASES.len()},
                },
                "composition": recognition_composition(recall.of),
            }),
        );
    }
    // Each measurement states only what it measured: the synthetic set and the
    // real-audio stream are independent, so a real-audio-only screening still
    // reports its false accepts, and a phase that did not finish is absent,
    // never a null.
    let mut false_reactions = serde_json::Map::new();
    if let Some(non_phrase) = &p.non_phrase
        && non_phrase.of > 0
    {
        false_reactions.insert(
            "non_phrase_set".into(),
            json!({
                "false_reactions": non_phrase.fired.len(),
                "of": non_phrase.of,
                "rate": non_phrase.fired.len() as f64 / non_phrase.of as f64,
                "fired": non_phrase.fired,
                "single_voice": true,
                "clip_seed_bands": {
                    "confusable": [sets::CONFUSABLE_BANDS[0].0, sets::CONFUSABLE_BANDS[1].0],
                    "unrelated": [sets::UNRELATED_BANDS[0].0, sets::UNRELATED_BANDS[1].0],
                    "phrase_in_passing": sets::PHRASE_IN_PASSING_SEED_BASE,
                    "detection_noise": sets::DETECTION_NOISE_SEED_BASE,
                },
                "composition": non_phrase_composition(),
            }),
        );
    }
    if let Some(real) = &p.real {
        false_reactions.insert("real_audio".into(), real.json());
    }
    if !false_reactions.is_empty() {
        report.insert("false_reactions".into(), Value::Object(false_reactions));
    }
    if let Some(real) = &p.real {
        let mut coverage = serde_json::Map::new();
        if let Some(recall) = &p.recall {
            coverage.insert("phrase_utterances".into(), json!(recall.of));
        }
        if let Some(non_phrase) = &p.non_phrase {
            coverage.insert("non_phrase_set".into(), json!(non_phrase.of));
        }
        coverage.insert("real_audio_audio_hours".into(), json!(real.audio_hours));
        coverage.insert("real_audio_speech_hours".into(), json!(real.speech_hours));
        coverage.insert("real_audio_noise_hours".into(), json!(real.noise_hours));
        report.insert("coverage".into(), Value::Object(coverage));
    }
    let synthesis = p.synthesis;
    report.insert(
        "corpus".into(),
        json!({
            "cache": cfg.cache.display().to_string(),
            "clips": synthesis.clips,
            "reused": synthesis.cache_hits,
            "synthesized": synthesis.clips - synthesis.cache_hits,
            "synthesis_secs": synthesis.synth_secs,
            "note": "The corpus is generated once, into this fixed cache directory, and reused \
                     by every later run: the key covers the text, the seed, the sample rate, both \
                     TTS files, the voice reference and the synthesis code (every code line of \
                     the modules that produce a clip, so a comment edit does not invalidate it), \
                     so a clip is re-synthesized only when one of those changes. Every clip is \
                     the one enrolled voice; there is no multi-voice rotation and no \
                     cold-versus-warm comparison.",
        }),
    );
    report.insert("phases".into(), json!(p.phases));
    // The size the stream actually resolved windows at, against the capacity the
    // pipeline was built for: a phase cut short may never fill a group, and a
    // slice boundary leaves the last group of a file short. Absent when the
    // real-audio phase did not run — a zero would read as "measured nothing".
    report.insert("group_capacity".into(), json!(listen::GROUP));
    if let Some(real) = &p.real {
        report.insert("windows_per_pass".into(), json!(real.max_group));
        report.insert("encoder_passes".into(), json!(real.encoder_passes));
        report.insert("partial_passes".into(), json!(real.partial_passes));
    }
    report.insert("wall_clock_secs".into(), json!(wall_clock_secs));
    report.insert("peak_rss_bytes".into(), json!(crate::report::peak_rss()));
    report.insert(
        "vad_gate".into(),
        json!({
            "retained": true,
            "hours_basis": "VAD-active speech hours",
            "carry_across_clips": true,
            "note": "Every score is computed on the VAD-gated speech window; the real-audio \
                     denominator is the VAD-active time the same detector marked over the one \
                     stream. The detector instance carries across files. Recall and false \
                     reactions run ungrouped, one window per encoder pass, so their decisions \
                     do not depend on a group boundary; the bound the sweep's grouping adds is \
                     stated with its rate in `false_reactions.real_audio.fa.basis_note`.",
        }),
    );
    Value::Object(report)
}

/// The non-phrase set's composition, in counts: the one place its shape is
/// stated, so the report cannot drift into three descriptions of the same set.
/// The total is the pinned constant the builder asserts against, not the count
/// this run happened to produce.
fn non_phrase_composition() -> Value {
    json!({
        "confusable_clips": sets::CONFUSABLE_BANDS.len() * sets::CONFUSABLE_PHRASES.len(),
        "unrelated_clips": sets::UNRELATED_BANDS.len() * sets::UNRELATED_PHRASES.len(),
        "silence_clips": sets::SILENCE_DURATIONS.len(),
        "noise_clips": sets::NOISE_PROFILES.len() + sets::NOISE_PROFILES_DETECTION_ONLY.len(),
        "phrase_in_passing_clips": sets::PHRASE_IN_PASSING_PHRASES.len(),
        "total": sets::EXPECTED_NON_PHRASE_TOTAL,
    })
}

/// The recall set's composition, in counts.
fn recognition_composition(total: usize) -> Value {
    json!({
        "wake_only_clips": sets::HELD_OUT_WAKE_ONLY_CLIPS,
        "command_plus_speech_clips": sets::COMMAND_PLUS_SPEECH_PHRASES.len(),
        "total": total,
    })
}

/// What the enrolment used: the clips that gated speech, the ones a quiet
/// render left below the gate, and the negatives a quiet render lost.
struct EnrollStats {
    configured: usize,
    utterances: usize,
    skipped: Vec<String>,
    skipped_negatives: Vec<String>,
}

/// The enrolment: ten conditioned clips, each VAD-gated through a **fresh**
/// detector, embedded, gated on consistency, and calibrated against a negative
/// pool built from disjoint clips.
///
/// A clip whose conditioned render carries no VAD-positive speech is skipped and
/// named: a quiet rendering is a property of the model and the seed, not a
/// reason to abandon the run. The consistency gate and
/// [`wake::MIN_ENROLLMENT_UTTERANCES`] are what keep a half-empty enrolment from
/// producing a threshold, and every skipped negative is reported so a shrunken
/// calibration pool cannot look like a full one.
fn enroll(
    synth: &mut Synth,
    listener: &mut listen::Listener,
    thresholds: wake::Thresholds,
) -> Result<(Enrollment, EnrollStats)> {
    let clips = audio::enrollment_clips(synth)?;
    let mut utterances = Vec::with_capacity(clips.len());
    let mut skipped = Vec::new();
    for (samples, label) in &clips {
        let speech = detect::vad_gate(samples, &mut earshot::Detector::default());
        if speech.is_empty() {
            eprintln!("[wake-word] enrolment clip `{label}`: no VAD-positive speech — skipped");
            skipped.push(label.clone());
            continue;
        }
        utterances.push(listener.embed(&speech)?);
    }
    let prototype = consistency(&utterances)?;

    // Ambient → owner → confusable → unrelated.
    let mut skipped_negatives = Vec::new();
    let mut negatives = audio::ambient_negatives(listener)?;
    negatives.extend(audio::owner_negatives(
        synth,
        listener,
        &mut skipped_negatives,
    )?);
    negatives.extend(audio::phrase_negatives(
        synth,
        listener,
        sets::CONFUSABLE_PHRASES,
        sets::CONFUSABLE_SEEDS_PER_PHRASE,
        sets::CONFUSABLE_SEED_BASE,
        "confusable",
        None,
        &mut skipped_negatives,
    )?);
    negatives.extend(audio::phrase_negatives(
        synth,
        listener,
        sets::UNRELATED_PHRASES,
        sets::UNRELATED_SEEDS_PER_PHRASE,
        sets::UNRELATED_SEED_BASE,
        "unrelated",
        None,
        &mut skipped_negatives,
    )?);
    audio::require_negatives(&negatives)?;

    let stats = EnrollStats {
        configured: clips.len(),
        utterances: utterances.len(),
        skipped,
        skipped_negatives,
    };
    let calibration = wake::calibrate_negatives(&prototype, &negatives);
    let enrollment = Enrollment::build(
        sets::WAKE_PHRASE.to_string(),
        &utterances,
        calibration,
        &negatives,
    )
    .ok_or_else(|| {
        anyhow!(
            "the enrollment built no prototype from {} utterances",
            utterances.len()
        )
    })?
    .with_thresholds(thresholds)
    .map_err(|why| anyhow!("the threshold set is unusable: {why}"))?;
    Ok((enrollment, stats))
}

/// The enrolment consistency gate: centroid + the fraction of utterances at
/// cosine ≥ [`wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY`] to it, and at least
/// [`wake::MIN_ENROLLMENT_UTTERANCES`] utterances.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn consistency(utterances: &[Vec<f32>]) -> Result<Vec<f32>> {
    if utterances.len() < wake::MIN_ENROLLMENT_UTTERANCES {
        bail!(
            "only {} enrolment utterances (need ≥{}) — the enrolment is too small to gate",
            utterances.len(),
            wake::MIN_ENROLLMENT_UTTERANCES
        );
    }
    let centroid = wake::normalized_centroid(utterances)
        .map_err(|width| anyhow!("an enrolment embedding has width {width}"))?;
    let required =
        (utterances.len() as f32 * wake::ENROLLMENT_CONSISTENCY_MIN_FRACTION).ceil() as usize;
    let passed = utterances
        .iter()
        .filter(|embedding| {
            wake::cosine(embedding, &centroid) >= wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY
        })
        .count();
    if passed < required {
        bail!(
            "only {passed}/{} enrolment utterances match the prototype (need ≥{required} at \
             cosine ≥ {})",
            utterances.len(),
            wake::ENROLLMENT_CONSISTENCY_MIN_SIMILARITY
        );
    }
    Ok(centroid)
}

/// The stream a clip set is measured on: ungrouped, so a group boundary can never
/// move a decision in the two rates reported over clips. Both the recall and the
/// false-reaction phase take their stream from here, which is what
/// `a_clip_set_stream_is_ungrouped` pins.
fn clip_set_stream(enrollment: &Enrollment) -> DetectionStream {
    DetectionStream::warm(enrollment.clone())
}

/// Feed the recall clips through one warm stream, two seconds of silence
/// between them. Returns `(detected, first_pass, missed labels)`. The stop and
/// the resident-set ceiling are checked per clip, on top of the per-chunk check
/// inside [`DetectionStream::feed`].
fn measure_recognition(
    enrollment: &Enrollment,
    listener: &mut listen::Listener,
    clips: &[(Vec<f32>, String)],
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<(usize, bool, Vec<String>)> {
    let mut stream = clip_set_stream(enrollment);
    let gap = vec![0.0f32; sets::FILE_GAP_SAMPLES];
    let mut detected = 0usize;
    let mut first_pass = false;
    let mut missed = Vec::new();
    for (index, (samples, label)) in clips.iter().enumerate() {
        ceiling.check_process("recall")?;
        let hit = feed_clip(&mut stream, listener, samples, &gap, deadline, "recall")?;
        if index == 0 {
            first_pass = hit;
        }
        if hit {
            detected += 1;
        } else {
            missed.push(label.clone());
        }
    }
    Ok((detected, first_pass, missed))
}

/// Feed the non-phrase clips through one warm stream. Returns the labels that
/// fired — every one is a false reaction.
fn measure_false_reactions(
    enrollment: &Enrollment,
    listener: &mut listen::Listener,
    clips: &[(Vec<f32>, String)],
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<Vec<String>> {
    let mut stream = clip_set_stream(enrollment);
    let gap = vec![0.0f32; sets::FILE_GAP_SAMPLES];
    let mut fired = Vec::new();
    for (samples, label) in clips {
        ceiling.check_process("non-phrase")?;
        if feed_clip(&mut stream, listener, samples, &gap, deadline, "non-phrase")? {
            fired.push(label.clone());
        }
    }
    Ok(fired)
}

/// One clip's turn in a continuous stream.
///
/// A fire during the clip's own samples is a detection of that clip, and so is
/// a fire during the trailing silence while that clip's segment is still open
/// — the trailing-silence flush, which is how a match on the phrase's last
/// frames is counted. Once the segment boundary closes no further fire is
/// possible; the rest of the two-second gap only re-arms the stream.
fn feed_clip(
    stream: &mut DetectionStream,
    listener: &mut listen::Listener,
    samples: &[f32],
    gap: &[f32],
    deadline: Deadline,
    phase: &'static str,
) -> Result<bool> {
    let mut hit = feed_fired(stream, samples, listener, deadline, phase)?;
    let mut remaining = gap;
    while !remaining.is_empty() {
        let take = remaining.len().min(detect::FRAME_LENGTH);
        let (chunk, rest) = remaining.split_at(take);
        if stream.segment_open() {
            hit |= feed_fired(stream, chunk, listener, deadline, phase)?;
        } else {
            stream.feed(remaining, listener, deadline, phase)?;
            break;
        }
        remaining = rest;
    }
    Ok(hit)
}

/// Feed one slice and report whether it fired, whether or not the slice ran to
/// its end.
fn feed_fired(
    stream: &mut DetectionStream,
    samples: &[f32],
    listener: &mut listen::Listener,
    deadline: Deadline,
    phase: &'static str,
) -> Result<bool> {
    let fed = stream.feed(samples, listener, deadline, phase);
    let fires = stream.take_fires();
    fed.map(|()| fires > 0)
}

/// The honesty floor: a rate the run cannot divide is an error, not a report.
///
/// The clip lists and the file feed are pinned sizes and a failed synthesis or
/// an empty feed already aborts, so what this guards is the one denominator the
/// report divides by: VAD-active speech time. A rate with no denominator is
/// never written.
fn require_measured(p: &Progress, phases: config::Phases) -> Result<()> {
    if !phases.real_audio {
        return Ok(());
    }
    let real = p
        .real
        .as_ref()
        .ok_or_else(|| anyhow!("the real-audio phase did not complete — no false-accept rate"))?;
    if !real.vad_active_hours.is_finite() || real.vad_active_hours <= 0.0 {
        bail!(
            "the real-audio phase fed {} file(s) but the VAD marked no speech — the \
             false-accept rate has no denominator",
            real.files_fed
        );
    }
    Ok(())
}

/// Write the report only after the whole measurement completed: a temporary
/// file beside the target, then a rename.
fn write_report(path: &Path, report: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(report).context("cannot serialise the report")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    if let Err(error) = std::fs::write(&tmp, text.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("cannot write {}", tmp.display()));
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error)
            .with_context(|| format!("cannot rename {} to {}", tmp.display(), path.display()));
    }
    Ok(())
}

/// The progress lock, recovering from a poisoned mutex: a panic elsewhere must
/// not take the guard's own reading, or the report, down with it.
fn state<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The run's report, written at most once: one lock, held across the claim and
/// the write, and set only after the write succeeds — so a failed write (a full
/// disk beside the cache, say) leaves the claim free and the stop keeps watching.
#[derive(Default)]
struct Report {
    written: Mutex<bool>,
}

impl Report {
    /// `Ok(Some(report))` is the JSON this call wrote; `Ok(None)` means an
    /// earlier call had already written it; `Err` means it is not on disk, and
    /// the caller is expected to say so.
    fn write(
        &self,
        cfg: &BenchConfig,
        progress: &Mutex<Progress>,
        started: Instant,
        status: &str,
        stop: Option<(&str, &str)>,
    ) -> Result<Option<Value>> {
        let mut written = state(&self.written);
        if *written {
            return Ok(None);
        }
        let report = report_json(
            &state(progress),
            status,
            started.elapsed().as_secs_f64(),
            cfg,
            stop,
        );
        write_report(&cfg.report, &report)?;
        *written = true;
        Ok(Some(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir =
            std::env::temp_dir().join(format!("mahbot-wake-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn test_config() -> BenchConfig {
        BenchConfig {
            models: Some(std::path::PathBuf::from("/models")),
            voice: std::path::PathBuf::from("/voice.wav"),
            corpus: std::path::PathBuf::from("/corpus"),
            report: std::path::PathBuf::from("/report.json"),
            cache: std::path::PathBuf::from("/cache"),
            lock: std::path::PathBuf::from("/report.lock"),
            timeout_secs: 900,
            max_rss_mib: 6_144,
            asr_body: std::path::PathBuf::from("/models/asr.gguf"),
            asr_sound: std::path::PathBuf::from("/models/asr-mmproj.gguf"),
            tts_body: std::path::PathBuf::from("/models/tts.gguf"),
            tts_sound: std::path::PathBuf::from("/models/tts-mmproj.gguf"),
            thresholds: wake::Thresholds::DEFAULT,
            phases: config::Phases::ALL,
        }
    }

    /// The clip sets are measured ungrouped: both phases take their stream from
    /// `clip_set_stream`, so a phase that switched to the grouped path — or a
    /// change to the helper — would move a decision at a group boundary and fail
    /// here. The sweep's own stream is the one that groups.
    #[test]
    fn a_clip_set_stream_is_ungrouped() {
        use crate::wake::{Calibration, Enrollment};
        let utterances = vec![vec![1.0f32, 0.0]; crate::wake::MIN_ENROLLMENT_UTTERANCES];
        let enrollment = Enrollment::build("wake".into(), &utterances, Calibration::default(), &[])
            .expect("an enrolment of identical utterances");
        assert_eq!(clip_set_stream(&enrollment).group(), 1);
        assert_eq!(DetectionStream::grouped(enrollment).group(), listen::GROUP);
    }

    /// A run stopped before a phase measured anything must not carry that
    /// phase's rate at all — an absent section, never a null or a zero.
    #[test]
    fn an_aborted_run_omits_the_phases_it_did_not_measure() {
        let report = report_json(
            &Progress::default(),
            "aborted",
            1.0,
            &test_config(),
            Some(("synthesis", "stop")),
        );
        assert_eq!(report["status"], "aborted");
        assert_eq!(report["aborted_in"], "synthesis");
        assert!(report.get("recognition").is_none());
        assert!(report.get("false_reactions").is_none());
        assert_no_null_rates(&report, "$");
    }

    /// Measured phases carry finite rates, and no key named `...rate...` is
    /// ever null.
    #[test]
    fn a_measured_run_carries_finite_rates() {
        let progress = Progress {
            recall: Some(Recall {
                detected: 3,
                of: 4,
                first_pass: true,
                missed: vec!["clip".into()],
            }),
            non_phrase: Some(NonPhrase {
                fired: Vec::new(),
                of: 121,
            }),
            ..Progress::default()
        };
        let report = report_json(&progress, "complete", 1.0, &test_config(), None);
        assert_eq!(report["recognition"]["rate"], 0.75);
        assert_eq!(report["false_reactions"]["non_phrase_set"]["rate"], 0.0);
        assert_no_null_rates(&report, "$");
    }

    /// The two measurement aids are a classification, not a different report: a
    /// run that priced another threshold set, or a subset of the phases, is
    /// `screened` — the status the record gate and the console both read, even
    /// when the run took all three phases.
    #[test]
    fn a_run_that_prices_a_candidate_is_screened() {
        let benchmark = test_config();
        assert_eq!(classify(&benchmark, &Ok(()), None, "inputs").0, "complete");

        let mut overridden = test_config();
        overridden.thresholds.anchor = 0.86;
        assert_eq!(
            classify(&overridden, &Ok(()), None, "inputs").0,
            "screened",
            "another threshold set is a candidate point"
        );

        let mut subset = test_config();
        subset.phases = config::Phases {
            recall: true,
            non_phrase: true,
            real_audio: false,
        };
        assert_eq!(
            classify(&subset, &Ok(()), None, "inputs").0,
            "screened",
            "a subset of the phases is a candidate point"
        );

        let error = anyhow!("no denominator");
        let (status, stop) = classify(&benchmark, &Ok(()), Some(&error), "real audio");
        assert_eq!(status, "incomplete");
        assert_eq!(stop.expect("named").0, "real audio");
    }

    /// The point a run priced is stated whether or not a phase got far enough to
    /// derive anything: the set and the invocation are run inputs, like the stop
    /// and the ceiling.
    #[test]
    fn a_run_that_stopped_before_enrolment_still_states_its_point() {
        let mut cfg = test_config();
        cfg.thresholds.anchor = 0.86;
        let report = report_json(
            &Progress::default(),
            "aborted",
            1.0,
            &cfg,
            Some(("inputs", "stop")),
        );
        assert_eq!(report["thresholds"], cfg.thresholds.to_string(), "{report}");
        assert_ne!(
            report["thresholds"],
            wake::Thresholds::DEFAULT.to_string(),
            "the requested point is the one stated before a derivation exists"
        );
        let command = report["command"].as_str().expect("a command line");
        assert!(command.contains("--wake-thresholds"), "{command}");
    }

    /// The real-audio phase is independent of the synthetic set: a run that took
    /// only the real audio still states its false accepts, in the JSON and not
    /// only on the console.
    #[test]
    fn a_real_audio_only_run_states_its_false_accepts() {
        let progress = Progress {
            real: Some(measured_real(1.066, 0.807)),
            ..Progress::default()
        };
        let report = report_json(&progress, "screened", 1.0, &test_config(), None);
        assert!(
            report["false_reactions"].get("non_phrase_set").is_none(),
            "{report}"
        );
        assert_eq!(report["false_reactions"]["real_audio"]["fa"]["events"], 2);
        assert_eq!(
            report["false_reactions"]["real_audio"]["fa"]["fa_per_hour_vad_active"],
            2.0 / 0.807
        );
        // Coverage names the measurements it has, not the ones it does not.
        assert!(
            report["coverage"].get("phrase_utterances").is_none(),
            "{report}"
        );
        assert!(report["coverage"].get("real_audio_audio_hours").is_some());
        assert_no_null_rates(&report, "$");
    }

    /// A figure the report does not carry is named as unavailable, in the same
    /// words every row uses — never printed as a zero the run did not measure.
    #[test]
    fn a_figure_the_report_lacks_is_named_not_zeroed() {
        let body = record_body(&json!({}));
        assert!(body.contains(&format!("status: `{UNAVAILABLE}`")), "{body}");
        for role in ["ASR body", "ASR sound", "TTS body", "TTS sound", "voice"] {
            assert!(
                body.contains(&format!("- {role}: {UNAVAILABLE}")),
                "the {role} pin does not name its absence:\n{body}"
            );
        }
        for row in [
            "recall",
            "false reactions",
            "false accepts",
            "thresholds",
            "derived point",
            "wall clock",
            "peak resident",
            "corpus",
            "windows per encoder pass",
        ] {
            assert!(
                body.contains(&format!("| {row} | {UNAVAILABLE} |")),
                "the {row} row does not name its missing figure:\n{body}"
            );
        }
    }

    /// A resolved input set: five pins with a byte count and a digest, so the
    /// record's provenance block has the figures a real run states.
    fn test_inputs() -> config::Inputs {
        let pin = |name: &str| config::Pin {
            path: std::path::PathBuf::from(format!("/models/{name}")),
            bytes: 1,
            sha256: "00".into(),
        };
        config::Inputs {
            asr_body: pin("asr.gguf"),
            asr_sound: pin("asr-mmproj.gguf"),
            tts_body: pin("tts.gguf"),
            tts_sound: pin("tts-mmproj.gguf"),
            voice: pin("voice.wav"),
        }
    }

    /// The same writer over a measured report states every figure: the
    /// unavailable words appear only where the report has no number.
    #[test]
    fn a_measured_report_states_every_figure() {
        let progress = Progress {
            inputs: Some(test_inputs()),
            recall: Some(Recall {
                detected: 46,
                of: 50,
                first_pass: false,
                missed: Vec::new(),
            }),
            non_phrase: Some(NonPhrase {
                fired: vec!["clip".into()],
                of: 121,
            }),
            real: Some(measured_real(1.066, 0.807)),
            calibration: Some(wake::Calibration::default()),
            thresholds: Some(wake::Thresholds::DEFAULT),
            synthesis: audio::SynthStats {
                clips: 394,
                cache_hits: 394,
                synth_secs: 0.0,
            },
            ..Progress::default()
        };
        let report = report_json(&progress, "complete", 287.9, &test_config(), None);
        let body = record_body(&report);
        assert!(!body.contains(UNAVAILABLE), "{body}");
        assert!(body.contains("| recall | 46/50 (92.0 %) |"), "{body}");
        assert!(
            body.contains(&format!(
                "| false accepts | 2 events · {:.4}/h over VAD-active speech (0.807 h)",
                2.0 / 0.807
            )),
            "{body}"
        );
        assert!(
            body.contains("| peak resident | ") && body.contains("| wall clock | 287.9 s |"),
            "{body}"
        );
    }

    fn assert_no_null_rates(value: &Value, path: &str) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    let here = format!("{path}.{key}");
                    if key.contains("rate") || key.starts_with("fa_per_hour") {
                        assert!(!child.is_null(), "{here} is null");
                        assert!(
                            child.as_f64().is_some_and(f64::is_finite),
                            "{here} is not a finite number"
                        );
                    }
                    assert_no_null_rates(child, &here);
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    assert_no_null_rates(child, &format!("{path}[{index}]"));
                }
            }
            _ => {}
        }
    }

    #[test]
    fn a_report_is_written_atomically_and_parses_back() {
        let dir = temp_dir("report");
        let path = dir.join("wake_word_report.json");
        let report = json!({"benchmark": "wake_word", "rate": 0.5});

        write_report(&path, &report).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert_eq!(serde_json::from_str::<Value>(&text).expect("parse"), report);

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "a temporary file was left behind");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A measured real-audio phase with the given denominators.
    fn measured_real(audio_hours: f64, vad_active_hours: f64) -> real_audio::RealAudio {
        real_audio::RealAudio {
            files_total: 33,
            speech_files: 18,
            noise_files: 15,
            corpus: std::path::PathBuf::from("/corpus"),
            files_fed: 33,
            audio_hours,
            speech_hours: audio_hours,
            noise_hours: 0.0,
            vad_active_hours,
            wall_secs: 1.0,
            max_group: listen::GROUP,
            encoder_passes: 3,
            partial_passes: 1,
            listener_bytes: 1,
            events: 2,
            cut: None,
            failures: Vec::new(),
        }
    }

    /// A write that fails must leave the claim free: the stop keeps watching and
    /// the report can still be written later. A path whose parent is a file is
    /// the cheapest way to make the write fail.
    #[test]
    fn a_failed_write_does_not_spend_the_claim() {
        let dir = temp_dir("report-retry");
        let mut cfg = test_config();
        let progress = Mutex::new(Progress::default());
        let report = Report::default();
        let started = Instant::now();

        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"x").expect("write the blocker");
        cfg.report = blocker.join("wake_word_report.json");
        assert!(
            report
                .write(
                    &cfg,
                    &progress,
                    started,
                    "aborted",
                    Some(("recall", "stop"))
                )
                .is_err(),
            "the write must fail"
        );
        assert!(
            !*state(&report.written),
            "a failed write must not spend the claim"
        );

        cfg.report = dir.join("wake_word_report.json");
        report
            .write(
                &cfg,
                &progress,
                started,
                "aborted",
                Some(("recall", "stop")),
            )
            .expect("the retry writes");
        assert!(cfg.report.exists());
        let text = std::fs::read_to_string(&cfg.report).expect("read");
        assert_eq!(
            serde_json::from_str::<Value>(&text).expect("parse")["status"],
            "aborted"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A real-audio phase the guard stopped still carries what the stream
    /// measured, and says it was cut and why.
    #[test]
    fn a_cut_real_audio_phase_reports_partial_with_its_reason() {
        let mut real = measured_real(1.0, 0.5);
        real.cut = Some("the `real audio` phase exceeded the 60 s stop".into());
        let json = real.json();
        assert_eq!(json["status"], "partial");
        assert_eq!(
            json["feed"]["cut"],
            "the `real audio` phase exceeded the 60 s stop"
        );
        assert_eq!(json["feed"]["files_fed"], 33);
        assert_eq!(json["fa"]["fa_per_hour_vad_active"], 4.0);

        let ran = measured_real(1.0, 0.5).json();
        assert_eq!(ran["status"], "ran");
        assert!(ran["feed"]["cut"].is_null());
    }

    #[test]
    fn a_run_that_measured_no_speech_is_refused() {
        assert!(require_measured(&Progress::default(), config::Phases::ALL).is_err());
        let no_speech = Progress {
            real: Some(measured_real(1.0, 0.0)),
            ..Progress::default()
        };
        assert!(require_measured(&no_speech, config::Phases::ALL).is_err());
        let measured = Progress {
            real: Some(measured_real(1.0, 0.5)),
            ..Progress::default()
        };
        assert!(require_measured(&measured, config::Phases::ALL).is_ok());
    }

    /// A rate whose denominator measured no time is absent, never null.
    #[test]
    fn a_degenerate_false_accept_rate_is_omitted() {
        let fa = measured_real(0.0, 0.0).json();
        let fa = &fa["fa"];
        assert!(fa.get("fa_per_hour_audio").is_none(), "{fa}");
        assert!(fa.get("fa_per_hour_vad_active").is_none(), "{fa}");
        assert_eq!(fa["events"], 2);
        let note = fa["omitted"].as_str().expect("a reason is stated");
        assert!(note.contains("fa_per_hour_vad_active"), "{note}");
        assert_no_null_rates(fa, "$.false_reactions.real_audio.fa");

        let fa = measured_real(1.0, 0.5).json();
        assert_eq!(fa["fa"]["fa_per_hour_audio"], 2.0);
        assert_eq!(fa["fa"]["fa_per_hour_vad_active"], 4.0);
    }

    /// The report states the set's shape once, in counts that add up to the
    /// pinned total.
    #[test]
    fn the_reported_composition_adds_up() {
        let non_phrase = non_phrase_composition();
        let summed = non_phrase["confusable_clips"].as_u64().unwrap()
            + non_phrase["unrelated_clips"].as_u64().unwrap()
            + non_phrase["silence_clips"].as_u64().unwrap()
            + non_phrase["noise_clips"].as_u64().unwrap()
            + non_phrase["phrase_in_passing_clips"].as_u64().unwrap();
        assert_eq!(summed as usize, sets::EXPECTED_NON_PHRASE_TOTAL);
        assert_eq!(
            non_phrase["total"].as_u64().unwrap() as usize,
            sets::EXPECTED_NON_PHRASE_TOTAL
        );

        let recall = recognition_composition(50);
        assert_eq!(
            recall["wake_only_clips"].as_u64().unwrap()
                + recall["command_plus_speech_clips"].as_u64().unwrap(),
            50
        );
    }

    #[test]
    fn an_intolerable_enrolment_is_refused() {
        assert!(consistency(&[]).is_err());
        assert!(consistency(&vec![vec![1.0f32, 0.0]; 4]).is_err());
    }
}
