// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake-word bench: recall, false reactions and false accepts per hour on
//! this crate's own models.
//!
//! One run measures three things:
//!
//! 1. **recall** — 40 bare wake-phrase clips plus 10 wake-phrase-plus-command
//!    clips, in each of eleven voices: the enrolled one and ten clones;
//! 2. **false reactions** — 1 161 clips. Eleven speech streams of 104 clips each
//!    (confusables, unrelated speech, phrase-in-passing), one per voice, plus one
//!    stream of the 17 non-speech clips (silence, noise), which have no voice;
//! 3. **false accepts per hour** — the pinned 33-file real-audio subset, fed in
//!    order through one continuous stream.
//!
//! The enrolment is the one enrolled voice: ten conditioned renderings, a
//! consistency gate, and a negative pool that shares no clip with either
//! measurement set. The ten clones are cut from single-reader corpus recordings
//! the false-accept phase never measures ([`voices`]); they are scored against the
//! enrolled voice's prototype, which is the point of the exercise. Everything the
//! report states was measured on the models the run was given; a run that
//! measured nothing is an error, not a report.
//!
//! # Voices and streams
//!
//! Every speech measurement runs in **one stream per voice**, never one shared
//! stream: the detector's audio-clock cooldown then stays inside the stream that
//! caused it, so every voice's rows are measured under identical scoring
//! conditions, and the enrolled voice's own negatives run in a stream of their
//! own rather than sharing the non-speech clips. The prototype, the calibration
//! pool and the anti-prototypes stay built from the enrolled voice alone.
//!
//! The enrolled voice's row is measured at thresholds fitted on that same voice;
//! the ten clone rows are measured at thresholds fitted on the enrolled voice.
//! That asymmetry is deliberate and the record states it: the clone rows are the
//! generalisation evidence, the enrolled row is not.
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
//! Recall and false reactions run ungrouped, one window per pass: the ungrouped
//! path keeps their decisions free of any group boundary, which grouping would
//! move at the group's edges. The group facts are a property of the run and the
//! report states them.
//!
//! # Guards
//!
//! A run holds an exclusive lock (`--lock` / `MAHBOT_LOCK`, defaulting to the one
//! shared path every heavy run of this crate takes) so a second launch refuses
//! instead of competing for the same device. A measurement carries a wall-clock
//! stop (3 600 s by default) and a memory guard covering the process's resident
//! set, swap growth since the run started and the system's free-memory percentage;
//! the corpus build carries the memory guard alone, because it has no wall-clock
//! bound.
//!
//! The work items check the deadline at every clip read, detection chunk and
//! corpus file, and the cheap process figure at every clip and corpus file;
//! the two-subprocess system figures run at phase and file boundaries. A guard
//! thread watches the wall clock every second and both figures every five, so a
//! stop fires even while the measuring thread is inside one blocked device call.
//! A report is written once, by whichever thread gets there first — a stop or a
//! fatal error leaves what it measured, names the phase it stopped in, and exits
//! non-zero. The corpus counts a stopped run states on its console come from the
//! same lock the guard reads.
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
//! over all three measurements and all eleven voices, inside the default bounds,
//! over a corpus the build left complete, reports `complete`, and rewrites this
//! surface's section of the repository's `MEASUREMENTS.md` (see
//! [`crate::measurements`]); `MAHBOT_MEASUREMENTS` points that write at another
//! file, which is how a run can be checked without editing the tree.
//!
//! # Pricing an operating point
//!
//! `--wake-thresholds` overrides the firing-threshold set ([`crate::wake`]),
//! `--phases` takes a subset of the three measurements and `--voices` measures
//! fewer than the ten clones. All three are measurement aids, and the
//! classification is by value rather than by flag: a run that prices another set,
//! measures a subset, measures a short voice set, or is given a wall-clock, resident
//! or swap-growth bound larger than the defaults reports `screened`, says why on its
//! console and in the report, and leaves `MEASUREMENTS.md` alone even when it
//! measured all three phases; while a run that states the shipped set — naming it
//! or not — over all three phases and all eleven voices, inside the defaults, is
//! the benchmark and writes the record. A voice set short of eleven is labelled as
//! such wherever it is printed, so a three-row table cannot be mistaken for the
//! eleven-voice measurement.
//!
//! # The synthesized corpus
//!
//! The corpus is 1 956 clips: the enrolled voice's 396 (ten enrolment clips, its
//! 230 calibration negatives, 50 positives, 104 speech negatives, and the two
//! intelligibility probes), and 156 per clone — 104 speech negatives, 50
//! positives and the two probes — for the ten.
//!
//! It is **built once, deliberately**, by the `wake-word-build` binary: rendering
//! is never a side effect of measuring. That operation's only job is to leave the
//! cache holding every clip a measurement needs and to say how many clips it reused
//! and how many it rendered; it writes no report, and the one-hour measurement bound
//! does not apply to it (the memory guards do). A measurement never renders: a clip
//! it finds missing is a stop naming the shortfall and the build, not the start of a
//! ninety-minute synthesis.
//!
//! The cache is one fixed, durable directory
//! (`$XDG_CACHE_HOME/mahbot/wake-word-corpus`, else `~/.cache/…`), not a per-run
//! one. A clip's key is content — the text, the seed, the sample rate, both TTS
//! files, the reference voice (or, for a clone, the span it was cut from) and the
//! code lines of the modules that produce it — so the enrolled voice's existing
//! clips stay valid, one voice's audio can never stand in for another's, and a
//! second build renders nothing. The one legitimate rebuild is a change to the
//! generation path itself: the build then sweeps the entries every recorded
//! generation claimed and it did not produce — never a file no record names — so a
//! superseded corpus does not accumulate. A record belongs to the generation that
//! wrote it and survives until a later build reclaims its entries; the record is
//! rewritten only after the sweep, so a build killed mid-sweep leaves the generation
//! it was replacing recorded and reclaimable by the next one. The trade is stated
//! rather than hidden: a measurement of a superseded generation that shares this
//! cache under another lock finds its clips gone and stops with a miss, because the
//! alternative is a cache that grows without bound. A cache that cannot be written
//! stops the run instead of quietly costing the corpus again.

mod audio;
pub mod build;
pub(crate) mod config;
mod detect;
mod real_audio;
mod record;
mod sets;
mod voices;
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
use crate::wake::{self, Enrollment};

use crate::guard::{Abort, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock, Watch};
use audio::{Category, Synth};
use config::BenchConfig;
use detect::DetectionStream;
use voices::Speaker;

/// The bench's exit contract, shared with the corpus build and the harnesses that
/// call them: a refused input — a flag, a bound, a corpus cache that cannot be
/// created, a record or manifest it holds, a pinned recording that no longer matches
/// its pin — is the stand's argument refusal (2); anything the run reached later is a
/// failed run (1); a guard stop exits 75 from the stop itself.
pub fn exit_code(error: &anyhow::Error) -> i32 {
    if error.is::<config::Refused>() { 2 } else { 1 }
}

/// Run the bench once: measure, write the report atomically.
///
/// # Errors
/// A missing or malformed input; a model or corpus failure; a corpus cache short
/// of the clips the measurement needs (a measurement never renders); an
/// enrolment that fails its consistency gate; the wall-clock stop; or the
/// resident-set ceiling. The report is written for every one of those — a stop
/// or a failure leaves what it measured, and a run whose denominator measured
/// nothing writes its report and then exits non-zero.
pub fn run() -> Result<()> {
    let started = Instant::now();
    let cfg = BenchConfig::from_env(config::Mode::Measure)?;
    let _lock = RunLock::acquire(&cfg.lock)?;
    let deadline = Deadline::new(cfg.timeout_secs);
    // The swap ceiling the record states is the one the guard enforces: both come
    // from `cfg`, never from a second read of the environment.
    let ceiling = MemoryCeiling::with_swap_growth(cfg.max_rss_mib, cfg.swap_growth_mib);
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
                // The corpus counts are printed from the progress lock, not left
                // to a report a stopped run may or may not write: what a cold
                // cache built is the point of the run that stopped on it.
                corpus_line(&cfg, &state(&progress));
                match report.write(
                    &cfg,
                    &progress,
                    started,
                    "aborted",
                    Some((phase, message.as_str())),
                ) {
                    Ok(_) => eprintln!(
                        "[wake-word] partial report: {}",
                        cfg.report_path().display()
                    ),
                    Err(error) => eprintln!(
                        "[wake-word] cannot write the report {}: {error:#} — stopping without one",
                        cfg.report_path().display()
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
         sets run one per pass) · {} verification voices · stop {}s · ceiling {} MiB · lock {}",
        cfg.models_label(),
        cfg.corpus.display(),
        listen::GROUP,
        cfg.voices,
        deadline.secs(),
        cfg.max_rss_mib,
        cfg.lock.display()
    );
    for reason in cfg.screened_reasons() {
        eprintln!("[wake-word] screened: {reason}");
    }

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
    // The corpus counts are stated on every path this thread reaches — completed,
    // stopped at a work item, or failed — before the outcome lines. A guard-thread stop
    // prints the same line from its own stop action and exits there, without this thread.
    corpus_line(&cfg, &state(&progress));
    if let Some((phase, message)) = &stop {
        eprintln!("[wake-word] stopped in `{phase}`: {message}");
    }
    match (&write, &stop) {
        (Ok(_), Some(_)) => eprintln!(
            "[wake-word] partial report: {}",
            cfg.report_path().display()
        ),
        (Ok(_), None) => eprintln!("[wake-word] report: {}", cfg.report_path().display()),
        (Err(error), _) => eprintln!(
            "[wake-word] cannot write the report {}: {error:#}",
            cfg.report_path().display()
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
                cfg.report_path().display()
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
             {:.4}/h over wall-clock audio · one stream · {:.1}s wall · \
             peak={}",
            real.files_fed,
            real.audio_hours,
            real.speech_hours,
            real.vad_active_hours,
            real.events,
            real.events as f64 / real.vad_active_hours,
            real.events as f64 / real.audio_hours,
            real.wall_secs,
            crate::report::peak_mib(),
        );
    }
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
            cfg.report_path().display()
        );
        std::process::exit(EXIT_STOPPED);
    };
    // The rows, read out of the report this run just wrote.
    record::console(&written);
    // Only the benchmark itself is the record: a screened run priced a candidate
    // point and says so instead.
    if status == COMPLETE {
        measurements::refresh("bench", &record::block(&written)).with_context(|| {
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
/// phases, all eleven voices, inside the default wall-clock, resident and
/// swap-growth bounds. A run that priced a candidate — another set, a subset of the
/// phases, a short voice set, or a larger bound — is `screened` and never writes the
/// record, even when it measured everything; anything that stopped is named with the
/// phase and the reason.
fn classify(
    cfg: &BenchConfig,
    measured: &Result<()>,
    unmeasured: Option<&anyhow::Error>,
    phase: &'static str,
) -> (&'static str, Option<(&'static str, String)>) {
    let benchmark = cfg.screened_reasons().is_empty();
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

/// The corpus line: what this run read from the cache, how much of that it had to
/// render, and what the rendering cost — in the same words the build's own account
/// uses. Stated as **clip reads**, because a run the guard stopped has read only
/// part of the corpus and must not print its own progress as the corpus size.
/// Printed from the progress lock, so a stopped run states the same counts as one
/// that finished.
fn corpus_line(cfg: &BenchConfig, progress: &Progress) {
    let corpus = progress.synthesis;
    eprintln!(
        "[wake-word] corpus {}: {} clip reads — {} reused, {} rendered in {:.1}s",
        cfg.cache.display(),
        corpus.reads,
        corpus.cache_hits,
        corpus.rendered(),
        corpus.synth_secs
    );
}

/// A rate over a denominator a run measured. A zero denominator produces no rate
/// at all in the report's sections — this helper is only reached with a count the
/// run's own rows produced, which is never zero clips.
fn rate(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

/// What the run has measured so far. The complete and the stopped report are
/// both built from this, so a phase cannot appear in one shape and not the
/// other.
pub(crate) struct Progress {
    /// The phase the run is inside, for a stop or a failure to name.
    current: &'static str,
    inputs: Option<config::Inputs>,
    enrolment: Option<EnrollStats>,
    calibration: Option<wake::Calibration>,
    /// The threshold set the enrolment scores with — the point this run prices.
    thresholds: Option<wake::Thresholds>,
    /// The voices the synthetic side was checked with: the enrolled one's check,
    /// the selected clones, and the witness that they are distinct.
    voices: Option<voices::Selection>,
    recall: Option<Recall>,
    non_phrase: Option<NonPhrase>,
    real: Option<real_audio::RealAudio>,
    /// The corpus cache's counters, updated by [`Synth`] through this same lock:
    /// a report written mid-run counts the clips the phases have already read
    /// instead of the zero they started from.
    synthesis: audio::SynthStats,
    /// What the build left beside the corpus: the generation it completed and what
    /// rendering it cost. Absent when the cache carries no manifest for this
    /// generation, which the record then names.
    corpus_build: Option<Value>,
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
            voices: None,
            recall: None,
            non_phrase: None,
            real: None,
            synthesis: audio::SynthStats::default(),
            corpus_build: None,
            phases: std::collections::BTreeMap::new(),
        }
    }
}

/// The positive side: one row per voice.
struct Recall {
    voices: Vec<VoicePositives>,
}

impl Recall {
    /// `(detected, of)` over every voice's clips: the pooled figure, which is not
    /// a number of independent utterances.
    fn pooled(&self) -> (usize, usize) {
        self.voices.iter().fold((0, 0), |(detected, of), row| {
            (detected + row.detected, of + row.of)
        })
    }

    fn clips_per_voice(&self) -> Option<usize> {
        self.voices.first().map(|row| row.of)
    }
}

/// One voice's positive row: its own stream, its own 50 clips.
struct VoicePositives {
    voice: String,
    detected: usize,
    of: usize,
    missed: Vec<String>,
}

/// The negative side: one row per speech stream, plus the non-speech stream.
struct NonPhrase {
    voices: Vec<VoiceNegatives>,
    non_speech: NonSpeech,
}

impl NonPhrase {
    fn speech_pooled(&self) -> (usize, usize) {
        self.voices.iter().fold((0, 0), |(fired, of), row| {
            (fired + row.fired(), of + row.of())
        })
    }
}

/// One voice's negative row: how many of its clips fired, per category, in
/// [`Category::SPEECH`] order.
struct VoiceNegatives {
    voice: String,
    fired: [usize; 3],
    of: [usize; 3],
    /// Every clip that fired, named with its stream.
    fired_labels: Vec<String>,
}

impl VoiceNegatives {
    fn fired(&self) -> usize {
        self.fired.iter().sum()
    }

    fn of(&self) -> usize {
        self.of.iter().sum()
    }
}

/// The 17 non-speech clips, in a stream of their own: they have no voice.
struct NonSpeech {
    fired: usize,
    of: usize,
    fired_labels: Vec<String>,
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

/// Both role pairs, opened once: the measurement and the corpus build read the same
/// four inputs and fail with the same words.
fn open_pairs(cfg: &BenchConfig) -> Result<(Arc<Pair>, Arc<Pair>)> {
    let asr = Arc::new(
        Pair::open(&cfg.asr_body, &cfg.asr_sound).context("cannot open the recognition pair")?,
    );
    let tts = Arc::new(
        Pair::open(&cfg.tts_body, &cfg.tts_sound).context("cannot open the synthesis pair")?,
    );
    Ok((asr, tts))
}

/// The corpus cache is an input both entry points own: it is the run's durable
/// artifact, so [`prepare_cache`] creates it and drops only this process's own
/// half-written entries. Called before anything reads inside it, so a cache root that
/// is not a directory is named here rather than by the manifest's reader.
pub(crate) fn prepare_cache(cache: &Path) -> Result<(), config::Refused> {
    std::fs::create_dir_all(cache).with_context(|| {
        format!(
            "cannot create the corpus cache {} — the synthesized corpus is the run's durable \
             artifact",
            cache.display()
        )
    })?;
    audio::clean_cache(cache);
    Ok(())
}

/// The synthesis front end over the cache the caller prepared ([`prepare_cache`]): a
/// measurement requires it filled rather than synthesizing quietly somewhere else.
fn open_corpus(
    cfg: &BenchConfig,
    tts: Arc<Pair>,
    synthesis_hash: String,
    missing: audio::Missing,
    progress: &Arc<Mutex<Progress>>,
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<Synth> {
    Ok(Synth::new(
        tts,
        cfg.voice.clone(),
        cfg.cache.clone(),
        synthesis_hash,
        missing,
        Arc::clone(progress),
        deadline,
        ceiling,
    ))
}

/// The measurement, phase by phase. Each phase marks itself current, so a stop
/// names where it happened, and every long phase checks the stop at its own work
/// item — a clip read, a detection chunk, a corpus file — on top of the guard
/// thread's own watch.
fn measure(
    cfg: &BenchConfig,
    deadline: Deadline,
    ceiling: MemoryCeiling,
    progress: &Arc<Mutex<Progress>>,
) -> Result<()> {
    // The cache is the run's durable artifact: created and cleaned here once, before
    // anything reads inside it, so a root that is not a directory is a refused input
    // named as such rather than a failure the manifest reader reports.
    prepare_cache(&cfg.cache)?;
    let inputs = phase(progress, "inputs", || {
        deadline.check("inputs")?;
        let inputs = config::pin_inputs(cfg)?;
        // The corpus is verified here, before any phase reads it: a bad pin stops
        // the run in its first seconds.
        real_audio::verify(&cfg.corpus, sets::REAL_AUDIO_FILES)
            .context("the pinned real-audio corpus")?;
        Ok(inputs)
    })?;
    let (asr, tts) = open_pairs(cfg)?;
    let mut synth = open_corpus(
        cfg,
        Arc::clone(&tts),
        inputs.synthesis_hash(),
        audio::Missing::Stop,
        progress,
        deadline,
        ceiling,
    )?;
    let mut listener =
        listen::Listener::new(Arc::clone(&asr)).context("cannot build the listening path")?;
    // The build's own account of this generation, so the record can state what the
    // render cost instead of leaving a warm run looking free. A manifest that cannot
    // be read is refused, not read as absent: the cost row would understate it.
    let generation = inputs.synthesis_hash();
    state(progress).corpus_build = audio::read_build_manifest(&cfg.cache, &generation)?;
    state(progress).inputs = Some(inputs);

    // A measurement renders nothing: a cache short of what it needs is a stop
    // naming the shortfall and the build that fills it, never a synthesis the
    // measurement quietly starts. The pre-flight is its own step — its failure is a
    // missing corpus, not the `inputs` phase, which has already finished.
    let synthetic = cfg.phases.recall || cfg.phases.non_phrase;
    state(progress).current = "corpus pre-flight";
    synth.require_corpus(
        &audio::enrolment_requests(),
        "the enrolment and its calibration pool",
    )?;
    if synthetic {
        // Read by the voices phase's enrolled check, before the plan that lists them
        // is pre-flighted: a broken probe stops here rather than after enrolment.
        synth.require_corpus(
            &audio::probe_requests(&Speaker::Enrolled),
            "the enrolled voice's intelligibility probes",
        )?;
        // The clone spans the voices phase reads: every cut the build recorded for this
        // generation must be there and whole, checked here rather than inside the phase
        // that reads it. The record is read here, not above, because this pre-flight is
        // its only consumer: a phase subset that selects no voice does not need it.
        let records = audio::read_build_keys(&cfg.cache)?;
        voices::require_spans(&cfg.cache, records.get(&generation))?;
    }

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

    // ── Voices: the enrolled one's own two-sentence check, then the ten clones
    // the pinned pool fills. Both sides go through the same synthesis path and are
    // read back by the same recogniser; the selection's spans are the only thing
    // the clones differ in. ──
    let speakers = if synthetic {
        let selection = phase(progress, "voices", || {
            deadline.check("voices")?;
            ceiling.check("voices")?;
            voices::select(&tts, &asr, &mut synth, cfg, deadline, ceiling)
        })?;
        let speakers = selection.speakers();
        state(progress).voices = Some(selection);
        speakers
    } else {
        Vec::new()
    };
    // A real-audio-only run measures no synthetic clips at all, so it asks the
    // corpus for none of them; a screening run asks only for the phases it reads.
    if synthetic {
        state(progress).current = "corpus pre-flight";
        synth.require_corpus(&audio::plan(&speakers, cfg.phases), "the measured set")?;
    }

    // ── Recall: one warm continuous stream per voice over that voice's 50
    // phrase clips. ──
    let recall = if cfg.phases.recall {
        let clips = phase(progress, "recall synthesis", || {
            speakers
                .iter()
                .map(|speaker| {
                    audio::recall_clips(&mut synth, speaker).map(|clips| (speaker.clone(), clips))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let recall = phase(progress, "recall", || {
            deadline.check("recall")?;
            ceiling.check("recall")?;
            let mut rows = Vec::with_capacity(clips.len());
            // One warm stream per voice: a stream carries the rolling window, the
            // adaptive arm and the audio-clock cooldown, so a shared one would let one
            // voice's decision reach another's clips.
            for (speaker, clips) in clips {
                let mut stream = warm_stream(&enrollment);
                rows.push(measure_recognition(
                    &mut stream,
                    &mut listener,
                    &speaker,
                    &clips,
                    deadline,
                    ceiling,
                )?);
            }
            Ok(Recall { voices: rows })
        })?;
        for row in &recall.voices {
            if !row.missed.is_empty() {
                eprintln!(
                    "[wake-word] missed positives in {}: {}",
                    row.voice,
                    row.missed.join(", ")
                );
            }
        }
        Some(recall)
    } else {
        None
    };

    // ── False reactions: one warm continuous stream per voice over that voice's
    // 104 speech clips, plus one stream over the 17 non-speech clips. ──
    let non_phrase = if cfg.phases.non_phrase {
        let speech = phase(progress, "non-phrase synthesis", || {
            speakers
                .iter()
                .map(|speaker| {
                    audio::speech_negatives(&mut synth, speaker)
                        .map(|clips| (speaker.clone(), clips))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let non_phrase = phase(progress, "non-phrase", || {
            deadline.check("non-phrase")?;
            ceiling.check("non-phrase")?;
            let mut rows = Vec::with_capacity(speech.len());
            // The same per-voice streams as the recall side, for the same reason.
            for (speaker, clips) in speech {
                let mut stream = warm_stream(&enrollment);
                rows.push(measure_false_reactions(
                    &mut stream,
                    &mut listener,
                    &speaker,
                    &clips,
                    deadline,
                    ceiling,
                )?);
            }
            let non_speech = measure_non_speech(
                &enrollment,
                &mut listener,
                &audio::non_speech_clips()?,
                deadline,
                ceiling,
            )?;
            Ok(NonPhrase {
                voices: rows,
                non_speech,
            })
        })?;
        for row in &non_phrase.voices {
            if !row.fired_labels.is_empty() {
                eprintln!(
                    "[wake-word] false reactions in {}: {}",
                    row.voice,
                    row.fired_labels.join(", ")
                );
            }
        }
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
    // The bounds this run was actually taken under, so a record cannot be read as
    // one taken under the defaults when a guard was widened.
    report.insert(
        "guards".into(),
        json!({
            "timeout_secs": cfg.timeout_secs,
            "max_rss_mib": cfg.max_rss_mib,
            "swap_growth_mib": cfg.swap_growth_mib,
        }),
    );
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
                },
            }),
        );
    }
    if let Some(recall) = &p.recall
        && !recall.voices.is_empty()
    {
        let (detected, of) = recall.pooled();
        let voices: Vec<Value> = recall
            .voices
            .iter()
            .map(|row| {
                json!({
                    "voice": row.voice,
                    "detected": row.detected,
                    "of": row.of,
                    "rate": rate(row.detected, row.of),
                    "missed": row.missed,
                })
            })
            .collect();
        report.insert(
            "recognition".into(),
            json!({
                "detected": detected,
                "of": of,
                "rate": rate(detected, of),
                "set": {
                    "clips": of,
                    "clips_pinned": sets::EXPECTED_POSITIVE_TOTAL,
                    "voices": recall.voices.len(),
                    "clips_per_voice": recall.clips_per_voice(),
                    "distinct_texts": 1 + sets::COMMAND_PLUS_SPEECH_PHRASES.len(),
                    "wake_only_clips_per_voice": sets::HELD_OUT_WAKE_ONLY_CLIPS,
                    "command_plus_speech_clips_per_voice": sets::COMMAND_PLUS_SPEECH_PHRASES.len(),
                },
                "voices": voices,
                "clip_seed_bands": {
                    "wake_only": {"base": sets::HELD_OUT_SEED_BASE, "clips": sets::HELD_OUT_WAKE_ONLY_CLIPS},
                    "command_plus_speech": {"base": sets::COMMAND_PLUS_SPEECH_SEED_BASE, "clips": sets::COMMAND_PLUS_SPEECH_PHRASES.len()},
                },
            }),
        );
    }
    // Each measurement states only what it measured: the synthetic set and the
    // real-audio stream are independent, so a real-audio-only screening still
    // reports its false accepts, and a phase that did not finish is absent,
    // never a null.
    let mut false_reactions = serde_json::Map::new();
    if let Some(non_phrase) = &p.non_phrase
        && !non_phrase.voices.is_empty()
    {
        let voices: Vec<Value> = non_phrase
            .voices
            .iter()
            .map(|row| {
                let mut cell = serde_json::Map::new();
                cell.insert("voice".into(), json!(row.voice));
                cell.insert("of".into(), json!(row.of()));
                cell.insert("false_reactions".into(), json!(row.fired()));
                cell.insert("rate".into(), json!(rate(row.fired(), row.of())));
                cell.insert("fired".into(), json!(row.fired_labels));
                for category in Category::SPEECH {
                    let slot = category.index();
                    cell.insert(
                        category.key().into(),
                        json!({
                            "false_reactions": row.fired[slot],
                            "of": row.of[slot],
                            "rate": rate(row.fired[slot], row.of[slot]),
                        }),
                    );
                }
                Value::Object(cell)
            })
            .collect();
        let (fired, of) = non_phrase.speech_pooled();
        let mut speech = serde_json::Map::new();
        speech.insert("false_reactions".into(), json!(fired));
        speech.insert("of".into(), json!(of));
        speech.insert("rate".into(), json!(rate(fired, of)));
        for category in Category::SPEECH {
            let slot = category.index();
            let category_fired = non_phrase.voices.iter().map(|row| row.fired[slot]).sum();
            let category_of = non_phrase.voices.iter().map(|row| row.of[slot]).sum();
            speech.insert(
                category.key().into(),
                json!({
                    "false_reactions": category_fired,
                    "of": category_of,
                    "rate": rate(category_fired, category_of),
                }),
            );
        }
        speech.insert("voices".into(), Value::Array(voices));
        false_reactions.insert(
            "non_phrase_set".into(),
            json!({
                "set": {
                    "clips": of + non_phrase.non_speech.of,
                    "clips_pinned": sets::EXPECTED_NON_PHRASE_TOTAL,
                    "speech_clips": of,
                    "non_speech_clips": non_phrase.non_speech.of,
                    "voices": non_phrase.voices.len(),
                    "speech_clips_per_voice": sets::SPEECH_NEGATIVE_CLIPS,
                    "confusable_clips_per_voice": sets::CONFUSABLE_CLIPS,
                    "unrelated_clips_per_voice": sets::UNRELATED_CLIPS,
                    "phrase_in_passing_clips_per_voice": sets::PHRASE_IN_PASSING_CLIPS,
                    "distinct_texts": sets::CONFUSABLE_PHRASES.len()
                        + sets::UNRELATED_PHRASES.len()
                        + sets::PHRASE_IN_PASSING_PHRASES.len(),
                    "silence_clips": sets::SILENCE_DURATIONS.len(),
                    "noise_clips": sets::NOISE_PROFILES.len()
                        + sets::NOISE_PROFILES_DETECTION_ONLY.len(),
                },
                "speech": Value::Object(speech),
                "non_speech": {
                    "false_reactions": non_phrase.non_speech.fired,
                    "of": non_phrase.non_speech.of,
                    "rate": rate(non_phrase.non_speech.fired, non_phrase.non_speech.of),
                    "fired": non_phrase.non_speech.fired_labels,
                },
                "clip_seed_bands": {
                    "confusable": [sets::CONFUSABLE_BANDS[0].0, sets::CONFUSABLE_BANDS[1].0],
                    "unrelated": [sets::UNRELATED_BANDS[0].0, sets::UNRELATED_BANDS[1].0],
                    "phrase_in_passing": sets::PHRASE_IN_PASSING_SEED_BASE,
                    "detection_noise": sets::DETECTION_NOISE_SEED_BASE,
                },
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
            coverage.insert("phrase_utterances".into(), json!(recall.pooled().1));
        }
        if let Some(non_phrase) = &p.non_phrase {
            coverage.insert("non_phrase_set".into(), json!(non_phrase.speech_pooled().1));
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
            "reads": synthesis.reads,
            "reused": synthesis.cache_hits,
            "rendered": synthesis.rendered(),
            "synthesis_secs": synthesis.synth_secs,
        }),
    );
    if let Some(selection) = &p.voices {
        report.insert("voices".into(), selection.json());
    }
    if let Some(build) = &p.corpus_build {
        report.insert("corpus_build".into(), build.clone());
    }
    let screened = cfg.screened_reasons();
    if !screened.is_empty() {
        report.insert("screened_because".into(), json!(screened));
    }
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
    Value::Object(report)
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
    for clip in &clips {
        let speech = detect::vad_gate(&clip.samples, &mut earshot::Detector::default());
        if speech.is_empty() {
            eprintln!(
                "[wake-word] enrolment clip `{}`: no VAD-positive speech — skipped",
                clip.label
            );
            skipped.push(clip.label.clone());
            continue;
        }
        utterances.push(listener.embed(&speech)?);
    }
    let prototype = consistency(&utterances)?;

    // Ambient → the pinned calibration pool (owner, confusable, unrelated), from
    // the same request list the plan pre-flights and the build renders.
    let mut skipped_negatives = Vec::new();
    let mut negatives = audio::ambient_negatives(listener)?;
    negatives.extend(audio::calibration_negatives(
        synth,
        listener,
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

/// One warm, **ungrouped** scoring stream. Groups exist only in the real-audio
/// sweep: grouping the clip sets would move a decision at a group boundary, and
/// `a_clip_set_stream_is_ungrouped` pins that this helper never switches to it.
fn warm_stream(enrollment: &Enrollment) -> DetectionStream {
    DetectionStream::warm(enrollment.clone())
}

/// Feed one voice's positive clips through its own warm stream, two seconds of
/// silence between them. The stop and the resident-set ceiling are checked per
/// clip, on top of the per-chunk check inside [`DetectionStream::feed`].
fn measure_recognition(
    stream: &mut DetectionStream,
    listener: &mut listen::Listener,
    speaker: &Speaker,
    clips: &[audio::Clip],
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<VoicePositives> {
    let gap = vec![0.0f32; sets::FILE_GAP_SAMPLES];
    let mut detected = 0usize;
    let mut missed = Vec::new();
    for clip in clips {
        ceiling.check_process("recall")?;
        let hit = feed_clip(stream, listener, &clip.samples, &gap, deadline, "recall")?;
        if hit {
            detected += 1;
        } else {
            missed.push(clip.label.clone());
        }
    }
    Ok(VoicePositives {
        voice: speaker.name().to_string(),
        detected,
        of: clips.len(),
        missed,
    })
}

/// Feed one voice's speech negatives through its own warm stream. Every clip that
/// fired is a false reaction, counted in its category's slot.
fn measure_false_reactions(
    stream: &mut DetectionStream,
    listener: &mut listen::Listener,
    speaker: &Speaker,
    clips: &[(Category, audio::Clip)],
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<VoiceNegatives> {
    let gap = vec![0.0f32; sets::FILE_GAP_SAMPLES];
    let mut fired = [0usize; 3];
    let mut of = [0usize; 3];
    let mut fired_labels = Vec::new();
    for (category, clip) in clips {
        ceiling.check_process("non-phrase")?;
        let slot = category.index();
        of[slot] += 1;
        if feed_clip(
            stream,
            listener,
            &clip.samples,
            &gap,
            deadline,
            "non-phrase",
        )? {
            fired[slot] += 1;
            fired_labels.push(clip.label.clone());
        }
    }
    Ok(VoiceNegatives {
        voice: speaker.name().to_string(),
        fired,
        of,
        fired_labels,
    })
}

/// Feed the non-speech clips through a stream of their own: they have no voice,
/// so they cannot share a stream with one.
fn measure_non_speech(
    enrollment: &Enrollment,
    listener: &mut listen::Listener,
    clips: &[audio::Clip],
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<NonSpeech> {
    let mut stream = warm_stream(enrollment);
    let gap = vec![0.0f32; sets::FILE_GAP_SAMPLES];
    let mut fired_labels = Vec::new();
    for clip in clips {
        ceiling.check_process("non-phrase")?;
        if feed_clip(
            &mut stream,
            listener,
            &clip.samples,
            &gap,
            deadline,
            "non-phrase",
        )? {
            fired_labels.push(clip.label.clone());
        }
    }
    Ok(NonSpeech {
        fired: fired_labels.len(),
        of: clips.len(),
        fired_labels,
    })
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
            "the real-audio phase fed a {}-file corpus but the VAD marked no speech — the \
             false-accept rate has no denominator",
            real.files_fed
        );
    }
    Ok(())
}

/// Atomic, durable write: a temporary beside the target, flushed to the device, then a
/// rename — so a crash leaves either the old file or the new one, never a half-written
/// one and never a name pointing at bytes that did not reach the disk. The one idiom
/// every cache entry, record, manifest and report goes through. The temporary is removed
/// when the write fails.
pub(crate) fn write_atomic(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let written = write(&tmp)
        .and_then(|()| {
            // Flushed to the device, so the rename that follows cannot make a name
            // visible before its bytes are.
            Ok(std::fs::OpenOptions::new()
                .write(true)
                .open(&tmp)?
                .sync_all()?)
        })
        .and_then(|()| {
            std::fs::rename(&tmp, path)
                .with_context(|| format!("cannot move {} into place", tmp.display()))
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written.with_context(|| format!("cannot write {}", path.display()))
}

/// Write the report only after the whole measurement completed.
fn write_report(path: &Path, report: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(report).context("cannot serialise the report")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    write_atomic(path, |tmp| Ok(std::fs::write(tmp, text.as_bytes())?))
}

/// The progress lock, recovering from a poisoned mutex: a panic elsewhere must
/// not take the guard's own reading, or the report, down with it.
pub(crate) fn state<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
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
        write_report(cfg.report_path(), &report)?;
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
            mode: config::Mode::Measure,
            models: Some(std::path::PathBuf::from("/models")),
            voice: std::path::PathBuf::from("/voice.wav"),
            corpus: std::path::PathBuf::from("/corpus"),
            report: Some(std::path::PathBuf::from("/report.json")),
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
            voices: sets::NUM_VERIFICATION_VOICES,
            swap_growth_mib: crate::guard::MAX_SWAP_GROWTH_MIB,
        }
    }

    fn test_voice(index: usize) -> Arc<voices::Voice> {
        Arc::new(voices::Voice {
            name: format!("voice{index:02}"),
            recording: "speech/librivox/speech-librivox-0002.wav",
            sha256: "f907f96e69eb8e680c86964f63ff570d2c24458beacd7d94520c44b2a5c10868",
            offset_secs: 11.4,
            secs: 15.0,
            wav: std::path::PathBuf::from("/cache/voice.wav"),
        })
    }

    fn test_selection(count: usize) -> voices::Selection {
        let voices: Vec<Arc<voices::Voice>> = (1..=count).map(test_voice).collect();
        let similarity = (0..voices.len())
            .flat_map(|a| (a + 1..voices.len()).map(move |b| (a, b, 0.5)))
            .collect();
        voices::Selection {
            voices,
            replacements: Vec::new(),
            enrolled: vec![voices::SentenceCheck {
                text: "the weather is sunny today",
                seed: 5_000,
                heard: "the weather is sunny today.".into(),
                passed: true,
                truncated: false,
            }],
            similarity,
            spans: Vec::new(),
        }
    }

    /// One negative row for `voice`, with `fired` split over the categories.
    fn test_negatives(voice: &str) -> VoiceNegatives {
        VoiceNegatives {
            voice: voice.to_string(),
            fired: [2, 1, 0],
            of: [
                sets::CONFUSABLE_CLIPS,
                sets::UNRELATED_CLIPS,
                sets::PHRASE_IN_PASSING_CLIPS,
            ],
            fired_labels: vec![format!("{voice}/confusable_s800")],
        }
    }

    fn test_positives(voice: &str) -> VoicePositives {
        VoicePositives {
            voice: voice.to_string(),
            detected: 40,
            of: sets::POSITIVE_CLIPS,
            missed: Vec::new(),
        }
    }

    /// The clip sets are measured ungrouped: both phases take their stream from
    /// `warm_stream`, so a phase that switched to the grouped path — or a change to
    /// the helper — would move a decision at a group boundary and fail here. The
    /// sweep's own stream is the one that groups.
    #[test]
    fn a_clip_set_stream_is_ungrouped() {
        let enrollment = enrolment();
        assert_eq!(warm_stream(&enrollment).group(), 1);
        assert_eq!(DetectionStream::grouped(enrollment).group(), listen::GROUP);
    }

    fn enrolment() -> crate::wake::Enrollment {
        use crate::wake::{Calibration, Enrollment};
        let utterances = vec![vec![1.0f32, 0.0]; crate::wake::MIN_ENROLLMENT_UTTERANCES];
        Enrollment::build("wake".into(), &utterances, Calibration::default(), &[])
            .expect("an enrolment of identical utterances")
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
            voices: Some(test_selection(2)),
            recall: Some(Recall {
                voices: vec![test_positives("enrolled"), test_positives("voice01")],
            }),
            non_phrase: Some(NonPhrase {
                voices: vec![test_negatives("enrolled"), test_negatives("voice01")],
                non_speech: NonSpeech {
                    fired: 0,
                    of: sets::NON_SPEECH_CLIPS,
                    fired_labels: Vec::new(),
                },
            }),
            ..Progress::default()
        };
        let report = report_json(&progress, "complete", 1.0, &test_config(), None);
        assert_eq!(report["recognition"]["rate"], 0.8);
        assert_eq!(
            report["false_reactions"]["non_phrase_set"]["speech"]["rate"],
            3.0 / 104.0
        );
        assert_eq!(
            report["false_reactions"]["non_phrase_set"]["non_speech"]["rate"],
            0.0
        );
        assert_no_null_rates(&report, "$");
    }

    /// The report's invariant: one row per voice, each over its own clips, and a
    /// pooled figure that is labelled as pooled instead of being read as that many
    /// independent trials. The set size and the population the overall divides are
    /// different things, and both are stated.
    #[test]
    fn the_report_states_one_row_per_voice_and_its_pooled_population() {
        let mut positives = vec![test_positives("enrolled")];
        positives.extend((1..=2).map(|i| test_positives(&format!("voice{i:02}"))));
        let mut negatives = vec![test_negatives("enrolled")];
        negatives.extend((1..=2).map(|i| test_negatives(&format!("voice{i:02}"))));
        let progress = Progress {
            voices: Some(test_selection(2)),
            recall: Some(Recall { voices: positives }),
            non_phrase: Some(NonPhrase {
                voices: negatives,
                non_speech: NonSpeech {
                    fired: 0,
                    of: sets::NON_SPEECH_CLIPS,
                    fired_labels: Vec::new(),
                },
            }),
            ..Progress::default()
        };
        let report = report_json(&progress, "complete", 1.0, &test_config(), None);
        let speech = &report["false_reactions"]["non_phrase_set"]["speech"];
        let rows = speech["voices"].as_array().expect("rows");
        assert_eq!(rows.len(), 3, "enrolled plus two clones");
        for row in rows {
            assert_eq!(row["of"], sets::SPEECH_NEGATIVE_CLIPS);
            assert!(row.get("fired").is_some(), "{row}");
        }
        assert_eq!(speech["of"], 3 * sets::SPEECH_NEGATIVE_CLIPS);
        assert_eq!(speech["false_reactions"], 3 * 3);
        // The set size is what this run measured, and the population the overall
        // divides is stated beside it: a three-voice run cannot claim the pinned set.
        let set = &report["false_reactions"]["non_phrase_set"]["set"];
        assert_eq!(
            set["clips"],
            3 * sets::SPEECH_NEGATIVE_CLIPS + sets::NON_SPEECH_CLIPS
        );
        assert_eq!(set["voices"], 3);
        assert_eq!(
            set["clips_pinned"],
            sets::EXPECTED_NON_PHRASE_TOTAL,
            "the pinned set the run falls short of is named, not silently equal to it"
        );
        assert_eq!(report["recognition"]["of"], 3 * sets::POSITIVE_CLIPS);
    }

    /// The measurement aids are a classification, not a different report: a run
    /// that priced another threshold set, a subset of the phases, a short voice
    /// set, or a bound larger than the defaults is `screened` — the status the
    /// record gate and the console both read, even when the run measured
    /// everything.
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

        let mut short = test_config();
        short.voices = 3;
        assert_eq!(
            classify(&short, &Ok(()), None, "inputs").0,
            "screened",
            "a subset of the voices is a screening run"
        );

        let mut raised = test_config();
        raised.timeout_secs = crate::guard::DEFAULT_TIMEOUT_SECS + 1;
        assert_eq!(
            classify(&raised, &Ok(()), None, "inputs").0,
            "screened",
            "a run given a larger wall-clock bound produces no recorded measurement"
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
        let body = record::block(&json!({}));
        assert!(
            body.contains(&format!("status: `{}`", record::UNAVAILABLE)),
            "{body}"
        );
        for role in ["ASR body", "ASR sound", "TTS body", "TTS sound", "voice"] {
            assert!(
                body.contains(&format!("- {role}: {}", record::UNAVAILABLE)),
                "the {role} pin does not name its absence:\n{body}"
            );
        }
        for row in [
            "negatives",
            "negative overall (pooled speech)",
            "non-speech",
            "positives (pooled)",
            "false accepts",
            "thresholds",
            "derived point",
            "wall clock",
            "peak resident",
            "corpus",
            "corpus build",
            "windows per encoder pass",
        ] {
            assert!(
                body.contains(&format!("| {row} | {} |", record::UNAVAILABLE)),
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

    /// The plan is the corpus: every clip the synthetic side renders is a request,
    /// its labels are unique — so a missed or fired list names one clip — and a
    /// phase the run does not take asks the corpus for nothing.
    #[test]
    fn the_plan_is_the_whole_corpus_and_its_labels_are_unique() {
        let speakers = test_selection(sets::NUM_VERIFICATION_VOICES).speakers();
        let plan = audio::plan(&speakers, config::Phases::ALL);
        assert_eq!(plan.len(), 1_956, "the corpus the module doc states");
        let mut labels: Vec<String> = plan.iter().map(audio::Request::label).collect();
        let planned = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), planned, "two requests share a label");
        // The probes are the two pinned sentences, not the wake phrase.
        let probes: Vec<&str> = audio::probe_requests(&Speaker::Enrolled)
            .iter()
            .map(|request| request.text)
            .collect();
        assert_eq!(probes, sets::INTELLIGIBILITY_SENTENCES);
        // A screening run asks for the phases it reads, not the whole corpus.
        let recall_only = audio::plan(
            &speakers,
            config::Phases {
                recall: true,
                non_phrase: false,
                real_audio: false,
            },
        );
        assert_eq!(
            recall_only.len(),
            audio::enrolment_requests().len()
                + sets::SPEECH_VOICES * (probes.len() + sets::POSITIVE_CLIPS)
        );
    }

    /// One voice's negatives land in the category their row reports them under: the
    /// per-category percentages divide those counts, so a category that lost a band
    /// would move a row without moving the overall.
    #[test]
    fn one_voice_gets_the_pinned_clip_counts() {
        let mut categories = [0usize; 3];
        for (category, _) in audio::negative_requests(&Speaker::Enrolled) {
            categories[category.index()] += 1;
        }
        assert_eq!(
            categories,
            [
                sets::CONFUSABLE_CLIPS,
                sets::UNRELATED_CLIPS,
                sets::PHRASE_IN_PASSING_CLIPS
            ]
        );
    }

    /// A measurement never renders: the corpus build is the only writer, and the
    /// sweep is what keeps a superseded corpus from accumulating. It removes the
    /// entries any recorded generation claimed — clips and span cuts alike — and this
    /// one did not produce: never an entry no build of this cache claimed, and never a
    /// half-written one. A generation whose entries could not be removed stays
    /// recorded for the next build.
    #[test]
    fn the_sweep_removes_only_the_corpus_a_recorded_build_replaced() {
        let dir = std::env::temp_dir().join(format!(
            "mahbot-wake-sweep-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let live = "a".repeat(64);
        let dead = "b".repeat(64);
        let replaced = "e".repeat(64);
        let foreign = "d".repeat(64);
        let span = "voice-0123456789abcdef-11.40-15.00-16000.wav".to_string();
        let stale_span = "voice-fedcba9876543210-2.60-15.00-16000.wav".to_string();
        let torn = format!("{}.tmp-999999", "c".repeat(64));
        for name in [
            &live,
            &dead,
            &replaced,
            &foreign,
            &span,
            &stale_span,
            &torn,
            "notes.txt",
        ] {
            std::fs::write(dir.join(name), b"x").expect("a temp file");
        }

        let produced = std::collections::HashSet::from([live.clone(), span.clone()]);
        // Two recorded generations: the corpus this build replaces is claimed by both.
        let records = audio::Records::from([
            (
                "gen-1".to_string(),
                std::collections::HashSet::from([
                    live.clone(),
                    dead.clone(),
                    span.clone(),
                    stale_span.clone(),
                ]),
            ),
            (
                "gen-2".to_string(),
                std::collections::HashSet::from([replaced.clone()]),
            ),
        ]);

        let (removed, survivors) = audio::sweep(&dir, &produced, &records);

        assert_eq!(
            removed,
            Some(3),
            "the replaced clip, the stale span and the other generation's clip go"
        );
        assert!(
            survivors.is_empty(),
            "every entry was reclaimed: {survivors:?}"
        );
        assert!(dir.join(&live).exists(), "the generation stays");
        assert!(dir.join(&span).exists(), "its span stays");
        assert!(!dir.join(&dead).exists(), "a replaced clip goes");
        assert!(
            !dir.join(&replaced).exists(),
            "another generation's clip goes"
        );
        assert!(!dir.join(&stale_span).exists(), "a stale span goes");
        assert!(
            dir.join(&foreign).exists(),
            "a clip no recorded build claimed is kept"
        );
        assert!(dir.join(&torn).exists(), "a half-written entry is not ours");
        assert!(dir.join("notes.txt").exists(), "an unrelated file stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A measured run of all eleven voices, as the report writer sees it: one row
    /// per voice on each side, the corpus entirely reused, and a generation whose
    /// build is in the manifest.
    fn eleven_voice_progress() -> Progress {
        let mut positives = vec![test_positives(voices::ENROLLED)];
        let mut negatives = vec![test_negatives(voices::ENROLLED)];
        for index in 1..=sets::NUM_VERIFICATION_VOICES {
            positives.push(test_positives(&format!("voice{index:02}")));
            negatives.push(test_negatives(&format!("voice{index:02}")));
        }
        Progress {
            inputs: Some(test_inputs()),
            voices: Some(test_selection(sets::NUM_VERIFICATION_VOICES)),
            recall: Some(Recall { voices: positives }),
            non_phrase: Some(NonPhrase {
                voices: negatives,
                non_speech: NonSpeech {
                    fired: 1,
                    of: sets::NON_SPEECH_CLIPS,
                    fired_labels: vec!["non-speech/pink_noise_s42".into()],
                },
            }),
            real: Some(measured_real(1.066, 0.807)),
            calibration: Some(wake::Calibration::default()),
            thresholds: Some(wake::Thresholds::DEFAULT),
            synthesis: audio::SynthStats {
                reads: 1_956,
                cache_hits: 1_956,
                synth_secs: 0.0,
            },
            corpus_build: Some(json!({
                "clips": 1_956,
                "built": 1_956,
                "builds": 2,
                "synthesis_secs": 2_797.2,
                "last": {
                    "built": 0,
                    "swept": 0,
                    "wall_secs": 66.6,
                    "stopped": null,
                },
            })),
            ..Progress::default()
        }
    }

    /// The console a completed run prints states both pooled overalls, one line per
    /// voice on each side and the non-speech line — the surfaces the record's own
    /// table cannot vouch for.
    #[test]
    fn the_console_states_both_pooled_overalls_and_every_voice() {
        let report = report_json(
            &eleven_voice_progress(),
            "complete",
            287.9,
            &test_config(),
            None,
        );
        let lines = record::lines(&report);
        assert!(
            !lines.iter().any(|line| line.contains(record::UNAVAILABLE)),
            "{lines:#?}"
        );
        let pooled = |side: &str| {
            lines
                .iter()
                .find(|line| line.contains(&format!("{side} pooled")))
                .unwrap_or_else(|| panic!("no pooled {side} line: {lines:#?}"))
                .clone()
        };
        let negatives = pooled("negatives");
        assert!(
            negatives.contains("33/1144 (2.9 %) — pooled over 104 clips"),
            "{negatives}"
        );
        let positives = pooled("positives");
        assert!(
            positives.contains("440/550 (80.0 %) — pooled over 50 clips"),
            "{positives}"
        );
        let rows = |side: &str| {
            lines
                .iter()
                .filter(|line| {
                    line.starts_with(&format!("[wake-word] {side} ")) && !line.contains(" pooled")
                })
                .count()
        };
        assert_eq!(rows("negatives"), sets::SPEECH_VOICES, "{lines:#?}");
        assert_eq!(rows("positives"), sets::SPEECH_VOICES, "{lines:#?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("non-speech: 1/17 (5.9 %) in a stream of its own")),
            "{lines:#?}"
        );
    }

    /// A build that stopped is recorded as stopped, and one stopped after the
    /// corpus was already whole is not described as an incomplete corpus: the
    /// manifest's own `corpus_complete` decides the wording, and a sweep count the
    /// stop left unknown is named as unstated rather than printed as a zero.
    #[test]
    fn a_stopped_build_is_never_recorded_as_a_clean_one() {
        // `swept` is stated once the sweep returned and `null` while it has not;
        // neither a stop before the sweep nor one inside it measured a count.
        let cases = [
            (false, Value::Null, "stopped before the corpus was complete"),
            (true, Value::Null, "stopped after the corpus was complete"),
            (true, json!(7), "stopped after the corpus was complete"),
        ];
        for (complete, swept, expected) in cases {
            let mut progress = eleven_voice_progress();
            progress.corpus_build = Some(json!({
                "clips": 1_956,
                "built": 900,
                "builds": 1,
                "synthesis_secs": 1_000.0,
                "last": {
                    "built": 900,
                    "swept": swept,
                    "wall_secs": 1_200.0,
                    "stopped": {"reason": "the resident ceiling", "corpus_complete": complete},
                },
            }));
            let report = report_json(&progress, "stopped", 1_200.0, &test_config(), None);
            let body = record::block(&report);
            assert!(body.contains(expected), "{complete}: {body}");
            let sweep = match (&swept, complete) {
                // Both stopped cases leave the count unknown: neither a stop before
                // the sweep nor one inside it measured how many entries it cleared.
                (Value::Null, _) => "and its sweep count is not stated",
                _ => "clearing 7 superseded entries",
            };
            assert!(body.contains(sweep), "{complete}: {body}");
        }
    }

    /// The same writer over a measured report states every figure: the
    /// unavailable words appear only where the report has no number.
    #[test]
    fn a_measured_report_states_every_figure() {
        let progress = eleven_voice_progress();
        let report = report_json(&progress, "complete", 287.9, &test_config(), None);
        let body = record::block(&report);
        assert!(!body.contains(record::UNAVAILABLE), "{body}");
        assert!(
            body.contains(
                "| negatives | 1161 clips — 1144 speech across 11 voices, 17 \
                           non-speech in a stream of their own |"
            ),
            "{body}"
        );
        assert!(
            body.contains(
                "| negative overall (pooled speech) | 33/1144 (2.9 %) — pooled over 104 clips \
                 (56 distinct texts) across 11 voices, not 1144 independent trials |"
            ),
            "{body}"
        );
        assert!(body.contains("| non-speech | 1/17 (5.9 %)"), "{body}");
        assert!(
            body.contains(
                "| positives (pooled) | 440/550 (80.0 %) — pooled over 50 clips (11 \
                           distinct texts) across 11 voices, not 550 independent utterances |"
            ),
            "{body}"
        );
        // One row per voice on each side, with its own denominators, and the
        // pooled row below them.
        assert!(
            body.contains("Negatives by voice — 104 speech clips each:"),
            "{body}"
        );
        assert!(
            body.contains(
                "| enrolled | 2/56 (3.6 %) | 1/40 (2.5 %) | 0/8 (0.0 %) | 3/104 (2.9 %) |"
            ),
            "{body}"
        );
        // Each voice's recording and span is stated once, in the provenance block.
        assert!(
            body.contains(
                "- voice01: `speech/librivox/speech-librivox-0002.wav` · 11.40 s + 15.00 s · \
                 sha256 f907f96e69eb8e680c86964f63ff570d2c24458beacd7d94520c44b2a5c10868"
            ),
            "{body}"
        );
        assert!(
            body.contains("| pooled over 11 voices | 22/616 (3.6 %)"),
            "{body}"
        );
        assert!(
            body.contains("| pooled over 11 voices | 440 | 550 | 80.0 % |"),
            "{body}"
        );
        assert!(
            body.contains(
                "pooled over 104 clips (56 distinct texts) across 11 voices, not 1144 \
                 independent trials"
            ),
            "{body}"
        );
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
        // The build's cost is stated, so a warm record cannot make the render free.
        assert!(
            body.contains(
                "| corpus build | 1956 clips · 1956 rendered over 2 builds of this generation, \
                 2797 s of synthesis; the last build took 67 s wall and rendered 0, clearing 0 \
                 superseded entries |"
            ),
            "{body}"
        );
        assert!(
            body.contains("is min 0.5000, median 0.5000, max 0.5000 over 45 pairs"),
            "{body}"
        );
        // The record says what the numbers are, what they are not, and whose the
        // prototype is.
        for claim in [
            "generalisation evidence",
            "not comparable with the enrolled row",
            "not a regression against anything",
            "single-reader provenance, not speaker identity",
            "distinct recordings, not verified different people",
            "passed",
            "The check is deliberately not built on the wake phrase",
        ] {
            assert!(
                body.contains(claim),
                "the record does not state `{claim}`:\n{body}"
            );
        }
        // The record states this run only: nothing of the old aggregate survives.
        assert!(!body.contains("46/121"), "{body}");
        assert!(!body.contains("| recall |"), "{body}");
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
        cfg.report = Some(blocker.join("wake_word_report.json"));
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

        cfg.report = Some(dir.join("wake_word_report.json"));
        report
            .write(
                &cfg,
                &progress,
                started,
                "aborted",
                Some(("recall", "stop")),
            )
            .expect("the retry writes");
        assert!(cfg.report_path().exists());
        let text = std::fs::read_to_string(cfg.report_path()).expect("read");
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

    /// Each side states its set's shape once, in the `set` object the printers read,
    /// and the parts add up to the pinned total.
    #[test]
    fn the_reported_set_adds_up() {
        let progress = Progress {
            recall: Some(Recall {
                voices: (0..sets::SPEECH_VOICES)
                    .map(|index| test_positives(&format!("voice{index:02}")))
                    .collect(),
            }),
            non_phrase: Some(NonPhrase {
                voices: (0..sets::SPEECH_VOICES)
                    .map(|index| test_negatives(&format!("voice{index:02}")))
                    .collect(),
                non_speech: NonSpeech {
                    fired: 0,
                    of: sets::NON_SPEECH_CLIPS,
                    fired_labels: Vec::new(),
                },
            }),
            ..Progress::default()
        };
        let report = report_json(&progress, "complete", 1.0, &test_config(), None);
        let set = &report["false_reactions"]["non_phrase_set"]["set"];
        let whole = |key: &str| set[key].as_u64().expect(key) as usize;
        let per_voice = whole("confusable_clips_per_voice")
            + whole("unrelated_clips_per_voice")
            + whole("phrase_in_passing_clips_per_voice");
        assert_eq!(per_voice, sets::SPEECH_NEGATIVE_CLIPS);
        assert_eq!(per_voice, whole("speech_clips_per_voice"));
        assert_eq!(
            whole("speech_clips") + whole("non_speech_clips"),
            whole("clips")
        );
        assert_eq!(
            per_voice * whole("voices") + whole("silence_clips") + whole("noise_clips"),
            sets::EXPECTED_NON_PHRASE_TOTAL
        );
        assert_eq!(
            set["clips_pinned"].as_u64().expect("pinned"),
            sets::EXPECTED_NON_PHRASE_TOTAL as u64
        );

        let set = &report["recognition"]["set"];
        let whole = |key: &str| set[key].as_u64().expect(key) as usize;
        assert_eq!(
            whole("wake_only_clips_per_voice") + whole("command_plus_speech_clips_per_voice"),
            sets::POSITIVE_CLIPS
        );
        assert_eq!(whole("clips_per_voice"), sets::POSITIVE_CLIPS);
        assert_eq!(whole("clips"), sets::EXPECTED_POSITIVE_TOTAL);
        assert_eq!(
            set["clips_pinned"].as_u64().expect("pinned"),
            sets::EXPECTED_POSITIVE_TOTAL as u64
        );
    }

    #[test]
    fn an_intolerable_enrolment_is_refused() {
        assert!(consistency(&[]).is_err());
        assert!(consistency(&vec![vec![1.0f32, 0.0]; 4]).is_err());
    }
}
