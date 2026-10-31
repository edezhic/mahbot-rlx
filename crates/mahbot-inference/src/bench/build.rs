// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The corpus build: the one deliberate operation that renders the wake-word
//! bench's clips.
//!
//! It is not a measurement — no thresholds, no report, no record — and its only
//! job is to leave the cache holding every clip a measurement needs and the 16 kHz
//! span cut each clone is built from, to say how many clips it reused and how many it
//! rendered, and to exit. The one-hour measurement bound does not apply, and the build
//! refuses a `--timeout`: an empty cache takes hours on purpose. The safety guards do
//! apply: the exclusive lock, the resident-set ceiling, the swap-growth ceiling and
//! the free-memory floor, with the guard thread watching them while a clip renders.
//!
//! A clip is keyed by content — text, seed, sample rate, both TTS files, the
//! reference voice (or a clone's span) and the synthesis code — so a second build
//! renders nothing, the enrolled voice's existing clips stay valid, and the one
//! legitimate rebuild is a change to the generation path itself. Once the
//! generation is complete the build sweeps every corpus a recorded build has
//! superseded; the rule, and the cache-sharing trade it implies, are stated with
//! the bench's corpus docs ([`crate::bench`]).
//!
//! The counts are stated on two named bases: the corpus [`audio::plan`] lists, and
//! the clip reads this invocation made, each one either reused or rendered. The two
//! differ when the voice selection renders a candidate's probes and then rejects it
//! — that is work the plan does not list — so it shows up in the reads and never
//! inflates the corpus.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::{Value, json};

use crate::guard::{Abort, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock, Watch};

use super::audio::{self, SynthStats};
use super::config::{BenchConfig, Inputs, Mode, Phases, Refused};
use super::voices;
use super::{Progress, open_corpus, open_pairs, state};

/// The corpus-build binary's entry point: one code per outcome, so a caller can
/// tell a refused input (2) from a failed render (1) from a guard stop (75).
pub fn main_entry() {
    if let Err(error) = run() {
        eprintln!("wake-word build: {error:#}");
        std::process::exit(super::exit_code(&error));
    }
}

/// Render every clip a measurement needs, then exit.
///
/// # Errors
/// A missing or malformed input; a pinned voice recording that is missing or no
/// longer hashes to its pin; a corpus record or build manifest that cannot be read;
/// a pool that cannot fill the voices asked for; a synthesis failure; the
/// resident-set or swap ceiling. No report is written: the build's product is the
/// cache. A build that fails or is stopped still states the counts it reached, and
/// records them in the generation's manifest unless it had rendered nothing yet.
fn run() -> Result<()> {
    let started = std::time::Instant::now();
    let cfg = BenchConfig::from_env(Mode::Build)?;
    let _lock = RunLock::acquire(&cfg.lock)?;
    // Resolved before the guard: the generation names the manifest that a stopped
    // build records as well as the one a completed build writes.
    let inputs = super::config::pin_inputs(&cfg)?;
    // No wall-clock stop: this is the deliberate render, not a measurement.
    let deadline = Deadline::unbounded();
    let ceiling = MemoryCeiling::with_swap_growth(cfg.max_rss_mib, cfg.swap_growth_mib);
    let progress = Arc::new(Mutex::new(Progress::default()));
    let account = Account::new(&cfg.cache, inputs.synthesis_hash(), started);
    // The manifest is read before anything renders, and every terminal path states
    // the counts it reached: one that cannot be read refuses here, with the empty
    // ones, rather than after a corpus whose cost it would then record as zero.
    if let Some(refusal) = &account.refusal {
        account.stop(state(&progress).synthesis, &refusal.to_string());
        return Err(refusal.clone().into());
    }
    let account = Arc::new(account);
    let watchdog = match Watch::install(
        "wake-word corpus build",
        deadline,
        ceiling,
        true,
        {
            let progress = Arc::clone(&progress);
            move || state(&progress).current
        },
        {
            let progress = Arc::clone(&progress);
            let account = Arc::clone(&account);
            move |abort: Abort| {
                if account.claim() {
                    eprintln!("[wake-word build] stopped: {abort}");
                    account.stop(state(&progress).synthesis, &abort.to_string());
                    std::process::exit(EXIT_STOPPED);
                }
            }
        },
    ) {
        Ok(watchdog) => watchdog,
        Err(error) => {
            // Without the guard thread a build could outlive every bound, so it does
            // not start — and states the counts, like every other path past the account.
            account.stop(state(&progress).synthesis, &format!("{error:#}"));
            return Err(error.context("cannot start the guard thread"));
        }
    };

    eprintln!(
        "[wake-word build] models {} · corpus {} · cache {} · {} verification voices · \
         ceiling {} MiB · swap growth {} MiB · no wall-clock bound (a build is not a \
         measurement) · lock {}",
        cfg.models_label(),
        cfg.corpus.display(),
        cfg.cache.display(),
        cfg.voices,
        cfg.max_rss_mib,
        cfg.swap_growth_mib,
        cfg.lock.display()
    );

    match render(&cfg, &inputs, &progress, &account, deadline, ceiling) {
        Ok((clips, stats)) => {
            watchdog.finish();
            account.finish(clips, stats)
        }
        Err(error) => {
            // The guard thread is joined first, so it cannot state this same stop
            // from its side while this thread states it.
            watchdog.finish();
            if account.claim() {
                let abort = error.downcast_ref::<Abort>();
                if let Some(abort) = abort {
                    // The binary exits on an abort without printing the error, so
                    // the bound it hit is stated here.
                    eprintln!("[wake-word build] stopped: {abort}");
                }
                // The counts are the point of a run that failed after hours of
                // rendering, exactly as they are of one the guard stopped.
                account.stop(state(&progress).synthesis, &format!("{error:#}"));
                if abort.is_some() {
                    std::process::exit(EXIT_STOPPED);
                }
            }
            Err(error)
        }
    }
}

/// A build's one account, whichever way it ends: the corpus it is filling, the clip
/// reads it reached, the sweep, and the manifest both a completed and a stopped
/// build record. The guard thread and the caller read the same state, so a stop
/// states what the build had reached, and the first of them to claim it is the one
/// that states it: one console line and one write of the manifest.
struct Account {
    cache: PathBuf,
    generation: String,
    started: std::time::Instant,
    /// Why this build refuses to start, when it is the cache's root or manifest it
    /// could not use.
    refusal: Option<Refused>,
    /// The corpus [`audio::plan`] lists, known once the voices are selected.
    clips: Mutex<Option<usize>>,
    /// The sweep's count, known once it returns.
    swept: Mutex<Option<u64>>,
    /// Set once the plan is verified whole: a stop after that point leaves a
    /// complete corpus, which the record must not call an incomplete build.
    complete: AtomicBool,
    claimed: AtomicBool,
}

impl Account {
    /// Read once, before anything renders: the cache directory is prepared and the
    /// generation's manifest checked. A cache root that is not a directory, and a
    /// manifest that cannot be interpreted, both refuse here — naming what is wrong —
    /// rather than after hours of synthesis whose cost would then be unknown.
    fn new(cache: &Path, generation: String, started: std::time::Instant) -> Self {
        let refusal = super::prepare_cache(cache)
            .and_then(|()| audio::read_build_manifest(cache, &generation).map(|_| ()))
            .err();
        Self {
            cache: cache.to_path_buf(),
            generation,
            started,
            refusal,
            clips: Mutex::new(None),
            swept: Mutex::new(None),
            complete: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
        }
    }

    /// The corpus this build is filling, stated as soon as the plan exists.
    fn plan(&self, clips: usize) {
        *state(&self.clips) = Some(clips);
    }

    /// The corpus is whole: a later stop is this invocation's failure, not the
    /// generation's shortfall.
    fn whole(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }

    /// The sweep returned, having cleared `entries` — or `None` when it could not
    /// examine every entry, which a later reader states as unfinished.
    fn swept(&self, entries: Option<u64>) {
        *state(&self.swept) = entries;
    }

    /// Claim the account: `true` for the first caller only, so the guard thread and
    /// the error boundary state one stop between them. A caller that claims it
    /// prints the reason and calls [`Self::stop`].
    fn claim(&self) -> bool {
        !self.claimed.swap(true, Ordering::SeqCst)
    }

    /// A completed build: its figures, then the generation's manifest. `clips` is the
    /// plan `render` verified whole before it returned, passed in rather than re-read:
    /// a completed build must not lose its manifest write to an assertion.
    fn finish(&self, clips: usize, stats: SynthStats) -> Result<()> {
        eprintln!("{}", self.line(&self.figures(stats)));
        eprintln!(
            "[wake-word build] all {clips} planned clips are present; the measurement renders \
             none of them"
        );
        audio::write_build_manifest(&self.cache, &self.manifest(stats, None))
    }

    /// A claimed stop: the same figures, plus what the cache now holds. A stop after
    /// nothing was rendered leaves the generation's manifest untouched on purpose:
    /// this invocation has no render to record, and rewriting the manifest would
    /// replace the build that filled the cache with a no-op's line. A *completed*
    /// no-op is still a build of the generation, and [`Self::finish`] records it.
    fn stop(&self, stats: SynthStats, reason: &str) {
        eprintln!("{}", self.line(&self.figures(stats)));
        if stats.rendered() == 0 {
            return;
        }
        eprintln!(
            "[wake-word build] what this build rendered is in the cache; the next build resumes \
             from it"
        );
        let manifest = self.manifest(stats, Some(reason));
        if let Err(error) = audio::write_build_manifest(&self.cache, &manifest) {
            // A stop is already a failure; the console line above states the counts.
            eprintln!("[wake-word build] cannot record the manifest: {error:#}");
        }
    }

    /// The figures between the cache path and the wall clock: the corpus, the reads
    /// (`reused + rendered`, by definition), the renders outside the corpus, the
    /// sweep and the synthesis time.
    fn figures(&self, stats: SynthStats) -> String {
        let clips = *state(&self.clips);
        let complete = self.complete.load(Ordering::SeqCst);
        let mut figures = Vec::new();
        match clips {
            Some(clips) => figures.push(format!("{clips} clips")),
            // `clips` is set by the plan, which the voice selection precedes: a stop
            // before it has no corpus to state, however many probes it rendered.
            None => figures.push("corpus not stated: the build stopped before its plan".into()),
        }
        figures.push(format!(
            "{} clip reads: {} reused, {} rendered",
            stats.reads,
            stats.cache_hits,
            stats.rendered()
        ));
        // A whole corpus has had every plan clip read once, so the reads above it
        // are the probes of the candidates the check rejected — read from the cache
        // on a warm build, rendered on a cold one.
        if let Some(clips) = clips
            && complete
            && stats.reads > clips
        {
            figures.push(format!(
                "{} of those reads are probes of candidates the check rejected",
                stats.reads - clips
            ));
        }
        figures.push(match *state(&self.swept) {
            Some(swept) => format!("{} cleared", audio::superseded(swept)),
            None if complete => "the sweep did not finish".into(),
            None => "nothing swept: the sweep waits for a complete corpus".into(),
        });
        figures.push(format!("{:.1} s of synthesis", stats.synth_secs));
        figures.join(" · ")
    }

    fn line(&self, figures: &str) -> String {
        format!(
            "[wake-word build] corpus {}: {figures} · {:.1} s wall · peak {}",
            self.cache.display(),
            self.started.elapsed().as_secs_f64(),
            crate::report::shown(crate::report::peak_rss())
        )
    }

    /// The generation's manifest, stating only what this invocation did: the corpus
    /// it planned, the reads it reused, the clips it rendered, the synthesis time it
    /// spent, the sweep it ran and how it ended. A build that stopped before its plan
    /// carries a null `clips` rather than borrowing an earlier build's figure.
    ///
    /// This is the corpus's own build record, not a measurement report: the next
    /// build reads `generation` to decide whether the cache is usable, and the
    /// figures say what the last build did with it — how many clips it rendered and
    /// swept, how long that took, and why a build that stopped early did. A
    /// measurement never writes one.
    fn manifest(&self, stats: SynthStats, stopped: Option<&str>) -> Value {
        let clips = *state(&self.clips);
        let swept = state(&self.swept).map_or(Value::Null, |entries| json!(entries));
        json!({
            "generation": self.generation,
            "clips": clips,
            "reused": stats.cache_hits,
            "rendered": stats.rendered(),
            "synthesis_secs": stats.synth_secs,
            "swept": swept,
            "wall_secs": self.started.elapsed().as_secs_f64(),
            "stopped": stopped.map_or(Value::Null, |reason| json!({
                "reason": reason,
                "corpus_complete": self.complete.load(Ordering::SeqCst),
            })),
        })
    }
}

/// Render the corpus: the voice selection's probes, then everything the plan lists.
/// The account states the plan and the sweep as it reaches them, so a guard stop
/// anywhere reports what the build had done. Returns the corpus the plan listed and
/// the reads this run made.
fn render(
    cfg: &BenchConfig,
    inputs: &Inputs,
    progress: &Arc<Mutex<Progress>>,
    account: &Account,
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<(usize, SynthStats)> {
    let (asr, tts) = open_pairs(cfg)?;
    let mut synth = open_corpus(
        cfg,
        Arc::clone(&tts),
        inputs.synthesis_hash(),
        audio::Missing::Render,
        progress,
        deadline,
        ceiling,
    )?;
    // Read before the first render, under the lock this build already holds: a record
    // it cannot interpret must refuse now — as a refused input, not a failed render —
    // not after hours of synthesis that no record would then claim.
    let records = audio::read_build_keys(&cfg.cache)?;

    // The selection renders each candidate's two probes: those are part of the
    // generation, and a candidate the check rejects leaves its probes behind for
    // the measurement's own replay of the same decision.
    state(progress).current = "voices";
    let selection = voices::select(&tts, &asr, &mut synth, cfg, deadline, ceiling)?;
    let speakers = selection.speakers();

    // Everything the measurement will ask for. Planning is its own phase so a stop
    // while the request list is built names planning rather than the render that
    // would follow. The probes are already present from the selection, so the loop
    // skips what this run has touched and every clip is counted once; `Synth::pcm`
    // checks the guards per clip, so the loop does not.
    state(progress).current = "plan";
    let requests = audio::plan(&speakers, Phases::ALL);
    account.plan(requests.len());
    state(progress).current = "corpus build";
    for request in &requests {
        if !synth.touched(request) {
            synth.clip(&request.speaker, request.text, request.seed)?;
        }
    }
    let (missing, names) = synth.missing(&requests);
    if missing > 0 {
        anyhow::bail!(
            "{missing} of {} clips are still missing after the build: {}",
            requests.len(),
            names.join(", ")
        );
    }
    // From here the corpus is what a measurement needs: a stop in the sweep below
    // is this invocation's failure, not a generation still short of clips.
    account.whole();

    // The sweep: every cache entry any recorded generation claimed and this one did
    // not produce — the clips and the span cuts alike. It runs once the generation
    // is complete, under the lock, against the record *before* that record is
    // rewritten.
    state(progress).current = "sweep";
    let produced: HashSet<String> = synth
        .live()
        .iter()
        .cloned()
        .chain(selection.spans.iter().filter_map(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        }))
        .collect();
    let (swept, survivors) = audio::sweep(&cfg.cache, &produced, &records);
    account.swept(swept);
    audio::write_build_keys(&cfg.cache, &account.generation, &produced, &survivors)?;

    let stats = state(progress).synthesis;
    eprintln!(
        "[wake-word build] {} voices selected from {} candidates tried, {} replaced",
        selection.voices.len(),
        selection.voices.len() + selection.replacements.len(),
        selection.replacements.len()
    );
    // The plan `finish` states: `Ok` here means it was verified whole, so the
    // completed build's corpus is this figure by construction.
    Ok((requests.len(), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir =
            std::env::temp_dir().join(format!("mahbot-build-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// `reads` clip reads of which `hits` were cache hits: the rest were rendered.
    fn stats(reads: usize, hits: usize) -> SynthStats {
        SynthStats {
            reads,
            cache_hits: hits,
            synth_secs: 1.5,
        }
    }

    /// An account for a generation whose corpus is `clips` long.
    fn account_for(dir: &Path, clips: usize) -> Account {
        let account = Account::new(dir, "gen".into(), std::time::Instant::now());
        account.plan(clips);
        account
    }

    /// A generation's manifest states this invocation only: a second build records
    /// its own reused/rendered counts and synthesis time, not a running total over
    /// the builds before it.
    #[test]
    fn a_manifest_states_this_build_only() {
        let dir = temp_dir("manifest");
        let first = account_for(&dir, 1_956).manifest(
            SynthStats {
                reads: 1_000,
                cache_hits: 0,
                synth_secs: 10.0,
            },
            None,
        );
        audio::write_build_manifest(&dir, &first).expect("a manifest");
        let second = account_for(&dir, 1_956).manifest(
            SynthStats {
                reads: 956,
                cache_hits: 900,
                synth_secs: 5.0,
            },
            None,
        );
        assert_eq!(second["reused"], 900);
        assert_eq!(second["rendered"], 56);
        assert_eq!(second["synthesis_secs"], 5.0);
        assert_eq!(second["clips"], 1_956);
        assert!(second.get("builds").is_none(), "{second}");
        assert!(second.get("built").is_none(), "{second}");
        assert!(second.get("last").is_none(), "{second}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The console has to add up in every case: the corpus is the plan's figure, and
    /// the reads are `reused + rendered`. A cold build renders the whole plan, a
    /// rejected candidate's two probe renders are named apart from it, and a warm
    /// build renders nothing — without any of them contradicting the corpus.
    #[test]
    fn a_build_states_the_corpus_apart_from_its_clip_reads() {
        let dir = temp_dir("cold");
        let account = Account::new(&dir, "gen".into(), std::time::Instant::now());
        assert_eq!(
            account.figures(stats(2, 0)),
            "corpus not stated: the build stopped before its plan · 2 clip reads: 0 reused, 2 \
             rendered · nothing swept: the sweep waits for a complete corpus · 1.5 s of synthesis"
        );

        // Cold, with one rejected candidate: the plan is whole, and the two probe
        // renders outside it are named rather than folded into the corpus.
        let account = account_for(&dir, 1_956);
        account.whole();
        assert_eq!(
            account.figures(stats(1_958, 0)),
            "1956 clips · 1958 clip reads: 0 reused, 1958 rendered · 2 of those reads are probes of \
             candidates the check rejected · the sweep did not finish · 1.5 s of synthesis"
        );

        // Warm: every clip reused, nothing rendered, nothing outside the plan.
        account.swept(Some(0));
        assert_eq!(
            account.figures(stats(1_956, 1_956)),
            "1956 clips · 1956 clip reads: 1956 reused, 0 rendered · 0 superseded entries cleared \
             · 1.5 s of synthesis"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stop has one account: the guard thread and the error boundary both reach
    /// it, and the second must neither print the counts again nor add a second
    /// build to the manifest. A stop after the plan was verified whole leaves a
    /// complete corpus, which the record must not call an incomplete build — and a
    /// stop after the sweep returned states its count instead of claiming the sweep
    /// never got through.
    #[test]
    fn a_stop_is_stated_once_and_a_whole_corpus_is_not_an_incomplete_build() {
        let dir = temp_dir("stop");
        let account = account_for(&dir, 1_956);
        assert!(account.claim(), "the first caller states the stop");
        account.stop(stats(10, 5), "the `corpus build` phase exceeded");
        assert!(
            !account.claim(),
            "the other writer returns instead of stating it twice"
        );
        let written = audio::read_build_manifest(&dir, "gen")
            .expect("the manifest")
            .expect("this generation");
        assert_eq!(
            written["rendered"], 5,
            "one statement of this build's renders: {written}"
        );
        assert_eq!(written["stopped"]["corpus_complete"], false);
        assert!(
            written["swept"].is_null(),
            "a sweep that never ran is not a zero: {written}"
        );

        // A stop inside the sweep: the corpus was whole, and the entries the sweep
        // got through are not known — the manifest says neither, rather than
        // reading as a clean completed build.
        let account = account_for(&dir, 1_956);
        account.whole();
        assert!(account.claim());
        account.stop(stats(20, 1), "the sweep failed");
        let written = audio::read_build_manifest(&dir, "gen")
            .expect("the manifest")
            .expect("this generation");
        assert_eq!(written["stopped"]["corpus_complete"], true);
        assert!(written["swept"].is_null(), "{written}");

        // A stop after the sweep returned: the corpus was whole and the count is
        // known, so neither is stated as unknown.
        let account = account_for(&dir, 1_956);
        account.whole();
        account.swept(Some(7));
        assert!(account.claim());
        account.stop(stats(30, 2), "the figures could not be written");
        let written = audio::read_build_manifest(&dir, "gen")
            .expect("the manifest")
            .expect("this generation");
        assert_eq!(written["swept"], 7, "{written}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A build that failed before rendering anything states its counts but leaves no
    /// manifest of zeroes.
    #[test]
    fn a_build_that_rendered_nothing_leaves_no_manifest() {
        let dir = temp_dir("quiet");
        let account = account_for(&dir, 1_956);
        assert!(account.claim());
        account.stop(stats(0, 0), "a missing model");
        assert!(
            audio::read_build_manifest(&dir, "gen")
                .expect("the manifest")
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A refused input is the shared argument refusal (2); a render that failed is a
    /// failed run (1), so a caller can tell the two apart. The marker carries the
    /// whole context chain as its message, so the console states what was refused.
    #[test]
    fn a_refused_input_is_not_a_failed_render() {
        let refused: anyhow::Error = Refused::new(anyhow::anyhow!("unknown flag `--nope`")).into();
        assert_eq!(
            crate::bench::exit_code(&refused),
            2,
            "a refused input exits 2"
        );
        assert!(
            format!("{refused:#}").contains("unknown flag"),
            "the refusal states its cause: {refused:#}"
        );
        let failed = anyhow::anyhow!("synthesis failed for `hey mahbot`");
        assert_eq!(
            crate::bench::exit_code(&failed),
            1,
            "a failed render exits 1"
        );
    }
}
