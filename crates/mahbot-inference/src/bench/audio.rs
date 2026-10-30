// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bench's audio material: the deterministic guided-prompt DSP
//! conditioning, the noise generators, the VAD gate, and TTS synthesis with its
//! on-disk PCM cache.
//!
//! Every generator is seeded, so a run is reproducible from the models it was
//! given. The calibration pool and the detection set render the same noise
//! profile names at **different seed bands** ([`sets::CALIBRATION_NOISE_SEED_BASE`]
//! against [`sets::DETECTION_NOISE_SEED_BASE`]) — no waveform is both
//! calibration material and measurement material.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use crate::listen::Listener;
use crate::loader::Pair;
use crate::synth::Synthesis;

use super::config::Refused;
use super::voices::Speaker;
use super::{detect, sets};
use crate::guard::{Deadline, MemoryCeiling};
use crate::measurements::hex;

/// A noise generator: its seed band decides which set it renders for.
pub(crate) type NoiseGenerator = fn(u64) -> Vec<f32>;

/// What one run has read from the corpus: how many clips it asked the cache for,
/// how many of those it served, and the wall time it spent generating the rest.
/// A counter of **reads**, not of the corpus: a run that reads one clip twice
/// counts it twice, while the corpus size is what [`plan`] lists.
#[derive(Clone, Copy, Default)]
pub(crate) struct SynthStats {
    pub reads: usize,
    pub cache_hits: usize,
    pub synth_secs: f64,
}

impl SynthStats {
    /// The clips this run rendered, out of the ones it read.
    pub(crate) fn rendered(&self) -> usize {
        self.reads - self.cache_hits
    }
}

/// What a cache miss does: the build renders it, a measurement stops and names it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Missing {
    Render,
    Stop,
}

/// What a measurement tells an engineer who finds a clip missing: the one
/// instruction, worded once, for both the pre-flight and the per-clip refusal.
pub(crate) const BUILD_FIRST: &str = "a measurement never renders, and a corpus build does: \
     `cargo run --release -p mahbot-inference --features bench --bin wake-word-build -- <the same \
     model, voice and corpus inputs>`";

/// One synthesised-and-cached TTS front end. It owns the run's guards so a
/// cache miss — the only phase that can be long — stops at the next clip.
/// The resident-set check runs per clip; the system figures (two subprocesses)
/// stay at the phase boundaries the caller checks.
///
/// One pipeline is resident at a time and it is built on demand: a warm corpus
/// synthesizes nothing, and a run that walks eleven voices drops one voice's
/// graphs before building the next.
pub(crate) struct Synth {
    pair: Arc<Pair>,
    /// The enrolled reference the owner's renders are built from.
    voice: PathBuf,
    cache: PathBuf,
    synthesis_hash: String,
    /// What a cache miss does: the build renders it, a measurement stops and names
    /// what the build owes it, so a measurement never turns into a synthesis.
    missing: Missing,
    /// The pipeline that is resident, under the speaker it was built for:
    /// `None` while nothing has been synthesized this run.
    pipeline: Option<(Option<String>, Synthesis)>,
    /// Every clip key this run touched — hits and renders. With the span cuts the
    /// selection named it is what the build's record claims, and so what the next
    /// build's sweep may reclaim.
    live: HashSet<String>,
    deadline: Deadline,
    ceiling: MemoryCeiling,
    progress: Arc<Mutex<super::Progress>>,
}

impl Synth {
    pub(crate) fn new(
        pair: Arc<Pair>,
        voice: PathBuf,
        cache: PathBuf,
        synthesis_hash: String,
        missing: Missing,
        progress: Arc<Mutex<super::Progress>>,
        deadline: Deadline,
        ceiling: MemoryCeiling,
    ) -> Self {
        Self {
            pair,
            voice,
            cache,
            synthesis_hash,
            missing,
            pipeline: None,
            live: HashSet::new(),
            progress,
            deadline,
            ceiling,
        }
    }

    /// Whether this run renders a miss (the build) or stops on it (a measurement).
    pub(crate) fn renders(&self) -> bool {
        self.missing == Missing::Render
    }

    /// The clip keys this run touched — what the build's record claims beside its
    /// span cuts.
    pub(crate) fn live(&self) -> &HashSet<String> {
        &self.live
    }

    /// How many of `requests` the cache does not hold whole, and the first few names —
    /// the figure a measurement refuses to start on. A torn or truncated entry is not
    /// a hit: the pre-flight reads each entry's header and length, so a broken clip
    /// stops the run before it spends its setup instead of surfacing as a miss at read
    /// time.
    pub(crate) fn missing(&self, requests: &[Request]) -> (usize, Vec<String>) {
        let mut missing = 0;
        let mut first = Vec::new();
        for request in requests {
            let key = self.key(request);
            if is_whole_entry(&self.cache.join(&key)) {
                continue;
            }
            missing += 1;
            if first.len() < 3 {
                first.push(request.label());
            }
        }
        (missing, first)
    }

    /// The one identity of a clip: text, seed, the sound half's rate, the generation
    /// and the voice. The pre-flight, the sweep and the read all go through it, so it
    /// has one composition.
    fn key_of(&self, speaker: &Speaker, text: &str, seed: u64) -> String {
        cache_key(
            text,
            seed,
            crate::synth::SAMPLE_RATE,
            &self.synthesis_hash,
            speaker.cache_id().as_deref(),
        )
    }

    fn key(&self, request: &Request) -> String {
        self.key_of(&request.speaker, request.text, request.seed)
    }

    /// Whether this run has already touched the clip `request` names — a hit or a
    /// render. The build skips those, so one clip is counted once.
    pub(crate) fn touched(&self, request: &Request) -> bool {
        self.live.contains(&self.key(request))
    }

    /// The corpus pre-flight. A measurement never renders: a cache short of what
    /// the run needs stops it before it spends anything, naming the shortfall and
    /// the build that fills it.
    pub(crate) fn require_corpus(&self, requests: &[Request], what: &str) -> Result<()> {
        let (missing, names) = self.missing(requests);
        if missing == 0 {
            return Ok(());
        }
        bail!(
            "the corpus cache is short {missing} of the {} clips in {what} — {BUILD_FIRST}. \
             Missing: {}",
            requests.len(),
            names.join(", ")
        )
    }

    /// One utterance in `speaker`'s voice at the listening rate (16 kHz). The
    /// sound half renders at [`crate::synth::SAMPLE_RATE`]; every listening path
    /// resamples.
    pub(crate) fn clip(&mut self, speaker: &Speaker, text: &str, seed: u64) -> Result<Vec<f32>> {
        let pcm = self.pcm(speaker, text, seed)?;
        Ok(crate::host::resample(
            &pcm,
            crate::synth::SAMPLE_RATE,
            crate::listen::SAMPLE_RATE,
        ))
    }

    /// The synthesised waveform at the sound half's own rate (24 kHz), from the
    /// cache when its key is present.
    fn pcm(&mut self, speaker: &Speaker, text: &str, seed: u64) -> Result<Vec<f32>> {
        // The phase is the one the caller recorded: a stop here has to name a
        // key the report's `phases` map carries (`enrolment`, `recall synthesis`,
        // `non-phrase synthesis`, `voices`), not a generic "synthesis".
        let phase = super::state(&self.progress).current;
        self.deadline.check(phase)?;
        self.ceiling.check_process(phase)?;
        let key = self.key_of(speaker, text, seed);
        let path = self.cache.join(&key);
        if let Some(samples) = read_cache(&path) {
            self.live.insert(key);
            let mut progress = super::state(&self.progress);
            progress.synthesis.reads += 1;
            progress.synthesis.cache_hits += 1;
            return Ok(samples);
        }
        if self.missing == Missing::Stop {
            bail!(
                "the corpus clip `{text}` seed {seed} in {} is not in {} — {BUILD_FIRST}",
                speaker.name(),
                self.cache.display()
            );
        }
        let started = std::time::Instant::now();
        let synthesized = {
            let pipeline = self.pipeline(speaker)?;
            pipeline.synthesize(text, seed)
        };
        let samples = synthesized.with_context(|| {
            format!(
                "synthesis failed for `{text}` seed {seed} in {}",
                speaker.name()
            )
        })?;
        let elapsed = started.elapsed().as_secs_f64();
        // The disk write happens before the progress lock is taken: the guard
        // thread reads its phase and its counters through that lock, and a slow
        // write must not be able to delay the wall-clock stop.
        write_cache(&path, &samples).with_context(|| {
            format!(
                "cannot write the corpus clip `{text}` seed {seed} in {} to {}",
                speaker.name(),
                path.display()
            )
        })?;
        self.live.insert(key);
        let mut progress = super::state(&self.progress);
        progress.synthesis.reads += 1;
        progress.synthesis.synth_secs += elapsed;
        Ok(samples)
    }

    /// The pipeline for `speaker`, built on first use and replaced when the voice
    /// changes. The previous one is dropped before the next is built, so two
    /// checkpoints' graphs are never resident at once.
    fn pipeline(&mut self, speaker: &Speaker) -> Result<&mut Synthesis> {
        let wanted = speaker.cache_id();
        let resident = self
            .pipeline
            .as_ref()
            .is_some_and(|(built, _)| *built == wanted);
        if !resident {
            self.pipeline = None;
            let path = speaker
                .voice()
                .map_or_else(|| self.voice.clone(), |voice| voice.wav.clone());
            let built = Synthesis::new(Arc::clone(&self.pair), &path).with_context(|| {
                format!("cannot build the synthesis path for {}", speaker.name())
            })?;
            self.pipeline = Some((wanted, built));
        }
        Ok(&mut self.pipeline.as_mut().expect("just built").1)
    }
}

/// The file the build leaves beside the corpus: the generation it completed and
/// what the render cost. A measurement states it, so a warm record cannot make the
/// build look like nothing.
pub(crate) const BUILD_MANIFEST: &str = "corpus-build.json";

/// Record what the build just did. `builds`, `built` and `synthesis_secs` are the
/// generation's totals over every build of it, so the recorded cost is not the last
/// invocation's no-op; `last` is that invocation. Written atomically when the build
/// completes, and again when a stop that rendered something claims its account.
pub(crate) fn write_build_manifest(cache: &Path, manifest: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(manifest)
        .context("cannot serialise the corpus build's manifest")?;
    super::write_atomic(&cache.join(BUILD_MANIFEST), |tmp| {
        Ok(std::fs::write(tmp, text.as_bytes())?)
    })
}

/// A whole-number cell of the manifest every build of a generation appends to, in the
/// format that build writes it. The build's running totals and the record's cost row
/// read counts through this one function, so the two cannot disagree.
pub(crate) fn manifest_count(manifest: Option<&serde_json::Value>, key: &str) -> Option<u64> {
    manifest
        .and_then(|manifest| manifest.get(key))
        .and_then(serde_json::Value::as_u64)
}

/// The manifest of the generation `synthesis_hash` names: `None` when the cache
/// carries no manifest or one from another generation. A manifest that exists but
/// cannot be interpreted — unreadable, not JSON, or naming no generation — is an
/// error, never an absent one: writing over it from nothing would reset the
/// generation's build totals and understate the cost the record carries.
pub(crate) fn read_build_manifest(
    cache: &Path,
    synthesis_hash: &str,
) -> Result<Option<serde_json::Value>, Refused> {
    let path = cache.join(BUILD_MANIFEST);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            let why = format!("cannot be read ({error})");
            return Err(Refused::new(anyhow::Error::new(error).context(refusal(
                &path,
                "build manifest",
                why,
            ))));
        }
    };
    let manifest: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| refusal(&path, "build manifest", "does not parse as JSON"))?;
    match manifest
        .get("generation")
        .and_then(serde_json::Value::as_str)
    {
        Some(generation) if generation == synthesis_hash => Ok(Some(manifest)),
        // Another generation's manifest is not this build's business; one that names
        // no generation at all is not any generation's, so it is refused like a
        // corrupt one rather than written over.
        Some(_) => Ok(None),
        None => Err(Refused::new(anyhow!(refusal(
            &path,
            "build manifest",
            "names no generation"
        )))),
    }
}

/// What a corpus file the run cannot interpret is refused with: each holds claims a
/// rewrite from less than all of it would lose — the record every generation's corpus,
/// the manifest a generation's build totals — so it is refused whole, never rewritten
/// from nothing.
fn refusal(path: &Path, noun: &str, why: impl std::fmt::Display) -> String {
    format!(
        "the corpus {noun} {} {why} — it is refused rather than rewritten from nothing, because \
         what it records would be lost; move it aside to start a fresh one",
        path.display()
    )
}

/// Remove this run's own half-written cache entries, left by an earlier launch
/// that died mid-write. Only `*.tmp-<this pid>` is swept: the lock path is an
/// input, so another run can legitimately share this cache directory under a
/// different lock, and its in-flight clips must survive.
pub(crate) fn clean_cache(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let suffix = format!(".tmp-{}", std::process::id());
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().ends_with(&suffix) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Every generation's record of the corpus a completed build produced: one section
/// per generation, headed by `generation <synthesis hash>` and then one cache entry
/// per line — a clip's key or a span cut's file name. A section outlives its build
/// until a later build reclaims the entries it claims, so a corpus superseded by a
/// synthesis change is still claimed by a record.
pub(crate) type Records = BTreeMap<String, HashSet<String>>;

/// The record file's name. It lives beside the corpus and names nothing but cache
/// entries, so the sweep never touches a file no record claims.
pub(crate) const BUILD_KEYS: &str = "corpus-keys.txt";

/// Write `generation`'s section from `produced`, plus every section in `survivors` —
/// the records whose entries could not all be reclaimed. Atomic, and called only
/// after the sweep.
pub(crate) fn write_build_keys(
    cache: &Path,
    generation: &str,
    produced: &HashSet<String>,
    survivors: &Records,
) -> Result<()> {
    let mut records = survivors.clone();
    records
        .entry(generation.to_string())
        .or_default()
        .extend(produced.iter().cloned());
    let mut text = String::new();
    for (generation, entries) in &records {
        text.push_str(&format!("generation {generation}\n"));
        let mut lines: Vec<&str> = entries.iter().map(String::as_str).collect();
        lines.sort_unstable();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
    }
    super::write_atomic(&cache.join(BUILD_KEYS), |tmp| {
        Ok(std::fs::write(tmp, text.as_bytes())?)
    })
}

/// Every generation's record this cache carries; empty when no build has left one.
///
/// A record that exists but cannot be interpreted is an error, never an empty or a
/// partial record: rewriting it from anything less than all of it would drop claims
/// and orphan the corpora they name. A header with no entries is a record, not a
/// broken one — the writer can emit it.
pub(crate) fn read_build_keys(cache: &Path) -> Result<Records, Refused> {
    let path = cache.join(BUILD_KEYS);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Records::new()),
        Err(error) => {
            let why = format!("cannot be read ({error})");
            return Err(Refused::new(
                anyhow::Error::new(error).context(refusal(&path, "record", why)),
            ));
        }
    };
    let mut records = Records::new();
    let mut generation: Option<&str> = None;
    for line in text.lines().filter(|line| !line.is_empty()) {
        let Some(name) = line.strip_prefix("generation ") else {
            let Some(generation) = generation else {
                return Err(Refused::new(anyhow!(refusal(
                    &path,
                    "record",
                    "carries an entry before its first generation header"
                ))));
            };
            records
                .entry(generation.to_string())
                .or_default()
                .insert(line.to_string());
            continue;
        };
        generation = Some(name);
        records.entry(name.to_string()).or_default();
    }
    Ok(records)
}

/// A count of entries the sweep cleared, as both printers state it: `1 superseded
/// entry` / `N superseded entries`.
pub(crate) fn superseded(entries: u64) -> String {
    match entries {
        1 => "1 superseded entry".to_string(),
        entries => format!("{entries} superseded entries"),
    }
}

/// Remove every corpus a recorded build has superseded: the entries any generation's
/// record claims and this build did not produce. Returns how many entries were
/// removed — `None` when the directory, or an entry in it, could not be read — and the
/// records that must survive because an entry they claim is still on disk: a removal
/// that failed keeps its generation recorded, so a later build retries instead of
/// leaking the file. A sweep cut short keeps every record as a survivor, which may
/// re-claim an entry this build already removed; it was not seen, so it is not
/// forgotten, and the next build's sweep settles it. Nothing else is touched: a file
/// no record names is kept, and so is a half-written entry ([`clean_cache`]'s).
pub(crate) fn sweep(
    dir: &Path,
    produced: &HashSet<String>,
    records: &Records,
) -> (Option<u64>, Records) {
    let mut claimed: HashMap<&str, Vec<&str>> = HashMap::new();
    for (generation, entries) in records {
        for name in entries {
            claimed
                .entry(name.as_str())
                .or_default()
                .push(generation.as_str());
        }
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        // Nothing could be examined, so nothing may be forgotten: every record is a
        // survivor and the next build retries.
        return (None, records.clone());
    };
    let mut removed = 0u64;
    let mut survivors = Records::new();
    for entry in entries {
        // A failed read is not an empty entry: its file may be one a record claims,
        // and forgetting that claim would leave the file unreclaimable.
        let Ok(entry) = entry else {
            return (None, records.clone());
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(generations) = claimed.get(name.as_str()) else {
            continue;
        };
        if produced.contains(&name) {
            continue;
        }
        if std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        } else {
            for generation in generations {
                survivors
                    .entry((*generation).to_string())
                    .or_default()
                    .insert(name.clone());
            }
        }
    }
    (Some(removed), survivors)
}

/// Deterministic cache key for one synthesised utterance: text, seed, sample
/// rate and the synthesis model's own hash — both TTS files' digests and the
/// reference voice's, because the voice WAV is part of every prompt.
///
/// A clone voice adds its span's identity after the code hash. The enrolled
/// voice passes none, so its key is the one its cached renders were stored under
/// and they stay valid; no clone can be served another voice's audio — or the
/// owner's.
fn cache_key(
    text: &str,
    seed: u64,
    sample_rate: u32,
    synthesis_hash: &str,
    voice: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update([0u8]);
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(sample_rate.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(synthesis_hash.as_bytes());
    if let Some(voice) = voice {
        hasher.update([0u8]);
        hasher.update(voice.as_bytes());
    }
    hex(&hasher.finalize())
}

/// An 8-byte sample count followed by raw little-endian f32 PCM. Anything that
/// does not carry a count matching its own length — a truncated, half-written or
/// zero-filled entry — is a miss, and the utterance is re-synthesised.
fn read_cache(path: &Path) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    match decode_cache(&data) {
        Some(samples) => Some(samples),
        None => {
            let _ = std::fs::remove_file(path);
            None
        }
    }
}

/// Whether `path` is a whole cache entry, from its header and its file length alone.
/// The pre-flight's version of [`decode_cache`]: a torn entry is a miss before the run
/// starts, without reading the samples it claims. One open, so the header and the
/// length are the same file's even if the entry is replaced concurrently.
fn is_whole_entry(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(meta) = file.metadata() else {
        return false;
    };
    let mut header = [0u8; 8];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let count = u64::from_le_bytes(header);
    count
        .checked_mul(4)
        .and_then(|bytes| 8u64.checked_add(bytes))
        == Some(meta.len())
}

/// The entry's shape, or `None` when the bytes are not one.
fn decode_cache(data: &[u8]) -> Option<Vec<f32>> {
    let count = u64::from_le_bytes(data.get(..8)?.try_into().ok()?);
    let count = usize::try_from(count).ok()?;
    if data.len() != 8 + count.checked_mul(4)? {
        return None;
    }
    Some(
        data[8..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// A cache entry, written through [`super::write_atomic`] — flushed before the rename,
/// so a crash leaves either the old entry or the new one and never a half-written clip.
///
/// The corpus is the run's durable artifact, so a cache that cannot be written
/// is an error: silently re-synthesizing the whole corpus on every run is how the
/// bench spent hours producing nothing reusable.
fn write_cache(path: &Path, samples: &[f32]) -> Result<()> {
    let mut data = Vec::with_capacity(8 + samples.len() * 4);
    data.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for &sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    super::write_atomic(path, |tmp| Ok(std::fs::write(tmp, &data)?))
}

// ── Guided-prompt DSP conditioning ────────────────────────────────────────

/// The conditioning floor's seed band: pink noise at `base + clip index`. Its
/// own band, separate from every phrase and noise seed band.
pub(crate) const CONDITIONING_NOISE_SEED_BASE: u64 = 4000;

/// The four enrolment conditions, by clip index.
#[derive(Clone, Copy)]
enum GuidedPromptGroup {
    Normal,
    Distance,
    Angle,
    Morning,
}

impl GuidedPromptGroup {
    const fn for_clip_index(index: usize) -> Self {
        match index {
            0..=2 => Self::Normal,
            3..=5 => Self::Distance,
            6..=7 => Self::Angle,
            _ => Self::Morning,
        }
    }

    /// The noise floor's SNR (dB). The relationship is chosen so every group's
    /// noise lands at the same absolute RMS as the normal 20 dB floor.
    const fn noise_snr_db(self) -> f32 {
        match self {
            Self::Normal => 20.0,
            Self::Distance => 14.0,
            Self::Angle | Self::Morning => 17.0,
        }
    }
}

/// Apply the guided-prompt conditioning to one raw enrolment clip:
/// - normal: a 20 dB noise floor;
/// - distance: −6 dB + a 3.2 kHz one-pole lowpass + a 14 dB floor;
/// - angle: a −4 dB high-shelf cut at 3 kHz + −3 dB + a 17 dB floor;
/// - morning: a 0.92× speed perturbation + −3 dB + a 2.2 kHz lowpass + a 17 dB
///   floor.
///
/// The floor is seeded pink noise at
/// [`CONDITIONING_NOISE_SEED_BASE`]` + clip index`. Pure arithmetic plus seeded
/// RNG: identical inputs give identical output on every run.
fn condition_enrollment_clip(pcm: &[f32], clip_index: usize) -> Vec<f32> {
    let group = GuidedPromptGroup::for_clip_index(clip_index);
    let conditioned = match group {
        GuidedPromptGroup::Normal => pcm.to_vec(),
        GuidedPromptGroup::Distance => {
            let attenuated = apply_gain(pcm, -6.0);
            one_pole_lowpass(&attenuated, 3200.0, crate::listen::SAMPLE_RATE)
        }
        GuidedPromptGroup::Angle => {
            let tilted = high_shelf_cut(pcm, -4.0, 3000.0, crate::listen::SAMPLE_RATE);
            apply_gain(&tilted, -3.0)
        }
        GuidedPromptGroup::Morning => {
            let slower = speed_perturbation(pcm, crate::listen::SAMPLE_RATE, 0.92);
            let reduced = apply_gain(&slower, -3.0);
            one_pole_lowpass(&reduced, 2200.0, crate::listen::SAMPLE_RATE)
        }
    };
    add_noise(
        &conditioned,
        group.noise_snr_db(),
        CONDITIONING_NOISE_SEED_BASE + clip_index as u64,
    )
}

/// Deterministic first-order IIR lowpass: `y[n] = y[n-1] + α(x[n] − y[n-1])`,
/// `α = 1 − exp(−2π fc / fs)`.
fn one_pole_lowpass(pcm: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    if cutoff_hz <= 0.0 || cutoff_hz >= sample_rate as f32 * 0.5 {
        return pcm.to_vec();
    }
    let alpha = 1.0 - (-2.0 * core::f32::consts::PI * cutoff_hz / sample_rate as f32).exp();
    let mut y = 0.0f32;
    pcm.iter()
        .map(|&x| {
            y += alpha * (x - y);
            y
        })
        .collect()
}

/// Deterministic first-order high-shelf cut (RBJ cookbook, S = 1): attenuates
/// frequencies above `fc_hz` by `gain_db`.
fn high_shelf_cut(pcm: &[f32], gain_db: f32, fc_hz: f32, sample_rate: u32) -> Vec<f32> {
    if gain_db.abs() < 1e-6 || fc_hz <= 0.0 {
        return pcm.to_vec();
    }
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * core::f32::consts::PI * fc_hz / sample_rate as f32;
    let alpha = w0.sin() / 2.0 * core::f32::consts::SQRT_2;
    let cos_w0 = w0.cos();
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let inv_a0 = 1.0 / a0;
    let (b0, b1, b2, a1, a2) = (
        b0 * inv_a0,
        b1 * inv_a0,
        b2 * inv_a0,
        a1 * inv_a0,
        a2 * inv_a0,
    );
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;
    pcm.iter()
        .map(|&x| {
            let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            y
        })
        .collect()
}

// ── Gain, noise and perturbance primitives ────────────────────────────────

/// Fixed gain: `10^(gain_db / 20)`. Deterministic.
fn apply_gain(pcm: &[f32], gain_db: f32) -> Vec<f32> {
    let amp = 10.0_f32.powf(gain_db / 20.0);
    pcm.iter().map(|&s| s * amp).collect()
}

/// Noise colours the negative recipe mixes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NoiseColor {
    Pink,
    Brown,
}

/// Pink-noise alias of [`add_noise_color`], the enrolment recipe's floor.
fn add_noise(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    add_noise_color(pcm, snr_db, NoiseColor::Pink, seed)
}

/// Add colour noise at the given SNR, deterministic from `seed`. The noise is
/// scaled to the signal's own RMS, so the SNR is the level relationship.
fn add_noise_color(pcm: &[f32], snr_db: f32, color: NoiseColor, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let signal_rms = compute_rms(pcm).max(1e-10);
    let noise: Vec<f32> = match color {
        NoiseColor::Pink => generate_pink_noise(pcm.len(), &mut rng),
        NoiseColor::Brown => {
            let mut acc = 0.0f32;
            (0..pcm.len())
                .map(|_| {
                    acc = 0.999 * acc + (rng.random::<f32>() * 2.0 - 1.0);
                    acc
                })
                .collect()
        }
    };
    let noise_rms_target = signal_rms * 10.0_f32.powf(-snr_db / 20.0);
    let noise_rms_current = compute_rms(&noise).max(1e-10);
    let scale = noise_rms_target / noise_rms_current;
    pcm.iter()
        .zip(noise.iter())
        .map(|(&s, &n)| (s + n * scale).clamp(-1.0, 1.0))
        .collect()
}

/// Root mean square; `0.0` for empty input.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Pink noise (1/f) via Voss–McCartney, high-pass delta variant, normalized to
/// unit RMS.
fn generate_pink_noise(len: usize, mut rng: impl Rng) -> Vec<f32> {
    const NUM_OCTAVES: usize = 16;
    let mut values = [0.0f32; NUM_OCTAVES];
    let mut outputs = [0.0f32; NUM_OCTAVES];
    let mut sample_count = 0u64;
    let mut noise = Vec::with_capacity(len);

    for _ in 0..len {
        sample_count += 1;
        let mut sum = 0.0;
        for octave in 0..NUM_OCTAVES {
            if sample_count.is_multiple_of(1u64 << octave) {
                values[octave] = rng.random::<f32>() * 2.0 - 1.0;
            }
            let new_val = values[octave];
            let delta = new_val - outputs[octave];
            outputs[octave] = new_val;
            sum += delta;
        }
        noise.push(sum);
    }

    let rms = compute_rms(&noise).max(1e-10);
    for sample in &mut noise {
        *sample /= rms;
    }
    noise
}

/// Both Box–Muller branches from two uniform draws in `(0, 1]`.
fn gaussian_pair_from_uniforms(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// A standard-normal pair with the draws floored at [`f32::EPSILON`] rather
/// than re-rolled — the bench's exact draw sequence.
fn sample_gaussian_pair_clamped(rng: &mut impl Rng) -> (f32, f32) {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
    gaussian_pair_from_uniforms(u1, u2)
}

/// Speed perturbation by time-domain resampling: both speed and pitch move.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn speed_perturbation(samples: &[f32], sample_rate: u32, factor: f32) -> Vec<f32> {
    if samples.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return samples.to_vec();
    }
    let effective_rate = (sample_rate as f32 * factor) as u32;
    resample_audio(samples, effective_rate, sample_rate)
}

/// Linear-interpolation resample with a three-tap anti-aliasing prefilter on
/// downsampling — the arithmetic the enrolment conditioning was measured with.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    let filtered: Vec<f32> = if from_rate > to_rate && samples.len() >= 3 {
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0] * 0.75 + samples[1] * 0.25);
        for i in 1..samples.len() - 1 {
            out.push(samples[i - 1] * 0.25 + samples[i] * 0.5 + samples[i + 1] * 0.25);
        }
        out.push(samples[samples.len() - 2] * 0.25 + samples[samples.len() - 1] * 0.75);
        out
    } else {
        samples.to_vec()
    };

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < filtered.len() {
            output.push(
                (f64::from(filtered[src_idx]) * (1.0 - frac)
                    + f64::from(filtered[src_idx + 1]) * frac) as f32,
            );
        } else if src_idx < filtered.len() {
            output.push(filtered[src_idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}

// ── Noise profiles ────────────────────────────────────────────────────────

/// White uniform noise in [-1, 1].
pub(crate) fn white_uniform_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..sets::NOISE_LEN)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

/// White Gaussian noise (Box–Muller), scaled into [-1, 1].
pub(crate) fn white_gaussian_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(sets::NOISE_LEN);
    let mut i = 0;
    while i < sets::NOISE_LEN {
        let (z1, z2) = sample_gaussian_pair_clamped(&mut rng);
        samples.push((z1 * 0.333).clamp(-1.0, 1.0));
        if i + 1 < sets::NOISE_LEN {
            samples.push((z2 * 0.333).clamp(-1.0, 1.0));
        }
        i += 2;
    }
    samples
}

/// Pink noise profile.
pub(crate) fn pink_noise(seed: u64) -> Vec<f32> {
    generate_pink_noise(sets::NOISE_LEN, StdRng::seed_from_u64(seed))
}

/// Brown noise: a leaky integration of white noise.
pub(crate) fn brown_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(sets::NOISE_LEN);
    let mut prev = 0.0;
    for _ in 0..sets::NOISE_LEN {
        let white: f32 = rng.random::<f32>() * 2.0 - 1.0;
        prev = (prev + white * 0.125) * 0.98;
        samples.push(prev.clamp(-1.0, 1.0));
    }
    samples
}

/// A 200 Hz tonal hum under broadband noise — a distant-speech-like floor.
pub(crate) fn mixed_speech_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(sets::NOISE_LEN);
    for i in 0..sets::NOISE_LEN {
        let t = i as f32 / crate::listen::SAMPLE_RATE as f32;
        let tone = (2.0 * core::f32::consts::PI * 200.0 * t).sin() * 0.15;
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        samples.push((tone + noise * 0.85).clamp(-1.0, 1.0));
    }
    samples
}

/// Blue noise: a first-order difference of white noise.
pub(crate) fn blue_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prev: f32 = 0.0;
    (0..sets::NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let blue = (white - prev) * 0.5;
            prev = white;
            blue.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Violet noise: a second-order difference of white noise.
pub(crate) fn violet_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prev1 = 0.0f32;
    let mut prev2 = 0.0f32;
    (0..sets::NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let violet = (white - 2.0 * prev1 + prev2) * 0.25;
            prev2 = prev1;
            prev1 = white;
            violet.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Low-frequency rumble: white noise through a strong one-pole lowpass.
pub(crate) fn low_freq_rumble(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut low: f32 = 0.0;
    let alpha = 0.95;
    (0..sets::NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            low = low * alpha + white * (1.0 - alpha);
            low.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Amplitude-modulated noise (4 Hz tremolo).
pub(crate) fn modulated_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(sets::NOISE_LEN);
    for i in 0..sets::NOISE_LEN {
        let t = i as f32 / crate::listen::SAMPLE_RATE as f32;
        let modulator = (2.0 * core::f32::consts::PI * 4.0 * t).sin() * 0.5 + 0.5;
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        samples.push((noise * modulator).clamp(-1.0, 1.0));
    }
    samples
}

/// High-frequency hiss: a one-pole high-pass above roughly 4 kHz.
pub(crate) fn high_freq_hiss(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prev: f32 = 0.0;
    (0..sets::NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let hiss = 0.9 * (white - prev) + 0.8 * prev;
            prev = white;
            hiss.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Impulse bursts: sparse broadband clicks.
pub(crate) fn impulse_burst_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..sets::NOISE_LEN)
        .map(|_| {
            if rng.random::<f32>() < 0.02 {
                (rng.random::<f32>() * 2.0 - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// A slow frequency sweep mixed with white noise.
pub(crate) fn oscillating_tone_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..sets::NOISE_LEN)
        .map(|i| {
            let t = i as f32 / crate::listen::SAMPLE_RATE as f32;
            let freq = 200.0 + 300.0 * (2.0 * core::f32::consts::PI * 0.5 * t).sin();
            let tone = (2.0 * core::f32::consts::PI * freq * t).sin() * 0.3;
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (tone + noise * 0.5).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Sparse high-amplitude transients over a quiet floor.
pub(crate) fn crackle_static_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prev: f32 = 0.0;
    (0..sets::NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let crackle = if rng.random::<f32>() < 0.005 {
                white * 3.0
            } else {
                white * 0.1
            };
            let hi = 0.9 * (crackle - prev) + 0.8 * prev;
            prev = crackle;
            hi.clamp(-1.0, 1.0)
        })
        .collect()
}

/// White noise gated by a slow square envelope.
pub(crate) fn pulsed_broadband_noise(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..sets::NOISE_LEN)
        .map(|i| {
            let t = i as f32 / crate::listen::SAMPLE_RATE as f32;
            let gate = if (t * 2.0).sin() > 0.0 { 1.0 } else { 0.0 };
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (noise * gate).clamp(-1.0, 1.0)
        })
        .collect()
}

// ── Clip construction ─────────────────────────────────────────────────────

/// One measured clip, named by the stream it belongs to.
pub(crate) struct Clip {
    pub samples: Vec<f32>,
    /// Unique across the run: the stream it belongs to is part of it, so a
    /// missed or fired list never needs a second lookup to name its voice.
    pub label: String,
}

/// The category a speech negative's row reports it under. Only the negative side
/// has categories: positives and probes are not grouped.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Category {
    Confusable,
    Unrelated,
    PhraseInPassing,
}

impl Category {
    /// The three categories a negative row states, in report order.
    pub(crate) const SPEECH: [Self; 3] = [Self::Confusable, Self::Unrelated, Self::PhraseInPassing];

    pub(crate) const fn key(self) -> &'static str {
        match self {
            Self::Confusable => "confusable",
            Self::Unrelated => "unrelated",
            Self::PhraseInPassing => "phrase_in_passing",
        }
    }

    /// The words the record's tables use for the category.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Confusable => "phonetic near-misses",
            Self::Unrelated => "unrelated sentences",
            Self::PhraseInPassing => "wake phrase inside a sentence",
        }
    }

    /// This category's slot in [`Self::SPEECH`].
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Confusable => 0,
            Self::Unrelated => 1,
            Self::PhraseInPassing => 2,
        }
    }
}

/// One clip a measurement needs: the speaker it is rendered in, the text, the
/// seed, the tag its label carries, and — for the owner's non-wake speech — the
/// SNR its second, noisy embedding is scored at.
///
/// One list drives the corpus build and the measurement's own clip construction,
/// so a seed, a label or a phrase cannot drift between what is rendered and what
/// is scored. The measurement never renders: it asks the cache, and a miss is a
/// stop naming the clip the build owes it.
pub(crate) struct Request {
    pub speaker: Speaker,
    pub text: &'static str,
    pub seed: u64,
    pub tag: &'static str,
    /// `Some(snr_db)` when the clip contributes a second embedding under brown
    /// noise at that SNR — the owner-negative cell of the calibration pool.
    pub noise_snr_db: Option<f32>,
}

impl Request {
    /// The clip's name inside its stream: `voice03/confusable_s800`.
    pub(crate) fn label(&self) -> String {
        format!("{}/{}_s{}", self.speaker.name(), self.tag, self.seed)
    }
}

/// `seeds_per_phrase` requests per phrase, seeded `seed_base + phrase·n + variant`.
fn phrase_requests(
    speaker: &Speaker,
    phrases: &[&'static str],
    seeds_per_phrase: usize,
    seed_base: u64,
    tag: &'static str,
) -> Vec<Request> {
    let mut requests = Vec::with_capacity(phrases.len() * seeds_per_phrase);
    for (index, phrase) in phrases.iter().enumerate() {
        for variant in 0..seeds_per_phrase {
            requests.push(Request {
                speaker: speaker.clone(),
                text: phrase,
                seed: seed_base + (index * seeds_per_phrase + variant) as u64,
                tag,
                noise_snr_db: None,
            });
        }
    }
    requests
}

/// One seeded rendering per phrase, seeded `seed_base + index`.
fn one_per_phrase(
    speaker: &Speaker,
    phrases: &[&'static str],
    seed_base: u64,
    tag: &'static str,
) -> Vec<Request> {
    phrase_requests(speaker, phrases, 1, seed_base, tag)
}

/// The two intelligibility probes of one voice: the list the build renders, the
/// measurement pre-flights and the check scores.
pub(crate) fn probe_requests(speaker: &Speaker) -> Vec<Request> {
    sets::INTELLIGIBILITY_SENTENCES
        .iter()
        .enumerate()
        .map(|(index, text)| Request {
            speaker: speaker.clone(),
            text,
            seed: sets::INTELLIGIBILITY_SEED_BASE + index as u64,
            tag: "probe",
            noise_snr_db: None,
        })
        .collect()
}

/// The enrolment: ten seeded renders of the wake phrase in the enrolled voice.
pub(crate) fn enrolment_variants() -> Vec<Request> {
    (0..sets::NUM_ENROLLMENT_VARIANTS)
        .map(|i| Request {
            speaker: Speaker::Enrolled,
            text: sets::WAKE_PHRASE,
            seed: sets::ENROLLMENT_SEED_BASE + i as u64,
            tag: "enroll",
            noise_snr_db: None,
        })
        .collect()
}

/// The calibration pool: the enrolled voice's non-wake speech (scored twice, the
/// second time under brown noise at [`sets::OWNER_NEGATIVE_SNR_DB`]), then the
/// confusable and unrelated bands.
pub(crate) fn calibration_requests() -> Vec<Request> {
    let mut requests = phrase_requests(
        &Speaker::Enrolled,
        sets::OWNER_NEGATIVE_PHRASES,
        sets::OWNER_NEGATIVE_SEEDS_PER_PHRASE,
        sets::OWNER_NEGATIVE_SEED_BASE,
        "owner",
    );
    for request in &mut requests {
        request.noise_snr_db = Some(sets::OWNER_NEGATIVE_SNR_DB);
    }
    requests.extend(phrase_requests(
        &Speaker::Enrolled,
        sets::CONFUSABLE_PHRASES,
        sets::CONFUSABLE_SEEDS_PER_PHRASE,
        sets::CONFUSABLE_SEED_BASE,
        "confusable",
    ));
    requests.extend(phrase_requests(
        &Speaker::Enrolled,
        sets::UNRELATED_PHRASES,
        sets::UNRELATED_SEEDS_PER_PHRASE,
        sets::UNRELATED_SEED_BASE,
        "unrelated",
    ));
    requests
}

/// Everything the enrolment phase reads: its ten variants and its calibration pool.
pub(crate) fn enrolment_requests() -> Vec<Request> {
    let mut requests = enrolment_variants();
    requests.extend(calibration_requests());
    requests
}

/// One voice's positives: 40 bare wake-phrase renders and 10 with a command.
pub(crate) fn positive_requests(speaker: &Speaker) -> Vec<Request> {
    let mut requests: Vec<Request> = (0..sets::HELD_OUT_WAKE_ONLY_CLIPS)
        .map(|i| Request {
            speaker: speaker.clone(),
            text: sets::WAKE_PHRASE,
            seed: sets::HELD_OUT_SEED_BASE + i as u64,
            tag: "heldout_wake",
            noise_snr_db: None,
        })
        .collect();
    requests.extend(one_per_phrase(
        speaker,
        sets::COMMAND_PLUS_SPEECH_PHRASES,
        sets::COMMAND_PLUS_SPEECH_SEED_BASE,
        "command_other",
    ));
    requests
}

/// One voice's speech negatives, in report order — the two confusable bands, the
/// two unrelated bands, then phrase-in-passing — each tagged with the category its
/// row states.
pub(crate) fn negative_requests(speaker: &Speaker) -> Vec<(Category, Request)> {
    let mut requests: Vec<(Category, Request)> = Vec::with_capacity(sets::SPEECH_NEGATIVE_CLIPS);
    for &(band, tag) in sets::CONFUSABLE_BANDS {
        requests.extend(category_band(
            speaker,
            sets::CONFUSABLE_PHRASES,
            band,
            tag,
            Category::Confusable,
        ));
    }
    for &(band, tag) in sets::UNRELATED_BANDS {
        requests.extend(category_band(
            speaker,
            sets::UNRELATED_PHRASES,
            band,
            tag,
            Category::Unrelated,
        ));
    }
    requests.extend(category_band(
        speaker,
        sets::PHRASE_IN_PASSING_PHRASES,
        sets::PHRASE_IN_PASSING_SEED_BASE,
        "phrase_in_passing",
        Category::PhraseInPassing,
    ));
    requests
}

/// One seeded rendering per phrase, each tagged with the category its row states.
fn category_band(
    speaker: &Speaker,
    phrases: &[&'static str],
    seed_base: u64,
    tag: &'static str,
    category: Category,
) -> Vec<(Category, Request)> {
    one_per_phrase(speaker, phrases, seed_base, tag)
        .into_iter()
        .map(|request| (category, request))
        .collect()
}

/// Everything a measurement over `speakers` needs: the enrolled voice's enrolment
/// and calibration clips, and — for the phases it runs — each voice's probes,
/// positives and speech negatives. A screening run asks the corpus only for what it
/// reads.
pub(crate) fn plan(speakers: &[Speaker], phases: super::config::Phases) -> Vec<Request> {
    let mut requests = enrolment_requests();
    for speaker in speakers {
        requests.extend(probe_requests(speaker));
        if phases.recall {
            requests.extend(positive_requests(speaker));
        }
        if phases.non_phrase {
            requests.extend(
                negative_requests(speaker)
                    .into_iter()
                    .map(|(_, request)| request),
            );
        }
    }
    requests
}

/// The enrolled voice's enrolment clips, conditioned by
/// [`condition_enrollment_clip`].
pub(crate) fn enrollment_clips(synth: &mut Synth) -> Result<Vec<Clip>> {
    let requests = enrolment_variants();
    let mut clips = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let pcm = synth.clip(&request.speaker, request.text, request.seed)?;
        clips.push(Clip {
            samples: condition_enrollment_clip(&pcm, index),
            label: request.label(),
        });
    }
    Ok(clips)
}

/// One clip per request, in request order.
pub(crate) fn clips(synth: &mut Synth, requests: &[Request]) -> Result<Vec<Clip>> {
    requests
        .iter()
        .map(|request| {
            Ok(Clip {
                samples: synth.clip(&request.speaker, request.text, request.seed)?,
                label: request.label(),
            })
        })
        .collect()
}

/// One voice's positives.
pub(crate) fn recall_clips(synth: &mut Synth, speaker: &Speaker) -> Result<Vec<Clip>> {
    let clips = clips(synth, &positive_requests(speaker))?;
    anyhow::ensure!(
        clips.len() == sets::POSITIVE_CLIPS,
        "the positive set built {} clips, not the pinned {}",
        clips.len(),
        sets::POSITIVE_CLIPS
    );
    Ok(clips)
}

/// One voice's speech negatives, each with the category its row reports it under.
pub(crate) fn speech_negatives(
    synth: &mut Synth,
    speaker: &Speaker,
) -> Result<Vec<(Category, Clip)>> {
    let requests = negative_requests(speaker);
    let mut clips = Vec::with_capacity(requests.len());
    for (category, request) in &requests {
        clips.push((
            *category,
            Clip {
                samples: synth.clip(&request.speaker, request.text, request.seed)?,
                label: request.label(),
            },
        ));
    }
    anyhow::ensure!(
        clips.len() == sets::SPEECH_NEGATIVE_CLIPS,
        "a voice's speech negatives built {} clips, not the pinned {}",
        clips.len(),
        sets::SPEECH_NEGATIVE_CLIPS
    );
    Ok(clips)
}

/// The non-speech negatives, in a fixed order: digital silence, then the noise
/// profiles. Their stream has no voice.
pub(crate) fn non_speech_clips() -> Result<Vec<Clip>> {
    let mut clips = Vec::with_capacity(sets::NON_SPEECH_CLIPS);
    for &(label, len) in sets::SILENCE_DURATIONS {
        clips.push(Clip {
            samples: vec![0.0f32; len],
            label: format!("non-speech/{label}"),
        });
    }
    for (i, (label, generator)) in sets::all_noise_profiles().enumerate() {
        let seed = sets::DETECTION_NOISE_SEED_BASE + i as u64;
        clips.push(Clip {
            samples: generator(seed),
            label: format!("non-speech/{label}_s{seed}"),
        });
    }
    anyhow::ensure!(
        clips.len() == sets::NON_SPEECH_CLIPS,
        "the non-speech set built {} clips, not the pinned {}",
        clips.len(),
        sets::NON_SPEECH_CLIPS
    );
    Ok(clips)
}

// ── Calibration negatives ────────────────────────────────────────────────

/// The calibration pool's embeddings, from [`calibration_requests`]: one per
/// request, plus the noisy second embedding of every band that names an SNR. A
/// clip the VAD gate finds no speech in is skipped and named — a quiet render is a
/// property of the model and the seed, not a reason to abandon the run.
pub(crate) fn calibration_negatives(
    synth: &mut Synth,
    listener: &mut Listener,
    skipped: &mut Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::new();
    for request in calibration_requests() {
        let pcm = synth.clip(&request.speaker, request.text, request.seed)?;
        let speech = detect::vad_gate(&pcm, &mut earshot::Detector::default());
        if speech.is_empty() {
            let label = request.label();
            eprintln!("[wake-word] {label}: no VAD-positive speech — skipped");
            skipped.push(label);
            continue;
        }
        embeddings.push(listener.embed(&speech)?);
        if let Some(snr_db) = request.noise_snr_db {
            let noisy = add_noise_color(&speech, snr_db, NoiseColor::Brown, request.seed);
            embeddings.push(listener.embed(&noisy)?);
        }
    }
    Ok(embeddings)
}

/// Ambient negatives: every training noise profile at full level and at −6 dB,
/// encoded directly (no VAD gate — the profiles are the ambient floor). The
/// profiles render at the calibration seed band.
pub(crate) fn ambient_negatives(listener: &mut Listener) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(sets::NOISE_PROFILES.len() * 2);
    for (i, (_, generator)) in sets::NOISE_PROFILES.iter().enumerate() {
        let seed = sets::CALIBRATION_NOISE_SEED_BASE + i as u64;
        let raw = generator(seed);
        embeddings.push(listener.embed(&raw)?);
        embeddings.push(listener.embed(&apply_gain(&raw, -6.0))?);
    }
    Ok(embeddings)
}

/// Reject a calibration pool that produced nothing: a silently empty pool
/// would calibrate a weaker enrolment than the report claims.
pub(crate) fn require_negatives(negatives: &[Vec<f32>]) -> Result<()> {
    if negatives.is_empty() {
        bail!("the negative calibration pool is empty — no threshold can be derived");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mahbot-wake-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// The cache is shared by every run that shares the directory, and the lock
    /// path is an input, so two runs can differ there. Clean-up may only remove
    /// this process's own half-written entries.
    #[test]
    fn clean_up_leaves_another_runs_in_flight_clip_alone() {
        let dir = std::env::temp_dir().join(format!("mahbot-clean-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let mine = dir.join(format!("a.f32.tmp-{}", std::process::id()));
        let theirs = dir.join("a.f32.tmp-999999");
        let clip = dir.join("a.f32");
        for path in [&mine, &theirs, &clip] {
            std::fs::write(path, b"x").expect("a temp file");
        }

        clean_cache(&dir);

        assert!(!mine.exists(), "this run's leftover is swept");
        assert!(theirs.exists(), "another run's in-flight clip survives");
        assert!(clip.exists(), "a finished clip survives");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn conditioning_is_deterministic_and_grouped_by_clip_index() {
        let pcm = ramp(4000);
        for i in 0..sets::NUM_ENROLLMENT_VARIANTS {
            assert_eq!(
                condition_enrollment_clip(&pcm, i),
                condition_enrollment_clip(&pcm, i),
                "clip {i}"
            );
        }
        // The four groups render different variants of the same clip.
        let normal = condition_enrollment_clip(&pcm, 0);
        let distance = condition_enrollment_clip(&pcm, 3);
        let angle = condition_enrollment_clip(&pcm, 6);
        let morning = condition_enrollment_clip(&pcm, 8);
        assert_ne!(normal, distance);
        assert_ne!(normal, angle);
        assert_ne!(normal, morning);
    }

    #[test]
    fn the_noise_profiles_are_seeded() {
        for (label, generator) in sets::all_noise_profiles() {
            let a = generator(sets::DETECTION_NOISE_SEED_BASE);
            let b = generator(sets::DETECTION_NOISE_SEED_BASE);
            let c = generator(sets::CALIBRATION_NOISE_SEED_BASE);
            assert_eq!(a, b, "{label} is not deterministic");
            assert_ne!(a, c, "{label} ignores its seed band");
            assert_eq!(a.len(), sets::NOISE_LEN);
            assert!(a.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn the_cache_key_covers_text_seed_rate_model_and_voice() {
        let base = cache_key("hey mahbot", 7, 24_000, "model", None);
        assert_eq!(base, cache_key("hey mahbot", 7, 24_000, "model", None));
        assert_ne!(base, cache_key("hey mahbot", 8, 24_000, "model", None));
        assert_ne!(base, cache_key("hey mahbot", 7, 16_000, "model", None));
        assert_ne!(base, cache_key("hey mahbot ", 7, 24_000, "model", None));
        assert_ne!(base, cache_key("hey mahbot", 7, 24_000, "other", None));
        assert_eq!(base.len(), 64);

        // The enrolled voice's key is byte for byte the historical one, so the
        // renders already cached under it stay valid; a clone adds its own span
        // and no two voices share an entry.
        let clone_a = cache_key("hey mahbot", 7, 24_000, "model", Some("a:0.00:15.00"));
        let clone_b = cache_key("hey mahbot", 7, 24_000, "model", Some("b:0.00:15.00"));
        assert_ne!(base, clone_a);
        assert_ne!(clone_a, clone_b);
        assert_eq!(
            clone_a,
            cache_key("hey mahbot", 7, 24_000, "model", Some("a:0.00:15.00"))
        );
    }

    #[test]
    fn the_cache_round_trips_and_rejects_a_torn_file() {
        let dir = temp_dir("cache");
        let path = dir.join("entry");
        let samples = vec![0.0f32, 0.5, -0.25, 1.0];
        write_cache(&path, &samples).expect("write");
        assert!(is_whole_entry(&path), "a whole entry passes the pre-flight");
        assert_eq!(read_cache(&path).expect("hit"), samples);

        std::fs::write(&path, [0u8; 6]).expect("torn");
        assert!(!is_whole_entry(&path), "a truncated entry is a miss");
        assert!(read_cache(&path).is_none());
        assert!(!path.exists(), "a torn entry is deleted");

        // A header whose count disagrees with the file is not whole either.
        let mut lying = 8u64.to_le_bytes().to_vec();
        lying.extend_from_slice(&1.0f32.to_le_bytes());
        lying.extend_from_slice(&2.0f32.to_le_bytes());
        std::fs::write(&path, &lying).expect("short for its count");
        assert!(
            !is_whole_entry(&path),
            "a count longer than the file is a miss"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The record the sweep reads back: a completed build's section survives for its
    /// own generation, a superseded generation's entries are reclaimed only by a
    /// completed build, and a record written after that reclaim keeps the survivor.
    #[test]
    fn the_build_record_survives_for_its_generation_and_a_superseded_one_is_reclaimed() {
        let dir = temp_dir("keys");
        let old_clip = "a".repeat(64);
        let new_clip = "b".repeat(64);
        let span = "voice-0123456789abcdef-11.40-15.00-16000.wav".to_string();
        for name in [&old_clip, &new_clip, &span] {
            std::fs::write(dir.join(name), b"x").expect("a cache entry");
        }
        let first = HashSet::from([old_clip.clone(), span.clone()]);
        write_build_keys(&dir, "gen-1", &first, &Records::new()).expect("write the record");
        assert_eq!(
            read_build_keys(&dir).expect("the record"),
            Records::from([("gen-1".to_string(), first.clone())]),
            "a completed build's record survives for its own generation"
        );

        // The next build of another generation produces the shared span and its own
        // clip; the sweep removes what only the earlier generation claimed. A build
        // killed here would leave the file untouched, and the next build would see
        // the same record.
        let produced = HashSet::from([new_clip.clone(), span.clone()]);
        let records = read_build_keys(&dir).expect("the record");
        let (removed, survivors) = sweep(&dir, &produced, &records);
        assert_eq!(removed, Some(1), "the superseded clip is reclaimed");
        assert!(
            survivors.is_empty(),
            "nothing was left behind: {survivors:?}"
        );
        assert_eq!(
            read_build_keys(&dir).expect("the record"),
            records,
            "the sweep writes nothing"
        );
        write_build_keys(&dir, "gen-2", &produced, &survivors).expect("write the record");
        assert_eq!(
            read_build_keys(&dir).expect("the record"),
            Records::from([("gen-2".to_string(), produced)]),
            "the reclaimed generation's section goes with its entries"
        );
        assert!(dir.join(&span).is_file(), "the shared span stays");

        // A directory that cannot be read reclaims nothing and forgets nothing.
        let unreadable = dir.join("not-a-directory");
        std::fs::write(&unreadable, b"x").expect("a file where a directory is expected");
        let kept = Records::from([("gen-1".to_string(), HashSet::from([old_clip.clone()]))]);
        assert_eq!(
            sweep(&unreadable, &HashSet::new(), &kept),
            (None, kept.clone()),
            "nothing examined, nothing forgotten"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An entry whose removal fails stays recorded, so the next build retries it
    /// instead of the file leaking. `remove_file` refuses a directory, which makes
    /// the unlink fail without depending on permissions.
    #[test]
    fn an_entry_that_cannot_be_removed_stays_recorded() {
        let dir = temp_dir("stuck");
        let stuck = "a".repeat(64);
        std::fs::create_dir_all(dir.join(&stuck)).expect("an entry that cannot be unlinked");
        let records = Records::from([("gen-1".to_string(), HashSet::from([stuck.clone()]))]);
        let (removed, survivors) = sweep(&dir, &HashSet::new(), &records);
        assert_eq!(removed, Some(0), "the entry is still there");
        assert_eq!(
            survivors, records,
            "its generation stays recorded for the next build"
        );
        // The rewrite carries the survivor, so the claim is not lost with the sweep.
        write_build_keys(&dir, "gen-2", &HashSet::new(), &survivors).expect("write the record");
        assert_eq!(
            read_build_keys(&dir).expect("the record"),
            Records::from([
                ("gen-1".to_string(), HashSet::from([stuck.clone()])),
                // A section with no entries is a record of its own: this build
                // produced nothing, because the only entry it had was the stuck one.
                ("gen-2".to_string(), HashSet::new()),
            ]),
            "the surviving claim is on disk"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record that cannot be interpreted is refused rather than read as empty or
    /// partial: a build that read it as less than it is would rewrite it with only
    /// what it understood and orphan every corpus the lost lines named. A header
    /// with no entries is a record — the writer can emit one — and is not refused.
    #[test]
    fn a_record_that_cannot_be_interpreted_is_refused_whole() {
        let dir = temp_dir("record");
        assert!(
            read_build_keys(&dir).expect("no record yet").is_empty(),
            "a cache with no record records nothing"
        );
        std::fs::write(dir.join(BUILD_KEYS), b"").expect("an empty record");
        assert!(
            read_build_keys(&dir).expect("an empty record").is_empty(),
            "an empty file records nothing"
        );
        // The writer emits this for a section with no entries; the reader accepts it.
        write_build_keys(&dir, "gen-2", &HashSet::new(), &Records::new()).expect("a bare section");
        assert_eq!(
            read_build_keys(&dir).expect("a header with no entries"),
            Records::from([("gen-2".to_string(), HashSet::new())]),
            "a section with no entries is a record, not a broken one"
        );
        // An entry before the first header belongs to no generation: the record is
        // refused whole rather than dropped line by line.
        std::fs::write(
            dir.join(BUILD_KEYS),
            format!("{}\ngeneration gen-2\n", "a".repeat(64)),
        )
        .expect("an entry before its header");
        let error = read_build_keys(&dir).expect_err("a leading entry is refused");
        assert!(
            format!("{error:#}").contains("before its first generation header"),
            "{error:#}"
        );
        std::fs::write(dir.join(BUILD_KEYS), b"not a section\n").expect("a corrupt record");
        let error = read_build_keys(&dir).expect_err("a record with no generation is refused");
        assert!(
            format!("{error:#}").contains("before its first generation header"),
            "{error:#}"
        );
        std::fs::remove_file(dir.join(BUILD_KEYS)).expect("remove the record");
        std::fs::create_dir(dir.join(BUILD_KEYS)).expect("a directory where the record was");
        let error = read_build_keys(&dir).expect_err("an unreadable record is refused");
        assert!(format!("{error:#}").contains("cannot be read"), "{error:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A manifest that cannot be read is refused rather than read as absent: a build
    /// that wrote over it from nothing would reset the generation's build totals and
    /// understate the cost the record is required to carry. A manifest of another
    /// generation is simply not this one's; one that names no generation is refused
    /// like the record's header-less case, not treated as another's.
    #[test]
    fn a_manifest_that_cannot_be_read_is_not_an_absent_one() {
        let dir = temp_dir("manifest");
        assert_eq!(
            read_build_manifest(&dir, "gen").expect("no manifest is not a broken one"),
            None
        );
        write_build_manifest(
            &dir,
            &serde_json::json!({"generation": "other", "builds": 3}),
        )
        .expect("write a manifest");
        assert_eq!(
            read_build_manifest(&dir, "gen").expect("another generation's manifest"),
            None
        );
        write_build_manifest(&dir, &serde_json::json!({"builds": 3})).expect("a manifest header");
        let error =
            read_build_manifest(&dir, "gen").expect_err("a manifest with no generation is refused");
        assert!(
            format!("{error:#}").contains("names no generation"),
            "{error:#}"
        );
        std::fs::write(dir.join(BUILD_MANIFEST), b"{not json").expect("a corrupt manifest");
        let error = read_build_manifest(&dir, "gen").expect_err("a corrupt manifest is refused");
        assert!(
            format!("{error:#}").contains("corpus build manifest"),
            "{error:#}"
        );
        std::fs::remove_file(dir.join(BUILD_MANIFEST)).expect("remove the manifest");
        std::fs::create_dir(dir.join(BUILD_MANIFEST)).expect("a directory where the manifest was");
        let error =
            read_build_manifest(&dir, "gen").expect_err("an unreadable manifest is refused");
        assert!(format!("{error:#}").contains("cannot be read"), "{error:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
