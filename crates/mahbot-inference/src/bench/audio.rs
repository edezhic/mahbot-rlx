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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use crate::listen::Listener;
use crate::synth::Synthesis;

use super::{detect, sets};
use crate::guard::{Deadline, MemoryCeiling};
use crate::measurements::hex;

/// A noise generator: its seed band decides which set it renders for.
pub(crate) type NoiseGenerator = fn(u64) -> Vec<f32>;

/// What one run's synthesis cost: the clips asked for, how many came from the
/// cache, and the wall time spent generating the rest.
#[derive(Clone, Copy, Default)]
pub(crate) struct SynthStats {
    pub clips: usize,
    pub cache_hits: usize,
    pub synth_secs: f64,
}

/// One synthesised-and-cached TTS front end. It owns the run's guards so a
/// cache miss — the only phase that can be long — stops at the next clip.
/// The resident-set check runs per clip; the system figures (two subprocesses)
/// stay at the phase boundaries the caller checks.
pub(crate) struct Synth {
    inner: Synthesis,
    cache: PathBuf,
    synthesis_hash: String,
    deadline: Deadline,
    ceiling: MemoryCeiling,
    progress: Arc<Mutex<super::Progress>>,
}

impl Synth {
    pub(crate) fn new(
        inner: Synthesis,
        cache: PathBuf,
        synthesis_hash: String,
        progress: Arc<Mutex<super::Progress>>,
        deadline: Deadline,
        ceiling: MemoryCeiling,
    ) -> Self {
        Self {
            inner,
            cache,
            synthesis_hash,
            progress,
            deadline,
            ceiling,
        }
    }

    /// One utterance at the listening rate (16 kHz). The sound half renders at
    /// [`crate::synth::SAMPLE_RATE`]; every listening path resamples.
    pub(crate) fn clip(&mut self, text: &str, seed: u64) -> Result<Vec<f32>> {
        let pcm = self.pcm(text, seed)?;
        Ok(crate::host::resample(
            &pcm,
            crate::synth::SAMPLE_RATE,
            crate::listen::SAMPLE_RATE,
        ))
    }

    /// The synthesised waveform at the sound half's own rate (24 kHz), from the
    /// cache when its key is present.
    fn pcm(&mut self, text: &str, seed: u64) -> Result<Vec<f32>> {
        // The phase is the one the caller recorded: a stop here has to name a
        // key the report's `phases` map carries (`enrolment`, `recall synthesis`,
        // `non-phrase synthesis`), not a generic "synthesis".
        let phase = super::state(&self.progress).current;
        self.deadline.check(phase)?;
        self.ceiling.check_process(phase)?;
        let key = cache_key(text, seed, crate::synth::SAMPLE_RATE, &self.synthesis_hash);
        let path = self.cache.join(key);
        if let Some(samples) = read_cache(&path) {
            let mut progress = super::state(&self.progress);
            progress.synthesis.clips += 1;
            progress.synthesis.cache_hits += 1;
            return Ok(samples);
        }
        let started = std::time::Instant::now();
        let samples = self
            .inner
            .synthesize(text, seed)
            .with_context(|| format!("synthesis failed for `{text}` seed {seed}"))?;
        let elapsed = started.elapsed().as_secs_f64();
        // The disk write happens before the progress lock is taken: the guard
        // thread reads its phase and its counters through that lock, and a slow
        // write must not be able to delay the wall-clock stop.
        write_cache(&path, &samples)?;
        let mut progress = super::state(&self.progress);
        progress.synthesis.clips += 1;
        progress.synthesis.synth_secs += elapsed;
        Ok(samples)
    }
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

/// Deterministic cache key for one synthesised utterance: text, seed, sample
/// rate and the synthesis model's own hash — both TTS files' digests and the
/// reference voice's, because the voice WAV is part of every prompt.
fn cache_key(text: &str, seed: u64, sample_rate: u32, synthesis_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update([0u8]);
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(sample_rate.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(synthesis_hash.as_bytes());
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

/// Atomic write: a unique temporary file in the cache directory, flushed to the
/// device before the rename, so a crash leaves either the old entry or the new
/// one and never a half-written clip.
///
/// The corpus is the run's durable artifact, so a cache that cannot be written
/// is an error: silently re-synthesizing all 394 clips on every run is how the
/// bench spent hours producing nothing reusable.
fn write_cache(path: &Path, samples: &[f32]) -> Result<()> {
    let mut data = Vec::with_capacity(8 + samples.len() * 4);
    data.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for &sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let written = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&data)?;
        file.sync_all()
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(error)
            .with_context(|| format!("cannot write the corpus clip {}", tmp.display()));
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| {
            format!(
                "cannot move the corpus clip {} into place at {}",
                tmp.display(),
                path.display()
            )
        });
    }
    Ok(())
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

/// The ten conditioned enrolment clips: one seeded rendering of the wake
/// phrase each, conditioned by [`condition_enrollment_clip`].
pub(crate) fn enrollment_clips(synth: &mut Synth) -> Result<Vec<(Vec<f32>, String)>> {
    let mut clips = Vec::with_capacity(sets::NUM_ENROLLMENT_VARIANTS);
    for i in 0..sets::NUM_ENROLLMENT_VARIANTS {
        let seed = sets::ENROLLMENT_SEED_BASE + i as u64;
        let pcm = synth.clip(sets::WAKE_PHRASE, seed)?;
        clips.push((
            condition_enrollment_clip(&pcm, i),
            format!("enroll{i}_s{seed}"),
        ));
    }
    Ok(clips)
}

/// One clip per phrase, seeded `seed_base + index`.
pub(crate) fn phrase_clips(
    synth: &mut Synth,
    phrases: &[&str],
    seed_base: u64,
    tag: &str,
) -> Result<Vec<(Vec<f32>, String)>> {
    phrases
        .iter()
        .enumerate()
        .map(|(i, phrase)| {
            let seed = seed_base + i as u64;
            Ok((synth.clip(phrase, seed)?, format!("{tag}_s{seed}")))
        })
        .collect()
}

/// The recall set: 40 bare wake-phrase clips plus 10 command-plus-speech clips.
pub(crate) fn recall_clips(synth: &mut Synth) -> Result<Vec<(Vec<f32>, String)>> {
    let mut clips = Vec::with_capacity(
        sets::HELD_OUT_WAKE_ONLY_CLIPS + sets::COMMAND_PLUS_SPEECH_PHRASES.len(),
    );
    for i in 0..sets::HELD_OUT_WAKE_ONLY_CLIPS {
        let seed = sets::HELD_OUT_SEED_BASE + i as u64;
        clips.push((
            synth.clip(sets::WAKE_PHRASE, seed)?,
            format!("heldout_wake_s{seed}"),
        ));
    }
    clips.extend(phrase_clips(
        synth,
        sets::COMMAND_PLUS_SPEECH_PHRASES,
        sets::COMMAND_PLUS_SPEECH_SEED_BASE,
        "command_other",
    )?);
    Ok(clips)
}

/// The 121-clip false-reaction set, in a fixed order: confusable bands,
/// unrelated bands, silence, noise profiles, then phrase-in-passing.
pub(crate) fn non_phrase_clips(synth: &mut Synth) -> Result<Vec<(Vec<f32>, String)>> {
    let mut clips = Vec::new();
    for &(band, tag) in sets::CONFUSABLE_BANDS {
        clips.extend(phrase_clips(synth, sets::CONFUSABLE_PHRASES, band, tag)?);
    }
    for &(band, tag) in sets::UNRELATED_BANDS {
        clips.extend(phrase_clips(synth, sets::UNRELATED_PHRASES, band, tag)?);
    }
    for &(label, len) in sets::SILENCE_DURATIONS {
        clips.push((vec![0.0f32; len], label.to_string()));
    }
    for (i, (label, generator)) in sets::all_noise_profiles().enumerate() {
        let seed = sets::DETECTION_NOISE_SEED_BASE + i as u64;
        clips.push((generator(seed), format!("{label}_s{seed}")));
    }
    clips.extend(phrase_clips(
        synth,
        sets::PHRASE_IN_PASSING_PHRASES,
        sets::PHRASE_IN_PASSING_SEED_BASE,
        "phrase_in_passing",
    )?);
    anyhow::ensure!(
        clips.len() == sets::EXPECTED_NON_PHRASE_TOTAL,
        "the non-phrase set built {} clips, not the pinned {}",
        clips.len(),
        sets::EXPECTED_NON_PHRASE_TOTAL
    );
    Ok(clips)
}

// ── Calibration negatives ────────────────────────────────────────────────

/// Owner negatives: the enrolled voice's non-wake speech, VAD-gated, plus one
/// brown-noise-at-[`sets::OWNER_NEGATIVE_SNR_DB`] embedding per cell.
pub(crate) fn owner_negatives(
    synth: &mut Synth,
    listener: &mut Listener,
    skipped: &mut Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    phrase_negatives(
        synth,
        listener,
        sets::OWNER_NEGATIVE_PHRASES,
        sets::OWNER_NEGATIVE_SEEDS_PER_PHRASE,
        sets::OWNER_NEGATIVE_SEED_BASE,
        "owner",
        Some(sets::OWNER_NEGATIVE_SNR_DB),
        skipped,
    )
}

/// Phrase negatives: `seeds_per_phrase` VAD-gated renderings per phrase, each
/// embedded once — and, when `noise_snr_db` is given, a second time under brown
/// noise at that SNR.
pub(crate) fn phrase_negatives(
    synth: &mut Synth,
    listener: &mut Listener,
    phrases: &[&str],
    seeds_per_phrase: usize,
    seed_base: u64,
    tag: &str,
    noise_snr_db: Option<f32>,
    skipped: &mut Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::new();
    for (i, phrase) in phrases.iter().enumerate() {
        for variant in 0..seeds_per_phrase {
            let seed = seed_base + (i * seeds_per_phrase + variant) as u64;
            let pcm = synth.clip(phrase, seed)?;
            let speech = detect::vad_gate(&pcm, &mut earshot::Detector::default());
            if speech.is_empty() {
                eprintln!(
                    "[wake-word] {tag} negative `{phrase}` seed {seed}: no VAD-positive speech — skipped"
                );
                skipped.push(format!("{tag} `{phrase}` seed {seed}"));
                continue;
            }
            embeddings.push(listener.embed(&speech)?);
            if let Some(snr_db) = noise_snr_db {
                let noisy = add_noise_color(&speech, snr_db, NoiseColor::Brown, seed);
                embeddings.push(listener.embed(&noisy)?);
            }
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
    fn the_cache_key_covers_text_seed_rate_and_model() {
        let base = cache_key("hey mahbot", 7, 24_000, "model");
        assert_eq!(base, cache_key("hey mahbot", 7, 24_000, "model"));
        assert_ne!(base, cache_key("hey mahbot", 8, 24_000, "model"));
        assert_ne!(base, cache_key("hey mahbot", 7, 16_000, "model"));
        assert_ne!(base, cache_key("hey mahbot ", 7, 24_000, "model"));
        assert_ne!(base, cache_key("hey mahbot", 7, 24_000, "other"));
        assert_eq!(base.len(), 64);
    }

    #[test]
    fn the_cache_round_trips_and_rejects_a_torn_file() {
        let dir = std::env::temp_dir().join(format!(
            "mahbot-wake-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("entry");
        let samples = vec![0.0f32, 0.5, -0.25, 1.0];
        write_cache(&path, &samples).expect("write");
        assert_eq!(read_cache(&path).expect("hit"), samples);

        std::fs::write(&path, [0u8; 6]).expect("torn");
        assert!(read_cache(&path).is_none());
        assert!(!path.exists(), "a torn entry is deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
