// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The synthetic verification voices: clones cut from single-reader corpus
//! recordings the false-accept phase never measures.
//!
//! A clone's only per-voice input is the speaker embedding of its reference audio
//! ([`crate::synth::Synthesis::new`] embeds the whole file), so a voice *is* the
//! recording and the span taken out of it, not a name: the span is cached as a
//! 16 kHz WAV and every clip rendered from it carries the span's own identity in
//! its cache key, so one voice's audio can never stand in for another's.
//!
//! Selection walks [`sets::VERIFICATION_POOL`] in its pinned order and takes the
//! first [`sets::NUM_VERIFICATION_VOICES`] whose two-sentence render comes back
//! from the workspace's own recogniser word for word. A candidate that fails is
//! named and the next one is tried; a voice is never kept by lowering the bar. The
//! enrolled voice is checked the same way, is reported, and is never replaced.
//!
//! The reference embeddings' pairwise similarity is reported as a witness: ten
//! distinct recordings can still carry the same person, and nothing here rejects
//! a voice for resembling another — rejection is the intelligibility check alone.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::graphs::{self, Compiled};
use crate::host;
use crate::listen;
use crate::loader::Pair;
use crate::measurements::sha256_file;
use crate::tokenizer::Tokenizer;
use crate::{synth, wake};

use super::audio::Synth;
use super::config::{BenchConfig, Refused};
use super::sets;
use crate::guard::{Deadline, MemoryCeiling};

/// The enrolled voice: the one the enrolment and every threshold above it are
/// built from. Its clips are the ones the cache already holds, so it has no span
/// identity of its own.
pub(crate) const ENROLLED: &str = "enrolled";

/// One selected clone voice.
pub(crate) struct Voice {
    /// `voice01`..`voice10`, numbered by position in the selected set.
    pub name: String,
    /// The recording the span was cut from, relative to the corpus directory.
    pub recording: &'static str,
    pub sha256: &'static str,
    pub offset_secs: f64,
    pub secs: f64,
    /// The 16 kHz span WAV the clone is built from, in the corpus cache.
    pub wav: PathBuf,
}

impl Voice {
    /// The render cache key's voice component: the span itself, so a re-pinned
    /// pool order cannot serve one recording's audio under another's name.
    pub(crate) fn cache_id(&self) -> String {
        format!("{}:{:.2}:{:.2}", self.sha256, self.offset_secs, self.secs)
    }
}

/// The stream a measurement runs on: the enrolled voice or one clone.
#[derive(Clone)]
pub(crate) enum Speaker {
    Enrolled,
    Voice(Arc<Voice>),
}

impl Speaker {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Enrolled => ENROLLED,
            Self::Voice(voice) => &voice.name,
        }
    }

    pub(crate) fn voice(&self) -> Option<&Voice> {
        match self {
            Self::Enrolled => None,
            Self::Voice(voice) => Some(voice),
        }
    }

    /// The render cache key's voice component: absent for the enrolled voice,
    /// whose cached renders must stay valid.
    pub(crate) fn cache_id(&self) -> Option<String> {
        self.voice().map(Voice::cache_id)
    }
}

/// One intelligibility sentence's outcome.
pub(crate) struct SentenceCheck {
    pub text: &'static str,
    /// What the recogniser answered.
    pub heard: String,
    /// Whether the answer carries the sentence's words, in order, ignoring case
    /// and punctuation.
    pub passed: bool,
    /// Whether the render ran past the probe window, so the recogniser never heard
    /// its last words. A truncated render is a render-length failure, not a
    /// failure of the check.
    pub truncated: bool,
}

/// Whether every sentence of the enrolled voice's own check came back.
fn enrolled_passed(checks: &[SentenceCheck]) -> bool {
    checks.iter().all(|check| check.passed)
}

/// A candidate the intelligibility check rejected, named with what it answered.
pub(crate) struct Replacement {
    pub recording: &'static str,
    pub sha256: &'static str,
    pub offset_secs: f64,
    pub secs: f64,
    /// The first sentence that did not come back, and what was heard instead.
    pub reason: String,
}

/// What the selection produced: the clones and the witness that their references
/// are distinct.
pub(crate) struct Selection {
    pub voices: Vec<Arc<Voice>>,
    pub replacements: Vec<Replacement>,
    /// Pairwise cosine over the selected voices' reference embeddings, `i < j`.
    pub similarity: Vec<(usize, usize, f32)>,
    /// The span WAV of every candidate that was tried, so the build's sweep keeps
    /// the references its replacements were judged on.
    pub spans: Vec<PathBuf>,
}

impl Selection {
    /// The streams a measurement runs on, enrolled first.
    pub(crate) fn speakers(&self) -> Vec<Speaker> {
        let mut speakers = Vec::with_capacity(self.voices.len() + 1);
        speakers.push(Speaker::Enrolled);
        speakers.extend(
            self.voices
                .iter()
                .map(|voice| Speaker::Voice(Arc::clone(voice))),
        );
        speakers
    }

    /// The report cell: what was selected, from what, and the similarity witness —
    /// the one place the witness is summarised.
    pub(crate) fn json(&self) -> Value {
        let voices: Vec<Value> = self
            .voices
            .iter()
            .map(|voice| {
                json!({
                    "name": voice.name,
                    "recording": voice.recording,
                    "sha256": voice.sha256,
                    "span": {
                        "offset_secs": voice.offset_secs,
                        "secs": voice.secs,
                    },
                })
            })
            .collect();
        let mut similarity = serde_json::Map::new();
        if let Some(summary) = similarity_summary(&self.similarity) {
            similarity.insert("min".into(), json!(summary.min));
            similarity.insert("median".into(), json!(summary.median));
            similarity.insert("max".into(), json!(summary.max));
            let (a, b, cosine) = summary.closest;
            similarity.insert(
                "closest".into(),
                json!({"a": self.voices[a].name, "b": self.voices[b].name, "cosine": cosine}),
            );
        }
        json!({
            "selected": self.voices.len(),
            "voices": voices,
            "replacements": self.replacements.iter().map(|replacement| json!({
                "recording": replacement.recording,
                "sha256": replacement.sha256,
                "span": {"offset_secs": replacement.offset_secs, "secs": replacement.secs},
                "reason": replacement.reason,
            })).collect::<Vec<_>>(),
            "reference_similarity": Value::Object(similarity),
        })
    }
}

/// The intelligibility probe's length: six seconds at the listening rate, which is
/// the longest a short sentence renders at a slow clone's pace. A fixed length lets
/// one compiled recognition path serve every probe of every voice, so the check
/// uploads the recogniser once instead of once per probe.
const PROBE_SECS: usize = 6;
const PROBE_FRAMES: usize = PROBE_SECS * crate::listen::SAMPLE_RATE as usize / graphs::MEL_HOP;

/// Pad or truncate a probe render to the probe length, reporting whether the render
/// ran past the window.
fn probe_audio(mut samples: Vec<f32>) -> (Vec<f32>, bool) {
    let len = PROBE_FRAMES * graphs::MEL_HOP;
    let truncated = samples.len() > len;
    samples.resize(len, 0.0);
    (samples, truncated)
}

/// Walk the pinned pool and select `cfg.voices` usable clones.
///
/// # Errors
/// A pinned recording that is missing or no longer hashes to its pin; a span the
/// recording cannot supply; a clip or a recognition that fails; a pool that cannot
/// fill the voices it was asked for, naming the candidates that failed and the
/// shortfall.
pub(crate) fn select(
    tts: &Arc<Pair>,
    asr: &Arc<Pair>,
    synth: &mut Synth,
    cfg: &BenchConfig,
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Result<Selection> {
    let tokenizer =
        Tokenizer::from_model(&asr.body).context("a tokenizer for the intelligibility check")?;
    let mut clip = listen::Clip::build(asr, &tokenizer, PROBE_FRAMES)
        .context("the intelligibility check's recognition path")?;
    deadline.check("voices")?;
    ceiling.check("voices")?;
    // The build's own prefix, from the operation the config was resolved for: its
    // lines run under two entry points and must not read as a measurement's.
    let label = cfg.mode.label();
    let enrolled = check(synth, &mut clip, &Speaker::Enrolled, deadline)?;
    eprintln!(
        "[{label}] enrolled voice: two-sentence check {}",
        if enrolled_passed(&enrolled) {
            "passed"
        } else {
            "failed (reported, never replaced)"
        }
    );

    let mut voices: Vec<Arc<Voice>> = Vec::with_capacity(cfg.voices);
    let mut replacements = Vec::new();
    let mut spans = Vec::new();
    for candidate in sets::VERIFICATION_POOL {
        if voices.len() == cfg.voices {
            break;
        }
        ceiling.check("voices")?;
        deadline.check("voices")?;
        let wav = span_path(&cfg.corpus, &cfg.cache, candidate)?;
        spans.push(wav.clone());
        let voice = Arc::new(Voice {
            name: format!("voice{:02}", voices.len() + 1),
            recording: candidate.recording,
            sha256: candidate.sha256,
            offset_secs: candidate.offset_secs,
            secs: candidate.secs,
            wav,
        });
        let speaker = Speaker::Voice(Arc::clone(&voice));
        if synth.renders() {
            // The build cuts the span every render of this voice reads; a measurement
            // reads the cut the build left — pre-flighted before the enrolment — and
            // never writes one.
            cut_span(&cfg.corpus, &voice.wav, candidate)?;
        } else {
            // A measurement replays this decision from the cache, so it stops here
            // with the shortfall rather than inside a clip it will not render.
            synth.require_corpus(
                &super::audio::probe_requests(&speaker),
                &format!(
                    "the intelligibility probes for {} ({})",
                    voice.name, voice.recording
                ),
            )?;
        }
        let sentences = check(synth, &mut clip, &speaker, deadline)?;
        match sentences.iter().find(|check| !check.passed) {
            None => {
                eprintln!(
                    "[{label}] {}: {} ({}) renders both sentences intelligibly",
                    voice.name,
                    voice.recording,
                    span_label(voice.offset_secs, voice.secs)
                );
                voices.push(voice);
            }
            Some(failed) => {
                let reason = if failed.truncated {
                    format!(
                        "`{}` rendered longer than the {PROBE_SECS} s probe window, so the check \
                         heard only its first words: `{}`",
                        failed.text, failed.heard
                    )
                } else {
                    format!("`{}` came back as `{}`", failed.text, failed.heard)
                };
                eprintln!(
                    "[{label}] {} ({}) rejected: {reason}; the next pinned candidate is tried",
                    candidate.recording,
                    span_label(candidate.offset_secs, candidate.secs)
                );
                replacements.push(Replacement {
                    recording: candidate.recording,
                    sha256: candidate.sha256,
                    offset_secs: candidate.offset_secs,
                    secs: candidate.secs,
                    reason,
                });
            }
        }
    }
    if voices.len() < cfg.voices {
        bail!(
            "{}",
            shortfall(
                sets::VERIFICATION_POOL.len(),
                voices.len(),
                cfg.voices,
                &replacements
            )
        );
    }
    // The speaker encoder runs here, once per voice that passed: an embedding of a
    // candidate the check rejected has nothing to witness.
    let embeddings: Vec<Vec<f32>> = voices
        .iter()
        .map(|voice| reference_embedding(tts, &voice.wav))
        .collect::<Result<_>>()?;
    let similarity = pairwise(&embeddings);
    Ok(Selection {
        voices,
        replacements,
        similarity,
        spans,
    })
}

/// The refusal a pool that cannot fill the voices asked for states: how short it
/// is and which candidates failed, in the order they were tried.
fn shortfall(pool: usize, filled: usize, asked: usize, failed: &[Replacement]) -> String {
    let named: Vec<String> = failed
        .iter()
        .map(|replacement| format!("{} ({})", replacement.recording, replacement.reason))
        .collect();
    format!(
        "the pinned pool of {pool} candidates filled {filled} of the {asked} verification voices \
         asked for — {} short — and has no candidate left; {} failed the check: {}",
        asked - filled,
        named.len(),
        named.join("; ")
    )
}

/// Render the two intelligibility sentences in one voice and read them back with
/// the workspace's own recogniser. The renders are [`super::audio::probe_requests`]
/// — the same two requests the plan pre-flights and the build renders.
fn check(
    synth: &mut Synth,
    clip: &mut listen::Clip<'_>,
    speaker: &Speaker,
    deadline: Deadline,
) -> Result<Vec<SentenceCheck>> {
    let requests = super::audio::probe_requests(speaker);
    let mut checks = Vec::with_capacity(requests.len());
    for request in &requests {
        deadline.check("voices")?;
        let samples = synth.clip(&request.speaker, request.text, request.seed)?;
        let (probe, truncated) = probe_audio(samples);
        let heard = clip
            .run(probe, None)
            .with_context(|| {
                format!(
                    "the intelligibility probe `{}` for {}",
                    request.text,
                    speaker.name()
                )
            })?
            .text;
        checks.push(SentenceCheck {
            text: request.text,
            passed: same_words(request.text, &heard),
            heard,
            truncated,
        });
    }
    Ok(checks)
}

/// Whether the recogniser's answer carries the sentence's words, in order,
/// ignoring case and punctuation. Everything that is not a letter or a digit
/// separates words, in both the expected sentence and the answer.
fn same_words(expected: &str, heard: &str) -> bool {
    fn words(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_string)
            .collect()
    }
    words(expected) == words(heard)
}

/// The pinned span as a 16 kHz WAV in the corpus cache: its path, after checking the
/// recording still hashes to its pin.
///
/// The recording is re-hashed on every run, so a re-pinned or replaced recording
/// cannot pass silently, and the name carries the recording's digest, the span and
/// the rate it was cut at, so a stale cut cannot stand in for another. The file is
/// the build's to write ([`cut_span`]): nothing on the measurement path writes it.
///
/// # Errors
/// Anything the named recording decides is a [`Refused`] input (exit 2).
fn span_path(
    corpus: &Path,
    cache: &Path,
    candidate: &sets::VoiceCandidate,
) -> Result<PathBuf, Refused> {
    let source = corpus.join(candidate.recording);
    let digest = sha256_file(&source)
        .with_context(|| format!("cannot hash the voice recording {}", source.display()))?;
    if digest != candidate.sha256 {
        return Err(Refused::new(anyhow!(
            "the pinned voice recording {} hashes {digest}, not {} — the pool names a recording \
             this corpus does not carry",
            source.display(),
            candidate.sha256
        )));
    }
    if candidate.secs < sets::MIN_VOICE_SPAN_SECS {
        return Err(Refused::new(anyhow!(
            "{} spans {} s, shorter than the {} s a voice needs",
            candidate.recording,
            candidate.secs,
            sets::MIN_VOICE_SPAN_SECS
        )));
    }
    Ok(cache.join(span_name(candidate)))
}

/// The name a pinned span is cached under: the recording's digest, the span and the
/// rate it was cut at, so a stale cut cannot stand in for another. Derivable without
/// reading the recording, which is what lets the pre-flight name the cuts a record
/// claims.
fn span_name(candidate: &sets::VoiceCandidate) -> String {
    format!(
        "voice-{}-{:.2}-{:.2}-{}.wav",
        &candidate.sha256[..16],
        candidate.offset_secs,
        candidate.secs,
        listen::SAMPLE_RATE
    )
}

/// The samples the pin's span asks for, at the listening rate.
fn span_samples(candidate: &sets::VoiceCandidate) -> usize {
    (candidate.secs * f64::from(listen::SAMPLE_RATE)).round() as usize
}

/// Cut the span the cache does not carry whole: the build's own step, over the
/// recording [`span_path`] checked. A whole cut is left alone and the recording is
/// not decoded, because the name carries the digest, span and rate it was cut at; a
/// torn one is cut again.
fn cut_span(corpus: &Path, path: &Path, candidate: &sets::VoiceCandidate) -> Result<()> {
    let samples = span_samples(candidate);
    if is_whole_span(path, samples) {
        return Ok(());
    }
    let source = corpus.join(candidate.recording);
    let offset = (candidate.offset_secs * f64::from(listen::SAMPLE_RATE)).round() as usize;
    let pcm = host::wav_mono_16k(&source.display().to_string(), offset, samples)
        .with_context(|| format!("cannot read the span of {}", source.display()))?;
    if pcm.len() != samples {
        bail!(
            "{} carries {} of the {} samples {} s + {} s asks for",
            source.display(),
            pcm.len(),
            samples,
            candidate.offset_secs,
            candidate.secs
        );
    }
    crate::measurements::write_atomic(path, |tmp| {
        host::write_wav16(&tmp.display().to_string(), &pcm, listen::SAMPLE_RATE)
    })
}

/// The corpus pre-flight's span requirement, before the enrolment is spent: every span
/// cut the build recorded for this generation must be in the cache, whole. The record
/// is the build's own statement of what it produced, so the pre-flight reads it instead
/// of re-deriving where the build stopped; a cache without a section for this generation
/// claims nothing, and the selection's own read is then the only check.
///
/// Every claimed cut is required, whatever `--voices` narrows the selection to: the
/// record states what the build produced, not what one run reads. The recordings' pins
/// are left to [`span_path`], which hashes one when the selection reaches the candidate
/// it names — a recording replaced after its cut was made is refused there, in the
/// `voices` phase, before that candidate's clips are used.
pub(crate) fn require_spans(cache: &Path, claims: Option<&HashSet<String>>) -> Result<()> {
    let Some(claims) = claims else {
        return Ok(());
    };
    for candidate in sets::VERIFICATION_POOL {
        let name = span_name(candidate);
        if claims.contains(&name) {
            require_span(&cache.join(&name), candidate)?;
        }
    }
    Ok(())
}

/// The measurement's span check: the cut must be there and whole, because a
/// measurement renders and writes nothing. A miss names the candidate and the build
/// that cuts it, before any clip is read.
fn require_span(path: &Path, candidate: &sets::VoiceCandidate) -> Result<()> {
    if is_whole_span(path, span_samples(candidate)) {
        return Ok(());
    }
    bail!(
        "the corpus cache carries no whole span cut of {} ({}) at {} — {}",
        candidate.recording,
        span_label(candidate.offset_secs, candidate.secs),
        path.display(),
        super::audio::BUILD_FIRST
    )
}

/// Whether `path` is the whole 16 kHz cut of `samples`: the header `host::write_wav16`
/// writes, the data length it declares, and a file of exactly that length. The
/// pre-flight's rule — one open, no sample decode — in the shape the clip pre-flight
/// uses, so a torn cut is a miss before the run starts rather than a decode failure
/// after it.
fn is_whole_span(path: &Path, samples: usize) -> bool {
    use std::io::Read;
    const HEADER: usize = 44;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(meta) = file.metadata() else {
        return false;
    };
    let mut header = [0u8; HEADER];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let data = u64::from(u32::from_le_bytes([
        header[40], header[41], header[42], header[43],
    ]));
    header.starts_with(b"RIFF")
        && &header[8..12] == b"WAVE"
        && data == samples as u64 * 2
        && meta.len() == HEADER as u64 + data
}

/// The reference embedding a clone is built from. It repeats what
/// [`crate::synth::Synthesis::new`] does with the reference audio — the speaker
/// encoder over the 24 kHz mel — rather than reading the pipeline's field,
/// because the modules that produce clips are part of the corpus cache key: a
/// code change there would invalidate every cached render.
fn reference_embedding(tts: &Arc<Pair>, wav: &Path) -> Result<Vec<f32>> {
    let path = wav.display().to_string();
    let audio = host::wav_mono(&path, synth::SAMPLE_RATE, 0, usize::MAX)
        .map_err(|error| anyhow!("speaker audio `{path}`: {error}"))?;
    if audio.is_empty() {
        bail!("the voice span `{path}` is empty");
    }
    let (mel, frames) = host::mel_24k(&audio);
    let cfg = tts.body.config();
    let mut speaker = Compiled::new(
        rlx_runtime::Device::Gpu,
        graphs::speaker_encoder(tts.sound.clone(), frames, cfg.d)?,
    )?;
    let mut embed = speaker.run_slices(&[("mel", &mel)]);
    embed
        .drain(..)
        .next()
        .ok_or_else(|| anyhow!("the speaker encoder returned no embedding for `{path}`"))
}

fn pairwise(embeddings: &[Vec<f32>]) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for a in 0..embeddings.len() {
        for b in a + 1..embeddings.len() {
            pairs.push((a, b, wake::cosine(&embeddings[a], &embeddings[b])));
        }
    }
    pairs
}

/// min / median / max of the pairwise similarities and the closest pair — the
/// record's one-line witness, computed once here and printed from the report.
pub(crate) struct Similarity {
    pub min: f32,
    pub median: f32,
    pub max: f32,
    /// The pair with the highest cosine.
    pub closest: (usize, usize, f32),
}

pub(crate) fn similarity_summary(similarity: &[(usize, usize, f32)]) -> Option<Similarity> {
    if similarity.is_empty() {
        return None;
    }
    let mut values: Vec<f32> = similarity.iter().map(|(_, _, cosine)| *cosine).collect();
    values.sort_by(f32::total_cmp);
    let closest = *similarity
        .iter()
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .expect("a non-empty slice");
    Some(Similarity {
        min: values[0],
        median: values[values.len() / 2],
        max: values[values.len() - 1],
        closest,
    })
}

/// A span as the record prints it: `16.10 s + 15.00 s`.
pub(crate) fn span_label(offset_secs: f64, secs: f64) -> String {
    format!("{offset_secs:.2} s + {secs:.2} s")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own, named so two runs cannot collide.
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir =
            std::env::temp_dir().join(format!("mahbot-voice-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Word-for-word is the rule the check implements: case and punctuation do
    /// not matter, order and missing or extra words do.
    #[test]
    fn the_probe_comparison_ignores_case_and_punctuation_only() {
        assert!(same_words(
            "the weather is sunny today",
            "The weather is sunny today."
        ));
        assert!(same_words(
            "please open the window",
            "Please, open the window!"
        ));
        assert!(!same_words(
            "the weather is sunny today",
            "the weather is sunny"
        ));
        assert!(!same_words(
            "please open the window",
            "open the window please"
        ));
        assert!(!same_words(
            "the weather is sunny today",
            "the whether is sunny today"
        ));
    }

    /// The probe sentences are not the wake phrase, and there are two of them.
    #[test]
    fn the_probes_are_not_the_wake_phrase() {
        for text in sets::INTELLIGIBILITY_SENTENCES {
            assert!(!text.contains(sets::WAKE_PHRASE), "{text}");
        }
        assert_eq!(sets::INTELLIGIBILITY_SENTENCES.len(), 2);
    }

    #[test]
    fn a_similarity_summary_is_min_median_max_and_the_closest_pair() {
        assert!(similarity_summary(&[]).is_none());
        let summary = similarity_summary(&[(0, 1, 0.9), (0, 2, 0.1), (1, 2, 0.5)]).expect("three");
        assert_eq!((summary.min, summary.median, summary.max), (0.1, 0.5, 0.9));
        assert_eq!(summary.closest, (0, 1, 0.9));
    }

    /// A probe render that runs past the window is reported as truncated, not as an
    /// unintelligible sentence.
    #[test]
    fn a_probe_past_its_window_is_marked_truncated() {
        let (padded, truncated) = probe_audio(vec![0.0; 16]);
        assert_eq!(padded.len(), PROBE_FRAMES * graphs::MEL_HOP);
        assert!(!truncated);
        let (_, truncated) = probe_audio(vec![0.0; 2 * PROBE_FRAMES * graphs::MEL_HOP]);
        assert!(truncated);
    }

    /// The shortfall names what is missing — how many voices, and every candidate
    /// that failed with what it answered — so a pool that cannot fill the set says
    /// what to re-pin instead of only that it failed.
    #[test]
    fn an_unfilled_pool_names_the_failures_and_the_shortfall() {
        let failed = vec![
            Replacement {
                recording: "speech/librivox/speech-librivox-0020.wav",
                sha256: "0",
                offset_secs: 1.0,
                secs: 15.0,
                reason: "`please open the window` came back as `please open the`".into(),
            },
            Replacement {
                recording: "speech/librivox/speech-librivox-0021.wav",
                sha256: "1",
                offset_secs: 2.0,
                secs: 15.0,
                reason: "`the weather is sunny today` came back as ``".into(),
            },
        ];
        let text = shortfall(20, 8, 10, &failed);
        assert!(text.contains("filled 8 of the 10"), "{text}");
        assert!(text.contains("2 short"), "{text}");
        assert!(text.contains("2 failed the check"), "{text}");
        assert!(text.contains("speech-librivox-0020.wav"), "{text}");
        assert!(text.contains("speech-librivox-0021.wav"), "{text}");
    }

    /// The witness is over every pair of the voices that passed: `n` embeddings
    /// give `n(n-1)/2` cosines, so a voice cannot be dropped from the comparison.
    #[test]
    fn the_similarity_witness_covers_every_pair() {
        let embeddings = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let pairs = pairwise(&embeddings);
        assert_eq!(pairs.len(), 3);
        let tuple = |i: usize, j: usize| {
            pairs
                .iter()
                .find(|(a, b, _)| *a == i && *b == j)
                .map(|(_, _, cosine)| *cosine)
        };
        assert_eq!(tuple(0, 1), Some(0.0));
        for pair in [(0, 2), (1, 2)] {
            let cosine = tuple(pair.0, pair.1).expect("a pair");
            assert!((cosine - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        }
    }

    /// A recording the corpus input no longer matches its pin on is a refused input
    /// (exit 2), not a failed run: the build never starts, and a script keying on the
    /// code reads it as the stale input it is.
    #[test]
    fn a_recording_that_no_longer_matches_its_pin_is_refused() {
        let dir = temp_dir("pin");
        let candidate = &sets::VERIFICATION_POOL[0];
        let source = dir.join(candidate.recording);
        std::fs::create_dir_all(source.parent().expect("a parent")).expect("temp dir");
        std::fs::write(&source, b"not the pinned recording").expect("a stale recording");
        // The classification is the return type, and the entry point reads it off the
        // error: a stale pin is the refused-input code, not a failed run.
        let refused = span_path(&dir, &dir, candidate).expect_err("a stale recording is refused");
        let error: anyhow::Error = refused.into();
        assert!(format!("{error:#}").contains("hashes"), "{error:#}");
        assert_eq!(crate::bench::exit_code(&error), 2, "{error:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A span is whole only when its header, its declared data length and its file
    /// length agree: a torn cut is a miss the measurement stops on, not a clip it
    /// reads half of — and the same rule is what makes the build re-cut one.
    #[test]
    fn a_span_is_whole_only_when_its_header_and_length_agree() {
        let dir = temp_dir("whole");
        let path = dir.join("span.wav");
        let pcm = vec![0.25f32; 1_600];
        host::write_wav16(&path.display().to_string(), &pcm, listen::SAMPLE_RATE).expect("a cut");
        assert!(is_whole_span(&path, pcm.len()), "the cut the build writes");
        assert!(
            !is_whole_span(&path, pcm.len() + 1),
            "a file shorter than the span asks for is torn"
        );
        assert!(!is_whole_span(&dir.join("absent.wav"), pcm.len()));
        let cut = std::fs::read(&path).expect("the cut");
        std::fs::write(&path, &cut[..cut.len() - 4]).expect("a torn cut");
        assert!(!is_whole_span(&path, pcm.len()), "a truncated cut");

        // The measurement's refusal names the candidate and the build, so an
        // operator reads a torn cut as a corpus miss rather than a broken run.
        let candidate = &sets::VERIFICATION_POOL[0];
        let error = require_span(&path, candidate).expect_err("a torn cut is a miss");
        assert!(error.to_string().contains(candidate.recording), "{error}");
        assert!(error.to_string().contains("a corpus build"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The span pre-flight takes its list from the build's record, not from a second
    /// walk of the pool: a cut the record claims must be whole, and one it does not
    /// claim is not required — so a cache with no section for the generation, or a
    /// section naming other entries, asks for nothing.
    #[test]
    fn the_pre_flight_requires_the_span_cuts_the_record_claims() {
        let dir = temp_dir("claims");
        let candidate = &sets::VERIFICATION_POOL[0];
        let name = span_name(candidate);
        require_spans(&dir, None).expect("a cache without a record claims nothing");
        require_spans(&dir, Some(&HashSet::new()))
            .expect("a section with no entries claims nothing");

        let claims = HashSet::from([name.clone()]);
        let error =
            require_spans(&dir, Some(&claims)).expect_err("a claimed cut is required whole");
        assert!(error.to_string().contains(candidate.recording), "{error}");

        let pcm = vec![0.0f32; span_samples(candidate)];
        host::write_wav16(
            &dir.join(&name).display().to_string(),
            &pcm,
            listen::SAMPLE_RATE,
        )
        .expect("a cut");
        require_spans(&dir, Some(&claims)).expect("the whole cut the build writes");
        let cut = std::fs::read(dir.join(&name)).expect("the cut");
        std::fs::write(dir.join(&name), &cut[..cut.len() - 8]).expect("a torn cut");
        require_spans(&dir, Some(&claims)).expect_err("a torn cut the record claims is a miss");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
