// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! "Transcribe any audio": where the speech is, how a long recording is cut into
//! recognition chunks, which regions tile the input, which transcripts are worth
//! trusting — and the model-backed call that runs the detector, the tagger and
//! the recogniser over that plan.
//!
//! The plan is pure. The span builder takes the detector's per-hop mask as an
//! argument, the cut takes the audio only to find the quietest hop near it, and the
//! guard takes the region's own duration — so it is testable on a machine with no
//! models and no audio, and the one-window listening path is untouched: this module
//! wires the recogniser and the tagger to the plan instead.
//! [`Transcriber`] is the other half: one call over one input, the regions of the
//! result tiling it exactly, or an error saying why the call could not finish.
//!
//! The rules are deliberately asymmetric. A start is the detector's own hop and
//! is never moved: conditioning and hangover may not shift where speech begins.
//! An end reaches into the silence after it, clamped to the next run and to the
//! input. A transcript is never dropped, only marked: the caller knows whether a
//! doubtful region is worth printing. The product wants to filter more rather
//! than less, but never to lose what the model actually said.
//!
//! What this module does **not** establish is anything about a transcript's
//! content: the guards detect loops and density, not wrong words. The pinned music
//! pool the stand measures against lives in [`crate::music`], beside the record
//! rather than in this surface.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use crate::ced::{Tag, Tagger};
use crate::graphs;
use crate::host;
use crate::listen;
use crate::loader::Pair;
use crate::report;
use crate::tokenizer::Tokenizer;
use crate::vad;

// ── the plan ────────────────────────────────────────────────────────

/// The analysis rate of every model in this crate — [`listen::SAMPLE_RATE`]'s,
/// not a second copy of it.
pub const RATE: u32 = listen::SAMPLE_RATE;
/// A speech run shorter than this (0.4 s) is not worth a recognition pass.
pub const MIN_SPEECH_HOPS: usize = 25;
/// A gap shorter than this (0.8 s) inside a run is a pause, not a region boundary.
pub const MERGE_GAP_HOPS: usize = 50;
/// How far a run's end reaches into the silence that follows it (0.4 s), so the
/// last phoneme is inside the chunk.
pub const HANGOVER_HOPS: usize = 25;
/// The encoder's stride in mel frames: three stride-2 convolutions, so one audio
/// token is eight frames. A chunk's frame count is rounded up to it and no further,
/// which keeps the padding under one token — the encoder's attention is dense, so a
/// padded frame becomes a key every real token attends to, and rounding to a coarser
/// grid changes the reading. The record's `parity-span` row is that check.
pub const FRAME_ALIGNMENT: usize = 8;
/// The length every cut aims at, in seconds: the span this recogniser's encoder was
/// trained over.
pub const CUT_SECONDS: usize = 8;
/// How far before its aim a cut may search. The narrower of this and
/// [`CUT_SLACK_PCT`] of the piece share applies, so a cut never leaves its even share of
/// the span by more than that: the caps bound the search over a long span, and the share
/// bounds it over a short one, where a wide search would walk to the end of the span —
/// whose hangover's silence is the quietest hop there is — and leave a fragment.
pub const CUT_EARLY_HOPS: usize = 187;
/// How far after its aim a cut may search, under the same two bounds.
pub const CUT_LATE_HOPS: usize = 437;
/// How far a cut may leave its even share of the span, in percent of that share: half a
/// piece. A piece is therefore between half a share and one and a half — about four to
/// twelve seconds at the aim — rather than the whole span the caps allow.
pub const CUT_SLACK_PCT: usize = 50;
/// The longest word overlap [`overlap_words`] reports: an overlap longer than a region
/// can end with is not a seam's.
const OVERLAP_WORDS: usize = 6;
/// The overlap a seam may not carry: two words can coincide between two regions by
/// chance, so the record's gate reads an overlap of at least this many words.
pub const SEAM_WORDS: usize = 3;

const HOP_SAMPLES: usize = crate::vad::HOP_LENGTH;

/// A span of the analysed input, in samples.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Span {
    start: usize,
    end: usize,
}

/// The speech spans a per-hop detector mask marks, in samples.
///
/// Runs of consecutive `true` hops are merged across gaps shorter than
/// `MERGE_GAP_HOPS`, runs shorter than `MIN_SPEECH_HOPS` are dropped, and each
/// surviving run's end reaches `HANGOVER_HOPS` into the silence after it —
/// clamped to the next run's start and to `samples`. A start boundary is the
/// detector's own hop and is never moved: conditioning and hangover may not shift
/// where speech begins.
fn speech_spans(mask: &[bool], samples: usize) -> Vec<Span> {
    // Every maximal run of `true` hops, as half-open hop ranges.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut hop = 0;
    while hop < mask.len() {
        if !mask[hop] {
            hop += 1;
            continue;
        }
        let start = hop;
        while hop < mask.len() && mask[hop] {
            hop += 1;
        }
        runs.push((start, hop));
    }
    // A short gap is a pause inside one utterance, not a region boundary.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (start, end) in runs {
        if let Some(last) = merged.last_mut() {
            if start - last.1 < MERGE_GAP_HOPS {
                last.1 = end;
                continue;
            }
        }
        merged.push((start, end));
    }
    // A run has to be long enough to be worth a pass; the filter runs after the
    // merge, so a burst broken by small pauses is judged as the whole it is.
    let kept: Vec<(usize, usize)> = merged
        .into_iter()
        .filter(|(start, end)| end - start >= MIN_SPEECH_HOPS)
        .collect();
    let mut spans = Vec::with_capacity(kept.len());
    for (i, &(start, end)) in kept.iter().enumerate() {
        let start = start * HOP_SAMPLES;
        // A run that begins at or past the end of the analysed input describes
        // audio the caller does not have; it is not a span of it.
        if start >= samples {
            break;
        }
        let mut stop = end * HOP_SAMPLES + HANGOVER_HOPS * HOP_SAMPLES;
        if let Some(&(next, _)) = kept.get(i + 1) {
            stop = stop.min(next * HOP_SAMPLES);
        }
        spans.push(Span {
            start,
            end: stop.min(samples),
        });
    }
    spans
}

/// The shortest piece a cut may leave, in samples (0.4 s — the shortest run
/// segmentation keeps at all): a span cut into pass-length pieces can otherwise end
/// on a 0.2 s fragment that is inside the hangover and near-silent, which is not a
/// piece of speech and must not be reported as one.
pub const MIN_CHUNK_SAMPLES: usize = MIN_SPEECH_HOPS * HOP_SAMPLES;

/// The speech spans cut into recognition chunks, each aimed at [`CUT_SECONDS`]:
/// a span shorter than that plus one fragment is one chunk, and a longer span is cut
/// into [`cut`]'s pieces. Spans that need no cut are returned unchanged.
fn chunks(spans: Vec<Span>, audio: &[f32]) -> Vec<Span> {
    spans
        .into_iter()
        .flat_map(|span| cut(span, audio))
        .collect()
}

/// `span` cut into pieces aimed at [`CUT_SECONDS`]: each cut lands on the quietest hop
/// within [`CUT_EARLY_HOPS`]/[`CUT_LATE_HOPS`] of its even share of the span — and never
/// further than [`CUT_SLACK_PCT`] of that share, so a cut sits in a pause rather than in
/// a word and the span's trailing silence never becomes a piece of its own. The piece
/// count follows the span's length alone, so moving a cut never adds or removes a pass.
fn cut(span: Span, audio: &[f32]) -> Vec<Span> {
    let aim = CUT_SECONDS * RATE as usize;
    let len = span.end.saturating_sub(span.start);
    if len <= aim + MIN_CHUNK_SAMPLES {
        return vec![span];
    }
    // The number of pieces the span's length asks for, rounded to the nearest aim:
    // twelve seconds is two six-second pieces rather than one of twelve.
    let count = ((len + aim / 2) / aim).max(1);
    let (early, late) = (CUT_EARLY_HOPS * HOP_SAMPLES, CUT_LATE_HOPS * HOP_SAMPLES);
    // Each piece's own share of the span, and how far a cut may leave it.
    let slack = len / count * CUT_SLACK_PCT / 100;
    let mut out = Vec::with_capacity(count);
    let mut start = span.start;
    for k in 1..count {
        let target = span.start + len * k / count;
        let lo = target
            .saturating_sub(early.min(slack))
            .max(start + MIN_CHUNK_SAMPLES);
        let hi = (target + late.min(slack)).min(span.end.saturating_sub(MIN_CHUNK_SAMPLES));
        // `lo` is itself a legal cut, which is what the fallback uses when the window is
        // empty or the audio does not reach it.
        let at = quietest_hop(audio, lo, hi)
            .unwrap_or(lo)
            .min(span.end.saturating_sub(MIN_CHUNK_SAMPLES))
            .max(start + MIN_CHUNK_SAMPLES);
        out.push(Span { start, end: at });
        start = at;
    }
    out.push(Span {
        start,
        end: span.end,
    });
    out
}

/// The first hop in `[lo, hi]` with the lowest RMS, on the detector's own
/// 256-sample grid. `None` when the range holds no whole hop inside `audio`.
fn quietest_hop(audio: &[f32], lo: usize, hi: usize) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    let mut at = lo.div_ceil(HOP_SAMPLES) * HOP_SAMPLES;
    while at <= hi {
        let Some(hop) = audio.get(at..at + HOP_SAMPLES) else {
            break;
        };
        let rms = (hop
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / HOP_SAMPLES as f64)
            .sqrt();
        if best.is_none_or(|(lowest, _)| rms < lowest) {
            best = Some((rms, at));
        }
        at += HOP_SAMPLES;
    }
    best.map(|(_, at)| at)
}

/// The regions that tile `[0, samples)` with no hole and no overlap: one region
/// per speech chunk and one for each span of background between them, in order.
/// The first region starts at 0 and the last ends at exactly `samples`, so the
/// total coverage is the input's length by construction. Each region carries its
/// chunk index — `Some` for a recognition chunk's own region, `None` for
/// background — and a caller that has a transcript per chunk takes it from here
/// rather than by comparing spans: the clamp below is defensive, and a clamp that
/// fired would silently pair a chunk's answer with its neighbour's region.
fn tile_indexed(chunks: &[Span], samples: usize) -> Vec<(Span, Option<usize>)> {
    let mut regions = Vec::with_capacity(chunks.len() * 2 + 1);
    let mut cursor = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        // The clamp costs nothing on ordered chunks (the only kind `chunks`
        // returns) and keeps the invariant even if a caller hands over a list
        // that overlaps or begins before 0.
        let start = chunk.start.min(samples).max(cursor);
        let end = chunk.end.min(samples).max(start);
        if start > cursor {
            regions.push((
                Span {
                    start: cursor,
                    end: start,
                },
                None,
            ));
        }
        if end > start {
            regions.push((Span { start, end }, Some(index)));
        }
        cursor = cursor.max(end);
    }
    if cursor < samples {
        regions.push((
            Span {
                start: cursor,
                end: samples,
            },
            None,
        ));
    }
    regions
}

/// Each chunk's reading in chunk order, from the `(index, answer)` pairs the walks
/// produced: a shape is a frame count, so the walks are taken shape by shape and the
/// chunks of one shape sit scattered through the plan. A reading is placed by the
/// chunk's own index and never by the order the walks ran, which is what keeps a
/// transcript with the audio it describes — a chunk's text attached to its neighbour's
/// region would hold the region times right and the text wrong.
fn readings_by_chunk(count: usize, readings: Vec<(usize, Guarded)>) -> Vec<Option<Guarded>> {
    let mut answers: Vec<Option<Guarded>> = (0..count).map(|_| None).collect();
    for (index, answer) in readings {
        answers[index] = Some(answer);
    }
    answers
}

// ── the text guards ─────────────────────────────────────────────────

/// The repetition and density limits of the text guards.
pub const REPEAT_GRAM: usize = 4;
pub const REPEAT_LIMIT: usize = 3;
pub const WORD_REPEAT_LIMIT: usize = 12;
/// Characters per second of the region's audio above which a transcript is
/// denser than anyone can speak.
pub const DENSITY_CEILING: f64 = 30.0;

/// Generated tokens a chunk may spend per second of its own audio. The pinned
/// 17.46 s recognition clip generates 3.7 a second, so ten leaves a wide margin
/// over dense speech; what it stops is a loop — eighteen seconds of solfège once
/// generated four thousand `la` and cost forty-five seconds of decoding.
pub const CHUNK_TOKENS_PER_S: f64 = 10.0;

/// The floor of a chunk's token budget, so a half-second region can still say a
/// short sentence.
pub const MIN_CHUNK_TOKENS: usize = 48;

/// One chunk's decode budget from its own length, in tokens.
fn chunk_budget(seconds: f64) -> usize {
    let tokens = (CHUNK_TOKENS_PER_S * seconds.max(0.0)).ceil();
    if tokens.is_finite() && tokens >= MIN_CHUNK_TOKENS as f64 {
        tokens as usize
    } else {
        MIN_CHUNK_TOKENS
    }
}

/// One region's transcript as the guards leave it.
#[derive(Clone, PartialEq, Debug)]
struct Guarded {
    text: Option<String>,
    reliable: bool,
}

/// Whether a reading says anything a language is written with: the model's own
/// "no speech" answer arrives empty, as whitespace, or as a lone punctuation mark.
fn spoken(text: &str) -> bool {
    text.trim().chars().any(char::is_alphanumeric)
}

/// The three text guards, in the direction the product wants them: filter more
/// rather than less.
/// * a transcript with nothing a language is written with — empty, whitespace, or
///   punctuation only — is the model's own "no speech": `text: None`, and nothing
///   to distrust, so `reliable: true`;
/// * a repetition loop — any `REPEAT_GRAM`-word gram occurring more than
///   `REPEAT_LIMIT` times, or one word run more than `WORD_REPEAT_LIMIT` times —
///   is `reliable: false` and the text is kept;
/// * text denser than `DENSITY_CEILING` characters per second of the region's own
///   audio is `reliable: false`, the text kept.
/// Text is never dropped: the call marks, the caller cuts.
fn guard(text: &str, seconds: f64) -> Guarded {
    // Nothing at all, or nothing a language is written with: the model's own
    // "no speech" answer arrives both ways, and a lone punctuation mark is not a
    // reading of the audio either.
    if !spoken(text) {
        return Guarded {
            text: None,
            reliable: true,
        };
    }
    let trimmed = text.trim();
    let words: Vec<String> = trimmed.split_whitespace().map(str::to_lowercase).collect();
    let mut reliable = true;
    // One word held for more than a dozen turns.
    let mut run = 1;
    for pair in words.windows(2) {
        if pair[0] == pair[1] {
            run += 1;
            if run > WORD_REPEAT_LIMIT {
                reliable = false;
            }
        } else {
            run = 1;
        }
    }
    // Any short gram that keeps coming back. Occurrences may overlap, which is
    // what makes "la la la la la la la" four four-grams and not one.
    if words.len() >= REPEAT_GRAM {
        let last = words.len() - REPEAT_GRAM;
        for i in 0..=last {
            let gram = &words[i..i + REPEAT_GRAM];
            let mut count = 1;
            for j in i + 1..=last {
                if &words[j..j + REPEAT_GRAM] == gram {
                    count += 1;
                }
            }
            if count > REPEAT_LIMIT {
                reliable = false;
                break;
            }
        }
    }
    // A region with no audio and text is infinitely dense; so is one whose
    // duration is not a number. Both are marks, not crashes.
    let density = trimmed.chars().count() as f64 / seconds;
    if seconds <= 0.0 || !density.is_finite() || density > DENSITY_CEILING {
        reliable = false;
    }
    Guarded {
        text: Some(text.to_string()),
        reliable,
    }
}

// ── the aligned input ───────────────────────────────────────────────

/// The frame count a chunk of `frames` mel frames is compiled at:
/// `ceil(frames / FRAME_ALIGNMENT) * FRAME_ALIGNMENT`, and never 0.
fn aligned_frames(frames: usize) -> usize {
    frames.div_ceil(FRAME_ALIGNMENT).max(1) * FRAME_ALIGNMENT
}

/// `audio` truncated (or zero-padded) to exactly `samples` samples — the
/// `frames * 160` an [`aligned_frames`] graph takes.
fn pad_to(audio: &[f32], samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples);
    out.extend_from_slice(&audio[..audio.len().min(samples)]);
    out.resize(samples, 0.0);
    out
}

/// A mel frame count from a sample count at the crate's hop (`graphs::MEL_HOP`).
/// A partial trailing hop is not a frame; the caller pads to the alignment so the
/// count it compiles at is whole.
fn frames_of(samples: usize) -> usize {
    samples / graphs::MEL_HOP
}

// ── the model-backed half ───────────────────────────────────────────

/// One region of the returned result.
#[derive(Clone, PartialEq, Debug)]
pub struct Region {
    /// Start of the region, in samples at [`Transcription::sample_rate`].
    pub start: usize,
    /// End of the region, exclusive.
    pub end: usize,
    /// The transcript of the audio in this region, as the model produced it —
    /// `None` only where the model answered nothing (`guard` read its empty
    /// answer as "no speech").
    pub text: Option<String>,
    /// What the tagger heard here, when the region carries one: always for a
    /// region without text, and also for a region whose transcript the guards
    /// rejected. A transcript that survived carries no label.
    pub label: Option<String>,
    /// `false` when the guards rejected the transcript: it is returned anyway,
    /// marked, because the call does not decide for the caller.
    pub reliable: bool,
}

/// What one call produced, in the input's own time base. The regions tile the
/// input exactly: the first starts at 0, each starts where the last ended, and
/// the last ends at `samples`.
#[derive(Clone, PartialEq, Debug)]
pub struct Transcription {
    pub sample_rate: u32,
    pub samples: usize,
    /// The gain the conditioning applied to the whole input (1.0: none). A gain
    /// is time-preserving, so it cannot move a boundary.
    pub gain: f32,
    pub regions: Vec<Region>,
}

impl Transcription {
    /// The input's length in seconds.
    pub fn duration_s(&self) -> f64 {
        self.samples as f64 / f64::from(self.sample_rate)
    }

    /// `at` — a sample of [`Transcription::sample_rate`] — as seconds.
    pub fn seconds(&self, at: usize) -> f64 {
        at as f64 / f64::from(self.sample_rate)
    }

    /// The result's own contract, checked in every build: a region answers with a
    /// text or with a label, a label never sits beside a reliable transcript, the
    /// regions are ordered without a hole or an overlap inside `[0, samples]`, the
    /// first starts at 0 and the last ends at `samples`. A caller that composes a
    /// `Transcription` — the tests do — gets the same check the call itself runs,
    /// rather than a `debug_assert` a release build strips.
    pub fn check(&self) -> Result<()> {
        let (Some(first), Some(last)) = (self.regions.first(), self.regions.last()) else {
            bail!("the result has no region");
        };
        if first.start != 0 {
            bail!("the first region starts at {} instead of 0", first.start);
        }
        if last.end != self.samples {
            bail!(
                "the last region ends at {} instead of the input's {} samples",
                last.end,
                self.samples
            );
        }
        let mut cursor = 0usize;
        for region in &self.regions {
            let at = self.seconds(region.start);
            if region.text.is_none() && region.label.is_none() {
                bail!("the region at {at:.3} s answers neither with a text nor with a label");
            }
            if region.label.is_some() && region.text.is_some() && region.reliable {
                bail!("the region at {at:.3} s carries a label beside a reliable transcript");
            }
            if region.start != cursor || region.end <= region.start {
                bail!(
                    "the region at {at:.3} s does not meet its neighbour or has no length \
                     ({}..{} after {cursor})",
                    region.start,
                    region.end
                );
            }
            if region.end > self.samples {
                bail!(
                    "the region at {at:.3} s ends at {} samples, past the input's {}",
                    region.end,
                    self.samples
                );
            }
            cursor = region.end;
        }
        Ok(())
    }
}

/// What one call cost, split by stage — the figures the measurement record
/// reports. A caller that only wants the transcript ignores this.
#[derive(Clone, Copy, Default, Debug)]
pub struct Costs {
    /// Reading the file, for a call that came through [`Transcriber::transcribe_file`]:
    /// zero for a caller that hands the audio over itself.
    pub read_ms: f64,
    /// The detector's per-hop mask and the span/chunk/tile plan.
    pub segment_ms: f64,
    /// The tagger's log-mel and its windows over the whole input.
    pub tag_ms: f64,
    /// Every recognition graph built by this call.
    pub build_ms: f64,
    /// The recognition passes themselves (front end, prefill, decode).
    pub recognise_ms: f64,
    /// The recognition passes: one per chunk.
    pub chunks: usize,
    /// The distinct frame counts the graphs were compiled at — the front end, prefill
    /// and decode a chunk pays for when no other chunk shares its length.
    pub shapes: usize,
    pub windows: usize,
    /// Seconds of audio the recognition passes actually covered.
    pub speech_s: f64,
    /// The part of [`Costs::speech_s`] whose region came back with no transcript at
    /// all: the detector called it speech and the chunk's reading produced nothing. On
    /// input that holds speech this is speech the path failed to transcribe, in
    /// seconds. On input that holds none it is the detector's false positives, which is
    /// why the record keeps the two pools apart. A region's whole seconds are counted,
    /// the hangover into silence included, so this is never more than
    /// [`Costs::speech_s`].
    pub textless_s: f64,
    /// [`Costs::textless_s`] over regions whose tagger label is one of
    /// [`WORD_LABELS`]: a region the recogniser had no reading for at all — lost
    /// speech, in seconds.
    pub textless_voiced_s: f64,
    /// [`Costs::textless_s`] over regions whose label says otherwise: the detector
    /// marked audio the tagger calls something other than words, so there was nothing
    /// there for the recogniser to miss. The two figures are the split the residue is
    /// judged by.
    pub textless_other_s: f64,
    /// Seconds of merged speech spans the tagger calls something that cannot carry
    /// words and that were not presented to the recogniser at all: the marked hops
    /// among them are cleared from the plan, and their audio is a region's label
    /// instead. Never more than [`Costs::masked_s`], because only marked hops are
    /// cleared. Music and singing are always presented (`WORD_LABELS`), and so is any
    /// span shorter than one tagger window, which the tagger has judged only once.
    pub suppressed_s: f64,
    /// Chunks that ended with no transcript.
    pub textless_chunks: usize,
    /// The part of [`Costs::speech_s`] whose transcript the guards rejected: the
    /// text is returned, marked, with the tagger's label beside it, and a caller
    /// that filters loses these seconds.
    pub flagged_s: f64,
    /// Seconds the detector marked speech, before the plan dropped the runs too
    /// short to be worth a pass.
    pub masked_s: f64,
    /// The part of [`Costs::masked_s`] that no chunk covers: the audio
    /// segmentation threw away before any word existed. It is not
    /// `masked_s - speech_s`, because a chunk's hangover reaches into silence the
    /// detector did not mark, so that difference can be negative.
    pub uncovered_s: f64,
}

/// The detector, the tagger and the recogniser over one recognition pair.
///
/// The three are held together because one call uses all three: the detector
/// answers where the speech is, the tagger what the region sounds like, the
/// recogniser what was said. All three are the caller's — the pair, its tagger —
/// so a caller that opened a pair from an explicit path cannot be paired with a
/// tagger resolved from somewhere else. The clip a shape's chunks run on is built
/// and dropped inside the call. A cache across calls is ruled out by residency, not
/// by the types — the clip could own an `Arc<Tokenizer>` as it owns `Arc<Pair>`: one
/// call already holds two body-sized graphs while a chunk's decode graph stays
/// resident and the next chunk's prefill is compiled, so a cross-call cache would
/// hold a third between calls. A sweep of files therefore pays one front end, one
/// decode graph and one prefill per distinct chunk length per call; the record's
/// `build` row prices exactly that, and the weight upload is not part of it (the
/// shared weight buffer skips a model's bytes a compile already placed).
pub struct Transcriber {
    pair: Arc<Pair>,
    tokenizer: Tokenizer,
    tagger: Tagger,
    costs: Costs,
}

impl Transcriber {
    /// Open the tokenizer over `pair` and hold `tagger`. Both are required:
    /// without the tagger a non-speech region could not carry a label, and a region
    /// never comes back empty.
    pub fn open(pair: Arc<Pair>, tagger: Tagger) -> Result<Self> {
        let tokenizer = Tokenizer::from_model(&pair.body).context("the recognition tokenizer")?;
        Ok(Self {
            pair,
            tokenizer,
            tagger,
            costs: Costs::default(),
        })
    }

    /// Transcribe any length of `audio` at `sample_rate`. Whole input, every
    /// time: the result's regions tile it, or the call returns an error saying
    /// why it could not. It never truncates.
    pub fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<Transcription> {
        // A call that fails reports no figures at all: the previous call's describe
        // audio this one never saw.
        self.costs = Costs::default();
        if audio.is_empty() {
            bail!("the input has no samples");
        }
        if sample_rate == 0 {
            bail!("the input declares a zero sample rate, so no mark in it is a time");
        }
        let samples = audio.len();
        // Refused before the buffers exist: the 16 kHz length this input will
        // occupy, which a source far below 16 kHz makes far more than its own count.
        let projected = samples.saturating_mul(RATE as usize) / sample_rate as usize;
        self.ceiling(projected, 0, None)?;
        // The audio only has to be copied when it needs converting or lifting:
        // an hour of 16 kHz input that is already loud enough is read where it is.
        let converted;
        let mut audio16: Cow<'_, [f32]> = if sample_rate == RATE {
            Cow::Borrowed(audio)
        } else {
            converted = host::resample(audio, sample_rate, RATE);
            Cow::Owned(converted)
        };
        // Conditioning, once and time-preserving: a quiet recording is lifted by
        // one gain over the whole input, so every mark the plan finds stays where
        // it is. Nothing is trimmed and nothing is dropped — a trim would move
        // every boundary in the file. `stand::prepare_speech` is the single-shot
        // path's conditioning and does drop a loud pre-roll; that is why this
        // path may not reuse it.
        let gain = host::speech_gain(&audio16);
        if gain != 1.0 {
            audio16.to_mut().iter_mut().for_each(|value| *value *= gain);
        }
        let audio16: &[f32] = &audio16;

        // The tagger's log-mel and its windows, once each, over the whole input: one
        // mel for every label lookup below, and the windows every lookup reads. It
        // runs before the plan because the plan asks it which merged spans can hold
        // words at all. A signal no longer than one detector hop is shorter than the
        // tagger's own reflect pad, so its copy is extended with silence — which
        // moves no boundary and no transcript — because a region that short still
        // has to come back labelled.
        let t = report::Stopwatch::start();
        let extended;
        let signal: &[f32] = if audio16.len() <= vad::HOP_LENGTH {
            extended = pad_to(audio16, vad::HOP_LENGTH + self.tagger.samples_per_frame());
            &extended
        } else {
            audio16
        };
        let mel = self
            .tagger
            .log_mel(signal)
            .context("the tagger's log-mel")?;
        let tags = self.tagger.tags(&mel).context("the tagger's windows")?;
        let tag_ms = t.ms();

        // Where the speech is, how it is cut, and how the cuts tile the input.
        let t = report::Stopwatch::start();
        let mut detector = earshot::Detector::default();
        let mask = vad::speech_mask(audio16, &mut detector);
        let masked_s = mask.iter().filter(|hop| **hop).count() as f64 * vad::HOP_LENGTH as f64
            / f64::from(RATE);
        // A merged span the tagger calls non-speech is not presented to the recogniser
        // as speech: short bursts over a gap shorter than the merge reach join into an
        // arbitrarily long span, and a span of noise given to the recogniser is a pass
        // that can only invent words. Those seconds are the region's label instead,
        // and they are not seconds segmentation dropped.
        let mut suppressed_s = 0f64;
        let mut kept = mask.clone();
        let mut spans = Vec::new();
        // One label per surviving span, asked once: it decides whether the span can
        // hold words at all, and it is the label of every region that is the span
        // uncut.
        let mut span_labels: BTreeMap<Span, Option<String>> = BTreeMap::new();
        let window = self.tagger.window_frames() * self.tagger.samples_per_frame();
        for span in speech_spans(&mask, audio16.len()) {
            let label = region_label(&self.tagger, &tags, span);
            if removable(
                self.tagger.labels(),
                window,
                self.tagger.samples_per_frame(),
                &tags,
                span,
            ) {
                // Only the hops the detector marked are cleared, so the figure stays a
                // part of `masked_s`: a span's extent reaches into silence the detector
                // never called speech.
                suppressed_s +=
                    cleared_hops(&kept, span) as f64 * HOP_SAMPLES as f64 / f64::from(RATE);
                suppress(&mut kept, span);
            } else {
                span_labels.insert(span, label);
                spans.push(span);
            }
        }
        let chunk_list = chunks(spans, audio16);
        let plan = tile_indexed(&chunk_list, audio16.len());
        // The seconds the plan dropped: masked hops no chunk covers and no label
        // accounts for (`suppressed_s` above). Counted against the chunks themselves
        // rather than as `masked - covered`, because a hangover covers silence the
        // mask never marked.
        let mut uncovered = 0usize;
        let mut at = 0usize;
        for (hop, speech) in kept.iter().enumerate() {
            if !speech {
                continue;
            }
            let from = hop * vad::HOP_LENGTH;
            while chunk_list.get(at).is_some_and(|chunk| chunk.end <= from) {
                at += 1;
            }
            if !chunk_list.get(at).is_some_and(|chunk| from >= chunk.start) {
                uncovered += vad::HOP_LENGTH;
            }
        }
        let uncovered_s = uncovered as f64 / f64::from(RATE);
        let segment_ms = t.ms();

        // The chunks grouped by the frame count their graphs are compiled at, so a
        // file whose chunks share a length pays one front end and one decode graph
        // for both rather than one each. Each chunk's prefill is dropped once it has
        // run and rebuilt for the next, while the decode graph stays resident for the
        // shape: a rebuild is half a second against a body upload, and the two graphs a
        // call holds at once are what the record's peak pays.
        let mut shapes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, chunk) in chunk_list.iter().enumerate() {
            shapes.entry(chunk_frames(*chunk)).or_default().push(index);
        }
        // Each chunk's reading: the walk's own text through the guards, and never a
        // second pass — the plan cuts at the span the encoder was trained over, so a
        // chunk is read once. A walk is taken per shape and a shape holds chunks
        // scattered through the plan, so the readings are collected with the chunk's
        // own index and placed by it ([`readings_by_chunk`]), never by the order the
        // shapes were visited: a reading attached to its neighbour's region would
        // mistime the text and miscount the textless seconds.
        let mut readings: Vec<(usize, Guarded)> = Vec::with_capacity(chunk_list.len());
        let (mut build_ms, mut recognise_ms) = (0f64, 0f64);
        for (frames, indexes) in &shapes {
            let mut clip = listen::Clip::build(&self.pair, &self.tokenizer, *frames)
                .with_context(|| format!("building the {frames}-frame recognition graphs"))?;
            for &index in indexes {
                let chunk = chunk_list[index];
                let seconds = (chunk.end - chunk.start) as f64 / f64::from(RATE);
                // The chunk's own length bounds what it can say: a walk capped
                // there cannot spend a minute looping over eighteen seconds of
                // singing, and the loop, if it happens, is still in the words.
                clip.set_max_new(chunk_budget(seconds));
                let walk = clip
                    .run(chunk_audio(audio16, chunk, *frames), 0)
                    .with_context(|| {
                        format!("transcribing samples {}..{}", chunk.start, chunk.end)
                    })?;
                // Each walk reports the build it paid: the shape's own graphs on its
                // first chunk (front end, prefill, decode) and a prefill rebuild on the
                // rest — so summing every walk's figure prices every compile the call
                // made, which is what the record's `build` row counts. The walk's own
                // cold pass is what produced its transcript, so it is what the
                // recognition figure counts.
                build_ms += walk.build_ms;
                recognise_ms += passage_ms(&walk);
                let mut answer = guard(&walk.text, seconds);
                answer.reliable &= !walk.truncated;
                readings.push((index, answer));
            }
            // The clip is dropped here, at the end of its shape: one shape's graphs are
            // resident at a time.
        }
        let answers = readings_by_chunk(chunk_list.len(), readings);
        let textless_chunks = answers
            .iter()
            .flatten()
            .filter(|answer| answer.text.is_none())
            .count();
        // `f64`'s `Sum` starts at `-0.0`, so an input with no chunk at all would report
        // a covered length of `-0`; adding zero normalises the sign.
        let speech_s: f64 = chunk_list
            .iter()
            .map(|chunk| (chunk.end - chunk.start) as f64 / f64::from(RATE))
            .sum::<f64>()
            + 0.0;

        // The plan's regions, each with the answer that speaks for it: a speech
        // region takes its own chunk's transcript (the plan carries the index, so
        // no two regions can be confused for each other), a background region has
        // none and is described by the tagger alone.
        let mut regions: Vec<Region> = Vec::with_capacity(plan.len());
        let (mut textless_s, mut textless_voiced_s, mut textless_other_s) = (0f64, 0f64, 0f64);
        let mut flagged_s = 0f64;
        for (span, chunk) in &plan {
            let mut text = None;
            let mut reliable = true;
            if let Some(index) = *chunk {
                let answer = answers[index]
                    .as_ref()
                    .with_context(|| format!("the reading of chunk {index}"))?;
                text.clone_from(&answer.text);
                reliable = answer.reliable;
            }
            // A transcript the guards rejected keeps its text but is marked, so
            // the tagger says what was heard there beside it; one that survived
            // carries no label. A region that is a span uncut takes the label the
            // span already has.
            let label = if text.is_none() || !reliable {
                match span_labels.get(span) {
                    Some(label) => label.clone(),
                    None => region_label(&self.tagger, &tags, *span),
                }
            } else {
                None
            };
            // A region with no text is not left unattributed: its seconds go to the
            // side its own label puts them on, so the residue can be read as lost
            // speech or as the detector's artifact. Only a region that came from a
            // recognition pass is counted — a background region is audio the detector
            // never called speech, and a suppressed span is `suppressed_s`.
            let seconds = (span.end - span.start) as f64 / f64::from(RATE);
            if chunk.is_some() {
                if text.is_none() {
                    textless_s += seconds;
                    if label.as_deref().is_some_and(bears_words) {
                        textless_voiced_s += seconds;
                    } else {
                        textless_other_s += seconds;
                    }
                } else if !reliable {
                    flagged_s += seconds;
                }
            }
            regions.push(Region {
                start: span.start,
                end: span.end,
                text,
                label,
                reliable,
            });
        }
        let out = Transcription {
            sample_rate,
            samples,
            gain,
            regions: retime(merge(regions), sample_rate, samples),
        };
        // The result's contract, checked here rather than only in a debug build: a
        // product that calls this gets the same guarantee the stand's gates read.
        out.check().context("the transcription's own regions")?;

        self.costs = Costs {
            read_ms: 0.0,
            segment_ms,
            tag_ms,
            build_ms,
            recognise_ms,
            chunks: chunk_list.len(),
            shapes: shapes.len(),
            windows: tags.len(),
            speech_s,
            textless_s,
            textless_voiced_s,
            textless_other_s,
            suppressed_s,
            textless_chunks,
            flagged_s,
            masked_s,
            uncovered_s,
        };
        Ok(out)
    }

    /// A WAV file read whole, at [`RATE`], through [`Transcriber::transcribe`] —
    /// no offset and no limit, so the result describes the file.
    pub fn transcribe_file(&mut self, path: &Path) -> Result<Transcription> {
        // A call that returns before `transcribe` reports no figures rather than the
        // last call's: a refusal is not a pass over anything.
        self.costs = Costs::default();
        let name = path.display().to_string();
        // Refused from the header, before the file is read: what the input will
        // occupy at [`RATE`] plus the file's own bytes, against the run's resident-set
        // ceiling. A file too large to hold is a diagnostic here rather than an
        // allocation the guard stops, and the length in it is the one the header
        // declares, which is also what the read is bounded by.
        let (rate, frames) =
            host::wav_header(&name).with_context(|| format!("reading the header of `{name}`"))?;
        let projected = usize::try_from(frames.saturating_mul(u64::from(RATE)) / u64::from(rate))
            .unwrap_or(usize::MAX);
        let disk = std::fs::metadata(path).map_or(0, |file| file.len());
        self.ceiling(projected, disk, Some(path))?;
        let t = report::Stopwatch::start();
        let audio = host::wav_mono_16k(&name, 0, projected.saturating_add(1))
            .with_context(|| format!("reading `{name}`"))?;
        let read_ms = t.ms();
        let out = self.transcribe(&audio, RATE)?;
        // The read is part of the call's cost, and `transcribe` priced the rest.
        self.costs.read_ms = read_ms;
        Ok(out)
    }

    /// The costs of the most recent `transcribe` call. A call that returned an
    /// error reports zeros: it has no complete pass to price, and the call before it
    /// describes audio this one never saw.
    pub fn costs(&self) -> Costs {
        self.costs
    }

    /// The tokenizer this call encodes with — the pair's own, opened once in
    /// [`Transcriber::open`]. A caller that has to encode alongside a call (the
    /// stand's parity span goes through `listen::recognise`) reads it here rather
    /// than parsing the same file a second time.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Refuse an input the run cannot hold, naming both figures: the call's own
    /// whole-input buffers (`samples` of f32 audio, the tagger's log-mel over them,
    /// and `disk` bytes of the file being read) against the run's resident-set
    /// ceiling. Called before those buffers are allocated, so an impossible input is
    /// a diagnostic rather than an allocation the guard stops.
    fn ceiling(&self, samples: usize, disk: u64, path: Option<&Path>) -> Result<()> {
        let need = self.need(samples, disk);
        if need > crate::guard::max_rss_bytes() {
            return Err(self.too_large(need, samples, path, disk));
        }
        Ok(())
    }

    /// What a whole input of `samples` at [`RATE`] needs of host memory: `samples` of
    /// f32 audio, the tagger's log-mel over them, and `disk` bytes of file while it is
    /// read — the three buffers the call holds at once.
    fn need(&self, samples: usize, disk: u64) -> u64 {
        let frames = samples / self.tagger.samples_per_frame().max(1);
        disk.saturating_add(samples as u64 * 4)
            .saturating_add(self.tagger.mel_bytes(frames))
    }

    /// The one refusal the ceiling gives, so an input refused before its conversion
    /// and one refused as a file say the same thing — and the `need` figure it prints
    /// is the one that was just compared with the ceiling, itemised the way it was
    /// summed (the file's own bytes are part of it only when a file is being read).
    fn too_large(
        &self,
        need: u64,
        samples: usize,
        path: Option<&Path>,
        disk: u64,
    ) -> anyhow::Error {
        let name = match path {
            Some(path) => format!("`{}`", path.display()),
            None => "the input".to_string(),
        };
        let disk = if disk > 0 {
            format!(
                " and the file's own bytes ({:.0} MiB) while it is read",
                disk as f64 / 1048576.0
            )
        } else {
            String::new()
        };
        anyhow::anyhow!(
            "{name} of {:.0} s ({samples} samples at {RATE} Hz) needs ~{} MiB of host memory — \
             its audio and its tagger log-mel{disk} — past the run's {} MiB resident-set ceiling \
             (`MAHBOT_MAX_RSS_MIB`)",
            samples as f64 / f64::from(RATE),
            need / (1024 * 1024),
            crate::guard::max_rss_bytes() / (1024 * 1024),
        )
    }
}

/// What one walk's cold pass cost the recogniser: every phase of its first
/// iteration, which is what produced the transcript.
fn passage_ms(walk: &listen::Walk) -> f64 {
    walk.mel.first_ms
        + walk.encoder.first_ms
        + walk.projector.first_ms
        + walk.prefill.first_ms
        + walk.decode_times.iter().sum::<f64>()
}

/// The frame count a chunk's graphs are compiled at: its own length, rounded to the
/// encoder's stride and no further ([`FRAME_ALIGNMENT`]).
fn chunk_frames(chunk: Span) -> usize {
    aligned_frames(frames_of(chunk.end - chunk.start))
}

/// The audio one chunk is recognised from: the samples of `read` (its own, or one of its
/// pieces), padded to the frame count it is compiled at.
fn chunk_audio(audio: &[f32], read: Span, frames: usize) -> Vec<f32> {
    pad_to(
        audio.get(read.start..read.end).unwrap_or(&[]),
        frames * graphs::MEL_HOP,
    )
}

/// The marked hops inside a span, before it is cleared from the mask.
fn cleared_hops(mask: &[bool], span: Span) -> usize {
    let last = mask.len();
    let (from, to) = (span.start / HOP_SAMPLES, span.end / HOP_SAMPLES);
    mask[from.min(last)..to.min(last)]
        .iter()
        .filter(|marked| **marked)
        .count()
}

/// Clear a span from the detector's per-hop mask, so the span is not planned into a
/// recognition chunk at all: the audio becomes a label instead.
fn suppress(mask: &mut [bool], span: Span) {
    let last = mask.len();
    let (from, to) = (span.start / HOP_SAMPLES, span.end / HOP_SAMPLES);
    mask[from.min(last)..to.min(last)].fill(false);
}

/// Whether a span may be removed from the recogniser's input rather than transcribed.
///
/// Only when the tagger has seen it more than once, and only when **no** window over it
/// names a label that can carry words ([`WORD_LABELS`]) — one such window keeps the
/// whole span, whatever the span's aggregate label says. A span shorter than a tagger
/// window is judged by a single window, itself padded and so out of distribution: a
/// three-second excerpt of a pop song came back labelled `Meow` and lost its transcript
/// that way, and a guess of that weight may not take audio away from the recogniser.
fn removable(labels: &[String], window: usize, per_frame: usize, tags: &[Tag], span: Span) -> bool {
    if per_frame == 0 || span.end.saturating_sub(span.start) < window {
        return false;
    }
    let (from, to) = (span.start / per_frame, span.end / per_frame);
    !tags
        .iter()
        .filter(|tag| tag.end > from && tag.start < to)
        .filter_map(|tag| labels.get(tag.index))
        .any(|label| bears_words(label))
}

/// The words at the head of `text` that `previous` already ended with: two adjacent
/// regions that end and begin with the same words carry one of them twice. Whole words,
/// the longest overlap up to [`OVERLAP_WORDS`], compared without case or punctuation.
pub fn overlap_words(previous: &str, text: &str) -> usize {
    let bare = |word: &str| {
        word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let head: Vec<&str> = text.split_whitespace().collect();
    let tail: Vec<&str> = previous.split_whitespace().collect();
    let most = head.len().min(tail.len()).min(OVERLAP_WORDS);
    (1..=most)
        .rev()
        .find(|&k| {
            (0..k).all(|i| {
                bare(head[i]) == bare(tail[tail.len() - k + i]) && !bare(head[i]).is_empty()
            })
        })
        .unwrap_or(0)
}

/// The tagger's labels that mean words, taken from the file's own 527-name vocabulary:
/// the speech classes, the singing ones, the word-bearing music ones, and the vocal
/// noises a word can survive. A label outside this list is a span the recogniser is not
/// asked about — its audio becomes the region's label instead.
///
/// The list is explicit rather than a family match on purpose. A substring rule admits
/// `Musical instrument`, `Whale vocalization`, `Singing bowl` and `Hum` and excludes
/// `Song`, `Lullaby` and `A capella`, which is the wrong side of the line in both
/// directions: an instrumental clip is the pool that must never be transcribed, and a
/// sung line is the one the song rule must offer. Everything a word-bearing name could be
/// spelled as in this ontology is here, and the record prints the list with the seconds it
/// removed, so the rule is visible where it acts.
pub const WORD_LABELS: &[&str] = &[
    // Speech.
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Children shouting",
    "Whispering",
    "Speech synthesizer",
    "Hubbub, speech noise, speech babble",
    // Singing.
    "Singing",
    "Choir",
    "Yodeling",
    "Chant",
    "Mantra",
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
    "Rapping",
    "Humming",
    "Vocal music",
    "A capella",
    "Song",
    "Lullaby",
    // Music: offered to the recogniser as the song rule asks — if the words come out they
    // are transcribed, and if they do not the guards and the label say so.
    "Ambient music",
    "Angry music",
    "Background music",
    "Carnatic music",
    "Christian music",
    "Christmas music",
    "Classical music",
    "Dance music",
    "Electronic dance music",
    "Electronic music",
    "Exciting music",
    "Folk music",
    "Funny music",
    "Gospel music",
    "Happy music",
    "Hip hop music",
    "House music",
    "Independent music",
    "Jingle (music)",
    "Middle Eastern music",
    "Music",
    "Music for children",
    "Music of Africa",
    "Music of Asia",
    "Music of Bollywood",
    "Music of Latin America",
    "Musical instrument",
    "New-age music",
    "Pop music",
    "Rock music",
    "Sad music",
    "Salsa music",
    "Scary music",
    "Soul music",
    "Soundtrack music",
    "Swing music",
    "Tender music",
    "Theme music",
    "Traditional music",
    "Trance music",
    "Video game music",
    "Wedding music",
];

/// Whether a tagger label says the audio there can hold words: one of [`WORD_LABELS`],
/// matched on the name the file carries.
fn bears_words(label: &str) -> bool {
    let name = label.trim();
    WORD_LABELS
        .iter()
        .any(|word| word.eq_ignore_ascii_case(name))
}

/// The tagger's label for a region: the windows overlapping it, weighted by how
/// much of it each covers ([`Tagger::label_of`]), with mel frames from samples at
/// the tagger's own hop — not the ASR front end's, which is a different number in
/// general ([`Tagger::samples_per_frame`]). A region shorter than one frame is
/// widened to a frame, and a region no window overlaps at all takes the first
/// window's label: every region this call returns is described by its text or by
/// its label.
fn region_label(tagger: &Tagger, tags: &[Tag], span: Span) -> Option<String> {
    let hop = tagger.samples_per_frame();
    let start = span.start / hop;
    let end = (span.end / hop).max(start + 1);
    let index = match tagger.label_of(tags, start, end) {
        Some((index, _)) => index,
        None => tags.first()?.index,
    };
    tagger.labels().get(index).cloned()
}

/// Adjacent regions that say nothing and carry the same label are one region: a
/// background run and the textless speech chunk beside it make the same answer
/// about the same audio, so the seam between them is not a boundary the caller
/// can use. A region with text never merges — its transcript is the boundary.
fn merge(regions: Vec<Region>) -> Vec<Region> {
    let mut out: Vec<Region> = Vec::with_capacity(regions.len());
    for region in regions {
        if let Some(last) = out.last_mut() {
            if last.text.is_none() && region.text.is_none() && last.label == region.label {
                last.end = region.end;
                continue;
            }
        }
        out.push(region);
    }
    out
}

/// A sample index of [`RATE`] in the caller's own time base: scaled to `rate`,
/// rounded to the nearest sample and clamped to `samples`, so a boundary never
/// names a sample the caller does not have.
fn at_rate(at: usize, rate: u32, samples: usize) -> usize {
    let scaled = (at as u64 * u64::from(rate) + u64::from(RATE) / 2) / u64::from(RATE);
    (scaled as usize).min(samples)
}

/// The plan's boundaries in the caller's own time base. The first region starts
/// at exactly 0 and the last ends at exactly `samples` — the two are the input's
/// own ends, not rounded — and every region is at least as late as the one
/// before it. A region whose boundaries round onto the same sample has no audio
/// in it and is not a region of the result.
fn retime(regions: Vec<Region>, rate: u32, samples: usize) -> Vec<Region> {
    let last = regions.len().saturating_sub(1);
    let mut out: Vec<Region> = Vec::with_capacity(regions.len());
    let mut cursor = 0usize;
    for (i, mut region) in regions.into_iter().enumerate() {
        region.start = if i == 0 {
            0
        } else {
            at_rate(region.start, rate, samples).max(cursor)
        };
        region.end = if i == last {
            samples
        } else {
            at_rate(region.end, rate, samples).max(region.start)
        };
        cursor = region.end;
        if region.end > region.start {
            out.push(region);
        }
    }
    // A first region that rounded away would otherwise leave the result starting
    // in the middle of the input.
    if let Some(first) = out.first_mut() {
        first.start = 0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(audio: &[f32]) -> f32 {
        (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt()
    }

    /// The plan's regions without their chunk indexes: what the tiling tests
    /// compare against.
    fn tile(chunks: &[Span], samples: usize) -> Vec<Span> {
        tile_indexed(chunks, samples)
            .into_iter()
            .map(|(span, _)| span)
            .collect()
    }

    /// The whole-file-coverage invariant, in the form every tiling must hold.
    fn assert_tiling(cuts: &[Span], samples: usize) {
        let regions = tile(cuts, samples);
        if regions.is_empty() {
            assert_eq!(samples, 0, "only an empty input may have no region");
            return;
        }
        assert_eq!(regions[0].start, 0, "the first region starts at 0");
        assert_eq!(
            regions
                .last()
                .expect("a non-empty tiling has a last region")
                .end,
            samples,
            "the last region ends at the input's length"
        );
        let mut total = 0;
        let mut cursor = 0;
        for region in &regions {
            assert!(region.start < region.end, "a region has a positive length");
            assert_eq!(
                region.start, cursor,
                "regions meet without a hole or an overlap"
            );
            total += region.end - region.start;
            cursor = region.end;
        }
        assert_eq!(total, samples, "the regions cover the input exactly once");
    }

    #[test]
    fn an_empty_mask_marks_no_speech() {
        assert!(speech_spans(&[], 10_000).is_empty(), "nothing is marked");
    }

    #[test]
    fn one_full_mask_is_one_span_clamped_to_the_input() {
        let mask = vec![true; 40];
        let samples = 40 * 256;
        let spans = speech_spans(&mask, samples);
        assert_eq!(spans.len(), 1, "one run is one span");
        assert_eq!(
            spans[0],
            Span {
                start: 0,
                end: samples
            },
            "the end is the input's"
        );
    }

    #[test]
    fn a_run_shorter_than_the_minimum_is_dropped() {
        assert!(
            speech_spans(&[true; MIN_SPEECH_HOPS - 1], 100 * 256).is_empty(),
            "a burst under 0.4 s is not worth a pass"
        );
        assert_eq!(
            speech_spans(&[true; MIN_SPEECH_HOPS], 100 * 256).len(),
            1,
            "exactly the minimum is kept",
        );
    }

    #[test]
    fn a_short_gap_merges_and_a_long_one_does_not() {
        let mut short = vec![true; 30];
        short.extend(std::iter::repeat_n(false, MERGE_GAP_HOPS - 1));
        short.extend(std::iter::repeat_n(true, 30));
        let merged = speech_spans(&short, 200 * 256);
        assert_eq!(merged.len(), 1, "a 0.79 s pause is inside one utterance");
        assert_eq!(merged[0].start, 0, "the merged run begins at its first hop");

        let mut long = vec![true; 30];
        long.extend(std::iter::repeat_n(false, MERGE_GAP_HOPS));
        long.extend(std::iter::repeat_n(true, 30));
        let split = speech_spans(&long, 200 * 256);
        assert_eq!(split.len(), 2, "a 0.8 s gap is a region boundary");
        assert!(
            split[0].end <= split[1].start,
            "spans never overlap after a split"
        );
    }

    #[test]
    fn hangover_extends_an_end_without_crossing_the_next_start_or_the_input() {
        let mut mask = vec![true; 30];
        mask.extend(std::iter::repeat_n(false, MERGE_GAP_HOPS));
        mask.extend(std::iter::repeat_n(true, 30));
        let samples = 200 * 256;
        let spans = speech_spans(&mask, samples);
        assert_eq!(
            spans[0].end,
            30 * 256 + HANGOVER_HOPS * 256,
            "the end reaches one hangover into the silence"
        );
        let clipped = speech_spans(&[true; 40], 40 * 256 + 100);
        assert_eq!(
            clipped[0].end,
            40 * 256 + 100,
            "hangover is clamped to the samples that exist"
        );
    }

    #[test]
    fn a_start_is_the_first_true_hop_and_is_never_moved() {
        let mut mask = vec![false; 7];
        mask.extend(std::iter::repeat_n(true, 30));
        let spans = speech_spans(&mask, 100 * 256);
        assert_eq!(spans[0].start, 7 * 256, "a start is the detector's own hop");
    }

    /// A long span is cut into pieces aimed at the encoder's span, each cut landing in
    /// the pause nearest its even share of the span.
    #[test]
    fn a_twenty_four_second_span_cuts_into_three_pauses() {
        const NOTCH: usize = 2_048;
        let mut audio = vec![0.1f32; 24 * RATE as usize];
        for at in [8 * RATE as usize, 16 * RATE as usize] {
            for value in &mut audio[at - NOTCH..at + NOTCH] {
                *value = 0.0;
            }
        }
        let average = audio.chunks_exact(256).map(rms).sum::<f32>() / (audio.len() / 256) as f32;
        let span = Span {
            start: 0,
            end: audio.len(),
        };
        let cuts = chunks(vec![span], &audio);
        assert_eq!(cuts.len(), 3, "24 s is three pieces of eight");
        assert_eq!(
            cuts[0].start, 0,
            "the first piece starts where the span does"
        );
        assert_eq!(
            cuts.last().expect("three pieces").end,
            audio.len(),
            "the last piece ends where the span does"
        );
        for (i, at) in [8 * RATE as usize, 16 * RATE as usize]
            .into_iter()
            .enumerate()
        {
            let cut = cuts[i].end;
            assert!(
                (at - NOTCH..at + NOTCH).contains(&cut),
                "cut {i} lands in the quiet notch at {at}"
            );
            assert!(
                rms(&audio[cut..cut + 256]) < average,
                "cut {i} is quieter than the span's average hop"
            );
        }
    }

    /// A cut lands on the quietest hop inside its own window — its share of the span,
    /// no further off than [`CUT_SLACK_PCT`] of that share — so a piece is the length
    /// the encoder was trained over and no piece is a fragment.
    #[test]
    fn a_cut_lands_in_the_quietest_hop_near_the_target() {
        let seconds = 40usize;
        // On the detector's own hop grid: a quiet hop is a hop, not a sample index.
        let notches: Vec<usize> = [7, 16, 24, 33]
            .iter()
            .map(|s| (s * RATE as usize) / HOP_SAMPLES * HOP_SAMPLES)
            .collect();
        let average = 0.5f32;
        let audio: Vec<f32> = (0..seconds * RATE as usize)
            .map(|i| {
                let hop = i / HOP_SAMPLES * HOP_SAMPLES;
                if notches.contains(&hop) { 0.0 } else { average }
            })
            .collect();
        let cuts = chunks(
            vec![Span {
                start: 0,
                end: audio.len(),
            }],
            &audio,
        );
        assert_eq!(cuts.len(), 5, "40 s is five pieces of eight");
        for (cut, notch) in cuts.iter().zip(&notches) {
            assert_eq!(
                cut.end, *notch,
                "each cut takes the pause inside its own window"
            );
        }
        for cut in &cuts {
            let len = cut.end - cut.start;
            assert!(
                len >= MIN_CHUNK_SAMPLES,
                "no piece is a fragment: {:?}",
                cut
            );
            assert!(
                len <= CUT_SECONDS * RATE as usize + CUT_LATE_HOPS * HOP_SAMPLES,
                "no piece reaches past its aim and the cut's own window: {:?}",
                cut
            );
        }
    }

    #[test]
    fn spans_that_fit_a_chunk_are_returned_unchanged() {
        let audio = vec![0.1f32; 600_000];
        let exact = Span {
            start: 0,
            end: CUT_SECONDS * RATE as usize,
        };
        assert_eq!(
            chunks(vec![exact], &audio),
            vec![exact],
            "exactly one pass is left alone"
        );
        let shorter = Span {
            start: 1_000,
            end: 101_000,
        };
        assert_eq!(
            chunks(vec![shorter], &audio),
            vec![shorter],
            "a span under the limit is left alone"
        );
    }

    /// A span the tagger calls non-speech is cleared from the mask before the plan is
    /// made, so no chunk is planned over it: its audio becomes a label instead of a
    /// recognition pass that can only invent words.
    #[test]
    fn a_suppressed_span_leaves_no_speech_hop_behind() {
        let mut mask = vec![false; 100];
        mask[10..20].fill(true);
        mask[40..50].fill(true);
        let span = Span {
            start: 10 * HOP_SAMPLES,
            end: 20 * HOP_SAMPLES,
        };
        suppress(&mut mask, span);
        assert!(
            mask[10..20].iter().all(|hop| !hop),
            "the suppressed span's hops are no longer speech"
        );
        assert!(
            mask[40..50].iter().all(|hop| *hop),
            "the span beside it is untouched"
        );
        // A span that reaches past the mask (a hangover at the input's end) is clamped
        // rather than panicking.
        suppress(
            &mut mask,
            Span {
                start: 40 * HOP_SAMPLES,
                end: usize::MAX / 2,
            },
        );
        assert!(mask.iter().all(|hop| !hop));
    }

    /// A span is removed from the recogniser's input only when the tagger has seen it
    /// more than once *and* no window over it names a label that can carry words: one
    /// window's guess — a three-second song excerpt labelled `Meow` — may not take a
    /// transcript away, and one word-bearing window keeps a longer span whatever its
    /// aggregate label says.
    #[test]
    fn only_a_span_the_tagger_saw_twice_and_never_called_words_is_removed() {
        let labels = ["Music".to_string(), "Meow".to_string(), "Bird".to_string()];
        let per_frame = 160; // samples in one mel frame: 100 frames a second
        let window = 1_000 * per_frame; // ten seconds, the tagger's own window
        let at = |start: usize, end: usize, index: usize| Tag {
            start,
            end,
            index,
            score: 0.5,
        };
        let tags = [
            at(0, 1_000, 2),     // Bird over the first ten seconds
            at(500, 1_500, 2),   // Bird again, overlapping it
            at(1_500, 2_500, 0), // Music from the fifteenth second
        ];
        let span = |from_s: usize, to_s: usize| Span {
            start: from_s * 16_000,
            end: to_s * 16_000,
        };
        assert!(
            !removable(&labels, window, per_frame, &tags, span(1, 4)),
            "a three-second span is one window's guess, whatever that window says"
        );
        assert!(
            !removable(&labels, window, per_frame, &tags, span(0, 30)),
            "one window naming music keeps the whole span offered to the recogniser"
        );
        assert!(
            removable(&labels, window, per_frame, &tags, span(0, 15)),
            "fifteen seconds of windows that never name words is a span the tagger can remove"
        );
    }

    /// The seam rule, which the record's gate reads: two adjacent text regions that end
    /// and begin with the same whole words carry one of them twice.
    #[test]
    fn two_regions_that_share_their_words_are_one_seam() {
        assert_eq!(
            overlap_words("и это было сказано", "Сказано, что дальше"),
            1,
            "the seam gate reads whole words, whatever their case or punctuation"
        );
        assert_eq!(
            overlap_words("a b c d", "c d e f g"),
            2,
            "the longest overlap is the one counted"
        );
        assert_eq!(
            overlap_words("a b c d", "x y z"),
            0,
            "unrelated text carries no seam"
        );
        assert_eq!(
            overlap_words("a b c d e f g h i", "c d e f g h i j k"),
            0,
            "an overlap longer than a region can end with is not the seam's"
        );
    }

    /// No cut leaves a fragment: a span just over the aim stays one chunk, and a long
    /// span's pieces are all longer than the floor a cut may leave.
    #[test]
    fn no_cut_leaves_a_piece_shorter_than_the_minimum() {
        let aim = CUT_SECONDS * RATE as usize;
        // One aim plus a fifth of a second: the overflow is not worth a pass of its own.
        let short_overflow = aim + MIN_CHUNK_SAMPLES / 2;
        let audio = vec![0.1f32; short_overflow];
        let span = Span {
            start: 0,
            end: audio.len(),
        };
        assert_eq!(
            chunks(vec![span], &audio),
            vec![span],
            "an overflow shorter than the minimum stays in the chunk"
        );
        // 61 s: every piece inside the aim and its window, none of them a fragment.
        let audio = vec![0.1f32; 61 * RATE as usize];
        let span = Span {
            start: 0,
            end: audio.len(),
        };
        let cuts = chunks(vec![span], &audio);
        assert!(cuts.len() >= 4, "61 s is at least four pieces");
        // A cut never leaves its even share by more than `CUT_SLACK_PCT` of it, so no
        // piece is shorter than the share less that slack.
        let share = audio.len() / cuts.len();
        for piece in &cuts[..cuts.len() - 1] {
            assert!(
                piece.end - piece.start >= share - share * CUT_SLACK_PCT / 100,
                "no piece is a fragment: {piece:?}"
            );
            assert!(
                piece.end - piece.start <= share + share * CUT_SLACK_PCT / 100,
                "no piece reaches past the share and its own slack: {piece:?}"
            );
        }
        for piece in &cuts {
            assert!(
                piece.end - piece.start <= aim + CUT_LATE_HOPS * HOP_SAMPLES,
                "no piece reaches past the aim and its own window: {piece:?}"
            );
        }
        for pair in cuts.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "the pieces tile the span");
        }
    }

    /// The wider bound a cut may never use: the caps bound the search over a long span,
    /// and the share's own slack bounds it over a short one, so a cut on a span of two
    /// pieces lands inside its share and not at the far end of the cap.
    #[test]
    fn a_short_spans_cut_is_bounded_by_its_share_not_by_the_cap() {
        let audio = vec![0.1f32; 13 * RATE as usize];
        let cuts = chunks(
            vec![Span {
                start: 0,
                end: audio.len(),
            }],
            &audio,
        );
        assert_eq!(cuts.len(), 2, "13 s is two pieces");
        let first = cuts[0];
        let share = audio.len() / 2;
        assert!(
            first.end <= share + share * CUT_SLACK_PCT / 100 + HOP_SAMPLES,
            "the cut stayed within its share's slack rather than the cap: {} against {share}",
            first.end
        );
        assert!(
            first.end < CUT_EARLY_HOPS * HOP_SAMPLES + share,
            "the cap is not what bounded it"
        );
    }

    /// A chunk's reading is its own: the walks are taken shape by shape, so the order
    /// they ran in is not the plan's order and a reading is placed by its chunk index.
    #[test]
    fn each_chunk_keeps_its_own_reading() {
        let reading = |text: &str| Guarded {
            text: Some(text.to_string()),
            reliable: true,
        };
        // Two shapes, visited shortest first: the second chunk of the plan is the one
        // the first shape read.
        let answers = readings_by_chunk(
            2,
            vec![
                (1, reading("the short piece")),
                (0, reading("the long piece")),
            ],
        );
        assert_eq!(
            answers[0].as_ref().and_then(|a| a.text.as_deref()),
            Some("the long piece")
        );
        assert_eq!(
            answers[1].as_ref().and_then(|a| a.text.as_deref()),
            Some("the short piece")
        );
        // The plan carries the index, so the region for a chunk takes that chunk's text
        // and the region beside it takes its own.
        let plan = tile_indexed(
            &[
                Span {
                    start: 0,
                    end: 4 * RATE as usize,
                },
                Span {
                    start: 4 * RATE as usize,
                    end: 5 * RATE as usize,
                },
            ],
            6 * RATE as usize,
        );
        for (span, index) in plan {
            match index {
                Some(0) => assert!(span.start == 0 && span.end == 4 * RATE as usize),
                Some(1) => assert!(span.start == 4 * RATE as usize),
                _ => assert!(span.end > 5 * RATE as usize, "the tail is background"),
            }
        }
    }

    /// A merged span the tagger calls non-speech is not presented to the recogniser
    /// as speech; the labels that mean words are.
    #[test]
    fn a_label_that_cannot_hold_words_is_not_sent_to_the_recogniser() {
        for label in [
            "Speech",
            "Male speech, man speaking",
            "Conversation",
            "Narration, monologue",
            "Whispering",
            "Singing",
            "Synthetic singing",
            "Choir",
            "Rapping",
            "Vocal music",
            "A capella",
            "Song",
            "Lullaby",
            "Chant",
        ] {
            assert!(bears_words(label), "{label} can hold words");
        }
        // Music and singing are offered even when no word comes out of them: the
        // recogniser and the guards decide, and an instrumental clip produces no text.
        for label in [
            "Music",
            "Musical instrument",
            "Classical music",
            "Pop music",
            "Background music",
            "Speech synthesizer",
        ] {
            assert!(bears_words(label), "{label} can carry words");
        }
        // What cannot hold a word is cleared before the recogniser sees it.
        for label in [
            "Silence",
            "Noise",
            "Bird",
            "Train",
            "Walk, footsteps",
            "Animal",
            "Whale vocalization",
            "Singing bowl",
            "Mains hum",
        ] {
            assert!(!bears_words(label), "{label} cannot");
        }
    }

    #[test]
    fn regions_tile_the_input_without_a_hole_or_an_overlap() {
        for (cuts, samples) in [
            (Vec::new(), 1_000usize),
            (Vec::new(), 0),
            (
                vec![Span {
                    start: 200,
                    end: 400,
                }],
                1_000,
            ),
            (
                vec![
                    Span { start: 0, end: 100 },
                    Span {
                        start: 100,
                        end: 900,
                    },
                ],
                1_000,
            ),
            (
                vec![Span {
                    start: 0,
                    end: 1_000,
                }],
                1_000,
            ),
        ] {
            assert_tiling(&cuts, samples);
        }
        assert_eq!(
            tile(
                &[Span {
                    start: 200,
                    end: 400
                }],
                1_000
            ),
            vec![
                Span { start: 0, end: 200 },
                Span {
                    start: 200,
                    end: 400
                },
                Span {
                    start: 400,
                    end: 1_000
                },
            ],
            "background and speech alternate in order"
        );
        assert!(tile(&[], 0).is_empty(), "an empty input has no region");
    }

    #[test]
    fn every_mask_tiles_its_input_exactly() {
        use rand::rngs::StdRng;
        use rand::{Rng as _, SeedableRng as _};

        // A fixed seed makes a failure reproducible; the generator is the crate's
        // own (`wake` seeds `StdRng` the same way).
        let mut rng = StdRng::seed_from_u64(0x5eed_1234_abcd_0001);
        let audio = vec![0.0f32; 1_500_000];
        for case in 0..200 {
            let samples = rng.random_range(0..=1_500_000usize);
            let mask: Vec<bool> = (0..samples.div_ceil(256))
                .map(|_| rng.random_ratio(9, 10))
                .collect();
            let cuts = chunks(speech_spans(&mask, samples), &audio);
            assert!(
                cuts.iter().all(|c| c.start < c.end && c.end <= samples),
                "case {case}: every chunk lies inside the input with a positive length"
            );
            assert_tiling(&cuts, samples);
        }
    }

    #[test]
    fn an_empty_transcript_is_no_speech_and_not_distrusted() {
        // The same answer arrives as punctuation: the model answered nothing, and a
        // lone mark is not a reading of the audio either.
        for text in ["", "   ", "\n\t ", ".", "。", " — ", "..."] {
            assert_eq!(
                guard(text, 1.0),
                Guarded {
                    text: None,
                    reliable: true
                },
                "an empty transcript is the model's own no-speech"
            );
        }
    }

    #[test]
    fn an_ordinary_sentence_is_reliable() {
        let guarded = guard("the meeting starts in ten minutes", 2.5);
        assert_eq!(
            guarded.text.as_deref(),
            Some("the meeting starts in ten minutes"),
            "the text is kept as it came"
        );
        assert!(guarded.reliable, "a short ordinary sentence is trustworthy");
    }

    #[test]
    fn a_repeated_gram_is_unreliable_and_its_text_is_kept() {
        let guarded = guard("la la la la la la la", 2.0);
        assert!(guarded.text.is_some(), "the text is never dropped");
        assert!(
            !guarded.reliable,
            "a four-word gram four times over is a loop"
        );
    }

    #[test]
    fn one_word_written_thirteen_times_is_unreliable() {
        let guarded = guard(&"word ".repeat(13), 5.0);
        assert!(guarded.text.is_some(), "the text is never dropped");
        assert!(!guarded.reliable, "a run of 13 words is a loop");
    }

    #[test]
    fn density_marks_text_that_outruns_its_audio() {
        let text = "x".repeat(40);
        assert!(
            !guard(&text, 0.5).reliable,
            "40 characters in 0.5 s is 80 per second"
        );
        assert!(
            guard(&text, 3.0).reliable,
            "the same 40 characters over 3 s are 13.3 per second"
        );
    }

    // ── the result, without a model ────────────────────────────────

    fn region(start: usize, end: usize, text: Option<&str>, label: Option<&str>) -> Region {
        Region {
            start,
            end,
            text: text.map(str::to_string),
            label: label.map(str::to_string),
            reliable: true,
        }
    }

    #[test]
    fn adjacent_textless_regions_with_one_label_collapse() {
        let merged = merge(vec![
            region(0, 100, None, Some("Music")),
            region(100, 200, None, Some("Music")),
            region(200, 300, None, Some("Speech")),
        ]);
        assert_eq!(
            merged,
            vec![
                region(0, 200, None, Some("Music")),
                region(200, 300, None, Some("Speech")),
            ],
            "a run of one label is one region, and the next label still splits it"
        );
    }

    #[test]
    fn two_labels_do_not_collapse() {
        let merged = merge(vec![
            region(0, 100, None, Some("Music")),
            region(100, 200, None, Some("Singing")),
        ]);
        assert_eq!(
            merged.len(),
            2,
            "two different labels are two answers and stay two regions"
        );
        assert_eq!(
            merged[1].start, 100,
            "the second region keeps its own start"
        );
    }

    #[test]
    fn a_region_with_text_never_collapses() {
        let merged = merge(vec![
            region(0, 100, None, Some("Music")),
            region(100, 200, Some("hello"), None),
            region(200, 300, None, Some("Music")),
        ]);
        assert_eq!(
            merged.len(),
            3,
            "a transcript is a boundary a merge may not cross"
        );
        assert_eq!(
            merged[1].text.as_deref(),
            Some("hello"),
            "the transcript is untouched by the merge"
        );
    }

    #[test]
    fn merging_leaves_the_tiled_coverage_alone() {
        let samples = 1_000;
        let plan = tile(
            &[
                Span {
                    start: 200,
                    end: 400,
                },
                Span {
                    start: 700,
                    end: 900,
                },
            ],
            samples,
        );
        let regions: Vec<Region> = plan
            .iter()
            .map(|span| region(span.start, span.end, None, Some("Music")))
            .collect();
        let merged = merge(regions.clone());
        assert_eq!(
            merged,
            vec![region(0, samples, None, Some("Music"))],
            "one label over the whole input is one region"
        );
        let covered = |list: &[Region]| list.iter().map(|r| r.end - r.start).sum::<usize>();
        assert_eq!(
            covered(&merged),
            covered(&regions),
            "a merge changes the seams, not the coverage"
        );
        assert_eq!(
            (merged[0].start, merged[0].end),
            (0, samples),
            "the merged region spans exactly what its parts did"
        );
    }

    #[test]
    fn the_time_base_scales_sixteen_kilohertz_boundaries_three_fold() {
        let regions = vec![
            region(0, 100, None, Some("Music")),
            region(100, 1_000, Some("hello"), None),
            region(1_000, 16_000, None, Some("Speech")),
        ];
        let out = retime(regions, 48_000, 48_000);
        assert_eq!(out.len(), 3, "three regions in, three regions out");
        assert_eq!(out[0].start, 0, "the first region starts at 0");
        assert_eq!(
            out.last().map(|r| r.end),
            Some(48_000),
            "the last region ends at the input's own length"
        );
        assert_eq!(
            out[1].start, 300,
            "a boundary is three samples of 48 kHz for one of 16 kHz"
        );
        assert_eq!(out[1].end, 3_000, "and the end scales the same way");
        assert_eq!(out[2].end, 48_000, "16 s of 16 kHz is 48 s of 48 kHz");
    }

    #[test]
    fn every_region_list_retimes_to_an_ordered_time_base() {
        for seed in 0..200u64 {
            let values = report::pseudo_rand(64, seed);
            let total = 1_600 + (values[0].abs() * 100_000.0) as usize;
            let mut cuts: Vec<usize> = vec![0];
            for (i, value) in values.iter().enumerate().skip(1) {
                let step = ((value + 1.0) / 2.0 * 1_000.0) as usize + 1;
                cuts.push((cuts[i - 1] + step).min(total));
            }
            cuts.push(total);
            cuts.dedup();
            let regions: Vec<Region> = cuts
                .windows(2)
                .map(|pair| region(pair[0], pair[1], None, None))
                .collect();
            let samples = total * 3;
            let out = retime(regions, 48_000, samples);
            assert!(
                out.first().is_some_and(|region| region.start == 0),
                "seed {seed}: the first region starts at 0"
            );
            assert_eq!(
                out.last().map(|region| region.end),
                Some(samples),
                "seed {seed}: the last region ends at the input's length"
            );
            assert!(
                out.iter().all(|region| region.start < region.end),
                "seed {seed}: every region has a positive length"
            );
            assert!(
                out.windows(2).all(|pair| pair[0].end == pair[1].start),
                "seed {seed}: regions meet without a hole or an overlap"
            );
        }
    }

    fn transcription(regions: Vec<Region>) -> Transcription {
        Transcription {
            sample_rate: RATE,
            samples: 300,
            gain: 1.0,
            regions,
        }
    }

    #[test]
    fn the_result_contract_holds_in_every_build() {
        let good = transcription(vec![
            region(0, 100, None, Some("Music")),
            region(100, 300, Some("hello"), None),
        ]);
        assert!(good.check().is_ok(), "a tiling result is accepted");

        let mut hole = good.clone();
        hole.regions[1].start = 120;
        assert!(hole.check().is_err(), "a gap between regions is refused");

        let mut short = good.clone();
        short.regions[1].end = 299;
        assert!(
            short.check().is_err(),
            "a result short of its input ends early is refused"
        );

        let mut empty = good.clone();
        empty.regions[1].text = None;
        assert!(
            empty.check().is_err(),
            "a region with neither answer is refused"
        );

        let mut both = good.clone();
        both.regions[1].label = Some("Speech".to_string());
        assert!(
            both.check().is_err(),
            "a label beside a reliable transcript is refused"
        );
    }
}
