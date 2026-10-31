// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! "Transcribe any audio": where the speech is, how a long recording is cut into
//! recognition chunks, which regions tile the input, which transcripts are readings of
//! the audio rather than the model's placeholders — and the model-backed call that runs
//! the detector, the tagger and the recogniser over that plan.
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
//! input. A transcript the guards refuse is not returned at all: it is the model's
//! placeholder or a loop, not a reading of the audio, and the region says so with
//! its label and `read = false` rather than handing a caller words nobody said.
//!
//! What this module does **not** establish is anything about a transcript's
//! content: the guards detect loops, density and placeholders, not wrong words. The
//! pinned music pool the music check reads lives in [`crate::music`], not in this
//! surface.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use crate::ced::{Tag, Tagger};
use crate::graphs;
use crate::host;
use crate::listen;
use crate::loader::{Pair, Role};
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
/// The frame counts a chunk's graphs are compiled at, in the encoder's own frames
/// (eight per audio token). A chunk is padded up to the next rung and no further, so
/// the graphs a transcription compiles come from this closed set and each rung's
/// compile serves every chunk of the rung for the life of the process
/// (the ladder's `rung` cache) — a chunk's exact length is not a shape.
///
/// The padding is silence past the end of the piece: it cannot move a word, and the
/// measure that keeps it honest is the reading itself — every product call runs with it
/// in place. The top rung is above the longest
/// piece a cut can leave — [`CUT_SECONDS`] plus the window a cut may search back over —
/// so no piece is ever truncated to fit.
pub const FRAME_RUNGS: [usize; 6] = [128, 256, 384, 512, 640, 768];
/// The length every cut aims at, in seconds: above it this recogniser spends its context
/// on preamble rather than on the words, and a longer piece costs a quadratic prompt
/// prefill. A piece is at most this plus [`CUT_EARLY_HOPS`].
///
/// Seven is where the two bars meet, measured both ways on the pinned corpus: a chunk
/// costs about 0.7 s of wall time whatever its length — the prompt's own prefill and a
/// decode apiece — so a shorter aim multiplies that cost while the recogniser gains only
/// the hard spans it could not read (five-second pieces read four of them and cost
/// 125.9 ms per audio second against 98.9 here). The trade is recorded, not tuned away.
pub const CUT_SECONDS: usize = 7;
/// How far before its aim a cut may search for a quiet hop (0.5 s). A cut never moves
/// later than its aim, so the piece count is the span's length alone.
pub const CUT_EARLY_HOPS: usize = 32;
/// The ladder covers the cut: the top rung is at or above the longest piece a cut can
/// leave, so no chunk ever needs a rung the ladder does not have.
const _: () = assert!(
    FRAME_RUNGS[FRAME_RUNGS.len() - 1]
        >= (CUT_SECONDS * RATE as usize + CUT_EARLY_HOPS * HOP_SAMPLES) / crate::graphs::MEL_HOP
);
/// The least spread, in dB, that the frame loudness of a chunk must have before the
/// recognition prompt declares a language for it: the tenth to the ninetieth percentile
/// of its 20 ms frame energies. Speech and singing vary by tens of dB over a few
/// seconds — pauses, syllables, breaths — while a steady signal (noise, a hum, a drone)
/// stays within a few. A steady span is still sent to the recogniser; it is only not
/// *told* the language, so the model's own "no speech" answer stands for it, which is
/// what keeps a hallucinated transcript off audio that holds no words.
pub const SPEECH_SPREAD_DB: f32 = 11.0;
/// The frame the spread is measured over: 20 ms at [`RATE`].
const SPREAD_FRAME: usize = 320;
/// The longest word overlap [`overlap_words`] reports: an overlap longer than a region
/// can end with is not a seam's.
const OVERLAP_WORDS: usize = 6;

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

/// `span` cut into pieces of at most [`CUT_SECONDS`] plus one search window: the piece
/// count is the span's length divided by the aim, rounded up, and each cut lands on the
/// quietest hop in the half second before its even share — so a cut sits in a pause rather
/// than in a word, never moves later than its share, and the span's trailing silence never
/// becomes a piece of its own.
///
/// The bound is the aim plus [`CUT_EARLY_HOPS`]: a cut may move earlier, which lengthens
/// the piece after it by at most that window. [`FRAME_RUNGS`]' top rung covers it, which is
/// what keeps a piece from ever being truncated to fit.
fn cut(span: Span, audio: &[f32]) -> Vec<Span> {
    let aim = CUT_SECONDS * RATE as usize;
    let len = span.end.saturating_sub(span.start);
    if len <= aim {
        return vec![span];
    }
    // Enough pieces that every one is at most the aim: a span is never one long piece with
    // a short remainder, which is what left twelve- and fourteen-second pieces the
    // recogniser answered with nothing.
    let count = len.div_ceil(aim).max(1);
    let early = CUT_EARLY_HOPS * HOP_SAMPLES;
    let mut out = Vec::with_capacity(count);
    let mut start = span.start;
    for k in 1..count {
        // The even share, and the quietest hop in the half second before it: a cut lands in
        // a pause rather than in a word, and never after its share.
        let target = span.start + len * k / count;
        let from = target.saturating_sub(early).max(start + MIN_CHUNK_SAMPLES);
        let at = quietest_hop(audio, from, target)
            .unwrap_or(target)
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
fn readings_by_chunk(count: usize, readings: Vec<(usize, Option<String>)>) -> Vec<Option<String>> {
    let mut answers: Vec<Option<String>> = (0..count).map(|_| None).collect();
    for (index, answer) in readings {
        answers[index] = answer;
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

/// Words in a transcript at or below which, over a chunk of at least [`FILLER_SECONDS`], it
/// is the model's placeholder rather than a reading of what is there: the recogniser answers
/// audio it cannot read with one generic word ("Okay.", "Hello.", "Thank you.") over
/// seconds of it. Three words and up is kept — a short utterance can be that short, and the
/// reading is the caller's to judge. See `guard`.
pub const FILLER_WORDS: usize = 2;

/// The shortest chunk the filler count applies to: two words in a second and a half is a
/// short utterance, not a placeholder, and a chunk shorter than this has too little audio
/// for the answer's length to mean anything.
pub const FILLER_SECONDS: f64 = 3.0;

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

/// Whether a reading says anything a language is written with: the model's own
/// "no speech" answer arrives empty, as whitespace, or as a lone punctuation mark.
fn spoken(text: &str) -> bool {
    text.trim().chars().any(char::is_alphanumeric)
}

/// What the guards leave of one chunk's transcript: `Some` when it is a reading the call
/// stands behind, `None` when it is not a reading at all.
///
/// A transcript is refused when it
/// * holds nothing a language is written with — empty, whitespace, or punctuation only,
///   which is the model's own "no speech" answer;
/// * loops — any [`REPEAT_GRAM`]-word gram more than [`REPEAT_LIMIT`] times, or one word
///   run more than [`WORD_REPEAT_LIMIT`] times — the model circling instead of reading;
/// * is denser than [`DENSITY_CEILING`] characters per second of its own audio, which is
///   faster than anyone speaks;
/// * says no more than [`FILLER_WORDS`] over a chunk at least [`FILLER_SECONDS`] long: the
///   one generic word ("Okay.", "Hello.") the recogniser answers audio it cannot read with,
///   over seconds of it.
///
/// A refused transcript is never returned, not even marked: a caller that cannot tell
/// filler from a reading has no use for either. The region is then speech found and not
/// read, and carries its label.
fn guard(text: &str, seconds: f64) -> Option<String> {
    // Nothing at all, or nothing a language is written with: the model's own
    // "no speech" answer arrives both ways, and a lone punctuation mark is not a
    // reading of the audio either.
    if !spoken(text) {
        return None;
    }
    let trimmed = text.trim();
    let words: Vec<String> = trimmed.split_whitespace().map(str::to_lowercase).collect();
    // One word held for more than a dozen turns.
    let mut run = 1;
    for pair in words.windows(2) {
        if pair[0] == pair[1] {
            run += 1;
            if run > WORD_REPEAT_LIMIT {
                return None;
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
                return None;
            }
        }
    }
    // A region with no audio and text is infinitely dense; so is one whose
    // duration is not a number. Both are marks, not crashes.
    let density = trimmed.chars().count() as f64 / seconds;
    if seconds <= 0.0 || !density.is_finite() || density > DENSITY_CEILING {
        return None;
    }
    // The placeholder: two words, or one, over seconds of audio.
    if seconds >= FILLER_SECONDS && words.len() <= FILLER_WORDS {
        return None;
    }
    Some(text.to_string())
}

// ── the aligned input ───────────────────────────────────────────────

/// The frame count a chunk of `frames` mel frames is compiled at: the first rung of
/// [`FRAME_RUNGS`] at or above it. A chunk longer than the top rung is refused rather
/// than truncated — [`cut`] never leaves one, so this is the refusal that keeps a
/// future change to [`CUT_SECONDS`] from silently dropping the tail of every piece.
fn rung_frames(frames: usize) -> Result<usize> {
    FRAME_RUNGS
        .iter()
        .copied()
        .find(|rung| *rung >= frames)
        .with_context(|| {
            format!(
                "a chunk of {frames} frames is longer than the ladder's top rung ({})",
                FRAME_RUNGS[FRAME_RUNGS.len() - 1]
            )
        })
}

/// The spread of a chunk's frame loudness, in dB: the tenth to the ninetieth percentile
/// of its [`SPREAD_FRAME`] frame energies. See [`SPEECH_SPREAD_DB`].
fn spread_db(audio: &[f32], span: Span) -> f32 {
    let mut levels: Vec<f32> =
        Vec::with_capacity(span.end.saturating_sub(span.start) / SPREAD_FRAME + 1);
    let mut at = span.start;
    while at + SPREAD_FRAME <= span.end.min(audio.len()) {
        let energy: f32 = audio[at..at + SPREAD_FRAME]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / SPREAD_FRAME as f32;
        levels.push(10.0 * energy.max(1e-12).log10());
        at += SPREAD_FRAME;
    }
    if levels.is_empty() {
        return 0.0;
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |f: f64| levels[((levels.len() - 1) as f64 * f) as usize];
    at(0.9) - at(0.1)
}

/// `audio` truncated (or zero-padded) to exactly `samples` samples — the
/// `frames * 160` a rung's graph takes.
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
    /// `Some` exactly when [`Region::read`]: a reading the call stands behind. A refused
    /// transcript is not returned (`transcribe::guard`), so there is no third state.
    pub text: Option<String>,
    /// What the tagger heard here: always on a region no reading produced — what the
    /// audio is, which may be a non-speech class or a speech one. A region that was read
    /// carries no label.
    pub label: Option<String>,
    /// `true` when the plan sent this region's audio to the recogniser and came back with a
    /// transcript. A region with `read: false` is not silence: it is audio the plan found
    /// no reading for — speech the model answered nothing on, or speech whose answer the
    /// guards refused as filler, or audio segmentation never called speech — and its label
    /// says which. The caller can therefore tell "nothing there" from "we failed to read
    /// it" without inferring it from an empty string.
    pub read: bool,
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

    /// The result's own contract, checked in every build: a region answers with a text or
    /// with a label and never both, the regions are ordered without a hole or an overlap
    /// inside `[0, samples]`, the first starts at 0 and the last ends at `samples`. A
    /// caller that composes a `Transcription` — the tests do — gets the same check the
    /// call itself runs, rather than a `debug_assert` a release build strips.
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
            if region.read != region.text.is_some() {
                bail!(
                    "the region at {at:.3} s says read={} with {} text",
                    region.read,
                    if region.text.is_some() { "a" } else { "no" }
                );
            }
            if !region.read && region.label.is_none() {
                bail!(
                    "the region at {at:.3} s was not read and carries no label: a caller could \
                     not tell what the audio is"
                );
            }
            if region.read && region.label.is_some() {
                bail!("the region at {at:.3} s carries a label beside its transcript");
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

/// The detector, the tagger and the recogniser over one recognition pair.
///
/// The three are held together because one call uses all three: the detector
/// answers where the speech is, the tagger what the region sounds like, the
/// recogniser what was said. All three are the caller's — the pair, its tagger —
/// so a caller that opened a pair from an explicit path cannot be paired with a
/// tagger resolved from somewhere else.
///
/// What a call compiles is the **ladder's** rungs and nothing else: a chunk is padded
/// up to one of [`FRAME_RUNGS`] and that rung's graphs are compiled once for the process
/// (the ladder's `rung` cache), so a pass pays for the rungs no earlier call placed rather than a
/// graph set per chunk length. The cache is bounded by construction — its key comes from
/// the closed ladder and it holds at most the ladder's own number of rungs — and a rung the
/// process has never placed is a compile, never a wrong answer.
/// [`crate::release_process_memory`] gives all of it back.
pub struct Transcriber {
    pair: Arc<Pair>,
    tokenizer: Tokenizer,
    tagger: Tagger,
    /// The resident-set ceiling an input is refused against: the bound the caller's
    /// run is held to, so an input is refused against the figure that run enforces
    /// rather than against a second read of the environment.
    ceiling: u64,
}

impl Transcriber {
    /// Open the tokenizer over `pair` and hold `tagger`. Both are required:
    /// without the tagger a non-speech region could not carry a label, and a region
    /// never comes back empty. `ceiling_mib` is the resident-set bound the run is
    /// held to — [`crate::guard::Bounds::max_rss_mib`] for a run whose bounds were
    /// resolved once.
    pub fn open(pair: Arc<Pair>, tagger: Tagger, ceiling_mib: u64) -> Result<Self> {
        let tokenizer = Tokenizer::from_model(&pair.body).context("the recognition tokenizer")?;
        Ok(Self {
            pair,
            tokenizer,
            tagger,
            ceiling: ceiling_mib.saturating_mul(1024 * 1024),
        })
    }

    /// The product's recogniser, opened from the inputs a run resolves: the
    /// recognition pair and the audio tagger every recognition call depends on, and
    /// the resident-set bound the run itself enforces.
    pub fn open_product(ceiling_mib: u64) -> Result<Self> {
        let pair = Arc::new(Pair::of_role(Role::Recognition)?);
        let tagger = Tagger::open(&crate::loader::models_dir()?.join(crate::ced::CED_MODEL))?;
        Self::open(pair, tagger, ceiling_mib)
    }

    /// Transcribe any length of `audio` at `sample_rate`. Whole input, every
    /// time: the result's regions tile it, or the call returns an error saying
    /// why it could not. It never truncates.
    pub fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<Transcription> {
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
        // every boundary in the file. `prepare_speech` (the tests' harness) is
        // the single-shot path's conditioning and does drop a loud pre-roll; that
        // is why this path may not reuse it.
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

        // Where the speech is, how it is cut, and how the cuts tile the input.
        let mut detector = earshot::Detector::default();
        let mask = vad::speech_mask(audio16, &mut detector);
        let mut spans = Vec::new();
        // One label per surviving span, asked once: it decides whether the span can
        // hold words at all, and it is the label of every region that is the span
        // uncut.
        let mut span_labels: BTreeMap<Span, Option<String>> = BTreeMap::new();
        let window = self.tagger.window_frames() * self.tagger.samples_per_frame();
        for span in speech_spans(&mask, audio16.len()) {
            let label = region_label(&self.tagger, &tags, span);
            // A merged span the tagger calls non-speech is not presented to the
            // recogniser: its audio becomes a region's label instead of a pass that can
            // only invent words.
            if !removable(
                self.tagger.labels(),
                window,
                self.tagger.samples_per_frame(),
                &tags,
                span,
            ) {
                span_labels.insert(span, label);
                spans.push(span);
            }
        }
        // Each chunk's own label, from the span it was cut out of: it decides whether the
        // recogniser is given the language (`SPEECH_LABELS`) or left to answer with its
        // own.
        let mut chunk_list: Vec<Span> = Vec::new();
        let mut chunk_labels: Vec<Option<String>> = Vec::new();
        for span in spans {
            let label = span_labels.get(&span).cloned().flatten();
            for piece in cut(span, audio16) {
                chunk_list.push(piece);
                chunk_labels.push(label.clone());
            }
        }
        let plan = tile_indexed(&chunk_list, audio16.len());

        // The chunks grouped by the rung their graphs are compiled at, so a file whose
        // chunks share a rung — most of them, since a full piece is always the same
        // length — visits that rung's graphs once. A rung is compiled the first time
        // any call asks for it and kept process-wide (the ladder's `rung` cache): the call pays
        // only the rungs no earlier call placed, so the build line is a ladder, not a
        // per-chunk cost.
        let mut shapes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, chunk) in chunk_list.iter().enumerate() {
            shapes.entry(chunk_frames(*chunk)?).or_default().push(index);
        }
        // Each chunk's reading: the walk's own text through the guards, and never a
        // second pass — the plan cuts at the span the encoder was trained over, so a
        // chunk is read once. A walk is taken per shape and a shape holds chunks
        // scattered through the plan, so the readings are collected with the chunk's
        // own index and placed by it ([`readings_by_chunk`]), never by the order the
        // shapes were visited: a reading attached to its neighbour's region would
        // mistime the text.
        let mut readings: Vec<(usize, Option<String>)> = Vec::with_capacity(chunk_list.len());
        for (frames, indexes) in &shapes {
            let (mut clip, _) = listen::Clip::rung(&self.pair, &self.tokenizer, *frames)
                .with_context(|| format!("building the {frames}-frame recognition graphs"))?;
            for &index in indexes {
                let chunk = chunk_list[index];
                let seconds = (chunk.end - chunk.start) as f64 / f64::from(RATE);
                // The chunk's own length bounds what it can say: a walk capped
                // there cannot spend a minute looping over eighteen seconds of
                // singing, and the loop, if it happens, is still in the words.
                clip.set_max_new(chunk_budget(seconds));
                // The prompt's language rows, from the chunk's own tagger label and its
                // own loudness dynamics: a speech or singing chunk that varies the way
                // speech does is told the language, and everything else — instrumental
                // music, a noise span the detector called speech, a drone — is left with
                // the prompt the one-shot path uses, so the model answers with its own
                // language or with none. A steady signal asked for a language answers with
                // a generic word rather than words; `guard` refuses that answer.
                let told = chunk_labels[index].as_deref().is_some_and(speaks_words)
                    && spread_db(audio16, chunk) >= SPEECH_SPREAD_DB;
                let walk = clip
                    .run(
                        chunk_audio(audio16, chunk, *frames),
                        told.then_some(listen::ASR_LANGUAGE_IDS),
                    )
                    .with_context(|| {
                        format!("transcribing samples {}..{}", chunk.start, chunk.end)
                    })?;
                // A walk that ran out of its own budget was circling, not reading: the
                // words it did produce are not a reading of the chunk either.
                let answer = if walk.truncated {
                    None
                } else {
                    guard(&walk.text, seconds)
                };
                readings.push((index, answer));
            }
            // The clip is dropped here, at the end of its rung: a rung's graphs stay
            // resident in the ladder's cache, and this handle is only the walk.
        }
        let answers = readings_by_chunk(chunk_list.len(), readings);

        // The plan's regions, each with the answer that speaks for it: a speech
        // region takes its own chunk's transcript (the plan carries the index, so
        // no two regions can be confused for each other), a background region has
        // none and is described by the tagger alone.
        let mut regions: Vec<Region> = Vec::with_capacity(plan.len());
        for (span, chunk) in &plan {
            // A speech region takes its own chunk's reading; `None` is a chunk the pass
            // produced no reading for, which the region says with its label and `read`.
            let text = chunk
                .map(|index| answers[index].clone())
                .unwrap_or_default();
            // A region with no transcript says what it is: the tagger's label for the span
            // it is part of, asked for this region only when the span is not the region. A
            // region that was read carries no label — there is nothing to explain.
            let label = if text.is_none() {
                match span_labels.get(span) {
                    Some(label) => label.clone(),
                    None => region_label(&self.tagger, &tags, *span),
                }
            } else {
                None
            };
            regions.push(Region {
                start: span.start,
                end: span.end,
                read: text.is_some(),
                text,
                label,
            });
        }
        let out = Transcription {
            sample_rate,
            samples,
            gain,
            regions: retime(merge(regions), sample_rate, samples),
        };
        // The result's contract, checked here rather than only in a debug build: a
        // caller gets the same guarantee the measurement checks read.
        out.check().context("the transcription's own regions")?;

        Ok(out)
    }

    /// A WAV file read whole, at [`RATE`], through [`Transcriber::transcribe`] —
    /// no offset and no limit, so the result describes the file.
    pub fn transcribe_file(&mut self, path: &Path) -> Result<Transcription> {
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
        let audio = host::wav_mono_16k(&name, 0, projected.saturating_add(1))
            .with_context(|| format!("reading `{name}`"))?;
        self.transcribe(&audio, RATE)
    }

    /// Refuse an input the run cannot hold, naming both figures: the call's own
    /// whole-input buffers (`samples` of f32 audio, the tagger's log-mel over them,
    /// and `disk` bytes of the file being read) against the run's resident-set
    /// ceiling. Called before those buffers are allocated, so an impossible input is
    /// a diagnostic rather than an allocation the guard stops.
    fn ceiling(&self, samples: usize, disk: u64, path: Option<&Path>) -> Result<()> {
        let need = self.need(samples, disk);
        if need > self.ceiling {
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
             its audio and its tagger log-mel{disk} — past the run's {} MiB resident-set ceiling",
            samples as f64 / f64::from(RATE),
            need / (1024 * 1024),
            self.ceiling / (1024 * 1024),
        )
    }
}

/// The frame count a chunk's graphs are compiled at: its own length rounded up to the
/// next rung ([`FRAME_RUNGS`]).
fn chunk_frames(chunk: Span) -> Result<usize> {
    rung_frames(frames_of(chunk.end - chunk.start))
}

/// The audio one chunk is recognised from: the samples of `read` (its own, or one of its
/// pieces), padded to the frame count it is compiled at.
fn chunk_audio(audio: &[f32], read: Span, frames: usize) -> Vec<f32> {
    pad_to(
        audio.get(read.start..read.end).unwrap_or(&[]),
        frames * graphs::MEL_HOP,
    )
}

/// Whether a span may be removed from the recogniser's input rather than transcribed.
///
/// Only when the tagger has seen it more than once, and only when **no** window over it
/// names a label that can carry words ([`bears_words`]) — one such window keeps the
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
/// the longest overlap up to `OVERLAP_WORDS`, compared without case or punctuation.
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

/// The tagger's labels that say the audio there is *speech or singing* — the two
/// families where the recogniser is given the language in its prompt
/// ([`listen::ASR_LANGUAGE_IDS`]) instead of being left to identify it. Every song
/// rule's clips are here; every instrumental music class is not, because a declared
/// language over an orchestra produces a confident, wrong transcript, which is the one
/// thing the music pool must never see.
pub const SPEECH_LABELS: &[&str] = &[
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
];

/// The word-bearing music classes the song rule offers to the recogniser. Separated from
/// [`SPEECH_LABELS`] because the two families are used for different decisions — a music
/// class may hold words but never speaks them.
const MUSIC_LABELS: &[&str] = &[
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

/// Whether a tagger label says the audio there is speech or singing.
fn speaks_words(label: &str) -> bool {
    let name = label.trim();
    SPEECH_LABELS
        .iter()
        .any(|class| name.eq_ignore_ascii_case(class))
}

/// Whether a tagger label says the audio there can hold words: a speech or singing class
/// ([`SPEECH_LABELS`]) or one of the word-bearing music ones ([`MUSIC_LABELS`]), matched
/// on the name the file carries.
fn bears_words(label: &str) -> bool {
    let name = label.trim();
    MUSIC_LABELS
        .iter()
        .any(|word| name.eq_ignore_ascii_case(word))
        || speaks_words(name)
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
            if !last.read && !region.read && last.label == region.label {
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

    /// Spans walked through [`cut`] one by one: the production path the tests below take,
    /// with the span's label left out (that is what the call itself carries).
    fn chunks(spans: Vec<Span>, audio: &[f32]) -> Vec<Span> {
        spans
            .into_iter()
            .flat_map(|span| cut(span, audio))
            .collect()
    }

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

    /// A span of three aims is cut into three pieces, each cut landing in the pause
    /// nearest its even share of the span.
    #[test]
    fn a_span_of_three_aims_cuts_into_three_pauses() {
        const NOTCH: usize = 2_048;
        let aim = CUT_SECONDS * RATE as usize;
        let mut audio = vec![0.1f32; 3 * aim];
        for at in [aim, 2 * aim] {
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
        assert_eq!(cuts.len(), 3, "three aims are three pieces");
        assert_eq!(
            cuts[0].start, 0,
            "the first piece starts where the span does"
        );
        assert_eq!(
            cuts.last().expect("three pieces").end,
            audio.len(),
            "the last piece ends where the span does"
        );
        for (i, at) in [aim, 2 * aim].into_iter().enumerate() {
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

    /// A cut lands on the quietest hop in the half second before its share of the span,
    /// and never after it, so no piece is longer than the aim and no piece is a
    /// fragment.
    #[test]
    fn a_cut_lands_in_the_quietest_hop_before_its_share() {
        let aim = CUT_SECONDS * RATE as usize;
        let span = 5 * aim;
        // On the detector's own hop grid: a quiet hop is a hop, not a sample index.
        let notches: Vec<usize> = (1..5)
            .map(|k| (k * span / 5 - HOP_SAMPLES * 8) / HOP_SAMPLES * HOP_SAMPLES)
            .collect();
        let average = 0.5f32;
        let audio: Vec<f32> = (0..span)
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
        assert_eq!(cuts.len(), 5, "five aims are five pieces");
        for (cut, notch) in cuts.iter().zip(&notches) {
            assert!(
                cut.end.abs_diff(*notch) <= HOP_SAMPLES * 8,
                "each cut takes the pause inside its own window: {} against {notch}",
                cut.end
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
                len <= aim + CUT_EARLY_HOPS * HOP_SAMPLES,
                "no piece reaches past the aim and the cut's own window: {:?}",
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
            end: 1_000 + CUT_SECONDS * RATE as usize,
        };
        assert_eq!(
            chunks(vec![shorter], &audio),
            vec![shorter],
            "a span under the limit is left alone"
        );
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

    /// The seam rule: two adjacent text regions that end and begin with the same whole
    /// words carry one of them twice.
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
        let audio = vec![0.1f32; 2 * aim];
        let span = Span {
            start: 0,
            end: aim + MIN_CHUNK_SAMPLES / 2,
        };
        assert_eq!(
            chunks(vec![span], &audio).len(),
            2,
            "anything past the aim is cut, so no piece is longer than one aim"
        );
        // 61 s: every piece inside the aim and its window, none of them a fragment.
        let audio = vec![0.1f32; 61 * RATE as usize];
        let span = Span {
            start: 0,
            end: audio.len(),
        };
        let cuts = chunks(vec![span], &audio);
        assert!(cuts.len() >= 4, "61 s is at least four pieces");
        // A cut never moves later than its share, so a piece is at most one aim plus
        // the window a cut may search back over.
        for piece in &cuts {
            assert!(
                piece.end - piece.start <= aim + CUT_EARLY_HOPS * HOP_SAMPLES,
                "no piece reaches past the aim and its own window: {piece:?}"
            );
            assert!(
                piece.end - piece.start >= MIN_CHUNK_SAMPLES,
                "no piece is a fragment: {piece:?}"
            );
        }
        for pair in cuts.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "the pieces tile the span");
        }
    }

    /// A cut searches only backwards: the window is the half second before its share,
    /// so a span of two aims is two pieces of at most one aim each rather than one long
    /// piece and a fragment.
    #[test]
    fn a_short_spans_cut_never_reaches_past_its_share() {
        let aim = CUT_SECONDS * RATE as usize;
        let audio = vec![0.1f32; 2 * aim - aim / 3];
        let cuts = chunks(
            vec![Span {
                start: 0,
                end: audio.len(),
            }],
            &audio,
        );
        assert_eq!(cuts.len(), 2, "a span of nearly two aims is two pieces");
        assert!(
            cuts[0].end <= audio.len() / 2,
            "the cut stayed at or before its share: {} against {}",
            cuts[0].end,
            audio.len() / 2
        );
    }

    /// A chunk's reading is its own: the walks are taken shape by shape, so the order
    /// they ran in is not the plan's order and a reading is placed by its chunk index.
    #[test]
    fn each_chunk_keeps_its_own_reading() {
        let reading = |text: &str| Some(text.to_string());
        // Two shapes, visited shortest first: the second chunk of the plan is the one
        // the first shape read.
        let answers = readings_by_chunk(
            2,
            vec![
                (1, reading("the short piece")),
                (0, reading("the long piece")),
            ],
        );
        assert_eq!(answers[0].as_deref(), Some("the long piece"));
        assert_eq!(answers[1].as_deref(), Some("the short piece"));
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
                None,
                "an empty transcript is the model's own no-speech"
            );
        }
    }

    #[test]
    fn an_ordinary_sentence_is_a_reading() {
        assert_eq!(
            guard("the meeting starts in ten minutes", 2.5).as_deref(),
            Some("the meeting starts in ten minutes"),
            "the text is kept as it came"
        );
    }

    #[test]
    fn a_repeated_gram_is_refused() {
        assert_eq!(
            guard("la la la la la la la", 2.0),
            None,
            "a four-word gram four times over is a loop"
        );
    }

    #[test]
    fn one_word_written_thirteen_times_is_refused() {
        assert_eq!(
            guard(&"word ".repeat(13), 5.0),
            None,
            "a run of 13 words is a loop"
        );
    }

    #[test]
    fn density_refuses_text_that_outruns_its_audio() {
        let text = "the meeting starts in ten minutes tod";
        assert_eq!(
            guard(text, 0.5),
            None,
            "38 characters in 0.5 s is 76 per second"
        );
        assert!(
            guard(text, 3.0).is_some(),
            "the same 38 characters over 3 s are 12.7 per second, and eight words is a sentence"
        );
    }

    /// The failure this feature exists to remove: a generic word over seconds of
    /// unclear audio is the model's placeholder, and it is never returned as a reading.
    #[test]
    fn a_generic_word_over_seconds_of_audio_is_refused() {
        for text in ["Okay.", "Hello.", "Yeah.", "Thank you."] {
            assert_eq!(
                guard(text, 7.0),
                None,
                "`{text}` over seven seconds is not a reading of them"
            );
        }
        // A short utterance over short audio is a reading: the floor is a rate, and it
        // only applies once there is enough audio for a rate to mean anything.
        assert!(
            guard("Yes.", 1.0).is_some(),
            "one word in a second is speech"
        );
        assert!(
            guard("Okay, let us begin.", 2.0).is_some(),
            "four words in two seconds is speech"
        );
    }

    // ── the result, without a model ────────────────────────────────

    fn region(start: usize, end: usize, text: Option<&str>, label: Option<&str>) -> Region {
        Region {
            start,
            end,
            read: text.is_some(),
            text: text.map(str::to_string),
            label: label.map(str::to_string),
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
            let values = crate::report::pseudo_rand(64, seed);
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

        // A reading that produced nothing keeps its text out of the answer and must say
        // what the audio is instead: the label is the whole answer of an unread region.
        let mut unread = good.clone();
        unread.regions[1] = region(100, 300, None, None);
        assert!(
            unread.check().is_err(),
            "a region with neither a transcript nor a label is refused"
        );

        // "Speech, found but not read" is a legal region: the label says the audio holds
        // words and nothing says they were read.
        let mut claimed = good.clone();
        claimed.regions[1] = region(100, 300, None, Some("Speech"));
        assert!(
            claimed.check().is_ok(),
            "\"speech, found but not read\" is a legal region"
        );
        claimed.regions[1].read = true;
        assert!(
            claimed.check().is_err(),
            "a region marked read with no transcript is refused"
        );

        let mut both = good.clone();
        both.regions[1].label = Some("Speech".to_string());
        assert!(
            both.check().is_err(),
            "a label beside a transcript is refused"
        );
    }

    #[test]
    fn the_ladder_covers_the_cut_and_refuses_a_longer_chunk() {
        // The refusal instead of a clamp: a chunk longer than the top rung would
        // otherwise have its tail cut off silently by the padding.
        assert!(
            rung_frames(FRAME_RUNGS[FRAME_RUNGS.len() - 1]).is_ok(),
            "the top rung is a rung"
        );
        assert!(
            rung_frames(FRAME_RUNGS[FRAME_RUNGS.len() - 1] + 1).is_err(),
            "a chunk past the ladder is refused, not truncated"
        );
        // And no cut can leave one: with a span far longer than the aim, every piece is
        // inside the aim and the window a cut may search back over, which the top rung
        // covers.
        let aim = CUT_SECONDS * RATE as usize;
        let audio = vec![0.5f32; 4 * aim + 1_234];
        let span = Span {
            start: 0,
            end: audio.len(),
        };
        let pieces = cut(span, &audio);
        assert!(
            pieces.len() > 3,
            "four aims and a remainder are four pieces"
        );
        for piece in &pieces {
            let frames = frames_of(piece.end - piece.start);
            assert!(
                piece.end - piece.start <= aim + CUT_EARLY_HOPS * HOP_SAMPLES,
                "a piece is inside the aim and its own search window"
            );
            assert!(
                frames <= FRAME_RUNGS[FRAME_RUNGS.len() - 1],
                "a piece of {frames} frames is inside the ladder"
            );
            assert!(
                rung_frames(frames).is_ok(),
                "every piece a cut leaves has a rung"
            );
        }
    }

    #[test]
    fn loudness_spread_tells_a_steady_signal_from_a_modulated_one() {
        // A steady level is what a noise span the detector called speech looks like: no
        // p10-to-p90 spread, so no declared language for it.
        let steady = vec![0.1f32; SPREAD_FRAME * 200];
        let span = Span {
            start: 0,
            end: steady.len(),
        };
        assert!(
            spread_db(&steady, span) < SPEECH_SPREAD_DB,
            "a constant level has no spread"
        );
        // Syllables: a quiet and a loud frame alternating is tens of dB.
        let mut syllables = vec![0.0f32; SPREAD_FRAME * 200];
        for (i, frame) in syllables.chunks_mut(SPREAD_FRAME).enumerate() {
            let level = if i % 2 == 0 { 0.005 } else { 0.5 };
            frame.fill(level);
        }
        assert!(
            spread_db(&syllables, span) >= SPEECH_SPREAD_DB,
            "a modulated signal spreads past the gate"
        );
    }
}
