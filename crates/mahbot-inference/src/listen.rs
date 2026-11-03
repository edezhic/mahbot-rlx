// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The listening path: mel front end, encoder and recognition on this crate's
//! own ASR pair.
//!
//! [`Listener::embed`] is the wake decision's input — one VAD-gated speech
//! window through the sound encoder and the projector, mean-pooled and
//! L2-normalized — and [`Listener::embed_pair`] scores two consecutive windows in
//! one batched pass, which is the form the detector runs. The real-audio sweep
//! takes the same front end cut in two (`GroupFront`): the conv front per window,
//! the encoder tail and the projector once for a group of windows.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::graphs::{self, Compiled};
use crate::host;
use crate::loader::{Config, Head, Pair};
use crate::report;
use crate::tokenizer::Tokenizer;

/// Audio rate the listening path takes.
pub const SAMPLE_RATE: u32 = 16_000;

/// Mel frames in the wake window: 76 frames at a 10 ms hop. The mel front end
/// computes one frame past the window and drops it.
pub const MEL_FRAMES: usize = 76;

/// Samples in the wake window: [`MEL_FRAMES`] hops.
pub const WINDOW_SAMPLES: usize = 12_160;
/// The front end refuses a window that is not whole hops, so the wake window's own
/// length and the front end's must be the same number.
const _: () = assert!(WINDOW_SAMPLES == MEL_FRAMES * graphs::MEL_HOP);

/// Samples between two consecutive scoring windows: 16 mel frames at the
/// encoder's 160-sample hop, i.e. the 160 ms stride a live ear advances by. It is
/// also the distance between the two windows of one [`PAIR`] and, therefore, the
/// most a fire can be resolved later than a one-window pass would resolve it.
pub const SCORE_STRIDE_SAMPLES: usize = 16 * graphs::MEL_HOP;

/// The most audio the harnesses that read a pinned recording take: 30 s (3 000 mel
/// frames) — a bound on the fixture, not a limit of the reading path, which cuts any
/// length into pieces.
pub const PASS_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

/// The decode budget — the most tokens one segment may generate; the cache is
/// sized in 128-row buckets above the prompt, not by this number.
pub const ASR_MAX_NEW: usize = 2_048;

/// The recognition prompt's fixed ends, in the tokenizer's own ids: 3 head rows
/// and 6 text rows before the encoder's audio rows, 6 tail rows after them. The
/// audio count is the piece's (80 rows for a 640-frame piece), so the prompt's
/// length is not a constant.
pub const ASR_HEAD_IDS: [usize; 3] = [151644, 8948, 198];
pub const ASR_TEXT_IDS: [usize; 6] = [151645, 198, 151644, 872, 198, 151669];
pub const ASR_TAIL_IDS: [usize; 6] = [151670, 151645, 198, 151644, 77091, 198];
pub const ASR_EOS: usize = 151645;
/// The three tokens a recognition answer opens with — `language English <asr_text>`,
/// written **into the prompt** by the transcription path instead of being generated. The
/// recogniser identifies the language itself, and on noisy speech it answers
/// `language None` and stops, so the whole chunk comes back empty — one 26.8 s voice
/// message did, and every one of its pieces. With the tag supplied the model transcribes
/// the audio's own language regardless of it (a tag naming another language does not
/// translate the answer) and the tag never reaches a transcript: everything up to
/// [`ASR_TEXT`] is stripped. Only a chunk the tagger calls speech or singing *and* whose
/// loudness varies the way speech does is told the language
/// ([`crate::transcribe::SPEECH_SPREAD_DB`]); every other chunk gets the prompt the
/// one-shot path uses, whose answer is the model's own. That boundary is what keeps a
/// declared language from putting a confident transcript on a steady signal.
pub const ASR_LANGUAGE_IDS: [usize; 3] = [11528, 6364, 151704];
/// The `<asr_text>` token the transcript follows; also the marker a zero-shot answer's
/// language tag ends with.
pub const ASR_TEXT: usize = 151704;

/// The ASR config's `eos_token_id` is 151645; 151643 is the vocabulary's
/// `<|endoftext|>`, not a second config EOS. The path stops on either.
pub const ASR_EOS_ALSO: usize = 151643;

pub fn is_asr_eos(id: usize) -> bool {
    id == ASR_EOS || id == ASR_EOS_ALSO
}

/// The front end's three graphs for one frame count: mel, encoder and projector.
/// Built and run in this order everywhere the path measures — the ear's window, a
/// recognition segment, the tagger's own front — so the input names and the frame
/// arithmetic are stated once.
///
/// `batch` independent windows ride one pass over each stage: the mel front end and
/// the convs carry their own batch axis and the tail and projector take the rows of
/// the whole batch, so a window's result does not depend on which slot it took.
///
/// The mel graph is not this clip's own: it has no weights and its shape is the frame
/// count and the batch, so every clip that reads the same shape runs the same compiled
/// graph ([`mel_graph`]). A transcription cuts its chunks onto a closed ladder of frame
/// counts ([`crate::transcribe::FRAME_RUNGS`]), so the one-window mel graphs a process
/// compiles are that ladder's — one each, however many chunks the pass reads, and shared
/// by every call rather than recompiled with the rung — and the ear's pair adds one more,
/// at its own batch.
pub(crate) struct Front {
    mel: Arc<Mutex<Compiled>>,
    encoder: Arc<Mutex<Compiled>>,
    projector: Arc<Mutex<Compiled>>,
    /// Independent windows one pass carries.
    batch: usize,
    /// Samples one independent window of this front is (`MEL_HOP * frames`), so a
    /// recognition chunk of `frames` frames is read in full rather than as a wake
    /// window. A pass hands the mel graph `batch` of them, `batch * samples` values.
    samples: usize,
    build_ms: f64,
}

/// The mel graphs a process keeps, keyed by `(frames, bins, batch)`: a small LRU, so the
/// ear's two (its window and its pair) stay warm while a transcription walks the rungs of
/// [`crate::transcribe::FRAME_RUNGS`], whose six are more than this holds — a mel graph it
/// comes back to is recompiled. An evicted entry is a recompile, never a wrong answer.
const MEL_GRAPH_ENTRIES: usize = 4;

/// The process's mel graphs and their least-recently-used order.
type MelGraphs = (
    HashMap<(usize, usize, usize), Arc<Mutex<Compiled>>>,
    std::collections::VecDeque<(usize, usize, usize)>,
);

fn mel_graphs() -> &'static Mutex<MelGraphs> {
    static MEL: OnceLock<Mutex<MelGraphs>> = OnceLock::new();
    MEL.get_or_init(|| Mutex::new((HashMap::new(), std::collections::VecDeque::new())))
}

/// The mel graph for one frame count, bin count and batch, compiled once for the process:
/// it has no weights, so two clips of the same shape share it exactly. Held behind a lock
/// because a run needs `&mut`.
///
/// Keyed by the caller's own audio shape, so the cache could otherwise collect one graph
/// per length a long-lived process is handed. It is bounded instead ([`MEL_GRAPH_ENTRIES`])
/// and cleared with the decode graph ([`clear_graph_caches`]).
fn mel_graph(frames: usize, bins: usize, batch: usize) -> Result<Arc<Mutex<Compiled>>> {
    let batch = batch.max(1);
    let mut guard = mel_graphs().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(graph) = guard.0.get(&(frames, bins, batch)) {
        let graph = Arc::clone(graph);
        guard.1.retain(|key| *key != (frames, bins, batch));
        guard.1.push_back((frames, bins, batch));
        return Ok(graph);
    }
    let graph = Arc::new(Mutex::new(Compiled::new(
        rlx_runtime::Device::Gpu,
        graphs::mel_frontend(frames, bins, batch)?,
    )?));
    while guard.0.len() >= MEL_GRAPH_ENTRIES {
        let Some(oldest) = guard.1.pop_front() else {
            break;
        };
        guard.0.remove(&oldest);
    }
    guard.0.insert((frames, bins, batch), Arc::clone(&graph));
    guard.1.push_back((frames, bins, batch));
    Ok(graph)
}

impl Front {
    pub(crate) fn build(pair: &Arc<Pair>, frames: usize, batch: usize) -> Result<Self> {
        let batch = batch.max(1);
        let t = crate::report::Stopwatch::start();
        let tokens = graphs::asr_tokens(frames);
        let samples = graphs::MEL_HOP * frames;
        let mel = mel_graph(frames, pair.sound.clip().mel_bins, batch)?;
        let encoder = Arc::new(Mutex::new(Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::asr_encoder(pair.sound.clone(), frames, batch)?,
        )?));
        let projector = Arc::new(Mutex::new(Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::projector_chain(pair.clone(), batch * tokens)?,
        )?));
        Ok(Self {
            mel,
            encoder,
            projector,
            batch,
            samples,
            build_ms: t.ms(),
        })
    }

    pub(crate) fn build_ms(&self) -> f64 {
        self.build_ms
    }

    /// One pass of the three graphs over `batch` windows, each exactly this front's
    /// own length ([`Self::samples`]), in slot order: each window's projected token
    /// rows (`tokens × projection_dim`). The caller pads — [`trailing_window`] is the
    /// wake path's own — so a window of another length is a named error, never a
    /// silent trim.
    fn forward(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let (batch, samples) = (self.batch, self.samples);
        if windows.len() != batch {
            return Err(anyhow!(
                "{} windows in a front end built for {batch}",
                windows.len()
            ));
        }
        let mut audio = Vec::with_capacity(batch * samples);
        for window in windows {
            if window.len() != samples {
                return Err(anyhow!(
                    "a {}-sample window in a front end built for {samples}",
                    window.len()
                ));
            }
            audio.extend_from_slice(window);
        }
        split_windows(self.pass(audio)?, batch)
    }

    /// The three graphs over one mel input of [`Self::batch`] windows: the projected
    /// token rows of the whole pass, `batch × tokens × projection_dim` values.
    fn pass(&self, audio: Vec<f32>) -> Result<Vec<f32>> {
        let mel = self
            .mel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .run(&[("audio".into(), audio)]);
        let encoder = self
            .encoder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .run(&[("mel".into(), first(mel, "mel front end")?)]);
        first(
            self.projector
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .run(&[("features".into(), first(encoder, "encoder")?)]),
            "projector",
        )
    }

    /// The three runs for one recognition chunk: its mel, the encoder and the projector.
    /// The caller's buffer is the mel graph's input — a chunk of exactly this front's
    /// length ([`Self::samples`]) is handed over rather than copied into a batch of one,
    /// and a chunk of any other length is a named error, never a silent trim. Only a
    /// one-window front serves this path; every caller here builds one
    /// ([`Self::build`] with `batch = 1`).
    pub(crate) fn run(&self, audio: Vec<f32>) -> Result<Vec<f32>> {
        let samples = self.samples;
        if audio.len() != samples {
            return Err(anyhow!(
                "a {}-sample chunk in a front end built for {samples}",
                audio.len()
            ));
        }
        self.pass(audio)
    }
}

/// One frame count of the transcription ladder: the front end and the prompt prefill, over
/// the pair's one shared decode graph. A transcription cuts its chunks onto the closed
/// ladder of [`crate::transcribe::FRAME_RUNGS`], so the ladder's rungs are the only frame
/// counts it ever compiles — each compiled once, held for the chunks that need it, and
/// dropped with the last of them ([`Clip::rung`]) instead of for the life of the process.
struct Rung {
    /// The pair these graphs were built over, and the pair a prefill compiled for a later
    /// chunk of the same rung is built over too.
    pair: Arc<Pair>,
    front: Front,
    /// The prompt prefill, one per prompt length: the prompt carries three declared
    /// language rows or not at all, and the two lengths are two graphs. A variant the
    /// rung's chunks never ask for is never compiled.
    prefills: Mutex<HashMap<usize, Arc<Mutex<Compiled>>>>,
    /// The decode graph: one per pair, shared by every rung of the ladder, because the
    /// capacity is one value for all of them. A step is weight-read bound, so the
    /// attention reading a few more masked rows costs a fraction of the step, and a
    /// decode graph per rung would park an arena, a k/v plane pair and a weight layout
    /// apiece for no arithmetic gain.
    decode: Arc<Shared>,
    tokens: usize,
}

/// The capacity every rung's decode graph is compiled at: the longest rung's prompt rows —
/// 3 head + 6 text + 768/8 = 96 audio + 6 tail = 111, or 114 with the three declared language
/// rows — plus the most tokens a chunk of the longest piece may generate
/// ([`crate::transcribe::CHUNK_TOKENS_PER_S`] over the 7.5 s a cut can leave = 76).
/// 190 at the worst, rounded up to the next multiple of eight. A walk that outgrows it grows
/// its own graph, as any clip does.
const RUNG_CAPACITY: usize = 192;

impl Rung {
    fn build(pair: &Arc<Pair>, frames: usize) -> Result<Self> {
        Ok(Self {
            pair: Arc::clone(pair),
            front: Front::build(pair, frames, 1)?,
            prefills: Mutex::new(HashMap::new()),
            decode: shared_decode(pair)?,
            tokens: graphs::asr_tokens(frames),
        })
    }

    /// This rung's prefill for `prompt_rows`, compiled the first time one of the rung's
    /// chunks asks for that length and kept for the rung's life. Built over this rung's
    /// own pair, so every graph of one rung is one pair's.
    fn prefill(&self, prompt_rows: usize) -> Result<Arc<Mutex<Compiled>>> {
        let mut guard = self.prefills.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(prefill) = guard.get(&prompt_rows) {
            return Ok(Arc::clone(prefill));
        }
        let prefill = Arc::new(Mutex::new(prefill_graph(&self.pair, prompt_rows)?));
        guard.insert(prompt_rows, Arc::clone(&prefill));
        Ok(prefill)
    }
}

/// The one decode graph of a pair, built on first ask: every rung of the ladder and every
/// chunk of a rung runs the same `[RUNG_CAPACITY]`-deep body. Held beside the clips rather
/// than inside one because the rungs come and go with the chunk groups, and this graph does
/// not: it is the pair's, not a rung's.
///
/// The pair is held **weakly**: the graph is what a warm call reuses, and it must not keep
/// a model file mapped after its last user has dropped it. A caller that opens the pair
/// again gets a new decode graph, because a second `Pair` over the same files is a second
/// set of weights.
fn decode_slot() -> &'static Mutex<Option<(Weak<Pair>, Arc<Shared>)>> {
    static DECODE: OnceLock<Mutex<Option<(Weak<Pair>, Arc<Shared>)>>> = OnceLock::new();
    DECODE.get_or_init(|| Mutex::new(None))
}

fn shared_decode(pair: &Arc<Pair>) -> Result<Arc<Shared>> {
    let mut guard = decode_slot().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((held, decode)) = guard.as_ref() {
        // Pointer identity, not the file path: a second `Pair` opened over the same files
        // is a second set of weights, and its graphs are not this one's.
        if held.upgrade().is_some_and(|held| Arc::ptr_eq(&held, pair)) {
            return Ok(Arc::clone(decode));
        }
    }
    let decode = Shared::new(decode_graph(pair, RUNG_CAPACITY)?);
    *guard = Some((Arc::downgrade(pair), Arc::clone(&decode)));
    Ok(decode)
}

/// Empty the decode graph that outlives the rungs and the process's mel graphs: what
/// [`crate::release_process_memory`] asks for. A clip still holding either keeps it alive;
/// nothing else does. A rung's own graphs are not here to clear — they belong to the clip
/// that built them and go when it does.
pub(crate) fn clear_graph_caches() {
    *decode_slot().lock().unwrap_or_else(|e| e.into_inner()) = None;
    let mut mel = mel_graphs().lock().unwrap_or_else(|e| e.into_inner());
    mel.0.clear();
    mel.1.clear();
}

/// The same front end cut where a group pass needs it: the conv front runs one
/// window per call and the tail — `conv_out`, the position table, the 24 CLIP
/// blocks, `post_ln` — runs over the whole group in one pass, with the projector
/// after it. That is where the weight reads and the multiplies amortise.
///
/// The conv front is left one window per call on purpose: `batch` here is the
/// sweep's sixteen, and the im2col panel a batched conv front reserves is
/// `batch` panels wide ([`crate::graphs::EncoderConvs`] takes a windows-wide batch
/// for the pair, where that panel is two). The tail is what sixteen windows share.
pub(crate) struct GroupFront {
    mel: Compiled,
    convs: Compiled,
    tail: Compiled,
    projector: Compiled,
    batch: usize,
    tokens: usize,
    dim: usize,
    flat: usize,
}

impl GroupFront {
    pub(crate) fn build(pair: &Arc<Pair>, frames: usize, batch: usize) -> Result<Self> {
        let batch = batch.max(1);
        let tokens = graphs::asr_tokens(frames);
        let bins = pair.sound.clip().mel_bins;
        let mel = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::mel_frontend(frames, bins, 1)?,
        )?;
        let convs = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::asr_encoder_front(pair.sound.clone(), frames)?,
        )?;
        let tail = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::asr_encoder_tail(pair.sound.clone(), frames, batch)?,
        )?;
        let projector = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::projector_chain(pair.clone(), batch * tokens)?,
        )?;
        Ok(Self {
            mel,
            convs,
            tail,
            projector,
            batch,
            tokens,
            dim: pair.sound.clip().projection_dim,
            flat: graphs::flat_width(bins),
        })
    }

    /// `windows.len() <= batch` windows in one tail pass and one projector pass,
    /// each window's projection taken out of its own token block. Unused group
    /// slots are zeros and their results are discarded.
    fn forward_group(&mut self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let (batch, tokens, flat) = (self.batch, self.tokens, self.flat);
        let mut rows = Vec::with_capacity(batch * tokens * flat);
        for slot in 0..batch {
            match windows.get(slot) {
                Some(window) => {
                    let mel = first(self.mel.run_slices(&[("audio", window)]), "mel front end")?;
                    let front = first(self.convs.run_slices(&[("mel", &mel)]), "conv front")?;
                    rows.extend_from_slice(&front);
                }
                None => rows.resize(rows.len() + tokens * flat, 0.0),
            }
        }
        let encoded = first(self.tail.run_slices(&[("flat", &rows)]), "encoder tail")?;
        let projected = first(
            self.projector.run_slices(&[("features", &encoded)]),
            "projector",
        )?;
        let mut split = split_windows(projected, batch)?;
        split.truncate(windows.len());
        Ok(split)
    }
}

/// Windows per encoder pass: one weight read for the group instead of one per
/// window per layer. Sixteen is where a group stops paying for itself — on this
/// machine a group of sixteen costs 227 ms against 626 ms for sixteen
/// single-window passes (2.75×), eight windows cost 22.7 ms per window and
/// thirty-two 14.2, no better than sixteen. Those three readings were taken
/// while the split was introduced, with a probe that is not in the tree. The
/// sweep's windows arrive at a 0.16 s stride, so a group of sixteen spans 2.56 s
/// of audio.
pub const GROUP: usize = 16;

/// Windows per scoring pass: one pass through the encoder's weights for two
/// consecutive windows, at the price of a fire resolving up to one stride
/// (0.16 s) later than a one-window pass would. The product stops at two — a
/// wider group amortises more of the pass and delays a fire by the group's whole
/// span, and one stride is the delay the ear accepts. The offline sweep, where no
/// fire has to be on time, buys its amortisation with [`GROUP`].
pub const PAIR: usize = 2;

/// The listening path for the ear: the wake decision's embedding, from one window
/// (the enrolment, the negative pool and the ungrouped streams) or from two
/// consecutive windows in one batched pass ([`PAIR`]). The sweep's group front end
/// (`GroupFront`) is built and held on its own, so neither path compiles a graph
/// the other runs.
///
/// Each front is built on first use: a caller that only ever calls [`embed`](Self::embed)
/// compiles the one-window graphs and not the pair's, and one that only ever calls
/// [`embed_pair`](Self::embed_pair) the other way round. A stream that fed a lone
/// window through the pair's slot would end up holding both, which is what a stream
/// that flushes does — the lone window is the one case a pair has no work for.
pub struct Listener {
    pair: Arc<Pair>,
    tokens: usize,
    dim: usize,
    single: Option<Front>,
    paired: Option<Front>,
}

impl Listener {
    /// The ear's front ends for the wake window (76 frames): the pair's graphs are
    /// compiled by the first [`embed_pair`](Self::embed_pair), the single window's
    /// by the first [`embed`](Self::embed).
    pub fn new(pair: Arc<Pair>) -> Self {
        Self {
            tokens: graphs::asr_tokens(MEL_FRAMES),
            dim: pair.sound.clip().projection_dim,
            pair,
            single: None,
            paired: None,
        }
    }

    /// The front end for `batch` windows, built by the first call that asks for it: a
    /// caller of one form never compiles the other's graphs. The two slots are the only
    /// batches this listener holds, so a third one is refused rather than served from
    /// the wrong front.
    fn front(&mut self, batch: usize) -> Result<&mut Front> {
        let slot = match batch {
            1 => &mut self.single,
            PAIR => &mut self.paired,
            other => {
                return Err(anyhow!(
                    "a {other}-window pass in a listener that holds a one-window and a \
                     {PAIR}-window front end"
                ));
            }
        };
        if slot.is_none() {
            *slot = Some(Front::build(&self.pair, MEL_FRAMES, batch)?);
        }
        Ok(slot.as_mut().expect("the front end was built just above"))
    }

    /// `batch` windows of the front end's own length in one pass → one pooled,
    /// unit-length embedding each, in slot order.
    fn embed_windows(&mut self, batch: usize, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let (tokens, dim) = (self.tokens, self.dim);
        let projected = self.front(batch)?.forward(windows)?;
        projected
            .iter()
            .map(|rows| embed_rows(rows, tokens, dim))
            .collect()
    }

    /// One window of mono 16 kHz audio → the encoder's window embedding, through
    /// the one-window front end — enrolment and the negative pool embed a clip at
    /// a time, and padding a lone window into a batched pass costs the pass's work.
    pub fn embed(&mut self, samples_16k: &[f32]) -> Result<Vec<f32>> {
        let window = trailing_window(samples_16k);
        Ok(self.embed_windows(1, &[&window])?.remove(0))
    }

    /// Two consecutive windows → their two embeddings from one pass through the
    /// mel front end, the convs, every encoder block and the projector. Each
    /// window keeps its own mel, its own conv rows and its own attention, so a
    /// window's embedding does not depend on the slot it took or on its neighbour.
    pub fn embed_pair(&mut self, first: &[f32], second: &[f32]) -> Result<[Vec<f32>; PAIR]> {
        let both = [trailing_window(first), trailing_window(second)];
        Ok(self
            .embed_windows(PAIR, &[&both[0], &both[1]])?
            .try_into()
            .expect("a two-window pass answers two embeddings"))
    }
}

/// Whatever turns a group of detection windows into one embedding each: the
/// sweep's [`GroupFront`], or a [`Listener`] for the ungrouped and paired streams.
pub(crate) trait Embedder {
    /// Up to [`GROUP`] windows → one embedding each. [`GroupFront`] spends one
    /// encoder tail pass and one projector pass over the whole group; a
    /// [`Listener`] pairs two windows in one pass and embeds a lone one alone.
    fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;
}

impl Embedder for GroupFront {
    fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        if windows.len() > self.batch {
            return Err(anyhow!(
                "a group of {} windows in a pipeline built for {}",
                windows.len(),
                self.batch
            ));
        }
        let windows: Vec<Vec<f32>> = windows.iter().map(|w| trailing_window(w)).collect();
        let refs: Vec<&[f32]> = windows.iter().map(Vec::as_slice).collect();
        let (tokens, dim) = (self.tokens, self.dim);
        self.forward_group(&refs)?
            .into_iter()
            .map(|projected| embed_rows(&projected, tokens, dim))
            .collect()
    }
}

impl Embedder for Listener {
    fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if let [first, second] = windows {
            return Ok(self.embed_pair(first, second)?.to_vec());
        }
        windows.iter().map(|window| self.embed(window)).collect()
    }
}

/// The trailing [`WINDOW_SAMPLES`] of a window, zero-padded at the front when it is
/// shorter: the wake path's own pad, so a [`Front`] is only ever handed a window of
/// exactly its length.
pub(crate) fn trailing_window(samples_16k: &[f32]) -> Vec<f32> {
    let mut window = vec![0f32; WINDOW_SAMPLES];
    let take = samples_16k.len().min(WINDOW_SAMPLES);
    window[WINDOW_SAMPLES - take..].copy_from_slice(&samples_16k[samples_16k.len() - take..]);
    window
}

/// One window's projected token rows → its embedding: the tokens pooled over, the result
/// put on the unit sphere. What the ear and the grouped front end both report.
fn embed_rows(projected: &[f32], tokens: usize, dim: usize) -> Result<Vec<f32>> {
    Ok(l2(&mean_pool(projected, tokens, dim)?))
}

fn first(outs: Vec<Vec<f32>>, what: &str) -> Result<Vec<f32>> {
    outs.into_iter()
        .next()
        .ok_or_else(|| anyhow!("the {what} returned no output"))
}

/// One batched pass's projected rows split into one block per window: `batch` blocks of
/// `projected.len() / batch` values from a pass that answered `batch` windows.
fn split_windows(projected: Vec<f32>, batch: usize) -> Result<Vec<Vec<f32>>> {
    let per_window = projected.len() / batch;
    if per_window == 0 || !projected.len().is_multiple_of(batch) {
        return Err(anyhow!(
            "the projector returned {} values for {batch} windows",
            projected.len()
        ));
    }
    Ok((0..batch)
        .map(|slot| projected[slot * per_window..(slot + 1) * per_window].to_vec())
        .collect())
}

pub(crate) fn argmax(v: &[f32]) -> u32 {
    let mut best = (f32::NEG_INFINITY, 0u32);
    for (i, x) in v.iter().enumerate() {
        if *x > best.0 {
            best = (*x, i as u32);
        }
    }
    best.1
}

/// The mean of `rows` rows of width `dim`: the pooling [`Listener::embed`] and the group
/// front end do before [`l2`].
pub fn mean_pool(values: &[f32], rows: usize, dim: usize) -> Result<Vec<f32>> {
    if rows == 0 || dim == 0 {
        return Err(anyhow!("mean pool over {rows} rows of width {dim}"));
    }
    if values.len() != rows * dim {
        return Err(anyhow!(
            "mean pool over {rows} rows of width {dim} needs {} values, the buffer has {}",
            rows * dim,
            values.len()
        ));
    }
    let mut out = vec![0f32; dim];
    for r in 0..rows {
        for i in 0..dim {
            out[i] += values[r * dim + i];
        }
    }
    let inv = 1.0 / rows as f32;
    for v in &mut out {
        *v *= inv;
    }
    Ok(out)
}

/// `v` at unit length: what the wake decision compares two embeddings by, once the pooled
/// row is this.
pub fn l2(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// What one recognition reading answered: the transcript every caller reads and the
/// reading's own cost.
///
/// A reading is one piece or several — [`recognise`] cuts its input and reads each piece
/// on its own prompt — and a piece the guards refuse contributes no words at all, so the
/// reading is shorter than the audio rather than filled with words nobody said. The pieces'
/// fields are joined in piece order.
pub struct Walk {
    /// The input's own frame count.
    pub frames: usize,
    /// Prompt rows the body read.
    pub prompt_rows: usize,
    /// The widest decode capacity grown to.
    pub capacity: usize,
    /// The generated ids, in piece order.
    pub generated: Vec<usize>,
    /// The greedy choice that stopped the last piece: EOS, or the token its budget ended
    /// on.
    pub next: usize,
    /// Whether any piece ran out of its own budget.
    pub truncated: bool,
    /// The pieces' readings with a space between.
    pub text: String,
    /// One piece's answer as floats — its ids, then `next`, its EOS flag and the cache rows
    /// written — concatenated over the pieces: what the answer digest is taken over
    /// ([`crate::report::digest`]).
    pub values: Vec<f32>,
    /// Milliseconds spent building graphs.
    pub build_ms: f64,
    /// One entry per generated token: the row's own price.
    pub decode_times: Vec<f64>,
}

impl Walk {
    /// One reading from the pieces' own, in piece order: the lists concatenated, the counts
    /// and the build time summed, `capacity` the widest any piece's decode grew to, `next`
    /// the last piece's, and `text` the readings the caller kept, a space between.
    pub(crate) fn pieces(frames: usize, pieces: Vec<(Walk, Option<String>)>) -> Walk {
        let mut read = Walk {
            frames,
            prompt_rows: 0,
            capacity: 0,
            generated: Vec::new(),
            next: ASR_EOS,
            truncated: false,
            text: String::new(),
            values: Vec::new(),
            build_ms: 0.0,
            decode_times: Vec::new(),
        };
        let mut readings: Vec<String> = Vec::new();
        for (walk, answer) in pieces {
            read.build_ms += walk.build_ms;
            read.prompt_rows += walk.prompt_rows;
            read.capacity = read.capacity.max(walk.capacity);
            read.truncated |= walk.truncated;
            read.generated.extend(walk.generated);
            read.values.extend(walk.values);
            read.decode_times.extend(walk.decode_times);
            read.next = walk.next;
            if let Some(answer) = answer {
                readings.push(answer.trim().to_string());
            }
        }
        read.text = readings.join(" ");
        read
    }
}

/// A decode graph more than one chunk runs on, with the lock a walk holds while it uses
/// it. The graph's device buffers are *this walk's* k/v planes once
/// `resident_cache_planes` has bound them, so two walks must not interleave into them:
/// the second waits for the first rather than rewriting its cache, which would produce a
/// plausible, wrong transcript. A walk holds [`Shared::walk`] from its bind to its last
/// step.
pub(crate) struct Shared {
    graph: Mutex<Compiled>,
    walk: Mutex<()>,
}

impl Shared {
    fn new(graph: Compiled) -> Arc<Self> {
        Arc::new(Self {
            graph: Mutex::new(graph),
            walk: Mutex::new(()),
        })
    }

    fn graph(&self) -> std::sync::MutexGuard<'_, Compiled> {
        self.graph.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn walk(&self) -> std::sync::MutexGuard<'_, ()> {
        self.walk.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// One chunk's compiled recognition path, built for a frame count and run per chunk: the
/// mel front end, the encoder, the projector, the prompt prefill and the one-row decode.
///
/// Three residencies, one per caller. A **ladder chunk** ([`Clip::rung`]) builds the rung of
/// its frame count and holds it, so a pass compiles each rung once and the rung is given back
/// when the group of chunks that share it has run. A **caller with several chunks of one
/// length** ([`Clip::build`]) keeps its front end and its decode graph and rebuilds only the
/// prefill between chunks, because holding all three would keep three compiled graphs whose
/// uploads carry the body. The **one-shot** clip ([`Clip::once`]) drops the front end as soon
/// as it has run, so the front end is never resident beside the prefill.
///
/// A clip is one pass, not a reading: a pass over a long input loops, so the reading
/// entry points ([`recognise`], and the product's [`crate::transcribe::Transcriber`])
/// cut the input and read the pieces through this.
pub(crate) struct Clip<'a> {
    pair: Arc<Pair>,
    tokenizer: &'a Tokenizer,
    /// The front end, built with the clip and held across chunks when a caller
    /// transcribes several of them. A one-shot clip ([`Clip::once`]) drops it after
    /// its single run, so the front end's graphs are never resident beside the
    /// prefill.
    front: Option<Front>,
    /// The rung this clip runs on, when it is a ladder chunk: the rung's front end and
    /// prefills are the ones the clip runs, so it builds neither and leaves `front` empty.
    /// The clip owns the rung, so the rung's graphs go with it. A clip of its own leaves
    /// this `None`.
    shared: Option<Rung>,
    /// The decode graph this clip runs. On a rung it is the pair's, cloned here at
    /// construction; a walk that outgrows the compiled capacity grows a graph of its own
    /// and replaces this handle, leaving the pair's as it was. On a clip of its own
    /// it is built by the first `run` that needs it.
    decode: Option<Arc<Shared>>,
    cfg: Config,
    frames: usize,
    tokens: usize,
    prompt_rows: usize,
    hkv: usize,
    capacity: usize,
    /// `false` for a one-shot clip: the front end goes as soon as it has run.
    reuse_front: bool,
    /// The most tokens one walk may generate: the crate's budget by default, and
    /// the caller's own when a chunk's length bounds what it can say.
    max_new: usize,
    /// The build the next `run` reports: the construction price, then any graph a
    /// previous run rebuilt. A caller that sums the walks' figures prices every
    /// compile the clip made, exactly once.
    owed_ms: f64,
}

/// The prompt's prefill graph: the whole body over `prompt_rows` rows, logits only on the
/// last, with the fused qkv and gate/up projections the decode graph also uses — the same
/// parameter names, so the device's weight pool holds one copy of the body for both
/// layouts.
fn prefill_graph(pair: &Arc<Pair>, prompt_rows: usize) -> Result<Compiled> {
    graphs::body_new_rows_head_chain(pair.body.clone(), prompt_rows, 0, true)
        .and_then(|m| graphs::Compiled::new(rlx_runtime::Device::Gpu, m))
        .context("prompt build")
}

/// The decode graph for one compiled capacity: the whole body, one row per step,
/// on the per-layer plane cache.
fn decode_graph(pair: &Arc<Pair>, capacity: usize) -> Result<Compiled> {
    graphs::body_capacity_chain_planes(pair.body.clone(), capacity)
        .and_then(|m| graphs::Compiled::new(rlx_runtime::Device::Gpu, m))
        .context("decode build")
}

impl<'a> Clip<'a> {
    /// One chunk's compiled path, for a caller that will run several chunks of this
    /// length: the front end is compiled with the clip and kept, the decode graph by
    /// its first `run` and kept, and every `run` compiles its own prefill and drops
    /// it once it has run rather than hold a third graph whose upload carries the
    /// body.
    pub(crate) fn build(pair: &Arc<Pair>, tokenizer: &'a Tokenizer, frames: usize) -> Result<Self> {
        Self::new(pair, tokenizer, frames, true)
    }

    /// The clip an input that fits one piece is read on ([`recognise`]): [`Clip::build`]'s
    /// graphs, with the front end dropped as soon as it has run, so the front end is not
    /// resident beside the prefill. A second `run` on this
    /// clip rebuilds the front end, so a caller with several chunks uses
    /// [`Clip::build`] instead.
    pub(crate) fn once(pair: &Arc<Pair>, tokenizer: &'a Tokenizer, frames: usize) -> Result<Self> {
        Self::new(pair, tokenizer, frames, false)
    }

    /// A chunk of a **rung** of the transcription ladder: the rung is built by this call and
    /// the clip owns it, so the front end, the prefills and their arenas are given back when
    /// the clip is dropped after the group's last chunk. The decode graph and the mel graph
    /// are not the rung's — the pair and the process keep those — so they outlive the drop.
    pub(crate) fn rung(pair: &Arc<Pair>, tokenizer: &'a Tokenizer, frames: usize) -> Result<Self> {
        let t = report::Stopwatch::start();
        let rung = Rung::build(pair, frames)?;
        let owed_ms = t.ms();
        let decode = Arc::clone(&rung.decode);
        let tokens = rung.tokens;
        let cfg = pair.body.config().clone();
        let hkv = cfg.kv_heads * cfg.head_dim;
        Ok(Self {
            pair: Arc::clone(pair),
            tokenizer,
            front: None,
            decode: Some(decode),
            shared: Some(rung),
            cfg,
            frames,
            tokens,
            prompt_rows: ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + tokens + ASR_TAIL_IDS.len(),
            hkv,
            capacity: RUNG_CAPACITY,
            reuse_front: true,
            max_new: ASR_MAX_NEW,
            owed_ms,
        })
    }

    fn new(
        pair: &Arc<Pair>,
        tokenizer: &'a Tokenizer,
        frames: usize,
        reuse_front: bool,
    ) -> Result<Self> {
        let tokens = graphs::asr_tokens(frames);
        let cfg = pair.body.config().clone();
        let hkv = cfg.kv_heads * cfg.head_dim;
        let prompt_rows = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + tokens + ASR_TAIL_IDS.len();
        // The compiled capacity starts one bucket above the prompt and grows only as
        // the walk fills it; a capacity fixed at the token budget would make every
        // step's attention read thousands of masked rows.
        let capacity = prompt_rows + graphs::CACHE_BUCKET;
        let front = Front::build(pair, frames, 1).context("front build")?;
        let build_ms = front.build_ms();
        Ok(Self {
            pair: Arc::clone(pair),
            tokenizer,
            front: Some(front),
            decode: None,
            shared: None,
            cfg,
            frames,
            tokens,
            prompt_rows,
            hkv,
            capacity,
            reuse_front,
            max_new: ASR_MAX_NEW,
            owed_ms: build_ms,
        })
    }

    /// One chunk of 16 kHz mono audio: exactly `frames * graphs::MEL_HOP` samples (the
    /// caller truncates or pads before calling). `declared` is the three language rows
    /// written into the prompt instead of being generated, or `None` to leave the
    /// recogniser its own answer — the prompt is three rows longer with them, and the two
    /// lengths are two prefills.
    pub(crate) fn run(
        &mut self,
        mut audio: Vec<f32>,
        declared: Option<[usize; ASR_LANGUAGE_IDS.len()]>,
    ) -> Result<Walk> {
        let frames = self.frames;
        audio.truncate(frames * graphs::MEL_HOP);
        let cfg = self.cfg.clone();
        let hkv = self.hkv;
        let vocab = match cfg.head {
            Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
        };
        let prompt_rows = self.prompt_rows + declared.map_or(0, |ids| ids.len());
        let mut capacity = self.capacity;
        let mut build_ms = std::mem::take(&mut self.owed_ms);

        let projected = match &self.shared {
            Some(rung) => rung.front.run(audio)?,
            None => {
                if self.front.is_none() {
                    let t = report::Stopwatch::start();
                    self.front = Some(Front::build(&self.pair, frames, 1).context("front build")?);
                    build_ms += t.ms();
                }
                let out = self
                    .front
                    .as_ref()
                    .expect("the front end was built just above")
                    .run(audio)?;
                if !self.reuse_front {
                    self.front = None;
                }
                out
            }
        };

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
            text_ids.push((
                ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + self.tokens + i,
                *id,
            ));
        }
        if let Some(declared) = &declared {
            let at = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + self.tokens + ASR_TAIL_IDS.len();
            for (i, id) in declared.iter().enumerate() {
                text_ids.push((at + i, *id));
            }
        }
        let picks: Vec<(usize, usize)> = text_ids.iter().map(|(_, id)| (0, *id)).collect();
        let rows = self
            .pair
            .body
            .table_rows_f32("token_embd.weight", &picks, cfg.d)
            .with_context(|| "reading prompt embeddings")?;
        for ((at, _), row) in text_ids.iter().zip(rows) {
            prompt[at * cfg.d..(at + 1) * cfg.d].copy_from_slice(&row);
        }
        for a in 0..self.tokens {
            let at = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + a;
            prompt[at * cfg.d..(at + 1) * cfg.d]
                .copy_from_slice(&projected[a * cfg.d..(a + 1) * cfg.d]);
        }

        // Prompt prefill on a growing cache: all `prompt_rows` rows, logits on the
        // last. A rung keeps its prefill resident; a clip of its own drops it once it
        // has run, so the prefill is never resident beside the decode graph it seeds.
        let (cos, sin) = host::rope_tables_at(0, prompt_rows, cfg.head_dim, cfg.rope_base as f64);
        let inputs = vec![
            ("x".into(), prompt),
            ("cos".into(), cos.iter().map(|v| *v as f32).collect()),
            ("sin".into(), sin.iter().map(|v| *v as f32).collect()),
        ];
        let mut p = match &self.shared {
            Some(rung) => {
                let t = report::Stopwatch::start();
                let prefill = rung.prefill(prompt_rows)?;
                build_ms += t.ms();
                prefill
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .run(&inputs)
            }
            None => {
                let mut prefill = {
                    let t = report::Stopwatch::start();
                    let prefill = prefill_graph(&self.pair, prompt_rows)?;
                    build_ms += t.ms();
                    prefill
                };
                let p = prefill.run(&inputs);
                drop(prefill);
                p
            }
        };
        // The decode is one whole-body graph on a per-layer plane cache: each
        // layer's `[capacity + 1, hkv]` rows are already in the attention's row
        // layout, and the step's own row is appended in-graph.
        // `mask[0..written]` is the history the decode may attend to;
        // `mask[capacity]` is the one query slot, not a cache row.
        let graphs::PlaneCache {
            k: mut k_cache,
            v: mut v_cache,
            mut mask,
        } = graphs::seed_plane_cache(cfg.layers, capacity, prompt_rows, hkv, &p[1], &p[2]);
        let mut next = argmax(&p[3][..vocab]) as usize;
        p.clear();
        let mut decode = match &self.decode {
            Some(decode) => Arc::clone(decode),
            None => {
                let t = report::Stopwatch::start();
                let decode = decode_graph(&self.pair, capacity)?;
                build_ms += t.ms();
                Shared::new(decode)
            }
        };
        // This walk owns the graph's planes until its last step. The lock is taken through
        // its own handle so a grown capacity can replace `decode` below: a graph this call
        // builds is nobody else's, so it needs no lock of its own.
        let held = Arc::clone(&decode);
        let _walk = held.walk();
        if !decode
            .graph()
            .runner
            .resident_cache_planes(&k_cache, &v_cache)
        {
            return Err(anyhow!("resident cache refused"));
        }
        for rows in k_cache.iter_mut().chain(v_cache.iter_mut()) {
            rows.clear();
        }
        let mut generated: Vec<usize> = Vec::new();
        let mut written = prompt_rows;
        let mut decode_times: Vec<f64> = Vec::new();
        let max_new = self.max_new;
        while !is_asr_eos(next) && generated.len() < max_new && written < capacity {
            let t_row = report::Stopwatch::start();
            let mut row = self
                .pair
                .body
                .table_rows_f32("token_embd.weight", &[(0, next)], cfg.d)
                .with_context(|| format!("reading token {next}"))?;
            let hidden = row.remove(0);
            let row_read_ms = t_row.ms();
            let (cos, sin) = host::rope_tables_at(written, 1, cfg.head_dim, cfg.rope_base as f64);
            let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
            let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
            let t = report::Stopwatch::start();
            let inputs = vec![
                ("x".into(), hidden),
                ("cos".into(), cos.clone()),
                ("sin".into(), sin.clone()),
                ("mask".into(), mask.clone()),
            ];
            // Output order: `hidden`, each layer's K row, its V row, then logits.
            // Only the logits are read back: the next step's input is the sampled
            // token's embedding, not this row, so asking for it would be a host
            // return nothing consumes. The cache rows are fed device-side.
            let o = decode
                .graph()
                .runner
                .run_resident_planes(
                    &inputs,
                    written,
                    hkv,
                    &[graphs::plane_step_logits(cfg.layers)],
                )
                .ok_or_else(|| anyhow!("resident feeder refused at {written}"))?;
            decode_times.push(t.ms() + row_read_ms);
            generated.push(next);
            next = argmax(&o[0][..vocab]) as usize;
            mask[written] = 1.0;
            written += 1;
            // A compiled graph's attention reads its whole `[capacity + 1]` plane,
            // so the cache grows by one bucket when the live rows reach it: the
            // written rows move in a plain row copy, the rest stays zero, and the
            // mask is rebuilt with the same key order (history, then the spare row
            // now at `cap`). The arithmetic and the graph are unchanged.
            if written == capacity && !is_asr_eos(next) && generated.len() < max_new {
                let cap = capacity + graphs::CACHE_BUCKET;
                let (k_new, v_new) =
                    graphs::lift_planes(&decode.graph(), cfg.layers, hkv, written, cap)?;
                k_cache = k_new;
                v_cache = v_new;
                mask = graphs::resident_mask(written, cap);
                let t = report::Stopwatch::start();
                let grown = decode_graph(&self.pair, cap)?;
                build_ms += t.ms();
                // A grown capacity is this clip's own: a rung keeps the pair's one graph,
                // which its other chunks still run. The walk lock is still held, so nothing
                // else is inside the graph this replaces.
                decode = Shared::new(grown);
                if !decode
                    .graph()
                    .runner
                    .resident_cache_planes(&k_cache, &v_cache)
                {
                    return Err(anyhow!("resident cache refused at capacity {cap}"));
                }
                for rows in k_cache.iter_mut().chain(v_cache.iter_mut()) {
                    rows.clear();
                }
                capacity = cap;
            }
        }
        let finished = is_asr_eos(next);
        let truncated = !finished && generated.len() >= max_new;
        let text = match generated.iter().position(|&id| id == ASR_TEXT) {
            Some(at) => self.tokenizer.decode(&generated[at + 1..]),
            None => self.tokenizer.decode(&generated),
        };
        // The digest covers the answer itself, not just its length: the generated
        // ids, the next token that stopped the walk, and the EOS flag.
        let mut values: Vec<f32> = generated.iter().map(|&v| v as f32).collect();
        values.push(next as f32);
        values.push(if is_asr_eos(next) { 1.0 } else { 0.0 });
        values.push(written as f32);
        self.capacity = capacity;
        self.decode = Some(decode);
        Ok(Walk {
            frames,
            prompt_rows,
            capacity,
            generated,
            next,
            truncated,
            text,
            values,
            build_ms,
            decode_times,
        })
    }

    /// Cap one walk at `max_new` generated tokens. A caller transcribing a bounded
    /// chunk sets one from the chunk's own length, so a repetition loop cannot cost
    /// more than the audio it covers — the loop is reported either way, by the
    /// transcript's words; a single-pass reading keeps the crate's own budget.
    pub(crate) fn set_max_new(&mut self, max_new: usize) {
        self.max_new = max_new.max(1);
    }
}

/// The one-shot path: any length of 16 kHz mono audio in, its reading out. The input is
/// cut and read piece by piece on the transcription path's own plan
/// (`crate::transcribe::read_whole`) — one pass over twenty-six seconds is a loop of
/// invented words, not a reading — and an input that fits one piece is read by the
/// single pass exactly as it always was.
pub fn recognise(pair: &Arc<Pair>, tokenizer: &Tokenizer, audio: Vec<f32>) -> Result<Walk> {
    let frames = audio.len() / graphs::MEL_HOP;
    if frames == 0 {
        bail!("the audio gave no speech");
    }
    let mut audio = audio;
    audio.truncate(frames * graphs::MEL_HOP);
    crate::transcribe::read_whole(pair, tokenizer, audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reading of several pieces is the pieces' own: a piece the guards refused keeps its
    /// cost and contributes no words, and the fields are joined in piece order.
    #[test]
    fn a_multi_piece_reading_is_its_pieces_in_order() {
        let piece = |prompt_rows: usize, capacity: usize, next: usize, truncated: bool| Walk {
            frames: 96,
            prompt_rows,
            capacity,
            generated: vec![next],
            next,
            truncated,
            text: String::new(),
            values: vec![next as f32],
            build_ms: 10.0,
            decode_times: vec![1.0],
        };
        let read = Walk::pieces(
            192,
            vec![
                (piece(95, 16, 7, false), Some("first".to_string())),
                // Ran out of its own budget: what it said is not a reading of the piece.
                (piece(90, 32, 9, true), None),
                (piece(95, 8, 11, false), Some("third".to_string())),
            ],
        );
        assert_eq!(read.frames, 192);
        assert_eq!(read.prompt_rows, 280);
        assert_eq!(read.build_ms, 30.0);
        assert_eq!(read.capacity, 32, "the widest piece's decode");
        assert_eq!(read.next, 11, "the last piece's");
        assert!(read.truncated, "one piece ran out");
        assert_eq!(read.text, "first third", "the refused piece says nothing");
        assert_eq!(read.generated, [7, 9, 11]);
        assert_eq!(read.values.len(), 3);
        assert_eq!(read.decode_times.len(), 3);
    }

    /// A grouped pass must compute what the single-window pass computes: the
    /// bench's decisions are identical only if a window's embedding does not
    /// depend on which slot of the group it took.
    ///
    /// The tree carries no models, so no gate can run this: it is opt-in on both
    /// the models being installed and `MAHBOT_GROUP_PARITY=1`, and the second
    /// exists so a plain `cargo test` in a shell that has the models does not
    /// compile seven graphs (~1.5 GiB of device work) by accident. Run it
    /// deliberately with `MAHBOT_GROUP_PARITY=1 cargo test -p mahbot-inference --lib
    /// a_grouped_pass`.
    #[test]
    fn a_grouped_pass_matches_the_single_window_pass() {
        let Ok(models) = crate::loader::models_dir() else {
            eprintln!(
                "skipped: the model directory is not installed, so the grouped-pass equivalence \
                 was not checked"
            );
            return;
        };
        if std::env::var_os("MAHBOT_GROUP_PARITY").is_none() {
            eprintln!(
                "skipped: set MAHBOT_GROUP_PARITY=1 to compile the graphs and check the \
                 grouped-pass equivalence"
            );
            return;
        }
        use crate::loader::Role;
        let Some((body, sound)) = crate::loader::pair_files(Role::Recognition) else {
            eprintln!("skipped: the loader names no ASR pair");
            return;
        };
        let (body, sound) = (models.join(body), models.join(sound));
        if !body.exists() || !sound.exists() {
            eprintln!(
                "skipped: the ASR pair is not under {} — the grouped-pass equivalence was not \
                 checked",
                models.display()
            );
            return;
        }
        let pair = std::sync::Arc::new(Pair::open(body, sound).expect("open the ASR pair"));

        // Three distinguishable windows, each shorter than the wake window so the
        // padding path is exercised too.
        let windows: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                let len = 4_000 + i * 1_500;
                (0..len)
                    .map(|n| {
                        let t = n as f32 / 16_000.0;
                        (2.0 * std::f32::consts::PI * (220.0 + 55.0 * i as f32) * t).sin() * 0.4
                    })
                    .collect()
            })
            .collect();

        let tokens = graphs::asr_tokens(MEL_FRAMES);
        let dim = pair.sound.clip().projection_dim;
        let embedding =
            |projected: &[f32]| embed_rows(projected, tokens, dim).expect("a window's rows");
        let single = Front::build(&pair, MEL_FRAMES, 1).expect("the single-window front");
        let paired = Front::build(&pair, MEL_FRAMES, PAIR).expect("the pair front");
        let mut grouped = GroupFront::build(&pair, MEL_FRAMES, GROUP).expect("a group front");
        let padded: Vec<Vec<f32>> = windows.iter().map(|w| trailing_window(w)).collect();
        let references: Vec<&[f32]> = padded.iter().map(Vec::as_slice).collect();
        let batched = grouped
            .forward_group(&references)
            .expect("the grouped pass");
        assert_eq!(batched.len(), windows.len());

        for (window, grouped) in windows.iter().zip(batched) {
            let one = trailing_window(window);
            let single = single.run(one).expect("the single-window pass");
            let a = embedding(&single);
            let b = embedding(&grouped);
            let cosine: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                cosine > 0.999_9,
                "a grouped window's embedding matches the single-window pass: cosine {cosine}"
            );
        }

        // The pair pass, under the same rule: two windows in one batched pass, each
        // slot's embedding what the single-window pass gives that window.
        let batched = paired
            .forward(&[&padded[0], &padded[1]])
            .expect("the pair pass");
        assert_eq!(batched.len(), 2);
        for slot in 0..PAIR {
            let single = single.run(padded[slot].clone()).expect("the single pass");
            let a = embedding(&single);
            let b = embedding(&batched[slot]);
            let cosine: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                cosine > 0.999_9,
                "the pair pass's slot {slot} matches the single-window pass: cosine {cosine}"
            );
        }
    }

    /// A batched mel clamps each window on its own. The log-mel clamp is `max − 8`
    /// over one utterance's every frame and bin, so a pass that reduced over the whole
    /// batch would move every window's mel; two windows at very different levels make
    /// that failure loud. No model is needed — the mel front end carries no weights —
    /// so this runs wherever a wgpu adapter does.
    #[test]
    fn a_batched_mel_clamps_each_window_on_its_own() {
        if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
            return;
        }
        // 128 bins — the recognition front's own — at the wake window's frame count.
        let (frames, bins) = (MEL_FRAMES, 128);
        let samples = graphs::MEL_HOP * frames;
        let tone = |amplitude: f32, hz: f32| -> Vec<f32> {
            (0..samples)
                .map(|n| {
                    let t = n as f32 / SAMPLE_RATE as f32;
                    amplitude * (2.0 * std::f32::consts::PI * hz * t).sin()
                })
                .collect()
        };
        // 26 dB apart: a clamp shared by the pair would hold one window's log-mel
        // against the other's maximum.
        let quiet = tone(0.05, 220.0);
        let loud = tone(0.9, 660.0);
        let mut batched = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::mel_frontend(frames, bins, PAIR).expect("the batched mel front end"),
        )
        .expect("the batched mel compiles");
        let mut single = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::mel_frontend(frames, bins, 1).expect("the mel front end"),
        )
        .expect("the mel compiles");
        let mut audio = quiet.clone();
        audio.extend_from_slice(&loud);
        let two = batched
            .run(&[("audio".into(), audio)])
            .pop()
            .expect("the batched mel answers");
        for (slot, window) in [quiet, loud].into_iter().enumerate() {
            let alone = single
                .run(&[("audio".into(), window)])
                .pop()
                .expect("the mel answers");
            assert_eq!(alone.len(), two.len() / PAIR, "one block per window");
            let block = &two[slot * alone.len()..(slot + 1) * alone.len()];
            let delta = alone
                .iter()
                .zip(block)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("slot {slot}: batch 2 vs batch 1 max|Δ| {delta:e}");
            assert!(
                delta <= 1e-6,
                "slot {slot}: the batched mel is the window's own mel (max|Δ| {delta:e})"
            );
        }
    }
}
