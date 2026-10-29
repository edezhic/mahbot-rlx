// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The listening path: mel front end, encoder and recognition on this crate's
//! own ASR pair.
//!
//! [`Listener::embed`] is the wake decision's input — one VAD-gated speech
//! window through the sound encoder and the projector, mean-pooled and
//! L2-normalized. The real-audio sweep takes the same front end cut in two
//! (`GroupFront`): the conv front per window, the encoder tail and the projector
//! once for a group of windows.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::graphs::{self, Compiled, Outcome};
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

/// The most audio one recognition pass takes: 30 s, the length the stand's own
/// measured walk is taken at (3 000 mel frames, 375 audio tokens). The encoder
/// admits more, but the prefill's attention plane grows with the square of it, so
/// a longer input is cut into passes of this length rather than compiled whole.
pub const PASS_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

/// The decode budget — the most tokens one segment may generate; the cache is
/// sized in 128-row buckets above the prompt, not by this number.
pub const ASR_MAX_NEW: usize = 2_048;

/// The recognition prompt's fixed ends, in the tokenizer's own ids: 3 head rows
/// and 6 text rows before the encoder's audio rows, 6 tail rows after them. The
/// audio count is the clip's (219 rows for the recorded 17.46 s reference), so
/// the prompt's length is not a constant.
pub const ASR_HEAD_IDS: [usize; 3] = [151644, 8948, 198];
pub const ASR_TEXT_IDS: [usize; 6] = [151645, 198, 151644, 872, 198, 151669];
pub const ASR_TAIL_IDS: [usize; 6] = [151670, 151645, 198, 151644, 77091, 198];
pub const ASR_EOS: usize = 151645;
/// The three tokens a recognition answer opens with — `language English <asr_text>`,
/// written **into the prompt** by the transcription path instead of being generated. The
/// recogniser identifies the language itself, and on noisy speech it answers
/// `language None` and stops, so the whole chunk comes back empty — one 26.8 s voice
/// message did, and every one of its pieces. With the tag supplied the model transcribes
/// the audio's own language regardless of it (the pinned Russian clip returns its pinned
/// words under an English tag) and the tag never reaches a transcript: everything up to
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

/// The decode's compiled cache grows in these row buckets: the attention operand
/// is the whole `[capacity + 1]` plane, so a capacity fixed at the token budget
/// makes every step read the masked rows too. One bucket above the prompt, then
/// one more each time the live cache reaches the compiled one, keeps the operand
/// within a bucket of the live span. The first bucket holds a whole short
/// utterance (a walk of 65 tokens here), so the common case pays one compile;
/// the bucket exists to bound the operand, not to avoid a second graph.
pub const DECODE_CACHE_BUCKET: usize = 128;

/// The front end's three graphs for one frame count: mel, encoder and projector.
/// Built and run in this order everywhere the cells measure — the ear's window,
/// the ear's shorter cells, a recognition segment — so the input names and the
/// frame arithmetic are stated once. One window per call, which is the graph the
/// stand's numbers belong to.
///
/// The mel graph is not this clip's own: it has no weights and its shape is the frame
/// count, so every clip that reads the same count runs the same compiled graph
/// ([`mel_graph`]). A transcription cuts its chunks onto a closed ladder of frame counts
/// ([`crate::transcribe::FRAME_RUNGS`]), so the mel graphs a process compiles are that
/// ladder's — one each, however many chunks the pass reads.
pub(crate) struct Front {
    mel: Arc<Mutex<Compiled>>,
    encoder: Arc<Mutex<Compiled>>,
    projector: Arc<Mutex<Compiled>>,
    build_ms: f64,
}

/// The mel graphs a process keeps: the length the ear's window asks for and the rung a pass
/// of one piece settles on, with room for a second rung without evicting the first. An
/// evicted entry is a recompile, never a wrong answer; before the bound a process handed
/// many lengths kept one graph each for the life of the process.
const MEL_GRAPH_ENTRIES: usize = 4;

/// The process's mel graphs and their least-recently-used order.
type MelGraphs = (
    HashMap<(usize, usize), Arc<Mutex<Compiled>>>,
    std::collections::VecDeque<(usize, usize)>,
);

fn mel_graphs() -> &'static Mutex<MelGraphs> {
    static MEL: OnceLock<Mutex<MelGraphs>> = OnceLock::new();
    MEL.get_or_init(|| Mutex::new((HashMap::new(), std::collections::VecDeque::new())))
}

/// The mel graph for one frame count and bin count, compiled once for the process: it
/// has no weights, so two clips of the same length share it exactly. Held behind a lock
/// because a run needs `&mut`.
///
/// Keyed by the caller's own audio length, so the cache could otherwise collect one graph
/// per length a long-lived process is handed. It is bounded instead ([`MEL_GRAPH_ENTRIES`])
/// and cleared with the ladder ([`release_rungs`]).
fn mel_graph(frames: usize, bins: usize) -> Result<Arc<Mutex<Compiled>>> {
    let mut guard = mel_graphs().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(graph) = guard.0.get(&(frames, bins)) {
        let graph = Arc::clone(graph);
        guard.1.retain(|key| *key != (frames, bins));
        guard.1.push_back((frames, bins));
        return Ok(graph);
    }
    let graph = Arc::new(Mutex::new(Compiled::new(
        rlx_runtime::Device::Gpu,
        graphs::mel_frontend(frames, bins)?,
    )?));
    while guard.0.len() >= MEL_GRAPH_ENTRIES {
        let Some(oldest) = guard.1.pop_front() else {
            break;
        };
        guard.0.remove(&oldest);
    }
    guard.0.insert((frames, bins), Arc::clone(&graph));
    guard.1.push_back((frames, bins));
    Ok(graph)
}

impl Front {
    pub(crate) fn build(pair: &Arc<Pair>, frames: usize) -> Result<Self> {
        let t = crate::report::Stopwatch::start();
        let tokens = graphs::asr_tokens(frames);
        let mel = mel_graph(frames, pair.sound.clip().mel_bins)?;
        let encoder = Arc::new(Mutex::new(Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::asr_encoder(pair.sound.clone(), frames)?,
        )?));
        let projector = Arc::new(Mutex::new(Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::projector_chain(pair.clone(), tokens)?,
        )?));
        Ok(Self {
            mel,
            encoder,
            projector,
            build_ms: t.ms(),
        })
    }

    pub(crate) fn build_ms(&self) -> f64 {
        self.build_ms
    }

    /// The three measured runs, for the stand's per-phase figures.
    pub(crate) fn run(&self, audio: Vec<f32>, reps: usize) -> (Outcome, Outcome, Outcome) {
        let mel = self
            .mel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .run(&[("audio".into(), audio)], reps);
        let encoder = self
            .encoder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .run(&[("mel".into(), mel.outs[0].clone())], reps);
        let projector = self
            .projector
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .run(&[("features".into(), encoder.outs[0].clone())], reps);
        (mel, encoder, projector)
    }
}

/// One frame count of the transcription ladder, compiled once for the process: the front
/// end and the prompt prefill, over the pair's one shared decode graph. A transcription cuts
/// its chunks onto the closed ladder of [`crate::transcribe::FRAME_RUNGS`], so the ladder's
/// rungs are the only frame counts it ever compiles — and each is compiled once, not once
/// per chunk.
pub(crate) struct Rung {
    /// The pair these graphs were built over. A cached rung is served only to a caller
    /// holding this same pair: two models whose graphs share tensor names would
    /// otherwise run one model's weights on the other's audio.
    pair: Arc<Pair>,
    front: Front,
    /// The prompt prefill, one per prompt length: the prompt carries three declared
    /// language rows or not at all, and the two lengths are two graphs. A variant the
    /// process never asks for is never compiled.
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
pub(crate) const RUNG_CAPACITY: usize = 192;

impl Rung {
    fn build(pair: &Arc<Pair>, frames: usize) -> Result<Self> {
        Ok(Self {
            pair: Arc::clone(pair),
            front: Front::build(pair, frames)?,
            prefills: Mutex::new(HashMap::new()),
            decode: shared_decode(pair)?,
            tokens: graphs::asr_tokens(frames),
        })
    }

    /// This rung's prefill for `prompt_rows`, compiled the first time the ladder is asked
    /// for it and kept for the process. Built over this rung's own pair: a rung is served
    /// only to a caller holding it.
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

/// The rungs kept resident: the ladder's own size, so a transcription never evicts a rung
/// it is about to ask for again. A process that walks a longer ladder evicts its least
/// recently used rung, which is a recompile, never a wrong answer.
const RUNG_ENTRIES: usize = crate::transcribe::FRAME_RUNGS.len();

type RungCache = (HashMap<usize, Arc<Rung>>, std::collections::VecDeque<usize>);

fn rung_cache() -> &'static Mutex<RungCache> {
    static RUNGS: OnceLock<Mutex<RungCache>> = OnceLock::new();
    RUNGS.get_or_init(|| Mutex::new((HashMap::new(), std::collections::VecDeque::new())))
}

/// The one decode graph of a pair, built on first ask: every rung of the ladder and every
/// chunk of a rung runs the same `[RUNG_CAPACITY]`-deep body. Held beside the ladder
/// rather than inside it because the ladder's entries come and go with the frame counts,
/// and this graph does not.
fn decode_slot() -> &'static Mutex<Option<(Arc<Pair>, Arc<Shared>)>> {
    static DECODE: OnceLock<Mutex<Option<(Arc<Pair>, Arc<Shared>)>>> = OnceLock::new();
    DECODE.get_or_init(|| Mutex::new(None))
}

fn shared_decode(pair: &Arc<Pair>) -> Result<Arc<Shared>> {
    let mut guard = decode_slot().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((held, decode)) = guard.as_ref() {
        // Pointer identity, not the file path: a second `Pair` opened over the same files
        // is a second set of weights, and its graphs are not this one's.
        if Arc::ptr_eq(held, pair) {
            return Ok(Arc::clone(decode));
        }
    }
    let decode = Shared::new(decode_graph(pair, RUNG_CAPACITY)?);
    *guard = Some((Arc::clone(pair), Arc::clone(&decode)));
    Ok(decode)
}

/// Empty the ladder, the decode graph that outlives its entries, and the process's mel
/// graphs: what [`crate::release_process_memory`] asks for. A clip still holding a rung's
/// graphs keeps them alive; nothing else does.
pub(crate) fn release_rungs() {
    let mut guard = rung_cache().lock().unwrap_or_else(|e| e.into_inner());
    guard.0.clear();
    guard.1.clear();
    drop(guard);
    *decode_slot().lock().unwrap_or_else(|e| e.into_inner()) = None;
    let mut mel = mel_graphs().lock().unwrap_or_else(|e| e.into_inner());
    mel.0.clear();
    mel.1.clear();
}

/// The rung for `frames` of `pair`, compiled by this call when no earlier one placed it.
/// The second value is the build this call paid, `None` for a rung already resident.
pub(crate) fn rung(pair: &Arc<Pair>, frames: usize) -> Result<(Arc<Rung>, Option<f64>)> {
    let mut guard = rung_cache().lock().unwrap_or_else(|e| e.into_inner());
    let resident = guard
        .0
        .get(&frames)
        .filter(|rung| Arc::ptr_eq(&rung.pair, pair))
        .map(Arc::clone);
    if let Some(rung) = resident {
        if let Some(at) = guard.1.iter().position(|f| *f == frames) {
            guard.1.remove(at);
        }
        guard.1.push_back(frames);
        return Ok((rung, None));
    }
    let t = crate::report::Stopwatch::start();
    let built = Arc::new(Rung::build(pair, frames)?);
    let build_ms = t.ms();
    // The pair is part of the key, so an entry another pair placed is evicted, not served.
    if guard.0.contains_key(&frames) {
        guard.1.retain(|f| *f != frames);
    }
    while guard.0.len() >= RUNG_ENTRIES {
        let Some(oldest) = guard.1.pop_front() else {
            break;
        };
        guard.0.remove(&oldest);
    }
    guard.0.insert(frames, Arc::clone(&built));
    guard.1.push_back(frames);
    Ok((built, Some(build_ms)))
}

/// The same front end cut where a group pass needs it: the conv front runs one
/// window per call (its route costs a window's work per window whatever the
/// batch axis says), and the tail — `conv_out`, the position table, the 24 CLIP
/// blocks, `post_ln` — runs over the whole group in one pass, with the projector
/// after it. That is where the weight reads and the multiplies amortise.
pub(crate) struct GroupFront {
    pair: Arc<Pair>,
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
            graphs::mel_frontend(frames, bins)?,
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
            pair: Arc::clone(pair),
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

    /// One listener's memory: the pair's two model files (a graph binds its weights
    /// from those files, so their sizes are the weight figure the planned arenas do not
    /// carry), plus every compiled graph's planned arena and uploaded parameters.
    ///
    /// The backend's scratch tail is excluded — it is reserved, not resident, for
    /// graphs whose attention does not fill it — and the run's system-memory guard
    /// backstops the estimate.
    #[must_use]
    pub fn footprint_bytes(&self) -> u64 {
        let file = |path: &std::path::Path| std::fs::metadata(path).map_or(0, |m| m.len());
        let weights = file(&self.pair.body.path) + file(&self.pair.sound.path);
        let graph = |g: &Compiled| g.device_bytes() as u64;
        weights
            + [&self.mel, &self.convs, &self.tail, &self.projector]
                .into_iter()
                .map(graph)
                .sum::<u64>()
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
        let per_slot = projected.len() / batch;
        if per_slot == 0 || projected.len() % batch != 0 {
            return Err(anyhow!(
                "the projector returned {} values for {batch} windows",
                projected.len()
            ));
        }
        Ok((0..windows.len())
            .map(|slot| projected[slot * per_slot..(slot + 1) * per_slot].to_vec())
            .collect())
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

/// The listening path for everything that embeds one window at a time: the
/// enrolment, the negative pool and the ungrouped detection streams. The sweep's
/// group front end (`GroupFront`) is built and held on its own, so neither path
/// compiles a graph the other runs.
pub struct Listener {
    tokens: usize,
    dim: usize,
    single: Front,
}

impl Listener {
    /// Compile the mel front end, the encoder and the projector for the wake
    /// window (76 frames), one window per call.
    pub fn new(pair: Arc<Pair>) -> Result<Self> {
        let tokens = graphs::asr_tokens(MEL_FRAMES);
        let dim = pair.sound.clip().projection_dim;
        let single = Front::build(&pair, MEL_FRAMES)?;
        Ok(Self {
            tokens,
            dim,
            single,
        })
    }

    /// One window of mono 16 kHz audio → the encoder's window embedding, through
    /// the one-window front end — enrolment and the negative pool embed a clip at
    /// a time, and padding a lone window into a [`GROUP`]-wide pass costs the
    /// whole group's work.
    pub fn embed(&mut self, samples_16k: &[f32]) -> Result<Vec<f32>> {
        let (_, _, projector) = self.single.run(trailing_window(samples_16k), 0);
        let pooled = mean_pool(&projector.outs[0], self.tokens, self.dim)?;
        Ok(l2(&pooled))
    }
}

/// Whatever turns a group of detection windows into one embedding each: the
/// sweep's [`GroupFront`], or a [`Listener`] for the ungrouped streams.
pub(crate) trait Embedder {
    /// Up to [`GROUP`] windows → one embedding each. [`GroupFront`] spends one
    /// encoder tail pass and one projector pass over the whole group; a
    /// [`Listener`] embeds them one at a time.
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
            .map(|projected| Ok(l2(&mean_pool(&projected, tokens, dim)?)))
            .collect()
    }
}

impl Embedder for Listener {
    fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        windows.iter().map(|window| self.embed(window)).collect()
    }
}

/// The trailing [`WINDOW_SAMPLES`] of a window, zero-padded at the front when
/// it is shorter.
pub(crate) fn trailing_window(samples_16k: &[f32]) -> Vec<f32> {
    let mut window = vec![0f32; WINDOW_SAMPLES];
    let take = samples_16k.len().min(WINDOW_SAMPLES);
    window[WINDOW_SAMPLES - take..].copy_from_slice(&samples_16k[samples_16k.len() - take..]);
    window
}

fn first(outs: Vec<Vec<f32>>, what: &str) -> Result<Vec<f32>> {
    outs.into_iter()
        .next()
        .ok_or_else(|| anyhow!("the {what} returned no output"))
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

pub(crate) fn mean_pool(values: &[f32], rows: usize, dim: usize) -> Result<Vec<f32>> {
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

pub(crate) fn l2(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// What one recognition walk answered and what it cost: the answer every caller
/// transcribes, and the materials the stand folds into its per-phase figures.
pub struct Walk {
    pub frames: usize,
    pub samples: usize,
    pub tokens: usize,
    pub prompt_rows: usize,
    pub capacity: usize,
    pub generated: Vec<usize>,
    /// The greedy choice that stopped the walk (EOS, or the token the budget ended on).
    pub next: usize,
    pub written: usize,
    pub finished: bool,
    pub truncated: bool,
    pub text: String,
    pub values: Vec<f32>,
    pub build_ms: f64,
    /// The prompt-embedding table read, priced into the prefill phase.
    pub prompt_read_ms: f64,
    pub mel: Outcome,
    pub encoder: Outcome,
    pub projector: Outcome,
    pub prefill: Outcome,
    pub decode_nodes: usize,
    /// One entry per generated token: the row's own price.
    pub decode_times: Vec<f64>,
    pub kv_bytes: usize,
    pub decode_stream_bytes: u64,
    pub counters: (u64, u64),
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
/// Three residencies, one per caller. A **ladder chunk** ([`Clip::rung`]) borrows the
/// process-wide graphs of its frame count, so a pass compiles each rung once. A **caller
/// with several chunks of one length** ([`Clip::build`]) keeps its front end and its
/// decode graph and rebuilds only the prefill between chunks, because holding all three
/// would keep three compiled graphs whose uploads carry the body. The **one-shot** clip
/// ([`Clip::once`]) drops the front end as soon as it has run, so the front end is never
/// resident beside the prefill — the residency the listen cell's peak records.
pub struct Clip<'a> {
    pair: Arc<Pair>,
    tokenizer: &'a Tokenizer,
    /// The front end, built with the clip and held across chunks when a caller
    /// transcribes several of them. A one-shot clip ([`Clip::once`]) drops it after
    /// its single run, so the front end's graphs are never resident beside the
    /// prefill — the residency the listen cell's peak records.
    front: Option<Front>,
    /// The rung this clip runs on, when it is a ladder chunk: its front end and its
    /// prefills are the process-wide ones ([`rung`]), so this clip builds neither and
    /// leaves `front` empty. `decode` starts as the rung's own graph, which this clip may
    /// replace with a grown one. A clip of its own leaves this `None`.
    shared: Option<Arc<Rung>>,
    /// The decode graph this clip runs. On a rung that is the rung's own, cloned here at
    /// construction; a walk that outgrows the compiled capacity grows a graph of its own
    /// and replaces this handle, leaving the rung's entry as it was. On a clip of its own
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
/// blob names, so the device's weight pool holds one copy of the body for both layouts.
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
    pub fn build(pair: &Arc<Pair>, tokenizer: &'a Tokenizer, frames: usize) -> Result<Self> {
        Self::new(pair, tokenizer, frames, true)
    }

    /// The one-shot path's clip: [`Clip::build`]'s graphs, with the front end
    /// dropped as soon as it has run, so the front end is not resident beside the
    /// prefill — the residency the listen cell's peak records. A second `run` on this
    /// clip rebuilds the front end, so a caller with several chunks uses
    /// [`Clip::build`] instead.
    pub fn once(pair: &Arc<Pair>, tokenizer: &'a Tokenizer, frames: usize) -> Result<Self> {
        Self::new(pair, tokenizer, frames, false)
    }

    /// A chunk of a **rung** of the transcription ladder: every graph is the process-wide
    /// one for that frame count ([`rung`]), so the call compiles nothing a previous chunk
    /// of the rung compiled. The second value is the build this call paid, `None` when the
    /// rung was already resident.
    pub(crate) fn rung(
        pair: &Arc<Pair>,
        tokenizer: &'a Tokenizer,
        frames: usize,
    ) -> Result<(Self, Option<f64>)> {
        let (rung, build_ms) = rung(pair, frames)?;
        let cfg = pair.body.config().clone();
        let hkv = cfg.kv_heads * cfg.head_dim;
        let clip = Self {
            pair: Arc::clone(pair),
            tokenizer,
            front: None,
            decode: Some(Arc::clone(&rung.decode)),
            shared: Some(Arc::clone(&rung)),
            cfg,
            frames,
            tokens: rung.tokens,
            prompt_rows: ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + rung.tokens + ASR_TAIL_IDS.len(),
            hkv,
            capacity: RUNG_CAPACITY,
            reuse_front: true,
            max_new: ASR_MAX_NEW,
            owed_ms: 0.0,
        };
        Ok((clip, build_ms))
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
        let capacity = prompt_rows + DECODE_CACHE_BUCKET;
        let front = Front::build(pair, frames).context("front build")?;
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
    /// lengths are two prefills. `reps` is the warm repetition count of every measured
    /// phase; the gate passes 0.
    pub fn run(
        &mut self,
        mut audio: Vec<f32>,
        declared: Option<[usize; ASR_LANGUAGE_IDS.len()]>,
        reps: usize,
    ) -> Result<Walk> {
        let frames = self.frames;
        audio.truncate(frames * graphs::MEL_HOP);
        let samples = audio.len();
        let cfg = self.cfg.clone();
        let hkv = self.hkv;
        let vocab = match cfg.head {
            Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
        };
        let prompt_rows = self.prompt_rows + declared.map_or(0, |ids| ids.len());
        let mut capacity = self.capacity;
        let mut build_ms = std::mem::take(&mut self.owed_ms);
        let c0 = graphs::device_counts();

        let (mel, encoder, projector) = match &self.shared {
            Some(rung) => rung.front.run(audio, reps),
            None => {
                if self.front.is_none() {
                    let t = report::Stopwatch::start();
                    self.front = Some(Front::build(&self.pair, frames).context("front build")?);
                    build_ms += t.ms();
                }
                let out = self
                    .front
                    .as_ref()
                    .expect("the front end was built just above")
                    .run(audio, reps);
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
        let t_rows = report::Stopwatch::start();
        let rows = self
            .pair
            .body
            .table_rows_f32("token_embd.weight", &picks, cfg.d)
            .with_context(|| "reading prompt embeddings")?;
        let prompt_read_ms = t_rows.ms();
        for ((at, _), row) in text_ids.iter().zip(rows) {
            prompt[at * cfg.d..(at + 1) * cfg.d].copy_from_slice(&row);
        }
        for a in 0..self.tokens {
            let at = ASR_HEAD_IDS.len() + ASR_TEXT_IDS.len() + a;
            prompt[at * cfg.d..(at + 1) * cfg.d]
                .copy_from_slice(&projector.outs[0][a * cfg.d..(a + 1) * cfg.d]);
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
                    .run(&inputs, reps)
            }
            None => {
                let mut prefill = {
                    let t = report::Stopwatch::start();
                    let prefill = prefill_graph(&self.pair, prompt_rows)?;
                    build_ms += t.ms();
                    prefill
                };
                let p = prefill.run(&inputs, reps);
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
        } = graphs::seed_plane_cache(
            cfg.layers,
            capacity,
            prompt_rows,
            hkv,
            &p.outs[1],
            &p.outs[2],
        );
        let mut next = argmax(&p.outs[3][..vocab]) as usize;
        p.outs.clear();
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
        let (nodes, decode_stream_bytes) = {
            let guard = decode.graph();
            (
                guard.ops.iter().map(|(_, n)| n).sum::<usize>(),
                graphs::stream_bytes(&guard.runner.graph),
            )
        };
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
                let cap = capacity + DECODE_CACHE_BUCKET;
                let mut k_new: Vec<Vec<f32>> = (0..cfg.layers)
                    .map(|_| vec![0f32; (cap + 1) * hkv])
                    .collect();
                let mut v_new: Vec<Vec<f32>> = (0..cfg.layers)
                    .map(|_| vec![0f32; (cap + 1) * hkv])
                    .collect();
                let live = written * hkv;
                for l in 0..cfg.layers {
                    let k_rows = decode
                        .graph()
                        .runner
                        .read_plane(&format!("k{l}"))
                        .ok_or_else(|| anyhow!("reading k{l} at row {written}"))?;
                    let v_rows = decode
                        .graph()
                        .runner
                        .read_plane(&format!("v{l}"))
                        .ok_or_else(|| anyhow!("reading v{l} at row {written}"))?;
                    k_new[l][..live].copy_from_slice(&k_rows[..live]);
                    v_new[l][..live].copy_from_slice(&v_rows[..live]);
                }
                k_cache = k_new;
                v_cache = v_new;
                mask = vec![0f32; cap + 1];
                for r in 0..written {
                    mask[r] = 1.0;
                }
                mask[cap] = 1.0;
                let t = report::Stopwatch::start();
                let grown = decode_graph(&self.pair, cap)?;
                build_ms += t.ms();
                // A grown capacity is this clip's own: a rung keeps the one graph every
                // chunk of its ladder shares. The walk lock is still held, so nothing else
                // is inside the graph this replaces.
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
        // The planes as the walk left them: `2 · layers · (capacity + 1) · hkv` f32.
        let kv_bytes = 2 * cfg.layers * (capacity + 1) * hkv * 4;
        let c = graphs::device_counts();
        self.capacity = capacity;
        self.decode = Some(decode);
        Ok(Walk {
            frames,
            samples,
            tokens: self.tokens,
            prompt_rows,
            capacity,
            generated,
            next,
            written,
            finished,
            truncated,
            text,
            values,
            build_ms,
            prompt_read_ms,
            mel,
            encoder,
            projector,
            prefill: p,
            decode_nodes: nodes,
            decode_times,
            kv_bytes,
            decode_stream_bytes,
            counters: (c.0 - c0.0, c.1 - c0.1),
        })
    }

    /// Cap one walk at `max_new` generated tokens. The one-shot path keeps the
    /// crate's budget; a caller transcribing a bounded chunk sets one from the
    /// chunk's own length, so a repetition loop cannot cost more than the audio it
    /// covers — the loop is reported either way, by the transcript's words.
    pub fn set_max_new(&mut self, max_new: usize) {
        self.max_new = max_new.max(1);
    }
}

/// The one-shot path: one [`Clip::once`] for this length, one run — the clip the
/// listen cell and the speak cell's round-trip gate transcribe with.
pub fn recognise(
    pair: &Arc<Pair>,
    tokenizer: &Tokenizer,
    audio: Vec<f32>,
    reps: usize,
) -> Result<Walk> {
    let frames = audio.len() / graphs::MEL_HOP;
    if frames == 0 {
        bail!("the audio gave no speech");
    }
    let mut audio = audio;
    audio.truncate(frames * graphs::MEL_HOP);
    Clip::once(pair, tokenizer, frames)?.run(audio, None, reps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grouped pass must compute what the single-window pass computes: the
    /// bench's decisions are identical only if a window's embedding does not
    /// depend on which slot of the group it took.
    ///
    /// The tree carries no models, so no gate can run this: it is opt-in on both
    /// `MAHBOT_MODELS_DIR` and `MAHBOT_GROUP_PARITY=1`, and the second exists so a
    /// plain `cargo test` in a shell that has the first does not compile seven
    /// graphs (~1.5 GiB of device work) by accident. Run it deliberately with
    /// `MAHBOT_MODELS_DIR=<dir> MAHBOT_GROUP_PARITY=1 cargo test -p
    /// mahbot-inference --lib a_grouped_pass`.
    #[test]
    fn a_grouped_pass_matches_the_single_window_pass() {
        let Ok(models) = crate::loader::models_dir() else {
            eprintln!(
                "skipped: {} is not set, so the grouped-pass equivalence was not checked",
                crate::loader::MODELS_DIR_ENV
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
            eprintln!(
                "skipped: {} names no ASR pair",
                crate::loader::MODELS_DIR_ENV
            );
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

        let single = Front::build(&pair, MEL_FRAMES).expect("the single-window front");
        let mut grouped = GroupFront::build(&pair, MEL_FRAMES, GROUP).expect("a group front");
        let padded: Vec<Vec<f32>> = windows.iter().map(|w| trailing_window(w)).collect();
        let references: Vec<&[f32]> = padded.iter().map(Vec::as_slice).collect();
        let batched = grouped
            .forward_group(&references)
            .expect("the grouped pass");
        assert_eq!(batched.len(), windows.len());

        for (window, grouped) in windows.iter().zip(batched) {
            let one = trailing_window(window);
            let (_, _, projector) = single.run(one, 0);
            let single = &projector.outs[0];
            let a = l2(&mean_pool(
                single,
                graphs::asr_tokens(MEL_FRAMES),
                pair.sound.clip().projection_dim,
            )
            .unwrap());
            let b = l2(&mean_pool(
                &grouped,
                graphs::asr_tokens(MEL_FRAMES),
                pair.sound.clip().projection_dim,
            )
            .unwrap());
            let cosine: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                cosine > 0.999_9,
                "a grouped window's embedding matches the single-window pass: cosine {cosine}"
            );
        }
    }
}
