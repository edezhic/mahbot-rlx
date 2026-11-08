// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthesis: text to speech on this crate's own talker, code predictor and
//! sound half.
//!
//! The graphs whose shapes do not depend on the request are built and compiled
//! once ([`Synthesis::new`] — the pinned speaker and the code predictor's whole
//! sixteen-level frame as one graph). The prompt and body graphs are built on a
//! length they have not seen and kept while a request lands within a bucket of that
//! length. The sound half's graphs belong to the call that builds them: built for a
//! window length, given back once the windows of that length have run, so a call
//! leaves no sound graph resident. [`Synthesis::synthesize`] is the reusable entry
//! point the wake-word bench calls for every corpus clip. Output is mono f32 PCM at
//! [`SAMPLE_RATE`] (24 kHz, the sound half's own rate), `frames · 1920` samples for the
//! frames the talker produced — as many as the text takes to say, since the render runs
//! to end-of-speech rather than to a frame budget.

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

use crate::graphs::{Compiled, DAC_MARGIN, DAC_WINDOW, FRONT_WINDOW, windows};
use crate::loader::{Config, Head, Pair};
use crate::report::Stopwatch;
use crate::tokenizer::Tokenizer;
use crate::{graphs, host};

/// Waveform rate the sound half produces.
pub const SAMPLE_RATE: u32 = 24_000;

/// The longest block a runaway render may repeat, in level-0 codes. A block of at
/// most sixteen codes — 1.28 s of audio — said over and over for [`REPEAT_SPAN`] is
/// the talker stuck on one phrase rather than carrying a text forward. Nothing here
/// stops a render for being long: the guard is the talker's own stuckness, never the
/// text's length. A trip ends the call with an error, so a corpus build stops rather
/// than caching a clip the render never finished.
const REPEAT_BLOCK: usize = 16;
/// The frames a loop is seen in, whatever its period. Two and a half seconds of one
/// block over and over is longer than any pause, sustained note or single word a
/// legitimate render holds — and a render stuck on one code reaches it in the same
/// two and a half seconds as one stuck on a phrase.
const REPEAT_SPAN: usize = 32;
/// The frames a render may run for each row of text before the call stops it as a
/// runaway: ten times the three frames per token every text measured here takes, so a
/// text said far slower than any of them still finishes. This is what a loop wider than
/// [`REPEAT_BLOCK`] — the one [`self_repeating`] cannot see — runs into: minutes, with an
/// error naming the render, rather than the talker's whole context and a killed process.
/// Ten times a text's rows fits inside the talker's context up to about a thousand rows;
/// past that the context is the bound, and a render that fills it is refused the same way.
const BACKSTOP_PER_ROW: usize = 32;
/// The floor under that backstop: forty seconds of audio, more than a text of a few
/// words has ever taken, so the ratio never binds a short request.
const BACKSTOP_MIN_FRAMES: usize = 512;

/// The talker's own special ids and the codec control block. The body's
/// `token_embd` table is `[projected text 151936; codec 3072]`, so a codec id is
/// the row `151936 + id`, not a text row.
pub const TTS_TEXT_ROWS: usize = 151_936;
pub const TTS_EOS_ID: usize = 151_673;
pub const TTS_PAD_ID: usize = 151_671;
pub const TTS_BOS_ID: usize = 151_672;
pub const TTS_ROLE_IDS: [usize; 3] = [151_644, 77_091, 198];
/// The rows the talker's prompt takes, whatever the text: the role header, the five
/// codec rows and the row the first text token rides. The text's other rows follow one
/// per frame, which is what a text's length is bounded by.
const TTS_PROMPT_ROWS: usize = TTS_ROLE_IDS.len() + 6;
pub const TTS_CODEC_PAD: usize = 2_148;
pub const TTS_CODEC_BOS: usize = 2_149;
pub const TTS_CODEC_EOS: usize = 2_150;
pub const TTS_CODEC_NOTTHINK: usize = 2_155;
pub const TTS_CODEC_THINK_BOS: usize = 2_156;
pub const TTS_CODEC_THINK_EOS: usize = 2_157;
pub const TTS_REPEAT_PENALTY: f32 = 1.05;

/// The codec's service block is the last 1024 codec ids. Only EOS is a
/// legitimate answer from it; everything else is not a codebook entry.
pub(crate) fn suppress_service(logits: &mut [f32], vocab: usize) {
    let first = vocab.saturating_sub(1024);
    for id in first..vocab.min(logits.len()) {
        if id != TTS_CODEC_EOS {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}

pub(crate) fn apply_repetition(logits: &mut [f32], seen: &[usize], penalty: f32) {
    for &id in seen {
        if let Some(v) = logits.get_mut(id) {
            *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
        }
    }
}

/// The block a render has got stuck on: `Some(period)` when its last [`REPEAT_SPAN`]
/// level-0 codes are one block of at most [`REPEAT_BLOCK`] codes said over and over.
/// A phrase said once is not a loop; the same stretch of audio again and again is the
/// talker filling the render instead of finishing the text.
fn self_repeating(level0: &[usize]) -> Option<usize> {
    if level0.len() < REPEAT_SPAN {
        return None;
    }
    let tail = &level0[level0.len() - REPEAT_SPAN..];
    (1..=REPEAT_BLOCK).find(|&period| (period..REPEAT_SPAN).all(|i| tail[i] == tail[i - period]))
}

/// The frames a render of `rows` text rows may run: [`BACKSTOP_PER_ROW`] for each, at
/// least [`BACKSTOP_MIN_FRAMES`], and never more than the `limit` the context leaves — a
/// text long enough for ten times its rows to exceed that limit is bounded by the context.
fn frame_bound(rows: usize, limit: usize) -> usize {
    (rows * BACKSTOP_PER_ROW)
        .max(BACKSTOP_MIN_FRAMES)
        .min(limit)
}

/// Whether a text of `rows` rows can be spoken inside a `context`-row talker at all: the
/// prompt and the text's own rows must come to less than the context, because the frame
/// that answers end-of-speech reads one row more.
fn fits_context(rows: usize, context: usize) -> bool {
    TTS_PROMPT_ROWS + rows < context
}

/// The talker's prompt for one request: role header → codec control prefix → first text
/// token + codec BOS, the real trailing text followed by EOS, and the pad
/// embedding the loop falls back to once the text is spent.
pub(crate) struct TalkerPrompt {
    pub x: Vec<f32>,
    pub rows: usize,
    pub trailing: Vec<Vec<f32>>,
    pub pad: Vec<f32>,
}

pub(crate) fn talker_prompt(
    body: &crate::loader::Model,
    d: usize,
    speaker_embed: &[f32],
    text_ids: &[usize],
) -> Result<TalkerPrompt> {
    let first_text_id = *text_ids
        .first()
        .ok_or_else(|| anyhow!("the tokenizer gave no text tokens"))?;
    // The table is `[projected text rows; codec rows]`, so a codec row is
    // `151936 + id`.
    let mut wanted: Vec<usize> = Vec::new();
    wanted.extend_from_slice(&TTS_ROLE_IDS);
    wanted.extend_from_slice(text_ids);
    wanted.push(first_text_id);
    wanted.extend_from_slice(&[TTS_EOS_ID, TTS_PAD_ID, TTS_BOS_ID]);
    for &id in &[
        TTS_CODEC_NOTTHINK,
        TTS_CODEC_THINK_BOS,
        TTS_CODEC_THINK_EOS,
        TTS_CODEC_PAD,
        TTS_CODEC_BOS,
    ] {
        wanted.push(TTS_TEXT_ROWS + id);
    }
    wanted.sort_unstable();
    wanted.dedup();
    let picks: Vec<(usize, usize)> = wanted.iter().map(|&id| (0, id)).collect();
    let table_rows = body.table_rows_f32("token_embd.weight", &picks, d)?;
    let row = |id: usize| -> Result<Vec<f32>> {
        let at = wanted
            .iter()
            .position(|&v| v == id)
            .ok_or_else(|| anyhow!("prompt row {id} was not read from `token_embd.weight`"))?;
        Ok(table_rows[at].clone())
    };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let codec = |id: usize| row(TTS_TEXT_ROWS + id);

    let mut codec_input: Vec<Vec<f32>> = Vec::with_capacity(6);
    for id in [TTS_CODEC_NOTTHINK, TTS_CODEC_THINK_BOS, TTS_CODEC_THINK_EOS] {
        codec_input.push(codec(id)?);
    }
    codec_input.push(speaker_embed.to_vec());
    codec_input.push(codec(TTS_CODEC_PAD)?);
    codec_input.push(codec(TTS_CODEC_BOS)?);
    let pad = row(TTS_PAD_ID)?;
    let bos = row(TTS_BOS_ID)?;
    let mut prompt: Vec<Vec<f32>> = TTS_ROLE_IDS
        .iter()
        .map(|&id| row(id))
        .collect::<Result<_>>()?;
    for i in 0..codec_input.len() - 1 {
        let base = if i + 2 == codec_input.len() {
            &bos
        } else {
            &pad
        };
        prompt.push(add(base, &codec_input[i]));
    }
    prompt.push(add(
        &row(first_text_id)?,
        &codec_input[codec_input.len() - 1],
    ));
    // The reference's trailing text is the real tokens followed by EOS; the pad
    // embedding sits after EOS, not before it.
    let mut trailing: Vec<Vec<f32>> = Vec::with_capacity(text_ids.len());
    for &id in &text_ids[1..] {
        trailing.push(row(id)?);
    }
    trailing.push(row(TTS_EOS_ID)?);
    let rows = prompt.len();
    if rows != TTS_PROMPT_ROWS {
        bail!(
            "the talker's prompt came to {rows} rows where its shape is {TTS_PROMPT_ROWS}: this \
             pair cannot render text"
        );
    }
    let mut x = Vec::with_capacity(rows * d);
    for r in &prompt {
        x.extend_from_slice(r);
    }
    Ok(TalkerPrompt {
        x,
        rows,
        trailing,
        pad,
    })
}

/// The codec tensor's `[C, frames · per]` narrowed to the frames `[p0, p1)`:
/// the per-window slice of a group's input, channel-major. `per` is the input
/// stage's own positions per frame, not the waveform's.
pub(crate) fn codec_slice(
    wave: &[f32],
    frames: usize,
    p0: usize,
    p1: usize,
    per: usize,
) -> Vec<f32> {
    let positions = frames * per;
    let c = wave.len() / positions;
    let mut out = Vec::with_capacity(c * (p1 - p0) * per);
    for ch in 0..c {
        let base = ch * positions;
        out.extend_from_slice(&wave[base + p0 * per..base + p1 * per]);
    }
    out
}

/// Copy one window's core `[a, b)` out of a group's output into the part's
/// stitched tensor, channel-major at `per_out`.
pub(crate) fn stitch_window(
    stitched: &mut [f32],
    chunk: &[f32],
    frames: usize,
    part: (usize, usize, usize, usize),
    channels: usize,
    per_out: usize,
) {
    let (p0, _, a, b) = part;
    let span = (part.1 - p0) * per_out;
    for ch in 0..channels {
        let src = ch * span + (a - p0) * per_out;
        let dst = ch * (frames * per_out) + a * per_out;
        stitched[dst..dst + (b - a) * per_out]
            .copy_from_slice(&chunk[src..src + (b - a) * per_out]);
    }
}

/// A render the engine refused: it did not speak the text it was asked for, for one of
/// the reasons [`Synthesis::synthesize`] names — the text has no tokens, it cannot fit the
/// talker's context, the render filled that context, the talker answered end-of-speech
/// before the text was read, it started repeating itself, or it ran to the runaway
/// backstop.
///
/// The type is the classification: a caller that may reject one candidate reads it, and
/// every other failure out of the same call — a device error, a missing tensor row, a
/// feed the talker refused — ends the run as the fault it is.
#[derive(Debug)]
pub struct Unspoken(String);

impl Unspoken {
    /// The refusal as an error carrying its reason.
    fn raised(reason: String) -> anyhow::Error {
        anyhow::Error::new(Self(reason))
    }
}

impl std::fmt::Display for Unspoken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unspoken {}

/// One compiled synthesis pipeline for one talker pair and one speaker.
pub struct Synthesis {
    pair: Arc<Pair>,
    tokenizer: Tokenizer,
    cfg: Config,
    body_vocab: usize,
    hkv: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    steps: usize,
    d_talker: usize,
    speaker_embed: Vec<f32>,
    /// The code predictor's whole sixteen-level frame in **one** graph: the
    /// head-less body at level 0, every projecting level with its own head, each
    /// draw an `Op::Sample` on the device and the cache carried in the arena. A
    /// frame is one run and one fence, its dispatches submitted in encode chunks;
    /// [`Synthesis::generate`] rewrites the draw seeds and runs it.
    frame: Compiled,
    /// The prompt and body graphs are shape-only, and a TTS request whose length lands
    /// within a bucket of the last one reuses them instead of paying the compile again —
    /// the bench's corpus cycles phrases across seeds, and a served voice answers
    /// requests of similar length. One entry each; a body grown by a much longer text
    /// is given back rather than run a short render over its masked rows
    /// ([`Synthesis::take_body`]), and one grown past that window stays cached until a
    /// later call drops it — a long render's planes live until the next request, not
    /// forever.
    prefill: Option<(usize, Compiled)>,
    body: Option<(usize, Compiled)>,
    /// The rows the talker's context holds: the one bound a render cannot outrun. For a
    /// text short enough that [`BACKSTOP_PER_ROW`] frames per row fits inside it,
    /// [`frame_bound`]'s backstop stops a runaway far sooner; past that the context is
    /// the bound, and reaching it is refused the same way.
    context: usize,
    /// What the last [`Synthesis::synthesize`] paid to build: the prompt and body
    /// graphs when the request's lengths are new, the body graph's growth when a text
    /// outlasts its cache, and the sound half's graphs, which every call builds for
    /// the window lengths it runs. Counted as the call goes, with the front's pair
    /// added once its window loop ends, so a failed call reports the builds it
    /// finished — the front's only if that loop did.
    build_ms: f64,
}

impl Synthesis {
    /// The product's synthesis path: the pinned voice is
    /// [`crate::speaker::SPEAKER_EMBED`], so no recording is read and no
    /// speaker-encoder graph is built. The code predictor's whole sixteen-level frame
    /// and the speaker's numbers are compiled once; the prompt and body graphs are
    /// built on first use and reused while a request's length stays within a bucket of
    /// the one they were built for (`Synthesis::prefill`, `Synthesis::body`); the sound
    /// half's graphs are built and given back inside the call that runs them.
    pub fn new(pair: Arc<Pair>) -> Result<Self> {
        Self::with_embedding(pair, crate::speaker::SPEAKER_EMBED.to_vec())
    }

    /// A caller that already holds a speaker vector — the bench's clone voices, whose
    /// vector comes from a corpus span ([`crate::speaker::embed`]), and
    /// [`Synthesis::new`] with the pinned one. The vector must be as wide as the
    /// talker's prompt row, or the call is refused before any graph is built.
    pub fn with_embedding(pair: Arc<Pair>, speaker_embed: Vec<f32>) -> Result<Self> {
        let cfg = pair.body.config().clone();
        // The prompt adds the vector to `token_embd.weight`'s pad row, which is the
        // talker's own width (`cfg.d`) — the same value `generate` builds the prompt
        // with. A vector of another width cannot be conditioned on at all, so it is
        // refused here, before any graph exists.
        if speaker_embed.len() != cfg.d {
            bail!(
                "pair `{}` + `{}`: the speaker vector is {} wide — expected the \
                 talker's {}-wide prompt row (`token_embd.weight`'s pad row)",
                pair.body.path.display(),
                pair.sound.path.display(),
                speaker_embed.len(),
                cfg.d
            );
        }

        let tokenizer = Tokenizer::from_model(&pair.body)?;
        // A render's length is the text's, bounded by the talker's context: a file that
        // states none cannot say how long a render may run.
        let context = pair.body.context_length()?;
        let body_vocab = match cfg.head {
            Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
        };
        let hkv = cfg.kv_heads * cfg.head_dim;
        let (top_k, top_p, temperature) = pair.body.sampling();
        let steps = graphs::code_predictor_steps(&pair.sound)?;
        let capacity_pred = steps + 1;
        let (d_talker, _) = pair.sound.code_proj_widths()?;

        // The draw seed is a uniform word, not a pipeline constant: the graph is
        // built once with a neutral default and every frame rewrites the seeds
        // (`Runner::set_sample_seeds`) before its one run.
        let sampler = graphs::SampleSpec {
            top_k,
            top_p,
            temperature,
            seed: 0,
        };
        let frame = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::tts_code_predictor_frame(pair.sound.clone(), capacity_pred, sampler)?,
        )?;

        Ok(Self {
            pair,
            tokenizer,
            cfg,
            body_vocab,
            hkv,
            top_k,
            top_p,
            temperature,
            steps,
            d_talker,
            speaker_embed,
            frame,
            prefill: None,
            body: None,
            context,
            build_ms: 0.0,
        })
    }

    /// Synthesize `text` with the given sampling seed. Returns mono 24 kHz PCM
    /// of `frames · graphs::samples_per_frame()` samples — as long as the text is.
    ///
    /// The whole text is spoken: the call returns the audio of the render that
    /// reached end-of-speech, and an error naming what stopped it when one did not. Six
    /// outcomes are refusals, and they are [`Unspoken`] so a caller can tell them from a
    /// fault: the text has no tokens to speak, or is longer than the talker's context
    /// can hold; the render filled that context without answering end-of-speech; the
    /// talker answered end-of-speech before the last row of the text was read; it
    /// started repeating itself; or the render ran past the frames its text could
    /// plausibly need. A shorter audio is never returned as though it were the text.
    pub fn synthesize(&mut self, text: &str, seed: u64) -> Result<Vec<f32>> {
        self.build_ms = 0.0;
        let text_ids = self.tokenizer.encode(text);
        if text_ids.is_empty() {
            return Err(Unspoken::raised(format!(
                "the tokenizer gave no text tokens for `{text}`"
            )));
        }
        let frames = self.generate(&text_ids, seed)?;
        self.render_sound(&frames)
    }

    /// Milliseconds the last call spent compiling and uploading graphs: the prompt
    /// and body graphs on a length it had not built, the body graph's growth when a
    /// text outlasts its cache, and the sound half's graphs, which every call builds.
    #[must_use]
    pub fn build_ms(&self) -> f64 {
        self.build_ms
    }

    /// The talker's prompt, then one body step and a fifteen-level predictor
    /// walk per frame, until the talker answers end-of-speech. Returns each
    /// frame's codes, level 0 first.
    ///
    /// The length is the text's, not a budget: the loop's bounds are the frames the text
    /// could plausibly need ([`frame_bound`]) and, past that, the talker's context — and
    /// the repetition guard, the backstop and a context that fills all leave through an
    /// error rather than a shorter answer.
    fn generate(&mut self, text_ids: &[usize], seed: u64) -> Result<Vec<Vec<usize>>> {
        let d = self.cfg.d;
        // The talker reads the text one row per frame, so a text that leaves no room for
        // its own rows and the frame that answers end-of-speech cannot be spoken at all —
        // refused before the prompt's rows are read out of the file or this request's
        // graphs exist.
        if !fits_context(text_ids.len(), self.context) {
            return Err(Unspoken::raised(format!(
                "the text is {} tokens and the talker's {}-row context leaves {} rows beside \
                 the prompt, one of them for the frame that answers end-of-speech: this text \
                 cannot be spoken whole",
                text_ids.len(),
                self.context,
                self.context.saturating_sub(TTS_PROMPT_ROWS)
            )));
        }
        let prompt = talker_prompt(&self.pair.body, d, &self.speaker_embed, text_ids)?;
        let TalkerPrompt {
            x: prompt_x,
            rows: prompt_rows,
            trailing,
            pad,
            ..
        } = prompt;
        let device = rlx_runtime::Device::Gpu;
        if self
            .prefill
            .as_ref()
            .is_none_or(|(rows, _)| *rows != prompt_rows)
        {
            // Dropped before the replacement is compiled, as the body graph is
            // below: the two would otherwise be resident at once.
            self.prefill = None;
            let t = Stopwatch::start();
            let compiled = Compiled::new(
                device,
                graphs::body_new_rows_head_chain(self.pair.body.clone(), prompt_rows, 0, true)?,
            )?;
            self.build_ms += t.ms();
            self.prefill = Some((prompt_rows, compiled));
        }
        let prefill = &mut self.prefill.as_mut().expect("prompt graph").1;
        let (cos, sin) =
            host::rope_tables_at(0, prompt_rows, self.cfg.head_dim, self.cfg.rope_base as f64);
        let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
        let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
        let p = prefill.run_slices(&[("x", &prompt_x), ("cos", &cos), ("sin", &sin)]);
        if p.len() < 4 {
            return Err(anyhow!(
                "the talker prefill returned {} outputs, expected hidden, k, v and logits",
                p.len()
            ));
        }
        let hidden = p[0]
            .get((prompt_rows - 1) * d..prompt_rows * d)
            .ok_or_else(|| anyhow!("the talker prefill returned no last prompt row"))?
            .to_vec();
        let mut logits = p[3]
            .get(..self.body_vocab)
            .ok_or_else(|| anyhow!("the talker prefill returned {} logits", p[3].len()))?
            .to_vec();

        // The talker on a plane cache sized to this render: one row per layer
        // per frame, the prompt rows copied into each layer's own
        // `[capacity + 1, hkv]` cache before the first step. The capacity starts
        // one bucket above the prompt and grows as the render fills it — a
        // capacity fixed at the longest text a caller might ask for would make
        // every step's attention read thousands of masked rows and park their
        // planes for the whole render.
        //
        // A call starts one bucket above its own prompt and reuses the cached graph
        // when that start lands within a bucket of the last one's capacity: a short or
        // typical text pays its compile once. A text that grew the cache past that
        // window compiles its own graph and gives it back, so a long render never
        // leaves the short one running over masked rows.
        let (mut capacity, mut body) = self.take_body(prompt_rows + graphs::CACHE_BUCKET)?;
        let layers = self.cfg.layers;
        let kv_span = layers * prompt_rows * self.hkv;
        let k_rows = p
            .get(1)
            .filter(|v| v.len() >= kv_span)
            .ok_or_else(|| anyhow!("the talker prefill returned no prompt keys"))?;
        let v_rows = p
            .get(2)
            .filter(|v| v.len() >= kv_span)
            .ok_or_else(|| anyhow!("the talker prefill returned no prompt values"))?;
        // The prompt's rows go straight from the prefill's output into the body's own
        // arena slots: the device holds the cache from here, and the rows this render
        // writes go to it through the row feed, so no host plane is kept.
        let mut mask = body.runner.resident_cache_planes(
            layers,
            capacity,
            prompt_rows,
            self.hkv,
            k_rows,
            v_rows,
        )?;

        // The code predictor: **one** graph for the whole sixteen-level frame —
        // the head-less body at level 0, every projecting level with its own
        // head — with each draw an `Op::Sample` on the device and the cache
        // carried in the arena. A frame is one run and one fence — its dispatches
        // are submitted in encode chunks, not per level — and only its draw seeds
        // cross the boundary.
        let mut code_seen: Vec<usize> = Vec::new();
        let mut draw_index = 0u64;
        suppress_service(&mut logits, self.body_vocab);
        apply_repetition(&mut logits, &code_seen, TTS_REPEAT_PENALTY);
        let mut code0 = host::draw(
            &logits,
            self.top_k,
            self.top_p,
            self.temperature,
            seed + draw_index,
        );
        draw_index += 1;
        let mut hidden = hidden;
        let mut all_codes: Vec<Vec<usize>> = Vec::new();
        let mut written = prompt_rows;
        // The rows the talker's context holds, and the far tighter bound a runaway runs
        // into: a render may say its text slowly, but not ten times over.
        let limit = self.context.saturating_sub(prompt_rows);
        let bound = frame_bound(trailing.len(), limit);
        // The period of the block a runaway render got stuck on, if it did.
        let mut looped: Option<usize> = None;
        for f in 0..bound {
            // The reference stops on codec EOS at level 0, but never before two
            // frames; EOS itself is neither penalised nor fed to the sound half.
            if f >= 2 && code0 == TTS_CODEC_EOS {
                break;
            }
            code_seen.push(code0);
            if let Some(period) = self_repeating(&code_seen) {
                looped = Some(period);
                break;
            }
            if written == capacity {
                let (cap, grown, grown_mask) = self.grow_body(capacity, body)?;
                body = grown;
                capacity = cap;
                // The mask keeps its key order: history first, then the spare row the
                // step's own row is appended to, now at `cap`.
                mask = grown_mask;
            }
            let mut frame_codes = vec![code0];
            let code0_embed = self
                .pair
                .body
                .table_rows_f32("token_embd.weight", &[(0, TTS_TEXT_ROWS + code0)], d)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no `token_embd` row for codec id {code0}"))?;

            // The seeds are uniform words, so a compiled
            // frame serves every frame and every request.
            let seeds = graphs::predictor_draw_seeds(
                seed,
                draw_index,
                self.steps,
                self.top_k,
                self.temperature,
            );
            if !self.frame.runner.set_sample_seeds(&seeds) {
                return Err(anyhow!("the code predictor frame refused its draw seeds"));
            }
            let out = self.frame.runner.run_read(
                &[
                    ("hidden".into(), hidden.clone()),
                    ("code0_embed".into(), code0_embed.clone()),
                ],
                &[0],
            );
            let codes = out
                .first()
                .filter(|v| v.len() >= self.steps)
                .ok_or_else(|| {
                    anyhow!("the code predictor frame returned no codes at frame {f}")
                })?;
            for k in 0..self.steps {
                frame_codes.push(codes[k] as usize);
            }
            draw_index += self.steps as u64;

            // The next frame's input: the sixteen code embeddings, summed. The
            // rows are picked out of the file one level at a time — the frame
            // graph's gather reads its own copy in the arena, so the host keeps
            // none.
            let mut sum = code0_embed;
            let picks: Vec<(usize, usize)> = (1..=self.steps)
                .map(|level| {
                    frame_codes
                        .get(level)
                        .map(|&id| (level - 1, id))
                        .ok_or_else(|| anyhow!("level {level} has no code"))
                })
                .collect::<Result<_>>()?;
            let embeds =
                self.pair
                    .sound
                    .table_rows_f32("a.gen.code.embd.weight", &picks, self.d_talker)?;
            for e in &embeds {
                for (o, v) in sum.iter_mut().zip(e) {
                    *o += *v;
                }
            }
            if f + 1 < bound {
                let mut x = trailing.get(f).cloned().unwrap_or_else(|| pad.clone());
                for (v, s) in x.iter_mut().zip(&sum) {
                    *v += *s;
                }
                let (cos, sin) =
                    host::rope_tables_at(written, 1, self.cfg.head_dim, self.cfg.rope_base as f64);
                let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
                let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
                let body_o = body
                    .runner
                    .run_resident_planes(
                        &[
                            ("x".into(), x),
                            ("cos".into(), cos),
                            ("sin".into(), sin),
                            ("mask".into(), mask.clone()),
                        ],
                        written,
                        self.hkv,
                        &graphs::plane_step_reads(layers),
                    )
                    .ok_or_else(|| anyhow!("the talker refused the resident feed at frame {f}"))?;
                mask[written] = 1.0;
                written += 1;
                hidden = body_o
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("the talker returned no hidden row at frame {f}"))?;
                let mut logits = body_o
                    .get(1)
                    .and_then(|v| v.get(..self.body_vocab))
                    .ok_or_else(|| anyhow!("the talker returned no logits at frame {f}"))?
                    .to_vec();
                suppress_service(&mut logits, self.body_vocab);
                apply_repetition(&mut logits, &code_seen, TTS_REPEAT_PENALTY);
                code0 = host::draw(
                    &logits,
                    self.top_k,
                    self.top_p,
                    self.temperature,
                    seed + draw_index,
                );
                draw_index += 1;
            }
            all_codes.push(frame_codes);
        }
        // The graph goes back to the cache before the loop's verdict is acted on, so
        // a render that stopped short costs the next call no compile. It keeps the
        // capacity the render ended with: a text that grew it past the window the next
        // call reads is dropped then, not now, because the next call may well be another
        // long one.
        let frames = all_codes.len();
        self.body = Some((capacity, body));
        if let Some(period) = looped {
            return Err(Unspoken::raised(format!(
                "the talker is repeating itself — its last {REPEAT_SPAN} level-0 codes are one \
                 {period}-code block over and over — after {frames} frames: the text was not \
                 spoken whole"
            )));
        }
        if frames == bound {
            if bound < limit {
                return Err(Unspoken::raised(format!(
                    "the render ran to {frames} frames — its whole backstop of {BACKSTOP_PER_ROW} \
                     frames for each of the {} rows the text takes, floored at \
                     {BACKSTOP_MIN_FRAMES} — without answering end-of-speech, so it is treated \
                     as a runaway: the text was not spoken whole",
                    trailing.len()
                )));
            }
            return Err(Unspoken::raised(format!(
                "the talker filled the {limit} rows its context leaves behind the prompt without \
                 answering end-of-speech: the text was not spoken whole"
            )));
        }
        // The talker reads one text row per frame, in order, so a render that ends with
        // rows still unread answered end-of-speech without ever being given the rest of
        // the text. The count includes the EOS marker, so a render that came within one
        // row of the text is refused rather than passed — never seen at the three frames
        // per row these renders take.
        if frames < trailing.len() {
            return Err(Unspoken::raised(format!(
                "the talker answered end-of-speech after {frames} of the {} rows the text takes: \
                 the text was not spoken whole",
                trailing.len()
            )));
        }
        Ok(all_codes)
    }

    /// The talker graph this call runs on: the cached one when its capacity is within a
    /// bucket of what this call asks for, and a fresh compile at `start` otherwise. That
    /// window is what a short or typical text lands in, so it reuses the graph; one grown
    /// by a much longer text is given back, because every step of a short render at that
    /// capacity reads rows the render has not written — a quarter more per frame than a
    /// graph of its own size. The cache is emptied either way: the graph belongs to the
    /// call until it hands it back, so a grown one never sits beside the graph it replaces.
    fn take_body(&mut self, start: usize) -> Result<(usize, Compiled)> {
        if let Some((capacity, graph)) = self.body.take() {
            if capacity >= start && capacity <= start + graphs::CACHE_BUCKET {
                return Ok((capacity, graph));
            }
            drop(graph);
        }
        Ok((start, self.compile_body(start)?))
    }

    /// A talker graph at `capacity` rows, its compile counted into [`Synthesis::build_ms`].
    fn compile_body(&mut self, capacity: usize) -> Result<Compiled> {
        let t = Stopwatch::start();
        let compiled = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::body_capacity_chain_planes(self.pair.body.clone(), capacity)?,
        )?;
        self.build_ms += t.ms();
        Ok(compiled)
    }

    /// [`Synthesis::take_body`]'s graph grown by one [`graphs::CACHE_BUCKET`]: the
    /// rows the render has written move from the old graph's planes into the new one's
    /// — one readback, then a write per plane — and the rest of the fresh arena stays
    /// zero, so the rows the render reads are the same rows. Answers the grown graph's
    /// mask. The old graph is dropped before its replacement is compiled, so the peak
    /// holds one compiled body.
    fn grow_body(
        &mut self,
        capacity: usize,
        graph: Compiled,
    ) -> Result<(usize, Compiled, Vec<f32>)> {
        let grown = capacity + graphs::CACHE_BUCKET;
        // Every row of the old cache is live here; read them out before the
        // replacement is compiled, so the old arena is dropped first and the peak holds
        // one compiled body.
        let rows = graph
            .runner
            .read_plane_rows(self.cfg.layers, self.hkv, capacity)?;
        drop(graph);
        let mut compiled = self.compile_body(grown)?;
        compiled.runner.write_plane_rows(self.cfg.layers, &rows)?;
        Ok((grown, compiled, graphs::resident_mask(capacity, grown)))
    }

    /// The sound half: codes → codebook → wave transformer → DAC, windowed to
    /// keep each graph's plan small. Stitches into one waveform of
    /// `frames · graphs::samples_per_frame()` samples.
    fn render_sound(&mut self, all_codes: &[Vec<usize>]) -> Result<Vec<f32>> {
        let frames = all_codes.len();
        if frames == 0 {
            return Err(anyhow!("the talker produced no codec frames"));
        }
        let sound = self.pair.sound.clone();
        let levels = sound
            .plan("a.gen.wav.quant.rest.codebook.weight")?
            .ggml_shape[2];
        let mut flat: Vec<f32> = Vec::with_capacity((levels + 1) * frames);
        for level in 0..=levels {
            for f in 0..frames {
                let id = *all_codes
                    .get(f)
                    .and_then(|row| row.get(level))
                    .ok_or_else(|| anyhow!("frame {f} has no code at level {level}"))?;
                flat.push(id as f32);
            }
        }

        let parts = windows(frames, DAC_WINDOW, DAC_MARGIN);
        let groups = graphs::codec_groups();
        let (front_group, dac_groups) = groups
            .split_first()
            .ok_or_else(|| anyhow!("the sound half has no front group"))?;
        let (_wave_d_in, wave_d_tfm) = sound.plan("a.gen.wav.tfm.in_proj.weight")?.in_out();
        let wave_layers = sound
            .plans
            .iter()
            .filter(|p| p.name.starts_with("a.gen.wav.tfm.blk.") && p.name.ends_with(".ln1.weight"))
            .count();
        let wave_heads = 16usize;
        let wave_hd = sound.plan("a.gen.wav.tfm.blk.0.attn_q.weight")?.ggml_shape[1] / wave_heads;
        let wave_hkv = wave_heads * wave_hd;
        let wave_k = sound.plan("a.gen.wav.pre_conv.weight")?.ggml_shape[0];

        let device = rlx_runtime::Device::Gpu;
        let prefix = graphs::FRONT_CARRY;

        // The wave transformer carries a 72-frame window: 71 K/V frames per
        // layer, the last `pre_conv` frames and the global position. Carry the
        // state window to window.
        let (front_ch, front_per) = graphs::dac_stage_io(front_group.end, 1);
        let mut wave = vec![0f32; front_ch * frames * front_per];
        let mut kv_k = vec![0f32; wave_layers * prefix * wave_hkv];
        let mut kv_v = vec![0f32; wave_layers * prefix * wave_hkv];
        let mut left = vec![0f32; wave_d_tfm * (wave_k - 1)];
        // The front's graphs live only until this loop has run: dropped before the
        // first stage group is built, they cannot sit beside the decoder's arenas and
        // set the peak at their sum.
        let mut front = graphs::FrontGraphs::build(device, &sound, frames, front_group.clone())?;
        for (a, b) in windows(frames, FRONT_WINDOW, 0)
            .into_iter()
            .map(|(_, _, a, b)| (a, b))
        {
            let core = b - a;
            let (graph, run_len) = front.pick(a)?;
            let mut chunk = vec![0f32; (levels + 1) * run_len];
            for level in 0..=levels {
                for f in a..b {
                    chunk[level * run_len + (f - a)] = flat[level * frames + f];
                }
            }
            let (cos, sin) = host::rope_tables_at(a, run_len, wave_hd, 1e4);
            let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
            let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
            let mut inputs: Vec<(&str, &[f32])> = vec![
                ("codes", &chunk),
                ("cos", &cos),
                ("sin", &sin),
                ("left", &left),
            ];
            if a > 0 && prefix > 0 {
                inputs.push(("kpre", &kv_k));
                inputs.push(("vpre", &kv_v));
            }
            let run = graph.run_slices(&inputs);
            if run.len() < 4 {
                return Err(anyhow!(
                    "the wave transformer returned {} outputs, expected waveform + left + k/v",
                    run.len()
                ));
            }
            let chunk = &run[0];
            // The run length is the graph's own; a graph whose declared width
            // disagreed with it would slice past the end here, so refuse by name.
            if chunk.len() != front_ch * run_len * front_per {
                return Err(anyhow!(
                    "the wave transformer ran a {run_len}-frame window and returned {} samples, \
                     expected {} for {front_ch} channels × {run_len} frames × {front_per}",
                    chunk.len(),
                    front_ch * run_len * front_per
                ));
            }
            for ch in 0..front_ch {
                let src = ch * run_len * front_per;
                let dst = ch * frames * front_per + a * front_per;
                wave[dst..dst + core * front_per]
                    .copy_from_slice(&chunk[src..src + core * front_per]);
            }
            // Side outputs are published in this order: `left`, `knew`, `vnew`.
            // Carry the last 71 rows of a full chunk; a part's first window is one
            // whenever another window follows it, so the carry is filled before the
            // window that reads it.
            let knew = &run[2];
            let vnew = &run[3];
            let rows = knew.len() / (wave_layers * wave_hkv);
            if prefix > 0 && rows > prefix {
                for l in 0..wave_layers {
                    let src = (l * rows + (rows - prefix)) * wave_hkv;
                    let dst = l * prefix * wave_hkv;
                    kv_k[dst..dst + prefix * wave_hkv]
                        .copy_from_slice(&knew[src..src + prefix * wave_hkv]);
                    kv_v[dst..dst + prefix * wave_hkv]
                        .copy_from_slice(&vnew[src..src + prefix * wave_hkv]);
                }
            }
            left = run[1].clone();
        }
        self.build_ms += front.built_ms();
        drop(front);

        // The distinct window lengths the part runs, the same for every stage group.
        let mut lengths: Vec<usize> = parts.iter().map(|&(p0, p1, _, _)| p1 - p0).collect();
        lengths.sort_unstable();
        lengths.dedup();

        for group in dac_groups {
            let (ch_in, per_in) = graphs::dac_stage_io(group.start, 1);
            let expected = ch_in * frames * per_in;
            if wave.len() != expected {
                return Err(anyhow!(
                    "the sound stage {}..{} input is {} values, its stage reads {expected}",
                    group.start,
                    group.end,
                    wave.len()
                ));
            }
            let (ch_out, per_out) = if group.end == graphs::DAC_STAGES {
                (1, graphs::samples_per_frame())
            } else {
                graphs::dac_stage_io(group.end, 1)
            };
            // The group's output is a tensor of its own shape, not a waveform
            // until the group that carries `post`. Stitch it channel-major at
            // `per_out`, exactly as the next group's input expects it.
            let mut stitched = vec![0f32; ch_out * frames * per_out];
            // One length's graph at a time: built, run over every window of that
            // length, and gone before the next length's is built.
            for &len in &lengths {
                let t = Stopwatch::start();
                let mut graph = graphs::sound_graph(device, &sound, len, group.clone(), 0)?;
                self.build_ms += t.ms();
                for &(p0, p1, a, b) in &parts {
                    if p1 - p0 != len {
                        continue;
                    }
                    let chunk = codec_slice(&wave, frames, p0, p1, per_in);
                    let run = graph.run_slices(&[("codec_in", &chunk)]);
                    let chunk = run
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("the sound stage returned no output"))?;
                    stitch_window(
                        &mut stitched,
                        &chunk,
                        frames,
                        (p0, p1, a, b),
                        ch_out,
                        per_out,
                    );
                }
            }
            wave = stitched;
            // Every graph of this group went with its length, so no two groups can
            // coexist.
        }

        let expected = frames * graphs::samples_per_frame();
        if wave.len() != expected {
            return Err(anyhow!(
                "the sound half left {} values, a {frames}-frame waveform is {expected}",
                wave.len()
            ));
        }
        Ok(wave)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runaway guard's own boundaries: a render shorter than the span is never a
    /// loop, a full span that is not one block repeated is not a loop either, and a
    /// block repeated over the whole span is — at every period the guard looks for.
    #[test]
    fn a_phrase_said_once_is_not_a_loop() {
        // Shorter than the span, however often the phrase is said.
        let phrase: Vec<usize> = (0..7).collect();
        assert_eq!(self_repeating(&[&phrase[..], &phrase[..]].concat()), None);
        // A full span of distinct codes.
        let distinct: Vec<usize> = (0..REPEAT_SPAN).collect();
        assert_eq!(self_repeating(&distinct), None);
        // A full span that is periodic but for one code.
        let mut nearly: Vec<usize> = (0..REPEAT_SPAN).map(|i| 100 + i % 3).collect();
        nearly[REPEAT_SPAN - 1] += 1;
        assert_eq!(self_repeating(&nearly), None);
        // A full span whose period is one the guard does not look for.
        let wide: Vec<usize> = (0..REPEAT_SPAN)
            .map(|i| 100 + i % (REPEAT_BLOCK + 1))
            .collect();
        assert_eq!(self_repeating(&wide), None);
    }

    #[test]
    fn a_block_said_over_and_over_is_a_loop() {
        for period in 1..=REPEAT_BLOCK {
            let block: Vec<usize> = (0..period).map(|i| 100 + i).collect();
            let over: Vec<usize> = (0..REPEAT_SPAN).map(|i| block[i % period]).collect();
            assert_eq!(
                self_repeating(&over),
                Some(period),
                "a {period}-code block over the whole span is the guard's own case"
            );
        }
        // The span, not the repetition count, is the threshold: one frame short of it
        // is not a loop, however many times the block has said itself.
        assert_eq!(self_repeating(&[7; REPEAT_SPAN - 1]), None);
    }

    /// The refusal marker is the classification, and it does not over-reach: a refusal is
    /// found through the context a caller adds around it, and a fault out of the same call
    /// is not.
    #[test]
    fn a_refusal_is_told_from_a_fault_through_its_context() {
        let refusal = Unspoken::raised("the text was not spoken whole".into())
            .context("synthesis failed for `hello`");
        assert!(refusal.downcast_ref::<Unspoken>().is_some());
        let fault =
            anyhow!("the talker returned no hidden row").context("synthesis failed for `hello`");
        assert!(fault.downcast_ref::<Unspoken>().is_none());
    }

    /// The runaway backstop's own boundaries: a short text is bounded by the floor, a
    /// longer one by the per-row ratio, and neither by more than the context leaves.
    #[test]
    fn the_backstop_leaves_a_text_ten_times_its_rows() {
        assert_eq!(frame_bound(1, 32_000), BACKSTOP_MIN_FRAMES);
        assert_eq!(frame_bound(100, 32_000), 100 * BACKSTOP_PER_ROW);
        assert_eq!(frame_bound(100, 1_000), 1_000);
    }

    /// A text is admitted only when the context leaves room for its own rows and the frame
    /// that answers end-of-speech: the last row the talker can read is the EOS row, and the
    /// frame after it is where the render ends.
    #[test]
    fn a_text_needs_its_rows_and_one_to_answer_end_of_speech() {
        let room = TTS_PROMPT_ROWS + 10;
        assert!(fits_context(9, room));
        assert!(!fits_context(10, room));
    }

    /// What precedes the loop is not part of it: a render carrying a sentence
    /// forward and then sticking on one is caught by the tail it ends on.
    #[test]
    fn the_guard_reads_the_tail_only() {
        let mut codes: Vec<usize> = (0..40).map(|i| i * 3 + 1).collect();
        codes.extend([7; REPEAT_SPAN]);
        assert_eq!(self_repeating(&codes), Some(1));
    }
}
