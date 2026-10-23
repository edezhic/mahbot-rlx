// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthesis: text to speech on this crate's own talker, code predictor and
//! sound half.
//!
//! The graphs whose shapes do not depend on the request are built and compiled
//! once ([`Synthesis::new`] — the speaker encoder and the code predictor's whole
//! sixteen-level frame as one graph); the prompt and the sound half are built per call exactly as the
//! stand builds them. [`Synthesis::synthesize`] is the reusable entry point the
//! wake-word bench calls for every corpus clip. Output is mono f32 PCM at
//! [`SAMPLE_RATE`] (24 kHz, the sound half's own rate), `frames · 1920` samples
//! for the frames the talker produced.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::graphs::Compiled;
use crate::loader::{Config, Head, Pair};
use crate::tokenizer::Tokenizer;
use crate::{graphs, host};

/// Waveform rate the sound half produces.
pub const SAMPLE_RATE: u32 = 24_000;

/// The talker's text capacity: longer requests are truncated.
pub const MAX_TEXT_TOKENS: usize = 300;

/// The generation cap: 330 codec rows, about 26.4 s at 24 kHz.
pub const MAX_CODEC_ROWS: usize = 330;

/// The talker's own special ids and the codec control block. The body's
/// `token_embd` table is `[projected text 151936; codec 3072]`, so a codec id is
/// the row `151936 + id`, not a text row.
pub const TTS_TEXT_ROWS: usize = 151_936;
pub const TTS_EOS_ID: usize = 151_673;
pub const TTS_PAD_ID: usize = 151_671;
pub const TTS_BOS_ID: usize = 151_672;
pub const TTS_ROLE_IDS: [usize; 3] = [151_644, 77_091, 198];
pub const TTS_CODEC_PAD: usize = 2_148;
pub const TTS_CODEC_BOS: usize = 2_149;
pub const TTS_CODEC_EOS: usize = 2_150;
pub const TTS_CODEC_NOTTHINK: usize = 2_155;
pub const TTS_CODEC_THINK_BOS: usize = 2_156;
pub const TTS_CODEC_THINK_EOS: usize = 2_157;
pub const TTS_REPEAT_PENALTY: f32 = 1.05;

/// The codec's windows over a part: the DAC stages are convolutional (their
/// LayerNorm is per position), so a window with a margin gives the same samples
/// as the whole part away from the edges, and the planned arena — which is a
/// graph's — shrinks with the window instead of being the part's.
///
/// The window is 144 frames — two of the front's chunks ([`FRONT_WINDOW`]): every
/// window runs its graph over every frame in it, so a longer window is fewer
/// graph runs over the same frames, each convolution at twice the columns and for
/// twice the frames. The measurement is the record's (three windows a group
/// instead of five, the same waveform digest, peak unmoved); doubling again is the
/// same curve's next step and is not taken, because the arena grows with the
/// window.
///
/// Shared with the stand.
pub(crate) const DAC_WINDOW: usize = 144;
/// The wave transformer's own window: the front is cut at this many frames so
/// its output does not depend on the part's total length.
///
/// Shared with the stand.
pub(crate) const FRONT_WINDOW: usize = 72;
/// Frames of margin on each side, clipped at the part's own edges: the four DAC
/// blocks' kernels and strides reach this far into a window, so a core sample
/// never reads a window edge the part does not have.
///
/// Shared with the stand.
pub(crate) const DAC_MARGIN: usize = 4;

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

/// The talker's prompt for one request, shared by [`Synthesis::generate`] and
/// the stand's measured cell: role header → codec control prefix → first text
/// token + codec BOS, the real trailing text followed by EOS, and the pad
/// embedding the loop falls back to once the text is spent.
pub(crate) struct TalkerPrompt {
    pub x: Vec<f32>,
    pub rows: usize,
    pub trailing: Vec<Vec<f32>>,
    pub pad: Vec<f32>,
    pub speaker_norm: f32,
    pub text0_norm: f32,
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
    let mut x = Vec::with_capacity(rows * d);
    for r in &prompt {
        x.extend_from_slice(r);
    }
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let speaker_norm = norm(speaker_embed);
    let text0_norm = norm(&row(first_text_id)?);
    Ok(TalkerPrompt {
        x,
        rows,
        trailing,
        pad,
        speaker_norm,
        text0_norm,
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

/// Codec windows over the part's frames: `(p0, p1, a, b)` — the padded span the
/// DAC graphs read (positions `p0 · 32 .. p1 · 32` of the front's output) and
/// the core `[a, b)` the stitch keeps. The first and last window are not padded
/// past the part's own edges.
pub(crate) fn windows(
    rows: usize,
    window: usize,
    margin: usize,
) -> Vec<(usize, usize, usize, usize)> {
    debug_assert!(margin < window);
    let mut v = Vec::new();
    let mut a = 0;
    if rows > 0 {
        let b = window.min(rows);
        v.push((0, (b + margin).min(rows), 0, b));
        a = b;
    }
    while a < rows {
        let b = (a + window).min(rows);
        v.push((a.saturating_sub(margin), (b + margin).min(rows), a, b));
        a = b;
    }
    v
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

/// One compiled synthesis pipeline for one talker pair and one reference voice.
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
    /// The prompt and body graphs are shape-only, and a TTS request that repeats
    /// a prompt length reuses them instead of paying the compile again — the
    /// bench's corpus cycles phrases across seeds, and a served voice answers
    /// requests of similar length. One entry each: the sound graphs' arenas are
    /// far larger and consecutive requests are the ones that repeat.
    prefill: Option<(usize, Compiled)>,
    body: Option<(usize, Compiled)>,
}

impl Synthesis {
    /// Compile the code predictor's whole sixteen-level frame as one graph, and
    /// the speaker encoder, and embed the reference voice once. The prompt and
    /// body graphs are built on first use and reused while the shape repeats
    /// (`Synthesis::prefill`, `Synthesis::body`); the sound graphs are built per
    /// call.
    pub fn new(pair: Arc<Pair>, voice_wav: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_model(&pair.body)?;
        let cfg = pair.body.config().clone();
        let body_vocab = match cfg.head {
            Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
        };
        let hkv = cfg.kv_heads * cfg.head_dim;
        let (top_k, top_p, temperature) = pair.body.sampling();
        let steps = graphs::code_predictor_steps(&pair.sound)?;
        let capacity_pred = steps + 1;
        let (d_talker, _) = pair.sound.code_proj_widths()?;

        // The reference loads the voice at its own length and resamples it to
        // 24 kHz: no offset and no 5-second cut.
        let voice = voice_wav.display().to_string();
        let audio = host::wav_mono(&voice, SAMPLE_RATE, 0, usize::MAX)
            .map_err(|e| anyhow!("speaker audio `{voice}`: {e}"))?;
        if audio.is_empty() {
            return Err(anyhow!("speaker audio `{voice}` is empty"));
        }
        let (mel, frames) = host::mel_24k(&audio);
        let mut speaker = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::speaker_encoder(pair.sound.clone(), frames, cfg.d)?,
        )?;
        let mut embed = speaker.run_slices(&[("mel", &mel)]);
        let speaker_embed = embed
            .drain(..)
            .next()
            .ok_or_else(|| anyhow!("the speaker encoder returned no embedding"))?;

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
        })
    }

    /// Synthesize `text` with the given sampling seed. Returns mono 24 kHz PCM
    /// of `frames · graphs::samples_per_frame()` samples.
    pub fn synthesize(&mut self, text: &str, seed: u64) -> Result<Vec<f32>> {
        let mut text_ids = self.tokenizer.encode(text);
        text_ids.truncate(MAX_TEXT_TOKENS);
        if text_ids.is_empty() {
            return Err(anyhow!("the tokenizer gave no text tokens for `{text}`"));
        }
        let frames = self.generate(&text_ids, seed)?;
        self.render_sound(&frames)
    }

    /// The talker's prompt, then one body step and a fifteen-level predictor
    /// walk per frame. Returns each frame's codes, level 0 first.
    fn generate(&mut self, text_ids: &[usize], seed: u64) -> Result<Vec<Vec<usize>>> {
        let d = self.cfg.d;
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
            let compiled = Compiled::new(
                device,
                graphs::body_new_rows_head_chain(self.pair.body.clone(), prompt_rows, 0, true)?,
            )?;
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

        // The talker on the part's fixed-capacity plane cache: one row per layer
        // per frame, the prompt rows copied into each layer's own
        // `[capacity + 1, hkv]` cache before the first step.
        let capacity = prompt_rows + MAX_CODEC_ROWS;
        if self.body.as_ref().is_none_or(|(rows, _)| *rows != capacity) {
            self.body = None;
            let compiled = Compiled::new(
                device,
                graphs::body_capacity_chain_planes(self.pair.body.clone(), capacity)?,
            )?;
            self.body = Some((capacity, compiled));
        }
        let body = &mut self.body.as_mut().expect("talker graph").1;
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
        let graphs::PlaneCache {
            k: k_cache,
            v: v_cache,
            mut mask,
        } = graphs::seed_plane_cache(layers, capacity, prompt_rows, self.hkv, k_rows, v_rows);
        if !body.runner.resident_cache_planes(&k_cache, &v_cache) {
            return Err(anyhow!("the talker refused the resident K/V cache"));
        }

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
        let mut all_codes: Vec<Vec<usize>> = Vec::with_capacity(MAX_CODEC_ROWS);
        let mut written = prompt_rows;
        for f in 0..MAX_CODEC_ROWS {
            // The reference stops on codec EOS at level 0, but never before two
            // frames; EOS itself is neither penalised nor fed to the sound half.
            if f >= 2 && code0 == TTS_CODEC_EOS {
                break;
            }
            code_seen.push(code0);
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
            if f + 1 < MAX_CODEC_ROWS {
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
        Ok(all_codes)
    }

    /// The sound half: codes → codebook → wave transformer → DAC, windowed to
    /// keep each graph's plan small. Stitches into one waveform of
    /// `frames · graphs::samples_per_frame()` samples.
    fn render_sound(&self, all_codes: &[Vec<usize>]) -> Result<Vec<f32>> {
        let frames = all_codes.len();
        if frames == 0 {
            return Err(anyhow!("the talker produced no codec frames"));
        }
        let sound = &self.pair.sound;
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
        let prefix = FRONT_WINDOW.saturating_sub(1);

        let device = rlx_runtime::Device::Gpu;
        let mut front_first = Compiled::new(
            device,
            graphs::synthesis_sound_group(sound.clone(), FRONT_WINDOW, front_group.clone(), 0)?,
        )?;
        let mut front_next = if frames > FRONT_WINDOW {
            Some(Compiled::new(
                device,
                graphs::synthesis_sound_group(
                    sound.clone(),
                    FRONT_WINDOW,
                    front_group.clone(),
                    prefix,
                )?,
            )?)
        } else {
            None
        };

        // The wave transformer carries a 72-frame window: 71 K/V frames per
        // layer, the last `pre_conv` frames and the global position. Carry the
        // state window to window.
        let (front_ch, front_per) = graphs::dac_stage_io(front_group.end, 1);
        let mut wave = vec![0f32; front_ch * frames * front_per];
        let mut kv_k = vec![0f32; wave_layers * prefix * wave_hkv];
        let mut kv_v = vec![0f32; wave_layers * prefix * wave_hkv];
        let mut left = vec![0f32; wave_d_tfm * (wave_k - 1)];
        for (a, b) in windows(frames, FRONT_WINDOW, 0)
            .into_iter()
            .map(|(_, _, a, b)| (a, b))
        {
            let len = b - a;
            let mut chunk = Vec::with_capacity((levels + 1) * FRONT_WINDOW);
            for level in 0..=levels {
                for f in a..b {
                    chunk.push(flat[level * frames + f]);
                }
                chunk.resize(chunk.len() + (FRONT_WINDOW - len), 0.0);
            }
            let (cos, sin) = host::rope_tables_at(a, FRONT_WINDOW, wave_hd, 1e4);
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
            let graph = if a == 0 {
                &mut front_first
            } else {
                front_next.as_mut().ok_or_else(|| {
                    anyhow!("a carried wave-transformer front is missing for frame {a}")
                })?
            };
            let run = graph.run_slices(&inputs);
            if run.len() < 4 {
                return Err(anyhow!(
                    "the wave transformer returned {} outputs, expected waveform + left + k/v",
                    run.len()
                ));
            }
            let chunk = &run[0];
            for ch in 0..front_ch {
                let src = ch * FRONT_WINDOW * front_per;
                let dst = ch * frames * front_per + a * front_per;
                wave[dst..dst + len * front_per]
                    .copy_from_slice(&chunk[src..src + len * front_per]);
            }
            // Side outputs are published in this order: `left`, `knew`, `vnew`.
            // Keep the last 71 rows of this window for the next one.
            let knew = &run[2];
            let vnew = &run[3];
            if prefix > 0 && knew.len() >= wave_layers * FRONT_WINDOW * wave_hkv {
                for l in 0..wave_layers {
                    let src = (l * FRONT_WINDOW + (FRONT_WINDOW - prefix)) * wave_hkv;
                    let dst = l * prefix * wave_hkv;
                    kv_k[dst..dst + prefix * wave_hkv]
                        .copy_from_slice(&knew[src..src + prefix * wave_hkv]);
                    kv_v[dst..dst + prefix * wave_hkv]
                        .copy_from_slice(&vnew[src..src + prefix * wave_hkv]);
                }
            }
            left = run[1].clone();
        }

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
            let mut lengths: Vec<usize> = parts.iter().map(|&(p0, p1, _, _)| p1 - p0).collect();
            lengths.sort_unstable();
            lengths.dedup();
            let mut graphs_by_len: Vec<(usize, Compiled)> = Vec::with_capacity(lengths.len());
            for &len in &lengths {
                graphs_by_len.push((
                    len,
                    Compiled::new(
                        device,
                        graphs::synthesis_sound_group(sound.clone(), len, group.clone(), 0)?,
                    )?,
                ));
            }
            // The group's output is a tensor of its own shape, not a waveform
            // until the group that carries `post`. Stitch it channel-major at
            // `per_out`, exactly as the next group's input expects it.
            let mut stitched = vec![0f32; ch_out * frames * per_out];
            for &(p0, p1, a, b) in &parts {
                let len = p1 - p0;
                let chunk = codec_slice(&wave, frames, p0, p1, per_in);
                let gi = graphs_by_len
                    .iter()
                    .position(|(l, _)| *l == len)
                    .ok_or_else(|| anyhow!("no sound graph was built for a {len}-frame window"))?;
                let run = graphs_by_len[gi].1.run_slices(&[("codec_in", &chunk)]);
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
            wave = stitched;
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
    use super::windows;

    /// The codec's windows tile the part: every frame falls in exactly one core, a
    /// core sits inside its padded span, no span leaves the part or reaches more
    /// than one margin past its core — and the first window's span is the window
    /// plus a margin, not the front's chunk, which is what lets a window wider than
    /// [`super::FRONT_WINDOW`] run at all.
    #[test]
    fn the_codec_windows_tile_the_part_and_clip_at_its_edges() {
        for (rows, window, margin) in [
            (0usize, 144usize, 4usize),
            (1, 144, 4),
            (144, 144, 4),
            (145, 144, 4),
            (330, 144, 4),
            (200, 72, 4),
            (7, 3, 0),
        ] {
            let v = windows(rows, window, margin);
            for w in v.windows(2) {
                assert_eq!(
                    w[0].3, w[1].2,
                    "rows={rows}: a core starts where the last ended"
                );
            }
            for &(p0, p1, a, b) in &v {
                assert!(
                    p0 <= a && a < b && b <= p1 && p1 <= rows,
                    "rows={rows}: core {a}..{b} is not inside its span {p0}..{p1}"
                );
                assert!(
                    p0 + margin >= a && p1 - p0 <= window + 2 * margin,
                    "rows={rows}: span {p0}..{p1} reaches past its core's margin"
                );
            }
            let covered: Vec<(usize, usize)> = v.iter().map(|w| (w.2, w.3)).collect();
            assert!(
                covered.is_empty() || (covered[0].0 == 0 && covered[covered.len() - 1].1 == rows),
                "rows={rows}: the cores cover the part"
            );
        }
        // The part's own shape: three windows, the first padded forward past the
        // front's 72-frame chunk.
        assert_eq!(
            windows(330, 144, 4),
            vec![(0, 148, 0, 144), (140, 292, 144, 288), (284, 330, 288, 330)]
        );
    }
}
