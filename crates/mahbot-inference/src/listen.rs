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
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::graphs::{self, Compiled, Outcome};
use crate::loader::Pair;

/// Audio rate the listening path takes.
pub const SAMPLE_RATE: u32 = 16_000;

/// Mel frames in the wake window: 76 frames at a 10 ms hop. The mel front end
/// computes one frame past the window and drops it.
pub const MEL_FRAMES: usize = 76;

/// Samples in the wake window: [`MEL_FRAMES`] hops.
pub const WINDOW_SAMPLES: usize = 12_160;

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
/// The ASR config's `eos_token_id` is 151645; 151643 is the vocabulary's
/// `<|endoftext|>`, not a second config EOS. The path stops on either.
pub const ASR_EOS_ALSO: usize = 151643;

pub fn is_asr_eos(id: usize) -> bool {
    id == ASR_EOS || id == ASR_EOS_ALSO
}

/// The front end's three graphs for one frame count: mel, encoder and projector.
/// Built and run in this order everywhere the cells measure — the ear's window,
/// the ear's shorter cells, a recognition segment — so the input names and the
/// frame arithmetic are stated once. One window per call, which is the graph the
/// stand's numbers belong to.
pub(crate) struct Front {
    mel: Compiled,
    encoder: Compiled,
    projector: Compiled,
    build_ms: f64,
}

impl Front {
    pub(crate) fn build(pair: &Arc<Pair>, frames: usize) -> Result<Self> {
        let t = crate::report::Stopwatch::start();
        let tokens = graphs::asr_tokens(frames);
        let mel = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::mel_frontend(frames, pair.sound.clip().mel_bins)?,
        )?;
        let encoder = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::asr_encoder(pair.sound.clone(), frames)?,
        )?;
        let projector = Compiled::new(
            rlx_runtime::Device::Gpu,
            graphs::projector_chain(pair.clone(), tokens)?,
        )?;
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
    pub(crate) fn run(&mut self, audio: Vec<f32>, reps: usize) -> (Outcome, Outcome, Outcome) {
        let mel = self.mel.run(&[("audio".into(), audio)], reps);
        let encoder = self
            .encoder
            .run(&[("mel".into(), mel.outs[0].clone())], reps);
        let projector = self
            .projector
            .run(&[("features".into(), encoder.outs[0].clone())], reps);
        (mel, encoder, projector)
    }
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

    /// One listener's memory: the pair's two model files (its graphs bind their
    /// weights from the mapped file, so the file sizes are the weight figure the
    /// planned arenas do not carry), plus every compiled graph's planned arena and
    /// uploaded parameters.
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

        let mut single = Front::build(&pair, MEL_FRAMES).expect("the single-window front");
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
