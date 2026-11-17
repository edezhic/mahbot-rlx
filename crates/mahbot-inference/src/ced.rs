// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CED audio tagger — an AudioSet classifier over a log-mel spectrogram, on
//! the wgpu backend.
//!
//! The model is a DeiT-shaped ViT (`mispeech/ced-small`, from `mudler/ced-gguf`):
//! a non-overlapping 16×16 convolution over a 64-bin log-mel, one position table
//! over the 4×63 patches it makes, twelve pre-norm blocks, a mean over the
//! tokens, and a 527-way sigmoid head over the AudioSet ontology. Everything the
//! file states is read from the file ([`CedConfig`]), including the front end's
//! own filterbank, window and clamp: nothing about this model is a constant here
//! except the pins below.
//!
//! Three decisions shape the split between host and device, and each is the
//! reason a piece sits where it does:
//!
//! * **the log-mel is on the host.** The filterbank and the window are tensors,
//!   the transform is a 512-point real FFT, and the clamp's threshold is one
//!   maximum over the whole utterance — a graph would state all three for one
//!   input length at a time, and the file's front end is cheap (one FFT and one
//!   small dot product per frame). The FFT is radix-2 and written here: the crate
//!   takes no dependency for it, and `n_fft` is the file's (`512 = 2⁹`), so the
//!   front end's constructor refuses anything that is not a power of two.
//! * **the patch embedding is a matmul.** The convolution's kernel and stride are
//!   equal — checked, not assumed — so each token is a disjoint 16×16 tile of the
//!   (batch-norm-folded) log-mel and the projection is exactly one `[252, 256] ·
//!   [256, 384]` multiply. The kernel is transposed once at open, and the two
//!   position tables are summed into one `[252, 384]` table at the same time:
//!   both are functions of the file, not of the window.
//! * **the twelve blocks are one graph.** They are the same weights over the same
//!   rows, so one compile serves every window; [`Tagger::probabilities`] refills
//!   the input and reads the 527 columns back.
//!
//! Weights: 49 of the file's 160 tensors are rank-2 `F16` matrices; the other 111
//! are `F32` — among them the norms' weights and biases, the attention and MLP
//! biases, the two position tables, the front end's filterbank and window, the
//! batch-norm statistics and the patch projection, which alone is nearly half of the
//! file's f32 bytes. [`Store::F16Linear`] declares those 49 as f16 params — the
//! only plan in this crate that does. It does not avoid a widening: each matrix is
//! widened to f32, transposed into `MatMul`'s layout and narrowed back, and the
//! session widens those bytes again into the f32 arena. What it buys is the size of
//! the typed image the build hands the session — [`Model::linear_f16`] states it.
//!
//! Which kernel serves a multiply is the adapter's answer as much as the shape's.
//! The cooperative f16 multiply wants whole 64-row output tiles, whole 64-column
//! tiles and 32-deep K stages, and this device has that kernel — a window is 252
//! tokens, which is not a multiple of 64 on any device, so these multiplies are
//! refused there. The blocks' projections run on the f32 cooperative form over the
//! weight's f32 arena image on this machine, and on the plain tiled kernel wherever
//! that form is not admitted. The 527-wide head is one row by 527 columns — it
//! misses the tile's 64-row term and its 64-column term both, so padding the rows
//! would not admit it — and is served by the plain tiled path, at one row by that
//! path's vector-matrix arm.
//!
//! Admitting the f16 tile would mean padding the token axis to 256, and four
//! tokens of nothing inside the attention are a different model, not a faster one —
//! not worth an arithmetic the file did not describe, and there is no padded variant
//! in the tree to re-measure one with. Measured on an M3 Pro, one window costs
//! **10.5 ms** the way it runs against **22.2 ms** with `RLX_WGPU_MATMUL_F32_ONLY=1`
//! (every multiply on the portable tiled kernel) — `tests/ced_tagger.rs` prints the
//! first figure, one window per music-pool clip, seven of the eight agreeing to a
//! tenth of a millisecond and the first carrying the process's first dispatch; the
//! same test with that switch set prints the second. The tagger is a few percent of
//! a transcription pass. The graph is compiled with the weight-concat fusion off,
//! which is what keeps an f16 `Concat` out of it: that node would make the runtime
//! promote the whole graph to f32 arithmetic.
//!
//! What this module does **not** establish: that its numbers are the reference
//! model's. There is no golden fixture for CED here; what `tests/ced_tagger.rs`
//! pins is that the labels are the ones the audio deserves — every vocal clip
//! keeps a voice label among its five best, the a cappella one is not called an
//! instrument, and the instrumental clips are called music rather than speech.
//! The front end's arithmetic is checked against a naive DFT and against a
//! tone's known mel band in this module's own tests, which need neither a device
//! nor the file.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rlx_flow::{FlowCtx, FlowValue, LayerStage, ModelFlow, StageArtifacts};
use rlx_ir::hir::HirGraphExt;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, HirNodeId, Shape};
use rlx_runtime::Device;

use crate::emit::{self, LoaderWeights};
use crate::graphs;
use crate::loader::{CedConfig, Model, Store};

/// The tagger file's name. It is an input like every other model: the caller joins
/// it to the one fixed model directory ([`crate::paths::root`]).
pub const CED_MODEL: &str = "ced-small-f16.gguf";

/// The artefact the labels are taken with, as `(repository, revision, sha256)`:
/// `mudler/ced-gguf`'s f16 conversion of `mispeech/ced-small` (Apache-2.0) at
/// the revision whose LFS object carries that hash. A run whose own
/// `ced-small-f16.gguf` hashes differently did not measure this file.
pub const CED_MODEL_SOURCE: (&str, &str, &str) = (
    "mudler/ced-gguf",
    "b5e9a4aad6438763c8da16079d77563fbed35c65",
    "c391ed8697a1b08d7c1a463e4940a5c3a2f670e0544ab0d8ee23b544583602a8",
);

/// One file's log-mel, in the model's own front end: `n_mels · frames`, mel-major
/// (`values[m · frames + t]`), in dB with the file's clamp applied.
pub struct Mel {
    pub values: Vec<f32>,
    pub frames: usize,
    pub n_mels: usize,
    /// The clamp's floor, `max - top_db`, over the whole utterance — the value a
    /// window past the signal is padded with. `loader` refuses a non-positive
    /// `top_db`, so the clamp only raises the bottom: the maximum of the clamped
    /// mel is the unclamped one, and this is `max(values) - top_db`.
    /// [`Tagger::log_mel`] computes it, once per mel.
    ///
    /// `patch_matrix` used to recompute it per window, folding over the whole mel
    /// again: 62 windows over 2.03 M values on the pinned 318 s recording, 12 ms of
    /// the tagger phase — 826.4/831.5 ms cold and warm before it, 817.8/816.3
    /// after, takes at load average 2.37 and 2.20.
    pub floor: f32,
}

/// One window's decision: the mel frames it covers and the label it named.
pub struct Tag {
    pub start: usize,
    pub end: usize,
    pub index: usize,
    pub score: f32,
}

// ── the front end ──────────────────────────────────────────────────

/// A real input's transform of `n` points: the `n/2`-point complex radix-2 FFT of
/// the frame's even/odd interleave, untangled into the `n/2 + 1` bins a real
/// spectrum has.
///
/// A real frame's spectrum is conjugate-symmetric, so half of a complex
/// transform's butterflies compute bins the caller throws away. Packing
/// `x[2k] + i·x[2k+1]` into a half-length transform and splitting it back out is
/// `n/4 · log₂(n/2)` butterflies against `n/2 · log₂ n` — 1024 against 2304 at
/// `n = 512`. Both twiddle tables are computed once here: computing the split's
/// per frame is what makes a naive half-length form slower than the complex one
/// it replaces.
///
/// On the pinned 318 s recording this is 34 ms of the tagger phase: 863.9/862.7 ms
/// cold and warm before it, 826.4/831.5 after, takes on one M3 Pro at load average
/// 1.24 and 2.37. The two forms are not bit-identical — the untangle rounds where
/// the complex transform does not, 7.1e-6 against 3.8e-6 of the spectrum's peak
/// against a naive DFT — and no label of that recording moved.
struct Fft {
    /// `n/2`: the packed transform's length.
    half: usize,
    /// The packed transform's bit-reversal permutation and its `half/2` twiddles.
    rev: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// `exp(-2πi k/n)` for `k` in `0..=half`: the untangle's twiddles.
    split_cos: Vec<f32>,
    split_sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Result<Self> {
        if n < 4 || !n.is_power_of_two() {
            bail!(
                "the tagger's front end is a radix-2 FFT of at least four points and `n_fft` is {n}"
            );
        }
        let half = n / 2;
        let bits = half.trailing_zeros();
        let rev = (0..half)
            .map(|i| (i as u32).reverse_bits() >> (32 - bits))
            .collect();
        let mut cos = vec![0f32; half / 2];
        let mut sin = vec![0f32; half / 2];
        for k in 0..half / 2 {
            let a = -2.0 * std::f64::consts::PI * k as f64 / half as f64;
            cos[k] = a.cos() as f32;
            sin[k] = a.sin() as f32;
        }
        let mut split_cos = vec![0f32; half + 1];
        let mut split_sin = vec![0f32; half + 1];
        for k in 0..=half {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            split_cos[k] = a.cos() as f32;
            split_sin[k] = a.sin() as f32;
        }
        Ok(Self {
            half,
            rev,
            cos,
            sin,
            split_cos,
            split_sin,
        })
    }

    /// `re² + im²` of the `n/2 + 1` bins a real transform has, into `out`, which
    /// is that long — the front end's `n_freqs`, and the length of the untangle's
    /// twiddle table. A `frame` shorter than `n` is zero-padded (the front end
    /// refuses that case). `re`/`im` are the caller's scratch, `n/2` each.
    fn power_into(&self, frame: &[f32], out: &mut [f32], re: &mut [f32], im: &mut [f32]) {
        let half = self.half;
        // `z[k] = x[2k] + i·x[2k+1]`, permuted into the transform's own order on
        // the way in, so the frame is read once.
        for (i, &j) in self.rev.iter().enumerate() {
            let j = j as usize;
            re[j] = frame.get(2 * i).copied().unwrap_or(0.0);
            im[j] = frame.get(2 * i + 1).copied().unwrap_or(0.0);
        }
        let mut len = 2;
        while len <= half {
            let (h, step) = (len / 2, half / len);
            for start in (0..half).step_by(len) {
                for k in 0..h {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (ar, ai) = (re[start + k], im[start + k]);
                    let (br, bi) = (re[start + k + h], im[start + k + h]);
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    re[start + k] = ar + tr;
                    im[start + k] = ai + ti;
                    re[start + k + h] = ar - tr;
                    im[start + k + h] = ai - ti;
                }
            }
            len *= 2;
        }
        // Untangle: `X[k] = E[k] + exp(-2πi k/n)·O[k]`, with `E` and `O` the
        // even and odd halves' spectra — `(Z[k] ± conj Z[-k])/2` and that
        // difference over `2i`.
        for (k, slot) in out.iter_mut().enumerate() {
            let i = k % half;
            let j = (half - i) % half;
            let (zr, zi) = (re[i], im[i]);
            let (cr, ci) = (re[j], -im[j]);
            let (dr, di) = (zr - cr, zi - ci);
            let (er, ei) = ((zr + cr) * 0.5, (zi + ci) * 0.5);
            let (or_, oi) = (di * 0.5, -dr * 0.5);
            let (wr, wi) = (self.split_cos[k], self.split_sin[k]);
            let xr = er + wr * or_ - wi * oi;
            let xi = ei + wr * oi + wi * or_;
            *slot = xr * xr + xi * xi;
        }
    }
}

/// The file's front end: its filterbank, its window and its amplitude-to-dB
/// constants, with the frames cut out of the signal the way torchaudio
/// `MelSpectrogram(center=true, power=2.0)` does it.
struct FrontEnd {
    /// Each band's non-zeros as `(first bin, weights)`: the file's filters are
    /// triangles over a handful of bins, so the dense product over `n_freqs` is
    /// almost all zeros. The fold below is the same arithmetic — an exact zero
    /// contributes exactly nothing — without them.
    bands: Vec<(usize, Vec<f32>)>,
    window: Vec<f32>,
    n_mels: usize,
    n_freqs: usize,
    n_fft: usize,
    hop: usize,
    multiplier: f32,
    amin: f32,
    top_db: f32,
    fft: Fft,
}

impl FrontEnd {
    fn new(config: &CedConfig, filterbank: Vec<f32>, window: Vec<f32>) -> Result<Self> {
        let want = config.n_mels * config.n_freqs;
        if filterbank.len() != want {
            bail!(
                "`ced.mel_filterbank` has {} values, `{} × {}` mel bins want {want}",
                filterbank.len(),
                config.n_mels,
                config.n_freqs
            );
        }
        if window.len() != config.win_size {
            bail!(
                "`ced.mel_window` has {} values, `ced.win_size` is {}",
                window.len(),
                config.win_size
            );
        }
        if config.hop_size == 0 {
            bail!("`ced.hop_size` is 0");
        }
        // The transform's own bins: a mismatch here is an out-of-range write in
        // the power spectrum, so the constructor states it even though the file
        // loader already refused such a metadata block.
        if config.n_freqs != config.n_fft / 2 + 1 {
            bail!(
                "`ced.n_freqs` is {}, not the {} bins a {}-point real transform has",
                config.n_freqs,
                config.n_fft / 2 + 1,
                config.n_fft
            );
        }
        let bands = (0..config.n_mels)
            .map(|m| {
                let row = &filterbank[m * config.n_freqs..(m + 1) * config.n_freqs];
                let first = row.iter().position(|v| *v != 0.0).unwrap_or(row.len());
                let last = row.iter().rposition(|v| *v != 0.0).map_or(first, |i| i + 1);
                (first, row[first..last].to_vec())
            })
            .collect();
        Ok(Self {
            bands,
            window,
            n_mels: config.n_mels,
            n_freqs: config.n_freqs,
            n_fft: config.n_fft,
            hop: config.hop_size,
            multiplier: config.a2db_multiplier,
            amin: config.a2db_amin,
            top_db: config.a2db_top_db,
            fft: Fft::new(config.n_fft)?,
        })
    }

    /// The whole signal's log-mel.
    ///
    /// The signal is reflect-padded by half a window on both sides (torch's
    /// `mode="reflect"`: the edge sample is not repeated), so there are
    /// `1 + samples / hop` frames and frame `t` starts at `t · hop`. Each frame
    /// is windowed, transformed, squared into a power spectrum, filtered, and
    /// finally turned into dB and clamped at `max - top_db` over the whole
    /// utterance — the order torchaudio's `AmplitudeToDB` uses, whose `ref=1.0`
    /// is why there is no offset.
    fn log_mel(&self, samples: &[f32]) -> Result<Mel> {
        let pad = self.n_fft / 2;
        if samples.len() <= pad {
            bail!(
                "the front end reflect-pads {pad} samples each side, so it needs more than \
                 {pad} samples — got {}",
                samples.len()
            );
        }
        let frames = 1 + samples.len() / self.hop;
        let mut values = vec![0f32; self.n_mels * frames];
        let mut power = vec![0f32; self.n_freqs];
        let mut frame = vec![0f32; self.n_fft];
        // Two scratch halves, allocated once for the whole utterance rather than
        // two per frame.
        let mut re = vec![0f32; self.fft.half];
        let mut im = vec![0f32; self.fft.half];
        // A reflect-padded read: the left pad mirrors `x[pad - i]`, the right one
        // `x[n - 2 - i]`, and inside the signal it is the signal.
        let at = |index: usize| -> f32 {
            let i = index as isize - pad as isize;
            if i < 0 {
                samples[(-i) as usize]
            } else if (i as usize) < samples.len() {
                samples[i as usize]
            } else {
                samples[2 * samples.len() - 2 - i as usize]
            }
        };
        for t in 0..frames {
            for (j, slot) in frame.iter_mut().enumerate() {
                *slot = at(t * self.hop + j) * self.window[j];
            }
            self.fft.power_into(&frame, &mut power, &mut re, &mut im);
            for (m, (first, weights)) in self.bands.iter().enumerate() {
                let mut acc = 0f32;
                for (w, p) in weights.iter().zip(&power[*first..]) {
                    acc += w * p;
                }
                values[m * frames + t] = acc;
            }
        }
        let mut max_db = f32::NEG_INFINITY;
        for v in values.iter_mut() {
            let db = self.multiplier * v.max(self.amin).log10();
            *v = db;
            max_db = max_db.max(db);
        }
        let floor = max_db - self.top_db;
        for v in values.iter_mut() {
            *v = v.max(floor);
        }
        Ok(Mel {
            values,
            frames,
            n_mels: self.n_mels,
            floor,
        })
    }
}

// ── the file's tables, computed once at open ───────────────────────

/// `init_bn` folded into two per-bin vectors, `x · scale + shift`.
///
/// The model applies the batch norm to the log-mel *before* the patches are cut,
/// so it folds into the patch lift exactly: an eval-mode `BatchNorm1d` is an
/// affine map per bin.
fn batch_norm_fold(model: &Model, c: &CedConfig) -> Result<(Vec<f32>, Vec<f32>)> {
    let read = |name: &str| -> Result<Vec<f32>> { Ok(model.f32_file_order(name)?.0) };
    let weight = read("encoder.init_bn.weight")?;
    let bias = read("encoder.init_bn.bias")?;
    let mean = read("encoder.init_bn.running_mean")?;
    let var = read("encoder.init_bn.running_var")?;
    for (name, v) in [
        ("encoder.init_bn.weight", &weight),
        ("encoder.init_bn.bias", &bias),
        ("encoder.init_bn.running_mean", &mean),
        ("encoder.init_bn.running_var", &var),
    ] {
        if v.len() != c.n_mels {
            bail!(
                "`{name}` has {} values, `ced.n_mels` is {}",
                v.len(),
                c.n_mels
            );
        }
    }
    let mut scale = vec![0f32; c.n_mels];
    let mut shift = vec![0f32; c.n_mels];
    for m in 0..c.n_mels {
        let s = weight[m] / (var[m] + c.bn_eps).sqrt();
        scale[m] = s;
        shift[m] = bias[m] - mean[m] * s;
    }
    Ok((scale, shift))
}

/// The patch kernel as the matmul's weight `[patch², embed_dim]`.
///
/// The file holds the convolution's kernel as `ne = [kw, kh, 1, oc]` with the
/// data in the framework's `(oc, 1, kh, kw)` C order — `oc` slowest, the kernel
/// index `j = kw + patch · kh` fastest — while the multiply wants one row per
/// kernel element with `oc` fastest. So this is a real transpose, written out so
/// the order is checkable rather than a re-interpretation of the same bytes.
fn patch_weight(model: &Model, c: &CedConfig) -> Result<Vec<f32>> {
    let name = "encoder.patch_embed.proj.weight";
    let (conv, dims) = model.f32_file_order(name)?;
    let patch = c.patch_width();
    let want = [c.embed_dim, 1, c.patch_size, c.patch_size];
    if dims != want {
        bail!("`{name}` is {dims:?}, the config says {want:?}");
    }
    if conv.len() != c.embed_dim * patch {
        bail!(
            "`{name}` has {} values, `{} × {patch}` patches want {}",
            conv.len(),
            c.embed_dim,
            c.embed_dim * patch
        );
    }
    let mut wp = vec![0f32; patch * c.embed_dim];
    for oc in 0..c.embed_dim {
        for j in 0..patch {
            wp[j * c.embed_dim + oc] = conv[oc * patch + j];
        }
    }
    Ok(wp)
}

/// The two position tables summed into one `[tokens, embed_dim]` table.
///
/// Both are **position-major in the data** — one `embed_dim`-wide row per
/// position, the embedding fastest (`time_pos[t · d + c]`,
/// `freq_pos[f · d + c]`) — and are broadcast over the other axis, so a token at
/// `(f, t)` reads the sum of its time position's row and its frequency
/// position's row.
///
/// The header's `ne` lists the position axis first (`[63, 1, 384, 1]`), which
/// read as the ggml convention would put `t` fastest; the bytes do not. Read that
/// way, six of the 384 channels are exactly zero across all 63 positions and all
/// four frequency positions — a trained table with six dead channels — while read
/// as one 384-wide row per position nothing is degenerate. The witness test on
/// real audio is what holds this reading: a transposed table moves every label to
/// another kind of sound.
fn position_table(model: &Model, c: &CedConfig) -> Result<Vec<f32>> {
    let (ow, oh, d) = (c.time_tokens(), c.freq_tokens(), c.embed_dim);
    let (time, time_dims) = model.f32_file_order("encoder.time_pos_embed")?;
    let (freq, freq_dims) = model.f32_file_order("encoder.freq_pos_embed")?;
    if time.len() != ow * d {
        bail!(
            "`encoder.time_pos_embed` has {} values, {ow} positions of {d} want {} ({time_dims:?})",
            time.len(),
            ow * d
        );
    }
    if freq.len() != oh * d {
        bail!(
            "`encoder.freq_pos_embed` has {} values, {oh} positions of {d} want {} ({freq_dims:?})",
            freq.len(),
            oh * d
        );
    }
    let mut pos = vec![0f32; ow * oh * d];
    for f in 0..oh {
        for t in 0..ow {
            let row = (f * ow + t) * d;
            for ch in 0..d {
                pos[row + ch] = time[t * d + ch] + freq[f * d + ch];
            }
        }
    }
    Ok(pos)
}

// ── the graph ──────────────────────────────────────────────────────

/// A projection of the tagger's, with its bias and — where the block has one — its
/// activation folded into the multiply.
///
/// The tagger's profile lowers `FusionPolicy::Direct` ([`graphs::profile`]), so a
/// `HirOp::Linear` with a bias is `Op::FusedMatMulBiasAct`: the bias and the
/// activation land in the multiply's epilogue instead of as a binary and a unary
/// step of their own. That is the op `FuseMatMulBiasAct` builds, emitted rather than
/// matched. The pass would refuse this graph on two gates, and neither holds here:
///
/// * **the weight's declared dtype.** The pass fuses only an `F32` weight, because
///   its epilogue reads the weight as f32 and a backend that keeps the half bytes
///   would read garbage. This one stores an `F16` parameter as one f32 lane per
///   element, and the fused op derives its compute path from the same
///   `derive_matmul_compute` call as a bare `MatMul` — so it reads the weight out of
///   the same f32 arena image the unfused multiply read.
/// * **the epilogue's activation.** The pass folds `Gelu` and `Silu` only, so the
///   head's `Sigmoid` would stay a step of its own. Every kernel this graph's shapes
///   can select carries the sigmoid arm: `matmul.wgsl`'s `apply_act` and each
///   cooperative form's.
///
/// The one difference from [`emit::linear`] is the weight's declared dtype: a
/// [`Store::F16Linear`] plan is declared as an f16 `Param`, and its bytes
/// [`Tagger::open`] hands the session. Every other store is [`emit::linear`]'s —
/// the packed route for a quantised file, the widened f32 param for one that is
/// not — with the bias and the activation added beside it, since a packed multiply
/// has no epilogue to fold them into.
///
/// A bias that is not the weight's width is refused — see [`check_bias_width`] —
/// since the pass's own width refusal is never reached on an op emitted rather
/// than matched.
///
/// It is not [`emit::linear_encoder`]: that entry refuses a row count the
/// cooperative f16 tile does not admit, and a tagger window is 252 tokens, which
/// is not a multiple of 64 — the refusal is the row count's, on every device, not a
/// device-family limit. Refusing here would leave this graph with no route at all,
/// so each multiply keeps whatever arm its own geometry and the device admit: the f32
/// cooperative form over the weight's f32 arena image where it is admitted, the plain
/// tiled kernel where it is not. The 527-wide head is one row by 527 columns, so it
/// misses the tile's row term and its column term both, and is on the plain tiled
/// path — at one row, on that path's vector-matrix arm.
///
/// On the pinned 318 s recording it is 102 ms of the tagger phase — 817.8/816.3 ms
/// cold and warm before it, 704.4/727.6 and 705.6/721.4 after, takes at load
/// average 2.20 and 2.40/1.84 — and 63 of the graph's 284 schedule steps: the
/// fifty bias adds and the thirteen activations. The census that follows carries
/// no binary and no unary step, and no step of the fused form lands on the host
/// path (`RLX_WGPU_DBG_COUNTERS`'s `stage` interval covers all 221 of them, and
/// `RLX_WGPU_COMPILE_REPORT` reports no restart).
///
/// The values move: the epilogue's GELU is the same A&S erf polynomial spelled
/// with ten-digit constants and explicit `fma`, where the standalone activation's
/// manifest carries eight, and its sigmoid clamps at ±88 where the standalone one
/// does not. That is ~1e-7 of a probability, and the pinned recording's transcript,
/// regions and digest do not move.
fn ced_linear(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    value: &FlowValue,
    key: &str,
    bias: Option<HirNodeId>,
    activation: Option<Activation>,
) -> Result<HirNodeId> {
    let p = model.plan(key)?;
    let (k, n) = p.in_out();
    // Every call site is one rank-2 `[rows, k]` value, and the out shape carries
    // that row axis and nothing else.
    let rows = value.shape.dims()[0].unwrap_static();
    if let Some(b) = bias {
        check_bias_width(key, &ctx.node_shape(b)?, n)?;
    }
    if p.store != Store::F16Linear {
        let y = emit::linear(ctx, value, key)?.hir_id();
        let y = match bias {
            Some(b) => emit::add(ctx, y, b),
            None => y,
        };
        return Ok(match activation {
            None => y,
            Some(a) => emit::act(ctx, a, y, &[rows, n]),
        });
    }
    let w = ctx.hir().param(key, Shape::new(&[k, n], DType::F16));
    Ok(ctx.hir().linear(
        value.hir_id(),
        w,
        bias,
        activation,
        Shape::new(&[rows, n], DType::F32),
    ))
}

/// A bias must be as wide as the weight's columns: the fused epilogue reads one
/// value per column and validates nothing, the bias width is metadata's and the
/// column count the tensor's.
fn check_bias_width(key: &str, bias: &Shape, n: usize) -> Result<()> {
    let width = bias.dims().last().map_or(0, |d| d.unwrap_static());
    if width != n {
        bail!("`{key}`: bias is {width} wide, the weight has {n} columns");
    }
    Ok(())
}

/// `[tokens, heads · head_dim]` → `[1, heads, tokens, head_dim]`: the operand
/// layout the attention op reads, head-major because a rows-first
/// `[1, tokens, heads, head_dim]` is only unambiguous while `tokens != heads`
/// (see `graphs::block`).
fn head_major(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    tokens: usize,
    heads: usize,
    hd: usize,
) -> HirNodeId {
    let r = emit::reshape(ctx, x, &[1, tokens as i64, heads as i64, hd as i64]);
    emit::h(ctx).transpose_(r, vec![0, 2, 1, 3])
}

/// The tagger's whole device graph: patch tokens in, 527 probabilities out.
struct TaggerStage {
    model: Arc<Model>,
    config: CedConfig,
    /// The patch kernel, transposed — see [`patch_weight`].
    patch: Vec<f32>,
    /// The combined position table — see [`position_table`].
    pos: Vec<f32>,
}

impl LayerStage for TaggerStage {
    fn name(&self) -> &str {
        "ced_tagger"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let c = &self.config;
        let d = c.embed_dim;
        let tokens = c.token_count();
        let (heads, hd) = (c.heads, d / c.heads);
        let rows = Shape::new(&[tokens, d], DType::F32);

        // ── patchify, bias, positions ──
        let wp = emit::synth_param(
            ctx,
            "ced.patch_weight",
            self.patch.clone(),
            &[c.patch_width(), d],
        );
        let bias = emit::f32_param(ctx, model, "encoder.patch_embed.proj.bias", &[d])?;
        let mut x = ctx
            .hir()
            .linear(input.hir_id(), wp, Some(bias), None, rows.clone());
        let pos = emit::synth_param(ctx, "ced.pos", self.pos.clone(), &[tokens, d]);
        x = emit::add(ctx, x, pos);

        // ── the twelve pre-norm blocks ──
        for il in 0..c.depth {
            let p = format!("encoder.blocks.{il}");
            let gamma = emit::f32_param(ctx, model, &format!("{p}.norm1.weight"), &[d])?;
            let beta = emit::f32_param(ctx, model, &format!("{p}.norm1.bias"), &[d])?;
            let n = emit::ln(ctx, x, gamma, beta, c.ln_eps_encoder);
            let v = FlowValue::new(n, rows.clone());
            // Lazy: a file without qkv biases declares no such parameter at all.
            let qkv_bias = c
                .qkv_bias
                .then(|| emit::f32_param(ctx, model, &format!("{p}.attn.qkv.bias"), &[3 * d]))
                .transpose()?;
            let qkv = ced_linear(
                ctx,
                model,
                &v,
                &format!("{p}.attn.qkv.weight"),
                qkv_bias,
                None,
            )?;
            let q = emit::narrow(ctx, qkv, 1, 0, d);
            let q = head_major(ctx, q, tokens, heads, hd);
            let k = emit::narrow(ctx, qkv, 1, d, d);
            let k = head_major(ctx, k, tokens, heads, hd);
            let v = emit::narrow(ctx, qkv, 1, 2 * d, d);
            let v = head_major(ctx, v, tokens, heads, hd);
            // No mask: the twelve blocks attend over every patch of the window.
            let a = emit::attention(
                ctx,
                q,
                k,
                v,
                heads,
                hd,
                MaskKind::None,
                &[1, heads, tokens, hd],
            );
            let a = emit::h(ctx).transpose_(a, vec![0, 2, 1, 3]);
            let a = emit::reshape(ctx, a, &[tokens as i64, d as i64]);
            let v = FlowValue::new(a, rows.clone());
            let base = emit::f32_param(ctx, model, &format!("{p}.attn.proj.bias"), &[d])?;
            let o = ced_linear(
                ctx,
                model,
                &v,
                &format!("{p}.attn.proj.weight"),
                Some(base),
                None,
            )?;
            x = emit::add(ctx, x, o);

            let ffn = c.ffn();
            let gamma = emit::f32_param(ctx, model, &format!("{p}.norm2.weight"), &[d])?;
            let beta = emit::f32_param(ctx, model, &format!("{p}.norm2.bias"), &[d])?;
            let n = emit::ln(ctx, x, gamma, beta, c.ln_eps_encoder);
            let v = FlowValue::new(n, rows.clone());
            let b = emit::f32_param(ctx, model, &format!("{p}.mlp.fc1.bias"), &[ffn])?;
            let h_lin = ced_linear(
                ctx,
                model,
                &v,
                &format!("{p}.mlp.fc1.weight"),
                Some(b),
                Some(Activation::Gelu),
            )?;
            let v = FlowValue::new(h_lin, Shape::new(&[tokens, ffn], DType::F32));
            let b = emit::f32_param(ctx, model, &format!("{p}.mlp.fc2.bias"), &[d])?;
            let h_lin = ced_linear(
                ctx,
                model,
                &v,
                &format!("{p}.mlp.fc2.weight"),
                Some(b),
                None,
            )?;
            x = emit::add(ctx, x, h_lin);
        }

        // ── the head: one mean over the tokens, one norm, one projection ──
        let gamma = emit::f32_param(ctx, model, "encoder.norm.weight", &[d])?;
        let beta = emit::f32_param(ctx, model, "encoder.norm.bias", &[d])?;
        let x = emit::ln(ctx, x, gamma, beta, c.ln_eps_encoder);
        let pooled = emit::h(ctx).mean(x, vec![0], true);
        let gamma = emit::f32_param(ctx, model, "outputlayer.0.weight", &[d])?;
        let beta = emit::f32_param(ctx, model, "outputlayer.0.bias", &[d])?;
        let pooled = emit::ln(ctx, pooled, gamma, beta, c.ln_eps_head);
        let v = FlowValue::new(pooled, Shape::new(&[1, d], DType::F32));
        let b = emit::f32_param(ctx, model, "outputlayer.1.bias", &[c.output_dim])?;
        let y = ced_linear(
            ctx,
            model,
            &v,
            "outputlayer.1.weight",
            Some(b),
            // The classifier is multi-label: one sigmoid per class, no softmax.
            Some(Activation::Sigmoid),
        )?;
        let out = Shape::new(&[1, c.output_dim], DType::F32);
        Ok((
            FlowValue::new(y, out.clone()),
            StageArtifacts::hidden_only(out),
        ))
    }
}

// ── the tagger ─────────────────────────────────────────────────────

/// A loaded tagger: the file's front end, its label list and one compiled graph.
///
/// The model file is dropped once the graph is compiled and its tables are on the
/// host — the weights live on the device from then on, and keeping the mapping
/// would keep 41.7 MiB of pages resident for nothing.
pub struct Tagger {
    config: CedConfig,
    labels: Vec<String>,
    front: FrontEnd,
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    compiled: graphs::Compiled,
}

impl Tagger {
    /// Open the tagger at `path` and compile its graph for the wgpu device.
    pub fn open(path: &Path) -> Result<Self> {
        let model = Arc::new(Model::open_ced(path)?);
        let config = model
            .ced
            .clone()
            .ok_or_else(|| anyhow!("{} opened without a `ced` config", path.display()))?;
        let (filterbank, window) = model.ced_front_end()?;
        let front = FrontEnd::new(&config, filterbank, window)
            .with_context(|| format!("{}: the front end", path.display()))?;
        let (bn_scale, bn_shift) = batch_norm_fold(&model, &config)?;
        let patch = patch_weight(&model, &config)?;
        let pos = position_table(&model, &config)?;
        let labels = config.labels.clone();

        let stage = TaggerStage {
            model: model.clone(),
            config: config.clone(),
            patch,
            pos,
        };
        let flow = ModelFlow::new("ced_tagger")
            .with_profile(graphs::profile())
            .input(
                "patches",
                Shape::new(&[config.token_count(), config.patch_width()], DType::F32),
            )
            .layer_stage(stage)
            .output("labels");
        let mut weights = LoaderWeights::new(model.clone());
        let mut built = flow.build(&mut weights)?;
        // The f16 route's bytes, one `set_param_typed(name, bytes, F16)` per
        // weight the store rule declared f16, drained by the runner like a packed
        // blob.
        for plan in model.plans.iter().filter(|p| p.store == Store::F16Linear) {
            let (bytes, ..) = model.linear_f16(&plan.name)?;
            built
                .typed_params
                .push((plan.name.clone(), bytes, DType::F16));
        }
        let compiled = graphs::compile_encoder(Device::Gpu, built)?;
        drop(model);
        Ok(Self {
            config,
            labels,
            front,
            bn_scale,
            bn_shift,
            compiled,
        })
    }

    /// The AudioSet ontology, one name per output column.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The mel frames one window holds (`ced.target_length`).
    pub fn window_frames(&self) -> usize {
        self.config.target_length
    }

    /// The mel frames between two windows — half a window.
    pub fn hop_frames(&self) -> usize {
        self.config.target_length / 2
    }

    /// 16 kHz samples between two mel frames of this file's front end, which is
    /// the only conversion the labels may be read through: a region's samples come
    /// from the detector's hop, its windows from `ced.hop_size`, and the two need
    /// not be equal.
    pub fn samples_per_frame(&self) -> usize {
        self.front.hop
    }

    /// The host bytes a log-mel of `frames` frames takes: `n_mels × frames` f32.
    /// The transcription call reads this before it starts, so an input whose own
    /// features do not fit the run's ceiling is refused with the figure rather than
    /// as an allocation the guard stops.
    pub fn mel_bytes(&self, frames: usize) -> u64 {
        self.config.n_mels as u64 * frames as u64 * 4
    }

    /// The whole signal's log-mel, in this file's own front end.
    pub fn log_mel(&self, samples_16k: &[f32]) -> Result<Mel> {
        self.front.log_mel(samples_16k)
    }

    /// One window's 527-class probabilities, from mel frame `start`.
    ///
    /// A window that runs past the signal's last frame is padded with the dB
    /// floor — the reference pads with silence, whose log-mel is the clamp's
    /// threshold, `max - top_db`.
    pub fn probabilities(&mut self, mel: &Mel, start: usize) -> Result<Vec<f32>> {
        let patches = self.patches(mel, start)?;
        let outs = self.compiled.runner.run_slices(&[("patches", &patches)]);
        let out = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the tagger graph returned no output"))?;
        if out.len() != self.config.output_dim {
            bail!(
                "the tagger graph returned {} values, `ced.outputdim` is {}",
                out.len(),
                self.config.output_dim
            );
        }
        Ok(out)
    }

    /// The labelled windows of a whole input: one per hop, and one over the tail
    /// so the last frame of the signal is inside a window. An input shorter than a
    /// window gets one padded window — a region is labelled from the windows that
    /// cover it, and an unlabelled region is the one thing this may not produce.
    pub fn tags(&mut self, mel: &Mel) -> Result<Vec<Tag>> {
        let (window, hop) = (self.window_frames(), self.hop_frames());
        if hop == 0 {
            bail!("`ced.target_length` is {window}: half a window is no hop at all");
        }
        let mut starts: Vec<usize> = (0..)
            .map(|i| i * hop)
            .take_while(|s| s + window <= mel.frames)
            .collect();
        let tail = mel.frames.saturating_sub(window);
        if starts.last().copied() != Some(tail) {
            starts.push(tail);
        }
        let mut tags = Vec::with_capacity(starts.len());
        for start in starts {
            let probabilities = self.probabilities(mel, start)?;
            let (index, score) = top1(&probabilities);
            tags.push(Tag {
                start,
                end: start + window,
                index,
                score,
            });
        }
        Ok(tags)
    }

    /// The single label of the mel range `[start_frame, end_frame)`.
    ///
    /// Each window's top-1 is weighted by how many frames of the range it covers,
    /// the label with the greatest total wins, a tie goes to the earliest window,
    /// and the score reported is that window's — a label that holds over most of a
    /// range beats one that spikes at its edge. `None` when no window overlaps the
    /// range at all. The rule itself is the `collapse` helper below, which needs none
    /// of this type's state.
    pub fn label_of(
        &self,
        tags: &[Tag],
        start_frame: usize,
        end_frame: usize,
    ) -> Option<(usize, f32)> {
        collapse(tags, self.config.output_dim, start_frame, end_frame)
    }

    /// One window's patch matrix `[tokens, patch²]`, batch norm folded in.
    fn patches(&self, mel: &Mel, start: usize) -> Result<Vec<f32>> {
        patch_matrix(mel, start, &self.config, &self.bn_scale, &self.bn_shift)
    }
}

/// The window's tokens as the matmul's input: `[tokens, patch²]`, one row per
/// patch, with `init_bn`'s fold applied per mel bin.
///
/// The row index is the reference's flatten order — `f · ow + t`, frequency
/// major and time fastest — and the column is the kernel element with the time
/// offset fastest (`kw + patch · kh`), which is the order the convolution's
/// weight holds.
fn patch_matrix(
    mel: &Mel,
    start: usize,
    c: &CedConfig,
    bn_scale: &[f32],
    bn_shift: &[f32],
) -> Result<Vec<f32>> {
    if mel.n_mels != c.n_mels || mel.values.len() != mel.n_mels * mel.frames {
        bail!(
            "mel is {} bins × {} frames, the model's front is {} bins",
            mel.n_mels,
            mel.frames,
            c.n_mels
        );
    }
    let (ps, ts) = (c.patch_size, c.patch_stride);
    let (ow, oh) = (c.time_tokens(), c.freq_tokens());
    let mut patches = vec![0f32; ow * oh * c.patch_width()];
    for f in 0..oh {
        for t in 0..ow {
            let row = (f * ow + t) * c.patch_width();
            for kh in 0..ps {
                let m = f * ts + kh;
                let (scale, shift) = (bn_scale[m], bn_shift[m]);
                for kw in 0..ps {
                    let u = start + t * ts + kw;
                    let v = if u < mel.frames {
                        mel.values[m * mel.frames + u]
                    } else {
                        mel.floor
                    };
                    patches[row + kw + ps * kh] = v * scale + shift;
                }
            }
        }
    }
    Ok(patches)
}

/// The weighted top-1 of `tags` over `[start_frame, end_frame)`: the label whose
/// windows cover the most frames of the range, the earliest window breaking a
/// tie, scored by that window.
fn collapse(
    tags: &[Tag],
    labels: usize,
    start_frame: usize,
    end_frame: usize,
) -> Option<(usize, f32)> {
    let mut weight = vec![0usize; labels];
    let mut first = vec![usize::MAX; labels];
    for (i, tag) in tags.iter().enumerate() {
        let overlap = tag
            .end
            .min(end_frame)
            .saturating_sub(tag.start.max(start_frame));
        if overlap == 0 || tag.index >= weight.len() {
            continue;
        }
        weight[tag.index] += overlap;
        if first[tag.index] == usize::MAX {
            first[tag.index] = i;
        }
    }
    let mut best: Option<(usize, usize, usize)> = None;
    for (index, &w) in weight.iter().enumerate() {
        if w == 0 {
            continue;
        }
        let at = first[index];
        let better = match best {
            None => true,
            Some((bw, bf, _)) => w > bw || (w == bw && at < bf),
        };
        if better {
            best = Some((w, at, index));
        }
    }
    let (_, at, index) = best?;
    Some((index, tags[at].score))
}

/// The greatest value and its index; ties go to the earliest column.
fn top1(values: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in values.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgufWriter, MetaValue};

    /// The pinned file's shape, as the metadata states it: what the model-free
    /// tests below take for granted is only this, never the weights.
    fn config(n_mels: usize, n_freqs: usize, n_fft: usize, hop: usize) -> CedConfig {
        CedConfig {
            embed_dim: 384,
            depth: 12,
            heads: 6,
            mlp_ratio: 4.0,
            output_dim: 527,
            qkv_bias: true,
            patch_size: 16,
            patch_stride: 16,
            target_length: 1012,
            pooling: "mean".into(),
            ln_eps_encoder: 1e-6,
            ln_eps_head: 1e-5,
            bn_eps: 1e-5,
            sample_rate: 16_000,
            n_mels,
            n_fft,
            win_size: n_fft,
            hop_size: hop,
            n_freqs,
            f_min: 0.0,
            f_max: 8000.0,
            center: true,
            a2db_multiplier: 10.0,
            a2db_amin: 1e-10,
            a2db_top_db: 120.0,
            a2db_ref: 1.0,
            labels: Vec::new(),
        }
    }

    /// The naive transform the FFT has to agree with: `|Σ x[i]·e^{-2πi k i/n}|²`
    /// for every non-negative bin, in f64, straight from the definition.
    fn naive_power(frame: &[f32], n_freqs: usize) -> Vec<f64> {
        let n = frame.len();
        (0..n_freqs)
            .map(|k| {
                let (mut re, mut im) = (0f64, 0f64);
                for (i, &x) in frame.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                    re += x as f64 * a.cos();
                    im += x as f64 * a.sin();
                }
                re * re + im * im
            })
            .collect()
    }

    /// A signal with structure at several scales: two tones, a slow ramp and a
    /// deterministic wobble, so the comparison is not one that a transform with a
    /// permuted or truncated stage could pass.
    fn probe_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                let mut state = (i as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                state ^= state >> 13;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t / 16_000.0).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3_100.0 * t / 16_000.0).cos()
                    + 0.2 * (t / n as f32)
                    + 0.05 * (state >> 24) as f32 / 255.0
            })
            .collect()
    }

    #[test]
    fn the_radix_two_fft_matches_a_naive_dft() {
        let n = 512;
        let signal = probe_signal(n);
        let fft = Fft::new(n).expect("512 is a power of two");
        let mut got = vec![0f32; n / 2 + 1];
        let (mut re, mut im) = (vec![0f32; fft.half], vec![0f32; fft.half]);
        fft.power_into(&signal, &mut got, &mut re, &mut im);
        let want = naive_power(&signal, n / 2 + 1);
        let mut worst = 0f64;
        let mut worst_bin = 0usize;
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            // Relative to the larger of the two: a bin of magnitude 1e6 and one
            // of magnitude 1e-3 cannot both be held to the same absolute bar.
            let scale = (g as f64).abs().max(w.abs()).max(1.0);
            let rel = ((g as f64 - w).abs() / scale).max(1e-12);
            if rel > worst {
                worst = rel;
                worst_bin = k;
            }
        }
        println!("fft: worst relative error {worst:.3e} at bin {worst_bin}");
        assert!(
            worst < 1e-3,
            "the FFT's bin {worst_bin} is off by {worst:.3e} relative to the DFT"
        );
        // The 440 Hz tone is in bin 14.08 at a 16 kHz rate: the transform's peak
        // has to be there and nowhere else.
        let peak = got
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (k, &v)| {
                if v > best.1 { (k, v) } else { best }
            })
            .0;
        assert_eq!(peak, 14, "the 440 Hz tone's peak bin");
    }

    /// A band filter bank: eight bands of eight bins from zero, so a tone's band
    /// is a known row rather than one the file's mel scale decides.
    fn band_front_end() -> (CedConfig, FrontEnd) {
        let config = config(8, 257, 512, 160);
        let mut filterbank = vec![0f32; 8 * 257];
        for m in 0..8 {
            for f in 8 * m..8 * m + 8 {
                filterbank[m * 257 + f] = 1.0;
            }
        }
        let window = vec![1f32; 512];
        let front = FrontEnd::new(&config, filterbank, window).expect("the band filter bank");
        (config, front)
    }

    #[test]
    fn a_440_hz_tone_peaks_in_the_mel_row_that_holds_440_hz() {
        let (_, front) = band_front_end();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = front.log_mel(&samples).expect("a one-second tone");
        assert_eq!(mel.frames, 1 + samples.len() / front.hop);
        assert_eq!(mel.values.len(), mel.n_mels * mel.frames);
        // 440 Hz is bin 14.08, and band 1 covers bins 8..16.
        let frames = mel.frames;
        let row = |m: usize| -> f32 {
            let mid = frames / 2;
            mel.values[m * frames + mid]
        };
        let peak = (0..mel.n_mels)
            .fold((0usize, f32::NEG_INFINITY), |best, m| {
                let v = row(m);
                if v > best.1 { (m, v) } else { best }
            })
            .0;
        println!(
            "tone: peak band {peak} at {:.1} dB, band 2 {:.1} dB",
            row(peak),
            row(2)
        );
        assert_eq!(peak, 1, "440 Hz belongs to band 1, not band {peak}");
        assert!(
            row(1) > row(2) + 20.0,
            "the tone's own band {:.1} dB is not well clear of the next ({:.1} dB)",
            row(1),
            row(2)
        );
    }

    /// A mel whose only non-zero cell is at `(freq bin, time frame)`.
    fn single_cell_mel(c: &CedConfig, m: usize, t: usize, frames: usize) -> Mel {
        let mut values = vec![0f32; c.n_mels * frames];
        values[m * frames + t] = 1.0;
        Mel {
            values,
            frames,
            n_mels: c.n_mels,
            // What the front end's clamp leaves on a mel whose maximum is the
            // one cell: `max - top_db`.
            floor: 1.0 - c.a2db_top_db,
        }
    }

    #[test]
    fn a_mel_cell_lands_in_the_token_and_column_the_reference_flatten_order_names() {
        let c = config(64, 257, 512, 160);
        let ones = vec![1f32; c.n_mels];
        let zeros = vec![0f32; c.n_mels];
        // The kernel offsets this cell falls on: freq offset 3 of patch row 2,
        // time offset 7 of patch column 5.
        let (f, kh, t, kw) = (2usize, 3usize, 5usize, 7usize);
        let (m, u) = (f * c.patch_stride + kh, t * c.patch_stride + kw);
        let frames = c.target_length;
        let mel = single_cell_mel(&c, m, u, frames);
        let patches = patch_matrix(&mel, 0, &c, &ones, &zeros).expect("one window");

        let (ow, oh) = (c.time_tokens(), c.freq_tokens());
        assert_eq!((ow, oh), (63, 4), "this file's token grid");
        assert_eq!(patches.len(), ow * oh * c.patch_width());

        let token = f * ow + t;
        let column = kw + c.patch_size * kh;
        let non_zero: Vec<(usize, usize)> = patches
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, _)| (i / c.patch_width(), i % c.patch_width()))
            .collect();
        println!("patch order: cell ({m}, {u}) → token {token}, column {column}");
        assert_eq!(
            non_zero,
            vec![(token, column)],
            "one cell, one (token, column)"
        );
        assert_eq!(token, 2 * 63 + 5, "token = f · ow + t");
        assert_eq!(column, 7 + 16 * 3, "column = kw + patch · kh");
    }

    /// The pinned file's own mel count, so [`CedConfig::token_count`]'s grid
    /// exists — eight rows is less than one patch. Band `m` is bin `m`, so a
    /// tone's band is known here too.
    fn full_front_end() -> (CedConfig, FrontEnd) {
        let config = config(64, 257, 512, 160);
        let mut filterbank = vec![0f32; 64 * 257];
        for m in 0..64 {
            filterbank[m * 257 + m] = 1.0;
        }
        let front =
            FrontEnd::new(&config, filterbank, vec![1f32; 512]).expect("a one-bin band per row");
        (config, front)
    }

    #[test]
    fn a_window_past_a_short_signal_is_padded_with_the_mel_s_own_floor() {
        let (c, front) = full_front_end();
        // A quarter second is 26 mel frames against a 1012-frame window, so the
        // last patch columns read past the signal and take the pad value.
        let samples: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mel = front.log_mel(&samples).expect("a quarter second");
        assert!(
            mel.frames < c.target_length,
            "the mel is not shorter than one window"
        );
        // The pad value is the one the clamp applied: the clamp only raises, so
        // the maximum of the clamped mel is the unclamped one.
        let max = mel.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            mel.floor,
            max - c.a2db_top_db,
            "the floor is `max - top_db`"
        );
        assert!(
            mel.floor < -1.0,
            "a zero pad would pass this test by accident"
        );

        let ones = vec![1f32; c.n_mels];
        let zeros = vec![0f32; c.n_mels];
        let patches = patch_matrix(&mel, 0, &c, &ones, &zeros).expect("one window");
        // The last time token covers frames `(ow-1)·stride ..`, entirely past the
        // signal at 26 frames: every cell of its row is the floor.
        let (ow, oh) = (c.time_tokens(), c.freq_tokens());
        let row = ((oh - 1) * ow + ow - 1) * c.patch_width();
        let last = &patches[row..row + c.patch_width()];
        println!("pad: {} cells at {:.3} dB", last.len(), mel.floor);
        assert!(
            last.iter().all(|v| *v == mel.floor),
            "the padded row starts {:?}, not the floor {}",
            &last[..4],
            mel.floor
        );
    }

    /// A bias must be as wide as the weight's columns, and the two numbers come
    /// from different places — metadata and the tensor — so the check is the only
    /// thing relating them.
    #[test]
    fn a_bias_that_is_not_the_weight_width_is_refused() {
        let bias = |dims: &[usize]| Shape::new(dims, DType::F32);
        assert!(check_bias_width("w", &bias(&[16]), 16).is_ok());
        let err = check_bias_width("w", &bias(&[8]), 16)
            .expect_err("half a bias")
            .to_string();
        assert!(err.contains("bias is 8 wide, the weight has 16"), "{err}");
        // A scalar bias has no width to compare, and the epilogue would add it to
        // one column of many.
        assert!(check_bias_width("w", &bias(&[]), 16).is_err());
    }

    /// A tag whose every frame is its own: one window per hop, no overlap.
    fn tag(start: usize, end: usize, index: usize, score: f32) -> Tag {
        Tag {
            start,
            end,
            index,
            score,
        }
    }

    #[test]
    fn the_range_s_label_is_the_window_that_covers_most_of_it() {
        let tags = vec![
            tag(0, 100, 4, 0.9),
            tag(100, 200, 7, 0.6),
            tag(200, 300, 7, 0.55),
        ];
        // The whole range: label 7 covers 200 frames, label 4 covers 100.
        assert_eq!(collapse(&tags, 527, 0, 300), Some((7, 0.6)));
        // Only the first window overlaps: label 4.
        assert_eq!(collapse(&tags, 527, 10, 90), Some((4, 0.9)));
        // A range no window covers.
        assert_eq!(collapse(&tags, 527, 300, 400), None);
        // Overlap is what is counted, not the window: half of each of the first
        // two is a tie, so the earliest window wins.
        assert_eq!(collapse(&tags, 527, 50, 150), Some((4, 0.9)));
        // An empty list is no label at all.
        assert_eq!(collapse(&[], 527, 0, 100), None);
    }

    #[test]
    fn a_tie_goes_to_the_earliest_window_that_named_the_label() {
        let tags = vec![tag(0, 100, 11, 0.7), tag(100, 200, 3, 0.8)];
        // 100 frames each, so the weights are equal and the first window wins.
        assert_eq!(collapse(&tags, 527, 0, 200), Some((11, 0.7)));
        // The same two, the other way round: the answer follows the order.
        let tags = vec![tag(0, 100, 3, 0.8), tag(100, 200, 11, 0.7)];
        assert_eq!(collapse(&tags, 527, 0, 200), Some((3, 0.8)));
    }

    /// The full metadata block of the pinned file, as a `GgufWriter` — no
    /// tensors, because every refusal tested here is decided by the metadata.
    fn ced_gguf(overrides: &[(&str, MetaValue)], drop: Option<&str>) -> GgufWriter {
        let mut w = GgufWriter::new();
        let mut meta: Vec<(&str, MetaValue)> = vec![
            ("general.architecture", MetaValue::String("ced".into())),
            ("ced.embed_dim", MetaValue::U32(384)),
            ("ced.depth", MetaValue::U32(12)),
            ("ced.num_heads", MetaValue::U32(6)),
            ("ced.mlp_ratio", MetaValue::F32(4.0)),
            ("ced.outputdim", MetaValue::U32(3)),
            ("ced.qkv_bias", MetaValue::Bool(true)),
            ("ced.patch_size", MetaValue::U32(16)),
            ("ced.patch_stride", MetaValue::U32(16)),
            ("ced.target_length", MetaValue::U32(1012)),
            ("ced.pooling", MetaValue::String("mean".into())),
            ("ced.ln_eps_encoder", MetaValue::F32(1e-6)),
            ("ced.ln_eps_head", MetaValue::F32(1e-5)),
            ("ced.bn_eps", MetaValue::F32(1e-5)),
            ("ced.sample_rate", MetaValue::U32(16_000)),
            ("ced.n_mels", MetaValue::U32(64)),
            ("ced.n_fft", MetaValue::U32(512)),
            ("ced.win_size", MetaValue::U32(512)),
            ("ced.hop_size", MetaValue::U32(160)),
            ("ced.n_freqs", MetaValue::U32(257)),
            ("ced.f_min", MetaValue::F32(0.0)),
            ("ced.f_max", MetaValue::F32(8000.0)),
            ("ced.center", MetaValue::Bool(true)),
            ("ced.a2db_multiplier", MetaValue::F32(10.0)),
            ("ced.a2db_amin", MetaValue::F32(1e-10)),
            ("ced.a2db_top_db", MetaValue::F32(120.0)),
            ("ced.a2db_ref", MetaValue::F32(1.0)),
            (
                "ced.labels",
                MetaValue::Array(
                    ["Speech", "Music", "Singing"]
                        .map(|s| MetaValue::String(s.into()))
                        .to_vec(),
                ),
            ),
        ];
        meta.retain(|(k, _)| Some(*k) != drop);
        for (key, value) in meta {
            w.set_meta(key, value);
        }
        for (key, value) in overrides {
            w.set_meta(*key, value.clone());
        }
        w
    }

    /// The refusal a synthetic file produces, or a panic naming it as accepted.
    fn refusal(w: &GgufWriter) -> String {
        let path = std::env::temp_dir().join(format!(
            "mahbot-ced-{}-{:p}.gguf",
            std::process::id(),
            w as *const _
        ));
        w.write_to_path(&path).expect("the synthetic file writes");
        let out = Model::open_ced(&path).err().map(|e| format!("{e:#}"));
        let _ = std::fs::remove_file(&path);
        out.unwrap_or_else(|| panic!("{} was accepted", path.display()))
    }

    #[test]
    fn an_embedding_width_that_is_not_whole_heads_is_refused_by_key() {
        let err = refusal(&ced_gguf(&[("ced.embed_dim", MetaValue::U32(385))], None));
        println!("refusal: {err}");
        assert!(
            err.contains("ced.embed_dim") && err.contains("ced.num_heads"),
            "the refusal names the keys it read: {err}"
        );
    }

    #[test]
    fn a_missing_metadata_key_is_refused_by_name() {
        let err = refusal(&ced_gguf(&[], Some("ced.n_mels")));
        println!("refusal: {err}");
        assert!(err.contains("ced.n_mels"), "{err}");
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn labels_that_do_not_fill_the_head_are_refused() {
        let err = refusal(&ced_gguf(&[("ced.outputdim", MetaValue::U32(4))], None));
        println!("refusal: {err}");
        assert!(
            err.contains("ced.labels") && err.contains("ced.outputdim"),
            "{err}"
        );
    }

    #[test]
    fn a_file_that_is_not_an_audio_tagger_is_refused_by_architecture() {
        let err = refusal(&ced_gguf(
            &[("general.architecture", MetaValue::String("clip".into()))],
            None,
        ));
        println!("refusal: {err}");
        assert!(err.contains("general.architecture"), "{err}");
        assert!(err.contains("ced"), "{err}");
    }
}
