// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Our own GGUF loader — the thing this stand exists for.
//!
//! It loads a body and its `mmproj` under three rules:
//!
//! * **metadata merged with a model config** — the `{arch}.*` keys become a
//!   typed `Config` (layers, width, heads, eps, RoPE, vocabulary, head variant)
//!   and the `clip.*` keys a `ClipConfig` instead of a bag of strings;
//! * **F16 widened on the host** — an F16 tensor is declared F32 on sight;
//! * **packed weights kept raw** — quantized tensors stay the file's own bytes
//!   and carry their own dequant, so the backend reserves scratch for them.
//!
//! Integrity is the bench's own SHA-256 pins; the loader reads files read-only
//! and checks the GGUF header/offset layout the parser hands it.
//!
//! Nothing here writes to the model files: they are opened read-only and
//! mmapped.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{GgmlType, GgufFile, MetaValue, bytes_for_public};
use rlx_ir::quant::QuantScheme;

/// The directory holding the six model files. The path is an input: the
/// checkout carries no models, so there is no crate-relative default.
pub const MODELS_DIR_ENV: &str = "MAHBOT_MODELS_DIR";

/// The recorded 16 kHz voice reference the ear, recognition and synthesis read.
pub const VOICE_WAV_ENV: &str = "MAHBOT_VOICE_WAV";

/// The article the two embedding cells are cut from — a plain-text file, not
/// committed: the checkout carries no corpus, only its pins.
pub const EMBED_TEXT_ENV: &str = "MAHBOT_EMBED_TEXT";

/// The embeddings file: EmbeddingGemma 2's text tower, the one role that is a
/// single file rather than a body and its `mmproj`.
pub const EMBED_MODEL: &str = "embeddinggemma-2-Q8_0.gguf";

/// The artefact the embedding rows are taken with, as `(url, revision,
/// sha256)`: the Q8_0 text tower from `ggml-org/embeddinggemma-2-GGUF`, at the
/// repository revision whose LFS object carries that hash. A run whose own
/// `embeddinggemma-2-Q8_0.gguf` hashes differently did not measure this file,
/// and the record prints that rather than folding the two together.
pub const EMBED_MODEL_SOURCE: (&str, &str, &str) = (
    "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF",
    "58faf90940886df2d62be555c8765d00dac362c5",
    "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
);

/// The model directory from [`MODELS_DIR_ENV`], or the error naming it.
pub fn models_dir() -> Result<PathBuf> {
    env_dir(MODELS_DIR_ENV, "the model directory")
}

/// The embedding cells' article from [`EMBED_TEXT_ENV`], or the error naming it.
pub fn embed_text() -> Result<PathBuf> {
    env_dir(
        EMBED_TEXT_ENV,
        "the plain-text article the embedding cells read",
    )
}

/// The voice reference from [`VOICE_WAV_ENV`], or the error naming it.
pub fn voice_wav() -> Result<PathBuf> {
    match std::env::var_os(VOICE_WAV_ENV) {
        Some(v) if !v.is_empty() => Ok(PathBuf::from(v)),
        _ => Err(anyhow!(
            "{VOICE_WAV_ENV} is not set — point it at the 16 kHz mono voice reference WAV"
        )),
    }
}

fn env_dir(name: &str, what: &str) -> Result<PathBuf> {
    match std::env::var_os(name) {
        Some(v) if !v.is_empty() => Ok(PathBuf::from(v)),
        _ => Err(anyhow!("{name} is not set — point it at {what}")),
    }
}

/// What a tensor is, as far as loading rules are concerned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    /// 2-D matmul weight: the only class `layout: "bt_nk"` is correct for.
    Linear2D,
    /// Embedding table (`token_embd`) — rows are read by index, or the same
    /// bytes serve as a tied head.
    Embedding,
    /// `position_embd` — a table of positions, not a matmul weight.
    PositionTable,
    /// VQ codebook — decoded by index, not by nearest-neighbour search.
    Codebook,
    /// A per-level stack (`a.gen.code.head.weight` / `…embd.weight`, last axis
    /// 15): one slice per depth level, never a matmul weight as a whole.
    Stacked,
    /// 2-D/3-D convolution kernel (`[C_out, C_in, k…]`).
    Conv,
    /// 1-D ConvTranspose kernel (`[C_in, C_out, k]`) — the DAC upsamplers.
    ConvTranspose1D,
    /// 1-D norm weight/bias, biases, Snake `alpha`/`beta`, layer scales.
    Vector,
    /// Anything the rules above do not name.
    Other,
}

/// How a tensor is held before it reaches a graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Store {
    /// The file already stores F32: bytes go through as they are.
    F32File,
    /// F16 in the file: widened to F32 on the host.
    F16Widened,
    /// Quantized, kept as the file's own packed bytes; the graph multiplies
    /// through `Op::DequantMatMul` and the backend reserves scratch for it.
    Packed,
    /// Quantized, unpacked to F32 on the host and multiplied as a plain
    /// `MatMul` — the path the sound stage's non-linear tensors take.
    HostDequant,
    /// Quantized, unpacked to **f16** on the host and multiplied as a plain
    /// `MatMul` whose weight is an f16 `Param` — the embedding encoder's route
    /// when [`Model::f16_linears`] takes it. The file's bytes are read once and
    /// never re-quantised; the packed path stays the default and the fallback.
    F16Linear,
}

/// One tensor, planned.
#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    /// Dimensions as the file gives them (innermost axis first).
    pub ggml_shape: Vec<usize>,
    /// The same tensor in RLX row-major order (innermost axis last).
    pub dims: Vec<usize>,
    pub file: GgmlType,
    pub class: Class,
    pub store: Store,
}

impl Plan {
    /// `(in, out)` of a linear weight — `ne0`, `ne1` of the file.
    pub fn in_out(&self) -> (usize, usize) {
        (self.ggml_shape[0], self.ggml_shape[1])
    }
}

/// Quantized types that travel as packed bytes.
pub fn quant_of(dt: GgmlType) -> Option<QuantScheme> {
    Some(match dt {
        GgmlType::Q4K => QuantScheme::GgufQ4K,
        GgmlType::Q5K => QuantScheme::GgufQ5K,
        GgmlType::Q6K => QuantScheme::GgufQ6K,
        GgmlType::Q8_0 => QuantScheme::GgufQ8_0,
        GgmlType::Q4_0 => QuantScheme::GgufQ4_0,
        GgmlType::Q5_0 => QuantScheme::GgufQ5_0,
        GgmlType::Q8K => QuantScheme::GgufQ8K,
        _ => return None,
    })
}

/// `MAHBOT_EMBED_GEMM=packed` keeps the embedding encoder's multiplies on the
/// packed route — the measurement switch the two arms are taken with, not a
/// tuning knob: the default is the fast path, and the packed route is also what
/// every device without the cooperative f16 kernel gets regardless.
pub const EMBED_GEMM_ENV: &str = "MAHBOT_EMBED_GEMM";

/// Which matmul route an embedding graph's linears took — printed on the cell's
/// own line, so a record states which route its figures price.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gemm {
    /// The file's own packed bytes through `Op::DequantMatMul`. The default for
    /// every other file and for this one on any device that does not admit
    /// [`Gemm::F16Coop`].
    Packed,
    /// The packed weights dequantised on the host into f16 `Param`s and
    /// multiplied by `Op::MatMul`, which the backend serves with
    /// `matmul_coop_f16_f32acc` (f16 operands, f32 accumulator).
    F16Coop,
}

impl Gemm {
    /// The token the cell line carries as `gemm=`.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Gemm::Packed => "packed",
            Gemm::F16Coop => "f16coop",
        }
    }
}

/// Whether the cooperative f16 kernel's tile admits a `[m, k] · [k, n]`
/// multiply, asked of the backend that owns the admission
/// ([`rlx_wgpu::coop_f16_f32acc_admits`]) rather than restated here.
///
/// The extents are the graph's `(m, k, n)` while the backend predicate takes
/// `(m, n, k)`; both are named here so a caller cannot forward them in the wrong
/// order.
///
/// The gate is not cosmetic. An f16 weight whose multiply does not admit this
/// tile is not served more slowly by the same kernel — it falls to
/// `matmul_f16_compute`, which is slower than the packed Q8_0 route it would
/// replace. So a graph that cannot put **every** linear on the cooperative
/// kernel keeps all of them packed.
#[must_use]
pub fn f16_multiply_admits(m: usize, k: usize, n: usize) -> bool {
    rlx_wgpu::coop_f16_f32acc_admits(m as u64, n as u64, k as u64)
}

/// The route rule, as a pure predicate the unit tests can drive without a
/// device: f16 only when the adapter's `f16_f32acc_path_ok()` self-check passed,
/// the route is not pinned off (see [`f16_pinned`]), and every multiply of
/// `shapes` — `(m, k, n)` — admits the kernel.
#[must_use]
pub fn f16_route(pinned: bool, probe_ok: bool, shapes: &[(usize, usize, usize)]) -> bool {
    !pinned
        && probe_ok
        && !shapes.is_empty()
        && shapes.iter().all(|&(m, k, n)| f16_multiply_admits(m, k, n))
}

/// Whether this process pins the encoder off the 16-bit route: the stand's own
/// `MAHBOT_EMBED_GEMM=packed`, or the backend's absolute f32 pin
/// (`RLX_WGPU_MATMUL_F32_ONLY`), under which every multiply computes in f32
/// whatever store the graph declared. The f32 pin has to be honoured here too, or
/// the stand would keep the 6 B-per-2 B f16 residency and a `gemm=f16coop` cell
/// line for a route the backend's arm no longer takes.
#[must_use]
pub fn f16_pinned(forced_packed: bool) -> bool {
    forced_packed || rlx_wgpu::f32_matmul_pinned()
}

/// Which head a body file carries — the three configurations the bodies use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Head {
    /// Own `output.weight` with its own vocabulary.
    Separate { name: &'static str, vocab: usize },
    /// No head of its own: the `token_embd` table is used again.
    Tied { table: &'static str, vocab: usize },
}

/// Merged model configuration — the `{arch}.*` metadata as typed fields.
#[derive(Clone, Debug)]
pub struct Config {
    pub layers: usize,
    pub d: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub vocab: usize,
    pub head: Head,
}

/// Audio-encoder configuration of an `mmproj` (`clip.audio.*`).
#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub dim: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mel_bins: usize,
    pub eps: f32,
    /// Width the sound stage is projected into (`clip.audio.projection_dim`) —
    /// the body's `d` for recognition.
    pub projection_dim: usize,
    /// TTS also carries the generative part (`clip.gen.audio.*`).
    /// `clip.gen.audio.projection_dim` — the talker's width, which is the body's
    /// `d` for a sound file that belongs to a body (checked in [`Pair::open`]).
    pub gen_projection_dim: Option<usize>,
    pub gen_layers: Option<usize>,
    pub gen_heads: Option<usize>,
    pub gen_kv_heads: Option<usize>,
    pub gen_ffn: Option<usize>,
    pub gen_eps: Option<f32>,
}

/// The embeddings file's configuration — EmbeddingGemma 2's **text tower**, as
/// its own `gemma-embedding2.*` metadata states it.
///
/// It is a type of its own rather than a [`Config`]: this file's layers do not
/// share a head width (four global layers at 512, twenty sliding ones at 256),
/// nor a KV-head count (one on a global layer, two on a sliding one), and what
/// the graph needs beyond the block shape — the band, the layer-type pattern,
/// the per-layer-input widths and the output width — has no field in the body
/// config at all. Squeezing it into `Config` would mean six defaults that are
/// wrong for every Qwen file.
///
/// Every field is read, none is derived from a nominal shape: a file whose
/// arrays disagree with its tensors is refused in [`Model::open`] rather than
/// read as if its metadata were a comment.
#[derive(Clone, Debug)]
pub struct GemmaConfig {
    pub layers: usize,
    pub d: usize,
    pub ffn: usize,
    pub heads: usize,
    /// `attention.key_length` — the head width of the four **global** layers.
    pub head_dim: usize,
    /// `attention.key_length_swa` — the head width of the twenty **sliding**
    /// layers.
    pub head_dim_swa: usize,
    /// `attention.head_count_kv`, one entry per layer.
    pub kv_heads: Vec<usize>,
    /// `attention.sliding_window_pattern`: `true` marks a sliding layer,
    /// `false` a global one.
    pub sliding: Vec<bool>,
    /// `attention.sliding_window` as the file states it — the **full** width.
    /// llama.cpp's `LLAMA_SWA_TYPE_SYMMETRIC` halves it into a radius, so the
    /// band a query sees is `|qi - ki| <= window / 2`, not `window`.
    pub window: usize,
    pub eps: f32,
    /// `rope.freq_base` of the global layers.
    pub rope_base: f32,
    /// `rope.freq_base_swa` of the sliding ones.
    pub rope_base_swa: f32,
    /// `embedding_length_per_layer_input` — the width of one layer's slice of
    /// the per-layer-input projection (`512` here, the block width).
    pub ple: usize,
    /// `embedding_length_out` — the width of the final projection (`768`).
    pub out_width: usize,
    pub vocab: usize,
}

impl GemmaConfig {
    /// `true` when layer `il` is one of the twenty sliding ones.
    pub fn is_sliding(&self, il: usize) -> bool {
        self.sliding[il]
    }

    /// The head width layer `il`'s Q/K/V and norms are built at.
    pub fn head_dim_of(&self, il: usize) -> usize {
        if self.is_sliding(il) {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    /// The KV-head count of layer `il`.
    pub fn kv_heads_of(&self, il: usize) -> usize {
        self.kv_heads[il]
    }

    /// The RoPE base layer `il` rotates at.
    pub fn rope_base_of(&self, il: usize) -> f32 {
        if self.is_sliding(il) {
            self.rope_base_swa
        } else {
            self.rope_base
        }
    }

    /// The band's radius: half the file's `sliding_window` — see [`Self::window`].
    pub fn window_radius(&self) -> usize {
        self.window / 2
    }
}

/// The CED audio tagger's configuration — the `ced.*` metadata as typed fields.
///
/// It is a type of its own for the reason [`GemmaConfig`] is: the file's own
/// front end (a log-mel whose filterbank and window it carries as tensors, and
/// whose amplitude-to-dB clamp is three numbers) has no field in a body config,
/// and the ViT over its patches has neither a KV-head count nor a vocabulary.
///
/// Every field is read from the file by [`Model::open_ced`], and the arrays are
/// checked against the tensors there: a file whose metadata and tensors
/// disagree is refused by name rather than read as if the metadata were a
/// comment. The front end's two tensors are given out by
/// [`Model::ced_front_end`], not recomputed here — the converter baked them.
#[derive(Clone, Debug)]
pub struct CedConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_ratio: f32,
    pub output_dim: usize,
    pub qkv_bias: bool,
    pub patch_size: usize,
    pub patch_stride: usize,
    /// `ced.target_length` — the mel frames one window holds.
    pub target_length: usize,
    pub pooling: String,
    pub ln_eps_encoder: f32,
    pub ln_eps_head: f32,
    pub bn_eps: f32,
    /// `ced.sample_rate` — the rate the signal must arrive at. Recorded rather
    /// than used: the file's filterbank already fixes the bins.
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_size: usize,
    pub hop_size: usize,
    pub n_freqs: usize,
    /// `ced.f_min`/`ced.f_max` — the band the file's own filterbank was built
    /// over. Recorded rather than used: the filters arrive as a tensor, so
    /// recomputing them from these two numbers would be a second front end.
    pub f_min: f32,
    pub f_max: f32,
    pub center: bool,
    /// `ced.a2db_multiplier` — 10 for a power spectrogram.
    pub a2db_multiplier: f32,
    pub a2db_amin: f32,
    pub a2db_top_db: f32,
    pub a2db_ref: f32,
    /// The AudioSet ontology, one name per output column.
    pub labels: Vec<String>,
}

/// The token grid the tagger's graph reshapes to, stated once: the shape check
/// against the file's tensors and the graph that computes them read these, so a
/// file the check blesses cannot be one the graph mis-reshapes.
impl CedConfig {
    /// Mel columns one window's patches span.
    pub fn time_tokens(&self) -> usize {
        (self.target_length - self.patch_size) / self.patch_stride + 1
    }

    /// Mel rows one window's patches span — `4` on this file.
    pub fn freq_tokens(&self) -> usize {
        (self.n_mels - self.patch_size) / self.patch_stride + 1
    }

    /// The tokens one window makes — `252` on this file.
    pub fn token_count(&self) -> usize {
        self.time_tokens() * self.freq_tokens()
    }

    /// The columns one token holds: one patch, `patch_size²`.
    pub fn patch_width(&self) -> usize {
        self.patch_size * self.patch_size
    }

    /// The MLP's hidden width, as the file's own ratio computes it.
    pub fn ffn(&self) -> usize {
        (self.embed_dim as f32 * self.mlp_ratio) as usize
    }
}

/// Which of the three roles a file belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Recognition,
    Synthesis,
    Embeddings,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Recognition => "recognition",
            Role::Synthesis => "synthesis",
            Role::Embeddings => "embeddings",
        }
    }
}

/// A loaded file: mmapped bytes + everything the loader decided about it.
pub struct Model {
    pub path: PathBuf,
    pub file: GgufFile,
    /// The role a pair names this file for — `None` for [`Model::open_ced`]'s
    /// tagger, which no pair holds and which therefore has no role to claim.
    pub role: Option<Role>,
    /// `true` for the `mmproj` (sound) files. What that means for the weights is
    /// the rule below: rank-2 weights keep the file's own quantised bytes, the
    /// non-linear sound tensors go to the host as f32.
    pub sound: bool,
    pub config: Option<Config>,
    pub clip: Option<ClipConfig>,
    /// The embeddings file's own config — set only by `gemma-embedding2`.
    pub gemma: Option<GemmaConfig>,
    /// The audio tagger's own config — set only by [`Model::open_ced`].
    pub ced: Option<CedConfig>,
    pub plans: Vec<Plan>,
    by_name: HashMap<String, usize>,
}

fn meta_u32(file: &GgufFile, key: &str) -> Option<u32> {
    file.metadata.get(key).and_then(MetaValue::as_u32)
}

/// The `MetaValue` variant's name, for a refusal that has to say what a key
/// actually held rather than what it should have held.
fn meta_type(value: &MetaValue) -> &'static str {
    match value {
        MetaValue::U8(_) => "u8",
        MetaValue::I8(_) => "i8",
        MetaValue::U16(_) => "u16",
        MetaValue::I16(_) => "i16",
        MetaValue::U32(_) => "u32",
        MetaValue::I32(_) => "i32",
        MetaValue::U64(_) => "u64",
        MetaValue::I64(_) => "i64",
        MetaValue::F32(_) => "f32",
        MetaValue::F64(_) => "f64",
        MetaValue::Bool(_) => "bool",
        MetaValue::String(_) => "string",
        MetaValue::Array(_) => "array",
    }
}

fn meta_f32(file: &GgufFile, key: &str) -> Option<f32> {
    match file.metadata.get(key) {
        Some(MetaValue::F32(v)) => Some(*v),
        Some(MetaValue::F64(v)) => Some(*v as f32),
        _ => None,
    }
}

fn meta_str<'a>(file: &'a GgufFile, key: &str) -> Option<&'a str> {
    file.metadata.get(key).and_then(MetaValue::as_str)
}

/// Classify a tensor by name and rank — the rule table of this loader.
///
/// The rank decides most of it (only a 2-D tensor can be a matmul weight), and
/// names only separate the cases the format does not: the embedding/position
/// tables, the codebooks, the stacked per-level heads, and the DAC's transposed
/// convolutions (a 1-D ConvTranspose is stored `[C_in, C_out, k]` — the opposite
/// channel order to a plain conv, and it must not be treated as one).
pub fn classify(name: &str, rank: usize) -> Class {
    if name.ends_with(".bias") || name.ends_with(".alpha") || name.ends_with(".beta") {
        return Class::Vector;
    }
    if name.ends_with("token_embd.weight") || name.ends_with("out_embd.weight") {
        return Class::Embedding;
    }
    if name.ends_with("position_embd.weight") {
        return Class::PositionTable;
    }
    if name.contains("codebook") {
        return Class::Codebook;
    }
    if name == "a.gen.code.head.weight" || name == "a.gen.code.embd.weight" {
        return Class::Stacked;
    }
    if rank == 3
        && name.ends_with(".conv.weight")
        && (name.starts_with("a.gen.wav.up.blk") || name.starts_with("a.gen.wav.dac.blk"))
    {
        return Class::ConvTranspose1D;
    }
    match rank {
        2 => Class::Linear2D,
        1 => Class::Vector,
        3 | 4 => Class::Conv,
        _ => Class::Other,
    }
}

impl Model {
    /// The code predictor's entry: `(talker width, predictor width)`. The 1.7B
    /// sound file carries `a.gen.code.proj_in.weight`, which takes the talker's
    /// row (2048) down into the code block's width (1024); the 0.6B file has no
    /// such weight at all — its talker is already the code block's width — so
    /// both readings fall back on the code block's own width.
    pub fn code_proj_widths(&self) -> Result<(usize, usize)> {
        if let Ok(p) = self.plan("a.gen.code.proj_in.weight") {
            return Ok(p.in_out());
        }
        let d = self.plan("a.gen.code.blk.0.ln1.weight")?.ggml_shape[0];
        Ok((d, d))
    }

    /// `true` when this sound file's talker has to be projected into the code
    /// predictor's width, i.e. when it carries `a.gen.code.proj_in.weight`.
    pub fn has_code_proj_in(&self) -> bool {
        self.plan("a.gen.code.proj_in.weight").is_ok()
    }

    /// Open a file read-only (mmap) and plan every tensor in it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = GgufFile::from_path_mmap(&path)
            .with_context(|| format!("open gguf {}", path.display()))?;

        let arch = meta_str(&file, "general.architecture")
            .unwrap_or("")
            .to_string();
        let sound = arch == "clip";
        let role = if sound {
            let generative = meta_u32(&file, "clip.gen.audio.block_count").is_some();
            if generative {
                Role::Synthesis
            } else {
                Role::Recognition
            }
        } else {
            match arch.as_str() {
                "qwen3vl" => Role::Recognition,
                "qwen3tts" => Role::Synthesis,
                "gemma-embedding2" => Role::Embeddings,
                other => bail!("unknown architecture `{other}` in {}", path.display()),
            }
        };

        let (config, gemma) = if sound {
            (None, None)
        } else if arch == "gemma-embedding2" {
            (None, Some(merge_gemma(&file, &arch)?))
        } else {
            (Some(merge_config(&file, &arch)?), None)
        };
        let clip = if sound { Some(merge_clip(&file)) } else { None };

        // ── the inventory: one plan per tensor, rules applied up front ──
        // Rank-2 quantised weights keep the file's own bytes, packed, for every
        // role. The sound file's non-linear tensors go to the host as f32; no
        // file is re-quantised in the stand.
        let mut plans: Vec<Plan> = Vec::with_capacity(file.tensors.len());
        for t in file.tensors.values() {
            let rank = t.shape.len();
            let class = classify(&t.name, rank);
            let dims: Vec<usize> = t.shape.iter().rev().copied().collect();
            let store = match quant_of(t.dtype) {
                Some(_) => {
                    if sound && rank != 2 {
                        Store::HostDequant
                    } else {
                        Store::Packed
                    }
                }
                None => match t.dtype {
                    GgmlType::F32 => Store::F32File,
                    // F16 is never declared as F16: it is widened on the host.
                    GgmlType::F16 | GgmlType::BF16 => Store::F16Widened,
                    other => bail!("tensor {} has unhandled type {}", t.name, other.name()),
                },
            };
            plans.push(Plan {
                name: t.name.clone(),
                ggml_shape: t.shape.clone(),
                dims,
                file: t.dtype,
                class,
                store,
            });
        }
        plans.sort_by(|a, b| a.name.cmp(&b.name));
        // The per-layer arrays are checked against the tensors themselves: a
        // file whose `head_count_kv` or `sliding_window_pattern` disagrees with
        // what its own projections are wide is refused here, not read as if the
        // arrays were a comment.
        if let Some(g) = &gemma {
            check_gemma_tensors(&file, g)?;
        }
        let by_name = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();

        Ok(Self {
            path,
            file,
            role: Some(role),
            sound,
            config,
            clip,
            gemma,
            ced: None,
            plans,
            by_name,
        })
    }

    /// Open a **CED** audio tagger (a `ced` architecture file) read-only and plan
    /// every tensor in it under this file's own store rule.
    ///
    /// It is a separate constructor rather than a fourth [`Role`]: a tagger is
    /// not a body, not an `mmproj` and not a text tower, it is never paired with
    /// another file, and the one rule that differs — the store — must not reach
    /// the three files [`Model::open`] serves. Its [`Model::role`] is therefore
    /// `None`.
    ///
    /// The store rule here is the tagger's own: a **rank-2 F16** tensor is kept
    /// as f16 (`Store::F16Linear`), everything F32 stays as the file's bytes, and
    /// any other F16 goes to the host as f32. This file is f16 end to end — its
    /// 49 weight matrices are `F16` in the file — which is why the rule exists at
    /// all: under [`Model::open`]'s rule every one of them would be widened.
    /// Which bytes a multiply then reads — the f16 halves or their f32 image —
    /// is the backend's decision per shape; [`crate::ced`] records which of the
    /// two this tagger's geometry lands on.
    pub fn open_ced(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = GgufFile::from_path_mmap(&path)
            .with_context(|| format!("open gguf {}", path.display()))?;
        let arch = meta_str(&file, "general.architecture").unwrap_or("");
        if arch != "ced" {
            bail!(
                "{} is not an audio tagger: general.architecture is `{arch}`, not `ced`",
                path.display()
            );
        }
        let ced = merge_ced(&file, arch)?;
        let mut plans: Vec<Plan> = Vec::with_capacity(file.tensors.len());
        for t in file.tensors.values() {
            let rank = t.shape.len();
            let class = classify(&t.name, rank);
            let dims: Vec<usize> = t.shape.iter().rev().copied().collect();
            let store = match quant_of(t.dtype) {
                Some(_) => Store::Packed,
                None => match t.dtype {
                    GgmlType::F16 if class == Class::Linear2D => Store::F16Linear,
                    GgmlType::F32 => Store::F32File,
                    GgmlType::F16 | GgmlType::BF16 => Store::F16Widened,
                    other => bail!("tensor {} has unhandled type {}", t.name, other.name()),
                },
            };
            plans.push(Plan {
                name: t.name.clone(),
                ggml_shape: t.shape.clone(),
                dims,
                file: t.dtype,
                class,
                store,
            });
        }
        plans.sort_by(|a, b| a.name.cmp(&b.name));
        check_ced_tensors(&file, &ced)?;
        let by_name = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();
        Ok(Self {
            path,
            file,
            role: None,
            sound: false,
            config: None,
            clip: None,
            gemma: None,
            ced: Some(ced),
            plans,
            by_name,
        })
    }

    /// The tagger's front end as the file bakes it: `(filterbank, window)`.
    ///
    /// The filterbank is mel-major — `n_mels` rows of `n_freqs` — and the window
    /// is the periodic Hann of `win_size` points. Neither is computed here: a
    /// front end that recomputed them would be a second definition of the
    /// model's own, free to drift from the one the weights were trained against.
    pub fn ced_front_end(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let filterbank = self.f32_file_order("ced.mel_filterbank")?.0;
        let window = self.f32_file_order("ced.mel_window")?.0;
        Ok((filterbank, window))
    }

    pub fn plan(&self, name: &str) -> Result<&Plan> {
        self.by_name
            .get(name)
            .map(|&i| &self.plans[i])
            .ok_or_else(|| {
                anyhow!(
                    "tensor `{name}` is not planned (not in {})",
                    self.path.display()
                )
            })
    }

    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// The store rule for an embedding encoder's linears: hand this file's
    /// packed rank-2 weights over as f16 params when — and only when — the
    /// adapter's cooperative f16 kernel is admitted for **every** multiply the
    /// graph will state at `rows` rows. Returns the route the graph then takes,
    /// which is what its cell line prints as `gemm=`.
    ///
    /// The gate has two halves, and both are needed:
    ///
    /// * `rlx_wgpu::coop_probe::f16_f32acc_path_ok()` — a per-device self-check
    ///   of `matmul_coop_f16_f32acc`'s operand order. Without it an f16 weight
    ///   is not routed to a slower variant of the same kernel, it falls to
    ///   `matmul_f16_compute`, measured at 0.098 TFLOP/s against the packed
    ///   route's measured 1.4 — an order worse than the path it would replace.
    ///   A device that cannot pass the probe keeps the packed bytes, which is
    ///   also the portable default.
    /// * every linear admitting the kernel's 64×64 tile
    ///   ([`f16_multiply_admits`]) — the encoder mixes `m` (the row count, so
    ///   the two cells differ), `k` and `n` per projection, and one multiply
    ///   that does not admit it would silently put that projection back on the
    ///   f16 fallback while claiming the fast route on the cell's line.
    ///
    /// And two pins stand the route down entirely ([`f16_pinned`]): the stand's
    /// `MAHBOT_EMBED_GEMM=packed`, and the backend's `RLX_WGPU_MATMUL_F32_ONLY`,
    /// under which the arm computes in f32 whatever store was declared — keeping
    /// the f16 residency (or printing `gemm=f16coop`) under that pin would cost
    /// memory and claim a route nothing runs.
    ///
    /// Only `Class::Linear2D` plans are flipped: the `token_embd` table shares
    /// the file's packed encoding but is read row by row on the host, and
    /// widening it to f16 would pay for 134 M weights no multiply reads.
    ///
    /// # What it costs, and what it wins
    ///
    /// An f16-declared param costs **6 B on the device** for every 2 B of
    /// weight: the f32-uniform arena keeps a 4 B-per-element slot that
    /// `set_param_typed` widens into, and the kernel reads a 2 B f16 mirror of
    /// the same slot. The 130 M weights this file multiplies therefore move from
    /// ~138 MiB packed to ~780 MB. Measured on this record's M3 Pro, the two
    /// arms taken back to back (`RLX_WGPU_DBG_COUNTERS=1` and the cell lines are
    /// the record): at 8 192 rows the arena is 980.8 MiB packed against 1.31 GiB
    /// f16, and the cell's own `peak` 0.74 GiB against 1.75–1.77 GiB; at 256 rows
    /// the cooperative cell's arena is 547.9 MiB and its `peak` 1.63 GiB.
    ///
    /// The win is in the multiplies, and this doc does not apportion the pass
    /// between them and the attention blocks — those have their own cost tests.
    /// At 8 192 rows the cooperative arm reads **~2.31 s** warm against the
    /// portable packed arm's **~3.45 s**, so the route is ~1.49× at this length;
    /// at 256 rows the cooperative cell reads **~68 ms** warm. Those are the
    /// record's two rows, quoted to the precision that does not move between
    /// takes (the warm spread is a few milliseconds); the record carries the
    /// figures and the runs that took them. The
    /// cooperative kernel's measured rate is 3.07–3.66 TFLOP/s on the encoder's
    /// six linear/FFN shapes (`tests/encoder_gemm_cost.rs`) and 3.42–3.58
    /// TFLOP/s on the dense score squares
    /// (`tests/attention_dense_coop_cost.rs`). What is left of the
    /// 8 192-row pass is the twenty sliding layers' banded walk and the four
    /// global layers' dense score squares; each call has its own cost pinned in
    /// those tests, and summing them here would over-count the floors the fused
    /// pass pays once.
    ///
    /// The win is **not portable** — it exists only where this device's
    /// cooperative f16 kernel does, which is why the probe gates it. What *is*
    /// portable is the route's shape (a plain `Op::MatMul` over an f16 param)
    /// and its arithmetic (f16 operands, f32 accumulator, the reference engine's
    /// own arrangement); a device without the kernel gets the packed route, the
    /// packed route's numbers and the packed route's memory.
    pub fn f16_linears(&mut self, rows: usize) -> Gemm {
        // Re-decided from scratch: a second call for another row count must not
        // inherit the first call's answer (the extents, not the file, decide).
        for p in &mut self.plans {
            if p.store == Store::F16Linear {
                p.store = Store::Packed;
            }
        }
        let shapes: Vec<(usize, usize, usize)> = self
            .plans
            .iter()
            .filter(|p| p.store == Store::Packed && p.class == Class::Linear2D)
            .map(|p| {
                let (k, n) = p.in_out();
                (rows, k, n)
            })
            .collect();
        let forced_packed = std::env::var_os(EMBED_GEMM_ENV)
            .is_some_and(|v| v.to_string_lossy().trim().eq_ignore_ascii_case("packed"));
        let pinned = f16_pinned(forced_packed);
        // The probe is a device round trip (it builds and runs the kernel), so a
        // caller who has already pinned the route off does not pay for it.
        let probe_ok = !pinned && rlx_wgpu::coop_probe::f16_f32acc_path_ok();
        if !f16_route(pinned, probe_ok, &shapes) {
            return Gemm::Packed;
        }
        for p in &mut self.plans {
            if p.store == Store::Packed && p.class == Class::Linear2D {
                p.store = Store::F16Linear;
            }
        }
        Gemm::F16Coop
    }

    /// The file's sampler, as the file states it: `(top_k, top_p, temperature)`.
    pub fn sampling(&self) -> (usize, f32, f32) {
        let top_k = meta_u32(&self.file, "general.sampling.top_k").unwrap_or(50) as usize;
        let top_p = meta_f32(&self.file, "general.sampling.top_p").unwrap_or(1.0);
        let temp = meta_f32(&self.file, "general.sampling.temp").unwrap_or(0.9);
        (top_k, top_p, temp)
    }

    pub fn config(&self) -> &Config {
        self.config.as_ref().expect("body file has a config")
    }

    pub fn clip(&self) -> &ClipConfig {
        self.clip.as_ref().expect("mmproj file has a clip config")
    }

    pub fn gemma(&self) -> &GemmaConfig {
        self.gemma.as_ref().expect("gemma file has a gemma config")
    }

    // ── bytes ───────────────────────────────────────────────────────

    /// The tensor's size in bytes, without reading it.
    pub fn raw_len(&self, name: &str) -> Result<usize> {
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` is not in {}", self.path.display()))?;
        bytes_for_public(t.dtype, t.n_elements())
            .ok_or_else(|| anyhow!("`{name}`: {} elements are not whole blocks", t.n_elements()))
    }

    /// The tensor's own bytes into `buf` with `pread`, so the checkpoint's pages never enter
    /// the process's resident set behind the copy the device holds
    /// ([`GgufFile::read_tensor_bytes_into`]). `buf` is grown and overwritten rather than
    /// cleared, so one scratch reused across tensors pays at most the largest of them; a
    /// file that cannot be read that way falls back to the mapping, which does clear it.
    pub fn read_into(&self, name: &str, buf: &mut Vec<u8>) -> Result<()> {
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` is not in {}", self.path.display()))?;
        if !self.file.read_tensor_bytes_into(t, buf)? {
            buf.clear();
            buf.extend_from_slice(self.file.tensor_bytes(t)?);
        }
        Ok(())
    }

    pub fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.read_into(name, &mut bytes)?;
        Ok(bytes)
    }

    /// The tensor as F32 values in the file's own flat order (which is also the
    /// RLX row-major order of `plan.dims` — GGUF stores the innermost axis
    /// first, so reversing the dims keeps the element order).
    pub fn f32_file_order(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let p = self.plan(name)?;
        let bytes = self.raw(name)?;
        let vals = rlx_gguf::dequant_typed(p.file, &bytes, p.ggml_shape.iter().product(), name)?;
        if vals.len() != p.dims.iter().product::<usize>() {
            bail!(
                "`{name}`: dequantized {} values, plan expects {}",
                vals.len(),
                p.dims.iter().product::<usize>()
            );
        }
        Ok((vals, p.dims.clone()))
    }

    /// Selected rows of a stacked table, dequantised a row at a time: `picks`
    /// are `(level, row)` pairs, `width` is the row's element count. The file's
    /// bytes are read once and only the picked rows leave it — a caller that
    /// needs one row of each level does not pay for the whole f32 table.
    pub fn table_rows_f32(
        &self,
        name: &str,
        picks: &[(usize, usize)],
        width: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let p = self.plan(name)?;
        // Borrow the tensor's slice of the mapping and touch only the rows asked
        // for. `raw` copies the whole tensor with `pread` on every call, which for
        // `token_embd` is hundreds of megabytes per lookup.
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` is not in {}", self.path.display()))?;
        let bytes = self.file.tensor_bytes(t)?;
        let row_bytes = bytes_for_public(p.file, width)
            .ok_or_else(|| anyhow!("`{name}`: no byte size for {width} elements"))?;
        let stride = p.ggml_shape[1];
        picks
            .iter()
            .map(|&(level, row)| {
                if row >= stride {
                    bail!("`{name}`: row {row} past the level's {stride} rows");
                }
                let start = (level * stride + row) * row_bytes;
                let end = start + row_bytes;
                if end > bytes.len() {
                    bail!("`{name}`: row {level}/{row} past {} bytes", bytes.len());
                }
                rlx_gguf::dequant_typed(p.file, &bytes[start..end], width, name)
            })
            .collect()
    }

    /// A 2-D linear weight in the layout `Op::MatMul` wants: `[k, n]` row-major
    /// with `n` fastest. The file holds it as `[k, n]` with `k` fastest, so this
    /// is the one place a materialized weight is transposed.
    pub fn linear_f32(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        let p = self.plan(name)?;
        if p.ggml_shape.len() != 2 {
            bail!("`{name}` is not 2-D (rank {})", p.ggml_shape.len());
        }
        let (k, n) = p.in_out();
        let (vals, _) = self.f32_file_order(name)?;
        let mut out = vec![0f32; k * n];
        for j in 0..n {
            for i in 0..k {
                out[i * n + j] = vals[j * k + i];
            }
        }
        Ok((out, k, n))
    }

    /// The same weight as **f16** bytes, in the same layout — what
    /// [`Store::F16Linear`] hands the session through
    /// `set_param_typed(name, bytes, DType::F16)`.
    ///
    /// The dequant is [`linear_f32`](Self::linear_f32)'s, the one the loaders
    /// already share (`rlx_gguf`'s per-scheme decode), and the narrowing is one
    /// round per element: f16 carries 11 bits of mantissa against Q8_0's 8 and
    /// a per-value exponent against Q8_0's per-32 block scale, so this is a
    /// widening of the file's own precision, not a second quantisation. Peak
    /// host cost is one weight's f32 image plus its f16 image — 4 B + 2 B per
    /// element, the largest here being a 512×2048 projection.
    pub fn linear_f16(&self, name: &str) -> Result<(Vec<u8>, usize, usize)> {
        let (vals, k, n) = self.linear_f32(name)?;
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for v in &vals {
            bytes.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes());
        }
        Ok((bytes, k, n))
    }
}

/// Merge the `{arch}.*` metadata into a typed config.
fn merge_config(file: &GgufFile, arch: &str) -> Result<Config> {
    let need = |suffix: &str| -> Result<String> {
        let key = format!("{arch}.{suffix}");
        if !file.metadata.contains_key(&key) {
            bail!("metadata key `{key}` is missing");
        }
        Ok(key)
    };
    let layers = meta_u32(file, &need("block_count")?).unwrap_or(0) as usize;
    let d = meta_u32(file, &need("embedding_length")?).unwrap_or(0) as usize;
    let ffn = meta_u32(file, &need("feed_forward_length")?).unwrap_or(0) as usize;
    let heads = meta_u32(file, &need("attention.head_count")?).unwrap_or(0) as usize;
    let kv_heads = meta_u32(file, &need("attention.head_count_kv")?).unwrap_or(0) as usize;
    let eps = meta_f32(file, &need("attention.layer_norm_rms_epsilon")?).unwrap_or(1e-6);
    let rope_base = meta_f32(file, &need("rope.freq_base")?).unwrap_or(10_000.0);
    // Head dimension: the explicit key when present, else derived from heads.
    let head_dim = meta_u32(file, &format!("{arch}.attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| d / heads.max(1));
    file.metadata
        .contains_key(&format!("{arch}.attention.key_length"));

    // ── the head: three configurations, decided by what the file carries ──
    let head = if file.get("output.weight").is_some() {
        // The head is stored like every other linear: `ne0` is the input width
        // (the model's `d`), `ne1` the vocabulary.
        let vocab = file
            .get("output.weight")
            .map(|t| t.shape[1])
            .ok_or_else(|| anyhow!("output.weight without a shape"))?;
        Head::Separate {
            name: "output.weight",
            vocab,
        }
    } else {
        let table = file
            .get("token_embd.weight")
            .ok_or_else(|| anyhow!("no output.weight and no token_embd.weight — no head at all"))?;
        // The table is `[d, vocab]` in the file; the vocabulary is its last axis.
        Head::Tied {
            table: "token_embd.weight",
            vocab: table.shape[1],
        }
    };
    let vocab = match head {
        Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
    };

    // Vocabulary cross-check: the file must also say so, when it says anything.
    let vocab_key = format!("{arch}.vocab_size");
    if let Some(v) = meta_u32(file, &vocab_key) {
        if v as usize != vocab {
            bail!("`{arch}.vocab_size` = {v} but the head has {vocab} rows");
        }
    }

    Ok(Config {
        layers,
        d,
        ffn,
        heads,
        kv_heads,
        head_dim,
        eps,
        rope_base,
        vocab,
        head,
    })
}

/// Merge the `gemma-embedding2.*` metadata of the embeddings file into a typed
/// config, then check it against the file's own tensors.
///
/// The arrays are the reason this config exists: `attention.head_count_kv` and
/// `attention.sliding_window_pattern` are per-layer in this file, and so is the
/// head width (`key_length` for the global layers, `key_length_swa` for the
/// sliding ones). Both arrays must be exactly as long as the block count, and
/// `key_length`/`value_length` must agree with each other on both widths — the
/// graph builds K and V at one width per layer, so a file that disagreed could
/// only be served by silently reading one of the two.
fn merge_gemma(file: &GgufFile, arch: &str) -> Result<GemmaConfig> {
    let need = |suffix: &str| -> Result<String> {
        let key = format!("{arch}.{suffix}");
        if !file.metadata.contains_key(&key) {
            bail!("metadata key `{key}` is missing");
        }
        Ok(key)
    };
    let u32_of = |suffix: &str| -> Result<usize> {
        let key = need(suffix)?;
        let value = file.metadata.get(&key).expect("key just checked");
        value.as_u32().map(|v| v as usize).ok_or_else(|| {
            anyhow!(
                "metadata key `{key}` is {}, expected a u32",
                meta_type(value)
            )
        })
    };
    let f32_of = |suffix: &str| -> Result<f32> {
        let key = need(suffix)?;
        let value = file.metadata.get(&key).expect("key just checked");
        match value {
            MetaValue::F32(v) => Ok(*v),
            MetaValue::F64(v) => Ok(*v as f32),
            other => bail!(
                "metadata key `{key}` is {}, expected an f32",
                meta_type(other)
            ),
        }
    };

    let layers = u32_of("block_count")?;
    if layers == 0 {
        bail!("`{arch}.block_count` is 0 — nothing to build");
    }
    // The per-layer arrays. `as_first_u32` is not what this wants: a file that
    // carried a single scalar here would silently claim every layer is layer 0.
    let array_of_u32 = |key: &str| -> Result<Vec<usize>> {
        let value = file
            .metadata
            .get(key)
            .ok_or_else(|| anyhow!("metadata key `{key}` is missing"))?;
        let items = value.as_array().ok_or_else(|| {
            anyhow!(
                "metadata key `{key}` is {}, expected an array",
                meta_type(value)
            )
        })?;
        items
            .iter()
            .map(|v| {
                v.as_u32().map(|v| v as usize).ok_or_else(|| {
                    anyhow!("`{key}` holds a {} entry, expected a u32", meta_type(v))
                })
            })
            .collect()
    };
    let kv_key = format!("{arch}.attention.head_count_kv");
    let kv_heads = array_of_u32(&kv_key)?;
    let pattern_key = format!("{arch}.attention.sliding_window_pattern");
    let pattern_value = file
        .metadata
        .get(&pattern_key)
        .ok_or_else(|| anyhow!("metadata key `{pattern_key}` is missing"))?;
    let pattern_items = pattern_value.as_array().ok_or_else(|| {
        anyhow!(
            "metadata key `{pattern_key}` is {}, expected an array",
            meta_type(pattern_value)
        )
    })?;
    let sliding: Vec<bool> = pattern_items
        .iter()
        .map(|v| match v {
            MetaValue::Bool(b) => Ok(*b),
            other => Err(anyhow!(
                "`{pattern_key}` holds a {} entry, expected a bool",
                meta_type(other)
            )),
        })
        .collect::<Result<_>>()?;
    if kv_heads.len() != layers {
        bail!(
            "`{kv_key}` has {} entries, `{arch}.block_count` is {layers}",
            kv_heads.len()
        );
    }
    if sliding.len() != layers {
        bail!(
            "`{pattern_key}` has {} entries, `{arch}.block_count` is {layers}",
            sliding.len()
        );
    }

    // One width per layer type, and K's width is V's: the graph reads both out
    // of one reshape, so two widths would have to be silently dropped.
    let head_dim = u32_of("attention.key_length")?;
    let head_dim_swa = u32_of("attention.key_length_swa")?;
    for (name, want) in [
        ("attention.value_length", head_dim),
        ("attention.value_length_swa", head_dim_swa),
        ("rope.dimension_count", head_dim),
        ("rope.dimension_count_swa", head_dim_swa),
    ] {
        let got = u32_of(name)?;
        if got != want {
            bail!("`{arch}.{name}` is {got}, expected {want} (the head width it pairs with)");
        }
    }
    if head_dim == 0 || head_dim_swa == 0 || !head_dim.is_multiple_of(2) {
        bail!("head widths {head_dim} / {head_dim_swa} cannot be rotated pairwise");
    }
    let causal_key = format!("{arch}.attention.causal");
    let causal = match file.metadata.get(&causal_key) {
        Some(MetaValue::Bool(v)) => *v,
        Some(other) => bail!(
            "metadata key `{causal_key}` is {}, expected a bool",
            meta_type(other)
        ),
        None => false,
    };
    if causal {
        bail!("`{causal_key}` is set — the embedding pass is bidirectional");
    }
    let pooling = u32_of("pooling_type")?;
    if pooling != 1 {
        bail!("`{arch}.pooling_type` is {pooling} — this loader serves MEAN (1) only");
    }
    let window = u32_of("attention.sliding_window")?;
    if sliding.iter().any(|s| *s) && window < 2 {
        bail!("`{arch}.attention.sliding_window` is {window} — a sliding layer needs a band");
    }

    let d = u32_of("embedding_length")?;
    let eps = f32_of("attention.layer_norm_rms_epsilon")?;
    if !(eps.is_finite() && eps > 0.0) {
        bail!(
            "`{arch}.attention.layer_norm_rms_epsilon` is {eps} — an RMSNorm needs a positive \
             finite epsilon, and a zero would divide by √mean with no floor"
        );
    }
    let vocab = file
        .get("token_embd.weight")
        .map(|t| t.shape[1])
        .ok_or_else(|| anyhow!("no `token_embd.weight` — no table to look rows up in"))?;
    Ok(GemmaConfig {
        layers,
        d,
        ffn: u32_of("feed_forward_length")?,
        heads: u32_of("attention.head_count")?,
        head_dim,
        head_dim_swa,
        kv_heads,
        sliding,
        window,
        eps,
        rope_base: f32_of("rope.freq_base")?,
        rope_base_swa: f32_of("rope.freq_base_swa")?,
        ple: u32_of("embedding_length_per_layer_input")?,
        out_width: u32_of("embedding_length_out")?,
        vocab,
    })
}

/// Check a [`GemmaConfig`] against the file's own tensors: every per-layer
/// array decides a projection's width, so each layer's `attn_q`, `attn_k`,
/// `attn_v`, `attn_q_norm` and `attn_k_norm` must be as wide as the config says.
/// A mismatch here is what "reject the file loudly" means — the alternative is a
/// graph that reshapes one width as another and computes nonsense.
fn check_gemma_tensors(file: &GgufFile, g: &GemmaConfig) -> Result<()> {
    let shape = |name: &str| -> Result<Vec<usize>> {
        file.get(name)
            .map(|t| t.shape.clone())
            .ok_or_else(|| anyhow!("tensor `{name}` is missing"))
    };
    let want = |name: &str, expect: &[usize]| -> Result<()> {
        let got = shape(name)?;
        if got.as_slice() != expect {
            bail!("`{name}` is {got:?}, the config says {expect:?}");
        }
        Ok(())
    };
    for il in 0..g.layers {
        let (hd, nkv) = (g.head_dim_of(il), g.kv_heads_of(il));
        let p = format!("blk.{il}");
        want(&format!("{p}.attn_q.weight"), &[g.d, g.heads * hd])?;
        want(&format!("{p}.attn_k.weight"), &[g.d, nkv * hd])?;
        want(&format!("{p}.attn_v.weight"), &[g.d, nkv * hd])?;
        want(&format!("{p}.attn_output.weight"), &[g.heads * hd, g.d])?;
        want(&format!("{p}.attn_q_norm.weight"), &[hd])?;
        want(&format!("{p}.attn_k_norm.weight"), &[hd])?;
        want(&format!("{p}.ffn_gate.weight"), &[g.d, g.ffn])?;
        want(&format!("{p}.ffn_up.weight"), &[g.d, g.ffn])?;
        want(&format!("{p}.ffn_down.weight"), &[g.ffn, g.d])?;
        want(&format!("{p}.inp_gate.weight"), &[g.d, g.ple])?;
        want(&format!("{p}.proj.weight"), &[g.ple, g.d])?;
    }
    want("token_embd.weight", &[g.d, g.vocab])?;
    want("per_layer_model_proj.weight", &[g.d, g.layers * g.ple])?;
    want("per_layer_proj_norm.weight", &[g.ple])?;
    want("output.weight", &[g.d, g.out_width])?;
    want("output_norm.weight", &[g.d])?;
    Ok(())
}

/// Merge the `ced.*` metadata of an audio tagger into a [`CedConfig`].
///
/// Every key is compulsory: the front end's arithmetic is the file's own — its
/// clamp, its window, its `amin` — so a default here would be a second, silently
/// different front end. Each refusal names the key it read.
fn merge_ced(file: &GgufFile, arch: &str) -> Result<CedConfig> {
    let u = |suffix: &str| -> Result<usize> {
        let key = format!("{arch}.{suffix}");
        match file.metadata.get(&key) {
            Some(v) => v
                .as_u32()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("metadata key `{key}` is {}, not a u32", meta_type(v))),
            None => Err(anyhow!("metadata key `{key}` is missing")),
        }
    };
    let f = |suffix: &str| -> Result<f32> {
        let key = format!("{arch}.{suffix}");
        match file.metadata.get(&key) {
            Some(MetaValue::F32(v)) => Ok(*v),
            Some(MetaValue::F64(v)) => Ok(*v as f32),
            Some(v) => Err(anyhow!(
                "metadata key `{key}` is {}, not a float",
                meta_type(v)
            )),
            None => Err(anyhow!("metadata key `{key}` is missing")),
        }
    };
    let b = |suffix: &str| -> Result<bool> {
        let key = format!("{arch}.{suffix}");
        match file.metadata.get(&key) {
            Some(MetaValue::Bool(v)) => Ok(*v),
            Some(v) => Err(anyhow!(
                "metadata key `{key}` is {}, not a bool",
                meta_type(v)
            )),
            None => Err(anyhow!("metadata key `{key}` is missing")),
        }
    };
    let s = |suffix: &str| -> Result<String> {
        let key = format!("{arch}.{suffix}");
        match file.metadata.get(&key) {
            Some(MetaValue::String(v)) => Ok(v.clone()),
            Some(v) => Err(anyhow!(
                "metadata key `{key}` is {}, not a string",
                meta_type(v)
            )),
            None => Err(anyhow!("metadata key `{key}` is missing")),
        }
    };
    let labels = {
        let key = format!("{arch}.labels");
        match file.metadata.get(&key) {
            Some(MetaValue::Array(items)) => items
                .iter()
                .map(|v| match v {
                    MetaValue::String(s) => Ok(s.clone()),
                    other => Err(anyhow!(
                        "metadata key `{key}` holds a {}, not a string",
                        meta_type(other)
                    )),
                })
                .collect::<Result<Vec<String>>>()?,
            Some(v) => bail!("metadata key `{key}` is {}, not an array", meta_type(v)),
            None => bail!("metadata key `{key}` is missing"),
        }
    };
    let config = CedConfig {
        embed_dim: u("embed_dim")?,
        depth: u("depth")?,
        heads: u("num_heads")?,
        mlp_ratio: f("mlp_ratio")?,
        output_dim: u("outputdim")?,
        qkv_bias: b("qkv_bias")?,
        patch_size: u("patch_size")?,
        patch_stride: u("patch_stride")?,
        target_length: u("target_length")?,
        pooling: s("pooling")?,
        ln_eps_encoder: f("ln_eps_encoder")?,
        ln_eps_head: f("ln_eps_head")?,
        bn_eps: f("bn_eps")?,
        sample_rate: u("sample_rate")?,
        n_mels: u("n_mels")?,
        n_fft: u("n_fft")?,
        win_size: u("win_size")?,
        hop_size: u("hop_size")?,
        n_freqs: u("n_freqs")?,
        f_min: f("f_min")?,
        f_max: f("f_max")?,
        center: b("center")?,
        a2db_multiplier: f("a2db_multiplier")?,
        a2db_amin: f("a2db_amin")?,
        a2db_top_db: f("a2db_top_db")?,
        a2db_ref: f("a2db_ref")?,
        labels,
    };

    // ── the arithmetic the metadata itself has to settle ──
    // These are relations between keys, not between keys and tensors, so they
    // are refused here, before any tensor is looked at: a caller that asks about
    // one of them gets the key it asked about.
    if config.heads == 0 || !config.embed_dim.is_multiple_of(config.heads) {
        bail!(
            "metadata key `{arch}.embed_dim` is {}, which is not a whole number of \
             head widths (`{arch}.num_heads` is {})",
            config.embed_dim,
            config.heads
        );
    }
    if config.patch_stride == 0 {
        bail!("metadata key `{arch}.patch_stride` is 0");
    }
    if config.patch_size != config.patch_stride {
        bail!(
            "metadata key `{arch}.patch_size` is {} but `{arch}.patch_stride` is {}: the \
             patch matrix is the convolution only when the kernel does not overlap itself",
            config.patch_size,
            config.patch_stride
        );
    }
    if config.target_length < config.patch_size {
        bail!(
            "metadata key `{arch}.target_length` is {}, shorter than one {} frame patch",
            config.target_length,
            config.patch_size
        );
    }
    if config.n_mels < config.patch_size {
        bail!(
            "metadata key `{arch}.n_mels` is {}, shorter than one {} band patch",
            config.n_mels,
            config.patch_size
        );
    }
    if config.win_size != config.n_fft {
        bail!(
            "metadata key `{arch}.win_size` is {} but `{arch}.n_fft` is {}: a short window \
             would have to be centred in the transform, and this front end does not",
            config.win_size,
            config.n_fft
        );
    }
    if config.n_freqs != config.n_fft / 2 + 1 {
        bail!(
            "metadata key `{arch}.n_freqs` is {}, not the {} bins a {}-point real transform has",
            config.n_freqs,
            config.n_fft / 2 + 1,
            config.n_fft
        );
    }
    if !config.center {
        bail!("metadata key `{arch}.center` is false: this front end reflect-pads half a window");
    }
    if config.a2db_ref != 1.0 {
        bail!(
            "metadata key `{arch}.a2db_ref` is {}: this front end's dB scale is relative to 1.0",
            config.a2db_ref
        );
    }
    if config.pooling != "mean" {
        bail!(
            "metadata key `{arch}.pooling` is `{}`: this graph pools its tokens by mean",
            config.pooling
        );
    }
    if config.labels.len() != config.output_dim {
        bail!(
            "metadata key `{arch}.labels` has {} entries but `{arch}.outputdim` is {}",
            config.labels.len(),
            config.output_dim
        );
    }
    Ok(config)
}

/// Check a [`CedConfig`] against the file's own tensors — every shape the graph
/// reshapes, narrows or multiplies is named, so a file whose metadata and
/// tensors disagree is refused here rather than computed with into nonsense.
fn check_ced_tensors(file: &GgufFile, c: &CedConfig) -> Result<()> {
    let want = |name: &str, expect: &[usize]| -> Result<()> {
        let got = file
            .get(name)
            .map(|t| t.shape.clone())
            .ok_or_else(|| anyhow!("tensor `{name}` is missing"))?;
        if got.as_slice() != expect {
            bail!("`{name}` is {got:?}, the config says {expect:?}");
        }
        Ok(())
    };
    let d = c.embed_dim;
    let ffn = c.ffn();
    let (ow, oh) = (c.time_tokens(), c.freq_tokens());

    want("ced.mel_filterbank", &[c.n_freqs, c.n_mels])?;
    want("ced.mel_window", &[c.win_size])?;
    // The two position tables are declared with a trailing unit axis and their
    // position axis first, so the axes that carry the numbers are ne[0], ne[2]
    // for the time table and ne[1], ne[2] for the frequency one.
    want("encoder.time_pos_embed", &[ow, 1, d, 1])?;
    want("encoder.freq_pos_embed", &[1, oh, d, 1])?;
    for t in ["weight", "bias", "running_mean", "running_var"] {
        want(&format!("encoder.init_bn.{t}"), &[c.n_mels])?;
    }
    want(
        "encoder.patch_embed.proj.weight",
        &[c.patch_size, c.patch_size, 1, d],
    )?;
    want("encoder.patch_embed.proj.bias", &[d])?;
    for il in 0..c.depth {
        let p = format!("encoder.blocks.{il}");
        for t in ["norm1", "norm2"] {
            want(&format!("{p}.{t}.weight"), &[d])?;
            want(&format!("{p}.{t}.bias"), &[d])?;
        }
        want(&format!("{p}.attn.qkv.weight"), &[d, 3 * d])?;
        if c.qkv_bias {
            want(&format!("{p}.attn.qkv.bias"), &[3 * d])?;
        }
        want(&format!("{p}.attn.proj.weight"), &[d, d])?;
        want(&format!("{p}.attn.proj.bias"), &[d])?;
        want(&format!("{p}.mlp.fc1.weight"), &[d, ffn])?;
        want(&format!("{p}.mlp.fc1.bias"), &[ffn])?;
        want(&format!("{p}.mlp.fc2.weight"), &[ffn, d])?;
        want(&format!("{p}.mlp.fc2.bias"), &[d])?;
    }
    want("encoder.norm.weight", &[d])?;
    want("encoder.norm.bias", &[d])?;
    want("outputlayer.0.weight", &[d])?;
    want("outputlayer.0.bias", &[d])?;
    want("outputlayer.1.weight", &[d, c.output_dim])?;
    want("outputlayer.1.bias", &[c.output_dim])?;
    Ok(())
}

/// Merge the `clip.*` metadata of an `mmproj` into a typed config.
fn merge_clip(file: &GgufFile) -> ClipConfig {
    let take = |key: &str| -> Option<u32> { meta_u32(file, key) };
    let dim = take("clip.audio.embedding_length").unwrap_or(0) as usize;
    let heads = take("clip.audio.attention.head_count").unwrap_or(0) as usize;
    let gen_projection_dim = take("clip.gen.audio.projection_dim").map(|v| v as usize);
    let gen_layers = take("clip.gen.audio.block_count").map(|v| v as usize);
    let gen_heads = take("clip.gen.audio.attention.head_count").map(|v| v as usize);
    let gen_kv_heads = take("clip.gen.audio.attention.head_count_kv").map(|v| v as usize);
    let gen_ffn = take("clip.gen.audio.feed_forward_length").map(|v| v as usize);
    let gen_eps = if file
        .metadata
        .contains_key("clip.gen.audio.attention.layer_norm_epsilon")
    {
        meta_f32(file, "clip.gen.audio.attention.layer_norm_epsilon")
    } else {
        None
    };
    file.metadata.contains_key("clip.audio.projector_type");
    file.metadata.contains_key("clip.gen.audio.projector_type");
    ClipConfig {
        dim,
        ffn: take("clip.audio.feed_forward_length").unwrap_or(0) as usize,
        layers: take("clip.audio.block_count").unwrap_or(0) as usize,
        heads,
        head_dim: dim.checked_div(heads).unwrap_or(0),
        mel_bins: take("clip.audio.num_mel_bins").unwrap_or(0) as usize,
        eps: meta_f32(file, "clip.audio.attention.layer_norm_epsilon").unwrap_or(1e-5),
        projection_dim: take("clip.audio.projection_dim").unwrap_or(0) as usize,
        gen_projection_dim,
        gen_layers,
        gen_heads,
        gen_kv_heads,
        gen_ffn,
        gen_eps,
    }
}

// ── the pair: a body and the `mmproj` that goes with it ────────────

/// A body file and its sound file as one unit — the notion the ready-made
/// importer does not have. Weights of both files are served
/// through one [`WeightSource`](rlx_flow::WeightSource), and nothing is glued
/// before the pair has been checked against its own metadata.
///
/// What a pair is checked for:
///
/// * **the roles agree** — a `qwen3tts` body with the recognition `mmproj` is
///   two files that load fine and mean nothing together;
/// * **the projector's width is the body's width** — `clip.audio.projection_dim`
///   (recognition) and `clip.gen.audio.projection_dim` (the synthesis talker's
///   generative half) against `{arch}.embedding_length` of the body; this is the
///   one dimension that must match across the two files, and it is what makes
///   "mel → logits" a chain rather than two graphs in a row;
/// * **the talker's width is the body's width** — the synthesis sound file's
///   `a.gen.code.proj_in.weight` input axis against the body's `d`;
/// * **the code vocabulary is the talker's head** — the synthesis body's
///   `output.weight` rows (3072) against `a.gen.code.out_embd.weight` rows;
/// * **the head configuration** — a body paired with a sound file has to carry
///   its own `output.weight`; the tied-table body (embeddings) has no `mmproj`
///   at all, and a pair for it is refused.
pub struct Pair {
    pub body: Arc<Model>,
    pub sound: Arc<Model>,
}

/// Which body goes with which `mmproj`: one row per role the stand runs. The
/// embeddings role has no sound file.
///
/// The pairing is a table rather than a rule off the names: the two halves of a
/// pair do not share a name (the 1.7B body is `Q8_0`, its sound file `Q8_0`),
/// and the half that says who it is — the `mmproj`'s `clip.*` metadata — cannot
/// be read without opening it. Both halves are checked against each other before
/// they are used: see [`Pair::open`].
pub const PAIRS: &[(Role, &str, &str)] = &[
    (
        Role::Recognition,
        "Qwen3-ASR-1.7B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
    ),
    (
        Role::Synthesis,
        "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
        "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
    ),
];

/// The two file names of a role. The embeddings role has none.
pub fn pair_files(role: Role) -> Option<(&'static str, &'static str)> {
    PAIRS
        .iter()
        .find(|(r, _, _)| *r == role)
        .map(|(_, body, sound)| (*body, *sound))
}

impl Pair {
    /// The pair of a role from [`PAIRS`], under [`models_dir`].
    pub fn of_role(role: Role) -> Result<Self> {
        Self::of_role_optional(role)?.ok_or_else(|| match pair_files(role) {
            Some((body, sound)) => anyhow!(
                "no `{}` pair — `{body}` and `{sound}` are not under the models directory",
                role.name()
            ),
            None => anyhow!("no `{}` pair — pairs: {}", role.name(), PAIRS.len()),
        })
    }

    /// [`Pair::of_role`], with a role whose files are not there reported as
    /// `Ok(None)` instead of an error.
    ///
    /// A caller that skips when the pair is absent but must fail when it is present
    /// and broken asks this, so "is this pair here" is resolved in one place rather
    /// than re-derived from [`PAIRS`] and [`models_dir`].
    pub fn of_role_optional(role: Role) -> Result<Option<Self>> {
        let Some((body, sound)) = pair_files(role) else {
            return Ok(None);
        };
        let dir = models_dir()?;
        let (body, sound) = (dir.join(body), dir.join(sound));
        if !(body.is_file() && sound.is_file()) {
            return Ok(None);
        }
        Self::open(body, sound).map(Some)
    }

    /// Open both files and check that they belong together. The body file must
    /// come first; the `mmproj` is recognised by its own `clip` architecture, not
    /// by the argument's position.
    pub fn open(body_path: impl AsRef<Path>, sound_path: impl AsRef<Path>) -> Result<Self> {
        // Wrapped in `Arc` up front: every graph of a pair clones the two
        // handles, and the second `Model` is what the 18 GiB machine pays for.
        let first = Arc::new(Model::open(body_path)?);
        let second = Arc::new(Model::open(sound_path)?);
        let (body, sound) = if first.sound && !second.sound {
            (second, first)
        } else if first.sound && second.sound {
            bail!("both files are `mmproj` — a pair is a body and its sound file");
        } else if !first.sound && !second.sound {
            bail!("neither file is an `mmproj` — a pair is a body and its sound file");
        } else {
            (first, second)
        };

        // Every check either agrees or names the disagreement; a failed one
        // refuses the pair, so a usable pair is one whose list passed whole.
        let check = |expected: String, got: String| -> bool { expected == got };
        let fail = |what: &str, expected: &str, got: &str| -> Result<Self> {
            bail!(
                "pair `{}` + `{}`: {what} — expected {expected}, got {got}",
                body.path.display(),
                sound.path.display()
            )
        };

        // 1. The roles must agree.
        let role_name = |m: &Model| {
            m.role
                .map_or_else(|| "no role".to_string(), |r| r.name().to_string())
        };
        let want_role = role_name(&body);
        let got_role = role_name(&sound);
        if !check(want_role.clone(), got_role.clone()) {
            return fail("roles agree", &want_role, &got_role);
        }

        // 2. The body carries its own head: the tied-table body has no sound file.
        let head_kind = match body.config().head {
            Head::Separate { vocab, .. } => format!("separate `output.weight`, {vocab} wide"),
            Head::Tied { table, vocab } => format!("tied `{table}`, {vocab} wide"),
        };
        let head_ok = matches!(body.config().head, Head::Separate { .. });
        if !check(
            "separate".into(),
            if head_ok {
                "separate".into()
            } else {
                head_kind.clone()
            },
        ) {
            return fail("body head", "a separate `output.weight`", &head_kind);
        }

        // 3. The projector's width is the body's width.
        let d = body.config().d.to_string();
        let proj = sound.clip().projection_dim.to_string();
        if !check(d.clone(), proj.clone()) {
            return fail("audio projector width == body d", &d, &proj);
        }

        // 4. When the sound file also carries the generative half: its projector
        //    width and the talker width are both the body's `d`.
        if let Some(gen_proj) = sound.clip().gen_projection_dim {
            let gen_proj = gen_proj.to_string();
            if !check(d.clone(), gen_proj.clone()) {
                return fail("generative projector width == body d", &d, &gen_proj);
            }
            // The talker's hidden state goes into the code predictor: through the
            // file's own projection where it carries one (`proj_in`, the 1.7B
            // file) and straight into the blocks where it does not (the 0.6B
            // file) — either way the width it arrives at is the body's.
            let talker = sound.code_proj_widths()?.0.to_string();
            if !check(d.clone(), talker.clone()) {
                return fail("code predictor input width == body d", &d, &talker);
            }
            // The talker's head *is* the predictor's first table: the same
            // vocabulary, seen from both files.
            let vocab = match body.config().head {
                Head::Separate { vocab, .. } | Head::Tied { vocab, .. } => vocab,
            }
            .to_string();
            let table = sound.plan("a.gen.code.out_embd.weight")?.ggml_shape[1].to_string();
            if !check(vocab.clone(), table.clone()) {
                return fail("talker head vocab == code out_embd rows", &vocab, &table);
            }
        }

        Ok(Self { body, sound })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned file's encoder config — the values its `gemma-embedding2.*`
    /// metadata states: 24 blocks, 20 sliding at head 256 then a global one at
    /// head 512 in every sixth slot, 2 KV heads throughout.
    fn pinned_gemma() -> GemmaConfig {
        GemmaConfig {
            layers: 24,
            d: 512,
            ffn: 2048,
            heads: 4,
            head_dim: 512,
            head_dim_swa: 256,
            kv_heads: vec![2; 24],
            sliding: (0..24).map(|il| il % 6 != 5).collect(),
            window: 1024,
            eps: 1e-6,
            rope_base: 1_000_000.0,
            rope_base_swa: 10_000.0,
            ple: 512,
            out_width: 768,
            vocab: 262_144,
        }
    }

    /// Every linear the encoder graph states at both cells' row counts, as
    /// `(m, k, n)`, from [`crate::graphs::encoder_shapes`] — the same list the
    /// stage reads its projections from. Whole on all three axes at either `m`.
    fn encoder_shapes(rows: usize) -> Vec<(usize, usize, usize)> {
        crate::graphs::encoder_shapes(&pinned_gemma(), rows)
    }

    /// The probe is the first gate: a device that cannot pass
    /// `f16_f32acc_path_ok()` — the f16 route's whole point — keeps the packed
    /// bytes, whatever the extents are. This is the device-free pin of that
    /// refusal: the predicate never asks a device, so it is exercised on a
    /// machine with no adapter at all.
    #[test]
    fn the_f16_route_is_refused_when_the_probe_fails() {
        for rows in [256usize, 8_192] {
            assert!(
                f16_route(false, true, &encoder_shapes(rows)),
                "the encoder's shapes admit the tile at {rows} rows"
            );
            assert!(
                !f16_route(false, false, &encoder_shapes(rows)),
                "a failed probe must keep the packed route at {rows} rows"
            );
        }
    }

    /// Either pin keeps the packed route: the stand's own, and the backend's
    /// absolute f32 pin, read through the predicate the store rule calls — so the
    /// pin reaches the store, not only the arm. Under it the f16 residency and the
    /// `gemm=f16coop` witness would both be false.
    #[test]
    fn the_backend_f32_pin_keeps_the_encoder_packed() {
        let shapes = encoder_shapes(8_192);
        assert!(
            f16_route(f16_pinned(false), true, &shapes),
            "with neither pin set the encoder's shapes take the cooperative route"
        );
        let _pin = rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_MATMUL_F32_ONLY", "1")));
        assert!(f16_pinned(false));
        assert!(!f16_route(f16_pinned(false), true, &shapes));
    }

    /// The explicit switch, and the extent gate underneath it: one multiply that
    /// does not admit the 64×64 tile keeps **every** linear packed, because an
    /// f16 weight on an unadmitted shape falls to `matmul_f16_compute` — slower
    /// than the packed route it would replace.
    #[test]
    fn one_unadmitted_multiply_keeps_the_whole_graph_packed() {
        let mut shapes = encoder_shapes(8_192);
        assert!(f16_route(false, true, &shapes));
        assert!(!f16_route(true, true, &shapes), "MAHBOT_EMBED_GEMM=packed");
        // `m` is the row count: a graph whose rows are not a whole number of
        // 64-row tiles is the case the two cells do not have and a caller can.
        assert!(!f16_route(false, true, &encoder_shapes(8_190)));
        // ... and so is one projection whose `n` or `k` misses its own alignment.
        shapes.push((8_192, 512, 7_680 + 8));
        assert!(!f16_route(false, true, &shapes));
        shapes.pop();
        shapes.push((8_192, 512 + 16, 512));
        assert!(!f16_route(false, true, &shapes));
        shapes.pop();
        assert!(
            f16_route(false, true, &shapes),
            "restoring it restores the route"
        );
    }

    /// An empty shape list is a graph with no linear at all — nothing to route.
    #[test]
    fn a_graph_with_no_linear_takes_no_f16_route() {
        assert!(!f16_route(false, true, &[]));
    }
}
