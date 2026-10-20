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

/// The model directory from [`MODELS_DIR_ENV`], or the error naming it.
pub fn models_dir() -> Result<PathBuf> {
    env_dir(MODELS_DIR_ENV, "the model directory")
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
    pub role: Role,
    /// `true` for the `mmproj` (sound) files. What that means for the weights is
    /// the rule below: rank-2 weights keep the file's own quantised bytes, the
    /// non-linear sound tensors go to the host as f32.
    pub sound: bool,
    pub config: Option<Config>,
    pub clip: Option<ClipConfig>,
    pub plans: Vec<Plan>,
    by_name: HashMap<String, usize>,
}

fn meta_u32(file: &GgufFile, key: &str) -> Option<u32> {
    file.metadata.get(key).and_then(MetaValue::as_u32)
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
                "qwen3" => Role::Embeddings,
                other => bail!("unknown architecture `{other}` in {}", path.display()),
            }
        };

        let config = if sound {
            None
        } else {
            Some(merge_config(&file, &arch)?)
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
        let by_name = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();

        Ok(Self {
            path,
            file,
            role,
            sound,
            config,
            clip,
            plans,
            by_name,
        })
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

    // ── bytes ───────────────────────────────────────────────────────

    /// The file's own bytes for a tensor, read on demand: `pread` into a buffer
    /// that lives only for this tensor, so the checkpoint's pages never enter the
    /// process's resident set behind the uploaded copy ([`GgufFile::read_tensor_bytes_into`]).
    pub fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` is not in {}", self.path.display()))?;
        let mut bytes = Vec::new();
        if self.file.read_tensor_bytes_into(t, &mut bytes)? {
            Ok(bytes)
        } else {
            Ok(self.file.tensor_bytes(t)?.to_vec())
        }
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
        let (body, sound) = pair_files(role)
            .ok_or_else(|| anyhow!("no `{}` pair — pairs: {}", role.name(), PAIRS.len()))?;
        let dir = models_dir()?;
        Self::open(dir.join(body), dir.join(sound))
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
        let want_role = body.role.name().to_string();
        let got_role = sound.role.name().to_string();
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
