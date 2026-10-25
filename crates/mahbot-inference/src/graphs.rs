// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The three role graphs, assembled on `rlx-flow`.
//!
//! Each graph is one `ModelFlow` plus one `LayerStage`; the stage emits the
//! blocks through the loader's helpers (`emit.rs`) and the flow carries the
//! result to a `Graph` a session can compile. Every decision that shapes the
//! graphs is visible here: the sound stage is a separate graph from the body and runs
//! host-unpacked weights, ConvTranspose uses the backend's transposed
//! convolution, the embeddings file's table rows are read by the stand and enter
//! the encoder as its input, and the codec reads its codebook by index.
//!
//! Shapes are taken from the files themselves (`Config` / `ClipConfig` /
//! `GemmaConfig` merged from the metadata), not written down as constants: that
//! is the point of doing the loading ourselves.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow, bail};
use rlx_flow::{BuiltModel, FlowCtx, FlowValue, LayerStage, ModelFlow, StageArtifacts};
use rlx_ir::hir::HirGraphExt;
use rlx_ir::op::{Activation, MaskKind, PadMode};
use rlx_ir::{DType, HirNodeId, Op, Shape};
use rlx_runtime::{Device, Session};

use crate::emit::{self, LoaderWeights, PairWeights};
use crate::loader::{Gemm, GemmaConfig, Head, Model, Pair, Store};

// ── generic block: Qwen3 body and CLIP encoder are the same skeleton ──

#[derive(Clone, Copy)]
pub struct BlkSpec {
    pub d: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    /// RMSNorm (Qwen3 / wave transformer) vs LayerNorm with bias (CLIP).
    pub rms: bool,
    /// gate+up+SiLU vs up/GELU/down.
    pub gated: bool,
    pub qk_norm: bool,
    pub rope: bool,
    pub bias: bool,
    pub causal: bool,
    /// LayerScale on both residual branches.
    pub ls: bool,
    pub norm1: &'static str,
    pub norm2: &'static str,
    pub o_name: &'static str,
}

/// Emit one block over `m` rows of `seq` sequence positions each, so `m / seq`
/// independent sequences share the weights in one pass. `seq == m` is the one
/// sequence every other graph here uses, and the graph it emits is unchanged.
pub fn block(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    p: &str,
    b: &BlkSpec,
    x: HirNodeId,
    m: usize,
    seq: usize,
    rope: Option<(HirNodeId, HirNodeId)>,
) -> Result<HirNodeId> {
    let h = if b.rms {
        let gamma = emit::f32_param(ctx, model, &format!("{p}.{}.weight", b.norm1), &[b.d])?;
        emit::rms_norm(ctx, x, gamma, b.eps, &[m, b.d])
    } else {
        let gamma = emit::f32_param(ctx, model, &format!("{p}.{}.weight", b.norm1), &[b.d])?;
        let beta = emit::f32_param(ctx, model, &format!("{p}.{}.bias", b.norm1), &[b.d])?;
        emit::ln(ctx, x, gamma, beta, b.eps)
    };

    let q3 = vec![(m / seq) as i64, seq as i64, (b.heads * b.head_dim) as i64];
    let kv3 = vec![
        (m / seq) as i64,
        seq as i64,
        (b.kv_heads * b.head_dim) as i64,
    ];

    // Projections: one call site, two weight paths (packed body / unpacked sound);
    // the biases ride outside the multiply, as the caller adds them below.
    let [q_mm, k_mm, v_mm] = emit::qkv(
        ctx,
        model,
        h,
        m,
        b.d,
        [
            &format!("{p}.attn_q.weight"),
            &format!("{p}.attn_k.weight"),
            &format!("{p}.attn_v.weight"),
        ],
    )?;

    let (q, k, v) = if b.bias {
        let bq = emit::f32_param(
            ctx,
            model,
            &format!("{p}.attn_q.bias"),
            &[1, b.heads * b.head_dim],
        )?;
        let q = emit::add(ctx, q_mm, bq);
        let bk = emit::f32_param(
            ctx,
            model,
            &format!("{p}.attn_k.bias"),
            &[1, b.kv_heads * b.head_dim],
        )?;
        let k = emit::add(ctx, k_mm, bk);
        let bv = emit::f32_param(
            ctx,
            model,
            &format!("{p}.attn_v.bias"),
            &[1, b.kv_heads * b.head_dim],
        )?;
        let v = emit::add(ctx, v_mm, bv);
        (q, k, v)
    } else {
        (q_mm, k_mm, v_mm)
    };

    let (q, k) = if b.qk_norm {
        let r = emit::reshape(ctx, q, &[(m * b.heads) as i64, b.head_dim as i64]);
        let gamma = emit::f32_param(
            ctx,
            model,
            &format!("{p}.attn_q_norm.weight"),
            &[b.head_dim],
        )?;
        let r = emit::rms_norm(ctx, r, gamma, b.eps, &[m * b.heads, b.head_dim]);
        let q = emit::reshape(ctx, r, &q3);
        let r = emit::reshape(ctx, k, &[(m * b.kv_heads) as i64, b.head_dim as i64]);
        let gamma = emit::f32_param(
            ctx,
            model,
            &format!("{p}.attn_k_norm.weight"),
            &[b.head_dim],
        )?;
        let r = emit::rms_norm(ctx, r, gamma, b.eps, &[m * b.kv_heads, b.head_dim]);
        let k = emit::reshape(ctx, r, &kv3);
        (q, k)
    } else {
        (emit::reshape(ctx, q, &q3), emit::reshape(ctx, k, &kv3))
    };
    let v = emit::reshape(ctx, v, &kv3);
    let (q, k) = if b.rope {
        let (c, s) = rope.ok_or_else(|| anyhow!("{p}: RoPE tables were not prepared"))?;
        (
            emit::h(ctx).rope(q, c, s, b.head_dim),
            emit::h(ctx).rope(k, c, s, b.head_dim),
        )
    } else {
        (q, k)
    };

    let q4 = emit::reshape(
        ctx,
        q,
        &[
            (m / seq) as i64,
            seq as i64,
            b.heads as i64,
            b.head_dim as i64,
        ],
    );
    let k4 = emit::reshape(
        ctx,
        k,
        &[
            (m / seq) as i64,
            seq as i64,
            b.kv_heads as i64,
            b.head_dim as i64,
        ],
    );
    let v4 = emit::reshape(
        ctx,
        v,
        &[
            (m / seq) as i64,
            seq as i64,
            b.kv_heads as i64,
            b.head_dim as i64,
        ],
    );
    // Head-major in, head-major out: the core picks the layout by "does axis 1
    // hold `num_heads`?" (`rlx-ir/src/attention_layout.rs::attention_geom`), so a
    // rows-first `[1, S, H, D]` is misread as head-major whenever `S == H`.
    // Head-major is unambiguous at any `S`, so every block here hands it over;
    // at `seq == 1` moving the length-1 axis keeps every element in place, so
    // the declaration alone is that layout and no step is needed.
    let (q4, k4, v4) = if seq == 1 {
        let head_major_dims = |heads: usize| {
            [
                (m / seq) as i64,
                heads as i64,
                seq as i64,
                b.head_dim as i64,
            ]
        };
        (
            emit::reshape(ctx, q4, &head_major_dims(b.heads)),
            emit::reshape(ctx, k4, &head_major_dims(b.kv_heads)),
            emit::reshape(ctx, v4, &head_major_dims(b.kv_heads)),
        )
    } else {
        (
            emit::h(ctx).transpose_(q4, vec![0, 2, 1, 3]),
            emit::h(ctx).transpose_(k4, vec![0, 2, 1, 3]),
            emit::h(ctx).transpose_(v4, vec![0, 2, 1, 3]),
        )
    };
    let kind = if b.causal {
        MaskKind::Causal
    } else {
        MaskKind::None
    };
    // The declared output shape is the operands' own head-major layout.
    let a = emit::attention(
        ctx,
        q4,
        k4,
        v4,
        b.heads,
        b.head_dim,
        kind,
        &[m / seq, b.heads, seq, b.head_dim],
        None,
    );
    let a = if seq == 1 {
        emit::reshape(ctx, a, &[m as i64, (b.heads * b.head_dim) as i64])
    } else {
        let a = emit::h(ctx).transpose_(a, vec![0, 2, 1, 3]);
        emit::reshape(ctx, a, &[m as i64, (b.heads * b.head_dim) as i64])
    };

    let w = emit::linear(
        ctx,
        &FlowValue::new(a, Shape::new(&[m, b.heads * b.head_dim], DType::F32)),
        &format!("{p}.{}", b.o_name),
    )?;
    let mut o = w.hir_id();
    if b.bias {
        let bb = emit::f32_param(ctx, model, &format!("{p}.attn_out.bias"), &[1, b.d])?;
        o = emit::add(ctx, o, bb);
    }
    if b.ls {
        let ls = emit::f32_param(ctx, model, &format!("{p}.ls1.weight"), &[1, b.d])?;
        o = emit::mul(ctx, o, ls);
    }
    let x = emit::add(ctx, x, o);

    let h = if b.rms {
        let gamma = emit::f32_param(ctx, model, &format!("{p}.{}.weight", b.norm2), &[b.d])?;
        emit::rms_norm(ctx, x, gamma, b.eps, &[m, b.d])
    } else {
        let gamma = emit::f32_param(ctx, model, &format!("{p}.{}.weight", b.norm2), &[b.d])?;
        let beta = emit::f32_param(ctx, model, &format!("{p}.{}.bias", b.norm2), &[b.d])?;
        emit::ln(ctx, x, gamma, beta, b.eps)
    };

    let f = if b.gated {
        let w = emit::linear(
            ctx,
            &FlowValue::new(h, Shape::new(&[m, b.d], DType::F32)),
            &format!("{p}.ffn_gate.weight"),
        )?;
        let gate = w.hir_id();
        let w = emit::linear(
            ctx,
            &FlowValue::new(h, Shape::new(&[m, b.d], DType::F32)),
            &format!("{p}.ffn_up.weight"),
        )?;
        let up = w.hir_id();
        let act = emit::silu(ctx, gate, &[m, b.ffn]);
        let act = emit::mul(ctx, act, up);
        let w = emit::linear(
            ctx,
            &FlowValue::new(act, Shape::new(&[m, b.ffn], DType::F32)),
            &format!("{p}.ffn_down.weight"),
        )?;
        w.hir_id()
    } else {
        let w = emit::linear(
            ctx,
            &FlowValue::new(h, Shape::new(&[m, b.d], DType::F32)),
            &format!("{p}.ffn_up.weight"),
        )?;
        let up = w.hir_id();
        let bb = emit::f32_param(ctx, model, &format!("{p}.ffn_up.bias"), &[1, b.ffn])?;
        let up = emit::add(ctx, up, bb);
        let act = emit::gelu(ctx, up, &[m, b.ffn]);
        let w = emit::linear(
            ctx,
            &FlowValue::new(act, Shape::new(&[m, b.ffn], DType::F32)),
            &format!("{p}.ffn_down.weight"),
        )?;
        let down = w.hir_id();
        let bb = emit::f32_param(ctx, model, &format!("{p}.ffn_down.bias"), &[1, b.d])?;
        emit::add(ctx, down, bb)
    };
    let f = if b.ls {
        let ls = emit::f32_param(ctx, model, &format!("{p}.ls2.weight"), &[1, b.d])?;
        emit::mul(ctx, f, ls)
    } else {
        f
    };
    let out = emit::add(ctx, x, f);
    Ok(out)
}

// ── the ASR audio encoder (recognition's sound stage) ──────────────

/// Recogntion's `mmproj`, split at the point where a group pass wants it: the
/// three stride-2 `3×3` convs and the flatten ([`EncoderConvs`]) are cheap but
/// scale badly with a batch axis — the conv route costs a window's worth of work
/// per window whatever `n` is — while everything after them
/// ([`EncoderTail`]) is linear, norm and attention, which one pass can carry for
/// a whole group. [`asr_encoder`] chains the two into the one graph the stand
/// measures; [`asr_encoder_front`] and [`asr_encoder_tail`] expose the pieces for
/// the bench's grouped passes.
pub struct EncoderConvs {
    pub model: Arc<Model>,
    pub frames: usize,
}

impl LayerStage for EncoderConvs {
    fn name(&self) -> &str {
        "asr_audio_convs"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let c = model.clip();
        let mut x = input.hir_id();

        // Three stride-2 3×3 convs: 128 mel bins → 16, frames → tokens. Both
        // axes halve every time, so the token count is frames/8 once the mel
        // axis has bottomed out at 128/8 = 16.
        let mut cur_len = self.frames;
        // Declared height of the current tensor (honest: halving from
        // `mel_bins`; the earlier stand's lie: a constant `mel_bins/2`).
        let mut hh = c.mel_bins;
        for i in 1..=3usize {
            let name = format!("a.conv2d.{i}.weight");
            let p = model.plan(&name)?;
            let cout = p.dims[0];
            let w = emit::conv_param(ctx, model, &name)?;
            let b = emit::conv_bias(ctx, model, &format!("a.conv2d.{i}.bias"), cout)?;
            let conv_h = (hh + 2 - 3) / 2 + 1;
            cur_len = (cur_len + 2 - 3) / 2 + 1;
            let y = emit::h(ctx).conv2d(
                x,
                w,
                [3, 3],
                [2, 2],
                [1, 1],
                1,
                Shape::new(&[1, cout, conv_h, cur_len], DType::F32),
            );
            let y = emit::add(ctx, y, b);
            let y = emit::gelu(ctx, y, &[1, cout, conv_h, cur_len]);
            hh = conv_h;
            x = emit::reshape(ctx, y, &[1, cout as i64, hh as i64, cur_len as i64]);
        }
        let tokens = cur_len;
        // Per token: (channel, mel) with the channel major — 480·16 = 7680.
        let flat = flat_width(c.mel_bins);
        let t = emit::h(ctx).transpose_(x, vec![0, 3, 1, 2]);
        let t = emit::reshape(ctx, t, &[tokens as i64, flat as i64]);
        let sh = Shape::new(&[tokens, flat], DType::F32);
        Ok((
            FlowValue::new(t, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The rest of the sound stage over `batch * tokens` rows of the front's width:
/// the `conv_out` linear, the position table, the CLIP blocks and `post_ln` —
/// one weight read for the whole group.
pub struct EncoderTail {
    pub model: Arc<Model>,
    pub frames: usize,
    /// Independent windows in the pass; each carries its own `tokens` rows.
    pub batch: usize,
}

impl EncoderTail {
    fn spec(&self) -> BlkSpec {
        let c = self.model.clip();
        BlkSpec {
            d: c.dim,
            ffn: c.ffn,
            heads: c.heads,
            kv_heads: c.heads,
            head_dim: c.head_dim,
            eps: c.eps,
            rms: false,
            gated: false,
            qk_norm: false,
            rope: false,
            bias: true,
            causal: false,
            ls: false,
            norm1: "ln1",
            norm2: "ln2",
            o_name: "attn_out.weight",
        }
    }
}

impl LayerStage for EncoderTail {
    fn name(&self) -> &str {
        "asr_audio_encoder"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let c = model.clip();
        let batch = self.batch.max(1);
        let tokens = asr_tokens(self.frames);
        let m = batch * tokens;
        let flat = flat_width(c.mel_bins);

        // `conv_out` is a linear layer, not a convolution: [flat → dim], no
        // bias. The checkpoint carries it as `a.conv_out.weight`.
        let proj = emit::linear(
            ctx,
            &FlowValue::new(input.hir_id(), Shape::new(&[m, flat], DType::F32)),
            "a.conv_out.weight",
        )?;
        let h = proj.hir_id();
        let pos = emit::prefix_param(
            ctx,
            model,
            "a.position_embd.weight",
            &[tokens, c.dim],
            tokens * c.dim,
        )?;
        // The position table is one sequence wide, so a group is added one
        // sequence at a time; with one sequence it is the plain add.
        let mut x = if batch == 1 {
            emit::add(ctx, h, pos)
        } else {
            let h = emit::reshape(ctx, h, &[batch as i64, tokens as i64, c.dim as i64]);
            let h = emit::add(ctx, h, pos);
            emit::reshape(ctx, h, &[m as i64, c.dim as i64])
        };

        let spec = self.spec();
        for i in 0..c.layers {
            x = block(ctx, model, &format!("a.blk.{i}"), &spec, x, m, tokens, None)?;
        }
        let gamma = emit::f32_param(ctx, model, "a.post_ln.weight", &[c.dim])?;
        let beta = emit::f32_param(ctx, model, "a.post_ln.bias", &[c.dim])?;
        let out = emit::ln(ctx, x, gamma, beta, c.eps);
        let sh = Shape::new(&[m, c.dim], DType::F32);
        Ok((
            FlowValue::new(out, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The recognition sound graph for a mel window of `frames` frames: the convs
/// and the tail chained into one graph — the graph the stand's cells measure.
pub fn asr_encoder(model: Arc<Model>, frames: usize) -> Result<BuiltModel> {
    let c = model.clip().clone();
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("asr_enc_{frames}"))
        .with_profile(profile())
        .input("mel", Shape::new(&[1, 1, c.mel_bins, frames], DType::F32))
        .layer_stage(EncoderConvs {
            model: model.clone(),
            frames,
        })
        .layer_stage(EncoderTail {
            model,
            frames,
            batch: 1,
        })
        .output("tokens");
    flow.build(&mut weights).map_err(Into::into)
}

/// The conv front on its own, one window per call: `mel` → `[tokens, flat]`.
pub fn asr_encoder_front(model: Arc<Model>, frames: usize) -> Result<BuiltModel> {
    let c = model.clip().clone();
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("asr_conv_{frames}"))
        .with_profile(profile())
        .input("mel", Shape::new(&[1, 1, c.mel_bins, frames], DType::F32))
        .layer_stage(EncoderConvs { model, frames })
        .output("flat");
    flow.build(&mut weights).map_err(Into::into)
}

/// The tail over `batch * tokens` rows of the front's width: `flat` →
/// `[batch * tokens, dim]`, one weight read for the whole group.
pub fn asr_encoder_tail(model: Arc<Model>, frames: usize, batch: usize) -> Result<BuiltModel> {
    let c = model.clip().clone();
    let flat = flat_width(c.mel_bins);
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("asr_tail_{frames}x{batch}"))
        .with_profile(profile())
        .input(
            "flat",
            Shape::new(&[batch.max(1) * asr_tokens(frames), flat], DType::F32),
        )
        .layer_stage(EncoderTail {
            model,
            frames,
            batch,
        })
        .output("tokens");
    flow.build(&mut weights).map_err(Into::into)
}

// ── the ASR projector: sound → body ────────────────────────────────

/// The projector that carries the recognition sound stage's tokens into the
/// body's width: two linear layers with an erf-GELU between them, straight from
/// `mm.a.mlp.1/2` of the `mmproj` (the reference builds it as
/// `build_ffn(..., FFN_GELU_ERF)`). It is a graph of its own, on the sound
/// file's host-unpacked weights, like the encoder it follows.
pub struct ProjectorStage {
    pub model: Arc<Model>,
    /// Mel-window tokens in (the encoder's `dim`), body width out.
    pub tokens: usize,
}

impl LayerStage for ProjectorStage {
    fn name(&self) -> &str {
        "asr_projector"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let dim = model.clip().dim;
        let out_dim = model.clip().projection_dim;
        let m = self.tokens;

        let h = emit::linear(
            ctx,
            &FlowValue::new(input.hir_id(), Shape::new(&[m, dim], DType::F32)),
            "mm.a.mlp.1.weight",
        )?;
        let b = emit::f32_param(ctx, model, "mm.a.mlp.1.bias", &[1, dim])?;
        let h = emit::add(ctx, h.hir_id(), b);
        // `Activation::Gelu` is the erf form on the processor path too, which is
        // what the reference asks for (`FFN_GELU_ERF`), not the tanh fit.
        let h = emit::gelu(ctx, h, &[m, dim]);
        let out = emit::linear(
            ctx,
            &FlowValue::new(h, Shape::new(&[m, dim], DType::F32)),
            "mm.a.mlp.2.weight",
        )?;
        let b = emit::f32_param(ctx, model, "mm.a.mlp.2.bias", &[1, out_dim])?;
        let out = emit::add(ctx, out.hir_id(), b);
        let sh = Shape::new(&[m, out_dim], DType::F32);
        Ok((
            FlowValue::new(out, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The compile profile every graph in this stand uses.
///
/// `Direct` lowering. A `fusion.skip` field is not a switch the engine reads;
/// fusion is controlled by `skip_fusion` / `RLX_METAL_NO_FUSION`, so
/// fusions do happen and no conclusion may rest on a "fusion off" profile.
pub fn profile() -> rlx_flow::CompileProfile {
    let mut p = rlx_flow::CompileProfile::encoder();
    p.fusion.policy = rlx_flow::FusionPolicyKind::Direct;
    p
}

// ── the embeddings encoder (EmbeddingGemma 2's text tower) ─────────

/// The band an encoder layer attends over, as the graph states it.
///
/// A two-sided band is only a restriction once the sequence is longer than the
/// band: at `rows <= radius + 1` the farthest pair of rows, `(0, rows - 1)`, is
/// still within `radius`, so every query reaches every key and
/// [`MaskKind::None`] names the same set of keys without stating a band. The
/// twenty sliding layers therefore state [`MaskKind::Window`] at 8 192 rows and
/// no mask at 256. The four global layers are dense at every length and state
/// [`MaskKind::None`]; which kernel serves either statement is the backend
/// router's decision, not the graph's.
#[must_use]
pub fn window_mask(rows: usize, radius: usize) -> MaskKind {
    if rows <= radius + 1 {
        MaskKind::None
    } else {
        MaskKind::Window(radius)
    }
}

/// The nine linear projections one encoder block multiplies, as
/// `(weight suffix, k, n)`, in the order the block emits them.
///
/// The stage reads its keys and extents from this one list, and the loader's
/// route tests admit their multiplies against it, so a projection added to a
/// block cannot drift between the graph and the f16 route's admission check.
#[must_use]
pub fn encoder_block_linears(
    d: usize,
    heads: usize,
    head_dim: usize,
    kv_heads: usize,
    ffn: usize,
    ple: usize,
) -> [(&'static str, usize, usize); 9] {
    [
        ("attn_q.weight", d, heads * head_dim),
        ("attn_k.weight", d, kv_heads * head_dim),
        ("attn_v.weight", d, kv_heads * head_dim),
        ("attn_output.weight", heads * head_dim, d),
        ("ffn_gate.weight", d, ffn),
        ("ffn_up.weight", d, ffn),
        ("ffn_down.weight", ffn, d),
        ("inp_gate.weight", d, ple),
        ("proj.weight", ple, d),
    ]
}

/// Every linear the encoder graph states at `rows` rows, as `(m, k, n)`: the
/// per-block list above for each layer, then the per-layer-input projection and
/// the output projection. The stage and the loader's route tests read this one
/// list.
#[must_use]
pub fn encoder_shapes(g: &GemmaConfig, rows: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::with_capacity(2 + g.layers * 9);
    out.push((rows, g.d, g.layers * g.ple));
    for il in 0..g.layers {
        for (_, k, n) in encoder_block_linears(
            g.d,
            g.heads,
            g.head_dim_of(il),
            g.kv_heads_of(il),
            g.ffn,
            g.ple,
        ) {
            out.push((rows, k, n));
        }
    }
    out.push((rows, g.d, g.out_width));
    out
}

/// The encoder's one pass: the looked-up embedding rows in, the pooled
/// 768-wide vector out — one graph over all rows, bidirectional, with no
/// padding anywhere.
///
/// The layer is the reference's (`models/gemma-embedding2.cpp`), in its order:
/// `attn_norm → q/k/v → per-head q/k norms and an unweighted V norm → RoPE at
/// the layer's own base → attention (scale 1: the q/k norms replace `1/√d`) →
/// `attn_output → post_attention_norm` → residual → `ffn_norm → gate·up (tanh
/// GELU) → down → post_ffw_norm` → residual → the per-layer-input block
/// (`inp_gate`·GELU · this layer's PLE slice → `proj` → `post_norm` → residual)
/// → one scalar `layer_output_scale`. The per-layer inputs are computed once,
/// before the stack, from the **scaled** embedding.
///
/// The mean over the rows — BOS and EOS included, as the reference engine's
/// `--pooling mean` — is in the graph; the L2 normalisation is not, so the
/// graph's output is the raw 768-vector and the record digests the normalised
/// one the stand makes of it.
pub struct EncoderStage {
    pub model: Arc<Model>,
    pub rows: usize,
}

impl LayerStage for EncoderStage {
    fn name(&self) -> &str {
        "gemma_encoder"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let g = model.gemma();
        let rows = self.rows;
        let (d, heads, eps) = (g.d, g.heads, g.eps);
        let rows_i = rows as i64;

        // The embedding is scaled by `√d` before anything reads it — the
        // reference's `build_inp_embd(tok_embd, sqrtf(n_embd))`, and the PLE
        // projection is over the scaled value, not the table's rows.
        let root = emit::constant(ctx, &[1], (d as f32).sqrt());
        let mut x = emit::mul(ctx, input.hir_id(), root);

        // ── the per-layer inputs, once for the whole stack ──
        // `[rows, d] → [rows, layers · ple]` in one multiply, the `1/√d` scale,
        // then the file's flat axis read as `[layers, ple]` with `ple` fastest
        // (the file is `[n_embd, n_embd_per_layer · n_layer]`), one RMSNorm over
        // each layer's own 512.
        let w = FlowValue::new(x, Shape::new(&[rows, d], DType::F32));
        let ple = emit::linear_encoder(ctx, model, &w, "per_layer_model_proj.weight")?;
        let inv_root = emit::constant(ctx, &[1], 1.0 / (d as f32).sqrt());
        let ple = emit::mul(ctx, ple.hir_id(), inv_root);
        let ple = emit::reshape(ctx, ple, &[rows_i, g.layers as i64, g.ple as i64]);
        let ple2 = emit::reshape(ctx, ple, &[(rows * g.layers) as i64, g.ple as i64]);
        let gamma = emit::f32_param(ctx, model, "per_layer_proj_norm.weight", &[g.ple])?;
        let ple2 = emit::rms_norm(ctx, ple2, gamma, eps, &[rows * g.layers, g.ple]);
        let per_layer = emit::reshape(ctx, ple2, &[rows_i, g.layers as i64, g.ple as i64]);

        // V carries no learnable norm in this file: an RMSNorm whose scale is
        // one at the layer's own head width, not a missing step.
        let ones_swa = emit::constant(ctx, &[g.head_dim_swa], 1.0);
        let ones = emit::constant(ctx, &[g.head_dim], 1.0);

        for il in 0..g.layers {
            let p = format!("blk.{il}");
            let hd = g.head_dim_of(il);
            let nkv = g.kv_heads_of(il);
            let lin = encoder_block_linears(d, heads, hd, nkv, g.ffn, g.ple);
            let key = |i: usize| format!("{p}.{}", lin[i].0);
            let qw = lin[0].2 as i64;
            let kw = lin[1].2 as i64;
            // One table pair per layer type: the twenty sliding layers rotate at
            // the same width and base, so they share the two leaves.
            let base = g.rope_base_of(il);
            let (cos, sin) = emit::rope_tables(
                ctx,
                if g.is_sliding(il) {
                    "rope_swa"
                } else {
                    "rope_global"
                },
                rows,
                hd,
                base,
            );

            let gamma = emit::f32_param(ctx, model, &format!("{p}.attn_norm.weight"), &[d])?;
            let h = emit::rms_norm(ctx, x, gamma, eps, &[rows, d]);
            let [q, k, v] = emit::qkv(ctx, model, h, rows, d, [&key(0), &key(1), &key(2)])?;
            // Per-head norms: reshape to `[rows · heads, head_dim]` so the
            // last-axis RMSNorm is one head's own, then back onto the row axis
            // for the rotation (the rope step reads the row as `[-2]`).
            let r = emit::reshape(ctx, q, &[(rows * heads) as i64, hd as i64]);
            let gamma = emit::f32_param(ctx, model, &format!("{p}.attn_q_norm.weight"), &[hd])?;
            let q = emit::rms_norm(ctx, r, gamma, eps, &[rows * heads, hd]);
            let r = emit::reshape(ctx, k, &[(rows * nkv) as i64, hd as i64]);
            let gamma = emit::f32_param(ctx, model, &format!("{p}.attn_k_norm.weight"), &[hd])?;
            let k = emit::rms_norm(ctx, r, gamma, eps, &[rows * nkv, hd]);
            let r = emit::reshape(ctx, v, &[(rows * nkv) as i64, hd as i64]);
            let unit = if hd == g.head_dim_swa { ones_swa } else { ones };
            let v = emit::rms_norm(ctx, r, unit, eps, &[rows * nkv, hd]);
            let q = emit::reshape(ctx, q, &[1, rows_i, qw]);
            let k = emit::reshape(ctx, k, &[1, rows_i, kw]);
            let q = emit::h(ctx).rope(q, cos, sin, hd);
            let k = emit::h(ctx).rope(k, cos, sin, hd);
            // Rows-first `[1, rows, heads, head_dim]`, declared and not
            // transposed: the route reads the layout off the extents (axis 1 is
            // the row count here, not the head count) and the strides follow, so
            // a transpose would only move the same elements twice.
            let q = emit::reshape(ctx, q, &[1, rows_i, heads as i64, hd as i64]);
            let k = emit::reshape(ctx, k, &[1, rows_i, nkv as i64, hd as i64]);
            let v = emit::reshape(ctx, v, &[1, rows_i, nkv as i64, hd as i64]);
            let mask = if g.is_sliding(il) {
                window_mask(rows, g.window_radius())
            } else {
                MaskKind::None
            };
            let a = emit::attention(
                ctx,
                q,
                k,
                v,
                heads,
                hd,
                mask,
                &[1, rows, heads, hd],
                Some(1.0),
            );
            let a = emit::reshape(ctx, a, &[rows_i, qw]);
            let o = emit::linear_encoder(
                ctx,
                model,
                &FlowValue::new(a, Shape::new(&[rows, lin[3].1], DType::F32)),
                &key(3),
            )?;
            let gamma =
                emit::f32_param(ctx, model, &format!("{p}.post_attention_norm.weight"), &[d])?;
            let o = emit::rms_norm(ctx, o.hir_id(), gamma, eps, &[rows, d]);
            let attn_out = emit::add(ctx, x, o);

            let gamma = emit::f32_param(ctx, model, &format!("{p}.ffn_norm.weight"), &[d])?;
            let f = emit::rms_norm(ctx, attn_out, gamma, eps, &[rows, d]);
            let w = FlowValue::new(f, Shape::new(&[rows, lin[4].1], DType::F32));
            let gate = emit::linear_encoder(ctx, model, &w, &key(4))?;
            // `GGML_GELU` in the reference is the **tanh** fit, not the erf form:
            // `0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`.
            let gate = emit::act(ctx, Activation::GeluApprox, gate.hir_id(), &[rows, g.ffn]);
            let w = FlowValue::new(f, Shape::new(&[rows, lin[5].1], DType::F32));
            let up = emit::linear_encoder(ctx, model, &w, &key(5))?;
            let f = emit::mul(ctx, gate, up.hir_id());
            let w = FlowValue::new(f, Shape::new(&[rows, lin[6].1], DType::F32));
            let f = emit::linear_encoder(ctx, model, &w, &key(6))?;
            let gamma = emit::f32_param(ctx, model, &format!("{p}.post_ffw_norm.weight"), &[d])?;
            let f = emit::rms_norm(ctx, f.hir_id(), gamma, eps, &[rows, d]);
            x = emit::add(ctx, attn_out, f);

            // ── the per-layer-input block ──
            let w = FlowValue::new(x, Shape::new(&[rows, lin[7].1], DType::F32));
            let gated = emit::linear_encoder(ctx, model, &w, &key(7))?;
            let gated = emit::act(ctx, Activation::GeluApprox, gated.hir_id(), &[rows, g.ple]);
            let slice = emit::narrow(ctx, per_layer, 1, il, 1);
            let slice = emit::reshape(ctx, slice, &[rows_i, g.ple as i64]);
            let gated = emit::mul(ctx, gated, slice);
            let w = FlowValue::new(gated, Shape::new(&[rows, lin[8].1], DType::F32));
            let out = emit::linear_encoder(ctx, model, &w, &key(8))?;
            let gamma = emit::f32_param(ctx, model, &format!("{p}.post_norm.weight"), &[d])?;
            let out = emit::rms_norm(ctx, out.hir_id(), gamma, eps, &[rows, d]);
            let x_ple = emit::add(ctx, x, out);
            // A scalar per layer (1.0 in this file), applied to the layer's whole
            // output rather than to one branch.
            let scale =
                emit::f32_param(ctx, model, &format!("{p}.layer_output_scale.weight"), &[1])?;
            x = emit::mul(ctx, x_ple, scale);
        }

        let gamma = emit::f32_param(ctx, model, "output_norm.weight", &[d])?;
        let x = emit::rms_norm(ctx, x, gamma, eps, &[rows, d]);
        let w = FlowValue::new(x, Shape::new(&[rows, d], DType::F32));
        let out = emit::linear_encoder(ctx, model, &w, "output.weight")?;
        // Mean over *all* rows, BOS and EOS included — no padding to exclude.
        // The projection is linear, so pooling before or after it is the same
        // arithmetic; this order is the reference's.
        let pooled = emit::h(ctx).mean(out.hir_id(), vec![0], true);
        let pooled = emit::reshape(ctx, pooled, &[g.out_width as i64]);
        let sh = Shape::new(&[g.out_width], DType::F32);
        Ok((
            FlowValue::new(pooled, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The embeddings graph: `rows` embedding rows (already looked up from the
/// table, `[rows, d]`, unscaled) → the pooled 768-wide vector.
///
/// The rows arrive looked up rather than as ids because the table's used rows
/// are the cell's own: the stand reads them with
/// [`Model::table_rows_f32`](crate::loader::Model::table_rows_f32) (one row per
/// position, ~16 MiB at 8 192 rows) and binds them as this graph's input. The
/// whole `token_embd` table as a `Param` would be half a gigabyte of f32 for
/// 262 144 rows, and an in-graph `Gather` over it would read from a leaf nobody
/// needs.
///
/// Which route the projections take is the *model's* decision, taken before the
/// graph is built ([`Model::f16_linears`]): this function builds whatever the
/// plans say. It also owns the other half of that route — the f16 image of each
/// weight the store rule flipped, dequantised here and handed to
/// `set_param_typed` — so a graph built from packed plans never pays for it.
/// The f16 route's compile needs the weight-concat fusions off; the switch is
/// installed by [`compile_encoder`], not here.
pub fn encoder(model: Arc<Model>, rows: usize) -> Result<BuiltModel> {
    let cfg = model.gemma().clone();
    if rows < 3 {
        bail!("an embedding cell needs at least BOS + one token + EOS, got {rows} rows");
    }
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("embed_{rows}"))
        .with_profile(profile())
        .input("x", Shape::new(&[rows, cfg.d], DType::F32))
        .layer_stage(EncoderStage {
            model: model.clone(),
            rows,
        })
        .output("emb");
    let mut built = flow.build(&mut weights)?;
    // The f16 route's bytes. `emit::linear_encoder` declared one f16 `Param` per
    // weight the store rule flipped and no f32 data, so the session is handed
    // the f16 image here — `set_param_typed(name, bytes, DType::F16)`, drained
    // by the runner exactly like a packed blob. Dequantising here rather than at
    // open keeps a route the device refuses from costing a byte of host memory.
    for plan in model.plans.iter().filter(|p| p.store == Store::F16Linear) {
        let (bytes, ..) = model.linear_f16(&plan.name)?;
        built
            .typed_params
            .push((plan.name.clone(), bytes, DType::F16));
    }
    Ok(built)
}

/// Compile the embedding graph with the weight-concat fusions off for the one
/// compile that reads the switch, restoring the process's previous overrides on
/// return.
///
/// `FuseSharedInputMatMul` builds q/k/v — and gate/up — as ONE `Concat` over
/// the three weights. That takes the multiply's B out of the `Param` role, and
/// every cooperative f16 family is admitted only for a weight-side `Param`;
/// worse, the F16 `Concat` node makes the runtime's `prepare_f32_exec_graph` see
/// a low-precision *arithmetic* node and promote the whole graph to f32, which
/// takes the remaining 97 multiplies off the kernel too. Measured at 8 192 rows
/// on the M3 Pro (the cell line's own figures): 5.27 s with the fusion on —
/// every multiply on the f32 cooperative kernel — and 3.97 s with it off, all
/// 217 on the f16 one.
///
/// The switch is process-global, so it is installed here, around this compile,
/// and restored on return. Installing it while the graph is built would leak it
/// into every later compile in the process, and the packed and f16 arms the
/// record compares would then differ in fusion as well as precision. The
/// override map has no atomicity of its own, so a second compile in the same
/// process would see this one's install or its restore mid-flight.
pub fn compile_encoder(device: Device, built: BuiltModel, gemm: Gemm) -> Result<Compiled> {
    static FUSION_SCOPE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Every compile takes the lock, not only the arm that installs: a compile
    // that installs nothing must still not observe another's override, and the
    // arm is chosen from a process-global read. Declared before the override, so
    // the override is restored before the lock is released.
    let _scope = FUSION_SCOPE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _fusions = (gemm == Gemm::F16Coop)
        .then(|| rlx_ir::env::RuntimeOverrides::install([("RLX_NO_WEIGHT_CONCAT_FUSION", "1")]));
    Compiled::new(device, built)
}

// ── the body with the attention state carried between windows ──────

/// The 28 blocks of a body **with the keys and values of everything already
/// seen carried in** — the KV cache as the graph sees it: one cell per layer,
/// head and element width, its size chosen by the caller.
///
/// The state is one tensor pair per layer, `[layers, pre, kv_heads·head_dim]`,
/// read as `kpre`/`vpre`. What is published back is only the window's own rows,
/// which the caller appends to its own cache — the band is never rebuilt just to
/// be handed on.
/// Attention is **causal over `[cache ; window]`** — not a sliding window: the
/// cache holds the past, the queries are the new rows, and a query must reach
/// every earlier position. (The state is RoPE'd before it is stored, exactly as
/// in [`BodyStateStage`], so the positions the caller puts in `cos`/`sin` are
/// the ones the query and key rows keep in the whole sequence.)
///
/// `prefix == 0` is the first window: nothing to read, ordinary causal attention
/// over the window alone, and the state published is the window's own rows.
///
/// [`BodyStateStage::layers`] is which of the file's blocks this graph carries:
/// the whole body (the default) or a range of the file's blocks.
pub struct BodyStateStage {
    pub model: Arc<Model>,
    /// Rows in this window.
    pub m: usize,
    /// Rows already in the cache.
    pub prefix: usize,
    /// The blocks this graph holds, `0..cfg.layers` for a whole body. The cache
    /// pair is `[layers.len(), pre, …]` — the **group's** slice, in the group's
    /// own order — so a block's row in it is its position in this range
    /// (`i - layers.start`), never the file's block number, which is what the
    /// weights are named by.
    pub layers: std::ops::Range<usize>,
    /// A **fixed-capacity** cache: the stage takes a `mask` input
    /// `[1, prefix + m]` (`1` = a row that is written) instead of a causal mask,
    /// so one graph serves the whole part and each step only rewrites the cache.
    /// False for the growing cache, where the shape itself says what is written.
    pub masked: bool,
    /// The cache's axes: `[capacity, layers, hkv]` when true (a step's row for
    /// all layers is contiguous), `[layers, capacity, hkv]` when false.
    pub row_first: bool,
    /// The file's own head on the same row, published as the `logits` side
    /// output: the body and its head are one graph and one fence per frame.
    pub head: bool,
    /// Per-layer carried caches of `[prefix + 1, hkv]` rows, the spare row at
    /// `prefix` written in-graph by `KvAppend` and one row fed device-side after
    /// the step. The attention operand is then the cache itself — the same
    /// `[prefix + 1]`-row operand the row-first layout built by transposing and
    /// concatenating, without either pass. Requires `m == 1`, `masked` and
    /// `row_first`.
    pub plane_cache: bool,
    /// Pack a block's cache input as `[capacity / rows, rows · layers · hkv]`
    /// and publish this block's new rows as one row of the same width. The
    /// memory is still the row-first `[capacity, layers, hkv]`; only the
    /// declared row is wider, so one `feed_kv_row` carries every row of a block
    /// instead of one submission per row.
    pub packed_feed: bool,
    /// q/k/v were fused at load time under `*.qkv.fused`; gate+up under
    /// `*.gate_up.fused`. The two are separate flags because on Q4_K_M `v` is
    /// Q6_K while `q`/`k` are Q4_K: q/k/v cannot share one packed scheme, while
    /// gate+up can.
    pub fused: bool,
    /// gate+up were fused at load time under `*.gate_up.fused`.
    pub fused_gate_up: bool,
}

impl LayerStage for BodyStateStage {
    fn name(&self) -> &str {
        "body_state"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let cfg = model.config().clone();
        let (m, pre) = (self.m, self.prefix);
        // The plane cache reads the whole `[prefix + 1]` row operand that the
        // in-graph append builds, so it only makes sense for one row per step,
        // against a mask, in the row-first layout the spare row assumes.
        assert!(
            !self.plane_cache || (m == 1 && self.masked && self.row_first),
            "a plane cache needs one row, a mask and the row-first layout \
             (plane_cache = true, m = {m}, masked = {}, row_first = {})",
            self.masked,
            self.row_first
        );
        let (d, nh, nkv, hd) = (cfg.d, cfg.heads, cfg.kv_heads, cfg.head_dim);
        let hkv = nkv * hd;
        // The group this graph carries, not the whole file: a split body's block
        // index (`blk.{i}`) stays the file's own, so the same weights are read.
        let layers = self.layers.len();
        let eps = cfg.eps;
        // Rows each layer hands on: this window's own, appended by the caller.
        let out_rows = m;

        let (cos, sin) = (
            emit::h(ctx).input("cos", Shape::new(&[m, hd / 2], DType::F32)),
            emit::h(ctx).input("sin", Shape::new(&[m, hd / 2], DType::F32)),
        );
        let mask = self
            .masked
            .then(|| emit::h(ctx).input("mask", Shape::new(&[1, pre + m], DType::F32)));
        // The plane cache's per-layer inputs, already in the attention's row
        // layout: `[prefix + 1, hkv]`, the spare row at `prefix` for this step's
        // own rows. Created here so the loop below reads them by layer index.
        let plane_in: Option<Vec<(HirNodeId, HirNodeId)>> = self.plane_cache.then(|| {
            (0..layers)
                .map(|gi| {
                    (
                        emit::h(ctx)
                            .input(format!("k{gi}"), Shape::new(&[pre + 1, hkv], DType::F32)),
                        emit::h(ctx)
                            .input(format!("v{gi}"), Shape::new(&[pre + 1, hkv], DType::F32)),
                    )
                })
                .collect()
        });
        let kv_in = if plane_in.is_some() {
            None
        } else if pre > 0 {
            // The fixed-capacity body keeps its cache **row-first**:
            // `[capacity, layers, hkv]`, so the new row of all layers of a step is
            // contiguous and the caller folds it in with one device-side feed. The
            // growing cache keeps the old layer-first layout. A packed feed
            // declares the same bytes as `[capacity / rows, rows · layers · hkv]`,
            // so one feed covers a whole block; the reshape below restores the
            // attention layout.
            let shape = if self.packed_feed && m > 1 && pre % m == 0 {
                Shape::new(&[pre / m, m * layers * hkv], DType::F32)
            } else if self.row_first {
                Shape::new(&[pre, layers, hkv], DType::F32)
            } else {
                Shape::new(&[layers, pre, hkv], DType::F32)
            };
            Some((
                emit::h(ctx).input("kpre", shape.clone()),
                emit::h(ctx).input("vpre", shape),
            ))
        } else {
            None
        };
        let kv_in = kv_in.map(|(kp, vp)| {
            if self.packed_feed && m > 1 && pre % m == 0 {
                (
                    emit::reshape(ctx, kp, &[pre as i64, layers as i64, hkv as i64]),
                    emit::reshape(ctx, vp, &[pre as i64, layers as i64, hkv as i64]),
                )
            } else {
                (kp, vp)
            }
        });
        // One transpose per submission, not one per block: the rest of the stage
        // reads a layer's `[pre, hkv]` slice exactly as it did layer-first.
        let kv_in = kv_in.map(|(kp, vp)| {
            if self.row_first {
                (
                    emit::h(ctx).transpose_(kp, vec![1, 0, 2]),
                    emit::h(ctx).transpose_(vp, vec![1, 0, 2]),
                )
            } else {
                (kp, vp)
            }
        });

        let mut x = input.hir_id();
        let mut k_next: Vec<HirNodeId> = Vec::with_capacity(layers);
        let mut v_next: Vec<HirNodeId> = Vec::with_capacity(layers);
        // `gi` is this block's row in the group's own cache slice — the layer
        // index `i` is the file's, which the weights are named by, but the
        // `kpre`/`vpre` input holds `layers.len()` rows, so indexing it by `i`
        // would read past the slice for every group but the first.
        for (gi, i) in self.layers.clone().enumerate() {
            let p = format!("blk.{i}");
            let gamma = emit::f32_param(ctx, model, &format!("{p}.attn_norm.weight"), &[d])?;
            let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d]);
            let lc = |x: HirNodeId| FlowValue::new(x, Shape::new(&[m, d], DType::F32));
            let (q, k, v) = if self.fused {
                let parts = emit::linear_fused(
                    ctx,
                    &lc(h),
                    &format!("{p}.qkv.fused"),
                    &[nh * hd, hkv, hkv],
                )?;
                (
                    FlowValue::new(parts[0], Shape::new(&[m, nh * hd], DType::F32)),
                    FlowValue::new(parts[1], Shape::new(&[m, hkv], DType::F32)),
                    FlowValue::new(parts[2], Shape::new(&[m, hkv], DType::F32)),
                )
            } else {
                (
                    emit::linear(ctx, &lc(h), &format!("{p}.attn_q.weight"))?,
                    emit::linear(ctx, &lc(h), &format!("{p}.attn_k.weight"))?,
                    emit::linear(ctx, &lc(h), &format!("{p}.attn_v.weight"))?,
                )
            };

            // Per-head qk-norm, then RoPE at the window's own positions.
            let q3 = vec![1, m as i64, (nh * hd) as i64];
            let kv3 = vec![1, m as i64, hkv as i64];
            let r = emit::reshape(ctx, q.hir_id(), &[(m * nh) as i64, hd as i64]);
            let g = emit::f32_param(ctx, model, &format!("{p}.attn_q_norm.weight"), &[hd])?;
            let r = emit::rms_norm(ctx, r, g, eps, &[m * nh, hd]);
            let q = emit::reshape(ctx, r, &q3);
            let r = emit::reshape(ctx, k.hir_id(), &[(m * nkv) as i64, hd as i64]);
            let g = emit::f32_param(ctx, model, &format!("{p}.attn_k_norm.weight"), &[hd])?;
            let r = emit::rms_norm(ctx, r, g, eps, &[m * nkv, hd]);
            let k = emit::reshape(ctx, r, &kv3);
            let v = emit::reshape(ctx, v.hir_id(), &kv3);
            let q = emit::h(ctx).rope(q, cos, sin, hd);
            let k = emit::h(ctx).rope(k, cos, sin, hd);

            let (q4, k4, v4) = if plane_in.is_some() {
                // The attention reads `[1, rows, heads, dim]` and the cache is
                // already in that layout, so there is nothing to permute: the
                // projections reshape into it directly.
                (
                    emit::reshape(ctx, q, &[1, m as i64, nh as i64, hd as i64]),
                    emit::reshape(ctx, k, &[1, m as i64, nkv as i64, hd as i64]),
                    emit::reshape(ctx, v, &[1, m as i64, nkv as i64, hd as i64]),
                )
            } else {
                (
                    head_major(ctx, q, m, nh, hd),
                    head_major(ctx, k, m, nkv, hd),
                    head_major(ctx, v, m, nkv, hd),
                )
            };
            // One reshape per plane serves both the in-graph append and the
            // new-row side output this layer publishes: the same node, so a
            // layer's row is not reshaped twice.
            let rows = plane_in.as_ref().map(|_| {
                (
                    emit::reshape(ctx, k4, &[m as i64, hkv as i64]),
                    emit::reshape(ctx, v4, &[m as i64, hkv as i64]),
                )
            });
            let (k_all, v_all) = if let (Some(planes), Some((k_row, v_row))) = (&plane_in, rows) {
                let (kp, vp) = planes[gi];
                // The step's own rows land in the spare row, in-graph: the
                // operand is then `[cache rows; the new row]`, the same operand
                // the concat used to build, with no pass over the cache.
                let k_app = emit::h(ctx).kv_append(kp, k_row, 0, pre);
                let v_app = emit::h(ctx).kv_append(vp, v_row, 0, pre);
                (
                    emit::reshape(ctx, k_app, &[1, (pre + m) as i64, nkv as i64, hd as i64]),
                    emit::reshape(ctx, v_app, &[1, (pre + m) as i64, nkv as i64, hd as i64]),
                )
            } else {
                match kv_in {
                    Some((kp, vp)) => {
                        let kpq = kv_prefix(ctx, kp, gi, pre, nkv, hd);
                        let vpq = kv_prefix(ctx, vp, gi, pre, nkv, hd);
                        (
                            emit::h(ctx).concat_(vec![kpq, k4], 2),
                            emit::h(ctx).concat_(vec![vpq, v4], 2),
                        )
                    }
                    None => (k4, v4),
                }
            };
            // Causal over `[cache ; window]`: query `i` of the window reaches the
            // whole cache and the window up to itself. A fixed-capacity cache says
            // the same thing with a mask: the rows past the written ones are 0.
            let sh = if plane_in.is_some() {
                Shape::new(&[1, m, nh, hd], DType::F32)
            } else {
                Shape::new(&[1, nh, m, hd], DType::F32)
            };
            let a = if let Some(mk) = mask {
                emit::h(ctx).attention(q4, k_all, v_all, mk, nh, hd, sh)
            } else {
                emit::h(ctx).attention_kind(q4, k_all, v_all, nh, hd, MaskKind::Causal, sh)
            };
            let a = if plane_in.is_some() {
                emit::reshape(ctx, a, &[m as i64, (nh * hd) as i64])
            } else {
                let a = emit::h(ctx).transpose_(a, vec![0, 2, 1, 3]);
                emit::reshape(ctx, a, &[m as i64, (nh * hd) as i64])
            };
            let o = emit::linear(
                ctx,
                &FlowValue::new(a, Shape::new(&[m, nh * hd], DType::F32)),
                &format!("{p}.attn_output.weight"),
            )?;
            x = emit::add(ctx, x, o.hir_id());

            let gamma = emit::f32_param(ctx, model, &format!("{p}.ffn_norm.weight"), &[d])?;
            let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d]);
            let (gate, up) = if self.fused_gate_up {
                let parts = emit::linear_fused(
                    ctx,
                    &lc(h),
                    &format!("{p}.gate_up.fused"),
                    &[cfg.ffn, cfg.ffn],
                )?;
                (
                    FlowValue::new(parts[0], Shape::new(&[m, cfg.ffn], DType::F32)),
                    FlowValue::new(parts[1], Shape::new(&[m, cfg.ffn], DType::F32)),
                )
            } else {
                (
                    emit::linear(ctx, &lc(h), &format!("{p}.ffn_gate.weight"))?,
                    emit::linear(ctx, &lc(h), &format!("{p}.ffn_up.weight"))?,
                )
            };
            let act = emit::silu(ctx, gate.hir_id(), &[m, cfg.ffn]);
            let act = emit::mul(ctx, act, up.hir_id());
            let down = emit::linear(
                ctx,
                &FlowValue::new(act, Shape::new(&[m, cfg.ffn], DType::F32)),
                &format!("{p}.ffn_down.weight"),
            )?;
            x = emit::add(ctx, x, down.hir_id());

            // What the next window reads: this window's own rows, which the
            // caller appends to its cache. A plane cache publishes each layer's
            // row on its own, already in the cache's row layout.
            if let Some((k_row, v_row)) = rows {
                k_next.push(k_row);
                v_next.push(v_row);
            } else {
                k_next.push(kv_tail(ctx, k4, 0, out_rows, nkv, hd));
                v_next.push(kv_tail(ctx, v4, 0, out_rows, nkv, hd));
            }
        }
        if plane_in.is_some() {
            for (gi, id) in k_next.iter().enumerate() {
                ctx.publish_side_output(
                    format!("knew{gi}"),
                    &FlowValue::new(*id, Shape::new(&[m, hkv], DType::F32)),
                );
            }
            for (gi, id) in v_next.iter().enumerate() {
                ctx.publish_side_output(
                    format!("vnew{gi}"),
                    &FlowValue::new(*id, Shape::new(&[m, hkv], DType::F32)),
                );
            }
        } else {
            let packed = self.packed_feed && out_rows > 1;
            let kv_shape = if packed {
                Shape::new(&[1, out_rows * layers * hkv], DType::F32)
            } else if self.row_first && out_rows > 1 {
                Shape::new(&[out_rows, layers, hkv], DType::F32)
            } else {
                Shape::new(&[layers, out_rows, hkv], DType::F32)
            };
            let k_out = emit::h(ctx).concat_(k_next, 0);
            let v_out = emit::h(ctx).concat_(v_next, 0);
            // A multi-row row-first cache hands each row's whole layer width to one
            // feed; a single-row step is already contiguous either way. `kv_tail` is
            // `[1, 1, rows, hkv]`, so the transpose is `[rows, layers, 1, hkv]`.
            let (k_out, v_out) = if self.row_first && out_rows > 1 {
                let k = emit::h(ctx).transpose_(k_out, vec![2, 0, 1, 3]);
                let v = emit::h(ctx).transpose_(v_out, vec![2, 0, 1, 3]);
                (
                    emit::reshape(ctx, k, &[out_rows as i64, layers as i64, hkv as i64]),
                    emit::reshape(ctx, v, &[out_rows as i64, layers as i64, hkv as i64]),
                )
            } else {
                (k_out, v_out)
            };
            let (k_out, v_out) = if packed {
                (
                    emit::reshape(ctx, k_out, &[1, (out_rows * layers * hkv) as i64]),
                    emit::reshape(ctx, v_out, &[1, (out_rows * layers * hkv) as i64]),
                )
            } else {
                (k_out, v_out)
            };
            ctx.publish_side_output("kpre", &FlowValue::new(k_out, kv_shape.clone()));
            ctx.publish_side_output("vpre", &FlowValue::new(v_out, kv_shape));
        }

        // `output_norm` closes the *body*, so only the group that ends at the
        // file's last block applies it: a split body's earlier groups hand on the
        // raw residual stream, which is what the next group's first block reads.
        let x = if self.layers.end == cfg.layers {
            let gamma = emit::f32_param(ctx, model, "output_norm.weight", &[d])?;
            emit::rms_norm(ctx, x, gamma, eps, &[m, d])
        } else {
            x
        };
        let sh = Shape::new(&[m, d], DType::F32);
        // The body's own head, when asked for: the vocabulary projection on this
        // row, published beside the cache rows.
        if self.head {
            let (name, vocab) = match cfg.head {
                Head::Separate { name, vocab } => (name, vocab),
                Head::Tied { table, vocab } => (table, vocab),
            };
            // Only the last row's next-token logits are read. Running the head on
            // every prompt row makes a `[prompt, vocab]` transient that is larger
            // than the decoder's own cache on a long prompt.
            let (xh, hs) = if m > 1 {
                (
                    emit::narrow(ctx, x, 0, m - 1, 1),
                    Shape::new(&[1, d], DType::F32),
                )
            } else {
                (x, Shape::new(&[1, d], DType::F32))
            };
            let out = emit::linear(ctx, &FlowValue::new(xh, hs), name)?;
            ctx.publish_side_output(
                "logits",
                &FlowValue::new(out.hir_id(), Shape::new(&[1, vocab], DType::F32)),
            );
        }
        Ok((
            FlowValue::new(x, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The body on a **growing** cache with the file's own head: `m` prompt rows,
/// `prefix` rows already carried, logits and the new cache rows out. This is the
/// recognition/synthesis prompt prefill.
pub fn body_new_rows_head_chain(
    model: Arc<Model>,
    m: usize,
    prefix: usize,
    head: bool,
) -> Result<BuiltModel> {
    let cfg = model.config().clone();
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("body_rows_{m}_p{prefix}_h{head}"))
        .with_profile(profile())
        .input("x", Shape::new(&[m, cfg.d], DType::F32))
        .layer_stage(BodyStateStage {
            model,
            m,
            prefix,
            layers: 0..cfg.layers,
            masked: false,
            row_first: false,
            head,
            plane_cache: false,
            packed_feed: false,
            fused: false,
            fused_gate_up: false,
        })
        .output("hidden");
    flow.build(&mut weights).map_err(Into::into)
}

/// The recognition projector alone: `[tokens, clip.dim]` features → the body's
/// width. The prompt mixes these with the text token embeddings.
pub fn projector_chain(pair: Arc<Pair>, tokens: usize) -> Result<BuiltModel> {
    let dim = pair.sound.clip().dim;
    let mut weights = PairWeights::new(pair.body.clone(), pair.sound.clone());
    let flow = ModelFlow::new(format!("projector_{tokens}"))
        .with_profile(profile())
        .input("features", Shape::new(&[tokens, dim], DType::F32))
        .layer_stage(ProjectorStage {
            model: pair.sound.clone(),
            tokens,
        })
        .output("x");
    flow.build(&mut weights).map_err(Into::into)
}

/// The whole body with a plane cache: one `[capacity + 1, hkv]` input per
/// layer, already in the attention's row layout, the step's own row appended
/// in-graph. The synthesis body's and recognition decode's fixed-capacity loops
/// are the callers.
pub fn body_capacity_chain_planes(model: Arc<Model>, capacity: usize) -> Result<BuiltModel> {
    let cfg = model.config().clone();
    let mut weights = LoaderWeights::new(model.clone());
    let mut fused = true;
    let mut fused_gate_up = true;
    for i in 0..cfg.layers {
        let p = format!("blk.{i}");
        fused &= weights.fuse_packed(
            &format!("{p}.qkv.fused"),
            &[
                &format!("{p}.attn_q.weight"),
                &format!("{p}.attn_k.weight"),
                &format!("{p}.attn_v.weight"),
            ],
        )?;
        fused_gate_up &= weights.fuse_packed(
            &format!("{p}.gate_up.fused"),
            &[
                &format!("{p}.ffn_gate.weight"),
                &format!("{p}.ffn_up.weight"),
            ],
        )?;
    }
    let flow = ModelFlow::new(format!("body_capacity_planes_{capacity}"))
        .with_profile(profile())
        .input("x", Shape::new(&[1, cfg.d], DType::F32))
        .layer_stage(BodyStateStage {
            model,
            m: 1,
            prefix: capacity,
            layers: 0..cfg.layers,
            masked: true,
            row_first: true,
            head: true,
            plane_cache: true,
            packed_feed: false,
            fused,
            fused_gate_up,
        })
        .output("hidden");
    flow.build(&mut weights).map_err(Into::into)
}

/// How many resident moves one plane-cache step collects: one K row and one V
/// row per layer. `resident_cache_planes` registers exactly this many, and
/// `run_resident_planes` folds all of them with one device-side copy — the
/// feed is one submission whatever the layer count.
pub fn plane_feed_moves(layers: usize) -> usize {
    2 * layers
}

/// The plane cache's output order is `hidden`, one K row and one V row per
/// layer, then the logits. A talker step reads the hidden row (its next frame's
/// input) and the logits; a recognition decode reads only the logits, because its
/// next input is the sampled token's embedding rather than this row — it passes
/// [`plane_step_logits`] alone. The `2 · layers` cache rows are fed device-side
/// rather than returned to the host, so one host return carries the step's answer.
pub fn plane_step_reads(layers: usize) -> [usize; 2] {
    [0, plane_step_logits(layers)]
}
/// Where the logits sit in a plane-cache step's outputs — the one read a step that
/// does not continue from its own hidden row needs.
pub fn plane_step_logits(layers: usize) -> usize {
    1 + plane_feed_moves(layers)
}

/// The full output count of a plane-cache body: the hidden row, the `2 · layers`
/// cache rows the feed consumes, and the logits.
pub fn plane_step_outputs(layers: usize) -> usize {
    plane_feed_moves(layers) + 2
}

/// A talker plane cache seeded from its prompt: one `[capacity + 1, hkv]` K and
/// V plane per layer — the attention's row layout already — with the prefill's
/// `[layers, prompt_rows, hkv]` rows scattered in, and a mask with the prompt's
/// rows and the spare query slot at `capacity` set.
pub struct PlaneCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub mask: Vec<f32>,
}

/// Build the talker's planes and mask before its first step. The prompt's rows
/// come from the prefill's layer-major output, one row at a time; every step
/// after the prefill writes one row of its own through the registered feed.
pub fn seed_plane_cache(
    layers: usize,
    capacity: usize,
    prompt_rows: usize,
    hkv: usize,
    k_rows: &[f32],
    v_rows: &[f32],
) -> PlaneCache {
    let mut k: Vec<Vec<f32>> = (0..layers)
        .map(|_| vec![0f32; (capacity + 1) * hkv])
        .collect();
    let mut v: Vec<Vec<f32>> = (0..layers)
        .map(|_| vec![0f32; (capacity + 1) * hkv])
        .collect();
    for l in 0..layers {
        for r in 0..prompt_rows {
            let dst = r * hkv;
            let src = (l * prompt_rows + r) * hkv;
            k[l][dst..dst + hkv].copy_from_slice(&k_rows[src..src + hkv]);
            v[l][dst..dst + hkv].copy_from_slice(&v_rows[src..src + hkv]);
        }
    }
    let mut mask = vec![0f32; capacity + 1];
    for r in 0..prompt_rows {
        mask[r] = 1.0;
    }
    mask[capacity] = 1.0;
    PlaneCache { k, v, mask }
}

// ── the TTS speaker encoder (ECAPA-TDNN, the `mmproj` voice half) ──

pub struct SpeakerStage {
    pub model: Arc<Model>,
    pub frames: usize,
    pub out: usize,
}

fn speaker_conv(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    name: &str,
    x: HirNodeId,
    c_out: usize,
    k: usize,
    dilation: usize,
    t: usize,
) -> Result<HirNodeId> {
    let pad = dilation * (k - 1) / 2;
    let x = emit::pad_axis(ctx, x, 4, 3, pad, pad, PadMode::Reflect);
    let w = emit::conv_param(ctx, model, &format!("{name}.weight"))?;
    let y = emit::conv2d_dilated(
        ctx,
        x,
        w,
        [1, k],
        [1, 1],
        [0, 0],
        [1, dilation],
        1,
        Shape::new(&[1, c_out, 1, t], DType::F32),
    );
    let b = emit::conv_bias(ctx, model, &format!("{name}.bias"), c_out)?;
    Ok(emit::add(ctx, y, b))
}

impl LayerStage for SpeakerStage {
    fn name(&self) -> &str {
        "speaker_encoder"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let t = self.frames;
        let relu = |ctx: &mut FlowCtx<'_>, x, c| emit::act(ctx, Activation::Relu, x, &[1, c, 1, t]);

        let mut x = speaker_conv(ctx, model, "a.conv1d.0", input.hir_id(), 512, 5, 1, t)?;
        x = relu(ctx, x, 512);
        let mut block_out = Vec::new();
        for (bi, dilation) in [2usize, 3, 4].into_iter().enumerate() {
            let p = format!("a.blk.{}", bi + 1);
            let res = x;
            let mut h = speaker_conv(ctx, model, &format!("{p}.conv_pw1"), x, 512, 1, 1, t)?;
            h = relu(ctx, h, 512);
            let mut parts = Vec::new();
            let chunk = 64usize;
            for i in 0..8 {
                let part = emit::narrow(ctx, h, 1, i * chunk, chunk);
                if i == 0 {
                    parts.push(part);
                } else {
                    let part = if i == 1 {
                        part
                    } else {
                        emit::add(ctx, part, parts[i - 1])
                    };
                    let part = speaker_conv(
                        ctx,
                        model,
                        &format!("{p}.res2.{}", i - 1),
                        part,
                        chunk,
                        3,
                        dilation,
                        t,
                    )?;
                    parts.push(relu(ctx, part, chunk));
                }
            }
            let h = emit::h(ctx).concat_(parts, 1);
            let mut h = speaker_conv(ctx, model, &format!("{p}.conv_pw2"), h, 512, 1, 1, t)?;
            h = relu(ctx, h, 512);
            // Squeeze-excitation: mean over time, 512→128→512, sigmoid gate.
            let mean = emit::h(ctx).mean(h, vec![2, 3], true);
            let se = speaker_conv(ctx, model, &format!("{p}.se_conv1"), mean, 128, 1, 1, 1)?;
            let se = emit::act(ctx, Activation::Relu, se, &[1, 128, 1, 1]);
            let se = speaker_conv(ctx, model, &format!("{p}.se_conv2"), se, 512, 1, 1, 1)?;
            let se = emit::act(ctx, Activation::Sigmoid, se, &[1, 512, 1, 1]);
            let se = emit::expand(ctx, se, &[1, 512, 1, t]);
            let h = emit::mul(ctx, h, se);
            x = emit::add(ctx, res, h);
            block_out.push(x);
        }
        let x = emit::h(ctx).concat_(block_out, 1);
        let x = speaker_conv(ctx, model, "a.conv_out", x, 1536, 1, 1, t)?;
        let x = relu(ctx, x, 1536);

        // Attentive statistics pooling.
        let mean = emit::h(ctx).mean(x, vec![2, 3], true);
        let sq = emit::mul(ctx, x, x);
        let ex2 = emit::h(ctx).mean(sq, vec![2, 3], true);
        let mm = emit::mul(ctx, mean, mean);
        let var = emit::sub(ctx, ex2, mm);
        let eps = emit::constant(ctx, &[1, 1536, 1, 1], 1e-8);
        let var = emit::add(ctx, var, eps);
        let std = emit::h(ctx).sqrt(var);
        let mean_b = emit::expand(ctx, mean, &[1, 1536, 1, t]);
        let std_b = emit::expand(ctx, std, &[1, 1536, 1, t]);
        let cat = emit::h(ctx).concat_(vec![x, mean_b, std_b], 1);
        let attn = speaker_conv(ctx, model, "a.asp_tdnn", cat, 128, 1, 1, t)?;
        let attn = relu(ctx, attn, 128);
        let attn = emit::act(ctx, Activation::Tanh, attn, &[1, 128, 1, t]);
        let attn = speaker_conv(ctx, model, "a.asp_attn", attn, 1536, 1, 1, t)?;
        let attn = emit::h(ctx).add_node(
            Op::Softmax { axis: 3 },
            vec![attn],
            Shape::new(&[1, 1536, 1, t], DType::F32),
        );
        let weighted = emit::mul(ctx, x, attn);
        let wmean = emit::h(ctx).sum(weighted, vec![3], true);
        let mean_b = emit::expand(ctx, wmean, &[1, 1536, 1, t]);
        let diff = emit::sub(ctx, x, mean_b);
        let diff2 = emit::mul(ctx, diff, diff);
        let weighted = emit::mul(ctx, diff2, attn);
        let wvar = emit::h(ctx).sum(weighted, vec![3], true);
        let wstd = emit::h(ctx).sqrt(wvar);
        let stats = emit::h(ctx).concat_(vec![wmean, wstd], 1);
        let out = self.out;
        let spk = speaker_conv(ctx, model, "mm.a.fc", stats, out, 1, 1, 1)?;
        let spk = emit::reshape(ctx, spk, &[1, out as i64]);
        let sh = Shape::new(&[1, out], DType::F32);
        Ok((
            FlowValue::new(spk, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

pub fn speaker_encoder(model: Arc<Model>, frames: usize, out: usize) -> Result<BuiltModel> {
    let mut weights = LoaderWeights::new(model.clone());
    let flow = ModelFlow::new(format!("speaker_{frames}"))
        .with_profile(profile())
        .input("mel", Shape::new(&[1, 128, 1, frames], DType::F32))
        .layer_stage(SpeakerStage { model, frames, out })
        .output("spk");
    flow.build(&mut weights).map_err(Into::into)
}

// ── the TTS codec (synthesis's sound stage): zeros + ordinary conv ──

/// The waveform stack of the synthesis `mmproj`: two stride-2 transposed convs
/// with ConvNeXt blocks (12.5 → 50 Hz), `entry` (1024 → 1536), four DAC blocks
/// at strides 8/5/4/3 (50 Hz → 24 kHz, 1536 → 96 channels), `post_snake` and
/// `post_conv`.
///
/// The transposed convs are the "insert zeros + ordinary conv" form.
///
/// [`DacStage::stages`] is which of the eight stages this graph carries: the
/// whole stack (the default) or one group of consecutive stages, so the same
/// stage builds a **split** codec — the arena is a graph's, and this is the
/// graph whose plan is over the binding at its real window.
pub struct DacStage {
    pub model: Arc<Model>,
    pub frames: usize,
    /// Stages of [`DAC_STAGES`] this graph runs, in order.
    pub stages: std::ops::Range<usize>,
}

/// The codec's eight stages: `up.blk.0`, `up.blk.1`, `dac.entry`,
/// `dac.blk.0…3`, `post` (snake + conv).
pub const DAC_STAGES: usize = 8;

/// The four DAC upsamplers, `(kernel, stride, out channels)`, at stages
/// [`DAC_UPSAMPLE_STAGE`]`..`[`DAC_UPSAMPLE_STAGE`]` + DAC_UPSAMPLE.len()`. Their
/// strides are [`dac_stage_io`]'s length ratios and together with the two leading
/// 2× they make [`samples_per_frame`]; `tests/sound_transpose_form.rs` holds the
/// table to that geometry.
pub const DAC_UPSAMPLE: [(usize, usize, usize); 4] =
    [(16, 8, 768), (10, 5, 384), (8, 4, 192), (6, 3, 96)];

/// The stage [`DAC_UPSAMPLE`]'s first upsampler runs at — the two `up.blk` stages
/// and the entry convolution before it.
pub const DAC_UPSAMPLE_STAGE: usize = 3;

/// `(channels, length)` stage `stage` reads when the codec is given `frames`
/// frames — the shape a group starting at that stage declares as its input, and
/// the shape a group ending at that stage hands on. Stage [`DAC_STAGES`] is the
/// waveform itself: one channel at [`samples_per_frame`] samples per frame.
pub fn dac_stage_io(stage: usize, frames: usize) -> (usize, usize) {
    let l = match stage {
        0 => frames,
        1 => frames * 2,
        2 | 3 => frames * 4,
        4 => frames * 32,
        5 => frames * 160,
        6 => frames * 640,
        _ => frames * 1920,
    };
    let ch = match stage {
        0..=2 => 1024,
        3 => 1536,
        4 => 768,
        5 => 384,
        6 => 192,
        _ => 96,
    };
    (ch, l)
}

impl DacStage {
    fn conv_conv1d(
        &self,
        ctx: &mut FlowCtx<'_>,
        x: HirNodeId,
        name: &str,
        cout: usize,
        l: usize,
        k: usize,
        pad: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<HirNodeId> {
        // The kernel keeps the file's own width; the op's native dilation field
        // walks it at the reference's spacing instead of a zero-spread kernel.
        let w = emit::conv_param(ctx, &self.model, name)?;
        // The decoder is causal: the reference prepends `k-1` left samples and
        // never pads on the right. The old symmetric `pad` changed every sample
        // after the first kernel, not just the edge. It is the conv's own padding
        // rather than a materialised zero pad in front of it: every route that reads
        // this conv reads the padding out of its own gather, so the copy is a whole
        // activation pass per conv and nothing more.
        //
        // The declared extent is a deliberate truncation: the right pad of that
        // symmetric field is never emitted, and every route takes the extent from
        // this node's shape and reads `padding` as a left origin.
        let nchw = emit::conv2d_dilated(
            ctx,
            x,
            w,
            [1, k],
            [1, 1],
            [0, pad],
            [1, dilation],
            groups,
            Shape::new(&[1, cout, 1, l], DType::F32),
        );
        Ok(nchw)
    }

    /// A 1-D transposed conv either way. Returns the new length.
    #[allow(clippy::too_many_arguments)]
    fn transpose_conv1d(
        &self,
        ctx: &mut FlowCtx<'_>,
        x: HirNodeId,
        name: &str,
        (_cin, cout): (usize, usize),
        l: usize,
        _k: usize,
        s: usize,
    ) -> Result<(HirNodeId, usize)> {
        // The reference trims only the right `k - s` samples; the op's own
        // transposed output is that full span, and the first `l · s` samples are
        // the causal answer.
        let h_out = l * s;
        let (w, _, _, k) = emit::conv_transpose_param(ctx, &self.model, name)?;
        let full = (l - 1) * s + k;
        let y = emit::h(ctx).conv_transpose2d(
            x,
            w,
            [1, k],
            [1, s],
            [0, 0],
            [1, 1],
            [0, 0],
            1,
            Shape::new(&[1, cout, 1, full], DType::F32),
        );
        Ok((emit::narrow(ctx, y, 3, 0, h_out), h_out))
    }
}

impl LayerStage for DacStage {
    fn name(&self) -> &str {
        "tts_codec_dac"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        // A group of stages starts where its first stage does: the running length
        // and channel count are the ones that stage reads, not the codec's input.
        let (mut ch, mut l) = dac_stage_io(self.stages.start, self.frames);
        let mut cur = input.hir_id();

        // Two transposed convs with stride 2 doubling the rate, each followed by
        // a ConvNeXt block (depthwise k=7 → LayerNorm2d → pw1 → GELU → pw2 →
        // gamma).
        for i in 0..2 {
            if !self.stages.contains(&i) {
                continue;
            }
            let p = format!("a.gen.wav.up.blk.{i}");
            let (cur_new, l_new) = self.transpose_conv1d(
                ctx,
                cur,
                &format!("{p}.conv.weight"),
                (1024, 1024),
                l,
                2,
                2,
            )?;
            cur = cur_new;
            l = l_new;
            let b = emit::conv_bias(ctx, model, &format!("{p}.conv.bias"), 1024)?;
            cur = emit::add(ctx, cur, b);

            let dw = emit::conv_param(ctx, model, &format!("{p}.dwconv.weight"))?;
            let dwb = emit::conv_bias(ctx, model, &format!("{p}.dwconv.bias"), 1024)?;
            let cur_pad = emit::pad_axis(ctx, cur, 4, 3, 6, 0, rlx_ir::op::PadMode::Constant(0.0));
            let d = emit::h(ctx).conv2d(
                cur_pad,
                dw,
                [1, 7],
                [1, 1],
                [0, 0],
                1024,
                Shape::new(&[1, 1024, 1, l], DType::F32),
            );
            let d = emit::add(ctx, d, dwb);
            let nw = emit::f32_param(ctx, model, &format!("{p}.norm.weight"), &[1024])?;
            let nb = emit::f32_param(ctx, model, &format!("{p}.norm.bias"), &[1024])?;
            let d = emit::h(ctx).layer_norm2d(d, nw, nb, 1e-6);
            let lc = emit::to_lc(ctx, d, 1024, l);
            let w = emit::linear(
                ctx,
                &FlowValue::new(lc, Shape::new(&[l, 1024], DType::F32)),
                &format!("{p}.pw1.weight"),
            )?;
            let h1 = w.hir_id();
            let b1 = emit::f32_param(ctx, model, &format!("{p}.pw1.bias"), &[1, 4096])?;
            let h1 = emit::add(ctx, h1, b1);
            let h1 = emit::gelu(ctx, h1, &[l, 4096]);
            let w = emit::linear(
                ctx,
                &FlowValue::new(h1, Shape::new(&[l, 4096], DType::F32)),
                &format!("{p}.pw2.weight"),
            )?;
            let h2 = w.hir_id();
            let b2 = emit::f32_param(ctx, model, &format!("{p}.pw2.bias"), &[1, 1024])?;
            let h2 = emit::add(ctx, h2, b2);
            let gam = emit::f32_param(ctx, model, &format!("{p}.gamma"), &[1, 1024, 1, 1])?;
            // `[l, 1024]` → `[1, 1024, 1, l]`: the **transpose is the whole
            // point**, because the two projections above work in the `[L, C]`
            // layout `to_lc` produced and a bare `Reshape` would read the flat
            // buffer as channel-major, mixing channels with positions. This is
            // the same conversion `LcToNchwStage` makes on the codec's input
            // (a missing transpose here is a real defect: the graph's
            // own numbers leave the f64 reference by the block's whole peak).
            let h2 = emit::h(ctx).transpose_(h2, vec![1, 0]);
            let h2 = emit::reshape(ctx, h2, &[1, 1024, 1, l as i64]);
            let h2 = emit::mul(ctx, h2, gam);
            cur = emit::add(ctx, cur, h2);
        }

        // entry: 1024 → 1536, kernel 7.
        if self.stages.contains(&2) {
            let e =
                self.conv_conv1d(ctx, cur, "a.gen.wav.dac.entry.weight", 1536, l, 7, 6, 1, 1)?;
            let eb = emit::conv_bias(ctx, model, "a.gen.wav.dac.entry.bias", 1536)?;
            cur = emit::add(ctx, e, eb);
            ch = 1536;
        }

        for (i, &(k, s, c_out)) in DAC_UPSAMPLE.iter().enumerate() {
            if !self.stages.contains(&(DAC_UPSAMPLE_STAGE + i)) {
                continue;
            }
            let p = format!("a.gen.wav.dac.blk.{i}");
            let a = emit::f32_param(ctx, model, &format!("{p}.snake.alpha"), &[1, ch, 1, 1])?;
            let b = emit::f32_param(ctx, model, &format!("{p}.snake.beta"), &[1, ch, 1, 1])?;
            cur = emit::snake_beta(ctx, cur, a, b, ch, l);
            let (y, l_new) =
                self.transpose_conv1d(ctx, cur, &format!("{p}.conv.weight"), (ch, c_out), l, k, s)?;
            cur = y;
            l = l_new;
            let bias = emit::conv_bias(ctx, model, &format!("{p}.conv.bias"), c_out)?;
            cur = emit::add(ctx, cur, bias);

            for (r, dilation) in [1usize, 3, 9].into_iter().enumerate() {
                let rp = format!("{p}.res.{r}");
                let a1 =
                    emit::f32_param(ctx, model, &format!("{rp}.act1.alpha"), &[1, c_out, 1, 1])?;
                let b1 =
                    emit::f32_param(ctx, model, &format!("{rp}.act1.beta"), &[1, c_out, 1, 1])?;
                let a2 =
                    emit::f32_param(ctx, model, &format!("{rp}.act2.alpha"), &[1, c_out, 1, 1])?;
                let b2 =
                    emit::f32_param(ctx, model, &format!("{rp}.act2.beta"), &[1, c_out, 1, 1])?;
                let h1 = emit::snake_beta(ctx, cur, a1, b1, c_out, l);
                let c1 = self.conv_conv1d(
                    ctx,
                    h1,
                    &format!("{rp}.conv1.weight"),
                    c_out,
                    l,
                    7,
                    6 * dilation,
                    1,
                    dilation,
                )?;
                let bb = emit::conv_bias(ctx, model, &format!("{rp}.conv1.bias"), c_out)?;
                let h1 = emit::add(ctx, c1, bb);
                let h2 = emit::snake_beta(ctx, h1, a2, b2, c_out, l);
                let c2 =
                    self.conv_conv1d(ctx, h2, &format!("{rp}.conv2.weight"), c_out, l, 1, 0, 1, 1)?;
                let bb = emit::conv_bias(ctx, model, &format!("{rp}.conv2.bias"), c_out)?;
                let h2 = emit::add(ctx, c2, bb);
                cur = emit::add(ctx, cur, h2);
            }
            ch = c_out;
        }

        // The group ends before `post`: it hands its `[1, ch, 1, l]` tensor on.
        if !self.stages.contains(&7) {
            let sh = Shape::new(&[1, ch, 1, l], DType::F32);
            return Ok((
                FlowValue::new(cur, sh.clone()),
                StageArtifacts::hidden_only(sh),
            ));
        }

        let a = emit::f32_param(ctx, model, "a.gen.wav.dac.post_snake.alpha", &[1, ch, 1, 1])?;
        let b = emit::f32_param(ctx, model, "a.gen.wav.dac.post_snake.beta", &[1, ch, 1, 1])?;
        cur = emit::snake_beta(ctx, cur, a, b, ch, l);
        let out = self.conv_conv1d(ctx, cur, "a.gen.wav.dac.post_conv.weight", 1, l, 7, 6, 1, 1)?;
        let pb = emit::f32_param(ctx, model, "a.gen.wav.dac.post_conv.bias", &[1, 1, 1, 1])?;
        let out = emit::add(ctx, out, pb);
        let out = emit::h(ctx).add_node(
            rlx_ir::Op::Clamp {
                min: -1.0,
                max: 1.0,
            },
            vec![out],
            Shape::new(&[1, 1, 1, l], DType::F32),
        );
        let sh = Shape::new(&[1, 1, 1, l], DType::F32);
        Ok((
            FlowValue::new(out, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

// ── the code predictor: the talker's 15 acoustic codes ─────────────

/// The talker's row, and the code embeddings with it, into the predictor's width:
/// through the file's own `proj_in` where the file carries one, and unchanged
/// where it does not (see [`Model::code_proj_widths`]).
fn code_proj_in_node(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    seq: HirNodeId,
    rows: usize,
    d_talker: usize,
    d_pred: usize,
) -> Result<HirNodeId> {
    if !model.has_code_proj_in() {
        return Ok(seq);
    }
    let proj = emit::linear(
        ctx,
        &FlowValue::new(seq, Shape::new(&[rows, d_talker], DType::F32)),
        "a.gen.code.proj_in.weight",
    )?;
    let b = emit::f32_param(ctx, model, "a.gen.code.proj_in.bias", &[1, d_pred])?;
    Ok(emit::add(ctx, proj.hir_id(), b))
}

/// The predictor's own step count — the file's LM-head stack depth.
pub fn code_predictor_steps(model: &Model) -> Result<usize> {
    Ok(model.plan("a.gen.code.head.weight")?.ggml_shape[2])
}

/// The draw seeds one predictor frame's `steps` samplers read: level `s` uses
/// `seed + draw_index + s`, the seed the reference walk hands `host::draw`. Empty
/// when the sampler lowered to `Argmax` — a greedy walk reads no seed. One helper
/// for both walks (the crate's own path and the measuring stand), because the two
/// must agree bit-exactly or the pinned waveform moves.
pub fn predictor_draw_seeds(
    seed: u64,
    draw_index: u64,
    steps: usize,
    top_k: usize,
    temperature: f32,
) -> Vec<u64> {
    if top_k == 1 || temperature <= 0.0 {
        return Vec::new();
    }
    (0..steps as u64).map(|k| seed + draw_index + k).collect()
}

/// What a step's cache is made of: `(blocks, kv_heads·head_dim, head_dim)`.
/// The step's own RoPE tables are the host's, at the row's position.
pub fn code_predictor_cache(model: &Model) -> Result<(usize, usize, usize)> {
    let c = model.clip();
    let head_dim = model
        .plan("a.gen.code.blk.0.attn_q_norm.weight")?
        .ggml_shape[0];
    Ok((
        c.gen_layers.unwrap_or(0),
        c.gen_kv_heads.unwrap_or(0) * head_dim,
        head_dim,
    ))
}

/// The code predictor's block stack for one row on a carried cache — the
/// arithmetic the one-submission frame walk spells once per level. `x` is the
/// row after `proj_in`; each block's key/value cache is its own preallocated
/// `[capacity, hkv]` plane, and the level's row is appended to it in-graph.
struct PredictorBlocks {
    m: usize,
    d_pred: usize,
    nh: usize,
    nkv: usize,
    ffn: usize,
    eps: f32,
    blocks: usize,
    head_dim: usize,
    /// One layer's key/value width, `nkv · head_dim`.
    hkv: usize,
    /// Rows in one block's key/value plane: one per level of the walk.
    capacity: usize,
}

/// One row through the predictor's block stack, appending it to each block's
/// key/value plane. The plane is the attention's row layout already
/// (`[rows, heads · head_dim]`), so no narrow, concat or transpose moves the
/// cache: a level writes one row per block through `Op::KvAppend` (the same
/// in-place append the body's plane cache uses) and reads the `[..pos + 1]`
/// prefix the append returns. The mask selects every one of those rows, so the
/// valid keys are the same past rows plus this level's own row, in the same
/// order, and the levels' rows never have to be re-collected.
///
/// Level 0 seeds each plane: `[its own row; zeros]` is one `Concat` per block,
/// and it is what makes the walk's first level live. Reading a plane later
/// through the append that wrote it is a buffer mutation, not a dataflow edge —
/// on its own, level 0's block stack would be unreachable from every output
/// (its hidden row is only the input to the next level's cache) and dead-code
/// elimination would drop the arithmetic that produced rows 0. Seeding the
/// plane with that row keeps it an ordinary value every later append reads.
/// Neither the cache nor the projections are permuted.
#[allow(clippy::too_many_arguments)]
fn emit_predictor_blocks(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    g: &PredictorBlocks,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    mask: HirNodeId,
    k_planes: &mut [Option<HirNodeId>],
    v_planes: &mut [Option<HirNodeId>],
    pos: usize,
    fused: bool,
) -> Result<HirNodeId> {
    let PredictorBlocks {
        m,
        d_pred,
        nh,
        nkv,
        ffn,
        eps,
        blocks,
        head_dim,
        hkv,
        capacity,
    } = *g;
    let mut x = x;
    for i in 0..blocks {
        let p = format!("a.gen.code.blk.{i}");
        let lc = |x: HirNodeId| FlowValue::new(x, Shape::new(&[m, d_pred], DType::F32));
        let gamma = emit::f32_param(ctx, model, &format!("{p}.ln1.weight"), &[d_pred])?;
        let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d_pred]);
        let [q, k, v] = if fused {
            let parts = emit::linear_fused(
                ctx,
                &lc(h),
                &format!("{p}.qkv.fused"),
                &[nh * head_dim, hkv, hkv],
            )?;
            [parts[0], parts[1], parts[2]]
        } else {
            emit::qkv(
                ctx,
                model,
                h,
                m,
                d_pred,
                [
                    &format!("{p}.attn_q.weight"),
                    &format!("{p}.attn_k.weight"),
                    &format!("{p}.attn_v.weight"),
                ],
            )?
        };
        let q3 = vec![1, m as i64, (nh * head_dim) as i64];
        let kv3 = vec![1, m as i64, hkv as i64];
        let r = emit::reshape(ctx, q, &[(m * nh) as i64, head_dim as i64]);
        let gq = emit::f32_param(ctx, model, &format!("{p}.attn_q_norm.weight"), &[head_dim])?;
        let r = emit::rms_norm(ctx, r, gq, eps, &[m * nh, head_dim]);
        let q = emit::reshape(ctx, r, &q3);
        let r = emit::reshape(ctx, k, &[(m * nkv) as i64, head_dim as i64]);
        let gk = emit::f32_param(ctx, model, &format!("{p}.attn_k_norm.weight"), &[head_dim])?;
        let r = emit::rms_norm(ctx, r, gk, eps, &[m * nkv, head_dim]);
        let k = emit::reshape(ctx, r, &kv3);
        let q = emit::h(ctx).rope(q, cos, sin, head_dim);
        let k = emit::h(ctx).rope(k, cos, sin, head_dim);
        let q4 = emit::reshape(ctx, q, &[1, m as i64, nh as i64, head_dim as i64]);
        let k4 = emit::reshape(ctx, k, &[1, m as i64, nkv as i64, head_dim as i64]);
        let v4 = emit::reshape(ctx, v, &[1, m as i64, nkv as i64, head_dim as i64]);
        // Rows-first is a declaration for one row, not a move. `attention_geom`
        // keys the layout off Q's axis 1, so the hazard is a single head: K/V
        // then read `[1, pos + 1, nkv, D]` with the sequence taken from dim 2.
        assert!(
            nh != 1,
            "rows-first attention keys the layout off Q's axis 1: one head would read \
             K/V with its sequence taken from dim 2"
        );
        // The level's own row lands in the block's plane at `pos`, and the
        // append's `[..pos + 1]` prefix is the attention operand: the past rows
        // are the rows earlier levels wrote, then this level's own row. One row
        // write per block per level replaces the narrow + concat the cache used
        // to be rebuilt with. Level 0 has no plane yet and seeds it instead.
        let k_row = emit::reshape(ctx, k4, &[m as i64, hkv as i64]);
        let v_row = emit::reshape(ctx, v4, &[m as i64, hkv as i64]);
        let k_src = match k_planes[i] {
            Some(plane) => emit::h(ctx).kv_append(plane, k_row, 0, pos),
            None => {
                assert_eq!(pos, 0, "a block's cache plane is seeded at level 0");
                let zeros = emit::constant(ctx, &[capacity - m, hkv], 0.0);
                let plane = emit::h(ctx).concat_(vec![k_row, zeros], 0);
                k_planes[i] = Some(plane);
                k_row
            }
        };
        let v_src = match v_planes[i] {
            Some(plane) => emit::h(ctx).kv_append(plane, v_row, 0, pos),
            None => {
                assert_eq!(pos, 0, "a block's cache plane is seeded at level 0");
                let zeros = emit::constant(ctx, &[capacity - m, hkv], 0.0);
                let plane = emit::h(ctx).concat_(vec![v_row, zeros], 0);
                v_planes[i] = Some(plane);
                v_row
            }
        };
        let k_all = emit::reshape(
            ctx,
            k_src,
            &[1, (pos + m) as i64, nkv as i64, head_dim as i64],
        );
        let v_all = emit::reshape(
            ctx,
            v_src,
            &[1, (pos + m) as i64, nkv as i64, head_dim as i64],
        );
        let sh = Shape::new(&[1, m, nh, head_dim], DType::F32);
        let a = emit::h(ctx).attention(q4, k_all, v_all, mask, nh, head_dim, sh);
        let a = emit::reshape(ctx, a, &[m as i64, (nh * head_dim) as i64]);
        let o = emit::linear(
            ctx,
            &FlowValue::new(a, Shape::new(&[m, nh * head_dim], DType::F32)),
            &format!("{p}.attn_out.weight"),
        )?;
        x = emit::add(ctx, x, o.hir_id());

        let gamma = emit::f32_param(ctx, model, &format!("{p}.ln2.weight"), &[d_pred])?;
        let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d_pred]);
        let (gate, up) = if fused {
            let parts =
                emit::linear_fused(ctx, &lc(h), &format!("{p}.gate_up.fused"), &[ffn, ffn])?;
            (
                FlowValue::new(parts[0], Shape::new(&[m, ffn], DType::F32)),
                FlowValue::new(parts[1], Shape::new(&[m, ffn], DType::F32)),
            )
        } else {
            (
                emit::linear(ctx, &lc(h), &format!("{p}.ffn_gate.weight"))?,
                emit::linear(ctx, &lc(h), &format!("{p}.ffn_up.weight"))?,
            )
        };
        let act = emit::silu(ctx, gate.hir_id(), &[m, ffn]);
        let act = emit::mul(ctx, act, up.hir_id());
        let down = emit::linear(
            ctx,
            &FlowValue::new(act, Shape::new(&[m, ffn], DType::F32)),
            &format!("{p}.ffn_down.weight"),
        )?;
        x = emit::add(ctx, x, down.hir_id());
    }
    Ok(x)
}

/// The stacked tensor the frame walk's per-level `#s-1` head slots slice — the
/// file's `[in, out, steps]` LM-head stack, one level per slot.
pub const PREDICTOR_HEAD_WEIGHT: &str = "a.gen.code.head.weight";

/// A constant F32 tensor from its values — the frame walk's per-level RoPE
/// tables and masks, and the gather's slot offset.
fn f32_values(ctx: &mut FlowCtx<'_>, dims: &[usize], values: &[f32]) -> HirNodeId {
    assert_eq!(values.len(), dims.iter().product::<usize>());
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    ctx.hir()
        .mir(Op::Constant { data }, vec![], Shape::new(dims, DType::F32))
}

/// One code-predictor frame as **one graph**: the sixteen-level walk that used to
/// be sixteen `run`s — level 0's head-less body, then every projecting level with
/// its own head — now emits every level into one schedule, draws each level's
/// code with `Op::Sample` on the device, and gathers the next level's row out of
/// the code table in the same graph. The cache never leaves the device: each
/// block keeps its own `[capacity, hkv]` plane, level 0 seeds it with its row
/// and every later level appends one row in place (`Op::KvAppend`), so a level's
/// attention reads the same rows the reference walk binds without rebuilding
/// the cache.
///
/// The walk is sequential by construction — level `s + 1` embeds the code level
/// `s` drew — so the single submit/wait is the only boundary a frame pays. A
/// draw's seed is a uniform word, not a pipeline constant: the caller rewrites
/// the `Sample` steps' seeds for each frame ([`Runner::set_sample_seeds`]), which
/// is why one compiled graph serves every frame and every request. The draw
/// reproduces [`crate::host::draw`] bit for bit; `sample_graph_one` pins that
/// arithmetic against it.
///
/// The engine's phase classifier marks `Sample` as epilogue, but this backend
/// schedules by graph node order and the runtime pipeline preserves it, so a
/// draw with a consumer after it follows that consumer's level; no phase edit is
/// needed for the mid-graph draw.
pub struct CodePredictorFrameStage {
    pub model: Arc<Model>,
    /// Rows the reference walk's cache holds — `steps + 1`.
    pub capacity: usize,
    /// Projecting levels: `steps` codes follow level 0's `code0`.
    pub steps: usize,
    pub fused: bool,
    pub sampler: SampleSpec,
}

impl LayerStage for CodePredictorFrameStage {
    fn name(&self) -> &str {
        "tts_code_predictor_frame"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let c = model.clip();
        let (d_talker, d_pred) = model.code_proj_widths()?;
        let head_dim = model
            .plan("a.gen.code.blk.0.attn_q_norm.weight")?
            .ggml_shape[0];
        let head_out = model.plan(PREDICTOR_HEAD_WEIGHT)?.ggml_shape[1];
        let (nh, nkv, ffn) = (
            c.gen_heads.unwrap_or(0),
            c.gen_kv_heads.unwrap_or(0),
            c.gen_ffn.unwrap_or(0),
        );
        let eps = c.gen_eps.unwrap_or(1e-6);
        let blocks = c.gen_layers.unwrap_or(0);
        let (m, hkv) = (1usize, nkv * head_dim);
        let geom = PredictorBlocks {
            m,
            d_pred,
            nh,
            nkv,
            ffn,
            eps,
            blocks,
            head_dim,
            hkv,
            capacity: self.capacity,
        };

        let hidden = emit::reshape(ctx, input.hir_id(), &[1, d_talker as i64]);
        let code0 = emit::h(ctx).input("code0_embed", Shape::new(&[1, d_talker], DType::F32));
        // The code table is one `[steps, vocab, d_talker]` stack; flattening the
        // first two axes makes a level's row a plain leading-axis gather, which
        // is the one embedding lookup this backend serves without a transpose.
        let p = model.plan("a.gen.code.embd.weight")?.clone();
        let (d_embd, vocab, slots) = (p.ggml_shape[0], p.ggml_shape[1], p.ggml_shape[2]);
        assert_eq!(
            d_embd, d_talker,
            "the code table's width must be the predictor's input width"
        );
        assert!(
            slots >= self.steps.saturating_sub(1),
            "the code table has {slots} levels, the walk needs {}",
            self.steps.saturating_sub(1)
        );
        // A level's row is `(slot, id)` into the flattened table, so an id past
        // the level's `vocab` rows lands on the next level's row: the head that
        // draws the id must be exactly as wide as one level.
        assert_eq!(
            head_out, vocab,
            "the LM head projects {head_out} rows, the code table has {vocab} rows per \
             level; a wider head would gather another level's row"
        );
        let table = emit::f32_param(
            ctx,
            model,
            "a.gen.code.embd.weight",
            &[slots * vocab, d_talker],
        )?;

        // Each block's key/value plane: level 0 seeds `[its row; zeros]` and
        // every later level appends one row into it, so the cache is never
        // copied or concatenated as the levels advance. The plane is carried
        // between levels as an ordinary value (see `emit_predictor_blocks`).
        let mut k_planes: Vec<Option<HirNodeId>> = vec![None; blocks];
        let mut v_planes: Vec<Option<HirNodeId>> = vec![None; blocks];
        let mut ids: Vec<HirNodeId> = Vec::with_capacity(self.steps);
        let mut logits1: Option<HirNodeId> = None;
        for s in 0..=self.steps {
            let row = if s == 0 {
                hidden
            } else if s == 1 {
                code0
            } else {
                // `s == 1` drew `ids[0]`, so level `s` embeds level `s - 1`'s
                // code from slot `s - 2`.
                let at = ((s - 2) * vocab) as f32;
                let at = f32_values(ctx, &[1], &[at]);
                let idx = emit::add(ctx, ids[s - 2], at);
                emit::gather(ctx, table, idx)
            };
            let (cos, sin) = crate::host::rope_tables_at(s, 1, head_dim, 1e6);
            let cos: Vec<f32> = cos.iter().map(|v| *v as f32).collect();
            let sin: Vec<f32> = sin.iter().map(|v| *v as f32).collect();
            let cos = f32_values(ctx, &[1, head_dim / 2], &cos);
            let sin = f32_values(ctx, &[1, head_dim / 2], &sin);
            // The plane already holds every row up to this level's own, so the
            // mask admits all of them; the operand's length is `s + 1`.
            let mask = f32_values(ctx, &[1, s + 1], &vec![1.0; s + 1]);
            let x = code_proj_in_node(ctx, model, row, m, d_talker, d_pred)?;
            let x = emit_predictor_blocks(
                ctx,
                model,
                &geom,
                x,
                cos,
                sin,
                mask,
                &mut k_planes,
                &mut v_planes,
                s,
                self.fused,
            )?;
            if s >= 1 {
                // The head projects the *normalised* hidden row the body returns.
                let gamma =
                    emit::f32_param(ctx, model, "a.gen.code.output_norm.weight", &[d_pred])?;
                let x = emit::rms_norm(ctx, x, gamma, eps, &[m, d_pred]);
                let key = format!("{PREDICTOR_HEAD_WEIGHT}#{}", s - 1);
                let logits = emit::linear(
                    ctx,
                    &FlowValue::new(x, Shape::new(&[1, d_pred], DType::F32)),
                    &key,
                )?;
                if s == 1 {
                    logits1 = Some(logits.hir_id());
                }
                let id = emit::h(ctx).add_node(
                    Op::Sample {
                        top_k: self.sampler.top_k,
                        top_p: self.sampler.top_p,
                        temperature: self.sampler.temperature,
                        seed: self.sampler.seed + (s - 1) as u64,
                    },
                    vec![logits.hir_id()],
                    Shape::new(&[1], DType::F32),
                );
                ids.push(id);
            }
        }
        let codes = emit::h(ctx).concat_(ids, 0);
        // Level 1's logits ride out as a side output: the stand's sampler check
        // reads that one row once, from the same head the draw consumed.
        if let Some(logits1) = logits1 {
            ctx.publish_side_output(
                "logits1",
                &FlowValue::new(logits1, Shape::new(&[1, head_out], DType::F32)),
            );
        }
        let sh = Shape::new(&[self.steps], DType::F32);
        Ok((
            FlowValue::new(codes, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// Build the one-graph frame walk: `[d_talker]` hidden row and `code0` row in,
/// `steps` drawn codes out. One [`Runner::set_sample_seeds`] + one run is a whole
/// frame, whatever the level count.
///
/// **Exclusion: the frame is one row, not `m`.** A frame graph carrying `m` frames
/// in flight would read the stack once instead of once a frame, and it is not kept:
/// the frames are not independent — frame `f`'s codes feed frame `f + 1`, so a
/// chunk of `m` frames is walked to a fixed point in `m` passes. The measurement
/// that closed the axis, and the passes it took, are in `MEASUREMENTS.md`'s
/// dismissed forms.
pub fn tts_code_predictor_frame(
    model: Arc<Model>,
    capacity: usize,
    sampler: SampleSpec,
) -> Result<BuiltModel> {
    let (d_talker, _) = model.code_proj_widths()?;
    let steps = code_predictor_steps(&model)?;
    assert_eq!(
        capacity,
        steps + 1,
        "the frame walk's cache is the file's `steps + 1` positions"
    );
    let mut weights = LoaderWeights::new(model.clone());
    let blocks = model.clip().gen_layers.unwrap_or(0);
    let mut fused = true;
    for i in 0..blocks {
        let p = format!("a.gen.code.blk.{i}");
        fused &= weights.fuse_packed(
            &format!("{p}.qkv.fused"),
            &[
                &format!("{p}.attn_q.weight"),
                &format!("{p}.attn_k.weight"),
                &format!("{p}.attn_v.weight"),
            ],
        )?;
        fused &= weights.fuse_packed(
            &format!("{p}.gate_up.fused"),
            &[
                &format!("{p}.ffn_gate.weight"),
                &format!("{p}.ffn_up.weight"),
            ],
        )?;
        if !fused {
            break;
        }
    }
    let flow = ModelFlow::new("tts_code_frame")
        .with_profile(profile())
        .input("hidden", Shape::new(&[d_talker], DType::F32))
        .layer_stage(CodePredictorFrameStage {
            model,
            capacity,
            steps,
            fused,
            sampler,
        })
        .output("codes");
    flow.build(&mut weights).map_err(Into::into)
}

// ── the wave transformer: the codebook hidden → the codec's width ──

/// The waveform part's transformer (`a.gen.wav.tfm.*`): a causal `pre_conv`
/// from the RVQ hidden up to 1024, down through `in_proj` to 512, eight blocks
/// with causal attention (window 72, RoPE 1e4, LayerScale on both branches) and
/// back up through `out_proj`.
///
/// The reference keeps the previous batch's 71 key/value frames, the last
/// `pre_conv` frames and a position counter between calls. This graph carries
/// the same state:
///   * `left` — the last `k − 1` frames before `pre_conv`;
///   * `kpre`/`vpre` — `[layers, prefix, heads·head_dim]` already RoPE'd;
///   * `cos`/`sin` — RoPE at the window's **global** position.
/// It publishes this window's `knew`/`vnew` (the caller keeps the last 71 rows)
/// and the `left` context for the next window. Without that state the first
/// window of a part is correct and every later one starts from zeros — exactly
/// the 72-row/330-row split.
pub struct WaveTfmStage {
    pub model: Arc<Model>,
    /// Frames in the window (the reference's 72).
    pub frames: usize,
    /// Attention heads — hardwired in the reference (`n_head = 16`), not in the
    /// file, so it is a parameter of the graph rather than a loaded shape.
    pub heads: usize,
    /// Rows carried from the previous window. `0` is the first window.
    pub prefix: usize,
}

impl LayerStage for WaveTfmStage {
    fn name(&self) -> &str {
        "tts_wave_transformer"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let l = self.frames;
        let pre = self.prefix;
        let (d_in, d_tfm) = model.plan("a.gen.wav.tfm.in_proj.weight")?.in_out();
        let ffn = model
            .plan("a.gen.wav.tfm.blk.0.ffn_gate.weight")?
            .ggml_shape[1];
        let head_dim = model.plan("a.gen.wav.tfm.blk.0.attn_q.weight")?.ggml_shape[1] / self.heads;
        let hkv = self.heads * head_dim;
        let layers = model
            .plans
            .iter()
            .filter(|p| p.name.starts_with("a.gen.wav.tfm.blk.") && p.name.ends_with(".ln1.weight"))
            .count();
        let eps = 1e-5f32;

        // The reference prepends the persisted left context (`k − 1` frames).
        // The hidden comes in with time on the **rows**, while a convolution
        // reads its channels on axis 1 and time on the last axis, so this is a
        // transpose, not a reshape.
        let ct = emit::h(ctx).transpose_(input.hir_id(), vec![1, 0]);
        let x = emit::reshape(ctx, ct, &[1, d_tfm as i64, 1, l as i64]);
        let k = model.plan("a.gen.wav.pre_conv.weight")?.ggml_shape[0];
        let left = emit::h(ctx).input("left", Shape::new(&[1, d_tfm, 1, k - 1], DType::F32));
        let x = emit::h(ctx).concat_(vec![left, x], 3);
        let w = emit::conv_param(ctx, model, "a.gen.wav.pre_conv.weight")?;
        let y = emit::h(ctx).conv2d(
            x,
            w,
            [1, k],
            [1, 1],
            [0, 0],
            1,
            Shape::new(&[1, d_in, 1, l], DType::F32),
        );
        let b = emit::conv_bias(ctx, model, "a.gen.wav.pre_conv.bias", d_in)?;
        let y = emit::add(ctx, y, b);
        let t = emit::to_lc(ctx, y, d_in, l);

        // `left` for the next window is the tail of this window's input.
        let left_tail = emit::narrow(ctx, x, 3, l, k - 1);
        ctx.publish_side_output(
            "left",
            &FlowValue::new(left_tail, Shape::new(&[1, d_tfm, 1, k - 1], DType::F32)),
        );

        let proj = emit::linear(
            ctx,
            &FlowValue::new(t, Shape::new(&[l, d_in], DType::F32)),
            "a.gen.wav.tfm.in_proj.weight",
        )?;
        let b = emit::f32_param(ctx, model, "a.gen.wav.tfm.in_proj.bias", &[1, d_tfm])?;
        let mut cur = emit::add(ctx, proj.hir_id(), b);

        // Global position, not the window's own offset.
        let half = head_dim / 2;
        let cos = emit::h(ctx).input("cos", Shape::new(&[l, half], DType::F32));
        let sin = emit::h(ctx).input("sin", Shape::new(&[l, half], DType::F32));
        let kv_in = if pre > 0 {
            Some((
                emit::h(ctx).input("kpre", Shape::new(&[layers, pre, hkv], DType::F32)),
                emit::h(ctx).input("vpre", Shape::new(&[layers, pre, hkv], DType::F32)),
            ))
        } else {
            None
        };
        let mut k_all: Vec<HirNodeId> = Vec::with_capacity(layers);
        let mut v_all: Vec<HirNodeId> = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("a.gen.wav.tfm.blk.{i}");
            let (o, k, v) = wave_block(
                ctx, model, &p, d_tfm, ffn, self.heads, head_dim, eps, cur, l, i, pre, kv_in, cos,
                sin,
            )?;
            cur = o;
            k_all.push(k);
            v_all.push(v);
        }
        let knew = emit::h(ctx).concat_(k_all, 0);
        let vnew = emit::h(ctx).concat_(v_all, 0);
        let kv_shape = Shape::new(&[layers, l, hkv], DType::F32);
        ctx.publish_side_output("knew", &FlowValue::new(knew, kv_shape.clone()));
        ctx.publish_side_output("vnew", &FlowValue::new(vnew, kv_shape));

        let gamma = emit::f32_param(ctx, model, "a.gen.wav.tfm.output_norm.weight", &[d_tfm])?;
        let cur = emit::rms_norm(ctx, cur, gamma, eps, &[l, d_tfm]);
        let out = emit::linear(
            ctx,
            &FlowValue::new(cur, Shape::new(&[l, d_tfm], DType::F32)),
            "a.gen.wav.tfm.out_proj.weight",
        )?;
        let b = emit::f32_param(ctx, model, "a.gen.wav.tfm.out_proj.bias", &[1, d_in])?;
        let out = emit::add(ctx, out.hir_id(), b);
        let sh = Shape::new(&[l, d_in], DType::F32);
        Ok((
            FlowValue::new(out, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// One wave-transformer block with carried K/V. The block is the generic one
/// (RMSNorm, QKV without bias, RoPE, gated FFN, LayerScale), but the keys and
/// values of the carried prefix are concatenated before attention and this
/// window's own K/V are handed back in `[l, heads·head_dim]`.
#[allow(clippy::too_many_arguments)]
fn wave_block(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    p: &str,
    d: usize,
    ffn: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    x: HirNodeId,
    m: usize,
    layer: usize,
    prefix: usize,
    kv_in: Option<(HirNodeId, HirNodeId)>,
    cos: HirNodeId,
    sin: HirNodeId,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    let hkv = heads * head_dim;
    let gamma = emit::f32_param(ctx, model, &format!("{p}.ln1.weight"), &[d])?;
    let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d]);
    let [q, k, v] = emit::qkv(
        ctx,
        model,
        h,
        m,
        d,
        [
            &format!("{p}.attn_q.weight"),
            &format!("{p}.attn_k.weight"),
            &format!("{p}.attn_v.weight"),
        ],
    )?;
    let q = emit::reshape(ctx, q, &[1, m as i64, (heads * head_dim) as i64]);
    let k = emit::reshape(ctx, k, &[1, m as i64, hkv as i64]);
    let v = emit::reshape(ctx, v, &[1, m as i64, hkv as i64]);
    let q = emit::h(ctx).rope(q, cos, sin, head_dim);
    let k = emit::h(ctx).rope(k, cos, sin, head_dim);
    let q4 = head_major(ctx, q, m, heads, head_dim);
    let k4 = head_major(ctx, k, m, heads, head_dim);
    let v4 = head_major(ctx, v, m, heads, head_dim);
    let (k_all, v_all) = match kv_in {
        Some((kp, vp)) => {
            let kpq = kv_prefix(ctx, kp, layer, prefix, heads, head_dim);
            let vpq = kv_prefix(ctx, vp, layer, prefix, heads, head_dim);
            (
                emit::h(ctx).concat_(vec![kpq, k4], 2),
                emit::h(ctx).concat_(vec![vpq, v4], 2),
            )
        }
        None => (k4, v4),
    };
    let sh = Shape::new(&[1, heads, m, head_dim], DType::F32);
    let a = emit::h(ctx).attention_kind(q4, k_all, v_all, heads, head_dim, MaskKind::Causal, sh);
    let a = emit::h(ctx).transpose_(a, vec![0, 2, 1, 3]);
    let a = emit::reshape(ctx, a, &[m as i64, (heads * head_dim) as i64]);
    let o = emit::linear(
        ctx,
        &FlowValue::new(a, Shape::new(&[m, heads * head_dim], DType::F32)),
        &format!("{p}.attn_out.weight"),
    )?;
    let ls = emit::f32_param(ctx, model, &format!("{p}.ls1.weight"), &[1, d])?;
    let o = emit::mul(ctx, o.hir_id(), ls);
    let x = emit::add(ctx, x, o);

    let gamma = emit::f32_param(ctx, model, &format!("{p}.ln2.weight"), &[d])?;
    let h = emit::rms_norm(ctx, x, gamma, eps, &[m, d]);
    let gate = emit::linear(
        ctx,
        &FlowValue::new(h, Shape::new(&[m, d], DType::F32)),
        &format!("{p}.ffn_gate.weight"),
    )?;
    let up = emit::linear(
        ctx,
        &FlowValue::new(h, Shape::new(&[m, d], DType::F32)),
        &format!("{p}.ffn_up.weight"),
    )?;
    let act = emit::silu(ctx, gate.hir_id(), &[m, ffn]);
    let act = emit::mul(ctx, act, up.hir_id());
    let down = emit::linear(
        ctx,
        &FlowValue::new(act, Shape::new(&[m, ffn], DType::F32)),
        &format!("{p}.ffn_down.weight"),
    )?;
    let ls = emit::f32_param(ctx, model, &format!("{p}.ls2.weight"), &[1, d])?;
    let down = emit::mul(ctx, down.hir_id(), ls);
    let out = emit::add(ctx, x, down);

    // `[1, heads, m, head_dim]` → `[m, heads·head_dim]` for the carried state.
    let kt = emit::h(ctx).transpose_(k4, vec![0, 2, 1, 3]);
    let vt = emit::h(ctx).transpose_(v4, vec![0, 2, 1, 3]);
    let kt = emit::reshape(ctx, kt, &[m as i64, hkv as i64]);
    let vt = emit::reshape(ctx, vt, &[m as i64, hkv as i64]);
    Ok((out, kt, vt))
}

// ── the wave transformer with the state of the previous window ─────

/// `[rows, heads·head_dim]` → `[1, heads, rows, head_dim]` (head-major).
fn head_major(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    rows: usize,
    heads: usize,
    hd: usize,
) -> HirNodeId {
    if rows == 1 {
        // A one-row block: the transposed axis is length 1, so the declaration
        // is the layout and no step is needed.
        return emit::reshape(ctx, x, &[1, heads as i64, rows as i64, hd as i64]);
    }
    let r = emit::reshape(ctx, x, &[1, rows as i64, heads as i64, hd as i64]);
    emit::h(ctx).transpose_(r, vec![0, 2, 1, 3])
}

/// One layer's slice of the carried K/V: `[layers, prefix, heads·head_dim]` →
/// `[1, heads, prefix, head_dim]`.
fn kv_prefix(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    layer: usize,
    prefix: usize,
    heads: usize,
    hd: usize,
) -> HirNodeId {
    let s = emit::narrow(ctx, x, 0, layer, 1);
    let s = emit::reshape(ctx, s, &[1, prefix as i64, heads as i64, hd as i64]);
    emit::h(ctx).transpose_(s, vec![0, 2, 1, 3])
}

/// The last `len` frames of `[1, heads, seq, head_dim]`, back in the layout the
/// next window reads: `[1, 1, len, heads·head_dim]`.
fn kv_tail(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    start: usize,
    len: usize,
    heads: usize,
    hd: usize,
) -> HirNodeId {
    let s = emit::narrow(ctx, x, 2, start, len);
    let s = emit::h(ctx).transpose_(s, vec![0, 2, 1, 3]);
    emit::reshape(ctx, s, &[1, 1, len as i64, (heads * hd) as i64])
}

// ── the codebook: the RVQ codes → the codec's hidden ───────────────

/// The decode half of the RVQ (`a.gen.wav.quant.*`): the semantic codebook's
/// row is projected on its own, the fifteen acoustic codebooks' rows are summed
/// and projected together, and the two projections add up.
///
/// Rows are read **by index** — a codebook is a lookup, not a nearest-neighbour
/// search — which is what keeps `ArgMin` out of the decoding path.
pub struct QuantDecodeStage {
    pub model: Arc<Model>,
    /// Frames; the codes arrive level-major (`16 × frames` numbers), exactly
    /// the layout the reference's per-level views take.
    pub frames: usize,
}

impl LayerStage for QuantDecodeStage {
    fn name(&self) -> &str {
        "tts_quant_decode"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let model = &self.model;
        let frames = self.frames;
        let first = model.plan("a.gen.wav.quant.first.codebook.weight")?.clone();
        let (vocab, d) = (first.dims[0], first.dims[1]);
        let out_dim = model
            .plan("a.gen.wav.quant.first.out_proj.weight")?
            .in_out()
            .1;

        let table = emit::f32_param(
            ctx,
            model,
            "a.gen.wav.quant.first.codebook.weight",
            &[vocab, d],
        )?;
        let ids = emit::narrow(ctx, input.hir_id(), 0, 0, frames);
        let sem = emit::gather(ctx, table, ids);
        let sem = emit::linear(
            ctx,
            &FlowValue::new(sem, Shape::new(&[frames, d], DType::F32)),
            "a.gen.wav.quant.first.out_proj.weight",
        )?;

        let levels = model
            .plan("a.gen.wav.quant.rest.codebook.weight")?
            .ggml_shape[2];
        let rest = model
            .f32_file_order("a.gen.wav.quant.rest.codebook.weight")?
            .0;
        let mut acc: Option<HirNodeId> = None;
        for level in 0..levels {
            let t = emit::table_slice(&rest, level, vocab, d);
            let node = emit::sliced_param(
                ctx,
                "a.gen.wav.quant.rest.codebook.weight",
                level,
                t,
                &[vocab, d],
            );
            let ids = emit::narrow(ctx, input.hir_id(), 0, (level + 1) * frames, frames);
            let e = emit::gather(ctx, node, ids);
            acc = Some(match acc {
                None => e,
                Some(a) => emit::add(ctx, a, e),
            });
        }
        let acc = acc.ok_or_else(|| anyhow!("the acoustic codebook has no levels"))?;
        let ac = emit::linear(
            ctx,
            &FlowValue::new(acc, Shape::new(&[frames, d], DType::F32)),
            "a.gen.wav.quant.rest.out_proj.weight",
        )?;
        let hidden = emit::add(ctx, sem.hir_id(), ac.hir_id());
        let sh = Shape::new(&[frames, out_dim], DType::F32);
        Ok((
            FlowValue::new(hidden, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

// ── running a built flow ───────────────────────────────────────────

/// Submissions and blocking waits issued by this process's walks. Each
/// `Runner::run`/`run_read`/`run_slices` is one submit and one wait. The
/// counters live here, around the whole cell; `waits` mirrors `submits`
/// because the backend calls are synchronous. Both count *calls*, not queue
/// submissions: a chunked run (`RLX_WGPU_ENCODE_CHUNK`) submits several command
/// buffers inside one `run`, and the queue-level figure is the backend's own
/// `rlx_wgpu::device_submit_count`.
static SUBMITS: AtomicU64 = AtomicU64::new(0);
static WAITS: AtomicU64 = AtomicU64::new(0);

pub fn device_counts() -> (u64, u64) {
    (
        SUBMITS.load(Ordering::Relaxed),
        WAITS.load(Ordering::Relaxed),
    )
}

fn bump(submits: u64, waits: u64) {
    SUBMITS.fetch_add(submits, Ordering::Relaxed);
    WAITS.fetch_add(waits, Ordering::Relaxed);
}

/// One graph × one backend: what it cost, what came out of it.
pub struct Outcome {
    pub compile_ms: f64,
    pub upload_ms: f64,
    pub first_ms: f64,
    pub step_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub outs: Vec<Vec<f32>>,
    /// The arena `rlx-wgpu`'s own planner sizes for this graph
    /// ([`rlx_wgpu::buffer::plan_f32_uniform`], align 16) — the graph before the
    /// backend's unfuse passes, and without its scratch tail.
    pub arena_bytes: usize,
    /// Packed `DequantMatMul` nodes in the graph — 0 on the sound stage.
    pub dequant_nodes: usize,
    pub params_f32: usize,
    pub params_packed: usize,
    /// Node count by op kind, in the lowered graph (what the backend schedules).
    pub ops: Vec<(String, usize)>,
    /// The runs a report's counters cover — the cold first run plus `reps` measured ones.
    pub passes: usize,
}

/// Compile a built flow and run it.
///
/// The flow's own outputs are taken apart by hand: `BuiltModel` keeps the f32
/// param map and the packed U8 blobs as public fields, but its `into_graph`
/// consumes the whole value and drops them, so the module is lowered here
/// directly (`hir_to_graph`) to keep both.
/// A graph compiled **once** and restarted frame after frame: the same plan, the
/// same buffers, new inputs each time. A graph per frame is a cost of the stand,
/// not of the model — the model's own run keeps its session and feeds the cache.
pub struct Runner {
    session: rlx_runtime::CompiledGraph,
    /// The compiled graph itself — this crate's source for the arena and the ops.
    pub graph: rlx_ir::Graph,
    pub compile_ms: f64,
    pub upload_ms: f64,
}

impl Runner {
    pub fn compile(device: Device, mut built: BuiltModel) -> Result<Self> {
        let hir = built
            .module
            .into_hir()
            .ok_or_else(|| anyhow!("flow built a non-HIR module"))?;
        let graph = rlx_ir::hir_to_graph(hir)?;
        let t = crate::report::Stopwatch::start();
        let mut session = Session::new(device).compile(graph.clone());
        let compile_ms = t.ms();
        // Upload and release one tensor at a time: the flow hands over the whole
        // weight map at once, but holding all of it while the device receives its
        // own copy would put two checkpoints in the process. So the entry leaves
        // the map before its clone goes to the device, not after the map has been
        // walked whole.
        let t = crate::report::Stopwatch::start();
        while let Some(name) = built.params.keys().next().cloned() {
            let values = built.params.remove(&name).expect("key just read");
            session.set_param(&name, &values);
        }
        while let Some((name, bytes, dt)) = built.typed_params.pop() {
            session.set_param_typed(&name, &bytes, dt);
        }
        let upload_ms = t.ms();
        Ok(Self {
            session,
            graph,
            compile_ms,
            upload_ms,
        })
    }

    /// Point the frame graph's `Sample` steps at this frame's draw seeds, in
    /// level order — one seed per projecting level. The seeds are uniform words
    /// the backend rewrites before the next run ([`rlx_runtime::CompiledGraph`]
    /// forwards this to the backend), so a compiled frame serves every frame and
    /// every request; `false` means the graph does not have exactly that many
    /// `Sample` steps and nothing was written.
    pub fn set_sample_seeds(&mut self, seeds: &[u64]) -> bool {
        self.session.set_sample_seeds(seeds)
    }

    /// The step kinds of the schedule this graph compiled to, in emission order.
    pub fn schedule_step_names(&self) -> Option<Vec<&'static str>> {
        self.session.schedule_step_names()
    }

    /// One pass with new inputs — the same compiled buffers, refilled.
    pub fn run(&mut self, inputs: &[(String, Vec<f32>)]) -> Vec<Vec<f32>> {
        let ins: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        bump(1, 1);
        self.session.run(&ins)
    }

    /// One pass that reads back only the outputs named in `read`; the others are
    /// still written (a registered row feed reads them on device).
    pub fn run_read(&mut self, inputs: &[(String, Vec<f32>)], read: &[usize]) -> Vec<Vec<f32>> {
        let ins: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        bump(1, 1);
        self.session.run_read_outputs(&ins, Some(read))
    }

    /// One pass whose inputs are already borrowed.
    pub fn run_slices(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        bump(1, 1);
        self.session.run(inputs)
    }

    /// A plane cache: each layer's `[capacity + 1, hkv]` rows are a resident
    /// handle of their own, registered to this graph's own new-row output. The
    /// output order is the stage's: the main output, this graph's K rows (one per
    /// layer), then its V rows.
    pub fn resident_cache_planes(&mut self, k: &[Vec<f32>], v: &[Vec<f32>]) -> bool {
        if k.len() != v.len() {
            return false;
        }
        for (gi, rows) in k.iter().enumerate() {
            if !self.session.bind_gpu_handle(&format!("k{gi}"), rows) {
                return false;
            }
        }
        for (gi, rows) in v.iter().enumerate() {
            if !self.session.bind_gpu_handle(&format!("v{gi}"), rows) {
                return false;
            }
        }
        let base = 1; // the graph's main output
        let n = k.len();
        for gi in 0..n {
            if !self
                .session
                .register_kv_row_feed(&format!("k{gi}"), base + gi)
            {
                return false;
            }
            if !self
                .session
                .register_kv_row_feed(&format!("v{gi}"), base + n + gi)
            {
                return false;
            }
        }
        true
    }

    /// One plane-cache step: run, read only `read`, and fold each layer's new
    /// row into its handle. `row_elems` is one layer's row width (`hkv`).
    pub fn run_resident_planes(
        &mut self,
        inputs: &[(String, Vec<f32>)],
        row: usize,
        row_elems: usize,
        read: &[usize],
    ) -> Option<Vec<Vec<f32>>> {
        let out = self.run_read(inputs, read);
        bump(1, 1);
        if !self.session.feed_kv_row(0, row, row_elems) {
            return None;
        }
        Some(out)
    }

    /// The whole current contents of a resident plane input, read back from the
    /// device. The decode's bucket rebuild is the one caller: the planes live on
    /// the device between steps, so a larger capacity's planes are filled from
    /// the smaller ones' live rows here, not from the host mirror the row feed
    /// emptied.
    pub fn read_plane(&self, name: &str) -> Option<Vec<f32>> {
        self.session.read_gpu_handle(name)
    }
}

/// One graph built and compiled: what a cell's build step leaves behind.
///
/// Everything the graph is made of — the planner's arena, its op census, the param
/// sizes — is read here, at the build, so the run step prices only the walk. The
/// graph is then fed again and again (a frame, a window, a submission) and never
/// rebuilt inside the run.
///
/// A compiled graph lives in this process only: the engine persists optimized LIR
/// (`rlx_runtime::AotCache`) but not the backend's executable, so a second process
/// still pays `backend.compile_lir` — there is nothing here to reuse across cells,
/// and that is what a cell's `build_ms` is.
pub struct Compiled {
    pub runner: Runner,
    pub compile_ms: f64,
    pub upload_ms: f64,
    pub arena_bytes: usize,
    pub dequant_nodes: usize,
    pub params_f32: usize,
    pub params_packed: usize,
    pub ops: Vec<(String, usize)>,
}

impl Compiled {
    /// The build step for one graph: emit was the caller's, this lower it, compiles
    /// it and uploads its params — the price paid once, before any pass.
    pub fn new(device: Device, built: BuiltModel) -> Result<Self> {
        let params_f32: usize = built.params.values().map(|v| v.len() * 4).sum();
        let params_packed: usize = built.typed_params.iter().map(|(_, b, _)| b.len()).sum();
        let runner = Runner::compile(device, built)?;
        let graph = &runner.graph;
        let arena_bytes = rlx_wgpu::buffer::plan_f32_uniform(graph, 16).arena_size;
        let dequant_nodes = graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, rlx_ir::Op::DequantMatMul { .. }))
            .count();
        let mut ops: BTreeMap<String, usize> = BTreeMap::new();
        for n in graph.nodes() {
            *ops.entry(format!("{:?}", n.op.kind())).or_default() += 1;
        }
        Ok(Self {
            compile_ms: runner.compile_ms,
            upload_ms: runner.upload_ms,
            runner,
            arena_bytes,
            dequant_nodes,
            params_f32,
            params_packed,
            ops: ops.into_iter().collect(),
        })
    }

    /// Bytes this graph's parameters occupy: the uploaded f32 copy and the packed
    /// weights as the file holds them.
    #[must_use]
    pub fn params_bytes(&self) -> usize {
        self.params_f32 + self.params_packed
    }

    /// Bytes this graph keeps on the device: [`Compiled::params_bytes`] plus its
    /// planned arena. The arena half is the *plan* ([`Compiled::arena_bytes`]), the
    /// pre-unfuse figure, which is larger than what the backend allocates — so this
    /// over-states what a graph holds.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.params_bytes() + self.arena_bytes
    }

    /// The run step: one cold pass (it pays the device's warm-up) and then `reps`
    /// measured passes on the same compiled graph.
    pub fn run(&mut self, inputs: &[(String, Vec<f32>)], reps: usize) -> Outcome {
        let (compile_ms, upload_ms) = (self.compile_ms, self.upload_ms);
        let arena_bytes = self.arena_bytes;
        let dequant_nodes = self.dequant_nodes;
        let (params_f32, params_packed) = (self.params_f32, self.params_packed);
        let ops = self.ops.clone();
        let runner = &mut self.runner;

        let t0 = crate::report::Stopwatch::start();
        let first = runner.run(inputs);
        let first_ms = t0.ms();

        let mut samples = Vec::with_capacity(reps);
        let mut outs_reps = vec![first];
        for _ in 0..reps {
            let t = crate::report::Stopwatch::start();
            let out = runner.run(inputs);
            samples.push(t.ms());
            outs_reps.push(out);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (step_ms, min_ms, max_ms) = if samples.is_empty() {
            (first_ms, first_ms, first_ms)
        } else {
            (
                samples[samples.len() / 2],
                samples[0],
                samples[samples.len() - 1],
            )
        };
        let outs = outs_reps[0].clone();

        Outcome {
            compile_ms,
            upload_ms,
            first_ms,
            step_ms,
            min_ms,
            max_ms,
            outs,
            arena_bytes,
            dequant_nodes,
            params_f32,
            params_packed,
            ops,
            passes: reps + 1,
        }
    }

    /// One pass with borrowed inputs, no repeated warm passes: the caller owns
    /// the state buffers and must not have them copied just to be read.
    pub fn run_slices(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.runner.run_slices(inputs)
    }
}

// ── the roles glued into one pass each ─────────────────────────────

/// How many tokens the recognition sound front produces from `frames` mel
/// frames (three stride-2, 3×3 convolutions).
pub fn asr_tokens(frames: usize) -> usize {
    let mut w = frames;
    for _ in 0..3 {
        w = (w + 2 - 3) / 2 + 1;
    }
    w
}

/// The conv front's width per token: 480 channels over the mel bins the three
/// stride-2 convolutions halve. Written once — the front's reshape, the tail's
/// `conv_out` and the group pass all read it, and a mismatch between them would
/// mis-shape the tail's input silently.
pub fn flat_width(mel_bins: usize) -> usize {
    480 * (mel_bins / 8)
}

/// `[L, C]` → `[1, C, 1, L]`: the layout a convolution reads, so the wave
/// transformer's output can feed the codec in the same graph.
pub struct LcToNchwStage {
    pub c: usize,
    pub l: usize,
}

impl LayerStage for LcToNchwStage {
    fn name(&self) -> &str {
        "lc_to_nchw"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let (c, l) = (self.c, self.l);
        let t = emit::h(ctx).transpose_(input.hir_id(), vec![1, 0]);
        let x = emit::reshape(ctx, t, &[1, c as i64, 1, l as i64]);
        let sh = Shape::new(&[1, c, 1, l], DType::F32);
        Ok((
            FlowValue::new(x, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The codec's windows over a part: the DAC stages are convolutional (their
/// LayerNorm is per position) and causal, so a window with its left margin gives
/// exactly the whole part's samples over its core, and the planned arena — which
/// is a graph's — shrinks with the window instead of being the part's.
///
/// The window is 144 frames — two of the front's chunks ([`FRONT_WINDOW`]): every
/// window runs its graph over every frame in it, so a longer window is fewer
/// graph runs over the same frames, each convolution at twice the columns and for
/// twice the frames. Doubling again is the same curve's next step and is not
/// taken, because the arena grows with the window.
pub const DAC_WINDOW: usize = 144;
/// The wave transformer's own window: the front is cut at this many frames so
/// its output does not depend on the part's total length.
pub const FRONT_WINDOW: usize = 72;
/// Frames of K/V state a full chunk carries into the next one: the chunk it runs
/// less the single frame it re-reads. The caller allocates its carry with it.
pub const FRONT_CARRY: usize = FRONT_WINDOW - 1;
/// Frames of left context a window's span carries before its core, clipped at the
/// part's own left edge: the four DAC blocks' kernels and strides reach this far
/// into a window, so a core sample never reads a window edge the part does not
/// have. There is no right margin — the decoder is causal, so frames past the core
/// would produce only outputs the stitch discards.
pub const DAC_MARGIN: usize = 4;

/// One window's span: the core `[a, b)` extended `left` frames before it and
/// `right` after, clipped at the part's own edges. [`windows`] uses it with a zero
/// right margin; a probe that prices a right margin the decoder no longer carries
/// reads both arms through it, from the one cut.
pub fn window_span(rows: usize, a: usize, b: usize, left: usize, right: usize) -> (usize, usize) {
    (a.saturating_sub(left), (b + right).min(rows))
}

/// Codec windows over the part's frames: `(p0, p1, a, b)` — the span the DAC
/// graphs read and the core `[a, b)` the stitch keeps. A window's span carries
/// `margin` frames of left context and none on the right: the decoder is causal
/// (every convolution pads its left edge and a transposed conv narrows to the
/// samples it owns), so frames past the core produce outputs `stitch_window`
/// discards. The first window is not padded past the part's own left edge.
pub fn windows(rows: usize, window: usize, margin: usize) -> Vec<(usize, usize, usize, usize)> {
    debug_assert!(margin < window);
    let mut v = Vec::new();
    let mut a = 0;
    if rows > 0 {
        let b = window.min(rows);
        let (p0, p1) = window_span(rows, a, b, margin, 0);
        v.push((p0, p1, a, b));
        a = b;
    }
    while a < rows {
        let b = (a + window).min(rows);
        let (p0, p1) = window_span(rows, a, b, margin, 0);
        v.push((p0, p1, a, b));
        a = b;
    }
    v
}

/// How the sound half is cut into graphs.
/// The arena is a graph's, so the cut follows the planned sizes at the window the
/// sound half actually runs ([`DAC_WINDOW`], 144 frames — twice what this cut
/// was first drawn at): the front and the codec's first four stages,
/// stages 4–5, stages 6–7 — three graphs instead of nine. A group whose plan
/// outgrew the backend's per-binding limit would show up as host routing or arena
/// striping in the stand's per-phase `arena` column, which is what the window's
/// own measurement is read against.
pub fn codec_groups() -> Vec<std::ops::Range<usize>> {
    vec![0..4, 4..6, 6..DAC_STAGES]
}

/// Waveform samples per code frame: two stride-2 transposed convs, then the four
/// DAC stages' strides 8/5/4/3, multiply to `2·2·8·5·4·3 = 1920`.
pub fn samples_per_frame() -> usize {
    dac_stage_io(DAC_STAGES, 1).1
}

/// One graph of the sound half: the front (codes → the RVQ codebook decode → the
/// wave transformer → the codec's `[1, C, 1, frames]` input) when the group starts
/// at the codec's input, then the group's own stages.
fn synthesis_sound_group(
    model: Arc<Model>,
    frames: usize,
    stages: std::ops::Range<usize>,
    prefix: usize,
) -> Result<BuiltModel> {
    let levels = model
        .plan("a.gen.wav.quant.rest.codebook.weight")?
        .ggml_shape[2];
    let d_out = model.plan("a.gen.wav.tfm.in_proj.weight")?.in_out().0;
    let (ch, l) = dac_stage_io(stages.start, frames);
    let mut weights = LoaderWeights::new(model.clone());
    let name = format!(
        "tts_sound_{}_{}_{frames}_p{prefix}",
        stages.start, stages.end
    );
    let flow = ModelFlow::new(name).with_profile(profile());
    let flow = if stages.start == 0 {
        flow.input("codes", Shape::new(&[(levels + 1) * frames], DType::F32))
            .layer_stage(QuantDecodeStage {
                model: model.clone(),
                frames,
            })
            .layer_stage(WaveTfmStage {
                model: model.clone(),
                frames,
                heads: 16,
                prefix,
            })
            .layer_stage(LcToNchwStage {
                c: d_out,
                l: frames,
            })
    } else {
        flow.input("codec_in", Shape::new(&[1, ch, 1, l], DType::F32))
    };
    flow.layer_stage(DacStage {
        model,
        frames,
        stages,
    })
    .output("waveform")
    .build(&mut weights)
    .map_err(Into::into)
}

/// One compiled sound-half graph, shared between a cache and its users: `Rc`
/// because a process synthesizes on one thread and a call holds several graphs at
/// once (a part's first and carried windows), while `Runner::run` takes
/// `&mut self`. The graph is read many times and rewritten never, and no borrow is
/// held across a cache call.
pub type SoundGraph = std::rc::Rc<std::cell::RefCell<Compiled>>;

/// Device bytes a decoder cache may hold, by [`Compiled::device_bytes`] — an arena
/// is resident for as long as its graph is, so parameters alone would not bound it.
/// This holds one part's set with the graphs a new length brings; the ceiling that
/// decides for real is `guard`'s, which reads the resident set.
const SOUND_GRAPHS_BYTES: usize = 5 * 1024 * 1024 * 1024;

/// What fixes a sound graph: the file it reads, its device, the frames it runs,
/// its stage group and its K/V prefix. Two requests with the same key want the
/// same weights in the same graph. For a decoder window `frames` is the span the
/// graph reads — its core plus the left context — not the core alone.
#[derive(PartialEq)]
struct SoundKey {
    path: std::path::PathBuf,
    device: rlx_runtime::Device,
    frames: usize,
    stages: std::ops::Range<usize>,
    prefix: usize,
}

/// The decoder graphs one synthesis call needs, most recently used last. Owned by
/// whoever synthesizes — [`crate::synth::Synthesis`] in the library, a local of the
/// synthesis half in the stand — so the graphs are released when that is, and the
/// process holds one model pair's decoder, not every length it ever ran.
#[derive(Default)]
pub struct SoundGraphs {
    entries: Vec<(SoundKey, SoundGraph)>,
}

impl SoundGraphs {
    /// What this cache holds, in the form the probes report and the budget reads:
    /// `(graphs, parameters, device bytes)`. Parameters are [`Compiled::params_bytes`]
    /// — pages of the mapped file; the third figure is the cache's own account,
    /// [`Compiled::device_bytes`] over its graphs. A graph a caller still holds —
    /// one part's first and carried windows during a call — stays resident after
    /// eviction, so it is the cache's figure, not the process's.
    #[must_use]
    pub fn census(&self) -> (usize, usize, usize) {
        self.entries
            .iter()
            .fold((0, 0, 0), |(n, params, device), (_, g)| {
                let g = g.borrow();
                (n + 1, params + g.params_bytes(), device + g.device_bytes())
            })
    }

    /// The graph for `frames` frames of `stages` at `prefix`, compiled and uploaded
    /// on first use and reused after. A hit moves to the newest end, so the graphs
    /// every request of a length reuses — a part's first and carried windows — are
    /// the last to be evicted when another length arrives.
    pub fn graph(
        &mut self,
        device: rlx_runtime::Device,
        model: &Arc<Model>,
        frames: usize,
        stages: std::ops::Range<usize>,
        prefix: usize,
    ) -> Result<SoundGraph> {
        let key = SoundKey {
            path: model.path.clone(),
            device,
            frames,
            stages,
            prefix,
        };
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            let hit = self.entries.remove(i);
            let graph = hit.1.clone();
            self.entries.push(hit);
            return Ok(graph);
        }
        // Build first and trim after, so a failed build leaves the cache as it
        // was. The transient is the whole cache plus this graph, which is one
        // part's set plus one graph: a graph's arena is only known once it exists,
        // so the trim below is the bound the budget can actually promise.
        let graph = std::rc::Rc::new(std::cell::RefCell::new(Compiled::new(
            device,
            synthesis_sound_group(model.clone(), frames, key.stages.clone(), prefix)?,
        )?));
        self.entries.push((key, graph.clone()));
        while self.entries.len() > 1 && self.census().2 > SOUND_GRAPHS_BYTES {
            self.entries.remove(0);
        }
        Ok(graph)
    }
}

/// A part's wave-transformer graphs: the first window at the part's own length and
/// the full chunk every later window runs.
pub struct FrontGraphs {
    first: SoundGraph,
    first_len: usize,
    next: Option<SoundGraph>,
}

/// The window lengths a part of `frames` runs a graph for: its first window, and
/// the full chunk every later window runs (a part that does not end on a chunk
/// boundary pads its last window with code 0, whose outputs the stitch discards).
/// Two graphs at most, because a graph per tail length costs more to build than
/// the padded width costs to run (`tests/sound_decoder_cost.rs`).
fn front_lengths(frames: usize) -> (usize, Option<usize>) {
    (
        frames.min(FRONT_WINDOW),
        (frames > FRONT_WINDOW).then_some(FRONT_WINDOW),
    )
}

impl FrontGraphs {
    /// Build (or fetch) the graphs a part of `frames` runs: its first window and,
    /// when the part is longer than one chunk, the carried full chunk.
    pub fn build(
        cache: &mut SoundGraphs,
        device: rlx_runtime::Device,
        model: &Arc<Model>,
        frames: usize,
        stages: std::ops::Range<usize>,
    ) -> Result<Self> {
        let (first, next) = front_lengths(frames);
        Ok(Self {
            first: cache.graph(device, model, first, stages.clone(), 0)?,
            first_len: first,
            next: next
                .map(|len| cache.graph(device, model, len, stages, FRONT_CARRY))
                .transpose()?,
        })
    }

    /// The graph the window at `a` runs and the length it runs over: the first
    /// window its own length, every later one the full chunk — zero-padded when it
    /// is the part's last, and the two always agree so a window's input is the
    /// width its graph declares.
    #[must_use]
    pub fn pick(&self, a: usize) -> (&SoundGraph, usize) {
        if a == 0 {
            (&self.first, self.first_len)
        } else {
            // A window at `a > 0` exists only for a part longer than one chunk, and
            // `front_lengths` builds the carried graph for exactly those parts — the
            // invariant its own unit test pins.
            let next = self.next.as_ref().expect(
                "a window past the first needs the carried chunk `front_lengths` builds for it",
            );
            (next, FRONT_WINDOW)
        }
    }
}

/// The sampler as a graph of **one node**: logits in, drawn ids out
/// (`Op::Sample`). Nothing computes after it, which is what the engine's own
/// phase schedule asks for — a `Sample` is assigned the **epilogue** phase
/// (`rlx-ir/src/phase.rs`), so in a graph that has compute after the draw the
/// node index and the execution order disagree and the slot it reads can be
/// reused. A draw at the head's output, in its own graph, has
/// no such disagreement, and it is what the stand checks against its own
/// sampler.
pub struct SampleStage {
    pub rows: usize,
    pub sampler: SampleSpec,
}

impl LayerStage for SampleStage {
    fn name(&self) -> &str {
        "sample"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let rows = self.rows;
        let ids = emit::h(ctx).add_node(
            Op::Sample {
                top_k: self.sampler.top_k,
                top_p: self.sampler.top_p,
                temperature: self.sampler.temperature,
                seed: self.sampler.seed,
            },
            vec![input.hir_id()],
            Shape::new(&[rows], DType::F32),
        );
        let sh = Shape::new(&[rows], DType::F32);
        Ok((
            FlowValue::new(ids, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// Build the sampler graph: `[rows, vocab]` logits in, `[rows]` drawn ids out.
pub fn sample_graph(vocab: usize, rows: usize, sampler: SampleSpec) -> Result<BuiltModel> {
    let mut weights = NoWeights;
    let flow = ModelFlow::new(format!("sample_{vocab}_{rows}"))
        .with_profile(profile())
        .input("logits", Shape::new(&[rows, vocab], DType::F32))
        .layer_stage(SampleStage { rows, sampler })
        .output("ids");
    flow.build(&mut weights).map_err(Into::into)
}

/// The sampler for **one** row at a time (`[1, vocab]` in, one id out) — the
/// shape a step-by-step walk needs.
pub fn sample_graph_one(vocab: usize, sampler: SampleSpec) -> Result<BuiltModel> {
    sample_graph(vocab, 1, sampler)
}

/// How the pipeline draws: the engine's fused `Op::Sample` at the head's output
/// (top-k, top-p, temperature, seed — one node, no host round trip).
#[derive(Clone, Copy, Debug)]
pub struct SampleSpec {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: u64,
}

// ── the mel front end (ours, not the file's) ───────────────────────

/// The sample rate the front is defined at. It is not in the files either: the
/// models are trained at 16 kHz and the filters are not stored at all.
pub const MEL_SAMPLE_RATE: f64 = 16000.0;
/// The front's window and hop: a 400-point window (25 ms) and a hop of 160
/// (10 ms) — the only two numbers of the front that are a choice rather than a
/// consequence of the files.
pub const MEL_WINDOW: usize = 400;
pub const MEL_HOP: usize = 160;

/// A periodic Hann window of `n` points, as an `[1, n]` row of the frame matrix.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// The real DFT as two `[n, n/2 + 1]` matrices — cosine and sine of
/// `2π·k·i/n`. A 400-point transform is 201 non-negative frequency bins; the
/// frame is multiplied by the window and then by these, which is the whole
/// transform in full (no FFT, and nothing to fuse).
pub fn dft_matrices(n: usize) -> (Vec<f32>, Vec<f32>) {
    let n_freq = n / 2 + 1;
    let mut cos = vec![0f32; n * n_freq];
    let mut sin = vec![0f32; n * n_freq];
    for k in 0..n_freq {
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            cos[i * n_freq + k] = a.cos() as f32;
            sin[i * n_freq + k] = a.sin() as f32;
        }
    }
    (cos, sin)
}

/// Triangular mel filters for the **recognition** front, `[n_freq, bins]`: the
/// Slaney mel scale and Slaney area normalisation the recognition front uses
/// (`norm="slaney"`, `mel_scale="slaney"`), not HTK. The speaker
/// encoder's 24 kHz front is separate (`host::mel_24k`).
pub fn mel_filters(bins: usize, n_freq: usize, sample_rate: f64, f_max: f64) -> Vec<f32> {
    // Slaney: linear below 1 kHz (`200/3` Hz per mel), logarithmic above it.
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let mel = |f: f64| {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let hz = |m: f64| {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };
    let (m_lo, m_hi) = (mel(0.0), mel(f_max));
    let points: Vec<f64> = (0..bins + 2)
        .map(|i| hz(m_lo + (m_hi - m_lo) * i as f64 / (bins + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_freq * bins];
    for m in 0..bins {
        let (lo, mid, hi) = (points[m], points[m + 1], points[m + 2]);
        let area = if hi > lo { 2.0 / (hi - lo) } else { 0.0 };
        for f in 0..n_freq {
            let freq = f as f64 * sample_rate / ((n_freq - 1) * 2) as f64;
            let v = if freq >= lo && freq <= mid {
                (freq - lo) / (mid - lo)
            } else if freq > mid && freq <= hi {
                (hi - freq) / (hi - mid)
            } else {
                0.0
            };
            fb[f * bins + m] = (v * area) as f32;
        }
    }
    fb
}

/// The recognition mel front end: raw samples in, the log-mel spectrogram the
/// recognition sound encoder reads, in its own `[1, 1, bins, frames]` layout.
/// It uses HTK filters and the **power** spectrum (`re² + im²`, no `sqrt`); the
/// speaker encoder's front is `host::mel_24k` (Slaney filters, amplitude).
///
/// Nothing here is stored in any file: the window, the DFT matrices and
/// the filters are computed, the frames are cut at `hop` with `window` points and
/// reflect padding of `window/2` on both sides (so frame `i` is centred on sample
/// `i·hop`), and the log-mel is `log10` of the filter outputs, clamped at
/// `max − 8` over the whole utterance and scaled `(x + 4) / 4` — the convention
/// the reference uses: the clamp and the scale are taken over the whole
/// utterance, not per frame.
pub struct MelStage {
    pub frames: usize,
    pub bins: usize,
    pub window: usize,
    pub hop: usize,
}

impl LayerStage for MelStage {
    fn name(&self) -> &str {
        "mel_frontend"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let (t, bins, n, hop) = (self.frames, self.bins, self.window, self.hop);
        let raw = t + 1;
        let n_freq = n / 2 + 1;

        // The frames: `window` points every `hop`, with the signal reflected half
        // a window on each side so every frame is centred. One gather with a
        // precomputed index row replaces a `Narrow` per frame plus a `Concat`
        // (3 001 nodes on the 30 s cell).
        let padded = emit::pad_axis(ctx, input.hir_id(), 1, 0, n / 2, n / 2, PadMode::Reflect);
        let mut indices = Vec::with_capacity(raw * n * 8);
        for i in 0..raw {
            for j in 0..n {
                indices.extend_from_slice(&(((i * hop + j) as i64).to_le_bytes()));
            }
        }
        let ids = ctx.hir().mir(
            Op::Constant { data: indices },
            vec![],
            Shape::new(&[raw * n], DType::I64),
        );
        let flat = emit::gather(ctx, padded, ids);
        let frames = emit::reshape(ctx, flat, &[raw as i64, n as i64]);

        // The window (a row, broadcast over the frames).
        let w = ctx.synth_param(
            "mel.window",
            hann_window(n),
            Shape::new(&[1, n], DType::F32),
        );
        let w = emit::expand(ctx, w, &[raw, n]);
        let xw = emit::mul(ctx, frames, w);

        // The DFT: |X|² = re² + im².
        let (cos, sin) = dft_matrices(n);
        let cos = ctx.synth_param("mel.dft.cos", cos, Shape::new(&[n, n_freq], DType::F32));
        let sin = ctx.synth_param("mel.dft.sin", sin, Shape::new(&[n, n_freq], DType::F32));
        let re = emit::h(ctx).mm(xw, cos);
        let im = emit::h(ctx).mm(xw, sin);
        let re2 = emit::mul(ctx, re, re);
        let im2 = emit::mul(ctx, im, im);
        let power = emit::add(ctx, re2, im2);

        // The filter bank, then the log.
        let fb = ctx.synth_param(
            "mel.filters",
            mel_filters(bins, n_freq, MEL_SAMPLE_RATE, MEL_SAMPLE_RATE / 2.0),
            Shape::new(&[n_freq, bins], DType::F32),
        );
        let mel = emit::h(ctx).mm(power, fb);
        // `max(mel, 1e-10)`: a zero energy would be `log10(0)`.
        let floor = emit::constant(ctx, &[raw, bins], 1e-10);
        let short = emit::sub(ctx, floor, mel);
        let short = emit::act(ctx, Activation::Relu, short, &[raw, bins]);
        let mel = emit::add(ctx, mel, short);
        let ln10 = emit::constant(ctx, &[raw, bins], 1.0 / std::f64::consts::LN_10 as f32);
        let log = emit::act(ctx, Activation::Log, mel, &[raw, bins]);
        let log = emit::mul(ctx, log, ln10);

        // The clamp over the whole utterance, then `(x + 4) / 4`.
        let mx = emit::reduce_max(ctx, log, &[0, 1], &[1, 1]);
        let minus8 = emit::constant(ctx, &[1, 1], -8.0);
        let thresh = emit::add(ctx, mx, minus8);
        let thresh = emit::expand(ctx, thresh, &[raw, bins]);
        let low = emit::sub(ctx, log, thresh);
        let low = emit::act(ctx, Activation::Relu, low, &[raw, bins]);
        let clamped = emit::add(ctx, thresh, low);
        let four = emit::constant(ctx, &[raw, bins], 4.0);
        let shifted = emit::add(ctx, clamped, four);
        let quarter = emit::constant(ctx, &[raw, bins], 0.25);
        let out = emit::mul(ctx, shifted, quarter);

        // `[raw, bins]` → drop the last frame → the encoder's `[1, 1, bins, T]`.
        let out = emit::narrow(ctx, out, 0, 0, t);
        let t1 = emit::h(ctx).transpose_(out, vec![1, 0]);
        let y = emit::reshape(ctx, t1, &[1, 1, bins as i64, t as i64]);
        let sh = Shape::new(&[1, 1, bins, t], DType::F32);
        Ok((
            FlowValue::new(y, sh.clone()),
            StageArtifacts::hidden_only(sh),
        ))
    }
}

/// The mel front end as a graph: `hop · frames` samples in (the front computes
/// one frame past the window and drops it), the `[1, 1, bins, frames]`
/// log-mel the recognition sound encoder takes out.
pub fn mel_frontend(frames: usize, bins: usize) -> Result<BuiltModel> {
    let (window, hop) = (MEL_WINDOW, MEL_HOP);
    let samples = hop * frames;
    // The front has no weights at all — every constant in it is computed — so the
    // weight source is the loader of no file. It still has to answer `has`.
    let mut weights = NoWeights;
    let flow = ModelFlow::new(format!("mel_{frames}"))
        .with_profile(profile())
        .input("audio", Shape::new(&[samples], DType::F32))
        .layer_stage(MelStage {
            frames,
            bins,
            window,
            hop,
        })
        .output("mel");
    flow.build(&mut weights).map_err(Into::into)
}

/// A weight source for a graph that has no weights (the mel front end, and the
/// one-op graphs).
pub struct NoWeights;

impl rlx_flow::WeightSource for NoWeights {
    fn take(&mut self, key: &str, _transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        bail!("the mel front end has no weights, but `{key}` was asked for")
    }

    fn has(&self, _key: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{FRONT_WINDOW, front_lengths, windows};

    /// The codec's windows tile the part: every frame falls in exactly one core, a
    /// core sits inside its span, no span leaves the part or reaches more than one
    /// left margin before its core — and a span carries no right margin, because
    /// the groups are causal and a sample past the core is one `stitch_window`
    /// discards.
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
                    p0 + margin >= a && p1 - p0 <= window + margin,
                    "rows={rows}: span {p0}..{p1} reaches past its core's left margin"
                );
            }
            let covered: Vec<(usize, usize)> = v.iter().map(|w| (w.2, w.3)).collect();
            assert!(
                covered.is_empty() || (covered[0].0 == 0 && covered[covered.len() - 1].1 == rows),
                "rows={rows}: the cores cover the part"
            );
        }
        // The part's own shape: three windows, two of them a full one and the tail
        // its true 42 frames.
        assert_eq!(
            windows(330, 144, 4),
            vec![(0, 144, 0, 144), (140, 288, 144, 288), (284, 330, 288, 330)]
        );
    }

    /// Two graphs at most: a part of one chunk or less has its first window only,
    /// and every longer part adds the one chunk every later window runs.
    #[test]
    fn a_part_runs_two_front_graphs_at_most() {
        assert_eq!(front_lengths(40), (40, None));
        assert_eq!(front_lengths(FRONT_WINDOW), (72, None));
        assert_eq!(front_lengths(FRONT_WINDOW + 1), (72, Some(72)));
        assert_eq!(front_lengths(330), (72, Some(72)));
    }
}
