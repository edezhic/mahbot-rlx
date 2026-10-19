// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight plumbing and op helpers shared by the three role graphs.
//!
//! Every graph in this stand is assembled on `rlx-flow`: `ModelFlow` carries the
//! topology, a [`LayerStage`](rlx_flow::LayerStage) per graph emits the blocks
//! through `FlowCtx`'s emission surface and `HirMut`'s op set, and the loader
//! above supplies the weights. Two weight paths:
//!
//! * **body and sound** rank-2 tensors (`Store::Packed`) — the file's own
//!   packed bytes through [`WeightSource::take_packed`], so
//!   `FlowCtx::linear` emits a fused `DequantMatMul`;
//! * **host** tensors (`Store::HostDequant`, `Store::F16Widened`) — unpacked to
//!   f32 on the host and multiplied as a plain `MatMul`.
//!
//! A `name#step` key is a stacked tensor read one step at a time (the code
//! predictor's fifteen LM heads), and takes the same paths on its slice.
//!
//! The helpers below also carry the other decisions: 1-D convolutions go
//! through 2-D with `[k, 1]` (never `[k, k]`, which silently takes the 2-D
//! branch), dilated and strided kernels use the op's own fields, ConvTranspose
//! uses the backend's transposed-convolution op, Snake uses the file's own
//! `exp(alpha)`/`exp(-beta)` tensors directly, and the codebook is read by
//! index rather than searched.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use rlx_flow::{FlowCtx, GgufPackedLinear, WeightSource};
use rlx_gguf::bytes_for_public;
use rlx_ir::hir::{HirGraphExt, HirMut};
use rlx_ir::op::{Activation, MaskKind, PadMode, ReduceOp};
use rlx_ir::{DType, HirNodeId, HirOp, Op, Shape};

use crate::loader::{self, Model, Store};

/// The weight source of one file: serves both weight paths.
pub struct LoaderWeights {
    pub model: Arc<Model>,
    /// Weights fused at load time under a synthetic key. The file is not
    /// rewritten and no per-step graph is rebuilt: `take_packed` serves the
    /// concatenated Q8 blob once, and the graph sees one `DequantMatMul`.
    fused: HashMap<String, GgufPackedLinear>,
}

impl LoaderWeights {
    pub fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            fused: HashMap::new(),
        }
    }

    /// Concatenate packed `[out, in]` Q8 blobs along their rows. False when any
    /// key is not packed or the schemes/in-dims differ — the caller then keeps
    /// the separate projections rather than changing arithmetic.
    pub fn fuse_packed(&mut self, name: &str, keys: &[&str]) -> Result<bool> {
        let mut total = 0usize;
        for &key in keys {
            let name = split_step(key).map_or(key, |(n, _)| n);
            let p = self.model.plan(name)?.clone();
            let elems = p.ggml_shape[0] * p.ggml_shape[1];
            let Some(n) = bytes_for_public(p.file, elems) else {
                return Ok(false);
            };
            total += n;
        }
        let mut bytes = Vec::with_capacity(total);
        let mut scheme = None;
        let mut in_dim = None;
        let mut out_dim = 0;
        for &key in keys {
            let part = if let Some((name, step)) = split_step(key) {
                self.take_step_packed(name, step)?
            } else {
                let p = self.model.plan(key)?.clone();
                if p.store != Store::Packed {
                    return Ok(false);
                }
                let Some(q) = loader::quant_of(p.file) else {
                    return Ok(false);
                };
                let (i, o) = p.in_out();
                Some(GgufPackedLinear {
                    w_q: self.model.raw(key)?,
                    scheme: q,
                    in_dim: i,
                    out_dim: o,
                    bias: Vec::new(),
                })
            };
            let Some(part) = part else { return Ok(false) };
            if scheme.is_none() {
                scheme = Some(part.scheme);
                in_dim = Some(part.in_dim);
            } else if scheme != Some(part.scheme) || in_dim != Some(part.in_dim) {
                return Ok(false);
            }
            bytes.extend_from_slice(&part.w_q);
            out_dim += part.out_dim;
        }
        let (Some(scheme), Some(in_dim)) = (scheme, in_dim) else {
            return Ok(false);
        };
        self.fused.insert(
            name.to_string(),
            GgufPackedLinear {
                w_q: bytes,
                scheme,
                in_dim,
                out_dim,
                bias: Vec::new(),
            },
        );
        Ok(true)
    }
}

/// Split a `name#step` key: a stacked tensor (`[k, n, steps]`, the code
/// predictor's LM heads) read one step at a time. `None` for a plain name — and
/// a name is only ever taken this way by a stage that means it.
fn split_step(key: &str) -> Option<(&str, usize)> {
    let (name, step) = key.rsplit_once('#')?;
    if step.is_empty() || !step.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((name, step.parse().ok()?))
}

impl WeightSource for LoaderWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some((name, step)) = split_step(key) {
            if transpose {
                bail!("`{key}`: transpose is only defined for a whole 2-D weight");
            }
            return self.take_step(name, step);
        }
        let p = self.model.plan(key)?.clone();
        if p.ggml_shape.len() == 2 {
            // A 2-D weight is handed over in the matmul layout `[k, n]` with `n`
            // fastest — the one place the file's element order is rearranged.
            let (vals, k, n) = self.model.linear_f32(key)?;
            if transpose {
                let mut out = vec![0f32; k * n];
                for i in 0..k {
                    for j in 0..n {
                        out[j * k + i] = vals[i * n + j];
                    }
                }
                return Ok((out, vec![n, k]));
            }
            return Ok((vals, vec![k, n]));
        }
        // Everything else keeps the file's element order (reversing the dims is
        // what makes that the RLX row-major order).
        let (vals, dims) = self.model.f32_file_order(key)?;
        if transpose {
            bail!("`{key}`: transpose is only defined for 2-D weights");
        }
        Ok((vals, dims))
    }

    fn has(&self, key: &str) -> bool {
        match split_step(key) {
            Some((name, _)) => self.model.has(name),
            None => self.model.has(key),
        }
    }

    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        if let Some(v) = self.fused.get(key) {
            return Ok(Some(v.clone()));
        }
        if let Some((name, step)) = split_step(key) {
            return self.take_step_packed(name, step);
        }
        let p = self.model.plan(key)?.clone();
        match p.store {
            // The file's own bytes, as the body hands them over.
            Store::Packed => {
                let scheme = loader::quant_of(p.file)
                    .ok_or_else(|| anyhow!("`{key}` is planned packed but has no scheme"))?;
                let (in_dim, out_dim) = p.in_out();
                Ok(Some(GgufPackedLinear {
                    w_q: self.model.raw(key)?,
                    scheme,
                    in_dim,
                    out_dim,
                    bias: Vec::new(),
                }))
            }
            // Host-dequant and F16 tensors deliberately do not take this path.
            Store::HostDequant | Store::F32File | Store::F16Widened => Ok(None),
        }
    }
}

impl LoaderWeights {
    /// One step of a stacked tensor as f32 in the matmul layout `[k, n]`
    /// (`n` fastest) — the fallback when the stack is not packed.
    fn take_step(&mut self, name: &str, step: usize) -> Result<(Vec<f32>, Vec<usize>)> {
        let p = self.model.plan(name)?.clone();
        let (k, n) = (p.ggml_shape[0], p.ggml_shape[1]);
        let (vals, _) = self.model.f32_file_order(name)?;
        let w = linear_slice(&vals, step, k, n);
        Ok((w, vec![k, n]))
    }

    /// The same step, kept packed: the step slice lies contiguously in the
    /// file's order (`[out, in]`, `in` fastest, then the fifteen steps), so the
    /// file's own bytes are handed over without re-quantising or widening.
    fn take_step_packed(&mut self, name: &str, step: usize) -> Result<Option<GgufPackedLinear>> {
        let p = self.model.plan(name)?.clone();
        if p.class != loader::Class::Stacked {
            bail!("`{name}#{step}`: `{name}` is not a stacked tensor");
        }
        let Some(scheme) = loader::quant_of(p.file) else {
            return Ok(None);
        };
        let (k, n) = (p.ggml_shape[0], p.ggml_shape[1]);
        let Some(per_step) = bytes_for_public(p.file, k * n) else {
            return Ok(None);
        };
        let bytes = self.model.raw(name)?;
        let at = step * per_step;
        if at + per_step > bytes.len() {
            bail!(
                "`{name}#{step}`: the file holds {} bytes, step wants {per_step}",
                bytes.len()
            );
        }
        Ok(Some(GgufPackedLinear {
            w_q: bytes[at..at + per_step].to_vec(),
            scheme,
            in_dim: k,
            out_dim: n,
            bias: Vec::new(),
        }))
    }
}

/// The weight source of a **pair**: one `WeightSource` over the body file and
/// its `mmproj`, so a graph that spans both files (mel → logits, say) has one
/// place its weights come from. Which file a key belongs to is decided by the
/// file it is found in — the two files' tensor names do not overlap, which the
/// loader's pair check is what makes sure of.
///
/// The two weight paths behave exactly as in [`LoaderWeights`]: body and sound
/// packed tensors go through `take_packed` (bytes, `DequantMatMul`), host-dequant
/// and F16 tensors through `take` (host f32, plain `MatMul`).
pub struct PairWeights {
    pub body: Arc<Model>,
    pub sound: Arc<Model>,
}

impl PairWeights {
    pub fn new(body: Arc<Model>, sound: Arc<Model>) -> Self {
        Self { body, sound }
    }

    fn file_of(&self, key: &str) -> Result<Arc<Model>> {
        // A `name#step` key belongs to whichever file owns `name`.
        let name = split_step(key).map(|(n, _)| n).unwrap_or(key);
        if self.body.has(name) {
            Ok(self.body.clone())
        } else if self.sound.has(name) {
            Ok(self.sound.clone())
        } else {
            bail!(
                "`{key}` is in neither file of the pair (`{}` / `{}`)",
                self.body.path.display(),
                self.sound.path.display()
            )
        }
    }

    /// The same call on whichever file owns the key.
    fn on_owner<T>(
        &mut self,
        key: &str,
        f: impl FnOnce(&mut LoaderWeights) -> Result<T>,
    ) -> Result<T> {
        let mut one = LoaderWeights::new(self.file_of(key)?);
        f(&mut one)
    }
}

impl WeightSource for PairWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        self.on_owner(key, |w| w.take(key, transpose))
    }

    fn has(&self, key: &str) -> bool {
        let name = split_step(key).map(|(n, _)| n).unwrap_or(key);
        self.body.has(name) || self.sound.has(name)
    }

    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        self.on_owner(key, |w| w.take_packed(key))
    }
}

// ── emission helpers ───────────────────────────────────────────────

/// A `HirMut` view of the flow context's module. Cheap (`&mut HirModule`), so
/// each helper just asks for one instead of threading a borrow around.
pub fn h<'a>(ctx: &'a mut FlowCtx<'_>) -> HirMut<'a> {
    HirMut::new(ctx.hir())
}

/// A constant F32 tensor — the zeros of the upsample insertion and of every
/// norm's beta.
pub fn constant(ctx: &mut FlowCtx<'_>, dims: &[usize], value: f32) -> HirNodeId {
    let n: usize = dims.iter().product();
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&value.to_le_bytes());
    }
    ctx.hir()
        .mir(Op::Constant { data }, vec![], Shape::new(dims, DType::F32))
}

pub fn zeros(ctx: &mut FlowCtx<'_>, dims: &[usize]) -> HirNodeId {
    constant(ctx, dims, 0.0)
}

/// A tensor of the file, F32, declared at `dims` (its own element order). One
/// leaf a name: a tensor already in the graph is handed back, file untouched
/// (see `declared_param`).
pub fn f32_param(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    name: &str,
    dims: &[usize],
) -> Result<HirNodeId> {
    if let Some(id) = declared_param(ctx, name, &Shape::new(dims, DType::F32)) {
        return Ok(id);
    }
    let (vals, file_dims) = model.f32_file_order(name)?;
    let want: usize = dims.iter().product();
    if vals.len() != want {
        bail!(
            "`{name}`: {want} values wanted for {dims:?}, file has {} ({file_dims:?})",
            vals.len()
        );
    }
    Ok(synth_param(ctx, name, vals, dims))
}

/// A prefix of a tensor's values (the first `count` of its own order): how
/// `position_embd` contributes only the positions a window needs, and how a
/// stacked codebook gives up one level. One leaf a name, as [`f32_param`].
pub fn prefix_param(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    name: &str,
    dims: &[usize],
    count: usize,
) -> Result<HirNodeId> {
    if let Some(id) = declared_param(ctx, name, &Shape::new(dims, DType::F32)) {
        return Ok(id);
    }
    let (vals, _) = model.f32_file_order(name)?;
    if count > vals.len() {
        bail!(
            "`{name}`: prefix {count} beyond the tensor's {} values",
            vals.len()
        );
    }
    if dims.iter().product::<usize>() != count {
        bail!("`{name}`: prefix {count} does not fill {dims:?}");
    }
    Ok(synth_param(ctx, name, vals[..count].to_vec(), dims))
}

/// A convolution kernel as the graph declares it: a rank-4 tensor as it is, a
/// rank-3 one (a 1-D convolution) with the unit axis inserted before the kernel,
/// so the kernel stays `[1, k]` and never `[k, k]`.
pub fn conv_param(ctx: &mut FlowCtx<'_>, model: &Model, name: &str) -> Result<HirNodeId> {
    let p = model.plan(name)?.clone();
    let dims = match p.dims.len() {
        4 => p.dims.clone(),
        3 => {
            let mut d = p.dims.clone();
            let k = d.pop().expect("kernel");
            d.push(1);
            d.push(k);
            d
        }
        other => bail!("`{name}`: conv weight of rank {other}"),
    };
    f32_param(ctx, model, name, &dims)
}

/// A 1-D convolution with a stride and a dilation on the ordinary `Op::Conv`
/// path. The HIR helper fixes dilation to `[1, 1]`; the op has carried a real
/// dilation field all along, so a dilated kernel does not have to be spread with
/// zeros (which costs both weights and multiplications).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_dilated(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    out_shape: Shape,
) -> HirNodeId {
    h(ctx).inner().mir(
        Op::Conv {
            kernel_size: kernel_size.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
            groups,
        },
        vec![x, weight],
        out_shape,
    )
}

/// A ConvTranspose kernel as the op wants it: `[C_in, C_out, 1, k]` in the
/// file's own order. The backend's transposed convolution scatters the kernel
/// forward, so it is **not** flipped.
pub fn conv_transpose_param(
    ctx: &mut FlowCtx<'_>,
    model: &Model,
    name: &str,
) -> Result<(HirNodeId, usize, usize, usize)> {
    let p = model.plan(name)?.clone();
    if p.dims.len() != 3 {
        bail!(
            "`{name}`: expected a rank-3 ConvTranspose kernel, got {:?}",
            p.dims
        );
    }
    let (cin, cout, k) = (p.dims[0], p.dims[1], p.dims[2]);
    let (vals, _) = model.f32_file_order(name)?;
    let node = synth_param(ctx, name, vals, &[cin, cout, 1, k]);
    Ok((node, cin, cout, k))
}

/// A linear projection: `FlowCtx::linear` picks the packed path for body
/// weights and the f32 `MatMul` for sound ones — one call site, two paths.
pub fn linear(
    ctx: &mut FlowCtx<'_>,
    x: &rlx_flow::FlowValue,
    key: &str,
) -> Result<rlx_flow::FlowValue> {
    ctx.linear(x, key, false)
}

/// One fused multiply, split along its output rows into `dims` — q/k/v or
/// gate/up. The split is a `narrow` per part; the weight pass is one node.
pub fn linear_fused(
    ctx: &mut FlowCtx<'_>,
    x: &rlx_flow::FlowValue,
    key: &str,
    dims: &[usize],
) -> Result<Vec<HirNodeId>> {
    let fused = linear(ctx, x, key)?;
    let axis = ctx.node_shape(fused.hir_id())?.rank() - 1;
    let mut out = Vec::with_capacity(dims.len());
    let mut start = 0;
    for &len in dims {
        out.push(narrow(ctx, fused.hir_id(), axis, start, len));
        start += len;
    }
    Ok(out)
}

/// One block's three attention projections, one call site: each key through
/// [`linear`] on its own — three multiplies.
pub fn qkv(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    m: usize,
    d: usize,
    keys: [&str; 3],
) -> Result<[HirNodeId; 3]> {
    let value = rlx_flow::FlowValue::new(x, Shape::new(&[m, d], DType::F32));
    let mut out = [x; 3];
    for (i, key) in keys.iter().enumerate() {
        out[i] = linear(ctx, &value, key)?.hir_id();
    }
    Ok(out)
}

/// A bias tensor as an NCHW broadcastable `[1, C, 1, 1]`.
pub fn conv_bias(ctx: &mut FlowCtx<'_>, model: &Model, name: &str, c: usize) -> Result<HirNodeId> {
    f32_param(ctx, model, name, &[1, c, 1, 1])
}

/// `[1, C, 1, L]` → `[L, C]`.
pub fn to_lc(ctx: &mut FlowCtx<'_>, x: HirNodeId, c: usize, l: usize) -> HirNodeId {
    let t = h(ctx).reshape_(x, vec![c as i64, l as i64]);
    h(ctx).transpose_(t, vec![1, 0])
}

/// SnakeBeta as the file stores it: `x + beta · sin²(alpha · x)`. The GGUF's
/// `alpha` tensor already holds `exp(alpha)` and `beta` holds `exp(-beta)`, which
/// is the reference's inverse magnitude; applying `exp` here again would invert
/// it and is the difference that made the codec waveform wrong.
pub fn snake_beta(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    alpha: HirNodeId,
    beta: HirNodeId,
    c: usize,
    l: usize,
) -> HirNodeId {
    let full = Shape::new(&[1, c, 1, l], DType::F32);
    let mut g = h(ctx);
    let a = g.reshape_(alpha, vec![1, c as i64, 1, 1]);
    let b = g.reshape_(beta, vec![1, c as i64, 1, 1]);
    let arg = g.mul(a, x);
    let s = g.activation(Activation::Sin, arg, full);
    let s2 = g.mul(s, s);
    let scale = g.mul(b, s2);
    g.add(x, scale)
}

/// `y = softmax(q·kᵀ/√d)·v` by the core's own attention op.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &mut FlowCtx<'_>,
    q: HirNodeId,
    k: HirNodeId,
    v: HirNodeId,
    heads: usize,
    head_dim: usize,
    kind: MaskKind,
    out: &[usize],
) -> HirNodeId {
    h(ctx).attention_kind(q, k, v, heads, head_dim, kind, Shape::new(out, DType::F32))
}

/// RoPE tables `[rows, head_dim/2]` computed at the file's base (the checkpoints
/// do not carry them). These are the graph's **synthetic constants** (f32): the
/// wave transformer consumes a whole window at once. The body's cached steps use
/// [`crate::host::rope_tables_at`] instead — global positions and f64, so a long
/// walk does not accumulate f32 phase drift.
pub fn rope_tables(
    ctx: &mut FlowCtx<'_>,
    name: &str,
    rows: usize,
    hd: usize,
    base: f32,
) -> (HirNodeId, HirNodeId) {
    let half = hd / 2;
    let mut cos = vec![0f32; rows * half];
    let mut sin = vec![0f32; rows * half];
    for p in 0..rows {
        for i in 0..half {
            let theta = p as f32 / base.powf(2.0 * i as f32 / hd as f32);
            cos[p * half + i] = theta.cos();
            sin[p * half + i] = theta.sin();
        }
    }
    let c = synth_param(ctx, &format!("{name}.cos"), cos, &[rows, half]);
    let s = synth_param(ctx, &format!("{name}.sin"), sin, &[rows, half]);
    (c, s)
}

/// `x + y` with the shape of `x` (broadcast where the caller means it to).
pub fn add(ctx: &mut FlowCtx<'_>, x: HirNodeId, y: HirNodeId) -> HirNodeId {
    h(ctx).add(x, y)
}

pub fn mul(ctx: &mut FlowCtx<'_>, x: HirNodeId, y: HirNodeId) -> HirNodeId {
    h(ctx).mul(x, y)
}

pub fn sub(ctx: &mut FlowCtx<'_>, x: HirNodeId, y: HirNodeId) -> HirNodeId {
    h(ctx).sub(x, y)
}

pub fn gelu(ctx: &mut FlowCtx<'_>, x: HirNodeId, dims: &[usize]) -> HirNodeId {
    h(ctx).activation(Activation::Gelu, x, Shape::new(dims, DType::F32))
}

pub fn silu(ctx: &mut FlowCtx<'_>, x: HirNodeId, dims: &[usize]) -> HirNodeId {
    h(ctx).activation(Activation::Silu, x, Shape::new(dims, DType::F32))
}

pub fn ln(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    beta: HirNodeId,
    eps: f32,
) -> HirNodeId {
    h(ctx).ln(x, gamma, beta, eps)
}

pub fn rms_norm(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    eps: f32,
    dims: &[usize],
) -> HirNodeId {
    let beta = zeros(ctx, &[dims[dims.len() - 1]]);
    h(ctx).rms_norm(x, gamma, beta, eps)
}

pub fn reshape(ctx: &mut FlowCtx<'_>, x: HirNodeId, dims: &[i64]) -> HirNodeId {
    h(ctx).reshape_(x, dims.to_vec())
}

pub fn narrow(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    axis: usize,
    start: usize,
    len: usize,
) -> HirNodeId {
    h(ctx).narrow_(x, axis, start, len)
}

/// Gather rows of a table by index (`axis = 0`).
pub fn gather(ctx: &mut FlowCtx<'_>, table: HirNodeId, indices: HirNodeId) -> HirNodeId {
    h(ctx).gather_(table, indices, 0)
}

/// An elementwise activation the sound stages need beyond the body's set
/// (`ReLU`/`tanh`/`sigmoid`/`sqrt` in the speaker encoder; `Log` in the mel
/// front's log10).
pub fn act(ctx: &mut FlowCtx<'_>, kind: Activation, x: HirNodeId, dims: &[usize]) -> HirNodeId {
    h(ctx).activation(kind, x, Shape::new(dims, DType::F32))
}

/// `max` over `axes`, with the axes kept as singletons (`out` says the shape) —
/// the mel front's clamp threshold is the maximum over the whole utterance, and
/// that is the one reduction `HirMut` only offers as `sum`/`mean`.
pub fn reduce_max(ctx: &mut FlowCtx<'_>, x: HirNodeId, axes: &[usize], out: &[usize]) -> HirNodeId {
    h(ctx).add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: axes.to_vec(),
            keep_dim: true,
        },
        vec![x],
        Shape::new(out, DType::F32),
    )
}

/// Broadcast to a target shape (NumPy rules — the size-1 axes expand).
pub fn expand(ctx: &mut FlowCtx<'_>, x: HirNodeId, target: &[usize]) -> HirNodeId {
    h(ctx).expand_(x, target.iter().map(|d| *d as i64).collect())
}

/// Pad one axis (`Constant(0)` for the causal left context, `Reflect` for the
/// speaker encoder's `same` convolutions — the reference pads reflect).
pub fn pad_axis(
    ctx: &mut FlowCtx<'_>,
    x: HirNodeId,
    rank: usize,
    axis: usize,
    before: usize,
    after: usize,
    mode: PadMode,
) -> HirNodeId {
    if before == 0 && after == 0 {
        return x;
    }
    let mut pads = vec![[0usize, 0]; rank];
    pads[axis] = [before, after];
    h(ctx).pad_(x, pads, mode)
}

/// One 2-D slice of a stacked `[k, n, steps]` file tensor, rearranged into the
/// layout `MatMul` wants (`[k, n]` row-major, `n` fastest) — the same move
/// [`Model::linear_f32`](crate::loader::Model::linear_f32) makes for a whole
/// weight, one step at a time (the code predictor's per-step LM heads).
pub fn linear_slice(values: &[f32], step: usize, k: usize, n: usize) -> Vec<f32> {
    let base = step * k * n;
    let mut out = vec![0f32; k * n];
    for j in 0..n {
        for i in 0..k {
            out[i * n + j] = values[base + j * k + i];
        }
    }
    out
}

/// One table of a stacked `[d, rows, steps]` file tensor: step `step`'s
/// `[rows, d]` table in row-major order (`d` fastest) — already the layout
/// `Gather` reads rows in (the code predictor's per-step code embeddings and
/// the RVQ codebooks).
pub fn table_slice(values: &[f32], step: usize, rows: usize, cols: usize) -> Vec<f32> {
    values[step * rows * cols..(step + 1) * rows * cols].to_vec()
}

/// A `Param` node for values the stage built itself (a slice, a table).
pub fn sliced_param(
    ctx: &mut FlowCtx<'_>,
    name: &str,
    step: usize,
    values: Vec<f32>,
    dims: &[usize],
) -> HirNodeId {
    synth_param(ctx, &format!("{name}#{step}"), values, dims)
}

/// A `Param` node for values the stage computed itself, under its own name —
/// **one leaf a name**: a second call under the same name and shape returns the
/// leaf already in the graph, so the one node every reader shares is the one a
/// binding feeds (see `declared_param`).
pub fn synth_param(
    ctx: &mut FlowCtx<'_>,
    name: &str,
    values: Vec<f32>,
    dims: &[usize],
) -> HirNodeId {
    let shape = Shape::new(dims, DType::F32);
    if let Some(id) = declared_param(ctx, name, &shape) {
        return id;
    }
    ctx.synth_param(name, values, shape)
}

/// The `Param` leaf already declared under this name and shape, if any.
///
/// A leaf is bound **by name** and a binding reaches **one** node (the pinned
/// fork's `verify` calls `rlx_ir::verify_unique_leaf_names`, so a duplicate is a
/// graph error), so a second node under a name never gets its bytes and reads
/// zeros. The flow keeps one node a name for the weights it loads
/// (`FlowCtx::load_param`), but `synth_param` declares a fresh leaf on every
/// call — and the predictor's steps are unrolled, so the same f32 parameters are
/// asked for once per step and frame. Asking the module for the leaf already
/// under the name is that same rule for the stand's own leaves. The **last**
/// declaration is the one a binding would win, so that is the one reused; a name
/// that reappears with another shape is left alone — those are genuinely
/// different leaves (the per-frame RoPE tables, the per-step slices, which differ
/// by name already).
fn declared_param(ctx: &mut FlowCtx<'_>, name: &str, shape: &Shape) -> Option<HirNodeId> {
    ctx.hir().nodes().iter().rev().find_map(|n| match &n.op {
        HirOp::Param { name: p } if p == name && &n.shape == shape => Some(n.id),
        _ => None,
    })
}
