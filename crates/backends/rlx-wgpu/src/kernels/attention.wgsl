// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Single-pass online-softmax SDPA (FlashAttention v1 inner-row form).
// One thread per (batch, head, q_row); each thread walks the K dimension
// exactly once, maintaining a running (m, l, O[D]) tuple:
//
//   m_new = max(m, s)
//   e_old = exp(m - m_new); e_cur = exp(s - m_new)
//   l     = e_old · l + e_cur
//   O[d]  = e_old · O[d] + e_cur · V[s][d]
//   m     = m_new
//
// At end: out[d] = O[d] / l. This does ≈ seq_k · 2·head_dim FMAs per row
// vs. the 3-pass form's ≈ seq_k · 4·head_dim — measured ~2.6× faster on
// MiniLM6/bge-* shapes — and the running max/scale rebase is also more
// numerically stable than the 3-pass "max then sum_exp then weighted sum".
//
// Inputs all live in the arena as [B, H, S, D] f32 tensors.
// Mask kinds:
//   2 = binary key-padding mask (Custom): element < 0.5 → score = -1e9.
//   4 = additive bias mask (Bias): score += mask element (e.g. a
//       block-diagonal window bias carrying 0 / large-negative).
// Both index the mask via the per-axis mask strides below. Caller is
// responsible for normalizing other shapes upstream.
//
// `O` is held in a per-thread private array<f32, MAX_HEAD_DIM>. Gemma 3
// decode uses head_dim=256; keep headroom for Llama-class 128 and future
// 512-dim heads without spilling on Apple GPUs (private limit ~10 KiB/thread).
//
// This module also carries a decode-shaped variant, `attention_decode`: one
// workgroup per (batch, head) with the key sequence walked across the
// workgroup's lanes and the lane partials merged in workgroup memory. The
// serial form above is the wrong shape for a decode step — one query row means
// one workgroup of 64 lanes of which only batch·heads are live, each walking
// the whole context alone. The host picks the variant from the extent the run
// will actually walk (`kernels::attention_decode_eligible`).
//
// The third entry point, `attention_mask`, is the many-row route's masked
// scale: the same scale, cap and mask the two above apply to a score, over a
// dense score matrix a multiply just wrote, in front of the ordinary dense
// softmax.

// The widest head the per-thread Q/O arrays below hold. A shape above it is
// refused at plan time (`backend::attention_route`), so the early return that
// reads this is a backstop, not the refusal.
const MAX_HEAD_DIM: u32 = 512u;

// Decode-shaped variant: workgroup width == key lanes, and the value dims merged
// per pass. DECODE_LANES * DECODE_DIM_CHUNK f32 is 8 KiB of workgroup storage
// (8.26 KiB with the merge scratch), half of WebGPU's guaranteed 16 KiB
// `maxComputeWorkgroupStorageSize` — the native backend's equivalent needs
// 16.25 KiB and is above that floor.
const DECODE_LANES: u32 = 64u;
const DECODE_DIM_CHUNK: u32 = 32u;

// The score a key position contributes after this operation's logit cap and
// mask: the same cap, sentinels, threshold and window for every entry point
// that applies one. `cap` is `attn_logit_softcap`'s own `c · tanh(s / c)`, over
// the scaled score and *before* the mask/bias — the operation's definition, not
// this kernel's order — and is disabled at 0 or below. `mask` is the value the
// caller loaded — only kinds 2 and 4 read the mask tensor, so every other kind
// passes 0 — and is read only for those two.
fn mask_score(score: f32, cap: f32, i: u32, q_abs: u32, kind: u32, window: u32, mask: f32) -> f32 {
    var out = score;
    if (cap > 0.0) { out = cap * tanh(out / cap); }
    if (kind == 1u) {
        if (i > q_abs) { out = -3.4e38; }
    } else if (kind == 2u) {
        // BERT-style binary multiplicative mask (1=valid, 0=padding).
        // Matches CPU/Metal: a position with mask < 0.5 → score driven
        // to -1e9. Hardcoded 0.5 keeps parity across backends.
        if (mask < 0.5) { out = -1e9; }
    } else if (kind == 3u) {
        if (i > q_abs) { out = -3.4e38; }
        else if (q_abs - i > window) { out = -3.4e38; }
    } else if (kind == 4u) {
        // Additive bias mask (e.g. block-diagonal window bias): the mask
        // carries additive values (0 to attend, large-negative to drop)
        // added to the score pre-softmax — NOT a 0/1 indicator.
        out = out + mask;
    }
    return out;
}

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,

    out_off: u32,
    mask_off: u32,
    mask_kind: u32,    // 0=None, 1=Causal, 2=Custom(binary), 3=SlidingWindow, 4=Bias(additive)
    scale_bits: u32,   // bitcast<f32>(1/sqrt(D))
    window: u32,       // SlidingWindow width (only used when mask_kind == 3)
    // MASK address strides. The kernel computes:
    //   mask_addr = mask_off
    //             + b  * mask_batch_stride
    //             + h  * mask_head_stride
    //             + qi * seq_q_stride
    //             + s  * seq_k_stride
    // Setting head/q strides to 0 lets the kernel read a [B, S]
    // padding mask directly without materializing the [B, H, S_q, S_k]
    // broadcast (saves the Expand pre-pass per attention block).
    seq_q_stride: u32,
    seq_k_stride: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    kv_heads: u32,     // GQA/MQA: #KV heads query heads share (heads=MHA, 0=unset→MHA)
    // Asymmetric SDPA (DeepSeek/Kimi MLA): V rows + output are `v_head_dim`
    // wide while Q/K scores still use `head_dim`. Equals `head_dim` for
    // ordinary symmetric attention; 0 (zero-init) falls back to `head_dim`.
    v_head_dim: u32,
    // `attn_logit_softcap`, bitcast: `c · tanh(s / c)` over the scaled score,
    // before the mask. 0 or below is disabled.
    logit_softcap_bits: u32,

    // Per-tensor strides (in f32 elements). Q/K/V/out can be in either
    // [B, H, S, D] or [B, S, H, D] layout — caller sets the strides.
    q_batch_stride: u32, q_head_stride: u32, q_seq_stride: u32, _pad_q: u32,
    k_batch_stride: u32, k_head_stride: u32, k_seq_stride: u32, _pad_k: u32,
    v_batch_stride: u32, v_head_stride: u32, v_seq_stride: u32, _pad_v: u32,
    o_batch_stride: u32, o_head_stride: u32, o_seq_stride: u32, _pad_o: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.batch * params.heads * params.seq_q;
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= total) { return; }
    let qi = row % params.seq_q;
    let q1 = row / params.seq_q;
    let h  = q1 % params.heads;
    let b  = q1 / params.heads;
    let scale = bitcast<f32>(params.scale_bits);
    let cap = bitcast<f32>(params.logit_softcap_bits);
    // Absolute query position for causal/window masking. In a decode step
    // seq_q=1 but the query sits at position seq_k-1 (past KV preceded it), so
    // causality must compare against `qi + (seq_k - seq_q)`, not the local
    // `qi`. Prefill (seq_q == seq_k) leaves this as `qi`.
    let q_abs = qi + select(0u, params.seq_k - params.seq_q, params.seq_k >= params.seq_q);

    // Mask address uses generic per-axis strides. Each axis is folded
    // independently; setting head/q strides to 0 lets us read a
    // broadcast mask without materializing it. The s-dependent part
    // is left to the inner loop.
    let mask_partial = params.mask_off
        + b  * params.mask_batch_stride
        + h  * params.mask_head_stride
        + qi * params.seq_q_stride;

    // Q, K, V, output base addresses use explicit per-axis strides so the
    // kernel works with [B, H, S, D] OR [B, S, H, D] layout uniformly.
    let q_base = params.q_off
        + b * params.q_batch_stride
        + h * params.q_head_stride
        + qi * params.q_seq_stride;
    // GQA / MQA: several query heads share one KV head. Map query head `h` to
    // its KV head. kv_heads is always set (== heads for plain MHA → group 1);
    // the max() guards protect against a stray 0.
    let kv_denom = max(params.kv_heads, 1u);
    let kv_h = h / max(params.heads / kv_denom, 1u);
    let k_bh   = params.k_off + b * params.k_batch_stride + kv_h * params.k_head_stride;
    let v_bh   = params.v_off + b * params.v_batch_stride + kv_h * params.v_head_stride;
    let o_base = params.out_off
        + b * params.o_batch_stride
        + h * params.o_head_stride
        + qi * params.o_seq_stride;

    let hd = params.head_dim;
    // V/output per-head width. For symmetric SDPA this equals `head_dim`;
    // for MLA the V rows and output are `v_head_dim` wide while the Q·K
    // score still contracts over `head_dim`. A zero-init params value
    // (older callers / defaulted uniforms) falls back to `head_dim`.
    let hd_v = select(params.v_head_dim, hd, params.v_head_dim == 0u);
    if (hd > MAX_HEAD_DIM || hd_v > MAX_HEAD_DIM) { return; }

    // Cache Q[qi, :] in registers — read seq_k times by the dot product.
    var q_reg: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        q_reg[d] = arena[q_base + d];
    }

    // Online softmax accumulators. `o` spans the V-width `hd_v`.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;
    var o: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd_v; d = d + 1u) { o[d] = 0.0; }

    for (var s: u32 = 0u; s < params.seq_k; s = s + 1u) {
        // Score: scale · Q·K[s], the logit cap, then the mask.
        let k_base = k_bh + s * params.k_seq_stride;
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            score = score + q_reg[d] * arena[k_base + d];
        }
        score = score * scale;
        // Only kinds 2 and 4 read the mask tensor, so the others must not
        // form a mask address at all.
        var mask_v: f32 = 0.0;
        if (params.mask_kind == 2u || params.mask_kind == 4u) {
            mask_v = arena[mask_partial + s * params.seq_k_stride];
        }
        score = mask_score(score, cap, s, q_abs, params.mask_kind, params.window, mask_v);

        // Online softmax update.
        let m_new = max(m, score);
        let e_old = exp(m - m_new);
        let e_cur = exp(score - m_new);
        l = e_old * l + e_cur;
        // V rows are `hd_v` wide (v_seq_stride already reflects that width).
        let v_base = v_bh + s * params.v_seq_stride;
        for (var d: u32 = 0u; d < hd_v; d = d + 1u) {
            o[d] = e_old * o[d] + e_cur * arena[v_base + d];
        }
        m = m_new;
    }

    // Normalize and emit `hd_v`-wide output rows. l is guaranteed > 0
    // (at least one finite score).
    let inv_l = 1.0 / l;
    for (var d: u32 = 0u; d < hd_v; d = d + 1u) {
        arena[o_base + d] = o[d] * inv_l;
    }
}

// Merge scratch: [0] merged max, [1] merged sum.
var<workgroup> dec_pub: array<f32, 2u>;
// Per-lane partial, reused: first the lane's running max, then its weighted sum.
var<workgroup> dec_lane: array<f32, DECODE_LANES>;
// Per-lane weighted O partials, `[dim * DECODE_LANES + lane]` (dim-major: the
// writes and the reduction reads are both across consecutive lanes).
var<workgroup> dec_dims: array<f32, DECODE_DIM_CHUNK * DECODE_LANES>;

// One workgroup per (batch, head); `row` is the (batch, head) pair. Every lane
// keeps its own running (m, l, O[]) over the keys it walks, and the merge at the
// bottom is the same online-softmax rebase the serial kernel does serially: a
// lane whose partial sits below the merged max contributes exp(m - max) of its
// own weighting. Only lane 0's reduction of a 64-lane array differs in shape
// from the serial form — at this width a barrier tree is not worth it.
@compute @workgroup_size(DECODE_LANES)
fn attention_decode(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wid3: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let lid = lid3.x;
    let row = wid3.x + wid3.y * ngs.x;
    let total = params.batch * params.heads;
    if (row >= total) { return; }
    let h = row % params.heads;
    let b = row / params.heads;
    let hd = params.head_dim;
    let hd_v = select(params.v_head_dim, hd, params.v_head_dim == 0u);

    let scale = bitcast<f32>(params.scale_bits);
    let cap = bitcast<f32>(params.logit_softcap_bits);
    // seq_q == 1 on this route, so the single query row is qi = 0, sitting at
    // absolute position seq_k - 1 — the same `q_abs` the serial kernel derives.
    let qi = 0u;
    let q_abs = select(0u, params.seq_k - params.seq_q, params.seq_k >= params.seq_q);
    let mask_partial = params.mask_off
        + b  * params.mask_batch_stride
        + h  * params.mask_head_stride
        + qi * params.seq_q_stride;
    let q_base = params.q_off
        + b * params.q_batch_stride
        + h * params.q_head_stride
        + qi * params.q_seq_stride;
    let kv_denom = max(params.kv_heads, 1u);
    let kv_h = h / max(params.heads / kv_denom, 1u);
    let k_bh = params.k_off + b * params.k_batch_stride + kv_h * params.k_head_stride;
    let v_bh = params.v_off + b * params.v_batch_stride + kv_h * params.v_head_stride;
    let o_base = params.out_off
        + b * params.o_batch_stride
        + h * params.o_head_stride
        + qi * params.o_seq_stride;

    var q_reg: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { q_reg[d] = arena[q_base + d]; }

    // Running (m, l, O[]) over this lane's share of the keys: lane `lid` takes
    // keys lid, lid + DECODE_LANES, ... . Zero-fill to a DECODE_DIM_CHUNK
    // multiple so the merge below never reads an indeterminate O element.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;
    var o: array<f32, MAX_HEAD_DIM>;
    let hd_v_pad = (hd_v + DECODE_DIM_CHUNK - 1u) / DECODE_DIM_CHUNK * DECODE_DIM_CHUNK;
    for (var d: u32 = 0u; d < hd_v_pad; d = d + 1u) { o[d] = 0.0; }

    // Mirrors the `attention` entry point's key loop — same cap, same mask
    // kinds, same sentinels, same online-softmax update — with the walk split
    // across lanes. Keep the two in sync. The key loop itself stays inlined
    // rather than
    // factored into a shared helper on purpose: that helper measured ~1.6% on
    // the prefill path. The mask rule it applies is the shared `mask_score`.
    for (var s: u32 = lid; s < params.seq_k; s = s + DECODE_LANES) {
        let k_base = k_bh + s * params.k_seq_stride;
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            score = score + q_reg[d] * arena[k_base + d];
        }
        score = score * scale;
        // Only kinds 2 and 4 read the mask tensor, so the others must not
        // form a mask address at all.
        var mask_v: f32 = 0.0;
        if (params.mask_kind == 2u || params.mask_kind == 4u) {
            mask_v = arena[mask_partial + s * params.seq_k_stride];
        }
        score = mask_score(score, cap, s, q_abs, params.mask_kind, params.window, mask_v);
        let m_new = max(m, score);
        let e_old = exp(m - m_new);
        let e_cur = exp(score - m_new);
        l = e_old * l + e_cur;
        let v_base = v_bh + s * params.v_seq_stride;
        for (var d: u32 = 0u; d < hd_v; d = d + 1u) {
            o[d] = e_old * o[d] + e_cur * arena[v_base + d];
        }
        m = m_new;
    }

    // Merge the lanes. Lane 0 does the two scalar reductions (DECODE_LANES adds
    // each, far cheaper than a barrier tree at this width); `w` rescales each
    // lane's partial onto the merged max. Empty lanes carry l = 0 and contribute
    // nothing, and a fully masked row keeps the serial kernel's l = 1 / O = 0.
    dec_lane[lid] = m;
    workgroupBarrier();
    if (lid == 0u) {
        var mm = -3.4e38;
        for (var i: u32 = 0u; i < DECODE_LANES; i = i + 1u) { mm = max(mm, dec_lane[i]); }
        dec_pub[0] = mm;
    }
    workgroupBarrier();
    let w = exp(m - dec_pub[0]);
    dec_lane[lid] = w * l;
    workgroupBarrier();
    if (lid == 0u) {
        var ll = 0.0;
        for (var i: u32 = 0u; i < DECODE_LANES; i = i + 1u) { ll = ll + dec_lane[i]; }
        dec_pub[1] = ll;
    }
    workgroupBarrier();
    let inv_l = 1.0 / dec_pub[1];

    // Output: value dims in DECODE_DIM_CHUNK-wide passes.
    for (var c: u32 = 0u; c < hd_v; c = c + DECODE_DIM_CHUNK) {
        for (var j: u32 = 0u; j < DECODE_DIM_CHUNK; j = j + 1u) {
            dec_dims[j * DECODE_LANES + lid] = w * o[c + j];
        }
        workgroupBarrier();
        if (lid < DECODE_DIM_CHUNK) {
            let d = c + lid;
            var acc = 0.0;
            for (var i: u32 = 0u; i < DECODE_LANES; i = i + 1u) {
                acc = acc + dec_dims[lid * DECODE_LANES + i];
            }
            if (d < hd_v) { arena[o_base + d] = acc * inv_l; }
        }
        workgroupBarrier();
    }
}

// The many-row route's masked scale (`kernels::attention_mask_kernel`): the
// scale, the logit cap and the five mask kinds the `attention` entry point
// above applies, in place over a [rows, cols] score matrix a dense multiply
// just wrote. One
// 64-lane workgroup per row — the same decomposition `softmax.wgsl` uses, and
// the same grid convention (`row` is `wid.x + wid.y * ngs.x`, so the dispatch
// is `rows` workgroups of 64) — with `cols` unbounded: the row is strided by
// the lane count rather than reduced over a padded extent, so no power-of-two
// width is required.
//
// Rows are (batch, head, query row) major, which is the order the score
// multiply's batch/row axes wrote them in. Row `r` therefore resolves to
// qi = r % seq_q, h = (r / seq_q) % heads, b = r / (seq_q · heads) — the same
// (b, h, qi) triple the fused kernel starts from, with `q_abs` derived from it
// the same way — and the mask is read through the same per-axis strides. The
// scale, the cap and the mask are applied in that order — the fused kernel's own
// order — and the rule itself is the shared `mask_score` both entry points call,
// so a masked position is treated identically on either.
//
// The normalisation that follows is the ordinary dense softmax
// (`softmax.wgsl`), run in place over the same matrix: this pass writes the
// scaled, capped, masked score back so that matrix can stay where the multiply
// left it.
//
// This pass writes every element exactly once and reads only that element, so
// it is elementwise and in-place safe under any dispatch that covers the rows.
struct AttentionMaskParams {
    rows: u32,
    cols: u32,
    seq_q: u32,
    heads: u32,
    scale_bits: u32,
    mask_kind: u32,
    window: u32,
    scores_off: u32,
    mask_off: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    seq_q_stride: u32,
    seq_k_stride: u32,
    // `attn_logit_softcap`, bitcast: `c · tanh(s / c)` over the scaled score,
    // before the mask. 0 or below is disabled.
    logit_softcap_bits: u32,
    _p1: u32, _p2: u32,
};

// The two module-scope bindings of the mask pass. A WGSL module declares one
// binding per (group, binding), and this one carries three entry points that
// share the layout but not the uniform's shape, so the mask entry's own pair is
// named for it (`mask_arena` / `mask_params`); `attention` and
// `attention_decode` above read the fused `Params`.
@group(0) @binding(0) var<storage, read_write> mask_arena: array<f32>;
@group(0) @binding(1) var<uniform>              mask_params: AttentionMaskParams;

@compute @workgroup_size(64)
fn attention_mask(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= mask_params.rows || mask_params.cols == 0u) { return; }
    let tid = lid.x;
    let qi = row % mask_params.seq_q;
    let q1 = row / mask_params.seq_q;
    let h  = q1 % mask_params.heads;
    let b  = q1 / mask_params.heads;
    let scale = bitcast<f32>(mask_params.scale_bits);
    let cap = bitcast<f32>(mask_params.logit_softcap_bits);
    // Same absolute query position as the fused kernel: cols is the key
    // sequence, so the offset is seq_k - seq_q.
    let q_abs = qi + select(0u, mask_params.cols - mask_params.seq_q, mask_params.cols >= mask_params.seq_q);
    let mask_partial = mask_params.mask_off
        + b  * mask_params.mask_batch_stride
        + h  * mask_params.mask_head_stride
        + qi * mask_params.seq_q_stride;
    let base = mask_params.scores_off + row * mask_params.cols;

    // ── Scale, cap then mask, in place: the same score the fused kernel turns
    // into an online-softmax update, before the softmax reads it. ──
    var i: u32 = tid;
    loop {
        if (i >= mask_params.cols) { break; }
        // Only kinds 2 and 4 read the mask tensor, so the others must not
        // form a mask address at all.
        var mask_v: f32 = 0.0;
        if (mask_params.mask_kind == 2u || mask_params.mask_kind == 4u) {
            mask_v = mask_arena[mask_partial + i * mask_params.seq_k_stride];
        }
        let score = mask_arena[base + i] * scale;
        mask_arena[base + i] =
            mask_score(score, cap, i, q_abs, mask_params.mask_kind, mask_params.window, mask_v);
        i = i + 64u;
    }
}
