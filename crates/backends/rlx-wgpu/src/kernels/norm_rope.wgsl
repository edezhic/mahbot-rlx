// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RmsNorm → RoPE in ONE dispatch, one workgroup per (position, head) row.
//
// A transformer layer walks `narrow(qkv) → RmsNorm(q/k head) → RoPE`. In a
// one-row decode frame each of those is a whole dispatch over a one-row
// activation, and q and k together are four of them. This kernel is the same
// normalization and the same rotation as `layernorm.wgsl`'s RmsNorm arm and
// `rope.wgsl`'s NeoX arm — the same passes, the same lane mapping, the same
// expression order — so folding them changes the dispatch count and not the
// numbers.
//
// Two parts ride in one dispatch when they share a position table (a block's q
// and k): workgroup `row < rows_a` normalizes+rotates part A, the rest part B.
// `row` is uniform across a workgroup, so the part branch never diverges.
//
// A third stretch of workgroups copies the split's leftover piece (the v of a
// fused qkv row) verbatim, so the cut that fed this step and the pair it feeds
// cost one dispatch rather than two. `copy_len == 0` adds none.
//
// Admission (lower.rs) requires the full NeoX rotation (`n_rot == head_dim`,
// even), a rank-3/2 operand with `batch == 1`, a norm whose inner extent is the
// head width and whose outer extent is `seq × heads`, and a single consumer of
// the norm.
struct Params {
    seq: u32,             // walked positions; scaled by the active extent
    heads_a: u32,         // 0 ⇒ this part is absent
    heads_b: u32,
    inner: u32,           // head width (even); rotation pair = (j, j + inner/2)
    cos_row_stride: u32,  // the table's own row width (head_dim / 2 here)
    eps_bits: u32,
    copy_in: u32,         // the split's leftover piece, copied verbatim
    copy_out: u32,
    copy_len: u32,        // 0 ⇒ no copy
    in_a: u32,
    out_a: u32,
    gamma_a: u32,
    beta_a: u32,
    in_b: u32,
    out_b: u32,
    gamma_b: u32,
    beta_b: u32,
    cos_off: u32,
    sin_off: u32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
var<workgroup> scratch: array<f32, 64>;
fn tree_sum(tid: u32) {
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
}
@compute @workgroup_size(64)
fn norm_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    let rows_a = params.seq * params.heads_a;
    let rows_b = params.seq * params.heads_b;
    let rows_copy = (params.copy_len + 63u) / 64u;
    if (row >= rows_a + rows_b + rows_copy) { return; }
    let tid = lid.x;
    if (row >= rows_a + rows_b) {
        // The split's leftover piece: the row's own bytes, one element a lane.
        let i = (row - rows_a - rows_b) * 64u + tid;
        if (i < params.copy_len) {
            arena[params.copy_out + i] = arena[params.copy_in + i];
        }
        return;
    }
    if (params.inner == 0u) { return; }
    var in_off = params.in_a;
    var out_off = params.out_a;
    var gamma_off = params.gamma_a;
    var beta_off = params.beta_a;
    var heads = params.heads_a;
    var local = row;
    if (row >= rows_a) {
        in_off = params.in_b;
        out_off = params.out_b;
        gamma_off = params.gamma_b;
        beta_off = params.beta_b;
        heads = params.heads_b;
        local = row - rows_a;
    }
    let in_base = in_off + local * params.inner;
    let out_base = out_off + local * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);

    // ── Pass 1: mean = Σx / n (tree). ──
    var partial: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.inner) { break; }
        partial = partial + arena[in_base + i];
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier();

    // ── Pass 2: mean of squared deviation (tree), the stable two-pass form. ──
    partial = 0.0;
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let d = arena[in_base + i] - mean;
        partial = partial + d * d;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let dev_mean = scratch[0] * n_inv;
    let inv_rms = inverseSqrt(dev_mean + mean * mean + eps);

    // ── Pass 3: normalize, then rotate the pair (j, j + half) as `rope.wgsl`'s
    // NeoX arm does — one lane owns both halves of a pair, so no exchange. ──
    let half = params.inner / 2u;
    let pos = local / heads;
    let cs = pos * params.cos_row_stride;
    var j: u32 = tid;
    loop {
        if (j >= half) { break; }
        let j2 = j + half;
        let y1 = arena[in_base + j] * inv_rms * arena[gamma_off + j] + arena[beta_off + j];
        let y2 = arena[in_base + j2] * inv_rms * arena[gamma_off + j2] + arena[beta_off + j2];
        let c = arena[params.cos_off + cs + j];
        let s = arena[params.sin_off + cs + j];
        arena[out_base + j] = y1 * c - y2 * s;
        arena[out_base + j2] = y2 * c + y1 * s;
        j = j + 64u;
    }
}
