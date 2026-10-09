// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Channel-first LayerNorm2d on NCHW (candle / SAM semantics): every
// (batch, spatial) position is normalised across the CHANNEL axis, so the
// reduction strides by `hw` where GroupNorm's strides by 1.
//
//   y[n, c, i] = (x[n, c, i] - mean(n, i)) * rsqrt(var(n, i) + eps)
//                * gamma[c] + beta[c]
//
// ONE WORKGROUP PER (batch, spatial) POSITION, 64 threads, shared-memory tree
// reduction over the channel axis — the same shape and the same STABLE
// TWO-PASS variance as `group_norm.wgsl` (subtracting the mean before squaring
// keeps the f32 result from drifting at 1024 channels, where the
// `E[x²] − E[x]²` form cancels catastrophically against a large DC offset).
//
// Offsets are in f32 elements.

struct Params {
    positions: u32,   // n * hw (one workgroup each)
    c: u32,
    hw: u32,
    in_off: u32,
    out_off: u32,
    gamma_off: u32,
    beta_off: u32,
    eps_bits: u32,    // bitcast-encoded f32 eps
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
fn layer_norm2d(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let pos = wid.x + wid.y * ngs.x;
    if (pos >= params.positions || params.c == 0u) { return; }
    let tid = lid.x;

    let ni = pos / params.hw;
    let s = pos - ni * params.hw;
    let plane = params.c * params.hw;
    let in_base = params.in_off + ni * plane + s;
    let out_base = params.out_off + ni * plane + s;
    let n_inv = 1.0 / f32(params.c);
    let eps = bitcast<f32>(params.eps_bits);

    // ── Pass 1: mean over the channel axis. ──
    var partial: f32 = 0.0;
    var ch: u32 = tid;
    loop {
        if (ch >= params.c) { break; }
        partial = partial + arena[in_base + ch * params.hw];
        ch = ch + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // every thread reads scratch[0] before pass 2 reuses it.

    // ── Pass 2: mean squared deviation (stable two-pass). ──
    partial = 0.0;
    ch = tid;
    loop {
        if (ch >= params.c) { break; }
        let d = arena[in_base + ch * params.hw] - mean;
        partial = partial + d * d;
        ch = ch + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let inv_std = inverseSqrt(scratch[0] * n_inv + eps);
    workgroupBarrier();

    // ── Pass 3: scale/shift per channel. ──
    ch = tid;
    loop {
        if (ch >= params.c) { break; }
        let g = arena[params.gamma_off + ch];
        let b = arena[params.beta_off + ch];
        let idx = in_base + ch * params.hw;
        arena[out_base + ch * params.hw] = (arena[idx] - mean) * inv_std * g + b;
        ch = ch + 64u;
    }
}
