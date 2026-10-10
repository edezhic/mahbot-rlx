// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused (residual add + optional bias add) + RMSNorm.
//
//   y = rms_norm(x + residual + [bias], gamma, beta)
//
// ONE WORKGROUP (64 threads) PER ROW, shared-memory tree reduction over the
// feature dim — see `fused_residual_ln.wgsl` for why, and for the identical
// lane mapping across the passes that keeps the in-place fold sound.
//
// Inputs (offsets in f32 elements):
//   in_off:       [outer, inner]
//   residual_off: [outer, inner]
//   bias_off:     [inner]   (only read when has_bias != 0)
//   gamma_off:    [inner]
//   beta_off:     [inner]
// Output:
//   out_off:      [outer, inner]

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> scratch: array<f32, 64>;

// Sum-reduce `scratch[0..64]` into `scratch[0]`. Caller must barrier before.
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
fn fused_residual_rms_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let out_base = params.out_off      + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1a: fuse residual (+ bias) into the output slot and accumulate this
    // lane's share of the sum.
    var partial: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.inner) { break; }
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[out_base + i] = v;
        partial = partial + v;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // all lanes read scratch[0] before Pass 1b overwrites it.

    // Pass 1b: STABLE TWO-PASS mean(x²) = mean((x − mean)²) + mean² (see the RmsNorm
    // note in layernorm.wgsl) — a large per-row DC offset otherwise swamps the
    // one-pass f32 Σx² and drifts vs CPU's f64, corrupting the per-row scale on wgpu.
    partial = 0.0;
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let d = arena[out_base + i] - mean;
        partial = partial + d * d;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let inv_rms = inverseSqrt(scratch[0] * n_inv + mean * mean + eps);

    // Pass 2: scale by gamma and shift by beta — each lane over its own lanes.
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off + i];
        arena[out_base + i] = arena[out_base + i] * inv_rms * g + b;
        i = i + 64u;
    }
}
