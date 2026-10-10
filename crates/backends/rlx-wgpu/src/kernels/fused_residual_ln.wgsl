// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused (residual add + optional bias add) + LayerNorm.
//
//   y = layer_norm(x + residual + [bias])
//
// ONE WORKGROUP (64 threads) PER ROW, shared-memory tree reduction over the
// feature dim — the same shape as the plain `layernorm.wgsl`, and for the same
// reason: a row reduced serially in one thread makes the step's cost grow with
// the row width. Every pass below strides over the same lanes (element index ≡
// lane mod 64), so an arena element is only ever read and written by the lane
// that owns it; the one cross-lane exchange is the reduction, through `scratch`,
// which the barriers order.
//
// Compared to running Add → [Add] → LayerNorm as three separate
// kernels, this saves 2 dispatches and reads/writes the [outer, inner]
// arena slot twice instead of four times.
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
fn fused_residual_ln(
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

    // Pass 1: fold residual (+ bias) into the OUTPUT slot and accumulate this
    // lane's share of the sum. (The summed value x+residual+bias is materialized
    // in out_base, so the variance pass reads it back cheaply.)
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

    // Pass 1b: STABLE TWO-PASS variance = mean((x − mean)²). The one-pass
    // identity E[x²] − (E[x])² catastrophically cancels in f32 when the
    // row carries a large DC offset (pre-norm transformer activations),
    // collapsing the variance to near-zero and corrupting the norm on
    // wgpu only. Two-pass matches CPU/Metal/MLX/CoreML; the extra read
    // over `inner` is worth correctness.
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
    let var_ = scratch[0] * n_inv;
    let inv_std = inverseSqrt(var_ + eps);

    // Pass 2: normalize, scale, shift in place — each lane over its own lanes.
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        arena[out_base + i] = (arena[out_base + i] - mean) * inv_std * g + b;
        i = i + 64u;
    }
}
