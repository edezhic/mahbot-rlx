// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised sibling of `fused_residual_ln_tee.wgsl` — the same dual-output
// (sum + norm) step, over `vec4<f32>` groups with a 256-lane workgroup. Read
// that module for what the step is and why the sum is written out; this one
// differs only in the access width and the lane count, which is the whole of
// its cost model: a 2 048-wide decode row is 512 vector accesses over eight
// warps instead of 2 048 scalar accesses in one dependent chain.
//
// The per-element lane mapping is identical to the scalar module (lane `t`
// owns element groups `t, t + 256, …`), so the in-place alias rule that
// `lower.rs`'s veto rests on is unchanged.
//
// Taken only when `inner % 4 == 0` and every offset is 4-word aligned.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    sum_off: u32,
    ln_out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    is_rms: u32,
    vec4: u32,
};

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> arena: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> scratch: array<f32, 256>;

// Sum-reduce `scratch[0..256]` into `scratch[0]`. Caller must barrier before.
fn tree_sum(tid: u32) {
    var stride: u32 = WG >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
}

@compute @workgroup_size(256)
fn fused_residual_ln_tee(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let inner4 = params.inner >> 2u;
    let in_base  = (params.in_off       >> 2u) + row * inner4;
    let res_base = (params.residual_off >> 2u) + row * inner4;
    let sum_base = (params.sum_off      >> 2u) + row * inner4;
    let out_base = (params.ln_out_off   >> 2u) + row * inner4;
    let bias4 = params.bias_off >> 2u;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1: write the SUM to sum_base and accumulate this lane's share of it.
    var partial: f32 = 0.0;
    var k: u32 = tid;
    loop {
        if (k >= inner4) { break; }
        var v = arena[in_base + k] + arena[res_base + k];
        if (with_bias) { v = v + arena[bias4 + k]; }
        arena[sum_base + k] = v;
        partial = partial + (v.x + v.y) + (v.z + v.w);
        k = k + WG;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // all lanes read scratch[0] before Pass 1b overwrites it.

    // Pass 1b: stable two-pass variance, reading the summed value back.
    partial = 0.0;
    k = tid;
    loop {
        if (k >= inner4) { break; }
        let d = arena[sum_base + k] - vec4<f32>(mean);
        partial = partial + (d.x * d.x + d.y * d.y) + (d.z * d.z + d.w * d.w);
        k = k + WG;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let var_ = scratch[0] * n_inv;
    let rms_only = params.is_rms != 0u;
    let inv_scale = select(
        inverseSqrt(var_ + eps),
        inverseSqrt(var_ + mean * mean + eps),
        rms_only,
    );

    // Pass 2: read sum_base, write the norm result to out_base.
    let gamma4 = params.gamma_off >> 2u;
    let beta4 = params.beta_off >> 2u;
    var j: u32 = tid;
    loop {
        if (j >= inner4) { break; }
        let g = arena[gamma4 + j];
        let b = arena[beta4 + j];
        let v = arena[sum_base + j];
        let centred = select(v - vec4<f32>(mean), v, rms_only);
        arena[out_base + j] = centred * vec4<f32>(inv_scale) * g + b;
        j = j + WG;
    }
}
