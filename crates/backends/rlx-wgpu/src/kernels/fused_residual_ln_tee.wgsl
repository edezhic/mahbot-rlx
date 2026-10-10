// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// FusedResidualLN / FusedResidualRmsNorm with a "tee" output (`is_rms`): writes BOTH (h + residual + [bias])
// AND LN(h + residual + [bias]) to two separate arena slots. Used when
// the sum has multiple consumers downstream (the LN AND a later residual
// add) and the regular `fused_residual_ln` can't fire because of its
// single-consumer guard.
//
// Replaces the (Add → LN) sequence with one dispatch when Add is multi-
// consumer. Vision pre-norm transformers hit this 23× per forward
// (every residual feeds both the next LN and the *next-next* residual);
// w/o this kernel the wgpu schedule has 35 standalone Binary + 36
// standalone LayerNorm. With this kernel each (Add → LN) collapses to
// one FusedResidualLnTee.
//
// Memory layout:
//   sum_off:      [outer, inner]   ← (h + residual + [bias])
//   ln_out_off:   [outer, inner]   ← LN of the sum
// Other consumers of the sum read sum_off (= the eliminated Add's old
// arena slot, which the planner has already allocated for them).
//
// ONE WORKGROUP (64 threads) PER ROW, shared-memory tree reduction over the
// feature dim — the same shape and the same lane mapping as
// `fused_residual_ln.wgsl`. That mapping is what the arena-safety veto in
// `lower.rs` rests on: the norm's output may alias `h` or `delta` EXACTLY
// (row r's writes land on the lanes those same threads have already read),
// while a shift that maps one lane's write onto another lane's read is fatal.

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
fn fused_residual_ln_tee(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let sum_base = params.sum_off      + row * params.inner;
    let out_base = params.ln_out_off   + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1: write the SUM to sum_base and accumulate this lane's share of it.
    var partial: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.inner) { break; }
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[sum_base + i] = v;
        partial = partial + v;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // all lanes read scratch[0] before Pass 1b overwrites it.

    // Pass 1b: STABLE TWO-PASS variance = mean((x − mean)²), reading the
    // summed value back from sum_base. The one-pass E[x²] − (E[x])²
    // identity catastrophically cancels in f32 under a large DC offset
    // (pre-norm transformer activations) and corrupts the norm on wgpu
    // only; two-pass matches CPU/Metal/MLX/CoreML.
    partial = 0.0;
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let d = arena[sum_base + i] - mean;
        partial = partial + d * d;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let var_ = scratch[0] * n_inv;
    // RMS norm scales by 1/sqrt(mean(x^2)) and does NOT recentre. Reuse the
    // same stable two-pass accumulation: mean(x^2) = mean((x-mean)^2) + mean^2,
    // which avoids the f32 cancellation a one-pass sum(x^2) suffers under the
    // large DC offset pre-norm transformer activations carry.
    let rms_only = params.is_rms != 0u;
    let inv_scale = select(
        inverseSqrt(var_ + eps),
        inverseSqrt(var_ + mean * mean + eps),
        rms_only,
    );

    // Pass 2: read sum_base, write the norm result to out_base — each lane over
    // its own lanes. Two distinct slots — the sum stays available for the other
    // consumer.
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        let v = arena[sum_base + i];
        let centred = select(v - mean, v, rms_only);
        arena[out_base + i] = centred * inv_scale * g + b;
        i = i + 64u;
    }
}
