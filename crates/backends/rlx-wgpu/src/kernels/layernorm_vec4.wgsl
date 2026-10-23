// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised sibling of `layernorm.wgsl`'s `norm` entry point: same params,
// same row-per-workgroup shape, same three passes and the same stable two-pass
// variance. Two things differ, and they are the whole reason it exists:
//
//   * every pass reads and writes whole `vec4<f32>` (16 B) groups, so a decode
//     row of 2 048 elements is 512 vector accesses instead of 2 048 scalar ones
//     with a single access in flight per lane;
//   * the workgroup is 256 lanes, so those accesses are spread over eight warps
//     and the row's memory latency is covered by parallelism rather than by a
//     dependent scalar loop.
//
// The lane mapping stays an EXACT per-element identity (lane `t` reads and
// writes element groups `t, t + 256, …` in both the scalar and the vector
// form). `lower.rs`'s in-place veto rests on that: a norm whose output aliases
// its input is sound only while every lane writes back the elements it read.
//
// Taken at lowering only when `inner % 4 == 0` and every offset the kernel
// reads or writes is a multiple of 4 f32 words (the arena plans at 16 B, so
// this is the common case). Anything else keeps the 64-lane scalar module.
//
// Summation order differs from the scalar kernel (four adjacent elements fold
// first, then the tree), so a digest may move in its last bits; the arithmetic
// is the same two-pass form, not an approximation.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    gamma_off: u32,
    beta_off: u32,
    eps_bits: u32,    // bitcast-encoded f32 eps
    op: u32,          // 0=LayerNorm, 1=RmsNorm
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
fn norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let inner4 = params.inner >> 2u;
    let in_base  = (params.in_off  >> 2u) + row * inner4;
    let out_base = (params.out_off >> 2u) + row * inner4;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);

    // ── Pass 1: mean = Σx / n (tree). ──
    var partial: f32 = 0.0;
    var k: u32 = tid;
    loop {
        if (k >= inner4) { break; }
        let v = arena[in_base + k];
        partial = partial + (v.x + v.y) + (v.z + v.w);
        k = k + WG;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // all threads read scratch[0] before Pass 2 overwrites it.

    // ── Pass 2: mean of squared deviation (tree), stable two-pass form. ──
    partial = 0.0;
    k = tid;
    loop {
        if (k >= inner4) { break; }
        let v = arena[in_base + k];
        let d = v - vec4<f32>(mean);
        partial = partial + (d.x * d.x + d.y * d.y) + (d.z * d.z + d.w * d.w);
        k = k + WG;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let dev_mean = scratch[0] * n_inv; // mean((x − mean)²)

    // ── Pass 3: normalize (each thread strides over its vector groups). ──
    let gamma4 = params.gamma_off >> 2u;
    let beta4 = params.beta_off >> 2u;
    if (params.op == 0u) {
        let inv_std = inverseSqrt(dev_mean + eps);
        var j: u32 = tid;
        loop {
            if (j >= inner4) { break; }
            let g = arena[gamma4 + j];
            let b = arena[beta4 + j];
            arena[out_base + j] = (arena[in_base + j] - vec4<f32>(mean)) * vec4<f32>(inv_std) * g + b;
            j = j + WG;
        }
    } else {
        // mean(x²) = mean((x − mean)²) + mean². `+ beta` matches the CPU oracle
        // (RmsNorm carries beta; the lower now passes the real beta input).
        let inv_rms = inverseSqrt(dev_mean + mean * mean + eps);
        var j: u32 = tid;
        loop {
            if (j >= inner4) { break; }
            let g = arena[gamma4 + j];
            let b = arena[beta4 + j];
            arena[out_base + j] = arena[in_base + j] * vec4<f32>(inv_rms) * g + b;
            j = j + WG;
        }
    }
}
