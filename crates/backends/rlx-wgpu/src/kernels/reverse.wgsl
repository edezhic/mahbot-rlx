// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Batch-general flip (`Op::Reverse`): element `[..., i, ...]` on a reversed
// axis of size `d` reads input `[..., d-1-i, ...]`. The host pre-computes the
// per-axis dims, the source strides and the reverse flags; same 3-binding
// pattern as `expand.wgsl` (per-axis arrays in a STORAGE buffer, because a
// scalar array in a WGSL *uniform* is 16-byte strided).

struct Params {
    rank: u32,
    total: u32,
    in_off: u32,
    out_off: u32,
    // PLAN L1 — `1` when the bucket (leading) axis is NOT reversed, so the
    // active-extent path may scale `total`. `0` keeps the launch at full
    // extent: scaling an axis this kernel flips would flip a rescaled axis.
    scale_leading: u32,
    dim_0: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        axis_meta: array<u32>;

@compute @workgroup_size(64)
fn reverse(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.total) { return; }
    var rem = i;
    var src: u32 = 0u;
    for (var ax: u32 = params.rank; ax > 0u; ax = ax - 1u) {
        let a = ax - 1u;
        let dim = axis_meta[a];                          // dims[a]
        let stride = axis_meta[params.rank + a];         // src strides[a]
        var c = rem % dim;
        rem = rem / dim;
        if (axis_meta[2u * params.rank + a] == 1u) {
            c = dim - 1u - c;
        }
        src = src + c * stride;
    }
    arena[params.out_off + i] = arena[params.in_off + src];
}
