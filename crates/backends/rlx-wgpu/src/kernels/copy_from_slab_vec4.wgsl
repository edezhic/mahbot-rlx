// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vector sibling of `copy_from_slab.wgsl`: one `vec4<f32>` (16 B) per thread.
// Taken at encode time only when the span is whole vectors and both window
// offsets are 16 B aligned — the same rule `copy_vec4.wgsl` applies to the
// arena-only move. A separate module from the scalar form because a vector
// load needs both ends bound as `array<vec4<f32>>`, and a bind group layout
// entry carries no element type, so only the shader module differs.
//
// The four lanes are loaded and stored unchanged, so the result is
// bit-identical to the scalar kernel's.

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>       slab: array<vec4<f32>>;

@compute @workgroup_size(64)
fn copy_from_slab_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let v = gid.x + gid.y * ngs.x * 64u;
    // Guard the vector's start, as the scalar kernel guards its element: the
    // full extent is a multiple of 4, so a partially covered last vector still
    // lies inside both slots.
    if (v * 4u >= params.n) { return; }
    arena[(params.out_off >> 2u) + v] = slab[(params.in_off >> 2u) + v];
}
