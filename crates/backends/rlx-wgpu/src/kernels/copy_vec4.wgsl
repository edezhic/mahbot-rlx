// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised flat copy: one `vec4<f32>` (16 B) per thread instead of one
// f32, and no integer division — `gid.x` is already the vector index.
//
// Taken at encode time only when `n % 4 == 0` and both window offsets are
// 16 B aligned (f32-word offsets divisible by 4; the window base is 256 B
// aligned). Anything else keeps `copy.wgsl`'s scalar form, including a
// dispatch past the adapter's `max_compute_workgroups_per_dimension`.
//
// A separate module from `copy.wgsl` because a vector load needs the arena
// bound as `array<vec4<f32>>` while the scalar fallback indexes
// `array<f32>`. A bind group layout entry carries no element type, so
// `build_kernel`'s layout — and every bind group built from it — fits both;
// only the shader module and entry point differ.
//
// The copy is a byte move: the four lanes are loaded and stored unchanged,
// so the output is bit-identical to the scalar kernel's.

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn copy_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Active extent scales `n` at run time. Guard the vector's *start*, as
    // the scalar kernel guards its element: a vector that starts inside the
    // scaled prefix moves all four of its lanes, so the last one may carry up
    // to three elements past that prefix. They stay inside this op's own
    // destination slot — the slot's full extent is a whole number of vectors —
    // and outside the active region, so no other slot is touched; the scalar
    // kernel writes those elements, this one writes them as part of the vector.
    if (gid.x * 4u >= params.n) { return; }
    arena[(params.out_off >> 2u) + gid.x] = arena[(params.in_off >> 2u) + gid.x];
}
