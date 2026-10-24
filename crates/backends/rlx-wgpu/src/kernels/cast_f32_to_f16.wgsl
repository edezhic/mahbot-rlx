// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable f16;

// Pre-pass cast that materialises a region of the f32 arena as f16 in whichever
// f16 buffer the step binds. Two users: the weight mirror, before a f16-operand
// matmul (the mirror is indexed by the same logical element as the arena, so
// `dst_off == src_off` there), and the dense attention route's own operand buffer,
// where `dst_off` is that buffer's region start. One thread per element.

struct Params {
    src_off: u32,   // f32-element offset into the arena window
    len: u32,       // element count
    dst_off: u32,   // f16-element offset into the bound f16 buffer
    scale_bits: u32, // scalar applied on the way through, 0 = none
};

// Bind ordering matches `build_kernel_3`: 0=f16(rw), 1=uniform, 2=f32 arena(rw).
@group(0) @binding(0) var<storage, read_write> arena_f16: array<f16>;
@group(0) @binding(1) var<uniform>             params:    Params;
@group(0) @binding(2) var<storage, read_write> arena:     array<f32>;

@compute @workgroup_size(64)
fn cast_f32_to_f16(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.len) { return; }
    // `scale_bits == 0` is the unscaled cast every other caller takes; a set
    // scale multiplies before the downcast (see the dense attention route).
    let s = select(1.0, bitcast<f32>(params.scale_bits), params.scale_bits != 0u);
    arena_f16[params.dst_off + i] = f16(arena[params.src_off + i] * s);
}
