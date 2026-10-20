// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The GGUF packed-multiply output move: the pooled output slab → an arena
// window. Two storage bindings because the two ends are *different* buffers —
// the slab is a process-global scratch buffer, the destination is the arena —
// so `copy.wgsl`'s single-buffer form cannot carry it and a
// `copy_buffer_to_buffer` blit cannot be recorded inside a compute pass. This
// dispatch can: it is the move the joined packed multiply records after its
// kernel in the caller's own pass.
//
// Both windows start at a 256 B-aligned base (wgpu's storage binding offset
// requirement), so the intra-window offsets ride in the uniform exactly as the
// dequant kernels' do. The move is a byte-for-byte lane copy, so it is
// bit-identical to the blit it replaces.
//
// `copy_from_slab_vec4.wgsl` is the 16 B-per-thread sibling; `vec4` in the
// encoder is what picks one or the other.

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>       slab: array<f32>;

@compute @workgroup_size(64)
fn copy_from_slab(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    arena[params.out_off + i] = slab[params.in_off + i];
}
