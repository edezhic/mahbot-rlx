// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Split a row into 2–3 contiguous last-axis pieces, each into its own slot, in
// ONE dispatch.
//
// A fused qkv projection writes one `[1, q + k + v]` row; the three consumers
// read it as three tensors, and the decomposed form copies each piece with its
// own `narrow` dispatch. The pieces are contiguous in the source and the whole
// row is covered, so one thread per source element selects its destination by
// the piece boundaries. The values are moved, not recomputed — this is the same
// copy three `narrow` dispatches make.
//
// Admission requires `outer == 1` (a single row: with rows the pieces are
// strided and this flat walk would not describe them), contiguous pieces that
// cover the last axis, and a destination span disjoint from the source's (an
// exact alias is allowed — every thread reads its own element before writing
// it).
struct Params {
    src: u32,
    total: u32,
    end0: u32,
    end1: u32,
    dst0: u32,
    dst1: u32,
    dst2: u32,
    count: u32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@compute @workgroup_size(64)
fn narrow_split(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.total) { return; }
    var dst = params.dst0;
    var base = 0u;
    if (params.count >= 2u && i >= params.end0) {
        dst = params.dst1;
        base = params.end0;
    }
    if (params.count >= 3u && i >= params.end1) {
        dst = params.dst2;
        base = params.end1;
    }
    arena[dst + i - base] = arena[params.src + i];
}
