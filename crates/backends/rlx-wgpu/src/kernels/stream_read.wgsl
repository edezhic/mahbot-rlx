// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Read-only stream probe: the ceiling on a *read* stream, which is the
// quantity a packed weight sees — a weight is read, never written. The
// sibling copy (`copy_vec4.wgsl`, measured by `bandwidth::stream_ceiling`)
// moves two bytes of traffic for every byte it reads, so its rate is a
// different mix and neither figure bounds the other.
//
// Each thread loads `params.words_per_thread` (`WORDS = 4u` in the probe)
// `vec4<f32>` (64 B) at `gid.x`, `gid.x + stride`, `gid.x + 2 · stride`,
// `gid.x + 3 · stride`, where stride is the whole dispatch
// (`workgroup_size · num_workgroups.x`). The four sweeps therefore cover
// `[0, 4 · threads)` exactly once, and every `vec4` of the binding below
// `params.n` is read exactly once. The four addresses depend only on `gid.x`
// and the stride, so the loads are independent and a thread keeps four of them
// in flight; the running sum consumes them in order and is what makes them
// live.
//
// Geometry on the 64 MiB half the copy measures: the half is `n = 67 107 840
// / 16 = 4 194 240` `vec4`. 4096 workgroups × 256 threads = 1 048 576
// threads × 4 `vec4` = 4 194 304 `vec4`, 64 `vec4` (1 KiB) more than `n`, so
// the last 64 threads' fourth load lands past the extent and is guarded. That
// is deliberate: the guarded tail is the only out-of-range load, and the
// bytes read per dispatch stay exactly `n · 16 = 67 107 840` — the same
// per-direction figure the copy moves, so the two rates are comparable
// without a scale factor.
//
// The sink is one f32 per workgroup: 4096 × 4 B = 16 KiB stored per dispatch
// against 64 MiB read, ~0.02 % of the read side. Without it the loads would be
// dead and a driver is free to drop them. The tree in `partial_sum` exists to
// fold all 256 threads' words into that one store, not to be timed: the figure
// this kernel produces is a read rate, not a reduction rate.
struct Params {
    n: u32,
    // Loads per thread; the probe writes `WORDS = 4u`.
    words_per_thread: u32,
    _p1: u32,
    _p2: u32,
};

const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read>       src: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read_write> sink: array<f32>;

var<workgroup> partial_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn stream_read(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = WORKGROUP_SIZE * groups.x;
    var acc: f32 = 0.0;
    for (var w = 0u; w < params.words_per_thread; w = w + 1u) {
        let i = gid.x + w * stride;
        if (i < params.n) {
            let v = src[i];
            acc = acc + (v.x + v.y + v.z + v.w);
        }
    }
    partial_sum[lid.x] = acc;
    workgroupBarrier();
    // Fixed tree: 256 → 128 → … → 1, one barrier per level so no thread reads
    // a slot another is still overwriting.
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        sink[wid.x] = partial_sum[0];
    }
}
