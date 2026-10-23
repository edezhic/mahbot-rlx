// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// One `Op::Concat`'s inputs copied in ONE dispatch. `concat.wgsl` is dispatched
// once per input (each carries its own `start`); this walks the output as one run
// of contiguous input slices, so a 15-input concat costs one dispatch instead of
// fifteen. Each workgroup copies 64 consecutive elements of one input's run; the
// workgroup-to-input walk is uniform, and the copy is the same f32 move.
//
// Admission (lower.rs) requires a single row (`outer == 1`), at most
// `CONCAT_MANY_MAX` pieces, a real dtype, and every input plus the output inside
// one bind window — so no piece is ever staged, which would cost more than the
// dispatches it replaced. A walked extent does not cut the pieces: the axis a walk
// trims is the leading one and is not necessarily this concat's (a framing over a
// row source has it inside `inner`), so the step copies the planned run and the
// rows past the walk are rows nothing reads at that extent.
struct Params {
    inner: u32,    // product of the trailing dims (the row width below the axis)
    out_off: u32,
    count: u32,    // inputs, in output order
    ins: array<vec4<u32>, 16>, // .x = in_off, .y = axis_len, .z = axis start
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn concat_many(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // The grid is one workgroup a 64-element run: `gid` is a thread index across
    // the whole dispatch, so the workgroup is `gid.x / 64`.
    let wg = gid.x / 64u + gid.y * ngs.x;
    // Which input this workgroup's 64 elements belong to, and which workgroup
    // that input's run starts at.
    var k: u32 = 0u;
    var first: u32 = 0u;
    loop {
        if (k >= params.count) { return; }
        let e = params.ins[k];
        let wgs = (e.y * params.inner + 63u) / 64u;
        if (wg < first + wgs) { break; }
        first = first + wgs;
        k = k + 1u;
    }
    let e = params.ins[k];
    let i = (wg - first) * 64u + lid.x;
    if (i >= e.y * params.inner) { return; }
    // A piece's run is contiguous in the output from its own axis start, so the
    // output index needs no decomposition.
    arena[params.out_off + e.z * params.inner + i] = arena[e.x + i];
}
