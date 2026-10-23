// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised narrow: one `vec4<f32>` (16 B) per thread, and no division —
// `gid.z` is the outer index, `gid.y` the slice axis, `gid.x` the contiguous
// span in vec4 units. The scalar form in `narrow.wgsl` rebuilds all three
// from one flat index with `%` and `/` per element.
//
// Taken at lowering only when both window offsets are 16 B aligned (f32-word
// offsets divisible by 4; the window base is 256 B aligned) and the grid fits
// the adapter's `max_compute_workgroups_per_dimension`, in one of two shapes:
// `inner % 4 == 0` (many rows of whole vectors), or `inner == 1` with the slice
// start, its length and the input row stride all whole vectors (the trailing
// transposed-convolution slice). Anything else keeps the 1-D scalar form.
//
// A separate module from `narrow.wgsl` because a vector load needs the
// arena bound as `array<vec4<f32>>` while the scalar fallback indexes
// `array<f32>`. A bind group layout entry carries no element type, so
// `build_kernel`'s layout — and every bind group built from it — fits both;
// only the shader module and entry point differ.
//
// The copy moves lanes unchanged, so the output is bit-identical to the
// scalar kernel's.

struct Params {
    out_total: u32,
    outer: u32,
    inner: u32,
    axis_in_size: u32,
    axis_out_size: u32,
    start: u32,
    in_off: u32,
    out_off: u32,
    vec4: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn narrow_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Last-axis slice (`inner == 1`): the copied run is the slice itself, so the
    // contiguous span is `axis_out_size` elements of one input row rather than
    // `inner` elements of many rows. This is the trailing `narrow` after every
    // transposed convolution; its `inner` is 1, which the row form above cannot
    // express (inner4 == 0), and the old admission test looked only at `inner`
    // and sent it to the scalar kernel.
    if (params.inner == 1u) {
        if (gid.x * 4u >= params.axis_out_size) { return; }
        let out_elem = gid.z * params.axis_out_size + gid.x * 4u;
        if (out_elem >= params.out_total) { return; }
        let in_elem = gid.z * params.axis_in_size + params.start + gid.x * 4u;
        arena[(params.out_off >> 2u) + (out_elem >> 2u)] =
            arena[(params.in_off >> 2u) + (in_elem >> 2u)];
        return;
    }
    let inner4 = params.inner >> 2u;
    // `gid.y`/`gid.z` come from a grid sized exactly to the axis and outer
    // counts, so only x (rounded up to whole workgroups) needs a guard.
    if (gid.x >= inner4) { return; }
    let out_row = gid.z * params.axis_out_size + gid.y;
    // Active extent scales `out_total` at run time; see `concat_vec4.wgsl`
    // for why a whole-vector start guard matches the scalar prefix.
    if (out_row * params.inner + gid.x * 4u >= params.out_total) { return; }
    let in_row = gid.z * params.axis_in_size + (gid.y + params.start);
    arena[(params.out_off >> 2u) + out_row * inner4 + gid.x] =
        arena[(params.in_off >> 2u) + in_row * inner4 + gid.x];
}
