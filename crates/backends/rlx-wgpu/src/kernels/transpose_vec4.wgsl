// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised permute: one `vec4<f32>` (16 B) per thread, with the axis
// decomposition done once per thread instead of once per element.
//
// Three modes, chosen at lowering:
//
//   mode 1 — a general permutation that keeps the innermost axis in place
//            (`perm[rank-1] == rank-1`). That axis is contiguous on both
//            sides, so a thread copies one vector between two rows whose
//            outer index is decomposed over `axis_meta` once.
//   mode 2 — `perm == [0, 2, 1, 3]`, plus any trailing self-mapped axes
//            (the C64/C128 lane axis enters that way): the attention
//            head-major swap. 3-D grid: x over `inner / 4`, y over the
//            sequence axis, z over heads, with a division-free loop over
//            batch inside.
//   mode 3 — `perm == [0, 2, 1]`: the rank-3 head-major swap. Its input
//            innermost axis is contiguous but lands mid-output, so a
//            thread reads one vector and writes the four components at
//            their strided output positions. Those writes go through a
//            scalar alias of the same window (binding 3): a component
//            store through the `vec4` array is a read-modify-write of the
//            whole 16 B group on some backends, and threads holding
//            different lanes of the same group would clobber each other.
//            Scalar stores are 4 B each and do not overlap.
//
// A separate module from `transpose.wgsl` because a vector load needs the
// arena bound as `array<vec4<f32>>` while the scalar fallback indexes
// `array<f32>`. A bind group layout entry carries no element type, so the
// scalar kernel's layout (bindings 0–2) fits both; only the shader module
// and entry point differ. Binding 3 is the scalar alias only this module
// declares.
//
// Taken at lowering only when the innermost span is a multiple of 4, both
// window offsets are 16 B aligned (f32-word offsets divisible by 4; the
// window base is 256 B aligned), and every grid dimension fits the
// adapter's `max_compute_workgroups_per_dimension`. Everything else,
// including permutations that leave no axis contiguous (`perm == [1, 0]`),
// keeps the 1-D scalar kernel.
//
// Active extent scales `out_total` at run time. Modes 1 and 2 guard each
// vector's start; the span is a multiple of 4, so the vector stays inside
// one inner row and the cut matches the scalar element prefix within that
// row's last whole vector. Mode 3 guards every component because its four
// writes are not adjacent.
//
// The copy moves lanes unchanged (mode 3 moves them to strided slots), so
// every mode is bit-identical to the scalar kernel.

struct Params {
    rank: u32,
    out_total: u32,
    in_off: u32,
    out_off: u32,
    bucket_outermost: u32,
    out_dim_0: u32,
    mode: u32,
    grid_y: u32,
    grid_z: u32,
    inner: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        axis_meta: array<u32>;
// Scalar alias of the same window, for mode 3's non-adjacent stores.
@group(0) @binding(3) var<storage, read_write> arena_out: array<f32>;

@compute @workgroup_size(64)
fn transpose_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inner4 = params.inner >> 2u;
    if (gid.x >= inner4) { return; }
    let in_base = params.in_off >> 2u;
    let out_base = params.out_off >> 2u;

    if (params.mode == 2u) {
        // Output extents are [batch, seq, heads, inner..]; the input swaps
        // its seq and head axes. z is the head index and y the sequence index,
        // so the only loop is over batch and the row math is division-free.
        let batch = axis_meta[0];
        let seq = axis_meta[1];
        let heads = axis_meta[2];
        let s = gid.y;
        let h = gid.z;
        for (var b: u32 = 0u; b < batch; b = b + 1u) {
            let out_row = (b * seq + s) * heads + h;
            // Active extent trims the leading batch axis, so rows past the
            // scaled total only get further past it as `b` grows.
            if (out_row * params.inner + gid.x * 4u >= params.out_total) { return; }
            let in_row = (b * heads + h) * seq + s;
            arena[out_base + out_row * inner4 + gid.x] =
                arena[in_base + in_row * inner4 + gid.x];
        }
        return;
    }

    if (params.mode == 3u) {
        // Input is [outer, seq, hd] and the output is [outer, hd, seq]: read
        // along the input's contiguous hd axis, write each lane to its
        // strided output position.
        let hd = axis_meta[1];
        let seq = axis_meta[2];
        let v = arena[in_base + (gid.z * seq + gid.y) * inner4 + gid.x];
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let e = gid.x * 4u + c;
            let out_elem = (gid.z * hd + e) * seq + gid.y;
            if (out_elem < params.out_total) {
                arena_out[params.out_off + out_elem] = v[c];
            }
        }
        return;
    }

    // Mode 1: one vector per (outer index, inner vector). The outer index is
    // split across y and z to stay inside the per-dimension workgroup cap,
    // then peeled over the leading output axes exactly as the scalar kernel
    // peels the whole flat index.
    let o = gid.y + gid.z * params.grid_y;
    var rem = o;
    var src_row: u32 = 0u;
    for (var a: u32 = params.rank - 1u; a > 0u; a = a - 1u) {
        let ax = a - 1u;
        let dim = axis_meta[ax];
        let c = rem % dim;
        rem = rem / dim;
        src_row = src_row + c * axis_meta[params.rank + ax];
    }
    if (o * params.inner + gid.x * 4u >= params.out_total) { return; }
    arena[out_base + o * inner4 + gid.x] =
        arena[in_base + (src_row >> 2u) + gid.x];
}
