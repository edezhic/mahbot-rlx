// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// GPU im2col for the NCHW convolution im2col+GEMM path (groups == 1).
//
// Gathers the receptive field of every output position into a column matrix
// `col[k_total, cols]` laid out **row-major** so it feeds the standard
// `matmul` kernel directly as the B operand, one panel per image:
//
//   A = weight[c_out, c_in*kh*kw]   (row-major, exactly as stored)
//   B = col[n][c_in*kh*kw, cols]    (this kernel's output)
//   C = out[n][c_out, spatial]      (NCHW: the matmul's z is the image)
//
// One dispatch gathers one *chunk* of the panel — output columns `s0 .. s0+cols`
// of the conv's whole `spatial` extent (`ConvColPanel`, which sizes the tail the
// chunk lands in). A chunk is the panel's columns only: the reduction axis is
// never cut, so the GEMM that follows accumulates in the same order as it would
// over the whole extent, and the values are identical. The GEMM writes those
// columns into C at their own offset, C's row stride staying the whole extent.
//
// Image `i` of the batch is one `gid.z` slice, read from its own NCHW frame and
// written to its own panel, so a batched conv gathers every window's receptive
// field in one dispatch and the GEMM that follows reads the weight once per
// image with the batch's z axis.
//
// The flattened reduction index `k = ((ci*kh)+kr)*kw + kc` matches the
// (ci, kr, kc) nesting of the direct conv kernel, so the direct and gathered
// routes accumulate in the same order — which made their f32 outputs identical
// while the GEMM below stayed on the plain f32 kernel. That GEMM may now take
// the f32 cooperative kernel for a shape whose k is a multiple of 8 and whose
// spatial extent is a multiple of 8 (`conv_gemm` in `backend/mod.rs`), whose
// accumulation order the tree does not fix — so on those shapes the two routes
// are held to a tolerance, not to bit-identity.
//
// One thread per col element (`k_total * cols` threads per image).
// Out-of-bounds receptive-field taps (padding / dilation past the edge) store 0.0.

struct Params {
    c_in: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    in_off: u32,
    col_off: u32,
    k_total: u32,   // c_in * kh * kw
    cols: u32,      // output columns of this chunk (the panel's row stride)
    s0: u32,        // this chunk's first output column
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn im2col2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.k_total * params.cols;
    let idx = gid.x + gid.y * ngs.x * 64u;
    if (idx >= total) { return; }

    let img = gid.z;
    let c = idx % params.cols;      // column within this chunk
    let k = idx / params.cols;      // flattened (ci, kr, kc)

    let kc = k % params.kw;
    let tmp = k / params.kw;
    let kr = tmp % params.kh;
    let ci = tmp / params.kh;

    let s = params.s0 + c;          // output spatial position (ho*w_out + wo)
    let ho = s / params.w_out;
    let wo = s % params.w_out;

    let in_r = i32(ho * params.sh + kr * params.dh) - i32(params.ph);
    let in_c = i32(wo * params.sw + kc * params.dw) - i32(params.pw);

    var v: f32 = 0.0;
    if (in_r >= 0 && in_c >= 0 && in_r < i32(params.h) && in_c < i32(params.w)) {
        // Element (img, ci, in_r, in_c) of [n, c_in, h, w].
        let frame = params.c_in * params.h * params.w;
        let in_idx = img * frame + (ci * params.h + u32(in_r)) * params.w + u32(in_c);
        v = arena[params.in_off + in_idx];
    }
    let panel = params.k_total * params.cols;
    arena[params.col_off + img * panel + idx] = v;
}
