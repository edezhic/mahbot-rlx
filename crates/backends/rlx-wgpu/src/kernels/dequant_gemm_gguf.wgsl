// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused multi-row GGUF K-quant GEMM (prefill `m > 1`) for `Op::DequantMatMul`.
//
// One entry point per scheme — `dequant_gemm_q4k` (scheme 0) and
// `dequant_gemm_q6k` (scheme 2). Deliberately NOT a `scheme_id` branch chain
// like `dequant_gguf.wgsl`: an id routed to a kernel with no branch for it
// matches nothing and reads the weights as **zeros** instead of failing.
//
// Each thread owns one output column `j` and accumulates `TM = 8` rows of
// `Y[m,n] = X[m,k] @ dequant(W)[n,k]^T` straight from the packed weight (no f32
// scratch). Every packed weight byte is read **once per row group** and reused
// by all 8 rows — that is the whole point of this kernel, and why the per-row
// GEMV (which re-streams the weight once per row) does not scale in `m`.
//
// The X side is read as `vec4` straight from device memory rather than staged in
// threadgroup memory, which keeps the workgroup free of barriers. All 64 threads
// of a workgroup read the same X addresses (they differ only in output column),
// and each row's activations are read once per column group — the same traffic
// the per-row GEMV pays — so the weight read is what the group size buys back.
// Measured on a 777 MiB Q4_K weight (k=7168, n=202048 — the widest served
// shape): a tile of 8 rows costs ~1.2x one single-row pass, flat in `m` out to
// 276 rows.
//
// Binding scheme matches `dequant_gemv_gguf.wgsl`: `x` and `weight` are separate
// read-only windows of the arena (each under `max_storage_buffer_binding_size`)
// and `y` goes to a separate buffer the caller copies back — a storage binding
// bound read-write is exclusive in a dispatch and cannot also be read.
//
// Grid: x = ceil(ncols / 64) column groups of 64 columns, y = ceil(m / 8) row
// tiles. Rows past `m` clamp to the last valid row (their loads stay in bounds)
// and are dropped at the store.
//
// The dequant is the GEMV's, element for element and in the same k order; the
// accumulation is not — each thread keeps four vec4<f32> lanes per row and adds
// them horizontally at the store (`dot4`), where the one-row arms sum in their
// own orders (the tiled arm reduces lane partials in a fixed tree), so round-off
// differs. The measured CPU-reference parity is the real justification: max-abs
// 1.9e-5 .. 6.5e-4 over the prefill-parity shapes, against the 1e-2 this op is
// compared at.
//
// The dequant helpers (`read_w`, `read_w_i8`, `dq_read_f16`,
// `dq_get_scale_min_k4`) are copied from the one-row K-quant arm
// (`dequant_gemv_tiled`) of `dequant_gemv_gguf.wgsl`, rewritten to build one
// `vec4` per 4 weights. They are the third copy of the Q4_K/Q6_K packing in this
// directory (`dequant_gguf.wgsl`, `dequant_gemv_gguf.wgsl`): a fix to the
// nibble or scale/min layout has to land in all three.

struct Params {
    m: u32,           // rows
    k: u32,           // contraction dim (multiple of 256, hence of 4)
    ncols: u32,       // output columns in this dispatch (also the out row stride)
    x_vec4_off: u32,  // vec4 index of X[0,0] within the x binding
    w_byte_off: u32,  // byte offset of W[col0,0] within the weight binding
    // 8 words total: a uniform binding size must be a multiple of 16 bytes.
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>        xarr: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        warr: array<u32>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;

const TM: u32 = 8u;
const TG: u32 = 64u;
const BLK_ELEMS: u32 = 256u;

fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (warr[word] >> shift) & 0xffu;
}

fn read_w_i8(rel: u32) -> i32 {
    let b = read_w(rel);
    return select(i32(b), i32(b) - 256, b >= 128u);
}

fn dq_read_f16(rel: u32) -> f32 {
    let bits = read_w(rel) | (read_w(rel + 1u) << 8u);
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var v: f32;
    if (exp == 0u) { v = f32(mant) / 1024.0 * exp2(-14.0); }
    else if (exp == 31u) { v = select(0.0, bitcast<f32>(0x7f800000u), mant == 0u); }
    else { v = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp) - 15)); }
    return select(v, -v, sign != 0u);
}

fn dq_get_scale_min_k4(scales_rel: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(read_w(scales_rel + j) & 63u, read_w(scales_rel + j + 4u) & 63u);
    }
    let sc = (read_w(scales_rel + j + 4u) & 0x0Fu) | (((read_w(scales_rel + j - 4u) >> 6u) & 3u) << 4u);
    let mn = (read_w(scales_rel + j + 4u) >> 4u) | (((read_w(scales_rel + j) >> 6u) & 3u) << 4u);
    return vec2<u32>(sc, mn);
}

fn dot4(v: vec4<f32>) -> f32 { return v.x + v.y + v.z + v.w; }

// Q4_K: 144 bytes / 256 elems (f16 d + f16 dmin + 12 packed scales + 128 nibbles).
@compute @workgroup_size(64, 1, 1)
fn dequant_gemm_q4k(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x * TG + tid;
    let row0 = wid.y * TM;
    // Clamp out-of-range columns rather than returning: the row tile is shared,
    // and out-of-range results are dropped at the store.
    let colc = min(col, params.ncols - 1u);

    let nbk = params.k / BLK_ELEMS;
    let blk = 2u + 2u + 12u + 128u;
    let k4 = params.k >> 2u;
    // vec4 index of each tile row's start in the x window. Rows past `m` clamp
    // to the last valid row; their accumulators are dropped at the store.
    let last_row = params.m - 1u;
    let xb = params.x_vec4_off;
    let b0 = xb + min(row0, last_row) * k4;
    let b1 = xb + min(row0 + 1u, last_row) * k4;
    let b2 = xb + min(row0 + 2u, last_row) * k4;
    let b3 = xb + min(row0 + 3u, last_row) * k4;
    let b4 = xb + min(row0 + 4u, last_row) * k4;
    let b5 = xb + min(row0 + 5u, last_row) * k4;
    let b6 = xb + min(row0 + 6u, last_row) * k4;
    let b7 = xb + min(row0 + 7u, last_row) * k4;

    // TM named accumulators: an `array<vec4<f32>, TM>` indexed by a loop
    // variable spills to local memory and costs a load+store per accumulate.
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    for (var b: u32 = 0u; b < nbk; b = b + 1u) {
        let p0 = b * BLK_ELEMS;
        let off = (colc * nbk + b) * blk;
        let d = dq_read_f16(off);
        let dmin = dq_read_f16(off + 2u);
        let scales_rel = off + 4u;
        let qs_rel = off + 16u;
        var lp = 0u;   // k offset within the block for the current 32-elem half
        var is = 0u;
        for (var t: u32 = 0u; t < 8u; t = t + 2u) {
            let sm0 = dq_get_scale_min_k4(scales_rel, t);
            let sm1 = dq_get_scale_min_k4(scales_rel, t + 1u);
            let d0 = d * f32(sm0.x);
            let m0 = dmin * f32(sm0.y);
            let d1 = d * f32(sm1.x);
            let m1 = dmin * f32(sm1.y);
            // Two separate nibble passes, low half fully before high half, so a
            // sub-block's elements land in the same k positions here and in the
            // one-row arms' dequant.
            for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                var ww = vec4<f32>(0.0);
                ww.x = d0 * f32(read_w(qs_rel + is + l) & 0x0Fu) - m0;
                ww.y = d0 * f32(read_w(qs_rel + is + l + 1u) & 0x0Fu) - m0;
                ww.z = d0 * f32(read_w(qs_rel + is + l + 2u) & 0x0Fu) - m0;
                ww.w = d0 * f32(read_w(qs_rel + is + l + 3u) & 0x0Fu) - m0;
                let e = (p0 + lp + l) >> 2u;
                a0 = a0 + xarr[b0 + e] * ww;
                a1 = a1 + xarr[b1 + e] * ww;
                a2 = a2 + xarr[b2 + e] * ww;
                a3 = a3 + xarr[b3 + e] * ww;
                a4 = a4 + xarr[b4 + e] * ww;
                a5 = a5 + xarr[b5 + e] * ww;
                a6 = a6 + xarr[b6 + e] * ww;
                a7 = a7 + xarr[b7 + e] * ww;
            }
            lp = lp + 32u;
            for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                var ww = vec4<f32>(0.0);
                ww.x = d1 * f32(read_w(qs_rel + is + l) >> 4u) - m1;
                ww.y = d1 * f32(read_w(qs_rel + is + l + 1u) >> 4u) - m1;
                ww.z = d1 * f32(read_w(qs_rel + is + l + 2u) >> 4u) - m1;
                ww.w = d1 * f32(read_w(qs_rel + is + l + 3u) >> 4u) - m1;
                let e = (p0 + lp + l) >> 2u;
                a0 = a0 + xarr[b0 + e] * ww;
                a1 = a1 + xarr[b1 + e] * ww;
                a2 = a2 + xarr[b2 + e] * ww;
                a3 = a3 + xarr[b3 + e] * ww;
                a4 = a4 + xarr[b4 + e] * ww;
                a5 = a5 + xarr[b5 + e] * ww;
                a6 = a6 + xarr[b6 + e] * ww;
                a7 = a7 + xarr[b7 + e] * ww;
            }
            lp = lp + 32u;
            is = is + 32u;
        }
    }

    store_tile(a0, a1, a2, a3, a4, a5, a6, a7, row0, col, colc);
}

// Write the TM row results of one column. Split out so both entries share it.
fn store_tile(
    a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>,
    a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>,
    row0: u32, col: u32, colc: u32,
) {
    if (col != colc) { return; }   // clamped column: no store
    if (row0 < params.m) { outarr[row0 * params.ncols + col] = dot4(a0); }
    if (row0 + 1u < params.m) { outarr[(row0 + 1u) * params.ncols + col] = dot4(a1); }
    if (row0 + 2u < params.m) { outarr[(row0 + 2u) * params.ncols + col] = dot4(a2); }
    if (row0 + 3u < params.m) { outarr[(row0 + 3u) * params.ncols + col] = dot4(a3); }
    if (row0 + 4u < params.m) { outarr[(row0 + 4u) * params.ncols + col] = dot4(a4); }
    if (row0 + 5u < params.m) { outarr[(row0 + 5u) * params.ncols + col] = dot4(a5); }
    if (row0 + 6u < params.m) { outarr[(row0 + 6u) * params.ncols + col] = dot4(a6); }
    if (row0 + 7u < params.m) { outarr[(row0 + 7u) * params.ncols + col] = dot4(a7); }
}

// Q6_K: 210 bytes / 256 elems (128 ql + 64 qh + 16 i8 scales + f16 d).
// One vec4 of dequantized weights per quarter (element offsets +0/+32/+64/+96
// of the half), so each component's 6 bits are assembled from `b` (ql) and `h`
// (qh). `hi` picks the high nibble of `b`; `shift` the qh bit pair.
fn q6c(sc: f32, byte: u32, h: u32, shift: u32, hi: bool) -> f32 {
    let q = select(byte & 0x0Fu, byte >> 4u, hi);
    return sc * f32(i32((q | (((h >> shift) & 3u) << 4u)) - 32));
}

fn q6_quarter(sc: f32, b: vec4<u32>, h: vec4<u32>, shift: u32, hi: bool) -> vec4<f32> {
    return vec4<f32>(
        q6c(sc, b.x, h.x, shift, hi),
        q6c(sc, b.y, h.y, shift, hi),
        q6c(sc, b.z, h.z, shift, hi),
        q6c(sc, b.w, h.w, shift, hi),
    );
}

@compute @workgroup_size(64, 1, 1)
fn dequant_gemm_q6k(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x * TG + tid;
    let row0 = wid.y * TM;
    let colc = min(col, params.ncols - 1u);

    let nbk = params.k / BLK_ELEMS;
    let blk = 128u + 64u + 16u + 2u;
    let k4 = params.k >> 2u;
    let last_row = params.m - 1u;
    let xb = params.x_vec4_off;
    let b0 = xb + min(row0, last_row) * k4;
    let b1 = xb + min(row0 + 1u, last_row) * k4;
    let b2 = xb + min(row0 + 2u, last_row) * k4;
    let b3 = xb + min(row0 + 3u, last_row) * k4;
    let b4 = xb + min(row0 + 4u, last_row) * k4;
    let b5 = xb + min(row0 + 5u, last_row) * k4;
    let b6 = xb + min(row0 + 6u, last_row) * k4;
    let b7 = xb + min(row0 + 7u, last_row) * k4;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    for (var b: u32 = 0u; b < nbk; b = b + 1u) {
        let p0 = b * BLK_ELEMS;
        let off = (colc * nbk + b) * blk;
        let ql_rel = off;
        let qh_rel = off + 128u;
        let sc_rel = off + 192u;
        let d = dq_read_f16(off + 208u);
        for (var h: u32 = 0u; h < 2u; h = h + 1u) {
            let row_base = h * 128u;
            let ql_off = h * 64u;
            let qh_off_h = h * 32u;
            let sc_off = h * 8u;
            for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                let is = l / 16u;
                // One i8 scale per quarter of the half.
                let sc1 = d * f32(read_w_i8(sc_rel + sc_off + is));
                let sc2 = d * f32(read_w_i8(sc_rel + sc_off + is + 2u));
                let sc3 = d * f32(read_w_i8(sc_rel + sc_off + is + 4u));
                let sc4 = d * f32(read_w_i8(sc_rel + sc_off + is + 6u));
                let qa = vec4<u32>(
                    read_w(ql_rel + ql_off + l),
                    read_w(ql_rel + ql_off + l + 1u),
                    read_w(ql_rel + ql_off + l + 2u),
                    read_w(ql_rel + ql_off + l + 3u),
                );
                let qb = vec4<u32>(
                    read_w(ql_rel + ql_off + l + 32u),
                    read_w(ql_rel + ql_off + l + 33u),
                    read_w(ql_rel + ql_off + l + 34u),
                    read_w(ql_rel + ql_off + l + 35u),
                );
                let hb = vec4<u32>(
                    read_w(qh_rel + qh_off_h + l),
                    read_w(qh_rel + qh_off_h + l + 1u),
                    read_w(qh_rel + qh_off_h + l + 2u),
                    read_w(qh_rel + qh_off_h + l + 3u),
                );
                let v1 = q6_quarter(sc1, qa, hb, 0u, false);
                let v2 = q6_quarter(sc2, qb, hb, 2u, false);
                let v3 = q6_quarter(sc3, qa, hb, 4u, true);
                let v4 = q6_quarter(sc4, qb, hb, 6u, true);
                // Tile-local element base of the first quarter (a multiple of 4);
                // the quarters sit +32/+64/+96 elements, i.e. +8/+16/+24 vec4.
                let e0 = (p0 + row_base + l) >> 2u;
                let e1 = e0 + 8u;
                let e2 = e0 + 16u;
                let e3 = e0 + 24u;
                a0 = a0 + xarr[b0 + e0] * v1 + xarr[b0 + e1] * v2 + xarr[b0 + e2] * v3 + xarr[b0 + e3] * v4;
                a1 = a1 + xarr[b1 + e0] * v1 + xarr[b1 + e1] * v2 + xarr[b1 + e2] * v3 + xarr[b1 + e3] * v4;
                a2 = a2 + xarr[b2 + e0] * v1 + xarr[b2 + e1] * v2 + xarr[b2 + e2] * v3 + xarr[b2 + e3] * v4;
                a3 = a3 + xarr[b3 + e0] * v1 + xarr[b3 + e1] * v2 + xarr[b3 + e2] * v3 + xarr[b3 + e3] * v4;
                a4 = a4 + xarr[b4 + e0] * v1 + xarr[b4 + e1] * v2 + xarr[b4 + e2] * v3 + xarr[b4 + e3] * v4;
                a5 = a5 + xarr[b5 + e0] * v1 + xarr[b5 + e1] * v2 + xarr[b5 + e2] * v3 + xarr[b5 + e3] * v4;
                a6 = a6 + xarr[b6 + e0] * v1 + xarr[b6 + e1] * v2 + xarr[b6 + e2] * v3 + xarr[b6 + e3] * v4;
                a7 = a7 + xarr[b7 + e0] * v1 + xarr[b7 + e1] * v2 + xarr[b7 + e2] * v3 + xarr[b7 + e3] * v4;
            }
        }
    }

    store_tile(a0, a1, a2, a3, a4, a5, a6, a7, row0, col, colc);
}

