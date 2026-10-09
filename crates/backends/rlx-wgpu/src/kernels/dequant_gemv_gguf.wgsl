// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — fused GGUF packed GEMV (decode m=1) for `Op::DequantMatMul`.
//
// Computes one row of `y = x @ W^T` where `W` is a GGUF-packed weight `[n, k]`
// (row-major blocks) and `x` is `[1, k]`. Both entries dequantize on the fly
// (NO f32 scratch) and mirror `dequant_gguf.wgsl`'s math; only the sink differs
// (accumulate vs store). They differ in how one row's work is arranged:
//
//   * `dequant_gemv_tiled` (Q4_K, Q6_K): one WORKGROUP per output column, and
//     that workgroup's 32 lanes split the contraction instead of one thread
//     walking the whole row. A lane walks one 32-element sub-block with 4-byte
//     word loads, so a warp round's requests land in the same few packed blocks
//     rather than in one strided sector per lane, and it reads X as `vec4` (8
//     vector loads per lane instead of 32 scalar ones) — which is why this arm's
//     X window has to line up with a four-element group, i.e. `x_f32_off` be a
//     multiple of 4. The device work a one-row multiply issues is then the row's
//     own `columns x contraction`: at the model widths (1024…6144 columns) the
//     one-thread-per-column shape left only 16…96 workgroups, far below what
//     those widths can occupy.
//   * `dequant_gemv` (Q1_0): one thread per output column, as before. The 1-bit
//     scheme's blocks are 128 elements with an 8-element sign byte, a different
//     element/byte mapping than either K-quant, so it keeps this shape.
//
// Lane partials of `dequant_gemv_tiled` are combined by a fixed-order tree in
// `red`: deterministic, and no floating-point atomics (absent on some adapters).
//
// Binding scheme works around two wgpu limits that the whole-arena
// `as_entire_binding` path hits on multi-GiB models:
//   1. a storage binding may be at most `max_storage_buffer_binding_size`
//      (4 GiB) — so `x` and `weight` are bound as separate *windowed* slabs of
//      the arena, each well under 4 GiB;
//   2. a buffer bound STORAGE_READ_WRITE is exclusive in a dispatch (cannot
//      also be bound read-only) — so the arena is bound read-only twice (x +
//      weight) and `y` is written to a small *separate* output buffer that the
//      caller copies back into the arena.
//
// Block element order is identity (`out_i` == k-offset within the block), so
// `x[k_base + out_i]`.

struct Params {
    k: u32,            // contraction dim (multiple of 256)
    n: u32,            // output columns (weight rows)
    scheme_id: u32,    // 0 = Q4_K, 2 = Q6_K, 24 = Q1_0
    x_f32_off: u32,    // f32 index of x[0] within the x binding
    w_byte_off: u32,   // byte offset of weight[0,0] within the weight binding
    out_f32_off: u32,  // f32 index of y[0] within the output binding
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read>        xarr: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        warr: array<u32>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;
// The same `x` window as binding 0, seen four elements at a time: the tiled arm
// reads a lane's 32 contiguous X elements with 8 vector loads instead of 32
// scalar ones, so `x_f32_off` has to be a multiple of 4. The Rust side admits a
// call only when it is (`gemv_one_row_device_ok`) and sends the rest to the host
// rather than reading X shifted. The one-thread-per-column arm keeps the scalar
// view.
@group(0) @binding(4) var<storage, read>        xarr4: array<vec4<f32>>;

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

// The four packed bytes starting at `rel` as one word, for a `rel` that may be
// only 2-byte aligned: Q6_K's 210-byte blocks put every odd block's `ql`/`qh`
// slices there, where a plain word load would be misaligned.
fn read_word_at(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let sh = (abs % 4u) * 8u;
    if (sh == 0u) { return warr[word]; }
    return (warr[word] >> sh) | (warr[word + 1u] << (32u - sh));
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

// The four weights one Q4_K `qs` word carries for a sub-block, `i`-th first: the
// nibble a 4-element group of X multiplies, so the group's one vector load and
// this word's four extracts stay written together.
fn dq_q4k_x4(word: u32, i: u32, sh: u32) -> f32 {
    return f32((word >> (8u * i + sh)) & 0x0Fu);
}

// The four weights one Q6_K `ql`/`qh` word pair carries for a sub-block, `i`-th
// first (4 low bits from `ql`, 2 high bits from `qh`, biased by 32).
fn dq_q6k_x4(qlw: u32, qhw: u32, i: u32, sh: u32, qbits: u32) -> f32 {
    let q = ((qlw >> (8u * i + sh)) & 0x0Fu) | (((qhw >> (8u * i + qbits)) & 3u) << 4u);
    return f32(i32(q) - 32);
}

// Lane partials of one output column, reduced by `dequant_gemv_tiled`. 32 lanes
// x 4 bytes = 128 B of workgroup storage — inside the 16 KiB WebGPU floor, and
// both entries stay at or under 256 invocations, so the arm is portable shading
// rather than an adapter-specific shape. It uses no optional device feature: no
// floating-point atomics (absent on some adapters), no subgroup operation, and
// no assumption about subgroup width.
var<workgroup> red: array<f32, 32>;

@compute @workgroup_size(64)
fn dequant_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.n) { return; }
    let xb = params.x_f32_off;
    var acc = 0.0;

    // The scheme test stays an explicit branch (rather than being folded into
    // this being the Q1_0-only entry) so `gguf_kernel_scheme_coverage.rs` can
    // tie it to the schemes routed here: a scheme reaching this entry without a
    // branch reads back as zeros, which is visible, where a folded body would
    // decode it as something else.
    if (params.scheme_id == 24u) {
        // Q1_0 (prism-ml Bonsai-27B): 18 bytes / 128 elems. f16 scale `d` + 16
        // sign bytes, LSB-first, bit1 -> +d, bit0 -> -d. Reads packed weight
        // directly (no f32 scratch). k is a multiple of 128.
        let nbk = params.k / 128u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * 18u;
            let d = dq_read_f16(off);
            let nd = -d;
            let ko = b * 128u;
            for (var byte: u32 = 0u; byte < 16u; byte = byte + 1u) {
                let bits = read_w(off + 2u + byte);
                for (var bit: u32 = 0u; bit < 8u; bit = bit + 1u) {
                    let w = select(nd, d, ((bits >> bit) & 1u) != 0u);
                    acc = acc + xarr[xb + ko + byte * 8u + bit] * w;
                }
            }
        }
    }

    outarr[params.out_f32_off + j] = acc;
}

// One workgroup per output column; its 32 lanes split that column's
// contraction. A unit of work is one 32-element sub-block of the row, and a lane
// walks its own sub-block serially with 4-byte word loads — so the warp's 32
// lanes sweep the row's packed bytes in dense four-byte requests rather than
// reading one strided byte per output column — while its X side is four of those
// sub-block's elements per `vec4<f32>` load. The scale/min unpacking a sub-block
// needs stays amortized over its 32 elements (the reason a lane cannot be given
// a single element: the per-block constants then cost more than the arithmetic).
@compute @workgroup_size(32)
fn dequant_gemv_tiled(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j = wid.x;
    if (j >= params.n) { return; }
    let nbk = params.k / 256u;        // 256-elem blocks per weight row
    let lane = lid.x;
    let xb4 = params.x_f32_off / 4u;  // X as vec4 groups (see `xarr4`)
    var acc = 0.0;

    if (params.scheme_id == 0u) {
        // Q4_K: 144 bytes / 256 elems. Sub-block `s` (8 per block) covers
        // elements [32s, 32s+32): the low nibble (even s) or the high nibble
        // (odd s) of byte `l` of its block's 32-byte group `s/2`. One `qs` word
        // holds the four weights of a four-element X group, so the word and that
        // group's one vector load are consumed together.
        for (var s = lane; s < nbk * 8u; s = s + 32u) {
            let off = (j * nbk + s / 8u) * 144u;
            let sb = s % 8u;
            let sm = dq_get_scale_min_k4(off + 4u, sb);
            let d0 = dq_read_f16(off) * f32(sm.x);
            let m0 = dq_read_f16(off + 2u) * f32(sm.y);
            let qs = off + 16u + 32u * (sb / 2u);
            let e4 = s * 8u;
            let sh = select(0u, 4u, (sb % 2u) == 1u);
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = read_word_at(qs + 4u * w);
                let xv = xarr4[xb4 + e4 + w];
                acc = acc + xv.x * (d0 * dq_q4k_x4(word, 0u, sh) - m0);
                acc = acc + xv.y * (d0 * dq_q4k_x4(word, 1u, sh) - m0);
                acc = acc + xv.z * (d0 * dq_q4k_x4(word, 2u, sh) - m0);
                acc = acc + xv.w * (d0 * dq_q4k_x4(word, 3u, sh) - m0);
            }
        }
    } else if (params.scheme_id == 2u) {
        // Q6_K: 210 bytes / 256 elems. Sub-block `s` is quarter `q = s%4` of
        // half `h = s/4`: its 32 elements take one nibble of the quarter's
        // 32-byte `ql` slice, two bits of `qh`, and one of the half's 16 i8
        // scales per 16 elements.
        for (var s = lane; s < nbk * 8u; s = s + 32u) {
            let b = s / 8u;
            let h = (s % 8u) / 4u;
            let q = s % 4u;
            let off = (j * nbk + b) * 210u;
            let d = dq_read_f16(off + 208u);
            let sc0 = d * f32(read_w_i8(off + 192u + h * 8u + 2u * q));
            let sc1 = d * f32(read_w_i8(off + 192u + h * 8u + 2u * q + 1u));
            let qlb = off + h * 64u + (q % 2u) * 32u;
            let qhb = off + 128u + h * 32u;
            let sh = select(0u, 4u, q >= 2u);
            let qbits = 2u * q;
            let e4 = b * 64u + h * 32u + q * 8u;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let qlw = read_word_at(qlb + 4u * w);
                let qhw = read_word_at(qhb + 4u * w);
                let scl = select(sc0, sc1, w >= 4u);
                let xv = xarr4[xb4 + e4 + w];
                acc = acc + xv.x * (scl * dq_q6k_x4(qlw, qhw, 0u, sh, qbits));
                acc = acc + xv.y * (scl * dq_q6k_x4(qlw, qhw, 1u, sh, qbits));
                acc = acc + xv.z * (scl * dq_q6k_x4(qlw, qhw, 2u, sh, qbits));
                acc = acc + xv.w * (scl * dq_q6k_x4(qlw, qhw, 3u, sh, qbits));
            }
        }
    }

    // Fixed-order tree over the lane partials: the same order on every lane and
    // every run, so the sum is deterministic and needs no f32 atomics.
    red[lane] = acc;
    workgroupBarrier();
    if (lane < 16u) { red[lane] = red[lane] + red[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { red[lane] = red[lane] + red[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { red[lane] = red[lane] + red[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { red[lane] = red[lane] + red[lane + 2u]; }
    workgroupBarrier();
    if (lane == 0u) { outarr[params.out_f32_off + j] = red[0u] + red[1u]; }
}
