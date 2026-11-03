// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Embedding-style gather out of a PACKED GGUF Q8_0 table, dequantised to f32 as
// it is read — one dispatch, no host round trip and no intermediate f32 table.
//
// Table   bytes `rows * blocks_per_row * 34` (Q8_0: f16 d | 32 i8 quants per 32
//         elements), in a binding of its own: it is either a packed parameter
//         parked in the weight buffer or arena-resident, and only the caller
//         knows which.
// Indices flat, f32-encoded integers at `idx_off` (arena words).
// Output  `[n_idx, cols]` at `out_off` (arena words), `cols = blocks_per_row * 32`.
//
// One thread per Q8_0 block of the output: block `b` of the flattened
// `n_idx * blocks_per_row` grid is (index `b / blocks_per_row`, block
// `b % blocks_per_row`). The index arithmetic is `gather.wgsl`'s, with the
// folded bias added before truncation.

struct Params {
    n_idx: u32,          // number of indices (output rows)
    cols: u32,           // elements per row
    rows: u32,           // table rows (for clamping out-of-range indices)
    blocks_per_row: u32, // Q8_0 blocks per row, `cols / 32`
    tbl_byte_off: u32,   // byte offset of the table inside its own binding
    idx_off: u32,        // offset of the indices array (arena words)
    out_off: u32,        // offset of the output (arena words)
    idx_bias: f32,       // added to each index before truncation
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
// read_write even though this kernel only reads it: when the packed table is
// arena-resident, bindings 0 and 2 point at the same buffer, and a read-only +
// read-write pair on one buffer is a wgpu usage-scope conflict while rw+rw is
// legal.
@group(0) @binding(2) var<storage, read_write> table: array<u32>;

// Q8_0's block geometry: 32 elements per 34-byte block.
const Q8_0_BLOCK_ELEMS: u32 = 32u;
const Q8_0_BLOCK_BYTES: u32 = 34u;

// The byte-wise readers below are `dequant_gguf.wgsl`'s, over this kernel's
// table binding instead of the arena: Q8_0 blocks start on even byte offsets, so
// the 16-bit scale word can straddle a 32-bit arena word and has to be assembled
// byte by byte.
fn read_t(rel: u32) -> u32 {
    let abs = params.tbl_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (table[word] >> shift) & 0xffu;
}

fn read_t_i8(rel: u32) -> i32 {
    let b = read_t(rel);
    return select(i32(b), i32(b) - 256, b >= 128u);
}

fn dq_read_f16(rel: u32) -> f32 {
    let bits = read_t(rel) | (read_t(rel + 1u) << 8u);
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var v: f32;
    if (exp == 0u) { v = f32(mant) / 1024.0 * exp2(-14.0); }
    else if (exp == 31u) { v = select(0.0, bitcast<f32>(0x7f800000u), mant == 0u); }
    else { v = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp) - 15)); }
    return select(v, -v, sign != 0u);
}

@compute @workgroup_size(64)
fn gather_packed(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let b = gid.x + gid.y * ngs.x * 64u;
    let total = params.n_idx * params.blocks_per_row;
    if (b >= total) { return; }
    let i = b / params.blocks_per_row;
    let blk = b % params.blocks_per_row;
    let idx_f = arena[params.idx_off + i] + params.idx_bias;
    var idx_u: u32 = u32(max(idx_f, 0.0));
    // `rows - 1` would wrap at zero; the host refuses such a table, and this does not
    // depend on that.
    let last = max(params.rows, 1u) - 1u;
    if (idx_u > last) { idx_u = last; }
    // The row's block `blk`, as a byte offset relative to the table's start.
    let off = idx_u * (params.blocks_per_row * Q8_0_BLOCK_BYTES) + blk * Q8_0_BLOCK_BYTES;
    let d = dq_read_f16(off);
    let qs_rel = off + 2u;
    let dst = params.out_off + i * params.cols + blk * Q8_0_BLOCK_ELEMS;
    for (var j: u32 = 0u; j < Q8_0_BLOCK_ELEMS; j = j + 1u) {
        arena[dst + j] = d * f32(read_t_i8(qs_rel + j));
    }
}
