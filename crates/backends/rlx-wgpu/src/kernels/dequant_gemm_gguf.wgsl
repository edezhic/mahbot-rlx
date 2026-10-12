// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused multi-row GGUF packed-weight GEMM (prefill `m > 1`) for
// `Op::DequantMatMul`.
//
// One entry point per scheme — `dequant_gemm_q4k` (scheme 0), `dequant_gemm_q6k`
// (scheme 2) and `dequant_gemm_q8_0` (scheme 20, the byte-per-weight form).
// Deliberately NOT a `scheme_id` branch chain like `dequant_gguf.wgsl`: an id
// routed to a kernel with no branch for it matches nothing and reads the weights
// as **zeros** instead of failing.
//
// In the base geometry each thread owns one output column `j` and accumulates
// `TM = 8` rows of `Y[m,n] = X[m,k] @ dequant(W)[n,k]^T` straight from the packed
// weight (no f32 scratch) — that is what the Q6_K and Q8_0 entries run at every
// shape, and what the Q4_K entry runs wherever the encoder does not put it on the
// split geometry below. There a column's contraction is halved between two
// threads, so a thread owns half a column rather than a column. **Both**
// geometries read every packed quant byte **once per row group** and reuse it for
// all 8 rows — that is the whole point of this kernel, and why the per-row GEMV
// (which re-streams the weight once per row) does not scale in `m`. The split's
// block headers are the exception: its two lane groups each read one column's
// header, so every block's header is read twice under it (its own paragraph
// below).
//
// Except for the parity bands this tree measures itself further down, the timing
// figures in this header are the author's own out-of-tree A/B of the raw kernel
// and its variants on one adapter (wgpu over Metal, M3 Pro), timed per byte and
// per variant rather than as graph-call ratios, and are to be re-taken rather
// than trusted: the harnesses this header names time whole graph calls instead
// (`tests/gguf_gemm_grouped_weight_reuse.rs` answers this kernel's
// row-group-versus-single-row question that way).
//
// Both operands are read through the cheapest path that measured here, and which
// path that is was not obvious. The weight is read as whole 4-byte words: a byte
// at a time, with a shift and a mask each, as this kernel first did, was one of
// the two paths that put it at 2.1-2.7x a single-row pass. The Q4_K entry point
// also reads its block's `d|dmin` and 12 scale bytes as the four whole words they
// are (`read_u32`, `q4k_scale_min`), where a per-byte read carries a `/4`, a `%4`
// and a shift each; Q6_K keeps its own readers — `read_word_at` reassembles a
// word across the 2-byte boundary its 210-byte blocks put every odd block's
// `ql`/`qh` slices on, and its `d` plus its 16 i8 scales are read a byte at a
// time (`read_w`, `read_w_i8`). The activations go through the
// workgroup-staged [`xs`] tile below, because every thread of a workgroup
// multiplies the SAME X and reading it per thread made the X side the kernel's
// largest term.
//
// What still limits the kernel is that staged-X path: `TM` staged loads per
// weight word per output column is 8 loads per 32 MACs, and the only way past
// that is more output columns per thread, which this adapter pays for in
// registers — a two-column variant spilled its accumulators and measured 2x
// SLOWER. Nor is the number of times the weight is re-read what a step pays for:
// an 8-row group and a 1-row group of the same packed bytes cost the same per
// byte, a 16-byte vectorized weight fetch and a doubled FMA count both measured
// flat (so the kernel is not issue-bound), and a tile of 16 rows — which halves
// the re-reads and does the same total row work wherever the row count reaches
// into the tile's top half — measured a wash there and a regression below it,
// where the tile computes 16 rows' worth for fewer real ones.
//
// `tests/gguf_gemm_grouped_weight_reuse.rs` holds the measurement of what
// one row group costs against the single-row pass it replaces. It no longer
// asserts a bound on it: the bound was denominated in single-row passes, so
// speeding the single-row arm up tightened it without this kernel changing, and
// it was retired rather than re-denominated — so this route has no in-tree
// regression ceiling left, and the rig's own re-take of the grouped cells is what
// covers it.
//
// Binding scheme matches `dequant_gemv_gguf.wgsl`: `x` and `weight` are separate
// read-only windows of the arena (each under `max_storage_buffer_binding_size`)
// and `y` goes to a separate buffer the caller copies back — a storage binding
// bound read-write is exclusive in a dispatch and cannot also be read.
//
// Grid: x = ceil(ncols / TG) column groups of `TG` columns (one thread per
// column), y = ceil(m / TM) row tiles — the grid the Rust side sizes from the
// dispatch constants in `gguf_gpu.rs`, whose own test pins these two consts
// against them. The Q4_K entry point's split geometry (below) covers `COLS`
// columns per workgroup with a thread to each half-`k` of a column instead, so
// there its x is ceil(ncols / COLS) and its y is unchanged. Rows past `m` clamp
// to the last valid row (their loads stay in bounds) and are dropped at the
// store.
//
// The Q4_K entry point runs a second geometry when the dispatch is too small to
// fill the GPU. One dispatch costs about the same here whether it has 16
// workgroups or 500, so a dispatch of a few dozen leaves the device idle; at
// `GROUPED_SPLIT_MAX_BASE_WORKGROUPS` base workgroups and fewer the encoder sets
// `params.split`, and a workgroup then covers `COLS` = 32 columns with two
// threads to each, splitting `k` down the middle (`sl`): the same packed quant
// bytes over up to twice the warps, with the x grid at ceil(ncols / COLS) — twice
// the base grid's ceil(ncols / TG), or one workgroup short of twice where the
// chunk's last thread group is half-full or less (ncols = 65 is 2 column groups
// base and 3 split) — and y unchanged. A column's blocks are read by both of its
// threads, so the quant bytes read are the base geometry's while each block's
// 16-byte header is read twice. The extra warps win on the starved dispatches it
// is taken for, still win a little way above the ceiling, are a wash just past
// that, and lose further out — that const's own comment in `gguf_gpu.rs` carries
// the measured band and what it costs the base geometry to have this arm in the
// module — and the flag is set for scheme 0 alone: the Q6_K and Q8_0 entries have
// no split arm and run the base geometry whatever the word says.
//
// The dequant is the GEMV's, element for element and in the same k order; the
// accumulation is not — each thread keeps four vec4<f32> lanes per row and adds
// them horizontally at the store (`dot4`), where the one-row arms sum in their
// own orders (the tiled arm reduces lane partials in a fixed tree), so round-off
// differs. A split workgroup adds a third order again: the two half-`k` threads'
// partial sums meet in workgroup memory first — `sl` = 1's half written, `sl` =
// 0's thread adding it to its own before the `dot4` — the same terms in one fixed
// order, with no unordered updates. The measured CPU-reference parity is the
// real justification: max-abs 1.9e-5 .. 6.5e-4 over the prefill-parity shapes,
// against the 1e-2 this op is compared at. The Q8_0 entry's own band, over the
// cells `rlx-wgpu/tests/gguf_matmul_route.rs` measures it at, is 2.7e-5 .. 4.2e-4
// (identical against the CPU reference and against the scratch route it
// replaces), well inside that gate.
//
// The Q6_K dequant (`read_w`, `read_w_i8`, `dq_read_f16`, `bytes_x4`, `q6c`,
// `q6_quarter`) is the one-row tiled arm's, `dequant_gemv_tiled` of
// `dequant_gemv_gguf.wgsl`, rewritten to build one `vec4` per 4 weights; Q4_K's
// is that same math with its scale/min bytes and nibble words unpacked as whole
// words (`q4k_scale_min`, `q4k_nibbles`). A fix to the nibble or scale/min layout
// has four copies to reach: `dequant_gguf.wgsl`, `dequant_gemv_gguf.wgsl`, and
// this file's `dequant_gemm_q4k` **twice** — once per arm of its split branch,
// which are the same math over different spans of `k` (the arms' own comment
// says why they are two). Q8_0's (`q8_0_quants`, the entry at the end of this
// file) has one other copy, the `scheme_id == 20` branch of `dequant_gguf.wgsl`.

struct Params {
    m: u32,           // rows
    k: u32,           // contraction dim (multiple of 256, hence of 4)
    ncols: u32,       // output columns in this dispatch (also the out row stride)
    x_vec4_off: u32,  // vec4 index of X[0,0] within the x binding
    w_byte_off: u32,  // byte offset of W[col0,0] within the weight binding
    // 1 = the Q4_K entry point's split geometry, 0 = the base one (see the grid
    // paragraph above). Set by the encoder for a dispatch too small to fill the
    // GPU; the Q6_K and Q8_0 entries ignore it.
    split: u32,
    // 8 words total: a uniform binding size must be a multiple of 16 bytes.
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read>        xarr: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        warr: array<u32>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;

const TM: u32 = 8u;
const TG: u32 = 64u;
const BLK_ELEMS: u32 = 256u;
/// Columns one workgroup of the Q4_K entry point covers when the split geometry
/// is on: half of `TG`, the two halves being the workgroup's two k-lane groups.
/// The encoder's `GROUPED_SPLIT_COLS` sizes the x grid from it and
/// `grouped_geometry_matches_the_shader` ties the two.
const COLS: u32 = 32u;

/// One packed byte at `rel`, for a `rel` with no alignment at all.
fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (warr[word] >> shift) & 0xffu;
}

/// Four packed bytes at `rel` as one word. What has to be a multiple of 4 is the
/// whole address `params.w_byte_off + rel`, and both halves of it are: `rel` is a
/// multiple of the 144-byte Q4_K block, and `w_byte_off` is the window's own
/// offset into the packed weight — `w0 mod 256`, for a weight slot the memory
/// plan aligned — so it is one too. (The 256-aligned quantity is the binding base
/// this subtracts, `w_base`; that is not what makes the load legal.) Needs no
/// alignment branch, unlike [`read_word_at`], so the load has no divergent path
/// to sit behind. The encoder holds this precondition rather than assuming it:
/// a Q4_K window that is not 4-aligned declines to the per-row GEMV instead of
/// reading shifted weights, and the Q6_K entry point — whose reader reassembles
/// across a byte boundary — needs no such check.
fn read_u32(rel: u32) -> u32 {
    return warr[(params.w_byte_off + rel) >> 2u];
}

/// Four packed bytes at `rel` as one word, for a `rel` that need not be aligned:
/// Q6_K's 210-byte blocks put every odd block's `ql`/`qh` slices on a 2-byte
/// boundary. The branch prices the reassembly (two loads, a shift, an or) above
/// [`read_u32`], which is why the Q4_K entry point keeps its own loader.
fn read_word_at(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let sh = (abs % 4u) * 8u;
    if (sh == 0u) { return warr[word]; }
    return (warr[word] >> sh) | (warr[word + 1u] << (32u - sh));
}

fn read_w_i8(rel: u32) -> i32 {
    let b = read_w(rel);
    return select(i32(b), i32(b) - 256, b >= 128u);
}

/// The f16 whose bits are `bits`, as f32. One body for both readers: the K-quant
/// entries hand it two packed bytes ([`dq_read_f16`]), the Q4_K entry the half of
/// a word its block's `d`/`dmin` occupy.
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var v: f32;
    if (exp == 0u) { v = f32(mant) / 1024.0 * exp2(-14.0); }
    else if (exp == 31u) { v = select(0.0, bitcast<f32>(0x7f800000u), mant == 0u); }
    else { v = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp) - 15)); }
    return select(v, -v, sign != 0u);
}

fn dq_read_f16(rel: u32) -> f32 {
    return f16_bits_to_f32(read_w(rel) | (read_w(rel + 1u) << 8u));
}

/// The `j`-th (scale, min) pair of a Q4_K block's 12 packed scale bytes, with
/// those bytes already read as the three words they are — the Q4_K entry point
/// reads them once per block per thread, and so twice per block under the split
/// geometry, whose two lane groups each read the header — rather than byte by byte,
/// because a byte read costs a `/4`, a `%4` and a shift on top of the load, while
/// `j` folds to a constant in the base arm's unrollable sub-block loop (the split
/// arm passes its runtime sub-block index there instead).
fn q4k_scale_min(w0: u32, w1: u32, w2: u32, j: u32) -> vec2<u32> {
    let i = j & 3u;
    let a = (w0 >> (8u * i)) & 0xFFu;   // scales 0..3, and the high bits of 4..7
    let b = (w1 >> (8u * i)) & 0xFFu;   // mins 0..3, and the high bits of 4..7
    let c = (w2 >> (8u * i)) & 0xFFu;   // low nibble: scales 4..7, high: mins 4..7
    if (j < 4u) { return vec2<u32>(a & 63u, b & 63u); }
    return vec2<u32>((c & 0x0Fu) | (((a >> 6u) & 3u) << 4u), (c >> 4u) | (((b >> 6u) & 3u) << 4u));
}

fn dot4(v: vec4<f32>) -> f32 { return v.x + v.y + v.z + v.w; }

/// The four Q4_K weights one `qs` word carries for a sub-block, low nibble
/// (`sh == 0`) or high nibble (`sh == 4`) — the group of four X elements one
/// `vec4` load multiplies. Returned the way `unpack4x8unorm` gives them, i.e.
/// `nibble / 255`: one mask plus one unpack stand in for four shifts, four masks
/// and four `f32` conversions, and the caller folds the 255 back into the
/// sub-block scale instead, so the four weights cost one `fma`.
fn q4k_nibbles(word: u32, sh: u32) -> vec4<f32> {
    return unpack4x8unorm((word >> sh) & 0x0F0F0F0Fu);
}

/// `unpack4x8unorm`'s divisor, folded back into `d`/`dmin` per sub-block.
const DQ_UNORM: f32 = 255.0;

// Q4_K: 144 bytes / 256 elems (f16 d + f16 dmin + 12 packed scales + 128 nibbles).
@compute @workgroup_size(64, 1, 1)
fn dequant_gemm_q4k(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // Geometry, one thread per output column in the base form and one thread per
    // half-`k` of a column in the split one. Both are uniform across the
    // workgroup, so every branch and barrier below is too.
    let split = params.split == 1u;
    let cl = select(tid, tid % COLS, split);
    let sl = select(0u, tid / COLS, split);
    let col = wid.x * select(TG, COLS, split) + cl;
    let row0 = wid.y * TM;
    // Clamp out-of-range columns rather than returning: every thread of the
    // workgroup has to reach the staging barriers, and out-of-range results are
    // dropped at the store.
    let colc = min(col, params.ncols - 1u);

    let nbk = params.k / BLK_ELEMS;
    let blk = 2u + 2u + 12u + 128u;
    let k4 = params.k >> 2u;
    // Rows past `m` clamp to the last valid row; their accumulators are dropped
    // at the store.
    let last_row = params.m - 1u;
    let xb = params.x_vec4_off;

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
        // Stage this block's activation tile; the barrier pair keeps the
        // previous block's reads out of this one's writes.
        workgroupBarrier();
        stage_x(tid, row0, p0, k4, xb, last_row);
        workgroupBarrier();

        let off = (colc * nbk + b) * blk;
        // `off` is a multiple of 4 (144-byte blocks against a 256-aligned
        // window), so the block's d|dmin pair and its 12 scale bytes are three
        // and one whole words — one load each, not two per f16 and one per byte.
        let d_pack = read_u32(off);
        let d = f16_bits_to_f32(d_pack & 0xFFFFu);
        let dmin = f16_bits_to_f32(d_pack >> 16u);
        let scw0 = read_u32(off + 4u);
        let scw1 = read_u32(off + 8u);
        let scw2 = read_u32(off + 12u);
        let qs_rel = off + 16u;
        // Two arms, not one loop over a `split`-dependent range: the base arm's
        // sub-block index `t` has to stay a compile-time constant in each unrolled
        // iteration (that is what lets `q4k_scale_min` fold its byte indices), and
        // a shared loop would derive it from the runtime `sl` instead — costing
        // the base arm what the split arm pays anyway. The two arms are the same
        // math over different spans of `k` — the base arm's `t` loop runs the whole
        // block, the split arm one half of it — so a change to one is a change to
        // the other.
        if (split) {
            // This thread's half of the block: sub-blocks `4*sl` .. `4*sl+4`,
            // elements [sl*128, sl*128 + 128), whose quant bytes are `64*sl`
            // bytes into the block's quant area. Two `h` steps of two sub-blocks
            // each, paired the base loop pairs them — a 32-byte word group's low
            // nibbles then the same group's high nibbles — so each sub-block
            // keeps its own scale and min and the dequant is the base arm's,
            // element for element and in the same k order, over half the block.
            var lp = sl * 128u;   // k offset within the block of this slice
            var is = sl * 64u;    // its byte offset into the block's quant area
            for (var h: u32 = 0u; h < 2u; h = h + 1u) {
                let t = sl * 4u + h * 2u;
                let sm0 = q4k_scale_min(scw0, scw1, scw2, t);
                let sm1 = q4k_scale_min(scw0, scw1, scw2, t + 1u);
                let d0 = d * f32(sm0.x);
                let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x);
                let m1 = dmin * f32(sm1.y);
                // `q4k_nibbles` yields `nibble / DQ_UNORM`; folding the divisor
                // into the two per-sub-block constants keeps the weight to one
                // `fma`, as in the base arm.
                let d0u = vec4<f32>(d0 * DQ_UNORM);
                let d1u = vec4<f32>(d1 * DQ_UNORM);
                let nm0 = vec4<f32>(-m0);
                let nm1 = vec4<f32>(-m1);
                // The base arm's two nibble passes, low half fully before high
                // half, over this slice's 32-byte word group. Both warps of a
                // workgroup read the same 32 columns here, so the weight reads
                // stay as coalesced as the base arm's.
                let qsw = qs_rel + is;
                for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                    let ww = fma(q4k_nibbles(read_u32(qsw + l), 0u), d0u, nm0);
                    let o = lp + l;
                    a0 = a0 + xs_at(0u, o) * ww;
                    a1 = a1 + xs_at(1u, o) * ww;
                    a2 = a2 + xs_at(2u, o) * ww;
                    a3 = a3 + xs_at(3u, o) * ww;
                    a4 = a4 + xs_at(4u, o) * ww;
                    a5 = a5 + xs_at(5u, o) * ww;
                    a6 = a6 + xs_at(6u, o) * ww;
                    a7 = a7 + xs_at(7u, o) * ww;
                }
                lp = lp + 32u;
                for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                    let ww = fma(q4k_nibbles(read_u32(qsw + l), 4u), d1u, nm1);
                    let o = lp + l;
                    a0 = a0 + xs_at(0u, o) * ww;
                    a1 = a1 + xs_at(1u, o) * ww;
                    a2 = a2 + xs_at(2u, o) * ww;
                    a3 = a3 + xs_at(3u, o) * ww;
                    a4 = a4 + xs_at(4u, o) * ww;
                    a5 = a5 + xs_at(5u, o) * ww;
                    a6 = a6 + xs_at(6u, o) * ww;
                    a7 = a7 + xs_at(7u, o) * ww;
                }
                lp = lp + 32u;
                is = is + 32u;
            }
        } else {
            var lp = 0u;   // k offset within the block for the current 32-elem half
            var is = 0u;
            for (var t: u32 = 0u; t < 8u; t = t + 2u) {
                let sm0 = q4k_scale_min(scw0, scw1, scw2, t);
                let sm1 = q4k_scale_min(scw0, scw1, scw2, t + 1u);
                let d0 = d * f32(sm0.x);
                let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x);
                let m1 = dmin * f32(sm1.y);
                // `q4k_nibbles` yields `nibble / DQ_UNORM`; folding the divisor into
                // the two per-sub-block constants keeps the weight to one `fma`.
                let d0u = vec4<f32>(d0 * DQ_UNORM);
                let d1u = vec4<f32>(d1 * DQ_UNORM);
                let nm0 = vec4<f32>(-m0);
                let nm1 = vec4<f32>(-m1);
                // Two separate nibble passes, low half fully before high half, so a
                // sub-block's elements land in the same k positions here and in the
                // one-row arms' dequant. One 4-byte word feeds one 4-element group,
                // as in the one-row tiled arm: the weight is read as words, not one
                // byte at a time with a shift and a mask each.
                let qsw = qs_rel + is;
                for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                    let ww = fma(q4k_nibbles(read_u32(qsw + l), 0u), d0u, nm0);
                    let o = lp + l;
                    a0 = a0 + xs_at(0u, o) * ww;
                    a1 = a1 + xs_at(1u, o) * ww;
                    a2 = a2 + xs_at(2u, o) * ww;
                    a3 = a3 + xs_at(3u, o) * ww;
                    a4 = a4 + xs_at(4u, o) * ww;
                    a5 = a5 + xs_at(5u, o) * ww;
                    a6 = a6 + xs_at(6u, o) * ww;
                    a7 = a7 + xs_at(7u, o) * ww;
                }
                lp = lp + 32u;
                for (var l: u32 = 0u; l < 32u; l = l + 4u) {
                    let ww = fma(q4k_nibbles(read_u32(qsw + l), 4u), d1u, nm1);
                    let o = lp + l;
                    a0 = a0 + xs_at(0u, o) * ww;
                    a1 = a1 + xs_at(1u, o) * ww;
                    a2 = a2 + xs_at(2u, o) * ww;
                    a3 = a3 + xs_at(3u, o) * ww;
                    a4 = a4 + xs_at(4u, o) * ww;
                    a5 = a5 + xs_at(5u, o) * ww;
                    a6 = a6 + xs_at(6u, o) * ww;
                    a7 = a7 + xs_at(7u, o) * ww;
                }
                lp = lp + 32u;
                is = is + 32u;
            }
        }
    }

    if (split) {
        // Fold the two half-`k` partials per column, `sl` = 1 first: a slot per
        // row, so the two threads of a column never touch the same one, and a
        // stride of `TM + 1` slots rather than `TM` so that a warp's one-row-at-
        // a-time accesses spread across bank groups instead of landing in one.
        // The slots are [`xs`]'s own first `COLS * (TM + 1)` — the last block's
        // staged tile, dead once every thread has accumulated it, which the
        // barrier here establishes.
        //
        // They are [`xs`]'s rather than a second array's because workgroup storage
        // is declared per entry point, so a dedicated fold array would be storage
        // the base geometry's threads never touch but do pay for. The fold's
        // footprint inside the tile — and the pin that keeps `COLS * (TM + 1)`
        // inside it — are in `gguf_gpu.rs`'s `grouped_geometry_matches_the_shader`;
        // what this arm costs the base geometry is on
        // `GROUPED_SPLIT_MAX_BASE_WORKGROUPS`'s doc.
        workgroupBarrier();
        if (sl == 1u) {
            xs[cl * (TM + 1u)] = a0;
            xs[cl * (TM + 1u) + 1u] = a1;
            xs[cl * (TM + 1u) + 2u] = a2;
            xs[cl * (TM + 1u) + 3u] = a3;
            xs[cl * (TM + 1u) + 4u] = a4;
            xs[cl * (TM + 1u) + 5u] = a5;
            xs[cl * (TM + 1u) + 6u] = a6;
            xs[cl * (TM + 1u) + 7u] = a7;
        }
        workgroupBarrier();
        if (sl == 0u) {
            a0 = a0 + xs[cl * (TM + 1u)];
            a1 = a1 + xs[cl * (TM + 1u) + 1u];
            a2 = a2 + xs[cl * (TM + 1u) + 2u];
            a3 = a3 + xs[cl * (TM + 1u) + 3u];
            a4 = a4 + xs[cl * (TM + 1u) + 4u];
            a5 = a5 + xs[cl * (TM + 1u) + 5u];
            a6 = a6 + xs[cl * (TM + 1u) + 6u];
            a7 = a7 + xs[cl * (TM + 1u) + 7u];
        }
    }

    // Only a thread holding a whole column's sum stores it — the `sl` = 1 thread
    // of the split geometry holds half of it. `sl` is 0 for every thread of the
    // base geometry, so that arm's guard is just the column bound.
    if (sl == 0u && col < params.ncols) {
        store_col(a0, a1, a2, a3, a4, a5, a6, a7, row0, col);
    }
}

// Write the `TM` row results of one column. Split out so both entries share it.
fn store_col(
    a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>,
    a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>,
    row0: u32, col: u32,
) {
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

/// One 4-byte word as its four bytes, `i`-th first — what `q6_quarter` consumes,
/// so a quarter's `ql`/`qh` slice is one word load instead of four byte loads.
fn bytes_x4(w: u32) -> vec4<u32> {
    return vec4<u32>(w & 0xFFu, (w >> 8u) & 0xFFu, (w >> 16u) & 0xFFu, (w >> 24u) & 0xFFu);
}

/// The staged activation tile: `TM` rows x one block of `k` per row, one `vec4`
/// per four-element group — 8 KiB, and this entry point's whole workgroup
/// storage: the split geometry's fold reuses its first slots (see the fold below)
/// rather than declaring a second array, so what bounds `COLS` is `TG`/2 — the two
/// k-lane groups — and not the 16 KiB WebGPU workgroup-storage floor. Every thread
/// of a workgroup multiplies the SAME activations (they differ only in output
/// column), so reading them from device memory once per thread made the X side
/// this kernel's cost: `TM` device `vec4` loads per 4 weight bytes, each of the 64
/// columns paying them. Staging each block's tile once and reading it back from
/// workgroup memory is the third leg of the same load-path rebuild the one-row
/// tiled arm's `vec4` X read and this kernel's word-granular weight read are the
/// first two of.
var<workgroup> xs: array<vec4<f32>, TM * (BLK_ELEMS / 4u)>;

/// Stage `X[row0, row0+TM) x [p0, p0+BLK_ELEMS)` into [`xs`], one `vec4` per
/// thread per round. Rows past `m` clamp to the last valid row, exactly as the
/// weight reads do; their accumulators are dropped at the store.
fn stage_x(tid: u32, row0: u32, p0: u32, k4: u32, xb: u32, last_row: u32) {
    let per_row = BLK_ELEMS / 4u;
    for (var i: u32 = tid; i < TM * per_row; i = i + TG) {
        let r = i / per_row;
        let c4 = (i % per_row) * 4u;
        xs[i] = xarr[xb + min(row0 + r, last_row) * k4 + ((p0 + c4) >> 2u)];
    }
}

/// [`xs`] at element offset `o` of tile row `r`.
fn xs_at(r: u32, o: u32) -> vec4<f32> {
    return xs[r * (BLK_ELEMS / 4u) + (o >> 2u)];
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
        workgroupBarrier();
        stage_x(tid, row0, p0, k4, xb, last_row);
        workgroupBarrier();

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
                let qa = bytes_x4(read_word_at(ql_rel + ql_off + l));
                let qb = bytes_x4(read_word_at(ql_rel + ql_off + l + 32u));
                let hb = bytes_x4(read_word_at(qh_rel + qh_off_h + l));
                let v1 = q6_quarter(sc1, qa, hb, 0u, false);
                let v2 = q6_quarter(sc2, qb, hb, 2u, false);
                let v3 = q6_quarter(sc3, qa, hb, 4u, true);
                let v4 = q6_quarter(sc4, qb, hb, 6u, true);
                // Tile-local element base of the first quarter (a multiple of 4);
                // the quarters sit +32/+64/+96 elements, i.e. +8/+16/+24 vec4.
                let o = row_base + l;
                a0 = a0 + xs_at(0u, o) * v1 + xs_at(0u, o + 32u) * v2 + xs_at(0u, o + 64u) * v3 + xs_at(0u, o + 96u) * v4;
                a1 = a1 + xs_at(1u, o) * v1 + xs_at(1u, o + 32u) * v2 + xs_at(1u, o + 64u) * v3 + xs_at(1u, o + 96u) * v4;
                a2 = a2 + xs_at(2u, o) * v1 + xs_at(2u, o + 32u) * v2 + xs_at(2u, o + 64u) * v3 + xs_at(2u, o + 96u) * v4;
                a3 = a3 + xs_at(3u, o) * v1 + xs_at(3u, o + 32u) * v2 + xs_at(3u, o + 64u) * v3 + xs_at(3u, o + 96u) * v4;
                a4 = a4 + xs_at(4u, o) * v1 + xs_at(4u, o + 32u) * v2 + xs_at(4u, o + 64u) * v3 + xs_at(4u, o + 96u) * v4;
                a5 = a5 + xs_at(5u, o) * v1 + xs_at(5u, o + 32u) * v2 + xs_at(5u, o + 64u) * v3 + xs_at(5u, o + 96u) * v4;
                a6 = a6 + xs_at(6u, o) * v1 + xs_at(6u, o + 32u) * v2 + xs_at(6u, o + 64u) * v3 + xs_at(6u, o + 96u) * v4;
                a7 = a7 + xs_at(7u, o) * v1 + xs_at(7u, o + 32u) * v2 + xs_at(7u, o + 64u) * v3 + xs_at(7u, o + 96u) * v4;
            }
        }
    }

    if (col < params.ncols) { store_col(a0, a1, a2, a3, a4, a5, a6, a7, row0, col); }
}

// Q8_0: 34 bytes / 32 elements — f16 `d` and 32 i8 quants, the byte-per-weight
// form. Its block stride is not a multiple of four, so a block's quants sit two
// bytes off a word boundary for half the (column, block) pairs: neither the
// block's `d` nor its quants can use the whole-word reader the Q4_K entry has
// ([`read_u32`]), and both go through the reassembling [`read_word_at`], as
// Q6_K's 210-byte rows already do.
const Q8_0_BLK_ELEMS: u32 = 32u;
const Q8_0_BLK_BYTES: u32 = 34u;

/// One `qs` word's four quants, as `(b + 128) / 255`: the word is biased by 0x80
/// so the signed bytes land on the unsigned `unorm` unpack, which is exact,
/// where a signed unpack clamps the lowest quant (-128) at -1.0. The 255 and the
/// bias both fold back out in [`dequant_gemm_q8_0`]'s `fma`, so four weights cost
/// one unpack and one `fma` — the same shape as the Q4_K entry's nibble path, and
/// not the byte-at-a-time read with a shift, a mask and a sign fix each that
/// measured 2.1-2.7x here.
fn q8_0_quants(word: u32) -> vec4<f32> {
    return unpack4x8unorm(word ^ 0x80808080u);
}

/// The bias `q8_0_quants` adds, folded into the block's scale together with
/// [`DQ_UNORM`]: `fma(quants, d * 255, -128 * d)` recovers `d * b` for a quant
/// `b`, since `quants` is `(b + 128) / 255`.
const Q8_0_BIAS: f32 = 128.0;

@compute @workgroup_size(64, 1, 1)
fn dequant_gemm_q8_0(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x * TG + tid;
    let row0 = wid.y * TM;
    let colc = min(col, params.ncols - 1u);

    let nbk = params.k / Q8_0_BLK_ELEMS;
    let k4 = params.k >> 2u;
    let last_row = params.m - 1u;
    let xb = params.x_vec4_off;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    // One staged activation tile per `BLK_ELEMS` of `k` — the Q4_K/Q6_K entries'
    // tile, and eight of this scheme's blocks — so the barrier pair falls once per
    // 256 elements instead of once per 32-element block, with the same staging
    // work at the same granularity as the entries above.
    for (var b0: u32 = 0u; b0 < params.k; b0 = b0 + BLK_ELEMS) {
        workgroupBarrier();
        stage_x(tid, row0, b0, k4, xb, last_row);
        workgroupBarrier();

        for (var bi: u32 = 0u; bi < BLK_ELEMS / Q8_0_BLK_ELEMS; bi = bi + 1u) {
            let b = (b0 / Q8_0_BLK_ELEMS) + bi;
            let off = (colc * nbk + b) * Q8_0_BLK_BYTES;
            let d = f16_bits_to_f32(read_word_at(off) & 0xFFFFu);
            let dq = vec4<f32>(d * DQ_UNORM);
            let bias = vec4<f32>(-Q8_0_BIAS * d);
            let lp = bi * Q8_0_BLK_ELEMS;
            for (var l: u32 = 0u; l < Q8_0_BLK_ELEMS; l = l + 4u) {
                let ww = fma(q8_0_quants(read_word_at(off + 2u + l)), dq, bias);
                let o = lp + l;
                a0 = a0 + xs_at(0u, o) * ww;
                a1 = a1 + xs_at(1u, o) * ww;
                a2 = a2 + xs_at(2u, o) * ww;
                a3 = a3 + xs_at(3u, o) * ww;
                a4 = a4 + xs_at(4u, o) * ww;
                a5 = a5 + xs_at(5u, o) * ww;
                a6 = a6 + xs_at(6u, o) * ww;
                a7 = a7 + xs_at(7u, o) * ww;
            }
        }
    }

    if (col < params.ncols) { store_col(a0, a1, a2, a3, a4, a5, a6, a7, row0, col); }
}

