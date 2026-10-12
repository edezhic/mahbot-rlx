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
//   * `dequant_gemv_tiled` (Q4_K, Q6_K): one WORKGROUP per TWO output columns,
//     and that workgroup's 32 lanes split each column's contraction instead of
//     one thread walking the whole row. A lane owns one 32-element *unit* of the
//     row and reads X as `vec4` (8 vector loads per unit, 4 elements each),
//     which is why this arm's X window has to line up with a four-element group,
//     i.e. `x_f32_off` be a multiple of 4; that one X read is then reused for
//     both of the workgroup's columns. A row holds eight of these units per
//     256-element block, the same count the shape this arm replaced walked (one
//     32-element sub-block per lane), so the lane occupancy is unchanged at every
//     contraction length.
//
//     What a unit *is* differs per scheme, because their nibble-to-element maps
//     do:
//       - Q6_K: the 16-byte half of a quarter's 32-byte `ql` span, read at BOTH
//         of that span's nibble sets — byte `t` of the span holds element `t` of
//         quarter `q` in one nibble and of quarter `q ^ 2` in the other, so the
//         unit's 32 elements are 16 from each and the lane requests each of that
//         half's bytes exactly once — where the earlier shape had two lanes read
//         the same span's two nibble sets separately. That is where this arm's
//         halved packed-weight requests come from.
//       - Q4_K: one 32-element sub-block, i.e. one nibble set of its block's
//         32-byte `qs` group. A Q4_K group's byte `t` holds element `t` and
//         element `32 + t`, so a unit built from half of one at both nibbles
//         would take half of one sub-block and half of its sibling, whose 6-bit
//         mins then cancel against partial sums belonging to different scales:
//         that cover overshoots the one-row parity bar the k=256 cell of
//         `tests/gguf_batch_join.rs` asserts (1e-2). The sub-block cover is kept
//         for this scheme. Its header is still four whole-word loads per unit
//         instead of the six or seven one-byte loads it replaced; those touched
//         six or seven *distinct header bytes* per unit, where these four words
//         request the block's whole 16-byte header — once per unit, so eight
//         times per block — inside the 144-byte block that unit's own `qs` words
//         come from. Header bytes requested up, loads down.
//
//     Both schemes' block headers are read from that scheme's own layout: Q4_K's
//     16-byte header (two f16s then 12 scale/min bytes) as four `read_word_at`
//     words parsed with shifts, Q6_K's (an f16 `d` and 16 i8 scales) byte-wise.
//     Reading the packed weight at any alignment is `read_word_at`'s job — the
//     blocks are 144 and 210 bytes, so Q6_K's ql/qh slices are only two-byte
//     aligned.
//
//     The device work a one-row multiply issues is the row's own
//     `columns x contraction`: at the model widths (1024…6144 columns) the
//     one-thread-per-column shape left only 16…96 workgroups, far below what
//     those widths can occupy.
//   * `dequant_gemv` (Q1_0): one thread per output column, as before. The 1-bit
//     scheme's blocks are 128 elements with an 8-element sign byte, a different
//     element/byte mapping than either K-quant, so it keeps this shape.
//
// Lane partials of `dequant_gemv_tiled` are combined by a fixed-order tree per
// column in `red`: deterministic, and no floating-point atomics (absent on some
// adapters).
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
// reads X for a whole 32-element unit with 8 vector loads instead of 32 scalar ones
// — one contiguous span for Q4_K, two 16-element halves 64 elements apart for
// Q6_K (that span's other nibble set, two quarters along) — so `x_f32_off` has to
// be a multiple of 4.
// The Rust side admits a call only when it is (`gemv_one_row_device_ok`) and sends
// the rest to the host rather than reading X shifted. The one-thread-per-column arm
// keeps the scalar view.
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

fn dq_f16_bits(bits: u32) -> f32 {
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
    return dq_f16_bits(read_w(rel) | (read_w(rel + 1u) << 8u));
}

// The (scale, min) pair of one Q4_K sub-block, out of the four whole header
// words the tiled arm loads. After the two f16s (word `h0`, the block's `d` and
// `dmin`), the 16-byte header's 12 packed scale/min bytes are words `h1`, `h2`,
// `h3`: byte `i` of `h1` is scale `i` in its low six bits plus the top two bits
// of scale `i + 4`, byte `i` of `h2` says the same for min `i`, and byte `i` of
// `h3` is scale `i + 4`'s low nibble with min `i + 4`'s in its high one. So the
// three bytes a sub-block `j >= 4` needs are `h3`'s byte `j - 4` and the `h1`/
// `h2` bytes `j - 4`, which is where the shifts below come from. Four whole-word
// loads per sub-block replace the six or seven one-byte loads this arm made
// before. The grouped route's `q4k_scale_min` (`dequant_gemm_gguf.wgsl`) decodes
// the same bytes and has to stay in step with this.
fn dq_q4k_sc_min(h1: u32, h2: u32, h3: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>((h1 >> (8u * j)) & 63u, (h2 >> (8u * j)) & 63u);
    }
    let j4 = j - 4u;
    let lo = h3 >> (8u * j4);
    let sc_hi = (h1 >> (8u * j4)) >> 6u;
    let mn_hi = (h2 >> (8u * j4)) >> 6u;
    return vec2<u32>((lo & 0x0Fu) | ((sc_hi & 3u) << 4u),
                     ((lo >> 4u) & 0x0Fu) | ((mn_hi & 3u) << 4u));
}

// The four weights one Q4_K `qs` word carries for a unit's element set, `i`-th
// first, dotted against that set's four X elements: the word and the set's one
// vector load are consumed together.
fn dq_q4k_span(word: u32, sh: u32, xv: vec4<f32>, d: f32, m: f32) -> f32 {
    return xv.x * (d * f32((word >> sh) & 0x0Fu) - m)
         + xv.y * (d * f32((word >> (8u + sh)) & 0x0Fu) - m)
         + xv.z * (d * f32((word >> (16u + sh)) & 0x0Fu) - m)
         + xv.w * (d * f32((word >> (24u + sh)) & 0x0Fu) - m);
}

// The same for one Q6_K element set: `ql`/`qh` word pair, four low bits from
// `ql`, two high bits from `qh`, biased by 32, times that set's scale.
fn dq_q6k_span(qlw: u32, qhw: u32, sh: u32, qbits: u32, xv: vec4<f32>, scl: f32) -> f32 {
    let q0 = f32(i32(((qlw >> sh) & 0x0Fu) | (((qhw >> qbits) & 3u) << 4u)) - 32);
    let q1 = f32(i32(((qlw >> (8u + sh)) & 0x0Fu) | (((qhw >> (8u + qbits)) & 3u) << 4u)) - 32);
    let q2 = f32(i32(((qlw >> (16u + sh)) & 0x0Fu) | (((qhw >> (16u + qbits)) & 3u) << 4u)) - 32);
    let q3 = f32(i32(((qlw >> (24u + sh)) & 0x0Fu) | (((qhw >> (24u + qbits)) & 3u) << 4u)) - 32);
    return (xv.x * q0 + xv.y * q1 + xv.z * q2 + xv.w * q3) * scl;
}

// Packed bytes one 256-element block of each K-quant occupies — what each branch
// offsets its second column's block by.
const Q4K_BLOCK_BYTES: u32 = 144u;
const Q6K_BLOCK_BYTES: u32 = 210u;

// Threads one workgroup of `dequant_gemv_tiled` runs: its `@workgroup_size`, and
// the stride between one output column's reduction slots in `red` and the next
// column's. The Rust side's `GEMV_TILED_THREADS`.
const LANES: u32 = 32u;
// Output columns one workgroup of that entry covers — the Rust side's
// `GEMV_TILED_COLS_PER_WG`, which sizes the dispatch and the shape it records,
// and the count of slot sets in `red`.
const COLS_PER_WG: u32 = 2u;
// Such a workgroup's reduction storage: one f32 per lane per output column it
// covers, `LANES * COLS_PER_WG * 4` = 256 B — inside the 16 KiB workgroup-storage
// floor every WebGPU backend honours, at or under 256 invocations per workgroup,
// no optional device feature (no floating-point atomics, no subgroup operation,
// no assumption about subgroup width): portable shading rather than an
// adapter-specific shape. All three of these numbers are tied to the Rust
// constants by `gguf_gpu::tests::tiled_gemv_geometry_matches_the_shader`.
var<workgroup> red: array<f32, LANES * COLS_PER_WG>;

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

// One workgroup per two output columns; its 32 lanes split each column's
// contraction. A lane owns one 32-element **unit** of the row and reads X as
// eight `vec4<f32>` loads; both of the workgroup's columns are then given their
// weights' worth of work out of that one X read, so the X side is shared rather
// than re-requested per column. The row holds eight units per 256-element block
// under either shape, so the lane occupancy is unchanged at every contraction
// length. What a unit is differs per scheme — see the file header; Q6_K's is the
// one that also halves the packed-weight requests.
@compute @workgroup_size(32)
fn dequant_gemv_tiled(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j0 = wid.x * COLS_PER_WG;
    if (j0 >= params.n) { return; }
    let second = (j0 + 1u) < params.n;
    // Both schemes' blocks hold 256 elements — eight of this arm's 32-element
    // units — so only their packed size differs.
    let nbk = params.k / 256u;        // blocks per weight row
    // Units per weight row: eight per block, over the row's own block count, so the
    // bound stays inside the row's packed bytes — a `k` that is not a whole number
    // of blocks drops its tail, as `nbk` already floors it, instead of reading
    // block `nbk`, the next column's bytes.
    let nu = nbk * 8u;
    let lane = lid.x;
    let xb4 = params.x_f32_off / 4u;  // X as vec4 groups (see `xarr4`)
    var acc0 = 0.0;
    var acc1 = 0.0;

    if (params.scheme_id == 0u) {
        // Q4_K: 144 bytes / 256 elems. Unit `u` is sub-block `sb = u%8` of block
        // `u/8`: its 32 elements are one nibble set of byte `l` of its block's
        // 32-byte group `sb/2` — the low nibble for even `sb`, the high one for
        // odd — read as eight words. The block's header (`d`/`dmin` as two f16s,
        // then 12 scale/min bytes) is its first four words.
        for (var u = lane; u < nu; u = u + LANES) {
            let off0 = (j0 * nbk + u / 8u) * Q4K_BLOCK_BYTES;
            let sb = u % 8u;
            let h0 = read_word_at(off0);
            let h1 = read_word_at(off0 + 4u);
            let h2 = read_word_at(off0 + 8u);
            let h3 = read_word_at(off0 + 12u);
            let sm0 = dq_q4k_sc_min(h1, h2, h3, sb);
            let d0 = dq_f16_bits(h0 & 0xFFFFu) * f32(sm0.x);
            let m0 = dq_f16_bits(h0 >> 16u) * f32(sm0.y);
            let qs0 = off0 + 16u + 32u * (sb / 2u);
            let e4 = u * 8u;
            let sh = select(0u, 4u, (sb % 2u) == 1u);
            var qs1 = 0u;
            var d1 = 0.0;
            var m1 = 0.0;
            if (second) {
                let off1 = off0 + nbk * Q4K_BLOCK_BYTES;
                let g0 = read_word_at(off1);
                let g1 = read_word_at(off1 + 4u);
                let g2 = read_word_at(off1 + 8u);
                let g3 = read_word_at(off1 + 12u);
                let sm1 = dq_q4k_sc_min(g1, g2, g3, sb);
                d1 = dq_f16_bits(g0 & 0xFFFFu) * f32(sm1.x);
                m1 = dq_f16_bits(g0 >> 16u) * f32(sm1.y);
                qs1 = off1 + 16u + 32u * (sb / 2u);
            }
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let xv = xarr4[xb4 + e4 + w];
                let word0 = read_word_at(qs0 + 4u * w);
                acc0 = acc0 + dq_q4k_span(word0, sh, xv, d0, m0);
                if (second) {
                    let word1 = read_word_at(qs1 + 4u * w);
                    acc1 = acc1 + dq_q4k_span(word1, sh, xv, d1, m1);
                }
            }
        }
    } else if (params.scheme_id == 2u) {
        // Q6_K: 210 bytes / 256 elems. Unit `u` is the 16-byte half `rh` of a
        // quarter's `ql` span, within half `h`: bytes [16rh, 16rh+16) of it carry
        // element `16rh + j` of quarter `q` in one nibble and of quarter `q ^ 2`
        // in the other, and both sets' two-bit fields live in the same four `qh`
        // words (picked by the element index within the quarter, the same for
        // both sets). `rh` also picks which of a quarter's two scales covers
        // those 16 elements.
        for (var u = lane; u < nu; u = u + LANES) {
            let b = u / 8u;
            let ub = u % 8u;              // unit within the block
            let h = ub / 4u;              // its 128-element half
            let q = ub % 4u;              // its quarter of that half
            let rh = q / 2u;              // its 16-byte half of that quarter
            let q2 = q ^ 2u;
            let off0 = (j0 * nbk + b) * Q6K_BLOCK_BYTES;
            let qlb0 = off0 + h * 64u + (q % 2u) * 32u + 16u * rh;
            let qhb0 = off0 + 128u + h * 32u + 16u * rh;
            let d0 = dq_read_f16(off0 + 208u);
            let scb0 = off0 + 192u + h * 8u;
            let slo0 = select(d0 * f32(read_w_i8(scb0 + 2u * q)),
                              d0 * f32(read_w_i8(scb0 + 2u * q + 1u)), rh == 1u);
            let shi0 = select(d0 * f32(read_w_i8(scb0 + 2u * q2)),
                              d0 * f32(read_w_i8(scb0 + 2u * q2 + 1u)), rh == 1u);
            let shlo = select(0u, 4u, q >= 2u);
            let shhi = select(0u, 4u, q2 >= 2u);
            let e4 = b * 64u + h * 32u + 4u * rh;
            let xlo4 = e4 + 8u * q;
            let xhi4 = e4 + 8u * q2;
            var qlb1 = 0u;
            var qhb1 = 0u;
            var slo1 = 0.0;
            var shi1 = 0.0;
            if (second) {
                let off1 = off0 + nbk * Q6K_BLOCK_BYTES;
                let d1 = dq_read_f16(off1 + 208u);
                let scb1 = off1 + 192u + h * 8u;
                qlb1 = off1 + h * 64u + (q % 2u) * 32u + 16u * rh;
                qhb1 = off1 + 128u + h * 32u + 16u * rh;
                slo1 = select(d1 * f32(read_w_i8(scb1 + 2u * q)),
                              d1 * f32(read_w_i8(scb1 + 2u * q + 1u)), rh == 1u);
                shi1 = select(d1 * f32(read_w_i8(scb1 + 2u * q2)),
                              d1 * f32(read_w_i8(scb1 + 2u * q2 + 1u)), rh == 1u);
            }
            for (var w: u32 = 0u; w < 4u; w = w + 1u) {
                let qlw0 = read_word_at(qlb0 + 4u * w);
                let qhw0 = read_word_at(qhb0 + 4u * w);
                let xlo = xarr4[xb4 + xlo4 + w];
                let xhi = xarr4[xb4 + xhi4 + w];
                acc0 = acc0 + dq_q6k_span(qlw0, qhw0, shlo, 2u * q, xlo, slo0)
                            + dq_q6k_span(qlw0, qhw0, shhi, 2u * q2, xhi, shi0);
                if (second) {
                    let qlw1 = read_word_at(qlb1 + 4u * w);
                    let qhw1 = read_word_at(qhb1 + 4u * w);
                    acc1 = acc1 + dq_q6k_span(qlw1, qhw1, shlo, 2u * q, xlo, slo1)
                                + dq_q6k_span(qlw1, qhw1, shhi, 2u * q2, xhi, shi1);
                }
            }
        }
    }

    // Fixed-order tree over the lane partials, one tree per column: the same
    // order on every lane and every run, so the sum is deterministic and needs
    // no f32 atomics. Column `c`'s slots are `c * LANES ..`, so the second tree's
    // every step moves by one `LANES`-wide block.
    red[lane] = acc0;
    red[LANES + lane] = acc1;
    workgroupBarrier();
    if (lane < 16u) {
        red[lane] = red[lane] + red[lane + 16u];
        red[LANES + lane] = red[LANES + lane] + red[LANES + lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        red[lane] = red[lane] + red[lane + 8u];
        red[LANES + lane] = red[LANES + lane] + red[LANES + lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        red[lane] = red[lane] + red[lane + 4u];
        red[LANES + lane] = red[LANES + lane] + red[LANES + lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        red[lane] = red[lane] + red[lane + 2u];
        red[LANES + lane] = red[LANES + lane] + red[LANES + lane + 2u];
    }
    workgroupBarrier();
    if (lane == 0u) {
        outarr[params.out_f32_off + j0] = red[0u] + red[1u];
        // The width edge: an odd (or one-column) chunk has no second column to
        // store, and its `acc1` is zero because its body never ran.
        if (second) { outarr[params.out_f32_off + j0 + 1u] = red[LANES] + red[LANES + 1u]; }
    }
}
