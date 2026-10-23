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
//   * `dequant_gemv_tiled` (Q4_K, Q6_K, Q8_0): one WORKGROUP per `COLS_PER_WG`
//     output columns, and that workgroup's `LANES` threads split each column's
//     contraction instead of one thread walking the whole row. A lane owns one
//     32-element *unit* of the row and reads X as `vec4` (8 vector loads per unit,
//     4 elements each), which is why this arm's X window has to line up with a
//     four-element group, i.e. `x_f32_off` be a multiple of 4; that one X read is
//     then reused for both of the workgroup's columns. A row holds eight of these
//     units per 256-element block, the same count the shape this arm replaced
//     walked (one 32-element sub-block per lane), so the lane occupancy is
//     unchanged at every contraction length.
//
//     What a unit *is* differs per scheme, because their nibble-to-element maps
//     do:
//       - Q8_0: one whole block — its 32 i8 quants and the f16 `d` at their head
//         — so a row holds `k / 32` units rather than the K-quants' eight per
//         256-element block, and this branch counts them itself. Its 34-byte
//         block stride puts every odd block's `d` and quants two bytes off a word
//         boundary, which `read_w16`'s two-group assembly covers.
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
//         for this scheme. Its header is one 16-byte leg per unit (`read_w16`)
//         instead of the six or seven one-byte loads it replaced: the block is
//         144 bytes, so that leg sits on a group boundary and requests the whole
//         header — once per unit, eight times per block, header bytes requested
//         up and loads down.
//
//     Each scheme's block header is read from that scheme's own layout: Q4_K's
//     16-byte header (two f16s then 12 scale/min bytes) as the four words of one
//     16-byte leg parsed with shifts, Q6_K's (an f16 `d` and 16 i8 scales) and
//     Q8_0's (an f16 `d` and its quants) as legs whose bytes are picked, with `d`
//     read exactly at its two. Every leg is read 16 bytes at a time (`read_w16`) and
//     assembled across two groups where its offset is not a group boundary — the
//     blocks are 144, 210 and 34 bytes, so Q6_K's and Q8_0's legs land at two.
//
//     The device work a one-row multiply issues is the row's own
//     `columns x contraction`: at the model widths (1024…6144 columns) the
//     one-thread-per-column shape left only 16…96 workgroups, far below what
//     those widths can occupy.
//   * `dequant_gemv` (Q1_0): one thread per output column, as before. The 1-bit
//     scheme's blocks are 128 elements with an 8-element sign byte, a different
//     element/byte mapping than either K-quant, so it keeps this shape.
//   * `dequant_gemv_wide` (Q4_K, Q6_K, Q8_0): the same 32-element unit and the
//     same per-scheme decode as `dequant_gemv_tiled` — except a Q4_K row that
//     gives a lane two or more `qs` groups, which hand it one whole group a step
//     (`dq_q4k_group`) — with one **subgroup** per output column (128 invocations
//     per workgroup, four subgroups at a probed width of 32) and the lane partials
//     combined by `subgroupAdd`. Below the
//     marker at the end of this file, because a module that mentions a subgroup
//     built-in anywhere is refused by wgpu on a device without
//     `Features::SUBGROUP` — and the two entry points above must still be
//     creatable there.
//
// Lane partials of `dequant_gemv_tiled` are combined by a fixed-order tree per
// column in `red`: deterministic, and no floating-point atomics (absent on some
// adapters).
//
// # The subgroup arm, and why its width is probed
//
// The wide arm maps one output column onto one subgroup, so the shader has to
// know how wide a subgroup is in order to say which column an invocation
// belongs to and how far a lane strides through the row. `AdapterInfo` only
// buckets that number (`subgroup_min_size`/`max_size`; Metal reports 4..64), so
// the value is measured once per process by a probe workgroup
// (`GGUF_SUBGROUP_PROBE_WGSL` in `kernels/mod.rs`, cached by
// `gguf_gpu::gemv_subgroup_probe`) and handed to this entry as
// `@builtin(subgroup_size)`; the host reads the same probe to size the grid with
// `128 / sg` columns per workgroup. A width that does not divide 128, a device
// without `Features::SUBGROUP`, or `RLX_WGPU_GEMV_WIDE=0|shared` keeps the
// shared-memory arm above — the fallback is `dequant_gemv_tiled` itself, not a
// third entry point: both arms call the same `dq_*_unit` decode, so what differs
// between them is the reduction alone.
//
// On this machine (Apple M3 Pro, wgpu over Metal, 2026-10-21) the probe reads
// subgroup_size=32 with four subgroups in a 128-invocation workgroup, so a wide
// workgroup covers four
// output columns with 32 lanes each — half the workgroups and twice the lanes
// per column the tiled arm issues, and one X read per column instead of one per
// two. Measured through `tests/gguf_one_row_stream_rate.rs` in one session, five
// tiled runs against four wide runs interleaved, min GB/s (median of the runs'
// own minima; the per-form spread within a run is 1-30 %, so a ±5 % cell is not
// a difference):
//
// | form (scheme, k, n)                     | tiled | wide  |
// |-----------------------------------------|-------|-------|
// | 8-bit predictor down (Q8_0 1024×1024)   | 40.8  | 45.3  |
// | 8-bit predictor head (Q8_0 1024×2048)   | 57.6  | 61.1  |
// | 8-bit predictor gate+up (Q8_0 1024×6144)| 81.6  | 83.3  |
// | 8-bit predictor block (Q8_0 2048×2048)  | 74.7  | 78.3  |
// | 4-bit predictor block (Q4_K 2048×2048)  | 56.0  | 57.1  |
// | 4-bit FFN gate (Q4_K 2048×6144)         | 71.1  | 70.0  |
// | 6-bit FFN down (Q6_K 6144×2048)         | 71.4  | 73.5  |
// | 6-bit recognition head (Q6_K 2048×151936)| 82.7 | 86.0  |
// | 4-bit predictor, n=8192                 | 74.9  | 72.0  |
// | 4-bit FFN gate, n=49152                 | 83.7  | 80.3  |
// | 8-bit predictor, n=49152                | 96.3  | 100.3 |
//
// So: ahead on every Q8_0 cell the code predictor issues (+2…+11 %, the narrowest
// one most), ahead on the Q6_K cells, level on the Q4_K predictor block and the
// Q4_K FFN gate, and behind at the two Q4_K widths no level of the model issues
// (−4…−5 %). The cells this model runs are the Q8_0 and Q4_K narrow ones, and a
// later paired cell A/B put the whole generation loop 2.3 % faster on this arm
// (20 020.8 against 20 502.1 ms, predictor 38.98 against 40.39 ms a level), so it
// is the default and `RLX_WGPU_GEMV_WIDE=shared` restores the tiled arm. What the
// table does not establish: device time (the figure is wall time around one
// submission and one readback), the cause of the Q4_K wide-n losses, or any
// adapter but the one measured — the width is probed per device precisely because
// this one's 32 is not the next one's.
//
// Binding scheme: each arena window is bound **once** — X at binding 0, the
// weight at binding 2, the output at binding 3. Windowing is what keeps each
// binding under `max_storage_buffer_binding_size` (4 GiB) on multi-GiB models,
// where the whole-arena `as_entire_binding` path does not fit. All three windows
// are declared read_write, including the two the entry points only read: the
// one-row arm can bind them out of one buffer, whose windows are disjoint, and
// the planner keeps the input live until the store. The previous layout bound X
// and the weight twice (a scalar and a `vec4` view of each), so two writable
// bindings shared a range; that only worked on a driver that does not enforce
// WebGPU's rule against it, and a conforming one rejects the bind group.
//
// Block element order is identity (`out_i` == k-offset within the block), so
// `x[k_base + out_i]`.

struct Params {
    k: u32,            // contraction dim (a whole number of the scheme's blocks)
    n: u32,            // output columns (weight rows)
    scheme_id: u32,    // 0 = Q4_K, 2 = Q6_K, 20 = Q8_0, 24 = Q1_0
    x_f32_off: u32,    // f32 index of x[0] within the x binding
    w_byte_off: u32,   // byte offset of weight[0,0] within the weight binding
    out_f32_off: u32,  // f32 index of y[0] within the output window
    // Padding to the 32 bytes a WGSL uniform struct of eight words occupies (its
    // alignment is 16), so the record written and the record read agree.
    _p0: u32,
    _p1: u32,
};

// Every arena window is read_write even where the entry point only reads it:
// the one-row arm can bind X, output and weight out of one buffer, whose windows
// are disjoint (see the file header's binding note). Each window is bound once.
//
// The X window, four elements at a time: the tiled arm reads X for a whole
// 32-element unit with 8 vector loads instead of 32 scalar ones — one
// contiguous span for Q4_K and Q8_0, two 16-element halves 64 elements apart
// for Q6_K (that span's other nibble set, two quarters along) — so `x_f32_off`
// has to be a multiple of 4. The legacy Q1_0 arm reaches the same view
// component-wise (`xarr4[i >> 2u][i & 3u]`), so the gate covers it too. The Rust
// side admits a call only when the window is 16-byte aligned
// (`gemv_one_row_device_ok`) and sends the rest to the host rather than reading
// X shifted.
@group(0) @binding(0) var<storage, read_write>  xarr4: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>              params: Params;
// The packed weight window, 16 bytes at a time. Each scheme's decode reads a
// block as a few contiguous 16-byte legs — Q4_K's header and its `qs` half, Q6_K's
// `ql`/`qh` halves and its scale row, Q8_0's `d`-plus-quants — so one `vec4<u32>`
// fetch carries what four word fetches carried before. The block strides put those
// legs at a 4-byte (Q4_K's 144) or 2-byte (Q6_K's 210, Q8_0's 34) offset, so a leg
// that does not start on a group boundary is assembled from the two groups it
// straddles (`read_w16`), and byte-granular reads pick a byte out of their group.
@group(0) @binding(2) var<storage, read_write>  warr4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;


// The weight window's byte offset, set once at the top of every entry from
// `params.w_byte_off`. The read helpers below index `warr4` through it.
var<private> g_w_off: u32;

// Component `n` of the 16-byte group pair `(a, b)` — one group's four words, then
// the next group's — so `n` 0…7 walks two consecutive groups.
fn wcomp(a: vec4<u32>, b: vec4<u32>, n: u32) -> u32 {
    if (n >= 4u) { return b[n - 4u]; }
    return a[n];
}

// The 16 packed-weight bytes at `rel` as four words. A leg on a group boundary is
// the group itself; otherwise the two groups it straddles are fetched once each and
// the words assembled from them — one 16-byte fetch per group instead of one 4-byte
// fetch per word.
//
// `g_w_off + rel` must be **even**: an odd leg would take the 2-aligned branch below
// and shift every word by 16 bits. Every call site holds it — the block strides are
// even (144, 210, 34) and so is every offset added to them — and so is `g_w_off`,
// which is a row offset minus the storage-aligned base below it. That parity is part
// of the contract, not something the picker repairs.
fn read_w16(rel: u32) -> vec4<u32> {
    let abs = g_w_off + rel;
    let a = warr4[abs >> 4u];
    if ((abs & 15u) == 0u) { return a; }
    let b = warr4[(abs >> 4u) + 1u];
    let k = (abs >> 2u) & 3u;
    // A 4-aligned leg keeps each word whole; a 2-aligned one (Q6_K's and Q8_0's
    // odd blocks) takes each word's halves from two adjacent groups.
    if ((abs & 3u) == 0u) {
        return vec4<u32>(wcomp(a, b, k), wcomp(a, b, k + 1u),
                         wcomp(a, b, k + 2u), wcomp(a, b, k + 3u));
    }
    return vec4<u32>((wcomp(a, b, k) >> 16u) | (wcomp(a, b, k + 1u) << 16u),
                     (wcomp(a, b, k + 1u) >> 16u) | (wcomp(a, b, k + 2u) << 16u),
                     (wcomp(a, b, k + 2u) >> 16u) | (wcomp(a, b, k + 3u) << 16u),
                     (wcomp(a, b, k + 3u) >> 16u) | (wcomp(a, b, k + 4u) << 16u));
}

fn read_w(rel: u32) -> u32 {
    let abs = g_w_off + rel;
    return (warr4[abs >> 4u][(abs >> 2u) & 3u] >> ((abs & 3u) * 8u)) & 0xffu;
}

// Byte `i` (0…15) of one `read_w16` leg, sign-extended: Q6_K's block carries its
// 16 i8 scales as exactly that one leg. Q4_K's 12 scale/min bytes never pass
// through here — `dq_q4k_sc_min` parses them out of the header words with shifts.
fn wbyte_i8(v: vec4<u32>, i: u32) -> i32 {
    let b = (v[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
    return select(i32(b), i32(b) - 256, b >= 128u);
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

// The (scale, min) pair of one Q4_K sub-block, out of the block header's one 16-byte
// leg. After the two f16s (word `h0`, the block's `d` and
// `dmin`), the 16-byte header's 12 packed scale/min bytes are words `h1`, `h2`,
// `h3`: byte `i` of `h1` is scale `i` in its low six bits plus the top two bits
// of scale `i + 4`, byte `i` of `h2` says the same for min `i`, and byte `i` of
// `h3` is scale `i + 4`'s low nibble with min `i + 4`'s in its high one. So the
// three bytes a sub-block `j >= 4` needs are `h3`'s byte `j - 4` and the `h1`/
// `h2` bytes `j - 4`, which is where the shifts below come from. The leg replaces
// the six or seven one-byte loads this arm made before. The grouped route's
// `q4k_scale_min` (`dequant_gemm_gguf.wgsl`) decodes
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
// `ql`, two high bits from `qh`, biased by 32, times that set's scale. The four
// elements' fields sit in the same byte of both words, so each word is masked
// once and the two halves merged into one word before the four are unpacked.
fn dq_q6k_span(qlw: u32, qhw: u32, sh: u32, qbits: u32, xv: vec4<f32>, scl: f32) -> f32 {
    let q = ((qlw >> sh) & 0x0F0F0F0Fu) | (((qhw >> qbits) & 0x03030303u) << 4u);
    let q0 = f32(i32(q & 0xFFu) - 32);
    let q1 = f32(i32((q >> 8u) & 0xFFu) - 32);
    let q2 = f32(i32((q >> 16u) & 0xFFu) - 32);
    let q3 = f32(i32(q >> 24u) - 32);
    return (xv.x * q0 + xv.y * q1 + xv.z * q2 + xv.w * q3) * scl;
}

// The same for one Q8_0 element set: the four signed quants of the `qs` word at
// `w`, scaled by the block's `d` and multiplied against that set's four X
// elements. The word is biased by 0x80 and read through `unpack4x8unorm` — exact,
// where a signed unpack clamps the lowest quant (-128) at -1.0 — and the 255 and
// the bias fold back out in one `fma`, the trick the grouped Q8_0 entry's
// `q8_0_quants` uses (`dequant_gemm_gguf.wgsl`).
fn dq_q8_0_span(word: u32, xv: vec4<f32>, d: f32) -> f32 {
    let q = fma(unpack4x8unorm(word ^ 0x80808080u), vec4<f32>(d * 255.0), vec4<f32>(-128.0 * d));
    return xv.x * q.x + xv.y * q.y + xv.z * q.z + xv.w * q.w;
}

// Packed bytes one block of each scheme occupies — what each branch offsets its
// second column's block by: a 256-element block for the K-quants, Q8_0's own
// 32-element one.
const Q4_K_BLK_BYTES: u32 = 144u;
const Q6_K_BLK_BYTES: u32 = 210u;
const Q8_0_BLK_BYTES: u32 = 34u;

// One 32-element unit of one or two output columns' packed rows, decoded and
// dotted against that unit's X elements: `u` is the unit's index within the row
// (each arm strides it by its own width), `nbk` the row's 256-element block
// count, `xb4` X's offset in `vec4` groups, and
// `second` whether the caller's workgroup also serves column `j0 + 1` — the tiled
// arm's two columns and the subgroup arm's ILP entry both do, and their two
// columns' independent weight words are requested together rather than behind a
// per-unit branch; the subgroup arm's one-column entry does not. `x` is the first
// column's share, `y` the second's (zero when `second` is false). Both one-row
// arms call it, and Q4_K's cover below reads the same header through the same
// `dq_q4k_sc_min`, so a scheme's block addressing is one body.
fn dq_q4k_unit(j0: u32, u: u32, nbk: u32, xb4: u32, second: bool) -> vec2<f32> {
    // Sub-block `sb = u % 8` of block `u / 8`: its 32 elements are one nibble set
    // of byte `l` of the block's 32-byte `qs` group `sb / 2` — the low nibble for
    // even `sb`, the high one for odd — one 16-byte leg each. The block's header
    // (`d`/`dmin` as two f16s, then 12 scale/min bytes) is its first 16 bytes, so
    // it is one leg too; a 144-byte block stride keeps both on group boundaries.
    let off0 = (j0 * nbk + u / 8u) * Q4_K_BLK_BYTES;
    let sb = u % 8u;
    let h = read_w16(off0);
    let sm0 = dq_q4k_sc_min(h.y, h.z, h.w, sb);
    let d0 = dq_f16_bits(h.x & 0xFFFFu) * f32(sm0.x);
    let m0 = dq_f16_bits(h.x >> 16u) * f32(sm0.y);
    let qs0 = off0 + 16u + 32u * (sb / 2u);
    let q0a = read_w16(qs0);
    let q0b = read_w16(qs0 + 16u);
    let e4 = u * 8u;
    let sh = select(0u, 4u, (sb % 2u) == 1u);
    var d1 = 0.0;
    var m1 = 0.0;
    var q1a = vec4<u32>(0u);
    var q1b = vec4<u32>(0u);
    if (second) {
        let off1 = off0 + nbk * Q4_K_BLK_BYTES;
        let g = read_w16(off1);
        let sm1 = dq_q4k_sc_min(g.y, g.z, g.w, sb);
        d1 = dq_f16_bits(g.x & 0xFFFFu) * f32(sm1.x);
        m1 = dq_f16_bits(g.x >> 16u) * f32(sm1.y);
        let qs1 = off1 + 16u + 32u * (sb / 2u);
        q1a = read_w16(qs1);
        q1b = read_w16(qs1 + 16u);
    }
    var acc0 = 0.0;
    var acc1 = 0.0;
    // The two columns' words are independent legs, so the pair is issued together
    // rather than behind a per-iteration `if`: the branch's own predication would
    // sit between the two requests. The two legs of one `qs` group are walked in
    // order, so the accumulation is the word-at-a-time one.
    if (second) {
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xv = xarr4[xb4 + e4 + w];
            acc0 = acc0 + dq_q4k_span(q0a[w], sh, xv, d0, m0);
            acc1 = acc1 + dq_q4k_span(q1a[w], sh, xv, d1, m1);
        }
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xv = xarr4[xb4 + e4 + 4u + w];
            acc0 = acc0 + dq_q4k_span(q0b[w], sh, xv, d0, m0);
            acc1 = acc1 + dq_q4k_span(q1b[w], sh, xv, d1, m1);
        }
    } else {
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xv = xarr4[xb4 + e4 + w];
            acc0 = acc0 + dq_q4k_span(q0a[w], sh, xv, d0, m0);
        }
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xv = xarr4[xb4 + e4 + 4u + w];
            acc0 = acc0 + dq_q4k_span(q0b[w], sh, xv, d0, m0);
        }
    }
    return vec2<f32>(acc0, acc1);
}

// One whole Q4_K `qs` group — 64 elements, i.e. two adjacent sub-blocks — of one or
// two output columns' packed rows. `g` indexes the row's groups (`nbk * 4` of them),
// so the group's block is `g / 4` and its place among that block's four groups
// `gi = g % 4`. Byte `t` of a group holds element `t` of sub-block `2 * gi` in its
// low nibble and element `32 + t` of sub-block `2 * gi + 1` in its high one, each
// with its own scale/min pair out of the block's single header — so this form reads
// that header leg and the group's two `qs` legs once for both sub-blocks, where two
// `dq_q4k_unit` calls read the header twice and both legs twice. The even
// sub-block's 32 elements are X `vec4` groups `[16 g, 16 g + 8)` and the odd one's
// the eight after them, the same split that unit pair `2 g` and `2 g + 1` walks.
// One accumulator pair per column, summed at the end into `dq_q4k_unit`'s contract:
// `x` column `j0`, `y` column `j0 + 1`, zero when `second` is false.
fn dq_q4k_group(j0: u32, g: u32, nbk: u32, xb4: u32, second: bool) -> vec2<f32> {
    let off0 = (j0 * nbk + g / 4u) * Q4_K_BLK_BYTES;
    let gi = g % 4u;
    let h = read_w16(off0);
    let dblk = dq_f16_bits(h.x & 0xFFFFu);
    let mblk = dq_f16_bits(h.x >> 16u);
    let sm0 = dq_q4k_sc_min(h.y, h.z, h.w, 2u * gi);
    let sm1 = dq_q4k_sc_min(h.y, h.z, h.w, 2u * gi + 1u);
    let d0 = dblk * f32(sm0.x);
    let m0 = mblk * f32(sm0.y);
    let d0o = dblk * f32(sm1.x);
    let m0o = mblk * f32(sm1.y);
    let qs0 = off0 + 16u + 32u * gi;
    let q0a = read_w16(qs0);
    let q0b = read_w16(qs0 + 16u);
    let e4 = 16u * g;
    var d1 = 0.0;
    var m1 = 0.0;
    var d1o = 0.0;
    var m1o = 0.0;
    var q1a = vec4<u32>(0u);
    var q1b = vec4<u32>(0u);
    if (second) {
        let off1 = off0 + nbk * Q4_K_BLK_BYTES;
        let h1 = read_w16(off1);
        let dblk1 = dq_f16_bits(h1.x & 0xFFFFu);
        let mblk1 = dq_f16_bits(h1.x >> 16u);
        let sm10 = dq_q4k_sc_min(h1.y, h1.z, h1.w, 2u * gi);
        let sm11 = dq_q4k_sc_min(h1.y, h1.z, h1.w, 2u * gi + 1u);
        d1 = dblk1 * f32(sm10.x);
        m1 = mblk1 * f32(sm10.y);
        d1o = dblk1 * f32(sm11.x);
        m1o = mblk1 * f32(sm11.y);
        let qs1 = off1 + 16u + 32u * gi;
        q1a = read_w16(qs1);
        q1b = read_w16(qs1 + 16u);
    }
    var acc0 = 0.0;   // column `j0`, sub-block `2 * gi`
    var acc0o = 0.0;  // column `j0`, sub-block `2 * gi + 1`
    var acc1 = 0.0;
    var acc1o = 0.0;
    // The two sub-blocks' X groups are eight apart, so one weight word's nibble pair
    // is dotted against two different vector loads; both columns share them, and the
    // pair of columns' words are independent legs issued together as in
    // `dq_q4k_unit`.
    if (second) {
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xe = xarr4[xb4 + e4 + w];
            let xo = xarr4[xb4 + e4 + 8u + w];
            acc0 = acc0 + dq_q4k_span(q0a[w], 0u, xe, d0, m0);
            acc0o = acc0o + dq_q4k_span(q0a[w], 4u, xo, d0o, m0o);
            acc1 = acc1 + dq_q4k_span(q1a[w], 0u, xe, d1, m1);
            acc1o = acc1o + dq_q4k_span(q1a[w], 4u, xo, d1o, m1o);
        }
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xe = xarr4[xb4 + e4 + 4u + w];
            let xo = xarr4[xb4 + e4 + 12u + w];
            acc0 = acc0 + dq_q4k_span(q0b[w], 0u, xe, d0, m0);
            acc0o = acc0o + dq_q4k_span(q0b[w], 4u, xo, d0o, m0o);
            acc1 = acc1 + dq_q4k_span(q1b[w], 0u, xe, d1, m1);
            acc1o = acc1o + dq_q4k_span(q1b[w], 4u, xo, d1o, m1o);
        }
    } else {
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xe = xarr4[xb4 + e4 + w];
            let xo = xarr4[xb4 + e4 + 8u + w];
            acc0 = acc0 + dq_q4k_span(q0a[w], 0u, xe, d0, m0);
            acc0o = acc0o + dq_q4k_span(q0a[w], 4u, xo, d0o, m0o);
        }
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let xe = xarr4[xb4 + e4 + 4u + w];
            let xo = xarr4[xb4 + e4 + 12u + w];
            acc0 = acc0 + dq_q4k_span(q0b[w], 0u, xe, d0, m0);
            acc0o = acc0o + dq_q4k_span(q0b[w], 4u, xo, d0o, m0o);
        }
    }
    return vec2<f32>(acc0 + acc0o, acc1 + acc1o);
}

// One column's whole Q4_K row, walked a whole `qs` group a step where the row
// gives a lane two or more of them — `dq_q4k_group` then reads the block's header
// leg and the group's two `qs` legs once for 64 elements, where two sub-block
// steps read them twice — and a sub-block a step below that, where a group a lane
// would already read the same bytes and splitting it over two steps keeps twice as
// much of the row in flight. The row's group count is uniform across the subgroup,
// so the choice is one body for both entries of the arm.
fn dq_q4k_cover(j0: u32, sid: u32, sg: u32, nbk: u32, xb4: u32, second: bool) -> vec2<f32> {
    let ng = nbk * 4u;
    var acc = vec2<f32>(0.0);
    if (ng >= 2u * sg) {
        for (var g = sid; g < ng; g = g + sg) {
            acc = acc + dq_q4k_group(j0, g, nbk, xb4, second);
        }
    } else {
        for (var u = sid; u < nbk * 8u; u = u + sg) {
            acc = acc + dq_q4k_unit(j0, u, nbk, xb4, second);
        }
    }
    return acc;
}

// The same for Q6_K: unit `u` is the 16-byte half `rh` of a quarter's `ql` span,
// within half `h`. Bytes `[16 rh, 16 rh + 16)` of it carry element `16 rh + j` of
// quarter `q` in one nibble and of quarter `q ^ 2` in the other, and both sets'
// two-bit fields live in the same four `qh` words; `rh` also picks which of a
// quarter's two scales covers those 16 elements. The unit's 32 elements are 16
// from each set, and each of that half's bytes is requested once. Both columns
// share the unit's X reads, so the second column costs its weights and nothing of
// the activation.
fn dq_q6k_unit(j0: u32, u: u32, nbk: u32, xb4: u32, second: bool) -> vec2<f32> {
    let b = u / 8u;
    let ub = u % 8u;              // unit within the block
    let h = ub / 4u;              // its 128-element half
    let q = ub % 4u;              // its quarter of that half
    let rh = q / 2u;              // its 16-byte half of that quarter
    let q2 = q ^ 2u;
    let off0 = (j0 * nbk + b) * Q6_K_BLK_BYTES;
    let qlb0 = off0 + h * 64u + (q % 2u) * 32u + 16u * rh;
    let qhb0 = off0 + 128u + h * 32u + 16u * rh;
    // The block's 16 i8 scales are one 16-byte leg, and the f16 `d` after it the
    // two bytes after that leg — read exactly, so no access here leaves the block.
    let sc0 = read_w16(off0 + 192u);
    let d0 = dq_read_f16(off0 + 208u);
    let slo0 = d0 * f32(wbyte_i8(sc0, h * 8u + 2u * q + rh));
    let shi0 = d0 * f32(wbyte_i8(sc0, h * 8u + 2u * q2 + rh));
    let shlo = select(0u, 4u, q >= 2u);
    let shhi = select(0u, 4u, q2 >= 2u);
    let e4 = b * 64u + h * 32u + 4u * rh;
    let xlo4 = e4 + 8u * q;
    let xhi4 = e4 + 8u * q2;
    let ql0 = read_w16(qlb0);
    let qh0 = read_w16(qhb0);
    var slo1 = 0.0;
    var shi1 = 0.0;
    var ql1 = vec4<u32>(0u);
    var qh1 = vec4<u32>(0u);
    if (second) {
        let off1 = off0 + nbk * Q6_K_BLK_BYTES;
        let sc1 = read_w16(off1 + 192u);
        let d1 = dq_read_f16(off1 + 208u);
        ql1 = read_w16(off1 + h * 64u + (q % 2u) * 32u + 16u * rh);
        qh1 = read_w16(off1 + 128u + h * 32u + 16u * rh);
        slo1 = d1 * f32(wbyte_i8(sc1, h * 8u + 2u * q + rh));
        shi1 = d1 * f32(wbyte_i8(sc1, h * 8u + 2u * q2 + rh));
    }
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
        let xlo = xarr4[xb4 + xlo4 + w];
        let xhi = xarr4[xb4 + xhi4 + w];
        acc0 = acc0 + dq_q6k_span(ql0[w], qh0[w], shlo, 2u * q, xlo, slo0)
                    + dq_q6k_span(ql0[w], qh0[w], shhi, 2u * q2, xhi, shi0);
        if (second) {
            acc1 = acc1 + dq_q6k_span(ql1[w], qh1[w], shlo, 2u * q, xlo, slo1)
                        + dq_q6k_span(ql1[w], qh1[w], shhi, 2u * q2, xhi, shi1);
        }
    }
    return vec2<f32>(acc0, acc1);
}

// The same for Q8_0, whose unit IS its block: the f16 `d` at the block's head and
// its 32 i8 quants, so the row holds `q8blk = k / 32` units and the caller counts
// them itself rather than through the K-quants' `nbk * 8`. Its 34-byte stride is
// not a multiple of four, so half the (column, block) pairs put their quants two
// bytes off a word boundary, and both 16-byte quant legs of a block are assembled
// from the two groups they straddle. The second leg reaches up to 14 bytes past the
// block's own 34, which is the over-read the Rust side's weight window covers: it
// is cut on a 16-byte boundary (`gemv_weight_window`), not at the row's last byte.
fn dq_q8_0_unit(j0: u32, u: u32, q8blk: u32, xb4: u32, second: bool) -> vec2<f32> {
    let off0 = (j0 * q8blk + u) * Q8_0_BLK_BYTES;
    let d0 = dq_read_f16(off0);
    // The 32 quants are the two 16-byte legs after `d`, wherever the 34-byte stride
    // puts them.
    let q0a = read_w16(off0 + 2u);
    let q0b = read_w16(off0 + 18u);
    let e4 = u * 8u;
    var d1 = 0.0;
    var q1a = vec4<u32>(0u);
    var q1b = vec4<u32>(0u);
    if (second) {
        let off1 = off0 + q8blk * Q8_0_BLK_BYTES;
        d1 = dq_read_f16(off1);
        q1a = read_w16(off1 + 2u);
        q1b = read_w16(off1 + 18u);
    }
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
        let xv = xarr4[xb4 + e4 + w];
        acc0 = acc0 + dq_q8_0_span(q0a[w], xv, d0);
        if (second) {
            acc1 = acc1 + dq_q8_0_span(q1a[w], xv, d1);
        }
    }
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
        let xv = xarr4[xb4 + e4 + 4u + w];
        acc0 = acc0 + dq_q8_0_span(q0b[w], xv, d0);
        if (second) {
            acc1 = acc1 + dq_q8_0_span(q1b[w], xv, d1);
        }
    }
    return vec2<f32>(acc0, acc1);
}

// Threads one workgroup of `dequant_gemv_tiled` runs: its `@workgroup_size`, and
// the stride between one output column's reduction slots in `red` and the next
// column's. The Rust side's `GEMV_TILED_THREADS`. A lane advances through its
// column's units by `LANES`, so this is also the serial depth one lane walks: a
// row holds `nu = k / 32` units, so `LANES` threads take `nu / LANES` units each,
// and that depth is the concurrency the arm's shape decides.
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
    g_w_off = params.w_byte_off;
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
                    let i = ko + byte * 8u + bit;
                    acc = acc + xarr4[(xb + i) >> 2u][(xb + i) & 3u] * w;
                }
            }
        }
    }

    outarr[params.out_f32_off + j] = acc;
}

// One workgroup per `COLS_PER_WG` output columns; its `LANES` threads split each
// column's contraction. A lane owns one 32-element **unit** of the row and reads X
// as eight `vec4<f32>` loads; both of the workgroup's columns are then given their
// weights' worth of work out of that one X read, so the X side is shared rather
// than re-requested per column. The row holds eight units per 256-element block
// under either shape, so the lane occupancy is unchanged at every contraction
// length. What a unit is differs per scheme — see the file header; Q6_K's is the
// one that also halves the packed-weight requests.
@compute @workgroup_size(LANES)
fn dequant_gemv_tiled(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    g_w_off = params.w_byte_off;
    let j0 = wid.x * COLS_PER_WG;
    if (j0 >= params.n) { return; }
    let second = (j0 + 1u) < params.n;
    // The two K-quants' blocks hold 256 elements — eight of this arm's 32-element
    // units — so only their packed size differs. Q8_0's own block IS that unit,
    // so its branch counts the row's blocks itself.
    let nbk = params.k / 256u;        // blocks per weight row
    // Units per weight row: eight per block, over the row's own block count, so the
    // bound stays inside the row's packed bytes — a `k` that is not a whole number
    // of blocks drops its tail, as `nbk` already floors it, instead of reading
    // block `nbk`, the next column's bytes.
    let nu = nbk * 8u;
    let lane = lid.x;
    let xb4 = params.x_f32_off / 4u;  // X as vec4 groups (see `xarr4`)
    // Every scheme reads its packed blocks in 16-byte legs (`read_w16`): Q4_K's
    // 144-byte stride keeps them on group boundaries, and the two 2-aligned strides
    // (Q6_K's 210, Q8_0's 34) are assembled from the two groups a leg straddles.
    var acc0 = 0.0;
    var acc1 = 0.0;

    if (params.scheme_id == 0u) {
        // Q4_K (144 bytes / 256 elems): one 32-element sub-block per unit, the
        // shared decode in `dq_q4k_unit`.
        for (var u = lane; u < nu; u = u + LANES) {
            let a = dq_q4k_unit(j0, u, nbk, xb4, second);
            acc0 = acc0 + a.x;
            acc1 = acc1 + a.y;
        }
    } else if (params.scheme_id == 2u) {
        // Q6_K (210 bytes / 256 elems): the 16-byte half of a quarter's `ql` span,
        // read at both of that span's nibble sets, both columns off one X read.
        for (var u = lane; u < nu; u = u + LANES) {
            let a = dq_q6k_unit(j0, u, nbk, xb4, second);
            acc0 = acc0 + a.x;
            acc1 = acc1 + a.y;
        }
    } else if (params.scheme_id == 20u) {
        // Q8_0 (34 bytes / 32 elems): unit `u` IS block `u`, so this branch counts
        // the row's own blocks rather than the K-quants' `nu` (which holds the
        // tail of a `k` that is not a whole number of 256-element blocks). The
        // Rust side admits this scheme's single row only over whole blocks, so
        // that tail is never served here.
        let q8blk = params.k / 32u;
        for (var u = lane; u < q8blk; u = u + LANES) {
            let a = dq_q8_0_unit(j0, u, q8blk, xb4, second);
            acc0 = acc0 + a.x;
            acc1 = acc1 + a.y;
        }
    }
    // Fixed-order tree over the lane partials, one tree per column: the same
    // order on every lane and every run, so the sum is deterministic and needs
    // no f32 atomics. Column `c`'s slots are `c * LANES ..`, so the second tree's
    // every step moves by one `LANES`-wide block. Written as a halving loop from
    // `LANES / 2`: at `LANES == 32` it takes exactly the five steps this arm
    // hand-wrote (16, 8, 4, 2, 1), so widening the workgroup does not change the
    // order the width that was taken measures.
    red[lane] = acc0;
    red[LANES + lane] = acc1;
    var step = LANES / 2u;
    loop {
        if (step == 0u) { break; }
        workgroupBarrier();
        if (lane < step) {
            red[lane] = red[lane] + red[lane + step];
            red[LANES + lane] = red[LANES + lane] + red[LANES + lane + step];
        }
        step = step / 2u;
    }
    workgroupBarrier();
    if (lane == 0u) {
        outarr[params.out_f32_off + j0] = red[0u];
        // The width edge: an odd (or one-column) chunk has no second column to
        // store, and its `acc1` is zero because its body never ran.
        if (second) { outarr[params.out_f32_off + j0 + 1u] = red[LANES]; }
    }
}

// >>> dequant_gemv_wide: subgroup-reduction arm (module split here)

// Everything below this line is the subgroup arm, and `DEQUANT_GEMV_GGUF_WGSL`
// in `kernels/mod.rs` stops at this line: the two entry points above are created
// on every device, and wgpu refuses a shader module that names a subgroup
// built-in on a device without `Features::SUBGROUP`. The wide module is this
// whole file, built only after `gguf_gpu::gemv_subgroup_probe` has returned a
// width on a device that carries the feature.
//
// # What this arm is
//
// One subgroup per output column. `@workgroup_size` is 128 (the Rust side's
// `GEMV_WIDE_THREADS`), so at the probed width `sg` a workgroup covers
// `128 / sg` columns — four on this machine, where the tiled arm covers two —
// and every lane of a column's subgroup walks that column's 32-element units by
// `sg` instead of by `LANES`. The per-scheme decode is the tiled arm's:
// `dq_q6k_unit` / `dq_q8_0_unit` above, called with this arm's lane stride, so
// the accumulated products are the same numbers in the same per-lane order; only
// the final combination differs (a `subgroupAdd` tree instead of `red`'s fixed
// five-step tree). Q4_K walks `dq_q4k_cover` — a whole `qs` group, its header and
// both `qs` legs read once — where the row gives a lane two or more groups, and
// `dq_q4k_unit`'s sub-block cover where it does not, so its per-lane accumulation
// order is whichever cover ran. `tests/gguf_gemv_wide_arm.rs` is the cell that
// bounds that difference against the CPU reference.
//
// # What it is not
//
// There is no shared-memory fallback *entry point* here: the fallback is
// `dequant_gemv_tiled` above, reached by the same route when the probe declines
// or the knob says `0`/`shared`, and the two arms share the decode, so this is
// one reduction against another and not a second kernel body.
//
// Threads one workgroup of `dequant_gemv_wide` runs — this entry's
// `@workgroup_size`, the Rust side's `GEMV_WIDE_THREADS`, and the numerator of
// the `columns per workgroup = this / sg` the host sizes the grid with.
// `gguf_gpu::tests::wide_gemv_geometry_matches_the_shader` ties the two
// spellings together.
const WIDE_THREADS: u32 = 128u;

// One subgroup per output column. A lane's index within its column's subgroup is
// `subgroup_invocation_id`, the column is the workgroup's base plus
// `subgroup_id` — both guaranteed builtins with a defined relation to the
// subgroup, where `local_invocation_id / sg` is only what they happen to equal.
@compute @workgroup_size(WIDE_THREADS)
fn dequant_gemv_wide(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    g_w_off = params.w_byte_off;
    let cols = WIDE_THREADS / sg;
    let j = wid.x * cols + sgid;
    // A whole subgroup shares `j`, so the tail column leaves the dispatch with
    // every lane of its subgroup inactive — no lane reaches `subgroupAdd` alone.
    if (j >= params.n) { return; }
    let nbk = params.k / 256u;        // K-quant blocks per weight row
    let nu = nbk * 8u;                // 32-element units per K-quant row
    let xb4 = params.x_f32_off / 4u;  // X as vec4 groups (see `xarr4`)
    // Q4_K's 144-byte stride keeps its words 4-aligned; Q6_K (210 B) and Q8_0
    // (34 B) straddle groups, which `read_w16` assembles, as in the tiled arm.
    var acc = 0.0;

    if (params.scheme_id == 0u) {
        acc = acc + dq_q4k_cover(j, sid, sg, nbk, xb4, false).x;
    } else if (params.scheme_id == 2u) {
        for (var u = sid; u < nu; u = u + sg) {
            acc = acc + dq_q6k_unit(j, u, nbk, xb4, false).x;
        }
    } else if (params.scheme_id == 20u) {
        let q8blk = params.k / 32u;
        for (var u = sid; u < q8blk; u = u + sg) {
            acc = acc + dq_q8_0_unit(j, u, q8blk, xb4, false).x;
        }
    }

    // The reduction: a subgroup tree, stored by the subgroup's first
    // invocation. No `workgroupBarrier` and no workgroup storage — the lanes of
    // a column are its own subgroup, so the tree never leaves it. The order is
    // the hardware's, not the tiled arm's fixed five steps;
    // `tests/gguf_gemv_wide_arm.rs` bounds that difference against the CPU
    // reference instead of asserting identity.
    let total = subgroupAdd(acc);
    if (sid == 0u) { outarr[params.out_f32_off + j] = total; }
}

// Columns per subgroup in `dequant_gemv_wide_ilp` — the host's `GEMV_WIDE_ILP`
// sizes its grid from the same number, and
// `gguf_gpu::tests::wide_gemv_geometry_matches_the_shader` ties the spellings.
const WIDE_ILP: u32 = 2u;

// The one-column-per-subgroup entry above leaves each lane one or two loads of
// a row in flight (`k = 1024` at Q8_0 is one 34-byte block per lane), so its
// rate is memory latency, not bandwidth: 39 GB/s against a 130 GB/s probe.
// Here one subgroup takes `WIDE_ILP` contiguous columns with an independent
// accumulator per column, the pair sharing its X read through the decode's
// `second` argument — two columns' weight legs outstanding at once, and
// `WIDE_THREADS / sg * WIDE_ILP` columns a workgroup. Why the width is two and
// what it measured is `gguf_gpu::GEMV_WIDE_ILP`'s and the route record's.
//
// The host dispatches this entry only for a chunk whose column count is a whole
// number of `WIDE_THREADS / sg * WIDE_ILP`; every other chunk keeps the entry
// above. That arithmetic is not trusted here: like the entry above, this one
// reads the dispatch-time `sg` while the host sizes the grid from the probed
// width, so the tail guard below skips a column the two disagree about instead
// of storing past the chunk's output window.
@compute @workgroup_size(WIDE_THREADS)
fn dequant_gemv_wide_ilp(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    g_w_off = params.w_byte_off;
    let cols = WIDE_THREADS / sg;
    let j = wid.x * (cols * WIDE_ILP) + sgid * WIDE_ILP;
    // A whole subgroup shares `j` and stores all `WIDE_ILP` columns, so a group
    // that starts past the chunk leaves with every lane inactive — no lane
    // reaches `subgroupAdd` alone.
    if (j + WIDE_ILP > params.n) { return; }
    let nbk = params.k / 256u;
    let nu = nbk * 8u;
    let xb4 = params.x_f32_off / 4u;
    var acc = vec2<f32>(0.0);

    if (params.scheme_id == 0u) {
        acc = acc + dq_q4k_cover(j, sid, sg, nbk, xb4, true);
    } else if (params.scheme_id == 2u) {
        for (var u = sid; u < nu; u = u + sg) {
            acc = acc + dq_q6k_unit(j, u, nbk, xb4, true);
        }
    } else if (params.scheme_id == 20u) {
        let q8blk = params.k / 32u;
        for (var u = sid; u < q8blk; u = u + sg) {
            acc = acc + dq_q8_0_unit(j, u, q8blk, xb4, true);
        }
    }

    let total = subgroupAdd(acc);
    if (sid == 0u) {
        outarr[params.out_f32_off + j] = total.x;
        outarr[params.out_f32_off + j + 1u] = total.y;
    }
}
