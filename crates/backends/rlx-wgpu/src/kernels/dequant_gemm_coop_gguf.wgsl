// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused packed-weight GEMM through the hardware matrix units (prefill `m > 1`)
// for `Op::DequantMatMul`: `Y[m,n] = X[m,k] @ dequant(W)[n,k]^T`, the weight
// decoded from its own packed bytes straight into a workgroup f16 stage and
// multiplied by `simdgroup_matrix` / `OpCooperativeMatrixMulAddKHR` with an f32
// accumulator.
//
// One entry point per scheme — `dequant_gemm_coop_q4k` (scheme 0),
// `dequant_gemm_coop_q6k` (scheme 2) and `dequant_gemm_coop_q8_0` (scheme 20) —
// for the reason `dequant_gemm_gguf.wgsl` gives: an id routed to an entry point
// with no branch for it matches nothing and reads the weights as zeros instead of
// failing.
//
// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
//
// `dequant_gemm_gguf.wgsl` accumulates in f32 scalars, one per output row, and is
// limited by that arithmetic: at the row counts the product's prompt prefill
// actually runs (31-111 rows, 95 typical) it measures 0.99-1.51 TFLOP/s on the
// stand's M3 Pro across the three schemes. It is not a memory limit — the same
// measurement shows the weight bytes accounting for at most a third of the call.
// This entry point moves the accumulation onto the matrix units, whose
// f16-operand/f32-accumulator form measures 3.07-3.66 TFLOP/s at m 8192 on the same
// adapter (`matmul_coop_f16_f32acc.wgsl`'s header carries that table), and keeps the
// weight in its packed form: the decode happens per staged tile, in workgroup memory,
// so no f16 copy of the weight is ever materialised in device memory.
//
// ── WHAT IT MEASURES ────────────────────────────────────────────────────────
//
// `tests/gguf_gemm_prefill_rate.rs`, Apple M3 Pro, release, the **pair** — both
// tiles on the same cells in adjacent rounds of one process, the `median_TFLOPs` of
// five rounds of three calls, one take (a re-take moves a rate by a few percent and a
// ratio by up to a tenth) — TFLOP/s at the product's own prefill row counts, this
// entry point against `dequant_gemm_gguf.wgsl`'s row-grouped tile:
//
//   q4k qkv      m 95 k 2048 n  4096    2.24   against 1.42   (1.58x)
//   q4k o        m 95 k 2048 n  2048    1.86   against 1.29   (1.44x)
//   q4k gate/up  m 95 k 2048 n 12288    2.68   against 1.51   (1.77x)
//   q6k down     m 95 k 6144 n  2048    1.74   against 0.99   (1.76x)
//   q8_0 enc     m 96 k 1024 n  4096    1.86   against 1.10   (1.69x)
//   q4k gate/up  m 31/47/63/79/95/111   2.16 / 1.99 / 2.56 / 2.29 / 2.69 / 2.44
//                                        against 1.42 / 1.44 / 1.49 / 1.49 / 1.50 / 1.51
//                                        (1.52x / 1.38x / 1.72x / 1.54x / 1.79x / 1.62x)
//
// The sawtooth in that last row is the 32-row tile's own: 31, 63 and 95 rows are one,
// two and three whole tiles (3 %, 2 % and 1 % of the dispatch spent on rows past `m`),
// while 47 and 79 round up to two and three tiles of a 64- and 96-row dispatch (27 %
// and 18 %), and 111 to four of a 128 (13 %). The m 95 sweep cell is a second
// independent build of the same shape — the harness's repeatability witness, 2.69
// against the table's 2.68. The endpoint is what the floor rests on: 47, the most
// rounded, is the weakest rung at 1.38x, which is what `COOP_GEMM_MAX_PAD_DIVISOR` in
// `gguf_gpu.rs` is the floor of.
//
// The encoder's own rate is the file's other reading, and it is the one that moved
// the floor: `q8_0 enc-down` at m 96 k 4096 n 1024 is eight column tiles by three
// row tiles — 24 workgroups, `COOP_MIN_WORKGROUPS` itself — so it ran the
// row-grouped tile while the floor was 48 and the pair read 1.00x, a repeatability
// witness rather than a comparison. At the floor of 24 it takes the cooperative
// tile and the pair reads 1.21x.
//
// The end-to-end figure the tile buys is the recognition stand row (`STAND.md`'s
// `recognition.typical`): about 0.4 s faster warm with the tile than with
// `RLX_WGPU_GEMM_COOP_DISABLE=1` (3.04 s warm, median of three alternating takes
// each). The prompt prefill is about a quarter of that call, which is why the tile's
// 1.77x on the cell reaches the row as ~12 %.
//
// The row outcome of the header cache is smaller: its +5-6 % on the cell reaches
// recognition long as about 1.2 % by median (0.3 % paired) against a round spread of
// about 5 %. The prompt prefill is 24 % of the long row and the in-shader decode is
// 14 % of the cell, so even removing the whole decode caps at ~3.4 % at the row. The
// cell win is real; the row move is inside the noise.
//
// ── THE TILE ────────────────────────────────────────────────────────────────
//
// 32 rows × 128 columns per 128-thread workgroup, over FOUR 32-wide subgroups that
// each own the whole 32 rows by 32 columns — 4 row sub-tiles by 4 column
// sub-tiles, 16 f32 accumulators per subgroup, one decoded column per thread. The
// load-to-MMA ratio is 8/TM + 8/TN_s = 0.5 per MMA.
//
// Four subgroups rather than two, measured: the same 32×128 tile over two subgroups
// (32 accumulators each, ratio 0.375) is 1.22x slower on the gate/up cell — the
// wider accumulator file does not pay for the warps it costs. What the warps buy is
// latency hiding at the 6 KiB of workgroup storage below, and it is the term this
// form is most sensitive to.
//
// The row tile is 32 and not 64 because the shapes are 22-111 rows: a 64-row tile
// rounds a 95-row prompt up to 128 rows of arithmetic, which is 35 % of the work
// spent on nothing. It is not a constant bump either: a 64-row A stage is two
// ping-pong 64x16 f16 panels, 4 KiB against today's 2 KiB, which puts the tile at
// 8 448 B — past the 8 KiB residency cliff below — so a taller row tile is a
// restructuring of the operand stage, not a loop bound. A 32-row tile rounds it to
// 96 — and the encoder clamps the last tile's row base to `m - 32` rather than
// rounding, so a tile never reaches past the output tensor and the overlap is
// recomputed rather than written. The row tile cannot go below 32: 16 rows per
// subgroup would put the load-to-MMA ratio at 0.75 and halve the arithmetic each
// staged weight tile is reused for.
//
// ── WHAT IT COSTS IN WORKGROUP STORAGE ──────────────────────────────────────
//
// 2 KiB of A stage (two ping-pong 32×16 f16 panels), 4 KiB of B stage (16 k by
// 128 columns of decoded f16) and 256 B of zero lanes the accumulators are
// seeded from: 6 400 B. That is under the 8 KiB at which
// `dequant_gemm_gguf.wgsl` measures its occupancy cliff (a tile merely *declared*
// 10 rows deep, identical arithmetic, read 2.2x its 8 KiB time), so two
// workgroups stay resident per core. Both stages are single-buffered deliberately:
// a second B panel takes the shader to 10.25 KiB for the one barrier per stage it
// would buy, and measured *slower* (17.8 against 16.3 ms on the gate/up cell) —
// the residency is worth more than the barrier. Padding the two strides to break
// the bank period lost for the same reason: on the two-subgroup form this one
// replaced it read 21.5 against 20.0 ms.
//
// ── THE DECODE'S SHARE, AND WHAT A CHEAPER UNPACK BUYS ───────────────────────
//
// A variant of THIS kernel whose decode stores a constant — same tile, barriers and
// stores, no packed reads and no unpack — is the decode's ceiling: at m 95 k 2048
// n 12288 it read **3.11 TFLOP/s against the shipped 2.68**, so the decode is about
// 14 % of the cell and the only lever left at this shape — the tile and its two
// barriers per stage are already inside that 3.11. The distance beyond it to the
// dense form's 3.07-3.66 (a different kernel over a half-precision mirror at m 8192)
// is m and operand, not this tile. The Q6_K `down` cell is dearest (2.53 against
// 1.74, 31 %); the encoder's Q8_0 cells sit at 14-23 %. The variant is not in the
// tree; reproduce it by replacing each `decode_b_*` body's `b_put` argument with
// `vec4(1.0)`, which leaves the tile, the barriers and the stores untouched.
//
// The cheap part is the per-block header, which the row-grouped entry hoists per block
// and this form recomputed per stage before the cache below. Caching it (`hdr_*`,
// keyed on the block's byte offset) is worth **+5-6 % on every Q4_K cell and +3-5 %
// on Q6_K/Q8_0**, bit-exact, and it is what this form ships; the table above is a
// post-cache take. The rest does not yield to a cheaper spelling: a `vec4` Q6_K unpack
// measured 0, issuing the stage's loads a stage ahead measured 0, and caching the
// per-sub-block `(scale, min)` pair — eight more f32 registers — read 2.55 against
// 2.68. The header cache above is not a counterexample: it spends registers to remove
// *global loads*, a per-block cost amortised over the block's stages (sixteen for the
// 256-element blocks, two for Q8_0's 32). Spending registers to remove per-stage
// arithmetic is what loses here — the residue is the decode's register footprint, not
// its instruction count. Reaching the ceiling needs the decode's operands out of
// registers, and no lever tried gets them there: the one obvious place to spend, more
// workgroup memory (a second B panel, above), is measured slower because the extra
// 4 KiB crosses the 8 KiB residency cliff.
//
// ── LAYOUT CONVENTION (read before editing) ─────────────────────────────────
//
// The METAL convention throughout, exactly as `matmul_coop_f16_f32acc.wgsl`
// documents it: `coopLoadT` on A, `coopLoadT` on B, `coopStoreT` on C. The `T`
// suffix reaches Metal's `simdgroup_load(..., transpose_matrix)` and SPIR-V's
// `RowMajorKHR` as different things, so the same WGSL does not mean the same
// thing on the two backends. `src/coop_probe.rs` establishes that this convention
// computes `a·b` on the adapter in play (through the f16/f32-accumulator kernel,
// whose operand layout this one reproduces index for index); a device where it
// does not is refused the route rather than served wrong numbers.
//
// A is `[TM][TK]` f16 row-major with stride TK; B is `[TK][TN]` f16 row-major
// with stride TN, which is the `[k][n]` form the packed weight's own decode
// produces column by column; C is `[m][ncols]` f32 row-major with stride ncols.
//
// ── GEOMETRY ────────────────────────────────────────────────────────────────
//
// The encoder admits a call whose `ncols % 128 == 0` and `k % 16 == 0`, and
// dispatches `(ceil(ncols / 128), ceil(m / 32))`. Whole tiles on both axes are what
// a cooperative store requires — it cannot be masked, so a partial tile would write
// past the output — and the row axis is made whole by clamping the last tile's base
// to `m - 32` instead of rounding the row count up, so the rows two tiles share are
// recomputed identically rather than the output being overrun.
//
// The row count is bounded from below by what that rounding costs: the dispatch
// covers `ceil(m / 32) * 32` rows, and `gguf_gpu.rs`'s `COOP_GEMM_MAX_PAD_DIVISOR`
// refuses a call where more than a third of them are past `m`, so the shortest call
// this tile takes is 22 rows. A call under 32 rows runs one tile whose rows past `m`
// land in the output slab's own surplus — the encoder sizes the slab `max(m, 32)`
// rows and copies only the valid ones out — and whose A panel is staged with its row
// clamped to `m - 1`, so neither axis ever reads or writes outside its own extent.
//
// ── THE READERS ─────────────────────────────────────────────────────────────
//
// `read_w`/`read_u32`/`read_word_at`/`f16_bits_to_f32`/`q4k_scale_min`/
// `q4k_nibbles`/`bytes_x4`/`q6c`/`q6_quarter`/`q8_0_quants` are
// `dequant_gemm_gguf.wgsl`'s, verbatim, so both files decode a block the same
// way: a fix to the nibble, scale or 2-bit layout has five copies to reach
// (`dequant_gguf.wgsl`, `dequant_gemv_gguf.wgsl`, this file, and
// `dequant_gemm_gguf.wgsl` twice — once per arm of its split branch).
//
// Each entry decodes ONE STAGE of 16 elements of `k` per column, which is a
// 16-element slice of one packing block for every scheme here (Q4_K/Q6_K blocks
// are 256 elements, Q8_0's 32), so a stage never straddles a block.

enable f16;
enable wgpu_cooperative_matrix;

/// The uniform is `DequantGemmGgufParams`' layout, shared with the row-grouped
/// entries so one Rust struct encodes both; `split` and the two pad words are
/// unused here.
struct Params {
    m: u32,
    k: u32,
    ncols: u32,
    x_vec4_off: u32,
    w_byte_off: u32,
    split: u32,
    _p0: u32,
    _p1: u32,
}

const TM: u32 = 32u;   // output rows per workgroup
const TN: u32 = 128u;  // output columns per workgroup
const TK: u32 = 16u;   // contraction elements per stage
/// Columns one subgroup owns: its quarter of the tile, and the quarter it decodes,
/// one column per lane of its 32 invocations.
const SG_COLS: u32 = TN / 4u;
/// Blocks of the two K-quant schemes: the granularity their readers stride in.
const K_BLK_ELEMS: u32 = 256u;
/// The two stages' strides, unpadded: a cooperative operand load is one hardware
/// instruction reading the whole 8×8 tile, so the bank-period padding a per-lane
/// load would want buys nothing and costs footprint (see the storage section).
const A_STRIDE: u32 = TK;
const B_STRIDE: u32 = TN;

@group(0) @binding(0) var<storage, read>       xarr: array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read>       warr: array<u32>;
@group(0) @binding(3) var<storage, read_write> outarr: array<f32>;

// A stage: two ping-pong 32×16 f16 panels, 1 KiB each.
var<workgroup> a_stage: array<f16, 2u * TM * A_STRIDE>;
// B stage: 16 k by 128 columns of decoded weight, 4 KiB. Written by the same
// subgroup that reads it (see `b_col`), so only the publication barrier orders it.
var<workgroup> b_stage: array<f16, TK * B_STRIDE>;
// The all-zero 8×8 f32 fragment the accumulators are seeded from: WGSL has no
// zero-valued cooperative-matrix constructor this lowering supports.
var<workgroup> zero_lanes: array<f32, 64u>;

// ── the packed readers, shared with `dequant_gemm_gguf.wgsl` ────────────────

/// One packed byte at `rel`, for a `rel` with no alignment at all.
fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (warr[word] >> shift) & 0xffu;
}

/// Four packed bytes at `rel` as one word, for a `rel` that is a multiple of 4.
fn read_u32(rel: u32) -> u32 {
    return warr[(params.w_byte_off + rel) / 4u];
}

/// Four packed bytes at `rel` as one word, for a `rel` on a 2-byte boundary.
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

/// The f16 whose bits are `bits`, as f32.
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
/// those bytes already read as the three words they are.
fn q4k_scale_min(w0: u32, w1: u32, w2: u32, j: u32) -> vec2<u32> {
    let i = j & 3u;
    let a = (w0 >> (8u * i)) & 0xFFu;
    let b = (w1 >> (8u * i)) & 0xFFu;
    let c = (w2 >> (8u * i)) & 0xFFu;
    if (j < 4u) { return vec2<u32>(a & 63u, b & 63u); }
    return vec2<u32>((c & 0x0Fu) | (((a >> 6u) & 3u) << 4u), (c >> 4u) | (((b >> 6u) & 3u) << 4u));
}

/// The four Q4_K weights one `qs` word carries for a sub-block, low nibble
/// (`sh == 0`) or high nibble (`sh == 4`), as `unpack4x8unorm` gives them.
fn q4k_nibbles(word: u32, sh: u32) -> vec4<f32> {
    return unpack4x8unorm((word >> sh) & 0x0F0F0F0Fu);
}

/// `unpack4x8unorm`'s divisor, folded back into `d`/`dmin` per sub-block.
const DQ_UNORM: f32 = 255.0;

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

/// One 4-byte word as its four bytes, `i`-th first.
fn bytes_x4(w: u32) -> vec4<u32> {
    return vec4<u32>(w & 0xFFu, (w >> 8u) & 0xFFu, (w >> 16u) & 0xFFu, (w >> 24u) & 0xFFu);
}

/// One Q8_0 `qs` word's four quants, as `(b + 128) / 255`.
fn q8_0_quants(word: u32) -> vec4<f32> {
    return unpack4x8unorm(word ^ 0x80808080u);
}

/// The bias [`q8_0_quants`] adds, folded into the block's scale with [`DQ_UNORM`].
const Q8_0_BIAS: f32 = 128.0;

// ── staging ─────────────────────────────────────────────────────────────────

/// Stage one 32×16 A panel, f32 arena → f16, into ping-pong buffer `buf`. Each
/// invocation covers four adjacent K columns of one row, so the load is one
/// `vec4` — the form `matmul_coop_f16_f32acc.wgsl` measures at 2.2 → 3.4 TFLOP/s
/// against one column per invocation — and the workgroup's 128 invocations are
/// exactly `TM * A_STRIDE / 4` such groups. The column is always whole (the
/// encoder admits only whole K stages); the row is clamped to `m - 1` because a
/// call below 32 rows runs one tile over an X window of exactly `m` rows, and
/// only the surplus rows of the output slab are the tile's to write.
fn stage_a(buf: u32, k_off: u32, row_base: u32, lid: u32) {
    let idx = lid * 4u;
    let r = idx / TK;
    let c = idx % TK;
    let row = min(row_base + r, params.m - 1u);
    let v = xarr[params.x_vec4_off + row * (params.k >> 2u) + ((k_off + c) >> 2u)];
    let dst = buf * (TM * A_STRIDE) + r * A_STRIDE + c;
    a_stage[dst] = f16(v.x);
    a_stage[dst + 1u] = f16(v.y);
    a_stage[dst + 2u] = f16(v.z);
    a_stage[dst + 3u] = f16(v.w);
}

/// The column this invocation decodes: its own lane of its subgroup's quarter, so
/// consecutive lanes read consecutive packed rows and write consecutive B slots,
/// and a subgroup never touches another's quarter.
fn b_col(sg: u32, lane: u32) -> u32 {
    return sg * SG_COLS + lane;
}

/// Four decoded weights for `col` at stage offsets `k0 .. k0+4`.
fn b_put(k0: u32, col: u32, v: vec4<f32>) {
    b_stage[(k0 + 0u) * B_STRIDE + col] = f16(v.x);
    b_stage[(k0 + 1u) * B_STRIDE + col] = f16(v.y);
    b_stage[(k0 + 2u) * B_STRIDE + col] = f16(v.z);
    b_stage[(k0 + 3u) * B_STRIDE + col] = f16(v.w);
}

// A block's header, cached across the stages that read it. `off` is constant for
// every stage inside one packing block — 16 of them for the two K-quants, two for
// Q8_0 — and the stage loop below would otherwise re-read and re-parse it per
// stage. The row-grouped entry point already hoists it per block; this carries the
// same fact to the cooperative one, where a stage's decode is between two barriers
// and every instruction in it is on the critical path. Keyed on the byte offset (a
// stage of one block has one offset; a different block changes it), and holding
// only the block-invariant part: Q4_K's `d`, `dmin` and three scale words, Q6_K's
// and Q8_0's `d`. The per-sub-block constants are still derived per stage.
//
// The three decoders share one cache, which is sound only because exactly one
// scheme decodes per dispatch and a thread's `off` names one block of one column:
// an arm that decoded two schemes, or a second weight operand with its own offset
// base, would reuse another scheme's header and return wrong weights rather than
// fail. A second operand needs its own `hdr_*` set.
var<private> hdr_off: u32 = 0xFFFFFFFFu;
var<private> hdr_d: f32;
var<private> hdr_dmin: f32;
var<private> hdr_scw0: u32;
var<private> hdr_scw1: u32;
var<private> hdr_scw2: u32;

/// Q4_K: 144 bytes / 256 elements. The stage covers half of one 32-element
/// sub-block, whose 16 weights are four packed words' low nibbles (an even
/// sub-block) or high nibbles (an odd one).
fn decode_b_q4k(k_off: u32, wcol: u32, bcol: u32) {
    let nbk = params.k / K_BLK_ELEMS;
    let b = k_off / K_BLK_ELEMS;
    let within = k_off % K_BLK_ELEMS;
    let t = within / 32u;
    let half = within % 32u;                 // 0 or 16
    let off = (wcol * nbk + b) * 144u;
    if (hdr_off != off) {
        hdr_off = off;
        let h = read_u32(off);
        hdr_d = f16_bits_to_f32(h & 0xFFFFu);
        hdr_dmin = f16_bits_to_f32(h >> 16u);
        hdr_scw0 = read_u32(off + 4u);
        hdr_scw1 = read_u32(off + 8u);
        hdr_scw2 = read_u32(off + 12u);
    }
    let d = hdr_d;
    let dmin = hdr_dmin;
    let sm = q4k_scale_min(hdr_scw0, hdr_scw1, hdr_scw2, t);
    // `q4k_nibbles` yields `nibble / DQ_UNORM`; folding the divisor into the two
    // per-sub-block constants keeps each weight to one `fma`, as the row-grouped
    // entry's arms do.
    let sc = vec4<f32>(d * f32(sm.x) * DQ_UNORM);
    let mn = vec4<f32>(-dmin * f32(sm.y));
    let qsw = off + 16u + (t / 2u) * 32u + half;
    let sh = (t % 2u) * 4u;
    for (var i: u32 = 0u; i < 16u; i = i + 4u) {
        b_put(i, bcol, fma(q4k_nibbles(read_u32(qsw + i), sh), sc, mn));
    }
}

/// Q6_K: 210 bytes / 256 elements. The stage covers half of one 32-element
/// quarter of a 128-element half; a quarter reads its own `ql` byte (low or high
/// nibble) and its own two bits of `qh`, and its own i8 scale.
fn decode_b_q6k(k_off: u32, wcol: u32, bcol: u32) {
    let nbk = params.k / K_BLK_ELEMS;
    let b = k_off / K_BLK_ELEMS;
    let within = k_off % K_BLK_ELEMS;
    let h = within / 128u;
    let q = (within % 128u) / 32u;
    let half = within % 32u;                 // 0 or 16
    let off = (wcol * nbk + b) * 210u;
    if (hdr_off != off) {
        hdr_off = off;
        hdr_d = dq_read_f16(off + 208u);
    }
    let d = hdr_d;
    let sc = d * f32(read_w_i8(off + 192u + h * 8u + half / 16u + 2u * q));
    let ql = off + h * 64u + (q % 2u) * 32u + half;
    let qh = off + 128u + h * 32u + half;
    let sh = (q % 2u) * 2u + select(0u, 4u, q >= 2u);
    let hi = q >= 2u;
    for (var i: u32 = 0u; i < 16u; i = i + 4u) {
        b_put(i, bcol, q6_quarter(sc, bytes_x4(read_word_at(ql + i)), bytes_x4(read_word_at(qh + i)), sh, hi));
    }
}

/// Q8_0: 34 bytes / 32 elements. The stage covers half a block's 32 quants.
fn decode_b_q8_0(k_off: u32, wcol: u32, bcol: u32) {
    let nbk = params.k / 32u;
    let b = k_off / 32u;
    let half = k_off % 32u;                  // 0 or 16
    let off = (wcol * nbk + b) * 34u;
    if (hdr_off != off) {
        hdr_off = off;
        hdr_d = dq_read_f16(off);
    }
    let d = hdr_d;
    let dq = vec4<f32>(d * DQ_UNORM);
    let bias = vec4<f32>(-Q8_0_BIAS * d);
    let qs = off + 2u + half;
    for (var i: u32 = 0u; i < 16u; i = i + 4u) {
        b_put(i, bcol, fma(q8_0_quants(read_word_at(qs + i)), dq, bias));
    }
}

// ── the multiply ────────────────────────────────────────────────────────────

/// The three entry points are one body: `decode` names the scheme's reader, and
/// everything else — geometry, staging, the MMA loop, the store — is shared.
fn dequant_gemm_coop(lid: u32, wid: vec3<u32>, decode: u32) {
    let col_base = wid.x * TN;
    // The last row tile's base is clamped to `m - TM` rather than rounded up: a
    // cooperative store cannot be masked, so the tile has to land inside the
    // output. The overlap rows are recomputed, and recomputing them is
    // deterministic — same operands, same k order, same tile — so the two tiles
    // that write them agree bit for bit.
    let row_base = min(wid.y * TM, params.m - min(params.m, TM));
    let sg = lid / 32u;
    let lane = lid % 32u;
    let sg_col = sg * SG_COLS;

    if (lid < 64u) {
        zero_lanes[lid] = 0.0;
    }
    workgroupBarrier();

    // 16 accumulators — 4 row sub-tiles by 4 column sub-tiles of this subgroup's
    // quarter — all f32 while A and B stay f16.
    var acc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);

    let n_stages = params.k / TK;
    // `bc` indexes the workgroup's own B stage (tile-local, 0 .. TN), `colw` the
    // packed weight (global). Two indices rather than one: the stage is one tile
    // wide and the weight is the whole chunk, which the admission makes whole
    // column tiles of, so `col_base + bc` is always inside `ncols`.
    let bc = b_col(sg, lane);
    let colw = col_base + bc;

    // Prime A[0] and B[0], then refill A one stage ahead of the MMAs that read it.
    stage_a(0u, 0u, row_base, lid);
    if (decode == 0u) { decode_b_q4k(0u, colw, bc); }
    else if (decode == 2u) { decode_b_q6k(0u, colw, bc); }
    else { decode_b_q8_0(0u, colw, bc); }
    workgroupBarrier();

    for (var t: u32 = 0u; t < n_stages; t = t + 1u) {
        let cur = t % 2u;
        if (t + 1u < n_stages) {
            stage_a((t + 1u) % 2u, (t + 1u) * TK, row_base, lid);
        }
        // Two K sub-steps of 8, each the hardware MMA's own K.
        for (var kk: u32 = 0u; kk < TK; kk = kk + 8u) {
            let a0 = coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TM * A_STRIDE) + 0u * A_STRIDE + kk], A_STRIDE);
            let a1 = coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TM * A_STRIDE) + 8u * A_STRIDE + kk], A_STRIDE);
            let a2 = coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TM * A_STRIDE) + 16u * A_STRIDE + kk], A_STRIDE);
            let a3 = coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TM * A_STRIDE) + 24u * A_STRIDE + kk], A_STRIDE);
            let b_row = kk * B_STRIDE + sg_col;
            let b0 = coopLoadT<coop_mat8x8<f16, B>>(&b_stage[b_row + 0u], B_STRIDE);
            let b1 = coopLoadT<coop_mat8x8<f16, B>>(&b_stage[b_row + 8u], B_STRIDE);
            let b2 = coopLoadT<coop_mat8x8<f16, B>>(&b_stage[b_row + 16u], B_STRIDE);
            let b3 = coopLoadT<coop_mat8x8<f16, B>>(&b_stage[b_row + 24u], B_STRIDE);

            acc_00 = coopMultiplyAdd(a0, b0, acc_00);
            acc_01 = coopMultiplyAdd(a0, b1, acc_01);
            acc_02 = coopMultiplyAdd(a0, b2, acc_02);
            acc_03 = coopMultiplyAdd(a0, b3, acc_03);
            acc_10 = coopMultiplyAdd(a1, b0, acc_10);
            acc_11 = coopMultiplyAdd(a1, b1, acc_11);
            acc_12 = coopMultiplyAdd(a1, b2, acc_12);
            acc_13 = coopMultiplyAdd(a1, b3, acc_13);
            acc_20 = coopMultiplyAdd(a2, b0, acc_20);
            acc_21 = coopMultiplyAdd(a2, b1, acc_21);
            acc_22 = coopMultiplyAdd(a2, b2, acc_22);
            acc_23 = coopMultiplyAdd(a2, b3, acc_23);
            acc_30 = coopMultiplyAdd(a3, b0, acc_30);
            acc_31 = coopMultiplyAdd(a3, b1, acc_31);
            acc_32 = coopMultiplyAdd(a3, b2, acc_32);
            acc_33 = coopMultiplyAdd(a3, b3, acc_33);
        }
        // Every read of this stage's B and of the A buffer it used is behind this
        // point, so the refill below may overwrite them.
        workgroupBarrier();
        if (t + 1u < n_stages) {
            let k_next = (t + 1u) * TK;
            if (decode == 0u) { decode_b_q4k(k_next, colw, bc); }
            else if (decode == 2u) { decode_b_q6k(k_next, colw, bc); }
            else { decode_b_q8_0(k_next, colw, bc); }
        }
        // Publishes the refilled B, and the A panel staged above.
        workgroupBarrier();
    }

    // Each subgroup stores its quarter straight into the slab: `ncols` is the
    // output row stride, so the four quarters tile the 32×128 output region.
    let dst = row_base * params.ncols + col_base + sg_col;
    coopStoreT(acc_00, &outarr[dst + 0u * params.ncols + 0u], params.ncols);
    coopStoreT(acc_01, &outarr[dst + 0u * params.ncols + 8u], params.ncols);
    coopStoreT(acc_02, &outarr[dst + 0u * params.ncols + 16u], params.ncols);
    coopStoreT(acc_03, &outarr[dst + 0u * params.ncols + 24u], params.ncols);
    coopStoreT(acc_10, &outarr[dst + 8u * params.ncols + 0u], params.ncols);
    coopStoreT(acc_11, &outarr[dst + 8u * params.ncols + 8u], params.ncols);
    coopStoreT(acc_12, &outarr[dst + 8u * params.ncols + 16u], params.ncols);
    coopStoreT(acc_13, &outarr[dst + 8u * params.ncols + 24u], params.ncols);
    coopStoreT(acc_20, &outarr[dst + 16u * params.ncols + 0u], params.ncols);
    coopStoreT(acc_21, &outarr[dst + 16u * params.ncols + 8u], params.ncols);
    coopStoreT(acc_22, &outarr[dst + 16u * params.ncols + 16u], params.ncols);
    coopStoreT(acc_23, &outarr[dst + 16u * params.ncols + 24u], params.ncols);
    coopStoreT(acc_30, &outarr[dst + 24u * params.ncols + 0u], params.ncols);
    coopStoreT(acc_31, &outarr[dst + 24u * params.ncols + 8u], params.ncols);
    coopStoreT(acc_32, &outarr[dst + 24u * params.ncols + 16u], params.ncols);
    coopStoreT(acc_33, &outarr[dst + 24u * params.ncols + 24u], params.ncols);
}

@compute @workgroup_size(128, 1, 1)
fn dequant_gemm_coop_q4k(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    dequant_gemm_coop(lid, wid, 0u);
}

@compute @workgroup_size(128, 1, 1)
fn dequant_gemm_coop_q6k(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    dequant_gemm_coop(lid, wid, 2u);
}

@compute @workgroup_size(128, 1, 1)
fn dequant_gemm_coop_q8_0(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    dequant_gemm_coop(lid, wid, 20u);
}
