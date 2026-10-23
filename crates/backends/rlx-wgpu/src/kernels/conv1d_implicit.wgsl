// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// The convolution GEMM routes' third form: the tile of
// `matmul_coop_f32_direct.wgsl` with the im2col panel taken away.
//
// The im2col route materialises `col[c_in·kh·kw, spatial]` into scratch and then
// multiplies it, so a conv pays one write and one read of `k_total · spatial ·
// 4` bytes for an operand it already has: the gathered row is a strided read of
// the activation through the conv's own taps. This entry gathers each 8-row
// k-step of B straight into `var<workgroup>` — 8 k-rows × 32 spatial columns,
// the workgroup's own column block — and multiplies it on the same 32×32 f32
// cooperative tile the direct entry uses. A is the weight, read from the arena
// with `coopLoadT` at stride `k_total` exactly as the direct entry reads it.
//
// The shape this wins on is the 96-channel decoder block's dilated conv, and its
// class is capped at 96 output channels — so **one workgroup covers every output
// row tile**: the gathered 8×32 B step is used by all `ceil(c_out / 32)` row tiles
// before the next k step overwrites it, and the activation is gathered once per
// column block rather than once per row tile. The host admits it only for that
// class — see `ConvImplicitShape` in `backend/mod.rs`, which is also what the
// panel reservation subtracts.
//
// Three accumulator sets are the ceiling this entry has (`ROW_TILES`): an output wider
// than `ROW_TILES * TILE_M` rows would have rows no set covers, so the cap is a host
// term and the kernel carries no guard against it. Everything inside a set is guarded
// by the tile count instead — the multiply and the store of a set the shape does not
// reach are skipped, so a narrower output costs a set's registers but not its
// arithmetic.
//
// Three things the gather does that the panel did not have to:
// * **`(c, tap)` is carried, not divided.** The k index decomposes as
//   `c = k / kw`, `tap = k % kw`; incrementing the pair once per gathered row
//   (and wrapping at `kw`) is one compare instead of a divide per row.
// * **One buffer, two uses.** The 8×32 B tile and the 32×32 store tile share
//   one 4 KiB `var<workgroup>` array — the B tile's last read and the epilogue's
//   first write are separated by the k loop's closing barrier, where two buffers
//   would cost 5 KiB a workgroup and a second set of barriers.
// * **The epilogue's tails are guarded.** The panel multiply could rely on its
//   alignment admission (whole 32×32 tiles); here an admitted shape may end in a
//   partial `c_out` row tile or a partial spatial column tile, so the fragments
//   are stored to the shared tile and copied out under
//   `row < c_out && col < out_len`.
//
// The A read past `c_out` (a padded row tile) is the same reliance
// `matmul_coop_f32.wgsl` states for its own padded reads: the store guard drops
// every row at or past `c_out`, and WebGPU robust buffer access deals with the last
// leg of the weight region. The k-step's tail is a different matter — it reads
// *inside* the arena, the next weight row's own values — and what makes it harmless
// is the B tile: every gathered row at or past `k_total` is zero, so those products
// contribute nothing. That zero fill is the guarantee, not the host's class terms,
// which is why a shape whose `k_total` is not a multiple of `TILE_K` is correct.
//
// Row-major load/store convention: see the long note in `matmul_coop_f32.wgsl`.
// `coopLoad`/`coopLoadT` mean different things on Metal and SPIR-V; this entry is
// Metal-only (`conv1d_implicit_kernel` refuses every other backend). Its B
// fragments load from `var<workgroup>` rather than the arena — the pointer form
// naga's MSL backend resolves to `NagaCooperativeLoad(const threadgroup float*,
// …)`, the one `matmul_coop_f32.wgsl` already stores through (`coopStoreT`).

struct Params {
    // Channels in and out; the weight is `[c_out][c_in·kw]` row-major.
    c_in: u32,
    c_out: u32,
    // The contraction and both extents, all in the same order the panel had.
    k_total: u32,
    // The 1-D input's own length (the conv's W extent, unit height).
    w: u32,
    // The output's length — the GEMM's `n`.
    out_len: u32,
    kw: u32,
    dilation: u32,
    pad: u32,
    in_off: u32,
    w_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
};

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;
// The accumulator sets this entry carries, `TILE_M` rows each: what bounds the output
// rows it covers. `ConvImplicitShape` refuses a wider `c_out` on the host, so the
// clamp below states the bound rather than guarding it.
const ROW_TILES: u32 = 3u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

// `[0, 64)` is the zero fragment every accumulator is loaded from, `[0, 256)` the
// B tile (`[k row][spatial column]`, stride `TILE_N`), and the whole array the
// epilogue's store tile (`[c_out row][spatial column]`, stride `TILE_N`).
var<workgroup> tile: array<f32, 1024>;

/// One row tile of the shared store tile, out to the output's own extents. The
/// host admits at most `ROW_TILES * TILE_M` output rows, which is what bounds the
/// tiles this entry has accumulator sets for.
fn copy_out(base_row: u32, col_base: u32, j: u32) {
    for (var e: u32 = 0u; e < TILE_M; e = e + 1u) {
        let row = base_row + e;
        let col = col_base + j;
        if (row < params.c_out && col < params.out_len) {
            arena[params.out_off + row * params.out_len + col] = tile[e * TILE_N + j];
        }
    }
}

@compute @workgroup_size(32)
fn conv1d_implicit(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col_base = wid.x * TILE_N;
    let j = lid.x;
    // The class caps `c_out` at the sets this entry has (`ROW_TILES`).
    let n_row_tiles = min((params.c_out + TILE_M - 1u) / TILE_M, ROW_TILES);

    if (j < 2u) {
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            tile[j + s * 2u] = 0.0;
        }
    }
    workgroupBarrier();

    // Every row tile's accumulators, read from the zero fragment before the
    // buffer becomes the B tile.
    var acc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accb_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    var accc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    workgroupBarrier();

    // The gather's own walk of the contraction's `(channel, tap)`
    // decomposition, carried forward one k row at a time.
    var gk: u32 = 0u;
    var c: u32 = 0u;
    var tap: u32 = 0u;

    let n_tiles = (params.k_total + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        // B: this workgroup's 32 columns, one gathered input row per k row. A
        // thread owns one column, so the 32 threads read 32 consecutive input
        // positions per row.
        for (var r: u32 = 0u; r < TILE_K; r = r + 1u) {
            var v: f32 = 0.0;
            if (gk < params.k_total) {
                // `pos = col_base + j + tap·dilation − pad` of the 1-D input;
                // `q` is that plus `pad`, so the two unsigned tests stand for
                // `0 <= pos < w` without a signed cast.
                let q = col_base + j + tap * params.dilation;
                if (q >= params.pad && q - params.pad < params.w) {
                    v = arena[params.in_off + c * params.w + (q - params.pad)];
                }
            }
            tile[r * TILE_N + j] = v;
            gk = gk + 1u;
            tap = tap + 1u;
            if (tap == params.kw) {
                tap = 0u;
                c = c + 1u;
            }
        }
        workgroupBarrier();
        let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&tile[0u],  TILE_N);
        let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&tile[8u],  TILE_N);
        let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&tile[16u], TILE_N);
        let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&tile[24u], TILE_N);
        let a_row = params.w_off + k_off;
        let a0_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row], params.k_total);
        let a0_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 8u  * params.k_total], params.k_total);
        let a0_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 16u  * params.k_total], params.k_total);
        let a0_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 24u  * params.k_total], params.k_total);
        acc_00 = coopMultiplyAdd(a0_0, b_0, acc_00);
        acc_01 = coopMultiplyAdd(a0_0, b_1, acc_01);
        acc_02 = coopMultiplyAdd(a0_0, b_2, acc_02);
        acc_03 = coopMultiplyAdd(a0_0, b_3, acc_03);
        acc_10 = coopMultiplyAdd(a0_1, b_0, acc_10);
        acc_11 = coopMultiplyAdd(a0_1, b_1, acc_11);
        acc_12 = coopMultiplyAdd(a0_1, b_2, acc_12);
        acc_13 = coopMultiplyAdd(a0_1, b_3, acc_13);
        acc_20 = coopMultiplyAdd(a0_2, b_0, acc_20);
        acc_21 = coopMultiplyAdd(a0_2, b_1, acc_21);
        acc_22 = coopMultiplyAdd(a0_2, b_2, acc_22);
        acc_23 = coopMultiplyAdd(a0_2, b_3, acc_23);
        acc_30 = coopMultiplyAdd(a0_3, b_0, acc_30);
        acc_31 = coopMultiplyAdd(a0_3, b_1, acc_31);
        acc_32 = coopMultiplyAdd(a0_3, b_2, acc_32);
        acc_33 = coopMultiplyAdd(a0_3, b_3, acc_33);
        if (n_row_tiles > 1u) {
            let a_rowb = params.w_off + TILE_M * params.k_total + k_off;
            let a1_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowb], params.k_total);
            let a1_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowb + 8u  * params.k_total], params.k_total);
            let a1_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowb + 16u  * params.k_total], params.k_total);
            let a1_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowb + 24u  * params.k_total], params.k_total);
            accb_00 = coopMultiplyAdd(a1_0, b_0, accb_00);
            accb_01 = coopMultiplyAdd(a1_0, b_1, accb_01);
            accb_02 = coopMultiplyAdd(a1_0, b_2, accb_02);
            accb_03 = coopMultiplyAdd(a1_0, b_3, accb_03);
            accb_10 = coopMultiplyAdd(a1_1, b_0, accb_10);
            accb_11 = coopMultiplyAdd(a1_1, b_1, accb_11);
            accb_12 = coopMultiplyAdd(a1_1, b_2, accb_12);
            accb_13 = coopMultiplyAdd(a1_1, b_3, accb_13);
            accb_20 = coopMultiplyAdd(a1_2, b_0, accb_20);
            accb_21 = coopMultiplyAdd(a1_2, b_1, accb_21);
            accb_22 = coopMultiplyAdd(a1_2, b_2, accb_22);
            accb_23 = coopMultiplyAdd(a1_2, b_3, accb_23);
            accb_30 = coopMultiplyAdd(a1_3, b_0, accb_30);
            accb_31 = coopMultiplyAdd(a1_3, b_1, accb_31);
            accb_32 = coopMultiplyAdd(a1_3, b_2, accb_32);
            accb_33 = coopMultiplyAdd(a1_3, b_3, accb_33);
        }
        if (n_row_tiles > 2u) {
            let a_rowc = params.w_off + 2u * TILE_M * params.k_total + k_off;
            let a2_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowc], params.k_total);
            let a2_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowc + 8u  * params.k_total], params.k_total);
            let a2_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowc + 16u  * params.k_total], params.k_total);
            let a2_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_rowc + 24u  * params.k_total], params.k_total);
            accc_00 = coopMultiplyAdd(a2_0, b_0, accc_00);
            accc_01 = coopMultiplyAdd(a2_0, b_1, accc_01);
            accc_02 = coopMultiplyAdd(a2_0, b_2, accc_02);
            accc_03 = coopMultiplyAdd(a2_0, b_3, accc_03);
            accc_10 = coopMultiplyAdd(a2_1, b_0, accc_10);
            accc_11 = coopMultiplyAdd(a2_1, b_1, accc_11);
            accc_12 = coopMultiplyAdd(a2_1, b_2, accc_12);
            accc_13 = coopMultiplyAdd(a2_1, b_3, accc_13);
            accc_20 = coopMultiplyAdd(a2_2, b_0, accc_20);
            accc_21 = coopMultiplyAdd(a2_2, b_1, accc_21);
            accc_22 = coopMultiplyAdd(a2_2, b_2, accc_22);
            accc_23 = coopMultiplyAdd(a2_2, b_3, accc_23);
            accc_30 = coopMultiplyAdd(a2_3, b_0, accc_30);
            accc_31 = coopMultiplyAdd(a2_3, b_1, accc_31);
            accc_32 = coopMultiplyAdd(a2_3, b_2, accc_32);
            accc_33 = coopMultiplyAdd(a2_3, b_3, accc_33);
        }
        // Every thread's B fragments are in registers; the next step may write
        // the tile again.
        workgroupBarrier();
    }

    // The epilogue, one row tile at a time: the fragments into the shared tile,
    // then the part of it that lands inside the output's own extents.
    coopStoreT(acc_00, &tile[0u * TILE_N], TILE_N);
    coopStoreT(acc_01, &tile[0u * TILE_N + 8u], TILE_N);
    coopStoreT(acc_02, &tile[0u * TILE_N + 16u], TILE_N);
    coopStoreT(acc_03, &tile[0u * TILE_N + 24u], TILE_N);
    coopStoreT(acc_10, &tile[8u * TILE_N], TILE_N);
    coopStoreT(acc_11, &tile[8u * TILE_N + 8u], TILE_N);
    coopStoreT(acc_12, &tile[8u * TILE_N + 16u], TILE_N);
    coopStoreT(acc_13, &tile[8u * TILE_N + 24u], TILE_N);
    coopStoreT(acc_20, &tile[16u * TILE_N], TILE_N);
    coopStoreT(acc_21, &tile[16u * TILE_N + 8u], TILE_N);
    coopStoreT(acc_22, &tile[16u * TILE_N + 16u], TILE_N);
    coopStoreT(acc_23, &tile[16u * TILE_N + 24u], TILE_N);
    coopStoreT(acc_30, &tile[24u * TILE_N], TILE_N);
    coopStoreT(acc_31, &tile[24u * TILE_N + 8u], TILE_N);
    coopStoreT(acc_32, &tile[24u * TILE_N + 16u], TILE_N);
    coopStoreT(acc_33, &tile[24u * TILE_N + 24u], TILE_N);
    workgroupBarrier();
    copy_out(0u, col_base, j);
    workgroupBarrier();
    if (n_row_tiles > 1u) {
        coopStoreT(accb_00, &tile[0u * TILE_N], TILE_N);
        coopStoreT(accb_01, &tile[0u * TILE_N + 8u], TILE_N);
        coopStoreT(accb_02, &tile[0u * TILE_N + 16u], TILE_N);
        coopStoreT(accb_03, &tile[0u * TILE_N + 24u], TILE_N);
        coopStoreT(accb_10, &tile[8u * TILE_N], TILE_N);
        coopStoreT(accb_11, &tile[8u * TILE_N + 8u], TILE_N);
        coopStoreT(accb_12, &tile[8u * TILE_N + 16u], TILE_N);
        coopStoreT(accb_13, &tile[8u * TILE_N + 24u], TILE_N);
        coopStoreT(accb_20, &tile[16u * TILE_N], TILE_N);
        coopStoreT(accb_21, &tile[16u * TILE_N + 8u], TILE_N);
        coopStoreT(accb_22, &tile[16u * TILE_N + 16u], TILE_N);
        coopStoreT(accb_23, &tile[16u * TILE_N + 24u], TILE_N);
        coopStoreT(accb_30, &tile[24u * TILE_N], TILE_N);
        coopStoreT(accb_31, &tile[24u * TILE_N + 8u], TILE_N);
        coopStoreT(accb_32, &tile[24u * TILE_N + 16u], TILE_N);
        coopStoreT(accb_33, &tile[24u * TILE_N + 24u], TILE_N);
        workgroupBarrier();
        copy_out(TILE_M, col_base, j);
        workgroupBarrier();
    }
    if (n_row_tiles > 2u) {
        coopStoreT(accc_00, &tile[0u * TILE_N], TILE_N);
        coopStoreT(accc_01, &tile[0u * TILE_N + 8u], TILE_N);
        coopStoreT(accc_02, &tile[0u * TILE_N + 16u], TILE_N);
        coopStoreT(accc_03, &tile[0u * TILE_N + 24u], TILE_N);
        coopStoreT(accc_10, &tile[8u * TILE_N], TILE_N);
        coopStoreT(accc_11, &tile[8u * TILE_N + 8u], TILE_N);
        coopStoreT(accc_12, &tile[8u * TILE_N + 16u], TILE_N);
        coopStoreT(accc_13, &tile[8u * TILE_N + 24u], TILE_N);
        coopStoreT(accc_20, &tile[16u * TILE_N], TILE_N);
        coopStoreT(accc_21, &tile[16u * TILE_N + 8u], TILE_N);
        coopStoreT(accc_22, &tile[16u * TILE_N + 16u], TILE_N);
        coopStoreT(accc_23, &tile[16u * TILE_N + 24u], TILE_N);
        coopStoreT(accc_30, &tile[24u * TILE_N], TILE_N);
        coopStoreT(accc_31, &tile[24u * TILE_N + 8u], TILE_N);
        coopStoreT(accc_32, &tile[24u * TILE_N + 16u], TILE_N);
        coopStoreT(accc_33, &tile[24u * TILE_N + 24u], TILE_N);
        workgroupBarrier();
        copy_out(2u * TILE_M, col_base, j);
    }
}
