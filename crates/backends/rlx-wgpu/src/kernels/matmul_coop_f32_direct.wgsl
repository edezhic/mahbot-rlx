// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// The convolution GEMM routes' form: the same 32×32 output tile and the same
// 16 cooperative f32 fragments as `matmul_coop_f32`, with two things removed.
//
// * **No shared-memory staging.** A is loaded straight from the arena with
//   `coopLoadT` at stride `k`. The staged form copied A through 1 KiB of
//   workgroup memory and paid two barriers per 8-element k step; at 5 KiB per
//   workgroup that also capped how many workgroups a core could hold. This entry
//   declares 256 B (a zero fragment) and no staging, so the tile's cost per k
//   step is four A loads, four B loads and sixteen `coopMultiplyAdd`s.
// * **No epilogue pass.** The convolution GEMMs carry no bias and no activation
//   (bias and Snake are separate nodes), so the fragments are stored straight to
//   C with `coopStoreT`. The host admits this entry only on the two convolution
//   GEMM routes, only when that is true, and only when `m` and `n` are multiples
//   of 32 (`coop_f32_direct_admitted`), so no padded row or column can be
//   written; every other call keeps the general kernel, whose scratch store and
//   scalar epilogue are what pay for ragged extents.
//
// Row-major load/store convention: see the long note in `matmul_coop_f32.wgsl`.
// `coopLoad`/`coopLoadT` mean different things on Metal and SPIR-V; this entry is
// Metal-only (`matmul_coop_f32_direct_kernel` refuses every other backend).

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

// `has_bias`, `bias_off` and `act_id` are part of the record but unread here:
// this entry shares `matmul_coop_f32`'s uniform layout, the host binds it at that
// layout, and the entry is admitted only where there is no epilogue to apply.
// Do not "clean them up" — the layout is not this kernel's to change.

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

// The zero fragment every accumulator is loaded from: 8×8 f32.
var<workgroup> zero: array<f32, 64>;

@compute @workgroup_size(32)
fn matmul_coop_f32_direct(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    if (lid.x < 2u) {
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            zero[lid.x + s * 2u] = 0.0;
        }
    }
    workgroupBarrier();

    var acc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero[0], 8u);

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        let a_row = a_base + row_base * params.k + k_off;
        let a_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row],                  params.k);
        let a_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 8u  * params.k], params.k);
        let a_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 16u * params.k], params.k);
        let a_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&arena[a_row + 24u * params.k], params.k);
        let b_row = b_base + k_off * params.n + col_base;
        let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row],        params.n);
        let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 8u ], params.n);
        let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 16u], params.n);
        let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 24u], params.n);
        acc_00 = coopMultiplyAdd(a_0, b_0, acc_00);
        acc_01 = coopMultiplyAdd(a_0, b_1, acc_01);
        acc_02 = coopMultiplyAdd(a_0, b_2, acc_02);
        acc_03 = coopMultiplyAdd(a_0, b_3, acc_03);
        acc_10 = coopMultiplyAdd(a_1, b_0, acc_10);
        acc_11 = coopMultiplyAdd(a_1, b_1, acc_11);
        acc_12 = coopMultiplyAdd(a_1, b_2, acc_12);
        acc_13 = coopMultiplyAdd(a_1, b_3, acc_13);
        acc_20 = coopMultiplyAdd(a_2, b_0, acc_20);
        acc_21 = coopMultiplyAdd(a_2, b_1, acc_21);
        acc_22 = coopMultiplyAdd(a_2, b_2, acc_22);
        acc_23 = coopMultiplyAdd(a_2, b_3, acc_23);
        acc_30 = coopMultiplyAdd(a_3, b_0, acc_30);
        acc_31 = coopMultiplyAdd(a_3, b_1, acc_31);
        acc_32 = coopMultiplyAdd(a_3, b_2, acc_32);
        acc_33 = coopMultiplyAdd(a_3, b_3, acc_33);
    }
    let c_row = c_base + row_base * params.n + col_base;
    coopStoreT(acc_00, &arena[c_row],        params.n);
    coopStoreT(acc_01, &arena[c_row + 8u ],  params.n);
    coopStoreT(acc_02, &arena[c_row + 16u],  params.n);
    coopStoreT(acc_03, &arena[c_row + 24u],  params.n);
    coopStoreT(acc_10, &arena[c_row + 8u  * params.n], params.n);
    coopStoreT(acc_11, &arena[c_row + 8u  * params.n + 8u ], params.n);
    coopStoreT(acc_12, &arena[c_row + 8u  * params.n + 16u], params.n);
    coopStoreT(acc_13, &arena[c_row + 8u  * params.n + 24u], params.n);
    coopStoreT(acc_20, &arena[c_row + 16u * params.n], params.n);
    coopStoreT(acc_21, &arena[c_row + 16u * params.n + 8u ], params.n);
    coopStoreT(acc_22, &arena[c_row + 16u * params.n + 16u], params.n);
    coopStoreT(acc_23, &arena[c_row + 16u * params.n + 24u], params.n);
    coopStoreT(acc_30, &arena[c_row + 24u * params.n], params.n);
    coopStoreT(acc_31, &arena[c_row + 24u * params.n + 8u ], params.n);
    coopStoreT(acc_32, &arena[c_row + 24u * params.n + 16u], params.n);
    coopStoreT(acc_33, &arena[c_row + 24u * params.n + 24u], params.n);
}
