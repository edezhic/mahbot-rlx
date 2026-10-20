// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Vectorised wide-tile matmul: 64x64 output tile, 16x16 threads, 4x4 per
// thread, every tile access a `vec4<f32>`.
//
// WHY: `matmul_wide.wgsl` measured ~120 GF/s on an M4 Pro — about 2.5% of peak.
// Four structural hypotheses were tested against it and ALL came back flat:
// fully unrolling the register block (naga/Metal already promotes
// constant-bounded local arrays), dropping naga's serial workgroup zero-init,
// padding `tile_a`'s row stride to kill a measured 4-way bank conflict, and
// raising the workgroup from 64 to 256 threads (the `matmul_wide_nv` geometry).
// None moved the number, and the generated MSL showed the arena loads are not
// bounds-checked either.
//
// What did move it was VECTORISATION: same tiling, same 4x4 register block,
// but `vec4` tile storage and a `vec4` FMA in the inner loop — 122 -> 346 GF/s
// on the same device through the same WGSL -> naga -> Metal path. The scalar
// form gives the driver no vector memory ops to schedule; the tiling was never
// the problem.
//
// The ROW AXIS is a second geometry. `matmul_wide_vec4` keeps whole 64-row
// tiles; `matmul_wide_vec4_tail` is the same body at 16x8 threads / 32 rows,
// over the rows the 64-row grid rounds up — so a 96-row node computes 96 rows
// and not 128, the padded quarter. For the node this was added for (96 rows,
// 48000 columns, 672 contraction: a dense 1x7 convolution's im2col GEMM), an
// interleaved A/B series against the single-tile build measured 9.37 -> 7.04 ms
// per node. `kernels::wide_grid` is the Rust side of that rule and the
// device-free tests at the end of `kernels/mod.rs` pin the two together;
// `tests/matmul_vec4_parity.rs` pins the pair bit-identical to the single
// rounded-up grid it replaces. (That node's GEMM may now take the f32
// cooperative kernel where its alignment admits it — see `conv_gemm` — which
// leaves these tiles to the shapes the cooperative alignment refuses.)
//
// This is the path an adapter without the cooperative-matrix extension gets:
// `matmul_coop_*` needs EXPERIMENTAL_COOPERATIVE_MATRIX and a shape the coop
// alignment admits, so this tile kernel is the fallback wherever either
// refuses.
//
// Accumulation order over k is unchanged (0..K-1, sequential), so results stay
// bit-identical to `matmul_wide` — `tests/matmul_vec4_parity.rs` asserts it.

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

const FULL_M: u32 = 64u;   // output rows per workgroup of the full-tile pipeline
const TAIL_M: u32 = 32u;   // output rows per workgroup of the remainder pipeline
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;


fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    switch (params.act_id) {
        case 0xFFFFu: {}
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u, 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        default: {}
    }
    return v;
}


// A tile: `4 * tile_m` vec4 = tile_m rows x TILE_K, indexed [row * 4 + (k >> 2)].
// Sized for the full tile; the remainder pipeline leaves the rows past its own
// tile_m unwritten and never reads them (see the row mapping below).
var<workgroup> ta: array<vec4<f32>, 256>;
// B tile: TILE_K x TILE_N = 256 vec4, indexed [k * 16 + (col >> 2)].
var<workgroup> tb: array<vec4<f32>, 256>;

/// One f32 from the arena, or 0.0 when the index is outside the operand. The
/// zero keeps out-of-range lanes from contributing to the dot product, which is
/// what lets M, N and K be arbitrary.
fn a_at(base: u32, row: u32, kk: u32, rows: u32) -> f32 {
    if (row >= rows || kk >= params.k) {
        return 0.0;
    }
    return arena[base + row * params.k + kk];
}

fn b_at(base: u32, kk: u32, col: u32) -> f32 {
    if (kk >= params.k || col >= params.n) {
        return 0.0;
    }
    return arena[base + kk * params.n + col];
}

/// One output tile of `tile_m` rows x TILE_N columns, `4 * tile_m` threads,
/// starting at output row `row_origin`.
fn wide_vec4_tile(
    lid: vec3<u32>,
    wid: vec3<u32>,
    tile_m: u32,
    row_origin: u32,
) {
    // No early return before workgroupBarrier — FXC rejects barriers under
    // varying control flow (X4026), the same rule the other tiled kernels obey.
    let bz = wid.z;
    let in_batch = bz < params.batch;
    let bz_safe = select(0u, bz, in_batch);

    let a_base = params.a_off + bz_safe * params.a_batch_stride;
    let b_base = params.b_off + bz_safe * params.b_batch_stride;
    let c_base = params.c_off + bz_safe * params.c_batch_stride;

    let lr = lid.y;
    let lc = lid.x;
    let li = lr * 16u + lc;          // one vec4 of each tile per thread
    let threads = 4u * tile_m;
    let row_base = row_origin + wid.y * tile_m;
    let col_base = wid.x * TILE_N;

    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k0 = t * TILE_K;

        // Cooperative load, one vec4 per thread per pass.
        // A: thread li covers row li/4, k chunk li%4 — consecutive li walk k
        // first, so each group of 4 threads reads one contiguous 64-byte run.
        // `threads == 4 * tile_m` lands exactly on the tile, so this load is
        // the same code at both geometries.
        let ar = li >> 2u;
        let ak = (li & 3u) * 4u;
        let arow = row_base + ar;
        ta[li] = vec4<f32>(
            a_at(a_base, arow, k0 + ak + 0u, params.m),
            a_at(a_base, arow, k0 + ak + 1u, params.m),
            a_at(a_base, arow, k0 + ak + 2u, params.m),
            a_at(a_base, arow, k0 + ak + 3u, params.m),
        );
        // B: the 16 x 64 tile is 256 vec4 — one per thread at the full
        // geometry, two at the half-height one. Thread i covers k i/16, col
        // chunk i%16 — consecutive i walk the column axis, so each group of 16
        // threads reads a contiguous 256-byte run of one B row.
        for (var i: u32 = li; i < 256u; i = i + threads) {
            let bk = k0 + (i >> 4u);
            let bc = col_base + (i & 15u) * 4u;
            tb[i] = vec4<f32>(
                b_at(b_base, bk, bc + 0u),
                b_at(b_base, bk, bc + 1u),
                b_at(b_base, bk, bc + 2u),
                b_at(b_base, bk, bc + 3u),
            );
        }

        workgroupBarrier();

        // 4x4 outer product per k, as four vec4 FMAs. k advances one at a time,
        // so the summation order matches the scalar kernel exactly.
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            let bv = tb[k * 16u + lc];
            let kc = k >> 2u;
            let ke = k & 3u;
            acc[0] = acc[0] + ta[(lr * 4u + 0u) * 4u + kc][ke] * bv;
            acc[1] = acc[1] + ta[(lr * 4u + 1u) * 4u + kc][ke] * bv;
            acc[2] = acc[2] + ta[(lr * 4u + 2u) * 4u + kc][ke] * bv;
            acc[3] = acc[3] + ta[(lr * 4u + 3u) * 4u + kc][ke] * bv;
        }

        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let global_row = row_base + lr * 4u + i;
        if (in_batch && global_row < params.m) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let global_col = col_base + lc * 4u + j;
                if (global_col < params.n) {
                    var v = acc[i][j];
                    if (params.has_bias != 0u) {
                        v = v + arena[params.bias_off + global_col];
                    }
                    v = apply_act(v);
                    arena[c_base + global_row * params.n + global_col] = v;
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn matmul_wide_vec4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    wide_vec4_tile(lid, wid, FULL_M, 0u);
}

/// The rows the `FULL_M` grid rounds the row axis up to: `wide_grid` records this
/// entry point once when the row count leaves a remainder of at most `TAIL_M`,
/// over `(params.m / FULL_M) * FULL_M .. + TAIL_M`. Rows past `params.m` are
/// still computed (as zeros, via `a_at`'s row bound) and never stored — the
/// epilogue is unchanged. Same thread mapping, same k order as the full tile, so
/// the pair is bit-identical to the single rounded-up grid it replaces.
@compute @workgroup_size(16, 8)
fn matmul_wide_vec4_tail(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    wide_vec4_tile(lid, wid, TAIL_M, (params.m / FULL_M) * FULL_M);
}
