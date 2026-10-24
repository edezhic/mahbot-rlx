// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// f16 operands, f32 cooperative accumulator, 64×64 output tile per workgroup
// (two 32-thread subgroups, each owning a 64×32 half), A staged 32 K rows at a
// time through a ping-pong workgroup buffer, B read straight from the f16
// weight buffer.
//
// This is the packed-weight GEMM the embedding encoder's text tower wants
// (EmbeddingGemma 2, one bidirectional 8 192-row pass): A (activations) is read
// from the f32 arena and downcast to f16 as it lands in workgroup staging, B
// (weights) is read as f16 straight out of the f16 shadow buffer, and the
// hardware MMA accumulates in f32. Nothing is materialised: there is no
// `Step::CastF32ToF16` pre-pass, which is what made the f16 cooperative path
// regress BERT end-to-end (see the END-TO-END PERF NOTE in
// `matmul_coop16.wgsl`) — the cast costs A's bytes again in bandwidth and loses
// to the matmul it enables.
//
// ── THE DENSE ATTENTION ROUTE'S USE ────────────────────────────────────────
//
// The other caller is `backend::attention_route`'s cooperative dense square
// form: a `MaskKind::None`, square, 8 192-row step at the embedding encoder's
// global-layer geometry (4 heads of width 512 over one KV head) runs both of its
// multiplies through this kernel — A is Q′ or the softmax's output, read from the
// f32 arena, and B is Kᵀ then V′ materialised as f16 in a buffer the route owns
// (`MatmulCompute::CoopF16F32AccScratch`). Both operands are row-major `[m, k]`
// and `[k, n]` at the same `coopLoadT`/`coopStoreT` convention this file
// documents, which is why the route keeps its Kᵀ and V′ transposes rather than
// teaching the kernel a second convention.
//
// Measured on the same rig and harness (`tests/attention_dense_coop_cost.rs`),
// one dispatch per chunk of 2 048 rows, batch = 4 `(batch, head)` tiles, median
// of 9 after 3 warm-ups, TFLOP/s:
//
//   scores (m 2048, k 512, n 8192)   3.59      out (m 2048, k 8192, n 512)   3.47
//   scores (m 8192, k 512, n 8192)   3.67      out (m 8192, k 8192, n 512)   3.58
//
// The route walks the row extent in chunks so its score matrix fits the 512 MiB
// scratch cap, which is what the two rows above price: 3.59 at the chunk against
// 3.67 whole. Its whole dense step — the transposes, the two f16 casts, four
// chunks of both multiplies and the softmax — reads 230 ms at the encoder's
// geometry with host I/O included, against the band kernel's ~535 ms at the same
// shape (that kernel's own header), and the stand's `embed.8192` cell moves
// 3 589 → 2 312 ms with four such layers in it. What this does *not* establish:
// that a different tile would not be faster for the value multiply's short, wide
// `n = 512`; only the 64×64 form above was tried at the route's shapes, and a
// 64×128 four-subgroup form measured 2.7–2.9 TFLOP/s there under the f32-operand
// variant this route started from, which is why it was not carried over.
//
// ── THE TILE, AND WHAT IT WAS MEASURED AT ───────────────────────────────────
//
// 64×64 per workgroup over TWO 32-wide subgroups, each owning a 64×32 half and
// 32 8×8 f32 accumulators — eight row sub-tiles by four column sub-tiles, twice
// the per-subgroup accumulator file of `matmul_coop_f32.wgsl` and of the four-
// subgroup form this replaced. Two things about that shape are load-bearing, and
// both were measured (see the table below): 32 accumulators per subgroup is what
// buys the MMA-to-operand-load ratio, and the 64×64 tile at a 32-deep K stage is
// what keeps the two ping-pong A buffers at 8 KiB in total, small enough that
// several workgroups stay resident per GPU core (32 KiB of workgroup storage per
// core would make that four).
//
// Measured on Apple M3 Pro (14-core GPU, wgpu 30 over Metal), the text tower's
// projections, best of 7 after 3 warm-ups, ONE dispatch with no readback
// (`tests/encoder_gemm_cost.rs`, `the_single_dispatch_rate_of_each_candidate_kernel`),
// TFLOP/s at m = 8192, k, n as the tower presents them:
//
//   shape                     this kernel   previous form   matmul_coop16
//   512→1024                      3.32          2.40            2.39
//   512→2048                      3.50          2.47            2.51
//   512→512                       3.07          2.24            2.19
//   1024→512                      3.38          2.36            2.43
//   2048→512                      3.55          2.44            2.57
//   512→12288 (PLE)               3.66          2.56            2.59
//
// Five things that table settles, and one it does not:
//
//   * The previous form — one 32×32 quadrant per subgroup, four subgroups, a
//     single 4 KiB A stage refilled under a barrier — is the column to read the
//     gain against: +37% to +45% per shape.
//   * The accumulator count, not the tile, is the lever: 32 per subgroup at
//     64×64 (two subgroups) measured 3.13–3.73, while 16 per subgroup at the
//     same 64×64 tile (four subgroups) measured 3.03–3.56 and 16 per subgroup
//     at 32×256 measured 3.26–3.53. A 32×32 quadrant is one operand load too
//     many per MMA.
//   * The occupancy term is workgroup storage, and it is sharp. 8 KiB of A
//     stage — this form — is the fastest point measured; 16 KiB (a 64-deep K
//     stage at 64 rows, or a 128-row tile) collapses to 0.6–2.6 TFLOP/s, which is
//     the same cliff the original 16 KiB f32 output tile fell off at 0.35
//     TFLOP/s. The residency count itself is inferred from that cliff, not read
//     off a counter.
//   * Staging B into workgroup memory loses. B is already f16 in its own buffer
//     and the hardware's strided load coalesces it, so the stage buys nothing but
//     its own residency: in the 64×128 form tried here it pushed the workgroup to
//     16 KiB and measured 0.68–1.31 TFLOP/s, and in the previous four-subgroup
//     form it measured −25% (1.86 against 2.39 at 512→1024). The barrier and the
//     stage loop were not the cost; the 4–8 KiB was.
//   * Downcast four adjacent K columns per invocation instead of one: the four
//     f32 loads are then one coalesced run and the compiler emits them as a
//     vector. 2.2 → 3.4 TFLOP/s at the same tile — the second-largest term here.
//   * NOT established: this is a local optimum among the forms measured. A
//     64×128 tile (four subgroups, same 32-accumulator halves) measured
//     3.24–3.90 and a 32×256 one (four subgroups over a single 32-row group)
//     3.35–4.05 — both 5–9% ahead, both rejected because they need
//     `n % 128 == 0` and `n % 256 == 0` respectively, and that does not pay for a
//     second kernel or for re-opening the unmasked-store design the GEOMETRY
//     section below rests on. The per-shape ordering is this machine's.
//
// ── LAYOUT CONVENTION (read before editing) ─────────────────────────────────
//
// METAL convention: `coopLoadT` on A, `coopLoadT` on B, `coopStoreT` on C —
// every operand. The `T` suffixes are not decoration. `coopLoad`/`coopLoadT`
// differ only by naga's `row_major = function_name.ends_with("T")`, but that
// bool reaches the backends as different things (Metal's
// `simdgroup_load(..., transpose_matrix)` versus SPIR-V's
// `RowMajorKHR`/`ColumnMajorKHR`), so the same WGSL does not mean the same
// thing on the two. The Vulkan siblings use the other convention deliberately;
// do not unify them. `src/coop_probe.rs` runs a non-commuting GEMM probe through
// this kernel on the actual device and `derive_matmul_compute` believes the
// result, so a device where this computes `b·a` reports ineligible instead of
// producing wrong numbers.
//
// The two subgroups are the second assumption: `workgroup_size(64)` is expected
// to be two 32-wide subgroups in `local_invocation_index` order, each executing
// its own cooperative instructions on its own column half. A device whose
// subgroup width is not 32 would partition the halves differently, compute the
// wrong product, and be rejected by that same probe.
//
// ── GEOMETRY ────────────────────────────────────────────────────────────────
//
//   * m must be a multiple of 64 and n a multiple of 64: a tile is stored whole,
//     because a cooperative store cannot be masked and a partial tile would write
//     past the output. `derive_matmul_compute` refuses the other geometries and
//     the multiply keeps the kernel it would have had — a tail is not rounded,
//     fused or written through a scalar fallback here. Every projection of the
//     text tower is a whole number of tiles (m = 8192 or 256, n ∈ 512…12288).
//   * k must be a multiple of 32, the K stage. A tail K stage would have to be
//     zero-filled inside the staging loop, which is a branch per staged element
//     on every tile of every call for a geometry the model never presents.
//
// Bind group: 0 = arena (f32, read-write: A in, C out), 1 = params (uniform),
// 2 = weights (f16, read-only).
//
// REQUIREMENTS: SHADER_F16 + EXPERIMENTAL_COOPERATIVE_MATRIX, and at least
// `8 KiB + 256 B` of workgroup storage (two 4 KiB A stage buffers + 256 B of
// zero lanes the accumulators are bootstrapped from). The output tile is NOT
// staged: the fragments are stored straight into the arena, which is what keeps
// the shader at 8.25 KiB and lets four workgroups be resident per GPU core — a
// 16 KiB f32 tile was the first form of this kernel and it ran 7× slower than
// `matmul_coop16` at these shapes (measured).
//
// The bias/activation epilogue is the one case that touches an output element
// twice: a fragment's lanes cannot be addressed individually, so the epilogue
// re-reads the tile it just stored after a `storageBarrier()`, applies the
// bias/activation in f32 and writes back. That is skipped entirely when the
// multiply has neither (`act_id == 0xFFFF`, `has_bias == 0`), which is what every
// projection measured above is.

enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,   // f32 element offset into `arena` (A is f32 there)
    b_off: u32,   // f16 element offset into `weights`
    c_off: u32,   // f32 element offset into `arena`
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;
const WG: u32 = 64u; // two 32-wide subgroups, each owning a 64×32 half

@group(0) @binding(0) var<storage, read_write> arena:   array<f32>;
@group(0) @binding(1) var<uniform>             params:  Params;
@group(0) @binding(2) var<storage, read>       weights: array<f16>;

// A stage: two ping-pong buffers of 64 rows × 32 K columns of f16 = 2048 f16 =
// 4 KiB each. The second buffer is what takes the refill off the critical path:
// the next panel's global loads are issued before this panel's MMAs, and the
// single `workgroupBarrier()` at the end of an iteration both publishes that
// panel and orders it behind the reads of the buffer it will overwrite.
var<workgroup> a_stage: array<f16, 4096>;
// The all-zero 8×8 f32 fragment the 32 accumulators are seeded from: WGSL has no
// zero-valued cooperative-matrix constructor this lowering supports, so one is
// loaded — the same bootstrap `matmul_coop16.wgsl` and `matmul_coop_f32.wgsl` use.
var<workgroup> zero_lanes: array<f32, 64>;

// Stage one 64×32 A panel, f32 arena → f16, into ping-pong buffer `buf`. Each
// invocation covers four adjacent K columns of one row, so the loads are a
// coalesced run the compiler can vectorise; the scalar one-column-per-invocation
// form of this loop measured 2.2 against 3.4 TFLOP/s for this one at the same
// tile. No bounds test: the geometry is a whole number of tiles and of K stages
// (see GEOMETRY), so every index is in range.
fn stage_a(buf: u32, k_off: u32, a_base: u32, row_base: u32, lid: u32) {
    let base = buf * (TILE_M * TILE_K);
    for (var s: u32 = 0u; s < 8u; s = s + 1u) {
        let idx = (lid + s * WG) * 4u;
        let r = idx / TILE_K;
        let c = idx % TILE_K;
        let g = a_base + (row_base + r) * params.k + k_off + c;
        let dst = base + r * TILE_K + c;
        a_stage[dst + 0u] = f16(arena[g + 0u]);
        a_stage[dst + 1u] = f16(arena[g + 1u]);
        a_stage[dst + 2u] = f16(arena[g + 2u]);
        a_stage[dst + 3u] = f16(arena[g + 3u]);
    }
}

// Exact GELU — A&S 7.1.26 erf, matches rlx-cpu's scalar_gelu.
fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    if (params.act_id == 0xFFFFu) { return v; }
    switch (params.act_id) {
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
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

@compute @workgroup_size(64)
fn matmul_coop_f16_f32acc(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    // This subgroup's half of the 64×64 tile: the whole 64 rows by 32 columns.
    let sg = lid / 32u;
    let sg_col = sg * 32u;

    if (lid < 64u) {
        zero_lanes[lid] = 0.0;
    }
    workgroupBarrier();

    // 32 accumulators — one per 8×8 sub-tile of this subgroup's half — all f32
    // while A and B stay f16: the hardware MMA form that keeps the tower's 512-
    // and 2048-term dot products out of f16's range.
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
    var acc_40: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_41: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_42: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_43: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_50: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_51: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_52: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_53: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_60: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_61: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_62: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_63: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_70: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_71: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_72: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);
    var acc_73: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&zero_lanes[0], 8u);

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    // Prime the ping-pong pair, then refill the other buffer ahead of the compute
    // that reads this one.
    stage_a(0u, 0u, a_base, row_base, lid);
    workgroupBarrier();
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        if (t + 1u < n_tiles) {
            stage_a((t + 1u) % 2u, (t + 1u) * TILE_K, a_base, row_base, lid);
        }
        let cur = t % 2u;

        // Four K sub-steps of 8, each the hardware MMA's own K.
        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 8u) {
            let a_0: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 0u * TILE_K + kk], TILE_K);
            let a_1: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 8u * TILE_K + kk], TILE_K);
            let a_2: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 16u * TILE_K + kk], TILE_K);
            let a_3: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 24u * TILE_K + kk], TILE_K);
            let a_4: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 32u * TILE_K + kk], TILE_K);
            let a_5: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 40u * TILE_K + kk], TILE_K);
            let a_6: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 48u * TILE_K + kk], TILE_K);
            let a_7: coop_mat8x8<f16, A> =
                coopLoadT<coop_mat8x8<f16, A>>(&a_stage[cur * (TILE_M * TILE_K) + 56u * TILE_K + kk], TILE_K);
            let b_row = b_base + (k_off + kk) * params.n + col_base + sg_col;
            let b_0: coop_mat8x8<f16, B> =
                coopLoadT<coop_mat8x8<f16, B>>(&weights[b_row + 0u], params.n);
            let b_1: coop_mat8x8<f16, B> =
                coopLoadT<coop_mat8x8<f16, B>>(&weights[b_row + 8u], params.n);
            let b_2: coop_mat8x8<f16, B> =
                coopLoadT<coop_mat8x8<f16, B>>(&weights[b_row + 16u], params.n);
            let b_3: coop_mat8x8<f16, B> =
                coopLoadT<coop_mat8x8<f16, B>>(&weights[b_row + 24u], params.n);

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
            acc_40 = coopMultiplyAdd(a_4, b_0, acc_40);
            acc_41 = coopMultiplyAdd(a_4, b_1, acc_41);
            acc_42 = coopMultiplyAdd(a_4, b_2, acc_42);
            acc_43 = coopMultiplyAdd(a_4, b_3, acc_43);
            acc_50 = coopMultiplyAdd(a_5, b_0, acc_50);
            acc_51 = coopMultiplyAdd(a_5, b_1, acc_51);
            acc_52 = coopMultiplyAdd(a_5, b_2, acc_52);
            acc_53 = coopMultiplyAdd(a_5, b_3, acc_53);
            acc_60 = coopMultiplyAdd(a_6, b_0, acc_60);
            acc_61 = coopMultiplyAdd(a_6, b_1, acc_61);
            acc_62 = coopMultiplyAdd(a_6, b_2, acc_62);
            acc_63 = coopMultiplyAdd(a_6, b_3, acc_63);
            acc_70 = coopMultiplyAdd(a_7, b_0, acc_70);
            acc_71 = coopMultiplyAdd(a_7, b_1, acc_71);
            acc_72 = coopMultiplyAdd(a_7, b_2, acc_72);
            acc_73 = coopMultiplyAdd(a_7, b_3, acc_73);
        }
        workgroupBarrier(); // publishes the next panel and frees the one just read
    }

    // Each subgroup stores its half straight into the arena: `stride` is the
    // output row stride, so the two halves tile the 64×64 output region.
    let srow = row_base;
    let scol = col_base + sg_col;
    let dst = c_base + srow * params.n + scol;
    coopStoreT(acc_00, &arena[dst + 0u * params.n + 0u], params.n);
    coopStoreT(acc_01, &arena[dst + 0u * params.n + 8u], params.n);
    coopStoreT(acc_02, &arena[dst + 0u * params.n + 16u], params.n);
    coopStoreT(acc_03, &arena[dst + 0u * params.n + 24u], params.n);
    coopStoreT(acc_10, &arena[dst + 8u * params.n + 0u], params.n);
    coopStoreT(acc_11, &arena[dst + 8u * params.n + 8u], params.n);
    coopStoreT(acc_12, &arena[dst + 8u * params.n + 16u], params.n);
    coopStoreT(acc_13, &arena[dst + 8u * params.n + 24u], params.n);
    coopStoreT(acc_20, &arena[dst + 16u * params.n + 0u], params.n);
    coopStoreT(acc_21, &arena[dst + 16u * params.n + 8u], params.n);
    coopStoreT(acc_22, &arena[dst + 16u * params.n + 16u], params.n);
    coopStoreT(acc_23, &arena[dst + 16u * params.n + 24u], params.n);
    coopStoreT(acc_30, &arena[dst + 24u * params.n + 0u], params.n);
    coopStoreT(acc_31, &arena[dst + 24u * params.n + 8u], params.n);
    coopStoreT(acc_32, &arena[dst + 24u * params.n + 16u], params.n);
    coopStoreT(acc_33, &arena[dst + 24u * params.n + 24u], params.n);
    coopStoreT(acc_40, &arena[dst + 32u * params.n + 0u], params.n);
    coopStoreT(acc_41, &arena[dst + 32u * params.n + 8u], params.n);
    coopStoreT(acc_42, &arena[dst + 32u * params.n + 16u], params.n);
    coopStoreT(acc_43, &arena[dst + 32u * params.n + 24u], params.n);
    coopStoreT(acc_50, &arena[dst + 40u * params.n + 0u], params.n);
    coopStoreT(acc_51, &arena[dst + 40u * params.n + 8u], params.n);
    coopStoreT(acc_52, &arena[dst + 40u * params.n + 16u], params.n);
    coopStoreT(acc_53, &arena[dst + 40u * params.n + 24u], params.n);
    coopStoreT(acc_60, &arena[dst + 48u * params.n + 0u], params.n);
    coopStoreT(acc_61, &arena[dst + 48u * params.n + 8u], params.n);
    coopStoreT(acc_62, &arena[dst + 48u * params.n + 16u], params.n);
    coopStoreT(acc_63, &arena[dst + 48u * params.n + 24u], params.n);
    coopStoreT(acc_70, &arena[dst + 56u * params.n + 0u], params.n);
    coopStoreT(acc_71, &arena[dst + 56u * params.n + 8u], params.n);
    coopStoreT(acc_72, &arena[dst + 56u * params.n + 16u], params.n);
    coopStoreT(acc_73, &arena[dst + 56u * params.n + 24u], params.n);

    // Epilogue. A fragment's lanes are not individually addressable, so a bias or
    // an activation is applied by re-reading the tile this workgroup just stored —
    // ordered by `storageBarrier()`, which is what makes a storage-space write
    // visible to the other invocations of the same workgroup. Skipped whole when
    // the multiply has neither, which is the case the rates above are measured on.
    if (params.has_bias == 0u && params.act_id == 0xFFFFu) {
        return;
    }
    storageBarrier();
    workgroupBarrier();
    for (var s: u32 = 0u; s < 64u; s = s + 1u) {
        let idx = lid + s * WG;
        let r = idx / TILE_N;
        let c = idx % TILE_N;
        let gc = col_base + c;
        let at = c_base + (row_base + r) * params.n + gc;
        var v: f32 = arena[at];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + gc];
        }
        arena[at] = apply_act(v);
    }
}
