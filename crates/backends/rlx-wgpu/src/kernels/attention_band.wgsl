// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Two-sided banded SDPA at prefill sizes: `MaskKind::Window(w)` at
// `seq_q == seq_k`, and a dense `MaskKind::None` step whose extents are square and
// whose many-row score matrix is past that arrangement's scratch cap. One pass
// over the keys inside each query block's band — the whole sequence when
// `w >= seq`, which is the dense form. Both are one dispatch: the plan writes
// `window = seq_k` for the dense step and lowers it to the same kernel
// (`params.mask_kind == 5`, `backend::attention_route::BAND_MASK_KIND`).
//
// What it replaces. Nothing served kind 5 before this entry point: the fused
// `attention.wgsl` entry knows kinds 0..4 and computes a *dense* pass for kind 5,
// which is silently wrong; the many-row route (`backend::attention_route`)
// decomposes into dense multiplies for the same reason. `plan_attention` routes
// `MaskKind::Window(_)` here instead and refuses, naming the shapes, any geometry
// this entry cannot hold. A dense step is taken here only where its many-row
// arrangement no longer fits its 512 MiB scratch cap *and* the cooperative dense
// square form (`backend::attention_route`'s `dense_coop`) does not take it: the
// stand's four global layers at 8 192 rows are that form's now, and this kernel is
// what they keep where the adapter cannot build or verify it — their previous
// arrangement was the fused one-thread-per-row walk (≈83 s for that cell, a
// reading of this workspace's `mahbot-inference` stand). Below
// the cap the f32 dense arrangement is affordable and its answer is the tighter
// one, so a dense step keeps it.
//
// The algorithm is FlashAttention-2's block form, not the fused entry's
// one-thread-per-query-row walk: a workgroup owns `BM` query rows of one
// `(batch, head)`, stages the key and value blocks it needs in workgroup memory
// (the K tile is reused for V once the block's scores are read) and keeps the
// running output, max and sum in **registers** as
//   m_new = max(m, max_k s_k);  alpha = exp(m - m_new)
//   l     = alpha·l + Σ_k exp(s_k - m_new)
//   O[d]  = alpha·O[d] + Σ_k exp(s_k - m_new)·V[k][d]
// per key *block* rather than per key, so one rescale of O covers `BK` keys
// instead of one key. `BM·C` threads cover the row block: thread `c` of a row
// owns `DPER` of the row's dims, which is `DPER` f32 of Q and the same of O in
// registers — `BM·head_dim / THREADS`, the same figure (32) at every head width.
//
// Four things about that inner loop are measured rather than stylistic:
//
//   1. Every loop over a thread's slots or over the key block is *expanded* by
//      the host (`kernels::expand_band_loops`) into statements whose index is a
//      literal. This is the largest lever in the file and it is not about the
//      arithmetic: a local array behind a loop index is thread-local memory in
//      the module wgpu builds, because naga gives every loop a termination guard
//      the backend cannot see through, so `q4[j]`, `o4[j]` and `score[kk]` each
//      became a load and a store. With nothing else changed, the slot loops cost
//      2 223 ms at the global cell as loops and 1 196 ms expanded; expanding the
//      key loops as well reads 656 ms; the `part_s` reduce's dependent chain
//      (below) another ~80 ms. Nothing here is Apple-specific — the same naga
//      output feeds Vulkan and DX12 — but the ratios are this rig's, and a
//      backend whose compiler already unrolls would gain less.
//   2. The Q/O slices are `vec4<f32>` and the score's dot product is four
//      independent accumulation chains. One scalar accumulator per row was the
//      first form of this kernel — a chain of `DPER` dependent FMAs, one per
//      dim, is ~30× off the machine's own multiply rate.
//   3. The tile is `vec4<f16>` a slot (or `vec4<f32>`, below), and a row's chunk
//      threads own *interleaved* slots (`j·C + c`): the lanes of a row then read
//      consecutive slots, and a thread's four dims are four consecutive arena
//      words, which is what the Q read and the O write coalesce across.
//   4. The block's scores are exchanged through `part_s`/`s_sh` once, not summed
//      redundantly by every chunk thread: one thread a key adds the `C` partials
//      and applies the cap and the mask, and the row's chunk threads read the
//      result back. The redundant form cost `C` reads a thread *per key*. The
//      `C`-long sum itself is four accumulators, not one: one accumulator is a
//      dependent chain of 8 or 16 shared-memory latencies per key, and splitting
//      it read 637 → 557 ms at the global cell.
//
// The band is the point. For query block `[q0, q0+BM)` the walked key range is
// `[max(0, q0 - w), min(seq, q0 + BM + w))`: a key block wholly outside it is
// never touched, and the in-band condition `|q - k| <= w` is applied per row
// inside the edge blocks. At the fork's encoder geometry (8 192 rows, w = 512,
// `BM = 64`) an interior query block walks 1 088 keys of 8 192, 1 052 on average
// over the clipped ends — 7.8× less work than a dense pass, and the figure
// `tests/attention_band_cost.rs` prints. `w >= seq` degenerates to "no band": the
// range is the whole sequence and the mask never fires, which is the global
// layers' shape and must equal `MaskKind::None` exactly.
//
// Staging is f16 with f32 accumulation — 16-bit operands, 32-bit accumulation,
// which is what the reference engine does. The same body is built with a
// `vec4<f32>` tile by `kernels::bind_band_stage` on an adapter that does not
// advertise `Features::SHADER_F16`, and behind `RLX_WGPU_ATTENTION_BAND_F32` on
// one that does: one source, one substituted token, so the two can never drift.
// The f32 tile is also the measured baseline the f16 form is held against.
//
// Measured on this machine (Apple M3 Pro, 14-core GPU, wgpu 30 over Metal),
// `tests/attention_band_cost.rs`, best of seven rounds after three warm-ups, one
// `Session` call per round — so a reading includes the run's submission and
// readback floor, which the same file measures separately (43–68 ms for the
// biggest cell's 64 MiB and 10–14 ms for the sliding cell's 32 MiB, with another
// process sharing the GPU while these were taken). "Before" is the 256-thread,
// 32-dims form this file shipped with — the floor every adapter can launch, and
// still what a sub-512 adapter runs — and "after" is the 512-thread wide form
// this rig's Metal admits.
// Both cells of the pair were re-taken twice, back to back, and the spreads below
// are the two runs' (the sliding cell's single-run readings move with load):
//
// | geometry                            | before (f16) | after (f16) | after (f32) |
// |-------------------------------------|--------------|-------------|-------------|
// | 8 192 rows, 4 heads, d = 256, ±512  | 46.3–46.4 ms, 0.74 TFLOP/s | 41.8–41.9 ms, 0.84 TFLOP/s | 69.4 ms, 0.51 TFLOP/s |
// | 8 192 rows, 4 heads, d = 512, none  | 564.9–566.2 ms, 0.97 TFLOP/s | 534.9–536.8 ms, 1.02 TFLOP/s | 1 062 ms, 0.52 TFLOP/s |
// | 256 rows, 4 heads, d = 256, w = 512 | 1.37 ms | 1.38 ms | 1.71 ms |
//
// The two big cells are 1.10× and 1.05×. An absolute reading taken later, with
// another process on the GPU, moves by more than that — compare the pair, not one
// run against another. The f32 tile loses 12% at the global cell: the form's
// per-key partials are `THREADS · BK · 4` B, so at 32 KiB the f32 tile's block
// halves to 4 where the f16 one keeps 8, and the walk pays in barriers. It is the
// fallback on an adapter without half precision and the baseline for the f16
// column; the form is one per width so the dispatch grid cannot disagree with the
// module, and the f16 win is the one that ships. The encoder's own per-layer
// shares are arithmetic on the table, not readings: the pass lives in the sibling
// `mahbot-inference` repo.
//
// The form table is the other half of the change, and it is what says the walk is
// not bandwidth-bound. A larger `BM` halves or quarters the bytes a call re-reads
// (`blocks · heads · keys walked · head_dim · 4 · 2`, `blocks = ceil(seq / BM)`),
// and the wall clock follows only where the per-thread registers did not move:
//
// | threads | dims/thread | BM at d=512 | global | sliding | small |
// |---------|-------------|-------------|--------|---------|-------|
// | 256     | 32          | 16          | 565 ms | 46.3 ms | 1.37 ms |
// | 512     | 32          | 32          | 535 ms | 41.8 ms | 1.38 ms |
// | 256     | 64          | 32          | 757 ms | 58.4 ms | 1.49 ms |
// | 512     | 64          | 64          | 763 ms | 67.9 ms | 2.31 ms |
// | 256     | 128         | 64          | 1 498 ms | 118.4 ms | 3.03 ms |
// | 512     | 128         | 128         | 1 560 ms | 133.9 ms | 3.89 ms |
// | 1 024   | 32          | 64          | 1 385 ms | 106 ms | — |
//
// 256/64 and 512/32 read the same 34.4 GB at d = 512 and differ by 41%: the wide
// chunk doubles a thread's Q and O to 128 f32, which halves how many workgroups a
// core holds, and that costs more than the bytes save. 512/32 is the one row that
// doubles `BM` with a thread's registers untouched. 1 024/32 quarters the bytes
// and loses 2.4× — one workgroup a core, and the key block squeezed to 4.
//
// What those readings say, and what they do not:
//
//   * The band is still most of the win over the fused walk. The same step with
//     no band and `MaskKind::None` at the sliding cell's shape read 7 897 ms
//     before this route existed (the fused walk; the many-row arrangement's score
//     matrix is 1 GiB there, past its cap), against 242 ms now — the same pass,
//     served here with the band switched off. That is 32× wall-clock at equal
//     output. Both figures are one run of the previous/current tree on this rig.
//   * The kernel is not shared-memory bound. With the slot loops expanded but not
//     the key loops, replacing every staged-tile read in both phases with a
//     register read moved the global cell 947 → 635 ms, i.e. the tile's loads
//     were ~33%; after the key loops and the reduce chain the same experiment
//     moves 557 → 545 ms, ~2%. What remains is the instruction stream itself:
//     ~2.2 issued instructions per FMA at head width 512 (one tile load, one
//     `vec4` FMA and two half→float converts per 4 dims, plus the per-key
//     exchange), against a device that this rig measures at 3.8 TFLOP/s of pure
//     f32 FMA through the same stack — an out-of-tree probe (one compute shader,
//     eight named `vec4` accumulators, no memory in the loop), not a harness in
//     this tree and not re-takeable from it. At 1.02 TFLOP/s the kernel sits at
//     ~27% of that peak and ~60% of the ~1.7 TFLOP/s its instruction mix allows.
//   * Reuse per *thread* is the lever the instruction mix points at, and it has
//     been taken to its limit and refused: a thread that owns two rows issues
//     `1 load + 2 converts` once instead of twice. That was built (a `RPT`
//     rows-per-thread axis, expanded by the host like the other two) and measured
//     on the 256-thread tile, on both big cells of `tests/attention_band_cost.rs`:
//
//     | arrangement                                   | global | sliding | small |
//     |-----------------------------------------------|--------|---------|-------|
//     | shipped then: 256 threads, 32 dims, 1 row     | 561 ms | 48 ms   | 1.39 ms |
//     | 2 rows a thread, 32 dims (BM 32 at 512)       | 773 ms | 62 ms   | 1.51 ms |
//     | 2 rows a thread, 16 dims (BM 16 at 512)       | 704 ms | 47 ms   | 1.36 ms |
//     | 2 rows, 32 dims, 128 threads (BM 16 at 512)   | 831 ms | 68 ms   | 1.71 ms |
//     | 1 row, 16 dims (BM 8 at 512)                  | 717 ms | 60 ms   | 1.55 ms |
//
//     The instruction count fell exactly as the mix predicts: the first of those
//     rows does twice the work a workgroup and runs at `2 · 561/773 = 1.45×` the
//     shipped rate, against 2.25/1.63 = 1.38× predicted. The *wall clock* got
//     worse, because a second row doubles a thread's Q + O to 128 f32 and that
//     halves how many workgroups a core holds. That is the same cost the wide
//     chunk pays in the form table above — and it is why the shipped form reaches
//     `BM = 32` at 512 by *doubling the workgroup* (512 threads, 32 dims a thread,
//     64 f32 of Q + O) rather than by doubling a thread's rows: the same `BM`, the
//     same bytes, and 535 ms against 773.
//   * Barriers are the other priced term. Forcing `BK = 4` at head width 512 on
//     the 256-thread tile (8.4 KiB of workgroup storage instead of 16.9) read
//     642 ms against 561 at `BK = 8`, and 16 keys a block would need 33.8 KiB,
//     above this adapter's 32 768 B. So the tile sits at the largest block the
//     storage allows, which is what `kernels::attention_band_key_block_within`
//     encodes — 8 keys a block at every width on this rig's 32 KiB, and 4 on
//     WebGPU's guaranteed 16 KiB.
//   * `THREADS = 512` is a lever at this rig's geometries: taken back to back
//     twice, 256 against 512 reads 564.9–566.2 against 534.9–536.8 ms at the
//     global cell and 46.3–46.4 against 41.8–41.9 at the sliding one. The width
//     is an add-on, not the base: `THREADS` comes from the adapter
//     (`attention_band_threads`), which takes 512 where
//     `maxComputeInvocationsPerWorkgroup` admits it — 1 024 on this rig's Metal,
//     and what `fft_gpu.wgsl` already asks for — and the 256-thread floor
//     otherwise. The dispatch grid takes the same figure from the same limit, so
//     the two cannot disagree; below the floor, which no conforming WebGPU
//     adapter is, the kernel panics naming the limit.
//   * The absolute rate is above this backend's own tiled f32 matmul, which reads
//     0.725 TFLOP/s at 8 192×512×8 192 on the same machine and harness
//     (`Op::MatMul`, one call) — the attention kernel's two multiplies per head
//     are the same arithmetic, so that is the honest reference point. Nothing in
//     the tables establishes a bound for another adapter: every figure is one
//     M3 Pro, so re-take them on the adapter in question.
//   * The 256-row cell's 0.19 TFLOP/s is a small-cell artefact, not a rate: it
//     visits 1 024 keys in all, and the ~11 ms run floor is most of its reading.
//
// Which part of that is portable, and which is this rig's:
//
//   * Portable by construction: the expansion (WGSL has no unroll, and naga's
//     loop guard reaches every wgpu backend the same way), the tile choice read
//     off the adapter's own `maxComputeWorkgroupStorageSize` (a 16 KiB adapter
//     takes `BK = 4` and computes the same answer), the f16 tile with four-lane
//     accumulation, the one body for both staging widths, and the dead-row guard
//     below. The *sizes* of the wins are not portable: 1.05× and 1.10× are one
//     M3 Pro's, and a backend whose driver already promotes local arrays to
//     registers would gain less from the expansion. Whether another adapter gains
//     from the 512-thread form is the same question on its own register file and
//     core count: the form needs 64 f32 a thread whatever the workgroup.
//   * This device's: the absolute rates (1.02 TFLOP/s at d = 512 against an
//     out-of-tree 3.8 TFLOP/s FMA probe on the same chip), the 5–10% the wider
//     workgroup buys, and the crossover at 1 024 threads.
//   * Not established at all: any bound for another adapter, any figure for a
//     head width the plan admits but this table does not cover (128 is pinned for
//     parity only), and the in-graph per-layer shares, which are arithmetic on
//     this table and not a reading of the model.
//
// The one lever this file does not take, and why it is not a rearrangement of it:
// a thread's 4 dims cost `1 tile load + 2 half→float converts + 4 fma` — 1.75
// instructions an MAC for the inner loop, ~2.2 with the per-key exchange counted
// — so at 3.8 TFLOP/s of pure FMA this rig caps the walk at ~1.7 TFLOP/s, and the
// 1.02 it reads is ~60% of that. A perfect schedule of this same instruction
// stream lands at ~0.32 s a global layer at best, against the 0.30 the encoder's
// pass needs, and only if every stall vanished. Below 1.75 is hardware matrix
// multiply (`coop_mat8x8<f16>` operands with an f32 accumulator), and that is a
// *device* path, not a portable one: `coop_probe::f16_f32acc_path_ok()` exists
// precisely because the operand order is a property of the adapter's
// cooperative-matrix implementation. It would be a second module behind a probe
// with this one as the fallback, and it is not in this tree — the softmax sits
// between the two products and would have to come back out of a fragment, and the
// running O rescale per key block becomes a store and reload of the output tile.
// Neither is measured here, and a wrong attention is worse than a slow one.
//
// What the two readings above say that route's ceiling is, at the global cell:
// the products are 26.84 + 28.44 ms a head through the shipping cooperative
// kernel — 2.56 and 2.42 TFLOP/s at `m = 8192, k = 512, n = 8192` and
// `m = 8192, k = 8192, n = 512`, one dispatch, no readback, this rig
// (`tests/encoder_gemm_cost.rs`) — so 221 ms for four heads, under half the 535
// spent here. The fused form then has to move its own operands, and that is where
// it closes or does not: with O resident (64 f32 a lane at 32 rows over 256
// threads) Q is re-staged per key block, so 34.4 GB of K/V plus 8.6 GB of Q, and
// at the ~110 GB/s that GEMM sustains beside its own multiplies that is 391 ms
// before a single `exp`. The arrangement that would reach the 0.30 s an encoder
// layer needs holds 64 rows instead — 128 f32 a lane of O, the residency the row-
// split experiment above measured as a 1.4× loss. That trade, and not the
// fragment plumbing, is what a cooperative attention has to settle first.
//
// Parity against the CPU reference (`tests/attention_band_parity.rs`, f16
// staging, max-abs and cosine over the full output): 300 rows / w = 32 /
// d = 64 reads 1.73e-4 and 0.999999493; 512 rows / w = 8 / d = 256 reads 2.89e-4
// and 0.999999992; the sliding cell reads 5.01e-5 and 0.999999263; the global
// cell reads 1.29e-5 and 0.999991535; a dense square step past the scratch cap
// (1 536 rows, 64 heads, d = 64, no mask) reads 2.75e-5 and 0.999992555 — the
// same arithmetic with the band switched off. The reference stages nothing in
// f16, so the residue is the tile's rounding as well as summation order, and the
// bar the parity test holds is 5e-3 — an order above every reading, so a
// regression that dropped an in-band key or mis-mapped a GQA head fails it by far
// more. The other cases read the same way: a GQA group of four *under* a band
// 1.03e-4 and 0.999997566; a window narrower than one key block 3.89e-4 and
// 0.999999963; head width 128, the four-chunk boundary of the reduce's
// accumulator form, 1.14e-4 and 0.999999438. A row past the sequence computes
// nothing (see the guards in the walk), which the 300-row case's ragged last
// block exercises.
//
// Tile parameters, and why:
//   * `THREADS = 512` and `DPER = 32`: a workgroup's rows are `THREADS / C`, so
//     the thread count is the lever on `BM` that does not touch a thread's
//     registers, and the form table above is what chose both. A thread's Q and O
//     are 32 f32 each whatever the head width.
//   * `C = head_dim / DPER` threads a row and `BM = THREADS / C` rows a
//     workgroup — 256 rows at head width 64, 128 at 128, 64 at 256, 32 at 512.
//     `C` is never 1: the reduce sums a row's chunks in pairs at `C = 2` and in
//     fours above, and a one-chunk row would read the next row's partials. A width
//     that would build one is refused by the plan (`ATTENTION_BAND_MIN_HEAD_DIM`).
//   * `BK` staged keys: the largest power of two in `{32, 16, 8, 4, 2}` whose K
//     tile (`BK·head_dim` f16) plus the score partials (`THREADS·BK` f32) and the
//     block's scores (`BM·BK` f32) fits the adapter's
//     `maxComputeWorkgroupStorageSize` less a sixteenth — 8 keys at every width on
//     this rig's 32 768 B, 4 on WebGPU's *guaranteed* 16 384 B. That degradation
//     is of the walk and not of the answer: a smaller block only means more
//     barriers. The partials alone are `THREADS·BK·4` bytes, which is what keeps
//     the block from being larger.
//
// A row past the walked extent — the last block of a sequence that is not a
// multiple of `BM` — runs none of the arithmetic: its threads skip the score
// dot, the chunk reduce and the P·V accumulation and only reach the barriers, so
// no compute is spent on rows that carry no input.
// Its Q load and its address math stay in bounds through `q_abs`, which is
// clamped to the last live row.
//
// Fully-masked rows do not arise under this mask kind (a window always contains
// the query itself), but the behaviour is defined: a row with `l == 0` — an empty
// key sequence, which `plan_attention` refuses before it gets here — writes
// zeros rather than a 0/0 NaN. A masked key scores `-3.4e38` (the sentinel the
// fused entry's `mask_score` uses) and contributes `exp(-3.4e38 - m) = 0`; a block
// wholly outside a row's band therefore leaves (m, l, O) untouched, and its
// `exp(-3.4e38 + 3.4e38)` prefix is scaled to zero by the first in-band block's
// `alpha`.
//
// `attn_logit_softcap` is applied over the scaled score and before the mask, the
// order `attention.wgsl`'s `mask_score` uses (0 disables it, which is this
// model's value).
//
// Layout: Q/K/V/out are read and written through the per-axis strides in
// `AttentionParams`, so `[B, H, S, D]` and `[B, S, H, D]` both work with no
// transpose in front. GQA maps the query head to its KV head the way the fused
// entry does (`h / (heads / kv_heads)`).
//
// The module-scope constants below are prepended by
// `kernels::attention_band_kernel`, which formats them from the head width the
// step declares — the file is not a standalone module. So is `enable f16;`: a
// global directive has to lead the module, ahead of the constants. `%STAGE%` is
// the one token the two staging widths differ in (`kernels::bind_band_stage`).

// The dims a thread owns are `SLOTS` four-wide slots, interleaved across the
// row's chunk threads: thread `c` of a row owns the dims of slots `j·C + c`. Two
// consequences, both load-bearing: the lanes of a row read consecutive slots of
// the staged tile (a `vec4` each, so a group of them covers whole banks), and a
// thread's four dims are four *consecutive* arena words, which is what the Q read
// and the output write coalesce across.
//
// Every loop over `j` and over the key block's `kk` below is a `// @unroll`
// block rather than a WGSL loop: the host repeats those statements with the
// index as a literal (`kernels::expand_band_loops`). A dynamically indexed
// local array is *not* a register array in the module wgpu builds — naga emits
// every loop with a termination guard that the backend cannot see through, so
// `q4[j]`, `o4[j]`, `score[kk]` and the rest land in thread-local memory and
// every use becomes a load and a store. Measured on the global cell above, with
// nothing else changed: 2 223 ms as loops, 656 ms as unrolled statements. The
// same mechanism is available to every backend behind wgpu, so this is not a
// property of one GPU family.
const SLOTS: u32 = DPER / 4u;

// The score a fully-masked key carries: `mask_score`'s own sentinel in
// `attention.wgsl`, so the two entries' masked rows read alike.
const NEG: f32 = -3.4e38;

// `kernels::AttentionParams`, verbatim — the uniform layout the host writes.
// The two move together; the fields this entry reads are the geometry, the
// strides, the scale and the cap (`mask_off` / `splits` / the mask strides are
// the other entries').
struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,

    out_off: u32,
    mask_off: u32,
    mask_kind: u32,
    scale_bits: u32,
    window: u32,
    seq_q_stride: u32,
    seq_k_stride: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    kv_heads: u32,
    v_head_dim: u32,
    logit_softcap_bits: u32,

    q_batch_stride: u32, q_head_stride: u32, q_seq_stride: u32,
    splits: u32,
    k_batch_stride: u32, k_head_stride: u32, k_seq_stride: u32, _pad_k: u32,
    v_batch_stride: u32, v_head_stride: u32, v_seq_stride: u32, _pad_v: u32,
    o_batch_stride: u32, o_head_stride: u32, o_seq_stride: u32, _pad_o: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
// The decode entries' key-range partials. This entry never splits, but the
// module keeps the three-binding layout every attention entry shares, so one
// bind group serves whichever entry the run picks.
@group(0) @binding(2) var<storage, read_write> attention_parts: array<f32>;

// The staged key block, four dims a slot, then the value block in the same
// buffer: the block's scores are in `s_sh` by the time V is needed, so the tile
// is written once per phase and no second buffer is allocated.
var<workgroup> stage: array<%STAGE%, BK * C * SLOTS>;
// Per-chunk score partials, `[(row·C + chunk)·BK + key]`: what a chunk thread
// could not sum alone (the dot product spans the whole row) waits here for one
// barrier.
var<workgroup> part_s: array<f32, BM * C * BK>;
// The block's scores, once per (row, key): one thread a key sums the chunks,
// applies the cap and the mask, and every chunk thread of the row reads them
// back — `C` reads a row instead of the `C·BK` a redundant sum would take.
var<workgroup> s_sh: array<f32, BM * BK>;

@compute @workgroup_size(THREADS)
fn attention_band(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    // Backstops, not refusals: `plan_attention` refuses a zero sequence, a zero
    // head count and a width this module was not built for, so a step that
    // reaches here has none of them.
    if (params.seq_q == 0u || params.heads == 0u) { return; }

    let c = lid % C;
    let r = lid / C;
    let hd4 = C * SLOTS;   // the staged tile's slot count a key row occupies
    let seq = params.seq_q;
    let q0 = wid.x * BM;
    let bh = wid.y;
    let b = bh / params.heads;
    let h = bh % params.heads;
    let w = params.window;
    let scale = bitcast<f32>(params.scale_bits);
    let cap = bitcast<f32>(params.logit_softcap_bits);

    // A row past the walked extent keeps its threads in the workgroup (the
    // barriers below are its whole cost) but reads and writes no output; its
    // addresses are clamped so the register load stays in bounds.
    let live = q0 + r < seq;
    let q_abs = min(q0 + r, seq - 1u);

    // GQA/MQA: the query head's KV head, the fused entry's own mapping.
    let kv_denom = max(params.kv_heads, 1u);
    let kv_h = h / max(params.heads / kv_denom, 1u);
    let k_base = params.k_off + b * params.k_batch_stride + kv_h * params.k_head_stride;
    let v_base = params.v_off + b * params.v_batch_stride + kv_h * params.v_head_stride;
    let q_base = params.q_off
        + b * params.q_batch_stride
        + h * params.q_head_stride
        + q_abs * params.q_seq_stride
        + c * 4u;
    let o_base = params.out_off
        + b * params.o_batch_stride
        + h * params.o_head_stride
        + q_abs * params.o_seq_stride
        + c * 4u;

    // Q, O and the block's scores live in registers for the whole walk. The four
    // lanes of a slot are one `vec4`, so the score's dot product is four
    // independent accumulation chains rather than one serialized one.
    var q4: array<vec4<f32>, SLOTS>;
    // @unroll j
    { let o = q_base + %j%u * 4u * C; q4[%j%u] = vec4<f32>(arena[o], arena[o + 1u], arena[o + 2u], arena[o + 3u]); }
    // @end
    var o4: array<vec4<f32>, SLOTS>;
    // @unroll j
    o4[%j%u] = vec4<f32>(0.0);
    // @end
    var m_run: f32 = NEG;
    var l_run: f32 = 0.0;

    // The keys this block's rows can reach: `[q0 - w, q0 + BM + w)` clipped to
    // the sequence. A key block outside it is never read.
    let k_start = select(q0 - w, 0u, w > q0);
    let k_end = select(min(seq, q0 + BM + w), seq, w >= seq);

    var k0 = k_start;
    loop {
        if (k0 >= k_end) { break; }
        let kn = min(BK, k_end - k0);

        // ── Stage the K block ────────────────────────────────────────────────
        for (var i: u32 = lid; i < BK * hd4; i = i + THREADS) {
            let kk = i / hd4;
            let d0 = (i % hd4) * 4u;
            var v4 = vec4<f32>(0.0);
            if (kk < kn) {
                let o = k_base + (k0 + kk) * params.k_seq_stride + d0;
                v4 = vec4<f32>(arena[o], arena[o + 1u], arena[o + 2u], arena[o + 3u]);
            }
            stage[i] = %STAGE%(v4);
        }
        workgroupBarrier();

        // ── This chunk's share of every score in the block ───────────────────
        // One statement a key, repeated by the host: `part_s` and the tile are
        // addressed with the key as a literal.
        //
        // A thread whose row is past the sequence runs none of this: the last
        // block of a sequence that is not a multiple of `BM` carries rows with
        // no input, and they must not pay for a dot product whose result is
        // discarded. Every thread still reaches the barriers below, and the
        // tile's staging above is the block's work rather than a row's.
        if (live) {
            // @unroll kk
            {
                var acc = vec4<f32>(0.0);
                // @unroll j
                acc = acc + q4[%j%u] * vec4<f32>(stage[%kk%u * hd4 + %j%u * C + c]);
                // @end
                part_s[(r * C + c) * BK + %kk%u] = acc.x + acc.y + acc.z + acc.w;
            }
            // @end
        }
        workgroupBarrier();

        // ── Stage the V block over K, and sum one key a chunk ────────────────
        //
        // Both run between the same pair of barriers: the K tile is dead once the
        // partials are in `part_s`, and the reduce touches only `part_s` and
        // `s_sh`, so one barrier serves V's readers and the scores' together.
        for (var i: u32 = lid; i < BK * hd4; i = i + THREADS) {
            let kk = i / hd4;
            let d0 = (i % hd4) * 4u;
            var v4 = vec4<f32>(0.0);
            if (kk < kn) {
                let o = v_base + (k0 + kk) * params.v_seq_stride + d0;
                v4 = vec4<f32>(arena[o], arena[o + 1u], arena[o + 2u], arena[o + 3u]);
            }
            stage[i] = %STAGE%(v4);
        }
        if (live) {
            for (var kk: u32 = c; kk < BK; kk = kk + C) {
                // Four accumulators over the row's chunks, two at the two-chunk
                // width: one accumulator made the sum a dependent chain of `C`
                // shared-memory latencies (637 → 557 ms at the global cell), and
                // `C` is `2, 4, 8` or `16` — `C == 1` is not a form the kernel is
                // built with (`attention_band_form`), and the pair would read the
                // next row's partials there. Both arms are compile-time.
                var s0: f32 = 0.0;
                var s1: f32 = 0.0;
                var s2: f32 = 0.0;
                var s3: f32 = 0.0;
                if (C >= 4u) {
                    for (var cc: u32 = 0u; cc < C; cc = cc + 4u) {
                        s0 = s0 + part_s[(r * C + cc) * BK + kk];
                        s1 = s1 + part_s[(r * C + cc + 1u) * BK + kk];
                        s2 = s2 + part_s[(r * C + cc + 2u) * BK + kk];
                        s3 = s3 + part_s[(r * C + cc + 3u) * BK + kk];
                    }
                } else {
                    for (var cc: u32 = 0u; cc < C; cc = cc + 2u) {
                        s0 = s0 + part_s[(r * C + cc) * BK + kk];
                        s1 = s1 + part_s[(r * C + cc + 1u) * BK + kk];
                    }
                }
                var s: f32 = (s0 + s1) + (s2 + s3);
                s = s * scale;
                // The op's order: the cap is over the scaled score, the mask after
                // it — `mask_score`'s order in `attention.wgsl`.
                if (cap > 0.0) { s = cap * tanh(s / cap); }
                let ki = k0 + kk;
                if (kk >= kn) {
                    s = NEG;
                } else if (w < seq && (ki + w < q_abs || q_abs + w < ki)) {
                    s = NEG;
                }
                s_sh[r * BK + kk] = s;
            }
        }
        workgroupBarrier();

        // ── Rebase the running triple, then accumulate P·V ───────────────────
        //
        // Guarded like the score above: a dead row has no `s_sh` entry of its
        // own, so it reads none and accumulates none.
        if (live) {
            var score: array<f32, BK>;
            var bmax: f32 = NEG;
            // @unroll kk
            { score[%kk%u] = s_sh[r * BK + %kk%u]; bmax = max(bmax, score[%kk%u]); }
            // @end
            let m_new = max(m_run, bmax);
            // `m_run == m_new == NEG` leaves `alpha = exp(0) = 1` and an untouched
            // O; a later in-band block is what rebases it.
            let alpha = exp(m_run - m_new);
            l_run = l_run * alpha;
            // @unroll j
            o4[%j%u] = o4[%j%u] * alpha;
            // @end
            m_run = m_new;
            // @unroll kk
            {
                var p: f32 = 0.0;
                if (score[%kk%u] > NEG) { p = exp(score[%kk%u] - m_run); }
                l_run = l_run + p;
                // @unroll j
                o4[%j%u] = o4[%j%u] + p * vec4<f32>(stage[%kk%u * hd4 + %j%u * C + c]);
                // @end
            }
            // @end
        }
        // The next block's K staging overwrites this tile.
        workgroupBarrier();
        k0 = k0 + BK;
    }

    if (live) {
        // Defined for `l == 0` (an empty sequence): zeros, not the 0/0 NaN.
        let inv = select(0.0, 1.0 / l_run, l_run > 0.0);
        // @unroll j
        { let o = o_base + %j%u * 4u * C; arena[o] = o4[%j%u].x * inv; arena[o + 1u] = o4[%j%u].y * inv; arena[o + 2u] = o4[%j%u].z * inv; arena[o + 3u] = o4[%j%u].w * inv; }
        // @end
    }
}
