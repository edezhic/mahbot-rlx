// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether a many-row `Op::Attention` step takes the dense arrangement — one
//! `Step::Matmul` of the scores, a `Step::AttentionMask` and a `Step::Softmax`
//! over them, and one `Step::Matmul` by V — in place of one
//! `Step::Attention` in which a single thread walks the whole key sequence
//! alone. Shapes, strides and the nodes that produce them — never a device — so
//! it is testable without one, and [`plan_attention`] the one derivation of it
//! from a node: the reserve in [`crate::backend`] and the `Op::Attention` arm of
//! `compile::lower` both read the plan it returns.
//!
//! A rank-4 operand's layout — `[B, H, S, D]` or `[B, S, H, D]` — is read from
//! the graph wherever the operand's own extents leave both readings possible, by
//! [`rank4_geometry`]: the extents settle it wherever they can, and the one extent
//! they cannot is refused by name rather than guessed — still device-free, the
//! walk over the producing nodes included.
//!
//! Why: the fused kernel runs one thread per (batch, head, query row), each
//! reading all of K and V itself, so the work never fills the machine as the row
//! count grows. The route was admitted from 32 query rows and is admitted from
//! two now: a single live query row stays on the existing one-row arrangement
//! (the decode kernel at this cache geometry), two or more take the dense
//! arrangement, and the key length does not enter as a term — the scratch
//! conjunct already puts an upper bound on the key extent, so a key floor would
//! enlarge the admission, though it would also avoid the empty-cache cell's
//! 9.9 ms — the price of the single-term admission, paid once per generation.
//!
//! The figures come from the fork's own measurement stand, an out-of-tree
//! harness (`l59_stand --part decode`) that patches this checkout's crates in
//! and reads a cell in five rounds inside one window; they cannot be re-taken in
//! this checkout. The body is Qwen3-ASR-0.6B's 28-layer attention — 16 heads of
//! width 128 over 8 KV heads, one batch — on M3 Pro / Metal; milliseconds. `band`
//! compiles the whole cache, `rows` the window's rows instead; the `interleaved`
//! session read the two builds back to back — `before → after` is that order,
//! and a one-row cell's single figure or `a / b` pair is the same reading on the
//! two builds — and `31-row control` and `empty cache` are the separate control
//! readings:
//!
//! | session         | form | cells | rows | ms            |
//! |-----------------|------|-------|------|---------------|
//! | interleaved     | band |   512 |    2 | 502.1 → 189.5 |
//! | interleaved     | band |  2048 |    2 | 2040 → 548.1  |
//! | interleaved     | rows |   512 |    2 | 515.2 → 147.0 |
//! | interleaved     | rows |  2048 |    2 | 1880 → 451.4  |
//! | interleaved     | band |   512 |    1 | 86.0          |
//! | interleaved     | rows |   512 |    1 | 65.1          |
//! | interleaved     | band |  2048 |    1 | 260.2 / 287.6 |
//! | interleaved     | rows |  2048 |    1 | 193.3 / 193.6 |
//! | 31-row control  | band |   512 |   31 | 527.1 → 177.6 |
//! | empty cache     | band |     0 |    2 | 33.3 → 43.2   |
//!
//! The gain from the table's own cells is x2.65 and x3.72 for the band form
//! (502.1 → 189.5 over 512 cells, 2040 → 548.1 over 2048) and x3.50 and x4.16
//! for the rows form (515.2 → 147.0, 1880 → 451.4). Against two single-row
//! passes from the same one-row rows, the two-row figure sits 10% above the band
//! form's bound over 512 cells (189.5 against 2 × 86.0) and 13% above the rows
//! form's (147.0 against 2 × 65.1); over 2048 cells the band form's 548.1 is 5%
//! above 2 × 260.2 and 5% inside 2 × 287.6, and the rows form's 451.4 is 17%
//! above either 2 × 193.3 or 2 × 193.6. That residue — the packed multiplies
//! switching from per-row GEMV to row-grouped GEMM — is the second admission,
//! recorded as the next subject and not tuned around.
//!
//! The classes it re-routes, one sentence each, with the tests that pin each
//! route and its parity:
//!
//! - the cached decode body at two live rows, in
//!   `attention_cached_decode_route::a_two_row_cached_decode_step_takes_the_route`,
//!   with the one-row decode-kernel control in
//!   `a_one_row_cached_decode_step_keeps_the_decode_kernel`;
//! - a step whose query extent differs from its key extent, pinned in
//!   `attention_rows_route` by
//!   `a_causal_step_shorter_in_keys_than_rows_takes_the_route`;
//! - a chunked prefill or padded walk compiled above its walked extent, in
//!   `attention_cached_decode_route::a_padded_run_keeps_the_live_rows_answers`;
//! - the crate's own attention fixtures from two rows up, in
//!   `attention_rows_route::route_mask_kinds_match_cpu`,
//!   `attention_softcap_route::the_dense_arrangement_caps_ahead_of_the_mask`,
//!   `attention_v_head_dim_parity::wgpu_attention_asymmetric_v_head_dim_parity`
//!   and `attention_long_seq_precision::attention_precision_across_key_lengths`.
//!
//! The embeddings path walks one live row, so it never meets this admission, and
//! its digests are unchanged.
//!
//! # The cooperative dense square form
//!
//! A dense `MaskKind::None` step whose extents are square and which is 2 048 query
//! rows or longer takes the same three passes on `matmul_coop_f16_f32acc` instead
//! of on the f32 tiled multiplies: 16-bit operands, 32-bit accumulation, and the
//! score matrix materialised [`DENSE_COOP_CHUNK_ROWS`] query rows at a time
//! (`dense_coop_admits`). That is the embedding encoder's four global layers — 8 192
//! rows, 4 heads of width 512 over one KV head, a 1 GiB score square — which were
//! the band kernel's with the band switched off. Measured on the fork's stand
//! (Apple M3 Pro / Metal, wgpu 30; `mahbot-inference`'s `embed.8192` cell, one
//! bidirectional pass, warm): the stand reads 3 589 ms before this form and
//! 2 312–2 313 ms after, three consecutive runs on the same rig, which puts the
//! block's four layers at ~773 ms against the ~2 050 ms they read on the band
//! kernel. The step itself, at that geometry, reads 230 ms in
//! `tests/attention_dense_coop_cost.rs` — two multiplies at 3.5–3.6 TFLOP/s
//! (155 ms), the softmax's three passes (16 ms), the transposes and the two f16
//! casts (10 ms), and the call's own upload/readback floor for the rest. That
//! leaves it ~3% over the 750 ms this block was given; the measured split says the
//! two multiplies are within 4% of the same kernel's whole-sequence rate
//! (`tests/attention_dense_coop_cost.rs`'s chunk table) and that the residue is the
//! score matrix's three softmax passes plus the transposes, not a mis-tiled
//! multiply.
//!
//! Four shape terms keep the form off a step: the [`DENSE_COOP_MIN_Q`] row floor
//! (below it the f32 arrangement's ~1e-6 answer is the one the tree's precision
//! pins read); the two kernels' whole-tile widths (`head_dim % 32`,
//! `v_head_dim % 64`, the sequence a multiple of 64); one batch, because the
//! per-chunk output permute writes each batch's block where the first chunk
//! already wrote; and a head count that is a whole multiple of the KV heads,
//! because the transposes replicate one tile per KV head while both multiplies
//! read `batch · heads`. [`dense_coop_admits`] carries three more the list above
//! leaves out: the mask kind and square extents (the band forms own every other
//! mask), both widths inside [`ATTENTION_DECOMPOSE_MAX_HEAD_DIM`]'s ceiling, and
//! the dispatch grid inside [`MAX_WORKGROUPS_PER_DIM`] on both axes. A step any
//! of them refuses keeps the band, many-row or fused arrangement it had. The
//! adapter is the term beyond the shapes — it has to build
//! `matmul_coop_f16_f32acc` and the device probe has to have shown it computes
//! `a·b` — and [`crate::backend::WgpuExecutable::attention_dense_coop_declines`]
//! counts its share together with the two switches that decline the form outright
//! (`RLX_WGPU_NO_DENSE_COOP`, and `RLX_WGPU_MATMUL_F32_ONLY`'s absolute f32 pin).
//!
//! A cooperative step is never the many-row arrangement's (see
//! [`AttentionPlan::decompose`]): the band kernel is what such a step keeps where
//! its admission took the shape, and the fused walk where it did not.
//!
//! Why steps and not a graph decomposition: on a discrete Vulkan/DX12 adapter
//! the ops this would decompose into are hosted — `Op::Softmax` / `Op::Binary` /
//! `Op::Where` by `should_pack_host_op` and `Op::Transpose` by its own
//! structure-host switch — while `Op::MatMul` is the one op exempt from that
//! rule. A graph-level decomposition would run the scale, the mask and the
//! normalisation on the CPU on exactly those adapters; as steps, every piece of
//! it stays on the device.
//!
//! Every admission term is the graph's own, including WebGPU's per-dimension
//! dispatch cap ([`MAX_WORKGROUPS_PER_DIM`], which both multiplies reach through
//! their `batch · heads` z extent and their `seq_q`-row y extent). The one
//! condition the geometry here cannot settle, so the `Op::Attention` arm applies
//! it, is the window: the route's steps bind their operands *and* the regions
//! this plan reserves as a single range, so the arena has to address itself whole
//! at the base the operand resolve left behind (`!Arena::is_sharded()`) and that
//! window has to reach the regions
//! (`route_end <= base + size`). Both hold wherever the arena fits one bind
//! window — the case the route is measured in; a step that fails either keeps the
//! fused kernel, which stages each operand into whatever window it has. Not a
//! silent decline: [`crate::backend::WgpuExecutable::attention_window_declines`]
//! counts it per compiled graph and the schedule report
//! (`RLX_WGPU_SCHEDULE`) prints it, and the arm's attention debug print
//! (`RLX_WGPU_DEBUG_ATTN_MASK`) names the term for the step's own addresses.

use rlx_ir::op::MaskKind;

use crate::kernels::{ATTENTION_PARTS_DIM_CHUNK, attention_decode_eligible};

use super::MAX_WORKGROUPS_PER_DIM;

/// Key-range splits one decode attention step runs as: `n` workgroups over
/// disjoint key ranges plus one merge dispatch, or 1 for the single-workgroup
/// form. The decode form's cost is set by how many workgroups the grid has —
/// `batch × heads` of them, one or two warps a workgroup depending on the
/// adapter's subgroup width — so a grid this small leaves the machine idle
/// while one workgroup walks the whole key range.
///
/// Called per run, twice on the same step — once where the uniform write records
/// the depth and once where the encoder dispatches on it — so the grid, the merge
/// guard and the shader all read the extent the run walks rather than the one the
/// graph was lowered at.
pub fn attention_decode_key_splits(
    seq_q: u32,
    seq_k: u32,
    rows: u32,
    head_dim: u32,
    v_head_dim: u32,
) -> u32 {
    let splits = if attention_decode_eligible(seq_q, seq_k, head_dim, v_head_dim)
        && rows <= ATTENTION_DECODE_SPLIT_MAX_ROWS
        && seq_k >= ATTENTION_DECODE_SPLIT_MIN_KEYS
    {
        ATTENTION_DECODE_SPLITS
    } else {
        1
    };
    // A depth of zero dispatches nothing and writes nothing — a silently empty
    // output slot, not a fault — so the one place that returns a depth holds it.
    assert!(
        splits >= 1,
        "the attention key-range split depth is never zero (got {splits})"
    );
    splits
}

/// Splits a decode step takes at most.
pub const ATTENTION_DECODE_SPLITS: u32 = 2;
/// Above this many rows the grid already fills the machine, so the split buys
/// nothing and costs a second dispatch.
pub const ATTENTION_DECODE_SPLIT_MAX_ROWS: u32 = 32;
/// A split needs keys to give every workgroup a share: below this the
/// single-workgroup form already covers the range with one pass of its lanes.
pub const ATTENTION_DECODE_SPLIT_MIN_KEYS: u32 = 256;

/// Bytes the key-range split's partials need: one `[m, l, O[0..pad]]` per
/// (row, split), the layout both decode entries derive from
/// [`ATTENTION_PARTS_DIM_CHUNK`].
pub fn attention_parts_bytes(rows: u32, splits: u32, v_head_dim: u32) -> u64 {
    let pad = v_head_dim.div_ceil(ATTENTION_PARTS_DIM_CHUNK) * ATTENTION_PARTS_DIM_CHUNK;
    (rows as u64) * (splits as u64) * ((2 + pad) as u64) * 4
}

/// The partials a decode step whose grid has `rows` workgroups must be sized for.
///
/// The depth itself is the run's decision ([`attention_decode_key_splits`] at the
/// uniform write), so the buffer is fixed for the deepest form the step can be
/// walked at: the row cap is static, while the query and key extents only ever
/// shrink — a padded prefill walked at one row may split where its compile did
/// not. A grid above the row cap never splits and binds the layout's minimum.
pub fn attention_parts_bytes_for_grid(rows: u32, v_head_dim: u32) -> u64 {
    if rows <= ATTENTION_DECODE_SPLIT_MAX_ROWS {
        attention_parts_bytes(rows, ATTENTION_DECODE_SPLITS, v_head_dim)
    } else {
        16
    }
}

/// Query rows below which the dense arrangement is not served: one, and only
/// one. A single query row stays on the one-row arrangement — the decode kernel
/// where [`crate::kernels::attention_decode_eligible`] allows it, the serial
/// fused kernel otherwise — which is cheaper, so two is the floor.
pub const ATTENTION_DECOMPOSE_MIN_Q: u32 = 2;

/// Head width the route serves, mirroring the fused kernel's own `MAX_HEAD_DIM`
/// (`kernels/attention.wgsl`). Neither multiply on this route needs the cap —
/// the tiled f32 kernels take any k and n — but a shape the fused kernel refuses
/// must not become covered by the route that replaces it, so the two ceilings
/// move together. It is also the ceiling [`plan_attention`] refuses above, so
/// this is the one width both the refusal and the route decision read.
pub(crate) const ATTENTION_DECOMPOSE_MAX_HEAD_DIM: u32 = 512;

/// The route must not need more scratch than this; a score matrix past it belongs
/// on the fused kernel (or on a chunked step) rather than in a per-step
/// allocation this backend does not have.
///
/// Raised from 64 MiB to 512 MiB. [`crate::backend::compute_scratch_bytes`]
/// raises the tail reserve to an admitted plan's own [`AttentionPlan::scratch_bytes`],
/// so the cap is what decides admission and the 64 MiB floor stays the minimum
/// every graph is guaranteed — no graph is sized below the plan it takes. The
/// floor was *not* moved with the cap: a 512 MiB tail on every graph makes a
/// small-shard arena stripe a tail it never writes and refuse on the stripe bound
/// (`sharded_arena`'s forced-small-shard case), for no gain. Raising the cap
/// alone is therefore safe only because the floor still sizes every graph to the
/// plan it takes; a cap moved without that reserve would admit a plan eight times
/// the guaranteed tail.
///
/// Measured on the stand's 8 192-key embedding cell (Apple M3 Pro / Metal): 362.6 s
/// under the 64 MiB cap against tens of seconds under this one, with its arena at
/// 670.2 MiB (the planner's figure at align 16, before the scratch tail) and its
/// output digest 17a3b918 unchanged under both caps — the values did not change,
/// the arrangement did. The cell's first pass moves with load from run to run, so
/// the stand's section of `MEASUREMENTS.md` carries the reading this tree was last
/// measured at rather than a figure restated here.
pub(crate) const ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES: u64 = 512 << 20;

/// Query rows the cooperative dense square form walks per pass. Its score matrix
/// is materialised in full for `chunk` rows over every key, and a chunk of 2 048
/// at the stand's 8 192 keys and four `(batch, head)` tiles is 268 MiB — with Kᵀ
/// and the two compactions that is 469 MiB of scratch, and the output region costs
/// nothing more because it reuses Kᵀ's bytes once the f16 cast has read them
/// (`AttentionPlan::coop_out_over_kt`). The chunk is a row count, not a tile: it
/// is a whole number of 64-row cooperative tiles and the last one is whatever the
/// sequence leaves, also a multiple of 64 because the admission requires the
/// sequence to be.
///
/// 2 048 is both what fits and what the multiplies want: a chunk is 32 row tiles
/// per `(batch, head)`, and the same two multiplies measured at 2 048 rows run
/// 3.59 and 3.47 TFLOP/s against 3.67 and 3.58 whole (`tests/attention_dense_coop_cost.rs`),
/// so the chunking costs ~4 ms a layer — while 2 560 would need 536 MiB, over the
/// cap, for no rate the 2 048 reading does not already have.
pub const DENSE_COOP_CHUNK_ROWS: u32 = 2048;

/// Query rows below which the cooperative dense square form is not taken. The
/// f16 operands are the reason this is a floor at all rather than just a speed
/// question: `tests/attention_long_seq_precision.rs` holds the many-row f32
/// arrangement to ~1e-6 at BrainBERT's geometry (12 heads × 64 head width, up to
/// 186 rows), and the cooperative form's operands are 16-bit. 2 048 is where the
/// dense square's own scratch is past what the f32 many-row form can hold at the
/// stand's width *and* where `tests/attention_dense_coop_cost.rs` measures the
/// cooperative form ahead; every shape below it keeps the route it has.
pub const DENSE_COOP_MIN_Q: u32 = 2048;

/// Rows per workgroup the route's multiplies dispatch their `m = seq_q` rows
/// in: 32 is the narrowest row tile of the kernels the run can pick for them
/// (the plain f32 kernel's, and `WideVariant::Std`'s — the vectorised variants
/// take 64 rows per workgroup, whole tiles plus a 32-row remainder tile), so it
/// is the conservative divisor for that extent's cap.
///
/// The `n` extent carries no admission term of its own: it is `v_head_dim` on
/// the value multiply (≤ 512, sixteen workgroups) and `seq_k` on the score one,
/// and the scratch conjunct bounds the latter instead — `tiles · seq_k · c`
/// with `c = head_dim + seq_q + v_head_dim` (Kᵀ plus the scores plus V's
/// compaction, the last term zero when V's tile is already dense) against the
/// reserve's 134.2M floats. That puts an upper bound on `seq_k` of
/// `134 217 728 / (tiles · c)`, so `seq_k.div_ceil(32)` stays inside the cap of
/// 65 535 wherever `tiles · c >= 65`, and at exactly 64 reaches 65 536 — one
/// workgroup past it. The body's 16 tiles of 128-wide heads give
/// `tiles · c = 16 · 258 = 4 128`, an upper bound of about thirty-two thousand
/// cells. What it leaves is the single-tile end of both widths: one-wide heads
/// with V's tile already dense put `tiles · c` at 3, so 44.7M keys pass the
/// budget with a grid of 1 398 102 workgroups, and one-wide heads of any layout
/// sit at 4. The device refuses such a dispatch rather than computing it
/// wrongly, and closing the corner with a key-length term would enlarge the
/// admission, so it is recorded rather than admitted around.
const MATMUL_TILE_M: u32 = 32;

/// Scratch regions are bumped through 256-byte aligned (`buffer::Arena`'s own
/// window granularity), so each region's element count is rounded up to that
/// alignment before it is summed. The rounded figure is what the lowering bumps
/// through, which is what lets the reserve and the allocation agree exactly.
const SCRATCH_ALIGN_FLOATS: u64 = 256 / 4;

/// The shape half of the route decision: the step's query rows, its two head
/// widths, the tiles both multiplies walk and the float count its scratch needs.
/// No device, no adapter, no arena — so the decision is testable without a
/// device, and the tests below build every field rather than describing a shape
/// wrongly.
pub(crate) struct AttentionGeometry {
    pub seq_q: u32,
    /// The key sequence the step walks.
    pub seq_k: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// `batch · heads`: the (batch, head) tiles both multiplies walk, which the
    /// dispatch carries as its own z extent.
    pub tiles: u64,
    /// Floats the route's scratch needs: Kᵀ + the score matrix + whatever
    /// compaction the operand layouts force. See [`scratch_regions`].
    pub scratch_floats: u64,
}

/// Whether the many-row form — the dense multiply of the scores, the
/// operation's own masked normalisation, and a dense multiply by V — serves
/// this step. Every conjunct is the graph's own — the two dispatch extents the
/// multiplies walk included — so the arm's route gate needs nothing from the
/// adapter (see the module doc), and the floor is
/// [`ATTENTION_DECOMPOSE_MIN_Q`]. Two of the conjuncts are degenerate shapes
/// the fused path still runs: `tiles >= 1`, since with no
/// `(batch, head)` tiles there is no multiply to dispatch and the `n`-extent
/// argument above holds vacuously however long the key sequence is; and
/// `seq_k >= 1`, since every region would be empty, the budget conjunct would
/// pass vacuously, and the route would write zeros where the fused kernel's 0/0
/// normalisation is the NaN the operation is.
///
/// `RLX_WGPU_ATTN_DENSE_ONE_ROW` lowers the floor to one query row: the fused
/// walk is the step's own form there, so the arm prices that form against the
/// decomposed arrangement (the same numbers, another summation order).
pub(crate) fn decompose_attention(g: &AttentionGeometry) -> bool {
    let min_q = if rlx_ir::env::flag("RLX_WGPU_ATTN_DENSE_ONE_ROW") {
        1
    } else {
        ATTENTION_DECOMPOSE_MIN_Q
    };
    g.seq_q >= min_q
        && g.seq_k >= 1
        && g.tiles >= 1
        && g.seq_q.div_ceil(MATMUL_TILE_M) <= MAX_WORKGROUPS_PER_DIM
        && g.tiles <= MAX_WORKGROUPS_PER_DIM as u64
        && g.head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        && g.v_head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        && g.scratch_floats.saturating_mul(4) <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES
}

/// Whether the cooperative dense square form serves this step, and the row chunk
/// it walks its score matrix in. `None` keeps the route the step has.
///
/// The form is the many-row arrangement's three passes — a dense scores multiply,
/// the masked normalisation over the scores, a dense multiply by V — with the two
/// multiplies served by `matmul_coop_f16_f32acc` (16-bit operands, f32
/// accumulation) and the score matrix materialised a chunk of query rows at a
/// time. Every conjunct is the graph's:
///
/// * `MaskKind::None` and square extents. The band forms own every other mask, and
///   the dense square is the shape this route exists for.
/// * one batch. The per-chunk output permute writes at `out_off + row · heads ·
///   v_head_dim` while the transpose it emits lands the batches' blocks
///   contiguously (`batch · m · heads · v_head_dim` from the permute's own start),
///   so with more than one chunk a second batch's block would begin where the
///   first chunk already wrote and the two would interleave and overwrite each
///   other. Per-batch offsets are a different change; a multi-batch step keeps the
///   band or many-row arrangement.
/// * [`DENSE_COOP_MIN_Q`] query rows.
/// * `head_dim % 32 == 0` (the score multiply's K stage) and `v_head_dim % 64 == 0`
///   (the value multiply's output columns); `seq_q % 64 == 0` is both the value
///   multiply's K extent and the row tile of every chunk. These are
///   [`crate::backend::coop_f16_f32acc_admits`]'s three terms, per multiply.
/// * both widths inside [`ATTENTION_DECOMPOSE_MAX_HEAD_DIM`], the fused kernel's
///   own ceiling, so the route covers nothing the fused path refuses.
/// * `heads % nkv == 0`. The arm's Kᵀ and V′ transposes write one tile per KV
///   head — the head axis splits into `(nkv, heads / nkv)` — while both multiplies
///   read `batch · heads` tiles, so a head count that is not a whole multiple of
///   the KV count would leave the extra heads reading zero-initialised f16. Both
///   sibling admissions carry the same term ([`band_attention`], and the many-row
///   arm's own `heads % nkv` check in `plan_attention`).
/// * the dispatch grid inside [`MAX_WORKGROUPS_PER_DIM`] on both axes, and the
///   chunked score region inside [`ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES`] — which
///   `plan_attention` checks, since it needs the operand layouts to size it.
///
/// The device half is not here: whether the adapter builds the kernel and whether
/// it computes `a·b` are the lowering's questions (`matmul_coop_f16_f32acc_kernel`
/// and `crate::coop_probe`), and a step whose device cannot serve it keeps the band
/// arrangement the plan admits underneath it.
pub(crate) fn dense_coop_admits(
    mask_kind_id: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    v_head_dim: u32,
    batch: u32,
    heads: u32,
    nkv: u32,
) -> Option<u32> {
    let tiles = batch as u64 * heads as u64;
    if mask_kind_id != 0
        || seq_q != seq_k
        || batch != 1
        || seq_q < DENSE_COOP_MIN_Q
        || !seq_q.is_multiple_of(64)
        || !seq_k.is_multiple_of(64)
        || !head_dim.is_multiple_of(32)
        || !v_head_dim.is_multiple_of(64)
        || head_dim > ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        || v_head_dim > ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        || nkv == 0
        || !heads.is_multiple_of(nkv)
        || tiles == 0
        || tiles > MAX_WORKGROUPS_PER_DIM as u64
        || seq_k.div_ceil(64) > MAX_WORKGROUPS_PER_DIM
        || seq_q.div_ceil(64) > MAX_WORKGROUPS_PER_DIM
    {
        return None;
    }
    Some(seq_q.min(DENSE_COOP_CHUNK_ROWS))
}

/// The `params.mask_kind` id the tiled band kernel is dispatched on — the
/// `MaskKind::Window` arm of the `0..=5` uniform contract, and one constant so
/// the router, the lowering, the run loop and the step name cannot disagree
/// about it. The kernel itself reads only `window` (and only through `w < seq`,
/// so `w >= seq` is "no band at all"), which is what lets a dense
/// `MaskKind::None` step whose extents are square be served through the same
/// entry point: the lowering writes this id and `window = seq_k` for it.
pub(crate) const BAND_MASK_KIND: u32 = 5;

/// Whether the tiled band kernel (`kernels/attention_band.wgsl`, dispatched on
/// [`BAND_MASK_KIND`]) can serve a `MaskKind::Window` step — or a dense step
/// whose extents are square — and the refusal naming the shapes where it cannot.
///
/// This is the whole admission: unlike the dense route there is no scratch, no
/// arena and no window term — the kernel walks the band in one pass over the
/// operands the step already has — so every conjunct is the graph's own and the
/// decision stays device-free (the one adapter term, `SHADER_F16`, picks between
/// the kernel's two staging tiles rather than a route, and both compute the same
/// answer).
///
/// The conjuncts, and why each is a refusal rather than a fallback:
///
/// - `seq_q == seq_k`: the pass walks one key sequence with no carried KV, so a
///   query extent that differs from the key extent has no band to walk. (A
///   windowed *decode* step would need the band applied against a cached key
///   sequence; this tree has no such route, and the fused kernel would read kind
///   5 as no mask.)
/// - `seq_k >= 1`: nothing to attend to.
/// - `head_dim == v_head_dim`: a thread owns one contiguous chunk of Q and of O,
///   and the two are the same chunk. Asymmetric V (MLA) needs a second chunk
///   mapping and is refused by name rather than mis-read.
/// - `head_dim >= ATTENTION_BAND_MIN_HEAD_DIM`: the reduce sums a row's chunk
///   partials two or four at a time, and a one-chunk row has nothing to sum — the
///   pair form would read the next row's partials (and past `part_s` on the last
///   row). A width that would build that form is refused rather than answered
///   wrongly; the fix is a kernel form, not a wider read.
/// - `head_dim` a multiple of `ATTENTION_BAND_DIMS_PER_THREAD` and at most
///   `ATTENTION_BAND_MAX_HEAD_DIM`: the chunk width is a compile-time constant in
///   the module, so a width the module was not built for would read the wrong
///   dims.
/// - `attention_band_form` exists: its chunk count has to divide both workgroup
///   widths the kernel builds (`ATTENTION_BAND_MIN_THREADS` at the floor and
///   `ATTENTION_BAND_MAX_THREADS` where the adapter admits the wide form) into
///   whole rows, or the workgroup's last threads belong to a row the block does
///   not have.
/// - `heads` a multiple of `nkv`: the KV-head group the shader divides by.
/// - both dispatch extents inside `MAX_WORKGROUPS_PER_DIM`: `seq_q / BM` blocks
///   along x, the `batch · heads` tiles along y.
pub(crate) fn band_attention(
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    v_head_dim: u32,
    heads: u32,
    nkv: u32,
    tiles: u64,
) -> Result<(), AttentionRefusal> {
    use crate::kernels::{
        ATTENTION_BAND_DIMS_PER_THREAD, ATTENTION_BAND_MAX_HEAD_DIM, ATTENTION_BAND_MIN_HEAD_DIM,
        attention_band_form, attention_band_rows,
    };
    let reason = if seq_q != seq_k {
        "the band pass walks one key sequence, so seq_q must equal seq_k"
    } else if seq_k == 0 {
        "there is no key to attend to"
    } else if head_dim != v_head_dim {
        "the tiled pass holds one chunk width, so v_head_dim must equal head_dim"
    } else if head_dim < ATTENTION_BAND_MIN_HEAD_DIM {
        "head_dim is below the kernel's 64-wide floor: its reduce sums a row's chunk \
         partials two or four at a time, and a one-chunk row would read the next row's"
    } else if !head_dim.is_multiple_of(ATTENTION_BAND_DIMS_PER_THREAD) {
        "head_dim must be a multiple of the kernel's 32-wide thread chunk"
    } else if attention_band_form(head_dim).is_none() {
        // The chunk count is `head_dim / ATTENTION_BAND_DIMS_PER_THREAD` and must
        // divide the workgroup into whole rows at both widths the kernel builds
        // (`ATTENTION_BAND_MIN_THREADS` = 256 and `ATTENTION_BAND_MAX_THREADS` =
        // 512). The sentence names the divisor and the admitted set rather than
        // enumerating every refused width, which would have to move in step with
        // the form on every change.
        "head_dim / 32 chunks must split the kernel's workgroup into whole rows at both \
         of its widths (256 threads at the floor, 512 where the adapter admits the wide \
         form): only 64, 128, 256 and 512 do"
    } else if head_dim > ATTENTION_BAND_MAX_HEAD_DIM {
        "head_dim is above the widest head the tiled pass holds"
    } else if nkv == 0 || heads == 0 || !heads.is_multiple_of(nkv) {
        "the head count must be a whole multiple of the KV-head count"
    } else if seq_q.div_ceil(attention_band_rows(head_dim)) > MAX_WORKGROUPS_PER_DIM
        || tiles > MAX_WORKGROUPS_PER_DIM as u64
    {
        "the dispatch grid is above WebGPU's per-dimension workgroup cap"
    } else {
        return Ok(());
    };
    Err(AttentionRefusal::BandGeometry {
        seq_q,
        seq_k,
        head_dim,
        v_head_dim,
        reason,
    })
}

/// Whether an operand's own per-`(batch, head)` tile is already a dense
/// `[seq, width]` block, and so can be read or written where it lies instead of
/// through a compaction transpose. `head_stride == seq · width` is what makes
/// the tiles contiguous *and* in `(batch, head)` order, which is the order the
/// multiplies' batch axis walks them in.
pub(crate) fn operand_tile_is_dense(
    seq_stride: u32,
    head_stride: u32,
    seq: u32,
    width: u32,
) -> bool {
    seq_stride == width && head_stride == seq * width
}

/// The route's scratch regions, in the fixed order the lowering bumps through
/// them, in f32 elements and already rounded up to [`SCRATCH_ALIGN_FLOATS`].
/// `0` marks a region the route does not emit (that operand's tiles are dense,
/// or its output slot is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionScratchRegions {
    /// Kᵀ, `[B, H, D, S_k]` — always: the score multiply reads B as `[k=D, n=S_k]`.
    pub kt: u64,
    /// The score matrix `[B, H, S_q, S_k]`, normalised in place into P.
    pub scores: u64,
    /// Q′ `[B, H, S_q, D]`, only when Q's tile is not dense.
    pub q: u64,
    /// V′ `[B, H, S_k, D_v]`, only when V's tile is not dense.
    pub v: u64,
    /// out′ `[B, H, S_q, D_v]`, only when the output slot is not dense per
    /// `(batch, head)` — otherwise the second multiply writes the slot directly.
    pub out: u64,
}

impl AttentionScratchRegions {
    pub(crate) fn total(self) -> u64 {
        self.kt + self.scores + self.q + self.v + self.out
    }
}

/// The shape a step's scratch regions are sized from: one struct rather than a
/// run of positional arguments, so a call site cannot hand the two row extents to
/// each other's parameter — they are both `u32` extents of the same kind and the
/// compiler would not say a word.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScratchShape {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// The query extent the score matrix and the output region are materialised
    /// for: the whole sequence for the f32 arrangement, one chunk for the
    /// cooperative form (which walks them in order through the same region).
    pub rows: u32,
    pub q_dense: bool,
    pub v_dense: bool,
    pub out_dense: bool,
}

/// The regions a step's layout forces. One place, so the figure the route
/// decision reads and the offsets the lowering bumps through cannot disagree.
pub(crate) fn scratch_regions(shape: ScratchShape) -> AttentionScratchRegions {
    let ScratchShape {
        batch,
        heads,
        seq_q,
        seq_k,
        head_dim,
        v_head_dim,
        rows,
        q_dense,
        v_dense,
        out_dense,
    } = shape;
    let align = |floats: u64| floats.div_ceil(SCRATCH_ALIGN_FLOATS) * SCRATCH_ALIGN_FLOATS;
    // The (batch, head) tiles every region is sized in. GQA/MQA expands here:
    // Kᵀ and V′ are written per query head, so that the multiplies' batch axis
    // is one dense tile per (b, h) and needs no per-head indirection.
    let tiles = (batch as u64) * (heads as u64);
    AttentionScratchRegions {
        kt: align(tiles * head_dim as u64 * seq_k as u64),
        scores: align(tiles * rows as u64 * seq_k as u64),
        q: if q_dense {
            0
        } else {
            align(tiles * seq_q as u64 * head_dim as u64)
        },
        v: if v_dense {
            0
        } else {
            v_compaction_floats(batch, heads, seq_k, v_head_dim)
        },
        out: if out_dense {
            0
        } else {
            align(tiles * rows as u64 * v_head_dim as u64)
        },
    }
}

/// The region a compacted V′ takes: one dense `[S_k, D_v]` tile per query head,
/// rounded up to the bump's alignment. The one home of its size — the layout
/// above and the plan's booked spare (which the lowering spends when the output
/// slot lands on V; see [`AttentionPlan::v_spare`]) both read it here.
fn v_compaction_floats(batch: u32, heads: u32, seq_k: u32, v_head_dim: u32) -> u64 {
    let floats = (batch as u64) * (heads as u64) * (seq_k as u64) * (v_head_dim as u64);
    floats.div_ceil(SCRATCH_ALIGN_FLOATS) * SCRATCH_ALIGN_FLOATS
}

/// The slack the tail reserve carries on top of the regions: the bump cursor is
/// only 16-byte aligned when the route starts (`Arena::scratch_off`), so the
/// first region can begin up to one alignment past it.
pub(crate) const SCRATCH_ALIGN_SLACK_BYTES: u64 = 256;

/// Everything the many-row route needs to know about one `Op::Attention` node:
/// the batch/head/sequence geometry, the KV-head count, the strides Q/K/V and the
/// output are read and written through, the mask's kind/numbering and strides,
/// which per-`(batch, head)` tiles are already dense, the scratch regions those
/// tiles force, and whether the route serves the step at all. Derived from the
/// graph alone — see [`plan_attention`].
///
/// A plan is one `Op::Attention` node's *geometry*, not its operands or their
/// addresses: the offsets the route reads them at come from the arena and the
/// schedule, which this module deliberately cannot see.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionPlan {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// KV heads (GQA/MQA). `heads` where a packed QKV parent carries K and V,
    /// whose strides are uniform across the three operands.
    pub nkv: u32,
    /// The `[B, S, 3, H, D]` parent Q/K/V are axis-2 narrows of, with the width
    /// of one head in elements, where one exists and
    /// `RLX_WGPU_NO_PACKED_BSHD_ATTN` is clear.
    pub packed_parent: Option<(rlx_ir::NodeId, u32)>,
    pub q_b: u32,
    pub q_h: u32,
    pub q_s: u32,
    pub k_b: u32,
    pub k_h: u32,
    pub k_s: u32,
    pub v_b: u32,
    pub v_h: u32,
    pub v_s: u32,
    pub o_b: u32,
    pub o_h: u32,
    pub o_s: u32,
    /// The mask's own kind, in the fused kernel's numbering: 0 none, 1 causal,
    /// 2 binary key padding, 3 sliding window, 4 additive bias, and
    /// [`BAND_MASK_KIND`] for the two-sided band — which a dense step whose
    /// extents are square also takes, with its [`window`](Self::window) set to
    /// the key extent.
    pub mask_kind_id: u32,
    /// The band's width, 0 for every other kind. A dense step served through the
    /// band kernel carries `seq_k` here: `w >= seq` is the kernel's "no band"
    /// path, and `u32::MAX` would overflow the mask test's address arithmetic.
    pub window: u32,
    pub m_b: u32,
    pub m_h: u32,
    pub m_q: u32,
    pub m_k: u32,
    /// Whether Q's / V's / the output's per-`(batch, head)` tile is already a
    /// dense `[seq, width]` block (see [`operand_tile_is_dense`]).
    pub q_dense: bool,
    pub v_dense: bool,
    pub out_dense: bool,
    /// The regions those three flags force, in the order the route bumps them.
    pub regions: AttentionScratchRegions,
    /// V′'s region for the case [`regions`](Self::regions) leaves out: V's tiles
    /// are dense, but the planned output slot lies on V's own bytes, so the
    /// lowering compacts V anyway. 0 whenever `regions` already holds it.
    pub v_spare: u64,
    /// Whether the route serves the step: the shape half of the decision
    /// ([`decompose_attention`]) over this plan's own geometry and regions, with
    /// `nkv` dividing `heads` for the grouped-KV layouts. Never true for a band
    /// step: kind 5 has its own kernel and the dense arrangement does not compute
    /// it, and a dense step is only routed to that kernel where this arrangement
    /// no longer fits. Nor for a cooperative one, whose booked regions hold one
    /// chunk of rows rather than the square (see where `decompose` is computed).
    pub decompose: bool,
    /// Whether the tiled band kernel ([`band_attention`]) is this step's route — a
    /// `MaskKind::Window` one, or a dense `MaskKind::None` one whose extents are
    /// square and whose many-row scratch would not fit, or which the cooperative
    /// form booked chunk-sized regions for. A `Step::Attention` dispatch whose
    /// pipeline the run picks from the mask kind, not a decomposition. Mutually
    /// exclusive with [`decompose`](Self::decompose), and with
    /// [`dense_coop`](Self::dense_coop) in *preference* only: where both are
    /// admitted the cooperative form is what the lowering emits first, and this is
    /// what the step keeps if the adapter declines it.
    pub band: bool,
    /// Whether the step is served by the cooperative dense square form
    /// ([`dense_coop_admits`]): the many-row arrangement's three passes, with its
    /// two multiplies on `matmul_coop_f16_f32acc` and its score matrix
    /// walked [`coop_chunk`](Self::coop_chunk) query rows at a time. Mutually
    /// exclusive with [`decompose`](Self::decompose); where [`band`](Self::band) is
    /// also admitted this is the form the lowering emits first and the band kernel
    /// is what the step keeps if the adapter declines it. The device half of the
    /// admission is the lowering's, so a `true` here is a shape the adapter may
    /// still decline.
    pub dense_coop: bool,
    /// Query rows each pass of the cooperative form covers — the row extent its
    /// score region is sized for, and the `m` of both its multiplies. 0 when
    /// [`dense_coop`](Self::dense_coop) is false.
    pub coop_chunk: u32,
    /// Whether the output region shares Kᵀ's bytes. The cooperative form's Kᵀ is
    /// read once, by its own f16 cast, before the first chunk's multiplies; the
    /// output region is written from the first chunk's value multiply onwards, so
    /// the two never overlap in time and the output grid — no larger than Kᵀ's —
    /// reuses them instead of asking for a sixth region. False on every other
    /// route, and false where the output is written where it lies.
    pub coop_out_over_kt: bool,
    /// Whether this is a dense square step the band kernel would have served had
    /// its admission accepted the shape, where the cooperative form was not booked
    /// for it. `band` is false for it, so the step keeps the fused walk; the
    /// lowering counts the flag for the schedule report, which is what keeps that
    /// fallback from being silent.
    pub band_declined: bool,
}

/// Where each of the route's scratch regions starts, in bytes from the bump's
/// start, plus one past the last of them — the order the lowering bumps through
/// them in, derived once here so the arm's offsets and the reserve cannot drift
/// apart (see [`AttentionPlan::scratch_offsets`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionScratchOffsets {
    pub kt: u64,
    pub scores: u64,
    pub q: u64,
    pub v: u64,
    pub out: u64,
    /// One past the last region: what a bump starting at the aligned cursor
    /// reaches, and never past [`AttentionPlan::scratch_bytes`].
    pub end: u64,
}

impl AttentionPlan {
    /// The tail-scratch bytes the route needs: the aligned regions — plus V′'s
    /// booked spare, since the lowering may have to compact a V it would
    /// otherwise read where it lies — and the one alignment the bump cursor
    /// itself can start short of ([`SCRATCH_ALIGN_SLACK_BYTES`]). One place, so
    /// the reserve and the bump cannot drift, and never less than the bump the
    /// arm performs.
    pub(crate) fn scratch_bytes(&self) -> usize {
        let total = self.regions.total()
            - if self.coop_out_over_kt {
                self.regions.out
            } else {
                0
            };
        ((total + self.v_spare) * 4 + SCRATCH_ALIGN_SLACK_BYTES) as usize
    }

    /// The region offsets for an arm that has (`v_compacted`) or has not had to
    /// compact V — the arm's own answer, since what decides it is its output
    /// slot's overlap with V (see the arm's note), and the plan books that
    /// compaction as [`v_spare`](Self::v_spare). `end` is at most
    /// [`scratch_bytes`](Self::scratch_bytes): the spare is booked for exactly
    /// this compaction and the slack covers the cursor's alignment.
    pub(crate) fn scratch_offsets(&self, v_compacted: bool) -> AttentionScratchOffsets {
        let r = self.regions;
        let kt = 0;
        let scores = kt + r.kt * 4;
        let q = scores + r.scores * 4;
        let v = q + r.q * 4;
        let v_end = v + if v_compacted {
            r.v * 4 + self.v_spare * 4
        } else {
            0
        };
        let out = if self.coop_out_over_kt { kt } else { v_end };
        AttentionScratchOffsets {
            kt,
            scores,
            q,
            v,
            out,
            end: v_end.max(out + r.out * 4),
        }
    }

    /// The arrangement this plan settles the step on, in the lowering's own
    /// precedence: the cooperative dense square form first, the band kernel where
    /// that is admitted underneath it, the f32 many-row arrangement, and the fused
    /// walk where none of the three applies.
    ///
    /// This is the *shape* half — a cooperative step whose adapter declines the
    /// kernel keeps the arrangement recorded under it, which is [`band`](Self::band)
    /// where that is true and the fused walk otherwise (`decompose` is false for a
    /// cooperative plan). The precedence lives here rather than in each reader's
    /// `if` chain so the lowering's `RLX_WGPU_DEBUG_ATTN_MASK` print and the arm
    /// cannot disagree about it. That print is this method's only reader; it
    /// qualifies the answer by the terms the plan cannot see (whether the adapter
    /// took the cooperative form, and for the many-row arm the arena's sharding and
    /// bind window).
    pub(crate) fn arrangement(&self) -> AttentionArrangement {
        if self.dense_coop {
            AttentionArrangement::DenseCoop
        } else if self.band {
            AttentionArrangement::Band
        } else if self.decompose {
            AttentionArrangement::ManyRow
        } else {
            AttentionArrangement::Fused
        }
    }
}

/// Which of the four arrangements a plan settles on — the field-level precedence
/// of [`AttentionPlan::arrangement`], named so a reader can match on it instead of
/// restating a chain of flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionArrangement {
    /// [`AttentionPlan::dense_coop`]: the cooperative dense square form, the
    /// lowering's first choice where its shape is admitted.
    DenseCoop,
    /// [`AttentionPlan::band`]: the tiled band kernel, including a dense square
    /// step it serves with the band switched off.
    Band,
    /// [`AttentionPlan::decompose`]: the f32 many-row arrangement.
    ManyRow,
    /// None of the three: the fused one-thread-per-row walk.
    Fused,
}

/// Why a node has no plan, carrying the message the lowering's own panic for
/// that shape prints. Every variant but [`NotAttention`](Self::NotAttention),
/// [`HeadWidth`](Self::HeadWidth) and [`AmbiguousRank4`](Self::AmbiguousRank4)
/// describes a step the fused path cannot derive a shape for either — the arms
/// that hit them index a dim that is not there — so a refusal is never a shape
/// the route could have taken. Both exceptions are shapes the fused path *can*
/// size, refused here rather than lowered to a wrong result: [`HeadWidth`](Self::HeadWidth)
/// is a width whose kernel would early-return without writing the output slot,
/// and [`AmbiguousRank4`](Self::AmbiguousRank4) is a rank-4 operand at the one
/// extent where both axis orders fit it with nothing declaring which — a reading
/// the fused path would take off the head axis and compute wrong in silence. See
/// [`rank4_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionRefusal {
    /// The node is not an `Op::Attention` — what the scratch reserve, which
    /// walks every node of a graph, gets for all of them.
    NotAttention,
    /// An `Op::Attention` with fewer than three inputs.
    Arity { inputs: usize },
    /// An `Op::Attention` with a zero head count or head width.
    ZeroWidth { num_heads: usize, head_dim: usize },
    /// An `Op::Attention` whose Q, K, V or output shape is not fully static.
    NonStatic,
    /// Q, K or V is neither rank-3 nor rank-4.
    Rank { rank: usize },
    /// Rank-3 `[B·H, S, D]` whose leading dim is not a multiple of the head
    /// count (and whose last dim is not the `[B, S, H·D]` feature width).
    Rank3Leading {
        leading: usize,
        heads: usize,
        head_dim: usize,
        last: usize,
    },
    /// A mask kind that reads `node.inputs[3]` on a node that has fewer inputs.
    MissingMask { kind: u32, inputs: usize },
    /// A head width — after the kernel's own "0 means V is as wide as Q"
    /// resolution — above [`ATTENTION_DECOMPOSE_MAX_HEAD_DIM`], which the fused
    /// kernel's `MAX_HEAD_DIM` and this route's decision both cap at.
    HeadWidth { head_dim: u32, v_head_dim: u32 },
    /// A rank-4 operand at the one extent where both readings fit it — axis 1 and
    /// axis 2 are both `heads`, one extent rather than two — with nothing on the
    /// operand's own path declaring which one it is. See [`rank4_layout`].
    AmbiguousRank4 { heads: u32, head_dim: u32 },
    /// A `MaskKind::Window` step the band kernel cannot hold, with the shapes it
    /// was asked for and the rule that broke. Refused rather than lowered: no
    /// other arrangement computes kind 5 — the fused entry and the dense route
    /// both read it as "no mask" and run a dense pass, which is a wrong answer
    /// rather than a slow one. See [`band_attention`].
    BandGeometry {
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        v_head_dim: u32,
        /// Which conjunct failed, in the words the panic prints.
        reason: &'static str,
    },
}

impl std::fmt::Display for AttentionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAttention => write!(f, "rlx-wgpu Attention: node is not an Attention node"),
            Self::Arity { inputs } => write!(
                f,
                "rlx-wgpu Attention: needs Q, K and V (node has {inputs} inputs)"
            ),
            Self::ZeroWidth {
                num_heads,
                head_dim,
            } => write!(
                f,
                "rlx-wgpu Attention: zero num_heads or head_dim ({num_heads} heads of {head_dim})"
            ),
            Self::NonStatic => write!(
                f,
                "rlx-wgpu Attention: Q, K, V and the output shape must be static"
            ),
            Self::Rank { rank } => write!(
                f,
                "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V inputs supported (got rank {rank})"
            ),
            Self::Rank3Leading {
                leading,
                heads,
                head_dim,
                last,
            } => write!(
                f,
                "rlx-wgpu Attention: rank-3 leading dim {leading} not divisible by num_heads \
                 {heads} (and last dim {last} ≠ H·D = {})",
                heads * head_dim
            ),
            Self::MissingMask { kind, inputs } => write!(
                f,
                "rlx-wgpu Attention: mask kind {kind} needs a mask input (node has {inputs} inputs)"
            ),
            Self::HeadWidth {
                head_dim,
                v_head_dim,
            } => write!(
                f,
                "rlx-wgpu Attention: head_dim {head_dim} / v_head_dim {v_head_dim} is above the \
                 {ATTENTION_DECOMPOSE_MAX_HEAD_DIM} this route serves"
            ),
            Self::AmbiguousRank4 { heads, head_dim } => write!(
                f,
                "rlx-wgpu Attention: rank-4 layout is ambiguous — axis 1 and axis 2 are both \
                 {heads}, so `[B, H, S, D]` and `[B, S, H, D]` fit the operand equally and the \
                 key length read off it differs — and nothing on its way into the step declares \
                 which one it is. Read one of them: unroll a packed `[.., {heads}·{head_dim}]` \
                 feature axis into `[B, S, H, D]`, or swap axes 1 and 2 into `[B, H, S, D]`"
            ),
            Self::BandGeometry {
                seq_q,
                seq_k,
                head_dim,
                v_head_dim,
                reason,
            } => write!(
                f,
                "rlx-wgpu Attention: MaskKind::Window (two-sided band) needs the tiled band \
                 kernel, and this step is one it cannot hold: {reason} — seq_q {seq_q}, seq_k \
                 {seq_k}, head_dim {head_dim}, v_head_dim {v_head_dim}. There is no dense \
                 fallback: the fused and decomposed arrangements read kind 5 as \"no mask\" \
                 and would answer a dense pass"
            ),
        }
    }
}

/// Which way round a rank-4 attention operand's axes run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rank4Layout {
    /// `[B, H, S, D]`: axis 1 is the head axis, axis 2 the query/key rows.
    HeadsFirst,
    /// `[B, S, H, D]`: axis 1 carries the rows, axis 2 the heads.
    RowsFirst,
}

impl Rank4Layout {
    /// The reading an axis 1 ↔ 2 swap of a rank-4 operand gives.
    fn flipped(self) -> Self {
        match self {
            Self::HeadsFirst => Self::RowsFirst,
            Self::RowsFirst => Self::HeadsFirst,
        }
    }
}

/// A dim the walk or the derivation has to read, or `None` where the graph has
/// not pinned it. Producer dims are read too, so a dynamic one is answered with
/// [`AttentionRefusal::NonStatic`] rather than panicking out of `unwrap_static`.
fn static_dim(d: rlx_ir::shape::Dim) -> Option<u32> {
    d.is_static().then(|| d.unwrap_static() as u32)
}

/// The extents of a shape, or `None` where the graph has not pinned them.
fn static_dims(shape: &rlx_ir::Shape) -> Option<Vec<u32>> {
    shape.dims().iter().map(|d| static_dim(*d)).collect()
}

/// The layout the graph declares for a rank-4 operand, or `None` where nothing
/// on the operand's own path says which one it is.
///
/// This tree spells the two presentations differently, and these are the
/// spellings read here:
///
/// * a `Reshape` that unrolls a packed head width into per-head axes — the
///   rank-3 `[B, S, H·D]` projection, or the `[B, S, 1, H, D]` axis-2 narrow of
///   a packed QKV parent — leaves the rows on axis 1: rows-first;
/// * a `Transpose` swapping axes 1 and 2 of a rank-4 operand moves the head axis
///   onto axis 1 (or off it): its own operand's layout, flipped;
/// * every other producer — a norm, a rope, an elementwise op, a cast — hands
///   the axes back as it found them, so the walk continues into the operand it
///   transforms: its *first* input, which is the data for the ops named here but
///   not for all of them (`Where` carries its condition there, and a scaled
///   `Binary` may carry the scale). No spelling in this tree reaches a coincident
///   rank-4 operand that way, and which input an op transforms is not something
///   its shape alone decides — so the walk is not widened to guess one.
///
/// Anything else ends the walk: the operand's layout is then one nothing
/// declares, which is what [`rank4_layout`] refuses at the coincident extent.
fn declared_rank4_layout(
    graph: &rlx_ir::Graph,
    id: rlx_ir::NodeId,
    heads: u32,
    head_dim: u32,
) -> Result<Option<Rank4Layout>, AttentionRefusal> {
    let node = graph.node(id);
    // The operand under discussion is the rank-4 one; a node on the way to it
    // that is not rank-4 itself is not a spelling of it.
    if node.shape.rank() != 4 {
        return Ok(None);
    }
    match &node.op {
        rlx_ir::Op::Reshape { .. } => {
            let Some(src_id) = node.inputs.first().copied() else {
                return Ok(None);
            };
            let src = static_dims(&graph.node(src_id).shape).ok_or(AttentionRefusal::NonStatic)?;
            let dims = static_dims(&node.shape).ok_or(AttentionRefusal::NonStatic)?;
            let unrolled = match src.len() {
                // `[B, S, H·D] → [B, S, H, D]` — a rank-1 to rank-3 source, so that
                // it has a last axis at all: the packed head width unrolled into
                // the head axes, with the source's own leading extents (the batch
                // and the sequence) ahead of them, a unit axis either side being no
                // extent at all. The rows stay on axis 1. A source with no axis to
                // unroll is not a spelling, and ends the walk below rather than
                // indexing one.
                n if (1..4).contains(&n) => {
                    src[n - 1] == heads * head_dim
                        && dims[2..4] == [heads, head_dim]
                        && src[..n - 1]
                            .iter()
                            .filter(|&&d| d != 1)
                            .eq(dims[..2].iter().filter(|&&d| d != 1))
                }
                // `[B, S, 1, H, D] → [B, S, H, D]`: the packed QKV parent's
                // axis-2 narrow unrolled — the spelling `packed_bshd_qkv_strides`
                // is read for. Its unit axis is dropped on the way, so the whole
                // target is checked here.
                5 => src[2] == 1 && [src[0], src[1], src[3], src[4]] == dims[..4],
                _ => false,
            };
            if unrolled {
                return Ok(Some(Rank4Layout::RowsFirst));
            }
            // A reshape that folds unit axes away — or is a no-op — leaves the
            // order alone, so its own operand answers; any other reshape is not a
            // spelling this route reads.
            if src != dims {
                return Ok(None);
            }
            declared_rank4_layout(graph, src_id, heads, head_dim)
        }
        rlx_ir::Op::Transpose { perm } if perm.len() == 4 => {
            let swaps = match perm[..] {
                [0, 2, 1, 3] => true,
                [0, 1, 2, 3] => false,
                _ => return Ok(None),
            };
            let Some(src_id) = node.inputs.first().copied() else {
                return Ok(None);
            };
            let Some(src) = declared_rank4_layout(graph, src_id, heads, head_dim)? else {
                return Ok(None);
            };
            Ok(Some(if swaps { src.flipped() } else { src }))
        }
        _ => match node.inputs.first().copied() {
            Some(src) => declared_rank4_layout(graph, src, heads, head_dim),
            None => Ok(None),
        },
    }
}

/// The layout of a rank-4 attention operand: what its own extents settle, and —
/// where they leave both readings possible — what the graph declares.
///
/// Heads-first needs axis 1 to be the head count and rows-first needs axis 2 to
/// be it, so the one operand both readings fit is the one whose first two axes
/// are *both* the head count. There the two presentations are shape-identical
/// and no shape-only tie-break is sound: a reading taken from the extents alone
/// comes out heads-first whatever the operand really is, and the key length is
/// then read off the head axis — the KV-head count in place of the number of
/// keys — so the tensor is computed wrong with no refusal and no report. The
/// graph breaks the tie ([`declared_rank4_layout`]); where it does not, this
/// refuses the step by name instead of guessing.
fn rank4_layout(
    graph: &rlx_ir::Graph,
    q_id: rlx_ir::NodeId,
    heads: u32,
    head_dim: u32,
) -> Result<Rank4Layout, AttentionRefusal> {
    let q = &graph.node(q_id).shape;
    let (Some(axis1), Some(axis2)) = (static_dim(q.dim(1)), static_dim(q.dim(2))) else {
        return Err(AttentionRefusal::NonStatic);
    };
    if axis1 != heads {
        return Ok(Rank4Layout::RowsFirst);
    }
    if axis2 != heads {
        return Ok(Rank4Layout::HeadsFirst);
    }
    declared_rank4_layout(graph, q_id, heads, head_dim)?
        .ok_or(AttentionRefusal::AmbiguousRank4 { heads, head_dim })
}

/// `(batch, seq_q, seq_k)` and whether the layout is heads-first, for the rank-4
/// Q and K of one attention step. The one derivation this route reads — the plan
/// below and the `Op::AttentionBackward` arm of the lowering both go through it,
/// so no graph is read two ways by the route. A dim it reads that the graph has
/// not pinned is the enum's own [`AttentionRefusal::NonStatic`].
pub(crate) fn rank4_geometry(
    graph: &rlx_ir::Graph,
    q_id: rlx_ir::NodeId,
    k_id: rlx_ir::NodeId,
    heads: u32,
    head_dim: u32,
) -> Result<(u32, u32, u32, bool), AttentionRefusal> {
    let dim = |id: rlx_ir::NodeId, i: usize| -> Result<u32, AttentionRefusal> {
        static_dim(graph.node(id).shape.dim(i)).ok_or(AttentionRefusal::NonStatic)
    };
    Ok(match rank4_layout(graph, q_id, heads, head_dim)? {
        Rank4Layout::HeadsFirst => (dim(q_id, 0)?, dim(q_id, 2)?, dim(k_id, 2)?, true),
        Rank4Layout::RowsFirst => (dim(q_id, 0)?, dim(q_id, 1)?, dim(k_id, 1)?, false),
    })
}

/// `(batch, seq_q, seq_k)` for the rank-3 spellings of one attention step: the
/// packed `[B, S, H·D]` projection against `[B·H, S, D]`, told apart by the last
/// dim. The route's one rank-3 derivation — the plan below and the
/// `Op::AttentionBackward` arm of the lowering both go through it, so a rank-3
/// graph is not read two ways by the route either.
pub(crate) fn rank3_geometry(
    graph: &rlx_ir::Graph,
    q_id: rlx_ir::NodeId,
    k_id: rlx_ir::NodeId,
    heads: u32,
    head_dim: u32,
) -> Result<(u32, u32, u32), AttentionRefusal> {
    let dim = |id: rlx_ir::NodeId, i: usize| -> Result<u32, AttentionRefusal> {
        static_dim(graph.node(id).shape.dim(i)).ok_or(AttentionRefusal::NonStatic)
    };
    let (leading, seq_q, seq_k) = (dim(q_id, 0)?, dim(q_id, 1)?, dim(k_id, 1)?);
    let last = dim(q_id, 2)?;
    if last == heads * head_dim {
        // [B, S, H·D] — the leading dim is the batch.
        return Ok((leading, seq_q, seq_k));
    }
    if !leading.is_multiple_of(heads) {
        // [B·H, S, D] with a leading dim that is not B·H.
        return Err(AttentionRefusal::Rank3Leading {
            leading: leading as usize,
            heads: heads as usize,
            head_dim: head_dim as usize,
            last: last as usize,
        });
    }
    Ok((leading / heads, seq_q, seq_k))
}

/// The route's geometry for one node of a graph, or why the node has none.
///
/// Every node of any graph gets an answer, because the scratch reserve in
/// [`crate::backend`] walks all of them: a node that is not an `Op::Attention`
/// gets [`AttentionRefusal::NotAttention`], and one whose shapes the arm cannot
/// derive geometry for gets the refusal that names the dim. The refusals are
/// what the lowering's own arm panics with — there is no plan with no geometry
/// for it to read instead, so a shape this cannot plan is loud rather than a
/// step the kernel early-returns from, the over-ceiling head width and the
/// rank-4 layout nothing declares included.
pub(crate) fn plan_attention(
    graph: &rlx_ir::Graph,
    node: &rlx_ir::Node,
) -> Result<AttentionPlan, AttentionRefusal> {
    let rlx_ir::Op::Attention {
        num_heads,
        head_dim,
        v_head_dim,
        mask_kind,
        ..
    } = &node.op
    else {
        return Err(AttentionRefusal::NotAttention);
    };
    if node.inputs.len() < 3 {
        return Err(AttentionRefusal::Arity {
            inputs: node.inputs.len(),
        });
    }
    if *head_dim == 0 || *num_heads == 0 {
        return Err(AttentionRefusal::ZeroWidth {
            num_heads: *num_heads,
            head_dim: *head_dim,
        });
    }
    let (q_id, k_id, v_id) = (node.inputs[0], node.inputs[1], node.inputs[2]);
    let q_ir = &graph.node(q_id).shape;
    let k_ir = &graph.node(k_id).shape;
    let v_ir = &graph.node(v_id).shape;
    let out_ir = &node.shape;
    if let Some(rank) = [q_ir.rank(), k_ir.rank(), v_ir.rank()]
        .into_iter()
        .find(|rank| !matches!(rank, 3 | 4))
    {
        return Err(AttentionRefusal::Rank { rank });
    }
    // A dim the derivation reads that the graph has not pinned means no route.
    // The arm calls `unwrap_static` on these and panics; this must not, and a
    // route with an unknown sequence or width could not be sized anyway.
    if !(q_ir.is_static() && k_ir.is_static() && v_ir.is_static() && out_ir.is_static()) {
        return Err(AttentionRefusal::NonStatic);
    }
    let h = *num_heads as u32;
    let hd = *head_dim as u32;
    // A zero value-head width is the kernel's own spelling of "V is as wide as Q"
    // (`kernels/attention.wgsl` reads `v_head_dim` through a `select` that takes
    // `head_dim` when it is 0), so the plan resolves it the same way. Sizing V′,
    // the output region and the P·V multiply at width 0 instead would dispatch no
    // workgroups and leave the output slot unwritten.
    let hd_v = match *v_head_dim {
        Some(v) if v > 0 => v as u32,
        _ => hd,
    };
    // The one width above which neither the fused kernel (whose per-thread Q/O
    // arrays are `MAX_HEAD_DIM` wide) nor the dense route serves the step: the
    // kernel would return without writing the output slot, so the step is
    // refused here instead — naming the shape — rather than lowered to a blank.
    if hd > ATTENTION_DECOMPOSE_MAX_HEAD_DIM || hd_v > ATTENTION_DECOMPOSE_MAX_HEAD_DIM {
        return Err(AttentionRefusal::HeadWidth {
            head_dim: hd,
            v_head_dim: hd_v,
        });
    }
    let dim = |d: rlx_ir::shape::Dim| d.unwrap_static() as u32;
    // Rank-4's layout is the graph's to declare and not its extents' — see
    // [`rank4_geometry`], which is also what the backward arm reads. Rank-3 has
    // its own two spellings, told apart below by the last dim (the packed
    // `[B, S, H·D]` feature width against `[B·H, S, D]`).
    let (batch, seq_q, seq_k, bhsd) = match q_ir.rank() {
        4 => rank4_geometry(graph, q_id, k_id, h, hd)?,
        _ => {
            let (batch, seq_q, seq_k) = rank3_geometry(graph, q_id, k_id, h, hd)?;
            (batch, seq_q, seq_k, false)
        }
    };
    let (mask_kind_id, window) = match mask_kind {
        MaskKind::None => (0u32, 0u32),
        MaskKind::Causal => (1, 0),
        // 2 = binary key-padding mask (Custom: <0.5 → -inf); 4 = additive bias
        // mask (Bias: score += mask). These are NOT interchangeable — the
        // encoder's block-diagonal winmask is additive, so folding it into the
        // binary path silently corrupts attention.
        MaskKind::Custom => (2, 0),
        MaskKind::Bias => (4, 0),
        MaskKind::SlidingWindow(w) => (3, *w as u32),
        MaskKind::Window(w) => (5, *w as u32),
    };
    let mask_read = mask_kind_id == 2 || mask_kind_id == 4;
    if mask_read && node.inputs.len() < 4 {
        return Err(AttentionRefusal::MissingMask {
            kind: mask_kind_id,
            inputs: node.inputs.len(),
        });
    }
    if mask_read && !graph.node(node.inputs[3]).shape.is_static() {
        return Err(AttentionRefusal::NonStatic);
    }
    // Mask address strides, from the mask's own shape for the kinds that read
    // one and the canonical [B, H, S_q, S_k] form otherwise (the kernel's
    // mask_partial computation is harmless when the mask is not read).
    let (m_b, m_h, m_q, m_k) = if mask_read {
        rlx_ir::mask_strides_for_shape(graph.node(node.inputs[3]).shape.dims(), h, seq_q, seq_k)
    } else {
        rlx_ir::mask_strides_bhsd(h, seq_q, seq_k)
    };
    let packed_parent = if rlx_ir::env::flag("RLX_WGPU_NO_PACKED_BSHD_ATTN") {
        None
    } else {
        rlx_ir::detect_packed_bshd_qkv_attention(graph, q_id, k_id, v_id)
            .map(|(parent, head_width, _)| (parent, head_width as u32))
    };
    // GQA/MQA: the KV-head count is layout-independent — K's element count over
    // B·S_k·D. Packed QKV shares Q's head count (uniform strides), so it is MHA.
    let nkv: u32 = match packed_parent {
        Some(_) => h,
        None => {
            let k_elems = k_ir
                .dims()
                .iter()
                .fold(1u32, |elems, d| elems.saturating_mul(dim(*d)));
            let tiles = batch
                .max(1)
                .saturating_mul(seq_k.max(1))
                .saturating_mul(hd.max(1));
            (k_elems / tiles).max(1)
        }
    };
    let (q_b, q_h, q_s, k_b, k_h, k_s, v_b, v_h, v_s) = match packed_parent {
        Some((_, head_width)) => {
            let (b, head, pack_seq) =
                rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
            (b, head, pack_seq, b, head, pack_seq, b, head, pack_seq)
        }
        None => {
            let (qb, qh, qs) = rlx_ir::strides_for_shape(q_ir.dims(), h, hd, seq_q, bhsd);
            // K/V carry `nkv` heads (GQA): the layout test inside steps by
            // nkv·D, and V's rows are `hd_v` wide (MLA-asymmetric).
            let (kb, kh, ks) = rlx_ir::strides_for_shape(k_ir.dims(), nkv, hd, seq_k, bhsd);
            let (vb, vh, vs) = rlx_ir::strides_for_shape(v_ir.dims(), nkv, hd_v, seq_k, bhsd);
            (qb, qh, qs, kb, kh, ks, vb, vh, vs)
        }
    };
    // The output rows are `hd_v` wide (asymmetric SDPA == V's width).
    let (o_b, o_h, o_s) = rlx_ir::strides_for_shape(out_ir.dims(), h, hd_v, seq_q, bhsd);
    // A per-(b, h) tile that is dense is read or written where it lies; one that
    // is not is compacted through a transpose.
    let q_dense = operand_tile_is_dense(q_s, q_h, seq_q, hd);
    let v_dense = nkv == h && operand_tile_is_dense(v_s, v_h, seq_k, hd_v);
    let out_dense = operand_tile_is_dense(o_s, o_h, seq_q, hd_v);
    let regions = scratch_regions(ScratchShape {
        batch,
        heads: h,
        seq_q,
        seq_k,
        head_dim: hd,
        v_head_dim: hd_v,
        rows: seq_q,
        q_dense,
        v_dense,
        out_dense,
    });
    // V′'s region for the case where the default layout reads V where it lies but
    // the lowering still has to compact it: an output slot the planner placed on
    // V's own bytes cannot be written by the multiply that reads V (see the arm's
    // output-overlap note). Booked either way, so that case never needs more
    // scratch than the reserve holds.
    let v_spare = if v_dense {
        v_compaction_floats(batch, h, seq_k, hd_v)
    } else {
        0
    };
    // Kind 5 is served by the tiled band kernel and by nothing else, so its
    // geometry is settled here: a step the kernel cannot hold is refused with the
    // shapes in the message rather than lowered to the fused or dense
    // arrangement, both of which read kind 5 as "no mask" and answer a dense pass.
    // The decision is the graph's own (see `band_attention`), so it stays
    // testable without a device.
    //
    // A dense `MaskKind::None` step whose extents are square is served by the
    // same kernel with the band switched off (`window = seq_k`, i.e. `w >= seq`),
    // where the many-row arrangement's own score matrix no longer fits its
    // scratch cap. That is the cell this route exists for: past the cap the step
    // kept the fused one-thread-per-row walk (the stand's four global layers read
    // 83 s there before the routes below), and the band pass re-reads K and V once
    // per query block over the whole sequence instead. Below the cap the f32
    // many-row arrangement is both affordable and the tighter answer — its bar is
    // 1e-5 against this kernel's 5e-3-of-scale f16 tile — so the step keeps the
    // route it had, as it does where the band admission refuses and the
    // cooperative form is not booked for it.
    //
    // The band and many-row decisions below are about the *whole* sequence's
    // score matrix, so they read this layout even when the cooperative form's
    // chunked one is the layout the plan ends up booking. The two are then
    // mutually exclusive in the lowering by preference, not by admission: the
    // cooperative block is tried first, the band kernel is what the step keeps if
    // the adapter declines it, and the many-row arrangement the step had before
    // neither was booked.
    let full_regions = regions;
    let dense_scratch_fits = full_regions
        .total()
        .saturating_add(v_spare)
        .saturating_mul(4)
        <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES;
    // The cooperative dense square form. Its score region is a chunk of query
    // rows rather than all of them ([`DENSE_COOP_CHUNK_ROWS`]), so the plan books
    // that layout whenever the form is taken — the lowering bumps through exactly
    // the regions `scratch_offsets` returns, and the reserve reads
    // `scratch_bytes`. The shape admission is device-free; the adapter's share of
    // it (does it build the kernel, does the kernel compute `a·b`) is the
    // lowering's, and a step the adapter declines keeps the arrangement this flag
    // does not take.
    let coop_chunk = dense_coop_admits(mask_kind_id, seq_q, seq_k, hd, hd_v, batch, h, nkv);
    let coop_regions = coop_chunk.map(|chunk| {
        scratch_regions(ScratchShape {
            batch,
            heads: h,
            seq_q,
            seq_k,
            head_dim: hd,
            v_head_dim: hd_v,
            rows: chunk,
            q_dense,
            v_dense,
            out_dense,
        })
    });
    let dense_coop = coop_regions.is_some_and(|r| {
        r.total().saturating_add(v_spare).saturating_mul(4) <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES
    });
    let (regions, coop_chunk, coop_out_over_kt) = if dense_coop {
        let r = coop_regions.expect("admitted above");
        // The output region reuses Kᵀ's bytes where it is no larger than them —
        // see the field's own note. Kᵀ is `tiles · head_dim · seq_k` and the output
        // `tiles · chunk · v_head_dim` with `chunk <= seq_q` and `v_head_dim <=
        // head_dim`, so the chunked square is `seq_k · head_dim` over
        // `chunk · v_head_dim` smaller — four times at the encoder's geometry —
        // and an asymmetric-V step smaller still.
        (r, coop_chunk.unwrap_or(0), !out_dense && r.out <= r.kt)
    } else {
        (regions, 0, false)
    };
    // A band step with the band switched off is what a dense square step keeps
    // where the plan booked *chunked* regions for it: the many-row square cannot
    // follow (see `decompose` below). `!dense_scratch_fits || dense_coop` is
    // exactly the condition under which the plan wants the band kernel: the
    // many-row square is past its scratch cap, or the cooperative form booked a
    // chunk-sized region the square would run past. The one `band_attention`
    // answer is shared with `band_declined` so the admission is asked once.
    let band_admission = band_attention(seq_q, seq_k, hd, hd_v, h, nkv, batch as u64 * h as u64);
    let band_wanted = mask_kind_id == 0
        && seq_q == seq_k
        && seq_q >= ATTENTION_DECOMPOSE_MIN_Q
        && (!dense_scratch_fits || dense_coop);
    let band = match mask_kind_id {
        // Kind 5 is admitted here or refused: no other arrangement computes it.
        BAND_MASK_KIND => {
            band_admission?;
            true
        }
        // `MaskKind::None`: a refusal is a fallback rather than an error — the
        // step keeps the arrangement below.
        0 => band_wanted && band_admission.is_ok(),
        _ => false,
    };
    // A dense square step the band kernel would have taken — the many-row square is
    // past its cap, or the cooperative form booked chunk-sized regions — but whose
    // shape the admission refused. `!dense_coop` is what makes "nothing else is
    // left for it" true: without it, `band_wanted` reduces to the square past its
    // many-row scratch cap, where `decompose_attention` is false as well, so the
    // step keeps the fused walk. The lowering counts it so that fallback is visible
    // instead of silent. A step the cooperative form booked is served by that form,
    // so its band refusal is not a decline of that step's route and is not counted.
    let band_declined = band_wanted && band_admission.is_err() && !dense_coop;
    // The band kernel walks `[q0 - w, q0 + BM + w)` and masks keys outside
    // `|q - k| <= w`; `w >= seq` is its "no band" path, so a dense step is handed
    // the sequence length itself rather than `u32::MAX`, which would overflow the
    // address arithmetic (`q_abs + w`) in the mask test.
    let window = if band && mask_kind_id == 0 {
        seq_k
    } else {
        window
    };
    // The shape half of the decision, plus the one layout condition it needs:
    // grouped KV reads K and V through one (b, h) tile per query head, so the KV
    // group has to divide the head count exactly — `heads / nkv` is the fused
    // kernel's own kv-head divisor, and the route's Kᵀ and V′ transposes walk
    // that group as an axis. A shape where it does not divide is one that kernel
    // already indexes past its KV heads.
    //
    // `!dense_coop` is not a preference: where the plan books the cooperative
    // form's regions they are sized at one chunk of query rows
    // ([`DENSE_COOP_CHUNK_ROWS`]), while this arrangement's `scores` region holds
    // the whole square — so taking it as the cooperative form's fallback would
    // write past the booked reserve. It is the band kernel's or the fused walk's,
    // never this arrangement's.
    let decompose = !band
        && !dense_coop
        && decompose_attention(&AttentionGeometry {
            seq_q,
            seq_k,
            head_dim: hd,
            v_head_dim: hd_v,
            tiles: batch as u64 * h as u64,
            scratch_floats: full_regions.total() + v_spare,
        })
        && h.is_multiple_of(nkv.max(1));
    Ok(AttentionPlan {
        batch,
        heads: h,
        seq_q,
        seq_k,
        head_dim: hd,
        v_head_dim: hd_v,
        nkv,
        packed_parent,
        q_b,
        q_h,
        q_s,
        k_b,
        k_h,
        k_s,
        v_b,
        v_h,
        v_s,
        o_b,
        o_h,
        o_s,
        mask_kind_id,
        window,
        m_b,
        m_h,
        m_q,
        m_k,
        q_dense,
        v_dense,
        out_dense,
        regions,
        v_spare,
        decompose,
        band,
        band_declined,
        dense_coop,
        coop_chunk,
        coop_out_over_kt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Dim, Graph, Op, Shape};

    /// The measured shape family: 8 heads of head_dim 64, both widths clear of
    /// the ceiling, a single batch, and the scratch computed from the same
    /// arguments the lowering passes (all three tiles compacted, the layout that
    /// needs the most).
    fn geom(seq_q: u32, seq_k: u32) -> AttentionGeometry {
        let (batch, heads, head_dim, v_head_dim) = (1u32, 8u32, 64u32, 64u32);
        AttentionGeometry {
            seq_q,
            seq_k,
            head_dim,
            v_head_dim,
            tiles: batch as u64 * heads as u64,
            scratch_floats: scratch_regions(ScratchShape {
                batch,
                heads,
                seq_q,
                seq_k,
                head_dim,
                v_head_dim,
                rows: seq_q,
                q_dense: false,
                v_dense: false,
                out_dense: false,
            })
            .total(),
        }
    }

    /// One `Op::Attention` node over Q/K/V inputs of the given shapes, and the
    /// plan its own geometry derives. No mask: the kinds that read one are
    /// exercised below by hand.
    fn plan_of(
        q: &[usize],
        k: &[usize],
        v: &[usize],
        out: &[usize],
        heads: usize,
        head_dim: usize,
        v_head_dim: Option<usize>,
    ) -> Result<AttentionPlan, AttentionRefusal> {
        let mut g = Graph::new("attention_plan");
        let (q_id, k_id, v_id) = (
            g.input("q", Shape::new(q, DType::F32)),
            g.input("k", Shape::new(k, DType::F32)),
            g.input("v", Shape::new(v, DType::F32)),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: heads,
                head_dim,
                v_head_dim,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_id, k_id, v_id],
            Shape::new(out, DType::F32),
        );
        plan_attention(&g, g.node(attn))
    }

    /// How a rank-4 operand reaches the step: as a bare input (nothing declares
    /// its layout), unrolled out of a rank-3 packed `[B, S, H·D]` projection, or
    /// out of the packed QKV parent's axis-2 narrow — the spellings this tree
    /// builds attention operands with — or swapped 1↔2 out of the unrolled one.
    #[derive(Clone, Copy)]
    enum Presented {
        Bare,
        Unrolled,
        Packed,
        Swapped,
    }

    /// One rank-4 attention step of `rows` query rows over `lk` keys presented
    /// the way `how` says. The operand's extents are `[B, rows, H, D]` /
    /// `[B, lk, H, D]` under both readings, so at `rows == H` the extents fit
    /// either one and only the producers tell them apart. A packed QKV parent
    /// carries one sequence extent for all three operands, so that spelling is a
    /// self-attention step (`rows` is both).
    fn presented_graph(
        b: usize,
        rows: usize,
        lk: usize,
        nh: usize,
        hd: usize,
        how: Presented,
    ) -> (Graph, rlx_ir::NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("attention_presented");
        let dim = nh * hd;
        let unroll = |g: &mut Graph, name: &str, seq: usize| -> rlx_ir::NodeId {
            let packed = g.input(name, Shape::new(&[b, seq, dim], f));
            g.add_node(
                Op::Reshape {
                    new_shape: vec![b as i64, seq as i64, nh as i64, hd as i64],
                },
                vec![packed],
                Shape::new(&[b, seq, nh, hd], f),
            )
        };
        let (q, k, v, out) = match how {
            Presented::Bare => (
                g.input("q", Shape::new(&[b, rows, nh, hd], f)),
                g.input("k", Shape::new(&[b, lk, nh, hd], f)),
                g.input("v", Shape::new(&[b, lk, nh, hd], f)),
                Shape::new(&[b, rows, nh, hd], f),
            ),
            Presented::Unrolled => (
                unroll(&mut g, "q", rows),
                unroll(&mut g, "k", lk),
                unroll(&mut g, "v", lk),
                Shape::new(&[b, rows, nh, hd], f),
            ),
            Presented::Swapped => {
                let swap = |g: &mut Graph, name: &str, seq: usize| -> rlx_ir::NodeId {
                    let unrolled = unroll(g, name, seq);
                    g.add_node(
                        Op::Transpose {
                            perm: vec![0, 2, 1, 3],
                        },
                        vec![unrolled],
                        Shape::new(&[b, nh, seq, hd], f),
                    )
                };
                (
                    swap(&mut g, "q", rows),
                    swap(&mut g, "k", lk),
                    swap(&mut g, "v", lk),
                    Shape::new(&[b, nh, rows, hd], f),
                )
            }
            Presented::Packed => {
                let parent = g.input("qkv", Shape::new(&[b, rows, 3, nh, hd], f));
                let narrow = |g: &mut Graph, start: usize| -> rlx_ir::NodeId {
                    let view = g.add_node(
                        Op::Narrow {
                            axis: 2,
                            start,
                            len: 1,
                        },
                        vec![parent],
                        Shape::new(&[b, rows, 1, nh, hd], f),
                    );
                    g.add_node(
                        Op::Reshape {
                            new_shape: vec![b as i64, rows as i64, nh as i64, hd as i64],
                        },
                        vec![view],
                        Shape::new(&[b, rows, nh, hd], f),
                    )
                };
                (
                    narrow(&mut g, 0),
                    narrow(&mut g, 1),
                    narrow(&mut g, 2),
                    Shape::new(&[b, rows, nh, hd], f),
                )
            }
        };
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            out,
        );
        (g, attn)
    }

    /// [`presented_graph`], and the plan its own graph derives.
    fn presented(
        b: usize,
        rows: usize,
        lk: usize,
        nh: usize,
        hd: usize,
        how: Presented,
    ) -> Result<AttentionPlan, AttentionRefusal> {
        let (g, attn) = presented_graph(b, rows, lk, nh, hd, how);
        plan_attention(&g, g.node(attn))
    }

    /// A test that reads the packed-parent path needs the reading it is gated by
    /// on: it is the default, and this forces it — `set("0")` overrides an exported
    /// `RLX_WGPU_NO_PACKED_BSHD_ATTN`, where `unset` would fall back to it.
    fn packed_parent_reading_on() {
        rlx_ir::env::set("RLX_WGPU_NO_PACKED_BSHD_ATTN", "0");
    }

    /// The floor, and the reading it rests on: the fork's cached decode cell —
    /// two rows over a full cache — was measured and is served from two rows,
    /// while one row stays with the decode kernel, which is cheaper still. The
    /// live row count is the whole term: the key length does not change the
    /// answer.
    #[test]
    fn the_floor_is_the_measured_one() {
        assert!(
            !decompose_attention(&geom(1, 1024)),
            "one query row is the decode kernel's, and cheaper there"
        );
        assert!(decompose_attention(&geom(2, 2)));
        assert!(decompose_attention(&geom(2, 32)));
        assert!(decompose_attention(&geom(2, 512)));
        assert!(decompose_attention(&geom(24, 512)));
        assert!(decompose_attention(&geom(31, 512)));
        assert!(decompose_attention(&geom(32, 512)));
        assert!(decompose_attention(&geom(100, 1024)));
    }

    /// The head ceiling is the fused kernel's: the width equal to it is served
    /// by both forms, one past it neither. Applies to V's own width as well,
    /// since the shared `MAX_HEAD_DIM` is compared against both.
    #[test]
    fn the_head_ceiling_is_the_fused_kernels_own() {
        let mut at = geom(64, 256);
        at.head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM;
        at.v_head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM;
        assert!(decompose_attention(&at));
        let mut over = geom(64, 256);
        over.head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM + 1;
        assert!(!decompose_attention(&over));
        let mut wide_v = geom(64, 256);
        wide_v.v_head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM + 1;
        assert!(!decompose_attention(&wide_v));
    }

    /// A shape past that ceiling is one the fused path *can* size — and whose
    /// kernel then returns without writing the output slot — so it has no plan at
    /// all rather than a plan the arm would lower to a blank. The refusal names
    /// the operator and the width, and the plan-side ceiling is the shader's own
    /// `MAX_HEAD_DIM`, so the two cannot be raised apart (the silent blank
    /// returns if only this one moves).
    #[test]
    fn a_head_width_past_the_ceiling_is_refused_rather_than_left_blank() {
        const SHADER: &str = include_str!("../kernels/attention.wgsl");
        assert!(
            SHADER.contains(&format!(
                "const MAX_HEAD_DIM: u32 = {ATTENTION_DECOMPOSE_MAX_HEAD_DIM}u;"
            )),
            "the plan's ceiling and the fused kernel's `MAX_HEAD_DIM` are one number: a shape \
             the plan serves must not be one the kernel early-returns from"
        );

        // The widths are the whole subject; the other dims are a served shape's.
        // `v_head_dim` of 0 is the kernel's own "as wide as Q" spelling, so the
        // shapes resolve it the same way the plan does.
        let plan = |hd: usize, v_hd: Option<usize>| {
            let vd = match v_hd {
                Some(v) if v > 0 => v,
                _ => hd,
            };
            plan_of(
                &[1, 4, hd],
                &[1, 8, hd],
                &[1, 8, vd],
                &[1, 4, vd],
                1,
                hd,
                v_hd,
            )
        };

        let at = ATTENTION_DECOMPOSE_MAX_HEAD_DIM as usize;
        assert!(
            plan(at, None).is_ok(),
            "the width equal to the ceiling is served"
        );
        assert_eq!(
            plan(at + 1, None).unwrap_err(),
            AttentionRefusal::HeadWidth {
                head_dim: at as u32 + 1,
                v_head_dim: at as u32 + 1,
            }
        );
        // V's own width, which the kernel caps the same way, and the 0 spelling.
        assert_eq!(
            plan(at, Some(at + 1)).unwrap_err(),
            AttentionRefusal::HeadWidth {
                head_dim: at as u32,
                v_head_dim: at as u32 + 1,
            }
        );
        assert!(
            plan(64, Some(0)).is_ok(),
            "a zero width means V is as wide as Q"
        );

        // The refusal names the operator and the shape, so a caller can act on it
        // rather than read it as a crash.
        let msg = AttentionRefusal::HeadWidth {
            head_dim: at as u32 + 1,
            v_head_dim: at as u32 + 1,
        }
        .to_string();
        assert!(msg.contains("Attention"), "the operator: {msg}");
        assert!(msg.contains(&(at + 1).to_string()), "the width: {msg}");
    }

    /// Both dispatch caps the multiplies walk are shape terms, at the tree's own
    /// bound rather than an adapter query: their tiles come out as the dispatch's
    /// z and their `m = seq_q` rows as its y, so a shape at either bound is served
    /// here and one tile past it is not. The y case is built by hand because its
    /// key sequence is tiny and its scratch field is the decision's own, not a
    /// graph's — setting it small leaves the other conjuncts standing.
    #[test]
    fn the_dispatch_caps_bound_both_extents_the_multiplies_walk() {
        let at_cap = AttentionGeometry {
            tiles: MAX_WORKGROUPS_PER_DIM as u64,
            ..geom(32, 1024)
        };
        assert!(decompose_attention(&at_cap));
        let over = AttentionGeometry {
            tiles: MAX_WORKGROUPS_PER_DIM as u64 + 1,
            ..geom(32, 1024)
        };
        assert!(!decompose_attention(&over));
        let rows = MAX_WORKGROUPS_PER_DIM as u64 * MATMUL_TILE_M as u64;
        let at_rows = AttentionGeometry {
            seq_q: rows as u32,
            scratch_floats: 1,
            ..geom(32, 1024)
        };
        assert!(decompose_attention(&at_rows));
        let over_rows = AttentionGeometry {
            seq_q: rows as u32 + MATMUL_TILE_M,
            ..at_rows
        };
        assert!(!decompose_attention(&over_rows));
        // No tiles at all: nothing for either multiply to dispatch, whatever the
        // key sequence is (the empty batch the fused path still runs).
        let no_tiles = AttentionGeometry {
            tiles: 0,
            ..geom(32, 1024)
        };
        assert!(!decompose_attention(&no_tiles));
    }

    /// The degenerate key sequence: every region would be empty and the budget
    /// conjunct would pass vacuously, so the route would take a shape whose
    /// masked scale, softmax and P·V all write nothing — where the fused kernel
    /// it replaces normalises `0/0` into the NaN the operation is. One key is
    /// enough to serve it, so it is the emptiness and not the length that refuses.
    #[test]
    fn an_empty_key_sequence_keeps_the_fused_kernel() {
        let empty = AttentionGeometry {
            seq_k: 0,
            scratch_floats: 0,
            ..geom(2, 1024)
        };
        assert!(!decompose_attention(&empty));
        assert!(decompose_attention(&AttentionGeometry {
            seq_k: 1,
            scratch_floats: 1,
            ..empty
        }));
    }

    /// The budget's own boundary, on the field the decision reads — the cap
    /// itself is admitted and one float past it is refused — and a shape the
    /// budget alone refuses: 4096 × 4096 keys per head is the cap in scores
    /// before Kᵀ, so the three other regions put it over the reserve every graph
    /// is guaranteed, while the other three conjuncts hold.
    #[test]
    fn the_scratch_budget_admits_its_own_figure() {
        let at = AttentionGeometry {
            seq_q: 64,
            seq_k: 128,
            head_dim: 64,
            v_head_dim: 64,
            tiles: 8,
            scratch_floats: ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES / 4,
        };
        assert!(decompose_attention(&at));
        let over = AttentionGeometry {
            scratch_floats: at.scratch_floats + 1,
            ..at
        };
        assert!(!decompose_attention(&over));
        let huge = geom(4096, 4096);
        assert!(huge.seq_q >= ATTENTION_DECOMPOSE_MIN_Q);
        assert!(huge.head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM);
        assert!(huge.scratch_floats * 4 > ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES);
        assert!(!decompose_attention(&huge));
    }

    /// The regions a layout forces: Kᵀ and the scores are always emitted, each
    /// operand's own region dropped exactly when its tiles are dense, every
    /// region a multiple of the alignment and never smaller than the store it
    /// holds — so size order is the bump order, a region's offset the sum before
    /// it, and the bump cannot run short.
    #[test]
    fn regions_are_aligned_and_dropped_by_dense_operands() {
        let shape = |q_dense, v_dense, out_dense| ScratchShape {
            batch: 2,
            heads: 4,
            seq_q: 64,
            seq_k: 128,
            head_dim: 64,
            v_head_dim: 64,
            rows: 64,
            q_dense,
            v_dense,
            out_dense,
        };
        let all_compacted = scratch_regions(shape(false, false, false));
        let all_dense = scratch_regions(shape(true, true, true));
        assert_eq!(all_compacted.kt, 2 * 4 * 64 * 128);
        assert_eq!(all_compacted.scores, 2 * 4 * 64 * 128);
        assert_eq!(all_compacted.q, 2 * 4 * 64 * 64);
        assert_eq!(all_compacted.v, 2 * 4 * 128 * 64);
        assert_eq!(all_compacted.out, 2 * 4 * 64 * 64);
        assert_eq!(all_dense.total(), all_compacted.kt + all_compacted.scores);
        assert!(all_compacted.total() > all_dense.total());
        // A tile that is dense is read where it lies; one that is not is
        // compacted. The two predicates are the caller's, but the strides that
        // decide them are compared here at the boundary they describe.
        assert!(operand_tile_is_dense(64, 64 * 64, 64, 64));
        assert!(!operand_tile_is_dense(64 * 8, 64, 64, 64));
        for (seq_q, seq_k) in [(32u32, 33u32), (64, 1024), (100, 128)] {
            let r = scratch_regions(ScratchShape {
                batch: 1,
                heads: 8,
                seq_q,
                seq_k,
                head_dim: 64,
                v_head_dim: 64,
                rows: seq_q,
                q_dense: false,
                v_dense: true,
                out_dense: false,
            });
            assert_eq!(r.kt % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.scores % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.q % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.out % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.v, 0, "a dense V needs no region");
            assert!(r.kt >= 8 * 64 * seq_k as u64);
            assert!(r.scores >= 8 * seq_q as u64 * seq_k as u64);
        }
    }

    /// The derivation `plan_attention` shares with the lowering: a dense rank-4
    /// layout, a strided rank-3 one, a grouped-KV step and an asymmetric
    /// `v_head_dim` one, each read off the plan rather than restated — then the
    /// two shapes it refuses, with the messages the arm gives them, and the
    /// shapes it declines without refusing.
    #[test]
    fn the_plan_derives_the_geometry_the_route_reads() {
        // Rank-4 [B, H, S, D]: every per-(b, h) tile is a dense [seq, width]
        // block, so Kᵀ and the scores are the route's only regions.
        let dense = plan_of(
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("a rank-4 plan");
        assert_eq!(
            (dense.batch, dense.heads, dense.seq_q, dense.seq_k),
            (2, 4, 64, 64)
        );
        assert_eq!((dense.head_dim, dense.v_head_dim, dense.nkv), (64, 64, 4));
        assert_eq!((dense.nkv, dense.mask_kind_id, dense.window), (4, 0, 0));
        assert_eq!(
            (dense.q_b, dense.q_h, dense.q_s),
            (4 * 64 * 64, 64 * 64, 64)
        );
        assert_eq!(
            (dense.o_b, dense.o_h, dense.o_s),
            (4 * 64 * 64, 64 * 64, 64)
        );
        assert_eq!(
            (dense.m_b, dense.m_h, dense.m_q, dense.m_k),
            (4 * 64 * 64, 64 * 64, 64, 1)
        );
        assert!(dense.q_dense && dense.v_dense && dense.out_dense);
        assert_eq!(
            (dense.regions.q, dense.regions.v, dense.regions.out),
            (0, 0, 0)
        );
        assert_eq!(
            dense.regions.total(),
            dense.regions.kt + dense.regions.scores
        );
        assert!(dense.regions.kt > 0 && dense.regions.scores > 0);
        assert_eq!(dense.packed_parent, None);
        assert!(dense.decompose);
        // V's tiles are dense, so the default layout reads V where it lies — but
        // V′'s region is booked anyway, for the output slot the planner may have
        // placed on V (see the arm's output-overlap note).
        assert_eq!(dense.v_spare, 2 * 4 * 64 * 64);
        assert_eq!(
            dense.scratch_bytes(),
            ((dense.regions.total() + dense.v_spare) * 4 + SCRATCH_ALIGN_SLACK_BYTES) as usize
        );

        // Rank-3 [B, S, H·D], told from [B·H, S, D] by its last dim: its tiles
        // are strided, so every operand region is emitted — and the step is still
        // the route's, 16 query rows being well above the floor.
        let strided = plan_of(
            &[2, 16, 256],
            &[2, 16, 256],
            &[2, 16, 256],
            &[2, 16, 256],
            4,
            64,
            None,
        )
        .expect("a rank-3 plan");
        assert_eq!(
            (strided.batch, strided.heads, strided.seq_q, strided.seq_k),
            (2, 4, 16, 16)
        );
        assert_eq!((strided.q_b, strided.q_h, strided.q_s), (16 * 256, 64, 256));
        assert!(!strided.q_dense && !strided.v_dense && !strided.out_dense);
        assert_eq!(strided.regions.q, 2 * 4 * 16 * 64);
        assert!(strided.decompose, "16 query rows are above the floor");

        // Grouped KV (nkv < heads): Q's and the output's tiles stay dense, V's
        // does not — the route reads one (b, h) V tile per query head.
        let gqa = plan_of(
            &[1, 8, 32, 64],
            &[1, 2, 8, 64],
            &[1, 2, 8, 64],
            &[1, 8, 32, 64],
            8,
            64,
            None,
        )
        .expect("a grouped-KV plan");
        assert_eq!(
            (gqa.batch, gqa.heads, gqa.seq_q, gqa.seq_k, gqa.nkv),
            (1, 8, 32, 8, 2)
        );
        assert!(gqa.q_dense && gqa.out_dense);
        assert!(!gqa.v_dense);
        assert_eq!(gqa.regions.v, 8 * 8 * 64);
        assert_eq!((gqa.k_b, gqa.k_h, gqa.k_s), (2 * 64 * 8, 8 * 64, 64));

        // Rank-4 [B, S, H, D] (the CPU / EEG-DINO convention), on a shape whose
        // own extents settle which way it runs — axis 2 is the 8 heads, axis 1 is
        // 32 rows, and no operand's production is consulted. Every tile is strided
        // — seq stride `H·D`, not the head width — so all three operand regions
        // are emitted on a step the floor admits.
        let bshd = plan_of(
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            8,
            64,
            None,
        )
        .expect("a rank-4 BSHD plan");
        assert_eq!(
            (bshd.batch, bshd.heads, bshd.seq_q, bshd.seq_k),
            (1, 8, 32, 32)
        );
        assert_eq!((bshd.q_h, bshd.q_s, bshd.o_s), (64, 8 * 64, 8 * 64));
        assert!(!bshd.q_dense && !bshd.v_dense && !bshd.out_dense);
        assert!(bshd.decompose);

        // Packed QKV: Q/K/V are axis-2 narrows of one [B, S, 3, H, D] parent,
        // read through the parent's own strides so the KV-head count is Q's.
        packed_parent_reading_on();
        let mut packed = Graph::new("packed_qkv");
        let parent = packed.input("fmb", Shape::new(&[1, 4, 3, 2, 8], DType::F32));
        let shape_5 = Shape::new(&[1, 4, 1, 2, 8], DType::F32);
        let heads_in: Vec<rlx_ir::NodeId> = (0..3)
            .map(|start| {
                let narrow = packed.add_node(
                    Op::Narrow {
                        axis: 2,
                        start,
                        len: 1,
                    },
                    vec![parent],
                    shape_5.clone(),
                );
                packed.add_node(
                    Op::Reshape {
                        new_shape: vec![1, 4, 2, 8],
                    },
                    vec![narrow],
                    Shape::new(&[1, 4, 2, 8], DType::F32),
                )
            })
            .collect();
        let attn = packed.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 8,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            heads_in,
            Shape::new(&[1, 4, 2, 8], DType::F32),
        );
        let plan = plan_attention(&packed, packed.node(attn)).expect("a packed-QKV plan");
        // One head is `1 · H · D` = 16 elements, so the packed sequence stride is
        // 3 · 16 and no operand of it is a dense tile.
        assert_eq!(plan.packed_parent, Some((parent, 16)));
        assert_eq!((plan.heads, plan.nkv, plan.seq_q), (2, 2, 4));
        assert_eq!((plan.q_s, plan.k_s, plan.v_s), (48, 48, 48));
        assert!(!plan.q_dense && !plan.v_dense && !plan.out_dense);
        assert!(plan.decompose, "4 query rows are above the floor");

        // Asymmetric SDPA: V's rows and the output are `v_head_dim` wide while
        // the scores still contract over `head_dim`.
        let mla = plan_of(
            &[1, 2, 32, 64],
            &[1, 2, 16, 64],
            &[1, 2, 16, 96],
            &[1, 2, 32, 96],
            2,
            64,
            Some(96),
        )
        .expect("an asymmetric plan");
        assert_eq!((mla.head_dim, mla.v_head_dim), (64, 96));
        assert_eq!((mla.seq_q, mla.seq_k), (32, 16));
        assert_eq!((mla.k_s, mla.v_s, mla.o_s), (64, 96, 96));
        assert_eq!((mla.v_b, mla.v_h, mla.o_h), (2 * 96 * 16, 16 * 96, 32 * 96));
        assert!(mla.decompose);

        // A zero value-head width is the kernel's own "V is as wide as Q": the
        // plan resolves it to `head_dim` rather than sizing V′, the output and
        // the P·V multiply at width 0 — which would emit a multiply with no
        // workgroups and leave the output slot unwritten.
        let zero_v = plan_of(
            &[1, 2, 32, 64],
            &[1, 2, 16, 64],
            &[1, 2, 16, 64],
            &[1, 2, 32, 64],
            2,
            64,
            Some(0),
        )
        .expect("a zero value-width plan");
        assert_eq!((zero_v.head_dim, zero_v.v_head_dim), (64, 64));
        assert_eq!((zero_v.v_s, zero_v.o_s), (64, 64));
        assert_eq!(zero_v.regions.out, 0);
        assert!(zero_v.decompose);

        // The two shapes no plan can be derived for, with the message the arm
        // panics with today.
        let rank = plan_of(
            &[1, 2, 32, 64, 1],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            2,
            64,
            None,
        )
        .unwrap_err();
        assert_eq!(rank, AttentionRefusal::Rank { rank: 5 });
        assert_eq!(
            rank.to_string(),
            "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V inputs supported (got rank 5)"
        );
        let leading = plan_of(
            &[7, 32, 3],
            &[7, 32, 3],
            &[7, 32, 3],
            &[7, 32, 3],
            2,
            64,
            None,
        )
        .unwrap_err();
        assert_eq!(
            leading.to_string(),
            "rlx-wgpu Attention: rank-3 leading dim 7 not divisible by num_heads 2 \
             (and last dim 3 ≠ H·D = 128)"
        );
        let mut g = Graph::new("maskless");
        let (q, k, v) = (
            g.input("q", Shape::new(&[1, 2, 32, 64], DType::F32)),
            g.input("k", Shape::new(&[1, 2, 32, 64], DType::F32)),
            g.input("v", Shape::new(&[1, 2, 32, 64], DType::F32)),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 64,
                v_head_dim: None,
                mask_kind: MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 2, 32, 64], DType::F32),
        );
        assert_eq!(
            plan_attention(&g, g.node(attn)).unwrap_err().to_string(),
            "rlx-wgpu Attention: mask kind 2 needs a mask input (node has 3 inputs)"
        );

        // A node that is not attention at all is a refusal too — the reserve
        // walks every node of a graph, and reads only the plans that name it a
        // route.
        let mut plain = Graph::new("plain");
        let x = plain.input("x", Shape::new(&[1, 2, 32, 64], DType::F32));
        assert_eq!(
            plan_attention(&plain, plain.node(x)).unwrap_err(),
            AttentionRefusal::NotAttention
        );
        // …and a shape under the floor is still a plan, with the route off it.
        let under = plan_of(
            &[1, 2, 1, 64],
            &[1, 2, 4, 64],
            &[1, 2, 4, 64],
            &[1, 2, 1, 64],
            2,
            64,
            None,
        )
        .expect("a plan under the floor");
        assert!(!under.decompose);
    }

    /// Grouped KV that does not divide the query heads is a shape the route
    /// declines even above its floor: its Kᵀ and V′ transpose K and V through a
    /// group axis, and `heads / nkv` is the fused kernel's own kv-head divisor.
    /// Both sides of the rule, on the layout the plan derives `nkv` from (K's
    /// element count over B·S_k·D).
    #[test]
    fn the_kv_group_must_divide_the_head_count() {
        let divides = plan_of(
            &[1, 4, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 4, 32, 64],
            4,
            64,
            None,
        )
        .expect("a grouped-KV plan");
        assert_eq!((divides.heads, divides.nkv), (4, 2));
        assert!(divides.decompose, "4 heads over 2 KV heads is served");

        let leaves_one_out = plan_of(
            &[1, 3, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 3, 32, 64],
            3,
            64,
            None,
        )
        .expect("a plan with an indivisible group");
        assert_eq!((leaves_one_out.heads, leaves_one_out.nkv), (3, 2));
        assert!(
            leaves_one_out.seq_q >= ATTENTION_DECOMPOSE_MIN_Q,
            "the floor must not be what refuses this shape"
        );
        assert!(
            !leaves_one_out.decompose,
            "3 heads over 2 KV heads is not a layout the route's group axis can walk"
        );
    }

    /// The degenerate empty batch the fused path still runs, and the reason the
    /// route must not take it: with no `(batch, head)` tiles there is no multiply
    /// to dispatch, and `tiles · c` — the coefficient the `n`-extent argument
    /// above bounds — is zero however long the key sequence is, so a shape no
    /// dispatch could run would pass every other conjunct. The fused path's grid
    /// is folded across the dispatch's own dimensions and takes it, as it did
    /// before.
    #[test]
    fn an_empty_batch_keeps_the_fused_kernel() {
        let empty = plan_of(
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("an empty-batch plan");
        assert_eq!((empty.batch, empty.seq_q), (0, 64));
        assert!(!empty.decompose, "no tiles means no multiply to dispatch");
        // One tile and the same lengths are served, so it is the empty batch and
        // not the geometry that refuses.
        let served = plan_of(
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("a one-batch plan");
        assert!(served.decompose);
    }

    /// The core pin: an operand whose first two axes are both the head count fits
    /// either reading, so the reading is its graph's and not its extents' — the
    /// key length comes off the row axis even where the head axis sits on it —
    /// and a presentation that declares neither is refused by name rather than
    /// read off the head axis.
    #[test]
    fn a_coincident_extent_is_read_the_way_the_graph_lays_the_operand_out() {
        const NH: usize = 16;
        const HD: usize = 64;
        const ROWS: usize = 16;
        const LK: usize = 40;
        let (nh, hd) = (NH as u32, HD as u32);

        // Rows-first, unrolled: the rank-3 packed projection reshaped. K's axis 2
        // is the head count and its axis 1 the key count, and `seq_k` is read off
        // axis 1 — 40, not 16 — which is what a reading taken from the extents
        // would have got wrong.
        let unrolled =
            presented(1, ROWS, LK, NH, HD, Presented::Unrolled).expect("a declared plan");
        assert_eq!(
            (unrolled.batch, unrolled.seq_q, unrolled.seq_k),
            (1, 16, 40)
        );
        assert_eq!(
            unrolled.nkv, nh,
            "the KV heads are K's own, not the key count"
        );
        assert_eq!(
            (unrolled.q_s, unrolled.q_h, unrolled.o_s),
            (nh * hd, hd, nh * hd),
            "a rows-first operand's sequence stride is the whole head width"
        );
        assert!(!unrolled.q_dense && !unrolled.v_dense && !unrolled.out_dense);
        assert_eq!(unrolled.packed_parent, None);

        // The packed spelling the bodies build: one `[B, S, 3, H, D]` parent whose
        // axis-2 narrows are unrolled per operand, at the self-attention length
        // that spelling has. Its axes 1 and 2 are the head count twice over, so
        // the extents fit either reading here too — the parent under the reading
        // its own strides give.
        packed_parent_reading_on();
        let packed = presented(1, ROWS, ROWS, NH, HD, Presented::Packed).expect("a packed plan");
        assert!(packed.packed_parent.is_some(), "the packed QKV parent");
        assert_eq!(
            (packed.batch, packed.seq_q, packed.seq_k, packed.nkv),
            (1, 16, 16, 16)
        );
        // …and its strides are the parent's own: one head width per pack slot
        // along the sequence, `3 · H · D`, and the head width across it.
        assert_eq!(
            (packed.q_s, packed.k_s, packed.v_s, packed.q_h),
            (3 * nh * hd, 3 * nh * hd, 3 * nh * hd, hd)
        );
        // The output is not packed: `H · D` along the sequence, which is the
        // number the extents-only reading flips to the head width.
        assert_eq!(packed.o_s, nh * hd);
        assert!(!packed.q_dense && !packed.v_dense && !packed.out_dense);

        // The other direction: the unrolled operand swapped 1↔2. Its extents are
        // the same numbers, so this is the case that would come out rows-first if
        // the graph were not read — and the swap is what makes it heads-first,
        // read rather than refused.
        let swapped = presented(1, ROWS, LK, NH, HD, Presented::Swapped)
            .expect("a declared heads-first operand must be read, not refused");
        assert_eq!((swapped.batch, swapped.seq_q, swapped.seq_k), (1, 16, 40));
        assert_eq!(
            (swapped.q_s, swapped.q_h, swapped.o_s),
            (hd, ROWS as u32 * hd, hd),
            "the head axis is axis 1, so the head width is the sequence stride"
        );
        assert!(swapped.q_dense && swapped.v_dense && swapped.out_dense);
        assert_eq!(swapped.nkv, nh);

        // Nothing declares this one: a bare rank-4 input at the same extents has
        // no plan at all, and the refusal names the head count — which is both of
        // the operand's first two extents there — rather than reading a length off
        // it.
        let bare = presented(1, ROWS, LK, NH, HD, Presented::Bare).unwrap_err();
        assert_eq!(
            bare,
            AttentionRefusal::AmbiguousRank4 {
                heads: 16,
                head_dim: 64,
            }
        );
        // The message names the ambiguity, the extent both readings fit and the two
        // spellings that would declare one — the parts a caller could act on, not
        // the whole sentence.
        let msg = bare.to_string();
        assert!(msg.contains("rank-4 layout is ambiguous"), "{msg}");
        assert!(msg.contains("axis 1 and axis 2 are both 16"), "{msg}");
        assert!(msg.contains("swap axes 1 and 2"), "{msg}");
    }

    /// The coincidence is not a property of one head count: sixteen is the count
    /// the measured band compiles at, and this is the rule the pin above reads, at
    /// 4, 8, 16 and 32 heads.
    #[test]
    fn the_coincident_extent_is_not_a_property_of_one_head_count() {
        const HD: usize = 64;
        for nh in [4usize, 8, 16, 32] {
            let lk = nh + 8;
            let plan = presented(1, nh, lk, nh, HD, Presented::Unrolled).expect("a declared plan");
            assert_eq!(
                (plan.seq_q, plan.seq_k, plan.nkv),
                (nh as u32, lk as u32, nh as u32),
                "heads={nh}: the reading is the reshape's, so the key count is K's axis 1"
            );
            assert_eq!((plan.q_s, plan.q_h), (nh as u32 * HD as u32, HD as u32));
            assert!(presented(1, nh, lk, nh, HD, Presented::Bare).is_err());
        }
    }

    /// The other side of the rule: where the extents settle the layout the graph
    /// is not consulted at all — a bare input and a declared operand of the same
    /// shape derive one plan, at 13 rows and at 32 — and the layout is a tie-break
    /// rather than a second reading, so no number moves off the shape-only answer.
    #[test]
    fn the_extents_still_settle_the_layout_where_they_can() {
        const NH: usize = 16;
        const HD: usize = 64;
        const LK: usize = 40;
        for rows in [13usize, 32] {
            let bare = presented(1, rows, LK, NH, HD, Presented::Bare).expect("a plan");
            let unrolled = presented(1, rows, LK, NH, HD, Presented::Unrolled).expect("a plan");
            // Every field a layout can move, so the two plans are one plan.
            assert_eq!(
                (bare.batch, bare.seq_q, bare.seq_k, bare.nkv),
                (unrolled.batch, unrolled.seq_q, unrolled.seq_k, unrolled.nkv),
                "rows={rows} != heads={NH}: the extents settle the layout, so the graph's own \
                 declaration cannot move a plan"
            );
            assert_eq!(
                (
                    bare.q_b,
                    bare.q_h,
                    bare.q_s,
                    bare.o_s,
                    bare.regions,
                    bare.decompose
                ),
                (
                    unrolled.q_b,
                    unrolled.q_h,
                    unrolled.q_s,
                    unrolled.o_s,
                    unrolled.regions,
                    unrolled.decompose
                ),
                "rows={rows}: one plan, and not just the same extents"
            );
            // The numbers are the shape-only ones: rows-first strides and the key
            // count off K's axis 1.
            assert_eq!(bare.seq_k, LK as u32);
            assert_eq!(
                (bare.q_s, bare.q_h, bare.o_s),
                (NH as u32 * HD as u32, HD as u32, NH as u32 * HD as u32)
            );
            // This operand is `[b, nh, rows, hd]`: heads-first is the only reading
            // that fits it, so it is read that way at both row counts.
            let swapped =
                presented(1, rows, LK, NH, HD, Presented::Swapped).expect("a heads-first plan");
            assert_eq!((swapped.seq_q, swapped.seq_k), (rows as u32, LK as u32));
            assert_eq!(
                (swapped.q_s, swapped.q_h),
                (HD as u32, rows as u32 * HD as u32)
            );
        }
    }

    /// The walk reads the producing nodes' dims as well as the operand's, and Q's
    /// own axes are read by both arms — the backward one without the plan's
    /// staticness guard — so an unpinned dim is refused by name wherever it is met.
    #[test]
    fn a_dynamic_dim_on_the_way_into_a_rank4_operand_is_refused_by_name() {
        let f = DType::F32;
        // Q static at the coincident extent over a projection whose batch the
        // graph has not pinned: the plan's own guards pass, so the walk is what
        // meets the dynamic dim.
        let mut g = Graph::new("attention_dynamic_producer");
        let packed = g.input(
            "qp",
            Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(16), Dim::Static(1024)], f),
        );
        let q = g.add_node(
            Op::Reshape {
                new_shape: vec![1, 16, 16, 64],
            },
            vec![packed],
            Shape::new(&[1, 16, 16, 64], f),
        );
        let k = g.input("k", Shape::new(&[1, 40, 16, 64], f));
        let v = g.input("v", Shape::new(&[1, 40, 16, 64], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: 16,
                head_dim: 64,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 16, 16, 64], f),
        );
        assert_eq!(
            plan_attention(&g, g.node(node)).unwrap_err(),
            AttentionRefusal::NonStatic
        );

        // …and Q's own axes, which the backward arm reads without that guard.
        let mut g = Graph::new("attention_dynamic_q");
        let q = g.input(
            "q",
            Shape::from_dims(
                &[
                    Dim::Static(1),
                    Dim::Dynamic(0),
                    Dim::Static(16),
                    Dim::Static(64),
                ],
                f,
            ),
        );
        let k = g.input("k", Shape::new(&[1, 40, 16, 64], f));
        assert_eq!(
            rank4_geometry(&g, q, k, 16, 64).unwrap_err(),
            AttentionRefusal::NonStatic,
            "the arm reads the derivation directly, so the dim it cannot read is a refusal and \
             not a panic"
        );
    }

    /// The route's two arms read one derivation, so a step whose forward reads a
    /// graph rows-first cannot have a backward that reads its head axis. Both
    /// entry points are pinned here, device-free, on the graphs the plan above
    /// derives from: the rank-4 geometry the arm's rank-4 branch reads, and the
    /// rank-3 geometry its rank-3 branch reads — the spelling whose own batch and
    /// key length must survive it. What the arm does with them is the device pin's
    /// (`tests/attention_rows_route.rs`), which shows a step and its gradients
    /// computing one graph's numbers in both spellings.
    #[test]
    fn the_backward_arm_reads_the_same_layout_derivation() {
        let (g, attn) = presented_graph(1, 16, 40, 16, 64, Presented::Unrolled);
        let inputs = g.node(attn).inputs.clone();
        assert_eq!(
            rank4_geometry(&g, inputs[0], inputs[1], 16, 64),
            Ok((1, 16, 40, false)),
            "the arm's rank-4 branch reads the reshape: the key length is the key count off K's \
             axis 1, 40, and not K's head axis' 16"
        );
        let (bare, attn) = presented_graph(1, 16, 40, 16, 64, Presented::Bare);
        let inputs = bare.node(attn).inputs.clone();
        assert_eq!(
            rank4_geometry(&bare, inputs[0], inputs[1], 16, 64),
            Err(AttentionRefusal::AmbiguousRank4 {
                heads: 16,
                head_dim: 64
            }),
            "and the residue the graph leaves open is refused by the arm too"
        );

        // The arm's rank-3 branch, on the two rank-3 spellings: the packed
        // `[B, S, H·D]` projection, whose leading dim is the batch, and the
        // `[B·H, S, D]` one, whose leading dim is not.
        let f = DType::F32;
        let mut packed = Graph::new("attention_rank3_packed");
        let q = packed.input("q", Shape::new(&[1, 16, 16 * 64], f));
        let k = packed.input("k", Shape::new(&[1, 40, 16 * 64], f));
        assert_eq!(
            rank3_geometry(&packed, q, k, 16, 64),
            Ok((1, 16, 40)),
            "a batch read off the head axis is the batch divided away: 1, not 1/16"
        );
        let mut bhsd = Graph::new("attention_rank3_bhsd");
        let q = bhsd.input("q", Shape::new(&[16, 16, 64], f));
        let k = bhsd.input("k", Shape::new(&[16, 40, 64], f));
        assert_eq!(
            rank3_geometry(&bhsd, q, k, 16, 64),
            Ok((1, 16, 40)),
            "the `[B·H, S, D]` spelling is the leading dim over the head count"
        );
    }

    /// The walk's unroll arm reads a source's last-but-one extent, so it treats
    /// rank-1 to rank-3 sources as the packed spellings: a rankless one has no
    /// axis to unroll, and is refused by name rather than indexed out of its
    /// shape.
    #[test]
    fn a_rankless_packed_source_is_refused_rather_than_indexed() {
        let f = DType::F32;
        let mut g = Graph::new("attention_rankless_source");
        let scalar = g.input("q", Shape::new(&[], f));
        let q = g.add_node(
            Op::Reshape {
                new_shape: vec![1, 16, 16, 64],
            },
            vec![scalar],
            Shape::new(&[1, 16, 16, 64], f),
        );
        let k = g.input("k", Shape::new(&[1, 40, 16, 64], f));
        let v = g.input("v", Shape::new(&[1, 40, 16, 64], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: 16,
                head_dim: 64,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 16, 16, 64], f),
        );
        assert_eq!(
            plan_attention(&g, g.node(node)).unwrap_err(),
            AttentionRefusal::AmbiguousRank4 {
                heads: 16,
                head_dim: 64
            }
        );
    }

    #[test]
    fn the_key_split_needs_a_small_grid_and_a_long_key_range() {
        // The talker body frame: 16 rows, 330 keys, width 128 — the grid leaves
        // most of the machine idle, so it splits into two key ranges.
        assert_eq!(attention_decode_key_splits(1, 330, 16, 128, 128), 2);
        // The code predictor's attention is one row over a few cached keys:
        // nothing to split.
        assert_eq!(attention_decode_key_splits(1, 17, 16, 128, 128), 1);
        // A grid that already fills the machine stays on one workgroup a head.
        assert_eq!(attention_decode_key_splits(1, 4096, 64, 128, 128), 1);
        // The serial form's shapes never split.
        assert_eq!(attention_decode_key_splits(2, 330, 16, 128, 128), 1);
        assert_eq!(attention_decode_key_splits(1, 330, 16, 192, 128), 1);
        // The partials' layout: two scalars plus a chunk-padded value row.
        assert_eq!(attention_parts_bytes(16, 2, 128), 16 * 2 * 130 * 4);
        assert_eq!(attention_parts_bytes(16, 1, 40), 16 * (2 + 64) * 4);
    }

    /// One `Op::Attention` with the two-sided band `w` over rank-4
    /// `[B, S, H, D]` / `[B, S, KV, D]` operands — the encoder's own layout — and
    /// the plan it derives. `kv` is K's and V's head count, so a GQA group is a
    /// step's own.
    fn plan_band(
        rows: usize,
        lk: usize,
        heads: usize,
        kv: usize,
        hd: usize,
        v_hd: Option<usize>,
        w: usize,
    ) -> Result<AttentionPlan, AttentionRefusal> {
        let f = DType::F32;
        let mut g = Graph::new("attention_band_plan");
        let q = g.input("q", Shape::new(&[1, rows, heads, hd], f));
        let k = g.input("k", Shape::new(&[1, lk, kv, hd], f));
        let v = g.input("v", Shape::new(&[1, lk, kv, v_hd.unwrap_or(hd)], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: heads,
                head_dim: hd,
                v_head_dim: v_hd,
                mask_kind: MaskKind::Window(w),
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, rows, heads, v_hd.unwrap_or(hd)], f),
        );
        plan_attention(&g, g.node(node))
    }

    /// The two-sided band is the band kernel's and nothing else's: at the model's
    /// own geometries the plan takes it, and the dense arrangement that would
    /// otherwise have served the shape (≥ 2 query rows, scratch inside the cap) is
    /// switched off — the arrangement reads kind 5 as "no mask" and answers a
    /// dense pass, which is a wrong answer rather than a slow one.
    #[test]
    fn a_windowed_step_is_the_band_kernels_at_the_models_geometries() {
        for (rows, heads, kv, hd) in [
            // The sliding layers: 8 192 rows, ±512, 4 heads over 2 KV heads.
            (8192usize, 4usize, 2usize, 256usize),
            // The global layers: no band, one KV head, the widest head.
            (8192, 4, 1, 512),
            // The small cell, and a narrow one whose band really clips.
            (256, 4, 2, 256),
            (300, 2, 2, 64),
        ] {
            let plan =
                plan_band(rows, rows, heads, kv, hd, None, 512).unwrap_or_else(|r| panic!("{r}"));
            assert!(plan.band, "rows={rows} hd={hd} must take the band kernel");
            assert!(
                !plan.decompose,
                "rows={rows} hd={hd}: the dense arrangement must not also claim it"
            );
            assert_eq!(plan.mask_kind_id, 5, "the kernel reads kind 5");
            assert_eq!(plan.window, 512, "and the band width with it");
        }
        // `w >= seq` is admitted: it is the global layers' shape, and the kernel
        // reads it as no band at all.
        for w in [8192usize, u32::MAX as usize] {
            let plan = plan_band(8192, 8192, 4, 1, 512, None, w).unwrap_or_else(|r| panic!("{r}"));
            assert!(plan.band, "w={w} >= seq is the no-band cell");
        }
    }

    /// One dense `Op::Attention` (`MaskKind::None`) over Q/K/V of the given
    /// shapes, and the plan its geometry derives — the spelling
    /// `plan_band` cannot build.
    fn plan_dense(
        rows: usize,
        heads: usize,
        kv: usize,
        hd: usize,
    ) -> Result<AttentionPlan, AttentionRefusal> {
        let f = DType::F32;
        let mut g = Graph::new("attention_dense_plan");
        let q = g.input("q", Shape::new(&[1, rows, heads, hd], f));
        let k = g.input("k", Shape::new(&[1, rows, kv, hd], f));
        let v = g.input("v", Shape::new(&[1, rows, kv, hd], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: heads,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, rows, heads, hd], f),
        );
        plan_attention(&g, g.node(node))
    }

    /// The stand's four global layers — 8 192 rows, 4 heads of width 512 over one
    /// KV head, `MaskKind::None` — take the cooperative dense square form, not the
    /// band kernel with the band switched off nor the f32 many-row arrangement:
    /// the score square is 1 GiB here, past both the many-row cap and the band
    /// kernel's instruction mix, and the two two-pass multiplies are what the
    /// stand's embedding cell is measured against
    /// (`tests/attention_dense_coop_cost.rs`). The chunk the plan books is the
    /// region the lowering walks, and it is the one the scratch cap admits.
    #[test]
    fn a_dense_square_step_at_the_encoder_width_takes_the_cooperative_form() {
        let plan = plan_dense(8192, 4, 1, 512).unwrap_or_else(|r| panic!("{r}"));
        assert!(
            plan.dense_coop,
            "the global layers' shape must take the cooperative dense square form"
        );
        assert!(
            plan.band,
            "the band kernel with the band switched off stays the fallback: it is what \
             the step keeps where the adapter cannot build or verify the cooperative \
             form, and it is admitted here for exactly that reason"
        );
        assert!(
            !plan.decompose,
            "and not the f32 many-row arrangement it replaces"
        );
        assert_eq!(plan.mask_kind_id, 0, "the op is still MaskKind::None");
        assert_eq!(
            plan.coop_chunk,
            DENSE_COOP_CHUNK_ROWS.min(plan.seq_q),
            "the score matrix is walked one chunk at a time"
        );
        assert!(
            plan.scratch_bytes() as u64 <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES,
            "an admitted plan fits the reserve it asks for"
        );
        // A square *window* step at the same shape is the band kernel's: kind 5
        // is the one mask the cooperative form does not read, and the sliding
        // layers are that step.
        let mut g = Graph::new("attention_dense_plan_kind5");
        let f = DType::F32;
        let q = g.input("q", Shape::new(&[1, 8192, 4, 512], f));
        let k = g.input("k", Shape::new(&[1, 8192, 1, 512], f));
        let v = g.input("v", Shape::new(&[1, 8192, 1, 512], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: 4,
                head_dim: 512,
                v_head_dim: None,
                mask_kind: MaskKind::Window(512),
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 8192, 4, 512], f),
        );
        let plan = plan_attention(&g, g.node(node)).unwrap_or_else(|r| panic!("{r}"));
        assert!(
            plan.band && !plan.dense_coop,
            "kind 5 keeps the band kernel"
        );
    }

    /// The cooperative form's scratch bookkeeping: the output region lives over
    /// Kᵀ's bytes, so the reserve must not charge for it and the bump must still
    /// end inside what the reserve holds. Both are the plan's own arithmetic, so
    /// they are checked here rather than left to a run.
    #[test]
    fn the_cooperative_form_reuses_kt_for_its_output_region() {
        let plan = plan_dense(8192, 4, 1, 512).unwrap_or_else(|r| panic!("{r}"));
        assert!(plan.dense_coop && plan.coop_out_over_kt);
        assert!(
            plan.regions.out <= plan.regions.kt,
            "the output grid only fits over Kᵀ because it is no larger"
        );
        // The booked extent is one chunk of query rows — what the lowering writes
        // per chunk before the next overwrites it — and not the whole sequence: at
        // the encoder's shape a full-sequence booking is four times this.
        assert_eq!(
            plan.regions.out,
            4 * DENSE_COOP_CHUNK_ROWS as u64 * 512,
            "the output region is booked a chunk of rows, not the whole sequence"
        );
        let offsets = plan.scratch_offsets(true);
        assert_eq!(offsets.out, offsets.kt, "the output region starts at Kᵀ's");
        assert!(
            offsets.end as usize <= plan.scratch_bytes(),
            "the bump ends inside the reserve: {} > {}",
            offsets.end,
            plan.scratch_bytes()
        );
        assert!(
            plan.scratch_bytes() as u64 <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES,
            "and the reserve is what the cap admitted"
        );
        // A step the cooperative form does not take books no overlap: the flag is
        // the cooperative layout's alone.
        let below = plan_dense(1024, 4, 1, 512).unwrap_or_else(|r| panic!("{r}"));
        assert!(!below.dense_coop && !below.coop_out_over_kt);
    }

    /// The gates the cooperative form adds over the band kernel's geometry, all
    /// the graph's: the query-row floor below which the answer is the f32
    /// arrangement's, the operand widths the kernels take, the single batch the
    /// per-chunk output permute can write, and the KV group the transposes can
    /// replicate. `dense_coop_admits` is the one derivation of them.
    #[test]
    fn the_cooperative_dense_square_form_admits_only_whole_wide_tiles() {
        // The stand's cell, at the widest head the fused path serves.
        assert_eq!(
            dense_coop_admits(0, 8192, 8192, 512, 512, 1, 4, 1),
            Some(2048)
        );
        // Below the floor: `attention_long_seq_precision`'s BrainBERT sweep and
        // the band parity's 1 536-row cell are these, and both must keep the
        // arrangement they have.
        assert_eq!(dense_coop_admits(0, 1024, 1024, 512, 512, 1, 4, 1), None);
        assert_eq!(dense_coop_admits(0, 1536, 1536, 64, 64, 1, 64, 64), None);
        // k % 32 and n % 64, per multiply; the sequence is both operands' inner
        // and outer extent, so it has to be a whole tile itself.
        assert_eq!(dense_coop_admits(0, 8192, 8192, 500, 512, 1, 4, 1), None);
        assert_eq!(dense_coop_admits(0, 8192, 8192, 512, 500, 1, 4, 1), None);
        assert_eq!(dense_coop_admits(0, 8190, 8190, 512, 512, 1, 4, 1), None);
        // Rectangular and masked steps are other routes'.
        assert_eq!(dense_coop_admits(0, 8192, 4096, 512, 512, 1, 4, 1), None);
        assert_eq!(
            dense_coop_admits(BAND_MASK_KIND, 8192, 8192, 512, 512, 1, 4, 1),
            None
        );
        // A head width past the fused kernel's ceiling is nobody's route.
        assert_eq!(dense_coop_admits(0, 8192, 8192, 1024, 1024, 1, 4, 1), None);
        // More than one batch: the per-chunk output permute writes each batch's
        // block where the first chunk already wrote, so the two would overwrite
        // each other. Refused rather than served wrongly.
        assert_eq!(dense_coop_admits(0, 8192, 8192, 512, 512, 2, 4, 1), None);
        // A head count that is not a whole multiple of the KV heads: the Kᵀ and
        // V′ transposes write one tile per KV head while both multiplies read
        // `batch · heads`, so the fourth of four heads over three KV heads would
        // read zero-initialised f16.
        assert_eq!(dense_coop_admits(0, 8192, 8192, 512, 512, 1, 4, 3), None);
        assert_eq!(dense_coop_admits(0, 8192, 8192, 512, 512, 1, 4, 0), None);
        // And the chunk is the plan's, whatever the sequence leaves.
        assert_eq!(
            dense_coop_admits(0, 2048, 2048, 512, 512, 1, 4, 1),
            Some(2048)
        );
        assert_eq!(
            dense_coop_admits(0, 4096, 4096, 512, 512, 1, 4, 1),
            Some(2048)
        );
    }

    /// Below the many-row arrangement's scratch cap a dense square step keeps it,
    /// even though the band kernel's geometry admits the shape: the f32
    /// arrangement is affordable there and its answer is the tighter one (1e-5
    /// against the f16 tile's 5e-3 of scale). The BrainBERT sweep in
    /// `tests/attention_long_seq_precision.rs` is what holds this: its 16–186-row
    /// cases would move 1e-6 → 1e-3 if the floor were dropped.
    #[test]
    fn a_dense_square_step_below_the_scratch_cap_keeps_the_dense_route() {
        for (rows, heads, kv, hd) in [
            (16usize, 12usize, 12usize, 64usize),
            (186, 12, 12, 64),
            (1024, 4, 4, 256),
        ] {
            let plan = plan_dense(rows, heads, kv, hd).unwrap_or_else(|r| panic!("{r}"));
            assert!(
                !plan.band,
                "rows={rows} hd={hd}: the band kernel must not take a dense step whose \
                 many-row scratch fits"
            );
            assert!(
                plan.decompose,
                "rows={rows} hd={hd}: and the dense arrangement must still serve it"
            );
            assert!(!plan.band_declined, "no refusal to report");
        }
    }

    /// Where the band admission refuses a dense square step the step keeps the
    /// arrangement it had — the dense one below the scratch cap, the fused one
    /// above it — rather than becoming unplannable.
    #[test]
    fn a_dense_square_step_the_band_kernel_refuses_keeps_its_route() {
        // 96-wide heads are a multiple of the chunk but their three chunks do not
        // divide the workgroup, so the band kernel has no form for them; the
        // cooperative form takes `v_head_dim` in whole 64-column tiles, which 96 is
        // not; and at 8 192 rows the many-row arrangement's scratch is past its
        // cap, which is the shape that would otherwise be the band kernel's.
        let plan = plan_dense(8192, 4, 4, 96).unwrap_or_else(|r| panic!("{r}"));
        assert!(!plan.band, "96-wide heads have no band form");
        assert!(!plan.dense_coop, "nor have they a whole tile width");
        assert!(
            !plan.decompose,
            "and at this length the dense arrangement is past its scratch cap"
        );
        assert!(
            plan.band_declined,
            "the refusal is reported rather than silent"
        );
        // A width the cooperative form books, refused by the band kernel for the
        // same chunk count: the refusal is not a decline of the step's route, so
        // the flag stays clear. The cooperative form is the shape's route; whether
        // the adapter takes it is the lowering's separate count.
        let cooperative = plan_dense(8192, 4, 4, 192).unwrap_or_else(|r| panic!("{r}"));
        assert!(
            cooperative.dense_coop,
            "192 is a whole tile width for both cooperative kernels"
        );
        assert!(
            !cooperative.band,
            "and six chunks do not divide either band workgroup width"
        );
        assert!(
            !cooperative.band_declined,
            "a band refusal on a step the cooperative form booked is not a decline"
        );
        // A non-square dense step never had a band to walk.
        let mut g = Graph::new("attention_dense_rect");
        let f = DType::F32;
        let q = g.input("q", Shape::new(&[1, 64, 2, 128], f));
        let k = g.input("k", Shape::new(&[1, 256, 2, 128], f));
        let v = g.input("v", Shape::new(&[1, 256, 2, 128], f));
        let node = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 128,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 64, 2, 128], f),
        );
        let rect = plan_attention(&g, g.node(node)).unwrap_or_else(|r| panic!("{r}"));
        assert!(
            !rect.band && rect.decompose,
            "a rectangular dense step keeps the dense route"
        );
    }

    /// A geometry the band kernel cannot hold is refused at plan time — naming
    /// the shapes — rather than lowered to the fused or dense arrangement, both of
    /// which would compute a dense pass. Each case is one conjunct of
    /// [`band_attention`], with the term it breaks.
    #[test]
    fn a_band_step_the_kernel_cannot_hold_is_refused_by_name() {
        // Each row is one conjunct of [`band_attention`] and the term it breaks.
        // (A width above the ceiling, 576, never reaches these conjuncts: the
        // plan's own `HeadWidth` refusal names it first, which
        // `a_head_width_past_the_ceiling_is_refused_rather_than_left_blank`
        // pins.)
        let cases: Vec<((usize, usize, usize, usize, Option<usize>, usize), &str)> = vec![
            ((1024, 512, 4, 256, None, 512), "seq_q must equal seq_k"),
            (
                (512, 512, 4, 256, Some(64), 512),
                "v_head_dim must equal head_dim",
            ),
            (
                (512, 512, 4, 80, None, 512),
                "multiple of the kernel's 32-wide",
            ),
            (
                // 192 is a multiple of the kernel's 32-wide chunk (6 of them),
                // but six chunks do not divide either workgroup width — 256
                // threads at the floor, 512 where the adapter admits the wide
                // form.
                (512, 512, 4, 192, None, 512),
                "must split the kernel's workgroup into whole rows",
            ),
            // The one-chunk width: a row split across a single thread has no
            // partials to reduce, and the pair form would read the next row's.
            ((512, 512, 4, 32, None, 512), "64-wide floor"),
        ];
        for ((rows, lk, heads, hd, v_hd, kv), want) in cases {
            let err = plan_band(rows, lk, heads, kv, hd, v_hd, 512)
                .err()
                .unwrap_or_else(|| panic!("rows={rows} lk={lk} hd={hd} must be refused"));
            let msg = err.to_string();
            assert!(
                msg.contains("MaskKind::Window"),
                "the refusal names the mask kind: {msg}"
            );
            assert!(msg.contains(want), "rows={rows} lk={lk} hd={hd}: {msg}");
            // And the shapes, so a caller can act on it rather than read a crash.
            assert!(msg.contains(&hd.to_string()), "the head width: {msg}");
            assert!(msg.contains(&rows.to_string()), "the row count: {msg}");
            // The message names the numbers the code divides by, not just the
            // sentence around them: 192 is refused over its chunk count, whose
            // denominator is the 32-wide chunk and whose workgroup width is 256
            // threads at the kernel's floor and 512 where the adapter admits the
            // wide form — so both figures, and the widths that are whole rows at
            // either.
            if hd == 192 {
                assert!(
                    msg.contains("head_dim / 32 chunks"),
                    "the chunk count: {msg}"
                );
                assert!(msg.contains("256 threads"), "the floor width: {msg}");
                assert!(
                    msg.contains("512 where the adapter admits the wide form"),
                    "the wide width, named as such rather than by the shapes: {msg}"
                );
                assert!(
                    msg.contains("only 64, 128, 256 and 512 do"),
                    "the widths that split into whole rows: {msg}"
                );
                // The sentence states the divisor and the admitted set; it does
                // not enumerate the refused widths, which would have to move in
                // step with every form and once mis-attributed 32 (refused by the
                // floor, not by the chunk count).
                assert!(
                    !msg.contains("are refused") && !msg.contains("480"),
                    "no enumeration of the refused widths: {msg}"
                );
            }
        }
        // The one-chunk refusal is also reachable without a plan, on the width
        // itself, and it names the floor rather than falling through the form.
        let err = band_attention(512, 512, 32, 32, 4, 4, 4)
            .expect_err("a one-chunk head has no reduce to walk");
        assert!(
            err.to_string().contains("64-wide floor"),
            "the one-chunk width: {err}"
        );
        // A KV group that does not divide the head count is refused too — the
        // shader's `heads / kv_heads` would index past its KV heads.
        let err = plan_band(512, 512, 6, 4, 256, None, 512)
            .expect_err("6 heads over 4 KV heads is not a GQA group");
        assert!(
            err.to_string().contains("KV-head count"),
            "the group: {err}"
        );
        // And a step with no keys has nothing to walk.
        let empty = plan_band(0, 0, 4, 1, 256, None, 512).expect_err("no keys to walk");
        assert!(
            empty.to_string().contains("no key to attend to"),
            "the empty sequence: {empty}"
        );
    }

    /// The band body's `// @unroll` expansion is that kernel's largest lever, so
    /// it gets a device-free witness: nesting, literal substitution, and the body
    /// coming out with no marker left and a statement a key a slot. The staged
    /// tile's own marker is the other pass (`bind_band_stage`), and both spellings
    /// of it have to leave a module the body can be read from.
    #[test]
    fn the_band_body_expands_to_literals() {
        use crate::kernels::{ATTENTION_BAND_WGSL, bind_band_stage, expand_band_loops};
        let nested = concat!(
            "// @unroll kk\n",
            "{\n",
            "  // @unroll j\n",
            "  a[%j%u] = b[%kk%u * 2u + %j%u];\n",
            "  // @end\n",
            "}\n",
            "// @end\n",
        );
        let out = expand_band_loops(nested, 2, 3);
        assert!(!out.contains('%'), "a marker survived: {out}");
        assert_eq!(out.matches("a[").count(), 6, "2 keys × 3 slots: {out}");
        assert!(
            out.contains("a[2u] = b[1u * 2u + 2u];"),
            "substitution: {out}"
        );
        for stage in ["vec4<f16>", "vec4<f32>"] {
            let bound = bind_band_stage(ATTENTION_BAND_WGSL, stage);
            assert!(
                bound.contains(&format!("array<{stage}, BK * C * SLOTS>")),
                "the tile is declared at the staging width"
            );
            assert!(
                bound.contains(&format!("stage[i] = {stage}(v4);")),
                "and stored at it"
            );
            assert!(!bound.contains("%STAGE%"), "a tile marker survived");
            let out = expand_band_loops(&bound, 8, 8);
            // The body's own `%` operators (`lid % C`) stay; its markers may not.
            assert!(
                !out.contains("%j%") && !out.contains("%kk%"),
                "a marker survived"
            );
            assert!(out.contains("lid % C"));
            // The last key's last slot, addressed by literals — the whole point:
            // a `stage[...]` read the backend can see is constant-indexed.
            assert!(out.contains("stage[7u * hd4 + 7u * C + c]"));
            assert!(!out.contains("kk * hd4"));
        }
    }

    /// The band kernel's grid and tile constants are one set of numbers in two
    /// languages: the form the host derives (`threads`, `dims_per_thread`) and the
    /// `BM`/`C`/`DPER`/`THREADS` the module is declared with. A `DPER` in the body
    /// would be a second source for the same number, so the body must not carry
    /// one.
    #[test]
    fn the_band_tile_constants_are_the_shaders_own() {
        use crate::kernels::{
            ATTENTION_BAND_MIN_THREADS, ATTENTION_BAND_WGSL, attention_band_form,
            attention_band_key_block_within, attention_band_rows,
        };
        assert!(
            !ATTENTION_BAND_WGSL.contains("const DPER"),
            "`DPER` is the host's declaration, not the body's: two sources would let the \
             dispatch's grid and the module's workgroup disagree"
        );
        assert!(
            ATTENTION_BAND_WGSL.contains("const SLOTS: u32 = DPER / 4u;"),
            "the body derives its slot count from the host's `DPER`"
        );
        // The widths the plan admits each land on one form whose chunk count is at
        // least two and divides the workgroup into whole rows.
        for hd in [64u32, 128, 256, 512] {
            let form = attention_band_form(hd).unwrap_or_else(|| panic!("hd={hd} has no form"));
            let chunks = form.chunks;
            assert!(chunks >= 2, "hd={hd} would build a one-chunk row");
            assert_eq!(attention_band_rows(hd), form.rows);
            assert_eq!(attention_band_rows(hd) * chunks, ATTENTION_BAND_MIN_THREADS);
            // The staged tile plus the partials and the block's scores fit the
            // budget the block was chosen for — WebGPU's guaranteed 16 KiB floor
            // and whatever an adapter reports. The device's own figure is asserted
            // against both of the sizes this rig has seen: the guaranteed floor,
            // where the block degrades, and the 32 KiB an M-series GPU reports.
            for budget in [16u32 * 1024, 32 * 1024] {
                for (stage_bytes, tag) in [(2u32, "f16"), (4, "f32")] {
                    let bk = attention_band_key_block_within(
                        hd,
                        stage_bytes,
                        budget,
                        ATTENTION_BAND_MIN_THREADS,
                    );
                    let stage = (bk * hd * stage_bytes) as usize;
                    let parts = (attention_band_rows(hd) * chunks * bk * 4) as usize;
                    let scores = (attention_band_rows(hd) * bk * 4) as usize;
                    assert!(
                        stage + parts + scores <= budget as usize,
                        "hd={hd} bk={bk} {tag} at {budget} B needs {} B",
                        stage + parts + scores
                    );
                }
            }
        }
        // Widths with no whole-row form, including the one-chunk width the
        // kernel refuses by name.
        for hd in [32u32, 48, 96, 192, 320, 448] {
            assert!(
                attention_band_form(hd).is_none(),
                "hd={hd} must have no form"
            );
        }
    }
}
