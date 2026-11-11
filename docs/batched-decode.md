# Batched multi-row decode: the reading, and why nothing shipped

A long recognition call cuts its recording into independent pieces and decodes them one
after another — about fifty pieces, about 1 300 decode steps on the pinned recording, each
streaming the body's ~1.1 GB of packed weights
([`decode_token_rate.rs`](../crates/mahbot-inference/tests/decode_token_rate.rs) pins the
geometry and that figure). The pieces are independent, so several could decode in one pass
sharing one weight read. The change judged that worth shipping only if a pass of `n` lanes
costs at most half a one-row pass per token — at four lanes, a pass at most 2× a one-row
pass. It missed on two counts. At four lanes — the largest group the attempt's per-lane
plane budget allowed — a pass costs 2.95–3.01× a one-row pass (0.74–0.75× per token), over
that bar, and the row's total peak rises by several times the allowance the change asked
for. Nothing shipped.

## The reading

One process on an M3 Pro 18 GiB, release, at capacity 181 — the top rung's 111-row prompt
plus a 7 s piece's 70-token budget. The product's rung path compiles its decode graph at 192
instead: the same 111-row prompt plus the 76 tokens a 7.5 s piece (the cut's 7 s plus its
0.5 s search window) may generate, 187 rounded up to the next multiple of eight
([`listen.rs`](../crates/mahbot-inference/src/listen.rs)'s `RUNG_CAPACITY`). Every lane
ratio below is at planes ~6% smaller than the product's, and the attempt's memory column is
at that same capacity, so a reader re-deriving either from 192 will not match; the margin to
the bound at the four-lane operating point is wide enough that a flip is implausible. Each
arm is one graph, warmed, then the median of nine passes; the arms ran in one process, one
after another (1, 4, 6, 8), not round-robin. The 6- and 8-lane arms are past the attempt's
per-lane plane budget; they show only how the pass scales. The one-row arm is the shipped
decode body, so the denominator is not a second route. The first-touch ordering bias
[`gguf_gemm_grouped_weight_reuse.rs`](../crates/backends/rlx-wgpu/tests/gguf_gemm_grouped_weight_reuse.rs)
records — a byte-identical control drifting, later arms reading fast — is therefore not
excluded, and it would flatter the grouped arms, which run after the one-row denominator;
the verdict does not rest on it.

| lanes | one pass | per token |
|---|---|---|
| 1 | 12.5–13.1 ms | 1.00× |
| 4 | 37.7–38.7 ms = 2.95–3.01× | 0.74–0.75× |
| 6 | 40.2–43.4 ms = 3.2–3.3× | 0.54–0.55× |
| 8 | 45.0–46.4 ms = 3.5–3.6× | 0.44–0.45× |

The attempt's own figures — the lane table, the product call's piece and rung counts, times
and transcript comparison, the cache-length arm, the paired take and the attempt's memory
columns — are unrecorded, and their exact figures cannot be reproduced from the tree; the
route-specific ones were measured once by a harness that was never committed and was deleted
with the route. The figures with a home are named where they appear: the decode geometry and
the 1 100 961 792 bytes per token in `decode_token_rate.rs`, the grouped multiply's 1.72–1.82×
band in `gguf_gemm_grouped_weight_reuse.rs`, `GROUPED_ROWS` and `ceil(m / 8)` in
`gguf_gpu.rs`, the product's 192-row capacity in `listen.rs`, the record's take in
`STAND.md`, the 68 regions in the stand's console detail line and the 1 287-token decode
count in its `MAHBOT_PHASE_SPLIT` `phases:` line — takes, not stored records.

## What the product call did

On the pinned long recording, 42 of its 48 pieces decoded in groups (the 19- and 23-piece
rungs; the six-piece rung stayed solo), no fallback, and the same transcript — the same
region digest. The stand's long-row detail line prints 68 regions for the same recording:
regions are the plan's tiling of the audio and pieces are the chunks the recogniser runs on,
so the two counts differ by construction. Decode 13.6 s over 1 287 tokens, read from
`MAHBOT_PHASE_SPLIT`'s `phases:` line (the `decode … ms over … tokens` field), and warm
25.1 s. The session's two takes of the unchanged binary read decode 16.5 and 16.9 s and warm
27.8 and 28.7 s; it did not record which decode take paired with which warm take, so the
gains are ranges — 2.9–3.3 s on the decode (1.21–1.24×) and 2.7–3.6 s warm, 3.15 s against
the takes' median of 28.25 s. The record's long-row take is 27.89 s warm.

## The two misses

**The multiply, first, because it is the one with headroom shown.** The pass barely grows
with rows — 2.95–3.01× the one-row pass at four lanes and 3.5–3.6× at eight, where a cost
proportional to rows would be 4× and 8× — so at the four-lane operating point one weight
read across the rows does not buy the saving the change asked for. The fixed term is the
row-grouped packed multiply's eight-row tile:
`GROUPED_ROWS` is the row tile one workgroup covers and `grouped_base_geometry` dispatches
`ceil(m / 8)` of them ([`gguf_gpu.rs`](../crates/backends/rlx-wgpu/src/gguf_gpu.rs)), so
four rows are one row group's work and the tile computes eight rows whatever `m` is. The
tree's own reading of that kernel
([`gguf_gemm_grouped_weight_reuse.rs`](../crates/backends/rlx-wgpu/tests/gguf_gemm_grouped_weight_reuse.rs))
is flat from two to eight rows at about 1.8× the one-row cell — at a stress shape
(`n = 202048`, `k = 7168`) far bigger than any of this body's, so it bounds the mechanism
rather than this body's ratio.

**Then memory: each peak reading alone is over the allowance.**

| | the record's take | the paired take | this attempt |
|---|---|---|---|
| peak GPU | 1.99 GiB | 2.12 GiB | 2.27 GiB |
| peak system | 457 MiB | 460 MiB | 828 MiB |
| total peak | 2.44 GiB | 2.57 GiB | 3.08 GiB |

The record's take is the long-row take in `STAND.md` — 456.6 MiB of peak system and
1.99 GiB of peak GPU. The paired take is the unchanged binary measured in the attempt's own
session — a measurement, not a record — and the third column is the attempt. The change
asked for 5% of total peak — 125 MiB against the record's take and 132 MiB against the
paired take. Against the record's take the attempt is +655 MiB (+26%), 5.2× that 125 MiB;
against the paired take +522 MiB (+20%), 4.0× that 132 MiB. The GPU peak alone rises
287 MiB over the record's take against that take's 125 MiB, and 154 MiB over the paired
take against its 132 MiB, so explaining the host side away would not have shipped it. The
derived host-side reading (peak system) rose 371 MiB over the record's take and 368 over
the paired take, and is **unexplained**; the direct resting-system reading is the attempt's
own and unrecorded, and the attempt's resting GPU was not recorded.

## What a re-attempt must answer

Two things nobody has explained. First, the derived host-side reading — 368–371 MiB that is
not the rung's front and prefill graphs (the one-row route already holds those for the rung's
life) and is distinct from the direct resting-system reading, which is the attempt's own and
unrecorded. Second, the residual between the multiply and the pass: the row-grouped tile
explains why the multiply stays flat in rows, not what the pass spends beyond it. The
cache-length arm does not explain the gap: 16 and 111 visible rows differ by under 5 ms, so
the feeds or attention at fixed length remain candidates.

The route is gone from the tree: the product code path is identical to the state before this
work, and the one multi-row decode graph that remains,
[`graphs::body_capacity_chain_planes_rows`](../crates/mahbot-inference/src/graphs.rs), is
the shared-cache arm
[`tests/draft_acceptance.rs`](../crates/mahbot-inference/tests/draft_acceptance.rs) uses to
price a pass rather than answer one — not a product route.

## License

MIT OR Apache-2.0.
