# Measurements

The latest numbers from this crate's two measuring surfaces. Each row is
replaced by the run that measured it: the section carries that run's date and
`code` line, and its `rows` line names which rows the run took — a row a later
run did not measure is carried, not re-taken, because its own inputs are not what
that run opens. The two surfaces are run at different times, so one half may be
older than the other. This file is not a log — history is what version control is
for.

<!-- BEGIN stand -->
## Per-model measurement (stand)

_Measured 2026-10-25 10:46 UTC._

| unit | cold ms | warm ms | speech s | s/speech s | loop ms | sound ms | build ms | peak | map | arena | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | 75.8 | 32.3 | - | - | - | - | 220.5 | 0.88 GiB | 1.53 GiB | 3.4MiB | 55a4972b484d3999 |
| ear.075 | 73.0 | 32.1 | - | - | - | - | 213.1 | 0.84 GiB | 1.53 GiB | 3.4MiB | a8ed106ea43eeb60 |
| ear.batch2 | 81.7 | 38.1 | - | - | - | - | 238.0 | 0.83 GiB | 1.53 GiB | 3.6MiB | cc5fcc79e6278b90 |
| ear.batch4 | 90.1 | 45.4 | - | - | - | - | 273.9 | 0.85 GiB | 1.53 GiB | 3.9MiB | 351492f41141665e |
| listen | 2127.9 | 1745.1 | 17.460 | 0.100 | - | - | 1274.9 | 2.31 GiB | 1.53 GiB | 1.10GiB | 2f93dfa1d7cdeeb1 |
| speak | 15874.6 | 14165.7 | 26.400 | 0.537 | 12986.1 | 1179.6 | 1766.7 | 2.42 GiB | 1.38 GiB | 702.9MiB | 9d837a9c4164ba94 |
| embed.256 | 99.3 | 66.7 | - | - | - | - | 727.1 | 1.63 GiB | 0.29 GiB | 547.9MiB | ec6e9fa56756e071 |
| embed.8192 | 2388.6 | 2322.6 | - | - | - | - | 1147.8 | 1.77 GiB | 0.29 GiB | 1.31GiB | bfa0336b63d2e3d0 |

The cell's number is the **`s/speech s`** column: the **warm** figure (`warm ms`) divided by the `speech s` column — the cell's own seconds of speech, `samples / 16 000` for recognition and `rows · 1 920 / 24 000` for synthesis, printed so the rate re-derives from this table alone. `loop ms` is the synthesis generation loop (speaker, prefill, body, predictor, sampler) and `sound ms` is the sound half; their sum is `warm ms`, so a ×N synthesis claim reads off both. Both are the same phase sums the cell's `loop_ms`/`sound_ms` tokens record; `speak`'s `sound_phase_ms` field is the sound graphs' summed `step_ms` — the same sum the `sound ms` column is taken from, carried as a field so a cell line is self-describing. For `speak` the recorded `digest` is the **waveform** digest, not the `prompt_digest` its own line prints. `-` is a cell with no speech denominator. Each cell line prints its own `warm_spread` (the warm runs' max − min, in ms). `peak` is the cell process's own high-water (`ru_maxrss`, which no later drop can lower) and each cell is its own process, so the figure is the maximum that unit's own process reached — what it held when it peaked, not what is live when the line prints. For `speak` the reading is taken before the round-trip gate loads the recogniser, so the column excludes it (the gate prints its own post-gate reading as `speak roundtrip peak_mb`). Across this machine's records the column moves between runs of one tree, and the summary line carries the spread each run saw as `peak_spread`. `embed.256` and `embed.8192` are one bidirectional pass each over the whole sequence on a Q8_0 stack. Their `digest` is over the L2-normalised 768-vector the row names — the cell's answer, not the graph's raw output.

The pinned answers: recognition must reproduce `обычные объявления метро делались только на каталонском языке, но не забывали переводить объявления в другие языки, например, итальянский, испанский, английский, французский, итальянский`, compared word for word with at most 2 of them different (its digest and generated-token count are witnesses, printed with what moved); the synthesis round trip — the waveform the `speak` row's digest names, resampled to 16 kHz and read back through the recogniser on its raw samples — must reproduce `Call me Ishmael. Some years ago, now I'm old, beside the sea. I've seen them all, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every ocean, every island, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every` exactly.

### Weight streams, against the machine's streaming ceiling

Ceiling, measured on this run's device before the cells: **135.3 GB/s** on a streaming read (133.9–135.9 over 5 rounds) and 133.9 GB/s of total traffic on a streaming copy (133.0–134.1). The read figure is the one the rows below are reported against, because a weight is read and never written; the copy figure is the same probe's other arm, recorded beside it. Both come from `rlx_wgpu::bandwidth::stream_ceiling`: a 64.0 MiB vectorised read and a 64.0 MiB vectorised copy, on `Apple M3 Pro (Metal)`. The same probe after the cells read 128.5 GB/s (127.9–129.3), which is the reading-to-reading spread to hold the figure to.

| phase | unit | weight bytes/unit | ms/unit | GB/s | of the read ceiling |
|---|---|---|---|---|---|
| recognition decode | one token | 1 100 961 792 | 15.84 | 69.5 | 51.4 % |
| talker body | one frame | 850 870 272 | 13.49 | 63.1 | 46.6 % |
| code predictor | one level | 87 109 632 | 1.59 | 54.8 | 40.5 % |
| sound half | one utterance | 1 348 042 624 | 1179.58 | 1.1 | 0.8 % |

Each row's numerator is the graph's own parameter bytes — packed and f32 — which its multiplies and convolutions read once per unit, and only operands that are parameters count; a graph that read one of them twice per unit would move more, and one that read a slice would move less. `talker body` is one frame, `code predictor` one of a frame's sixteen projecting levels (the frame's stream and time divided by the levels it walks), `sound half` the utterance's windows summed with the graph each window actually ran, and `recognition decode` one generated token. The figure in each row is the run whose warm figure is the table's.

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- driver: unknown
- profile: release
- code: c53125de change-sha256 54291d908f5a4696 (the commit and the SHA-256 of the tree's own difference from it — `MEASUREMENTS.md` excluded, because the run that measures rewrites it, and `crates/backends/rlx-mlx-sys` excluded, because the vendored MLX checkout is not this fork's content — which together are the content that produced these figures; a clean tree's digest is `e3b0c44298fc1c14`, the hash of an empty diff)
- rows: this run measured listen, speak, embed.256, embed.8192; every other row is carried from an earlier run of this file, and its figures belong to the run that took them — the `code` line above is this run's tree, not theirs. A carried row keeps its weight-stream row and the artefact lines of the run that took it
- thermal: unknown (pmset -g therm printed no CPU limit line) (`pmset -g therm`, the reading `scripts/check-throttle.sh` gates on)
- load: 2.48 2.88 4.09 (1/5/15-min) at the start, 2.65 2.76 3.80 at the table; every cell line carries its own `load=`
- swap: 5848 MiB in use at the start, 529 MiB grown during the run (ceiling 6000 MiB, raised by `MAHBOT_STAND_MAX_SWAP_MIB` from its default 1024)
- command: `mahbot-inference 4 5 6 7` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`, article from `MAHBOT_EMBED_TEXT`)
- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- model: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- model: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- model: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- model: `embeddinggemma-2-Q8_0.gguf` · 309855456 bytes · sha256 2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135 · url https://huggingface.co/ggml-org/embeddinggemma-2-GGUF · revision 58faf90940886df2d62be555c8765d00dac362c5
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
- text: `article.txt` · 52785 bytes · sha256 df0eb1cd6ed0844698e1b63bc34698f2c3591650fede12d4d58e2c48ce0bdc0f · https://en.wikipedia.org/w/index.php?title=Atlantic_Ocean&oldid=1381231733 (revision 1381231733) · fetched with `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&formatversion=2&revids=1381231733` · cells: `embed.8192` first 5846 words (37543 bytes, sha256 40ee7e6f2fa3f530f018d05e3538314ab9c9656b829d4654eb52a7810faaba3c) and `embed.256` 195 words from `Large quantities of basalt erupted at this time period` (1251 bytes, sha256 727dccf3fdbd6460acd836ef132e0679f2904206f940df20b10c2cbcd47ce960) · the text itself is not committed
<!-- END stand -->

## What this record is

A stand run of the tree its `code` line names, on `Apple M3 Pro (Metal)`, with
`RLX_WGPU_DBG_COUNTERS` and `RLX_WGPU_SCHEDULE` unset and `RLX_WGPU_ENCODE_CHUNK`
at its default 64 steps — the scheduling shape the product path walks, and the one
these figures belong to. The `rows` line names which of them that run measured;
the rest keep the figures of the run that took them.

### The cell, unit by unit

- `speak` — the row above and the run's own lines, nothing between them: the
  generation loop is the cell (`loop ms` against `sound ms`), the waveform digest
  `9d837a9c4164ba94`, the round trip equal. The waveform left the pin it carried
  before this change (`231d823496450627`) when the packed multiply's accumulator
  became one scalar per row: a different summation order in the talker body's
  prefill moves the seeded draws somewhere in the 330 rows, and the samples that
  follow are another waveform of the same sentence — 330 rows, 633 600 samples, the
  prompt digest unchanged, and the round-trip transcript still exact. The block
  below is what the run printed: the repeat whose `residency` line carries the
  peak the row quotes (each repeat prints one), the run's own summary `speak` line,
  the parent's
  `build_ms` line — this run took four cells, so it is their sum — and the witness
  lines all three repeats agree on. The row's four figures are the
  medians of the three (spreads: `cold` 828.8 ms, `warm` 177.1, `build` 981.2,
  `peak` 0.672 GiB).
  Two items were dropped, each priced on the
  decoder's own graphs by `tests/sound_decoder_cost.rs` (one subject per process):
  the DAC windows' right margin, **6.0 (5.1–8.8) ms** at 148→144 frames and
  **5.8 (4.7–6.0) ms** at 152→148 on group 4-6, **3.4 (2.4–4.1)** and
  **3.0 (1.6–4.3) ms** at the same two windows on group 6-8 — **18.2 ms** an
  utterance, 9/9 rounds on every window — and a front graph of the tail's own
  length, which builds in **89.5–103.5 ms** (medians of 7 alternating rounds in
  this session's two readings, spreads reaching 88.9–1 585.0) — one window's
  figure: the same probe on this tree has read 89–170 ms across windows — against
  the **19.7 ms** (9/9, spread 18.9–20.4) a padded last window costs to run. The
  build is the quantity that recurs and the padding the one that pays it off, so
  a one-shot request loses by the first and a repeated length does not pay it.
  **Known gap:** the shipping library path is the wake-word bench's corpus in one
  process, and that steady state — retained graphs against the guard's ceiling —
  has not been run.

  ```
  speak residency start_mb=196.7 prefill_mb=1431.6 body_mb=1541.1 loop_graphs_mb=2554.1 sound_graphs_mb=448.8 peak_mb=2603.9 map_mb=1482.4
  speak cold=15874.6 warm=14165.7 s_per_s=0.537 loop_ms=12986.1 sound_ms=1179.6 peak=2.42 map=1.38 cold_spread=828.8 warm_spread=177.1 peak_spread=0.672
  build_ms 4902.4 build_spread=981.2
  witness speak rows=330/330 samples=633600/633600 waveform_digest=9d837a9c4164ba94/9d837a9c4164ba94 prompt_digest=fb160ff9665b4331/fb160ff9665b4331 equal=true
  witness speak-roundtrip wave=raw transcript_sha256=f2e47729461ad4c95ed9c0c441a65ab5700613338c5132935945adfa8537b164/f2e47729461ad4c95ed9c0c441a65ab5700613338c5132935945adfa8537b164 equal=true
  ```

- `listen` — 1 745.1 ms, **0.100 s per second of speech** (warm spread 11.2 ms;
  cold 2 127.9 ms), with its transcript and `2f93dfa1d7cdeeb1` unchanged — the
  decode that dominates the cell has moved since the 2 257.6 ms row this replaces.
  The one-scalar-per-row packed multiply is inside that, from its own paired
  window: prefill 563.4–565.9 → 522.5–523.5 ms (−7.3 %), encoder 180.2–183.0 →
  177.0–179.3 ms — a move smaller than this row's own encoder repeats span
  (175.9–182.3 ms across its three runs), so read as flat-to-small rather than as
  a win. This
  row is another window of the same tree: its three runs read 1 734.2–1 745.4 ms
  and its prefill 528.8–530.5. The weight-stream row reads 15.84 ms a token at
  69.5 GB/s.
- `embed.8192` — **2 322.6 ms** for one 8 192-row bidirectional pass (three runs,
  warm spread 34.0 ms; cold 2 388.6 ms, `load=` 2.4–2.7), the model's own masking
  throughout: the twenty sliding layers walk their ±512 band, the four global
  ones all 8 192 keys. The same run's `build ms` is 1 147.8 ms — the graph
  compile plus the Q8_0→f16 widening, neither inside the timed path — and its
  `peak` and `arena` are 1.77 GiB and 1.31 GiB. The reference engine reads
  **~2.39 s of encode** for the same text on this machine — llama.cpp `b11724`
  (commit `99db719bdf07`), the same Q8_0 artefact, Metal, flash attention on,
  one 8 192-row physical batch, weights resident: 3.20–3.22 s of process over
  five runs against 0.81–0.83 s for a one-token prompt over three. This row is
  2.8 % under that figure and 3.2 % (77.4 ms) under the 2.4 s bar. `embed.256`
  is the same pass over 256 rows: **66.7 ms** warm (spread 0.2 ms; cold 99.3
  ms, `load=` 2.6–2.7), `build ms` 727.1, `peak` 1.63 GiB, `arena` 547.9 MiB —
  tens of milliseconds.

  **What is portable, and what is this device.** The 2.31 s row rides on the
  cooperative-matrix feature. The same graph with `MAHBOT_EMBED_GEMM=packed` —
  the file's Q8_0 bytes through `Op::DequantMatMul`, the portable route every
  wgpu device can run — measures **3 400.1 ms** warm at 8 192 rows (three runs with
  the stand's own cells, spread 12.0 ms; cold 3 464.7 ms; `peak` 0.74 GiB,
  `arena` 980.8 MiB, `build ms` 458.3) and **89.9 ms** at 256 rows (spread 0.5 ms),
  1 000.1 ms (41.7 %) over the 2.4 s bar and 1.46× the cooperative arm — 1.5 % and
  3.0 % under the 3 451.9 ms and 92.7 ms the same arm read before the accumulator
  became one scalar per row. Its fixture gate is where the recorded one stands:
  cosine 1.000000 at 256 rows and 0.999999 at 8 192, max-abs 1.3235e-4 and 8.6e-5.
  The win that *is* portable
  is the route's shape: the single bidirectional pass, the banded walk over the
  twenty sliding layers, the loop-expanded kernel and the
  f16-operand/f32-accumulator form are the same work on any wgpu device that runs
  them. The cooperative-matrix tile and this device's 3.5 TFLOP/s are not.

  Where the two figures changed from, measured rather than argued. The dense
  layers are one pass at **16-bit operands**: they split their 8 192 query rows
  into 2 048-row chunks and multiply the 8 192×8 192 score squares cooperatively
  (f16 operands downcast at the staged load, f32 accumulator, as the reference
  engine does), 236.6 ms a layer for the whole chunked step — of which the two
  multiplies are 157.2 ms at 3.58 and 3.42 TFLOP/s, the score matrix's softmax
  16 ms and the transposes and f16 casts ~20 ms
  (`tests/attention_dense_coop_cost.rs`); the same layers read ~0.57 s a layer on
  the tiled kernel before this form. The linear/FFN **multiplies** moved from 2.19–2.57 to
  3.07–3.66 TFLOP/s at the encoder's six shapes — a two-subgroup 64×64 tile with
  32 accumulators and a ping-pong stage, plain WGSL over the cooperative-matrix
  primitive (`tests/encoder_gemm_cost.rs`). The twenty sliding layers keep the
  tiled band kernel at 49.9–51.2 ms a call, 11.7 ms of which is the call's own
  upload/readback floor (`tests/attention_band_cost.rs`). Those figures are each a
  call's own cost with its floor included, so this record does not apportion the
  2.31 s between them: summing them over-counts the floors the fused pass pays
  once. Residency is not what the cooperative form buys: the dense layers' score
  matrix is chunked to 2 048 rows and the output region reuses Kᵀ's bytes, so
  the cooperative row's `peak` (1.77 GiB) and `arena` (1.31 GiB) stay where the
  f32 row's were. What separates the two arms' memory is
  the f16 route's own 6 B per 2 B of weight — the packed arm reads `peak`
  0.74 GiB and `arena` 980.8 MiB — and `build ms` is 1 147.8 against the packed
  arm's 458.3.

  Both rows produce the model's answer — the 8 192-row vector's cosine against the
  reference engine's own output for the same text is 1.000000 at max-abs 8.8e-5,
  and the 256-row one 1.000000 at 1.0e-4 on the cooperative route (0.999999 at
  1.3e-4 packed) (`tests/embed_model.rs`, whose header states the tolerance and
  both routes' residues, and whose fixture
  `tests/fixtures/embeddinggemma2-golden.json` records the engine build that
  produced it).

  **The article, and how a re-run gets it.** The extract endpoint renders the
  page's current text rather than the pinned revision's, so a fetch of the same
  `revids=1381231733` (`query.pages[0].extract` of the JSON) on 2026-10-24
  returned **52 787 bytes, sha256 `426ac83b…`** where the pin is 52 785 /
  `df0eb1cd…`: the pin plus the two characters of the one edit landed after it —
  revision 1381660382's `divided in two parts` → `divided into two parts`.
  Undoing that edit reproduces the pin byte for byte (52 593 characters, sha256
  `df0eb1cd…`), which is how the rows above were taken. The substitution is not
  cosmetic to the cells: the drifted extract's own cut is 37 545 bytes against
  the pinned 37 543 and tokenises to different ids, same 8 192 rows — so a wrong
  article changes what the model reads, and `reference::article` checks the file
  against both figures and refuses anything else. The text itself is not
  committed.

Each row of the weight-stream table divides that graph's own parameter bytes — the
ones a node reaching an output reads — by the ms the same run took for one unit; the
percentage is that quotient over the ceiling at the head of this table — its own
run's for a row this run measured, this section's for a carried row. Every `_mb`
figure in this file is decimal MB (÷10⁶); every GiB one is bytes
÷ 2³⁰ — `peak` is the cell process's own `ru_maxrss`, `map` the two model files it
mapped, and `arena` prints its own MiB/GiB — labelled once here. `peak` and
`build ms` follow the machine more than the graph: both move between runs of one
tree, and the summary line carries each run's own spread.

### The sound half's convolution blocks

The sound half's graphs are three windows of the same stage chain, and each block's
GEMMs are the shape this tree's route gives them: `im2col_gpu` + the f32
cooperative multiply (32×32 tile, 16 fragments, no staging, no epilogue —
`matmul_coop_f32_direct`) for the panel multiply, `conv1x1_gemm` for the pointwise
one, and `conv2d_tiled`/`conv_transpose2d_tiled` where the panel would not fit.
FLOPs are the graphs' own convolution arithmetic (`2 · output elems · k`, tallied
over the windows the cell runs); the two byte columns are *demand* — the panel
each output-channel tile reads and the weight each output-column tile reads, at the
32×32 tile — and the cache serves part of both, which is why they are reported
apart from the arithmetic rate rather than as traffic.

| block | ms | GFLOP | TFLOP/s | panel GB/s | weight GB/s |
|---|---|---|---|---|---|
| front (0..4) | 310.3 | 563.2 | 1.82 | 113 | 114 |
| stages 4-6 | 561.1 | 1 077.6 | 1.92 | 120 | 120 |
| stages 6-8 | 394.2 | 368.2 | 0.93 | 62 | 59 |

The three blocks are one session's window split, and that session's sound half read
1 265.3 ms with a 121.5 GB/s read probe; this stand's sound half reads 1 179.6 ms on
a box whose probe read the figure at the head of its own table, so the two are read
against their own probes and not subtracted from one another.

The two heavy blocks are at 1.8–1.9 TFLOP/s and their GEMM demand sits at the
probe's ceiling, so what a form change buys them is fewer reads for the same
arithmetic, not a wider tile. Stages 6-8 carry a smaller arithmetic load over more
structure: 114 steps per window set of which 9 are panel multiplies, 9 pointwise
GEMMs, 3 `conv2d_tiled`, 3 `conv_transpose2d_tiled` and 90 elementwise/concat/
narrow/activation passes over the 24 kHz activations. Its own read demand is
half the heavy blocks' and its rate is half theirs, so the 6-8 form to change is
the one that materialises a panel at all (an implicit-GEMM convolution reading the
input through its taps); the routing answer is measured and wrong — see the
dismissed forms, where taking the panel away and leaving `conv2d_tiled` costs it
6×.

### The frame's schedule

The body frame — the loop's dominant graph — compiles to **394 steps with no
`buffer_copy` at all** and an 8 KiB median dispatch: each `KvAppend` writes its
row where the attention reads it (`Arena::finish_plan`'s placement), so the frame
has no copy dispatch and no second pass over the arena. The predictor's frame is
1 315 steps at a 4 KiB median.

`RLX_WGPU_DBG_COUNTERS=1` splits the body frame in a counter run of its own — 433
frames of one run, with the switch's fence and per-step census inside its wall, so
its per-frame figure is not this table's 13.49 ms and the two are not subtracted
from one another — into
**19.27 ms of wall** (p10–p90 19.21–23.38), **18.17 ms of device time**, in
**6 dispatch passes**,
`encode` 2.87 ms, one `wait` of 16.11 ms and 0.01 ms of readback. So the frame is
94 % device and **wall − device = 1.10 ms**. Its steps, by kind: `dequant_matmul_gguf:169`,
`layer_norm:57`, `rope:56`, `fused_residual_ln_tee:55`, `attention:28`,
`fused_swiglu:28`, `fused_residual_rms_norm:1` — 394, with no `buffer_copy`
among them. The predictor's frame, same switch: 1 315 steps in 21 passes,
35.74 ms of wall over 34.71 of device.

### What one step costs, and what a dispatch's setup costs

`small_step_dispatch_cost` measures the marginal cost of an independent schedule
step directly: 1 024 `RmsNorm` steps over one `[1, 1024]` row is about 1 µs a
step, and 8 steps cost 0.28 ms of call. The one-row packed multiply's own fixed
cost, from `gguf_one_row_stream_rate`'s step-count sweep, is 192–526 µs a *call*,
and its marginal step 24–42 µs across forms (40–56 GB/s). So the predictor frame
is **not** dispatch-count-bound: its non-multiply steps cost about a millisecond
of it. The encode-side setup is not the explanation either: the frame's packed
multiplies pay the create-plus-write-plus-bind-group cost, ~2.9 ms a frame at
5.7 µs an op. The buffer placement is not it either
(`RLX_WGPU_ALL_PARAMS_WEIGHT=1` reads 129 ms for the body frame and 281 for the
predictor's against 23 and 41 — 5–7× worse — and `RLX_WGPU_SHARE_WEIGHTS=0` is
flat at 23.52/40.77 against 23.44/40.63).

### The loop's in-situ rate

`tests/gemm_ceiling.rs::synthesis_ceiling` measures the same forms in a joined
call of 32–128 steps: body 78.7 GB/s and predictor 90.2 GB/s on that session's box.
This record's cell reads the loop's own bytes — 850 870 272 B a body frame and
87 109 632 B a predictor level — at the weight-stream table's **63.1 and
54.8 GB/s** (13.49 ms a body frame and 1.59 ms a predictor level), so in situ the
loop runs **1.25× (body) and 1.65× (predictor) below what the same forms reach when
their dispatches are back to back**. The predictor's multiplies are the harness table's Q8_0 forms at
`k = 1 024` — one level is
`n = 2 048` (attention out, head, proj_in), `3 072` (down), `4 096` (qkv) and
`6 144` (gate+up) over four blocks — which read 79.7–102.2 GB/s in that table; the
body's are the Q4_K MLP and Q6_K down forms at 70.2–85.6. The gap is a property of
the work partition — one workgroup per output slab, weights streamed, and a
dependent `layer_norm`/`rope`/`attention` chain between the multiplies — not of the
memory system.

At the packings' own arithmetic intensity — 3.56 FLOP/B for Q4_K, 2.44 for Q6_K
and 1.88 for Q8_0 — that is **0.103 TFLOP/s for the predictor's Q8_0 stream in
situ** and 0.154–0.225 for the body's, against 0.12–0.37 across the harness table
(the predictor's own widths 0.15–0.19) and the sound half's 0.93–1.92. The
multiply's ceiling here is bytes over a dependent chain, not arithmetic: at this
row's width the work is one output slab a workgroup, and neither the X side nor
the workgroup shape is the term. One pass over a unit's eight X groups feeding four
columns instead of two measured 4–5 % *slower* on the frame's own widths (57.1
against 59.8 GB/s at `n = 1 024`, 71.1 against 74.2 at `n = 2 048`) and 4.3 %
faster at `k = 2 048 n = 2 048` (87.0 against 83.4), where a lane walks two units
instead of one; staging the row in workgroup storage so every subgroup of a
workgroup reads it once cost **7–20 % on every form**: 17.8 % and 19.8 % off the
two Q8_0 `n = 1 024` arms (48.8–49.3 against 59.4–61.5 GB/s), 14.9 % off the Q4_K
block (55.2 against 64.9) and 7.4 % off Q6_K (67.8 against 73.2) — the 8 KiB stage
trades the residency that pays; and a 64-thread workgroup is flat in situ (body frame 19.58
against 19.75 ms, predictor frame 36.77 against 36.68) while 256 threads is flat to
worse in the harness (56.1 against 61.5 at `n = 1 024`, 112.7 against 110.0 at
`n = 49 152`). What moves the rate is more of the chain in flight, not a narrower
fetch, a wider tile or a staged one.

### The weight read: 16-byte legs and the tail guard

The weight window is read as `array<vec4<u32>>` (`read_w16`): each scheme's block
is a few contiguous 16-byte legs — Q4_K's header and its two `qs` halves, Q6_K's
`ql`/`qh` halves and its 16-byte scale row, Q8_0's two quant legs — so one fetch
carries what four word fetches carried, and byte-granular reads pick a byte out of
the leg they already fetched. The bytes read are unchanged and so is the
accumulation order; the parity cells are the ones the arm already has
(`gguf_gemv_wide_arm` against the CPU arm, `gguf_batch_join` joined-vs-unjoined,
`gguf_gemv_in_place`, `gguf_dequant_matmul_prefill_parity`). Each scheme's block
scale `d` is read at its own two bytes (`dq_read_f16`, one spelling for Q6_K and
Q8_0), so neither scheme has an access leaving its block. A leg that does not
start on a 16-byte group boundary is assembled from the two groups it straddles,
so the fetch of a block's last leg reaches one group past the bytes it consumes;
the weight buffer and its shards therefore end on a 16-byte multiple, and the
window cut (`gguf_gpu::gemv_weight_window`) asserts the round-up fits instead of
clamping. The activation arena's own end is rounded to a 16-byte multiple as well
(`Arena::from_plan`, which asserts it): its slots are four bytes per element, and
a packed weight can be the last of them. `tests/gguf_gemv_tail_window.rs` is the
fixture (a 1 260-byte Q6_K weight, its last column against the CPU arm, which
fails loudly if the rounding is reverted).

The arm's rates, per form, four columns a subgroup with four rows' legs in
flight (`RLX_WGPU_GEMV_ILP`'s entry, which every chunk at or above 1 024 columns
takes), the minimum of five rounds in `gguf_one_row_stream_rate`, one session's;
`TFLOP/s` is `GB/s` at the scheme's own packing
intensity (3.56 FLOP/B for Q4_K, 2.44 for Q6_K, 1.88 for Q8_0):

| form | workgroups | MB per dispatch | GB/s | TFLOP/s |
|---|---|---|---|---|
| Q8_0 k=1024 n=1024 | 64 | 1.11 | 65.3 | 0.12 |
| Q8_0 k=1024 n=2048 | 128 | 2.23 | 79.7 | 0.15 |
| Q8_0 k=2048 n=2048 | 128 | 4.46 | 90.1 | 0.17 |
| Q4_K k=2048 n=2048 | 128 | 2.36 | 70.2 | 0.25 |
| Q8_0 k=1024 n=6144 | 384 | 6.68 | 102.2 | 0.19 |
| Q4_K k=2048 n=6144 | 384 | 7.08 | 85.6 | 0.30 |
| Q6_K k=6144 n=2048 | 128 | 10.32 | 76.3 | 0.19 |
| Q4_K k=2048 n=8192 | 512 | 9.44 | 91.8 | 0.33 |
| Q4_K k=2048 n=49152 | 3 072 | 56.6 | 103.8 | 0.37 |
| Q8_0 k=1024 n=49152 | 3 072 | 53.5 | 113.6 | 0.21 |
| Q6_K k=2048 n=151936 | 9 496 | 255.3 | 100.9 | 0.25 |

The rate a dispatch reaches rises with the work in it. These eleven are one
session's minima, and a session moves them: this session read 0.3–18 % above the
one it replaces on the same forms at the same 131–132 GB/s probe, and two runs a
few minutes apart differed by 3–10 %. A row moving inside that band is the box,
not the form; a claim that a form moved needs a pair from one window, which is
what the floor's A/B below is. What this table and the cell's own
figures separate is a **route floor**, not a second form: the
four-column entry already existed, and the question was which chunk widths reach
it. The floor was 4 096 columns; at 1 024 the frame's own widths take it. Paired
by rebuilding the crate with each floor and alternating the two binaries — no
switch moves this constant, so the pair is rebuild-to-rebuild, not a within-session
A/B — the Q4_K `k = n = 2 048` block reads 63.4 → 64.7 GB/s (0.225 → 0.230
TFLOP/s), the Q8_0 `n = 1 024` form 52.1 → 54.3 (0.098 → 0.102) and its `n = 2 048`
form 69.5 → 70.9 (0.131 → 0.133), while Q6_K `k = 6 144, n = 2 048` — the body's
down projection, whose per-column walk already keeps a quarter of the threads
idle — reads 79.9 → 74.8 (0.195 → 0.182). That session's box read the `n = 1 024`
form at 52.1/54.3 GB/s against this table's 65.3 from a different run, which is the
same reading-to-reading spread the paragraph above names. End to end, alternating
on one box: the predictor phase 12 080.6 → 11 788.6 ms and the cell 19 758.4 →
19 446.3 ms.

### Dismissed forms, measured on this tree

Numbers from the sessions that tried each, paired on one box back to back; none of
these is in the tree.

| form | measurement |
|---|---|
| 32×64 cooperative tile (32 accumulators) | sound 1 857.3 → 2 019.1 ms (+8.7 %), loop flat |
| 64×64 cooperative tile | sound +33 % |
| 64-thread workgroup, two 32×32 tiles | front 321.4 → 316.7, 4-6 580.7 → 572.6, 6-8 411.6 → 413.0 ms — flat |
| A/B fragment loads interleaved with the k-step's multiplies | sound 1 322.3 → 1 322.1 ms — flat |
| grouped panel-reuse entry (one panel read per output-channel block) | the panel's re-reads cut about four-fold, and every block slower: front +4.6 %, 4-6 +12.9 %, 6-8 +6.0 % |
| 4-warp N-split of the direct entry | front 313.2 → 401.4/469.5, 4-6 562.4 → 761.3/758.0, 6-8 391.7 → 482.1/487.9 ms |
| `conv2d_tiled` instead of im2col+GEMM for stages 6-8 (`RLX_WGPU_IM2COL_MIN_COUT=128`) | 6-8 409.7 → 2 599.3 ms, sound half 1 319.5 → 3 483.3 |
| the same for the front (`RLX_WGPU_IM2COL_MIN_SPATIAL=4096`) | front 327.5 → 2 505.5 ms, 4-6/6-8 flat |
| contraction split across workgroups (knob since removed) | same bytes slower at every form and monotonically with split: 51.3 → 36.7 → 24.2 GB/s at 1/2/4 |
| cross-window dispatch join; two whole multiplies in one dispatch | built, flat, removed |
| middle one-row partition (two columns a subgroup, two rows a lane) | harness Q4_K 63.0 → 67.6 GB/s (0.224 → 0.240 TFLOP/s), Q8_0 68.0 → 71.5 (0.128 → 0.135) at n = 2 048; in situ flat (predictor 36.96/36.49 → 37.25 ms) |
| the packed multiply re-rowed — one kernel decoding once for 2..16 rows | 3.2–5.9× slower than the per-row arm at sixteen rows; not kept |
| the code predictor's frame as one graph over `m` frames in flight | `m = 16` read the stack once — 230.1 ms for sixteen frames, 14.4 ms a frame against 26.4 ms for a one-row frame — but frame `f`'s codes feed frame `f + 1`, so a chunk of `m` frames needs `m` passes: 4 passes on 82 of 83 chunks at `m = 4`, 8 on 41 of 42 at `m = 8`, the loop 3.5× and 5.9× the serial walk |
| splitting the predictor's fused qkv at the code level | +1.9 % |
| `RLX_WGPU_ALL_PARAMS_WEIGHT=1` | 129 and 281 ms for the body and predictor frames against 23 and 41 |
| the decoder's transposed convs as phase panels onto the existing f32 multiply — `s` multiplies of `[c_out, 2·c_in]` by one shared `[2·c_in, l]` panel | block 330.5 → 412.1 ms over the cell's three windows, a **+24.7 %** time change (81.6 ms slower); the panel's rates come first — 0.777/0.713/0.544/0.343 TFLOP/s against the shipped `ConvTranspose2d` step's 0.802/0.841/0.754/0.450 — so the panel is 0.9 / 5.8 / 13.9 / 14.2 ms slower a form, and 7–142 MB a form (`tests/sound_transpose_form.rs`) |

<!-- BEGIN bench -->
## Wake-word bench

_Measured 2026-10-19 05:37 UTC._

status: `complete`

| metric | value |
|---|---|
| recall | 31/50 (62.0 %) |
| false reactions | 17/121 (14.0 %) |
| false accepts | 0 events · 0.0000/h over VAD-active speech (0.807 h) · 0.0000/h over wall-clock audio (1.066 h) |
| wall clock | 370.3 s |
| peak resident | 1280 MiB |
| corpus | 394 of 394 clips reused, 0 synthesized in 0.0 s |
| windows per encoder pass | at most 16 of 16 · 730 partial of 1567 passes |

| phase | seconds |
|---|---|
| enrolment | 12.7 |
| inputs | 7.3 |
| non-phrase | 40.3 |
| non-phrase synthesis | 0.2 |
| real audio | 295.4 |
| recall | 8.6 |
| recall synthesis | 0.1 |

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: ea6ea643-dirty
- command: `cargo bench -p mahbot-inference --features bench --bench wake_word -- --asr-body ~/.mahbot/userspaces/admin/models/Qwen3-ASR-1.7B-Q4_K_M.gguf --asr-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --tts-body ~/.mahbot/userspaces/admin/models/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf --tts-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf --voice ~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav --corpus ~/.mahbot/faph_corpus --report /tmp/mahbot-wake-report-final.json --lock /tmp/mahbot-wake.lock --timeout 3600 --max-rss 6144`
- ASR body: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- ASR sound: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- TTS body: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- TTS sound: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
<!-- END bench -->
