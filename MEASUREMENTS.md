# Measurements

The latest numbers from this crate's three measuring surfaces. Each row is
replaced by the run that measured it: the section carries that run's date and
`code` line, and its `rows` line names which rows the run took — a row a later
run did not measure is carried, not re-taken, because its own inputs are not what
that run opens. The surfaces are run at different times, so any one of the three
may be older than another. This file is not a log — history is what version
control is for.

<!-- BEGIN stand -->
## Per-model measurement (stand)

_Measured 2026-10-29 06:58 UTC._

| unit | cold_ms | run_ms | covered_s | s_per_covered_s | loop_ms | sound_ms | build_ms | peak GiB (readings, or one carried figure; / ceiling verdict where held) | map | weights | arena | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | 64.5 | 30.8 | - | - | - | - | 167.4 | median 0.54 (readings run 1 0.54, run 2 0.54, run 3 0.56) | 1.53 GiB | 309.2MiB | 3.4MiB | 5f96a55676652c30 |
| ear.075 | 65.2 | 30.7 | - | - | - | - | 166.6 | median 0.56 (readings run 1 0.55, run 2 0.56, run 3 0.56) | 1.53 GiB | 309.2MiB | 3.4MiB | 6c69c810e6e461a5 |
| ear.batch2 | 69.4 | 36.4 | - | - | - | - | 185.6 | median 0.56 (readings run 1 0.56, run 2 0.56, run 3 0.54) | 1.53 GiB | 309.2MiB | 3.6MiB | 035ccc1722ced5ba |
| ear.batch4 | 87.1 | 44.2 | - | - | - | - | 227.8 | median 0.55 (readings run 1 0.55, run 2 0.55, run 3 0.56) | 1.53 GiB | 309.2MiB | 3.9MiB | d41141fe08378af6 |
| listen | 2011.5 | 1751.8 | 17.460 | 0.100 | - | - | 1272.1 | median 1.89 (readings run 1 1.89, run 2 1.89, run 3 1.89) / 2.31 inside | 1.53 GiB | 1.33GiB | 1.10GiB | 2f93dfa1d7cdeeb1 |
| speak | 15331.3 | 14377.5 | 26.400 | 0.545 | 13216.5 | 1160.9 | 1251.6 | median 1.96 (readings run 1 1.66, run 2 1.98, run 3 1.96) / 2.42 inside | 1.38 GiB | 0B | 702.9MiB | 9d837a9c4164ba94 |
| embed.256 | 99.3 | 66.7 | - | - | - | - | 727.1 | 1.63 GiB | 0.29 GiB | - | 547.9MiB | ec6e9fa56756e071 |
| embed.8192 | 2388.6 | 2322.6 | - | - | - | - | 1147.8 | 1.77 GiB | 0.29 GiB | - | 1.31GiB | bfa0336b63d2e3d0 |
| transcribe | 1088199.4 | 377127.1 | 3296.220 | 0.114 | - | - | 1059.4 | cell 2.22 (readings message 10 s cold run 1 2.36, message 10 s cold run 2 2.36, message 10 s cold run 3 2.35, message 30 s cold run 1 2.35, message 30 s cold run 2 2.35, message 30 s cold run 3 2.36, corpus cold 2.36) / 2.35 at | 1.57 GiB | 1.33GiB | - | - |

The cell's number is the **`s_per_covered_s`** column: the **warm** figure (`run_ms`) divided by the `covered_s` column — the seconds of audio the cell's passes themselves covered, `samples / 16 000` for recognition and `rows · 1 920 / 24 000` for synthesis, printed so the rate re-derives from this table alone. `loop_ms` is the synthesis generation loop (speaker, prefill, body, predictor, sampler) and `sound_ms` is the sound half; their sum is `run_ms`, so a ×N synthesis claim reads off both. Both are the same phase sums the cell's `loop_ms`/`sound_ms` tokens record; `speak`'s `sound_phase_ms` field is the sound graphs' summed `step_ms` — the same sum the `sound_ms` column is taken from, carried as a field so a cell line is self-describing. For `speak` the recorded `digest` is the **waveform** digest, not the `prompt_digest` its own line prints. `-` marks a figure the row has none of: in `covered_s` and `s_per_covered_s`, a cell with no covered-audio denominator; in `loop_ms` and `sound_ms`, a cell with no synthesis split (only `speak` has one); in `arena`, a ladder cell whose graphs each have a plan of their own; and in `digest`, a cell with no single answer (the transcription cell's ladder of graphs). The `peak` column is a process high-water (`ru_maxrss`, which no later drop can lower): each cell run is its own process, so the figure the column leads with is the median of the cell's runs' high-waters, not what is live when the line prints. Two rows need a word on whose reading their figure is: `speak`'s is `synthesis_peak_bytes` — each run's high-water when the synthesis half had finished, before the round-trip gate loads the recogniser, whose later reading prints as `process_peak_bytes` and is not part of the row's verdict — and the transcription cell is one process, so its figure is the `process_peak_bytes` its `unit` row printed when the cell had finished, not a median over runs. Beside the leading figure the column lists every other reading the verdict compares, each named with the process that took it (`run 1` of the cell's runs, or the transcription cell's message- and corpus-size calls); a row held to no ceiling lists the readings it has. A cell's earlier moments are not in that list: the transcription cell's print as `moment_peak_bytes`. A row this run did not take carries the figures the run that took it printed — the readings list where that run printed one, the single figure it printed where it did not, which is what the header's `or one carried figure` names. Nothing outside that list enters a verdict, and no reading from another run of this code does. The ceilings beside them are policies, and the verdict those readings support is `inside` when every one is under the ceiling by more than 0.01 GiB, `over` when every one is above it by more, and `at` otherwise: the set meets the ceiling's 0.01 GiB band — readings on both sides of it, or every reading on one side with the nearest inside the band. The transcription cell's own reading is one figure — its `unit` row, printed when the cell had finished — so every reading its verdict compares is a process of that run: the cell's own and the ones it started for a call of message or corpus size, each cold message row's three runs and the corpus-cold call. The three hour-scale inputs are a population of their own and are not in that verdict: their rows carry them, and the section states them apart. The ceilings are 2.31 GiB for `listen`, 2.42 GiB for `speak`, 2.35 GiB for `transcribe` — the transcription cell's is the listening cell's class plus its tagger — and the verdict beside each peak in the table is computed from that row's own readings, not restated. The columns beside the peak name the allocations a reader can hold it against: `weights` (the device's shared weight pool — one copy of each model's parameter bytes, held once however many layouts and rungs bind them, and given back by `release_process_memory`), `arena` (the compiled graph's own plan, `-` on a cell that builds a ladder of graphs, because a sum of plans is not an allocation the peak contains) and `map` (the model files the cell binds, by their size: each is opened by mapping and its tensor bytes are read with `pread` through one scratch buffer, so the mapping is not the copy the process holds and the figure is not resident pages). Each is the cell's own reading of that allocation, taken where the cell prints it; a `-` in `weights` is the one figure a carried row's run did not print, because a measured cell always prints the pool's bytes, `0B` included. Their sum is not `peak` and the record does not claim it is: the peak is what the process touched, and neither the pool nor the files is all resident at once. For `speak` all three are read together, at the end of the synthesis half, so the row is one moment: the talker's pool and graphs are released before the decoder builds, so `weights` there is the pool as it stands then — the decoder's, or nothing where the release emptied it and the graphs keep their weights in their own buffers — and `arena` is the sound half's plan. The peak is `synthesis_peak_bytes`: the process's high-water when the synthesis half had finished, taken before the round-trip gate releases it and loads the recogniser — that gate's later reading prints as `process_peak_bytes` and is not this row's figure. `embed.256` and `embed.8192` are one bidirectional pass each over the whole sequence on a Q8_0 stack. Their `digest` is over the L2-normalised 768-vector the row names — the cell's answer, not the graph's raw output.

The pinned answers: recognition is recorded against `обычные объявления метро делались только на каталонском языке, но не забывали переводить объявления в другие языки, например, итальянский, испанский, английский, французский, итальянский` — compared word for word, at most 2 of them different, its digest and generated-token count printed as witnesses with what moved, and the run carries on; the synthesis round trip — the waveform the `speak` row's digest names, resampled to 16 kHz and read back through the recogniser on its raw samples — must reproduce `Call me Ishmael. Some years ago, now I'm old, beside the sea. I've seen them all, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every ocean, every island, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every` exactly.

### Weight streams, against the machine's streaming ceiling

This run took no stream reading of its own: the rows below are the run that measured them, restated against the ceiling printed below.

Ceiling, measured on this run's device before the cells: **127.1 GB/s** on a streaming read (124.5–132.5 over 5 rounds) and 124.9 GB/s of total traffic on a streaming copy (118.0–126.4). The read figure is the one the rows below are reported against, because a weight is read and never written; the copy figure is the same probe's other arm, recorded beside it. Both come from `rlx_wgpu::bandwidth::stream_ceiling`: a 64.0 MiB vectorised read and a 64.0 MiB vectorised copy, on `Apple M3 Pro (Metal)`. The same probe after the cells read 131.5 GB/s (126.4–132.8), which is the reading-to-reading spread to hold the figure to.

| phase | unit | weight bytes/unit | ms/unit | GB/s | of the read ceiling |
|---|---|---|---|---|---|
| recognition decode | one token | 1 100 961 792 | 16.05 | 68.6 | 54.0 % |
| talker body | one frame | 850 870 272 | 13.54 | 62.8 | 49.4 % |
| code predictor | one level | 87 109 632 | 1.64 | 53.1 | 41.8 % |
| sound half | one utterance | 1 348 042 624 | 1160.93 | 1.2 | 0.9 % |

Each row's numerator is the graph's own parameter bytes — packed and f32 — which its multiplies and convolutions read once per unit, and only operands that are parameters count; a graph that read one of them twice per unit would move more, and one that read a slice would move less. `talker body` is one frame, `code predictor` one of a frame's sixteen projecting levels (the frame's stream and time divided by the levels it walks), `sound half` the utterance's windows summed with the graph each window actually ran, and `recognition decode` one generated token. The figure in each row is the run whose warm figure is the table's.

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- driver: unknown
- profile: release
- code: 3915e22c change-sha256 b592a86c61a31521 (the commit and the SHA-256 of the tree's own difference from it — `MEASUREMENTS.md` excluded, because the run that measures rewrites it, and `crates/backends/rlx-mlx-sys` excluded, because the vendored MLX checkout is not this fork's content — which together are the content that produced these figures; a clean tree's digest is `e3b0c44298fc1c14`, the hash of an empty diff)
- rows: this run measured transcribe, and the transcription section this run wrote is built from its own rows; every other row of the stand table is carried from an earlier run of this file, and its figures belong to the run that took them — the `code` line above is this run's tree, not theirs. A carried row keeps its weight-stream row and, where this run could not read the files, the artefact lines of the run that took it
- thermal: unknown (pmset -g therm printed no CPU limit line) (`pmset -g therm`, the reading `scripts/check-throttle.sh` gates on)
- load: 2.46 3.00 3.10 (1/5/15-min) at the start, 3.10 2.97 2.85 at the table; every cell line carries its own `load=`
- ceiling: 6144 MiB resident set, the guard's default
- swap: 4503.8 MiB in use at the start, 4487.8 MiB at the table — 16.0 MiB, given back during the run (ceiling 1024 MiB growth, the guard's default)
- command: `mahbot-inference 8` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`, article from `MAHBOT_EMBED_TEXT`)
- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- model: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- model: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- model: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
- model: `embeddinggemma-2-Q8_0.gguf` · 309855456 bytes · sha256 2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135 · url https://huggingface.co/ggml-org/embeddinggemma-2-GGUF · revision 58faf90940886df2d62be555c8765d00dac362c5
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
  generation loop is the cell (`loop_ms` against `sound_ms`), the waveform digest
  `9d837a9c4164ba94`, the round trip equal. Its peak is **1.96 GiB**, the median of the
  three runs' `synthesis_peak_bytes` readings — 1.66, 1.98 and 1.96 GiB, each the process
  high-water when the synthesis half had finished, each printed with the run that took it
  beside this figure — and every one of those readings is under the 2.42 GiB ceiling set for
  the cell, the widest by more than the 0.01 GiB the printed numbers resolve, so it is reported **inside** it. The talker's graphs, its host copies and the
  pool's copy of its weights are released and their pages returned
  (`release_process_memory`) before the decoder is built, but a resident-set high-water cannot
  fall, so the decoder's build adds to it and does not replace it. The row's three allocation
  figures are read at that same moment, at
  the end of the synthesis half, and say what the half holds then: `weights` 0B — the
  release empties the shared pool, and the decoder keeps its weights in its own graphs —
  `arena` 702.9 MiB, the decoder's compiled plan, and `map` 1.38 GiB, the two model files
  by their size, opened by mapping and their tensor bytes read through one scratch buffer,
  so the mapping is not a copy the process holds. The row is
  the median of the three runs of its invocation.
  Two items were dropped, each priced on the
  decoder's own graphs by `tests/sound_decoder_cost.rs` (one subject per process, on
  this machine, one run of that harness each — not this run's readings):
  the DAC windows' right margin, **6.0 (5.1–8.8) ms** at 148→144 frames and
  **5.8 (4.7–6.0) ms** at 152→148 on group 4-6, **3.4 (2.4–4.1)** and
  **3.0 (1.6–4.3) ms** at the same two windows on group 6-8 — **18.2 ms** an
  utterance, 9/9 rounds on every window — and a front graph of the tail's own
  length, which that harness builds in **89.5–103.5 ms** against the **19.7 ms** a
  padded last window costs it to run. The build is the quantity that recurs and the
  padding the one that pays it off, so a one-shot request loses by the first and a
  repeated length does not pay it.

- `listen` — the row above is the figure, and its transcript and digest are the ones
  the recognition work recorded (`2f93dfa1d7cdeeb1`): a reading of the same audio.
  Its **1 751.8 ms** is the recognition cell's own speed on this machine, and its peak
  is **1.89 GiB** against the 2.31 GiB ceiling that was set for it: the body's parameter
  bytes are held once, in the device's weight pool (`weights` 1.33 GiB), the compiled
  graphs' plans are 1.10 GiB, and the model files the cell binds (1.53 GiB by their size)
  are opened by mapping with their tensor bytes read through one scratch buffer, so the
  mapping is not a copy the process holds and they are not pages inside that peak.
- `embed.8192` — **2 322.6 ms** for one 8 192-row bidirectional pass (cold 2 388.6
  ms), the model's own masking
  throughout: the twenty sliding layers walk their ±512 band, the four global
  ones all 8 192 keys. The same run's `build_ms` is 1 147.8 ms — the graph
  compile plus the Q8_0→f16 widening, neither inside the timed path — and its
  `peak` and `arena` are 1.77 GiB and 1.31 GiB. The reference engine reads
  **~2.39 s of encode** for the same text on this machine — llama.cpp `b11724`
  (commit `99db719bdf07`), the same Q8_0 artefact, Metal, flash attention on,
  one 8 192-row physical batch, weights resident. Run here for this record, that
  engine's own process times are 3.20–3.22 s over five runs against 0.81–0.83 s
  for a one-token prompt over three. This row is
  2.8 % under that figure and 3.2 % (77.4 ms) under the 2.4 s bar. `embed.256`
  is the same pass over 256 rows: **66.7 ms** warm (cold 99.3
  ms), `build_ms` 727.1, `peak` 1.63 GiB, `arena` 547.9 MiB —
  tens of milliseconds.

  **What is portable, and what is this device.** The 2.31 s row rides on the
  cooperative-matrix feature. The same `embed` cells with `MAHBOT_EMBED_GEMM=packed` —
  the file's Q8_0 bytes through `Op::DequantMatMul`, the portable route every
  wgpu device can run, one run of the stand with that switch — measure
  **3 400.1 ms** warm at 8 192 rows (cold 3 464.7
  ms; `peak` 0.74 GiB,
  `arena` 980.8 MiB, `build_ms` 458.3) and **89.9 ms** at 256 rows,
  1 000.1 ms (41.7 %) over the 2.4 s bar and 1.46× the cooperative arm. Its
  fixture gate is where the recorded one stands:
  cosine 1.000000 at 256 rows and 0.999999 at 8 192, max-abs 1.3235e-4 and 8.6e-5.
  The win that *is* portable
  is the route's shape: the single bidirectional pass, the banded walk over the
  twenty sliding layers, the loop-expanded kernel and the
  f16-operand/f32-accumulator form are the same work on any wgpu device that runs
  them. The cooperative-matrix tile and this device's 3.5 TFLOP/s are not.

  Where the two figures come from, measured rather than argued. The dense
  layers are one pass at **16-bit operands**: they split their 8 192 query rows
  into 2 048-row chunks and multiply the 8 192×8 192 score squares cooperatively
  (f16 operands downcast at the staged load, f32 accumulator, as the reference
  engine does), 236.6 ms a layer for the whole chunked step — of which the two
  multiplies are 157.2 ms at 3.58 and 3.42 TFLOP/s, the score matrix's softmax
  16 ms and the transposes and f16 casts ~20 ms
  (`tests/attention_dense_coop_cost.rs`). The linear/FFN **multiplies** read
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
  the f16 route's own 6 B per 2 B of weight — the same cells with
  `MAHBOT_EMBED_GEMM=packed` read `peak` 0.74 GiB and `arena` 980.8 MiB — and
  `build_ms` is 1 147.8 against that arm's 458.3.

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
percentage is that quotient over the ceiling at the head of this table, which is this
run's probe for every row. Every `MiB` figure in this file is bytes ÷ 2²⁰ and every
GiB one bytes ÷ 2³⁰ — the peak figures are process high-waters (`ru_maxrss`), `map`
the model files the cell binds by their size (opened by mapping, their tensor bytes
read through one scratch buffer, so not pages held as a mapping), and `arena` prints
its own MiB/GiB — labelled once here. The peak and `build_ms` follow the machine more
than the graph: both move between runs of one tree.

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
its per-frame figure is not this table's 13.54 ms and the two are not subtracted
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
87 109 632 B a predictor level — at the weight-stream table's **62.8 and
53.1 GB/s** (13.54 ms a body frame and 1.64 ms a predictor level), so in situ the
loop runs **1.25× (body) and 1.70× (predictor) below what the same forms reach when
their dispatches are back to back**. The predictor's multiplies are the harness table's Q8_0 forms at
`k = 1 024` — one level is
`n = 2 048` (attention out, head, proj_in), `3 072` (down), `4 096` (qkv) and
`6 144` (gate+up) over four blocks — which read 79.7–102.2 GB/s in that table; the
body's are the Q4_K MLP and Q6_K down forms at 70.2–85.6. The gap is a property of
the work partition — one workgroup per output slab, weights streamed, and a
dependent `layer_norm`/`rope`/`attention` chain between the multiplies — not of the
memory system.

At the packings' own arithmetic intensity — 3.56 FLOP/B for Q4_K, 2.44 for Q6_K
and 1.88 for Q8_0 — that is **0.100 TFLOP/s for the predictor's Q8_0 stream in
situ** and 0.153–0.224 for the body's, against 0.12–0.37 across the harness table
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
one it replaces on the same forms, and two runs a few minutes apart differed by 3–10 %. A row moving inside that band is the box,
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

_Measured 2026-10-27 03:39 UTC._

status: `complete`

| metric | value |
|---|---|
| recall | 46/50 (92.0 %) |
| false reactions | 46/121 (38.0 %) |
| false accepts | 2 events · 2.4773/h over VAD-active speech (0.807 h) · 1.8756/h over wall-clock audio (1.066 h) · one event = 1.2386/h |
| thresholds | `floor_min=0.55, floor_max=0.75, tail_z=2.33, fire_factor_cap=0.55, anchor=0.838, reset_ratio=0.95, adaptive_k=2.5, adaptive_ceiling=2.6` |
| derived point | floor 0.7500 · fire factor 0.3520 · static bar 1.0560 · reset 0.3344 |
| wall clock | 288.2 s |
| peak resident | 1339 MiB |
| corpus | 394 of 394 clips reused, 0 synthesized in 0.0 s |
| windows per encoder pass | at most 16 of 16 · 731 partial of 1567 passes |

| phase | seconds |
|---|---|
| enrolment | 9.8 |
| inputs | 7.1 |
| non-phrase | 27.3 |
| non-phrase synthesis | 0.2 |
| real audio | 234.0 |
| recall | 4.1 |
| recall synthesis | 0.1 |

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: dd4a1a53 change-sha256 6822e184bc966446
- command: `cargo bench -p mahbot-inference --features bench --bench wake_word -- --asr-body ~/.mahbot/userspaces/admin/models/Qwen3-ASR-1.7B-Q4_K_M.gguf --asr-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --tts-body ~/.mahbot/userspaces/admin/models/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf --tts-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf --voice ~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav --corpus ~/.mahbot/faph_corpus --report /tmp/mahbot-wake-report-final.json --lock /tmp/mahbot-wake.lock --timeout 3600 --max-rss 6144`
- ASR body: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- ASR sound: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- TTS body: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- TTS sound: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
<!-- END bench -->

<!-- BEGIN transcribe -->
### Transcription: one call over the pinned corpus and the music pool

_Measured 2026-10-29 06:58 UTC._

**The headline — the one figure the latency bar is stated against: 98.2 ms of call per second of audio** over the pinned corpus. It is the whole call from hand-over of the audio to the returned text: the file read, the gain and resample, the detector's mask, the tagger's labels, the cut, every compilation the call needs, the recognition passes, the guards and the joins. The bar is 100 ms per second of audio, 103 at the tolerance the plan allows, and this run is **98.2 ms/s inside the bar at the 103 it allows** — 0.98× the 100 ms/s the plan asks for, and 4.8 ms/s of headroom under a figure that already carries its tolerance. The `e2e` row carries the bar beside the figure and `gated=false`: reported against the bar, not gated by it, because nothing in this run fails when the figure misses it. The rows that do fail the run — `seams` and the noise pool's `invented` — say `gated=true`; every other one is a reading, a breakdown or an absolute figure.

The message rows are absolute figures and the bar does not gate them: a 10-second message takes **2411 ms cold** (419 ms of it the model load, so 241 ms/s for that message, spread 173 ms); **1161 ms warm** (116 ms/s, spread 18 ms); a 30-second message takes **5208 ms cold** (416 ms of it the model load, so 174 ms/s for that message, spread 42 ms); **3336 ms warm** (111 ms/s, spread 28 ms); a ten-second cold call pays 419 ms of model load before a sample is read — 42 ms/s of the message's own ten seconds on that alone — so no implementation inside this plan's rules could put it under the per-second bar. `cold` is a claim about this process and nothing else: it holds no mapping, host copy, device buffer or compiled graph of the models when the span starts. The operating system's own file cache is not this crate's to control, and is not part of the claim. Every one of those rows says `gated=false` and carries the same `bar_ms_per_audio_s`/`bar_tolerance_ms_per_audio_s` fields the headline does, so the bar is beside the figure it does not gate rather than a comparison the reader has to make. Every message row is the median of three runs — a cold row's in three processes of its own — and carries the range the three saw: `total_spread_ms` on a cold row, `warm_spread_ms` on a warm one. A cold row also carries `load_ms`, the model files it read before a sample; every message row carries its own split — `build_ms` with `compiles` (every graph the call built: a cold process builds the ladder's rungs and the prompt's shapes from nothing) and `recognise_ms` with `chunks`, the recognition passes inside it. The peak each row prints is in bytes and named after its scope: `process_peak_bytes` is the process the row's `process=` field names, read as that process's final reading — `own` on a cold row is that child, which exits as it printed it — and a cold row's `process_peaks_bytes` lists all three of its runs' high-waters in run order, the row's own figure being the median run's. A warm row's `process=cell` names the cell's own process, which prints its final reading on its `unit` row: the high-water a warm row, the `long` row or the `memory` row carries is an earlier moment of that process, so it is `moment_peak_bytes`. A cold row's `total_ms_per_audio_s` carries the load inside it; a warm row's `call_ms_per_audio_s` does not.

**The quality figure, and the second bar: 2.09 % of the covered speech came back with no transcript**, against 2 %: this run is **0.09 points above it**. Its denominator is `speech_pool_s`, the seconds of the speech pool that a recognition chunk covered — the noise pool's covered seconds are counted apart on the same row's `noise_*` fields, and `covered_s` on the `coverage` row and the `corpus` row is both pools together — and its numerator is the seconds of those regions no reading produced, so a region the detector marked but the tagger calls something other than words is counted where the `speech` row's split puts it. The `speech` row carries the bar beside the figure and `gated=false`: reported against it, the same way the headline is. Every speech file has its own `textless-file` row with that file's own percentage beside the same bar — a zero printed rather than a row left out — so the distribution can be read rather than the aggregate taken on trust.

These rows are one pass of `transcribe::Transcriber` over the pinned corpus of 33 real-audio files (64.0 min of speech and noise) and the eight-clip music pool, plus the constructed boundary material, the pinned clip's parity with the single-shot answer, the longest pinned track and three hour-scale inputs built from the corpus itself. Every transcription is held to the invariant that a region carries a text or a label and never both, and that the regions tile the input exactly — the first at 0, the last at its length. A region that was not read is not silence: it carries the tagger's label and `read=false`, which is how a caller tells nothing-there from failed-to-read. The times are the detector's own 16 ms hops and the labels are the tagger's per-window winners, not a transcript. The table's `unit` row is this cell's, and its two time columns do not mean what they mean on the other rows: its `cold_ms` is the whole cell — the messages, the corpus pass, the music pool, the constructed material, the pinned clip's parity, the hour-scale inputs (each a process of its own, whose wall time the span includes) and the longest track — while its `run_ms` is the corpus pass alone, one call per file, hand-over to text. Its `peak` is likewise the whole cell's own process and not the corpus pass's: the cell's figure is the high-water its `unit` row prints when the cell has finished, and every other reading its verdict compares is one of the processes it started for a call of message or corpus size — a `process_peak_bytes` entry of a row below, or, for a cold message row's three runs, a `process_peaks_bytes` one. The hour-scale inputs are a population of their own and are stated apart in the paragraph below. The cell's own earlier moments (its warm message rows, its `long` row, its `memory` row and its per-phase prints) are `moment_peak_bytes` entries: never part of the verdict, and never higher than the figure above.

| what | numbers |
|---|---|
| inputs | corpus=~/.mahbot/faph_corpus files=33 audio_s=3838.7 music=~/.mahbot/asr_audio clips=8 voice=~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav open_ms=165.3 |
| message | s=10 pass=cold process=own load_ms=419.4 call_ms=1991.6 audio_s=10.000 total_ms_per_audio_s=241.1 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 regions=4 words=16 chunks=2 rungs=1 compiles=1 build_ms=720.5 recognise_ms=1217.8 process_peak_bytes=2529329152 gated=false text="Dedicatória e introdução de amor por perdição. Esta é uma gr" total_ms=2411.0 runs=3 total_spread_ms=172.8 process_peaks_bytes=2529296384,2529329152,2527838208 |
| message | s=30 pass=cold process=own load_ms=415.6 call_ms=4792.1 audio_s=30.000 total_ms_per_audio_s=173.6 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 regions=10 words=49 chunks=6 rungs=3 compiles=3 build_ms=1344.0 recognise_ms=3292.6 process_peak_bytes=2528952320 gated=false text="Dedicatória e introdução de amor por perdição. Esta é uma gr" total_ms=5207.7 runs=3 total_spread_ms=42.1 process_peaks_bytes=2528493568,2525888512,2528952320 |
| corpus-cold | file=speech/librivox/speech-librivox-0027.wav process=own load_ms=415.1 call_ms=39285.0 audio_s=318.0 total_ms_per_audio_s=124.8 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 words=651 process_peak_bytes=2535194624 gated=false text="Dedicatória e introdução de amor por perdição. Esta é uma gr" |
| parity | moved=21 words=25 pinned=22 same_count=false clip_s=17.460 covered_s=15.392 text="Обычные объявления в метро делаются только на каталонском яз" |
| parity-span | span_s=4.720 moved=1 words=9 single=9 same_count=true text="Обычные объявления в метро делаются только на каталонском яз" |
| boundary | pastes=4 regions=17 err_ms=+0.0,+0.0,+0.0,+0.0 mean_ms=+0.0 max_abs_ms=0.0 |
| boundary | gain=1.0000 lifted_gain=29.4828 moved=0 identical=true |
| music instrumental/01-orchestra.wav | audio_s=20.00 regions=1 text_chars=0 labels=Music suppressed_s=0.0 read_ms=1.9 wall_ms=1735.2 call_ms_per_audio_s=86.8 |
| music instrumental/02-solo-piano.wav | audio_s=25.00 regions=1 text_chars=0 labels=Music suppressed_s=0.0 read_ms=2.2 wall_ms=2095.0 call_ms_per_audio_s=83.8 |
| music instrumental/03-electronic.wav | audio_s=20.00 regions=1 text_chars=0 labels=Music suppressed_s=0.0 read_ms=1.9 wall_ms=1271.3 call_ms_per_audio_s=63.6 |
| music instrumental/04-ambient.wav | audio_s=25.00 regions=1 text_chars=0 labels=Music suppressed_s=0.0 read_ms=2.2 wall_ms=1563.1 call_ms_per_audio_s=62.5 |
| music vocal/01-acappella-solfege.wav | audio_s=20.00 regions=4 text_chars=91 labels=Lullaby suppressed_s=0.0 read_ms=2.0 wall_ms=1907.8 call_ms_per_audio_s=95.4 |
| music-report vocal/01-acappella-solfege.wav | transcribed=true read_regions=3 suppressed_s=0.0 labels=Lullaby text="La la la la la la. La, la, so, so, la, la, la, la, mila. Sol" |
| music vocal/02-pop-lyrics.wav | audio_s=20.00 regions=4 text_chars=168 labels=Synthetic_singing suppressed_s=0.0 read_ms=2.0 wall_ms=1780.4 call_ms_per_audio_s=89.0 |
| music-report vocal/02-pop-lyrics.wav | transcribed=true read_regions=3 suppressed_s=0.0 labels=Synthetic_singing text="In the town where I was born, there's a fountain. Michigan, " |
| music vocal/03-vocal-over-band.wav | audio_s=20.00 regions=4 text_chars=97 labels=Music suppressed_s=0.0 read_ms=1.9 wall_ms=1898.4 call_ms_per_audio_s=94.9 |
| music-report vocal/03-vocal-over-band.wav | transcribed=true read_regions=2 suppressed_s=0.0 labels=Music text="Of darkness, oh, she says, I get so lonely, boy. In the sun," |
| music vocal/04-vocal-indian-classical.wav | audio_s=20.00 regions=1 text_chars=0 labels=Music suppressed_s=0.0 read_ms=1.9 wall_ms=2263.9 call_ms_per_audio_s=113.2 |
| music-report vocal/04-vocal-indian-classical.wav | transcribed=false read_regions=0 suppressed_s=0.0 labels=Music text="" |
| music-total | clips=8 audio_s=170.0 wall_s=14.5 call_ms_per_audio_s=85.4 suppressed_s=0.0 declared_chunks=6 chunks=27 instrumental_any_text=0 vocal_read=3 |
| peak | phase=music process=cell moment_peak_bytes=2387755008 weights=1.33GiB host=427819008 |
| textless-file speech/librivox/speech-librivox-0027.wav | audio_s=318.0 chunks=51 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=37662.4 labels=Speech:17 |
| textless-file speech/librivox/speech-librivox-0051.wav | audio_s=315.6 chunks=50 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=37808.0 labels=Speech:14 |
| textless-file speech/librivox/speech-librivox-0093.wav | audio_s=314.1 chunks=52 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=35338.4 labels=Speech:22 |
| textless-file speech/librivox/speech-librivox-0075.wav | audio_s=312.8 chunks=55 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=28519.7 labels=Speech:39 |
| textless-file speech/librivox/speech-librivox-0103.wav | audio_s=307.2 chunks=57 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=35918.6 labels=Speech:35 |
| textless-file speech/librivox/speech-librivox-0088.wav | audio_s=304.2 chunks=53 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=35252.4 labels=Speech:31 |
| textless-file speech/us-gov/speech-us-gov-0252.wav | audio_s=231.5 chunks=36 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=23683.1 labels=Speech:9 |
| textless-file speech/us-gov/speech-us-gov-0147.wav | audio_s=215.9 chunks=34 textless_chunks=1 textless_s=6.4 textless_pct=2.99 textless_bar_pct=2.00 textless_voiced_s=6.4 textless_other_s=0.0 e2e_ms=19256.7 labels=Speech:6 |
| textless-file speech/us-gov/speech-us-gov-0082.wav | audio_s=215.0 chunks=32 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=25986.7 labels=Speech:3 |
| textless-file speech/us-gov/speech-us-gov-0159.wav | audio_s=173.9 chunks=26 textless_chunks=1 textless_s=6.8 textless_pct=3.94 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=6.8 e2e_ms=16202.5 labels=Animal:1;Speech:2 |
| textless-file speech/us-gov/speech-us-gov-0067.wav | audio_s=157.8 chunks=25 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=17550.2 labels=Speech:7 |
| textless-file speech/us-gov/speech-us-gov-0250.wav | audio_s=126.1 chunks=23 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=14140.3 labels=Speech:9 |
| textless-file speech/us-gov/speech-us-gov-0210.wav | audio_s=123.8 chunks=18 textless_chunks=2 textless_s=13.7 textless_pct=11.54 textless_bar_pct=2.00 textless_voiced_s=13.7 textless_other_s=0.0 e2e_ms=10609.9 labels=Speech:5 |
| textless-file speech/us-gov/speech-us-gov-0229.wav | audio_s=100.5 chunks=16 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=10277.1 labels=Speech:3 |
| textless-file speech/us-gov/speech-us-gov-0201.wav | audio_s=58.0 chunks=9 textless_chunks=1 textless_s=6.7 textless_pct=11.60 textless_bar_pct=2.00 textless_voiced_s=6.7 textless_other_s=0.0 e2e_ms=4995.5 labels=Speech:2 |
| textless-file speech/us-gov/speech-us-gov-0130.wav | audio_s=53.3 chunks=8 textless_chunks=1 textless_s=6.1 textless_pct=11.55 textless_bar_pct=2.00 textless_voiced_s=6.1 textless_other_s=0.0 e2e_ms=5183.1 labels=Speech:2 |
| textless-file speech/us-gov/speech-us-gov-0005.wav | audio_s=26.8 chunks=4 textless_chunks=4 textless_s=26.7 textless_pct=100.00 textless_bar_pct=2.00 textless_voiced_s=26.7 textless_other_s=0.0 e2e_ms=2815.4 labels=Speech:1 |
| textless-file speech/us-gov/speech-us-gov-0109.wav | audio_s=19.7 chunks=3 textless_chunks=0 textless_s=0.0 textless_pct=0.00 textless_bar_pct=2.00 textless_voiced_s=0.0 textless_other_s=0.0 e2e_ms=2065.9 labels=Speech:1 |
| corpus | files=33 audio_s=3838.7 covered_s=3296.2 regions=772 text_regions=542 label_regions=230 wall_s=377.1 ladder=128,256,384,512,640,768 |
| peak | phase=corpus process=cell moment_peak_bytes=2387755008 weights=1.33GiB host=427819008 |
| message | s=10 pass=warm process=cell call_ms=1160.7 warm_spread_ms=17.6 runs=3 audio_s=10.000 call_ms_per_audio_s=116.1 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 regions=4 words=16 chunks=2 rungs=1 compiles=0 build_ms=0.0 recognise_ms=1124.1 moment_peak_bytes=2387755008 gated=false text="Dedicatória e introdução de amor por perdição. Esta é uma gr" |
| message | s=30 pass=warm process=cell call_ms=3336.2 warm_spread_ms=28.1 runs=3 audio_s=30.000 call_ms_per_audio_s=111.2 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 regions=10 words=49 chunks=6 rungs=3 compiles=0 build_ms=0.0 recognise_ms=3196.2 moment_peak_bytes=2387755008 gated=false text="Dedicatória e introdução de amor por perdição. Esta é uma gr" |
| e2e | scope=corpus files=33 audio_s=3838.7 wall_ms=377127.1 call_ms_per_audio_s=98.2 covered_s=3296.2 bar_ms_per_audio_s=100 bar_tolerance_ms_per_audio_s=103 gated=false |
| breakdown | read_ms_per_audio_min=3.4 mask_plan_ms_per_audio_min=16.2 tagger_ms_per_audio_min=229.9 compile_ms_per_audio_min=16.6 passes_ms_per_audio_min=5581.4 unattributed_ms_per_audio_min=47.0 read_ms=219.4 chunks=597 rungs_per_file_summed=86 windows=742 |
| tuning | min_speech_hops=25 merge_gap_hops=50 hangover_hops=25 cut_s=7 cut_early_s=0.512 frame_rungs=128,256,384,512,640,768 min_chunk_s=0.4 repeat_gram=4 repeat_limit=3 word_repeat_limit=12 density_ceiling=30.0 chunk_tokens_per_s=10.0 min_chunk_tokens=48 speech_spread_db=11.0 filler_words=2 filler_seconds=3.0 |
| speech | files=18 speech_pool_s=3174.8 chunks=552 declared_chunks=547 undeclared_chunks=5 textless_s=66.4 textless_pct=2.09 textless_bar_pct=2.00 textless_gap_pct=0.09 gated=false textless_chunks=10 textless_voiced_s=59.6 textless_other_s=6.8 suppressed_s=0.0 discarded_s=52.6 discarded_pct=1.66 noise_files=15 noise_masked_s=125.1 noise_suppressed_s=36.7 noise_covered_s=121.5 noise_textless_s=121.5 |
| seams | seams=0 seam_words=3 gated=true |
| coverage | masked_s=2868.8 covered_s=3296.2 uncovered_s=1.4 uncovered_pct=0.05 |
| invented | files=15 text_regions=0 chars=0 gated=true |
| refused-example 1 | file=speech/us-gov/speech-us-gov-0147.wav seconds=6.35 text="Right? Nope." |
| refused-example 2 | file=speech/us-gov/speech-us-gov-0210.wav seconds=6.68 text="Thank you." |
| refused-example 3 | file=speech/us-gov/speech-us-gov-0201.wav seconds=6.72 text="Nice." |
| refused-example 4 | file=speech/us-gov/speech-us-gov-0130.wav seconds=6.14 text="No, I." |
| refused-example 5 | file=speech/us-gov/speech-us-gov-0005.wav seconds=6.34 text="Okay." |
| refused-example 6 | file=speech/us-gov/speech-us-gov-0005.wav seconds=6.67 text="Okay." |
| refused-example 7 | file=speech/us-gov/speech-us-gov-0005.wav seconds=6.56 text="Hello." |
| refused-example 8 | file=speech/us-gov/speech-us-gov-0005.wav seconds=7.15 text="Oh yeah! Yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yea" |
| label-rule | offered=68 labels=Speech;Male speech, man speaking;Female speech, woman speaking;Child speech, kid speaking;Conversation;Narration, monologue;Babbling;Children shouting;Whispering;Speech synthesizer;Hubbub, speech noise, speech babble;Singing;Choir;Yodeling;Chant;Mantra;Male singing;Female singing;Child singing;Synthetic singing;Rapping;Humming;Vocal music;A capella;Song;Lullaby;Ambient music;Angry music;Background music;Carnatic music;Christian music;Christmas music;Classical music;Dance music;Electronic dance music;Electronic music;Exciting music;Folk music;Funny music;Gospel music;Happy music;Hip hop music;House music;Independent music;Jingle (music);Middle Eastern music;Music;Music for children;Music of Africa;Music of Asia;Music of Bollywood;Music of Latin America;Musical instrument;New-age music;Pop music;Rock music;Sad music;Salsa music;Scary music;Soul music;Soundtrack music;Swing music;Tender music;Theme music;Traditional music;Trance music;Video game music;Wedding music |
| speech-labels | offered=26 labels=Speech;Male speech, man speaking;Female speech, woman speaking;Child speech, kid speaking;Conversation;Narration, monologue;Babbling;Children shouting;Whispering;Speech synthesizer;Hubbub, speech noise, speech babble;Singing;Choir;Yodeling;Chant;Mantra;Male singing;Female singing;Child singing;Synthetic singing;Rapping;Humming;Vocal music;A capella;Song;Lullaby |
| noise-labels | files=15 labels=Animal:1;Bell:1;Bird:2;Chirp,_tweet:1;Emergency_vehicle:1;Fire_alarm:1;Fireworks:1;Insect:1;Music:4;Police_car_(siren):1;Silence:1;Sine_wave:1;Speech:1;Synthesizer:1;Theremin:1;Walk,_footsteps:1;Whistle:1 |
| long | file=speech/librivox/speech-librivox-0027.wav audio_s=318.0 regions=68 read_ms=20.2 build_ms=728.3 passes_s=35.3 wall_s=37.7 call_ms_per_audio_s=118.4 process=cell moment_peak_bytes=2387755008 |
| memory | process=cell moment_peak_bytes=2387755008 resident=1.27GiB weights=1.33GiB host=427819008 map=1681852576 ceiling_mib=6144 swap_ceiling_mib=1024 |
| long-input | label=1min process=own audio_s=60.0 regions=15 chunks=10 rungs=3 compiles=3 wall_s=8.6 call_ms_per_audio_s=142.8 input_bytes=3840000 rss_before=353026048 rss_after=1561755648 retained=1208729600 process_peak_bytes=2539634688 weights=1.33GiB arena=- |
| long-input | label=30min process=own audio_s=1800.0 regions=458 chunks=308 rungs=6 compiles=6 wall_s=204.0 call_ms_per_audio_s=113.3 input_bytes=115200000 rss_before=353075200 rss_after=252870656 retained=-100204544 process_peak_bytes=2777219072 weights=1.33GiB arena=- |
| long-input | label=60min process=own audio_s=3600.0 regions=740 chunks=574 rungs=6 compiles=6 wall_s=379.5 call_ms_per_audio_s=105.4 input_bytes=230400000 rss_before=353042432 rss_after=362217472 retained=9175040 process_peak_bytes=3015344128 weights=1.33GiB arena=- |

### The pass, phase by phase

One pass, split into what could be attributed to it. These are the parts of the
headline figure and none of them is the recognition number.

| phase | ms per audio second | share of the pass |
|---|---|---|
| read (the file) | 0.1 | 0 % |
| mask and plan | 0.3 | 0 % |
| tagger | 3.8 | 4 % |
| compilations | 0.3 | 0 % |
| recognition passes | 93.0 | 95 % |
| unattributed | 0.8 | 1 % |

The recognition passes are the floor the model imposes: a decode step reads the
body's weights whatever the row count, so what moves the passes is how many steps
the audio needs — fewer chunks, or less of it decoded — not the other phases. This
run's own figures are the row above; a change that raises them says so here rather
than trading them for a better line elsewhere.
The compilations are a **ladder**, not a per-chunk price: a chunk is padded up to one
of 6 closed frame counts and that rung's graphs are compiled once for the process,
so a pass pays for the rungs no earlier call placed. `breakdown`'s
`rungs_per_file_summed` adds each file's distinct rung count — a per-file figure, so
its total over 33 files is larger than the ladder it can only ever draw from, which
the `tuning` row prints whole; `compiles` counts the rungs a call had to build,
`chunks` is the work they price and `windows` the tagger's windows.

### The call's memory, and hour-scale input

The cell's own reading is 2.22 GiB: the high-water of the one process the cell is, taken when it printed its `unit` row, after its corpus pass and the hour-scale children's calls. Its `memory` row, printed after the corpus pass, read 2.22 GiB; the first phase print that carries it is `music` (2.22 GiB) — the phases print their reading in the order they run, so the high-water was reached at or before that phase. `weights` (1.33 GiB) is the device's
shared weight pool — one model's parameter bytes, held once however many layouts and
rungs bind them — and `host` (0.40 GiB) is the process's own kept copies of weights, the
fused projections a second layout re-reads. `host` is printed in whole bytes, like
`map`; `weights` in the display form the `unit` row prints. Both are read at
`music`'s print, the widest of the per-phase readings. `map` (1.57 GiB) is the
bytes of the three files the cell binds — the recognition pair and the tagger — by
their size: each is opened by mapping and its tensor bytes read with `pread` through one
scratch buffer, so the mapping is not the copy the process holds and the figure is not
resident pages. No `arena` figure: each rung's graphs have a plan of
their own, and a sum of plans is not an allocation. Its resident set when the `memory`
row was taken is 1.27 GiB. This cell is **at its ceiling**: the cell's own process and the 7 it started for a call of message or corpus size — cell 2.22 GiB, message 10 s cold run 1 2.36 GiB, message 10 s cold run 2 2.36 GiB, message 10 s cold run 3 2.35 GiB, message 30 s cold run 1 2.35 GiB, message 30 s cold run 2 2.35 GiB, message 30 s cold run 3 2.36 GiB, corpus cold 2.36 GiB — span
2.22 GiB to 2.36 GiB against 2.35 GiB. Its readings fall on both sides of it — 1 below, 7 above — so the row is reported at it. The hour-scale inputs are a population of their own and are not in it: their rows carry them, and the paragraph under their table says what their high-waters do between lengths.
The cell ran under a 6144 MiB resident-set ceiling and a 1024 MiB swap-growth ceiling;
the provenance block says which of those was raised from the guard's default.

Three inputs built from the corpus itself — real speech and real noise, repeated, no
synthetic tone — answer the bound the product needs. Each is one call in a process of
its own, each returns a segmentation covering the whole input, and all three are held
to the same closed ladder: whatever the input's length, no row compiles a shape the
ladder does not hold.

| input | audio s | regions | chunks | rungs | compiles | call ms/audio s | caller MiB | retained MiB | process peak GiB | weights |
|---|---|---|---|---|---|---|---|---|---|---|
| 1min | 60.0 | 15 | 10 | 3 | 3 | 142.8 | 3.7 | 1152.7 | 2.37 | 1.33GiB |
| 30min | 1800.0 | 458 | 308 | 6 | 6 | 113.3 | 109.9 | -95.6 | 2.59 | 1.33GiB |
| 60min | 3600.0 | 740 | 574 | 6 | 6 | 105.4 | 219.7 | 8.8 | 2.81 | 1.33GiB |

`compiles` is what the call asked the compiler for. Each row is a fresh process, so
each places the rungs its own call uses — `compiles` is that call's, not a figure the
earlier rows already paid; within one process a repeat call compiles nothing, which is
what the corpus pass above shows. `rungs` cannot exceed the ladder the `tuning` row
prints, whatever the input's length: the count follows the input up to the ladder's own
size and then stops, and the rows print 3, 6, 6 against a ladder of 6.
`weights` is the device pool each row held: one copy of the same model's parameter
bytes, whatever the length.
`retained` is the difference between two resident-set readings of one process —
before the call, and after it with the caller's audio dropped and the freed pages
returned. At the granularity of a process that has just built and dropped hundreds of
megabytes that difference is not resolvable: it moves between repeats by more than the
effect it would show, so it is printed as a reading rather than as evidence of growth
or of its absence — the three rows read 1152.7 MiB, -95.6 MiB, 8.8 MiB. `process peak GiB` is that process's high-water, and it does not fall with the input here: 2.37 GiB, 2.59 GiB, 2.81 GiB
for the three inputs. The caller's audio (`caller MiB`) and the tagger's log-mel
over it are linear in the input by construction, and they are the part of a call's
working set no implementation can hold in nothing. What the table shows not following
the length is the model side: `rungs` stays inside the closed ladder however long the
input, and `weights` holds one copy of the same parameter bytes on every row.

The `tuning` row is the decision surface this run was taken with — the pause it merges
across, the run it drops, the hangover it reaches into silence, the length every cut aims
at and how far before that aim a cut may search for a pause, the ladder of frame counts a
chunk's graphs are compiled at, and the three text guards — because a choice that is not
printed cannot be reviewed. A cut never moves later than its share, so a span's piece count
follows its length and no piece is longer than the aim plus the search.

The `speech` row answers how much of the speech the segmented path failed to transcribe, in
audio rather than in words. Its denominator is `speech_pool_s` — the speech pool's seconds a
recognition chunk covered, the noise pool counted apart on the same row — and not the seconds
the detector marked: a chunk's hangover reaches into silence, so the covered figure is larger
than `masked_s` in the coverage row, and `covered_s` there is both pools together. All three
are named so none has to be derived from another. `textless_s` is the covered audio whose region came back with no
transcript, `textless_chunks` the passes that produced it, and `discarded_s` the speech whose
transcript the guards refused — not returned at all, because a placeholder over audio nobody
can read is worse than an honest hole: the region is marked found-but-not-read, and the
first `REFUSAL_SAMPLES` of them are printed as `refused-example` rows with their own
audio, so the line between a reading and the model's placeholder is reviewed rather than
taken on trust. A region's
whole seconds are counted, the hangover into silence included. `textless_voiced_s` and
`textless_other_s` split the textless seconds by what the tagger says the audio there is:
the first a label that can hold words, the second a detector mark on audio the tagger calls
something else. `suppressed_s` is the third kind: the marked seconds inside merged spans the
tagger calls something that cannot carry a word, whose audio is their label and which are
not presented to the recogniser at all. Every speech file has its own
`textless-file` row — its `textless_s`, that file's own percentage against the same bar, how
that splits by label, what the labels are, and the call's own `e2e_ms` — so the aggregate is
attributed rather than taken on trust. The noise pool is counted apart, because there the
detector's marks are mostly false positives and a textless second is the detector's, not
lost speech: `noise_masked_s` is what the detector marked, `noise_suppressed_s` the marked
seconds the tagger's labels kept out of the recognition path and `noise_covered_s` what the
recogniser was given. `seams` counts the adjacent transcripts that share `seam_words` whole
words or more — a reading carried twice, which the plan's disjoint chunks rule out, and which
the round is held to at zero. The pool's `invented` row is its other gate: a region over the
noise pool that answered with words at all, with the first three printed so a reading can be
judged rather than counted. The bar is zero, and a run that prints a figure is a run that
failed it.

Two label lists do different work, and both are printed because a decision that is not
printed cannot be reviewed. `label-rule` decides whether a merged span is offered to the
recogniser at all: the tagger's names that can carry words — the speech classes, the singing
ones and the whole music family — matched on the name the file carries. Music and singing
are offered even when no word comes out of them, because a song over an orchestra is exactly
what the song rule must transcribe when its words come out, and an instrumental clip
produces no text at all: offering it costs a pass and invents nothing. `speech-labels` is
the narrower list where the prompt may carry a **declared language** instead of letting the
recogniser identify one: the speech and singing classes only, and only where the chunk's own
loudness varies the way speech does (`SPEECH_SPREAD_DB`, the tenth to the ninetieth percentile
of its 20 ms frame energies). The recogniser's own language answer stops a noisy chunk with
`language None` and no transcript — the failure this surface exists to remove — while a
declared language over an orchestra, or over a steady signal the detector mistook for speech,
produces a confident, wrong reading. So a speech or singing chunk that varies is told the
language, and everything else keeps the prompt the one-shot path uses and the model's own
answer; that gate is what holds the music and noise gates at zero text. `suppressed_s` is what the offering rule kept out of the recognition path — on
the speech pool, on the noise pool (`noise_suppressed_s`) and on the music pool, where the
`music` and `music-report` rows carry it so that a clip with no transcript and suppressed
seconds was never asked rather than misread — and `noise-labels` what the labelled regions
say. The tagger's own arithmetic, as this tree runs it: f16 weights read from the file and
served to f32 multiplies over their widened image — not f16 in compute, which the
cooperative f16 tile cannot do at this model's 252-token window without padding the token
axis and changing the arithmetic.

`parity` reads the pinned clip the listen cell reads — the same file at the same offset and
length, conditioned by the same `prepare_speech` — through the segmented call, whose text
describes the speech spans it detected (`covered_s` of the clip's `clip_s`) and whose
`moved` counts word positions against the pinned transcript. `parity-span` hands the same
detected span to the single-shot walk: that walk keeps the model's own language answer while
the segmented one declares it where the gate allows, so a word that moved between them is what
that difference costs, and the row is a witness rather than a gate. The `parity` row's remainder is the span's:
the recogniser's own cut sensitivity, which moves most word positions when only the second of
non-speech around the speech is added or removed — a reading difference, not lost speech. Both
`moved` figures compare word by position, so one inserted or missing word marks every position
after it.

`boundary` is the binary check on material whose truth is the construction: a speech excerpt
pasted into digital silence at known offsets. `err_ms` is each kept region's start against the
paste it came from, signed; a boundary that conditioning moved would show there. The gain row
repeats the same material as a copy at a level the library leaves alone against the lifted
one: `moved` counts boundaries that shifted, and `identical` says the two readings are the same.

`music` and `music-report` are the music pool: an instrumental clip must come back with no
transcript at all — any text is counted in `instrumental_any_text` and fails the run, because a
reading over music is a wrong answer rather than an observation — and a vocal clip is
transcribed where its words survive the guards. `music-total` is the pool's own call time per
second of audio (`call_ms_per_audio_s`) — the call, not the baseline of a phase.

The `unit` row is this cell's row in the table above, and its two time columns are read as its
introduction says: `cold_ms` is the whole cell, `run_ms` the corpus pass alone, which is what
`s_per_covered_s` re-derives from, and its `peak GiB` column the whole cell's own reading. This
cell's `cold_ms` and `run_ms` are therefore two different spans of the same call and not a phase
and its sum, which is why `s_per_covered_s` is the column to read. The table's `s_per_covered_s` is
therefore per second of *covered audio* while the headline above is per second of *audio*:
two denominators, each named where it is used. Its `build_ms` is the corpus pass's compilations,
the rungs that pass placed, not the whole cell's; `arena` `-` (the ladder's graphs each have a
plan, and a sum of plans is not an allocation) and `digest` `-` because the cell has no pinned
answer of its own.

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- driver: unknown
- profile: release
- code: 3915e22c change-sha256 b592a86c61a31521 (the commit and the SHA-256 of the tree's own difference from it — `MEASUREMENTS.md` excluded, because the run that measures rewrites it, and `crates/backends/rlx-mlx-sys` excluded, because the vendored MLX checkout is not this fork's content — which together are the content that produced these figures; a clean tree's digest is `e3b0c44298fc1c14`, the hash of an empty diff)
- rows: this run measured transcribe, and the transcription section this run wrote is built from its own rows; every other row of the stand table is carried from an earlier run of this file, and its figures belong to the run that took them — the `code` line above is this run's tree, not theirs. A carried row keeps its weight-stream row and, where this run could not read the files, the artefact lines of the run that took it
- thermal: unknown (pmset -g therm printed no CPU limit line) (`pmset -g therm`, the reading `scripts/check-throttle.sh` gates on)
- load: 2.46 3.00 3.10 (1/5/15-min) at the start, 3.10 2.97 2.85 at the table; every cell line carries its own `load=`
- ceiling: 6144 MiB resident set, the guard's default
- swap: 4503.8 MiB in use at the start, 4487.8 MiB at the table — 16.0 MiB, given back during the run (ceiling 1024 MiB growth, the guard's default)
- command: `mahbot-inference 8` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`, article from `MAHBOT_EMBED_TEXT`)
- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- model: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- model: `ced-small-f16.gguf` · 43707680 bytes · sha256 c391ed8697a1b08d7c1a463e4940a5c3a2f670e0544ab0d8ee23b544583602a8 · mudler/ced-gguf@b5e9a4aad6438763c8da16079d77563fbed35c65

> **The music pool's provenance.** instrumental/01-orchestra.wav — United States Navy Band, La Bayamesa: public domain; must not be transcribed. instrumental/02-solo-piano.wav — Gershwin, Rhapsody in Blue piano solo: public domain; must not be transcribed. instrumental/03-electronic.wav — Loyalty Freak Music, High Technologic Beat Explosion: CC0 1.0; must not be transcribed. instrumental/04-ambient.wav — Wilfredor, Ambient music test (Yamaha CK61): CC0 1.0; must not be transcribed. vocal/01-acappella-solfege.wav — anonymous, Star in the east solfege: CC BY 2.5, https://creativecommons.org/licenses/by/2.5/; shape-note solfège singing, no transcript required and its absence is not a miss, the tagger should answer singing/choir/a capella rather than an instrument. vocal/02-pop-lyrics.wav — Irving Berlin, Billy Murray (Edison), I Want to Go Back to Michigan: public domain; clear English lyrics, expected to be transcribed. vocal/03-vocal-over-band.wav — Norworth and Bayes, Ada Jones and Billy Murray, Shine On, Harvest Moon: public domain; voice over an orchestra, borderline — record which way it went. vocal/04-vocal-indian-classical.wav — Master Madan, Yun Na Reh Reh Kar Hamein Tarsaiye: public domain; vocal with tabla, not English, no text expectation — record what comes out.
<!-- END transcribe -->
