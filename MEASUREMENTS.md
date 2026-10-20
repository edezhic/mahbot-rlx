# Measurements

The latest numbers from this crate's two measuring surfaces. Each section is
replaced by the run that produced it and carries its own date; the two surfaces
are run at different times, so one half may be older than the other. This file
is not a log — history is what version control is for.

<!-- BEGIN stand -->
## Per-model measurement (stand)

_Measured 2026-10-20 15:20 UTC._

| unit | cold ms | warm ms | speech s | s/speech s | loop ms | sound ms | build ms | peak | map | arena | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | 73.0 | 34.2 | - | - | - | - | 185.9 | 0.88 GiB | 1.53 GiB | 3.4MiB | 92cc8b5117cffde0 |
| ear.075 | 71.7 | 34.8 | - | - | - | - | 185.4 | 0.84 GiB | 1.53 GiB | 3.4MiB | 569e7385523d15e7 |
| ear.batch2 | 74.3 | 40.3 | - | - | - | - | 203.4 | 0.84 GiB | 1.53 GiB | 3.6MiB | aac8192598768f05 |
| ear.batch4 | 89.5 | 51.6 | - | - | - | - | 236.9 | 0.84 GiB | 1.53 GiB | 3.9MiB | 4ea8fb8b0c7196c5 |
| listen | 2687.2 | 2487.5 | 17.460 | 0.142 | - | - | 1178.2 | 2.31 GiB | 1.53 GiB | 1.10GiB | 2f93dfa1d7cdeeb1 |
| speak | 30602.8 | 29208.7 | 26.400 | 1.106 | 26858.8 | 2349.9 | 1521.1 | 2.42 GiB | 1.38 GiB | 724.6MiB | bae7ba022d335038 |
| embed | 51254.4 | 51125.1 | - | - | - | - | 53149.0 | 3.05 GiB | 0.37 GiB | 670.2MiB | 17a3b918d44d2849 |

The cell's number is the **`s/speech s`** column: the **warm** figure (`warm ms`) divided by the `speech s` column — the cell's own seconds of speech, `samples / 16 000` for recognition and `rows · 1 920 / 24 000` for synthesis, printed so the rate re-derives from this table alone. `loop ms` is the synthesis generation loop (speaker, prefill, body, predictor, sampler) and `sound ms` is the sound half; their sum is `warm ms`, so a ×N synthesis claim reads off both. Both are the same phase sums the cell's `loop_ms`/`sound_ms` tokens record; `speak`'s `sound_phase_ms` field is the sound graphs' summed `step_ms` — the same sum the `sound ms` column is taken from, carried as a field so a cell line is self-describing. For `speak` the recorded `digest` is the **waveform** digest, not the `prompt_digest` its own line prints. `-` is a cell with no speech denominator. Each cell line prints its own `warm_spread` (the warm runs' max − min, in ms).

### Weight streams, against the machine's streaming ceiling

Ceiling, measured on this run's device before the cells: **130.2 GB/s** on a streaming read (129.7–132.0 over 5 rounds) and 127.5 GB/s of total traffic on a streaming copy (126.0–128.1). The read figure is the one the rows below are reported against, because a weight is read and never written; the copy figure is the same probe's other arm, recorded beside it. Both come from `rlx_wgpu::bandwidth::stream_ceiling`: a 64.0 MiB vectorised read and a 64.0 MiB vectorised copy, on `Apple M3 Pro (Metal)`. The same probe after the cells read 130.4 GB/s (129.9–131.4), which is the reading-to-reading spread to hold the figure to.

| phase | unit | weight bytes/unit | ms/unit | GB/s | of the read ceiling |
|---|---|---|---|---|---|
| recognition decode | one token | 1 100 961 792 | 26.27 | 41.9 | 32.2 % |
| talker body | one frame | 850 870 272 | 23.27 | 36.6 | 28.1 % |
| code predictor | one level | 87 875 584 | 3.61 | 24.3 | 18.7 % |
| sound half | one utterance | 1 414 403 200 | 2349.90 | 0.6 | 0.5 % |

Each row's numerator is the graph's own parameter bytes — packed and f32 — which its multiplies and convolutions read once per unit, and only operands that are parameters count; a graph that read one of them twice per unit would move more, and one that read a slice would move less. `talker body` is one frame, `code predictor` one of a frame's sixteen projecting levels (the frame's stream and time divided by the levels it walks), `sound half` the utterance's windows summed with the graph each window actually ran, and `recognition decode` one generated token. The figure in each row is the run whose warm figure is the table's.

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: e3488352 change-sha256 406ee0f2e2452f73 (the commit and the SHA-256 of the tree's own difference from it — `MEASUREMENTS.md` excluded, because the run that measures rewrites it, and `crates/backends/rlx-mlx-sys` excluded, because the vendored MLX checkout is not this fork's content — which together are the content that produced these figures; a clean tree's digest is `e3b0c44298fc1c14`, the hash of an empty diff)
- thermal: unknown (pmset -g therm printed no CPU limit line) (`pmset -g therm`, the reading `scripts/check-throttle.sh` gates on)
- load: 3.00 4.31 5.23 (1/5/15-min) at the start, 3.80 4.09 4.79 at the table; every cell line carries its own `load=`
- swap: 7969 MiB in use at the start, 229 MiB grown during the run (ceiling 4096 MiB, raised by `MAHBOT_STAND_MAX_SWAP_MIB` from its default 1024)
- command: `cargo run --release -p mahbot-inference` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`)
- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- model: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- model: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- model: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
<!-- END stand -->

## What this record is

One stand run of the tree the provenance block names, on `Apple M3 Pro (Metal)`,
with `RLX_WGPU_DBG_COUNTERS` and `RLX_WGPU_SCHEDULE` unset and
`RLX_WGPU_ENCODE_CHUNK` at its default 64 steps — the scheduling shape the product
path walks, and the one these figures belong to.

- `listen` — 2 487.5 ms warm, 0.142 s per second of speech, answer digest
  `2f93dfa1d7cdeeb1` in all three runs.
- `speak` — 29 208.7 ms warm = 26 858.8 loop + 2 349.9 sound, 1.106 s per second
  of speech, waveform digest `bae7ba022d335038` in all three runs.
- `embed` — 51 125.1 ms warm, no speech denominator.

Each row of the weight-stream table above divides that graph's own parameter bytes
by the ms the same run took for one unit; the percentage is that quotient over the
read ceiling the same run measured before the cells. `peak` is the run's high-water
RSS and `build ms` its compile time per cell: both moved against the committed
record's, the build to 1 178.2, 1 521.1 and 53 149.0 ms for the three surfaces,
and neither is adjudicated further here.

`139 GB/s` is a separate ad-hoc sweep of this machine (median of nine rounds over
six slice shapes, at wall clock over encode + submit + poll), recorded once here
and not re-taken by this run: the machine-level read rate, named as its own
measurement and not any row's denominator.

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
