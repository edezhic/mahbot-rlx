# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.37s | 1.14s | 3.59GiB | 4.26GiB |
| recognition, typical voice message (30 s) | 5.16s | 3.27s | 4.50GiB | 4.58GiB |
| recognition, long voice message (the whole pinned recording) | 39.30s | 36.68s | 6.08GiB | 6.16GiB |
| synthesis, short text | 6.98s | 1.12s | 4.17GiB | 4.31GiB |
| synthesis, typical text | 9.98s | 4.32s | 6.64GiB | 6.75GiB |
| wake word, two scoring windows in one pass | 653.0ms | 34.5ms | 847.6MiB | 1.18GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has loaded and built what it needs and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-01 01:11 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 51d263cc change-sha256 a4be85772af621d6 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `~/.mahbot/userspaces/admin/models`
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav`
- corpus: `~/.mahbot/faph_corpus`
