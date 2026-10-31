# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.81s | 1.13s | 3.59GiB | 4.26GiB |
| recognition, typical voice message (30 s) | 5.37s | 3.26s | 4.51GiB | 4.59GiB |
| recognition, long voice message (the whole pinned recording) | 39.62s | 36.37s | 5.74GiB | 6.15GiB |
| synthesis, short text | 7.03s | 1.11s | 4.17GiB | 4.30GiB |
| synthesis, typical text | 10.08s | 4.08s | 6.63GiB | 6.75GiB |
| wake word, one scoring step | 639.3ms | 27.7ms | 826.6MiB | 1.16GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has loaded and built what it needs and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-10-31 19:58 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 6aa0afa2 change-sha256 4c14684345ff5529 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `~/.mahbot/userspaces/admin/models`
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav`
- corpus: `~/.mahbot/faph_corpus`
