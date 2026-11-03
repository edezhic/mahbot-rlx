# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.45s | 1.35s | 3.44GiB | 4.12GiB |
| recognition, typical voice message (30 s) | 5.08s | 3.69s | 3.51GiB | 4.07GiB |
| recognition, long voice message (the whole pinned recording) | 38.36s | 36.94s | 3.67GiB | 4.09GiB |
| synthesis, short text | 2.52s | 1.29s | 2.68GiB | 3.48GiB |
| synthesis, typical text | 5.28s | 4.36s | 3.07GiB | 3.99GiB |
| wake word, two scoring windows in one pass | 527.4ms | 32.1ms | 809.1MiB | 1.12GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-02 23:09 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: d2750edb change-sha256 889347fba2202e3e (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
