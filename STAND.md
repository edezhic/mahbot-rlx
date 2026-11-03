# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.61s | 1.25s | 2.83GiB | 3.91GiB |
| recognition, typical voice message (30 s) | 4.65s | 3.43s | 2.81GiB | 3.86GiB |
| recognition, long voice message (the whole pinned recording) | 37.91s | 36.89s | 2.97GiB | 3.88GiB |
| synthesis, short text | 2.34s | 1.21s | 2.43GiB | 3.09GiB |
| synthesis, typical text | 5.12s | 4.28s | 2.67GiB | 3.49GiB |
| wake word, two scoring windows in one pass | 516.9ms | 32.4ms | 757.1MiB | 1.07GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-03 07:59 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 8fe9ce46 change-sha256 43f7267d9437f2ac (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
