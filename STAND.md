# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.62s | 1.25s | 2.84GiB | 3.91GiB |
| recognition, typical voice message (30 s) | 4.70s | 3.42s | 2.81GiB | 3.87GiB |
| recognition, long voice message (the whole pinned recording) | 37.82s | 36.93s | 2.97GiB | 3.88GiB |
| synthesis, short text | 2.32s | 1.21s | 2.29GiB | 2.74GiB |
| synthesis, typical text | 5.11s | 4.30s | 2.54GiB | 3.03GiB |
| wake word, two scoring windows in one pass | 514.1ms | 31.9ms | 755.1MiB | 1.07GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-03 15:25 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 6f816ed1 change-sha256 21b4110842c37379 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
