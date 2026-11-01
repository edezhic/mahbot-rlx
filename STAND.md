# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.41s | 1.37s | 3.58GiB | 4.26GiB |
| recognition, typical voice message (30 s) | 5.14s | 3.78s | 3.66GiB | 4.20GiB |
| recognition, long voice message (the whole pinned recording) | 38.80s | 37.78s | 3.84GiB | 4.21GiB |
| synthesis, short text | 7.11s | 1.13s | 4.17GiB | 4.30GiB |
| synthesis, typical text | 10.10s | 4.14s | 6.61GiB | 6.72GiB |
| wake word, two scoring windows in one pass | 645.9ms | 32.0ms | 848.0MiB | 1.18GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-01 08:59 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: bd158331 change-sha256 ebb7522eeb59963c (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- run form: RLX_ALLOW_THROTTLE=1
- models: `~/.mahbot/userspaces/admin/models`
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav`
- corpus: `~/.mahbot/faph_corpus`
