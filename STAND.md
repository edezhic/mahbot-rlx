# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.60s | 1.37s | 3.58GiB | 4.27GiB |
| recognition, typical voice message (30 s) | 5.17s | 3.83s | 3.66GiB | 4.20GiB |
| recognition, long voice message (the whole pinned recording) | 40.06s | 38.03s | 3.89GiB | 4.21GiB |
| synthesis, short text | 6.96s | 1.33s | 3.24GiB | 4.17GiB |
| synthesis, typical text | 9.82s | 4.28s | 3.59GiB | 4.60GiB |
| wake word, two scoring windows in one pass | 568.7ms | 30.6ms | 848.1MiB | 1.18GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-01 12:03 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: ad411a3a change-sha256 b747b6b2681680a3 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- run form: RLX_ALLOW_THROTTLE=1
- models: `~/.mahbot/userspaces/admin/models`
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav`
- corpus: `~/.mahbot/faph_corpus`
