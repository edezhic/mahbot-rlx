# Stand

| row | cold | warm | resting | peak |
|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.56s | 1.16s | 3.61GiB | 4.29GiB |
| recognition, typical voice message (30 s) | 5.46s | 3.33s | 4.54GiB | 4.67GiB |
| recognition, long voice message (the whole pinned recording) | 40.19s | 37.21s | 6.13GiB | 6.26GiB |
| synthesis, short text | 7.55s | 1.18s | 4.14GiB | 4.27GiB |
| synthesis, typical text | 10.78s | 4.14s | 6.63GiB | 6.76GiB |
| wake word, one scoring step | 665.3ms | 31.2ms | 843.0MiB | 1.18GiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has loaded and built what it needs and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-10-31 13:52 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: e199dfd4 change-sha256 768d65d2becdb727 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `~/.mahbot/userspaces/admin/models`
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav`
- corpus: `~/.mahbot/faph_corpus`
