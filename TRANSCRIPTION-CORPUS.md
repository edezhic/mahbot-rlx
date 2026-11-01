# Transcription corpus

| recording | audio s | s per audio s | resting | peak |
|---|---|---|---|---|
| `speech/librivox/speech-librivox-0027.wav` | 318.0 | 0.119 | 3.84GiB | 4.21GiB |
| `speech/librivox/speech-librivox-0051.wav` | 315.6 | 0.119 | 3.76GiB | 4.30GiB |
| `speech/librivox/speech-librivox-0093.wav` | 314.1 | 0.113 | 3.84GiB | 4.27GiB |
| `speech/librivox/speech-librivox-0075.wav` | 312.8 | 0.094 | 3.84GiB | 4.24GiB |
| `speech/librivox/speech-librivox-0103.wav` | 307.2 | 0.120 | 3.84GiB | 4.21GiB |
| `speech/librivox/speech-librivox-0088.wav` | 304.2 | 0.119 | 3.84GiB | 4.21GiB |
| `speech/us-gov/speech-us-gov-0252.wav` | 231.5 | 0.107 | 3.83GiB | 4.22GiB |
| `speech/us-gov/speech-us-gov-0147.wav` | 215.9 | 0.090 | 3.75GiB | 4.28GiB |
| `speech/us-gov/speech-us-gov-0082.wav` | 215.0 | 0.120 | 3.66GiB | 4.32GiB |
| `speech/us-gov/speech-us-gov-0159.wav` | 173.9 | 0.095 | 4.05GiB | 4.32GiB |
| `speech/us-gov/speech-us-gov-0067.wav` | 157.8 | 0.113 | 3.82GiB | 4.19GiB |
| `speech/us-gov/speech-us-gov-0250.wav` | 126.1 | 0.115 | 3.74GiB | 4.19GiB |
| `speech/us-gov/speech-us-gov-0210.wav` | 123.8 | 0.089 | 4.13GiB | 4.28GiB |
| `speech/us-gov/speech-us-gov-0229.wav` | 100.5 | 0.107 | 3.82GiB | 4.24GiB |
| `speech/us-gov/speech-us-gov-0201.wav` | 58.0 | 0.090 | 3.65GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0130.wav` | 53.3 | 0.102 | 3.65GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0005.wav` | 26.8 | 0.117 | 3.12GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0109.wav` | 19.7 | 0.119 | 3.65GiB | 4.31GiB |
| `noise/free-sound/noise-free-sound-0640.wav` | 32.8 | 0.080 | 3.60GiB | 4.21GiB |
| `noise/free-sound/noise-free-sound-0184.wav` | 32.4 | 0.004 | 455.0MiB | 579.3MiB |
| `noise/free-sound/noise-free-sound-0761.wav` | 32.4 | 0.073 | 3.65GiB | 4.31GiB |
| `noise/free-sound/noise-free-sound-0012.wav` | 32.2 | 0.070 | 3.51GiB | 4.18GiB |
| `noise/free-sound/noise-free-sound-0605.wav` | 32.0 | 0.083 | 3.51GiB | 4.18GiB |
| `noise/free-sound/noise-free-sound-0298.wav` | 32.0 | 0.004 | 455.1MiB | 579.4MiB |
| `noise/free-sound/noise-free-sound-0251.wav` | 32.0 | 0.042 | 3.63GiB | 4.21GiB |
| `noise/free-sound/noise-free-sound-0733.wav` | 31.9 | 0.004 | 455.0MiB | 579.4MiB |
| `noise/free-sound/noise-free-sound-0035.wav` | 31.9 | 0.017 | 3.43GiB | 4.18GiB |
| `noise/free-sound/noise-free-sound-0709.wav` | 31.9 | 0.041 | 3.51GiB | 4.18GiB |
| `noise/sound-bible/noise-sound-bible-0075.wav` | 32.7 | 0.004 | 455.0MiB | 579.3MiB |
| `noise/sound-bible/noise-sound-bible-0044.wav` | 30.0 | 0.004 | 455.0MiB | 579.3MiB |
| `noise/sound-bible/noise-sound-bible-0067.wav` | 27.7 | 0.028 | 3.48GiB | 4.21GiB |
| `noise/sound-bible/noise-sound-bible-0014.wav` | 27.5 | 0.019 | 3.44GiB | 4.18GiB |
| `noise/sound-bible/noise-sound-bible-0032.wav` | 25.2 | 0.074 | 3.40GiB | 4.18GiB |

The pinned real-audio corpus, one call per recording, each measured in a process of its own. Time per second of audio is the warm call — the models already resident — divided by the recording's own length. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-01 09:12 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: bd158331 change-sha256 ebb7522eeb59963c (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference --features bench --bin transcription-corpus
- run form: RLX_ALLOW_THROTTLE=1
- corpus: `~/.mahbot/faph_corpus`
