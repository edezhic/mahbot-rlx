# Transcription corpus

| recording | audio s | s per audio s | resting | peak |
|---|---|---|---|---|
| `speech/librivox/speech-librivox-0027.wav` | 318.0 | 0.116 | 6.07GiB | 6.16GiB |
| `speech/librivox/speech-librivox-0051.wav` | 315.6 | 0.118 | 4.73GiB | 4.81GiB |
| `speech/librivox/speech-librivox-0093.wav` | 314.1 | 0.111 | 5.22GiB | 5.29GiB |
| `speech/librivox/speech-librivox-0075.wav` | 312.8 | 0.090 | 5.66GiB | 5.75GiB |
| `speech/librivox/speech-librivox-0103.wav` | 307.2 | 0.116 | 6.08GiB | 6.16GiB |
| `speech/librivox/speech-librivox-0088.wav` | 304.2 | 0.115 | 6.08GiB | 6.16GiB |
| `speech/us-gov/speech-us-gov-0252.wav` | 231.5 | 0.101 | 5.66GiB | 5.74GiB |
| `speech/us-gov/speech-us-gov-0147.wav` | 215.9 | 0.088 | 4.71GiB | 4.80GiB |
| `speech/us-gov/speech-us-gov-0082.wav` | 215.0 | 0.120 | 4.17GiB | 4.32GiB |
| `speech/us-gov/speech-us-gov-0159.wav` | 173.9 | 0.092 | 4.73GiB | 4.82GiB |
| `speech/us-gov/speech-us-gov-0067.wav` | 157.8 | 0.110 | 5.13GiB | 5.21GiB |
| `speech/us-gov/speech-us-gov-0250.wav` | 126.1 | 0.110 | 4.96GiB | 5.04GiB |
| `speech/us-gov/speech-us-gov-0210.wav` | 123.8 | 0.085 | 5.11GiB | 5.21GiB |
| `speech/us-gov/speech-us-gov-0229.wav` | 100.5 | 0.101 | 5.20GiB | 5.28GiB |
| `speech/us-gov/speech-us-gov-0201.wav` | 58.0 | 0.084 | 4.17GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0130.wav` | 53.3 | 0.096 | 4.16GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0005.wav` | 26.8 | 0.104 | 4.16GiB | 4.31GiB |
| `speech/us-gov/speech-us-gov-0109.wav` | 19.7 | 0.102 | 4.16GiB | 4.31GiB |
| `noise/free-sound/noise-free-sound-0640.wav` | 32.8 | 0.064 | 4.38GiB | 4.45GiB |
| `noise/free-sound/noise-free-sound-0184.wav` | 32.4 | 0.004 | 454.9MiB | 579.2MiB |
| `noise/free-sound/noise-free-sound-0761.wav` | 32.4 | 0.062 | 4.16GiB | 4.31GiB |
| `noise/free-sound/noise-free-sound-0012.wav` | 32.2 | 0.054 | 4.23GiB | 4.30GiB |
| `noise/free-sound/noise-free-sound-0605.wav` | 32.0 | 0.068 | 4.24GiB | 4.30GiB |
| `noise/free-sound/noise-free-sound-0298.wav` | 32.0 | 0.004 | 454.9MiB | 579.3MiB |
| `noise/free-sound/noise-free-sound-0251.wav` | 32.0 | 0.030 | 4.00GiB | 4.21GiB |
| `noise/free-sound/noise-free-sound-0733.wav` | 31.9 | 0.004 | 454.9MiB | 579.3MiB |
| `noise/free-sound/noise-free-sound-0035.wav` | 31.9 | 0.010 | 3.45GiB | 4.18GiB |
| `noise/free-sound/noise-free-sound-0709.wav` | 31.9 | 0.025 | 4.23GiB | 4.30GiB |
| `noise/sound-bible/noise-sound-bible-0075.wav` | 32.7 | 0.004 | 454.8MiB | 579.2MiB |
| `noise/sound-bible/noise-sound-bible-0044.wav` | 30.0 | 0.004 | 454.8MiB | 579.1MiB |
| `noise/sound-bible/noise-sound-bible-0067.wav` | 27.7 | 0.020 | 3.50GiB | 4.21GiB |
| `noise/sound-bible/noise-sound-bible-0014.wav` | 27.5 | 0.011 | 3.45GiB | 4.18GiB |
| `noise/sound-bible/noise-sound-bible-0032.wav` | 25.2 | 0.061 | 3.74GiB | 4.18GiB |

The pinned real-audio corpus, one call per recording, each measured in a process of its own. Time per second of audio is the warm call — the models already resident — divided by the recording's own length. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has loaded and built what it needs and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-01 06:52 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 033bc236 change-sha256 48cf4514e1bbc8d7 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference --features bench --bin transcription-corpus
- run form: RLX_ALLOW_THROTTLE=1
- corpus: `~/.mahbot/faph_corpus`
