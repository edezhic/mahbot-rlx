# Transcription corpus

| recording | audio s | s per audio s | resting | peak |
|---|---|---|---|---|
| `speech/librivox/speech-librivox-0027.wav` | 318.0 | 0.117 | 2.97GiB | 3.88GiB |
| `speech/librivox/speech-librivox-0051.wav` | 315.6 | 0.118 | 2.89GiB | 3.94GiB |
| `speech/librivox/speech-librivox-0093.wav` | 314.1 | 0.111 | 2.97GiB | 3.92GiB |
| `speech/librivox/speech-librivox-0075.wav` | 312.8 | 0.093 | 2.97GiB | 3.90GiB |
| `speech/librivox/speech-librivox-0103.wav` | 307.2 | 0.119 | 2.97GiB | 3.88GiB |
| `speech/librivox/speech-librivox-0088.wav` | 304.2 | 0.117 | 2.97GiB | 3.88GiB |
| `speech/us-gov/speech-us-gov-0252.wav` | 231.5 | 0.109 | 2.96GiB | 3.88GiB |
| `speech/us-gov/speech-us-gov-0147.wav` | 215.9 | 0.087 | 2.97GiB | 3.92GiB |
| `speech/us-gov/speech-us-gov-0082.wav` | 215.0 | 0.117 | 2.79GiB | 3.96GiB |
| `speech/us-gov/speech-us-gov-0159.wav` | 173.9 | 0.093 | 3.18GiB | 3.96GiB |
| `speech/us-gov/speech-us-gov-0067.wav` | 157.8 | 0.110 | 2.95GiB | 3.86GiB |
| `speech/us-gov/speech-us-gov-0250.wav` | 126.1 | 0.111 | 2.89GiB | 3.86GiB |
| `speech/us-gov/speech-us-gov-0210.wav` | 123.8 | 0.087 | 3.02GiB | 3.92GiB |
| `speech/us-gov/speech-us-gov-0229.wav` | 100.5 | 0.103 | 2.95GiB | 3.89GiB |
| `speech/us-gov/speech-us-gov-0201.wav` | 58.0 | 0.087 | 2.78GiB | 3.95GiB |
| `speech/us-gov/speech-us-gov-0130.wav` | 53.3 | 0.097 | 2.78GiB | 3.95GiB |
| `speech/us-gov/speech-us-gov-0005.wav` | 26.8 | 0.109 | 2.48GiB | 3.95GiB |
| `speech/us-gov/speech-us-gov-0109.wav` | 19.7 | 0.109 | 2.78GiB | 3.95GiB |
| `noise/free-sound/noise-free-sound-0640.wav` | 32.8 | 0.070 | 2.76GiB | 3.87GiB |
| `noise/free-sound/noise-free-sound-0184.wav` | 32.4 | 0.004 | 412.2MiB | 495.1MiB |
| `noise/free-sound/noise-free-sound-0761.wav` | 32.4 | 0.066 | 2.79GiB | 3.95GiB |
| `noise/free-sound/noise-free-sound-0012.wav` | 32.2 | 0.059 | 2.69GiB | 3.85GiB |
| `noise/free-sound/noise-free-sound-0605.wav` | 32.0 | 0.073 | 2.70GiB | 3.85GiB |
| `noise/free-sound/noise-free-sound-0298.wav` | 32.0 | 0.004 | 413.1MiB | 496.0MiB |
| `noise/free-sound/noise-free-sound-0251.wav` | 32.0 | 0.035 | 2.77GiB | 3.87GiB |
| `noise/free-sound/noise-free-sound-0733.wav` | 31.9 | 0.004 | 413.2MiB | 496.1MiB |
| `noise/free-sound/noise-free-sound-0035.wav` | 31.9 | 0.013 | 2.73GiB | 3.85GiB |
| `noise/free-sound/noise-free-sound-0709.wav` | 31.9 | 0.032 | 2.70GiB | 3.85GiB |
| `noise/sound-bible/noise-sound-bible-0075.wav` | 32.7 | 0.004 | 412.3MiB | 495.2MiB |
| `noise/sound-bible/noise-sound-bible-0044.wav` | 30.0 | 0.004 | 411.2MiB | 494.2MiB |
| `noise/sound-bible/noise-sound-bible-0067.wav` | 27.7 | 0.024 | 2.76GiB | 3.87GiB |
| `noise/sound-bible/noise-sound-bible-0014.wav` | 27.5 | 0.015 | 2.73GiB | 3.85GiB |
| `noise/sound-bible/noise-sound-bible-0032.wav` | 25.2 | 0.064 | 2.59GiB | 3.85GiB |

The pinned real-audio corpus, one call per recording, each measured in a process of its own. Time per second of audio is the warm call — the models already resident — divided by the recording's own length. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-03 08:13 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: 8fe9ce46 change-sha256 43f7267d9437f2ac (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference --features bench --bin transcription-corpus
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
