# Transcription corpus

| recording | audio s | s per audio s | resting | peak |
|---|---|---|---|---|
| `speech/librivox/speech-librivox-0027.wav` | 318.0 | 0.116 | 6.18GiB | 6.25GiB |
| `speech/librivox/speech-librivox-0051.wav` | 315.6 | 0.129 | 4.78GiB | 4.91GiB |
| `speech/librivox/speech-librivox-0093.wav` | 314.1 | 0.131 | 4.69GiB | 5.44GiB |
| `speech/librivox/speech-librivox-0075.wav` | 312.8 | 0.100 | 5.76GiB | 5.87GiB |
| `speech/librivox/speech-librivox-0103.wav` | 307.2 | 0.119 | 5.60GiB | 6.16GiB |
| `speech/librivox/speech-librivox-0088.wav` | 304.2 | 0.118 | 5.55GiB | 6.34GiB |
| `speech/us-gov/speech-us-gov-0252.wav` | 231.5 | 0.106 | 5.71GiB | 5.83GiB |
| `speech/us-gov/speech-us-gov-0147.wav` | 215.9 | 0.091 | 4.79GiB | 4.92GiB |
| `speech/us-gov/speech-us-gov-0082.wav` | 215.0 | 0.123 | 4.23GiB | 4.38GiB |
| `speech/us-gov/speech-us-gov-0159.wav` | 173.9 | 0.094 | 4.66GiB | 4.76GiB |
| `speech/us-gov/speech-us-gov-0067.wav` | 157.8 | 0.114 | 5.18GiB | 5.31GiB |
| `speech/us-gov/speech-us-gov-0250.wav` | 126.1 | 0.120 | 5.01GiB | 5.14GiB |
| `speech/us-gov/speech-us-gov-0210.wav` | 123.8 | 0.087 | 4.64GiB | 4.74GiB |
| `speech/us-gov/speech-us-gov-0229.wav` | 100.5 | 0.106 | 5.31GiB | 5.39GiB |
| `speech/us-gov/speech-us-gov-0201.wav` | 58.0 | 0.087 | 4.22GiB | 4.37GiB |
| `speech/us-gov/speech-us-gov-0130.wav` | 53.3 | 0.100 | 4.21GiB | 4.36GiB |
| `speech/us-gov/speech-us-gov-0005.wav` | 26.8 | 0.122 | 4.22GiB | 4.36GiB |
| `speech/us-gov/speech-us-gov-0109.wav` | 19.7 | 0.121 | 4.21GiB | 4.36GiB |
| `noise/free-sound/noise-free-sound-0640.wav` | 32.8 | 0.066 | 4.45GiB | 4.58GiB |
| `noise/free-sound/noise-free-sound-0184.wav` | 32.4 | 0.004 | 454.9MiB | 579.2MiB |
| `noise/free-sound/noise-free-sound-0761.wav` | 32.4 | 0.063 | 4.20GiB | 4.35GiB |
| `noise/free-sound/noise-free-sound-0012.wav` | 32.2 | 0.055 | 4.29GiB | 4.45GiB |
| `noise/free-sound/noise-free-sound-0605.wav` | 32.0 | 0.069 | 4.30GiB | 4.46GiB |
| `noise/free-sound/noise-free-sound-0298.wav` | 32.0 | 0.004 | 455.0MiB | 579.4MiB |
| `noise/free-sound/noise-free-sound-0251.wav` | 32.0 | 0.032 | 4.06GiB | 4.23GiB |
| `noise/free-sound/noise-free-sound-0733.wav` | 31.9 | 0.004 | 457.4MiB | 581.8MiB |
| `noise/free-sound/noise-free-sound-0035.wav` | 31.9 | 0.010 | 3.47GiB | 4.20GiB |
| `noise/free-sound/noise-free-sound-0709.wav` | 31.9 | 0.027 | 4.29GiB | 4.45GiB |
| `noise/sound-bible/noise-sound-bible-0075.wav` | 32.7 | 0.004 | 455.0MiB | 579.3MiB |
| `noise/sound-bible/noise-sound-bible-0044.wav` | 30.0 | 0.004 | 454.5MiB | 578.9MiB |
| `noise/sound-bible/noise-sound-bible-0067.wav` | 27.7 | 0.022 | 3.56GiB | 4.23GiB |
| `noise/sound-bible/noise-sound-bible-0014.wav` | 27.5 | 0.011 | 3.47GiB | 4.20GiB |
| `noise/sound-bible/noise-sound-bible-0032.wav` | 25.2 | 0.068 | 3.79GiB | 4.20GiB |

The pinned real-audio corpus, one call per recording, each measured in a process of its own. Time per second of audio is the warm call — the models already resident — divided by the recording's own length. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has loaded and built what it needs and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-10-31 14:06 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: e199dfd4 change-sha256 768d65d2becdb727 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference --features bench --bin transcription-corpus
- corpus: `~/.mahbot/faph_corpus`
