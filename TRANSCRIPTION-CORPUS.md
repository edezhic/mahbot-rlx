# Transcription corpus

| recording | audio s | s per audio s | resting | peak |
|---|---|---|---|---|
| `speech/librivox/speech-librivox-0027.wav` | 318.0 | 0.118 | 3.66GiB | 4.09GiB |
| `speech/librivox/speech-librivox-0051.wav` | 315.6 | 0.118 | 3.59GiB | 4.15GiB |
| `speech/librivox/speech-librivox-0093.wav` | 314.1 | 0.112 | 3.67GiB | 4.13GiB |
| `speech/librivox/speech-librivox-0075.wav` | 312.8 | 0.093 | 3.67GiB | 4.11GiB |
| `speech/librivox/speech-librivox-0103.wav` | 307.2 | 0.119 | 3.66GiB | 4.09GiB |
| `speech/librivox/speech-librivox-0088.wav` | 304.2 | 0.118 | 3.67GiB | 4.09GiB |
| `speech/us-gov/speech-us-gov-0252.wav` | 231.5 | 0.106 | 3.65GiB | 4.09GiB |
| `speech/us-gov/speech-us-gov-0147.wav` | 215.9 | 0.089 | 3.58GiB | 4.13GiB |
| `speech/us-gov/speech-us-gov-0082.wav` | 215.0 | 0.120 | 3.49GiB | 4.17GiB |
| `speech/us-gov/speech-us-gov-0159.wav` | 173.9 | 0.094 | 3.87GiB | 4.17GiB |
| `speech/us-gov/speech-us-gov-0067.wav` | 157.8 | 0.112 | 3.64GiB | 4.07GiB |
| `speech/us-gov/speech-us-gov-0250.wav` | 126.1 | 0.114 | 3.58GiB | 4.08GiB |
| `speech/us-gov/speech-us-gov-0210.wav` | 123.8 | 0.088 | 3.94GiB | 4.13GiB |
| `speech/us-gov/speech-us-gov-0229.wav` | 100.5 | 0.106 | 3.65GiB | 4.10GiB |
| `speech/us-gov/speech-us-gov-0201.wav` | 58.0 | 0.089 | 3.48GiB | 4.16GiB |
| `speech/us-gov/speech-us-gov-0130.wav` | 53.3 | 0.101 | 3.48GiB | 4.16GiB |
| `speech/us-gov/speech-us-gov-0005.wav` | 26.8 | 0.115 | 3.19GiB | 4.16GiB |
| `speech/us-gov/speech-us-gov-0109.wav` | 19.7 | 0.118 | 3.49GiB | 4.16GiB |
| `noise/free-sound/noise-free-sound-0640.wav` | 32.8 | 0.078 | 3.45GiB | 4.08GiB |
| `noise/free-sound/noise-free-sound-0184.wav` | 32.4 | 0.004 | 411.3MiB | 494.3MiB |
| `noise/free-sound/noise-free-sound-0761.wav` | 32.4 | 0.071 | 3.48GiB | 4.16GiB |
| `noise/free-sound/noise-free-sound-0012.wav` | 32.2 | 0.067 | 3.38GiB | 4.06GiB |
| `noise/free-sound/noise-free-sound-0605.wav` | 32.0 | 0.081 | 3.38GiB | 4.06GiB |
| `noise/free-sound/noise-free-sound-0298.wav` | 32.0 | 0.004 | 411.6MiB | 494.5MiB |
| `noise/free-sound/noise-free-sound-0251.wav` | 32.0 | 0.041 | 3.47GiB | 4.08GiB |
| `noise/free-sound/noise-free-sound-0733.wav` | 31.9 | 0.004 | 413.3MiB | 496.2MiB |
| `noise/free-sound/noise-free-sound-0035.wav` | 31.9 | 0.017 | 3.34GiB | 4.07GiB |
| `noise/free-sound/noise-free-sound-0709.wav` | 31.9 | 0.041 | 3.38GiB | 4.06GiB |
| `noise/sound-bible/noise-sound-bible-0075.wav` | 32.7 | 0.004 | 412.4MiB | 495.4MiB |
| `noise/sound-bible/noise-sound-bible-0044.wav` | 30.0 | 0.004 | 412.0MiB | 495.0MiB |
| `noise/sound-bible/noise-sound-bible-0067.wav` | 27.7 | 0.028 | 3.37GiB | 4.08GiB |
| `noise/sound-bible/noise-sound-bible-0014.wav` | 27.5 | 0.018 | 3.33GiB | 4.06GiB |
| `noise/sound-bible/noise-sound-bible-0032.wav` | 25.2 | 0.072 | 3.29GiB | 4.07GiB |

The pinned real-audio corpus, one call per recording, each measured in a process of its own. Time per second of audio is the warm call — the models already resident — divided by the recording's own length. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), which is the only reading that sees memory allocated for the GPU: the resident set counts different things and is not comparable. Resting is taken once the cold call has returned and before the warm call; peak is the process's lifetime maximum.

_Measured 2026-11-02 23:28 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: d2750edb change-sha256 889347fba2202e3e (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference --features bench --bin transcription-corpus
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
