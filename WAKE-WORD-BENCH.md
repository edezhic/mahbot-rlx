# Wake-word bench

| metric | value |
|---|---|
| corpus | 1956 clips |
| negatives | 1161 clips — 1144 speech across 11 voices, 17 non-speech in a stream of their own |
| negative overall (pooled speech) | 54/1144 (4.7 %) |
| non-speech | 0/17 (0.0 %) — silence and noise, measured apart |
| positives (pooled) | 61/550 (11.1 %) |
| false accepts | 2 events · 2.4773/h over VAD-active speech (0.807 h) · 1.8756/h over wall-clock audio (1.066 h) · one event = 1.2386/h |
| derived point | floor 0.7500 · fire factor 0.3520 · static bar 1.0560 · reset 0.3344 |
| wall clock | 809.3 s |
| peak resident | 1.85GiB |

One complete run: recall and false reactions in each of the eleven voices, and the false accepts per hour over the pinned real-audio corpus.

| voice | phonetic near-misses (56) | unrelated sentences (40) | wake phrase inside a sentence (8) | overall (104) |
|---|---|---|---|---|
| enrolled | 31/56 (55.4 %) | 11/40 (27.5 %) | 2/8 (25.0 %) | 44/104 (42.3 %) |
| voice01 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice02 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice03 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice04 | 2/56 (3.6 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 2/104 (1.9 %) |
| voice05 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice06 | 1/56 (1.8 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 1/104 (1.0 %) |
| voice07 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice08 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice09 | 0/56 (0.0 %) | 0/40 (0.0 %) | 0/8 (0.0 %) | 0/104 (0.0 %) |
| voice10 | 6/56 (10.7 %) | 0/40 (0.0 %) | 1/8 (12.5 %) | 7/104 (6.7 %) |
| pooled over 11 voices | 40/616 (6.5 %) | 11/440 (2.5 %) | 3/88 (3.4 %) | 54/1144 (4.7 %) |

Pooled over 11 voices: 104 clips each, not 1144 independent trials.

| voice | detected | of | rate |
|---|---|---|---|
| enrolled | 46 | 50 | 92.0 % |
| voice01 | 0 | 50 | 0.0 % |
| voice02 | 0 | 50 | 0.0 % |
| voice03 | 0 | 50 | 0.0 % |
| voice04 | 0 | 50 | 0.0 % |
| voice05 | 0 | 50 | 0.0 % |
| voice06 | 7 | 50 | 14.0 % |
| voice07 | 0 | 50 | 0.0 % |
| voice08 | 0 | 50 | 0.0 % |
| voice09 | 0 | 50 | 0.0 % |
| voice10 | 8 | 50 | 16.0 % |
| pooled over 11 voices | 61 | 550 | 11.1 % |

Pooled over 11 voices: 50 clips each, not 550 independent utterances.

_Measured 2026-10-31 14:20 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: e199dfd4 change-sha256 768d65d2becdb727 (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo bench -p mahbot-inference --features bench --bench wake_word -- --report <file>
- models: `~/.mahbot/userspaces/admin/models`
- corpus: `~/.mahbot/faph_corpus`
- ASR body: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- ASR sound: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- TTS body: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- TTS sound: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
- voice01: `speech/librivox/speech-librivox-0002.wav` · 11.40 s + 15.00 s · sha256 f907f96e69eb8e680c86964f63ff570d2c24458beacd7d94520c44b2a5c10868
- voice02: `speech/librivox/speech-librivox-0039.wav` · 2.80 s + 15.00 s · sha256 41fb73636b15dc76815fff9f816c97e897d2d920acef4b89097f8f05fe682fca
- voice03: `speech/librivox/speech-librivox-0094.wav` · 14.80 s + 15.00 s · sha256 b529147404ffd239df9a5eaddfe19e7130393fb756e4f56f4ee85d68cd63cc7d
- voice04: `speech/librivox/speech-librivox-0092.wav` · 106.60 s + 15.00 s · sha256 00eaf9379dfd80075731d89a9f83df2139b4959037513f61597f848a5668fff0
- voice05: `speech/librivox/speech-librivox-0079.wav` · 131.60 s + 15.00 s · sha256 116937d90638753623eabc97626e0d18b7b73ffd523534caf055a48171c67d81
- voice06: `speech/librivox/speech-librivox-0168.wav` · 141.10 s + 15.00 s · sha256 d5f2d58fffc2ed0694e5362ac047e3f0386a864fa0ff5f1c9d6ce66c399e4731
- voice07: `speech/librivox/speech-librivox-0009.wav` · 3.10 s + 15.00 s · sha256 afdfc48bb6ac16ffd97cf0663cc90122d7defa17d07d6b3c5017a6523c086cef
- voice08: `speech/librivox/speech-librivox-0089.wav` · 13.80 s + 15.00 s · sha256 cf8dfec778ea249d67282b7a7396fa954d212280ff44cb723cdb45e15f2ed01b
- voice09: `speech/librivox/speech-librivox-0129.wav` · 40.60 s + 15.00 s · sha256 75252026b534c7bb730c44cb721e6c7105220cbf1b4023eee9c7d5af40256adf
- voice10: `speech/librivox/speech-librivox-0065.wav` · 5.90 s + 15.00 s · sha256 725adb1ab5289dbcd8c8016285ae34890c96ac4f150995f20d078a1cfd1ecbf3
