# Measurements

The latest numbers from this crate's two measuring surfaces. Each section is
replaced by the run that produced it and carries its own date; the two surfaces
are run at different times, so one half may be older than the other. This file
is not a log — history is what version control is for.

<!-- BEGIN stand -->
## Per-model measurement (stand)

_Measured 2026-10-19 05:44 UTC._

| unit | cold ms | warm ms | build ms | peak | map | arena | digest |
|---|---|---|---|---|---|---|---|
| ear | 109.3 | 39.5 | 202.8 | 1.05 GiB | 1.53 GiB | 3.4MiB | 92cc8b5117cffde0 |
| ear.075 | 107.9 | 39.3 | 188.6 | 0.83 GiB | 1.53 GiB | 3.4MiB | 569e7385523d15e7 |
| ear.batch2 | 113.8 | 45.4 | 207.7 | 0.83 GiB | 1.53 GiB | 3.6MiB | aac8192598768f05 |
| ear.batch4 | 122.0 | 56.3 | 243.2 | 0.85 GiB | 1.53 GiB | 3.9MiB | 4ea8fb8b0c7196c5 |
| listen | 9420.4 | 8894.8 | 1160.5 | 3.52 GiB | 1.53 GiB | 1.10GiB | 2f93dfa1d7cdeeb1 |
| speak | 48342.9 | 47233.7 | 1168.0 | 2.19 GiB | 1.38 GiB | 724.6MiB | fb160ff9665b4331 |
| embed | 47844.2 | 47717.7 | 49678.4 | 3.13 GiB | 0.37 GiB | 670.2MiB | 17a3b918d44d2849 |

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: ea6ea643-dirty
- command: `cargo run --release -p mahbot-inference` (models from `MAHBOT_MODELS_DIR`, voice from `MAHBOT_VOICE_WAV`)
- model: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- model: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- model: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- model: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
<!-- END stand -->

<!-- BEGIN bench -->
## Wake-word bench

_Measured 2026-10-19 05:37 UTC._

status: `complete`

| metric | value |
|---|---|
| recall | 31/50 (62.0 %) |
| false reactions | 17/121 (14.0 %) |
| false accepts | 0 events · 0.0000/h over VAD-active speech (0.807 h) · 0.0000/h over wall-clock audio (1.066 h) |
| wall clock | 370.3 s |
| peak resident | 1280 MiB |
| corpus | 394 of 394 clips reused, 0 synthesized in 0.0 s |
| windows per encoder pass | at most 16 of 16 · 730 partial of 1567 passes |

| phase | seconds |
|---|---|
| enrolment | 12.7 |
| inputs | 7.3 |
| non-phrase | 40.3 |
| non-phrase synthesis | 0.2 |
| real audio | 295.4 |
| recall | 8.6 |
| recall synthesis | 0.1 |

Provenance:

- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- code: ea6ea643-dirty
- command: `cargo bench -p mahbot-inference --features bench --bench wake_word -- --asr-body ~/.mahbot/userspaces/admin/models/Qwen3-ASR-1.7B-Q4_K_M.gguf --asr-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --tts-body ~/.mahbot/userspaces/admin/models/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf --tts-sound ~/.mahbot/userspaces/admin/models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf --voice ~/.mahbot/userspaces/admin/assets/rlx-4bit-tts/voice-sample.wav --corpus ~/.mahbot/faph_corpus --report /tmp/mahbot-wake-report-final.json --lock /tmp/mahbot-wake.lock --timeout 3600 --max-rss 6144`
- ASR body: `Qwen3-ASR-1.7B-Q4_K_M.gguf` · 1282435552 bytes · sha256 3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6
- ASR sound: `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` · 355709344 bytes · sha256 46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d
- TTS body: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` · 1035965280 bytes · sha256 8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129
- TTS sound: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` · 446422912 bytes · sha256 6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2
- voice: `voice-sample.wav` · 558764 bytes · sha256 0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace
<!-- END bench -->
