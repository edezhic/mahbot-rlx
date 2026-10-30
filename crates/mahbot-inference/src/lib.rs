// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MahBot inference: the fork's own model paths and its wake-word measurement.
//!
//! [`transcribe`] is the product's one transcription surface: any length of
//! audio in, the regions it holds — times, transcripts and sound labels — out.
//! [`synth`] turns text into speech, [`listen`] turns speech into the encoder's
//! window embedding and a transcript, and [`wake`] holds the enrolment,
//! calibration and rolling decision. [`ced`] is the sound tagger the
//! transcription's labels come from and [`vad`] the voice-activity detector both
//! it and the wake path decide on. [`music`] is the music pool the transcription's
//! measurements are taken on. [`stand`] is the measurement stand
//! (`mahbot-inference` binary) and [`bench`] is the wake-word bench, whose corpus
//! the `wake-word-build` binary renders. Every model path enters through an
//! argument or an environment variable — see [`loader::models_dir`] and
//! [`loader::voice_wav`].

pub mod bench;
pub mod ced;
pub mod emit;
pub mod graphs;
pub mod guard;
pub mod host;
pub mod listen;
pub mod loader;
pub mod measurements;
pub mod music;
pub mod reference;
pub mod report;
pub mod stand;
pub mod synth;
pub mod tokenizer;
pub mod transcribe;
pub mod vad;
pub mod wake;

/// Give back every process-wide copy of the models this crate keeps: the transcription
/// ladder's compiled graphs, the host copies of the weights and the device's shared weight
/// pool. A long-lived caller that has finished a pass calls this to return the memory; the
/// next call recompiles and re-reads, so it is a cost, not a state change. Nothing that
/// survives a completed call grows without a bound — the ladder is closed, the host caches
/// are keyed by tensor and the pool by scope — so this is how a process gets *down* from
/// that bound, not what makes it bounded.
///
/// The pages the release freed are handed back to the OS too: macOS keeps them in the
/// malloc zones, so a process that drops one half and builds the next peaks at their sum
/// unless the release also returns them.
pub fn release_process_memory() {
    listen::release_rungs();
    emit::clear_weight_caches();
    rlx_wgpu::buffer::clear_shared_weights();
    report::release_free();
}
