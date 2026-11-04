// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MahBot inference: the fork's own model paths and their measurement.
//!
//! [`transcribe`] is the product's one transcription surface: any length of
//! audio in, the regions it holds — times, transcripts and sound labels — out.
//! [`synth`] turns text into speech, [`listen`] turns speech into the encoder's
//! window embedding and a transcript, and [`wake`] holds the enrolment,
//! calibration and rolling decision. [`ced`] is the sound tagger the
//! transcription's labels come from and [`vad`] the voice-activity detector both
//! it and the wake path decide on. [`stand`] is the measurement stand
//! (`mahbot-inference` binary), [`corpus`] the transcription corpus run
//! (`transcription-corpus` binary) and [`bench`](mod@bench) the wake-word bench, whose corpus
//! the `wake-word-build` binary renders. Every input lives at one fixed path per
//! operating system ([`paths`]); nothing is configured, and the one directory the
//! crate creates is `wake-corpus/` below that root.

pub mod bench;
pub mod ced;
pub mod corpus;
pub mod emit;
pub mod graphs;
pub mod guard;
pub mod host;
pub mod listen;
pub mod loader;
pub mod measurements;
pub mod music;
pub mod paths;
pub mod reference;
pub mod report;
mod run;
pub mod speaker;
pub mod stand;
pub mod synth;
pub mod tokenizer;
pub mod transcribe;
pub mod vad;
pub mod wake;

/// Give back every process-wide copy of the models this crate keeps: the decode graph a
/// resident pair holds, the process's mel graphs, the host copies of the weights and the
/// device's shared weight pool. A long-lived caller that has finished a pass drops them
/// here; the next call recompiles and re-reads, so it is a cost, not a state change.
/// Nothing that survives a completed call grows without a bound — the host caches are keyed
/// by tensor and the pool by scope — so this is how a process gets *down* from that bound,
/// not what makes it bounded. A rung's own graphs are not part of this: they belong to the
/// call that built them and are already gone when it returns.
///
/// What this does not do is hand the freed pages back to the operating system: measured on
/// macOS 26.5/arm64, neither dropping a 378 MiB `Vec` nor `malloc_zone_pressure_relief`
/// returns one, so what keeps the footprint down is not keeping a copy, not releasing one
/// later.
pub fn release_process_memory() {
    listen::clear_graph_caches();
    emit::clear_weight_caches();
    rlx_wgpu::buffer::clear_shared_weights();
}
