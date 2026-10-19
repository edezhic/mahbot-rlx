// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MahBot inference: the fork's own model paths and its wake-word measurement.
//!
//! [`synth`] turns text into speech, [`listen`] turns speech into the encoder's
//! window embedding and a transcript, and [`wake`] holds the enrolment,
//! calibration and rolling decision. [`stand`] is the measurement stand
//! (`mahbot-inference` binary) and [`bench`] is the wake-word bench. Every model
//! path enters through an argument or an environment variable — see
//! [`loader::models_dir`] and [`loader::voice_wav`].

pub mod bench;
pub mod emit;
pub mod graphs;
pub mod guard;
pub mod host;
pub mod listen;
pub mod loader;
pub mod measurements;
pub mod report;
pub mod stand;
pub mod synth;
pub mod tokenizer;
pub mod wake;
