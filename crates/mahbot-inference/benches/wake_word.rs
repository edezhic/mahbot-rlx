// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake-word bench: recall, false reactions and false accepts per hour.
//!
//! A run that measured nothing exits non-zero; the report is written when the
//! measurement completes.

fn main() {
    if let Err(e) = mahbot_inference::bench::run() {
        eprintln!("wake-word bench: {e:#}");
        std::process::exit(1);
    }
}
