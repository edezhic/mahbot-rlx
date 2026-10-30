// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake-word bench: recall, false reactions and false accepts per hour.
//!
//! A run that measured nothing exits non-zero; the report is written when the
//! measurement completes. The exit code is the bench's own contract — 2 for an input
//! the invocation named and the run cannot use, 1 for anything it reached later — so a
//! script can tell a broken invocation from a broken run.
fn main() {
    if let Err(error) = mahbot_inference::bench::run() {
        eprintln!("wake-word bench: {error:#}");
        std::process::exit(mahbot_inference::bench::exit_code(&error));
    }
}
