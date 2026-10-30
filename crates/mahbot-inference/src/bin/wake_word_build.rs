// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake-word corpus build: render every clip the bench's measurement needs,
//! report how many clips it reused and how many it rendered, and exit. Not a
//! measurement — no record.

fn main() {
    mahbot_inference::bench::build::main_entry();
}
