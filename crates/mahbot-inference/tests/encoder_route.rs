// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The encoder's route, as a gate: every block of the wake-word path reads its three
//! attention input projections from one wide packed blob (`graphs::block`'s `qkv_fused`,
//! decided for the whole graph by `graphs::fuse_projection`), and none falls back to three
//! separate passes. The route is otherwise invisible — the fallback emits the same
//! arithmetic and differs only in how often the device is fed — so the compiled schedule is
//! what this counts.
//!
//! `tests/encoder_qkv.rs` prices the two forms and needs a second arm; this needs the
//! shipped one, and runs under a bare `cargo test`. Model- and device-dependent, so without
//! `MAHBOT_MODELS_DIR` or a wgpu adapter it says why it skipped.

use mahbot_inference::graphs;
use mahbot_inference::listen::{GROUP, MEL_FRAMES};
use mahbot_inference::loader::{Pair, Role};
use rlx_runtime::Device;

mod support;

use support::{PER_BLOCK_FUSED, PER_BLOCK_SPLIT, packed_multiplies};

#[test]
fn every_encoder_block_takes_one_wide_projection() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("skipped: {error:#} — the encoder's route was not checked");
            return;
        }
    };
    let blocks = pair.sound.clip().layers;
    let cases = [
        (
            "the wake window's whole encoder",
            graphs::asr_encoder(pair.sound.clone(), MEL_FRAMES),
        ),
        (
            "the batching tail, sixteen windows in one pass",
            graphs::asr_encoder_tail(pair.sound.clone(), MEL_FRAMES, GROUP),
        ),
    ];
    for (label, built) in cases {
        let compiled = graphs::Compiled::new(Device::Gpu, built.expect("the encoder builds"))
            .expect("the encoder compiles");
        let names = compiled
            .schedule_step_names()
            .expect("the wgpu backend exposes its schedule");
        let multiplies = packed_multiplies(&names);
        assert_eq!(
            multiplies,
            blocks * PER_BLOCK_FUSED,
            "{label}: {blocks} blocks of {PER_BLOCK_FUSED} packed multiplies is {}; {} means the \
             blocks fell back to three separate projections, or the file's q/k/v can no longer \
             share one blob",
            blocks * PER_BLOCK_FUSED,
            blocks * PER_BLOCK_SPLIT,
        );
    }
}
