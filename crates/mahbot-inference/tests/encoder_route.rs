// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The encoder's routes, as gates: every block of the wake-word path reads its three
//! attention input projections from one wide packed blob (`graphs::block`'s `qkv_fused`,
//! decided for the whole graph by `graphs::fuse_projection`), and none falls back to three
//! separate passes; and the encoder's conv front keeps its im2col panel route at the pair's
//! batch, which is the whole point of the batched front end (a conv that loses the panel
//! falls to the direct kernel and costs more per window than two single-window passes). Both
//! routes are otherwise invisible — the fallbacks emit the same arithmetic and differ only in
//! how the device is read — so the compiled schedule is what this counts.
//!
//! `tests/encoder_qkv.rs` prices the two projections and needs a second arm, and
//! `tests/wake_pair_cost.rs` prices the batched pass against two single ones; this needs the
//! shipped one, and runs under a bare `cargo test`. Model- and device-dependent, so without
//! the installed model directory or a wgpu adapter it says why it skipped.

use mahbot_inference::graphs;
use mahbot_inference::listen::{GROUP, MEL_FRAMES, PAIR};
use mahbot_inference::loader::{Pair, Role};
use rlx_runtime::Device;

mod support;

use support::{PER_BLOCK_FUSED, PER_BLOCK_SPLIT, packed_multiplies};

/// The steps of one kind in a compiled schedule, counted.
fn steps(names: &[&'static str], kind: &str) -> usize {
    names.iter().filter(|name| **name == kind).count()
}

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
    // Each graph is built and compiled in turn, never two builds before one compile:
    // the build's weight records (`emit`) live for the one compile that follows, and a
    // second build in between replaces them — the encoder's host f32 weights would then
    // have no resolvable bytes and the compile refuses rather than run on zeros.
    let cases: [(&str, fn(&Pair) -> anyhow::Result<rlx_flow::BuiltModel>); 2] = [
        ("the wake window's whole encoder", |pair| {
            graphs::asr_encoder(pair.sound.clone(), MEL_FRAMES, 1)
        }),
        ("the batching tail, sixteen windows in one pass", |pair| {
            graphs::asr_encoder_tail(pair.sound.clone(), MEL_FRAMES, GROUP)
        }),
    ];
    for (label, build) in cases {
        let built = build(&pair).expect("the encoder builds");
        let compiled = graphs::Compiled::new(Device::Gpu, built).expect("the encoder compiles");
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

/// The conv front's own route at one window and at the pair's two: the panel gather
/// and the multiply that consumes it, counted on the product's whole encoder — the
/// graph the pair's front end compiles — so a route decided only by the full graph's
/// arena (the whole-arena binding, the reserved scratch) is gated too. A batch that
/// drops the panel shows up as fewer `im2col_gpu` steps and more `conv2d` ones, which
/// is the per-window slow path the batched front end exists to avoid.
#[test]
fn the_conv_front_keeps_its_panel_route_at_the_pairs_batch() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("skipped: {error:#} — the conv front's route was not checked");
            return;
        }
    };
    let encoder_steps = |batch| {
        let built =
            graphs::asr_encoder(pair.sound.clone(), MEL_FRAMES, batch).expect("the encoder builds");
        let compiled = graphs::Compiled::new(Device::Gpu, built).expect("the encoder compiles");
        let names = compiled
            .schedule_step_names()
            .expect("the wgpu backend exposes its schedule");
        (
            steps(&names, "im2col_gpu"),
            steps(&names, "matmul"),
            steps(&names, "conv2d"),
        )
    };
    let one = encoder_steps(1);
    let two = encoder_steps(PAIR);
    assert_eq!(
        one, two,
        "the conv front routes the same at one window and at {PAIR}: {one:?} against {two:?}; a \
         batch that loses a panel gather reads the direct kernel instead, which is dearer per \
         window than two single-window passes",
    );
    assert!(
        one.0 > 0,
        "the shipped front convs take the im2col panel route at all: {one:?}",
    );
}
