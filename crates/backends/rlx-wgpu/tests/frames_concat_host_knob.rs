// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `RLX_WGPU_CONCAT_HOST` — the concat arm's own bisect lever — stands the mel
//! front-end's framing fold down with it.
//!
//! Its own test binary on purpose, as `binary_parked_param.rs` is: the lever is a
//! code override (`rlx_ir::env::set`), which is process-global, so the pins in
//! `frames_concat_fusion.rs` mask it off and this file is the one that reads it
//! on. Without the split the two would race.

use rlx_wgpu::backend::WgpuExecutable;

mod common;

#[test]
fn the_concat_host_knob_stands_the_fold_down() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (n_frames, frame_len, hop) = (8, 64, 16);
    let t = common::frames_signal_len(n_frames, frame_len, hop);
    let sig = common::framing_signal(t);
    let want = common::framing_reference(
        &sig,
        1,
        t,
        &common::framing_starts(n_frames, hop),
        frame_len,
    );
    // The premise: this graph is one the fold recognises, so what the census
    // below reads is the lever rather than a graph it would never have folded.
    // Every knob that moves this route is neutralised first — the shared set, plus
    // this fold's own concat lever, which belongs to one op and so is not in that
    // set — so the premise reads this backend's routing rather than the ambient
    // environment.
    common::neutralise_device_route_env();
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "0");
    let g = common::framed_over_input(n_frames, frame_len, hop);
    assert_eq!(
        WgpuExecutable::compile(g.clone())
            .step_kind_counts()
            .get("frames"),
        Some(&1),
        "premise: with the lever off this graph folds, so the lever is what the census below \
         reads"
    );
    let folded = WgpuExecutable::compile(g.clone())
        .run(&[("x", &sig)])
        .remove(0);
    common::assert_bit_identical(
        &folded,
        &want,
        "the folded route's matrix, which is what the lever's route has to match",
    );
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "1");
    let census = WgpuExecutable::compile(g.clone()).step_kind_counts();
    assert_eq!(
        census.get("frames"),
        None,
        "with RLX_WGPU_CONCAT_HOST=1 the fold must stand down; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the per-frame steps must be the ones left; census {census:?}"
    );
    // The lever only takes the concat off its direct window path: on this adapter it
    // comes back as a device concat staged through scratch, so what replaces the
    // fold still owes the matrix the fold produced.
    let unfolded = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    common::assert_bit_identical(
        &unfolded,
        &want,
        "the concat route the lever selects is the same framing",
    );
}
