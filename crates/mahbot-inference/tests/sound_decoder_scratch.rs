// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The sound half's stage-4..6 graph — `dac.blk.1` and `dac.blk.2`, the codec group
//! whose scratch the cooperative implicit-GEMM conv route books — and the two claims
//! that route is admitted there on: the group reserves the 384-channel panel alone, and
//! the convolutions it takes are not slower than the panel route they replace.
//!
//! Needs the installed model directory; a machine without it skips, and a pair that is
//! present but will not open fails. Run with `RLX_ALLOW_THROTTLE=1 cargo test -p
//! mahbot-inference --release --features hand-run --test sound_decoder_scratch --
//! --ignored --nocapture <subject>`, **one subject at a time**: the route switch is
//! process-global, and the cost subject holds two of the group's arenas at once (372 and
//! 527 MB) — a second subject's graphs would cross this machine's memory ceiling.
//!
//! **The reservation.** The graph's arena is `plan + scratch`: the activation plan is
//! `plan_f32_uniform`'s figure and the panel a conv route books is the tail, so only the
//! second moves with the route and only the built arena reports it
//! (`arena_scratch_bytes`; `RLX_WGPU_ARENA_REPORT=1`'s `plan … + scratch …` line is the
//! same figure on stderr). The step census is read beside it, because the same reservation
//! would appear if the 192-channel convs had simply fallen off the route — the group's
//! schedule must name three `conv1d_implicit` steps and three `im2col_gpu` ones.
//!
//! **The cost.** Both arms are built in one process through the route's own switch, so
//! the comparison carries no build-to-build scatter: the group at one length compiled
//! with `RLX_WGPU_CONV_IMPLICIT` at 1 and at 0, the two timed alternately — `ROUNDS`
//! rounds, the arm that runs first swapping each round — with one warm-up pass per arm
//! discarded, and the median of the per-round paired differences as the figure. The
//! sign is the route's own: route minus panel, so a **positive** median means the
//! convolutions the class took are slower than the panel route's.
use mahbot_inference::graphs::{self, DAC_WINDOW};
use mahbot_inference::loader::Role;

mod support;

/// The route's own switch: the class is admitted when it is unset or 1.
const ROUTE: &str = "RLX_WGPU_CONV_IMPLICIT";

/// Rounds the two arms are timed over.
const ROUNDS: usize = 9;

/// The stand's own synthesis length: `synthesis.typical` is a 90-frame part.
const TYPICAL_FRAMES: usize = 90;

/// The group this file is about: `codec_groups()[1]`, DAC stages 4..6 — `dac.blk.1`'s
/// three 384-channel dilated residuals and `dac.blk.2`'s three 192-channel ones.
const GROUP: usize = 1;

/// The panel each arm books at `frames`, in bytes — `c · 7 · out_len · 4` for the two
/// blocks the group runs, read from the codec's own stage table: the one the reservation
/// still covers after the class takes the other, and the one the class takes. Returned
/// as `(with_the_class, without_it)`.
fn panels(frames: usize) -> (u64, u64) {
    // The block's residual `conv1`s: seven taps, at the width each block hands on.
    const TAPS: u64 = 7;
    let group = graphs::codec_groups()[GROUP].clone();
    let panel = |stage: usize| {
        let (ch, len) = graphs::dac_stage_io(stage, frames);
        ch as u64 * TAPS * len as u64 * 4
    };
    // A group's blocks hand on the shapes one stage later than it starts at, so the
    // group `4..6` runs the blocks whose outputs are the shapes at 5 and 6.
    (panel(group.start + 1), panel(group.end))
}

/// The reservation, exactly, at both lengths the codec runs this group at, read with the
/// step census that says which convolutions the reservation is about.
#[test]
#[ignore = "needs the models; asserts the group's reservation and its step census"]
fn the_group_reserves_the_384_channel_panel_alone() {
    let _solo = support::solo("the_group_reserves_the_384_channel_panel_alone");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    for frames in [TYPICAL_FRAMES, DAC_WINDOW] {
        let (panel_384, panel_192) = panels(frames);
        // The route's own switch pinned on, so an exported `=0` cannot decide this
        // subject's figure.
        let graph = support::with_env(ROUTE, "1", || support::codec_group(&pair, frames, GROUP));
        let census = support::schedule_census(&graph);
        let scratch = graph
            .arena_scratch_bytes()
            .expect("the wgpu backend reports the arena's scratch");
        println!(
            "group={}..{} frames={frames} scratch={scratch} B panel_192={panel_192} B \
             panel_384={panel_384} B census={census}",
            graphs::codec_groups()[GROUP].start,
            graphs::codec_groups()[GROUP].end,
        );
        assert_eq!(
            scratch, panel_384,
            "frames={frames}: the group's tail is the 384-channel conv's panel alone \
             ({census})"
        );
        for (name, count) in [("conv1d_implicit", 3), ("im2col_gpu", 3)] {
            assert!(
                census
                    .split('+')
                    .any(|e| e == format!("{name}:{count}").as_str()),
                "frames={frames}: {count} × {name} in the group ({census})"
            );
        }
    }
}

/// The three convolutions the class took, against the panel route they replace: paired,
/// alternating, medians, at the stand's own synthesis length.
#[test]
#[ignore = "measurement only: the group's admitted convs against the panel route"]
fn the_admitted_convs_are_not_slower_than_the_panel_route() {
    let _solo = support::solo("the_admitted_convs_are_not_slower_than_the_panel_route");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    let frames = TYPICAL_FRAMES;
    let (panel_384, panel_192) = panels(frames);
    let mut on = support::with_env(ROUTE, "1", || support::codec_group(&pair, frames, GROUP));
    let mut off = support::with_env(ROUTE, "0", || support::codec_group(&pair, frames, GROUP));
    let (on_census, off_census) = (
        support::schedule_census(&on),
        support::schedule_census(&off),
    );
    let (on_scratch, off_scratch) = (
        on.arena_scratch_bytes()
            .expect("the wgpu backend reports the arena's scratch"),
        off.arena_scratch_bytes()
            .expect("the wgpu backend reports the arena's scratch"),
    );
    println!(
        "census frames={frames} route={on_census} panel={off_census} scratch {on_scratch} B vs \
         {off_scratch} B",
    );
    assert_eq!(
        on_scratch, panel_384,
        "the route's arm reserves the 384-channel panel"
    );
    assert_eq!(
        off_scratch, panel_192,
        "the panel arm reserves the 192-channel one"
    );
    assert!(on_census.contains("conv1d_implicit:3"), "{on_census}");
    assert!(!off_census.contains("conv1d_implicit"), "{off_census}");

    let on_in = support::filled_inputs(&on.runner.graph, 0.1);
    let off_in = support::filled_inputs(&off.runner.graph, 0.1);
    let (on_ms, off_ms, deltas) =
        support::paired_rounds(ROUNDS, &mut on, &on_in, &mut off, &off_in);
    let delta = support::median(&deltas);
    println!(
        "group={}..{} frames={frames} route_ms={:.3} panel_ms={:.3} delta_ms={delta:.3} \
         delta_spread={:.3}..{:.3} slower={}/{ROUNDS} ({ROUNDS} rounds)",
        graphs::codec_groups()[GROUP].start,
        graphs::codec_groups()[GROUP].end,
        support::median(&on_ms),
        support::median(&off_ms),
        support::least(&deltas),
        support::greatest(&deltas),
        deltas.iter().filter(|d| **d > 0.0).count(),
    );
    assert!(
        delta < 0.0,
        "the admitted convolutions are {delta:.3} ms slower than the panel route's \
         (route {:.3} ms, panel {:.3} ms)",
        support::median(&on_ms),
        support::median(&off_ms),
    );
}
