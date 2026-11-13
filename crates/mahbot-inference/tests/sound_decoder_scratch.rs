// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The sound half's stage-4..6 graph — `dac.blk.1` and `dac.blk.2`, the codec group
//! whose scratch the cooperative implicit-GEMM conv route books — and the claims the route
//! is admitted there on: the group reserves no panel at the stand's 90-frame part or at the
//! codec's 144- and 148-frame windows, and keeps it at the stand's 23-frame part below the
//! route's extent floor; the row's peak arena is the last group's and no longer this one's;
//! and the six convolutions together are faster than the panel route they replace. The width
//! this group adds to the class is admitted for its residence; its own per-shape time is
//! measured, not promised, in `rlx-wgpu`'s `conv_implicit_vs_panel_at_the_admitted_widths`.
//!
//! Needs the installed model directory; a machine without it skips, and a pair that is
//! present but will not open fails. Run with `RLX_ALLOW_THROTTLE=1 cargo test -p
//! mahbot-inference --release --features hand-run --test sound_decoder_scratch --
//! --ignored --nocapture <subject>`, **one subject at a time**: the route switch is
//! process-global, and the largest arena any subject builds is the reservation subject's
//! class-off arm at the 148-frame window, 846 654 336 B (the group's 337 439 616 B plan plus
//! 509 214 720 B of panel; its two arms are built in sequence, each dropped before the next),
//! while the cost subject holds two graphs, 745 068 288 B together, at once — so a second
//! subject's graphs would cross this machine's memory ceiling.
//!
//! **The reservation.** The graph's arena is its plan plus its scratch, and only the
//! scratch moves with the route; the built arena is what reports it (`arena_bytes`;
//! `RLX_WGPU_ARENA_REPORT=1`'s `plan … + scratch … -> size` line is the same figure on
//! stderr). The step census is read beside it, because a zero reservation would appear
//! just as well if the group's convolutions had fallen off every accelerated route — the
//! schedule must name six `conv1d_implicit` steps and no `im2col_gpu`.
//!
//! **The peak.** The row's peak is the largest arena a call's graphs are built at. The
//! sound half's front graph is `codec_groups()[0]` at `frames.min(FRONT_WINDOW)` — 72 frames
//! at these lengths, and gone before the decoder's groups are built — and the talker body's
//! graph is smaller at these lengths, so the largest of the three codec groups is the row's
//! peak: the front's first window and the decoder's two groups are built at the lengths the
//! call runs them at, and the largest is pinned with the group that sets it. The front's
//! carried chunk and the talker body are argued below the peak rather than built. The peak
//! the same call held with this group's panel booked is pinned beside it from the previous
//! tree's arm — the extent floor set between the two widths — which is the graph that tree
//! built.
//!
//! **The cost.** Both arms are built in one process through the route's own switch, so
//! the comparison carries no build-to-build scatter: the group at one length compiled
//! with `RLX_WGPU_CONV_IMPLICIT` at 1 and at 0, the two timed alternately — `ROUNDS`
//! rounds, the arm that runs first swapping each round — with one warm-up pass per arm
//! discarded, and the median of the per-round paired differences as the figure. The
//! sign is the route's own: route minus panel, so a **positive** median means the
//! convolutions the class took are slower than the panel route's. Both arms carry six
//! convolutions, three at each of the class's two widths, so this is the group's own
//! figure — the product path's — and not either width's.
//!
//! All of it is Metal-class only: the reservation is subtracted where the class's switch
//! and the adapter's cooperative-matrix entry agree, and the route is admitted only there.
//! An adapter without the entry books the panel, so the subjects here skip on it — as the
//! `rlx-wgpu` cells do — rather than pinning a figure the adapter decides.
use mahbot_inference::graphs::{self, DAC_MARGIN, DAC_WINDOW};
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

/// The extent floor that reproduces the previous tree's routing for this group: above the
/// 384-channel shapes' own extents (14 400 to 23 680 output columns) and below the
/// 192-channel ones' (57 600 to 94 720), so the class takes `dac.blk.2` and leaves
/// `dac.blk.1` on the panel. A shape change that moves either width past it turns the
/// previous-tree arm into this tree's, and the peaks below stop being two states of one row.
const PRE_CHANGE_FLOOR: &str = "30000";

/// The two extent floors, as a skip: each is a bisect lever, and under either the group's
/// routing is not the shipped one. The route's own switch is not one of them — every subject
/// pins it around the graph it builds — and the adapter's cooperative-matrix entry is asked
/// separately below. [`support::pair_of`] has created the device by the time this is asked.
fn skip_without_the_route() -> bool {
    for key in [
        "RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL",
        "RLX_WGPU_IM2COL_MIN_SPATIAL",
    ] {
        if let Some(value) = rlx_ir::env::var(key) {
            eprintln!("skip: {key}={value} is a bisect, not the default");
            return true;
        }
    }
    if support::implicit_route_is_live() {
        return false;
    }
    eprintln!("skip: the implicit conv route needs the adapter's cooperative-matrix entry");
    true
}

/// The reservation, exactly zero at the stand's 90-frame part and at the codec's 144- and
/// 148-frame windows, the census that says which convolutions the reservation is about, and
/// the panel the same graph books with the class switched off. The stand's 23-frame part is
/// below the route's extent floor (3 680 output columns against 4 800), so the 384-channel
/// convs keep their panel there and the short row's memory is not meant to move.
#[test]
#[ignore = "needs the models; asserts the group's reservation and its step census"]
fn the_group_reserves_no_panel() {
    let _solo = support::solo("the_group_reserves_no_panel");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    if skip_without_the_route() {
        return;
    }
    // The part's own window, the codec's window, and the span every later window of a
    // longer part runs: the recurring one is `DAC_WINDOW` plus its left margin.
    /// `(frames, the panel the group books without the class)`.
    const PANELS: [(usize, u64); 3] = [
        (TYPICAL_FRAMES, 309_657_600),
        (DAC_WINDOW, 495_452_160),
        (DAC_WINDOW + DAC_MARGIN, 509_214_720),
    ];
    for (frames, booked) in PANELS {
        // The route's own switch pinned on, so an exported `=0` cannot decide this
        // subject's figure. The block is a drop: the class-off arm below builds the same group
        // with up to 510 MB of panel booked, and holding both arenas at once would cross
        // this machine's ceiling.
        {
            let graph =
                support::with_env(ROUTE, "1", || support::codec_group(&pair, frames, GROUP));
            let census = support::schedule_census(&graph);
            let scratch = graph
                .arena_scratch_bytes()
                .expect("the wgpu backend reports the arena's scratch");
            println!(
                "group={}..{} frames={frames} scratch={scratch} B arena={} B census={census}",
                graphs::codec_groups()[GROUP].start,
                graphs::codec_groups()[GROUP].end,
                graph
                    .arena_bytes()
                    .expect("the wgpu backend reports the arena's size"),
            );
            assert_eq!(
                scratch, 0,
                "frames={frames}: the group reserves no panel ({census})"
            );
            // Six dilated residuals: `dac.blk.1`'s three at the class's widest output and
            // `dac.blk.2`'s three at half of it. The count is exact, so one that fell to the
            // direct kernel shows up here as five; `im2col_gpu` is the panel route, which
            // none of them may take. (A plain `conv2d` step is not checked — the group's 1×1
            // residuals would be one too if their own route refused.)
            assert!(
                census.split('+').any(|e| e == "conv1d_implicit:6"),
                "frames={frames}: six × conv1d_implicit in the group ({census})"
            );
            assert!(
                !census.split('+').any(|e| e.starts_with("im2col_gpu")),
                "frames={frames}: no im2col panel in the group ({census})"
            );
        }
        // …and the same graph with the class off books exactly the panel the six
        // convolutions' own arithmetic gives, which is what the class removes.
        let class_off =
            support::with_env(ROUTE, "0", || support::codec_group(&pair, frames, GROUP));
        let census = support::schedule_census(&class_off);
        let scratch = class_off
            .arena_scratch_bytes()
            .expect("the wgpu backend reports the arena's scratch");
        println!(
            "frames={frames} class_off_scratch={scratch} B arena={} B census={census}",
            class_off
                .arena_bytes()
                .expect("the wgpu backend reports the arena's size"),
        );
        assert_eq!(
            scratch, booked,
            "frames={frames}: the panel the class removes ({census})"
        );
    }

    // Below the floor the class does not reach, so the short part's group is the graph it
    // was: the three 384-channel convs on the panel, the three 192-channel ones on the
    // route. That is why the stand's short synthesis row's resting GPU has nothing to
    // move for — its graph does not change.
    let frames =
        mahbot_inference::reference::STAND_SYNTH_SHORT_SAMPLES / graphs::samples_per_frame();
    let graph = support::with_env(ROUTE, "1", || support::codec_group(&pair, frames, GROUP));
    let census = support::schedule_census(&graph);
    let scratch = graph
        .arena_scratch_bytes()
        .expect("the wgpu backend reports the arena's scratch");
    println!("group={GROUP} frames={frames} scratch={scratch} B census={census}");
    assert_eq!(
        scratch, 39_567_360,
        "frames={frames}: below the extent floor the 384-channel convs keep their panel \
         ({census})"
    );
    assert!(
        census.split('+').any(|e| e == "conv1d_implicit:3"),
        "frames={frames}: and the 192-channel ones stay on the route ({census})"
    );
    assert!(
        census.split('+').any(|e| e.starts_with("im2col_gpu")),
        "frames={frames}: on the panel ({census})"
    );
}

/// The arena the row peaks on, at the stand's own synthesis length and at both of the
/// codec's windows: the largest of the three codec groups' built arenas — group 0 at the
/// length the front runs it, the decoder's two at the part's own — pinned exactly and
/// attributed to the group that sets it, beside the peak the same call would have held with
/// the changed group's panel booked. Only that group's arena moves with the route, so the two
/// peaks are the two states of the same row. The talker body's graph is the one a call holds
/// without building here: the module doc argues it below every codec group at these lengths.
#[test]
#[ignore = "needs the models; asserts which codec group sets the row's peak and its size"]
fn the_row_peaks_on_the_last_group() {
    let _solo = support::solo("the_row_peaks_on_the_last_group");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    if skip_without_the_route() {
        return;
    }
    let groups = graphs::codec_groups();
    /// `(frames, the row's peak, the same peak with the changed group's panel booked)`.
    const PEAKS: [(usize, u64, u64); 3] = [
        (TYPICAL_FRAMES, 311_001_520, 372_534_144),
        (DAC_WINDOW, 496_796_080, 576_908_160),
        (DAC_WINDOW + DAC_MARGIN, 510_558_640, 592_046_976),
    ];
    for (frames, peak_pinned, booked_pinned) in PEAKS {
        let mut arenas = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            // Group 0 is the sound half's front, which the product builds for the first
            // window only — `frames.min(FRONT_WINDOW)` — and drops before the decoder's
            // groups are built. The other two run at the part's own length.
            let len = if i == 0 {
                frames.min(graphs::FRONT_WINDOW)
            } else {
                frames
            };
            let graph = support::with_env(ROUTE, "1", || support::codec_group(&pair, len, i));
            let arena = graph
                .arena_bytes()
                .expect("the wgpu backend reports the arena's size");
            println!(
                "group={}..{} frames={len} arena={arena} B",
                group.start, group.end,
            );
            arenas.push((i, arena));
        }
        let (peak_group, peak) = arenas
            .iter()
            .copied()
            .max_by_key(|&(_, arena)| arena)
            .expect("the sound half has groups");
        assert_eq!(
            peak_group,
            groups.len() - 1,
            "frames={frames}: the last group sets the row's peak, not group {peak_group}"
        );

        // The row's peak with the changed group's panel booked: the same three arenas with
        // that one taken from the previous tree's arm. The route switch is all-or-nothing,
        // so that tree's routing is reproduced with the extent floor instead — above the
        // 384-channel shapes' own extents and below the 192-channel ones' — which is the
        // class taking `dac.blk.2` and leaving `dac.blk.1` on the panel.
        let previous_tree = support::with_env(ROUTE, "1", || {
            support::with_env(
                "RLX_WGPU_CONV_IMPLICIT_MIN_SPATIAL",
                PRE_CHANGE_FLOOR,
                || support::codec_group(&pair, frames, GROUP),
            )
        });
        let mut booked = arenas.clone();
        booked[GROUP].1 = previous_tree
            .arena_bytes()
            .expect("the wgpu backend reports the arena's size");
        let (booked_group, booked_peak) = booked
            .iter()
            .copied()
            .max_by_key(|&(_, arena)| arena)
            .expect("the sound half has groups");
        println!(
            "frames={frames} peak={peak} B on group={peak_group} booked_peak={booked_peak} B on \
             group={booked_group}"
        );
        assert_eq!(
            booked_group, GROUP,
            "frames={frames}: the changed group's own arena is what sets the booked peak"
        );
        assert_eq!(peak, peak_pinned, "frames={frames}: the row's peak");
        assert_eq!(
            booked_peak, booked_pinned,
            "frames={frames}: and the peak with the changed group's panel booked"
        );
    }
}

/// The group's six convolutions, against the panel route they replace: paired, alternating,
/// medians, at the stand's own synthesis length. The figure is the group's, not either
/// width's — the per-shape figures, and the deliberate asymmetry between the width that is
/// gated there and the one that is only recorded, are
/// `conv_implicit_vs_panel_at_the_admitted_widths` in `rlx-wgpu`. This is the product path's
/// own time claim: the group the codec runs must beat the panel route it replaced, or the
/// width has bought its residence with the call's time. It asserts; no CI gate runs it —
/// like every subject in this file, it is hand-run.
#[test]
#[ignore = "hand-run: asserts the group's six convolutions beat the panel route"]
fn the_group_is_faster_than_the_panel_route() {
    let _solo = support::solo("the_group_is_faster_than_the_panel_route");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    if skip_without_the_route() {
        return;
    }
    let frames = TYPICAL_FRAMES;
    let mut on = support::with_env(ROUTE, "1", || support::codec_group(&pair, frames, GROUP));
    let mut off = support::with_env(ROUTE, "0", || support::codec_group(&pair, frames, GROUP));
    // Which arm is which is `the_group_reserves_no_panel`'s at this length; the two censuses
    // are printed here for the context the pair of timings is read in.
    println!(
        "census frames={frames} route={} panel={}",
        support::schedule_census(&on),
        support::schedule_census(&off),
    );

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
