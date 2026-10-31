// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measurement only and ignored: the decoder's graphs — what they cost to build,
//! what their cache is worth to a served request, and what the served path's two
//! synthesis calls cost in time and resident set.
//!
//! Run with `MAHBOT_MODELS_DIR=… RLX_ALLOW_THROTTLE=1 cargo test -p mahbot-inference
//! --release --test sound_decoder_cost -- --ignored --nocapture <subject>`.
//! Each subject is its own test and wants its own process: the device keeps what
//! a test built, so one run of all four crosses this machine's memory ceiling —
//! take them one at a time (`graphs`, `front`, `margin`, `library`).
//!
//! **The build.** The graphs one 330-row utterance runs, requested twice through
//! `graphs::SoundGraphs` — the cache `Synthesis` owns — so the second round is the
//! hit a repeated request of a length gets, and the first is the build a served
//! path would otherwise pay per request. The front is built by driving
//! `graphs::FrontGraphs::build`, the builder both call sites use, so the probe
//! cannot drift from what the sound half runs.
//!
//! **The front's last window.** The shipped arrangement — a full chunk padded with
//! code 0 — against the same frames at their true length through a graph of their
//! own. Paired, alternating, in one window: the difference is what the padding
//! costs an utterance, and the front-build arm above is what a graph of the tail's
//! own length costs to build, so the two decide that trade together.
//!
//! **The margins.** A DAC group's full window with its right margin against the
//! same window without it, per group, paired the same way. With the padded last
//! window above it is the whole of what the served path gained from carrying no
//! right margin. The absolute per-window costs move between runs; the paired delta
//! and the rounds that agree on it are the figure.
//!
//! **The served path.** One process, two `Synthesis::synthesize` calls of the
//! pinned text and a third of another so the cache meets a second length:
//! the second call is the reuse a repeated request gets, and the resident set
//! after each call is what the decoder's retained graphs cost the process.
use std::sync::Arc;

use mahbot_inference::graphs::{self, DAC_MARGIN, DAC_WINDOW, FRONT_WINDOW, windows};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::report;
use mahbot_inference::synth::{MAX_CODEC_ROWS, Synthesis};
use rlx_ir::Tick;
use rlx_runtime::Device;

mod support;

use support::TTS_TEXT;

/// Rounds one arm is timed over; the median is the arm's figure and the spread of
/// the paired deltas is the comparison's own.
const ROUNDS: usize = 9;

/// Rounds one front graph's build is timed over, alternating the tail's length with
/// a full chunk's so both arms see the same window.
const BUILD_ROUNDS: usize = 7;

/// One timed run of a compiled graph. The warm-up pass and the alternating loop
/// that pairs two arms live at the call sites, which own the comparison.
fn run_ms(graph: &graphs::SoundGraph, ins: &[(String, Vec<f32>)]) -> f64 {
    let slices: Vec<(&str, &[f32])> = ins
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let t = Tick::now();
    graph.borrow_mut().run_slices(&slices);
    Tick::now().elapsed_ns(t) as f64 / 1e6
}

/// The process's resident set, which is what the cache costs the ceiling rather
/// than what its graphs plan.
fn rss() -> Option<usize> {
    report::resident_rss().map(|bytes| bytes as usize)
}

/// Bytes in decimal MB as this harness prints them: its own report's unit, stated
/// on each line it writes so a figure is never a bare number.
fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

/// [`mb`] of a reading that may be unavailable: an unavailable reading prints as
/// `-`, never as a zero.
fn mb_of(reading: Option<usize>) -> String {
    reading.map_or_else(|| "-".to_string(), |bytes| format!("{:.1}", mb(bytes)))
}

/// [`mb_of`] of what `after` holds beyond `before`, `-` when either end is
/// unavailable.
fn grown(before: Option<usize>, after: Option<usize>) -> String {
    mb_of(
        before
            .zip(after)
            .map(|(before, after)| after.saturating_sub(before)),
    )
}

/// The synthesis pair a subject reads, or `None` after naming what stopped it: every
/// subject needs the device and `MAHBOT_MODELS_DIR` naming a pair.
fn pair_or_skip() -> Option<Pair> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return None;
    }
    if let Err(e) = loader::models_dir() {
        eprintln!("skip: {e} — the probe did not run");
        return None;
    }
    match Pair::of_role(Role::Synthesis) {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("skip: {e} — the probe did not run");
            None
        }
    }
}

/// The tail the probes pair against a padded chunk must not itself be one chunk
/// long: one build arm would be empty, and the other would compare a graph with
/// itself.
fn assert_tail_is_not_a_chunk(tail_len: usize) {
    assert_ne!(
        tail_len, FRONT_WINDOW,
        "the codec's rows must not end on a front-chunk boundary, or the tail arm prices a chunk"
    );
}

/// The decoder's graphs, built once and then hit: the eight one 330-row utterance
/// runs, their parameters, their arenas and the bytes the cache holds; the front's
/// graph set per part length; and one front graph's own build, averaged alternately
/// against a full chunk's — the figure a graph per tail length would be paid at.
#[test]
#[ignore = "measurement only: prints the decoder's graph build against its cache hit"]
fn the_decoders_graphs_build_once_and_are_hit() {
    let Some(pair) = pair_or_skip() else { return };
    let device = Device::Gpu;
    let groups = graphs::codec_groups();
    let parts = windows(MAX_CODEC_ROWS, DAC_WINDOW, DAC_MARGIN);
    let front_set = windows(MAX_CODEC_ROWS, FRONT_WINDOW, 0);
    let tail_len = front_set.last().map_or(0, |&(p0, p1, _, _)| p1 - p0);
    assert_tail_is_not_a_chunk(tail_len);
    // One request per window the utterance runs; a repeated length is a hit.
    let requests = front_set.len() + parts.len() * (groups.len() - 1);
    let mut spans: Vec<usize> = parts.iter().map(|&(p0, p1, _, _)| p1 - p0).collect();
    spans.sort_unstable();
    spans.dedup();
    let before = rss();
    let mut cache = graphs::SoundGraphs::default();
    for round in 0..2 {
        let t = Tick::now();
        graphs::FrontGraphs::build(
            &mut cache,
            device,
            &pair.sound,
            MAX_CODEC_ROWS,
            groups[0].clone(),
        )
        .expect("the front's graphs");
        for group in &groups[1..] {
            for &len in &spans {
                cache
                    .graph(device, &pair.sound, len, group.clone(), 0)
                    .expect("a decoder graph");
            }
        }
        // The build's own elapsed time into a local before anything else: `grown`
        // shells out to `ps`, and a fork inside the argument list would be timed as
        // part of the build.
        let build_ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
        let (held, params, device_bytes) = cache.census();
        println!(
            "census decoder round={round} graphs={held} requests={requests} \
             params_mb={:.1} planned_arenas_mb={:.1} planned_held_mb={:.1} \
             rss_mb={} build_ms={:.3}",
            mb(params),
            mb(device_bytes.saturating_sub(params)),
            mb(device_bytes),
            grown(before, rss()),
            build_ms,
        );
    }
    drop(cache);
    println!("census decoder released rss_mb={}", grown(before, rss()));
    // The front's graph set per part length, through the builder the sound half
    // uses: every part builds its first window, and a part longer than one chunk
    // the full chunk its later windows run — never a graph those windows cannot
    // dispatch.
    let carry = graphs::FRONT_CARRY;
    for frames in [FRONT_WINDOW, 100, 2 * FRONT_WINDOW, MAX_CODEC_ROWS] {
        let mut fresh = graphs::SoundGraphs::default();
        graphs::FrontGraphs::build(&mut fresh, device, &pair.sound, frames, groups[0].clone())
            .expect("a front graph");
        let (held, params, device_bytes) = fresh.census();
        println!(
            "census decoder front frames={frames} graphs={held} carry={carry} params_mb={:.1} \
             arena_mb={:.1}",
            mb(params),
            mb(device_bytes.saturating_sub(params)),
        );
    }
    // What one front graph costs to build, in the window the answer is read in:
    // the tail's length against a full chunk's, alternately and in fresh caches.
    // The tail's length is the graph the shipped builder no longer builds; it is
    // built here only to price the alternative. Each arm's median is its own
    // figure — the trade reads the tail's against what a padded last window costs
    // to run, not one median against the other.
    let (mut tail_ms, mut chunk_ms) = (Vec::new(), Vec::new());
    for round in 0..BUILD_ROUNDS {
        let order: [usize; 2] = if round % 2 == 0 {
            [tail_len, FRONT_WINDOW]
        } else {
            [FRONT_WINDOW, tail_len]
        };
        for len in order {
            let mut fresh = graphs::SoundGraphs::default();
            let t = Tick::now();
            let g = fresh
                .graph(device, &pair.sound, len, groups[0].clone(), carry)
                .expect("a front graph");
            let ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
            drop(g);
            if len == tail_len {
                tail_ms.push(ms);
            } else {
                chunk_ms.push(ms);
            }
        }
    }
    let (t_lo, t_hi) = (support::least(&tail_ms), support::greatest(&tail_ms));
    let (c_lo, c_hi) = (support::least(&chunk_ms), support::greatest(&chunk_ms));
    println!(
        "census front_build rounds={BUILD_ROUNDS} tail_len={tail_len} tail_ms={:.3} \
         tail_spread={t_lo:.3}..{t_hi:.3} chunk_ms={:.3} chunk_spread={c_lo:.3}..{c_hi:.3}",
        support::median(&tail_ms),
        support::median(&chunk_ms),
    );
}

/// The front's last window as the shipped arrangement runs it — a full chunk
/// padded with code 0 — against the same frames at their true length through a
/// graph of their own: what the tree pays per utterance for not building a graph per
/// tail length, and the number the tail-graph trade was decided on.
#[test]
#[ignore = "measurement only: prints the padded tail window against a graph of its own length"]
fn the_fronts_last_window_runs_padded() {
    let Some(pair) = pair_or_skip() else { return };
    let device = Device::Gpu;
    let front = graphs::codec_groups()[0].clone();
    let part = windows(MAX_CODEC_ROWS, FRONT_WINDOW, 0);
    let &(_, _, tail_at, tail_end) = part.last().expect("a part has a window");
    let tail_len = tail_end - tail_at;
    assert_tail_is_not_a_chunk(tail_len);
    let mut cache = graphs::SoundGraphs::default();
    let fronts = graphs::FrontGraphs::build(
        &mut cache,
        device,
        &pair.sound,
        MAX_CODEC_ROWS,
        front.clone(),
    )
    .expect("the front's graphs");
    // The two lengths side by side: what the shipped arrangement runs (the padded
    // chunk) and the graph of the tail's own length it does not build.
    let padded = fronts.pick(tail_at).0.clone();
    let carry = graphs::FRONT_CARRY;
    let tail = cache
        .graph(device, &pair.sound, tail_len, front, carry)
        .expect("the tail's own graph");
    println!(
        "census front_tail tail_len={tail_len} chunk={FRONT_WINDOW} carry={carry} \
         rounds={ROUNDS} windows={}",
        part.len(),
    );
    let (padded_in, tail_in) = (
        support::filled_inputs(&padded.borrow().runner.graph, 0.1),
        support::filled_inputs(&tail.borrow().runner.graph, 0.1),
    );
    run_ms(&padded, &padded_in);
    run_ms(&tail, &tail_in);
    let mut deltas = Vec::with_capacity(ROUNDS);
    let (mut p_ms, mut t_ms) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        // Alternate which arm runs first, so a within-round drift lands on both
        // arms rather than on the delta; the sign stays padded minus tail in
        // both orders.
        let (p, t) = if round % 2 == 0 {
            (run_ms(&padded, &padded_in), run_ms(&tail, &tail_in))
        } else {
            let true_ms = run_ms(&tail, &tail_in);
            (run_ms(&padded, &padded_in), true_ms)
        };
        p_ms.push(p);
        t_ms.push(t);
        deltas.push(p - t);
    }
    let (p_lo, p_hi) = (support::least(&p_ms), support::greatest(&p_ms));
    let (t_lo, t_hi) = (support::least(&t_ms), support::greatest(&t_ms));
    let (d_lo, d_hi) = (support::least(&deltas), support::greatest(&deltas));
    println!(
        "census front_tail padded_ms={:.3} true_ms={:.3} delta_ms={:.3} \
         padded_spread={p_lo:.3}..{p_hi:.3} true_spread={t_lo:.3}..{t_hi:.3} \
         delta_spread={d_lo:.3}..{d_hi:.3} wins={}/{}",
        support::median(&p_ms),
        support::median(&t_ms),
        support::median(&deltas),
        deltas.iter().filter(|d| **d > 0.0).count(),
        ROUNDS,
    );
}

/// The decoder's windows carry no right margin: a span padded past its core on the
/// right computes outputs the stitch discards. Paired per DAC group at the part's
/// own two full windows — the span with the margin against the span without it.
#[test]
#[ignore = "measurement only: prints the decoder's windows with and without their right margin"]
fn the_decoders_windows_carry_no_right_margin() {
    let Some(pair) = pair_or_skip() else { return };
    let device = Device::Gpu;
    let groups = graphs::codec_groups();
    // The spans the run makes, and the spans the same windows carried while the
    // right margin was on — four frames more on every window but the part's own
    // tail, which already ends at the part's end.
    let cuts = windows(MAX_CODEC_ROWS, DAC_WINDOW, DAC_MARGIN);
    let spans: Vec<usize> = cuts.iter().map(|&(p0, p1, _, _)| p1 - p0).collect();
    // Both sides through the window function, from the one cut: the margin arm is
    // the same core with `DAC_MARGIN` frames on its right, clipped at the part's end.
    let padded: Vec<usize> = cuts
        .iter()
        .map(|&(_, _, a, b)| {
            let (p0, p1) = graphs::window_span(MAX_CODEC_ROWS, a, b, DAC_MARGIN, DAC_MARGIN);
            p1 - p0
        })
        .collect();
    println!(
        "census dac_margin groups={} spans={spans:?} with_margin={padded:?} rounds={ROUNDS}",
        groups.len() - 1,
    );
    let mut utterance = 0.0;
    for group in &groups[1..] {
        let mut group_ms = 0.0;
        for (with, without) in padded.iter().zip(&spans) {
            if with == without {
                continue;
            }
            // One pair's graphs at a time: a cache that held every span of both
            // groups would be several gigabytes for a probe that only pairs two.
            let mut cache = graphs::SoundGraphs::default();
            let a = cache
                .graph(device, &pair.sound, *with, group.clone(), 0)
                .expect("a padded decoder window");
            let b = cache
                .graph(device, &pair.sound, *without, group.clone(), 0)
                .expect("a decoder window");
            let (a_in, b_in) = (
                support::filled_inputs(&a.borrow().runner.graph, 0.1),
                support::filled_inputs(&b.borrow().runner.graph, 0.1),
            );
            run_ms(&a, &a_in);
            run_ms(&b, &b_in);
            let mut deltas = Vec::with_capacity(ROUNDS);
            for round in 0..ROUNDS {
                // Alternate which arm runs first, so a within-round drift lands on
                // both arms rather than on the delta; the sign stays the margin's
                // own (with-margin minus without-margin) in both orders.
                let delta = if round % 2 == 0 {
                    let (with_ms, without_ms) = (run_ms(&a, &a_in), run_ms(&b, &b_in));
                    with_ms - without_ms
                } else {
                    let (without_ms, with_ms) = (run_ms(&b, &b_in), run_ms(&a, &a_in));
                    with_ms - without_ms
                };
                deltas.push(delta);
            }
            // The per-window absolute costs move between runs; the paired delta and
            // its agreeing rounds are the figure this probe is read for.
            let delta = support::median(&deltas);
            let (lo, hi) = (support::least(&deltas), support::greatest(&deltas));
            group_ms += delta;
            println!(
                "census dac_margin group={}..{} span={with}->{without} delta_ms={delta:.3} \
                 delta_spread={lo:.3}..{hi:.3} wins={}/{}",
                group.start,
                group.end,
                deltas.iter().filter(|d| **d > 0.0).count(),
                ROUNDS,
            );
        }
        utterance += group_ms;
        println!(
            "census dac_margin group={}..{} utterance_ms={group_ms:.3}",
            group.start, group.end
        );
    }
    println!("census dac_margin utterance_ms={utterance:.3}");
}

/// The served path: one process, two calls of the pinned text and a third of
/// another, through `Synthesis`, which is the path the wake-word bench walks. The
/// second call is the reuse the decoder cache exists for; the resident set after
/// each call is what the retained graphs cost against the guard's ceiling.
#[test]
#[ignore = "measurement only: two synthesis calls of one utterance through the library path"]
fn the_library_path_pays_the_decoder_once() {
    let Some(pair) = pair_or_skip() else { return };
    let voice = match loader::voice_wav() {
        Ok(voice) => voice,
        Err(e) => {
            eprintln!("skip: {e} — the probe did not run");
            return;
        }
    };
    let t = Tick::now();
    let mut synthesis = Synthesis::new(Arc::new(pair), &voice).expect("the synthesis object");
    println!(
        "census library open_ms={:.1} rss_mb={}",
        Tick::now().elapsed_ns(t) as f64 / 1e6,
        mb_of(rss())
    );
    // Two calls of the pinned text — the row's own utterance — and one of another
    // so the cache meets a second length the way the bench's clips do. The repeat
    // must build nothing and reproduce the first call's waveform: both are the
    // claims the decoder cache and the tail graph stand on, so both are asserted.
    let other = "Sit down and tell me what you saw on the way here.";
    let mut pinned = [0u64; 2];
    for (call, text) in [(0usize, TTS_TEXT), (1, TTS_TEXT), (2, other)] {
        let t = Tick::now();
        let wave = synthesis.synthesize(text, 12345).expect("a synthesis call");
        let ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
        let build = synthesis.build_ms();
        let digest = report::digest(&wave);
        println!(
            "census library call={call} chars={} ms={ms:.1} build_ms={build:.3} samples={} \
             digest={digest:016x} rss_mb={} process_peak={}",
            text.chars().count(),
            wave.len(),
            mb_of(rss()),
            report::shown(report::peak_rss()),
        );
        if call < 2 {
            pinned[call] = digest;
        }
        if call == 0 {
            assert!(
                build > 0.0,
                "the first call of a length must build its graphs"
            );
        } else if call == 1 {
            // A hit still walks the cache (microseconds); a compile or an upload
            // would be two orders of magnitude more.
            assert!(
                build < 1.0,
                "a repeat of a length must pay no build, read {build:.3} ms"
            );
        }
    }
    assert_eq!(
        pinned[0], pinned[1],
        "a repeat of a text must reproduce its waveform"
    );
}
