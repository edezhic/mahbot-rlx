// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The transcription path on the pinned music pool: what each clip comes back
//! as, region by region, and what the call cost.
//!
//! **What it pins.** Not words — the pool has no reference transcript here — but
//! the shape the product depends on, as the library's own `Transcription::check`
//! states it: every clip tiles its own input exactly (the first region at 0, the
//! last at the file's length, no hole and no overlap) and no region comes back
//! empty, so a caller can always name what it is looking at, by a transcript or by
//! a label. Around that assertion the run prints, for every clip, each region
//! (start, end, text, label, reliable) and the `Costs` split of the call — timings
//! and memory are printed and never asserted, so a busy machine is not a
//! regression.
//!
//! **What it does not establish.** Which clip transcribes. The song rule's ground
//! truth is the split in `music::CLIPS` — nothing in `instrumental/`
//! should be transcribed, `vocal/02-pop-lyrics.wav` should be — and those
//! assertions are the harness's, taken on the regions printed below, not this
//! test's. Nor does it say anything about a label's correctness: the tagger's own
//! gate is `ced_tagger`.
//!
//! **When it runs.** It needs the recognition pair and the tagger under
//! `MAHBOT_MODELS_DIR`, the eight clips verified by `music::verify` under
//! `MAHBOT_ASR_MUSIC`, and a wgpu adapter; without any of them it prints why and
//! returns, and an input the run was pointed at that fails to verify fails the
//! test.

use std::sync::Arc;

use mahbot_inference::ced::{CED_MODEL, Tagger};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::music;
use mahbot_inference::report;
use mahbot_inference::stand;
use mahbot_inference::transcribe::Transcriber;

mod support;

/// The recognition pair and the tagger, opened from `MAHBOT_MODELS_DIR`, or the
/// reason this run cannot happen.
fn open() -> Option<(Arc<Pair>, Tagger)> {
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(error) => {
            println!("skip: {error:#}");
            return None;
        }
    };
    let (body, sound) =
        loader::pair_files(Role::Recognition).expect("the recognition role is pinned");
    for name in [body, sound, CED_MODEL] {
        let path = dir.join(name);
        assert!(
            path.is_file(),
            "{} is set and {} is not on this machine",
            loader::MODELS_DIR_ENV,
            path.display()
        );
    }
    let pair = Pair::open(dir.join(body), dir.join(sound))
        .unwrap_or_else(|error| panic!("{}: {error:#}", dir.display()));
    let tagger = Tagger::open(&dir.join(CED_MODEL))
        .unwrap_or_else(|error| panic!("{}: {error:#}", dir.join(CED_MODEL).display()));
    Some((Arc::new(pair), tagger))
}

#[test]
fn every_music_clip_transcribes_whole_with_models() {
    // The adapter first: a host that has the models but no wgpu device skips here
    // rather than panicking inside the pair's first compile, mid-way through.
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some((pair, tagger)) = open() else {
        return;
    };
    let Some(paths) = support::pool() else {
        return;
    };
    assert_eq!(
        paths.len(),
        music::CLIPS.len(),
        "the pool's verify returns one path per pin"
    );
    let open = report::Stopwatch::start();
    let mut transcriber = Transcriber::open(pair, tagger).expect("the transcriber opens");
    println!("transcriber open_ms={:.1} clips={}", open.ms(), paths.len());

    for (clip, path) in music::CLIPS.iter().zip(&paths) {
        let name = clip.path;
        let t = report::Stopwatch::start();
        let out = transcriber
            .transcribe_file(path)
            .unwrap_or_else(|error| panic!("{name}: {error:#}"));
        let wall_ms = t.ms();
        let costs = transcriber.costs();
        println!(
            "clip={name} wall_ms={wall_ms:.1} sample_rate={} samples={} duration_s={:.3} \
             gain={:.3} regions={}",
            out.sample_rate,
            out.samples,
            out.duration_s(),
            out.gain,
            out.regions.len()
        );
        for region in &out.regions {
            println!(
                "  region {:.3}..{:.3} text={:?} label={:?} reliable={}",
                out.seconds(region.start),
                out.seconds(region.end),
                region.text,
                region.label,
                region.reliable
            );
        }
        println!(
            "  costs segment_ms={:.1} tag_ms={:.1} build_ms={:.1} recognise_ms={:.1} \
             chunks={} shapes={} windows={} speech_s={:.2}",
            costs.segment_ms,
            costs.tag_ms,
            costs.build_ms,
            costs.recognise_ms,
            costs.chunks,
            costs.shapes,
            costs.windows,
            costs.speech_s
        );
        println!("  transcript={:?}", stand::spoken(&out));

        // The library's own contract in the build the test ran — the tiling, the
        // text-or-label answer, the label never beside a reliable transcript — so
        // the assertions cannot drift from what a caller is promised.
        out.check()
            .unwrap_or_else(|error| panic!("{name}: {error:#}"));
    }
    println!("peak_rss={}", report::human(report::peak_rss()));
}
