// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measurement only and ignored: what the wgpu conv routes' spatial floor
//! (`RLX_WGPU_IM2COL_MIN_SPATIAL`) costs the synthesis front at 1 against the 100 it
//! came down from, and what a floor of 1 newly admits in the rest of the product.
//!
//! Needs the installed model directory; a machine without it skips, and a pair that
//! is present but will not open fails. Run with `RLX_ALLOW_THROTTLE=1 cargo test -p
//! mahbot-inference --release --features hand-run --test synthesis_front_route --
//! --ignored --nocapture <subject>`, **one subject at a time**: the floor is a
//! process-global lever, so two subjects running together would build their arms under
//! each other's floor and still print plausible numbers. Each subject holds a guard for
//! its body, so a run with no subject named fails rather than reads, while a sequential
//! run of both (`--test-threads=1`) is fine.
//!
//! **The protocol.** Both arms are taken in one binary through the floor's own
//! runtime lever, so the comparison carries no build-to-build scatter: each graph is
//! compiled once with the key at one value and once at the other, the two compiled
//! graphs are timed alternately — `ROUNDS` rounds, the arm that runs first swapping
//! each round — with one warm-up pass per arm discarded, and each arm's median is its
//! figure. The `delta_ms` beside them is the median of the per-round paired
//! differences, not the difference of the two medians: the pairing is what it is for.
//! The floor is read at plan and at lowering time, so a compiled graph keeps the
//! route it was built with and neither arm can drift into the other's.
//!
//! **The front's own cost.** One front-group graph (`codec_groups()[0]`: the RVQ
//! decode, the wave transformer and DAC stages 0–3) at a part's own frame count,
//! which is what one window of that part runs. The step census beside each row is
//! the route each arm took. Memory is not read here: the col panel the materialising
//! route asks for is booked at compile time on top of the activation plan, so it is
//! `RLX_WGPU_ARENA_REPORT=1`'s "plan + scratch -> size" line — the built arena the
//! routing's record quotes — that witnesses it.
//!
//! **The rest of the product.** Every other graph that holds a convolution, at the
//! lengths the product runs it: the recognition conv front at the shortest and
//! longest rung and at the short one-piece inputs that are newly admitted (priced, and
//! read on a pinned recording as well as on the ramp), the wake ear's own window, the
//! synthesis stage groups at the spans the stand's part lengths cut, and the speaker
//! encoder at the lengths the floor newly admits. A graph whose census is identical
//! under both floors is not timed — no step changed, so nothing can move but noise —
//! and its two arms are given back before the next is built; the one such shape timed
//! anyway is the speaker encoder at 128 mel frames, the control row where both floors
//! already took the materialising route.
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use mahbot_inference::graphs::{self, DAC_WINDOW, FRONT_WINDOW};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::stand;
use mahbot_inference::tokenizer::Tokenizer;
use rlx_runtime::Device;

mod support;

/// The floor the shipped 1 came down from, and the shipped one.
const PARENT_FLOOR: usize = 100;
const SHIPPED_FLOOR: usize = 1;

/// The frame counts the front is priced at: the shipped row's own length (23), the two
/// the floor's move was first reported at (18 and 20), the reference lengths that must
/// not rise (42, 72), the chunk boundary (72 = [`FRONT_WINDOW`]), and the short lengths
/// a lower floor most plausibly loses at — one, two and three frames are also where the
/// stage-3 residual lifts are the shape the floor newly admits, and three is the largest
/// panel the move books on its own.
const LENGTHS: [usize; 15] = [1, 2, 3, 8, 16, 18, 20, 22, 23, 24, 26, 32, 42, 60, 72];

/// The frame counts a one-piece recognition input reaches the conv front at. The third
/// convolution's extent is `16 · asr_tokens(frames)` and the second's `32 · t2`, so the
/// third is under the old floor of 100 at 48 frames and below and the second at 12 and
/// below — both clearing the K and channel floors there. Twelve is the second's own
/// crossover, measured rather than derived. The product gets these shapes from
/// `transcribe::read_whole`, which hands `frames_of(len)` straight to `Clip::once` for
/// an input that fits one piece.
const SHORT_ASR: [usize; 6] = [1, 8, 12, 16, 32, 48];

/// The mel frame counts the short one-piece witness reads on real audio: the second
/// convolution's own crossover (12, where that stride-2 conv is the one newly admitted)
/// and 48, the largest shape the floor newly admits (0.48 s of audio), where a reordered
/// sum would show first.
const SHORT_REAL: [usize; 2] = [12, 48];
/// Where in that recording the 0.48 s is taken from: past the lead-in, so the snippet is
/// speech rather than room tone.
const SHORT_OFFSET_SECS: usize = 60;

/// Rounds each arm is timed over; the median is the arm's figure and the paired
/// deltas' spread is the comparison's own.
const ROUNDS: usize = 9;

/// Rounds a shape outside the front is timed over: the same protocol, fewer of them,
/// because those graphs are not where the decision is made.
const QUIET_ROUNDS: usize = 5;

/// The floor is a process-global lever, so a subject runs alone: a second one in the
/// same process would build its arms under the other's floor and print plausible
/// numbers for a route neither arm took. The guard is held for the subject's body, so a
/// sequential run of both is fine.
fn solo(subject: &str) -> MutexGuard<'static, ()> {
    static RUNNING: Mutex<()> = Mutex::new(());
    match RUNNING.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::WouldBlock) => panic!(
            "{subject}: another subject of this file is running in this process, and the \
             floor lever is process-global — name one subject"
        ),
        // A subject that panicked has already failed the run; the lock is still held
        // by nobody, so the next one may proceed.
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
    }
}

/// Run `body` with the floor's key at `floor`, then put the key back to what it held
/// (unset if it held nothing): a graph it builds keeps the route it was built with, and a
/// walk it runs compiles and runs under it. The key is process-global, so the restore is
/// a guard — a panic in the body must not leave the process measuring under a floor
/// nobody asked for, and a nested call must not drop the outer arm's floor.
fn at_floor<T>(floor: usize, body: impl FnOnce() -> T) -> T {
    const KEY: &str = "RLX_WGPU_IM2COL_MIN_SPATIAL";
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match &self.0 {
                Some(previous) => rlx_ir::env::set(KEY, previous.clone()),
                None => rlx_ir::env::unset(KEY),
            }
            rlx_wgpu::reload_runtime_config();
        }
    }
    let previous = rlx_ir::env::var(KEY);
    rlx_ir::env::set(KEY, floor.to_string());
    rlx_wgpu::reload_runtime_config();
    let _restore = Restore(previous);
    body()
}

/// The pair a role reads: files that are not on this machine skip, and a present pair
/// that will not open fails — a subject that skipped on a broken model would let this
/// file's evidence vanish while the run read `ok`.
fn pair_of(role: Role) -> Option<Arc<Pair>> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return None;
    }
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("skip: {e:#} — the probe did not run");
            return None;
        }
    };
    let (body, sound) = loader::pair_files(role).expect("every role is pinned");
    for name in [body, sound] {
        let path = dir.join(name);
        if !path.is_file() {
            eprintln!("skip: {} is not on this machine", path.display());
            return None;
        }
    }
    Some(Arc::new(
        Pair::open(dir.join(body), dir.join(sound))
            .unwrap_or_else(|e| panic!("{}: {e:#}", dir.display())),
    ))
}

/// The recognition conv front at `frames`.
fn asr_front(pair: &Arc<Pair>, frames: usize) -> graphs::Compiled {
    graphs::Compiled::new(
        Device::Gpu,
        graphs::asr_encoder_front(pair.sound.clone(), frames).expect("the conv front"),
    )
    .expect("the conv front compiles")
}

/// One front-group graph at `frames`.
fn build_front(pair: &Arc<Pair>, frames: usize) -> graphs::Compiled {
    graphs::sound_graph(
        Device::Gpu,
        &pair.sound,
        frames,
        graphs::codec_groups()[0].clone(),
        0,
    )
    .expect("the front's graph")
}

/// One graph built under both floors and held together: its two censuses and its
/// paired cost over `rounds`.
fn paired(label: &str, rounds: usize, build: impl Fn() -> graphs::Compiled) {
    let mut high = at_floor(PARENT_FLOOR, &build);
    let mut low = at_floor(SHIPPED_FLOOR, &build);
    let (hi, lo) = (
        support::schedule_census(&high),
        support::schedule_census(&low),
    );
    let hi_in = support::filled_inputs(&high.runner.graph, 0.1);
    let lo_in = support::filled_inputs(&low.runner.graph, 0.1);
    support::frame_once(&mut high, &hi_in);
    support::frame_once(&mut low, &lo_in);
    let (mut hi_ms, mut lo_ms, mut deltas) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..rounds {
        // Alternate which arm runs first, so a within-round drift lands on both
        // arms rather than on the delta; the sign stays high minus low in both
        // orders.
        let (h, l) = if round % 2 == 0 {
            (
                support::frame_once(&mut high, &hi_in),
                support::frame_once(&mut low, &lo_in),
            )
        } else {
            let l = support::frame_once(&mut low, &lo_in);
            (support::frame_once(&mut high, &hi_in), l)
        };
        hi_ms.push(h);
        lo_ms.push(l);
        deltas.push(h - l);
    }
    // The value check runs on the deterministic ramp rather than the fill the timing
    // uses: a constant fill cannot show a reordered sum.
    let hi_out = high.run(&support::ramped_inputs(&high.runner.graph));
    let lo_out = low.run(&support::ramped_inputs(&low.runner.graph));
    let (differing, worst) = support::output_difference(&hi_out, &lo_out);
    println!(
        "census {label} rounds={rounds} hi_ms={:.3} lo_ms={:.3} delta_ms={:.3} \
         hi_spread={:.3}..{:.3} lo_spread={:.3}..{:.3} delta_spread={:.3}..{:.3} \
         wins={}/{} differing={differing} max_abs={worst:.3e}",
        support::median(&hi_ms),
        support::median(&lo_ms),
        support::median(&deltas),
        support::least(&hi_ms),
        support::greatest(&hi_ms),
        support::least(&lo_ms),
        support::greatest(&lo_ms),
        support::least(&deltas),
        support::greatest(&deltas),
        deltas.iter().filter(|d| **d > 0.0).count(),
        rounds,
    );
    print_steps(label, &hi, &lo);
}

/// The two censuses under one label, in the order every row kind prints them.
fn print_steps(label: &str, hi: &str, lo: &str) {
    println!("census {label} hi_steps=[{hi}]");
    println!("census {label} lo_steps=[{lo}] changed={}", hi != lo);
}

/// One graph built under each floor in turn and given back before the next is built:
/// its two censuses. For the graphs whose arena is too large to hold twice, and where a
/// step census is the whole question.
fn census_only(label: &str, build: impl Fn() -> graphs::Compiled) {
    let high = at_floor(PARENT_FLOOR, &build);
    let hi = support::schedule_census(&high);
    drop(high);
    let low = at_floor(SHIPPED_FLOOR, &build);
    let lo = support::schedule_census(&low);
    print_steps(label, &hi, &lo);
}

/// The front's own cost at every frame count: what the floor's spatial minimum
/// costs a part of that length, in one binary, one graph per arm.
#[test]
#[ignore = "measurement only: prices the synthesis front at every frame count, both floors"]
fn the_front_costs_its_frames() {
    let _solo = solo("the_front_costs_its_frames");
    let Some(pair) = pair_of(Role::Synthesis) else {
        return;
    };
    println!(
        "census front rounds={ROUNDS} floors={PARENT_FLOOR}->{SHIPPED_FLOOR} \
         window={FRONT_WINDOW} lengths={LENGTHS:?}"
    );
    for &frames in &LENGTHS {
        paired(&format!("front frames={frames}"), ROUNDS, || {
            build_front(&pair, frames)
        });
    }
}

/// The shapes the lower floor newly admits outside the front, at the lengths the
/// product runs them: recognition's conv front, the wake ear, the synthesis stage
/// groups, and the speaker encoder.
#[test]
#[ignore = "measurement only: what the lower floor admits in the other product graphs"]
fn what_the_lower_floor_admits_outside_the_front() {
    let _solo = solo("what_the_lower_floor_admits_outside_the_front");
    let groups = graphs::codec_groups();
    // Each role's rows stand on that role's own pair: a machine with one pair and not
    // the other loses only the half it cannot measure, rather than reporting `ok` for a
    // run that measured nothing.
    if let Some(recognition) = pair_of(Role::Recognition) {
        // The conv front at the shortest and the longest rung, where the census is the
        // whole question, and at the short one-piece inputs the floor newly admits,
        // which are priced because their route does change.
        for frames in [128usize, 768] {
            census_only(&format!("asr_front frames={frames}"), || {
                asr_front(&recognition, frames)
            });
        }
        for &frames in &SHORT_ASR {
            paired(&format!("asr_front frames={frames}"), QUIET_ROUNDS, || {
                asr_front(&recognition, frames)
            });
        }
        // The wake ear: its own window, the same front at 76 mel frames.
        census_only("asr_front frames=76 (wake)", || asr_front(&recognition, 76));
        // The short one-piece shapes on real audio, which the stand never reaches: a walk
        // of the pinned recording's first `SHORT_REAL` mel frames under both floors. Their
        // route moved, so their values are read here rather than taken from the ramp —
        // including the 12-frame cell, where the stride-2 convolution is the one that is
        // newly admitted and a 48-frame walk would not reach it.
        if let Some(path) = support::recording(stand::RECOGNITION_RECORDING) {
            let tokenizer =
                Tokenizer::from_model(&recognition.body).expect("the recognition tokenizer");
            for &frames in &SHORT_REAL {
                let mut audio = mahbot_inference::host::wav_mono_16k(
                    &path.display().to_string(),
                    SHORT_OFFSET_SECS * 16_000,
                    frames * graphs::MEL_HOP,
                )
                .unwrap_or_else(|e| panic!("reading {}: {e:#}", path.display()));
                support::prepare_speech(&mut audio);
                let walk = |floor: usize| {
                    at_floor(floor, || {
                        mahbot_inference::listen::recognise(&recognition, &tokenizer, audio.clone())
                            .expect("recognising the short input")
                    })
                };
                let (hi, lo) = (walk(PARENT_FLOOR), walk(SHIPPED_FLOOR));
                println!(
                    "census asr_front short_real frames={frames} hi_ids={} lo_ids={} \
                     ids_equal={} values_equal={} text_equal={} text={:?}",
                    hi.generated.len(),
                    lo.generated.len(),
                    hi.generated == lo.generated,
                    hi.values == lo.values,
                    hi.text == lo.text,
                    hi.text,
                );
            }
        }
    }
    // The recognition half's process-wide caches — the decode slot's `Arc` holding the ASR
    // decode graph, and the mel graph maps — outlive the dropped pair, and the synthesis
    // half builds graphs of up to half a gigabyte under them.
    mahbot_inference::release_process_memory();
    let Some(pair) = pair_of(Role::Synthesis) else {
        return;
    };
    // The synthesis stage groups at the spans the stand's part lengths cut: 23 and
    // 90 frames (the two pinned rows), and the harness cell's 330.
    for frames in [23usize, 90, 330] {
        for span in graphs::windows(frames, DAC_WINDOW, graphs::DAC_MARGIN) {
            let len = span.1 - span.0;
            for group in &groups[1..] {
                census_only(
                    &format!(
                        "stage_group {}..{} frames={frames} span={len}",
                        group.start, group.end
                    ),
                    || {
                        graphs::sound_graph(Device::Gpu, &pair.sound, len, group.clone(), 0)
                            .expect("a decoder graph")
                    },
                );
            }
        }
    }
    // The stage group whose stage reads the shortest input there is: stage 4 is handed
    // 32 frames of columns, so a part of three gives it 96 — which the 100 default
    // refused — but the group's own convolutions run after that stage's ×5 upsampler,
    // at 480, so they clear either floor and only the census is read here.
    for frames in [3usize, 4] {
        census_only(&format!("stage_group 4..6 frames={frames}"), || {
            graphs::sound_graph(Device::Gpu, &pair.sound, frames, groups[1].clone(), 0)
                .expect("a decoder graph")
        });
    }
    // The speaker encoder, whose first convolution is a unit-height lift at the
    // reference's mel frame count: the lengths the floor newly admits — the whole band
    // below 100, down to a reference shorter than a second — and one it already admitted.
    let out = pair.body.config().d;
    for frames in [16usize, 32, 64, 99, 128] {
        paired(&format!("speaker frames={frames}"), QUIET_ROUNDS, || {
            graphs::Compiled::new(
                Device::Gpu,
                graphs::speaker_encoder(pair.sound.clone(), frames, out)
                    .expect("the speaker encoder"),
            )
            .expect("the speaker encoder compiles")
        });
    }
}
