// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and `#[ignore]`d: the cell's pinned waveform through the
//! library path, the shipped predictor frame's time with its steps counted by
//! kind, and the sixteen-level stack's rate against sixteen single-row reads.
//!
//! **The waveform.** `Synthesis::synthesize` on the cell's own text and seed,
//! printed against the pinned digest as a witness: the stand walks its own copy of
//! the loop, so this is where the library path's waveform is seen, and a moved
//! digest is the record's line, not a failure here.
//!
//! **The frame.** One row, one compile, the cell's own sampler: the frame is
//! `steps + 1` levels over a carried plane cache, and its schedule is counted by
//! step. The counts are not a price — `multiply_census` stands the frame's kinds
//! down one at a time for that, each arm's delta taken against the shipped frame.
//!
//! **The stack.** Every level reads the same weight stack. In one pass the
//! row-grouped route serves sixteen rows, measured here against the sixteen
//! single-row reads the serial walk pays — the figure a batched predictor would
//! collect if the frames were independent, which they are not ([`graphs`]'
//! `tts_code_predictor_frame`).
//!
//! Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… RLX_ALLOW_THROTTLE=1 cargo
//! test -p mahbot-inference --release --test predictor_draft -- --ignored
//! --nocapture`.

use std::sync::Arc;

use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::synth::Synthesis;
use mahbot_inference::{graphs, stand};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{Op, Tick};

mod support;

use support::{
    arm_once, call_of, frame_arm, last_dim, least, median, packed_ramp, step_census, steps_for,
};

/// The stand's speak seed: the frame this file measures is the cell's frame.
const SEED: u64 = 12_345;
/// The gather index-add fold's arm the frame is timed under, installed around its
/// compile and printed with the frame, so the number carries the arm it belongs to.
/// `0` leaves the fold on — the value is the *disable* switch's, so the label says so.
const GATHER_ARM: &str = "0";
/// Pairs of rounds a shape's two calls are measured in. Each pair runs both orders,
/// so a monotone drift cancels inside the pair and the marginal is a median of
/// order-symmetric differences.
const ROUNDS: usize = 3;
/// Rows one stacked reading serves: the sixteen levels a predictor frame walks, so
/// this is one whole level-set's weights. The route tiles eight rows a weight read,
/// so sixteen rows are two tiles — the traffic is the packed bytes times
/// `ceil(rows / 8)`, not the bytes once.
const STACK_ROWS: usize = 16;

/// One weight read's own time at `rows` rows: two calls of the same shape at
/// `steps` and `2 * steps` copies, the second call's added bytes over the time it
/// added. Each round runs both orders, so neither call is always the one running
/// second.
fn read_ms(
    scheme: QuantScheme,
    k: usize,
    n: usize,
    rows: usize,
    bytes: &[u8],
    steps: usize,
) -> f64 {
    let mut small = call_of(scheme, k, n, rows, bytes, steps);
    let mut large = call_of(scheme, k, n, rows, bytes, 2 * steps);
    let mut added = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let small_first = {
            let small_ms = small.ms();
            large.ms() - small_ms
        };
        let large_first = {
            let large_ms = large.ms();
            large_ms - small.ms()
        };
        added.push(0.5 * (small_first + large_first));
    }
    median(&added) / steps as f64
}

/// One shape's two readings: a single row, and the sixteen rows the stack serves as
/// two tiles of the row-grouped route. Milliseconds are the measurement; the effective rate
/// printed beside them is packed bytes over time, not a stream rate, since the
/// grouped route's own traffic is the packed bytes times `ceil(rows / 8)`.
fn shape_ms(scheme: QuantScheme, k: usize, n: usize, packed: u64) -> (f64, f64) {
    let bytes = packed_ramp(scheme, packed as usize);
    let steps = steps_for(packed, support::TARGET_BYTES);
    let flops = 2.0 * k as f64 * n as f64;
    let mut ms = [0.0f64; 2];
    for (slot, rows) in [1usize, STACK_ROWS].into_iter().enumerate() {
        let t = read_ms(scheme, k, n, rows, &bytes, steps);
        ms[slot] = t;
        // Effective, not streamed: the packed bytes the route reads, times the tiles
        // they are read in.
        let tiles = rows.div_ceil(8) as f64;
        println!(
            "draft rows scheme={scheme} k={k} n={n} rows={rows} ms={t:.4} \
             eff_GBps={:.1} eff_TFLOPs={:.2}",
            tiles * packed as f64 / (t * 1e6),
            rows as f64 * flops / (t * 1e-3) / 1e12
        );
    }
    (ms[0], ms[1])
}

#[test]
#[ignore = "measurement only: needs the synthesis pair and the pinned voice"]
fn the_predictor_frame_its_chain_and_the_stacks_ratio() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("skip: {e} — the probe did not run");
            return;
        }
    };
    let pair = match Pair::of_role(Role::Synthesis) {
        Ok(p) => Arc::new(p),
        Err(err) => {
            eprintln!("skip: {} names no synthesis pair ({err})", dir.display());
            return;
        }
    };
    let voice = match loader::voice_wav() {
        Ok(voice) => voice,
        Err(e) => {
            eprintln!("skip: {e} — the probe did not run");
            return;
        }
    };

    // The shipped walk's waveform, through the library path the bench's corpus
    // synthesis uses. The pin is a **witness**: a moved digest is recorded with the
    // run that moved it, never rolled back to (`reference.rs` carries the rule and
    // the transcript, the finiteness and the PCM range are the hard gates).
    let mut synthesis =
        Synthesis::new(pair.clone(), &voice).expect("compile the synthesis pipeline");
    let t = Tick::now();
    let wave = synthesis
        .synthesize(stand::TTS_TEXT, SEED)
        .expect("the shipped walk's sound half");
    let synth_ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
    let digest = mahbot_inference::report::digest(&wave);
    let pinned = mahbot_inference::reference::SPEAK_WAVE_DIGEST;
    println!(
        "draft wave digest={digest:016x} samples={} pinned={pinned:016x} equal={} \
         voice_pinned={} synth_ms={synth_ms:.1}",
        wave.len(),
        digest == pinned,
        mahbot_inference::reference::voice_is_pinned(),
    );
    drop(synthesis);

    // The cell's own frame: its shapes, its time and its steps by kind. The fold's
    // switch is pinned, so an ambient value cannot move the baseline.
    let steps = graphs::code_predictor_steps(&pair.sound).expect("predictor head count");
    let (top_k, top_p, temperature) = pair.body.sampling();
    let spec = graphs::SampleSpec {
        top_k,
        top_p,
        temperature,
        seed: SEED,
    };
    let (mut frame, frame_set) = frame_arm(
        &pair,
        steps + 1,
        spec,
        &[("RLX_WGPU_NO_GATHER_INDEX_BIAS", GATHER_ARM)],
    );
    let mut shapes: Vec<((QuantScheme, usize, usize), (u64, usize))> = Vec::new();
    for node in frame.runner.graph.nodes() {
        let Op::DequantMatMul { scheme } = node.op else {
            continue;
        };
        let Some(&weight) = node.inputs.get(1) else {
            continue;
        };
        if !matches!(frame.runner.graph.node(weight).op, Op::Param { .. }) {
            continue;
        }
        let Some(bytes) = frame.runner.graph.node(weight).shape.size_bytes() else {
            continue;
        };
        let key = (
            scheme,
            last_dim(&frame.runner.graph.node(node.inputs[0]).shape),
            last_dim(&node.shape),
        );
        match shapes.iter_mut().find(|(seen, _)| *seen == key) {
            Some((_, entry)) => entry.1 += 1,
            None => shapes.push((key, (bytes as u64, 1))),
        }
    }
    let inputs = support::zero_inputs(&frame.runner.graph);
    // The arm's own switch values are installed around every run: the run-time
    // ones (`GEMV_*`) are read at the dispatch, so an ambient value would other-
    // wise move this figure without moving the arm's label.
    arm_once(&mut frame, &frame_set, &inputs);
    let samples: Vec<f64> = (0..ROUNDS)
        .map(|_| arm_once(&mut frame, &frame_set, &inputs))
        .collect();
    let frame_ms = least(&samples);
    // The walk covers `steps + 1` levels — level 0 is the head-less body — and the
    // per-level figure is the stand's convention, one convention for both.
    let levels = steps + 1;
    println!(
        "draft frame gather_index_bias_disabled={GATHER_ARM} ms={frame_ms:.3} levels={levels} \
         nodes={} frame_per_level_ms={:.3}",
        frame.runner.graph.nodes().len(),
        frame_ms / levels as f64
    );

    // What the frame's chain is made of, by step: the compiled schedule's own names,
    // fusion and planning included, one dispatch each. This is the half of the frame
    // the weight stream does not pay for — every normalisation, rotation, attention,
    // cache append, view and draw the sixteen levels walk.
    {
        let names = frame
            .runner
            .schedule_step_names()
            .expect("the frame's schedule is compiled");
        println!(
            "draft chain kinds={} total_steps={}",
            step_census(&names),
            names.len()
        );
    }
    drop(frame);

    // The stack: one packed read at one row and at sixteen, inside this window,
    // summed over the frame's own distinct shapes by each shape's share of a
    // level. `s_meas` is one level's marginal; sixteen serial levels read the
    // stack sixteen times where the grouped route serves all sixteen rows once.
    let (mut s_meas, mut flops_per_level, mut total_16) = (0.0f64, 0.0f64, 0.0f64);
    for &((scheme, k, n), (packed, count)) in &shapes {
        let (t1, t16) = shape_ms(scheme, k, n, packed);
        let share = count as f64 / levels as f64;
        s_meas += t1 * share;
        flops_per_level += 2.0 * k as f64 * n as f64 * share;
        total_16 += t16 * share;
    }
    let serial = s_meas * levels as f64;
    let serial_tflops = flops_per_level / (s_meas * 1e-3) / 1e12;
    // The stack's rate is FLOPs over the summed time, as the serial side is, so
    // the two are one comparison and not a mean of per-shape rates.
    let tflops_16 = STACK_ROWS as f64 * flops_per_level / (total_16 * 1e-3) / 1e12;
    println!(
        "draft ratio marginal_per_level_ms={s_meas:.3} serial_16x_ms={serial:.3} \
         serial_tflops={serial_tflops:.2} stack_16rows_grouped_ms={total_16:.3} \
         stack_speedup={:.2}x stack_tflops={tflops_16:.2}",
        serial / total_16,
    );
}
