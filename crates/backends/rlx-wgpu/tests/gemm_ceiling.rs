// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The ceiling a packed multiply is judged against: this machine's own dense
//! f32 GEMM through the same path (a `Graph::matmul` through `Session` on the
//! wgpu backend), at the row count and shapes the recognition prefill's
//! projections walk — and, below it, the same multiply in the dtype the prefill
//! actually stores (`Op::DequantMatMul` over a Q4_K or Q6_K region), which is the
//! figure that decides whether that path has room. A packed multiply reading
//! 4-bit weights moves a quarter of the bytes a dense f32 one does, so it can
//! beat the dense figure; only its own row can say whether it does.
//!
//! Taken on the stand (an Apple M3 Pro, 14-core GPU, wgpu over Metal), best of
//! seven per shape after three warm-up calls:
//!
//! | m | k | n | dense f32 GFLOP/s |
//! |---|---|---|---|
//! | 219 | 1024 | 6144 | 815-994 |
//! | 219 | 2432 | 1024 | 632-718 |
//! | 219 | 1024 | 2048 | 531-705 |
//! | 512 | 512 | 512 | 302-378 |
//!
//! | scheme | m | k | n | packed B | GFLOP/s |
//! |---|---|---|---|---|---|
//! | Q4_K | 219 | 2048 | 6144 | 7 077 888 | 1 100 |
//! | Q4_K | 219 | 1024 | 2048 | 1 179 648 | 759 |
//! | Q4_K | 1 | 2048 | 6144 | 7 077 888 | 95 (call floor) |
//! | Q6_K | 1 | 2048 | 151 936 | 255 252 480 | 180 (73.9 GB/s) |
//! | Q6_K | 219 | 2048 | 151 936 | 255 252 480 | 945 |
//!
//! The recognition prefill runs 745 GFLOP in 568.5 ms of wall — 1.31 TFLOP/s
//! over *all* its steps, not only its multiplies — against a packed Q4_K GEMM at
//! its own row count of **1.10 TFLOP/s** (0.76 at the narrower projection).
//!
//! Both sides of that comparison are hand-derived rather than tallied from a
//! schedule this tree can print: the 745 GFLOP is `2 · m · Σ params` over the
//! linear projections of the 24-layer encoder at `m = 219`, and the 568.5 ms is
//! the prefill's measured wall time, so the rate is a **floor** on the multiply
//! rate (the same wall also carries the transposes, layer norms and concats,
//! which add no FLOPs to that count). A floor above the isolated 1.10 TFLOP/s of
//! one call means the layer loop is not where the prefill's time goes — the
//! isolated figure is the pessimistic side, since one call pays the whole dispatch
//! floor with nothing to overlap it — and it does **not** mean the kernel has no
//! headroom: the gap this harness exposes is between 1.1 and this machine's f32
//! peak, inside the multiply, not in the schedule around it. The `m = 1` rows are
//! here to show the call floor of a single dispatch (95 GFLOP/s where 32 of the
//! same step in one call read 83.6 GB/s in `gguf_one_row_stream_rate`), not as a
//! rate to quote.
//!
//! # The synthesis loop's ceiling
//!
//! `synthesis_ceiling` takes the same accounting to the synthesis loop, whose
//! dominant work is the same `m = 1` packed multiply: 850 870 272 packed bytes a
//! frame in the talker body (169 packed Q4_K/Q6_K multiplies) and 1 393 754 112 in
//! the code predictor (348 packed Q8_0 ones), over 330 frames of a 26.4 s speech.
//! What the loop itself reads — its ms a frame, the sound half — belongs to the run
//! that takes it; no record carries a per-form table. In one window it takes the stream probe's
//! read and copy ceilings, then one joined call of `steps` identical
//! `DequantMatMul`s over distinct parameters at each of the frame's geometries
//! (best of 7 after 3 warm-up calls), `steps` chosen so a call moves 214-302 MB —
//! every form fits this adapter's per-buffer limit, and the helper names one that
//! would not. Two denominators follow: (a) every frame byte read once at the
//! probe's read ceiling, which unpacks nothing, multiplies nothing and takes no
//! input, and (b) the byte-weighted rate these forms themselves achieved here,
//! each side's rows weighted by their own packed bytes and the frame's two byte
//! totals applying each side's rate. Both are floors/measurements of the frame's
//! multiply bytes only: the frame's attention, transposes, layer norms and
//! sampling are not in either.
//!
//! Recorded 2026-10-21 on this machine (Apple M3 Pro, 14-core GPU, wgpu over
//! Metal), three standalone runs of the same build, `--release` (this test alone,
//! as the probe's own doc asks):
//!
//! | side | scheme | m | k | n | steps | bytes/step | GB/s |
//! |---|---|---|---|---|---|---|---|
//! | body | Q4_K | 1 | 2048 | 2048 | 128 | 2 359 296 | 62.9-64.4 |
//! | body | Q4_K | 1 | 6144 | 2048 | 32 | 7 077 888 | 83.7-86.5 |
//! | body | Q4_K | 1 | 2048 | 12 288 | 16 | 14 155 776 | 90.7-93.9 |
//! | body | Q6_K | 1 | 2048 | 1024 | 128 | 1 720 320 | 39.9-42.3 |
//! | predictor | Q8_0 | 1 | 1024 | 4096 | 48 | 4 456 448 | 70.2-71.1 |
//! | predictor | Q8_0 | 1 | 1024 | 2048 | 96 | 2 228 224 | 55.7-56.5 |
//! | predictor | Q8_0 | 1 | 1024 | 6144 | 32 | 6 684 672 | 76.6-78.3 |
//! | predictor | Q8_0 | 1 | 2048 | 1024 | 96 | 2 228 224 | 55.4-57.4 |
//! | predictor | Q8_0 | 1 | 3072 | 1024 | 64 | 3 342 336 | 66.7-67.8 |
//!
//! The stream probe read 110.8-129.9 GB/s (copy 110.1-130.8 GB/s) across those
//! runs, 17 % apart, and its own rounds spanned 104-131, so a derived figure
//! inherits that spread. A run with `gemm_ceiling` ahead of it in the same process
//! read 78.9 GB/s for the body and 60.3 for the predictor (0.510 s/speech), its
//! per-row figures up to a third lower, which is why this test is taken alone.
//!
//! The two denominators this test exists to produce — the frame's packed bytes
//! (2 244 624 384 = 850 870 272 + 1 393 754 112) at (a) the probe's read ceiling
//! and (b) the byte-weighted rate of the forms above — are **not restated here**:
//! a reading belongs to the run that takes it, and a second copy is a second
//! number to drift. What the test prints is (a), (b) and the frame total for its
//! own session, so two runs can be read against each other.
//!
//! The denominators bound what these geometries cost as one contiguous run over
//! one shared input row, and **not** what a frame costs — a frame's 169 or 348
//! multiplies are distinct calls with the frame's other work between them, none of
//! which this test executes. So the gap between them and the test's own frames is a
//! measurement of this harness, not an attribution of the frame's time.
//!
//! The dense f32 comparison at the same row and two
//! predictor widths is a call floor rather than a rate: 502.9-694.9 µs
//! (24.1-33.4 GB/s of weight bytes) at `k = 1024, n = 4096` and 757.7-813.8 µs
//! (10.3-11.1 GB/s) at `k = 2048, n = 1024` over four runs — the narrower column
//! count is the more floor-bound, and the wider row moved 38 % between runs, so at
//! `m = 1` neither says anything about a dense GEMM at the row counts the prefill
//! uses.
//!
//! Run the thermal gate first (`just throttle`).
mod common;

use common::data;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::wgpu_device;

/// Best-of-7 whole-call µs, three warm-up calls ahead of it — the crate's own
/// timing convention (`common::time_rounds`, whose clock is `Tick`).
fn best_us(mut one_call: impl FnMut()) -> f64 {
    for _ in 0..3 {
        one_call();
    }
    common::time_rounds(7, 1, &mut one_call)
        .into_iter()
        .fold(f64::MAX, f64::min)
        * 1e3
}

/// Best-of-7 µs at one dense f32 shape.
fn time(m: usize, k: usize, n: usize) -> f64 {
    let mut g = Graph::new("gemm_ceiling");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Gpu).compile(g);
    let wv = data(k * n, 0.001, true);
    let xv = data(m * k, 0.002, false);
    c.set_param("w", &wv);
    best_us(|| {
        let _ = c.run(&[("x", &xv)]).remove(0);
    })
}

/// Best-of-7 µs for one packed `DequantMatMul` at `(m, k, n)`, with its packed
/// weight's own byte count — the numerator of the rate.
fn packed_time(scheme: QuantScheme, m: usize, k: usize, n: usize) -> (f64, u64) {
    let packed = common::packed_gguf_weight(scheme, k, n);
    let mut g = Graph::new("packed_gemm_ceiling");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let xv = data(m * k, 0.002, false);
    let us = best_us(|| {
        let _ = c.run(&[("x", &xv)]).remove(0);
    });
    (us, packed.len() as u64)
}

#[test]
#[ignore = "measurement: prints a cost table, asserts no duration"]
fn gemm_ceiling() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    eprintln!("[rlx-gemm-ceiling] dense f32 matmul through Session on wgpu (best of 7)");
    for (m, k, n) in [
        (219usize, 1024usize, 6144usize),
        (219, 2432, 1024),
        (219, 1024, 2048),
        (512, 512, 512),
    ] {
        let us = time(m, k, n);
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        eprintln!(
            "[rlx-gemm-ceiling] m={m} k={k} n={n}: {us:8.1} us  {gflop:6.3} GFLOP  {:7.1} GFLOP/s",
            gflop * 1e6 / us
        );
    }
    eprintln!(
        "[rlx-gemm-ceiling] packed Q4_K/Q6_K DequantMatMul through Session on wgpu (best of 7): \
         the dtype the prefill's own multiplies are stored in"
    );
    for (scheme, m, k, n) in [
        (QuantScheme::GgufQ4K, 219usize, 2048usize, 6144usize),
        (QuantScheme::GgufQ4K, 219, 1024, 2048),
        (QuantScheme::GgufQ4K, 1, 2048, 6144),
        (QuantScheme::GgufQ6K, 1, 2048, 151936),
        (QuantScheme::GgufQ6K, 219, 2048, 151936),
    ] {
        let (us, bytes) = packed_time(scheme, m, k, n);
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        eprintln!(
            "[rlx-gemm-ceiling] {scheme:?} m={m} k={k} n={n}: {us:9.1} us  {} B packed  \
             {:7.1} GFLOP/s  {:6.1} GB/s",
            bytes,
            gflop * 1e6 / us,
            bytes as f64 / 1e3 / us
        );
    }
}

/// One joined call of `steps` packed `DequantMatMul`s over distinct parameters of
/// one packed region at `(1, k, n)` — the shape a synthesis frame issues — its
/// best-of-7 µs and one step's packed bytes. `None` when the call's steps' regions
/// would not fit this adapter's per-buffer limit; that form is named, not dropped.
fn packed_steps_time(
    dev: &rlx_wgpu::device::WgpuDevice,
    scheme: QuantScheme,
    k: usize,
    n: usize,
    steps: usize,
) -> Option<(f64, u64)> {
    let packed = common::packed_gguf_weight(scheme, k, n);
    let transient = packed.len() as u64 * steps as u64;
    let max_buffer = dev.device.limits().max_buffer_size;
    if transient > max_buffer {
        eprintln!(
            "[rlx-synthesis-ceiling] {scheme:?} k={k} n={n} steps={steps}: the steps' regions \
             total {transient} bytes, over this adapter's per-buffer limit ({max_buffer}) — \
             skipped rather than allowed to fail opaquely"
        );
        return None;
    }
    let mut g = Graph::new("synthesis_packed_steps");
    let x = g.input("x", Shape::new(&[1, k], DType::F32));
    let mut outs = Vec::with_capacity(steps);
    for s in 0..steps {
        let w = g.param(format!("w{s}"), Shape::new(&[packed.len()], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[1, n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    let mut c = Session::new(Device::Gpu).compile(g);
    for s in 0..steps {
        c.set_param_typed(&format!("w{s}"), &packed, DType::U8);
    }
    let xv = data(k, 0.002, false);
    // Structural pins only: one output per step, each of the step's width, every
    // value finite. A time is never asserted.
    let warm = c.run(&[("x", &xv)]);
    assert_eq!(warm.len(), steps, "one output per step");
    for (s, out) in warm.iter().enumerate() {
        assert_eq!(out.len(), n, "step {s} output width");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "step {s} produced a non-finite value"
        );
    }
    let us = best_us(|| {
        let _ = c.run(&[("x", &xv)]).remove(0);
    });
    Some((us, packed.len() as u64))
}

/// The synthesis shapes: one talker-body frame and one code-predictor frame over
/// 330 frames of a 26.4 s speech. The sound half is **not** a constant here: it
/// depends on the decoder's own arrangement, so this measures the loop's figure
/// alone and states it alone.
const BODY_FRAME_BYTES: f64 = 850_870_272.0;
const PREDICTOR_FRAME_BYTES: f64 = 1_393_754_112.0;
const FRAMES: f64 = 330.0;
const SPEECH_S: f64 = 26.4;

/// Byte-weighted harmonic mean of a set of `(bytes, GB/s)` rows: the rate a
/// stream of `bytes` sees when each row's bytes are read at that row's own rate.
fn byte_weighted_gb_s(rows: &[(u64, f64)]) -> f64 {
    let bytes: f64 = rows.iter().map(|&(b, _)| b as f64).sum();
    bytes / rows.iter().map(|&(b, r)| b as f64 / r).sum::<f64>()
}

#[test]
#[ignore = "measurement: prints the synthesis loop's denominators, asserts no duration"]
fn synthesis_ceiling() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let Some(dev) = wgpu_device() else {
        eprintln!("[rlx-synthesis-ceiling] skip: wgpu unavailable");
        return;
    };
    eprintln!(
        "[rlx-synthesis-ceiling] adapter={:?} — these figures hold for this adapter and \
         machine load only",
        rlx_wgpu::device::adapter_name()
    );

    // One window: the stream probe first, every packed call after it on the same
    // device.
    let ceiling = rlx_wgpu::bandwidth::stream_ceiling()
        .unwrap_or_else(|e| panic!("rlx-synthesis-ceiling: stream probe produced no figure: {e}"));
    let read_gb_s = ceiling.read_gb_s;
    eprintln!(
        "[rlx-synthesis-ceiling] stream ceiling: read {:.1} GB/s (min {:.1}, max {:.1}) \
         copy {:.1} GB/s (min {:.1}, max {:.1}) over {} rounds of {} B per direction",
        ceiling.read_gb_s,
        ceiling.read_min_gb_s,
        ceiling.read_max_gb_s,
        ceiling.gb_s,
        ceiling.min_gb_s,
        ceiling.max_gb_s,
        ceiling.rounds,
        ceiling.bytes_per_direction,
    );

    // The code predictor's own storage and geometry (Q8_0, one row), and the
    // talker body's K-quant geometries: one joined call each, best of 7 after
    // three warm-up calls. Byte totals over 100 MB per call, all under this
    // adapter's per-buffer limit (the helper names a form that is not).
    let predictor = [
        (QuantScheme::GgufQ8_0, 1024usize, 4096usize, 48usize),
        (QuantScheme::GgufQ8_0, 1024, 2048, 96),
        (QuantScheme::GgufQ8_0, 1024, 6144, 32),
        (QuantScheme::GgufQ8_0, 2048, 1024, 96),
        (QuantScheme::GgufQ8_0, 3072, 1024, 64),
    ];
    let body = [
        (QuantScheme::GgufQ4K, 2048usize, 2048usize, 128usize),
        (QuantScheme::GgufQ4K, 6144, 2048, 32),
        (QuantScheme::GgufQ4K, 2048, 12288, 16),
        (QuantScheme::GgufQ6K, 2048, 1024, 128),
    ];

    let mut predictor_rows = Vec::with_capacity(predictor.len());
    let mut body_rows = Vec::with_capacity(body.len());
    for (side, rows) in [("predictor", &predictor[..]), ("body", &body[..])] {
        for &(scheme, k, n, steps) in rows {
            let Some((us, bytes_per_step)) = packed_steps_time(dev, scheme, k, n, steps) else {
                continue;
            };
            let call_bytes = bytes_per_step as f64 * steps as f64;
            let gb_s = call_bytes / (us * 1e3);
            eprintln!(
                "[rlx-synthesis-ceiling] side={side} {scheme:?} m=1 k={k} n={n} steps={steps} \
                 packed_bytes_per_step={bytes_per_step} call_bytes={call_bytes:.0} \
                 call_us={us:.1} us_per_step={:.1} GB/s={gb_s:.1}",
                us / steps as f64
            );
            if side == "predictor" {
                predictor_rows.push((bytes_per_step, gb_s));
            } else {
                body_rows.push((bytes_per_step, gb_s));
            }
        }
    }
    assert_eq!(
        predictor_rows.len(),
        predictor.len(),
        "every predictor form fits"
    );
    assert_eq!(body_rows.len(), body.len(), "every body form fits");

    let frame_bytes = BODY_FRAME_BYTES + PREDICTOR_FRAME_BYTES;
    eprintln!(
        "[rlx-synthesis-ceiling] per-frame bytes: body {BODY_FRAME_BYTES:.0} + predictor \
         {PREDICTOR_FRAME_BYTES:.0} = {frame_bytes:.0}; frames={FRAMES} speech={SPEECH_S} s; \
         the figures below are the generation loop alone — the sound half is a \
         different arrangement and is not carried here"
    );

    // (a) The pure byte floor: every frame byte read once at the probe's read
    // ceiling, which unpacks nothing and multiplies nothing.
    let ms_at = |bytes: f64, gb_s: f64| bytes / (gb_s * 1e6);
    let floor_ms = ms_at(frame_bytes, ceiling.read_gb_s);
    let floor_loop_s = FRAMES * floor_ms / 1e3;
    eprintln!(
        "[rlx-synthesis-ceiling] (a) read ceiling {:.1} GB/s: per-frame {floor_ms:.3} ms \
         ({BODY_FRAME_BYTES:.0} B / {read_gb_s:.1} GB/s = {:.3} ms + \
         {PREDICTOR_FRAME_BYTES:.0} B = {:.3} ms); loop {FRAMES} x {floor_ms:.3} ms = \
         {floor_loop_s:.3} s; loop-only s/speech = {floor_loop_s:.3} / {SPEECH_S} = {:.3}",
        ceiling.read_gb_s,
        ms_at(BODY_FRAME_BYTES, ceiling.read_gb_s),
        ms_at(PREDICTOR_FRAME_BYTES, ceiling.read_gb_s),
        floor_loop_s / SPEECH_S,
    );

    // (b) The best rate these forms achieved here, each row weighted by its own
    // packed bytes inside its side, and each side by the frame's own byte totals.
    let body_gb_s = byte_weighted_gb_s(&body_rows);
    let predictor_gb_s = byte_weighted_gb_s(&predictor_rows);
    let body_ms = ms_at(BODY_FRAME_BYTES, body_gb_s);
    let predictor_ms = ms_at(PREDICTOR_FRAME_BYTES, predictor_gb_s);
    let measured_ms = body_ms + predictor_ms;
    let measured_loop_s = FRAMES * measured_ms / 1e3;
    eprintln!(
        "[rlx-synthesis-ceiling] (b) measured forms, byte-weighted per side: body \
         {body_gb_s:.1} GB/s ({BODY_FRAME_BYTES:.0} B = {body_ms:.3} ms), predictor \
         {predictor_gb_s:.1} GB/s ({PREDICTOR_FRAME_BYTES:.0} B = {predictor_ms:.3} ms); \
         per-frame {measured_ms:.3} ms; loop {FRAMES} x {measured_ms:.3} ms = \
         {measured_loop_s:.3} s; loop-only s/speech = {measured_loop_s:.3} / {SPEECH_S} = {:.3}",
        measured_loop_s / SPEECH_S,
    );
    // The dense f32 comparison at the same row and two of
    // the predictor's widths — a call floor at `m = 1`, not a rate to quote.
    eprintln!(
        "[rlx-synthesis-ceiling] dense f32 matmul through Session on wgpu (best of 7), m = 1:"
    );
    for (m, k, n) in [(1usize, 1024usize, 4096usize), (1, 2048, 1024)] {
        let us = time(m, k, n);
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        let weight_bytes = (k * n * 4) as f64;
        eprintln!(
            "[rlx-synthesis-ceiling] dense f32 m={m} k={k} n={n}: {us:8.1} us  {gflop:6.3} \
             GFLOP  {:7.1} GFLOP/s  {:6.1} GB/s of weight bytes",
            gflop * 1e6 / us,
            weight_bytes / (us * 1e3)
        );
    }
}
