// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measurement only, one ignored probe and one gate: what the sound half's
//! **transposed convolutions** cost in the shipped form, against the phase-panel
//! form, with the shapes held to the emitter's own table. Run the probe with
//! `cargo test -p mahbot-inference --release --test sound_transpose_form -- --ignored --nocapture`;
//! `the_forms_are_the_emitters` runs with the crate's tests.
//!
//! A stride-`s` transposed convolution with kernel `2s` and no padding gives
//! `y[p + m*s] = x[m]*w[p] + x[m-1]*w[p+s]`, so its output splits into `s` phases
//! that each read two taps — the panel form is `s` multiplies of `[c_out, 2*c_in]`
//! by one shared `[2*c_in, l]` panel. Both forms carry the same MAC count, so the
//! comparison is the form's own: one `ConvTranspose2d` step against `s` panel
//! multiplies at the same arithmetic, at the channel widths and lengths the
//! decoder's four upsamplers run at. The shapes are the emitter's own — it exports
//! the table (`graphs::DAC_UPSAMPLE`) and the stage geometry (`dac_stage_io`), so
//! no model is read and the probe cannot measure a shape the decoder dropped.
use mahbot_inference::graphs::{
    DAC_MARGIN, DAC_UPSAMPLE, DAC_UPSAMPLE_STAGE, DAC_WINDOW, dac_stage_io, samples_per_frame,
    windows,
};
use mahbot_inference::synth::MAX_CODEC_ROWS;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{Device, Session};

/// Rounds one arm is timed over; the least is the arm's own figure (the quietest
/// sample of one call).
const ROUNDS: usize = 7;

/// The cell's codec spans, from the window function the sound half itself uses —
/// the three windows a group runs the pinned cell at.
fn spans() -> Vec<usize> {
    windows(MAX_CODEC_ROWS, DAC_WINDOW, DAC_MARGIN)
        .iter()
        .map(|&(p0, p1, _, _)| p1 - p0)
        .collect()
}

fn plan(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// A deterministic ramp of `n` f32s: only the timing is read, so the values need
/// only be finite.
fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25 + (i as f32 * 1e-4).sin()
        })
        .collect()
}

/// One compiled graph and its inputs, timed over [`ROUNDS`] runs: the least.
fn time(mut compiled: rlx_runtime::CompiledGraph, inputs: &[(&str, &[f32])]) -> f64 {
    compiled.run(inputs);
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let t = Tick::now();
        compiled.run(inputs);
        best = best.min(Tick::now().elapsed_ns(t) as f64 / 1e6);
    }
    best
}

/// The shipped form: one `ConvTranspose2d` over the full `(l-1)*s + k` extent, then
/// the causal narrow to `l*s` — what `DacStage::transpose_conv1d` emits.
fn current_form(cin: usize, cout: usize, l: usize, k: usize, s: usize) -> f64 {
    let mut g = Graph::new("transpose_current");
    let x = g.input("x", plan(&[1, cin, 1, l]));
    let w = g.param("w", plan(&[cin, cout, 1, k]));
    let y = g.add_node(
        Op::ConvTranspose2d {
            kernel_size: vec![1, k],
            stride: vec![1, s],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            output_padding: vec![0, 0],
            groups: 1,
        },
        vec![x, w],
        plan(&[1, cout, 1, (l - 1) * s + k]),
    );
    let out = g.add_node(
        Op::Narrow {
            axis: 3,
            start: 0,
            len: l * s,
        },
        vec![y],
        plan(&[1, cout, 1, l * s]),
    );
    g.set_outputs(vec![out]);
    let xv = ramp(cin * l, 11);
    let wv = ramp(cin * cout * k, 13);
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param("w", &wv);
    time(c, &[("x", &xv)])
}

/// The phase-panel form: one shared `[2*c_in, l]` input — the row and the row
/// shifted by one — read by `s` multiplies with the phase's own weight slice.
fn panel_form(cin: usize, cout: usize, l: usize, s: usize) -> f64 {
    let mut g = Graph::new("transpose_panel");
    let panel = g.input("panel", plan(&[2 * cin, l]));
    let mut outs = Vec::with_capacity(s);
    for p in 0..s {
        let w = g.param(format!("w{p}"), plan(&[cout, 2 * cin]));
        outs.push(g.add_node(Op::MatMul, vec![w, panel], plan(&[cout, l])));
    }
    g.set_outputs(outs);
    let pv = ramp(2 * cin * l, 17);
    let ws: Vec<Vec<f32>> = (0..s)
        .map(|p| ramp(cout * 2 * cin, 19 + p as u32))
        .collect();
    let mut c = Session::new(Device::Gpu).compile(g);
    for (p, w) in ws.iter().enumerate() {
        c.set_param(&format!("w{p}"), w);
    }
    time(c, &[("panel", &pv)])
}

/// The forms this probe measures are the ones the emitter builds: a gate that runs
/// without a device or a model, so the pin holds in the default contour.
#[test]
fn the_forms_are_the_emitters() {
    for (stage, &(k, s, cout)) in DAC_UPSAMPLE.iter().enumerate() {
        let (_, l) = dac_stage_io(DAC_UPSAMPLE_STAGE + stage, 1);
        let (next_cin, next_l) = dac_stage_io(DAC_UPSAMPLE_STAGE + stage + 1, 1);
        assert_eq!(
            s,
            next_l / l,
            "stage {stage}: the stride is the stage's length ratio"
        );
        assert_eq!(k, 2 * s, "stage {stage}: the kernel is twice the stride");
        assert_eq!(
            cout, next_cin,
            "stage {stage}: writes the channels the next stage reads"
        );
    }
    assert_eq!(
        4 * DAC_UPSAMPLE.iter().map(|f| f.1).product::<usize>(),
        samples_per_frame(),
        "the four strides and the two leading 2× must make the samples-per-frame constant"
    );
}

#[test]
#[ignore = "measurement only: prints the decoder's transposed-conv forms against the panel form"]
fn the_sound_halfs_transposed_convs_against_the_phase_panel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let spans = spans();
    println!("census transpose rounds={ROUNDS} spans={spans:?}");
    let (mut current_total, mut panel_total, mut macs_total) = (0f64, 0f64, 0f64);
    for (stage, &(k, s, cout)) in DAC_UPSAMPLE.iter().enumerate() {
        // One span, so the four forms are read at the cell's own window; the block's
        // total scales them by the three windows below.
        let (cin, l) = dac_stage_io(DAC_UPSAMPLE_STAGE + stage, spans[0]);
        let macs = cin as f64 * cout as f64 * l as f64 * k as f64;
        let current = current_form(cin, cout, l, k, s);
        let panel = panel_form(cin, cout, l, s);
        println!(
            "census transpose form={stage} cin={cin} cout={cout} l={l} k={k} s={s} \
             macs={macs} current_ms={current:.3} current_TFLOPs={:.3} \
             panel_ms={panel:.3} panel_TFLOPs={:.3} delta_ms={:+.3} panel_bytes={}",
            tflops(current, macs),
            tflops(panel, macs),
            current - panel,
            2 * cin * l * 4,
        );
        current_total += current;
        panel_total += panel;
        macs_total += macs;
    }
    // The block as the cell runs it: three windows a group, the tail shorter than
    // the two full ones, and every one of them carrying all four forms.
    let scale: f64 = spans.iter().map(|&f| f as f64).sum::<f64>() / spans[0] as f64;
    println!(
        "census transpose block windows={:?} scale={scale:.3} macs={} \
         current_ms={:.3} current_TFLOPs={:.3} panel_ms={:.3} panel_TFLOPs={:.3} \
         delta_ms={:+.3}",
        spans,
        macs_total * scale,
        current_total * scale,
        tflops(current_total * scale, macs_total * scale),
        panel_total * scale,
        tflops(panel_total * scale, macs_total * scale),
        (current_total - panel_total) * scale,
    );
}

fn tflops(ms: f64, macs: f64) -> f64 {
    2.0 * macs / (ms * 1e-3) / 1e12
}
