// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d` when its weight is parked in the separate weight buffer.
//!
//! `from_plan_split` parks parameters whenever the arena needs the split plan and
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` forces that on any adapter.
//!
//! The decision this file pins: the device step **stages** a parked operand — as
//! its sibling 3-D step already did — so the device route reads the weight
//! correctly rather than from the wrong bytes. Only the host route, whose step
//! carries 32-bit arena byte offsets that drop `WEIGHT_BUF_TAG`, still refuses
//! loudly. This file used to assert that refusal as the op's only behaviour; the
//! pin is re-pointed because the device route made it the minority case.
//!
//! Its own test binary: the switch is process-global and lowering reads it at
//! compile time.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

const C_IN: usize = 4;
const C_OUT: usize = 2;
const L_IN: usize = 8;
const K: usize = 3;
const S: usize = 2;
/// `(L_IN − 1)·S + K`, the width the oracle writes.
const L_OUT: usize = (L_IN - 1) * S + K;

fn graph() -> Graph {
    let mut g = Graph::new("conv_transpose2d_parked");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, L_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 1, K], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, K], [1, S], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn a_parked_upsampler_weight_is_staged_on_the_device_route_and_refused_on_the_host_one() {
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // The lever this file is about, forced back on over the route env above: the
    // device arm below has to see a parked weight.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    // State 1: with the structure-host preference off, the parked weight is
    // staged into the act window and read from there, so the op computes the
    // operator on the device step.
    let mut exe = WgpuExecutable::compile(graph());
    let kinds = exe.step_kind_counts();
    eprintln!("parked, device route: {kinds:?}");
    assert_eq!(
        kinds.get("conv_transpose2d").copied().unwrap_or(0),
        1,
        "the parked weight must not move the op off the device step; schedule = {kinds:?}"
    );

    let x: Vec<f32> = (0..C_IN * L_IN)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..C_IN * C_OUT * K)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    exe.set_param("w", &w);
    let got = exe.run(&[("x", &x)]);

    let mut want = vec![0f32; C_OUT * L_OUT];
    rlx_cpu::kernels::conv_transpose2d_nchw(
        &x, &w, &mut want, 1, C_IN, 1, L_IN, C_OUT, 1, L_OUT, 1, K, 1, S, 0, 0, 1, 1, 1,
    );
    let max = common::max_abs_diff(&got[0], &want);
    eprintln!(
        "parked, device route: max abs diff {max:.2e} ({} elems)",
        want.len()
    );
    // The device step sums at most `c_in · ceil(kw / stride) = 8` f32 products
    // per output (4 input channels × the 2 of the 3-wide kernel's taps that land
    // at stride 2), over operands bounded by 0.55 and 0.24 — so
    // `γ_8 · Σ|terms| < 8·u/(1 − 8·u) · 1.06 ≈ 5.1e-7` with `u = 2^-24`, and
    // 1e-6 is that bound rounded up. Measured is ~1e-8.
    assert!(
        max <= 1e-6,
        "the parked weight's staged copy must be read correctly: max abs diff {max} > 1e-6"
    );

    // A mis-read staged operand reads the copy's other bytes — for a weight the
    // overwhelmingly likely outcome is zeros, so the output has to carry the
    // operator's own magnitude, read off the oracle.
    let got_max = got[0].iter().fold(0f32, |m, v| m.max(v.abs()));
    let want_max = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("parked, device route: max|out| {got_max:.3e}, oracle max|out| {want_max:.3e}");
    assert!(
        got_max > 1e-3 && got_max > 0.5 * want_max,
        "the staged operand must have been read: max|out| {got_max:e} vs oracle max {want_max:e}"
    );

    // The host route keeps the loud refusal: its step addresses its operands by
    // 32-bit arena byte offsets, which drop `WEIGHT_BUF_TAG`.
    rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(graph())
    }));
    rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
    let Err(payload) = caught else {
        panic!("a parked weight was accepted instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    eprintln!("parked, host route: {msg}");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (parked ConvTranspose2d operand)"),
        "refusal must carry the stable marker; got: {msg}"
    );
    // What the refusal names back is the device step, which stages a parked
    // parameter — not the `RLX_WGPU_ALL_PARAMS_WEIGHT` that parked it here.
    assert!(
        msg.contains("RLX_WGPU_CONV_HOST")
            && msg.contains("RLX_WGPU_FORCE_HOST")
            && msg.contains("let the op take its device step instead"),
        "the refusal must name the way out it offers — the device step; got: {msg}"
    );
}
