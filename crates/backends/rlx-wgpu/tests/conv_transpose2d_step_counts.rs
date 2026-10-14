// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What each of `Op::ConvTranspose2d`'s two wgpu routes costs the device: a
//! host-classified step carries its own submission and its own completion poll,
//! the device step does not.
//!
//! The counters are process-global, so this is a **differential** between two
//! runs of the same graph — the same geometry compiled once per route — not an
//! absolute per-run figure. Both runs are warmed first, and the run report is
//! refused because it adds a fence, a resolve submit and a map of its own
//! (`common::refuse_under_run_report`); `common::route_lock` is held throughout
//! because the counters are process-global.
//!
//! The geometry is the decoder's — kernel `[1, 16]`, stride `[1, 8]`, `c_in = 8`,
//! `c_out = 6`, `h = 1`, `w_in = 96`, pad `[0, 0]`, a 1-D-lifted kernel at high
//! stride on unit height, i.e. the shape a vocoder-style upsampler decoder
//! presents — which is the compact branch of the host route
//! (`accelerated_route_applies`). That matters: on the host route's whole-arena
//! branch the host step pays only one submission and one poll *for the whole
//! step*, so a geometry that fell to it would not show the pair this file
//! measures. The compact branch stages the two operands it touches, an input and
//! a weight, and each staged read is its own submission and drain.
//!
//! Measured on the decoder geometry: the device route spends one submission and
//! one wait (its dispatch and the output readback), the host route three and four
//! — the compact branch's two staged reads, plus the poll the run loop owes the
//! device once a host step is over, plus the same readback. This file pins that
//! pair exactly; see the assertion for why the wait differential is three rather
//! than the two a "one submission, one poll" reading of the host step suggests.
//!
//! Its own test binary: the route is read per compile and the lever is
//! process-global.

mod common;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// Decoder geometry: unit height, 1-D-lifted — kernel `[1, 16]`, stride `[1, 8]`.
const C_IN: usize = 8;
const C_OUT: usize = 6;
const L_IN: usize = 96;
const K: usize = 16;
const S: usize = 8;

fn graph() -> Graph {
    let mut g = Graph::new("conv_transpose2d_step_counts");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, L_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 1, K], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, K], [1, S], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn the_host_step_spends_its_own_submission_and_poll_and_the_device_step_does_not() {
    common::refuse_under_run_report();
    let _route = common::route_lock();
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    let x: Vec<f32> = (0..C_IN * L_IN)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..C_IN * C_OUT * K)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();

    let mut device = WgpuExecutable::compile(graph());
    let device_kinds = device.step_kind_counts();
    device.set_param("w", &w);
    // Warm up: the first run materialises the pooled output slab and stages the
    // uniforms, neither of which is a submission or a wait.
    let _ = device.run(&[("x", &x)]).remove(0);

    rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
    let mut host = WgpuExecutable::compile(graph());
    rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
    let host_kinds = host.step_kind_counts();
    host.set_param("w", &w);
    let _ = host.run(&[("x", &x)]).remove(0);

    reset_device_counts();
    let _ = device.run(&[("x", &x)]).remove(0);
    let (device_submits, device_waits) = (device_submit_count(), device_wait_count());

    reset_device_counts();
    let _ = host.run(&[("x", &x)]).remove(0);
    let (host_submits, host_waits) = (device_submit_count(), device_wait_count());

    eprintln!("device route: {device_kinds:?}");
    eprintln!("RLX_WGPU_CONV_HOST=1: {host_kinds:?}");
    eprintln!(
        "device route: {device_submits} submission(s), {device_waits} wait(s); \
         host route: {host_submits} submission(s), {host_waits} wait(s)"
    );
    assert_eq!(
        device_kinds.get("conv_transpose2d").copied().unwrap_or(0),
        1,
        "with the structure-host preference off the route must be the device step; schedule = \
         {device_kinds:?}"
    );
    assert_eq!(
        host_kinds
            .get("conv_transpose2d_host")
            .copied()
            .unwrap_or(0),
        1,
        "the levered executable must name the host step; schedule = {host_kinds:?}"
    );
    // The host step stages two operands — the input and the weight — so its own
    // cost is two D2H submission/drain pairs; the device step is one dispatch. On
    // top of that pair the run loop owes the device one poll after a host step
    // (`backend/run.rs`, the boundary taken once the host step is over), which a
    // schedule of device steps has no reason to pay, so the differential is two
    // extra submissions and three extra waits. This is the measured pair, not the
    // two-and-two this pin was first written against: the third wait is that
    // boundary poll, and it is stated rather than absorbed into a range.
    assert_eq!(
        (host_submits, host_waits),
        (device_submits + 2, device_waits + 3),
        "the host step's own two staged reads plus the run loop's post-host-step \
         poll must be the whole difference — measured device {device_submits} \
         submission(s)/{device_waits} wait(s), host {host_submits}/{host_waits}"
    );
}
