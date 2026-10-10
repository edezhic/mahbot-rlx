// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d`'s wgpu route: one host step, printed as
//! `conv_transpose2d_host`, and the numerics of the op behind it.
//!
//! The schedule names a step, not which of the two branches behind that name
//! served the geometry — `accelerated_route_applies`' compact conv-backward-input
//! staging, or the whole-arena scalar fallback — so what this file pins is the
//! printed name (the P5 marker) plus the op's numerics.
//!
//! The numeric arm compares against `rlx_cpu::kernels::conv_transpose2d_nchw`,
//! the crate's naive scalar reference kernel — an oracle the runtime parity suite
//! cannot substitute, because the CPU backend that suite compares against shares
//! the accelerated formulation by default (`conv_transpose2d_forward_gemm`, gated
//! by `RLX_FAST_CONV`, on by default).
//!
//! Its own test binary: the route is decided at the first compile.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// Decoder geometry: unit height, 1-D-lifted — kernel `[1, 16]`, stride `[1, 8]`.
const C_IN: usize = 8;
const C_OUT: usize = 6;
const L_IN: usize = 96;
const K: usize = 16;
const S: usize = 8;

fn graph() -> Graph {
    let mut g = Graph::new("conv_transpose2d_host_route");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, L_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 1, K], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, K], [1, S], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn the_upsampler_is_one_named_host_step_matching_the_scalar_reference() {
    // The routers that would otherwise decide the route for us; the shard-cap
    // lever is neutralised to a value the adapter clamps back to its own limit.
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    let mut exe = WgpuExecutable::compile(graph());
    let kinds = exe.step_kind_counts();
    eprintln!("schedule {kinds:?}");
    assert_eq!(
        kinds.get("conv_transpose2d_host").copied().unwrap_or(0),
        1,
        "the upsampler must be one host step named with the crate's host marker; schedule = {kinds:?}"
    );

    let x: Vec<f32> = (0..C_IN * L_IN)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..C_IN * C_OUT * K)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    exe.set_param("w", &w);
    let got = exe.run(&[("x", &x)]);

    let l_out = (L_IN - 1) * S + K;
    let mut want = vec![0f32; C_OUT * l_out];
    rlx_cpu::kernels::conv_transpose2d_nchw(
        &x, &w, &mut want, 1, C_IN, 1, L_IN, C_OUT, 1, l_out, 1, K, 1, S, 0, 0, 1, 1, 1,
    );
    let max = max_abs_diff(&got[0], &want);
    eprintln!("max abs diff {max:.2e} ({} elems)", want.len());
    assert!(
        max <= 1e-3,
        "route vs scalar reference: max abs diff {max} > 1e-3"
    );
}
