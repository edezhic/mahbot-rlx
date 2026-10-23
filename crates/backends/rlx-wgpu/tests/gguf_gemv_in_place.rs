// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A one-row packed multiply stores its row straight into the arena, so it
//! allocates no pooled output slab and issues no slab-copy dispatch.
//!
//! The tiled arm's kernel writes `y[0, :]` through the window it binds; that
//! window is the output node's own arena slot when `gemv_in_place_out_window`
//! resolves it (one buffer holding the whole row). This file pins the
//! consequence: its first geometry (the code predictor's own Q8_0 shape) leaves
//! `gguf_gpu::pooled_out_buf_bytes` at zero, so `with_pooled_out_buf` never ran
//! and neither did the `encode_out_slab_copy` dispatch that used to move the row
//! out of the slab. Reverting the arm to the slab (or making the window resolver
//! always decline) allocates the slab and turns that assertion red;
//! `RLX_WGPU_GEMV_IN_PLACE=0` is that decline, which is how this store's own A/B
//! is re-taken.
//!
//! It then runs a geometry whose window reaches X, where the resolver *must*
//! refuse and take the slab, and asserts the surface reports it — so the zero
//! above is the one-row arm's doing, not a counter that never moves. The order
//! matters: `pooled_out_buf_bytes` is this process's peak.
//!
//! What this does not claim: that a sharded arena or a truncated window stores in
//! place (those calls keep the slab by design — `gemv_in_place_out_window` names
//! them), or anything about timing. The numeric bar is the cell's per-multiply one —
//! a relative difference of at most 2e-6 of the row's own max |y| against the CPU
//! arm, whose summation order is the only difference — and it is held at Q8_0 as
//! well as Q4_K: Q8_0 is the scheme the code predictor stores, so a Q8_0 cell that
//! drifted to 1e-3 would otherwise pass every numeric gate in this crate.
mod common;

use common::max_abs_diff;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::pooled_out_buf_bytes;

/// `[m, k] x [n, k]^T` at Q4_K, with its packed weight and input row — the shape
/// and fixture `gguf_matmul_route`'s parity cell builds.
fn case(m: usize, k: usize, n: usize) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q4K).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_in_place");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x)
}

/// The same multiply at Q8_0 and at the code predictor's narrowest geometry
/// (`k = 1024`, `n = 1024` — its o/down projection), which is the shape the
/// synthesis cell's one-row arm issues most.
fn case_q8_0(m: usize, k: usize, n: usize) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q8_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_in_place_q8_0");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x)
}

fn run(device: Device, g: Graph, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn a_one_row_multiply_stores_into_the_arena_and_allocates_no_output_slab() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // The store is what this file pins, so the switch is installed rather than
    // inherited: an operator's `=0` would otherwise turn the gate into a no-op.
    let _in_place = rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_GEMV_IN_PLACE", "1")));
    // The admitting geometry, first: the code predictor's narrowest Q8_0 form, where
    // the output row's slot is clear of both the X row and the packed weight.
    let (kq, nq) = (1024usize, 1024usize);
    let (gq, packedq, xq) = case_q8_0(1, kq, nq);
    let gpuq = run(Device::Gpu, gq.clone(), &packedq, &xq);
    let cpuq = run(Device::Cpu, gq, &packedq, &xq);
    let max_abs_q = max_abs_diff(&gpuq, &cpuq);
    let scale_q = cpuq.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rel_q = max_abs_q / scale_q;
    eprintln!(
        "one-row Q8_0 k={kq} n={nq}: max_abs={max_abs_q:.6e} rel={rel_q:.3e} \
         pooled_out_slab={}",
        pooled_out_buf_bytes()
    );
    // A relative bar: a 1e-2 absolute one passes a result that has lost five digits
    // on a small row and fails an exact one on a large row. 2e-6 is the cell's
    // per-multiply bound, and the summation order is the only thing that differs.
    assert!(
        rel_q <= 2e-6,
        "one-row Q8_0 k={kq} n={nq}: |Δ| {max_abs_q:.3e} is {rel_q:.3e} of the row's max |y| \
         {scale_q:.3e}, above 2e-6"
    );
    assert_eq!(
        pooled_out_buf_bytes(),
        0,
        "k={kq} n={nq}: the row's window is clear, so the store must go straight \
         into the arena and allocate no slab"
    );

    // The refusing geometry: the row's slot sits 128 bytes below X, so the
    // align-down window a storage binding needs would reach into the row the same
    // dispatch reads. The guard must refuse and the pooled slab must appear. The
    // 128-byte slip is the arena plan's (`k = 256, n = 640`), not this file's: if
    // the plan's byte layout moves enough to place this slot on a 256-byte boundary
    // the row would store again and this assertion would read the plan, not the
    // guard — the resolver's own refusal is pinned device-free in `gguf_gpu`, by
    // `a_row_inside_a_storage_block_brings_the_slot_below_it_into_the_window`.
    let (k, n) = (256usize, 640usize);
    let (g, packed, x) = case(1, k, n);
    let _ = run(Device::Gpu, g, &packed, &x);
    eprintln!(
        "one-row Q4_K k={k} n={n}: pooled_out_slab={}",
        pooled_out_buf_bytes()
    );
    assert!(
        pooled_out_buf_bytes() > 0,
        "k={k} n={n}: the out window's align-down base reaches X's row, so the store \
         must be refused and the pooled slab used; a zero here means the alias guard \
         let an overlapping writable window through"
    );
}
