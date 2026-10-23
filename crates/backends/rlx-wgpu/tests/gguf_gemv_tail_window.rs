// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A packed weight whose **last row ends off a 16-byte boundary** still computes its
//! tail column.
//!
//! The one-row multiply decodes a block as 16-byte legs (`read_w16`), and a leg that
//! does not start on a group boundary is assembled from the two groups it straddles,
//! so the fetch of a block's last leg reaches one group past the bytes it consumes.
//! The weight binding is therefore cut on a 16-byte boundary from the last row's end
//! (`gguf_gpu::gemv_weight_window`), and the weight buffer under it has to end on one
//! too: `Arena::from_plan_split` sizes it and its shards that way. A window clamped to a
//! four-byte-rounded buffer instead (which is what the weight buffer used to be) leaves
//! that second group outside the binding — where a robust access answers zeros, and the
//! tail column reads a zero scale.
//!
//! Q6_K's 210-byte rows are what make the case reachable: six of them are 1 260 bytes,
//! so the last row's packed bytes end 12 bytes into a group. A packed `U8` param of this
//! size is parked in the weight buffer (`Arena::from_plan_split` puts a param there until
//! it outgrows one weight shard), so that is the buffer whose sizing this pins — and it
//! used to be rounded to four bytes, which is exactly the 1 260-byte end above.
//! Reverting that rounding turns this case into `gemv_weight_window`'s refusal rather
//! than a wrong tail column, which is the loud failure the assertion is for. Nothing
//! about the served models is needed to see it, and nothing about them is claimed
//! here — this file is the fixture, not a measurement.
mod common;

use common::max_abs_diff;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// One Q6_K `[n, k]` weight, its packed bytes and its `[1, k]` activation row. `k` is
/// a whole number of the scheme's 256-element blocks, so the packed weight is
/// `210 * k / 256 * n` bytes.
fn case(k: usize, n: usize) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q6K).expect("quantize");
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_gemv_tail");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ6K,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
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
fn a_weight_whose_last_row_ends_off_16_computes_its_tail_column() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    // `k = 256` is one Q6_K block, so the rows are 210 bytes each and six of them are
    // 1 260: the premise of the fixture, asserted rather than assumed — a quantizer
    // that padded a row to 16 would make the case vacuous.
    let (k, n) = (256usize, 6usize);
    let (g, packed, x) = case(k, n);
    assert_eq!(packed.len(), 210 * n, "Q6_K's row stride");
    assert_ne!(
        packed.len() % 16,
        0,
        "the fixture needs a weight whose last row ends off a 16-byte boundary"
    );

    let cpu = run(Device::Cpu, g.clone(), &packed, &x);
    // Which arm answered: the record is written by the one-row device route only, so
    // an empty record would mean this case fell back to the host and pinned nothing.
    rlx_wgpu::gguf_gpu::reset_gemv_one_row_shape();
    let gpu = run(Device::Gpu, g, &packed, &x);
    assert!(
        rlx_wgpu::gguf_gpu::gemv_one_row_shape().is_some(),
        "the one-row device arm must serve this shape, or the tail column is computed \
         by a route whose window this file does not pin"
    );
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "the tail column left a non-finite value: {gpu:?}"
    );
    let diff = max_abs_diff(&gpu, &cpu);
    assert!(
        diff <= 1e-2,
        "a weight whose last row ends off 16 must still compute its tail column — the \
         window is cut on the 16-byte boundary, not at the row's last byte (max|diff| \
         {diff:e}, gpu {gpu:?}, cpu {cpu:?})"
    );
    // The last row is the one the window's end covers: name it, so a failure is never
    // read as a general parity drift.
    assert!(
        (gpu[n - 1] - cpu[n - 1]).abs() <= 1e-2,
        "the last column (the row at the window's end) read {} against the CPU's {}",
        gpu[n - 1],
        cpu[n - 1]
    );
}
