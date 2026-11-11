// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A packed weight whose **last row's last access ends two bytes past its rows**
//! still computes its tail column.
//!
//! `gguf_gpu::gemv_weight_window` carries the reach and why the cut is 16 rather than
//! 4. This file is the fixture for it, in both crossing shapes: **seven** 210-byte Q6_K
//! rows (1 470 bytes) and **three** 34-byte Q8_0 blocks (102 bytes), each ending two
//! bytes off a word boundary. Reverting the round-up makes this file die at bind-group
//! creation with
//! `Effective buffer binding size 1470 for storage buffers is expected to align to 4`,
//! which is what makes the fixtures non-vacuous.
//!
//! A packed `U8` param of this size is parked in the weight buffer
//! (`Arena::from_plan_split` puts a param there until it outgrows one weight shard),
//! so that is the buffer whose sizing this pins. Nothing about the served models is
//! needed to see it, and nothing about them is claimed here — this file is the
//! fixture, not a measurement.
mod common;

use common::max_abs_diff;
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// One packed `[n, k]` weight, its bytes and its `[1, k]` activation row. `k` is a
/// whole number of the scheme's blocks, so the packed weight is `row_bytes * n`. The
/// scheme comes from the packing type rather than a second argument, so a fixture
/// cannot name one scheme and pack another.
fn case(ggml: GgmlType, k: usize, n: usize) -> (Graph, Vec<u8>, Vec<f32>, QuantScheme) {
    let scheme = rlx_gguf::quant_scheme(ggml).expect("a packed-quant dtype");
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_gemv_tail");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x, scheme)
}

fn run(device: Device, g: Graph, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

/// One fixture: `n` rows of `row_bytes`, ending two bytes off a word boundary, which
/// the one-row device arm must serve and match the CPU on.
fn tail_column_survives_a_crossing_last_access(
    ggml: GgmlType,
    k: usize,
    n: usize,
    row_bytes: usize,
) {
    let (g, packed, x, scheme) = case(ggml, k, n);
    // The premise, asserted rather than assumed: a row stride that padded to 16, or a
    // count that ended the rows 4-aligned, would make the case vacuous.
    assert_eq!(packed.len(), row_bytes * n, "{scheme:?}: the row stride");
    assert_eq!(
        packed.len() % 4,
        2,
        "{scheme:?}: the fixture needs the rows to end two bytes off a word boundary, \
         or their last access stays inside them"
    );

    let cpu = run(Device::Cpu, g.clone(), &packed, &x);
    // Which arm answered, per cell: the record is written by the one-row device route
    // only, and it is cleared here, so a value left by another cell cannot satisfy the
    // assertion. The recorded dispatch must also cover this fixture's width, which ties
    // the record to this cell's call rather than to any one-row dispatch at all.
    rlx_wgpu::gguf_gpu::reset_gemv_one_row_shape();
    let gpu = run(Device::Gpu, g, &packed, &x);
    let shape = rlx_wgpu::gguf_gpu::gemv_one_row_shape().unwrap_or_else(|| {
        panic!(
            "{scheme:?}: the one-row device arm must serve this shape, or the tail \
             column is computed by a route whose window this file does not pin"
        )
    });
    assert!(
        (shape.workgroups * shape.cols_per_workgroup) as usize >= n,
        "{scheme:?}: the recorded dispatch covers {} columns, short of the fixture's {n}",
        shape.workgroups * shape.cols_per_workgroup
    );
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "{scheme:?}: the tail column left a non-finite value: {gpu:?}"
    );
    let diff = max_abs_diff(&gpu, &cpu);
    assert!(
        diff <= 1e-2,
        "{scheme:?}: a weight whose last access reads one word past its rows must \
         still compute its tail column — the window is cut on the 16-byte boundary, \
         not at the row's last byte (max|diff| {diff:e}, gpu {gpu:?}, cpu {cpu:?})"
    );
    // The last row is the one the window's end covers: name it, so a failure is never
    // read as a general parity drift.
    assert!(
        (gpu[n - 1] - cpu[n - 1]).abs() <= 1e-2,
        "{scheme:?}: the last column (the row at the window's end) read {} against \
         the CPU's {}",
        gpu[n - 1],
        cpu[n - 1]
    );
}

#[test]
fn a_weight_whose_last_access_crosses_the_row_end_computes_its_tail_column() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    // One `#[test]` on purpose, as in `gguf_gemv_wide_arm`: the shape record read
    // above is process-global, so a second cell overwrites it — each cell clears it
    // before its dispatch and the width assertion ties the record to its own call.
    // The two crossing shapes: the last Q6_K row's `d`, and the last Q8_0 block's
    // second quant leg.
    tail_column_survives_a_crossing_last_access(GgmlType::Q6K, 256, 7, 210);
    tail_column_survives_a_crossing_last_access(GgmlType::Q8_0, 32, 3, 34);
}
