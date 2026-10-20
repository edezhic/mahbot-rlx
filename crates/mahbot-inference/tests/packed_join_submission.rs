// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A joined packed multiply costs one submission, not one per weight. The
//! counters are the backend's process-global [`rlx_wgpu::device_submit_count`] /
//! [`rlx_wgpu::device_wait_count`], read around one forward pass of a synthetic
//! graph built from packed weights — no model files are involved.
//!
//! The graph is the join set's shape: two row-grouped packed multiplies with a
//! dense device multiply between them. A single test in its own binary, because
//! the counters are process-global; it skips on a host with no wgpu adapter.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

mod support;

/// One Q4_K superblock: a whole number of the scheme's packing blocks.
const K: usize = 256;
const N: usize = 64;
/// Above one row, so both packed multiplies take the row-grouped arm.
const M: usize = 4;

/// Deterministic packed Q4_K bytes for a `[k, n]` weight.
fn packed(k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).expect("quantize")
}

/// `dq(Q4_K) → mm → dq(Q4_K)`: both packed multiplies are device work, and the
/// dense multiply between them keeps the batch from ending at the first call.
#[test]
fn one_run_of_a_joined_packed_multiply_issues_one_submission() {
    // The report's own fence is not this run's: under it the counts this test pins
    // are the instrument's, so refuse instead of asserting them.
    support::assert_run_report_off();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let w1 = packed(K, N);
    let w2 = packed(K, N);
    let mut g = Graph::new("mahbot_packed_join");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let p1 = g.param("w1", Shape::new(&[w1.len()], DType::U8));
    let p2 = g.param("w2", Shape::new(&[w2.len()], DType::U8));
    let dense = g.param("p", Shape::new(&[N, K], DType::F32));
    let y1 = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x, p1],
        Shape::new(&[M, N], DType::F32),
    );
    let h = g.mm(y1, dense);
    let y2 = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![h, p2],
        Shape::new(&[M, N], DType::F32),
    );
    g.set_outputs(vec![y2]);

    let mut compiled = Session::new(Device::Gpu).compile(g);
    compiled.set_param_typed("w1", &w1, DType::U8);
    compiled.set_param_typed("w2", &w2, DType::U8);
    compiled.set_param("p", &vec![0.01f32; N * K]);
    let input: Vec<f32> = (0..M * K).map(|i| ((i as f32) * 0.007).sin()).collect();

    rlx_wgpu::reset_device_counts();
    let out = compiled.run(&[("x", &input)]);
    assert_eq!(out.len(), 1);
    assert!(out[0].iter().all(|v| v.is_finite()), "outputs are finite");
    assert_eq!(
        rlx_wgpu::device_submit_count(),
        1,
        "the packed multiplies join the enclosing batch: one submission"
    );
    assert_eq!(
        rlx_wgpu::device_wait_count(),
        1,
        "one wait for the batch's readback"
    );
}
