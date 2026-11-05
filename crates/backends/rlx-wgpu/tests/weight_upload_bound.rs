// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A load submits before its upload outgrows one piece, and a run adds none.**
//!
//! `queue.write_buffer` stages every write in a buffer of exactly the written size and
//! keeps it until the next submission, so a graph whose whole weight set is written
//! before anything is submitted holds that set twice: the device peak of a cold call
//! becomes the model's size. [`UPLOAD_PIECE_BYTES`] bounds it — a piece that would put
//! more than one in flight submits what is pending first — and this pins both halves of
//! the contract: the bound fires on a load, and a run's own writes inject no submission.
mod common;

use common::refuse_under_run_report;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::buffer::UPLOAD_PIECE_BYTES;

/// One write of a piece and a quarter, in f32 elements: the first piece fits, and the
/// piece that would cross the bound submits it.
const ELEMENTS: usize = UPLOAD_PIECE_BYTES * 5 / 4 / size_of::<f32>();

/// `y = x * w` over [`ELEMENTS`]: a weight, and an input, each of that size, and one
/// cheap step to run.
fn one_wide_weight() -> Graph {
    let mut g = Graph::new("upload-bound");
    let x = g.input("x", Shape::new(&[ELEMENTS], DType::F32));
    let w = g.param("w", Shape::new(&[ELEMENTS], DType::F32));
    let y = g.binary(BinaryOp::Mul, x, w, Shape::new(&[ELEMENTS], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// The bound acts inside the write loop, not per parameter: a parameter larger than the
/// piece is what a flush per `set_param` would leave whole in flight.
#[test]
fn a_load_submits_before_its_upload_outgrows_one_piece() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let mut exe = WgpuExecutable::compile(one_wide_weight());
    let bytes = ELEMENTS * size_of::<f32>();
    let flushes = (bytes.div_ceil(UPLOAD_PIECE_BYTES) - 1) as u64;
    let before = rlx_wgpu::device_submit_count();
    exe.set_param("w", &vec![0.5f32; ELEMENTS]);
    let submits = rlx_wgpu::device_submit_count() - before;
    assert_eq!(
        submits,
        flushes,
        "writing {bytes} bytes in {} MiB pieces made {submits} submissions, not {flushes}",
        UPLOAD_PIECE_BYTES >> 20,
    );
    // A run stages its input, and this input is a piece and a quarter too: without the
    // run guard that upload would flush, giving a one-step call a second submission.
    let before = rlx_wgpu::device_submit_count();
    exe.run(&[("x", &vec![1.0f32; ELEMENTS])]);
    let submits = rlx_wgpu::device_submit_count() - before;
    assert_eq!(
        submits, 1,
        "a one-step run made {submits} submissions, not the one it makes"
    );
}
