// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A load submits before its upload outgrows one piece, settles once its last parameter
//! lands, and a run adds neither.**
//!
//! Staging outlives its own submission (`settle_uploads` in the backend is where that is
//! spelled out), so a load that submits nothing holds its whole weight set twice and a cold
//! call's peak becomes the model's size. [`UPLOAD_PIECE_BYTES`] bounds it — a piece that
//! would put more than one in flight submits what is pending first — and the load's own
//! last piece is retired by the settle the last parameter triggers. This pins all four: the
//! bound fires inside one parameter's write, the settle fires on the last *declared*
//! parameter rather than on each write, a weight swap settles no further, and a run's own
//! writes inject no submission.
mod common;

use common::refuse_under_run_report;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::buffer::UPLOAD_PIECE_BYTES;

/// One write of a piece and a quarter, in f32 elements: the first piece fits, and the
/// piece that would cross the bound submits it.
const ELEMENTS: usize = UPLOAD_PIECE_BYTES * 5 / 4 / size_of::<f32>();

/// `y = (x * w1) * w2` over [`ELEMENTS`]: two declared parameters, so the
/// settle's trigger — the *last* one arriving — can be told from one per write.
fn two_wide_weights() -> Graph {
    let mut g = Graph::new("upload-bound-two");
    let x = g.input("x", Shape::new(&[ELEMENTS], DType::F32));
    let w1 = g.param("w1", Shape::new(&[ELEMENTS], DType::F32));
    let w2 = g.param("w2", Shape::new(&[ELEMENTS], DType::F32));
    let y = g.binary(BinaryOp::Mul, x, w1, Shape::new(&[ELEMENTS], DType::F32));
    let y = g.binary(BinaryOp::Mul, y, w2, Shape::new(&[ELEMENTS], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// The bound acts inside the write loop, not per parameter: a parameter larger than the
/// piece is what a flush per `set_param` would leave whole in flight. The one submission
/// past the flushes is the settle, and it belongs to the *last declared* parameter: the
/// first of two weights flushes and does not wait. A second write of a weight already
/// handed over — a weight swap — settles no further.
#[test]
fn a_load_submits_before_its_upload_outgrows_one_piece() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let mut exe = WgpuExecutable::compile(two_wide_weights());
    let bytes = ELEMENTS * size_of::<f32>();
    let pieces = bytes.div_ceil(UPLOAD_PIECE_BYTES) as u64;
    let submits = rlx_wgpu::device_submit_count();
    let waits = rlx_wgpu::device_wait_count();
    exe.set_param("w1", &vec![0.5f32; ELEMENTS]);
    assert_eq!(
        rlx_wgpu::device_submit_count() - submits,
        pieces - 1,
        "writing {bytes} bytes in {} MiB pieces made more than the one flush the bound \
         fires inside the write",
        UPLOAD_PIECE_BYTES >> 20,
    );
    assert_eq!(
        rlx_wgpu::device_wait_count() - waits,
        0,
        "one parameter of two arrived, and the load settled already",
    );
    exe.set_param("w2", &vec![0.5f32; ELEMENTS]);
    assert_eq!(
        rlx_wgpu::device_submit_count() - submits,
        2 * pieces,
        "the two parameters made more than one flush per piece but the last, plus the \
         settle the last one triggers",
    );
    assert_eq!(
        rlx_wgpu::device_wait_count() - waits,
        1,
        "the second parameter completes the set, and the settle waits exactly once",
    );
    let before = rlx_wgpu::device_submit_count();
    exe.set_param("w1", &vec![0.25f32; ELEMENTS]);
    assert_eq!(
        rlx_wgpu::device_submit_count() - before,
        pieces - 1,
        "a weight swap re-writes the pieces without settling again",
    );
    // A run stages its input, and this input is a piece and a quarter too: without the
    // run guard that upload would flush, giving the call a second submission.
    let before = rlx_wgpu::device_submit_count();
    exe.run(&[("x", &vec![1.0f32; ELEMENTS])]);
    let submits = rlx_wgpu::device_submit_count() - before;
    assert_eq!(
        submits, 1,
        "a two-step run made {submits} submissions, not the one both steps share"
    );
}
