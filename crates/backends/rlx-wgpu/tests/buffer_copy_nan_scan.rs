// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The fail-fast NaN diagnostic must still scan a graph whose schedule ends with a
//! device-route buffer copy — the shape `KvAppend` lowers to on this backend, which
//! now rides the enclosing batch instead of ending the run through a host step. The
//! scan runs on every tail that reads outputs back
//! (`WgpuExecutable::debug_scans_after_run`), not only on the run loop's exit,
//! because a joined run leaves through its own tail. `buffer_copy_batch_join.rs`
//! guards `RLX_DEBUG_NANS` (the scan's reads move its counters), so it cannot cover
//! this, and the counters are process-global: one test fn in a binary of its own.
//!
//! The evidence is the scan's own device reads. A completed run reads only its
//! outputs back, so switching the scan on must grow the submit/wait pair by one of
//! each per node it reads — the `KvAppend` output plus its two f32 operands here —
//! and must not change the results. Fail-fast mode then has to localize a NaN that
//! the joined copy wrote.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// Rows in the cache, the row width, and where the new row lands: a non-zero offset
/// inside a long-lived slot, as in `buffer_copy_batch_join.rs`.
const CAP: usize = 64;
const W: usize = 64;
const POS: usize = CAP - 2;

/// The knobs that would change this graph's node residency or routing — and so the
/// number of slots the scan reads — or what a submission or a wait means. The
/// sibling counter pins guard the same set; `RLX_DEBUG_NANS` and its alias are this
/// pin's own switches, set and unset inside it.
const GUARDED: [&str; 15] = [
    "RLX_WGPU_SNAPSHOT",
    "RLX_BENCH_DISPATCH_ONLY",
    "RLX_WGPU_ALL_PARAMS_WEIGHT",
    "RLX_WGPU_HOST_EAGER_H2D",
    "RLX_WGPU_HOST_BUFFER_COPY",
    "RLX_WGPU_CONCAT_HOST",
    "RLX_WGPU_FORCE_HOST",
    "RLX_WGPU_HOST_MATMUL",
    "RLX_DEBUG_NANS",
    "RLX_WGPU_NAN_TRACE",
    "RLX_WGPU_DISCRETE_HOST",
    "RLX_WGPU_SHARD_CAP_MIB",
    "RLX_WGPU_SHARD_STAGE_MIB",
    "RLX_WGPU_SHARD_GPU",
    "RLX_WGPU_ALLOW_SHARD",
];

/// A row write into an existing cache slot — `KvAppend`'s lower on this backend —
/// as the schedule's only step, so the copy is what the run ends with.
fn kv_row_graph() -> Graph {
    let mut g = Graph::new("nan_scan_kv_row");
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let row = g.input("row", Shape::new(&[1, 1, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    g.set_outputs(vec![k]);
    g
}

#[test]
fn a_copy_ending_graph_is_still_scanned() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // This pin owns the scan's own switches, so its two readings cannot be mixed up
    // with an operator's; the rest would move what the readings count.
    for key in GUARDED {
        assert!(
            !rlx_ir::env::flag(key),
            "{key} changes this graph's residency, routing or submission shape; unset \
             it to run this pin"
        );
    }

    let g = kv_row_graph();
    // The premise: the row write is the schedule's last step, so the run returns
    // through a tail that has to scan — a graph ending in an unjoined device step
    // would not exercise this.
    assert_eq!(
        rlx_wgpu::backend::WgpuExecutable::compile(g.clone()).test_step_names(),
        ["buffer_copy"],
        "the row write must be the schedule's last step"
    );

    let mut exe = Session::new(Device::Gpu).compile(g);
    exe.set_param("cache_k", &vec![0.013f32; CAP * W]);
    let row = vec![0.5f32; W];

    reset_device_counts();
    let off = exe.run(&[("row", &row)]);
    let (s_off, w_off) = (device_submit_count(), device_wait_count());

    rlx_ir::env::set("RLX_DEBUG_NANS", "1");
    reset_device_counts();
    let on = exe.run(&[("row", &row)]);
    let (s_on, w_on) = (device_submit_count(), device_wait_count());
    rlx_ir::env::unset("RLX_DEBUG_NANS");
    eprintln!("[nan-scan] knob off: {s_off}/{w_off}, knob on: {s_on}/{w_on}");

    // One submit and one wait per node the scan reads: the `KvAppend` output and its
    // two f32 operands. A scan that never reached this tail adds none of them, which
    // is what this reading catches.
    assert_eq!(
        (s_on - s_off, w_on - w_off),
        (3, 3),
        "the per-node scan must read every node back on a copy-ending graph; measured \
         {s_off}/{w_off} submissions/waits without it and {s_on}/{w_on} with it"
    );
    // Turning the diagnostic on changes what the run reads, never what it computes.
    assert_eq!(on, off, "the scan must not change the results");

    // Fail-fast: the scan must localize a NaN the joined copy wrote. While the scan
    // did not reach this tail, the run completed and the NaN went unreported.
    rlx_ir::env::set("RLX_DEBUG_NANS", "abort");
    let nan_row = vec![f32::NAN; W];
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exe.run(&[("row", &nan_row)]);
    }));
    rlx_ir::env::unset("RLX_DEBUG_NANS");
    assert!(
        caught.is_err(),
        "the fail-fast scan must abort on the NaN the joined copy wrote"
    );
}
