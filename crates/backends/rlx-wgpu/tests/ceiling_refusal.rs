// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The refusing half of the per-binding-ceiling work: a configuration the
//! engine cannot model correctly is refused loudly at plan time, before
//! anything executes, with a message a caller can tell apart from a crash.
//!
//! `RLX_WGPU_ALLOW_SHARD` is deliberately NOT set here — and removed from the
//! ambient environment — because it is that opt-in which turns the refusal into
//! a warn-and-continue. `RLX_WGPU_SHARD_CAP_MIB` is the lever that reaches the
//! geometry without a multi-gigabyte allocation.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Small enough that a few megabyte-scale tensors cannot share one stripe — the
/// same pinned cap the other striping tests use.
const SHARD_MIB: usize = 4;
const STAGE_MIB: usize = 1;

#[test]
fn striping_is_refused_loudly_at_plan_time_before_anything_executes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: one test per file, so no other thread reads these; the knobs
    // clamp toward smaller and cannot ask an adapter for more than it supports.
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_STAGE_MIB", STAGE_MIB.to_string());
        std::env::remove_var("RLX_WGPU_ALLOW_SHARD");
    }

    const COLS: usize = 256;
    const F_ROWS: usize = 3 * 1024 * 1024 / 4 / COLS;
    const A_ROWS: usize = 1024 * 1024 / 4 / COLS;
    let mut g = Graph::new("refuse");
    let f = g.param("f", Shape::new(&[F_ROWS, COLS], F));
    let s1 = g.sum(f, vec![1], false);
    let a = g.input("a", Shape::new(&[A_ROWS, COLS], F));
    let cs = g.cumsum(a, -1, false, Shape::new(&[A_ROWS, COLS], F));
    let s2 = g.sum(f, vec![0], false);
    g.set_outputs(vec![cs, s1, s2]);

    // `compile` is where the arena is planned and the refusal fires, so a
    // returned executor would mean the graph was accepted with nothing executed
    // — the refusal must come first.
    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(g)));
    let Err(payload) = caught else {
        panic!("striping was accepted instead of refused at plan time");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("rlx-wgpu: REFUSED (striped activation arena)"),
        "refusal must carry a stable marker a caller can distinguish from a process \
         death; got: {msg}"
    );
    assert!(
        msg.contains("RLX_WGPU_ALLOW_SHARD"),
        "refusal must name the case and the opt-in that reaches it; got: {msg}"
    );
}
