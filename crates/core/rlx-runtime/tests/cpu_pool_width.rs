// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "cpu")]

//! A requested pool width is the width compiled graphs run with.
//!
//! Only test in this binary: Rayon's pool is process-wide and created once, so
//! `RLX_WORKERS` is set before anything else can claim it. The case where
//! another Rayon user created the pool first is pinned by `rlx-cpu`'s
//! `tests/pool_width_claimed.rs`, and the refusal lines by `pool.rs`'s
//! unit tests.

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{PrecisionPolicy, Session};

#[test]
fn a_requested_width_is_the_width_compiled_graphs_run_with() {
    // A request wider than the machine is capped (and reported), so ask for a
    // width this host can actually run.
    let want = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(3);
    rlx_ir::env::set("RLX_WORKERS", want.to_string());

    let f = DType::F32;
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], f));
    let y = g.input("y", Shape::new(&[4], f));
    let out = g.binary(rlx_ir::op::BinaryOp::Add, x, y, Shape::new(&[4], f));
    g.set_outputs(vec![out]);

    let mut exec = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(g);

    // Compiling claims the process's pool (see `build_cpu_executable`), so Rayon
    // itself must already be at the requested width — not at its own default,
    // which is what a pool created lazily by the first kernel would have used.
    assert_eq!(rayon::current_num_threads(), want);
    assert_eq!(rlx_runtime::cpu_pool_width(), want);

    let out = exec.run(&[
        ("x", &[1.0, 2.0, 3.0, 4.0]),
        ("y", &[10.0, 20.0, 30.0, 40.0]),
    ]);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}
