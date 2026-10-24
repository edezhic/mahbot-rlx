// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The single check every generic-host-route operand read passes through.
//!
//! `Step::HostOp` stages each operand as `max(size_bytes / 4, elements)` f32 lanes,
//! and the ops that reach it *without* the per-binding-ceiling rule — an op with no
//! GPU kernel, a non-contiguous reduce, a non-last-axis softmax — get no plan-time
//! check. `scan_host::assert_host_reads_fit` is the net for those: it refuses an
//! operand whose storage the read would run past, judged on the slot the plan gave
//! that storage — and only where the graph says the operand is a parameter's, since its
//! record is keyed by a byte offset an activation can share with a released slot.
//!
//! The interesting geometry needs no ceiling lever: a `Bool` parameter in the ordinary
//! unsplit arena keeps a one-byte-per-element slot, while both the typed upload and the
//! staging use four — so the read runs past it and must be refused, and so must the
//! same parameter read through a *view* (the planner aliases a view onto its root's
//! storage, and only the root's remainder bounds what the view can read). An `F32`
//! parameter of the same shape is the control: accepted, and computed.
//!
//! `F16` was this file's example until the flat plan started reserving four bytes per
//! element for an `F16` parameter — the width `set_param_typed` widens it to and
//! `from_plan_split` already gave it — so that parameter is now readable and is no
//! longer a narrow-slot case. `Bool` still is.
//!
//! One test per file: the crate's convention for arena-level state.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const ROWS: usize = 8;
const COLS: usize = 4;

/// Non-last-axis softmax: an in-arm `Step::HostOp` with no ceiling in play.
fn host_op_graph(name: &str, dtype: DType, view: bool) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[ROWS, COLS], dtype));
    let (x, shape) = if view {
        // A `Narrow` view: its offset is inside the parameter's storage rather than at
        // its start, so its bound has to come from the root.
        (
            g.narrow_(p, 0, 1, ROWS - 1),
            Shape::new(&[ROWS - 1, COLS], DType::F32),
        )
    } else {
        (p, Shape::new(&[ROWS, COLS], DType::F32))
    };
    let y = g.softmax(x, 0, shape);
    g.set_outputs(vec![y]);
    g
}

fn refusal(graph: Graph, why: &str) {
    let mut exe = WgpuExecutable::compile(graph);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exe.run(&[])));
    let Err(payload) = caught else {
        panic!("{why}: the operand was staged instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("rlx-wgpu: REFUSED (operand not stageable as f32)")
            && msg.contains("Bool")
            && msg.contains("1 byte per element"),
        "{why}: the refusal must name the operand's real encoding and width; got {msg}"
    );
    assert!(
        msg.contains("in the activation arena"),
        "{why}: the refusal must say where the parameter lives; got {msg}"
    );
}

#[test]
fn host_route_reads_are_bounded_by_what_the_arena_holds() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // Control: an F32 parameter of the same shape is one f32 lane per element in a slot
    // that holds four bytes per element, so the same host step runs and computes.
    let mut exe = WgpuExecutable::compile(host_op_graph("host-fit-f32", DType::F32, false));
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i * 7) % 11) as f32 * 0.25 - 1.0)
        .collect();
    let mut want = vec![0.0f32; ROWS * COLS];
    for c in 0..COLS {
        let col: Vec<f32> = (0..ROWS).map(|r| x[r * COLS + c].exp()).collect();
        let sum: f32 = col.iter().sum();
        for r in 0..ROWS {
            want[r * COLS + c] = col[r] / sum;
        }
    }
    exe.set_param("p", &x);
    let got = exe.run(&[]);
    for (i, (a, b)) in got[0].iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6,
            "F32 parameter through the host route: element {i} {a} != {b}"
        );
    }

    // A Bool parameter: the typed upload widens it to one f32 lane per element, while
    // the plan gave it one byte per element — the read runs past it.
    refusal(
        host_op_graph("host-fit-bool", DType::Bool, false),
        "an in-arena Bool parameter",
    );
    // ... and the same parameter read through a `Narrow`: the view's own slot says
    // nothing about the storage it aliases, so the bound has to come from the root.
    refusal(
        host_op_graph("host-fit-bool-view", DType::Bool, true),
        "a view of an in-arena Bool parameter",
    );
}
