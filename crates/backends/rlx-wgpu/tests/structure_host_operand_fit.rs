// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The structural host steps' operand reads are bounded as well.
//!
//! `RLX_WGPU_FORCE_HOST` routes Expand / Narrow / Transpose / Concat to the host (a
//! discrete Vulkan/DX12 adapter does that by itself), where each stages its operand
//! through its own descriptor rather than `Step::HostOp` — and so is out of reach of the
//! route's own check. `scan_host::assert_read_fits` is the net for those: the operand's
//! byte offset and element count against the arena's record of what that storage holds,
//! and only where the graph flagged the operand as a parameter's storage — the flag the
//! step carries, the same gate `scan_host::assert_host_reads_fit` applies.
//!
//! The interesting geometry needs no ceiling lever: a `Bool` parameter in the ordinary
//! unsplit arena keeps a one-byte-per-element slot, while the step reads four bytes per
//! element, so the read runs past it and must be refused. An `F32` parameter of the same
//! shape is the control — accepted, and computed. Running every arm the lever reaches
//! with that parameter is also what makes each arm's carried flag load-bearing: with the
//! flag false the net is a no-op and the `Bool` read is staged instead of refused.
//! (`F16` was this file's example until the flat plan started reserving the four bytes
//! per element its upload writes — the width `from_plan_split` already gave it; see
//! `tests/host_route_operand_fit.rs`.)
//! `ConcatHostPieces` is the one host arm the lever does not reach — it needs a mixed
//! GPU/host concat, which `RLX_WGPU_FORCE_HOST` never produces (it hosts every input,
//! giving `ConcatHost`) — and its flag is pushed in the same loop as its operand list, so
//! the two cannot drift apart.
//!
//! One test per file: the crate's convention for arena-level state.

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const ROWS: usize = 8;
const COLS: usize = 4;

/// A narrowed copy of a parameter. Under the host-structural routing the lowering emits
/// `Step::NarrowHost`, whose operand read is the one `assert_read_fits` bounds.
fn narrow_graph(name: &str, dtype: DType) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[ROWS, COLS], dtype));
    let y = g.narrow_(p, 0, 1, ROWS - 1);
    g.set_outputs(vec![y]);
    g
}

/// A transposed parameter — `Step::TransposeHost` under the same lever.
fn transpose_graph(name: &str, dtype: DType) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[ROWS, COLS], dtype));
    let y = g.transpose_(p, vec![1, 0]);
    g.set_outputs(vec![y]);
    g
}

/// A broadcast parameter — `Step::ExpandHost` under the same lever.
fn expand_graph(name: &str, dtype: DType) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[1, COLS], dtype));
    let y = g.add_node(
        Op::Expand {
            target_shape: vec![ROWS as i64, COLS as i64],
        },
        vec![p],
        Shape::new(&[ROWS, COLS], dtype),
    );
    g.set_outputs(vec![y]);
    g
}

/// Two parameters joined on the outer axis — `Step::ConcatHost` under the same lever.
fn concat_graph(name: &str, dtype: DType) -> Graph {
    let mut g = Graph::new(name);
    let a = g.param("a", Shape::new(&[ROWS, COLS], dtype));
    let b = g.param("b", Shape::new(&[ROWS, COLS], dtype));
    let y = g.concat_(vec![a, b], 0);
    g.set_outputs(vec![y]);
    g
}

fn refusal(graph: Graph, op: &str, why: &str) {
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
        msg.contains("rlx-wgpu: REFUSED (operand not stageable as f32)") && msg.contains(op),
        "{why}: got {msg}"
    );
    assert!(
        msg.contains("in the activation arena")
            && msg.contains("narrower than the four bytes per element"),
        "{why}: the refusal must say where the operand is and why the read does not fit; got {msg}"
    );
    // The message must name the operand's own declared encoding and width: these steps
    // read them from the arena's parameter record rather than from a descriptor, so a
    // stand-in would be a false statement about a parameter of another encoding.
    assert!(
        msg.contains("Bool") && msg.contains("1 byte per element"),
        "{why}: the refusal must name the operand's real encoding and width; got {msg}"
    );
}

/// The control for an arm: `F32` is one f32 lane per element in a slot that holds four
/// bytes per element, so the same host step reads it and computes.
fn accepted(graph: Graph, op: &str) {
    let mut exe = WgpuExecutable::compile(graph);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exe.run(&[])));
    if let Err(payload) = caught {
        panic!("{op}: an F32 parameter must be staged, not refused — {payload:?}");
    }
}

#[test]
fn structural_host_reads_are_bounded_by_what_the_arena_holds() {
    // Before the first compile: the lowering reads this to decide which ops the host
    // runs, and the planner to keep their slots reserved.
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    // Control: an F32 parameter is one f32 lane per element in a slot that holds four
    // bytes per element, so the same host step reads it and computes.
    let mut exe = WgpuExecutable::compile(narrow_graph("structure-fit-f32", DType::F32));
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i * 7) % 11) as f32 * 0.25 - 1.0)
        .collect();
    exe.set_param("p", &x);
    let got = exe.run(&[]);
    let want = &x[COLS..];
    assert_eq!(
        got[0].len(),
        want.len(),
        "the narrowed output is {ROWS}-1 rows of {COLS}"
    );
    for (i, (a, b)) in got[0].iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6,
            "F32 parameter through the host step: element {i} {a} != {b}"
        );
    }

    // A Bool parameter: the flat plan gave it one byte per element while the step reads
    // four, so the read runs past it.
    refusal(
        narrow_graph("structure-fit-bool", DType::Bool),
        "Narrow",
        "an in-arena Bool parameter at a structural host step",
    );

    // The other host-structural arms the lever reaches: each is a step variant of its
    // own, and each carries the parameter flag its net gates on, so each is run here —
    // the F32 graph accepted, the Bool one refused.
    for (arm, build, op) in [
        (
            "transpose",
            transpose_graph as fn(&str, DType) -> Graph,
            "Transpose",
        ),
        ("expand", expand_graph, "Expand"),
        ("concat", concat_graph, "Concat"),
    ] {
        accepted(build(&format!("structure-fit-{arm}-f32"), DType::F32), op);
        refusal(
            build(&format!("structure-fit-{arm}-bool"), DType::Bool),
            op,
            "an in-arena Bool parameter at a structural host step",
        );
    }
}
