// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The unattributable-encoding refusal, reached end to end.
//!
//! `buffer::param_storage_bounds` keys its record by byte offset alone. Where a *view of
//! one parked parameter* is aliased onto the start of *another* parked parameter, two
//! different parameter roots claim one offset: the record keeps the tighter bound — the
//! view's own `slot.saturating_sub(off - root_off)`, which is 0 — and carries **no**
//! encoding, because it cannot say which of the two a read there reads. That is the
//! `dtype == None` arm of `buffer::refuse_unstageable_operand`, and the only caller that
//! can pass `None` is `scan_host::assert_read_fits`, the structural host steps' net
//! (Expand / Narrow / Transpose / Concat), because it alone reads `ParamStorage::dtype`.
//! The emitter is pinned device-free by the unit test
//! `scan_host::tests::an_unattributable_encoding_is_refused`; this file pins the route
//! that reaches it.
//!
//! The geometry, in the parked weight buffer (`RLX_WGPU_ALL_PARAMS_WEIGHT` — every
//! parameter parked, the sort places the two equal-size names in order "a" then "b"):
//!
//! - `a`: F64, 63 elements. `weight_param_bytes` gives a non-U8/I8 parameter one f32 lane
//!   per element, so it occupies `63 * 4 == 252` bytes at cursor 0, and the next parked
//!   parameter starts at `align256(252) == 256`.
//! - `v = narrow_(a, 0, 32, 1)`: the planner aliases a view onto its root at
//!   `element_index * dtype.size_bytes() == 32 * 8 == 256` — the *next* parameter's start.
//! - `b`: F32, 63 elements, also 252 bytes, parked at 256.
//! - `y = narrow_(b, 0, 0, 8)`: a structural host step (`RLX_WGPU_FORCE_HOST`) reading
//!   `b`'s offset — 252 bytes of read against the 0 the record now bounds it with, and an
//!   encoding the record cannot attribute.
//!
//! The refusal is raised inside the structural host runner, so it comes out of `run`.
//! Both levers are process-global, hence one test per file.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const N: usize = 63;

/// The view's byte delta must be the offset the planner gives the *next* parked
/// parameter, and both parameters must be sized by `weight_param_bytes` to 252 bytes.
fn two_parameters_under_one_offset(name: &str) -> Graph {
    let mut g = Graph::new(name);
    let a = g.param("a", Shape::new(&[N], DType::F64));
    let v = g.narrow_(a, 0, 32, 1);
    let b = g.param("b", Shape::new(&[N], DType::F32));
    let y = g.narrow_(b, 0, 0, 8);
    // Both outputs, so neither the view nor the step over `b` is dropped as dead.
    g.set_outputs(vec![v, y]);
    g
}

#[test]
fn a_read_bounded_by_two_parameters_storages_is_refused() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // Code overrides, not process-env writes: one test per file, and nothing else in this
    // process has to see them. The first lever parks every parameter, the second routes
    // the structural ops to the host steps — the two geometries the record needs.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");

    let mut exe = WgpuExecutable::compile(two_parameters_under_one_offset("two-params-one-offset"));
    // The refusal is raised inside the structural host runner, not at compile time.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exe.run(&[])));
    let Err(payload) = caught else {
        panic!("a read the record cannot attribute was staged instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("rlx-wgpu: REFUSED (operand not stageable as f32)")
            && msg.contains("Two parameters' storages share this offset")
            && msg.contains("not attributable"),
        "the structural host step must refuse with the unattributable-encoding message; got {msg}"
    );
    assert!(
        msg.contains("Narrow") && msg.contains("parked in the weight buffer"),
        "the refusal must name the step and where the operand lives; got {msg}"
    );
}
