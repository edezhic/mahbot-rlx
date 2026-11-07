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
//! - `b`: F32, `[7, 9]` — 63 elements, also 252 bytes, parked at 256.
//! - `y = narrow_(b, 1, 2, 4)`: a structural host step (`RLX_WGPU_FORCE_HOST`) reading
//!   `b`'s offset — 252 bytes of read against the 0 the record now bounds it with, and an
//!   encoding the record cannot attribute.
//!
//! The read has to be *emitted* for the net to be asked at all, and `b`'s rank is what
//! makes it: the self-copy elision ahead of the host-route branch in
//! `backend/compile/lower.rs` skips a narrow whose slot is its parent's plus
//! `start * inner * 4`, which is exactly where the planner aliases a pure view — so a
//! contiguous narrow of a whole f32 parameter is never a step, at any `start`, and the
//! `[N]` form this fixture used to carry reached no net. A narrow on axis 1 of `[7, 9]`
//! is not a pure view (`pure_view_offset` asks for every dim before the axis to be 1),
//! so it keeps a slot of its own and copies; the net's read is still the parent's, at
//! `in_byte_off` and `outer * axis_in * inner == 63` elements (`run_narrow_host_cached`),
//! which is the 252 bytes the record refuses.
//!
//! The refusal is raised inside the structural host runner, so it comes out of `run`.
//! Both levers are process-global, hence one test per file.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const N: usize = 63;

/// `b`'s `[7, 9]` row-major form: the same 63 elements, so `weight_param_bytes` still
/// sizes both parameters to 252 bytes and the sort still parks "a" before "b".
const B_SHAPE: [usize; 2] = [7, 9];

/// The view's byte delta must be the offset the planner gives the *next* parked
/// parameter, and both parameters must be sized by `weight_param_bytes` to 252 bytes.
fn two_parameters_under_one_offset(name: &str) -> Graph {
    let mut g = Graph::new(name);
    let a = g.param("a", Shape::new(&[N], DType::F64));
    let v = g.narrow_(a, 0, 32, 1);
    let b = g.param("b", Shape::new(&B_SHAPE, DType::F32));
    // Not `narrow_(b, 0, 0, 8)`: on axis 0 of a rank-1 parameter the planner aliases the
    // slot onto `b`'s and the self-copy elision drops the step (see the module doc).
    let y = g.narrow_(b, 1, 2, 4);
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
