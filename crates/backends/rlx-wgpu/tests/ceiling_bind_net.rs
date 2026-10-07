// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The net under the per-binding-ceiling rule, pinned in its own process.
//!
//! `bind_op_window` is the one bind path the absolute-offset arms use, and
//! `lower::should_host_at_ceiling` is what normally keeps them off it when the
//! memory they address cannot be bound whole at base zero. The net is what holds
//! when an arm is unclassified or a guard is dropped, so it has to be checked
//! against the geometry the two measured failures fell through: a striped arena
//! whose window for the node still starts at base zero — a window that is only
//! refused by the strengthened `binds_whole_at_base_zero` clause, not by the
//! weaker `base == 0` it replaced. The base is read back below and the node whose
//! window starts at zero is the one bound, so this check cannot quietly start
//! testing the weaker predicate instead.
//!
//! Reaching that geometry takes the shrunken-ceiling lever, which is process-global
//! — in the crate's own library target this check would race every other test
//! sharing that process and its device — so it lives here, in its own file, with its
//! own sane cap.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Small enough that a few megabyte-scale tensors cannot share one stripe.
const SHARD_MIB: usize = 4;
/// The per-stripe cap needs a per-stripe reserve too, and this is the knob for it.
/// Every tensor here is 3 MiB of a 4 MiB cap, so no stripe can hold two of them
/// whether or not it is set: the offsets come out 0/4/8 MiB either way, and only the
/// arena's tail padding differs (~11 vs ~12 MiB). The refusal below fires on a striped
/// arena either way, so it does not depend on this.
const STAGE_MIB: usize = 1;

#[test]
fn a_window_that_cannot_bind_whole_at_base_zero_is_refused_loudly() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: one test per file, so nothing else in this process reads these; both
    // knobs clamp toward smaller, and `ALLOW_SHARD` is what lets the arena stripe at
    // all (it is refused by default).
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_STAGE_MIB", STAGE_MIB.to_string());
        std::env::set_var("RLX_WGPU_ALLOW_SHARD", "1");
    }

    // Three 3 MiB tensors, of which at least two are live at the peak, so the
    // arena is over the 4 MiB cap and must stripe. A node in the first stripe
    // then has a window that starts at base zero — the geometry that only the
    // strengthened clause refuses (the weakened `base == 0` predicate would let
    // it through and run an absolute-offset kernel against a stripe-sized
    // window). If the cap did not stripe the arena, no window is refused and the
    // `expect_err` below fails saying so. No absolute-offset op is compiled here
    // — this is the bind path alone.
    const COLS: usize = 256;
    const ROWS: usize = 3 * 1024 * 1024 / 4 / COLS;
    let mut g = Graph::new("bind-net");
    let a = g.input("a", Shape::new(&[ROWS, COLS], F));
    let b = g.input("b", Shape::new(&[ROWS, COLS], F));
    let out = g.add(a, b);
    g.set_outputs(vec![out]);
    let exe = WgpuExecutable::compile(g);

    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    // Bind a window that starts at base zero, which is the only geometry that
    // isolates the strengthened clause (the weakened `base == 0` predicate would
    // let it through). Which node lands in stripe 0 is the planner's business, so
    // ask rather than assume — and fail loudly if none does, since then this check
    // could not distinguish the two predicates.
    let id = [a, b, out]
        .into_iter()
        .find(|&id| exe.test_window_for_nodes(device, &[id]).0 == 0)
        .expect("no base-zero window in this geometry — the check cannot isolate the bind clause");
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        exe.test_windowed_bind_for_nodes(device, rlx_wgpu::kernels::cumsum_kernel(device), &[id])
    }));
    let payload = caught.expect_err(
        "the windowed bind must be refused when the arena cannot be bound whole at base zero — \
         if this failed, the pinned cap did not stripe the arena",
    );
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    // Catchable and self-describing: a caller can tell this apart from a process
    // death, and it names the node the caller must classify.
    assert!(
        msg.contains("rlx-wgpu: REFUSED (arena not bindable whole)"),
        "refusal must carry the stable marker: {msg}"
    );
    assert!(
        msg.contains("addresses_arena_by_absolute_offset") && msg.contains("NodeId("),
        "refusal must name the case and the node: {msg}"
    );
}
