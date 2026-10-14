// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d`'s device step when the operand it must read is larger
//! than the staging reserve it has to copy it into.
//!
//! The device step addresses its operands through a bind window, and a parked
//! parameter is never inside that window, so it stages a copy into the window's
//! reserve (`crates/backends/rlx-wgpu/src/backend/compile/lower.rs`,
//! `assert_conv_transpose_operands_addressable`). That guard asks the shared
//! staging path's own predicates, and where the shared path would refuse it refuses
//! first — quoting the operand and the cap it exceeds, which the shared message
//! cannot name.
//!
//! Reached with the striping levers the other pins use: a small shard cap and a
//! 1 MiB stage reserve (the reserve *is* what the cap is on a striped arena), a
//! weight past the reserve but inside the shard cap so it is really parked, and
//! `RLX_WGPU_SHARD_GPU=1` to keep the op on the device — without it a sharded
//! arena hosts this op and the device route is unreachable. `RLX_WGPU_QUIET_SHARD=1`
//! silences only the informational sharding logs, never the opt-in warning.
//!
//! **What the refusal names.** The guard belongs to this op — its only two call
//! sites are in `Op::ConvTranspose2d`'s device arm — so the message names the op,
//! the operand node and its byte length, and the cap it exceeds. What it names as
//! the way out is *not* the host route: this fixture's operand is parked, and the
//! host step refuses a parked operand too (it stages through the same reserves), so
//! sending the reader there would be advice that does not work. It names the escape
//! a parked operand really has — keep it in the activation arena (drop
//! `RLX_WGPU_ALL_PARAMS_WEIGHT`, or let the split plan keep the weights) — and
//! states the two things that are *not* ways out: the reserve knob, which only lowers
//! the cap, and the host route, which refuses a parked operand. Only the parked form
//! is reachable here: an oversized *activation* operand on a striped arena sits in
//! another stripe, where the straddle guard hosts this op before the staging guard
//! can fire.
//!
//! Its own test binary: every lever here is process-global and read at compile
//! time.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// Shard/staging levers, exercised at kilobyte-to-megabyte scale like
/// `ceiling_refusal.rs` / `sharded_arena.rs` do.
const SHARD_MIB: usize = 4;
const STAGE_MIB: usize = 1;
/// The weight the device step would have to stage: `8·8·8192·4 = 2 MiB`, past
/// the 1 MiB reserve but inside the 4 MiB shard cap, so the planner parks it
/// rather than leaving it in the act arena. `NodeId(1)` is its node and its
/// length is what the refusal quotes.
const C_IN: usize = 8;
const C_OUT: usize = 8;
const K: usize = 8192;
const W_IN: usize = 8;
const WEIGHT_NODE: &str = "NodeId(1)";
const WEIGHT_BYTES: usize = C_IN * C_OUT * K * 4;

#[test]
fn a_parked_operand_past_the_staging_reserve_is_refused_by_name_and_without_a_host_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", SHARD_MIB.to_string());
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", STAGE_MIB.to_string());
    rlx_ir::env::set("RLX_WGPU_QUIET_SHARD", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    let mut g = Graph::new("conv_transpose2d_stage_refusal");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, W_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 1, K], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, K], [1, 1], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);

    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(g)));
    let Err(payload) = caught else {
        panic!("an operand past the staging reserve was accepted instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    eprintln!("stage refusal: {msg}");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (Op::ConvTranspose2d operand too large to stage)"),
        "the refusal must carry this op's own guard template rather than the shared \
         staging helper's; got: {msg}"
    );
    assert!(
        msg.contains(WEIGHT_NODE) && msg.contains(&WEIGHT_BYTES.to_string()),
        "the refusal must be about the parked weight this op could not stage; got: {msg}"
    );
    assert!(
        msg.contains(&(STAGE_MIB * 1024 * 1024).to_string()),
        "the refusal must name the cap it exceeds; got: {msg}"
    );
    // The escapes it names belong to a *parked* operand: unpark it (the lever this
    // fixture forces, and what a split plan does when the arena cannot hold the
    // weights), which is the way out. `RLX_WGPU_SHARD_STAGE_MIB` appears for its
    // direction — it only lowers this cap — not as a lever that could widen it.
    assert!(
        msg.contains("RLX_WGPU_ALL_PARAMS_WEIGHT")
            && msg.contains("RLX_WGPU_SHARD_STAGE_MIB")
            && msg.contains("the host route cannot take a parked operand either"),
        "the refusal must name the escapes this operand really has; got: {msg}"
    );
    // The host levers are absent on purpose, and that absence is the point of the
    // fixture: a parked operand is refused by the host route too, so the shared
    // staging message's "run it on the host" would be advice that does not work.
    assert!(
        !msg.contains("RLX_WGPU_CONV_HOST") && !msg.contains("RLX_WGPU_FORCE_HOST"),
        "the refusal must not offer the host route for an operand that route refuses too; \
         got: {msg}"
    );
}
