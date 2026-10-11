// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sharded activation arenas (> wgpu max_buffer_size / ~4 GiB).

mod common;

use rlx_ir::NodeId;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::memory::{BufferSlot, MemoryPlan};
use rlx_runtime::{Device, Session};
use rlx_wgpu::buffer::{Arena, effective_shard_cap};
use std::collections::HashMap;

/// Shard cap and stage reserve forced small, so striping is exercised at
/// kilobyte scale.
///
/// These tests used to size themselves off the adapter's real
/// `max_buffer_size`. On a discrete Vulkan adapter that is 4 GiB, so a
/// `shard_cap * 2` plan snapped to three 4 GiB buffers and died with
/// `wgpu error: Out of Memory`; they passed on Apple only because its smaller
/// limit happened to fit in unified memory. The striping LOGIC is what is under
/// test, and it does not care how big a shard is — so pin a small one and the
/// same assertions run everywhere.
const TEST_SHARD_MIB: usize = 4;
const TEST_STAGE_MIB: usize = 1;

/// Install the small-shard environment. Both knobs clamp toward smaller, so this
/// can never ask an adapter for more than it supports.
fn force_small_shards() {
    // SAFETY: tests in this binary run single-threaded w.r.t. this setup, and
    // both vars are read on each `from_plan` call rather than cached.
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", TEST_SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_STAGE_MIB", TEST_STAGE_MIB.to_string());
        // Only the informational sharding logs are silenced. The opt-in warning
        // is not quietable by design, so each striping decision still prints one.
        std::env::set_var("RLX_WGPU_QUIET_SHARD", "1");
        // Striping is refused by default now — it produces wrong results
        // silently. These tests are about the striping machinery itself, so
        // they opt back in.
        std::env::set_var("RLX_WGPU_ALLOW_SHARD", "1");
    }
}

fn fake_plan(arena_size: usize, slots: &[(u32, usize, usize)]) -> MemoryPlan {
    let mut assignments = HashMap::new();
    for &(id, offset, size) in slots {
        assignments.insert(NodeId(id), BufferSlot { offset, size });
    }
    MemoryPlan {
        arena_size,
        assignments,
        schedule: Vec::new(),
    }
}

#[test]
fn sharded_from_plan_stripes_and_reserves_stage() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let shard_cap = effective_shard_cap(device);
    let usable = shard_cap
        .saturating_sub(rlx_wgpu::buffer::shard_stage_reserve())
        .max(256);
    // Two large slots that cannot share one stripe's usable region.
    let slot = usable - 4096;
    let plan = fake_plan(
        shard_cap * 2,
        &[
            (0, 0, slot),
            (1, shard_cap, slot),
            (2, shard_cap + slot, 4096),
        ],
    );
    let arena = Arena::from_plan(device, &plan);
    assert!(
        arena.is_sharded(),
        "expected sharded arena (logical {} > shard cap {})",
        arena.size,
        shard_cap
    );
    assert!(!arena.extra_shards.is_empty());
    assert_eq!(arena.shard_size, shard_cap);

    for (&id, &off) in &arena.offsets {
        let len = arena.len_of(id);
        let local = off % shard_cap;
        assert!(
            local + len <= usable,
            "node {id:?} @ {off}+{len} invades stage reserve (usable={usable})"
        );
    }

    let id = NodeId(2);
    let payload = vec![1.0f32, 2.0, 3.0, 4.0];
    arena.write_f32(&wgpu.queue, id, &payload);
    let got = arena.read_f32(device, &wgpu.queue, id);
    assert_eq!(&got[..4], &payload[..]);
}

#[test]
fn bind_spec_stays_inside_one_shard() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let shard_cap = effective_shard_cap(device);
    let usable = shard_cap
        .saturating_sub(rlx_wgpu::buffer::shard_stage_reserve())
        .max(256);
    let slot = usable - 4096;
    let plan = fake_plan(shard_cap * 2, &[(0, 0, slot), (1, shard_cap, 4096)]);
    let arena = Arena::from_plan(device, &plan);
    assert!(arena.is_sharded());

    let spec = arena.bind_spec_for_nodes(device, &[NodeId(0)]);
    assert_eq!(spec.local_base, 0);
    assert_eq!(spec.rebase, 0);

    let off1 = arena.offset(NodeId(1));
    let spec2 = arena.bind_spec_for_nodes(device, &[NodeId(1)]);
    assert_eq!(spec2.rebase, (off1 / shard_cap * shard_cap) as u64);
    assert_eq!(spec2.local_base, 0);
}

/// The coarse condition behind the per-binding-ceiling routing rule
/// (`should_host_at_ceiling` → `supported_ops::addresses_arena_by_absolute_offset`):
/// a kernel carrying absolute arena offsets is dispatchable only when the whole
/// arena binds at base zero, and only while its word offsets fit the `u32` the
/// uniform declares.
#[test]
fn binds_whole_at_base_zero_covers_striping_limit_and_u32_words() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let shard_cap = effective_shard_cap(device);

    // Striped: never bindable whole at base zero, whatever the limit says.
    let striped = Arena::from_plan(
        device,
        &fake_plan(shard_cap * 2, &[(0, 0, shard_cap - 4096)]),
    );
    assert!(striped.is_sharded());
    assert!(!striped.binds_whole_at_base_zero(u64::MAX));

    // Unsharded: the caller-supplied limit decides, so an arena over the
    // device's real per-binding ceiling is testable without allocating it.
    let plan = fake_plan(4096, &[(0, 0, 4096)]);
    let mut arena = Arena::from_plan(device, &plan);
    assert!(!arena.is_sharded());
    assert!(arena.binds_whole_at_base_zero(4096));
    assert!(!arena.binds_whole_at_base_zero(2048));

    // Word offsets: `size` is only read here, so a synthetic size past
    // `u32::MAX` words probes the truncation branch with no 16 GiB allocation.
    // (`checked_*` because only a 64-bit `usize` can express that size at all.)
    if let Some(beyond_u32_words) = (u32::MAX as usize)
        .checked_add(1)
        .and_then(|words| words.checked_mul(4))
    {
        arena.size = beyond_u32_words;
        assert!(!arena.binds_whole_at_base_zero(u64::MAX));
    }
}

/// The **default** sharded routing (host, not `RLX_WGPU_SHARD_GPU`) combined with
/// the per-channel operand the lowering peeled
/// (`rlx_wgpu::unfuse::elide_binary_broadcast_expands`): the peel is what puts
/// the operand back to `(i / rep) % len`, and the host route has to read it that
/// way out of a striped arena.
///
/// `[1,C,1]` against `[1,C,T]` is 2 MiB per tensor, and two outputs keep three of
/// them live at once — past the 4 MiB shard cap below, while each single tensor
/// still fits a stripe's usable region (stage reserve 1 MiB). A wrong-byte read
/// of the peeled operand is O(1) wrong, far above the 1e-4 bar.
#[test]
fn sharded_host_route_reads_a_peeled_per_channel_operand() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();

    const C: usize = 256;
    const T: usize = 2048; // C*T f32 = 2 MiB per tensor
    const F: DType = DType::F32;

    let xs: Vec<f32> = (0..C * T)
        .map(|i| ((i * 41 % 173) as f32) / 8.0 - 4.0)
        .collect();
    let bias: Vec<f32> = (0..C).map(|i| i as f32 * 0.125 - 0.5).collect();

    let act = Shape::new(&[1, C, T], F);
    let mut g = Graph::new("sharded_peeled_param");
    let x = g.input("x", act.clone());
    let b = g.param("b", Shape::new(&[1, C, 1], F));
    let y0 = g.add_node(Op::Binary(BinaryOp::Add), vec![x, b], act.clone());
    let y1 = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, b], act);
    g.set_outputs(vec![y0, y1]);

    // The count is what anchors the peel here: this route gives an un-peeled copy a
    // step of its own (`expand_host` where it can be staged, `host_op` where its
    // operands straddle stripes — the plan decides), so the peeled schedule is one
    // step per elementwise leg and nothing in front of it.
    let names = common::legalised_step_names(&g);
    assert!(
        names.iter().all(|n| *n == "host_op"),
        "expected every step on the sharded host route, got {names:?} — a device step here \
         means the arena did not stripe or the route changed"
    );
    // The count is what anchors the peel here: this route gives an un-peeled copy a
    // step of its own (`expand_host` where it can be staged, `host_op` where its
    // operands straddle stripes — the plan decides), so the peeled schedule is one
    // step per elementwise leg and nothing in front of it.
    assert_eq!(
        names.len(),
        2,
        "one step per elementwise leg and no copy for the per-channel operand: {names:?}"
    );
    eprintln!("  sharded host route: steps {names:?}");

    let mut gpu = Session::new(Device::Gpu).compile(g.clone());
    gpu.set_param("b", &bias);
    let got = gpu.run(&[("x", &xs)]).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    cpu.set_param("b", &bias);
    let want = cpu.run(&[("x", &xs)]).remove(0);
    let max_d = common::max_abs_diff(&got, &want);
    // This pair is Add/Mul in f32 through the same read form, so it has to be
    // bit-exact: the bar is the strictest one the suite can state, and a peeled
    // operand read out of the wrong stripe is wrong by O(1).
    assert_eq!(
        max_d, 0.0,
        "sharded host route: wgpu differs from the CPU reference by {max_d:.3e} — a peeled \
         per-channel operand read out of the wrong stripe is wrong by O(1)"
    );
}
