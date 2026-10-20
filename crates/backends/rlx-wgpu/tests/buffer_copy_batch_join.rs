// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a KV row write costs the run it sits in: submissions and device waits,
//! from the backend's process-global counters
//! ([`rlx_wgpu::device_submit_count`] / [`rlx_wgpu::device_wait_count`]).
//!
//! A decode step's row write lowers to `Step::BufferCopy`. This backend's plan
//! classifies that step as a host step, so by default every occurrence ended the
//! enclosing run of device work, and its run-time device route then created a
//! command encoder of its own, submitted it, and blocked on the device — per
//! call. A copy whose route computes on the device must ride the enclosing
//! batch, the way the one-row packed multiply does (`tests/gguf_batch_join.rs`).
//!
//! Four brackets, each one forward pass between a counter reset and a read of
//! the counters. **1 — the motivating geometry**, `x → flip → kv_append(cache_k)
//! → kv_append(cache_v) → flip → flip`: two row writes whose destination is a
//! long-lived cache parameter and whose source is an activation produced by
//! device work, inside a run of device work that reads the rows they wrote — so a
//! write that failed to reach the other dispatches, or reached them out of order,
//! changes the output. Each bracket warms up on a *different* row (`x_warm`): the
//! caches are uploaded once, so a warm-up with the bracketed row would leave the
//! bytes that run writes already in place and a copy that never ran would still
//! match. **2 — the same writes with nothing after them**, so the copies are the
//! schedule's last steps. **3 — the host route of graph 2**, forced with
//! `RLX_WGPU_HOST_BUFFER_COPY=1`. **4 — graph 1 with a packed multiply whose route
//! is the one-row device arm**, so one batch carries both kinds of joined call and
//! the two cannot disagree about the batch's own encoder; the multiply's route
//! census is read there too.
//!
//! The readings, per bracket, as `(submissions, waits)` for one forward pass —
//! *before* the join (measured with the counters in place, in the default mode,
//! from a tree with the join not yet applied in the source; re-measured on this
//! tree with nothing but this change's own join rule disabled — the device route
//! declining to join — which reads the same `(4, 5)` in all four brackets) and
//! after it:
//!
//! | bracket | before | after |
//! |---|---|---|
//! | 1 device route, device work after | (4, 5) | **(1, 1)** |
//! | 2 device route, copies last       | (4, 5) | **(1, 1)** |
//! | 3 host route forced               | (4, 5) | **(4, 5)** |
//! | 3b host route, device work after  | (4, 5) | parity only |
//! | 4 both join kinds, device work    | (4, 5) | **(1, 1)** |
//!
//! Each device-route copy used to pay one submission and two waits: its own
//! poll, plus the poll it forced at the boundary in front of the run of device
//! work it ended. Both now ride the batch's single submission, so what is left
//! in brackets 1, 2 and 4 is that batch's own submission and the output readback's
//! wait. Bracket 3 is unchanged on purpose: the host route keeps its own
//! submission, its own waits and its own readback (its D2H read and the drained
//! boundary), with the device drained before it goes through the host.
//!
//! `RLX_WGPU_ONE_OP_PER_PASS=1` — the step-by-step mode, which the snapshot also
//! forces — makes the join inert, so each copy submits on its own again: the same
//! binary then reads (5, 5) / (4, 5) / (4, 5) / (7, 7). That mode is the cross-check
//! of the self-submitting shape the debugging paths need, an upper bound rather
//! than the pre-change reading above.
//!
//! The counts are adapter-independent: every step of these graphs is chosen
//! without an adapter-kind term (the flip and the copy route have none, and
//! nothing around them has), and no elementwise step follows a copy, so the one
//! arm of `step_needs_pass_flush` that differs on Vulkan/DX12-class adapters
//! never decides here.
//!
//! Parity: a row write is a byte move and a flip is a permutation, so brackets 1,
//! 2 and 3 and bracket 4's two row-write outputs must equal the CPU reference bit
//! for bit — the bound this crate's own copy assertions use for a byte move
//! (`tests/buffer_copy_exact.rs`, and `kv_append_backends.rs`, whose graphs these
//! are). Bracket 4's packed-multiply output takes the bound its own shapes are
//! asserted at in `gguf_batch_join.rs` (Q4_K, a whole number of superblocks, one
//! row) with that test's accompanying non-zero check. The route decision itself is
//! pinned without a device, in `copy_route.rs`'s own ladder tests
//! (`only_the_device_route_joins_the_enclosing_batch`): this step has no runtime
//! route counter to read.
//!
//! Graph 2 exists for bracket 3: a host-route row write is carried to the device
//! by the run's own final flush, and having the copies last keeps that bracket
//! inside the mechanism the host route is pinned by elsewhere
//! (`gguf_batch_join.rs` bracket 5, for a following joined multiply).
//!
//! Single test fn on purpose: the counters are process-global, so this file owns
//! them, resets them per bracket, and runs as its own test binary. The one knob it
//! sets — `RLX_WGPU_HOST_BUFFER_COPY`, inside brackets 3 and 3b — is unset inside
//! each bracket, and the guard below rejects the knobs that would change what a
//! submission or a wait means.

mod common;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::gguf_gpu::{reset_route_counts, route_counts};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// Rows in each KV cache.
const CAP: usize = 512;
/// Row width.
const W: usize = 64;
/// Where the new row is written: a non-zero offset inside a long-lived slot, so
/// the copy's destination sits far below the activation it reads.
const POS: usize = CAP - 2;
/// The packed multiply's contraction — one Q4_K superblock, the whole number of
/// packing blocks the one-row arm answers over (bracket 4).
const KM: usize = 256;
/// Its output width.
const NM: usize = 64;

/// Deterministic values for an `n`-element tensor or parameter.
fn feed(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * step).sin()).collect()
}

/// The two row writes of a decode step, inside a run of device work.
///
/// The row is a flip's output (an activation, not an input), each destination
/// aliases a cache `Param`'s slot, and the flips after the writes read what they
/// wrote — so parity fails if a write does not reach the other dispatches.
fn kv_row_graph() -> Graph {
    let mut g = Graph::new("wgpu_buffer_copy_batch_join");
    let x = g.input("x", Shape::new(&[1, 1, W], DType::F32));
    let row = g.reverse(x, vec![2]);
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let cv = g.param("cache_v", Shape::new(&[1, CAP, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let v = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![cv, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let kr = g.reverse(k, vec![2]);
    let vr = g.reverse(v, vec![2]);
    g.set_outputs(vec![kr, vr]);
    g
}

/// [`kv_row_graph`] without the flips after the writes, so the copies are the
/// schedule's last steps (bracket 3's shape).
fn kv_row_only_graph() -> Graph {
    let mut g = Graph::new("wgpu_buffer_copy_batch_join_last");
    let x = g.input("x", Shape::new(&[1, 1, W], DType::F32));
    let row = g.reverse(x, vec![2]);
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let cv = g.param("cache_v", Shape::new(&[1, CAP, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let v = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![cv, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    g.set_outputs(vec![k, v]);
    g
}

/// The two row writes followed by device work that reads the rows they wrote.
///
/// The flips are whole-arena host steps, so a graph built only of them and the
/// copies never dispatches before the writes; this one starts with a device
/// step (`tanh`), which is what makes the mirror's invalidation path live when
/// the copies run, and ends at a device consumer (`softmax`) instead of the
/// final flush, which is what makes the copy's own delivery live.
fn kv_row_device_consumer_graph() -> Graph {
    let mut g = Graph::new("wgpu_buffer_copy_batch_join_device_consumer");
    let x = g.input("x", Shape::new(&[1, 1, W], DType::F32));
    let t = g.tanh(x);
    let row = g.reverse(t, vec![2]);
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let cv = g.param("cache_v", Shape::new(&[1, CAP, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let v = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![cv, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let shape = Shape::new(&[1, POS + 1, W], DType::F32);
    let sk = g.softmax(k, 2, shape.clone());
    let sv = g.softmax(v, 2, shape);
    g.set_outputs(vec![sk, sv]);
    g
}

/// [`kv_row_graph`] with a joined packed multiply in the same run of device work,
/// so one batch carries both kinds of joined call (bracket 4).
fn kv_row_mixed_graph() -> (Graph, Vec<(&'static str, Vec<u8>)>) {
    let mut g = Graph::new("wgpu_buffer_copy_batch_join_mixed");
    let x = g.input("x", Shape::new(&[1, 1, W], DType::F32));
    let row = g.reverse(x, vec![2]);
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let cv = g.param("cache_v", Shape::new(&[1, CAP, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let v = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![cv, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    let kr = g.reverse(k, vec![2]);
    let vr = g.reverse(v, vec![2]);
    // The packed multiply, after the copies in the schedule and in the same run of
    // device work: `m == 1` over a whole superblock, so its route is the one-row
    // device arm and it joins like the copies do.
    let xm = g.input("xm", Shape::new(&[1, KM], DType::F32));
    let bytes = common::packed_gguf_bytes(rlx_gguf::GgmlType::Q4K, KM, NM);
    let w = g.param("w", Shape::new(&[bytes.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![xm, w],
        Shape::new(&[1, NM], DType::F32),
    );
    g.set_outputs(vec![kr, vr, y]);
    (g, vec![("w", bytes)])
}

/// The two caches' contents, non-zero and different from the row written into
/// them, so a write that never happened is a mismatch rather than a coincidence.
fn params() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("cache_k", feed(CAP * W, 0.013)),
        ("cache_v", feed(CAP * W, 0.017)),
    ]
}

/// Compile `g` on `device`, upload the caches and any packed weights, and return
/// the runnable graph.
fn build(device: Device, g: &Graph, packed: &[(&str, Vec<u8>)]) -> rlx_runtime::CompiledGraph {
    let mut c = Session::new(device).compile(g.clone());
    for (name, data) in params() {
        c.set_param(name, &data);
    }
    for (name, bytes) in packed {
        c.set_param_typed(name, bytes, DType::U8);
    }
    c
}

/// The packed multiply's parity, at the bound and with the accompanying check the
/// closest existing parity test applies to these shapes (`gguf_batch_join.rs`:
/// Q4_K, a whole number of superblocks, one row).
fn assert_packed_close(got: &[f32], want: &[f32], what: &str) {
    const TOL: f32 = 1e-2;
    let max_abs = common::max_abs_diff(got, want);
    eprintln!("[copy-join] {what}: max_abs={max_abs:.6e}");
    assert!(max_abs <= TOL, "{what}: max_abs {max_abs} > {TOL}");
    assert!(
        got.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "{what}: the output is all zeros"
    );
}

/// Parity at a tolerance, for a consumer that rounds (`softmax`). A row that
/// never reached the device would read as `1/width` everywhere, orders of
/// magnitude away from a real row, so the tolerance is not carrying the check.
fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: output count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{what}: output {i} length");
        let max_abs = common::max_abs_diff(g, w);
        assert!(
            max_abs <= tol,
            "{what}: output {i} max_abs {max_abs:e} > {tol:e}"
        );
        let spread = g.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - g.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread > 0.0,
            "{what}: output {i} is uniform — the row the consumer read is missing"
        );
    }
}

/// Bit-for-bit equality against the reference, reported as the first mismatch
/// (the outputs are tens of thousands of floats).
fn assert_exact(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: output count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{what}: output {i} length");
        if let Some(at) = g.iter().zip(w).position(|(a, b)| a != b) {
            panic!(
                "{what}: output {i} differs at {at}: got {} want {} (of {} elements); \
                 max_abs {}",
                g[at],
                w[at],
                g.len(),
                common::max_abs_diff(g, w)
            );
        }
    }
}

/// The knobs that would change what a submission or a wait means for a forward
/// pass, or move these graphs' routing (the shard knobs stripe the arena, which
/// hosts every copy) — so an operator's environment cannot flip the pins instead
/// of tripping this guard.
const GUARDED: [&str; 15] = [
    "RLX_WGPU_SNAPSHOT",
    "RLX_BENCH_DISPATCH_ONLY",
    "RLX_WGPU_ALL_PARAMS_WEIGHT",
    "RLX_WGPU_HOST_EAGER_H2D",
    "RLX_DEBUG_NANS",
    "RLX_WGPU_CONCAT_HOST",
    "RLX_WGPU_FORCE_HOST",
    "RLX_WGPU_HOST_MATMUL",
    "RLX_WGPU_HOST_BUFFER_COPY",
    "RLX_WGPU_DISCRETE_HOST",
    "RLX_WGPU_SHARD_CAP_MIB",
    "RLX_WGPU_SHARD_STAGE_MIB",
    "RLX_WGPU_SHARD_GPU",
    "RLX_WGPU_ALLOW_SHARD",
    common::RUN_REPORT_SWITCH,
];

/// One test fn on purpose — see the module doc.
#[test]
fn a_device_route_row_write_rides_the_enclosing_batch_and_the_host_route_still_submits() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    for key in GUARDED {
        assert!(
            !rlx_ir::env::flag(key),
            "{key} changes the submission/wait shape of a forward pass; unset it to run this pin"
        );
    }
    // `RLX_WGPU_ONE_OP_PER_PASS` is the one mode this pin supports: it makes the
    // join inert, which is the reading of the self-submitting shape below.
    let step_by_step = rlx_ir::env::flag("RLX_WGPU_ONE_OP_PER_PASS");

    let g = kv_row_graph();
    let last = kv_row_only_graph();
    // The premise, from the schedules: both graphs must still lower each row
    // write to a `buffer_copy` step, with device steps around them (graph 1) and
    // as their run's last steps (graph 2). Without this the counters below would
    // be measuring a graph that no longer exercises the route.
    let names = WgpuExecutable::compile(g.clone()).test_step_names();
    let last_names = WgpuExecutable::compile(last.clone()).test_step_names();
    eprintln!("[copy-join] steps={names:?} last={last_names:?}");
    assert_eq!(
        names,
        [
            "reverse",
            "buffer_copy",
            "buffer_copy",
            "reverse",
            "reverse"
        ],
        "graph 1 must be two row writes inside device work"
    );
    assert_eq!(
        last_names,
        ["reverse", "buffer_copy", "buffer_copy"],
        "graph 2 must be the same two row writes as the schedule's last steps"
    );

    let x = feed(W, 0.031);
    // The warm-up row must differ from the one the brackets write: the caches are
    // uploaded once and never re-uploaded, so warming up with the same `x` would
    // leave exactly the bytes the bracketed run writes already in the destination
    // row — and a copy that never ran, or ran after the reads of that row, would
    // still match the reference. A different row makes every one of those detect.
    let x_warm = feed(W, 0.047);
    let reference = |graph: &Graph| build(Device::Cpu, graph, &[]).run(&[("x", &x)]);
    let cpu = reference(&g);
    let cpu_last = reference(&last);
    // A graph whose reference is all zeros would let a copy that never ran pass
    // the parity assertions below.
    assert!(
        cpu.iter()
            .chain(&cpu_last)
            .any(|o| o.iter().any(|v| *v != 0.0)),
        "the reference outputs must be non-zero"
    );

    // ── 1. The device route, inside a run of device work ─────────────────
    // A fresh executable and a warm-up run before the bracket: the pool, the
    // uniforms and the kernel cache are not submissions or waits, and with a
    // different row warmed in (see `x_warm`) the bracketed run's copies have to
    // actually move the bytes they claim to.
    let mut exe = build(Device::Gpu, &g, &[]);
    let _ = exe.run(&[("x", &x_warm)]);
    reset_device_counts();
    let out = exe.run(&[("x", &x)]);
    let (s1, w1) = (device_submit_count(), device_wait_count());
    eprintln!("[copy-join] (1) device route, device work after: submits={s1} waits={w1}");
    // The chosen pair, with the unjoined shapes in the message: 4 submissions and
    // 5 waits per forward pass before the join (one own submission and two waits
    // per copy, plus the batch and the readback), 1 and 1 after it.
    assert_eq!(
        (s1, w1),
        if step_by_step { (5, 5) } else { (1, 1) },
        "two device-route row writes inside a run of device work must ride that \
         run's single submission — 4 submissions and 5 waits before the join, 5 \
         and 5 in the step-by-step mode where each device step submits on its \
         own; measured {s1} submission(s) and {w1} wait(s)"
    );
    assert_exact(&out, &cpu, "graph 1, device route");

    // ── 2. The device route with the copies as the schedule's last steps ─
    let mut exe_last = build(Device::Gpu, &last, &[]);
    let _ = exe_last.run(&[("x", &x_warm)]);
    reset_device_counts();
    let out_last = exe_last.run(&[("x", &x)]);
    let (s2, w2) = (device_submit_count(), device_wait_count());
    eprintln!("[copy-join] (2) device route, copies last: submits={s2} waits={w2}");
    assert_eq!(
        (s2, w2),
        if step_by_step { (4, 5) } else { (1, 1) },
        "the copies being the schedule's last steps must change nothing about \
         their route: the carried encoder is the one the readback rides — 4 \
         submissions and 5 waits before the join, 4 and 5 in the step-by-step \
         mode; measured {s2} submission(s) and {w2} wait(s)"
    );
    assert_exact(&out_last, &cpu_last, "graph 2, device route");

    // ── 3. The host route of the same graph, forced ──────────────────────
    // The host route is the one that reads and writes through the host, so it
    // keeps its own submission, its own waits and its own readback: this pair
    // must not move. Scoped to its bracket — the knob is read where the run path
    // routes a copy (its own arm and the batch boundary), but leaving it set
    // would route every later copy.
    rlx_ir::env::set("RLX_WGPU_HOST_BUFFER_COPY", "1");
    let mut exe_host = build(Device::Gpu, &last, &[]);
    let _ = exe_host.run(&[("x", &x_warm)]);
    reset_device_counts();
    let out_host = exe_host.run(&[("x", &x)]);
    let (s3, w3) = (device_submit_count(), device_wait_count());
    rlx_ir::env::unset("RLX_WGPU_HOST_BUFFER_COPY");
    eprintln!("[copy-join] (3) host route: submits={s3} waits={w3}");
    assert_eq!(
        (s3, w3),
        (4, 5),
        "the host route's own readback must still show, exactly as before the \
         join: its D2H read's submission and drain, the boundary polls, the batch \
         and the readback — 4 submissions and 5 waits; measured {s3} \
         submission(s) and {w3} wait(s)"
    );
    assert_exact(&out_host, &cpu_last, "graph 2, host route");

    // ── 3b. The host route with device work after the copies ─────────────
    // Bracket 3's copies are the schedule's last steps, so the run's final flush
    // delivers them; its flips are host steps, so nothing dispatched before the
    // writes. Here a device step precedes them and a device consumer follows
    // them. Before the copy's deferred write reached the door in front of that
    // consumer, and before the mirror survived the host boundary between the two
    // copies, the consumer read zeros for the row the first copy had written.
    // Parity against the CPU reference is what fails then.
    rlx_ir::env::set("RLX_WGPU_HOST_BUFFER_COPY", "1");
    let consumer = kv_row_device_consumer_graph();
    let cpu_consumer = build(Device::Cpu, &consumer, &[]).run(&[("x", &x)]);
    let mut exe_consumer = build(Device::Gpu, &consumer, &[]);
    let _ = exe_consumer.run(&[("x", &x_warm)]);
    reset_device_counts();
    let out_consumer = exe_consumer.run(&[("x", &x)]);
    let (s3b, w3b) = (device_submit_count(), device_wait_count());
    rlx_ir::env::unset("RLX_WGPU_HOST_BUFFER_COPY");
    eprintln!("[copy-join] (3b) host route, device work after: submits={s3b} waits={w3b}");
    assert_close(
        &out_consumer,
        &cpu_consumer,
        1e-5,
        "two row writes between device work, host route",
    );

    // ── 4. Both join kinds in one run of device work ─────────────────────
    // The row writes and a packed multiply whose route is the one-row device arm,
    // in one schedule: both joins share one carried encoder here, so this is the
    // bracket that would catch the two of them disagreeing about it — a copy that
    // dropped the encoder, or a multiply that opened a second one, shows up as an
    // extra submission, and a multiply that fell to a host leg as the route
    // census. The multiply's parity is asserted at the packed-multiply bound and
    // the row writes exactly, each as the closest existing test of its shape does.
    let (mixed, mixed_packed) = kv_row_mixed_graph();
    let xm = feed(KM, 0.053);
    let mixed_names = WgpuExecutable::compile(mixed.clone()).test_step_names();
    eprintln!("[copy-join] mixed={mixed_names:?}");
    assert_eq!(
        mixed_names,
        [
            "reverse",
            "buffer_copy",
            "buffer_copy",
            "reverse",
            "reverse",
            "dequant_matmul_gguf"
        ],
        "graph 4 must be the two row writes with the packed multiply after them"
    );
    let cpu_mixed = {
        let mut c = build(Device::Cpu, &mixed, &mixed_packed);
        c.run(&[("x", &x), ("xm", &xm)])
    };
    let mut exe_mixed = build(Device::Gpu, &mixed, &mixed_packed);
    let _ = exe_mixed.run(&[("x", &x_warm), ("xm", &xm)]);
    reset_route_counts();
    reset_device_counts();
    let out_mixed = exe_mixed.run(&[("x", &x), ("xm", &xm)]);
    let (s4, w4) = (device_submit_count(), device_wait_count());
    let routes = route_counts();
    eprintln!("[copy-join] (4) both join kinds: submits={s4} waits={w4} routes={routes:?}");
    assert_eq!(
        (routes.per_row, routes.host),
        (1, 0),
        "the packed multiply must take the one-row device route, not a host leg: \
         {routes:?}"
    );
    assert_eq!(
        (s4, w4),
        if step_by_step { (7, 7) } else { (1, 1) },
        "a joined copy and a joined packed multiply in one run of device work must \
         ride that run's single submission — 7 and 7 in the step-by-step mode, \
         where the join is inert and every step submits on its own; measured {s4} \
         submission(s) and {w4} wait(s)"
    );
    assert_exact(&out_mixed[..2], &cpu_mixed[..2], "graph 4, row writes");
    assert_packed_close(&out_mixed[2], &cpu_mixed[2], "graph 4, packed multiply");
}
