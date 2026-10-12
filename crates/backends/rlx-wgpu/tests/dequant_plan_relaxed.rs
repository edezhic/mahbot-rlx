// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device-side companion to `rlx-compile/tests/dequant_guard_plan.rs`: the
//! same builder shrunk until both plan configurations build on one device — 2
//! layers, an 8-row window out of a 32-row context by a contiguous `Narrow`,
//! 256-wide hidden, 512-wide MLP, Q4_K/Q6_K weights. That file owns the
//! device-free plan goldens; this one asserts only what the device decides and
//! computes its own comparison plans, so no plan literal lives here. The builder
//! and those constants are duplicated rather than shared — the compile-side file
//! is another crate, so sharing would add a dev-dependency edge — and a drift
//! between them only leaves this paragraph stale.
//!
//! **What it runs.** Every arena route the backend can take — params in the act
//! arena, params parked via `RLX_WGPU_ALL_PARAMS_WEIGHT`, and an
//! `RLX_ARENA_NO_REUSE` superset — each bit-identical to the other two and none on
//! the packed host route; the shipped one is compared with the CPU backend at 1e-2
//! max-abs (measured 1.07e-4). Step timing is measured and printed for both pairs,
//! never asserted: a device timing that cannot separate a regression from machine
//! load is reported, not gated. There is no *guarded* run to time — the guard is
//! derived, not configurable.
//!
//! **Globals.** The device test pins `RLX_WGPU_HAZARD_REPORT`,
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` and the planner knobs in its own body, and the
//! plan site writes the global `PIN_HOST_STRUCTURE`; the other test here is
//! device-free and reads none of them.
//!
//! **Why no run takes the host route.** It needs packed f32 scratch past
//! `max_buffer_size`, so the premise rests on the route classification in
//! `gguf_gpu.rs` plus
//! `backend::step::tests::no_packed_multiply_step_is_deferred_to_the_host_tail`,
//! not on a run.

mod common;

use rlx_compile::memory::{GuardedPlanner, plan_memory_with_dequant_guard};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

const F: DType = DType::F32;

/// The route table's answer, asserted rather than left to the plan sizes: no
/// packed multiply this backend can run defers its host read, so every plan site
/// derives the guard *off*. A device-free tripwire, not a behavioural check.
///
/// It flips the moment a route gains a deferring arm (the classification is
/// exhaustive; `backend::step::tests::no_packed_multiply_step_is_deferred_to_the_host_tail`
/// covers the packed steps that are not route-table members). When it does, the
/// fix is *not* to relax this assertion: the guard re-arms at every plan site, so
/// the plan goldens in `dequant_guard_plan.rs` are what must move.
#[test]
fn no_packed_multiply_route_defers_a_host_read() {
    assert!(
        !rlx_wgpu::gguf_gpu::dequant_route_defers_host_read(),
        "a packed multiply route now defers its host read, so the deferred-host pin is live \
         again and the relaxed plan goldens are stale"
    );
}

/// The `RUNNABLE` geometry of `rlx-compile/tests/dequant_guard_plan.rs` — the
/// demonstration graph shrunk until both plan configurations build on one device.
const LAYERS: usize = 2;
const WINDOW: usize = 8;
const CTX: usize = 32;
const HIDDEN: usize = 256;
const INTER: usize = 512;

/// Timed runs per configuration; one warm-up run precedes them. Enough samples
/// that the minimum of each configuration is a run with little interference on it
/// — the statistic the plan-size comparison rests on.
const REPS: usize = 15;

/// Deterministic values for an `n`-element tensor or parameter.
///
/// The step is deliberately steep. The weights above are a *slow* sinusoid in
/// their row index (`sin(0.011·i)`), so a slow input correlates with every weight
/// row and the packed multiplies sum coherently: at the sibling tests' `0.017`
/// step this graph reaches `|out| ≈ 7e4`, where the 1e-2 absolute tolerance below
/// is under one f32 ulp and the check would measure summation order, not routing.
/// A fast sine decorrelates, every sum grows as `sqrt(k)`, and the tolerance is
/// the routing statement it is meant to be.
fn feed(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * step).sin()).collect()
}

/// Deterministic `[n, k]` weights, packed by `quantize` below — the sibling
/// route test's generator, so the error is a quantizer's and not a synthetic
/// byte pattern's.
fn w_row(k: usize, n: usize) -> Vec<f32> {
    (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect()
}

/// One `DequantMatMul` over a packed weight param of the scheme's own byte width,
/// recording the bytes the test has to upload for it.
fn dmm(
    g: &mut Graph,
    x: NodeId,
    name: &str,
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
    weights: &mut Vec<(String, Vec<u8>)>,
) -> NodeId {
    let bytes = n * (k / scheme.gguf_block_size() as usize) * scheme.gguf_block_bytes() as usize;
    let w = g.param(name, Shape::new(&[bytes], DType::U8));
    weights.push((
        name.to_string(),
        rlx_gguf::quantize(&w_row(k, n), ggml).expect("quantize"),
    ));
    g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[WINDOW, n], F),
    )
}

/// The graph, plus the packed bytes of every weight param and the names of the
/// per-layer RMSNorm gammas.
struct Built {
    graph: Graph,
    weights: Vec<(String, Vec<u8>)>,
    gammas: Vec<String>,
}

/// A decode-shaped transformer step — packed Q4_K/Q6_K projections, RMSNorm,
/// SwiGLU, and an 8-row context window read by a contiguous `Narrow` — at the
/// geometry `rlx-compile/tests/dequant_guard_plan.rs` shrinks its demonstration
/// graph to. That file's `demo_graph` is the same builder without the packed
/// weights; a change to one is a change to both.
fn decode_graph() -> Built {
    let h = HIDDEN;
    let mut g = Graph::new("wgpu_dequant_plan_relaxed");
    let mut weights = Vec::new();
    let mut gammas = Vec::new();
    let k_cache = g.input("k_cache", Shape::new(&[CTX, h], F));
    let v_cache = g.input("v_cache", Shape::new(&[CTX, h], F));
    let x_in = g.input("x", Shape::new(&[WINDOW, h], F));
    let kw = g.narrow_(k_cache, 0, 0, WINDOW);
    let vw = g.narrow_(v_cache, 0, 0, WINDOW);
    let mut x = x_in;
    for l in 0..LAYERS {
        let g1 = g.param(format!("L{l}.g1"), Shape::new(&[h], F));
        let g2 = g.param(format!("L{l}.g2"), Shape::new(&[h], F));
        gammas.push(format!("L{l}.g1"));
        gammas.push(format!("L{l}.g2"));
        let xn = g.rms_norm(x, g1, g1, 1e-5);
        let q = dmm(
            &mut g,
            xn,
            &format!("L{l}.wq"),
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            h,
            h,
            &mut weights,
        );
        let k = dmm(
            &mut g,
            xn,
            &format!("L{l}.wk"),
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            h,
            h,
            &mut weights,
        );
        let v = dmm(
            &mut g,
            xn,
            &format!("L{l}.wv"),
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            h,
            h,
            &mut weights,
        );
        let qk = g.add(q, k);
        let s = g.add(qk, v);
        let o = dmm(
            &mut g,
            s,
            &format!("L{l}.wo"),
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            h,
            h,
            &mut weights,
        );
        let hres = g.add(x, o);
        let xn2 = g.rms_norm(hres, g2, g2, 1e-5);
        let gate = dmm(
            &mut g,
            xn2,
            &format!("L{l}.wg"),
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            h,
            INTER,
            &mut weights,
        );
        let up = dmm(
            &mut g,
            xn2,
            &format!("L{l}.wu"),
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            h,
            INTER,
            &mut weights,
        );
        let act = g.silu(gate);
        let prod = g.mul(act, up);
        let down = dmm(
            &mut g,
            prod,
            &format!("L{l}.wd"),
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            INTER,
            h,
            &mut weights,
        );
        x = g.add(hres, down);
    }
    // Consume the window at the end of the step so the resident caches are live
    // across it, which is what puts them in the live set.
    let xk = g.add(x, kw);
    let out = g.add(xk, vw);
    g.set_outputs(vec![out]);
    Built {
        graph: g,
        weights,
        gammas,
    }
}

/// One timed run, in nanoseconds.
fn nanos(c: &mut CompiledGraph, inputs: &[(&str, &[f32])]) -> u128 {
    // `rlx_ir::Tick`, not `Instant`: the tolerance floor below is `min/200` — tens
    // of microseconds, under `Instant`'s resolution — and the tree mandates the
    // tick for measurements at that scale.
    let start = rlx_runtime::Tick::now();
    let _ = c.run(inputs);
    u128::from(rlx_runtime::Tick::now().elapsed_ns(start))
}

/// `(min, median, max)` of a configuration's timings, in nanoseconds.
fn min_median_max(mut t: Vec<u128>) -> (u128, u128, u128) {
    t.sort_unstable();
    (t[0], t[t.len() / 2], t[t.len() - 1])
}

/// Times two configurations against each other: one warm-up run each, then
/// `REPS` timed runs with the order alternated inside every pair, so neither a
/// session's clock drift nor "always the second of the pair" is what gets
/// measured. Returns `(a_min, a_median, a_max, b_min, b_median, b_max)` in
/// nanoseconds.
fn time_pair(
    a: &mut CompiledGraph,
    b: &mut CompiledGraph,
    runs: &[(&str, &[f32])],
) -> (u128, u128, u128, u128, u128, u128) {
    let _ = a.run(runs);
    let _ = b.run(runs);
    let (mut ta, mut tb) = (Vec::with_capacity(REPS), Vec::with_capacity(REPS));
    for i in 0..REPS {
        if i % 2 == 0 {
            ta.push(nanos(a, runs));
            tb.push(nanos(b, runs));
        } else {
            tb.push(nanos(b, runs));
            ta.push(nanos(a, runs));
        }
    }
    let (a_min, a_med, a_max) = min_median_max(ta);
    let (b_min, b_med, b_max) = min_median_max(tb);
    (a_min, a_med, a_max, b_min, b_med, b_max)
}

/// No packed multiply left the device: nothing on the host route, something on a
/// device route.
fn assert_device_routes(r: &rlx_wgpu::gguf_gpu::DequantMatmulRoutes, what: &str) {
    assert_eq!(
        r.host, 0,
        "{what}: a packed multiply took the host route: {r:?}"
    );
    assert!(
        r.per_row + r.grouped + r.q1_0_gemm + r.scratch > 0,
        "{what}: no packed multiply took a device route: {r:?}"
    );
}

/// One test fn on purpose — see the module doc's **Globals** paragraph for the
/// process-global state it touches.
#[test]
fn the_relaxed_plan_runs_both_arena_routes_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    common::refuse_under_run_report();

    // Pin every knob this test depends on, *before* the first compile: an exported
    // environment a developer has set would otherwise move the plan figures below,
    // and `RLX_WGPU_ALL_PARAMS_WEIGHT` in particular would put the "in-arena"
    // session on the parked route and silently turn the timing comparison into two
    // parked runs. `set`, not `unset`, throughout — unsetting drops only the code
    // override and falls back to the process environment.
    rlx_ir::env::set("RLX_PIN_OUTPUT_ANCESTORS", "0");
    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "0");
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "0");
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    rlx_ir::env::set("RLX_WGPU_HAZARD_REPORT", "0");

    let Built {
        graph: g,
        weights,
        gammas,
    } = decode_graph();
    let inputs: Vec<(&str, Vec<f32>)> = vec![
        ("k_cache", feed(CTX * HIDDEN, 1.3)),
        ("v_cache", feed(CTX * HIDDEN, 1.7)),
        ("x", feed(WINDOW * HIDDEN, 1.9)),
    ];
    let gamma_values: Vec<(String, Vec<f32>)> = gammas
        .iter()
        .map(|name| (name.clone(), feed(HIDDEN, 2.1)))
        .collect();

    // ── Plans ──────────────────────────────────────────────────────────────
    // The four figures are pinned device-free in
    // `rlx-compile/tests/dequant_guard_plan.rs`; printed here for the record.
    let relaxed = rlx_wgpu::buffer::plan_f32_uniform(&g, 16);
    let guarded = plan_memory_with_dequant_guard(&g, 16, GuardedPlanner::F32Uniform, true);
    let guarded_act =
        plan_memory_with_dequant_guard(&g, 16, GuardedPlanner::F32UniformNoParams, true);
    let relaxed_act =
        plan_memory_with_dequant_guard(&g, 16, GuardedPlanner::F32UniformNoParams, false);
    eprintln!(
        "plan bytes: relaxed(full)={} guarded(full)={} guarded(acts)={} relaxed(acts)={}",
        relaxed.arena_size, guarded.arena_size, guarded_act.arena_size, relaxed_act.arena_size
    );

    // The point of the relaxation: the relaxed arena is one allocation.
    let limits = rlx_wgpu::wgpu_device()
        .expect("wgpu device (skip_unless_device said it was available)")
        .device
        .limits();
    eprintln!(
        "relaxed arena {} B vs per-allocation ceiling {} B",
        relaxed.arena_size, limits.max_buffer_size
    );
    assert!(
        relaxed.arena_size < limits.max_buffer_size as usize,
        "the relaxed arena {} must fit one allocation (ceiling {})",
        relaxed.arena_size,
        limits.max_buffer_size
    );

    // ── The shipping route: parameters inside the activation arena ─────────
    rlx_ir::env::set("RLX_WGPU_HAZARD_REPORT", "1");
    let mut gpu = Session::new(Device::Gpu).compile(g.clone());
    for (name, bytes) in &weights {
        gpu.set_param_typed(name, bytes, DType::U8);
    }
    for (name, v) in &gamma_values {
        gpu.set_param(name, v);
    }
    let runs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();

    rlx_wgpu::gguf_gpu::reset_route_counts();
    let before = rlx_wgpu::gguf_gpu::route_counts();
    let relaxed_out = gpu.run(&runs).remove(0);
    let relaxed_routes = rlx_wgpu::gguf_gpu::route_counts();
    eprintln!("routes before the run: {before:?}");
    eprintln!("routes after the relaxed run: {relaxed_routes:?}");
    assert_device_routes(&relaxed_routes, "relaxed run");

    // ── The CPU reference ──────────────────────────────────────────────────
    // The engine's own CPU backend — the same reference the GGUF parity tests use.
    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    for (name, bytes) in &weights {
        cpu.set_param_typed(name, bytes, DType::U8);
    }
    for (name, v) in &gamma_values {
        cpu.set_param(name, v);
    }
    let cpu_out = cpu.run(&runs).remove(0);

    assert_eq!(relaxed_out.len(), WINDOW * HIDDEN);
    assert_eq!(cpu_out.len(), WINDOW * HIDDEN);
    assert!(
        relaxed_out.iter().all(|v| v.is_finite()),
        "non-finite value in the relaxed-plan output"
    );
    assert!(
        cpu_out.iter().all(|v| v.is_finite()),
        "non-finite value in the CPU reference"
    );
    let max_abs = cpu_out
        .iter()
        .zip(&relaxed_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("relaxed plan vs cpu: max_abs={max_abs:.6e} (tolerance 1e-2)");
    assert!(
        max_abs <= 1e-2,
        "relaxed plan vs cpu: max_abs {max_abs} > 1e-2"
    );

    // ── The other arena route: every parameter parked in the weight buffer ──
    // `RLX_WGPU_ALL_PARAMS_WEIGHT` forces `Arena::from_plan_split` — the regime a
    // model larger than the binding limit lands in — so the packed multiply reads
    // its weight from a non-zero base instead of the act arena. Same graph, same
    // bytes, same inputs, so the results must be bit-identical. (The knob is read
    // by the split planner's constructor only, so it goes back off right after.)
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    // The parked route's own plan site: `Arena::from_plan_split` re-plans the
    // activations compactly before parking the parameters, so the plan it lays
    // the arena out under — `size` minus the scratch zone — is the relaxed
    // compact golden. Reverting that call site to
    // `plan_memory_f32_uniform_no_params` (guard inherited) makes it 385_408
    // instead, which is what this pins. The scratch reserve is a placeholder —
    // the assertion is on `size - scratch_bytes`, so its value does not matter —
    // and under `RLX_WGPU_ALL_PARAMS_WEIGHT=1` every param parks, so only
    // activations are left in the arena.
    let dev = rlx_wgpu::wgpu_device().expect("wgpu device (checked above)");
    let split_arena =
        rlx_wgpu::buffer::Arena::from_plan_split(&dev.device, &relaxed, &g, 16 * 1024);
    let split_plan = split_arena.size - split_arena.scratch_bytes;
    eprintln!(
        "parked route compact activation plan: {split_plan} B (relaxed acts {})",
        relaxed_act.arena_size
    );
    assert_eq!(
        split_plan, relaxed_act.arena_size,
        "the parked route's compact activation plan must be the relaxed one"
    );
    let mut parked = Session::new(Device::Gpu).compile(g.clone());
    for (name, bytes) in &weights {
        parked.set_param_typed(name, bytes, DType::U8);
    }
    for (name, v) in &gamma_values {
        parked.set_param(name, v);
    }
    rlx_wgpu::gguf_gpu::reset_route_counts();
    let parked_out = parked.run(&runs).remove(0);
    let parked_routes = rlx_wgpu::gguf_gpu::route_counts();
    eprintln!("routes after the parked-weight run: {parked_routes:?}");
    assert_device_routes(&parked_routes, "parked-weight run");
    assert_eq!(
        parked_out, relaxed_out,
        "the parked-weight arena route must produce bit-identical results"
    );
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    rlx_ir::env::set("RLX_WGPU_HAZARD_REPORT", "0");

    // ── The relaxation's own pair: plan size moves, route does not ─────────
    // Where the relaxation changes a plan's *size* without moving it across the
    // split threshold — the case here, and for every graph whose live set fits one
    // allocation — the step is the same sequence of dispatches on the same route.
    // `RLX_ARENA_NO_REUSE` is the existing knob that builds such a pair: every
    // tensor gets a private slot, so the plan is a strict superset of the relaxed
    // one, while the arena is still one flat buffer with the parameters inside it.
    // Its results and route must match the relaxed configuration's, which is what
    // makes the timing below a size-only comparison.
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
    let private_plan = rlx_wgpu::buffer::plan_f32_uniform(&g, 16).arena_size;
    let mut private = Session::new(Device::Gpu).compile(g.clone());
    for (name, bytes) in &weights {
        private.set_param_typed(name, bytes, DType::U8);
    }
    for (name, v) in &gamma_values {
        private.set_param(name, v);
    }
    rlx_wgpu::gguf_gpu::reset_route_counts();
    let private_out = private.run(&runs).remove(0);
    let private_routes = rlx_wgpu::gguf_gpu::route_counts();
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "0");
    eprintln!("routes after the no-reuse-plan run: {private_routes:?}");
    eprintln!(
        "plan bytes, no-reuse: {private_plan} (relaxed {})",
        relaxed.arena_size
    );
    assert!(
        private_plan > relaxed.arena_size,
        "the no-reuse plan ({private_plan}) must inflate beyond the shipped one ({})",
        relaxed.arena_size
    );
    assert!(
        private_plan < limits.max_buffer_size as usize,
        "the no-reuse plan must still take the same route, or this pair compares routes"
    );
    assert_eq!(
        private_out, relaxed_out,
        "an inflated plan on the same route must produce bit-identical results"
    );
    assert_eq!(
        (
            private_routes.grouped,
            private_routes.per_row,
            private_routes.q1_0_gemm,
            private_routes.scratch,
            private_routes.host,
        ),
        (
            relaxed_routes.grouped,
            relaxed_routes.per_row,
            relaxed_routes.q1_0_gemm,
            relaxed_routes.scratch,
            relaxed_routes.host,
        ),
        "the no-reuse plan took a different route, so its timing is not comparable"
    );

    // ── Timing: plan size alone, route unchanged ───────────────────────────
    // The two configurations' *minimums* are the comparison: the fastest observed
    // run is the one least disturbed by whatever else is on the machine, so it is
    // the stable statistic for "unchanged". The delta is printed against a
    // tolerance derived from the observed repeat spread (floored at 0.5% of the
    // faster minimum so a degenerate spread is still a measurement), never
    // asserted — a device timing that cannot separate a regression from machine
    // load is reported, not gated.
    let (a_min, a_med, a_max, n_min, n_med, n_max) = time_pair(&mut gpu, &mut private, &runs);
    let (a_spread, n_spread) = (a_max - a_min, n_max - n_min);
    let tolerance = a_spread.max(n_spread).max(a_min.min(n_min) / 200);
    let delta = a_min.abs_diff(n_min);
    eprintln!(
        "timing, plan size only: relaxed min={a_min}ns median={a_med}ns max={a_max}ns; \
         no-reuse min={n_min}ns median={n_med}ns max={n_max}ns; spreads={a_spread}/{n_spread}ns \
         tolerance={tolerance}ns delta(min)={delta}ns"
    );

    // ── Timing: the route change, reported rather than gated ───────────────
    // The parked route is what the relaxation *can* select for a graph whose
    // relaxed plan stays under the ceiling but whose guarded plan does not. It is
    // not the same speed — resolving every packed weight against the separate
    // weight buffer costs per-step host work the in-arena route does not pay — so
    // "unchanged within the configuration's own repeat spread" is not a statement
    // this pair can make, and the ratio is only *observed* — it has measured
    // anywhere from 1.0 to 1.9 with machine load — which is why it is printed
    // rather than bounded. Gating it would fail a default-run test on a busy
    // machine.
    let (a_min, a_med, a_max, b_min, b_med, b_max) = time_pair(&mut gpu, &mut parked, &runs);
    let ratio = b_med as f64 / a_med as f64;
    eprintln!(
        "timing, params in the act arena: min={a_min}ns median={a_med}ns max={a_max}ns \
         spread={}ns",
        a_max - a_min
    );
    eprintln!(
        "timing, params parked in the weight buffer: min={b_min}ns median={b_med}ns max={b_max}ns \
         spread={}ns",
        b_max - b_min
    );
    eprintln!("parked / in-arena median step ratio: {ratio:.3}");
}
