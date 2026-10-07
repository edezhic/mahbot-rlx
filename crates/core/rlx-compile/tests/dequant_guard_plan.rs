// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wgpu planner derives the deferred-host `DequantMatMul` guard instead of
//! inheriting it — `rlx_wgpu::gguf_gpu::dequant_route_defers_host_read` — and this
//! is what that buys, with the two other conservative extensions shown to be
//! untouched and no other backend's plan moving.
//!
//! The graph is a decode-shaped transformer — 32 layers, Q4_K/Q6_K weights on
//! every projection, a 4096-row context read through a 512-row contiguous
//! `Narrow`, its window caches graph `Input`s — planned device-free at `rlx-wgpu`'s
//! own width and align. `rlx-wgpu/tests/dequant_plan_relaxed.rs` replays the same
//! builder at a runnable size; keep the two in lockstep.
//!
//! Every test here reads the process-global planner switches, so they share one
//! lock.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use rlx_compile::memory::{
    ArenaWidthPolicy, GuardedPlanner, MemoryPlan, MemoryPlanOptions, plan_memory,
    plan_memory_aligned, plan_memory_f32_uniform, plan_memory_f32_uniform_no_params,
    plan_memory_hybrid, plan_memory_native, plan_memory_with_dequant_guard,
    plan_memory_with_options_policy,
};
use rlx_compile::plan_check::check_plan;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

const F: DType = DType::F32;

/// The width the f32-uniform plan reports activations at.
const WIDTH: usize = 4;
/// `u32::MAX`: the per-allocation ceiling `rlx-wgpu` enforces at its default
/// (`device.limits().max_buffer_size`, clamped by wgpu's indirect-validation
/// flag; `RLX_WGPU_LARGE_BUFFERS=1` removes the clamp).
const CEILING: usize = u32::MAX as usize;

/// The demonstration graph's geometry: a 512-row window read out of a 4096-row
/// resident context, 4096-wide hidden, 11008-wide MLP — a ~6B-parameter model
/// at Q4_K/Q6_K.
const DEMO: Demo = Demo {
    layers: 32,
    window: 512,
    ctx: 4096,
    hidden: 4096,
    inter: 11008,
};

/// The same graph shrunk until both plan configurations build on one device —
/// the configuration the runnable comparison in `rlx-wgpu` uses.
const RUNNABLE: Demo = Demo {
    layers: 2,
    window: 8,
    ctx: 32,
    hidden: 256,
    inter: 512,
};

#[derive(Clone, Copy)]
struct Demo {
    layers: usize,
    window: usize,
    ctx: usize,
    hidden: usize,
    inter: usize,
}

/// Packed byte width of an `[k, n]` GGUF weight.
fn packed_bytes(scheme: QuantScheme, k: usize, n: usize) -> usize {
    let blocks_per_row = k / scheme.gguf_block_size() as usize;
    n * blocks_per_row * scheme.gguf_block_bytes() as usize
}

fn dmm(
    g: &mut Graph,
    x: NodeId,
    scheme: QuantScheme,
    k: usize,
    n: usize,
    m: usize,
    name: &str,
) -> NodeId {
    let w = g.param(name, Shape::new(&[packed_bytes(scheme, k, n)], DType::U8));
    g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[m, n], F),
    )
}

/// A decode-shaped transformer step: packed projections, RMSNorm, SwiGLU, and a
/// 512-row context window read by a contiguous `Narrow`.
fn demo_graph(d: Demo) -> Graph {
    let h = d.hidden;
    let mut g = Graph::new("wgpu_dequant_window_decode");
    let k_cache = g.input("k_cache", Shape::new(&[d.ctx, h], F));
    let v_cache = g.input("v_cache", Shape::new(&[d.ctx, h], F));
    let x_in = g.input("x", Shape::new(&[d.window, h], F));
    let kw = g.narrow_(k_cache, 0, 0, d.window);
    let vw = g.narrow_(v_cache, 0, 0, d.window);
    let mut x = x_in;
    for l in 0..d.layers {
        let g1 = g.param(format!("L{l}.g1"), Shape::new(&[h], F));
        let g2 = g.param(format!("L{l}.g2"), Shape::new(&[h], F));
        let xn = g.rms_norm(x, g1, g1, 1e-5);
        let q = dmm(
            &mut g,
            xn,
            QuantScheme::GgufQ4K,
            h,
            h,
            d.window,
            &format!("L{l}.wq"),
        );
        let k = dmm(
            &mut g,
            xn,
            QuantScheme::GgufQ6K,
            h,
            h,
            d.window,
            &format!("L{l}.wk"),
        );
        let v = dmm(
            &mut g,
            xn,
            QuantScheme::GgufQ4K,
            h,
            h,
            d.window,
            &format!("L{l}.wv"),
        );
        let qk = g.add(q, k);
        let s = g.add(qk, v);
        let o = dmm(
            &mut g,
            s,
            QuantScheme::GgufQ4K,
            h,
            h,
            d.window,
            &format!("L{l}.wo"),
        );
        let hres = g.add(x, o);
        let xn2 = g.rms_norm(hres, g2, g2, 1e-5);
        let gate = dmm(
            &mut g,
            xn2,
            QuantScheme::GgufQ4K,
            h,
            d.inter,
            d.window,
            &format!("L{l}.wg"),
        );
        let up = dmm(
            &mut g,
            xn2,
            QuantScheme::GgufQ6K,
            h,
            d.inter,
            d.window,
            &format!("L{l}.wu"),
        );
        let act = g.silu(gate);
        let prod = g.mul(act, up);
        let down = dmm(
            &mut g,
            prod,
            QuantScheme::GgufQ4K,
            d.inter,
            h,
            d.window,
            &format!("L{l}.wd"),
        );
        x = g.add(hres, down);
    }
    // Consume the window at the end of the step so the resident caches are live
    // across it, which is what makes them part of the live set.
    let xk = g.add(x, kw);
    let out = g.add(xk, vw);
    g.set_outputs(vec![out]);
    g
}

/// The demonstration graph plus a data-dependent index read — extension (a).
fn gather_graph(d: Demo) -> Graph {
    let mut g = demo_graph(d);
    let h = d.hidden;
    let table = g.param("emb", Shape::new(&[64, h], F));
    let ids = g.input("ids", Shape::new(&[d.window], DType::I32));
    let rows = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, ids],
        Shape::new(&[d.window, h], F),
    );
    let out = g.outputs[0];
    let o = g.add(out, rows);
    g.set_outputs(vec![o]);
    g
}

/// A graph whose host-lowered structural ops are the extension — extension (b).
/// No packed multiply, so the guard under test is irrelevant to it.
fn structural_graph() -> Graph {
    let mut g = Graph::new("structural");
    let a = g.input("a", Shape::new(&[64, 512], F));
    let b = g.input("b", Shape::new(&[64, 512], F));
    let c = g.concat_(vec![a, b], 1);
    let t = g.transpose_(c, vec![1, 0]);
    let n = g.narrow_(t, 0, 0, 64);
    let y = g.input("y", Shape::new(&[64, 64], F));
    let mut h = n;
    for _ in 0..12 {
        let p = g.mul(h, y);
        let q = g.add(p, h);
        h = g.mul(q, p);
    }
    g.set_outputs(vec![h]);
    g
}

/// Peak bytes concurrently live, at 4 B/element — the width the f32-uniform plan
/// reports at. Computed from the graph alone; no planner is called.
///
/// Model, stated so it can be checked against the planner's: a tensor is live
/// from the step that produces it to the step of its last reader; a view (a node
/// [`is_pure_view`](rlx_compile::memory::is_pure_view) says aliases its producer)
/// shares that producer's buffer, so a read through a view is a read of the
/// producer; `Input`/`Constant` are resident for the whole execution (the planner
/// keeps them live to the end). Below, this is compared against the
/// activations-only plan, so `Param` is excluded. No alignment padding, no
/// offset search, no pin policy beyond the two rules above.
fn peak_live_bytes(g: &Graph) -> usize {
    let nodes = g.nodes();
    let root = |mut id: NodeId| -> NodeId {
        loop {
            let n = g.node(id);
            if !rlx_compile::memory::is_pure_view(g, n) || n.inputs.is_empty() {
                return id;
            }
            id = n.inputs[0];
        }
    };
    let mut birth: HashMap<NodeId, usize> = HashMap::new();
    let mut death: HashMap<NodeId, usize> = HashMap::new();
    let mut size: HashMap<NodeId, usize> = HashMap::new();
    for (step, n) in nodes.iter().enumerate() {
        let r = root(n.id);
        birth.entry(r).or_insert(step);
        death.entry(r).or_insert(step);
        if r == n.id && !matches!(&n.op, Op::Param { .. }) {
            size.insert(r, n.shape.num_elements().unwrap_or(0) * WIDTH);
        }
    }
    for (step, n) in nodes.iter().enumerate() {
        for &i in &n.inputs {
            if let Some(d) = death.get_mut(&root(i)) {
                *d = (*d).max(step);
            }
        }
    }
    for &o in &g.outputs {
        death.insert(root(o), nodes.len());
    }
    for n in nodes {
        if matches!(&n.op, Op::Input { .. } | Op::Constant { .. }) {
            let r = root(n.id);
            birth.insert(r, 0);
            death.insert(r, nodes.len());
        }
    }
    (0..=nodes.len())
        .map(|upto| {
            birth
                .iter()
                .filter(|(id, b)| **b <= upto && upto <= death[id])
                .map(|(id, _)| size.get(id).copied().unwrap_or(0))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0)
}

/// Offset/size of every slot, so two plans can be compared for exact equality
/// (`MemoryPlan` is only `Clone + Debug`).
fn fingerprint(p: &MemoryPlan) -> Vec<(NodeId, usize, usize)> {
    let mut v: Vec<(NodeId, usize, usize)> = p
        .assignments
        .iter()
        .map(|(id, s)| (*id, s.offset, s.size))
        .collect();
    v.sort();
    v
}

/// The planner entry point `rlx-wgpu` plans with, at its own align: `with_params`
/// picks between its two sites — `buffer::plan_f32_uniform` (params in the arena)
/// and the `Arena::from_plan_split` re-plan (params parked, activations only).
fn wgpu_plan(g: &Graph, guard: bool, with_params: bool) -> MemoryPlan {
    let planner = if with_params {
        GuardedPlanner::F32Uniform
    } else {
        GuardedPlanner::F32UniformNoParams
    };
    plan_memory_with_dequant_guard(g, 16, planner, guard)
}

/// Every test here reads or writes process-global planner switches (`rlx_ir::env`
/// overrides are process-wide, and `PIN_HOST_STRUCTURE` outlives a test), so they
/// are serialized through one lock and the three knobs an exported environment
/// could otherwise move every golden with are pinned to a known value.
fn plan_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let g = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // `RLX_PIN_OUTPUT_ANCESTORS` only ever forces the pin on; a developer with it
    // exported would otherwise move every golden below. `"0"` is not truthy and
    // is not `"1"`, so the derivation stays graph-only. The other two knobs are
    // pinned the same way rather than unset, which would fall back to the
    // process environment a developer could have set.
    rlx_ir::env::set("RLX_PIN_OUTPUT_ANCESTORS", "0");
    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "0");
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "0");
    g
}

/// With the guard armed, the additive entry point reproduces **each** named
/// planner it stands in for, slot for slot, whether `RLX_PIN_OUTPUT_ANCESTORS` is
/// off or on. Blanket-honouring the override at a site whose named planner
/// ignores it (`native` / `hybrid` / the no-params planner) fails this.
///
/// Run on a graph with host indexing as well as without: three of the four
/// variants derive `graph_has_host_indexing(graph)` for their pin, and on a
/// gather-free graph that expectation is `false` for all of them, so collapsing
/// the arm to a constant would otherwise stay green.
#[test]
fn the_derived_entry_point_mirrors_every_planner_it_replaces() {
    let _lock = plan_lock();
    for g in [demo_graph(DEMO), gather_graph(DEMO)] {
        for override_on in ["0", "1"] {
            rlx_ir::env::set("RLX_PIN_OUTPUT_ANCESTORS", override_on);
            let cases = [
                (GuardedPlanner::F32Uniform, plan_memory_f32_uniform(&g, 16)),
                (GuardedPlanner::Native, plan_memory_native(&g, 16)),
                (GuardedPlanner::Hybrid, plan_memory_hybrid(&g, 16)),
                (
                    GuardedPlanner::F32UniformNoParams,
                    plan_memory_f32_uniform_no_params(&g, 16),
                ),
            ];
            for (planner, named) in cases {
                let derived = plan_memory_with_dequant_guard(&g, 16, planner, true);
                let what = format!("override={override_on} planner={planner:?}");
                assert_eq!(
                    derived.arena_size, named.arena_size,
                    "{what}: arena differs"
                );
                assert_eq!(
                    fingerprint(&derived),
                    fingerprint(&named),
                    "{what}: slots differ"
                );
            }
        }
    }
    rlx_ir::env::set("RLX_PIN_OUTPUT_ANCESTORS", "0");
}

/// The before/after figures for the demonstration graph, and the ceiling claim.
///
/// Three figures per configuration, named separately because there is no
/// single "plan size" in this tree:
/// activation arena (params unassigned), the largest single allocated buffer
/// (for a single-buffer arena the whole activation arena, since `rlx-wgpu`
/// refuses to stripe), and the total including the separately parked parameter
/// buffer.
#[test]
fn the_relaxed_activation_arena_fits_one_allocation_and_the_inherited_one_does_not() {
    let _lock = plan_lock();
    let g = demo_graph(DEMO);

    // Activation arena, both configurations.
    let before = wgpu_plan(&g, true, false).arena_size;
    let after = wgpu_plan(&g, false, false).arena_size;
    assert_eq!(before, 5_981_077_888, "before: activation arena");
    assert_eq!(after, 218_628_480, "after: activation arena");

    // The largest single allocation is the activation arena itself: it is under
    // the ceiling after, and over it before.
    assert!(
        after < CEILING,
        "after: {after} must fit one allocation (ceiling {CEILING})"
    );
    assert!(
        before > CEILING,
        "before: {before} must exceed the per-allocation ceiling {CEILING}"
    );

    // Total including the separately parked parameter buffer (packed weights at
    // their native byte width, which `from_plan_split` parks). The four figures:
    let before_full = wgpu_plan(&g, true, true).arena_size;
    let after_full = wgpu_plan(&g, false, true).arena_size;
    assert_eq!(before_full, 10_135_310_720);
    assert_eq!(after_full, 4_372_861_312);
    // Both *full* plans are over the ceiling too, so the split predicate
    // (`plan.arena_size + scratch_bytes > max_storage_buffer_binding_size`,
    // `lower.rs`) is already true on either side: the "before" configuration
    // cannot be built as one buffer, and the relaxation moves the *allocation* it
    // splits into, not the route.
    assert!(before_full > CEILING && after_full > CEILING);
    // The parked-parameter contribution is identical in both configurations, so
    // the whole before/after delta is the activation arena.
    assert_eq!(before_full - before, after_full - after);
    assert_eq!(before_full - before, 4_154_232_832);

    // ... and the plan's own assignments back that figure, not only the identity
    // above: the parked bytes are the params-allocated plan's slots for its
    // `Op::Param` nodes, at each one's byte width (packed weights stay packed,
    // the f32 gammas at 4 B/elem).
    let with_params = wgpu_plan(&g, true, true);
    let param_ids: Vec<NodeId> = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Param { .. }))
        .map(|n| n.id)
        .collect();
    let slots: usize = param_ids
        .iter()
        .filter_map(|id| with_params.assignments.get(id))
        .map(|s| s.size)
        .sum();
    // Each slot additionally carries the planner's `BOUNDARY_TAIL_GUARD_BYTES`
    // (128 B) perimeter reserve, which is not part of `BufferSlot::size`.
    assert_eq!(slots, 4_154_195_968, "the params' own bytes");
    assert_eq!(slots + param_ids.len() * 128, before_full - before);

    // Independent reference: the live set at the same width fits the ceiling too,
    // so the "before" figure is inflation, not a genuinely large graph.
    let live = peak_live_bytes(&g);
    assert_eq!(live, 218_628_096, "independent peak-live reference");
    assert!(live < CEILING);
    // The relaxed plan is the live set plus per-slot alignment padding, and
    // nothing else.
    assert!(
        after >= live && after - live < 4096,
        "after {after} should be the live set {live} plus alignment only"
    );
    assert!(
        before > 27 * live,
        "before {before} is not the live set {live} blown up by the pin"
    );
}

/// The two other extensions keep their own conditions, and the guard change
/// cannot move either.
#[test]
fn host_indexing_still_pins_and_dominates_the_guard() {
    let _lock = plan_lock();
    let g = gather_graph(DEMO);
    // Extension (a) armed: the graph's own host indexing, not a switch.
    let before = wgpu_plan(&g, true, false).arena_size;
    let after = wgpu_plan(&g, false, false).arena_size;
    assert_eq!(before, 6_014_634_496);
    assert_eq!(
        before, after,
        "removing the dequant guard changed a host-indexing graph's plan"
    );
    assert_eq!(wgpu_plan(&g, true, true).arena_size, 10_169_916_032);
    assert_eq!(wgpu_plan(&g, false, true).arena_size, 10_169_916_032);

    // And on this graph the derived pin really is `true`: an explicitly unpinned
    // activations-only plan is smaller, which is what the pin buys back on a
    // host-indexing graph.
    let mut opts = MemoryPlanOptions::inference();
    opts.pin_output_ancestors = false;
    opts.allocate_params = false;
    let unpinned =
        plan_memory_with_options_policy(&g, 16, opts, ArenaWidthPolicy::F32Uniform).arena_size;
    assert!(
        after > unpinned,
        "host indexing must still pin: {after} vs explicitly unpinned {unpinned}"
    );
}

/// The demonstration graph arms neither extension, shown through the planner's
/// own decisions rather than by reading the op list.
#[test]
fn the_demonstration_graph_arms_neither_other_extension() {
    let _lock = plan_lock();
    let g = demo_graph(DEMO);
    let derived = wgpu_plan(&g, false, true).arena_size;

    // (a) The derived output-ancestor pin is `false`: an explicitly unpinned plan
    // is byte-identical to it.
    let mut opts = MemoryPlanOptions::inference();
    opts.pin_output_ancestors = false;
    opts.dequant_host_fallback = false;
    let unpinned = plan_memory_with_options_policy(&g, 16, opts, ArenaWidthPolicy::F32Uniform);
    assert_eq!(
        fingerprint(&unpinned),
        fingerprint(&wgpu_plan(&g, false, true))
    );

    // (b) Arming the host-structure pin changes nothing for this graph — the
    // window's `Narrow` reads a cache that is resident across the whole step
    // anyway — so it cannot account for any of the "before" figure.
    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "1");
    let armed = wgpu_plan(&g, false, true).arena_size;
    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "0");
    assert_eq!(
        armed, derived,
        "host-structure pin moved the demonstration plan"
    );
}

/// Extension (b) is still conditional and still derived from its own condition:
/// on a graph whose structural ops would otherwise be reused, arming it grows
/// the plan, and the guard change does not move it either way.
#[test]
fn host_structural_arming_still_grows_the_plan_and_ignores_the_guard() {
    let _lock = plan_lock();
    let g = structural_graph();
    let off = wgpu_plan(&g, false, true).arena_size;
    assert_eq!(off, 803_200);

    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "1");
    let on = wgpu_plan(&g, false, true).arena_size;
    let on_guarded = wgpu_plan(&g, true, true).arena_size;
    rlx_ir::env::set("RLX_ARENA_PIN_HOST_STRUCTURE", "0");

    assert_eq!(
        on, 852_352,
        "arming the host-structure pin must grow the plan"
    );
    assert_eq!(on, on_guarded, "the guard moved a host-structure plan");
    assert_eq!(
        fingerprint(&wgpu_plan(&g, true, true)),
        fingerprint(&plan_memory_f32_uniform(&g, 16)),
        "a graph with no packed multiply must plan exactly as the shared planner does"
    );
}

/// No other backend's plan moves: the change is additive, and the entry points
/// every other backend reaches are the ones pinned here.
///
/// Method: the repository carries no plan-size baseline, so this pins the shared
/// planner's output for the demonstration graph instead. `rlx-vulkan` builds its
/// arena through private wrappers around exactly these functions (plus the
/// `_no_params` one in its `from_plan_split`), so a golden on them is a golden on
/// Vulkan's plan for the same graph; `fit_batch` and the runtime's budget axis
/// reach the same functions with default options. No shared planner body changed,
/// so a move here can only come from something that was not additive.
#[test]
fn the_shared_planners_do_not_move_for_this_graph() {
    let _lock = plan_lock();
    let g = demo_graph(DEMO);
    let cases = [
        (
            "plan_memory_f32_uniform(g, 16)",
            plan_memory_f32_uniform(&g, 16).arena_size,
            10_135_310_720,
        ),
        (
            "plan_memory_f32_uniform_no_params(g, 16)",
            plan_memory_f32_uniform_no_params(&g, 16).arena_size,
            5_981_077_888,
        ),
        (
            "plan_memory_native(g, 16)",
            plan_memory_native(&g, 16).arena_size,
            10_135_310_720,
        ),
        (
            "plan_memory_hybrid(g, 16)",
            plan_memory_hybrid(&g, 16).arena_size,
            10_135_310_720,
        ),
        (
            "plan_memory_aligned(g, 64)",
            plan_memory_aligned(&g, 64).arena_size,
            10_152_087_936,
        ),
        (
            "plan_memory_aligned(g, 256)",
            plan_memory_aligned(&g, 256).arena_size,
            10_152_125_184,
        ),
        ("plan_memory(g)", plan_memory(&g).arena_size, 10_152_087_936),
    ];
    for (label, actual, expected) in cases {
        assert_eq!(actual, expected, "{label} moved");
    }
}

/// The relaxed plan passes the repository's own program-safety gate — with the
/// guard armed and with it derived off, since the gate is what says the smaller
/// plan's schedule is still legal.
#[test]
fn the_relaxed_plan_passes_the_program_safety_gate() {
    let _lock = plan_lock();
    let g = demo_graph(DEMO);
    for guard in [true, false] {
        let report = check_plan(&g, &wgpu_plan(&g, guard, true));
        assert!(report.is_clean(), "guard={guard}: {}", report.render());
    }
}

/// The figures at the geometry `rlx-wgpu` can actually build (`RUNNABLE`), which
/// is what the device-side test replays: both configurations' arena sizes, and
/// the independent max-concurrent-live-bytes reference the relaxed plan sits on.
#[test]
fn the_runnable_geometry_goldens() {
    let _lock = plan_lock();
    let small = demo_graph(RUNNABLE);
    assert_eq!(
        wgpu_plan(&small, true, false).arena_size,
        385_408,
        "guarded, acts only"
    );
    assert_eq!(
        wgpu_plan(&small, false, false).arena_size,
        131_456,
        "relaxed, acts only"
    );
    assert_eq!(
        wgpu_plan(&small, true, true).arena_size,
        1_230_464,
        "guarded, params in"
    );
    assert_eq!(
        wgpu_plan(&small, false, true).arena_size,
        976_512,
        "relaxed, params in"
    );
    assert_eq!(peak_live_bytes(&small), 131_072);
}
