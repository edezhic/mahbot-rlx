// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The wgpu lowering peels the broadcast copy `LegalizeBroadcast` materialises in
//! front of an `Op::Binary` operand
//! (`rlx_wgpu::unfuse::elide_binary_broadcast_expands`), putting the operand back
//! to the `(i/rep)%len` form `binary_main.wgsl` reads. This is that change's
//! evidence at decoder-plausible elementwise-chain shapes — a channel-major
//! `[1,C,T]` activation combined with per-channel `[1,C,1]` operands, 8 legs:
//!
//! * **A — attribution, device-free.** The runtime path's legalised graph against
//!   the peeled one: `profile_graph_io`, node counts and byte totals per `OpKind`
//!   (in `rlx_compile::fusion_benefit`'s own read+write recipe), and the copy count
//!   and bytes the peel removes. Those bytes are a **lower bound** on what a run
//!   moves: `profile_graph_io` charges one read and one write per node and knows
//!   nothing about a kernel's actual access pattern.
//! * **B — the scheduled steps.** The peeled graph's step table and arena against
//!   the same graph with the removed copies pinned as graph outputs (the peel
//!   refuses an expand that is an output), so the two schedules differ by exactly
//!   the copies. Alongside them, the arena the planner itself produces before and
//!   after the peel — device-free `plan_memory_f32_uniform` on the state the
//!   lowering plans (`unfuse` → `collapse_reshapes`), which is what the copies
//!   actually cost once dead slots are reused, where the pinned variant is only an
//!   upper bound (its copies stay live to the end).
//! * **C — the addressing cost.** A chain of `ARM_LEGS` (32) identical `Add` legs over
//!   the same `[1,C,T]` activation, every leg reading the same operand, so an arm's
//!   traffic scales with the leg count while the single output readback does not. Four
//!   arms of *the same output shape, different operand addressing* — a dense same-shape
//!   operand (no division or remainder), a per-channel `[1,C,1]` one (`(i/T)%C`: a real
//!   quotient and a real remainder), a scalar `[1]` (`(i/1)%1`, the run `simple_broadcast`
//!   gives an all-ones source: one division + one remainder per element), and a trailing
//!   `[T]` one (`(i/1)%T`: the same shape of
//!   work with the quotient left at 1), so `channel/trailing` moves only the quotient
//!   and isolates the division. The change is
//!   *not* made: this only reports what the div+mod costs, as an upper bound on it.
//!
//! A wall-clock reading here includes submission, the device wait and the output
//! readback, a floor that dominates a single `run()` — which is why part C times a chain
//! rather than one leg: the kernel's own traffic has to be lifted above that floor
//! before a few percent of per-element addressing can be seen at all. The variants are
//! therefore timed round-robin after a warm-up (see `gguf_gemm_grouped_weight_reuse.rs`
//! for the ordering-bias rationale) and only the min and median are reported. Nothing
//! asserts a duration; the asserted facts are structural and machine-independent —
//! the peeled schedule has no expand step, the pinned one has one per removed copy,
//! and the per-`OpKind` bytes add up to `profile_graph_io`'s traffic. The peel's own
//! guard is pinned by a non-ignored device-free gate in `complex_parity.rs`.
//!
//! Run the thermal gate first (`just throttle`).

mod common;

use std::collections::BTreeMap;

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::Tick;
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Decoder-plausible channel-major shapes — the same element count, a different
/// channel count, so the per-channel operand's own bytes differ between the sets.
const SHAPES: [(usize, usize); 2] = [(512, 2000), (256, 8000)];
/// Elementwise legs per chain; each one is a single `Op::Binary`.
const LEGS: usize = 8;
/// Identical `Add` legs per part-C arm, all reading the one operand. One leg at ~1M
/// elements is pure `run()` overhead (submission + device wait + readback); `ARM_LEGS`
/// legs multiply the kernel's traffic while that overhead stays put, which is what puts
/// the addressing difference above it.
const ARM_LEGS: usize = 32;
/// Round-robin rounds per part-C arm — odd, so the median is one sample.
const ROUNDS: usize = 7;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The alignment `lower.rs` hands the planner when it compiles this graph
/// (`plan_f32_uniform(&graph, 16)`), so the device-free planning in part B answers
/// with the same planner the lowering uses.
const ALIGN: usize = 16;

fn tensor_bytes(shape: &Shape) -> u64 {
    shape.size_bytes().map_or(0, |b| b as u64)
}

/// Every `Op::Expand` the peeling pass removes, as `(id, read, write)` — the
/// selection is the production one, `rlx_wgpu::unfuse::binary_broadcast_expand_elisions`
/// (the `(expand, source)` pairs the peel itself drops, so this evidence cannot
/// drift from it); `read`/`write` are the removed copy's source and output bytes.
fn removable_expands(g: &Graph) -> Vec<(NodeId, u64, u64)> {
    rlx_wgpu::unfuse::binary_broadcast_expand_elisions(g)
        .into_iter()
        .map(|(id, src)| {
            (
                id,
                tensor_bytes(&g.node(src).shape),
                tensor_bytes(&g.node(id).shape),
            )
        })
        .collect()
}

/// Per `OpKind`: `(node count, read bytes, write bytes)`, read being the node's
/// input bytes and write its own output bytes — the production
/// `rlx_compile::fusion_benefit::node_io_bytes` recipe, split at its output so a
/// kind's two sides are visible. `is_metadata_op` decides what that recipe charges
/// nothing, so nothing is charged here either.
fn kind_bytes(g: &Graph) -> BTreeMap<String, (usize, u64, u64)> {
    let mut m: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for n in g.nodes() {
        let e = m.entry(format!("{:?}", n.op.kind())).or_default();
        e.0 += 1;
        if rlx_compile::fusion_benefit::is_metadata_op(&n.op) {
            continue;
        }
        let write = tensor_bytes(&n.shape);
        e.1 += rlx_compile::fusion_benefit::node_io_bytes(n, g) - write;
        e.2 += write;
    }
    m
}

/// Exactly what the runtime shim does ahead of fusion
/// (`rlx-runtime/src/backend/wgpu_backend.rs`).
fn legalise(g: &Graph) -> Graph {
    use rlx_opt::pass::Pass as _;
    rlx_opt::LegalizeBroadcast.run(g.clone())
}

/// `y ← y (+|*) b_i` over `LEGS` legs of `[1,C,T]`, per-channel `[1,C,1]` params.
/// Add and Mul alternate so both elementwise shapes of the family are exercised.
fn chain(c: usize, t: usize) -> Graph {
    let mut g = Graph::new("elementwise_broadcast_chain");
    let x = g.input("x", Shape::new(&[1, c, t], F));
    let mut y = x;
    for i in 0..LEGS {
        let b = g.param(format!("b{i}"), Shape::new(&[1, c, 1], F));
        y = if i.is_multiple_of(2) {
            g.add(y, b)
        } else {
            g.mul(y, b)
        };
    }
    g.set_outputs(vec![y]);
    g
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.2
        })
        .collect()
}

/// One timed `run`, in nanoseconds. `Tick`, not `Instant`: the tree mandates the
/// tick at this scale.
fn nanos(exe: &mut WgpuExecutable, x: &[f32]) -> u64 {
    let start = Tick::now();
    let _ = exe.run(&[("x", x)]);
    Tick::now().elapsed_ns(start)
}

/// Part A: the legalised graph against the peeled one, all device-free.
fn part_a(label: &str, legalised: &Graph, peeled: &Graph) {
    let removed = removable_expands(legalised);
    let (read, write) = removed
        .iter()
        .fold((0, 0), |(r, w), &(_, r1, w1)| (r + r1, w + w1));
    eprintln!("[part A] {label}");
    eprintln!(
        "  (per-kind read/write is `fusion_benefit::node_io_bytes`' own recipe — a node's inputs' \
         bytes plus its output's — cut at its output: write = the node's own shape, read = \
         node_io_bytes − write; `is_metadata_op`'s kinds (Input/Param/Constant/Reshape/Transpose/\
         Narrow) are charged 0, their bytes are their reader's read)"
    );
    let mut prof = Vec::new();
    for (name, g) in [
        ("legalised (runtime path, pre-lowering)", legalised),
        ("peeled    (backend lowering, post-peel)", peeled),
    ] {
        let p = rlx_compile::fusion_benefit::profile_graph_io(g);
        prof.push(p);
        eprintln!(
            "  {name}: launches={} sync_points={} host_output={} B ({:.2} MiB) traffic={} B ({:.2} MiB)",
            p.kernel_launches,
            p.sync_points,
            p.host_output_bytes,
            mib(p.host_output_bytes),
            p.device_traffic_bytes,
            mib(p.device_traffic_bytes)
        );
        let (mut r, mut w) = (0, 0);
        for (kind, (nodes, kr, kw)) in kind_bytes(g) {
            r += kr;
            w += kw;
            eprintln!("    {kind:<16} nodes={nodes:>2} read={kr:>10} write={kw:>10}");
        }
        // The two halves are `fusion_benefit`'s own per-node recipe (the read is
        // `node_io_bytes` minus the node's output, which is the write), so they add
        // up to its traffic by construction.
        assert_eq!(
            r + w,
            p.device_traffic_bytes,
            "per-kind bytes vs profile_graph_io"
        );
    }
    let (d_launch, d_traffic) = (
        prof[1].kernel_launches as i64 - prof[0].kernel_launches as i64,
        prof[1].device_traffic_bytes as i64 - prof[0].device_traffic_bytes as i64,
    );
    eprintln!(
        "  peel delta: {d_launch:+} launches, {d_traffic:+} B ({:+.2} MiB) — the removed copies' own \
         bytes plus the consuming Binary's read dropping from a full-size copy to the per-channel \
         operand",
        d_traffic as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  removed copies: {} expand nodes, read {} B + write {} B = {} B ({:.2} MiB written) — this \
         family's share, not the model's total: only the Binary legs and the broadcast copies feeding \
         a Binary operand can change (a real decoder's unary legs, norms and matmuls are untouched)",
        removed.len(),
        read,
        write,
        read + write,
        mib(write)
    );
}

/// Part B: the scheduled steps of the peeled graph and of the copies-pinned one.
fn part_b(label: &str, legalised: &Graph, peeled: Graph) {
    let removed = removable_expands(legalised);
    let mut pinned = legalised.clone();
    pinned.set_outputs(
        std::iter::once(legalised.outputs[0])
            .chain(removed.iter().map(|&(id, _, _)| id))
            .collect(),
    );
    // The arena the planner itself produces, device-free: the lowering plans the
    // `unfuse` → `collapse_reshapes` state (lower.rs), so re-run that chain and ask
    // the planner it calls (`plan_memory_f32_uniform` at the same alignment ALIGN).
    // This is the number the copies actually cost after dead slots are reused; the
    // pinned arena below only bounds it, because its copies stay live to the end.
    let lowered =
        |g: &Graph| rlx_wgpu::unfuse::collapse_reshapes(rlx_wgpu::unfuse::unfuse(g.clone()));
    let plan_before =
        rlx_compile::memory::plan_memory_f32_uniform(&lowered(legalised), ALIGN).arena_size;
    let plan_after =
        rlx_compile::memory::plan_memory_f32_uniform(&lowered(&peeled), ALIGN).arena_size;
    let peel_exe = WgpuExecutable::compile(peeled);
    let pin_exe = WgpuExecutable::compile(pinned);
    let peel_names = peel_exe.test_step_names();
    let pin_names = pin_exe.test_step_names();
    eprintln!("[part B] {label}");
    let plan_delta = plan_after as i64 - plan_before as i64;
    let plan_delta_mib = plan_delta as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  planner (`plan_memory_f32_uniform` at align {ALIGN}, on `collapse_reshapes(unfuse(·))` — \
         the planner's own number for the state the lowering plans): before peel {plan_before} B \
         ({:.2} MiB), after peel {plan_after} B ({:.2} MiB), delta {plan_delta:+} B ({plan_delta_mib:+.2} MiB) — \
         the arena the planner produced, with the removed copies' slots recycled or not; the pinned \
         number below is an upper bound on it, because the pinned copies stay live to the end",
        mib(plan_before as u64),
        mib(plan_after as u64)
    );
    eprintln!(
        "  peel_exe.test_arena_bytes() = {} B ({:.2} MiB) against the planner's peeled {plan_after} B \
         ({:+} B): the executable's arena is the plan plus the lowering's tail scratch \
         (`compute_scratch_bytes` floors it at 64 MiB), so a fixed offset — the delta is what \
         compares",
        peel_exe.test_arena_bytes(),
        mib(peel_exe.test_arena_bytes()),
        peel_exe.test_arena_bytes() as i64 - plan_after as i64
    );
    eprintln!(
        "  peeled: {} steps {:?}, arena {} B ({:.2} MiB)",
        peel_names.len(),
        peel_exe.step_kind_counts(),
        peel_exe.test_arena_bytes(),
        mib(peel_exe.test_arena_bytes())
    );
    eprintln!(
        "  pinned: {} steps {:?}, arena {} B ({:.2} MiB)",
        pin_names.len(),
        pin_exe.step_kind_counts(),
        pin_exe.test_arena_bytes(),
        mib(pin_exe.test_arena_bytes())
    );
    eprintln!(
        "  sanity: peeled expand steps={} pinned expand steps={} (one per kept copy)",
        common::expand_steps(&peel_names),
        common::expand_steps(&pin_names)
    );
    assert_eq!(
        common::expand_steps(&peel_names),
        0,
        "peeled schedule must carry no expand step"
    );
    assert_eq!(
        common::expand_steps(&pin_names),
        removed.len(),
        "pinned keeps one expand step per copy"
    );
    let d_steps = pin_names.len() as i64 - peel_names.len() as i64;
    let d_arena = pin_exe.test_arena_bytes() as i64 - peel_exe.test_arena_bytes() as i64;
    eprintln!(
        "  delta: steps {d_steps:+} (exact — each kept copy is exactly one step); arena {d_arena:+} B \
         ({:+.2} MiB, an upper bound: the pinned copies stay live to the end of the graph)",
        mib(d_arena.unsigned_abs())
    );
}

/// One part-C arm: its executable, the steps it actually dispatches, the counted
/// bytes for its whole `[1,C,T]` chain, and one wall-clock sample per round.
struct Arm {
    label: &'static str,
    exe: WgpuExecutable,
    steps: Vec<&'static str>,
    bytes: u64,
    samples: Vec<u64>,
}

impl Arm {
    /// [`ARM_LEGS`] identical `Add` legs of `[1,C,T]`, every leg reading the same
    /// operand of shape `operand`, compiled the way the runtime path leaves the graph
    /// (the shim's `LegalizeBroadcast`); `WgpuExecutable::compile` runs the peel
    /// itself (`elide_binary_broadcast_expands` is in its lowering chain), which turns
    /// the copy the pass materialises for a mid-broadcast operand back into the
    /// kernel's own `(i/rep)%len` read, so the arm times the device `binary` step
    /// rather than the copy. Counted bytes are, per leg, the operand read plus the
    /// output read and write; the arm's total is that times `ARM_LEGS`.
    fn new(label: &'static str, operand: &[usize], c: usize, t: usize, x: &[f32]) -> Self {
        let act = Shape::new(&[1, c, t], F);
        let mut g = Graph::new("binary_add_chain");
        let xi = g.input("x", act.clone());
        let b = g.param("b", Shape::new(operand, F));
        let mut y = xi;
        for _ in 0..ARM_LEGS {
            y = g.add(y, b);
        }
        g.set_outputs(vec![y]);
        let mut exe = WgpuExecutable::compile(legalise(&g));
        exe.set_param("b", &fill(operand.iter().product(), 7));
        let bytes =
            ARM_LEGS as u64 * (tensor_bytes(&Shape::new(operand, F)) + 2 * tensor_bytes(&act));
        let steps = exe.test_step_names();
        let _ = exe.run(&[("x", x)]); // warm-up: first touch of the arena and pipelines
        Self {
            label,
            exe,
            steps,
            bytes,
            samples: Vec::new(),
        }
    }

    fn min(&self) -> f64 {
        *self.samples.iter().min().unwrap() as f64
    }

    /// Lower median; `ROUNDS` is odd so this is one of the samples.
    fn median(&self) -> f64 {
        let mut s = self.samples.clone();
        s.sort_unstable();
        s[s.len() / 2] as f64
    }

    /// Whatever the step table says, so a host-routed arm is never reported as the
    /// device kernel. The chain makes this "every step is `binary`", not "one step".
    fn route_note(&self) -> &'static str {
        if !self.steps.is_empty() && self.steps.iter().all(|s| *s == "binary") {
            ""
        } else if self.steps.contains(&"host_op") {
            "  ← HOST-ROUTED on this machine: not the device kernel"
        } else {
            "  ← not every leg is the plain device binary step"
        }
    }

    /// The dispatched step names, collapsed when the whole chain is the same step —
    /// which is what a 32-leg arm would otherwise print 32 times.
    fn steps_repr(&self) -> String {
        match self.steps.split_first() {
            Some((first, rest)) if rest.iter().all(|s| s == first) => {
                format!("[{first:?}; {} legs]", self.steps.len())
            }
            _ => format!("{:?}", self.steps),
        }
    }
}

/// Part C: the div+mod `(i/rep)%len` per element, as an upper bound on it, at a leg
/// count that puts the kernel above the per-`run()` floor.
fn part_c(label: &str, c: usize, t: usize) {
    let elems = c * t;
    let x = fill(elems, 21);
    let mut arms: Vec<Arm> = [
        ("dense    [1,C,T]", vec![1, c, t]),
        ("channel  [1,C,1]", vec![1, c, 1]),
        ("scalar   [1]", vec![1]),
        ("trailing [T]", vec![t]),
    ]
    .iter()
    .map(|(l, o)| Arm::new(l, o, c, t, &x))
    .collect();
    for _ in 0..ROUNDS {
        for a in arms.iter_mut() {
            a.samples.push(nanos(&mut a.exe, &x));
        }
    }
    eprintln!(
        "[part C] {label}: chain of {ARM_LEGS} `Add` legs over [1,{c},{t}] ({elems} elements each), \
         {ROUNDS} round-robin rounds + warm-up each; every number below is for the whole chain, so the \
         per-`run()` overhead (submission + device wait + one readback) is amortised over {ARM_LEGS} \
         legs of kernel traffic. `trailing [T]` is an addressing-isolation arm, not a decoder operand \
         shape — it reads the same remainder as `channel` with the quotient left at 1."
    );
    for a in &arms {
        let min = a.min();
        eprintln!(
            "  {:<16} steps={} min={:.0} ns median={:.0} ns {:.3} ns/elem·leg {:.1} GB/s @min \
             ({} B counted for the chain){}",
            a.label,
            a.steps_repr(),
            min,
            a.median(),
            min / (ARM_LEGS * elems) as f64,
            a.bytes as f64 / min,
            a.bytes,
            a.route_note()
        );
    }
    let (dense, channel, scalar, trailing) =
        (arms[0].min(), arms[1].min(), arms[2].min(), arms[3].min());
    // `channel/dense` carries both the addressing and the smaller operand read (a
    // dense operand is read in full), so it is an upper bound on the addressing.
    // `channel/trailing` holds the operand read and the remainder constant and
    // moves only the quotient (`i/T` vs `i/1`), so it isolates the division.
    let (cd, ct, sd) = (channel / dense, channel / trailing, scalar / dense);
    // This harness's own floor: how far a single arm moves between its best and its
    // typical sample in *this* run. A cross-arm difference below it is not a signal.
    let spread = arms
        .iter()
        .map(|a| a.median() / a.min())
        .fold(0.0f64, f64::max);
    eprintln!(
        "[part C] conclusion (whole chain, min wall time): channel/dense={cd:.2}x, \
         channel/trailing={ct:.2}x (the quotient isolated against the same remainder), \
         scalar/dense={sd:.2}x, against this harness's own spread within the run (worst arm \
         min→median {spread:.2}x). Read as an upper bound on the addressing, the two ratios that \
         carry it put it at at most ~{:+.0}% (channel/dense) and ~{:+.0}% (channel/trailing) of the \
         step's own time — a difference this run's spread does not resolve below, and an overstatement \
         of the addressing either way, since both addressed arms read a much smaller operand than the \
         dense one. Not acted on: the shader already skips the addressing entirely for a dense operand \
         (`len == 0`), so the only fix left is another kernel variant, which is not a small, \
         merge-safe change.",
        // A ratio below 1 is noise inside the spread, so the bounds floor at 0 —
        // a negative "at most" would be a nonsense bound on the addressing cost.
        (cd - 1.0).max(0.0) * 100.0,
        (ct - 1.0).max(0.0) * 100.0
    );
}

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn elementwise_broadcast_cost_over_decoder_chain_shapes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    common::refuse_under_run_report();
    eprintln!(
        "[elementwise_broadcast_cost] wgpu adapter: {}",
        rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into())
    );
    for &(c, t) in &SHAPES {
        let label = format!("(C={c}, T={t}), {LEGS} legs of [1,{c},{t}] over [1,{c},1]");
        let legalised = legalise(&chain(c, t));
        let peeled = rlx_wgpu::unfuse::elide_binary_broadcast_expands(legalised.clone());
        part_a(&label, &legalised, &peeled);
        part_b(&label, &legalised, peeled);
        part_c(&label, c, t);
    }
}
