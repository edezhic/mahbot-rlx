// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and `#[ignore]`d: the synthesis loop's per-shape rate census.
//! Needs the model files (`MAHBOT_MODELS_DIR`); without them it prints why and
//! returns. Run with `MAHBOT_MODELS_DIR=… RLX_ALLOW_THROTTLE=1 cargo test
//! -p mahbot-inference --release --test multiply_census -- --ignored --nocapture`.
//! `frame_share_ms` is each family's bytes at the frame's own measured aggregate
//! rate — a per-side constant, so within a side ordering by it is ordering by
//! bytes. A packed multiply's own in-frame time is not attributable: it dispatches
//! inside a pass the frame's other steps share, so the run report prices the pass.
//!
//! **What it does.** It compiles the shipped predictor frame, then stands one
//! subject down at a time — each non-multiply kind, the packed multiply's two
//! encode-time arms, and the gather index-add fold — and pairs every arm against the
//! shipped frame. Three things make an arm a price rather than a print: its schedule
//! census is **asserted** against the shipped one (equal for an encode-time arm,
//! moved for every other), its outputs are checked against the shipped frame's on
//! the same inputs, and its delta comes from paired rounds with the arm that runs
//! second alternating. A frame's milliseconds, its `loop_ms` over the cell's frames
//! and `wins` (the rounds the stood-down arm was faster) are what it prints, and
//! the file states no price of its own.
//!
//! **The reconciliation.** `census reconcile` puts the measured frame against the
//! sums the harness already prints — the multiply stream at the shapes' marginal
//! rates, the savings the priced arms collect, and what stays unattributed — so the
//! chain's total comes from the frame the harness measures rather than from a
//! per-dispatch price times a dispatch count. Each arm is a delta against the
//! shipped frame, so their sum is an upper bound on the part they explain, not a
//! partition. Attention is a **bound, not a saving**: its only switch changes the
//! arrangement rather than the dispatch count, so the kind is closed as bounded,
//! not priced. What each kind's arm does to the graph is in the arms table below.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Pair, Role};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{Graph, NodeId, Op, Tick};
use rlx_runtime::Device;

mod support;

use mahbot_inference::synth::MAX_CODEC_ROWS;
use support::{
    arm_once, arm_outputs, call_of, frame_arm, greatest, last_dim, least, median, packed_ramp,
    ramped_inputs, step_census, steps_for, zero_inputs,
};

/// The body's cache capacity: any capacity builds the same multiplies.
const BODY_CAPACITY: usize = 330;
/// Packed bytes one harness call streams, before the shape's own step count is
/// rounded down to whole copies and capped. Larger than the shared
/// [`support::TARGET_BYTES`] (96 MiB) on purpose: this census prices every shape in
/// the models against the stream cell
/// (`rlx-wgpu/tests/gguf_one_row_stream_rate.rs`, 214-302 MB a call), so its call
/// carries the same bytes and its rates are comparable with that cell's.
const CENSUS_TARGET_BYTES: u64 = 224 * 1024 * 1024;
/// Rounds a shape's two calls are alternated in; the marginal rate is the median
/// of the paired differences, a difference of two calls being already robust to a
/// level shift, where the least of them would report the quietest round.
const HARNESS_ROUNDS: usize = 5;
/// The frame's own wall time is the least of its rounds — the least perturbed
/// sample of one whole frame.
const FRAME_ROUNDS: usize = 5;
/// Rounds each ablation's two arms alternate in. One frame's run-to-run spread is
/// several times the effect, so a single pair cannot resolve it; the median of
/// this many paired differences either separates a 0.1-0.5 ms effect or bounds
/// it, and the win count is the sign test that says which.
const PAIR_ROUNDS: usize = 25;

fn pair(role: Role) -> Option<Arc<Pair>> {
    let Ok(dir) = loader::models_dir() else {
        eprintln!(
            "skip: {} is not set — the census needs a model",
            loader::MODELS_DIR_ENV
        );
        return None;
    };
    match Pair::of_role(role) {
        Ok(pair) => Some(Arc::new(pair)),
        Err(err) => {
            eprintln!(
                "skip: {} names no pair for this role ({err}) — the census did not run",
                dir.display()
            );
            None
        }
    }
}

/// The tensor a weight site belongs to: `q`'s marker off, then the stacked
/// tensor's step, then the block index — `a.gen.code.blk.3.gate_up.fused\0q`
/// and `blk.7.gate_up.fused\0q` are one family.
fn tensor_of(name: &str) -> String {
    let name = name.strip_suffix("\0q").unwrap_or(name);
    let name = name.split('#').next().unwrap_or(name);
    let mut parts: Vec<&str> = name.split('.').collect();
    if let Some(at) = parts.iter().position(|part| *part == "blk") {
        parts.drain(..=at + 1);
    }
    parts.join(".")
}

/// One priced multiply site: a `DequantMatMul` over a packed parameter.
struct Site {
    tensor: String,
    scheme: QuantScheme,
    k: usize,
    n: usize,
    bytes: u64,
}

/// Every packed multiply of `graph` in node order, live or not: the second
/// operand is the weight, and only a parameter in that position is a read.
fn sites(graph: &Graph) -> (Vec<(NodeId, Site)>, usize) {
    let mut out = Vec::new();
    let mut activation_weighted = 0usize;
    for node in graph.nodes() {
        let Op::DequantMatMul { scheme } = node.op else {
            continue;
        };
        let Some(&weight) = node.inputs.get(1) else {
            continue;
        };
        let Op::Param { name } = &graph.node(weight).op else {
            activation_weighted += 1;
            continue;
        };
        let Some(bytes) = graph.node(weight).shape.size_bytes() else {
            continue;
        };
        out.push((
            node.id,
            Site {
                tensor: tensor_of(name),
                scheme,
                k: last_dim(&graph.node(node.inputs[0]).shape),
                n: last_dim(&node.shape),
                bytes: bytes as u64,
            },
        ));
    }
    (out, activation_weighted)
}

/// The ids the graph's outputs reach, which is the set the stand's
/// [`stream_bytes`](mahbot_inference::stand::stream_bytes) counts reads over.
fn live_ids(graph: &Graph) -> HashSet<NodeId> {
    let mut live: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let mut stack: Vec<NodeId> = graph.outputs.clone();
    while let Some(id) = stack.pop() {
        for &input in &graph.node(id).inputs {
            if live.insert(input) {
                stack.push(input);
            }
        }
    }
    live
}

fn gbps(bytes: f64, ms: f64) -> f64 {
    bytes / (ms / 1000.0) / 1e9
}

/// One shape's isolated rates, `(call, marginal)` in GB/s, from two calls of the
/// same multiply at `steps` and `2 * steps` copies, alternated inside every round so
/// drift lands on both alike. A call carries its run's fixed cost — one submission,
/// one readback and their fence — and the marginal rate, the second call's own bytes
/// over the time it added, is the rate a frame's dispatches see. `None` when the
/// adapter cannot hold the shape, with the reason printed.
fn harness_rate(scheme: QuantScheme, k: usize, n: usize, packed_len: u64) -> Option<(f64, f64)> {
    let wgpu = rlx_wgpu::device::wgpu_device()?;
    let max_buffer = wgpu.device.limits().max_buffer_size;
    if packed_len > max_buffer {
        println!(
            "census harness scheme={scheme} k={k} n={n} packed={packed_len} skip=does_not_fit_adapter \
             max_buffer={max_buffer}"
        );
        return None;
    }
    let bytes = packed_ramp(scheme, packed_len as usize);
    let steps = steps_for(packed_len, CENSUS_TARGET_BYTES);
    let mut small = call_of(scheme, k, n, 1, &bytes, steps);
    let mut large = call_of(scheme, k, n, 1, &bytes, 2 * steps);
    let mut small_ms = Vec::with_capacity(HARNESS_ROUNDS);
    let mut added_ms = Vec::with_capacity(HARNESS_ROUNDS);
    for _ in 0..HARNESS_ROUNDS {
        let a = small.ms();
        let b = large.ms();
        small_ms.push(a);
        added_ms.push(b - a);
    }
    let total = (steps as u64 * packed_len) as f64;
    let (call_ms, added_ms) = (least(&small_ms), median(&added_ms));
    let (call, marginal) = (gbps(total, call_ms), gbps(total, added_ms));
    println!(
        "census harness scheme={scheme} k={k} n={n} packed={packed_len} steps={steps} \
         call_bytes={} call_ms={call_ms:.3} call_GBps={call:.1} added_ms={added_ms:.3} \
         marginal_GBps={marginal:.1}",
        total as u64,
    );
    Some((call, marginal))
}

/// Two arms of one frame in one paired window: `rounds` pairs with the arm that runs
/// second alternating, so a level shift or a warm-state drift lands on both arms.
/// Each run installs its arm's own switch values, so a run-time switch is the
/// arm's at the dispatch. Returns each arm's rounds and their per-round
/// differences, `b - a`.
fn paired_frame(
    a: &mut graphs::Compiled,
    a_set: &[(&str, &str)],
    a_in: &[(String, Vec<f32>)],
    b: &mut graphs::Compiled,
    b_set: &[(&str, &str)],
    b_in: &[(String, Vec<f32>)],
    rounds: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    arm_once(a, a_set, a_in);
    arm_once(b, b_set, b_in);
    let (mut a_ms, mut b_ms, mut diffs) = (Vec::new(), Vec::new(), Vec::with_capacity(rounds));
    for round in 0..rounds {
        let (x, y) = if round % 2 == 0 {
            (arm_once(a, a_set, a_in), arm_once(b, b_set, b_in))
        } else {
            let y = arm_once(b, b_set, b_in);
            let x = arm_once(a, a_set, a_in);
            (x, y)
        };
        a_ms.push(x);
        b_ms.push(y);
        diffs.push(y - x);
    }
    (a_ms, b_ms, diffs)
}

/// The largest difference between two arms' output lists, element for element: a
/// stand-down's value witness, with the codes themselves compared by the caller.
fn output_delta(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    assert_eq!(a.len(), b.len(), "both arms produce the same outputs");
    let mut max_abs = 0f64;
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.len(), y.len(), "both arms' outputs have one width");
        for (u, v) in x.iter().zip(y) {
            max_abs = max_abs.max((u - v).abs() as f64);
        }
    }
    max_abs
}

/// One paired window's line: both arms' medians, the paired difference's median and
/// spread, the stood-down arm's wins, and what the difference is worth over the
/// cell's own 330 generated frames -- a per-frame figure, not a per-level one.
fn print_pair(subject: &str, a_ms: &[f64], b_ms: &[f64], diffs: &[f64]) {
    let delta = median(diffs);
    println!(
        "census pair subject={subject} rounds={} shipped_ms={:.3} {subject}_ms={:.3} \
         median_delta_ms={:+.3} min_delta_ms={:+.3} max_delta_ms={:+.3} \
         {subject}_wins={}/{} loop_ms={:+.1}",
        diffs.len(),
        median(a_ms),
        median(b_ms),
        delta,
        least(diffs),
        greatest(diffs),
        diffs.iter().filter(|d| **d < 0.0).count(),
        diffs.len(),
        delta * MAX_CODEC_ROWS as f64,
    );
}

fn frame_ms(compiled: &mut graphs::Compiled, switches: &[(&str, &str)]) -> f64 {
    let inputs = zero_inputs(&compiled.runner.graph);
    arm_once(compiled, switches, &inputs);
    let samples: Vec<f64> = (0..FRAME_ROUNDS)
        .map(|_| arm_once(compiled, switches, &inputs))
        .collect();
    least(&samples)
}

/// The body's resident plane-cache step, which is the path the synthesis loop
/// takes: every input but `k*`/`v*` zero-filled at its graph shape, the per-layer
/// planes bound once as resident handles, and the step's own row folded in
/// device-side. Feeding the caches as ordinary inputs instead uploads both planes
/// on every run — `2 · layers · (capacity + 1) · hkv` floats — so that figure
/// prices the upload rather than the cell.
fn resident_plane_ms(
    compiled: &mut graphs::Compiled,
    row: usize,
    row_elems: usize,
    reads: &[usize],
) -> f64 {
    let inputs: Vec<(String, Vec<f32>)> = compiled
        .runner
        .graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Input { name } if matches!(name.as_str(), "x" | "cos" | "sin" | "mask") => {
                let len = node.shape.num_elements().expect("a static input shape");
                let values = if name == "mask" {
                    // The step's own mask: rows `0..row` carry the history the
                    // cell has written and the spare slot (`len - 1`) is this
                    // step's own query, exactly as the loop feeds it. Zeros left
                    // over that would make the step skip every key the plane
                    // holds and price an empty cache.
                    let mut m = vec![0f32; len];
                    for v in &mut m[..row.min(len)] {
                        *v = 1.0;
                    }
                    m[len - 1] = 1.0;
                    m
                } else {
                    vec![0f32; len]
                };
                Some((name.clone(), values))
            }
            _ => None,
        })
        .collect();
    let run = |compiled: &mut graphs::Compiled| {
        assert!(
            compiled
                .runner
                .run_resident_planes(&inputs, row, row_elems, reads)
                .is_some(),
            "the body refused its resident plane feed"
        );
    };
    run(compiled);
    let mut samples = Vec::with_capacity(FRAME_ROUNDS);
    for _ in 0..FRAME_ROUNDS {
        let start = Tick::now();
        run(compiled);
        samples.push(Tick::now().elapsed_ns(start) as f64 / 1e6);
    }
    least(&samples)
}

/// One shape family's row of the census. `bytes` is the family's whole-frame
/// bytes for the body and one level's share for the predictor; `count` stays the
/// frame's own site count.
struct Row {
    side: &'static str,
    tensor: String,
    scheme: QuantScheme,
    k: usize,
    n: usize,
    count: usize,
    levels: usize,
    bytes_per_site: u64,
    bytes: u64,
    frame_ms: f64,
    frame_bytes: u64,
    /// The shape's own isolated rates in GB/s: a whole call's, and the marginal one
    /// a frame's dispatch sees (`None` where the adapter could not hold the shape).
    call_gbps: Option<f64>,
    marginal_gbps: Option<f64>,
    ceiling_gbps: f64,
}

impl Row {
    /// The row's bytes at the frame's aggregate rate — not the multiply's own
    /// in-frame time, which a shared pass makes unattributable.
    fn frame_share_ms(&self) -> f64 {
        self.bytes as f64 * self.frame_ms / self.frame_bytes as f64
    }
    /// The row's share of its side's per-level bytes.
    fn share_pct(&self) -> f64 {
        100.0 * (self.bytes * self.levels as u64) as f64 / self.frame_bytes as f64
    }
    /// The row's bytes at the rate its own shape reaches inside a call, once the
    /// call's fixed cost is removed — the number a frame's dispatch sees.
    fn marginal_ms(&self) -> Option<f64> {
        self.marginal_gbps
            .map(|marginal| self.bytes as f64 / (marginal * 1e9) * 1000.0)
    }
    fn stream_ms(&self) -> f64 {
        self.bytes as f64 / (self.ceiling_gbps * 1e9) * 1000.0
    }
    fn lost_ms(&self) -> f64 {
        self.frame_share_ms() - self.stream_ms()
    }
    fn marginal_gap_ms(&self) -> Option<f64> {
        self.marginal_ms().map(|ms| self.frame_share_ms() - ms)
    }
    /// What the shape's own bytes still cost above the device's read ceiling: the
    /// form's remaining headroom, as distinct from the frame's other dispatches.
    fn headroom_ms(&self) -> Option<f64> {
        self.marginal_ms().map(|ms| ms - self.stream_ms())
    }
    /// The contracted work the row's bytes carry, over `ms`.
    fn tflops(&self, ms: f64) -> f64 {
        2.0 * self.k as f64 * self.n as f64 * self.count as f64
            / self.levels as f64
            / (ms * 1e-3)
            / 1e12
    }
}

/// One family's live site count and packed size, in `(tensor, scheme, k, n)`
/// order.
type Families = BTreeMap<(String, String, usize, usize), (QuantScheme, usize, u64)>;

/// The families of one side, priced and described; the frame's own byte total
/// comes from one compiled graph, its time from the caller — the two legs are fed
/// differently, so each measures its own.
fn side_rows(
    side: &'static str,
    compiled: &mut graphs::Compiled,
    levels: usize,
    ceiling_gbps: f64,
    ms: f64,
) -> Vec<Row> {
    let (fams, live_bytes) = {
        let graph = &compiled.runner.graph;
        let (sites, activation_weighted) = sites(graph);
        let live = live_ids(graph);
        let total = sites.len();
        let live_sites: Vec<&Site> = sites
            .iter()
            .filter(|(id, _)| live.contains(id))
            .map(|(_, site)| site)
            .collect();
        let dead = total - live_sites.len();
        let live_bytes: u64 = live_sites.iter().map(|site| site.bytes).sum();
        let graph_params: u64 = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::Param { .. }))
            .filter_map(|node| node.shape.size_bytes().map(|bytes| bytes as u64))
            .sum();
        println!(
            "census sites side={side} total={total} live={} dead={dead} \
             activation_weighted={activation_weighted} live_bytes={live_bytes} \
             stream_bytes={} graph_params={graph_params}",
            live_sites.len(),
            mahbot_inference::stand::stream_bytes(graph),
        );
        assert_eq!(
            activation_weighted, 0,
            "{side}: every packed multiply reads a parameter in its weight slot"
        );
        assert_eq!(
            live_bytes,
            mahbot_inference::stand::stream_bytes(graph),
            "{side}: the priced packed multiplies are the live weight bytes"
        );

        let mut fams: Families = BTreeMap::new();
        for site in &live_sites {
            let entry = fams
                .entry((site.tensor.clone(), site.scheme.to_string(), site.k, site.n))
                .or_insert((site.scheme, 0, site.bytes));
            assert_eq!(
                entry.2, site.bytes,
                "{side}: one (tensor, scheme, k, n) has one packed size"
            );
            entry.1 += 1;
        }
        let family_bytes: u64 = fams
            .values()
            .map(|(_, count, bytes)| *count as u64 * *bytes)
            .sum();
        assert_eq!(
            family_bytes, live_bytes,
            "{side}: the frame's families cover every live multiply byte"
        );
        (fams, live_bytes)
    };

    // The predictor's rows are one level each, so its frame line is one level too.
    let per_level_ms = ms / levels as f64;
    println!(
        "census frame side={side} ms={ms:.3} per_level_ms={per_level_ms:.3} bytes={live_bytes} \
         GBps={:.1} stream_ms={:.3} levels={levels} note=frame_share_ms_is_bytes_at_this_rate",
        gbps(live_bytes as f64, ms),
        live_bytes as f64 / (ceiling_gbps * 1e9) * 1000.0,
    );
    fams.into_iter()
        .map(|((tensor, _, k, n), (scheme, count, bytes_per_site))| Row {
            side,
            tensor,
            scheme,
            k,
            n,
            count,
            levels,
            bytes_per_site,
            bytes: bytes_per_site * count as u64 / levels as u64,
            frame_ms: ms,
            frame_bytes: live_bytes,
            call_gbps: None,
            marginal_gbps: None,
            ceiling_gbps,
        })
        .collect()
}

#[test]
#[ignore = "measurement only: prints the loop's per-shape rate census"]
fn the_synthesis_loops_per_shape_rate_census_with_models() {
    // The report's own passes perturb the frame time this census measures.
    support::assert_run_report_off();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair(Role::Synthesis) else {
        return;
    };
    let ceiling = rlx_wgpu::bandwidth::stream_ceiling().expect("the stream probe runs");
    println!(
        "census ceiling adapter={:?} read_GBps={:.1} copy_GBps={:.1} rounds={} \
         bytes_per_direction={}",
        rlx_wgpu::device::adapter_name(),
        ceiling.read_gb_s,
        ceiling.gb_s,
        ceiling.rounds,
        ceiling.bytes_per_direction,
    );

    let mut rows = Vec::new();

    // The synthesis body: a plane-cache decode step at one row, fed the way the
    // loop feeds it — the per-layer planes resident, only this step's row crossing.
    // The shipped switch values are installed across the body's build, compile and
    // runs, as they are for the predictor arms: a run-time switch is read at the
    // dispatch, so an ambient one would otherwise move this row without moving the
    // label it is recorded under.
    let (layers, hkv) = {
        let cfg = pair.body.config();
        (cfg.layers, cfg.kv_heads * cfg.head_dim)
    };
    let mut body = {
        let _switch = rlx_ir::env::RuntimeOverrides::install(support::FRAME_SWITCHES);
        let body = graphs::body_capacity_chain_planes(pair.body.clone(), BODY_CAPACITY)
            .expect("build the plane-cache decode body");
        let mut body = graphs::Compiled::new(Device::Gpu, body).expect("compile the body");
        let k = vec![vec![0f32; (BODY_CAPACITY + 1) * hkv]; layers];
        let v = vec![vec![0f32; (BODY_CAPACITY + 1) * hkv]; layers];
        assert!(
            body.runner.resident_cache_planes(&k, &v),
            "the body took the resident plane cache"
        );
        body
    };
    let row = BODY_CAPACITY / 2;
    let reads = graphs::plane_step_reads(layers);
    let body_ms = {
        let _switch = rlx_ir::env::RuntimeOverrides::install(support::FRAME_SWITCHES);
        resident_plane_ms(&mut body, row, hkv, &reads)
    };
    // The step's own row, and the two outputs it reads — the hidden row and the
    // logits; the `2 · layers` cache rows stay device-side.
    println!(
        "census body resident row={row} capacity={BODY_CAPACITY} layers={layers} hkv={hkv} \
         hidden_out={} logits_out={}",
        reads[0], reads[1],
    );
    rows.extend(side_rows("body", &mut body, 1, ceiling.read_gb_s, body_ms));
    drop(body);

    // The predictor frame: one whole sixteen-level walk, reported per level.
    let steps = graphs::code_predictor_steps(&pair.sound).expect("predictor head count");
    let (top_k, top_p, temperature) = pair.body.sampling();
    let spec = graphs::SampleSpec {
        top_k,
        top_p,
        temperature,
        seed: 12345,
    };
    // Each subject stood down against the shipped frame in one paired window, its
    // own census printed and asserted against the shipped schedule, and its outputs
    // checked against the shipped frame's on the same inputs. The arms, in table
    // order, and what each stands down:
    //
    //   gather_index_bias  the gather index-add fold — the one arm a lowering
    //                      fusion makes, so its stand-down *adds* the fold's step
    //                      back rather than removing one
    //   residual_ln_tee    `fused_residual_ln_tee` -> a binary add and a
    //                      `layer_norm` each
    //   norm_rope          `norm_rope` -> a norm and a `rope` each
    //   gated_product      `fused_swiglu` -> `narrow`, an activation and a
    //                      `binary` multiply each
    //   rms_norm           `fused_residual_rms_norm` -> a binary add and a rms
    //                      norm each
    //   attention          the many-row dense arrangement's multiplies, mask and
    //                      softmax
    //   concat             the concat, run on the host
    //   multiply_setup     the packed multiply's cached uniform and bind group,
    //                      read when the pass is encoded
    //   multiply_inplace   the packed multiply's in-place store, stood down to
    //                      the pooled slab and its copy
    //
    // The sampler is not an arm: `Argmax` is the same dispatch count as `Sample`,
    // so its stand-down would price a kernel rather than a dispatch, and a greedy
    // draw is not the frame's values. One arm beyond the shipped frame is resident
    // at a time: each is built, censused, valued, paired, dropped.
    let (mut shipped, shipped_set) = frame_arm(&pair, steps + 1, spec, &[]);
    // Every stand-down must leave the frame computing the same thing: its outputs
    // are checked against the shipped frame's on the same inputs, against
    // [`VALUE_BAR`], and the max difference is printed as the arm's witness — the
    // fusions' own sums are the same products in another order, not another
    // operation. `saving` is whether the delta is a saving the shipped form
    // collects (a fusion the frame keeps, a store it makes in place) or a bound on
    // the shipped form's worth (the attention arrangement): the two do not add up.
    // The arms whose switch acts when the pass is encoded rather than at the
    // lowering: they change what the frame dispatches, not the schedule it
    // compiled, so the census must be the shipped one.
    const ENCODE_TIME_ARMS: [&str; 2] = ["multiply_setup", "multiply_inplace"];
    let arms: [(&str, &[(&str, &str)], bool); 9] = [
        (
            "gather_index_bias",
            &[("RLX_WGPU_NO_GATHER_INDEX_BIAS", "1")],
            true,
        ),
        ("residual_ln_tee", &[("RLX_WGPU_NO_LN_TEE", "1")], true),
        ("norm_rope", &[("RLX_WGPU_NO_NORM_ROPE", "1")], true),
        ("gated_product", &[("RLX_WGPU_NO_FUSED_SWIGLU", "1")], true),
        ("rms_norm", &[("RLX_WGPU_NO_FUSED_RESIDUAL_RMS", "1")], true),
        // A bound on the shipped arrangement's worth, not a saving: at `1` a
        // one-row step takes the many-row form, which no frame would ship.
        ("attention", &[("RLX_WGPU_ATTN_DENSE_ONE_ROW", "1")], false),
        ("concat", &[("RLX_WGPU_CONCAT_HOST", "1")], true),
        ("multiply_setup", &[("RLX_WGPU_GEMV_POOL", "0")], true),
        ("multiply_inplace", &[("RLX_WGPU_GEMV_IN_PLACE", "0")], true),
    ];
    let shipped_in = zero_inputs(&shipped.runner.graph);
    let values_in = ramped_inputs(&shipped.runner.graph);
    let shipped_steps = shipped
        .runner
        .schedule_step_names()
        .expect("wgpu reports its schedule");
    println!(
        "census steps side=predictor arm=shipped steps={} kinds={}",
        shipped_steps.len(),
        step_census(&shipped_steps)
    );
    let mut priced: Vec<(&str, f64)> = Vec::with_capacity(arms.len());
    let mut bound: Vec<(&str, f64)> = Vec::new();
    // The bar a stand-down's outputs are held to: another summation order of the
    // same products, never another operation. The pins in `reference.rs` accept a
    // moved digest as a witness; here the witness is the max difference printed
    // beside it.
    const VALUE_BAR: f64 = 1e-4;
    let shipped_census = step_census(&shipped_steps);
    for (subject, set, saving) in arms {
        let (mut arm, arm_set) = frame_arm(&pair, steps + 1, spec, set);
        let names = arm
            .runner
            .schedule_step_names()
            .expect("wgpu reports its schedule");
        let census = step_census(&names);
        let moved = names.len() != shipped_steps.len() || census != shipped_census;
        println!(
            "census steps side=predictor arm={subject} stood_down steps={} kinds={census} \
             schedule={}",
            names.len(),
            if moved { "changed" } else { "as_shipped" },
        );
        assert_eq!(
            moved,
            !ENCODE_TIME_ARMS.contains(&subject),
            "{subject}: a lowering stand-down must move the schedule and an encode-time one \
             must not ({} steps against the shipped {shipped_census})\n{census}",
            names.len(),
        );
        let a_out = arm_outputs(&mut shipped, &shipped_set, &values_in);
        let b_out = arm_outputs(&mut arm, &arm_set, &values_in);
        let max_abs = output_delta(&a_out, &b_out);
        assert!(
            a_out
                .iter()
                .chain(b_out.iter())
                .all(|out| out.iter().all(|v| v.is_finite())),
            "{subject}: every output value is finite"
        );
        println!(
            "census values side=predictor subject={subject} outputs={} max_abs={max_abs:.3e} \
             codes_equal={}",
            a_out.len(),
            a_out.first() == b_out.first(),
        );
        assert!(
            max_abs <= VALUE_BAR && a_out.first() == b_out.first(),
            "{subject}: the stand-down moved the frame's values (max |delta| {max_abs:e})"
        );
        let arm_in = zero_inputs(&arm.runner.graph);
        let (a_ms, b_ms, diffs) = paired_frame(
            &mut shipped,
            &shipped_set,
            &shipped_in,
            &mut arm,
            &arm_set,
            &arm_in,
            PAIR_ROUNDS,
        );
        print_pair(subject, &a_ms, &b_ms, &diffs);
        let delta = median(&diffs);
        if saving {
            priced.push((subject, delta));
        } else {
            bound.push((subject, delta));
        }
        drop(arm);
    }
    let shipped_ms = frame_ms(&mut shipped, &shipped_set);
    rows.extend(side_rows(
        "predictor",
        &mut shipped,
        steps + 1,
        ceiling.read_gb_s,
        shipped_ms,
    ));
    drop(shipped);

    // One harness run per distinct (scheme, k, n): the same bytes at one shape's
    // own rate.
    let mut rates: BTreeMap<(String, usize, usize), Option<(f64, f64)>> = BTreeMap::new();
    for row in &rows {
        let key = (row.scheme.to_string(), row.k, row.n);
        if rates.contains_key(&key) {
            continue;
        }
        let rate = harness_rate(row.scheme, row.k, row.n, row.bytes_per_site);
        rates.insert(key, rate);
    }
    for row in &mut rows {
        let rate = rates[&(row.scheme.to_string(), row.k, row.n)];
        row.call_gbps = rate.map(|(call, _)| call);
        row.marginal_gbps = rate.map(|(_, marginal)| marginal);
    }

    // What the frame's own measured time is made of, as two arms. The chained
    // arm is the frame itself: every multiply waits on the norms, the
    // rotation, the attention and the draw before it, at the frame's own shapes.
    // The back-to-back arm is the same shapes at the marginal rate their own call
    // reaches with independent inputs (`harness_rate`). Both rates are taken over
    // the bytes an arm could drive — a shape the adapter cannot hold has no
    // back-to-back rate, so its bytes are out of both sums and `bytes_not_driven`
    // says how much of the frame the gap is not talking about — and the chained
    // share is the frame's time at the bytes rate `frame_share_ms` attributes every
    // per-shape figure with, because a shared pass makes one multiply's own
    // in-frame time unattributable.
    for side in ["body", "predictor"] {
        let side_rows: Vec<&Row> = rows.iter().filter(|row| row.side == side).collect();
        let Some(first) = side_rows.first() else {
            continue;
        };
        let (frame_ms, levels) = (first.frame_ms, first.levels);
        let per_level = frame_ms / levels as f64;
        let bytes: u64 = side_rows.iter().map(|row| row.bytes).sum();
        let driven: u64 = side_rows
            .iter()
            .filter(|row| row.marginal_ms().is_some())
            .map(|row| row.bytes)
            .sum();
        let not_driven = side_rows
            .iter()
            .filter(|row| row.marginal_ms().is_none())
            .count();
        let marginal: f64 = side_rows.iter().filter_map(|row| row.marginal_ms()).sum();
        let stream: f64 = side_rows.iter().map(|row| row.stream_ms()).sum();
        let chained_ms = per_level * driven as f64 / bytes as f64;
        // A shape the adapter cannot hold has no back-to-back rate, so a side with
        // none of them driven has no rate and no gap to divide by.
        if driven == 0 {
            println!(
                "census explain side={side} per_level_ms={per_level:.3} \
                 shapes_not_driven={not_driven} bytes_not_driven={} note=no_shape_driven",
                bytes - driven,
            );
            continue;
        }
        println!(
            "census explain side={side} per_level_ms={per_level:.3} \
             marginal_sum_ms={marginal:.3} marginal_sum_GBps={:.1} stream_sum_ms={stream:.3} \
             other_ms={:.3} gap_pct={:.1} shapes_not_driven={not_driven} bytes_not_driven={}",
            gbps(driven as f64, marginal),
            per_level - marginal,
            100.0 * (chained_ms - marginal) / chained_ms,
            bytes - driven,
        );
        if side == "predictor" {
            // The chain's accounting in one frame's milliseconds, all of it the
            // harness's own: the frame it measured, the multiply stream at the
            // shapes' marginal rates over the same levels, and the arms it paired.
            // The `priced` arms are savings the shipped frame collects, and each is
            // a delta against the shipped frame, so their sum is an upper bound on
            // the part they explain — they overlap in the dispatch setup each
            // re-pays. `unattributed_ms` is what the frame still spends that no
            // saving moved: the non-multiply kernels' own work, attention's above
            // all. The `bound` arm is not a saving to subtract: it is the shipped
            // form's worth against a worse arrangement of the same operation.
            let multiply = marginal * levels as f64;
            let priced_ms: f64 = priced.iter().map(|(_, d)| *d).sum();
            let unattributed = frame_ms - multiply - priced_ms;
            println!(
                "census reconcile side={side} frame_ms={frame_ms:.3} levels={levels} \
                 multiply_ms={multiply:.3} remainder_ms={:.3} priced_ms={priced_ms:+.3} \
                 unattributed_ms={unattributed:+.3} unattributed_pct={:.1} arms=[{}] bound=[{}]",
                frame_ms - multiply,
                100.0 * unattributed / frame_ms,
                priced
                    .iter()
                    .map(|(subject, d)| format!("{subject}:{d:+.3}"))
                    .collect::<Vec<_>>()
                    .join(","),
                bound
                    .iter()
                    .map(|(subject, d)| format!("{subject}:{d:+.3}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }

    for row in &rows {
        let (marginal, gap) = match (row.marginal_ms(), row.marginal_gap_ms()) {
            (Some(marginal_ms), Some(gap_ms)) => (
                format!(
                    "call_GBps={:.1} marginal_ms={marginal_ms:.3} marginal_GBps={:.1} \
                     marginal_TFLOPs={:.3}",
                    row.call_gbps.expect("a measured rate"),
                    row.marginal_gbps.expect("a measured rate"),
                    row.tflops(marginal_ms)
                ),
                format!("marginal_gap_ms={gap_ms:+.3}"),
            ),
            _ => (
                "call_GBps=skip marginal_ms=skip marginal_GBps=skip marginal_TFLOPs=skip"
                    .to_string(),
                "marginal_gap_ms=skip".to_string(),
            ),
        };
        println!(
            "census shape side={} tensor={} scheme={} k={} n={} count={} levels={} \
             bytes_per_site={} bytes={} frame_share_ms={:.3} share_pct={:.1} \
             {marginal} stream_ms={:.3} stream_TFLOPs={:.3} headroom_ms={} lost_ms={:+.3} {gap}",
            row.side,
            row.tensor,
            row.scheme,
            row.k,
            row.n,
            row.count,
            row.levels,
            row.bytes_per_site,
            row.bytes,
            row.frame_share_ms(),
            row.share_pct(),
            row.stream_ms(),
            row.tflops(row.stream_ms()),
            row.headroom_ms()
                .map_or("skip".to_string(), |ms| format!("{ms:+.3}")),
            row.lost_ms(),
        );
    }

    let mut by_lost: Vec<&Row> = rows.iter().collect();
    by_lost.sort_by(|a, b| {
        b.lost_ms()
            .partial_cmp(&a.lost_ms())
            .expect("values compare")
    });
    println!("census rank by=lost_ms note=within_a_side_this_is_the_byte_order");
    for (rank, row) in by_lost.iter().enumerate() {
        println!(
            "rank {} side={} tensor={} scheme={} k={} n={} lost_ms={:+.3}",
            rank + 1,
            row.side,
            row.tensor,
            row.scheme,
            row.k,
            row.n,
            row.lost_ms(),
        );
    }
    let mut by_gap: Vec<&Row> = rows
        .iter()
        .filter(|row| row.marginal_gap_ms().is_some())
        .collect();
    by_gap.sort_by(|a, b| {
        b.marginal_gap_ms()
            .expect("a measured rate")
            .partial_cmp(&a.marginal_gap_ms().expect("a measured rate"))
            .expect("values compare")
    });
    println!("census rank by=marginal_gap_ms");
    for (rank, row) in by_gap.iter().enumerate() {
        println!(
            "rank {} side={} tensor={} scheme={} k={} n={} marginal_gap_ms={:+.3}",
            rank + 1,
            row.side,
            row.tensor,
            row.scheme,
            row.k,
            row.n,
            row.marginal_gap_ms().expect("a measured rate"),
        );
    }
}
