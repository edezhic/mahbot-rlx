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

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Pair, Role};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op, Shape, Tick};
use rlx_runtime::{Device, Session};

mod support;

/// The body's cache capacity: any capacity builds the same multiplies.
const BODY_CAPACITY: usize = 330;
/// Packed bytes one harness call streams, before the shape's own step count is
/// rounded down to whole copies and capped. The stream cell
/// (`rlx-wgpu/tests/gguf_one_row_stream_rate.rs`) carries 214-302 MB a call, and
/// the rate a shape reaches rises with the call it is measured in.
const HARNESS_TARGET_BYTES: u64 = 224 * 1024 * 1024;
/// A ceiling on the small call's copies — the marginal leg carries twice as many —
/// so a small shape does not compile a graph of thousands of steps.
const HARNESS_MAX_STEPS: usize = 256;
/// Rounds a shape's two calls are alternated in; the marginal rate is the median
/// of the paired differences, a difference of two calls being already robust to a
/// level shift, where the least of them would report the quietest round.
const HARNESS_ROUNDS: usize = 5;
/// The frame's own wall time is the least of its rounds — the least perturbed
/// sample of one whole frame.
const FRAME_ROUNDS: usize = 5;

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

fn last_dim(shape: &Shape) -> usize {
    shape
        .dims()
        .last()
        .expect("a multiply has a rank")
        .unwrap_static()
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

fn least(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings compare"));
    samples[0]
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings compare"));
    samples[samples.len() / 2]
}

/// One packed weight's bytes, generated rather than quantized: a deterministic
/// ramp with the f16 scale fields pinned to a finite exponent, the shape
/// `rlx-wgpu/tests/common/mod.rs::packed_gguf_weight` uses. Only the timing is
/// read, so the values need only be finite.
fn packed_ramp(scheme: QuantScheme, len: usize) -> Vec<u8> {
    let block_bytes = scheme.gguf_block_bytes() as usize;
    // Byte offsets within a block whose high half is an f16 exponent: Q4_K's `d`
    // and `dmin` at its head, Q6_K's `d` at its tail, Q8_0's `d` at its head.
    let f16_high: &[usize] = match scheme {
        QuantScheme::GgufQ4K => &[1, 3],
        QuantScheme::GgufQ6K => &[209],
        QuantScheme::GgufQ8_0 => &[1],
        other => panic!("{other:?} has no packed ramp"),
    };
    let mut state = 7u32;
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let byte = (state >> 24) as u8;
            if f16_high.contains(&(i % block_bytes)) {
                0x08 | (byte & 0x03)
            } else {
                byte & 0x3F
            }
        })
        .collect()
}

/// One call of `steps` copies of the same packed multiply over one row: the
/// compiled call and its input row, ready to be timed. Its outputs are checked once
/// against `steps` and `n` when it is built.
struct Call {
    compiled: rlx_runtime::CompiledGraph,
    input: Vec<f32>,
}

fn call_of(scheme: QuantScheme, k: usize, n: usize, bytes: &[u8], steps: usize) -> Call {
    let len = bytes.len();
    let mut g = Graph::new(format!("census_{scheme}_{k}x{n}_{steps}"));
    let x = g.input("x", Shape::new(&[1, k], DType::F32));
    let mut outs = Vec::with_capacity(steps);
    for step in 0..steps {
        let w = g.param(format!("w{step}"), Shape::new(&[len], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[1, n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    let mut compiled = Session::new(Device::Gpu).compile(g);
    for step in 0..steps {
        compiled.set_param_typed(&format!("w{step}"), bytes, DType::U8);
    }
    let input: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let mut call = Call { compiled, input };
    let out = call.run();
    assert_eq!(out.len(), steps, "one output per priced multiply");
    assert!(
        out.iter()
            .all(|row| row.len() == n && row.iter().all(|v| v.is_finite())),
        "harness {scheme} k={k} n={n}: every output row is {n} finite values"
    );
    call
}

impl Call {
    fn run(&mut self) -> Vec<Vec<f32>> {
        self.compiled.run(&[("x", self.input.as_slice())])
    }
    fn ms(&mut self) -> f64 {
        let start = Tick::now();
        let _ = self.run();
        Tick::now().elapsed_ns(start) as f64 / 1e6
    }
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
    let steps = ((HARNESS_TARGET_BYTES / packed_len.max(1)) as usize).clamp(1, HARNESS_MAX_STEPS);
    let mut small = call_of(scheme, k, n, &bytes, steps);
    let mut large = call_of(scheme, k, n, &bytes, 2 * steps);
    let mut small_ms = Vec::with_capacity(HARNESS_ROUNDS);
    let mut added_ms = Vec::with_capacity(HARNESS_ROUNDS);
    for _ in 0..HARNESS_ROUNDS {
        let a = small.ms();
        let b = large.ms();
        small_ms.push(a);
        added_ms.push(b - a);
    }
    let total = (steps as u64 * packed_len) as f64;
    let (call_ms, added_ms) = (least(small_ms), median(added_ms));
    let (call, marginal) = (gbps(total, call_ms), gbps(total, added_ms));
    println!(
        "census harness scheme={scheme} k={k} n={n} packed={packed_len} steps={steps} \
         call_bytes={} call_ms={call_ms:.3} call_GBps={call:.1} added_ms={added_ms:.3} \
         marginal_GBps={marginal:.1}",
        total as u64,
    );
    Some((call, marginal))
}

/// The frame's own wall time: every graph input filled with zeros, one warm run,
/// then [`FRAME_ROUNDS`] timed runs, the least.
fn frame_ms(compiled: &mut graphs::Compiled) -> f64 {
    let inputs: Vec<(String, Vec<f32>)> = compiled
        .runner
        .graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Input { name } => Some((
                name.clone(),
                vec![0f32; node.shape.num_elements().expect("a static input shape")],
            )),
            _ => None,
        })
        .collect();
    compiled.runner.run(&inputs);
    let mut samples = Vec::with_capacity(FRAME_ROUNDS);
    for _ in 0..FRAME_ROUNDS {
        let start = Tick::now();
        compiled.runner.run(&inputs);
        samples.push(Tick::now().elapsed_ns(start) as f64 / 1e6);
    }
    least(samples)
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
            Op::Input { name } if matches!(name.as_str(), "x" | "cos" | "sin" | "mask") => Some((
                name.clone(),
                vec![0f32; node.shape.num_elements().expect("a static input shape")],
            )),
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
    least(samples)
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
    let body = graphs::body_capacity_chain_planes(pair.body.clone(), BODY_CAPACITY)
        .expect("build the plane-cache decode body");
    let mut body = graphs::Compiled::new(Device::Gpu, body).expect("compile the body");
    let (layers, hkv) = {
        let cfg = pair.body.config();
        (cfg.layers, cfg.kv_heads * cfg.head_dim)
    };
    let k = vec![vec![0f32; (BODY_CAPACITY + 1) * hkv]; layers];
    let v = vec![vec![0f32; (BODY_CAPACITY + 1) * hkv]; layers];
    assert!(
        body.runner.resident_cache_planes(&k, &v),
        "the body took the resident plane cache"
    );
    let row = BODY_CAPACITY / 2;
    let reads = graphs::plane_step_reads(layers);
    let body_ms = resident_plane_ms(&mut body, row, hkv, &reads);
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
    let frame = graphs::tts_code_predictor_frame(
        pair.sound.clone(),
        steps + 1,
        graphs::SampleSpec {
            top_k,
            top_p,
            temperature,
            seed: 12345,
        },
    )
    .expect("build the frame graph");
    let mut frame = graphs::Compiled::new(Device::Gpu, frame).expect("compile the frame");
    let frame_ms = frame_ms(&mut frame);
    rows.extend(side_rows(
        "predictor",
        &mut frame,
        steps + 1,
        ceiling.read_gb_s,
        frame_ms,
    ));
    drop(frame);

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

    // What the frame's own measured time is made of: the multiplies at the rates
    // they reach in isolation, and the rest — attention, the norms, the sampling
    // and every dispatch around them, which no multiply form can move.
    for side in ["body", "predictor"] {
        let side_rows: Vec<&Row> = rows.iter().filter(|row| row.side == side).collect();
        let frame_ms = side_rows[0].frame_ms;
        let levels = side_rows[0].levels;
        let marginal: f64 = side_rows.iter().filter_map(|row| row.marginal_ms()).sum();
        let stream: f64 = side_rows.iter().map(|row| row.stream_ms()).sum();
        println!(
            "census explain side={side} per_level_ms={:.3} marginal_sum_ms={marginal:.3} \
             stream_sum_ms={stream:.3} other_ms={:.3}",
            frame_ms / levels as f64,
            frame_ms / levels as f64 - marginal,
        );
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
