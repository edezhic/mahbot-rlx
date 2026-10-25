// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assertions the cell's gates share, and the measurement helpers the harnesses
//! that price packed multiplies share with them.

#![allow(dead_code)] // each binary takes the subset it uses

use mahbot_inference::graphs;
use mahbot_inference::loader::Pair;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{Device, Session};

/// The least of `samples`: the quietest round of one call.
pub fn least(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::INFINITY, f64::min)
}

/// The greatest of `samples`: the loudest round, `least`'s mirror, seeded so an
/// all-negative arm reports its own bound rather than zero.
pub fn greatest(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The median of `samples`; every caller passes an odd count, so it is the middle
/// sample after sorting.
pub fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("timings compare"));
    sorted[sorted.len() / 2]
}

/// One packed weight's bytes, generated rather than quantized: a deterministic
/// ramp with the f16 scale fields pinned to a finite exponent, the shape
/// `rlx-wgpu/tests/common/mod.rs::packed_gguf_weight` uses. Only the timing is
/// read, so the values need only be finite.
pub fn packed_ramp(scheme: QuantScheme, len: usize) -> Vec<u8> {
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

/// Every input the graph declares, at its declared shape, its `i`-th element from
/// `value` — the timing does not read the inputs, but a denormal or a NaN would
/// change what the arithmetic costs.
fn inputs_with(graph: &Graph, value: impl Fn(usize) -> f32) -> Vec<(String, Vec<f32>)> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Input { name } => Some((
                name.clone(),
                (0..node.shape.num_elements().expect("a static input shape"))
                    .map(&value)
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

/// [`inputs_with`] at one value.
pub fn filled_inputs(graph: &Graph, fill: f32) -> Vec<(String, Vec<f32>)> {
    inputs_with(graph, |_| fill)
}

/// [`filled_inputs`] at zero, ready for one run of the whole frame.
pub fn zero_inputs(graph: &Graph) -> Vec<(String, Vec<f32>)> {
    filled_inputs(graph, 0.0)
}

/// Refuse to pin a submission/wait shape under the run-report switch: it carries
/// a timestamp write on its own passes and adds a fence to a dispatch-only run, so
/// the counts beside it are the instrument's, not the run's.
pub fn assert_run_report_off() {
    assert!(
        !rlx_ir::env::flag("RLX_WGPU_DBG_COUNTERS"),
        "RLX_WGPU_DBG_COUNTERS changes the submission/wait shape of a forward pass; \
         unset it to run this pin"
    );
}

/// The switches that stand a speech-frame subject down, each with its shipped
/// value: a compile-time stand-down is `0` (off), a run-time one its on value. Each
/// frame compile — the predictor arm's and the body's — states all of them, and the
/// harness installs the same list around each arm's runs, so neither an ambient
/// value nor a run-time switch can make an arm silently differ from the one it is
/// labelled as.
pub const FRAME_SWITCHES: [(&str, &str); 10] = [
    ("RLX_WGPU_CONCAT_HOST", "0"),
    ("RLX_WGPU_GEMV_IN_PLACE", "1"),
    ("RLX_WGPU_GEMV_POOL", "1"),
    ("RLX_WGPU_ATTN_DENSE_ONE_ROW", "0"),
    ("RLX_WGPU_NO_ATTN_MASK_SKIP", "0"),
    ("RLX_WGPU_NO_FUSED_RESIDUAL_RMS", "0"),
    ("RLX_WGPU_NO_FUSED_SWIGLU", "0"),
    ("RLX_WGPU_NO_GATHER_INDEX_BIAS", "0"),
    ("RLX_WGPU_NO_LN_TEE", "0"),
    ("RLX_WGPU_NO_NORM_ROPE", "0"),
];

/// The predictor frame compiled with every [`FRAME_SWITCHES`] entry at its shipped
/// value, then `set` over it — the arm is named exactly by the pairs it passes.
/// The returned pairs are what [`arm_once`] installs around the arm's runs.
pub fn frame_arm(
    pair: &Pair,
    capacity: usize,
    sampler: graphs::SampleSpec,
    set: &[(&'static str, &'static str)],
) -> (graphs::Compiled, Vec<(&'static str, &'static str)>) {
    let mut values = FRAME_SWITCHES.to_vec();
    for (name, value) in set {
        let slot = values
            .iter_mut()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not a frame switch"));
        slot.1 = value;
    }
    let _switch = rlx_ir::env::RuntimeOverrides::install(values.clone());
    let g = graphs::tts_code_predictor_frame(pair.sound.clone(), capacity, sampler)
        .expect("build the frame graph");
    (
        graphs::Compiled::new(Device::Gpu, g).expect("compile the frame"),
        values,
    )
}

/// One run of a compiled frame, in milliseconds, with the arm's own switch values
/// installed — a run-time switch is read at the dispatch, not at the compile.
pub fn arm_once(
    compiled: &mut graphs::Compiled,
    switches: &[(&str, &str)],
    inputs: &[(String, Vec<f32>)],
) -> f64 {
    let _switch = rlx_ir::env::RuntimeOverrides::install(switches.iter().copied());
    frame_once(compiled, inputs)
}

/// Every output of one run of an arm, switches installed — the value half of a
/// stand-down's evidence.
pub fn arm_outputs(
    compiled: &mut graphs::Compiled,
    switches: &[(&str, &str)],
    inputs: &[(String, Vec<f32>)],
) -> Vec<Vec<f32>> {
    let _switch = rlx_ir::env::RuntimeOverrides::install(switches.iter().copied());
    compiled.runner.run(inputs)
}

/// The harnesses' deterministic non-zero ramp, one row: a value check on all-zero
/// inputs would compare two draws off a degenerate distribution, and a denormal or
/// a NaN would change what the arithmetic costs.
pub fn ramped_row(len: usize) -> Vec<f32> {
    (0..len).map(ramped_at).collect()
}

fn ramped_at(i: usize) -> f32 {
    ((i as f32) * 0.0007 - 0.31).sin()
}

/// [`filled_inputs`] at [`ramped_row`]'s ramp.
pub fn ramped_inputs(graph: &Graph) -> Vec<(String, Vec<f32>)> {
    inputs_with(graph, ramped_at)
}

/// One run of a compiled frame, in milliseconds.
fn frame_once(compiled: &mut graphs::Compiled, inputs: &[(String, Vec<f32>)]) -> f64 {
    let start = Tick::now();
    compiled.runner.run(inputs);
    Tick::now().elapsed_ns(start) as f64 / 1e6
}

/// A schedule's `Step::Kind:count+…` census, one entry per step kind the schedule
/// names, so an arm's printed census says what it changed.
pub fn step_census(names: &[&'static str]) -> String {
    let mut kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for n in names {
        *kinds.entry(*n).or_default() += 1;
    }
    kinds
        .iter()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join("+")
}

/// Packed bytes one harness call streams, so a call is long enough to leave its own
/// submission behind.
pub const TARGET_BYTES: u64 = 96 * 1024 * 1024;
/// A ceiling on a call's copies, so a small shape does not compile a graph of
/// thousands of steps.
const MAX_STEPS: usize = 256;

/// The contraction or output width of one tensor.
pub fn last_dim(shape: &Shape) -> usize {
    shape
        .dims()
        .last()
        .expect("a multiply has a rank")
        .unwrap_static()
}

/// `steps` copies of one packed multiply over `rows` rows: the compiled call and its
/// input, ready to be timed.
pub struct Call {
    compiled: rlx_runtime::CompiledGraph,
    input: Vec<f32>,
}

/// One call of `steps` copies of the same packed multiply over `rows` rows. Its
/// outputs are checked once against `steps`, `rows` and `n` when it is built, so a
/// timing never rests on a call that wrote nothing.
pub fn call_of(
    scheme: QuantScheme,
    k: usize,
    n: usize,
    rows: usize,
    bytes: &[u8],
    steps: usize,
) -> Call {
    let len = bytes.len();
    let mut g = Graph::new(format!("harness_{scheme}_{k}x{n}_{rows}x{steps}"));
    let x = g.input("x", Shape::new(&[rows, k], DType::F32));
    let mut outs = Vec::with_capacity(steps);
    for step in 0..steps {
        let w = g.param(format!("w{step}"), Shape::new(&[len], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[rows, n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    let mut compiled = Session::new(Device::Gpu).compile(g);
    for step in 0..steps {
        compiled.set_param_typed(&format!("w{step}"), bytes, DType::U8);
    }
    let input: Vec<f32> = (0..rows * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let mut call = Call { compiled, input };
    let out = call.run();
    assert_eq!(out.len(), steps, "one output per priced multiply");
    assert!(
        out.iter()
            .all(|row| row.len() == rows * n && row.iter().all(|v| v.is_finite())),
        "harness {scheme} k={k} n={n} rows={rows}: every output row is finite"
    );
    call
}

/// `steps` that stream about [`TARGET_BYTES`].
pub fn steps_for(packed: u64, target: u64) -> usize {
    ((target / packed.max(1)) as usize).clamp(1, MAX_STEPS)
}

impl Call {
    pub fn run(&mut self) -> Vec<Vec<f32>> {
        self.compiled.run(&[("x", self.input.as_slice())])
    }
    pub fn ms(&mut self) -> f64 {
        let start = Tick::now();
        let _ = self.run();
        Tick::now().elapsed_ns(start) as f64 / 1e6
    }
}
