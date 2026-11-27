// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The decode attention entry's two bodies, priced against each other and held
//! to the same answers.
//!
//! **The table is measurement only, `#[ignore]`d — it prints, it never asserts a
//! time.** Run it with `--ignored --nocapture`.
//!
//! The measured shape is the product's: one query row over a cached key/value
//! sequence, 16 heads over 8 KV heads at width 128, `LAYERS` attention steps
//! chained through one query — the body's whole attention chain, so the reading
//! is per layer rather than per dispatch.
//!
//! # Why two arms in one session
//!
//! A lone absolute reading of this chain moves by more than the difference
//! between the bodies: the machine's own load, the process's warm-up and the
//! device's clock all land on it. So every cell here is a **paired** take — both
//! bodies built from the same graph definition in the same process, the arm that
//! runs first alternating by round, and the statistic read is the median of the
//! per-round differences ([`common::paired_round_robin`]). The arms differ in
//! nothing but the `RLX_WGPU_ATTN_DECODE_ONLINE` value each run carries — the body
//! is a per-run pick, not a compile-time one — so the pairing cancels the per-run
//! cost both pay.
//!
//! The extents are the ones the product's decode steps walk: 138 keys is the
//! synthesis talker's cache bucket, 193 a recognition rung's plane, and 363 a
//! clip's plane past its prompt. 64 and 96 sit below the weighted body's key-range
//! split floor, and 152/160/176/192 straddle it, so the table shows both forms and
//! where the run switches between them; 512 and 2 438 keep a long context — and the
//! weighted body's tile boundaries — in view. Each row prints the split depth the
//! run takes.
mod common;

use common::{data, max_abs_diff, median, paired_round_robin, refuse_under_run_report, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::backend::attention_decode_key_splits;

/// Head count, KV head count and width of the 1.7B-class decode body.
const NH: usize = 16;
const NKV: usize = 8;
const HD: usize = 128;
/// A body's layer count: the chain is measured whole and divided by it.
const LAYERS: usize = 28;
/// Paired rounds and calls per arm per round. Odd, so the median is a sample.
const ROUNDS: usize = 15;
const CALLS: usize = 2;
/// The switch the one-pass body answers to.
const ONLINE: &str = "RLX_WGPU_ATTN_DECODE_ONLINE";

/// The extents the table walks, with the product step each one is.
const EXTENTS: [(usize, &str); 11] = [
    (64, "below the split floor"),
    (96, "below the split floor"),
    (138, "talker bucket"),
    (152, "just below the floor"),
    (160, "at the split floor"),
    (176, "over the split floor"),
    (192, "over the split floor"),
    (193, "recognition rung"),
    (363, "clip plane"),
    (512, "long context"),
    (2438, "digest shape"),
];

/// `layers` decode-shaped attention steps chained through the query, reading one
/// K/V plane with the binary key mask the plane cache carries.
fn chain(lk: usize, layers: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_step");
    let mut q = g.input("q", Shape::new(&[1, 1, NH, HD], f));
    let k = g.input("k", Shape::new(&[1, lk, NKV, HD], f));
    let v = g.input("v", Shape::new(&[1, lk, NKV, HD], f));
    let m = g.input("m", Shape::new(&[1, lk], f));
    for _ in 0..layers {
        q = g.add_node(
            Op::Attention {
                num_heads: NH,
                head_dim: HD,
                v_head_dim: None,
                mask_kind: MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, m],
            Shape::new(&[1, 1, NH, HD], f),
        );
    }
    g.set_outputs(vec![q]);
    g
}

/// The graph's operands: a ramp for Q/K/V and an all-ones key mask, so no key is
/// dropped and the whole walk is priced.
struct Operands {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    m: Vec<f32>,
}

impl Operands {
    fn new(lk: usize) -> Self {
        Self {
            q: data(NH * HD, 0.011, false),
            k: data(lk * NKV * HD, 0.013, true),
            v: data(lk * NKV * HD, 0.017, false),
            m: vec![1.0; lk],
        }
    }

    fn inputs(&self) -> Vec<(&str, &[f32])> {
        vec![
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("m", &self.m),
        ]
    }
}

/// One arm of the cost table: a compiled graph and the switch value its runs carry.
/// The body is a per-run pick, so the graph is not bound to either.
type Arm = (CompiledGraph, &'static str);

/// Run one body on a compiled chain — the switch installed for the call and
/// restored on the way out, panic included — and return its output.
fn run_form(graph: &mut CompiledGraph, form: &str, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let _form = rlx_ir::env::RuntimeOverrides::install([(ONLINE, form)]);
    graph.run(inputs).remove(0)
}

/// The two bodies of `chain(lk, LAYERS)`, compiled and warmed in one process.
fn arms(lk: usize, inputs: &[(&str, &[f32])]) -> (Arm, Arm) {
    let mut weighted: Arm = (Session::new(Device::Gpu).compile(chain(lk, LAYERS)), "0");
    let mut online: Arm = (Session::new(Device::Gpu).compile(chain(lk, LAYERS)), "1");
    // Warm both, so neither pays the other's first-run cost inside the take.
    for _ in 0..3 {
        run_form(&mut weighted.0, weighted.1, inputs);
        run_form(&mut online.0, online.1, inputs);
    }
    (weighted, online)
}

#[test]
#[ignore = "measurement: prints a cost table, asserts no duration"]
fn the_two_bodies_cost() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    refuse_under_run_report();
    let _guard = route_lock();

    for (lk, what) in EXTENTS {
        let operands = Operands::new(lk);
        let inputs = operands.inputs();
        let (mut weighted, mut online) = arms(lk, &inputs);
        let (own, deltas) = paired_round_robin(
            &mut weighted,
            &mut online,
            ROUNDS,
            CALLS,
            |arm: &mut Arm| {
                run_form(&mut arm.0, arm.1, &inputs);
            },
        );
        let layers = LAYERS as f64;
        let a = median(&own[0]) * 1e3 / layers;
        let b = median(&own[1]) * 1e3 / layers;
        let paired = median(&deltas) * 1e3 / layers;
        let splits = attention_decode_key_splits(1, lk as u32, NH as u32, HD as u32, HD as u32);
        eprintln!(
            "[attn-form] lk={lk:5} ({what:22}) splits={splits} weighted={a:9.2} us/layer \
             one-pass={b:9.2} us/layer  paired={paired:+8.2} us/layer ({:+.1}%)",
            paired / b * 100.0
        );
    }
}

#[test]
fn the_two_bodies_agree() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let mut digests = Vec::new();
    for (lk, what) in EXTENTS {
        let operands = Operands::new(lk);
        let inputs = operands.inputs();
        // One compiled graph runs both arms: the body is a per-run pick, so the
        // parity needs no second compile (and the gate does not pay for one).
        let mut graph = Session::new(Device::Gpu).compile(chain(lk, LAYERS));
        let a = run_form(&mut graph, "0", &inputs);
        let b = run_form(&mut graph, "1", &inputs);
        let d = max_abs_diff(&a, &b);
        let (da, db) = (digest(&a), digest(&b));
        eprintln!(
            "[attn-form-parity] lk={lk:5} ({what:22}) weighted={da:016x} one-pass={db:016x} \
             max_abs={d:.3e}"
        );
        // Both bodies compute the same softmax; they combine its partial sums in
        // a different order (per value dim against per key), which is the
        // reassociation a reduction is allowed. Anything larger is a defect.
        assert!(
            d < 1e-6,
            "decode bodies disagree at lk={lk}: max_abs {d:.3e}"
        );
        digests.push((lk, da, db));
    }
    // The switch's own witness: at the long extent the reassociation is visible
    // in the bits, so a run where the two arms came out identical means the arm
    // switch did not take.
    let (_, da, db) = digests[digests.len() - 1];
    assert_ne!(
        da,
        db,
        "the two decode bodies produced the same bits at {}: the arm switch did not take",
        EXTENTS[EXTENTS.len() - 1].0
    );
}

/// FNV-1a 64 over the output's f32 bytes, the digest form the fork's witnesses
/// use.
fn digest(values: &[f32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}
