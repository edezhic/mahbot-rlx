// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The dense square attention route's cooperative form against the CPU
//! reference and against the band kernel it replaces.
//!
//! What this file pins: that a `MaskKind::None`, square, 8 192-row step at the
//! encoder's global-layer geometry (4 heads of width 512 over one KV head) takes
//! the cooperative form — two `matmul_coop_f16_f32acc` dispatches per chunk, not
//! one band pass and not the f32 many-row multiplies — and lands inside the
//! tolerance that form's 16-bit operands set, against both the CPU reference and
//! the band kernel's own answer at the same shape.
//!
//! The two arms are held to the f16 tile's bar (`5e-3` max-abs here, where the
//! band parity file's own dense cell is held to the same figure) plus a cosine,
//! so a case whose output magnitude shrinks cannot hide a shape error behind a
//! small absolute deviation. What it does *not* claim: that the new form is
//! closer to the reference than the band kernel. Its operands are f16 for Q, K, V
//! and P, against the band kernel's f16 staging of the same operands, and the
//! printed pair is the measured difference of each arm — not a bound on either.
//!
//! The dispatch counter is process-global, so the cases that bracket it take
//! [`route_lock`]. Timings are printed, never asserted.
mod common;

use common::{data, max_abs_diff, refuse_under_run_report, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// The bar the two arms are held to: the f16 operand rounding, which is the
/// dominant term in both.
const BAR: f32 = 5e-3;

/// Cosine similarity of two equal-length vectors — the second number beside
/// max-abs, so a case whose output magnitude shrinks cannot hide a shape error
/// behind a small absolute deviation.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine needs equal lengths");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Rank-4 `[1, rows, H, D]` Q over `[1, rows, KV, D]` K/V, one `Op::Attention`
/// of the given mask kind — the stand's encoder layout.
fn attention_graph(rows: usize, nh: usize, nkv: usize, hd: usize, kind: MaskKind) -> Graph {
    attention_graph_scaled(rows, nh, nkv, hd, kind, None, None)
}

/// The same graph with the op's optional `score_scale` and `attn_logit_softcap`.
fn attention_graph_scaled(
    rows: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    kind: MaskKind,
    score_scale: Option<f32>,
    softcap: Option<f32>,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attention_dense_coop");
    let q = g.input("q", Shape::new(&[1, rows, nh, hd], f));
    let k = g.input("k", Shape::new(&[1, rows, nkv, hd], f));
    let v = g.input("v", Shape::new(&[1, rows, nkv, hd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale,
            attn_logit_softcap: softcap,
        },
        vec![q, k, v],
        Shape::new(&[1, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// The shapes the encoder's four global layers present, and the smallest one the
/// form is admitted at — both a whole number of cooperative tiles.
const CASES: &[(usize, &str)] = &[(8192, "the encoder's global layers"), (2048, "the floor")];

/// One case: the GPU's cooperative answer against the CPU reference and against
/// the band kernel's answer at the same shape, with the dispatch count that says
/// which route the step took.
fn run_case(rows: usize, label: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (nh, nkv, hd) = (4usize, 1usize, 512usize);
    let q = data(rows * nh * hd, 0.0005, false);
    let k = data(rows * nkv * hd, 0.0007, true);
    let v = data(rows * nkv * hd, 0.0004, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let _gate = route_lock();
    let mut c =
        Session::new(Device::Gpu).compile(attention_graph(rows, nh, nkv, hd, MaskKind::None));
    let _ = c.run(&inputs).remove(0);
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let start = rlx_runtime::Tick::now();
    let gpu = c.run(&inputs).remove(0);
    let ms = rlx_runtime::Tick::now().elapsed_us(start) / 1000.0;
    let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
    let chunk = rlx_wgpu::backend::DENSE_COOP_CHUNK_ROWS.min(rows as u32) as u64;
    let chunks = (rows as u64).div_ceil(chunk);
    eprintln!("{label} rows={rows}: gpu {ms:.2} ms, {disps} cooperative dispatches");
    assert_eq!(
        disps,
        2 * chunks,
        "rows={rows}: every chunk runs two cooperative multiplies ({} chunks), so the \
         step did not take the dense cooperative form",
        chunks
    );
    let cpu = Session::new(Device::Cpu)
        .compile(attention_graph(rows, nh, nkv, hd, MaskKind::None))
        .run(&inputs)
        .remove(0);
    // The band kernel at the same shape: a square window of the whole sequence
    // is its no-band path, which is what the dense step's cells were served by
    // before this route.
    let band = Session::new(Device::Gpu)
        .compile(attention_graph(rows, nh, nkv, hd, MaskKind::Window(rows)))
        .run(&inputs)
        .remove(0);
    (gpu, cpu, band)
}

#[test]
fn the_dense_square_form_matches_the_cpu_and_the_band_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    for &(rows, label) in CASES {
        let (gpu, cpu, band) = run_case(rows, label);
        let d_cpu = max_abs_diff(&gpu, &cpu);
        let cos_cpu = cosine(&gpu, &cpu);
        let d_band = max_abs_diff(&gpu, &band);
        let cos_band = cosine(&gpu, &band);
        let d_band_cpu = max_abs_diff(&band, &cpu);
        let cos_band_cpu = cosine(&band, &cpu);
        eprintln!(
            "  rows={rows}: vs CPU max_abs {d_cpu:.3e} cosine {cos_cpu:.9}; \
             vs band max_abs {d_band:.3e} cosine {cos_band:.9}; \
             band vs CPU max_abs {d_band_cpu:.3e} cosine {cos_band_cpu:.9}"
        );
        assert!(
            d_cpu < BAR,
            "rows={rows}: max-abs {d_cpu:.3e} against the CPU is over the {BAR:.1e} bar"
        );
        assert!(
            cos_cpu > 0.9999,
            "rows={rows}: cosine {cos_cpu:.9} against the CPU is short of the bar"
        );
        assert!(
            d_band < BAR,
            "rows={rows}: max-abs {d_band:.3e} against the band kernel is over the {BAR:.1e} bar"
        );
        assert!(
            cos_band > 0.9999,
            "rows={rows}: cosine {cos_band:.9} against the band kernel is short of the bar"
        );
    }
}

/// The form is admitted from [`rlx_wgpu::backend::DENSE_COOP_MIN_Q`] and not a
/// row below it: the band kernel stays what a shorter dense square step takes
/// where the many-row arrangement's scratch no longer fits. Device-free except
/// for the session it compiles, which is the point — the schedule is the fact.
#[test]
fn a_dense_square_step_below_the_floor_keeps_the_band_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let _gate = route_lock();
    let (rows, nh, nkv, hd) = (1536usize, 64usize, 64usize, 64usize);
    // The band parity file's own dense cell: 64 heads put the 512 MiB cap at
    // 1 449 rows, so 1 536 clears it and the step is the band kernel's.
    let inputs = (
        data(rows * nh * hd, 0.0007, false),
        data(rows * nkv * hd, 0.0009, true),
        data(rows * nkv * hd, 0.0005, true),
    );
    let inputs: [(&str, &[f32]); 3] = [("q", &inputs.0), ("k", &inputs.1), ("v", &inputs.2)];
    let mut c =
        Session::new(Device::Gpu).compile(attention_graph(rows, nh, nkv, hd, MaskKind::None));
    let _ = c.run(&inputs).remove(0);
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let routes_before = rlx_wgpu::kernels::attention_route_counts();
    let _ = c.run(&inputs).remove(0);
    let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
    let routes = rlx_wgpu::kernels::attention_route_counts();
    eprintln!("below the floor: rows={rows} dispatches {disps}, routes {routes:?}");
    assert_eq!(
        disps, 0,
        "a step below the floor must not take the cooperative form"
    );
    assert_eq!(
        routes.band,
        routes_before.band + 1,
        "and it keeps the band kernel's route"
    );
    assert_eq!(routes.decomposed, routes_before.decomposed);
}

/// The two switches that keep the cooperative form off a shape it admits: this
/// form's own decline lever (`RLX_WGPU_NO_DENSE_COOP`) and the crate's absolute
/// f32 pin (`RLX_WGPU_MATMUL_F32_ONLY`), which the conv routes honour the same
/// way. Each must leave the answer the cooperative route gave — the band kernel
/// is what the plan admits underneath it — and each must be visible on
/// [`WgpuExecutable::attention_dense_coop_declines`], which this file and
/// `tests/attention_rows_route.rs`'s band-decline pin are the only readers of.
///
/// The counter comes from a compile of the same graph, dropped before the session
/// that produces the numbers is built, so the two arms do not hold the device's
/// buffers at once.
#[test]
fn the_decline_levers_keep_the_answer_and_count_themselves() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let (rows, nh, nkv, hd) = (2048usize, 4usize, 1usize, 512usize);
    let q = data(rows * nh * hd, 0.0005, false);
    let k = data(rows * nkv * hd, 0.0007, true);
    let v = data(rows * nkv * hd, 0.0004, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let _gate = route_lock();
    let cpu = Session::new(Device::Cpu)
        .compile(attention_graph(rows, nh, nkv, hd, MaskKind::None))
        .run(&inputs)
        .remove(0);
    for switch in ["RLX_WGPU_NO_DENSE_COOP", "RLX_WGPU_MATMUL_F32_ONLY"] {
        // A guard rather than a set/unset pair: a failing assertion below would
        // otherwise leak the switch into the sibling tests of this binary.
        let _lever = rlx_ir::env::RuntimeOverrides::install(Some((switch, "1")));
        let declines = WgpuExecutable::compile(attention_graph(rows, nh, nkv, hd, MaskKind::None))
            .attention_dense_coop_declines();
        let gpu = Session::new(Device::Gpu)
            .compile(attention_graph(rows, nh, nkv, hd, MaskKind::None))
            .run(&inputs)
            .remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        let cos = cosine(&gpu, &cpu);
        eprintln!(
            "{switch}=1: {declines} cooperative decline(s), max_abs {d:.3e}, cosine {cos:.9}"
        );
        assert!(
            declines >= 1,
            "{switch}: the switch must keep the step off the cooperative form"
        );
        assert!(
            d < BAR,
            "{switch}: max-abs {d:.3e} against the CPU is over the {BAR:.1e} bar"
        );
        assert!(
            cos > 0.9999,
            "{switch}: cosine {cos:.9} against the CPU is short of the bar"
        );
    }
}

/// The route spends no pass on the masked scale for a plain `MaskKind::None`
/// step: the scale is folded into the Kᵀ f16 cast, which is the pass that already
/// walks that operand, so the schedule carries the two casts, eight multiplies and
/// four softmaxes — and no `attention_mask` step. A step with a softcap keeps that
/// step, because `c·tanh(s/c)` is not a scale of an operand. Both are schedule
/// facts, so they are read off the compiled step names.
#[test]
fn the_dense_form_folds_the_scale_into_the_kt_cast() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let names = common::legalised_step_names(&attention_graph(8192, 4, 1, 512, MaskKind::None));
    let count = |what: &str| names.iter().filter(|n| **n == what).count();
    eprintln!("plain: {} steps, {names:?}", names.len());
    assert_eq!(
        count("attention_mask"),
        0,
        "the masked scale is folded into the Kᵀ cast, not spent on a pass"
    );
    assert_eq!(
        count("cast_f32_to_f16"),
        2,
        "Kᵀ and V′ are materialised as f16"
    );
    assert_eq!(count("matmul"), 8, "two multiplies a chunk, four chunks");
    assert_eq!(count("softmax"), 4);

    // The same shape with a logit softcap keeps the masked-scale step.
    let f = DType::F32;
    let mut g = Graph::new("attention_dense_coop_softcap");
    let q = g.input("q", Shape::new(&[1, 8192, 4, 512], f));
    let k = g.input("k", Shape::new(&[1, 8192, 1, 512], f));
    let v = g.input("v", Shape::new(&[1, 8192, 1, 512], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: 4,
            head_dim: 512,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: Some(50.0),
        },
        vec![q, k, v],
        Shape::new(&[1, 8192, 4, 512], f),
    );
    g.set_outputs(vec![y]);
    let names = common::legalised_step_names(&g);
    eprintln!("softcap: {} steps, {names:?}", names.len());
    assert_eq!(
        names.iter().filter(|n| **n == "attention_mask").count(),
        4,
        "a softcapped step keeps one masked-scale pass a chunk"
    );
}

/// A logit softcap on the same form: the cap is `c·tanh(s/c)` over the scaled
/// score, which is not a scale of an operand — so the route keeps its masked-scale
/// step (`the_dense_form_folds_the_scale_into_the_kt_cast`) and this pins that the
/// step still means what it did on the f32 arrangement. A non-default
/// `score_scale` rides the same step rather than the Kᵀ cast.
#[test]
fn a_softcapped_dense_step_keeps_its_masked_scale_and_matches_the_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let (rows, nh, nkv, hd) = (2048usize, 4usize, 1usize, 512usize);
    let q = data(rows * nh * hd, 0.0005, false);
    let k = data(rows * nkv * hd, 0.0007, true);
    let v = data(rows * nkv * hd, 0.0004, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let _gate = route_lock();
    let build = |scale: Option<f32>, cap: Option<f32>| {
        attention_graph_scaled(rows, nh, nkv, hd, MaskKind::None, scale, cap)
    };
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let gpu = Session::new(Device::Gpu)
        .compile(build(Some(0.031), Some(50.0)))
        .run(&inputs)
        .remove(0);
    let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
    let cpu = Session::new(Device::Cpu)
        .compile(build(Some(0.031), Some(50.0)))
        .run(&inputs)
        .remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    let cos = cosine(&gpu, &cpu);
    eprintln!(
        "softcapped dense step: {disps} cooperative dispatches, max_abs {d:.3e}, cosine {cos:.9}"
    );
    assert_eq!(
        disps, 2,
        "the capped step still runs both multiplies on the form"
    );
    assert!(d < BAR, "max-abs {d:.3e} is over the {BAR:.1e} bar");
    assert!(cos > 0.9999, "cosine {cos:.9} is short of the bar");
}
