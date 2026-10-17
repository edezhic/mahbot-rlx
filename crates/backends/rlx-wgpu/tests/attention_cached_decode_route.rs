// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fork's cached decode cell, pinned on the device: **two query rows over a
//! full key cache**, on the recognition body's own attention geometry (16 query
//! heads over 8 KV heads, head width 128, one batch, causal), with the cache and
//! the window concatenated exactly as the body hands them to the step.
//!
//! That cell is why the many-row route's query-row floor sits at two
//! (`backend::ATTENTION_DECOMPOSE_MIN_Q`); the readings behind it — their
//! provenance and what the move does not pay back — are in that module's note.
//! What this file pins is the arrangement and the parity, not a time: the step must
//! take the route and reach the CPU backend inside `common::ATTENTION_BAR`, and the
//! graph must report the active-extent veto that comes with it. The single-row cell
//! over the same cache is pinned next to it, because that row's decode kernel is the
//! cheaper arrangement the change must leave alone; the padded run the band also
//! lands in — compiled above the walked extent — is pinned for the live rows'
//! answers, its time being unmeasured here.
//!
//! Four of the body's 28 blocks: the route decision and the parity are per
//! attention step, and the whole-body figure is the stand's reading rather than a
//! test's.
mod common;

use common::{ATTENTION_BAR, data, max_abs_diff, route_lock};
use rlx_ir::op::{BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

/// The body's geometry (Qwen3-ASR-0.6B): 16 query heads, 8 KV heads, width 128,
/// one batch.
const NH: usize = 16;
const NKV: usize = 8;
const HD: usize = 128;
const DIM: usize = NH * HD;
const KV_DIM: usize = NKV * HD;
/// The cached body's cell: 512 cells in the cache, and the two rows a generation
/// step walks.
const CACHE: usize = 512;
const ROWS: usize = 2;
/// The body's blocks this file walks — see the module note.
const LAYERS: usize = 4;

/// `layers` cached decode blocks over one cache pair and one window: each block
/// narrows its own row of the cache, concatenates the window's keys and values
/// onto it, and attends causally over `[cache ; window]` — the construction the
/// cached body runs. The blocks are summed so the graph has one output; the sum is
/// a device step like any other and moves nothing in the route decision.
fn cached_body(rows: usize, layers: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("cached_decode_route");
    let q = g.input("q", Shape::new(&[1, rows, DIM], f));
    let kpre = g.input("kpre", Shape::new(&[layers, CACHE, KV_DIM], f));
    let vpre = g.input("vpre", Shape::new(&[layers, CACHE, KV_DIM], f));
    let kwin = g.input("kwin", Shape::new(&[1, rows, KV_DIM], f));
    let vwin = g.input("vwin", Shape::new(&[1, rows, KV_DIM], f));

    // `[1, seq, H, D] → [1, H, seq, D]`, the body's head-major operand.
    let head_major = |g: &mut Graph, x: rlx_ir::NodeId, seq: usize, heads: usize| {
        let view = g.add_node(
            Op::Reshape {
                new_shape: vec![1, seq as i64, heads as i64, HD as i64],
            },
            vec![x],
            Shape::new(&[1, seq, heads, HD], f),
        );
        g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![view],
            Shape::new(&[1, heads, seq, HD], f),
        )
    };
    let cache_row = |g: &mut Graph, x: rlx_ir::NodeId, layer: usize| {
        let row = g.add_node(
            Op::Narrow {
                axis: 0,
                start: layer,
                len: 1,
            },
            vec![x],
            Shape::new(&[1, CACHE, KV_DIM], f),
        );
        head_major(g, row, CACHE, NKV)
    };
    let window = |g: &mut Graph, x: rlx_ir::NodeId| head_major(g, x, rows, NKV);
    let q4 = head_major(&mut g, q, rows, NH);

    let mut acc: Option<rlx_ir::NodeId> = None;
    for layer in 0..layers {
        let (kp, kw) = (cache_row(&mut g, kpre, layer), window(&mut g, kwin));
        let (vp, vw) = (cache_row(&mut g, vpre, layer), window(&mut g, vwin));
        let k = g.add_node(
            Op::Concat { axis: 2 },
            vec![kp, kw],
            Shape::new(&[1, NKV, CACHE + rows, HD], f),
        );
        let v = g.add_node(
            Op::Concat { axis: 2 },
            vec![vp, vw],
            Shape::new(&[1, NKV, CACHE + rows, HD], f),
        );
        let a = g.add_node(
            Op::Attention {
                num_heads: NH,
                head_dim: HD,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q4, k, v],
            Shape::new(&[1, NH, rows, HD], f),
        );
        // Back to `[1, rows, H·D]`, the body's output shape.
        let t = g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![a],
            Shape::new(&[1, rows, NH, HD], f),
        );
        let out = g.add_node(
            Op::Reshape {
                new_shape: vec![1, rows as i64, DIM as i64],
            },
            vec![t],
            Shape::new(&[1, rows, DIM], f),
        );
        acc = Some(match acc {
            Some(prev) => g.binary(BinaryOp::Add, prev, out, Shape::new(&[1, rows, DIM], f)),
            None => out,
        });
    }
    g.set_outputs(vec![acc.expect("at least one block")]);
    g
}

/// The body's inputs at `rows` window rows and `layers` cache rows, one
/// deterministic stream each.
fn body_inputs(rows: usize, layers: usize) -> [(&'static str, Vec<f32>); 5] {
    [
        ("q", data(rows * DIM, 0.011, false)),
        ("kpre", data(layers * CACHE * KV_DIM, 0.013, true)),
        ("vpre", data(layers * CACHE * KV_DIM, 0.017, false)),
        ("kwin", data(rows * KV_DIM, 0.019, true)),
        ("vwin", data(rows * KV_DIM, 0.023, false)),
    ]
}

/// The named slices `run` takes, borrowed from [`body_inputs`]' own buffers.
fn as_refs<'a>(xs: &'a [(&'static str, Vec<f32>)]) -> Vec<(&'static str, &'a [f32])> {
    xs.iter().map(|(name, v)| (*name, v.as_slice())).collect()
}

/// Run both devices at the body's own data and return the routes one run took and
/// the max-abs against the CPU.
fn run_both(g: Graph, rows: usize, layers: usize) -> (AttentionRoutes, f32) {
    let xs = body_inputs(rows, layers);
    let inputs = as_refs(&xs);
    let cpu = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&inputs)
        .remove(0);
    let mut gpu_run = Session::new(Device::Gpu).compile(g);
    let _ = gpu_run.run(&inputs); // warm the pipelines, then bracket the counters
    reset_attention_route_counts();
    let gpu = gpu_run.run(&inputs).remove(0);
    let routes = attention_route_counts();
    (routes, max_abs_diff(&gpu, &cpu))
}

/// The cell itself: two query rows, both carrying work, over a full 512-cell
/// cache, must take the dense arrangement — and with it the whole-body saving the
/// module note records — while staying on the CPU reference's numbers.
#[test]
fn a_two_row_cached_decode_step_takes_the_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let (routes, d) = run_both(cached_body(ROWS, LAYERS), ROWS, LAYERS);
    eprintln!(
        "[rlx-cached-decode] rows={ROWS} cache={CACHE} layers={LAYERS} heads={NH}/{NKV} \
         head_dim={HD} routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: LAYERS as u64,
            serial: 0,
        },
        "the cached two-row step must take the dense arrangement"
    );
    assert!(
        d < ATTENTION_BAR,
        "cached two-row decode: max_abs {d} vs CPU"
    );
}

/// The arrangement that must not move: one query row over the same cache stays on
/// the decode kernel, which is cheaper than both the fused walk and this route.
#[test]
fn a_one_row_cached_decode_step_keeps_the_decode_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let (routes, d) = run_both(cached_body(1, 1), 1, 1);
    eprintln!(
        "[rlx-cached-decode] rows=1 cache={CACHE} layers=1 heads={NH}/{NKV} head_dim={HD} \
         routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 1,
            decomposed: 0,
            serial: 0,
        },
        "one query row is the decode kernel's, and the route must not take it"
    );
    assert!(
        d < ATTENTION_BAR,
        "cached one-row decode: max_abs {d} vs CPU"
    );
}

/// What the route costs the graph besides time: its masked scale answers `false` to
/// `safe_for_active_extent`, so a graph containing one is walked at its compiled
/// extent, and a caller that compiles at a larger extent and patches down loses that
/// mode for the whole graph. Pinned at 48 rows and a patched key extent in
/// `attention_rows_route`'s
/// `the_route_keeps_its_planned_extent_under_an_active_extent_hint`; this is the
/// same property at the band's own extent, on the cached body, where the one-row
/// cell on the decode kernel keeps the mode alive.
#[test]
fn the_cached_step_vetoes_the_graphs_active_extent_mode() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let two = WgpuExecutable::compile(cached_body(ROWS, LAYERS));
    assert!(
        !two.test_all_steps_safe_for_active_extent(),
        "the dense arrangement's masked scale must veto the active-extent mode"
    );
    let one = WgpuExecutable::compile(cached_body(1, LAYERS));
    assert!(
        one.test_all_steps_safe_for_active_extent(),
        "the one-row cell on the decode kernel must keep the mode alive"
    );
}

/// The band's other re-routed class: a caller that compiles at a larger extent and
/// walks fewer live rows — the hint `CompileCache::run_padded` hands a backend as
/// `set_active_extent((actual, upper))`, i.e. a bucketed chunk. The dense
/// arrangement's masked scale vetoes that hint for the whole graph, so the step
/// computes at `UPPER`; what must not move is the answer of the rows that are live,
/// which is what this pins. Its cost is not asserted here: the veto makes the step
/// compute at `UPPER`, inside the rows-flat range the module note measures.
#[test]
fn a_padded_run_keeps_the_live_rows_answers() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const UPPER: usize = 8;
    let xs = body_inputs(UPPER, LAYERS);
    let inputs = as_refs(&xs);
    let mut padded_run = Session::new(Device::Gpu).compile(cached_body(UPPER, LAYERS));
    let _ = padded_run.run(&inputs).remove(0); // warm, then bracket the counters
    reset_attention_route_counts();
    padded_run.set_active_extent(Some((ROWS, UPPER)));
    let padded = padded_run.run(&inputs).remove(0);
    padded_run.set_active_extent(None);
    let routes = attention_route_counts();

    // The live rows are the first `ROWS` of the `[1, UPPER, H·D]` output, and their
    // reference is the same body at the live extent: the causal offset
    // `qi + (seq_k - seq_q)` puts row `qi` at the same key in both, so the two are
    // one computation over the same cache and the same window rows.
    let live_xs = body_inputs(ROWS, LAYERS);
    let live = as_refs(&live_xs);
    let cpu = Session::new(Device::Cpu)
        .compile(cached_body(ROWS, LAYERS))
        .run(&live)
        .remove(0);
    let d = max_abs_diff(&padded[..ROWS * DIM], &cpu);
    eprintln!(
        "[rlx-cached-decode] padded upper={UPPER} live={ROWS} cache={CACHE} layers={LAYERS} \
         routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: LAYERS as u64,
            serial: 0,
        },
        "the padded {UPPER}-row body must take the dense arrangement"
    );
    assert!(
        d < ATTENTION_BAR,
        "the live rows of a padded {UPPER}-row walk: max_abs {d} vs CPU"
    );
}
