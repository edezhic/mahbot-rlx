// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The many-row attention route on wgpu (`backend::attention_route`): the score
//! multiply, the operation's own masked normalise and the multiply by V, in
//! place of the fused kernel's per-thread walk of the key sequence.
//!
//! What this file pins on the device is that a many-row step takes that
//! arrangement and reaches the CPU reference, that a shape below its query-row
//! floor keeps the fused kernel and reaches it too, that a shape past its scratch
//! budget keeps it as well, and that all five mask kinds round-trip on the route.
//! The layouts the route's regions depend on are pinned alongside the masks: the
//! dense `[B, H, S, D]` tile order (where nothing needs compacting but Kᵀ), the
//! strided `[B, S, H, D]` one, grouped KV heads (whose head axis is replicated on
//! the way into Kᵀ and V′) and an asymmetric value width.
//!
//! One test compares the two arrangements directly. The route is chosen
//! from the step's own query-row count, so the pair is one per-(batch, head)
//! problem twice — the same prefill computed by the fused kernel and by the
//! dense arrangement — and under a causal mask with `seq_q == seq_k` the rows
//! they share attend to identical keys by construction. That pair is a
//! cross-route comparison and not bit identity: the fused form rebases its
//! softmax key by key and the dense form does it once against the row max, so the
//! two sum in different orders. The test asserts the crate's bar and prints the
//! measured deviation; it does not claim the outputs are equal.
//!
//! Several tests here bracket the process-global route counters, so they take
//! [`route_lock`] instead of living in one single-test binary. The timings are
//! printed, never asserted: what a regression moves first is the route.
mod common;

use common::{max_abs_diff, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::{ATTENTION_DECOMPOSE_MIN_Q, WgpuExecutable};
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

fn data(n: usize, step: f32, cos: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 * step;
            if cos { x.cos() } else { x.sin() }
        })
        .collect()
}

/// Rank-3 `[B, rows, heads·head_dim]` MHA over `lk` keys. `mask_kind` decides
/// whether the graph carries a mask input and of which shape: `[B, lk]` binary
/// key padding for [`MaskKind::Custom`], `[B, heads, rows, lk]` additive
/// per-head scores for [`MaskKind::Bias`] — the shapes the CPU reference reads.
fn chain(b: usize, rows: usize, lk: usize, nh: usize, hd: usize, kind: MaskKind) -> (Graph, bool) {
    let dim = nh * hd;
    let f = DType::F32;
    let mut g = Graph::new("attn_rows");
    let q = g.input("q", Shape::new(&[b, rows, dim], f));
    let k = g.input("k", Shape::new(&[b, lk, dim], f));
    let v = g.input("v", Shape::new(&[b, lk, dim], f));
    let carries_mask = matches!(kind, MaskKind::Custom | MaskKind::Bias);
    let mut inputs = vec![q, k, v];
    if carries_mask {
        let mask = g.input(
            "mask",
            match kind {
                MaskKind::Bias => Shape::new(&[b, nh, rows, lk], f),
                _ => Shape::new(&[b, lk], f),
            },
        );
        inputs.push(mask);
    }
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        inputs,
        Shape::new(&[b, rows, dim], f),
    );
    g.set_outputs(vec![y]);
    (g, carries_mask)
}

/// Rank-4 `[B, H, S, D]` MHA: the layout whose per-`(batch, head)` tiles *are*
/// dense, so Q, V and the output slot are read and written where they lie and
/// the route's only compaction is Kᵀ.
fn chain_bhsd(
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    kind: MaskKind,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_rows_bhsd");
    let q = g.input("q", Shape::new(&[b, nh, rows, hd], f));
    let k = g.input("k", Shape::new(&[b, nkv, lk, hd], f));
    let v = g.input("v", Shape::new(&[b, nkv, lk, hd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, nh, rows, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// Rank-4 `[B, S, H, D]` MHA (the CPU / EEG-DINO convention, told from
/// `[B, H, S, D]` by its second dim): every operand's per-`(batch, head)` tile is
/// strided — its `seq_stride` is `H·D`, not the head width — so the route
/// compacts Q, V and the output as well as Kᵀ. A caller must keep `rows != nh`,
/// or the second dim reads as the head axis.
fn chain_bshd(b: usize, rows: usize, lk: usize, nh: usize, hd: usize, kind: MaskKind) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_rows_bshd");
    let q = g.input("q", Shape::new(&[b, rows, nh, hd], f));
    let k = g.input("k", Shape::new(&[b, lk, nh, hd], f));
    let v = g.input("v", Shape::new(&[b, lk, nh, hd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// Rank-3 `[B, S, H·head_dim]` Q/K with `[B, S, H·v_head_dim]` V/out — the
/// asymmetric (DeepSeek/Kimi MLA) family `attention_v_head_dim_parity` pins:
/// scores contract over `head_dim` while V rows and the output are
/// `v_head_dim` wide.
fn chain_asym(
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    hd: usize,
    v_hd: usize,
    kind: MaskKind,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_rows_asym");
    let q = g.input("q", Shape::new(&[b, rows, nh * hd], f));
    let k = g.input("k", Shape::new(&[b, lk, nh * hd], f));
    let v = g.input("v", Shape::new(&[b, lk, nh * v_hd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: Some(v_hd),
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, nh * v_hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// The mask values for [`chain`], at the length its shape implies: ~14% of the
/// keys padding-masked out for `Custom`, a positional-plus-per-head additive
/// bias for `Bias` (so a kernel that collapsed the head axis would show).
fn mask_data(kind: MaskKind, b: usize, rows: usize, lk: usize, nh: usize) -> Vec<f32> {
    match kind {
        MaskKind::Custom => (0..b * lk)
            .map(|i| if i % 7 == 0 { 0.0 } else { 1.0 })
            .collect(),
        _ => (0..b * nh * rows * lk)
            .map(|i| {
                let s = (i % lk) as f32;
                let h = (i / lk) % nh;
                (s * 0.02 + h as f32 * 0.05).sin() * 0.5
            })
            .collect(),
    }
}

/// One GPU run of `g` at `inputs` under the route counters: warmed first so the
/// pipeline is compiled before the bracket, then one measured run — its output,
/// the routes that run took and the wall time in µs.
fn run_graph(g: Graph, inputs: &[(&str, &[f32])]) -> (Vec<f32>, AttentionRoutes, f64) {
    let mut c = Session::new(Device::Gpu).compile(g);
    // Warm, so the pipeline is compiled before the bracket and the deviation
    // printed is the steady-state one.
    let _ = c.run(inputs).remove(0);
    reset_attention_route_counts();
    let start = rlx_runtime::Tick::now();
    let out = c.run(inputs).remove(0);
    let us = rlx_runtime::Tick::now().elapsed_us(start);
    (out, attention_route_counts(), us)
}

/// One attention step of a [`chain`] on both devices, under one route bracket,
/// at the shape's own data: the GPU output, the routes that run took, the
/// max-abs against the CPU, and the measured wall time of the GPU run in µs.
fn run_case(
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    hd: usize,
    kind: MaskKind,
) -> (Vec<f32>, AttentionRoutes, f32, f64) {
    let dim = nh * hd;
    let (g, carries_mask) = chain(b, rows, lk, nh, hd, kind);
    let q = data(b * rows * dim, 0.011, false);
    let k = data(b * lk * dim, 0.013, true);
    let v = data(b * lk * dim, 0.017, false);
    let mask = mask_data(kind, b, rows, lk, nh);
    let mut inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    if carries_mask {
        inputs.push(("mask", &mask));
    }
    let (gpu, routes, us) = run_graph(g.clone(), &inputs);
    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    (gpu, routes, d, us)
}

/// The bar every route parity in this crate is held to at these shapes: a
/// max-abs against the CPU backend, not a relative one, so a diffuse softmax
/// leaving the output small cannot loosen it.
const ATTENTION_BAR: f32 = 1e-6;

/// A prefill-shaped step — `rows` query rows over `lk` keys — asserted to take
/// the dense arrangement and to reach the CPU reference.
fn rows_shape(b: usize, rows: usize, lk: usize, kind: MaskKind) {
    const NH: usize = 8;
    const HD: usize = 64;
    let (_, routes, d, us) = run_case(b, rows, lk, NH, HD, kind);
    eprintln!(
        "[rlx-attn-rows] batch={b} rows={rows} heads={NH} head_dim={HD} lk={lk} {kind:?} \
         routes={routes:?} vs CPU: max_abs={d:.6e} ({us:.1} us)"
    );
    assert!(
        rows >= ATTENTION_DECOMPOSE_MIN_Q as usize,
        "rows={rows} is below the route's floor, so this case would not test it"
    );
    // One attention step per run, so one step on the dense arrangement.
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "batch={b} rows={rows} lk={lk} {kind:?}: the step must take the dense arrangement"
    );
    assert!(
        d < ATTENTION_BAR,
        "batch={b} rows={rows} lk={lk} {kind:?}: max_abs {d} vs CPU"
    );
}

#[test]
fn many_row_shapes_take_the_route_and_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    // 8 heads x 64 query rows over 256 keys, where the fused form would run 512
    // threads and each would walk all 256 keys alone. Two batch elements: the
    // route's multiplies are one tile per (batch, head) either way.
    rows_shape(1, 64, 256, MaskKind::Causal);
    rows_shape(2, 64, 256, MaskKind::Causal);
}

/// The rank-4 `[B, H, S, D]` layout — the one whose per-(batch, head) tiles are
/// dense — takes the same route and reaches the reference: `q_dense`,
/// `v_dense` and `out_dense` all hold there, so the step compacts K alone.
#[test]
fn a_dense_layout_takes_the_route_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const B: usize = 2;
    const NH4: usize = 4;
    const HD4: usize = 64;
    const LK: usize = 192;
    const ROWS: usize = 48;
    let g = chain_bhsd(B, ROWS, LK, NH4, NH4, HD4, MaskKind::Causal);
    let q = data(B * NH4 * ROWS * HD4, 0.011, false);
    let k = data(B * NH4 * LK * HD4, 0.013, true);
    let v = data(B * NH4 * LK * HD4, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    let (gpu, routes, _) = run_graph(g.clone(), &inputs);
    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!(
        "[rlx-attn-rows-bhsd] batch={B} heads={NH4} head_dim={HD4} rows={ROWS} lk={LK} \
         routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "a dense [B, H, S, D] step must take the dense arrangement"
    );
    assert!(d < ATTENTION_BAR, "dense layout: max_abs {d} vs CPU");
}

/// The other rank-4 layout, `[B, S, H, D]` (the CPU / EEG-DINO convention): its
/// `seq_stride` is `H·D`, not the head width, so Q's, V's and the output's tiles
/// are all strided and the route compacts every operand as well as Kᵀ. That is
/// the arrangement `attention_decode_route::decode_route_rank4_bshd_matches_cpu`
/// covers at one query row; this is the many-row side of it.
#[test]
fn a_strided_rank4_layout_takes_the_route_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const B: usize = 2;
    const NH4: usize = 8;
    const HD4: usize = 64;
    const LK: usize = 192;
    const ROWS: usize = 48;
    let g = chain_bshd(B, ROWS, LK, NH4, HD4, MaskKind::Causal);
    let q = data(B * ROWS * NH4 * HD4, 0.011, false);
    let k = data(B * LK * NH4 * HD4, 0.013, true);
    let v = data(B * LK * NH4 * HD4, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    let (gpu, routes, _) = run_graph(g.clone(), &inputs);
    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!(
        "[rlx-attn-rows-bshd] batch={B} heads={NH4} head_dim={HD4} rows={ROWS} lk={LK} \
         routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "a strided [B, S, H, D] step must take the dense arrangement"
    );
    assert!(
        d < ATTENTION_BAR,
        "strided rank-4 layout: max_abs {d} vs CPU"
    );
}

/// The two families that change what the route's regions hold, both on a
/// many-row shape: grouped KV heads (GQA/MQA — the KV head axis has to be
/// replicated per query head on the way into Kᵀ and V′, so an implementation
/// that ignored the group would read the wrong head) and an asymmetric value
/// width (MLA — V's own last axis and the output are `v_head_dim`, while the
/// score still contracts over `head_dim`).
#[test]
fn grouped_kv_and_asymmetric_v_take_the_route_and_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    // GQA: 8 query heads over 2 KV heads (group 4), rank-4 dense layout.
    {
        const NH4: usize = 8;
        const NKV: usize = 2;
        const HD4: usize = 64;
        const LK: usize = 192;
        const ROWS: usize = 48;
        let g = chain_bhsd(1, ROWS, LK, NH4, NKV, HD4, MaskKind::Causal);
        let q = data(NH4 * ROWS * HD4, 0.011, false);
        let k = data(NKV * LK * HD4, 0.013, true);
        let v = data(NKV * LK * HD4, 0.017, false);
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        let (gpu, routes, _) = run_graph(g.clone(), &inputs);
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-rows-gqa] heads={NH4} nkv={NKV} head_dim={HD4} rows={ROWS} lk={LK} \
             routes={routes:?} vs CPU: max_abs={d:.6e}"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 0,
                decomposed: 1,
                serial: 0,
            },
            "a grouped-KV step must take the dense arrangement"
        );
        assert!(d < ATTENTION_BAR, "grouped KV: max_abs {d} vs CPU");
    }
    // MLA: scores over head_dim 64, V rows and the output 32 wide.
    {
        const NH: usize = 4;
        const HD: usize = 64;
        const V_HD: usize = 32;
        const LK: usize = 48;
        const ROWS: usize = 40;
        let g = chain_asym(1, ROWS, LK, NH, HD, V_HD, MaskKind::Causal);
        let q = data(ROWS * NH * HD, 0.011, false);
        let k = data(LK * NH * HD, 0.013, true);
        let v = data(LK * NH * V_HD, 0.017, false);
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        let (gpu, routes, _) = run_graph(g.clone(), &inputs);
        let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
        let d = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "[rlx-attn-rows-mla] heads={NH} head_dim={HD} v_head_dim={V_HD} rows={ROWS} \
             lk={LK} routes={routes:?} vs CPU: max_abs={d:.6e}"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 0,
                decomposed: 1,
                serial: 0,
            },
            "an asymmetric-V step must take the dense arrangement"
        );
        assert!(d < ATTENTION_BAR, "asymmetric V: max_abs {d} vs CPU");
    }
}

/// The other side of the floor, on device: the same geometry with fewer query
/// rows than the route serves keeps the fused kernel, and its numbers. This is
/// the side the fused form is the right one for, so it is the side that must
/// not silently move.
#[test]
fn below_the_floor_keeps_the_fused_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const NH: usize = 8;
    const HD: usize = 64;
    // 24 rows is the top of the range the floor was chosen from — nothing
    // measured between 13 and 24 query rows, an order of magnitude at about 100 —
    // and 31 is the row just below `ATTENTION_DECOMPOSE_MIN_Q`. All are above one
    // query row, so none is decode-shaped.
    for rows in [24usize, 31] {
        let (_, routes, d, us) = run_case(1, rows, 256, NH, HD, MaskKind::Causal);
        eprintln!(
            "[rlx-attn-rows-below] rows={rows} heads={NH} head_dim={HD} lk=256 \
             routes={routes:?} vs CPU: max_abs={d:.6e} ({us:.1} us)"
        );
        assert!(
            rows < ATTENTION_DECOMPOSE_MIN_Q as usize,
            "the floor's own points must sit below it"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 0,
                decomposed: 0,
                serial: 1,
            },
            "rows={rows} is below the route's floor ({ATTENTION_DECOMPOSE_MIN_Q}) and is \
             not decode-shaped: the fused kernel must serve it"
        );
        assert!(
            d < ATTENTION_BAR,
            "rows={rows} on the fused kernel: max_abs {d} vs CPU"
        );
    }
}

/// Every mask kind the fused kernel implements on the dense arrangement against
/// the CPU: the route re-arranges the same rule, so each kind has to reach the
/// reference by the same numbers and none of them may fall off the route.
#[test]
fn route_mask_kinds_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    for kind in [
        MaskKind::None,
        MaskKind::Causal,
        MaskKind::SlidingWindow(2),
        MaskKind::Custom,
        MaskKind::Bias,
    ] {
        rows_shape(1, 48, 192, kind);
    }
}

/// Query rows the cross-route pair below shares.
const CROSS_ROWS: usize = 24;
const CROSS_NH: usize = 8;
const CROSS_HD: usize = 64;

/// The two arrangements on the same problem: a prefill of [`CROSS_ROWS`] query
/// rows is below the route's floor and takes the fused kernel, the same prefill
/// padded out to [`ATTENTION_DECOMPOSE_MIN_Q`] rows takes the dense arrangement —
/// and because both walk `seq_q == seq_k` under a causal mask, row `qi` of either
/// attends to keys `0..=qi` only, so the shared rows are one computation.
///
/// The two outputs are compared as a max-abs on those shared rows — two
/// arrangements of the same f32 work, whose deviation here is ≈2.4e-7 and whose
/// expected band is 1.2e-7…2.5e-7, so [`ATTENTION_BAR`] covers it. It is
/// explicitly **not** bit identity: the fused form rebases its softmax key by key
/// where the dense form rebases once against the row max. The measured deviation
/// is printed.
#[test]
fn the_two_arrangements_agree_across_the_floor() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let dim = CROSS_NH * CROSS_HD;
    let long = ATTENTION_DECOMPOSE_MIN_Q as usize;
    let fused = chain(
        1,
        CROSS_ROWS,
        CROSS_ROWS,
        CROSS_NH,
        CROSS_HD,
        MaskKind::Causal,
    )
    .0;
    let split = chain(1, long, long, CROSS_NH, CROSS_HD, MaskKind::Causal).0;
    // One data stream, read at each graph's own length, so the shared rows and
    // keys carry identical values.
    let q = data(long * dim, 0.011, false);
    let k = data(long * dim, 0.013, true);
    let v = data(long * dim, 0.017, false);
    let shared = CROSS_ROWS * dim;
    let run = |g: Graph, n: usize| -> (Vec<f32>, AttentionRoutes, f64) {
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q[..n]), ("k", &k[..n]), ("v", &v[..n])];
        run_graph(g, &inputs)
    };
    let (fused_out, fused_routes, fused_us) = run(fused, shared);
    let (split_out, split_routes, split_us) = run(split, long * dim);
    assert_eq!(
        fused_routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: 1,
        },
        "{CROSS_ROWS} query rows must keep the fused kernel"
    );
    assert_eq!(
        split_routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "{long} query rows must take the dense arrangement"
    );
    let cross = max_abs_diff(&split_out[..shared], &fused_out);
    eprintln!(
        "[rlx-attn-rows-cross] fused({CROSS_ROWS} rows, {fused_us:.1} us) vs \
         dense({long} rows, {split_us:.1} us) on their shared rows: max_abs={cross:.6e}"
    );
    assert!(
        cross < ATTENTION_BAR,
        "the fused and dense arrangements disagree on the shared prefill rows: max_abs {cross}"
    );
}

/// The route's scratch budget, on the device: a shape tens of MiB into it still
/// takes the route, and one whose plan would need more than the 64 MiB cap falls
/// back to the fused kernel rather than to the route.
#[test]
fn the_scratch_budget_bounds_the_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    // Just inside the budget: Kᵀ and the scores are 8x64x4096 f32 (8 MiB each) and
    // the rank-3 compactions — V′ 8x4096x64 plus the Q′/out′ tiles — bring the
    // scratch to ≈25 MB. The route is exercised far above its floor and well into
    // the reserve it lives in, so `rows_shape` asserts both the arrangement and
    // the CPU bar.
    rows_shape(1, 64, 4096, MaskKind::Causal);
    // Past the budget: the same geometry over 12288 keys is ≈75 MB of scratch,
    // over the route's 64 MiB cap, so its plan refuses and the fused kernel serves
    // the step. The counters are the whole assertion here — a 12288-key diffuse
    // softmax on the fused kernel is not this file's bar to claim, so there is
    // deliberately no parity bar: the point is which arrangement the shape took.
    let (_, routes, _, _) = run_case(1, 64, 12288, 8, 64, MaskKind::Causal);
    eprintln!(
        "[rlx-attn-rows-budget] batch=1 rows=64 heads=8 head_dim=64 lk=12288 Causal \
         routes={routes:?}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: 1,
        },
        "a shape over the route's 64 MiB scratch cap must keep the fused kernel"
    );
}

/// The route's masked scale vetoes the active-extent mode for the whole graph
/// (`Step::AttentionMask::safe_for_active_extent`) — its rows are
/// (batch, head, query row) major, so no single folded extent can leave exactly the
/// rows the caller asked for, and the mode is all-or-nothing per graph. This pins
/// the vetoed case: under the same hint the decode route follows, a many-row step
/// still computes its planned extent, takes the dense arrangement, and reaches the
/// CPU.
#[test]
fn the_route_keeps_its_planned_extent_under_an_active_extent_hint() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const UPPER: usize = 192;
    const ACTUAL: usize = 96;
    const NH: usize = 8;
    const HD: usize = 64;
    const ROWS: usize = 48;
    let dim = NH * HD;
    let (g, _) = chain(1, ROWS, UPPER, NH, HD, MaskKind::Custom);
    let q = data(ROWS * dim, 0.011, false);
    let k = data(UPPER * dim, 0.013, true);
    let v = data(UPPER * dim, 0.017, false);
    // The same key-padding mask the patched decode test uses: the tail beyond
    // `ACTUAL` is masked out, so a walk over all `UPPER` keys and one over the
    // first `ACTUAL` are the same computation.
    let mask: Vec<f32> = (0..UPPER)
        .map(|i| if i < ACTUAL { 1.0 } else { 0.0 })
        .collect();
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v), ("mask", &mask)];
    let mut c = Session::new(Device::Gpu).compile(g.clone());
    // Warm at the declared extent so the measured run is not the first one.
    let _ = c.run(&inputs).remove(0);
    reset_attention_route_counts();
    c.set_active_extent(Some((ACTUAL, UPPER)));
    let gpu = c.run(&inputs).remove(0);
    c.set_active_extent(None);
    let routes = attention_route_counts();
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "the masked scale vetoes the active extent for the whole graph, so the \
         planned {ROWS}-row step must take the dense arrangement"
    );
    // The hint asks for a walk of `ACTUAL` of `UPPER` keys and the veto leaves the
    // graph at its planned `UPPER`, so the CPU run — all `UPPER` keys, the mask
    // dropping the tail — is the same work.
    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!(
        "[rlx-attn-rows-extent] actual={ACTUAL} upper={UPPER} rows={ROWS} heads={NH} \
         head_dim={HD} routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert!(
        d < ATTENTION_BAR,
        "patched many-row walk: max_abs {d} vs CPU"
    );
}

/// The one term of the route decision the graph's shapes do not settle, read per
/// compiled graph rather than from a process-global total: a shape above the
/// floor is *asked for* the route and gets it wherever the arena binds whole, so
/// a non-zero count here is the arena and not the geometry (the lowering's
/// attention debug print says which term it was). A shape below the floor never
/// asks, so it is not a decline either.
#[test]
fn the_decline_count_is_per_graph_and_zero_where_the_route_fires() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let rows = ATTENTION_DECOMPOSE_MIN_Q as usize;
    let (above, _) = chain(1, rows, 192, 8, 64, MaskKind::Causal);
    assert_eq!(
        WgpuExecutable::compile(above).attention_window_declines(),
        0,
        "{rows} query rows are above the floor, so the step asks for the route; a count here \
         means this adapter's bind window does not reach the plan's regions"
    );
    let (below, _) = chain(1, 24, 192, 8, 64, MaskKind::Causal);
    assert_eq!(
        WgpuExecutable::compile(below).attention_window_declines(),
        0,
        "24 query rows are below the floor: the shape's own answer, not a decline"
    );
}

/// A width past the 64 the rest of this file runs at: the route's two multiplies
/// take any head width (only the fused kernel's own 512 ceiling carries over), so
/// the interior widths the Gemma family runs at are pinned on the dense
/// arrangement here.
#[test]
fn a_wide_head_dim_takes_the_route_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _guard = route_lock();
    const NH: usize = 2;
    const HD: usize = 256;
    let (_, routes, d, us) = run_case(1, 32, 64, NH, HD, MaskKind::Causal);
    eprintln!(
        "[rlx-attn-rows-wide] heads={NH} head_dim={HD} rows=32 lk=64 routes={routes:?} \
         vs CPU: max_abs={d:.6e} ({us:.1} us)"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "head_dim {HD} must take the dense arrangement"
    );
    assert!(d < ATTENTION_BAR, "head_dim {HD}: max_abs {d} vs CPU");
}

/// [`chain`] with the mask owned by the graph instead of fed by the caller: the
/// route's mask offset then resolves out of a parameter slot rather than an input,
/// which is the other path its mask resolution reads.
fn chain_param_mask(b: usize, rows: usize, lk: usize, nh: usize, hd: usize) -> Graph {
    let dim = nh * hd;
    let f = DType::F32;
    let mut g = Graph::new("attn_rows_param_mask");
    let q = g.input("q", Shape::new(&[b, rows, dim], f));
    let k = g.input("k", Shape::new(&[b, lk, dim], f));
    let v = g.input("v", Shape::new(&[b, lk, dim], f));
    let mask = g.param("mask", Shape::new(&[b, lk], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, rows, dim], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// A key-padding mask the graph owns rather than one the caller feeds, on the
/// dense arrangement.
#[test]
fn a_param_mask_takes_the_route_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _guard = route_lock();
    const NH: usize = 8;
    const HD: usize = 64;
    const ROWS: usize = 32;
    const LK: usize = 192;
    let dim = NH * HD;
    let mask = mask_data(MaskKind::Custom, 1, ROWS, LK, NH);
    let g = chain_param_mask(1, ROWS, LK, NH, HD);
    let q = data(ROWS * dim, 0.011, false);
    let k = data(LK * dim, 0.013, true);
    let v = data(LK * dim, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    let mut gpu_session = Session::new(Device::Gpu).compile(g.clone());
    gpu_session.set_param("mask", &mask);
    // Warm, so the pipeline is compiled before the bracket.
    let _ = gpu_session.run(&inputs).remove(0);
    reset_attention_route_counts();
    let gpu = gpu_session.run(&inputs).remove(0);
    let routes = attention_route_counts();
    let mut cpu_session = Session::new(Device::Cpu).compile(g);
    cpu_session.set_param("mask", &mask);
    let cpu = cpu_session.run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!(
        "[rlx-attn-rows-param-mask] rows={ROWS} lk={LK} heads={NH} head_dim={HD} \
         routes={routes:?} vs CPU: max_abs={d:.6e}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "a parameter-carried mask must take the dense arrangement"
    );
    assert!(d < ATTENTION_BAR, "parameter mask: max_abs {d} vs CPU");
}
