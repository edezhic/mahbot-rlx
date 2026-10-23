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
//! Which of the two layouts a rank-4 operand is *read* in is the graph's to
//! declare and not its extents': at the extent where the two presentations are
//! shape-identical, one body spelled both ways computes the same numbers. The
//! operand nothing declares is refused by name rather than read off the head
//! axis, which `backend::attention_route`'s own unit tests pin device-free.
//!
//! One test compares the two arrangements directly, on one per-(batch, head)
//! problem twice: one query row over 32 keys is the fused kernel's, two rows over
//! the same keys are the dense arrangement's, and under a causal mask both put
//! their last row at the cache's end — so that row attends all 32 keys in either
//! arrangement and is one computation written twice. The pair is a cross-route
//! comparison and not bit identity: the fused form rebases its softmax key by key
//! and the dense form does it once against the row max, so the two sum in
//! different orders. The test asserts the crate's bar and prints the measured
//! deviation; it does not claim the outputs are equal.
//!
//! Several tests here bracket the process-global route counters, so they take
//! [`route_lock`] instead of living in one single-test binary. The timings are
//! printed, never asserted: what a regression moves first is the route.
mod common;

use common::{ATTENTION_BAR, data, max_abs_diff, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::{ATTENTION_DECOMPOSE_MIN_Q, WgpuExecutable};
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

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

/// Rank-4 `[B, S, H, D]` MHA (the CPU / EEG-DINO convention): every operand's
/// per-`(batch, head)` tile is strided — its `seq_stride` is `H·D`, not the head
/// width — so the route compacts Q, V and the output as well as Kᵀ. The operands
/// are bare inputs, so their own extents are all there is to settle the layout
/// and a caller must keep `rows != nh`: at `rows == nh` the two readings are
/// shape-identical and the step is refused by name. A rows-first operand at that
/// extent has to be built from the projection ([`unrolled_rank4`] below).
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

/// The bar the backward step's gradients are held to against the CPU backend.
/// They are sums over the keys of products of exponentials rather than one
/// softmax, so the two backends' own summation orders separate further than a
/// forward's does — measured at ~1e-6 on these shapes, against the ~1e0 the one
/// that used to come back as zeros separates by.
const BACKWARD_BAR: f32 = 1e-5;

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

/// One query row over a handful of keys takes the lane-parallel form, not the
/// serial one, at the geometry the fork's code predictor runs: 16 heads of width
/// 128 over 1..=16 keys (one cache row per level of its block, so a level's step
/// walks as many keys as the level's index plus one). The route carries no key
/// floor (`kernels::attention_decode_eligible`), so every one of those steps is
/// this form's; reinstating any floor above 16 on the route puts this cell back
/// on the serial kernel and turns the assertion red.
#[test]
fn a_one_row_short_key_step_takes_the_decode_form() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const NH: usize = 16;
    const HD: usize = 128;
    // The bar scales with the head width: the two backends' summation orders
    // separate about three times further at 16 heads of 128 than on the 8 × 64
    // cells `ATTENTION_BAR` was set on (measured 1.0e-6 at four keys here), so this
    // cell carries its own rather than loosening the shared one.
    const BAND_BAR: f32 = 4.0 * ATTENTION_BAR;
    // The mask is `None` because this cell pins the kernel, not a mask kind.
    for lk in [1usize, 2, 3, 4, 8, 16] {
        let (_, routes, d, us) = run_case(1, 1, lk, NH, HD, MaskKind::None);
        eprintln!(
            "[rlx-attn-rows-decode] rows=1 heads={NH} head_dim={HD} lk={lk} \
             routes={routes:?} vs CPU: max_abs={d:.6e} ({us:.1} us)"
        );
        assert_eq!(
            routes,
            AttentionRoutes {
                decode: 1,
                decomposed: 0,
                serial: 0,
            },
            "one query row over {lk} keys is the lane-parallel form's"
        );
        assert!(d < BAND_BAR, "the decode form: max_abs {d} vs CPU");
    }
}

/// Causal with fewer keys than query rows — a shape the two-row floor newly
/// admits, and one where the causal offset both arrangements derive
/// (`qi + (seq_k - seq_q)`) has to clamp at zero, since the subtraction would go
/// negative. The leading rows are masked hardest and the step must still reach the
/// CPU's numbers on the route.
#[test]
fn a_causal_step_shorter_in_keys_than_rows_takes_the_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const NH: usize = 8;
    const HD: usize = 64;
    let (_, routes, d, us) = run_case(1, 4, 2, NH, HD, MaskKind::Causal);
    eprintln!(
        "[rlx-attn-rows-short] rows=4 heads={NH} head_dim={HD} lk=2 \
         routes={routes:?} vs CPU: max_abs={d:.6e} ({us:.1} us)"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "four query rows over two causal keys is the route's"
    );
    assert!(
        d < ATTENTION_BAR,
        "causal rows past the key sequence: max_abs {d} vs CPU"
    );
}

/// Every mask kind the fused kernel implements on the dense arrangement against
/// the CPU: the route re-arranges the same rule, so each kind has to reach the
/// reference by the same numbers and none of them may fall off the route. Walked
/// at the band's own two query rows as well as the 48-row shape the kinds were
/// pinned on before the floor moved.
#[test]
fn route_mask_kinds_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    for rows in [ATTENTION_DECOMPOSE_MIN_Q as usize, 48] {
        for kind in [
            MaskKind::None,
            MaskKind::Causal,
            MaskKind::SlidingWindow(2),
            MaskKind::Custom,
            MaskKind::Bias,
        ] {
            rows_shape(1, rows, 192, kind);
        }
    }
}

/// The shapes the cross-arrangement pair below walks: one row for the serial
/// kernel, two for the route, over one key sequence.
const CROSS_ONE_ROW_ROWS: usize = 1;
const CROSS_DENSE_ROWS: usize = 2;
const CROSS_LK: usize = 32;
const CROSS_NH: usize = 8;
const CROSS_HD: usize = 64;

/// The two arrangements on the same problem: one query row over [`CROSS_LK`]
/// keys takes the lane-parallel form, two rows over the same keys take the dense
/// arrangement, and because both derive the causal position the same way —
/// `qi + (seq_k - seq_q)` — their last shared row, row
/// [`CROSS_DENSE_ROWS`] − 1, attends all [`CROSS_LK`] keys in both, so it is one
/// computation.
///
/// The shared row is compared as a max-abs — two arrangements of the same f32
/// work, whose measured deviation is printed. It is explicitly **not** bit
/// identity: the one-row form rebases per lane and merges, where the dense form
/// rebases once against the row max.
#[test]
fn the_two_arrangements_agree_on_their_shared_row() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    let dim = CROSS_NH * CROSS_HD;
    let one_row = chain(
        1,
        CROSS_ONE_ROW_ROWS,
        CROSS_LK,
        CROSS_NH,
        CROSS_HD,
        MaskKind::Causal,
    )
    .0;
    let dense = chain(
        1,
        CROSS_DENSE_ROWS,
        CROSS_LK,
        CROSS_NH,
        CROSS_HD,
        MaskKind::Causal,
    )
    .0;
    // One data stream, read at each graph's own length, so the shared row and
    // every key carry identical values.
    let q = data(CROSS_DENSE_ROWS * dim, 0.011, false);
    let k = data(CROSS_LK * dim, 0.013, true);
    let v = data(CROSS_LK * dim, 0.017, false);
    let run = |g: Graph, qs: &[f32]| -> (Vec<f32>, AttentionRoutes, f64) {
        let inputs: Vec<(&str, &[f32])> = vec![("q", qs), ("k", &k), ("v", &v)];
        run_graph(g, &inputs)
    };
    // The serial leg walks the same query row the dense leg's last row does, so the
    // two are one computation; reading the first row instead would compare two
    // different queries.
    let shared = (CROSS_DENSE_ROWS - 1) * dim;
    let (one_row_out, one_row_routes, one_row_us) = run(one_row, &q[shared..shared + dim]);
    let (dense_out, dense_routes, dense_us) = run(dense, &q[..CROSS_DENSE_ROWS * dim]);
    assert_eq!(
        one_row_routes,
        AttentionRoutes {
            decode: 1,
            decomposed: 0,
            serial: 0,
        },
        "one query row over {CROSS_LK} keys must take the lane-parallel form"
    );
    assert_eq!(
        dense_routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 1,
            serial: 0,
        },
        "{CROSS_DENSE_ROWS} query rows over {CROSS_LK} keys must take the dense arrangement"
    );
    let cross = max_abs_diff(&dense_out[shared..shared + dim], &one_row_out);
    eprintln!(
        "[rlx-attn-rows-cross] one-row({CROSS_ONE_ROW_ROWS} row, {one_row_us:.1} us) vs \
         dense({CROSS_DENSE_ROWS} rows, {dense_us:.1} us) on their shared row: max_abs={cross:.6e}"
    );
    assert!(
        cross < ATTENTION_BAR,
        "the one-row and dense arrangements disagree on the shared row: max_abs {cross}"
    );
}

/// The route's scratch budget, on the device: a shape tens of MiB into it still
/// takes the route, and one whose plan would need more than the 512 MiB cap falls
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
    // Past the budget: 4096 query rows over 4096 keys is the cap in scores alone
    // (512 MiB) and ≈544 MiB once Kᵀ and the three 8 MiB compactions are counted,
    // over the route's 512 MiB cap, so its plan refuses and the fused kernel
    // serves the step. Only the route counters are read, so this case runs no
    // CPU reference and builds no mask: a 4096-row parity bar is not this file's
    // claim, and the point is which arrangement the shape took.
    const DIM: usize = 8 * 64;
    let (g, carries_mask) = chain(1, 4096, 4096, 8, 64, MaskKind::Causal);
    assert!(!carries_mask, "a causal step carries no mask");
    let q = data(4096 * DIM, 0.011, false);
    let k = data(4096 * DIM, 0.013, true);
    let v = data(4096 * DIM, 0.017, false);
    let (_, routes, _) = run_graph(g, &[("q", &q), ("k", &k), ("v", &v)]);
    eprintln!(
        "[rlx-attn-rows-budget] batch=1 rows=4096 heads=8 head_dim=64 lk=4096 Causal \
         routes={routes:?}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: 1,
        },
        "a shape over the route's 512 MiB scratch cap must keep the fused kernel"
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
/// compiled graph rather than from a process-global total: a shape at or above the
/// floor is *asked for* the route and gets it wherever the arena binds whole, so
/// a non-zero count here is the arena and not the geometry (the lowering's
/// attention debug print says which term it was). A shape the decode form or the
/// floor keeps off the route never asks, so it is not a decline either.
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
        "{rows} query rows are at or above the floor, so the step asks for the route; a count \
         here means this adapter's bind window does not reach the plan's regions"
    );
    let (below, _) = chain(1, 1, 192, 8, 64, MaskKind::Causal);
    assert_eq!(
        WgpuExecutable::compile(below).attention_window_declines(),
        0,
        "one query row over 192 keys is the decode form's: the shape's own answer, not a \
         decline"
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

/// Rank-4 `[B, S, H, D]` MHA spelled the way the bodies spell it: each rank-3
/// `[B, S, H·D]` projection unrolled into per-head axes. At `rows == nh` its
/// first two axes are both the head count — the two readings are then
/// shape-identical — so the reshape is the only thing that says which way round
/// the operand runs.
fn unrolled_rank4(b: usize, rows: usize, lk: usize, nh: usize, hd: usize) -> Graph {
    let f = DType::F32;
    let dim = nh * hd;
    let mut g = Graph::new("attn_rows_unrolled");
    let unroll = |g: &mut Graph, name: &str, seq: usize| -> rlx_ir::NodeId {
        let packed = g.input(name, Shape::new(&[b, seq, dim], f));
        g.add_node(
            Op::Reshape {
                new_shape: vec![b as i64, seq as i64, nh as i64, hd as i64],
            },
            vec![packed],
            Shape::new(&[b, seq, nh, hd], f),
        )
    };
    let (q, k, v) = (
        unroll(&mut g, "q", rows),
        unroll(&mut g, "k", lk),
        unroll(&mut g, "v", lk),
    );
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// The packed QKV spelling the decoder bodies build, at a self-attention shape:
/// one `[B, S, 3·H·D]` input viewed as `[B, S, 3, H, D]`, narrowed on axis 2 per
/// operand and each narrow unrolled into `[B, S, H, D]`. Its sequence is the
/// query rows as well as the keys, so a packed step is one `rows` over all three.
fn packed_rank4(b: usize, rows: usize, nh: usize, hd: usize) -> Graph {
    let f = DType::F32;
    let dim = nh * hd;
    let mut g = Graph::new("attn_rows_packed_rank4");
    let packed = g.input("qkv", Shape::new(&[b, rows, 3 * dim], f));
    let view = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, rows as i64, 3, nh as i64, hd as i64],
        },
        vec![packed],
        Shape::new(&[b, rows, 3, nh, hd], f),
    );
    let operand = |g: &mut Graph, start: usize| -> rlx_ir::NodeId {
        let narrow = g.add_node(
            Op::Narrow {
                axis: 2,
                start,
                len: 1,
            },
            vec![view],
            Shape::new(&[b, rows, 1, nh, hd], f),
        );
        g.add_node(
            Op::Reshape {
                new_shape: vec![b as i64, rows as i64, nh as i64, hd as i64],
            },
            vec![narrow],
            Shape::new(&[b, rows, nh, hd], f),
        )
    };
    let (q, k, v) = (operand(&mut g, 0), operand(&mut g, 1), operand(&mut g, 2));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// The rank-3 reading of that same `[B, S, 3·H·D]` input: the feature axis
/// narrowed per operand directly, which is the spelling [`chain`] takes.
fn packed_rank3(b: usize, rows: usize, nh: usize, hd: usize) -> Graph {
    let f = DType::F32;
    let dim = nh * hd;
    let mut g = Graph::new("attn_rows_packed_rank3");
    let packed = g.input("qkv", Shape::new(&[b, rows, 3 * dim], f));
    let operand = |g: &mut Graph, start: usize| -> rlx_ir::NodeId {
        g.add_node(
            Op::Narrow {
                axis: 2,
                start,
                len: dim,
            },
            vec![packed],
            Shape::new(&[b, rows, dim], f),
        )
    };
    let (q, k, v) = (
        operand(&mut g, 0),
        operand(&mut g, dim),
        operand(&mut g, 2 * dim),
    );
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, dim], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// A backward step of the same shapes, spelled both ways one graph can spell it:
/// the rank-3 `[B, S, H·D]` projections [`chain`] uses, or those projections
/// unrolled into per-head axes — the rows-first rank-4 spelling. `dy` carries the
/// step's output shape, and the three gradients are the graph's outputs.
fn backward_graph(b: usize, rows: usize, lk: usize, nh: usize, hd: usize, unrolled: bool) -> Graph {
    let f = DType::F32;
    let dim = nh * hd;
    let mut g = Graph::new("attn_rows_backward");
    let unroll = |g: &mut Graph, name: &str, seq: usize| -> rlx_ir::NodeId {
        let packed = g.input(name, Shape::new(&[b, seq, dim], f));
        g.add_node(
            Op::Reshape {
                new_shape: vec![b as i64, seq as i64, nh as i64, hd as i64],
            },
            vec![packed],
            Shape::new(&[b, seq, nh, hd], f),
        )
    };
    let (q, k, v, dy) = if unrolled {
        (
            unroll(&mut g, "q", rows),
            unroll(&mut g, "k", lk),
            unroll(&mut g, "v", lk),
            unroll(&mut g, "dy", rows),
        )
    } else {
        (
            g.input("q", Shape::new(&[b, rows, dim], f)),
            g.input("k", Shape::new(&[b, lk, dim], f)),
            g.input("v", Shape::new(&[b, lk, dim], f)),
            g.input("dy", Shape::new(&[b, rows, dim], f)),
        )
    };
    let (dq, dk, dv) = g.attention_backward_all(q, k, v, dy, nh, hd, MaskKind::None, None);
    g.set_outputs(vec![dq, dk, dv]);
    g
}

/// Rows-first `[B, S, H, D]` SDPA — `rows` query rows over `lk` keys, no mask —
/// written out here rather than taken from the crate's own reference
/// (`basic.rs::cpu_attention_bshd`), so that two paths misreading an operand
/// alike cannot satisfy the pin between them. The scale is `1/sqrt(head_dim)`,
/// the softmax rebases once per row on the row max, and V is read at the key's
/// own row.
fn cpu_attention_rows_first(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    rows: usize,
    lk: usize,
    nh: usize,
    hd: usize,
) -> Vec<f32> {
    let hs = nh * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; b * rows * hs];
    for bi in 0..b {
        for qi in 0..rows {
            for hi in 0..nh {
                let q_base = bi * rows * hs + qi * hs + hi * hd;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; hd];
                for ki in 0..lk {
                    let k_base = bi * lk * hs + ki * hs + hi * hd;
                    let mut score = 0.0f32;
                    for d in 0..hd {
                        score += q[q_base + d] * k[k_base + d];
                    }
                    score *= scale;
                    let m_new = m.max(score);
                    let e_old = (m - m_new).exp();
                    let e_cur = (score - m_new).exp();
                    l = e_old * l + e_cur;
                    let v_base = bi * lk * hs + ki * hs + hi * hd;
                    for d in 0..hd {
                        acc[d] = e_old * acc[d] + e_cur * v[v_base + d];
                    }
                    m = m_new;
                }
                let inv_l = 1.0 / l;
                for d in 0..hd {
                    out[q_base + d] = acc[d] * inv_l;
                }
            }
        }
    }
    out
}

/// The coincident extent on the device: a body-shaped step whose Q carries the
/// head count on axis 1 *and* axis 2, in the two presentations one graph can
/// spell it in. Both must compute the same numbers — the reading is the graph's
/// and not the extents' — and the rows-first one must reach a reference written
/// in this file, so a pair of paths misreading the operand alike cannot pass
/// between them. The head counts are what makes the two axes coincide — 16 rows at
/// the body's 16 heads and 32 at 32 — and all four cases are above the route's
/// query-row floor, so what they bracket is the layout reading and not the floor.
/// The pair at 13 and at 32 rows over 16 heads is the guard that nothing moved where
/// the extents settle the layout: before the graph was read, `rows = 16` alone came
/// out `1.159` off its rank-3 spelling at these shapes, while 13 and 32 were bitwise
/// equal.
#[test]
fn a_coincident_extent_computes_as_its_graph_lays_it_out() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const HD: usize = 64;
    const LK: usize = 40;
    for (nh, rows) in [(16usize, 13usize), (16, 16), (16, 32), (32, 32)] {
        let dim = nh * HD;
        // One data stream per step, read by both of its presentations, so the two
        // carry identical operands.
        let q = data(rows * dim, 0.011, false);
        let k = data(LK * dim, 0.013, true);
        let v = data(LK * dim, 0.017, false);
        let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        let (rank3, rank3_routes, _) =
            run_graph(chain(1, rows, LK, nh, HD, MaskKind::None).0, &inputs);
        let (rank4, rank4_routes, _) = run_graph(unrolled_rank4(1, rows, LK, nh, HD), &inputs);
        let cross = max_abs_diff(&rank4, &rank3);
        // The other half of the pin: the rows-first reading is the one this file's
        // own reference computes, not just the one the other spelling does.
        let want = cpu_attention_rows_first(&q, &k, &v, 1, rows, LK, nh, HD);
        let against_reference = max_abs_diff(&rank4, &want);
        eprintln!(
            "[rlx-attn-coincident] rows={rows} heads={nh} head_dim={HD} lk={LK} \
             rank4={rank4_routes:?} rank3={rank3_routes:?} rank4-vs-rank3: max_abs={cross:.6e} \
             rank4-vs-reference: max_abs={against_reference:.6e}"
        );
        assert_eq!(
            rank4_routes, rank3_routes,
            "rows={rows} heads={nh}: which way an operand is spelled must not move the step's \
             arrangement"
        );
        assert!(
            cross <= ATTENTION_BAR,
            "rows={rows} heads={nh}: the rank-4 rows-first spelling must compute the rank-3 \
             spelling's own numbers — the key length off the row axis, {LK}, and not the head \
             axis' {nh}: max_abs {cross}"
        );
        assert!(
            against_reference <= 1e-5,
            "rows={rows} heads={nh}: the rows-first spelling against the reference written in \
             this file: max_abs {against_reference}"
        );
    }

    // The other spelling the bodies build — one packed QKV parent — at a
    // self-attention shape where its axis 2 is the head count too, against the
    // rank-3 narrows of the same input.
    const NH: usize = 16;
    let rows = NH;
    let dim = NH * HD;
    let packed = data(rows * 3 * dim, 0.007, false);
    let inputs: Vec<(&str, &[f32])> = vec![("qkv", &packed)];
    let (unrolled, unrolled_routes, _) = run_graph(packed_rank4(1, rows, NH, HD), &inputs);
    let (narrowed, narrowed_routes, _) = run_graph(packed_rank3(1, rows, NH, HD), &inputs);
    let d = max_abs_diff(&unrolled, &narrowed);
    eprintln!(
        "[rlx-attn-coincident-packed] rows={rows} heads={NH} head_dim={HD} \
         rank4={unrolled_routes:?} rank3={narrowed_routes:?} max_abs={d:.6e}"
    );
    assert_eq!(
        unrolled_routes, narrowed_routes,
        "the packed step's two spellings must take one arrangement"
    );
    assert!(
        d <= ATTENTION_BAR,
        "the packed rank-4 spelling must compute the rank-3 narrows of the same parent's own \
         numbers: max_abs {d}"
    );
}

/// The backward half of the same step, on the device: a body-shaped step's three
/// gradients, spelled the rank-3 `[B, S, H·D]` way and the rows-first rank-4 way,
/// are one set of numbers — and neither spelling's own batch and key length may
/// be lost on the way, which is how the rank-3 spelling once came back as zeros
/// while the rank-4 spelling of the same graph was right. The rank-4 half is at
/// the coincident extent, so it is also the backward movement of the forward pin
/// above: the layout the graph declares, read by the arm that computes these
/// gradients. The CPU backend's own rank-3 reading is the second oracle, rank-3
/// being the spelling with no coincidence to settle.
#[test]
fn a_backward_step_computes_the_numbers_its_spelling_declares() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _guard = route_lock();
    const NH: usize = 16;
    const HD: usize = 64;
    const ROWS: usize = 16;
    const LK: usize = 40;
    let dim = NH * HD;
    let (q, k, v, dy) = (
        data(ROWS * dim, 0.011, false),
        data(LK * dim, 0.013, true),
        data(LK * dim, 0.017, false),
        data(ROWS * dim, 0.019, true),
    );
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v), ("dy", &dy)];
    let run = |device: Device, unrolled: bool| -> Vec<Vec<f32>> {
        Session::new(device)
            .compile(backward_graph(1, ROWS, LK, NH, HD, unrolled))
            .run(&inputs)
    };
    let cpu = run(Device::Cpu, false);
    let rank3 = run(Device::Gpu, false);
    let rank4 = run(Device::Gpu, true);
    for (i, name) in ["dq", "dk", "dv"].into_iter().enumerate() {
        let (rank3, rank4, cpu) = (&rank3[i], &rank4[i], &cpu[i]);
        let declared = max_abs_diff(rank3, rank4);
        let against_cpu = max_abs_diff(rank3, cpu);
        eprintln!(
            "[rlx-attn-coincident-backward] {name} rows={ROWS} heads={NH} head_dim={HD} lk={LK} \
             rank3-vs-rank4: max_abs={declared:.6e} rank3-vs-cpu: max_abs={against_cpu:.6e}"
        );
        assert!(
            rank3.iter().any(|x| x.is_finite() && x.abs() > 1e-8),
            "{name}: the step's own gradient, which is not all zeros"
        );
        assert!(
            declared <= ATTENTION_BAR,
            "{name}: the two spellings are one graph, so one set of numbers — the reading is the \
             graph's and no extent of it is read off the head axis: max_abs {declared}"
        );
        assert!(
            against_cpu <= BACKWARD_BAR,
            "{name}: and the CPU backend's own rank-3 reading of it: max_abs {against_cpu}"
        );
    }
}
