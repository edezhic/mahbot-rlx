// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Asymmetric `Op::Attention { v_head_dim: Some(v) }` (DeepSeek/Kimi MLA)
//! parity — wgpu vs CPU.
//!
//! Q/K scores contract over `head_dim`; V rows + output are `v_head_dim`
//! wide (v != head_dim). The CPU backend already implements the asymmetric
//! path (`dh`=head_dim for Q/K, `dh_v`=v_head_dim for V-read/score@V/O-write);
//! this pins the wgpu kernel to it.
//!
//! The decode shapes (`lq = 1`, long key sequence) take the decode-shaped
//! kernel, whose merge walks the value dims in fixed-width passes — so a
//! `v_head_dim` narrower than one pass, and one that is asymmetric but wide,
//! both have to round-trip. The asymmetric width is pinned on all three
//! arrangements a shape can land on: the dense route at six rows, the fused
//! kernel at one row under the decode form's key floor, and the decode kernel
//! over a long cache.

mod common;

use common::expected_attention_routes;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::kernels::{
    AttentionRoutes, attention_decode_eligible, attention_route_counts,
    reset_attention_route_counts,
};

/// Rank-3 `[B, S, H·head_dim]` Q/K, `[B, S, H·v_head_dim]` V/out, causal.
/// `lq` query rows over `lk` cached keys (`lq = 1` is the decode shape).
fn run_case(nh: usize, hd: usize, v_hd: usize, lq: usize, lk: usize) -> Option<f32> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let b = 1usize;
    let q_dim = nh * hd; // Q/K per-head width = head_dim
    let v_dim = nh * v_hd; // V/out per-head width = v_head_dim (asymmetric)

    let q: Vec<f32> = (0..b * lq * q_dim)
        .map(|i| ((i as f32) * 0.021).sin())
        .collect();
    let k: Vec<f32> = (0..b * lk * q_dim)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let v: Vec<f32> = (0..b * lk * v_dim)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();

    let mut g = Graph::new("attn_vhd");
    let qi = g.input("q", Shape::new(&[b, lq, q_dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, lk, q_dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, lk, v_dim], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: Some(v_hd), // asymmetric MLA-style SDPA
            mask_kind: MaskKind::Causal,
            score_scale: None, // default = 1/sqrt(head_dim)
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[b, lq, v_dim], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
        ])
        .remove(0)
    };
    reset_attention_route_counts();
    let gpu = run(Device::Gpu);
    let routes = attention_route_counts();
    let cpu = run(Device::Cpu);
    // One attention step per run: the decode form's for a single query row over a
    // long enough key sequence — the predicate also carries this file's asymmetric
    // width term — and otherwise the row/key floors `expected_attention_routes`
    // encodes (dense from the floor up, serial below it).
    // That helper reads the counts alone; these widths clear the decode caps.
    let decode = attention_decode_eligible(lq as u32, lk as u32, hd as u32, v_hd as u32);
    assert_eq!(
        routes,
        if decode {
            AttentionRoutes {
                decode: 1,
                decomposed: 0,
                serial: 0,
            }
        } else {
            expected_attention_routes(lq, lk, 1, 1)
        },
        "nh={nh} hd={hd} v_hd={v_hd} lq={lq} lk={lk}: attention_decode_eligible says \
         {decode}, the run says {routes:?}"
    );
    assert_eq!(
        gpu.len(),
        b * lq * v_dim,
        "gpu output width == H·v_head_dim"
    );
    assert_eq!(
        cpu.len(),
        b * lq * v_dim,
        "cpu output width == H·v_head_dim"
    );
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("attn asym nh={nh} hd={hd} v_hd={v_hd} lq={lq} lk={lk}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_attention_asymmetric_v_head_dim_parity() {
    // head_dim=8, v_head_dim=4, 2 heads, short seq, Causal, rank-3 — the dense
    // route's, now that the floor is two rows.
    let Some(max_abs) = run_case(2, 8, 4, 6, 6) else {
        return;
    };
    assert!(
        max_abs < 1e-3,
        "wgpu asymmetric v_head_dim max_abs {max_abs} >= 1e-3"
    );
    // The same asymmetric path on the serial kernel: one query row over 48 keys
    // sits at or under the decode form's key floor, so the serial kernel serves
    // it, and its merge must cover the V dims the Q/K contraction does not.
    let Some(max_abs) = run_case(2, 64, 32, 1, 48) else {
        return;
    };
    assert!(
        max_abs < 1e-3,
        "wgpu asymmetric v_head_dim on the fused kernel: max_abs {max_abs} >= 1e-3"
    );
    // Decode shapes on the same asymmetric path: a value width below one merge
    // pass (4 of head_dim 8), a wide asymmetric one (64 of head_dim 128), and one
    // wider than the score width (128 of head_dim 64) — the last is inside the
    // decode form's eligibility (both widths ≤ the cap) and puts the merge on V
    // dims the Q/K contraction does not cover.
    for (nh, hd, v_hd) in [(2usize, 8usize, 4usize), (4, 128, 64), (4, 64, 128)] {
        let Some(max_abs) = run_case(nh, hd, v_hd, 1, 186) else {
            return;
        };
        assert!(
            max_abs < 1e-3,
            "wgpu decode asymmetric v_head_dim nh={nh} hd={hd} v_hd={v_hd} \
             max_abs {max_abs} >= 1e-3"
        );
    }
}
