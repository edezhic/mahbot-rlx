// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Attention precision on wgpu as the sequence crosses the kernel-selection
//! boundary.
//!
//! `rlx-brainbert` is 12 heads x 64 head_dim at **seq = 186**, and its relative
//! L2 on wgpu is ~240x CPU's while every isolated op (matmul, LayerNorm, GELU,
//! softmax) is bit-exact or near it. The backend has two attention kernel
//! shapes — a serial one (one thread per (batch, head, query row)) and a
//! decode-shaped one that only a single query row over a long key sequence
//! takes — and this file now covers both; the earlier cross-backend check only
//! covered seq = 16, i.e. not the path this model uses.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let den: f64 = a
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}

fn case_masked(nh: usize, hd: usize, seq: usize, masked: bool) -> Option<f64> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return None;
    }
    let dim = nh * hd;
    let mut g = Graph::new("attn_long");
    let q = g.input("q", Shape::new(&[1, seq, dim], DType::F32));
    let k = g.input("k", Shape::new(&[1, seq, dim], DType::F32));
    let v = g.input("v", Shape::new(&[1, seq, dim], DType::F32));
    // BrainBERT feeds a padding mask ([batch, key_len]; 1 = valid).
    let mask = g.input("mask", Shape::new(&[1, seq], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: if masked {
                MaskKind::Custom
            } else {
                MaskKind::None
            },
            score_scale: None,
            attn_logit_softcap: None,
        },
        if masked {
            vec![q, k, v, mask]
        } else {
            vec![q, k, v]
        },
        Shape::new(&[1, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let qs: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.011).sin()).collect();
    let ks: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.013).cos()).collect();
    let vs: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.017).sin()).collect();
    // ~15% of positions masked out, as BrainBERT's pretraining does.
    let ms: Vec<f32> = (0..seq)
        .map(|i| if i % 7 == 0 { 0.0 } else { 1.0 })
        .collect();
    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        if masked {
            c.run(&[
                ("q", qs.as_slice()),
                ("k", ks.as_slice()),
                ("v", vs.as_slice()),
                ("mask", ms.as_slice()),
            ])
            .remove(0)
        } else {
            c.run(&[
                ("q", qs.as_slice()),
                ("k", ks.as_slice()),
                ("v", vs.as_slice()),
            ])
            .remove(0)
        }
    };
    Some(rel_l2(&run(Device::Cpu), &run(Device::Gpu)))
}

/// One query row over a long key sequence with grouped KV heads (the decode
/// shape the backend now has a kernel for), `MaskKind::Causal`. The single query
/// row sits at absolute position `lk - 1`, so the causal branch must compare
/// against that, not against a local query index of 0 — a kernel that got that
/// wrong would keep only key 0 instead of all of them.
fn case_decode_causal(nh: usize, nkv: usize, hd: usize, lk: usize) -> Option<f64> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return None;
    }
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let mut g = Graph::new("attn_decode");
    let q = g.input("q", Shape::new(&[1, 1, q_dim], DType::F32));
    let k = g.input("k", Shape::new(&[1, lk, kv_dim], DType::F32));
    let v = g.input("v", Shape::new(&[1, lk, kv_dim], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, 1, q_dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let qs: Vec<f32> = (0..q_dim).map(|i| ((i as f32) * 0.011).sin()).collect();
    let ks: Vec<f32> = (0..lk * kv_dim)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let vs: Vec<f32> = (0..lk * kv_dim)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        c.run(&[
            ("q", qs.as_slice()),
            ("k", ks.as_slice()),
            ("v", vs.as_slice()),
        ])
        .remove(0)
    };
    Some(rel_l2(&run(Device::Cpu), &run(Device::Gpu)))
}

#[test]
fn attention_precision_across_the_seq_boundary() {
    // 12 x 64 is BrainBERT's geometry; 186 is its sequence. Every length here has
    // more than one query row, so all of them take the serial kernel (the decode
    // shape is one query row only, covered at the end of the test).
    let mut rows = Vec::new();
    for seq in [16usize, 64, 65, 186] {
        let Some(r) = case_masked(12, 64, seq, false) else {
            eprintln!("skip: wgpu unavailable");
            return;
        };
        eprintln!("attention 12x64 seq={seq:<4} rel_l2 = {r:.3e}");
        rows.push((seq, r));
        if let Some(rm) = case_masked(12, 64, seq, true) {
            eprintln!("attention 12x64 seq={seq:<4} MASKED rel_l2 = {rm:.3e}");
            rows.push((seq, rm));
        }
    }
    // Decode shapes: one query row, grouped KV heads, causal. A 512-key softmax
    // over this data is diffuse, so the output is small (|out| ~ 7e-3) and this
    // relative measure amplifies the CPU/wgpu summation-order difference. Measured
    // on the lk=512 shape: serial kernel 4.004e-5 vs CPU, decode kernel 3.995e-5
    // vs CPU, and 1.9e-6 between the two kernels. The bar is what the shape
    // supports; the prefill cases above keep the tighter one.
    let mut worst_decode = (0usize, 0f64);
    // Routing evidence, because both kernels clear the bar below on this shape
    // (the serial one reads 4.0e-5 there) and only this test in this binary
    // dispatches attention, so bracketing the counters is enough. Keep the two
    // together: the bar is what the shape supports, so it cannot be the thing a
    // routing regression trips.
    reset_attention_route_counts();
    let mut decode_cases = 0u64;
    for (nh, nkv, hd, lk) in [(12usize, 4usize, 64usize, 186usize), (16, 4, 128, 512)] {
        let Some(r) = case_decode_causal(nh, nkv, hd, lk) else {
            eprintln!("skip: wgpu unavailable");
            return;
        };
        eprintln!("attention {nh}x{hd} nkv={nkv} decode lk={lk:<4} rel_l2 = {r:.3e}");
        decode_cases += 1;
        if r > worst_decode.1 {
            worst_decode = (lk, r);
        }
    }
    assert_eq!(
        attention_route_counts(),
        AttentionRoutes {
            decode: decode_cases,
            serial: 0,
        },
        "a decode-shaped case ran on the serial kernel"
    );
    assert!(
        worst_decode.1 < 1e-4,
        "wgpu decode attention loses precision at lk={} (rel_l2 {:.3e})",
        worst_decode.0,
        worst_decode.1
    );
    let worst = rows
        .iter()
        .cloned()
        .fold((0usize, 0f64), |a, b| if b.1 > a.1 { b } else { a });
    assert!(
        worst.1 < 1e-5,
        "wgpu attention loses precision at seq={} (rel_l2 {:.3e}) — this is the \
         path rlx-brainbert takes",
        worst.0,
        worst.1
    );
}
