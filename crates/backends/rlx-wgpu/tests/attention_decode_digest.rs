// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The decode-shaped attention kernel's output bits, pinned.
//!
//! The pinned FNV-1a 64 digest fixes the decode kernel's rounding at the
//! recognition decode shape — one query row, 2 438 cached keys (a coverage case:
//! the live cell's cache grows in 128-row buckets from its prompt), 16 heads over
//! 8 KV heads at width 128, the width `DECODE_MAX_DIM` bounds and for which the
//! shader sizes the workgroup's Q row and its per-thread O array. A move in its
//! last bits is the
//! change the digest exists to catch; it is over the little-endian f32 bytes,
//! the same witness form `mahbot-inference`'s report uses, and is pinned from
//! the adapter it was taken on — `Apple M3 Pro` (Metal), by that exact name — so
//! another adapter's `exp` rounding prints the digest without asserting it,
//! while the route and the CPU max-abs are always checked.
//!
//! Runs only with a real device (`RLX_REQUIRE_DEVICE=1` in the GPU recipes turns
//! a missing one into a failure). The process-global route counter is bracketed,
//! hence [`common::route_lock`].
mod common;

use common::{data, max_abs_diff, route_lock};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

/// The recognition decode cell: `seq_q = 1`, `seq_k = 2438`, 16 heads over 8
/// KV heads at width 128. A single query row is the lane-parallel form's, so
/// the route is taken and the key walk spans 39 blocks of 64.
const SEQ_K: usize = 2438;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

/// FNV-1a 64 of the decode output on `PINNED_ADAPTER`. Any change to it is a
/// changed rounding, not a tolerance question.
const PINNED_DIGEST: u64 = 0x4a8b_4aa1_ea1f_dcb7;
/// The adapter the digest was taken on, by its exact name: another adapter — an
/// M2, a later Metal version — may round `exp` differently, so the pin asserts
/// only where it was recorded.
const PINNED_ADAPTER: &str = "Apple M3 Pro";

/// FNV-1a 64 over the output's f32 bytes: the same digest form the fork's
/// end-to-end witnesses use, so a last-bit move is one word.
fn digest(values: &[f32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

#[test]
fn the_decode_kernel_leaves_every_output_bit_where_it_was() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _guard = route_lock();

    let f = DType::F32;
    let mut g = Graph::new("attn_decode_digest");
    // Rank-4 [B, S, H, D] so the grouped-KV layout is the one the model walks.
    let q = g.input("q", Shape::new(&[1, 1, HEADS, HEAD_DIM], f));
    let k = g.input("k", Shape::new(&[1, SEQ_K, KV_HEADS, HEAD_DIM], f));
    let v = g.input("v", Shape::new(&[1, SEQ_K, KV_HEADS, HEAD_DIM], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: HEADS,
            head_dim: HEAD_DIM,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, 1, HEADS, HEAD_DIM], f),
    );
    g.set_outputs(vec![y]);

    let qd = data(HEADS * HEAD_DIM, 0.011, false);
    let kd = data(SEQ_K * KV_HEADS * HEAD_DIM, 0.013, true);
    let vd = data(SEQ_K * KV_HEADS * HEAD_DIM, 0.017, false);
    let inputs: Vec<(&str, &[f32])> = vec![("q", &qd), ("k", &kd), ("v", &vd)];

    reset_attention_route_counts();
    let gpu = Session::new(Device::Gpu)
        .compile(g.clone())
        .run(&inputs)
        .remove(0);
    let routes = attention_route_counts();
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 1,
            decomposed: 0,
            serial: 0,
            band: 0,
        },
        "the decode shape must take the decode kernel; the digest below is worthless otherwise"
    );

    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let bar = max_abs_diff(&gpu, &cpu);
    let got = digest(&gpu);
    let adapter = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!(
        "[rlx-attn-decode-digest] adapter={adapter} seq_k={SEQ_K} heads={HEADS} \
         kv_heads={KV_HEADS} head_dim={HEAD_DIM} digest={got:016x} vs CPU: max_abs={bar:.6e}"
    );
    assert!(bar < 1e-6, "decode shape vs CPU: max_abs {bar}");
    if adapter == PINNED_ADAPTER {
        assert_eq!(
            got, PINNED_DIGEST,
            "the decode kernel moved an output bit on {adapter}"
        );
    }
}
