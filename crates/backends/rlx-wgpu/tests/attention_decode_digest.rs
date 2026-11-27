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
use rlx_runtime::{CompiledGraph, Device, Session};
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
///
/// Re-pinned by the 0.2.20 sync, from `0x4a8b_4aa1_ea1f_dcb7`: upstream replaced
/// this kernel's softmax with the register-resident form, and the same change put
/// the correctly-rounded `rlx_exp` / `rlx_rcp` helpers in place of WGSL's native
/// `exp` / `1.0/x`. The output still agrees with the CPU reference to
/// `max_abs ≈ 1.35e-7` on the same shape, so this is the helpers' rounding
/// landing in the last bits rather than a change of computation.
///
/// Re-pinned again from `0x5a73_8a91_bd04_e719` by the weighted decode body
/// (`kernels/attention.wgsl`), which combines a step's partial sums per value dim
/// rather than per key: this shape is 2 438 keys, over the split floor, so it walks
/// two ranges of three `DECODE_KEY_TILE` tiles — the tile boundaries, the value-dim
/// split and the merge are all in the digest. Same computation, same helpers, a
/// different order of the sums — the CPU reference is *closer* now (`max_abs ≈
/// 7.2e-8`, was `≈ 1.35e-7`), and every transcript, label and score the product
/// emits is unchanged. A pin on the one-pass body's bits is
/// `RLX_WGPU_ATTN_DECODE_ONLINE`'s to hold, not this constant's.
const PINNED_DIGEST: u64 = 0x269f_bcb4_fb37_5b10;
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

/// The switch the one-pass body answers to.
const ONLINE: &str = "RLX_WGPU_ATTN_DECODE_ONLINE";

/// Keys the masked case walks and how many of them the mask keeps: a recognition
/// rung's plane, whose tail is the part a resident plane cache has not written —
/// the product's own arrangement of masked keys, and enough keys that the step
/// splits its range across two workgroups.
const MASKED_SEQ_K: usize = 193;
const KEPT: usize = 133;

/// One decode step over a binary key mask, at the widths above.
fn masked_step() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_decode_masked");
    let q = g.input("q", Shape::new(&[1, 1, HEADS, HEAD_DIM], f));
    let k = g.input("k", Shape::new(&[1, MASKED_SEQ_K, KV_HEADS, HEAD_DIM], f));
    let v = g.input("v", Shape::new(&[1, MASKED_SEQ_K, KV_HEADS, HEAD_DIM], f));
    let m = g.input("m", Shape::new(&[1, MASKED_SEQ_K], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: HEADS,
            head_dim: HEAD_DIM,
            v_head_dim: None,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, m],
        Shape::new(&[1, 1, HEADS, HEAD_DIM], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// A dropped key contributes nothing, in either body — the one-pass body's
/// `e_cur == 0` early-out and the weighted body's `select` are the same property,
/// and it is what keeps a non-finite V at a key the mask dropped out of the sum.
/// The mask here is a plane cache's: a written prefix that attends and a tail that
/// does not, at an extent the step splits in two.
#[test]
fn a_masked_key_contributes_nothing_whatever_its_value() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _guard = route_lock();

    let g = masked_step();
    let qd = data(HEADS * HEAD_DIM, 0.011, false);
    let kd = data(MASKED_SEQ_K * KV_HEADS * HEAD_DIM, 0.013, true);
    let vd = data(MASKED_SEQ_K * KV_HEADS * HEAD_DIM, 0.017, false);
    let mut mask = vec![1.0f32; MASKED_SEQ_K];
    mask[KEPT..].fill(0.0);
    let open = vec![1.0f32; MASKED_SEQ_K];
    // The same step with a non-finite V at every dropped key: a body that lost the
    // property carries it into the sum and answers with a NaN.
    let mut wild = vd.clone();
    wild[KEPT * KV_HEADS * HEAD_DIM..].fill(f32::NAN);

    // One compiled graph runs both arms: the body is a per-run pick, so the parity
    // needs no second compile.
    let mut graph = Session::new(Device::Gpu).compile(g);
    let run = |cg: &mut CompiledGraph, v: &[f32], m: &[f32], form: &str| {
        let _form = rlx_ir::env::RuntimeOverrides::install([(ONLINE, form)]);
        cg.run(&[("q", &qd), ("k", &kd), ("v", v), ("m", m)])
            .remove(0)
    };

    reset_attention_route_counts();
    let a = run(&mut graph, &vd, &mask, "0");
    let b = run(&mut graph, &vd, &mask, "1");
    let routes = attention_route_counts();
    assert_eq!(
        (routes.decode, routes.serial),
        (2, 0),
        "the masked step must take the decode kernel in both bodies; the case below \
         is worthless otherwise"
    );
    let d = max_abs_diff(&a, &b);
    assert!(
        d < 1e-6,
        "the decode bodies disagree on a masked step: max_abs {d:.3e}"
    );

    // The mask is not a no-op: the same step attending every key answers elsewhere,
    // so an output that matched it would mean the dropped keys were never dropped.
    let all = run(&mut graph, &vd, &open, "0");
    let kept_out = max_abs_diff(&all, &a);
    assert!(
        kept_out > 1e-2,
        "masking {MASKED_SEQ_K} keys to {KEPT} moved the answer by {kept_out:.3e}: the \
         mask did not take, and the non-finite V below would prove nothing"
    );

    // The property itself, on both bodies: the dropped keys' V cannot reach the sum,
    // so its bytes make no difference to the answer at all.
    let a_wild = run(&mut graph, &wild, &mask, "0");
    let b_wild = run(&mut graph, &wild, &mask, "1");
    for (name, got, reference) in [("weighted", &a_wild, &a), ("one-pass", &b_wild, &b)] {
        assert!(
            got.iter().all(|x| x.is_finite()),
            "the {name} body carried a dropped key's non-finite V into the output"
        );
        assert_eq!(
            digest(got),
            digest(reference),
            "the {name} body's answer moved when the dropped keys' V did"
        );
    }
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
