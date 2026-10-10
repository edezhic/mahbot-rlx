// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-staging ↔ `HostTensorCache` coherence.
//!
//! Host steps share the arena in three families, and the first two used to
//! disagree about who owns it:
//!
//! * **cache-aware** steps (`HostOp`, `Conv2dHost`, `ExpandHost`, `NarrowHost`,
//!   `TransposeHost`, `ConcatHost`, `BufferCopy`) read and write through
//!   `HostTensorCache`, keeping results in a host mirror and deferring the
//!   device write;
//! * **whole-arena** steps (`GroupNormHost`, `LayerNorm2dHost`, `ReverseHost`,
//!   `GruHost`, … — everything routed via `rlx_gpu_host::with_whole_arena`)
//!   read the device, recompute on the host, and rewrite the arena directly;
//! * **compact** steps (`Step::ConvTranspose2dHost`) stage only the tensors they
//!   touch, straight to and from the device: like the whole-arena family they
//!   need the mirror flushed in front of them and dropped behind them, which is
//!   why they sit outside the cache-aware list `bypasses_host_cache` tests.
//!
//! A whole-arena step got the mirror flushed *before* it ran, but nothing
//! invalidated the mirror *after*. The next cache-aware step could then serve a
//! pre-step copy of a region the whole-arena step had just overwritten, so the
//! arena and the mirror silently disagreed. `host_cache.clear()` is only
//! reached after real GPU work (`pass_dispatched`), and a run of consecutive
//! host steps never gets there.
//!
//! It stayed hidden because it needs *both* families adjacent in one schedule,
//! which normally only happens on discrete NVIDIA (where elementwise, conv and
//! norms are all hosted). `RLX_WGPU_FORCE_HOST=1` reproduces it on any adapter,
//! so this runs everywhere — in its own test binary because that env var has to
//! be set before the first compile and is process-global.

mod common;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// GroupNorm over NCHW with one group per channel (instance norm).
fn instance_norm_ref(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    c: usize,
    hw: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0f32; c * hw];
    for ch in 0..c {
        let base = ch * hw;
        let mean = (0..hw).map(|i| x[base + i]).sum::<f32>() / hw as f32;
        let var = (0..hw).map(|i| (x[base + i] - mean).powi(2)).sum::<f32>() / hw as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..hw {
            y[base + i] = (x[base + i] - mean) * inv * gamma[ch] + beta[ch];
        }
    }
    y
}

#[test]
fn whole_arena_host_step_invalidates_the_host_mirror() {
    // Must precede the first compile: the lowering reads this to decide which
    // ops take the host fallback.
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let (c, h, w) = (4usize, 8usize, 8usize);
    let hw = h * w;
    let sh = Shape::new(&[1, c, h, w], DType::F32);
    let eps = 1e-5f32;

    let xs: Vec<f32> = (0..c * hw)
        .map(|i| ((i * 41 % 173) as f32 / 173.0) - 0.5 + (i % 5) as f32 * 0.3)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..c).map(|i| i as f32 * 0.05 - 0.1).collect();
    let addend: Vec<f32> = (0..c * hw).map(|i| (i % 11) as f32 * 0.01).collect();

    // The stale entry has to belong to a *dead* tensor whose arena slot the
    // norm then reuses — that is what puts a pre-norm copy in the mirror at an
    // offset the norm has since overwritten. So: build up several cache-aware
    // intermediates first (they get mirrored), let them die, then norm, then
    // read the result back through more cache-aware steps.
    let mut g = Graph::new("stage_coherence");
    let x = g.input("x", sh.clone());
    let a = g.input("a", sh.clone());
    let ga = g.param("g", Shape::new(&[c], DType::F32));
    let be = g.param("b", Shape::new(&[c], DType::F32));

    let t0 = g.add_node(Op::Activation(Activation::Relu), vec![x], sh.clone());
    let t1 = g.add_node(Op::Binary(BinaryOp::Add), vec![t0, a], sh.clone());
    let t2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![t1, a], sh.clone());
    let t3 = g.add_node(Op::Binary(BinaryOp::Add), vec![t2, a], sh.clone());
    // t0..t2 are dead from here; the norm output may land on one of their slots.
    let n = g.add_node(
        Op::GroupNorm { num_groups: c, eps },
        vec![t3, ga, be],
        sh.clone(),
    );
    let r = g.add_node(Op::Activation(Activation::Relu), vec![n], sh.clone());
    let s = g.add_node(Op::Binary(BinaryOp::Add), vec![r, a], sh.clone());
    let s2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![s, a], sh.clone());
    g.set_outputs(vec![s2]);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("g", &gamma);
    exe.set_param("b", &beta);
    let got = exe.run(&[("x", &xs), ("a", &addend)]);

    let t0: Vec<f32> = xs.iter().map(|v| v.max(0.0)).collect();
    let t1: Vec<f32> = t0.iter().zip(&addend).map(|(v, d)| v + d).collect();
    let t2: Vec<f32> = t1.iter().zip(&addend).map(|(v, d)| v * d).collect();
    let t3: Vec<f32> = t2.iter().zip(&addend).map(|(v, d)| v + d).collect();
    let normed = instance_norm_ref(&t3, &gamma, &beta, c, hw, eps);
    let want: Vec<f32> = normed
        .iter()
        .zip(&addend)
        .map(|(v, d)| (v.max(0.0) + d) * d)
        .collect();

    let max_d = got[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_d < 1e-3,
        "consumers of a whole-arena host step read a stale mirror \
         (max|Δ| = {max_d:.3e})\n got {:?}\nwant {want:?}",
        got[0]
    );
}

/// A cache-aware host step's deferred write must reach the device before the
/// compact transposed-conv step stages its operands off it, and that step's own
/// write must not leave a pre-step mirror entry behind for its consumers.
///
/// `RLX_WGPU_FORCE_HOST` puts the elementwise chain (and the trailing relu) on
/// the cache-aware `HostOp` path on any adapter; the conv-transpose step is
/// outside that list, so it is the `bypasses_host_cache` door that has to flush
/// the mirror in front of it. The dead intermediates are mirrored before the
/// conv output reuses a slot, so a stale entry served to the trailing relu would
/// show up too.
#[test]
fn a_deferred_host_write_reaches_the_compact_conv_transpose_step() {
    // Must precede the first compile: the lowering reads this to decide which
    // ops take the host fallback.
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let sh = Shape::new(&[1, 4, 1, 8], DType::F32);
    let (c, len) = (4usize, 8usize);
    let xs: Vec<f32> = (0..c * len)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.1)
        .collect();
    let addend: Vec<f32> = (0..c * len).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let ws: Vec<f32> = (0..c * c)
        .map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.25)
        .collect();

    // A 1×1 transposed conv is a channel mix, so its output shape equals `sh`
    // and the plan may give it a dead intermediate's slot.
    let mut g = Graph::new("compact_step_coherence");
    let x = g.input("x", sh.clone());
    let a = g.input("a", sh.clone());
    let w = g.param("w", Shape::new(&[c, c, 1, 1], DType::F32));

    let t0 = g.add_node(Op::Activation(Activation::Relu), vec![x], sh.clone());
    let t1 = g.add_node(Op::Binary(BinaryOp::Add), vec![t0, a], sh.clone());
    let t2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![t1, a], sh.clone());
    let t3 = g.add_node(Op::Binary(BinaryOp::Add), vec![t2, a], sh.clone());
    // t0..t2 are dead from here; the conv output may land on one of their slots.
    let y = g.conv_transpose2d(t3, w, [1, 1], [1, 1], [0, 0], [1, 1], [0, 0], 1);
    let r = g.add_node(Op::Activation(Activation::Relu), vec![y], sh.clone());
    let out = g.add_node(Op::Binary(BinaryOp::Add), vec![r, a], sh.clone());
    g.set_outputs(vec![out]);

    let mut exe = WgpuExecutable::compile(g.clone());
    let kinds = exe.step_kind_counts();
    eprintln!("RLX_WGPU_FORCE_HOST=1: {kinds:?}");
    // The premise, not just the outcome: the conv-transpose step must be the
    // compact one and the elementwise chain plus the trailing relu must be
    // hosted, or the flush this test is about is never in play.
    assert_eq!(
        kinds.get("conv_transpose2d_host").copied().unwrap_or(0),
        1,
        "the transposed conv must be the compact host step; schedule = {kinds:?}"
    );
    assert!(
        kinds.get("host_op").copied().unwrap_or(0) >= 2,
        "the elementwise chain and the trailing relu must be hosted; schedule = {kinds:?}"
    );

    exe.set_param("w", &ws);
    let got = exe.run(&[("x", &xs), ("a", &addend)]);

    let mut cpu = Session::new(Device::Cpu).compile(g);
    cpu.set_param("w", &ws);
    let want = cpu.run(&[("x", &xs), ("a", &addend)]);

    let max_d = common::max_abs_diff(&got[0], &want[0]);
    eprintln!("max abs diff {max_d:.2e}");
    assert!(
        max_d < 1e-4,
        "the compact step staged a stale device value or left one for its \
         consumers (max|Δ| = {max_d:.3e})\n got {:?}\nwant {:?}",
        got[0],
        want[0]
    );
}
