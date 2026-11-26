// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Head-major attention runs the same arithmetic it ran before, a query row at a
//! time on the pool instead of the whole head on the caller's thread: at each
//! geometry the MahBot product emits (tagger, ASR encoder tail, decoder prefill,
//! decode steps, the wake window) the pooled result is compared with the same route
//! on one thread, and the lane block with the scalar loop, and each pair must be the
//! same bits. Bit-identity is the bar because the product's
//! transcripts are greedy-decoded from these graphs, and the reassociating routes —
//! measurably faster still — move words on the long recognition row.
//!
//! The route's inner loops run a block of lanes at a time, and both threading arms
//! share them, so `the_lane_kernels_are_the_scalar_loops` is what holds the lane
//! block to the scalar loop's bits; `RLX_CPU_ATTN_HEAD_MAJOR_SCALAR` keeps that
//! scalar loop reachable as the reference. `RLX_CPU_ATTN_HEAD_MAJOR_SERIAL` puts
//! head-major back on one thread, so both threading arms are levers in one binary
//! rather than two builds. That each switch selects its arm is a counter test in the
//! lib unit tests (`thunk::exec_dispatch::head_major_switch_tests`), not here.
//!
//! An f64 reference at two geometries is the independent check: two arms agreeing
//! cannot hide an offset both of them share.
//!
//! Every test here holds [`arm`]: both switches are process-global (see [`serial`])
//! and the binary's tests run in parallel, so without the lock one test would flip
//! an arm under another.
mod common;

use std::sync::{Mutex, MutexGuard};

use common::attn::{CASES, Case, Run, graph, keyed, scalar_body, serial};
use rlx_ir::op::MaskKind;

/// Held for the duration of every test in this binary: both switches are
/// process-global, so the tests take turns rather than racing on them.
fn arm() -> MutexGuard<'static, ()> {
    static ARM: Mutex<()> = Mutex::new(());
    // A one-worker pool runs both arms inline, where the comparison is vacuous.
    assert!(
        rlx_cpu::pool::num_threads() > 1,
        "a one-worker pool runs both head-major arms inline"
    );
    ARM.lock().unwrap_or_else(|e| e.into_inner())
}

/// One head-major call under the arms `(one_thread, scalar)` select.
fn run_arms(c: &Case, one_thread: bool, scalar: bool) -> Vec<f32> {
    let g = graph(c, true);
    let mut r = Run::new(&g, keyed);
    serial(one_thread);
    scalar_body(scalar);
    r.call();
    r.output(&g)
}

/// Bit-identical, not close: the two arms run one arithmetic, so any difference at all
/// is a defect.
fn assert_same_bits(want: &[f32], got: &[f32], name: &str) {
    assert_eq!(want.len(), got.len(), "{name}: output length");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            w.to_bits(),
            g.to_bits(),
            "{name}: element {i} is {g:e}, the reference gives {w:e}"
        );
    }
}

#[test]
fn the_pool_is_the_one_thread_route() {
    let _arm = arm();
    println!("pool width {}", rlx_cpu::pool::num_threads());
    for c in CASES {
        let one_thread = run_arms(c, true, false);
        let pooled = run_arms(c, false, false);
        assert_same_bits(&one_thread, &pooled, c.name);
        // A second pooled call: a race between tasks would have to land on the same
        // bits twice to get past here.
        assert_same_bits(&pooled, &run_arms(c, false, false), c.name);
        println!("{:<18} identical over {} elements", c.name, pooled.len());
    }
}

/// The two inner-loop arms must agree bit for bit at every geometry the product
/// emits. This is the comparison [`the_pool_is_the_one_thread_route`] can
/// no longer make: once both threading arms share the lane block, a one-bit drift
/// in it would pass every gate here — the f64 reference is 1e-5 loose and the
/// product's witnesses are not gates. The two decode rows have a single query row, so
/// both arms take the scalar gate there and that comparison is the route's, not the
/// lane block's.
#[test]
fn the_lane_kernels_are_the_scalar_loops() {
    let _arm = arm();
    for c in CASES {
        let want = run_arms(c, false, true);
        let got = run_arms(c, false, false);
        assert_same_bits(&want, &got, c.name);
    }
    println!(
        "{} geometries identical across the two inner-loop arms",
        CASES.len()
    );
}

/// The cap is applied per score, after the mask and before the softmax, and both
/// arms apply it: the capped row keeps its values, and the cap still bites.
#[test]
fn keeps_the_logit_softcap() {
    let _arm = arm();
    let capped = *CASES
        .iter()
        .find(|c| c.softcap.is_some())
        .expect("a capped row in the table");
    let one_thread = run_arms(&capped, true, false);
    let pooled = run_arms(&capped, false, false);
    let uncapped = Case {
        softcap: None,
        ..capped
    };
    let bites = common::max_rel_err(&pooled, &run_arms(&uncapped, false, false));
    println!("{:<18} cap effect {bites:.3e}", capped.name);
    assert_same_bits(&one_thread, &pooled, capped.name);
    assert!(
        bites > 1e-2,
        "the logit soft-cap did not reach a head-major input ({bites:e})"
    );
}

/// An f64 reference at two geometries: two arms agreeing cannot hide an offset both
/// of them share. Loose by design — it catches a wrong offset, not a rounding.
#[test]
fn matches_an_f64_reference() {
    let _arm = arm();
    for c in CASES
        .iter()
        .filter(|c| c.name == "tagger 252" || c.name == "prefill 111")
    {
        let got = run_arms(c, false, false);
        let (b, s, ks, h, kv, d) = (c.batch, c.seq, c.kv_seq, c.heads, c.kv_heads, c.head_dim);
        let scale = 1.0 / (d as f64).sqrt();
        let q = keyed("q", b * h * s * d);
        let kk = keyed("k", b * kv * ks * d);
        let vv = keyed("v", b * kv * ks * d);
        let mask = keyed("mask", b * ks);
        let mut worst_d = 0f64;
        for bi in 0..b {
            for hi in 0..h {
                let kv_hi = hi * kv / h;
                for qi in 0..s {
                    let qrow = &q[(bi * h + hi) * s * d + qi * d..][..d];
                    let sc: Vec<f64> = (0..ks)
                        .map(|ki| {
                            let live =
                                !matches!(c.mask, MaskKind::Custom) || mask[bi * ks + ki] > 0.0;
                            let causal = !matches!(c.mask, MaskKind::Causal)
                                || ki <= qi + ks.saturating_sub(s);
                            if !live || !causal {
                                return f64::NEG_INFINITY;
                            }
                            let krow = &kk[(bi * kv + kv_hi) * ks * d + ki * d..][..d];
                            qrow.iter()
                                .zip(krow)
                                .map(|(&a, &c)| a as f64 * c as f64)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let max = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = sc.iter().map(|&x| (x - max).exp()).collect();
                    let sum: f64 = ex.iter().sum();
                    for di in 0..d {
                        let want: f64 = (0..ks)
                            .map(|ki| {
                                let vrow = &vv[(bi * kv + kv_hi) * ks * d + ki * d..][..d];
                                ex[ki] / sum * vrow[di] as f64
                            })
                            .sum();
                        let have = got[(bi * h + hi) * s * d + qi * d + di] as f64;
                        worst_d = worst_d.max((have - want).abs());
                    }
                }
            }
        }
        println!("{:<18} vs f64 {worst_d:.3e}", c.name);
        assert!(
            worst_d < 1e-5,
            "{}: {worst_d:e} from the f64 reference",
            c.name
        );
    }
}

/// An attention node nothing reads is still scheduled, with its own slot: it must
/// write only that slot and leave the live output bit-identical, since this route
/// addresses its output through a pointer rather than by indexing.
#[test]
fn an_extra_attention_writes_only_its_own_slot() {
    let _arm = arm();
    let c = *CASES
        .iter()
        .find(|c| c.name == "prefill 47")
        .expect("a prefill row in the table");
    let mut g = graph(&c, true);
    let (b, s, h, d) = (c.batch, c.seq, c.heads, c.head_dim);
    let input = |g: &rlx_ir::Graph, want: &str| {
        g.nodes()
            .iter()
            .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == want))
            .map(|n| n.id)
            .expect("an input of that name")
    };
    let (q, k, v) = (input(&g, "q"), input(&g, "k"), input(&g, "v"));
    // A second attention nothing reads, still in the schedule.
    let orphan = g.attention_kind_opts(
        q,
        k,
        v,
        h,
        d,
        MaskKind::None,
        rlx_ir::Shape::new(&[b, h, s, d], rlx_ir::DType::F32),
        None,
        None,
    );
    let mut r = Run::new(&g, keyed);
    assert!(
        rlx_opt::memory::plan_memory(&g).schedule.contains(&orphan),
        "the extra attention left the schedule, so nothing was exercised"
    );
    assert_ne!(
        r.offset(orphan),
        usize::MAX,
        "the planner left the orphan without a slot, so the empty-output guard is \
         what this test would be exercising"
    );
    r.call();
    assert_same_bits(
        &run_arms(&c, false, false),
        &r.output(&g),
        "extra attention",
    );
}

/// A node the planner left without a slot reaches the dispatcher as an empty
/// output slice, and the accessor's contract for it is that writes are discarded.
/// The slot is filled with a sentinel first, so a route that wrote zeros would fail
/// here as well. Both layouts must take the no-op: the head-major route addresses
/// its output through a raw pointer, so without the guard it would write outside
/// the arena.
#[test]
fn a_node_with_no_slot_writes_nothing() {
    const SENTINEL: f32 = -7.5;
    let _arm = arm();
    let c = *CASES
        .iter()
        .find(|c| c.name == "prefill 47")
        .expect("a prefill row in the table");
    for head_major in [true, false] {
        let g = graph(&c, head_major);
        let mut r = Run::new(&g, keyed);
        r.fill_output(&g, SENTINEL);
        r.detach_attention_output();
        r.call();
        let out = r.output(&g);
        let moved = out.iter().filter(|&&v| v != SENTINEL).count();
        assert_eq!(
            moved, 0,
            "{head_major}: the route wrote {moved} element(s) of a slot it was given no \
             arena for"
        );
    }
}
