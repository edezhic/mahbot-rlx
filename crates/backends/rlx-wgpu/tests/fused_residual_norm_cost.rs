// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The folded residual+norm steps (`fused_residual_ln`,
//! `fused_residual_rms_norm`, the dual-output `…_tee`, `ada_layer_norm`) used to
//! reduce a whole outer row in ONE thread, which made them the largest
//! identified decode term. This harness is the marginal-cost evidence for the
//! rewrite to one workgroup per row: it pins the cost of the folded step against
//! the dispatching it replaces — `Add` + `RmsNorm`, both already row-parallel —
//! at the decode width of a real body (`H = 2048`, `rows = 1`), and at a
//! prefill-shaped `rows`.
//!
//! **Why the arms are separate chains rather than one graph with a fold toggled.**
//! The upstream `FuseResidualRmsNorm` requires the residual sum to have exactly
//! one consumer, and the tee requires it to have a second consumer *after* the
//! norm — so the same graph cannot be compiled both folded and unfolded without
//! editing the pipeline. Instead each piece is its own dependency chain of `n`
//! steps, and the pair is the *sum* of the `Add` and `RmsNorm` marginals:
//!
//!   * `add`    — `h ← h + d` per step (one Binary dispatch);
//!   * `norm`   — `h ← rms_norm(h)` per step (one row-parallel LayerNorm);
//!   * `folded` — `h ← fused_residual_rms_norm(h, d)` per step (one dispatch —
//!     the kernel under test);
//!   * `tee`    — a transformer layer's shape, `s ← h + d; n ← rms_norm(s);
//!     h ← n + s`, per step (one tee + one Binary — the decoder hot path).
//!
//! The tee's own kernel cost is `tee − add`, so every comparison below is
//! between kernels that each move the same bytes.
//!
//! **Marginal, not absolute.** `run()` includes submission, the device wait and
//! the readback — a fixed floor that dominates a single reading. Each arm is
//! therefore timed at two chain lengths in one submission, with the *same*
//! inputs and arena, and the figure reported is `(long − short) / (LONG − SHORT)`
//! over the minimum of `REPS` alternated pairs (see `Arm::per_step_us`).
//!
//! `RLX_WGPU_HOST_NORM=1` or a host-routing adapter makes this a measurement of
//! the host fallback; the arm step counts are asserted so that cannot pass
//! silently. Run the thermal gate first (`just throttle`).

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::Tick;
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Hidden width of one decode row — the owner's rig measures there.
const H: usize = 2048;
const EPS: f32 = 1e-5;
/// Chain lengths: the marginal is the slope between them, so the fixed
/// submission/readback floor cancels.
const SHORT: usize = 4;
const LONG: usize = 52;
/// Timed runs per length; the reported statistic is the minimum of each, the
/// run with the least interference on it.
const REPS: usize = 15;
/// Decode, then a prefill-shaped batch — the same kernels serve both.
const ROWS: [usize; 2] = [1, 8];

fn shape(rows: usize) -> Shape {
    Shape::new(&[rows, H], F)
}

/// `h ← h + d`, `n` times. One Binary dispatch per step.
fn add_chain(rows: usize, n: usize) -> Graph {
    let mut g = Graph::new("add_chain");
    let x = g.input("x", shape(rows));
    let d = g.param("d", shape(rows));
    let mut h = x;
    for _ in 0..n {
        h = g.add(h, d);
    }
    g.set_outputs(vec![h]);
    g
}

/// `h ← rms_norm(h)`, `n` times. One LayerNorm dispatch per step.
fn norm_chain(rows: usize, n: usize) -> Graph {
    let mut g = Graph::new("norm_chain");
    let x = g.input("x", shape(rows));
    let ga = g.param("g", Shape::new(&[H], F));
    let be = g.param("b", Shape::new(&[H], F));
    let mut h = x;
    for _ in 0..n {
        h = g.rms_norm(h, ga, be, EPS);
    }
    g.set_outputs(vec![h]);
    g
}

/// `h ← fused_residual_rms_norm(h, d)`, `n` times — the folded pair.
fn folded_chain(rows: usize, n: usize) -> Graph {
    let mut g = Graph::new("folded_chain");
    let x = g.input("x", shape(rows));
    let d = g.param("d", shape(rows));
    let ga = g.param("g", Shape::new(&[H], F));
    let be = g.param("b", Shape::new(&[H], F));
    let mut h = x;
    for _ in 0..n {
        h = g.fused_residual_rms_norm(h, d, None, ga, be, EPS, shape(rows));
    }
    g.set_outputs(vec![h]);
    g
}

/// A transformer layer's residual stream `n` times over: `s ← h + d`,
/// `n ← rms_norm(s)`, `h ← n + s`. The sum has a second consumer *after* the
/// norm, which is what the tee exists for, so each step is one tee dispatch
/// (carrying the Add) plus the plain Binary that adds the norm back.
fn tee_chain(rows: usize, n: usize) -> Graph {
    let mut g = Graph::new("tee_chain");
    let x = g.input("x", shape(rows));
    let d = g.param("d", shape(rows));
    let ga = g.param("g", Shape::new(&[H], F));
    let be = g.param("b", Shape::new(&[H], F));
    let mut h = x;
    for _ in 0..n {
        let s = g.add(h, d);
        let nn = g.rms_norm(s, ga, be, EPS);
        h = g.add(nn, s);
    }
    g.set_outputs(vec![h]);
    g
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.2
        })
        .collect()
}

/// One timed `run`, in nanoseconds. `Tick`, not `Instant`: the per-step figures
/// are microseconds and the tree mandates the tick at that scale.
fn nanos(exe: &mut WgpuExecutable, x: &[f32]) -> u128 {
    let start = Tick::now();
    let _ = exe.run(&[("x", x)]);
    u128::from(Tick::now().elapsed_ns(start))
}

/// One measured step, as a chain of `LONG` steps and of `SHORT` steps over the
/// same inputs and arena. Their difference is the marginal cost of `LONG−SHORT`
/// steps, which is a quantity the fixed submission/readback floor does not
/// enter.
struct Arm {
    label: &'static str,
    long: WgpuExecutable,
    short: WgpuExecutable,
    /// Minimum over `REPS` runs of each chain length, in nanoseconds.
    min_long: u128,
    min_short: u128,
}

impl Arm {
    /// `kinds` names the step kinds the chain dispatches exactly once per step —
    /// asserted at both chain lengths so an arm that silently stopped
    /// dispatching the kernel it names (host routing, an upstream pass fusing
    /// the pattern away) cannot be timed as if it had.
    fn new(
        label: &'static str,
        rows: usize,
        build: fn(usize, usize) -> Graph,
        kinds: &[&'static str],
    ) -> Self {
        let mk = |n: usize| {
            let mut exe = WgpuExecutable::compile(build(rows, n));
            exe.set_param("d", &fill(rows * H, 11));
            exe.set_param("g", &fill(H, 12));
            exe.set_param("b", &fill(H, 13));
            let counts = exe.step_kind_counts();
            for kind in kinds {
                let got = counts.get(*kind).copied().unwrap_or(0);
                assert_eq!(
                    got, n,
                    "{label} ({n} steps): expected {n} `{kind}` steps, got {got} \
                     — the arm is not measuring the kernel it names ({counts:?})"
                );
            }
            exe
        };
        Self {
            label,
            long: mk(LONG),
            short: mk(SHORT),
            min_long: u128::MAX,
            min_short: u128::MAX,
        }
    }

    /// One untimed `run` of each chain length — first touch of the arena and the
    /// pipeline's first dispatch of each step kind.
    fn warm(&mut self, x: &[f32]) {
        let _ = self.long.run(&[("x", x)]);
        let _ = self.short.run(&[("x", x)]);
    }

    fn sample(&mut self, x: &[f32], short_first: bool) {
        let (l, s) = if short_first {
            let s = nanos(&mut self.short, x);
            let l = nanos(&mut self.long, x);
            (l, s)
        } else {
            let l = nanos(&mut self.long, x);
            let s = nanos(&mut self.short, x);
            (l, s)
        };
        self.min_long = self.min_long.min(l);
        self.min_short = self.min_short.min(s);
    }

    /// Marginal microseconds per step.
    fn per_step_us(&self) -> f64 {
        assert!(
            self.min_long > self.min_short,
            "{}: the long chain ({} ns) is not slower than the short one ({} ns) — \
             too noisy to difference",
            self.label,
            self.min_long,
            self.min_short
        );
        (self.min_long - self.min_short) as f64 / (LONG - SHORT) as f64 / 1_000.0
    }
}

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn folded_residual_norm_marginal_cost() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    common::refuse_under_run_report();
    eprintln!(
        "wgpu folded residual+norm: H={H}, chains {SHORT} vs {LONG} steps, {REPS} \
         runs each, minimum per length; marginal = (long − short)/(LONG − SHORT)"
    );
    for rows in ROWS {
        // One arm per measured step; `want` is the per-`LONG` step count, scaled
        // to each chain length. Interleaved round-robin so a slow drift cannot
        // favour one arm, and the short/long order flips with the round.
        let mut arms = [
            Arm::new("add", rows, add_chain, &["binary"]),
            Arm::new("norm", rows, norm_chain, &["layer_norm"]),
            Arm::new("folded", rows, folded_chain, &["fused_residual_rms_norm"]),
            Arm::new("tee", rows, tee_chain, &["fused_residual_ln_tee", "binary"]),
        ];
        let x = fill(rows * H, 7);
        for arm in arms.iter_mut() {
            arm.warm(&x);
        }
        for i in 0..REPS {
            for k in 0..arms.len() {
                let a = if i % 2 == 0 { k } else { arms.len() - 1 - k };
                arms[a].sample(&x, i % 2 == 1);
            }
        }
        let [add, norm, folded, tee] = arms.each_ref();
        let (a, n, f, t) = (
            add.per_step_us(),
            norm.per_step_us(),
            folded.per_step_us(),
            tee.per_step_us(),
        );
        eprintln!("\n  rows={rows}: marginal per step, µs");
        eprintln!("    residual Add (1 dispatch)           {a:7.1}");
        eprintln!("    row-parallel RmsNorm (1 dispatch)   {n:7.1}");
        eprintln!("    the pair it replaces (2 dispatches) {:7.1}", a + n);
        eprintln!("    folded fused_residual_rms_norm      {f:7.1}");
        eprintln!(
            "    tee step, decoder hot path (1 of 2 dispatches)  {t:7.1}  → \
             tee kernel ≈ {:7.1}",
            t - a
        );
    }
}
