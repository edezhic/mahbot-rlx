// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The residual→norm tee must fire on transformer blocks, and must not change
//! the numbers.**
//!
//! `FuseResidualLN` / `FuseResidualRmsNorm` both require the residual sum to
//! have exactly one consumer. A transformer block gives it two — the norm and
//! the next residual (`h += attn; n = norm(h); h += ffn(n)`) — so the fusion
//! declines on every layer. `detect_residual_ln_tee_pattern` recovers that case
//! by emitting one step that writes the sum AND the normalised result.
//!
//! It was LayerNorm-only, which misses every Llama-class model. Widening it to
//! RmsNorm means both norms now share one kernel that selects its scaling on an
//! `is_rms` flag — and a mode flag on a shared kernel is exactly where a silent
//! numeric divergence hides. So this asserts both halves:
//!
//! * the tee actually fires (a dispatch-count fact, machine-independent), and
//! * wgpu still matches the CPU reference elementwise.
//!
//! Without the second check the first would happily pass on a kernel that
//! computes LayerNorm for an RmsNorm graph.
//!
//! The second test runs the same shape at **two rows** and at a width that is
//! **not** a multiple of the 64-wide workgroup (130 = 2·64 + 2, so the last
//! stride over the row is partial) under a large per-row **DC offset**, and the
//! first runs at a **decode-shaped width** (512 = 8 whole strides per lane) as
//! well. The prior scalar one-thread-per-row kernel had no strides and no
//! cross-lane tree to get wrong; this locks both, that the two rows stay
//! independent, and the DC offset is what makes a one-pass `E[x²] − E[x]²`
//! rewrite of the stable two-pass form fail the bar.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

const F: DType = DType::F32;
const H: usize = 128;
/// Partial-stride width: 130 = 2·64 + 2. `H` is an exact multiple of the
/// workgroup, so only this case can see a wrong (or missing) tail stride.
const H_STRIDED: usize = 130;
/// Decode-shaped width: 512 = 8 whole strides per lane, the shape a real
/// transformer body runs this step at.
const H_WIDE: usize = 512;
/// Per-row DC offset for that case: at 1e3 the one-pass identity's `ulp(Σx²)`
/// error (~0.06 for a 130-wide row) is a ~1% error on a variance of ~6 — small
/// enough that the row stays in the norm's linear range, large enough that the
/// substitution misses the case's bar by 23x.
const DC: f32 = 1_000.0;
const BLOCKS: usize = 6;

/// `BLOCKS` residual blocks whose sum feeds BOTH the norm and the next
/// residual — the shape the single-consumer guard rejects.
fn block_graph(rms: bool, rows: usize, h: usize) -> (Graph, Vec<(String, usize)>) {
    let mut g = Graph::new("resid_norm");
    let x = g.input("x", Shape::new(&[rows, h], F));
    let mut cur = x;
    let mut names = Vec::new();
    for l in 0..BLOCKS {
        let (gn, bn, wn) = (format!("g{l}"), format!("b{l}"), format!("w{l}"));
        let gam = g.param(&gn, Shape::new(&[h], F));
        let bet = g.param(&bn, Shape::new(&[h], F));
        let w = g.param(&wn, Shape::new(&[h, h], F));
        names.push((gn, h));
        names.push((bn, h));
        names.push((wn, h * h));
        let n = if rms {
            g.rms_norm(cur, gam, bet, 1e-5)
        } else {
            g.layer_norm(cur, gam, bet, -1, 1e-5, Shape::new(&[rows, h], F))
        };
        let proj = g.matmul(n, w, Shape::new(&[rows, h], F));
        cur = g.add(cur, proj);
    }
    g.set_outputs(vec![cur]);
    (g, names)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.5
        })
        .collect()
}

/// A row that is `DC` plus a small deterministic ramp — the cancellation-prone
/// regime (a row whose mean dominates its spread), i.e. pre-norm residual
/// streams with an input-projection bias or a positional table. The ramp is
/// wide enough (±4 on a mean of 1000) that the row is inside the norm's linear
/// range rather than at its inverse-square-root sensitivity wall, which is what
/// keeps the CPU reference's own sum error (see the case's tolerance) small
/// while the one-pass identity still loses ~1% of the variance to `ulp(Σx²)`.
fn dc_ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| DC + (((i + seed) % 17) as f32 - 8.0) * 0.5)
        .collect()
}

fn run_cpu(rms: bool, rows: usize, h: usize, x: &[f32]) -> Vec<f32> {
    let (g, names) = block_graph(rms, rows, h);
    let mut c = Session::new(Device::Cpu).compile(g);
    for (n, len) in &names {
        c.set_param(n, &fill(*len, n.len() as u64 + 7));
    }
    c.finalize_params();
    c.run(&[("x", x)]).remove(0)
}

/// One norm at one shape/input: device vs CPU elementwise, plus the route fact
/// that the tee is what ran.
fn check(rms: bool, rows: usize, h: usize, x: &[f32], tol: f32, case: &str) {
    let label = if rms { "RmsNorm" } else { "LayerNorm" };
    let (g, names) = block_graph(rms, rows, h);
    let mut exe = WgpuExecutable::compile(g);
    for (n, len) in &names {
        exe.set_param(n, &fill(*len, n.len() as u64 + 7));
    }
    let got = exe.run(&[("x", x)])[0].clone();

    // Numerics first — it applies on every adapter, however the ops routed.
    // `max_abs_diff` censuses finiteness before it folds (a hand-rolled
    // `f32::max` silently drops a NaN), and this comparison is the one place a
    // broken reduction would show up.
    let want = run_cpu(rms, rows, h, x);
    let maxd = common::max_abs_diff(&got, &want);
    assert!(
        maxd < tol,
        "{case}: {label} at h={h}: wgpu differs from CPU by {maxd:.3e} (bar {tol:.1e}) \
         — the tee kernel is computing the wrong norm (check the `is_rms` selector, \
         the tree reduction's tail stride, or the two-pass variance)"
    );

    let counts = exe.step_kind_counts();
    let tees = counts.get("fused_residual_ln_tee").copied().unwrap_or(0);
    let binaries = counts.get("binary").copied().unwrap_or(0);
    let norms = counts.get("layer_norm").copied().unwrap_or(0);
    eprintln!(
        "  {case}: {label} rows={rows} h={h}: {tees} tee, {binaries} binary, {norms} norm, \
         maxd={maxd:.3e}"
    );

    // The tee is a GPU-path optimisation. Discrete Vulkan/DX12 route
    // elementwise ops and norms to the host entirely
    // (`wgpu_prefer_host_fallback`), so there is no GPU step to fuse and
    // ZERO of each is the correct outcome — not a regression. Asserting a
    // tee count unconditionally fails there for the wrong reason.
    if tees == 0 && binaries == 0 && norms == 0 {
        eprintln!(
            "  {case}: {label} rows={rows} h={h}: ops are host-routed on this adapter — tee N/A"
        );
        return;
    }
    assert!(
        tees >= BLOCKS - 1,
        "{case}: {label} rows={rows} h={h}: only {tees} of {BLOCKS} blocks used the tee \
         ({binaries} binary + {norms} norm steps remain) — the multi-consumer residual \
         pattern is not being recognised, so every block pays a separate \
         Add dispatch"
    );
}

#[test]
fn the_tee_fires_for_both_norms_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let x = fill(H, 4);
    for rms in [false, true] {
        check(rms, 1, H, &x, 1e-4, "tee blocks");
    }
    // Bar 1e-3, not the 1e-4 of the h=128 case: this graph is a 6-block matmul
    // chain, and at a 512-wide dot the CPU's sequential f32 sum and the kernel's
    // differ by more than the norm does — measured 1.53e-4 here against 1.9e-5 at
    // h=128. The rewritten kernel itself is exact at this width (the folded
    // single-step cases in `basic.rs` match an f64 reference to 2.4e-7), so this
    // case is here for the stride/lane mapping, where a defect costs orders of
    // magnitude more.
    let wide = fill(H_WIDE, 11);
    for rms in [false, true] {
        check(rms, 1, H_WIDE, &wide, 1e-3, "decode width");
    }
}

/// The striding tail, per-row independence and the two-pass form, on the folded step.
///
/// Two rows, so the whole-row reduction has to keep its rows apart as well as
/// stride correctly: the rewrite moved the row decode from a per-invocation
/// `gid.x` to `wid.x + wid.y · ngs.x`, which has to land on the same row in
/// every pass. Bar 2e-3, not the 1e-4 of the zero-mean case: the CPU side is a
/// plain sequential f32 sum over a row whose mean is ~1e3, so its own mean error
/// is amplified by `inv_std` in every one of the 6 blocks — measured 6.1e-4 here
/// (RmsNorm 1.2e-4), against 1.9e-5 when the same shape has no DC offset. A
/// one-pass `E[x²] − E[x]²` variance substitution misses this bar by 23x
/// (measured 4.6e-2 with the substitution in place): `ulp(Σx²)` is ~0.06 at
/// DC=1e3, which is what the stable two-pass form exists to avoid.
#[test]
fn the_folded_step_at_a_partial_stride_with_dc_offset_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let x = dc_ramp(2 * H_STRIDED, 4);
    for rms in [false, true] {
        check(rms, 2, H_STRIDED, &x, 2e-3, "strided width + DC offset");
    }
}
