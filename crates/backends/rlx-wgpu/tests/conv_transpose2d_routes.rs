// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d`'s three wgpu steps: the device route's tiled GEMM,
//! printed `conv_transpose2d_tiled`, which serves the decoder family (unit height
//! with no height padding, unit batch, one group, no dilation, `s ≤ k ≤ 2·s` —
//! `conv_transpose2d_tiled_fits` in `lower.rs` is the authority on it,
//! `conv_transpose2d_gather_family.rs` witnesses its forms); the device gather,
//! printed `conv_transpose2d`, which serves every shape outside that family; and
//! the host step, printed `conv_transpose2d_host`.
//!
//! This file used to pin the host step as the op's *only* route: with the other
//! routers zeroed, `step_kind_counts()` had to be `conv_transpose2d_host == 1`.
//! The device route inverts that pin — with the structure-host preference off the
//! router picks one of the two device steps (on an integrated adapter that
//! preference is off on its own, so this is the default there; on a discrete
//! adapter the preference hosts the op instead) — and the boundary between them is
//! pinned here from both sides: this file's decoder geometry is inside the tiled
//! family and takes `conv_transpose2d_tiled`, while two shapes outside it keep the
//! incumbent gather — the same geometry at batch 4 (`n == 1`) and a unit-height
//! form that carries height padding (`ph`, which the gather answers by dropping its
//! single height tap). A move of the route is therefore a move and not a shape
//! losing one. The host route did not disappear either, it is one lever away
//! (`RLX_WGPU_CONV_HOST=1`, or the structure-host preference itself:
//! `RLX_WGPU_FORCE_HOST`, a discrete adapter, a sharded arena).
//!
//! The numeric arm compares against `rlx_cpu::kernels::conv_transpose2d_nchw`,
//! the crate's naive scalar scatter reference kernel — an oracle the runtime
//! parity suite cannot substitute, because the CPU *device* that suite compares
//! against shares the accelerated formulation by default
//! (`conv_transpose2d_forward_gemm`, gated by `RLX_FAST_CONV`, on by default)
//! rather than being an independent reference.
//!
//! Its own test binary: the route is read per compile and the
//! `RLX_WGPU_CONV_HOST` lever is read live by the lowering, so both states live
//! here (as in `vision_norm2d_host_switch.rs`), each with its env mutation
//! scoped and unset afterwards. The second test here drives the same arm's other
//! refusal, so both take `common::route_lock()` — with the levers process-global,
//! a compile next to this file's own `RLX_WGPU_CONV_HOST` window would take the
//! host route and be refused for the wrong reason.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// Decoder geometry: unit height, 1-D-lifted — kernel `[1, 16]`, stride `[1, 8]`.
const C_IN: usize = 8;
const C_OUT: usize = 6;
const L_IN: usize = 96;
const K: usize = 16;
const S: usize = 8;
/// The batch the behavioural active-extent arm compiles at: its reduced `(2, 4)`
/// extent scales `n` to exactly half. It is also the family-boundary pin's shape —
/// `DECODER` is unit batch and this is not, so the two census readings in this file
/// sit on either side of `conv_transpose2d_tiled_fits`'s `n == 1`.
const N: usize = 4;

fn graph(batch: usize) -> Graph {
    let mut g = Graph::new("conv_transpose2d_routes");
    let x = g.input("x", Shape::new(&[batch, C_IN, 1, L_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 1, K], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, K], [1, S], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);
    g
}

/// The decoder geometry as the shared bound walk reads it — the same `[1, K]` /
/// `[1, S]` attribute set `graph()` builds the op with, so
/// `common::conv_transpose2d_pair_error_bound` derives this file's bar exactly as
/// the padded/dilated/grouped family in `conv_transpose2d_gather_family.rs` derives
/// its own.
///
/// The decoder is unit height (`h = kH = sH = 1`, `pH = 0`), so that walk's single
/// height tap lands at `oy = 0` inside `h_out = 1` and what is left of it is the x
/// taps.
const DECODER: common::ConvTranspose2dGeometry = common::ConvTranspose2dGeometry {
    n: 1,
    c_in: C_IN,
    c_out: C_OUT,
    h: 1,
    w: L_IN,
    k: [1, K],
    s: [1, S],
    p: [0, 0],
    d: [1, 1],
    op: [0, 0],
    groups: 1,
};

#[test]
fn the_upsampler_takes_the_device_step_and_falls_back_to_the_host_step_behind_its_lever() {
    let _route = common::route_lock();
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    // State 1: with the structure-host preference off the route is a device step,
    // and on this geometry it is the tiled one — the decoder form is inside the
    // family `conv_transpose2d_tiled_fits` describes. This assertion is where a
    // route pin that used to name the gather (`conv_transpose2d == 1` with exactly
    // these gates zeroed, before that the host step) now names the tiled step.
    let mut device = WgpuExecutable::compile(graph(1));
    let device_kinds = device.step_kind_counts();
    eprintln!("device route: {device_kinds:?}");
    assert_eq!(
        device_kinds
            .get("conv_transpose2d_tiled")
            .copied()
            .unwrap_or(0),
        1,
        "this is a decoder form, so with the structure-host preference off the route must be \
         the tiled device step; schedule = {device_kinds:?}"
    );
    assert!(
        !device_kinds.contains_key("conv_transpose2d"),
        "and the incumbent gather must not also be in the schedule; schedule = {device_kinds:?}"
    );
    assert!(
        !device_kinds.contains_key("conv_transpose2d_host"),
        "that route must not leave the host step in place; schedule = {device_kinds:?}"
    );

    // State 2: the lever still selects the host route — it did not disappear.
    rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
    let mut host = WgpuExecutable::compile(graph(1));
    rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
    let host_kinds = host.step_kind_counts();
    eprintln!("RLX_WGPU_CONV_HOST=1: {host_kinds:?}");
    assert_eq!(
        host_kinds
            .get("conv_transpose2d_host")
            .copied()
            .unwrap_or(0),
        1,
        "the conv-host lever must still select the host step; schedule = {host_kinds:?}"
    );
    assert!(
        !host_kinds.contains_key("conv_transpose2d"),
        "the conv-host lever must not leave the device step in place; schedule = {host_kinds:?}"
    );

    let x: Vec<f32> = (0..C_IN * L_IN)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..C_IN * C_OUT * K)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    let l_out = (L_IN - 1) * S + K;
    let mut want = vec![0f32; C_OUT * l_out];
    rlx_cpu::kernels::conv_transpose2d_nchw(
        &x, &w, &mut want, 1, C_IN, 1, L_IN, C_OUT, 1, l_out, 1, K, 1, S, 0, 0, 1, 1, 1,
    );

    // Both routes compute the operator. The host route keeps its pre-existing
    // witness and its absolute bar: it measures ~6e-8 here.
    host.set_param("w", &w);
    let host_out = host.run(&[("x", &x)]);
    let host_max = max_abs_diff(&host_out[0], &want);
    eprintln!(
        "host route: max abs diff {host_max:.2e} ({} elems)",
        want.len()
    );
    assert!(
        host_max <= 1e-3,
        "host route vs scalar reference: max abs diff {host_max} > 1e-3"
    );

    // The device route's bar is derived from the operands, not absolute.
    device.set_param("w", &w);
    let got = device.run(&[("x", &x)]);
    let measured = max_abs_diff(&got[0], &want);
    let (bound, worst_terms, worst_abs) =
        common::conv_transpose2d_pair_error_bound(&DECODER, &x, &w);
    eprintln!(
        "device route: measured {measured:.2e}, bound {bound:.2e} (worst terms {worst_terms}, \
         worst Σ|terms| {worst_abs:.3}), ratio {:.3} of bound ({} elems)",
        measured as f64 / bound,
        want.len()
    );
    assert!(
        measured as f64 <= bound,
        "device route vs scalar reference: measured {measured:e} > derived bound {bound:e}"
    );

    // The device step is safe under an active extent: it scales its batch in the
    // uniform pre-pass, exactly like its 3-D sibling. A step that was not would
    // silently veto the mode for the whole graph.
    assert!(
        device.test_all_steps_safe_for_active_extent(),
        "the device step must honour (not veto) the active-extent mode; schedule = {device_kinds:?}"
    );

    // Behavioural arm: the same shape at batch 4, run at the full extent and then
    // under a `(2, 4)` extent, whose uniform pre-pass rewrites `n` to 2. The two
    // batches that remain are the same work in the same arrangement, so their
    // outputs must agree bit for bit.
    let mut batched = WgpuExecutable::compile(graph(N));

    // The family boundary from the other side, read before the arm below runs:
    // batch 4 is outside the tiled form's family (`conv_transpose2d_tiled_fits`
    // wants `n == 1`), so the shape the change excludes must keep the incumbent
    // gather — a shape outside a fast path's family is a shape whose route moves
    // back, not one that loses its route.
    let batched_kinds = batched.step_kind_counts();
    eprintln!("batch {N} route: {batched_kinds:?}");
    assert_eq!(
        batched_kinds.get("conv_transpose2d").copied().unwrap_or(0),
        1,
        "batch {N} is outside the tiled family, so it must keep the incumbent device \
         gather; schedule = {batched_kinds:?}"
    );
    assert!(
        !batched_kinds.contains_key("conv_transpose2d_tiled")
            && !batched_kinds.contains_key("conv_transpose2d_host"),
        "and neither the tiled step nor the host step may serve it; schedule = {batched_kinds:?}"
    );

    // The family's other boundary, on the height axis: unit height and a single
    // height tap, but with height padding — `h_out = 1` only allows `ph = 1` at
    // `output_padding[0] = 2`, and that is exactly the form where the gather drops
    // its one height tap (it lands outside the input) while the tiled kernel, which
    // knows nothing of the height axis, would still sum the width walk. No terms
    // land on such a shape, so this is a census pin and nothing else.
    let mut padded = Graph::new("conv_transpose2d_height_padding");
    let px = padded.input("x", Shape::new(&[1, C_IN, 1, L_IN], DType::F32));
    let pw = padded.param("w", Shape::new(&[C_IN, C_OUT, 1, 1], DType::F32));
    let py = padded.conv_transpose2d(px, pw, [1, 1], [1, 1], [1, 0], [1, 1], [2, 0], 1);
    padded.set_outputs(vec![py]);
    let padded_kinds = WgpuExecutable::compile(padded).step_kind_counts();
    eprintln!("height-padded (ph = 1) route: {padded_kinds:?}");
    assert_eq!(
        padded_kinds.get("conv_transpose2d").copied().unwrap_or(0),
        1,
        "a height-padded form is outside the tiled family, so it must keep the incumbent device \
         gather; schedule = {padded_kinds:?}"
    );
    assert!(
        !padded_kinds.contains_key("conv_transpose2d_tiled")
            && !padded_kinds.contains_key("conv_transpose2d_host"),
        "and neither the tiled step nor the host step may serve it; schedule = {padded_kinds:?}"
    );

    batched.set_param("w", &w);
    let x4: Vec<f32> = (0..N * C_IN * L_IN)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.05)
        .collect();
    let full = batched.run(&[("x", &x4)]).remove(0);
    batched.set_active_extent(Some((2, N)));
    let reduced = batched.run(&[("x", &x4)]).remove(0);
    batched.set_active_extent(None);
    let prefix = 2 * C_OUT * l_out;
    assert_eq!(
        &reduced[..prefix],
        &full[..prefix],
        "the (2, {N}) extent must reproduce the full-extent run's first two batches"
    );
    eprintln!(
        "active extent: compared the first {prefix} elements (2 of {N} batches × {C_OUT} c_out × \
         {l_out} width) of the (2, {N}) run against the full-extent run, bit for bit"
    );
}

/// The device arm's other refusal: `kernel_size` against the weight's own taps.
///
/// The IR validates the weight's *channel* dims against the input's but not its taps,
/// so `conv_transpose2d(x, w, [4, 4], …)` over a `[C_IN, C_OUT, 3, 3]` weight is a
/// graph inference accepts — and the gather indexes the weight with the op's
/// `kernel_size`, so walking 4×4 taps over a 3×3 weight reads across the operand
/// inside the binding instead of failing. Only the device arm is asked: the host
/// route sizes its staging from the same declared attribute, so an over-declared tap
/// count makes it read neighbouring arena bytes just as silently — its own
/// pre-existing behaviour, which this route reports rather than changes, and so
/// nothing here pins it.
#[test]
fn a_declared_kernel_size_the_weight_disagrees_with_is_refused_on_the_device_route() {
    let _route = common::route_lock();
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    let mut g = Graph::new("conv_transpose2d_routes");
    let x = g.input("x", Shape::new(&[1, C_IN, 1, L_IN], DType::F32));
    let w = g.param("w", Shape::new(&[C_IN, C_OUT, 3, 3], DType::F32));
    // 4×4 declared, 3×3 on the weight. Only the taps disagree — the weight's channel
    // dims are the ones the IR checks, and those are right.
    let y = g.conv_transpose2d(x, w, [4, 4], [1, S], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);

    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(g)));
    let Err(payload) = caught else {
        panic!("a kernel size the weight disagrees with was accepted instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    eprintln!("kernel-size refusal: {msg}");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (Op::ConvTranspose2d kernel size)"),
        "the refusal must carry this op's own guard template; got: {msg}"
    );
    assert!(
        msg.contains("declares 4x4 taps and the weight is 3x3 per channel"),
        "the refusal must name both the declared taps and the weight's own; got: {msg}"
    );
    assert!(
        msg.contains("Declare the kernel size the weight really has."),
        "the refusal must name the way out it offers — the graph's own attribute; got: {msg}"
    );
}
