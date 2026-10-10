// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::ConvTranspose2d` for wgpu arenas.
//!
//! A transposed conv is a forward conv's input gradient, so this route is the
//! compact-staging, BLAS GEMM + col2im one this crate already runs for that
//! operation ([`crate::conv_bwd_host::run_conv2d_backward_input`]) with the channel
//! roles swapped: the conv-transpose input plays `dy`, its output plays `dx`, and a
//! `[C_in, C_out/g, kH, kW]` weight already is the `[C_out, C_in/g, kH, kW]`
//! layout that route declares. Only the tensors this operator touches cross the
//! bus; the whole-arena mirror is left to the geometry the swap cannot mean
//! exactly (`accelerated_route_applies`).
//!
//! The executor flushes the host mirror before a host step that is not one of
//! its cache-aware kinds, which is where this route reads its operands from —
//! keeping it out of that list is what stops the staging from reading a stale
//! value a previous host step left in the mirror.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

/// Can the swapped conv-backward-input mapping serve this geometry exactly?
///
/// The reference kernel this replaces groups on `C_in / groups` and returns
/// zeros outright when `groups == 0` or `C_in < groups`; the swapped route groups
/// on the opposite channel count, so gating on `groups >= 1` and on divisibility
/// of *both* counts is what keeps those degenerate geometries producing the zeros
/// they produce today instead of a dense conv or a different channel mapping.
/// A non-divisible group count is not a shape either route computes faithfully:
/// the reference's own scatter indexes its channel slices out of range (panicking)
/// or silently covers non-group-consistent channels, so it keeps the reference's
/// behaviour rather than silently redefining the operator.
///
/// The `col` panel below is capped exactly like this crate's device im2col route
/// caps its own, so a large general 2-D shape cannot turn the compact route into
/// a memory regression.
fn accelerated_route_applies(
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    groups: usize,
) -> bool {
    groups >= 1
        && c_in.is_multiple_of(groups)
        && c_out.is_multiple_of(groups)
        && col_panel_bytes(c_out, h, w_in, kh, kw, groups)
            <= crate::backend::CONV_IM2COL_MAX_COL_BYTES
}

/// `col` panel the swapped route materializes: `(C_out/groups)·kH·kW` rows of
/// `H_in·W_in` f32 columns.
fn col_panel_bytes(
    c_out: usize,
    h: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    groups: usize,
) -> u64 {
    (c_out / groups)
        .saturating_mul(kh)
        .saturating_mul(kw)
        .saturating_mul(h)
        .saturating_mul(w_in)
        .saturating_mul(4) as u64
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    debug_assert!(
        src.is_multiple_of(4) && weight.is_multiple_of(4) && dst.is_multiple_of(4),
        "the host step's arena offsets are f32-lane aligned"
    );
    if accelerated_route_applies(c_in, h, w_in, c_out, kh, kw, groups) {
        // Roles swapped: the conv-transpose input plays `dy`, its output plays
        // `dx`, and `c_in`/`h`/`w_in` describe that conv-backward-input `dx`.
        crate::conv_bwd_host::run_conv2d_backward_input(
            arena,
            device,
            queue,
            src as u32,
            weight as u32,
            dst as u32,
            n as u32,
            c_out as u32,
            h_out as u32,
            w_out as u32,
            c_in as u32,
            h as u32,
            w_in as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            ph as u32,
            pw as u32,
            dh as u32,
            dw as u32,
            groups as u32,
        );
        return;
    }
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_conv_transpose2d_nchw(
        &mut a, src, weight, dst, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw,
        dh, dw, groups,
    );
}
