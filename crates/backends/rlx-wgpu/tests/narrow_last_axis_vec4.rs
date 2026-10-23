// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The last-axis `narrow`: the shape every transposed convolution ends in.
//!
//! `narrow_vec4.wgsl` used to describe only a copy of `inner` elements per row,
//! so a slice along the LAST axis (`inner == 1`) had nothing to index and the
//! lowering's admission (`inner % 4 == 0`) sent it to the scalar kernel. The
//! contiguous run there is the slice itself, and the sound half's DAC upsamplers
//! slice their output 4-aligned along that axis (start 0, length a multiple of
//! four, row stride a multiple of four), so the vector kernel serves it.
//!
//! This file pins three things: the vector kernel's last-axis indexing against
//! the CPU's own `Narrow`, which of the two modules the admission actually
//! selected (`vec4_move_count` — both compute the same slice, so parity alone
//! cannot tell them apart), and the fact that the shapes which do NOT meet the
//! alignment terms (an odd start, an odd length, an odd row stride) still compute
//! the same slice — they take the scalar module, and both modules must agree.
//!
//! A copy moves lanes unchanged, so the comparison is exact, not a tolerance.
mod common;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// `[1, channels, 1, length]` — the DAC's NCHW row.
fn shaped(channels: usize, length: usize) -> Shape {
    Shape::new(&[1, channels, 1, length], DType::F32)
}

/// One last-axis slice out of `input`, `len` elements from `start`.
fn narrow_graph(channels: usize, length: usize, start: usize, len: usize) -> Graph {
    let mut g = Graph::new("narrow_last_axis");
    let x = g.input("x", shaped(channels, length));
    let out = g.add_node(
        Op::Narrow {
            axis: 3,
            start,
            len,
        },
        vec![x],
        shaped(channels, len),
    );
    g.set_outputs(vec![out]);
    g
}

fn values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 4.0
        })
        .collect()
}

fn run(device: Device, g: Graph, x: &[f32]) -> Vec<f32> {
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("graph produced no output")
}

/// The device run's output, and how many `vec4` movement dispatches it took: the
/// scalar module produces the same bytes, so the counter is what says which one
/// ran.
fn run_counted(g: Graph, x: &[f32]) -> (Vec<f32>, u64) {
    rlx_wgpu::reset_vec4_move_count();
    let out = run(Device::Gpu, g, x);
    (out, rlx_wgpu::vec4_move_count())
}

/// Every case must equal the CPU slice exactly, on the module its shape admits.
/// The first two meet the vector admission (`inner == 1`, whole-vector
/// start/length/stride); the last two do not and take the scalar module.
#[test]
fn a_last_axis_slice_matches_cpu_on_both_kernels() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    const CHANNELS: usize = 8;
    const LENGTH: usize = 40;
    let x = values(CHANNELS * LENGTH);
    // The counter is process-global: no other test of this binary may bracket a
    // run while this one does.
    let _guard = common::route_lock();
    for (start, len, vector, why) in [
        (
            0usize,
            36usize,
            true,
            "whole-vector start and length, whole-vector stride",
        ),
        (4, 16, true, "a non-zero whole-vector start"),
        (
            2,
            13,
            false,
            "an odd start and an odd length — the scalar module",
        ),
        (7, 1, false, "a one-element slice — the scalar module"),
    ] {
        let (got, vec4) = run_counted(narrow_graph(CHANNELS, LENGTH, start, len), &x);
        assert_eq!(
            vec4,
            u64::from(vector),
            "case ({start}, {len}) ({why}): expected the {} module, got {vec4} `vec4` \
             dispatch(es) — both modules compute this slice, so only the counter \
             separates them",
            if vector { "vector" } else { "scalar" }
        );
        let want = run(Device::Cpu, narrow_graph(CHANNELS, LENGTH, start, len), &x);
        assert_eq!(got.len(), want.len(), "case ({start}, {len}): length");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "case ({start}, {len}) ({why}): element {i} is {a} on the device, {b} on \
                 the CPU — a slice is a lane move and must be bit-identical"
            );
        }
    }
}

/// A row stride that is not a whole number of vectors keeps the scalar module
/// even when the slice itself is 4-aligned: the vector kernel's row start
/// (`outer * axis_in_size`) would land mid-vector.
#[test]
fn an_unaligned_row_stride_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    const CHANNELS: usize = 6;
    const LENGTH: usize = 30;
    let x = values(CHANNELS * LENGTH);
    let _guard = common::route_lock();
    let (got, vec4) = run_counted(narrow_graph(CHANNELS, LENGTH, 4, 16), &x);
    assert_eq!(
        vec4, 0,
        "a 30-wide row stride must keep the scalar module: the vector kernel's row \
         start would be mid-vector"
    );
    let want = run(Device::Cpu, narrow_graph(CHANNELS, LENGTH, 4, 16), &x);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}");
    }
}
