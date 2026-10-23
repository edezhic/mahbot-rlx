// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What one *tiny* schedule step costs on this device, as a marginal figure.
//!
//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test small_step_dispatch_cost -- --ignored --nocapture
//! ```
//!
//! # Why
//!
//! The fork's decode graphs are chains of small steps: one code-predictor frame
//! lowers to 1 465 steps of which only 348 are multiplies (the rest are
//! `layer_norm`, `rope`, `fused_residual_ln_tee`, `buffer_copy`, `narrow`,
//! `attention`, `fused_swiglu`), and one talker body frame to 450 with 169
//! multiplies. Their tensors are one row — 1 024 or 2 048 floats — so a step's
//! own arithmetic is nothing next to what the device spends per dispatch. This
//! cell measures that per-dispatch share directly instead of inferring it from a
//! whole graph: `N` independent `RmsNorm` nodes over one `[1, 1024]` input, `N`
//! from 8 to 1 024, one call each, built and timed in one session.
//!
//! # What it reads
//!
//! For each `N`: the call's best-of-`ROUNDS` ms, and the marginal µs per step
//! between consecutive `N` — `(ms(N₂) − ms(N₁)) / (N₂ − N₁)`. The marginal
//! figure is the one to quote: the call's own submission, wait and readback sit
//! in every point and cancel out of the difference. The bytes moved per step are
//! printed beside it, so the same table says what a *linear* step costs.
//!
//! The nodes are independent (each reads the graph's one input), so the
//! dispatches are not serialised by a dependency; a dependent chain in a real
//! frame pays at least this per step and usually a memory round trip more. What
//! this does **not** establish: any single step kind's cost (one kind is used
//! throughout), the effect of switching pipelines between steps (every step here
//! shares one pipeline), or the cost of a step whose operand is in the separate
//! weight buffer.
mod common;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Independent `RmsNorm`s: `n` nodes over one row of `d` floats.
fn norm_graph(n: usize, d: usize) -> Graph {
    let mut g = Graph::new("small_step_dispatch_cost");
    let x = g.input("x", Shape::new(&[1, d], DType::F32));
    let w = g.param("w", Shape::new(&[d], DType::F32));
    let mut outs = Vec::with_capacity(n);
    // Every node reads the same row, so no node waits on another: the schedule
    // is `n` dispatches with nothing between them but the encoder.
    for _ in 0..n {
        outs.push(g.add_node(
            Op::RmsNorm { axis: 1, eps: 1e-6 },
            vec![x, w],
            Shape::new(&[1, d], DType::F32),
        ));
    }
    g.set_outputs(outs);
    g
}

/// Best-of-`ROUNDS` whole-call ms.
fn best_ms(mut one_call: impl FnMut()) -> f64 {
    for _ in 0..3 {
        one_call();
    }
    common::time_rounds(7, 1, &mut one_call)
        .into_iter()
        .fold(f64::MAX, f64::min)
}

/// One operation's own cost in µs: 1 000 calls a round, best of five.
fn per_op_us(mut one_op: impl FnMut()) -> f64 {
    const CALLS: usize = 1000;
    common::time_rounds(5, CALLS, &mut one_op)
        .into_iter()
        .fold(f64::MAX, f64::min)
        * 1e3
}

#[test]
#[ignore = "measurement: prints a cost table, asserts no duration"]
fn small_step_dispatch_cost() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    const D: usize = 1024;
    const NS: [usize; 5] = [8, 32, 128, 512, 1024];
    let x = common::ramp(1, D, 0.011);
    let w = common::ramp(1, D, 0.003);

    eprintln!("[small-step] one workload per point: N independent RmsNorm steps over [1, {D}]");
    let mut prev: Option<(usize, f64)> = None;
    for n in NS {
        let g = norm_graph(n, D);
        let mut c = Session::new(Device::Gpu).compile(g);
        c.set_param("w", &w);
        let out = c.run(&[("x", &x)]);
        assert_eq!(out.len(), n, "one output per node");
        assert!(
            out.iter()
                .all(|v| v.len() == D && v.iter().all(|u| u.is_finite()))
        );
        let ms = best_ms(|| {
            let _ = c.run(&[("x", &x)]);
        });
        let per_step = if let Some((pn, pms)) = prev {
            (ms - pms) * 1e3 / (n - pn) as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "[small-step] n={n:4} call_ms={ms:8.3} marginal_us_per_step={per_step:7.3} \
             per_call_us_per_step={:7.3} bytes_per_step={}",
            ms * 1e3 / n as f64,
            D * 4 * 2
        );
        prev = Some((n, ms));
    }
    dispatch_setup();
}

/// The per-dispatch CPU-side setup a packed multiply pays: a 32-byte uniform
/// buffer created and written, and a bind group over one storage window plus that
/// uniform. Printed per op, so the round that pools or caches them has a before —
/// the one-row arm pays both on every call, 516 times a synthesis frame.
fn dispatch_setup() {
    let Some(dev) = rlx_wgpu::wgpu_device() else {
        eprintln!("[dispatch-setup] skip: wgpu unavailable");
        return;
    };
    let device = &dev.device;
    let queue = &dev.queue;
    let window = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dispatch-setup window"),
        size: 1 << 20,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rlx-wgpu dispatch-setup bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let uniform = |mapped: bool| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu dispatch-setup uniform"),
            size: 32,
            usage: if mapped {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            },
            mapped_at_creation: mapped,
        })
    };
    let bind = |uniform: &wgpu::Buffer, off: u64| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu dispatch-setup bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &window,
                        offset: off,
                        size: wgpu::BufferSize::new(4096),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    };
    // Each op a thousand times per round, best of five: a mean within a round is
    // honest here because the figure is a per-call cost and not a rate, and the
    // rounds go through the crate's own clock (`common::time_rounds`).
    let create_us = per_op_us(|| {
        let _ = uniform(false);
    });
    let u = uniform(false);
    let write_us = per_op_us(|| {
        queue.write_buffer(&u, 0, &[0u8; 32]);
    });
    let mapped_us = per_op_us(|| {
        // Created mapped, written through the mapping, unmapped — the alternative
        // the one-row arm measured and did not keep (a mapping is not re-mappable
        // after `unmap`, and it measured slower than the queued write beside it).
        let b = uniform(true);
        {
            let mut view = b
                .slice(..)
                .get_mapped_range_mut()
                .expect("a buffer created mapped is mappable");
            view.copy_from_slice(&[1u8; 32]);
        }
        b.unmap();
    });
    let mut i = 0u64;
    let bind_us = per_op_us(|| {
        let _ = bind(&u, (i % 64) * 256);
        i += 1;
    });
    eprintln!(
        "[dispatch-setup] per op: create_buffer {create_us:.2} us, write_buffer {write_us:.2} us, \
         mapped write {mapped_us:.2} us, create_bind_group {bind_us:.2} us — a packed multiply \
         pays one creation plus one of the two writes plus the bind group, 516 times a \
         synthesis frame"
    );
}
