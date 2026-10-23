// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! The many-row route's cost, on the step's own query-row count, over four key
//! lengths. A `seq_q` sweep at a fixed head geometry, plus a decode-shaped
//! regression row — the arrangement the other two forms must keep their
//! numbers on — printing per-layer µs, µs per query row, GF/s (2·q·k·d FMAs
//! per row, the FLOP count `attention.wgsl`'s own header quotes) and the
//! arrangement each shape took, read from the process-global route counters.
//! At this file's head width (`HD`, under the lane-parallel form's cap) a single
//! query row takes that form at every key length and the fused kernel is never
//! reachable, so the one-row cells here are the decode form's short-key cost
//! rather than a many-row comparison.
//!
//! The cells that straddle the floor are the point: one query row against two or
//! more over the same keys — the row axis is the only one that still chooses
//! between the forms — read side by side on the real adapter. Nothing here
//! asserts a duration: `run()` includes submission, the device wait and the
//! readback, so a reading is mostly that fixed floor, and a regression in it is
//! not one in the kernel. What is asserted is only the route each shape took.
//!
//! Run the thermal gate first (`just throttle`).
mod common;

use common::data;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, Tick};
use rlx_wgpu::kernels::{attention_route_counts, reset_attention_route_counts};

/// The chain is measured whole and divided by its layer count, so the timing
/// reaches the per-layer cost rather than the submission floor.
const LAYERS: usize = 8;
const NH: usize = 8;
const HD: usize = 64;
const REPS: usize = 5;
const WARMUP: usize = 2;

/// `LAYERS` attention ops chained through the query, all reading one K/V.
fn chain(rows: usize, lk: usize) -> Graph {
    let dim = NH * HD;
    let f = DType::F32;
    let mut g = Graph::new("attn_decompose_cost");
    let mut q = g.input("q", Shape::new(&[1, rows, dim], f));
    let k = g.input("k", Shape::new(&[1, lk, dim], f));
    let v = g.input("v", Shape::new(&[1, lk, dim], f));
    for _ in 0..LAYERS {
        q = g.add_node(
            Op::Attention {
                num_heads: NH,
                head_dim: HD,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, rows, dim], f),
        );
    }
    g.set_outputs(vec![q]);
    g
}

#[test]
#[ignore = "measurement: prints a cost table, asserts no duration"]
fn attention_decompose_cost() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let dim = NH * HD;
    eprintln!(
        "[rlx-attn-decompose-cost] heads={NH} head_dim={HD} layers={LAYERS} mask=causal \
         (best of {REPS} per shape; per-layer figures)"
    );
    for lk in [32usize, 64, 256, 1024] {
        // 1 is the decode regression row (the lane-parallel form, at every key
        // length), 2 the fork's cached decode cell, 24 and 31 the top of the band
        // the many-row floor excludes, 32 and up what the dense route serves.
        for rows in [1usize, 2, 24, 31, 32, 64, 100, 256] {
            let g = chain(rows, lk);
            let q = data(rows * dim, 0.011, false);
            let k = data(lk * dim, 0.013, true);
            let v = data(lk * dim, 0.017, false);
            let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
            let mut c = Session::new(Device::Gpu).compile(g);
            for _ in 0..WARMUP {
                let _ = c.run(&inputs).remove(0);
            }
            reset_attention_route_counts();
            let mut best_us = f64::MAX;
            for _ in 0..REPS {
                let start = Tick::now();
                let _ = c.run(&inputs).remove(0);
                best_us = best_us.min(Tick::now().elapsed_us(start));
            }
            let routes = attention_route_counts();
            let per_layer = best_us / LAYERS as f64;
            let per_row = per_layer / rows as f64;
            // 2·q·k·d FMAs per query row, the fused kernel's own FLOP count.
            let gflops = 2.0 * (rows * lk * HD * NH) as f64 / (per_layer * 1e3);
            eprintln!(
                "[rlx-attn-decompose-cost] lk={lk:<5} rows={rows:<4} {per_layer:8.2} us/layer \
                 {per_row:8.3} us/row/layer {gflops:7.1} GF/s {routes:?}"
            );
            assert_eq!(
                routes,
                common::expected_attention_routes(rows, lk, HD, 0, LAYERS, REPS),
                "rows={rows} lk={lk}: the shape took a different arrangement than its \
                 query rows and key cells say"
            );
        }
    }
}
