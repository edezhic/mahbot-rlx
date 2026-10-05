// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight-reuse acceptance for the row-grouped GGUF GEMM at the widest served
//! shape: a ~800 MiB packed Q4_K weight, wider than one bind window, so the
//! encoder has to re-resolve and re-bind the weight several times per call. A
//! batched call must then cost roughly one pass over the packed weight per group
//! of 8 rows instead of one pass per row, and must agree with the per-row path it
//! replaces on the same rows.
//!
//! The weight is in the act arena here: the executable only moves a param into the
//! separate packed-weight buffer when the plan exceeds the device binding limit —
//! 4 GiB of packed weights on this hardware — which a single 777 MiB weight does
//! not reach. `gguf_gemm_parked_weight.rs` drives the split arena (harness size and
//! kilobyte size) directly instead.
//!
//! `#[ignore]`d — it materialises ~800 MiB of packed weights next to the device
//! copy and takes seconds. Run it deliberately:
//!
//! ```text
//! cargo test -p rlx-wgpu --release --test gguf_gemm_grouped_weight_reuse \
//!     -- --ignored --nocapture
//! ```
//!
//! Its own binary, and one test fn, so the process-global route counters have no
//! concurrent writers.

use std::time::Instant;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{DequantMatmulRoutes, reset_route_counts, route_counts};

/// One `DequantMatMul` over `packed` (`[n, k]`, `k/256 × 144` bytes per column)
/// and its row-major `x`.
fn graph(k: usize, n: usize, m: usize, packed_len: usize) -> (Graph, Vec<f32>) {
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 1e-4).sin()).collect();
    let mut g = Graph::new("wgpu_gguf_weight_reuse");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed_len], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, x)
}

/// The packed weight: a deterministic function of `(k, n)` alone, so a
/// single-row graph can reuse a wider case's weight. Nothing here is a real
/// quantization, but every byte is masked to six bits and the high halves of the
/// two f16 scale fields of each 144-byte block get a fixed small exponent, so
/// `d`/`dmin` land in [2^-13, 2^-12) and the dequantized weights stay ~1e-2 in
/// magnitude: finite, and small enough that the grouped and per-row results can be
/// compared under the op's absolute 1e-2 tolerance rather than measuring the
/// test's own weight scale (large random weights cancel catastrophically, and the
/// round-off of that cancellation would be what such a bound caught).
fn packed_bytes(k: usize, n: usize) -> Vec<u8> {
    let mut s = 7u32;
    (0..(k / 256) * 144 * n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 24) as u8;
            match i % 144 {
                // High half of an f16: sign 0, exponent 2, low mantissa bits from
                // the generator.
                1 | 3 => 0x08 | (b & 0x03),
                _ => b & 0x3F,
            }
        })
        .collect()
}

/// One timed call of the `m`-row graph against the shared weight, and the route
/// it took.
fn one_call(packed: &[u8], k: usize, n: usize, m: usize) -> (f64, DequantMatmulRoutes) {
    let (g, x) = graph(k, n, m, packed.len());
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    let _ = c.run(&[("x", &x)]); // warm: upload + first-run setup
    reset_route_counts();
    let t0 = Instant::now();
    let _ = c.run(&[("x", &x)]);
    (t0.elapsed().as_secs_f64() * 1e3, route_counts())
}

#[test]
#[ignore = "~800 MiB of packed weights; run with --ignored --nocapture"]
fn a_row_group_costs_one_pass_over_the_packed_weight() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // The widest served shape: 202048 × 7168 Q4_K = 777 MiB packed. The bind cap
    // splits it into four column chunks, standing in for a device whose binding
    // limit is under the weight size.
    let (k, n) = (7168usize, 202048usize);
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "256");
    let packed = packed_bytes(k, n);

    // 2 is the smallest batch that has to be served and 276 the widest matrix.
    let measured: Vec<(usize, (f64, DequantMatmulRoutes))> = [1usize, 2, 3, 4, 8, 96, 276]
        .into_iter()
        .map(|m| (m, one_call(&packed, k, n, m)))
        .collect();
    let t1 = measured[0].1.0;
    eprintln!(
        "weight reuse (Q4_K k={k} n={n}, 777 MiB + 4 bind windows): {}",
        measured
            .iter()
            .map(|(m, (t, _))| format!("m={m} {t:.1} ms ({:.2}x)", t / t1))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (m, (t, r)) in &measured {
        let (m, t) = (*m, *t);
        let want = if m == 1 { (0, 1) } else { (1, 0) };
        assert_eq!(
            (r.grouped, r.per_row),
            want,
            "m={m} must take {}: {r:?}",
            if m == 1 {
                "the per-row GEMV"
            } else {
                "one grouped dispatch"
            }
        );
        // Acceptance: the weight is read once per group of 8 rows, so a batch
        // costs `ceil(m / 8)` passes over it — not the `m` passes of the per-row
        // path it replaces. The bound below is what separates the two: a
        // one-pass-per-row fallback would blow past it (m=4 → 4 passes against a
        // 1.5-pass bound) while the measured cost sits at ~1.1 passes per group.
        assert!(
            t <= (m.div_ceil(8) as f64 * 1.5) * t1,
            "m={m}: {t:.1} ms is more than {:.1} ms — {} row groups over a \
             single-row pass",
            (m.div_ceil(8) as f64 * 1.5) * t1,
            m.div_ceil(8)
        );
    }

    // Accuracy against today's path — the same rows computed one at a time (the m=1
    // GEMV) — at the full 202048-column width, where the four column chunks and
    // their copy-back are all in play. Same absolute 1e-2 the packed-multiply
    // CPU-parity checks use; the reference's own magnitude is printed so the margin
    // is visible. `gguf_matmul_route.rs` pins the same comparison on a small weight
    // in the default run.
    let (g, x) = graph(k, n, 4, packed.len());
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let _ = c.run(&[("x", &x)]);
    let grouped = c.run(&[("x", &x)]).remove(0);
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for r in 0..4 {
        let (g1, _) = graph(k, n, 1, packed.len());
        let mut c1 = Session::new(Device::Gpu).compile(g1);
        c1.set_param_typed("w", &packed, DType::U8);
        let row_x = &x[r * k..(r + 1) * k];
        let _ = c1.run(&[("x", row_x)]);
        let one = c1.run(&[("x", row_x)]).remove(0);
        scale = one.iter().fold(scale, |m, v| m.max(v.abs()));
        for (a, b) in grouped[r * n..(r + 1) * n].iter().zip(&one) {
            worst = worst.max((a - b).abs());
        }
    }
    eprintln!(
        "grouped m=4 vs per-row m=1 at k={k} n={n}: max_abs={worst:e} (per-row |y|max={scale:e})"
    );
    assert!(worst <= 1e-2, "grouped vs per-row max_abs {worst:e} > 1e-2");

    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");

    // The group path's output slab is real device memory the arena plan does not
    // own, so it has to be visible to the backend's memory report — and capped, so a
    // wide head at a few thousand rows cannot allocate to the binding limit. 256 MiB
    // is that cap here (the widest served shape needs 213 MiB of it).
    let pooled = rlx_wgpu::gguf_gpu::pooled_out_buf_bytes();
    assert!(
        pooled > 0 && pooled <= 256 * 1024 * 1024,
        "pooled out slab {pooled} bytes must be reported and inside the 256 MiB cap"
    );
}
