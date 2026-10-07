// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which route a packed GGUF `DequantMatMul` took is readable from the backend
//! (`gguf_gpu::route_counts`), and a weight window wider than the binding limit
//! must be N-chunked rather than pushed onto the per-row path.
//!
//! Its own test binary and a single test fn, so the process-global counters have
//! no concurrent writers.
//!
//! Everything here is an IN-ARENA weight: the executable only moves a param into
//! the separate packed-weight buffer when the *plan* exceeds the device binding
//! limit (4 GiB of packed weights on this hardware), which no small test reaches —
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` would force the split, but nothing in this file
//! sets it. `gguf_gemm_parked_weight.rs` drives the split arena directly instead.
//! The `RLX_WGPU_MAX_BIND_MB` cases below drive the window/chunk arithmetic: a
//! weight *window* over the binding limit must be N-chunked, and no chunk window
//! may be sized from the weight's position relative to offset 0.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{DequantMatmulRoutes, reset_route_counts, route_counts};

/// Packed `w` and its row-major `x` for one `DequantMatMul`.
fn case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_route");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x)
}

fn run(device: Device, g: Graph, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

/// Brackets one call — reset, run, snapshot — so the route is this call's own.
fn route_of(scheme: QuantScheme, ggml: rlx_gguf::GgmlType, m: usize) -> DequantMatmulRoutes {
    let (g, packed, x) = case(scheme, ggml, m, 256, 64);
    reset_route_counts();
    let _ = run(Device::Gpu, g, &packed, &x);
    route_counts()
}

/// One test fn on purpose — see the module doc.
#[test]
fn gguf_matmul_routes_are_reported_and_weight_windows_are_chunked() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    // m == 1 keeps today's single-row path.
    let one = route_of(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1);
    assert_eq!(
        (one.per_row, one.grouped),
        (1, 0),
        "m=1 must stay on the per-row GEMV: {one:?}"
    );

    // Any m > 1 — including 2 and 3, the lowest sizes that must be served — takes
    // the grouped path, up to the ~96 rows the model bodies use and the ~276 of the
    // widest matrices.
    for m in [2usize, 3, 4, 5, 6, 7, 8, 9, 96, 276] {
        let c = route_of(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, m);
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "Q4_K m={m} must be one grouped dispatch: {c:?}"
        );
    }
    for m in [2usize, 4, 96] {
        let c = route_of(QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, m);
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "Q6_K m={m} must be one grouped dispatch: {c:?}"
        );
    }
    // The one-bit scheme keeps its own multi-row GEMM.
    let q1 = route_of(QuantScheme::GgufQ1_0, rlx_gguf::GgmlType::Q1_0, 4);
    assert_eq!(
        (q1.q1_0_gemm, q1.grouped),
        (1, 0),
        "Q1_0 must keep its tiled GEMM: {q1:?}"
    );
    // A scheme with no fused kernel still takes the host / scratch route — the
    // case the compile-time report shows as native because the op is claimed.
    let q5 = route_of(QuantScheme::GgufQ5_0, rlx_gguf::GgmlType::Q5_0, 4);
    assert_eq!(q5.host + q5.scratch, 1, "Q5_0 route: {q5:?}");

    // Q4_K parity and routing at a second shape (k=256, n=640), m=1 against m=96:
    // the m=1 GEMV must stay the m=1 route, the 12-row-tile case must be one
    // grouped call, and both must match the CPU reference.
    for m in [1usize, 96] {
        let scheme = QuantScheme::GgufQ4K;
        let (g, packed, x) = case(scheme, rlx_gguf::GgmlType::Q4K, m, 256, 640);
        reset_route_counts();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let c = route_counts();
        let cpu = run(Device::Cpu, g, &packed, &x);
        let want = if m == 1 { (0, 1) } else { (1, 0) };
        assert_eq!(
            (c.grouped, c.per_row),
            want,
            "{scheme:?} m={m} route: {c:?}"
        );
        let max_abs = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{scheme:?} m={m} k=256 n=640: max_abs={max_abs:.6e}");
        assert!(
            max_abs <= 1e-2,
            "{scheme:?} m={m}: max_abs {max_abs} > 1e-2"
        );
    }

    // A weight window wider than the binding limit must be N-chunked (the
    // regime a model bigger than the window lands in), not pushed onto the
    // per-row path — and chunking must not move the result against the CPU
    // reference. 1 MiB windows force several chunks; m=96 (12 exact row tiles,
    // so only the last column chunk is partial) also makes the output slab
    // (`by_out`) rather than the weight the binding constraint, which is the
    // widest chunk count the encoder can be made to produce here. Q6_K is a
    // different packed layout (210-byte blocks, quarter-sized sub-blocks) with
    // its own chunk arithmetic.
    for (scheme, ggml, m) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 4usize),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, 4),
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 96),
    ] {
        let (g, packed, x) = case(scheme, ggml, m, 256, 16384);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let c = route_counts();
        let cpu = run(Device::Cpu, g, &packed, &x);
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "{scheme:?} m={m}: a weight window over the binding limit must still \
             be grouped: {c:?}"
        );
        let max_abs = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("chunked-window {scheme:?} m={m} k=256 n=16384: max_abs={max_abs:.6e}");
        assert!(
            max_abs <= 1e-2,
            "chunked-window {scheme:?} m={m}: max_abs {max_abs} > 1e-2"
        );
    }

    // Accuracy against *today's* path — the same rows computed one at a time, i.e.
    // the m=1 GEMV — under the same chunked-window regime and at the same 1e-2
    // the CPU-parity checks above use. A wrong chunk stride or row tile is O(1)
    // here rather than round-off, which is what this pins. The packed bytes are a
    // function of (scheme, k, n), so the single-row graphs reuse this weight; the
    // ignored `gguf_gemm_grouped_weight_reuse.rs` repeats this at the widest served
    // width, too big to build in the default run.
    {
        let (k, n, m) = (256usize, 16384usize, 4usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, m, k, n);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        let grouped = run(Device::Gpu, g, &packed, &x);
        let c = route_counts();
        let mut worst = 0.0f32;
        for r in 0..m {
            let (g1, _, _) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
            let row = run(Device::Gpu, g1, &packed, &x[r * k..(r + 1) * k]);
            worst = worst.max(
                grouped[r * n..(r + 1) * n]
                    .iter()
                    .zip(&row)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max),
            );
        }
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "the reference run must itself be the grouped path: {c:?}"
        );
        eprintln!("grouped m={m} vs per-row at k={k} n={n} chunked: max_abs={worst:.6e}");
        assert!(worst <= 1e-2, "grouped vs per-row max_abs {worst} > 1e-2");
    }

    // Two packed weights in one graph, so the second weight's region starts at a
    // NON-zero base: `from_plan_split` and the planner both place params
    // largest-first, and the single-param cases above never move that base off 0.
    // Both weights N-chunk (1 MiB bind cap), so every chunk resolves
    // `w_byte_off + n0 * row_bytes` against a non-zero base — the arithmetic a
    // multi-weight model needs and nothing else here exercises. The per-weight
    // CPU reference reuses `case`, whose packed bytes are a function of
    // (scheme, k, n) alone. (`gguf_gemm_parked_weight.rs` covers the same thing
    // for a weight parked in the weight buffer.)
    {
        let (k, na, nb, m) = (256usize, 8192usize, 8192usize, 4usize);
        let packed_of = |ggml: rlx_gguf::GgmlType, n: usize| {
            let w_row: Vec<f32> = (0..k * n)
                .map(|i| ((i as f32) * 0.011).sin() * 0.5)
                .collect();
            rlx_gguf::quantize(&w_row, ggml).expect("quantize")
        };
        // Q6_K is 210 bytes per column against Q4_K's 144, so `wb` is the larger
        // param and sorts first: `wa` lands at a non-zero offset.
        let (wa, wb) = (
            packed_of(rlx_gguf::GgmlType::Q4K, na),
            packed_of(rlx_gguf::GgmlType::Q6K, nb),
        );
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

        let mut g = Graph::new("wgpu_gguf_route_two_weights");
        let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
        let pa = g.param("wa", Shape::new(&[wa.len()], DType::U8));
        let pb = g.param("wb", Shape::new(&[wb.len()], DType::U8));
        let ya = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K,
            },
            vec![x_in, pa],
            Shape::new(&[m, na], DType::F32),
        );
        let yb = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ6K,
            },
            vec![x_in, pb],
            Shape::new(&[m, nb], DType::F32),
        );
        g.set_outputs(vec![ya, yb]);

        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        let mut c = Session::new(Device::Gpu).compile(g);
        c.set_param_typed("wa", &wa, DType::U8);
        c.set_param_typed("wb", &wb, DType::U8);
        let _ = c.run(&[("x", &x)]);
        reset_route_counts();
        let got = c.run(&[("x", &x)]);
        let counts = route_counts();
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (counts.grouped, counts.per_row),
            (2, 0),
            "both packed ops must be grouped: {counts:?}"
        );

        for (i, (scheme, ggml, n)) in [
            (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, na),
            (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, nb),
        ]
        .into_iter()
        .enumerate()
        {
            let (g1, packed, x1) = case(scheme, ggml, m, k, n);
            let want = run(Device::Cpu, g1, &packed, &x1);
            let max_abs = want
                .iter()
                .zip(&got[i])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("two weights: op{i} {scheme:?} m={m} k={k} n={n} max_abs={max_abs:.6e}");
            assert!(
                max_abs <= 1e-2,
                "two weights: op{i} {scheme:?} max_abs {max_abs} > 1e-2"
            );
        }
    }

    // The grouped path's output slab is device memory the arena plan does not own,
    // so it has to be visible to the backend's memory report — the reported figure
    // must not understate real usage — and inside its cap.
    let pooled = rlx_wgpu::gguf_gpu::pooled_out_buf_bytes();
    assert!(
        pooled > 0 && pooled <= 256 * 1024 * 1024,
        "pooled out slab {pooled} bytes must be reported and inside the 256 MiB cap"
    );
}
