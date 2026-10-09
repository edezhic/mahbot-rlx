// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which route a packed GGUF `DequantMatMul` took is readable from the backend
//! (`gguf_gpu::route_counts`), and a weight window wider than the binding limit
//! must be N-chunked rather than pushed onto the per-row path. The one-row route
//! is a single route with a shape of its own, which is readable the same way
//! (`gguf_gpu::gemv_one_row_shape`).
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

mod common;
use common::max_abs_diff;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{
    DequantMatmulRoutes, gemv_one_row_shape, reset_gemv_one_row_shape, reset_route_counts,
    route_counts,
};

/// Packed `w` and its row-major `x` for one `DequantMatMul`.
fn case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    case_batched(scheme, ggml, &[], m, k, n)
}

/// As [`case`], with leading batch dims: the 3-D single-token decode shape the
/// models use has to reach the same route counters as the 2-D one.
fn case_batched(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    batch: &[usize],
    m: usize,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut in_shape = batch.to_vec();
    in_shape.extend_from_slice(&[m, k]);
    let mut out_shape = batch.to_vec();
    out_shape.extend_from_slice(&[m, n]);

    let mut g = Graph::new("wgpu_gguf_route");
    let x_in = g.input("x", Shape::new(&in_shape, DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&out_shape, DType::F32),
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

    // Q4_K parity at a second shape (k=256, n=640), m=1 against m=96, against the
    // CPU reference. The routes for these two cells are asserted by the shape grid
    // below, alongside every other shape; this loop is what pins their parity.
    for m in [1usize, 96] {
        let scheme = QuantScheme::GgufQ4K;
        let (g, packed, x) = case(scheme, rlx_gguf::GgmlType::Q4K, m, 256, 640);
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let cpu = run(Device::Cpu, g, &packed, &x);
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!("{scheme:?} m={m} k=256 n=640: max_abs={max_abs:.6e}");
        assert!(
            max_abs <= 1e-2,
            "{scheme:?} m={m}: max_abs {max_abs} > 1e-2"
        );
    }

    // The in-scope schemes at every shape the parity suite's grid uses, printed
    // from the process-global counters this binary owns. That file reports a route
    // DERIVED from the selector; these lines are the backend's own census for the
    // same shapes, so neither the small-grid shapes nor the 3-D single-token decode
    // rests on the derivation alone.
    //
    // Q5_K and Q8_0 have NO fused route — `gemv_supports_scheme` is exactly
    // {Q4_K(0), Q6_K(2), Q1_0(24)} — so both take dequant-scratch + matmul_bt at
    // every m: one route, no change point for them to measure, and `host == 0` is
    // what an in-arena weight produces (the host route is reserved for a sharded
    // arena, a parked weight, or a refused scratch, and — the case the k=384 cell
    // below is here to reach — a fused family whose k is not a whole number of
    // blocks, which books no slab for the selector to fall back into). Q4_K / Q6_K
    // change at m == 1: one per-row GEMV below, one row-grouped dispatch above.
    //
    // The model-class graph's own widths (H=1024, I=4096) are counter-checked here
    // in both regimes rather than only derived: gate/up is k=1024→n=4096, down is
    // k=4096→n=1024, and the o-projection is k=1024→n=1024. Each appears in the
    // decode regime (the 2-D m=1 form and the 3-D single-token `[1,1,k]` form) and
    // in the prefill regime where the model graph uses it (gate/up and o at m>1,
    // down at large m).
    let shapes: &[(&'static [usize], usize, usize, usize)] = &[
        (&[], 2, 256, 8),
        // k=384 is 1.5 Q4_K/Q6_K superblocks and 12 whole Q8_0 blocks: the cell
        // that reaches the selector's host fallback, so that arm is exercised
        // rather than left as scaffolding. Q5_K is non-fused at any k, so it still
        // takes its single scratch route here.
        (&[], 4, 384, 8),
        (&[], 96, 256, 640),
        (&[], 128, 256, 640),
        (&[], 64, 2048, 512),
        (&[], 192, 1024, 2048),
        (&[], 1, 256, 640),
        (&[1], 1, 2048, 640),
        (&[], 192, 4096, 1024),
        (&[], 1, 1024, 4096),
        (&[], 1, 4096, 1024),
        (&[], 96, 1024, 4096),
        (&[], 1, 1024, 1024),
        (&[], 96, 1024, 1024),
        (&[1], 1, 1024, 4096),
        (&[1], 1, 4096, 1024),
        (&[1], 1, 1024, 1024),
    ];
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ5K, rlx_gguf::GgmlType::Q5K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0),
    ] {
        for &(batch, m, k, n) in shapes {
            let (g, packed, x) = case_batched(scheme, ggml, batch, m, k, n);
            reset_route_counts();
            let gpu = run(Device::Gpu, g.clone(), &packed, &x);
            let c = route_counts();
            // `Debug` is the backend's own census — all five counters, rather than
            // one name picked for them.
            eprintln!(
                "[rlx-gguf-route] scheme={scheme:?} m={m} k={k} n={n} batch={batch:?} route={c:?}"
            );
            let fused = matches!(scheme, QuantScheme::GgufQ4K | QuantScheme::GgufQ6K);
            let aligned = k.is_multiple_of(scheme.gguf_block_size() as usize);
            let want = match (fused, m == 1, aligned) {
                (false, ..) => (0, 0, 1, 0), // Q5_K / Q8_0: one scratch route at every m
                // Decode takes one row per call so long as k is a multiple of
                // four (the tiled arm's `vec4` X read); the block gate is the
                // m > 1 GEMM's.
                (true, true, _) => (0, 1, 0, 0),
                (true, false, true) => (1, 0, 0, 0),
                (true, false, false) => (0, 0, 0, 1), // host fallback: no slab booked, no row slicing
            };
            assert_eq!(
                (c.grouped, c.per_row, c.scratch, c.host),
                want,
                "{scheme:?} m={m} k={k} n={n} batch={batch:?}: {c:?}"
            );
            if want == (0, 0, 0, 1) {
                // The host fallback's numbers are checked here as well as its
                // counter: the row slices are invalid and a broken staging would
                // show as a wrong or non-finite value, not as a wrong counter.
                let cpu = run(Device::Cpu, g, &packed, &x);
                let max_abs = max_abs_diff(&gpu, &cpu);
                eprintln!("host-fallback {scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e}");
                assert!(
                    max_abs <= 1e-2,
                    "{scheme:?} m={m} k={k} n={n}: host fallback max_abs {max_abs} > 1e-2"
                );
            }
        }
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
        let max_abs = max_abs_diff(&gpu, &cpu);
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
            worst = worst.max(max_abs_diff(&grouped[r * n..(r + 1) * n], &row));
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
            let max_abs = max_abs_diff(&got[i], &want);
            eprintln!("two weights: op{i} {scheme:?} m={m} k={k} n={n} max_abs={max_abs:.6e}");
            assert!(
                max_abs <= 1e-2,
                "two weights: op{i} {scheme:?} max_abs {max_abs} > 1e-2"
            );
        }
    }

    // The one-bit scheme's one-row arm: it keeps the scalar X view while the
    // route's bind group also carries the tiled arm's fifth (`vec4` X) binding,
    // so this is its route and parity at m == 1 — not a reshaped shape, which it
    // deliberately does not have.
    {
        let (k, n) = (256usize, 640usize);
        let (g, packed, x) = case(QuantScheme::GgufQ1_0, rlx_gguf::GgmlType::Q1_0, 1, k, n);
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let c = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!("GgufQ1_0 m=1 k={k} n={n}: routes={c:?} shape={shape:?} max_abs={max_abs:.6e}");
        assert_eq!(
            (c.per_row, c.q1_0_gemm),
            (1, 0),
            "Q1_0 m=1 must take the one-row arm, not the tiled GEMM: {c:?}"
        );
        assert_eq!(
            (
                shape.cols_per_workgroup,
                shape.threads_per_workgroup,
                shape.workgroups
            ),
            (64, 64, n.div_ceil(64) as u32),
            "this arm keeps one thread per output column, 64 columns to a \
             workgroup: {shape:?}"
        );
        assert!(max_abs <= 1e-2, "GgufQ1_0 m=1: max_abs {max_abs} > 1e-2");
    }

    // One row: the packed multiply's shape at the widths the model bodies use
    // (H=1024 and I=6144 columns) and at the contraction lengths those
    // projections have, in the 2-D `[1, k]` form and the 3-D `[1, 1, k]`
    // single-token form those calls take. What is recorded is the arrangement
    // actually used — one workgroup per output column with the workgroup's lanes
    // splitting that column's contraction over its sub-blocks, and
    // `workgroups x cols_per_workgroup` exactly the call's output width. That
    // last equality is the "no tile row is computed and discarded" half; a change
    // back to one unit of work per output column reading the weight byte-wise
    // would move `cols_per_workgroup` to 64 and `workgroups` to `ceil(n/64)`.
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
    ] {
        for &(k, n, batches) in &[
            (1024usize, 1024usize, &[&[][..], &[1][..]][..]),
            (1024, 6144, &[&[][..], &[1][..]][..]),
            // The down-projection's longer contraction: the split is over k, so
            // this is the same shape with more work per column.
            (4096, 1024, &[&[][..]][..]),
        ] {
            for &batch in batches {
                let (g, packed, x) = case_batched(scheme, ggml, batch, 1, k, n);
                reset_route_counts();
                reset_gemv_one_row_shape();
                let gpu = run(Device::Gpu, g.clone(), &packed, &x);
                let counts = route_counts();
                let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
                eprintln!(
                    "[rlx-gguf-gemv-shape] scheme={scheme:?} m=1 k={k} n={n} batch={batch:?} \
                     cols_per_workgroup={} threads_per_workgroup={} workgroups={}",
                    shape.cols_per_workgroup, shape.threads_per_workgroup, shape.workgroups
                );
                assert_eq!(
                    (counts.per_row, counts.grouped),
                    (1, 0),
                    "{scheme:?} m=1 k={k} n={n} batch={batch:?}: {counts:?}"
                );
                assert_eq!(
                    (
                        shape.cols_per_workgroup,
                        shape.threads_per_workgroup,
                        shape.workgroups
                    ),
                    (1, 32, n as u32),
                    "{scheme:?} k={k} n={n} batch={batch:?}: one row's work is its output \
                     columns times its contraction — one workgroup per column, the \
                     workgroup's threads splitting that column, no tile rows: {shape:?}"
                );
                let cpu = run(Device::Cpu, g, &packed, &x);
                let max_abs = max_abs_diff(&gpu, &cpu);
                eprintln!(
                    "one-row shape {scheme:?} k={k} n={n} batch={batch:?}: max_abs={max_abs:.6e}"
                );
                assert!(
                    max_abs <= 1e-2,
                    "{scheme:?} k={k} n={n} batch={batch:?}: max_abs {max_abs} > 1e-2"
                );
            }
        }
    }

    // Throughput at those shapes and beyond over repeated calls — a measurement
    // for the record, not a threshold. One call's own submission and device wait
    // swamp the streaming it does, which is why this is not a single call, and
    // the best of a few rounds is reported rather than one round: at the narrower
    // width the per-call floor is a large part of the figure, so the spread
    // between rounds is real information about the shape, not noise to smooth
    // away silently.
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
    ] {
        for (k, n, iters) in [
            (1024usize, 1024usize, 200usize),
            (1024, 6144, 200),
            // Wider than the model widths (1024…6144 columns), where the arm
            // must not get worse than the shape it replaced, with a shorter run
            // so the cell stays cheap. n = 65536 also crosses the tiled arm's
            // one-workgroup-per-column dispatch cap, so that figure carries the
            // extra chunk dispatch the cell above pins.
            (1024, 16384, 40),
            (1024, 32768, 40),
            (1024, 65536, 40),
        ] {
            let (g, packed, x) = case(scheme, ggml, 1, k, n);
            let mut c = Session::new(Device::Gpu).compile(g);
            c.set_param_typed("w", &packed, DType::U8);
            let warm = c.run(&[("x", &x)]);
            assert_eq!(warm[0].len(), n, "one row of {n} columns");
            let mut best_us = f64::MAX;
            for _ in 0..3 {
                // `Tick`, not `Instant`: this measures sub-millisecond per-call
                // work, and the tree mandates the tick at that scale.
                let start = rlx_runtime::Tick::now();
                for _ in 0..iters {
                    let _ = c.run(&[("x", &x)]);
                }
                best_us = best_us.min(rlx_runtime::Tick::now().elapsed_us(start));
            }
            let streamed_mib = (packed.len() * iters) as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[rlx-gguf-gemv-throughput] scheme={scheme:?} m=1 k={k} n={n} iters={iters} \
                 streamed_MiB={streamed_mib:.1} best_ms={:.2} per_call_us={:.0} \
                 streamed_MiB_per_s={:.0}",
                best_us * 1e-3,
                best_us / iters as f64,
                streamed_mib / (best_us * 1e-6)
            );
        }
    }

    // The same arm when the weight window exceeds the binding limit: rows are
    // N-chunked, so one call is several dispatches and the recorded one is the
    // last chunk's — never wider than the width asked for, and never zero. The
    // chunk stride is the share of the row indexing a wrong chunking would move,
    // which is what the parity check against CPU pins.
    {
        let (k, n) = (256usize, 16384usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let counts = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (counts.per_row, counts.grouped),
            (1, 0),
            "a 1 MiB window must still take the one-row arm: {counts:?}"
        );
        assert_eq!(
            (shape.cols_per_workgroup, shape.threads_per_workgroup),
            (1, 32),
            "{shape:?}"
        );
        assert!(
            shape.workgroups as usize > 0 && (shape.workgroups as usize) < n,
            "the window must have been split into more than one chunk: {shape:?}"
        );
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "chunked one-row GgufQ4K m=1 k={k} n={n}: last-chunk workgroups={} max_abs={max_abs:.6e}",
            shape.workgroups
        );
        assert!(
            max_abs <= 1e-2,
            "chunked one-row GgufQ4K k={k} n={n}: max_abs {max_abs} > 1e-2"
        );
    }

    // The same arm when the output is wider than one dispatch dimension's cap:
    // the tiled arm spends one workgroup per output column, and 65535 is the
    // WebGPU floor for a dimension (`max_compute_workgroups_per_dimension`), so
    // a wider output has to be another chunk of the same arm. Reaching it in one
    // dispatch is a wgpu validation error rather than a slow path — this cell is
    // what keeps that from coming back. The weight window is far under the
    // binding limit, so the split is the cap's and nothing else's, and the chunk
    // boundary is a fresh stride through the packed rows, which the CPU parity
    // pins.
    {
        let (k, n) = (256usize, 65536usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let counts = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        assert_eq!(
            (counts.per_row, counts.grouped),
            (1, 0),
            "an output over the dispatch cap must still take the one-row arm: {counts:?}"
        );
        assert_eq!(
            (
                shape.cols_per_workgroup,
                shape.threads_per_workgroup,
                shape.workgroups as usize
            ),
            (1, 32, n - 65535),
            "the dispatch cap (the WebGPU floor for one dimension, \
             max_compute_workgroups_per_dimension) must have split {n} columns \
             into the cap plus a remainder: {shape:?}"
        );
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "capped one-row GgufQ4K m=1 k={k} n={n}: last-chunk workgroups={} max_abs={max_abs:.6e}",
            shape.workgroups
        );
        assert!(
            max_abs <= 1e-2,
            "capped one-row GgufQ4K k={k} n={n}: max_abs {max_abs} > 1e-2"
        );
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
