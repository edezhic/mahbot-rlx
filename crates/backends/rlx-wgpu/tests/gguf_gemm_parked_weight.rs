// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The row-grouped GGUF GEMM against a weight parked in the backend's separate
//! packed-weight buffer — the regime a model larger than the device's binding
//! limit lands in — and, through the same hand-built arena, the one-row route's
//! X-window alignment guard.
//!
//! The executable only takes that regime when the *plan* exceeds the binding
//! limit: 4 GiB of packed weights on this hardware, which no test can afford to
//! build. So the split arena is built directly — `Arena::from_plan_split` parks
//! every packed quant param — and the same runner the executable calls is driven
//! against it.
//!
//! Two packed params, placed largest-first, put the second one's region at a
//! NON-zero weight-buffer base, and the 1 MiB window cap makes it several column
//! chunks: every chunk resolves `w_byte_off + n0 * row_bytes` against that base.
//! The reference is the same graph and the same bytes through the non-split
//! arena, i.e. the path the CPU-parity tests already cover.
//!
//! The second test repeats the exercise at the size the harness uses (777 MiB,
//! four bind windows); it is `#[ignore]`d for the memory it needs. The third is
//! the alignment guard: the runner is handed an X window four bytes off its slot,
//! which is the only way to reach it — the planner aligns every activation.

use rlx_compile::memory::plan_memory_f32_uniform;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_wgpu::buffer::{Arena, is_weight_off, raw_weight_off};
use rlx_wgpu::gguf_gpu::run_dequant_matmul_gguf_gemv_rows;
use rlx_wgpu::gguf_host::gguf_scheme_id;
use rlx_wgpu::wgpu_device;

/// Deterministic Q4_K-shaped bytes: `k/256 × 144` per column, every nibble and
/// scale field masked the way the real packer writes them.
fn packed(k: usize, n: usize) -> Vec<u8> {
    let mut s = 11u32;
    (0..(k / 256) * 144 * n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 24) as u8;
            match i % 144 {
                1 | 3 => 0x08 | (b & 0x03),
                _ => b & 0x3F,
            }
        })
        .collect()
}

/// x values for an `m × k` activation block.
fn activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k).map(|i| ((i as f32) * 1e-3).sin()).collect()
}

/// One `DequantMatMul` over a packed Q4_K weight of `w_bytes`, and the node ids
/// (x, w, y) a caller needs to place and read them.
fn one_weight_graph(
    k: usize,
    n: usize,
    m: usize,
    w_bytes: usize,
) -> (Graph, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("wgpu_gguf_parked_weight");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[w_bytes], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, x, w, y)
}

/// `from_plan_split` with this graph, asserting the packed weight landed in the
/// weight buffer rather than the act arena.
fn parked_arena(dev: &rlx_wgpu::device::WgpuDevice, g: &Graph, w: NodeId) -> Arena {
    let plan = plan_memory_f32_uniform(g, 16);
    let arena = Arena::from_plan_split(&dev.device, &plan, g, 0);
    assert!(
        arena.weight_buffer.is_some() && is_weight_off(arena.offset(w)),
        "a packed quant param must be parked in the weight buffer"
    );
    arena
}

#[test]
fn grouped_gemm_reads_a_parked_weight_from_the_weight_buffer() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let (k, m) = (256usize, 4usize);
    // Columns per weight: the larger one is placed first, so the smaller one's
    // region starts at a non-zero base. Both are wider than one 1 MiB window, so
    // each is dispatched in several column chunks.
    let (na, nb) = (12288usize, 8192usize);
    let (wa, wb) = (packed(k, na), packed(k, nb));
    let x = activations(m, k);

    let mut g = Graph::new("wgpu_gguf_parked_weight");
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
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x_in, pb],
        Shape::new(&[m, nb], DType::F32),
    );
    g.set_outputs(vec![ya, yb]);

    let plan = plan_memory_f32_uniform(&g, 16);
    let plain = Arena::from_plan(&dev.device, &plan);
    let parked = parked_arena(dev, &g, pa);

    // The premise of the test: both packed params really are in the weight
    // buffer, at distinct offsets, and one of them is at a non-zero base.
    let (oa, ob) = (parked.offset(pa), parked.offset(pb));
    assert_eq!(
        raw_weight_off(oa).min(raw_weight_off(ob)),
        0,
        "largest-first placement puts one weight at the buffer base"
    );
    assert!(
        raw_weight_off(oa).max(raw_weight_off(ob)) > 0,
        "the other weight must sit at a non-zero base (got {oa:#x} / {ob:#x})"
    );
    for (arena, label) in [(&plain, "plain"), (&parked, "parked")] {
        arena.write_bytes_range(&dev.queue, arena.offset(pa), &wa);
        arena.write_bytes_range(&dev.queue, arena.offset(pb), &wb);
        arena.write_f32(&dev.queue, x_in, &x);
        eprintln!(
            "{label}: wa @{:#x}, wb @{:#x}",
            arena.offset(pa),
            arena.offset(pb)
        );
    }

    // Multi-chunk windows for both weights, so the chunk loop (not just the first
    // chunk) has to resolve against the parked base.
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
    for (n, y) in [(na, ya), (nb, yb)] {
        run_dequant_matmul_gguf_gemv_rows(
            &parked,
            &dev.device,
            &dev.queue,
            m,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            parked.offset(x_in),
            parked.offset(if n == na { pa } else { pb }),
            parked.offset(y),
        );
        run_dequant_matmul_gguf_gemv_rows(
            &plain,
            &dev.device,
            &dev.queue,
            m,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            plain.offset(x_in),
            plain.offset(if n == na { pa } else { pb }),
            plain.offset(y),
        );
        let (got, want) = (
            parked.read_f32(&dev.device, &dev.queue, y),
            plain.read_f32(&dev.device, &dev.queue, y),
        );
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("parked vs in-arena weight, n={n}: max_abs={max_abs:e}");
        assert!(want.iter().any(|v| v.abs() > 1e-3), "reference is all-zero");
        assert!(
            max_abs <= 1e-6,
            "parked weight n={n}: max_abs {max_abs} against the same bytes in the act arena"
        );
    }
    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
}

/// The one-row route's X-window guard, driven through the same runner as the
/// executable uses: the tiled arm reads X as `vec4`, so a window that does not
/// line up with a four-element group must go to the host instead. The reference
/// is the same bytes read from an aligned window — the device arm — so an arm
/// that read the misaligned window would take it shifted by up to three elements
/// and not match.
#[test]
fn a_misaligned_x_window_is_not_read_as_vec4() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let (k, n) = (256usize, 1024usize);
    let w = packed(k, n);
    // `+1 / -1` rather than a smooth ramp: an arm that reads the window one
    // element off must show it here, and consecutive samples of a smooth input
    // are too close for that.
    let x: Vec<f32> = (0..k)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (g, x_in, w_id, y) = one_weight_graph(k, n, 1, w.len());
    let arena = Arena::from_plan(&dev.device, &plan_memory_f32_uniform(&g, 16));
    arena.write_bytes_range(&dev.queue, arena.offset(w_id), &w);
    let (x_off, w_off, y_off) = (arena.offset(x_in), arena.offset(w_id), arena.offset(y));
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    // X four bytes past its slot: a 4-byte-aligned window of the same tensor.
    let mis = x_off + 4;
    assert!(
        mis + x_bytes.len() <= arena.buffer.size() as usize,
        "X's slot needs four spare bytes to shift its window"
    );
    arena.write_bytes_range(&dev.queue, mis, &x_bytes);
    run_dequant_matmul_gguf_gemv_rows(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        mis,
        w_off,
        y_off,
    );
    let via_host = arena.read_f32(&dev.device, &dev.queue, y);

    arena.write_bytes_range(&dev.queue, x_off, &x_bytes);
    run_dequant_matmul_gguf_gemv_rows(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        x_off,
        w_off,
        y_off,
    );
    let on_device = arena.read_f32(&dev.device, &dev.queue, y);

    assert!(
        on_device.iter().any(|v| v.abs() > 1e-3),
        "the aligned reference is all-zero"
    );
    let max_abs = via_host
        .iter()
        .zip(&on_device)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = on_device.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    eprintln!("misaligned X window vs aligned: max_abs={max_abs:e} ref_scale={scale:e}");
    assert!(
        max_abs <= 1e-3,
        "a misaligned X window must not be read as vec4: max_abs {max_abs}"
    );
}

/// The same exercise at the size the harness uses — 202048 × 7168 Q4_K = 777 MiB
/// parked, four bind windows, where the weight is far larger than one window —
/// checked against the per-row GEMV it replaces, which in a parked arena binds
/// the weight buffer directly too.
///
/// `#[ignore]`d: it materialises ~800 MiB of packed weights next to the device
/// copy and builds two arenas over them.
#[test]
#[ignore = "~800 MiB of packed weights; run with --ignored --nocapture"]
fn harness_size_parked_weight_matches_the_per_row_path() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let (k, n, m) = (7168usize, 202048usize, 4usize);
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "256");
    let packed = packed(k, n);
    let x = activations(m, k);

    let (g, x_in, w, y) = one_weight_graph(k, n, m, packed.len());
    let parked = parked_arena(dev, &g, w);
    parked.write_bytes_range(&dev.queue, parked.offset(w), &packed);
    parked.write_f32(&dev.queue, x_in, &x);
    run_dequant_matmul_gguf_gemv_rows(
        &parked,
        &dev.device,
        &dev.queue,
        m,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        parked.offset(x_in),
        parked.offset(w),
        parked.offset(y),
    );
    let grouped = parked.read_f32(&dev.device, &dev.queue, y);

    // Per-row reference: a graph of its own (m == 1 picks the GEMV), one arena
    // reused for every row by rewriting just the activation.
    let (g1, x1, w1, y1) = one_weight_graph(k, n, 1, packed.len());
    let rows = parked_arena(dev, &g1, w1);
    rows.write_bytes_range(&dev.queue, rows.offset(w1), &packed);
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for r in 0..m {
        rows.write_f32(&dev.queue, x1, &x[r * k..(r + 1) * k]);
        run_dequant_matmul_gguf_gemv_rows(
            &rows,
            &dev.device,
            &dev.queue,
            1,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            rows.offset(x1),
            rows.offset(w1),
            rows.offset(y1),
        );
        let one = rows.read_f32(&dev.device, &dev.queue, y1);
        scale = one.iter().fold(scale, |a, v| a.max(v.abs()));
        for (a, b) in grouped[r * n..(r + 1) * n].iter().zip(&one) {
            worst = worst.max((a - b).abs());
        }
    }
    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
    eprintln!(
        "parked 777 MiB weight, grouped m={m} vs per-row at k={k} n={n}: \
         max_abs={worst:e} (per-row |y|max={scale:e})"
    );
    assert!(scale > 1.0, "per-row reference is degenerate ({scale:e})");
    assert!(worst <= 1e-2, "grouped vs per-row max_abs {worst:e} > 1e-2");
}
