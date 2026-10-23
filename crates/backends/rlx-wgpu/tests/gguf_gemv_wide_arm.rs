// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution of the one-row packed multiply's two arms: the subgroup arm —
//! `dequant_gemv_wide`, the shipped default (`gguf_gpu`'s `GEMV_WIDE_DEFAULT`),
//! with its ILP entry — and the workgroup-memory arm it falls back to on an adapter
//! without `Features::SUBGROUP`, reached here with `RLX_WGPU_GEMV_WIDE=shared`.
//!
//! This file is the only gate that *executes* either entry point: every other test
//! either leaves the knob unset or pins its own parsing (the device-free
//! `the_wide_knob_refuses_on_zero_and_shared`), so a wrong `subgroupAdd` tree, a wrong
//! column-to-subgroup map or a wrong lane stride in the wide kernel, and a broken
//! fallback, would otherwise be exercised by nothing.
//!
//! What it pins, per scheme the arm carries (Q4_K, Q6_K, Q8_0) at `m = 1` and the
//! widths the model bodies use: the overrides reach the encoder (`gemv_one_row_shape`
//! records which entry ran), the result matches the CPU reference of the same op, and
//! no value is non-finite. The bar is not bit identity: the wide arm reduces a column
//! with `subgroupAdd`, a tree whose shape the subgroup width sets, while the tiled arm's
//! is a fixed workgroup-memory one, so a different order of the same fp32 sum is
//! expected. `1e-2` is the bar every packed multiply in this crate is compared at,
//! these shapes included (`gguf_matmul_route.rs`).
//!
//! One `#[test]` on purpose: the override is process-global (`RuntimeOverrides`
//! installs into one shared map), so a second cell running in parallel in this process
//! would race it. On a device without `Features::SUBGROUP` the subgroup cells are
//! reported as skipped and the workgroup-memory arm — the one such a device runs —
//! is still checked.

mod common;

use common::max_abs_diff;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{
    GEMV_ILP_MIN_COLS, GEMV_ILP_Q6K_MIN_COLS, GEMV_WIDE_ILP, gemv_one_row_shape,
    gemv_subgroup_width, gemv_wide_cols_per_wg, reset_gemv_one_row_shape,
};

/// One packed `[n, k]` weight, its row-major `[1, k]` activation, and the
/// `DequantMatMul` that contracts them.
fn case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_gemv_wide");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x)
}

fn run(device: Device, g: Graph, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn the_subgroup_arm_matches_the_cpu_reference_on_every_scheme_it_carries() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let dev =
        rlx_wgpu::wgpu_device().expect("wgpu device: the availability check just asked for one");
    let width = gemv_subgroup_width(&dev.device);

    // The knob is read at the point of decision — once per dispatch — so the override
    // has to be live while the calls run, not just while the graph is compiled.
    // `None` is the shipped default (the subgroup arm), `"shared"` the fallback.
    let schemes = [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0),
    ];
    // The widths the model bodies use: H=1024 and the down-projection's doubled
    // contraction, at the square and the model-width output — plus the ILP entry's
    // admission edges (`n` exactly at its floor, and a chunk above the floor that is
    // not a whole number of its workgroups, which the narrow entry serves) and an
    // `n` below the floor, which the floor rejects. The off-width chunk is 4 100
    // because it sits 4 columns off a whole workgroup at any subgroup width this
    // crate has probed (4 mod 8 at 32, 4 mod 16 at 16); a value that divided by the
    // width would silently take the ILP entry and drop this case.
    let shapes = [
        (1024usize, 1024usize),
        (2048usize, 2048usize),
        // The body's two-unit narrow row: the chunk `GEMV_ILP_Q6K_MIN_COLS`
        // keeps on the one-column entry.
        (2048usize, 1024usize),
        // The body's Q4_K ffn_down: a row that gives a lane several `qs` groups,
        // which is where the subgroup arm walks whole groups instead of sub-blocks.
        (6144usize, 2048usize),
        // The same row four columns under the ILP entry's floor, so the one-column
        // entry runs that group cover rather than the pair entry running it.
        (6144usize, 1020usize),
        (1024usize, 6144usize),
        (1024usize, 1020usize),
        (1024usize, 4096usize),
        (1024usize, 4100usize),
    ];

    // `"1"` states the shipped arm rather than inheriting whatever the window
    // exported (`None` would pin the ambient `RLX_WGPU_GEMV_WIDE`, which the documented
    // `=shared` spelling turns into the other arm); `"shared"` is the fallback. The
    // ILP knob is pinned for the same reason: `=0` is a documented spelling, and a
    // window that exported it would make these cells assert an environment.
    for (knob, want_wide) in [("1", true), ("shared", false)] {
        let label = knob;
        if want_wide && !dev.device.features().contains(wgpu::Features::SUBGROUP) {
            // The subgroup arm can never be admitted here; the fallback below still
            // runs, so this file keeps covering the arm such a device executes.
            eprintln!("[gguf-gemv-wide] skipped the subgroup arm: no wgpu::Features::SUBGROUP");
            continue;
        }
        let _knob = rlx_ir::env::RuntimeOverrides::install([
            ("RLX_WGPU_GEMV_WIDE", knob),
            ("RLX_WGPU_GEMV_ILP", "1"),
        ]);
        for (scheme, ggml) in schemes {
            for (k, n) in shapes {
                let (g, packed, x) = case(scheme, ggml, k, n);
                reset_gemv_one_row_shape();
                let gpu = run(Device::Gpu, g.clone(), &packed, &x);
                // The recorded shape is what says the knob reached the encoder: only the
                // subgroup arm records one of its entries, and at m = 1 the ladder has no
                // other subgroup route.
                let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
                assert_eq!(
                    matches!(shape.arm, 1 | 2),
                    want_wide,
                    "{scheme:?} m=1 k={k} n={n} [{label}]: the probed subgroup width is \
                     {width:?}, so the record must say whether the wide arm ran: {shape:?}"
                );
                // Which entry the chunk took, from the arm's own rules: `cols *
                // WIDE_ILP` when the chunk clears the entry's floor and divides
                // into whole workgroups — the Q6_K floor where the row is two
                // units a lane — and the one-column-per-subgroup entry otherwise.
                if want_wide {
                    let cols = gemv_wide_cols_per_wg(&dev.device)
                        .expect("the subgroup arm is admitted here");
                    let sg = gemv_subgroup_width(&dev.device)
                        .expect("the subgroup arm is admitted here");
                    let narrow_q6k = scheme == QuantScheme::GgufQ6K && k / 32 == 2 * sg as usize;
                    let floor = if narrow_q6k {
                        GEMV_ILP_Q6K_MIN_COLS
                    } else {
                        GEMV_ILP_MIN_COLS
                    };
                    let pair = cols * GEMV_WIDE_ILP;
                    let (want_arm, expected) = if n >= floor && (n as u32).is_multiple_of(pair) {
                        (1u32, pair)
                    } else {
                        (2, cols)
                    };
                    assert_eq!(
                        (shape.arm, shape.cols_per_workgroup),
                        (want_arm, expected),
                        "{scheme:?} k={k} n={n} [{label}]: the record must say which \
                         subgroup entry ran"
                    );
                }
                let cpu = run(Device::Cpu, g, &packed, &x);
                let max_abs = max_abs_diff(&gpu, &cpu);
                eprintln!(
                    "[gguf-gemv-wide] knob={label} scheme={scheme:?} m=1 k={k} n={n} \
                     arm={} cols_per_workgroup={} workgroups={} max_abs={max_abs:.6e}",
                    shape.arm, shape.cols_per_workgroup, shape.workgroups
                );
                // The entries sum the same fp32 products in different orders —
                // `subgroupAdd`'s tree, one tree a column, against the workgroup-memory
                // tree — so exact agreement is not the claim; `1e-2` is this crate's bar
                // for a packed multiply at these shapes.
                assert!(
                    max_abs <= 1e-2,
                    "{scheme:?} m=1 k={k} n={n} [{label}]: the arm is {max_abs} from the CPU \
                     reference"
                );
                assert!(
                    gpu.iter().all(|v| v.is_finite()),
                    "{scheme:?} m=1 k={k} n={n} [{label}]: the arm produced a non-finite value"
                );
            }
        }
    }
}
