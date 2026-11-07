// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The im2col+GEMM convolution route serves a batch, one panel per image. The batch
//! axis is the panel's: the gather's `gid.z` is the image, so image `i` is read from
//! `in_off + i·c_in·h·w` and written to `col_off + i·k_total·spatial`, and the GEMM
//! carries `batch: n` over the same z with a zero weight batch stride (the weight is
//! shared by every image) and panel/output strides of `k_total·spatial` /
//! `c_out·spatial`; the reservation is `n · k_total · spatial`, so the planner and
//! the lowering agree.
//!
//! Two of the wake/recognition front's own convolutions — the first front conv
//! (`[n, 1, 128, 76]` → `[n, 480, 64, 38]`) and the channel-heavy `[n, 480, 32, 19]`
//! → `[n, 480, 16, 10]` form — are compiled at batch 1 and batch 2. Each must carry
//! the same `im2col_gpu` step — not the direct kernel, not a second path — and the
//! unit-channel case's batch-2 schedule must hold one panel step and one GEMM. The
//! whole `[2, c_out, h_out, w_out]` output and each image are held to
//! `rlx_runtime::Session` on `Device::Cpu` running the *same* batch-2 graph under
//! `common::assert_close_under`'s `1e-5` bar; image 0 is bit-identical when image 1
//! is replaced by another window, and image 0 of batch 2 is held to batch 1 under
//! the same bar — the bar is the contract, not the reading, since the GEMM's
//! accumulation order is not fixed.
//!
//! The first shape's contraction is `c_in·kh·kw = 9`, below the shipped
//! `RLX_WGPU_IM2COL_MIN_K` floor of 256, so the *admission* is not what this file
//! pins: the floor is lowered to 1 for the duration of the test (the lever class
//! `conv_route_cost.rs` uses) so the batched arrangement is exercised on the
//! product's geometry, while the channel-heavy case clears every shipped floor on
//! its own. The floors come from the runtime config snapshot, so the test calls
//! `rlx_wgpu::reload_runtime_config()` after setting them.
//!
//! The operands are `common::ramp` sweeps (never a NaN or denormal), and no time is
//! asserted anywhere. Its own test binary: the K floor is process-global and the
//! loaded config is one snapshot per process.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

/// One convolution: the input's `[c_in, h, w]` and the weight's `[c_out, c_in]`,
/// the kernel/stride/padding, and the label the assertions carry.
struct Case {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
}

/// The two product forms this file pins: the wake/recognition front's first conv
/// (unit channel, heavy output) and its channel-heavy sibling (480 → 480).
const CASES: &[Case] = &[
    Case {
        label: "front conv [n,1,128,76] 1->480 k=3 s=2 p=1",
        c_in: 1,
        c_out: 480,
        h: 128,
        w: 76,
        k: [3, 3],
        s: [2, 2],
        p: [1, 1],
    },
    Case {
        label: "channel-heavy [n,480,32,19] 480->480 k=3 s=2 p=1",
        c_in: 480,
        c_out: 480,
        h: 32,
        w: 19,
        k: [3, 3],
        s: [2, 2],
        p: [1, 1],
    },
];

impl Case {
    /// The output's `[c_out, h_out, w_out]` from the tree's own shape rule.
    fn out_hw(&self) -> (usize, usize) {
        let axis = |size: usize, k: usize, s: usize, p: usize| {
            rlx_ir::shape::conv2d_spatial_output(size, k, s, p, 1)
        };
        (
            axis(self.h, self.k[0], self.s[0], self.p[0]),
            axis(self.w, self.k[1], self.s[1], self.p[1]),
        )
    }

    /// Elements in one image of the input and of the output.
    fn in_per_image(&self) -> usize {
        self.c_in * self.h * self.w
    }
    fn out_per_image(&self) -> usize {
        let (h, w) = self.out_hw();
        self.c_out * h * w
    }
    fn weight_len(&self) -> usize {
        self.c_out * self.c_in * self.k[0] * self.k[1]
    }
}

/// The conv graph at `batch`, one `Op::Conv` node and nothing else.
fn graph(case: &Case, batch: usize) -> Graph {
    let mut g = Graph::new("conv_batched_im2col_parity");
    let x = g.input(
        "x",
        Shape::new(&[batch, case.c_in, case.h, case.w], DType::F32),
    );
    let w = g.param(
        "w",
        Shape::new(&[case.c_out, case.c_in, case.k[0], case.k[1]], DType::F32),
    );
    let y = g.conv2d(x, w, case.k, case.s, case.p, [1, 1], 1);
    g.set_outputs(vec![y]);
    g
}

/// The process-global switches this file needs: the host/route flips `common`
/// neutralises, the routes themselves on, the tuning cache pinned off (a stored
/// per-shape override would change the kernel under the comparison), and the K
/// floor lowered so the unit-channel front conv reaches the panel route. The floor
/// is read from the runtime config snapshot, so the snapshot is reloaded last.
fn route_switches() {
    common::neutralise_device_route_env();
    for key in [
        "RLX_WGPU_NO_TILED_CONV",
        "RLX_WGPU_CONV_HOST",
        "RLX_WGPU_MATMUL_F32_ONLY",
        "RLX_WGPU_FORCE_COOP_F32",
        "RLX_WGPU_NO_COOP_F32",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_CONV_IM2COL", "1");
    // The shipped spatial/channel floors, spelled so an operator's lever cannot decide
    // these rows; the K floor is the one this file lowers (see the doc comment).
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_SPATIAL", "100");
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_COUT", "64");
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_K", "1");
    common::pin_tuning_cache_off("conv-batched-im2col");
    rlx_wgpu::reload_runtime_config();
}

/// How many schedule steps carry this name.
fn count(steps: &[&'static str], name: &str) -> usize {
    steps.iter().filter(|s| **s == name).count()
}

/// Image `img`'s contiguous block of a `[n, c_out, h_out, w_out]` output.
fn image_block(v: &[f32], img: usize, per_image: usize) -> &[f32] {
    &v[img * per_image..(img + 1) * per_image]
}

#[test]
fn a_batched_im2col_conv_keeps_the_route_and_each_image_in_its_own_rows() {
    // The K floor and the route switches are process-global; the lock keeps a
    // neighbouring harness's compile from reading this file's window.
    let _routes = common::route_lock();
    route_switches();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    let device = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!("conv batched im2col parity on {device}");

    for case in CASES {
        let ws = common::ramp(1, case.weight_len(), 0.11);
        // Image 0 and image 1 are different windows; the two runs below only ever
        // change the second one.
        let a = common::ramp(1, case.in_per_image(), 0.37);
        let b = common::ramp(1, case.in_per_image(), 0.53);
        let b2 = common::ramp(1, case.in_per_image(), 0.71);
        let two: Vec<f32> = a.iter().chain(&b).copied().collect();
        let one = a.clone();

        // ── 1. the route, at both batches ────────────────────────────────────
        let mut exe1 = WgpuExecutable::compile(graph(case, 1));
        exe1.set_param("w", &ws);
        let steps1 = exe1.test_step_names();
        let mut exe2 = WgpuExecutable::compile(graph(case, 2));
        exe2.set_param("w", &ws);
        let steps2 = exe2.test_step_names();
        let (h_out, w_out) = case.out_hw();
        eprintln!(
            "{}: out [2, {}, {h_out}, {w_out}]; batch-1 steps {:?}; batch-2 steps {:?}",
            case.label, case.c_out, steps1, steps2
        );
        assert!(
            count(&steps2, "im2col_gpu") >= 1,
            "{}: batch 2 lost the panel route ({steps2:?})",
            case.label
        );
        assert!(
            count(&steps1, "im2col_gpu") >= 1,
            "{}: batch 1 does not take the panel route ({steps1:?})",
            case.label
        );
        if case.c_in == 1 {
            assert_eq!(
                count(&steps2, "im2col_gpu"),
                1,
                "{}: one conv, one panel step",
                case.label
            );
            assert_eq!(
                count(&steps2, "matmul"),
                1,
                "{}: one conv, one panel GEMM",
                case.label
            );
        }

        // ── 2. parity against the CPU conv, whole output and per image ───────
        let gpu2 = exe2.run(&[("x", &two)]).remove(0);
        let mut cpu = Session::new(Device::Cpu).compile(graph(case, 2));
        cpu.set_param("w", &ws);
        let cpu2 = cpu.run(&[("x", &two)]).remove(0);
        assert_eq!(gpu2.len(), 2 * case.out_per_image());
        let whole = common::assert_close_under(case.label, "gpu batch 2", &gpu2, &cpu2);
        let per_out = case.out_per_image();
        let d0 = common::max_abs_diff(
            image_block(&gpu2, 0, per_out),
            image_block(&cpu2, 0, per_out),
        );
        let d1 = common::max_abs_diff(
            image_block(&gpu2, 1, per_out),
            image_block(&cpu2, 1, per_out),
        );
        eprintln!("  vs CPU conv: max|Δ| whole {whole:e}, image 0 {d0:e}, image 1 {d1:e}");

        // ── 3. no leak between images ────────────────────────────────────────
        let two_b2: Vec<f32> = a.iter().chain(&b2).copied().collect();
        let gpu2_b2 = exe2.run(&[("x", &two_b2)]).remove(0);
        common::assert_bit_identical(
            image_block(&gpu2, 0, per_out),
            image_block(&gpu2_b2, 0, per_out),
            &format!("{}: image 0 moved when image 1 changed", case.label),
        );
        let gpu1 = exe1.run(&[("x", &one)]).remove(0);
        let d_batch = common::assert_close_under(
            case.label,
            "gpu batch 2 image 0 vs batch 1",
            image_block(&gpu2, 0, per_out),
            &gpu1,
        );
        eprintln!(
            "  image 0: bit-identical when image 1 moves; batch 2 vs batch 1 max|Δ| {d_batch:e}"
        );
    }
}
