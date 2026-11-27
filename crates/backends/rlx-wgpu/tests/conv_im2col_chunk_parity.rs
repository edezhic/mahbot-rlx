// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The im2col col panel is cut over the output's spatial axis, so a conv whose panel is
//! past `CONV_IM2COL_PANEL_BYTES` gathers and multiplies it a chunk at a time on the one
//! reserved tail instead of materialising the whole extent. The reduction axis is never
//! cut, and the chunk is the step's `n` while the GEMM's kernel pick, tuning bucket and C
//! row stride stay the conv's whole extent (`Step::Matmul::n_geometry`) — so the cut must
//! move no number and no route.
//!
//! Both are held here against the *same binary's* whole-panel arm
//! (`RLX_WGPU_IM2COL_PANEL_BYTES=0`, the reservation this replaces), on the two shapes the
//! fork's own graphs supply: the encoder front's top-rung third conv (`480 → 480`, 3×3,
//! stride 2, on `[480, 32, 192]`) and the synthesis codec's residual lift (`256 → 64`,
//! 1×7, on `[256, 1, 4096]`). Each whole panel is past one chunk, so each is cut in two.
//!
//! What is asserted, per shape: the capped arm's step census (one gather and one GEMM per
//! chunk, no fallback), its tail scratch at or under the budget and under the whole-panel
//! reservation, the uncapped arm's tail as the whole panel 256-byte aligned, the two
//! outputs bit-identical, and — the route half, on an adapter that takes one (`conv_gemm`
//! admits `CoopF32` on Metal alone; elsewhere both arms read zero and the census is the
//! route check) — the same cooperative entry dispatched once per chunk where the whole
//! extent dispatches it once.
//!
//! The budget is a runtime-config snapshot, so each arm sets its key and calls
//! `rlx_wgpu::reload_runtime_config()`. Its own test binary: the config is one snapshot per
//! process and the neighbouring harnesses set their own.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

/// One convolution and the chunk count the shipped budget gives its panel.
struct Case {
    label: &'static str,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    /// `ceil(spatial / chunk_cols)` at the shipped 16 MiB budget.
    chunks: usize,
}

/// The two product forms, each with a whole panel past one chunk:
///
/// * the front's `[480, 32, 192]` → `[480, 16, 96]`: reduction `480·3·3 = 4320`, spatial
///   `16·96 = 1536`, whole panel `4320·1536·4 = 26 542 080 B`. The widest chunk the 16 MiB
///   budget fits is 960 columns, so the extent needs two and they are split evenly: 768
///   columns each, `768·4320·4 = 13 271 040 B`;
/// * the lift's `[256, 1, 4096]` → `[64, 1, 4096]`: reduction `256·7 = 1792`, spatial 4096,
///   whole panel `1792·4096·4 = 29 360 128 B`. The widest fit is 2336 columns, so two chunks
///   of 2048: `2048·1792·4 = 14 680 064 B`.
const CASES: &[Case] = &[
    Case {
        label: "front conv 480->480 [480,32,192] k3 s2 p1",
        c_in: 480,
        c_out: 480,
        h: 32,
        w: 192,
        k: [3, 3],
        s: [2, 2],
        p: [1, 1],
        chunks: 2,
    },
    Case {
        label: "lift 256->64 [256,1,4096] k7",
        c_in: 256,
        c_out: 64,
        h: 1,
        w: 4096,
        k: [1, 7],
        s: [1, 1],
        p: [0, 3],
        chunks: 2,
    },
];

impl Case {
    fn out_hw(&self) -> (usize, usize) {
        let axis = |size: usize, k: usize, s: usize, p: usize| {
            rlx_ir::shape::conv2d_spatial_output(size, k, s, p, 1)
        };
        (
            axis(self.h, self.k[0], self.s[0], self.p[0]),
            axis(self.w, self.k[1], self.s[1], self.p[1]),
        )
    }

    /// The whole panel this conv would gather without the cut, in bytes.
    fn whole_panel(&self) -> u64 {
        let (h_out, w_out) = self.out_hw();
        (self.c_in * self.k[0] * self.k[1]) as u64 * (h_out * w_out) as u64 * 4
    }
}

fn graph(case: &Case) -> Graph {
    let mut g = Graph::new("conv_im2col_chunk_parity");
    let x = g.input("x", Shape::new(&[1, case.c_in, case.h, case.w], DType::F32));
    let w = g.param(
        "w",
        Shape::new(&[case.c_out, case.c_in, case.k[0], case.k[1]], DType::F32),
    );
    let y = g.conv2d(x, w, case.k, case.s, case.p, [1, 1], 1);
    g.set_outputs(vec![y]);
    g
}

/// The shipped budget, spelled here so an operator's lever cannot decide these rows.
const BUDGET: u64 = 16 * 1024 * 1024;

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
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_SPATIAL", "1");
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_COUT", "64");
    rlx_ir::env::set("RLX_WGPU_IM2COL_MIN_K", "256");
    common::pin_tuning_cache_off("conv-im2col-chunk");
}

/// One arm's compile: the budget key as the arm wants it, the snapshot reloaded, then the
/// graph compiled and its weight set.
fn arm(case: &Case, ws: &[f32], budget: Option<u64>) -> WgpuExecutable {
    match budget {
        Some(bytes) => rlx_ir::env::set("RLX_WGPU_IM2COL_PANEL_BYTES", bytes.to_string()),
        None => rlx_ir::env::unset("RLX_WGPU_IM2COL_PANEL_BYTES"),
    }
    rlx_wgpu::reload_runtime_config();
    let mut exe = WgpuExecutable::compile(graph(case));
    exe.set_param("w", ws);
    exe
}

fn count(steps: &[&'static str], name: &str) -> usize {
    steps.iter().filter(|s| **s == name).count()
}

#[test]
fn a_chunked_panel_moves_no_number_and_no_route() {
    // The budget, the route switches and the floors are process-global; the lock keeps a
    // neighbouring harness's compile from reading this file's window.
    let _routes = common::route_lock();
    route_switches();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let device = rlx_wgpu::device::adapter_name().unwrap_or_else(|| "wgpu".into());
    eprintln!("conv im2col chunk parity on {device}");

    for case in CASES {
        let ws = common::ramp(1, case.c_out * case.c_in * case.k[0] * case.k[1], 0.11);
        let x = common::ramp(1, case.c_in * case.h * case.w, 0.37);
        let whole_panel = case.whole_panel();
        assert!(
            whole_panel > BUDGET,
            "{}: the fixture's panel is past one chunk, or it proves nothing",
            case.label
        );

        // ── the whole-panel arm: the reservation this cut replaces ──────────
        let before = (
            rlx_wgpu::coop_f32_dispatch_count(),
            rlx_wgpu::coop_f32_direct_dispatch_count(),
        );
        let mut whole = arm(case, &ws, Some(0));
        let whole_steps = whole.test_step_names();
        let whole_scratch = whole.test_scratch_bytes();
        let out_whole = whole.run(&[("x", &x)]).remove(0);
        let whole_coop = (
            rlx_wgpu::coop_f32_dispatch_count() - before.0,
            rlx_wgpu::coop_f32_direct_dispatch_count() - before.1,
        );

        // ── the shipped arm: the panel cut to the budget ────────────────────
        let before = (
            rlx_wgpu::coop_f32_dispatch_count(),
            rlx_wgpu::coop_f32_direct_dispatch_count(),
        );
        let mut capped = arm(case, &ws, None);
        let capped_steps = capped.test_step_names();
        let capped_scratch = capped.test_scratch_bytes();
        let out_capped = capped.run(&[("x", &x)]).remove(0);
        let capped_coop = (
            rlx_wgpu::coop_f32_dispatch_count() - before.0,
            rlx_wgpu::coop_f32_direct_dispatch_count() - before.1,
        );

        eprintln!(
            "{}: whole {:?} scratch {whole_scratch} B; capped {:?} scratch {capped_scratch} B",
            case.label, whole_steps, capped_steps
        );

        // ── the cut, in steps and in the tail ───────────────────────────────
        assert_eq!(
            count(&whole_steps, "im2col_gpu"),
            1,
            "{}: the uncapped arm gathers the whole panel once ({whole_steps:?})",
            case.label
        );
        assert_eq!(
            (
                count(&capped_steps, "im2col_gpu"),
                count(&capped_steps, "matmul")
            ),
            (case.chunks, case.chunks),
            "{}: one gather and one GEMM per chunk, and no fallback ({capped_steps:?})",
            case.label
        );
        assert_eq!(
            count(&capped_steps, "conv2d") + count(&capped_steps, "conv2d_tiled"),
            0,
            "{}: the cut does not push a conv off the panel route ({capped_steps:?})",
            case.label
        );
        assert_eq!(
            whole_scratch,
            whole_panel.div_ceil(256) * 256,
            "{}: the uncapped tail is the whole panel, 256-byte aligned",
            case.label
        );
        assert!(
            capped_scratch <= BUDGET,
            "{}: the cut's tail fits the budget ({capped_scratch} B)",
            case.label
        );
        assert!(
            capped_scratch < whole_scratch,
            "{}: and it is smaller than the reservation it replaces ({capped_scratch} vs \
             {whole_scratch} B)",
            case.label
        );

        // ── exactness: the same bytes, chunked or not ───────────────────────
        common::assert_bit_identical(
            &out_capped,
            &out_whole,
            &format!("{}: chunked against whole-panel output", case.label),
        );

        // ── the route: the same entry, once per chunk ───────────────────────
        assert_eq!(
            capped_coop,
            (
                whole_coop.0 * case.chunks as u64,
                whole_coop.1 * case.chunks as u64
            ),
            "{}: every chunk dispatches the kernel the whole extent takes — {} against {} \
             cooperative dispatches ({:?} vs {:?})",
            case.label,
            capped_coop.0,
            whole_coop.0,
            capped_steps,
            whole_steps
        );
    }
}
