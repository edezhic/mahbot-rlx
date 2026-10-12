// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Conv2d` grouped/depthwise cross-backend parity — the detection U-Net is
//! depthwise-separable (`conv2d_bias_groups` with groups = channels), a classic
//! GPU-backend trap. CPU is the reference (matches rten); every GPU backend must
//! match it.

#![allow(dead_code)]

use rlx_ir::*;
use rlx_runtime::{Device, Session};

mod common;

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    /// Per-axis kernel taps. The unit-height 1-D lifts below carry the length
    /// taps in slot 1 (`k: [1, 7]`), which is the shape a decoder/vocoder body
    /// presents; the ONNX 1-D lift carries them in slot 0 (`k: [7, 1]`).
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    /// Per-axis dilation. Unit everywhere except the mirrored case below, which
    /// is what makes the relabel's `dh` (the only field it takes from slot 1
    /// besides taps/stride/padding) carry a parity reference.
    d: [usize; 2],
    groups: usize,
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv2d");
    let x = g.input("x", Shape::new(&[cfg.n, cfg.c_in, cfg.h, cfg.w], f));
    // Conv2d weight layout: [C_out, C_in/groups, kH, kW].
    let wt = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k[0], cfg.k[1]], f),
    );
    let y = g.conv2d(x, wt, cfg.k, cfg.s, cfg.p, cfg.d, cfg.groups);
    g.set_outputs(vec![y]);
    g
}

fn inputs(cfg: &Cfg) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.n * cfg.c_in * cfg.h * cfg.w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k[0] * cfg.k[1])
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.03)
        .collect();
    (x, w)
}

fn run_on(cfg: &Cfg, device: Device) -> Vec<f32> {
    let (x, w) = inputs(cfg);
    let mut c = Session::new(device).compile(build(cfg));
    c.run(&[("x", x.as_slice()), ("w", w.as_slice())])
        .pop()
        .unwrap()
}

fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "regular",
            Cfg {
                n: 1,
                c_in: 4,
                c_out: 6,
                h: 6,
                w: 6,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "depthwise-3x3",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 6,
                w: 6,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 8,
            },
        ),
        (
            "depthwise-s2",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 7,
                w: 7,
                k: [3, 3],
                s: [2, 2],
                p: [1, 1],
                d: [1, 1],
                groups: 8,
            },
        ),
        (
            "grouped-2",
            Cfg {
                n: 2,
                c_in: 8,
                c_out: 8,
                h: 5,
                w: 5,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 2,
            },
        ),
        // U-Net scale — exposes size/channel-dependent GPU tiling bugs.
        (
            "large-regular",
            Cfg {
                n: 1,
                c_in: 64,
                c_out: 64,
                h: 64,
                w: 64,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "large-depthwise",
            Cfg {
                n: 1,
                c_in: 128,
                c_out: 128,
                h: 50,
                w: 50,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 128,
            },
        ),
        (
            "large-1x1",
            Cfg {
                n: 1,
                c_in: 128,
                c_out: 64,
                h: 50,
                w: 50,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        // Just inside the band the wgpu im2col spatial floor's new default (800,
        // from 2048) newly admits: 40×40 = 1600 output elements, so on that backend
        // this row takes the materialised im2col route. It is the 2-D half of that
        // band's regression surface — the unit-height body rows below reach the
        // floor only through the length axis, and this is the only 2-D map in the
        // band (`dense-1x7-l800` sits on the floor itself, at its inclusive lower bound).
        (
            "band-3x3-40x40",
            Cfg {
                n: 1,
                c_in: 64,
                c_out: 64,
                h: 40,
                w: 40,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 1,
            },
        ),
        // Detection in_conv resolution (800×600) — exposes spatial-size kernel bugs.
        (
            "hires-3x3",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "hires-dw",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: [3, 3],
                s: [1, 1],
                p: [1, 1],
                d: [1, 1],
                groups: 8,
            },
        ),
        (
            "hires-1x1",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "hires-1ch",
            Cfg {
                n: 1,
                c_in: 1,
                c_out: 8,
                h: 800,
                w: 600,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        // ── Decoder/vocoder body, unit-height 1-D lifts ──────────────────────
        // The body presents `[N,C,1,L]` with the length taps in slot 1
        // (`k: [1,7]`, `p: [0,3]`, `s: [1,1]`), so an input of width L gives an
        // output of width L. These pin the *values* on the arm of
        // `RLX_WGPU_CONV_IM2COL` in force (the default: routes on, which is the
        // first live exercise of the accelerated routes) — the *routing* each
        // shape must take is asserted by the wgpu crate's `conv_route_cost`
        // route table, on both arms.
        //
        // 1×7 dense — clears every im2col floor ⇒ accelerated im2col+GEMM.
        (
            "dense-1x7-l4000",
            Cfg {
                n: 1,
                c_in: 384,
                c_out: 384,
                h: 1,
                w: 4000,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "dense-1x7-l16000",
            Cfg {
                n: 1,
                c_in: 192,
                c_out: 192,
                h: 1,
                w: 16000,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "dense-1x7-l48000",
            Cfg {
                n: 1,
                c_in: 96,
                c_out: 96,
                h: 1,
                w: 48000,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        // 1×7 dense sitting exactly on the 800 im2col spatial floor (out 800 is
        // that floor's own boundary, inclusive) ⇒ im2col — the shape the decoder
        // body's `blk.*.res.*.conv1` presents.
        (
            "dense-1x7-l800",
            Cfg {
                n: 1,
                c_in: 768,
                c_out: 768,
                h: 1,
                w: 800,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        // The body's mirrored final projection: one output channel clears the
        // tiled floor but not the im2col `c_out` floor ⇒ mirrored relabel.
        (
            "mirror-projection",
            Cfg {
                n: 1,
                c_in: 96,
                c_out: 1,
                h: 1,
                w: 48000,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        // ... and with a stride: out 2000 clears the 800 im2col spatial floor, but
        // `c_out = 32` does not clear its 64-channel floor ⇒ mirrored relabel with
        // a stride.
        (
            "mirror-strided-s2",
            Cfg {
                n: 1,
                c_in: 768,
                c_out: 32,
                h: 1,
                w: 4000,
                k: [1, 7],
                s: [1, 2],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        // ... and with dilation: out 794 sits above the 256 tiled floor and under
        // the 800 im2col spatial floor, so this is the *mirrored* relabel with a
        // non-unit `dh` — the one relabel field the strided case above cannot
        // cover — compared against CPU like every other case here.
        (
            "mirror-dilated-d2",
            Cfg {
                n: 1,
                c_in: 384,
                c_out: 384,
                h: 1,
                w: 800,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 2],
                groups: 1,
            },
        ),
        // The body's entry shape: below every route's floor ⇒ the plain direct
        // kernel; this is the shape the switch must leave alone.
        (
            "entry-1x7-l100",
            Cfg {
                n: 1,
                c_in: 1024,
                c_out: 1536,
                h: 1,
                w: 100,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1,
            },
        ),
        // The body's depthwise 1×7 (`groups == c_in`): excluded from every
        // accelerated route ⇒ the plain direct kernel.
        (
            "depthwise-1x7-l50",
            Cfg {
                n: 1,
                c_in: 1024,
                c_out: 1024,
                h: 1,
                w: 50,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1024,
            },
        ),
        (
            "depthwise-1x7-l100",
            Cfg {
                n: 1,
                c_in: 1024,
                c_out: 1024,
                h: 1,
                w: 100,
                k: [1, 7],
                s: [1, 1],
                p: [0, 3],
                d: [1, 1],
                groups: 1024,
            },
        ),
        // The shape the mirror must *refuse* (H a real axis: height padding 1 ⇒
        // out 3×200) is deliberately NOT here: the shared suite runs every case
        // on every arm, and the MLX backend's own 1-D lift takes such a shape and
        // fails its reshape — its guard, like the wgpu mirror's, has to exclude a
        // padded height axis, but that is a sibling backend's defect to fix, not
        // this change's. The refusal is pinned instead in the wgpu-only route
        // table (`conv_route_cost::expected_routes`, row `refuse-height-pad`),
        // where the shape is discriminable: its output spatial sits above the 256
        // tiled floor and under the 800 im2col floor, so a lost refusal relabels it
        // into the tiled route and fails that assertion.
        // 1×1 pointwise: routed to the `Step::Matmul` GEMM when the activation
        // clears that route's own frozen bar (`c_in * h_out * w_out` at least
        // `256 × 2048`) — all four are. The bar is a number of its own, not the
        // im2col route's `min_k * min_spatial`, which has to be free to move.
        (
            "pointwise-1x1-l800",
            Cfg {
                n: 1,
                c_in: 768,
                c_out: 768,
                h: 1,
                w: 800,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "pointwise-1x1-l4000",
            Cfg {
                n: 1,
                c_in: 384,
                c_out: 384,
                h: 1,
                w: 4000,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "pointwise-1x1-l16000",
            Cfg {
                n: 1,
                c_in: 192,
                c_out: 192,
                h: 1,
                w: 16000,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        (
            "pointwise-1x1-l48000",
            Cfg {
                n: 1,
                c_in: 96,
                c_out: 96,
                h: 1,
                w: 48000,
                k: [1, 1],
                s: [1, 1],
                p: [0, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
        // The existing ONNX 1-D lift — length taps in slot 0 (`k: [7,1]`,
        // `p: [3,0]`) — took `conv1d_tiled` before the flip and now takes the
        // im2col route when the routes are on.
        (
            "overlap-1x7-slot0",
            Cfg {
                n: 1,
                c_in: 384,
                c_out: 384,
                h: 1,
                w: 4000,
                k: [7, 1],
                s: [1, 1],
                p: [3, 0],
                d: [1, 1],
                groups: 1,
            },
        ),
    ]
}

fn assert_close(what: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{what}: length {} vs {}",
        actual.len(),
        reference.len()
    );
    // Finiteness census FIRST, the rule the wgpu suite's `common::max_abs_diff`
    // carries: `f32::max` returns the non-NaN operand, so an all-NaN or
    // infinity-poisoned output would fold to a small number and pass below.
    assert!(
        actual.iter().chain(reference).all(|v| v.is_finite()),
        "{what}: non-finite in the parity pair — it must not fold through `f32::max`"
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

#[test]
fn conv2d_groups_cpu_runs() {
    let _gpu = common::serialize_gpu();
    for (name, cfg) in cfgs() {
        let out = run_on(&cfg, Device::Cpu);
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
        eprintln!("cpu {name}: {} elems ok", out.len());
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn conv2d_groups_metal_matches_cpu() {
    let _gpu = common::serialize_gpu();
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("metal {name}"),
            &run_on(&cfg, Device::Metal),
            &run_on(&cfg, Device::Cpu),
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn conv2d_groups_mlx_matches_cpu() {
    let _gpu = common::serialize_gpu();
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("mlx {name}"),
            &run_on(&cfg, Device::Mlx),
            &run_on(&cfg, Device::Cpu),
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn conv2d_groups_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(rlx_runtime::Device::Gpu, "wgpu") {
        return;
    }
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("wgpu {name}"),
            &run_on(&cfg, Device::Gpu),
            &run_on(&cfg, Device::Cpu),
        );
    }
}
