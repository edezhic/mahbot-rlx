// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::ConvTranspose2d` cross-backend parity. The CPU kernel matches the
//! reference OCR (ocrs/rten) detection U-Net, so it is the reference here; every
//! GPU backend must reproduce it. This is the regression scaffold for the
//! Metal (wrong mask) / MLX (NHWC channel mismatch) / wgpu (op unsupported)
//! bugs surfaced by the OCR detection decoder.

#![allow(dead_code)]

use rlx_ir::*;
use rlx_runtime::{Device, Session};

mod common;

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    c_in: usize,
    /// The weight's `C_out/groups` — the output carries `c_out · groups` channels.
    c_out: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    op: [usize; 2],
    groups: usize,
}

impl Cfg {
    /// Both axes alike: square kernel, one stride and one pad, no dilation, no
    /// output padding, one group — the shape the OCR U-Net upsamples take.
    const fn sym(
        n: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: usize,
        s: usize,
        p: usize,
    ) -> Self {
        Self {
            n,
            c_in,
            c_out,
            h,
            w,
            k: [k, k],
            s: [s, s],
            p: [p, p],
            d: [1, 1],
            op: [0, 0],
            groups: 1,
        }
    }
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_transpose2d");
    let x = g.input("x", Shape::new(&[cfg.n, cfg.c_in, cfg.h, cfg.w], f));
    // ConvTranspose2d weight layout: [C_in, C_out/groups, kH, kW].
    let wt = g.input(
        "w",
        Shape::new(&[cfg.c_in, cfg.c_out, cfg.k[0], cfg.k[1]], f),
    );
    let y = g.conv_transpose2d(x, wt, cfg.k, cfg.s, cfg.p, cfg.d, cfg.op, cfg.groups);
    g.set_outputs(vec![y]);
    g
}

fn inputs(cfg: &Cfg) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.n * cfg.c_in * cfg.h * cfg.w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..cfg.c_in * cfg.c_out * cfg.k[0] * cfg.k[1])
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

/// U-Net-style upsamples (stride 2) plus a no-stride case.
fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        ("up2-k2", Cfg::sym(1, 8, 4, 6, 6, 2, 2, 0)),
        ("up2-k4-p1", Cfg::sym(1, 6, 5, 5, 7, 4, 2, 1)),
        ("s1-k3-p1", Cfg::sym(2, 4, 3, 5, 5, 3, 1, 1)),
        // The detection U-Net decoder's actual upsample: kernel 3, stride 2, pad 1.
        ("up2-k3s2-p1", Cfg::sym(1, 8, 4, 6, 6, 3, 2, 1)),
        ("up2-k3s2-p1-odd", Cfg::sym(1, 5, 3, 7, 5, 3, 2, 1)),
        // U-Net scale.
        ("large-up2-k3s2", Cfg::sym(1, 64, 32, 32, 32, 3, 2, 1)),
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
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

#[test]
fn conv_transpose2d_cpu_runs() {
    let _gpu = common::serialize_gpu();
    for (name, cfg) in cfgs() {
        let out = run_on(&cfg, Device::Cpu);
        let expect = cfg.n
            * cfg.c_out
            * ((cfg.h - 1) * cfg.s[0] + cfg.k[0] - 2 * cfg.p[0])
            * ((cfg.w - 1) * cfg.s[1] + cfg.k[1] - 2 * cfg.p[1]);
        assert_eq!(
            out.len(),
            expect,
            "cpu {name}: output len {} != {expect}",
            out.len()
        );
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
        eprintln!("cpu {name}: {} elems ok", out.len());
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn conv_transpose2d_metal_matches_cpu() {
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
fn conv_transpose2d_mlx_matches_cpu() {
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
fn conv_transpose2d_wgpu_matches_cpu() {
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

/// The geometry a vocoder actually presents — a 1-D-lifted kernel `[1, 16]` at
/// stride `[1, 8]` on unit height — plus the grouped / dilated / padded /
/// output-padded family the swapped conv-backward-input route has to reproduce
/// exactly, and the degenerate grouping its whole-arena fallback keeps zeroed.
#[test]
#[cfg(feature = "gpu")]
fn conv_transpose2d_wgpu_decoder_and_family_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let cases = [
        (
            "wgpu decoder-1d-lifted-k16s8",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 6,
                h: 1,
                w: 96,
                k: [1, 16],
                s: [1, 8],
                p: [0, 0],
                d: [1, 1],
                op: [0, 0],
                groups: 1,
            },
        ),
        (
            "wgpu grouped-g2",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 6,
                h: 6,
                w: 6,
                k: [3, 3],
                s: [2, 2],
                p: [1, 1],
                d: [1, 1],
                op: [0, 0],
                groups: 2,
            },
        ),
        (
            "wgpu dilated-d2",
            Cfg {
                n: 1,
                c_in: 5,
                c_out: 3,
                h: 5,
                w: 7,
                k: [3, 3],
                s: [2, 2],
                p: [2, 2],
                d: [2, 2],
                op: [0, 0],
                groups: 1,
            },
        ),
        (
            "wgpu output-padding-op1",
            Cfg {
                n: 1,
                c_in: 6,
                c_out: 4,
                h: 6,
                w: 6,
                k: [3, 3],
                s: [2, 2],
                p: [1, 1],
                d: [1, 1],
                op: [1, 1],
                groups: 1,
            },
        ),
        (
            "wgpu grouped-dilated-padded-asymmetric",
            Cfg {
                n: 1,
                c_in: 12,
                c_out: 6,
                h: 5,
                w: 5,
                k: [3, 2],
                s: [2, 1],
                p: [1, 0],
                d: [2, 1],
                op: [1, 0],
                groups: 3,
            },
        ),
        (
            "wgpu degenerate-grouping-g8",
            Cfg {
                n: 1,
                c_in: 4,
                c_out: 6,
                h: 4,
                w: 4,
                k: [3, 3],
                s: [2, 2],
                p: [1, 1],
                d: [1, 1],
                op: [0, 0],
                groups: 8,
            },
        ),
    ];
    for (name, cfg) in cases {
        let got = run_on(&cfg, Device::Gpu);
        let want = run_on(&cfg, Device::Cpu);
        assert_close(name, &got, &want);
        if cfg.groups > cfg.c_in {
            // Today's behaviour, preserved: `groups > C_in` has no channel to
            // accumulate from, so both sides leave the output zeroed rather than
            // define a different operator out of a shape nothing computes.
            assert!(
                want.iter().all(|v| *v == 0.0),
                "{}: the cpu reference must leave this degenerate grouping zeroed",
                name
            );
            assert!(
                got.iter().all(|v| *v == 0.0),
                "{}: the wgpu route must leave this degenerate grouping zeroed",
                name
            );
        }
    }
}
