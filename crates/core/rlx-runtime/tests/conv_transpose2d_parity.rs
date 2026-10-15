// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::ConvTranspose2d` cross-backend parity. The CPU kernel matches the
//! reference OCR (ocrs/rten) detection U-Net, so it is the reference here; every
//! GPU backend must reproduce it. This is the regression scaffold for the
//! Metal (wrong mask) / MLX (NHWC channel mismatch) / wgpu (then unsupported)
//! bugs surfaced by the OCR detection decoder.
//!
//! wgpu's default route for this op is now a device step: the phase-decomposed
//! tiled GEMM (`conv_transpose2d_tiled`) for the unit-height, unit-batch,
//! ungrouped, `k ≤ 2·s` family — the decoder's six forms among them — and the
//! gather over the whole operator for every shape outside it. The CPU-kernel host
//! route is kept behind `RLX_WGPU_CONV_HOST` (or the structure-host preference),
//! and is the default only where `coop_discrete_backend()` is true: on a discrete
//! Vulkan/DX12 adapter that preference hosts this op. Against that CPU reference
//! this file pins the wgpu device route at a fixed `1e-4` bar, and the worst
//! deviation measured over the whole case set, the decoder forms at the bottom
//! included, is `2.1e-6` (on `blk.0`), so the bar holds with a ~48× margin and is
//! deliberately NOT re-based. The per-case numbers
//! are NOT enumerated here: they go stale silently and nothing in this file
//! checks them, and the run prints them itself (`-- --nocapture`). The tolerance
//! is *argued* rather than asserted in
//! `crates/backends/rlx-wgpu/tests/conv_transpose2d_gather_family.rs`, which
//! derives its bar per case from the operands and the terms that land in the
//! worst output element — so reading this header is no substitute for running
//! the module. The geometry family the device route cannot serve is the
//! degenerate grouping (`groups == 0`, or a `groups` that does not divide `c_in`),
//! which it refuses rather than running as a different operator. Those cases are
//! re-pointed at the bottom of this file, and the two non-divisible shapes are
//! kept apart on purpose: `groups > c_in` is the zeroing shape the host route
//! reproduced, so its refusal still names that route, while `c_in > groups` with a
//! remainder is a geometry no route computes (`groups == 0` divides by a zero, and
//! the host route's scatter walks past its own output), so those refusals name no
//! way back to the host.

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

    /// Kernel, stride, padding and output padding per axis, no dilation and one
    /// group — the 1-D-lifted decoder shapes `sym` cannot express.
    const fn asym(
        n: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: [usize; 2],
        s: [usize; 2],
        p: [usize; 2],
        op: [usize; 2],
    ) -> Self {
        Self {
            n,
            c_in,
            c_out,
            h,
            w,
            k,
            s,
            p,
            d: [1, 1],
            op,
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

/// The six real decoder forms, in the order the run prints them: single batch,
/// unit height (`h = kH = sH = 1`, `pH = 0`), one group, no dilation, and the
/// width axis the decoder's transposed convolutions carry. The labels are the
/// decoder's own blocks and `c_in`/`c_out`/`w` its real channel counts and
/// input widths. The width padding and output padding are the stand's
/// generator's, `padding = (k − s + 1)/2` and `output_padding = (k − s) % 2`:
/// the two `up.blk` forms are the upsample `k = s = 2`, where both are zero, and
/// the four `blk` forms are `k = 2·s`, so the extra output column appears exactly
/// where `k − s` is odd — `blk.1` and `blk.3`.
///
/// Kept out of `cfgs()` so that the Metal and MLX arms stay on the small
/// geometries this reading is not about.
fn decoder_cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "up.blk.0",
            Cfg::asym(1, 1024, 1024, 1, 25, [1, 2], [1, 2], [0, 0], [0, 0]),
        ),
        (
            "up.blk.1",
            Cfg::asym(1, 1024, 1024, 1, 50, [1, 2], [1, 2], [0, 0], [0, 0]),
        ),
        (
            "blk.0",
            Cfg::asym(1, 1536, 768, 1, 100, [1, 16], [1, 8], [0, 4], [0, 0]),
        ),
        (
            "blk.1",
            Cfg::asym(1, 768, 384, 1, 800, [1, 10], [1, 5], [0, 3], [0, 1]),
        ),
        (
            "blk.2",
            Cfg::asym(1, 384, 192, 1, 4000, [1, 8], [1, 4], [0, 2], [0, 0]),
        ),
        (
            "blk.3",
            Cfg::asym(1, 192, 96, 1, 16000, [1, 6], [1, 3], [0, 2], [0, 1]),
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
    // `f32::max` ignores NaN, so folding with it would let a NaN on either side
    // (or an `inf - inf` pair) disappear into the running maximum and pass; a NaN
    // maximum does not compare `<= 1e-4`, so propagate it instead.
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, |m, d| {
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        });
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

/// The six real decoder forms on the wgpu device route: the same `1e-4` bar and the
/// same CPU-device reference as the arm above, on the geometries the decoder actually
/// presents rather than the small uniform ones. Its own list, so the Metal and MLX
/// arms above keep the case set they were written for.
#[test]
#[cfg(feature = "gpu")]
fn conv_transpose2d_wgpu_decoder_forms_match_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    for (name, cfg) in decoder_cfgs() {
        assert_close(
            &format!("wgpu {name}"),
            &run_on(&cfg, Device::Gpu),
            &run_on(&cfg, Device::Cpu),
        );
    }
}

/// Compile `cfg` on the wgpu **device** route and return the refusal it panics
/// with, panicking itself if the route accepted the graph.
///
/// `RLX_WGPU_SHARD_GPU=1` forces the device route for this compile: it
/// short-circuits `wgpu_prefer_structure_host`, including its discrete-adapter
/// term, so the route is deterministic on any adapter kind (README: "Opt out of
/// discrete/sharded host packing").
fn device_route_refusal(name: &str, cfg: &Cfg) -> String {
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_on(cfg, Device::Gpu)));
    rlx_ir::env::unset("RLX_WGPU_SHARD_GPU");
    let Err(payload) = caught else {
        panic!("{name}: the device route accepted a degenerate grouping");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    eprintln!("{name}: device route refused — {msg}");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (degenerate Op::ConvTranspose2d grouping)"),
        "{name}: the refusal must name the geometry; got: {msg}"
    );
    msg
}

/// The geometry a vocoder actually presents — a 1-D-lifted kernel `[1, 16]` at
/// stride `[1, 8]` on unit height — plus the grouped / dilated / padded /
/// output-padded family the device routes have to reproduce, and the two
/// degenerate groupings they refuse in place of the pre-device-route behaviour.
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
            "wgpu degenerate-grouping-g0",
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
                // Accepted by the IR, whose shape rule checks the rank and the
                // weight's channel dims but not `groups`, and then computes
                // `c_out/g · groups == 0` output channels — so the graph builds
                // and the lowering is what refuses it.
                groups: 0,
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
        (
            // The other non-divisible grouping, where the host route is *not* a way
            // back: more input channels than groups, with a remainder.
            "wgpu degenerate-grouping-c6g4",
            Cfg {
                n: 1,
                c_in: 6,
                c_out: 3,
                h: 4,
                w: 4,
                k: [3, 3],
                s: [2, 2],
                p: [1, 1],
                d: [1, 1],
                op: [0, 0],
                groups: 4,
            },
        ),
    ];
    for (name, cfg) in cases {
        if cfg.groups == 0 {
            // The other refusal: a zero group size, which the device gather would
            // divide by. Tested before the `c_in % groups` condition the
            // non-divisible arm would use, which divides by this very zero.
            //
            // Unlike that arm, this message deliberately names no lever back to
            // the host: no host route serves this geometry either, because the
            // host step divides by the same zero group size. Asserting that
            // absence is the point here.
            let msg = device_route_refusal(name, &cfg);
            assert!(
                msg.contains("groups=0") && msg.contains("divide by a zero group size"),
                "{name}: the refusal must name this case's geometry; got: {msg}"
            );
            assert!(
                !msg.contains("RLX_WGPU_CONV_HOST=1") && !msg.contains("RLX_WGPU_FORCE_HOST=1"),
                "{name}: this refusal must offer no route back to the host — none serves a \
                 zero group size; got: {msg}"
            );
            continue;
        }
        if !cfg.c_in.is_multiple_of(cfg.groups) {
            // The device route refuses every non-divisible grouping, and the two
            // shapes of that refusal are different failures with different ways out,
            // so they are asserted apart.
            let msg = device_route_refusal(name, &cfg);
            if cfg.c_in < cfg.groups {
                // The *zeroing* shape: with fewer input channels than groups there is
                // no channel to accumulate from, so the gather leaves every output it
                // never accumulates at zero — which is exactly what the replaced host
                // route produced (the witness below pins that). The refusal names that
                // route, because for this geometry it is a real way back.
                assert!(
                    msg.contains("groups=8 exceeds the 4 input channels")
                        && msg.contains("leave its output zero"),
                    "{name}: the refusal must name this case's zeroing geometry; got: {msg}"
                );
                assert!(
                    msg.contains("RLX_WGPU_CONV_HOST=1") && msg.contains("RLX_WGPU_FORCE_HOST=1"),
                    "{name}: the refusal must name both routes back to the host; got: {msg}"
                );

                // The host route keeps the old witness — it is where this geometry's
                // previous behaviour lives. `groups > C_in` has no channel to
                // accumulate from, so both sides leave the output zeroed rather than
                // define a different operator out of a shape nothing computes.
                //
                // `RLX_WGPU_CONV_HOST` is process-global, but this is safe: every test
                // in this binary holds `common::serialize_gpu()` for its whole body,
                // the process-global GPU mutex (`common/mod.rs`'s `needs_serializing`
                // includes `Device::Gpu`), so no other test in the binary can be
                // compiling a wgpu graph in this window.
                rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
                let got = run_on(&cfg, Device::Gpu);
                rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
                let want = run_on(&cfg, Device::Cpu);
                assert_close(name, &got, &want);
                assert!(
                    want.iter().all(|v| *v == 0.0),
                    "{name}: the cpu reference must leave this degenerate grouping zeroed"
                );
                assert!(
                    got.iter().all(|v| *v == 0.0),
                    "{name}: the wgpu host route must leave this degenerate grouping zeroed"
                );
                continue;
            }
            // The *dropping* shape: more input channels than groups, with a remainder,
            // so the gather would accumulate only the channels inside the whole groups
            // it can form and silently drop the rest. No route computes it — the host
            // route's own scatter walks its channel indices past its output and traps —
            // so this refusal must name no way back to the host, and asserting that
            // absence is the point here, as for the zero group size above.
            assert!(
                msg.contains("groups=4 does not divide 6 input channels")
                    && msg.contains("silently drop the rest"),
                "{name}: the refusal must name this case's dropping geometry; got: {msg}"
            );
            assert!(
                !msg.contains("RLX_WGPU_CONV_HOST=1") && !msg.contains("RLX_WGPU_FORCE_HOST=1"),
                "{name}: this refusal must offer no route back to the host — the host route \
                 traps on this grouping too; got: {msg}"
            );
            // The claim that refusal makes about the *other* route, pinned rather than
            // left in prose: for this grouping the host route's scatter does not zero
            // the surplus channels, it walks its channel indices past its own output
            // and traps.
            rlx_ir::env::set("RLX_WGPU_CONV_HOST", "1");
            let host = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_on(&cfg, Device::Gpu)
            }));
            rlx_ir::env::unset("RLX_WGPU_CONV_HOST");
            let Err(payload) = host else {
                panic!("{name}: the host route computed a grouping the device route refuses");
            };
            let host_msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            eprintln!("{name}: host route traps — {host_msg}");
            assert!(
                host_msg.contains("index out of bounds"),
                "{name}: the host route must fail the way the refusal describes — on its own \
                 output index; got: {host_msg}"
            );
            continue;
        }
        let got = run_on(&cfg, Device::Gpu);
        let want = run_on(&cfg, Device::Cpu);
        assert_close(name, &got, &want);
    }
}
