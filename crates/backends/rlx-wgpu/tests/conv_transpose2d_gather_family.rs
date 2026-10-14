// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d`'s device gather over the 2-D geometries its kernel's
//! padding, dilation and group arms exist for, each against the scalar oracle.
//!
//! The route's other pins cover it only at padding 0 / dilation 1 / groups 1 —
//! `conv_transpose2d_routes.rs` on the decoder geometry, the runtime parity suite
//! at stride 2 — and that suite compares against the CPU *device*, which shares the
//! accelerated GEMM formulation (`conv_transpose2d_forward_gemm`, gated by
//! `RLX_FAST_CONV`, on by default) and so is not an independent reference. The
//! padded, dilated and grouped arms therefore had no witness that could tell a
//! mis-walked tap from a correctly walked one. Every case here is compared against
//! `rlx_cpu::kernels::conv_transpose2d_nchw`, the naive scalar scatter whose tap
//! walk is written out one input voxel at a time and shares nothing with either GPU
//! formulation.
//!
//! The tolerance is derived per case from the operands and the terms that land in
//! the worst output element (`common::conv_transpose2d_pair_error_bound`), never from
//! an absolute bar, and every case asserts its census (`step_kind_counts()`) so a
//! measurement that silently took the host route cannot pass.
//!
//! Its own test binary: the route levers below are process-global and read per compile.

mod common;

use common::ConvTranspose2dGeometry;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// One case: the geometry the shared bound walk and the graph builder both read,
/// under the label the run prints it by. The geometry is field for field
/// `conv_transpose2d_parity.rs`'s `Cfg`, so the two files witness the same shapes.
struct Cfg {
    name: &'static str,
    geo: ConvTranspose2dGeometry,
}

impl Cfg {
    /// Both axes alike: square kernel, one stride and one pad, no dilation, no output
    /// padding, one group — the shape the OCR U-Net upsamples take. The modifiers
    /// below add what this cannot express.
    const fn sym(
        name: &'static str,
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
            name,
            geo: ConvTranspose2dGeometry {
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
            },
        }
    }

    /// Kernel, stride, pad and dilation per axis — the arms a symmetric case cannot
    /// reach. Still one group and no output padding, which the modifiers add.
    const fn asym(
        name: &'static str,
        n: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: [usize; 2],
        s: [usize; 2],
        p: [usize; 2],
        d: [usize; 2],
    ) -> Self {
        Self {
            name,
            geo: ConvTranspose2dGeometry {
                n,
                c_in,
                c_out,
                h,
                w,
                k,
                s,
                p,
                d,
                op: [0, 0],
                groups: 1,
            },
        }
    }

    /// The same geometry at another dilation.
    const fn dilation(mut self, d: [usize; 2]) -> Self {
        self.geo.d = d;
        self
    }

    /// The same geometry at another group count.
    const fn groups(mut self, groups: usize) -> Self {
        self.geo.groups = groups;
        self
    }

    /// The same geometry at another output padding.
    const fn output_padding(mut self, op: [usize; 2]) -> Self {
        self.geo.op = op;
        self
    }

    /// The case as printed, so a reading can be read against the shape it came from
    /// without the reader counting fields in the label.
    fn geometry(&self) -> String {
        format!(
            "n={} c_in={} c_out={}×g{} {}×{} k={}×{} s={}×{} p={}×{} d={}×{} op={}×{}",
            self.geo.n,
            self.geo.c_in,
            self.geo.c_out,
            self.geo.groups,
            self.geo.h,
            self.geo.w,
            self.geo.k[0],
            self.geo.k[1],
            self.geo.s[0],
            self.geo.s[1],
            self.geo.p[0],
            self.geo.p[1],
            self.geo.d[0],
            self.geo.d[1],
            self.geo.op[0],
            self.geo.op[1],
        )
    }
}

/// The cases, in the order the run prints them.
///
/// The first six are `conv_transpose2d_parity.rs`'s `cfgs()` verbatim — the
/// stride/pad family the two files must witness alike. The rest add the arms that
/// list does not reach: padding 2, stride 3, dilation 2 alone and combined with
/// padding, groups 2 and 3, an asymmetric kernel/stride/pad/dilation case, and one
/// output-padded case.
const CASES: &[Cfg] = &[
    Cfg::sym("up2-k2", 1, 8, 4, 6, 6, 2, 2, 0),
    Cfg::sym("up2-k4-p1", 1, 6, 5, 5, 7, 4, 2, 1),
    Cfg::sym("s1-k3-p1", 2, 4, 3, 5, 5, 3, 1, 1),
    Cfg::sym("up2-k3s2-p1", 1, 8, 4, 6, 6, 3, 2, 1),
    Cfg::sym("up2-k3s2-p1-odd", 1, 5, 3, 7, 5, 3, 2, 1),
    Cfg::sym("large-up2-k3s2", 1, 64, 32, 32, 32, 3, 2, 1),
    // Padding 2, and stride 3 — the other two values of both families than the list
    // above reaches.
    Cfg::sym("p2-k3s2", 1, 6, 4, 6, 6, 3, 2, 2),
    Cfg::sym("s3-k3-p1", 1, 4, 3, 5, 5, 3, 3, 1),
    // Dilation 2 on its own, then folded under a stride and a wider pad.
    Cfg::sym("d2-k3s1-p1", 1, 5, 3, 5, 7, 3, 1, 1).dilation([2, 2]),
    Cfg::sym("d2-k3s2-p2", 1, 5, 3, 5, 7, 3, 2, 2).dilation([2, 2]),
    // Groups 2, then 3 under a kernel, stride and dilation that differ per axis.
    Cfg::sym("g2-k3s2-p1", 1, 8, 6, 6, 6, 3, 2, 1).groups(2),
    Cfg::asym("g3-asym-d2", 1, 12, 6, 5, 5, [3, 2], [2, 1], [1, 0], [2, 1]).groups(3),
    // Padding, dilation and groups at once, at pads that differ per axis.
    Cfg::asym("p2-p1-d2-g2", 1, 8, 4, 6, 6, [3, 3], [2, 2], [2, 1], [2, 1]).groups(2),
    // Output padding: the rows and columns no tap reaches must still come back
    // zeroed rather than left as whatever the slab held.
    Cfg::sym("op1-k3s2-p1", 1, 6, 4, 6, 6, 3, 2, 1).output_padding([1, 1]),
];

fn graph(geo: &ConvTranspose2dGeometry) -> Graph {
    let mut g = Graph::new("conv_transpose2d_gather_family");
    let x = g.input(
        "x",
        Shape::new(&[geo.n, geo.c_in, geo.h, geo.w], DType::F32),
    );
    let w = g.param(
        "w",
        Shape::new(&[geo.c_in, geo.c_out, geo.k[0], geo.k[1]], DType::F32),
    );
    let y = g.conv_transpose2d(x, w, geo.k, geo.s, geo.p, geo.d, geo.op, geo.groups);
    g.set_outputs(vec![y]);
    g
}

/// The case's operands, filled deterministically the way the neighbouring pins fill
/// theirs. Nothing in the comparison reads the fill itself, but the values set the
/// magnitudes the derived bound is built from.
fn inputs(geo: &ConvTranspose2dGeometry) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..geo.n * geo.c_in * geo.h * geo.w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..geo.c_in * geo.c_out * geo.k[0] * geo.k[1])
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.03)
        .collect();
    (x, w)
}

#[test]
fn the_device_gather_matches_the_scalar_oracle_across_its_padding_dilation_and_group_arms() {
    common::conv_transpose2d_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    for cfg in CASES {
        let geo = &cfg.geo;
        let (h_out, w_out) = geo.out();
        let (x, w) = inputs(geo);
        let mut want = vec![0f32; geo.n * geo.c_out_total() * h_out * w_out];
        rlx_cpu::kernels::conv_transpose2d_nchw(
            &x,
            &w,
            &mut want,
            geo.n,
            geo.c_in,
            geo.h,
            geo.w,
            geo.c_out_total(),
            h_out,
            w_out,
            geo.k[0],
            geo.k[1],
            geo.s[0],
            geo.s[1],
            geo.p[0],
            geo.p[1],
            geo.d[0],
            geo.d[1],
            geo.groups,
        );

        let mut exe = WgpuExecutable::compile(graph(geo));
        let kinds = exe.step_kind_counts();
        assert_eq!(
            kinds.get("conv_transpose2d").copied().unwrap_or(0),
            1,
            "{}: the census must name the device gather rather than the host route this \
             measurement would not be about; schedule = {kinds:?}",
            cfg.name
        );
        exe.set_param("w", &w);
        let got = exe.run(&[("x", &x)]).remove(0);

        let measured = common::max_abs_diff(&got, &want);
        let (bound, worst_terms, worst_abs) =
            common::conv_transpose2d_pair_error_bound(geo, &x, &w);
        eprintln!(
            "{}: {} — max abs diff {measured:.3e}, bound {bound:.3e} (worst terms {worst_terms}, \
             worst Σ|terms| {worst_abs:.3}), ratio {:.3} of bound ({} elems)",
            cfg.name,
            cfg.geometry(),
            measured as f64 / bound,
            want.len()
        );
        assert!(
            measured as f64 <= bound,
            "{}: device gather vs scalar oracle: measured {measured:e} > derived bound {bound:e}",
            cfg.name
        );
    }
}
