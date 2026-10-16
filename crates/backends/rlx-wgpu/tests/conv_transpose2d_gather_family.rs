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
//! A second list, `TILED_CASES`, covers the tiled device step's family
//! (`conv_transpose2d_tiled`, `conv_transpose2d_tiled_fits`) instead: same oracle,
//! same derived bound, but the census — the one thing the two readings assert
//! differently — is what makes the reading attributable to that step. Its first two
//! rows are the decoder's `up.blk` forms, read off `common::DECODER_FORMS` so this file
//! and the two other holders of those shapes cannot disagree about one; the other five
//! are the family's edge arms — a form at the family's `k = 2·s` ceiling whose
//! channel counts do not fill a tile (so the panels' zero-fill and the store guards are
//! witnessed against the independent oracle), a form whose row is shorter than one
//! output phase, which is where the shader's empty-phase return is taken, a form
//! whose width pad is past the stride (`p = 3 > s = 2`), the one clause the predicate
//! admits that no other row reaches, a form at stride 1, and a form in the interior
//! `s < k < 2·s`, where a phase's tap count differs inside one dispatch. Between them
//! the rows witness both ends of the family's `s ≤ k ≤ 2·s` (`k = s` and `k = 2·s`) and
//! its interior, padding at zero, under the stride and over it, output padding present
//! and absent, a row too short to hold a phase, and panel-filling against panel-sparse
//! channel counts; every row is single batch, unit height, one group and no dilation,
//! which is the rest of the predicate. The gather's own cases keep the assertion they
//! had.
//!
//! The decoder's four `blk.*` forms are the same family at `k = 2·s`, but the oracle and
//! the bound walk are both `O(c_in · c_out · k · w_in)` there and the four of them are
//! ~8.4 G terms, so their derived-bound reading is the `#[ignore]`d test at the bottom of
//! this file rather than a row of the list the default device suite runs. Their `1e-4`
//! reading against the CPU device is
//! `crates/core/rlx-runtime/tests/conv_transpose2d_parity.rs`'s.
//!
//! One geometry here is deliberately not a parity case: `non_finite_weight_on_a_dropped_tap`
//! pins the single place the tiled step and the gather are not the same operator, so the
//! shader header's note about it is a checked claim rather than a story.
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
/// padding, groups 2 and 3, an asymmetric kernel/stride/pad/dilation case, one
/// output-padded case, and the boundary the tiled step's family stops at — unit
/// height like `TILED_CASES`, but at `k > 2·s`, which is the gather's side of it.
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
    // Unit height like the family above, but `k = 5 > 2·s = 4`: three taps a
    // phase is outside the tile's two, so this is the gather's side of the
    // boundary.
    Cfg::asym("k5s2-3taps", 1, 4, 4, 1, 8, [1, 5], [1, 2], [0, 2], [1, 1]).output_padding([0, 1]),
];

/// The tiled device step's family, in the order the run prints them: the decoder's
/// two `up.blk` forms first — read off `common::DECODER_FORMS`, where the six real
/// shapes and the generator that derives their padding live, so this file and the
/// route harness cannot disagree about one — then the edge arms the module header
/// names.
///
/// The decoder's four `blk.*` forms are the same family at `k = 2·s`; their
/// derived-bound reading is the `#[ignore]`d test below, for the cost reason the module
/// header gives.
///
/// Laid out by hand (`#[rustfmt::skip]`): rustfmt breaks these rows into one field or
/// argument per line, which buries the table the reading is read against.
#[rustfmt::skip]
const TILED_CASES: &[Cfg] = &[
    Cfg { name: common::DECODER_FORMS[0].0, geo: common::DECODER_FORMS[0].1 },
    Cfg { name: common::DECODER_FORMS[1].0, geo: common::DECODER_FORMS[1].1 },
    // The family's tap ceiling, `k = 2·s`, at channel counts that fill neither a
    // `CHUNK`-wide A panel nor a `TILE_CO`-wide B panel.
    Cfg::asym("k4s2-in-family", 1, 4, 4, 1, 8, [1, 4], [1, 2], [0, 1], [1, 1]),
    // `w_out = 1` at `s = 2` and `pad = 1`: the phase whose first position is past
    // the row's own end, i.e. the shader's `n_r = 0` return.
    Cfg::asym("phase-past-the-row", 1, 4, 4, 1, 1, [1, 2], [1, 2], [0, 1], [1, 1])
        .output_padding([0, 1]),
    // The width pad past the stride, `p = 3 > s = 2`, at the ceiling `k = 2·s`: the one
    // admitted clause the rows above leave unwitnessed. One group, no dilation, `h = 1`,
    // so the predicate and the derived bound are the same ones the rows above use and
    // what is new is the tap walk's pad arithmetic (`w_out = 12`).
    Cfg::asym("pw-ge-sw", 1, 4, 4, 1, 8, [1, 4], [1, 2], [0, 3], [1, 1]),
    // `s = 1`: the stride the predicate admits and the rows above leave unwitnessed.
    // One phase carries both taps, and the output row is shorter than the input
    // (`w_out = 7 < w_in = 8`), which no other row does.
    Cfg::asym("k2s1-in-family", 1, 4, 4, 1, 8, [1, 2], [1, 1], [0, 1], [1, 1]),
    // The interior `s < k < 2·s` (`3 < 5 < 6`), where a phase's tap count differs inside
    // one dispatch: three phases carrying 2, 2 and 1 taps.
    Cfg::asym("k5s3-mixed-taps", 1, 4, 4, 1, 8, [1, 5], [1, 3], [0, 1], [1, 1]),
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

/// Every case in `cases` through the one device step `step` names — its census
/// **asserted**, because a geometry that quietly kept the other step would be read
/// as a witness for this one — against the scalar oracle, at the bar derived from
/// the case's own operands and the terms that land in its worst output element.
///
/// The route env and the adapter skip are the harness's, not the case list's, so
/// both lists take them alike; the two lists then differ in nothing but their
/// geometries and the step they must reach.
fn compare_device_step_against_the_scalar_oracle(cases: &[Cfg], step: &str) {
    common::neutralise_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    for cfg in cases {
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
            kinds.get(step).copied().unwrap_or(0),
            1,
            "{}: the census must name {step} — the step this reading is about — rather than \
             another route; schedule = {kinds:?}",
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
            "{}: {step} vs scalar oracle: measured {measured:e} > derived bound {bound:e}",
            cfg.name
        );
    }
}

#[test]
fn the_device_gather_matches_the_scalar_oracle_across_its_padding_dilation_and_group_arms() {
    compare_device_step_against_the_scalar_oracle(CASES, "conv_transpose2d");
}

/// The same reading on the other device step: `TILED_CASES` belong to
/// `conv_transpose2d_tiled`'s family, so their census has to name that step. Same
/// oracle and same derived bound as the arm above, so these are the same measurement
/// of two steps rather than two bars.
#[test]
fn the_tiled_device_step_matches_the_scalar_oracle_across_its_family() {
    compare_device_step_against_the_scalar_oracle(TILED_CASES, "conv_transpose2d_tiled");
}

/// The decoder's four heavy forms, against the same oracle at the same derived
/// bound — the coverage the two light forms above cannot give, since a shared
/// wrong accumulation order would not be visible cross-side.
///
/// `#[ignore]`d for cost, not for doubt: the oracle and the bound walk are both
/// `O(all terms)` and these four carry ~8.4 G of them (the naive scatter and the
/// walk together run for a minute or more), which the default device suite must
/// not pay. Run it with `--ignored --nocapture` and read the printed
/// measured/bound pairs against the other file's.
///
/// Laid out by hand (`#[rustfmt::skip]`): rustfmt breaks these rows one field a line.
#[rustfmt::skip]
const HEAVY_CASES: &[Cfg] = &[
    Cfg { name: common::DECODER_FORMS[2].0, geo: common::DECODER_FORMS[2].1 },
    Cfg { name: common::DECODER_FORMS[3].0, geo: common::DECODER_FORMS[3].1 },
    Cfg { name: common::DECODER_FORMS[4].0, geo: common::DECODER_FORMS[4].1 },
    Cfg { name: common::DECODER_FORMS[5].0, geo: common::DECODER_FORMS[5].1 },
];

#[test]
#[ignore = "cost: the naive oracle plus the derived-bound walk over ~8.4 G terms"]
fn the_heavy_decoder_forms_match_the_scalar_oracle_too() {
    compare_device_step_against_the_scalar_oracle(HEAVY_CASES, "conv_transpose2d_tiled");
}

/// The one place the tiled step is not the same operator as the gather: a
/// **non-finite** weight on a tap the gather's own bounds test drops.
///
/// Kernel `[1, 2]`, stride `[1, 1]`, pad `[0, 0]`, `w_in = 4` → `w_out = 5`. The
/// only phase has both taps and `off = 0`, so at output position 0 the tap
/// `kx = 1` reads input position `iw = -1` — outside the row. The gather tests
/// `iw >= 0` before it reads the weight and so drops the term; the tiled kernel
/// reads the weight and multiplies it by the A panel's zero fill, which for an
/// infinite weight is `0 · inf = NaN`. This test pins that, so the shader
/// header's note is a checked claim rather than a story: every element other
/// than position 0 — where the NaN sits and the gather stays finite — agrees
/// with the naive scalar oracle bit for bit, infinities included.
#[test]
fn non_finite_weight_on_a_dropped_tap() {
    common::neutralise_device_route_env();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // Deliberately not a `TILED_CASES` row: a derived bound is meaningless on a weight
    // one of the two sides drops and the other reads.
    const GEO: Cfg = Cfg::asym("inf-tap", 1, 1, 1, 1, 4, [1, 2], [1, 1], [0, 0], [1, 1]);
    let geo = &GEO.geo;

    let mut exe = WgpuExecutable::compile(graph(geo));
    let kinds = exe.step_kind_counts();
    assert_eq!(
        kinds.get("conv_transpose2d_tiled").copied().unwrap_or(0),
        1,
        "{}: the divergence is the tiled step's, so a census naming anything else \
         witnesses nothing; schedule = {kinds:?}",
        GEO.name
    );

    let (x, mut w) = inputs(geo);
    // The weight the dropped tap reads, made non-finite; `w[0]` stays finite and
    // non-zero, so the other four positions carry a real product rather than a NaN.
    w[1] = f32::INFINITY;
    exe.set_param("w", &w);
    let got = exe.run(&[("x", &x)]).remove(0);

    let (h_out, w_out) = geo.out();
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

    assert_eq!(got.len(), want.len(), "parity needs equal lengths");
    eprintln!(
        "non-finite dropped tap: oracle[0] = {}, device[0] = {}",
        want[0], got[0]
    );
    assert!(
        want[0].is_finite(),
        "the gather tests `iw >= 0` before reading the weight and must drop that tap, \
         so oracle[0] = {} has to stay finite",
        want[0]
    );
    assert!(
        got[0].is_nan(),
        "the tiled step reads that tap's weight into the A panel's zero fill, so \
         device[0] = {} has to be the `0 · inf` NaN the divergence is about",
        got[0]
    );
    for i in 1..want.len() {
        eprintln!(
            "non-finite dropped tap: element {i}: device {} vs oracle {}",
            got[i], want[i]
        );
        assert_eq!(
            got[i].to_bits(),
            want[i].to_bits(),
            "every element but position 0 must agree with the oracle bit for bit, \
             infinities included: element {i} reads device {} against oracle {}",
            got[i],
            want[i]
        );
    }
}
