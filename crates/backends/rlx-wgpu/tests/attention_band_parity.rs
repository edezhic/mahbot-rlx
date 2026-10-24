// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The two-sided band kernel (`kernels/attention_band.wgsl`) against the CPU
//! reference: `Op::Attention { mask_kind: MaskKind::Window(w) }` at shapes where
//! the band genuinely excludes keys, and at the shapes the fork's encoder hands
//! over.
//!
//! What this file pins: that a kind-5 step reaches the CPU backend's answer
//! within the f16 staging floor, at nine geometries — a small one whose bands are
//! a fraction of the sequence, a window narrower than one key block (so whole key
//! blocks fall outside every row's band and the masked sentinel, not the range
//! clip, is what removes them), head width 128 (a row of four chunk threads), a
//! GQA group of four *under* a band, the model's 8 192-row sliding layers (±512 of
//! 8 192 keys), its global layers (no band, width 512), and a dense
//! `MaskKind::None` step whose many-row scratch is past its cap (the route's
//! second entry, with `window` at the key extent) — and that the route the step
//! took is the band kernel's, not the fused entry's (which ignores kind 5 and
//! answers a dense pass, so the parity below would fail by a wide margin rather
//! than a rounding).
//!
//! What it does not claim: that the kernel is *bit*-identical to the CPU, or that
//! the deviation is the kernel's alone. The reference stages nothing in f16, so
//! the two arms differ by the tile's rounding as well as by summation order, and
//! the numbers printed per case are the measured deviation of that pair — not a
//! bound on either arm. The bars are wide enough to hold that quantization and no
//! wider; they are read against the module this process actually builds — the f16
//! tile's rounding, or the f32 twin's summation order where
//! `RLX_WGPU_ATTENTION_BAND_F32` (or a missing `SHADER_F16`) forces it — so a
//! regression that dropped an in-band key or mis-mapped a GQA head moves them by
//! orders of magnitude.
//!
//! Every case is rank-4 `[B, S, H, D]` — the stand's encoder layout, and the one
//! that must not be transposed before the kernel reads it. The bands are printed
//! per case with the number of query blocks whose band is clipped at the sequence
//! start and at its end, so a case whose band never clips cannot silently stand
//! in for one that does. A block is `attention_band_rows_for` at the adapter's
//! own thread count ([`band_rows`]), not the plan-side floor figure, because the
//! band arithmetic printed beside a case has to be the one the kernel ran.
//!
//! The route counter is process-global, so the one case that brackets it takes
//! [`route_lock`]. Timings are printed, never asserted.
mod common;

use common::{
    band_device, band_key_block, band_rows, band_stages_f16, data, max_abs_diff,
    refuse_under_run_report, route_lock,
};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::kernels::{AttentionRoutes, attention_route_counts, reset_attention_route_counts};

/// The bar the two arms are held to, per staging precision. The f16 tile is the
/// shipped form: its rounding is the dominant term, and it is a *relative* 5e-3 of
/// the output's own scale rather than an absolute figure — see the printed
/// deviations, which sit an order of magnitude inside it.
const BAND_BAR_F16: f32 = 5e-3;
/// The bar on an adapter that falls back to the f32 staging module: the same walk,
/// so only summation order differs from the CPU — the crate's dense-route bar.
const BAND_BAR_F32: f32 = 1e-5;

/// The bar the module actually in use earns, which is what every case asserts
/// against: the f16 tile's rounding, or the f32 twin's summation-order difference.
fn staging_bar() -> f32 {
    if band_stages_f16() {
        BAND_BAR_F16
    } else {
        BAND_BAR_F32
    }
}

/// Rank-4 `[B, S, H, D]` Q and `[B, S, KV, D]` K/V, one `Op::Attention` with the
/// two-sided band `w`. The layout the fork's encoder builds, which the kernel
/// reads through its stride triple rather than through a transpose.
fn band_graph(b: usize, rows: usize, nh: usize, nkv: usize, hd: usize, w: usize) -> Graph {
    attention_graph(b, rows, nh, nkv, hd, MaskKind::Window(w))
}

/// The same graph under any mask kind — `MaskKind::None` is the dense step the
/// plan routes through the band kernel with `window = seq_k` where the many-row
/// arrangement's scratch no longer fits.
fn attention_graph(
    b: usize,
    rows: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    kind: MaskKind,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attention_band");
    let q = g.input("q", Shape::new(&[b, rows, nh, hd], f));
    let k = g.input("k", Shape::new(&[b, rows, nkv, hd], f));
    let v = g.input("v", Shape::new(&[b, rows, nkv, hd], f));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// The key band every query block walks, as the kernel derives it:
/// `[max(0, q0 - w), min(seq, q0 + BM + w))`. Returns the per-block widths, how
/// many blocks are clipped at the start (their band would have begun below 0) and
/// how many at the end.
fn bands(rows: usize, w: usize, head_dim: usize) -> (Vec<usize>, usize, usize) {
    let bm = band_rows(head_dim as u32);
    let mut widths = Vec::new();
    let (mut clip_start, mut clip_end) = (0usize, 0usize);
    let mut q0 = 0usize;
    while q0 < rows {
        let lo = q0.saturating_sub(w);
        let hi = (q0 + bm + w).min(rows);
        widths.push(hi - lo);
        if q0 < w {
            clip_start += 1;
        }
        if q0 + bm + w > rows {
            clip_end += 1;
        }
        q0 += bm;
    }
    (widths, clip_start, clip_end)
}

/// Cosine similarity of two equal-length vectors — the second number this suite
/// reports beside max-abs, so a case whose output magnitude shrinks cannot hide a
/// shape error behind a small absolute deviation.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine needs equal lengths");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The two numbers every case is held to, asserted in one place: `run_case` calls
/// it, and so does the one case that brackets the route counter and therefore
/// takes its own reading. The bar follows the module in use, which is why the
/// label is carried: a failure names the case the reader is looking at.
fn assert_within_bar(label: &str, d: f32, cos: f64) {
    let bar = staging_bar();
    assert!(
        d < bar,
        "{label}: max-abs {d:.3e} is over the {bar:.1e} bar"
    );
    assert!(
        cos > 0.99999,
        "{label}: cosine {cos:.9} is short of the bar"
    );
}

/// One geometry on both devices: GPU output, CPU output, max-abs, cosine and the
/// GPU's wall time. `label` is what the print names the case by.
///
/// The bars are asserted here rather than in every case: both numbers are the
/// ones this function computes, and a case that took its own reading asserts them
/// through [`assert_within_bar`]. The route counter is process-global and every
/// case here runs a band step, so the runs are taken under [`route_lock`]: the
/// one case that brackets the counter would otherwise read another case's
/// dispatch as its own, which it did as soon as the cells got fast enough to
/// overlap.
fn run_case(
    label: &str,
    g: Graph,
    inputs: &[(&str, &[f32])],
) -> (Vec<f32>, Vec<f32>, f32, f64, f64) {
    let _gate = route_lock();
    let mut c = Session::new(Device::Gpu).compile(g.clone());
    // Warm: the pipeline is compiled and the band module is built on the first
    // call, which is not the kernel's cost.
    let _ = c.run(inputs).remove(0);
    let start = rlx_runtime::Tick::now();
    let gpu = c.run(inputs).remove(0);
    let ms = rlx_runtime::Tick::now().elapsed_us(start) / 1000.0;
    let cpu = Session::new(Device::Cpu).compile(g).run(inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    let cos = cosine(&gpu, &cpu);
    eprintln!("{label:<44} max_abs = {d:.3e}  cosine = {cos:.9}  gpu = {ms:.2} ms");
    assert_within_bar(label, d, cos);
    (gpu, cpu, d, cos, ms)
}

/// The geometries, one `#[test]` each so a failure names its own case and the
/// heavy cells can be run alone.
///
/// Every case's data is the crate's deterministic sweep, scaled so a score is
/// O(1) after `1/sqrt(d)` — a ramped input pair whose dot over 512 dims is
/// dominated by the low-frequency components, which is what makes the softmax
/// peaked enough that a dropped in-band key shows.
#[test]
fn the_small_band_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // rows 300 is a couple of query blocks (BM is 128 at this width's floor, 256
    // at the wide form): the first blocks' bands reach back past position 0 and
    // the last ones' past `rows`, and w = 32 is a tenth of the sequence, so most
    // key blocks are skipped outright.
    let (b, rows, nh, nkv, hd, w) = (1usize, 300usize, 2usize, 2usize, 64usize, 32usize);
    let (widths, clip_start, clip_end) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={}, clipped at the start \
         {clip_start} and at the end {clip_end}, of {rows} keys",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    assert!(
        clip_start > 0 && clip_end > 0,
        "this case must exercise both band clips"
    );
    assert!(
        widths.iter().max().copied().unwrap_or(0) < rows,
        "and its bands must actually exclude keys"
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.01, false);
    let k = data(b * rows * nkv * hd, 0.013, true);
    let v = data(b * rows * nkv * hd, 0.007, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=300 w=32 nh=2 nkv=2 d=64", g, &inputs);
}

#[test]
fn a_narrow_band_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // rows 512 / w = 8: 512 keys becomes `BM + 2w` a query block — 48 rows at this
    // width's floor (BM = 32), 80 at the wide form (BM = 64) — a fraction of a
    // dense pass, and the block whose band is *wholly* inside its neighbour's is
    // the case where an off-by-one in the key range would drop nothing but a real
    // key would.
    let (b, rows, nh, nkv, hd, w) = (1usize, 512usize, 2usize, 2usize, 256usize, 8usize);
    let (widths, clip_start, clip_end) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={}, clipped at the start \
         {clip_start} and at the end {clip_end}, of {rows} keys",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.011, false);
    let k = data(b * rows * nkv * hd, 0.017, true);
    let v = data(b * rows * nkv * hd, 0.005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=512 w=8 nh=2 nkv=2 d=256", g, &inputs);
}

/// The model's sliding layers: 8 192 rows, 4 heads over 2 KV heads, head width
/// 256, ±512 band. This is the cell the route exists for — a query block visits
/// 1 088 of 8 192 keys — and the only one here with a GQA group above one, so a
/// head→KV-head mapping that read the group as 4 would show as a large deviation
/// on two of the four heads.
#[test]
fn the_sliding_layer_geometry_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (b, rows, nh, nkv, hd, w) = (1usize, 8192usize, 4usize, 2usize, 256usize, 512usize);
    let (widths, clip_start, clip_end) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={}, clipped at the start \
         {clip_start} and at the end {clip_end}, of {rows} keys",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.0007, false);
    let k = data(b * rows * nkv * hd, 0.0009, true);
    let v = data(b * rows * nkv * hd, 0.0005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=8192 w=512 nh=4 nkv=2 d=256", g, &inputs);
}

/// The model's global layers: no band (`w` ≥ rows, which must read exactly as
/// `MaskKind::None`), 4 heads over one KV head, head width 512 — the widest head
/// the kernel holds and the whole sequence in every block's range.
#[test]
fn the_global_layer_geometry_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (b, rows, nh, nkv, hd, w) = (1usize, 8192usize, 4usize, 1usize, 512usize, 8192usize);
    let (widths, _, _) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={} — no band, the whole \
         sequence in every block",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    assert!(
        widths.iter().all(|&n| n == rows),
        "w >= rows must leave every block's range the whole sequence"
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.0005, false);
    let k = data(b * rows * nkv * hd, 0.0007, true);
    let v = data(b * rows * nkv * hd, 0.0004, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=8192 w>=rows nh=4 nkv=1 d=512", g, &inputs);
}

/// A GQA group of four (`heads / kv_heads = 4`) *with* a band: the group the
/// global layers have, at a geometry where the band actually excludes keys, so a
/// head→KV-head mapping that read the group as 1 or as 4 shows on three heads of
/// four rather than only at the widest cell.
#[test]
fn a_four_wide_kv_group_under_a_band_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (b, rows, nh, nkv, hd, w) = (1usize, 1024usize, 4usize, 1usize, 256usize, 128usize);
    let (widths, clip_start, clip_end) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={}, clipped at the start \
         {clip_start} and at the end {clip_end}, of {rows} keys, 4 heads over 1 KV head",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    assert!(
        widths.iter().max().copied().unwrap_or(0) < rows,
        "the band must exclude keys, or this case cannot see a mis-mapped group"
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.011, false);
    let k = data(b * rows * nkv * hd, 0.013, true);
    let v = data(b * rows * nkv * hd, 0.005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=1024 w=128 nh=4 nkv=1 d=256", g, &inputs);
}

/// A window narrower than one key block, and narrower than the query block: every
/// row of every block therefore has whole key blocks wholly outside its band, and
/// the masked sentinel (not just the range clip) is what keeps them out of the
/// softmax. `w = 2` against a `BK` of 8 at this width on this rig's storage (4 on
/// a 16 KiB adapter staging f32) is the point of the case, so the test reads both
/// figures off the module in force and asserts it rather than assuming it.
#[test]
fn a_band_narrower_than_a_key_block_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (b, rows, nh, nkv, hd, w) = (1usize, 512usize, 2usize, 1usize, 256usize, 2usize);
    let bm = band_rows(hd as u32);
    let bk = band_key_block(hd as u32);
    assert!(
        w < bk && w * 2 < bm,
        "the case needs a window narrower than a key block ({bk}) and than a query \
         block's own band ({bm} rows): w={w}"
    );
    let (widths, _, _) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks of {bm} rows, band widths {}..={}, of which \
         {bk} keys a block — whole key blocks fall outside every row's band",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.011, false);
    let k = data(b * rows * nkv * hd, 0.017, true);
    let v = data(b * rows * nkv * hd, 0.005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=512 w=2 nh=2 nkv=1 d=256", g, &inputs);
}

/// Head width 128, where a row is four chunk threads: the boundary of the
/// reduce's four-accumulator form (`C >= 4`) and a `BM` of 64 rows at the floor
/// (128 at the wide form), the widest query block the plan builds. Cheap, and the
/// only case at that chunk count.
#[test]
fn the_four_chunk_width_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (b, rows, nh, nkv, hd, w) = (1usize, 256usize, 2usize, 2usize, 128usize, 64usize);
    let (widths, clip_start, clip_end) = bands(rows, w, hd);
    eprintln!(
        "rows={rows} w={w}: {} query blocks, band widths {}..={}, clipped at the start \
         {clip_start} and at the end {clip_end}, of {rows} keys",
        widths.len(),
        widths.iter().min().copied().unwrap_or(0),
        widths.iter().max().copied().unwrap_or(0),
    );
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.011, false);
    let k = data(b * rows * nkv * hd, 0.017, true);
    let v = data(b * rows * nkv * hd, 0.005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    run_case("rows=256 w=64 nh=2 nkv=2 d=128", g, &inputs);
}

/// A dense (`MaskKind::None`) step whose many-row scratch is past its cap is
/// served by the same kernel with the band switched off, where the cooperative
/// dense square form does not take it — this shape is below that form's 2 048-row
/// floor, which is what keeps it here. Sized so the CPU reference is affordable:
/// 64 heads put the 512 MiB cap at 1 449 rows, so 1 536 rows clears it, and at
/// head width 64 (two chunk threads, `BM = 128` at the floor, 256 at the wide
/// form) the band walk is cheap. The same run is also the route evidence:
/// `band` 1, `decomposed` 0.
///
/// The bar is the staged tile's, as for every other case — the f16 tile's
/// rounding, or the f32 twin's summation order where that module is in force: the
/// reference stages nothing, so the residue is the tile's rounding as much as the
/// summation order — and this case must land on the band kernel, not on the dense
/// arrangement it replaces, or the printed deviation would drop to ~1e-6 and the
/// route counter would say so.
#[test]
fn a_dense_square_step_past_the_scratch_cap_matches_the_cpu_reference() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let (b, rows, nh, nkv, hd) = (1usize, 1536usize, 64usize, 64usize, 64usize);
    let g = attention_graph(b, rows, nh, nkv, hd, MaskKind::None);
    let q = data(b * rows * nh * hd, 0.0007, false);
    let k = data(b * rows * nkv * hd, 0.0009, true);
    let v = data(b * rows * nkv * hd, 0.0005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let _gate = route_lock();
    let mut c = Session::new(Device::Gpu).compile(g.clone());
    let _ = c.run(&inputs).remove(0);
    reset_attention_route_counts();
    let gpu = c.run(&inputs).remove(0);
    let routes = attention_route_counts();
    let cpu = Session::new(Device::Cpu).compile(g).run(&inputs).remove(0);
    let d = max_abs_diff(&gpu, &cpu);
    let cos = cosine(&gpu, &cpu);
    eprintln!(
        "dense square rows={rows} nh={nh} d={hd} routes={routes:?} max_abs = {d:.3e}  \
         cosine = {cos:.9}"
    );
    assert_eq!(
        routes,
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: 0,
            band: 1,
        },
        "a dense square step past the many-row scratch cap must take the band kernel, \
         with the band switched off"
    );
    assert_within_bar("dense square past the scratch cap", d, cos);
}

/// The route the step took, named where the counter is read: a kind-5 step is the
/// band kernel's and never `serial`'s or the dense route's — the two arrangements
/// that ignore kind 5 and would have answered a dense pass.
#[test]
fn a_windowed_step_is_counted_as_the_band_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    refuse_under_run_report();
    let _gate = route_lock();
    let (b, rows, nh, nkv, hd, w) = (1usize, 128usize, 2usize, 2usize, 64usize, 16usize);
    let g = band_graph(b, rows, nh, nkv, hd, w);
    let q = data(b * rows * nh * hd, 0.01, false);
    let k = data(b * rows * nkv * hd, 0.013, true);
    let v = data(b * rows * nkv * hd, 0.007, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let mut c = Session::new(Device::Gpu).compile(g);
    let _ = c.run(&inputs).remove(0);
    reset_attention_route_counts();
    let _ = c.run(&inputs).remove(0);
    assert_eq!(
        attention_route_counts(),
        AttentionRoutes {
            decode: 0,
            decomposed: 0,
            serial: 0,
            band: 1,
        },
        "a MaskKind::Window step must take the band kernel: the other two entries \
         read kind 5 as no mask"
    );
}

/// The workgroup the module is built with is one this adapter can launch, and its
/// rows are the figure this file and the run loop dispatch with.
///
/// Two claims, both one-directional and device-free in form:
///
/// * `attention_band_threads(&device) <= device.limits().max_compute_invocations_per_workgroup`
///   — the 512-thread form is an add-on, so an adapter that reports less than
///   that must get the 256 floor and no module it cannot launch. This is what the
///   route used to assume rather than check: the plan is device-free, so a
///   512-wide module on a 256-wide adapter was a panic at pipeline creation.
/// * `rows · chunks == threads` for the form at that width — the whole-row split
///   the module's `@workgroup_size(THREADS)` / `BM` / `C` constants are declared
///   from, and the reason a grid built from [`band_rows`] is the kernel's own
///   blocks rather than an estimate of them.
///
/// What it does not establish: which width the adapter *takes* (this rig reports
/// 1024 and so takes 512; a lavapipe or a reduced-limit host would take 256), or
/// that the wide form is faster — that is the cost file's table.
#[test]
fn the_band_workgroup_is_the_adapters_and_splits_into_whole_rows() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let dev = band_device();
    let threads = rlx_wgpu::kernels::attention_band_threads(&dev.device);
    let limit = dev.device.limits().max_compute_invocations_per_workgroup;
    assert!(
        threads <= limit,
        "the band module's {threads}-thread workgroup is above this adapter's \
         maxComputeInvocationsPerWorkgroup ({limit})"
    );
    for hd in [64u32, 128, 256, 512] {
        let form = rlx_wgpu::kernels::attention_band_form_at(hd, threads)
            .unwrap_or_else(|| panic!("hd={hd} has no form at {threads} threads"));
        assert_eq!(
            form.rows * form.chunks,
            threads,
            "hd={hd}: one row must be whole chunks of the {threads}-thread workgroup"
        );
        assert_eq!(
            band_rows(hd),
            form.rows as usize,
            "hd={hd}: the rows this file's band arithmetic uses must be the module's BM"
        );
    }
}
