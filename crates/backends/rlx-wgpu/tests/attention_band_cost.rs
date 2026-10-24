// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The two-sided band kernel's cost at the fork's encoder geometries.
//! Measurement only: every figure here is printed and none is asserted — a
//! timing on this machine moves with load and with the machine's thermal state
//! (`just throttle` before a reading), so a pinned number would be a pin on the
//! runner rather than on the kernel.
//!
//! A reading is one `Session::run` of a graph that is *one* `Op::Attention`, so
//! it includes the submission and the readback floor of a run —
//! [`the_runs_own_floor_at_the_biggest_cell`] takes that floor apart, at the
//! biggest cell's size, so the kernel's own share can be read off. It is not the
//! kernel's device time; `RLX_WGPU_DBG_COUNTERS=1` gives the device's own lower
//! bound for one run when the split matters.
//!
//! GFLOP/s counts the band's own work: `2 · 2 · heads · BM · Σ_blocks width ·
//! head_dim` for the score and the value multiply — the second factor of two
//! being the two multiplies, and `BM` the rows each block's walked keys serve.
//! The achieved band is printed beside the rate, so the skip the kernel exists
//! for is visible: at 8 192 rows, ±512 and this rig's `BM = 64` (the 512-thread
//! form; 32 at the 256-thread floor), a query block visits 1 088 of 8 192 keys
//! (1 052 on average over the clipped ends).
//!
//! The GB and GB/s beside each rate are the K and V rows the walk re-reads from
//! the arena — `heads · Σ_blocks width · head_dim · 8` bytes, nothing cached
//! between query blocks. They are what says whether a cell is waiting on its
//! bytes or on its instructions, and they are the portable part of this file:
//! the byte count follows from the band and the block size alone, while the GB/s
//! a given adapter sustains for it does not. This rig reads ~64 GB/s at the
//! global cell and ~23 at the sliding one, and the traffic is *not* what binds:
//! the form table in the kernel header shows the same walk at half the traffic
//! (512 threads, `BM` 16 → 32 at d = 512 — the floor's rows against this rig's
//! wide form) gaining 5%, the same traffic with a doubled per-thread chunk losing
//! 41%, and a quarter of the traffic (1 024 threads) losing 2.4×. A
//! cooperative-matrix form would change the instruction side only; the kernel
//! header carries that arithmetic and where each figure in it came from.
//!
//! Both staging tiles are measured: the f16 one (the shipped form) and the f32
//! one behind `RLX_WGPU_ATTENTION_BAND_F32=1`, which is also what an adapter
//! without `SHADER_F16` takes. They are one WGSL body with one substituted token
//! (`kernels::bind_band_stage`), so the two tables differ in the tile alone. The
//! tables take a lock around themselves because that switch is process-global:
//! run in parallel, the f16 table's cells were measured on the module the f32
//! test had just forced.
//!
//! Run this file with `--ignored --nocapture`.
mod common;

use common::{band_rows, data, median, time_rounds};
use std::sync::Mutex;

/// Serialises the two staging tables: `RLX_WGPU_ATTENTION_BAND_F32` is read per
/// dispatch, so a table measured beside the other one measured the wrong module.
static TABLE: Mutex<()> = Mutex::new(());
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// One geometry of the table: the fork's encoder's own shapes, plus the small
/// cell that admits every key through a window wider than the sequence.
struct Cell {
    label: &'static str,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
}

const CELLS: [Cell; 3] = [
    // The sliding layers: ±512 over 8 192 keys, four heads over two KV heads.
    Cell {
        label: "sliding 8192 w=512 h=4 kv=2 d=256",
        rows: 8192,
        heads: 4,
        kv_heads: 2,
        head_dim: 256,
        window: 512,
    },
    // The global layers: no band — `w` at the sequence length, four heads over
    // one KV head, the widest head the kernel holds.
    Cell {
        label: "global  8192 w>=seq h=4 kv=1 d=512",
        rows: 8192,
        heads: 4,
        kv_heads: 1,
        head_dim: 512,
        window: 8192,
    },
    // The small cell: 256 rows under a window wider than the sequence,
    // which must read exactly as no mask at all.
    Cell {
        label: "small    256 w=512 h=4 kv=2 d=256",
        rows: 256,
        heads: 4,
        kv_heads: 2,
        head_dim: 256,
        window: 512,
    },
];

/// Rank-4 `[B, S, H, D]` attention — the encoder's layout — over `kind`.
fn attn_graph(rows: usize, nh: usize, nkv: usize, hd: usize, kind: MaskKind) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attention_band_cost");
    let q = g.input("q", Shape::new(&[1, rows, nh, hd], f));
    let k = g.input("k", Shape::new(&[1, rows, nkv, hd], f));
    let v = g.input("v", Shape::new(&[1, rows, nkv, hd], f));
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
        Shape::new(&[1, rows, nh, hd], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// Keys the band walks, summed over query blocks, and how many blocks there are.
fn band_keys(rows: usize, w: usize, head_dim: usize) -> (usize, usize) {
    let bm = band_rows(head_dim as u32);
    let (mut keys, mut blocks) = (0usize, 0usize);
    let mut q0 = 0usize;
    while q0 < rows {
        let lo = q0.saturating_sub(w);
        let hi = (q0 + bm + w).min(rows);
        keys += hi - lo;
        blocks += 1;
        q0 += bm;
    }
    (keys, blocks)
}

/// One cell on the module in force: best of 7 rounds after 3 warm-ups, in
/// milliseconds, with the band's TFLOP/s and the average keys a query block
/// visited. Returns the best round's milliseconds, which the encoder sum below
/// adds up.
fn measure(cell: &Cell) -> f64 {
    const WARM: usize = 3;
    const ROUNDS: usize = 7;
    let g = attn_graph(
        cell.rows,
        cell.heads,
        cell.kv_heads,
        cell.head_dim,
        MaskKind::Window(cell.window),
    );
    let q = data(cell.rows * cell.heads * cell.head_dim, 0.0007, false);
    let k = data(cell.rows * cell.kv_heads * cell.head_dim, 0.0009, true);
    let v = data(cell.rows * cell.kv_heads * cell.head_dim, 0.0005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let mut c = Session::new(Device::Gpu).compile(g);
    for _ in 0..WARM {
        let _ = c.run(&inputs).remove(0);
    }
    let ms = time_rounds(ROUNDS, 1, || {
        let _ = c.run(&inputs).remove(0);
    });
    let best = ms.iter().copied().fold(f64::INFINITY, f64::min);
    let bm = band_rows(cell.head_dim as u32) as f64;
    let (keys, blocks) = band_keys(cell.rows, cell.window, cell.head_dim);
    // Both multiplies: Q·Kᵀ over head_dim and P·V over the value width, two flops
    // a term, once per (head, block row, visited key).
    let flops = 2.0 * 2.0 * cell.heads as f64 * bm * keys as f64 * cell.head_dim as f64;
    let tflops = flops / (best * 1e-3) / 1e12;
    // The K and V rows the walk re-reads from the f32 arena: one row of each per
    // visited key, per head — nothing is cached between query blocks. This is the
    // figure that says whether a cell is bound by its bytes or by its
    // instructions: the two are not separable from a wall clock alone, but two
    // arrangements of this same walk differing only in this number were measured
    // (see the kernel header), and the GB/s below is what a reading of each puts
    // beside the other.
    let gb = (cell.heads as f64) * keys as f64 * cell.head_dim as f64 * 4.0 * 2.0 / 1e9;
    let gbps = gb / (best * 1e-3);
    let staged = if rlx_ir::env::flag("RLX_WGPU_ATTENTION_BAND_F32") {
        "f32"
    } else if rlx_wgpu::shader_f16_supported() {
        "f16"
    } else {
        "f32 (no SHADER_F16)"
    };
    eprintln!(
        "{:<38} {staged:<4} best {best:8.2} ms  median {:8.2} ms  {tflops:7.3} TFLOP/s  \
         {gb:6.2} GB re-read ({gbps:5.0} GB/s)  \
         band {} keys/block over {blocks} blocks (of {} rows), {} keys walked",
        cell.label,
        median(&ms),
        keys / blocks,
        cell.rows,
        keys,
    );
    best
}

/// The floor a reading includes: one elementwise pass over Q's own extent —
/// upload, one dispatch, readback — at both cells' sizes. A band reading is only
/// the kernel's own cost above this line, which is why the sliding cell's 47 ms
/// is not a kernel time: its floor is 11 ms of the same order, so its kernel
/// share is ~36 ms. The global cell's 574 ms is 28× its floor, so that one is
/// the kernel's.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_runs_own_floor_at_the_biggest_cell() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for (rows, heads, head_dim) in [(8192usize, 4usize, 512usize), (8192, 4, 256)] {
        let f = DType::F32;
        let mut g = Graph::new("attention_band_floor");
        let x = g.input("x", Shape::new(&[1, rows, heads, head_dim], f));
        let y = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Neg),
            vec![x],
            Shape::new(&[1, rows, heads, head_dim], f),
        );
        g.set_outputs(vec![y]);
        let data = data(rows * heads * head_dim, 0.0005, false);
        let inputs: [(&str, &[f32]); 1] = [("x", &data)];
        let mut c = Session::new(Device::Gpu).compile(g);
        for _ in 0..3 {
            let _ = c.run(&inputs).remove(0);
        }
        let ms = time_rounds(7, 1, || {
            let _ = c.run(&inputs).remove(0);
        });
        eprintln!(
            "floor: neg over [1,{rows},{heads},{head_dim}] best {:.2} ms, median {:.2} ms — \
             upload + one dispatch + readback of {} MiB",
            ms.iter().copied().fold(f64::INFINITY, f64::min),
            median(&ms),
            rows * heads * head_dim * 4 / (1 << 20),
        );
    }
}

/// The dense square cell at the sliding cell's shape (`MaskKind::None`): the
/// stand's four global layers, which say `None` over the whole pass. Its
/// preferred route is the cooperative chunked square, measured in the sibling
/// `tests/attention_dense_coop_cost.rs`; this line is what such a step keeps
/// where an adapter declines that form, so the run declines it by hand
/// (`RLX_WGPU_NO_DENSE_COOP`) and the band kernel walks the square with its band
/// switched off. Before the band kernel served dense squares the step fell to the
/// fused walk — the many-row arrangement's score matrix is `4 · 8192² · 4 B = 1 GiB`
/// here, past its 512 MiB cap — and read **7 897 ms** on this rig (same harness,
/// the previous tree; `RLX_WGPU_DEBUG_ATTN_MASK=1` printed `route=fused`). Three
/// rounds, not seven: it is the reference line.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_dense_square_cell_the_band_kernel_replaces() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _no_coop = rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_NO_DENSE_COOP", "1")));
    let cell = &CELLS[0];
    let g = attn_graph(
        cell.rows,
        cell.heads,
        cell.kv_heads,
        cell.head_dim,
        MaskKind::None,
    );
    let q = data(cell.rows * cell.heads * cell.head_dim, 0.0007, false);
    let k = data(cell.rows * cell.kv_heads * cell.head_dim, 0.0009, true);
    let v = data(cell.rows * cell.kv_heads * cell.head_dim, 0.0005, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let mut c = Session::new(Device::Gpu).compile(g);
    let _ = c.run(&inputs).remove(0);
    let ms = time_rounds(3, 1, || {
        let _ = c.run(&inputs).remove(0);
    });
    eprintln!(
        "dense (band, coop declined) {}: best {:.2} ms — every one of {} keys a row, \
         7 897 ms before the dense square step took a tiled walk",
        cell.label,
        ms.iter().copied().fold(f64::INFINITY, f64::min),
        cell.rows,
    );
}

/// The f16-staging table — the shipped module — followed by the sum the encoder
/// pass pays: twenty sliding layers and four global ones, which is the reason the
/// table has those two cells at all.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_band_kernels_cost_at_the_encoder_geometries() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _gate = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let mut per_layer = [0.0f64; 3];
    for (cell, best) in CELLS.iter().zip(per_layer.iter_mut()) {
        *best = measure(cell);
    }
    // 20 sliding + 4 global, the encoder's own counts (`EncoderStage`), on the
    // per-call figures above — each of which still carries the run's own
    // upload/readback floor, so this sum is an upper bound on the pass's share.
    // It is the band kernel's own arithmetic, not the pass: the four dense layers
    // stopped taking this kernel when the cooperative chunked form landed
    // (`tests/attention_dense_coop_cost.rs`), and only the twenty banded ones do.
    eprintln!(
        "band kernel if it served the whole pass: 20 × sliding {:.2} ms + 4 × global {:.2} ms = \
         {:.2} s of attention",
        20.0 * per_layer[0],
        4.0 * per_layer[1],
        (20.0 * per_layer[0] + 4.0 * per_layer[1]) / 1000.0,
    );
}

/// The same table over the f32 staging module, forced by
/// `RLX_WGPU_ATTENTION_BAND_F32=1` rather than by an adapter without half
/// precision — the two modules' readings have to come from one machine to be
/// comparable.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_f32_staged_band_kernels_cost_at_the_encoder_geometries() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // The switch is process-global and the f16 table reads the same one per
    // dispatch, so the two tables are taken one at a time — and installed through
    // the override guard, so a panic inside `measure` restores it for the other
    // tests in this binary rather than leaving the f32 module selected.
    let _gate = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let _f32_staging =
        rlx_ir::env::RuntimeOverrides::install(Some(("RLX_WGPU_ATTENTION_BAND_F32", "1")));
    for cell in &CELLS {
        measure(cell);
    }
}
