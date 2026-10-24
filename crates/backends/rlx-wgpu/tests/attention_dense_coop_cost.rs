// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The dense attention block's cooperative square form, measured. Measurement
//! only: every figure is printed and none is asserted — a timing on this machine
//! moves with load and with its thermal state (`just throttle` before a reading),
//! so a pinned number would be a pin on the runner rather than on the kernel.
//!
//! # The arms
//!
//! * **the multiplies** — the dense route's two, through
//!   `matmul_coop_f16_f32acc` with B in an f16 buffer (the route's form), one
//!   dispatch per chunk at the encoder's geometry: `scores = Q·Kᵀ`
//!   (`m = chunk, k = 512, n = 8192`) and `out = P·V` (`m = chunk, k = 8192,
//!   n = 512`), four `(batch, head)` tiles each, at the chunk sizes the route
//!   walks and at the whole sequence. Single-dispatch readings: the submit and
//!   device wait are paid, the output is not copied.
//! * **the step** — one `Op::Attention` at 8 192 rows, 4 heads over 1 KV head,
//!   head width 512, dense (`MaskKind::None`), through `Session::run`: the
//!   reading the stand's embedding cell is a multiple of.
//! * **the softmax** — `Op::Softmax` over one chunk's rows of keys, which is the
//!   route's largest pass outside the multiplies.
//!
//! Run with `--ignored --nocapture`. The figures the route's docs and the kernel
//! header record were taken from this file on an otherwise idle Apple M3 Pro
//! (14-core GPU, wgpu 30 over Metal).

mod common;

use common::{data, median, time_rounds};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

/// `Params` as every matmul shader in this crate declares it: 16 × `u32`. The
/// arena form reads `b_off`/`b_batch_stride` as f32 arena elements.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct P {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    batch: u32,
    a_bs: u32,
    b_bs: u32,
    c_bs: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _0: u32,
    _1: u32,
    _2: u32,
}

fn tflops(m: u32, k: u32, n: u32, batch: u32, ms: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 * batch as f64 / (ms * 1e-3) / 1e12
}

/// One dispatch of the route's kernel — `matmul_coop_f16_f32acc`, A f32 in the
/// arena and B f16 in its own buffer — over freshly allocated buffers, median of
/// 9 after 3 warm-ups, no readback. `batch` is the number of `(batch, head)`
/// tiles, which the route dispatches as the grid's `z`.
fn time_raw_weights(
    dev: &rlx_wgpu::device::WgpuDevice,
    label: &str,
    kernel: &rlx_wgpu::kernels::Kernel,
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
) -> f64 {
    let a_len = (m * k) as u64 * batch as u64;
    let b_len = (k * n) as u64 * batch as u64;
    let c_len = (m * n) as u64 * batch as u64;
    let arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_dense_coop_cost f32 arena"),
        size: (a_len + c_len) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let weights = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_dense_coop_cost f16 B"),
        size: b_len * 2,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let p = P {
        m,
        k,
        n,
        a_off: 0,
        b_off: 0,
        c_off: a_len as u32,
        batch,
        a_bs: m * k,
        b_bs: k * n,
        c_bs: m * n,
        has_bias: 0,
        bias_off: 0,
        act_id: 0xFFFF,
        _0: 0,
        _1: 0,
        _2: 0,
    };
    let pbuf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_dense_coop_cost params"),
        size: std::mem::size_of::<P>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&p));
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attention_dense_coop_cost bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights.as_entire_binding(),
            },
        ],
    });
    let grid = rlx_wgpu::kernels::coop_f16_f32acc_grid(m, n);
    let mut one = || {
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, batch);
        }
        dev.queue.submit(std::iter::once(enc.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    };
    for _ in 0..3 {
        one();
    }
    let rounds = time_rounds(9, 1, &mut one);
    let ms = median(&rounds);
    let best = rounds.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "  {label:<34} med {ms:>8.2} (best {best:>7.2}) ms  {:>7.3} TFLOP/s  grid {}×{}×{}",
        tflops(m, k, n, batch, ms),
        grid.0,
        grid.1,
        batch
    );
    ms
}

#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_two_dense_attention_gemms_at_the_encoder_geometry() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        return;
    };
    let Some(kr) = rlx_wgpu::kernels::matmul_coop_f16_f32acc_kernel(&dev.device) else {
        return;
    };
    let (rows, hd, heads) = (8192u32, 512u32, 4u32);
    eprintln!("── the route's kernel, one dispatch per chunk, batch = {heads} ──");
    for chunk in [1024u32, 2048, 8192] {
        eprintln!("scores  m={chunk} k={hd} n={rows}");
        let one = time_raw_weights(dev, "scores", kr, chunk, hd, rows, heads);
        eprintln!("out     m={chunk} k={rows} n={hd}");
        let two = time_raw_weights(dev, "out", kr, chunk, rows, hd, heads);
        eprintln!(
            "  both, ×{} chunk(s): {:.2} ms",
            rows / chunk,
            (one + two) * (rows / chunk) as f64
        );
    }
}

/// One dense `Op::Attention` at the encoder's global-layer shape, `[1, 8 192,
/// 4, 512]` Q over `[1, 8 192, 1, 512]` K/V. The whole route: the transposes,
/// both multiplies, the masked scale and the softmax.
fn dense_attention_graph(rows: usize, heads: usize, nkv: usize, hd: usize) -> Graph {
    let mut g = Graph::new("attention_dense_coop_cost");
    let q = g.input("q", Shape::new(&[1, rows, heads, hd], F32));
    let k = g.input("k", Shape::new(&[1, rows, nkv, hd], F32));
    let v = g.input("v", Shape::new(&[1, rows, nkv, hd], F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: heads,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, rows, heads, hd], F32),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_dense_step_at_the_encoder_geometry() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let rows = 8192usize;
    let (heads, nkv, hd) = (4usize, 1usize, 512usize);
    let g = dense_attention_graph(rows, heads, nkv, hd);
    let q = data(rows * heads * hd, 0.0005, false);
    let k = data(rows * nkv * hd, 0.0007, true);
    let v = data(rows * nkv * hd, 0.0004, true);
    let inputs: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
    let mut c = Session::new(Device::Gpu).compile(g);
    let _ = c.run(&inputs).remove(0);
    let mut one = || {
        let _ = c.run(&inputs).remove(0);
    };
    let ms = median(&time_rounds(5, 1, &mut one));
    let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
    eprintln!(
        "dense step rows={rows} heads={heads} nkv={nkv} hd={hd}: {ms:.2} ms \
         ({:.3} TFLOP/s over the two 2·m·k·n squares), cooperative dispatches {disps}",
        2.0 * 2.0 * (rows as f64) * (rows as f64) * (hd as f64) * (heads as f64)
            / (ms * 1e-3)
            / 1e12
    );
}

/// The two passes the dense route spends outside its multiplies: the score
/// matrix's masked scale (`Step::AttentionMask`) and its row softmax
/// (`Step::Softmax`), each at one chunk's shape — `4 · 1 024` rows of 8 192 keys,
/// the stand's global-layer geometry at `DENSE_COOP_CHUNK_ROWS`.
///
/// The softmax is timed through its own graph with a `Param` input, so no upload
/// is paid per run, and under `RLX_BENCH_DISPATCH_ONLY` so the output is not read
/// back either: the reading is the dispatch, which is what a step inside the
/// attention route costs. The masked scale has no op of its own — it is a step the
/// route emits — so its figure is derived from the softmax's per-pass rate: one
/// read and one write of the same 128 MiB.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_score_matrix_passes_at_the_chunk_shape() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (outer, inner) = (4096usize, 8192usize);
    let mut g = Graph::new("softmax_chunk");
    let x = g.param("x", Shape::new(&[outer, inner], F32));
    let y = g.add_node(
        Op::Softmax { axis: -1 },
        vec![x],
        Shape::new(&[outer, inner], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param("x", &data(outer * inner, 0.0001, false));
    let _dispatch_only =
        rlx_ir::env::RuntimeOverrides::install(Some(("RLX_BENCH_DISPATCH_ONLY", "1")));
    let mut one = || {
        let _ = c.run(&[]).remove(0);
    };
    let ms = median(&time_rounds(5, 1, &mut one));
    let bytes = (outer * inner * 4) as f64;
    eprintln!(
        "softmax [{outer}, {inner}] one dispatch (3 passes over the row): {ms:.2} ms, \
         {:.1} GB/s if all three were one pass, {:.1} GB/s over 2 reads + 1 write",
        bytes / (ms * 1e-3) / 1e9,
        3.0 * bytes / (ms * 1e-3) / 1e9
    );
}

/// What the route emits, in order, for the encoder's global-layer shape: the
/// transposes, the two f16 casts, then each chunk's two multiplies and its
/// softmax. Printed rather than asserted — the parity file pins the route, this
/// one shows the schedule a reading above is a reading of.
#[test]
#[ignore = "measurement only: prints the schedule, asserts nothing"]
fn the_dense_form_schedule_at_the_encoder_geometry() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let g = dense_attention_graph(8192, 4, 1, 512);
    let names = common::legalised_step_names(&g);
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for n in &names {
        *counts.entry(n).or_default() += 1;
    }
    eprintln!("{} steps: {counts:?}", names.len());
    eprintln!("{names:?}");
}
