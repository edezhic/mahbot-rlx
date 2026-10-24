// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The encoder's packed-weight GEMM, three ways, at the shapes the model
//! actually multiplies. Measurement only: every figure is printed and none is
//! asserted — a timing on this machine moves with load and with its thermal
//! state (`just throttle` before a reading), so a pinned number would be a pin on
//! the runner rather than on the kernel.
//!
//! # The arms
//!
//! * **packed Q8_0** — `Op::DequantMatMul` over a Q8_0 param, `x [m, k]` f32:
//!   how the embedding encoder's text tower reaches the multiply when the store
//!   rule keeps the packed bytes (`gemm=packed` on the stand's cell line). One
//!   `Session::run`.
//! * **f16 param** — an f16 `Param` and a plain `Op::MatMul`: the graph the stand
//!   hands over as `gemm=f16coop`. One `Session::run`. The dispatch counter is
//!   printed beside it so the reading is known to have been served by
//!   `matmul_coop_f16_f32acc` rather than by a fallback.
//! * **`matmul_f16_compute`** and **`matmul_coop16`** — the two kernels this path
//!   replaces, timed as *single dispatches* through their own bind groups, since
//!   no route dispatches either at these shapes today (`matmul_f16_compute` is
//!   reachable only through the >768 MiB forced-f16 route; on an aligned f16
//!   `Param` the cascade picks the cooperative path). Single-dispatch numbers are
//!   not the same reading as a full run and are labelled `raw`.
//!
//! # What a reading includes
//!
//! A full-run reading is one `Session::run`: upload of `x`, the dispatch (or
//! dispatches), and the readback of the `[m, n]` output. It is therefore
//! `kernel + the call's own floor`, and at these extents the floor is not a
//! rounding term. Measured on this rig
//! ([`the_runs_own_floor_at_the_tables_extents`], an elementwise neg over the same
//! extents, so it is upload + one dispatch + readback and nothing else):
//!
//! | extent      | floor  | host I/O  |
//! |-------------|--------|-----------|
//! | 8192×12288  | 105.9 ms | 7.3 GB/s |
//! | 8192×512    |   4.97 ms | 6.4 GB/s |
//! | 256×12288   |   3.81 ms | 6.3 GB/s |
//!
//! That is ~3.3 GB/s each way for the host boundary, and it is why the PLE row
//! below reads 1.77 TFLOP/s end to end while its kernel runs at 2.57: 384 MiB of
//! output costs ~50 ms to read back on its own, against a ~40 ms kernel. A
//! production pass does not pay this per projection — it keeps the tensors on the
//! device and reads back once — so the full-run columns compare *routes*, and the
//! single-dispatch table compares *kernels*. Neither is the pass's rate.
//!
//! The dispatch counter is printed per row: a full-run f16 reading whose count is
//! not 10 (one per call) was served by a fallback and is not this kernel's.
//!
//! TFLOP/s counts `2 · m · k · n` — the multiply's own work — and uses the best
//! of 7 rounds after 3 warm-ups. Run this file with `--ignored --nocapture`.

mod common;

use common::{data, time_rounds};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

/// One projection of the encoder: `[m, k] × [k, n]`.
struct GemmShape {
    label: &'static str,
    k: usize,
    n: usize,
}

/// The encoder's projections: the nine per-layer ones at the sliding-layer
/// geometry, the two the global layers use instead, and the PLE projection. The
/// "nine" are the per-layer rows; the global variants and PLE are
/// separate shapes the same pass runs, so a table that left them out would not
/// add up to the pass's 2.2 TFLOP.
const SHAPES: &[GemmShape] = &[
    GemmShape {
        label: "q         512→1024 ",
        k: 512,
        n: 1024,
    },
    GemmShape {
        label: "q_global  512→2048 ",
        k: 512,
        n: 2048,
    },
    GemmShape {
        label: "k         512→512  ",
        k: 512,
        n: 512,
    },
    GemmShape {
        label: "v         512→512  ",
        k: 512,
        n: 512,
    },
    GemmShape {
        label: "o        1024→512  ",
        k: 1024,
        n: 512,
    },
    GemmShape {
        label: "o_global 2048→512  ",
        k: 2048,
        n: 512,
    },
    GemmShape {
        label: "gate      512→2048 ",
        k: 512,
        n: 2048,
    },
    GemmShape {
        label: "up        512→2048 ",
        k: 512,
        n: 2048,
    },
    GemmShape {
        label: "down     2048→512  ",
        k: 2048,
        n: 512,
    },
    GemmShape {
        label: "inp_gate  512→512  ",
        k: 512,
        n: 512,
    },
    GemmShape {
        label: "proj      512→512  ",
        k: 512,
        n: 512,
    },
    GemmShape {
        label: "ple       512→12288",
        k: 512,
        n: 12288,
    },
];

const ROWS: [usize; 2] = [8192, 256];

/// `2·m·k·n` in TFLOP/s, from a best-of-7 figure in milliseconds.
fn tflops(m: usize, k: usize, n: usize, ms: f64) -> f64 {
    2.0 * (m * k * n) as f64 / (ms * 1e-3) / 1e12
}

/// `y = x · w` with `w` an f16 param, `x` and `y` f32 — the graph the new kernel
/// serves.
fn f16_graph(label: &str, m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new(label);
    let x = g.input("x", Shape::new(&[m, k], F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F16));
    let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[m, n], F32));
    g.set_outputs(vec![y]);
    g
}

/// The same multiply as the model reaches it: Q8_0-packed weights, one byte
/// stream param of the packed length.
fn packed_graph(label: &str, m: usize, k: usize, n: usize) -> (Graph, usize) {
    let scheme = QuantScheme::GgufQ8_0;
    let blocks = k / scheme.gguf_block_size() as usize;
    let bytes = n * blocks * scheme.gguf_block_bytes() as usize;
    let mut g = Graph::new(label);
    let x = g.input("x", Shape::new(&[m, k], F32));
    let w = g.param("w", Shape::new(&[bytes], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[m, n], F32),
    );
    g.set_outputs(vec![y]);
    (g, bytes)
}

/// Best-of-7 after 3 warm-ups, in milliseconds.
fn best_ms(mut run: impl FnMut()) -> f64 {
    for _ in 0..3 {
        run();
    }
    time_rounds(7, 1, &mut run)
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// The full-run arms, per row count.
fn measure_full_runs(m: usize) {
    eprintln!("── full runs, m = {m} ──");
    eprintln!(
        "{:<20} {:>10} {:>11}  | {:>10} {:>11}  {:>6}  {:>9}",
        "shape", "Q8_0 ms", "Q8_0 TF/s", "f16 ms", "f16 TF/s", "disps", "weights MB"
    );
    for s in SHAPES {
        let (k, n) = (s.k, s.n);
        let x = data(m * k, 0.0013, false);
        // Q8_0 arm.
        let (g, bytes) = packed_graph(s.label, m, k, n);
        let w: Vec<f32> = data(k * n, 0.0007, true);
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q8_0).expect("quantize");
        assert_eq!(packed.len(), bytes, "{}: packed length", s.label);
        let mut c = Session::new(Device::Gpu).compile(g);
        c.set_param_typed("w", &packed, DType::U8);
        let q_ms = best_ms(|| {
            let _ = c.run(&[("x", &x)]).remove(0);
        });
        // f16 arm.
        let mut c16 = Session::new(Device::Gpu).compile(f16_graph(s.label, m, k, n));
        c16.set_param("w", &w);
        let before = rlx_wgpu::coop_f16_f32acc_dispatch_count();
        let f_ms = best_ms(|| {
            let _ = c16.run(&[("x", &x)]).remove(0);
        });
        let disps = rlx_wgpu::coop_f16_f32acc_dispatch_count() - before;
        // The two weight images, which is also the memory the f16 hand-over costs
        // on top of the packed one: `packed.len()` bytes against `k·n·2`.
        eprintln!(
            "{:<20} {q_ms:>10.2} {:>11.3}  | {f_ms:>10.2} {:>11.3}  {disps:>6}  {:>4.1}/{:<4.1}",
            s.label,
            tflops(m, k, n, q_ms),
            tflops(m, k, n, f_ms),
            packed.len() as f64 / 1e6,
            (k * n * 2) as f64 / 1e6,
        );
    }
}

#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_encoder_gemms_at_the_big_row_count() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    measure_full_runs(ROWS[0]);
}

#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_encoder_gemms_at_the_small_row_count() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    measure_full_runs(ROWS[1]);
}

// ── single-dispatch baselines ───────────────────────────────────────────────

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

/// One dispatch of `pipeline` over freshly allocated buffers of this shape, best
/// of 7 after 3 warm-ups. No readback: the submit and the device wait are paid,
/// the output is not copied.
fn time_raw(
    dev: &rlx_wgpu::device::WgpuDevice,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    m: u32,
    k: u32,
    n: u32,
    grid: (u32, u32),
) {
    let arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("encoder_gemm_cost arena"),
        size: ((m * k + m * n) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let weights = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("encoder_gemm_cost f16 weights"),
        size: ((k * n) as u64) * 2,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let p = P {
        m,
        k,
        n,
        a_off: 0,     // A: f32 arena, m·k lanes
        b_off: 0,     // B: f16 weights, k·n lanes
        c_off: m * k, // C: f32 arena, right after A
        batch: 1,
        a_bs: 0,
        b_bs: 0,
        c_bs: 0,
        has_bias: 0,
        bias_off: 0,
        act_id: 0xFFFF,
        _0: 0,
        _1: 0,
        _2: 0,
    };
    let pbuf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("encoder_gemm_cost params"),
        size: std::mem::size_of::<P>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&p));
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("encoder_gemm_cost bg"),
        layout: bgl,
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
    let one = || {
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        dev.queue.submit(std::iter::once(enc.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    };
    let ms = best_ms(one);
    eprintln!(
        "  raw {label:<26} {ms:>9.2} ms  {:>8.3} TFLOP/s  grid {}×{}",
        tflops(m as usize, k as usize, n as usize, ms),
        grid.0,
        grid.1
    );
}

#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_single_dispatch_rate_of_each_candidate_kernel() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        return;
    };
    use rlx_wgpu::kernels as kern;
    let new = kern::matmul_coop_f16_f32acc_kernel(&dev.device);
    let coop16 = kern::matmul_coop16_kernel(&dev.device);
    let simt = kern::matmul_f16_compute_kernel(&dev.device);
    for m in ROWS {
        eprintln!("── one dispatch, m = {m} ──");
        for s in SHAPES {
            let (m, k, n) = (m as u32, s.k as u32, s.n as u32);
            eprintln!("{}", s.label);
            if let Some(kr) = new {
                time_raw(
                    dev,
                    "matmul_coop_f16_f32acc",
                    &kr.pipeline,
                    &kr.bgl,
                    m,
                    k,
                    n,
                    kern::coop_f16_f32acc_grid(m, n),
                );
            }
            if let Some(kr) = coop16 {
                time_raw(
                    dev,
                    "matmul_coop16",
                    &kr.pipeline,
                    &kr.bgl,
                    m,
                    k,
                    n,
                    (n.div_ceil(32), m.div_ceil(32)),
                );
            }
            if let Some(kr) = simt {
                time_raw(
                    dev,
                    "matmul_f16_compute",
                    &kr.pipeline,
                    &kr.bgl,
                    m,
                    k,
                    n,
                    (n.div_ceil(32), m.div_ceil(32)),
                );
            }
        }
    }
}

/// The floor a reading includes: one elementwise pass over the same extent —
/// upload, one dispatch, readback — at the two ends of the table's output sizes,
/// so a GEMM reading can be read as `kernel + floor` rather than as a kernel time.
#[test]
#[ignore = "measurement only: prints rates, asserts nothing"]
fn the_runs_own_floor_at_the_tables_extents() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for (m, n) in [
        (8192usize, 12288usize),
        (8192, 512),
        (256, 12288),
        (256, 512),
    ] {
        let mut g = Graph::new("encoder_gemm_floor");
        let x = g.input("x", Shape::new(&[m, n], F32));
        let y = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Neg),
            vec![x],
            Shape::new(&[m, n], F32),
        );
        g.set_outputs(vec![y]);
        let data = data(m * n, 0.0005, false);
        let mut c = Session::new(Device::Gpu).compile(g);
        let ms = best_ms(|| {
            let _ = c.run(&[("x", &data)]).remove(0);
        });
        let mib = m * n * 4 / (1 << 20);
        eprintln!(
            "floor [{m},{n}]: best {ms:8.2} ms — upload + one dispatch + readback of {mib} MiB \
             each way ({:.1} GB/s of the {:.0} MiB moved)",
            (2 * mib) as f64 / (ms * 1e-3) / 1e3,
            (2 * mib) as f64,
        );
    }
}

/// Which steps the packed arm takes at one shape — so the readings above are
/// attributable to a device kernel rather than a host fallback. Device-free in
/// what it asserts (it prints), but it needs the device to compile.
#[test]
#[ignore = "measurement only: prints the step names, asserts nothing"]
fn the_packed_arm_lands_on_a_device_step() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for (m, k, n) in [(256usize, 512usize, 512usize), (8192, 512, 12288)] {
        let (g, _) = packed_graph("packed steps", m, k, n);
        eprintln!(
            "packed Q8_0 m={m} k={k} n={n}: {:?}",
            common::legalised_step_names(&g)
        );
        let g16 = f16_graph("f16 steps", m, k, n);
        eprintln!(
            "f16 param  m={m} k={k} n={n}: {:?}",
            common::legalised_step_names(&g16)
        );
    }
}
