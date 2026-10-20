// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming bandwidth probe: the machine's read and copy ceilings on this
//! device.
//!
//! [`stream_ceiling`] measures a 64 MiB vectorised read (`stream_read.wgsl`)
//! and a 64 MiB vectorised copy (`copy_vec4.wgsl`) over a buffer far larger than
//! any cache, and reports bytes per second of wall time. A packed weight is read
//! and never written, so the read figure is the one a weight stream is compared
//! against; the copy's read + written rate is recorded beside it. Both are
//! wall-time rates around one submission and poll, not spec-sheet peaks. The
//! figures a run recorded are in `MEASUREMENTS.md`.
//!
//! # The round
//!
//! A round is `PASSES_PER_ROUND` dispatches in **one** compute pass and **one**
//! submission, then a wait poll: dispatches in a pass are ordered and the moves
//! are independent, so a pass boundary per dispatch would buy nothing. The wall
//! time is taken around encoder + submit + poll, so it counts the host's encode
//! work too — the shape a caller sees when it submits a phase. The copy's
//! direction flips on odd rounds, so both halves are read and written about
//! equally. `calibrate::Calibration`'s hard-coded `memory_bw_gbps` is neither
//! read nor fed here.

use rlx_ir::Tick;

use crate::kernels::{CopyParams, copy_vec4_kernel, stream_read_kernel};

/// Bytes one direction of the copy moves: 65 535 workgroups
/// (`max_compute_workgroups_per_dimension`'s guaranteed floor) of 64 invocations,
/// one `vec4<f32>` each. Both halves live in one buffer of twice this size.
const BYTES_PER_DIRECTION: u64 = 65_535 * 64 * 16;

/// Workgroups one copy dispatch issues.
const WORKGROUPS: u32 = 65_535;

/// Threads per workgroup in `stream_read.wgsl`.
#[cfg(test)]
const READ_WORKGROUP_SIZE: u32 = 256;

/// Workgroups one read dispatch issues.
const READ_WORKGROUPS: u32 = 4096;

/// `vec4<f32>` loads per thread in one read dispatch.
const READ_WORDS: u32 = 4;

/// Dispatches per round, in one pass and one submission.
const PASSES_PER_ROUND: usize = 8;

/// Timed rounds; the median is reported, the min/max kept as spread.
const ROUNDS: usize = 5;

/// Untimed rounds before `ROUNDS`, so the pipeline and buffer are resident.
const WARMUP_ROUNDS: usize = 2;

/// Uniform layout for `stream_read.wgsl`. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StreamReadParams {
    /// `vec4<f32>` count in the bound half.
    n: u32,
    /// Loads per thread; always [`READ_WORDS`].
    words_per_thread: u32,
    _p1: u32,
    _p2: u32,
}

/// The machine's streaming ceilings, measured on this device: one vectorised
/// copy pass over a buffer far larger than any cache (read + written bytes per
/// second of wall time), and one read-only pass over the same half (bytes read
/// per second).
pub struct StreamCeiling {
    /// Copy: read + written bytes per second, median over the rounds.
    pub gb_s: f64,
    pub min_gb_s: f64,
    pub max_gb_s: f64,
    /// Read-only stream: bytes read per second, median over the rounds.
    pub read_gb_s: f64,
    pub read_min_gb_s: f64,
    pub read_max_gb_s: f64,
    pub rounds: usize,
    /// The buffer one round copies. The read stream reads exactly this many
    /// bytes per dispatch.
    pub bytes_per_direction: u64,
}

/// Why a probe produced no figure. Three readings a record must keep apart: a
/// host with no adapter has nothing to measure on, a device fault means the probe
/// ran and its work failed, and a panic means the probe itself came apart — the
/// last two must not be recorded as the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CeilingFailure {
    /// The process has no wgpu adapter.
    NoAdapter,
    /// The probe ran and its own device work produced no figure: a validation
    /// error, a wait poll that did not complete, or a round its clock did not
    /// measure.
    DeviceWork(String),
    /// The probe panicked. Constructed by the caller that catches it, since only
    /// the caller can see a panic.
    Panicked(String),
}

impl CeilingFailure {
    /// The reading's own name, for a record's `unavailable=` field: the three
    /// ways a probe can produce no figure stay distinguishable there.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NoAdapter => "no_adapter",
            Self::DeviceWork(_) => "device_work",
            Self::Panicked(_) => "panicked",
        }
    }

    /// The detail behind a device fault or a panic; `None` for a missing adapter,
    /// which has nothing more to say.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::NoAdapter => None,
            Self::DeviceWork(reason) | Self::Panicked(reason) => Some(reason),
        }
    }
}

impl std::fmt::Display for CeilingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no wgpu adapter"),
            Self::DeviceWork(reason) => write!(f, "device work failed: {reason}"),
            Self::Panicked(reason) => write!(f, "probe panicked: {reason}"),
        }
    }
}

/// Measure the copy and read ceilings on the process-global device.
///
/// `Err(NoAdapter)` when the process has no wgpu adapter, `Err(DeviceWork)` when
/// the probe's own device work raised a validation error or its wait poll failed
/// — i.e. when the probe ran and did not produce a figure. A caller that catches
/// the probe's panic records it as `Panicked`.
///
/// An error scope is per device, not per call, so for the probe's duration any
/// validation error the shared device reports is read as this probe's, so a run
/// that measures it should not be running other device work alongside.
pub fn stream_ceiling() -> Result<StreamCeiling, CeilingFailure> {
    let Some(dev) = crate::device::wgpu_device() else {
        return Err(CeilingFailure::NoAdapter);
    };
    // The probe's own device work is fenced in an error scope: wgpu reports a
    // validation failure by dropping the command, not by aborting, so a bad
    // buffer or binding must not read as a missing adapter, nor as a plausible
    // rate from a dispatch that never ran. `pop` returns a future and this path
    // is synchronous, so it is driven by `pollster` the way device bring-up is; a
    // guard dropped without popping pops itself, so a panic inside `measure`
    // cannot leave the scope on the process-global device.
    let scope = dev.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let measured = measure(dev);
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(CeilingFailure::DeviceWork(format!(
            "validation error: {error}"
        )));
    }
    measured
}

/// [`stream_ceiling`]'s measurement, run inside the caller's error scope.
fn measure(dev: &crate::device::WgpuDevice) -> Result<StreamCeiling, CeilingFailure> {
    let copy = copy_vec4_kernel(&dev.device);
    // A bind group over `layout` with one entry per bound buffer, in binding
    // order. wgpu holds each buffer alive through the bind group, so nothing
    // else has to.
    let bind = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu stream ceiling bg"),
            layout,
            entries: &entries,
        })
    };
    let uniform = |bytes: &[u8]| {
        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu stream ceiling params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&u, 0, bytes);
        u
    };

    // Both halves of the working set live in one buffer: the kernel copies
    // `arena[out_off>>2 + gid.x] = arena[in_off>>2 + gid.x]`, so a direction is
    // just which half is the source. `n` is f32 words per direction.
    let n = (BYTES_PER_DIRECTION / 4) as u32;
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu stream ceiling"),
        size: 2 * BYTES_PER_DIRECTION,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // One bind group per direction: the uniform write and the submission are two
    // queue operations, and a bind group that already names its half cannot be
    // re-pointed by a later write in the same submission. The warm-up alternates
    // too, so the first timed round starts from the other half and neither
    // direction is systematically first.
    let copy_bg: Vec<_> = (0..2u32)
        .map(|half| {
            let p = CopyParams {
                n,
                in_off: half * n,
                out_off: (1 - half) * n,
                _p0: 0,
                _p1: 0,
                _p2: 0,
                _p3: 0,
                _p4: 0,
            };
            bind(&copy.bgl, &[&buf, &uniform(bytemuck::bytes_of(&p))])
        })
        .collect();
    let mut rates = timed_rounds(
        dev,
        &copy.pipeline,
        &copy_bg,
        WORKGROUPS,
        PASSES_PER_ROUND as u64 * 2 * BYTES_PER_DIRECTION,
    )
    .map_err(CeilingFailure::DeviceWork)?;

    // Read-only stream on the first half of the same buffer. `n` is the half's
    // vec4 count; the kernel reads exactly that many vectors, so the bytes are
    // the copy's per-direction figure and the two rates compare without a
    // scale factor.
    let read = stream_read_kernel(&dev.device);
    let n_vec4 = (BYTES_PER_DIRECTION / 16) as u32;
    let sink = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu stream read sink"),
        size: READ_WORKGROUPS as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let rp = StreamReadParams {
        n: n_vec4,
        words_per_thread: READ_WORDS,
        _p1: 0,
        _p2: 0,
    };
    let read_bg = bind(&read.bgl, &[&buf, &uniform(bytemuck::bytes_of(&rp)), &sink]);
    let mut read_rates = timed_rounds(
        dev,
        &read.pipeline,
        &[read_bg],
        READ_WORKGROUPS,
        PASSES_PER_ROUND as u64 * BYTES_PER_DIRECTION,
    )
    .map_err(CeilingFailure::DeviceWork)?;

    rates.sort_by(f64::total_cmp);
    read_rates.sort_by(f64::total_cmp);
    Ok(StreamCeiling {
        gb_s: rates[ROUNDS / 2],
        min_gb_s: rates[0],
        max_gb_s: rates[ROUNDS - 1],
        read_gb_s: read_rates[ROUNDS / 2],
        read_min_gb_s: read_rates[0],
        read_max_gb_s: read_rates[ROUNDS - 1],
        rounds: ROUNDS,
        bytes_per_direction: BYTES_PER_DIRECTION,
    })
}

/// Time [`ROUNDS`] rounds of one arm after [`WARMUP_ROUNDS`] untimed ones, and
/// return the timed rounds' rates in GB/s. `bgs` alternate per round so a two-
/// half arm does not read the same half first every time; `bytes_per_round` is
/// one round's traffic. The error string says which way the arm produced no
/// figure.
fn timed_rounds(
    dev: &crate::device::WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    bgs: &[wgpu::BindGroup],
    workgroups: u32,
    bytes_per_round: u64,
) -> Result<Vec<f64>, String> {
    let mut rates = Vec::with_capacity(ROUNDS);
    for r in 0..WARMUP_ROUNDS + ROUNDS {
        let t0 = Tick::now();
        if !run_pass(dev, pipeline, &bgs[r % bgs.len()], workgroups) {
            return Err(format!("round {r}: the wait poll did not complete"));
        }
        if r >= WARMUP_ROUNDS {
            let ns = Tick::now().elapsed_ns(t0);
            // A round the clock did not measure has no rate; the probe reports
            // that round rather than an infinite one.
            if ns == 0 {
                return Err(format!("round {r}: the clock read zero"));
            }
            rates.push(bytes_per_round as f64 / (ns as f64 / 1e9) / 1e9);
        }
    }
    Ok(rates)
}

/// [`PASSES_PER_ROUND`] dispatches into one compute pass and one submission,
/// then a wait poll. Dispatches in a pass are ordered and these moves are
/// independent, so a pass boundary per dispatch would buy nothing. `false` when
/// the poll does not complete: the dispatches did not run to completion, so the
/// round has no figure.
fn run_pass(
    dev: &crate::device::WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: u32,
) -> bool {
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu stream ceiling"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu stream ceiling"),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        for _ in 0..PASSES_PER_ROUND {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    crate::backend::note_device_submit();
    dev.queue.submit([enc.finish()]);
    crate::backend::note_device_wait();
    dev.device.poll(wgpu::PollType::wait_indefinitely()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the read dispatch's arithmetic without a device: 4096 × 256 threads ×
    /// 4 vectors covers the 64 MiB half with 64 vectors (1 KiB) to spare. The
    /// overhang has to be positive and smaller than one thread sweep (256 · 4
    /// vectors), so the shader's guard skips the tail of the last sweep only.
    #[test]
    fn the_read_geometry_covers_the_half_it_reads() {
        let vec4_per_direction = BYTES_PER_DIRECTION / 16;
        assert!(
            crate::kernels::STREAM_READ_WGSL.contains("@workgroup_size(256)"),
            "the shader's workgroup size must match READ_WORKGROUP_SIZE"
        );
        let threads = READ_WORKGROUPS as u64 * READ_WORKGROUP_SIZE as u64;
        assert_eq!(
            threads * READ_WORDS as u64 - vec4_per_direction,
            64,
            "64 vec4 = 1 KiB over, inside one thread's sweep"
        );
    }

    /// Prints both measurements. `#[ignore]`d like the crate's other device
    /// measurements: a round pair moves ~1.5 GiB, which does not belong in the
    /// suite's runtime, and the stand takes the figures a record quotes. The read
    /// dispatch's geometry is pinned above; the only assertions here are that each
    /// rate is finite and positive with its min ≤ median ≤ max, because a rate is
    /// a property of the machine under test.
    ///
    /// ```text
    /// cargo test -p rlx-wgpu --release --lib bandwidth -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "device measurement: the stand takes the figures a record quotes"]
    fn the_stream_ceiling_is_measured_on_this_device() {
        if rlx_ir::env::skip_unless_device("wgpu", true, crate::is_available()) {
            return;
        }
        let dev = crate::device::wgpu_device().expect("adapter present");
        let c = stream_ceiling()
            .unwrap_or_else(|failure| panic!("the probe produced no ceiling: {failure}"));
        eprintln!(
            "device={} backend={:?} gb_s={:.1} min_gb_s={:.1} max_gb_s={:.1} \
             read_gb_s={:.1} read_min_gb_s={:.1} read_max_gb_s={:.1} \
             bytes_per_direction={} passes_per_round={}",
            dev.name,
            dev.backend,
            c.gb_s,
            c.min_gb_s,
            c.max_gb_s,
            c.read_gb_s,
            c.read_min_gb_s,
            c.read_max_gb_s,
            c.bytes_per_direction,
            PASSES_PER_ROUND,
        );
        eprintln!(
            "copy per-direction share={:.1} GB/s  read={:.1} GB/s  read/share={:.2}",
            c.gb_s / 2.0,
            c.read_gb_s,
            c.read_gb_s / (c.gb_s / 2.0),
        );
        assert!(c.gb_s.is_finite() && c.gb_s > 0.0, "rate {} GB/s", c.gb_s);
        assert!(c.min_gb_s > 0.0 && c.min_gb_s <= c.gb_s && c.gb_s <= c.max_gb_s);
        assert!(
            c.read_gb_s.is_finite() && c.read_gb_s > 0.0,
            "read rate {} GB/s",
            c.read_gb_s
        );
        assert!(
            c.read_min_gb_s > 0.0
                && c.read_min_gb_s <= c.read_gb_s
                && c.read_gb_s <= c.read_max_gb_s,
            "read min/median/max {:.1}/{:.1}/{:.1}",
            c.read_min_gb_s,
            c.read_gb_s,
            c.read_max_gb_s,
        );
    }
}
