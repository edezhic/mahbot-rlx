// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Frame slicing — the mel front-end's framing. The crate README carries the
//! measured table, the pre-fold reading and the verdict; the figures are recorded
//! there and not here, so a re-take has one place to correct. What this file adds
//! is the reading itself: the forms the API already expresses measured against each
//! other at the shapes that stage is run at (76 and 100 frames of 400 samples at
//! hop 160, and their 512-sample neighbours), on every device built in, with each
//! row carrying the four figures a form has to answer for:
//!
//! * **median µs** over [`ROUNDS`] runs after [`WARMUP`] — the time line;
//! * **steps** — the schedule the backend's own lowering produces for the graph
//!   after the legalization that backend itself applies (`RLX_WGPU_SCHEDULE=1`
//!   prints the same census as text for the timed graph, fusion included). A
//!   dispatch count is a fact about the schedule, not about the machine, so it is
//!   the witness that survives a contended box;
//! * **sub/wait** — device submissions and blocking waits per run;
//! * **peak KiB / arena KiB / tables** — the **Rust** heap high-water mark over
//!   the arm's own compile plus its runs (a counting global allocator, so the
//!   figure is exact *for that heap*; driver-side allocations — Metal buffers,
//!   wgpu staging — never pass through it, so the device side is the arena line),
//!   the bytes the backend's own planner asks for, and the
//!   constant tables the form carries: a bigger arena is what a batched form buys
//!   its dispatch count with.
//!
//! The forms:
//!
//! * **one slice per frame, stacked by `concat_`** — the construction
//!   `Graph::stft` and `Graph::spectrogram` build, and the one a front-end that
//!   rolls its own framing writes. `Op::Slice` is the other spelling of a frame
//!   slice and is not a separate row: wgpu has no arm for it, so the pipeline
//!   rewrites it (`rlx_fusion::lower_slice`, `step == 1`) to exactly this form —
//!   pinned end to end, rewrite and values, by
//!   `frames_concat_fusion.rs`'s `a_slice_spelled_framing_reaches_the_fold`.
//!   The wgpu lowering folds this shape into one `Step::Frames` (the crate README
//!   carries that reading), so on that device this row measures the fused step,
//!   not the per-frame spelling whose pre-fold figures the README records and this
//!   tool can no longer re-take: the one lever that stands the fold down
//!   (`RLX_WGPU_CONCAT_HOST`, recorded in the crate README) only takes the concat
//!   off its direct window path, not back to the pre-fold spelling;
//! * **`Op::Im2Col`** — the framing as a 1×`frame_len` kernel at stride `hop`,
//!   the same matrix by construction, through an op that stages a span per call
//!   on wgpu;
//! * **batched** — a single `Op::Gather` whose constant index table carries the
//!   whole frame matrix (`indices[f][j] = f·hop + j`), one node instead of at
//!   least one step per frame.
//!
//! The stages:
//!
//! * **frames** — the matrix alone;
//! * **front-end** — the framing plus the cell it feeds: the transform's own
//!   tables and ordinary multiplies, the band projection, the log chain;
//! * **transform** — the framing plus the transform's own tables alone, so the
//!   cell's remainder (band projection, log chain) is a difference between two
//!   rows on the sheet rather than an inference;
//! * **in pass** — the front-end folded into a pass that consumes its mel, with
//!   the consumer's (small) result read back. Read against the same pass with the
//!   mel handed in — **the pass floor** — it is what the stage costs *inside* a
//!   walk, where it is one piece of that walk's work rather than a graph whose own
//!   (much larger) result is read back. The difference between the two is the
//!   per-window hand-off, measured rather than implied;
//! * **host front-end** — the whole front-end on the host instead, its mel handed
//!   to the same pass as an input, so the mel's upload is inside the call rather
//!   than beside it. This is the stage's other home, and the row the device form
//!   has to beat.
//!
//! Every row whose `baseline` is another row prints **max|Δ|** against that witness:
//! the plain-Rust framing for a framing row — computed without the backend, since the
//! matrix is the whole output and its witness needs no stage — and this device's own
//! slice-per-frame form of the same stage for a whole-stage row.
//! A row that is its own baseline prints `-` in that column: a slice-per-frame
//! whole-stage row (transform, front-end, in-pass) and the pass floor. A `ZERO` (the
//! whole output zero) or a `NaN` there is the signature of a consumer reading a slot
//! nothing wrote — the failure this measurement exists to rule out for this route — so
//! it is checked for, not assumed. The host (cpu) form of the same stage is on the same
//! sheet, because "comfortably below the CPU route" is the other end of the bar.
//!
//! Take the reading cool. `just throttle` (or `scripts/check-throttle.sh`) refuses
//! under thermal pressure, and these medians are quoted in the crate README; on a
//! shared box the µs column moves with the load, so it is the step census, the
//! arena and the table bytes — facts about the schedule, not the machine — that
//! carry the finding.
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_frame_slicing           # CPU
//! cargo run -p rlx-bench --release --example bench_frame_slicing --features gpu
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use rlx_driver::Device;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape, Tick};
use rlx_runtime::{CompiledGraph, Session, is_available};

/// Timed runs kept after warm-up. The median of them is the figure reported.
const ROUNDS: usize = 21;
const WARMUP: usize = 5;

/// Peak-heap witness: counts live bytes and keeps the high-water mark, so a form
/// that trades memory for time has to say so.
struct PeakHeap;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static HEAP: PeakHeap = PeakHeap;

fn heap_reset() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn heap_peak_kib() -> usize {
    PEAK.load(Ordering::Relaxed) / 1024
}

/// `(n_frames, frame_len, hop)`: 76 and 100 frames of 400 samples at hop 160 — the
/// two shapes this stage is measured at — each with its 512-sample neighbour, the
/// transform length at the next power of two.
const SHAPES: [(usize, usize, usize); 4] = [
    (76, 400, 160),
    (100, 400, 160),
    (76, 512, 160),
    (100, 512, 160),
];

/// Mel bands the front-end projects onto.
const N_MELS: usize = 80;

/// How the frames are sliced out of the signal.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Framing {
    /// `narrow_` + `reshape_` per frame, stacked by `concat_`.
    NarrowPerFrame,
    /// `Op::Im2Col` at kernel `[1, frame_len]`, stride `[1, hop]`.
    Im2Col,
    /// One `Op::Gather` over a constant `[n_frames, frame_len]` index table.
    BatchedGather,
}

/// How far a row runs: the frame matrix alone, the front-end's own cell on top of
/// it, and the front-end folded into the pass that consumes it. The last pair
/// differ only by the front-end, so their difference is what the stage costs
/// *inside* a walk — where it is one call among the walk's work rather than a
/// graph whose own result is read back — which is the front-end-to-pass hand-off
/// this measurement exists to price. `HostFrontEnd` is the same stage with the
/// hand-off spent the other way: the front-end on the host and its mel uploaded.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Frames,
    /// The framing plus the transform's own tables — the cell below the band
    /// projection, so what remains of the front-end is visible as its own row.
    Transform,
    FrontEnd,
    /// Front-end + the consumer it feeds, with the consumer's result read back.
    InPass,
    /// The same consumer with the mel handed in, so the difference from `InPass`
    /// is the front-end alone — the pass's own floor.
    PassFloor,
    /// The front-end on the host and the same consumer over the mel it produced,
    /// handed to the pass as an input.
    HostFrontEnd,
}

/// A stage and the framing it rides on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Arm {
    stage: Stage,
    framing: Framing,
}

const ARMS: [Arm; 10] = [
    arm(Stage::Frames, Framing::NarrowPerFrame),
    arm(Stage::Frames, Framing::Im2Col),
    arm(Stage::Frames, Framing::BatchedGather),
    arm(Stage::Transform, Framing::NarrowPerFrame),
    arm(Stage::FrontEnd, Framing::NarrowPerFrame),
    arm(Stage::FrontEnd, Framing::BatchedGather),
    arm(Stage::InPass, Framing::NarrowPerFrame),
    arm(Stage::InPass, Framing::BatchedGather),
    arm(Stage::PassFloor, Framing::NarrowPerFrame),
    arm(Stage::HostFrontEnd, Framing::NarrowPerFrame),
];

/// One arm, spelled for the tables above.
const fn arm(stage: Stage, framing: Framing) -> Arm {
    Arm { stage, framing }
}

impl Arm {
    fn name(self) -> &'static str {
        match (self.stage, self.framing) {
            (Stage::Frames, Framing::NarrowPerFrame) => "frames: slice-per-frame",
            (Stage::Frames, Framing::Im2Col) => "frames: im2col",
            (Stage::Frames, Framing::BatchedGather) => "frames: batched-gather",
            (Stage::Transform, _) => "transform: slice-per-frame",
            (Stage::FrontEnd, Framing::NarrowPerFrame) => "front-end: slice-per-frame",
            (Stage::FrontEnd, Framing::BatchedGather) => "front-end: batched-gather",
            (Stage::InPass, Framing::NarrowPerFrame) => "in pass: slice-per-frame",
            (Stage::InPass, Framing::BatchedGather) => "in pass: batched-gather",
            (Stage::PassFloor, _) => "in pass: floor (mel handed in)",
            (Stage::HostFrontEnd, _) => "in pass: host front-end",
            // `Op::Im2Col` is wired at the frames stage only, and the pass floor
            // and the host front-end have no framing of their own: the cell they
            // feed is the same cell whichever framing produced the matrix, so
            // neither is a row of its own there. A table that gains such a row
            // reads as this rather than panicking the harness mid-measurement.
            (_, Framing::Im2Col) => "im2col (frames stage only)",
        }
    }

    /// The row this form is read against: for a batched spelling, the same stage's
    /// slice-per-frame form; for the host front-end, the same stage with the
    /// framing left on the device. The difference is the form's own cost.
    fn baseline(self) -> Option<Arm> {
        match self.stage {
            Stage::HostFrontEnd => Some(arm(Stage::InPass, Framing::NarrowPerFrame)),
            _ => (self.framing != Framing::NarrowPerFrame).then_some(Arm {
                framing: Framing::NarrowPerFrame,
                ..self
            }),
        }
    }
}

/// The deterministic signal a window's front-end is run over.
fn signal(t: usize) -> Vec<f32> {
    (0..t).map(|i| (i as f32 * 0.017).sin() * 0.5).collect()
}

/// The framing, in plain Rust — the witness the frames stage's rows are read against.
fn reference_frames(sig: &[f32], n_frames: usize, frame_len: usize, hop: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_frames * frame_len];
    for f in 0..n_frames {
        out[f * frame_len..(f + 1) * frame_len].copy_from_slice(&sig[f * hop..f * hop + frame_len]);
    }
    out
}

/// The signal length a framing of `n_frames` rows at this hop needs.
fn signal_len(n_frames: usize, frame_len: usize, hop: usize) -> usize {
    frame_len + (n_frames - 1) * hop
}

fn const_f32(g: &mut Graph, data: Vec<f32>, dims: &[usize]) -> NodeId {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(dims, DType::F32),
    )
}

/// A constant table of deterministic, non-trivial values in `dims`.
fn table(g: &mut Graph, dims: &[usize], scale: f32) -> NodeId {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * scale).collect();
    const_f32(g, data, dims)
}

/// The same, non-negative — a mel filterbank is a matrix of non-negative weights.
/// The stage has to be one: against a signed bank the power spectrum projects to a
/// negative "mel", and `log` of that is NaN, so every front-end row's `max|Δ|` would
/// be a NaN of the harness's own making and the column could no longer tell a route
/// defect from the fixture (see `max_abs_diff`).
fn bank(g: &mut Graph, dims: &[usize], scale: f32) -> NodeId {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * scale).collect();
    const_f32(g, data, dims)
}

/// `[T] → [n_frames, frame_len]`, in `framing`'s own spelling.
fn frames(
    g: &mut Graph,
    x: NodeId,
    framing: Framing,
    n_frames: usize,
    frame_len: usize,
    hop: usize,
) -> NodeId {
    match framing {
        Framing::BatchedGather => {
            // `indices[f][j] = f·hop + j`, so one `Op::Gather` reads the whole
            // frame matrix out of the signal in one node.
            let idx: Vec<f32> = (0..n_frames)
                .flat_map(|f| (0..frame_len).map(move |j| (f * hop + j) as f32))
                .collect();
            let idx = const_f32(g, idx, &[n_frames, frame_len]);
            g.add_node(
                Op::Gather { axis: 0 },
                vec![x, idx],
                Shape::new(&[n_frames, frame_len], DType::F32),
            )
        }
        Framing::Im2Col => {
            // A `1 × frame_len` kernel at stride `hop` over `[1, 1, 1, T]` walks
            // the same `col[f][j] = x[f·hop + j]` matrix out of the signal.
            let t = signal_len(n_frames, frame_len, hop);
            let nchw = g.reshape_(x, vec![1, 1, 1, t as i64]);
            let col = g.im2col(nchw, [1, frame_len], [1, hop], [0, 0], [1, 1]);
            assert_eq!(
                (
                    g.shape(col).dim(0).unwrap_static(),
                    g.shape(col).dim(1).unwrap_static()
                ),
                (n_frames, frame_len),
                "im2col framing is not [n_frames, frame_len]"
            );
            col
        }
        Framing::NarrowPerFrame => {
            let mut rows = Vec::with_capacity(n_frames);
            for f in 0..n_frames {
                let frame = g.narrow_(x, 0, f * hop, frame_len);
                rows.push(g.reshape_(frame, vec![1, frame_len as i64]));
            }
            g.concat_(rows, 0)
        }
    }
}

/// The transform's own cell: the DFT tables and the square, which is where the
/// 400-point transform's cost sits.
fn transform(g: &mut Graph, frames: NodeId, frame_len: usize) -> NodeId {
    let bins = frame_len / 2 + 1;
    let dft = table(g, &[frame_len, 2 * bins], 0.001);
    let spec = g.mm(frames, dft);
    g.mul(spec, spec)
}

/// The cell the framing feeds: the transform, then the band projection and the
/// log chain — all on device. Split at `transform` so the row above shows what
/// the transform itself costs and this one what the rest of the cell adds.
///
/// The band matrix is the non-negative [`bank`] rather than a signed [`table`], so
/// the power spectrum projects to a non-negative mel and the `log` above it stays
/// finite: this stage has to be a mel chain, or every comparison against it is a NaN
/// of the fixture's own making.
fn front_end(g: &mut Graph, frames: NodeId, frame_len: usize) -> NodeId {
    let bins = frame_len / 2 + 1;
    let pow = transform(g, frames, frame_len);
    let bank = bank(g, &[2 * bins, N_MELS], 0.002);
    let mel = g.mm(pow, bank);
    let eps = const_f32(g, vec![1e-10], &[1, 1]);
    let shifted = g.add(mel, eps);
    let log_mel = g.log(shifted);
    let four = const_f32(g, vec![4.0], &[1, 1]);
    let quarter = const_f32(g, vec![0.25], &[1, 1]);
    let lifted = g.add(log_mel, four);
    g.mul(lifted, quarter)
}

/// The consumer the front-end's mel feeds — the op whose answer is read back, so
/// an in-pass row is the stage inside a walk and not a standalone call.
fn consume(g: &mut Graph, mel: NodeId) -> NodeId {
    let head = table(g, &[N_MELS, 128], 0.003);
    g.mm(mel, head)
}

fn build(arm: Arm, shape: (usize, usize, usize)) -> Graph {
    let (n_frames, frame_len, hop) = shape;
    let t = signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new(arm.name());
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let out = match arm.stage {
        Stage::Frames => frames(&mut g, x, arm.framing, n_frames, frame_len, hop),
        Stage::Transform => {
            let fr = frames(&mut g, x, arm.framing, n_frames, frame_len, hop);
            transform(&mut g, fr, frame_len)
        }
        Stage::FrontEnd | Stage::InPass => {
            let fr = frames(&mut g, x, arm.framing, n_frames, frame_len, hop);
            let mel = front_end(&mut g, fr, frame_len);
            if arm.stage == Stage::InPass {
                consume(&mut g, mel)
            } else {
                mel
            }
        }
        // The same consumer, over a mel that is handed in: the difference from
        // the `InPass` rows is the front-end alone.
        Stage::PassFloor => {
            let mel = table(&mut g, &[n_frames, N_MELS], 0.002);
            consume(&mut g, mel)
        }
        // The same consumer again, but over a mel the host is expected to hand
        // over — an input rather than a constant, so the call pays the upload.
        Stage::HostFrontEnd => {
            let mel = g.input("mel", Shape::new(&[n_frames, N_MELS], DType::F32));
            consume(&mut g, mel)
        }
    };
    g.set_outputs(vec![out]);
    g
}

/// Bytes the graph carries as constant tables — the index table the batched form
/// buys its dispatch count with sits here.
fn table_bytes(graph: &Graph) -> usize {
    graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Constant { data } => Some(data.len()),
            _ => None,
        })
        .sum()
}

/// The backend's own schedule and plan for `graph`, where that backend is built
/// in: the step census and the planned arena bytes, over the graph the backend's
/// own legalization step produces (the timed run applies the rest of the
/// pipeline — fusion, DCE — on top, which only ever removes steps). The
/// alignment is the one the lowering itself plans with (`lower.rs`:
/// `plan_f32_uniform(&graph, 16)`), so this column is the arena the run holds.
#[cfg(feature = "gpu")]
fn wgpu_plan(graph: &Graph) -> Option<(BTreeMap<&'static str, usize>, usize)> {
    if !rlx_wgpu::is_available() {
        return None;
    }
    let graph = rlx_opt::legalize_or_rewrite_for_backend(graph.clone(), rlx_wgpu::SUPPORTED_OPS)
        .expect("wgpu legalization");
    let exe = rlx_wgpu::backend::WgpuExecutable::compile(graph.clone());
    let arena = rlx_wgpu::buffer::plan_f32_uniform(&graph, 16).arena_size;
    Some((exe.step_kind_counts(), arena))
}

#[cfg(not(feature = "gpu"))]
fn wgpu_plan(_graph: &Graph) -> Option<(BTreeMap<&'static str, usize>, usize)> {
    None
}

#[cfg(feature = "gpu")]
fn reset_device_counts() {
    rlx_wgpu::backend::reset_device_counts();
}

#[cfg(not(feature = "gpu"))]
fn reset_device_counts() {}

#[cfg(feature = "gpu")]
fn device_counts() -> (u64, u64) {
    (
        rlx_wgpu::backend::device_submit_count(),
        rlx_wgpu::backend::device_wait_count(),
    )
}

#[cfg(not(feature = "gpu"))]
fn device_counts() -> (u64, u64) {
    (0, 0)
}

/// Largest absolute difference between a row's output and its witness, after a
/// finiteness census.
///
/// The census runs FIRST, because `f64::max` returns the non-NaN operand: a
/// NaN- or infinity-poisoned output would otherwise fold to a small number and print
/// in this column as a clean row — the very signature the column exists to catch. It
/// is the crate's own test idiom (`crates/backends/rlx-wgpu/tests/common/mod.rs`,
/// `max_abs_diff`), so a row's figure here means what a pin's figure means there.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut census = (0usize, 0usize, 0usize);
    for v in a.iter().chain(b) {
        if v.is_nan() {
            census.0 += 1;
        } else if *v == f32::INFINITY {
            census.1 += 1;
        } else if *v == f32::NEG_INFINITY {
            census.2 += 1;
        }
    }
    if census != (0, 0, 0) {
        return f64::NAN;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

struct Row {
    arm: Arm,
    out: Vec<f32>,
    steps: Option<usize>,
    census: String,
    submits: Option<(u64, u64)>,
    median_us: f64,
    /// The 10th and 90th percentile of the same runs, in µs. On a shared box the
    /// median alone cannot say whether a difference between two rows is real, so
    /// each row carries its own spread.
    spread_us: (f64, f64),
    peak_kib: usize,
    arena_kib: Option<usize>,
    tables_kib: usize,
    max_diff: Option<f64>,
    zeros: bool,
}

/// Compile and run one arm: warm it up, take [`ROUNDS`] timed runs, and read the
/// output back against `witness`. `census` reads the backend's own schedule and
/// its device counters, which only the wgpu pass has.
fn measure(
    dev: Device,
    arm: Arm,
    shape: (usize, usize, usize),
    sig: &[f32],
    witness: Option<&[f32]>,
    census: bool,
) -> Row {
    let graph = build(arm, shape);
    let tables_kib = table_bytes(&graph) / 1024;
    heap_reset();
    let plan = if census { wgpu_plan(&graph) } else { None };
    let mut exec: CompiledGraph = Session::new(dev).compile(graph);
    // The host front-end row calls two graphs a time: the front-end on the host
    // and the pass on `dev`, which is handed its mel as an input — so the upload
    // of the mel is inside the call rather than beside it. The device half of the
    // census below is the pass alone; the front-end is not a step of it.
    let mut host = (arm.stage == Stage::HostFrontEnd).then(|| {
        Session::new(Device::Cpu).compile(build(
            Arm {
                stage: Stage::FrontEnd,
                ..arm
            },
            shape,
        ))
    });
    let call = |exec: &mut CompiledGraph, host: &mut Option<CompiledGraph>| match host {
        Some(h) => {
            let mel = h.run(&[("x", sig)]).remove(0);
            exec.run(&[("mel", &mel)]).remove(0)
        }
        None => exec.run(&[("x", sig)]).remove(0),
    };
    for _ in 0..WARMUP {
        let _ = call(&mut exec, &mut host);
    }
    // A second batch is run and dropped before the timed one. A device's pipelines
    // are created on first use, and without this the arm that first runs a stage
    // the device has not run yet carries that creation cost into its own median —
    // measured: the first front-end row read ~1.8× the same stage at the next
    // shape.
    for _ in 0..ROUNDS {
        let _ = call(&mut exec, &mut host);
    }
    if census {
        reset_device_counts();
    }
    let mut samples = Vec::with_capacity(ROUNDS);
    let mut out = Vec::new();
    for _ in 0..ROUNDS {
        let t0 = Tick::now();
        out = call(&mut exec, &mut host);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    samples.sort_unstable();
    Row {
        arm,
        steps: plan.as_ref().map(|p| p.0.values().sum()),
        census: plan
            .as_ref()
            .map(|p| {
                p.0.iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default(),
        submits: census.then(|| {
            let (submits, waits) = device_counts();
            (submits / ROUNDS as u64, waits / ROUNDS as u64)
        }),
        median_us: samples[ROUNDS / 2] as f64 / 1000.0,
        spread_us: (
            samples[ROUNDS / 10] as f64 / 1000.0,
            samples[ROUNDS - 1 - ROUNDS / 10] as f64 / 1000.0,
        ),
        peak_kib: heap_peak_kib(),
        arena_kib: plan.map(|p| p.1 / 1024),
        tables_kib,
        max_diff: witness.map(|w| max_abs_diff(&out, w)),
        zeros: out.iter().all(|v| *v == 0.0),
        out,
    }
}

fn key(arm: Arm, shape: (usize, usize, usize)) -> String {
    format!("{}@{shape:?}", arm.name())
}

/// One device's sheet: every row's time, keyed by arm and shape.
struct Report {
    timings: BTreeMap<String, f64>,
}

fn bench(dev: Device, name: &str, census: bool) -> Report {
    let mut report = Report {
        timings: BTreeMap::new(),
    };
    // Whether this device is one where the fold is live, which decides whether the
    // footnote below applies (see it).
    let mut folded = false;
    println!("frame slicing on {name}");
    println!(
        "  {:>6} {:>5} {:>5}  {:<28} {:>5} {:>7} {:>9} {:>15} {:>9} {:>9} {:>7} {:>10}",
        "frames",
        "frame",
        "hop",
        "arm",
        "steps",
        "sub/wait",
        "median µs",
        "p10–p90 µs",
        "peak KiB",
        "arena KiB",
        "tables",
        "max|Δ|"
    );
    let _ = std::io::stdout().flush();
    // Pipelines, the arena's first allocation and the queue's first fence land on
    // whichever row happens to be first; warm the device on the same arm the
    // first row measures so that row is measured like every other one.
    {
        let (n_frames, frame_len, hop) = SHAPES[0];
        let warm = build(ARMS[0], SHAPES[0]);
        let sig = signal(signal_len(n_frames, frame_len, hop));
        let mut exec = Session::new(dev).compile(warm);
        for _ in 0..WARMUP {
            let _ = exec.run(&[("x", &sig)]);
        }
    }
    for &shape in &SHAPES {
        let (n_frames, frame_len, hop) = shape;
        let t = signal_len(n_frames, frame_len, hop);
        let sig = signal(t);
        let sliced = reference_frames(&sig, n_frames, frame_len, hop);
        // Every row whose `baseline` is another row is read against it: the
        // plain-Rust framing for a framing row, this device's own slice-per-frame
        // form of the same stage for the rest — so a reading is a difference
        // between two forms and not between two numerics. Only a slice-per-frame
        // whole-stage row (transform, front-end, in-pass) and the pass floor print
        // `-` in that column: those are the rows that are their own baseline, while
        // a framing row's own witness is the plain-Rust framing, so it prints one.
        let mut measured: BTreeMap<Arm, Vec<f32>> = BTreeMap::new();
        let mut rows: Vec<Row> = Vec::new();
        for &arm in &ARMS {
            let witness: Option<Vec<f32>> = match arm.stage {
                Stage::PassFloor => None,
                Stage::Frames => Some(sliced.clone()),
                _ => arm.baseline().and_then(|b| measured.get(&b).cloned()),
            };
            let row = measure(dev, arm, shape, &sig, witness.as_deref(), census);
            measured.insert(arm, row.out.clone());
            println!(
                "  {n_frames:>6} {frame_len:>5} {hop:>5}  {:<28} {:>5} {:>7} {:>9.1}µs {:>7.1}–{:<7.1} {:>9} {:>9} {:>7} {:>10}{}",
                row.arm.name(),
                row.steps
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into()),
                row.submits
                    .map(|(s, w)| format!("{s}/{w}"))
                    .unwrap_or_else(|| "-".into()),
                row.median_us,
                row.spread_us.0,
                row.spread_us.1,
                row.peak_kib,
                row.arena_kib
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| "-".into()),
                row.tables_kib,
                row.max_diff
                    .map(|d| format!("{d:.2e}"))
                    .unwrap_or_else(|| "-".into()),
                if row.zeros { "  ZERO" } else { "" },
            );
            if !row.census.is_empty() {
                println!("         {:<24} {}", "", row.census);
            }
            let _ = std::io::stdout().flush();
            report.timings.insert(key(arm, shape), row.median_us);
            folded |= row.census.contains("frames");
            rows.push(row);
        }
        // Rows are read back by their arm, never by their position: reordering
        // `ARMS` must not re-point a comparison at a different row.
        let at = |a: Arm| {
            rows.iter()
                .find(|r| r.arm == a)
                .expect("every arm in ARMS is measured")
        };
        // The comparisons the bar is stated in: each batched spelling against the
        // slice-per-frame form of the same stage, and the host front-end against
        // the front-end left inside the walk.
        for other in [
            arm(Stage::Frames, Framing::Im2Col),
            arm(Stage::Frames, Framing::BatchedGather),
            arm(Stage::FrontEnd, Framing::BatchedGather),
            arm(Stage::InPass, Framing::BatchedGather),
            arm(Stage::HostFrontEnd, Framing::NarrowPerFrame),
        ] {
            let base = at(other
                .baseline()
                .expect("a non-baseline form has a baseline"));
            let other = at(other);
            // The direction is spelled out rather than left to the reader: a ratio
            // of medians reads as "fewer" or "more" depending on which way it came
            // out, and the bar is stated in both directions (a batched spelling is
            // the faster one; the device front-end is the slower one).
            let (direction, ratio) = if other.median_us <= base.median_us {
                ("faster", base.median_us / other.median_us.max(0.001))
            } else {
                ("slower", other.median_us / base.median_us.max(0.001))
            };
            println!(
                "    {n_frames:>3} frames · {:<28} → {:<28} {:>6.2}× {direction:<6} µs, {:>5} → \
                 {:>5} steps, {:>4} → {:>4} KiB arena",
                base.arm.name(),
                other.arm.name(),
                ratio,
                base.steps
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into()),
                other
                    .steps
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into()),
                base.arena_kib
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| "-".into()),
                other
                    .arena_kib
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
        }
        let floor = at(arm(Stage::PassFloor, Framing::NarrowPerFrame));
        for row in [
            at(arm(Stage::InPass, Framing::NarrowPerFrame)),
            at(arm(Stage::InPass, Framing::BatchedGather)),
        ] {
            println!(
                "    {n_frames:>3} frames · {:<28} = {:>7.1} µs of the pass's own {:>6.1} µs",
                row.arm.name(),
                (row.median_us - floor.median_us).max(0.0),
                floor.median_us,
            );
        }
        // The front-end's own cells, with the steps each adds — the framing is a
        // count that follows the work and the transform is its own cell, so the
        // decomposition is a fact about the schedule. The µs are the table's rows
        // above: on a shared box their p10–p90 windows overlap between adjacent
        // cells, which is why the counts are what this line states.
        for row in [
            at(arm(Stage::Frames, Framing::NarrowPerFrame)),
            at(arm(Stage::Transform, Framing::NarrowPerFrame)),
            at(arm(Stage::FrontEnd, Framing::NarrowPerFrame)),
        ] {
            println!(
                "    {n_frames:>3} frames · {:<28} {:>5} steps, {:>7.1}–{:<7.1} µs",
                row.arm.name(),
                row.steps
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into()),
                row.spread_us.0,
                row.spread_us.1,
            );
        }
        println!();
    }
    if folded {
        println!(
            "  note: `frames: slice-per-frame` is the spelling the fold replaces, so on this \
             device that row is the one fused step — the pre-fold reading of the same spelling \
             cannot be re-taken here and is recorded, with the method, in `rlx-wgpu`'s README."
        );
        println!();
    }
    report
}

/// The devices to measure: the ones built in *and* instantiable here. A device
/// compiled in without an adapter is skipped rather than handed to `Session::new`,
/// which panics on one it cannot instantiate, and the skip is printed so a sheet
/// missing a line is not read as a measurement that was not taken. The sibling
/// benches gate the same way; the census below gates on `rlx_wgpu::is_available`.
fn devices() -> Vec<(Device, &'static str)> {
    #[allow(unused_mut)]
    let mut out = vec![(Device::Cpu, "cpu")];
    let candidates: &[(&str, Device)] = &[
        #[cfg(feature = "metal")]
        ("metal", Device::Metal),
        #[cfg(feature = "gpu")]
        ("wgpu", Device::Gpu),
    ];
    for &(name, dev) in candidates {
        if is_available(dev) {
            out.push((dev, name));
        } else {
            println!("skipping {name}: built in, but no adapter on this host");
        }
    }
    out
}

fn main() {
    let devices = devices();
    let mut reports: Vec<(String, Report)> = Vec::new();
    for (dev, name) in devices {
        reports.push((
            name.to_string(),
            bench(dev, name, matches!(dev, Device::Gpu)),
        ));
    }
    // The other end of the bar: the same stage as a host pass, which is the form
    // the device rows have to beat to be worth the device at all.
    let Some((host_name, host)) = reports.first() else {
        return;
    };
    for (name, report) in reports.iter().skip(1) {
        println!("{name} against the host form ({host_name}) of the same stage:");
        for &shape in &SHAPES {
            for arm in [
                arm(Stage::FrontEnd, Framing::NarrowPerFrame),
                arm(Stage::InPass, Framing::NarrowPerFrame),
                arm(Stage::InPass, Framing::BatchedGather),
            ] {
                let (Some(device_us), Some(host_us)) = (
                    report.timings.get(&key(arm, shape)),
                    host.timings.get(&key(arm, shape)),
                ) else {
                    continue;
                };
                println!(
                    "    {:>3} frames × {:<3} hop {:<3} · {:<28} {name} {device_us:>7.1} µs vs host \
                     {host_us:>6.1} µs ({:>5.2}×)",
                    shape.0,
                    shape.1,
                    shape.2,
                    arm.name(),
                    device_us / host_us.max(0.001),
                );
            }
            // The stage's other home, head to head: the front-end inside the walk
            // on `dev` against the same front-end on the host with its mel handed
            // to the same walk. The second row's call carries the mel's upload,
            // `n_frames × N_MELS × 4` bytes.
            let mel_kib = shape.0 * N_MELS * 4 / 1024;
            let (Some(in_pass), Some(host_pass)) = (
                report
                    .timings
                    .get(&key(arm(Stage::InPass, Framing::NarrowPerFrame), shape)),
                report.timings.get(&key(
                    arm(Stage::HostFrontEnd, Framing::NarrowPerFrame),
                    shape,
                )),
            ) else {
                continue;
            };
            println!(
                "    {:>3} frames × {:<3} hop {:<3} · {:<28} {name} in the walk {in_pass:>7.1} µs vs \
                 host front-end + pass {host_pass:>6.1} µs (+{mel_kib} KiB mel upload, enclosed) \
                 ({:>5.2}×)",
                shape.0,
                shape.1,
                shape.2,
                "in pass: front-end",
                in_pass / host_pass.max(0.001),
            );
        }
        println!();
    }
}
