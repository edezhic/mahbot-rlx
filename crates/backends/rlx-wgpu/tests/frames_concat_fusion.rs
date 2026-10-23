// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The mel front-end's framing folds into ONE device step on wgpu.
//!
//! `Graph::stft` / `Graph::spectrogram` — and any front-end that rolls its own
//! framing out of the public builders — slice the frame matrix one frame at a
//! time (`narrow_` + `reshape_` per frame) and stack the rows with one
//! `concat_`. On device that is a `Step::Narrow` and a `Step::Concat` input per
//! frame: 200 dispatches for 100 frames of 400 samples, for a matrix that is one
//! strided read of the source. `lower::detect_frames_pattern` recognises the
//! pattern and emits a single `Step::Frames` for it.
//!
//! What this file pins, per shape:
//!
//! * **the census** — the fold happened, and it is the *whole* schedule: one
//!   `frames` step, no `narrow` and no `concat` left behind. A step count is a
//!   fact about the schedule rather than about the machine, so it is the witness
//!   that survives a contended box;
//! * **the values** — the fused step's output is the plain-Rust framing
//!   (`out[f, o, j] = x[o, f·hop + j]`), bit for bit. Nothing here is a sum, so
//!   anything but identity would be a kernel addressing bug rather than a
//!   precision story;
//! * **the builders** — `Graph::stft` / `Graph::spectrogram` are what the fold is
//!   for, and their own front-end must reach it (what `narrow`/`concat` they keep
//!   is `rfft`'s block layout, and must not scale with the frame count);
//! * **a consumer** — the framing feeding a `matmul` reaches the CPU backend's
//!   answer and is not zeros. A route whose step writes where nothing reads looks
//!   exactly like this from the census alone, so the deviceless reference is what
//!   rules it out;
//! * **the refusals** — what is pinned, and what is only read. Pinned here, each
//!   with the premise that makes it that clause's case and the refused graph's own
//!   values: a source with rows of its own, a row `reshape` that permutes them, a
//!   hop the rows do not share, a row with a second consumer, a graph-output row, a
//!   complex source (its values read as the lane view the element-wise kernel would
//!   have moved), a parameter framing the concat arm bakes, a destination the plan
//!   places on the source's live bytes, and one graph carrying a foldable framing
//!   *beside* a refused one, so a refusal cannot take the foldable one's route with
//!   it. Route only, and this is the note that says so: a zero-length leading dim,
//!   whose destination has no elements — its values are empty whatever the route
//!   does — so its premise is the source's element count and its reading the census.
//!   In its own binary, because its class needs one slot per tensor to be the clause
//!   that decides: a source the split-QKV fold rewrites
//!   (`frames_split_qkv_source.rs`).
//!
//!   Read-verified rather than pinned, and this is the note that says so: the
//!   sharded / not-bindable-whole arena pair (a case needs an arena past the shard
//!   cap), the source's home (only a parameter can be parked in the weight buffer,
//!   and the destination's provenance refuses such a framing first — the bake pin
//!   below is that graph), a shape that is not static (`WgpuExecutable::compile`
//!   defers a graph with dynamic dims before lowering, so the clause is a guard
//!   rather than a route here), and the rows' range (each row is a `narrow` of the
//!   source and the hop is uniform, so the widest row is the last one and its own
//!   validity is the clause). The fold is a fold or nothing: a refused graph keeps
//!   its per-frame steps and still computes the right matrix.
//!
//! * **the mode the fold's step replaced** — the fold is taken only where the
//!   trimmed axis is the one a caller buckets: the frame axis, which is the only
//!   row axis a framing over a source with no rows of its own presents. There a
//!   folded framing scales with the active extent, pinned by
//!   `a_folded_framing_leaves_the_frames_past_the_extent_unwritten` (which takes
//!   the framing tensor itself as the graph's output, and reads both a dividing and
//!   a non-dividing trim). Where the source carries rows of its own the fold is not
//!   taken at all, so those rows — and the mode that trims them — stay with the
//!   per-frame steps: `a_framing_over_a_source_with_rows_keeps_its_per_frame_steps`
//!   for the census and the values, `a_framing_over_a_row_source_keeps_the_mode`
//!   for the trim itself.
//!
//! * **the builders, consumed** — `Graph::spectrogram` over an input, fed to a
//!   matmul and read against the CPU device: the census half of the builders'
//!   coverage is in `the_front_ends_own_builders_fold_too`, and
//!   `a_builder_framing_over_an_input_feeds_a_consumer_with_the_cpus_answer` is
//!   the values half. A parameter or constant source is the case the bake refusal
//!   keeps off this route by design, so no coverage is owed there.
//!
//! The routing knobs are neutralised as a set (`common::neutralise_device_route_env`),
//! since the fold stands down under more than one of them; this fold's own concat
//! lever is the one `pin_the_device_route` adds, and its on-state reading is a test
//! binary of its own (`frames_concat_host_knob.rs`).
//!
//! The two shapes are the ones this stage is run at, and the framing is measured at:
//! 76 frames × 400 samples at hop 160, and the same matrix at 100 frames.

mod common;

use common::{
    assert_bit_identical, frames_signal_len, framing, framing_reference as reference,
    framing_signal as signal, framing_starts as uniform_starts, max_abs_diff, planned_slots,
};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// `(n_frames, frame_len, hop)` — the two shapes this stage is measured at.
const SHAPES: [(usize, usize, usize); 2] = [(76, 400, 160), (100, 400, 160)];

/// `[B, T] → [n_frames, B, frame_len]` — the same framing over a source whose
/// last axis is the one being sliced, so the narrow is an inner-axis one.
fn rank2_rows(b: usize, n_frames: usize, frame_len: usize, hop: usize) -> Graph {
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_rank2");
    let x = g.input("x", Shape::new(&[b, t], DType::F32));
    let y = framing(
        &mut g,
        x,
        1,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, b as i64, frame_len as i64],
    );
    g.set_outputs(vec![y]);
    g
}

/// The step census `g`'s own lowering produces on wgpu, by step kind. A fold
/// that stopped firing is then the caller's own census assertion rather than a
/// value comparison that a slower route would also pass.
fn device_census(g: &Graph) -> std::collections::BTreeMap<&'static str, usize> {
    WgpuExecutable::compile(g.clone()).step_kind_counts()
}

/// The concat work a census carries, as dispatches: one per input in the plain
/// form (`concat`), one for the whole op where the fold takes it
/// (`concat_many`).
fn concat_steps(census: &std::collections::BTreeMap<&'static str, usize>) -> usize {
    census.get("concat").copied().unwrap_or(0) + census.get("concat_many").copied().unwrap_or(0)
}

/// The pins below read the fold's *default* route: the shared set of routing knobs
/// (`common::neutralise_device_route_env`) plus this fold's own op levers, which the
/// shared set leaves to the op that owns them — the framed concat is a concat, so
/// `RLX_WGPU_CONCAT_HOST` decides it — and the planner knob the fold's *placement*
/// premises are read off (`RLX_ARENA_NO_REUSE`): a graph refused because the plan put
/// the destination on the source's bytes is only that case under the planner's reuse,
/// so an environment that turns reuse off would have the live-range pin reading a
/// different graph than it names. A masked read is the only way to say what these
/// pins say when the ambient environment disagrees; a lever's own test needs the
/// process to itself, so those are test binaries of their own
/// (`frames_concat_host_knob.rs`, `frames_split_qkv_source.rs`).
fn pin_the_device_route() {
    common::neutralise_device_route_env();
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "0");
    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "0");
}

#[test]
fn the_framing_folds_to_one_step_and_reproduces_the_matrix_exactly() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    for (n_frames, frame_len, hop) in SHAPES {
        let (g, t) = (
            common::framed_over_input(n_frames, frame_len, hop),
            common::frames_signal_len(n_frames, frame_len, hop),
        );
        let census = device_census(&g);
        assert_eq!(
            census.get("frames"),
            Some(&1),
            "{n_frames}×{frame_len}: the framing must be one `frames` step; census {census:?}"
        );
        assert_eq!(
            (census.get("narrow").copied(), concat_steps(&census)),
            (None, 0),
            "{n_frames}×{frame_len}: the per-frame steps must be gone; census {census:?}"
        );
        assert_eq!(
            census.values().sum::<usize>(),
            1,
            "{n_frames}×{frame_len}: the frame matrix is the whole graph, so it is the whole \
             schedule; census {census:?}"
        );

        let sig = signal(t);
        let mut exe = WgpuExecutable::compile(g);
        let got = exe.run(&[("x", &sig)]).remove(0);
        let want = reference(&sig, 1, t, &uniform_starts(n_frames, hop), frame_len);
        assert_bit_identical(
            &got,
            &want,
            &format!("{n_frames}×{frame_len} fused framing"),
        );
    }
}

/// The kernel's other dispatch branch. Past `MAX_WORKGROUPS_PER_DIM · FRAMES_WORKGROUP`
/// words the grid needs a second dimension and the shader's
/// `gid.x + gid.y · nwg.x · WG` linearization has to agree with the grid the runner
/// dispatches. Its geometry is pinned device-free in `backend/mod.rs`; this is that
/// same boundary executed, which no other shape here reaches (76 × 400 is 30 400
/// words). The matrix is read back whole rather than sampled, so a frame lost at the
/// dimension seam shows up as a value.
#[test]
fn a_framing_past_one_dispatch_dimension_reads_its_own_matrix() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    // The hop is narrower than the frame, so the source is narrower than the
    // destination and the plan cannot put the destination on its bytes — that case
    // has its own pin.
    let (n_frames, frame_len, hop) = (76, 55_200, 54_000);
    let t = frames_signal_len(n_frames, frame_len, hop);
    assert!(
        (n_frames * frame_len).div_ceil(rlx_wgpu::kernels::FRAMES_WORKGROUP as usize) > 65_535,
        "premise: this framing must need a second grid dimension (`dispatch_dims` splits past \
         65 535 workgroups, `MAX_WORKGROUPS_PER_DIM`), or this is the one-dimensional case again"
    );
    let g = common::framed_over_input(n_frames, frame_len, hop);
    assert_eq!(
        device_census(&g).get("frames"),
        Some(&1),
        "the framing must fold for this to read the folded step"
    );

    let sig = signal(t);
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    assert_bit_identical(
        &got,
        &reference(&sig, 1, t, &uniform_starts(n_frames, hop), frame_len),
        "a framing across the grid's second dimension is still the matrix it reads",
    );
}

/// A framing over a source that carries rows of its own. The fold's trim is the
/// frame axis, while the steps it would replace trim the *source's* rows — so the
/// fold stands down and those steps keep their route, their scaling and their
/// values: `frames` is absent, one `narrow` per frame is left, and the matrix is
/// the plain-Rust framing bit for bit.
#[test]
fn a_framing_over_a_source_with_rows_keeps_its_per_frame_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    const B: usize = 2;
    let (n_frames, frame_len, hop) = (10, 64, 32);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let g = rank2_rows(B, n_frames, frame_len, hop);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a source with rows of its own must keep its per-frame steps; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the per-frame steps must be the ones left; census {census:?}"
    );

    let sig = signal(B * t);
    let mut exe = WgpuExecutable::compile(g);
    let got = exe.run(&[("x", &sig)]).remove(0);
    let want = reference(&sig, B, t, &uniform_starts(n_frames, hop), frame_len);
    assert_bit_identical(&got, &want, "the row source's per-frame framing");
}

/// The concat fold's mode: the **planned** run is copied at a partial extent too.
///
/// Two 2-row pieces at half extent: the walked axis is the pieces' own leading dim,
/// which their producers write, so every piece's planned rows must reach the
/// output — a rule that cut the concat axis to the walked total would leave the
/// second piece's rows holding the previous run's bytes.
/// `a_framing_over_a_row_source_keeps_the_mode` is the same rule through the
/// framing that builds it.
#[test]
fn a_concat_copies_its_planned_run_at_a_partial_extent() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    const W: usize = 24;
    let mut g = Graph::new("concat_many_walked");
    let a = g.input("a", Shape::new(&[2, W], DType::F32));
    let b = g.input("b", Shape::new(&[2, W], DType::F32));
    let out = g.add_node(
        Op::Concat { axis: 0 },
        vec![a, b],
        Shape::new(&[4, W], DType::F32),
    );
    g.set_outputs(vec![out]);
    assert_eq!(
        concat_steps(&device_census(&g)),
        1,
        "the fold must take the concat for this pin to be about its mode"
    );
    let mut exe = WgpuExecutable::compile(g);
    let a1: Vec<f32> = (0..2 * W).map(|i| i as f32).collect();
    let b1: Vec<f32> = (0..2 * W).map(|i| 100.0 + i as f32).collect();
    let full = exe.run(&[("a", &a1), ("b", &b1)]).remove(0);
    assert_bit_identical(
        &full,
        &[a1.as_slice(), b1.as_slice()].concat(),
        "the full run",
    );

    let a2: Vec<f32> = (0..2 * W).map(|i| 7.0 + i as f32).collect();
    let b2: Vec<f32> = (0..2 * W).map(|i| 500.0 + i as f32).collect();
    exe.set_active_extent(Some((2, 4)));
    let walked = exe.run(&[("a", &a2), ("b", &b2)]).remove(0);
    let want = [a2.as_slice(), b2.as_slice()].concat();
    assert_bit_identical(
        &walked,
        &want,
        "both pieces' planned rows reach the output at a partial extent — the \
         walk shortens the pieces' own leading dim, not the concat axis",
    );
}

/// And the mode still means what it always meant there: the per-frame steps trim
/// the source's **rows**, which is the axis a caller bucketing rows reads. A full
/// run over one input, then a half-extent run over a *different* input, must leave
/// the first row of every frame the new input's and the second row the old one's —
/// a fold's trim (whole frames) or a missing trim (every row new) both fail that.
#[test]
fn a_framing_over_a_row_source_keeps_the_mode() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    const B: usize = 2;
    let (n_frames, frame_len, hop) = (4, 16, 16);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let starts = uniform_starts(n_frames, hop);
    let g = rank2_rows(B, n_frames, frame_len, hop);
    assert_eq!(
        device_census(&g).get("frames"),
        None,
        "the framing must stay per-frame for this pin to be about the mode it keeps"
    );

    let sig_a = signal(B * t);
    let sig_b: Vec<f32> = sig_a.iter().map(|v| v * 3.0 + 1.0).collect();
    let want_a = reference(&sig_a, B, t, &starts, frame_len);
    let want_b = reference(&sig_b, B, t, &starts, frame_len);
    let mut exe = WgpuExecutable::compile(g);
    let full = exe.run(&[("x", &sig_a)]).remove(0);
    assert_bit_identical(&full, &want_a, "the untrimmed run is the reference");

    exe.set_active_extent(Some((1, B)));
    let trimmed = exe.run(&[("x", &sig_b)]).remove(0);
    assert_eq!(trimmed.len(), full.len());
    for f in 0..n_frames {
        let base = f * B * frame_len;
        assert_bit_identical(
            &trimmed[base..base + frame_len],
            &want_b[base..base + frame_len],
            &format!("frame {f}'s first row must be the trimmed run's own input"),
        );
        let second = base + frame_len;
        assert_bit_identical(
            &trimmed[second..second + frame_len],
            &want_a[second..second + frame_len],
            &format!(
                "frame {f}'s second row must still be the full run's — the per-frame steps trim \
                 the source's rows, the axis a caller bucketing rows reads"
            ),
        );
    }
}

/// The other spelling of a frame slice — `Op::Slice` rows rather than `narrow_`.
/// wgpu has no step for `Op::Slice`: the compiler rewrites it into this fold's own
/// pattern before any backend lowers it (`rlx_fusion::lower_slice::LowerSlice`, run
/// by `rlx-compile`'s rewrite pass), so a publisher who slices instead of narrowing
/// reaches the fold rather than a fallback. Pinned as both halves: the rewrite the
/// compiler applies, and what it leaves (one `frames` step, the plain-Rust matrix),
/// then the whole runtime route on the values.
#[test]
fn a_slice_spelled_framing_reaches_the_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (8, 64, 16);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_slice_spelling");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let rows: Vec<NodeId> = (0..n_frames)
        .map(|f| {
            let row = g.add_node(
                Op::Slice {
                    axis: 0,
                    start: f * hop,
                    len: frame_len,
                    step: 1,
                },
                vec![x],
                Shape::new(&[frame_len], DType::F32),
            );
            g.reshape_(row, vec![1, frame_len as i64])
        })
        .collect();
    let y = g.concat_(rows, 0);
    g.set_outputs(vec![y]);
    // The premise: as built, this backend cannot lower the spelling at all — `Op::Slice`
    // has no step here, so the compiler's rewrite is what carries it. (Read, not
    // asserted from a census: the compile panics rather than scheduling nothing.)
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(g.clone())
    }));
    let Err(payload) = caught else {
        panic!("a bare `Op::Slice` was scheduled instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("op Slice") && msg.contains("not yet lowered"),
        "the premise is this backend's own refusal for the un-rewritten spelling; got: {msg}"
    );

    // The rewrite, applied by the pass the compiler applies: what it leaves is the
    // narrow/reshape/concat framing, and that is what the fold recognises.
    use rlx_opt::pass::Pass as _;
    let rewritten = rlx_opt::rlx_fusion::lower_slice::LowerSlice.run(g.clone());
    let census = device_census(&rewritten);
    assert_eq!(
        census.get("frames"),
        Some(&1),
        "the rewritten spelling must be the fold's own pattern; census {census:?}"
    );
    let sig = signal(t);
    let want = reference(&sig, 1, t, &uniform_starts(n_frames, hop), frame_len);
    assert_bit_identical(
        &WgpuExecutable::compile(rewritten)
            .run(&[("x", &sig)])
            .remove(0),
        &want,
        "the rewritten spelling is the plain-Rust matrix",
    );
    // And the publisher's route: the runtime compiles the Slice spelling itself.
    let got = Session::new(Device::Gpu)
        .compile(g)
        .run(&[("x", &sig)])
        .remove(0);
    assert_bit_identical(&got, &want, "the runtime route's own matrix");
}

#[test]
fn the_front_ends_own_builders_fold_too() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    // The framing the fold is for is the framing THIS TREE builds: `spectrogram`
    // and `stft` slice one frame at a time and stack the rows, which is the
    // pattern being recognised. A fold that only reached a hand-rolled matrix
    // would leave the builders' own front-end — and any consumer of it — paying
    // two dispatches a frame.
    //
    // What the census still carries under `narrow`/`concat` is the block layout
    // `rfft` builds for itself (pad, spectrum split), which is not framing: it
    // does not scale with the frame count, so the assertion is that it stays put
    // while the frames do.
    let census_of = |name: &str, stft: bool, (n_frames, frame_len, hop): (usize, usize, usize)| {
        let t = frames_signal_len(n_frames, frame_len, hop);
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(&[t], DType::F32));
        let y = if stft {
            g.stft(x, frame_len, hop, rlx_ir::FftNorm::Forward)
        } else {
            g.spectrogram(
                x,
                frame_len,
                hop,
                rlx_ir::ops::spectral::WindowKind::Hann,
                true,
                true,
            )
        };
        g.set_outputs(vec![y]);
        device_census(&g)
    };
    for (name, stft) in [("spectrogram", false), ("stft", true)] {
        let censuses: Vec<_> = SHAPES
            .iter()
            .map(|&shape| (shape, census_of(name, stft, shape)))
            .collect();
        for (shape, census) in &censuses {
            assert_eq!(
                census.get("frames"),
                Some(&1),
                "{name} at {shape:?} must fold its framing; census {census:?}"
            );
            eprintln!("[rlx-wgpu frames] {name} at {shape:?}: {census:?}");
        }
        let own: Vec<(usize, usize)> = censuses
            .iter()
            .map(|(_, c)| (c.get("narrow").copied().unwrap_or(0), concat_steps(c)))
            .collect();
        assert!(
            own.windows(2).all(|w| w[0] == w[1]),
            "{name}: narrow/concat counts {own:?} scale with the frame count, so they are framing \
             rather than `rfft`'s own block layout"
        );
    }
}

#[test]
fn a_consumer_of_the_fused_frames_reaches_the_cpu_answer() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    // The framing feeding an ordinary device op, which is where the fold could
    // go wrong invisibly: the concat's slot is written by `frames` and read by
    // the matmul, so a step whose output nothing read back would still look
    // right on the census. The CPU device has no such fold, so its answer over
    // the same graph is the independent witness.
    let (n_frames, frame_len, hop) = (16, 64, 32);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_consumed");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let frames = framing(
        &mut g,
        x,
        0,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    let table: Vec<f32> = (0..frame_len * 8)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[frame_len, 8], DType::F32),
    );
    let y = g.mm(frames, c);
    g.set_outputs(vec![y]);

    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        Some(&1),
        "the framing feeding the matmul must still fold; census {census:?}"
    );

    let sig = signal(t);
    let mut exe = WgpuExecutable::compile(g.clone());
    let gpu = exe.run(&[("x", &sig)]).remove(0);
    let cpu = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", &sig)])
        .remove(0);
    assert!(
        gpu.iter().any(|v| *v != 0.0),
        "the consumed framing is all zeros: the matmul read a slot nothing wrote"
    );
    let d = max_abs_diff(&gpu, &cpu);
    eprintln!("[rlx-wgpu frames] consumed framing vs CPU: max|Δ|={d:.3e}");
    assert!(
        d <= 1e-5 * (1.0 + cpu.iter().fold(0f32, |m, v| m.max(v.abs()))),
        "the consumed framing disagrees with the CPU backend by {d}"
    );
}

/// Two near-misses that are not the pattern: a hop the rows do not share, and a
/// frame sliced for a second purpose. Both keep their per-frame steps and their
/// values.
#[test]
fn a_near_miss_keeps_the_per_frame_steps_and_the_matrix() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (8, 64, 16);
    let t = frames_signal_len(n_frames, frame_len, hop) + frame_len;

    // (a) Starts that are not a uniform arithmetic progression: the fourth frame
    // is pulled out of line, so no single hop describes the rows.
    let mut starts = uniform_starts(n_frames, hop);
    starts[3] += 13;
    let mut g = Graph::new("frames_irregular_hop");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let y = framing(&mut g, x, 0, &starts, frame_len, &[1, frame_len as i64]);
    g.set_outputs(vec![y]);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a non-uniform hop must not fold; census {census:?}"
    );
    assert_eq!(
        (census.get("narrow").copied(), concat_steps(&census)),
        (Some(n_frames), 1),
        "the per-frame steps must be the ones left, the concat in one dispatch; census {census:?}"
    );
    let sig = signal(t);
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    assert_bit_identical(
        &got,
        &reference(&sig, 1, t, &starts, frame_len),
        "the refused irregular-hop framing still computes the matrix",
    );

    // (b) A frame sliced for a second purpose. Its narrow has two consumers, so
    // the concat's rows are no longer the only readers of a slot the fold would
    // leave unwritten.
    let starts = uniform_starts(n_frames, hop);
    let mut g = Graph::new("frames_shared_row");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let mut rows = Vec::with_capacity(n_frames);
    let mut shared = None;
    for (f, &s) in starts.iter().enumerate() {
        let frame = g.narrow_(x, 0, s, frame_len);
        if f == 2 {
            shared = Some(g.add(frame, frame));
        }
        rows.push(g.reshape_(frame, vec![1, frame_len as i64]));
    }
    let y = g.concat_(rows, 0);
    g.set_outputs(vec![y, shared.expect("the shared frame")]);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a frame with a second consumer must not fold; census {census:?}"
    );
    assert_eq!(
        (census.get("narrow").copied(), concat_steps(&census)),
        (Some(n_frames), 1),
        "the per-frame steps must be the ones left, the concat in one dispatch; census {census:?}"
    );
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    assert_bit_identical(
        &got,
        &reference(&sig, 1, t, &starts, frame_len),
        "the refused shared-row framing still computes the matrix",
    );
}

/// The builders' own front-end, consumed and read back: `Graph::spectrogram` over
/// an **input** source (a parameter or a constant source is the framing the bake
/// refusal keeps off this route on purpose), a matmul over its frames, and the
/// values against the same graph on the CPU device. The census alone does not
/// cover the path the builders ship; this does.
#[test]
fn a_builder_framing_over_an_input_feeds_a_consumer_with_the_cpus_answer() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (76, 400, 160);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("spectrogram_consumed");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let spec = g.spectrogram(
        x,
        frame_len,
        hop,
        rlx_ir::ops::spectral::WindowKind::Hann,
        true,
        // The power spectrum, not its log: a log of a small power is steep, and
        // this pin is about the framing reaching a consumer rather than about the
        // front-end chain's last cell (which the bench prices separately). The
        // framing's own values are pinned bit-exact by the pins above.
        false,
    );
    let bins = g
        .shape(spec)
        .dims()
        .last()
        .expect("the spectrum's bin axis")
        .unwrap_static();
    let w = g.param("w", Shape::new(&[bins, 32], DType::F32));
    let y = g.mm(spec, w);
    g.set_outputs(vec![y]);
    assert_eq!(
        device_census(&g).get("frames"),
        Some(&1),
        "the builder's framing must fold ahead of the consumer it feeds"
    );

    let sig = signal(t);
    let wv = signal(bins * 32);
    let want = {
        let mut exe = Session::new(Device::Cpu).compile(g.clone());
        exe.set_param("w", &wv);
        exe.run(&[("x", &sig)]).remove(0)
    };
    // The device side goes through the user-facing runtime, not the backend
    // directly: this is the arrangement the front-end is consumed in, so the
    // path from `Session` through the pipeline's lowering to the folded step is
    // what reaches the answer below.
    let mut exe = Session::new(Device::Gpu).compile(g);
    exe.set_param("w", &wv);
    let got = exe.run(&[("x", &sig)]).remove(0);
    // Read relatively: this route carries `rfft` and a `log` over an f32
    // accumulation, whose order this tree fixes on neither side, and a log of a
    // small power is steep. The framing's own values are pinned bit-exact by the
    // pins above; what this pin covers is the builder's framing reaching a
    // consumer's answer rather than a slot nothing wrote.
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&got, &want);
    assert!(
        d.is_finite() && scale > 1.0 && d / scale < 1e-4,
        "the consumed builder framing must reach the CPU answer; max|Δ| = {d:.3e} against values \
         up to {scale:.3e} ({:.1e} relative)",
        d / scale.max(1e-30),
    );
    assert!(
        got.iter().any(|v| *v != 0.0),
        "a consumer of the framing read all zeros — the signature of a slot nothing wrote"
    );
}

/// A row that is a graph output. The elided `narrow` no longer materializes the
/// frame where it stands, and the folded dispatch writes the frame matrix rather
/// than the row the graph promises — so a consumer of that output would read a slot
/// nothing wrote, which is the signature this stage's safety is about. The row keeps
/// its per-frame steps, and the exposed output is the frame they write.
#[test]
fn a_graph_output_row_refuses_the_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (8, 64, 16);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let starts = uniform_starts(n_frames, hop);
    let mut g = Graph::new("frames_output_row");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let mut rows = Vec::with_capacity(n_frames);
    let mut exposed = None;
    for (f, &s) in starts.iter().enumerate() {
        let frame = g.narrow_(x, 0, s, frame_len);
        if f == 3 {
            exposed = Some(frame);
        }
        rows.push(g.reshape_(frame, vec![1, frame_len as i64]));
    }
    let y = g.concat_(rows, 0);
    g.set_outputs(vec![y, exposed.expect("the exposed frame")]);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a graph-output row must not fold; census {census:?}"
    );
    assert_eq!(
        (census.get("narrow").copied(), concat_steps(&census)),
        (Some(n_frames), 1),
        "the per-frame steps must be the ones left, the concat in one dispatch; census {census:?}"
    );
    let sig = signal(t);
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]);
    assert_bit_identical(
        &got[0],
        &reference(&sig, 1, t, &starts, frame_len),
        "the refused framing still computes the matrix",
    );
    let want_row: Vec<f32> = (0..frame_len).map(|j| sig[starts[3] + j]).collect();
    assert_bit_identical(
        &got[1],
        &want_row,
        "the row the graph exposes is the frame the per-frame steps write",
    );
}

/// The active-extent mode (the trim a caller bucketing the frame axis gets) over a
/// folded framing: `total` counts whole frames, so the runner's `actual/upper`
/// scaling leaves the destination's first `active` frames written and the rest
/// untouched.
///
/// The pin is built so that it *can* fail: it reads the framing tensor itself rather
/// than a consumer of it, and it reads the difference from the side that a wrong
/// answer changes — the graph runs once over one input at full extent, then at half
/// extent over a *different* input, and the frames past the extent must still hold
/// the first run's values. A `false` here (the mode off for the whole graph, as the
/// per-frame steps' own scaling used to be vetoed) writes the second input
/// everywhere and fails that; so does dropping the step's own `total` scaling.
///
/// The frame count is deliberately odd and the total is not a multiple of the
/// kernel's workgroup, so the trim lands mid-workgroup and the last workgroup's
/// `lin >= params.total` return is exercised too.
#[test]
fn a_folded_framing_leaves_the_frames_past_the_extent_unwritten() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (13, 30, 15);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let starts = uniform_starts(n_frames, hop);
    let mut g = Graph::new("frames_active_extent");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let y = framing(&mut g, x, 0, &starts, frame_len, &[1, frame_len as i64]);
    g.set_outputs(vec![y]);
    assert_eq!(
        device_census(&g).get("frames"),
        Some(&1),
        "the framing must fold for this pin to be about the folded step"
    );

    let sig_a = signal(t);
    let sig_b: Vec<f32> = sig_a.iter().map(|v| v * 3.0 + 1.0).collect();
    assert!(
        sig_a.iter().zip(&sig_b).all(|(a, b)| a != b),
        "premise: the two runs' inputs must differ everywhere, or the values below could not \
         tell the runs apart"
    );
    let want_a = reference(&sig_a, 1, t, &starts, frame_len);
    let want_b = reference(&sig_b, 1, t, &starts, frame_len);

    let mut exe = WgpuExecutable::compile(g);
    let full = exe.run(&[("x", &sig_a)]).remove(0);
    assert_bit_identical(
        &full,
        &want_a,
        "the untrimmed run is the reference: the whole matrix, from the first input",
    );

    let active = n_frames / 2;
    exe.set_active_extent(Some((active, n_frames)));
    let trimmed = exe.run(&[("x", &sig_b)]).remove(0);
    assert_eq!(trimmed.len(), full.len());
    let written = active * frame_len;
    assert_bit_identical(
        &trimmed[..written],
        &want_b[..written],
        "the frames inside the extent must be the trimmed run's own input",
    );
    assert_bit_identical(
        &trimmed[written..],
        &full[written..],
        "the frames past the extent must still be the full run's — the folded step trims whole \
         frames",
    );

    // A ratio that does *not* divide the frame count: the documented rounding is
    // the runner's own `ceil`, so the prefix ends inside the frame the extent
    // names — words `[..130]` of 390 — and everything past it is untouched.
    let sig_c: Vec<f32> = sig_a.iter().map(|v| v * -2.0 - 0.5).collect();
    let want_c = reference(&sig_c, 1, t, &starts, frame_len);
    exe.set_active_extent(Some((1, 3)));
    let partial = exe.run(&[("x", &sig_c)]).remove(0);
    let partial_written = (full.len() as u32).div_ceil(3) as usize;
    assert!(
        !partial_written.is_multiple_of(frame_len),
        "premise: this ratio must not divide the frame count, or this run is the case above again"
    );
    assert_bit_identical(
        &partial[..partial_written],
        &want_c[..partial_written],
        "the rounded prefix must be the trimmed run's own input",
    );
    assert_bit_identical(
        &partial[partial_written..],
        &trimmed[partial_written..],
        "and past the rounding the previous run's values must still stand",
    );

    // And an upper bound wider than the frame count: the trim is still the
    // destination's own prefix — `ceil(390 · 1/100)` = 4 words, inside the first
    // frame — where the per-row steps this step replaces would have trimmed each
    // row's own word count and so cut a word out of every frame. The scaled word
    // counts the destination, which is what makes this the prefix a caller reads.
    let sig_d: Vec<f32> = sig_a.iter().map(|v| v * 7.0 - 2.0).collect();
    let want_d = reference(&sig_d, 1, t, &starts, frame_len);
    exe.set_active_extent(Some((1, 100)));
    let narrow = exe.run(&[("x", &sig_d)]).remove(0);
    let narrow_written = (full.len() as u32).div_ceil(100) as usize;
    assert!(
        narrow_written < frame_len,
        "premise: this ratio must round to a prefix shorter than one frame, or it is not the \
         wider-upper case"
    );
    assert_bit_identical(
        &narrow[..narrow_written],
        &want_d[..narrow_written],
        "the rounded prefix must be the trimmed run's own input",
    );
    assert_bit_identical(
        &narrow[narrow_written..],
        &partial[narrow_written..],
        "and past it the previous run's values must still stand",
    );
}

/// A complex source: `frames.wgsl` moves one arena word per element, while a
/// C64 element is two contiguous lanes — so the fold must refuse rather than
/// write half a frame matrix. The narrow/concat steps it keeps are lane-aware.
#[test]
fn a_complex_source_keeps_its_lane_aware_per_frame_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (4, 8, 4);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_complex_src");
    let x = g.input("x", Shape::new(&[t], DType::C64));
    let y = framing(
        &mut g,
        x,
        0,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    g.set_outputs(vec![y]);
    // The premise: only the dtype separates this graph from one that folds.
    let mut twin = Graph::new("frames_complex_src_twin");
    let xt = twin.input("x", Shape::new(&[t], DType::F32));
    let yt = framing(
        &mut twin,
        xt,
        0,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    twin.set_outputs(vec![yt]);
    assert_eq!(
        device_census(&twin).get("frames"),
        Some(&1),
        "premise: the f32 twin of this framing folds, so the dtype clause is what refuses it"
    );
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a C64 source must not fold; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the refused framing keeps its per-frame steps; census {census:?}"
    );

    // A C64 element is two lanes and the route it keeps is lane-aware, so its
    // values are the same framing read in lane units: `t` elements of the source
    // are `2t` lanes, and a frame is `2 · frame_len` of them at twice the hop.
    let lanes: Vec<f32> = (0..2 * t).map(|i| (i as f32 * 0.011).cos()).collect();
    let got = WgpuExecutable::compile(g).run(&[("x", &lanes)]).remove(0);
    assert_bit_identical(
        &got,
        &reference(
            &lanes,
            1,
            2 * t,
            &uniform_starts(n_frames, 2 * hop),
            2 * frame_len,
        ),
        "the refused C64 framing is the matrix its lanes say",
    );
}

/// A framing whose destination is a tensor the concat arm bakes: a source built
/// only out of parameters makes such a concat, and the arm dispatches it once
/// ever (`once` / `slot_is_exclusive`), where the folded step has no bake and
/// would dispatch it on every call. So the fold stands down and leaves that arm
/// its route.
///
/// `RLX_WGPU_CONCAT_HOST`'s own reading is a test binary of its own —
/// `frames_concat_host_knob.rs`.
#[test]
fn a_parameter_framing_keeps_the_route_that_bakes_it() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (8, 64, 16);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_param_src");
    let w = g.param("w", Shape::new(&[t], DType::F32));
    let y = framing(
        &mut g,
        w,
        0,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    g.set_outputs(vec![y]);
    // The premise: the parameter sits in the activation arena rather than the
    // weight buffer, so it is the destination's bake that refuses this graph and
    // not the parked-source clause.
    let (w_off, _w_size) = planned_slots(&g, &[w])[0].expect("premise: the parameter has a slot");
    assert!(
        !rlx_wgpu::buffer::is_weight_off(w_off),
        "premise: the parameter must be in the activation arena for this shape to be the bake \
         refusal; its slot starts at byte {w_off}"
    );
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a framing the concat arm bakes must not fold; census {census:?}"
    );
    let sig = signal(t);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &sig);
    let got = exe.run(&[]).remove(0);
    assert_bit_identical(
        &got,
        &reference(&sig, 1, t, &uniform_starts(n_frames, hop), frame_len),
        "the refused parameter framing still computes the matrix",
    );
}

/// A source with a zero-length leading dim. Such a source has a leading dim of its
/// own — the fold's source-rows rule refuses it before the empty destination could
/// be dispatched over (the kernel's bound counts frames, so a step folded here would
/// read and write the arena's head). The per-frame steps it keeps are empty too.
#[test]
fn a_zero_length_leading_dim_keeps_the_per_frame_steps() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    const B: usize = 0;
    let (n_frames, frame_len, hop) = (4, 8, 4);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_zero_leading_dim");
    let x = g.input("x", Shape::new(&[B, t], DType::F32));
    let src = g.add(x, x);
    let y = framing(
        &mut g,
        src,
        1,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, B as i64, frame_len as i64],
    );
    g.set_outputs(vec![y]);
    // The premise: the source holds no elements, which is what makes the kernel's
    // frame-counted bound the wrong bound for this graph. Read off the shape, and
    // route only below: the destination has no elements either, so its values are
    // empty whatever the route does.
    assert_eq!(
        g.node(src).shape.num_elements(),
        Some(0),
        "premise: the source must have no elements for this to be the zero-length case"
    );
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a zero-length source must not fold; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the refused framing keeps its per-frame steps; census {census:?}"
    );
    // No value check here: this graph has no elements, which is exactly what it
    // is a pin about — `run` of an empty output is its own path, and the values
    // of a framing are pinned by the shapes above.
}

/// Two framings of one source in one graph, one the fold takes and one it refuses:
/// the walk is per concat, so a refusal must not take the foldable framing's route
/// with it — and each of the two matrices must still be the one its own rows say.
#[test]
fn a_foldable_framing_beside_a_refused_one_keeps_its_own_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (6, 32, 8);
    let t = frames_signal_len(n_frames, frame_len, hop) + frame_len;
    let mut g = Graph::new("frames_foldable_beside_refused");
    let x = g.input("x", Shape::new(&[t], DType::F32));
    let foldable = framing(
        &mut g,
        x,
        0,
        &uniform_starts(n_frames, hop),
        frame_len,
        &[1, frame_len as i64],
    );
    // The same source framed with one row out of line: a hop no single step
    // describes, so this framing is the one the fold refuses.
    let mut starts = uniform_starts(n_frames, hop);
    starts[2] += 3;
    let refused = framing(&mut g, x, 0, &starts, frame_len, &[1, frame_len as i64]);
    g.set_outputs(vec![foldable, refused]);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        Some(&1),
        "the foldable framing must fold — one `frames` step for it and none for the refused one; \
         census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the refused framing keeps one narrow per frame; census {census:?}"
    );
    let sig = signal(t);
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]);
    assert_bit_identical(
        &got[0],
        &reference(&sig, 1, t, &uniform_starts(n_frames, hop), frame_len),
        "the folded framing is the matrix its rows say",
    );
    assert_bit_identical(
        &got[1],
        &reference(&sig, 1, t, &starts, frame_len),
        "the refused framing beside it is the matrix its rows say",
    );
}

/// The source's own live range: the plan can place a tensor on the source's bytes
/// once its last reader is done, and this fold reads the source at the *concat*,
/// later than that reader. The fixture is the reachable form of the case: an
/// op-output source `[1, t]` whose last-axis narrows are copies — an inner-axis
/// narrow never aliases its source, whatever the hop, so the source dies at its last
/// narrow — with the hop wider than a frame, which makes the source wider than the
/// destination and is what lets the planner reuse its slot for the destination, on
/// the source's own head, where one dispatch would read words it has already
/// written. Nothing here argues that placement: the plan below is what asserts it.
/// The leading dim is a unit, so the rows rule does not cover this graph and this
/// clause is the one that refuses it.
#[test]
fn a_destination_on_the_sources_bytes_refuses_the_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (n_frames, frame_len, hop) = (4usize, 8usize, 16usize);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_destination_on_source");
    let x = g.input("x", Shape::new(&[1, t], DType::F32));
    let src = g.add(x, x);
    // No reshape between the narrow and the concat, so nothing here gives
    // `collapse_reshapes` a view to rewire and the slots below are this graph's own.
    let rows: Vec<NodeId> = (0..n_frames)
        .map(|f| g.narrow_(src, 1, f * hop, frame_len))
        .collect();
    let frames = g.concat_(rows, 0);
    g.set_outputs(vec![frames]);

    assert_eq!(
        g.node(src).shape.dims()[0].unwrap_static(),
        1,
        "premise: the source has no rows of its own, or the rows rule below is what refuses this \
         graph instead of the clause this pin is about"
    );
    assert!(
        hop > frame_len,
        "premise: a hop wider than a frame, so the source is wider than the destination and the \
         planner can hand the source's own slot to the destination; that it did so is asserted \
         from the plan below, not argued here"
    );
    let slots = planned_slots(&g, &[src, frames]);
    let (s, d) = (
        slots[0].expect("premise: the source has a slot"),
        slots[1].expect("premise: the destination has a slot"),
    );
    assert!(
        d.0 < s.0 + s.1 && s.0 < d.0 + d.1,
        "premise: the plan must place the destination on the source's bytes for this to be the \
         case; src {}..{}, destination {}..{}",
        s.0,
        s.0 + s.1,
        d.0,
        d.0 + d.1
    );

    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a destination on the source's live bytes must not fold; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the refused framing keeps its per-frame steps; census {census:?}"
    );

    let sig = signal(t);
    let src_values: Vec<f32> = sig.iter().map(|v| v + v).collect();
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    assert_bit_identical(
        &got,
        &reference(&src_values, 1, t, &uniform_starts(n_frames, hop), frame_len),
        "the refused framing is still the source's own values",
    );
}

/// A row whose `reshape` keeps the element count but permutes the source's leading
/// dims (`[2, 3, L]` viewed as `[3, 2, L]`). `frames.wgsl` enumerates the
/// destination as (frame, leading index, sample) with the leading index in the
/// source's row-major order, so folding this would write the matrix transposed: the
/// source-rows rule refuses it, and this is the case that says why the rule is
/// stated over the *source's dims* rather than over the reshape alone. The
/// per-frame route it keeps is the CPU device's answer.
#[test]
fn a_permuted_row_reshape_refuses_the_fold() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    pin_the_device_route();
    let (b0, b1, n_frames, frame_len, hop) = (2usize, 3usize, 4usize, 8usize, 8usize);
    let t = frames_signal_len(n_frames, frame_len, hop);
    let mut g = Graph::new("frames_permuted_rows");
    let x = g.input("x", Shape::new(&[b0, b1, t], DType::F32));
    let row_nodes: Vec<NodeId> = (0..n_frames)
        .map(|f| {
            let frame = g.narrow_(x, 2, f * hop, frame_len);
            g.reshape_(frame, vec![1, b1 as i64, b0 as i64, frame_len as i64])
        })
        .collect();
    let y = g.concat_(row_nodes, 0);
    g.set_outputs(vec![y]);
    let census = device_census(&g);
    assert_eq!(
        census.get("frames"),
        None,
        "a row reshape that permutes the source's leading dims must not fold; census {census:?}"
    );
    assert_eq!(
        census.get("narrow"),
        Some(&n_frames),
        "the per-frame steps must be the ones left; census {census:?}"
    );
    let sig = signal(b0 * b1 * t);
    let want = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &sig)])
        .remove(0);
    let got = WgpuExecutable::compile(g).run(&[("x", &sig)]).remove(0);
    assert!(
        want.iter().any(|v| *v != 0.0),
        "the CPU device's answer is the witness, so it must not be all zeros"
    );
    assert_bit_identical(
        &got,
        &want,
        "the refused permuted-reshape framing is the per-frame route's matrix on both backends",
    );
}
