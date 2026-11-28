// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measurement only and ignored: what the im2col conv panel's cut over the output's spatial
//! axis costs and buys on the product's own encoder — the recognition front at every rung of
//! the ladder ([`FRAME_RUNGS`]) — in the arena it reserves, in the schedule it emits, and in
//! the time it runs, beside the exactness the cut has to keep.
//!
//! The whole-panel arm is the reservation this tree replaced: `RLX_WGPU_IM2COL_PANEL_BYTES=0`
//! lifts the budget and the panel is the conv's whole extent, which is what the encoder front
//! booked before the cut (17.7 → 106.2 MB across the ladder). The chunked arm is the shipped
//! one: the key unset, the 16 MiB budget, the panel cut into whole 32-column tiles. Both arms
//! are the same graph and the same weights, so the pair prices the reservation and nothing
//! else; the budget is a snapshot of the process's configuration, so each arm reloads it
//! around its own build and compile.
//!
//! Per rung the run prints both arms' `arena_scratch_bytes` — the memory half: the chunked
//! arm's is at or under the budget and smaller than the whole arm's wherever the whole panel
//! was over it, and it runs 8 847 360 → 15 482 880 B across this ladder (2 → 7 even chunks)
//! against 17 694 720 → 106 168 320 B whole, both pinned by
//! `backend::conv_route_decision_tests`' ladder — beside both arms' `arena_bytes`, which is
//! what a row actually holds, the compact activation plan the split arena is built from, and
//! both arms' `im2col_gpu` / `matmul` / `conv2d` counts, which are the schedule half: a chunk
//! is one gather and one multiply, so the two arms differ by exactly the chunks the cut added
//! and by nothing else. Both columns of the elementwise in-place ladder are this probe:
//! `RLX_WGPU_NO_INPLACE=1` plans the plain layout, unset the shipped one.
//!
//! The exactness half is the reason the cut is over the spatial axis and not the reduction
//! one: a column carries no accumulation, so the two arms must produce the same output bit
//! for bit on the same input, and the run asserts it rather than a tolerance. A differing
//! element means a chunk boundary moved a sum.
//!
//! The time is paired and alternating at two grains. The front's own pair — both arms held in
//! one process, `ROUNDS` rounds, the arm that runs first swapping each round, [`WARMUP`] passes
//! per arm discarded first — is the row's reading, and its spread on a shared machine is
//! several times the effect. The second grain is the conv the cut actually changes, priced on
//! its own over [`PROBE_ROUNDS`] rounds of [`PROBE_ITER`] passes: conv 2 at the rung's own
//! shape, both arms, plus a third arm at a quarter of the budget that amplifies the chunk count
//! about four times, whose paired difference divided by the chunks it added is the slope — one
//! chunk's cost — that the shipped arm's figure is read from. Nothing here asserts a time: the
//! printed pair and the implied percentage are the reading, and only each median's finiteness
//! is checked.
//!
//! Needs the installed model directory; a machine without it or without a wgpu adapter skips
//! and says why. Run with `RLX_ALLOW_THROTTLE=1 cargo test -p mahbot-inference --release
//! --features hand-run --test conv_panel_cost -- --ignored --nocapture`, after `just
//! throttle`; the whole sweep is a couple of minutes.
use mahbot_inference::graphs::{self, Compiled};
use mahbot_inference::loader::{Pair, Role};
use mahbot_inference::transcribe::FRAME_RUNGS;
use rlx_wgpu::backend::WgpuExecutable;

mod support;

/// The shipped budget, and the chunked arm's ceiling.
const BUDGET: u64 = 16 * 1024 * 1024;

/// Rounds each arm is timed over, and the passes per arm discarded before them.
const ROUNDS: usize = 20;
const WARMUP: usize = 3;

/// The second of the conv front's three `3×3` stride-2 convs: `480·3·3` floats of reduction
/// per output column.
const CONV2_K: u64 = 480 * 3 * 3;

/// The encoder front's two panel-routed convs' output columns at `frames`: the second conv's
/// `32 · len₂` and the third's `16 · len₃`, both axes halving at every conv (128 mel bins →
/// 64 → 32 → 16 tall, `frames` → `len₂` → `len₃` wide). The third conv's reduction is the
/// same `480·3·3`, so its panel is the smaller of the two and the second's is the graph's.
fn panel_columns(frames: usize) -> (u64, u64) {
    let halve = |n: usize| (n - 1) / 2 + 1;
    let len2 = halve(halve(frames));
    let len3 = halve(len2);
    (32 * len2 as u64, 16 * len3 as u64)
}

/// The whole panel the uncapped arm books, in bytes: the widest conv's `480·3·3` reduction
/// over its output columns, one float per element.
fn whole_panel_bytes(frames: usize) -> u64 {
    let (conv2, _) = panel_columns(frames);
    CONV2_K * conv2 * 4
}

/// The gathers each of the front's two panel-routed convs needs at the shipped budget, per
/// rung: conv 2's `32 · len₂` columns and conv 3's `16 · len₃`, cut at the width
/// `backend::CONV_IM2COL_PANEL_BYTES` fits — 512, 704, 768, 832, 864 and 896 columns, so
/// 2+1, 3+1, 4+1, 5+2, 6+2 and 7+2. The figures are pinned, not derived: the rule that cuts
/// the panel is the backend's (`ConvColPanel::chunk_cols`), pinned device-free by
/// `backend::conv_route_decision_tests`' ladder, and this row is the product's own census
/// against it.
fn front_chunks(frames: usize) -> (u64, u64) {
    match frames {
        128 => (2, 1),
        256 => (3, 1),
        384 => (4, 1),
        512 => (5, 2),
        640 => (6, 2),
        768 => (7, 2),
        other => panic!("no pinned chunk census for {other} frames — the ladder grew"),
    }
}

/// The product's encoder at `frames`, one window, built and compiled in one call — so a build
/// is always followed by its own compile before the next build replaces the weight records
/// the build left for it.
fn front(pair: &Pair, frames: usize) -> Compiled {
    Compiled::new(graphs::asr_encoder(pair.sound.clone(), frames, 1).expect("the encoder builds"))
        .expect("the encoder compiles")
}

/// The steps of one kind in a compiled schedule, counted.
fn steps(compiled: &Compiled, kind: &str) -> usize {
    compiled
        .schedule_step_names()
        .expect("the wgpu backend exposes its schedule")
        .iter()
        .filter(|name| **name == kind)
        .count()
}

/// `(im2col_gpu, matmul, conv2d)` — the panel route's own census, in the order the arms print
/// it.
fn census(compiled: &Compiled) -> (usize, usize, usize) {
    (
        steps(compiled, "im2col_gpu"),
        steps(compiled, "matmul"),
        steps(compiled, "conv2d"),
    )
}

/// Passes of the probe per round, and the rounds each arm is timed over. The probe is one
/// conv — a few milliseconds — so a round of several passes is long enough to read and short
/// enough that a drift lands on both arms of a round rather than on the delta.
const PROBE_ITER: usize = 16;
const PROBE_ROUNDS: usize = 9;

/// The second of the front's three convs as a graph of its own: the one the reservation is
/// about, `[1, 480, 64, len₁]` through a `3×3` stride-2 pad-1 `480 → 480` conv, so its panel
/// is the graph's and nothing else runs. Its weight is the same `480·3·3` reduction the
/// front's is, and the ramp is finite, so the timing is that conv's gather and multiply.
fn conv2_probe(frames: usize) -> rlx_ir::Graph {
    let halve = |n: usize| (n - 1) / 2 + 1;
    let len1 = halve(frames);
    let mut g = rlx_ir::Graph::new("conv2_panel_probe");
    let x = g.input(
        "x",
        rlx_ir::Shape::new(&[1, 480, 64, len1], rlx_ir::DType::F32),
    );
    let w = g.param(
        "w",
        rlx_ir::Shape::new(&[480, 480, 3, 3], rlx_ir::DType::F32),
    );
    let y = g.conv2d(x, w, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    g.set_outputs(vec![y]);
    g
}

/// One probe arm: the conv compiled under `budget` (`Some("0")` = the whole panel).
fn probe_arm(frames: usize, ws: &[f32], budget: Option<&str>) -> WgpuExecutable {
    support::with_panel_budget(budget, || {
        let mut exe = WgpuExecutable::compile(conv2_probe(frames));
        exe.set_param("w", ws);
        exe
    })
}

/// The two probe arms timed alternately over [`PROBE_ROUNDS`] rounds of [`PROBE_ITER`] passes
/// each: `(whole, chunked)` per-pass milliseconds and the paired difference `whole − chunked`.
fn probe_rounds(
    whole: &mut WgpuExecutable,
    chunked: &mut WgpuExecutable,
    xs: &[f32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let round = |exe: &mut WgpuExecutable| {
        let start = std::time::Instant::now();
        for _ in 0..PROBE_ITER {
            exe.run(&[("x", xs)]);
        }
        start.elapsed().as_secs_f64() * 1e3 / PROBE_ITER as f64
    };
    support::paired_rounds_with(PROBE_ROUNDS, || round(whole), || round(chunked))
}

/// The budget of the amplifying arm: a quarter of the shipped one, so the panel is cut into
/// about four times as many chunks and one chunk's cost stands that much further above a
/// loaded machine's noise. What it measures is a *slope* — the difference against the
/// uncapped arm, divided by the chunks it added — and the shipped arm's cost is that slope at
/// the shipped chunk count.
const AMPLIFY_BUDGET: &str = "4194304";

/// The least sample: the pass that ran under the quietest moment, which is the statistic a
/// loaded machine leaves readable.
fn least(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "measurement: prices the chunked conv panel against the whole panel on the encoder"]
fn the_cut_prices_itself_on_the_encoder_front() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    mahbot_inference::device::pin_gpu();
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("skipped: {error:#} — the panel cut was not priced");
            return;
        }
    };
    for frames in FRAME_RUNGS {
        // Both arms of one rung held together: the rounds alternate between them, and each
        // build is compiled before the next build starts.
        let mut whole = support::with_panel_budget(Some("0"), || front(&pair, frames));
        let mut chunked = support::with_panel_budget(None, || front(&pair, frames));
        let scratch = |arm: &Compiled| {
            arm.arena_scratch_bytes()
                .expect("the wgpu backend reports the arena's scratch")
        };
        let (whole_scratch, chunked_scratch) = (scratch(&whole), scratch(&chunked));
        let arena = |arm: &Compiled| {
            arm.arena_bytes()
                .expect("the wgpu backend reports the arena's size")
        };
        let (whole_arena, chunked_arena) = (arena(&whole), arena(&chunked));
        let (whole_steps, chunked_steps) = (census(&whole), census(&chunked));
        let (conv2, _) = panel_columns(frames);
        let (conv2_chunks, conv3_chunks) = front_chunks(frames);
        // The booked tail *is* conv 2's chunk — it is the larger of the two convs' — so the
        // backend's own figure gives the chunk width, rather than a copy of the rule that
        // sized it.
        let chunk_columns = chunked_scratch / (CONV2_K * 4);
        println!(
            "frames={frames} panel={} B whole_scratch={whole_scratch} B \
             chunked_scratch={chunked_scratch} B whole_arena={whole_arena} B \
             chunked_arena={chunked_arena} B chunk_columns={chunk_columns} \
             chunks={conv2_chunks}+{conv3_chunks}",
            whole_panel_bytes(frames),
        );
        // The activation plan the split arena is built from — the ladder the
        // elementwise in-place rule is measured on.
        println!(
            "frames={frames} compact_plan={} B",
            rlx_wgpu::buffer::compact_plan(&chunked.runner.graph).arena_size,
        );
        println!(
            "frames={frames} whole_census=im2col_gpu:{} matmul:{} conv2d:{} \
             chunked_census=im2col_gpu:{} matmul:{} conv2d:{}",
            whole_steps.0,
            whole_steps.1,
            whole_steps.2,
            chunked_steps.0,
            chunked_steps.1,
            chunked_steps.2,
        );

        // The memory half: the uncapped arm books the whole panel the arithmetic gives, the
        // chunked arm at most the budget and less than the panel it cut.
        assert_eq!(
            whole_scratch,
            whole_panel_bytes(frames),
            "frames={frames}: the uncapped arm books the whole panel",
        );
        assert!(
            chunked_scratch <= BUDGET,
            "frames={frames}: the chunked arm books at most the {BUDGET} B budget, \
             {chunked_scratch} B",
        );
        assert!(
            chunked_scratch < whole_scratch,
            "frames={frames}: and less than the {whole_scratch} B panel it cut, \
             {chunked_scratch} B",
        );

        // The schedule half: both convs keep the panel route, conv 1 keeps the direct kernel,
        // and the arms differ by exactly the chunks — one gather and one multiply each.
        let extra = (conv2_chunks + conv3_chunks - 2) as usize;
        assert_eq!(
            (whole_steps.0, whole_steps.2),
            (2, 1),
            "frames={frames}: two whole panels gathered and conv 1 on the direct kernel",
        );
        assert_eq!(
            (
                chunked_steps.2,
                chunked_steps.0 - whole_steps.0,
                chunked_steps.1 - whole_steps.1
            ),
            (1, extra, extra),
            "frames={frames}: the chunked arm's extra gathers are the chunks the cut added, \
             each consumed by its own multiply and nothing else moved",
        );

        // The exactness half: one fixed input through both arms, compared exactly. The ramp is
        // the harness's, so a degenerate draw cannot make two different arithmetic agree.
        let input = support::ramped_inputs(&whole.runner.graph);
        let whole_out = whole.run(&input);
        let chunked_out = chunked.run(&input);
        let (differing, worst) = support::output_difference(&whole_out, &chunked_out);
        println!("frames={frames} exactness differing={differing} max|Δ|={worst:e}");
        assert_eq!(
            differing, 0,
            "frames={frames}: the two arms' outputs are bit-identical — the cut is over the \
             spatial axis, which carries no accumulation",
        );
        assert_eq!(
            worst, 0.0,
            "frames={frames}: and the largest difference is zero"
        );

        // The time: `WARMUP` passes per arm, then the paired, alternating rounds — the helper
        // discards one more pass per arm of its own.
        for _ in 0..WARMUP {
            support::frame_once(&mut whole, &input);
            support::frame_once(&mut chunked, &input);
        }
        let (whole_ms, chunked_ms, deltas) =
            support::paired_rounds(ROUNDS, &mut whole, &input, &mut chunked, &input);
        let (whole_median, chunked_median) =
            (support::median(&whole_ms), support::median(&chunked_ms));
        println!(
            "frames={frames} whole_ms={whole_median:.3} chunked_ms={chunked_median:.3} \
             ratio={:.4} delta={:+.2}% median_paired_delta={:+.3} ms chunked_won={}/{ROUNDS}",
            chunked_median / whole_median,
            (chunked_median - whole_median) / whole_median * 100.0,
            support::median(&deltas),
            deltas.iter().filter(|delta| **delta > 0.0).count(),
        );
        assert!(
            whole_median.is_finite() && whole_median > 0.0,
            "frames={frames}: the whole arm's median is a reading, {whole_median}",
        );
        assert!(
            chunked_median.is_finite() && chunked_median > 0.0,
            "frames={frames}: and so is the chunked arm's, {chunked_median}",
        );

        // ── the changed work, priced on its own ─────────────────────────────
        // The front's whole pair above is what the row costs; the *difference* the cut makes
        // is conv 2's, and one conv is a few milliseconds against the front's hundreds, so
        // pricing it alone is what resolves the effect under a loaded machine. The two arms
        // of the probe are the same conv at the same weights, and their outputs are held to
        // bit-identity before the timing, so a reading is of the same arithmetic.
        let ws: Vec<f32> = (0..480 * 480 * 3 * 3)
            .map(|i| ((i as f32) * 0.0007 - 0.31).sin())
            .collect();
        let mut probe_whole = probe_arm(frames, &ws, Some("0"));
        let mut probe_chunked = probe_arm(frames, &ws, None);
        let (probe_whole_scratch, probe_chunked_scratch) = (
            probe_whole.test_scratch_bytes(),
            probe_chunked.test_scratch_bytes(),
        );
        let halve = |n: usize| (n - 1) / 2 + 1;
        let xs: Vec<f32> = (0..480 * 64 * halve(frames))
            .map(|i| ((i as f32) * 0.0007 - 0.31).sin())
            .collect();
        let probe_out_whole = probe_whole.run(&[("x", &xs)]).remove(0);
        let probe_out_chunked = probe_chunked.run(&[("x", &xs)]).remove(0);
        let (differing, worst) = support::output_difference(
            std::slice::from_ref(&probe_out_whole),
            std::slice::from_ref(&probe_out_chunked),
        );
        assert_eq!(
            (differing, worst),
            (0, 0.0),
            "frames={frames}: the probe's two arms are bit-identical too",
        );
        assert_eq!(
            (probe_whole_scratch, probe_chunked_scratch),
            (whole_panel_bytes(frames), chunked_scratch),
            "frames={frames}: and they book the same two tails the front's arms do",
        );
        let (pw, pc, pd) = probe_rounds(&mut probe_whole, &mut probe_chunked, &xs);
        let (pw_med, pc_med) = (support::median(&pw), support::median(&pc));
        let paired = support::median(&pd);
        // The amplifying pair: the same conv against a quarter of the budget, so the slope of
        // the cost with the chunk count is read where it stands four times further above the
        // machine's noise. The shipped arm's cost is that slope at the chunks it has.
        let mut amplify = probe_arm(frames, &ws, Some(AMPLIFY_BUDGET));
        let (aw, _, ad) = probe_rounds(&mut probe_whole, &mut amplify, &xs);
        let amp_columns = amplify.test_scratch_bytes() / (CONV2_K * 4);
        let amp_chunks = conv2.div_ceil(amp_columns);
        let per_chunk = if amp_chunks > 1 {
            support::median(&ad) / (amp_chunks - 1) as f64
        } else {
            0.0
        };
        let implied = per_chunk * (conv2_chunks - 1) as f64;
        println!(
            "frames={frames} probe_whole_ms={pw_med:.4} (least {:.4}) probe_chunked_ms={pc_med:.4} \
             (least {:.4}) probe_paired_delta={paired:+.4} ms = {:+.3}% of the conv, \
             {:+.3}% of the front's {whole_median:.3} ms; amplified arm {amp_chunks} chunks \
             reads {per_chunk:+.4} ms a chunk (whole {:.4} ms), so the shipped {} chunks imply \
             {implied:+.4} ms = {:+.3}% of the front",
            least(&pw),
            least(&pc),
            paired / pw_med * 100.0,
            paired / whole_median * 100.0,
            support::median(&aw),
            conv2_chunks,
            implied / whole_median * 100.0,
        );
        assert!(
            pw_med.is_finite() && pw_med > 0.0 && pc_med.is_finite() && pc_med > 0.0,
            "frames={frames}: both probe medians are readings",
        );
    }
}

/// The wake path's own graph, held to the same exactness: its front runs at the pair's two
/// windows, which is the batch the resting columns of those rows hold, and the cut splits that
/// panel into two even chunks (`11 059 200 B` against `21 012 480 B` whole). Both arms are
/// compiled from the same graph and weights and run on one fixed input, and their outputs must
/// be bit-identical — the whole of what "every number the product produces is unchanged" means
/// at this graph.
#[test]
#[ignore = "measurement: the wake pair's front, chunked against whole, bit-identical"]
fn the_wake_pairs_front_is_bit_identical_across_the_cut() {
    use mahbot_inference::listen::{MEL_FRAMES, PAIR};

    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    mahbot_inference::device::pin_gpu();
    let pair = match Pair::of_role(Role::Recognition) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("skipped: {error:#} — the wake pair's front was not checked");
            return;
        }
    };
    let front = |budget: Option<&str>| {
        support::with_panel_budget(budget, || {
            Compiled::new(
                graphs::asr_encoder(pair.sound.clone(), MEL_FRAMES, PAIR)
                    .expect("the pair's front builds"),
            )
            .expect("the pair's front compiles")
        })
    };
    let mut whole = front(Some("0"));
    let mut chunked = front(None);
    let (whole_scratch, chunked_scratch) = (
        whole.arena_scratch_bytes().expect("the arena's scratch"),
        chunked.arena_scratch_bytes().expect("the arena's scratch"),
    );
    println!(
        "wake pair: whole_scratch={whole_scratch} B chunked_scratch={chunked_scratch} B \
         whole_census={} chunked_census={}",
        support::schedule_census(&whole),
        support::schedule_census(&chunked),
    );
    assert_eq!(
        (whole_scratch, chunked_scratch),
        (21_012_480, 11_059_200),
        "the pair's panel is cut, and to the figure the ladder's first rung books"
    );
    let input = support::ramped_inputs(&whole.runner.graph);
    let (differing, worst) = support::output_difference(&whole.run(&input), &chunked.run(&input));
    assert_eq!(
        (differing, worst),
        (0, 0.0),
        "the pair's front produces the same bytes on both arms"
    );
}
