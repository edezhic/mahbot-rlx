// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a hosted audio step may sit on this route. `welch_peaks_host`,
//! `log_mel_host` and `log_mel_backward_host` are the steps the plan *defers*
//! (`step_is_tail_host`): the pass loop walks past them and the run performs them
//! all in `run_tail_host_audio_ops`, which the run reaches only at the end of the
//! schedule, from the device batch that got there. They are not
//! `step_runs_on_host` steps, so the two predicates are not a partition — and the
//! single arrangement they are served in is: **the hosted audio steps in a row as the
//! schedule's last steps, with the device work that feeds them in the same batch.**
//! Every other placement is refused at plan time, and what only a run-time answer can
//! still reach — the crate's own route switches, or a joinable predecessor whose route
//! declines the join — stops loudly in the walk instead of spinning.
//!
//! The arrangements are pinned by what they do rather than by an argument, and the
//! preserved ones are run in their own process and waited for with a deadline
//! ([`common::rerun_self`]), so a regression that makes a run unable to advance
//! reports the child's stderr instead of hanging the gate — an unbounded wait here
//! hung `run_report`'s own tail-host case, which is why that case's wait is bounded
//! too.
//!
//! The reference the preserved fixtures are compared against is the one the route
//! itself calls: `rlx_gpu_host::run_log_mel` reaches
//! `rlx_cpu::thunk::execute_log_mel_f32`, which is `rlx_ir::audio::log_mel_block_f32`
//! and nothing else, and the backward arm is `log_mel_block_vjp` the same way. The step
//! in front of the mel is exact in f32 — a multiply by two, or the row copy's own write —
//! so the comparison is equality up to the host path's own arithmetic and the bar is
//! `1e-6`.
//!
//! Both kinds the rule admits in front of the tail are pinned as graphs here: the packed
//! multiply (`packed_then_mel`, whose join no lever declines) and the buffer copy
//! (`copy_then_mel`, whose join `RLX_WGPU_HOST_BUFFER_COPY=1` declines — the same fixture
//! is therefore both a preserved arrangement and the one run-time state the plan cannot
//! settle).
//!
//! What each arrangement did before the guard — and every reading this ruling rests on —
//! is recorded on the guard (`src/backend/step.rs::assert_hosted_audio_step_placement`).

mod common;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// The child re-exec marker: its value is the fixture to run.
const MARKER: &str = "HOSTED_AUDIO_CHILD";

/// The one test fn a child is re-run with (`common::rerun_self` builds the child's own
/// arguments from it).
const FILTER: &str = "hosted_audio_steps_are_walked_only_where_the_route_can_reach_them";

/// The fixture names the one test fn dispatches on.
const DEVICE_THEN_MEL: &str = "device_then_mel";
const MEL_THEN_BACKWARD: &str = "mel_then_backward";
const PACKED_THEN_MEL: &str = "packed_then_mel";
const COPY_THEN_MEL: &str = "copy_then_mel";

/// Log-mel geometry: `n_fft = 32` is a whole number of the FFT block's two halves
/// and `n_bins = n_fft/2 + 1` is what `Op::LogMel`'s shape rule asks for.
const N_FFT: usize = 32;
const N_BINS: usize = N_FFT / 2 + 1;
const N_MELS: usize = 8;

/// Frames in the spectrum the device step produces: `[FRAMES, 2*n_fft]`.
const FRAMES: usize = 3;

/// The packed multiply's contraction: one 256-element packing block of Q4_K, the
/// whole number of blocks the fused arms answer over — the contraction
/// `run_report`'s one-row packed fixture pins as joined.
const PACKED_K: usize = 256;

/// The mel filterbank, at the ramp `run_report`'s tail-host fixture fills it with.
fn filters() -> Vec<f32> {
    common::ramp(N_MELS * N_BINS, 1, 0.007)
}

/// The `[FRAMES, 2*n_fft]` spectrum block layout the device step produces. Its
/// magnitude is the caller's: a mel parity bar has to be read against it.
fn spectrum() -> Vec<f32> {
    common::ramp(FRAMES * 2 * N_FFT, 1, 0.013)
}

/// The spectrum the multiply in front of the mel produces: the input doubled.
/// Exact in f32, so the mel comparison is a comparison of the log-mel arithmetic
/// alone.
fn doubled(spec: &[f32]) -> Vec<f32> {
    spec.iter().map(|v| v * 2.0).collect()
}

/// The preserved arrangement, in its smallest form: one device step, one deferred
/// mel over its output, the mel the graph's own output.
///
/// The multiply is the one device step because a device step is the only thing that
/// can put device work in front of the mel — and the operands matter: it is a
/// multiply by a `2.0` constant and not an `add(spec, 0.0)`, whose identity the
/// front end folds away, leaving the mel alone in the schedule and this fixture on
/// the `mel_only` arrangement instead.
fn device_then_mel_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_device_then_mel");
    let spec = g.input("spec", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let two = g.constant(2.0, DType::F32);
    let doubled = g.mul(spec, two);
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(doubled, filters);
    g.set_outputs(vec![mel]);
    g
}

/// The preserved arrangement with the audio family's whole tail: the mel and its
/// backward, both deferred, both after the device step. Two family steps in a row
/// is the shape the deferral has to serve as a run, not as one step.
fn mel_then_backward_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_mel_then_backward");
    let spec = g.input("spec", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let two = g.constant(2.0, DType::F32);
    let doubled = g.mul(spec, two);
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(doubled, filters);
    let dspec = g.log_mel_backward(doubled, filters, mel);
    g.set_outputs(vec![mel, dspec]);
    g
}

/// The preserved arrangement with a hosted step the route *joins* into the device
/// submission in front of the deferred mel: a one-row Q4_K packed multiply, whose
/// contraction is a whole number of the scheme's own packing blocks. Whether the
/// join really happens is a run-time route answer — this fixture's own job is that
/// the arrangement still runs, so a route change that declines the join shows up as
/// the loud stop rather than as a hang.
fn packed_then_mel_graph() -> (Graph, Vec<u8>) {
    let bytes = common::packed_gguf_weight(QuantScheme::GgufQ4K, PACKED_K, 2 * N_FFT);
    let mut g = Graph::new("wgpu_hosted_audio_packed_then_mel");
    let x = g.input("x", Shape::new(&[1, PACKED_K], DType::F32));
    let w = g.param("w", Shape::new(&[bytes.len()], DType::U8));
    let spec = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x, w],
        Shape::new(&[1, 2 * N_FFT], DType::F32),
    );
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(spec, filters);
    g.set_outputs(vec![mel]);
    (g, bytes)
}

/// The preserved arrangement whose joined predecessor is the *other* kind the rule
/// admits: a `KvAppend` lowers to `Step::BufferCopy`, which this route joins into the
/// device submission when its route answers the device one, with the deferred mel over
/// the slot it writes.
///
/// `axis = 0` over a `[FRAMES, 2*n_fft]` cache has nothing in front of the axis, so the
/// append is a contiguous row write and this backend keeps the `KvAppend` instead of
/// lowering it away; the append aliases the cache's own slot (`rlx-compile`'s
/// `pure_view_offset`), which is the slot the mel reads.
fn copy_then_mel_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_copy_then_mel");
    let cache = g.input("cache", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let row = g.input("row", Shape::new(&[1, 2 * N_FFT], DType::F32));
    let appended = g.add_node(
        Op::KvAppend {
            axis: 0,
            pos: FRAMES - 1,
        },
        vec![cache, row],
        Shape::new(&[FRAMES, 2 * N_FFT], DType::F32),
    );
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(appended, filters);
    g.set_outputs(vec![mel]);
    g
}

/// The row the copy-join fixture appends: the cache's own last frame, doubled. The mel is
/// then read off a spectrum whose last frame the copy wrote, so a run that dropped the
/// write fails the parity bar instead of passing on the cache's own values.
fn appended_row(spec: &[f32]) -> Vec<f32> {
    spec[(FRAMES - 1) * 2 * N_FFT..]
        .iter()
        .map(|v| v * 2.0)
        .collect()
}

/// The refused arrangement with no device work anywhere: the mel is the schedule.
fn mel_only_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_mel_only");
    let spec = g.input("spec", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(spec, filters);
    g.set_outputs(vec![mel]);
    g
}

/// The refused arrangement with a device step *after* the deferred one.
fn mel_then_device_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_mel_then_device");
    let spec = g.input("spec", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let two = g.constant(2.0, DType::F32);
    let doubled = g.mul(spec, two);
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(doubled, filters);
    let out = g.mul(mel, two);
    g.set_outputs(vec![out]);
    g
}

/// The refused arrangement whose consumer is an ordinary host step: `ArgMax` lowers
/// to `Step::ArgReduceHost` unconditionally, so the argmax reads the mel's slot
/// while the step that would have written it is still deferred. The mel is an output
/// as well, so the slot the argmax reads is the one this fixture reads back.
fn mel_then_argmax_graph() -> Graph {
    let mut g = Graph::new("wgpu_hosted_audio_mel_then_argmax");
    let spec = g.input("spec", Shape::new(&[FRAMES, 2 * N_FFT], DType::F32));
    let two = g.constant(2.0, DType::F32);
    let doubled = g.mul(spec, two);
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(doubled, filters);
    let idx = g.argmax(mel, 1, false, Shape::new(&[FRAMES], DType::F32));
    g.set_outputs(vec![mel, idx]);
    g
}

/// Assert that `g` is refused at plan time with the placement marker, naming every one
/// of `names`. `WgpuExecutable::compile` returns no `Result`, so the refusal arrives as
/// a panic — the loud form of it, before anything executes, and the reason the catch is
/// `catch_unwind` rather than a `Result` match (`WgpuExecutable` carries no `Debug`).
fn assert_refused(g: Graph, names: &[&str]) {
    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(g)));
    let Err(payload) = caught else {
        panic!("this placement was accepted instead of refused at plan time");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.starts_with("rlx-wgpu: REFUSED (hosted audio step placement)"),
        "the refusal must carry its stable marker, got: {msg}"
    );
    for name in names {
        assert!(
            msg.contains(name),
            "the refusal must name `{name}`, got: {msg}"
        );
    }
}

/// The preserved fixture's own check: the mel the run reads back is the host
/// log-mel of the doubled spectrum.
///
/// The mel's slot is also what a stranded run reads as zeros, which this comparison
/// cannot mistake for a mel: the log-mel of this spectrum is finite and not all
/// zeros, so the unwritten slot fails the bar by construction.
fn run_device_then_mel() {
    let mut exe = WgpuExecutable::compile(device_then_mel_graph());
    exe.set_param("filters", &filters());
    let got = exe.run(&[("spec", &spectrum())]).remove(0);
    let mut want = vec![0f32; FRAMES * N_MELS];
    rlx_ir::audio::log_mel_block_f32(
        &doubled(&spectrum()),
        &filters(),
        FRAMES,
        N_FFT,
        N_BINS,
        N_MELS,
        &mut want,
    );
    let diff = common::max_abs_diff(&got, &want);
    assert!(
        diff <= 1e-6,
        "the deferred mel is not the host log-mel of its spectrum: max|delta|={diff:e}"
    );
}

/// The two-step tail's own check: the mel, and the backward against the vjp of that
/// same mel — the tensor the graph hands `Op::LogMelBackward` as `dy`.
fn run_mel_then_backward() {
    let mut exe = WgpuExecutable::compile(mel_then_backward_graph());
    exe.set_param("filters", &filters());
    let outs = exe.run(&[("spec", &spectrum())]);
    let spec = doubled(&spectrum());
    let mut want_mel = vec![0f32; FRAMES * N_MELS];
    rlx_ir::audio::log_mel_block_f32(
        &spec,
        &filters(),
        FRAMES,
        N_FFT,
        N_BINS,
        N_MELS,
        &mut want_mel,
    );
    let mut want_dspec = vec![0f32; FRAMES * 2 * N_FFT];
    rlx_ir::audio::log_mel_block_vjp(
        &spec,
        &filters(),
        &want_mel,
        FRAMES,
        N_FFT,
        N_BINS,
        N_MELS,
        &mut want_dspec,
    );
    let mel_diff = common::max_abs_diff(&outs[0], &want_mel);
    let dspec_diff = common::max_abs_diff(&outs[1], &want_dspec);
    assert!(
        mel_diff <= 1e-6 && dspec_diff <= 1e-6,
        "the audio family's tail is not the host pair it defers to: \
         max|delta| mel={mel_diff:e} dspec={dspec_diff:e}"
    );
}

/// The joined-fixture's own check: the run returns a mel that is finite and neither all
/// zeros nor constant.
///
/// The bar is degeneracy, not the parity the other two fixtures hold to: this fixture's
/// weight is [`common::packed_gguf_weight`]'s hand-laid Q4_K bytes, which have no f32 form
/// to compare against — a parity bar would first have to decode them on the host, a second
/// dequant to keep in step with the fused arm. What this arrangement has to show is that
/// the run returns and its deferred step produced something rather than a slot nothing
/// wrote, and a zero or constant mel is exactly what a stranded slot reads as.
fn run_packed_then_mel() {
    let (g, bytes) = packed_then_mel_graph();
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param_bytes("w", &bytes);
    exe.set_param("filters", &filters());
    let got = exe
        .run(&[("x", &common::ramp(PACKED_K, 1, 0.023))])
        .remove(0);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the joined fixture's mel is not finite: {got:?}"
    );
    assert!(
        got.iter().any(|v| *v != 0.0),
        "the joined fixture's mel is all zeros, which a stranded slot also reads: {got:?}"
    );
    assert!(
        got.iter().any(|v| *v != got[0]),
        "the joined fixture's mel is constant, which a stranded slot also reads: {got:?}"
    );
}

/// The copy-join fixture's own check: the mel is the host log-mel of the cache with the
/// appended row written into its last frame.
fn run_copy_then_mel() {
    let mut exe = WgpuExecutable::compile(copy_then_mel_graph());
    exe.set_param("filters", &filters());
    let spec = spectrum();
    let row = appended_row(&spec);
    let got = exe.run(&[("cache", &spec), ("row", &row)]).remove(0);
    let mut written = spec.clone();
    written[(FRAMES - 1) * 2 * N_FFT..].copy_from_slice(&row);
    let mut want = vec![0f32; FRAMES * N_MELS];
    rlx_ir::audio::log_mel_block_f32(
        &written,
        &filters(),
        FRAMES,
        N_FFT,
        N_BINS,
        N_MELS,
        &mut want,
    );
    let diff = common::max_abs_diff(&got, &want);
    assert!(
        diff <= 1e-6,
        "the mel over the copied row is not the host log-mel of the spectrum it wrote: \
         max|delta|={diff:e}"
    );
}

/// The recorded census of the preserved arrangement, as the run prints it: one
/// device step, no step the walk takes on the host as it goes, one step it defers,
/// and one uniform and one bind group for the device step's single dispatch.
///
/// The line's trailing `final_gpu_ui` is the uniform pre-pass's own figure and is
/// deliberately not pinned — it counts the pre-pass's cursor, which a later change
/// to how the pre-pass walks a deferred step is free to move without moving any
/// arrangement this target pins.
const COUNTER_PREFIX: &str = "[wgpu-counters] steps=2 host=0 tail_host=1 uniforms=1 bind_groups=1";

/// The one test fn, four child fixtures: which placements of a deferred audio step
/// this route serves, and what it does with the ones it does not.
#[test]
fn hosted_audio_steps_are_walked_only_where_the_route_can_reach_them() {
    if let Ok(fixture) = std::env::var(MARKER) {
        // The sentinel `common::rerun_self` waits for: it proves the child ran this test,
        // not just that the filter matched something.
        eprintln!("{MARKER} {fixture}");
        match fixture.as_str() {
            DEVICE_THEN_MEL => run_device_then_mel(),
            MEL_THEN_BACKWARD => run_mel_then_backward(),
            PACKED_THEN_MEL => run_packed_then_mel(),
            COPY_THEN_MEL => run_copy_then_mel(),
            other => panic!("`{other}` is not a fixture this test runs"),
        }
        return;
    }
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    // The premise, read off the plan: each preserved fixture is one step in front of the
    // audio family's own tail, in that order, and the family's steps are the ones the run
    // defers rather than walks (`step_is_tail_host`; `Op::LogMel` lowers to
    // `Step::LogMelHost` unconditionally). What the *predecessor* is is not a
    // classification the plan settles for the packed fixture: `dequant_matmul_gguf` is
    // host-classified, and whether the walk carries a batch over it is the run-time route
    // answer the guard admits here and the walk stops on if this route declines it.
    assert_eq!(
        WgpuExecutable::compile(device_then_mel_graph()).test_step_names(),
        ["binary", "log_mel_host"],
        "the device-then-mel fixture must be one device step and one deferred mel"
    );
    assert_eq!(
        WgpuExecutable::compile(mel_then_backward_graph()).test_step_names(),
        ["binary", "log_mel_host", "log_mel_backward_host"],
        "the mel-then-backward fixture must be the audio family's whole tail"
    );
    let (packed, _) = packed_then_mel_graph();
    assert_eq!(
        WgpuExecutable::compile(packed).test_step_names(),
        ["dequant_matmul_gguf", "log_mel_host"],
        "the packed fixture must be one packed multiply and one deferred mel"
    );
    assert_eq!(
        WgpuExecutable::compile(copy_then_mel_graph()).test_step_names(),
        ["buffer_copy", "log_mel_host"],
        "the copy-join fixture must be one buffer copy and one deferred mel"
    );

    // The refused placements, and the names their refusals must carry — a refusal has
    // to be about the arrangement it names, and these are the ones it names.
    // `mel_only` has no device step anywhere, so its deferred mel is not reached from
    // a batch; `mel_then_device` puts a `binary` after the mel; `mel_then_argmax`'s
    // consumer is the unconditional `Step::ArgReduceHost`.
    assert_refused(
        mel_only_graph(),
        &["log_mel_host", "it is the first step of the schedule"],
    );
    // The step after the deferred mel: before the guard this arrangement panicked on
    // the later device step's own uniform write, `index out of bounds: the len is 2 but
    // the index is 2`.
    assert_refused(mel_then_device_graph(), &["binary", "log_mel_host"]);
    // The host consumer after the deferred mel: before the guard it read the mel's
    // never-written slot as 24 zeros against a finite host log-mel, argmax `[0,0,0]`
    // against `[7,7,7]`.
    assert_refused(mel_then_argmax_graph(), &["argreduce_host", "log_mel_host"]);

    // The arrangements this route serves: each runs, and each checks its own output
    // against the host function the route itself calls. A run that cannot advance
    // fails here with the child's stderr rather than spinning.
    for fixture in [
        DEVICE_THEN_MEL,
        MEL_THEN_BACKWARD,
        PACKED_THEN_MEL,
        COPY_THEN_MEL,
    ] {
        let (ok, stderr) =
            common::rerun_self(FILTER, MARKER, fixture, &common::CLEARED_SWITCHES, &[]);
        assert!(ok, "the `{fixture}` child failed:\n{stderr}");
    }

    // The states the plan cannot settle: the crate's own debugging switches force one
    // operation per submission and carry no batch to the deferred tail either way, so
    // the walk reaches the hosted step with nothing in hand to advance it. Both used to
    // hang; both must now stop loudly. The deadline inside `rerun_self` is the bound
    // that turns a regression here into a failure rather than a hung gate.
    for switch in ["RLX_WGPU_ONE_OP_PER_PASS", "RLX_WGPU_SNAPSHOT"] {
        let (ok, stderr) = common::rerun_self(
            FILTER,
            MARKER,
            DEVICE_THEN_MEL,
            &common::CLEARED_SWITCHES,
            &[(switch, "1")],
        );
        assert!(
            !ok && stderr.contains("rlx-wgpu: STOPPED (hosted audio step stranded)"),
            "`{switch}` over the preserved arrangement must stop loudly instead of \
             hanging: ok={ok}\n{stderr}"
        );
    }

    // The other half of the same state, through a lever that reaches it without a
    // switch of the walk's own: a copy's route is a run-time answer, so forcing it onto
    // the host route takes the batch away from the deferred mel behind it. The plan
    // admits this arrangement (the kind may join), so what stops it is the walk's own
    // assert — the residual that reads as a refusal's twin.
    let (ok, stderr) = common::rerun_self(
        FILTER,
        MARKER,
        COPY_THEN_MEL,
        &common::CLEARED_SWITCHES,
        &[("RLX_WGPU_HOST_BUFFER_COPY", "1")],
    );
    assert!(
        !ok && stderr.contains("rlx-wgpu: STOPPED (hosted audio step stranded)"),
        "a copy whose route declines the join must leave the mel behind it stopped \
         loudly, not spinning or reading an unwritten slot: ok={ok}\n{stderr}"
    );

    // The census the run records for the preserved arrangement must not move: the
    // route's own report counts the deferred step on `tail_host` and on no other
    // line, which is the same claim `run_report` makes of its tail-host case.
    let (ok, stderr) = common::rerun_self(
        FILTER,
        MARKER,
        DEVICE_THEN_MEL,
        &common::CLEARED_SWITCHES,
        &[(common::RUN_REPORT_SWITCH, "1")],
    );
    assert!(ok, "the `{DEVICE_THEN_MEL}` census child failed:\n{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[wgpu-counters]"))
        .unwrap_or_else(|| panic!("no counters line in the child's stderr:\n{stderr}"));
    assert!(
        line.starts_with(COUNTER_PREFIX),
        "the recorded census of the preserved arrangement moved: `{line}`"
    );
}
