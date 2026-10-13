// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The printed half of `RLX_WGPU_DBG_COUNTERS`: what a run's report says about
//! itself, read as text. Every assertion here is about **structure** — which line
//! kinds appear, which labels they carry, and that each line parses into the
//! fields its kind is defined to have. **No time is asserted anywhere**: a figure
//! is checked for its shape (six decimals, `unavailable` where the device view
//! has no sample) and against the identities the report states about itself, never
//! against a value.
//!
//! Printed text is asserted the way this tree does it: the test re-runs itself in a
//! child process (`crates/backends/rlx-cpu/tests/pool_width_claimed.rs`'s
//! `the_refusal_is_reported_on_stderr` is the idiom) and reads the child's stderr.
//! The ten cases differ by environment — the switches are read by the run path,
//! and the environment is a fact of the run rather than of the test's own process
//! — so each case gets its own child and the parser sees one run's report and
//! nothing else: the region mix in full; the same graph under
//! `RLX_BENCH_DISPATCH_ONLY`, which must not claim a read-back it never did and
//! must still report the schedule as walked to its end; the switch off, which must
//! print no report line at all; a snapshot call, which is a replay session end to
//! end and reports nothing at all; a graph whose host step is one the run *defers*
//! to the end of the schedule, which the report must count on the `host` line that
//! ran it and on no other line; a dynamic-shape graph the run resolves itself,
//! whose compile is an interval of the bracket rather than an omission; and that
//! same graph under `RLX_BENCH_DISPATCH_ONLY`, whose schedule is device work
//! throughout, so its run leaves from inside the one batch that did it.
//!
//! The graph is the mix the report's two halves need: one device step and one host
//! step. `KvAppend` lowers to `Step::BufferCopy`, forced onto the host route by
//! `RLX_WGPU_HOST_BUFFER_COPY=1`, and the row it writes is a `MatMul`'s output —
//! a device step on every adapter, since `RLX_WGPU_HOST_MATMUL` is opt-in. The
//! premise is asserted before any child runs, so a graph that stopped having the
//! mix fails loudly instead of asserting about a schedule that never had one.
//!
//! The deferred-tail case's graph is the other kind of mix: a `MatMul` and an
//! `Op::LogMel`, which lowers to `Step::LogMelHost` — a step `step_is_tail_host`
//! names, so the run does not execute it where the schedule puts it but at the
//! end, alone under `host`. Its premise is asserted the same way.
//!
//! The device half of the report — `device`, `device_pass`, `device_unusable` —
//! exists only where a wgpu device does and can carry a figure, which is why this
//! target rides `backend-tests`. It is also the half that says whether a figure
//! could be obtained at all: `view=unavailable` is pinned together with
//! `measured=0` and `device_ms=unavailable`, and a lost sample is never a zero.
//!
//! The packed-multiply cases — 8 to 10 — are the ones the device view's coverage
//! is about. Their graph is one packed weight multiply and the dense multiply over
//! its rows, run at three shapes because the row count and the scheme together
//! decide the multiply's route: a K-quant multiply at one row is the one-row tiled
//! arm the run *joins* into its batch, and the same scheme at four rows is the
//! row-grouped GEMM, which the run joins the same way — both multiply passes are
//! `encode` passes beside the batch's own. An 8-bit (Q8_0) multiply at four rows is
//! not joined: it takes its group's own scratch entry point, so its pass is that
//! call's own submission and lands in `host` — the interval the run's own dispatch
//! loop never opens a pass in. All three cases assert the same thing about the
//! view: the packed multiply's pass is recorded, `passes` counts it, and the
//! interval the report names for it is the one that carried the multiply. A disabled
//! report cannot print, so the same fixture under the switch off would only restate
//! case 3, and the crate's other packed-multiply targets run these same routes with
//! the switch off — the marks' ordinary path needs no case of its own here.

use std::collections::{BTreeMap, BTreeSet};

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

/// Printed prefix of every report line. The parser keys on it.
const PREFIX: &str = "[wgpu-run-report]";

/// Set on the child re-exec; the test fn takes its child path when it is, and its
/// value is the fixture that child runs.
const MARKER: &str = "RUN_REPORT_CHILD";

/// The fixtures a child can be asked for: [`region_mix_graph`], the mix the
/// report's two halves need; [`tail_host_graph`], whose one host step is a step the
/// run defers to the end of the schedule; [`dynamic_graph`], which the run resolves
/// — and so compiles — inside the reported bracket; and [`packed_graph`] at each of
/// [`PACKED_ROWS`]'s rows and schemes, the only fixtures whose schedule holds a
/// packed multiply.
const REGION_MIX: &str = "region_mix";
const TAIL_HOST: &str = "tail_host";
const DYNAMIC: &str = "dynamic";
const PACKED_JOIN: &str = "packed_join";
const PACKED_JOIN_M4: &str = "packed_join_m4";
const PACKED_UNJOINED: &str = "packed_unjoined";

/// The packed-multiply fixtures: each row count with its fixture name, its scheme
/// and whether its route is one the run *joins* into its batch — the one place all
/// four are written down, so the cases and the premise assertions below cannot
/// drift apart. Two K-quant cases ride the join at one and four rows; the 8-bit
/// scheme's four-row call keeps its scratch entry point and its own submission,
/// which is why one case of each kind is here.
const PACKED_ROWS: [(usize, &str, QuantScheme, bool); 3] = [
    (1, PACKED_JOIN, QuantScheme::GgufQ4K, true),
    (4, PACKED_JOIN_M4, QuantScheme::GgufQ4K, true),
    (4, PACKED_UNJOINED, QuantScheme::GgufQ8_0, false),
];

/// The row count and scheme a packed-multiply fixture runs at, or `None` for
/// another fixture.
fn packed_rows(fixture: &str) -> Option<(usize, QuantScheme)> {
    PACKED_ROWS
        .iter()
        .find(|(_, name, _, _)| *name == fixture)
        .map(|(rows, _, scheme, _)| (*rows, *scheme))
}

/// The seven phase tokens an interval, an omission or a pass may name.
const PHASES: [&str; 7] = [
    "stage", "compile", "encode", "wait", "readback", "host", "probe",
];

/// Every line kind one report prints.
const KINDS: [&str; 7] = [
    "run",
    "interval",
    "omitted",
    "device",
    "device_unusable",
    "device_pass",
    "extra_work",
];

/// The five intervals every run has, each with the reason printed when it never
/// happened — `run_report.rs`'s own `REQUIRED`, as the fixed token pairs a caller
/// can key on.
const REQUIRED: [(&str, &str); 5] = [
    ("stage", "no_staging_edge"),
    ("encode", "no_device_work_in_this_schedule"),
    ("wait", "no_fence_edge_in_this_path"),
    ("readback", "no_readback_in_this_path"),
    ("host", "no_host_step_in_this_schedule"),
];

/// The one label these fixtures have no edge for: the in-call compile belongs to
/// a graph the run resolves itself, and every fixture but the dynamic one is
/// compiled by `Session::compile` before the bracket opens. A label without an
/// edge is omitted *and said so*, so the omission is asserted where it must
/// happen and the interval where it must not.
const COMPILE: (&str, &str) = ("compile", "no_deferred_compile_in_this_path");

/// The switches that would move this graph's routing — which of its two steps runs
/// where — or that add work the report itself names. Cleared on every child before
/// the case's own switches are applied: what a case reads is the case's
/// environment, never the operator's. (`buffer_copy_batch_join.rs` guards the same
/// set with an assertion; here the child is re-exec'd, so clearing is what that
/// list buys.)
const CLEARED: [&str; 15] = [
    "RLX_WGPU_SNAPSHOT",
    "RLX_BENCH_DISPATCH_ONLY",
    "RLX_WGPU_ALL_PARAMS_WEIGHT",
    "RLX_WGPU_HOST_EAGER_H2D",
    "RLX_WGPU_HOST_BUFFER_COPY",
    "RLX_WGPU_CONCAT_HOST",
    "RLX_WGPU_FORCE_HOST",
    "RLX_WGPU_HOST_MATMUL",
    common::RUN_REPORT_SWITCH,
    "RLX_WGPU_DISCRETE_HOST",
    "RLX_WGPU_SHARD_CAP_MIB",
    "RLX_WGPU_SHARD_STAGE_MIB",
    "RLX_WGPU_SHARD_GPU",
    "RLX_WGPU_ALLOW_SHARD",
    "RLX_WGPU_ONE_OP_PER_PASS",
];

/// Rows in the cache, its row width, and where the new row lands.
const CAP: usize = 64;
const W: usize = 64;
const POS: usize = CAP - 2;

/// Log-mel geometry of the tail-host fixture: the spectrum the multiply produces
/// is `[1, 2*n_fft]` in the FFT block layout, and the filterbank it is projected
/// with is `[n_mels, n_bins]` for `n_bins = n_fft/2 + 1`.
const NFFT: usize = 32;
const N_BINS: usize = NFFT / 2 + 1;
const N_MELS: usize = 8;

/// Geometry of the packed-multiply fixture: the contraction is one 256-element
/// packing block — the whole number of blocks the fused arms answer over, and the
/// staged tile the grouped GEMM's weight loop barriers fall at — against a `[k, n]`
/// packed weight. It is a whole number of Q8_0's 32-element blocks too, so the
/// 8-bit fixture reads the same `k`.
const PACKED_K: usize = 256;
const PACKED_N: usize = 64;

/// One line's fields, as the line spelled them: the key is the name and the value
/// is the text after `=`, quotes stripped — both borrowed from the line.
type Fields<'a> = BTreeMap<&'a str, &'a str>;

/// The region mix: one host step and one device step, and neither can be mistaken
/// for the other in the report.
///
/// The row `KvAppend` writes is the multiply's output, so the device step is what
/// the host step consumes and the appended cache is what the run reads back.
fn region_mix_graph() -> Graph {
    let mut g = Graph::new("wgpu_run_report_region_mix");
    let x = g.input("x", Shape::new(&[1, W], DType::F32));
    let wgt = g.param("wgt", Shape::new(&[W, W], DType::F32));
    let row = g.matmul(x, wgt, Shape::new(&[1, W], DType::F32));
    // The multiply's `[1, W]` row as the `[1, 1, W]` row `KvAppend` takes.
    let row = g.reshape_(row, vec![1, 1, W as i64]);
    let ck = g.param("cache_k", Shape::new(&[1, CAP, W], DType::F32));
    let k = g.add_node(
        Op::KvAppend { axis: 1, pos: POS },
        vec![ck, row],
        Shape::new(&[1, POS + 1, W], DType::F32),
    );
    g.set_outputs(vec![k]);
    g
}

/// The tail-host mix: the multiply the run dispatches, and an `Op::LogMel` over
/// its spectrum — the one step of this schedule the report must count on `host`,
/// as the interval that ran it, rather than on the `stage` pre-pass or the
/// `encode` batch that walked past its cursor entry.
///
/// The output is the log-mel row, which is also what the run reads back, so the
/// deferred step's result is the graph's own result and not a side branch.
fn tail_host_graph() -> Graph {
    let mut g = Graph::new("wgpu_run_report_tail_host");
    let x = g.input("x", Shape::new(&[1, NFFT], DType::F32));
    let wgt = g.param("wgt", Shape::new(&[NFFT, 2 * NFFT], DType::F32));
    let spec = g.matmul(x, wgt, Shape::new(&[1, 2 * NFFT], DType::F32));
    let filters = g.param("filters", Shape::new(&[N_MELS, N_BINS], DType::F32));
    let mel = g.log_mel(spec, filters);
    g.set_outputs(vec![mel]);
    g
}

/// The dynamic-shape graph — the one `tests/basic.rs`'s
/// `dynamic_shape_auto_infers_at_run_time` builds: an input whose leading extent
/// is `Dim::Dynamic(0)`, a scalar constant expanded to `[3, 4]`, and their
/// product.
///
/// Its point here is the compile: `WgpuExecutable::compile` defers a graph it
/// cannot resolve without the input's shape, so the run resolves and compiles it
/// *inside* the reported bracket — the one path where `compile` is an interval
/// rather than an omission.
fn dynamic_graph() -> Graph {
    use rlx_ir::op::BinaryOp;
    use rlx_ir::shape::Dim;
    let mut g = Graph::new("wgpu_run_report_dynamic");
    let dyn_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32);
    let x = g.input("x", dyn_shape);
    let two = g.add_node(
        Op::Constant {
            data: 2.0_f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::from_dims(&[Dim::Static(1), Dim::Static(1)], DType::F32),
    );
    let two_b = g.add_node(
        Op::Expand {
            target_shape: vec![3, 4],
        },
        vec![two],
        Shape::new(&[3, 4], DType::F32),
    );
    let dyn_out = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32);
    let y = g.binary(BinaryOp::Mul, x, two_b, dyn_out);
    g.set_outputs(vec![y]);
    g
}

/// The packed-multiply fixture: one packed weight multiply in `scheme` and a dense
/// one over its rows, at `m` rows. Returns the graph and the packed weight's bytes.
///
/// `m` and the scheme are the whole route decision — the row counts [`PACKED_ROWS`]
/// names put a K-quant multiply on the joined tiled arm and, above one row, on the
/// joined row-grouped GEMM, and the 8-bit scheme's four-row multiply on its scratch
/// entry point's own submission — and the dense multiply beside it keeps the run's
/// own dispatch loop in the schedule, so the report has a batched pass to read the
/// packed one against.
fn packed_graph(m: usize, scheme: QuantScheme) -> (Graph, Vec<u8>) {
    // The scheme's own GGML packing, so the bytes back the `Op::DequantMatMul` the
    // graph carries and every kernel in the route can read them.
    let ty = match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::GgmlType::Q4K,
        QuantScheme::GgufQ8_0 => rlx_gguf::GgmlType::Q8_0,
        other => panic!("no packed fixture for {other:?}"),
    };
    let bytes = common::packed_gguf_bytes(ty, PACKED_K, PACKED_N);
    let mut g = Graph::new("wgpu_run_report_packed");
    let x = g.input("x", Shape::new(&[m, PACKED_K], DType::F32));
    let w = g.param("w", Shape::new(&[bytes.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[m, PACKED_N], DType::F32),
    );
    let p = g.param("p", Shape::new(&[PACKED_N, PACKED_N], DType::F32));
    let out = g.mm(y, p);
    g.set_outputs(vec![out]);
    (g, bytes)
}

/// The child's whole job, for the fixture `run_case` asked it for: one forward pass
/// of that graph. The report goes to stderr; this prints nothing of its own.
fn run_region_mix() {
    let mut exe = Session::new(Device::Gpu).compile(region_mix_graph());
    exe.set_param("cache_k", &common::ramp(CAP * W, 1, 0.013));
    exe.set_param("wgt", &common::ramp(W * W, 1, 0.011));
    let _ = exe.run(&[("x", &common::ramp(W, 1, 0.031))]);
}

/// The tail-host child's job: one forward pass of the graph whose `Op::LogMel` the
/// run defers to the end of its schedule.
fn run_tail_host() {
    let mut exe = Session::new(Device::Gpu).compile(tail_host_graph());
    exe.set_param("wgt", &common::ramp(NFFT * 2 * NFFT, 1, 0.011));
    exe.set_param("filters", &common::ramp(N_MELS * N_BINS, 1, 0.007));
    let _ = exe.run(&[("x", &common::ramp(NFFT, 1, 0.031))]);
}

/// The dynamic child's job: one forward pass of the graph the run has to resolve,
/// fed the way `tests/basic.rs` feeds it.
///
/// The backend is compiled through directly, not through `Session`: `Session::compile`
/// resolves and compiles the graph before the bracket opens, while
/// `WgpuExecutable::compile` leaves it in `unresolved` for the run to resolve — and
/// that resolution is the in-call compile this case is about.
fn run_dynamic() {
    let mut exe = rlx_wgpu::backend::WgpuExecutable::compile(dynamic_graph());
    let xv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let _ = exe.run(&[("x", &xv)]);
}

/// A packed-multiply child's job: one forward pass at `m` rows of `scheme`, the
/// packed weight uploaded as the bytes its param is declared to hold.
fn run_packed(m: usize, scheme: QuantScheme) {
    let (g, bytes) = packed_graph(m, scheme);
    let mut exe = Session::new(Device::Gpu).compile(g);
    exe.set_param_typed("w", &bytes, DType::U8);
    exe.set_param("p", &common::ramp(PACKED_N * PACKED_N, 1, 0.019));
    let _ = exe.run(&[("x", &common::ramp(m * PACKED_K, 1, 0.023))]);
}

/// The child's stderr for one case: this binary re-run with `MARKER` naming
/// `fixture`, under exactly the switches the case is about.
fn run_case(fixture: &str, env: &[(&str, &str)]) -> String {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        "a_run_reports_where_its_time_went",
        "--nocapture",
    ])
    .env(MARKER, fixture);
    for key in CLEARED {
        cmd.env_remove(key);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("re-run this test in a child process");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "the child failed:\n{stderr}");
    stderr
}

/// The kind and the `key=value` fields of one report line, or `None` for a line
/// that is not one of the report's at all — a run prints diagnostics of its own
/// and those are ignored here.
///
/// Strict wherever it does claim a line, which is the strictness this test exists
/// to pin: a report line whose kind is none of [`KINDS`], a field that is not
/// `key=value`, a key that carries a space, a value whose quote is never closed,
/// and a key seen twice all panic. A quoted value — only the adapter's name has
/// one — keeps its spaces and loses its quotes.
fn parse_line(line: &str) -> Option<(&str, Fields<'_>)> {
    let rest = line.strip_prefix(PREFIX)?.trim_start_matches(' ');
    let (kind, mut rest) = rest.split_once(' ').unwrap_or((rest, ""));
    assert!(
        KINDS.contains(&kind),
        "`{line}`: `{kind}` is not a report line kind"
    );
    let mut fields = Fields::new();
    while !rest.is_empty() {
        let (key, tail) = rest
            .split_once('=')
            .unwrap_or_else(|| panic!("`{line}`: `{rest}` is not a key=value field"));
        assert!(
            !key.is_empty() && !key.contains(' '),
            "`{line}`: `{key}` is not a field name"
        );
        let (value, tail) = match tail.strip_prefix('"') {
            Some(quoted) => {
                let (value, tail) = quoted.split_once('"').unwrap_or_else(|| {
                    panic!("`{line}`: the quoted value of `{key}` is never closed")
                });
                (value, tail.strip_prefix(' ').unwrap_or(tail))
            }
            None => tail.split_once(' ').unwrap_or((tail, "")),
        };
        assert!(
            fields.insert(key, value).is_none(),
            "`{line}`: `{key}` appears twice"
        );
        rest = tail;
    }
    Some((kind, fields))
}

/// Every report line in `stderr`, parsed, in the order the report printed them.
fn report_lines(stderr: &str) -> Vec<(&str, Fields<'_>)> {
    stderr.lines().filter_map(parse_line).collect()
}

/// A field map's names, in the order the map keeps them — what a line's shape is
/// asserted against.
fn names<'a>(fields: &Fields<'a>) -> Vec<&'a str> {
    fields.keys().copied().collect()
}

/// The parsed field maps of every line of one kind.
fn fields_of<'a>(lines: &[(&'a str, Fields<'a>)], kind: &str) -> Vec<Fields<'a>> {
    lines
        .iter()
        .filter(|(k, _)| *k == kind)
        .map(|(_, fields)| fields.clone())
        .collect()
}

/// Assert one line's shape: the names its kind is defined to print, and no others.
/// `expected` is spelled in the report's own field order and compared as a set, so
/// the call reads like the line it pins.
fn assert_shape(fields: &Fields<'_>, expected: &[&str], what: &str, stderr: &str) {
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(names(fields), expected, "{what}:\n{stderr}");
}

/// A printed millisecond figure, returned as its value — its shape is asserted,
/// never the figure. `what` names the field in a failure.
fn ms_of(value: &str, what: &str) -> f64 {
    let (whole, frac) = value
        .split_once('.')
        .unwrap_or_else(|| panic!("{what}: `{value}` is not a millisecond figure"));
    assert!(
        !whole.is_empty()
            && whole.bytes().all(|b| b.is_ascii_digit())
            && frac.len() == 6
            && frac.bytes().all(|b| b.is_ascii_digit()),
        "{what}: `{value}` is not six decimals of a millisecond"
    );
    let ms: f64 = value.parse().expect("a decimal figure");
    assert!(ms >= 0.0, "{what}: `{value}` is negative");
    ms
}

/// A printed figure as the integer nanoseconds it is: six decimals of a
/// millisecond is exactly one nanosecond, so this is the report's own clock unit
/// and summing these is exact — which is what lets the test check that the printed
/// column closes on the printed total rather than trusting `convergence=ok`.
fn ns_of(value: &str, what: &str) -> u64 {
    // Shape only: the digits are the figure, not the float `ms_of` returns.
    ms_of(value, what);
    let (whole, frac) = value.split_once('.').expect("six decimals, checked");
    whole.parse::<u64>().expect("a whole part") * 1_000_000
        + frac.parse::<u64>().expect("a six-digit fraction")
}

/// A `kinds` field — `name:count,...`, empty when the interval carried no step.
fn kinds_of(field: &str) -> BTreeMap<&str, usize> {
    if field.is_empty() {
        return BTreeMap::new();
    }
    field
        .split(',')
        .map(|entry| {
            let (name, count) = entry
                .split_once(':')
                .unwrap_or_else(|| panic!("`{field}` is not name:count,..."));
            let count: usize = count
                .parse()
                .unwrap_or_else(|_| panic!("`{field}`: `{count}` is not a count"));
            (name, count)
        })
        .collect()
}

/// The reason `label` was omitted, if the run omitted it.
fn omitted_reason<'a>(omitted: &[Fields<'a>], label: &str) -> Option<&'a str> {
    omitted
        .iter()
        .find(|fields| fields["label"] == label)
        .map(|fields| fields["reason"])
}

/// Every interval line, checked the same way wherever it is read: the fields its
/// kind is defined to print, a phase token for a label, a figure of the printed
/// shape for `host_ms`, and a `kinds` histogram that sums to its line's `steps`.
/// The identity is the report's own — it is asserted, never read as a value.
fn assert_interval_lines(intervals: &[Fields<'_>], stderr: &str) {
    for interval in intervals {
        assert_shape(
            interval,
            &["label", "host_ms", "steps", "kinds"],
            "an interval line's fields",
            stderr,
        );
        let label = interval["label"];
        assert!(
            PHASES.contains(&label),
            "`{label}` is not a phase token:\n{stderr}"
        );
        ms_of(interval["host_ms"], &format!("interval {label} host_ms"));
        let steps: usize = interval["steps"].parse().expect("a step count");
        assert_eq!(
            steps,
            kinds_of(interval["kinds"]).values().sum::<usize>(),
            "interval {label}'s kinds must sum to its steps:\n{stderr}"
        );
    }
}

/// The device view's battery — what **every** case that reads the device half owes
/// wherever it reads it: one `device` line; its fields, with `reason` exactly when
/// the view is `unavailable`, and `measured=0` with `device_ms=unavailable` when it
/// is; the `measured + unusable == passes` identity; one `device_pass` record per
/// pass, numbered in dispatch order, each `ms` either a six-decimal figure or the
/// literal `unavailable` — never a zero — with their split agreeing with the line's
/// counts; and one `device_unusable` line per interval that carried unusable passes,
/// each restating that interval's own label and span, summing to `unusable`. The
/// records come back in dispatch order: which pass is whose, and which interval
/// carried it, is the calling case's own claim.
fn assert_device_view<'a>(lines: &[(&'a str, Fields<'a>)], stderr: &str) -> Vec<Fields<'a>> {
    let devices = fields_of(lines, "device");
    assert_eq!(devices.len(), 1, "one run names one adapter:\n{stderr}");
    let device = &devices[0];
    let view = device["view"];
    assert_eq!(
        device["scope"], "dispatch_passes+packed_multiplies",
        "{stderr}"
    );
    assert_eq!(device["per"], "stretch", "{stderr}");
    assert!(!device["adapter"].is_empty(), "{stderr}");
    assert!(!device["backend"].is_empty(), "{stderr}");
    assert!(
        view == "lower_bound" || view == "unavailable",
        "`{view}` is not a view of the device figures:\n{stderr}"
    );
    // `reason` is exactly what the unavailable view has to explain, so its
    // presence is the same claim as the view's value.
    let mut expected = vec![
        "adapter",
        "backend",
        "view",
        "scope",
        "per",
        "passes",
        "measured",
        "unusable",
        "device_ms",
    ];
    if view == "unavailable" {
        expected.push("reason");
    }
    assert_shape(device, &expected, "the device line's fields", stderr);
    let passes: usize = device["passes"].parse().expect("a pass count");
    let measured: usize = device["measured"].parse().expect("a count");
    let unusable: usize = device["unusable"].parse().expect("a count");
    assert_eq!(
        measured + unusable,
        passes,
        "every pass is measured or unusable:\n{stderr}"
    );

    let pass_lines = fields_of(lines, "device_pass");
    assert_eq!(
        pass_lines.len(),
        passes,
        "one pass record per pass:\n{stderr}"
    );
    let mut measured_passes = 0;
    for (index, pass) in pass_lines.iter().enumerate() {
        assert_shape(
            pass,
            &["index", "interval", "ms"],
            "a device_pass line's fields",
            stderr,
        );
        assert_eq!(
            pass["index"].parse::<usize>().expect("a pass index"),
            index,
            "the passes are numbered in dispatch order:\n{stderr}"
        );
        let interval = pass["interval"];
        assert!(
            PHASES.contains(&interval),
            "`{interval}` is not a phase token:\n{stderr}"
        );
        // A figure this pass could not obtain is `unavailable`, never a zero.
        if pass["ms"] != "unavailable" {
            ms_of(pass["ms"], &format!("device_pass {index} ms"));
            measured_passes += 1;
        }
    }
    assert_eq!(
        (measured_passes, passes - measured_passes),
        (measured, unusable),
        "the pass lines must split into `measured` figures and `unusable` ones:\n{stderr}"
    );

    // Unusable passes are restated one line per interval that carried them —
    // never added to the partition — so they are accounted for per interval and
    // sum to `unusable`. `index` is the ordinal of the interval line the row
    // restates, which is how a consumer matches it when a label repeats; a record
    // no interval ever closed over has no ordinal and no span to restate, and the
    // report says so rather than inventing a zero.
    let intervals = fields_of(lines, "interval");
    let mut unusable_restated = 0;
    for line in fields_of(lines, "device_unusable") {
        assert_shape(
            &line,
            &["passes", "interval", "index", "host_ms"],
            "a device_unusable line's fields",
            stderr,
        );
        let interval = line["interval"];
        assert!(
            PHASES.contains(&interval),
            "`{interval}` is not a phase token:\n{stderr}"
        );
        match line["index"].parse::<usize>() {
            Ok(index) => {
                assert!(
                    (1..=intervals.len()).contains(&index),
                    "device_unusable index={index} names no interval line:\n{stderr}"
                );
                assert_eq!(
                    intervals[index - 1]["label"],
                    interval,
                    "device_unusable index={index} must restate that interval's own label:\n\
                     {stderr}"
                );
                ms_of(
                    line["host_ms"],
                    &format!("device_unusable {interval} host_ms"),
                );
            }
            Err(_) => assert_eq!(
                (line["index"], line["host_ms"]),
                ("none", "unavailable"),
                "a pass no interval closed over has no ordinal and no span:\n{stderr}"
            ),
        }
        unusable_restated += line["passes"].parse::<usize>().expect("a pass count");
    }
    assert_eq!(
        unusable_restated, unusable,
        "every unusable pass is restated on one line:\n{stderr}"
    );

    if view == "unavailable" {
        assert_eq!(
            (measured, device["device_ms"]),
            (0, "unavailable"),
            "a view with no figure has no measured pass and no figure to print:\n{stderr}"
        );
    } else {
        ms_of(device["device_ms"], "device device_ms");
    }

    pass_lines
}

/// Case 1 — the region mix, whose report is the whole of it.
///
/// Structure only, and each claim is about a line's own fields or about the
/// identities the report states between them: the interval count is the `run`
/// line's `intervals`, a `kinds` histogram sums to its line's `steps`, the pass
/// lines add up to `passes`/`measured`/`unusable`. The five required labels are
/// all present, so `host` is reported as an interval rather than as an omission,
/// while `probe` — the instrument's own fence, resolve and map — and `compile` are
/// the further labels a reporting run may carry. This graph is compiled outside the
/// bracket, so its `compile` must be an omission that names the reason instead.
fn assert_region_mix(stderr: &str, schedule: usize) {
    let lines = report_lines(stderr);

    // ── the run line ─────────────────────────────────────────────────────
    let runs = fields_of(&lines, "run");
    assert_eq!(runs.len(), 1, "one run is one report block:\n{stderr}");
    let run = &runs[0];
    assert_shape(
        run,
        &[
            "total_ms",
            "partition_ms",
            "convergence",
            "intervals",
            "schedule",
            "reached",
        ],
        "the run line's fields",
        stderr,
    );
    assert_eq!(
        run["convergence"], "ok",
        "the partition must cover the bracket:\n{stderr}"
    );
    // The printed convergence, not a time assertion: the two are the same integer
    // nanosecond sum and are therefore the same text.
    assert_eq!(
        run["total_ms"], run["partition_ms"],
        "convergence=ok means one figure printed twice:\n{stderr}"
    );
    ms_of(run["total_ms"], "run total_ms");
    assert_eq!(
        run["schedule"].parse::<usize>().expect("a step count"),
        schedule,
        "the run line must name the schedule the premise asserts:\n{stderr}"
    );
    // The furthest schedule cursor the run reached — its own field, not an
    // interval's `steps=`, which counts only the steps that interval covered.
    run["reached"].parse::<usize>().expect("a step count");

    // ── the intervals ────────────────────────────────────────────────────
    let intervals = fields_of(&lines, "interval");
    assert_eq!(
        intervals.len(),
        run["intervals"]
            .parse::<usize>()
            .expect("an interval count"),
        "the interval lines are the count the run line declares:\n{stderr}"
    );
    assert_interval_lines(&intervals, stderr);
    // The printed column closes on the printed total at printed precision — six
    // decimals of a millisecond is the integer nanosecond, so the sum is exact —
    // which is what a scripted consumer can check instead of trusting the flag.
    let printed: u64 = intervals
        .iter()
        .map(|interval| ns_of(interval["host_ms"], "interval host_ms"))
        .sum();
    assert_eq!(
        printed,
        ns_of(run["total_ms"], "run total_ms"),
        "the printed interval figures must sum to the printed total:\n{stderr}"
    );
    let labels: BTreeSet<&str> = intervals.iter().map(|interval| interval["label"]).collect();
    for (label, _) in REQUIRED {
        assert!(
            labels.contains(label),
            "this run's schedule has a {label} interval:\n{stderr}"
        );
    }
    // The run's own `probe` and — for a graph the run resolves itself — `compile`
    // are the further intervals a reporting run may carry.
    let permitted: BTreeSet<&str> = REQUIRED
        .iter()
        .map(|(label, _)| *label)
        .chain([COMPILE.0, "probe"])
        .collect();
    assert!(
        labels.is_subset(&permitted),
        "the required five plus the reporting run's own `probe` and `compile` are the intervals \
         here, found {labels:?}:\n{stderr}"
    );
    let hosts: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == "host")
        .collect();
    assert_eq!(
        hosts.len(),
        1,
        "the one host step of this schedule is one interval:\n{stderr}"
    );
    // The intervals this run's real work landed in must not read as nothing: a
    // bound, not a time — nothing here is compared with a value or a threshold —
    // and it is what says the partition measured the work instead of closing over
    // it. The step-by-step and the host copy are tens of microseconds at their
    // smallest, so a zero here is a broken reading, not a fast machine.
    for interval in &intervals {
        if ["encode", "wait", "host", "readback"].contains(&interval["label"]) {
            let figure: f64 = interval["host_ms"].parse().expect("a millisecond figure");
            assert!(
                figure > 0.0,
                "the `{}` interval covers real work but measured nothing:\n{stderr}",
                interval["label"]
            );
        }
    }
    assert_eq!(
        kinds_of(hosts[0]["kinds"])
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        ["buffer_copy"],
        "the host interval is the copy and nothing else:\n{stderr}"
    );
    let encodes: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == "encode")
        .collect();
    assert!(
        encodes
            .iter()
            .any(|interval| kinds_of(interval["kinds"]).contains_key("matmul")),
        "the multiply is the device step this run encodes:\n{stderr}"
    );
    for interval in &encodes {
        assert!(
            !kinds_of(interval["kinds"]).contains_key("buffer_copy"),
            "the copy is not a device step of this run:\n{stderr}"
        );
    }

    // ── the omissions ────────────────────────────────────────────────────
    let omitted = fields_of(&lines, "omitted");
    for omission in &omitted {
        assert_shape(
            omission,
            &["label", "reason"],
            "an omitted line's fields",
            stderr,
        );
        let (label, reason) = (omission["label"], omission["reason"]);
        assert!(
            REQUIRED
                .iter()
                .chain([&COMPILE])
                .any(|(required, token)| *required == label && *token == reason),
            "`{label}`/`{reason}` is not one of the five fixed pairs, nor the in-call compile's:\n\
             {stderr}"
        );
    }
    for (label, _) in REQUIRED {
        assert!(
            omitted_reason(&omitted, label).is_none(),
            "every required interval of this run happened, so `{label}` must not be omitted:\n\
             {stderr}"
        );
    }
    // This run's graph is compiled by `Session::compile`, before the bracket opens,
    // so the in-call compile has no edge here: the run must say so rather than print
    // a zero-steps line beside the name.
    assert_eq!(
        omitted_reason(&omitted, COMPILE.0),
        Some(COMPILE.1),
        "a graph compiled outside the bracket has no `compile` edge to report:\n{stderr}"
    );

    // ── the device view ──────────────────────────────────────────────────
    assert_device_view(&lines, stderr);

    // The switch that inflates these intervals is named on every line it touched.
    for line in fields_of(&lines, "extra_work") {
        assert_shape(
            &line,
            &["switch", "effect"],
            "an extra_work line's fields",
            stderr,
        );
        assert!(
            !line["switch"].is_empty() && !line["effect"].is_empty(),
            "{stderr}"
        );
    }
}

/// Case 2 — a dispatch-only run: no read-back happened, so none may be reported,
/// and the run has to say that it omitted one. The switch that skips the read-back
/// is named as extra work, since it is what the intervals are read against.
///
/// A dispatch-only run leaves the loop from its own exit rather than through the
/// read-back mark, so what this pins is that such a run still reports its schedule
/// as walked to the end — `reached` is the `schedule` the premise read off the same
/// graph — and that the interval which did the work counts the steps it covered: a
/// `steps=0 kinds=` line for work an interval ran is the failure this case catches.
fn assert_dispatch_only(stderr: &str, schedule: usize) {
    let lines = report_lines(stderr);
    let runs = fields_of(&lines, "run");
    assert_eq!(runs.len(), 1, "one report block:\n{stderr}");
    assert_eq!(
        runs[0]["reached"].parse::<usize>().expect("a step count"),
        schedule,
        "a dispatch-only run walks its schedule to the end:\n{stderr}"
    );
    let intervals = fields_of(&lines, "interval");
    assert_interval_lines(&intervals, stderr);
    assert!(
        !intervals
            .iter()
            .any(|interval| interval["label"] == "readback"),
        "a dispatch-only run must not claim a read-back it never did:\n{stderr}"
    );
    // The interval that ran the host step is the one that counts it: this graph's
    // host step is the `KvAppend` copy `RLX_WGPU_HOST_BUFFER_COPY=1` forces onto the
    // host route, and the loop's exit is where the interval still open closes over
    // the cursor range it really covered. `host steps=0 kinds=` for a copy that ran
    // is what this asserts against.
    let hosts: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == "host")
        .collect();
    assert_eq!(
        hosts.len(),
        1,
        "the one host step of this schedule is one interval:\n{stderr}"
    );
    assert_eq!(
        (
            hosts[0]["steps"].parse::<usize>().expect("a step count"),
            kinds_of(hosts[0]["kinds"]),
        ),
        (1, BTreeMap::from([("buffer_copy", 1)])),
        "the host interval must count the copy it ran:\n{stderr}"
    );
    let omitted = fields_of(&lines, "omitted");
    assert_eq!(
        omitted_reason(&omitted, "readback"),
        Some("no_readback_in_this_path"),
        "the read-back it did not do must be named as omitted:\n{stderr}"
    );
    assert!(
        fields_of(&lines, "extra_work")
            .iter()
            .any(|line| line["switch"] == "RLX_BENCH_DISPATCH_ONLY"),
        "the report must name the switch these intervals are read against:\n{stderr}"
    );
}

/// Case 3: no line of the report at all — the report's cost and its lines are the
/// switch's, so an ordinary run prints exactly what it always did.
fn assert_no_report_lines(stderr: &str, why: &str) {
    let lines = report_lines(stderr);
    assert!(
        lines.is_empty(),
        "{why}, so no report line may appear, found {}:\n{stderr}",
        lines.len()
    );
}

/// Case 5 — the deferred tail-host step, counted by the interval that ran it and
/// by nothing else.
///
/// `step` is the kind the schedule defers. The `host` interval is that step alone,
/// counted once: the run's `stage` pre-pass and its `encode` batch both walk past
/// the step's cursor entry without running it, so a count on their lines would be
/// an attribution to the wrong interval, and a second count under `host` would be
/// one executed step on two lines. `encode` still carries the multiply, which is
/// the device work of this graph and not the deferred step.
fn assert_tail_host_counted_once(stderr: &str, step: &str) {
    let lines = report_lines(stderr);
    let runs = fields_of(&lines, "run");
    assert_eq!(runs.len(), 1, "one run is one report block:\n{stderr}");
    assert_eq!(
        runs[0]["schedule"]
            .parse::<usize>()
            .expect("a schedule length"),
        2,
        "this fixture's schedule is the multiply and the deferred step:\n{stderr}"
    );
    let intervals = fields_of(&lines, "interval");
    let hosts: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == "host")
        .collect();
    assert_eq!(
        hosts.len(),
        1,
        "the deferred step is one interval, the `host` one that ran it:\n{stderr}"
    );
    assert_eq!(
        kinds_of(hosts[0]["kinds"]),
        BTreeMap::from([(step, 1)]),
        "the `host` interval must be the deferred step, counted once:\n{stderr}"
    );
    for interval in &intervals {
        if interval["label"] == "host" {
            continue;
        }
        assert!(
            !kinds_of(interval["kinds"]).contains_key(step),
            "the `{}` interval never ran the deferred step:\n{stderr}",
            interval["label"]
        );
    }
    assert!(
        intervals
            .iter()
            .filter(|interval| interval["label"] == "encode")
            .any(|interval| kinds_of(interval["kinds"]).contains_key("matmul")),
        "the multiply is the device step this run encodes:\n{stderr}"
    );
}

/// Case 6 — the in-call compile: a dynamic-shape graph `WgpuExecutable::compile`
/// deferred, so the run resolves and compiles it inside the bracket and the
/// `compile` interval is a real interval above the first staged step rather than an
/// omission.
///
/// A compile is real work and must not read as nothing — the same bound every
/// interval carrying work is held to, not a threshold on a value — and it runs no
/// schedule step, since the schedule it produces is what the run then walks. The
/// label's omission is asserted absent: the one path that has the edge must not
/// print the reason that says it has none.
fn assert_in_call_compile(stderr: &str) {
    let lines = report_lines(stderr);
    assert_eq!(
        fields_of(&lines, "run").len(),
        1,
        "one report block:\n{stderr}"
    );
    let intervals = fields_of(&lines, "interval");
    assert_interval_lines(&intervals, stderr);
    let compiles: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == COMPILE.0)
        .collect();
    assert_eq!(
        compiles.len(),
        1,
        "the run resolves its graph once, so the in-call compile is one interval:\n{stderr}"
    );
    let figure: f64 = compiles[0]["host_ms"]
        .parse()
        .expect("a millisecond figure");
    assert!(
        figure > 0.0,
        "the compile covers real work but measured nothing:\n{stderr}"
    );
    assert_eq!(
        (
            compiles[0]["steps"].parse::<usize>().expect("a step count"),
            compiles[0]["kinds"],
        ),
        (0, ""),
        "the compile runs no step of the schedule it produces:\n{stderr}"
    );
    assert!(
        omitted_reason(&fields_of(&lines, "omitted"), COMPILE.0).is_none(),
        "this is the path where the compile has an edge, so `compile` must not be omitted:\n\
         {stderr}"
    );
}

/// Case 7 — the same dynamic-shape graph, under `RLX_BENCH_DISPATCH_ONLY`: a graph
/// whose **every** step is device work, dispatched in one batch, so the run returns
/// from inside that batch rather than through the loop's own exit. That is the edge
/// the schedule's end is read at second, and this case pins its effect.
///
/// `schedule` is that schedule's length, asserted against the run line's own
/// `schedule` field and its `reached`: a run that stopped short must not read as a
/// completed one, and a completed one must not read as short. The `encode`
/// interval(s) then have to be the whole schedule — the sum of their `steps` is
/// `schedule` and their kinds are exactly its one step kind, `binary` — and the run,
/// which reads nothing back, must name that omission.
fn assert_dispatch_only_device_only(stderr: &str, schedule: usize) {
    let lines = report_lines(stderr);
    let runs = fields_of(&lines, "run");
    assert_eq!(runs.len(), 1, "one report block:\n{stderr}");
    let run = &runs[0];
    assert_eq!(
        run["schedule"].parse::<usize>().expect("a step count"),
        schedule,
        "the run line must name the schedule this case asserts:\n{stderr}"
    );
    assert_eq!(
        run["reached"].parse::<usize>().expect("a step count"),
        schedule,
        "the batch that did the work walks its schedule to the end:\n{stderr}"
    );
    let intervals = fields_of(&lines, "interval");
    assert_interval_lines(&intervals, stderr);
    let encodes: Vec<&Fields<'_>> = intervals
        .iter()
        .filter(|interval| interval["label"] == "encode")
        .collect();
    let (mut steps, mut kinds) = (0usize, BTreeMap::new());
    for interval in &encodes {
        steps += interval["steps"].parse::<usize>().expect("a step count");
        for (kind, count) in kinds_of(interval["kinds"]) {
            *kinds.entry(kind).or_insert(0) += count;
        }
    }
    assert_eq!(
        (steps, kinds),
        (schedule, BTreeMap::from([("binary", schedule)])),
        "the `encode` interval(s) must cover the whole schedule, counted by kind:\n{stderr}"
    );
    assert_eq!(
        omitted_reason(&fields_of(&lines, "omitted"), "readback"),
        Some("no_readback_in_this_path"),
        "the read-back this run never did must be named as omitted:\n{stderr}"
    );
}

/// Cases 8 to 10 — the packed multiply's own pass in the device view.
///
/// [`assert_device_view`] owns the device line's full battery (its fields, its
/// `unavailable` reason, the `device_unusable` cross-checks); what a packed
/// multiply adds is read here: the multiply's pass must be one of the recorded
/// passes and not the dispatch loop's alone, and the interval the report names for
/// it must be the one that carried the multiply.
///
/// `joined` is the route this case's row count takes: `true` when the run encodes
/// the call into its own batch. The joined arms — the tiled one-row arm and the
/// row-grouped GEMM above it — ride that batch, so the multiply's pass is an
/// `encode` pass beside the batch's: the `encode` interval then carries the batch
/// pass opened before its first remaining step turns out to be the host-classified
/// multiply, the multiply's own pass, and the dense step's batch pass. The 8-bit
/// scheme's multi-row multiply is not joined: its group entry point submits on its
/// own inside the `host` step, so its pass is the only `device_pass` line outside
/// `encode` — the interval the run's own dispatch loop never opens a pass in, which
/// is what makes such a line readable as a packed multiply's.
fn assert_packed_covered(stderr: &str, joined: bool) {
    let lines = report_lines(stderr);
    let pass_lines = assert_device_view(&lines, stderr);

    // The premise, read off the report: this run's schedule held a packed multiply,
    // and the interval whose kinds count it is the one that carried it — the link a
    // reader uses, which is why the cases read it rather than assuming it.
    let intervals = fields_of(&lines, "interval");
    assert_interval_lines(&intervals, stderr);
    let carrier = intervals
        .iter()
        .find(|interval| kinds_of(interval["kinds"]).contains_key("dequant_matmul_gguf"))
        .expect("no interval counts the packed multiply this graph has");

    let outside_encode: Vec<&Fields<'_>> = pass_lines
        .iter()
        .filter(|pass| pass["interval"] != "encode")
        .collect();
    if joined {
        // The joined arm: the multiply rides the run's own batch, so its pass is an
        // `encode` line among the batch's — one more there than the run's own dispatch
        // loop opened, since the batch pass before it is the empty one the loop opens
        // before it sees that its step is the host-classified multiply.
        assert_eq!(
            carrier["label"], "encode",
            "the row the batch joined is the interval that carried it:\n{stderr}"
        );
        assert!(
            outside_encode.is_empty(),
            "the joined multiply rides the batch it joined:\n{stderr}"
        );
        let encodes = pass_lines
            .iter()
            .filter(|pass| pass["interval"] == "encode")
            .count();
        assert!(
            encodes >= 3,
            "the `encode` interval must carry the packed multiply's own pass beside the \
             run's own — the empty batch pass, the multiply's, and the dense step's — \
             found {encodes}:\n{stderr}"
        );
    } else {
        // The unjoined case — the 8-bit scheme's four-row multiply, which takes its
        // group's own entry point: a submission of its own, inside the `host` step's
        // interval — the one the run's own dispatch loop never opens a pass in.
        assert_eq!(
            carrier["label"], "host",
            "the unjoined multiply's own submission runs in the `host` step:\n{stderr}"
        );
        assert_eq!(
            outside_encode.len(),
            1,
            "an unjoined packed multiply's pass must be the run's one pass outside \
             `encode`:\n{stderr}"
        );
        assert_eq!(
            outside_encode[0]["interval"], carrier["label"],
            "the pass outside `encode` must be the one the packed multiply's interval \
             carried:\n{stderr}"
        );
    }
}

/// One test fn, ten child cases: the report's structure is one claim per case,
/// and a case's switch set is its own process's environment.
#[test]
fn a_run_reports_where_its_time_went() {
    if let Ok(fixture) = std::env::var(MARKER) {
        match fixture.as_str() {
            REGION_MIX => run_region_mix(),
            TAIL_HOST => run_tail_host(),
            DYNAMIC => run_dynamic(),
            other => match packed_rows(other) {
                Some((rows, scheme)) => run_packed(rows, scheme),
                None => panic!("`{other}` is not a fixture this test runs"),
            },
        }
        return;
    }
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    // The premise, from the schedule: one device step and one host step, in that
    // order — the mix both halves of the report are read against. Both routes are
    // runtime decisions (step names are the plan's), so this holds with and
    // without the switches a case sets.
    let steps = rlx_wgpu::backend::WgpuExecutable::compile(region_mix_graph()).test_step_names();
    assert_eq!(
        steps,
        ["matmul", "buffer_copy"],
        "the region-mix graph must be one device step and one host step"
    );

    // The tail-host premise: the multiply, then the one step the run defers to the
    // end of the schedule — not a host-route step it walks with the rest.
    let tail_steps =
        rlx_wgpu::backend::WgpuExecutable::compile(tail_host_graph()).test_step_names();
    assert_eq!(
        tail_steps,
        ["matmul", "log_mel_host"],
        "the tail-host graph must be one device step and one step the run defers"
    );

    // The dynamic premise: `WgpuExecutable::compile` defers a graph whose input
    // extents it cannot resolve, so this one names no step until the run resolves
    // it — which is what puts the compile inside the bracket. The other two
    // fixtures resolve at compile time and keep the steps asserted above.
    let dynamic_steps =
        rlx_wgpu::backend::WgpuExecutable::compile(dynamic_graph()).test_step_names();
    assert!(
        dynamic_steps.is_empty(),
        "the dynamic graph is unresolved until the run resolves it, found {dynamic_steps:?}"
    );

    // The packed-multiply premise: one packed multiply and the dense step over its
    // rows, in that order, at each row count and scheme a case runs. A packed
    // multiply whose scheme had no fused arm would leave this list without its step
    // — and the report could then only say the view covers nothing here.
    for (m, _, scheme, _) in PACKED_ROWS {
        let (g, _) = packed_graph(m, scheme);
        let packed_steps = rlx_wgpu::backend::WgpuExecutable::compile(g).test_step_names();
        assert_eq!(
            packed_steps,
            ["dequant_matmul_gguf", "matmul"],
            "the packed fixture at {m} rows of {scheme:?} must be one packed multiply \
             and one dense step"
        );
    }

    // 1. The region mix: the report in full, structure only.
    assert_region_mix(
        &run_case(
            REGION_MIX,
            &[
                (common::RUN_REPORT_SWITCH, "1"),
                ("RLX_WGPU_HOST_BUFFER_COPY", "1"),
            ],
        ),
        steps.len(),
    );

    // 2. Dispatch-only, on the same graph.
    assert_dispatch_only(
        &run_case(
            REGION_MIX,
            &[
                (common::RUN_REPORT_SWITCH, "1"),
                ("RLX_WGPU_HOST_BUFFER_COPY", "1"),
                ("RLX_BENCH_DISPATCH_ONLY", "1"),
            ],
        ),
        steps.len(),
    );

    // 3. The switch off: an ordinary run is unchanged in its printed lines.
    assert_no_report_lines(
        &run_case(REGION_MIX, &[("RLX_WGPU_HOST_BUFFER_COPY", "1")]),
        "the report switch is off",
    );

    // 4. A snapshot call on the same graph: the whole session is a replay, so it
    // reports nothing at all — not even the full pass it ends with.
    let snap = run_case(
        REGION_MIX,
        &[
            (common::RUN_REPORT_SWITCH, "1"),
            ("RLX_WGPU_HOST_BUFFER_COPY", "1"),
            ("RLX_WGPU_SNAPSHOT", "1"),
        ],
    );
    assert!(
        snap.contains("[wgpu-snapshot] replaying"),
        "this case is about a replay only if the child really replayed:\n{snap}"
    );
    assert_no_report_lines(&snap, "the call is a snapshot replay session");

    // 5. The deferred tail-host step: the interval that ran it counts it once, and
    // no interval that walked past it counts it too.
    assert_tail_host_counted_once(
        &run_case(TAIL_HOST, &[(common::RUN_REPORT_SWITCH, "1")]),
        tail_steps[1],
    );

    // 6. The in-call compile: the graph the run resolves itself, whose compile is
    // an interval of the bracket rather than the omission case 1 asserts.
    assert_in_call_compile(&run_case(DYNAMIC, &[(common::RUN_REPORT_SWITCH, "1")]));

    // 7. Dispatch-only on a graph whose every step is device work, so the run
    // returns from inside the batch that did it: the dynamic fixture's schedule is
    // one step — its multiply by two — and the batch encodes all of it.
    assert_dispatch_only_device_only(
        &run_case(
            DYNAMIC,
            &[
                (common::RUN_REPORT_SWITCH, "1"),
                ("RLX_BENCH_DISPATCH_ONLY", "1"),
            ],
        ),
        1,
    );

    // 8–10. The packed multiplies, on the routes their row counts and schemes
    // decide: the joined K-quant cases at one and four rows, and the 8-bit scheme's
    // four-row call, which takes its group's own scratch entry point and so keeps a
    // submission of its own inside the `host` interval.
    for (_, fixture, _, joined) in PACKED_ROWS {
        assert_packed_covered(
            &run_case(fixture, &[(common::RUN_REPORT_SWITCH, "1")]),
            joined,
        );
    }
}
