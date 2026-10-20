// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one-row K-quant packed multiply's own packed-weight stream rate.
//!
//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test gguf_one_row_stream_rate -- --ignored --nocapture
//! ```
//!
//! # The accounting
//!
//! The figure is **the tensor's own packed weight bytes one step reads, per second
//! of call wall time** (`packed_bytes_per_step / (call_seconds / steps)`, bytes per
//! second). Nothing else is in the numerator: not the input row, not the output.
//! Inside the denominator sit the call's one submission, one readback and one
//! blocking wait, the per-step host work (a uniform write and a bind group per step
//! and per chunk), the input row's own load traffic and the unpack arithmetic. This
//! backend exposes no usable per-dispatch device clock, so the denominator is wall
//! time, not device time.
//!
//! **A step** is one invocation of the packed multiply over one weight region and
//! one input row; **a call** is one run of a graph holding the form's fixed number
//! of them, each step's region its own parameter in the arena and each region read
//! once within a call. All of a call's steps share **one** input row, so the
//! input-side term is one row requested once per output-unit workgroup per step, not
//! one row per step.
//!
//! The figure is a **proxy** and deliberately term-agnostic: any wall-time reduction
//! raises it, including one in the fixed call floor, the input row's traffic or the
//! per-step host work. On its own it says nothing about weight reads, and it is not
//! a comparison against a plain dense read of the same bytes — that read unpacks
//! nothing, multiplies nothing and takes no input, so it is a ceiling on reading,
//! not a target for this arm.
//!
//! # The forms
//!
//! Each row is a call of `steps` such multiplies over distinct weight regions
//! (`bytes/step` = one region's packed bytes) — the `FORMS` array below mirrors
//! this table, and only `packed_bytes_per_step` is tied by an assertion, so an
//! edit to either side has to move both:
//!
//! | form | scheme | k | n | steps | row bytes | bytes/step |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 4-bit FFN gate | Q4_K | 2048 | 6144 | 32 | 1152 | 7,077,888 |
//! | 6-bit FFN down | Q6_K | 6144 | 2048 | 32 | 5040 | 10,321,920 |
//! | 4-bit FFN gate, widened x8 | Q4_K | 2048 | 49152 | 8 | 1152 | 56,623,104 |
//! | 4-bit predictor block | Q4_K | 2048 | 2048 | 128 | 1152 | 2,359,296 |
//! | 4-bit predictor block, wide n | Q4_K | 2048 | 8192 | 32 | 1152 | 9,437,184 |
//! | 6-bit recognition head | Q6_K | 2048 | 151936 | 3 | 1680 | 255,252,480 |
//! | 8-bit predictor down | Q8_0 | 1024 | 1024 | 128 | 1088 | 1,114,112 |
//! | 8-bit predictor head | Q8_0 | 1024 | 2048 | 64 | 1088 | 2,228,224 |
//! | 8-bit predictor gate+up | Q8_0 | 1024 | 6144 | 32 | 1088 | 6,684,672 |
//! | 8-bit predictor block, k=2048 | Q8_0 | 2048 | 2048 | 32 | 2176 | 4,456,448 |
//! | 8-bit predictor, widened x8 | Q8_0 | 1024 | 49152 | 4 | 1088 | 53,477,376 |
//!
//! The five Q8_0 rows are the synthesis cell's code predictor, which stores 8-bit
//! weights: `k = 1024` is its attention-projection width (its `ffn_down` contracts
//! over 3072), and `n = 1024` is that projection's output, `2048` its head, `6144`
//! its fused gate+up. Row bytes are 1088 for `k = 1024` — 32 Q8_0 blocks, each 2
//! bytes of scale and 32 of quants — and 2176 for `k = 2048`.
//!
//! The widened form's region is the first form's own packed bytes repeated eight
//! times into one buffer — a synthetic widening, no such tensor being in any file —
//! and its eight regions are byte-identical copies of it, so the first form's bytes
//! appear sixty-four times in that call's address space. The two K-quant arms'
//! predictor forms are the geometry the model cells' multiplies have at that
//! scheme — `k = n = 2048` for one level's block, and the same `k` at four times
//! the width to separate a dispatch's column count from everything else; the
//! Q8_0 rows above are the predictor's own stored scheme, at its own widths —
//! which is why their figures are the ones the cells'
//! rates are read against. The narrowest form is the most floor-bound and the
//! noisiest, so a conclusion resting on it alone is labelled as such; the second is
//! the only marginal-occupancy one, having by far the fewest output columns. Each
//! form is built and dropped on its own — two at once is the order of two gigabytes
//! of transient device memory — and a form whose transient bytes exceed the
//! adapter's `max_buffer_size` is **named as not fitting and skipped**, never
//! silently dropped, with the forms that do fit still measured.
//!
//! # What the arm does now
//!
//! A workgroup's two columns share one `vec4` read of the input row (input-side,
//! halved — not a weight-read win); Q6_K's lane unit requests each byte of a `ql`
//! span once instead of twice; Q4_K reads its 16-byte header as four whole words
//! instead of byte-wise, a request-volume increase confined to the block that unit
//! already reads. The host work per chunk (one uniform write and one bind group,
//! nothing pooled) and one submission and one wait per call are unchanged, so the
//! arm rides the enclosing batch. A development-time step-count sweep found a
//! fixed per-call cost of the order of a third of the narrow form's 32-step call,
//! which is why that form's ratio is the one that does not reproduce; the
//! recognition head's derived dispatch count moved 3 -> 2, two columns covering
//! twice the columns per workgroup there.
//!
//! # The workgroup's shape: two studies, no winner
//!
//! Two whole-shader constants place one workgroup on a narrow row:
//! `GEMV_TILED_COLS_PER_WG` output columns it covers and `LANES` threads splitting
//! each column's contraction (the Rust side's `GEMV_TILED_THREADS`, tied to the
//! shader by `gguf_tiled_gemv_geometry_matches_the_shader`). Each study moved one
//! of them and nothing else, in one back-to-back window per table: min GB/s over
//! five rounds, each column pair measured in the same session.
//!
//! Columns per workgroup:
//!
//! | form | n | 1 col | 2 cols | 4 cols |
//! | --- | --- | --- | --- | --- |
//! | 4-bit FFN gate | 6144 | 58.3 | 58.2 | 58.5 |
//! | 6-bit FFN down | 2048 | 67.9 | 65.6 | 48.2 |
//! | 4-bit FFN gate, widened x8 | 49152 | 71.1 | 86.7 | 79.3 |
//! | 4-bit predictor block | 2048 | 41.1 | 45.7 | — |
//! | 4-bit predictor block, wide n | 8192 | 62.4 | 73.6 | — |
//! | 6-bit recognition head | 151936 | 79.0 | 85.0 | 58.2 |
//!
//! Two columns — the tree's — leads on two of the four forms where all three ran
//! (49 152 by 9 %, 151 936 by 7 %) and is within 3.5 % of the best on the other
//! two; four columns ran on four forms, leading 6 144 by 0.5 %, sitting in the
//! middle of 49 152's 71.1/86.7/79.3, and trailing on 2 048 (48.2) and the head
//! (58.2). Per-setting spreads were not recorded in that sweep, so a difference
//! under ~4 % is not a difference, and the sweep predates the paired-column issue
//! below, so its two-column cell is not the arm as it stands.
//!
//! Lanes per column, at `k = 2048` (64 units a row: 16 lanes walk four each, 32
//! two, 64 one). Window A moved 16 → 32; window B moved 32 → 64 in a second
//! session, so its 32 column is that session's and not window A's:
//!
//! | form | n | A: 16 lanes | A: 32 lanes |
//! | --- | --- | --- | --- |
//! | 4-bit FFN gate | 6144 | 43.4 | 59.8 |
//! | 6-bit FFN down | 2048 | 43.8 | 65.1 |
//! | 4-bit FFN gate, widened x8 | 49152 | 54.2 | 81.9 |
//! | 4-bit predictor block | 2048 | 32.4 | 41.6 |
//! | 4-bit predictor block, wide n | 8192 | 47.5 | 67.9 |
//! | 6-bit recognition head | 151936 | 52.8 | 82.3 |
//!
//! | form | n | B: 32 lanes | B: 64 lanes |
//! | --- | --- | --- | --- |
//! | 4-bit FFN gate | 6144 | 59.4 | 60.3 |
//! | 6-bit FFN down | 2048 | 65.7 | 68.1 |
//! | 4-bit FFN gate, widened x8 | 49152 | 81.8 | 74.6 |
//! | 4-bit predictor block | 2048 | 43.1 | 42.1 |
//! | 4-bit predictor block, wide n | 8192 | 69.2 | 64.0 |
//! | 6-bit recognition head | 151936 | 82.3 | 74.7 |
//!
//! 16 lanes loses on all six by 22–36 %. 64 lanes wins two forms by 1.5–3.7 % and
//! loses four by 2.3–9.2 %, one of the losses being the 4-bit predictor block at
//! `n = 2 048` — the predictor's own shape. At cell scale the predictor frame ran
//! 66.34 ms at 32 lanes (66.17–66.38 over three runs) against 66.84 at 64 (one
//! run, same session), the waveform witness held on both. 32 lanes, the tree's,
//! stands; 16 lanes was not taken through the cell.
//!
//! **In the arm, and not.** The two columns' packed words are issued together
//! (`word0`/`word1` read before either is consumed) and Q4_K's block header and
//! `qs` words go through `read_word_q4`'s aligned single load — both bit-identical,
//! both part of every column above and of the after side of the table below. The
//! further step of hoisting a unit's whole eighteen loads is **not** in the tree:
//! on the four Q4_K forms it read 56.1, 93.3, 44.7 and 75.4 GB/s against that
//! sweep's two-column 58.2, 86.7, 45.7 and 73.6 — two above (+7.6 % at 49 152,
//! +2.4 % at 8 192), two below (−3.6 %, −2.2 %), and no spread recorded for that
//! sweep, so none of the four is a difference.
//!
//! What separates the forms is the dispatch's column count, not its tile: in
//! window A the same per-workgroup work reads 41.6 GB/s at 2 048 columns and 81.9
//! at 49 152. One predictor frame is sixteen levels, 1 465 dispatch slots (348 of
//! them multiplies) over 1 341 MiB, so the frame's next step is how many
//! dispatches it issues, not the shape of any one of them.
//!
//! # Asserted, and not
//!
//! Asserted, never a time: the tabulated packed bytes per step, the joined one-row
//! route for every step in one submission with one blocking wait, and finite
//! outputs. Printed: the figure, the derived per-step dispatch count, the
//! submission/wait structure, the recorded one-row shape and the per-round times.
//! Nothing here checks the shader's byte-wise helper reads, the host-side wiring
//! this route shares with the multi-row fallback, or any adapter but the one
//! printed.
//!
//! # Recorded figures
//!
//! Through this cell on the two revisions either side of the one-row arm's change,
//! adapter `Apple M3 Pro`, `--release`, medians never paired with mins:
//!
//! | form | before, min | after, min |
//! | --- | --- | --- |
//! | 4-bit FFN gate | 50.7 | 55.2 |
//! | 6-bit FFN down | 50.4 | 59.6 |
//! | 4-bit FFN gate, widened | 65.7 | 77.1 |
//! | 6-bit recognition head | 62.3 | 83.1 |
//!
//! Cross-revision by construction — two runs on two revisions, fixed order, no
//! in-run control — so each side's own spread is the only bound on it, and the
//! narrowest form does not reproduce to within its spread (47.4…57.8 GB/s).
//!
mod common;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{
    gemv_one_row_dispatches, gemv_one_row_shape, reset_gemv_one_row_shape, reset_route_counts,
    route_counts,
};
use rlx_wgpu::gguf_host::gguf_scheme_id;
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts, wgpu_device};

/// One measured form — see the table in the module doc.
struct Form {
    label: &'static str,
    scheme: QuantScheme,
    k: usize,
    /// Output columns the form's call dispatches. The widened form's `n` is its
    /// base width times [`Form::widen`].
    n: usize,
    /// Copies of a base geometry's packed bytes laid end to end, for the
    /// synthetic widening: the buffer a step reads is the base geometry's own
    /// packed bytes (`n / widen` columns) repeated `widen` times, contiguous —
    /// whole blocks, so it stays a valid weight whose row stride and block phase
    /// each step's region starts on. `1` for every form that is a model file's
    /// own tensor, `8` for the one that is not.
    widen: usize,
    steps: usize,
    packed_bytes_per_step: usize,
}

/// The forms measured: the K-quant arms' predictor geometry (`k = n = 2048`, the
/// narrowest call a 4-bit model's blocks issue, which its in-graph rate is set by)
/// and the Q8_0 rows at the widths the code predictor actually multiplies at.
const FORMS: [Form; 11] = [
    Form {
        label: "4-bit FFN gate",
        scheme: QuantScheme::GgufQ4K,
        k: 2048,
        n: 6144,
        widen: 1,
        steps: 32,
        packed_bytes_per_step: 7_077_888,
    },
    Form {
        label: "6-bit FFN down",
        scheme: QuantScheme::GgufQ6K,
        k: 6144,
        n: 2048,
        widen: 1,
        steps: 32,
        packed_bytes_per_step: 10_321_920,
    },
    Form {
        label: "4-bit FFN gate, widened x8",
        scheme: QuantScheme::GgufQ4K,
        k: 2048,
        n: 49152,
        widen: 8,
        steps: 8,
        packed_bytes_per_step: 56_623_104,
    },
    Form {
        label: "4-bit predictor block",
        scheme: QuantScheme::GgufQ4K,
        k: 2048,
        n: 2048,
        widen: 1,
        steps: 128,
        packed_bytes_per_step: 2_359_296,
    },
    Form {
        label: "4-bit predictor block, wide n",
        scheme: QuantScheme::GgufQ4K,
        k: 2048,
        n: 8192,
        widen: 1,
        steps: 32,
        packed_bytes_per_step: 9_437_184,
    },
    Form {
        label: "6-bit recognition head",
        scheme: QuantScheme::GgufQ6K,
        k: 2048,
        n: 151936,
        widen: 1,
        steps: 3,
        packed_bytes_per_step: 255_252_480,
    },
    // The code predictor's own storage and geometry: Q8_0 at `k = 1024` (its
    // `ffn_down` is `[3072, 1024]`, its head `[1024, 2048, 15]`), the narrowest
    // calls the synthesis cell issues. `n = 1024` is its o/down shape, `2048` its
    // head, `6144` its fused gate+up, and the widened form the same per-workgroup
    // work at a width no level of it issues — the anchor the others are read
    // against.
    Form {
        label: "8-bit predictor down",
        scheme: QuantScheme::GgufQ8_0,
        k: 1024,
        n: 1024,
        widen: 1,
        steps: 128,
        packed_bytes_per_step: 1_114_112,
    },
    Form {
        label: "8-bit predictor head",
        scheme: QuantScheme::GgufQ8_0,
        k: 1024,
        n: 2048,
        widen: 1,
        steps: 64,
        packed_bytes_per_step: 2_228_224,
    },
    Form {
        label: "8-bit predictor gate+up",
        scheme: QuantScheme::GgufQ8_0,
        k: 1024,
        n: 6144,
        widen: 1,
        steps: 32,
        packed_bytes_per_step: 6_684_672,
    },
    Form {
        label: "8-bit predictor block, k=2048",
        scheme: QuantScheme::GgufQ8_0,
        k: 2048,
        n: 2048,
        widen: 1,
        steps: 32,
        packed_bytes_per_step: 4_456_448,
    },
    Form {
        label: "8-bit predictor, widened x8",
        scheme: QuantScheme::GgufQ8_0,
        k: 1024,
        n: 49152,
        widen: 8,
        steps: 4,
        packed_bytes_per_step: 53_477_376,
    },
];

/// Timed rounds, and calls per round: the per-round figure is that round's own
/// mean per call, and the spread across rounds is reported as it stands rather
/// than smoothed away — at the narrow forms the fixed per-call floor is a large
/// part of the number, so run-to-run spread is information about the shape.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 3;

/// The form's graph: `steps` packed multiplies, each over its own parameter and
/// all of them over one input row.
fn form_graph(form: &Form, packed_bytes: usize) -> Graph {
    let mut g = Graph::new("wgpu_one_row_stream_rate");
    let x = g.input("x", Shape::new(&[1, form.k], DType::F32));
    let mut outs = Vec::with_capacity(form.steps);
    for s in 0..form.steps {
        let w = g.param(format!("w{s}"), Shape::new(&[packed_bytes], DType::U8));
        outs.push(g.add_node(
            Op::DequantMatMul {
                scheme: form.scheme,
            },
            vec![x, w],
            Shape::new(&[1, form.n], DType::F32),
        ));
    }
    g.set_outputs(outs);
    g
}

/// One call's steps share this row, so that the input-side term stays "one row
/// requested once per output-unit workgroup per step" — see the module doc.
fn input_row(k: usize) -> Vec<f32> {
    common::ramp(1, k, 0.017)
}

/// One form, built and dropped on its own — the module doc says why that is
/// load-bearing rather than tidy.
fn measure(form: &Form, dev: &rlx_wgpu::device::WgpuDevice) {
    // The form's transient device memory is its steps' own regions — one weight
    // parameter per step, which the plan places in the arena. The outputs are orders
    // smaller and are not in this number. Name the figure and skip a form that does
    // not fit rather than letting an allocation fail opaquely. This compares the
    // steps' total against a *per-buffer* limit, so it is a conservative aggregate
    // check rather than the exact one: a form the adapter could still place passes
    // it, a form it could not fails it.
    let transient = form.steps as u64 * form.packed_bytes_per_step as u64;
    let max_buffer = dev.device.limits().max_buffer_size;
    if transient > max_buffer {
        eprintln!(
            "[one-row packed stream] form=\"{}\" may not fit: its {} steps' regions total \
             {transient} bytes, over this adapter's per-buffer limit ({max_buffer}) — skipped \
             rather than allowed to fail opaquely",
            form.label, form.steps
        );
        return;
    }

    // The widened form's region is the base geometry's own packed bytes repeated
    // `widen` times into one contiguous buffer; every other form's regions are
    // its own geometry. Only the bytes are shared — each step still gets its own
    // parameter region.
    let packed = if form.widen > 1 {
        let base = common::packed_gguf_weight(form.scheme, form.k, form.n / form.widen);
        let mut widened = Vec::with_capacity(base.len() * form.widen);
        for _ in 0..form.widen {
            widened.extend_from_slice(&base);
        }
        widened
    } else {
        common::packed_gguf_weight(form.scheme, form.k, form.n)
    };
    assert_eq!(
        packed.len(),
        form.packed_bytes_per_step,
        "{}: the fixture must reproduce the form's tabulated packed bytes",
        form.label
    );

    let g = form_graph(form, packed.len());
    let mut c = Session::new(Device::Gpu).compile(g);
    for s in 0..form.steps {
        c.set_param_typed(&format!("w{s}"), &packed, DType::U8);
    }
    let x = input_row(form.k);

    let warm = c.run(&[("x", &x)]);
    assert_eq!(
        warm.len(),
        form.steps,
        "{}: one output per step",
        form.label
    );
    for (s, out) in warm.iter().enumerate() {
        assert_eq!(out.len(), form.n, "{}: step {s} output width", form.label);
        let non_finite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "{}: step {s} produced {non_finite} non-finite values — the fixture's \
             scale slots are not the finite patterns it writes",
            form.label
        );
    }

    // The census: which route, what shape, how many submissions and waits — one
    // call's own counts, taken outside the timing.
    reset_route_counts();
    reset_gemv_one_row_shape();
    reset_device_counts();
    let _ = c.run(&[("x", &x)]);
    let routes = route_counts();
    let shape = gemv_one_row_shape().expect("the one-row arm records its shape");
    let (submits, waits) = (device_submit_count(), device_wait_count());
    assert_eq!(
        (routes.per_row, routes.grouped, routes.host),
        (form.steps as u64, 0, 0),
        "{}: every step must take the one-row route: {routes:?}",
        form.label
    );
    // The join: this route is the one that rides the caller's own submission, so
    // one call's steps cost one submission and one blocking wait however many
    // dispatches and chunks they take. Asserted rather than only printed —
    // per-step submissions would substitute exactly the quantity this cell's
    // accounting forbids.
    assert_eq!(
        (submits, waits),
        (1, 1),
        "{}: one call must be one submission and one wait (its {} steps took \
         {submits} submissions and {waits} waits) — if this fails, the one-row \
         route has left the enclosing batch",
        form.label,
        form.steps
    );
    let dispatches =
        gemv_one_row_dispatches(&dev.device, gguf_scheme_id(form.scheme), form.k, form.n)
            .expect("the one-row arm's chunking rule answers for this scheme");

    // The timed rounds: this form alone, built and dropped on its own (the module
    // doc's form table), each round's figure the mean per call over its
    // `CALLS_PER_ROUND` calls.
    let per_call_ms = common::time_rounds(ROUNDS, CALLS_PER_ROUND, || {
        let _ = c.run(&[("x", &x)]);
    });
    let min_ms = per_call_ms.iter().cloned().fold(f64::MAX, f64::min);
    let med_ms = common::median(&per_call_ms);
    let max_ms = per_call_ms.iter().cloned().fold(0.0f64, f64::max);
    // The figure: one step's packed bytes over one step's share of the call.
    let rate = |call_ms: f64| {
        form.packed_bytes_per_step as f64 * form.steps as f64 / (call_ms * 1e-3) / 1e9
    };

    eprintln!(
        "[one-row packed stream] form=\"{}\" scheme={:?} k={} n={} steps={} \
         packed_bytes_per_step={} dispatches_per_step={} submits={submits} waits={waits} \
         cols_per_workgroup={} threads_per_workgroup={} workgroups={}",
        form.label,
        form.scheme,
        form.k,
        form.n,
        form.steps,
        form.packed_bytes_per_step,
        dispatches,
        shape.cols_per_workgroup,
        shape.threads_per_workgroup,
        shape.workgroups,
    );
    eprintln!(
        "[one-row packed stream] form=\"{}\" per_call_ms=[{}] min_ms={min_ms:.3} \
         median_ms={med_ms:.3} max_ms={max_ms:.3} spread_pct={:.1} min_GBps={:.1} \
         median_GBps={:.1}",
        form.label,
        per_call_ms
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        (max_ms - min_ms) / med_ms * 100.0,
        rate(min_ms),
        rate(med_ms),
    );
}

/// One test fn on purpose: what it reads is process-global (the route counters,
/// the recorded shape, the device submit/wait counters), and one form's
/// transient footprint must be gone before the next is built.
#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn one_row_packed_stream_rate() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    common::refuse_under_run_report();
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    eprintln!(
        "[one-row packed stream] adapter={:?} — this cell's figures hold for this adapter and \
         machine load only",
        rlx_wgpu::device::adapter_name()
    );
    for form in &FORMS {
        measure(form, dev);
    }
}
