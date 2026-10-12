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
//! # The four forms
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
//! | 6-bit recognition head | Q6_K | 2048 | 151936 | 3 | 1680 | 255,252,480 |
//!
//! The widened form's region is the first form's own packed bytes repeated eight
//! times into one buffer — a synthetic widening, no such tensor being in any file —
//! and its eight regions are byte-identical copies of it, so the first form's bytes
//! appear sixty-four times in that call's address space. The narrowest form is the
//! most floor-bound and the noisiest, so a conclusion resting on it alone is labelled
//! as such; the second is the only marginal-occupancy one, having by far the fewest
//! output columns. Each form is built and dropped on its own — four at once is the
//! order of two gigabytes of transient device memory — and a form whose transient
//! bytes exceed the adapter's `max_buffer_size` is **named as not fitting and
//! skipped**, never silently dropped, with the forms that do fit still measured.
//!
//! # What the arm does now, and what did not change
//!
//! A workgroup's two columns share one `vec4` read of the input row (**input-side**,
//! halved — a gain here is **not** a weight-read win); Q6_K's lane unit requests each
//! byte of a `ql` span once instead of twice; Q4_K keeps its sub-block cover (see the
//! shader header) but reads its 16-byte header as four whole words, a request-volume
//! increase confined to the block that unit already reads; and Q4_K's per-unit decode
//! against a `qs` group's bytes being requested by both sibling sub-blocks is
//! **unattributed** (one term, not two).
//!
//! Unchanged: the host work *per chunk* (one uniform write and one bind group,
//! nothing pooled); one submission and one wait per call, so the arm still rides the
//! enclosing batch; and the fixed call floor, *quantified rather than cancelled* — a
//! development-time step-count sweep found the narrow form's per-step time falling
//! steeply as its call's step count rose (a steps-independent per-call cost of the
//! order of a third of that 32-step call), which is why the narrow form's ratio is
//! the one that does not reproduce. The one per-step figure that moved is the
//! recognition head's derived dispatch count, 3 -> 2 — where the binding budget does
//! not bind the chunk width (it does not at these forms' row widths): the cap counts
//! workgroups, so two columns per workgroup covers twice the columns, i.e. one chunk
//! fewer and so one fewer uniform write and bind group per step on that call. The
//! widened form is one dispatch per step either way.
//!
//! # Asserted, and not
//!
//! Asserted, never a time: the tabulated packed bytes per step; the joined one-row
//! route for every step (the route counter conflates it with the multi-row per-row
//! fallback, so this census is what ties the figures to this arm), in one submission
//! with one blocking wait; finite outputs. Printed: the figure, the derived per-step
//! dispatch count ([`gemv_one_row_dispatches`], from the arm's chunking rule), the
//! submission/wait structure, the recorded one-row shape, and the per-round times.
//!
//! Not in this cell's sight, and so not to be read as checked: the shared shader's
//! byte-wise helper reads, which the legacy one-bit arm uses — **uninstrumented**;
//! the host-side wiring this route shares with that arm and with the multi-row
//! per-row fallback (which re-enters this arm one row at a time), so a wiring change
//! reaches further than the kernel does; the grouped multi-row route (which
//! duplicates this arm's unpack arithmetic rather than sharing it) and that fallback
//! — both re-taken by the owner's rig, not here; the comparable dense read, which
//! lives on that rig too, so only absolute figures are recorded here; and any
//! adapter but the one printed.
//!
//! # Recorded figures
//!
//! This accounting's bytes per second, both sides through this cell on the two
//! revisions either side of the one-row arm's change; adapter `Apple M3 Pro`,
//! `--release`, no other workload running; each figure that side's better run, a round
//! contributing the mean per call over `CALLS_PER_ROUND` calls, and medians are never
//! paired with mins.
//!
//! | form | before, min (median) | after, min (median) | ratio, min-for-min |
//! | --- | --- | --- | --- |
//! | 4-bit FFN gate | 50.7 (45.8) | 55.2 (51.3) | 1.09× |
//! | 6-bit FFN down | 50.4 (49.2) | 59.6 (57.8) | 1.18× |
//! | 4-bit FFN gate, widened | 65.7 (63.9) | 77.1 (75.9) | 1.17× |
//! | 6-bit recognition head | 62.3 (62.2) | 83.1 (80.7) | 1.33× |
//!
//! GB/s, one point in time and not a reproducible number — the modified side's later
//! runs have read 54.6-55.3, 57.6-61.3, 77.7-79.0 and 82.0-84.4, so a re-run **moves
//! this table deliberately** — and **cross-revision by construction**: two runs on
//! two revisions, fixed order, no in-run control, no interleaving, so each side's own
//! spread is the only bound on the ratio. The development-time interleaved measurement
//! the claim rests on (both arms round-robin in one process, the unmodified arm kept
//! alive as a duplicate control) is deliberately not landed, so **the tree cannot
//! re-derive the pairing** and the before side exists only here.
//!
//! **The narrowest form does not reproduce to within its own spread** (47.4…57.8 GB/s
//! across runs), so its 1.09× ratio is inside the noise, nothing here rests on it, and
//! it is the form the floor dominates; the three wider forms have stayed above their
//! `before` sides in every run so far, which is the change's support — run-to-run
//! evidence under a varying machine load, not an interleaved pair.

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

const FORMS: [Form; 4] = [
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
        label: "6-bit recognition head",
        scheme: QuantScheme::GgufQ6K,
        k: 2048,
        n: 151936,
        widen: 1,
        steps: 3,
        packed_bytes_per_step: 255_252_480,
    },
];

/// Timed rounds, and calls per round: the per-round figure is that round's own
/// mean per call, and the spread across rounds is reported as it stands rather
/// than smoothed away — at the narrow forms the fixed per-call floor is a large
/// part of the number, so run-to-run spread is information about the shape.
const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 3;

/// The form's packed weight: the crate's shared fixture at this scheme's own block
/// geometry, narrowed to the two schemes this arm has an entry point for — the
/// shared fixture also accepts Q8_0, which has none.
fn packed_weights(scheme: QuantScheme, k: usize, n: usize) -> Vec<u8> {
    assert!(
        matches!(scheme, QuantScheme::GgufQ4K | QuantScheme::GgufQ6K),
        "{scheme:?} has no one-row arm"
    );
    common::packed_gguf_weight(scheme, k, n)
}

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
        let base = packed_weights(form.scheme, form.k, form.n / form.widen);
        let mut widened = Vec::with_capacity(base.len() * form.widen);
        for _ in 0..form.widen {
            widened.extend_from_slice(&base);
        }
        widened
    } else {
        packed_weights(form.scheme, form.k, form.n)
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
    // doc's "The four forms"), each round's figure the mean per call over its
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
