// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A second compile for the same declared model skips re-uploading the weights the
//! first one placed in the shared weight buffer.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` parks every parameter in the weight buffer — the one
//! buffer the sharing is about — and the scope `set_param_sharing_scope` installs is a
//! process-global, so the cases here take turns behind one lock rather than running in
//! parallel. Each asks the same two questions around a second compile: what
//! `shared_weight_skip_stats` moved, and whether the second executable still computes
//! the first one's answer. The second question is the one that matters — a skip that
//! skipped bytes no compile wrote would read as a win on the counter alone.
//!
//! Every graph here is a set of `Cumsum`s over parked `F32` parameters, which is a
//! geometry whose route proves the parking: a parked operand sends `Op::Cumsum` to the
//! generic host route, so `ceiling_hosted_steps() >= params` says every parameter really
//! is where this test thinks it is — without it, "nothing was skipped" would also be
//! true of parameters that never reached the weight buffer at all.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::buffer::{set_param_sharing_scope, shared_weight_skip_stats};
use std::sync::{Mutex, MutexGuard};

const ROWS: usize = 4;
const COLS: usize = 64;

/// Bytes one parked parameter occupies in the weight buffer (four per element), and
/// so the figure the skip counter must move by when a compile skips it.
const PARAM_BYTES: u64 = (ROWS * COLS * 4) as u64;

/// The scope label, the sharing switch and the counters are all process-global, so
/// every case waits for this.
static EXCLUSIVE: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Parking is what puts a parameter in the shared weight buffer; the ambient switches
/// that could route `Cumsum` to a device step are neutralised so that parking alone
/// decides, exactly as the other parked-parameter cases do. Every key here is unset
/// again by [`release`], so a later case in this binary starts from the environment it
/// would have had.
const AMBIENT: [(&str, &str); 8] = [
    ("RLX_WGPU_FORCE_HOST", "0"),
    ("RLX_WGPU_DISCRETE_HOST", "0"),
    ("RLX_WGPU_HOST_NORM", "0"),
    ("RLX_WGPU_GPU_NORM", "0"),
    ("RLX_WGPU_SHARD_GPU", "0"),
    ("RLX_WGPU_ALLOW_SHARD", "0"),
    ("RLX_WGPU_SHARD_CAP_MIB", "1048576"),
    ("RLX_WGPU_ALL_PARAMS_WEIGHT", "1"),
];

fn park_every_param() {
    for (key, value) in AMBIENT {
        rlx_ir::env::set(key, value);
    }
}

/// Leave no scope behind for the next case, and let the process environment and the
/// sharing switch say what they said before.
fn release() {
    set_param_sharing_scope("");
    rlx_ir::env::unset("RLX_WGPU_SHARE_WEIGHTS");
    for (key, _) in AMBIENT {
        rlx_ir::env::unset(key);
    }
}

fn two_param_graph() -> Graph {
    let mut g = Graph::new("weight-scope");
    let shape = Shape::new(&[ROWS, COLS], DType::F32);
    let a = g.param("a", shape.clone());
    let ca = g.cumsum(a, -1, false, shape.clone());
    let b = g.param("b", shape.clone());
    let cb = g.cumsum(b, -1, false, shape);
    g.set_outputs(vec![ca, cb]);
    g
}

fn param_data(base: f32) -> Vec<f32> {
    param_data_rows(base, ROWS)
}

/// One parameter of `rows` rows, so a case can use a length the other layout does not
/// (`2 * ROWS` for a fused projection, `3 * ROWS` for a longer name).
fn param_data_rows(base: f32, rows: usize) -> Vec<f32> {
    (0..rows * COLS)
        .map(|i| base + (i % 16) as f32 * 0.25)
        .collect()
}

/// What `exe.run(&[])` must answer: the running sum of `data` along the last axis.
fn running_sum(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..data.len() / COLS {
        let mut acc = 0.0f32;
        for c in 0..COLS {
            acc += data[r * COLS + c];
            out[r * COLS + c] = acc;
        }
    }
    out
}

/// A graph of one hosted `Cumsum` per `(name, rows)` parameter. The route proves the
/// parking: a parked operand sends `Op::Cumsum` to the generic host route, so
/// `ceiling_hosted_steps() >= params.len()` says every parameter really is where this
/// test thinks it is — without it, "nothing was re-uploaded" would also be true of
/// parameters that never reached the weight buffer at all.
fn cumsum_graph(params: &[(&str, usize)]) -> Graph {
    let mut g = Graph::new("weight-scope-layout");
    let mut outputs = Vec::with_capacity(params.len());
    for (name, rows) in params {
        let shape = Shape::new(&[*rows, COLS], DType::F32);
        let param = g.param(*name, shape.clone());
        outputs.push(g.cumsum(param, -1, false, shape));
    }
    g.set_outputs(outputs);
    g
}

/// Compile `graph`, upload `data`, then run. Answers the skipped-byte/write figures
/// those uploads moved — measured around the uploads only — and the executable with its
/// outputs.
fn compile_graph_upload_and_run(
    graph: Graph,
    data: &[(&str, Vec<f32>)],
) -> (u64, u64, WgpuExecutable, Vec<Vec<f32>>) {
    let mut exe = WgpuExecutable::compile(graph);
    assert!(
        exe.ceiling_hosted_steps() >= data.len(),
        "every parked parameter must reach the host route ({} hosted, {} parameters) — \
         without that this case is not about the weight buffer",
        exe.ceiling_hosted_steps(),
        data.len()
    );
    let before = shared_weight_skip_stats();
    for (name, values) in data {
        exe.set_param(name, values);
    }
    let after = shared_weight_skip_stats();
    let out = exe.run(&[]);
    (after.0 - before.0, after.1 - before.1, exe, out)
}

/// One compile of the graph, both weights uploaded, then a run. Answers the
/// skipped-byte/write figures those uploads moved — measured around the uploads only,
/// so the compile itself cannot be credited with them — and the two outputs.
fn compile_upload_and_run() -> (u64, u64, Vec<Vec<f32>>) {
    let mut exe = WgpuExecutable::compile(two_param_graph());
    assert!(
        exe.ceiling_hosted_steps() >= 2,
        "both parked parameters must reach the host route ({} hosted) — without that \
         this case is not about the weight buffer",
        exe.ceiling_hosted_steps()
    );
    let before = shared_weight_skip_stats();
    exe.set_param("a", &param_data(0.0));
    exe.set_param("b", &param_data(1.0));
    let after = shared_weight_skip_stats();
    (after.0 - before.0, after.1 - before.1, exe.run(&[]))
}

/// The two answers a correct compile must give, on this test's data.
fn want_outputs() -> Vec<Vec<f32>> {
    vec![running_sum(&param_data(0.0)), running_sum(&param_data(1.0))]
}

#[test]
fn a_second_compile_of_the_same_model_skips_every_weight_write() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("weight-scope/same-model");

    let (skipped, writes, first) = compile_upload_and_run();
    assert_eq!(
        (skipped, writes),
        (0, 0),
        "the first compile for a scope has nothing in the shared buffer to skip"
    );
    assert_eq!(first, want_outputs(), "the first executable must compute");

    let (skipped, writes, second) = compile_upload_and_run();
    assert_eq!(
        skipped,
        2 * PARAM_BYTES,
        "the second compile must skip exactly the two parameters' bytes"
    );
    assert_eq!(writes, 2, "one skipped write per parameter");
    assert_eq!(
        second,
        want_outputs(),
        "the second executable must read the bytes the first compile placed — a skip \
         that left them unwritten would compute something else"
    );

    release();
}

#[test]
fn a_different_scope_label_uploads_the_weights_again() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("weight-scope/model-one");
    let (skipped, _, _) = compile_upload_and_run();
    assert_eq!(skipped, 0, "the first model's compile has nothing to skip");

    // Another model: same layout, same parameter names, different weights. Nothing
    // declares them the same, so its compile must upload.
    set_param_sharing_scope("weight-scope/model-two");
    let (skipped, writes, out) = compile_upload_and_run();
    assert_eq!(
        (skipped, writes),
        (0, 0),
        "a different label is a different model: its parameters must be uploaded"
    );
    assert_eq!(out, want_outputs(), "and computed from its own upload");

    release();
}

#[test]
fn with_no_scope_declared_no_weight_write_is_skipped() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("");

    let (skipped, _, _) = compile_upload_and_run();
    assert_eq!(skipped, 0, "no scope: the first compile uploads");
    let (skipped, writes, out) = compile_upload_and_run();
    assert_eq!(
        (skipped, writes),
        (0, 0),
        "without a declared scope nothing says the two compiles are one model, so \
         nothing may be skipped even though the buffer is shared"
    );
    assert_eq!(out, want_outputs(), "and every compile must still compute");

    release();
}

#[test]
fn the_sharing_switch_off_skips_nothing_even_with_a_scope() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    rlx_ir::env::set("RLX_WGPU_SHARE_WEIGHTS", "0");
    set_param_sharing_scope("weight-scope/no-sharing");

    let (skipped, _, _) = compile_upload_and_run();
    assert_eq!(skipped, 0, "sharing off: the first compile uploads");
    let (skipped, writes, out) = compile_upload_and_run();
    assert_eq!(
        (skipped, writes),
        (0, 0),
        "`RLX_WGPU_SHARE_WEIGHTS=0` turns sharing and the skip off together, so a \
         scope cannot make a write skippable"
    );
    assert_eq!(
        out,
        want_outputs(),
        "and each compile uploads its own bytes"
    );

    release();
}

/// Clearing the shared weight buffer is what makes a cold measurement cold: the next
/// compile of the same model uploads its weights again instead of skipping every write
/// a compile from an earlier call left behind.
#[test]
fn clearing_the_shared_buffer_makes_the_next_compile_upload_again() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("weight-scope/cold-start");
    let _ = compile_upload_and_run();
    let (skipped, _, _) = compile_upload_and_run();
    assert_eq!(
        skipped,
        2 * PARAM_BYTES,
        "the buffer holds the model's weights"
    );

    rlx_wgpu::buffer::clear_shared_weights();
    let (skipped, writes, out) = compile_upload_and_run();
    assert_eq!(
        (skipped, writes),
        (0, 0),
        "an emptied buffer has nothing to skip, so the compile pays the upload"
    );
    assert_eq!(out, want_outputs(), "and the executable still computes");

    release();
}

/// A layout that spells one projection as a fused tensor and the next as two unfused
/// ones — the shape a prefill/decode pair of one body has — must keep every name the
/// two layouts *share* at the offset the first compile gave it. The fused tensor is a
/// different name, so its bytes stay resident and unreferenced; what must not happen is
/// the rest of the body being allocated and uploaded a second time.
#[test]
fn a_fused_projection_spelling_does_not_re_upload_the_shared_parameters() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("weight-scope/fused-spelling");

    let a = param_data(0.0);
    let b = param_data(1.0);
    let fused = param_data_rows(2.0, 2 * ROWS);
    let (skipped, _, mut first, first_out) = compile_graph_upload_and_run(
        cumsum_graph(&[("a", ROWS), ("b", ROWS), ("blk.0.gate_up.fused", 2 * ROWS)]),
        &[
            ("a", a.clone()),
            ("b", b.clone()),
            ("blk.0.gate_up.fused", fused.clone()),
        ],
    );
    assert_eq!(
        skipped, 0,
        "the first compile for a scope has nothing to skip"
    );
    assert_eq!(
        first_out,
        vec![running_sum(&a), running_sum(&b), running_sum(&fused)],
        "the first executable must compute from its own upload"
    );

    let gate = param_data_rows(3.0, ROWS);
    let up = param_data_rows(4.0, ROWS);
    let (skipped, writes, _second, second_out) = compile_graph_upload_and_run(
        cumsum_graph(&[
            ("a", ROWS),
            ("b", ROWS),
            ("blk.0.ffn_gate.weight", ROWS),
            ("blk.0.ffn_up.weight", ROWS),
        ]),
        &[
            ("a", a.clone()),
            ("b", b.clone()),
            ("blk.0.ffn_gate.weight", gate.clone()),
            ("blk.0.ffn_up.weight", up.clone()),
        ],
    );
    assert_eq!(
        (skipped, writes),
        (2 * PARAM_BYTES, 2),
        "the two names both layouts hold must ride the first compile's bytes; only the \
         two names the fused spelling replaces are new, and they are one upload each"
    );
    assert_eq!(
        second_out,
        vec![
            running_sum(&a),
            running_sum(&b),
            running_sum(&gate),
            running_sum(&up),
        ],
        "the second executable must read the bytes the first compile placed for the \
         shared names, and its own bytes for the appended ones"
    );

    // The first executable is still live. The second compile appended a shard and wrote
    // only offsets the first one never binds, so re-running it must answer what it
    // answered before — a scheme that moved bytes, or reused an offset for a different
    // tensor, would show here.
    assert_eq!(
        first.run(&[]),
        first_out,
        "the first executable's bytes must not have been moved or overwritten"
    );

    release();
}

/// A name the entry does not hold yet — longer than any parameter of the first layout —
/// is appended at an offset of its own and computes; and the same name with a different
/// length is a *different* placement, not a silent alias of the shorter or longer one.
#[test]
fn a_new_or_longer_parameter_is_appended_and_still_computes() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _exclusive = exclusive();
    park_every_param();
    set_param_sharing_scope("weight-scope/growing-layout");

    let a = param_data(0.0);
    let b = param_data(1.0);
    let (skipped, _, _first, _) = compile_graph_upload_and_run(
        cumsum_graph(&[("a", ROWS), ("b", ROWS)]),
        &[("a", a.clone()), ("b", b.clone())],
    );
    assert_eq!(skipped, 0, "the first compile has nothing to skip");

    let long = param_data_rows(2.0, 3 * ROWS);
    let (skipped, writes, mut longer, longer_out) = compile_graph_upload_and_run(
        cumsum_graph(&[("a", ROWS), ("b", ROWS), ("c.long", 3 * ROWS)]),
        &[("a", a.clone()), ("b", b.clone()), ("c.long", long.clone())],
    );
    assert_eq!(
        (skipped, writes),
        (2 * PARAM_BYTES, 2),
        "the shared names ride the first compile's bytes; the longer name is new"
    );
    assert_eq!(
        longer_out,
        vec![running_sum(&a), running_sum(&b), running_sum(&long)],
        "the longer layout must compute from the shared bytes and its own"
    );

    // Same name, shorter length: (name, byte length) is the identity, so this is a new
    // placement — keying on the name alone would bind it to the 3-row bytes and both
    // the skip count and the answer would be wrong.
    let short = param_data_rows(3.0, ROWS);
    let (skipped, writes, _, short_out) = compile_graph_upload_and_run(
        cumsum_graph(&[("a", ROWS), ("b", ROWS), ("c.long", ROWS)]),
        &[
            ("a", a.clone()),
            ("b", b.clone()),
            ("c.long", short.clone()),
        ],
    );
    assert_eq!(
        (skipped, writes),
        (2 * PARAM_BYTES, 2),
        "a shorter tensor of the same name is not the longer one: only `a` and `b` are \
         shared bytes"
    );
    assert_eq!(
        short_out,
        vec![running_sum(&a), running_sum(&b), running_sum(&short)],
        "and it must compute from its own upload, not from the longer bytes"
    );

    // The longer layout's executable is still live and must keep its answer.
    assert_eq!(
        longer.run(&[]),
        longer_out,
        "appending a different length must not move the bytes the longer executable binds"
    );

    release();
}
