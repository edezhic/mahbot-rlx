// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The loop's shape, as counters rather than times: how a carried cache is
//! written, how many times a step returns to the host, and how the prompt's
//! packed weights are re-read.
//!
//! Every test whose name ends in `_with_models` needs the model files: without
//! `MAHBOT_MODELS_DIR` `pair` prints why and the body does not run. That is a
//! skip by environment — this checkout carries no models, so a host without them
//! cannot execute the gate at all — the same class as
//! `rlx_ir::env::skip_unless_device`. The run-report switch is the opposite case:
//! it is a switch the operator sets, so `support::assert_run_report_off` refuses
//! instead of skipping. The model-free guards live in this file and
//! `packed_join_submission.rs`, so a gate run without models still executes them.

use std::sync::Arc;

use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::reference::LISTEN_PROMPT_ROWS;
use rlx_flow::BuiltModel;
use rlx_ir::Op;

mod support;

fn pair(role: Role) -> Option<Arc<Pair>> {
    let Ok(dir) = loader::models_dir() else {
        eprintln!(
            "skip: {} is not set — the graph shapes need a model",
            loader::MODELS_DIR_ENV
        );
        return None;
    };
    // An incomplete pair is the same environment skip as no directory at all: a
    // checkout does not carry the models, so this gate says it did not run.
    match Pair::of_role(role) {
        Ok(pair) => Some(Arc::new(pair)),
        Err(err) => {
            eprintln!(
                "skip: {} names no pair for this role ({err}) — the shape gate did not run",
                dir.display()
            );
            None
        }
    }
}

fn graph_of(built: BuiltModel) -> rlx_ir::Graph {
    let hir = built.module.into_hir().expect("the flow built HIR");
    rlx_ir::hir_to_graph(hir).expect("HIR lowers to a graph")
}

fn census(graph: &rlx_ir::Graph) -> (usize, usize, usize) {
    let (mut append, mut transpose, mut concat) = (0, 0, 0);
    for node in graph.nodes() {
        match node.op {
            Op::KvAppend { .. } => append += 1,
            Op::Transpose { .. } => transpose += 1,
            Op::Concat { .. } => concat += 1,
            _ => {}
        }
    }
    (append, transpose, concat)
}

fn input_names(graph: &rlx_ir::Graph) -> Vec<String> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// A recognition decode body writes its cache with `KvAppend` in the attention's
/// own row layout: two appends per layer, no whole-cache transpose and no
/// concatenation. That is the fact the loop's submissions-per-token count rests
/// on — a transpose or a concat here is one more dispatch per layer.
#[test]
fn a_recognition_decode_body_carries_its_cache_without_a_transpose_or_concat_with_models() {
    let Some(pair) = pair(Role::Recognition) else {
        return;
    };
    let layers = pair.body.config().layers;
    let built = graphs::body_capacity_chain_planes(pair.body.clone(), 2_438)
        .expect("build the plane-cache decode body");
    let graph = graph_of(built);
    assert_eq!(
        census(&graph),
        (2 * layers, 0, 0),
        "one `KvAppend` for K and one for V per layer, and nothing else moving \
         the cache"
    );
    let names = input_names(&graph);
    for gi in 0..layers {
        for plane in ["k", "v"] {
            let want = format!("{plane}{gi}");
            assert!(
                names.iter().any(|n| n == &want),
                "the body takes `{want}` as its own `[capacity + 1, hkv]` cache"
            );
        }
    }
    // `hidden` is the main output, then each layer's K row, its V row, and the
    // logits last: the one tensor a step reads back. A step that read its cache
    // rows back too would return `2 · layers` more tensors to the host.
    assert_eq!(graph.outputs.len(), graphs::plane_step_outputs(layers));
}

/// The synthesis body is the same shape at its own width: one appends pair per
/// layer, no transpose and no concat.
#[test]
fn the_synthesis_body_carries_its_cache_without_a_transpose_or_concat_with_models() {
    let Some(pair) = pair(Role::Synthesis) else {
        return;
    };
    let layers = pair.body.config().layers;
    let built = graphs::body_capacity_chain_planes(pair.body.clone(), 370)
        .expect("build the plane-cache body");
    let graph = graph_of(built);
    assert_eq!(census(&graph), (2 * layers, 0, 0));
    assert_eq!(graph.outputs.len(), graphs::plane_step_outputs(layers));
}

/// One predictor frame is **one fence** and one submission per encode chunk of
/// the sixteen-level walk, and its cache stays in the arena. The frame graph is
/// the live walk, so these are its pins: two inputs (`hidden`, `code0_embed`)
/// and two outputs (the drawn codes, then level 1's logits for the stand's
/// sampler check), one `Op::Sample` draw per projecting level, no whole-cache
/// transpose, one in-place `KvAppend` per block and cache at every projecting
/// level (level 0 seeds each block's plane with a concat instead), and no
/// layer-axis narrow. Both counts are read from the backend's process-global
/// `device_submit_count` / `device_wait_count` after the compile and upload,
/// around two frames of the compiled graph.
#[test]
fn a_predictor_frame_is_one_fence_and_one_submission_per_encode_chunk_with_models() {
    // `RLX_WGPU_DBG_COUNTERS` changes the submission/wait shape of a forward pass.
    support::assert_run_report_off();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair(Role::Synthesis) else {
        return;
    };
    let steps = graphs::code_predictor_steps(&pair.sound).expect("predictor head count");
    let capacity = steps + 1;
    let (d_talker, _) = pair.sound.code_proj_widths().expect("predictor widths");
    let (top_k, top_p, temperature) = pair.body.sampling();
    let mut frame = graphs::Compiled::new(
        rlx_runtime::Device::Gpu,
        graphs::tts_code_predictor_frame(
            pair.sound.clone(),
            capacity,
            graphs::SampleSpec {
                top_k,
                top_p,
                temperature,
                seed: 12345,
            },
        )
        .expect("build the frame graph"),
    )
    .expect("compile the predictor frame");
    let graph = &frame.runner.graph;
    assert_eq!(
        input_names(graph),
        ["hidden", "code0_embed"],
        "one hidden row and one code0 row in; no cache and no per-level mask"
    );
    assert_eq!(
        graph.outputs.len(),
        2,
        "the drawn codes, then level 1's logits for the stand's sampler check"
    );
    let draws = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::Sample { .. }))
        .count();
    assert_eq!(
        draws, steps,
        "one device draw per projecting level, no draw on level 0"
    );
    // The cache is already row-first, which is the sequence-major layout the
    // attention reads, so nothing moves it whole: every projecting level writes
    // one row per block into that block's plane, in place.
    let (blocks, _, _) = graphs::code_predictor_cache(&pair.sound).expect("predictor cache shape");
    assert_eq!(
        census(graph).0,
        2 * blocks * steps,
        "one `KvAppend` per block and cache at every projecting level"
    );
    let cache_transposes = graph
        .nodes()
        .iter()
        .filter(|node| matches!(&node.op, Op::Transpose { perm } if perm.as_slice() == [1, 0, 2]))
        .count();
    assert_eq!(cache_transposes, 0, "no whole-cache transpose");
    assert_eq!(
        census(graph).2,
        2 * blocks + 1,
        "each block's two planes are seeded once at level 0, plus the drawn-code \
         concat; no level rebuilds a cache by concatenation"
    );
    let layer_narrows = graph
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                &node.op,
                Op::Narrow {
                    axis: 1,
                    len: 1,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        layer_narrows, 0,
        "the cache is appended in place, not re-sliced per level"
    );
    let seeds: Vec<u64> = (0..steps as u64).map(|k| 12345 + k).collect();
    assert!(
        frame.runner.set_sample_seeds(&seeds),
        "the frame graph has one seed per projecting level"
    );

    // The counters are process-global, so reset them after the compile and the
    // upload: what is left is the walk's own submissions.
    rlx_wgpu::reset_device_counts();
    for _ in 0..2 {
        let out = frame.runner.run_read(
            &[
                ("hidden".into(), vec![0f32; d_talker]),
                ("code0_embed".into(), vec![0f32; d_talker]),
            ],
            &[0],
        );
        assert_eq!(out.len(), 1, "one returned tensor: the drawn codes");
        assert_eq!(out[0].len(), steps, "one code per projecting level");
    }
    assert_eq!(
        rlx_wgpu::device_wait_count(),
        2,
        "and one wait per frame: no chunk is fenced, and no submission carries only a \
         cache row"
    );
    // Submissions are the encode chunks, not the frame and not the multiplies: the
    // backend hands one encoder on every `RLX_WGPU_ENCODE_CHUNK` steps so that a
    // chunk's execution overlaps the next one's encode (`run.rs`'s own doc has the
    // measurement). One per frame would expose the whole 1 465-step encode with the
    // device idle; one per multiply (348 a frame) is the shape the join removed.
    let per_frame = rlx_wgpu::device_submit_count() / 2;
    assert!(
        (2..=64).contains(&per_frame),
        "one submission per encode chunk per frame, not one per frame (1) and not one \
         per packed multiply (348): got {per_frame}"
    );
}

/// The 234-row prompt graph is the graph the model-free row-tile guard is about:
/// every packed multiply contracts all prompt rows at once, so the dispatch's
/// tile count is the prompt's, not a chunk's.
#[test]
fn the_recognition_prompt_graph_contracts_all_its_rows_with_models() {
    let Some(pair) = pair(Role::Recognition) else {
        return;
    };
    let graph = graph_of(
        graphs::body_new_rows_head_chain(pair.body.clone(), LISTEN_PROMPT_ROWS, 0, true)
            .expect("build the 234-row prompt graph"),
    );
    let packed: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::DequantMatMul { .. }))
        .collect();
    assert!(
        !packed.is_empty(),
        "the prompt's weights are the packed ones the grouped arm serves"
    );
    let mut over_prompt = 0usize;
    for node in packed {
        let rows = graph.node(node.inputs[0]).shape.dims()[0].unwrap_static();
        assert!(
            rows == LISTEN_PROMPT_ROWS || rows == 1,
            "a packed multiply contracts either the whole prompt or the head's last row"
        );
        over_prompt += usize::from(rows == LISTEN_PROMPT_ROWS);
    }
    assert!(
        over_prompt > 0,
        "the block multiplies contract all prompt rows, so the row-tile guard applies"
    );
}

/// The recognition prompt's row tiles, without a model or a device: how the
/// pinned prompt maps to the row-grouped GEMM's tiles.
///
/// The count comes from the backend's own
/// [`rlx_wgpu::gguf_gpu::grouped_base_geometry`] rather than a literal here, so
/// the assertion moves only with the kernel's geometry: the 234-row prompt over
/// the lowerer's eight-row tiles is 30 dispatches' worth of row groups.
#[test]
fn the_recognition_prompt_is_thirty_grouped_row_tiles() {
    let (_, rows) = rlx_wgpu::gguf_gpu::grouped_base_geometry(LISTEN_PROMPT_ROWS, 64);
    assert_eq!(
        rows, 30,
        "234 prompt rows over the lowerer's eight-row tiles"
    );
}
