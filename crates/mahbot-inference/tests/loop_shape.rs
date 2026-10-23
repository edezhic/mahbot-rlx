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

/// The schedule pins a plane-cache frame must keep: no `buffer_copy` and no
/// `host_op`. Both are asserted on the kinds the runtime's compile produced, with
/// the append census printed beside them so a frame that stopped writing its cache
/// through `KvAppend` cannot satisfy the pins vacuously.
fn assert_no_row_copy(names: &[&'static str], appends: usize, frame: &str) {
    let copies = names.iter().filter(|n| **n == "buffer_copy").count();
    let hosted = names.iter().filter(|n| **n == "host_op").count();
    println!(
        "census {frame} schedule_steps={} buffer_copy={copies} host_op={hosted} appends={appends}",
        names.len()
    );
    assert_eq!(
        copies, 0,
        "the {frame} carries {appends} appends, and each must write its row in place \
         rather than into a slot and then copy it: {names:?}"
    );
    // A copy is not the only way to lose the row's placement: an append routed to
    // the host has no copy step either, and it returns the frame to the slower
    // path all the same.
    assert_eq!(
        hosted, 0,
        "the {frame} has no host step of its own, so a `host_op` here is an append \
         (or a step reading its row) that left the device path: {names:?}"
    );
}

/// The synthesis body's schedule carries **no `buffer_copy` step and no
/// `host_op`**: each `KvAppend` writes its row where the attention reads it
/// (`Arena::finish_plan`'s placement), so the frame has no copy dispatch and no
/// second pass over the arena — and it stays on the device, which a copy-only pin
/// cannot tell apart from an append the host ran. Pinned on the real body graph
/// **as the product
/// compiles it** — `Session::compile` through the wgpu backend, fusion and all —
/// because the placement is a plan-time decision, and a regression that keeps the
/// copy would otherwise show up only as a slower frame, never as a failure.
#[test]
fn the_synthesis_body_schedule_has_no_row_copy_with_models() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair(Role::Synthesis) else {
        return;
    };
    let layers = pair.body.config().layers;
    let built = graphs::body_capacity_chain_planes(pair.body.clone(), 370)
        .expect("build the plane-cache body");
    let graph = graph_of(built);
    // Non-vacuous: the row-copy question only exists while the body writes its
    // cache through appends.
    let appends = census(&graph).0;
    assert_eq!(appends, 2 * layers);
    let session = rlx_runtime::Session::new(rlx_runtime::Device::Gpu).compile(graph);
    let names = session
        .schedule_step_names()
        .expect("the wgpu backend reports its schedule");
    assert_no_row_copy(&names, appends, "body");
    // Every layer's q and k norm+rope pairs are one step each, not four: the
    // fold that keeps the body's dispatch count off the one-row chain.
    assert_eq!(
        names.iter().filter(|n| **n == "norm_rope").count(),
        layers,
        "one fused norm+rope step per layer"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "rope").count(),
        0,
        "no rope keeps a dispatch of its own"
    );
}

/// One predictor frame is **one fence** and one submission per encode chunk of
/// the sixteen-level walk, and its cache stays in the arena. The frame graph is
/// the live walk, so these are its pins: two inputs (`hidden`, `code0_embed`)
/// and two outputs (the drawn codes, then level 1's logits for the stand's
/// sampler check), one `Op::Sample` draw per projecting level, no whole-cache
/// transpose, one in-place `KvAppend` per block and cache at every projecting
/// level (level 0 seeds each block's plane with a concat instead), and no
/// layer-axis narrow. Its schedule carries no `buffer_copy` and no `host_op`
/// either: each append already stands at the row its attention reads. The two
/// counters are read from the backend's process-global
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
    // The rows the appends write are the rows the attention reads: no copy step,
    // and no host step standing in for one, on the frame the product compiles.
    let names = frame
        .runner
        .schedule_step_names()
        .expect("the wgpu backend reports its schedule");
    assert_no_row_copy(&names, census(graph).0, "predictor frame");
    // The fused qkv's row is cut and its q/k head norms and rotations ride in the
    // same dispatch — one step a block-level where the decompose emits seven.
    let count = |name: &str| names.iter().filter(|n| **n == name).count();
    // Every block-level's q/k norm+rope pair is one step.
    assert_eq!(
        count("norm_rope"),
        blocks * (steps + 1),
        "one fused norm+rope step per block-level"
    );
    assert_eq!(count("rope"), 0, "no rope keeps a dispatch of its own");
    // Level 0's project-in reads two of its three parts, so that one keeps its
    // two `narrow`s; every other split is absorbed by the pair it feeds.
    assert_eq!(
        count("narrow_split"),
        0,
        "the split that feeds a pair rides in the pair's dispatch"
    );
    assert_eq!(count("narrow"), 2, "the one partial split stays decomposed");
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

/// The code-predictor frame's multiplies, counted and priced — the figures the
/// frame census is reconciled against, printed rather than asserted because they
/// are a census of one model's graph. Ignored by default and skipped without
/// `MAHBOT_MODELS_DIR`, like every `_with_models` case here.
///
/// It prints what the census needs and none of which can be inferred from
/// another: the multiply sites the graph has (packed and f32 apart), the ones that
/// are **dead** — no output of the graph reads them, found here by a backward walk
/// from the graph's own outputs — the `dequant_matmul_gguf` steps the backend's
/// lowering emits for the graph, and the parameter bytes each site reads, totalled
/// and as a tensor × byte × count table, beside the graph's own parameter-byte sum.
///
/// On the models `MEASUREMENTS.md` names the reconciliation closes: 351 packed
/// sites and 80 block instances (5 blocks × the frame's 16 levels: level 0 seeds
/// the cache and draws nothing, each of levels 1..=15 appends a row and projects);
/// 3 of those sites are dead — `blk.4`'s `attn_out`, `gate_up` and `ffn_down` at
/// level 0, whose row nothing reads because level 1 embeds the drawn code rather
/// than the hidden, while the block's `qkv` stays live as the cache seed — and the
/// live 348 is exactly what the cell dispatches. The backend lowering of the flow
/// graph still emits all 351 steps, so the other half of the method is the cell's
/// own schedule: `RLX_WGPU_SCHEDULE=1 ./target/release/mahbot-inference 5` prints
/// one 1 465-step graph with `348 × dequant_matmul_gguf`, `79 × attention` and
/// `79 × fused_swiglu` — the same one instance short, dropped as dead work rather
/// than shared or reused.
///
/// The compile is `WgpuExecutable::compile` on the graph the flow built, which is
/// the graph the cell's `Runner` holds; the runtime's pre-passes (the cleanup and
/// the fusions) run inside `Session::compile` and can move a step between kinds,
/// so a step figure here is the backend's lowering of this graph, not a promise
/// about the cell's schedule — which is why the schedule dump is the other half.
#[test]
#[ignore = "measurement only: prints the predictor frame's multiply census and asserts the live-site reconciliation"]
fn the_predictor_frames_multiply_census_with_models() {
    use std::collections::BTreeMap;

    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair(Role::Synthesis) else {
        return;
    };
    let steps = graphs::code_predictor_steps(&pair.sound).expect("predictor head count");
    let (top_k, top_p, temperature) = pair.body.sampling();
    let built = graphs::tts_code_predictor_frame(
        pair.sound.clone(),
        steps + 1,
        graphs::SampleSpec {
            top_k,
            top_p,
            temperature,
            seed: 12345,
        },
    )
    .expect("build the frame graph");
    let graph = graph_of(built);
    let (blocks, _, _) = graphs::code_predictor_cache(&pair.sound).expect("predictor cache shape");

    // The multiply sites: a packed `DequantMatMul` reads its quantised weight as
    // the second operand, a plain `MatMul` its f32 one. Only an operand that is a
    // parameter counts as a weight read — the same rule the record's stream table
    // uses — so an activation in that position is not priced here.
    let bytes_of = |id: rlx_ir::NodeId| -> Option<(String, u64)> {
        let node = graph.node(id);
        let Op::Param { name } = &node.op else {
            return None;
        };
        node.shape
            .size_bytes()
            .map(|bytes| (name.clone(), bytes as u64))
    };
    let (mut packed, mut plain) = (0usize, 0usize);
    let (mut packed_bytes, mut plain_bytes) = (0u64, 0u64);
    let mut histogram: BTreeMap<(&'static str, String, u64), usize> = BTreeMap::new();
    for node in graph.nodes() {
        let kind = match node.op {
            Op::DequantMatMul { .. } => "dequant_matmul",
            Op::MatMul => "matmul",
            _ => continue,
        };
        let (name, bytes) = node
            .inputs
            .get(1)
            .and_then(|id| bytes_of(*id))
            .unwrap_or_default();
        // A per-level weight is named with the level's slot (`…#s-1`); the census
        // groups by the tensor and its bytes, so the slot is read off into the
        // count rather than into the key.
        let tensor = name.split('#').next().unwrap_or(&name).to_string();
        *histogram.entry((kind, tensor, bytes)).or_default() += 1;
        if kind == "dequant_matmul" {
            packed += 1;
            packed_bytes += bytes;
        } else {
            plain += 1;
            plain_bytes += bytes;
        }
    }
    let graph_param_bytes: u64 = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::Param { .. }))
        .filter_map(|node| node.shape.size_bytes().map(|bytes| bytes as u64))
        .sum();

    // Which of those sites are dead from the graph's own outputs: a multiply
    // whose value cannot reach any output is work no schedule has to do, and the
    // runtime's pre-passes drop it. The flow graph still carries it — the frame's
    // level-0 block stack ends in a row nothing reads (level 1 embeds the drawn
    // code, not the hidden), so the last block's output path there is dead while
    // its K/V rows seed the cache — which is why the site count can exceed the
    // steps the cell actually dispatches.
    let mut live: std::collections::HashSet<rlx_ir::NodeId> =
        graph.outputs.iter().copied().collect();
    let mut stack: Vec<rlx_ir::NodeId> = graph.outputs.clone();
    while let Some(id) = stack.pop() {
        for &input in &graph.node(id).inputs {
            if live.insert(input) {
                stack.push(input);
            }
        }
    }
    let dead: Vec<(&'static str, String, u64)> = graph
        .nodes()
        .iter()
        .filter(|node| !live.contains(&node.id))
        .filter_map(|node| {
            let kind = match node.op {
                Op::DequantMatMul { .. } => "dequant_matmul",
                Op::MatMul => "matmul",
                _ => return None,
            };
            let (name, bytes) = node.inputs.get(1).and_then(|id| bytes_of(*id))?;
            Some((kind, name, bytes))
        })
        .collect();
    let dead_bytes: u64 = dead.iter().map(|(_, _, bytes)| *bytes).sum();
    println!(
        "census predictor dead_sites={} dead_bytes={dead_bytes} live_sites={}",
        dead.len(),
        packed + plain - dead.len(),
    );
    for (kind, name, bytes) in &dead {
        println!("census predictor dead_site kind={kind} tensor={name} bytes={bytes}");
    }

    // The backend's own lowering, after the runtime's pre-passes: one step per
    // settled multiply, named by kind. The direct compile entry runs no cleanup, so
    // the dead sites above would otherwise still be lowered — the cell's schedule is
    // the cleaned graph's.
    let cleaned = rlx_runtime::precompile::precompile_cleanup(
        graph.clone(),
        &rlx_runtime::CompileOptions::new(),
    );
    let exe = rlx_wgpu::backend::WgpuExecutable::compile(cleaned);
    let names = exe.test_step_names();
    let mut kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    for name in &names {
        *kinds.entry(*name).or_default() += 1;
    }
    let dequant_steps = kinds.get("dequant_matmul_gguf").copied().unwrap_or(0);

    println!(
        "census predictor blocks={blocks} levels={} multiply_sites={} packed={packed} \
         matmul={plain} flow_graph_steps={dequant_steps} flow_graph_schedule_steps={}",
        steps + 1,
        packed + plain,
        names.len(),
    );
    println!(
        "census predictor weight_bytes packed={packed_bytes} plain={plain_bytes} \
         sum={} graph_params={graph_param_bytes} per_packed_mean={:.0}",
        packed_bytes + plain_bytes,
        packed_bytes as f64 / packed.max(1) as f64,
    );
    for ((kind, tensor, bytes), count) in &histogram {
        println!("census predictor site kind={kind} tensor={tensor} bytes={bytes} count={count}");
    }
    for (kind, count) in &kinds {
        println!("census predictor step_kind kind={kind} count={count}");
    }
    // The reconciliation itself, as a check rather than a reading: the backend
    // dispatches one packed step per *live* packed site, so the flow graph's packed
    // site count minus the dead packed sites is the schedule's step count — nothing
    // unaccounted for in either direction. The dead set carries plain multiplies too,
    // so it is counted by kind: a dead `MatMul` says nothing about the packed count.
    let dead_packed = dead
        .iter()
        .filter(|(kind, _, _)| *kind == "dequant_matmul")
        .count();
    assert_eq!(
        dequant_steps,
        packed - dead_packed,
        "the backend's packed steps must be the live sites: {packed} sites, {dead_packed} dead \
         packed ({} dead over both kinds), {dequant_steps} steps",
        dead.len()
    );
}
