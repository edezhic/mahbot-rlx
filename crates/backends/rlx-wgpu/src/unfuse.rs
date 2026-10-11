// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IR-level "unfusion" pass for the wgpu backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies wgpu's [`DecomposePolicy`] and re-exports `collapse_reshapes`.
//!
//! The wgpu backend lowers `FusedMatMulBiasAct` (matmul folds bias +
//! activation into its WGSL epilogue) and `FusedResidualLN`
//! (`fused_residual_ln.wgsl` does Add[+bias] + LayerNorm in one pass)
//! natively — so the decompose pass folds biased projections into
//! FusedMatMulBiasAct and residual+norm pairs into FusedResidualLN. Its
//! Attention kernel reads Q/K/V (and the mask) via per-axis strides, so it
//! accepts rank-3 `[B, S, H·D]` inputs directly and skips the
//! reshape/transpose to `[B, H, S, D]`.

use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;

/// wgpu's decompose policy: fold biased matmuls into `FusedMatMulBiasAct`,
/// fold residual+norm pairs into `FusedResidualLN`, and pass rank-3 Q/K/V/
/// mask straight to the stride-driven Attention kernel.
pub(crate) struct WgpuPolicy;

impl DecomposePolicy for WgpuPolicy {
    fn fold_matmul_bias_act(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_MMBA")
    }

    fn fold_residual_ln(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_RESLN")
    }

    fn attention_accepts_rank3(&self) -> bool {
        true
    }

    /// `binary_main.wgsl` reads each operand through `(i / rep) % len`, so a
    /// scalar or per-channel operand does not need materialising. Without this
    /// every broadcast became a full-size `Expand`: on a 192^3 SynthStrip, 52
    /// of them, 5.57 GiB of a 7.05 GiB arena, which pushed the arena past the
    /// 4 GiB binding limit into striping — and striped arenas are refused.
    fn binary_broadcast_native(&self) -> bool {
        true
    }

    // Keep `Op::FusedSwiGLU` fused — wgpu has a native kernel
    // (`fused_swiglu.wgsl`) instead of the Narrow+Silu+Mul decompose.
    fn swiglu_native(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    let g = fold_conv_bias_into_conv3d(expand_unsupported_fusions(graph));
    rlx_unfuse::unfuse(g, &WgpuPolicy)
}

/// Peel the broadcast copy `LegalizeBroadcast` materialises in front of an
/// `Op::Binary` operand that the elementwise kernel reads itself.
///
/// That pass runs before fusion (the runtime backend calls it ahead of the
/// pipeline, so no backend pass ever sees the mismatch), and the shapes it
/// leaves alone — equal shape, scalar, clean trailing slice — are *strictly*
/// fewer than the ones `rlx_unfuse::simple_broadcast` can describe. So the
/// canonical per-channel operand (`[1,C,1]` against `[1,C,T]`) gets a full-size
/// copy even though [`WgpuPolicy::binary_broadcast_native`] exists to leave
/// exactly that operand in place, and the copy is a real step: the planner gives
/// it a slot and every run writes and reads it in full.
///
/// Peeling here — after all fusion, before memory planning — puts the operand
/// back to the form the kernel wants (`binary_main.wgsl` reads `(i/rep)%len`, and
/// the host route reads the same form in `rlx-cpu`'s `eval_elementwise_fast`) and
/// drops the `expand` step and the slot. What the peel refuses to touch is
/// [`binary_broadcast_expand_elisions`]'s business; a pattern no single contiguous
/// run describes keeps its copy, because the lowering reads an operand it cannot
/// describe *densely* — its `bcast` closure falls back to `(1, 0)` and
/// `binary_main.wgsl` indexes flat then — so peeling one would read the source's
/// own bytes where the output shape's are wanted.
pub fn elide_binary_broadcast_expands(graph: Graph) -> Graph {
    let peeled: HashMap<NodeId, NodeId> = binary_broadcast_expand_elisions(&graph)
        .into_iter()
        .collect();
    if peeled.is_empty() {
        return graph;
    }
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match peeled.get(&node.id) {
            // The source precedes its expand in node order, so its remap is
            // known: hand it on and every reader reads the source directly.
            Some(&src) => id_map[&src],
            None => out.add_node(node.op.clone(), inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// The `(expand, source)` pairs [`elide_binary_broadcast_expands`] drops, in node
/// order — the one definition of the peel's selection, so evidence gathered about
/// the peel cannot drift from it.
///
/// Selected when the expand is not itself a graph output and every one of its
/// readers is an `Op::Binary` that reads the operand itself: a different element
/// count from the source, a dtype whose f32 lanes *are* the elements, a shape
/// `simple_broadcast` describes, and a source whose storage the device step may
/// read as lanes. An expand with no readers at all goes too.
///
/// `Op::Compare`/`Op::Where`/`Op::Fma` are not in it: only `Compare` even requires
/// equal shapes, and all three index their operands flat, so the copy is their
/// only way to see a broadcast operand. A complex consumer, and any dtype staging
/// more lanes than elements (F64/I64, a complex lane pair), keep theirs for the
/// same reason — the host replay reads those flat too. So does a parameter
/// operand whose storage is not one f32 lane per element.
pub fn binary_broadcast_expand_elisions(graph: &Graph) -> Vec<(NodeId, NodeId)> {
    let outputs: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    // Per input: "every reader of it reads the broadcast itself", folded over the
    // edges, so nothing is collected per consumer.
    let mut readable_by_all: HashMap<NodeId, bool> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            // A reader of an expand reads the expand's *source* once it is peeled,
            // so that is the shape the kernel has to describe.
            let src = match &graph.node(input).op {
                Op::Expand { .. } if graph.node(input).inputs.len() == 1 => {
                    graph.node(input).inputs[0]
                }
                _ => input,
            };
            let src_shape = &graph.node(src).shape;
            let ok = matches!(&node.op, Op::Binary(_))
                // `(i/rep)%len` indexes f32 lanes, so it describes the operand
                // only when a lane is an element.
                && node.shape.dtype().size_bytes() <= 4
                // An equal element count is already read flat.
                && node.shape.num_elements() != src_shape.num_elements()
                // Both shapes have to be static before `simple_broadcast` is asked:
                // it demands them, and a dynamic dim would otherwise reach it by
                // making the count test above `None != Some(..)`. The host twin
                // (`rlx-cpu`'s `broadcast_run`) guards the same way.
                && node.shape.is_static()
                && src_shape.is_static()
                && rlx_unfuse::simple_broadcast(&node.shape, src_shape).is_some()
                // A parameter's storage is only read as lanes when its dtype says so:
                // the typed upload stores `U8`/`I8`/`BF16` raw, so peeling would hand
                // the step the model's packed bytes. Keeping the copy keeps whatever
                // refusal that operand already hit at the expand's own arm. An
                // activation is not judged here — the planner widens those at the
                // boundary, a rule stated and applied by `lower.rs`'s
                // `param_operands_are_lane_readable`.
                && (!crate::buffer::is_parameter_storage(graph, src)
                    || crate::buffer::unstageable_by_dtype(src_shape.dtype()).is_none());
            readable_by_all
                .entry(input)
                .and_modify(|all| *all &= ok)
                .or_insert(ok);
        }
    }
    graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(&n.op, Op::Expand { .. })
                && n.inputs.len() == 1
                // An expand that is itself an output keeps its storage.
                && !outputs.contains(&n.id)
                && readable_by_all.get(&n.id).copied().unwrap_or(true)
        })
        .map(|n| (n.id, n.inputs[0]))
        .collect()
}

/// Rewrite a 3-D [`Op::FusedConvBiasAct`] into an [`Op::Conv3d`] that keeps the
/// bias as a third input.
///
/// The conv3d shaders add the bias in their store, so this costs nothing at
/// run time and saves two full-size tensors per convolution: the `Expand` of
/// the bias to the output shape, and the `Add` that consumed it. On SynthStrip
/// that is 27 of each — most of the reason a 192^3 activation arena did not
/// fit under the 4 GiB binding limit.
///
/// The extra input is invisible to everything except this backend's lowering:
/// the memory planner reads it as one more liveness edge, which is exactly
/// what it is.
/// Rewrite a rank-5 `FusedConvBiasAct` (no activation, no residual) into a
/// **3-input** `Op::Conv3d` so `conv3d.wgsl` can add the bias in its store.
///
/// This is a **backend-private** representation. `Op::Conv3d` is
/// `Arity::Exact(2)` in the IR contract ("bias via Add"), so the graph this
/// produces would not pass `rlx_ir::verify` — which is fine, because it is
/// created after verification and never leaves this crate.
///
/// The arity is deliberately *not* widened to `Range { min: 2, max: 3 }` in
/// `Op::arity`: CPU's `compile_conv3d` does not read a third operand, so a
/// bias-carrying `Conv3d` reaching any backend other than this one would have
/// its bias silently dropped. Keeping the global contract at 2 means only the
/// backend that implements the fused form can construct it.
fn fold_conv_bias_into_conv3d(g: Graph) -> Graph {
    let is_target = |n: &rlx_ir::Node| {
        matches!(
            &n.op,
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            }
        ) && n.shape.rank() == 5
            && n.inputs.len() == 3
    };
    if !g.nodes().iter().any(is_target) {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let op = if is_target(node) {
            let Op::FusedConvBiasAct {
                stride,
                padding,
                dilation,
                groups,
                ..
            } = &node.op
            else {
                unreachable!()
            };
            let a3 = |v: &Vec<usize>| [v[0], v[1], v[2]];
            Op::Conv3d {
                stride: a3(stride),
                padding: a3(padding),
                dilation: a3(dilation),
                groups: *groups,
            }
        } else {
            node.op.clone()
        };
        let nid = out.add_node(op, ins, node.shape.clone());
        id_map.insert(node.id, nid);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Expand any [`Op::FusedConvBiasAct`] that is not the 2-D case wgpu lowers.
///
/// `SUPPORTED_OPS` lists the op unconditionally and `compile::lower` asserts
/// `only 2D NCHW convs are fused`, so a 3-D graph aborts at compile time
/// rather than running. (CUDA and ROCm share the claim without the assert and
/// silently compute a 2-D convolution instead.) Sending other ranks back to
/// primitives lets them run on `conv3d.wgsl`.
fn expand_unsupported_fusions(g: Graph) -> Graph {
    // Rank-5 with no activation and no residual is lowered natively — the
    // conv3d shaders apply the bias in their store. Expanding it instead costs
    // a full-size `Expand` of the bias and a full-size `Add` per convolution:
    // 27 of each on SynthStrip, which is most of why a 192^3 arena did not fit.
    let unsupported = |n: &rlx_ir::Node| -> bool {
        match &n.op {
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            } => n.shape.rank() != 5 || n.inputs.len() != 3,
            Op::FusedConvBiasAct { .. } => n.shape.rank() != 4,
            _ => false,
        }
    };
    if !g.nodes().iter().any(unsupported) {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if unsupported(node) {
            inline_unfused(&mut out, &node.op, &new_inputs, &node.shape)
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Run one node through the shared decomposer and splice the result back in.
fn inline_unfused(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    let mut mini = Graph::new("wgpu_unfuse_fcba");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, &src) in inputs.iter().enumerate() {
        let sh = out.node(src).shape.clone();
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op
            && let Some(rest) = name.strip_prefix("in")
            && let Ok(i) = rest.parse::<usize>()
        {
            map.insert(n.id, inputs[i]);
            continue;
        }
        let mapped: Vec<NodeId> = n.inputs.iter().map(|id| map[id]).collect();
        let nid = out.add_node(n.op.clone(), mapped, n.shape.clone());
        map.insert(n.id, nid);
    }
    map[&expanded.outputs[0]]
}

pub fn collapse_reshapes(graph: Graph) -> Graph {
    rlx_unfuse::collapse_reshapes(graph)
}
