// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Integration tests — compiler hardening (pipeline snapshots + rewrite).

use rlx_compile::{CompilePipeline, FusionTarget, inspect_pipeline, rewrite_for_backend};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphModule, Op, OpKind, Shape, verify::verify_all};

fn f32_shape(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

#[test]
fn pipeline_inspect_covers_hir_mir_lir() {
    let module = GraphModule::define("layer", |m| {
        let x = m.input("x", f32_shape(&[2, 64]));
        let w = m.param("w", f32_shape(&[64, 64]));
        m.linear(x, w, None, None, f32_shape(&[2, 64]))
    });
    let hir = module.into_hir().expect("hir");
    let pipe = CompilePipeline::new(FusionTarget::Cpu);
    let dump = inspect_pipeline(&pipe, hir).expect("inspect");

    assert!(dump.hir.contains("hir @layer"));
    assert!(dump.hir.contains("linear"));
    assert!(dump.mir_lowered.contains("mir @layer"));
    assert!(dump.mir_optimized.contains("graph @layer"));
    assert!(dump.lir.contains("lir @layer"));
    assert!(dump.lir.contains("arena:"));
    assert!(dump.lir.contains("fingerprint:"));
    assert!(dump.fusion.contains("fusion"));
}

#[test]
fn compile_module_passes_verifier() {
    let module = GraphModule::define("ffn", |m| {
        let x = m.input("x", f32_shape(&[4, 128]));
        let up = m.param("up", f32_shape(&[128, 256]));
        let gate = m.param("gate", f32_shape(&[128, 256]));
        let down = m.param("down", f32_shape(&[256, 128]));
        m.swiglu_ffn(x, up, gate, down, f32_shape(&[4, 128]))
    });
    let pipe = CompilePipeline::new(FusionTarget::Cpu);
    let result = pipe.compile_module(module).expect("compile");
    assert!(verify_all(result.lir.as_graph()).is_empty());
}

#[test]
fn fusion_clean_hir_linear() {
    let module = GraphModule::define("layer", |m| {
        let x = m.input("x", f32_shape(&[4, 32]));
        let w = m.param("w", f32_shape(&[32, 32]));
        let b = m.param("b", f32_shape(&[32]));
        m.linear_fused(x, w, b, None, f32_shape(&[4, 32]))
    });
    let pipe = CompilePipeline::new(FusionTarget::Cpu).with_assert_fusion_clean(true);
    pipe.compile_module(module)
        .expect("linear_fused should be fusion-clean");
}

#[test]
fn rewrite_unfuses_for_strict_backend() {
    let f = DType::F32;
    let mut g = Graph::new("f");
    let x = g.input("x", Shape::new(&[2, 8], f));
    let w = g.param("w", Shape::new(&[8, 4], f));
    let b = g.param("b", Shape::new(&[4], f));
    let out = g.fused_matmul_bias_act(x, w, b, None, Shape::new(&[2, 4], f));
    g.set_outputs(vec![out]);

    let strict = &[
        OpKind::Input,
        OpKind::Param,
        OpKind::MatMul,
        OpKind::Binary,
        OpKind::Expand,
        OpKind::Activation,
    ];
    let rewritten = rewrite_for_backend(g, strict);
    assert!(verify_all(&rewritten).is_empty());
    assert!(
        !rewritten
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. }))
    );
}

/// A duplicate `Op::Input` name is fatal on the release path too: the pipeline
/// checks it unconditionally, not only under `debug_assertions`. Both
/// declarations are consumed, i.e. this is the *live* duplicate — the one whose
/// second copy would otherwise be fed to nothing.
///
/// `expected = "duplicate"` rather than `"duplicate leaf names"` because in a
/// debug build the HIR verifier can fire first (it reads `verify`, which now
/// includes `verify_unique_leaf_names`) with `duplicate Input name …` rather
/// than the pipeline's `… declares duplicate leaf names …`. Both refuse, and
/// this pins the refusal in either profile.
#[test]
#[should_panic(expected = "duplicate")]
fn duplicate_input_name_is_refused_by_the_pipeline() {
    let mut g = Graph::new("dup-input");
    let a = g.input("x", f32_shape(&[2]));
    let b = g.input("x", f32_shape(&[2]));
    let sum = g.binary(BinaryOp::Add, a, b, f32_shape(&[2]));
    g.set_outputs(vec![sum]);
    CompilePipeline::new(FusionTarget::Cpu).compile_graph(g);
}

/// The dead half of the same rule: the second declaration is consumed by
/// nothing, so DCE would prune it and a post-pass check would never see it.
/// The refusal has to happen before the passes, and this is what pins that.
#[test]
#[should_panic(expected = "duplicate")]
fn duplicate_dead_leaf_name_is_refused_by_the_pipeline() {
    let mut g = Graph::new("dup-dead");
    let x = g.input("x", f32_shape(&[2]));
    let _unused = g.input("x", f32_shape(&[2]));
    let y = g.activation(rlx_ir::op::Activation::Relu, x, f32_shape(&[2]));
    g.set_outputs(vec![y]);
    CompilePipeline::new(FusionTarget::Cpu).compile_graph(g);
}
