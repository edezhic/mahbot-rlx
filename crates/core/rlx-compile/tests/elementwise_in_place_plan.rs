// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The elementwise in-place rule: a chain like `matmul → bias-add → activation`
//! must share one arena block, and every operand the rule cannot prove is the
//! same element must be refused — a broadcast (plain or inside a fused region),
//! a fused prologue, a repeated operand, a second reader, a boundary tensor, a
//! graph output, a view that is not an identity.
//!
//! The rule is only correct if the refusals are real: an accepted alias writes
//! the bytes it read, so a wrong acceptance silently changes an answer rather
//! than raising a plan size. Each test builds a graph the rule must refuse, and
//! asserts the rule stays out.

use rlx_compile::memory::{
    ArenaWidthPolicy, BufferSlot, GuardedPlanner, MemoryPlan, MemoryPlanOptions,
    plan_memory_with_dequant_guard, plan_memory_with_options_policy,
};
use rlx_compile::plan_check::{PlanFindingKind, check_plan};
use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

const F: DType = DType::F32;

fn plain(g: &Graph) -> MemoryPlan {
    plan_memory_with_dequant_guard(g, 16, GuardedPlanner::F32Uniform, false, false)
}

fn in_place(g: &Graph) -> MemoryPlan {
    plan_memory_with_dequant_guard(g, 16, GuardedPlanner::F32Uniform, false, true)
}

/// `matmul → bias-add → activation`: the encoder front's chain in miniature.
/// Returns the three activation nodes in order.
fn chain() -> (Graph, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("chain");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let b = g.param("b", shape.clone());
    let add = g.add(mm, b);
    let act = g.gelu(add);
    g.set_outputs(vec![act]);
    (g, mm, add, act)
}

#[test]
fn the_chain_shares_one_block_and_the_plan_is_clean() {
    let (g, mm, add, act) = chain();
    let plain = plain(&g);
    let in_place = in_place(&g);

    assert!(plain.in_place.is_empty(), "the plain planner never aliases");
    assert_eq!(in_place.in_place.len(), 2, "the add and the activation");
    let root = in_place.assignments[&mm].offset;
    assert_eq!(in_place.assignments[&add].offset, root);
    assert_eq!(in_place.assignments[&act].offset, root);
    assert_eq!(
        plain.arena_size - in_place.arena_size,
        8 * 64 * 4,
        "exactly the second block"
    );
    assert_eq!(
        plain.total_unshared_bytes() - in_place.total_unshared_bytes(),
        2 * 8 * 64 * 4,
        "a shared block is counted once, not once per member"
    );

    let report = check_plan(&g, &in_place);
    assert!(
        report.is_clean(),
        "the plan gate must resolve a deliberate in-place write:\n{}",
        report.render()
    );
}

/// The gate covers the alias the map asserts: a member the plan gave a slot of
/// its own is a wrong-bytes write, and folding it onto its root would otherwise
/// drop it from the overlap scan and report the plan clean.
#[test]
fn the_gate_catches_a_member_that_is_not_on_its_roots_block() {
    let (g, mm, add, _) = chain();
    let mut plan = in_place(&g);
    let slot = &plan.assignments[&mm];
    let off = slot.offset + slot.size;
    let size = slot.size;
    plan.assignments
        .insert(add, BufferSlot { offset: off, size });
    let report = check_plan(&g, &plan);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.kind == PlanFindingKind::InPlaceSlotMismatch),
        "a member off its root's block is a finding:\n{}",
        report.render()
    );
}

/// A `[1,64]` operand broadcast against `[8,64]` changes the element count, so the
/// add may not take its bytes and the alias goes to the dense operand. With the
/// size/shape guard gone the broadcast operand comes first in the scan and wins, so
/// this fails instead of passing on the second operand's merits.
#[test]
fn a_broadcast_operand_is_refused() {
    let mut g = Graph::new("broadcast");
    let x = g.input("x", Shape::new(&[1, 64], F));
    let w = g.param("w", Shape::new(&[1, 64], F));
    let narrow = g.matmul(x, w, Shape::new(&[1, 64], F));
    let p = g.param("p", Shape::new(&[8, 64], F));
    let dense = g.gelu(p);
    let add = g.add(narrow, dense); // [1,64] + [8,64] → [8,64]
    let act = g.gelu(add);
    g.set_outputs(vec![act]);
    let plan = in_place(&g);
    assert_eq!(
        plan.in_place.get(&add),
        Some(&dense),
        "the broadcast operand is refused; the dense one is taken"
    );
}

/// `y + y` reads the same tensor twice; overwriting it would clobber the second
/// read on an implementation that reads after writing.
#[test]
fn a_repeated_operand_is_refused() {
    let mut g = Graph::new("repeated");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let y = g.matmul(x, w, shape.clone());
    let add = g.add(y, y);
    let act = g.gelu(add);
    g.set_outputs(vec![act]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&add),
        "the same tensor read twice is not the same element once"
    );
}

/// A tensor with a second reader is still needed after this step.
#[test]
fn a_second_reader_is_refused() {
    let mut g = Graph::new("second_reader");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let y = g.matmul(x, w, shape.clone());
    let b = g.param("b", shape.clone());
    let first = g.gelu(y);
    let second = g.add(y, b);
    g.set_outputs(vec![first, second]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&first) && !plan.in_place.contains_key(&second),
        "neither consumer can take a tensor the other still reads"
    );
}

/// A second reader that reaches the root through a view is still a second read:
/// the view shares the root's bytes, so overwriting them changes what the view's
/// consumer sees, even though that consumer never names the root's node.
#[test]
fn a_reader_through_a_view_is_a_second_reader() {
    let mut g = Graph::new("view_reader");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let act = g.gelu(mm);
    let view = g.reshape(mm, vec![8, 64], shape.clone());
    let b = g.param("b", shape.clone());
    let other = g.add(view, b);
    g.set_outputs(vec![act, other]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&act),
        "a view of the root hands its bytes to another reader"
    );
}

/// A chain rooted at a reshape of a different shape is refused even though the
/// element order would have been preserved: the size equality cannot catch it, so
/// this is the case the shape check alone refuses.
#[test]
fn a_reshape_rooted_chain_is_refused() {
    let mut g = Graph::new("reshaped_root");
    let shape = Shape::new(&[64, 8], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let view = g.reshape(mm, vec![8, 64], Shape::new(&[8, 64], F));
    let act = g.gelu(view);
    g.set_outputs(vec![act]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&act),
        "the root's shape is not the output's, so the alias is not proved"
    );
}

/// A member whose slot is narrower than its root's, at the same shape: under
/// `NativeHalfWidened` a custom kernel's input keeps its declared two bytes while the
/// f16 activation feeding it is widened to four, so the shapes match and only the
/// size check can refuse the alias.
#[test]
fn a_member_narrower_than_its_root_is_refused() {
    let mut g = Graph::new("half_widened");
    let shape = Shape::new(&[8, 64], DType::F16);
    let x = g.input("x", shape.clone());
    let wide = g.gelu(x); // no custom reader: widened to four bytes
    let native = g.gelu(wide); // a custom's input: keeps its two
    let c = g.add_node(
        Op::Custom {
            name: "test.passthrough".into(),
            num_inputs: 1,
            attrs: Vec::new(),
        },
        vec![native],
        shape.clone(),
    );
    g.set_outputs(vec![c]);
    let opts = MemoryPlanOptions {
        elementwise_in_place: true,
        ..MemoryPlanOptions::inference()
    };
    let plan = plan_memory_with_options_policy(&g, 16, opts, ArenaWidthPolicy::NativeHalfWidened);
    assert!(
        !plan.in_place.contains_key(&native),
        "the member's slot is half the root's, so the write would not fill it"
    );
}

/// A graph output is read back after the last step; its bytes are not free.
#[test]
fn a_graph_output_root_is_refused() {
    let mut g = Graph::new("output_root");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let act = g.gelu(mm);
    g.set_outputs(vec![mm, act]);
    assert!(
        in_place(&g).in_place.is_empty(),
        "an output root must keep its own bytes"
    );
}

/// A parameter or an input is the caller's storage, not the arena's to reuse.
#[test]
fn a_boundary_root_is_refused() {
    let mut g = Graph::new("boundary_root");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let b = g.param("b", shape.clone());
    let add = g.add(x, b);
    let act = g.gelu(add);
    g.set_outputs(vec![act]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&add),
        "an input and a parameter are both boundary storage"
    );
}

/// A length-preserving narrow at offset 0 happens to keep the index, and the
/// rule still refuses it: it proves the identity, it does not test for it.
#[test]
fn a_view_that_is_not_an_identity_is_refused() {
    let mut g = Graph::new("narrow_view");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let view = g.narrow_(mm, 0, 0, 8);
    let act = g.gelu(view);
    g.set_outputs(vec![act]);
    let plan = in_place(&g);
    assert!(
        !plan.in_place.contains_key(&act),
        "a narrow is not a reshape, so the alias is not proved"
    );
}

/// A fused region with a prologue reads through the transform before the chain,
/// so the element it writes is not the element it read.
#[test]
fn a_fused_prologue_is_refused() {
    let mut g = Graph::new("prologue");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let region = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![ChainStep::Activation(
                Activation::Relu,
                ChainOperand::Input(0),
            )],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::ResizeNearest2x,
            prologue_input: 0,
        },
        vec![mm],
        shape.clone(),
    );
    g.set_outputs(vec![region]);
    assert!(
        !in_place(&g).in_place.contains_key(&region),
        "a prologue breaks the one-element-per-read identity"
    );
}

/// A region's `input_modulus` is its trailing broadcast: a non-zero entry reads
/// operand element `(i / modulus) % len`, not `i`.
#[test]
fn a_broadcast_inside_a_fused_region_is_refused() {
    let mut g = Graph::new("region_modulus");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let mm = g.matmul(x, w, shape.clone());
    let mut input_modulus = [0; 16];
    input_modulus[0] = 4;
    let region = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![ChainStep::Activation(
                Activation::Relu,
                ChainOperand::Input(0),
            )],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus,
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![mm],
        shape.clone(),
    );
    g.set_outputs(vec![region]);
    assert!(
        !in_place(&g).in_place.contains_key(&region),
        "a broadcast operand is read at a different index than it is written"
    );
}

/// A graph the rule cannot help keeps exactly the plain plan: the aliased
/// layout is applied only when it is smaller, never because it was asked for.
#[test]
fn a_graph_the_rule_cannot_help_is_planned_as_before() {
    let mut g = Graph::new("no_chain");
    let shape = Shape::new(&[8, 64], F);
    let x = g.input("x", shape.clone());
    let b = g.param("b", shape.clone());
    let add = g.add(x, b); // both operands are boundary storage
    g.set_outputs(vec![add]);
    let plan = in_place(&g);
    assert!(plan.in_place.is_empty());
    assert_eq!(plan.arena_size, plain(&g).arena_size);
    assert!(check_plan(&g, &plan).is_clean());
}
