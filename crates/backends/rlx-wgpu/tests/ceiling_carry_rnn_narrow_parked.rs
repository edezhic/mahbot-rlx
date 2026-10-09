// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The narrow carry-bearing recurrent refusal, in its own process.
//!
//! The pair to `ceiling_carry_rnn_parked.rs`: that file pins the WIDE arm, whose
//! op keeps its own whole-arena host step (`CarryLowering::WholeArenaHostStep`);
//! this is the narrow arm of the same pair, at or below
//! `supported_ops::NARROW_RNN_HIDDEN_MAX` (256), where
//! `should_host_at_ceiling` finds `CarryLowering::WindowedKernel`. The final state
//! is written back into a state input *through the windowed bind*, and the generic
//! host route carries a single output offset with no field that can express an
//! input mutation — hosting would return a plausible `y` while the state silently
//! never advances. A parked operand is what makes the ceiling rule run at all on a
//! small graph: without it the arena binds whole at base zero and the rule returns
//! before it classifies the op.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` is what parks the parameters here; it is
//! process-global, hence one test per file.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
const B: usize = 1;
const S: usize = 2;
const INPUT_SIZE: usize = 3;
/// At or below `supported_ops::NARROW_RNN_HIDDEN_MAX` (256), so the op lowers to the
/// narrow workgroup kernel rather than its own whole-arena host step.
const HIDDEN: usize = 32;

/// A carried single-layer RNN. `h0` is a parameter, which is what parks it; the
/// `[batch, hidden]` state layout is the one `gru_carry_native` uses for a
/// single-layer, unidirectional op.
fn narrow_carry_rnn(name: &str) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", Shape::new(&[B, S, INPUT_SIZE], F));
    let w_ih = g.param("w_ih", Shape::new(&[HIDDEN, INPUT_SIZE], F));
    let w_hh = g.param("w_hh", Shape::new(&[HIDDEN, HIDDEN], F));
    let bias = g.param("bias", Shape::new(&[HIDDEN], F));
    let h0 = g.param("h0", Shape::new(&[B, HIDDEN], F));
    let y = g.rnn(
        x,
        w_ih,
        w_hh,
        bias,
        Some(h0),
        HIDDEN,
        1,
        false,
        false,
        Shape::new(&[B, S, HIDDEN], F),
    );
    g.set_outputs(vec![y]);
    g
}

fn refusal_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn a_narrow_carry_rnn_with_a_parked_operand_is_refused() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // The lever only makes the split planner park every parameter. It is a code override
    // rather than a process-env write: one test per file, but no need to touch the real
    // environment.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(narrow_carry_rnn("rnn-narrow-carry"))
    }));
    let Err(payload) = caught else {
        panic!("a narrow carry RNN over a parked operand was accepted instead of refused");
    };
    let msg = refusal_message(payload);
    assert!(
        msg.contains("rlx-wgpu: REFUSED (carry state)"),
        "the refusal must name the rule that fired; got {msg}"
    );
    assert!(
        msg.contains("Rnn"),
        "the refusal must name the op, whose final state the host route cannot write back; \
         got {msg}"
    );
    assert!(
        msg.contains("writes its final state back into a state input buffer"),
        "the refusal must say what the host route cannot express; got {msg}"
    );

    // Control: with nothing parked the same geometry compiles, so the refusal above is
    // attributable to the parked operand rather than to the graph.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(narrow_carry_rnn("rnn-narrow-carry-control"))
    }));
    if let Err(payload) = caught {
        panic!(
            "the same carry RNN must compile with nothing parked — {}",
            refusal_message(payload)
        );
    }
}
