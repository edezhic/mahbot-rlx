// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wide carry-bearing recurrent refusal, in its own process.
//!
//! `OpKind::Rnn` addresses the arena by absolute offset, so `lower::should_host_at_ceiling`
//! runs as the first statement of the per-node lowering loop and decides before the
//! `Op::Rnn` arm. A carry-bearing `Rnn` wider than `NARROW_RNN_HIDDEN_MAX` keeps its own
//! whole-arena host step (`supported_ops::carry_lowering` → `WholeArenaHostStep`), which
//! mirrors the activation arena with 32-bit byte offsets and therefore cannot address an
//! operand parked in the separate weight buffer — the `WEIGHT_BUF_TAG` is truncated away
//! in the cast and the step reads the mirror instead. The rule refuses it loudly instead
//! of computing a recurrence over the wrong bytes.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` is what parks the parameters here; it is process-global,
//! hence one test per file.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
const B: usize = 1;
const S: usize = 2;
const INPUT_SIZE: usize = 3;
/// Above `supported_ops::NARROW_RNN_HIDDEN_MAX` (256), so the op lowers to its own
/// whole-arena host step rather than the narrow workgroup kernel.
const HIDDEN: usize = 300;

/// A carried single-layer RNN. `h0` is a parameter, which is what parks it; the
/// `[batch, hidden]` state layout is the one `gru_carry_native` uses for a
/// single-layer, unidirectional op.
fn wide_carry_rnn(name: &str) -> Graph {
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
        true,
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
fn a_wide_carry_rnn_with_a_parked_operand_is_refused() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(wide_carry_rnn("rnn-wide-carry"))
    }));
    let Err(payload) = caught else {
        panic!("a wide carry RNN over a parked operand was accepted instead of refused");
    };
    let msg = refusal_message(payload);
    assert!(
        msg.contains("rlx-wgpu: REFUSED (wide carry RNN)"),
        "the refusal must name the rule that fired; got {msg}"
    );
    assert!(
        msg.contains("Rnn"),
        "the refusal must name the op, whose whole-arena host step cannot take a parked \
         operand; got {msg}"
    );
    assert!(
        msg.contains("parked in the weight buffer"),
        "the refusal must say where the operand it cannot address lives; got {msg}"
    );

    // Control: with nothing parked the same geometry compiles, so the refusal above is
    // attributable to the parked operand rather than to the graph.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(wide_carry_rnn("rnn-wide-carry-control"))
    }));
    if let Err(payload) = caught {
        panic!(
            "the same carry RNN must compile with nothing parked — {}",
            refusal_message(payload)
        );
    }
}
