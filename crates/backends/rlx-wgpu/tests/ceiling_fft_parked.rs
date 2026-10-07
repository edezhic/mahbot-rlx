// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parked-operand refusals of both FFT ops, in its own process.
//!
//! Neither `OpKind::Fft` nor `OpKind::FftQ` is on the ceiling rule's list — their arms
//! ask the rule's own question in place and refuse a parked operand themselves, because
//! every step either op has carries offsets that drop the `WEIGHT_BUF_TAG`. Both a
//! device step and a host step would then read the activation arena instead of the
//! weight buffer, silently, so a parameter operand is refused rather than mis-addressed.
//!
//! The operand is a `param`, not a graph input: an input lives in the activation arena
//! and never parks, which is exactly the geometry the control compiles.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` is process-global, hence one test per file.

use rlx_ir::fft::{FftNorm, FftQScale};
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// `fft_meta` reads a F32 last axis as the 2N real-block layout, so `[4]` is two complex
/// points — the smallest radix-2 geometry both ops accept.
const LEN: usize = 4;

fn fft_graph(name: &str) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[LEN], F));
    let y = g.fft_norm(p, false, FftNorm::Backward);
    g.set_outputs(vec![y]);
    g
}

fn fft_q_graph(name: &str) -> Graph {
    let mut g = Graph::new(name);
    let p = g.param("p", Shape::new(&[LEN], F));
    let y = g.fft_q(p, false, FftNorm::Backward, FftQScale::None);
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

/// Compile `graph` and return the refusal text; a returned executor means the parked
/// operand was accepted.
fn refused(graph: Graph, why: &str) -> String {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(graph)
    }));
    let Err(payload) = caught else {
        panic!("{why}: the parked operand was accepted instead of refused");
    };
    refusal_message(payload)
}

#[test]
fn both_fft_ops_refuse_a_parked_operand_and_compile_without_one() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    let msg = refused(fft_graph("fft-parked"), "a parked Fft operand");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (parked FFT operand)"),
        "the Fft arm's own refuse must fire for a parked operand; got {msg}"
    );
    assert!(
        msg.contains("Fft") && msg.contains("parked in the weight buffer"),
        "the Fft refusal must name the op and where the operand it cannot address lives; \
         got {msg}"
    );

    let msg = refused(fft_q_graph("fft-q-parked"), "a parked FftQ operand");
    assert!(
        msg.contains("rlx-wgpu: REFUSED (FftQ geometry)"),
        "the FftQ arm's own refuse must fire for a parked operand; got {msg}"
    );
    assert!(
        msg.contains("FftQ") && msg.contains("parked in the weight buffer"),
        "the FftQ refusal must name the op and where the operand it cannot address lives; \
         got {msg}"
    );

    // Control: with nothing parked the same two geometries compile — neither op's
    // parked-operand assert is about the transform's own well-formedness.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    for (graph, why) in [
        (fft_graph("fft-control"), "Fft with nothing parked"),
        (fft_q_graph("fft-q-control"), "FftQ with nothing parked"),
    ] {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuExecutable::compile(graph)
        }));
        if let Err(payload) = caught {
            panic!("{why}: must compile — {}", refusal_message(payload));
        }
    }
}
