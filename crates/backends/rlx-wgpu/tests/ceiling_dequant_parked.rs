// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The quantized (non-GGUF, non-MLX) `Op::DequantMatMul` sub-path's parked-operand
//! refusal, in its own process.
//!
//! That sub-path is not on the ceiling rule's list, so it asks the rule's own question
//! in-arm and routes to the generic host route when an operand is parked. That route
//! reads every operand as one f32 lane per element, so a *packed* weight (I8) or a
//! two-byte encoding (BF16) parked in the weight buffer is refused by
//! `assert_host_route_stageable` rather than read four times past its length.
//!
//! The walk takes `node.inputs` in order `[x, w_q, scale, zp]` and refuses the first
//! operand the route cannot stage. Case 2/3 therefore declare the weight `F32` — with a
//! packed I8/U8 weight the walk refuses the weight first and the scale/zero-point half
//! of the walk could not be reached under this lever.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` is process-global, hence one test per file.

use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
const M: usize = 2;
const K: usize = 4;
const N: usize = 4;
const BLOCK: u32 = 2;

/// The shipped fixture's geometry: `x[m,k] @ wq[k,n]` with `k.div_ceil(block_size)`
/// scale/zero-point blocks, output `[m,n] F32`. Returns the weight's, scale's and
/// zero-point's node ids so a refusal can be attributed to the operand the walk refused.
fn int4_graph(
    name: &str,
    wq_dt: DType,
    sc_dt: DType,
    zp_dt: DType,
) -> (Graph, NodeId, NodeId, NodeId) {
    use rlx_ir::QuantScheme;
    let n_blocks = K.div_ceil(BLOCK as usize);
    let mut g = Graph::new(name);
    let x = g.input("x", Shape::new(&[M, K], F));
    let wq = g.param("wq", Shape::new(&[K, N], wq_dt));
    let sc = g.param("sc", Shape::new(&[n_blocks, N], sc_dt));
    let zp = g.param("zp", Shape::new(&[n_blocks, N], zp_dt));
    let y = g.dequant_matmul(
        x,
        wq,
        sc,
        zp,
        QuantScheme::Int4Block { block_size: BLOCK },
        Shape::new(&[M, N], F),
    );
    g.set_outputs(vec![y]);
    (g, wq, sc, zp)
}

/// Compile `graph` and return the refusal text. The refusal is decided at compile
/// time, so a returned executor would mean the route staged an operand it cannot stage.
fn refused(graph: Graph, why: &str) -> String {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(graph)
    }));
    let Err(payload) = caught else {
        panic!("{why}: the operand was staged instead of refused");
    };
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// The words every refusal here carries: the stable marker, the refused operand's own
/// declared encoding, where it lives, and why the route cannot stage it.
fn assert_parked_raw(msg: &str, why: &str, dtype: &str, id: NodeId) {
    assert!(
        msg.contains("rlx-wgpu: REFUSED (operand not stageable as f32)"),
        "{why}: the refusal must carry a stable marker a caller can distinguish from a \
         process death; got {msg}"
    );
    assert!(
        msg.contains(dtype) && msg.contains("parked in the weight buffer"),
        "{why}: the refusal must name the operand's encoding and where it lives; got {msg}"
    );
    assert!(
        msg.contains("the upload stores it raw"),
        "{why}: the refusal must state why this parameter cannot be staged; got {msg}"
    );
    assert!(
        msg.contains(&format!("{id:?}")),
        "{why}: the refusal must name the operand's own node; got {msg}"
    );
}

/// The other side: `graph` must compile with nothing parked, or the refusal above is
/// attributable to some other defect.
fn compiles(graph: Graph, why: &str) {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(graph)
    }));
    if let Err(payload) = caught {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        panic!("{why}: compile panicked unexpectedly — {msg}");
    }
}

#[test]
fn a_parked_quantized_operand_is_refused_by_the_walk_in_input_order() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    // 1. The packed weight: one byte per element in the weight buffer while the route
    // reads four, so it is the first operand the walk cannot stage.
    let (g, wq, _, _) = int4_graph("dq-parked-weight", DType::I8, F, F);
    let msg = refused(g, "a parked Int4 weight");
    assert_parked_raw(&msg, "a parked Int4 weight", "I8", wq);

    // 2. A stageable weight, a two-byte scale: the walk skips `x` (a graph input lives
    // in the activation arena, not a parameter's storage), accepts the F32 weight, and
    // refuses the scale.
    let (g, _, sc, _) = int4_graph("dq-parked-scale", F, DType::BF16, F);
    let msg = refused(g, "a parked BF16 scale");
    assert_parked_raw(&msg, "a parked BF16 scale", "BF16", sc);

    // 3. The same, one operand further along: the F32 scale is stageable and the
    // zero-point is not.
    let (g, _, _, zp) = int4_graph("dq-parked-zp", F, F, DType::BF16);
    let msg = refused(g, "a parked BF16 zero-point");
    assert_parked_raw(&msg, "a parked BF16 zero-point", "BF16", zp);

    // Control: the same packed-I8 graph compiles once nothing is parked — the lever,
    // not the graph, is what the three refusals above are attributable to.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "0");
    let (g, _, _, _) = int4_graph("dq-control", DType::I8, F, F);
    compiles(g, "the Int4 fixture with nothing parked");
}
