// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the ceiling rule's host route may stage, in its own process.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` parks every parameter, which sends a listed op
//! that reads one to the generic host route (`ceiling_parked.rs`) — and that route
//! stages every operand as one f32 lane per element. A parked `F16` parameter is
//! stored that way (the typed upload widens it, and the weight buffer sizes its
//! slot `ne * 4`), so it must run and compute the widened values; a parked *packed*
//! `U8` one is stored at a byte per element, so it must be refused loudly rather
//! than read four times past its length. One test per file: the flag is
//! process-global.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

const ROWS: usize = 4;
const COLS: usize = 64;

#[test]
fn a_packed_parameter_is_refused_and_a_widened_one_is_staged() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: one test per file, so nothing else in this process reads it. The flag
    // only makes the split planner park every parameter.
    unsafe { std::env::set_var("RLX_WGPU_ALL_PARAMS_WEIGHT", "1") };

    // Packed: one stored byte per element, so the f32-lane staging would read three
    // quarters of the operand from whatever follows it.
    let mut gp = Graph::new("parked-packed");
    let p = gp.param("p", Shape::new(&[ROWS, COLS], DType::U8));
    let cp = gp.cumsum(p, -1, false, Shape::new(&[ROWS, COLS], DType::U8));
    gp.set_outputs(vec![cp]);
    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(gp)));
    let Err(payload) = caught else {
        panic!("a parked U8 parameter was accepted instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("rlx-wgpu: REFUSED (operand not stageable as f32)"),
        "the refusal must carry a stable marker a caller can distinguish from a \
         process death; got: {msg}"
    );
    assert!(
        msg.contains("U8") && msg.contains("parked in the weight buffer"),
        "the refusal must name the operand, its dtype and where it lives; got: {msg}"
    );
    assert!(
        msg.contains("the upload stores it raw"),
        "the refusal must state why this parameter cannot be staged; got: {msg}"
    );

    // Widened: the same geometry with the two-byte dtype whose typed upload arm
    // widens it to one f32 lane per element, which the weight buffer's `ne * 4` slot
    // holds — so this must run, on the host route, over the widened values.
    let v: Vec<f32> = (0..ROWS * COLS).map(|i| (i % 16) as f32 * 0.25).collect();
    let f16: Vec<f32> = v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect();
    let mut want = vec![0.0f32; f16.len()];
    for r in 0..ROWS {
        let mut acc = 0.0f32;
        for c in 0..COLS {
            acc += f16[r * COLS + c];
            want[r * COLS + c] = acc;
        }
    }
    let build = |name: &str| {
        let mut g = Graph::new(name);
        let p = g.param("p", Shape::new(&[ROWS, COLS], DType::F16));
        let c = g.cumsum(p, -1, false, Shape::new(&[ROWS, COLS], DType::F16));
        g.set_outputs(vec![c]);
        g
    };
    // The direct entry makes the route visible: a parked parameter must reach the host
    // route, or the value assertion below says nothing about the staging.
    let direct = WgpuExecutable::compile(build("parked-f16"));
    assert!(
        direct.ceiling_hosted_steps() >= 1,
        "a parked parameter must route the op to the host route ({} hosted) — without \
         that, this case proves nothing",
        direct.ceiling_hosted_steps()
    );
    // And through `set_param_typed`, which is the arm the rule is about.
    let session = Session::new(Device::Gpu);
    let mut exe = session.compile(build("parked-f16-typed"));
    let bytes: Vec<u8> = f16
        .iter()
        .flat_map(|x| half::f16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    exe.set_param_typed("p", &bytes, DType::F16);
    assert_eq!(
        exe.run(&[])[0],
        want,
        "hosted running sum over a parked f16 parameter must read the widened values"
    );
}
