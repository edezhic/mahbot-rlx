// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The two fusion stand-downs the synthesis census prices:
//! `RLX_WGPU_NO_FUSED_SWIGLU` expands `Op::FusedSwiGLU` into narrow, silu and
//! multiply, and `RLX_WGPU_NO_FUSED_RESIDUAL_RMS` expands
//! `Op::FusedResidualRmsNorm` into its add and its norm. Both are measurement
//! arms, and both must compute the fused step's numbers — the census holds the
//! arms to that on a whole predictor frame; this file pins it on one node, on the
//! device, with no model and no frame.
//!
//! A `gate_first` SwiGLU is the case that matters: the expansion used to read up
//! from the low half regardless of the flag, which is `silu(up) · gate` and not a
//! rounding of the operation (`crates/core/rlx-fusion/tests/fused_swiglu_unfuse_order.rs`
//! pins the decomposition itself). The GPU's native fused kernel is the reference
//! here rather than the CPU executor because both arms must run the same backend:
//! the switch picks a decomposition, not a device, so a device-to-device
//! comparison is the fused step's own arithmetic in another set of steps.
//!
//! Both arms state their switch and both schedules are asserted — the shipped arm
//! compiles the fused step and the stood-down arm does not — because the
//! stand-down is read at the lowering, so an arm that inherited the operator's
//! value, or a switch that stopped reaching the lowering, would compare one
//! compiled form against itself. The override map is process-global, so the two
//! tests take `common::route_lock()`.
mod common;

use common::{data, max_abs_diff, route_lock};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// The two forms are the same arithmetic in different steps: a device-to-device
/// comparison of one node should agree to the last bits, and this bar leaves
/// room only for another summation order.
const BAR: f32 = 1e-6;

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn gpu(g: &Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(Device::Gpu)
        .compile(g.clone())
        .run(inputs)
        .remove(0)
}

/// The two arms' schedules for one node, asserting the stand-down reached the
/// lowering: the shipped arm must compile a `fused` step and the stood-down arm
/// must not, or a value comparison would be one compiled form against itself.
fn arm_schedules(g: &Graph, switch: &str, fused: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    let shipped = {
        let _on = rlx_ir::env::RuntimeOverrides::install([(switch, "0")]);
        common::legalised_step_names(g)
    };
    let stood_down = {
        let _off = rlx_ir::env::RuntimeOverrides::install([(switch, "1")]);
        common::legalised_step_names(g)
    };
    assert!(
        shipped.contains(&fused),
        "the shipped arm compiled no `{fused}` step ({shipped:?})"
    );
    assert!(
        !stood_down.contains(&fused),
        "{switch} left the `{fused}` step in the schedule ({stood_down:?})"
    );
    (shipped, stood_down)
}

/// Whether the two arms of one node agree, with the difference and both schedules
/// printed. `run` executes the node under whatever overrides are installed.
fn agree(name: &str, g: &Graph, switch: &str, fused: &str, run: impl Fn() -> Vec<f32>) {
    let (shipped, stood_down) = arm_schedules(g, switch, fused);
    let native = {
        let _on = rlx_ir::env::RuntimeOverrides::install([(switch, "0")]);
        run()
    };
    let expanded = {
        let _off = rlx_ir::env::RuntimeOverrides::install([(switch, "1")]);
        run()
    };
    let d = max_abs_diff(&expanded, &native);
    eprintln!(
        "[rlx-fusion-standdown] node={name} switch={switch} shipped={shipped:?} \
         stood_down={stood_down:?} max_abs={d:.6e}"
    );
    assert!(
        d < BAR,
        "{name}: the stand-down moved a value (max_abs {d:e} against the fused step)"
    );
}

#[test]
fn the_swiglu_stand_down_computes_the_fused_steps_numbers() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _gate = route_lock();
    // Both concat orders: `gate_first` is what the expansion used to ignore.
    for gate_first in [false, true] {
        let mut g = Graph::new("swiglu_standdown");
        let x = g.input("x", shape(&[2, 8]));
        let y = g.add_node(
            Op::FusedSwiGLU {
                cast_to: None,
                gate_first,
            },
            vec![x],
            shape(&[2, 4]),
        );
        g.set_outputs(vec![y]);
        let x = data(16, 0.019, gate_first);
        agree(
            if gate_first {
                "fused_swiglu_gate_first"
            } else {
                "fused_swiglu_up_first"
            },
            &g,
            "RLX_WGPU_NO_FUSED_SWIGLU",
            "fused_swiglu",
            || gpu(&g, &[("x", &x)]),
        );
    }
}

#[test]
fn the_residual_rms_stand_down_computes_the_fused_steps_numbers() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let _gate = route_lock();
    let mut g = Graph::new("residual_rms_standdown");
    let x = g.input("x", shape(&[2, 16]));
    let residual = g.input("residual", shape(&[2, 16]));
    let gamma = g.param("gamma", shape(&[16]));
    let beta = g.param("beta", shape(&[16]));
    let y = g.add_node(
        Op::FusedResidualRmsNorm {
            has_bias: false,
            eps: 1e-6,
        },
        vec![x, residual, gamma, beta],
        shape(&[2, 16]),
    );
    g.set_outputs(vec![y]);
    let x = data(32, 0.011, false);
    let residual = data(32, 0.013, true);
    let gamma: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.05).collect();
    let beta: Vec<f32> = (0..16).map(|i| 0.1 - i as f32 * 0.01).collect();
    agree(
        "fused_residual_rms",
        &g,
        "RLX_WGPU_NO_FUSED_RESIDUAL_RMS",
        "fused_residual_rms_norm",
        || {
            let mut c = Session::new(Device::Gpu).compile(g.clone());
            c.set_param("gamma", &gamma);
            c.set_param("beta", &beta);
            c.run(&[("x", &x), ("residual", &residual)]).remove(0)
        },
    );
}
