// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parked-operand half of the per-binding-ceiling rule, in its own process.
//!
//! `lower::should_host_at_ceiling` decides on two geometries. The striping one
//! needs a shrunken ceiling (`ceiling_route.rs`); this is the other: a parameter
//! parked in the separate weight buffer, whose `Arena::offset` is
//! `WEIGHT_BUF_TAG`-tagged and therefore cannot be handed to a kernel bound to the
//! activation arena through a `u32` uniform. The arena here fits one bind window,
//! so any hosting can only have come from the parked operand — the input-only
//! control below pins exactly that.
//!
//! `RLX_WGPU_ALL_PARAMS_WEIGHT` is process-global, hence this file.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
const ROWS: usize = 4;
const COLS: usize = 64;

/// Exactly representable operands, so the running sums below are exact and the
/// assertion can be an equality. Sums stay under 2^24.
fn values() -> Vec<f32> {
    (0..ROWS * COLS).map(|i| (i % 16) as f32).collect()
}

fn running_sums(v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for r in 0..ROWS {
        let mut acc = 0.0f32;
        for c in 0..COLS {
            acc += v[r * COLS + c];
            out[r * COLS + c] = acc;
        }
    }
    out
}

#[test]
fn a_parked_parameter_sends_a_listed_op_to_the_host_route() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: one test per file, so nothing else in this process reads it. The flag
    // only makes the split planner park every parameter — the geometry the default
    // route (no switches set) reaches when a plan exceeds the bind window.
    unsafe { std::env::set_var("RLX_WGPU_ALL_PARAMS_WEIGHT", "1") };

    let v = values();
    let want = running_sums(&v);

    // Parked parameter: the op's only operand lives in the weight buffer.
    let mut gp = Graph::new("parked-param");
    let p = gp.param("p", Shape::new(&[ROWS, COLS], F));
    let cp = gp.cumsum(p, -1, false, Shape::new(&[ROWS, COLS], F));
    gp.set_outputs(vec![cp]);
    let mut exe = WgpuExecutable::compile(gp);
    exe.set_param("p", &v);
    let got = exe.run(&[]);
    assert!(
        exe.ceiling_hosted_steps() >= 1,
        "a parked parameter must route the op to the host route, or this test proves nothing \
         ({} step(s) hosted)",
        exe.ceiling_hosted_steps()
    );
    assert_eq!(got[0], want, "hosted running sum over a parked parameter");

    // Control: the same op over an INPUT, same lever. Nothing is parked and the arena
    // fits one bind window, so the rule must not fire — which is what makes the
    // hosting above attributable to the parked operand rather than to the geometry.
    let mut gi = Graph::new("act-input");
    let a = gi.input("a", Shape::new(&[ROWS, COLS], F));
    let ci = gi.cumsum(a, -1, false, Shape::new(&[ROWS, COLS], F));
    gi.set_outputs(vec![ci]);
    let mut exe = WgpuExecutable::compile(gi);
    let got = exe.run(&[("a", &v)]);
    assert_eq!(
        exe.ceiling_hosted_steps(),
        0,
        "an operand in the activation arena must not be hosted"
    );
    assert_eq!(got[0], want, "GPU running sum over an activation operand");
}
