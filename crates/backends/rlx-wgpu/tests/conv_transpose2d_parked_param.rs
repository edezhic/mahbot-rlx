// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d` when its weight is parked in the separate weight buffer.
//!
//! `from_plan_split` parks parameters whenever the arena needs the split plan and
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` forces that on any adapter. The op's only route
//! addresses its operands by 32-bit byte offsets into the activation arena, which
//! drop `WEIGHT_BUF_TAG`, so this geometry must be refused loudly rather than read
//! from the wrong bytes.
//!
//! Its own test binary: the switch is process-global and lowering reads it at the
//! first compile.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn a_parked_upsampler_weight_is_refused_loudly() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");

    let mut g = Graph::new("conv_transpose2d_parked");
    let x = g.input("x", Shape::new(&[1, 4, 1, 8], DType::F32));
    let w = g.param("w", Shape::new(&[4, 2, 1, 3], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, 3], [1, 2], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);

    let caught =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| WgpuExecutable::compile(g)));
    let Err(payload) = caught else {
        panic!("a parked weight was accepted instead of refused");
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("rlx-wgpu: REFUSED (parked"),
        "refusal must carry the stable marker; got: {msg}"
    );
    assert!(
        msg.contains("ConvTranspose2d"),
        "the refusal must name the op; got: {msg}"
    );
    assert!(
        msg.contains("RLX_WGPU_ALL_PARAMS_WEIGHT"),
        "the refusal must name the opt-in that reaches it; got: {msg}"
    );
}
