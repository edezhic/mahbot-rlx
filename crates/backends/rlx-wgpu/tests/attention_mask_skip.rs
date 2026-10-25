// RLX — versatile ML compiler & runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The decode entry's masked-key stand-down: a binary key-padding mask (kind 2)
//! states which keys a row may attend to, and a masked key's score is the
//! kernel's sentinel whatever its K dot product says. Once the lane's running
//! max has been set by a valid key, such a key's `e_cur` is exactly 0 and its
//! `e_old` exactly 1, so its K and V reads cannot reach `m`, `l` or `O` and the
//! kernel stands them down. (A lane whose *first* key is masked still raises the
//! max to the sentinel with `e_cur = 1`, so it reads V exactly as before — the
//! degenerate row's own arithmetic, unchanged.) The body's fixed-capacity plane
//! cache is the shape this pays on — the rows past the ones the step has written
//! are exactly those masked keys, and its own query slot is always valid.
//! `RLX_WGPU_NO_ATTN_MASK_SKIP=1` restores the masked key's K dot; its V read is
//! stood down either way by the separate, ungated `e_cur == 0` early-out, so the
//! switch prices the K side only. Both arms compute the CPU reference's numbers,
//! bit for bit the same: the skipped keys contributed `e_cur = 0` and `0 · V`,
//! which is what makes the switch safe to hold a frame to.
mod common;

use common::{ATTENTION_BAR, data, max_abs_diff};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
/// More than two passes of the entry's 64 lanes, so a lane walks several keys:
/// at one key a lane every masked key is a *first* masked key (whose `e_cur` is
/// 1, not 0), and the V-read stand-down would never be reached.
const KEYS: usize = 200;
const WRITTEN: usize = 61;

/// A one-row `[1, 1, H, D]` step over `KEYS` keys, in the plane cache's own
/// layout (`[1, keys, H, D]` K/V, a `[1, keys]` binary mask), with the mask the
/// body feeds: the written prefix is `1`, every unwritten row is `0`.
fn one_row() -> (Graph, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (h, d) = (HEADS, HEAD_DIM);
    let mut g = Graph::new("attn_mask_skip");
    let q = g.input("q", Shape::new(&[1, 1, h, d], DType::F32));
    let k = g.input("k", Shape::new(&[1, KEYS, h, d], DType::F32));
    let v = g.input("v", Shape::new(&[1, KEYS, h, d], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, KEYS], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        Shape::new(&[1, 1, h, d], DType::F32),
    );
    g.set_outputs(vec![y]);
    let q = data(h * d, 0.011, false);
    let k = data(KEYS * h * d, 0.013, true);
    let v = data(KEYS * h * d, 0.017, false);
    let mask = (0..KEYS)
        .map(|i| if i < WRITTEN { 1.0 } else { 0.0 })
        .collect();
    (g, q, k, v, mask)
}

/// The switches at their shipped values: the arm an arm is compared against must
/// not inherit the operator's, or the two arms are one arm.
const SHIPPED: [(&str, &str); 2] = [
    ("RLX_WGPU_NO_ATTN_MASK_SKIP", "0"),
    // At `1` a one-row step takes the many-row arrangement and this kernel is
    // never dispatched, so both arms would compare something else.
    ("RLX_WGPU_ATTN_DENSE_ONE_ROW", "0"),
];

/// The same, with the masked key's K dot put back. Spelled out rather than
/// chained onto [`SHIPPED`]: a repeated key restores the earlier value first, so a
/// chain would leave the process holding this arm's `1`.
const READ_ALL: [(&str, &str); 2] = [
    ("RLX_WGPU_NO_ATTN_MASK_SKIP", "1"),
    ("RLX_WGPU_ATTN_DENSE_ONE_ROW", "0"),
];

/// One run of `g` on the device, compiled under whatever overrides are installed.
fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut c = Session::new(Device::Gpu).compile(g);
    c.run(inputs).remove(0)
}

#[test]
fn a_masked_key_costs_no_read_and_changes_no_value() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let (g, q, k, v, mask) = one_row();
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v), ("mask", &mask)];
    let cpu = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&inputs)
        .remove(0);

    // The switch is read at the lowering, so the two arms compile apart — and this
    // is what holds it there: a switch that stopped reaching the step would give
    // two identical compiles and identical values, and pass every check below
    // without standing a read down.
    let words = |overrides: [(&str, &str); 2]| {
        let _guard = rlx_ir::env::RuntimeOverrides::install(overrides);
        common::legalised_executable(&g).test_attention_skip_masked()
    };
    let shipped = words(SHIPPED);
    let restored = words(READ_ALL);
    eprintln!("[rlx-attn-mask-skip] compiled shipped={shipped:?} stand_down={restored:?}");
    assert!(
        !shipped.is_empty() && !restored.is_empty(),
        "no attention step in the compiled form: shipped {shipped:?}, stand-down {restored:?}"
    );
    assert!(
        shipped.iter().all(|w| *w == 1) && restored.iter().all(|w| *w == 0),
        "the mask-skip switch did not reach the compiled step: shipped {shipped:?}, \
         stand-down {restored:?}"
    );

    let skipped = {
        let _shipped = rlx_ir::env::RuntimeOverrides::install(SHIPPED);
        run(g.clone(), &inputs)
    };
    let read_all = {
        let _read_all = rlx_ir::env::RuntimeOverrides::install(READ_ALL);
        run(g.clone(), &inputs)
    };
    for (arm, out) in [("skipped", &skipped), ("read_all", &read_all)] {
        let d = max_abs_diff(out, &cpu);
        eprintln!(
            "[rlx-attn-mask-skip] arm={arm} keys={KEYS} written={WRITTEN} \
             heads={HEADS} head_dim={HEAD_DIM} max_abs={d:.6e}"
        );
        assert!(
            d < ATTENTION_BAR,
            "{arm}: max_abs {d} against the CPU reference"
        );
    }
    // `e_cur = 0` and `0 · V` are exact, so the stand-down is a read count and
    // nothing else — this is the equality the plane-cache digest rests on.
    assert_eq!(
        skipped, read_all,
        "skipping a masked key moved a value the frame computes"
    );
}
