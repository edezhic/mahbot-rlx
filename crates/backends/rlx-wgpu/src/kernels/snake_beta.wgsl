// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// One DAC snake, `y = x + beta·sin²(alpha·x)`, as a single dispatch.
//
// The sound half's activation is written as five graph nodes — `arg = alpha·x`,
// `s = sin(arg)`, `s2 = s·s`, `scale = beta·s2`, `out = x + scale` — and the
// elementwise REGION kernel cannot carry it: `alpha`/`beta` are `[1, C, 1, 1]`
// against a `[1, C, 1, L]` output, and a flat `i % C` modulus walks the channel
// with the time axis on the way out (`region_input_modulus` refuses that
// mid-axis singleton), so the chain stays five dispatches over the same tensor.
// A DAC block has seven snakes.
//
// The layout this kernel relies on is the flattened one: element `(row c,
// column l)` sits at `base + c*inner + l`, which is exact for the contiguous
// `[1, C, 1, L]` the pattern is matched on. The channel index is therefore the
// row, and a 2-D grid (`gid.y` = row over `inner`-wide columns) reads `alpha[c]`
// and `beta[c]` without a division per element.
//
// The arithmetic is the chain's, in the chain's order, so the fused step's
// output is the unfused chain's output and not an approximation of it:
//
//   v = x[i] + bias[c]              (bias folded out of the snake's input Add)
//   s = sin(alpha[c]·v)
//   y = v + beta[c]·(s·s)           — the product groups as `(s·s)` first,
//                                     matching `s2 = s*s; scale = beta*s2`
//
// `has_bias` is 0 when the lowering did not fold the snake's input `Add`.

struct Params {
    outer: u32,        // rows = channels (C)
    inner: u32,        // columns per row (L)
    x_off: u32,        // f32-word offsets into the arena
    alpha_off: u32,
    beta_off: u32,
    bias_off: u32,     // read as `bias_off + row` when `has_bias` is set
    out_off: u32,
    has_bias: u32,     // 8 words: 32 bytes, 16-aligned for the uniform space
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// One workgroup per row covers `SNAKE_BETA_WG` columns of it; `run.rs` builds
// the grid from that width, and the crate's unit test holds this literal to it.
@compute @workgroup_size(256)
fn snake_beta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.outer || col >= params.inner) { return; }
    // Contiguous `[1, C, 1, L]`: the row is the channel and `inner` is its
    // stride, so this is also the address `alpha`/`beta` are indexed by.
    let i = row * params.inner + col;

    var v = arena[params.x_off + i];
    if (params.has_bias != 0u) {
        v = v + arena[params.bias_off + row];
    }
    let a = arena[params.alpha_off + row];
    let b = arena[params.beta_off + row];
    let s = sin(a * v);
    arena[params.out_off + i] = v + b * (s * s);
}
