// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// The frame matrix of a mel/STFT front-end, written straight from the signal.
//
// The per-frame form of this is one `Step::Narrow` per frame plus one
// `Step::Concat` input per frame — `n_frames` of each — for a matrix that is a
// single strided read of the source. This kernel is the lowering's fused spelling
// of exactly that pattern (`lower.rs`, `detect_frames_pattern`): it reads the
// source with the same addressing the narrows did, so it needs no index table —
// which is what the batched-`Op::Gather` spelling of the same matrix buys and
// pays for (see this crate's README for the figures).
// `outer` is the source's leading-dims product, so the destination is
// [n_frames, outer, frame_len] in row-major order and the source element for
// (f, o, j) is at `o * t + f * hop + j`. One dispatch both reads and writes the
// arena, so the destination must not overlap the source: lowering refuses that case
// against the plan it lowers from — the plan predates the fold, so its liveness for
// the source may already have handed those bytes on — rather than dispatching into
// a race.

// The invocation count per workgroup, declared here and in `FRAMES_WORKGROUP`
// (one number in two languages, pinned by `frames_shader_matches_its_dispatch_-
// geometry`): the dispatch covers `total` elements in groups of `WG`, and this
// kernel linearises by the same stride.
const WG: u32 = 64u;

struct Params {
    src_off: u32,   // f32 word offset of the source signal [.., t]
    dst_off: u32,   // f32 word offset of the frame matrix [n_frames, .., frame_len]
    t: u32,         // source last-axis length
    outer: u32,     // product of the source's leading dims (>= 1)
    frame_len: u32,
    hop: u32,
    total: u32,     // outer · n_frames · frame_len: the covered prefix, in (f, .., j) order
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(WG)
fn frames(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let lin = gid.x + gid.y * nwg.x * WG;
    if (lin >= params.total) { return; }

    let j = lin % params.frame_len;
    let o = (lin / params.frame_len) % params.outer;
    let f = lin / (params.frame_len * params.outer);
    arena[params.dst_off + lin] = arena[params.src_off + o * params.t + f * params.hop + j];
}
