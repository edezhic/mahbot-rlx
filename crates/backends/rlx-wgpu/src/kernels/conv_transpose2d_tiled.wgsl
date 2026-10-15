// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 2-D NCHW transposed conv as a phase-decomposed tiled GEMM over the arena.
//
// `conv_transpose3d.wgsl`'s `conv_transpose2d` entry point — the incumbent this
// kernel is a fast path over, not a replacement for — gives one thread one
// output element and walks *every* declared tap, letting the
// `(o + pad − k·dil) % stride` test drop the ones that do not land: at stride
// `s` it pays the index arithmetic of all `k` taps to keep `k/s` of them. On the
// decoder's `k = 2·s` forms that is half the walk thrown away, and it is what the
// six real forms in `tests/conv_transpose2d_route_cost.rs` read it by.
//
// Two things change here. The taps that land are walked and no others: for
// output position `ow` only `kx = r + s·m` lands, with `r = (ow + pad) mod s`,
// and grouping the output by `r` — its positions are `ow = e + s·u` — makes
// `kx` uniform across the group, so each phase is an ordinary GEMM
//
//     y[co, e + s·u] = Σ_ci Σ_m x[ci, off + u − m] · w[ci, co, r + s·m]
//
// with `e` the phase's first output position and `off = (e + pad − r)/s` its
// input shift. `grid.z` carries the phase, so one dispatch covers all `s` of
// them. And each operand load feeds a 4×4 register tile instead of a single
// accumulator.
//
// Both panels are stamped as complete tiles, so no lane's index depends on a
// per-phase count: the A panel zero-fills the input positions before the row's
// start and past its end, and the B panel is stamped with both of its rows
// whatever the phase — the row a phase has no tap for is zero and the phase's
// tap count bounds the loop that reads it. What the inner loop multiplies is
// therefore exactly the terms that land, branch-free.
//
// That is exactly equivalent to the gather for finite operands — the terms this
// kernel multiplies are the terms the gather sums, so a parity check derives its
// bar from them either way — but not for non-finite ones: a zero-filled A entry
// can still meet a non-finite weight where the gather's own skip drops the term
// instead. The B panel's own zero-fills cannot diverge: `ci ≥ c_in` pairs with an
// A entry that is zero for the same reason, `co ≥ c_out` is discarded by the
// store guard, and the `kx ≥ k` row is stamped but never walked. So what diverges
// is NaN in the output rather than a silently different number, and no shape in
// the family carries such an operand. The case is pinned by
// `tests/conv_transpose2d_gather_family.rs`'s
// `non_finite_weight_on_a_dropped_tap`.
//
// The named accumulators are load-bearing: a dynamically indexed local array is
// spilled to thread-local memory by the shader compiler, which costs an order
// of magnitude more than the arithmetic it saves.
//
// Applicability is `lower.rs`'s `conv_transpose2d_tiled_fits` — unit batch, unit
// height with no height padding, one group, no dilation, and `s ≤ k ≤ 2·s`, the
// last because the tile holds `TAPS` taps per output phase and that family gives
// every phase one or two of them, never none. Every other shape keeps the gather,
// which serves the whole operator. `TILE_U`/`TILE_CO` are declared a second time
// in `kernels/mod.rs`, whose `conv_transpose2d_tiled_grid` is the one place they
// become this kernel's dispatch and whose own test keeps the two copies equal.
//
// Adapted from the stand's device-against-host probe for this operation: the same
// tiling, panel staging and register tile, carried from that probe's three
// buffers to the arena's single binding, with the family it is proven for written
// down rather than assumed.

struct Params {
    n: u32, c_in: u32, c_out: u32,
    d: u32, h: u32, w: u32,
    d_out: u32, h_out: u32, w_out: u32,
    kd: u32, kh: u32, kw: u32,
    sd: u32, sh: u32, sw: u32,
    pd: u32, ph: u32, pw: u32,
    dd: u32, dh: u32, dw: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
    // Only the fields this step reads are declared, as in the sibling gather: the Rust
    // layout's `bias_off`/`has_bias` and the two words after them are not, since the
    // 2-D lowering leaves them zero and this step has no bias arm.
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Output positions and output channels per workgroup, and the workgroup's thread
// grid: each thread owns the 4×4 tile of accumulators below, and the two tile
// axes are strided by the lane counts in the inner loop, so no two lanes of a
// row read the same bank.
const TILE_U: u32 = 64u;
const TILE_CO: u32 = 64u;
const CHUNK: u32 = 16u;
const LANES_U: u32 = 16u;
const LANES_CO: u32 = 16u;
// Taps an output phase can carry (`k ≤ 2·s`), and the input span the A panel
// must hold for one chunk: the tile's positions plus the taps' reach.
const TAPS: u32 = 2u;
const APANEL: u32 = TILE_U + TAPS - 1u;
const LANES: u32 = LANES_U * LANES_CO;

var<workgroup> a_panel: array<f32, CHUNK * APANEL>;
var<workgroup> b_panel: array<f32, CHUNK * TAPS * TILE_CO>;

@compute @workgroup_size(16, 16, 1)
fn conv_transpose2d_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = tx + ty * LANES_U;

    let s = params.sw;
    let k = params.kw;
    let pad = params.pw;
    let w_in = params.w;
    let w_out = params.w_out;
    let c_in = params.c_in;
    let c_out = params.c_out;

    // This workgroup's phase, and the geometry it implies: the phase's first
    // output position `e`, its input shift `off`, and how many positions the
    // phase has. `e ≡ r − pad (mod s)`, so every position of the phase is
    // `e + s·u` and only the taps `kx = r + s·m` land on it.
    let r = wid.z;
    let e = (r + s - (pad % s)) % s;
    let off = (e + pad - r) / s;
    let n_r = select(0u, (w_out - e + s - 1u) / s, w_out > e);
    let u0 = wid.x * TILE_U;
    let co0 = wid.y * TILE_CO;
    if (u0 >= n_r) { return; }

    var a00: f32 = 0.0; var a01: f32 = 0.0; var a02: f32 = 0.0; var a03: f32 = 0.0;
    var a10: f32 = 0.0; var a11: f32 = 0.0; var a12: f32 = 0.0; var a13: f32 = 0.0;
    var a20: f32 = 0.0; var a21: f32 = 0.0; var a22: f32 = 0.0; var a23: f32 = 0.0;
    var a30: f32 = 0.0; var a31: f32 = 0.0; var a32: f32 = 0.0; var a33: f32 = 0.0;

    // This phase's taps: `kx = r + s·m < k`, so `m < ceil((k − r)/s)` — the
    // second B row is walked only when it is a real tap, and a phase with no tap
    // left (`r ≥ k`, which needs `k < s`, outside the family
    // `conv_transpose2d_tiled_fits` admits) walks none and stores zeros. Skipping
    // a phantom row rather than letting its zero weight stand is not the same
    // operator on non-finite operands: the zero-filled row would turn a
    // non-finite input into `0 · inf = NaN` where this route's incumbent, which
    // skips the tap, stays finite.
    var taps: u32 = 0u;
    if (r < k) {
        taps = (k - r + s - 1u) / s;
    }

    var c0: u32 = 0u;
    loop {
        if (c0 >= c_in) { break; }

        // A panel: the input row this chunk's channels contribute, one entry
        // per (channel, position of the tile widened by the taps' reach).
        for (var idx: u32 = tid; idx < CHUNK * APANEL; idx = idx + LANES) {
            let c = idx / APANEL;
            let pp = idx % APANEL;
            let ci = c0 + c;
            let iw = i32(u0) + i32(pp) - i32(TAPS - 1u) + i32(off);
            var v = 0.0;
            if (ci < c_in && iw >= 0 && iw < i32(w_in)) {
                v = arena[params.in_off + ci * w_in + u32(iw)];
            }
            a_panel[idx] = v;
        }

        // B panel: this chunk's weight slab, `TAPS` rows of `co` per channel.
        // The `m`-th row is the tap `r + s·m`, stamped down as a full `TAPS`-row
        // panel by one loop so that no lane has to agree on `taps` to index it —
        // the row a phase does not walk is filled with zeros and never read.
        for (var idx: u32 = tid; idx < CHUNK * TAPS * TILE_CO; idx = idx + LANES) {
            let kk = idx / TILE_CO;
            let i = idx % TILE_CO;
            let c = kk / TAPS;
            let m = kk % TAPS;
            let ci = c0 + c;
            let co = co0 + i;
            let kx = r + s * m;
            var v = 0.0;
            if (ci < c_in && co < c_out && kx < k) {
                v = arena[params.w_off + (ci * c_out + co) * k + kx];
            }
            b_panel[idx] = v;
        }
        workgroupBarrier();

        for (var m: u32 = 0u; m < taps; m = m + 1u) {
            let nb = m * TILE_CO + ty;
            for (var c: u32 = 0u; c < CHUNK; c = c + 1u) {
                // A entry for the position `u0 + tx + i·LANES_U` at tap `m`:
                // `off + u − m` is the panel's `tx + (TAPS − 1) − m` entry.
                let na = c * APANEL + tx + (TAPS - 1u) - m;
                let va0 = a_panel[na];
                let va1 = a_panel[na + LANES_U];
                let va2 = a_panel[na + 2u * LANES_U];
                let va3 = a_panel[na + 3u * LANES_U];
                let nb0 = c * TAPS * TILE_CO + nb;
                let vb0 = b_panel[nb0];
                let vb1 = b_panel[nb0 + LANES_CO];
                let vb2 = b_panel[nb0 + 2u * LANES_CO];
                let vb3 = b_panel[nb0 + 3u * LANES_CO];
                a00 = a00 + va0 * vb0; a01 = a01 + va1 * vb0;
                a02 = a02 + va2 * vb0; a03 = a03 + va3 * vb0;
                a10 = a10 + va0 * vb1; a11 = a11 + va1 * vb1;
                a12 = a12 + va2 * vb1; a13 = a13 + va3 * vb1;
                a20 = a20 + va0 * vb2; a21 = a21 + va1 * vb2;
                a22 = a22 + va2 * vb2; a23 = a23 + va3 * vb2;
                a30 = a30 + va0 * vb3; a31 = a31 + va1 * vb3;
                a32 = a32 + va2 * vb3; a33 = a33 + va3 * vb3;
            }
        }
        workgroupBarrier();
        c0 = c0 + CHUNK;
    }

    // Each phase's positions are `e + s·u`, and `u < n_r` is the whole bound on
    // them: `n_r = ⌈(w_out − e)/s⌉` puts `e + s·(n_r − 1)` below `w_out`, so a
    // position this stores is inside the row and the last tile's overrun needs no
    // second test. `w_out` is the *declared* output width, so `out_pad` needs no
    // handling of its own — it is already the length of the row. `co < c_out` is
    // the channel tiles past the last one.
    let ua = u0 + tx;
    let ub = u0 + tx + LANES_U;
    let uc = u0 + tx + 2u * LANES_U;
    let ud = u0 + tx + 3u * LANES_U;
    let co_a = co0 + ty;
    let co_b = co0 + ty + LANES_CO;
    let co_c = co0 + ty + 2u * LANES_CO;
    let co_d = co0 + ty + 3u * LANES_CO;
    if (co_a < c_out) {
        if (ua < n_r) { let ow = e + s * ua; arena[params.out_off + co_a * w_out + ow] = a00; }
        if (ub < n_r) { let ow = e + s * ub; arena[params.out_off + co_a * w_out + ow] = a01; }
        if (uc < n_r) { let ow = e + s * uc; arena[params.out_off + co_a * w_out + ow] = a02; }
        if (ud < n_r) { let ow = e + s * ud; arena[params.out_off + co_a * w_out + ow] = a03; }
    }
    if (co_b < c_out) {
        if (ua < n_r) { let ow = e + s * ua; arena[params.out_off + co_b * w_out + ow] = a10; }
        if (ub < n_r) { let ow = e + s * ub; arena[params.out_off + co_b * w_out + ow] = a11; }
        if (uc < n_r) { let ow = e + s * uc; arena[params.out_off + co_b * w_out + ow] = a12; }
        if (ud < n_r) { let ow = e + s * ud; arena[params.out_off + co_b * w_out + ow] = a13; }
    }
    if (co_c < c_out) {
        if (ua < n_r) { let ow = e + s * ua; arena[params.out_off + co_c * w_out + ow] = a20; }
        if (ub < n_r) { let ow = e + s * ub; arena[params.out_off + co_c * w_out + ow] = a21; }
        if (uc < n_r) { let ow = e + s * uc; arena[params.out_off + co_c * w_out + ow] = a22; }
        if (ud < n_r) { let ow = e + s * ud; arena[params.out_off + co_c * w_out + ow] = a23; }
    }
    if (co_d < c_out) {
        if (ua < n_r) { let ow = e + s * ua; arena[params.out_off + co_d * w_out + ow] = a30; }
        if (ub < n_r) { let ow = e + s * ub; arena[params.out_off + co_d * w_out + ow] = a31; }
        if (uc < n_r) { let ow = e + s * uc; arena[params.out_off + co_d * w_out + ow] = a32; }
        if (ud < n_r) { let ow = e + s * ud; arena[params.out_off + co_d * w_out + ow] = a33; }
    }
}
