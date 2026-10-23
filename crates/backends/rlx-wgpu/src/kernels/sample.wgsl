// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Multinomial sampling kernel with optional top-k / top-p / temperature.
// One thread per batch row; serial inside each row (vocab N typically
// 32k–50k — fast enough on a single SIMD lane for batch=1 inference).
//
// A one-row dispatch (`outer == 1`) instead splits the row across the
// workgroup: the draw sits on the critical path of a decode step, where the row
// is browsed once per sampled token. Every reduction that decides
// the drawn id keeps the serial order — the exponent sum and the
// renormalisation sum stay lane-0 loops in index order, the max scans keep the
// lower index on ties — so the cooperative path picks exactly the id the serial
// one does. A row that fits `STAGE` is staged in workgroup storage first, so
// those passes read it instead of the arena. That scratch — the 8 KiB row, the
// selection tree's per-lane lists and the picked indices, about 11 KiB — is
// reserved by every dispatch of this entry point, the per-row path included.
//
// Algorithm:
//   1. Scale logits by 1/temperature into the scratch slot; track row max.
//   2. Convert to probabilities (exp(x - max) / sum_exp) in the scratch
//      slot. The logits are read-only, as on CPU and Metal: nothing here
//      clobbers the input, so every other consumer of the same logits
//      still sees the original values.
//   3. If top_k > 0 OR top_p < 1: do a serial selection-sort of the
//      probabilities; for each pick mark as "selected" by storing the
//      probability as a negative value. Stop when we've collected k
//      tokens *and* cumulative ≥ top_p.
//   4. Renormalize the selected set; zero the rest.
//   5. Sample by Gumbel-max over Threefry-2×32-20 noise (see below).

struct Params {
    outer: u32,
    inner: u32,        // vocab
    in_off: u32,
    out_off: u32,
    top_k: u32,
    top_p_bits: u32,   // bitcast<f32>
    temp_bits: u32,    // bitcast<f32>
    seed_lo: u32,
    seed_hi: u32,
    scratch_off: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const NEG_INF: f32 = -3.4e38;
const SENTINEL_EPS: f32 = 1e-30;

// ── Threefry-2×32-20 ─────────────────────────────────────────────
//
// Counter-based PRNG, same family MLX (and JAX) use. Stateless: feed
// in (counter, key), get out two 32-bit words. 20 rounds with key
// injections every 4 rounds.
//
// Reference: Salmon et al. "Parallel random numbers: as easy as 1, 2, 3"
// (Random123 paper, 2011). Rotation constants R_2x32 verbatim from
// the spec; KS_PARITY is the 32-bit Skein parity word.

fn rotl32(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry2x32_20(c_in: vec2<u32>, k_in: vec2<u32>) -> vec2<u32> {
    let KS_PARITY: u32 = 0x1BD11BDAu;
    let ks0 = k_in.x;
    let ks1 = k_in.y;
    let ks2 = ks0 ^ ks1 ^ KS_PARITY;

    var x0 = c_in.x + ks0;
    var x1 = c_in.y + ks1;

    // Rotation schedule: 8 constants, cycled across the 20 rounds.
    // The macro below interleaves (ks-extend round + injection) so
    // we don't need an `if` branch each round.

    // Rounds 1-4
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 1
    x0 = x0 + ks1; x1 = x1 + ks2; x1 = x1 + 1u;

    // Rounds 5-8
    x0 = x0 + x1; x1 = rotl32(x1, 17u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 29u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 16u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 24u); x1 = x1 ^ x0;
    // Inject 2
    x0 = x0 + ks2; x1 = x1 + ks0; x1 = x1 + 2u;

    // Rounds 9-12
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 3
    x0 = x0 + ks0; x1 = x1 + ks1; x1 = x1 + 3u;

    // Rounds 13-16
    x0 = x0 + x1; x1 = rotl32(x1, 17u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 29u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 16u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 24u); x1 = x1 ^ x0;
    // Inject 4
    x0 = x0 + ks1; x1 = x1 + ks2; x1 = x1 + 4u;

    // Rounds 17-20
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 5
    x0 = x0 + ks2; x1 = x1 + ks0; x1 = x1 + 5u;

    return vec2<u32>(x0, x1);
}

/// One index's Gumbel-max score: `log p + g`, with `g` from the counter-based
/// draw the sampler (and the host reference it is checked against) uses. `ctr` is
/// the row — `0` for a one-row draw, the batch index on the per-row path, which
/// is what makes the two draw the same uniform stream row for row.
fn gumbel_score(p: f32, i: u32, ctr: u32, key: vec2<u32>) -> f32 {
    let rng2 = threefry2x32_20(vec2<u32>(ctr, i), key);
    // The high 24 bits as a uniform on [0, 1); `1e-30` guards the log of a zero.
    let u: f32 = f32(rng2.x >> 8u) / 16777216.0;
    let g: f32 = -log(-log(max(u, 1e-30)));
    return log(p) + g;
}

/// The workgroup width of every entry point in this module, written once: the
/// scratch arrays, the cooperative scans' strides and the grid's row index are
/// all this number (it is also each `@workgroup_size`). The dispatch is computed
/// with `kernels::SAMPLE_LANES`, which a unit test holds to this literal.
const LANES: u32 = 64u;

/// The widest row `sample_one_staged` can hold in workgroup storage (8 KiB of the
/// device's 16 KiB minimum). A wider one-row draw keeps the global-memory path;
/// `kernels::SAMPLE_STAGE` is this number, held to it by a unit test.
const STAGE: u32 = 2048u;

/// The picked indices a draw records, so the restore sum and the discharge can
/// visit the selected set instead of the row. A wider `top_k` (or `top_k == 0`
/// with `top_p < 1`, whose count is the row's) keeps the row scans.
const PICK_MAX: u32 = 64u;

/// A lane's candidate list: descending by value, ascending by index on a tie. Four
/// entries — one round of the selection tree's pick count. A function-scope list is
/// indexed by constant only: a runtime index would put it in function-local memory,
/// and the scan pushes into it once per element. The list is constructible, so
/// `wg4` holds the lanes' lists as they are.
struct Cand {
    v: array<f32, 4>,
    i: array<u32, 4>,
}

// Workgroup scratch for a one-row draw. `wg_best` reduces (value, index) pairs;
// `wg_one` broadcasts the one scalar the serial order must produce on lane 0.
var<workgroup> wg_best_v: array<f32, LANES>;
var<workgroup> wg_best_i: array<u32, LANES>;
var<workgroup> wg_one: f32;
// The staged row: a one-row draw's probabilities, in the workgroup's own address
// space instead of the arena. Every pass below touches it `LANES`-strided, so the
// two serial reductions are the only lane-0 work left.
var<workgroup> row: array<f32, STAGE>;
// The selection tree's per-lane candidate lists, and the picked indices of a
// draw whose `k_limit` fits `PICK_MAX` (ascending, so the restore sums them in
// the serial pass's own order).
var<workgroup> wg4: array<Cand, LANES>;
var<workgroup> picks: array<u32, PICK_MAX>;

/// The best **value** over the workgroup: higher wins, and on an exact tie the
/// **lower index** wins — the serial scan's `v > best` order. The index is
/// selected and carried as a `u32` (`wg_best_i[0]`); only the output slot the
/// caller finally writes it to is `f32`, which is not what indexes a row.
fn wg_best(lane: u32, value: f32, index: u32) -> f32 {
    wg_best_v[lane] = value;
    wg_best_i[lane] = index;
    workgroupBarrier();
    var stride: u32 = LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let ov = wg_best_v[lane + stride];
            let oi = wg_best_i[lane + stride];
            let sv = wg_best_v[lane];
            let si = wg_best_i[lane];
            if (ov > sv || (ov == sv && oi < si)) {
                wg_best_v[lane] = ov;
                wg_best_i[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return wg_best_v[0];
}

// ── the selection's candidates ───────────────────────────────────
//
// (value desc, index asc) is a total order, so the row's next four values are
// the top four of the union of the lanes' own top four — a tree of merges,
// one barrier per level. That is what makes a pick cheap: one round of the tree
// yields four picks, and a round costs the same as the single-pick reduce it
// replaces.

fn cand_better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
    return av > bv || (av == bv && ai < bi);
}

fn cand_empty() -> Cand {
    return Cand(array<f32, 4>(-1.0, -1.0, -1.0, -1.0), array<u32, 4>(0u, 0u, 0u, 0u));
}

/// Insert one candidate into a list that is already in order: the newcomer takes
/// the tail if it beats it, then bubbles up while it beats the one above. The list
/// stays the four best of what has been offered, which is what the merge relies on.
fn cand_push(c: Cand, v: f32, i: u32) -> Cand {
    var r = c;
    if (!cand_better(v, i, r.v[3], r.i[3])) { return r; }
    r.v[3] = v;
    r.i[3] = i;
    if (cand_better(r.v[3], r.i[3], r.v[2], r.i[2])) {
        let tv1 = r.v[2];
        let ti1 = r.i[2];
        r.v[2] = r.v[3]; r.i[2] = r.i[3];
        r.v[3] = tv1; r.i[3] = ti1;
        if (cand_better(r.v[2], r.i[2], r.v[1], r.i[1])) {
            let tv2 = r.v[1];
            let ti2 = r.i[1];
            r.v[1] = r.v[2]; r.i[1] = r.i[2];
            r.v[2] = tv2; r.i[2] = ti2;
            if (cand_better(r.v[1], r.i[1], r.v[0], r.i[0])) {
                let tv3 = r.v[0];
                let ti3 = r.i[0];
                r.v[0] = r.v[1]; r.i[0] = r.i[1];
                r.v[1] = tv3; r.i[1] = ti3;
            }
        }
    }
    return r;
}

fn cand_merge(a: Cand, b: Cand) -> Cand {
    var r = cand_push(a, b.v[0], b.i[0]);
    r = cand_push(r, b.v[1], b.i[1]);
    r = cand_push(r, b.v[2], b.i[2]);
    r = cand_push(r, b.v[3], b.i[3]);
    return r;
}

/// `outer == 1`: the whole workgroup draws row 0. The loops whose *order*
/// decides the bits (the exponent sum, the renormalisation sum, the pick order)
/// stay serial on lane 0; the scans are split across `LANES`.
fn sample_one_coop(lane: u32) {
    let inner = params.inner;
    let base = params.in_off;
    let sbase = params.scratch_off;
    let temp = bitcast<f32>(params.temp_bits);
    let top_p = bitcast<f32>(params.top_p_bits);
    let inv_temp = 1.0 / max(temp, 1e-6);

    // 1. Apply temperature into scratch; row max.
    var local_max: f32 = NEG_INF;
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        let v = arena[base + i] * inv_temp;
        arena[sbase + i] = v;
        local_max = max(local_max, v);
    }
    let m: f32 = wg_best(lane, local_max, 0u);

    // 2. exp into scratch, then the sum in index order on lane 0.
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        arena[sbase + i] = exp(arena[sbase + i] - m);
    }
    storageBarrier();
    if (lane == 0u) {
        var sum_e: f32 = 0.0;
        for (var i: u32 = 0u; i < inner; i = i + 1u) {
            sum_e = sum_e + arena[sbase + i];
        }
        wg_one = sum_e;
    }
    workgroupBarrier();
    // Lane 0 read the scratch in the sum above; the rescale loop writes it, so
    // the storage accesses need their own barrier — `workgroupBarrier` orders
    // the workgroup address space only.
    storageBarrier();
    let inv_sum = 1.0 / wg_one;
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        arena[sbase + i] = arena[sbase + i] * inv_sum;
    }
    storageBarrier();

    // 3. Top-K / Top-P: each pick is a workgroup-wide max scan; the pick
    //    order and the running sum stay serial (all lanes hold the same
    //    value, so the loop is uniform).
    let need_filter = (params.top_k > 0u) || (top_p < 1.0 && top_p > 0.0);
    if (need_filter) {
        let k_limit = select(inner, params.top_k, params.top_k > 0u);
        var cum: f32 = 0.0;
        var picked_count: u32 = 0u;
        loop {
            if (picked_count >= k_limit) { break; }
            var best_v: f32 = -1.0;
            var best_i: u32 = 0u;
            for (var i: u32 = lane; i < inner; i = i + LANES) {
                let v = arena[sbase + i];
                if (v >= 0.0 && v > best_v) {
                    best_v = v;
                    best_i = i;
                }
            }
            let best = wg_best(lane, best_v, best_i);
            if (best < 0.0) { break; }
            if (lane == 0u) {
                arena[sbase + wg_best_i[0]] = -best - SENTINEL_EPS;
            }
            cum = cum + best;
            picked_count = picked_count + 1u;
            storageBarrier();
            if (top_p < 1.0 && cum >= top_p) { break; }
        }
        // Restore selected, zero the rest, and sum in index order on lane 0.
        if (lane == 0u) {
            var new_sum: f32 = 0.0;
            for (var i: u32 = 0u; i < inner; i = i + 1u) {
                let v = arena[sbase + i];
                if (v < 0.0) {
                    let restored = -v - SENTINEL_EPS;
                    arena[sbase + i] = restored;
                    new_sum = new_sum + restored;
                } else {
                    arena[sbase + i] = 0.0;
                }
            }
            wg_one = new_sum;
        }
        workgroupBarrier();
        // Lane 0 restored/zeroed the scratch; the scaling loop reads the
        // elements other lanes wrote, so order the storage accesses too.
        storageBarrier();
        let inv_new = 1.0 / max(wg_one, 1e-12);
        for (var i: u32 = lane; i < inner; i = i + LANES) {
            arena[sbase + i] = arena[sbase + i] * inv_new;
        }
        storageBarrier();
    }

    // 4. Gumbel-max: per-index scores are independent; the reduction keeps the
    //    lower index on an exact tie, as the serial `score > best` scan does.
    let raw_key = vec2<u32>(params.seed_lo, params.seed_hi);
    let key = threefry2x32_20(vec2<u32>(0u, 0u), raw_key);
    var best_score: f32 = NEG_INF;
    var best_i: u32 = 0u;
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        let p = arena[sbase + i];
        if (p <= 0.0) { continue; }
        let score = gumbel_score(p, i, 0u, key);
        if (score > best_score) {
            best_score = score;
            best_i = i;
        }
    }
    wg_best(lane, best_score, best_i);
    if (lane == 0u) {
        arena[params.out_off] = f32(wg_best_i[0]);
    }
}

/// The ascending sum of `n` staged values. The one reduction whose **order**
/// decides the drawn bits, so it is one serial chain on lane 0; the unrolled body
/// only hoists the loads (the additions stay in index order).
fn staged_sum(n: u32) -> f32 {
    var s: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i + 8u > n) { break; }
        let a0 = row[i];
        let a1 = row[i + 1u];
        let a2 = row[i + 2u];
        let a3 = row[i + 3u];
        let a4 = row[i + 4u];
        let a5 = row[i + 5u];
        let a6 = row[i + 6u];
        let a7 = row[i + 7u];
        s = s + a0;
        s = s + a1;
        s = s + a2;
        s = s + a3;
        s = s + a4;
        s = s + a5;
        s = s + a6;
        s = s + a7;
        i = i + 8u;
    }
    loop {
        if (i >= n) { break; }
        s = s + row[i];
        i = i + 1u;
    }
    return s;
}

/// `outer == 1` with the row staged: the arithmetic of `sample_one_coop`, in the
/// same order — the same picks, the same two serial sums, the same Gumbel scores —
/// with every pass over `row` instead of the arena. Staging is what removes the
/// global-memory latency a wide row's draw is dominated by: the passes are
/// `LANES`-strided workgroup accesses, and only the two order-deciding sums stay
/// on lane 0. The two paths differ in the address space they read, not in what
/// they compute — and a WGSL `ptr` is bound to one address space, so a helper
/// cannot take the arena and the staged row as one parameter; the passes are
/// therefore spelled twice rather than shared.
fn sample_one_staged(lane: u32) {
    let inner = params.inner;
    let base = params.in_off;
    let temp = bitcast<f32>(params.temp_bits);
    let top_p = bitcast<f32>(params.top_p_bits);
    let inv_temp = 1.0 / max(temp, 1e-6);
    let need_filter = (params.top_k > 0u) || (top_p < 1.0 && top_p > 0.0);
    let k_limit = select(inner, params.top_k, params.top_k > 0u);
    // A picked set that fits `PICK_MAX` is recorded, and the restore sum and the
    // discharge visit it instead of walking the row. `picks` is filled only inside
    // the filter block, so both short paths also test `need_filter`.
    let recording = k_limit <= PICK_MAX;

    // 1. Apply temperature into the staged row; row max.
    var local_max: f32 = NEG_INF;
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        let v = arena[base + i] * inv_temp;
        row[i] = v;
        local_max = max(local_max, v);
    }
    let m: f32 = wg_best(lane, local_max, 0u);

    // 2. exp into the staged row, then the sum in index order on lane 0.
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        row[i] = exp(row[i] - m);
    }
    workgroupBarrier();
    if (lane == 0u) {
        wg_one = staged_sum(inner);
    }
    workgroupBarrier();
    let inv_sum = 1.0 / wg_one;
    for (var i: u32 = lane; i < inner; i = i + LANES) {
        row[i] = row[i] * inv_sum;
    }
    workgroupBarrier();

    // 3. Top-K / Top-P: the picks of `sample_one_coop`, in its order and with its
    //    tie-break (`cand_better` keeps the lower index), four a round.
    var cum: f32 = 0.0;
    var picked_count: u32 = 0u;
    if (need_filter) {
        var done: bool = false;
        loop {
            if (done) { break; }
            // The row's next four values: each lane's own top four, merged
            // pairwise up the workgroup — one barrier a level, and the total order
            // makes the merge exact, so the order is the serial scan's.
            var c = cand_empty();
            for (var i: u32 = lane; i < inner; i = i + LANES) {
                let v = row[i];
                // The tail test is inline: a candidate that cannot enter the list
                // must not cost a `Cand` copy, and most of the row cannot.
                if (v >= 0.0 && cand_better(v, i, c.v[3], c.i[3])) {
                    c = cand_push(c, v, i);
                }
            }
            wg4[lane] = c;
            workgroupBarrier();
            var stride: u32 = LANES / 2u;
            loop {
                if (stride == 0u) { break; }
                if (lane < stride) {
                    wg4[lane] = cand_merge(wg4[lane], wg4[lane + stride]);
                }
                workgroupBarrier();
                stride = stride / 2u;
            }
            // Slot 0 now holds the row's next four, in the order they are picked.
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (picked_count >= k_limit) {
                    done = true;
                    break;
                }
                let v = wg4[0u].v[j];
                let i = wg4[0u].i[j];
                if (v < 0.0) {
                    done = true;
                    break;
                }
                if (lane == 0u) {
                    row[i] = -v - SENTINEL_EPS;
                    if (recording) {
                        // Keep `picks` ascending by index: the restore sum then adds
                        // the selected values in the serial pass's own order.
                        var at: u32 = picked_count;
                        loop {
                            if (at == 0u) { break; }
                            if (picks[at - 1u] < i) { break; }
                            picks[at] = picks[at - 1u];
                            at = at - 1u;
                        }
                        picks[at] = i;
                    }
                }
                cum = cum + v;
                picked_count = picked_count + 1u;
                if (top_p < 1.0 && cum >= top_p) {
                    done = true;
                    break;
                }
            }
            workgroupBarrier();
        }
        // Restore the selected set and zero the rest — summed in index order on
        // lane 0, exactly as the serial pass does, with the writes spread over the
        // workgroup. The selected set is all the sum visits when it was recorded.
        if (lane == 0u) {
            var new_sum: f32 = 0.0;
            if (recording) {
                for (var j: u32 = 0u; j < picked_count; j = j + 1u) {
                    new_sum = new_sum + (-row[picks[j]] - SENTINEL_EPS);
                }
            } else {
                for (var i: u32 = 0u; i < inner; i = i + 1u) {
                    let v = row[i];
                    if (v < 0.0) {
                        new_sum = new_sum + (-v - SENTINEL_EPS);
                    }
                }
            }
            wg_one = new_sum;
        }
        workgroupBarrier();
        for (var i: u32 = lane; i < inner; i = i + LANES) {
            let v = row[i];
            row[i] = select(0.0, -v - SENTINEL_EPS, v < 0.0);
        }
        workgroupBarrier();
        let inv_new = 1.0 / max(wg_one, 1e-12);
        for (var i: u32 = lane; i < inner; i = i + LANES) {
            row[i] = row[i] * inv_new;
        }
        workgroupBarrier();
    }

    // 4. Gumbel-max; the same reduction and tie-break. Every element outside the
    //    selected set is zero, so a recorded set is all a draw has to score.
    let raw_key = vec2<u32>(params.seed_lo, params.seed_hi);
    let key = threefry2x32_20(vec2<u32>(0u, 0u), raw_key);
    var best_score: f32 = NEG_INF;
    var best_i: u32 = 0u;
    if (need_filter && recording) {
        for (var j: u32 = lane; j < picked_count; j = j + LANES) {
            let i = picks[j];
            let p = row[i];
            if (p <= 0.0) { continue; }
            let score = gumbel_score(p, i, 0u, key);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }
    } else {
        for (var i: u32 = lane; i < inner; i = i + LANES) {
            let p = row[i];
            if (p <= 0.0) { continue; }
            let score = gumbel_score(p, i, 0u, key);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }
    }
    wg_best(lane, best_score, best_i);
    if (lane == 0u) {
        arena[params.out_off] = f32(wg_best_i[0]);
    }
}

@compute @workgroup_size(LANES)
fn sample(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    // One row is a decode step's critical path: split it across the workgroup —
    // staged in workgroup storage when it fits, which is what a 2 048-wide code
    // head's row does.
    if (params.outer == 1u) {
        if (params.inner <= STAGE) {
            sample_one_staged(lid.x);
        } else {
            sample_one_coop(lid.x);
        }
        return;
    }
    let row_idx = gid.x + gid.y * ngs.x * LANES;
    if (row_idx >= params.outer) { return; }
    let base = params.in_off + row_idx * params.inner;
    let sbase = params.scratch_off + row_idx * params.inner;
    let temp = bitcast<f32>(params.temp_bits);
    let top_p = bitcast<f32>(params.top_p_bits);
    let inv_temp = 1.0 / max(temp, 1e-6);

    // 1. Apply temperature into scratch; track row max. The logits at
    //    `base` stay read-only.
    var m: f32 = NEG_INF;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[base + i] * inv_temp;
        arena[sbase + i] = v;
        m = max(m, v);
    }
    // 2. exp / sum → probabilities (scratch slot).
    var sum_e: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let e = exp(arena[sbase + i] - m);
        arena[sbase + i] = e;
        sum_e = sum_e + e;
    }
    let inv_sum = 1.0 / sum_e;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        arena[sbase + i] = arena[sbase + i] * inv_sum;
    }

    // 3. Top-K / Top-P filter via selection.
    let need_filter = (params.top_k > 0u) || (top_p < 1.0 && top_p > 0.0);
    if (need_filter) {
        var cum: f32 = 0.0;
        var picked_count: u32 = 0u;
        let k_limit = select(params.inner, params.top_k, params.top_k > 0u);
        loop {
            if (picked_count >= k_limit) { break; }
            // Find the largest still-positive probability.
            var best_v: f32 = -1.0;
            var best_i: u32 = 0u;
            var found: bool = false;
            for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
                let v = arena[sbase + i];
                if (v >= 0.0 && v > best_v) {
                    best_v = v;
                    best_i = i;
                    found = true;
                }
            }
            if (!found) { break; }
            // Mark as selected by negating + sentinel offset.
            arena[sbase + best_i] = -best_v - SENTINEL_EPS;
            cum = cum + best_v;
            picked_count = picked_count + 1u;
            // Stop early if top-p satisfied AND we picked at least 1.
            if (top_p < 1.0 && cum >= top_p) { break; }
        }
        // Restore selected; zero unselected; sum for renormalization.
        var new_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let v = arena[sbase + i];
            if (v < 0.0) {
                let restored = -v - SENTINEL_EPS;
                arena[sbase + i] = restored;
                new_sum = new_sum + restored;
            } else {
                arena[sbase + i] = 0.0;
            }
        }
        let inv_new = 1.0 / max(new_sum, 1e-12);
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            arena[sbase + i] = arena[sbase + i] * inv_new;
        }
    }

    // 4. Multinomial sample via Gumbel-max: argmax(log(p) + g) where
    //    g_i = -log(-log(u_i)) is Gumbel(0, 1) noise. Equivalent to
    //    softmax-then-categorical but a single argmax pass — and
    //    matches MLX's `categorical` algorithm so picks are bit-
    //    equivalent given the same uniform stream + seed convention.
    //
    //    JAX-style seed → key derivation: rather than feed the raw
    //    u64 seed into per-cell Threefry, we first split the seed
    //    once (Threefry on counter=(0,0) with the raw seed as key) and
    //    use *that* output as the working key. Matches the protocol
    //    JAX/MLX use to turn a `PRNGKey(seed)` into the key the
    //    categorical sampler actually consumes — a no-op statistically
    //    but the bit pattern of u_i shifts to match.
    let raw_key = vec2<u32>(params.seed_lo, params.seed_hi);
    let key = threefry2x32_20(vec2<u32>(0u, 0u), raw_key);
    var best_score: f32 = NEG_INF;
    var picked: u32 = 0u;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let p = arena[sbase + i];
        if (p <= 0.0) { continue; }   // filtered by top_k / top_p
        let score = gumbel_score(p, i, row_idx, key);
        if (score > best_score) {
            best_score = score;
            picked = i;
        }
    }
    arena[params.out_off + row_idx] = f32(picked);
}
