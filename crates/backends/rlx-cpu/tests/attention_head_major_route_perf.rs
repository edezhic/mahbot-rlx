// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-shape cost of attention in the two operand layouts, at the geometries the MahBot
//! product emits: the decoder's prefill ladder (16 heads, 128-wide, causal, 8 KV heads),
//! the ASR encoder tail (16 heads, 64-wide, no mask), the tagger's ViT blocks (six heads,
//! 64-wide) and the wake window.
//!
//! `#[ignore]`d — it prints a table and asserts only the non-regression bounds. Gate it
//! on `just throttle` first, then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --features backend-tests \
//!     --test attention_head_major_route_perf -- --ignored --nocapture
//! ```
//!
//! Run it once more with `RLX_CPU_PORTABLE=1` (the latch is one-shot, so that is a
//! second process): portable swaps the softmax for its scalar mirror, and swaps the
//! rows-first dot for the scalar loop, so the same alternating reading applies.
//!
//! Every row is timed in both layouts and in three arms of the head-major route —
//! the pooled lane block, the pooled scalar loop it replaced, and the lane block on
//! one thread — alternating the arm order every round (`common::paired`), so a
//! drifting machine moves all four together. The `scalar/lane` column is the lane block
//! against the scalar loop it replaced: the same call, same bits, one arm with the inner
//! loops vectorised and one without. The one-thread arm is not the pre-change route byte
//! for byte: that loop softmaxed a head's rows in one call where this calls per row. For
//! these shapes that costs the call alone — `par_softmax_rows` fans out only above
//! 327 680 elements, and the largest score matrix here is 16 128 — so the printed
//! gain is the route's, not a softmax artefact.
//!
//! `RLX_SDPA_THRESHOLD` is read once per process: the default (32) leaves the short
//! rows-first rows on the sequential NEON route and the long ones on the
//! strided-BLAS route.
//!
//! The two layouts are separate graphs, so this measures a layout's cost, not one
//! graph's share of a row: that share is the per-call figure times the call count.
//! The µs/r² and GFLOP/s columns are the lane arm's.
//!
//! `decode 32 keyed` is the run's noise floor, not a gain: `grain_for` gives it one
//! chunk, so the two head-major inner-loop arms execute the identical call and their
//! printed ratio is whatever the machine did between them. It is also the shape the
//! lane block leaves alone — one query row cannot amortise the K transpose — so its
//! `scalar/lane` is 1.0x by design, not by accident.
mod common;

use common::attn::{CASES, Run, dense, graph, scalar_body, serial};

/// A sample lasts long enough that the clock and the pool's wake-up are noise, and
/// short enough that the whole table stays a few seconds.
const TARGET_MS: f64 = 2.0;
const ROUNDS: usize = 9;

/// Nothing may be traded for the big shapes: neither the lane block nor the pool split
/// may end up slower than what it replaced. The bound is deliberately loose — a real
/// regression here is a multiple, not a few percent — so the printed table, not these
/// asserts, is the evidence. It is not a load test: on a heavily oversubscribed machine
/// (a load several times the pool width) the lane arm can read slower than the scalar
/// loop by more than the bound and the assert fires, which says the machine is busy, not
/// that the kernel regressed.
const NO_SLOWER: f64 = 1.5;

#[test]
#[ignore = "prints a table; run by hand after `just throttle`"]
fn per_shape_cost_of_both_layouts() {
    // What a split can reach is bounded by the pool's width (the tuned default, or
    // `RLX_WORKERS`) and by the work per task. One worker runs the pooled arm inline,
    // where the comparison this table makes is vacuous.
    println!("pool width {}", rlx_cpu::pool::num_threads());
    assert!(
        rlx_cpu::pool::num_threads() > 1,
        "a one-worker pool runs both head-major arms inline"
    );
    println!(
        "{:<18} {:>4} {:>10} {:>10} {:>11} {:>8} {:>10} {:>7} {:>9} {:>9}",
        "case",
        "s",
        "rows-first",
        "head-lane",
        "head-scalar",
        "scalar/lane",
        "head-1thr",
        "gain",
        "µs/r²",
        "GFLOP/s"
    );
    for c in CASES {
        let mut rows = Run::new(&graph(c, false), dense);
        let mut head = Run::new(&graph(c, true), dense);
        serial(false);
        scalar_body(false);
        let one = head.sample(1);
        let reps = (TARGET_MS / one).round().clamp(1.0, 500.0) as usize;
        // Four arms per round, in the fork's alternating order: rows-first, the
        // pooled lane block, the pooled scalar loop, the lane block on one thread.
        let med = common::paired(ROUNDS, 4, |arm| {
            serial(arm == 3);
            scalar_body(arm == 2);
            match arm {
                0 => rows.sample(reps),
                _ => head.sample(reps),
            }
        });
        let (rf, lane, scalar, one_thread) = (med[0], med[1], med[2], med[3]);
        let (b, s, ks, h, d) = (c.batch, c.seq, c.kv_seq, c.heads, c.head_dim);
        let flops = 4.0 * (b * h * s * ks * d) as f64;
        println!(
            "{:<18} {s:>4} {rf:>10.4} {lane:>10.4} {scalar:>11.4} {:>7.1}x {one_thread:>10.4} \
             {:>6.1}x {:>9.3} {:>9.0}",
            c.name,
            scalar / lane,
            one_thread / lane,
            lane * 1e3 / (s * s) as f64,
            flops / (lane * 1e-3) / 1e9,
        );
        assert!(
            lane <= scalar * NO_SLOWER,
            "{}: the lane block is {lane:.4} ms against the scalar loop's {scalar:.4} ms",
            c.name
        );
        assert!(
            lane <= one_thread * NO_SLOWER,
            "{}: the pool is {lane:.4} ms against one thread's {one_thread:.4} ms",
            c.name
        );
    }
}
