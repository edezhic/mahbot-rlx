// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-shape cost of attention in the two operand layouts, and of both arms of the
//! head-major route, at the geometries the MahBot product emits: the decoder's
//! prefill ladder (16 heads, 128-wide, causal, 8 KV heads), the ASR encoder tail
//! (16 heads, 64-wide, no mask), the tagger's ViT blocks (six heads, 64-wide) and the
//! wake window.
//!
//! `#[ignore]`d — it prints a table and asserts only the one thing that must not
//! regress. Gate it on `just throttle` first, then:
//!
//! ```text
//! cargo test -p mahbot-rlx-cpu --release --features backend-tests \
//!     --test attention_head_major_route_perf -- --ignored --nocapture
//! ```
//!
//! Run it once more with `RLX_CPU_PORTABLE=1` (the latch is one-shot, so that is a
//! second process): portable only swaps the vector kernels for their scalar
//! mirrors, and this route is a plain scalar loop either way, so the same
//! alternating reading applies.
//!
//! Every row is timed in both layouts and in both arms of
//! `RLX_CPU_ATTN_HEAD_MAJOR_SERIAL` — the same per-row loop on one thread and on the
//! pool — alternating the arm order every round (`common::paired`), so a drifting
//! machine moves all three together. The one-thread arm is not the pre-change route
//! byte for byte: that loop softmaxed a head's rows in one call where this calls per
//! row. For these shapes that costs the call alone — `par_softmax_rows` fans out only
//! above 327 680 elements, and the largest score matrix here is 16 128 — so the
//! printed gain is the route's, not a softmax artefact.
//! `RLX_SDPA_THRESHOLD` is read once per process: the default (32) leaves the short
//! rows-first rows on the sequential NEON route and the long ones on the
//! strided-BLAS route.
//!
//! The two layouts are separate graphs, so this measures a layout's cost, not one
//! graph's share of a row: that share is the per-call figure times the call count.
//!
//! `decode 32 keyed` is the run's noise floor, not a gain: `grain_for` gives it one
//! chunk, so both arms execute the identical call and its printed ratio is whatever
//! the machine did between them. The µs/r² and GFLOP/s columns are the pooled arm's.
mod common;

use common::attn::{CASES, Run, dense, graph, serial};

/// A sample lasts long enough that the clock and the pool's wake-up are noise, and
/// short enough that the whole table stays a few seconds.
const TARGET_MS: f64 = 2.0;
const ROUNDS: usize = 9;

/// Nothing may be traded for the big shapes: no rung or window may end up slower
/// than the serial loop it replaced. The bound is deliberately loose — a machine
/// carrying load must not flake the harness, and a real regression here is a
/// multiple, not a few percent — so the printed table, not this assert, is the
/// evidence.
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
        "{:<18} {:>4} {:>10} {:>10} {:>10} {:>7} {:>9} {:>9}",
        "case", "s", "rows-first", "head-pool", "head-serial", "gain", "µs/r²", "GFLOP/s"
    );
    for c in CASES {
        let mut rows = Run::new(&graph(c, false), dense);
        let mut head = Run::new(&graph(c, true), dense);
        serial(false);
        let one = head.sample(1);
        let reps = (TARGET_MS / one).round().clamp(1.0, 500.0) as usize;
        // Three arms per round, in the fork's alternating order: rows-first, the
        // pooled head-major loop, the serial one.
        let med = common::paired(ROUNDS, 3, |arm| {
            serial(arm == 2);
            match arm {
                0 => rows.sample(reps),
                _ => head.sample(reps),
            }
        });
        let (rf, pooled, one_thread) = (med[0], med[1], med[2]);
        let (b, s, ks, h, d) = (c.batch, c.seq, c.kv_seq, c.heads, c.head_dim);
        let flops = 4.0 * (b * h * s * ks * d) as f64;
        println!(
            "{:<18} {s:>4} {rf:>10.4} {pooled:>10.4} {one_thread:>10.4} {:>6.1}x {:>9.3} {:>9.0}",
            c.name,
            one_thread / pooled,
            pooled * 1e3 / (s * s) as f64,
            flops / (pooled * 1e-3) / 1e9,
        );
        assert!(
            pooled <= one_thread * NO_SLOWER,
            "{}: the pooled head-major loop is {pooled:.4} ms against the serial \
             loop's {one_thread:.4} ms",
            c.name
        );
    }
}
