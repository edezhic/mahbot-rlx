// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The head-major attention route's two inner-loop arms on the product's own call: a
//! recognition row on the processor, timed alternately over rounds, with the transcripts
//! held against each other.
//!
//! `RLX_CPU_ATTN_HEAD_MAJOR_SCALAR=1` is the reference arm — the route's pre-lane scalar
//! loops, which `crates/backends/rlx-cpu/tests/attention_head_major_route.rs` holds to the
//! lane block's bits at the product's geometries — so every reading here is a same-session
//! pair rather than two runs compared across a rebuild. The route is a phase inside
//! `transcribe`, which prints no attention field, so the call is the instrument: a kernel
//! win that does not reach this row is not a win.
//!
//! Needs the installed model tree and the pinned corpus. `--test-threads=1` is part of
//! the invocation, not cosmetic: the row take forces `MAHBOT_DEVICE=cpu`, which is
//! process-global and read once at the product's first compile, and `support::solo`
//! refuses a second subject rather than serialising it.
//!
//! ```text
//! cargo test --release -p mahbot-inference --features hand-run --test attention_arm \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use mahbot_inference::report;

mod support;

/// The row: 30 s of the pinned recording — the stand's `recognition.typical` shape, long
/// enough that the route's share of the call is not lost in the model open.
const ROW_SECONDS: usize = 30;

/// Rounds of the paired take. The call is seconds long, so five rounds a side is already
/// tens of seconds, and an odd count puts the median on a real round rather than between
/// two.
const ROUNDS: usize = 5;

#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_recognition_row_on_the_processor() {
    let _alone = support::force_processor("the attention route's row take");
    let Some(mut row) = support::Row::open(Some(ROW_SECONDS)) else {
        return;
    };
    let call = |row: &mut support::Row| -> (f64, f64) {
        let (t, cpu) = (Instant::now(), support::cpu_seconds());
        row.call();
        let cpu = support::cpu_seconds() - cpu;
        (t.elapsed().as_secs_f64() * 1e3, cpu * 1e3)
    };
    // One warm-up each, so the model open and the graph build fall outside the take.
    let warm_lane = support::with_env("RLX_CPU_ATTN_HEAD_MAJOR_SCALAR", "0", || call(&mut row));
    support::assert_settled_on_the_processor();
    let warm_scalar = support::with_env("RLX_CPU_ATTN_HEAD_MAJOR_SCALAR", "1", || call(&mut row));
    println!(
        "warm-up: lane {:.0} ms wall / {:.0} ms cpu, scalar {:.0} ms wall / {:.0} ms cpu",
        warm_lane.0, warm_lane.1, warm_scalar.0, warm_scalar.1
    );
    let baseline = report::peak_rss();
    let [lane, scalar] =
        support::paired_env(ROUNDS, "RLX_CPU_ATTN_HEAD_MAJOR_SCALAR", ["0", "1"], || {
            call(&mut row)
        });
    // The wall clock is the row's own figure; the CPU clock is the paired one — a loaded
    // machine moves the first by multiples and the second by little, so a share of the
    // call this small is only visible there.
    let clock = |pick: fn(&(f64, f64)) -> f64| {
        [lane.as_slice(), scalar.as_slice()].map(|s| s.iter().map(pick).collect::<Vec<_>>())
    };
    for (name, samples) in [("wall", clock(|s| s.0)), ("cpu", clock(|s| s.1))] {
        let (l, s) = (support::median(&samples[0]), support::median(&samples[1]));
        println!(
            "{name}: lane {:?} ms, scalar {:?} ms — {s:.0} ms → {l:.0} ms, {:.1} % off the call",
            samples[0],
            samples[1],
            100.0 * (s - l) / s
        );
    }
    // Recorded, not gated. The route retains nothing across calls — the transposed K is a
    // local `Vec` dropped with the call — and this product's own per-call growth already
    // moves the high-water mark by hundreds of MiB between runs, so a bound here would
    // price the allocator rather than the route. The stand's peak column is where a rise
    // the route caused would be attributed.
    println!(
        "process peak: {} after the warm-up, {} after the take",
        report::shown(baseline),
        report::shown(report::peak_rss())
    );
}
