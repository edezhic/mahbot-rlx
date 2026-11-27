// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decode attention entry's two bodies on the product's own call: a recognition
//! row on the graphics path, timed alternately over rounds, with the transcripts held
//! against each other.
//!
//! `RLX_WGPU_ATTN_DECODE_ONLINE=1` is the reference arm — the one-pass body this fork
//! shipped before the weighted one, which
//! `crates/backends/rlx-wgpu/tests/attention_decode_step_cost.rs` holds to the weighted
//! body's numbers at the product's key extents — so every reading here is a
//! same-session pair rather than two runs compared across a rebuild. The body is a
//! dispatch-time pick, so one compiled graph runs either arm and the product's own
//! per-call graph cache is not a confound.
//!
//! The row effect sits at the stand's own take-to-take spread, which is why the arms
//! are paired rather than read one after the other, and why the statistic is the median
//! of the per-round differences: the machine's load moves either arm's absolute figure
//! by more than the effect.
//!
//! Needs the installed model tree and the pinned corpus. `--test-threads=1` is part of
//! the invocation, not cosmetic: the row's device is process-global and read once at the
//! product's first compile, and `support::solo` refuses a second subject rather than
//! serialising it.
//!
//! ```text
//! cargo test --release -p mahbot-inference --features hand-run --test attention_decode_form \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use mahbot_inference::report;

mod support;

/// The row: 30 s of the pinned recording — the stand's `recognition.typical` shape, and
/// the row `tests/attention_arm.rs` prices for the same reason: long enough that the
/// attention chain's share of the call is not lost in the model open.
const ROW_SECONDS: usize = 30;

/// Rounds of the paired take. The call is seconds long, so nine rounds a side is already
/// minutes, and an odd count puts the median on a real round rather than between two.
const ROUNDS: usize = 9;

/// The switch, and the two arms: `0` is the shipped weighted body, `1` the one-pass one.
const FORM: &str = "RLX_WGPU_ATTN_DECODE_ONLINE";

#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_recognition_row_on_the_graphics_path() {
    let _alone = support::solo("the decode body's row take");
    let Some(mut row) = support::Row::open(Some(ROW_SECONDS)) else {
        return;
    };
    // The row is the graphics path's: a processor arm here would price a kernel this
    // change does not touch.
    assert_eq!(
        mahbot_inference::device::settled(),
        "gpu",
        "the row settled on another device"
    );
    let call = |row: &mut support::Row| -> (f64, f64) {
        let (t, cpu) = (Instant::now(), support::cpu_seconds());
        row.call();
        let cpu = support::cpu_seconds() - cpu;
        (t.elapsed().as_secs_f64() * 1e3, cpu * 1e3)
    };
    // One warm-up each, so the model open and the graph build fall outside the take.
    let warm_weighted = support::with_env(FORM, "0", || call(&mut row));
    let warm_online = support::with_env(FORM, "1", || call(&mut row));
    println!(
        "warm-up: weighted {:.0} ms wall / {:.0} ms cpu, one-pass {:.0} ms wall / {:.0} ms cpu",
        warm_weighted.0, warm_weighted.1, warm_online.0, warm_online.1
    );
    let baseline = report::peak_rss();
    let [weighted, online] = support::paired_env(ROUNDS, FORM, ["0", "1"], || call(&mut row));
    let clock = |pick: fn(&(f64, f64)) -> f64| {
        [weighted.as_slice(), online.as_slice()].map(|s| s.iter().map(pick).collect::<Vec<_>>())
    };
    for (name, samples) in [("wall", clock(|s| s.0)), ("cpu", clock(|s| s.1))] {
        let (weighted_median, online_median) =
            (support::median(&samples[0]), support::median(&samples[1]));
        let paired = support::median(
            &samples[0]
                .iter()
                .zip(&samples[1])
                .map(|(w, o)| w - o)
                .collect::<Vec<_>>(),
        );
        println!(
            "{name}: weighted {weighted_median:.0} ms (of {:?}), one-pass {online_median:.0} ms \
             (of {:?}) — paired {paired:+.0} ms ({:.1} % off the one-pass row)",
            samples[0],
            samples[1],
            100.0 * paired / online_median
        );
    }
    // Recorded, not gated: the row's own per-call growth already moves the high-water
    // mark by hundreds of MiB between runs, so a bound here would price the allocator.
    // Neither body retains anything across calls — the weighted body's key tile is
    // workgroup storage and its accumulator is registers.
    println!(
        "process peak: {} after the warm-up, {} after the take",
        report::shown(baseline),
        report::shown(report::peak_rss())
    );
}
