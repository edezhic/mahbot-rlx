// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Measurement only, `#[ignore]`d — prints, never asserts a time.**
//!
//! ```text
//! cargo test -p rlx-wgpu --release --features backend-tests \
//!     --test gguf_gemm_grouped_loop_rate -- --ignored --nocapture
//! ```
//!
//! The row-grouped packed multiply's own loop, numbered for the one question this
//! measurement exists to answer: how much of a many-row call is its loop and how
//! much is the packed weight's traffic. The number is the **demand** — the bytes
//! the call's row
//! groups move over the weight (`groups x weight`), over the call's time, i.e. the
//! rate the weight would have to stream at for its own traffic to account for the
//! call — read against this machine's streaming read ceiling
//! ([`rlx_wgpu::bandwidth::stream_ceiling`], the same probe the stand's weight-stream
//! rows are quoted against). A demand well under that ceiling is a call the weight's
//! bytes cannot explain. Rates are in the unit the ceiling harness already uses
//! (`GFLOP/s`, `tests/gemm_ceiling.rs`).
//!
//! The resident sweep is the corroborating reading: the same multiply at the same
//! row count, over a weight small enough that the whole packed tensor stays inside
//! [`RESIDENT_BYTES`], across output widths. Its rate rising with the width while the
//! weight's bytes rise with it is the shape of a call that is not paying for those
//! bytes — but a fixed per-call cost being amortised has the same shape, and the
//! sweep stops short of the phase's node while its rate is still rising, so part of
//! that gap is work expansion rather than traffic. The sweep therefore says only
//! whether a weight that is smaller per pass buys a higher rate; the verdict is the
//! demand figure's.
//!
//! The two phases, and their row counts derived rather than guessed, both from
//! `mahbot_inference::graphs::asr_tokens` and `stand::recognise`:
//!
//! * the recognition **prefill** is one graph over the pinned clip's 234-row
//!   prompt — `3 + 6 + asr_tokens(1 746 mel frames) + 6`, and `asr_tokens(1 746)`
//!   is 219 — on the body's Q4_K weights, whose hidden width is the `k = 2 048` of
//!   its widest projections into `n = 6 144`;
//! * the **encoder** is one graph over the same clip's own **219** audio tokens, on
//!   the `mmproj`'s Q8_0 weights, at its `k = 1 024`/`n = 4 096` projection.
//!
//! "Resident" is a bound on bytes and not an observation, because nothing in the
//! tree reads this adapter's cache: a point is resident when its whole packed tensor
//! is at most [`RESIDENT_BYTES`]. The call's fixed term — one dispatch, the upload of
//! a few bytes, the readback of a 16-element output — is measured separately
//! ([`floor_us`]) and removed from every point before its rate is read; a point whose
//! own time does not leave a second floor behind is printed with what it measured and
//! not read as a rate. The floor does not remove what grows with the shape: the
//! activation upload and the output readback stay in, about 1 % of a node's time at
//! this machine's read rate, so a net rate is the conservative reading.
//!
//! Recorded 2026-10-25 on the stand (Apple M3 Pro, 14-core GPU, 18 GiB, macOS, wgpu
//! over Metal), `--release`, best of seven after three warm-ups, the two sides
//! interleaved in one window — the four-lane accumulators this kernel carried
//! and the one scalar per row it carries now:
//!
//! | phase | its node | four lanes (A) | one scalar per row (B) |
//! |---|---|---|---|
//! | prefill | Q4_K 234 x 2 048 x 6 144 | 5 076-5 114 us (1 152-1 160 GFLOP/s) | 4 673-4 729 us (1 245-1 260) |
//! | encoder | Q8_0 219 x 1 024 x 4 096 | 2 007-2 070 us (887-916) | 1 937-2 033 us (904-949) |
//!
//! The same two binaries over the raw kernel at 219 rows — the encoder's own row
//! count, on the body's K-quant weights rather than a phase's whole node, in a
//! second window whose microseconds are therefore not comparable with the table
//! above's: Q4_K 219 x 2 048 x 6 144 at 4 440-4 464 against 4 775-4 808 us (-7 %),
//! Q6_K 219 x 2 048 x 151 936 at 136.75-136.91 against 143.57-143.69 ms (-4.8 %),
//! and the 15 small-`m` cells of `gguf_gemm_grouped_split_rate.rs`, each side's own
//! binary run over them in that window, between 1.9 % slower and 7 % faster (median
//! 1.5 % faster — the one slower cell carried as measured, not re-taken here).
//!
//! The A side is the kernel before this form and cannot be re-taken from this tree —
//! that arm is gone, and the two binaries of those windows are what paired them.
//!
//! The measured reading, three runs of this harness in one window (a 228-277 us floor,
//! a 132.2-135.0 GB/s read ceiling on the same device): the prefill's node moves
//! 30 x 6.75 MiB in 4 722-4 738 us — 44.8-45.0 GB/s implied, 47.1-47.8 net of the
//! floor, 2.8-2.9x under the ceiling — so at most 35-36 % of its time can be the
//! weight's bytes, and halving the passes over them could recover at most 18 %. The
//! encoder's moves 28 x 4.25 MiB in 1 947-1 978 us — 63.1-64.1 implied, 71.3-74.3
//! net, 1.8-1.9x under — so at most 53-56 % of its time, a narrower margin stated
//! rather than rounded away. What a taller tile costs is far above either bound:
//! every measured height above 8 rows, on the same narrow accumulators, read
//! 1.5x-3.6x the 8-row time on the raw kernel at 219 rows (Q4_K 1.9x-3.6x, Q6_K
//! 1.5x-2.3x; `dequant_gemm_gguf.wgsl`'s header carries that sweep and the
//! declared-storage probe beside it). The resident sweep says the same from the
//! rate side: its best
//! point at the prefill's row count (3.94 MiB, 1.7x
//! lighter per pass than the node) reads 1 059-1 258 GFLOP/s net against the node's
//! 1 306-1 319 — 0.81-0.96x — and the encoder's own range barely reaches a lighter
//! weight at all (3.72 MiB is 1.1x lighter than its 4.25 MiB node), where it reads
//! 1 034-1 069 against 1 050-1 094 (0.97-0.99x). A lighter weight per pass buys no
//! rate here, so a taller tile's saved passes were not the saving the phases were
//! waiting for.
//!
//! Run the thermal gate first (`just throttle`).
mod common;

use common::data;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::gguf_gpu::{gguf_row_bytes, grouped_base_geometry};
use rlx_wgpu::gguf_host::gguf_scheme_id;

/// Best-of-7 µs of one packed multiply call, its input an input node so the graph
/// is not constant-folded, and with the output read back as the crate's own
/// timing convention does — so the reading carries the call's fixed term, which
/// [`floor_us`] measures separately rather than guesses.
fn call_us(scheme: QuantScheme, m: usize, k: usize, n: usize) -> f64 {
    let packed = common::packed_gguf_weight(scheme, k, n);
    let mut g = Graph::new("grouped_loop_rate");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Gpu).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let xv = data(m * k, 0.002, false);
    // The demand figure divides by this call's time, so the call has to be the one
    // the shape names: the width it was compiled for, and finite values — the check
    // the one-row harness's cell makes before it times one.
    let warm = c.run(&[("x", &xv)]);
    assert_eq!(warm.len(), 1, "one output");
    assert_eq!(warm[0].len(), m * n, "output width at m={m} k={k} n={n}");
    let non_finite = warm[0].iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "{non_finite} non-finite values at m={m} k={k} n={n}"
    );
    let mut one = || {
        let out = c.run(&[("x", &xv)]).remove(0);
        assert_eq!(out.len(), m * n, "output width at m={m} k={k} n={n}");
    };
    for _ in 0..2 {
        one();
    }
    let best = common::time_rounds(7, 1, &mut one)
        .into_iter()
        .fold(f64::MAX, f64::min);
    best * 1e3
}

/// The call's fixed term — one dispatch, the upload of a few bytes, the readback of
/// a 16-element output — at a shape whose arithmetic is nothing. Every point's own
/// time has this removed before a rate is read from it; the per-call part that grows
/// with the shape (the activation upload and the output readback) is not fixed and
/// stays in, at about 1 % of a node's time.
fn floor_us() -> f64 {
    call_us(QuantScheme::GgufQ8_0, 2, 32, 8)
}

/// The sweep's resident bound: a packed tensor at or under this many bytes is taken
/// as cache-served. A bound on bytes, not a reading of the cache.
const RESIDENT_BYTES: u64 = 4 << 20;

/// `us` with the call's fixed term removed — `None` when the point's own time does
/// not leave a second floor behind, i.e. when it measures the path's floor rather
/// than the loop.
fn net_us(us: f64, floor: f64) -> Option<f64> {
    (us >= 2.0 * floor).then_some(us - floor)
}

/// Sweep the output width at one phase's `(scheme, m, k)`, then print the phase's own
/// node with its demand — the bytes its row groups move over the weight, over the
/// call's time — against `ceiling_gbs`, this machine's streaming read ceiling (`0.0`
/// when the probe produced no figure).
fn sweep(
    phase: &str,
    scheme: QuantScheme,
    m: usize,
    k: usize,
    widths: &[usize],
    node: usize,
    floor: f64,
    ceiling_gbs: f64,
) {
    let row_bytes = gguf_row_bytes(gguf_scheme_id(scheme), k);
    let groups = grouped_base_geometry(m, node).1;
    eprintln!(
        "[grouped loop rate] {phase}: m={m} k={k} {scheme:?}, {groups} row groups; resident \
         sweep, best of 7 after 3 warm-ups, the call's own {floor:.1} us floor removed before a \
         rate is read"
    );
    let mut best: Option<(f64, usize, u64)> = None;
    let mut any_resident = false;
    for &n in widths {
        let bytes = n as u64 * row_bytes;
        let us = call_us(scheme, m, k, n);
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        let resident = bytes <= RESIDENT_BYTES;
        any_resident |= resident;
        let shape = format!(
            "{phase}: n={n} weight={:.2} MiB resident={}",
            bytes as f64 / (1024.0 * 1024.0),
            if resident { "yes" } else { "no" }
        );
        match net_us(us, floor) {
            Some(net) => {
                let net_rate = gflop * 1e6 / net;
                eprintln!(
                    "[grouped loop rate] {shape} {us:8.1} us net {net:8.1} {gflop:6.3} GFLOP \
                     {net_rate:7.1} GFLOP/s"
                );
                if resident && best.is_none_or(|b| net_rate > b.0) {
                    best = Some((net_rate, n, bytes));
                }
            }
            None => eprintln!(
                "[grouped loop rate] {shape} {us:8.1} us — at the call's own floor: not read as \
                 a rate"
            ),
        }
    }
    // The phase's own node — the call the stand runs — and the demand it implies.
    let node_bytes = node as u64 * row_bytes;
    let node_us = call_us(scheme, m, k, node);
    let node_gflop = 2.0 * (m * k * node) as f64 / 1e9;
    let node_rate = node_gflop * 1e6 / node_us;
    let moved = groups as f64 * node_bytes as f64;
    let demand = moved / node_us / 1e3;
    eprintln!(
        "[grouped loop rate] {phase}: the node m={m} k={k} n={node} moves {:.2} MiB ({groups} x \
         {:.2} MiB) in {node_us:.1} us ({node_rate:.1} GFLOP/s) — {demand:.1} GB/s implied",
        moved / (1024.0 * 1024.0),
        node_bytes as f64 / (1024.0 * 1024.0)
    );
    let verdict = match (net_us(node_us, floor), ceiling_gbs > 0.0) {
        (Some(net), true) => {
            let net_demand = moved / net / 1e3;
            let share = net_demand / ceiling_gbs;
            if share >= 0.8 {
                format!(
                    "{net_demand:.1} GB/s net of the floor against this machine's \
                     {ceiling_gbs:.1} GB/s read ceiling ({share:.2}x it): the weight's bytes can \
                     account for the call, so its traffic is a term this call pays for"
                )
            } else {
                format!(
                    "{net_demand:.1} GB/s net of the floor against this machine's \
                     {ceiling_gbs:.1} GB/s read ceiling ({:.1}x under): at most {:.0} % of the \
                     call's time can be the weight's bytes, so halving the passes over it could \
                     recover at most {:.0} % — the loop is the rest",
                    1.0 / share,
                    share * 100.0,
                    share * 50.0
                )
            }
        }
        (Some(net), false) => format!(
            "{:.1} GB/s net of the floor, with no ceiling figure from the probe to read it \
             against",
            moved / net / 1e3
        ),
        (None, _) => "the node's own time does not leave the call's floor behind, so no rate is \
                      read from it"
            .to_string(),
    };
    eprintln!("[grouped loop rate] {phase}: {verdict}");
    match (best, net_us(node_us, floor)) {
        (Some((rate, n, bytes)), Some(node_net)) => {
            let node_net_rate = node_gflop * 1e6 / node_net;
            eprintln!(
                "[grouped loop rate] {phase}: best resident point {rate:.1} GFLOP/s net at n={n} \
                 ({:.2} MiB, {:.1}x lighter per pass than the node); the node reads \
                 {node_net_rate:.1} GFLOP/s net — a lighter weight per pass buys {:.2}x its rate, \
                 and the sweep's rate still rising with the width is work expansion as much as \
                 traffic",
                bytes as f64 / (1024.0 * 1024.0),
                node_bytes as f64 / bytes as f64,
                rate / node_net_rate
            );
        }
        (Some(_), None) => eprintln!(
            "[grouped loop rate] {phase}: the node's own time does not leave the call's floor \
             behind, so there is no net rate to set the sweep's points against"
        ),
        (None, _) if !any_resident => eprintln!(
            "[grouped loop rate] {phase}: no swept width fit the {:.1} MiB resident bound, so the \
             sweep has no point to read; the demand figure above is the phase's reading",
            RESIDENT_BYTES as f64 / (1024.0 * 1024.0)
        ),
        (None, _) => eprintln!(
            "[grouped loop rate] {phase}: no resident point's own time left the call's floor \
             behind, so the sweep has no rate to read; the demand figure above is the phase's \
             reading"
        ),
    }
}

#[test]
#[ignore = "measurement: prints rates, asserts no duration"]
fn the_loop_with_the_weight_resident() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    common::refuse_under_run_report();
    let floor = floor_us();
    let ceiling_gbs = match rlx_wgpu::bandwidth::stream_ceiling() {
        Ok(c) => c.read_gb_s,
        Err(e) => {
            eprintln!(
                "[grouped loop rate] the read ceiling probe produced no figure ({e}); every demand \
                 below is printed without one"
            );
            0.0
        }
    };
    // The prefill's own row count and its widest projection, on the body's Q4_K
    // weights; the widths below stay inside RESIDENT_BYTES (1 152 B a row at
    // k = 2 048).
    sweep(
        "prefill",
        QuantScheme::GgufQ4K,
        234,
        2048,
        &[256, 512, 1024, 2048, 3584],
        6144,
        floor,
        ceiling_gbs,
    );
    // The encoder's own row count, on the mmproj's Q8_0 weights (1 088 B a row at
    // k = 1 024).
    sweep(
        "encoder",
        QuantScheme::GgufQ8_0,
        219,
        1024,
        &[256, 512, 1024, 2048, 3584],
        4096,
        floor,
        ceiling_gbs,
    );
}
