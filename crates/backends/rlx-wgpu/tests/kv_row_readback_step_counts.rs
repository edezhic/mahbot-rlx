// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The rows a step consumes leave the device at the step's price, not at one
//! device round trip per row.**
//!
//! A resident cache makes reading a row cost the row rather than the cache, but
//! a decode step reads one row per layer per K/V — fifty-six of them on the
//! recognition body — and each of those was its own submission and its own wait,
//! which puts a floor of about 5.6 ms under the step however small the rows are.
//! The bar is the region the step's own rows reach when they leave through a
//! graph output — about +1.2 ms at the short key and +3.6 ms at the long one over
//! a step that reads nothing back — and the shape that reaches it is the one that
//! carries the whole step: every row's copy is encoded together and answered by
//! one map, so what the read costs follows the rows asked for and not how many
//! handles they are spread across.
//!
//! What this file pins is that shape, which is what a timing pin cannot pin on an
//! adapter it does not know: the crate's process-wide submission/wait ledger, read
//! around one call. A step's fifty-six rows cost one of each — at either cache
//! capacity, which is the capacity half of the bar: the same rows read at a cache
//! eight times as long cost the same, not proportionally more. The same rows asked
//! one row at a time cost one of each *per row*, which is the price the batch
//! exists to remove, and a batch whose rows the host already holds asks nothing of
//! the device at all.
//!
//! Its own test binary, because the counters are process-global: a call whose work
//! is bracketed here must not run beside another test's device work. The
//! [`route_lock`](common::route_lock) the family takes is this file's statement of
//! that, and the answers are checked alongside the counts so a read that submitted
//! once but answered nothing does not pass.

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

const F: DType = DType::F32;
/// Layers in the step: one K and one V handle each. The recognition body's 28,
/// so its fifty-six rows.
const LAYERS: usize = 28;
/// Row width of every cache here.
const W: usize = 8;
/// Rows the short cache declares. The long one declares eight times as many, so
/// the same read spans two capacities.
const CAP: usize = 8;
const CAP_LONG: usize = CAP * 8;

/// One `[cap, W]` cache input per handle — `k0`, `v0`, `k1`, … — each doubled
/// into its own output, so the graph is a step's worth of layers and every handle
/// has a slot of its own.
fn caches(cap: usize) -> Graph {
    let mut g = Graph::new("kv_row_readback_step");
    let mut outs = Vec::new();
    for layer in 0..LAYERS {
        for kv in ["k", "v"] {
            let c = g.input(format!("{kv}{layer}"), Shape::new(&[cap, W], F));
            outs.push(g.add(c, c));
        }
    }
    g.set_outputs(outs);
    g
}

/// The handles one step reads, in the order it reads them.
fn step_handles() -> Vec<String> {
    (0..LAYERS)
        .flat_map(|l| [format!("k{l}"), format!("v{l}")])
        .collect()
}

/// The value handle `i`'s rows start at, so an answered row says which handle it
/// came from.
fn base_of(i: usize) -> f32 {
    100.0 * (i + 1) as f32
}

/// `rows` rows of `W`, `base` up.
fn rows(rows: usize, base: f32) -> Vec<f32> {
    (0..rows * W).map(|i| base + i as f32).collect()
}

/// Every handle bound, run once so the arena takes the seed rows, then marked
/// resident — the state a decode step is in, where a cache's rows are the arena's
/// and no host copy is re-uploaded per step.
fn resident_step(cap: usize) -> (CompiledGraph, Vec<String>) {
    let names = step_handles();
    let mut exe = Session::new(Device::Gpu).compile(caches(cap));
    for (i, name) in names.iter().enumerate() {
        assert!(
            exe.bind_gpu_handle(name, &rows(cap, base_of(i))),
            "{name}: binding a graph input was refused"
        );
    }
    let _ = exe.run(&[]);
    for name in &names {
        assert!(
            exe.prepare_resident_gpu_handle(name),
            "{name}: marking a bound input resident was refused"
        );
    }
    (exe, names)
}

/// Row 0 of every handle, which is the same request at either capacity.
fn step_requests(names: &[String]) -> Vec<(&str, usize, usize)> {
    names.iter().map(|n| (n.as_str(), 0, W)).collect()
}

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available())
}

#[test]
fn a_steps_rows_leave_in_one_submission_at_either_cache_capacity() {
    let _route = common::route_lock();
    if skip() {
        return;
    }
    for cap in [CAP, CAP_LONG] {
        let (exe, names) = resident_step(cap);
        let requests = step_requests(&names);

        reset_device_counts();
        let answers = exe.read_gpu_handle_rows(&requests);
        let (submits, waits) = (device_submit_count(), device_wait_count());
        eprintln!(
            "capacity {cap}: {} rows in one call → {submits} submission(s), {waits} wait(s)",
            requests.len()
        );
        assert_eq!(
            (submits, waits),
            (1, 1),
            "reading a step's {} rows at capacity {cap} cost {submits} submission(s) and \
             {waits} wait(s): the whole step must leave in one round trip, or the step \
             pays a round trip per row however small the rows are",
            requests.len()
        );

        // ...and they are the rows the caches hold, not merely plausible bytes.
        for (i, answer) in answers.iter().enumerate() {
            let want: Vec<f32> = (0..W).map(|j| base_of(i) + j as f32).collect();
            assert_eq!(
                answer.as_deref(),
                Some(want.as_slice()),
                "handle {} answered {answer:?}, not row 0 of the cache resident there",
                names[i]
            );
        }

        // The same rows one at a time is the shape the batch replaces: a
        // submission and a wait per row, and the same at either capacity — which
        // is what the batch's one of each is measured against.
        reset_device_counts();
        for &(name, row, row_inner) in &requests {
            assert!(
                exe.read_gpu_handle_row(name, row, row_inner).is_some(),
                "{name}: a row of a resident cache answered nothing"
            );
        }
        let (one_row_submits, one_row_waits) = (device_submit_count(), device_wait_count());
        eprintln!(
            "capacity {cap}: {} rows one at a time → {one_row_submits} submission(s), \
             {one_row_waits} wait(s)",
            requests.len()
        );
        assert_eq!(
            (one_row_submits, one_row_waits),
            (requests.len() as u64, requests.len() as u64),
            "a row asked for on its own is a round trip of its own — the batch is the \
             only shape that carries a step"
        );
    }
}

#[test]
fn a_batch_answered_from_the_host_mirror_asks_nothing_of_the_device() {
    let _route = common::route_lock();
    if skip() {
        return;
    }
    let names = step_handles();
    let mut exe = Session::new(Device::Gpu).compile(caches(CAP));
    for (i, name) in names.iter().enumerate() {
        assert!(
            exe.bind_gpu_handle(name, &rows(CAP, base_of(i))),
            "{name}: binding a graph input was refused"
        );
    }
    // No run and no residency mark: the rows are the mirror's, and the arena has
    // none of them.
    let requests = step_requests(&names);

    reset_device_counts();
    let answers = exe.read_gpu_handle_rows(&requests);
    let (submits, waits) = (device_submit_count(), device_wait_count());
    assert_eq!(
        (submits, waits),
        (0, 0),
        "a batch whose rows are all in the host mirror cost {submits} submission(s) and \
         {waits} wait(s): a row the device does not hold needs no round trip to read"
    );
    for (i, answer) in answers.iter().enumerate() {
        let want: Vec<f32> = (0..W).map(|j| base_of(i) + j as f32).collect();
        assert_eq!(
            answer.as_deref(),
            Some(want.as_slice()),
            "handle {} answered {answer:?}, not the row the host mirror holds for it",
            names[i]
        );
    }
}
