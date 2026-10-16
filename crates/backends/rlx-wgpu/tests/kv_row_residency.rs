// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A device-resident KV cache on this route is a value this executable holds,
//! and it says so.**
//!
//! Residency is what removes the O(context) cost of a decode step: the cache
//! lives in the arena, a registered feed folds each new token's row into it
//! device-side, and the runtime stops re-uploading the cache as an ordinary
//! input every step. Two things had to be true for a consumer to rely on it, and
//! neither was:
//!
//! * a feed answered `true` for every registered name whether or not the row
//!   reached the cache — a name that is not a graph input, a cache this
//!   executable never bound, a `dst_row` past the cache input's own extent (which
//!   wrote into whatever the arena holds next to the slot) all reported success;
//! * the whole cache could only be read back, so a consumer that needs one row
//!   paid for every row. A row now costs the row, and a step's rows — one per
//!   layer per K/V, across as many handles — cost one device round trip: the
//!   batch readback is the shape a step's price turns on, since a call that
//!   resolves one handle pays a submission and a wait for each row.
//!
//! The answer must also stay honest as the executable's storage changes: a clone
//! and a recompile both rebuild it, and a name whose rows went with the old
//! storage is no longer held — a feed says so, and the consumer re-seeds, rather
//! than the graph quietly computing against a zeroed cache.

use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Row width of every cache here.
const W: usize = 8;
/// Rows `cache_a` declares.
const ROWS_A: usize = 4;
/// Rows `cache_b` declares — deliberately fewer than `cache_a`, so a call that
/// covers both names has a destination one cache has and the other does not.
const ROWS_B: usize = 2;

/// `out_a = cache_a + extra`, `out_b = cache_b + cache_b`: `extra` moves the
/// output row a feed folds in without touching either cache, which is what makes
/// a copy that should not have happened visible in the cache's own readback.
fn two_caches() -> Graph {
    let mut g = Graph::new("kv_row_residency");
    let a = g.input("cache_a", Shape::new(&[ROWS_A, W], F));
    let b = g.input("cache_b", Shape::new(&[ROWS_B, W], F));
    let x = g.input("extra", Shape::new(&[ROWS_A, W], F));
    let out_a = g.add(a, x);
    let out_b = g.add(b, b);
    g.set_outputs(vec![out_a, out_b]);
    g
}

/// The graph's output IS its cache input, so a feed of a row onto itself has
/// nothing to move.
fn cache_is_the_output() -> Graph {
    let mut g = Graph::new("kv_row_identity");
    let c = g.input("cache", Shape::new(&[ROWS_A, W], F));
    g.set_outputs(vec![c]);
    g
}

/// `rows` rows of `W`, `base` up, so a row says where it came from.
fn rows(rows: usize, base: f32) -> Vec<f32> {
    (0..rows * W).map(|i| base + i as f32).collect()
}

fn row_of(v: &[f32], row: usize) -> &[f32] {
    &v[row * W..(row + 1) * W]
}

/// Bind, register and run both caches once: the state a decode loop reaches after
/// its first step, with both caches held by this executable and their rows in the
/// arena.
fn resident_caches() -> WgpuExecutable {
    let mut exe = WgpuExecutable::compile(two_caches());
    assert!(
        exe.bind_gpu_handle("cache_a", &rows(ROWS_A, 100.0)),
        "binding a graph input was refused"
    );
    assert!(
        exe.bind_gpu_handle("cache_b", &rows(ROWS_B, 200.0)),
        "binding a graph input was refused"
    );
    assert!(exe.register_kv_row_feed("cache_a", 0), "cache_a → output 0");
    assert!(exe.register_kv_row_feed("cache_b", 1), "cache_b → output 1");
    let _ = exe.run(&[]);
    exe
}

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available())
}

#[test]
fn a_registration_answers_for_exactly_the_names_binding_accepts() {
    if skip() {
        return;
    }
    let mut exe = WgpuExecutable::compile(two_caches());
    assert!(
        !exe.register_kv_row_feed("not_an_input", 0),
        "a name that is not a graph input registered as a working row feed — a \
         consumer would feed it forever and never learn nothing moved"
    );
    assert!(
        !exe.set_gpu_handle_feed("not_an_input", 0),
        "a name that is not a graph input registered as a working prefix feed"
    );
    for (name, cap) in [("cache_a", ROWS_A), ("cache_b", ROWS_B)] {
        assert!(
            exe.bind_gpu_handle(name, &rows(cap, 0.0)),
            "{name}: bind_gpu_handle refused a graph input"
        );
        assert!(
            exe.register_kv_row_feed(name, 0),
            "{name}: register_kv_row_feed refused a name binding accepts"
        );
        assert!(
            exe.set_gpu_handle_feed(name, 0),
            "{name}: set_gpu_handle_feed refused a name binding accepts"
        );
    }
}

#[test]
fn a_feed_into_a_cache_this_executable_does_not_hold_reports_failure() {
    if skip() {
        return;
    }
    let mut exe = WgpuExecutable::compile(two_caches());
    assert!(exe.register_kv_row_feed("cache_a", 0), "cache_a → output 0");
    // Registered, but never bound or seeded here: there is no cache in this
    // executable's storage for a row to go into.
    assert!(
        !exe.feed_kv_row(0, 0, W),
        "a feed into a cache this executable never bound reported success"
    );
}

#[test]
fn a_row_outside_the_extent_the_cache_declares_is_refused_and_nothing_moves() {
    if skip() {
        return;
    }
    let mut exe = resident_caches();
    // A fingerprinted `extra` row: a copy that should not happen would put the
    // row it folded (cache_a's row 0 plus `extra`) in cache_a's row 2, where the
    // readback below would find it.
    let extra = rows(ROWS_A, 900.0);
    let _ = exe.run(&[("extra", &extra)]);
    assert!(exe.feed_kv_row(0, ROWS_B - 1, W), "a row both caches have");
    let before_a = exe.read_gpu_handle("cache_a").expect("cache_a readable");
    let before_b = exe.read_gpu_handle("cache_b").expect("cache_b readable");
    let folded: Vec<f32> = row_of(&before_a, 0)
        .iter()
        .zip(&extra)
        .map(|(a, x)| a + x)
        .collect();
    assert_eq!(
        row_of(&before_a, ROWS_B - 1),
        folded.as_slice(),
        "the feed did not fold the output row into the cache"
    );

    // `dst_row = ROWS_B` exists in cache_a and not in cache_b. One name that
    // cannot take its row makes the call report failure, and a failure may not
    // leave cache_a's row already written.
    assert!(
        !exe.feed_kv_row(0, ROWS_B, W),
        "a feed asked for a row past the extent cache_b declares reported success"
    );
    // A row index near `usize::MAX` is out of the extent by any reading, and must
    // be refused rather than have a wrapping product pass the bound check.
    assert!(
        !exe.feed_kv_row(0, usize::MAX, W),
        "a feed at usize::MAX reported success — the extent bound wrapped"
    );
    // The symmetric case: the source row is bounded by the output's own extent
    // too, so a row it does not hold is not read out of whatever sits beside it.
    assert!(
        !exe.feed_kv_row(usize::MAX, 0, W),
        "a feed from usize::MAX reported success — the source bound wrapped"
    );
    assert_eq!(
        exe.read_gpu_handle("cache_a").expect("cache_a readable"),
        before_a,
        "a failed feed left cache_a written into — the call reports failure for \
         the whole set, so no name may have moved"
    );
    assert_eq!(
        exe.read_gpu_handle("cache_b").expect("cache_b readable"),
        before_b,
        "a failed feed changed cache_b"
    );
}

#[test]
fn a_feed_whose_row_already_sits_in_the_resident_cache_still_succeeds() {
    if skip() {
        return;
    }
    let mut exe = WgpuExecutable::compile(cache_is_the_output());
    let seed = rows(ROWS_A, 1.0);
    assert!(exe.bind_gpu_handle("cache", &seed));
    assert!(exe.register_kv_row_feed("cache", 0));
    assert_eq!(
        exe.run(&[])[0],
        seed,
        "the graph did not hand its cache input back"
    );
    // The graph's output IS the cache input: nothing to move, and a correctly
    // wired consumer must not be pushed onto its host fallback for it.
    assert!(
        exe.feed_kv_row(1, 1, W),
        "a row write where the row already sits reported failure"
    );
    assert_eq!(
        exe.read_gpu_handle("cache").expect("cache readable"),
        seed,
        "the no-op row write moved bytes"
    );
    // A different destination is a legitimate intra-buffer relocate, not a
    // no-op: the aliased case has to keep working.
    assert!(exe.feed_kv_row(0, 1, W), "row 0 → row 1");
    assert_eq!(
        row_of(&exe.read_gpu_handle("cache").expect("cache readable"), 1),
        row_of(&seed, 0),
        "the aliased row write did not relocate the row"
    );
}

#[test]
fn a_row_readback_answers_what_the_whole_handle_readback_answers() {
    if skip() {
        return;
    }
    let mut exe = resident_caches();
    let extra = rows(ROWS_A, 900.0);
    let _ = exe.run(&[("extra", &extra)]);
    assert!(
        exe.feed_kv_row(0, 1, W),
        "fold the output row into both caches"
    );

    let whole = exe.read_gpu_handle("cache_a").expect("cache_a readable");
    assert_ne!(
        row_of(&whole, 0),
        row_of(&whole, 1),
        "the two rows are identical, so the assertions below cannot tell a row \
         read from a read of the wrong one"
    );
    for row in 0..ROWS_A {
        let got = exe
            .read_gpu_handle_row("cache_a", row, W)
            .unwrap_or_else(|| panic!("row {row} of a resident cache answered nothing"));
        assert_eq!(
            got.len(),
            W,
            "the readback returned the handle ({} values), not the {W}-wide row",
            got.len()
        );
        assert_eq!(
            got,
            row_of(&whole, row),
            "row {row} disagrees with the whole-handle readback of the same cache"
        );
    }
    // Rows that are not there: nothing, rather than another part of the cache.
    assert!(
        exe.read_gpu_handle_row("cache_a", ROWS_A, W).is_none(),
        "a row past the cache's extent answered bytes"
    );
    assert!(
        exe.read_gpu_handle_row("cache_a", 0, 0).is_none(),
        "a zero-length row answered bytes"
    );
    assert!(
        exe.read_gpu_handle_row("cache_a", usize::MAX, W).is_none(),
        "a row readback at usize::MAX answered bytes — the extent bound wrapped"
    );
    // Bound by what the cache declares, not by how long the binding was: an
    // over-long host mirror does not make a row that does not exist readable.
    assert!(
        exe.bind_gpu_handle("cache_a", &rows(ROWS_A + 2, 400.0)),
        "binding a graph input was refused"
    );
    assert!(
        exe.read_gpu_handle_row("cache_a", ROWS_A, W).is_none(),
        "a row past the declared extent answered from an over-long host mirror"
    );
    assert!(
        exe.read_gpu_handle_row("cache_a", ROWS_A - 1, W).is_some(),
        "the last row the cache declares is inside its extent"
    );
}

#[test]
fn a_batch_readback_answers_a_steps_requests_in_one_call() {
    if skip() {
        return;
    }
    let mut exe = resident_caches();
    let extra = rows(ROWS_A, 900.0);
    let _ = exe.run(&[("extra", &extra)]);
    assert!(
        exe.feed_kv_row(0, 1, W),
        "fold the output row into both caches"
    );
    let whole_a = exe.read_gpu_handle("cache_a").expect("cache_a readable");
    let whole_b = exe.read_gpu_handle("cache_b").expect("cache_b readable");

    // A step asks after many handles in one call, so the requests interleave
    // three this executable cannot serve with five it can: a row past cache_b's
    // extent, a name it does not hold, and a zero-width row. Every answer is a
    // position in the same list, which is what an implementation that dropped or
    // shifted a request would get wrong.
    let requests = [
        ("cache_a", 0, W),
        ("cache_b", 0, W),
        ("cache_b", ROWS_B, W),
        ("cache_a", 2, W),
        ("not_here", 0, W),
        ("cache_a", 0, 0),
        ("cache_b", ROWS_B - 1, W),
        ("cache_a", ROWS_A - 1, W),
    ];
    let want = [
        Some(row_of(&whole_a, 0)),
        Some(row_of(&whole_b, 0)),
        None,
        Some(row_of(&whole_a, 2)),
        None,
        None,
        Some(row_of(&whole_b, ROWS_B - 1)),
        Some(row_of(&whole_a, ROWS_A - 1)),
    ];
    let answers = exe.read_gpu_handle_rows(&requests);
    assert_eq!(
        answers.len(),
        requests.len(),
        "the batch answered {} of {} requests",
        answers.len(),
        requests.len()
    );
    for (i, (got, want)) in answers.iter().zip(want).enumerate() {
        assert_eq!(
            got.as_deref(),
            want,
            "request {i} {:?} answered {got:?}, not {want:?}",
            requests[i]
        );
    }
    // Asked again, the same rows answer the same way — reading a cache leaves it
    // reading back as it did.
    assert_eq!(
        exe.read_gpu_handle_rows(&requests),
        answers,
        "the same requests answered differently on the second call"
    );
    // That holds for the requests the batch refused too: they took nothing with
    // them, and the caches read back whole.
    for (name, whole) in [("cache_a", &whole_a), ("cache_b", &whole_b)] {
        assert_eq!(
            exe.read_gpu_handle(name).as_deref(),
            Some(whole.as_slice()),
            "{name} reads back differently after a batch that refused a request for it"
        );
    }

    // Rows this executable holds only in its host mirror are answered from there:
    // a batch must not need an arena slot to answer a row the host holds.
    let mirror = rows(ROWS_A, 400.0);
    assert!(
        exe.bind_gpu_handle("cache_a", &mirror),
        "binding a graph input was refused"
    );
    assert_eq!(
        exe.read_gpu_handle_rows(&[("cache_a", 1, W), ("cache_a", 0, W)])[0].as_deref(),
        Some(row_of(&mirror, 1)),
        "a row of a handle this executable holds only on the host answered nothing"
    );
    // Both sources in one call: cache_a is the host's alone now while cache_b
    // still holds its rows in the arena, and each answer lands at its own
    // request's position.
    let mixed = exe.read_gpu_handle_rows(&[("cache_a", 0, W), ("cache_b", 0, W)]);
    assert_eq!(
        (mixed[0].as_deref(), mixed[1].as_deref()),
        (Some(row_of(&mirror, 0)), Some(row_of(&whole_b, 0))),
        "a batch mixing a host-held row with an arena-held one answered the wrong rows"
    );
}

#[test]
fn a_row_readback_of_a_cache_this_executable_does_not_hold_answers_nothing() {
    if skip() {
        return;
    }
    let mut exe = resident_caches();
    assert!(
        exe.feed_kv_row(0, 1, W),
        "fold the output row into both caches"
    );
    let whole = exe.read_gpu_handle("cache_a").expect("cache_a readable");
    assert_eq!(
        exe.read_gpu_handle_row("cache_a", 0, W).as_deref(),
        Some(row_of(&whole, 0))
    );

    // A clone recompiles its own storage and drops the parent's registrations.
    // Registering the same name on the clone reaches its residency check — a
    // clone refuses this from the empty-feed-map early return otherwise, which a
    // broken residency check would still pass — and it holds no rows, so the row
    // must not go anywhere.
    let mut clone = exe.clone_for_cache();
    assert!(
        clone.register_kv_row_feed("cache_a", 0),
        "a resolved clone cannot register a name it can bind"
    );
    assert!(
        !clone.feed_kv_row(0, 1, W),
        "a clone fed a cache whose rows its storage does not hold"
    );
    assert!(
        clone.read_gpu_handle_row("cache_a", 0, W).is_none(),
        "a clone answered a row of a cache it does not hold"
    );

    // ... and that is recoverable, which is what reporting it is for: re-seeding
    // the cache in the clone makes it a cache this executable holds again, and
    // the row goes where it was asked to.
    let extra = rows(ROWS_A, 900.0);
    assert!(clone.bind_gpu_handle("cache_a", &rows(ROWS_A, 7.0)));
    assert!(clone.register_kv_row_feed("cache_a", 0));
    let _ = clone.run(&[("extra", &extra)]);
    assert!(
        clone.feed_kv_row(0, 1, W),
        "a re-seeded clone still refuses its own cache's row"
    );
    let re = clone.read_gpu_handle("cache_a").expect("cache_a readable");
    let folded: Vec<f32> = row_of(&re, 0)
        .iter()
        .zip(&extra)
        .map(|(a, x)| a + x)
        .collect();
    assert_eq!(
        row_of(&re, ROWS_B - 1),
        folded.as_slice(),
        "the re-seeded clone did not fold the output row in"
    );
}

#[test]
fn a_clone_keeps_the_prefix_feed_its_parent_was_registered_with() {
    if skip() {
        return;
    }
    // A dynamic row count, so the parent's compile is deferred and the clone
    // compiles its own storage rather than inheriting the parent's.
    let dyn_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(W)], F);
    let mut g = Graph::new("kv_prefix_feed_clone");
    let cache = g.input("cache", dyn_shape);
    let out = g.add(cache, cache);
    g.set_outputs(vec![out]);
    let mut exe = WgpuExecutable::compile(g);
    let seed = rows(ROWS_A, 5.0);
    let _ = exe.run(&[("cache", &seed)]);
    assert!(
        exe.set_gpu_handle_feed("cache", 0),
        "register a prefix feed"
    );

    let mut clone = exe.clone_for_cache();
    let _ = clone.run(&[("cache", &seed)]);
    // The fold each run performs for a prefix feed is what installs the handle
    // here: a clone that dropped the record to a registration its still-deferred
    // storage could not answer for would keep running against a cache nothing
    // folds into, and nothing would report that.
    assert!(
        clone.has_gpu_handle("cache"),
        "the clone ran without folding its parent's prefix feed in — the \
         registration was dropped with the storage it was made in"
    );
}

#[test]
fn a_recompile_drops_the_resident_marks_the_arena_went_with() {
    if skip() {
        return;
    }
    // Two rows on the first run, three on the second: the second binding
    // recompiles the graph and rebuilds the arena the resident cache lived in.
    let dyn_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(W)], F);
    let mut g = Graph::new("kv_row_recompile");
    let cache = g.input("cache", dyn_shape);
    let out = g.add(cache, cache);
    g.set_outputs(vec![out]);
    let mut exe = WgpuExecutable::compile(g);

    let seed = rows(ROWS_A, 5.0);
    let _ = exe.run(&[("cache", &seed[..ROWS_B * W])]);
    assert!(exe.bind_gpu_handle("cache", &seed[..ROWS_B * W]), "bind");
    assert!(exe.register_kv_row_feed("cache", 0), "cache → output 0");
    let _ = exe.run(&[("cache", &seed[..ROWS_B * W])]);
    assert!(exe.feed_kv_row(0, 1, W), "fold the output row in");
    let doubled: Vec<f32> = row_of(&seed, 0).iter().map(|v| v * 2.0).collect();
    assert_eq!(
        exe.read_gpu_handle_row("cache", 1, W).as_deref(),
        Some(doubled.as_slice()),
        "the resident cache does not hold the fed row"
    );

    // Same feed, same names, but a rebound shape rebuilt the slot's storage: the
    // rows that were resident went with the old arena.
    let _ = exe.run(&[("cache", &seed)]);
    assert!(
        !exe.feed_kv_row(0, 1, W),
        "a feed reported success against a rebuilt, zeroed cache"
    );
    assert!(
        exe.read_gpu_handle_row("cache", 0, W).is_none(),
        "a row readback answered from the storage that was rebuilt"
    );
}
