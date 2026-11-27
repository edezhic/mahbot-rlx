// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acceptance harness for the cold read the upload's request is issued over: the extents a
//! compile really reads, walked in the three orders a cold read is measured in, on four
//! subjects.
//!
//! # The zero-residency method
//!
//! Nothing in the product purges the page cache. The stand does, through `src/cold.rs`,
//! which is this harness's method and carries its reasoning; this file is where it was
//! proved, and the figures below are its first use. The reproducible no-root method is an
//! invalidation over a read-only mapping of the whole model file: `msync(MS_INVALIDATE)`
//! drops its clean pages from the page cache on this platform, where
//! `madvise(MADV_DONTNEED)` and `MADV_FREE` do not. Residency is then read from a second,
//! never-faulted mapping of the same file, so a zero is the page cache's answer rather
//! than the invalidated mapping's own page-table entries; it is asserted over the exact
//! extents immediately before every arm, printed beside the take, and re-established
//! between every one of them: a second round on the same file reads in a fraction of the
//! time because the first round brought the pages in, so paired alternating takes mean
//! nothing without the invalidation.
//!
//! The note in `GgufFile::release_mapped_pages` that these primitives are no-ops on macOS
//! measured the mapping's resident set, not the page cache; both observations hold, and that
//! note lives in the upstream-facing loader this harness does not touch.
//!
//! # The three orders
//!
//! The upload loop walks its reads in ascending file offset and asks the kernel for one group
//! of extents ahead of the copies it feeds; [`readahead`]'s module doc is where that mechanism,
//! its one-group-ahead bound and its refusal to bridge a hole are argued, and the doc of the
//! request itself (`loader::request_readahead`) owns the figures it was measured at. This
//! harness holds the walk to its claim: one extent list, copied into one reused sink, in the
//! orders a cold read has had.
//!
//! - `Stack` — the upload's order before the readahead route: the build's own list, drained
//!   backwards from a stack, each window asking the kernel for itself.
//! - `Forward` — the same extents in ascending file offset, each window still asking for
//!   itself: what the order alone is worth.
//! - `Grouped` — [`readahead::read_extents`]: the forward walk with one group of requests
//!   issued ahead of the copies it feeds. This is the route the product takes.
//!
//! Every arm maps and releases one window per extent and copies its bytes into one reused
//! buffer, which is what the device's staging write pays; the arms differ only in order and in
//! what they ask the kernel for.
//!
//! # What is measured, and the thresholds
//!
//! Four subjects, each one file, in this order: the two bodies' prefills, then the graph each
//! pair compiles over its `mmproj` — the recognition encoder, and the synthesis sound front
//! (codec, wave transformer and DAC; that pair's `mmproj` holds no projector). The recognition
//! pair's own projector is a few megabytes, a couple of milliseconds that five paired takes
//! cannot resolve, so it is not a subject. A subject's graph is built once and the file extents
//! its compile would read are taken from the build ([`plan_extents`]): a fused projection
//! contributes one extent per part, and a half-precision tensor contributes the file bytes its
//! widening reads. The build is then dropped, so no figure below is charged for a graph nobody
//! compiles.
//!
//! [`ROUNDS`] paired rounds per subject, the three arms' order rotated every round so a
//! within-round drift lands on all of them rather than on one delta, and each arm's median taken
//! over the rounds. The grouped route must beat the forward walk and the stack order — those are
//! the claims the route makes; the order's own figure is printed and not gated, because what it
//! is for is the grouping under it and `readahead`'s doc owns its readings. The plan is also held
//! to its two invariants before a page is read — a group of several extents stays within
//! [`readahead::GROUP_BYTES`], and no group's span touches the vocabulary table — and the two
//! bodies assert, right after a grouped arm, that the table's own pages are still not resident.
//!
//! The upload of the synthesis body's bytes is timed through [`graphs::Compiled::new`] with the
//! request as it is — the route carries no switch to turn it off, so its baseline is the loader
//! doc's figure rather than an arm of its own here — and `UPLOAD_TAKES` times on a resident file,
//! the warm path the request must not move. The cold ceilings are **2.7 GB/s** for a read — the
//! 300 ms the synthesis prefill's 811.5 MiB was accepted against, as a rate so every subject
//! carries the same bound — and **500 ms** for the upload, the latter the median of three takes;
//! the figures they are drawn from, and the residency they were taken at, are in
//! `loader::request_readahead`'s doc comment, which owns them. Every figure here moves with the
//! machine, so the load average is printed beside each and a red take at high load is a take to
//! re-run rather than a regression.
//!
//! The invalidation is this platform's — Linux's equivalent is
//! `posix_fadvise(POSIX_FADV_DONTNEED)` — which is why the harness is macOS-only. The request's
//! own Linux arm is not even compiled on this host and has never been run on a Linux one, so
//! every ceiling here is the macOS arm's.
//!
//! A grouped arm can end with the last group's request still delivering what its copies did not
//! fault — the bridged slack between extents, under a mebibyte on these subjects — and a page
//! that lands after the next arm's invalidation refuses that take rather than making it look
//! warm. Nothing here can wait that out: the request's coverage is wider than the pages a
//! residency reading over the extents can see. A refused take is re-run.
//!
//! Needs the installed model directory and the device; a machine without either skips, and a
//! pair that is present but will not open fails. Run with `cargo test --release -p
//! mahbot-inference --features hand-run --test cold_read -- --ignored --nocapture`.

#![cfg(target_os = "macos")]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use mahbot_inference::emit;
use mahbot_inference::graphs;
use mahbot_inference::loader::{Model, Role};
use mahbot_inference::readahead;
use rlx_flow::BuiltModel;
use rlx_ir::Tick;

mod support;

/// The prefill's prompt length. Any length compiles the same parameter set, and this harness
/// reads bytes rather than activations.
const PREFILL_ROWS: usize = 16;
/// The sound front group's frame count, for the same reason.
const SOUND_FRAMES: usize = 8;
/// The encoder window's frame count, for the same reason.
const ENCODER_FRAMES: usize = 128;
/// The ceiling the cold read is accepted against — the 300 ms the synthesis prefill's 811.5 MiB
/// was taken at, as a rate, so a subject of another size carries the same bound — and the cold
/// upload's.
const READ_CEILING_GB_S: f64 = 2.7;
const UPLOAD_CEILING_MS: f64 = 500.0;
/// Paired rounds of the three read arms, in a rotated order every round; the warm-up they would
/// otherwise pay is the untimed `pin_arms` pass before them.
const ROUNDS: usize = 5;
/// Upload takes per arm, of which that arm's median is read: one take is a figure the machine's
/// load can carry to the ceiling on its own.
const UPLOAD_TAKES: usize = 3;

/// The file extents a compile of `built` reads, in the order the build declared them: the list
/// the upload loop walks by file offset once it is planned.
///
/// The predicates below are the ones `graphs::Runner::compile` routes by, and this must not
/// quietly grow another: a name that is neither fused, bound nor kept at the file's width is a
/// parameter the build carried, and a carried parameter always carries bytes. A file-backed
/// route the compile adds and this does not know about leaves the build's bytes empty and fails
/// here, rather than shrinking the subject without saying so. A route that both carried bytes
/// and read a file would pass this check; the upload subject's `upload.bytes == bound`
/// assertion is what catches that one.
fn plan_extents(built: &BuiltModel) -> Vec<readahead::Extent<()>> {
    let mut extents = Vec::new();
    for (name, bytes, _) in &built.typed_params {
        if let Some(fused) = emit::fused_of(name) {
            for part in &fused.parts {
                let (start, len) = fused.model.file_extent(part).expect("a fused part's bytes");
                extents.push(readahead::Extent {
                    model: Arc::clone(&fused.model),
                    key: part.clone(),
                    start,
                    len,
                    tag: (),
                });
            }
        } else if let Some((model, key)) = emit::bound_weight(name) {
            let (start, len) = model.file_extent(&key).expect("a bound tensor's bytes");
            extents.push(readahead::Extent {
                model,
                key,
                start,
                len,
                tag: (),
            });
        } else {
            assert!(
                !bytes.is_empty(),
                "`{name}` is neither fused nor bound and carries no bytes: the compile has a \
                 file-backed route this harness does not know, so the subject it prices here \
                 would be smaller than the read"
            );
        }
    }
    // The parameters the build handed over at the file's width: their bytes never enter the
    // build, and the compile widens them out of the file as it writes the device — reads of
    // this subject like any other.
    for (_, model, key) in emit::host_width_reads() {
        let (start, len) = model
            .file_extent(&key)
            .expect("a file-width tensor's bytes");
        extents.push(readahead::Extent {
            model,
            key,
            start,
            len,
            tag: (),
        });
    }
    extents
}

/// The three orders a cold read is measured in: the stack order the upload had before the
/// readahead route (its build list drained backwards, each window asking for itself), the same
/// extents walked forward by file offset with each window still asking for itself, and the
/// grouped forward walk the upload takes.
#[derive(Clone, Copy)]
enum Route {
    Stack,
    Forward,
    Grouped,
}

impl Route {
    /// Where this route's takes sit in a round's array: the same slot every round, whatever
    /// order the round ran them in.
    fn slot(self) -> usize {
        self as usize
    }
}

/// The routes in the order round `round` runs them: the round starts one further along each
/// time and cycles, so a within-round drift lands on all of them rather than on one delta.
fn route_order(round: usize) -> [Route; 3] {
    const ROUTES: [Route; 3] = [Route::Stack, Route::Forward, Route::Grouped];
    let mut order = ROUTES;
    order.rotate_left(round % ROUTES.len());
    order
}

/// The indices of `extents` in ascending file offset: the order the product's walk takes, which
/// the build's own list is only by accident.
fn forward_order(extents: &[readahead::Extent<()>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..extents.len()).collect();
    order.sort_by_key(|&i| extents[i].start);
    order
}

/// The page cache, dropped on demand and read back through a mapping of its own: `map` is what
/// the invalidation walks, and `probe` is never faulted, so a zero read through it is the page
/// cache's answer rather than the invalidated mapping's page-table entries. The descriptor is
/// the harness's own, so an arm that does not go through the route maps the same windows.
struct Cold {
    file: File,
    map: memmap2::Mmap,
    probe: memmap2::Mmap,
    scratch: Vec<u8>,
    page: usize,
}

impl Cold {
    fn new(path: &Path) -> Cold {
        let file = File::open(path).expect("the body file this harness was pointed at");
        // SAFETY: read-only mappings of a model file, which nothing in this tree writes.
        let (map, probe) = unsafe {
            (
                memmap2::MmapOptions::new()
                    .map(&file)
                    .expect("a model mapping"),
                memmap2::MmapOptions::new()
                    .map(&file)
                    .expect("a model mapping"),
            )
        };
        Cold {
            file,
            map,
            probe,
            scratch: Vec::new(),
            page: page_size(),
        }
    }

    /// Drop the file's clean pages from the page cache and answer the share of `extents` that
    /// was resident before the invalidation and the share that still is — zero when it did its
    /// job.
    fn reestablish(&mut self, extents: &[readahead::Extent<()>]) -> (f64, f64) {
        let before = self.resident(extents);
        // SAFETY: the mapping is read-only and live; the kernel only walks it.
        let rc = unsafe {
            libc::msync(
                self.map.as_ptr() as *mut libc::c_void,
                self.map.len(),
                libc::MS_INVALIDATE,
            )
        };
        assert_eq!(
            rc,
            0,
            "msync(MS_INVALIDATE): {}",
            std::io::Error::last_os_error()
        );
        (before, self.resident(extents))
    }

    /// The resident share of `extents`, from `mincore` over each one's pages through the
    /// never-faulted mapping.
    fn resident(&mut self, extents: &[readahead::Extent<()>]) -> f64 {
        let page = self.page;
        let mut resident = 0usize;
        let mut pages = 0usize;
        for e in extents {
            let lo = e.start as usize & !(page - 1);
            let hi = (((e.start + e.len) as usize).div_ceil(page) * page).min(self.probe.len());
            let n = (hi - lo).div_ceil(page);
            self.scratch.resize(n, 0);
            // SAFETY: `lo` is page-aligned inside the read-only mapping, and `scratch` holds
            // one byte per page of the range, which `mincore` writes.
            let rc = unsafe {
                libc::mincore(
                    self.probe.as_ptr().add(lo) as *const libc::c_void,
                    hi - lo,
                    self.scratch.as_mut_ptr().cast(),
                )
            };
            assert_eq!(rc, 0, "mincore: {}", std::io::Error::last_os_error());
            resident += self.scratch.iter().filter(|b| **b & 1 != 0).count();
            pages += n;
        }
        resident as f64 / pages as f64
    }
}

/// The file at zero residency, asserted: every arm and every take below starts cold, and a
/// refusal here is the one thing that would make a figure read partly warm without saying so.
/// Answers the share that was resident before the invalidation.
fn cold_start(cold: &mut Cold, extents: &[readahead::Extent<()>], what: &str) -> f64 {
    let (before, after) = cold.reestablish(extents);
    assert_eq!(
        after, 0.0,
        "{what}: the invalidation left {after:.3} of the extents resident (was {before:.3}), so \
         this arm would not be cold"
    );
    before
}

/// One extent's window, mapped by hand: the pin's addressing, which the route must reproduce
/// byte for byte.
fn window(file: &File, e: &readahead::Extent<()>) -> memmap2::Mmap {
    // SAFETY: a read-only window of a model file, opened read-only and never written; the
    // tensor's bytes lie inside it.
    unsafe {
        memmap2::MmapOptions::new()
            .offset(e.start)
            .len(e.len as usize)
            .map(file)
    }
    .expect("a window of the body file")
}

/// One extent's bytes through the route's own window, the per-window request included: what the
/// device's staging write pays.
fn copy_window(e: &readahead::Extent<()>, sink: &mut Vec<u8>) {
    e.model
        .with_mapped(&e.key, |bytes| {
            sink.clear();
            sink.extend_from_slice(bytes);
        })
        .expect("a window of the body file");
}

/// One arm over every extent: map the window, copy its bytes into one reused buffer — what the
/// device's staging write pays — and release the window. `route` picks both the order and what
/// the kernel is asked for: `Stack` walks the build's list backwards with each window asking for
/// itself, `Forward` walks the same extents by file offset with each window still asking for
/// itself, and `Grouped` hands the whole list to [`readahead::read_extents`], which asks for one
/// group ahead of the copies it feeds.
fn read_arm(
    route: Route,
    extents: &[readahead::Extent<()>],
    forward: &[usize],
    sink: &mut Vec<u8>,
) -> f64 {
    let t = Tick::now();
    match route {
        Route::Stack => {
            for e in extents.iter().rev() {
                copy_window(e, sink);
            }
        }
        Route::Forward => {
            for &i in forward {
                copy_window(&extents[i], sink);
            }
        }
        Route::Grouped => readahead::read_extents(extents, |_, bytes| {
            sink.clear();
            sink.extend_from_slice(bytes);
            Ok(())
        })
        .expect("a grouped cold read"),
    }
    Tick::now().elapsed_ns(t) as f64 / 1e6
}

/// Every route must address the same bytes, so this pass pins the extent list to the hand-mapped
/// baseline extent by extent: a route change that moved a window shows up here rather than as a
/// plausible number, and the same bytes reaching the device is the property every arm keeps. It
/// is also the arms' warm-up — the first window of a process pays the mapping and the descriptor,
/// which no arm should be charged for — so it costs no read of its own.
fn pin_arms(file: &File, extents: &[readahead::Extent<()>], sink: &mut Vec<u8>) {
    for e in extents {
        sink.clear();
        sink.extend_from_slice(&window(file, e)[..]);
        let same = e
            .model
            .with_mapped(&e.key, |bytes| bytes == sink.as_slice())
            .expect("a window of the body file");
        assert!(
            same,
            "`{}` is a different byte range in the route's window than in the hand-mapped one",
            e.key
        );
    }
    println!(
        "the arms address the same bytes over {} extents",
        extents.len()
    );
}

/// The state an arm is measured in: the file's pages dropped from the page cache before every
/// arm, or resident with nothing invalidated.
#[derive(Clone, Copy)]
enum State {
    Cold,
    Warm,
}

impl State {
    /// The state every arm and every take must start in, asserted: `Cold` invalidates and
    /// demands zero residency, `Warm` demands the file resident and invalidates nothing.
    fn ensure(self, cold: &mut Cold, extents: &[readahead::Extent<()>], what: &str) {
        match self {
            State::Cold => {
                cold_start(cold, extents, what);
            }
            State::Warm => assert_eq!(
                cold.resident(extents),
                1.0,
                "{what}: the file was not resident, so this arm would not be warm"
            ),
        }
    }
}

/// [`ROUNDS`] paired takes over the three routes, the order rotated every round so a
/// within-round drift lands on all of them rather than on the deltas, answering each route's
/// median. Every arm starts cold — the file's pages dropped and zero residency asserted
/// immediately before it — and `vocab` is the vocabulary table's own extent, checked right after
/// a grouped arm, before the next invalidation: a group that reached across the table would have
/// read bytes no parameter of this graph asks for.
fn paired(
    cold: &mut Cold,
    extents: &[readahead::Extent<()>],
    forward: &[usize],
    vocab: Option<&readahead::Extent<()>>,
    sink: &mut Vec<u8>,
) -> [f64; 3] {
    let mut takes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for round in 0..ROUNDS {
        let mut times = [0f64; 3];
        for route in route_order(round) {
            State::Cold.ensure(cold, extents, "cold");
            times[route.slot()] = read_arm(route, extents, forward, sink);
            if let (Route::Grouped, Some(vocab)) = (route, vocab) {
                let resident = cold.resident(std::slice::from_ref(vocab));
                assert_eq!(
                    resident, 0.0,
                    "the group reached across the vocabulary table: {resident:.6} of its pages \
                     were resident after a grouped cold read"
                );
            }
        }
        println!(
            "cold round {round}: stack {:.1} ms, forward {:.1} ms, grouped {:.1} ms",
            times[0], times[1], times[2]
        );
        for (slot, time) in times.iter().enumerate() {
            takes[slot].push(*time);
        }
    }
    [
        support::median(&takes[0]),
        support::median(&takes[1]),
        support::median(&takes[2]),
    ]
}

fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name reads a process constant.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// The one-minute load average: the figure that says whether a take is worth reading.
fn load_average() -> f64 {
    let mut load = [0f64; 3];
    // SAFETY: `getloadavg` writes at most three values into the array it is given.
    unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
    load[0]
}

/// The vocabulary table's extent, cut to the pages that lie wholly inside it: a tensor's first
/// and last page are shared with whatever the file put beside it, and a group that read that
/// neighbour leaves them resident — what the assertion is about is the table's own bytes, which
/// no group may reach.
fn interior(model: &Arc<Model>, key: &str, page: usize) -> readahead::Extent<()> {
    let (start, len) = model.file_extent(key).expect("the vocabulary table");
    let page = page as u64;
    let lo = start.next_multiple_of(page);
    let hi = (start + len) / page * page;
    readahead::Extent {
        model: Arc::clone(model),
        key: key.to_string(),
        start: lo,
        len: hi.saturating_sub(lo),
        tag: (),
    }
}

/// One subject: take the extents its compile would read, drop the build, and measure the three
/// routes over them at zero residency before every arm. `vocab` says whether the subject's file
/// holds the vocabulary table, whose own extent no group may reach — checked in the plan here and
/// in the page cache after every grouped arm. Answers the extents, so a caller can hand them to
/// the compile that follows.
fn subject(
    label: &str,
    built: BuiltModel,
    vocab: bool,
    sink: &mut Vec<u8>,
) -> Vec<readahead::Extent<()>> {
    let extents = plan_extents(&built);
    drop(built);
    let path = extents
        .first()
        .expect("a subject reads at least one file extent")
        .model
        .path
        .clone();
    assert!(
        extents.iter().all(|e| e.model.path == path),
        "{label}: the extents span more than one file, so this subject is not one file"
    );
    let bytes: u64 = extents.iter().map(|e| e.len).sum();
    let spans: Vec<(usize, u64, u64)> = extents.iter().map(|e| (0, e.start, e.len)).collect();
    let groups = readahead::groups(&spans);
    let span_bytes: u64 = groups.iter().map(|g| g.len).sum();
    let forward = forward_order(&extents);
    let mut cold = Cold::new(&path);
    println!(
        "{label}: {} extents, {:.1} MiB of {}; {} groups requesting {:.1} MiB ({:.1} MiB of \
         slack); load {:.2}",
        extents.len(),
        bytes as f64 / (1 << 20) as f64,
        path.display(),
        groups.len(),
        span_bytes as f64 / (1 << 20) as f64,
        (span_bytes - bytes) as f64 / (1 << 20) as f64,
        load_average(),
    );
    // The plan's own two invariants, before a page is read: a group of several extents stays
    // within [`readahead::GROUP_BYTES`] (one extent larger than a group cannot be split), and no
    // group's span touches the vocabulary table.
    let table_extent = vocab.then(|| {
        extents[0]
            .model
            .file_extent("token_embd.weight")
            .expect("the vocabulary table")
    });
    for group in &groups {
        assert!(
            group.extents.len() == 1 || group.len <= readahead::GROUP_BYTES,
            "{label}: a group of {} extents spans {} bytes",
            group.extents.len(),
            group.len
        );
        if let Some((start, len)) = table_extent {
            assert!(
                group.start + group.len <= start || group.start >= start + len,
                "{label}: a group's span [{}, {}) reaches into the vocabulary table at [{start}, \
                 {})",
                group.start,
                group.start + group.len,
                start + len
            );
        }
    }

    // The arms' warm-up and their pinning in one untimed pass: every window mapped by hand and
    // again through the route, the two compared byte for byte. Nothing below is charged for the
    // first window's mapping or the descriptor.
    pin_arms(&cold.file, &extents, sink);
    // The invalidation's own effect, printed where a reader can check it: the pass above read
    // every window, so all of them must now be resident, and the invalidation must drop them.
    let before = cold_start(&mut cold, &extents, "the first arm");
    println!(
        "invalidation: resident {before:.3} → 0.000 over {} extents",
        extents.len()
    );

    let table = vocab.then(|| interior(&extents[0].model, "token_embd.weight", cold.page));
    let [stack, forward_ms, grouped] = paired(&mut cold, &extents, &forward, table.as_ref(), sink);
    let load = load_average();
    println!(
        "{label} cold: stack {stack:.1} ms ({:.2} GB/s), forward {forward_ms:.1} ms ({:.2} GB/s), \
         grouped {grouped:.1} ms ({:.2} GB/s); forward - stack {:+.1} ms, grouped - forward {:+.1} \
         ms, grouped - stack {:+.1} ms; load {load:.2}",
        bytes as f64 / (stack * 1e6),
        bytes as f64 / (forward_ms * 1e6),
        bytes as f64 / (grouped * 1e6),
        forward_ms - stack,
        grouped - forward_ms,
        grouped - stack,
    );
    assert!(
        grouped < forward_ms,
        "{label}: the grouped cold read took {grouped:.1} ms against the forward walk's \
         {forward_ms:.1} ms at one-minute load {load:.2}"
    );
    assert!(
        grouped < stack,
        "{label}: the grouped cold read took {grouped:.1} ms against the stack order's \
         {stack:.1} ms at one-minute load {load:.2}"
    );
    let rate = bytes as f64 / (grouped * 1e6);
    assert!(
        rate >= READ_CEILING_GB_S,
        "{label}: the grouped cold read took {grouped:.1} ms for {:.1} MiB ({rate:.2} GB/s) at \
         one-minute load {load:.2}, under the {READ_CEILING_GB_S} GB/s ceiling",
        bytes as f64 / (1 << 20) as f64
    );
    extents
}

/// The figures, at zero residency before every arm.
#[test]
#[ignore = "measurement only: reads the model file cold; run by hand"]
fn the_prefills_cold_read_and_upload() {
    let _solo = support::solo("the_prefills_cold_read_and_upload");
    let Some(recognition) = support::pair_of(Role::Recognition) else {
        return;
    };
    let Some(synthesis) = support::pair_of(Role::Synthesis) else {
        return;
    };
    let mut sink = Vec::new();

    let built = graphs::body_new_rows_head_chain(recognition.body.clone(), PREFILL_ROWS, 0, true)
        .expect("the recognition prefill graph builds");
    subject("recognition body", built, true, &mut sink);

    let built = graphs::body_new_rows_head_chain(synthesis.body.clone(), PREFILL_ROWS, 0, true)
        .expect("the synthesis prefill graph builds");
    let extents = subject("synthesis body", built, true, &mut sink);

    // The rest of the recognition pair's `mmproj` is the encoder, which is the big cold read of
    // that file and the graph every recognition call compiles over it.
    let built = graphs::asr_encoder(recognition.sound.clone(), ENCODER_FRAMES, 1)
        .expect("the recognition encoder graph builds");
    subject("recognition encoder", built, false, &mut sink);

    // The synthesis pair's `mmproj` is the sound half — codec, wave transformer, DAC — not a
    // projector, so the subject over it is the front group a synthesis call compiles.
    let built = graphs::synthesis_sound_group(
        synthesis.sound.clone(),
        SOUND_FRAMES,
        graphs::codec_groups()[0].clone(),
        0,
    )
    .expect("the sound front group builds");
    subject("synthesis sound", built, false, &mut sink);

    // The upload through the real compile: the same bytes, the device write included. Each take
    // compiles its own graph, and the device's shared weights are released first so that take
    // uploads rather than skips what an earlier one placed. The extents are the synthesis body's,
    // taken from the same build the subject above measured, so the cold state invalidates exactly
    // the bytes the compile reads. The compile's byte count is those extents plus the f32
    // parameters the other route carries, so a tensor that moved out of either set shows up here.
    let body = synthesis.body.clone();
    let mut cold = Cold::new(&body.path);
    let bound: usize = body
        .plans
        .iter()
        .filter(|p| p.name != "token_embd.weight")
        .map(|p| body.raw_len(&p.name).expect("a planned tensor's bytes"))
        .sum();
    let upload_ms = upload_median(
        "cold upload",
        State::Cold,
        &mut cold,
        &body,
        &extents,
        bound,
    );
    let load = load_average();
    println!(
        "cold upload: {upload_ms:.1} ms median of {UPLOAD_TAKES} takes ({:.2} GB/s, load \
         {load:.2})",
        bound as f64 / (upload_ms * 1e6)
    );
    assert!(
        upload_ms <= UPLOAD_CEILING_MS,
        "the prefill's cold upload took {upload_ms:.1} ms at one-minute load {load:.2}, over the \
         {UPLOAD_CEILING_MS} ms ceiling"
    );

    // The warm upload: the same compile on a resident file, which is where the request must cost
    // nothing — the takes above read every window, so nothing is invalidated here. The bound is
    // an ordering smoke test, not a regression detector: a resident file must not upload more
    // slowly than a cold one, and no threshold can tell a small warm regression from the
    // threefold load swings this hand-run test sees, so it is not to be tightened.
    let warm_ms = upload_median(
        "warm upload",
        State::Warm,
        &mut cold,
        &body,
        &extents,
        bound,
    );
    println!(
        "warm upload: {warm_ms:.1} ms median of {UPLOAD_TAKES} takes ({:.2} GB/s, load {:.2})",
        bound as f64 / (warm_ms * 1e6),
        load_average()
    );
    assert!(
        warm_ms <= upload_ms,
        "the prefill's warm upload took {warm_ms:.1} ms against a cold median of {upload_ms:.1} ms"
    );
}

/// `UPLOAD_TAKES` takes of one upload arm, answering their median.
fn upload_median(
    label: &str,
    state: State,
    cold: &mut Cold,
    body: &Arc<Model>,
    extents: &[readahead::Extent<()>],
    bound: usize,
) -> f64 {
    let takes: Vec<f64> = (0..UPLOAD_TAKES)
        .map(|round| {
            upload_take(
                state,
                &format!("{label} {round}"),
                cold,
                body,
                extents,
                bound,
            )
        })
        .collect();
    support::median(&takes)
}

/// One upload through the real compile, from the state `state` names: the device's shared
/// weights are released so the take uploads rather than skips what an earlier compile placed,
/// and the byte count and skip are asserted against the harness's own extent set. Answers the
/// parameter upload's interval — the take's wall clock, which also pays the lowering, is printed
/// beside it.
fn upload_take(
    state: State,
    label: &str,
    cold: &mut Cold,
    body: &Arc<Model>,
    extents: &[readahead::Extent<()>],
    bound: usize,
) -> f64 {
    // Only the device's copy has to go: the host weight cache the f32 parameters are bound from
    // must stay, so this is `clear_shared_weights` rather than `release_process_memory`.
    rlx_wgpu::buffer::clear_shared_weights();
    let built = graphs::body_new_rows_head_chain(body.clone(), PREFILL_ROWS, 0, true)
        .expect("the prefill graph builds");
    // Immediately before the timed call: a build that ever touched a packed window would
    // otherwise leave the take partly warm, and the device's skip counter would not catch it.
    state.ensure(cold, extents, label);
    let t = Tick::now();
    let compiled = graphs::Compiled::new(built).expect("the prefill graph compiles");
    let compile_ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
    let upload = compiled.upload();
    println!(
        "{label}: {:.1} ms of {compile_ms:.1} ms ({:.1} MiB, {:.2} GB/s, skipped {:.1} MiB, \
         load {:.2})",
        upload.ms,
        upload.bytes as f64 / (1 << 20) as f64,
        upload.gb_per_s(),
        upload.skipped as f64 / (1 << 20) as f64,
        load_average(),
    );
    assert_eq!(
        upload.bytes as usize, bound,
        "the prefill bound a different tensor set than this harness measured"
    );
    assert_eq!(
        upload.skipped, 0,
        "{label}: a parameter was already resident, so this was not an upload"
    );
    upload.ms
}
