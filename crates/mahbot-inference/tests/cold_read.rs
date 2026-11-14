// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acceptance harness for the readahead request `Model::with_mapped` issues before each
//! window's bytes are read: the TTS prefill's packed windows read cold with the request and
//! without it, and the same tensors' upload through the real compile.
//!
//! # The zero-residency method
//!
//! Nothing in the product purges the page cache, and the stand's cold column is a process
//! with no models loaded rather than a cold cache — on a machine whose cache is warm the
//! synthesis cold row does not move at all, so an unchanged `STAND.md` is not a failed
//! measurement. The reproducible no-root method is an invalidation over a read-only mapping
//! of the whole model file: `msync(MS_INVALIDATE)` drops its clean pages from the page cache
//! on this platform, where `madvise(MADV_DONTNEED)` and `MADV_FREE` do not. Residency is then
//! read from a second, never-faulted mapping of the same file, so a zero is the page cache's
//! answer rather than the invalidated mapping's own page-table entries; it is asserted over
//! the exact extents immediately before every arm, printed beside the take, and re-established
//! between every one of them: a second round on the same file reads in a fraction of the time
//! because the first round brought the pages in, so paired alternating takes mean nothing
//! without the invalidation.
//!
//! The note in `GgufFile::release_mapped_pages` that these primitives are no-ops on macOS
//! measured the mapping's resident set, not the page cache; both observations hold, and that
//! note lives in the upstream-facing loader this harness does not touch.
//!
//! # What is measured, and the thresholds
//!
//! The read is 197 windows of the body file — every packed tensor the prefill binds,
//! `token_embd` excluded because the talker reads it row-wise on the host — timed through the
//! route with the request against the same windows mapped and copied without it. The upload
//! of the same bytes is timed through [`graphs::Compiled::new`] with the request only — the
//! route carries no switch to turn it off, so its baseline is the loader doc's figure rather
//! than a second arm here — and `UPLOAD_TAKES` times on a resident file, the warm path the
//! request must not move. The cold ceilings are **300 ms** for the read and **500 ms** for the
//! upload, the latter the median of three takes; the figures they are drawn from, and the
//! residency they were taken at, are in `loader::request_readahead`'s doc comment, which owns
//! them. Every figure here moves with the machine, so the load average is printed beside each
//! and a red take at high load is a take to re-run rather than a regression.
//!
//! The invalidation is this platform's — Linux's equivalent is
//! `posix_fadvise(POSIX_FADV_DONTNEED)` — which is why the harness is macOS-only. The request's
//! own Linux arm is not even compiled on this host and has never been run on a Linux one, so
//! every ceiling here is the macOS arm's.
//!
//! Needs the installed model directory and the device; a machine without either skips, and a
//! pair that is present but will not open fails. Run with `cargo test --release -p
//! mahbot-inference --features hand-run --test cold_read -- --ignored --nocapture`.

#![cfg(target_os = "macos")]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use mahbot_inference::graphs;
use mahbot_inference::loader::{Model, Role, Store};
use rlx_ir::Tick;
use rlx_runtime::Device;

mod support;

/// The prefill's prompt length. Any length compiles the same parameter set, and this harness
/// reads bytes rather than activations.
const PREFILL_ROWS: usize = 16;
/// The ceiling the cold read is accepted against, and the cold upload's.
const READ_CEILING_MS: f64 = 300.0;
const UPLOAD_CEILING_MS: f64 = 500.0;
/// Paired rounds of the two read arms, in the other arm's order every round; the warm-up they
/// would otherwise pay is the untimed `pin_arms` pass before them.
const ROUNDS: usize = 5;
/// Upload takes per arm, of which that arm's median is read: one take is a figure the machine's
/// load can carry to the ceiling on its own.
const UPLOAD_TAKES: usize = 3;

/// One tensor's byte range in the model file, as [`Model::with_mapped`] addresses it.
struct Extent {
    name: String,
    start: u64,
    len: usize,
}

/// The prefill's windows: every packed tensor of the body file except `token_embd`, which the
/// talker reads row-wise on the host and which therefore never reaches the route.
fn prefill_windows(model: &Model) -> Vec<Extent> {
    let data = model.file.data_offset();
    model
        .plans
        .iter()
        .filter(|p| p.store == Store::Packed && p.name != "token_embd.weight")
        .map(|p| Extent {
            name: p.name.clone(),
            start: data + model.file.get(&p.name).expect("a planned tensor").offset,
            len: model.raw_len(&p.name).expect("a planned tensor's bytes"),
        })
        .collect()
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
    fn reestablish(&mut self, extents: &[Extent]) -> (f64, f64) {
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
    fn resident(&mut self, extents: &[Extent]) -> f64 {
        let page = self.page;
        let mut resident = 0usize;
        let mut pages = 0usize;
        for e in extents {
            let lo = e.start as usize & !(page - 1);
            let hi = ((e.start as usize + e.len).div_ceil(page) * page).min(self.probe.len());
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
fn cold_start(cold: &mut Cold, extents: &[Extent], what: &str) -> f64 {
    let (before, after) = cold.reestablish(extents);
    assert_eq!(
        after, 0.0,
        "{what}: the invalidation left {after:.3} of the extents resident (was {before:.3}), so \
         this arm would not be cold"
    );
    before
}

/// One extent's window, mapped by hand: the baseline arm's addressing, which
/// [`Model::with_mapped`] must reproduce byte for byte.
fn window(file: &File, e: &Extent) -> memmap2::Mmap {
    // SAFETY: a read-only window of a model file, opened read-only and never written; the
    // tensor's bytes lie inside it.
    unsafe {
        memmap2::MmapOptions::new()
            .offset(e.start)
            .len(e.len)
            .map(file)
    }
    .expect("a window of the body file")
}

/// One arm over every extent: map the window, copy its bytes into one reused buffer — what
/// the device's staging write pays — and release the window. `request` picks the route under
/// test, or the same window mapped and copied without its request. The compile issues a
/// request per fused part as well as per window — 309 extents against the 197 walked here — and
/// a request is one `fcntl`, inside this arm's own spread.
fn read_arm(
    file: &File,
    model: &Arc<Model>,
    extents: &[Extent],
    request: bool,
    sink: &mut Vec<u8>,
) -> f64 {
    let t = Tick::now();
    for e in extents {
        if request {
            model
                .with_mapped(&e.name, |bytes| {
                    sink.clear();
                    sink.extend_from_slice(bytes);
                })
                .expect("a window of the body file");
        } else {
            sink.clear();
            sink.extend_from_slice(&window(file, e)[..]);
        }
    }
    Tick::now().elapsed_ns(t) as f64 / 1e6
}

/// Both arms must address the same bytes, so this pass pins the route to the hand-mapped
/// baseline extent by extent: a route change that moved a window shows up here rather than as
/// a plausible number, and the same bytes reaching the device is the property the route keeps.
/// It is also the arms' warm-up — the first window of a process pays the mapping and the
/// descriptor, which neither arm should be charged for — so it costs no read of its own.
fn pin_arms(file: &File, body: &Arc<Model>, extents: &[Extent], sink: &mut Vec<u8>) {
    for e in extents {
        sink.clear();
        sink.extend_from_slice(&window(file, e)[..]);
        let same = body
            .with_mapped(&e.name, |bytes| bytes == sink.as_slice())
            .expect("a window of the body file");
        assert!(
            same,
            "`{}` is a different byte range in the two arms",
            e.name
        );
    }
    println!(
        "the two arms address the same bytes over {} extents",
        extents.len()
    );
}

/// The arm that runs first, which swaps every round. `false` is the route without the request.
fn arm_order(round: usize) -> [bool; 2] {
    if round.is_multiple_of(2) {
        [false, true]
    } else {
        [true, false]
    }
}

/// The state an arm is measured in: the file's pages dropped from the page cache before every
/// arm, or resident with nothing invalidated.
#[derive(Clone, Copy)]
enum State {
    Cold,
    Warm,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            State::Cold => "cold",
            State::Warm => "warm",
        }
    }

    /// The state every arm and every take must start in, asserted: `Cold` invalidates and
    /// demands zero residency, `Warm` demands the file resident and invalidates nothing.
    fn ensure(self, cold: &mut Cold, extents: &[Extent], what: &str) {
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

/// [`ROUNDS`] paired takes over both arms, the order swapped every round so a within-round
/// drift lands on both rather than on the delta, answering each arm's median. `state` is what
/// every arm must start in: `Cold` invalidates and asserts zero residency, `Warm` asserts the
/// file is resident and invalidates nothing.
fn paired(
    state: State,
    cold: &mut Cold,
    body: &Arc<Model>,
    extents: &[Extent],
    sink: &mut Vec<u8>,
) -> (f64, f64) {
    let (mut plain, mut routed) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        let mut times = [0f64; 2];
        for request in arm_order(round) {
            state.ensure(cold, extents, state.label());
            times[usize::from(request)] = read_arm(&cold.file, body, extents, request, sink);
        }
        println!(
            "{} round {round}: without {:.1} ms, with {:.1} ms",
            state.label(),
            times[0],
            times[1]
        );
        plain.push(times[0]);
        routed.push(times[1]);
    }
    (support::median(&plain), support::median(&routed))
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

/// The figures, at zero residency before every arm.
#[test]
#[ignore = "measurement only: reads the model file cold; run by hand"]
fn the_prefills_cold_read_and_upload() {
    let _solo = support::solo("the_prefills_cold_read_and_upload");
    let Some(pair) = support::pair_of(Role::Synthesis) else {
        return;
    };
    let body = pair.body.clone();
    let extents = prefill_windows(&body);
    let bytes: usize = extents.iter().map(|e| e.len).sum();
    let mut cold = Cold::new(&body.path);
    let mut sink = Vec::new();
    println!(
        "prefill: {} windows, {:.1} MiB; load {:.2}",
        extents.len(),
        bytes as f64 / (1 << 20) as f64,
        load_average()
    );

    // The arms' warm-up and their pinning in one untimed pass: every window mapped by hand and
    // again through the route, the two compared byte for byte. Nothing below is charged for the
    // first window's mapping or the descriptor.
    pin_arms(&cold.file, &body, &extents, &mut sink);
    // The invalidation's own effect, printed where a reader can check it: the pass above read
    // every window, so all of them must now be resident, and the invalidation must drop them.
    let before = cold_start(&mut cold, &extents, "the first arm");
    println!(
        "invalidation: resident {before:.3} → 0.000 over {} extents",
        extents.len()
    );

    let (plain_median, routed_median) = paired(State::Cold, &mut cold, &body, &extents, &mut sink);
    let load = load_average();
    println!(
        "cold read: without {plain_median:.1} ms ({:.2} GB/s), with {routed_median:.1} ms \
         ({:.2} GB/s, delta {:.1} ms, load {load:.2})",
        bytes as f64 / (plain_median * 1e6),
        bytes as f64 / (routed_median * 1e6),
        routed_median - plain_median
    );
    assert!(
        routed_median <= READ_CEILING_MS,
        "the prefill's cold read took {routed_median:.1} ms at one-minute load {load:.2}, over \
         the {READ_CEILING_MS} ms ceiling"
    );

    // The warm path: the same arms on a resident file, which is where the request must cost
    // nothing. Nothing is invalidated here — the residency is asserted instead.
    let (warm_plain, warm_routed) = paired(State::Warm, &mut cold, &body, &extents, &mut sink);
    println!(
        "warm read: without {warm_plain:.1} ms, with {warm_routed:.1} ms (delta {:.1} ms, load {:.2})",
        warm_routed - warm_plain,
        load_average()
    );
    // One request per extent is one `fcntl` each: a couple of per cent of the read, against an
    // arm whose own spread is that wide. A tenth is the bound the move has to stay inside.
    assert!(
        warm_routed <= warm_plain * 1.10,
        "the request moved the warm read from {warm_plain:.1} ms to {warm_routed:.1} ms"
    );

    // The upload through the real compile: the same bytes, the device write included. Each take
    // compiles its own graph, and the device's shared weights are released first so that take
    // uploads rather than skips what an earlier one placed. The compile's byte count is this
    // harness's extent set plus the f32 parameters the other route carries, so a tensor that
    // moved out of either set shows up here.
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
    extents: &[Extent],
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
    extents: &[Extent],
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
    let compiled = graphs::Compiled::new(Device::Gpu, built).expect("the prefill graph compiles");
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
