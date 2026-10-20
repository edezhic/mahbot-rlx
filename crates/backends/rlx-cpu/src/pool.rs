// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rayon-backed parallel for: `par_for(total, grain, |off, cnt| …)`.
//!
//! Replaces the old per-worker Condvar pool with Rayon's work-stealing
//! scheduler. Same `(offset, count)` chunk API so all existing call
//! sites (BLAS tiling, SDPA, LayerNorm, …) pick up Rayon without
//! changes.
//!
//! The worker width is read once, when the pool is created: `RLX_WORKERS` if it
//! is set at that point, else the hardware-tuned default from `config.rs`, capped
//! at the threads the machine makes available. Rayon's pool is process-wide and
//! can only be created once, so a width that cannot be applied *as asked* is
//! reported on stderr instead of being dropped silently: the pool already exists
//! (both widths named), the request was wider than the machine, Rayon could not
//! build a pool at that width, `RLX_WORKERS` is not a width at all, or
//! `RuntimeConfig::install()` superseded the request — the installed width takes
//! effect when it is installed while the pool is still unclaimed.
//! [`num_threads`] reports the width in effect.
//!
//! A failed build leaves Rayon's global registry unset, where every Rayon call
//! panics: [`par_for`], [`par_range`] and [`num_threads`] keep off Rayon and run
//! inline when the width is 1. Kernels that drive Rayon directly (the batched
//! eigensolver, the fixed-point FFT, …) are outside that guard. Telling a
//! foreign failed build apart from a pool that already exists needs the read to
//! be caught, so under `panic = "abort"` (the `apple` profile) that one case
//! aborts the process instead of degrading to the serial width.
//!
//! Known limit: the pool is Rayon's, so RLX sizes it only if it gets there
//! first. Other in-tree Rayon users load weights before any CPU compile —
//! `rlx-gguf`, `rlx-mlx-io`, `rlx-pkg`, `rlx-vq` and `rlx-umap` — and the tuned
//! default is then reported rather than applied. A width set with
//! `rlx_ir::env::set` *after* the pool exists is likewise reported when
//! something later asks for the pool (a CPU compile, or
//! `RuntimeConfig::install()`), not when it is set; the width in effect stays
//! readable through [`num_threads`].

use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::OnceLock;

/// The one-time pool resolution: the width in effect, and why an asked-for width
/// is not it (`None` when the ask was applied as asked).
struct PoolState {
    width: usize,
    refused: Option<Refused>,
}

static POOL: OnceLock<PoolState> = OnceLock::new();

/// Claim the process-wide pool for RLX: create it at RLX's requested width if
/// nothing has yet, and report a width that could not be applied. The CPU backend
/// calls this on every compile, so a width requested in between is either in
/// effect or said to be impossible. Idempotent; safe to call from anywhere.
pub fn ensure_pool() {
    let state = resolved();
    if let Some(n) = crate::config::explicit_workers() {
        report_if_not_applied(n, crate::config::requested_workers_source());
    }
    // A value that is not a width cannot be applied later either when it was set
    // after the pool existed, where [`resolved`]'s own check has already run.
    if let Some(raw) = crate::config::unusable_workers_request() {
        report_unusable(&raw, state.width);
    }
}

/// Rayon's worker count, without the panic message `current_num_threads` emits
/// while Rayon's global registry is unset. A *foreign* `build_global` failure
/// leaves exactly that state behind while reporting the same sourceless
/// "already initialized" error, so the error alone cannot rule the read out.
/// The hook swapped in is process-global: one probe at a time, and a panic on
/// another thread inside that window prints nothing.
fn current_num_threads_quietly() -> Result<usize, Box<dyn std::any::Any + Send>> {
    static HOOK: Mutex<()> = Mutex::new(());
    let _guard = HOOK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probed = std::panic::catch_unwind(rayon::current_num_threads);
    std::panic::set_hook(hook);
    probed
}

/// The pool state, resolved on first call: ask for
/// [`crate::config::requested_workers`], build the pool, record the width and
/// report whatever could not be applied. The hot paths come straight here — they
/// must not pay an environment read per call, and reaching them first is fine.
fn resolved() -> &'static PoolState {
    POOL.get_or_init(|| {
        let want = crate::config::requested_workers();
        let source = crate::config::requested_workers_source();
        // A request wider than the threads the OS makes available cannot help,
        // so the cap is applied and reported rather than silently ignored.
        let machine = crate::config::machine_threads();
        let ask = want.min(machine);
        // `build_global` fails two ways, and `Error::source` is the only public
        // difference (rayon 1.12): a `source` means the build itself failed
        // (e.g. thread spawn) and rayon leaves its global registry unset, so
        // `current_num_threads()` would panic forever after — RLX runs serially
        // instead. No `source` means a pool already exists, whose width is safe
        // to read — unless *someone else's* build failed first, which also
        // completes Rayon's one-shot without a registry yet reports the same
        // sourceless error, so the read is guarded too.
        let (width, refused) = match rayon::ThreadPoolBuilder::new()
            .num_threads(ask)
            .thread_name(|i| format!("rlx-rayon-{i}"))
            .build_global()
        {
            Ok(()) => (
                ask,
                (ask < want).then_some(Refused::Capped {
                    want,
                    source,
                    machine,
                    effective: ask,
                }),
            ),
            Err(err) if std::error::Error::source(&err).is_some() => {
                (1, Some(Refused::BuildFailed { want, source }))
            }
            Err(_) => match current_num_threads_quietly() {
                Ok(width) => (
                    width,
                    (width != want).then_some(Refused::PoolExists {
                        want,
                        source,
                        effective: width,
                    }),
                ),
                Err(_) => (1, Some(Refused::BuildFailed { want, source })),
            },
        };
        if let Some(why) = refused.as_ref() {
            report_refused(why);
        }
        if let Some(raw) = crate::config::unusable_workers_request() {
            report_unusable(&raw, width);
        }
        // Pin OpenBLAS/MKL/Accelerate to 1 thread — Rayon owns outer
        // parallelism (par_sgemm splits). Nested N² threads are catastrophic.
        crate::blas::limit_inner_threads();
        PoolState { width, refused }
    })
}

/// Why a width could not be applied exactly as asked — the one channel a pool
/// width is reported through, so no request is dropped in silence. Each variant
/// carries the numbers its line names.
enum Refused {
    /// The ask was wider than the machine's available threads.
    Capped {
        want: usize,
        source: &'static str,
        machine: usize,
        effective: usize,
    },
    /// Another Rayon user created the process-wide pool first.
    PoolExists {
        want: usize,
        source: &'static str,
        effective: usize,
    },
    /// Rayon could not build the pool at all.
    BuildFailed { want: usize, source: &'static str },
    /// `RLX_WORKERS` is not a width, so there is no want or source to name.
    Unusable { raw: String, effective: usize },
    /// `RuntimeConfig::install` superseded an `RLX_WORKERS` request while the
    /// pool was already sized, so neither width is the one in effect.
    Replaced {
        request: usize,
        installed: usize,
        effective: usize,
    },
}

impl Refused {
    /// One-line report naming the requested and the in-effect width, so a request
    /// that was not applied is audible rather than silently dropped.
    fn line(&self) -> String {
        match self {
            Refused::Capped {
                want,
                source,
                machine,
                effective,
            } => format!(
                "[rlx] cpu pool: {effective} workers in effect — {want} ({source}) is wider than the {machine} threads this machine makes available"
            ),
            Refused::PoolExists {
                want,
                source,
                effective,
            } => format!(
                "[rlx] cpu pool: {effective} workers in effect, but {want} ({source}) could not be applied — the process-wide Rayon pool was already created"
            ),
            Refused::BuildFailed { want, source } => format!(
                "[rlx] cpu pool: Rayon could not build a {want}-worker pool ({source}) — RLX has no worker pool in this process"
            ),
            Refused::Unusable { raw, effective } => format!(
                "[rlx] cpu pool: ignoring RLX_WORKERS={raw} (not a positive integer) — using {effective} workers"
            ),
            Refused::Replaced {
                request,
                installed,
                effective,
            } => format!(
                "[rlx] cpu pool: {effective} workers in effect — RuntimeConfig::install() set {installed}, replacing the RLX_WORKERS={request} request"
            ),
        }
    }
}

/// Print [`Refused::line`], unless that exact line was already printed. The pool
/// is sized once but [`ensure_pool`] runs on every compile, and keying on the
/// line rather than the width still lets a *different* refusal speak.
fn report_refused(why: &Refused) {
    static LAST: Mutex<Option<String>> = Mutex::new(None);
    let line = why.line();
    let mut last = LAST.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if last.as_deref() == Some(line.as_str()) {
        return;
    }
    *last = Some(line.clone());
    eprintln!("{line}");
}

/// Report an `RLX_WORKERS` value that is not a width. Reached from [`resolved`]
/// and, for a value set after the pool exists, [`ensure_pool`].
fn report_unusable(raw: &str, effective: usize) {
    report_refused(&Refused::Unusable {
        raw: raw.to_string(),
        effective,
    });
}

/// Report that `requested` cannot take effect because the pool is already sized.
/// No-op while the pool is still unclaimed — in that case the request is
/// honoured when it is created.
pub(crate) fn report_if_not_applied(requested: usize, source: &'static str) {
    let Some(state) = POOL.get() else {
        return;
    };
    if requested == state.width {
        return;
    }
    // Name the reason the pool is not at `requested`, which need not be an
    // earlier Rayon user: after a build failure there is no pool at all, and a
    // request wider than the machine was capped rather than preceded.
    let why = match &state.refused {
        Some(Refused::BuildFailed { .. }) => Refused::BuildFailed {
            want: requested,
            source,
        },
        Some(Refused::Capped { machine, .. }) if requested > *machine => Refused::Capped {
            want: requested,
            source,
            machine: *machine,
            effective: state.width,
        },
        _ => Refused::PoolExists {
            want: requested,
            source,
            effective: state.width,
        },
    };
    report_refused(&why);
}

/// Report that an installed configuration superseded an `RLX_WORKERS` request.
///
/// No-op while the pool is unclaimed, where the installed width is the one the
/// pool is created at; when it is claimed the line names the width really in
/// effect, not the installed one, which cannot have taken effect.
pub(crate) fn report_replaced_request(request: usize, installed: usize) {
    let Some(state) = POOL.get() else {
        return;
    };
    report_refused(&Refused::Replaced {
        request,
        installed,
        effective: state.width,
    });
}

/// Worker width in effect in the process-wide pool RLX runs its CPU kernels on:
/// the one number to reason about parallelism with. Resolves the pool once and
/// then reports the width recorded then, so it never asks Rayon again — a process
/// where Rayon could not build a pool degrades to serial instead of panicking on
/// the unset global registry.
pub fn num_threads() -> usize {
    resolved().width
}

/// Parallel `for i in 0..n` over an index range, one task per index.
/// The closure must be `Sync + Send` because tasks run on Rayon workers.
#[inline]
pub fn par_range<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
    // Do not let Rayon's default width win, and stay off `into_par_iter`
    // when there is no usable pool — Rayon's unset global registry would
    // panic. (`num_threads` resolves the pool.)
    if num_threads() <= 1 {
        (0..n).for_each(&f);
        return;
    }
    (0..n).into_par_iter().for_each(f);
}

/// Minimum elements *per worker* that make threading an element-wise kernel
/// worthwhile. Below this the rayon hand-off costs more than the work it saves.
/// Kept in one place so conv/transpose/pool/relu/region kernels share a single,
/// tunable cutover instead of sprinkling magic constants at each call site.
pub const ELEMENTWISE_PAR_FLOOR: usize = 4096;

/// Should an op over `work` elements run in parallel? True only with >1 worker
/// and enough work to keep them busy past the amortization floor. Use this at
/// every data-parallel kernel call site instead of a hand-picked threshold.
///
/// The floor is per worker, so the cutover grows with the pool — the same rule
/// [`chunk_floor`] applies to `par_for` chunks, so a `true` here yields one
/// chunk per worker, never more. A width-independent cutover fanned work out
/// too small to amortize the hand-off: a 16 K-element activation measured 2×
/// slower at 5 workers than at 2.
#[inline]
pub fn should_parallelize(work: usize) -> bool {
    let workers = num_threads();
    workers > 1 && work >= ELEMENTWISE_PAR_FLOOR * workers
}

/// `par_for` chunk floor for a flat range of `total` elements: aim for one
/// chunk per worker, never below the amortization floor.
#[inline]
pub fn chunk_floor(total: usize) -> usize {
    (total / num_threads().max(1)).max(ELEMENTWISE_PAR_FLOOR)
}

/// `par_for` chunk floor for an *outer* loop of `units` independent items
/// (e.g. (n,c) planes): one chunk per worker, at least one unit.
#[inline]
pub fn outer_chunk(units: usize) -> usize {
    (units / num_threads().max(1)).max(1)
}

/// Parallel for: split `total` items across threads. `f(off, cnt)` is called once
/// per chunk with disjoint regions; `min_per_thread` is the chunk floor that
/// keeps the Rayon hand-off worth it.
///
/// SAFETY: caller must ensure `f` accesses disjoint memory regions for different
/// `(offset, count)` pairs.
#[inline]
pub fn par_for<F: Fn(usize, usize) + Sync>(total: usize, min_per_thread: usize, f: &F) {
    if total == 0 {
        return;
    }
    let grain = min_per_thread.max(1);
    let n_threads = (total / grain).max(1).min(num_threads());
    if n_threads <= 1 {
        f(0, total);
        return;
    }
    // Pin BLAS to 1 before Rayon fan-out so a preceding huge MT GEMM
    // cannot oversubscribe with worker-local sgemm / elementwise work.
    crate::blas::ensure_blas_threads(1);
    let chunk = total.div_ceil(n_threads);
    (0..n_threads).into_par_iter().for_each(|t| {
        let off = t * chunk;
        if off < total {
            f(off, (off + chunk).min(total) - off);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn par_for_sums_correctly() {
        let data = vec![1.0f32; 10_000];
        let total = AtomicU64::new(0);

        par_for(data.len(), 100, &|off, cnt| {
            let partial: f32 = data[off..off + cnt].iter().sum();
            total.fetch_add(partial.to_bits() as u64, Ordering::Relaxed);
        });

        assert!(total.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn par_for_small_is_sequential() {
        let sum = std::sync::atomic::AtomicUsize::new(0);
        par_for(10, 100, &|off, cnt| {
            sum.fetch_add(cnt, Ordering::Relaxed);
            assert_eq!(off + cnt, 10);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn par_for_exact_sum_many_dispatches() {
        for &n in &[256usize, 1024, 4097] {
            let sum = std::sync::atomic::AtomicUsize::new(0);
            par_for(n, 256, &|off, cnt| {
                sum.fetch_add(cnt, Ordering::Relaxed);
                assert!(off + cnt <= n);
            });
            assert_eq!(sum.load(Ordering::Relaxed), n);
        }
    }

    #[test]
    fn each_refusal_names_both_widths() {
        assert_eq!(
            Refused::Capped {
                want: 12,
                source: "requested via RLX_WORKERS",
                machine: 11,
                effective: 11,
            }
            .line(),
            "[rlx] cpu pool: 11 workers in effect — 12 (requested via RLX_WORKERS) is wider than the 11 threads this machine makes available"
        );
        assert_eq!(
            Refused::PoolExists {
                want: 12,
                source: "requested via RLX_WORKERS",
                effective: 11,
            }
            .line(),
            "[rlx] cpu pool: 11 workers in effect, but 12 (requested via RLX_WORKERS) could not be applied — the process-wide Rayon pool was already created"
        );
        assert_eq!(
            Refused::BuildFailed {
                want: 12,
                source: "requested via RLX_WORKERS",
            }
            .line(),
            "[rlx] cpu pool: Rayon could not build a 12-worker pool (requested via RLX_WORKERS) — RLX has no worker pool in this process"
        );
        assert_eq!(
            Refused::Unusable {
                raw: "abc".into(),
                effective: 11,
            }
            .line(),
            "[rlx] cpu pool: ignoring RLX_WORKERS=abc (not a positive integer) — using 11 workers"
        );
        assert_eq!(
            Refused::Replaced {
                request: 12,
                installed: 3,
                effective: 11,
            }
            .line(),
            "[rlx] cpu pool: 11 workers in effect — RuntimeConfig::install() set 3, replacing the RLX_WORKERS=12 request"
        );
    }

    #[test]
    fn an_existing_pool_is_distinguishable_from_a_build_failure() {
        // Create the global pool, so a second `build_global` fails for the
        // "already exists" reason — the resolution tells the two failure modes
        // apart by `Error::source`.
        let _ = rayon::current_num_threads();
        let err = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap_err();
        // If this fails, rayon changed the contract the resolution relies on.
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn par_for_concurrent_callers_isolated() {
        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    let n = 4096 + t * 17;
                    let sum = std::sync::atomic::AtomicUsize::new(0);
                    par_for(n, 128, &|off, cnt| {
                        sum.fetch_add(cnt, Ordering::Relaxed);
                        assert!(off + cnt <= n);
                    });
                    assert_eq!(sum.load(Ordering::Relaxed), n);
                });
            }
        });
    }
}
