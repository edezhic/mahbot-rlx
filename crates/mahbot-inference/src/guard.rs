// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two guards every heavy run carries: a wall-clock stop and a resident-set
//! ceiling. A synthesized clip, a corpus file and a detection chunk check the
//! deadline, and a clip or a corpus file also checks the cheap process figure, so
//! a run that is out of time or out of memory stops at the next work item instead
//! of at the end. The system figures (swap growth and free memory) cost two
//! subprocesses, so they belong to phase and file boundaries — and to [`Watch`],
//! the guard thread, which is what makes the stop hard: it ends the process at
//! the deadline even when a single device call is blocked, after running the
//! caller's own stop action (the bench writes its report there).
//!
//! The resident-set figure comes from `ps`, so on a platform without it the
//! process arm reads zero and never trips; the swap and free-memory arms report
//! their own absence. On the platform this stand runs on, all three read.
//!
//! The figures are inputs (`--timeout` / `MAHBOT_TIMEOUT_SECS`,
//! `--max-rss` / `MAHBOT_MAX_RSS_MIB`); the swap-growth ceiling has
//! `MAHBOT_STAND_MAX_SWAP_MIB` for a reproduction on a busy machine. The
//! defaults are stated here so a run without them is still bounded, and a
//! measurement taken under a larger bound is `screened`: it produces no recorded
//! measurement.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::report;

/// Default wall-clock stop. With the corpus in place — the normal case, since it
/// is built once, deliberately — a measured eleven-voice run finishes well inside
/// this bound, most of its wall clock going to the two clip-stream phases and the
/// real-audio sweep. `MEASUREMENTS.md`'s bench block is where those costs are
/// stated; this paragraph deliberately does not repeat them.
///
/// Building the corpus is a separate operation with no wall-clock bound of its
/// own, and a measurement never renders: a cache short of a clip stops and names
/// the build instead of starting that synthesis. The bound is not raised to fit a
/// build into a measurement either — a run given a larger one produces no recorded
/// measurement.
pub const DEFAULT_TIMEOUT_SECS: u64 = 3_600;

/// Default resident-set ceiling. It leaves room for the mapped models and the one
/// synthesis pipeline a run holds beside them on an 18 GiB machine, while still
/// aborting before the machine starts swapping. The record states the peak a
/// measured run reached; a corpus build carries the same guards and no wall clock.
pub const DEFAULT_MAX_RSS_MIB: u64 = 6_144;

/// Swap **growth since the run started** above which it stops, unless
/// `MAHBOT_STAND_MAX_SWAP_MIB` names another ceiling — and a measurement taken
/// under a raised ceiling is `screened`, so it produces no recorded measurement. GPU
/// allocations on a unified-memory machine are not visible in the process's
/// resident set, so this is the figure that catches the pressure a sweep creates:
/// the run that stopped working held 6.9 GiB of an 8.2 GiB swap file while its own
/// resident set read under 700 MiB. Growth rather than the absolute figure, because
/// a machine that has swapped before this run started is not this run's doing.
pub const MAX_SWAP_GROWTH_MIB: u64 = 1_024;

/// The swap-growth ceiling `env` names: [`MAX_SWAP_GROWTH_MIB`] unless
/// `MAHBOT_STAND_MAX_SWAP_MIB` holds a whole number. The environment it reads is
/// an input, so a resolved run can be classified under the same closure it was
/// resolved from instead of the live process environment.
#[must_use]
pub fn swap_growth_ceiling(env: &impl Fn(&str) -> Option<std::ffi::OsString>) -> u64 {
    env("MAHBOT_STAND_MAX_SWAP_MIB")
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(MAX_SWAP_GROWTH_MIB)
}

/// [`swap_growth_ceiling`] over this process's environment — the bound
/// [`MemoryCeiling::check_system`] enforces.
#[must_use]
pub fn max_swap_growth_mib() -> u64 {
    swap_growth_ceiling(&|name| std::env::var_os(name))
}

/// The resident-set ceiling a run is under, in MiB — `MAHBOT_MAX_RSS_MIB`, or
/// [`DEFAULT_MAX_RSS_MIB`]. One parse, one owner: the enforced ceiling and every
/// figure or refusal reported against it read this accessor, so a run cannot be
/// described under a bound it was not held to.
#[must_use]
pub fn max_rss_mib() -> u64 {
    std::env::var("MAHBOT_MAX_RSS_MIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_RSS_MIB)
}

/// [`max_rss_mib`] in bytes, for the library, which has to refuse an input the run
/// could not hold: the record and that refusal read the same accessor, so the
/// figure a cell says it exceeded is the ceiling the run was actually under.
#[must_use]
pub fn max_rss_bytes() -> u64 {
    max_rss_mib().saturating_mul(1024 * 1024)
}

/// System-wide free memory, as a percentage, below which a run stops.
pub const MIN_FREE_PERCENT: u64 = 8;

/// Exit code of a run a guard stopped (the platform's `EX_TEMPFAIL`): non-zero,
/// and distinct from a fatal error so a caller can tell the two apart. Both
/// entry points use it — the bench and the stand.
pub const EXIT_STOPPED: i32 = 75;

/// Why a run stopped early. It is carried inside an `anyhow::Error` so every
/// work-item boundary can use `?`; the top of the run downcasts it to decide
/// between a partial report and a fatal exit.
#[derive(Debug)]
pub enum Abort {
    /// The wall-clock stop expired.
    Timeout { phase: &'static str, secs: u64 },
    /// A memory figure passed its ceiling.
    Memory {
        phase: &'static str,
        detail: &'static str,
        reading: u64,
        ceiling: u64,
        unit: Unit,
    },
}

/// The unit a memory figure is in — a byte count or a percentage. The two are
/// never formatted alike, so a stop reason cannot read "0 MiB of pressured
/// system memory".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Bytes,
    Percent,
}

impl Abort {
    /// The phase that was running when the run stopped.
    #[must_use]
    pub fn phase(&self) -> &'static str {
        match self {
            Self::Timeout { phase, .. } | Self::Memory { phase, .. } => phase,
        }
    }
}

impl std::fmt::Display for Abort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout { phase, secs } => write!(
                f,
                "the `{phase}` phase exceeded the {secs} s wall-clock stop"
            ),
            Self::Memory {
                phase,
                detail,
                reading,
                ceiling,
                unit,
            } => match unit {
                Unit::Bytes => write!(
                    f,
                    "the `{phase}` phase reached {} MiB of {detail}, over the {} MiB ceiling — \
                     stopping before the machine swaps",
                    reading / (1024 * 1024),
                    ceiling / (1024 * 1024)
                ),
                Unit::Percent => write!(
                    f,
                    "the `{phase}` phase found {reading}% of {detail}, under the {ceiling}% floor — \
                     stopping before the machine swaps"
                ),
            },
        }
    }
}

impl std::error::Error for Abort {}

/// A wall-clock stop shared by every phase.
#[derive(Clone, Copy)]
pub struct Deadline {
    /// `None` for the corpus build, which has no bound to expire.
    end: Option<Instant>,
    secs: u64,
}

impl Deadline {
    #[must_use]
    pub fn new(secs: u64) -> Self {
        Self {
            end: Some(Instant::now() + Duration::from_secs(secs)),
            secs,
        }
    }

    /// A deadline that never expires: the corpus build is a deliberate operation
    /// with no wall-clock bound — its safety guards are the memory ones. `secs`
    /// reads as zero because there is no bound to name, and no stop can be
    /// produced to print one.
    #[must_use]
    pub fn unbounded() -> Self {
        Self { end: None, secs: 0 }
    }

    #[must_use]
    pub fn secs(&self) -> u64 {
        self.secs
    }

    #[must_use]
    pub fn expired(&self) -> bool {
        self.end.is_some_and(|end| Instant::now() >= end)
    }

    /// The error a caller at a work-item boundary returns when the stop expired.
    pub fn check(&self, phase: &'static str) -> Result<()> {
        if self.expired() {
            return Err(anyhow::Error::new(Abort::Timeout {
                phase,
                secs: self.secs,
            }));
        }
        Ok(())
    }
}

/// The memory guard: the run's own resident set, and the two system figures
/// that catch unified-memory allocations the resident set does not show.
#[derive(Clone, Copy)]
pub struct MemoryCeiling {
    rss_bytes: u64,
    /// The swap-growth ceiling, in MiB, resolved with the rest of the run's
    /// inputs: the bound the record states and the bound the run is held to are
    /// one figure, not two reads of one variable.
    swap_mib: u64,
    /// The free-memory floor; `None` leaves that arm inactive. Only the feed
    /// tests do, because they measure accounting, not a run, and must not fail
    /// because the host is busy.
    free_floor: Option<u64>,
    /// Swap in use when the run started. `None` where the platform reported no
    /// figure: without a baseline the growth cannot be told from the machine's
    /// pre-existing swap, so the swap arm stays inactive rather than tripping
    /// on it.
    swap_start: Option<u64>,
}

impl MemoryCeiling {
    /// The ceiling at the default swap-growth bound of this process's environment.
    #[must_use]
    pub fn new(mib: u64) -> Self {
        Self::with_swap_growth(mib, max_swap_growth_mib())
    }

    /// The ceiling at an explicit swap-growth bound, for a caller that resolved
    /// that input with the rest of its own.
    #[must_use]
    pub fn with_swap_growth(mib: u64, swap_mib: u64) -> Self {
        let swap_start = swap_used_bytes();
        if swap_start.is_none() {
            eprintln!(
                "[wake-word] the platform reported no swap figure — the swap-growth guard is \
                 inactive; the resident-set ceiling still applies"
            );
        }
        Self {
            rss_bytes: mib.saturating_mul(1024 * 1024),
            swap_mib,
            free_floor: Some(MIN_FREE_PERCENT),
            swap_start,
        }
    }

    /// A ceiling with only the process arm, for a test that exercises a work
    /// item's accounting rather than a run: no system figure may turn it red.
    #[cfg(test)]
    pub(crate) fn process_only(mib: u64) -> Self {
        Self {
            rss_bytes: mib.saturating_mul(1024 * 1024),
            swap_mib: MAX_SWAP_GROWTH_MIB,
            free_floor: None,
            swap_start: None,
        }
    }

    /// The ceiling's own figure, in bytes.
    #[must_use]
    pub fn rss_bytes(&self) -> u64 {
        self.rss_bytes
    }

    /// The current resident set, in bytes (`None` where the platform cannot
    /// report it, in which case this ceiling cannot trip).
    #[must_use]
    pub fn resident(&self) -> Option<u64> {
        report::resident_rss()
    }

    /// Both checks. Use at a phase or file boundary.
    pub fn check(&self, phase: &'static str) -> Result<()> {
        self.check_process(phase)?;
        self.check_system(phase)
    }

    /// The process's own resident set. Cheap enough for every work item.
    pub fn check_process(&self, phase: &'static str) -> Result<()> {
        let Some(rss) = self.resident() else {
            return Ok(());
        };
        if rss > self.rss_bytes {
            return Err(anyhow::Error::new(Abort::Memory {
                phase,
                detail: "resident memory",
                reading: rss,
                ceiling: self.rss_bytes,
                unit: Unit::Bytes,
            }));
        }
        Ok(())
    }

    /// Swap in use and free system memory. Two process spawns on macOS, so this
    /// is a phase / file boundary check, not a per-clip one. The swap-growth
    /// ceiling is the one the run was constructed with, so a reproduction on a
    /// busy machine can raise the bound without changing the default.
    pub fn check_system(&self, phase: &'static str) -> Result<()> {
        if let Some(start) = self.swap_start
            && let Some(now) = swap_used_bytes()
        {
            let growth = now.saturating_sub(start);
            let ceiling = self.swap_mib.saturating_mul(1024 * 1024);
            if growth > ceiling {
                return Err(anyhow::Error::new(Abort::Memory {
                    phase,
                    detail: "swap growth since the run started (override MAHBOT_STAND_MAX_SWAP_MIB)",
                    reading: growth,
                    ceiling,
                    unit: Unit::Bytes,
                }));
            }
        }
        if let Some(floor) = self.free_floor
            && let Some(percent) = free_percent()
            && percent < floor
        {
            return Err(anyhow::Error::new(Abort::Memory {
                phase,
                detail: "pressured system memory (free percentage below the floor)",
                reading: percent,
                ceiling: floor,
                unit: Unit::Percent,
            }));
        }
        Ok(())
    }
}

/// Swap in use, in bytes. `None` where the platform has no reading.
#[must_use]
pub fn swap_used_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        // `total = 8192.00M  used = 6904.94M  free = 1287.06M`
        let used = text.split("used =").nth(1)?.split_whitespace().next()?;
        let (number, unit) = used.split_at(used.len().checked_sub(1)?);
        let value: f64 = number.parse().ok()?;
        let scale = match unit {
            "G" => 1024.0 * 1024.0 * 1024.0,
            "M" => 1024.0 * 1024.0,
            "K" => 1024.0,
            _ => 1.0,
        };
        Some((value * scale) as u64)
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| -> Option<u64> {
            text.lines()
                .find(|line| line.starts_with(name))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u64>().ok())
                .map(|kb| kb * 1024)
        };
        let total = field("SwapTotal:")?;
        let free = field("SwapFree:")?;
        Some(total.saturating_sub(free))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// System-wide free memory as a percentage. `None` where the platform has no
/// reading.
///
/// `vm_stat`'s `Pages free` is not usable on macOS (it sits near zero while a
/// third of the machine is reclaimable), so this reads the same pressure figure
/// the operator sees.
#[must_use]
pub fn free_percent() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("memory_pressure")
            .output()
            .ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        text.lines()
            .find(|line| line.contains("memory free percentage"))?
            .rsplit(':')
            .next()?
            .trim()
            .trim_end_matches('%')
            .parse::<u64>()
            .ok()
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| -> Option<u64> {
            text.lines()
                .find(|line| line.starts_with(name))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u64>().ok())
        };
        let total = field("MemTotal:")?;
        let available = field("MemAvailable:")?;
        Some(available.saturating_mul(100) / total.max(1))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// The one lock every heavy run of this crate takes, resolved through `env`:
/// `MAHBOT_LOCK` when it names a non-empty path, else `mahbot-heavy-run.lock` in the
/// temp directory (`TMPDIR`, then `TMP`, `TEMP` — the order `std::env::temp_dir`
/// reads). One path for both entry points and for a cell run on its own, because
/// two heavy runs on the same device is what the lock exists to prevent.
///
/// A caller that resolves its inputs from something other than the process
/// environment passes that lookup in, so a configuration resolved in a test cannot
/// name the developer's lock. A lookup that names no temp directory either cannot
/// resolve one: the lock is refused, as the corpus cache's path is.
///
/// # Errors
/// Neither `MAHBOT_LOCK` nor `TMPDIR`, `TMP`, `TEMP` is set, so no shared path exists.
pub fn lock_path(env: &impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<std::path::PathBuf> {
    if let Some(path) = env("MAHBOT_LOCK").filter(|value| !value.is_empty()) {
        return Ok(std::path::PathBuf::from(path));
    }
    match ["TMPDIR", "TMP", "TEMP"]
        .into_iter()
        .find_map(|name| env(name).filter(|value| !value.is_empty()))
    {
        Some(dir) => Ok(std::path::PathBuf::from(dir).join("mahbot-heavy-run.lock")),
        None => bail!(
            "none of MAHBOT_LOCK, TMPDIR, TMP or TEMP is set — there is no shared lock path, \
             and two heavy runs on one device is what the lock exists to prevent"
        ),
    }
}

/// An exclusive advisory lock on the path [`lock_path`] resolves — `MAHBOT_LOCK`
/// when set, else one file in the temp directory, never a file beside the report
/// and never "no lock". A second launch refuses instead of competing for the same
/// GPU and the same memory.
///
/// The kernel releases the lock when the descriptor closes, so a killed run
/// leaves no stale lock to clear by hand.
#[derive(Debug)]
pub struct RunLock {
    _file: std::fs::File,
}

impl RunLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("cannot create the lock's directory {}", parent.display())
            })?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("cannot open the run lock {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(std::fs::TryLockError::WouldBlock) => bail!(
                "another wake-word bench is already running (lock: {}) — one heavy run at a time",
                path.display()
            ),
            Err(std::fs::TryLockError::Error(error)) => {
                Err(error).with_context(|| format!("cannot lock the run file {}", path.display()))
            }
        }
    }
}

/// The guard thread every heavy entry point installs: it polls the wall clock
/// every second and both memory figures every five, so a stop fires even while
/// the measuring thread is inside one blocked device call. `phase` is read fresh
/// on every tick; `on_stop` runs in this thread with the reason and is expected
/// to leave the process (the bench writes its report there, the stand prints and
/// exits) — either way the watch ends after it returns.
pub(crate) struct Watch {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watch {
    pub(crate) fn install(
        label: &'static str,
        deadline: Deadline,
        ceiling: MemoryCeiling,
        phase: impl Fn() -> &'static str + Send + 'static,
        mut on_stop: impl FnMut(Abort) + Send + 'static,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name(format!("{label}-guard"))
            .spawn(move || {
                let mut tick = 0u64;
                while !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_secs(1));
                    tick += 1;
                    if tick.is_multiple_of(60) {
                        eprintln!(
                            "[heartbeat] {label} still running — elapsed {}m{:02}s",
                            tick / 60,
                            tick % 60
                        );
                    }
                    let phase = phase();
                    let abort = if deadline.expired() {
                        Some(Abort::Timeout {
                            phase,
                            secs: deadline.secs(),
                        })
                    } else if tick.is_multiple_of(5) {
                        ceiling
                            .check_process(phase)
                            .err()
                            .or_else(|| ceiling.check_system(phase).err())
                            .and_then(|error| error.downcast::<Abort>().ok())
                    } else {
                        None
                    };
                    if let Some(abort) = abort {
                        on_stop(abort);
                        return;
                    }
                }
            })
            .context("cannot start the guard thread")?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    /// Tell the thread the run is over and wait for it.
    pub(crate) fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        // The thread wakes within a second and exits; joining here would delay an
        // error path for no benefit.
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ceiling under the process's own resident set trips on the process arm
    /// and reports bytes — not the percentage the system arm uses.
    #[test]
    fn a_tiny_ceiling_stops_on_the_resident_set() {
        let ceiling = MemoryCeiling::new(1);
        let error = ceiling.check_process("real audio").expect_err("over");
        let abort = error.downcast::<Abort>().expect("an Abort");
        let text = abort.to_string();
        assert!(text.contains("resident memory"), "{text}");
        assert!(text.contains("MiB"), "{text}");
        assert_eq!(abort.phase(), "real audio");
    }

    #[test]
    fn a_zero_or_elapsed_deadline_reports_the_phase() {
        let expired = Deadline::new(0);
        assert!(expired.expired());
        let error = expired.check("recall").expect_err("expired");
        match error.downcast::<Abort>() {
            Ok(Abort::Timeout { phase, secs }) => {
                assert_eq!(phase, "recall");
                assert_eq!(secs, 0);
            }
            Ok(other) => panic!("expected a timeout, got {other:?}"),
            Err(other) => panic!("expected a timeout, got {other:?}"),
        }
        let live = Deadline::new(60);
        assert!(live.check("recall").is_ok());
    }

    #[test]
    fn the_memory_ceiling_converts_and_trips_on_a_low_figure() {
        // A zero resident-set ceiling is below any live process's reading, so
        // it trips wherever the platform reports one.
        let ceiling = MemoryCeiling::new(0);
        if let Some(rss) = ceiling.resident() {
            assert!(rss > 0, "a live process reads more than zero bytes");
            let error = ceiling
                .check_process("real audio")
                .expect_err("over the ceiling");
            assert!(matches!(
                error.downcast::<Abort>(),
                Ok(Abort::Memory { phase, detail, .. })
                    if phase == "real audio" && detail == "resident memory"
            ));
        }
        // The system figures parse into bytes where the platform has them.
        if let Some(swap) = swap_used_bytes() {
            assert!(swap < 64 * 1024 * 1024 * 1024, "{swap} bytes of swap");
        }
        if let Some(percent) = free_percent() {
            assert!(percent <= 100, "{percent}% free");
        }
    }

    #[test]
    fn a_percentage_figure_formats_as_a_percentage() {
        let abort = Abort::Memory {
            phase: "real audio",
            detail: "pressured system memory",
            reading: 3,
            ceiling: MIN_FREE_PERCENT,
            unit: Unit::Percent,
        };
        let text = format!("{abort}");
        assert!(text.contains("3% of pressured system memory"), "{text}");
        assert!(text.contains("under the 8% floor"), "{text}");
        assert!(!text.contains("MiB"), "{text}");
    }

    #[test]
    fn the_lock_refuses_a_second_holder() {
        let dir = std::env::temp_dir().join(format!(
            "mahbot-wake-lock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("run.lock");
        let first = RunLock::acquire(&path).expect("first lock");
        let error = RunLock::acquire(&path).expect_err("second lock refuses");
        assert!(
            format!("{error:#}").contains("already running"),
            "{error:#}"
        );
        drop(first);
        // Release is the kernel's on close; a child forked around the drop can
        // hold the inherited description for a moment, so the check tolerates
        // that window rather than depending on scheduling.
        let mut relocked = false;
        for _ in 0..20 {
            if RunLock::acquire(&path).is_ok() {
                relocked = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            relocked,
            "the lock was not released when the holder dropped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The shared lock: an explicit path wins, otherwise the temp directory the
    /// platform names — and a lookup that names none refuses rather than falling
    /// back to the process environment.
    #[test]
    fn the_lock_path_wins_in_order_and_refuses_an_empty_lookup() {
        let lookup = |name: &str| match name {
            "MAHBOT_LOCK" => Some("/tmp/explicit.lock".into()),
            "TMPDIR" => Some("/tmp/dir".into()),
            "TMP" => Some("/var/tmp".into()),
            _ => None,
        };
        assert_eq!(
            lock_path(&lookup).unwrap(),
            std::path::PathBuf::from("/tmp/explicit.lock")
        );
        let implicit = |name: &str| (name != "MAHBOT_LOCK").then(|| lookup(name)).flatten();
        assert_eq!(
            lock_path(&implicit).unwrap(),
            std::path::PathBuf::from("/tmp/dir/mahbot-heavy-run.lock")
        );
        let only_tmp = |name: &str| (name == "TMP").then(|| lookup(name)).flatten();
        assert_eq!(
            lock_path(&only_tmp).unwrap(),
            std::path::PathBuf::from("/var/tmp/mahbot-heavy-run.lock")
        );
        let empty = lock_path(&|_: &str| None).expect_err("no path named");
        assert!(empty.to_string().contains("TMPDIR"), "{empty}");
    }
}
