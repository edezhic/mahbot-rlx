// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two guards every heavy run carries: a wall-clock stop and a resident-set
//! ceiling. A synthesized clip, a corpus file and a detection chunk check the
//! deadline, and a clip or a corpus file also checks the cheap process figure, so
//! a run that is out of time or out of memory stops at the next work item instead
//! of at the end. The system figures (swap growth and the machine's free
//! memory) cost a system call and a subprocess, so they belong to phase and file
//! boundaries — and to `Watch`, the guard thread, which is what makes the stop hard: it
//! ends the process at the deadline even when a single device call is blocked, after
//! running the caller's own stop action (the bench writes its report there).
//!
//! The swap arm reads this process's own swap-ins wherever a host reports them and the
//! machine's used swap where it does not; [`MAX_SWAP_GROWTH_MIB`] states which host produces
//! which, and why the per-process figure is the one a ceiling on a run wants.
//!
//! The resident-set figure comes from `ps`, so on a platform without it the
//! process arm reads zero and never trips. The system arms are armed only where the platform
//! has their figures, and an armed swap arm that cannot read its figure stops the run rather
//! than passing it: an arm that cannot report its own absence is not an arm, and a take it
//! silently skipped is a take recorded under a bound nobody applied. The free-memory arm is
//! best-effort, for the reason [`MemoryCeiling::check_system`] gives. On the platform this tree
//! runs on, all three read.
//!
//! The figures are inputs (`--timeout` / `MAHBOT_TIMEOUT_SECS`,
//! `--max-rss` / `MAHBOT_MAX_RSS_MIB`); the swap-growth ceiling has
//! `MAHBOT_STAND_MAX_SWAP_MIB`. The defaults are stated here so a run without them is
//! still bounded, and a measurement taken under a larger bound is `screened`: it produces
//! no recorded measurement.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::report;

/// Default wall-clock stop. With the corpus in place — the normal case, since it
/// is built once, deliberately — a measured eleven-voice run finishes well inside
/// this bound, most of its wall clock going to the two clip-stream phases and the
/// real-audio sweep.
///
/// Building the corpus is a separate operation with no wall-clock bound of its
/// own, and a measurement never renders: a cache short of a clip stops and names
/// the build instead of starting that synthesis. The bound is not raised to fit a
/// build into a measurement either — a run given a larger one produces no recorded
/// measurement.
pub const DEFAULT_TIMEOUT_SECS: u64 = 3_600;

/// Default resident-set ceiling. It leaves room for the mapped models and the one
/// synthesis pipeline a run holds beside them on an 18 GiB machine, while still
/// aborting before the machine starts swapping. The record's peak is the platform's
/// footprint, which is not this ceiling; a corpus build carries the same guards and
/// no wall clock.
pub const DEFAULT_MAX_RSS_MIB: u64 = 6_144;

/// Swap growth, in MiB, above which the run stops, unless
/// `MAHBOT_STAND_MAX_SWAP_MIB` names another ceiling — and a measurement taken
/// under a raised ceiling is `screened`, so it produces no recorded measurement.
///
/// The figure is this module's `swap_reading`, and which one a host produces is a property of
/// its kernel:
///
/// - **macOS**, a kernel that carries the `TASK_VM_INFO` tail: this process's own swap-ins,
///   from `ledger_swapins`. A byte count ([`report::swap_in_bytes`] states its width and
///   unit), so the growth is a difference of bytes and is compared against this ceiling in
///   bytes.
/// - **macOS**, an older kernel, and **Linux**: the machine's used swap — `vm.swapusage`
///   (`used =`) and `/proc/meminfo`'s `SwapTotal: - SwapFree:` respectively — read at the same
///   two moments, so the growth is the machine's.
/// - **Anywhere else**: neither, so the arm is not armed at all, since there is no reading to
///   be missing.
///
/// The per-process reading is preferred where it exists because the machine-wide figure moves
/// for reasons that have nothing to do with the run under measurement: another process's pages
/// going out to swap read as this run's growth, and another process's pages coming back in
/// shrink the growth this run appears to have spent. A ceiling on a figure the run did not
/// spend is not a ceiling on the run, and one a neighbour can lower is worse than that.
///
/// The arm is held to the **growth**, not to the absolute reading: a process that swapped
/// before this run started is not this run's doing, and the difference is what
/// `MemoryCeiling::spent` states on a row, so the figure a row prints is the figure this
/// ceiling held it to.
pub const MAX_SWAP_GROWTH_MIB: u64 = 1_024;

/// The swap-growth ceiling `env` names: [`MAX_SWAP_GROWTH_MIB`] unless
/// `MAHBOT_STAND_MAX_SWAP_MIB` holds a whole number, which a value that does not
/// parse is refused over rather than defaulted. The environment it reads is an
/// input, so a resolved run can be classified under the same closure it was
/// resolved from instead of the live process environment.
///
/// # Errors
/// The variable is set and is not a whole number of at least one.
pub fn swap_growth_ceiling(env: &impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<u64> {
    whole_number(
        env("MAHBOT_STAND_MAX_SWAP_MIB"),
        MAX_SWAP_GROWTH_MIB,
        "MAHBOT_STAND_MAX_SWAP_MIB",
    )
}

/// [`swap_growth_ceiling`] over this process's environment.
///
/// # Errors
/// The variable is set and is not a whole number of at least one.
pub fn max_swap_growth_mib() -> Result<u64> {
    swap_growth_ceiling(&|name| std::env::var_os(name))
}

/// A whole number of at least one from a bound's source, or `default` when the
/// source names nothing. A value that is set and does not parse, or that is zero, is
/// refused rather than defaulted: a mistyped bound must not let a run pass for one
/// held under the defaults. The one parser for the three bounds, in every entry
/// point that resolves them — the stand, the corpus run and the bench.
///
/// # Errors
/// The value is set and is not a whole number of at least one.
pub(crate) fn whole_number(
    value: Option<std::ffi::OsString>,
    default: u64,
    source: &str,
) -> Result<u64> {
    let Some(value) = value else {
        return Ok(default);
    };
    let text = value.to_string_lossy();
    let Ok(number) = text.trim().parse::<u64>() else {
        bail!("{source} `{text}` is not a whole number");
    };
    if number == 0 {
        bail!("{source} must be at least 1");
    }
    Ok(number)
}

/// [`whole_number`] over this process's environment.
fn whole_env(name: &str, default: u64) -> Result<u64> {
    whole_number(std::env::var_os(name), default, name)
}

/// The resident-set ceiling a run is under, in MiB — `MAHBOT_MAX_RSS_MIB`, or
/// [`DEFAULT_MAX_RSS_MIB`]. The bench resolves the same variable in its own input
/// table, through the same parser.
///
/// # Errors
/// The variable is set and is not a whole number of at least one.
pub fn max_rss_mib() -> Result<u64> {
    whole_env("MAHBOT_MAX_RSS_MIB", DEFAULT_MAX_RSS_MIB)
}

/// The wall-clock stop a run is under, in seconds — `MAHBOT_TIMEOUT_SECS`, or
/// [`DEFAULT_TIMEOUT_SECS`]. The bench resolves the same variable in its own input
/// table, through the same parser.
///
/// # Errors
/// The variable is set and is not a whole number of at least one.
pub fn timeout_secs() -> Result<u64> {
    whole_env("MAHBOT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS)
}

/// The three bounds a run is held to, resolved together so a mistyped variable is
/// refused where the run starts rather than from inside a measurement.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub timeout_secs: u64,
    pub max_rss_mib: u64,
    pub swap_growth_mib: u64,
}

impl Bounds {
    /// # Errors
    /// A bound variable is set and is not a whole number of at least one.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            timeout_secs: timeout_secs()?,
            max_rss_mib: max_rss_mib()?,
            swap_growth_mib: max_swap_growth_mib()?,
        })
    }
}

/// System-wide free memory, as a percentage, below which a run stops.
pub const MIN_FREE_PERCENT: u64 = 8;

/// Exit code of a run a guard stopped (the platform's `EX_TEMPFAIL`): non-zero,
/// and distinct from a fatal error so a caller can tell the two apart. Every
/// entry point uses it: the stand, the corpus run and the bench.
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
    /// A bound could not be applied because the figure it is held to could not be read.
    Unreadable {
        phase: &'static str,
        detail: &'static str,
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
            Self::Timeout { phase, .. }
            | Self::Memory { phase, .. }
            | Self::Unreadable { phase, .. } => phase,
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
            Self::Unreadable { phase, detail } => write!(
                f,
                "the `{phase}` phase cannot read {detail}, so the bound it is held to cannot be \
                 applied"
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
                    "the `{phase}` phase reached {} MiB of {detail}, over the {} MiB ceiling",
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
    /// inputs: the bound the console states and the bound the run is held to are
    /// one figure, not two reads of one variable.
    swap_mib: u64,
    /// The swap figure in use when the run started, with the one it was read from.
    /// `None` where the platform reported no figure: without a baseline the growth
    /// cannot be told from the machine's pre-existing swap. An armed ceiling refuses on
    /// a `None` rather than passing.
    swap_start: Option<SwapReading>,
    /// Whether the system arms — swap growth and the free-memory floor — are armed. They are
    /// served by the same platforms, so one field states both: `false` for a test's
    /// process-only ceiling, and for a platform with no system readings at all. An arm the
    /// platform cannot serve is not armed, while an armed arm whose figure then cannot be read
    /// refuses instead of leaving a take recorded under a bound nobody applied.
    system: bool,
}

impl MemoryCeiling {
    /// The ceiling at an explicit swap-growth bound: a caller resolves the run's
    /// bounds itself, so the number the run is held to is the number it read.
    #[must_use]
    pub fn with_swap_growth(mib: u64, swap_mib: u64) -> Self {
        // A platform with no system readings has no floor to hold the run to, and an arm it
        // cannot serve is not armed rather than refused — there is no reading to be missing.
        Self {
            rss_bytes: mib.saturating_mul(1024 * 1024),
            swap_mib,
            swap_start: swap_reading(),
            system: system_figures(),
        }
    }

    /// What the swap arm has spent since this ceiling took its baseline — the growth
    /// [`Self::check_system`] compares — and the figure it is a growth of. `None` where the
    /// platform reported no figure at one of the two reads. A row states this, so the figure a
    /// row carries is the one its own ceiling held it to.
    #[must_use]
    pub(crate) fn spent(&self) -> Option<(u64, SwapFigure)> {
        let start = self.swap_start?;
        Some((start.growth_to(swap_reading()?)?, start.figure))
    }

    /// Said where a run takes its ceiling, on a host whose figures the system arms are held to
    /// and that does not have them: the run is about to be recorded under bounds nobody applies.
    /// `None` when both arms are armed.
    #[must_use]
    pub fn unarmed(&self) -> Option<&'static str> {
        (!self.system).then_some(
            "this platform reports neither swap growth nor free memory — no system bound is \
             applied to this run",
        )
    }

    /// A ceiling with only the process arm, for a test that exercises a work
    /// item's accounting rather than a run: no system figure may turn it red.
    #[cfg(test)]
    pub(crate) fn process_only(mib: u64) -> Self {
        Self {
            rss_bytes: mib.saturating_mul(1024 * 1024),
            swap_mib: MAX_SWAP_GROWTH_MIB,
            swap_start: None,
            system: false,
        }
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

    /// The run's swap growth and the machine's free memory. A `task_info` call (or one file
    /// read) and one process spawn, so this is a phase / file boundary check, not a per-clip
    /// one. The swap-growth ceiling is the one the run was constructed with, so a reproduction
    /// under a raised bound does not change the default.
    ///
    /// An arm the platform cannot serve at all is not armed ([`MemoryCeiling::with_swap_growth`]);
    /// an armed swap arm whose figure then cannot be read is [`Abort::Unreadable`], which is a
    /// stop and not a pass. It is an `Abort` because the guard thread acts on those and only
    /// those — a plain error would be dropped there, and the stand's rows are stopped by nothing
    /// else. The free-memory arm is the exception, and stays best-effort: its probe is a
    /// subprocess that fails most easily under exactly the pressure the arm exists to catch, so
    /// a failed probe leaves the arm unapplied rather than refusing a run the swap arm still
    /// holds.
    ///
    /// # Errors
    /// Either figure passing its bound, or an armed swap arm's figure no longer being readable.
    pub fn check_system(&self, phase: &'static str) -> Result<()> {
        if !self.system {
            // Neither arm is armed: a platform with neither figure has no bound to hold the
            // run to, and there is no reading to be missing.
            return Ok(());
        }
        // Which figure a host reports is a property of its kernel, so the baseline and
        // this check read the same one. A baseline that was never there is a figure that
        // could not be read; the reason names no figure it did not measure.
        let unreadable = || {
            anyhow::Error::new(Abort::Unreadable {
                phase,
                detail: "the swap figure",
            })
        };
        let start = self.swap_start.ok_or_else(unreadable)?;
        let now = swap_reading().ok_or_else(unreadable)?;
        self.check_swap(start, now, phase)?;
        if let Some(percent) = free_percent()
            && percent < MIN_FREE_PERCENT
        {
            return Err(anyhow::Error::new(Abort::Memory {
                phase,
                detail: "pressured system memory (free percentage below the floor)",
                reading: percent,
                ceiling: MIN_FREE_PERCENT,
                unit: Unit::Percent,
            }));
        }
        Ok(())
    }

    /// The swap arm on one pair of readings: the ceiling is in MiB and the reading in bytes, and
    /// both sides of the comparison are bytes here, because the ledger is a byte count and not a
    /// page count. Taken as arguments rather than read here so a test can drive it with
    /// controlled figures — the live one is zero on nearly every row, which exercises nothing.
    fn check_swap(&self, start: SwapReading, now: SwapReading, phase: &'static str) -> Result<()> {
        // Both readings must be the same figure: a host whose per-process figure is missing on
        // one of the two calls falls back to the machine's, and a difference across them would
        // be the growth of two different quantities — a spurious stop in the adverse direction,
        // and a masked one in the other. A baseline that names a different one is a figure that
        // cannot be held to this ceiling, and the reason names none it did not measure.
        let Some(growth) = start.growth_to(now) else {
            return Err(anyhow::Error::new(Abort::Unreadable {
                phase,
                detail: "one swap figure (the host reported two)",
            }));
        };
        let ceiling = self.swap_mib.saturating_mul(1024 * 1024);
        if growth > ceiling {
            return Err(anyhow::Error::new(Abort::Memory {
                phase,
                detail: now.figure.name(),
                reading: growth,
                ceiling,
                unit: Unit::Bytes,
            }));
        }
        Ok(())
    }
}

/// Which of the two swap figures a reading came from. They are different figures — this
/// process's own swap-ins, or the machine's used swap — so a stop reason or a row must name
/// the one that was measured and never call one the other.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SwapFigure {
    /// This process's own swap-ins.
    Process,
    /// The machine's used swap.
    Machine,
}

impl SwapFigure {
    /// The figure as a memory stop names it: the quantity the row was held to, and the switch
    /// that sets that ceiling, so the stop says how to reproduce itself.
    fn name(self) -> &'static str {
        match self {
            Self::Process => {
                "swap this process faulted back in since the run started \
                 (MAHBOT_STAND_MAX_SWAP_MIB)"
            }
            Self::Machine => {
                "the machine's used swap above where the run started \
                 (MAHBOT_STAND_MAX_SWAP_MIB)"
            }
        }
    }

    /// The figure as a row's diagnostics line states it.
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Machine => "machine",
        }
    }
}

/// A swap reading with the figure it came from.
#[derive(Clone, Copy)]
struct SwapReading {
    bytes: u64,
    figure: SwapFigure,
}

impl SwapReading {
    /// The growth from `self` to `now`, when both are the same figure. A difference across
    /// figures is the growth of two quantities — a spurious stop in the adverse direction and
    /// a masked one in the other — so it is no reading at all.
    fn growth_to(self, now: Self) -> Option<u64> {
        (self.figure == now.figure).then(|| now.bytes.saturating_sub(self.bytes))
    }
}

/// The swap figure this host produces, with the name of it: the two are read here.
/// [`MAX_SWAP_GROWTH_MIB`] states which host produces which, and why the per-process one is
/// the figure a ceiling on a run wants.
fn swap_reading() -> Option<SwapReading> {
    report::swap_in_bytes()
        .map(|bytes| SwapReading {
            bytes,
            figure: SwapFigure::Process,
        })
        .or_else(|| {
            system_swap_used().map(|bytes| SwapReading {
                bytes,
                figure: SwapFigure::Machine,
            })
        })
}

/// Whether this platform has the two system readings the arms are held to. An arm a platform
/// cannot serve is not armed — it is not refused either, because there is no reading to be
/// missing — and a run on such a host says so on the console where it takes its ceiling
/// ([`MemoryCeiling::unarmed`]), since no column of a record states it.
fn system_figures() -> bool {
    cfg!(any(target_os = "macos", target_os = "linux"))
}

/// Used swap, in bytes, as the machine reports it. `None` where the platform has no reading;
/// on macOS [`report::swap_in_bytes`] is this process's own figure and this is the
/// machine-wide one it falls back to.
fn system_swap_used() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // Read only where the kernel's `task_info` tail is too old to carry `ledger_swapins`:
        // the per-process figure is the one a ceiling on this run means.
        let out = std::process::Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        used_swap_bytes(&text)
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

/// `used` from `vm.swapusage`'s `total = 6144.00M  used = 4566.06M  free = 1577.94M`, in
/// bytes. A figure the tool did not state is `None` rather than zero: zero would read as a
/// machine that is not swapping.
#[cfg(target_os = "macos")]
fn used_swap_bytes(text: &str) -> Option<u64> {
    let rest = text.split("used").nth(1)?.trim_start().strip_prefix('=')?;
    let megabytes: f64 = rest
        .split_whitespace()
        .next()?
        .trim_end_matches('M')
        .parse()
        .ok()?;
    (megabytes >= 0.0).then_some((megabytes * 1024.0 * 1024.0) as u64)
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
/// reads). One path for every entry point and for a row run on its own, because two
/// heavy runs on the same device is what the lock exists to prevent.
///
/// A caller that resolves its inputs from something other than the process
/// environment passes that lookup in, so a configuration resolved in a test cannot
/// name the developer's lock. A lookup that names no temp directory cannot resolve
/// one: the lock is refused, and nothing substitutes for it.
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
                "another heavy run is already running (lock: {}) — one at a time",
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
    /// `beat` prints the once-a-minute `still running` line. A child's output is
    /// drained and reprinted by its parent, which carries the beat, so only one
    /// process of a record run reports it.
    pub(crate) fn install(
        label: &'static str,
        deadline: Deadline,
        ceiling: MemoryCeiling,
        beat: bool,
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
                    if beat && tick.is_multiple_of(60) {
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

    /// A bound that is set and unparsable, or that is zero, is refused rather than
    /// defaulted: a mistyped variable must not let a run pass for one held under the
    /// defaults, and neither must a zero one.
    #[test]
    fn a_mistyped_or_zero_bound_is_refused_rather_than_defaulted() {
        let name = "MAHBOT_MAX_RSS_MIB";
        let error = whole_number(Some("6 144".into()), 1, name).expect_err("refused");
        assert!(
            error.to_string().contains("is not a whole number"),
            "{error}"
        );

        let error = whole_number(Some("0".into()), 1, name).expect_err("refused");
        assert!(error.to_string().contains("must be at least 1"), "{error}");

        assert_eq!(whole_number(None, 1, name).expect("the default"), 1);
        assert_eq!(
            whole_number(Some(" 2048 ".into()), 1, name).expect("parsed"),
            2048
        );
    }

    /// A ceiling under the process's own resident set trips on the process arm
    /// and reports bytes — not the percentage the system arm uses.
    #[test]
    fn a_tiny_ceiling_stops_on_the_resident_set() {
        let ceiling = MemoryCeiling::with_swap_growth(1, MAX_SWAP_GROWTH_MIB);
        let error = ceiling.check_process("real audio").expect_err("over");
        let abort = error.downcast::<Abort>().expect("an Abort");
        let text = abort.to_string();
        assert!(text.contains("resident memory"), "{text}");
        assert!(text.contains("MiB"), "{text}");
        assert_eq!(abort.phase(), "real audio");
    }

    /// The system arms are armed only where the platform has their figures, and an armed arm
    /// whose figure cannot be read stops the run as an `Abort` — the only kind of error the
    /// guard thread acts on, so a plain error there would leave the arm unapplied. An arm a
    /// platform cannot serve is not armed, so the same ceiling passes there because there is
    /// nothing to be missing.
    #[test]
    fn an_armed_arm_that_cannot_read_its_figure_stops_the_run() {
        let ceiling = MemoryCeiling::with_swap_growth(DEFAULT_MAX_RSS_MIB, MAX_SWAP_GROWTH_MIB);
        assert_eq!(
            ceiling.system,
            system_figures(),
            "the system arms are armed exactly where the platform has their figures"
        );
        assert_eq!(
            ceiling.unarmed().is_none(),
            system_figures(),
            "the notice that no system bound applies is said exactly where none does"
        );
        // No baseline is a figure that could not be read, and it is a stop rather than a pass.
        let mut unread = ceiling;
        unread.swap_start = None;
        if !system_figures() {
            // Nothing armed, so there is no figure to be missing and nothing to refuse.
            assert!(unread.check_system("recall").is_ok());
            return;
        }
        let error = unread.check_system("recall").expect_err("refused");
        match error.downcast::<Abort>() {
            Ok(Abort::Unreadable { phase, detail }) => {
                assert_eq!(phase, "recall");
                assert!(detail.contains("swap"), "{detail}");
            }
            Ok(other) => panic!("expected an unreadable figure, got {other:?}"),
            Err(other) => panic!("expected an unreadable figure, got {other:?}"),
        }
    }

    /// Two readings of different figures are not a growth: a host whose per-process figure is
    /// missing on one of the two calls falls back to the machine's, and subtracting one from the
    /// other would be a spurious stop of a take or a masked one. The mismatch is refused, and the
    /// reason names no figure it did not measure.
    #[test]
    fn a_baseline_from_another_figure_is_refused() {
        if !system_figures() {
            return;
        }
        let mut ceiling = MemoryCeiling::with_swap_growth(DEFAULT_MAX_RSS_MIB, MAX_SWAP_GROWTH_MIB);
        let Some(now) = swap_reading() else {
            return;
        };
        let (other, bytes) = match now.figure {
            SwapFigure::Process => (SwapFigure::Machine, now.bytes),
            SwapFigure::Machine => (SwapFigure::Process, now.bytes),
        };
        ceiling.swap_start = Some(SwapReading {
            // A baseline well below this reading, so only the figure can refuse it.
            bytes: bytes.saturating_sub(2 * ceiling.swap_mib * 1024 * 1024),
            figure: other,
        });
        let error = ceiling.check_system("recall").expect_err("refused");
        match error.downcast::<Abort>() {
            Ok(Abort::Unreadable { phase, detail }) => {
                assert_eq!(phase, "recall");
                assert!(detail.contains("one swap figure"), "{detail}");
            }
            Ok(other) => panic!("expected an unreadable figure, got {other:?}"),
            Err(other) => panic!("expected an unreadable figure, got {other:?}"),
        }
    }

    /// The machine-wide fallback reads the form `sysctl` actually prints, and a figure the
    /// tool did not state is missing rather than zero.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_swapusage_fallback_reads_what_sysctl_prints() {
        let printed = "total = 6144.00M  used = 4566.06M  free = 1577.94M  (encrypted)\n";
        assert_eq!(
            used_swap_bytes(printed),
            Some((4566.06_f64 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(used_swap_bytes("total = 0.00M  free = 0.00M"), None);
        assert_eq!(used_swap_bytes(""), None);
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
        let ceiling = MemoryCeiling::with_swap_growth(0, MAX_SWAP_GROWTH_MIB);
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
        if let Some(swap) = report::swap_in_bytes() {
            assert!(swap < 64 * 1024 * 1024 * 1024, "{swap} bytes of swap-ins");
        }
        if let Some(percent) = free_percent() {
            assert!(percent <= 100, "{percent}% free");
        }
    }

    /// The swap arm's arithmetic on controlled figures, driven through the comparison the run
    /// actually makes ([`MemoryCeiling::check_swap`]), because a live figure cannot be
    /// manufactured and every row's reads zero or a few MiB. The ceiling is in MiB and the
    /// reading in bytes, so both sides are bytes: 64 KiB of swap-ins is 64 KiB against a 1 MiB
    /// ceiling — read as a page count and multiplied by the page size, the same growth would
    /// have been a whole gigabyte — and a growth past 4 GiB is still the byte count it is, where
    /// a four-byte counter would have wrapped to its low half.
    #[test]
    fn the_swap_arm_is_the_growth_that_the_ceiling_is_expressed_in() {
        let at = |bytes: u64| SwapReading {
            bytes,
            figure: SwapFigure::Process,
        };
        // A 1 MiB ceiling, so the byte conversion is the ceiling's own.
        let ceiling = MemoryCeiling::with_swap_growth(0, 1);
        let check = |start: u64, now: u64| ceiling.check_swap(at(start), at(now), "recall");
        assert!(
            check(0, 64 * 1024).is_ok(),
            "64 KiB is under the 1 MiB ceiling"
        );

        // One byte over the ceiling: the stop states the growth, not the absolute reading.
        let start = 1 << 20;
        let error = check(start, start + 1024 * 1024 + 1).expect_err("over the ceiling");
        match error.downcast::<Abort>() {
            Ok(Abort::Memory {
                reading,
                ceiling,
                unit: Unit::Bytes,
                ..
            }) => {
                assert_eq!(reading, 1024 * 1024 + 1);
                assert_eq!(ceiling, 1024 * 1024, "the MiB bound, in the reading's unit");
            }
            Ok(other) => panic!("expected a memory stop, got {other:?}"),
            Err(other) => panic!("expected a memory stop, got {other:?}"),
        }

        // Past 4 GiB the figure is still the byte count it is: held to a bound above it, the
        // arm passes, where a four-byte field would have read its low half.
        let wide = MemoryCeiling::with_swap_growth(0, 8 * 1024);
        let past_four_gib = (4 << 30) + 4096;
        assert!(wide.check_swap(at(0), at(past_four_gib), "recall").is_ok());

        // Two readings of different figures are not a growth at all.
        let error = ceiling
            .check_swap(
                at(0),
                SwapReading {
                    bytes: 1 << 30,
                    figure: SwapFigure::Machine,
                },
                "recall",
            )
            .expect_err("two figures");
        match error.downcast::<Abort>() {
            Ok(Abort::Unreadable { detail, .. }) => assert!(detail.contains("one swap figure")),
            Ok(other) => panic!("expected an unreadable figure, got {other:?}"),
            Err(other) => panic!("expected an unreadable figure, got {other:?}"),
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
