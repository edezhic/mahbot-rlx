// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plumbing the two row-walking record runs share — the stand and the corpus
//! run: one run at a time, the wall-clock stop and memory ceiling, one row measured
//! in a process of its own, and the one line a row prints for its parent. The
//! wake-word bench measures in one process and shares only [`bound_reasons`] with
//! this module; its report, guards and record are its own.
//!
//! The parent holds the lock; a child carries the limits itself, so a parent that
//! dies leaves a bounded child behind rather than a multi-gigabyte process with
//! nothing over it.

use std::io::Read;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context as _, Result, bail};
use serde_json::{Map, Value};

use crate::guard::{Bounds, Deadline, EXIT_STOPPED, MemoryCeiling, RunLock, Watch, lock_path};
use crate::report::Memory;

/// The row a parent is waiting on, while it runs. A parent that leaves with its row
/// still measuring leaves a multi-gigabyte process on the device with the lock
/// released and nothing over it, so a stop path this program runs ends it first — the
/// guard thread's stop and the parent's own wait on every target. A parent killed by
/// a signal it does not handle ends nothing, on any target; the row then stops at its
/// own deadline, which it carries itself.
///
/// The child is published under this lock as soon as it exists and cleared in the
/// critical section that reaps it, so a stop can neither miss a running row nor
/// signal a pid that has been reaped and reused.
static ROW: Mutex<Option<std::process::Child>> = Mutex::new(None);

/// The slot a row's child is published in. The guard is taken across the wait that
/// reaps it, which is why a stop thread is deferred rather than lost.
fn row_slot() -> MutexGuard<'static, Option<std::process::Child>> {
    ROW.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// End the row this parent is waiting on, if one is still running. `Child::kill` is
/// the portable kill, so this ends the row on every target; it is not reaped here,
/// because the process that stops a row is leaving.
fn stop_row() {
    if let Some(row) = row_slot().as_mut() {
        let _ = row.kill();
    }
}

/// A row-walking record run's guard: the stop, the ceiling, the bounds this process
/// resolved, and the lock the parent took. Dropped at the end of the run.
pub(crate) struct Heavy {
    deadline: Deadline,
    bounds: Bounds,
    // Dropped with the guard: the thread wakes within a second and exits.
    _watch: Watch,
    _lock: Option<RunLock>,
}

impl Heavy {
    /// The parent of a row-walking record run: it takes the one shared lock and
    /// watches its own clock and memory.
    pub(crate) fn parent(label: &'static str, bounds: Bounds) -> Self {
        let lock = match lock_path(&|name: &str| std::env::var_os(name))
            .and_then(|path| RunLock::acquire(&path))
        {
            Ok(lock) => lock,
            Err(error) => {
                eprintln!("{label}: {error:#}");
                std::process::exit(1);
            }
        };
        Self::bounded(label, bounds, Some(lock))
    }

    /// A child the parent spawned: the parent holds the lock, so the child takes
    /// none — but it resolves the same bounds and carries the same stop and ceiling.
    pub(crate) fn child(label: &'static str) -> Self {
        Self::bounded(label, bounds(label), None)
    }

    /// The bounds this process is held to: the numbers its guard enforces, which are
    /// the numbers its rows measure under — one resolution, so a run cannot enforce
    /// one ceiling and refuse an input against another.
    pub(crate) fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// The lock holder prints the once-a-minute liveness line; a child's output is
    /// drained and reprinted by that parent, so it carries none.
    fn bounded(label: &'static str, bounds: Bounds, lock: Option<RunLock>) -> Self {
        let deadline = Deadline::new(bounds.timeout_secs);
        let ceiling = MemoryCeiling::with_swap_growth(bounds.max_rss_mib, bounds.swap_growth_mib);
        let watch = match Watch::install(
            label,
            deadline,
            ceiling,
            lock.is_some(),
            move || label,
            move |abort| {
                stop_row();
                eprintln!("{label}: {abort} — stopping");
                std::process::exit(EXIT_STOPPED);
            },
        ) {
            Ok(watch) => watch,
            Err(error) => {
                eprintln!("{label}: cannot start the guard thread: {error:#}");
                std::process::exit(1);
            }
        };
        Self {
            deadline,
            bounds,
            _watch: watch,
            _lock: lock,
        }
    }

    pub(crate) fn deadline(&self) -> Deadline {
        self.deadline
    }
}

/// A row-walking record run's child, entered through `marker`: it measures the one row
/// its argument names and prints the reading its parent reads back. An argument that
/// is not exactly one row, or a row that fails, ends the child with its own code.
pub(crate) fn child(
    label: &'static str,
    named: &[(usize, &'static str)],
    marker: &'static str,
    measure: impl Fn(usize, Bounds) -> Result<Reading>,
) {
    let [(index, name)] = named else {
        eprintln!(
            "{label}: {marker} takes one row index and {} are named — run the rows one at a \
             time, or name them without the marker",
            named.len()
        );
        std::process::exit(2);
    };
    let heavy = Heavy::child(label);
    match measure(*index, heavy.bounds()) {
        Ok(reading) => println!("{}", reading.line()),
        Err(error) => {
            println!("row {name} failed={error:#}");
            eprintln!("{label}: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Run this binary's child for one row and return its stdout: `index` names the
/// row, `marker` marks the child, so a re-exec of the same binary measures exactly
/// that row. A child that fails ends this process with its own code; a parent whose
/// stop expires kills it and exits [`EXIT_STOPPED`].
pub(crate) fn row_process(
    exe: &Path,
    index: usize,
    marker: &'static str,
    deadline: Deadline,
    label: &'static str,
) -> String {
    use std::process::{Command, Stdio};

    let (stdout, stderr) = {
        // The child is published before the lock is released, so a stop thread that
        // runs while the spawn is in flight waits for it and then ends the child.
        let mut slot = row_slot();
        let mut child = match Command::new(exe)
            .arg(index.to_string())
            .env(marker, "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                eprintln!("{label}: cannot start the row's process: {error}");
                std::process::exit(1);
            }
        };
        let stdout = drain(child.stdout.take().expect("piped stdout"));
        let stderr = drain(child.stderr.take().expect("piped stderr"));
        *slot = Some(child);
        (stdout, stderr)
    };
    let status = loop {
        let mut slot = row_slot();
        match slot.as_mut().expect("the published row").try_wait() {
            Ok(Some(status)) => {
                // Cleared with the reap, under the lock a stop takes: the child is
                // never signalled after it has been reaped and its pid reused.
                *slot = None;
                break status;
            }
            Ok(None) => {}
            Err(error) => {
                // The child is still on the device; it is killed before this
                // process leaves it holding gigabytes with nothing over it.
                drop(slot);
                end_row();
                eprintln!("{label}: cannot wait for a row: {error}");
                std::process::exit(1);
            }
        }
        drop(slot);
        if deadline.expired() {
            end_row();
            eprintln!("{label}: the wall-clock stop expired while a row ran — stopping");
            std::process::exit(EXIT_STOPPED);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let out = stdout.join().unwrap_or_default();
    let err = stderr.join().unwrap_or_default();
    print!("{out}");
    eprint!("{err}");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1).max(1));
    }
    out
}

/// Kill and reap the row. The slot is held across both, so a stop that finds it
/// taken waits for a child that is already dying instead of leaving it running.
fn end_row() {
    let mut slot = row_slot();
    if let Some(mut row) = slot.take() {
        let _ = row.kill();
        let _ = row.wait();
    }
}

/// Both of a child's pipes, drained while it runs — each on its own thread, because
/// a row that prints more than a pipe buffer would otherwise block on the pipe
/// nobody is reading.
fn drain(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    })
}

/// One row's figures as its child prints them and its parent reads them back: the
/// four numbers every product row carries, and the diagnostics behind them.
pub(crate) struct Reading {
    /// The row's machine name, the token its table entry is found by.
    pub(crate) row: String,
    pub(crate) cold_ms: f64,
    pub(crate) warm_ms: f64,
    pub(crate) memory: Memory,
    /// What the row is, per surface: regions and gain, frames and digest, a score.
    pub(crate) detail: Map<String, Value>,
}

impl Reading {
    /// The one line a child prints and its parent reads back.
    pub(crate) fn line(&self) -> String {
        serde_json::json!({
            "row": self.row,
            "cold_ms": self.cold_ms,
            "warm_ms": self.warm_ms,
            "resting_bytes": self.memory.now,
            "peak_bytes": self.memory.peak,
            "memory": if self.memory.footprint { "footprint" } else { "resident" },
            "detail": self.detail,
        })
        .to_string()
    }

    /// Read a child's stdout. Every figure the record states must be there: a row
    /// nobody measured may not be read as one.
    pub(crate) fn parse(stdout: &str) -> Result<Self> {
        let line = stdout
            .lines()
            .find(|line| line.trim_start().starts_with('{'))
            .with_context(|| "the row's process printed no reading")?;
        let value: Value = serde_json::from_str(line.trim())
            .with_context(|| format!("`{line}` is not a reading"))?;
        let object = value
            .as_object()
            .with_context(|| "a reading is a JSON object")?;
        let string = |key: &str| -> Result<&str> {
            object
                .get(key)
                .and_then(Value::as_str)
                .with_context(|| format!("the reading has no `{key}`"))
        };
        let number = |key: &str| -> Result<f64> {
            object
                .get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("the reading has no `{key}`"))
        };
        let bytes = |key: &str| -> Result<Option<u64>> {
            match object.get(key) {
                Some(Value::Null) => Ok(None),
                Some(value) => value
                    .as_u64()
                    .map(Some)
                    .with_context(|| format!("the reading's `{key}` is not a byte count")),
                None => bail!("the reading has no `{key}`"),
            }
        };
        let detail = object
            .get("detail")
            .and_then(Value::as_object)
            .with_context(|| "the reading has no `detail`")?
            .clone();
        let footprint = match string("memory")? {
            "footprint" => true,
            "resident" => false,
            other => bail!("the reading's `memory` is `{other}`, not `footprint` or `resident`"),
        };
        Ok(Self {
            row: string("row")?.to_string(),
            cold_ms: number("cold_ms")?,
            warm_ms: number("warm_ms")?,
            memory: Memory {
                now: bytes("resting_bytes")?,
                peak: bytes("peak_bytes")?,
                footprint,
            },
            detail,
        })
    }

    /// The diagnostics as one console line, `key=value` each.
    pub(crate) fn diagnostics(&self) -> String {
        self.detail
            .iter()
            .map(|(key, value)| match value {
                Value::String(text) => format!("{key}={text}"),
                _ => format!("{key}={value}"),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Walk a row-walking record run's rows, each in a process of its own, and return
/// their readings: the banner, the re-exec, the parse, and the refusal of a child that
/// answered under a name other than the row its index names. `name` is the run's
/// own row table.
pub(crate) fn walk(
    label: &'static str,
    rows: &[usize],
    marker: &'static str,
    deadline: Deadline,
    bounds: Bounds,
    name: impl Fn(usize) -> &'static str,
) -> Vec<Reading> {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            eprintln!("{label}: cannot find this binary: {error}");
            std::process::exit(1);
        }
    };
    let count = rows.len();
    eprintln!(
        "{label}: {count} {} · stop {}s · ceiling {} MiB",
        if count == 1 { "row" } else { "rows" },
        deadline.secs(),
        bounds.max_rss_mib,
    );
    let mut readings = Vec::with_capacity(rows.len());
    for &index in rows {
        let stdout = row_process(&exe, index, marker, deadline, label);
        let reading = match Reading::parse(&stdout) {
            Ok(reading) => reading,
            Err(error) => {
                eprintln!("{label}: {error:#}");
                std::process::exit(1);
            }
        };
        if reading.row != name(index) {
            eprintln!(
                "{label}: row {index} answered as `{}`, not `{}`",
                reading.row,
                name(index)
            );
            std::process::exit(1);
        }
        readings.push(reading);
    }
    readings
}

/// One row-walking record mode: the file it owns, its rows, and how a run's
/// readings become console lines and a record. The identity lives here once, so the
/// record a run prepares at start-up is the record it writes.
pub(crate) struct Mode {
    /// The run's name on its console lines and in its errors.
    pub(crate) label: &'static str,
    /// The one file this mode writes, and the record every one of its runs prepares
    /// or leaves alone.
    pub(crate) record: crate::measurements::Record,
    /// The environment variable a child of this run carries.
    pub(crate) marker: &'static str,
    /// How many rows the mode measures.
    pub(crate) total: usize,
    /// Row `index`'s name: the token its table entry is found by.
    pub(crate) name: fn(usize) -> &'static str,
    /// Measure row `index` in this process.
    pub(crate) measure: fn(usize, Bounds) -> Result<Reading>,
    /// Print the run's readings.
    pub(crate) print: fn(&[Reading]),
    /// The record's body and its input bullets, from the readings this run took.
    pub(crate) compose: fn(&[Reading]) -> Result<(String, String)>,
}

/// A record run's parent: take the record the mode owns, resolve the rows named on
/// the command line, measure each in a process of its own, print the readings, and
/// write the record only when the whole table was measured under the default bounds.
/// A child entered through `marker` measures the one row its argument names instead,
/// and writes nothing.
pub(crate) fn record_run(mode: Mode) {
    let Mode {
        label,
        record,
        marker,
        total,
        name,
        measure,
        print,
        compose,
    } = mode;
    // A row's child writes nothing, so it prepares nothing; the parent takes its
    // record before it reads its arguments, so a run refused on a bad argument still
    // clears a killed predecessor's temporary.
    let row_child = std::env::var_os(marker).is_some();
    if !row_child {
        if let Err(error) = record.prepare() {
            eprintln!("{label}: {error:#}");
            std::process::exit(1);
        }
    }
    let named = match indices(total, label) {
        Ok(named) => named,
        Err(error) => {
            eprintln!("{label}: {error:#}");
            std::process::exit(2);
        }
    };
    let rows = ordered(named, total);

    if row_child {
        let named: Vec<(usize, &'static str)> =
            rows.iter().map(|&index| (index, name(index))).collect();
        child(label, &named, marker, measure);
        return;
    }

    let bounds = bounds(label);
    // The lock is taken before the rows, whatever the run will be allowed to write:
    // screening decides whether a record is written, never whether the device is
    // shared, and a run under a raised bound measures exactly like any other.
    let heavy = Heavy::parent(label, bounds);
    // A run under a raised bound is screened: it measures and prints like any other,
    // and the record is what it does not write. Saying so before the rows run is
    // half of that.
    let screened = bound_reasons(
        bounds.timeout_secs,
        bounds.max_rss_mib,
        bounds.swap_growth_mib,
    );
    if !screened.is_empty() {
        refuse_record(label, &screened);
    }
    let readings = walk(label, &rows, marker, heavy.deadline(), bounds, name);
    print(&readings);
    let complete = readings.len() == total;
    if !complete {
        subset_note(label, readings.len(), total);
    }
    if complete && screened.is_empty() {
        // The record is the deliverable, so a run that cannot write it has not finished
        // — and this is the one write path of the two row-walking modes.
        let written = compose(&readings)
            .and_then(|(body, inputs)| crate::measurements::write(record, &body, &inputs));
        if let Err(error) = written {
            eprintln!("{label}: cannot write the record: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Say that a run measured a subset of its rows, and that no record is written.
pub(crate) fn subset_note(label: &str, named: usize, total: usize) {
    eprintln!("{label}: {named} of {total} rows named — a subset run writes no record");
}

/// One row's diagnostics, as the console prints them and the reading carries them.
pub(crate) fn detail(pairs: impl IntoIterator<Item = (&'static str, Value)>) -> Map<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// The rows a run measures, in the table's order: the whole table when none was
/// named, else the named ones sorted — a record's rows follow the product shapes,
/// never the order the indices happened to be typed in.
fn ordered(named: Vec<usize>, total: usize) -> Vec<usize> {
    if named.is_empty() {
        return (0..total).collect();
    }
    let mut named = named;
    named.sort_unstable();
    named
}

/// The row indices named on the command line: empty means the whole table. An
/// unparsable index, one past the table or a repeated one is refused before any
/// work starts.
pub(crate) fn indices(rows: usize, what: &str) -> Result<Vec<usize>> {
    let mut named: Vec<usize> = Vec::new();
    // Arguments are read as the platform hands them over: a byte sequence that is
    // not UTF-8 is not a row index, so it is refused like any other, not fatal.
    for arg in std::env::args_os().skip(1) {
        let arg = arg.to_string_lossy();
        let Ok(index) = arg.parse::<usize>() else {
            bail!("`{arg}` is not a row index — the {what} takes row indices and no flags");
        };
        if index >= rows {
            bail!("row {index} — there are {rows} rows");
        }
        if named.contains(&index) {
            bail!("row {index} is named twice");
        }
        named.push(index);
    }
    Ok(named)
}

/// Why a run under these bounds is not a record run, in the order the reasons are
/// checked: a bound larger than its default. The stand, the corpus run and the bench
/// all classify themselves by this one list, so the rule that decides whether a
/// record is written cannot drift between them.
pub(crate) fn bound_reasons(timeout: u64, max_rss: u64, swap_growth: u64) -> Vec<&'static str> {
    use crate::guard::{DEFAULT_MAX_RSS_MIB, DEFAULT_TIMEOUT_SECS, MAX_SWAP_GROWTH_MIB};

    let mut reasons = Vec::new();
    if timeout > DEFAULT_TIMEOUT_SECS {
        reasons.push("the wall-clock bound is larger than the default");
    }
    if max_rss > DEFAULT_MAX_RSS_MIB {
        reasons.push("the resident bound is larger than the default");
    }
    if swap_growth > MAX_SWAP_GROWTH_MIB {
        reasons.push("the swap-growth bound is larger than the default");
    }
    reasons
}

/// The bounds this process runs under, or the process stopped there: a bound
/// variable that is set and unparsable must not let the run pass for one held
/// under the defaults, so the entry point refuses it by name before any work.
pub(crate) fn bounds(label: &str) -> Bounds {
    match Bounds::from_env() {
        Ok(bounds) => bounds,
        Err(error) => {
            eprintln!("{label}: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Say why a run is screened, and that it writes nothing.
pub(crate) fn refuse_record(label: &str, reasons: &[&'static str]) {
    for reason in reasons {
        eprintln!("{label}: screened: {reason}");
    }
    eprintln!("{label}: a screened run writes no record");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record's rows follow the product shapes: naming them out of order still
    /// measures — and writes — the table's order.
    #[test]
    fn a_record_states_its_rows_in_table_order() {
        assert_eq!(ordered(vec![5, 4, 3, 2, 1, 0], 6), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(ordered(vec![2, 0], 6), vec![0, 2]);
        assert_eq!(ordered(Vec::new(), 3), vec![0, 1, 2]);
    }
}
