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
use crate::report::{Basis, Memory};

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
    /// The ceiling this process's guard holds. Kept beside the thread, not only inside it, so a
    /// row can state the swap growth *its own* ceiling measured.
    ceiling: MemoryCeiling,
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
        if let Some(note) = ceiling.unarmed() {
            eprintln!("{label}: {note}");
        }
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
            ceiling,
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
    // A row that panics is a row that failed: the child's whole job is to report the one
    // row it measured, so a refusal after the device was chosen is named here — with the
    // row and the device it stopped on — rather than ending the process at the layer.
    let measured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        measure(*index, heavy.bounds())
    }))
    .unwrap_or_else(|payload| Err(crate::device::stopped(payload.as_ref())));
    match measured {
        Ok(mut reading) => {
            reading.note_swap(heavy.ceiling.spent());
            println!("{}", reading.line())
        }
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
/// numbers every product row carries, and the diagnostics behind them.
pub(crate) struct Reading {
    /// The row's machine name, the token its table entry is found by.
    pub(crate) row: String,
    pub(crate) cold_ms: f64,
    pub(crate) warm_ms: f64,
    pub(crate) memory: Memory,
    /// The device the row ran on, stamped where the reading is built
    /// ([`Reading::new`]): the parent writes the record of that device, and a take whose
    /// readings disagree writes none.
    pub(crate) device: String,
    /// How the row's process arrived at that device, as [`crate::device::Arrival::token`]
    /// spells it — empty for the shipped arrangement. Carried rather than derived because the
    /// parent never compiles, so it cannot tell a machine with no graphics device from one
    /// whose graphics layer refused the first graph.
    pub(crate) arrival: String,
    /// What the row is, per surface: regions and gain, frames and digest, a score.
    pub(crate) detail: Map<String, Value>,
}

impl Reading {
    /// One row's figures, stamped with the device it ran on — the process's own decision,
    /// read here so no reading can be built without one.
    pub(crate) fn new(
        row: &'static str,
        cold_ms: f64,
        warm_ms: f64,
        memory: Memory,
        detail: Map<String, Value>,
    ) -> Self {
        Self {
            row: row.to_string(),
            cold_ms,
            warm_ms,
            memory,
            device: crate::device::label().to_string(),
            arrival: crate::device::arrival()
                .map_or("", crate::device::Arrival::token)
                .to_string(),
            detail,
        }
    }

    /// Add the figure the swap-growth guard held this row to, as the run's own ceiling reports
    /// it: the growth since that ceiling took its baseline — the quantity the ceiling is
    /// compared against, not the absolute reading, which includes whatever the process swapped
    /// before this run started. Rounded **up** to whole MiB, so a row whose guard held a
    /// non-zero growth never prints zero. The second key names which figure the growth is of,
    /// because a host with no per-process ledger reports the machine's used swap instead, and a
    /// row must not print one under the other's name. Diagnostics, not columns — the seven
    /// columns are the same on every device. Both keys are absent where the host reports no
    /// figure, and where it reports a different one at the two reads, in which case the ceiling
    /// refuses the growth rather than stating one.
    pub(crate) fn note_swap(&mut self, spent: Option<(u64, crate::guard::SwapFigure)>) {
        let Some((bytes, figure)) = spent else {
            return;
        };
        self.detail.insert(
            "swap_growth_mib".to_string(),
            Value::from(bytes.div_ceil(1024 * 1024)),
        );
        self.detail
            .insert("swap_figure".to_string(), Value::from(figure.token()));
    }

    /// The one line a child prints and its parent reads back.
    pub(crate) fn line(&self) -> String {
        serde_json::json!({
            "row": self.row,
            "cold_ms": self.cold_ms,
            "warm_ms": self.warm_ms,
            "resting_bytes": self.memory.now,
            "peak_bytes": self.memory.peak,
            "resting_gpu_bytes": self.memory.gpu_now,
            "peak_gpu_bytes": self.memory.gpu_peak,
            "memory": Basis::of([self.memory.footprint]).field(),
            "device": self.device,
            "arrival": self.arrival,
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
        let token = string("memory")?;
        let footprint = if token == Basis::Footprint.field() {
            true
        } else if token == Basis::Resident.field() {
            false
        } else {
            bail!(
                "the reading's `memory` is `{token}`, not `{}` or `{}`",
                Basis::Footprint.field(),
                Basis::Resident.field()
            );
        };
        Ok(Self {
            row: string("row")?.to_string(),
            cold_ms: number("cold_ms")?,
            warm_ms: number("warm_ms")?,
            arrival: string("arrival")?.to_string(),
            memory: Memory {
                now: bytes("resting_bytes")?,
                peak: bytes("peak_bytes")?,
                gpu_now: bytes("resting_gpu_bytes")?,
                gpu_peak: bytes("peak_gpu_bytes")?,
                footprint,
            },
            device: string("device")?.to_string(),
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

/// One row-walking record mode: the records it owns, its rows, and how a run's
/// readings become console lines and a record. The identity lives here once, so the
/// records a run prepares at start-up are the records it can write.
pub(crate) struct Mode {
    /// The run's name on its console lines and in its errors.
    pub(crate) label: &'static str,
    /// The record this mode writes on the graphics device, and the one it writes on
    /// the processor where the mode has one.
    pub(crate) record: crate::measurements::Record,
    pub(crate) processor_record: Option<crate::measurements::Record>,
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

/// A record run's parent: take the records the mode owns, resolve the rows named on
/// the command line, measure each in a process of its own, print the readings, and
/// write the record of the device the rows ran on only when the whole table was
/// measured under the default bounds. A child entered through `marker` measures the one
/// row its argument names instead, and writes nothing.
pub(crate) fn record_run(mode: Mode) {
    let Mode {
        label,
        record,
        processor_record,
        marker,
        total,
        name,
        measure,
        print,
        compose,
    } = mode;
    // A row's child writes nothing, so it prepares nothing; the parent takes every record it
    // may write before it reads its arguments, so a run refused on a bad argument still clears
    // a killed predecessor's temporary. Which ones it may write is the switch's: a forced
    // choice can only produce its own record, and an unforced run may produce either — so both
    // are prepared, and only a failure on the one the switch names stops the run.
    let row_child = std::env::var_os(marker).is_some();
    if row_child {
        // The child is the row: its device is settled before the sampler's window opens, so
        // a run forced onto the processor starts no thread that could only watch a device
        // it will not use.
        settle_device(label);
        // The GPU peak's window: from the first thing a row's process does until the
        // lifetime maximum is read in `Memory::finish`. The parent measures no row.
        crate::report::sample_gpu();
    }
    if !row_child {
        // The switch is read once. A value that is not understood is refused here, before any
        // work, rather than after the arguments and the lock; a value that is understood names
        // the record this take may write.
        let named = match crate::device::asked() {
            Err(message) => {
                eprintln!("{label}: {message}");
                std::process::exit(1);
            }
            Ok(crate::device::Asked::Cpu) => processor_record,
            Ok(crate::device::Asked::Gpu | crate::device::Asked::Auto) => Some(record),
        };
        // Both records this mode owns are prepared — swept, and proved writable — because an
        // unforced take cannot know its device before its first compile. A failure on the record
        // the switch names stops the run; a failure on the other is a warning, which for a forced
        // take names a file the take will not touch and for an unforced one says before the rows,
        // rather than after a whole table, that a fallback would have no record to write.
        for target in [Some(record), processor_record].into_iter().flatten() {
            let Err(error) = target.prepare() else {
                continue;
            };
            if Some(target) == named {
                eprintln!("{label}: {error:#}");
                std::process::exit(1);
            }
            eprintln!(
                "{label}: `{}` cannot be prepared ({error:#}) — a take that falls back to it \
                 writes nothing",
                target.file()
            );
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
    // The encoder-tail arm is resolved with the run's other inputs, so a value that is mistyped
    // or above its ceiling stops the run here — by name, before any work — rather than inside
    // the first recognition row. The cap itself is held by whoever builds the rungs
    // (`Transcriber::open`), which resolves it once and does not read the environment again.
    let tail_arm = tail_arm(label);
    // The host-width arm, resolved the same way and for the same reason: a value that is not a
    // boolean stops the run by name, before any work.
    let host_arm = host_arm(label);
    // Which files this run reads, on stderr: stdout carries the one-JSON-line row
    // protocol and gains nothing.
    eprintln!(
        "{label}: models {} · corpus {} · music {} · wake-corpus {}",
        crate::paths::root().display(),
        crate::paths::corpus().display(),
        crate::paths::music().display(),
        crate::paths::wake_corpus().display(),
    );
    // The lock is taken before the rows, whatever the run will be allowed to write:
    // screening decides whether a record is written, never whether the device is
    // shared, and a run under a raised bound measures exactly like any other.
    let heavy = Heavy::parent(label, bounds);
    // The device is settled after the records are prepared and the arguments are read, so a run
    // refused on either still leaves the temporaries swept, and a forced graphics run does not
    // create the device before its bounds are known. (The bench reads the switch itself, before
    // its record, because a forced processor take there writes nothing.)
    settle_device(label);
    // A run under a raised bound or a route switch is screened: it measures and prints like
    // any other, and the record is what it does not write. Saying so before the rows run is
    // half of that; the other half is the device a row actually ran on, which decides which
    // record the take writes once the rows are in.
    let screened = screening_reasons(bounds, tail_arm, host_arm);
    // A forced processor take of a mode that has no processor record is knowable here, so it is
    // said before the rows rather than after them: screening decides whether a record is written,
    // never whether the run measures, and an operator should not learn it hours in.
    let forced_unrecorded =
        crate::device::asked() == Ok(crate::device::Asked::Cpu) && processor_record.is_none();
    if forced_unrecorded {
        refuse_record(label, &[NO_PROCESSOR_RECORD]);
    }
    if !screened.is_empty() {
        refuse_record(label, &screened);
    }
    let unscreened = screened.is_empty() && !forced_unrecorded;
    let readings = walk(label, &rows, marker, heavy.deadline(), bounds, name);
    print(&readings);
    let complete = readings.len() == total;
    if !complete {
        subset_note(label, readings.len(), total);
    }
    // The record follows the device the rows actually ran on: a take writes the record of
    // its device and only that record. One run is one device, so the device is read back
    // from the readings — and a take whose readings disagree writes nothing.
    let target = if complete && unscreened {
        record_for(label, &readings, record, processor_record)
    } else {
        None
    };
    if let Some((target, on, arrival)) = target {
        // The record is the deliverable, so a run that cannot write it has not finished
        // — and this is the one write path of the two row-walking modes.
        let written = compose(&readings).and_then(|(body, inputs)| {
            crate::measurements::write(target, on, arrival, &body, &inputs)
        });
        if let Err(error) = written {
            eprintln!("{label}: cannot write the record: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Why a mode that has no processor record writes nothing on the processor. One rule, stated
/// once, for both routes to the processor: each mode applies it at the first moment the answer
/// is known — before its rows where the switch is forced, and at its record gate when a
/// fallback put the take there.
pub(crate) const NO_PROCESSOR_RECORD: &str =
    "this mode has no record for the cpu device — a processor take writes nothing";

/// The record this take writes and the device it ran on, or `None` when it writes none —
/// saying why, when the reason is one the run could not state before its rows.
///
/// The mode, not the device, decides whether it has a record for the device that ran: the
/// stand has a processor record and the corpus run and the wake-word bench do not.
fn record_for(
    label: &'static str,
    readings: &[Reading],
    record: crate::measurements::Record,
    processor_record: Option<crate::measurements::Record>,
) -> Option<(
    crate::measurements::Record,
    rlx_runtime::Device,
    Option<crate::device::Arrival>,
)> {
    let Taken {
        device: token,
        arrival,
    } = taken(label, readings)?;
    let Some(on) = crate::device::from_label(token) else {
        refuse_record(
            label,
            &[&format!(
                "the readings name a device this mode has no record for: `{token}`"
            )],
        );
        return None;
    };
    let record = match on {
        rlx_runtime::Device::Cpu => processor_record,
        _ => Some(record),
    };
    if record.is_none() {
        refuse_record(label, &[NO_PROCESSOR_RECORD]);
    }
    record.map(|record| (record, on, arrival))
}

/// How the take reached its device, as the rows stated it: one run is one device and one
/// arrival, and a take whose rows disagree writes no record.
struct Taken<'a> {
    device: &'a str,
    arrival: Option<crate::device::Arrival>,
}

/// The one device and one arrival a take's readings state, or `None` — saying why — when they
/// state neither. The two readings are one decision because a refusal and "no arrival" are not
/// the same answer: a take whose rows disagree, or whose token this build does not know, would
/// otherwise write a record stating a plain invocation for a run that needed a switch.
fn taken<'a>(label: &str, readings: &'a [Reading]) -> Option<Taken<'a>> {
    let first = readings.first()?;
    let device = first.device.as_str();
    if !readings.iter().all(|reading| reading.device == device) {
        let mut devices: Vec<&str> = readings
            .iter()
            .map(|reading| reading.device.as_str())
            .collect();
        devices.sort_unstable();
        devices.dedup();
        refuse_record(
            label,
            &[&format!(
                "the readings ran on {} devices ({}) — a mixed take writes no record",
                devices.len(),
                devices.join(", ")
            )],
        );
        return None;
    }
    let token = first.arrival.as_str();
    if !readings.iter().all(|reading| reading.arrival == token) {
        refuse_record(
            label,
            &[
                "the readings disagree about how the run reached its device — a mixed take writes \
              no record",
            ],
        );
        return None;
    }
    let arrival = if token.is_empty() {
        None
    } else {
        let Some(arrival) = crate::device::Arrival::from_token(token) else {
            refuse_record(
                label,
                &[&format!(
                    "the readings name an arrival this build does not know: `{token}`"
                )],
            );
            return None;
        };
        Some(arrival)
    };
    Some(Taken { device, arrival })
}

/// The run's device, settled before any work: a switch that cannot be honoured stops the
/// run by name, before it measures anything.
fn settle_device(label: &'static str) {
    if let Err(message) = crate::device::resolve(&format!("{label}: ")) {
        eprintln!("{label}: {message}");
        std::process::exit(1);
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

/// Why a run writes no record: [`bound_reasons`], and a route switch that makes the take
/// something other than the shipped arrangement. `MAHBOT_TAIL_GROUP` picks the encoder-tail
/// arm, and a record carries the shipped arrangement's numbers only — a take on the A/B arm
/// is a take of its own, so the variable being set at all (`tail_arm`, from the same
/// resolution that validated it) screens the run. `MAHBOT_HOST_F16` is the second
/// (`host_arm`): it picks between the file's width on the host and the f32 image that arm
/// replaced. The wake bench reads neither variable this way — its sweep groups by `GROUP`,
/// which `MAHBOT_TAIL_GROUP` does not touch, and its own `MAHBOT_HOST_F16` rule lives in
/// `bench::BenchConfig`. The device is not a route switch: a take writes the record of the
/// device it ran on, which the caller decides once the rows have run — and a mode with no
/// record for that device says so instead ([`NO_PROCESSOR_RECORD`]).
fn screening_reasons(bounds: Bounds, tail_arm: bool, host_arm: bool) -> Vec<&'static str> {
    let mut reasons = bound_reasons(
        bounds.timeout_secs,
        bounds.max_rss_mib,
        bounds.swap_growth_mib,
    );
    if tail_arm {
        reasons.push("MAHBOT_TAIL_GROUP selects the encoder-tail A/B arm");
    }
    if host_arm {
        reasons.push("MAHBOT_HOST_F16 selects the host-width A/B arm");
    }
    reasons
}

/// A variable the run reads before any work, or the process stopped there: a value that is
/// set and unparsable must not let the run pass for the shipped arrangement's, so it is
/// refused by name rather than defaulted.
fn resolved_or_exit<T>(label: &str, resolved: Result<T>) -> T {
    match resolved {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{label}: {error:#}");
            std::process::exit(1);
        }
    }
}

/// The bounds this process runs under.
pub(crate) fn bounds(label: &str) -> Bounds {
    resolved_or_exit(label, Bounds::from_env())
}

/// Whether `MAHBOT_TAIL_GROUP` names the encoder-tail A/B arm.
fn tail_arm(label: &str) -> bool {
    resolved_or_exit(label, crate::listen::tail_group()).1
}

/// Whether `MAHBOT_HOST_F16` names the host-width A/B arm.
fn host_arm(label: &str) -> bool {
    resolved_or_exit(label, crate::host_width::resolved()).1
}

/// Say why a run is screened, and that it writes nothing.
pub(crate) fn refuse_record(label: &str, reasons: &[&str]) {
    for reason in reasons {
        eprintln!("{label}: screened: {reason}");
    }
    eprintln!("{label}: a screened run writes no record");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reading on `device`, reached by `arrival`: the tests below differ only in what they
    /// state about where the row ran.
    fn reading(device: &str, arrival: &str) -> Reading {
        Reading {
            row: "recognition.short".to_string(),
            cold_ms: 1.0,
            warm_ms: 1.0,
            arrival: arrival.to_string(),
            memory: crate::report::Memory::take(),
            device: device.to_string(),
            detail: detail([("regions", Value::from(1))]),
        }
    }

    /// The record's rows follow the product shapes: naming them out of order still
    /// measures — and writes — the table's order.
    #[test]
    fn a_record_states_its_rows_in_table_order() {
        assert_eq!(ordered(vec![5, 4, 3, 2, 1, 0], 6), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(ordered(vec![2, 0], 6), vec![0, 2]);
        assert_eq!(ordered(Vec::new(), 3), vec![0, 1, 2]);
    }

    /// A row's swap figure is the growth its ceiling held it to, rounded **up** to whole MiB:
    /// a non-zero growth never prints as the zero that reads as "this row spent no swap".
    #[test]
    fn a_non_zero_swap_growth_never_prints_zero() {
        let mut held = reading("gpu", "");
        for (bytes, mib) in [
            (1u64, 1u64),
            (1 << 20, 1),
            ((1 << 20) + 1, 2),
            (3 << 30, 3 * 1024),
        ] {
            held.note_swap(Some((bytes, crate::guard::SwapFigure::Process)));
            assert_eq!(
                held.detail.get("swap_growth_mib"),
                Some(&Value::from(mib)),
                "{bytes} bytes of growth"
            );
        }
        assert_eq!(
            held.detail.get("swap_figure"),
            Some(&Value::from("process"))
        );
        // A host that reports no figure states neither key rather than a zero.
        let mut unread = reading("cpu", "absent");
        unread.note_swap(None);
        assert!(unread.detail.get("swap_growth_mib").is_none());
        assert!(unread.detail.get("swap_figure").is_none());
    }

    /// A take on the encoder-tail A/B arm is not the shipped arrangement's numbers, so it
    /// writes no record; a take under the defaults does.
    #[test]
    fn a_route_switch_screens_the_run() {
        let shipped = crate::guard::Bounds {
            timeout_secs: crate::guard::DEFAULT_TIMEOUT_SECS,
            max_rss_mib: crate::guard::DEFAULT_MAX_RSS_MIB,
            swap_growth_mib: crate::guard::MAX_SWAP_GROWTH_MIB,
        };
        assert!(screening_reasons(shipped, false, false).is_empty());
        assert_eq!(
            screening_reasons(shipped, true, false),
            vec!["MAHBOT_TAIL_GROUP selects the encoder-tail A/B arm"]
        );
        assert_eq!(
            screening_reasons(shipped, false, true),
            vec!["MAHBOT_HOST_F16 selects the host-width A/B arm"]
        );
    }

    /// One run is one device and one arrival: a take whose readings disagree did not measure
    /// one take, so no record states it, and the devices it saw are named instead.
    #[test]
    fn readings_that_disagree_are_not_one_take() {
        assert_eq!(
            taken("stand", &[reading("gpu", ""), reading("gpu", "")]).map(|taken| taken.device),
            Some("gpu")
        );
        assert_eq!(
            taken(
                "stand",
                &[reading("cpu", "absent"), reading("cpu", "absent")]
            )
            .map(|taken| taken.device),
            Some("cpu")
        );
        assert!(
            taken("stand", &[reading("gpu", ""), reading("cpu", "absent")]).is_none(),
            "readings that ran on two devices are not one take"
        );
        assert!(
            taken("stand", &[reading("gpu", "forced-gpu"), reading("gpu", "")]).is_none(),
            "readings that state two arrivals are not one take"
        );
    }

    /// A take writes the record of the device it ran on, and a forced choice can only produce
    /// its own: the graphics record is not written by a processor take, and a mode with no
    /// processor record writes nothing on the processor.
    #[test]
    fn a_take_writes_the_record_of_its_device() {
        use crate::measurements::Record;
        // The record follows the device, not the switch.
        assert_eq!(
            record_for(
                "stand",
                &[reading("gpu", ""), reading("gpu", "")],
                Record::Stand,
                Some(Record::StandProcessor)
            ),
            Some((Record::Stand, rlx_runtime::Device::Gpu, None))
        );
        assert_eq!(
            record_for(
                "stand",
                &[reading("cpu", "absent"), reading("cpu", "absent")],
                Record::Stand,
                Some(Record::StandProcessor)
            ),
            Some((
                Record::StandProcessor,
                rlx_runtime::Device::Cpu,
                Some(crate::device::Arrival::Absent)
            ))
        );
        // Two arrival tokens are two takes: no record states either, and the run is not
        // reported as one that needed no switch.
        assert_eq!(
            record_for(
                "stand",
                &[reading("gpu", "forced-gpu"), reading("gpu", "")],
                Record::Stand,
                Some(Record::StandProcessor)
            ),
            None
        );
        // A mode with no processor record writes nothing for a take that fell back.
        assert_eq!(
            record_for(
                "wake-word",
                &[reading("cpu", "refused"), reading("cpu", "refused")],
                Record::WakeWordBench,
                None
            ),
            None
        );
    }
}
