// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a run's own wall clock went — `RLX_WGPU_DBG_COUNTERS`.
//!
//! The bracket is one run of the compiled graph, entry to return: it opens in
//! [`RunReport::start`] and closes at the return edge, when the report drops.
//! Building the graph, loading weights and an AOT compile are outside it; a
//! dynamic-shape graph recompiled inside the call is inside, and that is what
//! [`Phase::Compile`] names. It fires once per outermost run, and only when the
//! switch is set — read once here, not at each mark. A schedule replay suppresses
//! it: a session under `RLX_WGPU_SNAPSHOT` reports nothing at all rather than a
//! figure that would be read as a plain run's. The browser build produces nothing
//! either, where `Tick` reads no clock. One caller-side call may still correspond
//! to several reported runs — a runtime wrapper can invoke the backend more than
//! once — while a per-token decode loop produces one report per token: each
//! bracket is that call's, never the caller's.
//!
//! The partition is host-visible and contiguous by construction — `stage`,
//! `compile`, `encode`, `wait`, `readback`, `host` and `probe` — read off the
//! edges the run already had (staging done, compile done, work submitted, fence,
//! outputs read, and the instrument's own query set, buffers and resolve). A step
//! belongs to the interval that *did* it: a step the run defers to the end of the
//! schedule is counted by `host`, the interval that runs it, and not by the
//! `encode` batch that walked past it. Its exactness is telescoping: every close
//! pushes exactly `now - last_ns` and moves `last_ns` with it, so the pushed spans
//! sum to the bracket whatever the run did in between, which is what
//! `convergence=ok` states. A label the run had no edge for is omitted with a
//! reason from [`REQUIRED`], never fabricated as a zero row.
//!
//! A boundary the run loop reaches with the schedule still in hand — its own exit,
//! or the end of its last batch — is [`RunReport::end_of_schedule`]: it closes the
//! interval still open over the steps it covered rather than over an empty range.
//!
//! The device view is a labelled lower bound, and is deliberately not part of the
//! partition: it is nested inside the host-visible intervals, is never added to,
//! subtracted from, or otherwise used to derive one, comes from a quantised,
//! implementation-defined device clock (so its figures are not comparable across
//! adapters, and the adapter they came from is named on every report), and is per
//! contiguous stretch of dispatch passes — `per=stretch`. `unavailable` is what a
//! pass reports when no figure can be obtained — it is never a zero and never a
//! measurement — and the wall time of such a pass stays inside the host-visible
//! interval that carried it; the `device_unusable` lines and
//! [`RunReport::pass_phase`]'s fallback for a pass no interval closed over are the
//! bookkeeping those two rules need.
//!
//! `README.md`'s "Where a run's time goes" is the consumer contract: field names,
//! per-label meaning, the sample block, and what a caller may and may not read out
//! of the figures. This module keeps only the rules a reader of the code needs.

use crate::device::WgpuDevice;

use super::step::{Step, step_name};

/// Printed prefix of every line of one report. A strict parser keys on it.
const PREFIX: &str = "[wgpu-run-report]";

/// Pass pairs the device view can hold: two query slots per dispatch pass. A
/// run with more passes than this still runs — see [`RunReport::begin_pass`].
/// `2 * MAX_PASSES` is exactly wgpu's `QUERY_SET_MAX_QUERIES`, the largest count
/// `create_query_set` accepts: this is the ceiling, not a round number.
const MAX_PASSES: u32 = 2048;

/// The intervals a run may have, with the reason printed when one never
/// happened. Fixed tokens: a caller reads them, so "absent" has to be as
/// specific as "present". `stage`, `encode`, `wait`, `readback` and `host` are
/// the five the run path always has; `compile` is the in-call compile a deferred
/// (dynamic-shape) graph performs, which a graph compiled outside the bracket
/// never has — and says so.
const REQUIRED: &[(Phase, &str)] = &[
    (Phase::Stage, "no_staging_edge"),
    (Phase::Compile, "no_deferred_compile_in_this_path"),
    (Phase::Encode, "no_device_work_in_this_schedule"),
    (Phase::Wait, "no_fence_edge_in_this_path"),
    (Phase::Readback, "no_readback_in_this_path"),
    (Phase::Host, "no_host_step_in_this_schedule"),
];

/// Per-step work or per-step output the run path itself reads a switch for. The
/// report names each because it inflates the very intervals the run is being asked
/// about, and a figure read without it is read against the wrong baseline. This is
/// the run path's own list, not every diagnostic switch in the crate: a switch a
/// dump path reads is not named here. `RLX_DBG_STEP` is set-ness rather than a
/// flag, matching how the run loop itself reads it. The first entry is the switch
/// this report rides on, and names both prints its census adds inside the bracket:
/// the pre-pass line, and then one line per *dispatched* step from inside the
/// batch's dispatch loop — so a run with no dispatched step has the pre-pass line
/// only. Those stderr writes are inside `stage` and `encode` on every reported run.
const EXTRA_WORK: &[(&str, &str, fn(&str) -> bool)] = &[
    (
        "RLX_WGPU_DBG_COUNTERS",
        "prepass_census_and_per_step_print",
        rlx_ir::env::flag,
    ),
    (
        "RLX_WGPU_ONE_OP_PER_PASS",
        "one_op_per_pass",
        rlx_ir::env::flag,
    ),
    (
        "RLX_BENCH_DISPATCH_ONLY",
        "dispatch_only",
        rlx_ir::env::flag,
    ),
    ("RLX_DBG_STEP", "per_step_print", is_set),
    (
        "RLX_WGPU_DBG_HOST_OP",
        "per_host_op_print",
        rlx_ir::env::flag,
    ),
    (
        "RLX_WGPU_DBG_INT8_HOST",
        "int8_host_trace",
        rlx_ir::env::flag,
    ),
    (
        "RLX_TRACE_PERFETTO",
        "perfetto_per_step_span",
        rlx_ir::env::flag,
    ),
    ("RLX_DEBUG_NANS", "per_node_nan_scan", rlx_ir::env::flag),
    ("RLX_WGPU_NAN_TRACE", "per_node_nan_scan", rlx_ir::env::flag),
    ("RLX_WGPU_DUMP_NODES", "per_node_dump", rlx_ir::env::flag),
    (
        "RLX_WGPU_ACTIVE_TRACE",
        "active_extent_trace",
        rlx_ir::env::flag,
    ),
];

fn is_set(key: &str) -> bool {
    rlx_ir::env::var(key).is_some()
}

/// One named interval's label. `name()` returns the printed token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    Stage,
    Compile,
    Encode,
    Wait,
    Readback,
    Host,
    Probe,
}

impl Phase {
    /// The token this interval is both opened as and printed under.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Phase::Stage => "stage",
            Phase::Compile => "compile",
            Phase::Encode => "encode",
            Phase::Wait => "wait",
            Phase::Readback => "readback",
            Phase::Host => "host",
            Phase::Probe => "probe",
        }
    }
}

/// One closed interval: the span attributed to it and the schedule steps that
/// span covered, by step kind. `steps` is not stored: it is `kinds`' counts
/// summed where the line is printed, so the figure and the histogram it came
/// from cannot disagree.
struct Interval {
    phase: Phase,
    ns: u64,
    kinds: Vec<(&'static str, usize)>,
}

/// One `begin_compute_pass` this run made. `interval_index` is the interval that
/// closed over it, attached when that interval is pushed — so the index always
/// names the stretch that carried the pass, and not one that later took the
/// index. It is `None` until then: a stretch the zero-span rule merged away is
/// attached at the next push, which is the interval that absorbed its time.
struct PassRecord {
    /// The device-view slot, or `None` for a pass past [`MAX_PASSES`] — counted
    /// and reported unusable, never dropped from the report.
    slot: Option<u32>,
    /// The interval that closed over this pass, or `None` for one no interval
    /// ever closed over — a remainder shorter than one clock tick, merged away.
    interval_index: Option<u32>,
}

/// The device-time view: one query pair per dispatch pass, resolved and mapped
/// once at the end of the run.
///
/// The browser build reads no clock there — `RunReport::start` cannot enable the
/// report there — so nothing ever builds one, and on `wasm32` neither the query
/// set nor the two buffers are ever read (what the `dead_code` allow covers).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct PassTimer {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// Thousandths of a nanosecond per tick, rounded. Nanoseconds would inflate
    /// a sub-nanosecond period to a whole one, and this view claims a lower
    /// bound; the conversion truncates, which can only ever lower a figure.
    period_ps: u64,
    /// The resolved samples, two per slotted pass, once the probe has read them.
    ticks: Vec<u64>,
}

impl PassTimer {
    /// Build the timer only where a figure can exist at all: the device
    /// advertises timestamps and the queue reports a period for them.
    fn new(dev: &WgpuDevice) -> Option<Self> {
        if !dev
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let period = dev.queue.get_timestamp_period();
        // An adapter that reports no period at all (zero, or a NaN that failed
        // to be one) has no figure to give.
        if period.is_nan() || period <= 0.0 {
            return None;
        }
        let bytes = 2 * u64::from(MAX_PASSES) * u64::from(wgpu::QUERY_SIZE);
        let qs = dev.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("rlx-wgpu run-report timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2 * MAX_PASSES,
        });
        let resolve = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu run-report timestamps resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu run-report timestamps read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            qs,
            resolve,
            read,
            period_ps: (period * 1000.0).round() as u64,
            ticks: Vec::new(),
        })
    }
}

pub(crate) struct RunReport {
    /// False makes every method a branch and a `return`. Nothing below reads a
    /// clock, allocates or records while it is false.
    enabled: bool,
    dev: Option<&'static WgpuDevice>,
    /// Bracket start, read once in [`RunReport::start`].
    t0: Option<rlx_ir::Tick>,
    /// End of the last pushed interval, in ns from `t0` — the bracket when the
    /// last close has run.
    last_ns: u64,
    /// The interval taking time right now, and the cursor range it took it over.
    open_phase: Phase,
    open_cursor: usize,
    /// The furthest cursor any boundary call has seen: the run's step count.
    cursor_seen: usize,
    schedule_len: usize,
    intervals: Vec<Interval>,
    /// Steps the uniform pre-pass counted itself, folded into the interval that
    /// closes next.
    pending: Vec<(&'static str, usize)>,
    /// Steps the run defers to the end of the schedule, folded into the `host`
    /// interval that runs them — see [`RunReport::note_step`].
    tail: Vec<(&'static str, usize)>,
    passes: Vec<PassRecord>,
    /// Passes recorded since the last pushed interval: they are attached to the
    /// interval that closes next, which is the one that carried them.
    passes_since_close: u32,
    timer: Option<PassTimer>,
}

impl RunReport {
    /// Bracket start. Reads the switch once; suppressed for a replay
    /// (`in_snapshot`, or a `RLX_WGPU_SNAPSHOT` session) and inert in the browser
    /// build.
    pub(crate) fn start(in_snapshot: bool) -> Self {
        let wanted = !cfg!(target_arch = "wasm32")
            && rlx_ir::env::flag("RLX_WGPU_DBG_COUNTERS")
            && !in_snapshot
            // A snapshot session is a replay from end to end — its last run is a
            // full pass, indistinguishable here from a production one — so the
            // session reports nothing rather than a figure read as a plain run's.
            && !rlx_ir::env::flag("RLX_WGPU_SNAPSHOT");
        let mut report = Self {
            enabled: false,
            dev: None,
            t0: None,
            last_ns: 0,
            // Inert while disabled, which never prints; an enabled report re-opens
            // this as `probe` before the bracket's first boundary call, because the
            // first thing it does is build its own query set.
            open_phase: Phase::Stage,
            open_cursor: 0,
            cursor_seen: 0,
            schedule_len: 0,
            intervals: Vec::new(),
            pending: Vec::new(),
            tail: Vec::new(),
            passes: Vec::new(),
            passes_since_close: 0,
            timer: None,
        };
        if !wanted {
            return report;
        }
        // The bracket opens before this instrument builds anything, and what it
        // builds is its own work, so that stretch is opened under `probe` rather
        // than charged to the run's `stage`.
        report.t0 = Some(rlx_ir::Tick::now());
        report.open_phase = Phase::Probe;
        // A run without an adapter panics on its own `wgpu_device().expect`, so
        // there is nothing to report and no `adapter=` line to print.
        let Some(dev) = crate::device::wgpu_device() else {
            return report;
        };
        report.dev = Some(dev);
        report.timer = PassTimer::new(dev);
        report.enabled = true;
        report
    }

    /// Attribute the span since the previous boundary to the interval already
    /// open, then open `phase` at `cursor` (the run's schedule step cursor).
    /// Idempotent: a call with the phase already open does nothing at all (no
    /// clock read) so consecutive marks of one kind merge into one interval.
    /// `sched` is used only to build the closed interval's step-kind histogram.
    pub(crate) fn open(&mut self, phase: Phase, cursor: usize, sched: &[Step]) {
        if !self.enabled {
            return;
        }
        // Only the drop-time probe closes over an empty slice, and it must not
        // erase the length of the schedule the run actually replayed.
        if !sched.is_empty() {
            self.schedule_len = sched.len();
        }
        self.cursor_seen = self.cursor_seen.max(cursor);
        if self.open_phase == phase {
            return;
        }
        self.boundary(cursor, sched);
        self.open_phase = phase;
    }

    /// Count one schedule step into the currently open interval (used by the
    /// uniform pre-pass, whose steps have no cursor of their own).
    ///
    /// A run whose uniforms already match the extent in use skips that pre-pass
    /// entirely, so its `stage` interval counts no steps — which is what it did.
    /// A step the run defers to the end of the schedule is not one of them: the
    /// pre-pass's own arm is a no-op for it and the interval that walks past its
    /// cursor entry is what counts it, into the tail ledger the `host` interval
    /// claims. Counting it here too would put one executed step on two lines.
    pub(crate) fn note_step(&mut self, step: &Step) {
        if !self.enabled || super::step::step_is_tail_host(step) {
            return;
        }
        merge(&mut self.pending, [(step_name(step), 1)]);
    }

    /// The schedule's end, read where the run loop reaches it — with the schedule
    /// still in hand, which is what makes a return edge countable.
    ///
    /// Not an interval of its own: the cursor range between the interval that is
    /// open and the end is folded into that interval's histogram here, and
    /// `open_cursor` moves with it, so the mark that closes that interval later
    /// walks an empty range and no step is counted twice. A run that returns from
    /// the loop or from inside its last batch (dispatch-only, a deferred tail)
    /// never reaches a mark of its own, and without this it would close its last
    /// interval over an empty range — losing the steps it covered, and reporting a
    /// `reached=` short of the schedule.
    pub(crate) fn end_of_schedule(&mut self, cursor: usize, sched: &[Step]) {
        if !self.enabled {
            return;
        }
        let end = cursor.min(sched.len());
        fold_range(
            self.open_cursor.min(end)..end,
            sched,
            &mut self.pending,
            &mut self.tail,
        );
        self.open_cursor = end;
        self.cursor_seen = self.cursor_seen.max(cursor);
    }

    /// Begin the run's own dispatch compute pass, sampling its device time when
    /// the adapter advertises the capability.
    ///
    /// One record per call, slotted or not: the device line's `passes` is what
    /// this run really created, so a run past [`MAX_PASSES`] reports its
    /// overflow instead of looking like a short one. The descriptor is a
    /// temporary inside this call and the returned pass borrows the encoder
    /// only, which is what lets the bookkeeping borrow `self` mutably here.
    pub(crate) fn begin_pass<'e>(
        &mut self,
        enc: &'e mut wgpu::CommandEncoder,
    ) -> wgpu::ComputePass<'e> {
        let label = Some("rlx-wgpu compute pass");
        if !self.enabled {
            return enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label,
                timestamp_writes: None,
            });
        }
        let slot = if self.timer.is_some() && (self.passes.len() as u32) < MAX_PASSES {
            Some(self.passes.len() as u32)
        } else {
            None
        };
        self.passes.push(PassRecord {
            slot,
            interval_index: None,
        });
        self.passes_since_close += 1;
        let timestamp_writes = match (slot, self.timer.as_ref()) {
            (Some(p), Some(timer)) => Some(wgpu::ComputePassTimestampWrites {
                query_set: &timer.qs,
                beginning_of_pass_write_index: Some(2 * p),
                end_of_pass_write_index: Some(2 * p + 1),
            }),
            _ => None,
        };
        enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes,
        })
    }

    /// Close the interval that is open at `cursor`. A non-zero span is pushed
    /// with the steps of the cursor range it covered; a zero span pushes nothing
    /// and advances nothing, so the stretch and its steps are absorbed by the
    /// next interval instead of printing a fabricated zero row.
    ///
    /// Telescoping is the whole point: every push adds exactly
    /// `now - last_ns` and moves `last_ns` with it, so the pushed spans sum to
    /// the bracket whatever the run did in between.
    fn boundary(&mut self, cursor: usize, sched: &[Step]) {
        let now = self.now_ns();
        let ns = now.saturating_sub(self.last_ns);
        if ns == 0 {
            return;
        }
        let mut kinds = std::mem::take(&mut self.pending);
        fold_range(self.open_cursor..cursor, sched, &mut kinds, &mut self.tail);
        // A tail ledger belongs to the `host` interval, which is the one that ran
        // those steps; `Drop` folds what is left into the close at the return edge,
        // so no step is lost.
        if self.open_phase == Phase::Host {
            merge(&mut kinds, std::mem::take(&mut self.tail));
        }
        kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        // Attach the passes this run began since the last push — exactly the tail
        // of `passes` — to the interval taking the index now: that is the stretch
        // that carried them, whatever a zero-span close did in between.
        let interval_index = self.intervals.len() as u32;
        let attach = self.passes.len() - self.passes_since_close as usize;
        for pass in &mut self.passes[attach..] {
            pass.interval_index = Some(interval_index);
        }
        self.passes_since_close = 0;
        self.intervals.push(Interval {
            phase: self.open_phase,
            ns,
            kinds,
        });
        self.last_ns = now;
        self.open_cursor = cursor;
    }

    /// Nanoseconds since the bracket start. Only the enabled path asks.
    fn now_ns(&self) -> u64 {
        self.t0.map_or(0, |t0| rlx_ir::Tick::now().elapsed_ns(t0))
    }

    /// Read the device view back. Its own work, and ordered: fence the last
    /// submission so the resolve sees every pass this run encoded, then a
    /// separate encoder that resolves and copies into a mappable buffer, then
    /// the map. Every submit and poll is annotated with the crate's counters,
    /// like every other one in `src/`.
    #[cfg(not(target_arch = "wasm32"))]
    fn probe(&mut self) {
        let Some(dev) = self.dev else {
            return;
        };
        let used = 2 * self.passes.iter().filter(|p| p.slot.is_some()).count() as u32;
        let Some(timer) = self.timer.as_mut() else {
            return;
        };
        if used == 0 {
            return;
        }
        let bytes = u64::from(used) * u64::from(wgpu::QUERY_SIZE);
        crate::backend::note_device_wait();
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu run-report resolve"),
            });
        enc.resolve_query_set(&timer.qs, 0..used, &timer.resolve, 0);
        enc.copy_buffer_to_buffer(&timer.resolve, 0, &timer.read, 0, bytes);
        crate::backend::note_device_submit();
        dev.queue.submit(std::iter::once(enc.finish()));
        let slice = timer.read.slice(..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        crate::backend::note_device_wait();
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        if !matches!(receiver.recv(), Ok(Ok(()))) {
            return;
        }
        if let Ok(view) = slice.get_mapped_range() {
            timer.ticks = bytemuck::cast_slice::<u8, u64>(&view[..bytes as usize]).to_vec();
        }
        timer.read.unmap();
    }

    /// No device clock in the browser, and no scheduled work that could be
    /// measured: the report there is the host-visible partition alone.
    #[cfg(target_arch = "wasm32")]
    fn probe(&mut self) {}

    /// The device view's figure for one pass, in ns — or `None` when no figure
    /// can be obtained from it, which is printed as `unavailable` and never as
    /// a zero.
    fn pass_ns(&self, pass: &PassRecord) -> Option<u64> {
        let slot = pass.slot? as usize;
        let timer = self.timer.as_ref()?;
        let pair = timer.ticks.get(2 * slot..2 * slot + 2)?;
        sample_ns(pair[0], pair[1], timer.period_ps, self.last_ns)
    }

    /// Which interval carried a pass, for the lines that name one. A record no
    /// interval ever closed over has no name of its own: every pass this run
    /// begins is begun in its dispatch work, so `encode` is what it is read as.
    fn pass_phase(&self, pass: &PassRecord) -> Phase {
        pass.interval_index
            .and_then(|index| self.intervals.get(index as usize))
            .map_or(Phase::Encode, |interval| interval.phase)
    }

    /// The whole report, one line per fact, joined — no trailing newline, so the
    /// caller can print it with a single `eprintln!` and a concurrent run cannot
    /// interleave its lines with these.
    fn render(&self) -> String {
        let mut lines = Vec::new();
        let partition_ns: u64 = self.intervals.iter().map(|interval| interval.ns).sum();
        lines.push(format!(
            "{PREFIX} run total_ms={} partition_ms={} convergence={} intervals={} schedule={} reached={}",
            ms(self.last_ns),
            ms(partition_ns),
            if partition_ns == self.last_ns {
                "ok"
            } else {
                "mismatch"
            },
            self.intervals.len(),
            self.schedule_len,
            // The furthest schedule cursor the run reached — not an interval's own
            // `steps=`, which counts only the steps that interval covered.
            self.cursor_seen,
        ));
        for interval in &self.intervals {
            lines.push(format!(
                "{PREFIX} interval label={} host_ms={} steps={} kinds={}",
                interval.phase.name(),
                ms(interval.ns),
                interval.kinds.iter().map(|(_, count)| count).sum::<usize>(),
                kinds_line(&interval.kinds),
            ));
        }
        for &(phase, reason) in REQUIRED {
            if !self.intervals.iter().any(|iv| iv.phase == phase) {
                lines.push(format!(
                    "{PREFIX} omitted label={} reason={reason}",
                    phase.name()
                ));
            }
        }
        if let Some(dev) = self.dev {
            let figures: Vec<Option<u64>> =
                self.passes.iter().map(|pass| self.pass_ns(pass)).collect();
            let measured = figures.iter().filter(|figure| figure.is_some()).count();
            let measured_ns: u64 = figures.iter().flatten().sum();
            // Why the view has no figure, three ways: a device with no clock this
            // instrument can read, a run that encoded no pass at all, or passes
            // that were encoded and yet yielded no usable sample. All are
            // `unavailable` — never a zero — and each says which it is.
            let view = if measured > 0 {
                "view=lower_bound".to_string()
            } else if self.timer.is_none() {
                "view=unavailable reason=no_timestamp_query_support".to_string()
            } else if self.passes.is_empty() {
                "view=unavailable reason=no_dispatch_pass_encoded".to_string()
            } else {
                "view=unavailable reason=no_usable_sample".to_string()
            };
            lines.push(format!(
                "{PREFIX} device adapter=\"{}\" backend={} {view} scope=dispatch_passes \
                 per=stretch passes={} measured={measured} unusable={} device_ms={}",
                dev.name,
                format!("{:?}", dev.backend).to_lowercase(),
                figures.len(),
                figures.len() - measured,
                if measured > 0 {
                    ms(measured_ns)
                } else {
                    "unavailable".to_string()
                },
            ));
            // Unusable passes are restated one line per interval that carried
            // them, in partition order: `index` is that interval line's 1-based
            // ordinal — the lines are printed in order, so it matches a repeated
            // label — and `host_ms` is that interval's own span, a figure already
            // in the partition, never a new quantity, never summed into the
            // partition and never a device figure. A record no interval ever
            // closed over has no ordinal and no span, and says so rather than
            // inventing a zero.
            let mut per_interval: Vec<(Option<u32>, usize)> = Vec::new();
            for (pass, figure) in self.passes.iter().zip(&figures) {
                if figure.is_some() {
                    continue;
                }
                match per_interval
                    .iter_mut()
                    .find(|(index, _)| *index == pass.interval_index)
                {
                    Some((_, count)) => *count += 1,
                    None => per_interval.push((pass.interval_index, 1)),
                }
            }
            // `None` sorts first, so the intervals that were pushed keep the order
            // their own lines are printed in.
            per_interval.sort_unstable();
            for (index, passes) in per_interval {
                let interval = index.and_then(|index| self.intervals.get(index as usize));
                lines.push(format!(
                    "{PREFIX} device_unusable passes={passes} interval={} index={} host_ms={}",
                    interval.map_or(Phase::Encode, |iv| iv.phase).name(),
                    index.map_or_else(|| "none".to_string(), |index| (index + 1).to_string()),
                    interval.map_or_else(|| "unavailable".to_string(), |iv| ms(iv.ns)),
                ));
            }
            for (index, (pass, figure)) in self.passes.iter().zip(&figures).enumerate() {
                lines.push(format!(
                    "{PREFIX} device_pass index={index} interval={} ms={}",
                    self.pass_phase(pass).name(),
                    figure.map_or_else(|| "unavailable".to_string(), ms),
                ));
            }
        }
        for &(key, effect, set) in EXTRA_WORK {
            if set(key) {
                lines.push(format!("{PREFIX} extra_work switch={key} effect={effect}"));
            }
        }
        lines.join("\n")
    }
}

impl Drop for RunReport {
    fn drop(&mut self) {
        if !self.enabled || std::thread::panicking() {
            return;
        }
        if self.timer.is_some() {
            // Opening the probe closes the run's last interval, so the probe's
            // own fence, resolve and map are measured into theirs alone.
            self.open(Phase::Probe, self.cursor_seen, &[]);
            self.probe();
        }
        // The close that carries the bracket to the return edge. Nothing else will
        // run, so a tail ledger no `host` interval claimed is folded into the
        // pending counts here rather than left on no line at all.
        merge(&mut self.pending, std::mem::take(&mut self.tail));
        self.boundary(self.cursor_seen, &[]);
        eprintln!("{}", self.render());
    }
}

/// Fold one cursor range of the schedule into histograms: a step the run defers to
/// the end of the schedule goes to the tail ledger, every other step to the
/// interval being counted. The close and the schedule's own end both read a range
/// this way, so the rule lives in one place.
fn fold_range(
    range: std::ops::Range<usize>,
    sched: &[Step],
    kinds: &mut Vec<(&'static str, usize)>,
    tail: &mut Vec<(&'static str, usize)>,
) {
    let Some(range) = sched.get(range) else {
        return;
    };
    for step in range {
        if super::step::step_is_tail_host(step) {
            merge(tail, [(step_name(step), 1)]);
        } else {
            merge(kinds, [(step_name(step), 1)]);
        }
    }
}

/// Fold step-kind counts into a histogram: one `(name, 1)` entry is the
/// single-step case, a taken ledger the fold. Kinds are few — one per step family
/// in the schedule — so a scan beats a map, and the histogram keeps one
/// deterministic order until a close sorts it.
fn merge(
    kinds: &mut Vec<(&'static str, usize)>,
    other: impl IntoIterator<Item = (&'static str, usize)>,
) {
    for (name, count) in other {
        match kinds.iter_mut().find(|(kind, _)| *kind == name) {
            Some((_, total)) => *total += count,
            None => kinds.push((name, count)),
        }
    }
}

/// The device figure one pair of samples yields, or `None` when it carries none:
/// a query that was never written, samples that do not advance (or run
/// backwards), a delta that truncates to no nanoseconds at all, or one larger
/// than the run's own bracket. `period_ps` is thousandths of a nanosecond per
/// tick, and a quantised, implementation-defined clock is what these come from —
/// every rule here turns a figure this instrument cannot trust into `None` rather
/// than into a small number.
fn sample_ns(begin: u64, end: u64, period_ps: u64, bracket_ns: u64) -> Option<u64> {
    // Both zero: the query was never written.
    if begin == 0 && end == 0 {
        return None;
    }
    // Not `begin < end`: a pass that shows no progress, or runs backwards,
    // is a dropped or wrapped sample, not a sub-tick pass.
    if end <= begin {
        return None;
    }
    // u128 because a quantised clock's samples are only as trustworthy as
    // the adapter that produced them; the conversion saturates rather than
    // wrapping, and truncating the thousandths only lowers the figure.
    let delta_ns =
        u64::try_from((end - begin) as u128 * period_ps as u128 / 1000).unwrap_or(u64::MAX);
    // Zero ns after truncation is the same non-measurement in another unit,
    // and `bracket_ns` is the run: nothing in it can be longer than it.
    if delta_ns == 0 || delta_ns > bracket_ns {
        return None;
    }
    Some(delta_ns)
}

/// Millisecond figure, six decimals, from the integer ns the report actually
/// compares: exactly that nanosecond, since 1 ns is 0.000001 ms. So the printed
/// intervals sum to the printed total, and a reader sees the partition close
/// rather than a flag claiming it does. One unit throughout: no ratio is printed.
fn ms(ns: u64) -> String {
    format!("{:.6}", ns as f64 / 1_000_000.0)
}

fn kinds_line(kinds: &[(&'static str, usize)]) -> String {
    kinds
        .iter()
        .map(|(name, count)| format!("{name}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::sample_ns;

    /// The five ways a sample pair carries no figure. None of these needs a device
    /// or a clock: they are the rules, not the samples they are applied to.
    #[test]
    fn a_lost_sample_is_never_a_figure() {
        // A query never written.
        assert_eq!(sample_ns(0, 0, 1_000, 1_000_000), None);
        // No advance at all, in a pair that is not both zero.
        assert_eq!(sample_ns(7, 7, 1_000, 1_000_000), None);
        // A pair that runs backwards, which is a dropped or wrapped sample.
        assert_eq!(sample_ns(9, 4, 1_000, 1_000_000), None);
        // Two ticks of a 42 ps period truncate to no nanoseconds at all.
        assert_eq!(sample_ns(1_000, 1_002, 42, 1_000_000), None);
        // A delta longer than the bracket containing it.
        assert_eq!(sample_ns(1_000, 4_000, 1_000, 2_000), None);
    }

    #[test]
    fn a_clean_pair_yields_its_nanoseconds() {
        // 3000 ticks at 1000 ps per tick, against a bracket that contains it.
        assert_eq!(sample_ns(1_000, 4_000, 1_000, 1_000_000), Some(3_000));
    }
}
