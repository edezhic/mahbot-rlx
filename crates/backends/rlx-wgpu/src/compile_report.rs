// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where one wgpu compile's own wall clock went — `RLX_WGPU_COMPILE_REPORT`.
//!
//! One ledger per compile, on the compiling thread, opened and closed by one guard.
//! A phase opens with `Phase::start` and records itself on drop, so the figures
//! telescope to the compile's total without a second clock: what the named phases do
//! not cover is printed as `other`, never folded into one of them. `steps` is the
//! lowering loop — the `Step` schedule together with the uniform buffers and bind
//! groups that point into the arena, because they are built at one site. `restarts`
//! counts the lowering's restarts: it compiles again without folds when a folded
//! activation would run on the host, the printed phases are the last attempt's, and
//! `total` and `other` cover them all. No product graph triggers one, so the field is
//! absent from every row of the stand (it is printed only when non-zero).
//!
//! These are the backend's phases. Graph construction and the parameter upload around
//! them belong to the caller, which prints a line of its own when it goes through the
//! reporting helper and this line alone when it does not. Without the switch the ledger
//! stays empty and a phase is one flag read.

use std::cell::RefCell;
use std::time::Instant;

/// Whether a compile records its phases.
pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_WGPU_COMPILE_REPORT")
}

/// One compile's figures: the backend's phases and total, and how many times the
/// lowering restarted.
#[derive(Default)]
struct Report {
    /// The backend compile's whole wall clock, every attempt included.
    total_ms: f64,
    phases: Vec<(&'static str, f64)>,
    restarts: usize,
}

impl Report {
    /// Print the decomposition: the phases in the order they closed, and `other` — the
    /// compile time the named ones do not cover.
    fn print(&self) {
        let mut line = format!("[wgpu-compile] total={:.2}ms", self.total_ms);
        if self.restarts > 0 {
            line.push_str(&format!(" restarts={}", self.restarts));
        }
        for (name, ms) in &self.phases {
            line.push_str(&format!(" {name}={ms:.2}"));
        }
        let covered: f64 = self.phases.iter().map(|(_, ms)| ms).sum();
        line.push_str(&format!(" other={:.2}", (self.total_ms - covered).max(0.0)));
        eprintln!("{line}");
    }
}

thread_local! {
    static LEDGER: RefCell<Report> = RefCell::new(Report::default());
}

/// One interval of a compile, recorded into this thread's ledger when it drops.
pub(crate) struct Phase {
    name: &'static str,
    start: Option<Instant>,
}

impl Phase {
    pub(crate) fn start(name: &'static str) -> Self {
        Self {
            name,
            start: enabled().then(Instant::now),
        }
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let ms = start.elapsed().as_secs_f64() * 1e3;
            LEDGER.with(|l| l.borrow_mut().phases.push((self.name, ms)));
        }
    }
}

/// One compile's bracket: opens the ledger and closes it on drop, printing however the
/// compile ends — including the `panic` path, which is the point of a guard here.
pub(crate) struct Bracket(Option<Instant>);

impl Bracket {
    pub(crate) fn open() -> Self {
        if enabled() {
            LEDGER.with(|l| *l.borrow_mut() = Report::default());
            Self(Some(Instant::now()))
        } else {
            Self(None)
        }
    }
}

impl Drop for Bracket {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            let total_ms = start.elapsed().as_secs_f64() * 1e3;
            LEDGER.with(|l| {
                let mut report = std::mem::take(&mut *l.borrow_mut());
                report.total_ms = total_ms;
                report.print();
            });
        }
    }
}

/// Note that the lowering is about to restart from the top: the phases of the
/// discarded attempt are dropped, since the printed line describes the one that
/// ended and `other` carries the rest.
pub(crate) fn restart() {
    if enabled() {
        LEDGER.with(|l| {
            let mut ledger = l.borrow_mut();
            ledger.restarts += 1;
            ledger.phases.clear();
        });
    }
}
