// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which device this process runs on: read once, decided once, said once.
//!
//! An unforced run asks for the graphics device first and continues on the processor when
//! the graphics path does not come up; the first compile is where that is decided, and it
//! is where the device has always been created, so the cold figures keep the bring-up they
//! were measured with. One row runs on one device: a failure after the first compile is a
//! product error, not a reason to switch — and the error names the switch that runs this
//! machine on the processor.
//!
//! # The bound
//!
//! "Some path works on every machine" is bounded by what the build compiles in, by the
//! installed model tree (`crate::paths`), and — until the processor path is complete — by
//! how far a processor run gets: a run reaches the product's own work and anything still
//! refusing there is named. The processor backend is named where this crate declares its
//! runtime dependency, so a build that loses it loses the fallback with it, and says so by
//! name at the first compile.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow};
use rlx_ir::Graph;
use rlx_runtime::{CompiledGraph, Device, Session};

/// The switch: `gpu`, `cpu` or `auto` (the default).
const VAR: &str = "MAHBOT_DEVICE";

/// The reason a forced choice's device line states.
const FORCED_GPU: &str = "MAHBOT_DEVICE=gpu";
const FORCED_CPU: &str = "MAHBOT_DEVICE=cpu";
/// The name a reading carries for the device its row ran on, and the one the record
/// prices.
pub(crate) const PRICED: &str = "gpu";
/// Why a forced take is not a record take, whatever device it ran on.
const FORCED: &str =
    "MAHBOT_DEVICE names the device (a forced take is not the shipped arrangement's)";
/// The reason a run that continued on the processor states, before or after its rows.
pub(crate) const FALLBACK: &str =
    "the run continued on the processor (the record prices the graphics device)";
/// What the product says when the graphics device is not there: the fallback's line and a
/// forced graphics failure state the same phrase.
const ABSENT: &str = "no usable graphics device";
/// The instruction a refusal on the graphics device carries: one device per run, so the way
/// out is the switch, and a dead end is not acceptable.
const WAY_OUT: &str = " — set MAHBOT_DEVICE=cpu to run this machine on the processor";

/// [`WAY_OUT`] where the switch is still the operator's to set, and nothing where the
/// process is pinned: a harness refuses the switch, so naming it would instruct the
/// operator to do what the harness refuses.
fn way_out(pinned: bool) -> &'static str {
    if pinned { "" } else { WAY_OUT }
}

/// The switch, parsed once.
static ASKED: OnceLock<Asked> = OnceLock::new();
/// The device this run settled on: the first `set` is the run's decision, and it is never
/// revised.
static CHOSEN: OnceLock<Device> = OnceLock::new();
/// The console prefix this run's lines carry, separator included (`stand: `).
static PREFIX: OnceLock<String> = OnceLock::new();
/// The undecided run's decision is one critical section: two compiles in one process must
/// not settle different devices.
static DECIDING: Mutex<()> = Mutex::new(());
/// Whether a harness pinned this process ([`pin_gpu`]): a pinned process refuses the switch,
/// so its refusals carry no way-out hint.
static PINNED: AtomicBool = AtomicBool::new(false);

/// What the switch asked for. `Auto` is the shipped arrangement: the graphics device
/// where it comes up, the processor where it does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Asked {
    Auto,
    Gpu,
    Cpu,
}

impl Asked {
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            _ => Err(format!(
                "{VAR}={} is not understood: name `gpu`, `cpu` or `auto`",
                raw.trim()
            )),
        }
    }
}

/// The switch, parsed once: `auto` where it names nothing, the value where it names one,
/// and the reason a value cannot be honoured where it cannot. Only a value that can be
/// honoured is latched, so an unparsable one is the error every caller refuses on rather
/// than a state a reader could interpret leniently.
fn switch() -> std::result::Result<Asked, String> {
    if let Some(asked) = ASKED.get() {
        return Ok(*asked);
    }
    let asked = match std::env::var_os(VAR) {
        None => Asked::Auto,
        Some(raw) => match raw.to_str() {
            None => return Err(format!("{VAR} is not valid UTF-8")),
            Some(raw) => Asked::parse(raw)?,
        },
    };
    let _ = ASKED.set(asked);
    Ok(asked)
}

fn chosen() -> Option<Device> {
    CHOSEN.get().copied()
}

/// Why a run cannot start on the device it was told to use.
pub(crate) enum Unusable {
    /// The switch is not understood: a bad invocation, refused before any work.
    Unreadable(String),
    /// The device the switch names is not on this host.
    Missing(String),
}

impl Unusable {
    /// What the console states, after the binary's own prefix.
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Unreadable(message) | Self::Missing(message) => message,
        }
    }

    /// Whether this is a refused invocation rather than a host that cannot deliver: the
    /// binaries that classify their exits say so with their own code.
    pub(crate) fn refused(&self) -> bool {
        matches!(self, Self::Unreadable(_))
    }
}

/// Settle what can be settled before any work: a forced choice is latched here, and a
/// value that is not understood is refused. An unforced run stays undecided until its
/// first compile. The parsed switch comes back, so a caller that screens on it reads the
/// value this run will honour rather than the environment a second time.
///
/// A forced graphics run asks for the device here, so a host that cannot deliver it fails
/// before the first row instead of at the first compile. That moves the bring-up out of
/// the timed window for that run only, and a forced run writes no record, so no record
/// figure moves with it.
pub(crate) fn resolve(prefix: &str) -> std::result::Result<Asked, Unusable> {
    let _ = PREFIX.set(prefix.to_string());
    let asked = match switch() {
        Ok(asked) => asked,
        Err(message) => return Err(Unusable::Unreadable(message)),
    };
    match asked {
        Asked::Auto => Ok(asked),
        Asked::Cpu => {
            settle(Device::Cpu, Some(FORCED_CPU));
            Ok(asked)
        }
        Asked::Gpu => {
            // The probe is the graphics path's own first step, so it is guarded like a
            // compile: a backend that dies in device creation is named, not fatal.
            let probed = catch_unwind(AssertUnwindSafe(|| rlx_runtime::is_available(Device::Gpu)));
            match probed {
                Ok(true) => {
                    settle(Device::Gpu, Some(FORCED_GPU));
                    Ok(asked)
                }
                Ok(false) => Err(Unusable::Missing(format!(
                    "{VAR}=gpu: {ABSENT} on this host"
                ))),
                Err(payload) => Err(Unusable::Missing(format!(
                    "{VAR}=gpu: the graphics device could not be created: {}",
                    panic_message(payload.as_ref())
                ))),
            }
        }
    }
}

/// The device this run compiles for and states: the one it settled on. A run that has not
/// settled yet is asking for the graphics device — an unforced run's first ask, and what a
/// forced graphics run has already latched — so that is the answer, and it is never a
/// reading of the switch.
pub(crate) fn of() -> Device {
    chosen().unwrap_or(Device::Gpu)
}

/// The device a row ran on, as its reading states it.
pub(crate) fn label() -> &'static str {
    name(of())
}

/// A device's name as a console line and a reading state it.
fn name(device: Device) -> &'static str {
    if device == Device::Cpu { "cpu" } else { PRICED }
}

/// Why this run's device makes it not a record run: the record prices the graphics
/// device, so a forced choice and a run that continued on the processor both write
/// nothing. The switch is the caller's — every binary resolves it before the rows and
/// refuses a value it cannot honour — so this reads no environment of its own.
pub(crate) fn screening_reasons(asked: Asked) -> Option<&'static str> {
    reasons(asked, chosen())
}

fn reasons(asked: Asked, chosen: Option<Device>) -> Option<&'static str> {
    match asked {
        Asked::Gpu | Asked::Cpu => Some(FORCED),
        Asked::Auto if chosen == Some(Device::Cpu) => Some(FALLBACK),
        Asked::Auto => None,
    }
}

/// Compile one graph for this run's device, deciding an unforced run at its first
/// compile.
///
/// The device is not a parameter: this is the run's one decision ([`of`]), so no caller
/// can compile for a device the run did not choose. A settled run compiles on its
/// device and a backend refusal is a named error; a forced graphics run never falls
/// back. An unforced, still-undecided run asks for the graphics device first — no
/// usable device, or a first graph the graphics layer refuses, moves the whole run to
/// the processor, which compiles the same graph. Nothing is decided after that.
pub(crate) fn compile(graph: &Graph) -> Result<CompiledGraph> {
    let asked = switch().map_err(|message| anyhow!("{message}"))?;
    match asked {
        // A forced choice is honoured exactly: the processor never asks for the graphics
        // device, and the graphics device never quietly becomes the processor. The
        // processor is settled before its compile — the device is not in question — and
        // the graphics device only after one succeeds, so its line never states a device
        // that did not come up.
        Asked::Cpu => {
            settle(Device::Cpu, Some(FORCED_CPU));
            attempt(Device::Cpu, graph).map_err(|r| r.error(Device::Cpu))
        }
        Asked::Gpu => {
            let compiled = attempt(Device::Gpu, graph).map_err(|r| r.error(Device::Gpu))?;
            settle(Device::Gpu, Some(FORCED_GPU));
            Ok(compiled)
        }
        Asked::Auto => match chosen() {
            Some(device) => attempt(device, graph).map_err(|r| r.error(device)),
            None => decide(graph),
        },
    }
}

/// The undecided run's one decision, taken under a lock: a second compile in the same
/// process finds the choice already made instead of settling one of its own.
fn decide(graph: &Graph) -> Result<CompiledGraph> {
    let _deciding = DECIDING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(device) = chosen() {
        return attempt(device, graph).map_err(|r| r.error(device));
    }
    match attempt(Device::Gpu, graph) {
        Ok(compiled) => {
            settle(Device::Gpu, None);
            Ok(compiled)
        }
        Err(refusal) => {
            settle(Device::Cpu, Some(&refusal.reason()));
            attempt(Device::Cpu, graph).map_err(|r| r.error(Device::Cpu))
        }
    }
}

/// Pin this process to the graphics device, for a harness whose subject is that device's
/// work.
///
/// The product decides its device once, at the first compile. An instrument that measured
/// whatever the run chose would change its own subject, and one that continued on the
/// processor would price another device's work — so a harness pins its device here: a
/// switch that names a device refuses by name, and a device that was already settled
/// before the pin is a panic rather than a measurement of the wrong subject. A pinned
/// process also drops the refusal's way-out hint, which names a switch the pin refuses.
#[doc(hidden)]
pub fn pin_gpu() {
    match switch() {
        Ok(Asked::Auto) => {}
        Ok(_) => panic!("{VAR} names the device; unset it to run this pin"),
        Err(message) => panic!("{message}"),
    }
    PINNED.store(true, Ordering::Relaxed);
    let _ = CHOSEN.set(Device::Gpu);
    assert_eq!(
        chosen(),
        Some(Device::Gpu),
        "the device was settled before this pin, so this instrument did not pin its subject"
    );
}

/// Settle the run's device, saying it once per process: the device, and — for a forced
/// choice or a fallback — why it is that one. The first call is the decision; a later one
/// finds it made and says nothing.
fn settle(device: Device, reason: Option<&str>) {
    if CHOSEN.set(device).is_err() {
        return;
    }
    match reason {
        Some(reason) => eprintln!("{}device {} — {reason}", prefix(), name(device)),
        None => eprintln!("{}device {}", prefix(), name(device)),
    }
}

/// The prefix a run that never resolved — a library, a unit test — gets instead of a blank
/// line. [`resolve`] sets the binary's own.
fn prefix() -> &'static str {
    PREFIX
        .get()
        .map(String::as_str)
        .unwrap_or("mahbot-inference: ")
}

/// Why one compile did not produce a graph: the device is not there, or the backend
/// refused the graph — by panicking, which is how a backend reports a refusal.
enum Refusal {
    Absent,
    Refused(String),
}

impl Refusal {
    /// The reason a fallback's line states.
    fn reason(&self) -> String {
        match self {
            Self::Absent => ABSENT.to_string(),
            Self::Refused(message) => {
                format!("the graphics layer refused the first graph: {message}")
            }
        }
    }

    fn error(&self, device: Device) -> anyhow::Error {
        let name = name(device);
        let message = match self {
            Self::Absent => format!("no usable {name} backend on this host"),
            Self::Refused(message) => {
                format!("the {name} backend could not compile this graph: {message}")
            }
        };
        if device == Device::Cpu {
            return anyhow!("{message}");
        }
        anyhow!("{message}{}", way_out(PINNED.load(Ordering::Relaxed)))
    }
}

/// One compile on one device, its refusal as a value instead of a panic. Obtaining the
/// device is inside the guard: creating it is the graphics path's first step, and a
/// backend that dies there dies inside this bracket rather than at the device layer.
fn attempt(device: Device, graph: &Graph) -> std::result::Result<CompiledGraph, Refusal> {
    let compiled = catch_unwind(AssertUnwindSafe(|| {
        if !rlx_runtime::is_available(device) {
            return Err(Refusal::Absent);
        }
        // The compile consumes its graph, so a fallback recompiles from a copy: this is
        // the copy, taken once per compile rather than held beside every attempt.
        Ok(Session::new(device).compile(graph.clone()))
    }));
    match compiled {
        Ok(compiled) => compiled,
        Err(payload) => Err(Refusal::Refused(panic_message(payload.as_ref()))),
    }
}

/// A panic that escaped a row or a measurement phase, as the error that reports it. The
/// bracket is the caller's — it is the widest thing a row or a phase can be wrapped in —
/// so the wording lives here, where both of them read it. It names where the run stopped,
/// not who is at fault: a backend's own `REFUSED` message carries the cause. A run that
/// stopped anywhere but the processor also names the way out, so a refusal after the
/// decision is not a dead end — and a run that panicked before it decided says that,
/// rather than naming a device it never chose.
pub(crate) fn stopped(payload: &(dyn Any + Send)) -> anyhow::Error {
    let message = panic_message(payload);
    match chosen() {
        Some(Device::Cpu) => anyhow!("the run stopped on the cpu device: {message}"),
        Some(device) => {
            anyhow!(
                "the run stopped on the {} device: {message}{}",
                name(device),
                way_out(PINNED.load(Ordering::Relaxed))
            )
        }
        None => anyhow!(
            "the run stopped before it chose a device: {message}{}",
            way_out(PINNED.load(Ordering::Relaxed))
        ),
    }
}

/// A caught panic's message: what a backend's refusal states, and what [`stopped`] reports.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
        })
        .unwrap_or_else(|| "no message".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_names_three_choices_and_refuses_everything_else() {
        assert_eq!(Asked::parse(""), Ok(Asked::Auto));
        assert_eq!(Asked::parse(" auto "), Ok(Asked::Auto));
        assert_eq!(Asked::parse("GPU"), Ok(Asked::Gpu));
        assert_eq!(Asked::parse("cpu"), Ok(Asked::Cpu));
        let refused = Asked::parse("tpu").unwrap_err();
        assert!(refused.contains("MAHBOT_DEVICE=tpu"), "{refused}");
    }

    #[test]
    fn a_forced_or_unpriced_device_writes_no_record() {
        assert_eq!(reasons(Asked::Auto, None), None);
        assert_eq!(reasons(Asked::Auto, Some(Device::Gpu)), None);
        assert_eq!(reasons(Asked::Auto, Some(Device::Cpu)), Some(FALLBACK));
        assert_eq!(reasons(Asked::Gpu, Some(Device::Gpu)), Some(FORCED));
        assert_eq!(reasons(Asked::Cpu, Some(Device::Cpu)), Some(FORCED));
    }

    /// The two lines a device-layer refusal produces: the fallback's reason (what the run
    /// says when it continues on the processor) and the settled run's error, which names
    /// the way out on every device but the processor — and never in a pinned process,
    /// where the way out is a switch the pin itself refuses.
    #[test]
    fn a_refusal_states_what_it_refused_and_names_the_way_out() {
        assert_eq!(Refusal::Absent.reason(), ABSENT);
        let absent = Refusal::Absent.error(Device::Cpu).to_string();
        assert!(!absent.contains(VAR), "{absent}");

        let refused = Refusal::Refused("boom".into());
        assert_eq!(
            refused.reason(),
            "the graphics layer refused the first graph: boom"
        );
        assert_eq!(
            refused.error(Device::Cpu).to_string(),
            "the cpu backend could not compile this graph: boom"
        );
        assert_eq!(way_out(false), WAY_OUT);
        assert_eq!(way_out(true), "");
    }
}
