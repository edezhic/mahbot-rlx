// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What every cell line needs: a stopwatch, the run's resident set, the answer's
//! digest, and the two labels (`human`, `sci`) the numbers are printed with.
//! Nothing here decides **what** is measured.

use std::time::Instant;

pub struct Stopwatch(Instant);

impl Stopwatch {
    pub fn start() -> Self {
        Self(Instant::now())
    }
    pub fn ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1e3
    }
}

/// The process's true high-water resident set, maintained by the kernel.
/// On macOS `ru_maxrss` is bytes; on Linux it is KiB. This is the peak over the
/// whole process lifetime, not the RSS left at the end of the run.
///
/// `None` when the kernel cannot be asked: a zero would read as the reading of a live
/// process, and a peak nobody measured may not be printed as one.
pub fn peak_rss() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc_rusage>::zeroed();
        let rc = unsafe { getrusage(0, usage.as_mut_ptr()) };
        if rc == 0 {
            let usage = unsafe { usage.assume_init() };
            let raw = usage.ru_maxrss as u64;
            return Some(if cfg!(target_os = "macos") {
                raw
            } else {
                raw * 1024
            });
        }
    }
    None
}

/// [`peak_rss`] as the raw byte count a `process_peak_bytes=` line states, or `-`
/// where the kernel did not answer — the rule a reading nobody took prints `-`,
/// never a zero, stated once rather than restated by each caller.
pub fn peak_text() -> String {
    peak_rss().map_or_else(|| "-".to_string(), |bytes| bytes.to_string())
}

/// [`peak_rss`] in MiB: the unit the wake-word bench's console and its record
/// state a peak in, so the two cannot report the same figure differently.
pub fn peak_mib() -> String {
    peak_rss().map_or_else(|| "-".to_string(), mib)
}

/// A byte count in MiB, the bench's unit for a peak residency.
pub fn mib(bytes: u64) -> String {
    format!("{} MiB", bytes / (1024 * 1024))
}

#[cfg(unix)]
#[repr(C)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    ru_ixrss: i64,
    ru_idrss: i64,
    ru_isrss: i64,
    ru_minflt: i64,
    ru_majflt: i64,
    ru_nswap: i64,
    ru_inblock: i64,
    ru_oublock: i64,
    ru_msgsnd: i64,
    ru_msgrcv: i64,
    ru_nsignals: i64,
    ru_nvcsw: i64,
    ru_nivcsw: i64,
}

#[cfg(unix)]
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

/// Resident set size right now, in bytes (the fallback / diagnostic reading).
///
/// `None` when the platform's process query fails or answers nothing usable: a
/// zero here would read as a measurement of a live process, which no run has.
pub fn resident_rss() -> Option<u64> {
    let pid = std::process::id().to_string();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
}
/// Return the pages freed so far to the OS. macOS keeps them in the malloc
/// zones, so a process that drops one phase and builds the next can peak at the
/// sum of the phases; the pressure call hands those pages back and the next
/// phase allocates from what the OS has. Platforms that return memory on free
/// do nothing here.
pub fn release_free() {
    #[cfg(target_os = "macos")]
    unsafe {
        // Twice, with a short settle: the first call starts returning pages, the
        // second catches what the allocator released in between. Without this the
        // post-run RSS depends on which phase happened to trigger the last free.
        malloc_zone_pressure_relief(core::ptr::null_mut(), 0);
        std::thread::sleep(std::time::Duration::from_millis(20));
        malloc_zone_pressure_relief(core::ptr::null_mut(), 0);
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn malloc_zone_pressure_relief(zone: *mut core::ffi::c_void, goal: usize) -> usize;
}

pub fn human(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if mib >= 1.0 {
        format!("{mib:.1}MiB")
    } else if bytes >= 1024 {
        format!("{:.0}KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

/// An exact count in space-grouped thousands (`1 101 234 567`), the way
/// `MEASUREMENTS.md` writes one. A rate printed beside it is that exact
/// division, where [`human`]'s rounding would leave a per-cent of slack.
#[must_use]
pub fn grouped(bytes: u64) -> String {
    let digits = bytes.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}

/// Milliseconds, in the unit the reports use.
pub fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{ms:.1}ms")
    } else {
        format!("{:.0}µs", ms * 1000.0)
    }
}

pub fn sci(v: f32) -> String {
    format!("{v:.3e}")
}

/// Deterministic pseudo-random f32 in `[-1, 1)` — the stand's own input; the
/// value is reported as read.
pub fn pseudo_rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

pub fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |m, v| m.max(v.abs()))
}

pub fn non_finite(values: &[f32]) -> usize {
    values.iter().filter(|v| !v.is_finite()).count()
}

/// FNV-1a 64 over the answer's f32 bits: an answer is compared by a word rather
/// than by a spread, so a kernel change inside a step must leave it unmoved.
pub fn digest(values: &[f32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// The machine's load, so a log records what else was running when it was taken.
pub fn load_average() -> String {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            s.replace(['{', '}'], "")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|| "unknown".into())
}

/// The thermal state the numbers were taken under, as `scripts/check-throttle.sh`
/// reads it from `pmset -g therm`: the limits the platform printed, naming only
/// those, or `unknown` where it printed none of them. `None` where the reading
/// itself is unavailable, so the record can say "no reading" instead of claiming
/// a cool run.
pub fn thermal() -> Option<String> {
    let text = std::process::Command::new("pmset")
        .args(["-g", "therm"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())?;
    let limit = |key: &str| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(key))
            .and_then(|rest| rest.split('=').nth(1)?.trim().parse::<u32>().ok())
    };
    let (speed, sched) = (limit("CPU_Speed_Limit"), limit("CPU_Scheduler_Limit"));
    let mut printed = Vec::new();
    if let Some(speed) = speed {
        printed.push(format!("CPU_Speed_Limit {speed}"));
    }
    if let Some(sched) = sched {
        printed.push(format!("CPU_Scheduler_Limit {sched}"));
    }
    if printed.is_empty() {
        return Some("unknown (pmset -g therm printed no CPU limit line)".into());
    }
    let state = if speed.is_some_and(|v| v < 100) || sched.is_some_and(|v| v < 100) {
        "throttled"
    } else {
        "not throttled"
    };
    Some(format!("{state} ({})", printed.join(", ")))
}

/// The `RLX_*` and `STAND_*` keys this process carries, sorted. The first are the
/// fork's route levers, the second the stand's own input/window levers; both are
/// printed with the numbers they changed, so a cell is never read apart from the
/// form it was taken on.
pub fn keys() -> String {
    let mut v: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("RLX_") || k.starts_with("STAND_"))
        .map(|(k, value)| format!("{k}={value}"))
        .collect();
    v.sort();
    if v.is_empty() {
        "none".into()
    } else {
        v.join(",")
    }
}
