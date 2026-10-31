// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What every measured row needs: a stopwatch, the process's memory as the
//! platform accounts for it, an answer's digest, and the labels the numbers are
//! printed with. Nothing here decides **what** is measured.

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

/// The process's physical footprint: `(now, lifetime maximum)`, as the platform
/// accounts for it, or `None` where the platform supplies no such reading.
///
/// The resident set cannot see memory allocated for the GPU on a unified-memory
/// machine, so the recorded figures come from here and fall back to
/// [`resident_rss`] only where the platform has no such reading. macOS answers
/// through `proc_pid_rusage`: `ri_phys_footprint` and
/// `ri_lifetime_max_phys_footprint`, which no later drop can lower.
#[cfg(target_os = "macos")]
pub fn footprint() -> Option<(u64, u64)> {
    let mut info = std::mem::MaybeUninit::<RusageInfoV4>::zeroed();
    // SAFETY: `info` is a properly aligned `rusage_info_v4`, the flavour that
    // fills exactly that layout, and the call only writes into it.
    let rc =
        unsafe { proc_pid_rusage(std::process::id() as i32, RUSAGE_INFO_V4, info.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: a zero return means the call filled the whole structure.
    let info = unsafe { info.assume_init() };
    Some((info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint))
}

#[cfg(not(target_os = "macos"))]
pub fn footprint() -> Option<(u64, u64)> {
    None
}

/// `proc_pid_rusage`'s `rusage_info_v4`: the layout of `sys/resource.h`, kept
/// whole because the call writes all of it and reads none of its padding.
#[cfg(target_os = "macos")]
#[repr(C)]
struct RusageInfoV4 {
    ri_uuid: [u8; 16],
    ri_user_time: u64,
    ri_system_time: u64,
    ri_pkg_idle_wkups: u64,
    ri_interrupt_wkups: u64,
    ri_pageins: u64,
    ri_wired_size: u64,
    ri_resident_size: u64,
    ri_phys_footprint: u64,
    ri_proc_start_abstime: u64,
    ri_proc_exit_abstime: u64,
    ri_child_user_time: u64,
    ri_child_system_time: u64,
    ri_child_pkg_idle_wkups: u64,
    ri_child_interrupt_wkups: u64,
    ri_child_pageins: u64,
    ri_child_elapsed_abstime: u64,
    ri_diskio_bytesread: u64,
    ri_diskio_byteswritten: u64,
    ri_cpu_time_qos_default: u64,
    ri_cpu_time_qos_maintenance: u64,
    ri_cpu_time_qos_background: u64,
    ri_cpu_time_qos_utility: u64,
    ri_cpu_time_qos_legacy: u64,
    ri_cpu_time_qos_user_initiated: u64,
    ri_cpu_time_qos_user_interactive: u64,
    ri_billed_system_time: u64,
    ri_serviced_system_time: u64,
    ri_logical_writes: u64,
    ri_lifetime_max_phys_footprint: u64,
    ri_instructions: u64,
    ri_cycles: u64,
    ri_billed_energy: u64,
    ri_serviced_energy: u64,
    ri_interval_max_phys_footprint: u64,
    ri_runnable_time: u64,
}

#[cfg(target_os = "macos")]
const RUSAGE_INFO_V4: i32 = 4;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
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

/// The two memory figures one row carries, from the platform's accounting: the
/// figure at rest and the process's lifetime peak.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub now: Option<u64>,
    pub peak: Option<u64>,
    /// The physical footprint where the platform supplies it, the resident set
    /// where it does not.
    pub footprint: bool,
}

impl Memory {
    /// The reading at rest. [`Memory::finish`] adds the lifetime peak once the
    /// timed calls are done, so a warm call's high-water is included: the peak is
    /// only ever read after them.
    #[must_use]
    pub fn take() -> Self {
        match footprint() {
            Some((now, _)) => Self {
                now: Some(now),
                peak: None,
                footprint: true,
            },
            None => Self {
                now: resident_rss(),
                peak: None,
                footprint: false,
            },
        }
    }

    /// The same reading with its lifetime peak, taken after the timed work. The
    /// peak comes from the basis [`Memory::take`] recorded, so one row never mixes
    /// the platform's footprint with the resident set.
    #[must_use]
    pub fn finish(self) -> Self {
        Self {
            peak: if self.footprint {
                footprint().map(|(_, peak)| peak)
            } else {
                peak_rss()
            },
            ..self
        }
    }

    #[must_use]
    pub fn now_text(&self) -> String {
        shown(self.now)
    }

    #[must_use]
    pub fn peak_text(&self) -> String {
        shown(self.peak)
    }
}

/// The basis a table's memory figures were taken on — the platform's physical
/// footprint, the resident set where the platform supplied none, or both across
/// the rows. It is read off the rows themselves, so a caption describes the
/// figures under it rather than the machine the caption is printed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Basis {
    Footprint,
    Resident,
    Mixed,
}

impl Basis {
    /// The basis the rows carrying these `footprint` flags were taken on. No rows
    /// means no figure for a caption to qualify — a table nobody filled has nothing a
    /// fallback could describe — so the platform's own reading is the one named.
    #[must_use]
    pub(crate) fn of(rows: impl IntoIterator<Item = bool>) -> Self {
        let mut flags = rows.into_iter();
        let Some(first) = flags.next() else {
            return Self::Footprint;
        };
        if flags.all(|flag| flag == first) {
            if first {
                Self::Footprint
            } else {
                Self::Resident
            }
        } else {
            Self::Mixed
        }
    }

    /// What the figures are, in the one sentence every caption that carries them
    /// uses: the two readings count different things, and a platform without the
    /// footprint falls back to the resident set.
    #[must_use]
    pub(crate) fn caption(self) -> &'static str {
        match self {
            Self::Footprint => {
                "Memory is the process's physical footprint as the platform accounts for \
                 it (`phys_footprint`), which is the only reading that sees memory allocated for \
                 the GPU: the resident set counts different things and is not comparable. \
                 Resting is taken once the cold call has loaded and built what it needs and \
                 before the warm call; peak is the process's lifetime maximum."
            }
            Self::Resident => {
                "Memory is the process's resident set, because this platform supplies no \
                 physical-footprint reading: resting is the set at the moment the cold call has \
                 loaded and built what it needs and before the warm call, and peak is the \
                 kernel's lifetime maximum of the same set. The resident set counts a different \
                 thing from the footprint, so these figures are not comparable with one taken \
                 where the platform supplies it."
            }
            Self::Mixed => {
                "Memory is the process's physical footprint on the rows the platform supplied \
                 one, and the resident set on the rows it did not: the two count different things \
                 and are not comparable with each other. Resting is taken once the cold call has \
                 loaded and built what it needs and before the warm call; peak is the process's \
                 lifetime maximum."
            }
        }
    }
}

/// One byte count as the record writes it, and `-` for a figure nobody took: a
/// zero would read as the reading of a live process.
pub fn shown(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "-".to_string(), human)
}

/// One byte count as the record writes it.
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

/// Deterministic pseudo-random f32 in `[-1, 1)` — a test's own input; the value
/// is reported as read.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The platform's footprint is a live reading, and its lifetime maximum is at
    /// least the current one. macOS supplies both; elsewhere the fallback is the
    /// resident set, which the caption names.
    #[test]
    fn the_memory_reading_comes_from_the_platform() {
        match footprint() {
            Some((now, peak)) => {
                assert!(now > 1024 * 1024, "a live process holds bytes: {now}");
                assert!(peak >= now, "the maximum cannot trail the current reading");
                assert!(Basis::Footprint.caption().contains("physical footprint"));
            }
            None => {
                assert!(Basis::Resident.caption().contains("resident set"));
            }
        }
    }

    /// The caption describes the basis its rows were taken on, so a table that
    /// mixed the two says so.
    #[test]
    fn the_basis_folds_its_rows() {
        assert_eq!(Basis::of([true, true]), Basis::Footprint);
        assert_eq!(Basis::of([false, false]), Basis::Resident);
        assert_eq!(Basis::of([true, false]), Basis::Mixed);
        assert_ne!(Basis::Mixed.caption(), Basis::Footprint.caption());
        assert_ne!(Basis::Mixed.caption(), Basis::Resident.caption());
    }

    /// [`human`] is the one byte-count rendering the records use: a whole count in
    /// the largest unit that keeps it whole.
    #[test]
    fn a_byte_count_renders_in_one_place() {
        assert_eq!(human(512), "512B");
        assert_eq!(human(2 * 1024), "2KiB");
        assert_eq!(human(3 * 1024 * 1024 + 512 * 1024), "3.5MiB");
        assert_eq!(human(3 * 1024 * 1024 * 1024), "3.00GiB");
    }
}
