// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What every measured row needs: a stopwatch, the process's memory as the
//! platform accounts for it (and the sampler that observes the GPU's share of it),
//! an answer's digest, and the labels the numbers are printed with. Nothing here
//! decides **what** is measured.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

pub struct Stopwatch(Instant);

impl Stopwatch {
    pub fn start() -> Self {
        Self(Instant::now())
    }
    pub fn ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1e3
    }
}

/// One graph's construction and compile, timed into `RLX_WGPU_COMPILE_REPORT` beside the
/// backend's own line ([`rlx_wgpu::compile_report`]): `build` is the flow's graph
/// construction and `compile` is everything from HIR conversion to the last parameter
/// upload. The difference between the two lines is the fusion pipeline and the upload —
/// the backend's phases are the second line's. Without the switch this is
/// [`crate::graphs::Compiled::new`] with the build timed. The device is fixed to
/// [`rlx_runtime::Device::Gpu`] rather than passed in: nothing in this crate compiles for
/// another backend.
pub(crate) fn compiled(
    kind: &str,
    build: impl FnOnce() -> anyhow::Result<rlx_flow::BuiltModel>,
) -> anyhow::Result<crate::graphs::Compiled> {
    let t = Stopwatch::start();
    let built = build()?;
    let build_ms = t.ms();
    let t = Stopwatch::start();
    let compiled = crate::graphs::Compiled::new(rlx_runtime::Device::Gpu, built)?;
    if rlx_wgpu::compile_report::enabled() {
        let compile_ms = t.ms();
        eprintln!("[compile {kind}] build={build_ms:.2}ms compile={compile_ms:.2}ms");
    }
    Ok(compiled)
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

/// The platform's page-based accounting of this process at one instant: the
/// physical footprint, its lifetime maximum, and the part of the footprint the
/// platform tags as graphics.
#[derive(Clone, Copy)]
struct Vm {
    footprint: u64,
    /// The same ledger's lifetime maximum: no later drop can lower it. `None` where
    /// the platform's accounting is too old to carry the ledger's peak.
    peak: Option<u64>,
    /// Graphics-tagged pages inside the footprint, compressed pages included — the
    /// platform's own itemisation of the GPU's share of it. `None` where the
    /// platform's accounting is too old to carry the tag ledgers.
    gpu: Option<u64>,
}

/// The process's physical footprint: `(now, lifetime maximum)`, as the platform
/// accounts for it, or `None` where the platform supplies no such reading.
///
/// The resident set cannot see memory allocated for the GPU on a unified-memory
/// machine, so the recorded figures come from here and fall back to
/// [`resident_rss`] only where the platform has no such reading. macOS answers
/// through `task_info(TASK_VM_INFO)`: `phys_footprint` and
/// `ledger_phys_footprint_peak`, which no later drop can lower.
#[cfg(target_os = "macos")]
pub fn footprint() -> Option<(u64, u64)> {
    vm().and_then(|vm| vm.peak.map(|peak| (vm.footprint, peak)))
}

#[cfg(not(target_os = "macos"))]
pub fn footprint() -> Option<(u64, u64)> {
    None
}

/// [`footprint`] with the platform's split of it: `ledger_tag_graphics_footprint`
/// plus its compressed twin are the graphics-tagged pages credited into the
/// footprint, and they nest inside it by construction.
#[cfg(target_os = "macos")]
fn vm() -> Option<Vm> {
    let mut info = TaskVmInfo::default();
    let mut count = (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<i32>()) as u32;
    // SAFETY: `info` is zeroed and `count` states its size in 32-bit words, the unit
    // `task_info` writes in; the call writes at most that many words and reports how
    // many it wrote.
    let rc = unsafe {
        task_info(
            mach_task_self_,
            TASK_VM_INFO,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    if rc != 0 || count < VM_INFO_PHYS_FOOTPRINT_COUNT {
        return None;
    }
    // The peak and the graphics ledgers share the revision the tags arrived with.
    let (peak, gpu) = if count >= VM_INFO_TAGS_COUNT {
        (
            u64::try_from(info.ledger_phys_footprint_peak).ok(),
            u64::try_from(
                info.ledger_tag_graphics_footprint + info.ledger_tag_graphics_footprint_compressed,
            )
            .ok(),
        )
    } else {
        (None, None)
    };
    Some(Vm {
        footprint: info.phys_footprint,
        peak,
        gpu,
    })
}

#[cfg(not(target_os = "macos"))]
fn vm() -> Option<Vm> {
    None
}

/// `task_vm_info_data_t`'s prefix, as `mach/task_info.h` lays it out: the fields
/// this crate reads end at the graphics ledgers, and the kernel's own tail is kept
/// as bytes so the buffer the call writes into is its full size. The header packs
/// the struct to four bytes, which is why it is `packed(4)` here too: a natural
/// layout rounds 372 bytes up to 376 and asks the kernel for a word more than it
/// has. The kernel fills only the fields its revision has, which `count` reports.
#[cfg(target_os = "macos")]
#[repr(C, packed(4))]
#[derive(Clone, Copy, Default)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    min_address: u64,
    max_address: u64,
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_novolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
    limit_bytes_remaining: u64,
    decompressions: i32,
    /// `ledger_swapins`, `ledger_tag_neural_nofootprint_total` and
    /// `ledger_tag_neural_nofootprint_peak` — three eight-byte fields this crate does
    /// not read.
    tail: [u8; 24],
}

#[cfg(target_os = "macos")]
const TASK_VM_INFO: u32 = 22;
/// The count at which `phys_footprint` is present (`TASK_VM_INFO_REV1_COUNT`).
#[cfg(target_os = "macos")]
const VM_INFO_PHYS_FOOTPRINT_COUNT: u32 = 38;
/// The count at which the tag ledgers are present (`TASK_VM_INFO_REV3_COUNT`).
#[cfg(target_os = "macos")]
const VM_INFO_TAGS_COUNT: u32 = 84;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// The calling task's own port: the kernel's `mach_task_self_` global.
    static mach_task_self_: u32;
    fn task_info(target_task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
}

/// How often the GPU sampler reads the platform's split. The shortest row's warm
/// call is tens of milliseconds, so this samples it many times over; each read is one
/// `task_info` call.
const GPU_SAMPLE_PERIOD: Duration = Duration::from_millis(1);

/// The process's GPU peak, once [`sample_gpu`] has started it.
static GPU_SAMPLER: OnceLock<Sampler> = OnceLock::new();

/// Start the GPU sampler: the highest graphics-tagged reading the platform reports,
/// from this call until [`Memory::finish`] stops it. A row's process calls this as
/// early as it can — before the arguments are read and long before the cold call —
/// because the total peak it sits beside is a process-lifetime maximum. Idempotent.
///
/// The sampler and the peak it takes are **process-wide**: one window, one reading,
/// consumed by the first [`Memory::finish`] on a footprint basis. That is why the
/// stand and the corpus measure one row per process — a process measuring two rows
/// would give the second the first one's window.
pub(crate) fn sample_gpu() {
    // Only macOS answers with a split; elsewhere the thread could never sample one.
    if cfg!(target_os = "macos") {
        GPU_SAMPLER.get_or_init(Sampler::start);
    }
}

/// What the sampler thread and its owner share: the stop flag and the highest
/// graphics-tagged reading so far, absent until the first one is taken.
#[derive(Default)]
struct State {
    stop: bool,
    peak: Option<u64>,
}

struct Sampler {
    state: Arc<Mutex<State>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Sampler {
    fn start() -> Self {
        let state = Arc::new(Mutex::new(State::default()));
        let thread = {
            let state = Arc::clone(&state);
            std::thread::Builder::new()
                .name("gpu-peak".into())
                .spawn(move || {
                    loop {
                        let mut state = lock(&state);
                        if state.stop {
                            break;
                        }
                        if let Some(gpu) = vm().and_then(|vm| vm.gpu) {
                            state.peak = Some(state.peak.map_or(gpu, |peak| peak.max(gpu)));
                        }
                        drop(state);
                        std::thread::sleep(GPU_SAMPLE_PERIOD);
                    }
                })
                .ok()
        };
        Self {
            state,
            thread: Mutex::new(thread),
        }
    }

    /// Stop the thread and give back the highest reading it took, or `None` where it
    /// never saw a GPU figure — never a zero nobody measured. Idempotent: a second
    /// call joins nothing and answers the same reading.
    fn finish(&self) -> Option<u64> {
        lock(&self.state).stop = true;
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
        lock(&self.state).peak
    }
}

/// A lock whose contents survive another thread's panic: a sampler that died must
/// not turn a reading it already took into a panic here.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The GPU peak the sampler observed, stopping it: `None` where no sampler ran or
/// the platform never reported a GPU figure.
fn gpu_peak() -> Option<u64> {
    GPU_SAMPLER.get().and_then(Sampler::finish)
}

/// The memory figures one row carries, from the platform's accounting: the figure
/// at rest and the process's lifetime peak, each a pair — the system's share and
/// the GPU's.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    /// The whole figure at rest: the footprint where the platform supplies it, the
    /// resident set where it does not.
    pub now: Option<u64>,
    pub peak: Option<u64>,
    /// Graphics-tagged pages inside `now`, from the same platform call, so the pair
    /// sums to it. `None` where the basis cannot see GPU pages.
    pub gpu_now: Option<u64>,
    /// The highest graphics-tagged reading over the window ending at `peak`: the
    /// sampler's maximum, with the split read at `peak`'s own instant folded in, so a
    /// run that started no sampler still carries the reading taken there. The window is
    /// the process's, not the row's: every `finish` in one process answers the same
    /// reading. `None` where the basis cannot see GPU pages, never a zero nobody
    /// measured.
    pub gpu_peak: Option<u64>,
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
        match vm() {
            Some(vm) => Self {
                now: Some(vm.footprint),
                peak: None,
                gpu_now: vm.gpu,
                gpu_peak: None,
                footprint: true,
            },
            None => Self {
                now: resident_rss(),
                peak: None,
                gpu_now: None,
                gpu_peak: None,
                footprint: false,
            },
        }
    }

    /// The same reading with its lifetime peak, taken after the timed work, and the
    /// GPU peak the sampler observed over the same lifetime. The peaks come from the
    /// basis [`Memory::take`] recorded, so one row never mixes the platform's
    /// footprint with the resident set.
    #[must_use]
    pub fn finish(self) -> Self {
        if !self.footprint {
            // Nothing to pair a GPU reading with here, and the sampler is left alone:
            // its window may still serve whatever this process measures next.
            return Self {
                peak: peak_rss(),
                gpu_peak: None,
                ..self
            };
        }
        // The sampler is stopped and joined first, so no sample lands after the peak is
        // read; the split from that read is folded in, so the window still reaches the
        // peak's instant. Nothing is observed between the last sample and it — that gap
        // is what makes this an observed maximum.
        let sampled = gpu_peak();
        let vm = vm();
        Self {
            peak: vm.and_then(|vm| vm.peak),
            gpu_peak: sampled.max(vm.and_then(|vm| vm.gpu)),
            ..self
        }
    }

    /// The system's share at rest: the whole figure minus the GPU-tagged pages read
    /// with it, so the pair always sums to the figure the surface carries.
    #[must_use]
    pub fn system_now_text(&self) -> String {
        shown(
            self.now
                .map(|total| total - pair_gpu(self.gpu_now, self.now).unwrap_or(0)),
        )
    }

    /// The system's share at peak: the lifetime maximum minus the highest GPU
    /// reading — a subtraction of two maxima, not a peak of its own.
    #[must_use]
    pub fn system_peak_text(&self) -> String {
        shown(
            self.peak
                .map(|total| total - pair_gpu(self.gpu_peak, self.peak).unwrap_or(0)),
        )
    }

    #[must_use]
    pub fn gpu_now_text(&self) -> String {
        measured(pair_gpu(self.gpu_now, self.now))
    }

    #[must_use]
    pub fn gpu_peak_text(&self) -> String {
        measured(pair_gpu(self.gpu_peak, self.peak))
    }
}

/// The GPU half of a pair, held inside its total: the platform's ledger reads are
/// not atomic, so a torn reading can put it a page above, and a pair that does not
/// sum would be worse than a page of rounding.
fn pair_gpu(gpu: Option<u64>, total: Option<u64>) -> Option<u64> {
    gpu.map(|gpu| total.map_or(gpu, |total| gpu.min(total)))
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

    /// What the figures are and what their pair means, as the caption under a table
    /// states it: the accounting each half comes from, what it includes and cannot
    /// see, how the two were taken, and what the pair means on this machine. Every
    /// arm is composed from the same sentences, so a mixed table's caption cannot
    /// drift from the uniform ones'.
    #[must_use]
    pub(crate) fn caption(self) -> String {
        match self {
            Self::Footprint => [
                FOOTPRINT_BASE,
                PAIR,
                OUTSIDE_PAIR,
                PEAK,
                SYSTEM_AT_PEAK,
                UNIFIED,
            ]
            .join(" "),
            Self::Resident => [RESIDENT_BASE, RESIDENT_GPU, BASES_DIFFER].join(" "),
            Self::Mixed => [
                MIXED_BASE,
                PAIR,
                OUTSIDE_PAIR,
                RESIDENT_GPU,
                PEAK,
                SYSTEM_AT_PEAK,
                UNIFIED,
            ]
            .join(" "),
        }
    }
}

/// The physical footprint's own definition: what the base counts and what lies
/// outside it.
const FOOTPRINT_BASE: &str = "Memory is the process's physical footprint as the platform \
     accounts for it (`phys_footprint`), the only reading that sees memory allocated for the \
     GPU; it counts dirty pages, page tables and IOKit mappings, and clean file-backed pages — \
     a read-only model mapping, say — are outside it.";

/// How the pair is read, and what each half holds.
const PAIR: &str = "On a footprint row, the resting figure is a pair read from one \
     `task_info(TASK_VM_INFO)` call, so the two halves sum to the footprint: system is the \
     footprint minus the pages the platform tags as graphics, which also holds IOKit mappings, \
     compressed pages, page tables and any pages tagged for something other than graphics, and \
     is not a CPU-only reading; GPU is those graphics-tagged pages, which include buffers the \
     host can also map and staging allocations, and is not the GPU's working set.";

/// What the platform books outside both halves.
const OUTSIDE_PAIR: &str = "Graphics-tagged pages the platform books outside the footprint \
     (`ledger_tag_graphics_nofootprint`) are in neither number, and GPU memory the platform \
     books as an IOKit mapping without a graphics tag cannot be separated from the system \
     figure.";

/// When the two readings are taken, and what the peak beside the maximum is.
const PEAK: &str = "Resting is taken once the cold call has returned and before the warm \
     call; peak is that figure's lifetime maximum. On a footprint row, the GPU peak beside it \
     is the highest graphics-tagged reading a sampler thread took every millisecond or so from \
     the first moment the row's process could start it until that maximum was read — an \
     observed maximum, so a spike before then or between two samples is not in it, and the two \
     halves were not taken at the same instant.";

/// What the system figure at peak is — a derived column — and why it can sit below
/// the one at rest.
const SYSTEM_AT_PEAK: &str = "The system figure at peak is derived, not measured: it is the \
     lifetime maximum of the whole figure minus the highest GPU reading, a subtraction of two \
     maxima, and is not a peak of anything on its own. A system figure at peak below the one at \
     rest is therefore legitimate — the GPU share can grow more than the total between rest and \
     peak — and a GPU spike the sampler missed makes the derived figure too large, while a GPU \
     reading taken at another instant can make it too small.";

/// What the pair means here, and what the same reading would be elsewhere.
const UNIFIED: &str = "Memory is unified on this machine, so where both halves are measured \
     they divide one pool rather than two budgets; on a machine with separate video memory this \
     reading would report the host-visible graphics share, not that machine's video memory.";

/// The resident set's own definition, for the platform that supplies no footprint.
const RESIDENT_BASE: &str = "Memory is the process's resident set, because this platform \
     supplies no physical-footprint reading: resting is the set at the moment the cold call has \
     returned and before the warm call, and peak is the kernel's lifetime maximum of the same \
     set.";

/// Why a resident row has no GPU half. The wake bench's record states the same
/// sentence, and its GPU cell carries the clause this one opens with.
pub(crate) const RESIDENT_GPU: &str = "The resident set cannot see pages allocated for the \
     GPU, so a resident row's GPU cell is not measured — not zero, and not the system figure \
     repeated — and there the two do not split one pool.";

/// The resident set and the footprint are different quantities.
const BASES_DIFFER: &str = "The resident set counts a different thing from the footprint, so \
     these figures are not comparable with one taken where the platform supplies it.";

/// A table that holds both bases, which are not comparable with each other.
const MIXED_BASE: &str = "Memory is the process's physical footprint on the rows the platform \
     supplied one and the resident set on the rows it did not: the two count different things \
     and are not comparable with each other.";

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

/// The words a cell carries when nobody measured the figure. A measured zero is a
/// reading, and [`human`] renders it as `0B`.
pub const NOT_MEASURED: &str = "not measured";

/// One GPU figure as the record writes it: a byte count — `0B` included, because a
/// measured zero is a reading — or [`NOT_MEASURED`].
fn measured(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| NOT_MEASURED.to_string(), human)
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

    /// The platform's footprint is a live reading and its lifetime maximum is at least
    /// the current one. macOS supplies both plus the graphics split; elsewhere the
    /// fallback is the resident set, which the caption names. The split is not asserted
    /// against the footprint here: [`pair_gpu`] exists because a torn ledger read can
    /// put it a page above, and a test that demanded otherwise would flake on the
    /// condition the pair is built to absorb.
    #[test]
    fn the_memory_reading_comes_from_the_platform() {
        match vm() {
            Some(vm) => {
                assert!(
                    vm.footprint > 1024 * 1024,
                    "a live process holds bytes: {}",
                    vm.footprint
                );
                if let Some(peak) = vm.peak {
                    assert!(
                        peak >= vm.footprint,
                        "the maximum cannot trail the current reading"
                    );
                }
                assert!(Basis::Footprint.caption().contains("physical footprint"));
            }
            None => {
                assert!(footprint().is_none());
                assert!(Basis::Resident.caption().contains("resident set"));
            }
        }
    }

    /// The pair sums to the figure the surface carries, and a measured zero stays
    /// distinguishable from an absent reading.
    #[test]
    fn the_pair_sums_and_a_zero_is_a_reading() {
        let measured = Memory {
            now: Some(2 << 30),
            peak: Some(3 << 30),
            gpu_now: Some(512 << 20),
            gpu_peak: Some(1 << 30),
            footprint: true,
        };
        assert_eq!(measured.system_now_text(), "1.50GiB");
        assert_eq!(measured.gpu_now_text(), "512.0MiB");
        assert_eq!(measured.system_peak_text(), "2.00GiB");
        assert_eq!(measured.gpu_peak_text(), "1.00GiB");

        let zero = Memory {
            gpu_now: Some(0),
            gpu_peak: Some(0),
            ..measured
        };
        assert_eq!(zero.gpu_now_text(), "0B");
        assert_eq!(zero.system_now_text(), "2.00GiB");

        let absent = Memory {
            gpu_now: None,
            gpu_peak: None,
            ..measured
        };
        assert_eq!(absent.gpu_now_text(), NOT_MEASURED);
        assert_ne!(zero.gpu_now_text(), absent.gpu_now_text());
        assert_eq!(absent.system_now_text(), "2.00GiB");
    }

    /// The FFI mirror is the kernel's own size: 372 bytes, 93 words. A natural
    /// layout would round up to 376 and ask the kernel for a word more than it has,
    /// which it logs on every call.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_kernel_layout_is_the_sdk_layout() {
        assert_eq!(std::mem::size_of::<TaskVmInfo>(), 372);
        assert_eq!(
            std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<i32>(),
            93
        );
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
