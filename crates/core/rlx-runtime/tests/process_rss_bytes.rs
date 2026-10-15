// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(target_os = "macos")]
//! `process_rss_bytes` must report the running process's *current* resident
//! size, not its peak — and the two agree while a working set is held, so only
//! a phase that releases one can tell them apart. Hence the anonymous mappings
//! and `munmap` below rather than a dropped `Vec`: on macOS a dropped `Vec`
//! keeps its pages resident and the reading would never be seen to fall.
//!
//! Auto-discovered (no `[[test]]` entry), so it runs in the default gate rather
//! than behind `backend-tests`; it is the only test in this binary, so the
//! gigabytes it maps are not contended by another test's allocations.
//!
//! Every threshold below is a fraction of what was mapped rather than a
//! near-exact delta, and the one absolute threshold is asserted only once the
//! independent reading has confirmed the process really holds that much: a
//! machine under memory pressure compresses this process's pages away, and the
//! check must not blame the reading for the machine.
use rlx_runtime::{available_unified_memory, process_rss_bytes};

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

const PROT_READ_WRITE: i32 = 1 | 2; // PROT_READ | PROT_WRITE
const MAP_PRIVATE_ANON: i32 = 0x0002 | 0x1000; // MAP_PRIVATE | MAP_ANON
/// 4 KiB: the smallest page any macOS target uses. A stride of Apple Silicon's
/// 16 KiB would leave three quarters of an Intel mapping non-resident and then
/// fail a reading that is fine.
const TOUCH_STRIDE: usize = 4 * 1024;

/// Tens of MB, then a few hundred MB: the two levels the reading is checked at.
const MODEST: usize = 64 << 20;
const LARGE: usize = 384 << 20;
/// Above 4 GiB — the regime that catches a reading truncated to 32 bits, which
/// is right at both levels above.
const HUGE: usize = 4600 << 20;
/// How far [`ps_rss_bytes`] may sit outside the window two readings of the
/// reading under test bracket it with. Both report the same kernel counter —
/// the resident bytes this process is charged for — through different doors
/// (the mach `task_info` call, and the proc table read by a separate process),
/// so in practice they agree to the page quantum. A few MiB is safe for a
/// process allocating concurrently and far below the working sets below, so it
/// cannot mask a broken reading.
const TOLERANCE: usize = 4 << 20;

/// An anonymous private mapping with every page written, so it is resident.
struct WorkingSet {
    base: *mut u8,
    len: usize,
}

impl WorkingSet {
    fn mapped(len: usize) -> Self {
        let base = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ_WRITE,
                MAP_PRIVATE_ANON,
                -1,
                0,
            )
        };
        assert!(base as isize != -1, "mmap of {len} bytes failed");
        // `write_volatile`: a plain store nothing reads back can be removed, and
        // a touch loop optimised away leaves the pages non-resident — which the
        // calling phase's growth assertion fails on rather than passes.
        let mut off = 0;
        while off < len {
            unsafe { std::ptr::write_volatile(base.add(off), 1) };
            off += TOUCH_STRIDE;
        }
        Self { base, len }
    }
}

impl Drop for WorkingSet {
    fn drop(&mut self) {
        assert_eq!(unsafe { munmap(self.base, self.len) }, 0, "munmap failed");
    }
}

/// The reading under test and [`ps_rss_bytes`]'s figure for the same process.
struct Samples {
    rss: usize,
    ps: usize,
}

/// Sample both: the reading under test, then `ps`, then the reading again.
///
/// Bracketing `ps` with two readings is what makes the comparison an assertion
/// about one instant. A machine under memory pressure compresses pages out of
/// this process *while* `ps` runs (measured: tens of MB of drift over a spawn
/// that took milliseconds), so a point comparison would compare two different
/// moments of a moving counter. A reading that were zero, or the peak rather
/// than the current size, or truncated to 32 bits, still falls outside the
/// window.
fn sample(at: &str) -> Samples {
    let before = reading(at);
    let ps = ps_rss_bytes();
    let after = reading(at);
    let lo = before.min(after).saturating_sub(TOLERANCE);
    let hi = before.max(after) + TOLERANCE;
    assert!(
        ps >= lo && ps <= hi,
        "at {at}: ps {ps} B falls outside [{lo}, {hi}] B, the window two \
         process_rss_bytes() readings of {before} B and {after} B bracket it with"
    );
    Samples { rss: after, ps }
}

fn reading(at: &str) -> usize {
    let rss = process_rss_bytes().expect("macOS must yield a resident size");
    assert!(rss > 0, "at {at}: process_rss_bytes() returned 0");
    rss
}

/// Resident size as `/bin/ps` reports it: 1024-byte blocks, from the proc table.
fn ps_rss_bytes() -> usize {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<usize>()
        .expect("ps rss")
        * 1024
}

#[test]
fn the_reading_is_the_current_resident_size() {
    let base = sample("baseline").rss;
    let mut held = Vec::new();

    held.push(WorkingSet::mapped(MODEST));
    let modest = sample("tens of MB held").rss;
    assert!(
        modest >= base + MODEST / 2,
        "at tens of MB: {base} B -> {modest} B for {MODEST} mapped and touched"
    );

    held.push(WorkingSet::mapped(LARGE));
    let peak = sample("a few hundred MB held").rss;
    assert!(
        peak >= modest + LARGE / 2,
        "at a few hundred MB: {modest} B -> {peak} B for {LARGE} mapped and touched"
    );

    // The release step, and the one thing a growing working set cannot show: a
    // *peak* reading (`resident_size_max`) reports `peak` here and drops by 0.
    // Only a quarter of the release is required, since a machine under pressure
    // may not have kept all of it resident, but a drop this large cannot happen
    // to a reading that only ever grows.
    held.clear();
    let released = sample("after releasing it").rss;
    assert!(
        peak.saturating_sub(released) >= (MODEST + LARGE) / 4,
        "releasing {MODEST} + {LARGE} dropped the reading by only {} B \
         ({peak} B -> {released} B)",
        peak.saturating_sub(released)
    );

    // Above 4 GiB. Skipped only where the machine could not hold it resident at
    // a third of its RAM, so a reading that is fine cannot fail for lack of
    // memory. (This guard uses total installed memory, never the reading.)
    if !matches!(available_unified_memory(), Some(ram) if ram >= 3 * HUGE) {
        eprintln!("skipping the above-4-GiB phase: too little RAM to hold {HUGE} B resident");
        return;
    }
    let huge = WorkingSet::mapped(HUGE);
    let big = sample("above 4 GiB held");
    drop(huge);
    let back = sample("after releasing it").rss;
    assert!(
        big.rss.saturating_sub(back) >= HUGE / 4,
        "releasing {HUGE} dropped the reading by only {} B ({} B -> {back} B)",
        big.rss.saturating_sub(back),
        big.rss
    );
    if big.ps > 4 << 30 {
        // The independent figure confirms this process really holds more than a
        // 32-bit count of resident bytes, so the reading has to be able to say
        // so; a reading truncated to 32 bits lands below both thresholds, and
        // nothing below this level can tell that apart.
        assert!(
            big.rss > 4 << 30,
            "the process holds {} B resident (ps) but process_rss_bytes() says {} B \
             — the count is being truncated",
            big.ps,
            big.rss
        );
    } else {
        eprintln!(
            "above-4-GiB assertion skipped: memory pressure left only {} B of {HUGE} B resident",
            big.ps
        );
    }
}
