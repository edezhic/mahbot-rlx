// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device's own memory ceiling — Apple's `recommendedMaxWorkingSetSize` —
//! as a value of its own.
//!
//! Every other memory figure this runtime reports describes the **process** or
//! the **machine**: [`crate::available_unified_memory`] is installed RAM,
//! [`crate::soft_memory_budget_bytes`] is a fraction of it, and the profiler's
//! `device_vram_bytes` calls that same total the Apple GPU's VRAM. The device's
//! own bound is none of those, and it cannot be derived from them: the
//! platform's fraction of RAM is machine-specific. So it is read from the
//! platform — [`device_working_set_bytes`].

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
use std::sync::OnceLock;

use rlx_driver::Device;

/// Bytes of device memory the platform recommends GPU work on `device` may
/// occupy, or `None` when this host has no such figure.
///
/// On Apple: `MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize`, the
/// read the Metal backend takes for the device it runs on, surfaced so a caller
/// that never activates that backend can take it before compiling anything. No
/// session or compiled graph needed — taken once, held for the process.
///
/// **A figure of its own.** The platform's value as it stands — not a fraction
/// of installed RAM, not the process soft budget, and not folded into either.
///
/// **What it means.** Apple's wording is a *recommendation*: an approximation of
/// what the device can hold without affecting runtime performance. It is a
/// ceiling on the device's whole footprint — not free memory, not enforced, and
/// not a per-allocation cap, which is a separate figure.
///
/// **Which devices answer.** Those that place work on the Apple GPU —
/// [`Device::Gpu`] (wgpu-on-Metal) and [`Device::WebGpu`], its label for that
/// same backend, [`Device::Metal`], [`Device::Mlx`], and [`Device::Vulkan`]
/// through MoltenVK. All answer with this one figure rather than with their own
/// readings, so they cannot disagree about a host. Its source is the platform's
/// default GPU — the same device the Metal backend and MLX read — which a
/// graphics API on a multi-GPU host may not pick. Every other device answers
/// `None`: the neural engine is a separate accelerator with its own budget, an
/// external GPU a different physical device.
///
/// **Absence.** `None` on non-Apple targets, on watchOS, and on an Apple host
/// with no Metal device — never `0`, this tree's in-band "unknown VRAM" that
/// callers act on destructively, and never installed RAM or the process budget.
///
/// ```
/// use rlx_runtime::{Device, device_working_set_bytes};
///
/// if let Some(ceiling) = device_working_set_bytes(Device::Gpu) {
///     println!("the device holds {ceiling} bytes of GPU work");
/// }
/// ```
pub fn device_working_set_bytes(device: Device) -> Option<u64> {
    match device {
        Device::Gpu | Device::WebGpu | Device::Metal | Device::Mlx | Device::Vulkan => {
            ceiling(platform_reading())
        }
        _ => None,
    }
}

/// The zero rule, on its own so it is unit-testable: a missing reading, or a
/// literal `0`, is absence rather than a number. Everything else passes through
/// unchanged — no multiple, no safety fraction.
fn ceiling(reading: Option<u64>) -> Option<u64> {
    reading.filter(|&bytes| bytes > 0)
}

/// Read once per process, not live: the setting is not chased mid-run.
#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn platform_reading() -> Option<u64> {
    static READING: OnceLock<Option<u64>> = OnceLock::new();
    *READING.get_or_init(apple::recommended_max_working_set)
}

/// No Metal framework, no device, no ceiling: the concept does not exist on
/// non-Apple targets, nor on watchOS (no public Metal API).
#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
fn platform_reading() -> Option<u64> {
    None
}

/// The platform read itself: Metal and the Objective-C runtime reached through
/// `dlopen`/`dlsym`.
#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
mod apple {
    use std::ffi::{CStr, c_char, c_int, c_void};
    use std::mem::{size_of, transmute_copy};

    /// `RTLD_NOW`.
    const RTLD_NOW: c_int = 0x2;

    /// The framework this probe must **not** link — it is compiled into every
    /// Apple build, including CPU-only artifacts that never touch Metal; a
    /// library that is not there is the absence case, not a failure.
    const METAL_FRAMEWORK: &CStr = c"/System/Library/Frameworks/Metal.framework/Metal";
    /// The Objective-C runtime, for `objc_msgSend` + `sel_registerName`.
    const OBJC_RUNTIME: &CStr = c"/usr/lib/libobjc.A.dylib";

    unsafe extern "C" {
        fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    // `MTLCreateSystemDefaultDevice` — a `…Create…` call, so +1: the caller
    // releases the device once the property has been read.
    type CreateSystemDefaultDevice = unsafe extern "C" fn() -> *mut c_void;
    // `objc_msgSend`, in the two shapes this probe sends. Cocoa selectors are
    // resolved at run time, so the send is cast to the signature it is sending.
    type RegisterName = unsafe extern "C" fn(*const c_char) -> *mut c_void;
    type RespondsToSelector = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
    type WorkingSetSize = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
    type Release = unsafe extern "C" fn(*mut c_void);

    /// Resolve `name` in an already-`dlopen`ed image.
    fn symbol<T: Copy>(handle: *mut c_void, name: &CStr) -> Option<T> {
        let ptr = unsafe { dlsym(handle, name.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        debug_assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
        Some(unsafe { transmute_copy(&ptr) })
    }

    /// `MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize`, or `None`
    /// when the framework, the device, or the property itself is not there.
    pub(super) fn recommended_max_working_set() -> Option<u64> {
        let metal = unsafe { dlopen(METAL_FRAMEWORK.as_ptr(), RTLD_NOW) };
        let objc = unsafe { dlopen(OBJC_RUNTIME.as_ptr(), RTLD_NOW) };
        if metal.is_null() || objc.is_null() {
            return None;
        }
        unsafe {
            let create: CreateSystemDefaultDevice = symbol(metal, c"MTLCreateSystemDefaultDevice")?;
            let register_name: RegisterName = symbol(objc, c"sel_registerName")?;
            let msg_send: *mut c_void = symbol(objc, c"objc_msgSend")?;
            let release: Release = symbol(objc, c"objc_release")?;

            let device = create();
            if device.is_null() {
                return None;
            }
            let working_set = register_name(c"recommendedMaxWorkingSetSize".as_ptr());
            // iOS/tvOS 16+, visionOS 1+: on an older system the send below is an
            // unrecognised selector — a crash, not absence.
            let responds: RespondsToSelector = transmute_copy(&msg_send);
            let send: WorkingSetSize = transmute_copy(&msg_send);
            let has_property = responds(
                device,
                register_name(c"respondsToSelector:".as_ptr()),
                working_set,
            ) != 0;
            let bytes = has_property.then(|| send(device, working_set));
            release(device);
            bytes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{available_unified_memory, soft_memory_budget_bytes};

    /// A reading passes through as it stands — an extra safety fraction would
    /// show up here — except for a literal `0`, which is absence rather than a
    /// zero-byte ceiling.
    #[test]
    fn the_reading_is_passed_through_without_arithmetic() {
        assert_eq!(ceiling(Some(4096)), Some(4096));
        assert_eq!(ceiling(Some(0)), None);
    }

    /// A device off the GPU path has no ceiling to report.
    #[test]
    fn devices_off_the_gpu_path_report_absence() {
        for device in [
            Device::Cpu,
            Device::Ane,
            Device::Cuda,
            Device::Rocm,
            Device::Egpu,
        ] {
            assert_eq!(device_working_set_bytes(device), None, "{device:?}");
        }
    }

    /// One ceiling per host rather than one per path — and not installed RAM.
    #[test]
    fn the_gpu_paths_answer_with_one_figure_of_their_own() {
        let reading = device_working_set_bytes(Device::Gpu);
        eprintln!(
            "device working set: {reading:?} B; soft budget: {:?} B; installed RAM: {:?} B",
            soft_memory_budget_bytes(),
            available_unified_memory(),
        );
        for device in [Device::WebGpu, Device::Metal, Device::Mlx, Device::Vulkan] {
            assert_eq!(device_working_set_bytes(device), reading, "{device:?}");
        }
        if let (Some(bytes), Some(ram)) = (reading, available_unified_memory()) {
            assert_ne!(bytes, ram as u64);
        }
    }

    /// The tests above are satisfied by an absent reading — which is what a host
    /// without a Metal device legitimately reports — so a probe that had silently
    /// degraded to `None` would look exactly like such a host. Where a
    /// Metal-backed backend this build contains is instantiable, the read must
    /// produce a figure. `just test-gpu` runs with one, and `RLX_REQUIRE_DEVICE=1`
    /// turns that pair from a skip into a failure.
    #[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
    #[test]
    fn a_usable_apple_gpu_backend_implies_a_figure() {
        let backend = [
            ("metal", Device::Metal),
            ("wgpu", Device::Gpu),
            ("vulkan", Device::Vulkan),
        ]
        .into_iter()
        .find(|(_, d)| crate::feature_compiled(*d));
        let Some((label, device)) = backend else {
            return; // No Apple GPU backend in this build to cross-check against.
        };
        if rlx_ir::env::skip_unless_device(label, true, crate::is_available(device)) {
            return;
        }
        assert!(
            device_working_set_bytes(device).is_some_and(|bytes| bytes > 0),
            "{label} is instantiable here, so the platform read must produce a figure"
        );
    }

    /// The Metal backend takes the same platform read for the device it runs on;
    /// the surface must hand out that number — folding a zero reading to absence,
    /// which the backend keeps raw.
    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
    #[test]
    fn the_metal_backend_reads_the_same_platform_figure() {
        let backend = rlx_metal::device::metal_device().map(|d| d.max_working_set);
        assert_eq!(
            device_working_set_bytes(Device::Metal),
            backend.filter(|&bytes| bytes > 0)
        );
    }
}
