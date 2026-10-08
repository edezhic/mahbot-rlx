// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A schedule target built from a live device, not a table.**
//!
//! CUDA and Metal capabilities are ISA facts: `mma.sync.m16n8k16` and
//! `simdgroup_half8x8` are fixed by the architecture, so a compile-time table
//! is the right shape for them.
//!
//! Vulkan is not like that. `VK_KHR_cooperative_matrix` **enumerates** the
//! supported `(MSize, NSize, KSize, AType, BType, CType, scope)` combinations
//! per physical device via `VkCooperativeMatrixPropertiesKHR`. Which shapes a
//! device supports is not derivable from the API version or the feature bit.
//!
//! rlx's current check is:
//!
//! ```ignore
//! if !feats.contains(SHADER_F16) || !feats.contains(EXPERIMENTAL_COOPERATIVE_MATRIX) {
//!     return None;
//! }
//! ```
//!
//! — the feature bit is present, therefore 8x8 f16 is assumed to work. That is
//! an assumption a conformant device may violate, and it is why
//! [`rlx_ir::kernel_schedule::Features`] distinguishes a queried capability
//! list from a static one.
//!
//! # What this can and cannot see
//!
//! wgpu surfaces the device's cooperative-matrix combinations
//! (`Adapter::cooperative_matrix_properties`, the table this crate reads in
//! `device::coop_f32_8x8_supported` / `coop_f16_16x16_supported` /
//! `coop_f16_16x16_f32_acc_supported`). [`target_for_device`] nevertheless
//! records only the 8x8x8 shape its WGSL kernel is written against **and marks
//! the set truncated**, because combinations it did not ask about may exist.
//!
//! That is the honest encoding of the situation: a `false` from
//! [`rlx_ir::kernel_schedule::Target::has`] here means "not known to be
//! supported", not "known to be unsupported", and
//! [`rlx_ir::kernel_schedule::Target::capabilities_may_be_incomplete`] says so.
//! Closing that gap needs a raw Vulkan query beneath wgpu, which is a larger
//! change than this module.

use rlx_ir::kernel_schedule::{Feature, FeatureSet, Features, Target};

/// Tile edge the `matmul_coop16` WGSL kernel is written against.
pub const COOP_TILE: u32 = 8;

/// Build a schedule target from a live wgpu device.
///
/// `max_shared_bytes` comes from the device's real
/// `max_compute_workgroup_storage_size` rather than a guess, so a schedule that
/// fits the adapter in hand is accepted and one that does not is rejected with
/// the adapter's own number.
pub fn target_for_device(device: &wgpu::Device) -> Target {
    let limits = device.limits();
    let feats = device.features();

    let mut set = FeatureSet::new();
    // Only claim the shape the shipped kernel actually uses: this gate reads the
    // feature bits, not the adapter's cooperative-matrix table, and inventing
    // entries would be worse than admitting the list is partial.
    if feats.contains(wgpu::Features::SHADER_F16)
        && feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        set.push(Feature::CoopMatrix {
            m: COOP_TILE,
            n: COOP_TILE,
            k: COOP_TILE,
        });
    }
    // Mark the list partial regardless: this records the one shape the kernel
    // is written against, not the device's whole table.
    set.truncated = true;

    Target {
        name: "wgpu-device",
        max_shared_bytes: limits.max_compute_workgroup_storage_size as usize,
        // wgpu speaks in invocations, not warps. 32 is the common subgroup
        // width; this is a floor, not a measurement.
        max_warps: (limits.max_compute_invocations_per_workgroup as usize).div_ceil(32),
        features: Features::Queried(set),
    }
}

/// A target for a device with no cooperative-matrix support, for testing the
/// rejection path without needing such a device present.
pub fn target_without_coop_matrix(max_shared_bytes: usize) -> Target {
    let mut set = FeatureSet::new();
    set.truncated = true;
    Target {
        name: "wgpu-device (no coop matrix)",
        max_shared_bytes,
        max_warps: 8,
        features: Features::Queried(set),
    }
}
