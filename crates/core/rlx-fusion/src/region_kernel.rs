// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ambient region-kernel judgement: which activation codes the region kernel that
//! will run a compile's regions actually implements.
//!
//! [`rlx_ir::region_encode`] declares such a code set (`activation_sub` numbering) and
//! the pure predicates over it — read that constant's doc for what the set is for. This
//! module is the compile-scoped context that says *which* set is in force, mirroring
//! [`crate::limits`]; the compile pipeline installs it
//! (`rlx_compile::fusion_target::with_fusion_context`) because that is the layer that
//! knows the device — a fusion pass cannot name a kernel.
//!
//! The judgement is keyed to the **kernel**, not to the fusion bucket: two device labels
//! that share a `FusionTarget` can run different region engines, and only the one whose
//! kernel this is may narrow. `None` — the default — installs no judgement and every
//! activation stays region-fusable exactly as it was, so devices whose region kernel
//! differs keep their present fusion decisions.
//!
//! Every site that can put an activation into a region consults it: the sites are those
//! that *choose* a step's activation, established by scanning every non-test
//! construction of `Op::ElementwiseRegion` / `Op::BatchElementwiseRegion` /
//! `Op::TransformRegion` and of `ChainStep::Activation`. They are
//! [`MarkElementwiseRegions`](crate::fusion::MarkElementwiseRegions)'s `chain_eligible`
//! and `unary_region_step_from_op` — which `FuseRegionPrologue`, `MarkBatchSliceRegions`
//! and the per-slice regions they mint all reach — through
//! [`region_kernel_honours_activation`], and `FuseBatchPreprocess` through
//! `region_signature` via [`region_kernel_honours_chain`]. The remaining constructions
//! copy or re-use a chain that already exists; those arrive already formed, which
//! `rlx-wgpu`'s expansion hook takes (`rlx_wgpu::unfuse::expand_unimplemented_regions`).

use std::cell::Cell;

use rlx_ir::op::{Activation, ChainStep};
use rlx_ir::region_encode::{activation_sub, activation_subs_cover_chain};

thread_local! {
    /// Codes the region kernel in force for this compile implements. `None` — the default,
    /// and every compile whose device runs a different region kernel — installs no
    /// kernel-keyed judgement, so every activation stays region-fusable.
    static ACTIVE_REGION_KERNEL_SUBS: Cell<Option<&'static [u32]>> = const { Cell::new(None) };
}

/// Codes the region kernel in force for this compile implements.
pub fn active_region_kernel_activation_subs() -> Option<&'static [u32]> {
    ACTIVE_REGION_KERNEL_SUBS.with(|c| c.get())
}

/// Run `f` with a region kernel's code set installed (single-threaded compile).
pub fn with_region_kernel_activation_subs<T>(
    subs: Option<&'static [u32]>,
    f: impl FnOnce() -> T,
) -> T {
    ACTIVE_REGION_KERNEL_SUBS.with(|c| {
        let prev = c.get();
        c.set(subs);
        let out = f();
        c.set(prev);
        out
    })
}

/// Does the region kernel in force for this compile implement `a`? `true` when no
/// kernel-keyed judgement is installed.
pub fn region_kernel_honours_activation(a: Activation) -> bool {
    match active_region_kernel_activation_subs() {
        None => true,
        Some(subs) => subs.contains(&activation_sub(a)),
    }
}

/// [`region_kernel_honours_activation`] over a whole chain: a region is unlowerable if
/// any of its activation steps is.
pub fn region_kernel_honours_chain(chain: &[ChainStep]) -> bool {
    match active_region_kernel_activation_subs() {
        None => true,
        Some(subs) => activation_subs_cover_chain(subs, chain),
    }
}
