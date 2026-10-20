// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assertions the cell's gates share.

/// Refuse to pin a submission/wait shape under the run-report switch: it carries
/// a timestamp write on its own passes and adds a fence to a dispatch-only run, so
/// the counts beside it are the instrument's, not the run's.
pub fn assert_run_report_off() {
    assert!(
        !rlx_ir::env::flag("RLX_WGPU_DBG_COUNTERS"),
        "RLX_WGPU_DBG_COUNTERS changes the submission/wait shape of a forward pass; \
         unset it to run this pin"
    );
}
