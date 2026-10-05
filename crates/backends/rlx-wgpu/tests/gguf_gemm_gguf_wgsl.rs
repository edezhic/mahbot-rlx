// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `dequant_gemm_gguf.wgsl` (the m > 1 row-grouped Q4_K / Q6_K GEMM) must
//! parse and validate.
//!
//! Device-free (naga only). Without this a WGSL mistake would surface as a
//! panic at `create_shader_module` on whatever graph first runs a batched
//! Q4_K / Q6_K multiply, long after the edit.

#[test]
fn wgsl_parses_validates_and_keeps_both_entry_points() {
    let module = naga::front::wgsl::parse_str(rlx_wgpu::kernels::DEQUANT_GEMM_GGUF_WGSL)
        .expect("wgsl parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("wgsl validate");

    // One entry point per scheme: routing Q4_K through the Q6_K entry (or a
    // scheme with no entry at all) would decode garbage rather than fail.
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    for want in ["dequant_gemm_q4k", "dequant_gemm_q6k"] {
        assert!(
            names.contains(&want),
            "missing entry point {want}: {names:?}"
        );
    }
    assert_eq!(names.len(), 2, "unexpected extra entry points: {names:?}");
}
