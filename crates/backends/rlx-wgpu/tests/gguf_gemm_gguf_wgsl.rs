// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `dequant_gemm_gguf.wgsl` (the m > 1 row-grouped Q4_K / Q6_K / Q8_0 GEMM)
//! must parse and validate.
//!
//! Device-free (naga only). Without this a WGSL mistake would surface as a
//! panic at `create_shader_module` on whatever graph first runs a batched
//! Q4_K / Q6_K / Q8_0 multiply, long after the edit.
//!
//! The dispatch geometry these entries have to match — their `@workgroup_size`
//! and their `TG`/`TM` consts against the Rust dispatch constants — is pinned by
//! `gguf_gpu.rs`'s own test, where those constants are private.

#[test]
fn wgsl_parses_validates_and_keeps_the_scheme_entry_points() {
    let module = naga::front::wgsl::parse_str(rlx_wgpu::kernels::DEQUANT_GEMM_GGUF_WGSL)
        .expect("wgsl parse");
    // The portability envelope, as far as naga can state it: each entry's workgroup
    // shape is pinned to `GROUPED_THREADS` = 64 by `gguf_gpu.rs`'s own test, inside
    // the WebGPU baseline's 256 invocations — and this validation is capability-free
    // on purpose. `Capabilities::empty()` is the strictest set, so it subsumes
    // validating with every capability allowed, and it is what pins the entry points
    // to the WGSL baseline: an arm reaching for an optional capability (f16, which
    // this adapter provides) would otherwise compile here and fail only on a device
    // that provides less.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("wgsl validate without any optional capability");

    // One entry point per scheme: routing Q4_K through the Q6_K entry (or a
    // scheme with no entry at all) would decode garbage rather than fail.
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    for want in ["dequant_gemm_q4k", "dequant_gemm_q6k", "dequant_gemm_q8_0"] {
        assert!(
            names.contains(&want),
            "missing entry point {want}: {names:?}"
        );
    }
    assert_eq!(names.len(), 3, "unexpected extra entry points: {names:?}");
    // None may use a capability this backend avoids outright: no atomics of any
    // kind, no subgroup operations.
    for avoided in ["atomic", "subgroup"] {
        assert!(
            !rlx_wgpu::kernels::DEQUANT_GEMM_GGUF_WGSL.contains(avoided),
            "the shader must not use `{avoided}`"
        );
    }
}
