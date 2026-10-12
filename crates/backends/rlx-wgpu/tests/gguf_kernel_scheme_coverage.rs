// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every GGUF scheme registered in the shared id table must have a branch in
//! the WGSL entry point that handles it — or be explicitly host-routed.
//!
//! Three entry points carry scheme dispatch:
//!   * `dequant_gguf` (`dequant_gguf.wgsl`) — device dequant into f32 scratch;
//!   * `dequant_gemv_tiled` / `dequant_gemv` (`dequant_gemv_gguf.wgsl`) — the
//!     two one-row packed-multiply arms, Q4_K / Q6_K and Q1_0 respectively.
//!
//! Each is an `if (params.scheme_id == N) { … }` chain with **no default arm**.
//! An id that reaches one without a matching branch falls off the end, leaves
//! the dequant scratch (or the output row) untouched, and reads back as
//! **zeros** — the model degrades silently instead of failing. `Q2_0` (25) sat
//! in the table with no WGSL branch for exactly this reason, which is how
//! `Ternary-Bonsai` and `Pestle-27B` would have "run" on wgpu. The same hazard
//! applies per *entry point*: routing Q4_K at the tiled arm's `dequant_gemv`
//! neighbour, which has no branch for it, would zero that column.
//!
//! Two directions matter and both are checked: claimed-but-not-branched →
//! silent zeros (the dangerous one); branched-but-not-claimed → correct, just
//! needlessly host-routed.

use std::collections::{BTreeMap, BTreeSet};

const NEEDLE: &str = "params.scheme_id == ";
const ENTRY: &str = "@compute";

/// Every `params.scheme_id == <N>u` a source text tests.
fn scheme_ids(src: &str) -> BTreeSet<u32> {
    let mut ids = BTreeSet::new();
    for (i, _) in src.match_indices(NEEDLE) {
        let tail = &src[i + NEEDLE.len()..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(id) = digits.parse::<u32>() {
            ids.insert(id);
        }
    }
    ids
}

/// Each `@compute` entry point of `src`, as `name -> ids it branches on`.
///
/// Per entry point rather than per file: the one-row shader's two arms cover
/// disjoint scheme sets, so a file-level union would accept a branch sitting in
/// the wrong arm — which is exactly the silent-zero case.
fn branches_by_entry(src: &str) -> BTreeMap<String, BTreeSet<u32>> {
    let mut by_entry = BTreeMap::new();
    let mut rest = src;
    while let Some(at) = rest.find(ENTRY) {
        let body = &rest[at..];
        let len = body[ENTRY.len()..]
            .find(ENTRY)
            .map_or(body.len(), |i| i + ENTRY.len());
        let chunk = &body[..len];
        let name = chunk
            .split("fn ")
            .nth(1)
            .and_then(|s| s.split(['(', ' ', '\n']).next())
            .unwrap_or_else(|| panic!("no `fn <name>` in entry point: {chunk}"));
        by_entry.insert(name.to_string(), scheme_ids(chunk));
        rest = &body[len..];
    }
    by_entry
}

/// Every `params.scheme_id == <N>u` the device-dequant shader actually tests.
fn branches_in_shader() -> BTreeSet<u32> {
    scheme_ids(rlx_wgpu::kernels::DEQUANT_GGUF_WGSL)
}

/// One `@compute` entry point's body, from its `fn` to the next entry point.
fn entry_body<'a>(src: &'a str, name: &str) -> &'a str {
    let head = format!("fn {name}(");
    let at = src
        .find(&head)
        .unwrap_or_else(|| panic!("no `{head}` in the shader"));
    let rest = &src[at..];
    &rest[..rest.find(ENTRY).map_or(rest.len(), |i| i + ENTRY.len())]
}

/// Ids in the shared table, via the public inverse of the id mapping.
fn registered_ids() -> BTreeSet<u32> {
    (0u32..=64)
        .filter(|id| rlx_ir::quant::QuantScheme::from_gpu_dequant_scheme_id(*id).is_some())
        .collect()
}

#[test]
fn wgsl_branch_set_matches_kernel_supports_scheme() {
    let in_shader = branches_in_shader();
    assert!(
        !in_shader.is_empty(),
        "parsed no `{NEEDLE}Nu` branches — the shader or this parser changed shape"
    );
    let claimed: BTreeSet<u32> = (0u32..=64)
        .filter(|id| rlx_wgpu::gguf_gpu::kernel_supports_scheme(*id))
        .collect();

    let claimed_not_branched: Vec<u32> = claimed.difference(&in_shader).copied().collect();
    assert!(
        claimed_not_branched.is_empty(),
        "kernel_supports_scheme claims {claimed_not_branched:?} but the WGSL \
         shader has no branch for them — these dispatch to the kernel, match \
         nothing, and read back as zeros"
    );

    let branched_not_claimed: Vec<u32> = in_shader.difference(&claimed).copied().collect();
    assert!(
        branched_not_claimed.is_empty(),
        "the WGSL shader branches on {branched_not_claimed:?} but \
         kernel_supports_scheme host-routes them — correct, but slower than \
         it needs to be; add them to the list"
    );
}

/// The one-row shader's arms, each against the predicate its routing uses: a
/// scheme routed to an arm with no branch for it reads back as zeros.
#[test]
fn each_one_row_arm_branches_on_exactly_the_schemes_routed_to_it() {
    /// The arm that keeps the original shape (`dequant_gemv`): what is left of
    /// the route's own claim once the tiled arm's schemes are taken out.
    fn legacy(id: u32) -> bool {
        rlx_wgpu::gguf_gpu::gemv_supports_scheme(id)
            && !rlx_wgpu::gguf_gpu::gemv_tiled_supports_scheme(id)
    }

    let entries = branches_by_entry(rlx_wgpu::kernels::DEQUANT_GEMV_GGUF_WGSL);
    let names: Vec<&str> = entries.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        ["dequant_gemv", "dequant_gemv_tiled"],
        "the one-row shader's entry points changed — routing and this check \
         both name them"
    );

    let arm = |entry: &str, routed: fn(u32) -> bool| {
        let claimed: BTreeSet<u32> = (0u32..=64).filter(|id| routed(*id)).collect();
        assert!(
            !claimed.is_empty(),
            "{entry}: no scheme is routed to this arm at all"
        );
        assert_eq!(
            entries[entry], claimed,
            "{entry}: branches {:?} vs the schemes routed to it {claimed:?}",
            entries[entry]
        );
    };
    arm(
        "dequant_gemv_tiled",
        rlx_wgpu::gguf_gpu::gemv_tiled_supports_scheme,
    );
    arm("dequant_gemv", legacy);
}

/// Both one-row entry points must parse and validate device-free. A WGSL
/// mistake in an arm only some shapes reach would otherwise surface as a panic
/// at `create_shader_module` on whatever graph first runs it.
#[test]
fn one_row_entry_points_parse_and_validate() {
    let module = naga::front::wgsl::parse_str(rlx_wgpu::kernels::DEQUANT_GEMV_GGUF_WGSL)
        .expect("wgsl parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("wgsl validate");

    // Each arm stays inside the WebGPU floor (256 invocations per workgroup), and
    // the legacy arm's declared size is the thread count its dispatch geometry
    // assumes: one thread per column, `GEMV_LEGACY_THREADS` to a workgroup. The
    // tiled arm's lane count is not repeated here — the device-free
    // `gguf_gpu::tests::tiled_gemv_geometry_matches_the_shader` derives it from
    // `GEMV_TILED_THREADS` itself, which the recorded shape in
    // `gguf_matmul_route.rs` then follows.
    let mut sizes = BTreeMap::new();
    for entry in &module.entry_points {
        assert!(
            entry.workgroup_size.iter().all(|&d| d <= 256),
            "{}: workgroup size {:?} leaves the portable floor",
            entry.name,
            entry.workgroup_size
        );
        sizes.insert(entry.name.as_str(), entry.workgroup_size);
    }
    assert_eq!(sizes["dequant_gemv"], [64, 1, 1], "legacy thread count");

    // How each arm reads the packed weight is part of the shape the one-row route
    // reports: the tiled arm's word reads go through the 4-byte `read_word_at`,
    // the arm that keeps one thread per column reads bytes. Pinned so a return to
    // byte-wise reads on the tiled arm is not silent — the recorded geometry alone
    // cannot see it. What this checks is the spelling of that arm's own reads:
    // Q6_K's header still goes through the byte-wise `dq_read_f16` / `read_w_i8`
    // (through a differently named helper the check would not trip).
    let wgsl = rlx_wgpu::kernels::DEQUANT_GEMV_GGUF_WGSL;
    let tiled = entry_body(wgsl, "dequant_gemv_tiled");
    assert!(
        tiled.contains("read_word_at("),
        "the tiled one-row arm must read the packed weight in 4-byte words"
    );
    assert!(
        !tiled.contains("read_w("),
        "the tiled one-row arm's own reads must not fall back to the byte-wise `read_w`"
    );
    assert!(
        entry_body(wgsl, "dequant_gemv").contains("read_w("),
        "the one-thread-per-column arm reads the packed weight a byte at a time"
    );
}

/// The one-row route's device guard. The tiled arm's `vec4` X read needs both the
/// window's offset and the row stride to line up with a four-element group; the
/// arm that reads X a scalar at a time does not. Pinned here because the decision
/// is arithmetic — the route consults this at its own entry, so a call it refuses
/// goes to the host instead of being read shifted.
#[test]
fn the_one_row_device_guard_requires_a_vec4_addressable_x_window() {
    use rlx_wgpu::gguf_gpu::{gemv_one_row_device_ok as ok, gemv_tiled_supports_scheme};

    let tiled = (0u32..=64)
        .find(|id| gemv_tiled_supports_scheme(*id))
        .expect("the tiled arm has at least one scheme");
    // A planned activation lands on a 16-byte boundary of its window and `k` is a
    // multiple of 256, so this is every call the models make.
    for aligned in [0usize, 16, 64, 1024, 256 * 3 + 32] {
        assert!(
            ok(tiled, aligned, 4096),
            "an X window {aligned} bytes in is readable as vec4"
        );
    }
    // A 4-byte-aligned view of one is not — neither as the window ...
    for misaligned in [4usize, 8, 12, 256 + 12, 1024 + 4] {
        assert!(
            !ok(tiled, misaligned, 4096),
            "an X window {misaligned} bytes in is not addressable as vec4"
        );
    }
    // ... nor as the row stride, which is the window's `k * 4` bytes: a
    // 257-element contraction strides 1028 B, four bytes off a group. The guard
    // is about addressability, not block geometry — a `k` that is a multiple of
    // four but not of a whole 256-element block (e.g. 384) passes it and is
    // truncated by the shader's `k / 256`, which is that shape's existing
    // behaviour.
    assert!(
        !ok(tiled, 16, 257 * 4),
        "a 257-element contraction's row stride"
    );
    // Q1_0 (the other arm) reads X a scalar at a time, so neither matters to it.
    assert!(ok(24, 4, 257 * 4));
}

/// The route ladder's answer, pinned device-free. The batch boundary joins
/// exactly on [`rlx_wgpu::gguf_gpu::DequantMatmulPath::OneRowDevice`], so the
/// predicate must be an exact test of the one-row tiled arm rather than its bare
/// window guard: it has to compose the arm's addressability check with the
/// tiled-arm scheme check. Q1_0 passes that guard (its arm reads X a scalar at a
/// time) but has no tiled branch, and a K-quant whose window is misaligned is
/// refused — joining either would encode work the tiled kernel cannot read, and
/// the batch boundary would have handed it a submission instead of the host.
///
/// The ladder is also deliberately pure: it records nothing, so asking it what
/// the route will be does not perturb [`rlx_wgpu::gguf_gpu::route_counts`].
#[test]
fn select_path_is_exact_for_the_one_row_device_arm() {
    use rlx_wgpu::gguf_gpu::{DequantMatmulPath, select_path};

    // Q4_K (0) / Q6_K (2) at one row with an addressable X window — the shape a
    // decode calls at, and the only one the boundary may join. `k = 384` is not a
    // whole number of 256-element blocks but the window is still addressable
    // (1536 B, a multiple of 16), and at m=1 that is enough: the arm's guard is
    // about addressability, not block geometry.
    for (name, id) in [("Q4_K", 0u32), ("Q6_K", 2)] {
        for k in [256usize, 384] {
            for scratch_off in [0usize, 4096] {
                assert_eq!(
                    select_path(1, k, id, 64, scratch_off),
                    DequantMatmulPath::OneRowDevice,
                    "{name} (id {id}) m=1 k={k} scratch_off={scratch_off}: an \
                     addressable one-row window is device-only and must join the \
                     enclosing batch"
                );
            }
        }
    }

    // Q1_0 (24) at one row passes the same window guard — its arm reads X a
    // scalar at a time — but is not the tiled arm, so it must not be joined.
    for k in [256usize, 384] {
        assert_eq!(
            select_path(1, k, 24, 64, 0),
            DequantMatmulPath::FusedArm,
            "Q1_0 (24) m=1 k={k}: the bare guard passes but the scheme is not the \
             tiled arm, so the ladder must say Fused"
        );
    }

    // Q4_K / Q6_K above one row: the row-grouped GEMM when k is a whole number of
    // packing blocks (whatever slab the plan booked), and the scratch / host
    // fallback when it is not.
    for id in [0u32, 2] {
        for scratch_off in [0usize, 4096] {
            assert_eq!(
                select_path(4, 256, id, 64, scratch_off),
                DequantMatmulPath::FusedArm,
                "id {id} m=4 k=256 scratch_off={scratch_off}: a whole number of \
                 blocks is the row-grouped GEMM, never the joined one-row arm"
            );
        }
        assert_eq!(
            select_path(4, 384, id, 64, 4096),
            DequantMatmulPath::ScratchArm,
            "id {id} m=4 k=384 with a slab: not a whole number of blocks, so the \
             scratch entry point — `ScratchArm` names the entry point, not the \
             recorded route, and for Q4_K/Q6_K it hands the call back to the fused \
             arm's per-row loop (`route_counts` is the arbiter)"
        );
        assert_eq!(
            select_path(4, 384, id, 64, 0),
            DequantMatmulPath::Host,
            "id {id} m=4 k=384 with no slab: nothing to dequantize into, so the host"
        );
    }

    // A scheme with no fused kernel is scratch / host at every row count. Q8_0 has
    // one arm inside the scratch entry point — the row-grouped GEMM at `m > 1` with
    // a staged-tile-aligned `k` — but that is the entry point's own choice, not
    // this selector's answer, so it is pinned by the route census in
    // `gguf_matmul_route.rs` and not here. One row takes the slab rather than a
    // per-row GEMV for both schemes, and that is the whole of this test's claim.
    for (name, scheme) in [
        ("Q5_0", rlx_ir::quant::QuantScheme::GgufQ5_0),
        ("Q8_0", rlx_ir::quant::QuantScheme::GgufQ8_0),
    ] {
        let id = rlx_wgpu::gguf_host::gguf_scheme_id(scheme);
        for m in [1usize, 4] {
            assert_eq!(
                select_path(m, 256, id, 64, 4096),
                DequantMatmulPath::ScratchArm,
                "{name} (id {id}) m={m}: no fused kernel, so a booked slab means scratch"
            );
            assert_eq!(
                select_path(m, 256, id, 64, 0),
                DequantMatmulPath::Host,
                "{name} (id {id}) m={m}: no fused kernel and no slab, so the host"
            );
        }
    }

    // A K-quant at one row with a window the arm's `vec4` X read cannot address
    // must not be joined either — the fused entry point refuses it and records the
    // host route.
    for id in [0u32, 2] {
        assert_eq!(
            select_path(1, 256, id, 4, 0),
            DequantMatmulPath::FusedArm,
            "id {id} m=1 with a 4-byte-aligned X window: not addressable as vec4, \
             so the boundary must not join it"
        );
    }

    // The selector is pure by construction; the route census it is checked
    // against lives in `gguf_matmul_route.rs` / `gguf_batch_join.rs`.
}

/// No registered scheme may be left unhandled. Today every one of them has a
/// WGSL branch somewhere; if that ever stops being true, the scheme must be
/// listed here so it is host-routed on purpose rather than silently mis-decoded.
#[test]
fn every_registered_scheme_is_branched_or_deliberately_host_routed() {
    /// `(id, why)` — schemes with no WGSL kernel, host-routed by choice.
    const HOST_ONLY: &[(u32, &str)] = &[];

    let in_shader = branches_in_shader();
    let registered = registered_ids();
    assert!(
        registered.len() >= 29,
        "expected the full shared id table, walked {}",
        registered.len()
    );

    for id in &registered {
        let branched = in_shader.contains(id);
        let host_only = HOST_ONLY.iter().any(|(h, _)| h == id);
        let scheme = rlx_ir::quant::QuantScheme::from_gpu_dequant_scheme_id(*id).unwrap();
        assert!(
            branched || host_only,
            "scheme {scheme:?} (id {id}) has no WGSL branch and is not listed \
             as host-only — this is the configuration that returns zeros"
        );
        assert!(
            !(branched && host_only),
            "scheme {scheme:?} (id {id}) is both branched and listed host-only"
        );
    }
}
