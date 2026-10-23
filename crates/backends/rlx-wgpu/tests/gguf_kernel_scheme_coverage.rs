// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every GGUF scheme registered in the shared id table must have a branch in
//! the WGSL entry point that handles it — or be explicitly host-routed.
//!
//! Three entry points carry scheme dispatch:
//!   * `dequant_gguf` (`dequant_gguf.wgsl`) — device dequant into f32 scratch;
//!   * `dequant_gemv_tiled` / `dequant_gemv` (`dequant_gemv_gguf.wgsl`) — the
//!     two one-row packed-multiply arms, Q4_K / Q6_K / Q8_0 and Q1_0
//!     respectively.
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

/// One module-scope `fn`'s body (up to its closing brace in column 0), for the
/// helpers [`entry_body`]'s cut does not reach: it stops at the next `@compute`,
/// which for a decode helper is several functions away.
fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let head = format!("fn {name}(");
    let at = src
        .find(&head)
        .unwrap_or_else(|| panic!("no `{head}` in the shader"));
    let rest = &src[at..];
    &rest[..rest.find("\n}").map_or(rest.len(), |i| i + 2)]
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
    // reports: both decoded arms read it 16 bytes at a time (`read_w16`) and pick
    // bytes out of the leg they fetched, where a word-at-a-time read cost one fetch
    // per word — the per-lane instruction count this latency-bound arm's rate
    // follows. The arm that keeps one thread per column still reads bytes. Both
    // one-row arms share one decode per scheme (`dq_q4k_unit` / `dq_q6k_unit` /
    // `dq_q8_0_unit`), so the spelling pins below read those bodies rather than
    // either entry point's.
    let wgsl = rlx_wgpu::kernels::DEQUANT_GEMV_GGUF_WGSL;
    let q4k = fn_body(wgsl, "dq_q4k_unit");
    let q6k = fn_body(wgsl, "dq_q6k_unit");
    let q8_0 = fn_body(wgsl, "dq_q8_0_unit");
    assert!(
        q6k.contains("read_w16(") && q8_0.contains("read_w16("),
        "the decoded one-row arms must read the packed weight in 16-byte legs"
    );
    assert!(
        !q6k.contains("read_w(") && !q8_0.contains("read_w("),
        "the decoded one-row arms' own reads must not fall back to the byte-wise `read_w`"
    );
    // …but the units *do* reach that reader, through `dq_read_f16` for a block's
    // exact two bytes of `d`. `!contains("read_w(")` above is satisfied by moving
    // a read one call away, so pin the indirection too: the module's byte-wise
    // reads are `dq_read_f16`'s two and the legacy column arm's one. A leg added
    // to any of the three bodies takes this to five.
    assert!(
        q6k.contains("dq_read_f16(") && q8_0.contains("dq_read_f16("),
        "a decoded arm's scale is read at its own two bytes through `dq_read_f16`"
    );
    assert_eq!(
        wgsl.matches("read_w(").count(),
        4,
        "byte-wise reads: the definition, `dq_read_f16`'s two, and the legacy arm's one"
    );
    // Q4_K's 144-byte stride puts its header and both `qs` halves on group
    // boundaries and off the second column's, so the body is six leg reads and no
    // word assembly. A word-at-a-time read computes bit-identical values, so no
    // parity or digest test can see this path change; like the reads above, this
    // file pins the spelling. One site, because the decode is one body both arms
    // call (it used to be two — one per column — inside the tiled arm).
    assert_eq!(
        q4k.matches("read_w16(").count(),
        6,
        "the Q4_K unit reads its header and both columns' two `qs` halves as legs"
    );
    assert!(
        q6k.contains("wbyte_i8("),
        "Q6_K's 16 i8 scales are picked out of the one leg they occupy"
    );
    assert!(
        entry_body(wgsl, "dequant_gemv").contains("read_w("),
        "the one-thread-per-column arm reads the packed weight a byte at a time"
    );
    // Each operand window is bound **once**. The layout this shader used to carry
    // bound the X window twice (`array<f32>` at binding 0 and `array<vec4<f32>>`
    // at 4) and the weight window twice (`array<u32>` at 2 and `array<vec4<u32>>`
    // at 5), all `read_write` — two writable bindings of one range, which WebGPU's
    // "bind groups alias a writable resource" rule forbids and this driver merely
    // happens not to enforce. Pinned by declaration count, because a re-added
    // alias is invisible to every parity and digest test: the arithmetic is
    // identical, only the conformance is not.
    let declarations: Vec<u32> = wgsl
        .match_indices("@binding(")
        .filter_map(|(at, tag)| {
            wgsl[at + tag.len()..]
                .split(')')
                .next()
                .and_then(|n| n.trim().parse().ok())
        })
        .collect();
    assert_eq!(
        declarations,
        [0, 1, 2, 3],
        "the one-row module binds each operand window once: X, params, weight, out"
    );
    assert!(
        !wgsl.contains("warr:") && !wgsl.contains("xarr:"),
        "a second view of a bound window is the aliasing defect, not an optimisation"
    );
}

/// The one-row route's device guard. Both arms read X through the buffer's
/// `vec4` view, so the window's offset must line up with a four-element group;
/// the tiled arm additionally reads across the row, so its stride must too.
/// Pinned here because the decision is arithmetic — the route consults this at
/// its own entry, so a call it refuses goes to the host instead of being read
/// shifted.
#[test]
fn the_one_row_device_guard_requires_a_vec4_addressable_x_window() {
    use rlx_wgpu::gguf_gpu::{gemv_one_row_device_ok as ok, gemv_tiled_supports_scheme};

    let tiled: Vec<u32> = (0u32..=64)
        .filter(|id| gemv_tiled_supports_scheme(*id))
        .collect();
    assert!(!tiled.is_empty(), "the tiled arm has at least one scheme");
    // Every scheme the tiled arm branches on, the 8-bit one included: the guard is
    // the arm's, not the first scheme's.
    for id in tiled {
        // A planned activation lands on a 16-byte boundary of its window and `k` is a
        // whole number of the scheme's blocks, so this is every call the models make.
        for aligned in [0usize, 16, 64, 1024, 256 * 3 + 32] {
            assert!(
                ok(id, aligned, 4096),
                "id {id}: an X window {aligned} bytes in is readable as vec4"
            );
        }
        // A 4-byte-aligned view of one is not — neither as the window ...
        for misaligned in [4usize, 8, 12, 256 + 12, 1024 + 4] {
            assert!(
                !ok(id, misaligned, 4096),
                "id {id}: an X window {misaligned} bytes in is not addressable as vec4"
            );
        }
        // ... nor as the row stride, which is the window's `k * 4` bytes: a
        // 257-element contraction strides 1028 B, four bytes off a group. The guard
        // is about addressability, not block geometry — a `k` that is a multiple of
        // four but not of a whole 256-element block (e.g. 384) passes it, and for
        // the K-quants the shader's `k / 256` then truncates it, which is that
        // shape's existing behaviour.
        assert!(
            !ok(id, 16, 257 * 4),
            "id {id}: a 257-element contraction's row stride"
        );
    }
    // Q1_0 (the other arm) reads X through the same `vec4` view, so the window's
    // offset has to line up too; its stride does not enter — it indexes the view
    // component-wise.
    assert!(!ok(24, 4, 257 * 4));
    assert!(ok(24, 16, 257 * 4));
}

/// The route ladder's answer, pinned device-free. The batch boundary joins
/// exactly on [`rlx_wgpu::gguf_gpu::DequantMatmulPath::is_provably_device_only`],
/// so that predicate must be an *exact* test of the ladder's device-only answers:
/// the tiled one-row arm
/// ([`rlx_wgpu::gguf_gpu::DequantMatmulPath::OneRowDevice`]) — the K-quants', and
/// new for the 8-bit scheme that scheme's own at one row, over its own 32-element
/// blocks — the row-grouped GEMM above it, which the same windowed entry point
/// reaches at the row count the run encodes it at
/// ([`rlx_wgpu::gguf_gpu::DequantMatmulPath::GroupedDevice`]), and — also new for
/// the 8-bit scheme — that scheme's multi-row grouped arm. Those last two are
/// refinements of the slab answer that scheme used to get: the multi-row one is
/// admitted on the scheme's *own packing block*
/// ([`rlx_wgpu::gguf_gpu::grouped_k_ok`]), which is the granularity that arm's
/// weight loop reads — for the K-quants that block is the 256-element staged tile
/// ([`rlx_wgpu::gguf_gpu::GROUPED_STAGE_ELEMS`]) the shared entry points stage, so
/// their admission is exactly what it was, while Q8_0's 32-element block is an
/// eighth of that tile, so a whole number of its blocks lets a row end inside one
/// and still be admitted, which the 8-bit entry point's own tail-tile bound is what
/// serves — and on the shared `vec4` X-window guard that kernel's encoder applies,
/// which the ladder restates device-free; the one-row one is that scheme's own
/// block-keyed branch of the tiled kernel, admitted on its shape terms alone,
/// exactly as the K-quants' is. Every other call keeps exactly the answer it had —
/// the scheme's slab when a slab was booked, the host when none was — so the
/// negative cases below are half of this claim and are pinned too.
///
/// It has to compose the arm's own X-window addressability guard
/// (`gemv_one_row_device_ok`), so a window the arm's `vec4` reads cannot address
/// stays out; Q1_0 passes that guard on the window's offset alone (its legacy arm
/// indexes the same view component-wise) but has no tiled branch, so it must not
/// be admitted either — joining it would encode work the tiled kernel cannot
/// read, and the batch boundary would have handed it a submission instead of the
/// host.
///
/// The ladder is also deliberately pure: it records nothing, so asking it what
/// the route will be does not perturb [`rlx_wgpu::gguf_gpu::route_counts`].
#[test]
fn select_path_is_exact_for_the_joined_device_arms() {
    use rlx_wgpu::gguf_gpu::{DequantMatmulPath, select_path};

    // The boundary's predicate is a total function of the variant, so it is pinned
    // once, over all five. The ladder's answers for the calls are pinned after it,
    // per call: together the two say the join set is exactly the two arms a caller
    // can encode, and that the three which can still reach the host stay out.
    for path in [
        DequantMatmulPath::OneRowDevice,
        DequantMatmulPath::GroupedDevice,
    ] {
        assert!(
            path.is_provably_device_only(),
            "{path:?} has no host arm and must join the batch"
        );
    }
    for path in [
        DequantMatmulPath::FusedArm,
        DequantMatmulPath::ScratchArm,
        DequantMatmulPath::Host,
    ] {
        assert!(
            !path.is_provably_device_only(),
            "{path:?} can reach the host and must stay out of the join set"
        );
    }

    // Q4_K (0) / Q6_K (2) at one row with an addressable X window — the shape a
    // decode calls at. `k = 384` is not a whole number of 256-element blocks but
    // the window is still addressable (1536 B, a multiple of 16), and at m=1 that
    // is enough: the arm's guard is about addressability, not block geometry.
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
    // Q1_0 (24) at one row passes the same window guard on the offset alone — its
    // legacy arm indexes the view component-wise, so the row stride does not enter
    // — but is not the tiled arm, so it must not be joined.
    for k in [256usize, 384] {
        assert_eq!(
            select_path(1, k, 24, 64, 0),
            DequantMatmulPath::FusedArm,
            "Q1_0 (24) m=1 k={k}: the bare guard passes but the scheme is not the \
             tiled arm, so the ladder must say Fused"
        );
    }

    // Pin (a) — the ladder's answer for a multi-row K-quant call, over the input
    // classes that decide it: the scheme, the slab the plan booked, a row count from
    // the bodies' measured class beside the smallest above one row, and two
    // contraction widths that are whole numbers of packing blocks. Each is the
    // row-grouped GEMM, whose entry point a caller encodes into its own batch
    // exactly as it does the one-row arm — and the predicate pinned above reads that
    // exact answer, so the admission is not a second rule that could disagree.
    for (name, id) in [("Q4_K", 0u32), ("Q6_K", 2)] {
        for m in [2usize, 13] {
            for k in [256usize, 2048] {
                for scratch_off in [0usize, 4096] {
                    assert_eq!(
                        select_path(m, k, id, 64, scratch_off),
                        DequantMatmulPath::GroupedDevice,
                        "{name} (id {id}) m={m} k={k} scratch_off={scratch_off}: a \
                         whole number of blocks above one row is the row-grouped GEMM"
                    );
                }
            }
        }
    }

    // The negative half — the calls that share the arm but must stay *out* of the
    // admission. Both a multi-row window the arm's `vec4` X read cannot address ...
    for id in [0u32, 2] {
        assert_eq!(
            select_path(4, 256, id, 4, 0),
            DequantMatmulPath::FusedArm,
            "id {id} m=4 with a 4-byte-aligned X window: not addressable as vec4, \
             so the boundary must not join it"
        );
    }
    // ... and Q1_0 above one row, which passes the bare guard but is not the tiled
    // arm.
    assert_eq!(
        select_path(4, 256, 24, 64, 0),
        DequantMatmulPath::FusedArm,
        "Q1_0 (24) m=4: the bare guard passes but the scheme is not the tiled arm, \
         so the ladder must say Fused"
    );

    // Q4_K / Q6_K above one row with a `k` that is not a whole number of packing
    // blocks: the scratch / host fallback, never the grouped device route.
    for id in [0u32, 2] {
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

    // The 8-bit scheme (Q8_0, 20) is the refinement's own case. It has no arm in
    // the fused entry point, so above one row it used to take its slab for every
    // call; it is now the device-only grouped arm whenever all four gates hold — the
    // plan booked the slab, more than one row, a whole number of this scheme's own
    // 32-element packing blocks, and an X window that kernel's `vec4` reads can
    // address. Its one row is the tiled arm's own branch, pinned below. The `k` term
    // is the packing block and not the arm's 256-element staged tile, so its cells
    // run through a single block up to several tiles.
    let q8_0 = rlx_wgpu::gguf_host::gguf_scheme_id(rlx_ir::quant::QuantScheme::GgufQ8_0);
    for k in [32usize, 128, 256, 288, 384, 896, 2048] {
        for m in [2usize, 4, 13] {
            assert_eq!(
                select_path(m, k, q8_0, 64, 4096),
                DequantMatmulPath::GroupedDevice,
                "Q8_0 (id {q8_0}) m={m} k={k} with a slab: a multi-row call over whole \
                 32-element blocks of this scheme's own (any multiple of 32, not only \
                 whole 256-element staged tiles) with a `vec4`-addressable X window is \
                 device-only and must join the enclosing batch"
            );
        }
    }

    // Each gate has its own negative. A contraction that is not a whole number of
    // those blocks, which is the one thing the arm's weight loop cannot read: the
    // format never carries a partial block with real packed bytes, so only this
    // device-free ladder can exercise the gate — the same reason the one-row arm's
    // `k = 100` cell below is device-free-only.
    assert_eq!(
        select_path(4, 100, q8_0, 64, 4096),
        DequantMatmulPath::ScratchArm,
        "Q8_0 (id {q8_0}) m=4 k=100 with a slab: 100 is not a whole number of \
         32-element blocks, so the grouped arm is not asked and the slab stands"
    );
    // A window the grouped kernel's `vec4` X read cannot address ...
    assert_eq!(
        select_path(4, 256, q8_0, 4, 4096),
        DequantMatmulPath::ScratchArm,
        "Q8_0 (id {q8_0}) m=4 with a 4-byte-aligned X window and a slab: the grouped \
         kernel reads X as `vec4`, so the slab stands"
    );
    // ... no slab to dequantize into ...
    assert_eq!(
        select_path(4, 256, q8_0, 64, 0),
        DequantMatmulPath::Host,
        "Q8_0 (id {q8_0}) m=4 with no slab: nothing to dequantize into, so the host"
    );
    // ... one row, which is now this scheme's own arm: the tiled kernel's branch
    // keyed on this scheme's own 32-element block, joined on its shape terms
    // alone ...
    assert_eq!(
        select_path(1, 256, q8_0, 64, 4096),
        DequantMatmulPath::OneRowDevice,
        "Q8_0 (id {q8_0}) m=1 with a slab: the 8-bit scheme's single row is the \
         tiled arm's own block-keyed branch, over whole 32-element blocks, so it \
         joins the enclosing batch"
    );
    // ... a contraction that is not a whole number of those blocks, which the
    // branch's own `k / 32` unit count would floor: the arm is not asked and
    // today's slab answer stands. The format never carries a partial block with
    // real packed bytes, so only this device-free ladder can exercise the gate.
    assert_eq!(
        select_path(1, 100, q8_0, 64, 4096),
        DequantMatmulPath::ScratchArm,
        "Q8_0 (id {q8_0}) m=1 k=100 with a slab: 100 is not a whole number of \
         32-element blocks, so the one-row arm is not asked and the slab stands"
    );
    // ... a one-row window the arm's `vec4` X read cannot address: the fused entry
    // point's own refusal, which hands the call to the host — the same outcome the
    // K-quants' one-row arm produces for that window, pinned at the end of this
    // test.
    assert_eq!(
        select_path(1, 256, q8_0, 4, 4096),
        DequantMatmulPath::FusedArm,
        "Q8_0 (id {q8_0}) m=1 with a 4-byte-aligned X window: not addressable as \
         vec4, so the boundary must not join it"
    );
    // ... and no slab booked at all, which the one-row arm does not read: it is
    // admitted on its shape terms alone, exactly as the K-quants' is
    // (`select_path(1, 256, 0, 64, 0)` is `OneRowDevice` today) — the slab is not
    // a term of it.
    assert_eq!(
        select_path(1, 256, q8_0, 64, 0),
        DequantMatmulPath::OneRowDevice,
        "Q8_0 (id {q8_0}) m=1 with no slab: the one-row arm's terms are its own \
         blocks and the X window, so the booking cannot keep it out"
    );
    // ... and a row count of zero, which is not above one row either, nor the new
    // term's `m == 1` exactly.
    assert_eq!(
        select_path(0, 256, q8_0, 64, 4096),
        DequantMatmulPath::ScratchArm,
        "Q8_0 (id {q8_0}) m=0 with a slab: a row count of zero is not `m > 1` and \
         not the one-row arm's `m == 1`, so the slab stands"
    );

    // Q5_0 (22) is the out-of-scope control: no fused kernel and no grouped kernel
    // at any row count, so its slab / host answer is exactly what it was.
    let q5_0 = rlx_wgpu::gguf_host::gguf_scheme_id(rlx_ir::quant::QuantScheme::GgufQ5_0);
    for m in [1usize, 4] {
        assert_eq!(
            select_path(m, 256, q5_0, 64, 4096),
            DequantMatmulPath::ScratchArm,
            "Q5_0 (id {q5_0}) m={m}: no fused and no grouped kernel, so a booked slab \
             means scratch"
        );
        assert_eq!(
            select_path(m, 256, q5_0, 64, 0),
            DequantMatmulPath::Host,
            "Q5_0 (id {q5_0}) m={m}: no kernel at all and no slab, so the host"
        );
    }

    // The pin on what that one list says, by value, and over the whole id range
    // rather than only the ids the `fused` term leaves to this admission: for Q4_K
    // and Q6_K that term answers the grouped arm first, so a per-call reading of the
    // ladder could not see them at all. `grouped_scheme_supported` is the kernel set
    // read without a device, and `select_path`'s multi-row admission is its one
    // caller — so the set is named here rather than read back from the predicate. It
    // must stay Q4_K, Q6_K and Q8_0; a scheme added to or dropped from that list has
    // to re-point this expectation deliberately.
    const GROUPED: [u32; 3] = [0, 2, 20]; // Q4_K, Q6_K, Q8_0
    let supported: Vec<u32> = (0u32..=64)
        .filter(|id| rlx_wgpu::gguf_gpu::grouped_scheme_supported(*id))
        .collect();
    assert_eq!(
        supported, GROUPED,
        "the row-grouped GEMM's scheme list must be exactly {GROUPED:?} — a scheme \
         added to or dropped from it has to re-point this pin deliberately"
    );

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

    // The grouped arm's `k` term is the scheme's own packing block in both schemes'
    // cases, and that is what the admission above inherits rather than restating:
    // `select_path` admits a multi-row K-quant only through its `fused` term, which
    // requires `k` a whole number of the scheme's own block, and for the two tiled
    // schemes that block *is* `GROUPED_STAGE_ELEMS` — the staged tile the grouped
    // GEMM's weight loop barriers fall at, so their admission is unchanged and an
    // admitted call still cannot be routed down the per-row loop, whose shader
    // truncates `k` to that same 256-element unit.
    assert_eq!(
        rlx_ir::quant::QuantScheme::GgufQ4K.gguf_block_size() as usize,
        rlx_wgpu::gguf_gpu::GROUPED_STAGE_ELEMS,
        "Q4_K's packing block must be the grouped GEMM's staged tile"
    );
    assert_eq!(
        rlx_ir::quant::QuantScheme::GgufQ6K.gguf_block_size() as usize,
        rlx_wgpu::gguf_gpu::GROUPED_STAGE_ELEMS,
        "Q6_K's packing block must be the grouped GEMM's staged tile"
    );
    // The 8-bit scheme's own block is a *fraction* of that tile — an eighth — so a
    // whole number of its blocks can end inside one, and the block term is what its
    // admission turns on: the tail tile of such a row is the entry point's own to
    // bound, which is why the tile is not the gate and why the cells above are
    // admitted at 32, 128, 288 and 896 alike.
    assert_eq!(
        rlx_ir::quant::QuantScheme::GgufQ8_0.gguf_block_size(),
        32,
        "Q8_0's own packing block"
    );
    assert_eq!(
        rlx_wgpu::gguf_gpu::GROUPED_STAGE_ELEMS,
        8 * rlx_ir::quant::QuantScheme::GgufQ8_0.gguf_block_size() as usize,
        "the grouped GEMM's staged tile is eight Q8_0 blocks"
    );

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

/// The one-row uniform record is written by the host and read by the shader, so
/// the two declarations are one layout: the host struct must be exactly the size
/// the WGSL struct occupies — its members' bytes rounded up to the 16-byte
/// alignment a uniform struct has. A word added on one side only leaves the
/// record the shader reads shifted by whatever the binding hands it, which this
/// driver accepts silently (the layout asks for no minimum size).
#[test]
fn the_one_row_uniform_record_is_the_size_the_shader_declares() {
    let src = rlx_wgpu::kernels::DEQUANT_GEMV_GGUF_WGSL;
    let at = src
        .find("struct Params {")
        .expect("the one-row shader declares `struct Params`");
    let body = &src[at..src[at..].find('}').expect("a struct closes") + at];
    let words = body.matches(':').count();
    let wgsl_bytes = words.div_ceil(4) * 16;
    assert_eq!(
        std::mem::size_of::<rlx_wgpu::kernels::DequantGemvGgufParams>(),
        wgsl_bytes,
        "the host record and the shader's `struct Params` ({words} words) must be one layout"
    );
}
