// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a packed multiply costs the pass it sits in: submissions and device
//! waits, from the backend's process-global counters
//! ([`rlx_wgpu::device_submit_count`] / [`rlx_wgpu::device_wait_count`]). A call
//! of the tiled K-quant arms' windowed route — the one-row arm, and the
//! row-grouped GEMM it becomes above one row — is device-only work that belongs
//! *inside* the enclosing run of device work: it takes the batch's own
//! submission (the batch submits once per encode chunk, `RLX_WGPU_ENCODE_CHUNK`)
//! and adds none of its own. The 8-bit scheme's multi-row calls over whole 32-element
//! blocks of its own are that same kind of work, through the same grouped arm, once
//! the ladder admits them.
//!
//! Nine brackets, each one forward pass between a counter reset and a read of
//! them. **1 — the join set**, `dq(Q4_K) → mm → dq(Q6_K) → rv → mm`, compiled at a
//! bucket and run at one row of it: the join set is the tiled one-row arm's two
//! K-quant schemes at **every row count** — the tiled one-row arm at one row and the
//! row-grouped GEMM above it — while Q1_0 keeps its own arm, the 8-bit scheme's
//! multi-row calls over whole 32-element blocks of its own join through that same
//! grouped arm (its single row takes the tiled one-row arm's own 8-bit branch, a
//! call off those blocks keeps the scratch entry point it had, and an admitted call
//! whose grouped encode refuses falls through to the host arm, which recomputes it
//! on the slab — brackets 8 and 9), and the scratch route answers as it did. So this
//! graph covers it completely — the
//! 4-bit half, the 6-bit half, and `rv`, an `Op::Reverse` on the last axis, the
//! filler that consumes a uniform, bind-group *and* meta-buffer lane while staying
//! device work on both legs. It reads **1 submission and 1 wait** against **5 and
//! 4** unjoined, with the route census `per_row = 2` and nothing else. **6 — the
//! same graph with the route's own N-chunking**: its first weight widened and the
//! weight-window budget lowered for the bracket (`RLX_WGPU_MAX_BIND_MB=1`, the
//! crate's own lever, used by `gguf_matmul_route.rs`), so the one-row arm splits the
//! multiply into column chunks — dispatches and their copies *inside* the joined
//! encoder, which must not add a submission or a wait: it reads the same **1 and 1**
//! against the same 5 and 4, which is what covers the chunked arm of the joined
//! encode. **7 — the same graph at its compiled row count**, no `set_active_extent`:
//! both packed multiplies are then above one row, so both take the row-grouped GEMM
//! and both are still joined — the route census is `grouped = 2`, the counted pair
//! again **1 and 1** against the same 5 and 4, and the pinned identity is that the
//! joined and the unjoined run of the same executable are **bit-identical**
//! element-wise: the mode changes the submission arrangement, not the route or the
//! arithmetic. **8 — the join set in the 8-bit scheme**: bracket 7's graph with
//! Q8_0 on both multiplies, the scheme the grouped arm's refinement added to the
//! join set. Its own packing block is 32 elements and that block — not the
//! 256-element staged tile the arm stages eight of — is the admission's `k` term,
//! and `K` is a whole number of both, so both
//! multiplies are the joined grouped arm — the census `grouped = 2`, the counted
//! pair again **1 and 1** against the same 5 and 4, and the same bit-identity
//! against the unjoined run. The counts are the witness here and the census is not:
//! the scratch entry point that took these calls before the arm reached this same
//! grouped kernel and recorded this same `grouped = 2`, so only the pair and the
//! identity tell the two arrangements apart. **9 — the refused admitted
//! call**: bracket 4's fixture at `BUCKET` rows on the 8-bit scheme, with the
//! arm's own bisect lever `RLX_WGPU_Q8_0_GEMM_DISABLE=1` (the ladder reads no
//! lever — `select_path` still answers the grouped arm) beside
//! `RLX_WGPU_CONCAT_HOST=1` + `RLX_WGPU_FORCE_HOST=1`: the call is still joined,
//! the encode inside the batch refuses, the tail submits that batch carrying no
//! work for the call, and the run's own arm recomputes on the f32 slab. One
//! executable and one arena serve both legs of that A/B — **3 and 4** lever off
//! against **6 and 5** lever on, the census `grouped = 1` then `scratch = 1`, and
//! the lever-on leg still matching the CPU reference is what says the flush the
//! boundary made before its (declined) encode carried the host step's deferred
//! write, so the recompute read fresh data. **2 — a host-adjacent graph**, device+host by construction, a
//! hosted call then a joined **row-grouped** K-quant multiply — above one row, so
//! the batch carries the grouped arm — then another hosted call: **6 and 8**
//! against 6 and 6 unjoined — so joining is not wait-neutral: it trades the
//! per-call submission for the two boundary polls. **3 — the one-row host case**, through the
//! route's own entry point with a `+4` X window (the only way in: the planner
//! aligns every activation slot) — **2 and 2**, counters covering the shared
//! readback helper that leg stages through. **4 — ConcatHost → joined**: a host
//! step whose output goes into the deferred host mirror instead of the arena,
//! `Step::ConcatHost` forced by the bracket's own `RLX_WGPU_CONCAT_HOST=1` +
//! `RLX_WGPU_FORCE_HOST=1`, must reach the device before the joined multiply reads
//! it — **3 and 4** against 4 and 5 unjoined. **5 — a host `BufferCopy` → joined**:
//! the one host arm that `continue`s past the shared post-host-step door, so only
//! the join block's own flush can carry its deferred write to the device
//! (`RLX_WGPU_HOST_BUFFER_COPY=1` forces the copy to the host route) — **2 and 2**
//! against 3 and 4 unjoined: the flush rides the batch's own submission, where the
//! multiply's pre-step door (the unjoined arm) still polls the device.
//!
//! The unjoined pairs are this same binary under `RLX_WGPU_ONE_OP_PER_PASS=1`,
//! which is also what pins the join not reaching into the step-by-step path (that
//! mode is inert by design). For brackets 2–5 that run reads exactly what the change
//! removed (6/6, 2/2, 4/5, 3/4 — bracket 5's own pair being 3/4 there, its flush then
//! polling for itself). Brackets 1, 6, 7 and 8 are five-step device runs
//! and that mode submits each of those steps, so they read 5/4 there, while the join
//! disabled with the counters in place reads 4/4 — the pair the join set's own gain
//! is measured on. (Brackets 7 and 8 measure the unjoined leg in-process when the
//! harness starts in the default mode; a harness already in that mode has it as the
//! bracket's first run.) Bracket 9 has no unjoined pair of that kind: its two legs
//! are the lever's own A/B, and the step-by-step mode does not arrange them
//! differently beyond leaving both unjoined (4 and 5, then 5 and 5).
//!
//! The `interleaved` graphs (1, 6, 7 and 8) are all-device on both legs: the packed
//! route comes from shape and slab only, a dense matmul is hosted only by
//! `RLX_WGPU_HOST_MATMUL` or a striped/unbindable arena, and `Reverse` only by
//! `RLX_WGPU_FORCE_HOST` / sharding / a sharded arena — no adapter-kind term in
//! either. So their integers are adapter-independent, and so is 6's budget lever:
//! `RLX_WGPU_MAX_BIND_MB` only lowers the device limit, and a different chunk count
//! adds dispatches and copies inside the encoder, never a submission or a wait. The
//! one opt-in that could add steps,
//! the Vulkan cooperative-matrix f16 path, needs
//! `RLX_WGPU_COOP_F16_VK_ENABLE` and an alignment these graphs lack. Brackets 2 and
//! 3 are adapter-independent too: 2's host legs come from the ladder's shape-and-slab
//! rule, and 3's arm refuses on the X window, not on the adapter. Brackets 4, 5 and 9
//! are knob-scoped instead: their own `RLX_WGPU_CONCAT_HOST` / `RLX_WGPU_FORCE_HOST`
//! / `RLX_WGPU_HOST_BUFFER_COPY` force the host step (9's the scheme's own bisect
//! lever as well, which only lowers the grouped arm's encode), but the steps around
//! it are the
//! adapter's choice (`wgpu_prefer_structure_host` includes `coop_discrete_backend()`,
//! true for any Vulkan or DX12 adapter), so their integers are asserted on a
//! non-discrete adapter only — leg-independence is *not* claimed for them. Their
//! parity and their route census are asserted on every leg.
//!
//! Brackets 4 and 5 are the load-bearing parity pins of the deferred-mirror
//! flushes: removing the flush that carries the host step to the device (both
//! sites for 4, the join block's own for 5) measures `max_abs` 5.70e1 and 5.45e1
//! against the CPU reference, where the flush is in place the two read 7.63e-6.
//! Bracket 9 reaches the same door with the call's work refused rather than
//! encoded, so there the flush has to cover a recompute: its lever-on leg is a pin
//! for the same reason, one step further along.
//!
//! `RLX_WGPU_SNAPSHOT=1` is rejected by the guard at the top: its replay runs the
//! schedule once per prefix, so the readings stop meaning one forward pass. With
//! that guard bypassed, the snapshot's per-step dump on this graph reads its final
//! step at 838.137 — the CPU `|out|` for the one-row graph — so "the debugging path
//! sees drained data".
//!
//! The two tests in this binary both bracket process-global counters, so both take
//! `common::route_lock()`: one runs at a time, and neither resets a counter under
//! the other. Each sets only its own knobs, unsets them inside its bracket, and the
//! guard at the top rejects the ones that would change what a submission or a wait
//! means.

mod common;

use common::{Q8_0_BISECT, RUN_REPORT_SWITCH, max_abs_diff};
use rlx_ir::op::Activation;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::gguf_gpu::{reset_route_counts, route_counts};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// The compiled bucket. One row is what the models run at, so this is the
/// shape the pin has to be wrong for if the decision is taken at compile time.
const BUCKET: usize = 4;
/// One Q4_K superblock — a whole number of packing blocks, which is what the
/// one-row arm answers over correctly.
const K: usize = 256;
/// Output width of every projection in the pin graph.
const N: usize = 64;
/// The contract of the hosted leg: 1.5 superblocks, so the one-row arm's own
/// route selection cannot take it at more than one row.
const HOST_K: usize = 384;

/// The tolerance pair the packed-multiply parity grids use for these shapes
/// (Q4_K, k a whole number of superblocks, m == 1).
const TOL: f32 = 1e-2;

/// The packed scheme a join fixture's multiplies carry: the tiled K-quants, the
/// pair the join set was pinned on first, or the 8-bit scheme, which the grouped
/// arm's refinement added to it.
enum JoinScheme {
    KQuant,
    Q8_0,
}

/// Deterministic packed Q4_K bytes for a `[k, n]` weight.
fn packed(k: usize, n: usize) -> Vec<u8> {
    common::packed_gguf_bytes(rlx_gguf::GgmlType::Q4K, k, n)
}

/// [`packed`] in the 6-bit K-block scheme (Q6_K) — the join set's other half.
/// Same 256-element superblock, so a graph may mix the two at one `K`.
fn packed_q6k(k: usize, n: usize) -> Vec<u8> {
    common::packed_gguf_bytes(rlx_gguf::GgmlType::Q6K, k, n)
}

/// [`packed`] in the 8-bit scheme (Q8_0). Its own block is 32 elements — the term
/// this scheme's grouped admission asks for, and not the 256-element staged tile
/// the arm stages eight of — so a `K` that is a whole number of these blocks is
/// admitted whether or not it is a whole number of tiles.
fn packed_q8_0(k: usize, n: usize) -> Vec<u8> {
    common::packed_gguf_bytes(rlx_gguf::GgmlType::Q8_0, k, n)
}

/// Deterministic values for an `n`-element tensor or parameter.
fn feed(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * step).sin()).collect()
}

/// The rebind fixture: `relu(dq(x) · w)` over `x: [Dynamic(0), k]` and a packed
/// `[n, k]` Q8_0 weight — the 8-bit scheme's one-row arm at one row, with a `k`
/// that is a whole number of its 32-element blocks. The relu is a second node so
/// the output is not the raw multiply, and the row axis stays dynamic so every
/// run's binding is the caller's to change.
fn dynamic_row_packed_multiply(k: usize, n: usize) -> (Graph, Vec<u8>) {
    let packed = packed_q8_0(k, n);
    let mut g = Graph::new("wgpu_gguf_dynamic_row_rebind");
    let x = g.input(
        "x",
        Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(k)], DType::F32),
    );
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let out_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(n)], DType::F32);
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![x, w],
        out_shape.clone(),
    );
    let out = g.activation(Activation::Relu, y, out_shape);
    g.set_outputs(vec![out]);
    (g, packed)
}

/// `dq → mm → dq → rv → mm`, at `m` rows: each packed multiply contracts over
/// `K` (a whole number of superblocks) and is immediately followed by a dense
/// device step whose lane would move if the packed one consumed one. The two
/// packed multiplies cover both halves of the join set — the first 4-bit
/// (Q4_K), the second 6-bit (Q6_K) — and `rv` is an `Op::Reverse` on the last
/// axis: the filler that consumes a uniform, bind-group *and* meta-buffer lane
/// while staying device work on both legs. Returns the graph, its packed
/// params, its f32 params, and the `x` input at `m` rows.
#[allow(clippy::type_complexity)]
fn interleaved(
    m: usize,
) -> (
    Graph,
    Vec<(&'static str, Vec<u8>)>,
    Vec<(&'static str, Vec<f32>)>,
    Vec<f32>,
) {
    interleaved_n(JoinScheme::KQuant, m, N)
}

/// [`interleaved`] with the first packed multiply's output width set to `wide`
/// (and its following dense weight to `[wide, K]`) — the join set's shape with a
/// weight the route's window budget can split into column chunks — and both
/// packed multiplies in `scheme`: the tiled K-quant pair, or Q8_0, which the
/// grouped arm's refinement added to the join set (one `K` serves both, a whole
/// number of the scheme's blocks and of that kernel's staged tile).
#[allow(clippy::type_complexity)]
fn interleaved_n(
    scheme: JoinScheme,
    m: usize,
    wide: usize,
) -> (
    Graph,
    Vec<(&'static str, Vec<u8>)>,
    Vec<(&'static str, Vec<f32>)>,
    Vec<f32>,
) {
    let (w1, w3, s1, s3) = match scheme {
        JoinScheme::KQuant => (
            packed(K, wide),
            packed_q6k(K, N),
            QuantScheme::GgufQ4K,
            QuantScheme::GgufQ6K,
        ),
        JoinScheme::Q8_0 => (
            packed_q8_0(K, wide),
            packed_q8_0(K, N),
            QuantScheme::GgufQ8_0,
            QuantScheme::GgufQ8_0,
        ),
    };
    let mut g = Graph::new("wgpu_gguf_batch_join");
    let x = g.input("x", Shape::new(&[m, K], DType::F32));
    let packed_params: Vec<(&'static str, Vec<u8>)> = vec![("w1", w1), ("w3", w3)];
    let dense: Vec<(&'static str, Vec<f32>)> =
        vec![("p2", feed(wide * K, 0.017)), ("p4", feed(N * N, 0.019))];
    let w1 = g.param("w1", Shape::new(&[packed_params[0].1.len()], DType::U8));
    let w3 = g.param("w3", Shape::new(&[packed_params[1].1.len()], DType::U8));
    let p2 = g.param("p2", Shape::new(&[wide, K], DType::F32));
    let p4 = g.param("p4", Shape::new(&[N, N], DType::F32));
    let y1 = g.add_node(
        Op::DequantMatMul { scheme: s1 },
        vec![x, w1],
        Shape::new(&[m, wide], DType::F32),
    );
    let h2 = g.mm(y1, p2);
    let y2 = g.add_node(
        Op::DequantMatMul { scheme: s3 },
        vec![h2, w3],
        Shape::new(&[m, N], DType::F32),
    );
    let r = g.reverse(y2, vec![1]);
    let out = g.mm(r, p4);
    g.set_outputs(vec![out]);
    (g, packed_params, dense, feed(m * K, 0.023))
}

/// Compile `g` on `device` and upload its params.
fn build(
    device: Device,
    g: &Graph,
    packed: &[(&str, Vec<u8>)],
    dense: &[(&str, Vec<f32>)],
) -> rlx_runtime::CompiledGraph {
    let mut c = Session::new(device).compile(g.clone());
    for (name, bytes) in packed {
        c.set_param_typed(name, bytes, DType::U8);
    }
    for (name, data) in dense {
        c.set_param(name, data);
    }
    c
}

/// How the host leg's calls and the joined one sit next to each other: a hosted
/// call, then the joined one, then another hosted call — so the batch that carries
/// the joined call both opens after a host step (mirror flush + poll) and closes
/// before one (submit + poll, because that host step reads the arena).
///
/// `[4, 384]` activations whose only job is to be the input of a call the ladder
/// sends to the host (1.5 superblocks at more than one row books no dequant slab,
/// so the ladder refuses the fused arms), and `[4, 256]` for the joined call —
/// above one row on purpose, so the batch carries a *row-grouped* K-quant multiply
/// with host steps on both sides of it.
fn host_adjacent() -> (
    Graph,
    Vec<(&'static str, Vec<u8>)>,
    Vec<(&'static str, Vec<f32>)>,
) {
    let mut g = Graph::new("wgpu_gguf_batch_join_host");
    let xb = g.input("xb", Shape::new(&[BUCKET, HOST_K], DType::F32));
    let xa = g.input("xa", Shape::new(&[BUCKET, K], DType::F32));
    let xc = g.input("xc", Shape::new(&[BUCKET, HOST_K], DType::F32));
    let wh = g.param("wh", Shape::new(&[packed(HOST_K, N).len()], DType::U8));
    let wj = g.param("wj", Shape::new(&[packed(K, N).len()], DType::U8));
    let wc = g.param("wc", Shape::new(&[packed(HOST_K, N).len()], DType::U8));
    let yh = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![xb, wh],
        Shape::new(&[BUCKET, N], DType::F32),
    );
    let yj = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![xa, wj],
        Shape::new(&[BUCKET, N], DType::F32),
    );
    let yc = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![xc, wc],
        Shape::new(&[BUCKET, N], DType::F32),
    );
    g.set_outputs(vec![yh, yj, yc]);
    (
        g,
        vec![
            ("wh", packed(HOST_K, N)),
            ("wj", packed(K, N)),
            ("wc", packed(HOST_K, N)),
        ],
        vec![
            ("xb", feed(BUCKET * HOST_K, 0.029)),
            ("xa", feed(BUCKET * K, 0.031)),
            ("xc", feed(BUCKET * HOST_K, 0.037)),
        ],
    )
}

/// `concat(x1, x2) → dq`, the X operand of the packed multiply produced by a host
/// step: two `[m, 128]` graph inputs joined on the last axis to `[m, 256]` — one
/// Q4_K superblock, and eight Q8_0 blocks — and contracted against a packed
/// `[256, 64]` weight.
///
/// It is built to make the concat the host-classified step whose output is staged
/// into the deferred host mirror (see the fourth bracket in the test) rather than
/// written to the arena, so without the flush before the joined multiply the
/// multiply reads the slot the host step never wrote.
///
/// `scheme`'s K-quant form is that one-row fixture, untouched; its Q8_0 form is
/// `BUCKET` rows, because above one row is where the 8-bit scheme's grouped arm is
/// what the ladder names — the shape the refusal bracket drives with the arm's own
/// bisect lever.
fn mirrored_x(
    scheme: JoinScheme,
) -> (
    Graph,
    Vec<(&'static str, Vec<u8>)>,
    Vec<(&'static str, Vec<f32>)>,
) {
    let half = K / 2;
    let m = match scheme {
        JoinScheme::KQuant => 1,
        JoinScheme::Q8_0 => BUCKET,
    };
    let (bytes, quant) = match scheme {
        JoinScheme::KQuant => (packed(K, N), QuantScheme::GgufQ4K),
        JoinScheme::Q8_0 => (packed_q8_0(K, N), QuantScheme::GgufQ8_0),
    };
    let mut g = Graph::new("wgpu_gguf_batch_join_host_mirror");
    let x1 = g.input("x1", Shape::new(&[m, half], DType::F32));
    let x2 = g.input("x2", Shape::new(&[m, half], DType::F32));
    let w = g.param("w", Shape::new(&[bytes.len()], DType::U8));
    let c = g.add_node(
        Op::Concat { axis: 1 },
        vec![x1, x2],
        Shape::new(&[m, K], DType::F32),
    );
    let y = g.add_node(
        Op::DequantMatMul { scheme: quant },
        vec![c, w],
        Shape::new(&[m, N], DType::F32),
    );
    g.set_outputs(vec![y]);
    (
        g,
        vec![("w", bytes)],
        vec![("x1", feed(m * half, 0.023)), ("x2", feed(m * half, 0.041))],
    )
}

/// `x → cast → cast → dq`: X of the packed multiply produced by a `Step::BufferCopy`
/// that the run routes to the *host*. Both casts are value-preserving on this
/// f32-uniform arena (F32→F16 and F16→F32 are `Op::Cast`'s mode-0 copy), and
/// neither is a planner view — `pure_view_offset` aliases a `Cast` only when the
/// dtypes match — so each gets its own slot and the second copy's destination is
/// exactly the packed multiply's X window. With `RLX_WGPU_HOST_BUFFER_COPY=1`
/// that copy takes the host route (the run path's one route answer for a copy,
/// read at the batch boundary and at the copy's own arm) and, its source not a
/// weight, the write is deferred into the host mirror — where only the join
/// block's own flush can carry it to the device before the joined multiply reads
/// it.
fn host_copied_x() -> (
    Graph,
    Vec<(&'static str, Vec<u8>)>,
    Vec<(&'static str, Vec<f32>)>,
) {
    let mut g = Graph::new("wgpu_gguf_batch_join_host_copy");
    let x = g.input("x", Shape::new(&[1, K], DType::F32));
    let bytes = packed(K, N);
    let w = g.param("w", Shape::new(&[bytes.len()], DType::U8));
    let c = g.cast(x, DType::F16);
    // Second value-preserving cast (`parent dtype != to`, so not a planner view):
    // its `Step::BufferCopy` destination is exactly the packed multiply's X slot.
    let c = g.cast(c, DType::F32);
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![c, w],
        Shape::new(&[1, N], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, vec![("w", bytes)], vec![("x", feed(K, 0.023))])
}

/// One bracket of the join set above one row: the fixture to compile and run, the
/// scheme word that heads the lines it prints, and the wording that says what its
/// integers and its route census mean.
struct JoinCase {
    /// This bracket's own word for the lines it prints — empty for the K-quant
    /// bracket, `q8-0 ` for the 8-bit one — carried with its trailing space so that
    /// one field composes every line and the two brackets stay distinguishable.
    scheme: &'static str,
    graph: Graph,
    packed: Vec<(&'static str, Vec<u8>)>,
    dense: Vec<(&'static str, Vec<f32>)>,
    x: Vec<f32>,
    /// Why the census reads `(0, 2, 0, 0)` at this bracket's row count.
    census: &'static str,
    /// Why the counted pair is bracket 1's in the same mode.
    pair: &'static str,
    /// Why the unjoined leg's census is still this bracket's.
    unjoined_census: &'static str,
    /// What the join removed from the unjoined leg's pair.
    unjoined: &'static str,
}

/// The join set above one row, once per scheme: compile `case`'s fixture and run it
/// **at the row count it was compiled at** (no `set_active_extent`), so both packed
/// multiplies are above one row, take the row-grouped GEMM, and are joined by the
/// enclosing batch. Reads the joined run's counters and census, then its `max_abs`
/// against the CPU reference, and — unless the harness started in that mode —
/// re-measures the unjoined leg in-process and pins the two outputs bit for bit.
fn joined_above_one_row(case: JoinCase, bucket_expected: (u64, u64), step_by_step: bool) {
    let JoinCase {
        scheme,
        graph,
        packed,
        dense,
        x,
        census,
        pair,
        unjoined_census,
        unjoined,
    } = case;
    let mut exe = build(Device::Gpu, &graph, &packed, &dense);
    let _ = exe.run(&[("x", &x)]).remove(0); // warm: pool + uniforms
    reset_route_counts();
    reset_device_counts();
    let out = exe.run(&[("x", &x)]).remove(0);
    let submits = device_submit_count();
    let waits = device_wait_count();
    let routes = route_counts();
    eprintln!(
        "[rlx-gguf-join] {scheme}grouped joined run: submits={submits} waits={waits} \
         routes={routes:?}"
    );
    assert_eq!(
        (routes.per_row, routes.grouped, routes.scratch, routes.host),
        (0, 2, 0, 0),
        "{census}: {routes:?}"
    );
    assert_eq!(
        (submits, waits),
        bucket_expected,
        "{pair}; measured {submits} submission(s) and {waits} wait(s)"
    );
    let cpu = {
        let mut c = build(Device::Cpu, &graph, &packed, &dense);
        c.run(&[("x", &x)]).remove(0)
    };
    let max_abs = max_abs_diff(&out, &cpu);
    eprintln!(
        "[rlx-gguf-join] {scheme}grouped joined run vs CPU: max_abs={max_abs:.6e} \
         (|out|~{:.3e})",
        cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
    assert!(
        max_abs <= TOL,
        "{scheme}grouped joined run: max_abs {max_abs} > {TOL}"
    );

    // Pin (c) — joined vs unjoined, in this process: the mode changes how the work
    // is submitted, not the route or the arithmetic. Running the same executable
    // again under `RLX_WGPU_ONE_OP_PER_PASS=1` removes the join, so the pair gains
    // the per-step submissions back (5 and 4) while the census is still the row
    // count's, and the two outputs must agree bit for bit. A binary the harness
    // already runs in that mode has the unjoined leg as its first run above, so the
    // identity is then against that same route and there is nothing to re-measure.
    if !step_by_step {
        rlx_ir::env::set("RLX_WGPU_ONE_OP_PER_PASS", "1");
        reset_route_counts();
        reset_device_counts();
        let unjoined_out = exe.run(&[("x", &x)]).remove(0);
        let u_submits = device_submit_count();
        let u_waits = device_wait_count();
        let u_routes = route_counts();
        // Unset only what this bracket set: a harness that started in that mode
        // must keep it for the brackets after this one.
        rlx_ir::env::unset("RLX_WGPU_ONE_OP_PER_PASS");
        assert_eq!(
            (u_submits, u_waits),
            (5, 4),
            "{unjoined}; measured {u_submits} submission(s), {u_waits} wait(s)"
        );
        assert_eq!(
            (
                u_routes.per_row,
                u_routes.grouped,
                u_routes.scratch,
                u_routes.host
            ),
            (0, 2, 0, 0),
            "{unjoined_census}: {u_routes:?}"
        );
        // Element-wise and bit for bit: the two runs are the same route at the same
        // shapes, so the only difference the mode may make is which submissions
        // carry the work.
        let identity = max_abs_diff(&out, &unjoined_out);
        eprintln!(
            "[rlx-gguf-join] {scheme}joined vs unjoined identity: max_abs_diff={identity:.6e}"
        );
        common::assert_bit_identical(
            &out,
            &unjoined_out,
            &format!("joined and unjoined {scheme}outputs must be bit-identical"),
        );
    } else {
        eprintln!(
            "[rlx-gguf-join] {scheme}grouped joined run: this harness is in \
             RLX_WGPU_ONE_OP_PER_PASS=1, so the first run above IS the unjoined leg — \
             nothing left to compare the join against here"
        );
    }
}

/// One test fn on purpose — see the module doc.
#[test]
fn a_packed_multiply_shares_the_enclosing_batch_and_the_host_leg_still_reads_back() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let _route = common::route_lock();

    // A `const` so the list is one place. Mode mismatch, not a route defect: each
    // of these changes what a submission or a wait means for a forward pass, or
    // moves this graph's routing/sharding, so the integers below would be
    // measuring another mode. `RLX_WGPU_ONE_OP_PER_PASS` is the one knob this pin
    // supports — it selects the unjoined expectation. `RLX_WGPU_CONCAT_HOST`,
    // `RLX_WGPU_FORCE_HOST`, `RLX_WGPU_HOST_BUFFER_COPY`,
    // `RLX_WGPU_MAX_BIND_MB` and the 8-bit scheme's bisect lever (whose own
    // refusal leg is bracket 9) are set and unset inside their own brackets, below.
    const GUARDED: [&str; 14] = [
        "RLX_WGPU_SNAPSHOT",
        "RLX_BENCH_DISPATCH_ONLY",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_HOST_EAGER_H2D",
        "RLX_WGPU_HOST_BUFFER_COPY",
        "RLX_WGPU_CONCAT_HOST",
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_HOST_MATMUL",
        "RLX_DEBUG_NANS",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALLOW_SHARD",
        RUN_REPORT_SWITCH,
        Q8_0_BISECT,
    ];
    for key in GUARDED {
        assert!(
            !rlx_ir::env::flag(key),
            "{key} changes the submission/wait shape of a forward pass; unset it to run this pin"
        );
    }

    // The counts of a knob-forced arrangement were measured on the unified-memory leg.
    // A discrete adapter's own rule (`wgpu_prefer_structure_host` includes
    // `coop_discrete_backend()`) can rewrite the steps around the forced-host one, so
    // only the leg-independent facts are asserted there.
    let measured_leg = !rlx_wgpu::device::coop_discrete_backend();

    // ── The join set: compiled at a bucket, run at one row ──────────────────
    // The expected integers, per mode. In the default mode the joined pair reads
    // (1, 1); run the same binary with `RLX_WGPU_ONE_OP_PER_PASS=1` and the same
    // graphs must read the UNJOINED integers instead — that mode is inert by
    // design, and it re-measures the before reading from a mode switch rather
    // than from a revision. One submission per device step, so the filler flip is
    // one more submission there (and no more waits: it sits between two device
    // steps, where neither the host door nor `need_host_sync` fires).
    let step_by_step = rlx_ir::env::flag("RLX_WGPU_ONE_OP_PER_PASS");
    let (bucket_expected, host_expected) = if step_by_step {
        ((5, 4), (6, 6))
    } else {
        ((1, 1), (6, 8))
    };
    let (bucket, bucket_packed, bucket_dense, bucket_x) = interleaved(BUCKET);
    // Premise, from the schedule alone: the filler flip is device work on every
    // adapter — its host gate is `wgpu_prefer_host_fallback_agnostic`
    // (`RLX_WGPU_FORCE_HOST` / `RLX_WGPU_SHARD_GPU` / a sharded arena, no
    // adapter-kind term) — so it must lower to `reverse`, never `reverse_host`.
    let bucket_steps = WgpuExecutable::compile(bucket.clone()).step_kind_counts();
    eprintln!("[rlx-gguf-join] bucket graph steps={bucket_steps:?}");
    assert_eq!(
        (
            bucket_steps.get("reverse").copied().unwrap_or(0),
            bucket_steps.get("reverse_host").copied().unwrap_or(0),
        ),
        (1, 0),
        "the filler flip must lower to the device step, not the host one: {bucket_steps:?}"
    );
    let mut exe = build(Device::Gpu, &bucket, &bucket_packed, &bucket_dense);
    // Warm up: the first run allocates the pooled output slab and stages the
    // uniforms, neither of which is a submission or a wait.
    let _ = exe.run(&[("x", &bucket_x)]).remove(0);
    exe.set_active_extent(Some((1, BUCKET)));
    reset_route_counts();
    reset_device_counts();
    let joined = exe.run(&[("x", &bucket_x)]).remove(0);
    let submits = device_submit_count();
    let waits = device_wait_count();
    let routes = route_counts();
    eprintln!("[rlx-gguf-join] bucket run: submits={submits} waits={waits} routes={routes:?}");
    assert_eq!(
        (routes.per_row, routes.grouped, routes.scratch, routes.host),
        (2, 0, 0, 0),
        "the pin graph's two packed multiplies (Q4_K and Q6_K) must both take the \
         one-row device route and nothing else: {routes:?}"
    );
    assert_eq!(
        (submits, waits),
        bucket_expected,
        "one contiguous run of device work must make one submission and wait only \
         for its own output readback — 5 and 4 with the join inert, since the two \
         packed multiplies and the filler flip then submit on their own; measured \
         {submits} submission(s) and {waits} wait(s)"
    );

    // Parity, against the CPU reference at the row count actually run.
    let (one_row, one_packed, one_dense, x_row) = interleaved(1);
    let cpu = {
        let mut c = build(Device::Cpu, &one_row, &one_packed, &one_dense);
        c.run(&[("x", &x_row)]).remove(0)
    };
    let gpu_one = {
        let mut c = build(Device::Gpu, &one_row, &one_packed, &one_dense);
        c.run(&[("x", &x_row)]).remove(0)
    };
    let bucket_row = joined[..N].to_vec();
    for (label, got) in [
        ("bucket-run row 0", &bucket_row),
        ("one-row graph", &gpu_one),
    ] {
        let max_abs = max_abs_diff(got, &cpu);
        eprintln!(
            "[rlx-gguf-join] {label} vs CPU: max_abs={max_abs:.6e} (|out|~{:.3e})",
            cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()))
        );
        assert!(max_abs <= TOL, "{label}: max_abs {max_abs} > {TOL}");
    }
    assert!(
        gpu_one.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "wgpu output is all zeros"
    );

    // Input dependence and an in-process repeat, from a fresh compile.
    let zero = {
        let mut c = build(Device::Gpu, &one_row, &one_packed, &one_dense);
        c.run(&[("x", &vec![0.0f32; K])]).remove(0)
    };
    let zero_max = zero.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let input_dep = max_abs_diff(&gpu_one, &zero);
    assert!(
        zero_max <= TOL,
        "zero input did not produce ~zero: {zero_max}"
    );
    assert!(input_dep > TOL, "output is input-independent");
    let gpu_again = {
        let mut c = build(Device::Gpu, &one_row, &one_packed, &one_dense);
        c.run(&[("x", &x_row)]).remove(0)
    };
    assert!(
        gpu_one == gpu_again,
        "two runs of the same graph disagree by {}",
        max_abs_diff(&gpu_one, &gpu_again)
    );

    // ── The join set with the route's own N-chunking ────────────────────────
    // Bracket 1's graph with its first weight widened past the weight-window budget
    // the bracket lowers for itself (`RLX_WGPU_MAX_BIND_MB=1`), so the one-row arm
    // splits that multiply into column chunks. Everything the chunk loop adds — one
    // dispatch and one copy per chunk — lands inside the joined encoder, so the
    // counted pair must equal bracket 1's in the same mode.
    let wide = 16384;
    // Premise, from the lever and the device: a Q4_K output column costs 144 packed
    // bytes (one 256-element superblock), the lever caps the weight window at 1 MiB,
    // and this weight is 16384 of those columns — so the one-row arm has to split
    // that multiply into several column chunks on any adapter.
    let wide_dev = rlx_wgpu::wgpu_device().expect("the device gate above passed");
    let bind_cap = (1024 * 1024u64).min(wide_dev.device.limits().max_storage_buffer_binding_size);
    assert!(
        (wide as u64) * 144 > bind_cap,
        "the widened weight must exceed the capped window: {wide} columns x 144 bytes \
         vs {bind_cap} bytes"
    );
    let (wide_graph, wide_packed, wide_dense, wide_x) = interleaved_n(JoinScheme::KQuant, 1, wide);
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
    let mut wexe = build(Device::Gpu, &wide_graph, &wide_packed, &wide_dense);
    let _ = wexe.run(&[("x", &wide_x)]).remove(0); // warm: pool + uniforms
    reset_route_counts();
    reset_device_counts();
    let wide_out = wexe.run(&[("x", &wide_x)]).remove(0);
    let w_submits = device_submit_count();
    let w_waits = device_wait_count();
    let w_routes = route_counts();
    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
    eprintln!(
        "[rlx-gguf-join] N-chunked joined run: submits={w_submits} waits={w_waits} \
         routes={w_routes:?}"
    );
    assert_eq!(
        (
            w_routes.per_row,
            w_routes.grouped,
            w_routes.scratch,
            w_routes.host
        ),
        (2, 0, 0, 0),
        "the widened graph's two packed multiplies must both stay on the one-row \
         device route: {w_routes:?}"
    );
    assert_eq!(
        (w_submits, w_waits),
        bucket_expected,
        "the route's own N-chunking must add dispatches and copies, not submissions \
         or waits — the pair is the one-row pair; measured {w_submits} submission(s) \
         and {w_waits} wait(s)"
    );
    let cpu_wide = {
        let mut c = build(Device::Cpu, &wide_graph, &wide_packed, &wide_dense);
        c.run(&[("x", &wide_x)]).remove(0)
    };
    let wide_max = max_abs_diff(&wide_out, &cpu_wide);
    eprintln!(
        "[rlx-gguf-join] N-chunked joined run vs CPU: max_abs={wide_max:.6e} \
         (|out|~{:.3e})",
        cpu_wide.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
    assert!(
        wide_max <= TOL,
        "N-chunked joined run: max_abs {wide_max} > {TOL}"
    );

    // ── The join set above one row ──────────────────────────────────────────
    // Bracket 1's graph at its *compiled* row count: no `set_active_extent`, so
    // the multiplies run as the m>1 form the compile bucketed. The premise is the
    // route census below — both packed multiplies are above one row, so both are
    // the row-grouped GEMM, and the boundary joins it exactly as it joins the
    // one-row arm. The counted pair is bracket 1's: joining is an arrangement of
    // submissions, so the row count does not move it.
    let (grp_graph, grp_packed, grp_dense, grp_x) = interleaved(BUCKET);
    joined_above_one_row(
        JoinCase {
            scheme: "",
            graph: grp_graph,
            packed: grp_packed,
            dense: grp_dense,
            x: grp_x,
            census: "at the compiled row count both packed multiplies (Q4_K and Q6_K) are \
                     above one row, so both take the row-grouped GEMM and both are joined",
            pair: "the joined row-grouped multiplies ride the enclosing batch exactly as the \
                   one-row arm does, so the pair is bracket 1's in the same mode",
            unjoined_census: "the route is the row count's, not the mode's",
            unjoined: "the join removed submits each of the graph's five device steps on its \
                       own and waits four times",
        },
        bucket_expected,
        step_by_step,
    );

    // ── The 8-bit scheme's multi-row calls, joined ──────────────────────────
    // Bracket 7's shape on the scheme the grouped arm's refinement added to the
    // join set: a multi-row Q8_0 call over whole 32-element blocks of its own is
    // device-only work the boundary joins, where before that arm the scratch entry
    // point took every such call. The admission's `k` term is that block — the
    // scheme's own packing granularity, not the 256-element staged tile the arm
    // stages eight of — and `K` is a whole number of both, which is why one `K`
    // serves the K-quant brackets and this one. The census
    // cannot witness the join: it is recorded inside the encoder, and the scratch
    // entry point took this same grouped arm for these calls before the change, so
    // `grouped = 2` is what the route is and not what the arrangement is. The
    // counted pair is that witness — bracket 1's in the same mode, against the 5
    // and 4 the same executable reads with the join inert — and the identity
    // against that unjoined run is the other half.
    let (q8_graph, q8_packed, q8_dense, q8_x) = interleaved_n(JoinScheme::Q8_0, BUCKET, N);
    // The arena does not grow — read as what it is: a plan identity, not a witness of
    // the arrangement. The arena is sized by the planner alone, and the executable's is
    // that plan plus the planner's fixed tail slack; read against the K-quant graph's
    // identical term, the slack cancels and what the identity compares is the two plans.
    // It pins that the admission's slab-booking term (`dequant_gguf_scratch_bytes`, a
    // *plan* fact about the graph — the premise asserted below) is the only thing the
    // 8-bit graph adds, and that the joined and the unjoined arrangement of the same
    // calls add nothing beyond it. What it does NOT witness is the join: the plan knows
    // nothing of the arrangement an encode takes, so it holds identically for both — the
    // counted pair below and the identity against the unjoined run are what witness the
    // join.
    let q8_plan = rlx_compile::memory::plan_memory_f32_uniform(&q8_graph, 64).arena_size
        + rlx_wgpu::gguf_gpu::dequant_gguf_scratch_bytes(&q8_graph);
    let kq_arena = {
        let (kq_graph, ..) = interleaved_n(JoinScheme::KQuant, BUCKET, N);
        let kq_plan = rlx_compile::memory::plan_memory_f32_uniform(&kq_graph, 64).arena_size
            + rlx_wgpu::gguf_gpu::dequant_gguf_scratch_bytes(&kq_graph);
        WgpuExecutable::compile(kq_graph).test_arena_bytes() as i64 - kq_plan as i64
    };
    let q8_arena = WgpuExecutable::compile(q8_graph.clone()).test_arena_bytes();
    eprintln!(
        "[rlx-gguf-join] q8-0 arena: exe={q8_arena} B (plan {q8_plan} B, slack {} B) \
         against the K-quant slack {kq_arena} B",
        q8_arena as i64 - q8_plan as i64
    );
    // The premise that gives that identity its meaning: the plan really books this
    // scheme's slab, or the arena comparison is vacuous.
    assert!(
        rlx_wgpu::gguf_gpu::dequant_gguf_scratch_bytes(&q8_graph) > 0,
        "the 8-bit graph must book this scheme's slab in the plan — the admission's \
         slab is a plan fact and the join a run-time arrangement of the encode over \
         it, so neither arrangement may add an arena byte"
    );
    assert_eq!(
        q8_arena as i64 - q8_plan as i64,
        kq_arena,
        "the 8-bit graph's arena beyond its own plan — the slab it books included — \
         must be the same tail slack the K-quant graph reads: the admission and the \
         join may add no arena bytes"
    );
    // Pin (c) again, for the scheme the refinement added: the mode changes how the
    // work is submitted, not the route or the arithmetic.
    joined_above_one_row(
        JoinCase {
            scheme: "q8-0 ",
            graph: q8_graph,
            packed: q8_packed,
            dense: q8_dense,
            x: q8_x,
            census: "both Q8_0 multiplies are above one row over whole 32-element blocks of \
                     this scheme's own, so both take the row-grouped GEMM — the one route this \
                     scheme had for them before the arm too, which is why the census cannot \
                     tell the arrangements apart",
            pair: "the 8-bit scheme's joined multi-row calls ride the enclosing batch exactly \
                   as the K-quant arms do, so the pair is bracket 1's in the same mode — and \
                   this pair, not the census, is what witnesses the join",
            unjoined_census: "the route is the scheme's and the row count's, not the mode's",
            unjoined: "the join removed submits each of the graph's five device steps on its \
                       own and waits four times, the 8-bit scheme's calls submitting through \
                       the grouped entry point one at a time",
        },
        bucket_expected,
        step_by_step,
    );

    // ── The refused admitted call: the grouped encode's own lever ────────────
    // The ladder is device-free and reads no lever: `select_path`'s gates are the
    // row count, `k`, the scheme, the slab and the X window's 16-alignment, and
    // `RLX_WGPU_Q8_0_GEMM_DISABLE` is read inside the grouped *encoder* only (a
    // lowering, as its comment there says). An admitted call can therefore be
    // declined where its work is encoded, which is what this bracket pins from the
    // outside: the run still joins the call at the batch boundary, the encode
    // inside the batch refuses, the tail submits that batch carrying no work for
    // the call, and the outer host dispatch then routes it to the entry point that
    // owns its recompute — `run_dequant_matmul_gguf_gpu`, this scheme's own ladder,
    // whose grouped arm declines too and whose f32 slab is behind it.
    //
    // One executable and one arena serve both legs of the A/B: the lever is read
    // where the run routes, not at compile time, so the two sides differ only in
    // the arrangement. The concat is the host step whose output goes into the
    // deferred host mirror (bracket 4's fixture, at `BUCKET` rows because that is
    // where the 8-bit scheme's grouped answer is the one the ladder names). The
    // contract this bracket asserts is the drain — the pre-encode flush, then the
    // batch's submission and its poll — and the mirror clear the host arm's tail
    // does for this step (`DequantMatmulGguf` is not one of the cache-aware arms, so
    // that tail drops the mirror after the arm's own recompute has run). Two of the
    // three have a reading naming them: the flush is what the lever-on leg's
    // `max_abs <= TOL` against the CPU reference witnesses, since a flush that had not
    // carried the deferred write would leave the recompute below reading the slot the
    // host step never wrote, and the drain is the one poll more the lever-on leg pays,
    // `rd_waits - rj_waits == d_waits`, which is that tail's own poll. The mirror clear
    // is the one with no counter of its own, and this fixture ends on the multiply, so
    // nothing reads a mirror entry after it: it is named here rather than witnessed. A
    // refused call changes only the batch's contents: the encode inside it declines, so
    // the submission goes out carrying no work for the call — the census above is what
    // says so, `grouped = 0` here where the joined leg reads `grouped = 1` — and the arm
    // behind it recomputes on the f32 slab.
    let (rq_graph, rq_packed, rq_dense) = mirrored_x(JoinScheme::Q8_0);
    let rq_inputs: Vec<(&str, &[f32])> = rq_dense.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let cpu_rq = {
        let mut c = build(Device::Cpu, &rq_graph, &rq_packed, &rq_dense);
        c.run(&rq_inputs).remove(0)
    };
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "1");
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    let mut rqexe = WgpuExecutable::compile(rq_graph.clone());
    for (name, bytes) in &rq_packed {
        rqexe.set_param_bytes(name, bytes);
    }
    let rq_kinds = rqexe.step_kind_counts();
    eprintln!("[rlx-gguf-join] refusal graph steps={rq_kinds:?}");
    // The premise: X is produced by the host step, and by nothing else — at this
    // row count too, which is what the flush before the (declined) encode has to
    // carry.
    assert_eq!(
        (
            rq_kinds.get("concat_host").copied().unwrap_or(0),
            rq_kinds.get("concat").copied().unwrap_or(0),
        ),
        (1, 0),
        "the concat must be the host step whose deferred write the flush carries; \
         schedule = {rq_kinds:?}"
    );
    // Both sides warmed before either is counted, and the ladder's admission read
    // with the lever set: the term that could keep this call off the grouped arm is
    // not in the ladder, so the refusal below is the encoder's and not the ladder's.
    // The offset stands for this call's X window — any 16-aligned one answers the
    // same, and 4096 is the slab the plan books.
    let _ = rqexe.run(&rq_inputs);
    rlx_ir::env::set("RLX_WGPU_Q8_0_GEMM_DISABLE", "1");
    let _ = rqexe.run(&rq_inputs);
    assert_eq!(
        rlx_wgpu::gguf_gpu::select_path(
            4,
            K,
            rlx_wgpu::gguf_host::gguf_scheme_id(QuantScheme::GgufQ8_0),
            16,
            4096
        ),
        rlx_wgpu::gguf_gpu::DequantMatmulPath::GroupedDevice,
        "the 8-bit scheme's bisect lever is outside the ladder by construction, so \
         an admitted call's encode can decline while the ladder still names the \
         grouped arm"
    );
    rlx_ir::env::unset("RLX_WGPU_Q8_0_GEMM_DISABLE");
    reset_route_counts();
    reset_device_counts();
    let rq_joined = rqexe.run(&rq_inputs).remove(0);
    let rj_submits = device_submit_count();
    let rj_waits = device_wait_count();
    let rj_routes = route_counts();
    rlx_ir::env::set("RLX_WGPU_Q8_0_GEMM_DISABLE", "1");
    reset_route_counts();
    reset_device_counts();
    let rq_declined = rqexe.run(&rq_inputs).remove(0);
    let rd_submits = device_submit_count();
    let rd_waits = device_wait_count();
    let rd_routes = route_counts();
    rlx_ir::env::unset("RLX_WGPU_Q8_0_GEMM_DISABLE");
    rlx_ir::env::unset("RLX_WGPU_FORCE_HOST");
    rlx_ir::env::unset("RLX_WGPU_CONCAT_HOST");
    eprintln!(
        "[rlx-gguf-join] refusal: lever off submits={rj_submits} waits={rj_waits} \
         routes={rj_routes:?}; lever on submits={rd_submits} waits={rd_waits} \
         routes={rd_routes:?}"
    );
    assert_eq!(
        (rj_routes.grouped, rj_routes.scratch, rj_routes.host),
        (1, 0, 0),
        "with the lever off the admitted call is the joined grouped one, and its \
         census is the only one recorded: {rj_routes:?}"
    );
    assert_eq!(
        (rd_routes.grouped, rd_routes.scratch, rd_routes.host),
        (0, 1, 0),
        "with the lever on the refused call recomputes on its slab — the route this \
         shape took before the grouped arm — and nothing records the grouped one: \
         {rd_routes:?}"
    );
    for (label, got) in [("lever off", &rq_joined), ("lever on", &rq_declined)] {
        let max_abs = max_abs_diff(got, &cpu_rq);
        eprintln!(
            "[rlx-gguf-join] refusal {label} vs CPU: max_abs={max_abs:.6e} \
             (|out|~{:.3e})",
            cpu_rq.iter().fold(0.0f32, |m, v| m.max(v.abs()))
        );
        assert!(max_abs <= TOL, "refusal {label}: max_abs {max_abs} > {TOL}");
    }
    assert!(
        rq_declined.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "the declined leg's output is all zeros"
    );
    if measured_leg {
        // Reconciled by difference, as brackets 2–5 are. With the join in place the
        // refused call costs the batch submission it leaves carrying no work for it,
        // plus the slab route's own two — the `dequant_gguf` pass and the `matmul_bt`
        // that follows it — and one poll more, which is not its own: five waits is
        // what an unjoined arrangement of this fixture reads (bracket 4's K-quant
        // form reads 5 with the join inert), so the refusal hands the call back the
        // boundary poll the join had taken off it. With the mode inert neither leg is
        // joined at all, so the whole difference is the two arms' own cost: the
        // grouped arm's one submission against the slab's two — which is why the two
        // modes' pairs differ by exactly the two submissions the arrangement moves,
        // the emptied batch's and the grouped arm's own. That the emptied one carried
        // nothing for the call is the census above, read against the lever-off leg's:
        // `grouped = 0` here where the joined leg reads `grouped = 1`.
        let (d_submits, d_waits) = if step_by_step { (1, 0) } else { (3, 1) };
        assert_eq!(
            rd_submits - rj_submits,
            d_submits,
            "the refused call costs the batch submission it leaves carrying no work \
             for it plus the slab route's own two — three with the join in place, the \
             slab's extra pass alone with it inert; measured {rd_submits} submission(s) \
             lever on against {rj_submits} lever off, a difference of {}",
            rd_submits as i64 - rj_submits as i64
        );
        assert_eq!(
            rd_waits - rj_waits,
            d_waits,
            "the refused call pays one poll more than the joined one — the boundary \
             poll a call that rides no batch costs, which is also what this fixture's \
             unjoined arrangement reads — and nothing with the join inert, the two \
             arms polling alike; measured {rd_waits} wait(s) lever on against \
             {rj_waits} lever off, a difference of {}",
            rd_waits as i64 - rj_waits as i64
        );
    } else {
        eprintln!(
            "[rlx-gguf-join] refusal: submits lever off={rj_submits} lever on=\
             {rd_submits}, waits lever off={rj_waits} lever on={rd_waits} \
             (counts not asserted on this adapter leg)"
        );
    }

    // ── The host leg: a call the ladder hosts still reads back ──────────────
    let (host_graph, host_packed, host_dense) = host_adjacent();
    let mut hx = build(Device::Gpu, &host_graph, &host_packed, &host_dense);
    let inputs: Vec<(&str, &[f32])> = host_dense
        .iter()
        .filter(|(n, _)| *n == "xb" || *n == "xa" || *n == "xc")
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    let _ = hx.run(&inputs);
    reset_route_counts();
    reset_device_counts();
    let host_out = hx.run(&inputs);
    let h_submits = device_submit_count();
    let h_waits = device_wait_count();
    let h_routes = route_counts();
    eprintln!(
        "[rlx-gguf-join] host-adjacent run: submits={h_submits} waits={h_waits} \
         routes={h_routes:?}"
    );
    assert!(
        h_routes.host >= 1 && h_routes.grouped >= 1,
        "the graph must mix a hosted call with a joined multi-row one, which the \
         grouped GEMM carries: {h_routes:?}"
    );
    // Reconciled by difference, not by naming every step: the joined arrangement
    // makes the SAME submissions as the inert mode (the middle call's own encode
    // becomes the batch's) and TWO MORE waits, the polls at the joined batch's two
    // boundaries — the door that flushes a host step's deferred mirror before
    // following device work (it does not fire in the inert mode, where the pack is
    // itself a host step) and the boundary the following hosted call needs (the
    // batch's own submit, then a drain). Hence 6 and 8 here, the unjoined 6 and 6.
    assert_eq!(
        (h_submits, h_waits),
        host_expected,
        "the hosted calls still read back on their own, and the joined call \
         contributes no submission of its own — the joined arrangement reads the \
         same submissions as the inert mode and two more waits (the two boundary \
         polls); measured {h_submits} submission(s), {h_waits} wait(s)"
    );
    let cpu_host = {
        let mut c = build(Device::Cpu, &host_graph, &host_packed, &host_dense);
        c.run(&inputs)
    };
    for (i, (got, want)) in host_out.iter().zip(&cpu_host).enumerate() {
        let max_abs = max_abs_diff(got, want);
        eprintln!("[rlx-gguf-join] host-adjacent output {i} vs CPU: max_abs={max_abs:.6e}");
        assert!(max_abs <= TOL, "output {i}: max_abs {max_abs} > {TOL}");
    }

    // ── The one-row host case, through the route's direct entry point ───────
    // A graph cannot reach it: the precondition it fails is the arm's X-window
    // addressability, and the planner aligns every activation slot (`+4` below is
    // the only way in — `gguf_gemm_parked_weight.rs` drives the same window to pin
    // the refusal itself). What this bracket adds is that the counters cover that
    // leg: the shared readback helper the host route stages through submits and
    // waits per ranged D2H read.
    let (d_k, d_n) = (K, N);
    let d_bytes = packed(d_k, d_n);
    let mut dg = Graph::new("wgpu_gguf_batch_join_host_one_row");
    let dx = dg.input("x", Shape::new(&[1, d_k], DType::F32));
    let dw = dg.param("w", Shape::new(&[d_bytes.len()], DType::U8));
    let dy = dg.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![dx, dw],
        Shape::new(&[1, d_n], DType::F32),
    );
    dg.set_outputs(vec![dy]);
    let dev = rlx_wgpu::wgpu_device().expect("the device gate above passed");
    let arena = rlx_wgpu::buffer::Arena::from_plan(
        &dev.device,
        &rlx_compile::memory::plan_memory_f32_uniform(&dg, 16),
    );
    let x_bytes: Vec<u8> = feed(d_k, 0.023)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (x_off, w_off, y_off) = (arena.offset(dx), arena.offset(dw), arena.offset(dy));
    assert!(
        x_off + 4 + x_bytes.len() <= arena.buffer.size() as usize,
        "X's slot needs four spare bytes to shift its window"
    );
    arena.write_bytes_range(&dev.queue, x_off + 4, &x_bytes);
    arena.write_bytes_range(&dev.queue, w_off, &d_bytes);
    reset_route_counts();
    reset_device_counts();
    // The counters below are this bracket's claim, not the ladder's answer.
    rlx_wgpu::gguf_gpu::run_dequant_matmul_gguf_gemv_rows(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        d_k,
        d_n,
        rlx_wgpu::gguf_host::gguf_scheme_id(QuantScheme::GgufQ4K),
        x_off + 4,
        w_off,
        y_off,
        &mut rlx_wgpu::backend::PassMarks::none(),
    );
    let d_submits = device_submit_count();
    let d_waits = device_wait_count();
    let d_routes = route_counts();
    eprintln!(
        "[rlx-gguf-join] direct one-row host: submits={d_submits} waits={d_waits} \
         routes={d_routes:?}"
    );
    assert_eq!(
        (d_routes.host, d_routes.per_row),
        (1, 0),
        "a misaligned one-row window must still be refused by the device arm: {d_routes:?}"
    );
    assert_eq!(
        (d_submits, d_waits),
        (2, 2),
        "the host route's two ranged D2H reads must each submit and wait: measured \
         {d_submits} submission(s), {d_waits} wait(s)"
    );

    // ── The host mirror flushed before joined device work ───────────────────
    // The concat below is the host-classified step whose output goes into the
    // deferred host mirror instead of the arena (its arm stages through the host
    // tensor cache), and the packed multiply that reads it is device-only work:
    // without the flush before that batch the multiply reads the arena slot the
    // concat never wrote. `RLX_WGPU_CONCAT_HOST=1` takes the concat off the direct
    // window path, but on this unified-memory adapter it is then still staged
    // through GPU scratch (`buffer_copy` + `concat` steps, no host step at all, so
    // parity would hold vacuously), which is why the bracket also sets
    // `RLX_WGPU_FORCE_HOST=1` — the crate's own lever for host structural steps on
    // any adapter, the one `structure_host_operand_fit.rs` drives `ConcatHost`
    // with.
    let (mir_graph, mir_packed, mir_dense) = mirrored_x(JoinScheme::KQuant);
    let mir_inputs: Vec<(&str, &[f32])> =
        mir_dense.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let cpu_mir = {
        let mut c = build(Device::Cpu, &mir_graph, &mir_packed, &mir_dense);
        c.run(&mir_inputs).remove(0)
    };
    rlx_ir::env::set("RLX_WGPU_CONCAT_HOST", "1");
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    let mut mirex = WgpuExecutable::compile(mir_graph.clone());
    for (name, bytes) in &mir_packed {
        mirex.set_param_bytes(name, bytes);
    }
    let mir_kinds = mirex.step_kind_counts();
    eprintln!("[rlx-gguf-join] mirror-flush graph steps={mir_kinds:?}");
    // The premise: X is produced by the host step, and by nothing else.
    assert_eq!(
        (
            mir_kinds.get("concat_host").copied().unwrap_or(0),
            mir_kinds.get("concat").copied().unwrap_or(0),
        ),
        (1, 0),
        "the concat must be the host step whose deferred write the flush carries; \
         schedule = {mir_kinds:?}"
    );
    // Warm up (arena + uniforms), then bracket one forward pass.
    let _ = mirex.run(&mir_inputs);
    reset_route_counts();
    reset_device_counts();
    let mir_out = mirex.run(&mir_inputs).remove(0);
    let m_submits = device_submit_count();
    let m_waits = device_wait_count();
    let m_routes = route_counts();
    rlx_ir::env::unset("RLX_WGPU_FORCE_HOST");
    rlx_ir::env::unset("RLX_WGPU_CONCAT_HOST");
    eprintln!(
        "[rlx-gguf-join] mirror-flush run: submits={m_submits} waits={m_waits} \
         routes={m_routes:?}"
    );
    // The load-bearing reading: a missing flush before the joined batch leaves the
    // multiply reading the arena slot the host step never wrote.
    let mir_abs = max_abs_diff(&mir_out, &cpu_mir);
    eprintln!("[rlx-gguf-join] mirror-flush output vs CPU: max_abs={mir_abs:.6e}");
    assert!(
        mir_abs <= TOL,
        "mirror-flush output: max_abs {mir_abs} > {TOL} — the host step's deferred \
         write did not reach the device before the joined multiply read it"
    );
    assert!(
        mir_out.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "mirror-flush output is all zeros"
    );
    assert!(
        m_routes.per_row >= 1 && m_routes.host == 0,
        "the packed multiply must still take the join-set device route and never \
         read its operand back on the host: {m_routes:?}"
    );
    if measured_leg {
        assert_eq!(
            (m_submits, m_waits),
            if step_by_step { (4, 5) } else { (3, 4) },
            "the host step's readback, its flush before the joined batch, and that \
             batch's own submission and readback: 3 and 4 with the join in place, 4 \
             and 5 with it inert, the multiply then reaching the mirror through the \
             host step's own entry flush; measured {m_submits} submission(s), {m_waits} \
             wait(s)"
        );
    } else {
        eprintln!(
            "[rlx-gguf-join] mirror-flush: submits={m_submits} waits={m_waits} \
             (counts not asserted on this adapter leg)"
        );
    }

    // ── The join block's own flush: a host BufferCopy feeding the join ───────
    // The bracket above reaches the join with a host step that falls through the
    // shared post-host-step door. The host `BufferCopy` arm `continue`s past that
    // door, so the join block's own flush is the only one that can carry its
    // deferred write to the device — and that arm is what this bracket exercises.
    // Without the flush the joined multiply reads the arena slot the copy never
    // wrote.
    let (hc_graph, hc_packed, hc_dense) = host_copied_x();
    let hc_inputs: Vec<(&str, &[f32])> = hc_dense.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let cpu_hc = {
        let mut c = build(Device::Cpu, &hc_graph, &hc_packed, &hc_dense);
        c.run(&hc_inputs).remove(0)
    };
    let mut hcexe = WgpuExecutable::compile(hc_graph.clone());
    for (name, bytes) in &hc_packed {
        hcexe.set_param_bytes(name, bytes);
    }
    let hc_kinds = hcexe.step_kind_counts();
    eprintln!("[rlx-gguf-join] host-copy graph steps={hc_kinds:?}");
    // The premise: X is produced by the copy step, and by nothing else.
    assert_eq!(
        (
            hc_kinds.get("buffer_copy").copied().unwrap_or(0),
            hc_kinds.get("dequant_matmul_gguf").copied().unwrap_or(0),
        ),
        (2, 1),
        "the two value-preserving casts must lower to two non-zero BufferCopies, \
         the second feeding the packed multiply: schedule = {hc_kinds:?}"
    );
    // Scoped to the run: the knob is read where the run path routes a copy (the
    // copy's own arm and the batch boundary), so compiling under it is not needed
    // — but leaving it set would re-route every later bracket's copies.
    rlx_ir::env::set("RLX_WGPU_HOST_BUFFER_COPY", "1");
    // Warm up (arena + uniforms), then bracket one forward pass.
    let _ = hcexe.run(&hc_inputs);
    reset_route_counts();
    reset_device_counts();
    let hc_out = hcexe.run(&hc_inputs).remove(0);
    let c_submits = device_submit_count();
    let c_waits = device_wait_count();
    let c_routes = route_counts();
    rlx_ir::env::unset("RLX_WGPU_HOST_BUFFER_COPY");
    eprintln!(
        "[rlx-gguf-join] host-copy run: submits={c_submits} waits={c_waits} \
         routes={c_routes:?}"
    );
    let hc_abs = max_abs_diff(&hc_out, &cpu_hc);
    eprintln!("[rlx-gguf-join] host-copy output vs CPU: max_abs={hc_abs:.6e}");
    assert!(
        hc_abs <= TOL,
        "host-copy output: max_abs {hc_abs} > {TOL} — the host copy's deferred \
         write did not reach the device before the joined multiply read it"
    );
    assert!(
        c_routes.per_row >= 1 && c_routes.host == 0,
        "the packed multiply must still take the join-set device route and never \
         read its operand back on the host: {c_routes:?}"
    );
    if measured_leg {
        assert_eq!(
            (c_submits, c_waits),
            if step_by_step { (3, 4) } else { (2, 2) },
            "the host copy's readback and the joined batch's own submission and readback: \
             2 and 2 with the join in place, 3 and 4 with it inert. The join block's flush of \
             the copy's deferred write rides the batch — the queue write is ordered before that \
             batch's submission — so it costs no poll of its own, where the multiply's pre-step \
             door (the inert arm's route) still drains the device before it reads X; measured \
             {c_submits} submission(s), {c_waits} wait(s)"
        );
    } else {
        eprintln!(
            "[rlx-gguf-join] host-copy: submits={c_submits} waits={c_waits} \
             (counts not asserted on this adapter leg)"
        );
    }
}

/// A packed multiply's one-row arm survives the row axis being rebound. The setup
/// cache that arm dispatches from is keyed by the record, the arm and the three
/// arena windows — no buffer identity — so an entry left over from an earlier
/// compile is served against that compile's arena (keeping it alive), and the
/// live arena's output slot is never written. Three runs through one executable at
/// `1, 4, 1` rows: the third binding is the first run's, so its windows are the
/// first run's and a surviving entry is hit. Each run is compared against the same
/// graph on `Device::Cpu`, whose deferred executable performs the same rebinds.
///
/// The same three runs pin the deferred param replay: while a graph is unresolved
/// `set_param_bytes` only buffers, so a replay left buffered would leave every
/// fresh arena's weight slot unwritten.
#[test]
fn a_packed_multiply_survives_rebinding_the_dynamic_row_axis() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // The route counters are process-global, and the counter pin above brackets
    // them; one of the two tests in this binary at a time.
    let _route = common::route_lock();
    const K: usize = 1024;
    const N: usize = 1024;
    let (graph, w_bytes) = dynamic_row_packed_multiply(K, N);
    let packed = vec![("w", w_bytes)];
    let mut cpu = build(Device::Cpu, &graph, &packed, &[]);
    let mut exe = WgpuExecutable::compile(graph);
    exe.set_param_bytes("w", &packed[0].1);
    // The third row differs from the first while its binding does not.
    let cases: [(usize, Vec<f32>, &str); 3] = [
        (1, feed(K, 0.023), "A"),
        (4, feed(4 * K, 0.017), "B"),
        (1, feed(K, 0.031), "C"),
    ];
    for (i, (m, row, label)) in cases.iter().enumerate() {
        if i == 0 {
            reset_route_counts();
        }
        let got = exe.run(&[("x", row)]).remove(0);
        if i == 0 {
            // The premise: the first run is the tiled one-row arm, not a host read
            // and not an arm that has no setup cache to survive.
            let routes = route_counts();
            assert_eq!(
                (routes.per_row, routes.host),
                (1, 0),
                "the one-row multiply must take the tiled arm's per-row route: {routes:?}"
            );
        }
        let want = cpu.run(&[("x", row)]).remove(0);
        let abs = max_abs_diff(&got, &want);
        eprintln!("[rlx-gguf-join] rebind run {label} (m={m}): max_abs={abs:.6e}");
        assert!(
            abs <= TOL,
            "rebind run {label} at {m} row(s): max_abs {abs} > {TOL} — the packed \
             multiply did not read the arena this rebind compiled"
        );
        assert!(
            got.iter().fold(0.0f32, |v, e| v.max(e.abs())) > 0.0,
            "rebind run {label} at {m} row(s) is all zeros — its output slot was \
             never written"
        );
    }

    // A weight written at the binding the executable already holds: no rebind runs,
    // so the write is the buffered kind — the one a deferred executable only lands
    // through the replay. The last run's values are the reference it must leave.
    let (last_m, last_row, _) = &cases[2];
    let before = exe.run(&[("x", last_row)]).remove(0);
    let mut w2 = packed[0].1.clone();
    for b in w2.iter_mut().skip(2).step_by(34) {
        *b = b.wrapping_add(1);
    }
    exe.set_param_bytes("w", &w2);
    let after = exe.run(&[("x", last_row)]).remove(0);
    let (ref_graph, _) = dynamic_row_packed_multiply(K, N);
    let mut cpu2 = build(Device::Cpu, &ref_graph, &[("w", w2)], &[]);
    let want2 = cpu2.run(&[("x", last_row)]).remove(0);
    let moved = max_abs_diff(&before, &after);
    let abs = max_abs_diff(&after, &want2);
    eprintln!(
        "[rlx-gguf-join] same-binding weight write at m={last_m}: moved={moved:.6e} \
         max_abs={abs:.6e}"
    );
    assert!(
        moved > 1e-3,
        "a weight written at an unchanged binding left the output bit-identical \
         (moved {moved:.6e}) — the buffered param never reached the arena"
    );
    assert!(
        abs <= TOL,
        "same-binding weight write at m={last_m}: max_abs {abs} > {TOL} — the run \
         used the weight the executable was compiled with, not the one written since"
    );
}
