// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed GGUF `Op::DequantMatMul` parity — wgpu vs CPU, prefill and decode.
//!
//! Per cell: counted finiteness (nan / pos_inf / neg_inf kept as three numbers and
//! asserted clean before anything is folded into `max_abs` — see [`Finiteness`]),
//! parity against the CPU reference, zero-input ~ zero, input dependence, and an
//! in-process repeat.
//!
//! wgpu's change point is `m == 1` vs `m > 1` on the scratch-free K-quant
//! routes: Q4_K / Q6_K decode with one fused per-row GEMV and prefill with a
//! row-grouped GEMM (8 rows/tile, one weight read per tile). Q5_K / Q8_0 have NO
//! fused route — they always dequantize into an in-arena f32 scratch and run
//! `matmul_bt`, so there is no change point for them to measure.
//!
//! wgpu DOES expose an authoritative route signal —
//! `rlx_wgpu::gguf_gpu::route_counts()` — but it is process-global, so the
//! authoritative assertion lives in `gguf_matmul_route.rs` (its own single-test
//! binary). This file reports a route DERIVED from the runtime selector instead
//! ([`wgpu_route`]).
//!
//! # Running
//!
//! Cells are behind the existing `backend-tests` opt-in:
//! `cargo test --release -p rlx-wgpu --features backend-tests --test
//! gguf_dequant_matmul_prefill_parity -- --nocapture` (reached by `just test-gpu`
//! on Darwin and by the Linux branch of that recipe). `-- --nocapture` is what
//! makes the per-cell `[rlx-gguf-*]` evidence visible; without it the run prints
//! none of it. Each cell's CPU reference is `rlx-cpu`'s
//! `gguf_matmul_bt_dispatch_at` behind an ordinary `Device::Cpu` `Session`; the
//! engine exposes NO signal naming which `rlx-cpu` route served a comparison, so
//! — rather than invent one — the reference is taken in the same process under
//! identical conditions, with no process-global knob pinned for it (this file
//! pins no knob at all). Each cell's `repeat_max_abs` re-runs that reference from
//! a fresh `Session` and compares it element for element, so a reference-route
//! change inside a cell would surface there rather than pass unnoticed.
//! Cross-process reproducibility is 3 serial invocations of that same command
//! with `--test-threads=1`, which a single test cannot observe.
//!
//! This suite and `rlx-metal`'s `metal_gguf_dequant_matmul_prefill_parity.rs` are
//! deliberate mirrors of each other: an edit to this harness must be applied to
//! both. They differ in backend-specific values — each file's tolerance
//! table (`1e-2` here), the rows its grid needs (this one drops the `m=96`
//! k=256/n=640 row Metal needs for its fused-GEMM cap, since the cases above
//! already run that shape and wgpu has no boundary above `m == 1`, and adds an
//! unaligned-k row only wgpu has a host route for), and the route strings, adapter
//! limits and memory figures, which are this backend's own. They also differ in
//! one structural thing, deliberately: that file serializes its device tests
//! under a lock, this one takes none, so the tests here run in parallel as they
//! always have.
//!
//! Two things make that difference safe rather than drift: this file pins no knob
//! (see above), and wgpu's process-global state is thread-safe by construction
//! (pipeline `OnceLock`s, the pooled out-slab behind its `Mutex`, the `Device`/
//! `Queue` a `Session` shares). Do not add a lock here to match the Metal file.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// One parity cell: a packed weight `[n, k]` applied to an input of
/// `batch ++ [m, k]`.
///
/// A struct rather than a long argument list keeps `run_case` inside the clippy
/// argument threshold and lets the grid and the existing prefill cases share one
/// signature.
#[derive(Clone)]
struct Cell {
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    /// Leading dims before `[m, k]`; `&[]` for a plain 2-D `[m, k]` cell and
    /// `&[1]` for the `[1, 1, k]` single-token decode shape real models use.
    batch: &'static [usize],
    m: usize,
    k: usize,
    n: usize,
    tol: f32,
    label: String,
}

/// Uploaded parameter: packed GGUF bytes go through the typed U8 path, the f32
/// norm gain/bias through the plain path.
enum Param {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

fn set_params(c: &mut CompiledGraph, params: &[(String, Param)]) {
    for (name, p) in params {
        match p {
            Param::U8(b) => c.set_param_typed(name, b, DType::U8),
            Param::F32(v) => c.set_param(name, v),
        }
    }
}

/// Finiteness census of one `[f32]` output: NaN, +inf and −inf counted as three
/// separate numbers, so an infinity cannot hide behind a NaN (or behind
/// `f32::max`, which returns the non-NaN operand). Cells assert this clean BEFORE
/// folding anything into a max-abs metric — an all-NaN output would otherwise
/// fold to `0.0` and pass.
///
/// This binary keeps its own census because it also *reports* the counts;
/// `tests/common/mod.rs` asserts the same rule before folding in `max_abs_diff`.
/// Keep the two in step.
#[derive(Clone, Copy, Debug)]
struct Finiteness {
    nan: usize,
    pos_inf: usize,
    neg_inf: usize,
}

impl Finiteness {
    fn count(v: &[f32]) -> Self {
        Self {
            nan: v.iter().filter(|x| x.is_nan()).count(),
            pos_inf: v.iter().filter(|x| **x == f32::INFINITY).count(),
            neg_inf: v.iter().filter(|x| **x == f32::NEG_INFINITY).count(),
        }
    }

    fn is_clean(self) -> bool {
        self.nan + self.pos_inf + self.neg_inf == 0
    }
}

/// Everything one packed-graph run reports. The non-finite counts are kept as
/// separate numbers so a report can never be made to pass by dropping one.
struct Measured {
    gpu: Finiteness,
    cpu: Finiteness,
    /// Finiteness of the wgpu output under a zero input — a non-zero count is a
    /// failure dressed up as "input dependence".
    zero: Finiteness,
    max_abs: f32,
    cpu_max: f32,
    gpu_max: f32,
    zero_max: f32,
    input_dep: f32,
    repeat_max_abs: f32,
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Negative control for the counted finiteness every cell asserts on: the census
/// must actually reject non-finite output, which no device run can demonstrate
/// without producing one. Device-free, so it holds even where no wgpu adapter
/// exists. Both `f32::max` traps are the point — an all-NaN output folds to `0.0`
/// and a NaN folded against a finite value disappears — so a cell that folded
/// without counting would still report clean.
#[test]
fn finiteness_census_counts_nan_and_both_infinities() {
    let v = [
        1.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        f32::NAN,
    ];
    let c = Finiteness::count(&v);
    assert_eq!((c.nan, c.pos_inf, c.neg_inf), (2, 1, 1), "{c:?}");
    assert!(
        !c.is_clean(),
        "a non-finite value must fail is_clean: {c:?}"
    );
    assert!(Finiteness::count(&[0.0, 1.0, -2.0]).is_clean());
    assert_eq!(
        max_abs(&[f32::NAN; 4]),
        0.0,
        "an all-NaN output folds to this, which is why the census precedes any fold"
    );
    assert_eq!(max_abs(&[f32::NAN, 1.0]), 1.0, "a NaN folded away entirely");
}

/// Run one graph on wgpu and on the CPU reference and gather every number the
/// cells assert on. `label` only shapes failure messages.
///
/// Real and zero input share ONE compiled graph, so `input_dep` measures the
/// graph's input dependence rather than two compiles' disagreement; the
/// in-process repeat check recompiles both backends from scratch.
fn measure(
    label: &str,
    g: &Graph,
    params: &[(String, Param)],
    x: &[f32],
    out_elems: usize,
) -> Measured {
    let zero = vec![0.0f32; x.len()];
    let mut cg = Session::new(Device::Gpu).compile(g.clone());
    set_params(&mut cg, params);
    let gpu = cg.run(&[("x", x)]).remove(0);
    let gpu_zero = cg.run(&[("x", zero.as_slice())]).remove(0);

    let mut cc = Session::new(Device::Cpu).compile(g.clone());
    set_params(&mut cc, params);
    let cpu = cc.run(&[("x", x)]).remove(0);

    // Fresh `Session`s, same graph and input: in-process reproducibility for
    // both backends. Reusing `cg`/`cc` would only re-measure the same objects.
    let mut cg2 = Session::new(Device::Gpu).compile(g.clone());
    set_params(&mut cg2, params);
    let gpu2 = cg2.run(&[("x", x)]).remove(0);
    let mut cc2 = Session::new(Device::Cpu).compile(g.clone());
    set_params(&mut cc2, params);
    let cpu2 = cc2.run(&[("x", x)]).remove(0);

    for (what, v) in [
        ("gpu", &gpu),
        ("gpu_zero", &gpu_zero),
        ("cpu", &cpu),
        ("gpu2", &gpu2),
        ("cpu2", &cpu2),
    ] {
        assert_eq!(v.len(), out_elems, "{label}: {what} output length");
    }

    // Counted before any fold — see `Finiteness`.
    let gpu_fin = Finiteness::count(&gpu);
    let cpu_fin = Finiteness::count(&cpu);
    let zero_fin = Finiteness::count(&gpu_zero);
    let gpu2_fin = Finiteness::count(&gpu2);
    let cpu2_fin = Finiteness::count(&cpu2);
    assert!(
        gpu_fin.is_clean()
            && cpu_fin.is_clean()
            && zero_fin.is_clean()
            && gpu2_fin.is_clean()
            && cpu2_fin.is_clean(),
        "{label}: non-finite output — gpu={gpu_fin:?} cpu={cpu_fin:?} zero={zero_fin:?} \
         gpu2={gpu2_fin:?} cpu2={cpu2_fin:?}"
    );

    let measured = Measured {
        gpu: gpu_fin,
        cpu: cpu_fin,
        zero: zero_fin,
        cpu_max: max_abs(&cpu),
        gpu_max: max_abs(&gpu),
        zero_max: max_abs(&gpu_zero),
        max_abs: max_diff(&cpu, &gpu),
        input_dep: max_diff(&gpu, &gpu_zero),
        repeat_max_abs: max_diff(&gpu, &gpu2).max(max_diff(&cpu, &cpu2)),
    };
    assert!(
        measured.cpu_max.is_finite() && measured.cpu_max > 0.0,
        "{label}: CPU reference is zero or non-finite (cpu_max={})",
        measured.cpu_max
    );
    measured
}

/// The per-cell assertions, all relative to the cell's own tolerance.
///
/// `max_abs <= tol` rather than `<` keeps the two sides of a bit-exact route
/// from failing on a representation-level tie; the non-finite values that would
/// make that comparison meaningless were already counted away in `measure`.
fn assert_measured(label: &str, r: &Measured, tol: f32) {
    assert!(r.gpu_max > 0.0, "{label}: wgpu output is all zeros");
    assert!(
        r.zero_max <= tol,
        "{label}: zero input did not produce ~zero output (zero_max={})",
        r.zero_max
    );
    assert!(
        r.max_abs <= tol,
        "{label}: wgpu vs CPU max_abs={} (tol {tol})",
        r.max_abs
    );
    assert!(
        r.input_dep > tol,
        "{label}: output is input-independent (zero-input delta={})",
        r.input_dep
    );
    assert!(
        r.repeat_max_abs <= tol,
        "{label}: in-process re-run disagrees by {}",
        r.repeat_max_abs
    );
}

/// Whether wgpu has a scratch-free fused route for `scheme` — derived from the
/// backend's own `gemv_supports_scheme` (exactly Q4_K=0, Q6_K=2, Q1_0=24) rather
/// than restated here, so a change there cannot go unnoticed by this report.
fn wgpu_has_fused_route(scheme: QuantScheme) -> bool {
    scheme
        .gpu_dequant_scheme_id()
        .is_some_and(rlx_wgpu::gguf_gpu::gemv_supports_scheme)
}

/// The fused decode route: one GEMV dispatch per output row, no f32 scratch.
const ROUTE_PER_ROW: &str = "per-row GEMV (fused, no scratch)";
/// The fused prefill route: one pass over each packed weight tile per 8 output
/// rows, no f32 scratch. Only Q4_K / Q6_K have it, and only at `m > 1`.
const ROUTE_GROUPED: &str =
    "row-grouped fused GEMM (8 rows/tile, one weight read per tile, no scratch)";
/// The only route Q5_K / Q8_0 have, at every `m`: device dequant into the
/// in-arena f32 scratch, then `matmul_bt`. The lowerer substitutes a whole-op host
/// D2H+H2D staging when no slab was booked (sharded arena, parked weight); that is
/// not distinguishable from here, and no cell in this file builds it.
const ROUTE_SCRATCH: &str = "dequant_scratch + matmul_bt (in-arena f32 scratch k*n*4)";
/// The fused GEMV/GEMM kernels read whole GGUF blocks, so at `m > 1` an unaligned
/// `k` on a scratch-free family takes no fused route: the multiply runs on the
/// host (where the graph booked no f32 slab for it), or through the slab.
const ROUTE_HOST: &str = "host dequant (k is not a whole number of GGUF blocks, which \
     the fused GEMV/GEMM reads, so at m > 1 no fused route applies)";

/// The route the wgpu encoder takes for a packed `DequantMatMul`, derived from
/// the runtime selector: `use_gemv` in `rlx-wgpu/src/backend/run.rs` and
/// `gguf_gpu::encode_dequant_matmul_gguf_into`'s Q1_0 → grouped K-quant →
/// per-row ladder.
///
/// There is deliberately no in-process probe here: wgpu's own signal
/// (`gguf_gpu::route_counts`) is process-global, so the authoritative assertion
/// lives in `gguf_matmul_route.rs`'s single-test binary. This helper mirrors the
/// same ladder so each cell can print which route it *should* have taken.
///
/// The selector is `use_gemv = m == 1 || scheme == Q1_0` FIRST and the `k` gate
/// only after it, so `m == 1` takes the fused path at any `k` — the `k % block`
/// requirement belongs to the `m > 1` arm alone, and printing it on the decode
/// cells would claim a gate the encoder never consults.
///
/// `arena_scratch_reserved` is the caller's statement of the graph-level fact the
/// lowerer tests as `dequant_scratch_off != 0`, taken from the graph's own
/// reservation (`gguf_gpu::dequant_gguf_scratch_bytes`) rather than decided here —
/// it is what separates the host fallback from the slab on the unaligned-`k` arm.
fn wgpu_route(scheme: QuantScheme, m: usize, k: usize, arena_scratch_reserved: bool) -> String {
    // Block geometry, so each printed route states the blocks the kernel reads.
    let block_elems = scheme.gguf_block_size() as usize;
    let blocks = format!(
        "{}-elem/{}-byte blocks",
        scheme.gguf_block_size(),
        scheme.gguf_block_bytes()
    );
    if !wgpu_has_fused_route(scheme) {
        // Q5_K and Q8_0 have NO fused route: one scratch route at every m, so
        // there is no change point between decode and prefill to measure.
        return match scheme {
            QuantScheme::GgufQ5K => {
                // The only scratch-route family here whose k this file can leave off
                // a block boundary (Q8_0's is always whole), so the note names the
                // geometry only where a cell has one — no cell reads a neighbour's.
                let partial = if k.is_multiple_of(block_elems) {
                    String::new()
                } else {
                    format!(" (verified by this cell: k={k} is not a whole number of blocks)")
                };
                format!(
                    "{ROUTE_SCRATCH}; {blocks}; no k gate — the scratch kernel has no block-multiple \
                     requirement{partial}; ONE route at every m"
                )
            }
            QuantScheme::GgufQ8_0 => format!(
                "{ROUTE_SCRATCH}; {blocks}; every k this file feeds is a whole number of 32-element \
                 blocks; ONE route at every m"
            ),
            _ => ROUTE_SCRATCH.to_string(),
        };
    }
    let aligned = k.is_multiple_of(block_elems);
    match scheme {
        // Q4_K / Q6_K: the change point is m == 1 (decode GEMV) vs m > 1
        // (row-grouped GEMM) — the latter only when k is a whole number of
        // 256-element blocks.
        QuantScheme::GgufQ4K | QuantScheme::GgufQ6K if m == 1 => {
            format!("{ROUTE_PER_ROW}; {blocks}; the m==1 arm consults no k")
        }
        QuantScheme::GgufQ4K | QuantScheme::GgufQ6K if aligned => {
            format!("{ROUTE_GROUPED}; {blocks}; k%256==0")
        }
        // Unaligned k with a slab booked elsewhere in the graph: the op dequantizes
        // into it instead of crossing to the host (the lowerer's
        // `dequant_scratch_off != 0` branch). No cell in this file builds that graph.
        QuantScheme::GgufQ4K | QuantScheme::GgufQ6K if arena_scratch_reserved => {
            format!("{ROUTE_SCRATCH}; {blocks}; unaligned k, so no fused GEMM")
        }
        QuantScheme::GgufQ4K | QuantScheme::GgufQ6K => format!("{ROUTE_HOST}; {blocks}"),
        // Q1_0 is the selector's other short-circuit (`scheme_id == 24`), so it
        // takes a fused path at every m and no k. Only the `m > 1` form is
        // modelled: no Q1_0 cell in this file is a decode cell.
        QuantScheme::GgufQ1_0 => {
            format!("tiled Q1_0 GEMM (one dispatch, no scratch); {blocks}; no k gate")
        }
        _ => ROUTE_SCRATCH.to_string(),
    }
}

/// The adapter limits the graph-wide dequant scratch is sized against: wgpu
/// reserves the scratch only when `align16(arena) + align16(scratch)` fits
/// `max_buffer_size`, and the scratch route binds the whole arena, so it also
/// needs the arena inside `max_storage_buffer_binding_size`. Read from the
/// device `rlx-wgpu` already holds.
fn adapter_limits() -> String {
    match rlx_wgpu::device::wgpu_device() {
        Some(d) => {
            let l = d.device.limits();
            format!(
                "adapter_max_buffer={} adapter_max_storage_binding={}",
                l.max_buffer_size, l.max_storage_buffer_binding_size
            )
        }
        None => "adapter_max_buffer=? (no wgpu device)".to_string(),
    }
}

/// The graph-wide in-arena dequant scratch against the device's buffer limit,
/// formatted for the cell and memory lines. Uses the same device accessor as
/// [`adapter_limits`], so the two cannot disagree.
fn scratch_vs_adapter(scratch: usize) -> String {
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    match rlx_wgpu::device::wgpu_device() {
        Some(d) => {
            let limit = d.device.limits().max_buffer_size;
            format!(
                "{:.2} MiB / {:.2} MiB max_buffer_size ({:.4}%)",
                mib(scratch),
                limit as f64 / (1024.0 * 1024.0),
                100.0 * scratch as f64 / limit as f64
            )
        }
        None => "no wgpu device".to_string(),
    }
}

/// One cell's report — the fields one cell's evidence has to carry.
struct CaseReport {
    scheme: String,
    m: usize,
    /// The projection shape(s) this cell's graph applies, `[k=…, n=…]` per op —
    /// the grid cells are one matmul, the model cell four different ones.
    weight_shape: String,
    label: String,
    route: String,
    /// The absolute bound this cell's parity, zero-input, input-dependence and
    /// repeat assertions use, printed so a run's evidence carries the tolerance
    /// it was judged by instead of leaving the reader to find the shape table.
    tol: f32,
    /// What this graph lets the backend fuse, and when — the grid cells are one
    /// bare matmul; the model-class graph is a multi-step block.
    graph_context: String,
    packed_bytes: usize,
    f32_image_bytes: usize,
    scratch_graph_max: usize,
    measured: Measured,
}

/// The grid cells' graph: one packed matmul and nothing around it.
const BARE_GRAPH_CONTEXT: &str = "bare single packed matmul (no norm/activation to fuse)";
/// The model-class graph's graph: what wgpu does with a real block.
const MODEL_GRAPH_CONTEXT: &str = "packed matmuls stay their own steps (never common-IR); \
     elementwise/mul ops fuse into the surrounding segments; the eight Q5_K/Q8_0 nodes share one \
     in-arena dequant scratch, which the Q4_K/Q6_K fused routes never materialize";

impl CaseReport {
    fn print(&self) {
        let r = &self.measured;
        eprintln!(
            "[rlx-gguf-cell] backend=wgpu scheme={} m={} weight_shape={} label=\"{}\" route=\"{}\" \
             packed={} f32_image={} scratch_graph_max={} scratch_vs_adapter={} \
             gpu_nan={} gpu_pos_inf={} gpu_neg_inf={} \
             cpu_nan={} cpu_pos_inf={} cpu_neg_inf={} \
             zero_nan={} zero_pos_inf={} zero_neg_inf={} \
             max_abs={:.6e} rel_err={:.3e} tol={:.3e} cpu_max={:.6} gpu_max={:.6} \
             zero_max={:.3e} input_dep={:.6e} repeat_max_abs={:.3e} \
             graph_context=\"{}\" cpu_ref_route=unreported_no_engine_signal {}",
            self.scheme,
            self.m,
            self.weight_shape,
            self.label,
            self.route,
            self.packed_bytes,
            self.f32_image_bytes,
            self.scratch_graph_max,
            scratch_vs_adapter(self.scratch_graph_max),
            r.gpu.nan,
            r.gpu.pos_inf,
            r.gpu.neg_inf,
            r.cpu.nan,
            r.cpu.pos_inf,
            r.cpu.neg_inf,
            r.zero.nan,
            r.zero.pos_inf,
            r.zero.neg_inf,
            r.max_abs,
            r.max_abs / r.cpu_max.max(1e-30),
            self.tol,
            r.cpu_max,
            r.gpu_max,
            r.zero_max,
            r.input_dep,
            r.repeat_max_abs,
            self.graph_context,
            adapter_limits(),
        );
    }
}

/// Build `x @ dequant(w)[n, k]^T` for a packed GGUF weight and compare wgpu
/// against the CPU reference over identical packed bytes.
///
/// One device gate per test: every caller has already passed `skip_unless_device`,
/// so this returns the cell report and asserts every invariant the cell reports on.
fn run_case(cell: &Cell) -> CaseReport {
    let Cell {
        scheme,
        ggml,
        batch,
        m,
        k,
        n,
        tol,
        label,
    } = cell.clone();

    // Weight is row-major [n, k]: output column c owns k contiguous values.
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut in_shape = batch.to_vec();
    in_shape.extend_from_slice(&[m, k]);
    let mut out_shape = batch.to_vec();
    out_shape.extend_from_slice(&[m, n]);

    let mut g = Graph::new("wgpu_gguf_dq_prefill");
    let x_in = g.input("x", Shape::new(&in_shape, DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&out_shape, DType::F32),
    );
    g.set_outputs(vec![y]);

    let packed_bytes = packed.len();
    let params = vec![("w".to_string(), Param::U8(packed))];
    let measured = measure(&label, &g, &params, &x, out_shape.iter().product());
    // The in-arena slab the planner booked for this graph, asked of the backend so
    // this evidence cannot drift from the route (`backend/compile/lower.rs` sizes
    // the scratch from the same call, on the post-fusion graph). It is zero by
    // SCHEME, not by route: the planner skips every scheme `gemv_supports_scheme`
    // accepts (Q4_K/Q6_K/Q1_0), so an unaligned-k Q4_K node at m > 1 runs on the
    // host — the k=384 cell below — yet still books nothing.
    let scratch_graph_max = rlx_wgpu::gguf_gpu::dequant_gguf_scratch_bytes(&g);
    let report = CaseReport {
        scheme: format!("{scheme:?}"),
        m,
        weight_shape: format!("[k={k}, n={n}]"),
        label: label.clone(),
        route: wgpu_route(scheme, m, k, scratch_graph_max > 0),
        graph_context: BARE_GRAPH_CONTEXT.to_string(),
        packed_bytes,
        f32_image_bytes: k * n * 4,
        scratch_graph_max,
        tol,
        measured,
    };
    // Print before asserting: a failing cell's numbers are the evidence.
    report.print();
    assert_measured(&label, &report.measured, tol);
    report
}

#[test]
fn wgpu_gguf_dequant_matmul_prefill_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let cases: &[(QuantScheme, rlx_gguf::GgmlType, usize, usize, usize, f32)] = &[
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            640,
            2048,
            1e-2,
        ),
        // Batched prefill sizes: the row-grouped GEMM covers 2..~276 rows, and
        // m=96 is the top of the model-body range Metal caps its fused GEMM at.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            16,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            96,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            96,
            512,
            1024,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            96,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            16,
            512,
            1024,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            21,
            2048,
            640,
            1e-2,
        ),
        // Gemma 4 E2B double-wide MLP shapes (intermediate=12288) — the wgpu
        // KV-shared block error concentrates in these. gate/up: k=1536→n=12288;
        // down: k=12288→n=1536.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            5,
            1536,
            12288,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            5,
            12288,
            1536,
            1e-2,
        ),
        // Gemma 3 270M unsloth GGUF: q/k Q5_0, v Q8_0 (hidden=640).
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            1024,
            1e-2,
        ),
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            256,
            1e-2,
        ),
        (
            QuantScheme::GgufQ8_0,
            rlx_gguf::GgmlType::Q8_0,
            21,
            640,
            256,
            1e-2,
        ),
        // Custom 1-bit Q1_0 (prism-ml Bonsai-27B): 128-elem blocks. k a
        // multiple of 128; wgpu on-device dequant_gguf path vs CPU.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            5,
            512,
            256,
            1e-2,
        ),
        // Bonsai-ish prefill tiles (hidden=5120).
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            33,
            512,
            512,
            1e-2,
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            8,
            5120,
            5120,
            2e-2,
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            33,
            5120,
            5120,
            2e-2,
        ),
    ];
    for &(scheme, ggml, m, k, n, tol) in cases {
        // `run_case` asserts the cell's invariants (counted finiteness, zero input
        // ~ zero, input dependence, in-process repeat) with this cell's own `tol`
        // and prints its report before failing.
        run_case(&Cell {
            scheme,
            ggml,
            batch: &[],
            m,
            k,
            n,
            tol,
            label: format!("prefill {scheme:?} m={m} k={k} n={n}"),
        });
    }
}

/// Pack one FV5 block (104 bytes) from 256 five-value codes in {-2,-1,0,1,2}.
fn pack_fv5_block(codes: &[i8], s_lo: f32, s_hi: f32) -> Vec<u8> {
    let mut b = vec![0u8; 104];
    b[0..4].copy_from_slice(&s_lo.to_le_bytes());
    b[4..8].copy_from_slice(&s_hi.to_le_bytes());
    for (j, &c) in codes.iter().enumerate() {
        let (byte, bit) = (j / 8, 1u8 << (j % 8));
        let (p, ng, hi) = match c {
            1 => (true, false, false),
            2 => (true, false, true),
            -1 => (false, true, false),
            -2 => (false, true, true),
            _ => (false, false, false),
        };
        if p {
            b[8 + byte] |= bit;
        }
        if ng {
            b[40 + byte] |= bit;
        }
        if hi {
            b[72 + byte] |= bit;
        }
    }
    b
}

// End-to-end FV5 (Neutrino ternary) DequantMatMul: exercises the full
// lower → dispatch → dequant-to-scratch → matmul path on wgpu vs CPU.
// FV5 has no float quantizer (packs are made offline), so we pack directly.
#[test]
fn wgpu_fv5_dequant_matmul_prefill_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let (m, k, n) = (4usize, 256usize, 8usize); // k a multiple of 256 → 1 block/row
    let mut packed = Vec::new();
    for row in 0..n {
        let codes: [i8; 256] = std::array::from_fn(|j| match (j + row) % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => -1,
            _ => -2,
        });
        packed.extend_from_slice(&pack_fv5_block(&codes, 0.05, 0.2));
    }
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_fv5_dq_prefill");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufFV5,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    // Counted before the fold — see `Finiteness`.
    let gpu_fin = Finiteness::count(&gpu);
    let cpu_fin = Finiteness::count(&cpu);
    assert!(
        gpu_fin.is_clean() && cpu_fin.is_clean(),
        "wgpu FV5 m={m} k={k} n={n}: non-finite output — gpu={gpu_fin:?} cpu={cpu_fin:?}"
    );
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("wgpu FV5 matmul m={m} k={k} n={n}: max_abs={max_abs:.6e}");
    assert!(max_abs <= 1e-4, "wgpu FV5 prefill max_abs {max_abs} > 1e-4");
}

/// Byte accounting for the bounded model-class graph, accumulated as it is
/// built. This is the engine's own accounting, not RSS: the repo ships no
/// OS-RSS utility, and the wgpu arena is a single storage buffer spanning
/// weights, activations and scratch, so process RSS cannot attribute the packed
/// bytes to this test. [`print_memory`] reports one number that is deliberately
/// outside this struct — `pooled_out_slab`, the backend's process-global pooled
/// output buffer the windowed routes write through, which the arena plan does not
/// own and which this test only ever reads as a high-water mark: it is a static
/// that grows and is never released, and the tests in this binary run in
/// parallel, so the figure may belong to another cell's route.
#[derive(Default)]
struct MemAcc {
    /// Sum of the packed param byte lengths uploaded via `set_param_typed(U8)`.
    packed_resident: usize,
    /// Sum of `k*n*4` over the block's weights — the f32 image an unpack-at-load
    /// workaround would hold for the whole graph. The build materializes one
    /// weight at a time, so its own host peak is the widest projection
    /// (`4096*1024*4` = 16 MiB), not this sum.
    f32_image_total: usize,
    /// Graph-wide in-arena dequant scratch the wgpu planner reserves, taken from
    /// `gguf_gpu::dequant_gguf_scratch_bytes` — `max(k*n*4)` over the nodes whose
    /// scheme has no fused route (Q5_K / Q8_0 here). The fused Q4_K / Q6_K routes
    /// materialize nothing.
    scratch_graph_max: usize,
}

/// A bounded transformer-MLP-block graph and the bytes it needs.
struct MlpBlock {
    graph: Graph,
    params: Vec<(String, Param)>,
    mem: MemAcc,
    x: Vec<f32>,
    out_elems: usize,
}

/// The model-class graph's memory line — numbers only, one line, once per run.
/// The accounting, the fields and the measurement's basis are [`MemAcc`]'s doc.
fn print_memory(mem: &MemAcc) {
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    let pooled = rlx_wgpu::gguf_gpu::pooled_out_buf_bytes();
    eprintln!(
        "[rlx-gguf-mem] backend=wgpu packed_resident={} B ({:.2} MiB) f32_image_total={} B \
         ({:.2} MiB) in_arena_dequant_scratch={} B ({:.2} MiB) pooled_out_slab={} B ({:.2} MiB; \
         process-wide high-water mark, NOT in the arena accounting) scratch_vs_adapter={} \
         basis=engine param bytes + planner scratch (no OS-RSS utility in-repo)",
        mem.packed_resident,
        mib(mem.packed_resident),
        mem.f32_image_total,
        mib(mem.f32_image_total),
        mem.scratch_graph_max,
        mib(mem.scratch_graph_max),
        pooled,
        mib(pooled as usize),
        scratch_vs_adapter(mem.scratch_graph_max),
    );
}

/// Packed weights already quantized, keyed by param name. The model-class graph's
/// weights do not depend on `m` (only its shapes do), so the `m` loop must not
/// re-quantize ~55M elements per pass; a name is used for one weight per block.
type WeightCache = Vec<(String, Vec<u8>)>;

/// Quantize a pseudo-random `[n, k]` weight in memory and register it as a U8
/// param. `seed` decorrelates the weights so a cross-wired copy cannot pass.
/// `cache` returns an already-quantized weight for the same name.
fn add_packed_weight(
    g: &mut Graph,
    params: &mut Vec<(String, Param)>,
    name: &str,
    ggml: rlx_gguf::GgmlType,
    n: usize,
    k: usize,
    seed: f32,
    mem: &mut MemAcc,
    cache: &mut WeightCache,
) -> rlx_ir::NodeId {
    let packed = match cache.iter().find(|(cached, _)| cached == name) {
        Some((_, bytes)) => bytes.clone(),
        None => {
            let w_row: Vec<f32> = (0..k * n)
                .map(|idx| ((idx as f32 + seed) * 0.011).sin() * 0.3)
                .collect();
            let bytes = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
            cache.push((name.to_string(), bytes.clone()));
            bytes
        }
    };
    mem.packed_resident += packed.len();
    mem.f32_image_total += k * n * 4;
    let node = g.param(name, Shape::new(&[packed.len()], DType::U8));
    params.push((name.to_string(), Param::U8(packed)));
    node
}

/// A transformer-MLP block graph, the way a decoder layer is actually shaped:
/// `layers` × (RmsNorm → {Q5_K gate, Q6_K up} → SiLU(gate)·up → Q8_0 down →
/// Q4_K o-projection folded into the residual). Every packed matmul is a step in
/// the wgpu execution plan, so the fused and scratch routes are all exercised in
/// one graph. Every intermediate is `batch ++ [m, d]` so a `[1]` batch builds the
/// models' `[1, 1, h]` single-token decode. `cache` carries the quantized weights
/// across the caller's `m` loop.
fn build_mlp_block(
    batch: &[usize],
    m: usize,
    h: usize,
    i: usize,
    layers: usize,
    cache: &mut WeightCache,
) -> MlpBlock {
    let mut g = Graph::new("wgpu_gguf_mlp_block");
    // The 1-D `gamma`/`beta` params stay 1-D; everything else carries the batch.
    let f32_shape = |d: usize| {
        let mut s = batch.to_vec();
        s.extend_from_slice(&[m, d]);
        Shape::new(&s, DType::F32)
    };
    let x_in = g.input("x", f32_shape(h));
    let mut params: Vec<(String, Param)> = Vec::new();
    let mut mem = MemAcc::default();
    let mut cur = x_in;

    for l in 0..layers {
        let gamma = g.param(format!("gamma{l}"), Shape::new(&[h], DType::F32));
        let beta = g.param(format!("beta{l}"), Shape::new(&[h], DType::F32));
        params.push((format!("gamma{l}"), Param::F32(vec![1.0; h])));
        params.push((format!("beta{l}"), Param::F32(vec![0.0; h])));
        let hn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![cur, gamma, beta],
            f32_shape(h),
        );

        let wg = add_packed_weight(
            &mut g,
            &mut params,
            &format!("wg{l}"),
            rlx_gguf::GgmlType::Q5K,
            i,
            h,
            l as f32 * 7.0,
            &mut mem,
            cache,
        );
        let gate = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ5K,
            },
            vec![hn, wg],
            f32_shape(i),
        );
        let wu = add_packed_weight(
            &mut g,
            &mut params,
            &format!("wu{l}"),
            rlx_gguf::GgmlType::Q6K,
            i,
            h,
            l as f32 * 7.0 + 1.0,
            &mut mem,
            cache,
        );
        let up = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ6K,
            },
            vec![hn, wu],
            f32_shape(i),
        );
        let act = g.add_node(Op::Activation(Activation::Silu), vec![gate], f32_shape(i));
        let a = g.add_node(Op::Binary(BinaryOp::Mul), vec![act, up], f32_shape(i));

        let wd = add_packed_weight(
            &mut g,
            &mut params,
            &format!("wd{l}"),
            rlx_gguf::GgmlType::Q8_0,
            h,
            i,
            l as f32 * 7.0 + 2.0,
            &mut mem,
            cache,
        );
        let d = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ8_0,
            },
            vec![a, wd],
            f32_shape(h),
        );
        let wo = add_packed_weight(
            &mut g,
            &mut params,
            &format!("wo{l}"),
            rlx_gguf::GgmlType::Q4K,
            h,
            h,
            l as f32 * 7.0 + 3.0,
            &mut mem,
            cache,
        );
        let o = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K,
            },
            vec![hn, wo],
            f32_shape(h),
        );
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![d, o], f32_shape(h));
        cur = g.add_node(Op::Binary(BinaryOp::Add), vec![cur, s], f32_shape(h));
    }
    g.set_outputs(vec![cur]);

    let x: Vec<f32> = (0..m * h).map(|j| ((j as f32) * 0.017).sin()).collect();
    // The slab the planner books for this block, asked of the backend rather than
    // summed here, so the memory line and the cells' routes agree by construction.
    mem.scratch_graph_max = rlx_wgpu::gguf_gpu::dequant_gguf_scratch_bytes(&g);
    MlpBlock {
        graph: g,
        params,
        mem,
        x,
        // `x` is one element per `[m, h]` slice; a `[1]` batch multiplies the output.
        out_elems: batch.iter().product::<usize>() * m * h,
    }
}

/// The scheme × shape grid and the model-class graph: the four packed families at
/// the Metal grid's shapes (minus one — see the shape list), plus a bounded
/// model-class graph.
#[test]
fn wgpu_gguf_dequant_prefill_route_grid_and_model_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    // Part (a): the scheme × shape grid. Q5_K is the only in-scope scheme this file
    // had no prefill cell for at all; Q8_0 had one at (m=21, k=640, n=256) and its
    // wide prefill shapes are new here, as is the 3-D single-token cell. The shapes
    // straddle wgpu's own change point (m == 1 vs m > 1 on the Q4_K/Q6_K fused
    // routes), so each cell prints the route it should have taken rather than
    // assuming one — and `gguf_matmul_route.rs` confirms those routes from the
    // backend's own counters at these same shapes, the model-class widths included.
    let schemes: &[(QuantScheme, rlx_gguf::GgmlType, &str)] = &[
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, "Q4_K"),
        (QuantScheme::GgufQ5K, rlx_gguf::GgmlType::Q5K, "Q5_K"),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, "Q6_K"),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, "Q8_0"),
    ];
    // (batch, m, k, n, tol). One tolerance for the whole grid, this file's own
    // convention: 1e-2 sits ~36x above the worst observed cell (2.75e-4 at k=2048)
    // rather than being tightened to a number measured on one adapter, and it is
    // still orders of magnitude away from a zeroed or grossly wrong projection.
    // The shape list is the Metal mirror's minus its `m=96` k=256/n=640 row (this
    // file's pre-existing cases above already run that shape for Q4_K and Q6_K,
    // and wgpu has no routing boundary above `m == 1`, so the row adds no side of a
    // change point here) and plus an unaligned-k row Metal has no host route for.
    let shapes: &[(&'static [usize], usize, usize, usize, f32)] = &[
        (&[], 2, 256, 8, 1e-2), // tiny, on the grouped side of the m==1 change point
        // Unaligned k (1.5 K-quant superblocks): no fused Q4_K/Q6_K GEMM reads it,
        // so those schemes take the documented HOST route while Q5_K/Q8_0 stay on
        // scratch. This is the shape the route census counter-asserts (`host: 1`)
        // for them; the pre-existing k=640 case is mid-superblock too, but has no
        // census cell behind its route.
        (&[], 4, 384, 8, 1e-2),
        (&[], 128, 256, 640, 1e-2),   // grouped side, wide n
        (&[], 64, 2048, 512, 1e-2),   // key width 2048
        (&[], 192, 1024, 2048, 1e-2), // hundreds of rows + low-thousands width
        (&[], 1, 256, 640, 1e-2),     // 2-D decode
        (&[1], 1, 2048, 640, 1e-2),   // 3-D single-token decode: [1,1,k] -> [1,1,n]
    ];
    for &(scheme, ggml, sname) in schemes {
        for &(batch, m, k, n, tol) in shapes {
            run_case(&Cell {
                scheme,
                ggml,
                batch,
                m,
                k,
                n,
                tol,
                label: format!("grid {sname} m={m} k={k} n={n} batch={batch:?}"),
            });
        }
    }

    // Part (b): a bounded model-class graph at whole-model scale. The bound is a
    // design bound, not a measurement: 4 decoder-style blocks × (gate Q5_K
    // [4096,1024] + up Q6_K [4096,1024] + down Q8_0 [1024,4096] + o Q4_K
    // [1024,1024]) = ~55M packed elements → ~45 MiB packed weights, and ~208 MiB of
    // f32 if the whole block were expanded at once (the build materializes one
    // weight at a time, so its own peak is the widest projection, 16 MiB), the 16 MiB
    // dequant slab the planner books for that projection (`max(k*n*4)` = 4096*1024*4)
    // reserved once for the eight Q5_K/Q8_0 nodes, and three cells: m=1 in the
    // models' 3-D single-token form `[1, 1, H]`, m=96 and m=192 at `[m, H]` — one
    // bounded test, weights built in memory with `rlx_gguf::quantize` (no file).
    // The measured figures are what the `[rlx-gguf-mem]` line prints.
    const H: usize = 1024;
    const I: usize = 4096;
    const LAYERS: usize = 4;
    // The block's weights are m-independent, so one cache spans the loop: each
    // weight is quantized once rather than once per m.
    let mut weight_cache = WeightCache::new();
    let mut mem_printed = false;
    for &(batch, m) in &[
        (&[1usize][..], 1usize),
        (&[][..], 96usize),
        (&[][..], 192usize),
    ] {
        let block = build_mlp_block(batch, m, H, I, LAYERS, &mut weight_cache);
        let label = format!("model mlp-block {LAYERS}L batch={batch:?} m={m} H={H} I={I}");
        // Q4_K (o-projection, k=H) and Q6_K (up, k=H) sit on the change point:
        // m=1 is the decode (per-row GEMV) side, m>=2 the row-grouped side.
        // Q5_K (gate, k=H) and Q8_0 (down, k=I) have only their single scratch
        // route at every m — all three m take the same one. Every projection has k a
        // whole number of blocks, so the slab's presence only affects the qualifier
        // the unaligned-k arm would print, and it is the block's own reservation.
        let scratch_reserved = block.mem.scratch_graph_max > 0;
        let mut routes = [
            wgpu_route(QuantScheme::GgufQ5K, m, H, scratch_reserved),
            wgpu_route(QuantScheme::GgufQ6K, m, H, scratch_reserved),
            wgpu_route(QuantScheme::GgufQ8_0, m, I, scratch_reserved),
            wgpu_route(QuantScheme::GgufQ4K, m, H, scratch_reserved),
        ]
        .to_vec();
        routes.sort();
        routes.dedup();

        let measured = measure(
            &label,
            &block.graph,
            &block.params,
            &block.x,
            block.out_elems,
        );
        // Per-m parity tolerance. The block's residual stream reaches |y| in the
        // low thousands (two un-normalized MLP expansions, no downstream norm), so
        // no grid-cell absolute bound holds here. Measured max_abs / cpu_max
        // (relative):
        //   m=1 (3-D) 1.12e-2 / 1617 = 6.9e-6
        //   m=96      2.01e-2 / 2939 = 6.8e-6
        //   m=192     2.25e-2 / 2940 = 7.6e-6
        // All three sit on the same ~7e-6 relative floor — the absolute value
        // tracks cpu_max, not the route — so one absolute bound with >=2x headroom
        // over the largest measured side is the honest form, PAIRED with the
        // relative bound asserted below. What actually bounds the fused GEMV/GEMM
        // and the scratch-dequant accumulation is the relative error.
        let tol = 5e-2f32;
        let report = CaseReport {
            scheme: "Q5K/Q6K/Q8_0/Q4K".to_string(),
            m,
            weight_shape: format!("gate/up [k={H}, n={I}], down [k={I}, n={H}], o [k={H}, n={H}]"),
            label,
            route: routes.join(" | "),
            graph_context: MODEL_GRAPH_CONTEXT.to_string(),
            packed_bytes: block.mem.packed_resident,
            f32_image_bytes: block.mem.f32_image_total,
            scratch_graph_max: block.mem.scratch_graph_max,
            tol,
            measured,
        };
        report.print();
        assert_measured(&report.label, &report.measured, tol);
        // The absolute bound above scales with |y|, so the per-element error is
        // also asserted against the reference magnitude: every observed cell sits
        // at ~7e-6 relative, and 1e-4 is the gross-divergence ceiling.
        let rel = report.measured.max_abs / report.measured.cpu_max.max(1e-30);
        assert!(
            rel <= 1e-4,
            "{}: max_abs is {rel:.3e} of the reference magnitude, over 1e-4",
            report.label
        );
        if !mem_printed {
            print_memory(&block.mem);
            mem_printed = true;
        }
    }
}
