// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed GGUF `Op::DequantMatMul` — Metal vs CPU.
//!
//! The encoder picks the route from a fixed if-ladder (see [`metal_route`],
//! which mirrors it because **Metal records no route signal**): m == 1 fuses
//! into a GEMV kernel, 1 < m <= the `RLX_METAL_Q4K_GEMM_MAX_M` cap (96 by
//! default) into a fused dequant-in-register GEMM, and m above the cap falls to
//! the `dequant_gguf` MSL kernel expanding the packed weight `[n, k]` into a
//! graph-wide shared f32 scratch, then `encode_mps_sgemm_bt`. That shared
//! scratch is the regime the coverage grid straddles. Packed GGUF never reaches
//! the MPSGraph path (`can_lower_dequant_in_mps` is hardcoded false); a graph
//! built from packed projections still runs the hybrid schedule, with every
//! `DequantMatMul` a thunk boundary — which is what the model-class graph below
//! exercises.
//!
//! # Running
//!
//! The cells are behind the `backend-tests` opt-in:
//! `cargo test --release -p rlx-metal --features backend-tests --test
//! metal_gguf_dequant_matmul_prefill_parity -- --nocapture` (on Darwin
//! `just test-gpu` reaches them). `-- --nocapture` is what makes the per-cell
//! `[rlx-gguf-*]` evidence visible; without it the run prints none of it.
//!
//! Each cell's CPU reference is `rlx-cpu`'s `gguf_matmul_bt_dispatch_at`
//! behind an ordinary `Device::Cpu` `Session`; the engine exposes NO signal
//! naming which `rlx-cpu` route served a comparison, so — rather than
//! invent one — the reference is taken in the same process under identical
//! conditions, with no process-global knob pinned for it. Each cell's
//! `repeat_max_abs` re-runs that reference from a fresh `Session` and compares it
//! element for element, so a reference-route change inside a cell would surface
//! there rather than pass unnoticed. Cross-process reproducibility is 3 serial
//! invocations of that same command with `--test-threads=1`, which a single test
//! cannot observe.
//!
//! The PTQ1_0 knob tests are the one place that does pin a knob
//! (`RLX_METAL_PTQ1_0_{NSG,FUSED_DISABLE,INT_PIPE}`, via [`EnvKnob`]'s
//! process-wide `rlx_ir::env` overrides). They take [`DEVICE_LOCK`], so a test's
//! setting cannot displace the route or memory figure another test prints. No
//! grid or model cell sets a knob.
//!
//! This suite and `rlx-wgpu`'s `gguf_dequant_matmul_prefill_parity.rs` are
//! deliberate mirrors of each other: an edit to this harness must be applied to
//! both. They differ in backend-specific values — each file's tolerance
//! table (`1e-3` / `2e-3` here), the change-point rows its grid needs (the
//! `m=96`/`m=128` pair below, which only Metal has a boundary for), and the route
//! strings, adapter limits and memory figures, which are this backend's own — and
//! in device serialization: every device test here takes [`DEVICE_LOCK`], wgpu's
//! take none, deliberately (see that file).

#![cfg(target_os = "macos")]

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

/// Overrides an encoder knob for the duration of a scope and clears the override
/// on the way out — including when an assertion inside the scope unwinds, which
/// would otherwise leak into every later test in this binary and silently change
/// what they measure.
///
/// Goes through [`rlx_ir::env`]'s code-override layer, which is what the encoder
/// reads (`rlx_ir::env::flag` / `parse_or`) and which takes precedence over the
/// process environment, rather than through `std::env`: the tests in this binary
/// run in parallel, so mutating the process environment would be a data race,
/// and a value the launching environment set applies again once the guard drops.
struct EnvKnob(&'static str);

impl EnvKnob {
    fn new(name: &'static str, value: Option<&str>) -> Self {
        match value {
            Some(v) => rlx_ir::env::set(name, v),
            None => rlx_ir::env::unset(name),
        }
        Self(name)
    }
}

impl Drop for EnvKnob {
    fn drop(&mut self) {
        rlx_ir::env::unset(self.0);
    }
}

/// Serializes the device work in this binary: every device test takes it, which —
/// not the grouping of cells into `#[test]`s — is what keeps them from running
/// concurrently. The two process-global things it keeps apart are documented in
/// the tests below:
///
/// - `MPSMatrixMultiplication`'s matrix/kernel cache, which the prefill cells past
///   the fused-GEMM cap reach from parallel test threads' independent `Session`s;
///   `mps_blas` guards it (`CACHE_GUARD` around every encode, and a `KernelKey`
///   that carries the thread, pinned by `tests/mps_matmul_concurrency.rs`);
/// - [`EnvKnob`] overrides are process-wide, so without this one test's setting
///   could change the route — and the memory figure — a concurrent test prints.
static DEVICE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The locked state, held for the whole of a device test. On release it does what
/// the crate's other device-serializing test guards do — command queue drained,
/// `MPSMatrixMultiplication` caches dropped.
struct DeviceGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

fn device_lock() -> DeviceGuard {
    // A cell that panics must not poison the tests queued behind it.
    DeviceGuard(DEVICE_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Finiteness census of one `[f32]` output: NaN, +inf and −inf counted as three
/// separate numbers, so an infinity cannot hide behind a NaN (or behind
/// `f32::max`, which returns the non-NaN operand). Cells assert this clean BEFORE
/// folding anything into a max-abs metric — an all-NaN output would otherwise
/// fold to `0.0` and pass.
///
/// `gguf_matmul_route.rs` (a separate wgpu test binary, so it cannot share this)
/// carries the same count-before-fold rule in `parity_max_abs`; keep the two in
/// step.
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
    /// Finiteness of the Metal output under a zero input — a non-zero count is
    /// a failure dressed up as "input dependence".
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
/// without producing one. Device-free, so it holds even where no Metal device
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

/// The effective `RLX_METAL_Q4K_GEMM_MAX_M` cap, read the way the encoder reads
/// it: the explicit override when set, else 96 (or 32 with the XS kernel off).
fn q4k_gemm_max_m() -> usize {
    match rlx_ir::env::var("RLX_METAL_Q4K_GEMM_MAX_M").and_then(|v| v.parse().ok()) {
        Some(v) => v,
        None if rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_XS_DISABLE") => 32,
        None => 96,
    }
}

/// The adapter limits the graph-wide dequant scratch is sized against:
/// `maxBufferLength` bounds the single arena `MTLBuffer` the scratch tail is
/// appended to (`backend/compile.rs`), and `recommendedMaxWorkingSetSize` is the
/// device's own budget for that buffer.
fn adapter_limits() -> String {
    match rlx_metal::device::metal_device() {
        Some(d) => format!(
            "adapter_max_buffer_length={} adapter_max_working_set={}",
            d.device.max_buffer_length(),
            d.max_working_set
        ),
        None => "adapter_max_buffer_length=? (no Metal device)".to_string(),
    }
}

/// The graph-wide dequant scratch against the single `MTLBuffer` the arena is
/// appended to, formatted for the cell and memory lines. Uses the same device
/// accessor as [`adapter_limits`], so the two cannot disagree.
fn scratch_vs_adapter(scratch: usize) -> String {
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    match rlx_metal::device::metal_device() {
        Some(d) => {
            let limit = d.device.max_buffer_length();
            format!(
                "{:.2} MiB / {:.2} MiB max_buffer_length ({:.4}%)",
                mib(scratch),
                mib(limit as usize),
                100.0 * scratch as f64 / limit as f64
            )
        }
        None => "no Metal device".to_string(),
    }
}

/// Every way the encoder's host fallback fires (`deferred_host`): the GPU path
/// switched off, a scheme with no Metal dequant kernel, or an op that needs the
/// f32 dequant slab in a graph whose arena reserved none.
const HOST_FALLBACK_ROUTE: &str = "host fallback (deferred_host)";
const SCRATCH_ROUTE: &str = "dequant_scratch + mps_sgemm_bt (shared graph-wide scratch)";

/// The scratch route, or the host fallback when the graph's arena reserved no
/// slab for it (the encoder's `needs_scratch && dequant_scratch_off == 0`).
/// Which graphs reserve one is asked of the engine itself
/// (`rlx_metal::backend::dequant_gguf_scratch_bytes`), never mirrored here.
fn scratch_or_host(arena_scratch_reserved: bool) -> String {
    if arena_scratch_reserved {
        SCRATCH_ROUTE.to_string()
    } else {
        HOST_FALLBACK_ROUTE.to_string()
    }
}

/// The route the Metal encoder takes for a packed `DequantMatMul`, derived
/// statically from the `Thunk::DequantMatMulGguf` if-ladder.
///
/// **There is NO runtime route probe on Metal** — the encoder
/// (`rlx-metal/src/backend/encode/mod.rs`) picks a branch and never records which
/// one served an op — so this mirrors the ladder's gates and every cell prints
/// the result, which is what shows a reader that the ladder moved. A mirror
/// drifts: it carries arms only for the schemes this file feeds (Q4_K / Q5_K /
/// Q6_K / Q8_0 in the grid, Q1_0 / PTQ1_0 in the cases that reuse `run_case`).
/// Anything the ladder would route differently from what is left here falls to
/// the scratch arm, except a scheme with no Metal dequant kernel at all, which
/// the host fallback below does cover.
///
/// `arena_scratch_reserved` is the caller's statement of the graph-level fact the
/// encoder tests as `dequant_scratch_off != 0`; the callers take it from the
/// graph's own reservation
/// (`rlx_metal::backend::dequant_gguf_scratch_bytes`, directly or through the
/// block accounting) rather than deciding it.
fn metal_route(
    scheme: QuantScheme,
    m: usize,
    k: usize,
    n: usize,
    arena_scratch_reserved: bool,
) -> String {
    if rlx_metal::runtime_config().dequant_gpu_disable
        || !rlx_metal::backend::has_metal_dequant_kernel(scheme)
    {
        return HOST_FALLBACK_ROUTE.to_string();
    }
    let fused = if m == 1 {
        metal_decode_gemv_route(scheme, k, n)
    } else {
        metal_prefill_gemm_route(scheme, m, k)
    };
    fused.unwrap_or_else(|| scratch_or_host(arena_scratch_reserved))
}

/// The fused single-pass GEMV a decode (`m == 1`) packed op takes, or `None` when
/// the encoder's ladder falls through to the scratch route.
///
/// Q4_K is the only one of the four grid families whose simdgroup GEMV constrains
/// `n` (8 output columns per simdgroup); the Q5_K / Q6_K / Q8_0 simdgroup kernels
/// reduce a row across 32 threads and are row-guarded.
///
/// Q1_0 has no arm here on purpose: every Q1_0 op that reaches `run_case` is
/// `m > 1` and takes [`metal_prefill_gemm_route`]'s uncapped Q1_0 GEMM instead.
fn metal_decode_gemv_route(scheme: QuantScheme, k: usize, n: usize) -> Option<String> {
    Some(match scheme {
        QuantScheme::GgufQ4K
            if k.is_multiple_of(256) && !rlx_ir::env::flag("RLX_METAL_Q4K_FUSED_DISABLE") =>
        {
            format!(
                "fused GEMV (q4k_mv_f32_sg when n%8==0 and RLX_METAL_Q4K_SG_DISABLE is off — \
                 8 outputs per simdgroup — else q4k_mv_f32; n={n}; k%256==0)"
            )
        }
        QuantScheme::GgufQ5K
            if k.is_multiple_of(256) && !rlx_ir::env::flag("RLX_METAL_Q5K_FUSED_DISABLE") =>
        {
            format!(
                "fused GEMV (q5k_mv_f32_sg unless RLX_METAL_Q5K_SG_DISABLE — 32 threads reduce \
                 a row, so no n constraint — else q5k_mv_f32; n={n}; k%256==0)"
            )
        }
        QuantScheme::GgufQ6K
            if k.is_multiple_of(256) && !rlx_ir::env::flag("RLX_METAL_Q6K_FUSED_DISABLE") =>
        {
            format!(
                "fused GEMV (q6k_mv_f32_sg unless RLX_METAL_Q6K_SG_DISABLE — 32 threads reduce \
                 a row, so no n constraint — else q6k_mv_f32; n={n}; k%256==0)"
            )
        }
        QuantScheme::GgufQ8_0
            if k.is_multiple_of(32) && !rlx_ir::env::flag("RLX_METAL_Q80_FUSED_DISABLE") =>
        {
            format!(
                "fused GEMV (q8_0_mv_f32_sg unless RLX_METAL_Q8_0_SG_DISABLE — 32 threads \
                 reduce a row, so no n constraint — else q8_0_mv_f32; n={n}; k%32==0)"
            )
        }
        QuantScheme::GgufPtq1_0
            if k.is_multiple_of(128)
                && n.is_multiple_of(8)
                && !rlx_ir::env::flag("RLX_METAL_PTQ1_0_FUSED_DISABLE") =>
        {
            "fused GEMV (q1_0_mv_f32_sg_flags; n%8==0; k%128==0)".to_string()
        }
        _ => return None,
    })
}

/// The fused dequant-in-register GEMM a prefill (`m > 1`) packed op takes, or
/// `None` when the encoder's ladder falls through to the scratch route.
///
/// Q1_0 carries its own fused GEMM with no `m` cap (the encoder's gate is shared
/// with Q2_0, which no cell here feeds); the K-quant and Q8_0 GEMMs share the
/// `RLX_METAL_Q4K_GEMM_MAX_M` cap (96 by default), above which the encoder
/// dequantizes into the graph-wide slab and runs MPS sgemm.
fn metal_prefill_gemm_route(scheme: QuantScheme, m: usize, k: usize) -> Option<String> {
    if matches!(scheme, QuantScheme::GgufQ1_0)
        && k.is_multiple_of(128)
        && !rlx_ir::env::flag("RLX_METAL_Q1_0_FUSED_DISABLE")
    {
        return Some("fused Q1_0 GEMM (q1_0_mm_f32; k%128==0; no cap)".to_string());
    }
    let cap = q4k_gemm_max_m();
    if m > cap || !k.is_multiple_of(256) {
        return None;
    }
    Some(match scheme {
        QuantScheme::GgufQ4K if !rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_DISABLE") => {
            let k_name = if rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_XS_DISABLE") {
                "q4k_mm_f32"
            } else {
                "q4k_mm_f32_xs"
            };
            format!("fused GEMM ({k_name}; cap={cap}; k%256==0)")
        }
        QuantScheme::GgufQ5K if !rlx_ir::env::flag("RLX_METAL_Q5K_GEMM_DISABLE") => {
            format!("fused XS GEMM (q5k_mm_f32_xs; cap={cap}; k%256==0)")
        }
        QuantScheme::GgufQ6K if !rlx_ir::env::flag("RLX_METAL_Q6K_GEMM_DISABLE") => {
            let k_name = if rlx_ir::env::flag("RLX_METAL_Q6K_GEMM_XS_DISABLE") {
                "q6k_mm_f32"
            } else {
                "q6k_mm_f32_xs"
            };
            format!("fused GEMM ({k_name}; cap={cap}; k%256==0)")
        }
        QuantScheme::GgufQ8_0 if !rlx_ir::env::flag("RLX_METAL_Q8_0_GEMM_DISABLE") => {
            format!("fused XS GEMM (q8_0_mm_f32_xs; cap={cap}; k%256==0)")
        }
        _ => return None,
    })
}

/// Run one compiled graph on Metal and the CPU reference and return every
/// number the cells assert on. `label` only shapes failure messages.
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
    let mut cm = Session::new(Device::Metal).compile(g.clone());
    set_params(&mut cm, params);
    let metal = cm.run(&[("x", x)]).remove(0);
    let metal_zero = cm.run(&[("x", zero.as_slice())]).remove(0);

    let mut cc = Session::new(Device::Cpu).compile(g.clone());
    set_params(&mut cc, params);
    let cpu = cc.run(&[("x", x)]).remove(0);

    // Fresh `Session`s, same graph and input: in-process reproducibility for
    // both backends. Reusing `cm`/`cc` would only re-measure the same objects.
    let mut cm2 = Session::new(Device::Metal).compile(g.clone());
    set_params(&mut cm2, params);
    let metal2 = cm2.run(&[("x", x)]).remove(0);
    let mut cc2 = Session::new(Device::Cpu).compile(g.clone());
    set_params(&mut cc2, params);
    let cpu2 = cc2.run(&[("x", x)]).remove(0);

    for (what, v) in [
        ("metal", &metal),
        ("metal_zero", &metal_zero),
        ("cpu", &cpu),
        ("metal2", &metal2),
        ("cpu2", &cpu2),
    ] {
        assert_eq!(v.len(), out_elems, "{label}: {what} output length");
    }

    // Counted before any fold — see `Finiteness`: this assert is what stops a NaN
    // from disappearing into `max_abs`, and it runs on every cell.
    let metal_fin = Finiteness::count(&metal);
    let cpu_fin = Finiteness::count(&cpu);
    let zero_fin = Finiteness::count(&metal_zero);
    let metal2_fin = Finiteness::count(&metal2);
    let cpu2_fin = Finiteness::count(&cpu2);
    assert!(
        metal_fin.is_clean()
            && cpu_fin.is_clean()
            && zero_fin.is_clean()
            && metal2_fin.is_clean()
            && cpu2_fin.is_clean(),
        "{label}: non-finite output — gpu={metal_fin:?} cpu={cpu_fin:?} zero={zero_fin:?} gpu2={metal2_fin:?} cpu2={cpu2_fin:?}"
    );

    let measured = Measured {
        gpu: metal_fin,
        cpu: cpu_fin,
        zero: zero_fin,
        cpu_max: max_abs(&cpu),
        gpu_max: max_abs(&metal),
        zero_max: max_abs(&metal_zero),
        max_abs: max_diff(&cpu, &metal),
        input_dep: max_diff(&metal, &metal_zero),
        repeat_max_abs: max_diff(&metal, &metal2).max(max_diff(&cpu, &cpu2)),
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
    assert!(r.gpu_max > 0.0, "{label}: Metal output is all zeros");
    assert!(
        r.zero_max <= tol,
        "{label}: zero input did not produce ~zero output (zero_max={})",
        r.zero_max
    );
    assert!(
        r.max_abs <= tol,
        "{label}: Metal vs CPU max_abs={} (tol {tol})",
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
    /// bare matmul; the model-class graph is a hybrid schedule.
    graph_context: String,
    packed_bytes: usize,
    f32_image_bytes: usize,
    scratch_graph_max: usize,
    measured: Measured,
}

/// The grid cells' graph: one packed matmul and nothing around it.
const BARE_GRAPH_CONTEXT: &str = "bare single packed matmul (no norm/activation to fuse)";
/// The model-class graph's graph: what the Metal encoder does with a real block.
const MODEL_GRAPH_CONTEXT: &str = "hybrid schedule: each RmsNorm lowers into an MPSGraph segment, \
     every packed matmul is a thunk boundary, and the dequant slab is sized for the widest \
     projection. At m=1 and m=96 every packed op routes fused and does not read the slab; at m=192, \
     past the fused cap, all 16 do";

impl CaseReport {
    fn print(&self) {
        let r = &self.measured;
        eprintln!(
            "[rlx-gguf-cell] backend=metal scheme={} m={} weight_shape={} label=\"{}\" route=\"{}\" \
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

/// Build `x @ dequant(w)[n, k]^T` for a packed GGUF weight and compare Metal
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

    let mut g = Graph::new("gguf_dq_prefill");
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
    // The graph-wide dequant slab the planner booked for this one-op graph, asked
    // of the backend so this evidence cannot drift from the encoder's rule
    // (`backend/compile.rs` sizes `dequant_scratch_off` from the same call, made on
    // the post-fusion graph; fusion does not change these graphs). What a zero
    // means, and the reason it is not "every op reads its weight directly", is
    // documented on `dequant_gguf_scratch_bytes` itself.
    let scratch_graph_max = rlx_metal::backend::dequant_gguf_scratch_bytes(&g);
    let report = CaseReport {
        scheme: format!("{scheme:?}"),
        m,
        weight_shape: format!("[k={k}, n={n}]"),
        label: label.clone(),
        route: metal_route(scheme, m, k, n, scratch_graph_max > 0),
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

/// Serialized by [`DEVICE_LOCK`], like every device test in this binary (see its
/// doc): the prefill cells past the fused-GEMM cap reach `MPSMatrixMultiplication`'s
/// process-global matrix/kernel cache, which is why this file's cases were once
/// kept in a single `#[test]` — the grouping is not what serializes them.
///
/// A CHAIN of DequantMatMuls interspersed with activations — forces many
/// MPSGraph/thunk segment boundaries (`mps_graph_hybrid` splits at every
/// DequantMatMul), the way the 64-layer 27B does. A single DequantMatMul is
/// bit-exact; this checks the metal segmentation/boundary-I/O across many.
#[test]
fn gguf_dequant_matmul_chain_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let m = 4usize;
    let k = 256usize; // square weights so we can chain
    let n_layers = 8usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("gguf_dq_chain");
    let mut cur = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    for l in 0..n_layers {
        let w_row: Vec<f32> = (0..k * k)
            .map(|i| ((i as f32 + l as f32 * 7.0) * 0.011).sin() * 0.3)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{l}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![cur, w],
            Shape::new(&[m, k], DType::F32),
        );
        // activation between layers → forces a segment boundary + keeps
        // values bounded (silu).
        cur = g.add_node(
            Op::Activation(Activation::Silu),
            vec![mm],
            Shape::new(&[m, k], DType::F32),
        );
        packs.push((name, packed));
    }
    g.set_outputs(vec![cur]);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = cpu.iter().zip(&metal).map(|(a, b)| a * b).sum();
    let nc: f32 = cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nm: f32 = metal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (nc * nm + 1e-9);
    eprintln!("gguf_dequant_matmul CHAIN({n_layers}L) Q1_0: max_abs={max_abs:.6e} cos={cos:.6}");
    assert!(
        cos > 0.999,
        "dequant-matmul chain metal diverges: cos={cos} max_abs={max_abs}"
    );
}

/// Isolates the Bonsai-27B **padded-prefill hybrid** divergence: a leading
/// `RmsNorm` (MPSGraph-lowered) feeding a Q1_0 `DequantMatMul` chain (thunk
/// boundaries) at a LARGE `m` (padded seq, e.g. 96). The 27B degenerated
/// (cos≈-0.14) only at max_seq≈96 through `run_via_mps_hybrid`; the m=4 chain
/// test above is bit-exact, so the trigger is the large-m boundary hand-off
/// between an MPSGraph sub-graph output and the next thunk's input.
/// `RLX_HYBRID_M` overrides m (default 96); set 4 to see it pass.
#[test]
fn hybrid_rmsnorm_dequant_large_m_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let m: usize = rlx_ir::env::var("RLX_HYBRID_M")
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    let k: usize = rlx_ir::env::var("RLX_HYBRID_K")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256); // square so we can chain
    let n_layers = 4usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("hybrid_rmsnorm_dq");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[k], DType::F32));
    let beta = g.param("beta", Shape::new(&[k], DType::F32));
    // RmsNorm → MPSGraph segment; its output feeds the first dequant thunk.
    let mut cur = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x_in, gamma, beta],
        Shape::new(&[m, k], DType::F32),
    );
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    for l in 0..n_layers {
        let w_row: Vec<f32> = (0..k * k)
            .map(|i| ((i as f32 + l as f32 * 7.0) * 0.011).sin() * 0.3)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{l}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![cur, w],
            Shape::new(&[m, k], DType::F32),
        );
        // RmsNorm between layers → another MPSGraph segment boundary, plus
        // keeps values bounded (mirrors qwen35's pre-attn / pre-ffn norms).
        cur = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![mm, gamma, beta],
            Shape::new(&[m, k], DType::F32),
        );
        packs.push((name, packed));
    }
    g.set_outputs(vec![cur]);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let gamma_v = vec![1.0f32; k];
    let beta_v = vec![0.0f32; k];
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param("gamma", &gamma_v);
        c.set_param("beta", &beta_v);
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = cpu.iter().zip(&metal).map(|(a, b)| a * b).sum();
    let nc: f32 = cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nm: f32 = metal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (nc * nm + 1e-9);
    eprintln!(
        "hybrid RmsNorm+Q1_0 chain m={m} k={k} {n_layers}L: max_abs={max_abs:.6e} cos={cos:.6}"
    );
    assert!(
        cos > 0.999,
        "hybrid rmsnorm+dequant m={m} metal diverges: cos={cos} max_abs={max_abs}"
    );
}

/// Fused `q1_0_mm_f32` parity: SEVERAL large-n Q1_0 `DequantMatMul`s in ONE
/// graph (all reading the same `x`), the way each Bonsai-27B layer fires
/// qkv/gate/ffn projections. Exercises the packed-weight-direct GEMM at
/// Bonsai's k=5120/n=6144 scale and asserts non-zero, NaN-free output matching
/// CPU. (NB: the shared-scratch race that zeroed these projections on the old
/// `dequant_gguf → scratch → MPS sgemm` path only reproduces at full 64-layer
/// graph scale — see the e2e Bonsai run — so this guards the fused path's
/// correctness rather than reproducing the race itself.)
#[test]
fn gguf_dequant_matmul_multi_large_n_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let m = 8usize;
    let k = 5120usize; // Bonsai hidden
    let n = 6144usize; // Bonsai GDN inner (large n that raced to zero)
    let n_proj = 4usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("gguf_dq_multi_large");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut outs = Vec::new();
    for p in 0..n_proj {
        let w_row: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 + p as f32 * 3.0) * 0.011).sin() * 0.4)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{p}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![x_in, w],
            Shape::new(&[m, n], DType::F32),
        );
        outs.push(mm);
        packs.push((name, packed));
    }
    g.set_outputs(outs);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let run = |device: Device| -> Vec<Vec<f32>> {
        let mut c = Session::new(device).compile(g.clone());
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())])
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    for p in 0..n_proj {
        let mmax = metal[p].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // Three-way count: the old `is_nan()`-only check let an infinity through.
        let fin = Finiteness::count(&metal[p]);
        let max_abs = cpu[p]
            .iter()
            .zip(&metal[p])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "proj {p}: metal_max={mmax:.4} nan={} pos_inf={} neg_inf={} max_abs_vs_cpu={max_abs:.4}",
            fin.nan, fin.pos_inf, fin.neg_inf
        );
        assert!(
            mmax > 0.0 && fin.is_clean(),
            "proj {p} Metal produced zeros/NaN/Inf (scratch race)"
        );
        assert!(max_abs < 2.0, "proj {p} Metal vs CPU max_abs={max_abs}");
    }
}

#[test]
fn gguf_dequant_matmul_prefill_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    // (scheme, ggml, m, k, n, tol, label)
    let cases: &[(
        QuantScheme,
        rlx_gguf::GgmlType,
        usize,
        usize,
        usize,
        f32,
        &str,
    )] = &[
        // Smallest prefill: m=4, one Q4_K superblock (k=256), n=8.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-3,
            "Q4_K prefill",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-3,
            "Q6_K prefill",
        ),
        // k=512 (two superblocks), wider n, larger m → real GEMM tiling.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            7,
            512,
            16,
            1e-3,
            "Q4_K multi-superblock",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            7,
            512,
            16,
            1e-3,
            "Q6_K multi-superblock",
        ),
        // Decode (m=1) regression guard: fused GEMV path must stay correct.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            1,
            256,
            8,
            1e-3,
            "Q4_K decode",
        ),
        // Q1_0 (prism-ml Bonsai-27B) — 128-elem blocks, on-device dequant.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            4,
            256,
            8,
            1e-2,
            "Q1_0 prefill",
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            7,
            512,
            16,
            2e-2,
            "Q1_0 multi-block",
        ),
        // Bonsai FFN scale: k=5120 (hidden), n=6144 (ffn). Directly the
        // 27B-on-Metal DequantMatMul the synth F32 path can't exercise.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            8,
            5120,
            6144,
            2.0,
            "Q1_0 bonsai-ffn-scale",
        ),
        // PADDED-PREFILL scale: m=96 (the max_seq bucket that degenerated the
        // real 27B), k=5120, n=10240 (GDN qkv projection). The dump showed
        // this exact node emit ALL zeros (nz=0/983040) on Metal at m=96 while
        // m=8/32 are correct — isolates encode_mps_sgemm_bt at large m·n.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            96,
            5120,
            10240,
            3.0,
            "Q1_0 bonsai m96 n10240",
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            96,
            5120,
            6144,
            3.0,
            "Q1_0 bonsai m96 n6144",
        ),
    ];
    for &(scheme, ggml, m, k, n, tol, label) in cases {
        run_case(&Cell {
            scheme,
            ggml,
            batch: &[],
            m,
            k,
            n,
            tol,
            label: label.to_string(),
        });
    }
}

#[test]
fn q1_0_decode_amp_f16_gemv_only_matches_f32() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let m = 1usize;
    let k = 256usize;
    let n = 64usize;
    let scheme = QuantScheme::GgufQ1_0;
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("q1_amp_gemv");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let feeds = [("x", x.as_slice())];
    let run = |device: Device, amp: bool| -> Vec<f32> {
        let mut opts = rlx_runtime::CompileOptions::new();
        if amp {
            opts = opts.policy(rlx_runtime::PrecisionPolicy::AutoMixed);
        }
        let mut c = Session::new(device).compile_with(g.clone(), &opts);
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&feeds).remove(0)
    };

    let cpu = run(Device::Cpu, false);
    let metal_f32 = run(Device::Metal, false);
    let metal_amp = run(Device::Metal, true);
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb + 1e-9)
    };
    let c_amp = cos(&cpu, &metal_amp);
    let c_pair = cos(&metal_f32, &metal_amp);
    eprintln!("q1_0 AMP f16 GEMV-only: cos(cpu,amp)={c_amp:.6} cos(f32,amp)={c_pair:.6}");
    assert!(
        c_amp > 0.99,
        "Metal AMP Q1 GEMV diverges vs CPU: cos={c_amp}"
    );
    assert!(
        c_pair > 0.99,
        "Metal AMP vs F32 Q1 GEMV diverges: cos={c_pair}"
    );
}

/// True f16 residual stream under AMP: decode Q1_0 GEMV + residual add.
/// Compares Metal AlwaysF32 vs Metal AutoMixed (f16 acts / residual) and
/// both against CPU F32. Exercises `x_f16` / `dst_f16` / `res_f16` on the
/// simdgroup Q1 kernels (weight-BW path: packed W + half activation traffic).
#[test]
fn q1_0_decode_amp_f16_residual_matches_f32() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let m = 1usize;
    let k = 256usize;
    let n = 64usize; // %8==0 → SG path
    let scheme = QuantScheme::GgufQ1_0;

    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let res: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 0.25).collect();

    let mut g = Graph::new("q1_amp_residual");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let res_in = g.input("res", Shape::new(&[m, n], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let mm = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    let y = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![res_in, mm],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let feeds = [("x", x.as_slice()), ("res", res.as_slice())];
    let run = |device: Device, amp: bool| -> Vec<f32> {
        let mut opts = rlx_runtime::CompileOptions::new();
        if amp {
            opts = opts.policy(rlx_runtime::PrecisionPolicy::AutoMixed);
        }
        let mut c = Session::new(device).compile_with(g.clone(), &opts);
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&feeds).remove(0)
    };

    let cpu = run(Device::Cpu, false);
    let metal_f32 = run(Device::Metal, false);
    let metal_amp = run(Device::Metal, true);

    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb + 1e-9)
    };
    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let c_f32 = cos(&cpu, &metal_f32);
    let c_amp = cos(&cpu, &metal_amp);
    let c_pair = cos(&metal_f32, &metal_amp);
    eprintln!(
        "q1_0 AMP f16 residual: cos(cpu,metal_f32)={c_f32:.6} cos(cpu,amp)={c_amp:.6} \
         cos(f32,amp)={c_pair:.6} max_abs(f32,amp)={:.4e}",
        max_abs(&metal_f32, &metal_amp)
    );
    assert!(c_f32 > 0.999, "Metal F32 Q1 residual diverges: cos={c_f32}");
    assert!(
        c_amp > 0.99,
        "Metal AMP Q1 residual diverges vs CPU: cos={c_amp}"
    );
    assert!(
        c_pair > 0.99,
        "Metal AMP vs F32 residual stream diverges: cos={c_pair}"
    );
}

#[test]
fn q2_0_decode_gemv_matches_reference() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let k = 256usize;
    let n = 16usize;
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let mut packed = Vec::with_capacity(n * (k / 128) * 34);
    let mut reference = vec![0.0f32; n];
    for row in 0..n {
        for block in 0..(k / 128) {
            let d = 0.125 + (row * 2 + block) as f32 * 0.003;
            let d_half = half::f16::from_f32(d);
            let d_ref = d_half.to_f32();
            packed.extend_from_slice(&d_half.to_le_bytes());
            for byte in 0..32usize {
                let mut bits = 0u8;
                for lane in 0..4usize {
                    let code = ((row + block + byte + lane) & 3) as u8;
                    bits |= code << (lane * 2);
                    let j = block * 128 + byte * 4 + lane;
                    reference[row] += x[j] * (code as f32 - 1.0) * d_ref;
                }
                packed.push(bits);
            }
        }
    }
    let mut g = Graph::new("q2_0_decode_gemv");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ2_0,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Metal).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let metal = c.run(&[("x", x.as_slice())]).remove(0);
    // Only the Metal side is counted — the reference is finite constants built
    // here, so counting it would test the test (see `Finiteness`).
    let m_fin = Finiteness::count(&metal);
    assert!(
        m_fin.is_clean(),
        "Q2_0 decode: non-finite output on Metal: {m_fin:?}"
    );
    let max_abs = reference
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "Q2_0 decode Metal max_abs={max_abs}");
}

/// G8_0 (Doses AI Pestle `token_embd` / lm_head): four bf16 scales per
/// 32-element block, one per group of 8. Exercises the finer scale
/// granularity — a block-wide scale would fail this by construction,
/// since group 3 is 64× group 0.
#[test]
fn g8_0_decode_gemv_matches_reference() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    use rlx_gguf::g8_dequant::{G8_0_BLOCK_BYTES, QKG8_0};
    let k = 128usize;
    let n = 16usize;
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let nblocks = k / QKG8_0;
    let mut packed = Vec::with_capacity(n * nblocks * G8_0_BLOCK_BYTES);
    let mut reference = vec![0.0f32; n];
    for row in 0..n {
        for block in 0..nblocks {
            let mut d_ref = [0.0f32; 4];
            for (g, slot) in d_ref.iter_mut().enumerate() {
                // Spread scales widely across the four groups.
                let d = half::bf16::from_f32(0.125 * 4f32.powi(g as i32) + row as f32 * 0.01);
                packed.extend_from_slice(&d.to_bits().to_le_bytes());
                *slot = d.to_f32();
            }
            for byte in 0..(QKG8_0 / 4) {
                let mut bits = 0u8;
                for lane in 0..4usize {
                    let code = ((row + block + byte + lane) % 3) as u8;
                    bits |= code << (lane * 2);
                    let j = byte * 4 + lane;
                    reference[row] += x[block * QKG8_0 + j] * (code as f32 - 1.0) * d_ref[j / 8];
                }
                packed.push(bits);
            }
        }
    }
    // The packed bytes must decode identically on the host.
    let host = rlx_gguf::g8_dequant::dequant_g8_0(&packed, n * k).unwrap();
    for (row, want) in reference.iter().enumerate() {
        let got: f32 = (0..k).map(|j| x[j] * host[row * k + j]).sum();
        assert!(
            (got - want).abs() < 1e-4,
            "host G8_0 row {row}: {got} vs {want}"
        );
    }
    let mut g = Graph::new("g8_0_decode_gemv");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufG8_0,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Metal).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let metal = c.run(&[("x", x.as_slice())]).remove(0);
    // Only the Metal side is counted — the reference is finite constants built
    // here, so counting it would test the test (see `Finiteness`).
    let m_fin = Finiteness::count(&metal);
    assert!(
        m_fin.is_clean(),
        "G8_0 decode: non-finite output on Metal: {m_fin:?}"
    );
    let max_abs = reference
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "G8_0 decode Metal max_abs={max_abs}");
}

/// Fused `PTQ1_0` DECODE GEMV (`m == 1`) vs the CPU reference.
///
/// This kernel is the whole decode budget for `Ternary-Bonsai-2-27B` — the
/// thunk profile puts `dequant_matmul_gguf` at ~90% before it exists and ~74%
/// after — so it runs on every token of every generation. It reads the packed
/// 28-byte blocks straight out of the arena rather than expanding the row to
/// f32, and it resolves PTQ1_0's *staged* element → (byte, base-3 digit) map
/// per lane. A wrong map still produces a well-formed ternary dot product from
/// permuted weights, which is fluent and wrong rather than obviously broken.
///
/// Shapes cover the gate's two constraints (`k % 128`, `n % 8`) and the real
/// model's widths.
#[test]
fn ptq1_0_decode_gemv_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    // `RLX_METAL_PTQ1_0_NSG` is a runtime tuning knob, so every value it
    // accepts has to be correct, not just the default — a sweep that trades
    // a wrong answer for a faster one is not a sweep. The guard clears the knob
    // after each value, and on failure too, so the sweep cannot leak.
    for nsg in ["", "1", "2", "4", "8"] {
        let _knob = EnvKnob::new("RLX_METAL_PTQ1_0_NSG", (!nsg.is_empty()).then_some(nsg));
        // With no override the encoder's own default applies — unless the
        // launching environment set the knob, so name what is under test.
        let effective =
            rlx_ir::env::var("RLX_METAL_PTQ1_0_NSG").unwrap_or_else(|| "default".into());
        for (k, n) in [(128usize, 8usize), (256, 16), (512, 64), (1024, 24)] {
            // `run_case` asserts the parity itself; the label carries the shape
            // and the knob value so a failure names the cell.
            run_case(&Cell {
                scheme: QuantScheme::GgufPtq1_0,
                ggml: rlx_gguf::GgmlType::PTQ1_0,
                batch: &[],
                m: 1,
                k,
                n,
                tol: 1e-3,
                label: format!("PTQ1_0 decode GEMV k={k} n={n} NSG={effective}"),
            });
        }
    }
}

/// The fused GEMV must agree with the scratch path it replaces.
///
/// `RLX_METAL_PTQ1_0_FUSED_DISABLE=1` selects the old expand-to-f32 +
/// `sgemm` route. Both are Metal, so this isolates the new kernel from any
/// Metal-vs-CPU difference.
#[test]
fn ptq1_0_fused_gemv_matches_the_scratch_path() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    let (k, n) = (512usize, 32usize);
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::PTQ1_0).expect("quantize");
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("ptq1_gemv");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufPtq1_0,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    // A decode-shaped PTQ1_0 op on the fused route needs no f32 slab at all —
    // the zero this asserts is the other half of the flip checked under the knob.
    assert_eq!(
        rlx_metal::backend::dequant_gguf_scratch_bytes(&g),
        0,
        "the fused PTQ1_0 decode route should book no dequant slab"
    );

    let run = || -> Vec<f32> {
        let mut c = Session::new(Device::Metal).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let fused = run();
    let scratch = {
        let _knob = EnvKnob::new("RLX_METAL_PTQ1_0_FUSED_DISABLE", Some("1"));
        // The override must actually reach the encoder, or the comparison below
        // passes with both runs fused. `dequant_gguf_scratch_bytes` reads the same
        // flag the fused-GEMV gate does: zero while the fused decode route is
        // available, `k*n*4` once it is switched off.
        assert_eq!(
            rlx_metal::backend::dequant_gguf_scratch_bytes(&g),
            k * n * 4,
            "RLX_METAL_PTQ1_0_FUSED_DISABLE did not reach the encoder"
        );
        run()
    };

    // Both sides are Metal outputs, so both are counted (see `Finiteness`).
    let (fused_fin, scratch_fin) = (Finiteness::count(&fused), Finiteness::count(&scratch));
    assert!(
        fused_fin.is_clean() && scratch_fin.is_clean(),
        "fused PTQ1_0 GEMV: non-finite output — fused={fused_fin:?} scratch={scratch_fin:?}"
    );
    let worst = fused
        .iter()
        .zip(&scratch)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "fused PTQ1_0 GEMV disagrees with the scratch path by {worst}"
    );
}

/// The non-default integer PTQ1_0 GEMV must compute the same thing.
///
/// `RLX_METAL_PTQ1_0_INT_PIPE=1` selects it (see `dequant_gguf.msl`). It is
/// kept as the A/B reference for the float-pipe default, so it has to stay
/// correct even while it is not the one that runs.
#[test]
fn ptq1_0_int_pipe_gemv_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    // The flag has to be set for the whole batch (the encoder reads it at encode
    // time); the guard clears it even if a cell below fails, so it cannot leak.
    let _int_pipe = EnvKnob::new("RLX_METAL_PTQ1_0_INT_PIPE", Some("1"));
    // The label carries the shape so a failure names the cell.
    for (k, n) in [(128usize, 8usize), (512, 64), (1024, 24)] {
        run_case(&Cell {
            scheme: QuantScheme::GgufPtq1_0,
            ggml: rlx_gguf::GgmlType::PTQ1_0,
            batch: &[],
            m: 1,
            k,
            n,
            tol: 1e-3,
            label: format!("PTQ1_0 integer-pipe GEMV k={k} n={n}"),
        });
    }
}

/// Microbenchmark + `NSG` sweep for the fused `PTQ1_0` decode GEMV, at
/// Ternary Bonsai 2's real projection shapes.
///
/// Two things make a naive timing loop useless here.
///
/// **Host overhead.** One `Session::run()` costs 0.11-0.19 ms of round trip,
/// which is larger than the kernel at these shapes and varies run to run. So
/// each graph packs `FANOUT` independent GEMVs — distinct `x` inputs against
/// one shared weight, distinct so CSE cannot collapse them into one node —
/// and the overhead is amortised `FANOUT`-fold instead of subtracted.
///
/// **Machine load.** End-to-end decode swings 2x with background jobs, and so
/// does this. Worse, running one variant after another puts that drift on the
/// variable under test: sequential readings claimed NSG=4 was 1.7x faster and
/// NSG=8 2.7x slower, and interleaved both vanished. So `NSG` is a runtime
/// flag (`RLX_METAL_PTQ1_0_NSG`, packed into the kernel's `flags` word) and
/// every round times all variants back to back, keeping the per-variant min.
///
/// `cargo test -p rlx-metal --release --test metal_gguf_dequant_matmul_prefill_parity --features backend-tests \
///      ptq1_0_gemv_bandwidth -- --ignored --nocapture`
#[test]
#[ignore = "benchmark, not a correctness check"]
fn ptq1_0_gemv_bandwidth() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();
    const NSGS: [u64; 3] = [2, 4, 8];
    const ROUNDS: usize = 5;
    const ITERS: usize = 8;
    const FANOUT: usize = 16;

    let shapes: [(usize, usize, &str); 4] = [
        (5120, 10240, "attn_qkv"),
        (6144, 5120, "ssm_out"),
        (5120, 17408, "ffn_up"),
        (17408, 5120, "ffn_down"),
    ];

    let mut best = vec![vec![f64::INFINITY; NSGS.len()]; shapes.len()];
    let mut sessions = Vec::new();
    let mut feeds_owned: Vec<Vec<(String, Vec<f32>)>> = Vec::new();
    let mut nbytes = Vec::new();

    for (k, n, _) in shapes {
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::PTQ1_0).expect("quantize");
        let mut g = Graph::new("ptq1_bench");
        let wp = g.param("w", Shape::new(&[packed.len()], DType::U8));
        let mut outs = Vec::with_capacity(FANOUT);
        let mut fx = Vec::with_capacity(FANOUT);
        for f in 0..FANOUT {
            let name = format!("x{f}");
            let x_in = g.input(&name, Shape::new(&[1, k], DType::F32));
            outs.push(g.add_node(
                Op::DequantMatMul {
                    scheme: QuantScheme::GgufPtq1_0,
                },
                vec![x_in, wp],
                Shape::new(&[1, n], DType::F32),
            ));
            fx.push((
                name,
                (0..k)
                    .map(|i| ((i + f) as f32 * 0.017).sin())
                    .collect::<Vec<f32>>(),
            ));
        }
        g.set_outputs(outs);
        let mut c = Session::new(Device::Metal).compile(g);
        c.set_param_typed("w", &packed, DType::U8);
        nbytes.push(packed.len() as f64 * FANOUT as f64);
        sessions.push(c);
        feeds_owned.push(fx);
    }

    macro_rules! run {
        ($c:expr, $fx:expr) => {{
            let f: Vec<(&str, &[f32])> = $fx
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let _ = $c.run(&f);
        }};
    }
    for (si, c) in sessions.iter_mut().enumerate() {
        for _ in 0..2 {
            run!(c, feeds_owned[si]);
        }
    }
    for _ in 0..ROUNDS {
        for (ni, nsg) in NSGS.iter().enumerate() {
            let nsg_text = nsg.to_string();
            let _nsg = EnvKnob::new("RLX_METAL_PTQ1_0_NSG", Some(nsg_text.as_str()));
            for (si, c) in sessions.iter_mut().enumerate() {
                let t = std::time::Instant::now();
                for _ in 0..ITERS {
                    run!(c, feeds_owned[si]);
                }
                let dt = t.elapsed().as_secs_f64() / ITERS as f64;
                if dt < best[si][ni] {
                    best[si][ni] = dt;
                }
            }
        }
    }

    eprintln!(
        "  effective weight GB/s, {FANOUT} GEMVs per run, min of {ROUNDS} interleaved rounds"
    );
    eprint!("  {:<10}", "shape");
    for nsg in NSGS {
        eprint!("   NSG={nsg:<8}");
    }
    eprintln!();
    for (si, (k, n, label)) in shapes.iter().enumerate() {
        eprint!("  {label:<10}");
        for ni in 0..NSGS.len() {
            eprint!("  {:>8.1}   ", nbytes[si] / best[si][ni] / 1e9);
        }
        eprintln!("  (k={k}, n={n})");
    }
}

/// Byte accounting for the bounded model-class graph, accumulated as it is
/// built. This is the engine's own accounting, not RSS: the repo ships no
/// OS-RSS utility, and the arena is a single `MTLBuffer` spanning weights,
/// activations and scratch, so process RSS cannot attribute the packed bytes to
/// this test.
#[derive(Default)]
struct MemAcc {
    /// Sum of the packed param byte lengths uploaded via `set_param_typed(U8)`.
    packed_resident: usize,
    /// Sum of `k*n*4` over the block's weights — the f32 image an unpack-at-load
    /// workaround would hold for the whole graph. The build materializes one
    /// weight at a time, so its own host peak is the widest projection
    /// (`4096*1024*4` = 16 MiB), not this sum.
    f32_image_total: usize,
    /// Graph-wide dequant slab the Metal planner reserves for the block, taken
    /// from `rlx_metal::backend::dequant_gguf_scratch_bytes`.
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
    eprintln!(
        "[rlx-gguf-mem] backend=metal packed_resident={} B ({:.2} MiB) f32_image_total={} B \
         ({:.2} MiB) in_arena_dequant_scratch={} B ({:.2} MiB) scratch_vs_adapter={} \
         basis=engine param bytes (no OS-RSS utility in-repo) arena=single MTLBuffer \
         (weights+activations+scratch)",
        mem.packed_resident,
        mib(mem.packed_resident),
        mem.f32_image_total,
        mib(mem.f32_image_total),
        mem.scratch_graph_max,
        mib(mem.scratch_graph_max),
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
/// Q4_K o-projection folded into the residual). The RmsNorm lowers into an
/// MPSGraph segment and every packed matmul is a thunk boundary, which is the
/// hybrid MPSGraph/thunk schedule a bare matmul chain never reaches. Every
/// intermediate is `batch ++ [m, d]` so a `[1]` batch builds the models'
/// `[1, 1, h]` single-token decode. `cache` carries the quantized weights across
/// the caller's `m` loop.
fn build_mlp_block(
    batch: &[usize],
    m: usize,
    h: usize,
    i: usize,
    layers: usize,
    cache: &mut WeightCache,
) -> MlpBlock {
    let mut g = Graph::new("gguf_mlp_block");
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
    // The slab the planner books for this block, asked of the engine rather than
    // summed here, so the memory line and the cells' routes agree by construction.
    mem.scratch_graph_max = rlx_metal::backend::dequant_gguf_scratch_bytes(&g);
    MlpBlock {
        graph: g,
        params,
        mem,
        x,
        // `x` is one element per `[m, h]` slice; a `[1]` batch multiplies the output.
        out_elems: batch.iter().product::<usize>() * m * h,
    }
}

/// The scheme × shape grid and the model-class graph, one `#[test]` for the whole
/// capture. What serializes it is [`DEVICE_LOCK`] (see the chain test above), not
/// the grouping: the two parts share one thread because they are one capture.
#[test]
fn gguf_dequant_prefill_route_grid_and_model_matches_cpu() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _serial = device_lock();

    // Part (a): the scheme × shape grid, the four packed families the Metal
    // encoder routes by. The shapes straddle Metal's own change point, which is
    // env-controlled, so `metal_route` reads the EFFECTIVE cap
    // (`RLX_METAL_Q4K_GEMM_MAX_M`, else 96/32) and re-labels the cells if it
    // changes. At the default cap=96 the sides are:
    //   fused XS GEMM — m=2, 64, 96   (1 < m <= cap, k%256==0)
    //   scratch+MPS   — m=128, 192    (m > cap)
    //   fused GEMV    — m=1 2-D and 3-D decode (k%256==0 for the K-quants,
    //                   k%32==0 for Q8_0)
    //
    // The straddle is the grid's whole point and the cap is a knob, so assert the
    // two sides from the cells' own reported routes below rather than assume them:
    // an exported `RLX_METAL_Q4K_GEMM_MAX_M` at or above 128 (or an XS disable,
    // which drops the cap to 32) would move every cell onto one side and leave the
    // test green with its property gone. The GPU dequant path also has to be on,
    // or every cell would be the host fallback and measure nothing on the device.
    assert!(
        !rlx_metal::runtime_config().dequant_gpu_disable,
        "RLX_METAL_DEQUANT_GPU_DISABLE (or its registered deprecated alias \
         RLX_DISABLE_METAL_DEQUANT_GPU) is set, so every cell would take the host fallback \
         and this grid could not measure an on-device route"
    );
    // The fused Q4_K GEMM's own off-switches would take the m=96 cell off the fused
    // side for a reason that is not the cap, so name them here rather than let the
    // cap assertion below blame RLX_METAL_Q4K_GEMM_MAX_M for one of them.
    assert!(
        !rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_DISABLE"),
        "RLX_METAL_Q4K_GEMM_DISABLE is set, so the grid's Q4_K cells fall to the scratch \
         path whatever RLX_METAL_Q4K_GEMM_MAX_M says"
    );
    // The XS switch is the other knob that moves the cap (96 → 32 in
    // `q4k_gemm_max_m`), so a failure under it would otherwise read as "the cap
    // moved" rather than as the switch it is.
    assert!(
        !rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_XS_DISABLE"),
        "RLX_METAL_Q4K_GEMM_XS_DISABLE is set, so the cap is 32 and the grid's m=96 cell \
         falls on the scratch side whatever RLX_METAL_Q4K_GEMM_MAX_M says"
    );
    let schemes: &[(QuantScheme, rlx_gguf::GgmlType, &str)] = &[
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, "Q4_K"),
        (QuantScheme::GgufQ5K, rlx_gguf::GgmlType::Q5K, "Q5_K"),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, "Q6_K"),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, "Q8_0"),
    ];
    // (batch, m, k, n, tol), chosen for these cells rather than inherited: 1e-3 at
    // k=256 (worst observed cell 8.0e-5) and 2e-3 once k is in the low thousands
    // (worst observed 3.1e-4), where f32 accumulation rather than the route sets
    // the floor. Each bound sits 6x or more above the worst cell it applies to, and
    // both reject a zeroed or grossly wrong projection by orders of magnitude.
    // Loosen only with a reason.
    let shapes: &[(&'static [usize], usize, usize, usize, f32)] = &[
        (&[], 2, 256, 8, 1e-3), // smallest prefill (m>1), single superblock, not covered elsewhere
        (&[], 96, 256, 640, 1e-3), // at the cap (fused)
        (&[], 128, 256, 640, 1e-3), // first rows past the cap (scratch)
        (&[], 64, 2048, 512, 2e-3), // key width 2048, below the cap
        (&[], 192, 1024, 2048, 2e-3), // hundreds of rows + low-thousands width, past the cap
        (&[], 1, 256, 640, 1e-3), // 2-D decode
        (&[1], 1, 2048, 640, 2e-3), // 3-D single-token decode: [1,1,k] -> [1,1,n]
    ];
    // The Q4_K k=256 n=640 pair that straddles the cap, kept as the cells route it.
    let (mut at_cap, mut past_cap) = (None, None);
    for &(scheme, ggml, sname) in schemes {
        for &(batch, m, k, n, tol) in shapes {
            let report = run_case(&Cell {
                scheme,
                ggml,
                batch,
                m,
                k,
                n,
                tol,
                label: format!("grid {sname} m={m} k={k} n={n} batch={batch:?}"),
            });
            if scheme == QuantScheme::GgufQ4K && k == 256 && n == 640 {
                match m {
                    96 => at_cap = Some(report.route),
                    128 => past_cap = Some(report.route),
                    _ => {}
                }
            }
        }
    }
    let cap = q4k_gemm_max_m();
    let at_cap = at_cap.expect("the grid has a Q4_K k=256 n=640 cell at m=96");
    let past_cap = past_cap.expect("the grid has a Q4_K k=256 n=640 cell at m=128");
    assert!(
        at_cap.contains("fused GEMM"),
        "the m=96 cell must be on the fused side of RLX_METAL_Q4K_GEMM_MAX_M={cap}: {at_cap:?}"
    );
    assert!(
        past_cap == SCRATCH_ROUTE,
        "the m=128 cell must be past RLX_METAL_Q4K_GEMM_MAX_M={cap}: {past_cap:?}"
    );
    eprintln!(
        "[rlx-gguf-route] backend=metal change_point=RLX_METAL_Q4K_GEMM_MAX_M cap={cap} \
         m=96=\"{at_cap}\" m=128=\"{past_cap}\""
    );

    // Part (b): a bounded model-class graph at whole-model scale. The bound is
    // a design bound, not a measurement: 4 decoder-style blocks of packed
    // projections (~55M packed elements → ~45 MiB packed, and ~208 MiB of f32 if the
    // whole block were expanded at once — the build materializes one weight at a
    // time, so its own peak is the widest projection, 16 MiB), the 16 MiB dequant
    // slab the planner books for that projection (`4096*1024*4`) shared by all 16
    // packed matmuls, and three cells: m=1 in the models' 3-D single-token form
    // `[1, 1, H]`, m=96 and m=192 at `[m, H]` — one bounded test, weights built in
    // memory with `rlx_gguf::quantize` (no file). The measured numbers print once.
    const H: usize = 1024;
    const I: usize = 4096;
    const LAYERS: usize = 4;
    let mut mem_printed = false;
    // The block's weights are m-independent, so one cache spans the loop: each
    // weight is quantized once rather than once per m.
    let mut weight_cache = WeightCache::new();
    for &(batch, m) in &[
        (&[1usize][..], 1usize),
        (&[][..], 96usize),
        (&[][..], 192usize),
    ] {
        let block = build_mlp_block(batch, m, H, I, LAYERS, &mut weight_cache);
        let label = format!("model mlp-block {LAYERS}L batch={batch:?} m={m} H={H} I={I}");
        // The block's four packed projections, in the shape each is built at:
        // gate Q5_K [n=I, k=H], up Q6_K [n=I, k=H], down Q8_0 [n=H, k=I] and the
        // o-projection Q4_K [n=H, k=H]. Each is routed on its own `(k, n)`, so
        // they are derived per op rather than from the block's H/I pair — and the
        // slab the routes fall back to is the one the planner booked for THIS
        // graph (`block.mem.scratch_graph_max`), not a restatement of it.
        let scratch_reserved = block.mem.scratch_graph_max > 0;
        let mut routes: Vec<String> = [
            (QuantScheme::GgufQ5K, H, I),
            (QuantScheme::GgufQ6K, H, I),
            (QuantScheme::GgufQ8_0, I, H),
            (QuantScheme::GgufQ4K, H, H),
        ]
        .into_iter()
        .map(|(s, k, n)| metal_route(s, m, k, n, scratch_reserved))
        .collect();
        routes.sort();
        routes.dedup();

        let measured = measure(
            &label,
            &block.graph,
            &block.params,
            &block.x,
            block.out_elems,
        );
        // Parity tolerance. The block's residual stream reaches |y| ~ 2.9e3
        // (two un-normalized MLP expansions, no downstream norm), so the absolute
        // 2e-3 that holds for the grid cells at |y| ~ 20 is below the f32
        // accumulation floor here. Observed max_abs / cpu_max (relative):
        //   m=1 (3-D) 1.95e-3 / 1617 = 1.2e-6
        //   m=96      2.67e-2 / 2939 = 9.1e-6
        //   m=192     1.73e-2 / 2940 = 5.9e-6
        // One absolute bound covers every m — the absolute value tracks cpu_max,
        // not the row count or the route — with headroom over the largest observed
        // side, and paired with a relative bound asserted below: what actually
        // bounds the fused GEMV/GEMM accumulation is the relative error, not an
        // absolute one.
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
        // at ~1e-6..9e-6 relative, and 1e-4 is the gross-divergence ceiling.
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
