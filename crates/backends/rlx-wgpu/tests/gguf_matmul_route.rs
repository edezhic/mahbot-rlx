// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which route a packed GGUF `DequantMatMul` took is readable from the backend
//! (`gguf_gpu::route_counts`), and a weight window wider than the binding limit
//! must be N-chunked rather than pushed onto the per-row path. The one-row route
//! is a single route with a shape of its own, which is readable the same way
//! (`gguf_gpu::gemv_one_row_shape`).
//!
//! The same counters pin Q8_0's change point: the byte-per-weight form's one row
//! is the tiled kernel's own block-keyed branch, and above one row, over whole
//! 32-element packing blocks of its own, it takes the row-grouped GEMM instead of
//! the slab — that arm's weight loop reads the scheme's blocks, and the 256-element
//! tile it stages is eight of them, so a row may end inside a tile. The A/B block
//! below measures such a call against the slab it used to take, at the shapes the
//! packed steps actually multiply, and *reports* the figures (its
//! `[rlx-gguf-q8-0-gemm]` lines) — the asserts there are the route, the recorded
//! route census, the step count and the values, not the timings.
//!
//! Its own test binary, and the one test fn that reads the process-global counters:
//! the other one, the device-free pin on the grid's Q8_0 expectation, reads none and
//! dispatches nothing, so those counters have no concurrent writers.
//!
//! Everything here is an IN-ARENA weight: the executable only moves a param into
//! the separate packed-weight buffer when the *plan* exceeds the device binding
//! limit (4 GiB of packed weights on this hardware), which no small test reaches —
//! `RLX_WGPU_ALL_PARAMS_WEIGHT=1` would force the split. This file does not set that
//! knob, and it does not guard it either: run the binary with the knob set and the
//! grid's non-Q8_0 cells (the Q5_K shapes among them) mis-route. That sensitivity is
//! pre-existing and no claim here rests on those cells' numbers under that knob;
//! `gguf_gemm_parked_weight.rs` is the file that drives a split arena directly.
//! The `RLX_WGPU_MAX_BIND_MB` cases below drive the window/chunk arithmetic: a
//! weight *window* over the binding limit must be N-chunked, and no chunk window
//! may be sized from the weight's position relative to offset 0.
//!
//! The knob is written through `rlx_ir::env` — the code-override layer the encoder
//! reads (`env::flag` / `parse_or`) and which outranks the process environment —
//! never through `std::env`, which is `unsafe` in edition 2024 and process-global.
//! Each bracket unsets it before its assertions, so a failing case leaves no cap
//! pinned behind it, and the counter read right after the run is what proves the
//! override reached the encoder at all: an A/B that changes nothing proves nothing.

mod common;
use common::{max_abs_diff, refuse_under_q8_0_bisect, refuse_under_run_report};
use rlx_compile::memory::plan_memory_f32_uniform;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::PassMarks;
use rlx_wgpu::buffer::Arena;
use rlx_wgpu::gguf_gpu::{
    DequantMatmulPath, DequantMatmulRoutes, dequant_gguf_scratch_bytes, gemv_one_row_shape,
    reset_gemv_one_row_shape, reset_route_counts, route_counts, run_dequant_matmul_gguf_gpu,
    select_path,
};
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts, wgpu_device};

/// Whether a Q8_0 call takes the scheme's one fused arm — the row-grouped GEMM —
/// rather than the dequant slab: more than one row, and a `k` that is a whole
/// number of this scheme's own 32-element packing blocks
/// (`QuantScheme::GgufQ8_0.gguf_block_size()`), which is the granularity that arm's
/// weight loop reads. Not the 256-element staged tile the arm shares with the
/// K-quant entries, which is eight of those blocks: a whole number of blocks can
/// end inside a tile. Both terms are exercised by the shape grid below — its k=384
/// cell is 12 whole blocks and 1.5 tiles, so it is the off-tile cell that pins the
/// block term rather than the tile, which is what keeps this falsifiable there —
/// while the A/B block's grid states the literal it is left with, its 128 / 288 /
/// 896 cells the class's own boundaries.
fn q8_0_byte_direct(m: usize, k: usize) -> bool {
    m > 1 && k.is_multiple_of(QuantScheme::GgufQ8_0.gguf_block_size() as usize)
}

/// Whether a Q8_0 call takes this scheme's own one-row arm — the tiled kernel's
/// 8-bit branch — rather than the slab: exactly one row, over whole 32-element
/// blocks (this scheme's own block, not the grouped arm's 256-element staged
/// tile).
fn q8_0_one_row(m: usize, k: usize) -> bool {
    m == 1 && k.is_multiple_of(QuantScheme::GgufQ8_0.gguf_block_size() as usize)
}

/// The shape grid's Q8_0 expectation is [`q8_0_byte_direct`] and
/// [`q8_0_one_row`], a restatement of the ladder's own shape terms — so it is
/// checked against the ladder here: a term the ladder gains or drops without a
/// matching line in those helpers fails this, instead of leaving the grid's
/// expectation to agree silently. Over every `m` this file's two Q8_0 grids drive
/// (the shape grid's 1…192 and the A/B block's 1…1024) and every `k` they drive,
/// plus the `k` below one staged tile that the negative needs — 100 is not a whole
/// number of this scheme's 32-element blocks, which is the grouped arm's `k` term
/// and the one-row arm's, so a packed weight cannot carry it and only a device-free
/// pin can exercise it.
///
/// Device-free — `select_path` reads no device — and the two constants are premises,
/// not the grid's own values: 64 is a 16-aligned X window (what this scheme's grouped
/// arm asks of it) and 4096 stands for the slab the plan books, which every grid cell
/// has on-arena.
#[test]
fn the_q8_0_shape_expectation_agrees_with_the_ladder() {
    let scheme = rlx_wgpu::gguf_host::gguf_scheme_id(QuantScheme::GgufQ8_0);
    for m in [
        1usize, 2, 4, 8, 13, 17, 64, 72, 96, 128, 192, 276, 512, 1024,
    ] {
        for k in [
            100usize, 128, 256, 288, 384, 512, 896, 1024, 2048, 3072, 4096,
        ] {
            // All three of the scheme's answers, one row included: a ladder that
            // gains a one-row term with no matching line in `q8_0_one_row` fails on
            // the grid's `m == 1` cells. That term is `m == 1` exactly, over this
            // scheme's own 32-element blocks rather than the staged tile — so its
            // `k = 128` and `k = 384` cells are that arm's own shapes, off the tile,
            // and only a row count other than one takes them off that arm. Above one
            // row those two are the grouped arm's, whose block term they satisfy
            // too; `k = 100` is the one contraction neither arm can read.
            let want = if q8_0_byte_direct(m, k) {
                DequantMatmulPath::GroupedDevice
            } else if q8_0_one_row(m, k) {
                DequantMatmulPath::OneRowDevice
            } else {
                DequantMatmulPath::ScratchArm
            };
            assert_eq!(
                select_path(m, k, scheme, 64, 4096),
                want,
                "Q8_0 m={m} k={k}: the grid's expectation and the ladder's own answer \
                 must agree"
            );
        }
    }
}

/// Packed `w` and its row-major `x` for one `DequantMatMul`.
fn case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    case_batched(scheme, ggml, &[], m, k, n)
}

/// As [`case`], with leading batch dims: the 3-D single-token decode shape the
/// models use has to reach the same route counters as the 2-D one.
fn case_batched(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    batch: &[usize],
    m: usize,
    k: usize,
    n: usize,
) -> (Graph, Vec<u8>, Vec<f32>) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut in_shape = batch.to_vec();
    in_shape.extend_from_slice(&[m, k]);
    let mut out_shape = batch.to_vec();
    out_shape.extend_from_slice(&[m, n]);

    let mut g = Graph::new("wgpu_gguf_route");
    let x_in = g.input("x", Shape::new(&in_shape, DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&out_shape, DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, packed, x)
}

fn run(device: Device, g: Graph, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

/// Brackets one call — reset, run, snapshot — so the route is this call's own.
fn route_of(scheme: QuantScheme, ggml: rlx_gguf::GgmlType, m: usize) -> DequantMatmulRoutes {
    let (g, packed, x) = case(scheme, ggml, m, 256, 64);
    reset_route_counts();
    let _ = run(Device::Gpu, g, &packed, &x);
    route_counts()
}

/// One row of a Q8_0 call driven straight from this shape's plan, through
/// `run_dequant_matmul_gguf_gpu` — the entry point this scheme's one-row calls
/// took before the tiled arm's own 8-bit branch, i.e. the slab arrangement the
/// A/B's lever cannot reach at `m == 1` (it lowers the row-grouped encoder alone).
/// Returns the leg's output, its own route census, and the two device counters
/// over the leg *including its readback* — the A side of the bracket is a graph
/// run, whose readback is inside the window its counters are read over, so a leg
/// that left its own out would compare two different things. One ranged D2H read
/// each, so an adapter that reads back in a submission of its own adds one to both
/// sides and the difference between them stays the two arms' own work.
fn one_row_slab_leg(
    dev: &rlx_wgpu::device::WgpuDevice,
    g: &Graph,
    k: usize,
    n: usize,
    packed: &[u8],
    x: &[f32],
) -> (Vec<f32>, DequantMatmulRoutes, u64, u64) {
    // The caller's graph — the fixture this test already builds for the shape — so
    // the leg's arena is that shape's own plan and its three points are the ids
    // the same graph hands out, not a restatement of them.
    let (x_in, w, y) = (
        g.input_id("x").expect("x input"),
        g.param_id("w").expect("w param"),
        g.outputs[0],
    );

    // The booking is the plan's own, exactly as the compiled graph's arena has it:
    // this scheme's op is one the scratch skip does not cover.
    let plan = plan_memory_f32_uniform(g, 16);
    let arena = Arena::from_plan_with_scratch(&dev.device, &plan, dequant_gguf_scratch_bytes(g));
    arena.write_bytes_range(&dev.queue, arena.offset(w), packed);
    arena.write_f32(&dev.queue, x_in, x);
    let (x_off, w_off, y_off) = (arena.offset(x_in), arena.offset(w), arena.offset(y));

    reset_route_counts();
    reset_device_counts();
    run_dequant_matmul_gguf_gpu(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        k,
        n,
        rlx_wgpu::gguf_host::gguf_scheme_id(QuantScheme::GgufQ8_0),
        x_off,
        w_off,
        arena.scratch_off,
        y_off,
        &mut PassMarks::none(),
    );
    let routes = route_counts();
    let bytes = arena.read_bytes_range(&dev.device, &dev.queue, y_off, n * 4);
    let (submits, waits) = (device_submit_count(), device_wait_count());
    let y_row: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (y_row, routes, submits, waits)
}

/// One test fn on purpose — see the module doc.
#[test]
fn gguf_matmul_routes_are_reported_and_weight_windows_are_chunked() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }

    refuse_under_run_report();

    // ... and out of the 8-bit scheme's bisect mode: this file's own A/B turns that
    // lever on and off inside one bracket, and the bars either side of it are the
    // shipped mode's.
    refuse_under_q8_0_bisect();

    // The one mode this binary's submission bars support, read once: the
    // step-by-step lever (`RLX_WGPU_ONE_OP_PER_PASS`) makes the batch join inert,
    // so a graph's steps submit one at a time instead of riding one submission.
    // Read here so every bar below states the floor its own arrangement has.
    let step_by_step = rlx_ir::env::flag("RLX_WGPU_ONE_OP_PER_PASS");

    // m == 1 keeps today's single-row path.
    let one = route_of(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1);
    assert_eq!(
        (one.per_row, one.grouped),
        (1, 0),
        "m=1 must stay on the per-row GEMV: {one:?}"
    );

    // Any m > 1 — including 2 and 3, the lowest sizes that must be served — takes
    // the grouped path, up to the ~96 rows the model bodies use and the ~276 of the
    // widest matrices.
    for m in [2usize, 3, 4, 5, 6, 7, 8, 9, 96, 276] {
        let c = route_of(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, m);
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "Q4_K m={m} must be one grouped dispatch: {c:?}"
        );
    }
    for m in [2usize, 4, 96] {
        let c = route_of(QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, m);
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "Q6_K m={m} must be one grouped dispatch: {c:?}"
        );
    }
    // The one-bit scheme keeps its own multi-row GEMM.
    let q1 = route_of(QuantScheme::GgufQ1_0, rlx_gguf::GgmlType::Q1_0, 4);
    assert_eq!(
        (q1.q1_0_gemm, q1.grouped),
        (1, 0),
        "Q1_0 must keep its tiled GEMM: {q1:?}"
    );
    // A scheme with no fused kernel still takes the host / scratch route — the
    // case the compile-time report shows as native because the op is claimed.
    let q5 = route_of(QuantScheme::GgufQ5_0, rlx_gguf::GgmlType::Q5_0, 4);
    assert_eq!(q5.host + q5.scratch, 1, "Q5_0 route: {q5:?}");

    // Q4_K parity at a second shape (k=256, n=640), m=1 against m=96, against the
    // CPU reference. The routes for these two cells are asserted by the shape grid
    // below, alongside every other shape; this loop is what pins their parity.
    for m in [1usize, 96] {
        let scheme = QuantScheme::GgufQ4K;
        let (g, packed, x) = case(scheme, rlx_gguf::GgmlType::Q4K, m, 256, 640);
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let cpu = run(Device::Cpu, g, &packed, &x);
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!("{scheme:?} m={m} k=256 n=640: max_abs={max_abs:.6e}");
        assert!(
            max_abs <= 1e-2,
            "{scheme:?} m={m}: max_abs {max_abs} > 1e-2"
        );
    }

    // The in-scope schemes at every shape the parity suite's grid uses, printed
    // from the process-global counters this binary owns. That file reports a route
    // DERIVED from the selector; these lines are the backend's own census for the
    // same shapes, so neither the small-grid shapes nor the 3-D single-token decode
    // rests on the derivation alone.
    //
    // Q5_K has NO fused route, so those calls take dequant-scratch + matmul_bt:
    // one route, no change point for it to measure, and `host == 0` is what an
    // in-arena weight produces (the host route is reserved for a sharded arena, a
    // parked weight, or a refused scratch, and — the case the k=384 cell below is
    // here to reach — a fused family whose k is not a whole number of blocks, which
    // books no slab for the selector to fall back into). Q8_0 changes at `m == 1`
    // too: one row of the tiled kernel's 8-bit branch, over this scheme's own
    // 32-element blocks, below; the row-grouped GEMM above, reading the packed
    // weight in place at any `k` that is a whole number of those same blocks — so
    // the 256-element staged tile that arm stages, eight blocks, is not what the
    // split turns on: the k=384 cell is 12 whole blocks and 1.5 tiles and is grouped
    // while the K-quants' cells at that same k fall to the host. That term's
    // negative is the `k = 100` cell of the device-free ladder pin above; no cell of
    // this grid is off it, since no packed weight carries a partial block. Q4_K /
    // Q6_K change at `m == 1` the same way: one per-row GEMV below, one row-grouped
    // dispatch above.
    //
    // The model-class graph's own widths (H=1024, I=4096) are counter-checked here
    // in both regimes rather than only derived: gate/up is k=1024→n=4096, down is
    // k=4096→n=1024, and the o-projection is k=1024→n=1024. Each appears in the
    // decode regime (the 2-D m=1 form and the 3-D single-token `[1,1,k]` form) and
    // in the prefill regime where the model graph uses it (gate/up and o at m>1,
    // down at large m).
    let shapes: &[(&'static [usize], usize, usize, usize)] = &[
        (&[], 2, 256, 8),
        // k=384 is 1.5 Q4_K/Q6_K superblocks and 12 whole Q8_0 blocks: the cell
        // that reaches the selector's host fallback for the K-quants and Q8_0's
        // grouped arm — 1.5 staged tiles, so off the tile, which pins that the arm's
        // `k` term is the scheme's own block and not the tile, and makes the pair of
        // cells the sharpest statement of the two answers at one k. Its m = 1 cell
        // beside it is the same shape on the one-row arm — Q8_0's own branch over
        // those 12 whole blocks, while the K-quants' one-row term inherits none of
        // their block gate. Q5_K is non-fused at any k, so it still takes its single
        // scratch route here.
        (&[], 4, 384, 8),
        (&[], 1, 384, 8),
        // The rest of Q8_0's grouped class, at the boundaries of that block term:
        // k=128 is half a staged tile — a row ending inside one, the sub-tile end the
        // tail tile's own bound exists for — k=288 is a tile plus one block, and
        // k=896 is three tiles plus four blocks, the width the A/B stand below
        // multiplies at. Each is a whole number of the scheme's 32-element blocks, so
        // each is admitted, and the grid's CPU parity covers them as it does k=384.
        // The other schemes read the same cells as they read k=384: none of the three
        // is a whole superblock, so the K-quants keep their host fallback there.
        //
        // n=16 rather than those cells' 8: the K-quants share every cell of this
        // grid, and k=288 of a Q6_K weight at n=8 is 9 blocks — 1890 B, which wgpu's
        // four-byte-aligned `write_buffer` refuses at the upload, before any route is
        // taken.
        (&[], 4, 128, 16),
        (&[], 4, 288, 16),
        (&[], 4, 896, 16),
        (&[], 96, 256, 640),
        (&[], 128, 256, 640),
        (&[], 64, 2048, 512),
        (&[], 192, 1024, 2048),
        (&[], 1, 256, 640),
        (&[1], 1, 2048, 640),
        (&[], 192, 4096, 1024),
        (&[], 1, 1024, 4096),
        (&[], 1, 4096, 1024),
        (&[], 96, 1024, 4096),
        (&[], 1, 1024, 1024),
        (&[], 96, 1024, 1024),
        (&[1], 1, 1024, 4096),
        (&[1], 1, 4096, 1024),
        (&[1], 1, 1024, 1024),
    ];
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ5K, rlx_gguf::GgmlType::Q5K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0),
    ] {
        for &(batch, m, k, n) in shapes {
            let (g, packed, x) = case_batched(scheme, ggml, batch, m, k, n);
            reset_route_counts();
            let gpu = run(Device::Gpu, g.clone(), &packed, &x);
            let c = route_counts();
            // `Debug` is the backend's own census — all five counters, rather than
            // one name picked for them.
            eprintln!(
                "[rlx-gguf-route] scheme={scheme:?} m={m} k={k} n={n} batch={batch:?} route={c:?}"
            );
            let fused = matches!(scheme, QuantScheme::GgufQ4K | QuantScheme::GgufQ6K);
            let aligned = k.is_multiple_of(scheme.gguf_block_size() as usize);
            // Q8_0's one-row arm is the tiled kernel's 8-bit branch, over this
            // scheme's own 32-element blocks; its multi-row one is the GEMM, whose
            // gate is those same blocks rather than the 256-element staged tile it
            // stages eight of. Off either of them the slab is all there is.
            let byte_direct = scheme == QuantScheme::GgufQ8_0 && q8_0_byte_direct(m, k);
            let one_row = scheme == QuantScheme::GgufQ8_0 && q8_0_one_row(m, k);
            // The one-row term: `fused` is where a K-quant's `m == 1` lands, and
            // Q8_0's is the scheme term above, whose block is its own 32 elements.
            let one_row_arm = m == 1 && (fused || one_row);
            let want = match (fused, byte_direct, one_row_arm, aligned) {
                (_, true, ..) => (1, 0, 0, 0), // Q8_0 m>1, whole-block k: the grouped GEMM
                (_, _, true, _) => (0, 1, 0, 0), // the tiled arm at one row: the K-quants, and Q8_0's own branch
                (false, false, ..) => (0, 0, 1, 0), // Q5_K / Q8_0 off its arms: the slab
                // Decode takes one row per call so long as k is a multiple of
                // four (the tiled arm's `vec4` X read); the block gate is the
                // m > 1 GEMM's.
                (true, _, false, true) => (1, 0, 0, 0),
                (true, _, false, false) => (0, 0, 0, 1), // host fallback: no slab booked, no row slicing
            };
            assert_eq!(
                (c.grouped, c.per_row, c.scratch, c.host),
                want,
                "{scheme:?} m={m} k={k} n={n} batch={batch:?}: {c:?}"
            );
            // The host fallback's numbers are checked here as well as its counter:
            // the row slices are invalid and a broken staging would show as a wrong
            // or non-finite value, not as a wrong counter. So are the cells that
            // left the slab for Q8_0's arms — the grouped GEMM's, whose first
            // evidence against the CPU reference is the grid's own shapes rather
            // than only the packed steps' the throughput block below uses, and the
            // one-row branch's, whose block stride a counter cannot see either.
            if want == (0, 0, 0, 1) || byte_direct || one_row {
                let cpu = run(Device::Cpu, g, &packed, &x);
                let max_abs = max_abs_diff(&gpu, &cpu);
                eprintln!(
                    "{}{scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e}",
                    if byte_direct {
                        "byte-direct "
                    } else if one_row {
                        "one-row "
                    } else {
                        "host-fallback "
                    }
                );
                assert!(
                    max_abs <= 1e-2,
                    "{scheme:?} m={m} k={k} n={n}: max_abs {max_abs} > 1e-2"
                );
            }
        }
    }

    // A weight window wider than the binding limit must be N-chunked (the
    // regime a model bigger than the window lands in), not pushed onto the
    // per-row path — and chunking must not move the result against the CPU
    // reference. 1 MiB windows force several chunks; m=96 (12 exact row tiles,
    // so only the last column chunk is partial) also makes the output slab
    // (`by_out`) rather than the weight the binding constraint, which is the
    // widest chunk count the encoder can be made to produce here. Q6_K is a
    // different packed layout (210-byte blocks, quarter-sized sub-blocks) with
    // its own chunk arithmetic, and Q8_0 a third (34-byte blocks, no word
    // alignment to read them through) — both ride the same loop, which is what
    // these cells are here to show. Q8_0's second cell is the same loop at a
    // contraction OFF the staged tile — 288 elements is nine of its 34-byte blocks,
    // a row stride of 306 bytes, not a whole word at every column boundary — which
    // is the class of shape this file's admission change newly hands to that loop.
    let chunked: &[(QuantScheme, rlx_gguf::GgmlType, usize, usize)] = &[
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 4, 256),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, 4, 256),
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 96, 256),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, 4, 256),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, 4, 288),
    ];
    for &(scheme, ggml, m, k) in chunked {
        let (g, packed, x) = case(scheme, ggml, m, k, 16384);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let c = route_counts();
        let cpu = run(Device::Cpu, g, &packed, &x);
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "{scheme:?} m={m} k={k}: a weight window over the binding limit must still \
             be grouped: {c:?}"
        );
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!("chunked-window {scheme:?} m={m} k={k} n=16384: max_abs={max_abs:.6e}");
        assert!(
            max_abs <= 1e-2,
            "chunked-window {scheme:?} m={m} k={k}: max_abs {max_abs} > 1e-2"
        );
    }

    // Accuracy against *today's* path — the same rows computed one at a time, i.e.
    // the m=1 GEMV — under the same chunked-window regime and at the same 1e-2
    // the CPU-parity checks above use. A wrong chunk stride or row tile is O(1)
    // here rather than round-off, which is what this pins. The packed bytes are a
    // function of (scheme, k, n), so the single-row graphs reuse this weight; the
    // ignored `gguf_gemm_grouped_weight_reuse.rs` repeats this at the widest served
    // width, too big to build in the default run.
    {
        let (k, n, m) = (256usize, 16384usize, 4usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, m, k, n);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        let grouped = run(Device::Gpu, g, &packed, &x);
        let c = route_counts();
        let mut worst = 0.0f32;
        for r in 0..m {
            let (g1, _, _) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
            let row = run(Device::Gpu, g1, &packed, &x[r * k..(r + 1) * k]);
            worst = worst.max(max_abs_diff(&grouped[r * n..(r + 1) * n], &row));
        }
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (c.grouped, c.per_row),
            (1, 0),
            "the reference run must itself be the grouped path: {c:?}"
        );
        eprintln!("grouped m={m} vs per-row at k={k} n={n} chunked: max_abs={worst:.6e}");
        assert!(worst <= 1e-2, "grouped vs per-row max_abs {worst} > 1e-2");
    }

    // Two packed weights in one graph, so the second weight's region starts at a
    // NON-zero base: `from_plan_split` and the planner both place params
    // largest-first, and the single-param cases above never move that base off 0.
    // Both weights N-chunk (1 MiB bind cap), so every chunk resolves
    // `w_byte_off + n0 * row_bytes` against a non-zero base — the arithmetic a
    // multi-weight model needs and nothing else here exercises. The per-weight
    // CPU reference reuses `case`, whose packed bytes are a function of
    // (scheme, k, n) alone. (`gguf_gemm_parked_weight.rs` covers the same thing
    // for a weight parked in the weight buffer.)
    {
        let (k, na, nb, m) = (256usize, 8192usize, 8192usize, 4usize);
        let packed_of = |ggml: rlx_gguf::GgmlType, n: usize| {
            let w_row: Vec<f32> = (0..k * n)
                .map(|i| ((i as f32) * 0.011).sin() * 0.5)
                .collect();
            rlx_gguf::quantize(&w_row, ggml).expect("quantize")
        };
        // Q6_K is 210 bytes per column against Q4_K's 144, so `wb` is the larger
        // param and sorts first: `wa` lands at a non-zero offset.
        let (wa, wb) = (
            packed_of(rlx_gguf::GgmlType::Q4K, na),
            packed_of(rlx_gguf::GgmlType::Q6K, nb),
        );
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

        let mut g = Graph::new("wgpu_gguf_route_two_weights");
        let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
        let pa = g.param("wa", Shape::new(&[wa.len()], DType::U8));
        let pb = g.param("wb", Shape::new(&[wb.len()], DType::U8));
        let ya = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K,
            },
            vec![x_in, pa],
            Shape::new(&[m, na], DType::F32),
        );
        let yb = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ6K,
            },
            vec![x_in, pb],
            Shape::new(&[m, nb], DType::F32),
        );
        g.set_outputs(vec![ya, yb]);

        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        let mut c = Session::new(Device::Gpu).compile(g);
        c.set_param_typed("wa", &wa, DType::U8);
        c.set_param_typed("wb", &wb, DType::U8);
        let _ = c.run(&[("x", &x)]);
        reset_route_counts();
        let got = c.run(&[("x", &x)]);
        let counts = route_counts();
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (counts.grouped, counts.per_row),
            (2, 0),
            "both packed ops must be grouped: {counts:?}"
        );

        for (i, (scheme, ggml, n)) in [
            (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, na),
            (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K, nb),
        ]
        .into_iter()
        .enumerate()
        {
            let (g1, packed, x1) = case(scheme, ggml, m, k, n);
            let want = run(Device::Cpu, g1, &packed, &x1);
            let max_abs = max_abs_diff(&got[i], &want);
            eprintln!("two weights: op{i} {scheme:?} m={m} k={k} n={n} max_abs={max_abs:.6e}");
            assert!(
                max_abs <= 1e-2,
                "two weights: op{i} {scheme:?} max_abs {max_abs} > 1e-2"
            );
        }
    }

    // The one-bit scheme's one-row arm: it keeps the scalar X view while the
    // route's bind group also carries the tiled arm's fifth (`vec4` X) binding,
    // so this is its route and parity at m == 1 — not a reshaped shape, which it
    // deliberately does not have.
    {
        let (k, n) = (256usize, 640usize);
        let (g, packed, x) = case(QuantScheme::GgufQ1_0, rlx_gguf::GgmlType::Q1_0, 1, k, n);
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let c = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!("GgufQ1_0 m=1 k={k} n={n}: routes={c:?} shape={shape:?} max_abs={max_abs:.6e}");
        assert_eq!(
            (c.per_row, c.q1_0_gemm),
            (1, 0),
            "Q1_0 m=1 must take the one-row arm, not the tiled GEMM: {c:?}"
        );
        assert_eq!(
            (
                shape.cols_per_workgroup,
                shape.threads_per_workgroup,
                shape.workgroups
            ),
            (64, 64, n.div_ceil(64) as u32),
            "this arm keeps one thread per output column, 64 columns to a \
             workgroup: {shape:?}"
        );
        assert!(max_abs <= 1e-2, "GgufQ1_0 m=1: max_abs {max_abs} > 1e-2");
    }

    // One row: the packed multiply's shape at the widths the model bodies use
    // (H=1024 and I=6144 columns) and at the contraction lengths those
    // projections have, in the 2-D `[1, k]` form and the 3-D `[1, 1, k]`
    // single-token form those calls take. What is recorded is the arrangement
    // actually used — one workgroup per two output columns, with the workgroup's
    // 32 lanes splitting each of those columns' contractions over its 32-element
    // units — so `workgroups` is `ceil(n / 2)` and
    // `workgroups x cols_per_workgroup` is a *ceiling* on the width the dispatch
    // covered, not that width: a chunk of odd width has its last workgroup's
    // second column off the end, for which nothing is computed and whose store is
    // skipped. The product is then within one workgroup's span of the width asked
    // for, which is the "no tile row is computed and discarded" half — no row and
    // no column is ever dropped, and no more than a workgroup's worth of the
    // dispatch is beyond the width. The 8-bit scheme's own branch of the same arm
    // is driven at those widths too, its 32-element blocks standing in for the
    // K-quants' 256-element tile.
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
        (QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0),
    ] {
        for &(k, n, batches) in &[
            (1024usize, 1024usize, &[&[][..], &[1][..]][..]),
            (1024, 6144, &[&[][..], &[1][..]][..]),
            // The down-projection's longer contraction: the split is over k, so
            // this is the same shape with more work per column.
            (4096, 1024, &[&[][..]][..]),
            // The width edge this arm's two-column workgroup introduces: widths
            // below its per-workgroup span and widths not a multiple of it, the
            // case where the recorded `workgroups x cols_per_workgroup` has to
            // round the width up to a whole workgroup. n = 6145 is a model width
            // (6144) with one extra column, at the down-projection's own
            // contraction.
            (1024, 1, &[&[][..]][..]),
            (1024, 3, &[&[][..]][..]),
            (2048, 6145, &[&[][..]][..]),
        ] {
            for &batch in batches {
                let (g, packed, x) = case_batched(scheme, ggml, batch, 1, k, n);
                reset_route_counts();
                reset_gemv_one_row_shape();
                let gpu = run(Device::Gpu, g.clone(), &packed, &x);
                let counts = route_counts();
                let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
                eprintln!(
                    "[rlx-gguf-gemv-shape] scheme={scheme:?} m=1 k={k} n={n} batch={batch:?} \
                     cols_per_workgroup={} threads_per_workgroup={} workgroups={}",
                    shape.cols_per_workgroup, shape.threads_per_workgroup, shape.workgroups
                );
                assert_eq!(
                    (counts.per_row, counts.grouped),
                    (1, 0),
                    "{scheme:?} m=1 k={k} n={n} batch={batch:?}: {counts:?}"
                );
                assert_eq!(
                    (
                        shape.cols_per_workgroup,
                        shape.threads_per_workgroup,
                        shape.workgroups
                    ),
                    (2, 32, n.div_ceil(2) as u32),
                    "{scheme:?} k={k} n={n} batch={batch:?}: one row's work is its output \
                     columns times its contraction — a workgroup per two columns, its 32 \
                     lanes splitting each column, `ceil(n / 2)` workgroups and no tile \
                     rows: {shape:?}"
                );
                let cpu = run(Device::Cpu, g, &packed, &x);
                let max_abs = max_abs_diff(&gpu, &cpu);
                eprintln!(
                    "one-row shape {scheme:?} k={k} n={n} batch={batch:?}: max_abs={max_abs:.6e}"
                );
                assert!(
                    max_abs <= 1e-2,
                    "{scheme:?} k={k} n={n} batch={batch:?}: max_abs {max_abs} > 1e-2"
                );
            }
        }
    }

    // Throughput at those shapes and beyond over repeated calls — a measurement
    // for the record, not a threshold. At these one-row shapes the arm's
    // dispatches ride in the enclosing batch, so what a call costs per call is its
    // encode plus that batch's own submission and output readback, and the
    // remainder is the streaming it does — which is why this is not a single call,
    // and the best of a few rounds is reported rather than one round: at the
    // narrower width the per-call floor is a large part of the figure, so the
    // spread between rounds is real information about the shape, not noise to
    // smooth away silently.
    for (scheme, ggml) in [
        (QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K),
        (QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K),
    ] {
        for (k, n, iters) in [
            (1024usize, 1024usize, 200usize),
            (1024, 6144, 200),
            // Wider than the model widths (1024…6144 columns), where the arm
            // must not get worse than the shape it replaced, with a shorter run
            // so the cell stays cheap. A workgroup now covers two output columns,
            // so one dispatch reaches 65535 * 2 = 131070 columns before the
            // dispatch-dimension cap and none of these figures crosses it — the
            // extra chunk the cap forces above that width is pinned by the
            // 131072-column cell below.
            (1024, 16384, 40),
            (1024, 32768, 40),
            (1024, 65536, 40),
        ] {
            let (g, packed, x) = case(scheme, ggml, 1, k, n);
            let mut c = Session::new(Device::Gpu).compile(g);
            c.set_param_typed("w", &packed, DType::U8);
            let warm = c.run(&[("x", &x)]);
            assert_eq!(warm[0].len(), n, "one row of {n} columns");
            let mut best_us = f64::MAX;
            for _ in 0..3 {
                // `Tick`, not `Instant`: this measures sub-millisecond per-call
                // work, and the tree mandates the tick at that scale.
                let start = rlx_runtime::Tick::now();
                for _ in 0..iters {
                    let _ = c.run(&[("x", &x)]);
                }
                best_us = best_us.min(rlx_runtime::Tick::now().elapsed_us(start));
            }
            let streamed_mib = (packed.len() * iters) as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[rlx-gguf-gemv-throughput] scheme={scheme:?} m=1 k={k} n={n} iters={iters} \
                 streamed_MiB={streamed_mib:.1} best_ms={:.2} per_call_us={:.0} \
                 streamed_MiB_per_s={:.0}",
                best_us * 1e-3,
                best_us / iters as f64,
                streamed_mib / (best_us * 1e-6)
            );
        }
    }

    // A multi-row Q8_0 call against the slab it used to take, at the shapes the
    // byte-per-weight steps multiply — the steps' own rows (13, 17, 72) against
    // `[1024,1024]`, `[1024,4096]` / `[4096,1024]`, `[1024,3072]` / `[3072,1024]`,
    // `[512,512]`, `[512,1024]` and `[1024,512]` — plus the class's own boundaries on
    // the arm's block term, and a row sweep either side of the change point at one of
    // those widths, since a row-grouped kernel can lose above the rows it was built
    // for.
    //
    // Both sides are one build and one instrument: A is the default route, B is
    // the same call with `RLX_WGPU_Q8_0_GEMM_DISABLE=1`, which is the dequant slab
    // — this scheme's route before the multi-row arm, at every m above one. That
    // switch lowers the row-grouped encoder and nothing else, so the single-row arm
    // is out of its reach: at m=1 both sides are that arm (the ~1 ratio it records,
    // and its census below), and that arm's own expansion — the slab this scheme's
    // one-row calls used to take — is driven from the plan instead. The two sides
    // share one compiled graph, one arena and one uploaded weight, and each side's
    // own census below proves which route it ran. The calls alternate A/B within a
    // round rather than running one side at a time: this crate's A/B protocol notes
    // record a first-touch ordering bias that made a later arm look up to 20% faster
    // than it was.
    {
        /// Timed rounds; each round runs both sides once. The figures are a
        /// measurement for the record, like `dequant_plan_relaxed.rs`'s — the
        /// asserted facts here are the routes, the route census, the step count and
        /// the values.
        const ROUNDS: usize = 5;

        for &(m, k, n) in &[
            // The packed steps' own cells, all multi-row.
            (13usize, 1024usize, 1024usize),
            (13, 1024, 4096),
            (13, 4096, 1024),
            (17, 1024, 1024),
            (17, 1024, 3072),
            (17, 3072, 1024),
            (72, 512, 512),
            (72, 512, 1024),
            (72, 1024, 512),
            // The class's own boundaries on the block term, at the stand's row count:
            // k=128 ends inside a staged tile, k=288 is a tile plus a block, and
            // k=896 is the width the stand multiplies at. All three are whole numbers
            // of the scheme's 32-element blocks and none is on the tile, so this is
            // where the admission's block-vs-tile question is measured rather than
            // reasoned about.
            (13, 128, 1024),
            (13, 288, 1024),
            (13, 896, 1024),
            // The sub-tile end of that same class: one whole 32-element block of k,
            // where the arm stages a 256-element tile — eight of them. The ticket
            // admits this end only if it is shown not to lose, so its figures are
            // printed beside the others' while its route, its join and its values are
            // asserted like the rest: one tile iteration is paid for a single real
            // block here.
            (13, 32, 1024),
            // The row sweep, on a model width, either side of the arm's change
            // point (m == 1) and past the ~96 rows one large-row measurement put
            // in doubt on Metal. m=1 is the control: both sides are this scheme's
            // one-row arm there, so a route away from it would say the switch does
            // more than lower this arm. m=2 and m=8 are the first row group and the
            // tile; 512 and 1024 are an order of magnitude past the cells' rows,
            // where a row-grouped kernel with no cap must still hold.
            (1, 1024, 1024),
            (2, 1024, 1024),
            (8, 1024, 1024),
            (96, 1024, 1024),
            (192, 1024, 1024),
            (276, 1024, 1024),
            (512, 1024, 1024),
            (1024, 1024, 1024),
        ] {
            let (g, packed, x) = case(QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, m, k, n);
            let mut c = Session::new(Device::Gpu).compile(g);
            c.set_param_typed("w", &packed, DType::U8);
            let _ = c.run(&[("x", &x)]);

            // The route each side takes, from this binary's own counters, and what
            // each side's steps cost the device. The arm's side is now work the batch
            // boundary *joins*: its multiply rides the graph's one submission, whose
            // readback is encoded into the same command buffer, so one submission
            // covers the whole forward pass. The lever's side is the refusal this
            // whole arrangement exists to reach: the ladder reads no lever, so the
            // same call is still admitted, the encode inside the batch declines, the
            // boundary's tail submits that batch carrying no work for the call — one
            // submission more than the arm's — and the run's own arm recomputes on the
            // slab, whose `dequant_gguf` and `matmul_bt` submit again (hence the four
            // against the one it prints, on an adapter that fuses the readback; a
            // separate readback adds its own to both sides). At the m=1 control both
            // sides are this scheme's one-row arm — the lever lowers the grouped
            // encoder alone — so that cell's ratio is ~1 and its two censuses agree:
            // the record that the single-row arm has no lever to be lowered by, and
            // why its own slab leg below is driven from the plan instead.
            reset_route_counts();
            reset_device_counts();
            reset_gemv_one_row_shape();
            let direct = c.run(&[("x", &x)]).remove(0);
            let a = route_counts();
            let a_submits = device_submit_count();
            let a_waits = device_wait_count();
            rlx_ir::env::set("RLX_WGPU_Q8_0_GEMM_DISABLE", "1");
            reset_route_counts();
            reset_device_counts();
            let slab = c.run(&[("x", &x)]).remove(0);
            let b = route_counts();
            let b_submits = device_submit_count();
            let b_waits = device_wait_count();
            rlx_ir::env::unset("RLX_WGPU_Q8_0_GEMM_DISABLE");
            // Only the arm is lowered: a call with no slab behind it must not
            // change route with the switch either, so this is exact for both.
            //
            // Every multi-row cell of this grid is admitted and the m=1 control is
            // not — that cell is this scheme's own one-row arm, which the lever does
            // not reach — so the expectation is the ladder's terms as a literal
            // rather than a restatement of them: each `k` in the grid is a whole
            // number of this scheme's own 32-element blocks, which is the admission's
            // `k` term, the planner aligns the X window, and the plan books this
            // scheme's slab unconditionally. The 128 / 288 / 896 cells are off the
            // 256-element staged tile that arm stages eight blocks of, and are still
            // admitted: the tile is not the gate. A lane that stops being admitted at
            // one of these shapes must re-point this literal deliberately; it fails
            // the census below rather than skipping the block's own checks.
            let direct_arm = m > 1;
            assert_eq!(
                (
                    a.per_row, a.grouped, a.scratch, b.per_row, b.grouped, b.scratch
                ),
                if direct_arm {
                    (0, 1, 0, 0, 0, 1)
                } else {
                    (1, 0, 0, 1, 0, 0)
                },
                "Q8_0 m={m} k={k} n={n}: the arm must be the default route and the \
                 switch must lower it to the slab above one row, while the single-row \
                 arm has no lever at all: {a:?} / {b:?}"
            );

            let mut a_us = f64::MAX;
            let mut b_us = f64::MAX;
            for _ in 0..ROUNDS {
                let t0 = rlx_runtime::Tick::now();
                let _ = c.run(&[("x", &x)]);
                a_us = a_us.min(rlx_runtime::Tick::now().elapsed_us(t0));

                rlx_ir::env::set("RLX_WGPU_Q8_0_GEMM_DISABLE", "1");
                let t0 = rlx_runtime::Tick::now();
                let _ = c.run(&[("x", &x)]);
                b_us = b_us.min(rlx_runtime::Tick::now().elapsed_us(t0));
                rlx_ir::env::unset("RLX_WGPU_Q8_0_GEMM_DISABLE");
            }
            let packed_mib = packed.len() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[rlx-gguf-q8-0-gemm] m={m} k={k} n={n} packed_MiB={packed_mib:.2} \
                 arm_us={a_us:.0} lever_us={b_us:.0} ratio={:.2}x arm_MiB_per_s={:.0} \
                 lever_MiB_per_s={:.0} arm_submits={a_submits} lever_submits={b_submits} \
                 arm_waits={a_waits} lever_waits={b_waits}",
                b_us / a_us,
                packed_mib / (a_us * 1e-6),
                packed_mib / (b_us * 1e-6)
            );

            if direct_arm {
                // The refused call's own cost, which is the other half of leaving
                // the slab. In the default arrangement the arm's multiply is joined
                // into the graph's one submission, so the lever's side pays the
                // batch submission the refused call leaves carrying no work for it,
                // plus the slab's dequant-then-`matmul_bt` pair: at least three
                // more than the arm's side. Under `RLX_WGPU_ONE_OP_PER_PASS` the
                // join is inert and both sides are unjoined, so only that pair
                // stands between them: at least one more. A separate readback adds
                // its own to both sides, so each is a floor and not an equality.
                let floor = if step_by_step { 1 } else { 3 };
                let arrangement = if step_by_step {
                    "step-by-step (`RLX_WGPU_ONE_OP_PER_PASS=1`)"
                } else {
                    "joined"
                };
                assert!(
                    b_submits >= a_submits + floor,
                    "Q8_0 m={m} k={k} n={n}: in the {arrangement} arrangement the \
                     refused call must cost at least {floor} submission(s) more than \
                     the arm's {a_submits}, not {b_submits}"
                );
                // The join witness for the class's boundaries, which is the ticket's
                // own acceptance for the sub-tile end: what the ladder newly admits
                // must be submitted exactly as the shapes it already admitted are —
                // the whole run one submission and one blocking wait, the counts the
                // m=1 branch pins and the stream-rate cells pin for this route. Stated
                // at the cells the class changed rather than for every cell of the
                // grid, since a separate readback would add its own to both and the
                // floor above is what covers the grid's older shapes. Guarded like
                // that branch's own pair: under `RLX_WGPU_ONE_OP_PER_PASS` the join is
                // inert and these cells' counts are the arrangement's, not the route's.
                if !step_by_step && matches!((m, k), (13, 128) | (13, 288) | (13, 896) | (13, 32)) {
                    assert_eq!(
                        (a_submits, a_waits),
                        (1, 1),
                        "Q8_0 m={m} k={k} n={n}: an admitted call must ride the graph's \
                         single submission and open none of its own, as the row-aligned \
                         ones do: took {a_submits} submission(s) and {a_waits} wait(s)"
                    );
                }
                // The arm's own values, against both references it has. The slab
                // side is the same weight and inputs through the f32 path this
                // arm replaced — a route-against-route check, independent of the
                // test's weight scale, that catches the stride and the byte
                // unpack of the arm's own read. The CPU reference is the op's
                // declared reference; same absolute 1e-2 every packed-multiply
                // check in this crate uses. The CPU side runs for the packed steps'
                // own rows (m <= 72) only: past them the grid above and
                // `gguf_dequant_matmul_prefill_parity.rs` carry the same
                // comparison at their own shapes.
                let vs_slab = max_abs_diff(&direct, &slab);
                eprintln!("byte-direct Q8_0 m={m} k={k} n={n}: vs_slab_max_abs={vs_slab:.6e}");
                assert!(
                    vs_slab <= 1e-2,
                    "byte-direct Q8_0 m={m} k={k} n={n}: {vs_slab} from the slab route"
                );
                if m <= 72 {
                    let (gc, _, _) = case(QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, m, k, n);
                    let cpu = run(Device::Cpu, gc, &packed, &x);
                    let max_abs = max_abs_diff(&direct, &cpu);
                    eprintln!("byte-direct Q8_0 m={m} k={k} n={n}: vs_cpu_max_abs={max_abs:.6e}");
                    assert!(
                        max_abs <= 1e-2,
                        "byte-direct Q8_0 m={m} k={k} n={n}: max_abs {max_abs} > 1e-2"
                    );
                }
            }

            if !direct_arm {
                // Which arrangement the control cell runs, not only that it is not
                // the slab: one workgroup per two output columns with its 32 lanes
                // splitting each column's contraction over this scheme's own
                // 32-element blocks — the geometry the K-quant one-row cells above
                // assert at their own widths, and not the row-grouped GEMM widened
                // down to one row (which would compute a group of eight rows' worth
                // of arithmetic for a single step).
                let shape = gemv_one_row_shape().expect("a single-row dispatch records its shape");
                assert_eq!(
                    (
                        shape.cols_per_workgroup,
                        shape.threads_per_workgroup,
                        shape.workgroups
                    ),
                    (2u32, 32u32, (n as u32).div_ceil(2)),
                    "Q8_0 m={m} k={k} n={n}: the single-row arm's own geometry: {shape:?}"
                );

                // The control's other half: the expansion this arm left behind,
                // which the lever cannot produce at this row count, is driven from
                // the plan instead. Its census is what proves the route — a slab leg
                // — and its submission count is the other half of the record, in the
                // arrangement this run is in, so it is read and printed beside the
                // arm's own counts: joined, this arm's multiply rides the graph's one
                // submission and opens none of its own, so the slab leg's
                // `dequant_gguf` pass and `matmul_bt` are one submit more; unjoined
                // (the step-by-step lever) that call opens its own submission and
                // pays the run's readback too, which is as many as the slab leg's
                // pair, so the counts meet and the census is what tells the two
                // arrangements apart there.
                let dev = wgpu_device().expect("wgpu device: the run above used one");
                // The public reporting hook answers for this scheme's one row as it
                // does for a K-quant's — the arm's chunking rule, not `None`. Printed
                // rather than barred: the count is the rule's own figure, which the
                // stream-rate cell reports the same way.
                let dispatches = rlx_wgpu::gguf_gpu::gemv_one_row_dispatches(
                    &dev.device,
                    rlx_wgpu::gguf_host::gguf_scheme_id(QuantScheme::GgufQ8_0),
                    k,
                    n,
                )
                .expect("this scheme's one-row arm: the chunking rule answers for it");
                let (gc, _, _) = case(QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, m, k, n);
                let (expanding, expanding_routes, slab_submits, slab_waits) =
                    one_row_slab_leg(dev, &gc, k, n, &packed, &x);
                let vs_expanding = max_abs_diff(&direct, &expanding);
                // Each leg's counters cover its own readback — the graph's is inside
                // its `run`, the driven leg's is inside `one_row_slab_leg` — so the two
                // are the two arms' own work, printed side by side for the record. What
                // is barred below is the arm's own count.
                eprintln!(
                    "[rlx-gguf-q8-0-one-row] m={m} k={k} n={n} joined_us={a_us:.0} \
                     joined_submits={a_submits} joined_waits={a_waits} \
                     expanding_submits={slab_submits} expanding_waits={slab_waits} \
                     vs_expanding_max_abs={vs_expanding:.6e} dispatches={dispatches} \
                     expanding_route={expanding_routes:?}"
                );
                assert_eq!(
                    (
                        expanding_routes.per_row,
                        expanding_routes.grouped,
                        expanding_routes.scratch
                    ),
                    (0, 0, 1),
                    "Q8_0 m={m} k={k} n={n}: the driven leg must be the slab this \
                     arm replaced: {expanding_routes:?}"
                );
                // The witness for "opens no submission of its own" is the arm's own
                // count, not a comparison with the slab leg's: a regression that
                // un-joined this call adds one submission to BOTH legs, so their
                // difference cannot see it. Joined, the whole control call — its
                // multiply and its readback — is one submission and one blocking wait,
                // the same counts the stream-rate cells pin for this route
                // (`gguf_one_row_stream_rate.rs`); unjoined it would read two. Under
                // `RLX_WGPU_ONE_OP_PER_PASS` the join is inert and that submission is
                // the arrangement's own, so the counts printed above are its record.
                if !step_by_step {
                    assert_eq!(
                        (a_submits, a_waits),
                        (1, 1),
                        "Q8_0 m={m} k={k} n={n}: joined, the one-row call must ride the \
                         graph's single submission and open none of its own: took \
                         {a_submits} submission(s) and {a_waits} wait(s)"
                    );
                }
                assert!(
                    vs_expanding <= 1e-2,
                    "the one-row arm against the slab it replaced at Q8_0 m={m} k={k} \
                     n={n}: {vs_expanding}"
                );
            }
        }
        // The pooled slab the arm's path allocates for a chunk's output — the
        // process-global, lazily created, never-released buffer the plan does not
        // know about. Printed here as this process's peak, measured rather than
        // assumed: this block sizes it, its widest cell being the sweep's m=1024
        // at n=1024 (4 MiB of output) against the grid cells' 1.5 MiB. Compare it
        // against the in-arena slab those calls no longer read (`scratch_graph_max`,
        // still 16 MiB in the model block of the prefill parity test: the planner
        // books this scheme's slab by scheme, whatever route a call takes).
        eprintln!(
            "[rlx-gguf-q8-0-gemm] pooled_out_buf_bytes={}",
            rlx_wgpu::gguf_gpu::pooled_out_buf_bytes()
        );
    }

    // The same arm when the weight window exceeds the binding limit: rows are
    // N-chunked, so one call is several dispatches and the recorded one is the
    // last chunk's — a whole-workgroup ceiling on that chunk's width, and never
    // zero. The chunk stride is the share of the row indexing a wrong chunking
    // would move, which is what the parity check against CPU pins.
    {
        let (k, n) = (256usize, 16384usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let counts = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
        assert_eq!(
            (counts.per_row, counts.grouped),
            (1, 0),
            "a 1 MiB window must still take the one-row arm: {counts:?}"
        );
        assert_eq!(
            (shape.cols_per_workgroup, shape.threads_per_workgroup),
            (2, 32),
            "a workgroup covers two output columns: {shape:?}"
        );
        assert!(
            shape.workgroups as usize > 0 && (shape.workgroups as usize) < n.div_ceil(2),
            "the window must have been split into more than one chunk: {shape:?}"
        );
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "chunked one-row GgufQ4K m=1 k={k} n={n}: last-chunk workgroups={} max_abs={max_abs:.6e}",
            shape.workgroups
        );
        assert!(
            max_abs <= 1e-2,
            "chunked one-row GgufQ4K k={k} n={n}: max_abs {max_abs} > 1e-2"
        );
    }

    // The same arm when the output is wider than one dispatch dimension's cap:
    // the tiled arm spends one workgroup per two output columns, and 65535 is the
    // WebGPU floor for a dimension (`max_compute_workgroups_per_dimension`), so
    // one dispatch covers 65535 * 2 = 131070 columns and a wider output has to be
    // another chunk of the same arm. Reaching the cap in one dispatch is a wgpu
    // validation error rather than a slow path — this cell is what keeps that
    // from coming back. The weight window is far under the binding limit, so the
    // split is the cap's and nothing else's, and the chunk boundary is a fresh
    // stride through the packed rows, which the CPU parity pins.
    {
        // One full cap-width plus two columns, so the remainder chunk is
        // 131072 - 131070 = 2 columns: exactly one workgroup of the two-column
        // arm, and the narrowest remainder there is.
        let (k, n) = (256usize, 131072usize);
        let (g, packed, x) = case(QuantScheme::GgufQ4K, rlx_gguf::GgmlType::Q4K, 1, k, n);
        reset_route_counts();
        reset_gemv_one_row_shape();
        let gpu = run(Device::Gpu, g.clone(), &packed, &x);
        let counts = route_counts();
        let shape = gemv_one_row_shape().expect("the one-row arm must record its shape");
        let cpu = run(Device::Cpu, g, &packed, &x);
        assert_eq!(
            (counts.per_row, counts.grouped),
            (1, 0),
            "an output over the dispatch cap must still take the one-row arm: {counts:?}"
        );
        assert_eq!(
            (
                shape.cols_per_workgroup,
                shape.threads_per_workgroup,
                shape.workgroups as usize
            ),
            (2, 32, 1),
            "the dispatch cap (the WebGPU floor for one dimension, \
             max_compute_workgroups_per_dimension) is 65535 * cols_per_workgroup = \
             131070 columns, so {n} columns must be split into that plus a \
             two-column remainder — one workgroup: {shape:?}"
        );
        let max_abs = max_abs_diff(&gpu, &cpu);
        eprintln!(
            "capped one-row GgufQ4K m=1 k={k} n={n}: last-chunk workgroups={} max_abs={max_abs:.6e}",
            shape.workgroups
        );
        assert!(
            max_abs <= 1e-2,
            "capped one-row GgufQ4K k={k} n={n}: max_abs {max_abs} > 1e-2"
        );
    }

    // The grouped path's output slab is device memory the arena plan does not own,
    // so it has to be visible to the backend's memory report — the reported figure
    // must not understate real usage — and inside its cap.
    let pooled = rlx_wgpu::gguf_gpu::pooled_out_buf_bytes();
    assert!(
        pooled > 0 && pooled <= 256 * 1024 * 1024,
        "pooled out slab {pooled} bytes must be reported and inside the 256 MiB cap"
    );
}
