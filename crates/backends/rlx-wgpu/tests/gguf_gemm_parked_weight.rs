// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The row-grouped GGUF GEMM against a weight parked in the backend's separate
//! packed-weight buffer — the regime a model larger than the device's binding
//! limit lands in — the same weight under the 8-bit scheme's multi-row call,
//! whose route the device-free admission moves, and, through the same hand-built
//! arena, the one-row route's X-window alignment guard.
//!
//! The executable only takes that regime when the *plan* exceeds the binding
//! limit: 4 GiB of packed weights on this hardware, which no test can afford to
//! build. So the split arena is built directly — `Arena::from_plan_split` parks
//! every packed quant param — and the same runner the executable calls is driven
//! against it.
//!
//! Two packed params, placed largest-first, put the second one's region at a
//! NON-zero weight-buffer base, and the 1 MiB window cap makes it several column
//! chunks: every chunk resolves `w_byte_off + n0 * row_bytes` against that base.
//! The reference is the same graph and the same bytes through the non-split
//! arena, i.e. the path the CPU-parity tests already cover.
//!
//! The Q8_0 test is the class whose route moves where no executable could be
//! made to: `gguf_gpu::select_path` answers `DequantMatmulPath::GroupedDevice`
//! for a multi-row 8-bit call with a slab booked, a whole staged tile and an
//! addressable X window, and it is *device-free* — the weight's placement never
//! enters its answer, so a parked weight is admitted to the batch boundary's
//! join just as the in-arena copy is. The scratch entry point's own
//! short-circuit, which exists precisely for the parked regime, would have put
//! that call on the CPU route. The joined arm reads the parked weight in place
//! through the row-grouped GEMM's chunked window (`resolve_w` N-chunks the
//! weight region, so a parked or oversized weight still takes that arm), and the
//! test's `[rlx-gguf-parked-q8-0]` line is the report that movement owes: the
//! route each arrangement reaches, the submissions and polls each pays, and the
//! difference between their outputs against this crate's 1e-2 cross-route bar for
//! packed multiplies.
//!
//! The virtual bind-stripe arena — one physical buffer, `shard_size > 0` and
//! `extra_shards` empty, the geometry `from_plan` chooses for a plan that fits one
//! allocation but exceeds the adapter's storage-buffer binding limit — is the
//! geometry this file measures the same movement on: `Arena::resolve_act` hands
//! back the single buffer's global offset, so the row-grouped GEMM's encode
//! declines nothing and an admitted multi-row Q8_0 call runs the device arm that
//! the scratch entry point's `is_sharded` test sends to the CPU. That
//! short-circuit's gate covers a physically striped arena too, and the movement
//! applies there as well whenever the call's windows fit their stripe —
//! `DequantMatmulPath::GroupedDevice` states that layout as the same movement,
//! unmeasured, and this one is what a fixture reaches with the shard levers off.
//! The plan is deliberately not snapped to stripes — these code paths read two
//! facts off the arena, `is_sharded()` and that global offset, and both hold
//! either way.
//!
//! The `#[ignore]`d test repeats the parked exercise at the size the harness uses
//! (777 MiB, four bind windows), and the alignment guard hands the one-row runner
//! an X window four bytes off its slot, which is the only way to reach it — the
//! planner aligns every activation.

use rlx_compile::memory::plan_memory_f32_uniform;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_wgpu::backend::PassMarks;
use rlx_wgpu::buffer::{Arena, is_weight_off, raw_weight_off};
use rlx_wgpu::gguf_gpu::{
    DequantMatmulPath, DequantMatmulRoutes, dequant_gguf_scratch_bytes,
    encode_dequant_matmul_gguf_into, reset_route_counts, route_counts,
    run_dequant_matmul_gguf_gemv_rows, run_dequant_matmul_gguf_gpu, select_path,
};
use rlx_wgpu::gguf_host::gguf_scheme_id;
use rlx_wgpu::wgpu_device;
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};
use std::sync::Mutex;

mod common;

/// The backend's device counters are process-global, and every test in this binary
/// dispatches — so the tests take this lock in turn. That is the discipline
/// `gguf_matmul_route.rs` gets from being one test fn in its own binary; this file's
/// claims are separate fns, so a bracketed window would otherwise read a
/// neighbour's submissions as much as its own.
static COUNTERS: Mutex<()> = Mutex::new(());

/// Deterministic Q4_K-shaped bytes: `k/256 × 144` per column, every nibble and
/// scale field masked the way the real packer writes them.
fn packed(k: usize, n: usize) -> Vec<u8> {
    let mut s = 11u32;
    (0..(k / 256) * 144 * n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 24) as u8;
            match i % 144 {
                1 | 3 => 0x08 | (b & 0x03),
                _ => b & 0x3F,
            }
        })
        .collect()
}

/// The all-zero guard every comparison's operand goes through: a route that
/// dispatched nothing reads an untouched slot, and a difference against one says
/// nothing.
fn assert_signal(v: &[f32], what: &str) {
    assert!(v.iter().any(|x| x.abs() > 1e-3), "{what} is all-zero");
}

/// [`common::max_abs_diff`] against the reference route's own magnitude, so the
/// relative figure says something about a call's values and not only about its shape.
fn cross_route(moved: &[f32], reference: &[f32]) -> (f32, f32) {
    let max_abs = common::max_abs_diff(moved, reference);
    (max_abs, max_abs / magnitude(reference))
}

/// A reference output's magnitude — the denominator of [`cross_route`]'s figure.
fn magnitude(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |a, x| a.max(x.abs()))
}

/// One `DequantMatMul` over a packed Q4_K weight of `w_bytes`, and the node ids
/// (x, w, y) a caller needs to place and read them.
fn one_weight_graph(
    k: usize,
    n: usize,
    m: usize,
    w_bytes: usize,
) -> (Graph, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("wgpu_gguf_parked_weight");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[w_bytes], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    (g, x, w, y)
}

/// `from_plan_split` with this graph, asserting the packed weight landed in the
/// weight buffer rather than the act arena.
fn parked_arena(dev: &rlx_wgpu::device::WgpuDevice, g: &Graph, w: NodeId) -> Arena {
    let plan = plan_memory_f32_uniform(g, 16);
    let arena = Arena::from_plan_split(&dev.device, &plan, g, 0);
    assert!(
        arena.weight_buffer.is_some() && is_weight_off(arena.offset(w)),
        "a packed quant param must be parked in the weight buffer"
    );
    arena
}

/// The graph and bytes both route-movement tests drive: `n` columns of a real
/// Q8_0-packed weight over `m` rows of activations, with an optional larger Q4_K
/// param that pushes the weight under test to a non-zero weight-buffer base.
struct Q8Case {
    graph: Graph,
    packed: Vec<u8>,
    x: Vec<f32>,
    x_in: NodeId,
    w: NodeId,
    y: NodeId,
}

/// The [`Q8Case`] both route-movement tests drive, built once so their arenas differ
/// in nothing but the geometry under test: `k` a whole number of the scheme's
/// 32-element blocks (eight of them) and so of the grouped GEMM's 256-element staged
/// tile, `m > 1` past the one-row arm this scheme does not have, and `n` wide enough
/// that the 1 MiB window cap those tests set splits the weight into several chunks,
/// each resolving `w_base + n0 * row_bytes`. Real Q8_0 bytes, through
/// [`common::packed_gguf_bytes`]: this arm unpacks the scheme's scale and quant
/// fields, so this file's synthetic Q4_K masks would be a decode of bytes the packer
/// never writes.
///
/// A non-zero `pad_n` adds that synthetic Q4_K param, declared before the weight
/// under test and larger than it — `from_plan_split` parks the largest first, so the
/// pad takes the weight buffer's base and the weight under test a NON-zero one — and
/// declared an output, so the only thing these tests use of it (its placement) is not
/// something a pruning pass could be free to drop. Its value is no claim's.
fn q8_case(name: &str, m: usize, k: usize, n: usize, pad_n: usize) -> Q8Case {
    let w_packed = common::packed_gguf_bytes(rlx_gguf::GgmlType::Q8_0, k, n);
    let x = common::ramp(m, k, 1e-3);

    let mut g = Graph::new(name);
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let pad_w = (pad_n > 0).then(|| {
        let pad = packed(k, pad_n);
        g.param("pad", Shape::new(&[pad.len()], DType::U8))
    });
    let w = g.param("w", Shape::new(&[w_packed.len()], DType::U8));
    let pad_y = pad_w.map(|pad_w| {
        g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K,
            },
            vec![x_in, pad_w],
            Shape::new(&[m, pad_n], DType::F32),
        )
    });
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(match pad_y {
        Some(pad_y) => vec![y, pad_y],
        None => vec![y],
    });
    Q8Case {
        graph: g,
        packed: w_packed,
        x,
        x_in,
        w,
        y,
    }
}

/// The case's operands written into `arena`, under the `{label}: x @…, w @…, y @…`
/// line every arrangement of the fixture prints — the bytes are a function of
/// `(k, n)` alone, so the arrangements write the same ones and those three offsets
/// are what tells them apart. Returns them.
fn place_q8_case(
    dev: &rlx_wgpu::device::WgpuDevice,
    case: &Q8Case,
    arena: &Arena,
    label: &str,
) -> (usize, usize, usize) {
    arena.write_bytes_range(&dev.queue, arena.offset(case.w), &case.packed);
    arena.write_f32(&dev.queue, case.x_in, &case.x);
    let (x_off, w_off, y_off) = (
        arena.offset(case.x_in),
        arena.offset(case.w),
        arena.offset(case.y),
    );
    eprintln!("{label}: x @{x_off:#x}, w @{w_off:#x}, y @{y_off:#x}");
    (x_off, w_off, y_off)
}

/// The admission's shared, device-free premises, returning the slab the plan books:
/// a positive booking (the plan's own `scratch_off`, which the Q8_0 op books — a
/// Q4_K pad is a scratch-free scheme), the multi-row call with that slab behind it
/// admitted to the join, that arm provably device-only, and the same answer over
/// `other_arena_x_off` — `blind` naming the fact the two arenas differ in.
///
/// That last comparison cannot prove the ladder blind to the arena: [`select_path`]
/// takes no arena, so the two answers can only differ when one window is not
/// four-element addressable, and this is what would catch a fixture whose window was
/// not. The offsets are printed with the message for exactly that case.
fn assert_q8_admission(
    case: &Q8Case,
    m: usize,
    k: usize,
    x_off: usize,
    other_arena_x_off: usize,
    blind: &str,
) -> usize {
    let booking = dequant_gguf_scratch_bytes(&case.graph);
    assert!(
        booking > 0,
        "the plan must book a dequant slab for this graph, got {booking}"
    );
    let scheme = gguf_scheme_id(QuantScheme::GgufQ8_0);
    let joined_path = select_path(m, k, scheme, x_off, booking);
    assert_eq!(
        joined_path,
        DequantMatmulPath::GroupedDevice,
        "a multi-row 8-bit call with a slab behind it must be admitted to the join"
    );
    assert!(
        joined_path.is_provably_device_only(),
        "the admission's answer for this call must be one of the two device-only arms"
    );
    assert_eq!(
        select_path(m, k, scheme, other_arena_x_off, booking),
        joined_path,
        "{blind} must not enter the admission's answer (X windows {x_off:#x} and \
         {other_arena_x_off:#x})"
    );
    booking
}

/// One arrangement's reading: its output, plus the route census and the device
/// counters its own bracketed window recorded.
struct Leg {
    y: Vec<f32>,
    routes: DequantMatmulRoutes,
    submits: u64,
    waits: u64,
}

/// One leg's route census must be the row-grouped GEMM and neither of the two it must
/// not fall back to; `what` is the claim the leg carries.
fn assert_grouped_routes(routes: &DequantMatmulRoutes, what: &str) {
    assert_eq!(
        (routes.grouped, routes.host, routes.scratch),
        (1, 0, 0),
        "{what}: {routes:?}"
    );
}

/// One leg's route census must be the host arm; `what` is the claim the leg carries.
fn assert_host_route(routes: &DequantMatmulRoutes, what: &str) {
    assert_eq!((routes.host, routes.grouped), (1, 0), "{what}: {routes:?}");
}

/// The movement's own difference, in the values: the device route against the CPU
/// route for one weight, which they read and accumulate in a different order — so
/// this is a tolerance figure, not a bit-identity claim (the bit-identity both
/// reports do carry is between the same route's two arrangements). Both operands are
/// guarded against a route that dispatched nothing and read an untouched slot.
fn movement(device_route: &[f32], cpu_route: &[f32]) -> (f32, f32) {
    assert_signal(device_route, "the device route's output");
    assert_signal(cpu_route, "the CPU route's output");
    cross_route(device_route, cpu_route)
}

/// The bar [`movement`]'s difference is held to: this crate's cross-route tolerance
/// for a packed multiply. `what` names the geometry it was measured on.
fn assert_under_cross_route_bar(max_abs: f32, what: &str) {
    assert!(
        max_abs <= 1e-2,
        "grouped against the CPU route on {what}: max_abs {max_abs:e} > 1e-2, the crate's \
         cross-route tolerance for packed multiplies — a route-to-route bar, not a \
         bit-identity claim"
    );
}

/// Drive one arrangement under the process-global counters: reset them, run the
/// caller's `drive` against a fresh marks handle, then read the census and the
/// device counters off that window. Those are read BEFORE the readback —
/// `read_f32` submits and polls a copy of its own, which is not the arrangement's
/// work and is not what a measured window is about.
fn leg(
    dev: &rlx_wgpu::device::WgpuDevice,
    arena: &Arena,
    case: &Q8Case,
    drive: impl FnOnce(&mut PassMarks<'_>),
) -> Leg {
    reset_device_counts();
    reset_route_counts();
    let mut marks = PassMarks::none();
    drive(&mut marks);
    let (routes, submits, waits) = (route_counts(), device_submit_count(), device_wait_count());
    Leg {
        y: arena.read_f32(&dev.device, &dev.queue, case.y),
        routes,
        submits,
        waits,
    }
}

#[test]
fn grouped_gemm_reads_a_parked_weight_from_the_weight_buffer() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let _counters = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let (k, m) = (256usize, 4usize);
    // Columns per weight: the larger one is placed first, so the smaller one's
    // region starts at a non-zero base. Both are wider than one 1 MiB window, so
    // each is dispatched in several column chunks.
    let (na, nb) = (12288usize, 8192usize);
    let (wa, wb) = (packed(k, na), packed(k, nb));
    let x = common::ramp(m, k, 1e-3);

    let mut g = Graph::new("wgpu_gguf_parked_weight");
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
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x_in, pb],
        Shape::new(&[m, nb], DType::F32),
    );
    g.set_outputs(vec![ya, yb]);

    let plan = plan_memory_f32_uniform(&g, 16);
    let plain = Arena::from_plan(&dev.device, &plan);
    let parked = parked_arena(dev, &g, pa);

    // The premise of the test: both packed params really are in the weight
    // buffer, at distinct offsets, and one of them is at a non-zero base.
    let (oa, ob) = (parked.offset(pa), parked.offset(pb));
    assert_eq!(
        raw_weight_off(oa).min(raw_weight_off(ob)),
        0,
        "largest-first placement puts one weight at the buffer base"
    );
    assert!(
        raw_weight_off(oa).max(raw_weight_off(ob)) > 0,
        "the other weight must sit at a non-zero base (got {oa:#x} / {ob:#x})"
    );
    for (arena, label) in [(&plain, "plain"), (&parked, "parked")] {
        arena.write_bytes_range(&dev.queue, arena.offset(pa), &wa);
        arena.write_bytes_range(&dev.queue, arena.offset(pb), &wb);
        arena.write_f32(&dev.queue, x_in, &x);
        eprintln!(
            "{label}: wa @{:#x}, wb @{:#x}",
            arena.offset(pa),
            arena.offset(pb)
        );
    }

    // Multi-chunk windows for both weights, so the chunk loop (not just the first
    // chunk) has to resolve against the parked base.
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
    for (n, y) in [(na, ya), (nb, yb)] {
        run_dequant_matmul_gguf_gemv_rows(
            &parked,
            &dev.device,
            &dev.queue,
            m,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            parked.offset(x_in),
            parked.offset(if n == na { pa } else { pb }),
            parked.offset(y),
            &mut PassMarks::none(),
        );
        run_dequant_matmul_gguf_gemv_rows(
            &plain,
            &dev.device,
            &dev.queue,
            m,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            plain.offset(x_in),
            plain.offset(if n == na { pa } else { pb }),
            plain.offset(y),
            &mut PassMarks::none(),
        );
        let (got, want) = (
            parked.read_f32(&dev.device, &dev.queue, y),
            plain.read_f32(&dev.device, &dev.queue, y),
        );
        let max_abs = common::max_abs_diff(&got, &want);
        eprintln!("parked vs in-arena weight, n={n}: max_abs={max_abs:e}");
        assert_signal(&want, "reference");
        assert!(
            max_abs <= 1e-6,
            "parked weight n={n}: max_abs {max_abs} against the same bytes in the act arena"
        );
    }
    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
}

/// The shape both route-movement tests drive, in their case and in every leg here:
/// `K` a whole number of this scheme's 32-element blocks (eight of them) and so of
/// the grouped GEMM's 256-element staged tile, `M > 1` past the one-row arm this
/// scheme does not have, and `N` wide enough that the 1 MiB window cap splits the
/// weight into several chunks, each resolving `w_base + n0 * row_bytes`.
const M: usize = 4;
const K: usize = 256;
const N: usize = 8192;

/// The one fact the two route-movement tests' arrangements differ in. Everything
/// else about them — the case, the plan, the whole-buffer arena and the operands
/// placed in each — is [`Arrangement`]'s, so each test's claim is about this field
/// alone.
enum Geometry {
    /// The weight in the backend's separate packed-weight buffer: the regime
    /// `from_plan_split` builds for a plan the adapter cannot bind whole. The case
    /// carries a Q4_K pad, placed first and larger, so the weight under test sits at
    /// a NON-zero weight-buffer base and its chunks resolve against that base.
    Parked,
    /// One physical buffer, `shard_size > 0` and `extra_shards` empty: the geometry
    /// `from_plan` chooses for a plan that fits one allocation but exceeds the
    /// adapter's storage-buffer binding limit.
    VirtualStripe,
}

/// The arrangement both route-movement tests drive: the case, the plan, both arenas
/// (this geometry's own and the whole-buffer copy), the operands placed in each, and
/// the device-free admission's premises asserted once. A claim in a test is about the
/// route one entry point takes on [`Self::arena`].
///
/// Holds `RLX_WGPU_MAX_BIND_MB=1` for its lifetime: the cap is what splits the weight
/// into several column chunks, so every leg resolves `w_base + n0 * row_bytes` past
/// the first one.
struct Arrangement<'a> {
    dev: &'a rlx_wgpu::device::WgpuDevice,
    case: Q8Case,
    /// The whole-buffer copy: the control, and — for [`Geometry::VirtualStripe`] —
    /// the device route this call has on an arena without the field under test.
    plain: Arena,
    /// The geometry's own arena: what every leg below is about.
    arena: Arena,
    x_off: usize,
    w_off: usize,
    y_off: usize,
    /// The slab the plan books for this graph — the admission's own term.
    booking: usize,
    scheme: u32,
}

impl<'a> Arrangement<'a> {
    fn build(dev: &'a rlx_wgpu::device::WgpuDevice, name: &str, geometry: Geometry) -> Self {
        // Both tests' claims are about the grouped device route these calls ship on.
        common::refuse_under_q8_0_bisect();
        // The pad param exists only to put a parked weight at a non-zero base.
        let pad = match geometry {
            Geometry::Parked => 16384,
            Geometry::VirtualStripe => 0,
        };
        let case = q8_case(name, M, K, N, pad);
        let plan = plan_memory_f32_uniform(&case.graph, 16);
        let plain = Arena::from_plan(&dev.device, &plan);
        let arena = match geometry {
            Geometry::Parked => parked_arena(dev, &case.graph, case.w),
            Geometry::VirtualStripe => {
                let mut arena = Arena::from_plan(&dev.device, &plan);
                // The plan is deliberately NOT snapped to stripes: what these code
                // paths read off the arena is `is_sharded()`, and that `resolve_act`
                // hands back the one physical buffer's global offset rather than a
                // per-stripe local one — both of which hold from this one field. Its
                // magnitude is only what the staging reads below are split by; no path
                // decides anything on it.
                arena.shard_size = 65536;
                arena
            }
        };
        assert!(
            !is_weight_off(plain.offset(case.w)),
            "the whole-buffer arena must hold this weight in the act arena, so the geometry \
             under test is the only fact the two placements differ in"
        );
        match geometry {
            Geometry::Parked => assert!(
                raw_weight_off(arena.offset(case.w)) > 0,
                "the weight must sit at a non-zero weight-buffer base (got {:#x})",
                arena.offset(case.w)
            ),
            Geometry::VirtualStripe => {
                assert!(
                    arena.is_sharded() && arena.extra_shards.is_empty(),
                    "this geometry is the virtual bind-stripe one: shard_size={} \
                     extra_shards={}",
                    arena.shard_size,
                    arena.extra_shards.len()
                );
                assert!(
                    !is_weight_off(arena.offset(case.w)),
                    "both arenas hold this weight in the act arena, so `is_sharded` is the \
                     only fact they differ in"
                );
            }
        }
        let (label, blind) = match geometry {
            Geometry::Parked => ("parked", "the weight's placement"),
            Geometry::VirtualStripe => ("stripe", "the arena's geometry"),
        };
        let _ = place_q8_case(dev, &case, &plain, "plain");
        let (x_off, w_off, y_off) = place_q8_case(dev, &case, &arena, label);
        let booking = assert_q8_admission(&case, M, K, x_off, plain.offset(case.x_in), blind);
        let scheme = gguf_scheme_id(QuantScheme::GgufQ8_0);
        // The cap is what splits the weight, so the chunk loop, not just the first
        // chunk, resolves `w_base + n0 * row_bytes`.
        rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "1");
        Arrangement {
            dev,
            case,
            plain,
            arena,
            x_off,
            w_off,
            y_off,
            booking,
            scheme,
        }
    }

    /// One untimed call of each device route, before the windows below: the first
    /// grouped encode builds this device's Q8_0 GEMM pipeline, and a bracketed window
    /// that long reads a neighbour's dispatches as much as this call's own.
    fn warm(&self) {
        assert!(
            self.encode_against(&self.arena, &mut PassMarks::none(), "rlx-wgpu q8_0 warm-up"),
            "the warm-up encode must be the grouped arm's own arm"
        );
        let _ = self.scratch();
    }

    /// The boundary's own encode of the call, submitted on its own: what the join
    /// branch does for the call it joins. `true` is the grouped arm reading this
    /// geometry's weight in place — for this scheme the one arm of that ladder that
    /// can answer it.
    fn encode_joined(&self) -> Leg {
        leg(self.dev, &self.arena, &self.case, |marks| {
            assert!(
                self.encode_against(&self.arena, marks, "rlx-wgpu q8_0 joined"),
                "the grouped arm must encode this call against this geometry's weight"
            );
        })
    }

    /// The whole-buffer copy through the same joined call: the control for the parked
    /// base and the chunk windows, so a difference against it is about the weight's
    /// region rather than about a route.
    fn plain_joined(&self) -> Vec<f32> {
        assert!(
            self.encode_against(
                &self.plain,
                &mut PassMarks::none(),
                "rlx-wgpu q8_0 in-arena control"
            ),
            "the grouped arm must encode the in-arena copy too"
        );
        self.plain
            .read_f32(&self.dev.device, &self.dev.queue, self.case.y)
    }

    /// Encode the call against `arena` into a fresh encoder and submit it, recording
    /// into `marks`. Returns that ladder's answer, which its own doc says no caller may
    /// discard: `false` is the refusal.
    fn encode_against(&self, arena: &Arena, marks: &mut PassMarks<'_>, label: &str) -> bool {
        let dev = self.dev;
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let ran = encode_dequant_matmul_gguf_into(
            arena,
            &dev.device,
            &dev.queue,
            &mut enc,
            M,
            K,
            N,
            self.scheme,
            arena.offset(self.case.x_in),
            arena.offset(self.case.w),
            arena.offset(self.case.y),
            marks,
        );
        dev.queue.submit(std::iter::once(enc.finish()));
        ran
    }

    /// The same call through the fused entry point the step arm takes — the route a run
    /// that does not batch (one-op-per-pass, or a snapshot) reaches.
    fn fused(&self) -> Leg {
        let dev = self.dev;
        leg(dev, &self.arena, &self.case, |marks| {
            assert!(
                run_dequant_matmul_gguf_gemv_rows(
                    &self.arena,
                    &dev.device,
                    &dev.queue,
                    M,
                    K,
                    N,
                    self.scheme,
                    self.x_off,
                    self.w_off,
                    self.y_off,
                    marks,
                ),
                "the fused entry point must run a multi-row Q8_0 call"
            );
        })
    }

    /// The scratch entry point's own route for this call.
    fn scratch(&self) -> Leg {
        let dev = self.dev;
        leg(dev, &self.arena, &self.case, |marks| {
            run_dequant_matmul_gguf_gpu(
                &self.arena,
                &dev.device,
                &dev.queue,
                M,
                K,
                N,
                self.scheme,
                self.x_off,
                self.w_off,
                self.booking,
                self.y_off,
                marks,
            );
        })
    }

    /// The whole-buffer copy through the scratch entry point: the device route this
    /// call has where the geometry's field is absent — the same arm at the same global
    /// offsets — so its census and its counted pair are both that route's.
    fn plain_route(&self) -> Leg {
        let dev = self.dev;
        leg(dev, &self.plain, &self.case, |marks| {
            run_dequant_matmul_gguf_gpu(
                &self.plain,
                &dev.device,
                &dev.queue,
                M,
                K,
                N,
                self.scheme,
                self.plain.offset(self.case.x_in),
                self.plain.offset(self.case.w),
                self.booking,
                self.plain.offset(self.case.y),
                marks,
            );
        })
    }
}

impl Drop for Arrangement<'_> {
    fn drop(&mut self) {
        rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
    }
}

/// The fixture's premise, not a reading of the admission: the graph holds exactly one
/// Q8_0 packed multiply, so the call a report line is about is one call site and every
/// figure beside it is that call's own.
fn q8_call_sites(case: &Q8Case) -> usize {
    let sites = case
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(&n.op, Op::DequantMatMul { scheme: s } if *s == QuantScheme::GgufQ8_0))
        .count();
    assert_eq!(sites, 1, "a report covers one Q8_0 call site");
    sites
}

/// The 8-bit scheme's multi-row call on a parked weight: the admission is
/// device-free, so it answers `GroupedDevice` for a placement it cannot see and
/// the call moves from the CPU route the scratch entry point's short-circuit
/// gives a parked weight to the joined row-grouped GEMM, which reads the parked
/// weight in place. Both arrangements are driven against the same arena, and the
/// `[rlx-gguf-parked-q8-0]` line this prints is their report.
#[test]
fn a_parked_weight_moves_a_multi_row_q8_0_call_to_the_grouped_route() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let _counters = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let a = Arrangement::build(dev, "wgpu_gguf_parked_q8_0", Geometry::Parked);
    // One untimed call of each route first: `warm` says why.
    a.warm();

    // The joined arrangement, driven exactly as the join branch drives it: the
    // boundary's own encode, submitted as part of its enclosing batch.
    //
    // The census below is process-global and this binary has two other tests that
    // drive the same routes (this file's Q4_K test the grouped route twice per
    // weight, the alignment guard the per-row one), so — as `gguf_matmul_route.rs`
    // notes of its own counters — only the routes no other test here takes can be
    // read as this call's own. `host` and `scratch` are those; `grouped` is a floor,
    // and the admission's own device-free answer (asserted in `Arrangement::build`)
    // is its exact source.

    let Leg {
        y: joined_y,
        routes: joined_routes,
        ..
    } = a.encode_joined();
    assert_grouped_routes(
        &joined_routes,
        "the joined call's own arm must be the grouped one, off the host and the slab",
    );

    // The same call in a run that does not batch — one-op-per-pass, or a snapshot:
    // the step arm asks the ladder the same question and takes the same fused entry
    // point, whose grouped encode answers the same way. What the counters can count
    // is this arrangement's own submission, where the boundary's submit is the
    // batch's and not this call's to count.
    let Leg {
        y: arm_y,
        routes: arm_routes,
        submits: arm_submits,
        waits: arm_waits,
    } = a.fused();
    assert_grouped_routes(
        &arm_routes,
        "the unjoined arrangement of the same call must reach the same device route, not the \
         CPU one the scratch entry point's short-circuit takes a parked weight to",
    );
    assert_eq!(
        (arm_submits, arm_waits),
        (1, 0),
        "the unjoined arrangement pays the arm's one submission and no poll of its \
         own; measured {arm_submits} submission(s), {arm_waits} wait(s)"
    );
    // Same route, same arithmetic, a different encoder: the mode changes the
    // submission arrangement and nothing else — the same identity the crate's
    // join pins assert for a K-quant multiply.
    common::assert_bit_identical(
        &joined_y,
        &arm_y,
        "the joined and the unjoined arrangement must be bit-identical",
    );

    // Today's route for the same call: the scratch entry point short-circuits a
    // parked weight to the CPU route — the regime that short-circuit exists for —
    // and pays the staging round trips that route needs.
    let Leg {
        y: today_y,
        routes: today_routes,
        submits: today_submits,
        waits: today_waits,
    } = a.scratch();
    assert_host_route(
        &today_routes,
        "the scratch entry point must keep a parked weight on the CPU route, and the grouped \
         arm that now takes it must not have run",
    );
    assert_eq!(
        (today_submits, today_waits),
        (2, 2),
        "the CPU route stages with its two ranged D2H reads, each of which submits \
         and waits; measured {today_submits} submission(s), {today_waits} wait(s)"
    );

    // The same bytes through the whole-buffer arena, encoded by the same joined
    // call: the control for the parked base and the chunk windows, so the
    // difference below is grouped-against-CPU and not a misread of the parked
    // weight region. Same claim and same bar as the Q4_K test's.
    let plain_y = a.plain_joined();
    let parked_abs = common::max_abs_diff(&joined_y, &plain_y);
    eprintln!("parked vs in-arena Q8_0 weight: max_abs={parked_abs:e}");
    assert_signal(&plain_y, "the in-arena control");
    assert!(
        parked_abs <= 1e-6,
        "the parked weight must read the same bytes as the act arena's: {parked_abs}"
    );

    let call_sites = q8_call_sites(&a.case);
    let (max_abs, rel) = movement(&joined_y, &today_y);
    // Only claimed when it holds: two routes that dequantize the same bytes with
    // a different accumulation order agree to within the bar below, and a report
    // that implied a tolerance was needed for an exact agreement would misread its
    // own numbers.
    let exact = if max_abs == 0.0 {
        " exact_bit_identity=1"
    } else {
        ""
    };
    eprintln!(
        "[rlx-gguf-parked-q8-0] call_sites={call_sites} joined_route=grouped unjoined_route=grouped \
         today_route=host joined_submits=in_enclosing_batch unjoined_submits={arm_submits} \
         unjoined_waits={arm_waits} today_submits={today_submits} today_waits={today_waits} \
         max_abs={max_abs:.3e} rel={rel:.3e}{exact}"
    );
    // The movement, in the counters that carry it. The joined arrangement's own
    // submission is the enclosing batch's — the boundary submits it, not this call —
    // so the countable pair is the unjoined arrangement's: the arm's one submission
    // and no poll of its own, against the CPU route's two staging submit+poll pairs,
    // which are its own work every round. And in the values, under the same bar.
    assert_under_cross_route_bar(max_abs, "a parked Q8_0 weight");
}

/// The virtual bind-stripe arena's movement: one physical buffer with a non-zero
/// `shard_size` and no extra shards is the geometry `from_plan` chooses for a plan
/// that fits one allocation but exceeds the adapter's storage-buffer binding limit
/// — a >2 GiB plan on this hardware — so, as for the parked regime above, the
/// arena is built directly (see the module doc). `Arena::resolve_act` hands back
/// that one buffer's GLOBAL offset, so the row-grouped GEMM's encode declines
/// nothing, and the admitted multi-row Q8_0 call runs the device arm that
/// `run_dequant_matmul_gguf_gpu`'s `is_sharded` short-circuit sends to the CPU
/// route. Both arrangements are driven against the same striped arena, and
/// `[rlx-gguf-virtual-stripe-q8-0]` is their report.
#[test]
fn a_virtual_stripe_arena_moves_a_multi_row_q8_0_call_to_the_grouped_route() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let _counters = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let a = Arrangement::build(
        dev,
        "wgpu_gguf_virtual_stripe_q8_0",
        Geometry::VirtualStripe,
    );

    // Today, on the striped arena: the scratch entry point short-circuits any
    // `is_sharded` arena to the CPU route before the ladder is even asked. That
    // route stages each read with its own ranged D2H — this arena's weight region
    // spans many of its 64 KiB stripes, so the range is stitched — and every one of
    // those submits and polls.
    let Leg {
        y: today_y,
        routes: today_routes,
        submits: today_submits,
        waits: today_waits,
    } = a.scratch();
    assert_host_route(
        &today_routes,
        "the scratch entry point must keep a sharded arena on the CPU route, and the grouped \
         arm that now takes it must not have run",
    );
    assert!(
        today_submits >= 2 && today_waits == today_submits,
        "the CPU route stages its X read and its weight read as ranged D2H reads, each \
         of which submits and polls; measured {today_submits} submission(s), \
         {today_waits} wait(s)"
    );

    // After, on the same striped arena: the fused entry point — the arm the batch
    // boundary joins — takes the row-grouped GEMM, because its X window resolves
    // through `resolve_act` to the one buffer's global offset and its weight window
    // is N-chunked inside.
    let Leg {
        y: joined_y,
        routes: joined_routes,
        submits: joined_submits,
        waits: joined_waits,
    } = a.fused();
    assert_grouped_routes(
        &joined_routes,
        "the striped arena's own device route is the grouped one, off the host and the slab",
    );
    assert_eq!(
        (joined_submits, joined_waits),
        (1, 0),
        "the grouped arm pays one submission and no poll of its own; measured \
         {joined_submits} submission(s), {joined_waits} wait(s)"
    );

    // The same call through the plain arena: the device route it already had, and
    // the control for the striped arena's grouped output. Its counted pair is the
    // one the striped arena's grouped arrangement pays — same arm, same offsets,
    // only the arena's stripe field differs.
    let Leg {
        y: plain_y,
        routes: plain_routes,
        submits: plain_submits,
        waits: plain_waits,
    } = a.plain_route();
    assert_grouped_routes(
        &plain_routes,
        "the plain arena's multi-row 8-bit call is the grouped arm it was before",
    );
    assert_eq!(
        (plain_submits, plain_waits),
        (joined_submits, joined_waits),
        "the plain arena's device route pays the pair the striped arena's grouped \
         arrangement pays; measured {plain_submits} submission(s), {plain_waits} wait(s)"
    );

    // Same kernel and the same global offsets: the striped arena must not move the
    // grouped arm's arithmetic.
    common::assert_bit_identical(
        &joined_y,
        &plain_y,
        "the virtual stripe must not change the grouped arm's output",
    );

    let (max_abs, rel) = movement(&joined_y, &today_y);

    let call_sites = q8_call_sites(&a.case);
    eprintln!(
        "[rlx-gguf-virtual-stripe-q8-0] call_sites={call_sites} today_route=host \
         joined_route=grouped grouped_vs_plain=bit_identical today_submits={today_submits} \
         today_waits={today_waits} joined_submits={joined_submits} joined_waits={joined_waits} \
         max_abs={max_abs:.3e} rel={rel:.3e} bar=1e-2"
    );
    assert_under_cross_route_bar(max_abs, "a virtual bind-stripe Q8_0 arena");
}

/// The one-row route's X-window guard, driven through the same runner as the
/// executable uses: the tiled arm reads X as `vec4`, so a window that does not
/// line up with a four-element group must go to the host instead. The reference
/// is the same bytes read from an aligned window — the device arm — so an arm
/// that read the misaligned window would take it shifted by up to three elements
/// and not match.
#[test]
fn a_misaligned_x_window_is_not_read_as_vec4() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let _counters = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let (k, n) = (256usize, 1024usize);
    let w = packed(k, n);
    // `+1 / -1` rather than a smooth ramp: an arm that reads the window one
    // element off must show it here, and consecutive samples of a smooth input
    // are too close for that.
    let x: Vec<f32> = (0..k)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (g, x_in, w_id, y) = one_weight_graph(k, n, 1, w.len());
    let arena = Arena::from_plan(&dev.device, &plan_memory_f32_uniform(&g, 16));
    arena.write_bytes_range(&dev.queue, arena.offset(w_id), &w);
    let (x_off, w_off, y_off) = (arena.offset(x_in), arena.offset(w_id), arena.offset(y));
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    // X four bytes past its slot: a 4-byte-aligned window of the same tensor.
    let mis = x_off + 4;
    assert!(
        mis + x_bytes.len() <= arena.buffer.size() as usize,
        "X's slot needs four spare bytes to shift its window"
    );
    arena.write_bytes_range(&dev.queue, mis, &x_bytes);
    run_dequant_matmul_gguf_gemv_rows(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        mis,
        w_off,
        y_off,
        &mut PassMarks::none(),
    );
    let via_host = arena.read_f32(&dev.device, &dev.queue, y);

    arena.write_bytes_range(&dev.queue, x_off, &x_bytes);
    run_dequant_matmul_gguf_gemv_rows(
        &arena,
        &dev.device,
        &dev.queue,
        1,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        x_off,
        w_off,
        y_off,
        &mut PassMarks::none(),
    );
    let on_device = arena.read_f32(&dev.device, &dev.queue, y);

    assert_signal(&on_device, "the aligned reference");
    let max_abs = common::max_abs_diff(&via_host, &on_device);
    let scale = magnitude(&on_device);
    eprintln!("misaligned X window vs aligned: max_abs={max_abs:e} ref_scale={scale:e}");
    assert!(
        max_abs <= 1e-3,
        "a misaligned X window must not be read as vec4: max_abs {max_abs}"
    );
}

/// The same exercise at the size the harness uses — 202048 × 7168 Q4_K = 777 MiB
/// parked, four bind windows, where the weight is far larger than one window —
/// checked against the per-row GEMV it replaces, which in a parked arena binds
/// the weight buffer directly too.
///
/// `#[ignore]`d: it materialises ~800 MiB of packed weights next to the device
/// copy and builds two arenas over them.
#[test]
#[ignore = "~800 MiB of packed weights; run with --ignored --nocapture"]
fn harness_size_parked_weight_matches_the_per_row_path() {
    let Some(dev) = wgpu_device() else {
        eprintln!("skip: wgpu unavailable");
        return;
    };
    let _counters = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let (k, n, m) = (7168usize, 202048usize, 4usize);
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", "256");
    let packed = packed(k, n);
    let x = common::ramp(m, k, 1e-3);

    let (g, x_in, w, y) = one_weight_graph(k, n, m, packed.len());
    let parked = parked_arena(dev, &g, w);
    parked.write_bytes_range(&dev.queue, parked.offset(w), &packed);
    parked.write_f32(&dev.queue, x_in, &x);
    run_dequant_matmul_gguf_gemv_rows(
        &parked,
        &dev.device,
        &dev.queue,
        m,
        k,
        n,
        gguf_scheme_id(QuantScheme::GgufQ4K),
        parked.offset(x_in),
        parked.offset(w),
        parked.offset(y),
        &mut PassMarks::none(),
    );
    let grouped = parked.read_f32(&dev.device, &dev.queue, y);

    // Per-row reference: a graph of its own (m == 1 picks the GEMV), one arena
    // reused for every row by rewriting just the activation.
    let (g1, x1, w1, y1) = one_weight_graph(k, n, 1, packed.len());
    let rows = parked_arena(dev, &g1, w1);
    rows.write_bytes_range(&dev.queue, rows.offset(w1), &packed);
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for r in 0..m {
        rows.write_f32(&dev.queue, x1, &x[r * k..(r + 1) * k]);
        run_dequant_matmul_gguf_gemv_rows(
            &rows,
            &dev.device,
            &dev.queue,
            1,
            k,
            n,
            gguf_scheme_id(QuantScheme::GgufQ4K),
            rows.offset(x1),
            rows.offset(w1),
            rows.offset(y1),
            &mut PassMarks::none(),
        );
        let one = rows.read_f32(&dev.device, &dev.queue, y1);
        assert_signal(&one, "the per-row arm's reference");
        scale = scale.max(magnitude(&one));
        worst = worst.max(common::max_abs_diff(&grouped[r * n..(r + 1) * n], &one));
    }
    rlx_ir::env::unset("RLX_WGPU_MAX_BIND_MB");
    eprintln!(
        "parked 777 MiB weight, grouped m={m} vs per-row at k={k} n={n}: \
         max_abs={worst:e} (per-row |y|max={scale:e})"
    );
    assert!(scale > 1.0, "per-row reference is degenerate ({scale:e})");
    assert!(worst <= 1e-2, "grouped vs per-row max_abs {worst:e} > 1e-2");
}
