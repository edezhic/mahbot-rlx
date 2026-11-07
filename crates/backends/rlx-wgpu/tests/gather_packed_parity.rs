// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `mahbot.gather_packed`: rows out of a packed GGUF Q8_0 table, dequantised to
//! f32 on device — one dispatch, no host round trip and no f32 table in between.
//!
//! The comparison is bit-for-bit against `rlx_gguf::dequant_q8_0`: Q8_0's device
//! decode is `d * f32(q)` on a f16 scale both sides widen exactly, so a difference
//! is a defect in the kernel, not a rounding budget. The indices are chosen so the
//! bias fold, the clamp and a raw negative cannot be confused for one another, and
//! the two bindings — the table in the arena, and the table parked in the weight
//! buffer — are each run and compared.

mod common;

use common::{cleared_switches, packed_gguf_weight, rerun_self, route_lock};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::{device_submit_count, device_wait_count, reset_device_counts};

/// The op's own name, as `crates/mahbot-inference` emits it.
const OP: &str = "mahbot.gather_packed";

/// The child re-exec marker: set in the child, printed by it and waited for by
/// `common::rerun_self`. Not an `RLX_*` name — it selects a fixture, it does not
/// configure the backend, so it stays out of the env registry.
const MARKER: &str = "GATHER_PACKED_PARKED";

/// Row width: two Q8_0 blocks, so a row is 68 bytes and the block stride within
/// a row is exercised rather than collapsing to one block.
const COLS: usize = 64;
const ROWS: usize = 5;

/// Added to every index before truncation.
const BIAS: u32 = 1;

/// `-1 + 1 = 0` (row 0, reachable only through the bias), `0 + 1 = 1` (a lost
/// bias answers row 0), `1 + 1 = 2`, `3 + 1 = 4` (the last row), `7 + 1 = 8`
/// (clamped to the last row), `2 + 1 = 3`.
const INDICES: [f32; 6] = [-1.0, 0.0, 1.0, 3.0, 7.0, 2.0];

/// The rows the kernel must select, derived here rather than read back from it:
/// `idx_u = min(u32(max(idx + bias, 0)), rows - 1)`.
fn selected_rows(indices: &[f32], bias: u32) -> Vec<usize> {
    indices
        .iter()
        .map(|&i| (i + bias as f32).max(0.0) as usize)
        .map(|i| i.min(ROWS - 1))
        .collect()
}

/// The fixture table: `[ROWS, COLS]` of Q8_0 blocks, laid out by hand by
/// `common::packed_gguf_weight` (its own doc says why the bytes are not
/// quantized from an f32 tensor).
fn packed_table() -> Vec<u8> {
    packed_gguf_weight(QuantScheme::GgufQ8_0, COLS, ROWS)
}

/// `[cols, rows, bias]`, little-endian u32 each — the contract. The scheme is not on
/// the wire: the route serves Q8_0 and nothing else.
fn attrs(cols: usize, rows: usize, bias: u32) -> Vec<u8> {
    [cols as u32, rows as u32, bias]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// One `mahbot.gather_packed` node: a U8 table `Param`, the f32-encoded indices
/// as an input, and a `[n_idx, COLS]` f32 output.
fn gather_packed(
    g: &mut Graph,
    n_idx: usize,
    bias: u32,
    table: rlx_ir::NodeId,
    idx: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    g.add_node(
        Op::Custom {
            name: OP.to_string(),
            num_inputs: 2,
            attrs: attrs(COLS, ROWS, bias),
        },
        vec![table, idx],
        Shape::new(&[n_idx, COLS], DType::F32),
    )
}

/// The parity graph: one gather, the table a `U8` parameter. Nothing here selects
/// the split plan, so the arena keeps the table and bindings 0 and 2 point into one
/// buffer (see the module doc).
fn graph(table_bytes: usize) -> Graph {
    let mut g = Graph::new("gather_packed");
    let table = g.param("table", Shape::new(&[table_bytes], DType::U8));
    let idx = g.input("idx", Shape::new(&[INDICES.len()], DType::F32));
    let out = gather_packed(&mut g, INDICES.len(), BIAS, table, idx);
    g.set_outputs(vec![out]);
    g
}

/// The reference: `rlx_gguf`'s Q8_0 dequant of the whole table, then the rows the
/// indices select.
fn reference(packed: &[u8]) -> Vec<f32> {
    let dequant = rlx_gguf::dequant_q8_0(packed, ROWS * COLS).expect("Q8_0 fixture dequantizes");
    let mut want = Vec::with_capacity(INDICES.len() * COLS);
    for row in selected_rows(&INDICES, BIAS) {
        want.extend_from_slice(&dequant[row * COLS..(row + 1) * COLS]);
    }
    want
}

/// The same comparison, element by element.
fn assert_is_the_reference(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "output element count");
    let rows = selected_rows(&INDICES, BIAS);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {i} (row {}) is not rlx_gguf's Q8_0 dequant: {a:e} vs {b:e}",
            rows[i / COLS]
        );
    }
}

#[test]
fn every_output_element_is_the_dequantized_row_the_index_selects() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let packed = packed_table();
    let mut s = Session::new(Device::Gpu).compile(graph(packed.len()));
    s.set_param_typed("table", &packed, DType::U8);
    let names = s.schedule_step_names().expect("wgpu reports its schedule");
    assert_eq!(
        names.iter().filter(|n| **n == "gather_packed").count(),
        1,
        "the op must lower to the device route, not fall back: {names:?}"
    );
    let got = s.run(&[("idx", &INDICES)]).pop().unwrap_or_default();
    assert_is_the_reference(&got, &reference(&packed));
}

/// The same rows out of a table parked in the **weight buffer** — the arm the
/// product's frame graph takes. `RLX_WGPU_ALL_PARAMS_WEIGHT=1` is what builds the
/// split plan, and inside it every packed `U8` parameter is parked; the switch is
/// read at compile time, so the case owns it by re-running itself with it and
/// everything else cleared. The child's `[rlx-wgpu split]` line is asserted below,
/// so a switch that stops parking a lone `Param` fails here instead of quietly
/// re-running the arena arm.
#[test]
fn a_parked_table_reads_the_same_rows() {
    // The device is checked in the parent: a child spawned without one prints its
    // marker and skips, which would read as a parked-arm failure here.
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    if std::env::var_os(MARKER).is_none() {
        let (ok, log) = rerun_self(
            "a_parked_table_reads_the_same_rows",
            MARKER,
            "1",
            &cleared_switches(),
            &[("RLX_WGPU_ALL_PARAMS_WEIGHT", "1"), ("RLX_WGPU_DEBUG", "1")],
        );
        assert!(ok, "the parked-table case failed: {log}");
        // The backend's own split report: the table is the graph's only parameter,
        // so `weight_params=1` is it, in the weight buffer.
        assert!(
            log.contains("weight_params=1") && log.contains("all_params=true"),
            "the child compiled without parking the table, so it re-ran the arena arm:\n{log}"
        );
        return;
    }
    // The sentinel `common::rerun_self` waits for: it proves the child ran this test.
    eprintln!("{MARKER} 1");
    let _guard = route_lock();

    let packed = packed_table();
    let mut s = Session::new(Device::Gpu).compile(graph(packed.len()));
    s.set_param_typed("table", &packed, DType::U8);
    let names = s.schedule_step_names().expect("wgpu reports its schedule");
    assert_eq!(
        names.iter().filter(|n| **n == "gather_packed").count(),
        1,
        "the parked table takes the same device step: {names:?}"
    );
    let got = s.run(&[("idx", &INDICES)]).pop().unwrap_or_default();
    assert_is_the_reference(&got, &reference(&packed));
}

#[test]
fn four_gathers_are_one_device_batch_with_one_wait_and_no_host_step() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    let packed = packed_table();
    let mut g = Graph::new("gather_packed_x4");
    let table = g.param("table", Shape::new(&[packed.len()], DType::U8));
    let idx = g.input("idx", Shape::new(&[INDICES.len()], DType::F32));
    // Four gathers that cannot be folded into one: each reads a different row of
    // the table, so each is its own step.
    let parts: Vec<_> = (0..4)
        .map(|k| gather_packed(&mut g, INDICES.len(), k, table, idx))
        .collect();
    let left = g.add(parts[0], parts[1]);
    let right = g.add(parts[2], parts[3]);
    let sum = g.add(left, right);
    g.set_outputs(vec![sum]);

    let mut s = Session::new(Device::Gpu).compile(g);
    s.set_param_typed("table", &packed, DType::U8);
    let names = s.schedule_step_names().expect("wgpu reports its schedule");
    assert_eq!(
        names.iter().filter(|n| **n == "gather_packed").count(),
        4,
        "every gather is its own device step: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("host")),
        "no gather may leave the device: {names:?}"
    );

    reset_device_counts();
    let out = s.run(&[("idx", &INDICES)]).pop().unwrap_or_default();
    let (submits, waits) = (device_submit_count(), device_wait_count());
    assert_eq!(
        waits, 1,
        "the whole run is one device batch and one wait (submits={submits}, names={names:?})"
    );
    assert!(!out.is_empty(), "the summed gathers produced no output");
}

/// The message a refused compile panicked with. The route's name has to be in it:
/// a caller can only change what it is told to change.
fn refusal<T>(compile: impl FnOnce() -> T) -> String {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(compile));
    let Err(payload) = caught else {
        panic!("the compile accepted a table it should have refused");
    };
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn a_table_whose_declared_bytes_disagree_with_its_geometry_is_refused_by_name() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    // `rows * row_bytes` for the attrs is 5 * 68 = 340 bytes; declare 336, the
    // length a table one block short would have.
    let packed = packed_table();
    let msg = refusal(|| WgpuExecutable::compile(graph(packed.len() - 4)));
    assert!(
        msg.contains(OP),
        "the refusal must name the route a caller has to change; got: {msg}"
    );
    assert!(
        msg.contains("340") && msg.contains("336"),
        "the refusal must quote the geometry and the declared bytes; got: {msg}"
    );
}

#[test]
fn a_table_that_is_not_u8_is_refused_by_name() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    // An f32 table declared at exactly `rows * row_bytes` elements passes the
    // byte-count guard (its element count *is* that number) and would be read as
    // Q8_0 blocks — the silent misread this route exists to prevent.
    let mut g = Graph::new("gather_packed_f32_table");
    let table = g.param("table", Shape::new(&[ROWS * COLS], DType::F32));
    let idx = g.input("idx", Shape::new(&[1], DType::F32));
    let out = g.add_node(
        Op::Custom {
            name: OP.to_string(),
            num_inputs: 2,
            attrs: attrs(COLS, ROWS, 0),
        },
        vec![table, idx],
        Shape::new(&[1, COLS], DType::F32),
    );
    g.set_outputs(vec![out]);

    let msg = refusal(|| WgpuExecutable::compile(g));
    assert!(
        msg.contains(OP) && msg.contains("U8"),
        "a non-U8 table must be refused by name; got: {msg}"
    );
}

#[test]
fn a_row_that_is_not_whole_q8_0_blocks_is_refused_by_name() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    // 100 elements is three blocks and four over: `gguf_row_bytes` truncates to
    // 102, which is exactly what a one-row table of that length declares, so the
    // byte-count check alone would pass it and every output row's tail would be
    // stale. The width has to be refused on its own.
    const RAGGED: usize = 100;
    let mut g = Graph::new("gather_packed_ragged");
    let table = g.param("table", Shape::new(&[3 * 34], DType::U8));
    let idx = g.input("idx", Shape::new(&[1], DType::F32));
    let out = g.add_node(
        Op::Custom {
            name: OP.to_string(),
            num_inputs: 2,
            attrs: attrs(RAGGED, 1, 0),
        },
        vec![table, idx],
        Shape::new(&[1, RAGGED], DType::F32),
    );
    g.set_outputs(vec![out]);

    let msg = refusal(|| WgpuExecutable::compile(g));
    assert!(
        msg.contains(OP) && msg.contains("whole number of Q8_0 blocks"),
        "a ragged row must be refused by name; got: {msg}"
    );
}

#[test]
fn a_node_that_declares_a_smaller_output_than_it_gathers_is_refused_by_name() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let _guard = route_lock();

    // The dispatch writes `n_idx * COLS` f32 words into the node's planned slot;
    // a node declaring one row fewer would spill a whole row into the next
    // planned occupant of the arena.
    let packed = packed_table();
    let mut g = Graph::new("gather_packed_short_output");
    let table = g.param("table", Shape::new(&[packed.len()], DType::U8));
    let idx = g.input("idx", Shape::new(&[INDICES.len()], DType::F32));
    let out = g.add_node(
        Op::Custom {
            name: OP.to_string(),
            num_inputs: 2,
            attrs: attrs(COLS, ROWS, BIAS),
        },
        vec![table, idx],
        Shape::new(&[INDICES.len() - 1, COLS], DType::F32),
    );
    g.set_outputs(vec![out]);

    let msg = refusal(|| WgpuExecutable::compile(g));
    assert!(
        msg.contains(OP) && msg.contains("output shape"),
        "a short output must be refused by name; got: {msg}"
    );
}
