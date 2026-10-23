// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A KV append's row is written where the attention reads it, not copied there.
//!
//! `Op::KvAppend` owns the cache's spare row: the producer of its row operand can
//! write at `cache + pos · row` instead of into a slot of its own, and then the
//! append has no step at all. The placement's refusals are exercised here — a
//! producer that is an input/parameter/constant, a cache whose own write is
//! ordered after the producer, a root longer than one row, two appends naming one
//! producer, and two appends naming one position — because a silent relocation is
//! exactly the class of defect the arrangement exists to remove. The striped-arena
//! refusal is a predicate's
//! (`Arena::binds_whole_at_base_zero`), not a schedule's: on that geometry the
//! append is hosted whatever the placement answers, so its case at the end pins
//! the geometry's outcome rather than the refusal.

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

mod common;

const ROWS: usize = 5;
const W: usize = 8;
/// The cache's spare row: `cache[4, :]`.
const POS: usize = 4;

/// Two caches, each with a row appended at its spare row. `shared` names one
/// producer for both appends (which the placement refuses); `from_input` makes
/// each row the graph input itself (also refused — an input has no step to
/// move).
fn graph(shared: bool, from_input: bool) -> Graph {
    let mut g = Graph::new("kv_append_in_place");
    let cache = g.input("cache", Shape::new(&[ROWS, W], DType::F32));
    let cache2 = g.input("cache2", Shape::new(&[ROWS, W], DType::F32));
    let x = g.input("x", Shape::new(&[1, W], DType::F32));
    let y = g.input("y", Shape::new(&[1, W], DType::F32));
    let row = |g: &mut Graph, src| {
        if from_input {
            src
        } else {
            g.add_node(
                Op::Activation(Activation::Relu),
                vec![src],
                Shape::new(&[1, W], DType::F32),
            )
        }
    };
    let r1 = row(&mut g, x);
    let r2 = if shared { r1 } else { row(&mut g, y) };
    let out = g.add_node(
        Op::KvAppend { axis: 0, pos: POS },
        vec![cache, r1],
        Shape::new(&[ROWS, W], DType::F32),
    );
    let out2 = g.add_node(
        Op::KvAppend { axis: 0, pos: POS },
        vec![cache2, r2],
        Shape::new(&[ROWS, W], DType::F32),
    );
    g.set_outputs(vec![out, out2]);
    g
}

/// One cache and one append whose row is a `[1, W]` narrow of a **two-row**
/// producer, at `start`: `start = 1` puts the row off its root's start, `start = 0`
/// gives it a root two rows long. Both are refused, and the copy stays.
fn graph_with_two_row_root(start: usize) -> Graph {
    let mut g = Graph::new("kv_append_two_row_root");
    let cache = g.input("cache", Shape::new(&[ROWS, W], DType::F32));
    let x = g.input("x", Shape::new(&[2, W], DType::F32));
    let wide = g.add_node(
        Op::Activation(Activation::Relu),
        vec![x],
        Shape::new(&[2, W], DType::F32),
    );
    let row = g.add_node(
        Op::Narrow {
            axis: 0,
            start,
            len: 1,
        },
        vec![wide],
        Shape::new(&[1, W], DType::F32),
    );
    let out = g.add_node(
        Op::KvAppend { axis: 0, pos: POS },
        vec![cache, row],
        Shape::new(&[ROWS, W], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

fn ramp(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| base + i as f32).collect()
}

/// `cache` with its spare row replaced by `row` (relu'd unless the row is the
/// input itself).
fn expected(cache_base: f32, row: &[f32], relu: bool) -> Vec<f32> {
    let mut want = ramp(ROWS * W, cache_base);
    want[POS * W..(POS + 1) * W]
        .iter_mut()
        .zip(row)
        .for_each(|(w, v)| *w = if relu { v.max(0.0) } else { *v });
    want
}

/// Every placement outcome — one spelling for all of them: the schedule keeps
/// exactly `copies` row copies (`0` where the placement is taken), and the device's
/// answers are what the schedule produces.
fn kept(g: Graph, inputs: &[(&str, &[f32])], copies: usize, why: &str) -> Vec<Vec<f32>> {
    let mut exe = WgpuExecutable::compile(g);
    let steps = exe.test_step_names();
    let got = steps.iter().filter(|s| **s == "buffer_copy").count();
    assert_eq!(got, copies, "{why}: {steps:?}");
    exe.run(inputs)
}

#[test]
fn the_row_producer_writes_the_spare_row_and_the_copy_steps_disappear() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 3.0).collect();
    let y: Vec<f32> = (0..W).map(|i| 2.0 - i as f32).collect();
    let out = kept(
        graph(false, false),
        &[
            ("cache", &ramp(ROWS * W, 0.0)),
            ("cache2", &ramp(ROWS * W, 100.0)),
            ("x", &x),
            ("y", &y),
        ],
        0,
        "each append's row is already where its cache reads it, so no copy is scheduled",
    );
    assert_eq!(out[0], expected(0.0, &x, true));
    assert_eq!(out[1], expected(100.0, &y, true));
}

#[test]
fn a_refused_placement_keeps_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 2.0).collect();
    let y: Vec<f32> = (0..W).map(|i| 1.0 - i as f32).collect();
    // 1 — the rows are graph inputs, so their bytes are uploaded and there is no
    //     producer step that could write the cache's spare row.
    let out = kept(
        graph(false, true),
        &[
            ("cache", &ramp(ROWS * W, 0.0)),
            ("cache2", &ramp(ROWS * W, 100.0)),
            ("x", &x),
            ("y", &y),
        ],
        2,
        "a row that is a graph input cannot be placed in the cache, so both appends \
         stay copies",
    );
    assert_eq!(out[0], expected(0.0, &x, false));
    assert_eq!(out[1], expected(100.0, &y, false));
    // 2 — one producer named by two appends cannot land in two spare rows, so
    //     both keep their copy rather than one of them being misplaced.
    let out = kept(
        graph(true, false),
        &[
            ("cache", &ramp(ROWS * W, 0.0)),
            ("cache2", &ramp(ROWS * W, 100.0)),
            ("x", &x),
            ("y", &y),
        ],
        2,
        "two appends of one producer must both keep their copy",
    );
    assert_eq!(out[0], expected(0.0, &x, true));
    assert_eq!(out[1], expected(100.0, &x, true));
}

/// A row whose root is longer than one row cannot be moved, whichever part of the
/// root it is: the producer's tail would reach the cache's next row. (A view off
/// its root's start is the same refusal, not a second one — a one-row view that
/// does not start at its root implies a root longer than a row.) The copy is the
/// only correct form, and the values are the copy's.
#[test]
fn a_root_longer_than_one_row_keeps_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let xs: Vec<f32> = (0..2 * W).map(|i| (i as f32) * 0.5 - 4.0).collect();
    for start in [0usize, 1] {
        let out = kept(
            graph_with_two_row_root(start),
            &[("cache", &ramp(ROWS * W, 0.0)), ("x", &xs)],
            1,
            "a two-row producer cannot be moved: its tail would reach the next row",
        );
        assert_eq!(
            out[0],
            expected(0.0, &xs[start * W..(start + 1) * W], true),
            "the copy must move that row and leave the rest of the cache"
        );
    }
}

/// A cache that a step of its own writes is only placeable when that step is
/// **ordered before** the row's producer: `cache = Relu(cache_in)` writing the
/// whole span after the producer would overwrite the placed row (the copy, at the
/// append, would have survived it). Built the other way round the move is the
/// copy's own write, and the values are the same in both orders.
#[test]
fn a_cache_written_after_the_producer_keeps_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let n = ROWS * W;
    let cache_values: Vec<f32> = (0..n).map(|i| i as f32 - 12.0).collect();
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 3.0).collect();
    let mut want: Vec<f32> = cache_values.iter().map(|v| v.max(0.0)).collect();
    for (w, v) in want[POS * W..(POS + 1) * W].iter_mut().zip(&x) {
        *w = v.max(0.0);
    }
    for cache_first in [false, true] {
        let mut g = Graph::new("kv_append_computed_cache");
        let cache_in = g.input("cache_in", Shape::new(&[ROWS, W], DType::F32));
        let xi = g.input("x", Shape::new(&[1, W], DType::F32));
        let relu = |g: &mut Graph, src, shape| {
            g.add_node(Op::Activation(Activation::Relu), vec![src], shape)
        };
        let row_of = Shape::new(&[1, W], DType::F32);
        let whole = || Shape::new(&[ROWS, W], DType::F32);
        let (cache, row) = if cache_first {
            let cache = relu(&mut g, cache_in, whole());
            (cache, relu(&mut g, xi, row_of))
        } else {
            let row = relu(&mut g, xi, row_of);
            (relu(&mut g, cache_in, whole()), row)
        };
        let out = g.add_node(
            Op::KvAppend { axis: 0, pos: POS },
            vec![cache, row],
            whole(),
        );
        g.set_outputs(vec![out]);
        let why = if cache_first {
            "the cache's own write is before the producer, so the row is the copy's own write"
        } else {
            "the cache's own write is after the producer, which would overwrite the row"
        };
        let out = kept(
            g,
            &[("cache_in", &cache_values), ("x", &x)],
            usize::from(!cache_first),
            why,
        );
        assert_eq!(out[0], want, "the relu'd cache with the appended row");
    }
}

/// Two appends naming one position write one row: their producers' step order is
/// not the appends' order, so a placement can leave the *earlier* producer's row
/// in the cache. Both keep their copy and the row is the later append's operand —
/// also when it is the **first** append that another term refuses, which is why
/// the position is decided over every append rather than as appends are accepted.
#[test]
fn two_appends_at_one_position_keep_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 3.0).collect();
    let y: Vec<f32> = (0..W).map(|i| 2.0 - i as f32).collect();
    let whole = || Shape::new(&[ROWS, W], DType::F32);
    let row_of = || Shape::new(&[1, W], DType::F32);
    let mut g = Graph::new("kv_append_duplicate_pos");
    let cache = g.input("cache", whole());
    let xi = g.input("x", row_of());
    let yi = g.input("y", row_of());
    let a = g.add_node(Op::Activation(Activation::Relu), vec![xi], row_of());
    let b = g.add_node(Op::Activation(Activation::Relu), vec![yi], row_of());
    // The appends name their producers in the opposite order to the steps that
    // write them: the first append takes the later producer.
    let one = g.add_node(Op::KvAppend { axis: 0, pos: POS }, vec![cache, b], whole());
    let two = g.add_node(Op::KvAppend { axis: 0, pos: POS }, vec![one, a], whole());
    g.set_outputs(vec![two]);
    let out = kept(
        g,
        &[("cache", &ramp(ROWS * W, 0.0)), ("x", &x), ("y", &y)],
        2,
        "one row cannot hold two producers' writes, so both appends keep their copy",
    );
    assert_eq!(
        out[0],
        expected(0.0, &x, true),
        "the later append's row is the one the cache carries"
    );

    // The first append's row is a graph input, so *it* is refused for that
    // reason. Were the position marked only by an accepted append, the second
    // append would be placed and the first one's kept copy would write over it.
    let mut g = Graph::new("kv_append_duplicate_pos_refused_first");
    let cache = g.input("cache", whole());
    let xi = g.input("x", row_of());
    let yi = g.input("y", row_of());
    let b = g.add_node(Op::Activation(Activation::Relu), vec![yi], row_of());
    let one = g.add_node(Op::KvAppend { axis: 0, pos: POS }, vec![cache, xi], whole());
    let two = g.add_node(Op::KvAppend { axis: 0, pos: POS }, vec![one, b], whole());
    g.set_outputs(vec![two]);
    let out = kept(
        g,
        &[("cache", &ramp(ROWS * W, 0.0)), ("x", &x), ("y", &y)],
        2,
        "both appends of one position keep their copy, whichever of them another \
         term also refuses",
    );
    assert_eq!(
        out[0],
        expected(0.0, &y, true),
        "the later append's row is the one the cache carries, not the refused \
         first one's"
    );
}

/// Two appends on **differently shaped views of one root** can name one
/// destination interval without sharing a position or an axis: a four-row slice
/// of an `[8, 8]` root written at `pos = 3` and a `[32, 2]` reshape of the same
/// root written at `pos = 12` both land at byte 96. Neither may be placed —
/// whichever kept its copy would write the bytes the other's producer moved.
#[test]
fn two_views_of_one_root_that_land_on_one_row_keep_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..8).map(|i| (i as f32) - 3.0).collect();
    let y = vec![9.5f32, -9.5];
    let mut g = Graph::new("kv_append_two_views_one_row");
    let cache = g.input("cache", Shape::new(&[8, 8], DType::F32));
    let xi = g.input("x", Shape::new(&[1, 8], DType::F32));
    let yi = g.input("y", Shape::new(&[1, 2], DType::F32));
    let four = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: 4,
        },
        vec![cache],
        Shape::new(&[4, 8], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![32, 2],
        },
        vec![cache],
        Shape::new(&[32, 2], DType::F32),
    );
    let row_a = g.add_node(
        Op::Activation(Activation::Relu),
        vec![xi],
        Shape::new(&[1, 8], DType::F32),
    );
    let row_b = g.add_node(
        Op::Activation(Activation::Relu),
        vec![yi],
        Shape::new(&[1, 2], DType::F32),
    );
    let a = g.add_node(
        Op::KvAppend { axis: 0, pos: 3 },
        vec![four, row_a],
        Shape::new(&[4, 8], DType::F32),
    );
    let b = g.add_node(
        Op::KvAppend { axis: 0, pos: 12 },
        vec![flat, row_b],
        Shape::new(&[13, 2], DType::F32),
    );
    g.set_outputs(vec![a, b]);
    let mut out = kept(
        g,
        &[("cache", &ramp(64, 0.0)), ("x", &x), ("y", &y)],
        2,
        "both appends write the same bytes, so both keep their copy",
    );
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    // Both copies write byte 96 of the root, so the later append's row is what
    // the cache carries where they meet and the rest of the four-row write stands.
    let mut want_a = ramp(64, 0.0)[..32].to_vec();
    for (w, v) in want_a[24..26].iter_mut().zip(&y) {
        *w = v.max(0.0);
    }
    for (w, v) in want_a[26..32].iter_mut().zip(&x[2..]) {
        *w = v.max(0.0);
    }
    assert_eq!(a, want_a);
    let mut want_b = ramp(64, 0.0)[..26].to_vec();
    for (w, v) in want_b[24..26].iter_mut().zip(&y) {
        *w = v.max(0.0);
    }
    assert_eq!(b, want_b);
}

/// A reader ordered between the row's producer and the append sees the cache as
/// it stood before the append. The move writes the row in the producer's own
/// dispatch, so such a reader would see it early — the copy therefore stays.
#[test]
fn a_reader_between_the_producer_and_the_append_keeps_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 3.0).collect();
    let mut g = Graph::new("kv_append_intermediate_reader");
    let cache = g.input("cache", Shape::new(&[ROWS, W], DType::F32));
    let xi = g.input("x", Shape::new(&[1, W], DType::F32));
    let row = g.add_node(
        Op::Activation(Activation::Relu),
        vec![xi],
        Shape::new(&[1, W], DType::F32),
    );
    // Reads the cache between the producer and the append.
    let before = g.add_node(
        Op::Activation(Activation::Relu),
        vec![cache],
        Shape::new(&[ROWS, W], DType::F32),
    );
    let out = g.add_node(
        Op::KvAppend { axis: 0, pos: POS },
        vec![cache, row],
        Shape::new(&[ROWS, W], DType::F32),
    );
    g.set_outputs(vec![out, before]);
    let mut out = kept(
        g,
        &[("cache", &ramp(ROWS * W, 0.0)), ("x", &x)],
        1,
        "a reader stands between the producer and the append, so the row is copied \
         there",
    );
    let before = out.pop().unwrap();
    let appended = out.pop().unwrap();
    assert_eq!(appended, expected(0.0, &x, true));
    assert_eq!(
        before,
        ramp(ROWS * W, 0.0),
        "the reader sees the cache before the row is appended"
    );
}

/// Two geometries the placement must get right for the destination to be the
/// lowering's own: a **batched** row (a rank-3 cache written along axis 0, where a
/// row is more than one `[W]` vector), and a cache that is a **view into a wider
/// root** — the case where the append's offset and its storage root's differ, so a
/// width or a base taken from the root rather than from the operand would land the
/// row in the wrong place while every containment check still passed.
#[test]
fn a_batched_cache_and_a_view_cache_take_the_row_in_place() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..2 * W).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let mut g = Graph::new("kv_append_batched");
    let cache = g.input("cache", Shape::new(&[ROWS, 2, W], DType::F32));
    let xi = g.input("x", Shape::new(&[1, 2, W], DType::F32));
    let row = g.add_node(
        Op::Activation(Activation::Relu),
        vec![xi],
        Shape::new(&[1, 2, W], DType::F32),
    );
    let out = g.add_node(
        Op::KvAppend { axis: 0, pos: POS },
        vec![cache, row],
        Shape::new(&[ROWS, 2, W], DType::F32),
    );
    g.set_outputs(vec![out]);
    let mut exe = WgpuExecutable::compile(g);
    let steps = exe.test_step_names();
    assert!(
        !steps.contains(&"buffer_copy"),
        "a batched row is still one row and is written in place: {steps:?}"
    );
    let got = exe
        .run(&[("cache", &ramp(ROWS * 2 * W, 0.0)), ("x", &x)])
        .remove(0);
    let mut want = ramp(ROWS * 2 * W, 0.0);
    for (w, v) in want[POS * 2 * W..(POS + 1) * 2 * W].iter_mut().zip(&x) {
        *w = v.max(0.0);
    }
    assert_eq!(
        got, want,
        "the batched cache keeps every row but the spare one"
    );

    // The cache is a **view into a wider root** whose row is not the root's row:
    // the second half of `root`, reshaped so its trailing extent is twice the
    // root's. A width taken from the root rather than from the operand would land
    // the row at half the offset — while every containment check still passed.
    // (A column slice would be the same disagreement, but the planner materialises
    // a `Narrow` off axis 0, so it is not a view the placement can see.)
    const VR: usize = 4;
    let mut g = Graph::new("kv_append_view_cache");
    let root = g.input("root", Shape::new(&[2 * VR, W], DType::F32));
    let view = g.add_node(
        Op::Narrow {
            axis: 0,
            start: VR,
            len: VR,
        },
        vec![root],
        Shape::new(&[VR, W], DType::F32),
    );
    let cache = g.add_node(
        Op::Reshape {
            new_shape: vec![(VR / 2) as i64, (2 * W) as i64],
        },
        vec![view],
        Shape::new(&[VR / 2, 2 * W], DType::F32),
    );
    let xi = g.input("x", Shape::new(&[1, 2 * W], DType::F32));
    let row = g.add_node(
        Op::Activation(Activation::Relu),
        vec![xi],
        Shape::new(&[1, 2 * W], DType::F32),
    );
    let out = g.add_node(
        Op::KvAppend {
            axis: 0,
            pos: VR / 2 - 1,
        },
        vec![cache, row],
        Shape::new(&[VR / 2, 2 * W], DType::F32),
    );
    g.set_outputs(vec![out, root]);
    let mut exe = WgpuExecutable::compile(g);
    let steps = exe.test_step_names();
    assert!(
        !steps.contains(&"buffer_copy"),
        "a cache that is a view is still addressable by its own offset: {steps:?}"
    );
    let xv: Vec<f32> = (0..2 * W).map(|i| (i as f32) * 0.5 - 5.0).collect();
    let mut outs = exe.run(&[("root", &ramp(2 * VR * W, 0.0)), ("x", &xv)]);
    let root_out = outs.pop().unwrap();
    let cache_out = outs.pop().unwrap();
    let mut want_cache = ramp(VR * W, (VR * W) as f32);
    for (w, v) in want_cache[(VR / 2 - 1) * 2 * W..].iter_mut().zip(&xv) {
        *w = v.max(0.0);
    }
    assert_eq!(
        cache_out, want_cache,
        "the view's rows, its last one rewritten"
    );
    let mut want_root = ramp(2 * VR * W, 0.0);
    let written = (2 * VR - 2) * W;
    for (w, v) in want_root[written..written + 2 * W].iter_mut().zip(&xv) {
        *w = v.max(0.0);
    }
    assert_eq!(
        root_out, want_root,
        "the row lands in the view's slice of the root, and the root's other rows keep theirs"
    );
}

/// The predictor's own cache shape, built without a model: one `Concat`-seeded
/// plane per block, then one `KvAppend` per level at that level's row. Level 0's row
/// is the seed's first row, so every row of the plane is written by a placement and
/// the frame has no copy step — the same arrangement the model-gated
/// `loop_shape` gate pins on the real frame, reproduced here so it can be checked
/// on a host without the model files.
#[test]
fn a_concat_seeded_plane_chain_takes_every_row_in_place() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let rows = 4;
    let mut g = Graph::new("kv_append_plane_chain");
    let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
    let row_of = || Shape::new(&[1, W], DType::F32);
    let relu =
        |g: &mut Graph, src| g.add_node(Op::Activation(Activation::Relu), vec![src], row_of());
    let row = |g: &mut Graph, at: usize, base: f32| -> (rlx_ir::NodeId, Vec<f32>) {
        let name = format!("x{at}");
        let values: Vec<f32> = (0..W).map(|i| base + i as f32 - 3.0).collect();
        let id = g.input(&name, row_of());
        (relu(g, id), values)
    };
    let (first, first_values) = row(&mut g, 0, 0.0);
    let zeros = g.add_node(
        Op::Constant {
            data: vec![0u8; (rows - 1) * W * 4],
        },
        vec![],
        Shape::new(&[rows - 1, W], DType::F32),
    );
    let plane = g.add_node(
        Op::Concat { axis: 0 },
        vec![first, zeros],
        Shape::new(&[rows, W], DType::F32),
    );
    let mut want = vec![0.0; rows * W];
    want[..W].copy_from_slice(&first_values.iter().map(|v| v.max(0.0)).collect::<Vec<_>>());
    inputs.push(("x0".into(), first_values));
    // Every level appends to the **seed**, not to the level before it: the plane
    // is sized to the capacity it grows to, and each append publishes the prefix
    // `[..pos + 1]` the verifier infers. The rows still land in one storage — the
    // placement moves each producer's write onto the cache's own row — so the last
    // level's output carries every row.
    let mut last = plane;
    for pos in 1..rows {
        let (produced, values) = row(&mut g, pos, pos as f32 * 10.0);
        last = g.add_node(
            Op::KvAppend { axis: 0, pos },
            vec![plane, produced],
            Shape::new(&[pos + 1, W], DType::F32),
        );
        want[pos * W..(pos + 1) * W]
            .copy_from_slice(&values.iter().map(|v| v.max(0.0)).collect::<Vec<_>>());
        inputs.push((format!("x{pos}"), values));
    }
    g.set_outputs(vec![last]);
    assert!(
        rlx_ir::verify::verify(&g).is_empty(),
        "a witness must be a graph the verifier accepts: {:?}",
        rlx_ir::verify::verify(&g)
    );
    let mut exe = WgpuExecutable::compile(g);
    let steps = exe.test_step_names();
    let copies = steps.iter().filter(|s| **s == "buffer_copy").count();
    assert_eq!(
        copies, 0,
        "every row of a concat-seeded plane is written by its producer: {steps:?}"
    );
    let borrowed: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    assert_eq!(
        exe.run(&borrowed).remove(0),
        want,
        "each level's row lands at its own position, the seed's row at position 0"
    );
}

/// Two appends on one cache root that disagree on `axis` count their positions in
/// different row lengths — `base + pos · kv_row_bytes(cache, axis)` — so one can
/// land inside the other's row. Here the axis-1 append's 4-byte row sits at
/// `base + 4`, inside the axis-0 append's whole-cache row, and its producer's step
/// runs *first*: placed, it would be overwritten by the whole-row write. The root
/// is refused whole and both copies stay, which is the graph's own order.
#[test]
fn a_cache_root_whose_appends_disagree_on_axis_keeps_the_copy() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let x: Vec<f32> = (0..W).map(|i| (i as f32) - 3.0).collect();
    let y = vec![7.5f32];
    let mut g = Graph::new("kv_append_mixed_axis");
    let cache = g.input("cache", Shape::new(&[1, W], DType::F32));
    let xi = g.input("x", Shape::new(&[1, W], DType::F32));
    let yi = g.input("y", Shape::new(&[1, 1], DType::F32));
    // The axis-1 append's producer is built first, so a placement would let the
    // later whole-row write cover it.
    let narrow_row = g.add_node(
        Op::Activation(Activation::Relu),
        vec![yi],
        Shape::new(&[1, 1], DType::F32),
    );
    let whole_row = g.add_node(
        Op::Activation(Activation::Relu),
        vec![xi],
        Shape::new(&[1, W], DType::F32),
    );
    let one = g.add_node(
        Op::KvAppend { axis: 0, pos: 0 },
        vec![cache, whole_row],
        Shape::new(&[1, W], DType::F32),
    );
    let two = g.add_node(
        Op::KvAppend { axis: 1, pos: 1 },
        vec![one, narrow_row],
        Shape::new(&[1, 2], DType::F32),
    );
    g.set_outputs(vec![two]);
    let out = kept(
        g,
        &[("cache", &ramp(W, 0.0)), ("x", &x), ("y", &y)],
        2,
        "a root whose appends disagree on the axis keeps both copies",
    );
    // The append's output is its `[..pos + 1]` prefix, so the readback is the
    // cache's first two columns: the whole-row write, then column 1 overwritten.
    let want = vec![x[0].max(0.0), y[0]];
    assert_eq!(
        out[0], want,
        "the whole-row write first, then the axis-1 row at column 1"
    );
}

/// A striped arena cannot be bound whole at base zero, so the placement declines —
/// but on that geometry the append's operands straddle shards and the append is
/// hosted either way, so what this case pins is the outcome (no copy step, the
/// row's values, the host route that says the straddle), not the refusal itself:
/// that is `Arena::binds_whole_at_base_zero`'s, and the placement asks it rather
/// than a second spelling. A lowered shard cap reaches the geometry without a
/// multi-gigabyte plan; the arena's own size is what exceeds it.
///
/// Its own process, because the cap is a process-global switch and a sibling test
/// must not see it.
#[test]
fn a_striped_arena_still_computes_its_row() {
    const MARKER: &str = "KV_IN_PLACE_CHILD";
    const FIXTURE: &str = "striped";
    const FILTER: &str = "a_striped_arena_still_computes_its_row";

    if let Ok(fixture) = std::env::var(MARKER) {
        eprintln!("{MARKER} {fixture}");
        assert_eq!(fixture, FIXTURE);
        let mut g = Graph::new("kv_append_striped");
        let cache = g.input("cache", Shape::new(&[512, 512], DType::F32));
        let x = g.input("x", Shape::new(&[1, 512], DType::F32));
        let row = g.add_node(
            Op::Activation(Activation::Relu),
            vec![x],
            Shape::new(&[1, 512], DType::F32),
        );
        let out = g.add_node(
            Op::KvAppend { axis: 0, pos: 511 },
            vec![cache, row],
            Shape::new(&[512, 512], DType::F32),
        );
        g.set_outputs(vec![out]);
        let mut exe = WgpuExecutable::compile(g);
        let steps = exe.test_step_names();
        assert!(
            !steps.contains(&"buffer_copy") && steps.contains(&"host_op"),
            "the append's operands straddle stripes, so no device copy step is \
             scheduled and the append runs on the host: {steps:?}"
        );
        let xv: Vec<f32> = (0..512).map(|i| (i as f32) * 0.5 - 100.0).collect();
        let got = exe
            .run(&[("cache", &ramp(512 * 512, 0.0)), ("x", &xv)])
            .remove(0);
        let mut want = ramp(512 * 512, 0.0);
        for (w, v) in want[511 * 512..512 * 512].iter_mut().zip(&xv) {
            *w = v.max(0.0);
        }
        assert_eq!(got, want, "the striped geometry still writes the spare row");
        return;
    }
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (ok, stderr) = common::rerun_self(
        FILTER,
        MARKER,
        FIXTURE,
        &common::CLEARED_SWITCHES,
        &[
            ("RLX_WGPU_SHARD_CAP_MIB", "2"),
            ("RLX_WGPU_SHARD_STAGE_MIB", "1"),
            ("RLX_WGPU_ALLOW_SHARD", "1"),
        ],
    );
    assert!(ok, "the striped-arena child failed:\n{stderr}");
    assert!(
        stderr.contains("hosting KvAppend"),
        "the striped geometry's append runs on the host:\n{stderr}"
    );
}
