// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Attention fixtures for the head-major route tests: the product's shapes, the two
//! operand layouts, and a compiled-graph runner.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{Thunk, ThunkSchedule, compile_thunks, execute_thunks};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};

/// One attention operand set: `batch` windows of `seq` queries over `kv_seq` keys.
#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub batch: usize,
    pub seq: usize,
    pub kv_seq: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub mask: MaskKind,
    pub softcap: Option<f32>,
}

/// The product's head-major attention at the shapes it runs: the tagger's ViT
/// blocks, the ASR encoder tail, the decoder's prefill ladder and its decode steps,
/// the wake window. Laid out by hand (`#[rustfmt::skip]`): rustfmt breaks these rows
/// one field a line, and the table is only readable as one line a shape.
#[rustfmt::skip]
pub const CASES: &[Case] = &[
    // The product's three head-major graphs, at the shapes they run.
    Case { name: "tagger 252",      batch: 1, seq: 252, kv_seq: 252, heads:  6, kv_heads:  6, head_dim:  64, mask: MaskKind::None,   softcap: None },
    Case { name: "encoder 64",      batch: 1, seq:  64, kv_seq:  64, heads: 16, kv_heads: 16, head_dim:  64, mask: MaskKind::None,   softcap: None },
    Case { name: "encoder 4x16",    batch: 4, seq:  16, kv_seq:  16, heads: 16, kv_heads: 16, head_dim:  64, mask: MaskKind::None,   softcap: None },
    Case { name: "encoder 16",      batch: 1, seq:  16, kv_seq:  16, heads: 16, kv_heads: 16, head_dim:  64, mask: MaskKind::None,   softcap: None },
    Case { name: "wake 2x10",       batch: 2, seq:  10, kv_seq:  10, heads: 16, kv_heads: 16, head_dim:  64, mask: MaskKind::None,   softcap: None },
    Case { name: "tail 16x8",       batch:16, seq:   8, kv_seq:   8, heads: 16, kv_heads: 16, head_dim:  64, mask: MaskKind::None,   softcap: None },
    // The decoder's prefill ladder, one row count per rung, causal, GQA 16/8.
    Case { name: "prefill 31",      batch: 1, seq:  31, kv_seq:  31, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 47",      batch: 1, seq:  47, kv_seq:  47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 63",      batch: 1, seq:  63, kv_seq:  63, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 79",      batch: 1, seq:  79, kv_seq:  79, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 95",      batch: 1, seq:  95, kv_seq:  95, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 111",     batch: 1, seq: 111, kv_seq: 111, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "prefill 96 keyed", batch: 1, seq: 96, kv_seq:  96, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Custom, softcap: None },
    // One decode step against a 512-row cache, and one against a short keyed one.
    Case { name: "decode 512",      batch: 1, seq:   1, kv_seq: 512, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    Case { name: "decode 32 keyed", batch: 1, seq:   1, kv_seq:  32, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Custom, softcap: None },
    // A capped head-major input: the cap is applied per score, before the softmax.
    Case { name: "capped 47",       batch: 1, seq:  47, kv_seq:  47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: Some(0.5) },
    // A batch of windows with GQA, so the K/V batch stride is `nkv` wide, not `nh`.
    Case { name: "prefill 47 b4 gqa", batch: 4, seq: 47, kv_seq: 47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Causal, softcap: None },
    // The masks the product does not emit, each applied per row; the bias is a
    // [b, h, s, ks] tensor, so its batch offset needs a batch to be visible.
    Case { name: "sliding 47",      batch: 1, seq:  47, kv_seq:  47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::SlidingWindow(8), softcap: None },
    Case { name: "window 47",       batch: 1, seq:  47, kv_seq:  47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Window(4), softcap: None },
    Case { name: "bias 2x47",      batch: 2, seq:  47, kv_seq:  47, heads: 16, kv_heads:  8, head_dim: 128, mask: MaskKind::Bias, softcap: None },
];

/// `[B, S, H·D]` rows-first, or `[B, H, S, D]` head-major.
pub fn graph(c: &Case, head_major: bool) -> Graph {
    let f = DType::F32;
    let (b, s, ks, h, kv, d) = (c.batch, c.seq, c.kv_seq, c.heads, c.kv_heads, c.head_dim);
    let mut g = Graph::new("attn");
    let (q_shape, k_shape, out_shape) = if head_major {
        (
            Shape::new(&[b, h, s, d], f),
            Shape::new(&[b, kv, ks, d], f),
            Shape::new(&[b, h, s, d], f),
        )
    } else {
        (
            Shape::new(&[b, s, h * d], f),
            Shape::new(&[b, ks, kv * d], f),
            Shape::new(&[b, s, h * d], f),
        )
    };
    let q = g.input("q", q_shape);
    let k = g.input("k", k_shape.clone());
    let v = g.input("v", k_shape);
    // Neither `attention` nor `attention_bias` takes a softcap, so a capped row
    // paired with either would measure nothing while looking covered.
    assert!(
        c.softcap.is_none() || !matches!(c.mask, MaskKind::Custom | MaskKind::Bias),
        "{}: this builder takes no softcap, so a capped row would measure nothing",
        c.name
    );
    let out = match c.mask {
        MaskKind::Custom => {
            let mask = g.input("mask", Shape::new(&[b, ks], f));
            g.attention(q, k, v, mask, h, d, out_shape)
        }
        MaskKind::Bias => {
            let bias = g.input("mask", Shape::new(&[b, h, s, ks], f));
            g.attention_bias(q, k, v, bias, h, d, out_shape)
        }
        kind => g.attention_kind_opts(q, k, v, h, d, kind, out_shape, None, c.softcap),
    };
    g.set_outputs(vec![out]);
    g
}

/// Deterministic pseudo-random activation in `[-0.5, 0.5)`.
pub fn val(seed: usize, i: usize) -> f32 {
    let z = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5
}

/// Operands for the correctness comparison: every fourth key is padding, so a route
/// that ignores the mask — or reads the wrong row — moves.
pub fn keyed(name: &str, n: usize) -> Vec<f32> {
    match name {
        "q" => (0..n).map(|i| val(1, i)).collect(),
        "k" => (0..n).map(|i| val(2, i)).collect(),
        "v" => (0..n).map(|i| val(3, i)).collect(),
        _ => (0..n).map(|i| if i % 4 == 3 { 0.0 } else { 1.0 }).collect(),
    }
}

/// Operands for timing: a dense `kv_seq × kv_seq` set, every key live.
pub fn dense(name: &str, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if name == "mask" { 1.0 } else { val(1, i) })
        .collect()
}

/// `RLX_CPU_ATTN_HEAD_MAJOR_SERIAL=1` keeps head-major on one thread. The kernel
/// reads it per call, so one process holds both arms.
///
/// `env::set` rather than `env::scoped`: `execute_thunks` hands the schedule to a
/// rayon worker, and the kernel reads the flag there, on a thread `scoped` would not
/// reach. The flag is therefore process-wide, which is why the tests in
/// `attention_head_major_route` take turns on a lock.
pub fn serial(on: bool) {
    rlx_ir::env::set("RLX_CPU_ATTN_HEAD_MAJOR_SERIAL", if on { "1" } else { "0" });
}

/// One compiled attention graph in its own arena, inputs filled once.
pub struct Run {
    arena: Arena,
    sched: ThunkSchedule,
}

impl Run {
    pub fn new(g: &Graph, fill: impl Fn(&str, usize) -> Vec<f32>) -> Self {
        let plan = rlx_opt::memory::plan_memory(g);
        let mut arena = Arena::from_plan(plan);
        let sched = compile_thunks(g, &arena);
        for node in g.nodes() {
            let Op::Input { name } = &node.op else {
                continue;
            };
            let n = node.shape.num_elements().expect("a static input shape");
            let data = fill(name, n);
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p, n);
            }
        }
        Self { arena, sched }
    }

    pub fn call(&mut self) {
        execute_thunks(&self.sched, self.arena.raw_buf_mut());
    }

    /// Point every attention thunk's output at the `usize::MAX` sentinel, which is
    /// what the arena accessor answers an empty slice for — how a node the planner
    /// left without a slot reaches the dispatcher.
    pub fn detach_attention_output(&mut self) {
        for t in &mut self.sched.thunks {
            if let Thunk::Attention { out, .. } = t {
                *out = usize::MAX;
            }
        }
    }

    /// Milliseconds of one call, over `reps` calls after a warm-up call.
    pub fn sample(&mut self, reps: usize) -> f64 {
        self.call();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            self.call();
        }
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    }

    /// Byte offset of a node's arena slot — `usize::MAX` where the planner left it
    /// none, which is how a DCE-orphaned node reads.
    pub fn offset(&self, id: rlx_ir::NodeId) -> usize {
        self.arena.byte_offset(id)
    }

    /// The graph's first output, read back out of the arena.
    pub fn output(&self, g: &Graph) -> Vec<f32> {
        let out = g.outputs[0];
        let n = g
            .node(out)
            .shape
            .num_elements()
            .expect("a static output shape");
        unsafe {
            let p = self
                .arena
                .raw_buf()
                .as_ptr()
                .add(self.arena.byte_offset(out)) as *const f32;
            std::slice::from_raw_parts(p, n).to_vec()
        }
    }

    /// Fill the graph's first output slot with `v`: a sentinel a call must not move.
    pub fn fill_output(&mut self, g: &Graph, v: f32) {
        let out = g.outputs[0];
        let n = g
            .node(out)
            .shape
            .num_elements()
            .expect("a static output shape");
        let off = self.arena.byte_offset(out);
        unsafe {
            let p = self.arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
            std::slice::from_raw_parts_mut(p, n).fill(v);
        }
    }
}
