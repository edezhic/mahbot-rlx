// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recognition decode **token's** own rate, at the shapes and quant schemes
//! the shipped Qwen3-ASR-1.7B body walks — model-free, `#[ignore]`d, and printing
//! rather than asserting every time.
//!
//! ```text
//! cargo test -p mahbot-inference --release --test decode_token_rate \
//!     -- --ignored --nocapture
//! ```
//!
//! **Measurement only.** Nothing here pins a millisecond or a byte rate; the
//! assertions are structural (the byte total of the plan, the schedule census and
//! the frame's step total, one submission and one wait for the frame call, an
//! output's finiteness).
//!
//! # Why the frame is not the stand
//!
//! The stand (the `mahbot-inference` binary) measures product calls off the model
//! files. This harness needs none of them: what it builds is the **shape** of one
//! decode step — the same 28 layers, the same 169 packed multiplies at the same
//! `(k, n, scheme)` triples, the same 28 attention calls over
//! `[1, 363, 8, 128]` key/value planes, and the same logits readback a product
//! decode step performs. The weights are a deterministic byte ramp
//! ([`support::packed_ramp`]) — nothing reads their values, so only their bytes
//! and their block layout are modelled.
//!
//! # The geometry
//!
//! `D = 2048`, `NH = 16`, `NKV = 8`, `HD = 128`, `FFN = 6144`, `LAYERS = 28`,
//! `VOCAB = 151936`; the cache capacity the shipped decode body allocates is
//! `KEYS = 363` rows (a 234-row prompt plus one 128-row prefill bucket, and the
//! step's own spare row). The walk enters the decode with `WRITTEN_FIRST = 234`
//! rows written, its first token reads 235 and its last reads
//! `WRITTEN_LAST = 299`; the step's own append lands in the spare row.
//!
//! Per layer, as `(k, n, scheme)` and the packed bytes those hold:
//!
//! | tensor | k | n | scheme | bytes |
//! | --- | --- | --- | --- | --- |
//! | `attn_q` | 2048 | 2048 | Q4_K | 2,359,296 |
//! | `attn_k` | 2048 | 1024 | Q4_K | 1,179,648 |
//! | `attn_v` | 2048 | 1024 | Q6_K even / Q4_K odd | 1,720,320 / 1,179,648 |
//! | `attn_output` | 2048 | 2048 | Q4_K | 2,359,296 |
//! | `gate+up` (fused) | 2048 | 12288 | Q4_K | 14,155,776 |
//! | `ffn_down` | 6144 | 2048 | Q6_K odd / Q4_K even | 10,321,920 / 7,077,888 |
//! | `output` head | 2048 | 151936 | Q6_K | 255,252,480 |
//!
//! The per-tensor scheme assignment is a **reconstruction**, not a read from a
//! file: it is the Q4_K_M mixture the loader reports for this pair, and the
//! contract it must satisfy is arithmetic — the sum over all 28 layers plus the
//! head is asserted against the recorded `1,100,961,792` bytes per token, so any
//! other assignment that lands on that number is byte-indistinguishable to this
//! harness. Which layer a tensor sits on cannot move a time at these sizes: every
//! tensor is its own parameter region, read once by its own multiply, so the 28
//! layers differ from one another only in whether `v` and `down` carry the 6-bit
//! scale bytes.
//!
//! The key/value planes are one f32 parameter per layer per plane,
//! `[1, 363, 8, 128]` = 1,486,848 B each, 83,263,488 B for all 28 layers' K+V —
//! allocated whole, as the plane cache allocates them, and read at whatever the
//! mask's fill says is written.
//!
//! Every arm runs under [`switches`]: the ten stand-down switches the frame's
//! route reads, at their shipped values, plus the shipped encode chunk —
//! installed as code overrides, so an ambient value of *those* names cannot
//! relabel an arm. The set is printed once per run. A knob outside it (the arena,
//! shard and reporting switches) is not pinned: one set there changes the machine
//! under every arm alike.
//!
//! # The arms
//!
//! **A — the frame.** One graph of the whole step, compiled once, run at four
//! mask fills: `written` = 2, 234, 299 and 363. The first is the read-free
//! control — two visible rows leave the decode attention's dispatch, softmax and
//! merge in place with essentially no key/value traffic — so `A(363) − A(2)`
//! prices the reads rather than the arrangement, which nothing else in the tree
//! separates. The rate is printed over three numerators, because which one is
//! "the" rate is the question: the packed weights alone, the weights plus the
//! key/value bytes fetched once, and the weights plus those bytes counted twice
//! for the 16-query-heads-on-8-kv-heads walk. The read itself is priced as an
//! **implied** rate — the fill's bytes over the paired difference — so a figure
//! above the machine's own streaming read ceiling says the fill is served below
//! DRAM rather than that the weights are slow.
//!
//! **A2 — the submission shape.** The same compiled frame, the same bytes and
//! the same dispatch count, sweeping `RLX_WGPU_ENCODE_CHUNK` over [`CHUNKS`] —
//! the shipped 64 encoded steps per queue submission, then 32, 16 and the whole
//! schedule — so the only thing that moves is how many submissions one call
//! becomes. Every value is paired per round against the shipped one, so the pair
//! prices the per-submission term directly.
//!
//! **B — the per-shape arms.** Seven graphs, one per distinct `(k, n, scheme)`
//! triple the layer walks, each `N` independent copies of one multiply over one
//! shared input row, `N` set so the call streams at least
//! [`support::TARGET_BYTES`] — the tree's own convention, so every arm sees one
//! load. Each arm also carries how many times a token walks that shape: 56 for
//! `q_or_o` (the query and the output projection are the same triple), 42 for
//! `k_or_v_q4k` (28 keys plus the 14 layers whose `v` is 4-bit), 28 for
//! `gate_up`, 14 for each of `v_q6k`, `down_q4k` and `down_q6k`, and 1 for the
//! head. A compile-time pin holds those copies to the token's 169 packed
//! multiplies, and the additive
//! floor is the copies times the arm's own per-step time. That floor is
//! **everything back to back with no dependence and no per-dispatch sharing** —
//! it is a lower bound on a fantasy machine that has this backend's per-step cost
//! and none of the frame's structure, not a target.
//!
//! Three further calls probe what fusing query and key into one multiply of
//! their summed width would buy, each streaming the same bytes per copy: the two
//! separate projections as shipped, the fused multiply alone, and the fused
//! multiply with the narrow split its emit leaves behind (both pieces are graph
//! outputs there, so the split's own dispatch is the only thing the last form
//! adds). Each prints its schedule census, so the dispatch count it costs is
//! printed beside the time it takes.
//!
//! **C — the dependency arm.** The same 28 layers and the same 169 packed
//! multiplies with the multiply-to-multiply edges cut: `o`, `gate+up` and
//! `q/k/v` all read the layer's own normed row, so only
//! `gate+up → swiglu → down` keeps an edge (it cannot go without dead-coding
//! the swiglu). Its attention output is added into the running residual so the
//! 28 attention calls stay in the graph, which is the one node this arm has and
//! the frame does not. `frame − dependency` is what the dependent chain costs.
//!
//! **D — the attention cells.** 28 chained attention calls (query of one feeds
//! the next) at four `(num_heads, nkv)` pairs over rank-4 `[1, 363, nkv, 128]`
//! planes: `(16, 8)` as shipped, then `(8, 8)`, `(16, 16)` and `(16, 4)`. The
//! four readings separate two hypotheses: that the time follows the key/value
//! bytes fetched once (which scale with `nkv`), and that it follows the
//! per-query-head walk (which scales with `nh`). At fixed `nkv`, halving `nh`
//! halves the query-head walk and leaves the fetched bytes alone, while doubling
//! `nkv` at fixed `nh` does the opposite. Read the four as a **bound, not a
//! decomposition** — compute
//! also scales with `nh`, though at 363 keys its share is tiny (2 × 363 × 128
//! FMAs per head per layer, against a plane fetch of 363 × 128 floats) — and
//! against the **grid**: the decode entry's grid is `heads × key_splits`
//! workgroups, so halving the query heads halves the workgroups with the walk.
//! Each cell prints its grid, its bytes and its time; the ratios between them
//! are the reading, and they are printed rather than scored.
//!
//! # What each arm cannot show
//!
//! Every arm is a **proxy**: wall time around one call, with the call's one
//! submission, one blocking wait and one output readback inside the
//! denominator, and no per-dispatch device clock to separate them (this backend
//! exposes none). A is the shape of the shipped step and nothing about its
//! numerics — the weights are a ramp, and no output value is compared against
//! anything. B prices one shape in isolation, so its rate is the ceiling a
//! shape's own bytes could reach if the frame gave it the machine alone; the
//! difference `A − B` is not a per-dispatch cost, because a frame's steps share
//! passes the shape arm's copies do not. The additive floor sums **seven**
//! medians, each weighted by the copies a token walks, and inherits their
//! spreads, so it prints the envelope of those spreads beside its point value: a
//! window whose arms were quiet gives a tighter floor than one whose arms were
//! not, and `A − floor` moves with it. C keeps the frame's bytes and its 169
//! multiplies and moves only the edges; its unfused norms cost the extra
//! elementwise steps its own census prints, so read the pair against those. D is
//! attention in isolation: its planes are never appended to, and the row the
//! real step writes into each before reading it is the *fold* a step pays (a
//! device-side copy of the rows the step appends) — the frame here writes each
//! row in-graph (`KvAppend`, one row, aliasing the cache) but pays no fold, so it
//! prices the chunked submissions and not the fold.
//!
//! The submission bracket is the one place the backend's own chunking shows: it
//! hands one command buffer per 64 encoded steps, so a schedule this long is
//! several *queue* submissions inside one call — five at the shipped chunk, one
//! with it pinned to the whole schedule. Both are printed, and A2 prices them.
//!
//! # The round protocol
//!
//! Three warm-up runs, then [`ROUNDS`] timed rounds. Within a round the four
//! mask fills, the submission shapes, the three q/k forms, and the frame and the
//! dependency arm run in an order that rotates with the round number, so no arm
//! is always first and each of their differences is the **median of the
//! per-round paired differences**. The four attention cells are the exception:
//! they are read as **ratios of their medians** against the shipped `(16, 8)`
//! cell, not as paired differences. The per-shape arms are the other exception:
//! one call of one already streams
//! [`support::TARGET_BYTES`], so each is measured in its own block (three
//! warm-ups, then [`ROUNDS`] rounds) and the floor they sum is paired with the
//! frame by position in the window rather than round by round. The reported
//! level is the median of the arm's own samples, with the least and the greatest
//! beside it because the spread is information about the shape.

mod support;

use mahbot_inference::report;
use rlx_ir::op::RopeStyle;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape, Tick};
use rlx_runtime::{Device, Session};

use support::{
    assert_run_report_off, call_of, greatest, least, median, packed_ramp, ramped_row, step_census,
    steps_for,
};

// ── The model geometry ──

const D: usize = 2048;
const NH: usize = 16;
const NKV: usize = 8;
const HD: usize = 128;
const FFN: usize = 6144;
const LAYERS: usize = 28;
const VOCAB: usize = 151936;
const KEYS: usize = 363;
/// The prompt rows the recognition walk starts its decode from.
const WRITTEN_FIRST: usize = 234;
/// The rows written by that walk's last generated token.
const WRITTEN_LAST: usize = 299;
const EPS: f32 = 1e-6;
/// The recorded packed bytes one token reads: 28 layers plus the head.
const TOTAL_PACKED_BYTES: u64 = 1_100_961_792;
/// The steps the shipped body's decode step compiles to on this backend. Its
/// equality to the product's step count cannot be re-derived model-free: the
/// assertion catches a schedule drift inside the frame this file builds, not a
/// drift between that frame and the shipped body.
const FRAME_STEPS: usize = 310;

/// Timed rounds; odd, so [`median`] is the middle sample.
const ROUNDS: usize = 7;
/// Warm-ups before the first timed round, so the first round is not the first
/// run.
const WARMUPS: usize = 3;
/// The mask fills the frame is timed at, in the order A reports them. The first
/// is the read-free control: two visible rows is the decode attention's own
/// dispatch and softmax with essentially no key/value traffic, so the difference
/// against the others prices the reads rather than the arrangement.
const WRITTENS: [usize; 4] = [2, WRITTEN_FIRST, WRITTEN_LAST, KEYS];
/// The index of the fill the token is read at: the walk's own last generated row.
const WRITTEN_CELL: usize = 2;
const _: () = assert!(WRITTENS[WRITTEN_CELL] == WRITTEN_LAST);
/// The encode-chunk values A2 sweeps, the shipped one first: one queue
/// submission per this many encoded steps (`run.rs`'s `RLX_WGPU_ENCODE_CHUNK`
/// default is 64; `0` means the whole schedule in one).
const CHUNKS: [usize; 4] = [64, 32, 16, 0];

/// The switches every arm runs under: [`support::FRAME_SWITCHES`] at its shipped
/// values plus the shipped encode chunk, installed as code overrides, so neither
/// an ambient value of those names nor a run-time switch can make an arm silently
/// differ from the one it is labelled as. Anything outside the set is the
/// machine's, not an arm's.
fn switches() -> Vec<(&'static str, &'static str)> {
    let mut values = support::FRAME_SWITCHES.to_vec();
    values.push(("RLX_WGPU_ENCODE_CHUNK", "64"));
    values
}

const Q4: QuantScheme = QuantScheme::GgufQ4K;
const Q6: QuantScheme = QuantScheme::GgufQ6K;

/// `(k, n, scheme)` of one packed weight, and the bytes it holds.
#[derive(Clone, Copy, PartialEq)]
struct Geo {
    k: usize,
    n: usize,
    scheme: QuantScheme,
}

impl Geo {
    /// The packed bytes a file holds for this tensor: a whole number of the
    /// scheme's blocks per row, `n` rows.
    fn bytes(self) -> u64 {
        (self.k as u64 / self.scheme.gguf_block_size() as u64)
            * self.scheme.gguf_block_bytes() as u64
            * self.n as u64
    }
}

/// One named packed weight: the parameter it is declared under and its geometry.
struct Packed {
    name: String,
    geo: Geo,
}

/// The six packed weights of one layer, in graph order: q, k, v, o, the fused
/// gate+up, down.
///
/// `v` is Q6_K on even layers and `down` is Q6_K on odd ones — the Q4_K_M
/// mixture, which is why q/k/v cannot be one fused multiply: they do not share
/// a scheme, and the load-time fusion is refused graph-wide rather than per
/// layer (`emit::fuse_packed`). gate+up does share one, and is emitted as one
/// multiply narrowed back into two halves.
fn layer_weights(layer: usize) -> [(&'static str, Geo); 6] {
    let q4 = |k: usize, n: usize| Geo { k, n, scheme: Q4 };
    let v = if layer.is_multiple_of(2) { Q6 } else { Q4 };
    let down = if layer.is_multiple_of(2) { Q4 } else { Q6 };
    [
        ("attn_q.weight", q4(D, D)),
        ("attn_k.weight", q4(D, NKV * HD)),
        (
            "attn_v.weight",
            Geo {
                k: D,
                n: NKV * HD,
                scheme: v,
            },
        ),
        ("attn_output.weight", q4(D, D)),
        ("gate_up.fused", q4(D, 2 * FFN)),
        (
            "ffn_down.weight",
            Geo {
                k: FFN,
                n: D,
                scheme: down,
            },
        ),
    ]
}

/// Every packed weight of the frame, in graph order — the one list the graph
/// builder and the upload both walk, so a name cannot drift from its bytes.
fn plan() -> Vec<Packed> {
    let mut out = Vec::with_capacity(6 * LAYERS + 1);
    for layer in 0..LAYERS {
        for (suffix, geo) in layer_weights(layer) {
            out.push(Packed {
                name: format!("blk.{layer}.{suffix}"),
                geo,
            });
        }
    }
    out.push(Packed {
        name: "output.weight".to_string(),
        geo: Geo {
            k: D,
            n: VOCAB,
            scheme: Q6,
        },
    });
    out
}

/// The key/value bytes one token reads at `written` keys: both planes of every
/// layer, one `nkv * hd` row each.
fn kv_bytes(written: usize, nkv: usize) -> u64 {
    (2 * LAYERS * written * nkv * HD * 4) as u64
}

/// The same bytes at the shipped body's 8 key/value heads.
fn kv_unique_bytes(written: usize) -> u64 {
    kv_bytes(written, NKV)
}

/// The same bytes counted twice: 16 query heads walking 8 key/value heads read
/// each row once per pair, and there are two pairs.
fn kv_twice_bytes(written: usize) -> u64 {
    2 * kv_bytes(written, NKV)
}

// ── The frame ──

/// Which of the two frame topologies a build produces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Topology {
    /// The frame: attention feeds `o`, and the residual feeds the FFN norm.
    Frame,
    /// The dependency arm: the same multiplies with the multiply-to-multiply
    /// edges cut, so each one reads the layer's own normed row.
    Dependency,
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Topology::Frame => "frame",
            Topology::Dependency => "dependency",
        }
    }
}

/// How an f32 parameter is filled. Only the planes' bytes are read as traffic;
/// the values exist so no denormal or NaN reaches the arithmetic.
#[derive(Clone, Copy)]
enum Fill {
    Ones,
    Zeros,
    Ramp,
}

/// One f32 parameter a graph declares, so the upload walks the same list the
/// graph was built from.
struct F32Param {
    name: String,
    elems: usize,
    fill: Fill,
}

fn f32_param(
    g: &mut Graph,
    list: &mut Vec<F32Param>,
    name: String,
    dims: &[usize],
    fill: Fill,
) -> NodeId {
    list.push(F32Param {
        name: name.clone(),
        elems: dims.iter().product(),
        fill,
    });
    g.param(name, Shape::new(dims, DType::F32))
}

/// One packed multiply over one row: the weight's bytes as a `u8` parameter,
/// read by the `DequantMatMul` that owns it.
fn dequant(g: &mut Graph, x: NodeId, w: &Packed) -> NodeId {
    let bytes = w.geo.bytes() as usize;
    let p = g.param(w.name.clone(), Shape::new(&[bytes], DType::U8));
    g.add_node(
        Op::DequantMatMul {
            scheme: w.geo.scheme,
        },
        vec![x, p],
        Shape::new(&[1, w.geo.n], DType::F32),
    )
}

/// Write one row into a plane's spare row and hand back the plane as the
/// attention reads it. `KvAppend` aliases the cache, so this is a one-row write,
/// not a copy of the cache; the operand shapes are the ones
/// `kv_append_shape` holds (`row[axis] == 1`, every other axis equal).
fn kv_append(g: &mut Graph, plane: NodeId, row: NodeId, pos: usize) -> NodeId {
    let shape = g.node(plane).shape.clone();
    g.add_node(Op::KvAppend { axis: 1, pos }, vec![plane, row], shape)
}

/// The frame graph: 28 layers of the decode token's topology, then the output
/// norm and the head.
///
/// The per-head norms and the rotations are laid out the way
/// `graphs::BodyStateStage::emit_layer` lays them out: the projection is
/// reshaped to `[heads, hd]` for its norm, flattened back to one `[1, 1, nh·hd]`
/// row for the rotation (one cos row per position, the head walk inside the
/// row), and only then split into the attention's `[1, 1, heads, hd]` plane
/// layout.
fn frame_graph(topo: Topology, plan: &[Packed]) -> (Graph, Vec<F32Param>) {
    let mut g = Graph::new(match topo {
        Topology::Frame => "decode_token_frame",
        Topology::Dependency => "decode_token_dependency",
    });
    let mut f32s = Vec::new();
    let x = g.input("x", Shape::new(&[1, D], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, KEYS], DType::F32));
    let cos = g.input("cos", Shape::new(&[1, HD / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[1, HD / 2], DType::F32));
    // The two beta widths the layers' norms need, declared once and reused.
    let zeros_d = f32_param(&mut g, &mut f32s, "zeros2048".into(), &[D], Fill::Zeros);
    let zeros_h = f32_param(&mut g, &mut f32s, "zeros128".into(), &[HD], Fill::Zeros);
    // The attention's output shape is its query's, one row per head.
    let query_shape = Shape::new(&[1, 1, NH, HD], DType::F32);

    // The frame carries its residual from layer to layer; the dependency arm
    // never reads it back into a multiply, so every layer's norm reads the
    // graph's own input row.
    let mut state = x;
    for layer in 0..LAYERS {
        let p = format!("blk.{layer}");
        let w = &plan[6 * layer..6 * layer + 6];
        let layer_input = match topo {
            Topology::Frame => state,
            Topology::Dependency => x,
        };

        let gamma = f32_param(
            &mut g,
            &mut f32s,
            format!("{p}.attn_norm.weight"),
            &[D],
            Fill::Ones,
        );
        let h = g.rms_norm(layer_input, gamma, zeros_d, EPS);
        let q = dequant(&mut g, h, &w[0]);
        let k = dequant(&mut g, h, &w[1]);
        let v = dequant(&mut g, h, &w[2]);

        let gq = f32_param(
            &mut g,
            &mut f32s,
            format!("{p}.attn_q_norm.weight"),
            &[HD],
            Fill::Ones,
        );
        let qr = g.reshape_(q, vec![NH as i64, HD as i64]);
        let qr = g.rms_norm(qr, gq, zeros_h, EPS);
        let q = g.reshape_(qr, vec![1, 1, (NH * HD) as i64]);
        let q = g.rope_styled(q, cos, sin, HD, RopeStyle::NeoX);
        let q4 = g.reshape_(q, vec![1, 1, NH as i64, HD as i64]);

        let gk = f32_param(
            &mut g,
            &mut f32s,
            format!("{p}.attn_k_norm.weight"),
            &[HD],
            Fill::Ones,
        );
        let kr = g.reshape_(k, vec![NKV as i64, HD as i64]);
        let kr = g.rms_norm(kr, gk, zeros_h, EPS);
        let k = g.reshape_(kr, vec![1, 1, (NKV * HD) as i64]);
        let k = g.rope_styled(k, cos, sin, HD, RopeStyle::NeoX);
        let k4 = g.reshape_(k, vec![1, 1, NKV as i64, HD as i64]);
        let v4 = g.reshape_(v, vec![1, 1, NKV as i64, HD as i64]);

        let plane = [1, KEYS, NKV, HD];
        let kp = f32_param(&mut g, &mut f32s, format!("k{layer}"), &plane, Fill::Ramp);
        let vp = f32_param(&mut g, &mut f32s, format!("v{layer}"), &plane, Fill::Ramp);
        // The step's own rows land in each plane's spare row, in-graph, exactly
        // as `BodyStateStage::emit_layer`'s plane cache writes them: the append
        // aliases the plane, so it is one row write inside the batch rather than
        // a pass over the cache — and it is what makes the two projections the
        // attention reads live, rather than a node the planner dead-codes.
        let k_all = kv_append(&mut g, kp, k4, KEYS - 1);
        let v_all = kv_append(&mut g, vp, v4, KEYS - 1);
        let a = g.attention(q4, k_all, v_all, mask, NH, HD, query_shape.clone());
        let a = g.reshape_(a, vec![1, (NH * HD) as i64]);

        let o = match topo {
            Topology::Frame => dequant(&mut g, a, &w[3]),
            Topology::Dependency => dequant(&mut g, h, &w[3]),
        };
        state = g.add(state, o);
        if topo == Topology::Dependency {
            // The one node the dependency arm adds: the attention result has to
            // reach an output, or the call is dead and drops out of the graph.
            state = g.add(state, a);
        }

        let gamma = f32_param(
            &mut g,
            &mut f32s,
            format!("{p}.ffn_norm.weight"),
            &[D],
            Fill::Ones,
        );
        let h2 = g.rms_norm(
            match topo {
                // The shipped frame's ffn block reads the **post-attention**
                // residual — that edge is what makes the layer one chain.
                Topology::Frame => state,
                Topology::Dependency => x,
            },
            gamma,
            zeros_d,
            EPS,
        );
        let gu = dequant(&mut g, h2, &w[4]);
        let gate = g.narrow_(gu, 1, 0, FFN);
        let up = g.narrow_(gu, 1, FFN, FFN);
        let act = g.silu(gate);
        let act = g.mul(act, up);
        let down = dequant(&mut g, act, &w[5]);
        state = g.add(state, down);
    }

    let gamma = f32_param(
        &mut g,
        &mut f32s,
        "output_norm.weight".into(),
        &[D],
        Fill::Ones,
    );
    let normed = g.rms_norm(state, gamma, zeros_d, EPS);
    let logits = dequant(&mut g, normed, &plan[6 * LAYERS]);
    g.set_outputs(vec![logits]);
    (g, f32s)
}

fn fill(p: &F32Param) -> Vec<f32> {
    match p.fill {
        Fill::Ones => vec![1.0; p.elems],
        Fill::Zeros => vec![0.0; p.elems],
        Fill::Ramp => ramped_row(p.elems),
    }
}

/// One compiled frame and the vectors a round feeds it.
struct Frame {
    topo: Topology,
    compiled: rlx_runtime::CompiledGraph,
    x: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    masks: Vec<Vec<f32>>,
}

/// One build of a frame, in milliseconds: graph construction, compile, upload,
/// and the bytes the upload wrote. This is the harness's own frame, not the
/// product's (whose `build_ms` is the same work over the model files).
struct BuildMs {
    build: f64,
    compile: f64,
    upload: f64,
    bytes: u64,
}

fn build_frame(topo: Topology, plan: &[Packed]) -> (Frame, BuildMs) {
    let start = Tick::now();
    let (graph, f32s) = frame_graph(topo, plan);
    let build = Tick::now().elapsed_ns(start) as f64 / 1e6;

    let start = Tick::now();
    let mut compiled = Session::new(Device::Gpu).compile(graph);
    let compile = Tick::now().elapsed_ns(start) as f64 / 1e6;

    let start = Tick::now();
    let mut bytes = 0u64;
    for w in plan {
        let packed = packed_ramp(w.geo.scheme, w.geo.bytes() as usize);
        compiled.set_param_typed(&w.name, &packed, DType::U8);
        bytes += packed.len() as u64;
    }
    for p in &f32s {
        let values = fill(p);
        compiled.set_param(&p.name, &values);
        bytes += (values.len() * 4) as u64;
    }
    let upload = Tick::now().elapsed_ns(start) as f64 / 1e6;

    let masks = WRITTENS
        .iter()
        .map(|&written| {
            (0..KEYS)
                .map(|i| if i < written { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    (
        Frame {
            topo,
            compiled,
            x: ramped_row(D),
            // A rotation by zero for the one position the table has: the table's
            // values are not what this harness times.
            cos: vec![1.0; HD / 2],
            sin: vec![0.0; HD / 2],
            masks,
        },
        BuildMs {
            build,
            compile,
            upload,
            bytes,
        },
    )
}

impl Frame {
    /// One run, in milliseconds, at one of [`WRITTENS`]' fills.
    fn run(&mut self, variant: usize) -> f64 {
        let start = Tick::now();
        self.compiled.run(&[
            ("x", &self.x),
            ("mask", &self.masks[variant]),
            ("cos", &self.cos),
            ("sin", &self.sin),
        ]);
        Tick::now().elapsed_ns(start) as f64 / 1e6
    }

    /// [`Frame::run`] with the encode chunk pinned. It is read at the dispatch,
    /// so both arms are the same compiled graph over the same bytes — the only
    /// thing that moves is how many queue submissions the call becomes.
    fn run_at_chunk(&mut self, variant: usize, chunk: usize) -> f64 {
        let _switch =
            rlx_ir::env::RuntimeOverrides::install([("RLX_WGPU_ENCODE_CHUNK", chunk.to_string())]);
        self.run(variant)
    }
}

/// A frame's schedule census and step total. The 169 packed multiplies and the
/// 28 attention steps are asserted for both topologies, and the frame's 310-step
/// total against the shipped body's own — so a schedule that drifted prints a
/// failure rather than a plausible new number. A backend that exposes no
/// schedule prints that and reports zero.
fn census(frame: &Frame) -> usize {
    let Some(names) = frame.compiled.schedule_step_names() else {
        eprintln!(
            "[decode token rate] arm={} census=none (this backend exposes no schedule)",
            frame.topo.label()
        );
        return 0;
    };
    let count = |kind: &str| names.iter().filter(|n| **n == kind).count();
    eprintln!(
        "[decode token rate] arm={} steps={} census={}",
        frame.topo.label(),
        names.len(),
        step_census(&names)
    );
    assert_eq!(
        count("dequant_matmul_gguf"),
        6 * LAYERS + 1,
        "the frame's 169 packed multiplies must each be one gguf step — a step \
         that left the route would change every byte figure this harness prints"
    );
    assert_eq!(
        count("attention"),
        LAYERS,
        "one attention step per layer, all of them on the decode route"
    );
    if frame.topo == Topology::Frame {
        assert_eq!(
            names.len(),
            FRAME_STEPS,
            "the frame must compile to the shipped body's {FRAME_STEPS}-step \
             schedule; a moved total means a fusion pattern moved under this \
             harness's arms"
        );
    }
    names.len()
}

/// The output the step reads back: one `[1, VOCAB]` logits row, finite.
fn check_logits(frame: &mut Frame) {
    let out = frame.compiled.run(&[
        ("x", &frame.x),
        ("mask", &frame.masks[WRITTEN_CELL]),
        ("cos", &frame.cos),
        ("sin", &frame.sin),
    ]);
    assert_eq!(out.len(), 1, "the graph's single output is the logits row");
    assert_eq!(
        out[0].len(),
        VOCAB,
        "the logits row is the decode step's own readback"
    );
    let non_finite = out[0].iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "the decode step's logits must be finite: a denormal or NaN here would \
         change what the arithmetic downstream costs"
    );
}

// ── B: the per-shape arms ──

/// One of the distinct shapes a layer walks, with the label it is reported under
/// and how many times one token walks it (`q` and `o`, and `k` and the Q4_K `v`,
/// are the same triple).
struct ShapeArm {
    label: &'static str,
    geo: Geo,
    copies: usize,
}

const SHAPE_ARMS: [ShapeArm; 7] = [
    ShapeArm {
        label: "q_or_o",
        geo: Geo {
            k: D,
            n: D,
            scheme: Q4,
        },
        copies: 2 * LAYERS,
    },
    ShapeArm {
        label: "k_or_v_q4k",
        geo: Geo {
            k: D,
            n: NKV * HD,
            scheme: Q4,
        },
        copies: LAYERS + LAYERS / 2,
    },
    ShapeArm {
        label: "v_q6k",
        geo: Geo {
            k: D,
            n: NKV * HD,
            scheme: Q6,
        },
        copies: LAYERS / 2,
    },
    ShapeArm {
        label: "gate_up",
        geo: Geo {
            k: D,
            n: 2 * FFN,
            scheme: Q4,
        },
        copies: LAYERS,
    },
    ShapeArm {
        label: "down_q4k",
        geo: Geo {
            k: FFN,
            n: D,
            scheme: Q4,
        },
        copies: LAYERS / 2,
    },
    ShapeArm {
        label: "down_q6k",
        geo: Geo {
            k: FFN,
            n: D,
            scheme: Q6,
        },
        copies: LAYERS / 2,
    },
    ShapeArm {
        label: "head_q6k",
        geo: Geo {
            k: D,
            n: VOCAB,
            scheme: Q6,
        },
        copies: 1,
    },
];

// The arms' weights must cover the token's packed multiplies (`6 × 28 + 1`), so a
// shape that gains or loses a walk fails here rather than in the floor's number.
const _: () = assert!({
    let mut copies = 0;
    let mut i = 0;
    while i < SHAPE_ARMS.len() {
        copies += SHAPE_ARMS[i].copies;
        i += 1;
    }
    copies == 6 * LAYERS + 1
});

/// Bytes one call streams over milliseconds, in GB/s.
fn gbps(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

/// One shape's own back-to-back per-step time: at its median round, and at its
/// quietest and loudest — the band the additive floor inherits from every arm it
/// sums.
#[derive(Clone, Copy)]
struct Steps {
    median: f64,
    least: f64,
    greatest: f64,
}

/// `steps` independent copies of one multiply over one input row, long enough
/// that the call leaves its own submission behind.
fn measure_shape(arm: &ShapeArm) -> Steps {
    let packed = packed_ramp(arm.geo.scheme, arm.geo.bytes() as usize);
    let steps = steps_for(arm.geo.bytes(), support::TARGET_BYTES);
    let mut call = call_of(arm.geo.scheme, arm.geo.k, arm.geo.n, 1, &packed, steps);
    for _ in 0..WARMUPS {
        call.run();
    }
    let samples: Vec<f64> = (0..ROUNDS).map(|_| call.ms()).collect();
    let rates: Vec<f64> = samples
        .iter()
        .map(|ms| gbps(arm.geo.bytes() * steps as u64, *ms))
        .collect();
    let per_step = |ms: f64| ms / steps as f64;
    eprintln!(
        "[decode token rate] shape={} scheme={:?} k={} n={} copies_per_token={} \
         steps={} packed_bytes_per_step={} ms_per_call=[{}] min_ms={:.3} \
         median_ms={:.3} max_ms={:.3} median_GBps={:.1} per_step_ms={:.4}",
        arm.label,
        arm.geo.scheme,
        arm.geo.k,
        arm.geo.n,
        arm.copies,
        steps,
        arm.geo.bytes(),
        samples
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        least(&samples),
        median(&samples),
        greatest(&samples),
        median(&rates),
        per_step(median(&samples)),
    );
    Steps {
        median: per_step(median(&samples)),
        least: per_step(least(&samples)),
        greatest: per_step(greatest(&samples)),
    }
}

// ── The query and key projections, separate and fused ──

/// The fused query+key multiply: one row of `D + NKV·HD` columns, which is the
/// two separate projections' bytes — so the three forms below stream one
/// geometry's bytes per copy.
const QK_FUSED: Geo = Geo {
    k: D,
    n: D + NKV * HD,
    scheme: Q4,
};

/// The three forms the two projections can be emitted in: the two separate
/// multiplies the shipped graph-wide fusion decision leaves, the one multiply of
/// their summed width a per-layer fusion would produce, and that multiply with
/// the narrow split it leaves behind.
#[derive(Clone, Copy)]
enum QkForm {
    Split,
    Fused,
    FusedSplit,
}

impl QkForm {
    fn label(self) -> &'static str {
        match self {
            QkForm::Split => "qk_split",
            QkForm::Fused => "qk_fused",
            QkForm::FusedSplit => "qk_fused_and_split",
        }
    }
}

/// One compiled probe call and the row it reads.
struct Probe {
    compiled: rlx_runtime::CompiledGraph,
    input: Vec<f32>,
}

impl Probe {
    /// `steps` copies of one [`QkForm`], all over one shared input row and all
    /// streaming the same bytes per copy, so the forms are compared under one
    /// load. The split form's pieces are graph outputs as they stand; the fused
    /// form's two halves are narrowed out of the one row, which is the whole of
    /// what the fusion adds.
    fn qk(form: QkForm, steps: usize) -> Self {
        let q = Geo {
            k: D,
            n: D,
            scheme: Q4,
        };
        let k = Geo {
            k: D,
            n: NKV * HD,
            scheme: Q4,
        };
        let mut g = Graph::new(format!("harness_{}", form.label()));
        let x = g.input("x", Shape::new(&[1, D], DType::F32));
        let named = |prefix: &str, step: usize, geo: Geo| Packed {
            name: format!("{prefix}{step}"),
            geo,
        };
        let mut outs = Vec::new();
        for step in 0..steps {
            match form {
                QkForm::Split => {
                    outs.push(dequant(&mut g, x, &named("q", step, q)));
                    outs.push(dequant(&mut g, x, &named("k", step, k)));
                }
                QkForm::Fused => outs.push(dequant(&mut g, x, &named("qk", step, QK_FUSED))),
                QkForm::FusedSplit => {
                    let row = dequant(&mut g, x, &named("qk", step, QK_FUSED));
                    outs.push(g.narrow_(row, 1, 0, D));
                    outs.push(g.narrow_(row, 1, D, NKV * HD));
                }
            }
        }
        g.set_outputs(outs);
        let mut compiled = Session::new(Device::Gpu).compile(g);
        // Every copy reads the same bytes, as `support::call_of`'s copies do.
        let (q_bytes, k_bytes, fused_bytes) = (
            packed_ramp(q.scheme, q.bytes() as usize),
            packed_ramp(k.scheme, k.bytes() as usize),
            packed_ramp(QK_FUSED.scheme, QK_FUSED.bytes() as usize),
        );
        for step in 0..steps {
            match form {
                QkForm::Split => {
                    compiled.set_param_typed(&format!("q{step}"), &q_bytes, DType::U8);
                    compiled.set_param_typed(&format!("k{step}"), &k_bytes, DType::U8);
                }
                _ => compiled.set_param_typed(&format!("qk{step}"), &fused_bytes, DType::U8),
            }
        }
        Probe {
            compiled,
            input: ramped_row(D),
        }
    }

    fn ms(&mut self) -> f64 {
        let start = Tick::now();
        self.compiled.run(&[("x", &self.input)]);
        Tick::now().elapsed_ns(start) as f64 / 1e6
    }

    /// The probe's schedule: its step total, its packed multiplies and the
    /// narrow split's own dispatches, printed so the dispatch count each form
    /// costs is read rather than assumed.
    fn census(&self, form: QkForm) {
        let Some(names) = self.compiled.schedule_step_names() else {
            eprintln!(
                "[decode token rate] qk_fusion form={} census=none \
                 (this backend exposes no schedule)",
                form.label()
            );
            return;
        };
        let count = |kind: &str| names.iter().filter(|n| **n == kind).count();
        eprintln!(
            "[decode token rate] qk_fusion form={} steps={} multiplies={} narrow_split={} \
             per_copy_bytes={}",
            form.label(),
            names.len(),
            count("dequant_matmul_gguf"),
            count("narrow_split"),
            QK_FUSED.bytes(),
        );
    }
}

// ── D: the attention cells ──

/// One attention-only cell: `nh` query heads over `nkv` key/value heads.
#[derive(Clone, Copy)]
struct Cell {
    nh: usize,
    nkv: usize,
}

const CELLS: [Cell; 4] = [
    // The shipped shape first: everything else is read against it.
    Cell { nh: 16, nkv: 8 },
    Cell { nh: 8, nkv: 8 },
    Cell { nh: 16, nkv: 16 },
    Cell { nh: 16, nkv: 4 },
];

impl Cell {
    fn label(self) -> String {
        format!("nh{}_nkv{}", self.nh, self.nkv)
    }

    /// The key-range depth the run takes for this cell, at the one query row and
    /// the cache's [`KEYS`] keys the decode step walks.
    fn key_splits(self) -> u32 {
        rlx_wgpu::backend::attention_decode_key_splits(
            1,
            KEYS as u32,
            self.nh as u32,
            HD as u32,
            HD as u32,
        )
    }

    /// The decode entry's grid: one workgroup per (batch, query head) per key
    /// range.
    fn grid(self) -> u64 {
        self.nh as u64 * self.key_splits() as u64
    }
}

/// 28 chained attention calls over `[1, KEYS, nkv, HD]` planes, the query of
/// one feeding the next — the shape `attention_decode_route.rs::chain` builds,
/// at the plane cache's own rank-4 layout.
fn cell_graph(cell: Cell) -> (Graph, Vec<F32Param>) {
    let mut g = Graph::new(format!("decode_attn_{}", cell.label()));
    let mut f32s = Vec::new();
    let mut q = g.input("q", Shape::new(&[1, 1, cell.nh, HD], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, KEYS], DType::F32));
    let plane = [1, KEYS, cell.nkv, HD];
    let query_shape = Shape::new(&[1, 1, cell.nh, HD], DType::F32);
    for layer in 0..LAYERS {
        let k = f32_param(&mut g, &mut f32s, format!("k{layer}"), &plane, Fill::Ramp);
        let v = f32_param(&mut g, &mut f32s, format!("v{layer}"), &plane, Fill::Ramp);
        q = g.attention(q, k, v, mask, cell.nh, HD, query_shape.clone());
    }
    g.set_outputs(vec![q]);
    (g, f32s)
}

/// One compiled cell and the vectors a round feeds it.
struct CellRun {
    cell: Cell,
    compiled: rlx_runtime::CompiledGraph,
    q: Vec<f32>,
    mask: Vec<f32>,
}

fn build_cell(cell: Cell) -> CellRun {
    let (graph, f32s) = cell_graph(cell);
    let mut compiled = Session::new(Device::Gpu).compile(graph);
    for p in &f32s {
        let values = fill(p);
        compiled.set_param(&p.name, &values);
    }
    CellRun {
        cell,
        compiled,
        q: ramped_row(cell.nh * HD),
        mask: (0..KEYS)
            .map(|i| if i < WRITTEN_LAST { 1.0 } else { 0.0 })
            .collect(),
    }
}

impl CellRun {
    fn run(&mut self) -> f64 {
        let start = Tick::now();
        self.compiled.run(&[("q", &self.q), ("mask", &self.mask)]);
        Tick::now().elapsed_ns(start) as f64 / 1e6
    }
}

// ── The harness ──

#[test]
#[ignore = "measurement, not a gate — run with `--ignored --nocapture`"]
fn token_rate() {
    // Under the run-report switch the submission/wait shape is the instrument's,
    // not the call's, and this harness pins `(1, 1)`.
    assert_run_report_off();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // Every arm's switches, installed for the whole run: both the compiles and
    // the dispatches read them.
    let switches = switches();
    let _switch = rlx_ir::env::RuntimeOverrides::install(switches.iter().copied());
    eprintln!(
        "[decode token rate] switch_provenance={}",
        switches
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let plan = plan();

    // ── The plan's own accounting: one line per distinct geometry and how many
    // tensors of it the token walks ────────────────────────────────────────
    let total: u64 = plan.iter().map(|w| w.geo.bytes()).sum();
    let mut geometries: Vec<(Geo, usize)> = Vec::new();
    for w in &plan {
        match geometries.iter_mut().find(|(geo, _)| *geo == w.geo) {
            Some((_, count)) => *count += 1,
            None => geometries.push((w.geo, 1)),
        }
    }
    for (geo, tensors) in &geometries {
        eprintln!(
            "[decode token rate] geometry k={} n={} scheme={:?} bytes={} tensors={tensors}",
            geo.k,
            geo.n,
            geo.scheme,
            geo.bytes()
        );
    }
    assert_eq!(
        total, TOTAL_PACKED_BYTES,
        "the reconstructed per-tensor scheme assignment must reproduce the \
         recorded {} bytes one token reads (it sums to {total})",
        TOTAL_PACKED_BYTES
    );
    eprintln!(
        "[decode token rate] weights_total_bytes={total} layers={LAYERS} \
         multiplies={} geometries={} kv_plane_bytes={} kv_all_planes_bytes={}",
        plan.len(),
        geometries.len(),
        kv_bytes(KEYS, NKV) / (2 * LAYERS) as u64,
        kv_bytes(KEYS, NKV)
    );
    for written in WRITTENS {
        eprintln!(
            "[decode token rate] kv_written={written} unique_bytes={} \
             twice_bytes={}",
            kv_unique_bytes(written),
            kv_twice_bytes(written)
        );
    }

    // The machine's own streaming read ceiling, so a fill's implied rate is read
    // against it instead of printed as a number no DRAM can reach. A probe that
    // produced no figure is named, not numbered.
    let ceiling = match rlx_wgpu::bandwidth::stream_ceiling() {
        Ok(measured) => {
            eprintln!(
                "[decode token rate] ceiling_read_gb_s={:.1}",
                measured.read_gb_s
            );
            Some(measured.read_gb_s)
        }
        Err(failure) => {
            eprintln!(
                "[decode token rate] ceiling_read_gb_s=unavailable kind={} reason={}",
                failure.kind(),
                failure.reason().unwrap_or("none")
            );
            None
        }
    };

    // ── A: the frame ──────────────────────────────────────────────────────
    let (mut frame, build) = build_frame(Topology::Frame, &plan);
    eprintln!(
        "[decode token rate] arm=frame graph_build_ms={:.1} compile_ms={:.1} \
         upload_ms={:.1} uploaded_bytes={}",
        build.build, build.compile, build.upload, build.bytes
    );
    let frame_steps = census(&frame);
    check_logits(&mut frame);

    // The bracket: one frame call, one submission and one blocking wait. The
    // backend hands one command buffer per `RLX_WGPU_ENCODE_CHUNK` encoded steps
    // (64 by default), so a schedule this long is several queue submissions at
    // the shipped value — the pinned `(1, 1)` is the shape of one *call*, one
    // submit and one wait as `graphs::device_counts` counts them, and it is read
    // with the chunk pinned to the whole schedule. Both figures are printed
    // because which one the caller wants depends on the question.
    rlx_wgpu::reset_device_counts();
    frame.run(WRITTEN_CELL);
    let (chunked_submits, chunked_waits) = (
        rlx_wgpu::device_submit_count(),
        rlx_wgpu::device_wait_count(),
    );
    rlx_wgpu::reset_device_counts();
    {
        let _one_pass = rlx_ir::env::RuntimeOverrides::install([("RLX_WGPU_ENCODE_CHUNK", "0")]);
        frame.run(WRITTEN_CELL);
    }
    let (submits, waits) = (
        rlx_wgpu::device_submit_count(),
        rlx_wgpu::device_wait_count(),
    );
    assert_eq!(
        (submits, waits),
        (1, 1),
        "one frame call must be one submission and one blocking wait (with the \
         encode chunk pinned it took {submits} and {waits}) — a step that left \
         the enclosing batch would substitute exactly the quantity this harness \
         prices"
    );
    assert_eq!(
        chunked_waits, 1,
        "one frame call is one blocking wait whatever the encode chunk is (it \
         took {chunked_waits})"
    );
    eprintln!(
        "[decode token rate] arm=frame submits={submits} waits={waits} \
         encode_chunk=whole_schedule chunked_submits={chunked_submits} \
         chunked_waits={chunked_waits} note=\"at the shipped chunk a product decode \
         step is {chunked_submits} queue submissions, one per encode chunk, plus the one \
         it spends folding the rows the step appends; this harness prices the \
         first term and not the fold\""
    );

    for _ in 0..WARMUPS {
        frame.run(WRITTEN_CELL);
    }
    let mut frame_samples: Vec<Vec<f64>> = vec![Vec::new(); WRITTENS.len()];
    for round in 0..ROUNDS {
        for k in 0..WRITTENS.len() {
            let variant = (k + round) % WRITTENS.len();
            let ms = frame.run(variant);
            frame_samples[variant].push(ms);
        }
    }
    for (variant, samples) in frame_samples.iter().enumerate() {
        let med = median(samples);
        // The read price, paired per round against the read-free control: the
        // fills all run in every round, so this difference sees one load.
        let delta = if variant == 0 {
            String::from("paired_minus_control_ms=+0.000 (the control)")
        } else {
            let paired: Vec<f64> = samples
                .iter()
                .zip(frame_samples[0].iter())
                .map(|(v, c)| v - c)
                .collect();
            let reads = kv_unique_bytes(WRITTENS[variant]) - kv_unique_bytes(WRITTENS[0]);
            let delta = median(&paired);
            let read = if delta > 0.0 {
                let rate = gbps(reads, delta);
                // A rate above what the machine can stream says the fill's reads
                // are served below DRAM, not that they ran at this rate.
                let note = match ceiling {
                    Some(ceiling) if rate > ceiling => {
                        format!(" (above the {ceiling:.1} GB/s ceiling — served below DRAM)")
                    }
                    None => String::from(" (no ceiling measured)"),
                    Some(_) => String::new(),
                };
                format!("implied_read_GBps={rate:.1}{note}")
            } else {
                // A fill that costs the control nothing: the reads ride the
                // weight stream, and a rate over a non-positive time would be a
                // number without a meaning.
                String::from("implied_read_GBps=none (the fill costs the control no time)")
            };
            format!("paired_minus_control_ms={delta:+.3} read_bytes={reads} {read}")
        };
        eprintln!(
            "[decode token rate] arm=frame written={} ms=[{}] min_ms={:.3} \
             median_ms={:.3} max_ms={:.3} GBps_weights={:.1} \
             GBps_weights_plus_kv={:.1} GBps_weights_plus_kv_twice={:.1} {delta}",
            WRITTENS[variant],
            samples
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            least(samples),
            med,
            greatest(samples),
            gbps(TOTAL_PACKED_BYTES, med),
            gbps(TOTAL_PACKED_BYTES + kv_unique_bytes(WRITTENS[variant]), med),
            gbps(TOTAL_PACKED_BYTES + kv_twice_bytes(WRITTENS[variant]), med),
        );
    }

    // The token's own figure, at the fill the walk ends on.
    let token_ms = median(&frame_samples[WRITTEN_CELL]);
    eprintln!(
        "[decode token rate] token_ms={token_ms:.3} token_ms_source=written_rows={}",
        WRITTENS[WRITTEN_CELL],
    );

    // ── A2: the submission shape at identical bytes and dispatches ─────────
    // The encode chunk is the one switch that turns one call into several queue
    // submissions without moving a byte or a step, so the sweep prices exactly
    // the per-submission term the decode's in-frame rate could be paying.
    let mut by_chunk: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); CHUNKS.len()];
    for round in 0..ROUNDS {
        for k in 0..CHUNKS.len() {
            let variant = (k + round) % CHUNKS.len();
            let ms = frame.run_at_chunk(WRITTEN_CELL, CHUNKS[variant]);
            by_chunk[variant].push(ms);
        }
    }
    let shipped = median(&by_chunk[0]);
    for (variant, samples) in by_chunk.iter().enumerate() {
        // The shipped chunk is the control the others are paired against, and is
        // labelled as one rather than printed against itself.
        let delta = if variant == 0 {
            String::from("paired_minus_shipped_ms=+0.000 (the shipped chunk)")
        } else {
            let paired: Vec<f64> = samples
                .iter()
                .zip(by_chunk[0].iter())
                .map(|(v, s)| v - s)
                .collect();
            format!(
                "paired_minus_shipped_ms={:+.3} faster_rounds={}",
                median(&paired),
                paired.iter().filter(|d| **d < 0.0).count(),
            )
        };
        eprintln!(
            "[decode token rate] arm=submission_shape chunk={} median_ms={:.3} \
             min_ms={:.3} max_ms={:.3} {delta}",
            CHUNKS[variant],
            median(samples),
            least(samples),
            greatest(samples),
        );
    }
    eprintln!(
        "[decode token rate] arm=submission_shape shipped_chunk={} \
         shipped_median_ms={shipped:.3} note=\"a product decode step is \
         ceil(steps/{}) submissions plus the one it spends folding the rows the \
         step appends; this harness prices the first term only\"",
        CHUNKS[0], CHUNKS[0],
    );

    // ── B: the per-shape arms, each built and dropped on its own ──────────
    let shapes: Vec<Steps> = SHAPE_ARMS.iter().map(measure_shape).collect();
    // The additive floor: every multiply back to back at its own per-step time,
    // weighted by how often one token walks that shape (a pin taken at compile
    // time beside the arms). Summing seven medians inherits all seven spreads, so
    // the floor is reported three times: at every arm's median, and at every
    // arm's least and greatest.
    let floor = |of: fn(&Steps) -> f64| {
        SHAPE_ARMS
            .iter()
            .zip(shapes.iter())
            .map(|(arm, s)| arm.copies as f64 * of(s))
            .sum::<f64>()
    };
    let additive_ms = floor(|s| s.median);
    let additive_least = floor(|s| s.least);
    let additive_greatest = floor(|s| s.greatest);
    let penalty = token_ms - additive_ms;
    eprintln!(
        "[decode token rate] additive_ms={additive_ms:.3} additive_least_ms={:.3} \
         additive_greatest_ms={:.3} token_ms={token_ms:.3} \
         in_frame_penalty_ms={penalty:+.3} \
         in_frame_penalty_pct_of_additive={:+.1}",
        additive_least,
        additive_greatest,
        penalty / additive_ms * 100.0,
    );

    // ── The query and key projections, separate and fused ─────────
    // Three calls at the same bytes per copy: the two separate multiplies, the
    // fused one, and the fused one with the narrow split it leaves behind. The
    // last is what a per-layer fusion would actually emit, so the saving it
    // reports is net of its own dispatch rather than a bound above one.
    let qk_steps = steps_for(QK_FUSED.bytes(), support::TARGET_BYTES);
    let forms = [QkForm::Split, QkForm::Fused, QkForm::FusedSplit];
    let mut probes: Vec<Probe> = forms.iter().map(|f| Probe::qk(*f, qk_steps)).collect();
    for (form, probe) in forms.iter().zip(probes.iter()) {
        probe.census(*form);
    }
    for probe in &mut probes {
        for _ in 0..WARMUPS {
            probe.ms();
        }
    }
    let mut qk_samples: Vec<Vec<f64>> = vec![Vec::new(); forms.len()];
    for round in 0..ROUNDS {
        for i in 0..forms.len() {
            let i = (i + round) % forms.len();
            let ms = probes[i].ms();
            qk_samples[i].push(ms);
        }
    }
    // Each form's level, and every difference as the **median of the per-round
    // paired differences**: the three forms run in every round, so a pair sees
    // one load and a level shift in the machine cancels.
    let per_token_ms = |ms: f64| ms / qk_steps as f64 * LAYERS as f64;
    let paired_saving = |a: &[f64], b: &[f64]| {
        let differences: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        per_token_ms(median(&differences))
    };
    let (split_ms, fused_ms, fused_split_ms) = (
        per_token_ms(median(&qk_samples[0])),
        per_token_ms(median(&qk_samples[1])),
        per_token_ms(median(&qk_samples[2])),
    );
    for (form, samples) in forms.iter().zip(qk_samples.iter()) {
        eprintln!(
            "[decode token rate] qk_fusion form={} per_copy_ms={:.4} \
             per_token_ms={:.3} ms_per_call=[{}]",
            form.label(),
            median(samples) / qk_steps as f64,
            per_token_ms(median(samples)),
            samples
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    eprintln!(
        "[decode token rate] qk_fusion split_ms_per_token={split_ms:.3} \
         fused_ms_per_token={fused_ms:.3} fused_and_split_ms_per_token={fused_split_ms:.3} \
         saving_before_split_ms_per_token={:+.3} \
         saving_net_of_split_ms_per_token={:+.3} narrow_split_cost_ms_per_token={:+.3} \
         note=\"the two savings are split minus the form, so positive is what a \
         per-layer q+k fusion would buy; the narrow-split figure is a cost, so \
         positive is what the split adds. Each is the median of the per-round paired \
         differences, at the bytes per copy the layer reads\"",
        paired_saving(&qk_samples[0], &qk_samples[1]),
        paired_saving(&qk_samples[0], &qk_samples[2]),
        paired_saving(&qk_samples[2], &qk_samples[1]),
    );

    // ── C: the dependency arm ─────────────────────────────────────────────
    let (mut dependency, dependency_build) = build_frame(Topology::Dependency, &plan);
    eprintln!(
        "[decode token rate] arm=dependency graph_build_ms={:.1} compile_ms={:.1} \
         upload_ms={:.1} uploaded_bytes={}",
        dependency_build.build,
        dependency_build.compile,
        dependency_build.upload,
        dependency_build.bytes
    );
    let dependency_steps = census(&dependency);
    check_logits(&mut dependency);
    // Both sides warm after the second frame's 1.1 GB upload, so neither pays
    // for the other's cold arena in the first paired round.
    for _ in 0..WARMUPS {
        frame.run(WRITTEN_CELL);
        dependency.run(WRITTEN_CELL);
    }
    let mut chain_frame = Vec::with_capacity(ROUNDS);
    let mut chain_dependency = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            chain_frame.push(frame.run(WRITTEN_CELL));
            chain_dependency.push(dependency.run(WRITTEN_CELL));
        } else {
            chain_dependency.push(dependency.run(WRITTEN_CELL));
            chain_frame.push(frame.run(WRITTEN_CELL));
        }
    }
    let paired: Vec<f64> = chain_frame
        .iter()
        .zip(chain_dependency.iter())
        .map(|(frame, dep)| frame - dep)
        .collect();
    let (frame_med, dependency_med) = (median(&chain_frame), median(&chain_dependency));
    eprintln!(
        "[decode token rate] arm=dependency frame_median_ms={frame_med:.3} \
         dependency_median_ms={dependency_med:.3} frame_steps={frame_steps} \
         dependency_steps={dependency_steps} dependency_extra_steps={} \
         frame_samples=[{}] dependency_samples=[{}] \
         paired_frame_minus_dependency_ms={:+.3} paired_pct={:+.1}",
        dependency_steps as i64 - frame_steps as i64,
        chain_frame
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        chain_dependency
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        median(&paired),
        median(&paired) / dependency_med * 100.0,
    );
    // The cells below hold 28 planes each; dropping the two frames first keeps
    // their weight streams from stacking on the cells' planes when the peak is read.
    drop(dependency);
    drop(frame);

    // ── D: the attention cells ────────────────────────────────────────────
    let mut cells: Vec<CellRun> = CELLS.iter().map(|c| build_cell(*c)).collect();
    for cell in &mut cells {
        for _ in 0..WARMUPS {
            cell.run();
        }
    }
    let mut cell_samples: Vec<Vec<f64>> = vec![Vec::new(); CELLS.len()];
    for round in 0..ROUNDS {
        for k in 0..CELLS.len() {
            let i = (k + round) % CELLS.len();
            let ms = cells[i].run();
            cell_samples[i].push(ms);
        }
    }
    for (i, cell) in cells.iter().enumerate() {
        let med = median(&cell_samples[i]);
        let unique = kv_bytes(WRITTEN_LAST, cell.cell.nkv);
        let per_query_head = unique * cell.cell.nh as u64 / cell.cell.nkv as u64;
        eprintln!(
            "[decode token rate] cell={} nh={} nkv={} ms=[{}] min_ms={:.3} \
             median_ms={:.3} max_ms={:.3} kv_unique_bytes={unique} \
             kv_per_query_head_bytes={per_query_head} \
             GBps_unique_kv={:.2} GBps_per_query_head_kv={:.2} \
             key_splits={} decode_grid_workgroups={}",
            cell.cell.label(),
            cell.cell.nh,
            cell.cell.nkv,
            cell_samples[i]
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            least(&cell_samples[i]),
            med,
            greatest(&cell_samples[i]),
            gbps(unique, med),
            gbps(per_query_head, med),
            cell.cell.key_splits(),
            cell.cell.grid(),
        );
    }
    // The four cells read against the shipped (16, 8): each of the other three
    // moves one of the three quantities the time could follow — the bytes fetched
    // once (with `nkv`), the per-query-head walk (with `nh`) and the decode grid
    // (`heads × key_splits`). Printed as ratios, not scored: a single verdict
    // would have to weigh three axes at once, and the (8, 8) cell moves two of
    // them together.
    let base = median(&cell_samples[0]);
    for (i, cell) in cells.iter().enumerate().skip(1) {
        eprintln!(
            "[decode token rate] cell={} ms_ratio={:.3} unique_bytes_ratio={:.3} \
             per_query_head_bytes_ratio={:.3} grid_ratio={:.3}",
            cell.cell.label(),
            median(&cell_samples[i]) / base,
            cell.cell.nkv as f64 / CELLS[0].nkv as f64,
            cell.cell.nh as f64 / CELLS[0].nh as f64,
            cell.cell.grid() as f64 / CELLS[0].grid() as f64,
        );
    }

    eprintln!(
        "[decode token rate] process_peak={} note=\"this process's peak resident \
         set, which includes the stream-ceiling probe's own buffers; the harness's \
         own frame, not a stand figure\"",
        report::shown(report::peak_rss())
    );
}
