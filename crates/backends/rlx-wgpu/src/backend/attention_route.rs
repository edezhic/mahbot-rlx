// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether a many-row `Op::Attention` step takes the dense arrangement — one
//! `Step::Matmul` of the scores, a `Step::AttentionMask` and a `Step::Softmax`
//! over them, and one `Step::Matmul` by V — in place of one
//! `Step::Attention` in which a single thread walks the whole key sequence
//! alone. Shapes and strides only, so it is testable without a device, and
//! [`plan_attention`] the one derivation of it from a node: the reserve in
//! [`crate::backend`] and the `Op::Attention` arm of `compile::lower` both read
//! the plan it returns.
//!
//! Why: the fused kernel runs one thread per (batch, head, query row), each
//! reading all of K and V itself, so the work never fills the machine as the row
//! count grows. Measured by `tests/attention_decompose_cost.rs` on this machine
//! (Metal, M3 Pro; 8 layers, 8 heads, head_dim 64, causal; a few percent run to
//! run): this route reaches **≈40 GF/s at 32 query rows over 1024 keys, ≈80 at
//! 64, ≈110 at 100 and ≈190 at 256**, and ≈33-130 GF/s at those row counts over
//! 256 keys. The fused side is deliberately not quoted as a rate: that harness's
//! own note is that a reading of it is mostly the fixed submit/wait/readback
//! floor, and the fused step at 24-31 rows is small enough to sit on that floor —
//! so what it gives is a bound on the fused step, not a measurement of it, and
//! the two arrangements are compared by their ratio there instead: 18x over 1024
//! keys and ≈14x over 256. The same reading does say two things about that step
//! without being a rate: it barely moves with the row count (≈1% from 24 to 31
//! rows, against a third more work), and it grows with the key sequence (≈4.2x
//! from 256 to 1024 keys at 24 rows) — the serial key walk showing on top of the
//! floor. The crossover the floor sits inside was read off-device: nothing at
//! 13-24 query rows, an order of magnitude at about 100.
//!
//! Why steps and not a graph decomposition: on a discrete Vulkan/DX12 adapter
//! the ops this would decompose into are hosted — `Op::Softmax` / `Op::Binary` /
//! `Op::Where` by `should_pack_host_op` and `Op::Transpose` by its own
//! structure-host switch — while `Op::MatMul` is the one op exempt from that
//! rule. A graph-level decomposition would run the scale, the mask and the
//! normalisation on the CPU on exactly those adapters; as steps, every piece of
//! it stays on the device.
//!
//! Every term above is the graph's own, including WebGPU's per-dimension
//! dispatch cap ([`MAX_WORKGROUPS_PER_DIM`], which both multiplies reach through
//! their `batch · heads` z extent and their `seq_q`-row y extent). The one
//! condition the geometry here cannot settle, so the `Op::Attention` arm applies
//! it, is the window: the route's steps bind their operands *and* the regions
//! this plan reserves as a single range, so the arena has to address itself whole
//! at the base the operand resolve left behind (`!Arena::is_sharded()`) and that
//! window has to reach the regions
//! (`route_end <= base + size`). Both hold wherever the arena fits one bind
//! window — the case the route is measured in; a step that fails either keeps the
//! fused kernel, which stages each operand into whatever window it has. Not a
//! silent decline: [`crate::backend::WgpuExecutable::attention_window_declines`]
//! counts it per compiled graph and the schedule report
//! (`RLX_WGPU_SCHEDULE`) prints it, and the arm's attention debug print
//! (`RLX_WGPU_DEBUG_ATTN_MASK`) names the term for the step's own addresses.

use rlx_ir::op::MaskKind;

use super::MAX_WORKGROUPS_PER_DIM;

/// Query rows below which the dense arrangement is not worth its two extra
/// passes. The re-grouping measures nothing at 13-24 query rows and an order of
/// magnitude at about 100, so the boundary has to fall inside that range, and
/// this is the first power of two past its lower end — the measured pair either
/// side of it is in `tests/attention_decompose_cost.rs` and in the note above.
pub const ATTENTION_DECOMPOSE_MIN_Q: u32 = 32;

/// Head width the route serves, mirroring the fused kernel's own `MAX_HEAD_DIM`
/// (`kernels/attention.wgsl`). Neither multiply on this route needs the cap —
/// the tiled f32 kernels take any k and n — but a shape the fused kernel refuses
/// must not become covered by the route that replaces it, so the two ceilings
/// move together. It is also the ceiling [`plan_attention`] refuses above, so
/// this is the one width both the refusal and the route decision read.
pub(crate) const ATTENTION_DECOMPOSE_MAX_HEAD_DIM: u32 = 512;

/// The route must not need more scratch than this. [`crate::backend::compute_scratch_bytes`]
/// floors the tail reserve at 64 MiB, so this is the figure every graph is
/// guaranteed to have; a route that needed more would have to size the tail
/// itself, and a score matrix past this belongs on the fused kernel (or on a
/// chunked step) rather than in a per-step allocation this backend does not have.
pub(crate) const ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES: u64 = 64 << 20;

/// Rows per workgroup the route's multiplies dispatch their `m = seq_q` rows in:
/// 32 is the narrowest row tile of the kernels the run can pick for them (the
/// plain f32 kernel's, and `WideVariant::Std`'s — the vectorised variants take
/// 64), so it is the conservative divisor for that extent's cap. Only that
/// extent needs one: `n` is `seq_k` on the score multiply and `v_head_dim` on the
/// value one, and the scratch conjunct already bounds `tiles · seq_k` by the
/// reserve's own `16.7M` floats, which keeps `n`'s tiles (at most 16384 keys per
/// multiply) far below the cap — an argument that needs the tile count to be at
/// least one, which [`decompose_attention`] requires.
const MATMUL_TILE_M: u32 = 32;

/// Scratch regions are bumped through 256-byte aligned (`buffer::Arena`'s own
/// window granularity), so each region's element count is rounded up to that
/// alignment before it is summed. The rounded figure is what the lowering bumps
/// through, which is what lets the reserve and the allocation agree exactly.
const SCRATCH_ALIGN_FLOATS: u64 = 256 / 4;

/// The shape half of the route decision: the step's query rows, its two head
/// widths, the tiles both multiplies walk and the float count its scratch needs.
/// No device, no adapter, no arena — so the decision is testable without a
/// device, and the tests below build every field rather than describing a shape
/// wrongly.
pub(crate) struct AttentionGeometry {
    pub seq_q: u32,
    /// The key sequence the step walks.
    pub seq_k: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// `batch · heads`: the (batch, head) tiles both multiplies walk, which the
    /// dispatch carries as its own z extent.
    pub tiles: u64,
    /// Floats the route's scratch needs: Kᵀ + the score matrix + whatever
    /// compaction the operand layouts force. See [`scratch_regions`].
    pub scratch_floats: u64,
}

/// Whether the many-row form — the dense multiply of the scores, the
/// operation's own masked normalisation, and a dense multiply by V — serves
/// this step. Every conjunct is the graph's own — the two dispatch extents the
/// multiplies walk included — so the arm's route gate needs nothing from the
/// adapter (see the module doc). Two of them are degenerate shapes the fused
/// path still runs: `tiles >= 1`, since with no `(batch, head)` tiles there is no
/// multiply to dispatch and the `n`-extent argument below holds vacuously
/// however long the key sequence is; and `seq_k >= 1`, since every region would
/// be empty, the budget conjunct would pass vacuously, and the route would write
/// zeros where the fused kernel's 0/0 normalisation is the NaN the operation is.
pub(crate) fn decompose_attention(g: &AttentionGeometry) -> bool {
    g.seq_q >= ATTENTION_DECOMPOSE_MIN_Q
        && g.seq_k >= 1
        && g.tiles >= 1
        && g.seq_q.div_ceil(MATMUL_TILE_M) <= MAX_WORKGROUPS_PER_DIM
        && g.tiles <= MAX_WORKGROUPS_PER_DIM as u64
        && g.head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        && g.v_head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM
        && g.scratch_floats.saturating_mul(4) <= ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES
}

/// Whether an operand's own per-`(batch, head)` tile is already a dense
/// `[seq, width]` block, and so can be read or written where it lies instead of
/// through a compaction transpose. `head_stride == seq · width` is what makes
/// the tiles contiguous *and* in `(batch, head)` order, which is the order the
/// multiplies' batch axis walks them in.
pub(crate) fn operand_tile_is_dense(
    seq_stride: u32,
    head_stride: u32,
    seq: u32,
    width: u32,
) -> bool {
    seq_stride == width && head_stride == seq * width
}

/// The route's scratch regions, in the fixed order the lowering bumps through
/// them, in f32 elements and already rounded up to [`SCRATCH_ALIGN_FLOATS`].
/// `0` marks a region the route does not emit (that operand's tiles are dense,
/// or its output slot is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionScratchRegions {
    /// Kᵀ, `[B, H, D, S_k]` — always: the score multiply reads B as `[k=D, n=S_k]`.
    pub kt: u64,
    /// The score matrix `[B, H, S_q, S_k]`, normalised in place into P.
    pub scores: u64,
    /// Q′ `[B, H, S_q, D]`, only when Q's tile is not dense.
    pub q: u64,
    /// V′ `[B, H, S_k, D_v]`, only when V's tile is not dense.
    pub v: u64,
    /// out′ `[B, H, S_q, D_v]`, only when the output slot is not dense per
    /// `(batch, head)` — otherwise the second multiply writes the slot directly.
    pub out: u64,
}

impl AttentionScratchRegions {
    pub(crate) fn total(self) -> u64 {
        self.kt + self.scores + self.q + self.v + self.out
    }
}

/// The regions a step's layout forces. One place, so the figure the route
/// decision reads and the offsets the lowering bumps through cannot disagree.
pub(crate) fn scratch_regions(
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    v_head_dim: u32,
    q_dense: bool,
    v_dense: bool,
    out_dense: bool,
) -> AttentionScratchRegions {
    let align = |floats: u64| floats.div_ceil(SCRATCH_ALIGN_FLOATS) * SCRATCH_ALIGN_FLOATS;
    // The (batch, head) tiles every region is sized in. GQA/MQA expands here:
    // Kᵀ and V′ are written per query head, so that the multiplies' batch axis
    // is one dense tile per (b, h) and needs no per-head indirection.
    let tiles = (batch as u64) * (heads as u64);
    AttentionScratchRegions {
        kt: align(tiles * head_dim as u64 * seq_k as u64),
        scores: align(tiles * seq_q as u64 * seq_k as u64),
        q: if q_dense {
            0
        } else {
            align(tiles * seq_q as u64 * head_dim as u64)
        },
        v: if v_dense {
            0
        } else {
            v_compaction_floats(batch, heads, seq_k, v_head_dim)
        },
        out: if out_dense {
            0
        } else {
            align(tiles * seq_q as u64 * v_head_dim as u64)
        },
    }
}

/// The region a compacted V′ takes: one dense `[S_k, D_v]` tile per query head,
/// rounded up to the bump's alignment. The one home of its size — the layout
/// above and the plan's booked spare (which the lowering spends when the output
/// slot lands on V; see [`AttentionPlan::v_spare`]) both read it here.
fn v_compaction_floats(batch: u32, heads: u32, seq_k: u32, v_head_dim: u32) -> u64 {
    let floats = (batch as u64) * (heads as u64) * (seq_k as u64) * (v_head_dim as u64);
    floats.div_ceil(SCRATCH_ALIGN_FLOATS) * SCRATCH_ALIGN_FLOATS
}

/// The slack the tail reserve carries on top of the regions: the bump cursor is
/// only 16-byte aligned when the route starts (`Arena::scratch_off`), so the
/// first region can begin up to one alignment past it.
pub(crate) const SCRATCH_ALIGN_SLACK_BYTES: u64 = 256;

/// Everything the many-row route needs to know about one `Op::Attention` node:
/// the batch/head/sequence geometry, the KV-head count, the strides Q/K/V and the
/// output are read and written through, the mask's kind/numbering and strides,
/// which per-`(batch, head)` tiles are already dense, the scratch regions those
/// tiles force, and whether the route serves the step at all. Derived from the
/// graph alone — see [`plan_attention`].
///
/// A plan is one `Op::Attention` node's *geometry*, not its operands or their
/// addresses: the offsets the route reads them at come from the arena and the
/// schedule, which this module deliberately cannot see.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionPlan {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// KV heads (GQA/MQA). `heads` where a packed QKV parent carries K and V,
    /// whose strides are uniform across the three operands.
    pub nkv: u32,
    /// The `[B, S, 3, H, D]` parent Q/K/V are axis-2 narrows of, with the width
    /// of one head in elements, where one exists and
    /// `RLX_WGPU_NO_PACKED_BSHD_ATTN` is clear.
    pub packed_parent: Option<(rlx_ir::NodeId, u32)>,
    pub q_b: u32,
    pub q_h: u32,
    pub q_s: u32,
    pub k_b: u32,
    pub k_h: u32,
    pub k_s: u32,
    pub v_b: u32,
    pub v_h: u32,
    pub v_s: u32,
    pub o_b: u32,
    pub o_h: u32,
    pub o_s: u32,
    /// The mask's own kind, in the fused kernel's numbering: 0 none, 1 causal,
    /// 2 binary key padding, 3 sliding window, 4 additive bias.
    pub mask_kind_id: u32,
    /// Sliding-window width, 0 for every other kind.
    pub window: u32,
    pub m_b: u32,
    pub m_h: u32,
    pub m_q: u32,
    pub m_k: u32,
    /// Whether Q's / V's / the output's per-`(batch, head)` tile is already a
    /// dense `[seq, width]` block (see [`operand_tile_is_dense`]).
    pub q_dense: bool,
    pub v_dense: bool,
    pub out_dense: bool,
    /// The regions those three flags force, in the order the route bumps them.
    pub regions: AttentionScratchRegions,
    /// V′'s region for the case [`regions`](Self::regions) leaves out: V's tiles
    /// are dense, but the planned output slot lies on V's own bytes, so the
    /// lowering compacts V anyway. 0 whenever `regions` already holds it.
    pub v_spare: u64,
    /// Whether the route serves the step: the shape half of the decision
    /// ([`decompose_attention`]) over this plan's own geometry and regions, with
    /// `nkv` dividing `heads` for the grouped-KV layouts.
    pub decompose: bool,
}

/// Where each of the route's scratch regions starts, in bytes from the bump's
/// start, plus one past the last of them — the order the lowering bumps through
/// them in, derived once here so the arm's offsets and the reserve cannot drift
/// apart (see [`AttentionPlan::scratch_offsets`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionScratchOffsets {
    pub kt: u64,
    pub scores: u64,
    pub q: u64,
    pub v: u64,
    pub out: u64,
    /// One past the last region: what a bump starting at the aligned cursor
    /// reaches, and never past [`AttentionPlan::scratch_bytes`].
    pub end: u64,
}

impl AttentionPlan {
    /// The tail-scratch bytes the route needs: the aligned regions — plus V′'s
    /// booked spare, since the lowering may have to compact a V it would
    /// otherwise read where it lies — and the one alignment the bump cursor
    /// itself can start short of ([`SCRATCH_ALIGN_SLACK_BYTES`]). One place, so
    /// the reserve and the bump cannot drift, and never less than the bump the
    /// arm performs.
    pub(crate) fn scratch_bytes(&self) -> usize {
        ((self.regions.total() + self.v_spare) * 4 + SCRATCH_ALIGN_SLACK_BYTES) as usize
    }

    /// The region offsets for an arm that has (`v_compacted`) or has not had to
    /// compact V — the arm's own answer, since what decides it is its output
    /// slot's overlap with V (see the arm's note), and the plan books that
    /// compaction as [`v_spare`](Self::v_spare). `end` is at most
    /// [`scratch_bytes`](Self::scratch_bytes): the spare is booked for exactly
    /// this compaction and the slack covers the cursor's alignment.
    pub(crate) fn scratch_offsets(&self, v_compacted: bool) -> AttentionScratchOffsets {
        let r = self.regions;
        let kt = 0;
        let scores = kt + r.kt * 4;
        let q = scores + r.scores * 4;
        let v = q + r.q * 4;
        let out = v + if v_compacted {
            r.v * 4 + self.v_spare * 4
        } else {
            0
        };
        AttentionScratchOffsets {
            kt,
            scores,
            q,
            v,
            out,
            end: out + r.out * 4,
        }
    }
}

/// Why a node has no plan, carrying the message the lowering's own panic for
/// that shape prints. Every variant but [`NotAttention`](Self::NotAttention) and
/// [`HeadWidth`](Self::HeadWidth) describes a step the fused path cannot derive a
/// shape for either — the arms that hit them index a dim that is not there — so a
/// refusal is never a shape the route could have taken. [`HeadWidth`](Self::HeadWidth)
/// is the exception: that shape has geometry the fused path could size, and the
/// kernel would then early-return without writing the output slot, which is why
/// the width is refused here rather than lowered to a blank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionRefusal {
    /// The node is not an `Op::Attention` — what the scratch reserve, which
    /// walks every node of a graph, gets for all of them.
    NotAttention,
    /// An `Op::Attention` with fewer than three inputs.
    Arity { inputs: usize },
    /// An `Op::Attention` with a zero head count or head width.
    ZeroWidth { num_heads: usize, head_dim: usize },
    /// An `Op::Attention` whose Q, K, V or output shape is not fully static.
    NonStatic,
    /// Q, K or V is neither rank-3 nor rank-4.
    Rank { rank: usize },
    /// Rank-3 `[B·H, S, D]` whose leading dim is not a multiple of the head
    /// count (and whose last dim is not the `[B, S, H·D]` feature width).
    Rank3Leading {
        leading: usize,
        heads: usize,
        head_dim: usize,
        last: usize,
    },
    /// A mask kind that reads `node.inputs[3]` on a node that has fewer inputs.
    MissingMask { kind: u32, inputs: usize },
    /// A head width — after the kernel's own "0 means V is as wide as Q"
    /// resolution — above [`ATTENTION_DECOMPOSE_MAX_HEAD_DIM`], which the fused
    /// kernel's `MAX_HEAD_DIM` and this route's decision both cap at.
    HeadWidth { head_dim: u32, v_head_dim: u32 },
}

impl std::fmt::Display for AttentionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAttention => write!(f, "rlx-wgpu Attention: node is not an Attention node"),
            Self::Arity { inputs } => write!(
                f,
                "rlx-wgpu Attention: needs Q, K and V (node has {inputs} inputs)"
            ),
            Self::ZeroWidth {
                num_heads,
                head_dim,
            } => write!(
                f,
                "rlx-wgpu Attention: zero num_heads or head_dim ({num_heads} heads of {head_dim})"
            ),
            Self::NonStatic => write!(
                f,
                "rlx-wgpu Attention: Q, K, V and the output shape must be static"
            ),
            Self::Rank { rank } => write!(
                f,
                "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V inputs supported (got rank {rank})"
            ),
            Self::Rank3Leading {
                leading,
                heads,
                head_dim,
                last,
            } => write!(
                f,
                "rlx-wgpu Attention: rank-3 leading dim {leading} not divisible by num_heads \
                 {heads} (and last dim {last} ≠ H·D = {})",
                heads * head_dim
            ),
            Self::MissingMask { kind, inputs } => write!(
                f,
                "rlx-wgpu Attention: mask kind {kind} needs a mask input (node has {inputs} inputs)"
            ),
            Self::HeadWidth {
                head_dim,
                v_head_dim,
            } => write!(
                f,
                "rlx-wgpu Attention: head_dim {head_dim} / v_head_dim {v_head_dim} is above the \
                 {ATTENTION_DECOMPOSE_MAX_HEAD_DIM} this route serves"
            ),
        }
    }
}

/// The route's geometry for one node of a graph, or why the node has none.
///
/// Every node of any graph gets an answer, because the scratch reserve in
/// [`crate::backend`] walks all of them: a node that is not an `Op::Attention`
/// gets [`AttentionRefusal::NotAttention`], and one whose shapes the arm cannot
/// derive geometry for gets the refusal that names the dim. The refusals are
/// what the lowering's own arm panics with — there is no plan with no geometry
/// for it to read instead, so a shape this cannot plan is loud rather than a
/// step the kernel early-returns from, the over-ceiling head width included.
pub(crate) fn plan_attention(
    graph: &rlx_ir::Graph,
    node: &rlx_ir::Node,
) -> Result<AttentionPlan, AttentionRefusal> {
    let rlx_ir::Op::Attention {
        num_heads,
        head_dim,
        v_head_dim,
        mask_kind,
        ..
    } = &node.op
    else {
        return Err(AttentionRefusal::NotAttention);
    };
    if node.inputs.len() < 3 {
        return Err(AttentionRefusal::Arity {
            inputs: node.inputs.len(),
        });
    }
    if *head_dim == 0 || *num_heads == 0 {
        return Err(AttentionRefusal::ZeroWidth {
            num_heads: *num_heads,
            head_dim: *head_dim,
        });
    }
    let (q_id, k_id, v_id) = (node.inputs[0], node.inputs[1], node.inputs[2]);
    let q_ir = &graph.node(q_id).shape;
    let k_ir = &graph.node(k_id).shape;
    let v_ir = &graph.node(v_id).shape;
    let out_ir = &node.shape;
    if let Some(rank) = [q_ir.rank(), k_ir.rank(), v_ir.rank()]
        .into_iter()
        .find(|rank| !matches!(rank, 3 | 4))
    {
        return Err(AttentionRefusal::Rank { rank });
    }
    // A dim the derivation reads that the graph has not pinned means no route.
    // The arm calls `unwrap_static` on these and panics; this must not, and a
    // route with an unknown sequence or width could not be sized anyway.
    if !(q_ir.is_static() && k_ir.is_static() && v_ir.is_static() && out_ir.is_static()) {
        return Err(AttentionRefusal::NonStatic);
    }
    let h = *num_heads as u32;
    let hd = *head_dim as u32;
    // A zero value-head width is the kernel's own spelling of "V is as wide as Q"
    // (`kernels/attention.wgsl` reads `v_head_dim` through a `select` that takes
    // `head_dim` when it is 0), so the plan resolves it the same way. Sizing V′,
    // the output region and the P·V multiply at width 0 instead would dispatch no
    // workgroups and leave the output slot unwritten.
    let hd_v = match *v_head_dim {
        Some(v) if v > 0 => v as u32,
        _ => hd,
    };
    // The one width above which neither the fused kernel (whose per-thread Q/O
    // arrays are `MAX_HEAD_DIM` wide) nor the dense route serves the step: the
    // kernel would return without writing the output slot, so the step is
    // refused here instead — naming the shape — rather than lowered to a blank.
    if hd > ATTENTION_DECOMPOSE_MAX_HEAD_DIM || hd_v > ATTENTION_DECOMPOSE_MAX_HEAD_DIM {
        return Err(AttentionRefusal::HeadWidth {
            head_dim: hd,
            v_head_dim: hd_v,
        });
    }
    let dim = |d: rlx_ir::shape::Dim| d.unwrap_static() as u32;
    // Rank-4 resolves through the shared geometry; rank-3 has the two layouts
    // `attention_geom` cannot tell apart, told apart here by the last dim.
    let geom = rlx_ir::attention_geom(q_ir, k_ir, *num_heads, *head_dim);
    let bhsd = geom.bhsd;
    let (batch, seq_q, seq_k) = match q_ir.rank() {
        4 => (geom.batch as u32, geom.seq_q as u32, geom.seq_k as u32),
        _ => {
            let (last, leading) = (dim(q_ir.dim(2)), dim(q_ir.dim(0)));
            let (seq_q, seq_k) = (dim(q_ir.dim(1)), dim(k_ir.dim(1)));
            if last == h * hd {
                // [B, S, H·D] — leading is the batch.
                (leading, seq_q, seq_k)
            } else if !leading.is_multiple_of(h) {
                // [B·H, S, D] with a leading dim that is not B·H.
                return Err(AttentionRefusal::Rank3Leading {
                    leading: leading as usize,
                    heads: h as usize,
                    head_dim: *head_dim,
                    last: last as usize,
                });
            } else {
                (leading / h, seq_q, seq_k)
            }
        }
    };
    let (mask_kind_id, window) = match mask_kind {
        MaskKind::None => (0u32, 0u32),
        MaskKind::Causal => (1, 0),
        // 2 = binary key-padding mask (Custom: <0.5 → -inf); 4 = additive bias
        // mask (Bias: score += mask). These are NOT interchangeable — the
        // encoder's block-diagonal winmask is additive, so folding it into the
        // binary path silently corrupts attention.
        MaskKind::Custom => (2, 0),
        MaskKind::Bias => (4, 0),
        MaskKind::SlidingWindow(w) => (3, *w as u32),
    };
    let mask_read = mask_kind_id == 2 || mask_kind_id == 4;
    if mask_read && node.inputs.len() < 4 {
        return Err(AttentionRefusal::MissingMask {
            kind: mask_kind_id,
            inputs: node.inputs.len(),
        });
    }
    if mask_read && !graph.node(node.inputs[3]).shape.is_static() {
        return Err(AttentionRefusal::NonStatic);
    }
    // Mask address strides, from the mask's own shape for the kinds that read
    // one and the canonical [B, H, S_q, S_k] form otherwise (the kernel's
    // mask_partial computation is harmless when the mask is not read).
    let (m_b, m_h, m_q, m_k) = if mask_read {
        rlx_ir::mask_strides_for_shape(graph.node(node.inputs[3]).shape.dims(), h, seq_q, seq_k)
    } else {
        rlx_ir::mask_strides_bhsd(h, seq_q, seq_k)
    };
    let packed_parent = if rlx_ir::env::flag("RLX_WGPU_NO_PACKED_BSHD_ATTN") {
        None
    } else {
        rlx_ir::detect_packed_bshd_qkv_attention(graph, q_id, k_id, v_id)
            .map(|(parent, head_width, _)| (parent, head_width as u32))
    };
    // GQA/MQA: the KV-head count is layout-independent — K's element count over
    // B·S_k·D. Packed QKV shares Q's head count (uniform strides), so it is MHA.
    let nkv: u32 = match packed_parent {
        Some(_) => h,
        None => {
            let k_elems = k_ir
                .dims()
                .iter()
                .fold(1u32, |elems, d| elems.saturating_mul(dim(*d)));
            let tiles = batch
                .max(1)
                .saturating_mul(seq_k.max(1))
                .saturating_mul(hd.max(1));
            (k_elems / tiles).max(1)
        }
    };
    let (q_b, q_h, q_s, k_b, k_h, k_s, v_b, v_h, v_s) = match packed_parent {
        Some((_, head_width)) => {
            let (b, head, pack_seq) =
                rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
            (b, head, pack_seq, b, head, pack_seq, b, head, pack_seq)
        }
        None => {
            let (qb, qh, qs) = rlx_ir::strides_for_shape(q_ir.dims(), h, hd, seq_q, bhsd);
            // K/V carry `nkv` heads (GQA): the layout test inside steps by
            // nkv·D, and V's rows are `hd_v` wide (MLA-asymmetric).
            let (kb, kh, ks) = rlx_ir::strides_for_shape(k_ir.dims(), nkv, hd, seq_k, bhsd);
            let (vb, vh, vs) = rlx_ir::strides_for_shape(v_ir.dims(), nkv, hd_v, seq_k, bhsd);
            (qb, qh, qs, kb, kh, ks, vb, vh, vs)
        }
    };
    // The output rows are `hd_v` wide (asymmetric SDPA == V's width).
    let (o_b, o_h, o_s) = rlx_ir::strides_for_shape(out_ir.dims(), h, hd_v, seq_q, bhsd);
    // A per-(b, h) tile that is dense is read or written where it lies; one that
    // is not is compacted through a transpose.
    let q_dense = operand_tile_is_dense(q_s, q_h, seq_q, hd);
    let v_dense = nkv == h && operand_tile_is_dense(v_s, v_h, seq_k, hd_v);
    let out_dense = operand_tile_is_dense(o_s, o_h, seq_q, hd_v);
    let regions = scratch_regions(
        batch, h, seq_q, seq_k, hd, hd_v, q_dense, v_dense, out_dense,
    );
    // V′'s region for the case where the default layout reads V where it lies but
    // the lowering still has to compact it: an output slot the planner placed on
    // V's own bytes cannot be written by the multiply that reads V (see the arm's
    // output-overlap note). Booked either way, so that case never needs more
    // scratch than the reserve holds.
    let v_spare = if v_dense {
        v_compaction_floats(batch, h, seq_k, hd_v)
    } else {
        0
    };
    // The shape half of the decision, plus the one layout condition it needs:
    // grouped KV reads K and V through one (b, h) tile per query head, so the KV
    // group has to divide the head count exactly — `heads / nkv` is the fused
    // kernel's own kv-head divisor, and the route's Kᵀ and V′ transposes walk
    // that group as an axis. A shape where it does not divide is one that kernel
    // already indexes past its KV heads.
    let decompose = decompose_attention(&AttentionGeometry {
        seq_q,
        seq_k,
        head_dim: hd,
        v_head_dim: hd_v,
        tiles: batch as u64 * h as u64,
        scratch_floats: regions.total() + v_spare,
    }) && h.is_multiple_of(nkv.max(1));
    Ok(AttentionPlan {
        batch,
        heads: h,
        seq_q,
        seq_k,
        head_dim: hd,
        v_head_dim: hd_v,
        nkv,
        packed_parent,
        q_b,
        q_h,
        q_s,
        k_b,
        k_h,
        k_s,
        v_b,
        v_h,
        v_s,
        o_b,
        o_h,
        o_s,
        mask_kind_id,
        window,
        m_b,
        m_h,
        m_q,
        m_k,
        q_dense,
        v_dense,
        out_dense,
        regions,
        v_spare,
        decompose,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Graph, Op, Shape};

    /// The measured shape family: 8 heads of head_dim 64, both widths clear of
    /// the ceiling, a single batch, and the scratch computed from the same
    /// arguments the lowering passes (all three tiles compacted, the layout that
    /// needs the most).
    fn geom(seq_q: u32, seq_k: u32) -> AttentionGeometry {
        let (batch, heads, head_dim, v_head_dim) = (1u32, 8u32, 64u32, 64u32);
        AttentionGeometry {
            seq_q,
            seq_k,
            head_dim,
            v_head_dim,
            tiles: batch as u64 * heads as u64,
            scratch_floats: scratch_regions(
                batch, heads, seq_q, seq_k, head_dim, v_head_dim, false, false, false,
            )
            .total(),
        }
    }

    /// One `Op::Attention` node over Q/K/V inputs of the given shapes, and the
    /// plan its own geometry derives. No mask: the kinds that read one are
    /// exercised below by hand.
    fn plan_of(
        q: &[usize],
        k: &[usize],
        v: &[usize],
        out: &[usize],
        heads: usize,
        head_dim: usize,
        v_head_dim: Option<usize>,
    ) -> Result<AttentionPlan, AttentionRefusal> {
        let mut g = Graph::new("attention_plan");
        let (q_id, k_id, v_id) = (
            g.input("q", Shape::new(q, DType::F32)),
            g.input("k", Shape::new(k, DType::F32)),
            g.input("v", Shape::new(v, DType::F32)),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: heads,
                head_dim,
                v_head_dim,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_id, k_id, v_id],
            Shape::new(out, DType::F32),
        );
        plan_attention(&g, g.node(attn))
    }

    /// The floor's two sides and the measured bracket it has to fall inside: the
    /// rule is a bare `>=` on the step's own query rows, the band below it
    /// (13-24 rows, where the re-grouping returned nothing) is refused and the
    /// order-of-magnitude end (about 100 rows) is served.
    #[test]
    fn the_floor_sits_inside_the_measured_bracket() {
        assert!(decompose_attention(&geom(32, 1024)));
        assert!(!decompose_attention(&geom(31, 1024)));
        assert!(!decompose_attention(&geom(24, 1024)));
        assert!(!decompose_attention(&geom(13, 1024)));
        assert!(decompose_attention(&geom(100, 1024)));
    }

    /// The head ceiling is the fused kernel's: the width equal to it is served
    /// by both forms, one past it neither. Applies to V's own width as well,
    /// since the shared `MAX_HEAD_DIM` is compared against both.
    #[test]
    fn the_head_ceiling_is_the_fused_kernels_own() {
        let mut at = geom(64, 256);
        at.head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM;
        at.v_head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM;
        assert!(decompose_attention(&at));
        let mut over = geom(64, 256);
        over.head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM + 1;
        assert!(!decompose_attention(&over));
        let mut wide_v = geom(64, 256);
        wide_v.v_head_dim = ATTENTION_DECOMPOSE_MAX_HEAD_DIM + 1;
        assert!(!decompose_attention(&wide_v));
    }

    /// A shape past that ceiling is one the fused path *can* size — and whose
    /// kernel then returns without writing the output slot — so it has no plan at
    /// all rather than a plan the arm would lower to a blank. The refusal names
    /// the operator and the width, and the plan-side ceiling is the shader's own
    /// `MAX_HEAD_DIM`, so the two cannot be raised apart (the silent blank
    /// returns if only this one moves).
    #[test]
    fn a_head_width_past_the_ceiling_is_refused_rather_than_left_blank() {
        const SHADER: &str = include_str!("../kernels/attention.wgsl");
        assert!(
            SHADER.contains(&format!(
                "const MAX_HEAD_DIM: u32 = {ATTENTION_DECOMPOSE_MAX_HEAD_DIM}u;"
            )),
            "the plan's ceiling and the fused kernel's `MAX_HEAD_DIM` are one number: a shape \
             the plan serves must not be one the kernel early-returns from"
        );

        // The widths are the whole subject; the other dims are a served shape's.
        // `v_head_dim` of 0 is the kernel's own "as wide as Q" spelling, so the
        // shapes resolve it the same way the plan does.
        let plan = |hd: usize, v_hd: Option<usize>| {
            let vd = match v_hd {
                Some(v) if v > 0 => v,
                _ => hd,
            };
            plan_of(
                &[1, 4, hd],
                &[1, 8, hd],
                &[1, 8, vd],
                &[1, 4, vd],
                1,
                hd,
                v_hd,
            )
        };

        let at = ATTENTION_DECOMPOSE_MAX_HEAD_DIM as usize;
        assert!(
            plan(at, None).is_ok(),
            "the width equal to the ceiling is served"
        );
        assert_eq!(
            plan(at + 1, None).unwrap_err(),
            AttentionRefusal::HeadWidth {
                head_dim: at as u32 + 1,
                v_head_dim: at as u32 + 1,
            }
        );
        // V's own width, which the kernel caps the same way, and the 0 spelling.
        assert_eq!(
            plan(at, Some(at + 1)).unwrap_err(),
            AttentionRefusal::HeadWidth {
                head_dim: at as u32,
                v_head_dim: at as u32 + 1,
            }
        );
        assert!(
            plan(64, Some(0)).is_ok(),
            "a zero width means V is as wide as Q"
        );

        // The refusal names the operator and the shape, so a caller can act on it
        // rather than read it as a crash.
        let msg = AttentionRefusal::HeadWidth {
            head_dim: at as u32 + 1,
            v_head_dim: at as u32 + 1,
        }
        .to_string();
        assert!(msg.contains("Attention"), "the operator: {msg}");
        assert!(msg.contains(&(at + 1).to_string()), "the width: {msg}");
    }

    /// Both dispatch caps the multiplies walk are shape terms, at the tree's own
    /// bound rather than an adapter query: their tiles come out as the dispatch's
    /// z and their `m = seq_q` rows as its y, so a shape at either bound is served
    /// here and one tile past it is not. The y case is built by hand because its
    /// key sequence is tiny and its scratch field is the decision's own, not a
    /// graph's — setting it small leaves the other conjuncts standing.
    #[test]
    fn the_dispatch_caps_bound_both_extents_the_multiplies_walk() {
        let at_cap = AttentionGeometry {
            tiles: MAX_WORKGROUPS_PER_DIM as u64,
            ..geom(32, 1024)
        };
        assert!(decompose_attention(&at_cap));
        let over = AttentionGeometry {
            tiles: MAX_WORKGROUPS_PER_DIM as u64 + 1,
            ..geom(32, 1024)
        };
        assert!(!decompose_attention(&over));
        let rows = MAX_WORKGROUPS_PER_DIM as u64 * MATMUL_TILE_M as u64;
        let at_rows = AttentionGeometry {
            seq_q: rows as u32,
            scratch_floats: 1,
            ..geom(32, 1024)
        };
        assert!(decompose_attention(&at_rows));
        let over_rows = AttentionGeometry {
            seq_q: rows as u32 + MATMUL_TILE_M,
            ..at_rows
        };
        assert!(!decompose_attention(&over_rows));
        // No tiles at all: nothing for either multiply to dispatch, whatever the
        // key sequence is (the empty batch the fused path still runs).
        let no_tiles = AttentionGeometry {
            tiles: 0,
            ..geom(32, 1024)
        };
        assert!(!decompose_attention(&no_tiles));
    }

    /// The degenerate key sequence: every region would be empty and the budget
    /// conjunct would pass vacuously, so the route would take a shape whose
    /// masked scale, softmax and P·V all write nothing — where the fused kernel
    /// it replaces normalises `0/0` into the NaN the operation is. One key is
    /// enough to serve it, so it is the emptiness and not the length that refuses.
    #[test]
    fn an_empty_key_sequence_keeps_the_fused_kernel() {
        let empty = AttentionGeometry {
            seq_k: 0,
            scratch_floats: 0,
            ..geom(32, 1024)
        };
        assert!(!decompose_attention(&empty));
        assert!(decompose_attention(&AttentionGeometry {
            seq_k: 1,
            scratch_floats: 1,
            ..empty
        }));
    }

    /// The budget's own boundary, on the field the decision reads — the cap
    /// itself is admitted and one float past it is refused — and a shape the
    /// budget alone refuses: 4096 × 4096 keys per head is 64 MiB of scores
    /// before Kᵀ, over the reserve every graph is guaranteed, while the other
    /// three conjuncts hold.
    #[test]
    fn the_scratch_budget_admits_its_own_figure() {
        let at = AttentionGeometry {
            seq_q: 64,
            seq_k: 128,
            head_dim: 64,
            v_head_dim: 64,
            tiles: 8,
            scratch_floats: ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES / 4,
        };
        assert!(decompose_attention(&at));
        let over = AttentionGeometry {
            scratch_floats: at.scratch_floats + 1,
            ..at
        };
        assert!(!decompose_attention(&over));
        let huge = geom(4096, 4096);
        assert!(huge.seq_q >= ATTENTION_DECOMPOSE_MIN_Q);
        assert!(huge.head_dim <= ATTENTION_DECOMPOSE_MAX_HEAD_DIM);
        assert!(huge.scratch_floats * 4 > ATTENTION_DECOMPOSE_MAX_SCRATCH_BYTES);
        assert!(!decompose_attention(&huge));
    }

    /// The regions a layout forces: Kᵀ and the scores are always emitted, each
    /// operand's own region dropped exactly when its tiles are dense, every
    /// region a multiple of the alignment and never smaller than the store it
    /// holds — so size order is the bump order, a region's offset the sum before
    /// it, and the bump cannot run short.
    #[test]
    fn regions_are_aligned_and_dropped_by_dense_operands() {
        let all_compacted = scratch_regions(2, 4, 64, 128, 64, 64, false, false, false);
        let all_dense = scratch_regions(2, 4, 64, 128, 64, 64, true, true, true);
        assert_eq!(all_compacted.kt, 2 * 4 * 64 * 128);
        assert_eq!(all_compacted.scores, 2 * 4 * 64 * 128);
        assert_eq!(all_compacted.q, 2 * 4 * 64 * 64);
        assert_eq!(all_compacted.v, 2 * 4 * 128 * 64);
        assert_eq!(all_compacted.out, 2 * 4 * 64 * 64);
        assert_eq!(all_dense.total(), all_compacted.kt + all_compacted.scores);
        assert!(all_compacted.total() > all_dense.total());
        // A tile that is dense is read where it lies; one that is not is
        // compacted. The two predicates are the caller's, but the strides that
        // decide them are compared here at the boundary they describe.
        assert!(operand_tile_is_dense(64, 64 * 64, 64, 64));
        assert!(!operand_tile_is_dense(64 * 8, 64, 64, 64));
        for (seq_q, seq_k) in [(32u32, 33u32), (64, 1024), (100, 128)] {
            let r = scratch_regions(1, 8, seq_q, seq_k, 64, 64, false, true, false);
            assert_eq!(r.kt % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.scores % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.q % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.out % SCRATCH_ALIGN_FLOATS, 0);
            assert_eq!(r.v, 0, "a dense V needs no region");
            assert!(r.kt >= 8 * 64 * seq_k as u64);
            assert!(r.scores >= 8 * seq_q as u64 * seq_k as u64);
        }
    }

    /// The derivation `plan_attention` shares with the lowering: a dense rank-4
    /// layout, a strided rank-3 one, a grouped-KV step and an asymmetric
    /// `v_head_dim` one, each read off the plan rather than restated — then the
    /// two shapes it refuses, with the messages the arm gives them, and the
    /// shapes it declines without refusing.
    #[test]
    fn the_plan_derives_the_geometry_the_route_reads() {
        // Rank-4 [B, H, S, D]: every per-(b, h) tile is a dense [seq, width]
        // block, so Kᵀ and the scores are the route's only regions.
        let dense = plan_of(
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            &[2, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("a rank-4 plan");
        assert_eq!(
            (dense.batch, dense.heads, dense.seq_q, dense.seq_k),
            (2, 4, 64, 64)
        );
        assert_eq!((dense.head_dim, dense.v_head_dim, dense.nkv), (64, 64, 4));
        assert_eq!((dense.nkv, dense.mask_kind_id, dense.window), (4, 0, 0));
        assert_eq!(
            (dense.q_b, dense.q_h, dense.q_s),
            (4 * 64 * 64, 64 * 64, 64)
        );
        assert_eq!(
            (dense.o_b, dense.o_h, dense.o_s),
            (4 * 64 * 64, 64 * 64, 64)
        );
        assert_eq!(
            (dense.m_b, dense.m_h, dense.m_q, dense.m_k),
            (4 * 64 * 64, 64 * 64, 64, 1)
        );
        assert!(dense.q_dense && dense.v_dense && dense.out_dense);
        assert_eq!(
            (dense.regions.q, dense.regions.v, dense.regions.out),
            (0, 0, 0)
        );
        assert_eq!(
            dense.regions.total(),
            dense.regions.kt + dense.regions.scores
        );
        assert!(dense.regions.kt > 0 && dense.regions.scores > 0);
        assert_eq!(dense.packed_parent, None);
        assert!(dense.decompose);
        // V's tiles are dense, so the default layout reads V where it lies — but
        // V′'s region is booked anyway, for the output slot the planner may have
        // placed on V (see the arm's output-overlap note).
        assert_eq!(dense.v_spare, 2 * 4 * 64 * 64);
        assert_eq!(
            dense.scratch_bytes(),
            ((dense.regions.total() + dense.v_spare) * 4 + SCRATCH_ALIGN_SLACK_BYTES) as usize
        );

        // Rank-3 [B, S, H·D], told from [B·H, S, D] by its last dim: its tiles
        // are strided, so every operand region is emitted and the floor refuses
        // the step.
        let strided = plan_of(
            &[2, 16, 256],
            &[2, 16, 256],
            &[2, 16, 256],
            &[2, 16, 256],
            4,
            64,
            None,
        )
        .expect("a rank-3 plan");
        assert_eq!(
            (strided.batch, strided.heads, strided.seq_q, strided.seq_k),
            (2, 4, 16, 16)
        );
        assert_eq!((strided.q_b, strided.q_h, strided.q_s), (16 * 256, 64, 256));
        assert!(!strided.q_dense && !strided.v_dense && !strided.out_dense);
        assert_eq!(strided.regions.q, 2 * 4 * 16 * 64);
        assert!(!strided.decompose, "16 query rows are under the floor");

        // Grouped KV (nkv < heads): Q's and the output's tiles stay dense, V's
        // does not — the route reads one (b, h) V tile per query head.
        let gqa = plan_of(
            &[1, 8, 32, 64],
            &[1, 2, 8, 64],
            &[1, 2, 8, 64],
            &[1, 8, 32, 64],
            8,
            64,
            None,
        )
        .expect("a grouped-KV plan");
        assert_eq!(
            (gqa.batch, gqa.heads, gqa.seq_q, gqa.seq_k, gqa.nkv),
            (1, 8, 32, 8, 2)
        );
        assert!(gqa.q_dense && gqa.out_dense);
        assert!(!gqa.v_dense);
        assert_eq!(gqa.regions.v, 8 * 8 * 64);
        assert_eq!((gqa.k_b, gqa.k_h, gqa.k_s), (2 * 64 * 8, 8 * 64, 64));

        // Rank-4 [B, S, H, D] (the CPU / EEG-DINO convention), told from BHSD by
        // its second dim: every tile is strided — seq stride `H·D`, not the head
        // width — so all three operand regions are emitted on a step the floor
        // admits.
        let bshd = plan_of(
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            &[1, 32, 8, 64],
            8,
            64,
            None,
        )
        .expect("a rank-4 BSHD plan");
        assert_eq!(
            (bshd.batch, bshd.heads, bshd.seq_q, bshd.seq_k),
            (1, 8, 32, 32)
        );
        assert_eq!((bshd.q_h, bshd.q_s, bshd.o_s), (64, 8 * 64, 8 * 64));
        assert!(!bshd.q_dense && !bshd.v_dense && !bshd.out_dense);
        assert!(bshd.decompose);

        // Packed QKV: Q/K/V are axis-2 narrows of one [B, S, 3, H, D] parent,
        // read through the parent's own strides so the KV-head count is Q's.
        let mut packed = Graph::new("packed_qkv");
        let parent = packed.input("fmb", Shape::new(&[1, 4, 3, 2, 8], DType::F32));
        let shape_5 = Shape::new(&[1, 4, 1, 2, 8], DType::F32);
        let heads_in: Vec<rlx_ir::NodeId> = (0..3)
            .map(|start| {
                let narrow = packed.add_node(
                    Op::Narrow {
                        axis: 2,
                        start,
                        len: 1,
                    },
                    vec![parent],
                    shape_5.clone(),
                );
                packed.add_node(
                    Op::Reshape {
                        new_shape: vec![1, 4, 2, 8],
                    },
                    vec![narrow],
                    Shape::new(&[1, 4, 2, 8], DType::F32),
                )
            })
            .collect();
        let attn = packed.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 8,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            heads_in,
            Shape::new(&[1, 4, 2, 8], DType::F32),
        );
        let plan = plan_attention(&packed, packed.node(attn)).expect("a packed-QKV plan");
        // One head is `1 · H · D` = 16 elements, so the packed sequence stride is
        // 3 · 16 and no operand of it is a dense tile.
        assert_eq!(plan.packed_parent, Some((parent, 16)));
        assert_eq!((plan.heads, plan.nkv, plan.seq_q), (2, 2, 4));
        assert_eq!((plan.q_s, plan.k_s, plan.v_s), (48, 48, 48));
        assert!(!plan.q_dense && !plan.v_dense && !plan.out_dense);
        assert!(!plan.decompose, "4 query rows are under the floor");

        // Asymmetric SDPA: V's rows and the output are `v_head_dim` wide while
        // the scores still contract over `head_dim`.
        let mla = plan_of(
            &[1, 2, 32, 64],
            &[1, 2, 16, 64],
            &[1, 2, 16, 96],
            &[1, 2, 32, 96],
            2,
            64,
            Some(96),
        )
        .expect("an asymmetric plan");
        assert_eq!((mla.head_dim, mla.v_head_dim), (64, 96));
        assert_eq!((mla.seq_q, mla.seq_k), (32, 16));
        assert_eq!((mla.k_s, mla.v_s, mla.o_s), (64, 96, 96));
        assert_eq!((mla.v_b, mla.v_h, mla.o_h), (2 * 96 * 16, 16 * 96, 32 * 96));
        assert!(mla.decompose);

        // A zero value-head width is the kernel's own "V is as wide as Q": the
        // plan resolves it to `head_dim` rather than sizing V′, the output and
        // the P·V multiply at width 0 — which would emit a multiply with no
        // workgroups and leave the output slot unwritten.
        let zero_v = plan_of(
            &[1, 2, 32, 64],
            &[1, 2, 16, 64],
            &[1, 2, 16, 64],
            &[1, 2, 32, 64],
            2,
            64,
            Some(0),
        )
        .expect("a zero value-width plan");
        assert_eq!((zero_v.head_dim, zero_v.v_head_dim), (64, 64));
        assert_eq!((zero_v.v_s, zero_v.o_s), (64, 64));
        assert_eq!(zero_v.regions.out, 0);
        assert!(zero_v.decompose);

        // The two shapes no plan can be derived for, with the message the arm
        // panics with today.
        let rank = plan_of(
            &[1, 2, 32, 64, 1],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            2,
            64,
            None,
        )
        .unwrap_err();
        assert_eq!(rank, AttentionRefusal::Rank { rank: 5 });
        assert_eq!(
            rank.to_string(),
            "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V inputs supported (got rank 5)"
        );
        let leading = plan_of(
            &[7, 32, 3],
            &[7, 32, 3],
            &[7, 32, 3],
            &[7, 32, 3],
            2,
            64,
            None,
        )
        .unwrap_err();
        assert_eq!(
            leading.to_string(),
            "rlx-wgpu Attention: rank-3 leading dim 7 not divisible by num_heads 2 \
             (and last dim 3 ≠ H·D = 128)"
        );
        let mut g = Graph::new("maskless");
        let (q, k, v) = (
            g.input("q", Shape::new(&[1, 2, 32, 64], DType::F32)),
            g.input("k", Shape::new(&[1, 2, 32, 64], DType::F32)),
            g.input("v", Shape::new(&[1, 2, 32, 64], DType::F32)),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 64,
                v_head_dim: None,
                mask_kind: MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 2, 32, 64], DType::F32),
        );
        assert_eq!(
            plan_attention(&g, g.node(attn)).unwrap_err().to_string(),
            "rlx-wgpu Attention: mask kind 2 needs a mask input (node has 3 inputs)"
        );

        // A node that is not attention at all is a refusal too — the reserve
        // walks every node of a graph, and reads only the plans that name it a
        // route.
        let mut plain = Graph::new("plain");
        let x = plain.input("x", Shape::new(&[1, 2, 32, 64], DType::F32));
        assert_eq!(
            plan_attention(&plain, plain.node(x)).unwrap_err(),
            AttentionRefusal::NotAttention
        );
        // …and a shape under the floor is still a plan, with the route off it.
        let under = plan_of(
            &[1, 2, 4, 64],
            &[1, 2, 4, 64],
            &[1, 2, 4, 64],
            &[1, 2, 4, 64],
            2,
            64,
            None,
        )
        .expect("a plan under the floor");
        assert!(!under.decompose);
    }

    /// Grouped KV that does not divide the query heads is a shape the route
    /// declines even above its floor: its Kᵀ and V′ transpose K and V through a
    /// group axis, and `heads / nkv` is the fused kernel's own kv-head divisor.
    /// Both sides of the rule, on the layout the plan derives `nkv` from (K's
    /// element count over B·S_k·D).
    #[test]
    fn the_kv_group_must_divide_the_head_count() {
        let divides = plan_of(
            &[1, 4, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 4, 32, 64],
            4,
            64,
            None,
        )
        .expect("a grouped-KV plan");
        assert_eq!((divides.heads, divides.nkv), (4, 2));
        assert!(divides.decompose, "4 heads over 2 KV heads is served");

        let leaves_one_out = plan_of(
            &[1, 3, 32, 64],
            &[1, 2, 32, 64],
            &[1, 2, 32, 64],
            &[1, 3, 32, 64],
            3,
            64,
            None,
        )
        .expect("a plan with an indivisible group");
        assert_eq!((leaves_one_out.heads, leaves_one_out.nkv), (3, 2));
        assert!(
            leaves_one_out.seq_q >= ATTENTION_DECOMPOSE_MIN_Q,
            "the floor must not be what refuses this shape"
        );
        assert!(
            !leaves_one_out.decompose,
            "3 heads over 2 KV heads is not a layout the route's group axis can walk"
        );
    }

    /// The degenerate empty batch the fused path still runs, and the reason the
    /// route must not take it: with no `(batch, head)` tiles there is no multiply
    /// to dispatch, and `tiles · seq_k` — the product the `n`-extent argument
    /// rests on — is zero however long the key sequence is, so a shape no dispatch
    /// could run would pass every other conjunct. The fused path's grid is folded
    /// across the dispatch's own dimensions and takes it, as it did before.
    #[test]
    fn an_empty_batch_keeps_the_fused_kernel() {
        let empty = plan_of(
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            &[0, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("an empty-batch plan");
        assert_eq!((empty.batch, empty.seq_q), (0, 64));
        assert!(!empty.decompose, "no tiles means no multiply to dispatch");
        // One tile and the same lengths are served, so it is the empty batch and
        // not the geometry that refuses.
        let served = plan_of(
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            &[1, 4, 64, 64],
            4,
            64,
            None,
        )
        .expect("a one-batch plan");
        assert!(served.decompose);
    }
}
