// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Buffer arena for the wgpu backend. Mirrors the rlx-metal arena
//! shape: pre-plan one big storage buffer at compile time, sub-allocate
//! per-node offsets at known positions, treat I/O as `write_buffer` /
//! `read_buffer` against those offsets.
//!
//! wgpu's storage buffers are fine for both reads and writes from
//! compute shaders; there's no shared-memory requirement at the API
//! level (unlike Metal where `StorageModeShared` matters). On Apple
//! Silicon wgpu's Metal backend gives us unified memory automatically.

use crate::gguf_gpu::dequant_route_defers_host_read;
use rlx_ir::{Graph, NodeId};
use rlx_opt::memory::MemoryPlan;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Process-wide reuse of weight buffers across prefill/decode compiles.
/// Prefill + decode each used to allocate a full ~2 GiB weight copy and OOM
/// VirtIO/DX12 (~8 GiB hostmem). Same **named** weight layout → share the GPU
/// allocation (params are re-uploaded on compile).
///
/// Disable with `RLX_WGPU_SHARE_WEIGHTS=0`. Fingerprint includes sorted
/// `(param_name, nbytes, offset)` so unrelated packages with coincidentally
/// similar byte cursors do not share buffers.
struct SharedWeightBuffers {
    /// FNV-ish fingerprint of layout + param names/sizes.
    fingerprint: u64,
    first: wgpu::Buffer,
    extras: Vec<wgpu::Buffer>,
    shard_size: usize,
    logical_bytes: usize,
}

// The browser WebGPU backend's `wgpu::Buffer` is `!Send` (holds an `Rc`), but
// wasm is single-threaded so the shared static is never actually accessed from
// another thread — assert `Send` there so it fits the `static Mutex`. On native
// `wgpu::Buffer` is already `Send` (no impl needed / would conflict).
#[cfg(target_arch = "wasm32")]
unsafe impl Send for SharedWeightBuffers {}

static SHARED_WEIGHTS: Mutex<Option<SharedWeightBuffers>> = Mutex::new(None);

fn weight_layout_fingerprint(
    weight_cursor: usize,
    named_slots: &[(String, usize, usize)], // (name, nbytes, offset)
) -> u64 {
    let mut fp = weight_cursor as u64;
    for (name, nbytes, off) in named_slots {
        fp = fp.wrapping_mul(0x100000001b3).wrapping_add(*nbytes as u64);
        fp = fp.wrapping_mul(0x100000001b3).wrapping_add(*off as u64);
        for b in name.as_bytes() {
            fp = fp.wrapping_mul(0x100000001b3).wrapping_add(u64::from(*b));
        }
    }
    fp
}

fn share_weights_enabled() -> bool {
    // Default on (OOM mitigation). Set RLX_WGPU_SHARE_WEIGHTS=0 to force
    // fresh allocations per compile. Prefer env registry flag when set to
    // truthy; treat explicit "0"/"false"/"off" as disable.
    match rlx_ir::env::var("RLX_WGPU_SHARE_WEIGHTS") {
        Some(v) => {
            let t = v.trim();
            !(t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("off"))
        }
        None => true,
    }
}

/// Byte end (exclusive) of an f16 shadow write for a slot starting at
/// `f32_byte_offset` with `f32_byte_len` bytes of f32 payload.
/// wgpu requires `queue.write_buffer` sizes to be 4-byte aligned; odd
/// f16 element counts are zero-padded by two bytes in `write_f32`.
fn f16_shadow_write_end(f32_byte_offset: usize, f32_byte_len: usize) -> usize {
    let f16_off = f32_byte_offset / 2;
    let f16_bytes = (f32_byte_len / 4) * 2;
    let padded = (f16_bytes + 3) & !3;
    f16_off + padded
}

/// Size the f16 side buffer so every planned slot's padded upload fits.
fn f16_shadow_arena_size(plan: &MemoryPlan) -> usize {
    plan.assignments
        .values()
        .map(|a| f16_shadow_write_end(a.offset, a.size))
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Bytes reserved at the end of every shard for cross-stripe staging copies.
/// Tensors are never placed here; ops that bind a shard can stage outliers into
/// this zone without clobbering live activations.
///
/// Must fit the largest tensor still staged through the arena.
/// Qwen3-0.6B tied lm_head/embedding is ~594 MiB F32; keep headroom above that
/// so NVIDIA's ~2 GiB storage-bind path can stage weight-buffer params into the
/// act window. Override with `RLX_WGPU_SHARD_STAGE_MIB` (KittenTTS defaults to
/// 64 on discrete wgpu so long-wave plans do not snap into multi-×4 GiB OOM).
pub const SHARD_STAGE_RESERVE: usize = 768 * 1024 * 1024;

/// Upper bound on how many stripes one arena may be split into.
///
/// Striping exists to get past `max_buffer_size`, so an adapter on its real
/// limits produces a handful of stripes — `size / max_buffer_size`. The lever
/// that makes the striping path testable, `RLX_WGPU_SHARD_CAP_MIB`, can instead
/// ask for thousands: a 4 GiB arena at a 1 MiB cap is 4096 buffers. That geometry
/// does not make the graph run — it multiplies bind windows, staging copies and
/// shader compilations until the host cannot keep up. Measured on one host: the
/// watchdog hung five times in a day while this backend's suite was run under a
/// few-MiB cap plus the sharding levers, every crash report showing that suite's
/// test binary alive with several Metal shader compilers.
///
/// So the count is bounded at the decision point, before the buffers it bounds
/// are created — the activation arena's own, and the parked weight buffer's (the
/// latter after the activation arena exists, but before any weight buffer does).
/// Tens of stripes stay usable (a 4 MiB cap on a 76 MiB arena is 19); a cap that
/// would explode the geometry is refused loudly and cheaply instead.
pub const MAX_SHARDS: usize = 64;

/// Effective stage reserve (see [`SHARD_STAGE_RESERVE`]).
pub fn shard_stage_reserve() -> usize {
    if let Some(raw) = rlx_ir::env::var("RLX_WGPU_SHARD_STAGE_MIB") {
        if let Ok(mib) = raw.parse::<usize>() {
            return (mib.max(1) * 1024 * 1024).min(SHARD_STAGE_RESERVE);
        }
    }
    SHARD_STAGE_RESERVE
}

/// Device shard capacity, honouring `RLX_WGPU_SHARD_CAP_MIB`.
///
/// The real cap is the adapter's `max_buffer_size` (4 GiB on discrete Vulkan,
/// smaller on Apple). Testing the striping path therefore meant building a plan
/// larger than that — `sharded_arena.rs` asked for `shard_cap * 2`, which on a
/// 4 GiB-limit adapter snapped to three 4 GiB buffers and died with
/// `wgpu error: Out of Memory`. The test passed on Apple only because its
/// smaller `max_buffer_size` happened to fit in unified memory.
///
/// A path that can only be exercised by allocating 12 GiB is a path that does
/// not get exercised — and this one carries a standing warning that sharded
/// arenas have produced *silently wrong results*. Allowing the cap to be lowered
/// makes striping testable on any device at kilobyte scale.
///
/// Clamped to the device limit: this can only make shards SMALLER, so it can
/// never be used to exceed what the adapter actually supports.
///
/// How far it may be lowered is bounded by [`MAX_SHARDS`]: a cap that would split
/// an arena into more stripes than that is refused at plan time — before the
/// buffers it bounds are created — rather than realized.
pub fn effective_shard_cap(device: &wgpu::Device) -> usize {
    let device_cap = ((device.limits().max_buffer_size as usize) / 256) * 256;
    if let Some(raw) = rlx_ir::env::var("RLX_WGPU_SHARD_CAP_MIB")
        && let Ok(mib) = raw.parse::<usize>()
    {
        let forced = (mib.max(1) * 1024 * 1024 / 256) * 256;
        return forced.min(device_cap).max(256);
    }
    device_cap
}

/// Whether the informational sharding diagnostics should print.
///
/// `RLX_WGPU_QUIET_SHARD` suppresses only these logs — never the warning that an
/// opt-in striping decision produced a knowingly wrong configuration
/// ([`Arena::refuse_or_warn_about_sharding`]), which is emitted every time, and
/// never [`bound_stripes`], which is a refusal.
///
/// Its own description — the registry summary in `rlx_ir`'s
/// `env_registry_data.inc.rs` and the `docs/rlx-env-vars.md` generated from it —
/// still reads as though it could silence that warning. Those two are gated together
/// by `check-rlx-env-vars`, so the correction lives here rather than there.
fn shard_log() -> bool {
    !rlx_ir::env::flag("RLX_WGPU_QUIET_SHARD")
        && (rlx_ir::env::flag("RLX_WGPU_DEBUG") || rlx_ir::env::flag("RLX_WGPU_SHARD_LOG"))
}

/// Buffers a `size`-byte region is split into at `cap` — the arithmetic every
/// allocation branch, and the parked weight buffer, shares.
///
/// Pure by design, so the bound ([`MAX_SHARDS`], applied by the callers where the
/// buffers it bounds do not exist yet) is decidable without a device and the
/// geometry table is testable where no GPU is involved.
fn shard_count(size: usize, cap: usize) -> usize {
    size.div_ceil(cap.max(1))
}

/// Bound the stripe count of a `size`-byte region about to be split at `cap`, and
/// return that count. Past [`MAX_SHARDS`] it refuses loudly instead, naming the
/// region, the count, the cap and the lever that reached it: the geometry that took
/// a host down here does not make the graph fit, it multiplies buffers, bind
/// windows, staging copies and kernel compilations until the machine cannot keep up.
///
/// Both callers apply it at their decision point, before creating the buffers it
/// bounds — the activation arena's own, and the parked weight buffer's. It is never
/// applied to a virtual bind-stripe geometry, which allocates one buffer.
///
/// Deliberately not gated on `RLX_WGPU_ALLOW_SHARD`: that opt-in downgrades the
/// *wrongness* warning to continue-and-warn, and it must not also downgrade a
/// bound into "compile until the machine gives up".
fn bound_stripes(what: &str, size: usize, cap: usize) -> usize {
    let shards = shard_count(size, cap);
    if shards <= MAX_SHARDS {
        return shards;
    }
    let size_mib = size as f64 / (1024.0 * 1024.0);
    let cap_mib = cap as f64 / (1024.0 * 1024.0);
    panic!(
        "rlx-wgpu: REFUSED (pathological stripe geometry): {what} is {size_mib:.1} MiB and would \
         be striped into {shards} buffers of {cap_mib:.2} MiB — over the {MAX_SHARDS}-stripe \
         bound.\n\
         \x20 A cap this small does not make the graph fit; it multiplies buffers, bind windows, \
         staging copies and kernel compilations until the machine cannot keep up.\n\
         \x20 The cap is RLX_WGPU_SHARD_CAP_MIB where that lever set one, and the adapter's own \
         max_buffer_size otherwise: raise or unset the lever if it is the cap, or keep the largest \
         intermediate under it."
    );
}

/// The allocated-buffer geometry of a logical arena at these caps, with the
/// stripe bound ([`MAX_SHARDS`]) already applied.
///
/// Pure — no device — so the decision is testable where no GPU is involved, and
/// it is the *one* place the activation arena's bound is asked. Asking it of
/// [`ArenaStripes::Physical`] only is the point: that is the only geometry that
/// allocates more than one buffer. A virtual bind-stripe geometry is one buffer
/// however many bind-sized windows its size spans, so counting the bind
/// dimension there would refuse a working arena on an adapter whose
/// `max_buffer_size` exceeds 64× its storage-bind limit.
#[derive(Debug, PartialEq, Eq)]
enum ArenaStripes {
    /// One buffer, addressed whole (`shard_size == 0`) or in bind-sized windows.
    One { shard_size: usize },
    /// `shards` buffers of `shard_cap` bytes each.
    Physical { shards: usize },
}

fn arena_stripes(what: &str, size: usize, shard_cap: usize, bind_shard_cap: usize) -> ArenaStripes {
    if size > shard_cap {
        ArenaStripes::Physical {
            shards: bound_stripes(what, size, shard_cap),
        }
    } else if size > bind_shard_cap {
        ArenaStripes::One {
            shard_size: bind_shard_cap,
        }
    } else {
        ArenaStripes::One { shard_size: 0 }
    }
}

/// Bytes one parameter occupies in the parked weight buffer: packed U8/I8 keep
/// their own byte per element, every other encoding is widened to one f32 lane
/// there. One function, because `from_plan_split` places the parameters and the
/// stageability rule asks about the slot they got.
pub(crate) fn weight_param_bytes(dt: rlx_ir::DType, ne: usize) -> usize {
    if matches!(dt, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
        ne * dt.size_bytes()
    } else {
        ne * 4
    }
}

/// The slot the plan assigned each node — the bytes a staged read of that node's
/// operand has to fit.
///
/// Its own function because [`Arena::lens`] cannot answer it: `lens` starts as these
/// numbers and is then overwritten with each node's *actual* byte length
/// (`Arena::set_actual_len`) once the arena exists — `elems * 4` for the widened
/// encodings, `elems * size_bytes()` for complex tensors.
fn plan_node_slots(plan: &MemoryPlan) -> HashMap<NodeId, usize> {
    plan.assignments
        .iter()
        .map(|(id, a)| (*id, a.size))
        .collect()
}

/// The node whose bytes `node`'s arena offset falls in — the planner's own view chain
/// (`rlx_compile::memory`'s `resolve_view_root`), so the record and the plan cannot
/// disagree about which storage a view lives in.
///
/// [`rlx_compile::memory::is_pure_view`] is that chain without `KvAppend`, which it
/// excludes because the backend must still *write* the appended row. The append still
/// aliases `cache` (input 0) at offset 0, so a view above one reads the cache's bytes and
/// has to be bounded by them.
fn storage_root<'a>(graph: &'a Graph, node: &'a rlx_ir::Node) -> &'a rlx_ir::Node {
    let mut root = node;
    while rlx_compile::memory::is_pure_view(graph, root)
        || matches!(root.op, rlx_ir::Op::KvAppend { .. })
    {
        root = graph.node(root.inputs[0]);
    }
    root
}

/// Whether `id`'s bytes are a parameter's: an `Op::Param`, or a node living in one's
/// storage ([`storage_root`]). Asked where the arena's own answer is not enough — see
/// [`first_unrecorded_parameter`].
pub(crate) fn is_parameter_storage(graph: &Graph, id: NodeId) -> bool {
    matches!(
        storage_root(graph, graph.node(id)).op,
        rlx_ir::Op::Param { .. }
    )
}

/// The first operand that is a parameter's storage — by the graph — for which `record`
/// has nothing to say, or `None` when it answers for every one of them.
///
/// The plan-time check asks this of the arena's record, and refuses what it finds: an
/// arena that was never recorded answers `None` for *every* offset, and there an operand
/// would read as an activation and be measured against nothing. Refusing is the safe
/// default for a decision the arena cannot answer, and it is the one this function makes
/// where the shared walk ([`first_unstageable_operand`]) cannot: it has the graph, and the
/// walk has only the offsets.
///
/// Pure — the graph the check already holds and the arena's map — so it is pinned without
/// a device: `an_unrecorded_parameter_is_not_an_activation`.
pub(crate) fn first_unrecorded_parameter(
    graph: &Graph,
    record: &HashMap<usize, ParamStorage>,
    operands: &[(NodeId, usize)],
) -> Option<(NodeId, usize)> {
    operands
        .iter()
        .copied()
        .find(|&(id, off)| !record.contains_key(&off) && is_parameter_storage(graph, id))
}

/// What a *parameter's* storage holds from a byte offset on: the bytes a staged read of
/// an operand there has to fit, and the encoding that parameter was declared with, where
/// the record can attribute one to that offset.
///
/// The encoding is here because the structural host steps carry element counts rather
/// than shapes (`scan_host::assert_read_fits`), so the refusal they raise cannot read the
/// operand's declared encoding off the operand itself — it names the parameter's real one
/// from this record, which is the same declaration the bytes were sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParamStorage {
    /// Bytes available at the offset, to the end of the parameter's storage. Where two
    /// parameters' storages reach one offset, this is the tighter of the two — the safe
    /// direction for a read whose owner the key cannot say.
    pub bytes: usize,
    /// The declared `DType` of the parameter whose storage the entry bounds, or `None`
    /// where the offset is claimed by two parameters' storages at once. The record is
    /// keyed by offset alone, so it cannot say which of them a read there belongs to, and
    /// an entry that cannot attribute an encoding names none rather than the co-located
    /// parameter's.
    pub dtype: Option<rlx_ir::DType>,
}

/// `(byte offset, what a parameter's storage holds there)` for every offset where such
/// storage starts, by the rule [`Arena::record_param_storage`] documents.
///
/// Pure — a graph and the three maps the arena holds — so the rule can be pinned without
/// a device: `a_shrinking_view_does_not_lower_the_parameters_bound` and
/// `an_offset_two_parameters_reach_names_no_encoding`.
fn param_storage_bounds(
    graph: &Graph,
    offsets: &HashMap<NodeId, usize>,
    weight_offsets: &HashMap<NodeId, usize>,
    plan_slots: &HashMap<NodeId, usize>,
) -> Vec<(usize, ParamStorage)> {
    // Where a node's bytes are: the activation arena, or — tagged — the weight buffer,
    // which is the same choice `Arena::offset` makes.
    let storage = |id: NodeId| -> Option<usize> {
        weight_offsets
            .get(&id)
            .map(|w| w | WEIGHT_BUF_TAG)
            .or_else(|| offsets.get(&id).copied())
    };
    let mut bounds: HashMap<usize, ParamStorage> = HashMap::new();
    // The parameter that first claimed each offset: a second, different one at the same
    // key — a view placed past its own root's slot lands on another parameter's start —
    // means the entry can no longer say whose storage a read at that offset reads.
    let mut owners: HashMap<usize, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let root = storage_root(graph, node);
        if !matches!(root.op, rlx_ir::Op::Param { .. }) {
            continue;
        }
        // A node the plan did not place (an elided transpose, say) has no storage.
        let (Some(off), Some(root_off)) = (storage(node.id), storage(root.id)) else {
            continue;
        };
        // What the parameter's storage holds from `off` on: a parked one is sized by the
        // weight buffer's own rule ([`weight_param_bytes`], the function that placed it),
        // one in the activation arena by the plan's slot. No slot on record asks with 0,
        // the safe default: nothing fits a zero-byte slot, so such an operand is refused
        // rather than assumed wide.
        let slot = if is_weight_off(root_off) {
            let ne = root.shape.num_elements().unwrap_or(0).max(1);
            weight_param_bytes(root.shape.dtype(), ne)
        } else {
            plan_slots.get(&root.id).copied().unwrap_or(0)
        };
        let bound = ParamStorage {
            bytes: slot.saturating_sub(off.saturating_sub(root_off)),
            dtype: Some(root.shape.dtype()),
        };
        match bounds.get_mut(&off) {
            Some(b) => {
                if owners.get(&off).is_some_and(|&id| id != root.id) {
                    b.dtype = None;
                }
                if bound.bytes < b.bytes {
                    b.bytes = bound.bytes;
                    // Only while the key still has one claimant: once two parameters have
                    // reached it, the silence about the encoding is final.
                    if b.dtype.is_some() {
                        b.dtype = bound.dtype;
                    }
                }
            }
            None => {
                owners.insert(off, root.id);
                bounds.insert(off, bound);
            }
        }
    }
    bounds.into_iter().collect()
}

/// Why the generic host route cannot stage a parameter operand — one variant per way the
/// bytes it reads and the bytes the parameter was given can disagree, plus the one way
/// the question cannot be answered at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unstageable {
    /// `set_param_typed` has no arm for the encoding, so the arena never receives
    /// the parameter and every lane the route reads is unwritten.
    NoUploadArm,
    /// The upload's raw arm keeps the operand's own bytes — one or two per element —
    /// while the route reads four.
    StoredRaw,
    /// The upload widens the operand to one f32 lane per element, while the route
    /// sizes its read from a declared element wider than that lane.
    DeclaredWiderThanALane,
    /// The slot the plan gave the operand is narrower than the four bytes per
    /// element the route reads.
    SlotTooNarrow,
    /// The arena holds no record of where this parameter's storage starts, so there is
    /// nothing to measure the read against — it was built without
    /// [`Arena::record_param_storage`].
    NoRecord,
}

impl Unstageable {
    /// The clause the refusal prints for it.
    fn sentence(self) -> &'static str {
        match self {
            Self::NoUploadArm => {
                "the typed upload has no arm for this encoding, so the arena never receives it"
            }
            Self::StoredRaw => {
                "the upload stores it raw, at the encoding's own bytes per element, while the \
                 staging reads four bytes per element"
            }
            Self::DeclaredWiderThanALane => {
                "the upload stores one f32 lane per element, while the staging reads the declared \
                 wider element"
            }
            Self::SlotTooNarrow => {
                "the slot it was given is narrower than the four bytes per element the staging \
                 reads"
            }
            Self::NoRecord => {
                "the arena holds no record of where this parameter's storage starts, so its read \
                 cannot be bounded"
            }
        }
    }
}

/// Why the generic host route cannot stage a parameter operand of `dt` with `ne`
/// elements whose storage the arena's record says holds `slot_bytes` from its offset on,
/// or `None` when it can.
///
/// That route transfers `max(ceil(size_bytes / 4), ne)` f32 lanes per operand
/// (`rlx_gpu_host::run_host_op_packed_cached` sizes the read from the declared
/// shape), so a parameter is stageable exactly where the bytes it reads really are
/// one f32 lane per element — in what the typed upload stored, and in the storage it was
/// given. An operand with no elements is no read at all and is never refused.
///
/// `slot_bytes` is what the storage itself was given, and it is the number the two-byte
/// encodings turn on: the flat plan sizes a non-F32 `Param` at its native width
/// (`rlx_compile`'s `node_slot_bytes`), while `from_plan_split` — the plan the
/// geometries this rule is about use — gives every parameter it places in the
/// activation arena four bytes per element (the ones it parks are sized by the weight
/// buffer's own rule, [`weight_param_bytes`], and the encodings where those two
/// disagree — packed `U8`/`I8` — are decided by the dtype arms below, before the slot is
/// read). It is not `Arena::lens`, which `compile_static_inner` overwrites with the
/// node's actual byte length: the callers ask the arena's record instead
/// (`Arena::param_storage_at` — see [`plan_node_slots`] for the numbers it holds).
///
/// Which encodings the upload stores as one lane is `set_param_typed` in
/// `crates/core/rlx-runtime/src/backend/wgpu_backend.rs`: this crate sits below it
/// and cannot call it, so a change there has to be mirrored here.
/// `tests/ceiling_stageable.rs` runs a parked parameter through that arm end to end.
pub(crate) fn unstageable_reason(
    dt: rlx_ir::DType,
    ne: usize,
    slot_bytes: usize,
) -> Option<Unstageable> {
    use rlx_ir::DType::*;
    // The transfer, in bytes, as `rlx_gpu_host` sizes it. Zero elements is a
    // zero-byte read, which nothing can be read past.
    let read = 4 * (dt.size_bytes().saturating_mul(ne)).div_ceil(4).max(ne);
    if read == 0 {
        return None;
    }
    // Exhaustive on purpose: a new encoding has to be decided here rather than fall
    // through to a verdict nobody checked.
    match dt {
        // One f32 lane per element, and a declared element no wider than that lane.
        F32 | I32 | U32 | F16 | Bool => {}
        F64 | I64 => return Some(Unstageable::DeclaredWiderThanALane),
        U8 | I8 | BF16 => return Some(Unstageable::StoredRaw),
        I16 | C64 | C128 => return Some(Unstageable::NoUploadArm),
    }
    (slot_bytes < read).then_some(Unstageable::SlotTooNarrow)
}

/// The first operand among `operands` the generic host route cannot stage, with the
/// reason — `operands` yields `(payload, byte offset, dtype, elements)`, and both the
/// payload and the offset come back so the caller can name the offender in its own terms
/// (a node id at plan time; nothing at all at run time, where the offset is the name).
/// `None` when every read fits, and for every operand whose storage is not a parameter's:
/// an activation's slot holds one f32 lane per element by construction.
///
/// The one walk both checkers use, asking one bound: the arena's record of where a
/// parameter's storage is and how much of it is there ([`Arena::param_storage_at`], filled
/// by `Arena::record_param_storage`) — so the plan-time check
/// (`lower::assert_host_route_stageable`) and the check at the route's staging point
/// (`scan_host::assert_host_reads_fit`) cannot drift apart.
pub(crate) fn first_unstageable_operand<T>(
    arena: &Arena,
    operands: impl IntoIterator<Item = (T, usize, rlx_ir::DType, usize)>,
) -> Option<(T, usize, rlx_ir::DType, Unstageable)> {
    operands.into_iter().find_map(|(payload, off, dt, ne)| {
        let slot = arena.param_storage_at(off)?.bytes;
        Some((payload, off, dt, unstageable_reason(dt, ne, slot)?))
    })
}

/// Where an operand's storage lives, for a refusal message.
pub(crate) fn operand_location(off: usize) -> &'static str {
    if is_weight_off(off) {
        "parked in the weight buffer"
    } else {
        "in the activation arena"
    }
}

/// The one refusal for a parameter operand the host route must not stage —
/// [`unstageable_reason`], from the plan-time check and the runtime backstop alike.
/// `operand` names it as that caller can, and `dtype` is its declared encoding where the
/// caller knows one. `None` says the arena's record cannot attribute an encoding to that
/// offset (two parameters' storages reach it), and the message then leaves the encoding
/// out rather than stating the co-located parameter's.
pub(crate) fn refuse_unstageable_operand(
    op: rlx_ir::OpKind,
    operand: &str,
    dtype: Option<rlx_ir::DType>,
    reason: Unstageable,
) -> ! {
    let Some(dtype) = dtype else {
        panic!(
            "rlx-wgpu: REFUSED (operand not stageable as f32): {op:?} reads {operand} — {}. Two \
             parameters' storages share this offset, so which encoding bounds the read is not \
             attributable; move the graph to a backend that can address them apart.",
            reason.sentence(),
        );
    };
    let width = dtype.size_bytes();
    panic!(
        "rlx-wgpu: REFUSED (operand not stageable as f32): {op:?} reads {operand} ({dtype:?}, \
         {width} {} per element) — {}. Avoid this encoding for the parameter, or move the graph \
         to a backend that can address it.",
        if width == 1 { "byte" } else { "bytes" },
        reason.sentence(),
    );
}

/// Re-place unique buffer slots so no slot crosses a shard boundary.
/// Preserves shared offsets (true reuse); may insert padding gaps and grow
/// `arena_size`. Tail bytes past the last slot (scratch) are preserved.
///
/// Each stripe keeps `stage_reserve` free at its end for staging.
fn snap_plan_to_shards_ex(plan: &mut MemoryPlan, shard_cap: usize, stage_reserve: usize) {
    let usable = shard_cap.saturating_sub(stage_reserve).max(256);
    let mut slot_size: HashMap<usize, usize> = HashMap::new();
    let mut max_end = 0usize;
    for a in plan.assignments.values() {
        let sz = a.size.max(1);
        slot_size
            .entry(a.offset)
            .and_modify(|s| *s = (*s).max(sz))
            .or_insert(sz);
        max_end = max_end.max(a.offset.saturating_add(sz));
    }
    let tail_extra = plan.arena_size.saturating_sub(max_end);

    // Place slots in the order they are first *used*, not in the order the
    // planner happened to lay them out.
    //
    // This is what keeps an op's operands in one stripe. A kernel binds a
    // single stripe, so when a convolution's input and output land either side
    // of a boundary the input reads as zero — silently, with a result that
    // looks real. Ordering by original offset made that likely: a 3.5 MiB graph
    // input assigned a high offset was pushed into stripe 1 while the 864 MiB
    // tensor consuming it sat in stripe 0, purely because of where the planner
    // had put them.
    //
    // First-use order groups tensors that are alive together, which is exactly
    // the set an op touches at once. It is a heuristic, not a guarantee — a
    // single op whose operands exceed one stripe cannot be satisfied by any
    // ordering — so the compile-time check in `lower.rs` still runs and still
    // refuses rather than computing something plausible.
    let mut first_use: HashMap<usize, usize> = HashMap::new();
    for (step, id) in plan.schedule.iter().enumerate() {
        if let Some(a) = plan.assignments.get(id) {
            first_use.entry(a.offset).or_insert(step);
        }
    }
    let unscheduled = plan.schedule.len();
    let mut ordered: Vec<(usize, usize)> = slot_size.into_iter().collect();
    // Ties and anything absent from the schedule fall back to offset order, so
    // the placement stays deterministic.
    ordered.sort_by_key(|(off, _)| (*first_use.get(off).unwrap_or(&unscheduled), *off));

    let mut remap: HashMap<usize, usize> = HashMap::with_capacity(ordered.len());
    let mut cursor = 0usize;
    for (old_off, size) in ordered {
        if size > shard_cap {
            panic!(
                "rlx-wgpu: tensor slot at byte {old_off} is {size} bytes \
                 ({:.2} GiB) — exceeds a whole buffer ({:.2} GiB shard cap). A \
                 single tensor cannot exceed one buffer; chunk the oversized op \
                 over its output dimension at the graph level.",
                size as f64 / (1u64 << 30) as f64,
                shard_cap as f64 / (1u64 << 30) as f64,
            );
        }
        cursor = cursor.div_ceil(16) * 16;
        let local = cursor % shard_cap;
        // A slot that fits a whole buffer but not the reserve-trimmed `usable`
        // zone (e.g. this model's ~2 GiB stacked 256-expert MoE codes, which is
        // read only via host-delegate COPY, never bound as a >usable storage
        // buffer) gets its OWN dedicated shard: start on a fresh shard boundary so
        // it can use the full buffer, and push the next slot to the following
        // shard. The stage reserve only matters for shards that hold staged ops.
        if size > usable {
            if local != 0 {
                cursor = cursor.div_ceil(shard_cap) * shard_cap;
            }
            remap.insert(old_off, cursor);
            cursor += size;
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
            continue;
        }
        // Jump to next stripe when the slot would enter the stage reserve or
        // cross a hard shard boundary.
        if local + size > usable {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        remap.insert(old_off, cursor);
        cursor += size;
    }

    for a in plan.assignments.values_mut() {
        a.offset = *remap.get(&a.offset).expect("rlx-wgpu: missing slot remap");
    }

    // Preserve scratch / padding past the last assigned slot.
    //
    // Scratch MAY live in the per-stripe stage reserve (that zone exists for
    // staging copies). Do **not** open a brand-new empty shard just to host
    // tail bytes — that turned a ~3 GiB KittenTTS wave arena into 2×4 GiB
    // buffers (8 GiB) and is an easy GPU OOM on 8–16 GiB cards.
    cursor = cursor.div_ceil(16) * 16;
    if tail_extra > 0 {
        let local = cursor % shard_cap;
        // Jump only when even usable+reserve (the full stripe) can't hold it.
        if local.saturating_add(tail_extra) > shard_cap {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        cursor = cursor.saturating_add(tail_extra);
    }
    // Pad to the end of the *current* stripe when we still sit in the data
    // zone, so that stripe keeps its stage reserve. If we're already in the
    // reserve (or exactly on a boundary), keep the exact byte size — the
    // last physical buffer is allocated as `(size - begin).min(shard_cap)`.
    let local = cursor % shard_cap;
    if local > 0 && local <= usable {
        cursor = cursor.div_ceil(shard_cap) * shard_cap;
    }
    plan.arena_size = cursor.max(1);
}

fn snap_plan_to_shards(plan: &mut MemoryPlan, shard_cap: usize) {
    snap_plan_to_shards_ex(plan, shard_cap, shard_stage_reserve());
}

#[cfg(test)]
mod shard_pack_tests {
    use super::*;
    use rlx_ir::NodeId;

    #[test]
    fn snap_plan_keeps_slots_out_of_stage_reserve() {
        // Cap smaller than SHARD_STAGE_RESERVE → usable collapses to 256 bytes.
        let shard_cap = 4096usize;
        let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
        assert_eq!(usable, 256);

        let mut plan = MemoryPlan {
            arena_size: 10_000,
            assignments: HashMap::from([
                (
                    NodeId(0),
                    rlx_opt::memory::BufferSlot {
                        offset: 0,
                        size: 200,
                    },
                ),
                (
                    NodeId(1),
                    rlx_opt::memory::BufferSlot {
                        offset: 200,
                        size: 200,
                    },
                ),
                (
                    NodeId(2),
                    rlx_opt::memory::BufferSlot {
                        offset: 5000,
                        size: 100,
                    },
                ),
            ]),
            schedule: Vec::new(),
        };
        snap_plan_to_shards(&mut plan, shard_cap);
        for a in plan.assignments.values() {
            let local = a.offset % shard_cap;
            assert!(
                local + a.size.max(1) <= usable,
                "slot @{}+{} invades reserve (usable={usable})",
                a.offset,
                a.size
            );
        }
        // Last stripe may be partial when scratch sits in the stage reserve.
        assert!(plan.arena_size >= usable);
    }

    #[test]
    fn snap_plan_does_not_open_empty_shard_for_scratch() {
        // Mimic KittenTTS wave: ~3 GiB of tensors + modest scratch under a
        // ~4 GiB shard. Old packing opened a second full 4 GiB stripe for the
        // tail → 8 GiB GPU alloc (OOM risk). Scratch must stay in-stripe.
        let shard_cap = 4 * 1024 * 1024 * 1024usize; // 4 GiB
        let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
        let data = usable - 64 * 1024 * 1024; // leave headroom in data zone
        let scratch = 32 * 1024 * 1024usize;
        let mut plan = MemoryPlan {
            arena_size: data + scratch,
            assignments: HashMap::from([(
                NodeId(0),
                rlx_opt::memory::BufferSlot {
                    offset: 0,
                    size: data,
                },
            )]),
            schedule: Vec::new(),
        };
        snap_plan_to_shards(&mut plan, shard_cap);
        assert!(
            plan.arena_size <= shard_cap,
            "scratch opened an extra shard: arena={} > shard_cap={shard_cap} (OOM risk)",
            plan.arena_size
        );
        let a = plan.assignments.get(&NodeId(0)).unwrap();
        assert!(a.offset + a.size <= usable);
    }
}

/// One contiguous *logical* arena + per-node byte offsets. Lives for the
/// entire executable graph's lifetime.
///
/// When the planned size exceeds wgpu `max_buffer_size` (~4 GiB), the
/// logical range is striped across [`Self::extra_shards`] physical buffers
/// of [`Self::shard_size`] bytes each (`buffer` is shard 0). Bind helpers
/// map a window onto one shard and report a byte `rebase` so uniform
/// offsets stay relative to the bound range.
pub struct Arena {
    /// Shard 0 (or the only buffer when unsharded).
    pub buffer: wgpu::Buffer,
    /// Shards 1..N-1 when the logical arena exceeds `max_buffer_size`.
    pub extra_shards: Vec<wgpu::Buffer>,
    /// Bytes per shard (0 = unsharded single buffer of [`Self::size`]).
    pub shard_size: usize,
    /// Optional shadow buffer holding f16 versions of every value
    /// written via `write_f32`. Sized at half the arena byte budget
    /// (each f32 element pairs with an f16 element at the same logical
    /// index — i.e. f16_off = f32_off / 2). Created only when the
    /// device exposes the `SHADER_F16` feature; matmul kernels with
    /// f16-typed B input bind both `buffer` (for f32 activations) and
    /// `f16_buffer` (for f16 weights). Halves global memory traffic
    /// on the dominant matmul reads.
    pub f16_buffer: Option<wgpu::Buffer>,
    /// Optional PACKED-BF16 weight side-buffer (raw bf16 bits, 2 per u32
    /// word). Holds `Op::MatMul` weight params whose IR dtype is BF16 that
    /// would otherwise widen to f32 in `buffer`. Indexed by the SAME logical
    /// element index as `buffer` (bf16 element i ↔ f32 element i), so the
    /// packed matmul kernel (`matmul_bf16w`) reads B at `b_off` = the global
    /// arena f32-word index and byte offset `f32_byte_off / 2`. Written by
    /// [`Arena::write_bf16_packed`] (and mirrored from [`Arena::write_f32`]);
    /// the shader unpacks `bitcast<f32>(bits << 16)`. Allocated on first
    /// [`Arena::register_bf16_packed`]. Halves B read bandwidth AND avoids the
    /// host bf16→f32 widen + ne*4 arena upload for these params.
    pub bf16_weight_buffer: Option<wgpu::Buffer>,
    /// Node ids stored PACKED in `bf16_weight_buffer` (not widened into
    /// `buffer`). Lets the runtime wrapper decide, at `set_param_typed(BF16)`
    /// time, whether to pack (2 B/elem) or widen to f32.
    pub bf16_packed_nodes: HashSet<NodeId>,
    /// Per-node byte offset into the logical arena (or weight buffer).
    pub offsets: HashMap<NodeId, usize>,
    /// Per-node byte length.
    pub lens: HashMap<NodeId, usize>,
    /// Total logical arena size in bytes.
    pub size: usize,
    /// Byte offset of the tail scratch zone (also `size - scratch_bytes`).
    /// Set when callers request scratch via `from_plan_with_scratch`.
    /// Reuseable across ops since scratch is temporary — only one
    /// op writes to it at a time within a schedule.
    pub scratch_off: usize,
    /// Size in bytes of the tail scratch zone (0 when not used).
    pub scratch_bytes: usize,
    /// Separate buffer(s) holding parked params (packed quant and/or large
    /// F32). When the logical weight range exceeds `max_buffer_size`, it is
    /// striped across [`Self::weight_extra_shards`] like the act arena.
    /// Only fused GGUF GEMV binds these directly; other ops stage into the
    /// act window.
    pub weight_buffer: Option<wgpu::Buffer>,
    /// Weight shards 1..N-1 when the logical weight range exceeds one buffer.
    pub weight_extra_shards: Vec<wgpu::Buffer>,
    /// Bytes per weight shard (0 = single unsharded `weight_buffer`).
    pub weight_shard_size: usize,
    /// Per-node byte offset into the logical weight buffer (quant / parked).
    pub weight_offsets: HashMap<NodeId, usize>,
    /// Per-node, the slot the plan assigned ([`plan_node_slots`]) — the bytes a
    /// staged read of that node's operand has to fit. Distinct from [`Self::lens`],
    /// which `compile_static_inner` overwrites with the node's *actual* byte length
    /// (`elems * 4`) as soon as the arena exists.
    ///
    /// Recorded at construction rather than recomputed by
    /// [`Self::record_param_storage`]'s caller: these are the slots of the plan the
    /// arena was laid out under — after `from_plan`'s physical snap, which re-places and
    /// re-sizes slots — which the caller's plan is not once a snap has happened.
    pub(crate) plan_slots: HashMap<NodeId, usize>,
    /// Per byte offset, what a *parameter's* storage holds from that offset on, in
    /// either region — the activation arena's plan slots, the parked weight buffer's own
    /// sizing. What [`Self::param_storage_at`] answers, filled by
    /// [`Self::record_param_storage`].
    pub(crate) param_slots: HashMap<usize, ParamStorage>,
}

/// How to bind a storage window onto an [`Arena`] (possibly sharded).
pub struct ArenaBindSpec<'a> {
    pub buffer: &'a wgpu::Buffer,
    /// Bind offset into `buffer` (256-byte aligned).
    pub local_base: u64,
    pub size: u64,
    /// Subtract from logical byte offsets before dividing by 4 for uniforms.
    pub rebase: u64,
}

/// High bit tagging a byte offset as living in [`Arena::weight_buffer`] rather
/// than the main arena buffer. Real arena offsets are < 4 GiB, so a high bit is
/// free. On 64-bit use bit 62; on 32-bit wasm (`usize` = 32 bits, `1 << 62`
/// overflows) use bit 31 — browser WebGPU arenas are far below 2 GiB.
#[cfg(target_pointer_width = "64")]
pub const WEIGHT_BUF_TAG: usize = 1usize << 62;
#[cfg(not(target_pointer_width = "64"))]
pub const WEIGHT_BUF_TAG: usize = 1usize << 31;

#[inline]
pub fn is_weight_off(off: usize) -> bool {
    off & WEIGHT_BUF_TAG != 0
}
#[inline]
pub fn raw_weight_off(off: usize) -> usize {
    off & !WEIGHT_BUF_TAG
}

/// Whether this device runs Expand/Concat/Transpose/Narrow on the *host*.
///
/// Mirrors `lower::wgpu_prefer_structure_host`'s device-level conditions. Host
/// structural steps are deferred — their arena write lands at a later flush —
/// so the planner has to keep those slots reserved. Without it a subsequent GPU
/// op takes the slot and the deferred write corrupts whatever now owns it,
/// nondeterministically, and only on backends that host: Metal keeps these on
/// the GPU, which is why it presented as a family of Vulkan-only DSP/eig/pad
/// bugs rather than one allocator issue.
fn hosts_structural_ops() -> bool {
    if rlx_ir::env::flag("RLX_WGPU_SHARD_GPU") {
        return false;
    }
    rlx_ir::env::flag("RLX_WGPU_FORCE_HOST") || crate::device::coop_discrete_backend()
}

/// Tell the planner about it before every plan — the device is resolved lazily,
/// so this cannot be done once at startup.
fn note_structural_hosting() {
    rlx_compile::memory::set_pin_host_structure(hosts_structural_ops());
}

/// Plan memory using f32-sized slots regardless of declared IR dtype, with
/// liveness-aware slot reuse — `plan_memory_f32_uniform`'s decisions, with the
/// deferred-host dequant guard derived here (`dequant_route_defers_host_read`).
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    note_structural_hosting();
    rlx_compile::memory::plan_memory_with_dequant_guard(
        graph,
        align,
        rlx_compile::memory::GuardedPlanner::F32Uniform,
        dequant_route_defers_host_read(),
    )
}

/// Native-width sibling of [`plan_f32_uniform`] — every slot at its true dtype
/// byte width (bf16/f16 = 2 B). For a uniformly low-precision graph whose kernels
/// all run native dtypes; do NOT use when bool/int activations widen to f32.
pub fn plan_native(graph: &Graph, align: usize) -> MemoryPlan {
    note_structural_hosting();
    rlx_compile::memory::plan_memory_with_dequant_guard(
        graph,
        align,
        rlx_compile::memory::GuardedPlanner::Native,
        dequant_route_defers_host_read(),
    )
}

/// Hybrid sibling of [`plan_f32_uniform`] — packs `Param` weights + F16/BF16
/// activations at native width, keeps everything else f32-uniform. Lets a
/// mixed-precision wgpu graph (e.g. a packed bf16 LM head + bf16 hidden states)
/// halve those slots while bool/int activations stay f32-safe.
pub fn plan_hybrid(graph: &Graph, align: usize) -> MemoryPlan {
    note_structural_hosting();
    rlx_compile::memory::plan_memory_with_dequant_guard(
        graph,
        align,
        rlx_compile::memory::GuardedPlanner::Hybrid,
        dequant_route_defers_host_read(),
    )
}

impl Arena {
    /// Build an arena from a memory plan with an extra tail scratch zone
    /// of `scratch_bytes` reserved past the plan's arena_size. Useful for
    /// ops that need throwaway temp storage that doesn't fit in a
    /// workgroup-shared variable.
    pub fn from_plan_with_scratch(
        device: &wgpu::Device,
        plan: &MemoryPlan,
        scratch_bytes: usize,
    ) -> Self {
        if scratch_bytes == 0 {
            return Self::from_plan(device, plan);
        }
        // Round up to 16 for storage-binding alignment. Fold scratch into the
        // planned size so sharding (when needed) covers the tail too.
        let scratch_aligned = scratch_bytes.div_ceil(16) * 16;
        let mut grown = plan.clone();
        grown.arena_size = plan.arena_size + scratch_aligned;
        let mut arena = Self::from_plan(device, &grown);
        arena.scratch_bytes = scratch_aligned;
        // After possible shard-boundary repack, scratch is the last zone.
        arena.scratch_off = arena.size.saturating_sub(scratch_aligned);
        arena
    }

    /// Build an arena from a memory plan. Allocates one big buffer when the
    /// plan fits `max_buffer_size`; otherwise stripes the logical range across
    /// multiple ≤4 GiB shards ([`Self::extra_shards`]).
    ///
    /// When the plan fits in one allocation but exceeds
    /// `max_storage_buffer_binding_size` (`RLX_WGPU_LARGE_BUFFERS`), snap into
    /// virtual bind-sized stripes (`shard_size > 0`, `extra_shards` empty) so
    /// staging uses per-stripe reserves instead of clobbering live params.
    /// Refuse to build a striped activation arena.
    ///
    /// This used to be a warning, and the warning said — correctly — that a
    /// striped arena produces wrong results silently. Continuing anyway means
    /// the caller gets numbers, not an error, and only finds out if they
    /// happen to check against another backend.
    ///
    /// Measured again on SynthStrip: exact at 64^3 and 128^3, and at 192^3 —
    /// the size `StripNet` pads a real head to — every one of 7,077,888 output
    /// voxels was wrong, 97.8% of full scale, while the run reported success.
    /// The straddle guard hosts the 108 ops whose operands cross a stripe and
    /// the host staging stitches per-shard reads correctly, so the mitigation
    /// is real but plainly incomplete.
    ///
    /// Since then the *addressing-honesty* class is covered: an op whose
    /// lowering hands the arena to its kernel by absolute offset is hosted below
    /// the per-binding ceiling (`supported_ops::addresses_arena_by_absolute_offset`
    /// + `should_host_at_ceiling`), so it computes correctly instead of reading a
    /// re-based window. Physical striping itself remains an unsupported opt-in
    /// that still produces wrong results elsewhere — the mitigation is partial,
    /// not a licence to stripe.
    ///
    /// `RLX_WGPU_ALLOW_SHARD=1` restores the old warn-and-continue for anyone
    /// whose graph happens to survive it. It cannot reach an unbounded stripe
    /// count either: [`MAX_SHARDS`] is checked before this is called, so the
    /// geometry is already bounded when the opt-in is consulted.
    fn refuse_or_warn_about_sharding(logical: usize, shards: usize, cap: usize) {
        if !rlx_ir::env::flag("RLX_WGPU_ALLOW_SHARD") {
            panic!(
                "rlx-wgpu: REFUSED (striped activation arena): the arena needs {:.2} GiB and \
                 would be striped across {shards} buffers of {:.2} GiB.\n\
                 \x20 Striped arenas produce WRONG RESULTS on this backend, silently — on \
                 SynthStrip at 192^3 every output voxel was wrong (97.8% of scale) while the \
                 run reported success.\n\
                 \x20 Keep the largest intermediate small enough to avoid striping, use \
                 another backend, or set RLX_WGPU_ALLOW_SHARD=1 to get the old \
                 warn-and-continue behaviour (unsupported — it warns on every use).",
                logical as f64 / (1u64 << 30) as f64,
                cap as f64 / (1u64 << 30) as f64,
            );
        }
        // Every use, not once per process: this opt-in is not a supported route,
        // and the resulting numbers are wrong. A diagnostic that a caller can see
        // once and then forget — or silence entirely — is not a diagnostic.
        eprintln!(
            "rlx-wgpu warning: the activation arena needs {:.2} GiB and is being striped \
             across {shards} buffers of {:.2} GiB.\n\
             \x20 Sharded arenas have produced WRONG RESULTS on this backend, silently: a \
             3-D convolution stack whose output tensor was 864 MiB came back at correlation \
             0.17 against a reference runtime, while the same graph at 666 MiB — which fits \
             without striping — was exact. Reserve size made no difference.\n\
             \x20 This opt-in (RLX_WGPU_ALLOW_SHARD) is NOT a supported route and cannot be \
             silenced; this warning is emitted on every striping decision.\n\
             \x20 Check any result from this graph against `cpu`, or keep the largest \
             intermediate small enough to avoid striping.",
            logical as f64 / (1u64 << 30) as f64,
            cap as f64 / (1u64 << 30) as f64,
        );
    }

    pub fn from_plan(device: &wgpu::Device, plan: &MemoryPlan) -> Self {
        let max_buf = device.limits().max_buffer_size as usize;
        let max_binding = device.limits().max_storage_buffer_binding_size as usize;
        // 256-byte alignment for storage-buffer bind offsets.
        let shard_cap = effective_shard_cap(device);
        let bind_shard_cap = (max_binding.min(max_buf) / 256) * 256;
        assert!(shard_cap >= 256, "rlx-wgpu: max_buffer_size too small");
        assert!(
            bind_shard_cap >= 256,
            "rlx-wgpu: max_storage_binding too small"
        );

        let mut plan = plan.clone();
        let size_hint = plan.arena_size.max(1);
        // The question throughout the sizing logic below is "does this fit in ONE
        // shard?", so it is asked against `shard_cap` rather than the raw device
        // limit. The two are identical unless `RLX_WGPU_SHARD_CAP_MIB` lowers the
        // cap — which is what makes the striping path testable without allocating
        // multiple gigabytes. `max_buf` is kept for the binding-limit maths.
        if size_hint > shard_cap {
            // Prefer shrinking the per-stripe stage reserve so a compact plan
            // that only slightly exceeds max_buf (act + modest scratch) still
            // lands in one allocation. Full 768 MiB reserves turn ~2 GiB into
            // 2×2 GiB shards and OOM VirtIO/DX12 (~4 GiB total budget).
            let compact = plan.clone();
            let default_reserve = shard_stage_reserve();
            let mut reserve = default_reserve.min(shard_cap / 2);
            let mut best_fit: Option<MemoryPlan> = None;
            let mut best_over: Option<MemoryPlan> = None;
            loop {
                let mut trial = compact.clone();
                snap_plan_to_shards_ex(&mut trial, shard_cap, reserve);
                if trial.arena_size <= shard_cap {
                    if reserve < default_reserve && shard_log() {
                        eprintln!(
                            "[rlx-wgpu] physical snap with {:.0} MiB stage reserve \
                             (default {:.0} MiB) → {:.3} GiB arena (fits max_buf)",
                            reserve as f64 / (1024.0 * 1024.0),
                            default_reserve as f64 / (1024.0 * 1024.0),
                            trial.arena_size as f64 / (1u64 << 30) as f64,
                        );
                    }
                    best_fit = Some(trial);
                    break;
                }
                best_over = Some(trial);
                if reserve <= 1024 * 1024 {
                    break;
                }
                reserve = (reserve / 2).max(1024 * 1024);
            }
            if let Some(trial) = best_fit {
                plan = trial;
            } else if let Some(trial) = best_over {
                if trial.arena_size < compact.arena_size.saturating_mul(2) {
                    plan = trial;
                } else {
                    plan = compact;
                    snap_plan_to_shards_ex(&mut plan, shard_cap, 1024 * 1024);
                }
            }
        } else if size_hint > bind_shard_cap {
            // One physical buffer, bind-sized virtual stripes. Full stage
            // reserve packing can inflate a compact plan that still fits
            // `max_buf` into multi-GiB waste (KittenTTS: 3.7 GiB → 6–8 GiB →
            // GPU OOM). Shrink the per-stripe reserve until the packed plan
            // fits in one allocation.
            let compact = plan.clone();
            let default_reserve = shard_stage_reserve();
            let mut reserve = default_reserve.min(bind_shard_cap / 2);
            loop {
                let mut trial = compact.clone();
                snap_plan_to_shards_ex(&mut trial, bind_shard_cap, reserve);
                if trial.arena_size <= shard_cap || reserve <= 1024 * 1024 {
                    if trial.arena_size <= shard_cap {
                        if reserve < default_reserve && shard_log() {
                            eprintln!(
                                "[rlx-wgpu] bind-stripe snap with {:.0} MiB stage reserve \
                                 (default {:.0} MiB) → {:.3} GiB arena (fits max_buf)",
                                reserve as f64 / (1024.0 * 1024.0),
                                default_reserve as f64 / (1024.0 * 1024.0),
                                trial.arena_size as f64 / (1u64 << 30) as f64,
                            );
                        }
                        plan = trial;
                    } else if shard_log() {
                        eprintln!(
                            "[rlx-wgpu] bind-stripe snap still {:.3} GiB (> shard_cap) at \
                             {:.0} MiB reserve; keeping compact {:.3} GiB layout",
                            trial.arena_size as f64 / (1u64 << 30) as f64,
                            reserve as f64 / (1024.0 * 1024.0),
                            size_hint as f64 / (1u64 << 30) as f64,
                        );
                        plan = compact;
                    } else {
                        plan = compact;
                    }
                    break;
                }
                reserve = (reserve / 2).max(1024 * 1024);
            }
        }
        let mut size = plan.arena_size.max(1); // wgpu hates zero-sized allocs
        // Bind-stripe snap can still leave size > max_buf when size_hint was
        // already over the cap; re-pack onto physical shard boundaries.
        // Guard: never prefer an inflated multi-shard layout when the compact
        // plan fit in one buffer (handled above).
        if size > shard_cap {
            snap_plan_to_shards(&mut plan, shard_cap);
            size = plan.arena_size.max(1);
        }

        // Which geometry this size allocates, decided by one pure function so the
        // stripe bound can only ever be asked of a *physical* geometry (see
        // `arena_stripes`).
        let layout = arena_stripes("the activation arena", size, shard_cap, bind_shard_cap);
        let (buffer, extra_shards, shard_size) = match layout {
            ArenaStripes::One { shard_size } => {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu arena"),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if shard_size > 0 && shard_log() {
                    let n = shard_count(size, shard_size);
                    eprintln!(
                        "[rlx-wgpu] virtual-sharded arena: physical={:.3} GiB → {n} × {:.3} GiB bind stripes",
                        size as f64 / (1u64 << 30) as f64,
                        shard_size as f64 / (1u64 << 30) as f64,
                    );
                }
                (buffer, Vec::new(), shard_size)
            }
            ArenaStripes::Physical { shards: n_shards } => {
                // Reject tensors that cannot fit in one shard.
                for (id, a) in &plan.assignments {
                    if a.size > shard_cap {
                        panic!(
                            "rlx-wgpu: node {id:?} slot {} bytes exceeds shard cap {} \
                         (max_buffer_size); shrink max_seq or split the graph",
                            a.size, shard_cap
                        );
                    }
                    let end = a.offset.saturating_add(a.size.max(1));
                    if a.offset / shard_cap != (end - 1) / shard_cap {
                        panic!(
                            "rlx-wgpu: node {id:?} @{}+{} spans a {}-byte shard boundary after snap",
                            a.offset, a.size, shard_cap
                        );
                    }
                }
                // Loud by default. Striping is where this backend has been measured
                // to return wrong numbers without saying so, and a diagnostic behind
                // a flag nobody sets is no diagnostic at all.
                Self::refuse_or_warn_about_sharding(size, n_shards, shard_cap);
                if shard_log() {
                    eprintln!(
                        "[rlx-wgpu] sharded arena: logical={:.3} GiB → {n_shards} × {:.3} GiB shards",
                        size as f64 / (1u64 << 30) as f64,
                        shard_cap as f64 / (1u64 << 30) as f64,
                    );
                }
                let mut shards = Vec::with_capacity(n_shards);
                for i in 0..n_shards {
                    let begin = i * shard_cap;
                    let this = (size - begin).min(shard_cap).max(256);
                    shards.push(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu arena shard"),
                        size: this as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                }
                let buffer = shards.remove(0);
                (buffer, shards, shard_cap)
            }
        };

        // Mirror f16 shadow buffer: half the byte size since each f32
        // slot maps to an f16 slot at the same logical element index.
        // On arenas larger than one bind window, allocate a capped f16
        // buffer (≤ max_storage_buffer_binding_size) so matmul can use
        // f16_weight_bind_range instead of staging multi‑GiB weights.
        //
        // Discrete Vulkan/DX12 Kitten paths host Int8 dequant and do not need
        // a 2 GiB f16 mirror — that alone OOMs 12 GiB cards next to a sharded
        // act arena. Skip unless `RLX_WGPU_F16_WEIGHTS=1`.
        let want_f16 = device.features().contains(wgpu::Features::SHADER_F16)
            && !rlx_ir::env::flag("RLX_WGPU_NO_F16_SHADOW")
            // VirtIO / DX12 often expose only ~2 GiB buffers *and* a small
            // total GPU budget — a full-size f16 mirror OOMs next to weights.
            && max_buf > (2usize << 30)
            && (shard_size == 0
                || rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS")
                || !crate::device::coop_discrete_backend());
        let f16_buffer = if want_f16 {
            // Never larger than the f16 image of the plan itself. The mirror is
            // addressed by logical offset / 2, so a plan-sized buffer holds every
            // range a binding can legitimately ask for, and `f16_shadow_bind_range`
            // clamps whatever a caller asks for to the buffer it actually got.
            //
            // A fixed single-window mirror at the device's binding limit (about
            // 4 GiB on Metal) allocated that much on *every* compile, so a run
            // that striped the graph — any small `RLX_WGPU_SHARD_CAP_MIB` turns
            // every graph into stripes — wired multiple GiB per compile, starved
            // the machine of memory and took the OS down with a watchdog panic.
            let f16_size = f16_shadow_arena_size(&plan).min(max_binding);
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena f16"),
                size: f16_size.max(256) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let mut offsets = HashMap::with_capacity(plan.assignments.len());
        let mut lens = HashMap::with_capacity(plan.assignments.len());
        for (id, a) in &plan.assignments {
            offsets.insert(*id, a.offset);
            lens.insert(*id, a.size);
        }
        let plan_slots = plan_node_slots(&plan);
        Self {
            buffer,
            extra_shards,
            shard_size,
            f16_buffer,
            bf16_weight_buffer: None,
            bf16_packed_nodes: HashSet::new(),
            offsets,
            lens,
            size,
            scratch_off: 0,
            scratch_bytes: 0,
            weight_buffer: None,
            weight_extra_shards: Vec::new(),
            weight_shard_size: 0,
            weight_offsets: HashMap::new(),
            plan_slots,
            param_slots: HashMap::new(),
        }
    }

    /// Build an arena that keeps large params in a SEPARATE `weight_buffer`
    /// so the activation arena stays under wgpu's `max_buffer_size` / bind
    /// caps. Packed U8/I8 quant always goes there; F32/F16 params larger than
    /// 512 MiB (e.g. Qwen3 tied lm_head) do too — otherwise NVIDIA's ~2 GiB
    /// storage bind virtual-shards the act arena and matmul panics. Smaller
    /// F32 params stay in the act buffer. `offset()` returns a WEIGHT_BUF_TAG-
    /// tagged offset for weight-buffer params; fused GGUF GEMV binds them
    /// directly, other ops stage into the act window.
    pub fn from_plan_split(
        device: &wgpu::Device,
        plan: &MemoryPlan,
        graph: &Graph,
        scratch_bytes: usize,
    ) -> Self {
        use rlx_ir::Op;
        let _ = plan;
        // Start from a COMPACT activations-only plan (params unassigned), then
        // place every param: packed quant weights (U8/I8) → dedicated weight
        // buffer; everything else → arena tail. Relocating from the full plan
        // does NOT shrink the arena (the planner aliases the persistent packed
        // weights onto low offsets and leaves an activation at the high offset),
        // so we re-plan without params to actually compact it.
        let mut new_plan = rlx_compile::memory::plan_memory_with_dequant_guard(
            graph,
            16,
            rlx_compile::memory::GuardedPlanner::F32UniformNoParams,
            dequant_route_defers_host_read(),
        );
        let walign = 256usize; // storage-buffer offset alignment for windowed binds
        let a = 16usize;
        let mut weight_offsets: HashMap<NodeId, usize> = HashMap::new();
        let mut weight_cursor = 0usize;
        let mut tail = new_plan.arena_size;
        // If activations + all params would exceed the storage bind limit,
        // park every param in the weight buffer so the act arena stays in one
        // bind window. Otherwise NVIDIA virtual-shards (~2 GiB stripes), which
        // disables matmul param-anchor and panics on large F32 lm_heads.
        let max_bind = device.limits().max_storage_buffer_binding_size as usize;
        let bind_cap = (max_bind / 256) * 256;
        // NOTE two distinct quantities. `max_buf` is the adapter's TRUE limit and
        // drives the "is this the NVIDIA ~2 GiB storage-bind path" heuristics
        // below; `weight_shard_cap` is the (possibly test-lowered) shard size.
        // Conflating them would let `RLX_WGPU_SHARD_CAP_MIB` flip a hardware
        // heuristic, which is not what that knob is for.
        let max_buf = device.limits().max_buffer_size as usize;
        let weight_shard_cap = effective_shard_cap(device);
        let scratch_aligned = scratch_bytes.div_ceil(16) * 16;
        let mut param_bytes_est = 0usize;
        for node in graph.nodes() {
            if matches!(&node.op, Op::Param { .. }) {
                let ne = node.shape.num_elements().unwrap_or(0).max(1);
                let dt = node.shape.dtype();
                param_bytes_est = param_bytes_est.saturating_add(weight_param_bytes(dt, ne));
            }
        }
        let park_all_params = rlx_ir::env::flag("RLX_WGPU_ALL_PARAMS_WEIGHT")
            || ((new_plan.arena_size + param_bytes_est + scratch_aligned) > bind_cap
                // Multi-shard weight buffers (~2 GiB × N) OOM on VirtIO/WARP
                // and other adapters whose *total* GPU budget is ≈ one
                // `max_buffer_size`. Only park-all when everything fits in one
                // weight buffer.
                && param_bytes_est <= weight_shard_cap)
            // Same small-budget adapters: even when compact act + params fit
            // one bind window, parking only the >512 MiB F32 tensors leaves
            // the rest in the act arena → multi-×2 GiB act shards + weight
            // OOM. If all params fit in one weight buffer, park them all.
            || (param_bytes_est <= weight_shard_cap && max_buf <= (2usize << 30));
        // Prefer parking the largest params first so lm_head / embeddings land
        // in the weight buffer when only a subset fits. Tie-break by param
        // name (not NodeId) so prefill/decode graphs pack identically and can
        // share one GPU weight buffer.
        let mut park_candidates: Vec<(NodeId, usize, usize, rlx_ir::DType, bool, String)> =
            Vec::new();
        for node in graph.nodes() {
            if new_plan.assignments.contains_key(&node.id) {
                continue;
            }
            if rlx_compile::memory::is_pure_view(graph, node) {
                continue;
            }
            let ne = node.shape.num_elements().unwrap_or(0).max(1);
            let dt = node.shape.dtype();
            let is_param = matches!(&node.op, Op::Param { .. });
            let nbytes = weight_param_bytes(dt, ne);
            let name = match &node.op {
                Op::Param { name } => name.clone(),
                _ => String::new(),
            };
            park_candidates.push((node.id, ne, nbytes, dt, is_param, name));
        }
        park_candidates.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.5.cmp(&b.5))
                .then_with(|| a.0.0.cmp(&b.0.0))
        });

        const LARGE_F32_PARAM: usize = 512 * 1024 * 1024;
        // On adapters whose max_buffer_size is ≈2 GiB (VirtIO/DX12 WARP),
        // packing only >512 MiB params into the weight buffer leaves the rest
        // in the act arena → multi-shard act + weight OOMs. Fill the single
        // weight shard with any param while room remains (largest first).
        let fill_weight_shard = max_buf <= (2usize << 30);
        let mut named_weight_slots: Vec<(String, usize, usize)> = Vec::new();
        for (id, ne, nbytes, dt, is_param, name) in park_candidates {
            let packed_quant = matches!(dt, rlx_ir::DType::U8 | rlx_ir::DType::I8);
            // A generic op reaches a weight-buffer param only by staging a copy
            // into the act bind window, and that copy has to fit the scratch
            // reserve. Parking a param bigger than the reserve is therefore a
            // dead end: `arena_off_in_window_or_stage` can only panic on it.
            // Keep such a param in the act arena instead, as long as the act
            // arena still fits one bind window — that is strictly better than
            // both alternatives (a panic, or a reserve grown to hundreds of MiB
            // that every graph then pays for). Packed U8/I8 quant is exempt:
            // fused GEMV binds the weight buffer directly and never stages.
            //
            // Qwen3-0.6B F32 is the case that needs this — its 622 MiB tied
            // lm_head and embedding are far past any sane reserve, while the
            // activations are ~1.2 GiB, so both fit the act arena comfortably.
            let unstageable = !packed_quant && nbytes > scratch_aligned;
            let act_after = tail.div_ceil(a) * a + nbytes + scratch_aligned;
            let keep_in_act = unstageable && act_after <= bind_cap;
            let to_weight = is_param
                && !keep_in_act
                && (packed_quant
                    || nbytes > LARGE_F32_PARAM
                    || park_all_params
                    || fill_weight_shard);
            if to_weight {
                if nbytes > weight_shard_cap {
                    // Fall through to act arena — caller must stage per op.
                    tail = tail.div_ceil(a) * a;
                    new_plan.assignments.insert(
                        id,
                        rlx_opt::memory::BufferSlot {
                            offset: tail,
                            size: ne * 4,
                        },
                    );
                    tail += ne * 4;
                    continue;
                }
                weight_cursor = weight_cursor.div_ceil(walign) * walign;
                // Single-shard budget: spill to act once the weight buffer is full.
                if weight_cursor.saturating_add(nbytes) > weight_shard_cap {
                    tail = tail.div_ceil(a) * a;
                    new_plan.assignments.insert(
                        id,
                        rlx_opt::memory::BufferSlot {
                            offset: tail,
                            size: ne * 4,
                        },
                    );
                    tail += ne * 4;
                    continue;
                }
                named_weight_slots.push((name, nbytes, weight_cursor));
                weight_offsets.insert(id, weight_cursor);
                weight_cursor += nbytes;
            } else {
                tail = tail.div_ceil(a) * a;
                new_plan.assignments.insert(
                    id,
                    rlx_opt::memory::BufferSlot {
                        offset: tail,
                        size: ne * 4,
                    },
                );
                tail += ne * 4;
            }
        }
        // Alias pure views onto roots now that params live in `assignments`
        // (or `weight_offsets` for packed quant). Iterate to a fixed point so
        // view-of-view chains resolve after the intermediate view is placed.
        let aliases = rlx_compile::memory::collect_view_aliases(graph);
        let mut pending: Vec<NodeId> = aliases.keys().copied().collect();
        pending.sort_by_key(|id| id.0);
        let mut guard = pending.len() + 2;
        while !pending.is_empty() && guard > 0 {
            guard -= 1;
            let mut next = Vec::new();
            for id in pending {
                if new_plan.assignments.contains_key(&id) || weight_offsets.contains_key(&id) {
                    continue;
                }
                let Some(&(root, off)) = aliases.get(&id) else {
                    continue;
                };
                if let Some(&w_off) = weight_offsets.get(&root) {
                    weight_offsets.insert(id, w_off + off);
                    continue;
                }
                let Some(root_slot) = new_plan.assignments.get(&root).cloned() else {
                    next.push(id);
                    continue;
                };
                let ne = graph.node(id).shape.num_elements().unwrap_or(0).max(1);
                new_plan.assignments.insert(
                    id,
                    rlx_opt::memory::BufferSlot {
                        offset: root_slot.offset + off,
                        size: ne * 4,
                    },
                );
            }
            pending = next;
        }
        // Any remaining unassigned node (shouldn't happen) gets a fresh slot.
        for node in graph.nodes() {
            if new_plan.assignments.contains_key(&node.id) || weight_offsets.contains_key(&node.id)
            {
                continue;
            }
            let ne = node.shape.num_elements().unwrap_or(0).max(1);
            tail = tail.div_ceil(a) * a;
            new_plan.assignments.insert(
                node.id,
                rlx_opt::memory::BufferSlot {
                    offset: tail,
                    size: ne * 4,
                },
            );
            tail += ne * 4;
        }
        new_plan.arena_size = tail;
        if rlx_ir::env::flag("RLX_WGPU_DEBUG") {
            eprintln!(
                "[rlx-wgpu split] weight_params={} weight_buf={:.3}GiB act_arena={:.3}GiB all_params={}",
                weight_offsets.len(),
                weight_cursor as f64 / (1u64 << 30) as f64,
                new_plan.arena_size as f64 / (1u64 << 30) as f64,
                rlx_ir::env::flag("RLX_WGPU_ALL_PARAMS_WEIGHT"),
            );
        }
        let mut arena = Self::from_plan_with_scratch(device, &new_plan, scratch_bytes);
        if weight_cursor > 0 {
            // Same bound as the activation arena, through the same helper (see
            // MAX_SHARDS): the weight buffer stripes on the same lever, and a
            // shrunken cap turns a model's weights into the same unbounded geometry.
            // The activation arena's buffers exist by now; this one's do not.
            let n_shards =
                bound_stripes("the parked weight buffer", weight_cursor, weight_shard_cap);
            if shard_log() && n_shards > 1 {
                eprintln!(
                    "[rlx-wgpu] sharded weight buffer: logical={:.3} GiB → {n_shards} × {:.3} GiB",
                    weight_cursor as f64 / (1u64 << 30) as f64,
                    weight_shard_cap as f64 / (1u64 << 30) as f64,
                );
            }
            named_weight_slots.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
            let fp = weight_layout_fingerprint(weight_cursor, &named_weight_slots);
            let shard_size = if n_shards > 1 { weight_shard_cap } else { 0 };
            let reused = if share_weights_enabled() {
                let guard = SHARED_WEIGHTS.lock().unwrap_or_else(|e| e.into_inner());
                match guard.as_ref() {
                    Some(shared)
                        if shared.fingerprint == fp
                            && shared.logical_bytes == weight_cursor
                            && shared.shard_size == shard_size
                            && shared.extras.len() + 1 == n_shards =>
                    {
                        if shard_log() {
                            eprintln!(
                                "[rlx-wgpu] reusing shared weight buffer ({:.3} GiB)",
                                weight_cursor as f64 / (1u64 << 30) as f64,
                            );
                        }
                        Some((
                            shared.first.clone(),
                            shared.extras.clone(),
                            shared.shard_size,
                        ))
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some((first, extras, ws)) = reused {
                arena.weight_buffer = Some(first);
                arena.weight_extra_shards = extras;
                arena.weight_shard_size = ws;
            } else {
                let mut shards = Vec::with_capacity(n_shards);
                for i in 0..n_shards {
                    let begin = i * weight_shard_cap;
                    // Rounded to 4 bytes: a weight window can end exactly at the
                    // buffer end, and a storage binding must be a multiple of 4
                    // bytes — an odd tail would make the last window of a weight
                    // that fits unbindable, declining the grouped path for it.
                    let this = (weight_cursor - begin)
                        .min(weight_shard_cap)
                        .max(4)
                        .next_multiple_of(4);
                    shards.push(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu weight buffer (packed quant)"),
                        size: this as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }));
                }
                let first = shards.remove(0);
                let extras = shards;
                if share_weights_enabled() {
                    if let Ok(mut guard) = SHARED_WEIGHTS.lock() {
                        *guard = Some(SharedWeightBuffers {
                            fingerprint: fp,
                            first: first.clone(),
                            extras: extras.clone(),
                            shard_size,
                            logical_bytes: weight_cursor,
                        });
                    }
                }
                arena.weight_buffer = Some(first);
                arena.weight_extra_shards = extras;
                arena.weight_shard_size = shard_size;
            }
            arena.weight_offsets = weight_offsets;
        }
        // This planner has the graph, so it records the parameter-storage bounds
        // itself; `compile_static_inner` does it for the unsplit arena, which does not.
        arena.record_param_storage(graph);
        arena
    }

    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id) || self.weight_offsets.contains_key(&id)
    }

    /// True when the logical arena is striped across multiple GPU buffers.
    #[inline]
    pub fn is_sharded(&self) -> bool {
        self.shard_size > 0
    }

    /// Bytes per stripe, for diagnostics.
    pub fn shard_size_bytes(&self) -> usize {
        self.shard_size.max(1)
    }

    /// The whole logical arena size, for diagnostics.
    pub fn logical_size(&self) -> usize {
        self.size
    }

    /// Can a kernel that addresses this arena by ABSOLUTE offset be dispatched?
    ///
    /// The arena is one logical address space presented to a shader as a single
    /// `arena` binding. A kernel that carries absolute offsets in its uniforms is
    /// correct only when the shared bind helpers bind the whole arena at base 0,
    /// and only while every word offset it must carry fits the `u32` its uniform
    /// declares. Through a re-based window an absolute offset reads and writes the
    /// wrong slots, and an offset past `u32::MAX` words would be truncated — both
    /// silently, because WGSL makes an out-of-range access a dynamic error whose
    /// permitted outcomes (return 0 or another in-bounds value, drop the store,
    /// discard the dispatch) never include telling the caller. This is the one
    /// coarse condition the routing rule asks; it deliberately does not look at
    /// any single op's window base.
    pub fn binds_whole_at_base_zero(&self, max_storage_binding: u64) -> bool {
        !self.is_sharded()
            && (self.size as u64) <= max_storage_binding
            && (self.size / 4) <= u32::MAX as usize
    }

    /// Which stripe a logical activation offset falls in.
    ///
    /// `None` when the arena is not striped, in which case everything is in
    /// the same place and there is nothing to compare.
    pub fn shard_of(&self, off: usize) -> Option<usize> {
        (self.is_sharded() && !is_weight_off(off)).then(|| off / self.shard_size)
    }

    /// Check that every operand of an op, and its output, live in one stripe.
    ///
    /// # Why this has to be checked, and checked here
    ///
    /// A kernel binds **one** stripe. Slot placement guarantees that no single
    /// tensor straddles a boundary, which is not the same as guaranteeing that
    /// an op's *operands* share a stripe — nothing in the planner knows which
    /// tensors are used together. When they do not, the operand outside the
    /// bound stripe reads as **zero**: no error, no warning, and a result that
    /// is entirely plausible. A convolution in that state returns its bias for
    /// every voxel, which looks like a real feature map.
    ///
    /// Only the matmul path checked for this, so every other operator failed
    /// silently. This is the check for all of them, run once at compile time —
    /// before any wrong number exists — rather than per dispatch.
    ///
    /// Returns the offending offsets if the op cannot be encoded.
    ///
    /// Wired from `lower.rs::op_straddles_shards`, which runs it per op
    /// (operands + output) at compile time. When it fires, that op is routed to
    /// the host path — which reads whole tensors and is stripe-agnostic — rather
    /// than refused, so `RLX_WGPU_SHARD_GPU` stays a throughput preference and
    /// never means "return zeros". Regression: `shard_straddle_guard`, which
    /// reproduces the all-zeros result (786432/786432 outputs 0.0 instead of
    /// 18.0) with the check removed.
    pub fn straddles_shards(&self, ids: &[(usize, usize)]) -> Option<Vec<(usize, usize)>> {
        if !self.is_sharded() {
            return None;
        }
        let mut seen: Option<usize> = None;
        for &(off, _len) in ids {
            let Some(shard) = self.shard_of(off) else {
                continue;
            };
            match seen {
                None => seen = Some(shard),
                Some(first) if first != shard => {
                    return Some(ids.to_vec());
                }
                Some(_) => {}
            }
        }
        None
    }

    /// Logical byte offset of the staging reserve for the shard that contains
    /// `logical_off` (end of that stripe minus [`shard_stage_reserve`]).
    pub fn shard_stage_off(&self, logical_off: usize) -> usize {
        if !self.is_sharded() {
            return self.scratch_off;
        }
        let s = self.shard_size;
        let shard_idx = logical_off / s;
        let shard_begin = shard_idx * s;
        let shard_end = (shard_begin + s).min(self.size);
        shard_end
            .saturating_sub(shard_stage_reserve())
            .max(shard_begin)
    }

    /// Physical buffer + local offset for a logical activation byte address.
    /// Device-to-device copy inside the arena: `bytes` from `src_off` to
    /// `dst_off`, both logical activation offsets.
    ///
    /// Needed by the resident-KV row feed, which folds the new token's row into
    /// a device-resident handle without a host round trip. `rlx-metal` does this
    /// with a plain memcpy because its arena is unified memory; here it has to
    /// be a real GPU copy.
    ///
    /// **WebGPU forbids `copy_buffer_to_buffer` where source and destination are
    /// the same buffer** ("Source and destination cannot be the same buffer"),
    /// and the arena is one buffer — so the copy bounces through a caller-owned
    /// `staging` buffer. Two copies instead of one, on a KV row of a few KiB,
    /// once per layer per token.
    ///
    /// The arena may also be SHARDED, so a range can straddle a shard boundary
    /// and is split at it. `copy_buffer_to_buffer` requires 4-byte-aligned
    /// offsets and sizes, which arena slots satisfy (the arena is f32-uniform)
    /// — asserted rather than assumed, because a silent misalignment here
    /// corrupts a cache instead of failing.
    pub fn copy_within_device(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: &wgpu::Buffer,
        src_off: usize,
        dst_off: usize,
        bytes: usize,
    ) {
        if bytes == 0 {
            return;
        }
        assert!(
            src_off.is_multiple_of(4) && dst_off.is_multiple_of(4) && bytes.is_multiple_of(4),
            "rlx-wgpu: copy_within_device needs 4-byte alignment \
             (src={src_off} dst={dst_off} bytes={bytes})"
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu kv row feed"),
        });
        // Walk the range in pieces that lie within one shard on BOTH sides.
        let mut done = 0usize;
        while done < bytes {
            let (src_buf, src_local) = self.resolve_act(src_off + done);
            let (dst_buf, dst_local) = self.resolve_act(dst_off + done);
            let remaining = bytes - done;
            let chunk = if self.is_sharded() && self.shard_size > 0 {
                let src_room = self.shard_size - (src_local % self.shard_size.max(1));
                let dst_room = self.shard_size - (dst_local % self.shard_size.max(1));
                remaining.min(src_room).min(dst_room)
            } else {
                remaining
            };
            // arena -> staging -> arena. A direct arena->arena copy is a
            // validation error even when the ranges do not overlap.
            enc.copy_buffer_to_buffer(src_buf, src_local as u64, staging, 0, chunk as u64);
            enc.copy_buffer_to_buffer(staging, 0, dst_buf, dst_local as u64, chunk as u64);
            done += chunk;
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    pub fn resolve_act(&self, global_off: usize) -> (&wgpu::Buffer, usize) {
        assert!(
            !is_weight_off(global_off),
            "rlx-wgpu: resolve_act called with weight-tagged offset {global_off:#x}"
        );
        assert!(
            global_off < self.size || self.size == 0,
            "rlx-wgpu: resolve_act off={global_off} past arena size={}",
            self.size
        );
        if !self.is_sharded() {
            return (&self.buffer, global_off);
        }
        // Virtual striping: one physical buffer, logical `shard_size` windows.
        if self.extra_shards.is_empty() {
            return (&self.buffer, global_off);
        }
        let idx = global_off / self.shard_size;
        let local = global_off % self.shard_size;
        if idx == 0 {
            (&self.buffer, local)
        } else {
            (
                self.extra_shards.get(idx - 1).unwrap_or_else(|| {
                    panic!(
                        "rlx-wgpu: shard {idx} out of range (off={global_off} size={} \
                             shard_size={} n_extra={})",
                        self.size,
                        self.shard_size,
                        self.extra_shards.len()
                    )
                }),
                local,
            )
        }
    }

    /// Bind spec for a set of activation nodes (same shard required when sharded).
    pub fn bind_spec_for_nodes(&self, device: &wgpu::Device, ids: &[NodeId]) -> ArenaBindSpec<'_> {
        const ALIGN: u64 = 256;
        let max_binding = device.limits().max_storage_buffer_binding_size;

        // Resolve span in *logical* addresses (skip weight-tagged nodes).
        let mut lo: u64 = u64::MAX;
        let mut hi: u64 = 0;
        for &id in ids {
            let off = self.offset(id);
            if is_weight_off(off) {
                continue;
            }
            let len = self.len_of(id) as u64;
            let o = off as u64;
            lo = lo.min(o);
            hi = hi.max(o.saturating_add(len));
        }
        if lo == u64::MAX {
            // No activation nodes — bind shard 0 / whole small buffer.
            let size = (self.buffer.size()).min(max_binding).max(256);
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: 0,
                size,
                rebase: 0,
            };
        }

        if !self.is_sharded() {
            // Prefer whole-arena bind when it fits (absolute offsets, rebase=0).
            if (self.size as u64) <= max_binding {
                let size = (self.size as u64).min(self.buffer.size()).max(256);
                return ArenaBindSpec {
                    buffer: &self.buffer,
                    local_base: 0,
                    size,
                    rebase: 0,
                };
            }
            let span = hi.saturating_sub(lo).max(1);
            if span > max_binding {
                panic!("rlx-wgpu: op needs {span} bytes of arena span (>{max_binding})");
            }
            let mut base = (lo / ALIGN) * ALIGN;
            let mut size = span.div_ceil(ALIGN) * ALIGN;
            size = size.max(256).min(max_binding);
            if base.saturating_add(size) > self.size as u64 {
                base = (self.size as u64).saturating_sub(size);
                base = (base / ALIGN) * ALIGN;
            }
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: base,
                size,
                rebase: base,
            };
        }

        // Sharded: entire window must live in one stripe.
        let s = self.shard_size as u64;
        let shard_lo = lo / s;
        let shard_hi = (hi.saturating_sub(1)) / s;
        if shard_lo != shard_hi {
            let mut details = String::new();
            for &id in ids.iter().take(12) {
                let off = self.offset(id);
                if is_weight_off(off) {
                    continue;
                }
                details.push_str(&format!(" {:?}@{}+{};", id, off, self.len_of(id)));
            }
            panic!(
                "rlx-wgpu: op span [{lo},{hi}) crosses shard boundary at {} \
                 (shard_size={s}). nodes:{details}",
                (shard_lo + 1) * s
            );
        }
        let shard_base = shard_lo * s;
        let span = hi.saturating_sub(lo).max(1);
        if span > max_binding {
            panic!("rlx-wgpu: op needs {span} bytes within a shard (>{max_binding})");
        }
        // Always bind the whole stripe so the per-shard stage reserve at the
        // tail stays addressable for cross-shard staging.
        if self.extra_shards.is_empty() {
            // Virtual stripes share one buffer — bind at the stripe's absolute base.
            let phys = (self.size as u64)
                .saturating_sub(shard_base)
                .min(s)
                .min(max_binding)
                .max(256);
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: shard_base,
                size: phys,
                rebase: shard_base,
            };
        }
        let (buf, _) = self.resolve_act(lo as usize);
        let shard_bytes = buf.size().min(s).min(max_binding).max(256);
        ArenaBindSpec {
            buffer: buf,
            local_base: 0,
            size: shard_bytes,
            rebase: shard_base,
        }
    }

    pub fn offset(&self, id: NodeId) -> usize {
        if let Some(&w) = self.weight_offsets.get(&id) {
            w | WEIGHT_BUF_TAG
        } else {
            *self.offsets.get(&id).unwrap_or_else(|| {
                panic!("rlx-wgpu arena: no offset for node {id:?} (not in arena or weight buffer)")
            })
        }
    }
    /// Resolve a (possibly weight-tagged) byte offset to (buffer, raw offset).
    pub fn resolve_w(&self, tagged: usize) -> (&wgpu::Buffer, usize) {
        if is_weight_off(tagged) {
            let raw = raw_weight_off(tagged);
            let Some(first) = self.weight_buffer.as_ref() else {
                panic!("weight-tagged off without weight_buffer");
            };
            if self.weight_shard_size == 0 {
                return (first, raw);
            }
            let s = self.weight_shard_size;
            let idx = raw / s;
            let local = raw % s;
            if idx == 0 {
                (first, local)
            } else {
                (
                    self.weight_extra_shards.get(idx - 1).unwrap_or_else(|| {
                        panic!(
                            "rlx-wgpu: weight off {raw} → shard {idx} missing \
                                 ({} extra shards, shard_size={s})",
                            self.weight_extra_shards.len()
                        )
                    }),
                    local,
                )
            }
        } else {
            self.resolve_act(tagged)
        }
    }
    pub fn len_of(&self, id: NodeId) -> usize {
        self.lens[&id]
    }

    /// Whether this node's f16 mirror fits in the capped f16 shadow buffer.
    pub fn param_fits_f16_mirror(&self, id: NodeId) -> bool {
        let Some(f16) = &self.f16_buffer else {
            return false;
        };
        // Weight-buffer params use a separate byte space; the f16 shadow is
        // laid out against *arena* offsets, so tagged offs never fit there.
        let off = self.offset(id);
        if is_weight_off(off) {
            return false;
        }
        let f16_off = off / 2;
        let f16_bytes = self.len_of(id) / 2;
        f16_off.saturating_add(f16_bytes) <= f16.size() as usize
    }

    /// Override the actual data length (in bytes) for a node. The
    /// backend calls this after planning to record true elem*4 sizes
    /// instead of the alignment-padded slot sizes.
    pub fn set_actual_len(&mut self, id: NodeId, bytes: usize) {
        self.lens.insert(id, bytes);
    }

    /// Write f32 data into the node's slot. The queue performs an
    /// async transfer; subsequent kernel dispatches on the same queue
    /// see the new bytes. When the device supports SHADER_F16, also
    /// downcasts and writes the same data into the f16 shadow buffer
    /// at offset `f32_offset / 2` — so matmul kernels with f16 weight
    /// bindings can read directly from there at half the bandwidth.
    pub fn write_f32(&self, queue: &wgpu::Queue, id: NodeId, data: &[f32]) {
        let off = self.offset(id);
        // PACKED-BF16 params live ONLY in `bf16_weight_buffer`; do not widen
        // them into the f32 arena slot. Downcast f32→bf16 (lossless for a
        // bf16-origin value) and write packed. Safety net for any f32-path
        // upload (e.g. deferred-graph pending replay) of a packed param.
        if self.bf16_packed_nodes.contains(&id) && !is_weight_off(off) {
            self.write_bf16_packed_at(queue, off, data);
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(data);
        // Route through chunked/shard-safe writer — large uploads (params,
        // ConcatHost, cross-shard staging) used to truncate on Metal.
        self.write_bytes_range(queue, off, bytes);
        if !is_weight_off(off) {
            self.write_f16_shadow_at(queue, off, data);
        }
    }

    /// True iff `id` is stored in the packed-bf16 weight side-buffer.
    #[inline]
    pub fn is_bf16_packed(&self, id: NodeId) -> bool {
        self.bf16_packed_nodes.contains(&id)
    }

    /// Total bytes of the packed-bf16 weight side-buffer (`None` if unused).
    pub fn bf16_weight_buffer_bytes(&self) -> Option<u64> {
        self.bf16_weight_buffer.as_ref().map(|b| b.size())
    }

    /// Record, per byte offset, what a *parameter's* storage holds from there on: the bytes
    /// [`Self::param_storage_at`] answers with — what the host route's staged read of an
    /// operand there has to fit — and the encoding it names for that operand.
    ///
    /// A view is recorded too, bounded by the storage it actually reads: the planner
    /// aliases a view onto its root, so what is available at the view's offset is the
    /// root's storage from there on — not the view's *own* slot, which says only how much
    /// that node needs and would bound the whole parameter wherever a shrinking view
    /// shares its offset. The flat plan sizes a view's own slot as if it were an
    /// activation while the parameter underneath it keeps its native width, so the
    /// root's remainder is the only number that tells the truth here.
    ///
    /// Where two parameters' storages reach one offset — a view the plan placed past its
    /// own root's slot lands on the next parameter's start — the entry keeps the tighter
    /// bound, the safe direction, and carries no encoding: the key is the offset alone, so
    /// it cannot say which of the two a read there belongs to.
    ///
    /// Both regions are recorded — the activation arena's plan slots, and the parked
    /// weight buffer's own sizing ([`weight_param_bytes`], the function that placed
    /// them) — so a parked operand and an in-arena one are measured the same way.
    ///
    /// Callers: `from_plan_split` (it has the graph, which the plan does not carry) and
    /// `compile_static_inner`, for the arena `from_plan_with_scratch` builds — that
    /// constructor never sees a graph. Either way an arena is recorded once, and every
    /// arena the compile path builds carries the record.
    pub(crate) fn record_param_storage(&mut self, graph: &Graph) {
        self.param_slots.extend(param_storage_bounds(
            graph,
            &self.offsets,
            &self.weight_offsets,
            &self.plan_slots,
        ));
    }

    /// What a *parameter's* storage holds from `off` on — its bytes, and the encoding that
    /// parameter was declared with — or `None` when no parameter's storage starts there: an
    /// activation's slot holds one f32 lane per element by construction, so a staged read of
    /// it always fits and needs no bound.
    ///
    /// The encoding is `None` as well where two parameters' storages reach the offset and the
    /// record cannot say which of them a read there reads ([`ParamStorage::dtype`]).
    ///
    /// `None` is also what an arena that was never recorded answers, and that reads as
    /// "no parameter here": the recording is an obligation of whoever builds the arena
    /// ([`Self::record_param_storage`]), and both constructions the compile path uses
    /// honour it. The plan-time check does not lean on that silence — a `Param` operand
    /// with no record is refused rather than read (`lower::assert_host_route_stageable`).
    pub(crate) fn param_storage_at(&self, off: usize) -> Option<ParamStorage> {
        self.param_slots.get(&off).copied()
    }

    /// Register `id` (a BF16 `Op::MatMul` weight Param) to live PACKED in the
    /// bf16 side-buffer, allocating that buffer on first use. Sized at half
    /// the arena byte budget — one bf16 element per f32 slot at the same
    /// logical index, mirroring the f16 shadow. Only valid for un-tagged,
    /// unsharded arena params (callers gate on [`Self::bf16_packed_eligible`]).
    pub fn register_bf16_packed(&mut self, device: &wgpu::Device, id: NodeId) {
        if self.bf16_weight_buffer.is_none() {
            // Half the arena: bf16 element i pairs with f32 element i, so the
            // packed buffer must span up to the highest weight element index.
            let bytes = (self.size / 2).max(256);
            self.bf16_weight_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena bf16 packed weights"),
                size: bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.bf16_packed_nodes.insert(id);
    }

    /// Whether `id` can be stored packed: unsharded main-arena param whose
    /// packed byte range fits one storage-binding window.
    pub fn bf16_packed_eligible(&self, device: &wgpu::Device, id: NodeId) -> bool {
        if self.is_sharded() {
            return false;
        }
        let Some(&off) = self.offsets.get(&id) else {
            return false;
        };
        if is_weight_off(off) {
            return false;
        }
        let max_binding = device.limits().max_storage_buffer_binding_size as usize;
        // Whole packed buffer is bound at offset 0 (global bf16 indexing), so
        // it must fit one binding.
        (self.size / 2).max(256) <= max_binding
    }

    /// Write raw bf16 weight bytes for `id` into the packed side-buffer.
    /// `data` is the native little-endian bf16 byte stream (`ne*2` bytes).
    pub fn write_bf16_packed(&self, queue: &wgpu::Queue, id: NodeId, data: &[u8]) {
        let off = self.offset(id);
        if is_weight_off(off) {
            return;
        }
        self.write_bf16_packed_bytes_at(queue, off, data);
    }

    /// Downcast host f32 → bf16 and write into the packed side-buffer at the
    /// slot for f32-byte-offset `off` (packed byte offset `off / 2`).
    fn write_bf16_packed_at(&self, queue: &wgpu::Queue, off: usize, data: &[f32]) {
        let mut bytes: Vec<u8> = Vec::with_capacity(data.len() * 2);
        for &v in data {
            bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        self.write_bf16_packed_bytes_at(queue, off, &bytes);
    }

    /// Write a raw bf16 byte stream into the packed side-buffer. The packed
    /// slot begins at byte `off / 2` (bf16 element i ↔ f32 element i). Pads to
    /// a u32-word boundary so the shader's `array<u32>` view is well-formed.
    fn write_bf16_packed_bytes_at(&self, queue: &wgpu::Queue, off: usize, data: &[u8]) {
        let Some(buf) = &self.bf16_weight_buffer else {
            return;
        };
        let packed_off = off / 2; // f32 byte off → bf16 byte off (parallel index)
        // Round the write up to a multiple of 4 bytes (one u32 word) so a
        // trailing odd bf16 element still lands in a fully-written word.
        let mut owned;
        let bytes: &[u8] = if data.len().is_multiple_of(4) {
            data
        } else {
            owned = data.to_vec();
            owned.resize(data.len().div_ceil(4) * 4, 0);
            &owned
        };
        if packed_off.saturating_add(bytes.len()) > buf.size() as usize {
            return;
        }
        queue.write_buffer(buf, packed_off as u64, bytes);
    }

    /// Downcast host f32 data into the f16 shadow buffer at `id`'s slot.
    pub fn write_f16_shadow(&self, queue: &wgpu::Queue, id: NodeId, data: &[f32]) {
        self.write_f16_shadow_at(queue, self.offset(id), data);
    }

    fn write_f16_shadow_at(&self, queue: &wgpu::Queue, off: usize, data: &[f32]) {
        if let Some(f16_buf) = &self.f16_buffer {
            let f16_off = off / 2;
            let mut f16_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            if !f16_data.len().is_multiple_of(2) {
                f16_data.push(half::f16::from_f32(0.0));
            }
            let f16_byte_len = f16_data.len() * 2;
            if f16_off.saturating_add(f16_byte_len) > f16_buf.size() as usize {
                return;
            }
            let f16_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_byte_len) };
            queue.write_buffer(f16_buf, f16_off as u64, f16_bytes);
        }
    }

    /// Read a node's bytes back to host f32. Uses a fresh staging buffer;
    /// hot paths should call [`read_f32_pooled`] with a reused [`ReadbackStaging`].
    pub fn read_f32(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: NodeId) -> Vec<f32> {
        read_f32_pooled(self, device, queue, id, &mut None)
    }

    /// Read a byte range from the arena (used for packed GGUF weights).
    ///
    /// Handles sharded arenas (range may span stripes) and large transfers.
    pub fn read_bytes_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        byte_off: usize,
        len: usize,
    ) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        // Weight buffer is contiguous per-shard — never use act-stripe splits.
        if is_weight_off(byte_off) {
            let raw = raw_weight_off(byte_off);
            let ws = self.weight_shard_size;
            if ws == 0 || (raw / ws) == ((raw + len - 1) / ws) {
                let (src, local) = self.resolve_w(byte_off);
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu readback bytes"),
                    size: len as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rlx-wgpu readback bytes enc"),
                });
                enc.copy_buffer_to_buffer(src, local as u64, &staging, 0, len as u64);
                queue.submit(std::iter::once(enc.finish()));

                let slice = staging.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = sender.send(r);
                });
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                receiver.recv().unwrap().unwrap();

                let view = slice.get_mapped_range().expect("buffer slice mapped");
                let out = view.to_vec();
                drop(view);
                staging.unmap();
                return out;
            }
            // Cross weight-shard: stitch.
            let mut out = vec![0u8; len];
            let mut done = 0usize;
            while done < len {
                let g = raw + done;
                let room = ws - (g % ws);
                let n = (len - done).min(room);
                let piece = self.read_bytes_range(device, queue, byte_off + done, n);
                out[done..done + n].copy_from_slice(&piece);
                done += n;
            }
            return out;
        }
        if !self.is_sharded()
            || (byte_off / self.shard_size) == ((byte_off + len - 1) / self.shard_size)
        {
            let (src, local) = self.resolve_w(byte_off);
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu readback bytes"),
                size: len as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu readback bytes enc"),
            });
            enc.copy_buffer_to_buffer(src, local as u64, &staging, 0, len as u64);
            queue.submit(std::iter::once(enc.finish()));

            let slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();

            let view = slice.get_mapped_range().expect("buffer slice mapped");
            let out = view.to_vec();
            drop(view);
            staging.unmap();
            return out;
        }
        // Cross-stripe: stitch per-shard copies.
        let mut out = vec![0u8; len];
        let mut done = 0usize;
        while done < len {
            let g = byte_off + done;
            let room = self.shard_size - (g % self.shard_size);
            let n = (len - done).min(room);
            let piece = self.read_bytes_range(device, queue, g, n);
            out[done..done + n].copy_from_slice(&piece);
            done += n;
        }
        out
    }

    /// Write raw bytes into the arena at `byte_off`.
    ///
    /// Chunks large payloads (Metal `queue.write_buffer` can truncate above
    /// ~64 MiB) and splits at shard boundaries so a single call never writes
    /// past the end of a physical stripe.
    pub fn write_bytes_range(&self, queue: &wgpu::Queue, byte_off: usize, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        const CHUNK: usize = 64 * 1024 * 1024;
        // Weight-buffer uploads must not use act-shard boundary math — the high
        // WEIGHT_BUF_TAG bit would make `% shard_size` nonsense and truncate.
        let weight = is_weight_off(byte_off);
        let mut off = 0usize;
        while off < data.len() {
            let mut n = (data.len() - off).min(CHUNK);
            if weight && self.weight_shard_size > 0 {
                let g = raw_weight_off(byte_off) + off;
                let room = self.weight_shard_size - (g % self.weight_shard_size);
                n = n.min(room);
            } else if !weight && self.is_sharded() {
                let g = byte_off + off;
                let room = self.shard_size - (g % self.shard_size);
                n = n.min(room);
            }
            // Non-final `write_buffer` sizes must be multiples of 4.
            if off + n < data.len() {
                n &= !3;
            }
            if n == 0 {
                n = (data.len() - off).min(4);
                if weight && self.weight_shard_size > 0 {
                    let g = raw_weight_off(byte_off) + off;
                    n = n.min(self.weight_shard_size - (g % self.weight_shard_size));
                } else if !weight && self.is_sharded() {
                    let g = byte_off + off;
                    n = n.min(self.shard_size - (g % self.shard_size));
                }
            }
            if n == 0 {
                break;
            }
            let (buf, local) = self.resolve_w(byte_off + off);
            queue.write_buffer(buf, local as u64, &data[off..off + n]);
            off += n;
        }
    }
}

/// Reusable MAP_READ staging buffer for output readback.
pub struct ReadbackStaging {
    buffer: wgpu::Buffer,
    capacity: usize,
}

/// Fixed 256 B MAP_READ staging for scalar (≤16 B) readback — avoids
/// `map_buffer_on_submit` + full-layout decode on MoltenVK hot paths.
pub struct TinyReadbackStaging {
    buffer: wgpu::Buffer,
}

impl TinyReadbackStaging {
    const CAPACITY: u64 = 256;

    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu tiny readback"),
                size: Self::CAPACITY,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// True when fused readback can use the tiny scalar fast path.
pub fn use_tiny_readback(layout: &ReadbackLayout, num_outputs: usize) -> bool {
    num_outputs == 1 && layout.total_bytes <= 16
}

/// After submit: decode one f32 vector from an already-mapped tiny staging buffer.
pub fn decode_tiny_mapped_f32(staging: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let len = len.max(4);
    let slice = staging.slice(..len as u64);
    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let out = bytemuck::cast_slice::<u8, f32>(&view[..len]).to_vec();
    drop(view);
    staging.unmap();
    out
}

/// After submit: map only `len` bytes and decode one f32 vector.
pub fn read_tiny_f32_after_submit(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let len = len.max(4);
    let slice = staging.slice(..len as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();
    decode_tiny_mapped_f32(staging, len)
}

impl ReadbackStaging {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn ensure(&mut self, device: &wgpu::Device, min_bytes: usize) {
        let need = min_bytes.max(256);
        if self.capacity >= need {
            return;
        }
        let cap = need.next_power_of_two().max(256);
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu readback staging"),
            size: cap as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.capacity = cap;
    }

    /// Grow-or-create staging for at least `min_bytes`.
    pub fn prepare(device: &wgpu::Device, staging: &mut Option<Self>, min_bytes: usize) {
        match staging {
            Some(s) => s.ensure(device, min_bytes),
            None => {
                let cap = min_bytes.max(256).next_power_of_two();
                *staging = Some(Self {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu readback staging"),
                        size: cap as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    capacity: cap,
                });
            }
        }
    }
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// Layout for batched output readback into a staging buffer.
#[derive(Debug, Clone)]
pub struct ReadbackLayout {
    pub regions: Vec<(usize, usize)>,
    pub total_bytes: usize,
}

impl ReadbackLayout {
    pub fn for_nodes(arena: &Arena, ids: &[NodeId]) -> Self {
        if ids.is_empty() {
            return Self {
                regions: Vec::new(),
                total_bytes: 0,
            };
        }
        if ids.len() == 1 {
            let len = arena.len_of(ids[0]);
            return Self {
                regions: vec![(0, len)],
                total_bytes: len,
            };
        }
        let mut regions = Vec::with_capacity(ids.len());
        let mut total = 0usize;
        for &id in ids {
            let len = arena.len_of(id);
            let start = total;
            total = align4(start + len);
            regions.push((start, len));
        }
        Self {
            regions,
            total_bytes: total,
        }
    }
}

/// Append arena→staging copies to an encoder (no submit).
pub fn encode_readback_copies(
    enc: &mut wgpu::CommandEncoder,
    arena: &Arena,
    staging: &wgpu::Buffer,
    ids: &[NodeId],
    layout: &ReadbackLayout,
) {
    for (&id, &(dst_off, len)) in ids.iter().zip(layout.regions.iter()) {
        let (src, local) = arena.resolve_w(arena.offset(id));
        enc.copy_buffer_to_buffer(src, local as u64, staging, dst_off as u64, len as u64);
    }
}

/// Map staging after submit and decode f32 outputs (one poll).
pub fn map_readback_f32(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    map_readback_f32_after_submit(device, staging, layout)
}

/// Block until the submission that produced the readback has finished and its
/// buffer-map callback has fired, then return.
///
/// A single submission-index `Wait` replaces the old 64–256 `poll(Poll)`
/// busy-spin followed by a full `Wait`. Each `Poll` maintain pass costs tens of
/// microseconds on Metal, so the spin alone added ~3 ms of pure CPU overhead per
/// run — the dominant cost for small graphs (an MNIST CNN forward dropped from
/// ~3.4 ms to sub-millisecond). wgpu invokes the buffer-map callback as part of
/// the poll, so the mapped range is ready when this returns.
pub fn wait_readback_map(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    _map_rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    _total_bytes: usize,
) {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    });
}

/// Schedule `map_async` on the encoder so mapping starts with submit (wgpu 29+).
pub fn schedule_readback_map(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let total = layout.total_bytes;
    let (sender, receiver) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(staging, wgpu::MapMode::Read, 0..total as u64, move |r| {
        let _ = sender.send(r);
    });
    receiver
}

fn map_readback_f32_after_submit(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    if layout.regions.is_empty() {
        return Vec::new();
    }
    let total = layout.total_bytes;
    let slice = staging.slice(..total as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let bytes = &view[..];
    let mut outs = Vec::with_capacity(layout.regions.len());
    for &(start, len) in &layout.regions {
        let chunk = &bytes[start..start + len];
        outs.push(bytemuck::cast_slice::<u8, f32>(chunk).to_vec());
    }
    drop(view);
    staging.unmap();
    outs
}

/// Decode f32 outputs after submit + map callback (used with [`schedule_readback_map`]).
pub fn decode_mapped_readback_f32(
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    if layout.regions.is_empty() {
        return Vec::new();
    }
    let total = layout.total_bytes;
    let slice = staging.slice(..total as u64);
    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let bytes = &view[..];
    let mut outs = Vec::with_capacity(layout.regions.len());
    for &(start, len) in &layout.regions {
        let chunk = &bytes[start..start + len];
        outs.push(bytemuck::cast_slice::<u8, f32>(chunk).to_vec());
    }
    drop(view);
    staging.unmap();
    outs
}

/// Dependency-free `map_async` await for wasm.
///
/// The browser is single-threaded and drives GPU completion via its event
/// loop, so an `Rc<RefCell<…>>` oneshot is sufficient: the `map_async`
/// callback fires from a microtask once the copy completes, stores the
/// result, and wakes the awaiting task. No `device.poll` is needed (and it
/// would be a no-op on the WebGPU backend anyway).
#[cfg(target_arch = "wasm32")]
pub mod wasm_async {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    #[derive(Default)]
    struct Shared {
        result: Option<Result<(), wgpu::BufferAsyncError>>,
        waker: Option<Waker>,
    }

    struct MapSignal(Rc<RefCell<Shared>>);

    /// Future resolving when the mapped buffer is ready.
    pub struct MapWait(Rc<RefCell<Shared>>);

    impl MapSignal {
        fn complete(self, r: Result<(), wgpu::BufferAsyncError>) {
            let mut b = self.0.borrow_mut();
            b.result = Some(r);
            if let Some(w) = b.waker.take() {
                w.wake();
            }
        }
    }

    impl Future for MapWait {
        type Output = Result<(), wgpu::BufferAsyncError>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let mut b = self.0.borrow_mut();
            match b.result.take() {
                Some(r) => Poll::Ready(r),
                None => {
                    b.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    /// Map `buffer[..len]` for read and return a future that resolves when the
    /// GPU has finished. Call AFTER the copy into `buffer` has been submitted.
    pub fn map_read_async(buffer: &wgpu::Buffer, len: usize) -> MapWait {
        let shared = Rc::new(RefCell::new(Shared::default()));
        let signal = MapSignal(shared.clone());
        buffer
            .slice(..len as u64)
            .map_async(wgpu::MapMode::Read, move |r| signal.complete(r));
        MapWait(shared)
    }
}

/// Read one node via a reused staging buffer (one submit + one poll).
pub fn read_f32_pooled(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    id: NodeId,
    staging: &mut Option<ReadbackStaging>,
) -> Vec<f32> {
    let off = arena.offset(id);
    let len = arena.len_of(id);
    let n_elems = len / 4;
    if n_elems == 0 {
        return Vec::new();
    }
    ReadbackStaging::prepare(device, staging, len);
    let staging = staging.as_ref().expect("staging");
    let (src, local) = arena.resolve_w(off);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu readback enc"),
    });
    enc.copy_buffer_to_buffer(src, local as u64, &staging.buffer, 0, len as u64);
    queue.submit(std::iter::once(enc.finish()));

    let slice = staging.buffer.slice(..len as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.buffer.unmap();
    out
}

/// Read several nodes with one submit + one poll (contiguous staging layout).
pub fn read_f32_many_pooled(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ids: &[NodeId],
    staging: &mut Option<ReadbackStaging>,
) -> Vec<Vec<f32>> {
    if ids.is_empty() {
        return Vec::new();
    }
    let layout = ReadbackLayout::for_nodes(arena, ids);
    ReadbackStaging::prepare(device, staging, layout.total_bytes);
    let staging_buf = staging.as_ref().expect("staging").buffer().clone();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu readback batch enc"),
    });
    encode_readback_copies(&mut enc, arena, &staging_buf, ids, &layout);
    queue.submit(std::iter::once(enc.finish()));
    map_readback_f32(device, &staging_buf, &layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::NodeId;
    use rlx_opt::memory::{BufferSlot, MemoryPlan};
    use std::collections::HashMap;

    #[test]
    fn f16_shadow_arena_accounts_for_copy_alignment_padding() {
        // Three f32 elements → six f16 bytes, padded to eight for wgpu
        // COPY_BUFFER_ALIGNMENT. The old `arena_size / 2` sizing was two
        // bytes short at this slot boundary.
        let mut assignments = HashMap::new();
        assignments.insert(
            NodeId(0),
            BufferSlot {
                offset: 32,
                size: 12,
            },
        );
        let plan = MemoryPlan {
            arena_size: 44,
            assignments,
            schedule: vec![],
        };
        assert_eq!(f16_shadow_arena_size(&plan), 24);
    }
}

/// The stripe bound ([`MAX_SHARDS`]). Pure arithmetic — no device, no shader, no
/// allocation — which is the point: the geometry that took this host down has to
/// be refusable, and inspectable, without running it.
#[cfg(test)]
mod stripe_bound_tests {
    use super::*;

    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;

    fn refusal_of(size: usize, cap: usize) -> String {
        let caught = std::panic::catch_unwind(|| bound_stripes("the activation arena", size, cap));
        let payload = caught.expect_err("this geometry must be refused");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    /// `(arena bytes, shard cap, stripes)`. The first rows are the killing lever — a
    /// few-MiB cap over a real workload's arena, which is what the bound exists to
    /// stop; the last ones are the deliberate small caps this backend's tests pin and
    /// which must keep working.
    const CASES: &[(usize, usize, usize)] = &[
        (4 * GIB, MIB, 4096),
        (4 * GIB, 4 * MIB, 1024),
        (76 * MIB, MIB, 76),
        (76 * MIB, 4 * MIB, 19),
        (256 * MIB, 4 * MIB, 64),
        (MIB, 4 * MIB, 1),
    ];

    #[test]
    fn a_cap_that_explodes_the_geometry_is_refused_and_a_sane_one_is_not() {
        for &(size, cap, expected) in CASES {
            if expected > MAX_SHARDS {
                let msg = refusal_of(size, cap);
                // Catchable and self-describing: a caller can tell this apart from a
                // process death, and the message names the case and the geometry.
                for needle in [
                    "rlx-wgpu: REFUSED (pathological stripe geometry)",
                    "the activation arena",
                    &format!("{expected} buffers"),
                ] {
                    assert!(
                        msg.contains(needle),
                        "refusal must mention {needle:?}: {msg}"
                    );
                }
            } else {
                assert_eq!(
                    bound_stripes("the activation arena", size, cap),
                    expected,
                    "{size} bytes at a {cap}-byte cap must stay usable"
                );
            }
        }
    }

    #[test]
    fn a_single_buffer_geometry_is_never_refused_over_the_bind_dimension() {
        // An adapter whose `max_buffer_size` exceeds 64× its storage-bind limit:
        // here an 8 GiB shard cap against a 64 MiB bind limit. A 6 GiB arena is
        // ONE allocated buffer, but spans 96 bind-sized windows. Counting the bind
        // dimension would exceed `MAX_SHARDS` and refuse a geometry that allocates
        // a single buffer — the wrong dimension — so the bound must be asked of
        // the physical geometry only.
        let (size, shard_cap, bind_cap) = (6 * GIB, 8 * GIB, 64 * MIB);
        assert!(
            shard_count(size, bind_cap) > MAX_SHARDS,
            "this check only means anything while the wrong dimension would exceed the bound"
        );
        assert_eq!(
            arena_stripes("the activation arena", size, shard_cap, bind_cap),
            ArenaStripes::One {
                shard_size: bind_cap
            },
            "a single-buffer geometry must never be refused, however many bind windows it spans"
        );
    }
}

/// The verdict of [`unstageable_reason`] for every encoding a parameter can have, in
/// both slots one can be given: a native one (what the flat plan gives a non-F32
/// `Param`) and a widened one — the four bytes per element the split plan gives a
/// parameter in the arena, and the weight buffer gives every parameter that is not
/// packed.
///
/// Pure — no device — and fed by hand, so it pins the verdict table and nothing else:
/// it does not go through the recording (`plan_node_slots` answers the flat plan's own
/// slots directly), so what a *real* arena records is pinned where the rule runs —
/// `tests/host_route_operand_fit.rs` (an in-arena `F16` parameter refused, its view
/// refused, an `F32` control computed) and `tests/ceiling_stageable.rs` (a parked `F16`
/// parameter staged and computed, a parked packed `U8` refused).
#[cfg(test)]
mod operand_stageability_tests {
    use super::*;
    use rlx_ir::{DType, GraphExt};

    const NE: usize = 256;

    /// One record entry: `bytes` available there, declared with `dtype`.
    fn stored(bytes: usize, dtype: DType) -> ParamStorage {
        ParamStorage {
            bytes,
            dtype: Some(dtype),
        }
    }

    /// An arena's parameter-storage record over `(offset, bytes)` pairs, declared `F32` —
    /// the encoding is irrelevant to what these tests ask of it.
    fn record(pairs: &[(usize, usize)]) -> HashMap<usize, ParamStorage> {
        pairs
            .iter()
            .map(|&(off, bytes)| (off, stored(bytes, DType::F32)))
            .collect()
    }

    /// `(dtype, does its slot hold four bytes per element, verdict)`.
    const CASES: &[(DType, bool, Option<Unstageable>)] = &[
        // A four-byte element's native slot is already one f32 lane wide, so these
        // three get one row: both slots come out the same size for them.
        (DType::F32, false, None),
        (DType::I32, false, None),
        (DType::U32, false, None),
        // Widened to one lane on upload: stageable exactly where the slot holds one.
        (DType::F16, true, None),
        (DType::F16, false, Some(Unstageable::SlotTooNarrow)),
        (DType::Bool, true, None),
        (DType::Bool, false, Some(Unstageable::SlotTooNarrow)),
        // The upload's raw arms: four bytes read per stored byte / pair.
        (DType::U8, true, Some(Unstageable::StoredRaw)),
        (DType::U8, false, Some(Unstageable::StoredRaw)),
        (DType::I8, false, Some(Unstageable::StoredRaw)),
        (DType::BF16, true, Some(Unstageable::StoredRaw)),
        (DType::BF16, false, Some(Unstageable::StoredRaw)),
        // Widened to one lane on upload, read as the declared two.
        (DType::F64, true, Some(Unstageable::DeclaredWiderThanALane)),
        (DType::F64, false, Some(Unstageable::DeclaredWiderThanALane)),
        (DType::I64, true, Some(Unstageable::DeclaredWiderThanALane)),
        // No upload arm: nothing to stage, in any slot.
        (DType::I16, true, Some(Unstageable::NoUploadArm)),
        (DType::C64, true, Some(Unstageable::NoUploadArm)),
        (DType::C128, false, Some(Unstageable::NoUploadArm)),
    ];

    #[test]
    fn stageability_is_one_f32_lane_per_element() {
        for &(dt, wide_slot, want) in CASES {
            let slot = if wide_slot {
                NE * 4
            } else {
                dt.size_bytes() * NE
            };
            assert_eq!(
                unstageable_reason(dt, NE, slot),
                want,
                "{dt:?} in a {slot} B slot"
            );
        }
    }

    /// A zero-element operand is no read at all — whatever its encoding, and whatever
    /// slot it got. Refusing one would be refusing a case the route computes, which is
    /// why the read is what decides: the zero-byte read returns before the encoding is
    /// consulted, so no dtype and no slot can refuse it.
    #[test]
    fn no_elements_is_no_read() {
        for dt in [DType::F64, DType::U8, DType::F16, DType::C128, DType::I16] {
            assert_eq!(unstageable_reason(dt, 0, 0), None, "{dt:?}");
        }
    }

    /// A shrinking view shares its root's offset, and the bound recorded there is what
    /// the *storage* holds from that offset on — the root's slot — not the view's own
    /// slot. Capping by the view's own slot would refuse a whole-parameter read that
    /// shares the offset, which the route computes correctly.
    ///
    /// The `KvAppend` row is the other half of the same question: the append writes a row
    /// but its output still aliases `cache` (input 0), so a view above one reads the
    /// cache's bytes and has to be bounded by them. `is_pure_view` alone stops the walk
    /// at the append, and the row fails if it does.
    #[test]
    fn a_shrinking_view_does_not_lower_the_parameters_bound() {
        let mut g = Graph::new("param-view-bounds");
        // `F16`, so the encoding an entry records is one a stand-in would not carry.
        let p = g.param("p", rlx_ir::Shape::new(&[8], DType::F16));
        let same = g.narrow_(p, 0, 0, 2); // aliased at the parameter's own offset
        let inside = g.narrow_(p, 0, 2, 1); // ... and two elements further in
        let row = g.input("row", rlx_ir::Shape::new(&[1], DType::F32));
        let kv = g.add_node(
            rlx_ir::Op::KvAppend { axis: 0, pos: 1 },
            vec![p, row],
            rlx_ir::Shape::new(&[2], DType::F32),
        );
        let after = g.narrow_(kv, 0, 1, 1); // reads the appended element, at p + 4 B
        g.set_outputs(vec![same, inside, after]);
        let nothing: HashMap<NodeId, usize> = HashMap::new();
        let offsets = HashMap::from([
            (p, 256usize),
            (same, 256),
            (inside, 264),
            (kv, 256),
            (after, 260),
        ]);
        let slots = HashMap::from([(p, 32usize), (same, 8), (inside, 4), (kv, 8), (after, 4)]);
        let mut bounds = param_storage_bounds(&g, &offsets, &nothing, &slots);
        bounds.sort_by_key(|&(off, _)| off);
        assert_eq!(
            bounds,
            [
                (256, stored(32, DType::F16)),
                (260, stored(28, DType::F16)),
                (264, stored(24, DType::F16)),
            ],
            "the parameter's own slot bounds its storage from each view's offset on, and every \
             entry records the root's declaration (F16) rather than the F32 a stand-in would \
             give"
        );
    }

    /// Two parameters' storages can reach one offset — a view the plan placed past its own
    /// root's slot lands on the next parameter's start. The key is the offset alone, so
    /// the entry keeps the tighter of the two bounds and stops naming an encoding, rather
    /// than stating the encoding of whichever parameter it happened to keep.
    #[test]
    fn an_offset_two_parameters_reach_names_no_encoding() {
        let mut g = Graph::new("two-params-one-offset");
        // The split plan gives an in-arena parameter `ne * 4` bytes, so this `F64`
        // parameter's view at element 4 is planned 32 bytes in — exactly where `b` starts.
        let a = g.param("a", rlx_ir::Shape::new(&[8], DType::F64));
        let view = g.narrow_(a, 0, 4, 2);
        let b = g.param("b", rlx_ir::Shape::new(&[8], DType::F32));
        g.set_outputs(vec![view, b]);
        let nothing: HashMap<NodeId, usize> = HashMap::new();
        let offsets = HashMap::from([(a, 0usize), (view, 32), (b, 32)]);
        let slots = HashMap::from([(a, 32usize), (b, 32)]);
        let mut bounds = param_storage_bounds(&g, &offsets, &nothing, &slots);
        bounds.sort_by_key(|&(off, _)| off);
        assert_eq!(
            bounds,
            [
                (0, stored(32, DType::F64)),
                (
                    32,
                    ParamStorage {
                        bytes: 0,
                        dtype: None
                    }
                ),
            ],
            "the shared offset keeps the tighter bound and names no encoding, since which \
             parameter a read there belongs to is not in the key"
        );
    }

    /// The safe default at the plan-time check: a parameter operand the arena's record
    /// cannot answer for is refused, not read as an activation. `None` where it answers —
    /// and for an operand that is an activation's own bytes, whatever the record says.
    #[test]
    fn an_unrecorded_parameter_is_not_an_activation() {
        let mut g = Graph::new("unrecorded-param");
        let p = g.param("p", rlx_ir::Shape::new(&[8], DType::F32));
        let view = g.narrow_(p, 0, 2, 4);
        let x = g.input("x", rlx_ir::Shape::new(&[8], DType::F32));
        g.set_outputs(vec![view, x]);
        let operands = [(p, 256usize), (view, 264), (x, 512)];

        // An arena that was never recorded: both parameter operands are unanswerable, the
        // activation is not a parameter's storage at all.
        let nothing = HashMap::new();
        assert_eq!(
            first_unrecorded_parameter(&g, &nothing, &operands),
            Some((p, 256))
        );
        // A complete record answers for the parameter and for the view over it.
        let recorded = record(&[(256, 32), (264, 24)]);
        assert_eq!(first_unrecorded_parameter(&g, &recorded, &operands), None);
        // A record that stops at the parameter: the view is a parameter's storage too,
        // and one the record misses is refused rather than assumed to be an activation.
        let partial = record(&[(256, 32)]);
        assert_eq!(
            first_unrecorded_parameter(&g, &partial, &operands),
            Some((view, 264))
        );
    }

    /// The slot the flat plan gives a two-byte `Param` is its native width, while the
    /// activation of the same dtype is widened — the fact the in-arena `F16`/`Bool`
    /// verdict rests on (`node_slot_bytes` in `rlx-compile`). Read through
    /// [`plan_node_slots`], the function the arena records its slots from, so this fails
    /// if that rule changes; what a *real* arena makes of the recording is pinned where
    /// the rule asks it, in `tests/host_route_operand_fit.rs`.
    #[test]
    fn flat_plan_slots_a_two_byte_parameter_native() {
        let mut g = Graph::new("flat-plan-slots");
        let p = g.param("p", rlx_ir::Shape::new(&[NE], DType::F16));
        let x = g.input("x", rlx_ir::Shape::new(&[NE], DType::F16));
        g.set_outputs(vec![p, x]);
        let slots = plan_node_slots(&rlx_compile::memory::plan_memory_f32_uniform(&g, 16));
        assert_eq!(
            (slots[&p], slots[&x]),
            (NE * 2, NE * 4),
            "the flat plan sizes a non-F32 Param native and widens activations"
        );
        assert_eq!(
            unstageable_reason(DType::F16, NE, slots[&p]),
            Some(Unstageable::SlotTooNarrow),
            "an F16 parameter in that slot is what the rule refuses in the act arena"
        );
    }
}
