// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a `Step::BufferCopy` runs, as a pure function of the copy's two
//! offsets, its byte count, the arena's size and the adapter's storage-binding
//! limit — plus the conditions that make a device copy impossible.
//!
//! Split out of the run loop so the decision is testable without a device.

/// Granularity a storage-buffer binding window is anchored to.
const WINDOW_ALIGN: u64 = 256;

/// The route a `Step::BufferCopy` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopyRoute {
    /// Zero bytes: nothing to move, on either route.
    Skip,
    /// One `copy_kernel` dispatch over the window `[base, base + size)` — a
    /// dispatch rather than a buffer-to-buffer transfer, because wgpu forbids
    /// `copy_buffer_to_buffer` between two regions of one buffer and both
    /// operands are arena offsets.
    Device { base: u64, size: u64 },
    /// A host round-trip: a surviving condition forbids a device copy, or no
    /// single window can cover the range.
    Host,
}

impl CopyRoute {
    /// Whether the run path may fold this copy into the enclosing batch of device
    /// work — the predicate form of [`Self::join_window`], which is the rule.
    pub(crate) fn joins_batch(self) -> bool {
        self.join_window().is_some()
    }

    /// The window a joining copy's dispatch covers, or `None` when the copy keeps
    /// its own submission. This is the whole of the join rule: only the device route
    /// joins, because `Host` reads and writes through the host (so it keeps its own
    /// submission, its own wait(s) and its own readback, with the device drained
    /// before it goes through the host) and `Skip` moves nothing.
    pub(crate) fn join_window(self) -> Option<(u64, u64)> {
        match self {
            CopyRoute::Device { base, size } => Some((base, size)),
            CopyRoute::Host | CopyRoute::Skip => None,
        }
    }
}

/// Inputs to [`buffer_copy_route`] — values rather than process state, so the
/// decision is deterministic and its tests order-independent.
#[derive(Debug)]
pub(crate) struct CopyRouteInputs {
    /// Absolute arena byte offsets of the source and destination.
    pub src: u64,
    pub dst: u64,
    pub nbytes: u64,
    /// [`crate::buffer::Arena::size`].
    pub arena_size: u64,
    /// The adapter's `max_storage_buffer_binding_size`.
    pub max_binding: u64,
    /// [`crate::buffer::Arena::shard_size`] when the arena is striped, else 0.
    pub shard_size: u64,
    /// Either operand lives in the separate weight buffer.
    pub weight_operand: bool,
    /// The host tensor cache holds uploads that have not reached the device.
    pub deferred_writes: bool,
    /// `RLX_WGPU_HOST_BUFFER_COPY=1`.
    pub force_host: bool,
}

/// The one storage-binding window that covers the byte range `[lo, hi)`, or
/// `None` when this adapter cannot bind one. `hi` must exceed `lo` — every
/// caller passes a non-empty range — so the rounded size is at least one block.
///
/// `base` is `lo` floored to the window granularity and the size is measured
/// **from `base`**: measuring from `lo` would leave the window ending short of
/// `hi` by at most the flooring, since the round-up to the window granularity
/// makes up part or all of the shortfall — and an operand at the far end then
/// falls outside the binding, where the shader reads or writes nothing,
/// silently. (Measured: a resident-training copy of `b` from 240 to 1216 got
/// `base` 0 with a 1024-byte window, so its destination at 1216..1232 was
/// outside it and the parameter never updated.)
///
/// At the arena's far end the size is *shrunk* to what is left rather than
/// sliding `base` down (`hi` is inside the arena by construction, and sliding
/// would drop it). When even then the window cannot cover `hi`, or would exceed
/// `max_binding`, there is no window: `buffer_copy_route` hosts the copy, and
/// the sibling path in `backend::dispatch` refuses.
pub(crate) fn copy_window(
    lo: u64,
    hi: u64,
    arena_size: u64,
    max_binding: u64,
) -> Option<(u64, u64)> {
    let base = (lo / WINDOW_ALIGN) * WINDOW_ALIGN;
    let mut size = hi.saturating_sub(base).div_ceil(WINDOW_ALIGN) * WINDOW_ALIGN;
    if base.saturating_add(size) > arena_size {
        size = arena_size.saturating_sub(base);
    }
    if size > max_binding || base.saturating_add(size) < hi {
        return None;
    }
    Some((base, size))
}

/// The byte span a copy touches: from the lower of the two offsets to the end
/// of whichever operand ends higher, since the window has to cover both.
pub(crate) fn copy_span(src: u64, dst: u64, nbytes: u64) -> (u64, u64) {
    (src.min(dst), src.max(dst).saturating_add(nbytes))
}

/// A copy runs on the device unless one of the conditions below says it cannot.
///
/// Two of the conditions are redundant by construction — `cross_shard` can only
/// hold when the arena is sharded, and a weight operand already fails the span
/// test through its forced `hi` — and are kept anyway, each for the failure it
/// was added for.
pub(crate) fn buffer_copy_route(i: &CopyRouteInputs) -> CopyRoute {
    if i.nbytes == 0 {
        return CopyRoute::Skip;
    }
    // A weight-tagged operand lives in the separate weight buffer, which the
    // copy kernel cannot reach through an arena binding: force the span over
    // the limit so this copy takes the host route.
    let (lo, hi) = if i.weight_operand {
        (0, i.max_binding.saturating_add(1))
    } else {
        copy_span(i.src, i.dst, i.nbytes)
    };
    // Same for sharded arenas when src and dst live on different stripes.
    let cross_shard = i.shard_size != 0 && !i.weight_operand && {
        let s = i.shard_size;
        let src_hi = i.src.saturating_add(i.nbytes.saturating_sub(1));
        let dst_hi = i.dst.saturating_add(i.nbytes.saturating_sub(1));
        i.src / s != src_hi / s || i.dst / s != dst_hi / s || i.src / s != i.dst / s
    };
    // A sharded arena hosts: GPU copy kernels corrupted Kitten NSF under
    // virtual sharding (NVIDIA 2 GiB bind), which is the failure these two
    // conditions were added for.
    if hi.saturating_sub(lo) > i.max_binding
        || cross_shard
        || i.shard_size != 0
        || i.weight_operand
        || i.deferred_writes
        || i.force_host
    {
        return CopyRoute::Host;
    }
    // Otherwise the copy runs on the device — unless no window can cover the
    // requested range, in which case it is hosted rather than clipped.
    match copy_window(lo, hi, i.arena_size, i.max_binding) {
        Some((base, size)) => CopyRoute::Device { base, size },
        None => CopyRoute::Host,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Big enough that no window is limited by the arena, and a binding limit
    /// that leaves room for `WINDOW_ALIGN` slack.
    const ARENA: u64 = 1 << 20;
    const BIND: u64 = 1 << 16;

    fn inputs(src: u64, dst: u64, nbytes: u64) -> CopyRouteInputs {
        CopyRouteInputs {
            src,
            dst,
            nbytes,
            arena_size: ARENA,
            max_binding: BIND,
            shard_size: 0,
            weight_operand: false,
            deferred_writes: false,
            force_host: false,
        }
    }

    fn device_window(i: &CopyRouteInputs) -> (u64, u64) {
        match buffer_copy_route(i) {
            CopyRoute::Device { base, size } => (base, size),
            other => panic!("expected a device copy, got {other:?} for {i:?}"),
        }
    }

    /// A copy no condition forbids runs on the device, over a window that
    /// covers BOTH operands — including when neither offset is aligned to the
    /// window granularity.
    #[test]
    fn unconstrained_copy_runs_on_the_device_and_its_window_covers_both_operands() {
        for (src, dst, nbytes) in [
            (0, 4096, 16),
            (8192, 256, 4096),
            (200, 4200, 1024),
            (BIND - 300, 8, 256),
        ] {
            let i = inputs(src, dst, nbytes);
            let (base, size) = device_window(&i);
            assert!(base <= src && src + nbytes <= base + size, "{i:?} window");
            assert!(base <= dst && dst + nbytes <= base + size, "{i:?} window");
            assert!(size <= BIND);
        }
    }

    /// A zero-byte copy moves nothing under every condition, including the ones
    /// that would otherwise host it.
    #[test]
    fn a_zero_byte_copy_is_skipped_under_every_condition() {
        for (src, dst) in [(0, 0), (0, 32896), (32896, 0), (ARENA - 4, 7)] {
            let i = inputs(src, dst, 0);
            assert_eq!(buffer_copy_route(&i), CopyRoute::Skip, "{i:?}");
            let forced = CopyRouteInputs {
                weight_operand: true,
                deferred_writes: true,
                force_host: true,
                shard_size: 4096,
                ..inputs(src, dst, 0)
            };
            assert_eq!(buffer_copy_route(&forced), CopyRoute::Skip, "{forced:?}");
        }
    }

    /// A range that ends in the arena's final partial block still gets a window
    /// that covers it — the arena's size is only 16-byte aligned, not window
    /// aligned, so the size has to shrink to keep `hi` inside the binding.
    #[test]
    fn a_range_ending_in_the_arena_tail_is_covered_not_clipped() {
        // 16-byte aligned sizes, as `Arena::size` is, none of them a whole
        // number of `WINDOW_ALIGN` blocks.
        for arena_size in [ARENA + 16, ARENA - 16, 272, 528] {
            let nbytes = 64;
            let src = arena_size - nbytes; // last block of the arena
            let mut i = inputs(src, src - 64, nbytes);
            i.arena_size = arena_size;
            let (base, size) = device_window(&i);
            assert!(base <= i.dst, "{i:?}");
            assert!(
                base + size >= src + nbytes,
                "window [{base}, {}) clips the source tail of {i:?}",
                base + size
            );
            assert!(base + size <= arena_size, "window past the arena: {i:?}");
        }
    }

    /// A span that fits `max_binding` from `lo` but not once `lo` is floored to
    /// the window granularity has no window at all, and is hosted rather than
    /// clipped — the defect that shipped as a short window.
    #[test]
    fn an_uncoverable_window_is_hosted_rather_than_clipped() {
        let nbytes = 1000;
        // lo sits 200 bytes into the window, so the span is exactly
        // `max_binding` while the window would have to be 256 bytes longer.
        let i = inputs(200 + BIND - nbytes, 200, nbytes);
        let (lo, hi) = copy_span(i.src, i.dst, i.nbytes);
        assert_eq!(
            hi - lo,
            BIND,
            "the span itself fits the binding limit exactly"
        );
        assert_eq!(copy_window(lo, hi, ARENA, BIND), None);
        assert_eq!(buffer_copy_route(&i), CopyRoute::Host);
    }

    /// Every surviving condition hosts exactly the copy it hosted before; the
    /// pair that is redundant by construction still hosts.
    #[test]
    fn each_hosting_condition_still_hosts() {
        let fits = inputs(0, 4096, 64);
        assert!(matches!(buffer_copy_route(&fits), CopyRoute::Device { .. }));

        let over_span = inputs(0, 0, BIND + 4);
        assert_eq!(buffer_copy_route(&over_span), CopyRoute::Host);

        let sharded = CopyRouteInputs {
            shard_size: 1 << 20,
            ..inputs(0, 4096, 64)
        };
        assert_eq!(buffer_copy_route(&sharded), CopyRoute::Host);

        // Different stripes within a sharded arena — subsumed by the sharded
        // test, kept for the failure it was added for.
        let cross_shard = CopyRouteInputs {
            shard_size: 4096,
            ..inputs(0, 8192, 64)
        };
        assert_eq!(buffer_copy_route(&cross_shard), CopyRoute::Host);

        // The weight case hosts through its forced span alone: `hi` is pushed
        // one byte past the binding limit, so dropping the weight terms would
        // not change the answer — which is why they are kept rather than tidied.
        let weight = CopyRouteInputs {
            weight_operand: true,
            ..inputs(0, 4096, 64)
        };
        assert_eq!(buffer_copy_route(&weight), CopyRoute::Host);

        let deferred = CopyRouteInputs {
            deferred_writes: true,
            ..inputs(0, 4096, 64)
        };
        assert_eq!(buffer_copy_route(&deferred), CopyRoute::Host);

        let forced = CopyRouteInputs {
            force_host: true,
            ..inputs(0, 4096, 64)
        };
        assert_eq!(buffer_copy_route(&forced), CopyRoute::Host);
    }

    /// The batch boundary's join rule — the whole of it: only the device route
    /// rides the enclosing batch, and it rides it with the window it picked; every
    /// answer that can read or write through the host keeps its own submission,
    /// its own wait and its own readback, and the zero-byte skip keeps the
    /// boundary treatment it has.
    ///
    /// This is [`CopyRoute::join_window`] on the same [`buffer_copy_route`] answer
    /// the run path's host arm routes by, so it is evaluated here without a device
    /// and cannot disagree with that arm at run time. Which conditions answer
    /// `Host` — and so decline to join — is the subject of the sibling tests
    /// above; what is pinned here is what the boundary does with each answer it
    /// can get.
    #[test]
    fn only_the_device_route_joins_the_enclosing_batch() {
        // The pin's own row-write geometry (`tests/buffer_copy_batch_join.rs`), as
        // this decision sees it: a 256-byte row (`W = 64`, f32) written far below
        // the activation it reads, in that pin's unsharded arena, on an adapter
        // whose binding limit covers the span. Device route, so it joins.
        let row_write = CopyRouteInputs {
            max_binding: 4 << 30,
            ..inputs(262_400, 130_560, 256)
        };
        let CopyRoute::Device { base, size } = buffer_copy_route(&row_write) else {
            panic!("the pin's row write must take the device route");
        };
        // ... with a window that covers the byte range, so both operands are inside
        // the one binding the dispatch reads and writes through.
        assert!(base <= 130_560 && 262_400 + 256 <= base + size);
        assert_eq!(
            buffer_copy_route(&row_write).join_window(),
            Some((base, size))
        );

        // Every answer that can read or write through the host declines, and the
        // one that matters is the pending deferred write: the host route is the one
        // that reads that mirror, so a copy that joined with one outstanding would
        // run device work over a mirror that still has to reach the device. Which
        // conditions answer `Host` at all is the sibling tests' subject; that no
        // `Host` or `Skip` answer carries a window is `join_window`'s own `match`.
        let deferred = CopyRouteInputs {
            deferred_writes: true,
            ..inputs(0, 4096, 64)
        };
        assert!(buffer_copy_route(&deferred).join_window().is_none());
    }
}
