// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cold upload's read order: one group of requests ahead of the copies they feed.
//!
//! A packed upload is a sequence of windows of one model file, each mapped for the length of
//! the write it feeds and released when that write returns (`Model::with_mapped_ready`). Read one
//! tensor at a time and asked for one tensor at a time, that sequence is the slowest route
//! the flow has: the kernel has no reason to read past the window it was just asked for, so
//! each window waits for its own pages, and the gaps between tensors — alignment padding, the
//! tensors of another role — are never in flight at all. The device sits idle for the whole
//! of it, because the upload is the first thing a compile does and nothing else can start
//! until it lands.
//!
//! The request has to stay inside the upload loop, after the compile. Before the compile
//! there is no plan: which parameters the schedule holds, which of them the device already
//! has from a previous rung and which fused projection each file tensor feeds are all
//! answers the compiled session gives, and a request made on a guess reads the wrong spans —
//! or reads at all on a warm compile that should touch nothing. Asking once for the whole
//! file up front has the same defect for the same reason, and pays for tensors no graph
//! binds.
//!
//! Forward order is a precondition, not a preference. A request only helps if the copy that
//! needs the pages comes after it, so where a request covers a group the walk has to visit the
//! file in ascending offset; the build's own list is in declaration order and is not that.
//! Sorting the plan by file offset also means each file is read front to back, which is the
//! order the filesystem's own readahead and the drive's own prefetch assume.
//!
//! One group ahead, not a hoist. A request made for every group before the first copy would
//! put the whole file in flight and hold the page cache — and the memory accounting — for
//! bytes the upload may not reach for a while. Issuing the next group's request just before
//! this group's copies keeps one group of readahead outstanding: enough to overlap the copy
//! with the next read, never more than a bounded window of the file. On the real files the
//! hoist is worse than no request at all — 762 ms at 32 MiB groups and 426 at 64, against 338
//! for the per-window walk — and a lookahead of sixteen groups is the same thing by another
//! name: it exceeds the synthesis body's entire upload set and degenerates into that hoist.
//!
//! A group never spans a hole. Two extents that are close together belong to one request;
//! two separated by a stretch the upload does not read — the vocabulary table, whose rows sit
//! between the body's tensors in some of these files and which no packed parameter reads —
//! must not, because the request would read those bytes into the page cache and count them
//! against nothing this process asked for. `SLACK_BYTES` is the largest gap a group
//! bridges: wide enough to absorb alignment padding and a small tensor between two large
//! ones, far too narrow to reach across a table.
//!
//! The window and the copy stay per tensor. Grouping the *requests* does not group the
//! *maps*: each extent is still handed to the device from its own window, released when the
//! write returns, so the resident cost stays the tensor being written rather than the group
//! behind it. A window over the whole group would fault the bytes between tensors into this
//! process and keep them there for the length of the group's copies, which is the cost the
//! per-tensor window exists to avoid.
//!
//! This is a macOS route. `F_RDADVISE` reads the span it is given; the Linux arm is clamped
//! by the kernel to its own readahead window and the Windows arm has no call at all, so on
//! those platforms a grouped request would cover one window of the group and the rest would
//! fault as before. There the walk is the one the flow had before this module existed — the
//! build's list reversed, each window asking for itself — because forward order without a
//! request that covers a group is a loss rather than a gain (the harness's `Forward` arm
//! prices it; see below).
//!
//! # What it is worth
//!
//! `tests/cold_read.rs` walks the extents a compile really reads, three arms in one process at
//! zero residency before every arm, five paired rounds with the arms' order rotated, median of
//! the rounds (M3 Pro, macOS 26.5/arm64, one-minute load 2.6, 2026-11-16). "Before" is the stack
//! order the upload had, each window asking for itself; "after" is this module's walk.
//!
//! | subject | extents | bytes | before | after |
//! |---|---|---|---|---|
//! | recognition body | 197 | 1050.0 MiB | 278.1 ms (3.96 GB/s) | **225.5 ms (4.88 GB/s)** |
//! | synthesis body | 197 | 811.5 MiB | 226.9 ms (3.75 GB/s) | **172.2 ms (4.94 GB/s)** |
//! | recognition encoder | 144 | 306.0 MiB | 107.1 ms (3.00 GB/s) | **65.6 ms (4.89 GB/s)** |
//! | synthesis sound front | 66 | 56.6 MiB | 27.0 ms (2.20 GB/s) | **12.1 ms (4.91 GB/s)** |
//!
//! The same bytes through the real compile — the synthesis prefill's 811.9 MiB, the device write
//! included — upload in 186–213 ms (median of three takes, three runs), where a request per
//! window left the 242–267 ms `loader::request_readahead` records.
//!
//! The order is not the gain on its own, and the harness prints it as its own arm rather than
//! folding it in: the same extents in file order with a request per window read 301.1 ms against
//! the stack order's 278.1 on the recognition body and 232.2 against 226.9 on the synthesis one,
//! so on the larger file the order alone is 8 % against. It is kept because the request is
//! worthless without it — a group asked for ahead of the copies pays only when the copies follow
//! it — while the forward walk under the request buys 52.6 and 54.7 ms on the two bodies.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::loader::{Model, covers_a_group};

/// How many bytes of one file one request covers: about sixteen tensor extents of these models —
/// 15 groups over each body's upload set, 24 over the recognition encoder, 5 over the sound
/// front — and one extent larger than a group (the recognition body's 243.4 MiB head table) is a
/// group of its own, since it cannot be split. The system's advisory request returns in ~3 ms at
/// 128 MiB and blocks for its own duration at 192 MiB and above, so this sits with headroom
/// below that boundary rather than on it — the boundary is filesystem-dispatched, undocumented,
/// and can move with an OS update.
pub const GROUP_BYTES: u64 = 64 << 20;

/// The largest gap between two extents one request bridges. The upload set is 86 runs on each
/// body file at zero slack and 58 at a page-sized 4 KiB; 16 KiB collapses it to 2 for 0.46 MiB
/// of extra bytes, and it cannot reach across a hole as large as an embedding table.
const SLACK_BYTES: u64 = 16 << 10;

/// One tensor's bytes an upload reads, with whatever the caller needs to know when they
/// arrive.
pub struct Extent<T> {
    pub model: Arc<Model>,
    pub key: String,
    pub start: u64,
    pub len: u64,
    pub tag: T,
}

/// One request the walk makes ahead of the copies it feeds: a span of one file and the
/// extents it covers, as indices into the list it was built from, in file order. A group
/// never spans two files.
pub struct Group {
    pub start: u64,
    pub len: u64,
    pub extents: Vec<usize>,
}

/// Group the `(file, start, len)` spans an upload reads into the requests it makes one group
/// ahead: the spans are sorted by file and then by offset, and consecutive extents merge while
/// the gap between them is at most `SLACK_BYTES` and the group's span stays within
/// [`GROUP_BYTES`]. A gap wider than the slack starts a new group, so a span never reaches
/// across a stretch of the file the upload does not read.
pub fn groups(spans: &[(usize, u64, u64)]) -> Vec<Group> {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by_key(|&i| (spans[i].0, spans[i].1));
    let mut out: Vec<Group> = Vec::new();
    let mut file: Option<usize> = None;
    for i in order {
        let (f, start, len) = spans[i];
        let end = start.saturating_add(len);
        if file == Some(f) {
            // Sorted by file first, so the group this extent may join is the last one pushed.
            let group = out.last_mut().expect("a group for this file");
            let group_end = group.start.saturating_add(group.len);
            let gap = start.saturating_sub(group_end);
            let span = end.max(group_end).saturating_sub(group.start);
            if gap <= SLACK_BYTES && span <= GROUP_BYTES {
                group.len = span;
                group.extents.push(i);
                continue;
            }
        }
        file = Some(f);
        out.push(Group {
            start,
            len,
            extents: vec![i],
        });
    }
    out
}

/// Walk `extents` in file order, one group of requests ahead of the copies they feed: request
/// the first group — without it the first group's copies would fault a page at a time — then,
/// before copying each group, request the next one, and hand every extent of this group to
/// `copy` with the bytes of its own window.
///
/// Where this platform's advice cannot cover a group there is nothing to be gained from
/// grouping and the walk is the one the flow had before: `extents` reversed, each window
/// asking the kernel for itself.
pub fn read_extents<T>(
    extents: &[Extent<T>],
    mut copy: impl FnMut(&Extent<T>, &[u8]) -> Result<()>,
) -> Result<()> {
    if !covers_a_group() {
        for extent in extents.iter().rev() {
            extent
                .model
                .with_mapped(&extent.key, |bytes| copy(extent, bytes))??;
        }
        return Ok(());
    }
    // One file index per path, in order of first appearance: a request is per file, and a
    // compile's reads are not assumed to be one file's.
    let mut files: Vec<&Path> = Vec::new();
    let mut spans: Vec<(usize, u64, u64)> = Vec::with_capacity(extents.len());
    for extent in extents {
        let path = extent.model.path.as_path();
        let file = match files.iter().position(|seen| *seen == path) {
            Some(file) => file,
            None => {
                files.push(path);
                files.len() - 1
            }
        };
        spans.push((file, extent.start, extent.len));
    }
    let plan = groups(&spans);
    if let Some(first) = plan.first() {
        let model = &extents[first.extents[0]].model;
        model.prefetch(first.start, first.len)?;
    }
    for (i, group) in plan.iter().enumerate() {
        if let Some(next) = plan.get(i + 1) {
            let model = &extents[next.extents[0]].model;
            model.prefetch(next.start, next.len)?;
        }
        for &index in &group.extents {
            let extent = &extents[index];
            // The window's failure and the copy's are both this walk's.
            extent
                .model
                .with_mapped_ready(&extent.key, |bytes| copy(extent, bytes))??;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span list as `read_extents` builds it: one file, offsets and lengths.
    fn spans(list: &[(u64, u64)]) -> Vec<(usize, u64, u64)> {
        list.iter().map(|&(start, len)| (0, start, len)).collect()
    }

    /// Every input index appears exactly once, in file order, and no group's span exceeds
    /// [`GROUP_BYTES`] — except one that is a single extent larger than that, which cannot
    /// be split.
    fn assert_whole(groups: &[Group], count: usize) {
        let mut seen: Vec<usize> = groups.iter().flat_map(|g| g.extents.clone()).collect();
        assert_eq!(seen.len(), count, "every extent appears once");
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..count).collect::<Vec<_>>(),
            "each index exactly once"
        );
        for group in groups {
            if group.extents.len() == 1 {
                continue;
            }
            assert!(
                group.len <= GROUP_BYTES,
                "a group's span stays within GROUP_BYTES: {} bytes",
                group.len
            );
        }
    }

    #[test]
    fn groups_are_walked_forward_by_file_offset() {
        let scrambled = spans(&[
            (40 << 20, 4 << 20),
            (8 << 20, 4 << 20),
            (0, 4 << 20),
            (24 << 20, 4 << 20),
        ]);
        let out = groups(&scrambled);
        let order: Vec<u64> = out
            .iter()
            .flat_map(|g| g.extents.iter().map(|&i| scrambled[i].1))
            .collect();
        assert_eq!(order, vec![0, 8 << 20, 24 << 20, 40 << 20]);
        for pair in out.windows(2) {
            assert!(
                pair[0].start < pair[1].start,
                "groups are in ascending offset"
            );
        }
        assert_whole(&out, scrambled.len());
    }

    #[test]
    fn a_slack_sized_gap_is_bridged_and_one_byte_more_is_not() {
        let close = spans(&[(0, 4 << 20), ((4 << 20) + SLACK_BYTES, 4 << 20)]);
        let out = groups(&close);
        assert_eq!(out.len(), 1, "a gap of exactly SLACK_BYTES is one group");
        assert_eq!(
            out[0].len,
            (8 << 20) + SLACK_BYTES,
            "the span covers the gap"
        );
        assert_whole(&out, close.len());

        let apart = spans(&[(0, 4 << 20), ((4 << 20) + SLACK_BYTES + 1, 4 << 20)]);
        let out = groups(&apart);
        assert_eq!(out.len(), 2, "a byte past the slack is two groups");
        assert_eq!(out[0].len, 4 << 20);
        assert_eq!(out[1].start, (4 << 20) + SLACK_BYTES + 1);
        assert_whole(&out, apart.len());
    }

    #[test]
    fn a_group_never_spans_a_hole_the_upload_does_not_read() {
        // A vocabulary table between two tensors: the gap is megabytes wide.
        let hole = 16 << 20;
        let list = spans(&[(0, 4 << 20), ((4 << 20) + hole, 4 << 20)]);
        let out = groups(&list);
        assert_eq!(out.len(), 2, "a multi-megabyte gap is two groups");
        assert_eq!(out[0].extents, vec![0]);
        assert_eq!(out[1].extents, vec![1]);
        assert!(
            out[0].start + out[0].len <= (4 << 20),
            "no group reaches the hole"
        );
        assert!(out[1].start >= (4 << 20) + hole, "nor starts inside it");
        assert_whole(&out, list.len());
    }

    #[test]
    fn a_group_never_spans_two_files() {
        // File 0's two extents sit back to back and file 1's starts at its own offset zero, so
        // nothing but the file boundary separates them.
        let list = vec![(0, 0, 4 << 20), (1, 0, 4 << 20), (0, 4 << 20, 4 << 20)];
        let out = groups(&list);
        assert_eq!(out.len(), 2, "a group stays inside one file");
        for group in &out {
            let files: Vec<usize> = group.extents.iter().map(|&i| list[i].0).collect();
            assert!(
                files.windows(2).all(|pair| pair[0] == pair[1]),
                "one group, one file: {files:?}"
            );
        }
        assert_eq!(out[0].extents, vec![0, 2], "file 0's extents merge");
        assert_eq!(out[1].extents, vec![1], "file 1's is its own");
    }

    #[test]
    fn a_run_wider_than_a_group_is_split_and_an_oversized_extent_stands_alone() {
        // Eight 4 MiB extents back to back: 32 MiB, inside one group. Twenty-four of them:
        // 96 MiB, which must split.
        let run: Vec<(u64, u64)> = (0..24).map(|i| (i * (4 << 20), 4 << 20)).collect();
        let out = groups(&spans(&run));
        assert!(out.len() > 1, "96 MiB does not fit one group");
        assert_whole(&out, run.len());

        let oversized = spans(&[
            (0, GROUP_BYTES + (4 << 20)),
            (GROUP_BYTES + (4 << 20), 4 << 20),
        ]);
        let out = groups(&oversized);
        assert_eq!(out.len(), 2, "an extent past GROUP_BYTES is its own group");
        assert_eq!(out[0].extents, vec![0]);
        assert_eq!(out[0].len, GROUP_BYTES + (4 << 20));
        assert_eq!(out[1].extents, vec![1]);
        assert_whole(&out, oversized.len());
    }
}
