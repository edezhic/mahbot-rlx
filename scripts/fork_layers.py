#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The pieces this fork's manifest tooling shares: the walk, the atomic writer and the
temp name it writes under, plus the version arithmetic the naming layer and the publish
gate both ask.

Both layers rewrite manifests in place. A manifest truncated by an interrupted walk
is a workspace that no longer parses, so the bytes go to a temporary beside the
target, are flushed to the platter, and are renamed into place — a reader sees
either the old file or the new one, never half of one, and a crash cannot leave the
new bytes in the page cache only.

The temp name is the record writer's own rule, `<target>.tmp-<pid>` (see
`measurements::temporary` in `crates/mahbot-inference`), which the tree digest's
leftover filter recognizes and leaves out of the tree-sha256 — so an interrupted
walk leaves nothing the digest sees, though nothing sweeps a manifest's temporary
the way a record run sweeps its own. A walk that raises removes its own on the way
out.
"""

import os
import re

# Not walked: build output, the vendored MLX tree (a submodule upstream owns), and
# VCS metadata.
SKIP_DIRS = {".git", "target", "vendor", "node_modules"}


def manifests(root):
    """Every `Cargo.toml` under `root`, skipping `SKIP_DIRS`, in a stable order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if "Cargo.toml" in filenames:
            yield os.path.join(dirpath, "Cargo.toml")


def read(path):
    """Read a manifest as UTF-8.

    Most manifests carry non-ASCII (the `──` section rules), and the encoding the
    locale happens to name is not something a gate should depend on.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_atomic(path, text):
    """Replace `path`'s contents in one step, via a pid-marked temporary beside it."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # A half-written temporary is not the target and nothing sweeps this name
        # outside a record run, so remove it here rather than leave it for `git add`.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def numbers(version):
    """The three numeric components of `X.Y.Z`, or None if `version` is not that."""
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\.(\d+)\s*", version or "")
    return tuple(int(part) for part in match.groups()) if match else None


def admits(requirement, version):
    """Whether a caret `requirement` provably admits `version`; None when it cannot be read.

    A bare version is cargo's caret requirement, and the only form the naming
    layer writes. Anything else — a comparator, a wildcard, a comma list — is not
    read at all, and comes back as None rather than False: a gate has to fail on
    what it cannot check, and to say which of the two happened.
    """
    match = re.fullmatch(r"\s*\^?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*", requirement)
    here = numbers(version)
    if match is None or here is None:
        return None
    groups = match.groups()
    named = [int(part or 0) for part in groups]
    # Cargo's compatibility boundary is the leftmost non-zero component the
    # requirement names, or the last one it names when they are all zero — so
    # `^0` stays inside the major version and `^0.0` inside the minor.
    count = sum(part is not None for part in groups)
    at = next((i for i, part in enumerate(named) if part), count - 1)
    high = tuple(named[:at] + [named[at] + 1] + [0] * (2 - at))
    return tuple(named) <= here < high

