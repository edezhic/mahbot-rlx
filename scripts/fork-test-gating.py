#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Gate every integration test target behind a crate's `backend-tests` feature.

This fork keeps a bare `cargo test` a fast smoke gate: a crate that declares
`backend-tests = []` in `[features]` also gives every file in `tests/` a
`[[test]]` entry whose `required-features` includes `backend-tests`, so the
target is only built when a gate asks for it. Upstream adds test files without
those entries, and a `tests/*.rs` with no `[[test]]` entry runs in the bare
default build — the opposite of the intent.

Re-run after every sync. Idempotent: an entry that is already there keeps its
own `required-features` and gains `backend-tests` appended, never replaced; a
test file with no entry gets one appended at the end of the manifest, in the
file's own block spacing.

Crates without a `backend-tests` feature are left alone: they have no gate to
attach to. Usage: `python3 scripts/fork-test-gating.py [--check]`.
"""

import os
import re
import sys

import fork_layers as layers

GATE = "backend-tests"
RF = re.compile(r"^([ \t]*)required-features[ \t]*=[ \t]*\[(.*?)\]", re.M | re.S)


def test_files(manifest_dir):
    """Integration-test target names: `tests/x.rs` → `x`, `tests/x/main.rs` → `x`."""
    tests = os.path.join(manifest_dir, "tests")
    if not os.path.isdir(tests):
        return []
    names = []
    for entry in sorted(os.listdir(tests)):
        path = os.path.join(tests, entry)
        if entry.endswith(".rs"):
            names.append(entry[:-3])
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "main.rs")):
            names.append(entry)
    return names


def blocks(text):
    """[(start, end, name)] for every `[[test]]` table, end exclusive."""
    out = []
    pos = 0
    header = re.compile(r"^\[\[test\]\][ \t]*$", re.M)
    any_header = re.compile(r"^\[", re.M)
    while True:
        m = header.search(text, pos)
        if not m:
            return out
        nxt = any_header.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        body = text[m.start():end]
        name = re.search(r'^[ \t]*name[ \t]*=[ \t]*"([^"]+)"', body, re.M)
        out.append((m.start(), end, name.group(1) if name else None))
        pos = end


def add_gate(body):
    """Append `backend-tests` to a block's `required-features`, or add the key."""
    m = RF.search(body)
    if m:
        items = m.group(2)
        if f'"{GATE}"' in items:
            return body
        # A trailing comma is legal TOML and `inner, "gate"` would then read
        # `"a",, "gate"`.
        inner = items.strip().rstrip(",").rstrip()
        new = f'{inner}, "{GATE}"' if inner else f'"{GATE}"'
        return body[:m.start(2)] + new + body[m.end(2):]
    name = re.search(r'^([ \t]*)name[ \t]*=[ \t]*"[^"]+"[ \t]*$', body, re.M)
    indent = name.group(1) if name else ""
    line = f'{indent}required-features = ["{GATE}"]'
    if name:
        return body[:name.end()] + "\n" + line + body[name.end():]
    return body.rstrip("\n") + "\n" + line + "\n"


def separator(text):
    """The blank line (or its absence) the file puts before its last `[[test]]`."""
    found = [m for m in re.finditer(r"^\[\[test\]\][ \t]*$", text, re.M)]
    if not found:
        return "\n"
    return "\n" if text[:found[-1].start()].endswith("\n\n") else ""


def gate_manifest(path):
    text = layers.read(path)
    if not re.search(r"^" + GATE + r"[ \t]*=[ \t]*\[", text, re.M):
        return text
    names = test_files(os.path.dirname(path))
    if not names:
        return text
    present = {n for _, _, n in blocks(text) if n}
    for start, end, name in reversed(blocks(text)):
        if name in names:
            body = text[start:end]
            new = add_gate(body)
            if new != body:
                text = text[:start] + new + text[end:]
    missing = [n for n in names if n not in present]
    if missing:
        sep = separator(text)
        add = "".join(f'{sep}[[test]]\nname = "{n}"\nrequired-features = ["{GATE}"]\n'
                      for n in missing)
        text = text.rstrip("\n") + "\n" + add
    return text


def main(argv):
    check = "--check" in argv[1:]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    changed = 0
    for path in layers.manifests(root):
        text = layers.read(path)
        new = gate_manifest(path)
        if new != text:
            changed += 1
            shown = os.path.relpath(path, root)
            if check:
                print(f"would gate: {shown}")
            else:
                layers.write_atomic(path, new)
                print(f"gated: {shown}")
    if check and changed:
        print(f"{changed} manifest(s) with ungated integration tests", file=sys.stderr)
        return 1
    if not changed:
        print("every integration test target is behind its crate's backend-tests gate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
