#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Apply this fork's package-name layer to every manifest in the tree.

Twenty-one crates are published from here under `mahbot-` names at the fork
version; every other crate keeps upstream's identity. The layer lives in the
manifests, not in the source: a dependency key keeps upstream's name, the
`[workspace.dependencies]` entry names the real package, and each renamed crate
pins `[lib] name` to the module its sources already use (see
docs/fork-publishing.md).

Upstream does not know about any of it, so every sync re-applies it here rather
than through a hand edit per manifest:

* the twenty-one crates get the fork's `[package] name` and version, and
  `mahbot-inference` — this fork's own subject, not a renamed upstream crate —
  gets the version alone;
* a dependency that names one of them by `path` gains `package = "mahbot-…"`,
  which cargo needs to resolve the edge at all;
* every `version` pin that is already there is made to name the version the
  dependency actually carries — the fork's for a renamed crate, the workspace's
  for the rest — and the workspace registry (`[workspace.dependencies]`) always
  pins. A path-only entry elsewhere stays path-only: cargo strips it from the
  published archive, which is how a dev-dep back-edge avoids a publish cycle.

Idempotent: running it twice changes nothing. Usage: `python3
scripts/fork-rename.py [--check | --print-version]`; `--check` exits 1 when a
manifest would change (for a gate) and prints what it would rewrite;
`--print-version` prints [`FORK_VERSION`] for the release scripts.
"""

import os
import re
import sys

import fork_layers as layers

# The fork's release version. The release process owns this number; a bump
# happens there, not here.
FORK_VERSION = "0.2.18-mahbot.3"

# Directory → package name, for the crates this fork publishes under its own
# name. Everything else in the tree keeps upstream's name and version.
RENAMED = {
    "crates/core/rlx-autodiff": "mahbot-rlx-autodiff",
    "crates/core/rlx-compile": "mahbot-rlx-compile",
    "crates/core/rlx-driver": "mahbot-rlx-driver",
    "crates/core/rlx-flow": "mahbot-rlx-flow",
    "crates/core/rlx-fusion": "mahbot-rlx-fusion",
    "crates/core/rlx-ir": "mahbot-rlx-ir",
    "crates/core/rlx-macros": "mahbot-rlx-macros",
    "crates/core/rlx-opt": "mahbot-rlx-opt",
    "crates/core/rlx-runtime": "mahbot-rlx-runtime",
    "crates/core/rlx-unfuse": "mahbot-rlx-unfuse",
    "crates/backends/rlx-cpu": "mahbot-rlx-cpu",
    "crates/backends/rlx-gpu-dispatch": "mahbot-rlx-gpu-dispatch",
    "crates/backends/rlx-gpu-host": "mahbot-rlx-gpu-host",
    "crates/backends/rlx-wgpu": "mahbot-rlx-wgpu",
    "crates/io/rlx-dduf": "mahbot-rlx-dduf",
    "crates/io/rlx-gguf": "mahbot-rlx-gguf",
    "crates/io/rlx-mlx-io": "mahbot-rlx-mlx-io",
    "crates/io/rlx-nemo": "mahbot-rlx-nemo",
    "crates/io/rlx-pkg": "mahbot-rlx-pkg",
    "crates/io/rlx-torch-ckpt": "mahbot-rlx-torch-ckpt",
    "crates/tooling/rlxsl": "mahbot-rlxsl",
}

# Crates that already carry the fork's own name and still take its version. This
# fork's own subject is not a renamed upstream crate, but restating the release
# number in its manifest by hand makes it a second source of truth for
# `FORK_VERSION`; stamped and checked like the rest, with the name left alone.
VERSION_ONLY = {"crates/mahbot-inference"}

DEP_START = re.compile(r"^([ \t]*)([A-Za-z0-9_.-]+)([ \t]*=[ \t]*)\{", re.M)
PKG_SECTION = re.compile(r"^\[package\][ \t]*$", re.M)
SECTION = re.compile(r"^\[[^\]\n]+\][ \t]*$", re.M)


def match_brace(text, open_at):
    """Index of the `}` closing the `{` at `open_at`, ignoring string bodies."""
    depth = 0
    i = open_at
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced braces from offset {open_at}: {text[open_at:open_at + 40]!r}")


def section_text(text, header):
    """The body of the table named `header` (`[package]`, `[workspace.package]`)."""
    m = re.search(r"^" + re.escape(header) + r"[ \t]*$", text, re.M)
    if not m:
        return ""
    end = SECTION.search(text, m.end())
    return text[m.end(): end.start() if end else len(text)]


def section_of(text, pos):
    """The header line of the table that contains character offset `pos`."""
    header = ""
    for m in SECTION.finditer(text):
        if m.start() > pos:
            break
        header = m.group(0).strip()
    return header


def field(body, key):
    m = re.search(r"^[ \t]*" + re.escape(key) + r"[ \t]*=[ \t]*\"([^\"]*)\"", body, re.M)
    return m.group(1) if m else None


def package_versions(root):
    """Directory (repo-relative) → the version that directory's package carries."""
    workspace_manifest = os.path.join(root, "Cargo.toml")
    workspace_version = field(section_text(layers.read(workspace_manifest),
                                          "[workspace.package]"), "version")
    versions = {}
    for path in layers.manifests(root):
        text = layers.read(path)
        body = section_text(text, "[package]")
        if not body:
            continue
        version = field(body, "version")
        if version is None:
            inherit = re.search(r"^[ \t]*version\.workspace[ \t]*=[ \t]*true", body, re.M)
            version = workspace_version if inherit else None
        rel = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
        if rel == ".":
            rel = ""
        if version:
            versions[rel] = version
    return versions


def rewrite_deps(text, manifest_dir, root, versions):
    """Apply the layer to every `path = …` dependency entry in one manifest."""
    edits = []  # (start, end, replacement)
    for m in DEP_START.finditer(text):
        open_at = m.end() - 1
        close_at = match_brace(text, open_at)
        body = text[open_at:close_at + 1]
        path = re.search(r'path[ \t]*=[ \t]*"([^"]+)"', body)
        if not path:
            continue
        target = os.path.normpath(os.path.join(manifest_dir, path.group(1)))
        rel = os.path.relpath(target, root).replace(os.sep, "/")
        if rel not in versions:
            continue
        # The workspace registry always pins a version — it is what a member
        # inherits and what the registry is resolved against. Elsewhere a
        # manifest may be path-only on purpose (a dev-dep with no `version` is
        # stripped from the published archive, which is how the back-edges
        # between tiers avoid a publish cycle), so only an existing pin moves.
        registry = section_of(text, open_at) == "[workspace.dependencies]"
        new = body
        if rel in RENAMED:
            name = RENAMED[rel]
            if re.search(r"package[ \t]*=", new):
                new = re.sub(r'(package[ \t]*=[ \t]*)"[^"]*"',
                             lambda mm: mm.group(1) + f'"{name}"', new)
            else:
                new = re.sub(r"^\{[ \t]*", f'{{ package = "{name}", ', new)
        # A pin, where the manifest carries one, must name the version the
        # dependency actually is. Path-only entries stay path-only — except in
        # the workspace registry, which always pins.
        pinned = rel in RENAMED and (registry or re.search(r'version[ \t]*=', new))
        if pinned:
            new = re.sub(r'(version[ \t]*=[ \t]*)"[^"]*"',
                         lambda mm: mm.group(1) + f'"{FORK_VERSION}"', new)
            if not re.search(r'version[ \t]*=', new):
                new = re.sub(r'(path[ \t]*=[ \t]*"[^"]*")',
                             lambda mm: mm.group(1) + f', version = "{FORK_VERSION}"',
                             new, count=1)
        else:
            new = re.sub(r'(version[ \t]*=[ \t]*)"[^"]*"',
                         lambda mm: mm.group(1) + f'"{versions[rel]}"', new)
            if registry and not re.search(r'version[ \t]*=', new):
                new = re.sub(r'(path[ \t]*=[ \t]*"[^"]*")',
                             lambda mm: mm.group(1) + f', version = "{versions[rel]}"',
                             new, count=1)
        if new != body:
            edits.append((open_at, close_at + 1, new))
    for start, end, new in reversed(edits):
        text = text[:start] + new + text[end:]
    return text


def rewrite_package(text, rel):
    """Pin a renamed crate's own name and version, or a version-only crate's version."""
    if rel not in RENAMED and rel not in VERSION_ONLY:
        return text
    m = PKG_SECTION.search(text)
    if not m:
        return text
    end = SECTION.search(text, m.end())
    end_at = end.start() if end else len(text)
    body = text[m.end():end_at]
    # Value-only substitutions: the key, its padding and its `=` stay as written.
    new = body
    if rel in RENAMED:
        new = re.sub(r'(^[ \t]*name[ \t]*=[ \t]*)"[^"]*"',
                     lambda mm: mm.group(1) + f'"{RENAMED[rel]}"', new, count=1, flags=re.M)
    new = re.sub(r'(^[ \t]*version[ \t]*=[ \t]*)"[^"]*"',
                 lambda mm: mm.group(1) + f'"{FORK_VERSION}"', new, count=1, flags=re.M)
    return text[:m.end()] + new + text[end_at:]


def main(argv):
    # `--print-version` is how the release scripts read the number as a *value*
    # rather than by parsing this file's source.
    if "--print-version" in argv[1:]:
        print(FORK_VERSION)
        return 0
    check = "--check" in argv[1:]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions = package_versions(root)
    changed = 0
    for path in layers.manifests(root):
        rel = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
        if rel == ".":
            rel = ""
        text = layers.read(path)
        new = rewrite_deps(rewrite_package(text, rel), os.path.dirname(path), root, versions)
        if new != text:
            changed += 1
            shown = os.path.relpath(path, root)
            if check:
                print(f"would rewrite: {shown}")
            else:
                layers.write_atomic(path, new)
                print(f"rewrote: {shown}")
    if check and changed:
        print(f"{changed} manifest(s) outside the fork's naming layer", file=sys.stderr)
        return 1
    if not changed:
        print("manifests already carry the fork's naming layer")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
