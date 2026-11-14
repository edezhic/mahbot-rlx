#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Apply this fork's package-name layer to every manifest in the tree.

Twenty-six crates are published from here under `mahbot-` names at the fork
version; every other crate keeps upstream's identity. The layer lives in the
manifests, not in the source: a dependency key keeps upstream's name, the
`[workspace.dependencies]` entry names the real package, and each renamed crate
pins `[lib] name` to the module its sources already use (see
docs/fork-publishing.md).

Upstream does not know about any of it, so every sync re-applies it here rather
than through a hand edit per manifest:

* the twenty-six crates get the fork's `[package] name` and version, and
  `mahbot-inference` — this fork's own subject, not a renamed upstream crate —
  gets the version alone;
* a dependency that names one of them by `path` gains `package = "mahbot-…"`,
  which cargo needs to resolve the edge at all;
* every `version` pin that is already there is made to name the version the
  dependency actually carries — the fork's for a renamed crate, the workspace's
  for the rest — and the workspace registry (`[workspace.dependencies]`) always
  pins. A path-only entry elsewhere stays path-only: cargo strips it from the
  published archive, which is how a dev-dep back-edge avoids a publish cycle.
  The exception is a pin that a published archive resolves against the registry
  for a crate the fork never publishes — the optional out-of-scope backends
  `mahbot-rlx-runtime` keeps under upstream names. That pin names
  [`UPSTREAM_VERSION`], the newest release the registry has, because the
  workspace version is upstream's and may not be released yet.

Idempotent: running it twice changes nothing. Usage: `python3
scripts/fork-rename.py [--check | --print-version | --print-upstream-version |
--selftest]`; `--check` exits 1 when a manifest would change (for a gate) and
prints what it would rewrite; the two `--print-*` forms print [`FORK_VERSION`]
and [`UPSTREAM_VERSION`] for the release scripts, whatever state the tree is in;
`--selftest` asserts the caret arithmetic the publish gate decides on and runs
the write path over a manifest that needs a rewrite, which `--check` never does.
"""

import os
import re
import sys

import fork_layers as layers

# The fork's release version. The release process owns this number; a bump
# happens there, not here.
FORK_VERSION = "0.2.18-mahbot.3"

# The newest upstream release on the registry. A pin that travels to the registry
# for a crate the fork never publishes has to name a version that exists there,
# and the family releases in lockstep, so one number covers it. A caret at this
# version admits the workspace version — that is what leaves the local build and
# `Cargo.lock` untouched — and a pin equal to it is the accepted state once
# upstream has published it; `main` refuses only once the caret stops admitting
# the workspace version.
UPSTREAM_VERSION = "0.2.19"

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
    "crates/numerics/rlx-fastmath": "mahbot-rlx-fastmath",
    "crates/numerics/rlx-libm": "mahbot-rlx-libm",
    "crates/numerics/rlx-linux-ref": "mahbot-rlx-linux-ref",
    "crates/numerics/rlx-pocketfft": "mahbot-rlx-pocketfft",
    "crates/numerics/rlx-search": "mahbot-rlx-search",
    "crates/tooling/rlxsl": "mahbot-rlxsl",
}

# Crates that already carry the fork's own name and still take its version. This
# fork's own subject is not a renamed upstream crate, but restating the release
# number in its manifest by hand makes it a second source of truth for
# `FORK_VERSION`; stamped and checked like the rest, with the name left alone.
VERSION_ONLY = {"crates/mahbot-inference"}

# The directories whose packages this fork publishes, and so the only ones whose
# dependencies may name a registry package: everything else keeps upstream's
# identity and is never uploaded from here. `scripts/publish.sh`'s `SCOPE` is the
# same set as names, in upload order; its gate fails a scoped crate that depends
# on a `mahbot-…` member this set does not contain.
PUBLISHED = set(RENAMED) | VERSION_ONLY

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


def relative_dir(root, path):
    """`path`'s directory, repo-relative and `/`-separated ("" for the root)."""
    rel = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
    return "" if rel == "." else rel


def workspace_version(root):
    """The version `[workspace.package]` hands every member that inherits it."""
    text = layers.read(os.path.join(root, "Cargo.toml"))
    return field(section_text(text, "[workspace.package]"), "version")


def package_versions(root, inherited):
    """Directory (repo-relative) → the version that directory's package carries."""
    versions = {}
    for path in layers.manifests(root):
        body = section_text(layers.read(path), "[package]")
        if not body:
            continue
        version = field(body, "version")
        if version is None:
            inherit = re.search(r"^[ \t]*version\.workspace[ \t]*=[ \t]*true", body, re.M)
            version = inherited if inherit else None
        if version:
            versions[relative_dir(root, path)] = version
    return versions


def entries(text):
    """Every `key = { … }` dependency entry: `(key, start, end, body)`."""
    for m in DEP_START.finditer(text):
        start = m.end() - 1
        end = match_brace(text, start) + 1
        yield m.group(2), start, end, text[start:end]


def workspace_registry(root):
    """Registry key → repo-relative directory of the member its `path` names."""
    text = layers.read(os.path.join(root, "Cargo.toml"))
    registry = {}
    for key, _, _, body in entries(section_text(text, "[workspace.dependencies]")):
        path = re.search(r'path[ \t]*=[ \t]*"([^"]+)"', body)
        if path:
            registry[key] = os.path.normpath(path.group(1)).replace(os.sep, "/")
    return registry


def upstream_keys(root, registry):
    """Registry keys a published crate inherits from a package the fork never publishes.

    Those are the edges a published archive resolves against the registry, so
    their pin has to name a version that exists there. Derived rather than
    listed: a sync that gives the runtime a new optional backend, or brings one
    of these crates into the fork's own set, moves the key in or out by itself.
    """
    keys = set()
    for path in layers.manifests(root):
        if relative_dir(root, path) not in PUBLISHED:
            continue
        for key, _, _, body in entries(layers.read(path)):
            if not re.search(r"workspace[ \t]*=[ \t]*true", body):
                continue
            target = registry.get(key)
            if target is not None and target not in PUBLISHED:
                keys.add(key)
    return keys


def rewrite_deps(text, manifest_dir, root, versions, published, upstream):
    """Apply the layer to every `path = …` dependency entry in one manifest."""
    edits = []  # (start, end, replacement) — `entries` already ends past the `}`
    for key, start, end, body in entries(text):
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
        registry = section_of(text, start) == "[workspace.dependencies]"
        new = body
        if rel in RENAMED:
            name = RENAMED[rel]
            if re.search(r"package[ \t]*=", new):
                new = re.sub(r'(package[ \t]*=[ \t]*)"[^"]*"',
                             lambda mm: mm.group(1) + f'"{name}"', new)
            else:
                new = re.sub(r"^\{[ \t]*", f'{{ package = "{name}", ', new)
        # A pin, where the manifest carries one, names the version the dependency
        # actually is: the fork's for a crate it publishes — `rewrite_package`
        # stamps every one of those — and UPSTREAM_VERSION where the pin travels
        # to the registry for one it never does (the workspace entry a published
        # crate inherits, or a pin already in a published crate's manifest). The
        # local version is right everywhere else, because nothing outside this
        # tree resolves it. Path-only entries stay path-only — except in the
        # workspace registry, which always pins.
        carries = re.search(r'version[ \t]*=', new) is not None
        if rel in PUBLISHED:
            wanted = FORK_VERSION
        elif (registry and key in upstream) or (published and carries):
            wanted = UPSTREAM_VERSION
        else:
            wanted = versions[rel]
        new = re.sub(r'(version[ \t]*=[ \t]*)"[^"]*"',
                     lambda mm: mm.group(1) + f'"{wanted}"', new)
        if registry and not carries:
            new = re.sub(r'(path[ \t]*=[ \t]*"[^"]*")',
                         lambda mm: mm.group(1) + f', version = "{wanted}"',
                         new, count=1)
        if new != body:
            edits.append((start, end, new))
    for start, end, new in reversed(edits):
        text = text[:start] + new + text[end:]
    return text


def rewrite_package(text, rel):
    """Pin a renamed crate's own name and version, or a version-only crate's version."""
    if rel not in PUBLISHED:
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


# What `admits` must say about cargo's caret semantics: the forms the naming
# layer writes, and the forms it has to refuse to guess at. The pre-flight gate
# decides whether a release proceeds on this arithmetic, so it is asserted here
# rather than trusted.
CARET_CASES = (
    ("0.2.19", "0.2.20", True),      # a bare version is a caret
    ("^0.2.19", "0.2.19", True),
    ("^0.2.19", "0.2.18", False),
    ("^0.2.20", "0.2.19", False),    # the pin names a newer release than the tree
    ("^0.2.19", "0.3.0", False),     # a caret stays inside the minor it names
    ("^1.2.3", "1.9.9", True),
    ("^1.2.3", "2.0.0", False),
    ("^0", "0.9.0", True),           # a zero major is its own boundary
    ("^0", "1.0.0", False),
    ("^0.0", "0.0.5", True),
    ("^0.0.3", "0.0.3", True),
    ("^0.0.3", "0.0.4", False),
    (">0.2.18", "0.2.19", None),     # not a caret: not read, and not admitted
    ("~0.2.19", "0.2.19", None),
    ("0.2.*", "0.2.19", None),
    ("^0.2.19", "0.2.19-rc.1", None),
)


def selftest():
    """The caret arithmetic the pre-flight decides on, and the write path.

    The write path matters because `--check` compares and never splices: a splice
    that eats the newline after an entry passes every gate in the tree while
    joining that entry to the line below it — a manifest cargo refuses to parse.
    The contract asserted here is that an entry is replaced by its own new body
    and nothing else moves, and that a second pass changes nothing.
    """
    for requirement, version, want in CARET_CASES:
        got = layers.admits(requirement, version)
        if got is not want:
            print(f"fork-rename selftest: admits({requirement!r}, {version!r}) is {got!r}, "
                  f"not {want!r}", file=sys.stderr)
            return 1
    # The fixture is the root manifest, which is not a published member — the
    # shape `main` actually walks it in.
    root, rel = "/tree", ""
    versions = {"crates/mahbot-inference": FORK_VERSION,
                "crates/backends/rlx-cuda": "0.0.0",
                "crates/core/rlx-optim": "0.0.0"}
    before = (
        "[workspace.dependencies]\n"
        'mahbot-inference = { path = "crates/mahbot-inference", version = "0.0.1" }\n'
        'rlx-cuda = { path = "crates/backends/rlx-cuda", version = "0.0.1" }\n'
        'rlx-optim = { path = "crates/core/rlx-optim" }\n'
        "\n[package]\n"
        'name = "probe"\n'
    )
    after = (
        "[workspace.dependencies]\n"
        f'mahbot-inference = {{ path = "crates/mahbot-inference", version = "{FORK_VERSION}" }}\n'
        f'rlx-cuda = {{ path = "crates/backends/rlx-cuda", version = "{UPSTREAM_VERSION}" }}\n'
        'rlx-optim = { path = "crates/core/rlx-optim", version = "0.0.0" }\n'
        "\n[package]\n"
        'name = "probe"\n'
    )
    for label, text in (("a rewrite", before), ("a second rewrite", after)):
        got = rewrite_deps(text, root, root, versions, rel in PUBLISHED, {"rlx-cuda"})
        if got != after:
            print(f"fork-rename selftest: {label} produced\n{got!r}\nnot\n{after!r}",
                  file=sys.stderr)
            return 1
    print("fork-rename selftest: the caret arithmetic holds and a rewrite replaces "
          "the entry and nothing else")
    return 0


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # `--print-version` and `--print-upstream-version` are how the release
    # scripts read the numbers as *values* rather than by parsing this file, so
    # they answer in whatever state the tree is in: a pin that needs fixing must
    # not take the version read down with it.
    for flag, value in (("--print-version", FORK_VERSION),
                        ("--print-upstream-version", UPSTREAM_VERSION)):
        if flag in argv[1:]:
            print(value)
            return 0
    # `--selftest` answers before the guard: the arithmetic it pins is what the
    # guard uses, so a regression there has to be named as such.
    if "--selftest" in argv[1:]:
        return selftest()
    inherited = workspace_version(root)
    if layers.admits(UPSTREAM_VERSION, inherited) is not True:
        print(f"UPSTREAM_VERSION {UPSTREAM_VERSION} does not admit the workspace version "
              f"{inherited}: a local build would take the registry copy", file=sys.stderr)
        return 1
    check = "--check" in argv[1:]
    versions = package_versions(root, inherited)
    upstream = upstream_keys(root, workspace_registry(root))
    changed = 0
    for path in layers.manifests(root):
        rel = relative_dir(root, path)
        text = layers.read(path)
        new = rewrite_deps(rewrite_package(text, rel), os.path.dirname(path), root,
                           versions, rel in PUBLISHED, upstream)
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
