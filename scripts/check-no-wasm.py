#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a crate this fork builds carries a wasm conditional or dependency.

This fork is native-only: the wgpu backend targets Metal, Vulkan and DX12 and is
never built for WebAssembly, so `rlx-web` / `rlx-webgl` are excluded from the
workspace and no buildable crate may keep a wasm branch behind them.

A wasm target name here means `wasm`, `wasm32`, `wasm64`, `wasi` or
`emscripten`, so `target_family = "wasm"` is caught as well as
`target_arch = "wasm32"`.

Matches code shapes only:
  * a `#[cfg(...)]` / `#[cfg_attr(...)]` attribute, outer or inner
    (`#![cfg(...)]`), naming one — multi-line attributes are accumulated until
    their parentheses balance;
  * a `cfg!(...)` macro naming one anywhere on the line;
  * in a `build.rs`, a quoted wasm target name (a comparison against it, or a
    `contains` / `starts_with` test);
  * a Cargo.toml `[target.…]` table header naming one, or a wasm-only crate as
    a dependency — inline (`web-sys = …`) or as a table
    (`[dependencies.web-sys]`).

Deliberately does not match:
  * comments or docs (`//`, `///`, `//!` lines), so prose that mentions wasm is
    not a failure;
  * third-party crates in the dependency graph, which are not ours to change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The directories the fork's recipes build: the main crate tree and the nested
# `android/` workspace.
ROOTS = ("crates", "android")
SKIP_PARTS = {"vendor", "target", ".venv"}
# Excluded from the workspace and upstream-owned: never scanned.
EXCLUDED_DIRS = ("crates/bindings/rlx-web", "crates/backends/rlx-webgl")

CFG_ATTR_RE = re.compile(r"^#!?\[cfg(?:_attr)?\(")
WASM_NAME_RE = re.compile(r"\b(?:wasm(?:32|64)?|wasi|emscripten)\b")
QUOTED_WASM_RE = re.compile(r'"[^"]*\b(?:wasm(?:32|64)?|wasi|emscripten)\b[^"]*"')
WASM_CRATE = r"wasm-bindgen[\w-]*|web-sys|js-sys|console_error_panic_hook"
WASM_DEP_RE = re.compile(rf"^(?:{WASM_CRATE})\s*=")
WASM_DEP_TABLE_RE = re.compile(rf"^\[.*\.(?:{WASM_CRATE})\]$")


def is_comment(line: str) -> bool:
    return line.lstrip().startswith("//")


def is_excluded(rel: str) -> bool:
    return any(rel == d or rel.startswith(d + "/") for d in EXCLUDED_DIRS)


def iter_files():
    for root in ROOTS:
        for path in sorted((ROOT / root).rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(part in SKIP_PARTS for part in parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if is_excluded(rel):
                continue
            if path.suffix == ".rs" or path.name == "Cargo.toml":
                yield path, rel


def scan_rs(path: Path, rel: str, violations: list[tuple[str, int, str]]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    is_build = path.name == "build.rs"
    i = 0
    while i < len(lines):
        line = lines[i]
        if is_comment(line):
            i += 1
            continue
        stripped = line.strip()
        if CFG_ATTR_RE.match(stripped):
            text = [stripped]
            depth = stripped.count("(") - stripped.count(")")
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                nxt = lines[j].strip()
                text.append(nxt)
                depth += nxt.count("(") - nxt.count(")")
            joined = " ".join(text)
            if WASM_NAME_RE.search(joined):
                violations.append((rel, i + 1, joined))
            i = j + 1
            continue
        if "cfg!(" in stripped and WASM_NAME_RE.search(stripped):
            violations.append((rel, i + 1, stripped))
        elif is_build and QUOTED_WASM_RE.search(stripped):
            violations.append((rel, i + 1, stripped))
        i += 1


def scan_cargo(path: Path, rel: str, violations: list[tuple[str, int, str]]) -> None:
    for i, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("[target.") and WASM_NAME_RE.search(stripped):
            violations.append((rel, i, stripped))
        elif WASM_DEP_RE.match(stripped) or WASM_DEP_TABLE_RE.match(stripped):
            violations.append((rel, i, stripped))


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    for path, rel in iter_files():
        if path.suffix == ".rs":
            scan_rs(path, rel, violations)
        else:
            scan_cargo(path, rel, violations)
    scan_cargo(ROOT / "Cargo.toml", "Cargo.toml", violations)

    if not violations:
        return 0
    for rel, line, text in sorted(violations):
        print(f"{rel}:{line}: {text}")
    print(f"{len(violations)} wasm code shape(s) found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
