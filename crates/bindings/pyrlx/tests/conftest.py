# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""This fork does not build the Python binding.

``crates/bindings/pyrlx`` is excluded from the workspace (see `exclude` in the
root `Cargo.toml`), so nothing here produces an extension and every module in
this directory fails at collection on `import pyrlx`. They stay on disk so the
next upstream sync merges against them rather than over a hole;
`binding_not_built_test.py` is what runs instead, and it asserts the exclusion
itself rather than a binding that does not exist.
"""

collect_ignore_glob = ["test_*.py"]

