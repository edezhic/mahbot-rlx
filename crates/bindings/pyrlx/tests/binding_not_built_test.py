# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Python binding is not built in this fork — assert that, do not exercise it.

The rest of this directory is upstream's suite for the extension. It is ignored
by `conftest.py` because no extension exists to test: `crates/bindings/pyrlx` is
`exclude`d from the workspace, so `cargo build` never reaches it and no
`_pyrlx.*.so` is produced. What is left to check is the decision itself.
"""

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]


def test_the_python_binding_is_outside_the_workspace():
    manifest = tomllib.loads((_ROOT / "Cargo.toml").read_text())
    workspace = manifest["workspace"]
    assert "crates/bindings/pyrlx" in workspace["exclude"]
    assert "crates/bindings/pyrlx" not in workspace["members"]
    assert "crates/bindings/pyrlx" not in workspace["default-members"]
