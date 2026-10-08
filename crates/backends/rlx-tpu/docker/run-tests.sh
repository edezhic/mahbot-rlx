#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Validation entry point inside the Docker container.
#
# Runs in three layers:
#   1. `cargo test -p rlx-tpu --release --features backend-tests` — host-agnostic
#      unit/check + (when LIBTPU_PATH is set, which it is in this image) the
#      pjrt_roundtrip + pjrt_bench tests against the real plugin.
#   2. `cargo test -p rlx-runtime --features cpu,tpu,backend-tests` —
#      compile-cache integration test that exercises Device::Tpu
#      through Session.
#   3. `validate_hlo.py` — round-trips emitted HLO bytes through
#      jax.lib.xla_extension.HloModule for parse validity.
#
# `backend-tests`: the default test selection is a fast smoke set, so TPU's test
# targets — and the tpu_* targets in rlx-runtime — are not in it, and each layer
# has to ask for them.

set -euo pipefail

cd /work

echo "═══════════════════════════════════════════════════════"
echo "[1/3] cargo test -p rlx-tpu --release --features backend-tests"
echo "═══════════════════════════════════════════════════════"
cargo test -p rlx-tpu --release --features backend-tests -- --nocapture --test-threads=1

echo
echo "═══════════════════════════════════════════════════════"
echo "[2/3] cargo test -p rlx-runtime --features cpu,tpu,backend-tests"
echo "═══════════════════════════════════════════════════════"
cargo test -p rlx-runtime --release --features cpu,tpu,backend-tests \
    --test tpu_compile_cache -- --nocapture --test-threads=1

echo
echo "═══════════════════════════════════════════════════════"
echo "[3/3] HLO byte validation via jax.xla_extension"
echo "═══════════════════════════════════════════════════════"

cargo run --release --quiet \
    -p rlx-tpu --example emit_hlo_samples 2>&1 | tee /tmp/hlo_samples.list

python3 /work/rlx-tpu/docker/validate_hlo.py /tmp/hlo_samples.list

echo
echo "✓ all rlx-tpu validation passed"
