# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

#  RLX dev recipes (plan #67).
#
#  Borrowed from MAX's pixi-tasks pattern: each common dev command
#  lives here so onboarding doesn't have to remember --features
#  combinations. Install just from https://just.systems if you don't
#  have it; everything also works as plain cargo invocations.
#
#  Run with `just <recipe>` or `just --list` to see all recipes.

# Default recipe — list available commands.
default:
    @just --list

# A GPU recipe that found no GPU has proved nothing, but a skipped test still
# prints `ok` — which is how a rig whose card had fallen off the PCIe bus
# reported a green suite while executing nothing on it. `RLX_REQUIRE_DEVICE`
# turns a skip into a failure for any backend that IS compiled in and still
# could not be instantiated; a backend the build does not contain is unaffected.
#
# On for the GPU recipes below, because asking for `just test-gpu` is asserting
# a GPU is present. A host without one wants `just require_device=0 test-gpu`.
require_device := "1"

# Run the throttle gate before benching. CI-friendly --warn variant
# never exits non-zero.
[no-cd]
throttle:
    {{justfile_directory()}}/scripts/check-throttle.sh

throttle-warn:
    {{justfile_directory()}}/scripts/check-throttle.sh --warn

# Build the whole workspace (release) — `--workspace` is explicit, not
# decorative: a bare `cargo build` here means this fork's default contour,
# which excludes the backends it does not develop (see `default-members` in
# Cargo.toml). `build-workspace` is the whole-set gate, run once per upstream
# sync rather than on every routine build.
build:
    cargo build --release

# Every member, including the backends and the embedded crates this fork does
# not build routinely. Slow and toolchain-dependent: the sync gate, not a
# development loop.
build-workspace:
    cargo build --release --workspace

# Build with everything turned on (Metal, kernel-trace, nan-check).
build-all:
    cargo build --release -p mahbot-rlx-runtime --features "cpu,metal,kernel-trace,nan-check,blas-accelerate"

# Build rlx-mlx. First build compiles MLX from source (~minutes).
# Requires `git submodule update --init rlx-mlx-sys/vendor/mlx`.
build-mlx:
    cargo build --release -p rlx-mlx

# Move vendor/mlx to an upstream ref (default: newest release tag; `main` for
# the tip) and re-apply rlx-mlx-sys/patches. `--build` builds rlx-mlx-sys after.
update-mlx *ARGS:
    {{justfile_directory()}}/crates/backends/rlx-mlx-sys/scripts/update-mlx.sh {{ARGS}}

# Run rlx-mlx tests (matmul+add parity check, both eager and lazy modes).
test-mlx:
    cargo test --release -p rlx-mlx --features backend-tests
    # `-p rlx-sparse`: the two `mlx_sparse_ops` targets register rlx-sparse's
    # MlxKernel, so rlx-sparse has to be built with `mlx`; a feature forward
    # from rlx-runtime's `mlx` does not reach it (see that feature's comment).
    # The `--features` list applies to both selected packages.
    cargo test --release -p mahbot-rlx-runtime -p rlx-sparse --features cpu,mlx,backend-tests \
        --test gru_rnn_mlx --test mlx_attention_parity --test mlx_compile_lir \
        --test mlx_depthwise_convtranspose --test mlx_fft --test mlx_im2col_parity \
        --test mlx_sparse_ops --test mlx_sparse_ops_v2 --test rnn_mlx_f16
    # Opt-in C++ quantized_matmul path for mxfp (must also pass default host path).
    cargo test --release -p rlx-mlx --features native-mxfp,backend-tests --test mxfp_path

# Run rlx-cerebras tests (CSL codegen + matmul oracle parity). Pure Rust, no SDK.
test-cerebras:
    cargo test -p rlx-cerebras

# Run rlx-lbm tests INCLUDING the graph path. `ir` is off by default (it pulls
# rlx-ir + rlx-runtime), so a bare `cargo test -p rlx-lbm` runs only the lib unit
# tests and silently skips `graph_stream` / `lane_reset` — the two that check the
# solver as an rlx Graph rather than as host arithmetic. The third integration
# target (`invariants`) is behind the `backend-tests` gate.
test-lbm:
    cargo test -p rlx-lbm --features ir,backend-tests

# Static graph checker (`cargo rlx check`) — dispatch, fusion, shape/dtype and
# numeric diagnostics. Device-free: CPU legality + all-target fusion, no GPU.
# Pass a graph JSON path or a demo, e.g. `just check-graph "--demo swiglu"`.
check-graph ARGS="--list-demos":
    cargo run -q -p rlx-check --bin cargo-rlx -- rlx check {{ARGS}}

# Install `cargo-rlx` so `cargo rlx check …` works from any RLX crate.
install-check:
    cargo install --path crates/tooling/rlx-check

# Emit CSL artifacts for an MxKxN matmul (default 32x64x32) into OUT.
# Compile + run on a Linux host with the Cerebras SDK container:
#   cd OUT && bash commands_wse2.sh
cerebras-emit M="32" K="64" N="32" OUT="cerebras-out":
    cargo run -q -p rlx-cerebras --bin rlx-cerebras-emit -- {{M}} {{K}} {{N}} {{OUT}}

# Emit + (if cslc is on PATH, i.e. inside the SDK container) compile & simulate.
cerebras-sim M="32" K="64" N="32" OUT="cerebras-out": (cerebras-emit M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v cslc >/dev/null 2>&1; then
        cd {{OUT}} && bash commands_wse2.sh
    else
        echo "cslc not found — run inside the Cerebras SDK container (Linux host)."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash commands_wse2.sh"
    fi

# Run rlx-egpu tests + the Device::Egpu runtime seam. Pure Rust, no SDK, no device.
test-egpu:
    cargo test -p rlx-egpu --all-features
    cargo test -p mahbot-rlx-runtime --features egpu,backend-tests --test egpu_seam

# Report what the PCIe tunnel is carrying (see docs/egpu.md).
egpu-probe:
    cargo run -q -p rlx-egpu --example egpu_probe

# Read a baked kernel pack or code object — needs no GPU toolchain.
egpu-inspect ARTIFACT:
    cargo run -q -p rlx-egpu --example egpu_inspect -- {{ARTIFACT}}

# Bake an ahead-of-time kernel pack. Needs hipcc (--hip) or ptxas (--ptx) on this host.
#   just egpu-bake "--hip kernels.hip:gfx1100,gfx1201" k.rlxisa
egpu-bake SPEC OUT="kernels.rlxisa":
    cargo run -q -p rlx-egpu --example egpu_bake -- -o {{OUT}} {{SPEC}}

# Fetch + verify the signed vendor firmware a bring-up hands to the card.
# FAMILIES: vendor (amd/nvidia) or IP family (gc_12_0 …); empty = everything.
egpu-firmware FAMILIES="":
    scripts/pull_gpu_firmware.sh {{FAMILIES}}
    cargo run -q -p rlx-egpu --features firmware --example egpu_firmware -- {{FAMILIES}}

# Run rlx-qnn tests (QNN model-C++ codegen + matmul oracle parity). Pure Rust, no SDK.
test-qnn:
    cargo test -p rlx-qnn

# Emit QNN model artifacts for an MxKxN matmul (default 32x64x32) into OUT.
# Build + run on a Linux host with the QNN SDK (QNN_SDK_ROOT set):
#   cd OUT && bash run_qnn.sh
qnn-emit M="32" K="64" N="32" OUT="qnn-out":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- {{M}} {{K}} {{N}} {{OUT}}

# Emit a MatMul+Softmax QNN model for offline qnn-net-run.
qnn-emit-matmul-softmax M="8" K="16" N="4" OUT="qnn-mmsm":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --matmul-softmax {{M}} {{K}} {{N}} {{OUT}}

# Emit a two-layer MLP (LinearRelu → Linear) QNN model for offline qnn-net-run.
qnn-emit-mlp2 M="8" K="16" H="32" N="4" OUT="qnn-mlp2":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --mlp2 {{M}} {{K}} {{H}} {{N}} {{OUT}}

# Emit a Linear with STATIC weight/bias (activation-only input) for offline qnn-net-run.
qnn-emit-linear-static M="8" K="16" N="4" OUT="qnn-linstatic":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-static {{M}} {{K}} {{N}} {{OUT}}

# Emit LinearStatic then run the offline context-binary path (needs QNN SDK on PATH).
qnn-run-context M="8" K="16" N="4" OUT="qnn-linstatic": (qnn-emit-linear-static M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v qnn-context-binary-generator >/dev/null 2>&1; then
        cd {{OUT}} && bash run_qnn_context.sh
    else
        echo "qnn-context-binary-generator not found — install the Qualcomm AI Engine Direct SDK."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash run_qnn_context.sh"
    fi

# Emit a Linear (in0·in1+in2) QNN model for offline qnn-net-run.
qnn-emit-linear M="8" K="16" N="4" OUT="qnn-linear":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear {{M}} {{K}} {{N}} {{OUT}}

# Emit a LinearRelu (relu(in0·in1+in2)) QNN model for offline qnn-net-run.
qnn-emit-linear-relu M="8" K="16" N="4" OUT="qnn-linrelu":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-relu {{M}} {{K}} {{N}} {{OUT}}

# Emit + (if the QNN SDK tools are on PATH) build & run on the x86 reference backend.
qnn-run M="32" K="64" N="32" OUT="qnn-out": (qnn-emit M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v qnn-net-run >/dev/null 2>&1; then
        cd {{OUT}} && bash run_qnn.sh
    else
        echo "qnn-net-run not found — install the Qualcomm AI Engine Direct SDK (Linux host)."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash run_qnn.sh"
    fi

# Run rlx-fpga tests (Verilog codegen + INT8 reference parity). Pure Rust.
test-fpga:
    cargo test -p rlx-fpga --features backend-tests --release
    cargo test -p mahbot-rlx-runtime --features cpu,fpga --lib export::

# Emit target-agnostic SystemVerilog for TinyConv-MNIST.
# TARGET = latency|size|energy|precision|bandwidth
# HW     = generic|ecp5|ice40|xilinx7:PART
fpga-emit TARGET="precision" HW="generic" OUT="":
    #!/usr/bin/env bash
    set -e
    if [ -n "{{OUT}}" ]; then
        cargo run -q -p rlx-fpga --release --bin rlx-fpga-emit -- --target {{TARGET}} --hw {{HW}} --out {{OUT}}
    else
        cargo run -q -p rlx-fpga --release --bin rlx-fpga-emit -- --target {{TARGET}} --hw {{HW}}
    fi

# Refresh the checked-in MNIST SystemVerilog demo (examples/mnist_sv/).
fpga-mnist-demo:
    cargo run -q -p rlx-fpga --release --example export_mnist

# SDK-free Docker self-test of the QNN host harness (verify.py + plumbing).
# Needs only Docker; uses a numpy stand-in for qnn-net-run.
qnn-docker-test M="8" K="16" N="4":
    python3 crates/backends/rlx-qnn/docker/validate.py harness-test --dims {{M}} {{K}} {{N}}

# Real Docker validation: build the model lib + run on libQnnCpu.so.
# Needs Docker AND the proprietary QNN SDK (set QNN_SDK_ROOT).
qnn-docker-run M="32" K="64" N="32":
    python3 crates/backends/rlx-qnn/docker/validate.py run --dims {{M}} {{K}} {{N}}

# Native (no Docker) QNN FFI + Session validation on a Linux host with the SDK.
# Expects QNN_SDK_ROOT (and typically LD_LIBRARY_PATH for libc++). Skips cleanly
# without the backend lib. Default backend: libQnnCpu.so.
qnn-ffi:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT / QNN SDK root}"
    export RLX_QNN_BACKEND_LIB="${RLX_QNN_BACKEND_LIB:-$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so}"
    export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cargo test -p rlx-qnn --features runtime -- --nocapture
    cargo test -p mahbot-rlx-runtime --features qnn,backend-tests --test qnn_hexagon_matmul -- --nocapture
    cargo test -p mahbot-rlx-runtime --features "cpu,qnn,backend-tests" --test fused_attention_block_parity -- --nocapture
    cargo test -p mahbot-rlx-runtime --features "cpu,qnn,backend-tests" --test qnn_dequant_matmul -- --nocapture
    cargo test -p mahbot-rlx-runtime --features qnn,backend-tests --test qnn_int8_matmul -- --nocapture
    cargo test -p mahbot-rlx-runtime --features qnn,backend-tests --test qnn_int4_matmul -- --nocapture

# x86 HTP *functional simulator* (libQnnHtp.so) — no Snapdragon silicon.
# Re-runs CPU soft-skip probes (sfixed8×sfixed8) plus int4/int8 MatMul and a
# LinearStatic offline model.so + context-binary path under HTP prepare.
# Forces HTP even if RLX_QNN_BACKEND_LIB already points at libQnnCpu.so
# (common in env.sh); override with RLX_QNN_HTP_LIB if needed.
qnn-htp-sim:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT / QNN SDK root}"
    export PYTHONPATH="${PYTHONPATH:-}"
    TARGET=x86_64-linux-clang
    export RLX_QNN_BACKEND_LIB="${RLX_QNN_HTP_LIB:-$QNN_SDK_ROOT/lib/$TARGET/libQnnHtp.so}"
    export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/$TARGET${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    test -f "$RLX_QNN_BACKEND_LIB" || { echo "missing HTP backend: $RLX_QNN_BACKEND_LIB"; exit 1; }
    echo "HTP functional sim: $RLX_QNN_BACKEND_LIB"
    # FFI probes that CPU soft-skips or that exercise quantized weight paths.
    # cargo test accepts one filter; run each name separately.
    for t in \
        ffi_sfixed8_matmul_probe \
        ffi_int4_static_matmul \
        ffi_int8_static_matmul \
        ffi_int8_per_channel_matmul \
        ffi_int8_param_matmul
    do
        echo "=== $t ==="
        cargo test -p rlx-qnn --features runtime --lib "$t" -- --nocapture
    done
    # Offline LinearStatic on HTP sim (model.so + context binary).
    OUT=$(mktemp -d)
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-static 4 8 2 "$OUT"
    (cd "$OUT" && bash run_qnn.sh)
    (cd "$OUT" && bash run_qnn_context.sh)
    echo "qnn-htp-sim OK"

# FKL region fusion parity (docs/fk-fusion.md). Metal MPS tests skip off macOS.
test-fk:
    cargo test -p mahbot-rlx-fusion fk_
    cargo test -p mahbot-rlx-compile --lib fusion_pipeline::tests
    cargo test -p rlx-tpu --features backend-tests --test fk_pipeline --test hlo_match batch_elementwise
    cargo test -p mahbot-rlx-runtime --features cpu,metal,gpu,tpu,backend-tests --test fk_prologue_parity
    cargo test -p rlx-metal --features backend-tests --test mps_graph_batch_region_lower --test mps_graph_prologue_region_lower
    cargo test -p rlx-mlx --features backend-tests --test basic batch_elementwise_region_matches_atomic

# Logical CPUs for cargo `-j` / libtest `--test-threads` (macOS, else Linux).
cpus := `sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4`

# Run the default fast smoke selection (see the body) at high parallelism.
# Integration targets anywhere in the workspace are opt-in: `--features
# backend-tests`.
# On Darwin, Metal / MoltenVK / MLX share one GPU — cap concurrent cargo
# jobs so crates don't deadlock the device; within a binary, libtest still
# uses the default (CPU count) thread pool. Linux keeps full -j / threads.
# Keep `--test-threads=1` only on recipes that share a GPU runtime unsafely.
test:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    # Integration test targets across the workspace sit behind
    # `required-features = "backend-tests"` and stay out of this default run —
    # the fast smoke gate. The per-backend recipes (`just test-gpu`, `test-mlx`,
    # `test-rocm`, …) pass the feature explicitly, and `required-features` gates
    # a test *target*, so in those recipes the crates' lib unit tests run too —
    # except `rlx-metal`, whose manifest (fork-local) drops its lib unit tests,
    # doctests, examples and auto-discovered test files from every run.
    # This run is the *selection*, not just a target filter: a bare `cargo test`
    # at the root means the fork's default contour, so no Apple backend is
    # compiled here at all. (Linux x86_64 excepted: rlx-collectives' retained
    # upstream dev-dependency row still pulls CUDA / ROCm / XDNA into this run.)
    # `just test-gpu` and the backend recipes below are what build the rest.
    # `mahbot-inference` gates twelve more targets this way, behind its own
    # `hand-run` feature rather than `backend-tests`: every test in them is
    # `#[ignore]`d, so ungated this run links and executes twelve binaries that run
    # nothing. `just test-workspace` and `just lint` enable that feature.
    if [[ "$(uname -s)" == Darwin ]]; then
        # A few parallel test binaries is enough; more contended Apple GPU.
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test --release -j "$JOBS"
    else
        # One test target that fails to compile aborts the whole workspace
        # run, `--no-fail-fast` or not — on the Linux rig that was seven
        # ungated Apple-only tests, and zero tests ran. Build every target
        # first so the report names all of them.
        cargo build --release --tests --keep-going -j "$CPUS"
        cargo test --release -j "$CPUS" -- --test-threads="$CPUS"
    fi
    # The default run builds each selected crate with its DEFAULT features, so
    # a test target gated behind a non-default feature (or behind the
    # `backend-tests` gate) is skipped without saying so. rlx-lbm's graph path
    # is the case in tree; keep it in the gate.
    cargo test --release -p rlx-lbm --features ir,backend-tests
    just test-debug-verifier restricted

# Every member, once per upstream sync (see `docs/fork-sync.md`): the whole-set
# counterpart of `test`. No package selection, so the backends and the embedded
# crates are compiled and run here and nowhere else. Three things it does not do:
# `cargo test --workspace` builds each package with its DEFAULT features, so a
# target behind `required-features` is still skipped — `test-gpu` and the
# per-backend recipes are what build those; `rlx-mlx-sys` / `rlx-mlx` are excluded
# because this fork's contour does not ship the MLX backend, which is a permanent
# decision and not a reading of the checkout — the vendored submodule being
# present does not put them back; and six of upstream's own tests need a dataset
# or a toolchain this host does not have, so they are skipped by name rather than
# left to redden the gate. A gate that is red out of the box is not a gate: a sync
# reads the red as a regression and chases it.
#
# One gated set is reached on purpose: `mahbot-inference`'s twelve hand-run targets
# are behind that crate's `hand-run` feature so the routine run does not link
# binaries that execute nothing, and `--features mahbot-inference/hand-run` below
# is what keeps this whole-set run's coverage where it was.
#
# `--skip` matches a test *name*, not a file, which is why the list below is
# six names rather than three targets. Each carries what it wants in the comment
# beside it; run one by hand on a rig that has the prerequisite.
# `--no-fail-fast` so anything else still reports.
test-workspace:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    excl="--exclude rlx-mlx-sys --exclude rlx-mlx"
    feat="--features mahbot-inference/hand-run"
    skip=""
    # MNIST IDX files at `RLX_MNIST_DIR`.
    skip="$skip --skip audit_pair_switch_graft_and_answer"
    skip="$skip --skip read_tasks_hide_the_answer_and_score_it"
    skip="$skip --skip steer_task_scores_both_halves"
    # A Ruby whose `Fiddle` has `TYPE_INT32_T`.
    skip="$skip --skip ruby_binding_tests_and_examples"
    # A Swift compiler that can type-check the examples.
    skip="$skip --skip the_swift_examples_run_against_the_c_abi"
    skip="$skip --skip the_swift_test_suite_passes"
    if [[ "$(uname -s)" == Darwin ]]; then
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test --release -j "$JOBS" --workspace $excl $feat --no-fail-fast -- $skip
    else
        cargo build --release --workspace --tests --keep-going $excl $feat -j "$CPUS"
        cargo test --release -j "$CPUS" --workspace $excl $feat --no-fail-fast -- \
            --test-threads="$CPUS" $skip
    fi

# Run the IR-verifier-bearing tests in a DEBUG profile.
#
# `rlx_ir::debug_assert_valid!` re-verifies the graph after every fusion pass
# that changed it — and it is `debug_assert`, so **`--release` never runs it**.
# Every other recipe here is `--release`, which meant the verifier was compiled
# out of the entire gate. Two real defects were sitting behind that:
#
#   * `Op::FusedTransformerLayer` declared 10 inputs while its lowering reads 8,
#     so a correctly-built no-bias node failed verification;
#   * rlx-vulkan emitted a `Step` with no matching `StepDep` on the `KvAppend`
#     path, tripping a `debug_assert_eq!` in its scheduler.
#
# Both passed `just test` and `just test-gpu` the whole time. Debug is slow, so
# this is scoped to the crates whose passes and schedulers carry the asserts
# rather than the whole workspace — the point is to run the checks at all.
#
# `mode=restricted` is how the default `test` recipe calls it: the runtime's
# default selection (lib unit tests plus the pool-width checks). `full` mode adds
# the gated targets of the four crates below.
#
# The runtime line runs this fork's backend only (`cpu,gpu`), and that applies to
# BOTH modes. Upstream's list — `cpu,gpu,coreml,mlx,vulkan` on an Apple host —
# is the way back, but re-adding it is a hand edit, and it does not restore what
# this trim gives up: the debug-assert graph-invariant route over the Apple and
# Vulkan graphs is now covered by no gate at all (the defect above was in a
# Vulkan target).
test-debug-verifier mode="full":
    #!/usr/bin/env bash
    set -euo pipefail
    # This fork develops wgpu (`gpu`) only, plus the CPU baseline.
    FEATURES="cpu,gpu"
    # Those three crates' integration targets carry the verifier asserts, so
    # they are gated too: `restricted` runs their lib unit tests only.
    case "{{mode}}" in
      restricted)
        cargo test -p mahbot-rlx-ir -p mahbot-rlx-fusion -p mahbot-rlx-compile ;;
      *)
        FEATURES="$FEATURES,backend-tests"
        cargo test -p mahbot-rlx-ir -p mahbot-rlx-fusion -p mahbot-rlx-compile --features backend-tests ;;
    esac
    cargo test -p mahbot-rlx-runtime --features "$FEATURES" -j 4 --no-fail-fast

# GPU backends + runtime feature tests for the host platform.
# Darwin: Metal, wgpu, CoreML, third-order.
# Linux: wgpu, optional CUDA/ROCm via `test-third-order-gpu` / `test-rocm`.
# MLX is not here: this fork's contour does not ship it, so neither this recipe
# nor `test-workspace` builds `-p rlx-mlx` whatever the checkout contains.
# When `$RLX_HF_CACHE` (or `~/.cache/rlx/hf`) already holds an mlx-community
# checkout, Metal's `metal_hf_mlx_one_linear` runs without `RLX_HF_MLX=1`.
# Cold download: `RLX_HF_MLX=1 just test-gpu`.
#
# The Vulkan rows are not here: they need a Vulkan ICD, which this fork does not
# ship and which `ci-workspace` cannot assume, so they live in `test-vulkan`
# instead. That keeps this recipe — and the whole-set gate that runs it — green
# on a host with only Metal and wgpu.
test-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    JOBS=$(( CPUS < 4 ? CPUS : 4 ))
    # A backend that skips for want of a device must not report `ok` here.
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      Darwin)
        cargo test --release -p rlx-metal --features backend-tests -j "$JOBS"
        cargo test --release -p mahbot-rlx-wgpu --features backend-tests -j "$JOBS"
        # Apple backends spelled out rather than as `apple`, which implies `mlx`:
        # this fork does not ship the MLX backend, so the test gates do not
        # select it (see the `exclude` note in Cargo.toml and `docs/fork-sync.md`
        # §3 — the whole-set build and lint recipes still cover the members).
        # Metal, portable-wgpu-on-Metal and CoreML are what `apple` adds on top.
        # `-p rlx-sparse`: this row runs the whole rlx-runtime suite, including
        # the `*_sparse_ops` targets that register rlx-sparse's Metal kernels.
        # The `metal` token is for the rlx-sparse half of `-p`, which needs its
        # own features (see rlx-runtime's `metal` feature comment).
        cargo test --release -p mahbot-rlx-runtime -p rlx-sparse --features cpu,metal,gpu,coreml,backend-tests -j "$JOBS"
        just test-third-order-gpu
        just check-corpus-device
        ;;
      *)
        cargo test --release -p mahbot-rlx-wgpu --features backend-tests -j "$CPUS" -- --test-threads="$CPUS"
        cargo test --release -p mahbot-rlx-runtime --features cpu,gpu,backend-tests -j "$CPUS" -- --test-threads="$CPUS"
        just test-third-order-gpu
        if cargo check -p mahbot-rlx-runtime --features cpu,cuda,gpu --quiet 2>/dev/null; then
          cargo test --release -p rlx-cuda --features backend-tests -j "$CPUS" -- --test-threads="$CPUS"
          cargo test --release -p mahbot-rlx-runtime --features cpu,cuda,gpu,backend-tests -j "$CPUS" -- --test-threads=1
        fi
        just test-rocm
        just check-corpus-device
        ;;
    esac

# The Vulkan rows, on their own: `rlx-vulkan` is its own `ash` instance with
# embedded SPIR-V, distinct from wgpu's Vulkan adapter, and it needs a Vulkan ICD
# registered (`MoltenVK` on macOS, the loader alone is not enough). Deliberately
# not in `ci-workspace` — a whole-set gate has to be green on a host without one.
# Run it on a rig: `just test-gpu test-vulkan`.
test-vulkan:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    JOBS=$(( CPUS < 4 ? CPUS : 4 ))
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      Darwin)
        cargo test --release -p rlx-vulkan --features backend-tests -j "$JOBS"
        cargo test --release -p mahbot-rlx-runtime --features cpu,metal,gpu,coreml,vulkan,backend-tests \
          --test vulkan_parity --test vulkan_dequant_grouped_matmul --test vulkan_dequant_moe_weights
        ;;
      *)
        cargo test --release -p rlx-vulkan --features backend-tests -j "$CPUS" -- --test-threads="$CPUS"
        ;;
    esac

# GGUF grouped MoE integration — serial when multiple GPU backends link in.
test-gguf-grouped:
    cargo test -p mahbot-rlx-runtime --features backend-tests --test dequant_grouped_matmul_gguf -- --test-threads=1

# rlx-js: the JS surface. `FEATURES` picks backends the same way rlx-runtime
# does. The default is the crate's own default set, so a bare `just test-js`
# covers the whole surface; tests for a feature that is off are cfg'd out, so
# a narrower set is green rather than full of "undefined is not a constructor".
test-js FEATURES="cpu,gguf,weights,training,text":
    cargo test -p rlx-js --no-default-features --features {{FEATURES}},backend-tests

# MNIST from JavaScript. Fetches the dataset into the cache dir if missing.
# Usage: just js-mnist            (MLP, ~8s on Metal)
#        just js-mnist --cnn
js-mnist *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    D="$HOME/.cache/torchvision-mnist/MNIST/raw"
    mkdir -p "$D"
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
        # Re-fetch on a truncated file too: a failed download leaves a short one.
        if [[ ! -s "$D/$f" || $(wc -c <"$D/$f") -lt 10000 ]]; then
            echo "fetching $f"
            curl -sSfL "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" -o "$D/$f.gz"
            gunzip -f "$D/$f.gz"
        fi
    done
    cargo run -q --release -p rlx-js --features "${FEATURES:-cpu,gguf,weights,training,text}" \
        --bin rlx-js -- crates/bindings/rlx-js/examples/mnist.js {{ARGS}}

# LoRA from JavaScript: adapt a frozen MNIST model to inverted pixels.
# Trains the base first if the checkpoint is missing.
js-lora *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    CK="${CKPT:-/tmp/rlx-mnist-base.gguf}"
    FEATURES="${FEATURES:-cpu,gguf,weights,training,text}"
    if [[ ! -s "$CK" ]]; then
        echo "no base checkpoint; training one first"
        FEATURES="$FEATURES" just js-mnist --save "$CK"
    fi
    cargo run -q --release -p rlx-js --features "$FEATURES" \
        --bin rlx-js -- crates/bindings/rlx-js/examples/mnist_lora.js "$CK" {{ARGS}}

# Run a JavaScript file against the RLX API.
# Usage: just js crates/bindings/rlx-js/examples/train.js
#        just js FEATURES=metal ... -- extra args for the script
js SCRIPT *ARGS:
    cargo run -q -p rlx-js --features "{{env('FEATURES', 'cpu,gguf')}}" --bin rlx-js -- {{SCRIPT}} {{ARGS}}

# PyTorch → RLX: convert a torch model file to an RLX bundle + generated crate.
# `MODEL` is a .py exposing `model` + `example_inputs` (or get_model()/build()).
# Usage: just torch-import path/to/model.py out/
torch-import MODEL OUT:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}/crates/bindings/pyrlx"
    if [[ ! -d .venv-torch ]]; then
        python3 -m venv .venv-torch
        .venv-torch/bin/pip install -q torch safetensors numpy
    fi
    # Run the front-end as a standalone script (no compiled _pyrlx needed); it
    # shells to the Rust `rlx-torch-import` worker via the cargo fallback.
    .venv-torch/bin/python python/pyrlx/torch_import.py "{{MODEL}}" -o "{{OUT}}"

# rlx-torch-import: run the aten→rlx importer's Rust tests.
test-torch-import:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    if [[ "$(uname -s)" == Darwin ]]; then
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test -p rlx-torch-import -j "$JOBS"
    else
        cargo test -p rlx-torch-import -j "$CPUS" -- --test-threads="$CPUS"
    fi

# Run a specific filter; use as `just testf narrow_attention`.
# Filters run against the default selection — this fork's contour, i.e. the
# `default-members` in Cargo.toml, not every member — so a FILTER naming a target
# behind `required-features = "backend-tests"` (every integration target in the
# workspace — see the `test` recipe) matches nothing and exits 0: add
# `--features backend-tests`, or name the package, as
# `cargo test --release --features backend-tests --test ...`.
testf FILTER:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    if [[ "$(uname -s)" == Darwin ]]; then
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test --release -j "$JOBS" {{FILTER}}
    else
        cargo test --release -j "$CPUS" {{FILTER}} -- --test-threads="$CPUS"
    fi

# Format check (no rewrite). Mirrors what CI should run.
fmt-check:
    cargo fmt --all -- --check

# Auto-format.
fmt:
    cargo fmt --all

# Clippy with warnings as errors.
#
# `--all-targets` covers every target of every crate, but only under DEFAULT
# features — so a `#[cfg(feature = "vulkan")]` test or a backend behind an
# optional dep is invisible to it. That is not a hypothetical gap: it is why
# feature-gated code has landed unlinted here before. `lint-features` covers the
# combinations this host can actually build.
lint: lint-features
    # Naming no package means this fork's default contour (see
    # `default-members` in Cargo.toml); `lint-workspace` is the whole-set gate.
    # `backend-tests`: the default test selection is a fast smoke set, and
    # `required-features` hides every gated target from `--all-targets` too —
    # without it they would not be type-checked by `just lint` at all. `hand-run`
    # is the same for `mahbot-inference`'s twelve hand-run targets.
    cargo clippy --all-targets --features backend-tests,hand-run -- -D warnings

# Every member, once per upstream sync (see `docs/fork-sync.md`).
lint-workspace: lint-features
    cargo clippy --workspace --all-targets --features backend-tests,hand-run -- -D warnings

# Clippy the feature combinations `--all-targets` alone cannot reach.
#
# Host-appropriate by design: linting `apple` on Linux or `cuda` without a
# toolkit fails for reasons that have nothing to do with the code. CUDA/ROCm are
# probed first and skipped when they cannot build, so this is safe to run
# anywhere — and says which combinations it actually checked, because a lint
# gate that silently covered two of five reads exactly like one that covered all
# five.
lint-features:
    #!/usr/bin/env bash
    set -euo pipefail
    # Every row also passes `backend-tests`: `required-features` hides those
    # targets from `--all-targets` too, so without it a row would type-check the
    # combinations it exists for and silently skip their bodies.
    checked=(); skipped=()
    lint() {  # lint <label> <cargo args...>
        local label="$1"; shift
        if cargo clippy "$@" --all-targets -- -D warnings; then
            checked+=("$label")
        else
            echo "FAILED: $label" >&2; return 1
        fi
    }
    case "$(uname -s)" in
      Darwin)
        lint "rlx-runtime cpu,metal,coreml,gpu,vulkan" -p mahbot-rlx-runtime --features cpu,metal,coreml,gpu,vulkan,backend-tests
        lint "rlx-corpus metal,coreml,gpu,vulkan"      -p rlx-corpus  --features metal,coreml,gpu,vulkan,backend-tests
        lint "rlx-ir test-support"          -p mahbot-rlx-ir      --features test-support,backend-tests
        # The AMX/SME/BNNS paths are opt-in and default-OFF, so nothing else
        # here compiles them. `amx-bnns` shipped with a Result-vs-Option arm
        # that did not build at all, for exactly that reason.
        lint "rlx-cpu amx+splat"            -p mahbot-rlx-cpu     --features amx-bnns,amx-dense,amx-sme,splat,parity-gemm,backend-tests
        ;;
      *)
        lint "rlx-runtime cpu,gpu,vulkan"   -p mahbot-rlx-runtime --features cpu,gpu,vulkan,backend-tests
        lint "rlx-corpus gpu,vulkan"        -p rlx-corpus  --features gpu,vulkan,backend-tests
        lint "rlx-ir test-support"          -p mahbot-rlx-ir      --features test-support,backend-tests
        ;;
    esac
    # Default-off feature gates every host compiles.
    lint "rlx-flow formats"                 -p mahbot-rlx-flow    --features formats,backend-tests
    lint "rlx-cpu int8-gemv,portable-gemm"  -p mahbot-rlx-cpu     --no-default-features --features int8-gemv,portable-gemm
    # `rlx-bench`'s frame-slicing harness measures the device path through a
    # `#[cfg(feature = "gpu")]` half that no other row's target set reaches.
    if cargo clippy -p rlx-bench --features gpu --example bench_frame_slicing -- -D warnings; then
        checked+=("rlx-bench gpu (frame-slicing example)")
    else
        echo "FAILED: rlx-bench gpu (frame-slicing example)" >&2; exit 1
    fi
    # The int8 GEMVs and the decode-attention lanes are `cfg(target_arch =
    # "x86_64")`: an Apple Silicon host never compiles them. Cross-lint them
    # when the x86-64 macOS target is installed (`rustup target add
    # x86_64-apple-darwin`).
    if [[ "$(uname -s)-$(uname -m)" == Darwin-arm64 ]]; then
        if rustup target list --installed 2>/dev/null | grep -qx x86_64-apple-darwin; then
            lint "rlx-cpu x86-64 int8-gemv,portable-gemm" -p mahbot-rlx-cpu --target x86_64-apple-darwin --no-default-features --features int8-gemv,portable-gemm
        else
            skipped+=("rlx-cpu x86-64 (rustup target add x86_64-apple-darwin)")
        fi
    fi
    for f in cuda rocm; do
        if cargo check -p mahbot-rlx-runtime --features "cpu,$f" --quiet 2>/dev/null; then
            lint "rlx-runtime cpu,$f" -p mahbot-rlx-runtime --features "cpu,$f,backend-tests"
        else
            skipped+=("rlx-runtime cpu,$f (does not build on this host)")
        fi
    done
    echo "lint-features checked: ${checked[*]}"
    [[ ${#skipped[@]} -eq 0 ]] || printf 'lint-features SKIPPED: %s\n' "${skipped[@]}"

# Cross-check every Metal dispatch's buffer bindings against the kernel's
# declared parameters. Buffers are bound by integer index against MSL signatures
# in kernels.rs, and a stale index is not a compile error, not a crash and not a
# GPU fault — it reads zero, so a kernel whose `len` moved does nothing at all.
# Off by default (a relaxed atomic load); this turns it on for the suite.
validate-metal-bindings *ARGS:
    RLX_METAL_VALIDATE_BINDINGS=1 cargo test --release -j 4 -p rlx-metal --features backend-tests --no-fail-fast {{ARGS}}

# Objective-C refcount gate for the Metal backend (macOS; no-ops elsewhere).
# The test suite cannot see a leak — an over-retained object still computes the
# right answer — so this runs Metal test binaries under `leaks --atExit`.
leak-check *TESTS:
    {{justfile_directory()}}/crates/backends/rlx-metal/scripts/leak-check.sh {{TESTS}}

# Install repo git hooks (auto-fmt + clippy on commit). Safe to re-run.
install-git-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(git rev-parse --show-toplevel)"
    mkdir -p "$ROOT/.git/hooks"
    ln -sfn ../../scripts/git-hooks/pre-commit "$ROOT/.git/hooks/pre-commit"
    chmod +x "$ROOT/scripts/git-hooks/pre-commit"
    echo "installed .git/hooks/pre-commit → scripts/git-hooks/pre-commit"

# Refresh docs/op-coverage.md checkmarks from backend SUPPORTED_OPS claims.
gen-op-coverage:
    python3 scripts/gen-op-coverage.py

# Fail if docs/op-coverage.md drifts from backend SUPPORTED_OPS claims.
check-op-coverage:
    python3 scripts/gen-op-coverage.py --check

# Regenerate the bindings derived from rlx-bind's op table: the C ABI glue
# (rlx-capi/src/graph_ops.rs), rlx.h, Swift's GraphOps.swift, the Graph
# section of pyrlx's .pyi, and docs/bindings-api.md. rlx-js and pyrlx expand
# the table at compile time and need no step. See docs/bindings.md.
gen-bindings:
    cargo run -q -p rlx-bindgen

# Standalone binding releases (docs/releasing-bindings.md). The workspace
# version in Cargo.toml is the one source: stamp it into every binding
# manifest after a bump, and `check-binding-versions` (in `just ci-workspace`)
# fails on drift.
sync-binding-versions:
    python3 scripts/bindings_release.py versions

check-binding-versions:
    python3 scripts/bindings_release.py versions --check

# Package every binding with prebuilt librlx_capi under natives/<platform>/
# (the release workflow does this for five platforms). Locally, for this host:
#   just package-bindings              # builds rlx-capi, stages into ./dist
package-bindings natives="" out="dist":
    #!/usr/bin/env bash
    set -euo pipefail
    natives="{{natives}}"
    if [[ -z "$natives" ]]; then
      cargo build --release -p rlx-capi
      case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) plat=macos-arm64 ;; Darwin-x86_64) plat=macos-x64 ;;
        Linux-aarch64) plat=linux-arm64 ;; Linux-x86_64) plat=linux-x64 ;;
        *) echo "unknown host; pass natives=<dir>" >&2; exit 1 ;;
      esac
      natives="$(mktemp -d)"
      mkdir -p "$natives/$plat"
      cp target/release/librlx_capi.* "$natives/$plat/"
    fi
    python3 scripts/bindings_release.py stage --natives "$natives" --out {{out}} --build

# Fail if any generated binding is stale (an op-table or rlx-capi change
# without `just gen-bindings`).
check-bindings:
    cargo run -q -p rlx-bindgen -- --check

# Every generated FFI binding (Go, Java, C#, Ruby, Julia, Zig, C++, Node,
# Kotlin, Dart, Lua, Octave, R, PHP, Nim, D, V, Haskell, OCaml, Elixir, Odin):
# its tests and its five examples, against a fresh librlx_capi. A language
# without a local toolchain is skipped, or with RLX_TEST_DOCKER=1 runs in its
# container image (after `just linux-capi-volume`).
test-ffi-bindings:
    cargo test -p rlx-capi --test ffi_languages -- --test-threads=1

# A Linux build of librlx_capi (and the reference model plugin) in the
# `rlx-linux-target` Docker volume, for the container runs of
# test-ffi-bindings (RLX_TEST_DOCKER=1).
linux-capi-volume:
    #!/usr/bin/env bash
    set -euo pipefail
    lock="$(mktemp)"; cp Cargo.lock "$lock"
    docker run --rm -v "$PWD":/src:ro -v "$lock":/src/Cargo.lock -v rlx-linux-target:/target \
        -v rlx-cargo-registry:/usr/local/cargo/registry -w /src -e CARGO_TARGET_DIR=/target \
        rust:latest cargo build -p rlx-capi -p rlx-model-plugin
    rm -f "$lock"

# Package the C ABI as RLX.xcframework (iOS + macOS, Metal) for Swift apps,
# then run the Swift test suite on it. See crates/bindings/rlx-swift/README.md.
swift-xcframework:
    ios/build-xcframework.sh --capi --apple
    RLX_XCFRAMEWORK=1 swift test --package-path crates/bindings/rlx-swift

# Print the curated RLX_* environment catalog (high-signal options only).
env-catalog:
    cargo run -q -p mahbot-rlx-ir --example env_catalog

# rlx-libm's ports against musl's independent algorithms (the real `libm`
# crate). A standalone workspace: rlx patches crates.io's `libm` with
# rlx-libm (`crates/numerics/libm-shim`), so the workspace has only that one.
test-libm-vs-musl:
    CARGO_TARGET_DIR=target/vs-musl cargo test --release --manifest-path crates/numerics/rlx-libm/tools/vs-musl/Cargo.toml

# rlx-libm against glibc 2.43 in every configuration the workspace `cargo test`
# skips: no_std, `export-glibc` (std's float methods are the ports), the
# Apple reference (`libm`), plus the musl second opinion. The hashed vectors
# (`tests/glibc_2_43_vectors.rs`) cover every exported function on both
# x86-64 and aarch64 glibc and fail on an export without vectors.
test-libm: test-libm-vs-musl
    cargo test --release -p rlx-libm
    cargo test --release -p rlx-libm --no-default-features
    cargo test --release -p rlx-libm --features export-glibc
    cargo test --release -p rlx-libm --features libm,export-glibc
    cargo test --release -p libm

# Every one-argument f32 port on all 2^32 inputs against the stored glibc
# hashes (x86-64 and aarch64), on any OS; minutes on every core. Regenerate
# the hashes on Linux with `tools/glibc-exhaustive` (it lists mismatches).
test-libm-exhaustive:
    cargo test --release -p rlx-libm --test f32_exhaustive -- --ignored --test-threads=1 --nocapture

# rlx-fpga once claimed `no_std` while depending on std-only rlx-ir, and no
# gate built it that way, so nothing noticed (GitHub issue #7).
# no_std crates (rlx-ir core, rlx-cortexm, rlx-fpga) build for bare metal + host tests
check-no-std:
    #!/usr/bin/env bash
    set -euo pipefail
    t=thumbv7em-none-eabihf
    rustup target list --installed | grep -qx "$t" || rustup target add "$t"
    cargo clippy -p mahbot-rlx-ir -p rlx-cortexm -p rlx-fpga -p rlx-mcu --no-default-features --target "$t" -- -D warnings
    # `rlx-micro` is checked with its defaults, not without: its halting
    # `#[panic_handler]` is the `panic-handler` default feature, and a bare-metal
    # staticlib without one fails codegen — which is the gate working, not a defect
    # to silence with an allow.
    cargo clippy -p rlx-micro --target "$t" -- -D warnings
    # Host side: every test/example target must still build (std-only ones gate
    # themselves on the `std` feature) and the no_std library tests must pass.
    # rlx-ir's integration tests use the full IR, so only its unit tests run here.
    cargo clippy -p rlx-fpga --no-default-features --all-targets -- -D warnings
    cargo clippy -p mahbot-rlx-ir --no-default-features --lib -- -D warnings
    cargo clippy -p mahbot-rlx-ir --no-default-features --lib --profile test -- -D warnings
    cargo test -p rlx-fpga --no-default-features
    cargo test -p mahbot-rlx-ir --no-default-features --lib

# README Rust examples as doctests (`crates/rlx`'s `ReadmeDoctests`): the
# first code a reader copies must compile and give the numbers it claims.
check-readme:
    cargo test -p rlx --doc --features cpu,tensor,optim

# Known-failures lists (`tests/known_failures/*.txt`, see rlx_ir::known_failures).
#   just known-failures list
#   RLX_KNOWN_FAILURES_REPORT=/tmp/kf.jsonl just test-gpu
#   just known-failures update /tmp/kf.jsonl --issue owner/repo#N
known-failures *args:
    python3 scripts/known-failures.py {{args}}

# Toolchain version matrix (scripts/toolchain_matrix.toml): every supported
# external toolchain version, each on its rig in a fresh TMPDIR.
#   just toolchain-matrix                 # run all, print a table
#   just toolchain-matrix --only xdna     # one device or component
#   just toolchain-matrix --update        # record results in tested_toolchains.tsv
toolchain-matrix *args:
    python3 scripts/toolchain_matrix.py {{args}}

# Exit 1 when an upstream toolchain (mlir-aie, XRT, …) has a release newer than
# anything the matrix has tested — run it nightly / before a release.
toolchain-upstream:
    python3 scripts/toolchain_matrix.py --upstream

# Regenerate docs/known-issues.md from rlx-runtime's known-issue table
# (`doctor/known_issues.rs`); a unit test fails while it is stale.
gen-known-issues:
    cargo run -q -p rlx-check --bin cargo-rlx -- doctor --known --markdown > docs/known-issues.md

# Regenerate docs/rlx-env-vars.md from the env registry (+ leftover mentions).
gen-rlx-env-vars:
    python3 scripts/gen-rlx-env-vars.py

# Named kernel corpus: every family through the device-free gate stack
# (verify -> repr -> memory-plan program-safety/schedule-semantics) under every
# planner configuration a backend uses. Answers "which families did my compiler
# change break", which the per-crate suites cannot.
check-corpus:
    cargo test -p rlx-corpus --features backend-tests -- --nocapture

# The same corpus, executed on every device this host can reach, scored against
# the INDEPENDENT f64 oracle rather than against rlx's own CPU path — a defect
# both share (rms_norm_backward's extra 1/r, in all seven impls at once) is
# invisible to backend-vs-backend parity. Rolled up per family per device.
#
# Backends are opt-in per host because cargo unifies features across a workspace
# build: an Apple-only backend enabled by default here would be enabled for
# every crate on Linux too.
check-corpus-device:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      # `apple` would imply `mlx`, which this fork does not ship, and `vulkan`
      # needs an ICD it does not ship either — so the Apple backends are named
      # one by one. `metal` and `coreml` are not `default-members`; they are here
      # because `test-gpu`'s Darwin row builds them, and the once-per-sync device
      # gate should cover what the routine GPU gate builds.
      Darwin) FEATURES="metal,coreml,gpu" ;;
      *)      FEATURES="gpu" ;;
    esac
    cargo test -p rlx-corpus --features "$FEATURES,backend-tests" --test device_gate -- --nocapture

# Env-registry gate, wired into `just ci`. Fails on any of:
#   * docs/rlx-env-vars.md drifting from the registry
#   * an RLX_* read with no registry entry
#   * an RLX_* read that bypasses the rlx_ir::env shim (std::env::var)
# The last one was a soft note until 208 call sites accumulated under it; the
# shim is what lets a test or an in-process A/B set a knob at all.
check-rlx-env-vars:
    python3 scripts/gen-rlx-env-vars.py --check

# Example-name gate, wired into `just ci`.
#
# Cargo writes every package's examples into one `target/<profile>/examples/`
# directory keyed by the bare target name, so two packages naming an example
# the same thing overwrite each other's binary and a harness run by that path
# measures whichever built last — under both labels. Prefix the example with
# its backend (`wgpu_schedule_matmul_ab`), or state in the script why the two
# can never share a target/ directory.
check-example-names:
    python3 scripts/check-example-names.py

# Markdown link gate, wired into `just ci`.
#
# Relative links rot silently: nothing renders an error, GitHub 404s only on a
# click, and `cargo doc` never reads a README. Regrouping the crates under
# crates/<group>/<crate>/ broke 59 links at once, and no gate noticed. This
# checks that every relative link resolves and that same-file `#anchor` links
# match a real heading.
check-doc-links:
    python3 scripts/check-doc-links.py

# rustdoc gate over every publishable crate, wired into `just ci-workspace`.
#
# `just ci` runs `check-rustdoc-contour` instead — the same script over the
# default contour, which is what the routine loop needs; this one is the
# once-per-sync whole set.
#
# Per-crate, with each crate's own `[package.metadata.docs.rs]` features —
# because that is what docs.rs builds. `cargo doc --workspace` uses ONE unified
# (default) feature set instead, so every intra-doc link into a feature-gated
# item reads as "unresolved": 26 of 28 such warnings were phantoms, and the two
# real ones were buried in them. Skips `publish = false` crates; docs.rs never
# builds those.
#
# Usage: `just check-rustdoc` or `just check-rustdoc rlx-tensor` for one crate.
check-rustdoc *CRATES:
    python3 scripts/check-rustdoc.py --keep-going {{CRATES}}

# The same gate over the fork's own build set, which is what `just ci` runs: the
# contour is what this fork maintains, and the twenty-one members upstream ships
# that it does not build have no reason to hold up a routine run. `--contour`
# reads `default-members` from cargo rather than repeating the list here.
check-rustdoc-contour:
    python3 scripts/check-rustdoc.py --contour --keep-going

# Dual-license header gate, wired into `just ci`.
#
# The writer half (`just license-headers`) existed; the gate did not, so the
# only thing keeping headers on new files was someone remembering to run it.
# Nine files had drifted — five tests/examples with no header and three docs
# with no `## License` footer.
check-license-headers:
    python3 scripts/add_license_headers.py --check

# Add any missing dual-license headers / footers in place.
license-headers:
    python3 scripts/add_license_headers.py

# Print the checklist / stub paths for adding a new Op (does not edit the tree).
# Usage: `just new-op MyOp` or `just new-op MyOp --write` to create empty stub files.
new-op NAME *ARGS:
    python3 scripts/new-op.py {{NAME}} {{ARGS}}

# Device-free WGSL gate: no kernel may reach a `workgroupBarrier()` from
# non-uniform control flow. naga → Metal/Vulkan accepts that shape, so a native
# run on those adapters passes even though the barrier is non-uniform; FXC (the
# DX12 target) rejects the whole entry point as X4026. Device-free by
# construction: it reads the shipped WGSL, it does not dispatch it.
check-wgsl-uniform:
    cargo test -p mahbot-rlx-wgpu --features backend-tests --test wgsl_uniform_control_flow

# Cross-compile gate: the Apple on-device stack must build for every Apple
# platform, device *and* simulator. The native backends are rlx-cpu
# (Accelerate/AMX), rlx-metal (Metal + MPS + MPSGraph) and rlx-coreml (ANE) —
# each compiles the *real* backend, not the non-Apple stub. Platform support
# matrix:
#   macOS / iOS / tvOS / visionOS → CPU + Metal + CoreML
#   watchOS                        → CPU/Accelerate only (no Metal API; CoreML
#                                    runtime model-compilation is unavailable)
#
# Compile-only. `just test-apple-sim` is the gate that actually *runs* on the
# simulators, and `ios/build-xcframework.sh` is what packages a build for a
# device.
#
# `check-ios` kept as an alias for the common iPhone/iPad case.
check-ios: check-apple
check-apple:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}

    # iOS has shipped a prebuilt std for years; tvOS, watchOS and visionOS only
    # got one in a recent stable. Use it where rustup has it and fall back to
    # building std from source on nightly where it does not, so this gate means
    # the same thing on either toolchain rather than quietly not running.
    apple_check() {
        local target="$1"; shift
        echo "==> $target ($*)"
        if rustup target add "$target" >/dev/null 2>&1; then
            cargo check -p mahbot-rlx-runtime "$@" --target "$target"
        else
            echo "note: no prebuilt std for $target — building std from source (nightly)" >&2
            rustup toolchain install nightly >/dev/null
            rustup component add rust-src --toolchain nightly >/dev/null
            cargo +nightly check -Zbuild-std -p mahbot-rlx-runtime "$@" --target "$target"
        fi
    }

    # Device targets carry the widest surface: the `apple` umbrella adds wgpu
    # (Metal-on-Apple) and MLX on top of CPU + Metal + CoreML.
    #
    # MLX on the device targets is the expensive part of this gate — it
    # cross-compiles libmlx (a large C++ tree) per platform — and it is also the
    # part that earns its keep. Every MLX defect on these platforms was invisible
    # to a `cpu,metal,coreml` check and showed up the moment libmlx was actually
    # built: a metallib stamped for macOS, `bfloat` missing below the Metal 3.1
    # floor, and `system()`/`popen()` being unavailable on tvOS and visionOS.
    for t in aarch64-apple-ios aarch64-apple-tvos aarch64-apple-visionos; do
        apple_check "$t" --features apple
    done

    # Simulators: CPU + Metal + CoreML. The simulator slices are exercised for
    # real by `just test-apple-sim`, and MLX is excluded there for the reason
    # that recipe documents — a headless `simctl spawn` has no Metal device.
    for t in aarch64-apple-ios-sim aarch64-apple-tvos-sim aarch64-apple-visionos-sim; do
        apple_check "$t" --no-default-features --features cpu,metal,coreml
    done

    # watchOS: CPU/Accelerate only — no Metal API, no CoreML runtime compile.
    for t in aarch64-apple-watchos aarch64-apple-watchos-sim; do
        apple_check "$t" --no-default-features --features cpu
    done

# Gates keyed on the WHOLE op set, not the subset some test happened to build.
#
# `rlx-ir`'s `test-support` feature exposes `sample_ops`: one constructible
# `Op` per `OpKind`, behind a `match` that fails to compile when an op is
# added. That is what lets a gate cover all 187 kinds — `rlx-corpus` reaches
# 13. The feature is off by default (it is scaffolding, not IR surface), so
# nothing else in the tree runs these.
check-ops:
    cargo test -p mahbot-rlx-ir --features test-support,backend-tests --test arity_exhaustive
    # Every op through `.rlxmodel` text, and the opset 1 fixture still reads.
    cargo test -p mahbot-rlx-ir --features test-support,serialize,backend-tests --test rlxmodel_every_op
    # Declared arity vs. what backends actually index — the only check here
    # that compares the table against reality rather than against itself.
    # `--lib` runs the scanner's own regression tests: every early version of
    # it passed while silently finding nothing, so the parser is pinned too.
    cargo test -p rlx-check --features op-gates --lib op_scan
    cargo test -p rlx-check --features op-gates,backend-tests --test backend_operand_indexing
    cargo clippy -p mahbot-rlx-ir --features test-support,backend-tests --all-targets -- -D warnings
    cargo clippy -p rlx-check --features op-gates,backend-tests --all-targets -- -D warnings

# Desktop coordinator for the mobile node demos. Defaults to `--self-test`,
# which runs the worker ranks here on loopback — so the desktop half is
# verifiable before a handset is involved.
#
#   just demo-coordinator                                  # loopback self-test
#   just demo-coordinator "--world 2 --peers 0.0.0.0:29500" # wait for a phone
demo-coordinator ARGS="--world 2 --self-test":
    cargo run -p rlx-ffi --example node_coordinator -- {{ARGS}}

# Build the iOS node demo app (xcframework → XcodeGen project → simulator
# build). Needs Xcode + `brew install xcodegen`.
demo-ios:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    ios/build-xcframework.sh
    cd ios/Demo
    xcodegen generate
    xcodebuild -project RlxDemo.xcodeproj -scheme RlxDemo \
        -sdk iphonesimulator -configuration Debug \
        CODE_SIGNING_ALLOWED=NO build
    echo "open ios/Demo/RlxDemo.xcodeproj, or install the .app with simctl"

# Build the Android node demo APK (cross-build the .so, then Gradle).
# Honors ANDROID_HOME / JAVA_HOME if already set; otherwise probes Homebrew.
demo-android:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}/android
    ./build.sh
    brew_prefix="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    export ANDROID_HOME="${ANDROID_HOME:-$brew_prefix/share/android-commandlinetools}"
    if [[ -z "${JAVA_HOME:-}" ]]; then
      for jdk in "$brew_prefix/opt/openjdk@17" "$brew_prefix/opt/openjdk"; do
        [[ -d "$jdk" ]] && export JAVA_HOME="$jdk" && break
      done
    fi
    if [[ ! -d "$ANDROID_HOME" ]]; then
      echo "Android SDK not found; set ANDROID_HOME." >&2; exit 1
    fi
    ./gradlew assembleDebug --console=plain
    echo "APK: android/app/build/outputs/apk/debug/app-debug.apk"

# Cross-compile gate: a distributed **node** must build for every non-desktop
# target we claim can join a mesh. Nodes are what make a cluster heterogeneous
# — a phone, an SBC, or the host driving an FPGA board is a rank like any
# other — and nothing else in the tree compiles the node stack off-desktop, so
# without this gate that support regresses silently.
#
# Layers, innermost first:
#   rlx-driver + rlx-collectives  transport + in-graph collectives (no platform gating)
#   rlx-runtime (dist::node)      the node driver + ship-graph worker
#   rlx-ffi                       C ABI shell (Apple xcframework, embedded C hosts)
#
# Android is gated on the NDK being installed: `cargo check` still runs C build
# scripts (zstd/bzip2 via the GGUF reader), which need the NDK's clang. The
# recipe reports the skip rather than passing quietly or failing on a machine
# that simply has no NDK.
check-nodes:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}

    # Every Apple OS, device slice. These are what ios/build-xcframework.sh
    # packages, so a break here is a break in the shipped xcframework. The
    # simulator slices are covered by `just test-apple-sim`, which runs on them.
    for t in aarch64-apple-ios aarch64-apple-tvos aarch64-apple-watchos aarch64-apple-visionos; do
      echo "==> Apple node ($t)"
      if ! rustup target add "$t" >/dev/null 2>&1; then
        echo "SKIPPED: no prebuilt std for $t in this toolchain." >&2
        echo "         That platform's node support was NOT verified by this run." >&2
        continue
      fi
      cargo check -p mahbot-rlx-driver -p rlx-collectives --target "$t"
      cargo check -p mahbot-rlx-runtime --no-default-features --features cpu --target "$t"
      # staticlib only: a cdylib must resolve every symbol at link time and
      # zstd-sys does not on these targets. This is the archive the
      # xcframework wraps.
      cargo build -p rlx-ffi --lib --target "$t"
    done

    echo "==> Android (aarch64-linux-android)"
    if ndk_env="$(android/ndk-env.sh 2>/dev/null)"; then
      eval "$ndk_env"
      rustup target add aarch64-linux-android >/dev/null
      cargo check -p mahbot-rlx-driver -p rlx-collectives --target aarch64-linux-android
      cargo check -p mahbot-rlx-runtime --no-default-features --features cpu --target aarch64-linux-android
      ( cd android && cargo check -p rlx-jni --target aarch64-linux-android )
    else
      echo "SKIPPED: no Android NDK (set ANDROID_NDK_HOME or ANDROID_HOME)." >&2
      echo "         Android node support was NOT verified by this run." >&2
    fi

    echo "==> node driver tests (host)"
    cargo test -p mahbot-rlx-runtime --no-default-features --features cpu,backend-tests --test dist_node_fixed_function
    cargo test -p rlx-ffi

# Run the Apple backend smoke + parity tests ON the simulators — iOS, tvOS,
# watchOS and visionOS. Real on-simulator execution, not just a cross-compile:
# scripts/apple-sim-runner.sh reads each test binary's Mach-O platform, boots a
# simulator of that family (override the device with RLX_SIM_DEVICE=<name|udid>)
# and runs the binary inside it via `simctl spawn`.
#
# Needs Xcode plus the simulator runtime for each platform; a platform whose
# runtime is not installed fails with the runner's own message naming it.
#
# Backends: cpu,metal,coreml. MLX is intentionally excluded from the *sim test*:
# a headless `simctl spawn` exposes no Metal device (so MLX/Metal can't run
# there anyway), and statically linking MLX pulls in newest-SDK Metal symbols
# (MTLTensor) that need a high link deployment target. MLX-on-iOS compile is
# covered by `just check-apple` (the `apple` umbrella includes mlx) + the host
# parity test in this same file.
#
# watchOS runs `apple_platform_sim` only: it has no Metal API and no CoreML
# runtime-compile path, so the accelerator parity test is compiled out there
# and the CPU floor is the whole of what there is to check.
#
# `just test-apple-sim apple` re-runs everything under the size-tuned `apple`
# profile — the one `ios/build-xcframework.sh` ships. That is a *precision*
# gate, not a second smoke test: fat LTO and one codegen unit change how the
# numeric kernels are inlined, and the point is that they must not change what
# the kernels compute. `apple_platform_sim` asserts hand-computed values and
# `apple_backends_sim` compares ANE against CPU, so a drift shows up as a
# failure rather than as a plausible-looking number in production.
#
# Run the backend smoke + parity tests on all four Apple simulators.
test-apple-sim profile="dev":
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}

    runner={{justfile_directory()}}/scripts/apple-sim-runner.sh
    export CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUNNER="$runner"
    export CARGO_TARGET_AARCH64_APPLE_TVOS_SIM_RUNNER="$runner"
    export CARGO_TARGET_AARCH64_APPLE_WATCHOS_SIM_RUNNER="$runner"
    export CARGO_TARGET_AARCH64_APPLE_VISIONOS_SIM_RUNNER="$runner"

    for t in aarch64-apple-ios-sim aarch64-apple-tvos-sim aarch64-apple-visionos-sim; do
        echo "==> $t (cpu,metal,coreml, profile {{profile}})"
        rustup target add "$t" >/dev/null
        cargo test --profile {{profile}} -p mahbot-rlx-runtime \
            --no-default-features --features cpu,metal,coreml,backend-tests \
            --target "$t" --test apple_backends_sim --test apple_platform_sim \
            -- --nocapture --test-threads=1
    done

    echo "==> aarch64-apple-watchos-sim (cpu, profile {{profile}})"
    rustup target add aarch64-apple-watchos-sim >/dev/null
    cargo test --profile {{profile}} -p mahbot-rlx-runtime --no-default-features --features cpu,backend-tests \
        --target aarch64-apple-watchos-sim --test apple_platform_sim \
        -- --nocapture --test-threads=1

# Android cross-compile gate — CPU (NEON) + wgpu via the `android` feature.
# Needs NDK (ANDROID_NDK_HOME / ANDROID_HOME) for C deps (bzip2-sys, etc.).
android-check:
    #!/usr/bin/env bash
    set -euo pipefail
    rustup target add aarch64-linux-android
    eval "$("{{justfile_directory()}}/android/ndk-env.sh")"
    cargo check -p mahbot-rlx-runtime --no-default-features --features android --target aarch64-linux-android
    cargo check -p rlx --no-default-features --features android --target aarch64-linux-android
    cargo check --manifest-path "{{justfile_directory()}}/android/rlx-jni/Cargo.toml" --target aarch64-linux-android

# Cross-build librlx_jni.so for the Android demo app (stages under jniLibs/).
# Pass --blas after ./android/build-openblas.sh for static OpenBLAS.
android-build *ARGS:
    {{justfile_directory()}}/android/build.sh {{ARGS}}

# End-to-end Android gate (emulator + instrumented tests). See android/e2e.sh.
android-e2e *ARGS:
    {{justfile_directory()}}/android/e2e.sh {{ARGS}}

# Verbose run — exposes [rlx] / [ktrace] log lines.
run-verbose CMD:
    RLX_VERBOSE=1 {{CMD}}

# The routine gate: this fork's default contour (`default-members` in
# Cargo.toml) plus the gates that only exist here. `build`, `test` and `lint`
# name no package, so they cover the contour and not the twenty-one upstream
# members this fork deliberately does not build. `check-corpus` and
# `test-third-order-cpu` stay here because every package they name *is* a default
# member — narrowing the contour was about the members this fork does not build,
# and neither drags any of them in. `check-rustdoc-contour` is the same gate over
# the contour rather than all eighty-four publishable crates. The Python binding
# is outside all of it: `crates/bindings/pyrlx` is `exclude`d from the workspace,
# so nothing builds it and no gate runs its tests — though gates do read files
# inside it, and `just torch-import` runs its pure-Python front-end (the `exclude`
# comment in Cargo.toml is the list).
# `ci-workspace` is the whole-set gate, run once per upstream sync (see
# `docs/fork-sync.md`).
ci: build test fmt-check lint check-fork-layers check-rlx-env-vars check-example-names check-doc-links check-license-headers check-wgsl-uniform check-op-coverage check-corpus test-third-order-cpu check-rustdoc-contour
# Everything, once per upstream sync: the routine gate above plus every member
# upstream ships — the Apple/NVIDIA/AMD/Intel/TPU/embedded backends, the no_std
# crates and the bindings. Slow and toolchain-dependent, which is why the routine
# loop above stays narrow. It depends on `ci` so it cannot fall behind it, and it
# names `test-gpu` because that is the recipe that builds the targets behind
# `required-features` — a bare `cargo test --workspace` skips every one of them.
# `test-workspace` names the one feature it has to (`mahbot-inference/hand-run`,
# see that recipe); every other gated set is `test-gpu`'s.
#
# Every recipe named here needs nothing beyond the Rust toolchain and the
# contour's own devices, so the gate is green on whatever host a sync runs on —
# `RLX_REQUIRE_DEVICE=1` makes a backend that skips for want of a device a
# failure, not a pass. The recipes that need something else are in `ci-rig`: HIP,
# `leaks`, `xcrun metal`, a crates.io fetch for the real `libm`, and a Vulkan ICD.
ci-workspace: ci build-workspace test-workspace lint-workspace check-no-std check-readme check-rustdoc check-corpus-device check-bindings check-binding-versions test-third-order-gpu test-gpu
# The whole-set gate's toolkit rows, run on a rig rather than a developer machine:
# HIP for `test-rocm`, `leaks` for `leak-check`, `xcrun metal` for
# `validate-metal-bindings`, an ICD for `test-vulkan`, and a crates.io fetch for
# `test-libm` — its comparison workspace depends on the real `libm`, which the
# root `[patch.crates-io]` shadows inside the workspace itself. `ci-workspace`
# does not name any of them — a gate that cannot pass on the machine it is run on
# teaches people to ignore gates.
ci-rig: ci-workspace test-vulkan test-libm test-rocm leak-check validate-metal-bindings
# The two layers this fork re-applies on every sync — the package rename and the
# narrow test contour — checked rather than trusted (see `docs/fork-sync.md` §2/§3).
check-fork-layers:
    python3 scripts/fork-rename.py --check
    python3 scripts/fork-test-gating.py --check

# ROCm compile check + graph_devices parity (tests skip when HIP unavailable).
#
# Sets no `require_device` itself: the compile check and the lib tests pass where
# no AMD GPU exists, which is every developer machine. On a Linux rig `test-gpu`
# calls this with `RLX_REQUIRE_DEVICE=1` already exported, so a row that skips for
# want of a device fails there instead of reporting `ok`.
test-rocm:
    cargo check -p mahbot-rlx-runtime --features cpu,rocm
    cargo test -p rlx-rocm --lib
    # Kernel argument-count check on. HIP reads exactly as many pointers as the
    # kernel declares and cannot see how many were passed, so a mismatch is
    # silent — this repo shipped one (see `gguf_gpu::launch_dequant_gguf`).
    RLX_GPU_VALIDATE_PARAMS=1 cargo test -p rlx-rocm --features backend-tests
    cargo test -p mahbot-rlx-runtime --features cpu,rocm,backend-tests --test graph_devices_parity
    cargo test -p mahbot-rlx-runtime --features cpu,rocm,backend-tests \
        --test rocm_op_parity --test rocm_gpu_custom_op

# HIP-CPU kernel validation (linux-gnu Docker only). Clones HIP-CPU into docker/vendor/.
test-hip-cpu-validate:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    docker build -f "$root/rlx-cuda/docker/Dockerfile.hip-cpu-validate" -t rlx-hip-cpu-validate "$root"
    docker run --rm -v "$root:/work" -w /work rlx-hip-cpu-validate \
        bash -c 'set -euo pipefail
            bash rlx-cuda/docker/fetch-hip-cpu.sh
            cargo test -p rlx-cuda --features hip-cpu-validate,backend-tests
            cargo test -p rlx-rocm --features hip-cpu-validate,backend-tests'

# Higher-order AD, CPU only — no device, so this is part of the routine gate
# (`just ci`) rather than the GPU parity run below it.
test-third-order-cpu:
    cargo test -p mahbot-rlx-runtime --release --features cpu,backend-tests --test nth_order_grad

# Higher-order AD: GPU parity (Apple backends on macOS, wgpu elsewhere).
test-third-order-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      Darwin)
        # `apple` would be the shorter spelling, but it implies `mlx` and this
        # fork does not ship the MLX backend; the per-device suites are
        # `cfg`-gated on `metal`/`mlx` anyway, so naming the backends is what
        # selects them.
        cargo test -p mahbot-rlx-runtime --release --features cpu,metal,gpu,coreml,backend-tests \
          --test higher_order_low_precision_parity
        cargo test -p mahbot-rlx-runtime --release --features cpu,metal,gpu,coreml,backend-tests \
          --test third_order_gpu_parity --test directional_nth_gpu_parity
        cargo test -p mahbot-rlx-runtime --release --features cpu,metal,gpu,coreml,backend-tests \
          --test higher_order_decompose_parity -- --test-threads=1 ;;
      *)
        cargo test -p mahbot-rlx-runtime --release --features cpu,gpu,backend-tests \
          --test higher_order_low_precision_parity
        cargo test -p mahbot-rlx-runtime --release --features cpu,gpu,backend-tests \
          --test third_order_gpu_parity --test directional_nth_gpu_parity
        cargo test -p mahbot-rlx-runtime --release --features cpu,gpu,backend-tests \
          --test higher_order_decompose_parity -- --test-threads=1
        if cargo check -p mahbot-rlx-runtime --features cpu,cuda,gpu --quiet 2>/dev/null; then
          cargo test -p mahbot-rlx-runtime --release --features cpu,cuda,gpu,backend-tests \
            --test higher_order_decompose_parity -- --test-threads=1
        fi
        if cargo check -p mahbot-rlx-runtime --features cpu,rocm --quiet 2>/dev/null; then
          cargo test -p mahbot-rlx-runtime --release --features cpu,rocm,backend-tests \
            --test rocm_op_parity --test graph_devices_parity
          cargo test -p mahbot-rlx-runtime --release --features cpu,rocm,backend-tests \
            --test third_order_gpu_parity --test directional_nth_gpu_parity \
            --test higher_order_low_precision_parity
          cargo test -p mahbot-rlx-runtime --release --features cpu,rocm,backend-tests \
            --test higher_order_decompose_parity -- --test-threads=1
        fi ;;
    esac

# Update the Cargo.lock (pinned dep refresh; commit the lockfile).
update-lock:
    cargo update --workspace

# Run a CPU kernel micro-bench (plan #52). `just micro sgemm`.
micro NAME:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cargo bench -p mahbot-rlx-cpu --bench {{NAME}}

# Run all CPU kernel micro-benches.
micro-all:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cargo bench -p mahbot-rlx-cpu

# ── rlx-wgpu portability validation ─────────────────────────────────
#
# macOS only ever exercises wgpu's Metal backend and wgpu-native's WGSL
# superset. This target covers the other surface the crate ships to.

# Builds for the host arch, so it runs native (not emulated) on Apple silicon.
# Found the zero-extent Expand divide-by-zero that Metal cannot reach.
# Run the rlx-wgpu suite on a real Linux Vulkan ICD (Mesa lavapipe) in Docker.
test-wgpu-linux:
    docker build -q -t rlx-wgpu-vk {{justfile_directory()}}/crates/backends/rlx-wgpu/docker
    docker run --rm \
        -v "{{justfile_directory()}}:/rlx" \
        -v rlx-vk-target:/target \
        rlx-wgpu-vk \
        cargo test -p mahbot-rlx-wgpu --features backend-tests --tests --no-fail-fast -- --test-threads=1

# Portability gate for the CPU backend: ONE binary has to run on the AVX-512
# host or M4 that built it *and* on an Atom box or a Raspberry Pi. That only
# holds while every above-baseline instruction sits inside a function reached
# through a runtime CPU-feature check — `cargo build` enforces nothing, and the
# failure mode is a bare `Illegal instruction` on hardware the author never
# sees.
#
#   just check-isa                                 # emulated Atom (x86-64)
#   just check-isa arm                             # emulated ARMv8.0 (Pi 3/4)
#   just check-isa scan target/release/cargo-rlx   # static scan, no Docker
#
# `atom` (the default) and `arm` need Docker. `--platform linux/amd64` supplies
# a real x86-64 Linux toolchain (Rosetta-backed on Apple Silicon, so near-native
# build speed) and `qemu-user-static -cpu Denverton` inside it emulates a
# Goldmont Atom that traps AVX; `arm` uses `linux/arm64` (fully native here)
# with `-cpu cortex-a53`, which has neither DotProd nor FP16 arithmetic.
#
# `scan` needs only objdump. It judges a binary against the baseline of the
# target it was BUILT for — `sdot` is baseline on aarch64-apple and a finding
# on aarch64-linux — and also catches the inverse problem, a
# `-C target-cpu=native` artifact that smears AVX across ordinary symbols
# (`Op::clone` included) where no runtime dispatch can rescue it.
check-isa *ARGS:
    python3 crates/backends/rlx-cpu/tools/isa_portability.py {{ARGS}}

# PS5 payloads (jailbroken console, ps5-payload-dev SDK). A standalone crate
# with its own workspace and pinned nightly toolchain, never part of the
# workspace build or `ci`. `just ps5 setup` · `just ps5 build` ·
# `just ps5 run rlx --host <ps5-ip>`. See crates/backends/rlx-ps5/README.md.
ps5 *ARGS:
    python3 crates/backends/rlx-ps5/scripts/ps5.py {{ARGS}}

# ── mahbot: this fork's own models ──────────────────────────────────
#
# A deliberate build renders the clip corpus; the measurements only read it. All
# three are behind the crate's non-default `bench` feature.
mahbot-wake-build *ARGS:
    cargo run --release -p mahbot-inference --features bench --bin wake-word-build -- {{ARGS}}

mahbot-wake-bench *ARGS:
    cargo bench -p mahbot-inference --features bench --bench wake_word -- {{ARGS}}

# The pinned real-audio corpus through the product's transcription call, one row
# per recording, each in its own process; writes TRANSCRIPTION-CORPUS.md.
mahbot-transcription-corpus *ARGS:
    cargo run --release -p mahbot-inference --features bench --bin transcription-corpus -- {{ARGS}}

# No other recipe compiles the bench target (whose `harness = false` means an
# executing gate would run a measurement); the hand-run targets are also in
# `just lint`, and this recipe is the crate's whole set under `-D warnings`.
lint-mahbot:
    cargo clippy -p mahbot-inference --features bench,hand-run --all-targets -- -D warnings

# The product checks the stand never runs: every one is `#[ignore]`d, so no
# routine command reaches them. A check whose input is absent prints why and
# returns, so this is safe on a machine with no installed tree.
#
# Deliberately not in `ci` or `ci-workspace`: a product check is a claim about
# behaviour, and one of them (`a_paste_onset_lands_within_one_detector_hop`) is a
# known-stale expectation the cut policy superseded — a gate that is red by
# construction teaches the operator to skip the gate. A sync runs this by hand
# (`docs/fork-sync.md` §10) and reads what it says.
test-product-checks:
    cargo test --release -p mahbot-inference --features hand-run --test product_checks -- --ignored --nocapture
