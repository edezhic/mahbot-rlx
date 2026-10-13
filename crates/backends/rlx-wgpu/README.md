# rlx-wgpu

Cross-platform GPU backend via the [wgpu](https://wgpu.rs/) crate.
Single backend serves Metal (macOS), Vulkan (Linux, and Windows
alongside DX12), DirectX 12 (Windows), and WebGPU (browsers). WGSL
kernels, pure Rust deps — no FFI, no submodules.

## What's here

- **WGSL kernels** — fp32 matmul (8×8 tile), cooperative-matrix
  matmul (32×32 tile, `simdgroup_matrix` / `KHR_cooperative_matrix`),
  f16-storage matmul.
- **`device.rs`** — wgpu instance/adapter/device singleton. Platform
  defaults: **Vulkan and DX12 both enabled on Windows**, with no preference of
  this crate's own — wgpu registers Vulkan before DX12 and sorts adapters only
  by device type, so DX12 is reached only when no Vulkan adapter survives.
  Vulkan on Linux/WSL, Metal (+ MoltenVK) on macOS. Override with
  `WGPU_BACKEND`. Sync wrapper via `pollster::block_on` so the rest of the
  backend matches the rlx-cpu / rlx-metal / rlx-mlx synchronous shape.
- **`buffer.rs` / `Arena`** — logical activation arena with per-node offsets
  from `plan_f32_uniform`. When the plan exceeds wgpu `max_buffer_size`
  (~4 GiB), the arena is **striped** across multiple ≤4 GiB buffers
  (`extra_shards`); each stripe keeps a stage reserve (`SHARD_STAGE_RESERVE`: a
  768 MiB constant, which `RLX_WGPU_SHARD_STAGE_MIB` can only lower) so
  cross-shard operand copies do not clobber live activations. A stripe's usable
  region is its size minus that reserve (with a 256-byte floor), and the snap
  that fits a compact plan into one buffer may halve the reserve while it tries.
  f32 host I/O via `queue.write_buffer` / pooled MAP_READ staging readback
  (`ReadbackStaging`, `read_f32_many_pooled`). Env: `RLX_WGPU_SHARD_LOG=1`.
  Packed quant weights may live in a separate `weight_buffer` (`from_plan_split`).
  Physical striping is **unsupported and opt-in only** (`RLX_WGPU_ALLOW_SHARD=1`,
  which warns uncutably on every striping decision), and the stripe count is
  bounded (`MAX_SHARDS`): a shrunken `RLX_WGPU_SHARD_CAP_MIB` that would split an
  arena into more stripes than that is refused loudly at plan time, before the buffers
  it bounds are created, instead of compiling its way through the machine. The
  engine does keep one class honest under it: an op whose kernel carries absolute
  arena offsets (`supported_ops::addresses_arena_by_absolute_offset`) is hosted by
  `should_host_at_ceiling` whenever the memory it addresses cannot bind at base 0
  — the activation arena, or a parameter parked in the separate weight buffer —
  and `WgpuExecutable::ceiling_hosted_steps()` reports that route. Two carry: a
  carry-bearing `Lstm`/`Gru`/`Rnn` on the narrow kernel is refused loudly instead,
  because the host route cannot write its state back into an input, while the wider
  form keeps its own whole-arena host step — which is refused too, for a parameter
  parked in the weight buffer or an arena past the step's 32-bit byte offsets; and
  on an adapter that parks every parameter (`from_plan_split` under a small
  `max_buffer_size`, or `RLX_WGPU_ALL_PARAMS_WEIGHT`), a listed op that reads a
  parameter hosts on every run, ceiling or not — unless that parameter is one the
  host route cannot stage (not one f32 lane per element, in what the upload stored or
  in the slot it was given: `buffer::unstageable_reason`), which is refused
  loudly, naming the operand and where it lives — and so is the *other* host route, the
  structural steps (`Expand`/`Narrow`/`Transpose`/`Concat`, `scan_host::assert_read_fits`),
  whose descriptors carry element counts rather than shapes: they ask the slot half alone,
  though the refusal they raise still names the operand's declared encoding, read from the
  arena's parameter record — and names none where the record cannot attribute one, since
  the record is keyed by byte offset. That key is also why not every operand is judged: an
  operand is measured against the record only where the graph — or, at run time, the flag
  the host step recorded — says it is a parameter's storage, since an activation a plan put
  on a released slot starts where a parameter's storage did. Six arms ask that question
  themselves instead: `Op::DequantMatMul`'s quantized sub-path refuses a packed
  weight rather than staging it (its GGUF and Int8 sub-paths keep their own route, but
  every other scheme's step reads the weight through the f32-lane staging),
  `Op::Fft`/`Op::FftQ` refuse a parked operand, which neither their GPU nor their host
  step can address, and the whole-arena host steps this backend still keeps for
  `Op::LayerNorm2d` and for `Op::Reverse` refuse a parked operand and an arena past
  their 32-bit byte offsets too (the flip also refuses a declared element width the
  arena does not store) — and so does `Op::ConvTranspose2d`'s host route, whose step
  carries 32-bit byte offsets into that arena while its compact form stages the weight
  from there. An absolute-offset arm
  that is not classified there and binds through the windowed path is refused at
  bind time instead of read through a re-based window. `RLX_WGPU_SHARD_GPU=1`
  cannot undo any of it.
- **`kernels/matmul.wgsl`** — fp32 matmul, one workgroup per 8×8 output
  tile. Functional, not optimized.
- **`kernels/mod.rs`** — `OnceLock`-cached pipeline + bind-group layout.
  First dispatch pays the WGSL → SPIR-V/MSL/HLSL translation cost
  (~ms); subsequent dispatches reuse the compiled pipeline.
- **`backend/`** — `WgpuExecutable`, decomposed into `compile`/`run`/`set`/
  `dispatch` submodules (each a sibling `impl WgpuExecutable`; see `mod.rs`).
  Unsupported ops panic at compile time with a clear "fall back to
  CPU/Metal/MLX" diagnostic.
- **GGUF dequant** — `kernels/dequant_gguf.wgsl`, `dequant_gemv_gguf.wgsl`,
  `dequant_gemm_gguf.wgsl` + `gguf_gpu.rs`: `select_path` picks the fused
  windowed kernels for Q4_K/Q6_K/Q1_0 (scratch-free, shard-safe). At `m == 1`
  Q4_K/Q6_K take a tiled per-row GEMV arm (one workgroup per two output columns,
  its lanes splitting each column's contraction over 32-element units of the
  packed weight, one X read four elements at a time shared across them)
  while Q1_0 keeps one thread per column on the scalar X view; for `m > 1`
  Q4_K/Q6_K **and Q8_0** a row-grouped GEMM reads each packed weight tile's quant
  bytes once per group of 8 rows instead of once per row — and that GEMM's Q4_K
  entry, when a dispatch would be too small to fill the GPU, covers half the
  columns per workgroup with two threads to each instead, splitting the contraction
  across them and reading each block's 16-byte header twice. Q8_0 has no one-row
  arm, so it takes the scratch slab at `m == 1` and at any multi-row shape the GEMM
  declines. Other schemes dequant into scratch then `matmul_bt` when the arena is
  a single buffer; sharded / split-weight arenas fall back to `gguf_host`. Grouped
  MoE GGUF uses the GPU path when scratch fits and the arena is unsharded. Scheme
  ids match Metal/CUDA. Details:
  [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
- **FFT** — `fft_gpu.wgsl` multi-kernel pow-2 dispatch, all stages in one
  compute pass with a **per-stage uniform pool** (each stage binds its own
  slot; a *shared* uniform would alias across stages because `write_buffer`
  lands before the pass runs). Batch (`outer`) spills across the y/z grid dims
  past wgpu's 65535 per-dimension cap. Non-pow2, **sharded** arenas and arenas past
  the storage-binding limit use `fft_host.rs` (host sync); f64 / C64 are rejected —
  the wgpu arena is f32-only.
  `RLX_BENCH_DISPATCH_ONLY=1` skips output readback for micro-benchmarks.
  `RLX_DISPATCH_REPORT=1` logs `fft_gpu` vs `fft_host` step counts.

## Op coverage

Broad inference surface (matmul, norms, attention, conv, vision, RNN/SSM,
quantized matmul, etc.) — see [docs/op-coverage.md](../../../docs/op-coverage.md).
GGUF `DequantMatMul` follows the `select_path` ladder — the fused windowed
kernels, or `dequant_gguf` + `matmul_bt` when arena scratch fits; otherwise
CPU dequant via `gguf_host`. MoE grouped GGUF remains host-only on WGPU.

## Install

```toml
[dependencies]
rlx-wgpu = "0.2"
```

Or via [`rlx`](https://crates.io/crates/rlx)'s `gpu` feature.

## Cost-model calibration

When a wgpu adapter is available, `rlx_wgpu::calibrate::Calibration::load_or_measure()`
benchmarks a 512³ matmul and caches results at `~/.cache/rlx/wgpu-calib-<adapter>.json`.
Feeds `WgpuCostModel` in `rlx-runtime` for backend ranking.

## Cooperative-matrix matmul (Vulkan / Metal)

This engine exposes **no cooperative matrices on DX12 at all** — the DX12
adapter reports an empty property list and its shader compilers cannot emit
cooperative-matrix code — so every cooperative path below is Vulkan or Metal.

The 16-bit path is **opt-in and off by default**: on a discrete Vulkan
adapter, aligned f32 matmul with Param weights selects **CoopF16Vk** only with
`RLX_WGPU_COOP_F16_VK_ENABLE=1`, and only when the adapter reports 16×16 f16
cooperative-matrix support:

- **`matmul_coop_f16_vulkan.wgsl`** — 16×16 f16 tensor-core tiles,
  `coopLoad` on A, and either `coopLoadT` or `coopLoad` on B — which of the two
  forms computes `a·b` is measured per device by `coop_probe`, with the size rule
  (the `_widen` form for N > 768) only a tie-break. By default N > 768 dispatches
  the wide f32 path instead (see auto wide fallback).
- **`matmul_qkv_coop_f16_vk.wgsl`** — split-write Q/K/V variant (same
  numerics, one dispatch instead of matmul + Narrow×3).
- **Auto wide fallback** — at `set_param`, an oscillation score on Param B
  selects wide f32 matmul for stress weights; **N > 768** also defaults to wide
  f32 because RTX coop B-load at large N stays ~4e-3 vs f16-ref.
  Opt back into coop for large N with `RLX_WGPU_COOP_F16_VK_LARGE_N=1`.
- Fallback wide path: the vectorised **`matmul_wide_vec4.wgsl`**. The 64×64
  **`matmul_wide_nv.wgsl`** is behind `RLX_WGPU_WIDE_NV=1`, and
  **`matmul_wide.wgsl`** behind `RLX_WGPU_WIDE_STD=1`.

When a matmul operand is an `Activation`, the lowering **host-mirrors** it
each `run()` (`apply_activation` + `write_f32`, same f16 shadow as Params)
and routes the matmul through the **wide f32** path. CoopF16Vk
remains on Input/Param-only operands (BERT QKV weights, etc.). A GPU
`cast_f32_to_f16` pre-pass still handles non-unary computed tensors when
CoopF16Vk applies. Pass flushes between unary → cast → coop matmul keep the
discrete-adapter visibility correct.

### Environment flags

| Variable | Effect |
|----------|--------|
| `WGPU_BACKEND` | Force one wgpu backend: `vulkan`, `dx12` (Windows), `metal` (macOS). Unset, this crate enables both Vulkan and DX12 on Windows — without preferring either, so wgpu's own ordering picks — and Metal on macOS |
| `RLX_WGPU_COOP_F16_VK_ENABLE=1` | Opt in to CoopF16Vk (default: off) |
| `RLX_WGPU_NO_COOP_F16_VK=1` | Legacy: skip CoopF16Vk — the matmul falls through the cascade (CoopF32, else the f32 tile path) |
| `RLX_WGPU_COOP_F16_VK_OSC_THRESH` | Oscillation score threshold for auto wide fallback (default `0.35`) |
| `RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE=1` | Disable oscillation-based wide fallback |
| `RLX_WGPU_COOP_F16_VK_NO_F32ACC=1` | Disable f16×f16→f32 coop accumulator kernels |
| `RLX_WGPU_COOP_F16_VK_FORCE_WIDE=1` | Always wide-fallback CoopF16Vk matmul at dispatch |
| `RLX_WGPU_COOP_F16_VK_LARGE_N=1` | Keep CoopF16Vk for N > 768 (default: wide f32 for accuracy) |
| `RLX_WGPU_COOP_F16_VK_LOAD_T=1` | Force `coopLoadT` on B when `RLX_WGPU_COOP_F16_VK_LARGE_N=1` and N > 768 |
| `RLX_WGPU_NO_COOP_F32=1` | Disable CoopF32 on Metal |
| `RLX_WGPU_FORCE_COOP_F32=1` | Force CoopF32 eligibility where the adapter reports no 8×8 f32 cooperative matrices (output is unreliable there); the portable 8×8 path is otherwise selected automatically |
| `RLX_WGPU_FORCE_INPUT_UPLOAD=1` | No-op (inputs are always re-uploaded; kept for older scripts) |
| `RLX_BENCH_DISPATCH_ONLY=1` | Skip output readback (micro-benchmarks) |
| `RLX_WGPU_F16_WEIGHTS=1` | Legacy f16-storage matmul experiment |
| `RLX_WGPU_SCHEDULE=1` | Log compiled dispatch schedule |
| `RLX_DISPATCH_REPORT=1` | Per-step dispatch report |
| `RLX_WGPU_NO_PACKED_BSHD_ATTN=1` | Disable packed QKV attention stride path |
| `RLX_WGPU_DUMP_NODES=1` | Per-node max-abs dump after run (debug) |
| `RLX_WGPU_DBG_COUNTERS=1` | Per-run report of where the run's time went, plus the step/uniform/bind-group census (see below) |
| `RLX_WGPU_FORCE_HOST=1` | Host-fallback all risky SPIR-V paths |
| `RLX_WGPU_SHARD_GPU=1` | Opt out of discrete/sharded host packing (unsafe for Kitten NSF) |
| `RLX_WGPU_DISCRETE_HOST=1` | Extra discrete hosting via `wgpu_prefer_host_fallback` |
| `RLX_WGPU_HOST_BUFFER_COPY=1` | Force host round-trip for BufferCopy |
| `RLX_WGPU_HOST_EAGER_H2D=1` | Write every HostOp/Conv output to the device immediately (default: defer until Expand/Concat/GPU/readback) |
| `RLX_WGPU_HOST_MATMUL=1` | Pack MatMul as HostOp |
| `RLX_WGPU_CONV_HOST=1` | Force Conv2d host path |

**Discrete Vulkan/DX12 (Kitten NSF):** Binary/Activation/Where/Fma, norms,
Expand/Concat/Transpose/Narrow, and Conv stay on the host (SPIR-V elementwise
and structure paths collapse audio). MatMul and unsharded BufferCopy stay on
GPU. Host→Host streaks reuse a tensor mirror (`HostTensorCache`) and skip empty
compute encodes between host steps; deferred H2D batches uploads before the
next device-reading step.

**Encoder parity notes:** compile runs `LegalizeBroadcast` before fusion
(mid-axis `[1,C,1,D]+[1,C,P,D]` needs `Expand`); the copies it inserts for a
binary operand `simple_broadcast` *can* describe (`[1,C,1]` against `[1,C,T]`)
are peeled back at lowering (`unfuse::elide_binary_broadcast_expands`). Compile
also unfuses `ElementwiseRegion` on wgpu (region kernel only supports trailing
broadcast). `Activation::Gelu` uses exact `erf` in `unary.wgsl` (not the tanh
approx).

## Where a run's time goes

`RLX_WGPU_DBG_COUNTERS=1` prints a `[wgpu-run-report]` block to stderr once per
run of the compiled graph — the bracket is entry to return, so building the
graph, loading weights and an AOT compile sit outside it, while a dynamic-shape
graph's in-call compile is one of the named intervals. A cold first run pays
pipeline creation too, so figures compare between warm runs of one graph. One
caller-side call can be several reported runs, and a per-token decode loop prints
one block per token; each block is a single `eprintln!`, so blocks never
interleave.

The partition is `stage` (inputs and params reaching the arena), `compile` (that
build), `encode` (this run's dispatches), `wait` (a fence on device progress),
`readback` (outputs crossing back), `host` (a host-route step's own compute) and
`probe` (the instrument's own work: the query set and buffers it builds with the
bracket, then its fence, resolve submit and map) — contiguous,
non-overlapping, summing to the bracket by construction. Every label is
**host-visible**: `encode`, `wait` and `readback` carry queue latency and the
fence rather than kernel time, so reading them as kernel cost is the mistake this
instrument exists to prevent. A read-back copy that rides the same submission as
the run's own device work is inside that `wait` too — the run path serves the
fence and the map with one edge, and splitting them would change the run being
measured — while the pooled read-back that closes a run reads its outputs through
a submission of its own, so its fence and its map are both `readback`. A path
with no such edge omits the label and says so on an `omitted` line, never a zero
beside it: a graph compiled outside the bracket has no `compile` edge, and the
run's report says exactly that rather than printing a zero-steps `compile` line.
Labels may repeat, so the line count is not fixed — parse by label, never by
position or count. `convergence=ok` compares the integer clock units of the
partition against the total: it proves the timeline is fully covered, **not** that
each step was attributed to the right interval. Every figure is printed as
milliseconds with six decimals, which is the integer nanosecond itself — one unit
throughout, never a ratio — so the printed interval figures sum to the printed
`total_ms` exactly and the convergence check is one a reader can repeat on the
block. An interval whose integer span is zero is never pushed at all, so no
interval line reads `0.000000` in the role of a row: a label the run had no edge
for is an `omitted` line, and a span the clock could not separate from the stretch
around it is that stretch's.

Each interval line's `kinds=` is the schedule steps that interval covered, by
step kind with counts, sorted by count descending — so the `host` line ranks the
host-executed kinds, including a step the run defers to the end of the schedule,
which `host` counts because that is the interval that ran it. A step the run
walked past *without* doing it — a weight pack already baked, a dispatch scaled to
zero — is counted by the interval whose cursor range spanned it, since no other
interval ran it. A step appears in **every** interval that covered it — the
uniform pre-pass stages one, the batch that dispatches it is another — so these
counts are per interval and are not a partition of the schedule. `wait`, `readback`
and `probe` cover no steps of their own (a wait covers device work issued in the
interval before it) and print `steps=0 kinds=`; `stage` prints none on a run whose
uniforms already match the extent in use, which is the pre-pass having nothing to
write. The run line's `reached=` is how far into the schedule the run got, read
where the schedule's end is reached — the run loop's own exit, or the end of the
last batch when that batch already carried the whole schedule — so a run that
finished it reports the schedule's length, and a figure short of `schedule=` is a
run that stopped early.

The `device` line is a **lower bound**, deliberately outside the partition and
never added to or subtracted from it: nested inside the host-visible intervals,
attributed to the interval that issued the pass. It is **per contiguous stretch
of dispatch passes, not per step** — the run packs a whole contiguous stretch of
device work into one pass on this adapter family, and the crate splits passes
differently elsewhere, so the figures are not comparable across adapters — and it
comes from a quantised, implementation-defined device clock: values are sanitised
before display, and the adapter they came from is named on the line. Nesting is by
issue and not by containment: the device runs ahead while the host encodes, so a
figure can exceed the interval that issued its pass, and nothing here compares the
two halves. Its `scope=` names both kinds of pass it covers: the run's own
dispatch loop's, and a **packed-weight multiply's** — the GEMV arm, the multiplies
a batch joins, the per-row fallback's one pass per row, the Q1_0 arm, and the
dequant-scratch route's two — each keeping the label its own site gave it.
`passes=` counts those records, so it is this instrument's ledger and not the run's
batched pass count.

Coverage is not completeness. A pass another site opens still carries no figure: an
arena copy's — the one site, whether the batch joins it or the call submits on its
own — a KV row feed's, the RNG's, `GatherSplit`'s, the splat route's, the
kernel-selection probe's, and every host route's own submission. Device work can
therefore be missing from the view, and subtracting these figures from an
interval's `host_ms` is **not** the run's waiting.

Reading a packed multiply's own figures: one the run joins into its batch opens its
pass beside the batch's own, so its line is an `encode` one, and one that runs as
its own submission — Q1_0's arm and the per-row fallback's, and the scratch route's
two — opens its pass inside the `host` interval its step runs in. The run's own
dispatch loop only ever opens passes in `encode`, so **a `device_pass` line outside
`encode` is a packed multiply's own submission**. Inside `encode` the interval's
`kinds=` names the packed step that batch carried, which is as far as attribution
goes: `per=stretch` is the finest a pass-level timer gives, so read those lines with
the interval's kinds rather than reading one `index=` as one multiply — and a
per-row fallback under one step can be many lines.

`unavailable` is what a pass reports when no figure can be obtained (the capability
absent, a run that encoded no pass, the query never written, samples that do not
advance, a delta larger than the bracket, a pass past the instrument's capacity, or
the empty pass the loop opens to submit an encoder it is still carrying, which
contains nothing to measure) — never a zero, never a measurement, never a silent
truncation — and the wall time of such a pass stays inside the host-visible interval
that carried it, which the `device_unusable` line restates by name and count: one
line per interval that carried unusable passes, `index=` its 1-based line ordinal
(so a repeated label is still matchable) and `host_ms=` that one interval's own
span — a figure already in the partition, never summed into it and never a device
figure. A pass no interval ever closed over has no span to restate and prints
`index=none host_ms=unavailable`, not a fabricated zero.

The instrument holds 2048 passes — wgpu's query-set ceiling, not a tunable — and a
run that creates more keeps every one of them: `passes=` counts them all, while the
query slots run out at 2048. `measured=` is at most those 2048 slots and can be
short of them, since a slotted pass whose samples give no figure — the empty batch
pass the loop opens to submit an encoder it is still carrying, a query never written
— counts `unusable` too, as does every pass past the ceiling. So what a reader sees
then is `measured=` at (at most) the 2048 slots, a non-zero `unusable=`, and a
`device_unusable` line per interval that carried them: never a dropped pass, never a
silent truncation. The count grows with the packed multiplies a step takes, so the
per-row fallback — one pass per row — is the shape that reaches the ceiling first.

The instrument's own work is the `probe` interval, not production time: its query
set and two buffers, then the resolve, which is fenced off from the passes it reads
and submitted on its own, since the same-encoder form read back all-zero samples on
Metal here, and a dispatch-only run gains that fence.

Under `RLX_WGPU_SNAPSHOT=1` a call is a replay session end to end, so nothing is
produced at all — its last pass is a full one, and the report cannot tell that
pass apart from a production run's. The browser build produces nothing either,
with no clock to read and the capability never requested there. Set, the report's
own query set, buffers, fence, resolve submit and map change a run's
submission/wait shape (adding one fence to a dispatch-only run), and its passes —
the batch's and the packed multiplies' — carry timestamp writes, so tests that pin
that shape or a rate refuse while it is set; an ordinary run is unchanged in work,
values and printed lines. Where an existing debug switch makes the profiled run do
extra work inside the bracket, an `extra_work` line names it — including this
switch itself, whose census adds both the pre-pass line printed inside `stage` and
one line per *dispatched* step from inside the batch's dispatch loop (a run with no
dispatched step has the pre-pass line only), so a reported `stage` and `encode`
carry those stderr writes. The table behind those lines is the switches the run
path itself reads, not every diagnostic switch in the crate: one a dump path, a
feed path or the compiler reads is not named.

```
[wgpu-run-report] run total_ms=4.153058 partition_ms=4.153058 convergence=ok intervals=7 schedule=2 reached=2
[wgpu-run-report] interval label=probe host_ms=0.041185 steps=0 kinds=
[wgpu-run-report] interval label=stage host_ms=0.029600 steps=1 kinds=matmul:1
[wgpu-run-report] interval label=encode host_ms=0.056577 steps=1 kinds=matmul:1
[wgpu-run-report] interval label=wait host_ms=2.663947 steps=0 kinds=
[wgpu-run-report] interval label=host host_ms=0.401103 steps=1 kinds=buffer_copy:1
[wgpu-run-report] interval label=readback host_ms=0.487822 steps=0 kinds=
[wgpu-run-report] interval label=probe host_ms=0.472824 steps=0 kinds=
[wgpu-run-report] omitted label=compile reason=no_deferred_compile_in_this_path
[wgpu-run-report] device adapter="Apple M3 Pro" backend=metal view=lower_bound scope=dispatch_passes+packed_multiplies per=stretch passes=1 measured=1 unusable=0 device_ms=0.130125
[wgpu-run-report] device_pass index=0 interval=encode ms=0.130125
[wgpu-run-report] extra_work switch=RLX_WGPU_DBG_COUNTERS effect=prepass_census_and_per_step_print
```

The packed multiply's own figures, on one packed weight multiply with the dense step
beside it (same rig and switch). **One row** — the arm the run joins into its batch:
the `encode` interval's `kinds=` name the packed step, and the multiply's pass is
the second of its three lines (the first is the empty batch pass the loop opens
before it sees that its step is the host-classified multiply, the third the dense
step's), so the multiply's own device figure — 0.017750 ms here — is read off that
line rather than inferred from the interval's host time:

```
[wgpu-run-report] interval label=encode host_ms=1.936316 steps=2 kinds=dequant_matmul_gguf:1,matmul:1
[wgpu-run-report] device adapter="Apple M3 Pro" backend=metal view=lower_bound scope=dispatch_passes+packed_multiplies per=stretch passes=3 measured=2 unusable=1 device_ms=0.106458
[wgpu-run-report] device_unusable passes=1 interval=encode index=3 host_ms=1.936316
[wgpu-run-report] device_pass index=0 interval=encode ms=unavailable
[wgpu-run-report] device_pass index=1 interval=encode ms=0.017750
[wgpu-run-report] device_pass index=2 interval=encode ms=0.088708
```

**Four rows, off tile** — the 8-bit scheme's contraction that is not a whole staged
tile, which the grouped arm's admission turns down: the call recomputes on the
scratch slab and so submits on its own, its two passes the lines outside `encode`,
inside the `host` interval that ran the step. A multiply the admission takes opens
its pass inside the enclosing encoder instead — the one-row case above — so no
`device_pass` line outside `encode` is its. `ms=0.015125` and `ms=1.757041` here
against that interval's `host_ms=11.813344` account for the slab's own device work
and not for the rest — with the passes this view still does not cover, the remainder
is "not this pass", never "waiting":

```
[wgpu-run-report] interval label=host host_ms=11.813344 steps=1 kinds=dequant_matmul_gguf:1
[wgpu-run-report] interval label=encode host_ms=0.226748 steps=1 kinds=matmul:1
[wgpu-run-report] device adapter="Apple M3 Pro" backend=metal view=lower_bound scope=dispatch_passes+packed_multiplies per=stretch passes=3 measured=3 unusable=0 device_ms=1.813541
[wgpu-run-report] device_pass index=0 interval=host ms=0.015125
[wgpu-run-report] device_pass index=1 interval=host ms=1.757041
[wgpu-run-report] device_pass index=2 interval=encode ms=0.041375
```

The `[wgpu-run-report]` prefix is unique to this block, so the lines can be
extracted from the interleaved diagnostics a run already prints; field names are
fixed and every time is in milliseconds with six decimals; an interval line
reads `interval label=<phase> host_ms=<ms> steps=<n> kinds=<name>:<count>,...`;
an unavailable device figure is the literal `unavailable`. The kinds and their
fields, which is all a parser needs:

```
run            total_ms partition_ms convergence intervals schedule reached
interval       label host_ms steps kinds
omitted        label reason
device         adapter backend view [reason] scope per passes measured unusable device_ms
device_unusable passes interval index host_ms
device_pass    index interval ms
extra_work     switch effect
```

No test asserts a time: the tests assert structure, labels and parseability — by
re-running a case in a child process and parsing its stderr — and anything
adapter-dependent rides the `backend-tests` feature.

## Build / test

```sh
cargo build -p rlx-wgpu --release
cargo test  -p rlx-wgpu --features backend-tests --release
cargo test  -p rlx-wgpu --features backend-tests --test coop_mat_probe --release -- --nocapture
# The CoopF16Vk tests only run with the opt-in switch; without it they self-skip
# and still report `ok`:
RLX_WGPU_COOP_F16_VK_ENABLE=1 \
  cargo test -p rlx-wgpu --features backend-tests --test coop_f16_vk_correctness --release -- --nocapture
# `coop_f16_vk_dx12` is DX12-only — run it with `WGPU_BACKEND=dx12` on Windows.
# Anywhere else it skips on its backend check, and on a DX12 adapter it skips on
# the cooperative-matrix property check — this engine exposes no cooperative
# matrices on DX12 — so its assertion is never reached:
cargo test  -p rlx-wgpu --features backend-tests --test coop_f16_vk_dx12 --release -- --nocapture
```

Through `rlx-runtime`:

```sh
cargo build -p rlx-runtime --features gpu --release
```

## Status

Functional, less battle-tested than `rlx-metal` / `rlx-mlx` on Apple
Silicon. CoopF16Vk on Vulkan is **not exercised by default**: its two tests,
`coop_f16_vk_correctness` and `coop_f16_vk_bert_baseline`, never set
`RLX_WGPU_COOP_F16_VK_ENABLE`, so they self-skip and still report `ok`. With the
switch set they compare against f16-ref on RTX-class GPUs (gentle BERT
weights vs f16-ref, oscillating weights auto-wide to f32). The legacy fp32
tile matmul kernel remains correctness-first for unaligned shapes.

## Gotchas

- **Scalar-output latency:** End-to-end `run()` on tiny graphs is often
  readback-bound (~1.3 ms on MoltenVK; ~100–150 µs on DX12/Vulkan).
  Kernels may be sub-µs; use `RLX_BENCH_DISPATCH_ONLY=1` to time dispatch
  without readback. See [`docs/benchmarks/higher-order-ad.md`](../../../docs/benchmarks/higher-order-ad.md).
- **WSL:** Linux cargo on the rig uses `~/rlx-workspace-mirror/rlx` (ext4),
  not virtio `D:` directly. `rig.sh bench-nth-order` syncs new examples there.
- Wgpu is async; we wrap with `pollster::block_on` for sync semantics.
  `run()` fuses output readback copies into the final compute encoder
  submit and schedules host mapping via `CommandEncoder::map_buffer_on_submit`
  (wgpu 29+), then reuses a pooled MAP_READ staging buffer across runs.
- The matmul kernel is correctness-first. It loops over K per thread
  with no register blocking or shared-memory tiling — order of
  magnitude slower than what's possible. Optimization comes after
  the op set is broad enough to run a real model.
- Shader compilation is lazy + cached via `OnceLock`. First dispatch
  pays the WGSL → SPIR-V/MSL/HLSL translation cost (~ms); subsequent
  dispatches reuse the compiled pipeline.

## License

MIT OR Apache-2.0.
