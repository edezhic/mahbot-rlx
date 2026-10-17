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
  (`ReadbackStaging`, `read_f32_many_pooled`), and `read_f32_ranges` — several
  rows of several slots in one submission and one map, which is the shape a
  decode step's resident-KV rows leave through. Env: `RLX_WGPU_SHARD_LOG=1`.
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
  arena does not store) — and so does `Op::ConvTranspose2d`'s host step, which
  carries 32-bit byte offsets into that arena while its compact form stages the weight
  from there; the op defaults to its own device step, so that host step is reached only
  behind `RLX_WGPU_CONV_HOST` or the structure-host preference
  (`RLX_WGPU_FORCE_HOST`, a discrete adapter, a sharded arena). An absolute-offset arm
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
  Q4_K/Q6_K/Q8_0 take a tiled per-row GEMV arm (one workgroup per two output
  columns, its lanes splitting each column's contraction over 32-element units of
  the packed weight, one X read four elements at a time shared across them)
  while Q1_0 keeps one thread per column on the scalar X view; for `m > 1`
  Q4_K/Q6_K **and Q8_0** a row-grouped GEMM reads each packed weight tile's quant
  bytes once per group of 8 rows instead of once per row — and that GEMM's Q4_K
  entry, when a dispatch would be too small to fill the GPU, covers half the
  columns per workgroup with two threads to each instead, splitting the contraction
  across them and reading each block's 16-byte header twice. That row-grouped arm's
  `k` gate is the scheme's own packing block — Q4_K/Q6_K: 256, which is the staged
  tile itself; Q8_0: 32 — so a Q8_0 row of several tiles plus a block, or of less
  than one tile, takes the arm and joins the enclosing batch: the shader's tail tile
  stops at the row's last whole block, and stages no activation past the row's own
  end (`stage_x_bounded`, its fetch guarded by `p0 + c4 < k`). The scratch slab is
  left for the shapes the arm refuses — a `k` off the scheme's blocks, an X window
  its `vec4` reads cannot address, the binding limit, the arm's bisect lever.
  Other schemes dequant into scratch then `matmul_bt` when the arena is a single
  buffer; sharded / split-weight arenas fall back to `gguf_host`. Grouped
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

`Op::WelchPeaks` / `Op::LogMel` / `Op::LogMelBackward` run in a deferred tail when their
lowering picks the host step (`step_is_tail_host`: `Step::WelchPeaksHost` / `LogMelHost` /
`LogMelBackwardHost`) — the walk skips those steps and performs them once the whole schedule
has been dispatched — so on this route a hosted step is usable only where such steps **end**
the schedule and the device work that feeds them precedes them in the same batch (a packed
multiply or buffer copy this route joins counts as that device work). A `Step::WelchPeaksGpu`
is device work, and may sit anywhere. Every other placement of a hosted step is refused at
plan time, and a run of such a graph under `RLX_WGPU_SNAPSHOT` or `RLX_WGPU_ONE_OP_PER_PASS`
stops loudly instead of spinning. See the placement guard in `src/backend/step.rs`
(fork-local; upstream carries the same deferral and settles none of these placements).

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
- Fallback wide path: the vectorised **`matmul_wide_vec4.wgsl`**, which ships a
  second, half-height entry point (`matmul_wide_vec4_tail`, 32 rows): the row axis
  is dispatched as whole 64-row tiles plus one remainder tile whenever the row
  count leaves a remainder that fits, so a 96-row node computes 96 rows, not 128
  — bit-identical to the single rounded-up grid it replaces. The 64×64
  **`matmul_wide_nv.wgsl`** is behind `RLX_WGPU_WIDE_NV=1`, and
  **`matmul_wide.wgsl`** behind `RLX_WGPU_WIDE_STD=1`; both pins keep the
  single-tile dispatch.

When a matmul operand is an `Activation`, the lowering **host-mirrors** it
each `run()` (`apply_activation` + `write_f32`, same f16 shadow as Params)
and routes the matmul through the **wide f32** path. CoopF16Vk
remains on Input/Param-only operands (BERT QKV weights, etc.). A GPU
`cast_f32_to_f16` pre-pass still handles non-unary computed tensors when
CoopF16Vk applies. Pass flushes between unary → cast → coop matmul keep the
discrete-adapter visibility correct.

The **f32 path** is on by default on Metal — `matmul_coop_f32.wgsl`, 32×32
`simdgroup_float8x8` tiles. It is admitted where `K` is a multiple of 8 and the
output-column extent a multiple of 32 (`M` is free: the padded rows are computed
and their stores dropped). The two **convolution GEMM routes** — the im2col panel
multiply and the 1×1 pointwise GEMM, whose weight is the FIRST operand — ask the
same admission for the **portable alignment** instead, a multiple of 8, because
their `n` is a conv's spatial output extent (see below). It reads both operands
straight out of the f32 arena, so the weight's operand role does not matter to
it: the ordinary Param-weight multiply and those two routes all reach it, and
`coop_f32_dispatch_count()` counts every dispatch of it. Opt out with
`RLX_WGPU_NO_COOP_F32=1`. The portable 8×8 sibling
(`matmul_coop_f32_portable.wgsl`) is a discrete Vulkan adapter's; the conv routes
never take it, whatever the adapter — their admission is Metal-scoped.

### The convolution GEMM routes' output-column extent

A convolution's panel multiply has a conv's spatial output extent for its `n`, and
such an extent is a multiple of 8 far more often than of 32: the recognition
front's third stride-2 convolution reads as `16 × N` on its frame axis, so its `n`
is 160 at N = 10, 208 at N = 13 and 256 at N = 16 — and 208, the middle one, is
the one the Metal form's own 32 refused, on the wide non-cooperative path while
its neighbours took the cooperative one. The kernel lays such an extent out either
way: with an `n` a multiple of 8 only the x grid rounds up exactly as the y grid
does, the `b_*` gathers read past the row, and the store guard's
`global_col >= params.n` term drops the padded columns — the same reliance the
padded rows already stand on, and the same test covers both. So `conv_gemm` asks
the admission for the **portable** alignment, and an extent that satisfies it is no
longer refused.

Measured on an M3 Pro with Metal on macOS, `--release`, the crate's own
`backend-tests` targets, shipped defaults, no tuning cache, `just throttle`
clean. The stage's own figures are the trailing block of
`conv_gemm_coop_f32.rs`: the front's third convolution as **one node** — 3×3
taps, stride 2, pad 1, 480 → 512, i.e. a contraction of 4320 (480 × 3 × 3) —
channels picked so that the kernel and not the submission floor decides (the
64-channel cell of the same extent, in the route-cost table below, sits at its
own floor), plus the other caller of the widened admission, the 1×1 pointwise
route at the extent the case loop admits. Each arm is compiled under the shaped
admission and under `RLX_WGPU_NO_COOP_F32=1`, and timed interleaved round by
round, one warm-up call each and then three rounds, with the shaped arm's one
dispatch and every arm's place against the crate's own CPU conv read with it:

| the node | wide | cooperative |
|---|---|---|
| front's third conv, `16 × 13 = 208` (8-only — the newly admitted extent) | 3.20–3.65 ms | **2.25–2.58 ms** |
| front's third conv, `16 × 10 = 160` (a multiple of 32 too) | 2.82–3.13 ms | 2.02–2.20 ms |
| front's third conv, `16 × 16 = 256` (a multiple of 32 too) | 2.98–3.12 ms | 2.19–2.42 ms |
| pointwise 1×1, `n = 5000` (8-only, the route's newly admitted extent) | 0.97–1.09 ms | 0.91–0.98 ms |

Each figure is the block's median over its three rounds and the range is six takes
of the block, in a window the box read `load1` 3.3–5.1; the rounds themselves are
printed per take, and they are the unit here — a co-tenant on the box can swamp
both arms of a take at once (the fork's `just throttle` gate exists for the same
reason). The three convolution rows are the front's own frame axis at N = 10, 13
and 16, reached by input width 19, 25 and 31. The box is shared and the absolutes
move with what else runs on it — an earlier window (three takes at `load1`
3.9–5.0) read the same three extents 12–20 % lower on **both** arms (2.55–2.67 wide
against 1.95–2.22 cooperative) — while the ratio the arms hold does not: 0.68–0.87
at the three convolution extents over all nine takes. The pointwise node's ratio is
0.90–1.00: at `m 64 × k 128 × n 5000` it sits in the submission floor, so the kernel
choice is not visible in its time — its rows are the parity and the dispatch count,
not a gain. `0.0` is what the cooperative arm reads
against the wide one and what each reads against the CPU conv, at every extent and
in every take. The same node under `RLX_WGPU_DBG_COUNTERS=1`, one run per arm — a
run that harness does not take, since the instrument changes a run's own
submission shape — is the second reading: wall `total_ms` 4.216 → 2.730 and the
`device` line (lower bound, `scope=dispatch_passes+packed_multiplies`)
2.969 → 1.753, with `passes=1` and one `wait` interval either way — the
submission shape does not move, what moves is the kernel inside it. The `kinds=`
line is `im2col_gpu:1,matmul:1` on both arms, the cooperative dispatches are 1 per
run on the cooperative arm and 0 on the wide one (`coop_f32_dispatch_count()`,
which is the count the block asserts on the shaped arm), and the arena is the same
77 870 464 B on both. This is a one-node graph, so its floor — the submission and
the read-back — sits in both arms; a front-level cut of the whole model, which
lives in the sibling `rlx-models` repo, cannot be re-taken here.

**Which other extents change route, and what they cost.** Only these two routes
have the widened term, so the shapes that move are convolution GEMM extents that
are a multiple of 8 and not of 32, with `K` a multiple of 8 and `M` free. Of the
crate's own route-cost cells that is three. The kernel the pre-change tree put
them on is what `RLX_WGPU_NO_COOP_F32=1` forces (the pre-change admission did
refuse these extents), so both columns are reproducible from this tree, from three
takes of the harness under each setting at `load1` 3.3:

| cell (its spatial extent) | pre-change kernel | shipped default | arena |
|---|---|---|---|
| `lift-slot0-l1000` (1000) | 433–476 µs | 439–452 µs | 64.6 MiB |
| `refuse-height-pad` (600) | 382–411 µs | 334–378 µs | 64.3 MiB |
| `map-3x3-s2-31x25-c64` (208, the front's extent at 64 channels) | 366–392 µs | 403–411 µs | 64.4 MiB |

Every one of these is a small cell — two or three steps in 64 channels, which the
harness prints as an unaccelerated arm of 656–731 µs against an accelerated arm of
334–476 µs — so the kernel sits inside the cell's submission floor, and what the
two columns read is a range and not a gain: their medians differ by ~±40 µs across
the three cells, and which side reads ahead is not stable across windows. That is
why the table's claim is bounded — no cell is slower by more than that, the arena
and the routes are the same on both columns, and no other cell changes route. The
208 cell is worth the one explanation: it is the front's own extent at a channel
count the model does not run, its multiply 60× smaller than the model's own node at
the same extent (which reads 0.70–0.77× the wide arm, table above), so what it
prices is the cooperative form's setup on a multiply too small to amortize it.
`conv_gemm_coop_f32.rs`'s `pointwise-n8-only` row is the fourth moved shape
(`n = 5000`, refused before and admitted now), priced by the pointwise row of the
table above — a wash, floor-bound — and carrying the parity and the dispatch count
beside it.

One rollout caveat, since these routes read the shared per-shape tuning table
(`src/tuning.rs`) as any other multiply does: an entry persisted there for one of
the newly admitted shapes — `RLX_GPU_TUNING_CACHE`, else
`~/.cache/rlx-wgpu/dispatch-tuning.tsv` — that records `F32` keeps the wide kernel
at that shape, so a re-shoot on a machine holding such a table reads as the
widening having done nothing. Remove that file (or point `RLX_GPU_TUNING_CACHE` at
a path that does not exist) to read the change itself; the test pins that path
away for exactly this reason, and the route-cost harness above does not — its
columns are read on a box without such a table.

**The shared widening and this one are not separated by cost.** Widening the
whole admission to the portable alignment is `RLX_WGPU_FORCE_COOP_F32`'s own
geometry term, and the extents it would move beyond the two conv routes are the
ordinary multiplies whose `n` is a multiple of 8 only. Timed the same way on
one-node matmul graphs, the pin (`RLX_WGPU_MATMUL_F32_ONLY=1`), the shipped
default and the widened shared rule:

| shape | f32 pin | default | widened shared |
|---|---|---|---|
| `191×200×600` (the EEG-QKV width) | 0.326 ms | 0.328 ms | 0.303 ms |
| `880×200×200` | 0.551 ms | 0.543 ms | 0.446 ms |
| `1×880×200` (a decode row) | 2.457 ms | 2.456 ms | 0.432 ms |
| `8×24×24` | 0.293 ms | 0.294 ms | 0.210 ms |
| `8×8×8` | 0.238 ms | 0.238 ms | 0.216 ms |

No extent measured slower under the shared rule either, so no cost argument
separates the two forms; the **shape-scoped** one is what lands because it moves
no shape outside these two routes, and the cell table above is the evidence that
nothing outside them changes route at all.

**The active-extent mode does not turn on.** `Step::Im2ColGpu` answers false to
`safe_for_active_extent`, which is all-or-nothing per graph, so any schedule that
walks a panel-routed convolution — the front's three-conv stack, whose first
convolution is on the direct kernel because its contraction is under the panel
route's 256 bar, among them — has the hint disabled on every pass. This change
does not lower that count: what it moves is the GEMM's kernel, not the panel
gather the veto names, so the mode is exactly as reachable after it as before.
The crate's own probe (the active-extent block of `conv_route_cost.rs`, which
asserts the veto per arm) reads the bucketed `(2, 4)` run at 0.85–1.00× of the
full one on those arms — 0 % of the run skippable — so the mode's worth stays
unmeasured as a gain, as it was before the change.

**Memory.** The cooperative kernel reads both operands out of the f32 arena — no
f16 shadow, no scratch — so the plan is the same on both arms: the block above
asserts arena equality per node and prints 77 870 464 B at the 208 node (77 415 040 B
at 160, 78 325 888 B at 256, 70 981 888 B at the pointwise extent), the route-cost
cells' arena lines are unchanged (64.6→64.6, 64.3→64.3, 64.4→64.4 MiB — the third
being 67 508 224 B for that same conv at 64 channels), and the process's resident
set reads 53.3 MB on each arm of a one-shot run of the 208 node. Memory counts
alongside speed here, so the model-level reading belongs beside this change's time
rather than being assumed still, and the pair to hand is a by-arm reading taken on
the path with `RLX_WGPU_FORCE_COOP_F32` on — an admission wider than this landing's,
so an upper bound and not this extent's own: the front's graph 65.8 → 69.5 MiB of
arena and 219.3 → 297.8 MiB of process peak, with the third convolution's own cell
39.0 MiB on either arm (that reading slices the model's plan per stage, a different
quantity from the one-node arena the block above prints). Nothing in this kernel's
mechanism produces a plan difference — it adds no buffer and the plans here are
byte-identical on the arms — so the pair is recorded as what that wider arm read
and not as this landing's price. The landing's own pair is the same graph read
**without** the lever, which is what a re-shoot takes, and it belongs next to the
pass's time.

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
| `RLX_WGPU_NO_COOP_F32=1` | Disable the f32 cooperative-matrix matmul, on Metal's 32×32 form and the portable Vulkan 8×8 one alike |
| `RLX_WGPU_FORCE_COOP_F32=1` | Force CoopF32 eligibility where the adapter reports no 8×8 f32 cooperative matrices (output is unreliable there), and widen it to the portable form's `K % 8` / `N % 8` alignment; the portable 8×8 path is otherwise selected automatically |
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
| `RLX_WGPU_CONV_HOST=1` | Force the Conv2d / transposed-conv host paths |

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

## The mel front-end's framing (one step instead of one per frame)

A mel front-end's framing — one `narrow`(+`reshape`) per frame, stacked by one
`concat`, which is what `Graph::stft` / `Graph::spectrogram` emit — is recognised
at lowering and becomes one `Step::Frames`, a strided read of the signal that
replaces two steps per frame and carries no index table
(`lower::detect_frames_pattern`): the dispatch count follows the work rather than
the frame count. `Op::Slice` is the other spelling of a frame slice and reaches the
same place — this backend has no step for it, so the compiler rewrites it to this
form (`rlx_fusion::lower_slice`), pinned end to end.

Measured on an M3 Pro with `cargo run -p rlx-bench --release --example
bench_frame_slicing --features gpu` at hop 160 (21 runs after warm-up, `just
throttle` clean). The counts and the planned bytes (arena, tables) are exact, and the
heap figure is the **Rust** heap's own peak — a counting global allocator, so it sees
nothing the driver allocates and the device side is the arena line; the µs move with a shared
box's load — the harness prints each row's own p10–p90 for that reason, and the
ranges below span four takes at `load1` 3.2–8.4 — so every row has a count beside
it. 512-at-hop-160 neighbours of the same rows carry the same counts (arena 201
and 264 KiB):

| the front-end's framing | 76 × 400 | 100 × 400 |
|---|---|---|
| slice-per-frame, pre-fold | 152 steps, 0.57–0.72 ms | 200 steps, 0.58–0.72 ms |
| fused `Step::Frames` | **1 step**, 225–234 µs, arena **167 KiB**, tables 0 | **1 step**, 215–234 µs, arena **219 KiB**, tables 0 |
| batched `Op::Gather` | 1 step, 169–220 µs, arena 286 KiB, tables 118 KiB | 1 step, 169–191 µs, arena 376 KiB, tables 156 KiB |
| `Op::Im2Col` | 1 host step, 2 sub/3 waits, 389–435 µs | 1 host step, 2 sub/3 waits, 404–427 µs |
| front-end (framing + transform + band + log) | 8 steps, 664–824 µs | 8 steps, 487–680 µs |
| the same front-end inside a consuming pass | 9 steps, 642–846 µs | 9 steps, 626–749 µs |
| that pass's own floor (mel handed in) | 1 step, 208–259 µs | 1 step, 214–269 µs |
| the front-end on the host, its mel handed to the same pass | 1 step, 354–412 µs (+24 KiB upload) | 1 step, 382–453 µs (+31 KiB) |

The first column's 152/200 dispatches are that spelling's pre-fold reading on this
backend: on the CPU device rows of the same sheet the spelling costs 14–17 µs a
call, which is what it costs where nothing folds, while the pre-fold *device*
spelling cannot be reached here any more — the concat arm's own lever
(`RLX_WGPU_CONCAT_HOST`, pinned in `tests/frames_concat_host_knob.rs`) stands the
fold down, but only takes the concat off its direct window path: on this
unified-memory adapter it is then a *device* concat staged per input through scratch
(`buffer_copy` + `concat`, 3 steps a frame, 228–300 steps at 6.1–8.1 ms), not the
pre-fold 2-steps-a-frame spelling, and the host spelling is reached where this
backend prefers hosting anyway (a discrete Vulkan/DX12 adapter, a sharded arena, or
`RLX_WGPU_FORCE_HOST`) — so the pre-fold *time* cannot be re-taken in this tree
either (0.57–0.72 ms a call on a quiet box, 0.85–1.56 ms on a loaded one, that form
being host-encode bound at 2 dispatches a frame). The lever corroborates the count
half of that reading and not the time half. The step census quoted here is taken for
this backend's schedule.

The batched `Op::Gather` reaches the same dispatch count and is consistently a
little faster on time, but buys it with a `[n_frames, frame_len]` index table:
+119…+157 KiB of arena (167→286 and 219→376 KiB on the table) plus 118…156 KiB as
constant tables. It is also ~9× *slower* than the per-frame spelling on the CPU
device (117 µs against 13 µs at 76 × 400), which is why the shared builders keep the
per-frame spelling and the fold lives in this backend's lowering. `Op::Im2Col` is one
*host* step and pays a span round trip (2 submissions, 3 waits) on every call.

The same harness prices the stage as a whole: the front-end inside a pass that
consumes its mel reads 642–846 / 626–749 µs against **354–412 / 382–453 µs** for
the front-end on the host with its mel handed to the same pass — disjoint ranges,
so the host form is the cheaper one at these shapes by 1.6–2.4×, which is the same
verdict the bar was set from (763/761 µs against 361/363 µs). One heavily loaded
re-take put the device form 0.81× ahead of the host form, which is the box's own
spread rather than a different answer: a single take's p10–p90 spans the gap. What
is not load-dependent is the framing's share of this: it is at the pass's own floor
(208–259 µs for one dispatch with its readback), while the front-end is 642–846 µs
— so the fold removes the framing from the bill and the cell that remains is the
one it feeds. That cell is the transform's own tables, a step of their own and on
this backend both the expensive one and a bimodal one: at 76 × 400,
within one take of 21 runs, it spans 607–869 µs while its neighbours span ~13 µs.
The bar's other end is ~0.15 ms — the caller's own figure for a host stage, which is
their CPU front-end (0.10 ms at their window) plus this pass's mel upload (24–31 KiB),
and which this repo cannot re-take; what it can re-take is the host arm on the sheet
above. So a device form of *this* front-end is not worth its dispatch: the fold is
here because the framing is this backend's, not because the stage is, and a stage
that wants the host can have it — the arms are on the same sheet, measured, rather
than implied. The caller's own instrument closed the question the same way, by
reading the staging cost as a slope and taking the pass floor with the same transfers
and no work at all — its readings, out of tree and not re-takeable here: the device
cannot get under that bar even at zero arithmetic (its floor alone is 0.28–0.29 ms
against 0.25 ms for this front-end on the CPU and about 0.10 ms for the product's own
function) — so **this stage's cheap form is the host one**, and the submission count
is unchanged either way (one submission, one wait, nothing host-served inside the
pass). No device work on the transform is owed.

Two things this landing must not be read as changing. It folds the framing half
only: `Graph::stft` / `Graph::spectrogram` pad the frame to a power of two and
return their own bin count, so the builder's whole spectrogram is not a drop-in for
the transform this stage computes. And the window maximum this front-end subtracts
is value-bearing, not decoration: removing it shifts the mel by exactly the constant
and moves the accepted answers with it (the caller's reading too, out of tree and not
re-takeable here: the encoder diverges by about 0.997 against a spread of 3.6), so it
stays — the other reading is not a quiet change but a re-acceptance of everything
this front-end feeds.

The mode the fold's step scales trims a row axis, so the fold is taken only where
the rows a caller buckets are the rows the step presents: a source with **no rows of
its own** (`outer == 1`), where the frame axis is the only row axis the folded step
has. A `total` of whole frames (`outer · n_frames · frame_len`, row-major in frame
order) then scales to the destination's own prefix — out to a frame boundary when the
ratio divides, and part-way into a frame when it does not (the runner's own `ceil`:
the pinned `ceil(390 · 1/100)` = 4 words keeps the first frame's head), where the
per-row steps this replaces would have cut a word out of every frame.
`a_folded_framing_leaves_the_frames_past_the_extent_unwritten` pins it the only way
it can fail: the framing is the graph's own output, and after a full run and then a
trimmed run over *different* input, the frames past the extent still hold the full
run's values.

Where the source carries rows of its own the fold is **not taken at all**, so the
per-frame `narrow`/`concat` steps — which trim the source's rows, the axis such a
caller buckets — keep both the route and the mode. Two pins say so:
`a_framing_over_a_source_with_rows_keeps_its_per_frame_steps` (the census and the
values) and `a_framing_over_a_row_source_keeps_the_mode` (a half-extent run over a
second input leaves the first row of every frame the new input's and the second row
the old one's). The cost is the fold's dispatch win on such a source, which the
green surface does not have: `Graph::stft` / `Graph::spectrogram` frame a rank-1
signal, so their framing folds.

Re-taken in a loaded window (`load1` 12.6–13.5, same box) the counts and the
memory lines of the table above are the same numbers to the byte, while the µs column moves
with the load exactly as the paragraph above says: the in-pass rows read
555–1744 µs against 438–1034 µs for the host form, and the pass floor 160–305 µs.

It refuses anything it cannot prove safe, and a refused graph keeps its per-frame
steps and its values. The clause list is recorded in **one** place —
`detect_frames_pattern`'s own doc, which is what the lowering reads — and every
class a pin can build has a case: `tests/frames_concat_fusion.rs`, plus
`tests/frames_split_qkv_source.rs` for the class whose own graph hides it (there the
planner hands the rewritten source's bytes to the destination, so that file's pins
run with one slot per tensor and this clause is the one that decides). Which case
asserts what — and the one case that is route-only — is that file header's own list,
which also names the clauses no case can be built for and why. The concat lever's
own reading is `tests/frames_concat_host_knob.rs`, a test binary of its own because the
rest of the pins neutralise this crate's routing knobs as a set
(`common::neutralise_device_route_env`).

**Re-taking that reading.** `cargo run -p rlx-bench --release --example
bench_frame_slicing --features gpu` measures the framing forms and the stages that
ride on them at 76 and 100 frames of 400 samples at hop 160, and at their
512-sample neighbours, and prints per row: the median of 21 runs after warm-up with
its p10–p90, the step census from this crate's own lowering, submissions and waits
per run, the Rust heap's own peak (a counting global allocator; the device side is
the arena line), the planned arena at the alignment the lowering
itself uses, the bytes a form carries as constant tables, and a max|Δ| against the
row's own baseline — the plain-Rust framing, which does not use the backend, for a
framing row, and this device's own slice-per-frame form of the same stage for a
whole-stage row. That procedure is the reusable part: an arm
table, a time line, a memory line and a census re-measure any stage whose cost is
suspected of following a count rather than the work, with the form that only exists
as a lowering fold compared both against the spellings the public API already
offers and against the stage's own host form. `tests/frames_concat_fusion.rs` is
the same reading as a gate: the census and the values at both shapes, the fold
firing from this crate's own builders (`Graph::stft` / `Graph::spectrogram`), a
consumer of the framing reaching the CPU answer, and a case per refusal clause
(that file's header names the clauses it cannot build a case for).

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

**Four rows, off the arms** — a scheme this route never admits at all (Q5_K): with
no fused arm and no grouped one, the call recomputes on the scratch slab and so
submits on its own, its two passes the lines outside `encode`, inside the `host`
interval that ran the step. A multiply the admission takes opens its pass inside the
enclosing encoder instead — the one-row case above — so no `device_pass` line
outside `encode` is its. `ms=0.185334` and `ms=2.877292` here against that interval's
`host_ms=100.792793` account for the slab's own device work and not for the rest —
with the passes this view still does not cover, the remainder is "not this pass",
never "waiting":

```
[wgpu-run-report] interval label=host host_ms=100.792793 steps=1 kinds=dequant_matmul_gguf:1
[wgpu-run-report] interval label=encode host_ms=0.276158 steps=1 kinds=matmul:1
[wgpu-run-report] device adapter="Apple M3 Pro" backend=metal view=lower_bound scope=dispatch_passes+packed_multiplies per=stretch passes=3 measured=3 unusable=0 device_ms=3.157834
[wgpu-run-report] device_pass index=0 interval=host ms=0.185334
[wgpu-run-report] device_pass index=1 interval=host ms=2.877292
[wgpu-run-report] device_pass index=2 interval=encode ms=0.095208
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
