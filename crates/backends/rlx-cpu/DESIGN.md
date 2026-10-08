# rlx-cpu DESIGN

## What this crate owns

CPU execution. SIMD kernels, BLAS dispatch, thread pool, arena
executor. Two coexisting execution paths in `thunk.rs`:

1. **Closure-based** — every Thunk variant builds a
   `Box<dyn Fn(*mut u8) + Send + Sync>` at compile time; the
   executor calls the boxed closure per thunk. Uniform but
   pays one indirect call per op.
2. **Direct execution** — single big `match` over `&Thunk` in
   `execute_thunks`. Zero closure overhead. The hot kernels
   (Attention, FusedAttnBlock, FusedNomicLayer, matmul) live here.

Both paths must stay in sync when changing a Thunk variant.

## Decisions

**Two execution paths, not one.** The closure path is the legacy
default and we kept it for ops where dispatch cost doesn't matter
(small element-wise stuff). The direct path is where perf-critical
ops have moved. Picking one would be a refactor; doing it right
needs benchmarks per op to verify the closure call cost.

**FusedAttnBlock and FusedNomicLayer are thunk-level, not IR-level.**
The fusion patterns scan the post-lowering thunk schedule because
the matchers need positional access to neighboring thunks (e.g.
"three Narrows then maybe two Ropes then Attention then matmul").
Doing this on the graph would require richer graph-walking
infrastructure than rlx-opt has today.

**Strided Q/K/V on Thunk::Attention (default = hidden, fused = parent
row stride).** Lets the Narrow×3→Attention fusion (#46 deep) elide
per-head buffers without rewriting the kernel — same code path,
different stride values.

**Persistent thread pool, no per-call init.** `pool::par_for(total,
grain, &|off, cnt| ...)` is the parallel primitive. The worker width
is fixed when the process-wide pool is created: `RLX_WORKERS` when
set, else all of the performance cores on Apple Silicon and ~¾ of the
logical CPUs elsewhere. `pool::num_threads()` is the one notion of
width — it reports the width in effect — and every width that could not
be applied as asked is reported on stderr; `src/pool.rs`'s module docs
are the reference for the resolution rules and their reports.

## What doesn't work / why

- **`autotune.rs` only tunes `par_threshold` and `sdpa_seq_threshold`.**
  The pool worker count is process-wide and fixed before the first
  parallel dispatch, so it is not a search axis. The swept thresholds
  and the returned `TuneResult::config` are values to install in a later
  process — the kernels read the frozen global config — so the tuner
  reports only what it measured, never a setting as if it were live.
- **The shared elementwise floor is set for the heavy kernels.** An op fans out
  only once every worker can get `ELEMENTWISE_PAR_FLOOR` (4 K) elements — right
  for conv/pool/transpose/region callers, generous for a bare activation, which
  is still dispatch-bound at 32 K (measured, min-of-40 `relu`: 37 µs at the
  5-worker default, where a serial pass is ~15 µs). One knob is shared on
  purpose; splitting it per kernel class is the tunable left on the table.
- **No fused dequant + matmul.** `QuantMap` exists in `rlx-ir` (#57)
  but blas.rs doesn't read it. The big LLM-bandwidth win on Apple
  Silicon is still on the table (plan #5).
- **Thunk-level fusion patterns are fragile.** Adding new ops
  upstream can subtly break the matchers without compile errors.
  The unit tests (e.g. `narrow_attention_fuses_in_unfused_path`)
  catch the common case but a regression test per fusion-pattern
  shape would be worth more than they currently are.

## Cross-crate contract

- `compile_thunks(graph, arena)` → `ThunkSchedule`. The arena's
  `MemoryPlan` (from rlx-opt) tells us byte offsets; we never
  allocate. The schedule is the executable artifact.
- Hot ops dispatch via `match thunk { ... }`; new variants need
  arms in BOTH execution paths and the `thunk_read_offsets` helper
  if they read any arena buffer (used by view / fusion safety
  checks).
## License

MIT OR Apache-2.0.
