# Syncing this fork with upstream

This fork is native-only. It exists to serve the MahBot models — Qwen3-ASR
recognition, Qwen3-TTS synthesis, the CED tagger and the wake-word ear path —
as fast and as small as possible on the wgpu backend. Everything else is kept
close to upstream so that a sync is a port rather than a rewrite.

A sync is the work of replaying upstream's commits on top of the fork's own, and
the rules below are what the last one settled. They are written down because the
alternative is re-deriving them from a hundred conflicted files.

## 1. Port direction is per file, by size

Whichever side holds the larger rewrite of a file is the base, and the smaller
side's changes are ported onto it. Fighting hundreds of conflict regions inside
one function is slower and less safe than rebuilding on the other side.

Measure it rather than guessing:

```sh
base=$(git merge-base HEAD upstream/main)
git diff --numstat "$base" HEAD       -- path/to/file   # fork's rewrite
git diff --numstat "$base" upstream/main -- path/to/file  # upstream's
```

As of the 0.2.20 sync: `gguf_gpu.rs`, `buffer.rs`, `backend/mod.rs` and
`backend/compile/lower.rs` are the fork's (upstream's deltas taken in);
`backend/run.rs`, `unfuse.rs`, `rlx-cpu`'s `blas.rs`/`exec_dispatch.rs`,
`rlx-compile`'s `memory.rs` and `rlx-flow`'s `weight.rs` are upstream's (the
fork's work re-applied). Re-measure each sync; these do not stay true.

## 2. The rename layer is a layer, not a change

Every crate this fork publishes carries a `mahbot-` package name at the fork
version, aliased through the workspace dependency table. Upstream's manifests
carry upstream's names. Taking them wholesale loses the rename; fighting hunk by
hunk through the manifests is worse. So the rename is re-applied on top of
whatever the merge produced, as its own step:

```sh
python3 scripts/fork-rename.py          # apply
python3 scripts/fork-rename.py --check  # gate: `just check-fork-layers`, and
                                        # the pre-commit hook when a manifest is staged
```

`FORK_VERSION` in that script is the release process's number, not the sync's, and
it is the one place the number is authored: the script stamps every manifest from
it, `--check` fails on any that disagrees, and `scripts/publish.sh` reads it from
the same file rather than restating it. See [fork-publishing.md](fork-publishing.md)
for what the layer covers.

## 3. The build contour stays narrow

The default build and test set is CPU, wgpu and the non-backend crates. The
Apple, NVIDIA, AMD, Intel, TPU and embedded backends are workspace members but
are not built by default; the two browser-oriented crates and the Python binding
are excluded outright. Upstream has no `default-members` narrowing at all, and a
bare `cargo build` on its root manifest compiles every backend — a silent,
expensive regression in the routine loop rather than in the product.

New upstream members are added to `members` (they are new directories and cannot
conflict); `default-members` and `exclude` stay exactly as they are. The gates
follow the same narrowing: `just build` / `just test` / `just lint` are bare, and
`just build-workspace` / `just test-workspace` / `just lint-workspace` are their
once-per-sync whole-set counterparts. The narrowing is about the members this
fork does not build, not about the recipes: a gate over a crate that *is* in the
contour (`check-corpus`, `test-third-order-cpu`, `check-rustdoc-contour`) belongs
in the routine loop.

An excluded crate is out of the build, not out of the tree: nothing compiles it
and no gate runs its tests, but files inside it are still read. `Cargo.toml`'s
`exclude` comment is the authority on which ones and why; the short form is that
two gates read into `crates/bindings/pyrlx` (`rlx-bindgen`'s `_pyrlx.pyi` and
`bindings_release.py`'s `pyproject.toml`), two more follow links or rows out of
it, and `just torch-import` runs its pure-Python front-end — so deleting the
directory reddens four gates. Its `tests/` keeps one stub that asserts the
exclusion, because a suite that needs the extension cannot run and a missing
one makes `pytest` exit non-zero on an empty collection.

Integration tests are gated the same way — upstream's wgpu targets are
auto-discovered, which would silently join them to the routine `cargo test`:

```sh
python3 scripts/fork-test-gating.py          # apply
python3 scripts/fork-test-gating.py --check  # gate: `just check-fork-layers`,
                                             # and the pre-commit hook when a manifest is staged
```

Both gates are wired into `just ci` (through `check-fork-layers`) and into
`scripts/git-hooks/pre-commit`, so a sync that forgets a layer fails the routine
loop rather than the next release. `just ci` is the narrow contour. `just
ci-workspace` *depends on* `ci` and adds the whole-set counterparts —
`build-workspace`, `test-workspace`, `lint-workspace` — plus the whole-set
recipes, so it cannot fall behind the routine gate the way a hand-kept parallel
list does. It runs once, as a sync's own verification.

Two things that whole-set run does not cover by itself. `test-workspace` is
`cargo test --workspace`, which builds each member at its *default* features, so
a target behind `required-features = "backend-tests"` is still skipped — the wgpu
integration suite among them. `test-gpu` is what builds those, which is why
`ci-workspace` names it. `test-workspace` names one feature itself —
`--features mahbot-inference/hand-run` — because that crate's eleven hand-run
targets were part of the whole-set run before they were gated (its manifest says
why), and a whole-set run that stopped reaching them would lose coverage without
saying so.

`ci-workspace` has to be green on whatever host a sync happens to run on. A gate
that is red out of the box is not a gate — a sync reads the red as a regression
and chases it — so every recipe it names needs nothing beyond the Rust toolchain
and the contour's own devices, and a row that needs a toolkit or a device this
fork does not ship is not in it. Those live in `ci-rig`, which `ci-workspace`
does not name and a rig runs: `test-vulkan` (an `ash` instance needing a
registered ICD), `test-rocm` (HIP), `leak-check` (`leaks`),
`validate-metal-bindings` (`xcrun metal`), `test-libm` (a crates.io fetch — its
comparison workspace depends on the real `libm`, which the root
`[patch.crates-io]` shadows inside the workspace itself). Everything
`ci-workspace` does name keeps `RLX_REQUIRE_DEVICE=1`: a backend that skips for
want of a device must not report `ok`. Six of upstream's own tests likewise want a
dataset or a toolchain a developer machine does not have, so `test-workspace` skips
them by name; the recipe names each and what it needs.

`rlx-mlx-sys` / `rlx-mlx` stay members, but the fork does not ship the MLX
backend — a permanent decision, not a reading of the checkout, and a vendored
submodule happening to be present does not change it. So: the routine loop skips
them (`default-members`), `test-workspace` runs neither, and the Apple rows of
the GPU recipes name `metal` / `coreml` / `gpu` individually rather than `apple`,
which implies `mlx`. The whole-set *build*, *lint* and *doc* recipes still cover
them, because a member nothing ever compiles rots — and these had: the merge took
upstream's `rlx_mlx_shim.cpp`, which passes a `window_size` argument that a newer
MLX takes, over the vendored MLX v0.32.4 the crate pins, and nothing compiled it
until `ci-workspace` did. The shim now matches the vendored signature.

## 4. One implementation per job

Two routes for one job is how the backend rots, and it is the defect a merge
produces most readily because both sides appended to the same files. Where both
exist, measure and keep one.

The packed-weight area is where this bites hardest, because several names are
shared across the two designs with *different meanings*, and a mixture compiles
and returns wrong numbers:

- `dequant_gemm_q8_0_kernel` — same name, same bindings, same 32-byte uniform on
  both sides, but the uniform's words mean different things and the activation is
  viewed differently.
- `DequantGemvGgufParams` — same name and size; two words are an output stride
  and a row count on one side and padding on the other.
- `RLX_WGPU_Q8_0_GEMM_DISABLE` — returns to the scratch slab at any row count
  here, to the per-row multiply at eight rows and above upstream.
- `CoopKernel` — the fork replaced the 32-wide Vulkan f16 variant with an
  f16-in-f32-accumulator variant.

Whichever side survives, the other side's call sites are re-pointed deliberately,
and the shader's entry points are pruned to the surviving set. The scheme-coverage
gate parses the shader's branches, so extend it to any new fused entry point
before the merge lands:

```sh
cargo test -p mahbot-rlx-wgpu --features backend-tests --test gguf_kernel_scheme_coverage
```

### What the `a7a6c021` sync settled, and why

Two upstream designs were retired in favour of the fork's. One by measurement, one by
inference from a measurement — the distinction matters, because the next sync will be
tempted to re-litigate both.

- **The workgroup floor, measured.** `gguf_gemm_prefill_rate`'s round-robin pair over
  the cells the boundary runs through is in `gguf_gpu::COOP_MIN_WORKGROUPS`'s doc and
  is not copied here: it is a harness reading, and a second copy drifts. What it
  settled is that the floor belongs at 24 rather than 48 — the one change the
  comparison produced. Above the floor the cooperative tile is the faster of the
  two; below it the row-grouped tile is, and the product's two cells there keep it.
- **Upstream's 64x64 Q8_0 prefill tile, by inference.** Its one advantage is a weight
  pass per 64 rows instead of the fork's per 32 — two passes against three at `m = 96`
  — and it pays for that with half the workgroups at the same width (16 against 24 at
  `n = 1 024`). The sweep above prices workgroup count at these sizes and cannot price
  a pass count, so this is an inference from it rather than a head-to-head: a wider
  tile moves the one cell the floor was set on the wrong way along the curve. A sync
  that wants the tile back owes the pair on `q8_0 enc-down m96 k4096 n1024` itself, not
  another workgroup sweep.
- **Upstream's `attention_coop` entry, measured and retired.** It is the same job
  as this fork's fused one-row-per-workgroup walk in `attention.wgsl` — 64
  threads, one `(batch, head, query row)` per workgroup, the key sequence walked
  serially with an online softmax — so keeping both would be two routes for one
  job. The fused walk is also not the faster of the two where both are
  available; the reading, at the ASR encoder's geometry, is in
  `backend::attention_route::decompose_attention`'s doc next to the
  `RLX_WGPU_NO_ATTN_DECOMPOSE` switch that prices it, and is not copied here. The
  many-row arrangement is kept, and the fused walk stays as the form below
  `ATTENTION_DECOMPOSE_MIN_Q` and for shapes the many-row one refuses.

## 5. The shader compiler's loop guard is off, and the expansion stays

`kernels::kernel_module` builds this crate's shaders through
`create_shader_module_trusted` with `force_loop_bounding: false`; user WGSL
(`wgpu_gpu_custom`) does not come through it and keeps the guard. The opt-out is
an `unsafe` API, so the argument for it is part of the code: every loop in this
crate's WGSL makes guaranteed progress toward its own exit test, audited over
the whole set (the three shapes that are not an obvious counter, and what a
stride may be, are listed in `kernel_module`'s comment). Re-audit it whenever a
kernel gains a loop whose exit test, or the stride that reaches it, is computed
rather than read.

The one place the guard was paid for by hand — `attention_band.wgsl`, whose
loops are expanded so the driver keeps them in registers — keeps that expansion.
The flag does not reach the expanded form's speed, so an expansion is not made
redundant by turning the guard off; the four readings are in that shader's own
header, next to the expansion they price.

## 6. What must survive
Each of these was measured when it landed, so losing one is a regression even
where the tree still compiles. The list is a checklist, not the authority: the
authority is the fork's own diff against its base, enumerated before the first
merge action.

- encode chunking, so chunk n+1's host encode overlaps chunk n's execution
- the row-grouped 8-row packed GEMM and its split-k arm
- the cooperative-matrix packed GEMM
- the many-row and split decode attention routes
- broadcast-`Expand` elision in front of binary ops
- the mel framing fold
- 1×1 pointwise-GEMM and convolution route thresholds
- fused residual-plus-normalisation reduction
- packed-multiply and buffer-copy batch joins, tail-host deferral, KV row feed and
  deferred mirror delivery — all in the executor's main run loop, where losing
  one produces no compile error
- bounded weight upload at 128 MiB in flight
- ladder-rung and sound-window release mid-call
- the lazy f16 arena mirror
- release of host weight duplicates
- the parameter host stash kept only for the cooperative host-activation mirror that
  reads it, with a clone taking its parameters off the device instead
- packed codec tables read as stored
- the buffered GGUF header parse (upstream reads one scalar per `read` syscall)
- the scope-keyed shared weight pool with re-upload skipping
- KV-append copy elision
- the ceiling policy as a whole-arena verdict
- the fork-only core surfaces the product calls: the packed gather op in the
  backend's custom lowering, the dequant-guarded memory plan, sample seeds, and
  the stream-ceiling probe
- the mel and decode graph caches
- the sixteen-window grouped encoder pass
- capacity-bucketed body planes and plane seeding
- f16 parameters for the tagger

## 7. Memory is not a currency for buying speed

Footprint has gone down across this fork's whole history. A sync that ends
faster by holding more in memory has failed, even if the times look good.
Upstream's compiled-graph caches retain graphs without releasing them, which is
directly opposed to what this fork does mid-call; that collision is resolved in
favour of releasing.

What the columns mean when a sync is judged, so the argument is about the right
numbers:

- **Resting GPU is the model.** It is the arena, the shared weight pool and the
  staging buffers, and it is the column that must be equal or lower on every row.
- **Resting system is the code.** It is the process's footprint minus the
  graphics-tagged pages: the binary, the compiled schedules, the per-step
  uniforms and bind groups, the pipeline objects, and whatever host-side weight
  copies the caches hold. A few MiB of drift here is the price of more kernels
  and more per-graph bookkeeping, and it is acceptable *because it is that* — say
  so, with the evidence. If instead it is a cache entry, an arena or a graph
  outliving its call, it comes out however small it is.
- **Total peak is the measured lifetime maximum**, and it is what "must not be
  higher" applies to. `peak system` and `peak GPU` are the same reading split at
  different instants, so their sum is that maximum, but neither column alone is a
  peak of anything. Compare it against the baseline's *highest across its takes*,
  never against one take: this column moves by more between two takes of one
  unchanged binary than a sync is likely to move it.

One change, a batched multi-row decode, asked for a decision on total peak. It did not
ship, so the rule that total peak must not be higher stands unweakened. The figures are in
[batched-decode.md](batched-decode.md), not here, and the next proposal needs its own
measurement.

## 8. Records are files, and the comparison is paired takes

`STAND.md`, `TRANSCRIPTION-CORPUS.md` and `WAKE-WORD-BENCH.md` at the repo root
are the only place *record* figures live — no ticket log, no separate note. Each
is rewritten whole by the one run that can produce it, so history in version
control *is* the record of every take.

A reading that justifies a code decision has one home too, and it is not here:
the doc comment of the thing it justifies (`COOP_MIN_WORKGROUPS`,
`attention_route::decompose_attention`, `attention_band.wgsl`'s header,
`kernel_module`'s SAFETY note). This file names the decision and points there,
because a second copy of a number drifts and a number with no home is the note
this section exists to forbid.

Comparing a sync against the previous take:

- **Repeated paired takes, alternating the baseline and merged binaries in one
  session.** The per-round spread on this machine is large; one run compared with
  one run proves nothing. Keep the baseline binary aside — the merge rebuilds it.
- **Median of the rounds** is the statistic; the tree already carries that idiom
  in its rate harnesses (`time_rounds`, `time_round_robin`, `median`, win counts).
- **Warm times and resting footprints discriminate.** The peak columns do not, per
  take: each is a maximum sampled about once a millisecond, and they move by double
  digits between takes that agree to half a percent on warm time. §7 says which of
  them is the one to compare and against what — the total peak, against the
  baseline's highest across its takes.
- **Cold time is watched, not gated.** A merge that changes compilation order can
  move it without saying anything about steady-state speed. If it moves
  materially, say so.
- "Strictly faster" means the product as a whole — recognition and synthesis as a
  set — with no row worse than the baseline outside that spread. It does not mean
  every row improves by a particular margin.

`MAHBOT_MEASUREMENTS=<dir>` redirects a whole record to another directory, which
is how a take can be checked without editing the tree; a subset run
(`mahbot-inference 0 3`) prints its rows and writes nothing.

`MAHBOT_PHASE_SPLIT=1` prints one recognition call's phase split — tagger, VAD and
plan, encoder front, prompt prefill and decode — on stderr. No record carries it:
the six-row stand measures whole calls.

## 9. Re-pin, do not widen

Numerical results move, because upstream changes arithmetic and a sync changes
routes. Where a pinned witness in `crates/mahbot-inference/src/reference.rs`
moves, it is re-pinned deliberately with the reason recorded in that file — never
by widening a tolerance until it passes. The product's own behaviour is the gate,
not an op-level tolerance: transcripts must still be right, the synthesis round
trip must still be intelligible, and the wake path must not get worse.

## 10. The sequence

1. Commit the records and any in-flight work. Keep the baseline binary.
2. `git fetch upstream --tags`; pin the target commit in the ticket, not the tag.
3. Merge in a scratch clone, never in the working tree, and export the result as
   one patch: `git diff --binary HEAD` minus vendored submodules.
4. Resolve by port direction (§1) — the trivial files first (lockfiles, generated
   docs, submodule pointers), then the per-area groups.
5. Re-apply the rename layer and the test gating (§2, §3).
6. Regenerate what is generated and gate it: `docs/rlx-env-vars.md`,
   `docs/op-coverage.md`, license headers, the WGSL uniform-control-flow gate.
7. Run the whole set once — `just ci-workspace`, or `just ci-rig` on a rig (§3) —
   and the product checks the stand never runs: `just test-product-checks`, which
   is the only thing that reaches those `#[ignore]`d tests. It is not part of
   either gate: read what it says rather than expecting it green.
8. Re-take the three records whole at one digest (§8), and compare (§8).

## License

MIT OR Apache-2.0.
