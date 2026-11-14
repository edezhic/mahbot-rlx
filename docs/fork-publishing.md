# Publishing this fork's crates

This fork is native-only and exists to serve the MahBot models on the wgpu
backend. It publishes the crates its product paths actually use, under its own
package names, and nothing else. Staging uploads nothing; the publish driver
itself is described at the end.

## The set

Twenty-seven crates, each at the fork version (a prerelease: a version that
differs from upstream's only in build metadata is rejected by the registry).
The number lives in one place, `FORK_VERSION` in
[`scripts/fork-rename.py`](../scripts/fork-rename.py) — the release process
bumps it there, re-runs the script to stamp every manifest, and
`scripts/publish.sh` reads it from the same file rather than restating it.
`python3 scripts/fork-rename.py --check` fails if any manifest disagrees.

| package | source |
|---|---|
| `mahbot-inference` | `crates/mahbot-inference` |
| `mahbot-rlx-ir` | `crates/core/rlx-ir` |
| `mahbot-rlx-flow` | `crates/core/rlx-flow` |
| `mahbot-rlx-fusion` | `crates/core/rlx-fusion` |
| `mahbot-rlx-autodiff` | `crates/core/rlx-autodiff` |
| `mahbot-rlx-compile` | `crates/core/rlx-compile` |
| `mahbot-rlx-opt` | `crates/core/rlx-opt` |
| `mahbot-rlx-driver` | `crates/core/rlx-driver` |
| `mahbot-rlx-macros` | `crates/core/rlx-macros` |
| `mahbot-rlx-runtime` | `crates/core/rlx-runtime` |
| `mahbot-rlx-unfuse` | `crates/core/rlx-unfuse` |
| `mahbot-rlx-cpu` | `crates/backends/rlx-cpu` |
| `mahbot-rlx-wgpu` | `crates/backends/rlx-wgpu` |
| `mahbot-rlx-gpu-dispatch` | `crates/backends/rlx-gpu-dispatch` |
| `mahbot-rlx-gpu-host` | `crates/backends/rlx-gpu-host` |
| `mahbot-rlx-gguf` | `crates/io/rlx-gguf` |
| `mahbot-rlx-mlx-io` | `crates/io/rlx-mlx-io` |
| `mahbot-rlx-pkg` | `crates/io/rlx-pkg` |
| `mahbot-rlx-nemo` | `crates/io/rlx-nemo` |
| `mahbot-rlx-dduf` | `crates/io/rlx-dduf` |
| `mahbot-rlx-torch-ckpt` | `crates/io/rlx-torch-ckpt` |
| `mahbot-rlxsl` | `crates/tooling/rlxsl` |
| `mahbot-rlx-libm` | `crates/numerics/rlx-libm` |
| `mahbot-rlx-fastmath` | `crates/numerics/rlx-fastmath` |
| `mahbot-rlx-search` | `crates/numerics/rlx-search` |
| `mahbot-rlx-linux-ref` | `crates/numerics/rlx-linux-ref` |
| `mahbot-rlx-pocketfft` | `crates/numerics/rlx-pocketfft` |

Every other package in the workspace keeps upstream's name and version and is
not published from here.

## What the rename does not touch

- **Dependency keys.** A manifest still writes `rlx-ir = { workspace = true }`;
  the workspace entry names the package (`package = "mahbot-rlx-ir"`) and the
  fork version. The keys, and the source files that import them, are unchanged.
- **Module names.** Every renamed crate pins `[lib] name` to the module
  upstream's sources use (`rlx_ir`, `rlx_wgpu`, …), so no source file changes.
- **Metadata.** Each crate in the set declares the fork's repository
  (`https://github.com/edezhic/mahbot-rlx`) and its own documentation page
  (`https://docs.rs/<package>`). `scripts/publish.sh` refuses a crate that
  does not.

## Readiness without a registry

`cargo package` resolves a crate's dependencies against the registry, so an
unpublished dependency blocks archive production entirely. `scripts/publish.sh
--stage [DIR]` demonstrates readiness locally instead:

1. It packages the crates tier by tier with `cargo package --offline
   --no-verify`, resolving each against the archives staged before it. The
   store (default `$TMPDIR/mahbot-publish-store`) stands in for the registry:
   `crates/` holds the archives, `src/` their unpacked copies, and `fork.toml`
   the `[patch.crates-io]` table that points at them.
2. It checks every packaged manifest for the name and version it should carry,
   and for the absence of any `path` dependency — an archive that names one
   could not stand alone.
3. It extracts every fork archive into `build/` and builds it from itself with
   only the store beside it. `mahbot-inference` runs its own test suite there —
   the packaged crate carries no file from the repository, so its two record
   checks skip when the records and the git tree are absent.

## The one limitation

`mahbot-rlx-runtime` keeps its fourteen optional out-of-scope dependencies
(twelve backends — `rlx-metal`, `rlx-mlx`, `rlx-coreml`, `rlx-cuda`,
`rlx-rocm`, `rlx-vulkan`, `rlx-oneapi`, `rlx-tpu`, `rlx-qnn`, `rlx-xdna`,
`rlx-egpu`, `rlx-fpga` — plus the `rlx-optim` and `rlx-hwprofile` helpers)
under their upstream names, because deleting them would break features and
churn source for no gain. They are off in the product configuration, so nothing
the MahBot paths build depends on an upstream `rlx-*` package. Enabling one of
them is not a supported configuration: it resolves the upstream family, whose
`rlx-ir` is a second `rlx_ir` library that cannot coexist with the fork's.

That is why staging packages seventeen out-of-scope crates from this tree as
*resolution stand-ins* — the fourteen above plus `rlx-gpu-kernels`, `rlx-mlx-sys`
and `rlx-cortexm`, which they pull in: the archive has to resolve their names,
and `--offline` forbids a registry contact. The stand-ins never enter the fork's
archives and are never uploaded; the packaged manifests still name the registry
packages.

## Publishing

`scripts/publish.sh --list` prints the tier order and versions,
`--plan` the crates the index does not have yet, and the default mode uploads
in tier order, pacing each upload against the crates.io bucket it draws from: a
brand-new name allows a burst of five and then one per ten minutes, a new
version a burst of thirty and then one per minute
(<https://crates.io/docs/rate-limits>). Its `SCOPE` array is the ground truth: a
tier naming a package the fork does not own, a crate missing from the tiers, a
crate that is not publishable, a version that is not the fork version, or a
product-configuration dependency on the upstream family all fail the run before
anything is built or uploaded.

## License

MIT OR Apache-2.0.
