#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# scripts/publish.sh — publish driver for the fork's own crates.
#
# Publishes the crates this fork owns (`SCOPE` below) in dependency-tier order,
# one at a time with rate-limit-respecting delays AND active polling of the
# sparse index until each just-published version is resolvable. This combination
# is what keeps the script safe to leave running unattended:
#
#   * Registry buckets: crates.io allows a burst of 5 brand-new names and then
#     one every 10 minutes, and a burst of 30 new versions of existing names and
#     then one a minute (https://crates.io/docs/rate-limits). The wait before
#     each upload follows the bucket that upload draws from — a brand-new name
#     is paced at `NEW_CRATE_REFILL`, a new version at `NEW_VERSION_REFILL` —
#     so the driver never plans an upload the registry is certain to refuse. An
#     unreadable index and a resumed run both take the slower reading.
#     `MIN_INTERVAL` is only the settle floor inside a burst.
#   * Active readiness check: after every successful `cargo publish`,
#     poll `https://index.crates.io/<partition>/<crate>` until the
#     just-uploaded version appears. Until that happens, the next
#     crate's `cargo publish` would either 404 on dep resolution or
#     get stale metadata.
#   * Skip published: before each upload, resolve that crate's effective
#     version from its Cargo.toml. Skip only when *that* version is already on
#     the sparse index — so a crate bumped to 0.2.2 is still published even
#     when 0.2.1 is on crates.io. Use `--no-skip-published` to force an upload.
#   * Registry HTTP errors (429, 408, 5xx, timeouts): parse cargo output as
#     it finishes, sleep until crates.io's "try again after <GMT>" when
#     present (+ pad), else status-specific backoff — then retry the same
#     crate automatically until it uploads. --max-retries 0 (default) never
#     gives up on retryable codes. The status is read from what cargo prints:
#     `(status 429 …)` or `(429 …)`, depending on the release.
#   * Missing registry deps: poll the sparse index for the dependency,
#     then retry (no manual --start-crate for index lag).
#
# Crates.io documented limits (https://crates.io/docs/rate-limits):
#   * New crates:        burst 5, then 1 per 10 minutes
#   * New versions:      burst 30, then 1 per minute
#   * Sparse-index serve: typically <30 s, occasionally up to ~5 min
#
# Default flow:
#
#   1. Pre-flight over the fork's own contour — the `default-members` selection
#      in Cargo.toml, which already includes `mahbot-inference` — and nothing
#      else: `cargo fmt --check`, `cargo clippy --all-targets --features
#      backend-tests -- -D warnings`, `cargo test --release --no-fail-fast
#      --features backend-tests`, each over that selection. Aborts if any fails.
#      The out-of-scope backends are never built, so their test suites are never
#      run: a defect in code this fork does not ship cannot stop a release.
#   2. Confirm prompt (skip with --yes; a run with nothing to read refuses).
#   3. Per tier, publish each crate sequentially. After each crate:
#        a. wait for the sparse index to report the new version
#           (poll interval `POLL_INTERVAL`, hard cap `POLL_TIMEOUT`),
#        b. sleep `MIN_INTERVAL` (rate-limit safety floor),
#      before issuing the next publish. After the last crate of a
#      tier the loop additionally sleeps `BETWEEN_DELAY` to let
#      downstream crates' dep resolution catch up.
#   4. `SCOPE` below is the publication tooling's ground truth: the crates this
#      fork owns, under fork package names at the fork version. `TIERS` must
#      cover exactly it, every name in it must be a workspace package, and every
#      crate in it must be publishable — `validate_scope` enforces all three.
#      Nothing else in the workspace is uploaded from here.
#
# Staging (no registry, no upload):
#
#   scripts/publish.sh --stage [DIR]                # default: $TMPDIR/mahbot-publish-store
#
#   `cargo package --offline` produces one archive per crate in tier order,
#   each resolved against the archives staged before it: the local store stands
#   in for the registry, so every crate in `SCOPE` yields an archive without a
#   registry contact. Every archive is then extracted and built from itself —
#   `mahbot-inference` runs its own test suite. See `docs/fork-publishing.md`,
#   including the one stated limitation: `mahbot-rlx-runtime`'s optional
#   out-of-scope backends keep upstream's names.
#
# Usage:
#
#   scripts/publish.sh --stage                      # local archives, no upload
#   scripts/publish.sh --stage /tmp/store           # …into a chosen directory
#   scripts/publish.sh --dry-run                    # safe — no upload
#   scripts/publish.sh --list                       # tier order + per-crate versions
#   scripts/publish.sh --plan                       # crates needing publish only
#   scripts/publish.sh --no-gate                    # skip fmt/clippy/test
#   scripts/publish.sh --yes                        # confirm without a prompt (needed when stdin is closed)
#   scripts/publish.sh --start-tier 3               # resume from tier 3
#   scripts/publish.sh --start-crate mahbot-rlx-runtime  # resume from a crate
#   scripts/publish.sh --min-interval 90            # settle floor inside a burst (sec)
#   scripts/publish.sh --between-delay 120          # between-tier extra (sec)
#   scripts/publish.sh --poll-interval 10           # index-poll interval
#   scripts/publish.sh --poll-timeout 600           # index-poll cap (sec)
#   scripts/publish.sh --max-retries 5              # cap registry retries (0 = unlimited)
#   scripts/publish.sh --rate-limit-pad 15          # pad after server retry-after
#   scripts/publish.sh --no-skip-published          # upload even if version exists
#   scripts/publish.sh --no-verify                  # skip cargo's local rebuild
#   scripts/publish.sh --no-poll                    # disable index polling
#
# Resuming:
#
#   If a publish fails (network blip, transient crates.io error),
#   re-run with `--start-crate <name>` (or `--start-tier N`) to pick
#   up where you stopped. Already-published crates aren't revisited.
#
# Dry-run limitations:
#
#   `cargo publish --dry-run` does a full local rebuild + dep
#   resolution against crates.io. For NON-LEAF crates (tier ≥ 1)
#   that fails until their leaf deps actually exist on crates.io at
#   the new version. The script's --dry-run mode handles this by
#   recording the resolution-failure crates separately at the end —
#   metadata + packaging are still validated for them. `--stage` is the
#   mode that demonstrates readiness without a registry at all. Run
#   `scripts/publish.sh --list` to print the full tier order;
#   `validate_scope` checks every dependency against that order before
#   publishing.
#
# Environment:
#
#   CARGO_REGISTRY_TOKEN — must be set, or run `cargo login` first.
#                          Not validated here; cargo errors clearly.

set -euo pipefail

DRY_RUN=0
LIST_ONLY=0
PLAN_ONLY=0
NO_GATE=0
NO_VERIFY=0
NO_POLL=0
SKIP_PUBLISHED=1
ASSUME_YES=0
MIN_INTERVAL=65          # settle floor between uploads inside a burst (sec)
NEW_CRATE_BURST=5        # brand-new names allowed back to back
NEW_CRATE_REFILL=600     # …then one every 10 minutes (sec)
NEW_VERSION_BURST=30     # new versions of existing names allowed back to back
NEW_VERSION_REFILL=60    # …then one a minute (sec)
BETWEEN_DELAY=90         # extra sleep at tier boundaries (sec)
POLL_INTERVAL=10         # index poll cadence (sec)
POLL_TIMEOUT=600         # max time we'll wait for the index (sec)
MAX_RETRIES=0            # cap on registry retries per crate (0 = unlimited)
RATE_LIMIT_PAD=15        # extra seconds after server "try again after" time
START_TIER=0
START_CRATE=""
LAST_PUBLISH_ERR=""      # temp log from the last failed publish attempt
STAGING=0
STAGE_DIR=""             # `--stage [DIR]`; empty means the default below

# The crates this fork owns: exactly what its product paths use, under fork
# package names at the fork version. This array is the ground truth the gates
# read — `TIERS` must cover exactly it, every name must be a workspace package,
# and every crate in it must be publishable. Everything else in the workspace
# keeps upstream's identity and is never uploaded from here.
#
# Written out a second time rather than flattened out of `TIERS` on purpose: the
# gate compares the two lists, so a crate dropped from `TIERS` fails here instead
# of silently shrinking the set a run would publish.
SCOPE=(
    mahbot-rlx-ir
    mahbot-rlxsl
    mahbot-rlx-gguf
    mahbot-rlx-torch-ckpt
    mahbot-rlx-nemo
    mahbot-rlx-gpu-dispatch
    mahbot-rlx-macros
    mahbot-rlx-dduf
    mahbot-rlx-unfuse
    mahbot-rlx-flow
    mahbot-rlx-fusion
    mahbot-rlx-driver
    mahbot-rlx-mlx-io
    mahbot-rlx-autodiff
    mahbot-rlx-pkg
    mahbot-rlx-compile
    mahbot-rlx-opt
    mahbot-rlx-cpu
    mahbot-rlx-gpu-host
    mahbot-rlx-wgpu
    mahbot-rlx-runtime
    mahbot-inference
)

# The fork version every crate in `SCOPE` carries. A prerelease on purpose: a
# version differing from upstream's only in build metadata is rejected by the
# registry.
FORK_VERSION="0.2.18-mahbot.1"

# Tier definitions. Each array entry is a single tier; space-separated crate
# names within. Order within a tier matters when one member depends on another
# in the same tier. `cargo publish` resolves every versioned dependency —
# normal, build and dev, optional included — against the registry, so a crate
# must follow everything it names; path-only dev-deps (no `version =`) are
# stripped from the packaged manifest and impose no order.
#
# Leaves first: ir, rlxsl (its build-dep), the weight-format readers, the
# dependency-free dispatch table and proc macros.
#   - rlx-unfuse / rlx-flow / rlx-fusion / rlx-driver / rlx-mlx-io sit on rlx-ir.
#   - rlx-autodiff (rlx-ir + rlx-fusion) and rlx-pkg (ir + the weight readers).
#   - rlx-compile (ir + fusion + optional autodiff), then rlx-opt (compile).
#   - rlx-cpu and rlx-gpu-host precede rlx-wgpu, which takes both; rlxsl is its
#     build-dependency and rides tier 0.
#   - rlx-runtime names every in-scope crate above it (ir, driver, opt, macros,
#     compile, gguf, gpu-dispatch, cpu, wgpu, pkg, mlx-io, dduf) and its
#     versioned dev-deps (autodiff, flow, fusion, rlxsl, compile).
#   - mahbot-inference names rlx-flow, rlx-gguf, rlx-ir, rlx-runtime and
#     rlx-wgpu: the product path, last.
#
# The out-of-scope crates this fork does not publish are absent on purpose;
# mahbot-rlx-runtime's optional backends still resolve upstream, which
# `validate_scope` reports and docs/fork-publishing.md states.
TIERS=(
    "mahbot-rlx-ir mahbot-rlxsl mahbot-rlx-gguf mahbot-rlx-torch-ckpt mahbot-rlx-nemo mahbot-rlx-gpu-dispatch mahbot-rlx-macros mahbot-rlx-dduf"
    "mahbot-rlx-unfuse mahbot-rlx-flow mahbot-rlx-fusion mahbot-rlx-driver mahbot-rlx-mlx-io"
    "mahbot-rlx-autodiff mahbot-rlx-pkg"
    "mahbot-rlx-compile"
    "mahbot-rlx-opt"
    "mahbot-rlx-cpu mahbot-rlx-gpu-host mahbot-rlx-wgpu"
    "mahbot-rlx-runtime"
    "mahbot-inference"
)

usage() {
    # Print the whole header comment block (line 2 through the last `#` line
    # before `set -euo pipefail`) — a hard-coded end line silently truncated
    # `--help` mid-section as the header grew.
    local last
    last="$(grep -n -m1 '^set -euo pipefail' "$0" | cut -d: -f1)"
    sed -n "2,$((last - 2))p" "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --stage)          STAGING=1
                          if (( $# > 1 )) && [[ "$2" != -* ]]; then STAGE_DIR="$2"; shift; fi
                          shift ;;
        --stage=*)        STAGE_DIR="${1#--stage=}"; STAGING=1; shift ;;
        --list)           LIST_ONLY=1; shift ;;
        --plan)           PLAN_ONLY=1; shift ;;
        --no-gate)        NO_GATE=1; shift ;;
        --no-verify)      NO_VERIFY=1; shift ;;
        --no-poll)        NO_POLL=1; shift ;;
        --no-skip-published) SKIP_PUBLISHED=0; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --min-interval)   MIN_INTERVAL="$2"; shift 2 ;;
        --between-delay)  BETWEEN_DELAY="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --poll-timeout)   POLL_TIMEOUT="$2"; shift 2 ;;
        --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
        --rate-limit-pad) RATE_LIMIT_PAD="$2"; shift 2 ;;
        --start-tier)     START_TIER="$2"; shift 2 ;;
        --start-crate)    START_CRATE="$2"; shift 2 ;;
        --help|-h)        usage ;;
        # Legacy aliases for the old --within-delay flag — map onto
        # --min-interval so older invocations don't silently no-op.
        --within-delay)   MIN_INTERVAL="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "run with --help for usage" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

# ── Gates ───────────────────────────────────────────────────────
#
# One gate, because both questions — does the tier list cover the scope, and
# does the order resolve — are answered from the same metadata. It reads
# `cargo metadata`, so it sees the manifests rather than a hand-copy of them. A
# gate that cannot see the crates must not pass: a missing python3 or a metadata
# failure is a hard error, never a skip.
validate_scope() {
    if ! command -v python3 >/dev/null 2>&1; then
        red "python3 not found — cannot validate the publish scope."
        exit 1
    fi
    local err notes rc=0
    err="$(python3 - "$PWD" "$FORK_VERSION" "${SCOPE[*]}" "${TIERS[@]}" 2>&1 <<'PY'
import json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
fork_version = sys.argv[2]
scope = sys.argv[3].split()
tiers = [tier.split() for tier in sys.argv[4:]]

try:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout)
except Exception as error:  # any failure here is a gate failure
    print(f"cargo metadata failed: {error}")
    sys.exit(2)

members = set(meta["workspace_members"])
packages = {p["name"]: p for p in meta["packages"] if p["id"] in members}

problems = []
notes = []
position = {}

for tier_index, tier in enumerate(tiers):
    for slot, name in enumerate(tier):
        if name in position:
            problems.append(f"{name} is listed twice in TIERS")
        position[name] = (tier_index, slot)
        if name not in scope:
            problems.append(f"{name} is in TIERS but not in SCOPE — the fork does not own that name")
        if name not in packages:
            problems.append(f"{name} is in TIERS but is not a workspace package")

for name in scope:
    package = packages.get(name)
    if name not in position:
        problems.append(f"{name} is in SCOPE but missing from TIERS")
    if package is None:
        problems.append(f"{name} is in SCOPE but is not a workspace package")
        continue
    if package.get("publish") == []:
        problems.append(f"{name} is in SCOPE but its manifest says `publish = false`")
    if package["version"] != fork_version:
        problems.append(f"{name} is at {package['version']}, not the fork version {fork_version}")
    if package.get("repository") != "https://github.com/edezhic/mahbot-rlx":
        problems.append(f"{name} does not declare the fork's repository")
    if package.get("documentation") != f"https://docs.rs/{name}":
        problems.append(f"{name} does not declare its own documentation page")

for name in scope:
    package = packages.get(name)
    if package is None or name not in position:
        continue
    mine = position[name]
    for dep in package["dependencies"]:
        dep_name = dep["name"]
        if dep_name in scope:
            # Path-only dev-deps are stripped from the packaged manifest, so
            # they impose no order.
            if dep["kind"] == "dev" and dep["req"] == "*":
                continue
            if dep_name in position and position[dep_name] >= mine:
                problems.append(
                    f"{name} (tier {mine[0]}) needs {dep_name} staged earlier "
                    f"(tier {position[dep_name][0]})"
                )
        elif dep_name.startswith("rlx"):
            # The upstream family. Anything the product configuration resolves
            # from it is a defect; an optional backend is the stated limitation.
            path_only = dep["req"] == "*"
            if dep["kind"] == "dev":
                if not path_only:
                    problems.append(
                        f"{name}: dev-dep {dep_name} carries a version, so publishing "
                        f"would resolve the upstream family — make it path-only"
                    )
            elif dep["optional"]:
                notes.append(f"{name}: optional {dep_name} resolves the upstream family when enabled")
            else:
                problems.append(
                    f"{name} depends on upstream {dep_name} in its product configuration"
                )

if problems:
    for problem in problems:
        print(problem)
    sys.exit(1)

for note in notes:
    print(f"note: {note}")
PY
)" || rc=$?
    # Exit 1 is the script's own "the scope does not match"; anything else — a
    # killed interpreter included — means the gate never reached a verdict.
    if (( rc > 1 )); then
        red "The publish scope gate could not run (python3 exit $rc)."
        [[ -n "$err" ]] && printf '%s\n' "$err" >&2
        exit 1
    fi
    notes="$(printf '%s\n' "$err" | grep '^note: ' || true)"
    err="$(printf '%s\n' "$err" | grep -v '^note: ' || true)"
    if [[ -n "$notes" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && yellow "  ${line#note: }"
        done <<< "$notes"
    fi
    if [[ -n "$err" ]]; then
        red "The publish scope does not match the manifests:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && red "  $line"
        done <<< "$err"
        exit 1
    fi
}

validate_scope

# ── Sparse-index helpers (needed for per-crate skip/plan checks) ─
sparse_index_path() {
    local name="$1"
    local n="${#name}"
    if   (( n == 1 )); then printf '1/%s\n'     "$name"
    elif (( n == 2 )); then printf '2/%s\n'     "$name"
    elif (( n == 3 )); then printf '3/%s/%s\n'  "${name:0:1}" "$name"
    else                    printf '%s/%s/%s\n' "${name:0:2}" "${name:2:2}" "$name"
    fi
}

# Returns 0 if the sparse index serves <version> for <crate>, 1 if
# the version isn't there yet, or 2 if the crate itself isn't in the
# index at all (pre-first-publish — perfectly fine, just means we
# should keep polling).
check_index() {
    local crate="$1"
    local version="$2"
    local url="https://index.crates.io/$(sparse_index_path "$crate")"
    local answer code body
    answer="$(curl -sS --max-time 10 -w '\n%{http_code}' "$url" 2>/dev/null || true)"
    code="${answer##*$'\n'}"
    body="${answer%$'\n'*}"
    case "$code" in
        404) return 2 ;;   # the name is not on the index at all
        200) ;;
        *)   return 3 ;;   # the index could not be read; not a classification
    esac
    if printf '%s' "$body" | grep -Fq "\"vers\":\"$version\""; then
        return 0
    fi
    return 1
}

# Per-crate publish versions, read from the manifests cargo itself resolved.
CRATE_VERSIONS_FILE=""

load_crate_versions() {
    CRATE_VERSIONS_FILE="$(mktemp)"
    python3 - "$PWD" "${SCOPE[*]}" <<'PY' > "$CRATE_VERSIONS_FILE"
import json, subprocess, sys

scope = sys.argv[2].split()
meta = json.loads(subprocess.run(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"],
    cwd=sys.argv[1], capture_output=True, text=True, check=True,
).stdout)
by_name = {p["name"]: p for p in meta["packages"]}
for name in scope:
    print(f'{name}\t{by_name[name]["version"]}')
PY
}

crate_version() {
    local crate="$1"
    local ver
    ver="$(awk -F'\t' -v c="$crate" '$1 == c { print $2; exit }' "$CRATE_VERSIONS_FILE")"
    if [[ -z "$ver" ]]; then
        red "unknown crate in version map: $crate" >&2
        exit 1
    fi
    echo "$ver"
}

load_crate_versions
trap 'rm -f "${CRATE_VERSIONS_FILE:-}"' EXIT

# Crates whose effective local version is not yet on the sparse index.
crates_needing_publish() {
    local tier c ver
    for tier in "${TIERS[@]}"; do
        for c in $tier; do
            ver="$(crate_version "$c")"
            if ! check_index "$c" "$ver"; then
                echo "$c@$ver"
            fi
        done
    done
}

list_tiers() {
    local with_index="${1:-0}"
    bold "Publish order (fork packages, version $FORK_VERSION):"
    local i=0 tier c ver suffix
    for tier in "${TIERS[@]}"; do
        echo "  tier $i:"
        for c in $tier; do
            ver="$(crate_version "$c")"
            suffix=""
            if (( with_index )) && check_index "$c" "$ver"; then
                suffix="  (already on crates.io)"
            fi
            echo "    - $c@$ver$suffix"
        done
        i=$(( i + 1 ))
    done
    echo
    echo "  Nothing else in this workspace is published from here."
}

print_publish_plan() {
    local -a need=()
    local entry
    while IFS= read -r entry; do
        [[ -z "$entry" ]] && continue
        need+=("$entry")
    done < <(crates_needing_publish)

    if (( ${#need[@]} == 0 )); then
        green "Nothing to publish — every listed crate is already at its local version on crates.io."
        return 0
    fi

    bold "Crates needing publish (${#need[@]}):"
    for entry in "${need[@]}"; do
        echo "  - $entry"
    done
}

# ── Local staging: readiness without a registry ──────────────────
#
# `cargo package --offline` produces one archive per crate, each resolved
# against the archives staged before it, so the store stands in for the
# registry:
#
#   <store>/crates/<pkg>-<ver>.crate   the archives
#   <store>/src/<pkg>-<ver>/           unpacked, what the patches point at
#   <store>/fork.toml                  [patch.crates-io] for everything staged
#   <store>/target/                    cargo's packaging target directory
#   <store>/build/<pkg>-<ver>/         a fresh extraction, built from itself
#
# The plan is the tier order with the out-of-scope crates the scope still
# resolves inserted before the crate that needs them. Those stand-ins are the
# stated limitation: `mahbot-rlx-runtime` keeps its optional out-of-scope
# backends (rlx-metal, rlx-cuda, …) under upstream names, so its archive has to
# resolve the upstream family. Staging them from this tree only lets that
# resolution finish offline — the packaged manifests still name the registry
# packages, and a consumer who enables one of those features pulls upstream.
# They are never verified as deliverables and never uploaded. See
# docs/fork-publishing.md.
STAGE_DEFAULT="${TMPDIR:-/tmp}/mahbot-publish-store"

# One [patch.crates-io] line per staged archive, rewritten whole.
write_fork_patch() {
    printf '[patch.crates-io]\n' > "$1"
    local c
    if (( ${#PATCHED[@]} )); then
        for c in "${PATCHED[@]}"; do
            printf '%s = { path = "%s/src/%s-%s" }\n' \
                "${c%@*}" "$STORE" "${c%@*}" "${c#*@}" >> "$1"
        done
    fi
}

# The flat staging plan: `<scope|stand-in>\t<package>\t<version>` per line, each
# package after everything it resolves from the registry.
staging_plan() {
    python3 - "$PWD" "${SCOPE[*]}" "${TIERS[@]}" <<'PY'
import json, subprocess, sys

root = sys.argv[1]
scope = sys.argv[2].split()
tiers = [tier.split() for tier in sys.argv[3:]]

meta = json.loads(subprocess.run(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"],
    cwd=root, capture_output=True, text=True, check=True,
).stdout)
packages = {p["name"]: p for p in meta["packages"]}

plan = []
seen = set()

def visit(name):
    if name in seen:
        return
    seen.add(name)
    for dep in packages[name]["dependencies"]:
        dep_name = dep["name"]
        if dep["kind"] == "dev" or dep_name in scope or dep_name not in packages:
            continue
        visit(dep_name)
    plan.append(("scope" if name in scope else "stand-in", name))

for tier in tiers:
    for name in tier:
        visit(name)

for kind, name in plan:
    print(f"{kind}\t{name}\t{packages[name]['version']}")
PY
}

# The packaged manifest must name the package and carry no path dependency: a
# path in it would mean the archive cannot stand alone.
check_archive() {
    local manifest="$1" crate="$2" version="$3"
    python3 - "$manifest" "$crate" "$version" <<'PY'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
name, version = sys.argv[2], sys.argv[3]

def table(header: str) -> str:
    match = re.search(rf"(?ms)^\[{re.escape(header)}\][^\n]*\n(.*?)(?=^\[|\Z)", text)
    return match.group(1) if match else ""

problems = []
if not re.search(rf'(?m)^name = "{re.escape(name)}"$', text):
    problems.append(f"packaged name is not {name}")
if not re.search(rf'(?m)^version = "{re.escape(version)}"$', text):
    problems.append(f"packaged version is not {version}")
for header in re.findall(r"(?m)^\[((?:target\.[^\]]+\.)?(?:dev-|build-)?dependencies(?:\.[^\]]+)?)\]", text):
    if re.search(r"(?m)^path = ", table(header)):
        problems.append(f"[{header}] carries a path dependency")
if problems:
    for problem in problems:
        print(f"{name}: {problem}")
    sys.exit(1)
PY
}

stage_one() {
    local kind="$1" crate="$2" version="$3" archive log
    bold ">> packaging $crate@$version ($kind)"
    log="$STORE/package-$crate.log"
    if ! cargo package --package "$crate" --no-verify --allow-dirty --offline \
            --target-dir "$STORE/target" --config "$STORE/fork.toml" \
            > "$log" 2>&1; then
        red "packaging $crate failed:"
        tail -20 "$log"
        exit 1
    fi
    archive="$STORE/target/package/$crate-$version.crate"
    if [[ ! -f "$archive" ]]; then
        red "cargo packaged $crate but left no archive at $archive"
        exit 1
    fi
    cp "$archive" "$STORE/crates/"
    rm -rf "${STORE:?}/src/$crate-$version"
    mkdir -p "$STORE/src/$crate-$version"
    tar -xzf "$archive" -C "$STORE/src/$crate-$version" --strip-components=1
    if ! check_archive "$STORE/src/$crate-$version/Cargo.toml" "$crate" "$version"; then
        red "$crate: the packaged manifest is not standalone"
        exit 1
    fi
    PATCHED+=("$crate@$version")
    write_fork_patch "$STORE/fork.toml"
}

stage_archives() {
    mkdir -p "$STORE"/{crates,src,target,build}
    PATCHED=()
    write_fork_patch "$STORE/fork.toml"
    local kind crate version plan rc=0
    plan="$(staging_plan)" || rc=$?
    if (( rc != 0 )) || [[ -z "$plan" ]]; then
        red "The staging plan could not be read from cargo metadata (exit $rc)."
        exit 1
    fi
    while IFS=$'\t' read -r kind crate version; do
        [[ -z "$crate" ]] && continue
        stage_one "$kind" "$crate" "$version"
    done <<< "$plan"
}

# Build each archive from itself, with nothing of the repository beside it but
# the store. `mahbot-inference` runs its own test suite there, which is the
# acceptance its packaged form has to meet.
verify_archives() {
    local tier crate dir log failed=0
    for tier in "${TIERS[@]}"; do
        for crate in $tier; do
            dir="$STORE/build/$crate-$FORK_VERSION"
            log="$STORE/build-$crate.log"
            rm -rf "$dir"
            mkdir -p "$dir"
            tar -xzf "$STORE/crates/$crate-$FORK_VERSION.crate" -C "$dir" --strip-components=1
            local -a run=(build)
            if [[ "$crate" == "mahbot-inference" ]]; then run=(test); fi
            bold ">> ${run[0]} $crate from its own archive"
            if ( cd "$dir" && cargo "${run[@]}" --offline \
                    --config "$STORE/fork.toml" > "$log" 2>&1 ); then
                green "   ok"
            else
                red "   FAILED to ${run[0]} from its own archive:"
                grep -E '^error' "$log" | head -5 || true
                failed=1
            fi
        done
    done
    return "$failed"
}

stage_all() {
    STORE="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' \
        "${STAGE_DIR:-$STAGE_DEFAULT}")"
    # Packaging under a patch table makes cargo record those patches in the
    # workspace lock (`[[patch.unused]]` and a rewritten dependency graph), so
    # the run puts back the lock it found however it ends.
    LOCK_BACKUP="$(mktemp)"
    cp Cargo.lock "$LOCK_BACKUP"
    trap 'cp "$LOCK_BACKUP" Cargo.lock; rm -f "$LOCK_BACKUP" "${CRATE_VERSIONS_FILE:-}"' EXIT
    local workspace_root
    workspace_root="$(pwd -P)"
    if [[ "$STORE" == "/" ]]; then
        red "the store must not be the filesystem root."
        exit 1
    fi
    case "$STORE/" in
        "$workspace_root"/*)
            red "the store must live outside the workspace — cargo would adopt the"
            red "extracted archives into it and refuse to build them from themselves."
            exit 1
            ;;
    esac
    mkdir -p "$STORE"
    bold "Staging the fork's crates into $STORE"
    echo
    stage_archives
    echo
    bold "Building every archive from itself"
    verify_archives || {
        red "At least one archive does not build from itself — see the logs in $STORE."
        exit 1
    }
    echo
    green "Staged ${#SCOPE[@]} fork archives and $(( ${#PATCHED[@]} - ${#SCOPE[@]} )) resolution stand-ins; every fork archive builds from itself."
    yellow "Nothing was uploaded."
}

if (( STAGING )); then
    stage_all
    exit 0
fi

if (( LIST_ONLY )); then
    list_tiers 1
    echo
    print_publish_plan
    exit 0
fi

if (( PLAN_ONLY )); then
    print_publish_plan
    exit 0
fi

# ── Pre-flight gates ────────────────────────────────────────────
#
# The packages a bare `cargo` invocation builds: the `default-members` selection
# in Cargo.toml, which already includes `mahbot-inference`. Read from the
# metadata, so the gates cannot drift from the manifest.
contour_packages() {
    python3 - <<'PY'
import json, subprocess

meta = json.loads(subprocess.run(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"],
    capture_output=True, text=True, check=True,
).stdout)
selected = set(meta["workspace_default_members"])
print("\n".join(sorted(p["name"] for p in meta["packages"] if p["id"] in selected)))
PY
}

if (( ! NO_GATE )); then
    # `clippy` and `test` take the contour by default; `cargo fmt` takes only an
    # explicit package list (bare, at the workspace root, it formats every
    # member), so the contour is passed to it by name.
    FMT_ARGS=()
    while IFS= read -r name; do FMT_ARGS+=(-p "$name"); done < <(contour_packages)
    if (( ${#FMT_ARGS[@]} == 0 )); then
        red "The fork's contour could not be read from cargo metadata."
        exit 1
    fi
    bold "[1/3] cargo fmt --check (the fork's contour)"
    cargo fmt "${FMT_ARGS[@]}" -- --check

    bold "[2/3] cargo clippy --all-targets --features backend-tests -- -D warnings"
    # `backend-tests`: the default test selection is a fast smoke set;
    # a release gate wants every target.
    cargo clippy --all-targets --features backend-tests -- -D warnings

    # --no-fail-fast: cargo stops at the first failing *binary* otherwise, so a
    # single early break hides every later crate's result. A release gate wants
    # the whole picture in one run; the non-zero exit still aborts the publish.
    #
    # --features backend-tests: the default test selection is a fast smoke set;
    # a release gate wants everything, so ask for the gated test targets back.
    bold "[3/3] cargo test --release --no-fail-fast --features backend-tests"
    cargo test --release --no-fail-fast --features backend-tests
    green "Pre-flight gates passed."
fi

# ── Confirmation ────────────────────────────────────────────────
list_tiers 1
echo
print_publish_plan
echo
if (( DRY_RUN )); then
    yellow "Mode: DRY RUN — no actual uploads, no rate-limit sleeps, no index polling."
else
    yellow "Mode: REAL PUBLISH — uploads to crates.io."
    yellow "Rate-limit floor:      ${MIN_INTERVAL}s between publishes."
    yellow "Between-tier extra:    ${BETWEEN_DELAY}s after each tier."
    if (( NO_POLL )); then
        yellow "Index polling:         DISABLED (--no-poll) — relying on fixed sleeps only."
    else
        yellow "Index polling:         every ${POLL_INTERVAL}s, hard cap ${POLL_TIMEOUT}s per crate."
    fi
    if (( MAX_RETRIES == 0 )); then
        yellow "Registry retries:        auto-wait + retry until success (unlimited)."
    else
        yellow "Registry retries:        up to ${MAX_RETRIES} backoff attempts per crate."
    fi
    yellow "Retry-after parsing:     HTTP-date + status-specific backoff (+${RATE_LIMIT_PAD}s pad)."
    if (( SKIP_PUBLISHED )); then
        yellow "Already on crates.io:    skip when local version matches index."
    else
        yellow "Already on crates.io:    still attempt upload (--no-skip-published)."
    fi
    if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
        # cargo will use ~/.cargo/credentials if no env var.
        yellow "CARGO_REGISTRY_TOKEN not set — relying on \`cargo login\` credentials."
    fi
fi
echo

if (( ! ASSUME_YES )); then
    # A closed stdin is not "no terminal": a piped answer and a run driven over a
    # remote shell both work. What cannot be answered is a stdin with nothing
    # left to read — `read` fails there, and under `set -e` that used to end the
    # run silently, leaving the operator believing something else happened.
    if ! read -p "Continue? [y/N] " -n 1 -r; then
        red "stdin is closed: nothing can answer the confirmation. Pass --yes to publish without confirming." >&2
        exit 1
    fi
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        red "Aborted."
        exit 1
    fi
fi

# ── Sparse-index URL builder ────────────────────────────────────
#
# crates.io's sparse-index partitioning mirrors crates.io-index's
# git-side layout — encoded in the URL prefix so each shard stays
# small:
#
#   1 char:   /1/<crate>
#   2 chars:  /2/<crate>
#   3 chars:  /3/<first_char>/<crate>
#   4+ chars: /<first_2>/<next_2>/<crate>
#
# Names are always lower-case and hyphens are kept verbatim. See
# https://doc.rust-lang.org/cargo/reference/registry-index.html
# (sparse_index_path, check_index are defined above for per-crate skip/plan
# checks.)

# Block until check_index says we're good or the timeout trips.
wait_for_index() {
    local crate="$1"
    local version="$2"
    if (( NO_POLL )); then
        yellow "  (--no-poll set; skipping index readiness check)"
        return 0
    fi
    local elapsed=0
    yellow "  polling https://index.crates.io/.../$crate for $version (every ${POLL_INTERVAL}s, cap ${POLL_TIMEOUT}s)..."
    while (( elapsed < POLL_TIMEOUT )); do
        if check_index "$crate" "$version"; then
            green "  index ready after ${elapsed}s — $crate@$version resolvable."
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$(( elapsed + POLL_INTERVAL ))
    done
    yellow "  index didn't report $crate@$version within ${POLL_TIMEOUT}s — continuing anyway,"
    yellow "    next crate's publish may fail dep resolution if it depends on this one."
    return 0
}

# True when a path dep was rewritten for publish but is not on crates.io yet.
is_missing_registry_dep_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qE 'no matching package named|location searched: crates\.io index' "$err_file"
}

missing_registry_dep_name() {
    local err_file="$1"
    grep -oE 'no matching package named `[^`]+`' "$err_file" 2>/dev/null \
        | head -1 \
        | sed -E 's/no matching package named `([^`]+)`/\1/' \
        || true
}

# True when cargo says this version was uploaded already.
is_already_exists_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qiE 'already exists on crates\.io|already exists on the registry' "$err_file"
}

# The HTTP status cargo reported. A registry error prints it parenthesised —
# `(status 429 Too Many Requests)` or, in releases that format the
# `http::StatusCode` directly, `(429 Too Many Requests)` — and the non-JSON
# response path prints `got 429`. The right boundary admits end of line: the
# shortest form ends there.
registry_http_status_from_log() {
    local err_file="$1" status
    [[ -f "$err_file" ]] || return 0
    status="$(grep -oiE '\((status )?[0-9]{3}([^0-9]|$)' "$err_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]{3}' || true)"
    if [[ -z "$status" ]]; then
        status="$(grep -oiE 'got [0-9]{3}([^0-9]|$)' "$err_file" 2>/dev/null \
            | tail -1 | grep -oE '[0-9]{3}' || true)"
    fi
    printf '%s\n' "$status"
}

# Transient registry / network failures — retry after a computed wait.
is_retryable_registry_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    local status
    status="$(registry_http_status_from_log "$err_file")"
    case "$status" in
        408|429|500|502|503|504) return 0 ;;
    esac
    grep -qiE \
        'Too Many Requests|rate.?limit|published too many new crates|try again after|temporarily unavailable|service unavailable|bad gateway|gateway timeout|timed out|connection reset|connection refused|unexpected eof|broken pipe|error sending request|operation timed out|dns error' \
        "$err_file"
}

# Convert crates.io HTTP-date ("Wed, 27 May 2026 11:09:09 GMT") → epoch.
http_date_to_epoch() {
    local when="$1"
    if [[ "$(uname -s)" == Darwin ]]; then
        date -j -u -f '%a, %d %b %Y %H:%M:%S GMT' "$when" '+%s' 2>/dev/null
    else
        date -u -d "$when" '+%s' 2>/dev/null
    fi
}

# Default backoff when no explicit retry window is in the log.
registry_status_default_wait() {
    local status="$1"
    case "$status" in
        429) echo $(( MIN_INTERVAL * 2 )) ;;
        408) echo 60 ;;
        500|502|503|504) echo 90 ;;
        *) echo "$MIN_INTERVAL" ;;
    esac
}

# How long to wait before retrying a registry error, the HTTP status cargo
# reported (0 when it named none), and the server's own window as printed
# (empty when it named none).
# Prints: wait_seconds http_status server_window
registry_retry_wait_seconds() {
    local err_file="$1"
    local when epoch now wait status
    status="$(registry_http_status_from_log "$err_file")"
    wait="$(registry_status_default_wait "$status")"

    when="$(
        grep -oiE '(please )?try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' \
            "$err_file" 2>/dev/null \
            | tail -1 \
            | sed -E 's/^[Pp]lease [Tt]ry again after //; s/^[Tt]ry again after //' \
            || true
    )"
    if [[ -n "$when" ]]; then
        epoch="$(http_date_to_epoch "$when" || true)"
        if [[ -n "${epoch:-}" ]]; then
            now="$(date -u '+%s')"
            wait=$(( epoch - now + RATE_LIMIT_PAD ))
            if (( wait < MIN_INTERVAL )); then
                wait=$MIN_INTERVAL
            fi
            if (( wait > 86400 )); then
                wait=86400
            fi
        else
            # A real window this `date` cannot parse: fall back to the doubled
            # settle floor and still report it.
            wait=$(( MIN_INTERVAL * 2 ))
        fi
        echo "$wait ${status:-0} $when"
        return 0
    fi

    local retry_after
    retry_after="$(
        grep -oiE 'retry-after: *[0-9]+' "$err_file" 2>/dev/null \
            | tail -1 \
            | grep -oE '[0-9]+' \
            || true
    )"
    if [[ -n "$retry_after" ]]; then
        echo "$(( retry_after + RATE_LIMIT_PAD )) ${status:-0}"
        return 0
    fi

    echo "$wait ${status:-0}"
}

# Sleep with a single live countdown line — easier to watch than
# silent waits during long publishes.
sleep_with_countdown() {
    local seconds="$1"
    local label="$2"
    local remaining=$seconds
    while (( remaining > 0 )); do
        printf "\r  %s — %3ds remaining " "$label" "$remaining"
        sleep 1
        remaining=$(( remaining - 1 ))
    done
    printf "\r  %s — done.                  \n" "$label"
}

# ── Walk tiers ──────────────────────────────────────────────────
DRY_RUN_PASS=()
DRY_RUN_FAIL=()
ALREADY_ON_CRATES_IO=()
PUBLISHED_THIS_RUN=()
NEW_CRATES_UPLOADED=0
NEW_VERSIONS_UPLOADED=0
# A resumed run continues a process whose uploads spent a bucket this one never
# saw, so it starts as if each burst were used: the first upload waits one
# refill instead of firing one the registry is likely to refuse.
if (( START_TIER > 0 )) || [[ -n "$START_CRATE" ]]; then
    NEW_CRATES_UPLOADED=$NEW_CRATE_BURST
    NEW_VERSIONS_UPLOADED=$NEW_VERSION_BURST
fi

# 0 = this version is on the index, 1 = the name is but this version is not,
# 2 = the name is not on the index at all. An unreadable index is reported as
# 2: with the bucket unknown, a first-time name (ten minutes a refill) is the
# reading that cannot be too fast.
crate_index_state() {
    local state=0
    check_index "$1" "$2" || state=$?
    if (( state == 3 )); then
        yellow "  index unreachable for $1 — pacing as a new crate name" >&2
        state=2
    fi
    echo "$state"
}

# Wait out the bucket the next upload draws from. crates.io refills a brand-new
# name every ten minutes after its burst of five, and a new version of an
# existing name every minute after its burst of thirty; inside a burst the only
# wait is the settle floor.
pace_before_upload() {
    local state="$1" bucket count burst refill
    if (( state == 2 )); then
        bucket="new crate"
        count=$NEW_CRATES_UPLOADED
        burst=$NEW_CRATE_BURST
        refill=$NEW_CRATE_REFILL
    else
        bucket="new version"
        count=$NEW_VERSIONS_UPLOADED
        burst=$NEW_VERSION_BURST
        refill=$NEW_VERSION_REFILL
    fi
    if (( count >= burst )); then
        sleep_with_countdown "$refill" "$bucket bucket spent — registry refill"
    elif (( count > 0 )); then
        sleep_with_countdown "$MIN_INTERVAL" "settle between uploads"
    fi
}

publish_one_attempt() {
    local crate="$1"
    local args=()
    args+=("--package" "$crate")
    if (( DRY_RUN )); then
        args+=("--dry-run" "--allow-dirty")
    fi
    if (( NO_VERIFY )); then
        args+=("--no-verify")
    fi
    bold ">> cargo publish ${args[*]}"
    local tmp_err
    tmp_err="$(mktemp)"
    LAST_PUBLISH_ERR=""
    # Do not trust cargo's exit code alone — crates.io 429 often returns 0.
    set +e
    cargo publish "${args[@]}" 2>&1 | tee "$tmp_err"
    local rc=${PIPESTATUS[0]}
    set -e

    if is_already_exists_error "$tmp_err"; then
        rm -f "$tmp_err"
        return 43
    fi

    if is_retryable_registry_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 42
    fi

    if is_missing_registry_dep_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 44
    fi

    if (( rc == 0 )) && ! grep -qE 'error(\[[0-9]+\])?: failed|error(\[[0-9]+\])?: failed to publish' "$tmp_err"; then
        rm -f "$tmp_err"
        return 0
    fi

    LAST_PUBLISH_ERR="$tmp_err"
    return 1
}

# 0 = published (or dry-run ok), 1 = hard failure, 2 = already on crates.io
publish_one() {
    local crate="$1"
    local bucket_refill="$2"
    local version
    version="$(crate_version "$crate")"
    local attempt=1
    local missing_dep_attempts=0
    while true; do
        # `|| rc=$?` keeps `set -e` from aborting on non-zero returns (42/44).
        local rc=0
        publish_one_attempt "$crate" || rc=$?
        if (( rc == 0 )); then
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            if (( DRY_RUN )); then
                DRY_RUN_PASS+=("$crate")
            fi
            return 0
        fi
        if (( rc == 43 )); then
            green "  $crate@$version already on crates.io (cargo confirmed) — skip."
            return 2
        fi
        if (( DRY_RUN )); then
            DRY_RUN_FAIL+=("$crate")
            yellow "  (dry-run: dep resolution failed — expected for non-leaf crates pre-publish)"
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            return 0
        fi
        if (( rc == 44 )); then
            local missing=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                missing="$(missing_registry_dep_name "$LAST_PUBLISH_ERR")"
            fi
            if [[ -n "$missing" ]]; then
                local missing_ver
                missing_ver="$(crate_version "$missing")"
                missing_dep_attempts=$(( missing_dep_attempts + 1 ))
                yellow "  $crate: waiting for $missing@$missing_ver on sparse index (attempt $missing_dep_attempts), then retry."
                wait_for_index "$missing" "$missing_ver"
                rm -f "${LAST_PUBLISH_ERR:-}"
                LAST_PUBLISH_ERR=""
                sleep_with_countdown "$POLL_INTERVAL" "index settle after $missing"
                continue
            fi
            red "Publish blocked for $crate: dependency not on crates.io yet."
            red "  Check scripts/publish.sh tier order."
            exit 1
        fi
        if (( rc == 42 )); then
            # rc 42 is only returned for an error `is_retryable_registry_error`
            # recognized, so a wait is always available here; the bucket refill
            # is the conservative seed if the log went missing.
            local wait_sec=$bucket_refill http_status=0 when_human=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                read -r wait_sec http_status when_human \
                    < <(registry_retry_wait_seconds "$LAST_PUBLISH_ERR")
                rm -f "$LAST_PUBLISH_ERR"
                LAST_PUBLISH_ERR=""
            fi
            # A 429 means the bucket is empty: an upload before its refill is a
            # certain refusal, whatever window the server named — none, or one
            # already elapsed and clamped to the settle floor.
            if (( http_status == 429 )) && (( wait_sec < bucket_refill )); then
                wait_sec=$bucket_refill
            fi
            if (( MAX_RETRIES > 0 && attempt > MAX_RETRIES )); then
                red "Publish failed for $crate after $MAX_RETRIES registry retries."
                red "Re-run with --start-crate $crate to resume later."
                exit 1
            fi
            if (( http_status > 0 )); then
                yellow "  crates.io HTTP $http_status for $crate — waiting ${wait_sec}s then retry (automatic)."
            else
                yellow "  registry transient error for $crate — waiting ${wait_sec}s then retry (automatic)."
            fi
            if [[ -n "$when_human" ]]; then
                yellow "    server window ends: $when_human (+${RATE_LIMIT_PAD}s pad)"
            fi
            attempt=$(( attempt + 1 ))
            sleep_with_countdown "$wait_sec" "crates.io registry cooldown"
            continue
        fi
        if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
            red "  last cargo log: $LAST_PUBLISH_ERR"
        fi
        red "Publish failed for $crate (non-retryable, exit $rc)."
        red "Re-run with --start-crate $crate (or a later one) to resume."
        exit 1
    done
}

# Resolve start position.
skip_until_tier=$START_TIER
skip_until_crate="$START_CRATE"

for tier_idx in "${!TIERS[@]}"; do
    if (( tier_idx < skip_until_tier )); then
        continue
    fi
    tier="${TIERS[$tier_idx]}"
    bold "── Tier $tier_idx ────────────────────────────────────────"

    published_in_tier=0
    for crate in $tier; do
        version="$(crate_version "$crate")"

        # Skip past --start-crate if specified.
        if [[ -n "$skip_until_crate" ]]; then
            if [[ "$crate" != "$skip_until_crate" ]]; then
                yellow "  skip $crate (resume target: $skip_until_crate)"
                continue
            fi
            skip_until_crate=""
        fi

        # One index reading answers both questions: skip when this version is
        # already there, and which bucket an upload would draw from.
        index_state=-1
        if (( ! DRY_RUN )); then
            index_state="$(crate_index_state "$crate" "$version")"
            if (( SKIP_PUBLISHED )) && (( index_state == 0 )); then
                green "  skip $crate@$version (already on crates.io)"
                ALREADY_ON_CRATES_IO+=("$crate@$version")
                continue
            fi
            pace_before_upload "$index_state"
        fi

        bucket_refill=$NEW_VERSION_REFILL
        if (( index_state == 2 )); then bucket_refill=$NEW_CRATE_REFILL; fi
        pub_rc=0
        publish_one "$crate" "$bucket_refill" || pub_rc=$?
        if (( pub_rc == 2 )); then
            ALREADY_ON_CRATES_IO+=("$crate@$version")
            continue
        fi
        if (( pub_rc != 0 )); then
            exit 1
        fi

        PUBLISHED_THIS_RUN+=("$crate@$version")
        if (( index_state == 2 )); then
            NEW_CRATES_UPLOADED=$(( NEW_CRATES_UPLOADED + 1 ))
        else
            NEW_VERSIONS_UPLOADED=$(( NEW_VERSIONS_UPLOADED + 1 ))
        fi
        published_in_tier=$(( published_in_tier + 1 ))

        # Post-publish: poll the index until the new version is
        # actually queryable, so the *next* crate's dep resolution
        # doesn't 404. Dry runs skip this entirely.
        if (( ! DRY_RUN )); then
            wait_for_index "$crate" "$version"
        fi
    done

    # Between-tier extra delay only when this tier uploaded something.
    if (( ! DRY_RUN )) && (( tier_idx + 1 < ${#TIERS[@]} )) && (( published_in_tier > 0 )); then
        sleep_with_countdown "$BETWEEN_DELAY" "between-tier (let crates.io index settle)"
    fi
done

if [[ -n "$skip_until_crate" ]]; then
    if [[ " ${TIERS[*]} " == *" $skip_until_crate "* ]]; then
        red "--start-crate $skip_until_crate sits below --start-tier $START_TIER — nothing was published."
    else
        red "--start-crate $skip_until_crate is not in TIERS — nothing was published."
    fi
    exit 1
fi

if (( DRY_RUN )); then
    echo
    bold "Dry-run summary:"
    green "  passed (full dry-run incl. dep resolution): ${#DRY_RUN_PASS[@]}"
    if (( ${#DRY_RUN_PASS[@]} > 0 )); then
        for c in "${DRY_RUN_PASS[@]}"; do echo "    ✓ $c"; done
    fi
    if (( ${#DRY_RUN_FAIL[@]} > 0 )); then
        yellow "  metadata + packaging ok, dep resolution fails (expected pre-publish): ${#DRY_RUN_FAIL[@]}"
        for c in "${DRY_RUN_FAIL[@]}"; do echo "    ⚠ $c"; done
        echo
        yellow "Non-leaf crates can only fully dry-run after their leaves are"
        yellow "actually published — that's why they show ⚠ above. Real publish"
        yellow "in tier order will succeed."
    fi
else
    echo
    if (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Published this run (${#PUBLISHED_THIS_RUN[@]}):"
        for c in "${PUBLISHED_THIS_RUN[@]}"; do echo "    ✓ $c"; done
    fi
    if (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        yellow "Already on crates.io — skipped (${#ALREADY_ON_CRATES_IO[@]}):"
        for c in "${ALREADY_ON_CRATES_IO[@]}"; do echo "    ○ $c"; done
    fi
    if (( ${#PUBLISHED_THIS_RUN[@]} == 0 )) && (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        green "Nothing left to publish — every listed crate is already at its local version on crates.io."
    elif (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Publish run finished."
    else
        green "All tiers processed."
    fi
fi
