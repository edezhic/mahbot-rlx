#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Gate the routine round's selection against the product and the build step.

`product_set` in the Justfile is the round's selection. Three lists, no
compilation, every query `--locked` so none can rewrite `Cargo.lock`:

  * every *default member* on `mahbot-inference`'s normal dependency closure is
    in `product_set`. The closure is the product's own resolution, not a
    manifest walk: `-e normal` drops dev-dependencies and build-dependencies, and
    an optional dependency behind a feature the product never enables is off the
    graph, so a crate reached only that way is not required. Turning a
    dependency's default features off keeps it on the graph.
  * the round's own normal closure holds no workspace crate the build step's
    does not. `product_features` exists to match the build step's resolution; a
    feature that pulled a crate into the round alone would be a new variant.
    Dev-dependency crates, which a test build compiles and the build step never
    does, are outside this walk.
  * every `product_features` entry that names a package (`pkg/feat`) names one in
    `product_set`, and the list is one comma-separated token with no spaces (the
    round's recipe word-splits it), so a departing crate's feature or a stray
    space fails here rather than in cargo when the round runs.

One-directional by design: a product dependency the round does not cover is a
failure; a round crate that has left the product path is not. Nothing here
decides what the round *should* run, so it is not a narrowing gate.

The closure also holds workspace members that are not default members; the round
never covered them and does not now, so they are printed as never covered
rather than failed.

It ends by printing the round's three commands at four jobs, so a caller that
runs the round from outside this workspace takes its command line from the same
list.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fail(msg):
    sys.stdout.flush()
    print(msg, file=sys.stderr)
    return 1


def justfile_list(name):
    """Read a `name := "..."` string variable out of the Justfile."""
    text = (ROOT / "Justfile").read_text()
    m = re.search(rf'^{re.escape(name)} := "(.*)"$', text, re.MULTILINE)
    if not m:
        raise SystemExit(f'Justfile has no `{name} := "..."` variable')
    return m.group(1)


def cargo(*args):
    r = subprocess.run(["cargo", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cargo {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def closure(*args):
    """The packages a `cargo tree -e normal` closure holds."""
    out = cargo("tree", "--locked", *args, "-e", "normal", "--prefix", "none",
                "--format", "{p}")
    return {ln.split(" v", 1)[0] for ln in out.splitlines() if ln.strip()}


def debug_commands():
    """The `restricted` arm of `test-debug-verifier`, at four jobs.

    Read out of the Justfile rather than copied, so the printed line cannot
    drift from the recipe the round actually runs.
    """
    text = (ROOT / "Justfile").read_text()
    body = re.search(r"^test-debug-verifier\b[^\n]*:\n((?:[ \t]+.*\n|\n)*)", text, re.MULTILINE)
    if not body:
        raise SystemExit("Justfile has no `test-debug-verifier` recipe")
    arm = re.search(r"^[ \t]+restricted\)\n(.*?);;", body.group(1), re.MULTILINE | re.DOTALL)
    features = re.search(r'FEATURES="([^"]+)"', body.group(1))
    if not arm or not features:
        raise SystemExit('`test-debug-verifier` has no `restricted` arm with `FEATURES="..."`')
    lines = [ln.strip() for ln in arm.group(1).splitlines()
             if ln.strip().startswith("cargo test")]
    if not lines:
        raise SystemExit("`test-debug-verifier`'s `restricted` arm runs no `cargo test`")
    out = [ln.replace('"$FEATURES"', features.group(1))
             .replace("cargo test", "cargo test -j 4", 1)
           for ln in lines]
    for ln in out:
        if ln.count("-j ") != 1 or re.search(r"[`$|&;<>()\\]", ln):
            raise SystemExit("`test-debug-verifier`'s `restricted` arm is no longer a "
                             f"literal command line, so it cannot be printed as one: {ln}")
    return out


def main():
    selection = justfile_list("product_set")
    features = justfile_list("product_features")
    if re.search(r"\s", features):
        return fail("FAIL: `product_features` has whitespace, and the round's recipe "
                    "word-splits it into separate arguments: write it as one "
                    "comma-separated list with no spaces")
    packages = re.findall(r"-p (\S+)", selection)
    if selection.strip() != " ".join(f"-p {p}" for p in packages):
        return fail("FAIL: `product_set` is not a sequence of `-p <package>` pairs: "
                    "an entry without `-p` would become a test-name filter at run "
                    "time and silently narrow the round")
    # A bare feature name is valid cargo (it resolves against the selected roots),
    # so only the `pkg/feat` form can name a package.
    feature_pkgs = {f.split("/", 1)[0] for f in features.split(",") if "/" in f}

    meta = json.loads(cargo("metadata", "--locked", "--no-deps", "--format-version", "1"))
    name_of = {p["id"]: p["name"] for p in meta["packages"]}
    members = {name_of[i] for i in meta["workspace_members"]}
    default_members = {name_of[i] for i in meta["workspace_default_members"]}

    unknown = sorted(set(packages) - members)
    if unknown:
        return fail("FAIL: `product_set` names packages that are not workspace members "
                    f"(cargo would reject the round): {' '.join(unknown)}")
    stray = sorted(feature_pkgs - set(packages))
    if stray:
        return fail("FAIL: `product_features` names packages outside `product_set` "
                    f"(cargo would reject the round): {' '.join(stray)}")

    product = closure("-p", "mahbot-inference")
    build = closure(*[a for n in sorted(default_members) for a in ("-p", n)])
    round_ = closure(*[a for n in packages for a in ("-p", n)],
                     *(["--features", features] if features else []))

    print("the round's selection (product_set):")
    print(f"  {len(packages)} packages: {' '.join(sorted(packages))}")
    print(f"  features: {features}")
    print()

    covered = product & members
    missing = sorted((covered & default_members) - set(packages))
    never = sorted(covered - default_members)
    extra = sorted((round_ - build) & members)

    print(f"product closure: {len(covered)} workspace crates, "
          f"{len(covered & default_members)} of them default members")
    print("never covered (on the closure, not a default member — the round does not "
          "select non-default members):")
    for n in never or ["(none)"]:
        print(f"  {n}")
    print()

    if missing:
        return fail("FAIL: default members on the product's closure are not in the "
                    "round's selection:\n  " + "\n  ".join(missing) +
                    "\n\nAdd them to `product_set` in the Justfile, or record in its "
                    "comment why the crate leaves the round.")
    if extra:
        return fail("FAIL: the round's normal graph holds workspace crates the build "
                    "step's does not, which is a new variant (a feature in "
                    "`product_features`, a selected crate's features, or a package "
                    "added to `product_set`):\n  " + "\n  ".join(extra))

    print("ok: every default member on the product's closure is in the round")
    print("ok: the round's normal graph adds no workspace crate the build step's "
          "does not hold")
    print()
    print("paste-ready, the round's three commands at four jobs "
          "(the recipes cap none):")
    # The release line is composed from the two variables rather than parsed out
    # of the `test` recipe, whose body branches per platform; the debug commands
    # below are parsed. Change this line whenever the recipe's flags change, and
    # re-paste all three wherever they are run: this script keeps them faithful to
    # the recipes, not to whatever the caller currently has.
    print(f"cargo test --release -j 4 --tests {selection}"
          + (f' --features "{features}"' if features else ""))
    print("  on Linux, `just test` prefixes a `cargo build --tests --keep-going` "
          "with the same arguments and appends `-- --test-threads=N`")
    for c in debug_commands():
        print(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
