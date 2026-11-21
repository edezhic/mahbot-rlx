# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.72s | 937.9ms | 203.4MiB | 1.78GiB | 194.9MiB | 1.88GiB |
| recognition 30s | 3.33s | 2.58s | 218.2MiB | 1.64GiB | 219.7MiB | 1.89GiB |
| recognition 318s | 27.43s | 27.23s | 260.9MiB | 1.67GiB | 262.6MiB | 2.04GiB |
| synthesis 25 chars | 1.51s | 953.7ms | 407.9MiB | 1.25GiB | 448.8MiB | 1.47GiB |
| synthesis 129 chars | 4.04s | 3.51s | 419.1MiB | 1.50GiB | 419.7MiB | 1.60GiB |
| wake background 1s | 168.6ms | 273µs | 129.9MiB | 465.2MiB | 137.6MiB | 529.8MiB |
| wake active 1s | 260.1ms | 86.2ms | 138.3MiB | 476.3MiB | 132.3MiB | 510.3MiB |

_Measured 2026-11-21 10:25 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 4.28 · 9.47 · 9.70 (1m · 5m · 15m)
- code: 906bf77f tree-sha256 cc20cc44d2a4d8b9 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
