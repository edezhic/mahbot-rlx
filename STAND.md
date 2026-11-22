# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 2.01s | 957.1ms | 203.9MiB | 1.78GiB | 193.5MiB | 1.83GiB |
| recognition 30s | 3.31s | 2.56s | 216.5MiB | 1.64GiB | 216.5MiB | 1.90GiB |
| recognition 318s | 30.61s | 28.73s | 263.8MiB | 1.67GiB | 268.6MiB | 2.04GiB |
| synthesis 25 chars | 1.67s | 1.06s | 410.1MiB | 1.25GiB | 418.7MiB | 1.64GiB |
| synthesis 129 chars | 4.26s | 4.05s | 420.8MiB | 1.50GiB | 409.1MiB | 1.65GiB |
| wake background 1s | 474.0ms | 258µs | 130.0MiB | 465.1MiB | 132.0MiB | 499.1MiB |
| wake active 1s | 260.4ms | 87.7ms | 138.2MiB | 476.3MiB | 132.9MiB | 508.9MiB |

_Measured 2026-11-22 14:38 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 4.70 · 6.40 · 10.69 (1m · 5m · 15m)
- code: d0d72e60 tree-sha256 5b5238d28d57c22d (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
