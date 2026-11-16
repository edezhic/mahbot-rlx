# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.52s | 958.4ms | 202.8MiB | 1.79GiB | 192.7MiB | 1.89GiB |
| recognition 30s | 3.17s | 2.55s | 216.1MiB | 1.64GiB | 216.3MiB | 1.89GiB |
| recognition 318s | 28.24s | 27.30s | 276.9MiB | 1.67GiB | 277.1MiB | 2.04GiB |
| synthesis 25 chars | 1.40s | 972.9ms | 408.5MiB | 1.25GiB | 415.0MiB | 1.52GiB |
| synthesis 129 chars | 3.96s | 3.56s | 419.5MiB | 1.50GiB | 448.4MiB | 1.66GiB |
| wake background 1s | 168.1ms | 280µs | 130.3MiB | 465.2MiB | 132.4MiB | 535.3MiB |
| wake active 1s | 263.4ms | 77.0ms | 138.0MiB | 476.3MiB | 135.6MiB | 506.6MiB |

_Measured 2026-11-16 16:38 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.49 · 4.75 · 6.52 (1m · 5m · 15m)
- code: 765666e5 tree-sha256 43d2141768cda8de (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
