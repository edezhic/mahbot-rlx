# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 7.88s | 6.54s | 207.3MiB | 1.55GiB | 198.0MiB | 2.08GiB |
| recognition 30s | 20.64s | 16.96s | 218.6MiB | 1.57GiB | 218.7MiB | 1.83GiB |
| recognition 318s | 197.51s | 207.40s | 268.0MiB | 1.59GiB | 268.5MiB | 1.95GiB |
| synthesis 25 chars | 11.08s | 6.87s | 410.0MiB | 1.25GiB | 292.3MiB | 1.61GiB |
| synthesis 129 chars | 31.88s | 31.66s | 421.9MiB | 1.46GiB | 235.8MiB | 1.75GiB |
| wake background 1s | 702.0ms | 291µs | 131.5MiB | 604.4MiB | 132.1MiB | 714.4MiB |
| wake active 1s | 636.0ms | 313.9ms | 138.9MiB | 466.8MiB | 136.8MiB | 500.9MiB |

_Measured 2026-11-27 00:38 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 6.82 · 4.22 · 4.91 (1m · 5m · 15m)
- code: 0934693f tree-sha256 18395c6f863c6843 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
