# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.84s | 983.0ms | 264.8MiB | 1.79GiB | 257.0MiB | 1.84GiB |
| recognition 30s | 3.41s | 2.61s | 277.1MiB | 1.84GiB | 294.7MiB | 1.92GiB |
| recognition 318s | 29.20s | 28.21s | 333.2MiB | 1.90GiB | 338.1MiB | 1.93GiB |
| synthesis 25 chars | 1.53s | 1.04s | 440.8MiB | 1.25GiB | 450.6MiB | 1.51GiB |
| synthesis 129 chars | 4.26s | 3.61s | 452.9MiB | 1.50GiB | 492.3MiB | 1.59GiB |
| wake background 1s | 229.0ms | 267µs | 154.1MiB | 465.1MiB | 154.6MiB | 500.7MiB |
| wake active 1s | 318.0ms | 74.5ms | 162.3MiB | 476.3MiB | 156.5MiB | 496.7MiB |

_Measured 2026-11-15 08:40 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 4.18 · 5.96 · 6.15 (1m · 5m · 15m)
- code: c7e5acb1 tree-sha256 821bb4507ab3db87 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
