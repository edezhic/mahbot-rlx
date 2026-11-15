# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.75s | 973.1ms | 201.6MiB | 1.79GiB | 196.5MiB | 1.86GiB |
| recognition 30s | 3.36s | 2.61s | 213.8MiB | 1.84GiB | 253.7MiB | 1.92GiB |
| recognition 318s | 29.11s | 28.23s | 272.8MiB | 1.90GiB | 273.5MiB | 1.93GiB |
| synthesis 25 chars | 1.52s | 993.6ms | 407.9MiB | 1.25GiB | 468.3MiB | 1.46GiB |
| synthesis 129 chars | 4.18s | 3.61s | 418.6MiB | 1.50GiB | 448.2MiB | 1.64GiB |
| wake background 1s | 210.2ms | 265µs | 130.0MiB | 465.1MiB | 135.1MiB | 521.3MiB |
| wake active 1s | 300.3ms | 75.5ms | 138.1MiB | 476.3MiB | 135.9MiB | 492.5MiB |

_Measured 2026-11-15 13:25 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 2.72 · 3.46 · 4.27 (1m · 5m · 15m)
- code: c3cc266e tree-sha256 8d8d13de82bab8a9 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
