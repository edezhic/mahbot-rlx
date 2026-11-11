# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.97s | 1.01s | 611.5MiB | 1.79GiB | 459.0MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.14s | 2.70s | 393.8MiB | 1.85GiB | 405.4MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 29.44s | 28.93s | 428.3MiB | 1.90GiB | 445.7MiB | 2.00GiB |
| synthesis, short text | 2.72s | 1.09s | 730.2MiB | 1.25GiB | 732.6MiB | 1.64GiB |
| synthesis, typical text | 4.30s | 3.82s | 732.8MiB | 1.50GiB | 748.0MiB | 2.06GiB |
| wake word, two scoring windows in one pass | 183.5ms | 24.6ms | 74.8MiB | 476.3MiB | 158.9MiB | 709.5MiB |

_Measured 2026-11-11 06:39 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.44 · 4.21 · 5.62 (1m · 5m · 15m)
- code: bda1d348 tree-sha256 5ae76e06195ce0de (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
