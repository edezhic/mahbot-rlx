# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 1.44s | 998.9ms | 611.9MiB | 1.79GiB | 369.1MiB | 2.08GiB |
| recognition, typical voice message (30 s) | 3.18s | 2.71s | 388.9MiB | 1.84GiB | 369.5MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 28.94s | 28.15s | 432.4MiB | 1.90GiB | 438.7MiB | 2.01GiB |
| synthesis, short text | 1.47s | 1.01s | 733.0MiB | 1.25GiB | 745.5MiB | 1.61GiB |
| synthesis, typical text | 4.13s | 3.63s | 735.0MiB | 1.50GiB | 768.7MiB | 2.04GiB |
| wake word, two scoring windows in one pass | 195.7ms | 24.3ms | 74.5MiB | 476.3MiB | 159.5MiB | 708.8MiB |

_Measured 2026-11-12 06:11 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.01 · 3.09 · 3.01 (1m · 5m · 15m)
- code: 5ed1ecbd tree-sha256 03c0002c31ef32ca (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
