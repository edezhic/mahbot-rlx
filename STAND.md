# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.07s | 1.08s | 607.5MiB | 1.79GiB | 621.6MiB | 2.08GiB |
| recognition, typical voice message (30 s) | 3.65s | 2.84s | 391.7MiB | 1.84GiB | 621.1MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 31.47s | 30.73s | 440.1MiB | 1.90GiB | 591.5MiB | 2.03GiB |
| synthesis, short text | 2.18s | 1.12s | 732.7MiB | 1.25GiB | 776.8MiB | 1.58GiB |
| synthesis, typical text | 4.72s | 3.94s | 734.7MiB | 1.50GiB | 756.3MiB | 1.92GiB |
| wake word, two scoring windows in one pass | 501.1ms | 25.1ms | 85.0MiB | 476.3MiB | 159.4MiB | 712.8MiB |

_Measured 2026-11-09 22:27 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.33 · 1.70 · 2.29 (1m · 5m · 15m)
- code: 049d6362 tree-sha256 686a6330a6367e5c (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
