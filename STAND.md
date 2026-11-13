# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 1.46s | 999.4ms | 612.5MiB | 1.79GiB | 369.1MiB | 2.08GiB |
| recognition, typical voice message (30 s) | 3.11s | 2.62s | 389.8MiB | 1.85GiB | 404.6MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 28.73s | 28.53s | 435.3MiB | 1.90GiB | 427.0MiB | 2.04GiB |
| synthesis, short text | 2.79s | 1.03s | 730.7MiB | 1.25GiB | 784.4MiB | 1.56GiB |
| synthesis, typical text | 4.21s | 3.52s | 733.6MiB | 1.50GiB | 769.1MiB | 1.75GiB |
| wake word, two scoring windows in one pass | 211.8ms | 24.1ms | 75.0MiB | 476.3MiB | 162.0MiB | 706.5MiB |

_Measured 2026-11-12 20:05 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.61 · 1.99 · 2.46 (1m · 5m · 15m)
- code: b2c1f877 tree-sha256 e14de7c93b696f01 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
