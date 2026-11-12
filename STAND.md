# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.53s | 1.00s | 612.2MiB | 1.78GiB | 460.4MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.22s | 2.67s | 390.8MiB | 1.84GiB | 405.7MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 29.09s | 28.55s | 432.9MiB | 1.90GiB | 448.4MiB | 1.99GiB |
| synthesis, short text | 1.65s | 1.05s | 733.5MiB | 1.25GiB | 785.3MiB | 1.66GiB |
| synthesis, typical text | 4.29s | 3.73s | 734.2MiB | 1.50GiB | 787.7MiB | 1.86GiB |
| wake word, two scoring windows in one pass | 190.6ms | 24.6ms | 74.8MiB | 476.3MiB | 161.7MiB | 710.0MiB |

_Measured 2026-11-12 15:59 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.37 · 1.85 · 2.58 (1m · 5m · 15m)
- code: 8c5c83e6 tree-sha256 dd549043d574759e (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
