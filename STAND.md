# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.14s | 1.06s | 612.1MiB | 1.79GiB | 460.0MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.32s | 2.84s | 388.0MiB | 1.84GiB | 402.8MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 31.23s | 30.72s | 433.3MiB | 1.90GiB | 423.1MiB | 2.03GiB |
| synthesis, short text | 2.74s | 1.10s | 734.0MiB | 1.25GiB | 753.3MiB | 1.68GiB |
| synthesis, typical text | 4.44s | 3.94s | 733.2MiB | 1.50GiB | 736.8MiB | 1.99GiB |
| wake word, two scoring windows in one pass | 186.9ms | 25.1ms | 74.2MiB | 476.3MiB | 159.1MiB | 709.5MiB |

_Measured 2026-11-10 18:35 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.57 · 4.23 · 5.45 (1m · 5m · 15m)
- code: 00ec82f1 tree-sha256 b1f0475213f6ac9e (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
