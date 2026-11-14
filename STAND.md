# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 1.80s | 979.0ms | 265.0MiB | 1.79GiB | 251.1MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.08s | 2.62s | 274.7MiB | 1.85GiB | 250.6MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 28.57s | 28.01s | 330.9MiB | 1.90GiB | 335.4MiB | 1.99GiB |
| synthesis, short text | 1.65s | 997.0ms | 439.4MiB | 1.25GiB | 452.1MiB | 1.57GiB |
| synthesis, typical text | 4.09s | 3.57s | 452.3MiB | 1.50GiB | 454.0MiB | 1.87GiB |
| wake word, two scoring windows in one pass | 192.0ms | 24.1ms | 161.1MiB | 476.3MiB | 158.5MiB | 706.3MiB |

_Measured 2026-11-14 14:07 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.38 · 3.04 · 3.03 (1m · 5m · 15m)
- code: 838754fc tree-sha256 bfd2d93a0c03b93a (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
