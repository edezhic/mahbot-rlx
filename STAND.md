# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.23s | 981.7ms | 612.5MiB | 1.79GiB | 460.2MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.15s | 2.81s | 389.2MiB | 1.85GiB | 407.5MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 29.22s | 27.89s | 446.1MiB | 1.90GiB | 456.6MiB | 1.99GiB |
| synthesis, short text | 2.67s | 1.02s | 730.0MiB | 1.25GiB | 738.3MiB | 1.68GiB |
| synthesis, typical text | 4.11s | 3.61s | 732.7MiB | 1.50GiB | 740.5MiB | 2.04GiB |
| wake word, two scoring windows in one pass | 186.1ms | 24.0ms | 74.0MiB | 476.3MiB | 158.1MiB | 713.0MiB |

_Measured 2026-11-11 13:05 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.39 · 3.77 · 4.26 (1m · 5m · 15m)
- code: 2e398abb tree-sha256 0f0eb05e8d1862ee (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
