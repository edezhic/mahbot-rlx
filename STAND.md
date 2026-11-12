# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.61s | 1.01s | 613.7MiB | 1.79GiB | 460.7MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.19s | 2.71s | 390.8MiB | 1.84GiB | 405.8MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 29.36s | 28.63s | 437.2MiB | 1.90GiB | 418.2MiB | 1.98GiB |
| synthesis, short text | 2.93s | 1.11s | 734.2MiB | 1.25GiB | 740.1MiB | 1.62GiB |
| synthesis, typical text | 4.33s | 3.82s | 735.0MiB | 1.50GiB | 791.9MiB | 1.90GiB |
| wake word, two scoring windows in one pass | 484.6ms | 25.0ms | 74.7MiB | 476.3MiB | 158.9MiB | 709.5MiB |

_Measured 2026-11-12 14:30 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.42 · 2.35 · 3.58 (1m · 5m · 15m)
- code: 53ddfd0a tree-sha256 bbfe067358b3fc09 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
