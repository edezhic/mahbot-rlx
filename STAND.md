# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 1.91s | 1.07s | 607.0MiB | 1.79GiB | 622.4MiB | 2.08GiB |
| recognition, typical voice message (30 s) | 3.32s | 2.84s | 392.5MiB | 1.84GiB | 622.2MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 31.20s | 30.78s | 440.3MiB | 1.90GiB | 650.8MiB | 1.98GiB |
| synthesis, short text | 1.98s | 1.10s | 733.8MiB | 1.25GiB | 742.0MiB | 1.64GiB |
| synthesis, typical text | 4.49s | 3.93s | 733.1MiB | 1.50GiB | 752.9MiB | 1.96GiB |
| wake word, two scoring windows in one pass | 189.9ms | 25.2ms | 83.6MiB | 476.3MiB | 159.3MiB | 710.2MiB |

_Measured 2026-11-10 02:22 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 2.17 · 2.89 · 5.66 (1m · 5m · 15m)
- code: 77c40a3a tree-sha256 71830117a124af8f (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
