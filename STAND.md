# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 1.67s | 980.7ms | 261.6MiB | 1.79GiB | 251.2MiB | 1.96GiB |
| recognition, typical voice message (30 s) | 3.06s | 2.62s | 274.2MiB | 1.85GiB | 253.5MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 28.93s | 28.35s | 330.9MiB | 1.90GiB | 393.9MiB | 1.98GiB |
| synthesis, short text | 2.48s | 1.04s | 466.8MiB | 1.25GiB | 481.8MiB | 1.57GiB |
| synthesis, typical text | 4.09s | 3.72s | 477.1MiB | 1.50GiB | 480.8MiB | 1.88GiB |
| wake word, two scoring windows in one pass | 181.5ms | 24.5ms | 161.9MiB | 476.3MiB | 159.2MiB | 706.5MiB |

_Measured 2026-11-13 02:51 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 2.95 · 2.91 · 2.70 (1m · 5m · 15m)
- code: 5b1e4d1e tree-sha256 6d66c2a0207c4252 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
