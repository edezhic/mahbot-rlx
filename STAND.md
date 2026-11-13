# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 3.49s | 976.6ms | 263.8MiB | 1.79GiB | 252.7MiB | 1.95GiB |
| recognition, typical voice message (30 s) | 3.07s | 2.60s | 279.7MiB | 1.85GiB | 253.2MiB | 2.02GiB |
| recognition, long voice message (the whole pinned recording) | 28.56s | 27.98s | 334.9MiB | 1.90GiB | 384.3MiB | 1.98GiB |
| synthesis, short text | 2.85s | 965.8ms | 467.6MiB | 1.26GiB | 488.8MiB | 1.58GiB |
| synthesis, typical text | 4.11s | 3.66s | 478.3MiB | 1.50GiB | 481.8MiB | 1.96GiB |
| wake word, two scoring windows in one pass | 194.6ms | 24.4ms | 161.6MiB | 476.3MiB | 159.0MiB | 707.3MiB |

_Measured 2026-11-13 20:52 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.93 · 4.21 · 3.60 (1m · 5m · 15m)
- code: 8d3572cd tree-sha256 7bdcf36b23a00a54 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
