# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.72s | 977.4ms | 262.1MiB | 1.79GiB | 252.6MiB | 1.95GiB |
| recognition 30s | 3.35s | 2.59s | 277.5MiB | 1.84GiB | 254.0MiB | 2.01GiB |
| recognition 318s | 29.15s | 28.37s | 333.0MiB | 1.90GiB | 352.8MiB | 2.04GiB |
| synthesis 25 chars | 1.60s | 1.05s | 440.3MiB | 1.25GiB | 451.7MiB | 1.66GiB |
| synthesis 129 chars | 4.26s | 3.71s | 450.5MiB | 1.50GiB | 463.5MiB | 1.98GiB |
| wake background 1s | 219.3ms | 256µs | 154.1MiB | 690.9MiB | 154.1MiB | 709.5MiB |
| wake active 1s | 307.6ms | 75.3ms | 162.1MiB | 476.3MiB | 156.3MiB | 709.3MiB |

_Measured 2026-11-15 05:49 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.85 · 3.89 · 4.97 (1m · 5m · 15m)
- code: 63d6a277 tree-sha256 67b734808ce11e1b (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
