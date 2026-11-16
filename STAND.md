# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.61s | 964.5ms | 201.4MiB | 1.79GiB | 193.6MiB | 1.86GiB |
| recognition 30s | 3.16s | 2.53s | 214.8MiB | 1.64GiB | 251.6MiB | 1.90GiB |
| recognition 318s | 28.31s | 27.51s | 276.5MiB | 1.67GiB | 276.6MiB | 2.04GiB |
| synthesis 25 chars | 1.41s | 972.6ms | 409.3MiB | 1.25GiB | 459.0MiB | 1.48GiB |
| synthesis 129 chars | 4.01s | 3.55s | 420.8MiB | 1.50GiB | 465.0MiB | 1.59GiB |
| wake background 1s | 165.5ms | 269µs | 129.9MiB | 467.4MiB | 130.9MiB | 509.2MiB |
| wake active 1s | 256.2ms | 80.1ms | 138.0MiB | 476.3MiB | 135.3MiB | 506.6MiB |

_Measured 2026-11-16 20:47 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 2.65 · 3.17 · 4.46 (1m · 5m · 15m)
- code: 2bb1d719 tree-sha256 33f623ba090740cc (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
