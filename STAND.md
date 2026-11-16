# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.69s | 990.8ms | 201.1MiB | 1.79GiB | 191.9MiB | 1.86GiB |
| recognition 30s | 3.30s | 2.56s | 213.9MiB | 1.64GiB | 251.5MiB | 1.89GiB |
| recognition 318s | 28.69s | 27.68s | 275.1MiB | 1.67GiB | 276.5MiB | 2.04GiB |
| synthesis 25 chars | 1.52s | 995.2ms | 407.3MiB | 1.25GiB | 412.6MiB | 1.52GiB |
| synthesis 129 chars | 4.24s | 3.70s | 418.8MiB | 1.50GiB | 419.4MiB | 1.67GiB |
| wake background 1s | 207.9ms | 257µs | 130.0MiB | 468.3MiB | 132.6MiB | 494.4MiB |
| wake active 1s | 304.3ms | 74.9ms | 138.1MiB | 476.3MiB | 132.8MiB | 495.4MiB |

_Measured 2026-11-16 07:41 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 1.36 · 4.97 · 5.11 (1m · 5m · 15m)
- code: 513ce211 tree-sha256 c80e8d00d67e79a0 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
