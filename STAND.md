# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 2.48s | 951.1ms | 120.1MiB | 1.73GiB | 99.7MiB | 1.80GiB |
| recognition 30s | 3.25s | 2.60s | 149.2MiB | 1.64GiB | 150.7MiB | 1.82GiB |
| recognition 318s | 27.73s | 27.92s | 219.8MiB | 1.67GiB | 223.6MiB | 1.95GiB |
| synthesis 25 chars | 1.60s | 912.1ms | 334.3MiB | 1.25GiB | 348.9MiB | 1.48GiB |
| synthesis 129 chars | 3.95s | 3.20s | 333.0MiB | 1.50GiB | 347.8MiB | 1.54GiB |
| wake background 1s | 199.2ms | 273µs | 37.1MiB | 455.8MiB | 37.2MiB | 501.8MiB |
| wake active 1s | 291.5ms | 76.1ms | 47.2MiB | 466.8MiB | 45.4MiB | 496.5MiB |

_Measured 2026-11-27 15:21 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 6.64 · 4.86 · 6.96 (1m · 5m · 15m)
- code: bde4f3ac tree-sha256 e812f814efed6063 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
