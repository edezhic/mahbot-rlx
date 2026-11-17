# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.83s | 958.3ms | 203.8MiB | 1.79GiB | 193.9MiB | 1.91GiB |
| recognition 30s | 3.11s | 2.51s | 213.9MiB | 1.64GiB | 236.6MiB | 1.90GiB |
| recognition 318s | 29.91s | 26.90s | 262.5MiB | 1.67GiB | 262.9MiB | 2.04GiB |
| synthesis 25 chars | 1.46s | 978.6ms | 410.3MiB | 1.25GiB | 469.3MiB | 1.47GiB |
| synthesis 129 chars | 3.99s | 3.56s | 421.2MiB | 1.50GiB | 458.1MiB | 1.62GiB |
| wake background 1s | 170.8ms | 260µs | 130.3MiB | 465.2MiB | 133.8MiB | 525.0MiB |
| wake active 1s | 268.2ms | 74.1ms | 138.7MiB | 476.3MiB | 134.7MiB | 508.2MiB |

_Measured 2026-11-17 19:31 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 2.81 · 3.95 · 5.01 (1m · 5m · 15m)
- code: 89ffaa73 tree-sha256 f1df12fdca2ff192 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
