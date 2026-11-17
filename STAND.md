# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.87s | 995.9ms | 203.6MiB | 1.78GiB | 192.9MiB | 1.89GiB |
| recognition 30s | 3.14s | 2.54s | 215.6MiB | 1.64GiB | 219.0MiB | 1.89GiB |
| recognition 318s | 28.11s | 28.08s | 262.2MiB | 1.67GiB | 267.9MiB | 2.03GiB |
| synthesis 25 chars | 1.68s | 1.02s | 408.1MiB | 1.25GiB | 415.9MiB | 1.49GiB |
| synthesis 129 chars | 4.17s | 3.67s | 420.2MiB | 1.50GiB | 429.7MiB | 1.60GiB |
| wake background 1s | 174.1ms | 269µs | 130.0MiB | 465.2MiB | 130.4MiB | 514.9MiB |
| wake active 1s | 262.9ms | 78.8ms | 137.9MiB | 476.3MiB | 135.2MiB | 505.6MiB |

_Measured 2026-11-17 05:15 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 5.02 · 5.96 · 5.88 (1m · 5m · 15m)
- code: 1db7ee83 tree-sha256 dafff8969c43f147 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
