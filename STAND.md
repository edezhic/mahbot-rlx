# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 2.08s | 976.2ms | 203.9MiB | 1.78GiB | 196.5MiB | 1.91GiB |
| recognition 30s | 3.24s | 2.69s | 218.6MiB | 1.64GiB | 218.6MiB | 1.90GiB |
| recognition 318s | 28.26s | 27.67s | 261.6MiB | 1.67GiB | 263.3MiB | 2.04GiB |
| synthesis 25 chars | 1.51s | 1.01s | 407.7MiB | 1.25GiB | 428.3MiB | 1.50GiB |
| synthesis 129 chars | 4.16s | 3.65s | 420.3MiB | 1.50GiB | 420.3MiB | 1.68GiB |
| wake background 1s | 165.7ms | 252µs | 130.6MiB | 468.4MiB | 133.9MiB | 507.1MiB |
| wake active 1s | 260.6ms | 82.5ms | 138.4MiB | 476.3MiB | 133.3MiB | 509.3MiB |

_Measured 2026-11-20 18:36 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.78 · 4.06 · 4.44 (1m · 5m · 15m)
- code: 40e9569f tree-sha256 b770d6ea79df51bb (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
