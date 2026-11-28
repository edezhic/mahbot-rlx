# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 1.55s | 924.8ms | 122.9MiB | 1.71GiB | 103.7MiB | 1.78GiB |
| recognition 30s | 3.53s | 2.87s | 152.6MiB | 1.63GiB | 152.9MiB | 1.79GiB |
| recognition 318s | 40.22s | 25.45s | 176.7MiB | 1.66GiB | 210.8MiB | 1.90GiB |
| synthesis 25 chars | 1.39s | 881.1ms | 332.0MiB | 1.25GiB | 338.0MiB | 1.49GiB |
| synthesis 129 chars | 3.62s | 3.11s | 331.7MiB | 1.50GiB | 334.5MiB | 1.65GiB |
| wake background 1s | 181.3ms | 242µs | 36.8MiB | 485.5MiB | 36.8MiB | 495.8MiB |
| wake active 1s | 278.8ms | 75.5ms | 45.7MiB | 459.4MiB | 43.2MiB | 489.7MiB |

_Measured 2026-11-28 05:18 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 5.11 · 4.15 · 3.19 (1m · 5m · 15m)
- code: d8660887 tree-sha256 c03e3a1fca4a8da2 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: cargo run --release -p mahbot-inference
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
