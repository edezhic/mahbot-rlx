# Stand (processor)

| row | cold | warm | resting system | peak system |
|---|---|---|---|---|
| recognition 10s | 3.96s | 2.12s | 374.1MiB | 561.2MiB |
| recognition 30s | 6.47s | 6.41s | 526.4MiB | 916.0MiB |
| recognition 318s | 63.22s | 66.90s | 759.7MiB | 1.55GiB |
| synthesis 25 chars | 1.96s | 1.54s | 839.5MiB | 1.27GiB |
| synthesis 129 chars | 7.14s | 7.00s | 1.48GiB | 3.12GiB |
| wake background 1s | 208.1ms | 276µs | 120.6MiB | 120.6MiB |
| wake active 1s | 2.45s | 1.14s | 186.1MiB | 186.7MiB |

_Measured 2026-11-26 05:52 UTC._
- machine: arm64 · Apple M3 Pro
- device: cpu (MAHBOT_DEVICE=cpu)
- profile: release
- load: 5.98 · 6.11 · 5.98 (1m · 5m · 15m)
- code: 84e1e75c tree-sha256 f39df85f8fa98abc (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: MAHBOT_DEVICE=cpu cargo run --release -p mahbot-inference
- run form: RLX_DEQUANT_CACHE=0
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
