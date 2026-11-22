# Stand (processor)

| row | cold | warm | resting system | peak system |
|---|---|---|---|---|
| recognition 10s | 4.46s | 3.68s | 370.0MiB | 557.5MiB |
| recognition 30s | 12.56s | 10.90s | 530.2MiB | 917.7MiB |
| recognition 318s | 142.40s | 136.43s | 765.3MiB | 1.56GiB |
| synthesis 25 chars | 9.78s | 9.06s | 848.9MiB | 1.30GiB |
| synthesis 129 chars | 32.08s | 33.29s | 1.52GiB | 3.23GiB |
| wake background 1s | 136.4ms | 267µs | 121.0MiB | 121.0MiB |
| wake active 1s | 706.7ms | 241.9ms | 185.6MiB | 185.6MiB |

_Measured 2026-11-22 13:06 UTC._
- machine: arm64 · Apple M3 Pro
- device: cpu (MAHBOT_DEVICE=cpu)
- profile: release
- load: 9.57 · 9.15 · 9.00 (1m · 5m · 15m)
- code: 09d9783f tree-sha256 f53c04b56db03b40 (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: MAHBOT_DEVICE=cpu cargo run --release -p mahbot-inference
- run form: RLX_DEQUANT_CACHE=0
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
