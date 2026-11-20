# Stand (processor)

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition 10s | 20.41s | 20.63s | 360.7MiB | not measured | 548.0MiB | not measured |
| recognition 30s | 66.93s | 62.11s | 259.3MiB | not measured | 905.5MiB | not measured |
| recognition 318s | 701.79s | 671.25s | 509.1MiB | not measured | 1.30GiB | not measured |
| synthesis 25 chars | 9.87s | 9.02s | 821.4MiB | not measured | 1.23GiB | not measured |
| synthesis 129 chars | 34.23s | 38.39s | 1.49GiB | not measured | 3.14GiB | not measured |
| wake background 1s | 432.3ms | 263µs | 120.8MiB | not measured | 120.8MiB | not measured |
| wake active 1s | 2.68s | 2.46s | 172.8MiB | not measured | 173.0MiB | not measured |

_Measured 2026-11-20 19:03 UTC._
- machine: arm64 · Apple M3 Pro
- device: cpu (MAHBOT_DEVICE=cpu)
- profile: release
- load: 10.83 · 9.53 · 8.52 (1m · 5m · 15m)
- code: 40e9569f tree-sha256 b770d6ea79df51bb (SHA-256 of the tree's content, not a difference from the commit: every file git tracks or lists as untracked and not ignored is read (ignored build output is not; a tracked file deleted from the worktree is marked absent); in Rust sources comments are stripped (literals respected), `#[cfg(test)]` and `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is collapsed; `*.md`, `tests`, `tests.rs`, `crates/backends/rlx-mlx-sys` and writer temporaries are left out; an unreadable file leaves the digest unavailable)
- command: MAHBOT_DEVICE=cpu cargo run --release -p mahbot-inference
- run form: RLX_DEQUANT_CACHE=0
- memory: footprint
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
