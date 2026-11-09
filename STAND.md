# Stand

| row | cold | warm | resting system | resting GPU | peak system | peak GPU |
|---|---|---|---|---|---|---|
| recognition, short voice message (10 s) | 2.04s | 1.04s | 608.1MiB | 1.79GiB | 621.9MiB | 2.08GiB |
| recognition, typical voice message (30 s) | 3.61s | 2.77s | 392.8MiB | 1.84GiB | 622.3MiB | 2.01GiB |
| recognition, long voice message (the whole pinned recording) | 30.67s | 29.80s | 431.8MiB | 1.90GiB | 650.6MiB | 1.98GiB |
| synthesis, short text | 2.16s | 1.07s | 732.4MiB | 1.25GiB | 735.6MiB | 1.64GiB |
| synthesis, typical text | 4.53s | 3.75s | 735.0MiB | 1.50GiB | 762.7MiB | 1.99GiB |
| wake word, two scoring windows in one pass | 443.5ms | 24.2ms | 84.6MiB | 476.3MiB | 161.2MiB | 707.4MiB |

Each row is a product shape measured in a process of its own. Cold is the process with no models loaded and the call that loads them; warm is the same call with the models already resident. Memory is the process's physical footprint as the platform accounts for it (`phys_footprint`), the only reading that sees memory allocated for the GPU; it counts dirty pages, page tables and IOKit mappings, and clean file-backed pages — a read-only model mapping, say — are outside it. On a footprint row, the resting figure is a pair read from one `task_info(TASK_VM_INFO)` call, so the two halves sum to the footprint: system is the footprint minus the pages the platform tags as graphics, which also holds IOKit mappings, compressed pages, page tables and any pages tagged for something other than graphics, and is not a CPU-only reading; GPU is those graphics-tagged pages, which include buffers the host can also map and staging allocations, and is not the GPU's working set. Graphics-tagged pages the platform books outside the footprint (`ledger_tag_graphics_nofootprint`) are in neither number, and GPU memory the platform books as an IOKit mapping without a graphics tag cannot be separated from the system figure. Resting is taken once the cold call has returned and before the warm call; peak is that figure's lifetime maximum. On a footprint row, the GPU peak beside it is the highest graphics-tagged reading a sampler thread took every millisecond or so from the first moment the row's process could start it until that maximum was read — an observed maximum, so a spike before then or between two samples is not in it, and the two halves were not taken at the same instant. The system figure at peak is derived, not measured: it is the lifetime maximum of the whole figure minus the highest GPU reading, a subtraction of two maxima, and is not a peak of anything on its own. A system figure at peak below the one at rest is therefore legitimate — the GPU share can grow more than the total between rest and peak — and a GPU spike the sampler missed makes the derived figure too large, while a GPU reading taken at another instant can make it too small. Memory is unified on this machine, so where both halves are measured they divide one pool rather than two budgets; on a machine with separate video memory this reading would report the host-visible graphics share, not that machine's video memory.

_Measured 2026-11-09 13:33 UTC._
- machine: arm64 · Apple M3 Pro
- device: Apple M3 Pro (Metal)
- profile: release
- load: 3.51 · 5.64 · 5.68 (1, 5 and 15 minutes when the record was written)
- code: ec916ae4 change-sha256 664a09f94975d73d (the commit and the SHA-256 of the tree's difference from it, `STAND.md`, `TRANSCRIPTION-CORPUS.md`, `WAKE-WORD-BENCH.md`, `crates/backends/rlx-mlx-sys` left out)
- command: cargo run --release -p mahbot-inference
- models: `/Users/Shared/MahBot/models`
- corpus: `/Users/Shared/MahBot/models/corpus/real-audio`
