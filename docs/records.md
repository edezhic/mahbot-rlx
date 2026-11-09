# The records and what their figures mean

The three records are the files the `measurements` module writes; what each run
records, and when a run writes nothing, is stated there. This file defines the
columns and the figures they carry.

## The stand's columns

| column | figure |
|---|---|
| `row` | the product shape measured, in a process of its own |
| `cold` | the call from a process with no models loaded, including loading them |
| `warm` | the same call with the models already resident |
| `resting system` | the system share of the physical footprint after the cold call and before the warm call |
| `resting GPU` | the graphics-tagged share of the same footprint reading |
| `peak system` | the lifetime maximum of the whole footprint minus the highest GPU reading — a derived figure, not a peak of its own |
| `peak GPU` | the highest graphics-tagged reading the sampler observed over the process's lifetime |

The corpus record's first columns are `recording` in place of `row`, and `audio s`
(the recording's own length) and `s per audio s` (the warm call divided by that
length) in place of the two time columns; its memory columns are the same. The wake
record reports `peak resident`, the resident set's high-water mark for the whole
run.

## The physical-footprint basis (`- memory: footprint`)

Memory is the process's physical footprint as the platform accounts for it
(`phys_footprint`), the only reading that sees memory allocated for the GPU; it
counts dirty pages, page tables and IOKit mappings, and clean file-backed pages —
a read-only model mapping, say — are outside it.

On a footprint row, the resting figure is a pair read from one
`task_info(TASK_VM_INFO)` call, so the two halves sum to the footprint: system is
the footprint minus the pages the platform tags as graphics, which also holds
IOKit mappings, compressed pages, page tables and any pages tagged for something
other than graphics, and is not a CPU-only reading; GPU is those graphics-tagged
pages, which include buffers the host can also map and staging allocations, and is
not the GPU's working set.

Graphics-tagged pages the platform books outside the footprint
(`ledger_tag_graphics_nofootprint`) are in neither number, and GPU memory the
platform books as an IOKit mapping without a graphics tag cannot be separated from
the system figure.

Resting is taken once the cold call has returned and before the warm call; peak is
that figure's lifetime maximum. On a footprint row, the GPU peak beside it is the
highest graphics-tagged reading a sampler thread took every millisecond or so from
the first moment the row's process could start it until that maximum was read — an
observed maximum, so a spike before then or between two samples is not in it, and
the two halves were not taken at the same instant.

The system figure at peak is derived, not measured: it is the lifetime maximum of
the whole figure minus the highest GPU reading, a subtraction of two maxima, and
is not a peak of anything on its own. A system figure at peak below the one at
rest is therefore legitimate — the GPU share can grow more than the total between
rest and peak — and a GPU spike the sampler missed makes the derived figure too
large, while a GPU reading taken at another instant can make it too small.

Memory is unified on this machine, so where both halves are measured they divide
one pool rather than two budgets; on a machine with separate video memory this
reading would report the host-visible graphics share, not that machine's video
memory.

## The resident-set basis (`- memory: resident`)

A record names `resident` when its figures are the resident set. On a platform
that supplies no physical-footprint reading, resting is the set at the moment the
cold call has returned and before the warm call, and peak is the kernel's lifetime
maximum of the same set. The wake bench always measures the resident set —
`peak resident` is its `getrusage` high-water mark — so its record names `resident`
on every platform.

The resident set cannot see pages allocated for the GPU, so a resident row's GPU
cell is not measured — not zero, and not the system figure repeated — and there the
two do not split one pool.

The resident set counts a different thing from the footprint, so these figures are
not comparable with one taken where the platform supplies it.

## A table with both bases (`- memory: mixed`)

Memory is the process's physical footprint on the rows the platform supplied one
and the resident set on the rows it did not: the two count different things and are
not comparable with each other. A record whose rows disagree names `mixed`; each
row was then taken on the basis its own reading supplied, and the field is the only
statement the record makes about it.

## Provenance fields

`_Measured ..._` is the take's timestamp. `machine`, `device`, `profile` and
`command` state where and how it ran; `load` is the kernel's load average when the
record was written, the three figures being the 1, 5 and 15 minute readings
(`1m · 5m · 15m`), and two takes are comparable only at the same load; `code` is
the commit and the digest; `memory` is the basis above; `models` and `corpus` name
the material the rows read; and the wake record's `run` names its completeness
(`complete`).

## The digest

The rule the `code:` line's digest is taken under is printed beside it in the stand
and corpus records. The wake record's line describes the take that produced its
numbers, not the current tree: it carries `change-sha256`, the earlier rule's digest
of the tree's difference from that commit.

## License

MIT OR Apache-2.0.
