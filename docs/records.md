# The records and what their figures mean

The four records are the files the `measurements` module writes; what each run
records, and when a run writes nothing, is stated there. This file defines the
columns and the figures they carry.

## The stand's columns

| column | figure |
|---|---|
| `row` | the product shape measured, in a process of its own |
| `cold` | the row's work on a machine whose page cache does not hold its input — the short-lived command-line path, which loads the models every time |
| `warm` | the same work with the models in the page cache and the engine already loaded — a request on an engine that is up |
| `resting system` | the system share of the physical footprint after the cold call and before the warm call |
| `resting GPU` | the graphics-tagged share of the same footprint reading |
| `peak system` | the lifetime maximum of the whole footprint minus the highest GPU reading — a derived figure, not a peak of its own |
| `peak GPU` | the highest graphics-tagged reading the sampler observed over the process's lifetime |

The corpus record's first columns are `recording` in place of `row`, and `audio s`
(the recording's own length) and `s per audio s` (the warm call divided by that
length) in place of the two time columns; its memory columns are the same. The wake
record reports `peak resident`, the resident set's high-water mark for the whole
run.

`STAND-CPU.md` is the same stand taken on the processor: the same seven rows, without
the two graphics columns — a processor run allocates nothing for the graphics device, so
those cells could only ever read `not measured`. Its `resting system` and `peak system`
are the footprint reading with no GPU half subtracted. The two records'
figures are device-relative, so a cell in one is not comparable with the same cell in
the other.

Its memory columns are the **anonymous** footprint, and do not include the model. The
processor reads its weights from files that are shared between the graphs of one model
(`rlx-cpu`'s weight pool), so those pages are the file's and clean, and the platform's
footprint figure — which is what these columns report on either device — does not charge
them to the process. A processor row's `resting` and `peak` therefore price what the row
allocated for itself, not what it read: a row can hold 1.3 GB of model and report a few
hundred MiB. The graphics record's columns include the models, because the graphics
backend's weights are device allocations. That is a second reason the two are not
comparable cell by cell.

Because they are anonymous only, they also move with what the kernel chooses to keep:
re-taking the same tree under a different load has moved a row's `resting system` by a
hundred MiB and more while the model stayed in the page cache. A cell is read against
the take's own `load:` line, not against the same cell in another take.

## The two states, and how they are established

`cold` and `warm` are properties of the machine, not of the process, so the stand
makes them instead of inheriting them: `src/cold.rs` owns the method, says why it
is the one that works here, how the state is established over the models root
rather than over a list of files, and what the warm arm is read against. The pages
the warm arm had to fetch are stated on the row's diagnostics line (`warm_reread`),
not in the record's table, and `stand` refuses a row whose arm fetched more than
half the pages the cold arm read: the product rebuilds the graphs it released, so
some read is part of what `warm` prices, but a column that re-read most of the
model off the disk is a cold measurement wearing a warm label. A platform without
the method is refused before the run starts rather than given a `cold` column that
means nothing.

The lazy engine load is in the cold column, because the cold arm is what opens the
engine and a daemon pays that once: `warm` is a request on an engine that is up,
which is what every request after the first one costs.

A record holds one take. A take is dropped only when something makes it *not a
measurement*, never because of what its cells say: the binary was not built from the
tree the `code:` line names; a row ran on a device other than the one the record's
device line names (one arrival — how the run reached that device, settled once — for
the whole run); a row was stopped by a bound; or a warm arm re-read the pages (the
rule above). `load:` is not one of those: it is recorded as measured and read as
context, nothing more. A take on a busy machine that meets none of those conditions *is*
the record, read by the cell rule above like any other. The take in `STAND.md` is of
that kind: its `load:` line states the load it was taken at.

A processor take is the same rule from the other side: the take *is* the load
there, and a long row's two arms cannot be separated from it. A processor row
whose warm figure comes out slower than its cold one is therefore not by itself
grounds to withhold the record — what settles it is the page count the row prints.
If the row's `warm_reread` is not above that half, the row is warm and the difference
between the arms is contention, not a cold measurement wearing a warm label.

## The two wake rows

Both rows stream one second of audio through the product's own detection stream
(`bench::detect`), on one listener brought up the way a daemon brings it up —
weights mapped, the scoring front end compiled — so neither row measures a
rebuild.

| row | audio | what it prices |
|---|---|---|
| `wake background 1s` | one second with no speech in it | an idle daemon: the voice-activity decision per hop and the ring's housekeeping, and nothing else |
| `wake active 1s` | one second of speech that does not fire | a daemon while somebody is talking: scoring at the product's cadence, the windows and paired encoder passes `stand::STREAM_SECONDS` derives from the stride |

`warm` is therefore the cost of one second of streamed audio, which is what a
real-time budget is read against — 1 000 ms of audio per second is the ceiling.
`cold` is that plus the listener's start-up from a cold cache: the pair read and
the front end compiled for the first time, the same work in both rows, and on the
active row the first pass's uploads as well. A fire is cheaper than the active
row, because it starts a cooldown that skips the gate and the scoring; the active
row measures the common case, the whole time somebody is talking.

These two rows are the table's cheapest, where a stall landing in one sample can be
several times the measurement, so their `warm` cell is the median of five requests
rather than one: the middle of five, no interpolation. The five are taken in
sequence on the one warm listener, each a fresh detection stream over the same
second of audio, each gated as a single sample is gated, and each dropped before
the next begins; the five times are printed on the console, and the record and the
row's diagnostics line carry the middle one, while `warm_reread` states the arm's
pages summed over the five. `cold` stays a single sample on both devices: a second
cold sample in the same process would not be cold, because the weight pools and the
compiled pipelines the column's definition depends on survive the page-cache
eviction. The memory columns are read as they are for every row — resting before the
arm and the lifetime peak after it — so these two rows' peak spans five samples.

The enrolment and its calibration are a one-time setup outside the product's
start-up and are in neither column: the stand's is a pinned prototype
(`src/enrolment.rs`), loaded as a stored enrolment rather than derived, so no
model is read before the arm. It carries no anti-prototypes, so the decision's
veto takes its empty branch and its cost — one cosine of the prototype's width per
anti-prototype, about a tenth of a millisecond per second of audio, a thirteenth
of a percent of the active row — is not in these figures; a full negative pool
would also mean shipping stand-in enrolment data the product does not have.

The two rows' memory columns are not comparable with each other: the background
row's listener has never scored, and reads about two hundred mebibytes more
graphics-tagged memory than the active row's, which has scored. What holds that
difference and what retires it is not something these rows establish, and is not
claimed here.

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

The resident set cannot see pages allocated for the GPU. The stand's table drops the
GPU columns outright when every row is resident, and reads `not measured` in the GPU
cells of the resident rows of a mixed one — not zero, and not the system figure
repeated. The wake bench measures the resident set only, so it has no second half to
drop and its one GPU cell reads `not measured` by construction. There the two do not
split one pool.

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
`command` state where and how it ran — `device` names the adapter, or the processor
with how it was reached (`MAHBOT_DEVICE=cpu`, or no usable graphics device), and a
forced take names its switch beside the device on either backend, because a forced
take is not the shipped arrangement's; `command` is the recipe that produced that
take; `load` is the kernel's load average when the
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
