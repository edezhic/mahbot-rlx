// RLX — versatile ML compiler & runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether reading the recognition decode's weights once for several candidate
//! tokens can pay: the **draft acceptance** a weight-free drafter reaches on the
//! stand's own clip, and the cost of a multi-row pass at the decode's own
//! shape on its own resident plane cache. Both come from one process, so they
//! belong to the same load window.
//!
//! ```text
//! MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… RLX_ALLOW_THROTTLE=1 \
//! STAND_ASR_WAV=… [STAND_ASR_OFFSET=…] \
//! cargo test -p mahbot-inference --release --test draft_acceptance -- --ignored --nocapture
//! ```
//!
//! A second recording is measured through the same walk when `STAND_ASR_WAV`
//! names one; it joins the acceptance curve, the confidence and the decision, and
//! its rounds are priced at the pinned clip's measured pass. `STAND_ASR_OFFSET`
//! moves the window both walks start at, as it does for the cell.
//!
//! **The number.** Per-position acceptance at depths 1..8: at depth `i`, the
//! share of positions where the drafter's `i`-th proposed token equals the
//! model's own greedy choice (and the chain-conditional rate a round composes its
//! yield from). Coverage is the share of rounds where the drafter had anything to
//! copy, printed with the distribution of draft lengths — a near-zero depth-8
//! acceptance because nothing was proposed is a fact about the drafter, not a
//! verdict on the route. The rounds a walk would take are
//! **simulated over the clip's own stream** (draft, accept the longest prefix,
//! emit the model's own next token), so the yield comes from the measured curve
//! rather than from one assumed `alpha`.
//!
//! **The pass cost** is the shipped decode graph at 1..=[`DEPTH`] + 1 query rows
//! (a full-depth round drafts [`DEPTH`] candidates and runs one more row), on the
//! resident plane cache the decode carries, joined into the step the way a step
//! is: one run plus its row feeds, one cold pass, then [`ROUNDS`] warm passes
//! whose median is reported. The planes hold a finite ramp — the pass is priced,
//! not answered: with more than one query row the plane cache's mask is one key
//! mask for all of them, so the rows are not causal among themselves and the arm's
//! logits are not the greedy choices (`graphs::body_capacity_chain_planes_rows`).
//!
//! **The decision** prices both sides from this one reading — the baseline is the
//! harness's own one-row pass times the clip's token count, never the cell's
//! figure, and the cell's own per-token time prints beside it — and takes the
//! build only if the measured curve, at the measured pass cost, predicts at least
//! [`DECISION_WIN`] on the decode with the 5th percentile of the walk it implies
//! clearing it too. The break-even acceptance, the acceptance a [`DECISION_WIN`]
//! would need, and the same curve at the pass ratio the route was proposed on
//! (1.75×) print either way.
//!
//! Asserted: the drafter and the round simulation on the hand-computed example in
//! `the_round_rule_is_the_hand_computed_one`; the byte figure's read count on a
//! full-depth round; the
//! nine sweep arms' logits finite and `rows × vocab`; and the two-row operand's
//! first row equal to the one-row step's with the second key shut, while a visible
//! candidate moves it. The pinned clip's answer prints through the cell's own
//! witness rule in both modes — the rule keys on the file the walk read, not on the
//! switch that named it — so a moved digest or transcript is reported and the round
//! carries on. No wall time is asserted.
//!
//! # What it read on this tree
//!
//! Apple M3 Pro / Metal, one process, four runs of this file on this tree. The
//! pinned clip's 65 generated ids take 6 of 9 proposals at depth one (0.667, Wilson
//! 0.354..0.879), 2 of 9 at depth two, then 2, 1, 0, 1, 0, 0 — over **16% coverage**
//! and 51 drafted tokens, with a second 132-token recording joining at 0.647 (11 of
//! 17) and, once a draft of two matches, carrying the rest of an eight-token draft
//! with it (0.59 for the whole draft). The weight-free drafter's tail is thin on
//! free speech and dense on repeated phrasing: over that second recording its
//! three-id minimum accepts all 9 full drafts it proposes (1.000 at every depth)
//! over 17% coverage, and it drafts nothing over 49 of its 59 rounds.
//!
//! The pass does not cost 1.7–1.8 one-row steps. On the decode's own resident
//! cache and shapes it is **6.01–6.06** one-row steps at eight rows (96.0–96.6 ms
//! against 15.82–16.06) and **7.14–7.29** at nine across runs (114.0–118.4: the
//! ninth row re-reads the weight tiles for one row), and the arms are flat from two
//! rows up (92.4 → 96.6 across 2..8) — not because a pass steps once per eight rows,
//! but because the row-grouped GEMM streams the same tiles at about a quarter of
//! the one-row kernel's bandwidth. The weight stream alone, cache out of the
//! picture, is **3.78–3.85×** at nine rows (15.7 → 60.2 ms, 18.1–18.3 GB/s where the
//! one-row arm streams 68.3–70.1 — the unique-byte rate: the ninth row's
//! 14.2–15.2 ms is its second read of the tiles), and the many-row attention
//! arrangement at eight query rows over 370 keys is 58.0–58.7 ms against the one-row
//! kernel's 4.0–4.2 at the same key count. The one-row arm re-taken after the sweep
//! read 1.8% faster in one run and 3.7% slower in another: that, and the ninth arm's
//! spread, is the drift these ratios carry.
//!
//! So break-even acceptance is **0.917–0.919** at seven candidates on eight rows,
//! and **0.941–0.943** at eight on nine — where a [`DECISION_WIN`] is unreachable
//! outright, because the yield it needs (9.3–9.4 tokens a round) is past the nine a
//! round can emit; 0.757–0.763 at the weight stream's own ratio, and 0.429 at the
//! pass the route was proposed on (1.75×), where a [`DECISION_WIN`] would need
//! 0.563. The measured curve predicts **0.63×** (0.71× and 0.89× for the two longer
//! minimums), 0.83–0.84× with the many-row attention's long-cache data free, 1.05×
//! at the 1.75× premise, and 0.29–0.30× for a drafter that proposed at every
//! position.
//!
//! What decides the route is the pass, not the drafter. The second recording's
//! three-id minimum accepts every full draft it proposes and yields 2.24 tokens a
//! round over 17% coverage, and it still reaches only **1.07–1.11×** at the measured
//! pass — while at the 1.75× premise the same curve would read 1.92–1.99×. So the
//! pass's cost is the number to argue with: reading each weight tile once for eight
//! candidates would pay on repetitive speech, and this decode's arrangement does
//! not read it that way.
//!
//! Three things about the arm, so the figure is read as what it is. With more than
//! one query row the plane cache's mask is one key mask for all of them, so the
//! rows are not causal among themselves and the arm's logits are not the greedy
//! choices — it prices a pass, and a verification route would have to change that
//! mask and the one-row KV write (see
//! [`graphs::body_capacity_chain_planes_rows`]); the one-row path the shipped
//! decode runs is instead witnessed by the pinned answer, which the run prints
//! through the cell's own rule as `all_pinned=true reads_as_pinned=true`. The rows
//! reach
//! the operand through a concat of the resident rows — 85 MB at eight rows (the two
//! the parity check prints are 83 MB: about 0.6 ms one way and 1.3 ms read and
//! written at this machine's ceiling) inside a 96 ms pass — where the route's own
//! write is one `KvAppend` per row. That operand is checked against the
//! one-row step, not only against its shape: with the second candidate's key shut
//! the first row reproduces the one-row logits to 2.0e-3, and with every key open
//! the same row moves by 1.9e1. The residue is a summation order, not an operand —
//! one query row runs the fused attention kernel and the head's per-row multiply,
//! two rows the decomposed attention and the grouped multiply — and it is four
//! orders of magnitude below the effect a visible future candidate has. And the pass
//! has one form per row count on this adapter: the packed multiply takes the
//! row-grouped f32 GEMM at every multi-row count (`grouped:169` at 2..9 rows) and
//! the per-row kernel at one (`per_row:169`), and the tree's only cooperative
//! attention arrangement starts at 2 048 query rows, so this shape has no
//! capability-gated second number. Bytes per accepted token is computed from the
//! reads its rounds take — one a pass, two for a round past eight rows — and the
//! tokens they emit: it falls (1.03 GiB → 936 MiB at the one-id minimum) while the
//! time rises, which is why the byte rate does not decide the route.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mahbot_inference::listen::{ASR_HEAD_IDS, ASR_TAIL_IDS, ASR_TEXT_IDS};
use mahbot_inference::loader::{self, Pair, Role};
use mahbot_inference::{graphs, host, report, stand};
use rlx_ir::Tick;
use rlx_runtime::Device;

mod support;

/// Candidate tokens one pass may carry — the depth the drafter is asked for and
/// the most rows a pass runs.
const DEPTH: usize = 8;
/// The drafter's minimum suffix length, one curve each: 1 is the most repeated
/// continuation of a single id, 3 a repeated phrase.
const MIN_MATCHES: [usize; 3] = [1, 2, 3];
/// The longest suffix the search considers: a cost cap on the search, not on the
/// context, which reaches the 15 prompt ids plus every generated one.
const MAX_MATCH: usize = 64;
/// Warm passes per arm, odd so the median is a sample.
const ROUNDS: usize = 9;
/// A build is taken only if the measured curve predicts this much on the decode.
const DECISION_WIN: f64 = 1.3;
/// Monte-Carlo walks of the model the curve implies, for its 5/95% interval.
const DRAWS: usize = 20_000;

/// One depth's tally: positions that carried a proposal this deep, and how many
/// matched the model's own choice at that position.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct Tally {
    proposed: u64,
    matched: u64,
}

impl Tally {
    fn rate(&self) -> Option<f64> {
        (self.proposed > 0).then(|| self.matched as f64 / self.proposed as f64)
    }

    /// The 95% Wilson interval on [`Self::rate`] — each depth carries its own `n`,
    /// so a thin tail reports its width instead of a point.
    fn wilson(&self) -> Option<(f64, f64)> {
        let n = self.proposed as f64;
        if n == 0.0 {
            return None;
        }
        let p = self.matched as f64 / n;
        let z = 1.96;
        let denom = 1.0 + z * z / n;
        let centre = (p + z * z / (2.0 * n)) / denom;
        let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
        Some(((centre - half).max(0.0), (centre + half).min(1.0)))
    }
}

/// The weight-free drafter: at a position, the continuation of the latest earlier
/// occurrence of the longest suffix of the context that is at least `min_match`
/// ids long, up to [`DEPTH`] ids. The context is the prompt's own text ids
/// followed by the generated ones — the transcript so far, with no new weights and
/// no head to train. The encoder's audio rows are not in it, so a suffix may span
/// the text rows' end and the tail's start, an adjacency the model's own prompt
/// does not have; that is a property of this drafter, and the measured curve is
/// what prices it.
struct Drafter {
    ids: Vec<usize>,
    /// How many ids the prompt's text contributes before the first generated one.
    prefix: usize,
    min_match: usize,
}

/// What the drafter proposes at one position: `start` is the id the first
/// proposal copies, `len` how many ids it carries (`0` = nothing to copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Proposal {
    start: usize,
    len: usize,
}

const NOTHING: Proposal = Proposal { start: 0, len: 0 };

impl Drafter {
    /// The proposal at `ctx` ids of context (`ids[..ctx]`), never reading past
    /// `ctx` — the draft is causal. Longest suffix first, then the latest
    /// occurrence, so a repeated phrase drafts the continuation nearest to where
    /// the walk is.
    fn at(&self, ctx: usize) -> Proposal {
        let max_n = ctx.saturating_sub(1).min(MAX_MATCH);
        for n in (self.min_match..=max_n).rev() {
            let suffix = &self.ids[ctx - n..ctx];
            for e in (n..ctx).rev() {
                if self.ids[e - n..e] == *suffix {
                    return Proposal {
                        start: e,
                        len: (ctx - e).min(DEPTH),
                    };
                }
            }
        }
        NOTHING
    }

    fn proposed(&self, p: Proposal) -> &[usize] {
        &self.ids[p.start..p.start + p.len]
    }
}

/// One clip's measured draft behaviour and the walk it implies.
struct Curve {
    min_match: usize,
    /// Per-depth per-position acceptance: the share of positions where the draft's
    /// `i`-th token equals the model's own greedy choice there.
    depths: [Tally; DEPTH],
    /// The draft length each round could use, `0` where nothing was proposed or
    /// the stream had no room left to check a draft.
    lengths: Vec<usize>,
    /// The prefix each round accepted.
    accepts: Vec<usize>,
    /// The rows each round's pass runs: the row of the token it emitted last,
    /// which yields the distribution its first candidate is checked against, plus
    /// one row per candidate.
    passes: Vec<usize>,
    /// Tokens the clip generated: what the rounds have to emit, and what the
    /// baseline pays one one-row step each.
    emitted: usize,
    /// The walk this curve implies: emitted tokens per round at the 5th, 50th and
    /// 95th percentile over [`DRAWS`] seeded walks.
    walk: (f64, f64, f64),
}

impl Curve {
    fn rounds(&self) -> usize {
        self.passes.len()
    }

    fn coverage(&self) -> f64 {
        let proposed = self.lengths.iter().filter(|&&l| l > 0).count();
        proposed as f64 / self.lengths.len() as f64
    }

    fn tokens_per_round(&self) -> f64 {
        self.emitted as f64 / self.rounds() as f64
    }

    /// The weight-stream reads a round's pass takes, per token the clip emits: the
    /// row groups the grouped multiply dispatches for that row count — its `y`
    /// geometry, read from the backend rather than restated here, so the figure
    /// moves with the kernel's tile width and not with [`DEPTH`].
    fn stream_reads_per_token(&self) -> f64 {
        let reads: usize = self
            .passes
            .iter()
            .map(|&rows| rlx_wgpu::gguf_gpu::grouped_base_geometry(rows, 64).1 as usize)
            .sum();
        reads as f64 / self.emitted as f64
    }

    fn length_histogram(&self) -> [u64; DEPTH + 1] {
        let mut h = [0u64; DEPTH + 1];
        for &l in &self.lengths {
            h[l.min(DEPTH)] += 1;
        }
        h
    }

    /// The step a round composes its yield from: `P(the i-th token is right | the
    /// first i were)`, i.e. the share of the rounds that reached depth `i + 1` and
    /// matched it. [`Tally::rate`] is the per-position share of drafts that matched
    /// — the same numerator over every round that *proposed* that depth, including
    /// the rounds whose earlier token had already missed.
    fn conditional(&self) -> [f64; DEPTH] {
        let mut q = [0.0; DEPTH];
        for (i, slot) in q.iter_mut().enumerate() {
            // A round reaches depth `i + 1` when it drafted that far and the tokens
            // before it matched; one that drafted `i + 1` ids but stopped at `i` did
            // not, and is not counted against the drafter.
            let reached = self
                .lengths
                .iter()
                .zip(&self.accepts)
                .filter(|&(&l, &a)| l > i && a >= i)
                .count();
            if reached > 0 {
                *slot = self.accepts.iter().filter(|&&a| a > i).count() as f64 / reached as f64;
            }
        }
        q
    }

    /// Tokens a round emits at full depth from chain-conditional rates `q`:
    /// `1 + q₁ + q₁q₂ + …`.
    fn yield_at_full_depth(q: &[f64]) -> f64 {
        let (mut chain, mut tokens) = (1.0, 1.0);
        for r in q.iter().take(DEPTH) {
            chain *= r;
            tokens += chain;
        }
        tokens
    }
}

/// The acceptance-measuring walk over one clip's own greedy choices.
///
/// `choices[k]` is the model's greedy choice after the first `k` generated ids, so
/// `choices` is the generated stream plus the choice that stopped it. The prefill
/// produces `choices[0]`, so the walk starts at `1`: every round drafts at `k`,
/// accepts the longest prefix that equals the model's own choices, and emits that
/// prefix plus the model's own next token — one token even when nothing was
/// proposed or nothing matched. Its pass runs one row for the token it emitted
/// last (that row yields the distribution the first candidate is checked against)
/// and one row per candidate.
fn measure(drafter: &Drafter, choices: &[usize]) -> Curve {
    let mut curve = Curve {
        min_match: drafter.min_match,
        depths: [Tally::default(); DEPTH],
        lengths: Vec::new(),
        accepts: Vec::new(),
        passes: Vec::new(),
        emitted: choices.len() - 1,
        walk: (f64::NAN, f64::NAN, f64::NAN),
    };
    let mut k = 1usize;
    while k < choices.len() {
        let p = drafter.at(drafter.prefix + k);
        // The last choice can be emitted but not checked against a draft: a round
        // emits one more token than the prefix it accepts.
        let len = p.len.min(choices.len() - k - 1);
        for i in 0..len {
            curve.depths[i].proposed += 1;
            if drafter.proposed(p)[i] == choices[k + i] {
                curve.depths[i].matched += 1;
            }
        }
        let mut accept = 0;
        while accept < len && drafter.proposed(p)[accept] == choices[k + accept] {
            accept += 1;
        }
        curve.lengths.push(len);
        curve.accepts.push(accept);
        curve.passes.push(1 + len);
        k += accept + 1;
    }
    curve.walk = model_walk(&curve, curve.emitted);
    curve
}

/// Measured pass costs in milliseconds, one per row count `1..=DEPTH + 1`: a
/// full-depth round drafts [`DEPTH`] candidates and runs one more row.
type PassCost = [f64; DEPTH + 2];

/// The predicted decode: every simulated round pays its own pass, and nothing
/// else — the row feeds and the logits readback sit inside the pass.
fn predicted_ms(curve: &Curve, cost: &PassCost) -> f64 {
    curve
        .passes
        .iter()
        .map(|&rows| cost[rows.min(DEPTH + 1)])
        .sum()
}

/// The geometric yield a constant acceptance `alpha` gives with `drafts`
/// candidates: `1 + alpha + … + alpha^drafts`.
fn yield_at(alpha: f64, drafts: usize) -> f64 {
    (0..=drafts).map(|i| alpha.powi(i as i32)).sum()
}

/// The constant acceptance whose yield is `target` with `drafts` candidates — the
/// geometric arithmetic the route was proposed with, on the measured pass cost.
/// `None` when no acceptance reaches it.
fn acceptance_for(target: f64, drafts: usize) -> Option<f64> {
    if target <= 1.0 {
        return Some(0.0);
    }
    if yield_at(1.0, drafts) < target {
        return None;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if yield_at(mid, drafts) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The walk the measured curve implies: a draft length drawn from the measured
/// length distribution, then each drafted position accepted independently at its
/// own chain-conditional rate. Answers the 5th, 50th and 95th percentile of
/// emitted tokens per round over [`DRAWS`] seeded walks — the interval is over the
/// walk, not over the per-depth rates' own uncertainty, which the Wilson bounds
/// beside each depth carry. A curve with nothing to draw from answers `NaN`.
fn model_walk(curve: &Curve, choices: usize) -> (f64, f64, f64) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(0x6472_6166_7421);
    let hist = curve.length_histogram();
    let total: u64 = hist.iter().sum();
    if total == 0 || choices == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let rates = curve.conditional();
    let mut per_round: Vec<f64> = Vec::with_capacity(DRAWS);
    for _ in 0..DRAWS {
        let mut emitted = 0usize;
        let mut rounds = 0usize;
        while emitted < choices {
            let mut pick = rng.random_range(0..total);
            let mut len = DEPTH;
            for (l, &count) in hist.iter().enumerate() {
                if pick < count {
                    len = l;
                    break;
                }
                pick -= count;
            }
            let mut accept = 0;
            while accept < len && rng.random::<f64>() < rates[accept] {
                accept += 1;
            }
            emitted += accept + 1;
            rounds += 1;
        }
        per_round.push(emitted as f64 / rounds as f64);
    }
    per_round.sort_by(|a, b| a.partial_cmp(b).expect("draws compare"));
    let at = |q: f64| per_round[((per_round.len() - 1) as f64 * q) as usize];
    (at(0.05), at(0.5), at(0.95))
}

/// One pass-cost arm's reading.
struct Pass {
    rows: usize,
    build_ms: f64,
    cold_ms: f64,
    warm_ms: f64,
    min_ms: f64,
    max_ms: f64,
    submits: (u64, u64),
    kinds: String,
    routes: (u64, u64, u64, u64, u64),
    finite: bool,
    readback: usize,
    /// Row moves the step's feeds carry, `rows · 2 · layers` of them.
    feeds: usize,
    /// Submissions they cost, one per row: the runtime's feed folds every layer's
    /// K and V in one call.
    feed_calls: usize,
}

/// The decode's own graph at `rows` query rows, on the resident plane cache the
/// decode carries: built, bound, one cold pass, then [`ROUNDS`] warm passes. It
/// prices a pass, not an answer — the arm's logits are not causal (see
/// [`graphs::body_capacity_chain_planes_rows`]) — so only its time and the route
/// it took are read.
fn pass_cost(pair: &Pair, capacity: usize, rows: usize) -> Result<Pass, String> {
    let cfg = pair.body.config().clone();
    let hkv = cfg.kv_heads * cfg.head_dim;
    let built = graphs::body_capacity_chain_planes_rows(pair.body.clone(), capacity, rows)
        .map_err(|e| format!("build: {e:#}"))?;
    let t = report::Stopwatch::start();
    let mut compiled =
        graphs::Compiled::new(Device::Gpu, built).map_err(|e| format!("compile: {e:#}"))?;
    let build_ms = t.ms();
    let names = compiled
        .runner
        .schedule_step_names()
        .ok_or("the schedule names no step")?;
    let kinds = support::step_census(&names);
    let plane = graphs::plane_rows(capacity, rows);
    let k: Vec<Vec<f32>> = (0..cfg.layers)
        .map(|_| support::ramped_row(plane * hkv))
        .collect();
    let v: Vec<Vec<f32>> = (0..cfg.layers)
        .map(|_| support::ramped_row(plane * hkv))
        .collect();
    if !compiled.runner.resident_cache_planes(&k, &v) {
        return Err("the device refused the resident planes".into());
    }
    // The device holds the planes now; dropping the host copies keeps the arm's
    // peak at its weights.
    drop(k);
    drop(v);
    let (cos, sin) = host::rope_tables_at(capacity, rows, cfg.head_dim, cfg.rope_base as f64);
    let inputs = vec![
        ("x".to_string(), support::ramped_row(rows * cfg.d)),
        ("cos".to_string(), f32s(&cos)),
        ("sin".to_string(), f32s(&sin)),
        ("mask".to_string(), vec![1f32; capacity + rows]),
    ];
    let read = [graphs::plane_step_logits(cfg.layers)];
    // The live cursor the feeds write below: any row inside the capacity is the
    // step's own slot once the prompt's rows are in the cache.
    let row = capacity - rows;
    let c0 = graphs::device_counts();
    let mut cold_ms = 0.0;
    let mut warm: Vec<f64> = Vec::with_capacity(ROUNDS);
    for i in 0..=ROUNDS {
        let t = Tick::now();
        compiled
            .runner
            .run_resident_planes_rows(&inputs, row, rows, hkv, &read)
            .ok_or_else(|| format!("the step or its {rows} row feeds were refused at {row}"))?;
        let ms = Tick::now().elapsed_ns(t) as f64 / 1e6;
        if i == 0 {
            cold_ms = ms;
        } else {
            warm.push(ms);
        }
    }
    let c = graphs::device_counts();
    // One more pass brackets the packed-multiply routes it took.
    rlx_wgpu::gguf_gpu::reset_route_counts();
    let o = compiled
        .runner
        .run_resident_planes_rows(&inputs, row, rows, hkv, &read)
        .ok_or("the bracketing pass was refused")?;
    let r = rlx_wgpu::gguf_gpu::route_counts();
    let finite = o.iter().all(|t| t.iter().all(|v| v.is_finite()));
    let readback: usize = o.iter().map(|t| t.len()).sum();
    Ok(Pass {
        rows,
        build_ms,
        cold_ms,
        warm_ms: support::median(&warm),
        min_ms: support::least(&warm),
        max_ms: support::greatest(&warm),
        submits: (c.0 - c0.0, c.1 - c0.1),
        kinds,
        routes: (r.grouped, r.per_row, r.q1_0_gemm, r.scratch, r.host),
        finite,
        readback,
        feeds: rows * graphs::plane_feed_moves(cfg.layers),
        feed_calls: rows,
    })
}

/// The prompt's own text ids, in the order the model reads them: the head rows,
/// then the text rows, then the tail rows — the encoder's 219 audio rows sit
/// between the text and tail rows and are not in the returned vector.
fn prompt_text_ids() -> Vec<usize> {
    ASR_HEAD_IDS
        .iter()
        .chain(ASR_TEXT_IDS.iter())
        .chain(ASR_TAIL_IDS.iter())
        .copied()
        .collect()
}

fn pair(role: Role) -> Option<Arc<Pair>> {
    match Pair::of_role(role) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

fn voice() -> Option<PathBuf> {
    match loader::voice_wav() {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

/// The second recording: the listen cell's own path selector, which answers the
/// reference voice when `STAND_ASR_WAV` is unset — that is the pinned clip, already
/// measured, so only a different file is a second recording.
fn alt_clip(pinned: &Path) -> Option<PathBuf> {
    let path = stand::listen_path().ok()?;
    if path == pinned {
        return None;
    }
    if path.exists() {
        return Some(path);
    }
    println!(
        "census draft alternate skipped=missing-path {}",
        path.display()
    );
    None
}

/// One clip through the listen cell's own walk, its stream printed against the
/// pin. The cell's own offset switch is honoured, so the harness reads the input
/// the cell would read, and the cell's own witness rule and alternate-input
/// predicate are reused rather than restated — the predicate keys on the file the
/// walk read, so the pinned clip is still witnessed beside a second recording.
fn clip(pair: &Arc<Pair>, name: &str, path: &Path) -> Option<stand::Answer> {
    let offset = stand::listen_offset();
    let a = stand::listen_answer(pair, path, offset)?;
    println!(
        "census draft clip {name} frames={} prompt={} generated={} capacity={} text_ids={} \
         decode_ms={:.3} stream_bytes={} answer_digest={:016x}",
        a.frames,
        a.prompt_rows,
        a.generated.len(),
        a.capacity,
        prompt_text_ids().len(),
        a.decode_ms,
        a.stream_bytes,
        a.digest,
    );
    // The cell's own witness rule, called rather than restated: the predicate inside
    // it decides from the file this walk read, so the pinned clip is witnessed even
    // beside a second recording.
    stand::witness_listen(
        path,
        offset,
        a.frames,
        a.prompt_rows,
        a.generated.len(),
        a.digest,
        &a.text,
    );
    Some(a)
}

/// Every curve a clip yields: per-depth acceptance with its Wilson interval and
/// its conditional rate, coverage, the length distribution, the walk the curve
/// implies and the chain product.
fn curves_of(name: &str, a: &stand::Answer) -> Vec<Curve> {
    let text = prompt_text_ids();
    let mut ids = text.clone();
    ids.extend_from_slice(&a.generated);
    let mut choices = a.generated.clone();
    choices.push(a.next);
    let mut curves = Vec::new();
    for min_match in MIN_MATCHES {
        let drafter = Drafter {
            ids: ids.clone(),
            prefix: text.len(),
            min_match,
        };
        let curve = measure(&drafter, &choices);
        let q = curve.conditional();
        let chain: f64 = q.iter().product();
        println!(
            "census draft curve {name} min_match={min_match} rounds={} emitted={} coverage={:.3} \
             tokens_per_round={:.3} proposed={} all8={chain:.4} geometric_alpha={:.3} \
             yield_if_always_proposed={:.3} lengths={:?}",
            curve.rounds(),
            curve.emitted,
            curve.coverage(),
            curve.tokens_per_round(),
            curve.depths.iter().map(|t| t.proposed).sum::<u64>(),
            chain.powf(1.0 / DEPTH as f64),
            Curve::yield_at_full_depth(&q),
            curve.length_histogram(),
        );
        for (i, t) in curve.depths.iter().enumerate() {
            let (lo, hi) = t.wilson().unwrap_or((0.0, 0.0));
            println!(
                "census draft depth {name} min_match={min_match} depth={} proposed={} \
                 matched={} rate={:.4} wilson95={lo:.4}..{hi:.4} conditional={:.4}",
                i + 1,
                t.proposed,
                t.matched,
                t.rate().unwrap_or(0.0),
                q[i],
            );
        }
        let (p05, p50, p95) = curve.walk;
        println!(
            "census draft model {name} min_match={min_match} emitted_per_round={p05:.3}/\
             {p50:.3}/{p95:.3} rounds_p50={:.2}",
            curve.emitted as f64 / p50,
        );
        curves.push(curve);
    }
    curves
}

fn alpha_text(alpha: Option<f64>) -> String {
    match alpha {
        Some(a) => format!("{a:.3}"),
        None => "unreachable".to_string(),
    }
}

/// The pass arms and the decision: measured costs at 1..=[`DEPTH`] + 1 rows, the
/// predicted decode from each curve, the break-even acceptance and the win a build
/// would have to reach. `sets` is one `(clip, curves)` pair per recording, the
/// pinned clip first.
fn decide(pair: &Pair, answer: &stand::Answer, sets: &[(&str, &[Curve])]) {
    let capacity = answer.capacity;
    let tokens = answer.generated.len() as f64;
    let mut cost: PassCost = [f64::NAN; DEPTH + 2];
    for rows in 1..=DEPTH + 1 {
        match pass_cost(pair, capacity, rows) {
            Ok(p) => {
                println!(
                    "census pass rows={} build_ms={:.1} feeds={} feed_calls={} cold_ms={:.3} \
                     warm_ms={:.3} min_ms={:.3} max_ms={:.3} submits={}/{} readback={} finite={} \
                     routes=grouped:{} per_row:{} q1_0:{} scratch:{} host:{} kinds={}",
                    p.rows,
                    p.build_ms,
                    p.feeds,
                    p.feed_calls,
                    p.cold_ms,
                    p.warm_ms,
                    p.min_ms,
                    p.max_ms,
                    p.submits.0,
                    p.submits.1,
                    p.readback,
                    p.finite,
                    p.routes.0,
                    p.routes.1,
                    p.routes.2,
                    p.routes.3,
                    p.routes.4,
                    p.kinds,
                );
                assert!(p.finite, "rows={} wrote a non-finite logit", p.rows);
                assert_eq!(
                    p.readback,
                    p.rows * cost_vocab(pair),
                    "rows={} read back {} values",
                    p.rows,
                    p.readback
                );
                cost[rows] = p.warm_ms;
            }
            Err(e) => {
                println!("census pass rows={rows} refused reason={e}");
            }
        }
    }
    // The operand's arithmetic, not only its shape: the first candidate row of a
    // two-row pass with the second key shut must equal the one-row step, and the
    // same pass with every key open shows what a visible future candidate does to
    // that row.
    match parity(pair, capacity) {
        Ok((masked, leaked, bytes)) => {
            println!(
                "census pass parity rows=2 keys={} masked_maxdiff={masked:.2e} \
                 leaked_maxdiff={leaked:.2e} operand_bytes={bytes}",
                capacity + 2
            );
            // Both sides of the check: the masked row must sit at the one-row step's
            // logits to a route-reordering residue — 28 layers of a different
            // summation order on logits of order 10 measure 2.0e-3, so 1e-2 is a few
            // times the measurement — and a *visible* future candidate must move it
            // by far more than that, or the second row was not carrying the candidate
            // the non-causality is about.
            assert!(masked < 1e-2, "the masked candidate row moved by {masked}");
            assert!(
                leaked > 1.0,
                "a visible future candidate moved the row by only {leaked}"
            );
        }
        Err(e) => println!("census pass parity refused reason={e}"),
    }
    let one = cost[1];
    // A full-depth round: `DEPTH` candidates plus the row that yields the
    // distribution the first of them is checked against.
    let deep = cost[DEPTH + 1];
    if cost[1..=DEPTH + 1].iter().any(|c| !c.is_finite()) {
        println!("census decision skipped=pass-arm-refused");
        return;
    }
    // Both sides of every ratio below come from this one reading: the baseline is
    // the harness's own one-row pass × the clip's token count, so a round is never
    // priced at one process's step and the baseline at another's. The cell's own
    // per-token figure prints beside it as context.
    let baseline_total = one * tokens;
    println!(
        "census pass cost one_ms={one:.3} deep_ms={deep:.3} ratio={:.3} \
         cell_step_ms={:.3} one_over_cell={:.3} tokens={tokens:.0} baseline_ms={baseline_total:.3} \
         cell_baseline_ms={:.3}",
        deep / one,
        answer.decode_ms,
        one / answer.decode_ms,
        answer.decode_ms * tokens,
    );
    // The sweep runs 1→9 in one process, so a machine that slows as it warms can
    // only inflate the ratio. Re-take the one-row arm after the sweep and print the
    // drift it shows: that bounds how much of the ratio is the clock rather than
    // the arrangement, and the verdict above already uses the first reading.
    match pass_cost(pair, capacity, 1) {
        Ok(again) => println!(
            "census pass cost drift one_ms={one:.3} one_again_ms={:.3} drift={:.3} \
             ratio={:.3} ratio_at_drift={:.3}",
            again.warm_ms,
            again.warm_ms / one,
            deep / one,
            deep / again.warm_ms,
        ),
        Err(e) => println!("census pass cost drift refused reason={e}"),
    }
    // The geometric arithmetic the route was proposed with, on the measured cost,
    // at both depths that matter: eight candidates run nine rows, and the ninth
    // re-reads the weight tiles (the grouped route steps once per eight rows), so
    // seven candidates on eight rows is the deepest draft that does not.
    for (drafts, rows) in [(DEPTH, DEPTH + 1), (DEPTH - 1, DEPTH)] {
        let ratio = cost[rows] / one;
        println!(
            "census break_even drafts={drafts} rows={rows} pass_ratio={ratio:.3} alpha={} \
             alpha_at_{DECISION_WIN}x={} yield_needed={:.3}",
            alpha_text(acceptance_for(ratio, drafts)),
            alpha_text(acceptance_for(DECISION_WIN * ratio, drafts)),
            DECISION_WIN * ratio,
        );
    }
    // Where the pass's time goes: the attention term alone at the same key count,
    // the same pass on a short cache (the weight stream and the decomposed
    // arrangement's dispatch cost), and the weight stream with the plane cache out
    // of the picture.
    let mut tiny = [f64::NAN; DEPTH + 1];
    for rows in [1usize, DEPTH] {
        match attention_only(pair, rows, capacity + rows) {
            Ok((ms, routes)) => println!(
                "census attention rows={rows} keys={} layers={} ms={ms:.3} route={routes}",
                capacity + rows,
                pair.body.config().layers,
            ),
            Err(e) => println!("census attention rows={rows} refused reason={e}"),
        }
        let short = DEPTH + 1;
        match pass_cost(pair, short, rows) {
            Ok(p) => {
                println!(
                    "census pass diagnostic rows={rows} capacity={short} warm_ms={:.3} \
                     routes=grouped:{} per_row:{} kinds={}",
                    p.warm_ms, p.routes.0, p.routes.1, p.kinds
                );
                tiny[rows] = p.warm_ms;
            }
            Err(e) => println!("census pass diagnostic rows={rows} refused reason={e}"),
        }
    }
    // The weight stream on its own, at the decode's own tensors, at the three row
    // counts the arms below charge: one read of every weight at one row, two at
    // nine — the grouped route takes eight rows a group, so the ninth row reads the
    // tiles again — hence `ninth` is what the ninth row's second read costs.
    let mut ninth = f64::NAN;
    match (
        weights_cost(pair, 1),
        weights_cost(pair, DEPTH),
        weights_cost(pair, DEPTH + 1),
    ) {
        (Ok(w1), Ok(w8), Ok(w9)) => {
            let ratio = w9 / w1;
            println!(
                "census pass weights rows=1 ms={w1:.3} rows={DEPTH} ms={w8:.3} rows={} ms={w9:.3} \
                 ratio={ratio:.3} ninth_row_ms={:.3}",
                DEPTH + 1,
                w9 - w8,
            );
            ninth = w9 - w8;
            // What the route's acceptance would have to be if a pass cost only the
            // weight stream it reads — the floor this decode's arrangement does not
            // reach, and a measured ratio rather than a modelled one.
            println!(
                "census break_even weights pass_ratio={ratio:.3} alpha={} \
                 alpha_at_{DECISION_WIN}x={}",
                alpha_text(acceptance_for(ratio, DEPTH)),
                alpha_text(acceptance_for(DECISION_WIN * ratio, DEPTH)),
            );
        }
        (a, b, c) => println!(
            "census pass weights refused reason={}",
            a.err().or(b.err()).or(c.err()).unwrap_or_default()
        ),
    }
    // The pass the route would have if its many-row attention's long-cache data
    // cost went away (the decomposed arrangement's own dispatch cost stays): the
    // one-row pass plus the increment a group of rows adds on the short cache —
    // measured as that arm's eight-row pass minus its one-row pass. Charging that one
    // increment to every row count above one is the *main sweep's* finding, not the
    // short-cache arm's, which measured only rows 1 and 8: the sweep is flat from two
    // rows up (92.4 → 96.6 across 2..8). The ninth row also pays the weight stream's
    // second read, measured above rather than inferred — and a refused measurement
    // refuses the whole arm, rather than pricing the ninth row's second read as free.
    let mut implied: PassCost = [f64::NAN; DEPTH + 2];
    if tiny[1].is_finite() && tiny[DEPTH].is_finite() && ninth.is_finite() {
        implied[1] = one;
        for d in 2..=DEPTH + 1 {
            implied[d] = one + (tiny[DEPTH] - tiny[1]);
        }
        implied[DEPTH + 1] += ninth;
    } else {
        println!("census pass implied refused=short-cache-or-ninth-row-arm");
    }
    // The pass cost the route was proposed on (1.7–1.8× the one-row step), so the
    // measured curve is also evaluated where the original premise put the pass
    // rather than only where this decode's arrangement puts it.
    const PREMISE_RATIO: f64 = 1.75;
    let mut premise_cost: PassCost = [f64::NAN; DEPTH + 2];
    premise_cost[1] = one;
    for d in 2..=DEPTH + 1 {
        premise_cost[d] = one * PREMISE_RATIO;
    }
    println!(
        "census break_even premise pass_ratio={PREMISE_RATIO:.3} alpha={} \
         alpha_at_{DECISION_WIN}x={}",
        alpha_text(acceptance_for(PREMISE_RATIO, DEPTH)),
        alpha_text(acceptance_for(DECISION_WIN * PREMISE_RATIO, DEPTH)),
    );
    // Every clip's curve is priced at *this* capacity's measured costs, so the
    // second recording's lines carry the first one's pass and are indicative: they
    // answer whether a more favourable drafter would change the verdict, not what
    // that clip's own decode costs.
    for (clip, set) in sets {
        for curve in *set {
            if curve.rounds() == 0 || curve.emitted == 0 {
                println!(
                    "census decision clip={clip} min_match={} skipped=no-rounds",
                    curve.min_match
                );
                continue;
            }
            // The clip's own token count: what the baseline pays one row each, and
            // what a round emits from.
            let tokens = curve.emitted as f64;
            let baseline_total = one * tokens;
            let predicted = predicted_ms(curve, &cost);
            let win = baseline_total / predicted;
            let mean_pass = predicted / curve.rounds() as f64;
            let (p05, p50, p95) = curve.walk;
            // The walk the curve implies pays `mean_pass` a round and needs
            // `emitted / tokens-per-round` rounds; its low end is the highest time.
            let win_p05 = baseline_total / (curve.emitted as f64 / p05 * mean_pass);
            // The same curve against the pass the route would have if its many-row
            // attention's long-cache data cost went away; this is what says whether
            // the route is closed on the drafter alone. A refused arm prints
            // `refused` rather than a ratio with a missing increment priced as free.
            let implied_ms = predicted_ms(curve, &implied);
            let win_implied = baseline_total / implied_ms;
            let (win_implied, implied_ms) = if win_implied.is_finite() && implied_ms.is_finite() {
                (format!("{win_implied:.3}"), format!("{implied_ms:.3}"))
            } else {
                ("refused".to_string(), "refused".to_string())
            };
            let win_premise = baseline_total / predicted_ms(curve, &premise_cost);
            // What the same chain-conditional rates would yield if the drafter
            // proposed at every position: precision, separated from coverage.
            let full = Curve::yield_at_full_depth(&curve.conditional());
            let win_full = baseline_total / (tokens / full * cost[DEPTH + 1]);
            let passes_per_token = curve.rounds() as f64 / tokens;
            // Bytes per *accepted* token, computed from the reads the curve's rounds
            // take — the row groups their passes dispatch.
            let reads_per_token = curve.stream_reads_per_token();
            println!(
                "census decision clip={clip} cost_capacity={capacity} min_match={} rounds={} \
                 tokens={tokens:.0} tokens_per_round={:.3} predicted_ms={predicted:.3} \
                 measured_ms={baseline_total:.3} win={win:.3} win_model_p05={win_p05:.3} \
                 win_no_attention_defect={win_implied} implied_ms={implied_ms} \
                 win_at_{PREMISE_RATIO:.2}x={win_premise:.3} \
                 yield_if_always_proposed={full:.3} \
                 win_if_always_proposed={win_full:.3} per_token_ms={:.3} \
                 passes_per_token={passes_per_token:.3} stream_reads_per_token={reads_per_token:.3} \
                 bytes_per_accepted_token={} bytes_per_read={} \
                 model={p05:.3}/{p50:.3}/{p95:.3} verdict={}",
                curve.min_match,
                curve.rounds(),
                curve.tokens_per_round(),
                predicted / tokens,
                report::human((reads_per_token * answer.stream_bytes as f64) as u64),
                report::human(answer.stream_bytes),
                if win >= DECISION_WIN && win_p05 >= DECISION_WIN {
                    "build"
                } else {
                    "closed"
                },
            );
        }
    }
}

/// RoPE tables are f64 on the host and f32 in the graph.
fn f32s(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

/// The largest absolute difference between two logit rows.
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The multi-row operand's arithmetic against the one-row step, not only its
/// shape: the first row of a two-row pass, with the second candidate's key masked
/// out, must reproduce the one-row step's logits from the same cache, input and
/// position — the operand the decode's own in-graph append produces. Its residue is
/// a summation order, not an operand: one query row runs the decode attention
/// kernel and the per-row packed multiply, two rows the decomposed attention and the
/// grouped multiply. The same pass with every key admitted measures how far a
/// *visible* future candidate moves that row: the non-causality the arm is documented
/// with, and the reason it is priced and not answered. Answers `(masked, leaked,
/// operand bytes)`.
fn parity(pair: &Pair, capacity: usize) -> Result<(f32, f32, usize), String> {
    let cfg = pair.body.config().clone();
    let (d, hkv, layers) = (cfg.d, cfg.kv_heads * cfg.head_dim, cfg.layers);
    let vocab = cost_vocab(pair);
    let read = [graphs::plane_step_logits(layers)];
    let ramp = |rows: usize| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (
            (0..layers)
                .map(|_| support::ramped_row(rows * hkv))
                .collect(),
            (0..layers)
                .map(|_| support::ramped_row(rows * hkv))
                .collect(),
        )
    };
    // The one-row step: its cache is `capacity + 1` rows (the spare row its
    // in-graph append writes) and its query sits at `capacity`.
    let planes = ramp(capacity + 1);
    let built = graphs::body_capacity_chain_planes(pair.body.clone(), capacity)
        .map_err(|e| format!("build: {e:#}"))?;
    let mut one =
        graphs::Compiled::new(Device::Gpu, built).map_err(|e| format!("compile: {e:#}"))?;
    if !one.runner.resident_cache_planes(&planes.0, &planes.1) {
        return Err("the device refused the resident planes".into());
    }
    drop(planes);
    let (cos1, sin1) = host::rope_tables_at(capacity, 1, cfg.head_dim, cfg.rope_base as f64);
    let expected = row_logits(
        &mut one,
        support::ramped_row(d),
        &cos1,
        &sin1,
        vec![1.0; capacity + 1],
        &read,
    )?;
    drop(one);
    // Two rows: the same cache without the spare row, the same input twice, and
    // the second candidate's key shut so both rows read the one-row operand. The
    // pass runs twice on one compiled graph — nothing feeds the cache between
    // them, so the second run reads the same planes.
    let planes = ramp(capacity);
    let built = graphs::body_capacity_chain_planes_rows(pair.body.clone(), capacity, 2)
        .map_err(|e| format!("build: {e:#}"))?;
    let mut two =
        graphs::Compiled::new(Device::Gpu, built).map_err(|e| format!("compile: {e:#}"))?;
    if !two.runner.resident_cache_planes(&planes.0, &planes.1) {
        return Err("the device refused the resident planes".into());
    }
    drop(planes);
    let (cos2, sin2) = host::rope_tables_at(capacity, 2, cfg.head_dim, cfg.rope_base as f64);
    let row = support::ramped_row(d);
    let x = [row.clone(), row].concat();
    let mut shut = vec![1.0; capacity + 1];
    shut.push(0.0);
    let masked = row_logits(&mut two, x.clone(), &cos2, &sin2, shut, &read)?;
    let leaked = row_logits(&mut two, x, &cos2, &sin2, vec![1.0; capacity + 2], &read)?;
    assert_eq!(
        expected.len(),
        vocab,
        "the one-row step read {} values",
        expected.len()
    );
    assert_eq!(
        masked.len(),
        2 * vocab,
        "the two-row pass read {} values",
        masked.len()
    );
    Ok((
        max_diff(&expected, &masked[..vocab]),
        max_diff(&expected, &leaked[..vocab]),
        2 * layers * (capacity + 2) * hkv * 4,
    ))
}

/// One plane step's logits, fed the way [`parity`] compares them.
fn row_logits(
    c: &mut graphs::Compiled,
    x: Vec<f32>,
    cos: &[f64],
    sin: &[f64],
    mask: Vec<f32>,
    read: &[usize],
) -> Result<Vec<f32>, String> {
    let inputs = vec![
        ("x".to_string(), x),
        ("cos".to_string(), f32s(cos)),
        ("sin".to_string(), f32s(sin)),
        ("mask".to_string(), mask),
    ];
    c.runner
        .run_read(&inputs, read)
        .into_iter()
        .next()
        .ok_or_else(|| "the pass read nothing".to_string())
}

fn cost_vocab(pair: &Pair) -> usize {
    match pair.body.config().head {
        loader::Head::Separate { vocab, .. } | loader::Head::Tied { vocab, .. } => vocab,
    }
}

/// The weight stream on its own, cache out of the picture: the prompt graph at
/// `rows` query rows (`prefix = 0`, the body's own head), warm median. It reads
/// every base tensor once — gate and up as two multiplies where the plane arm
/// fuses them (197 ops against that arm's 169, on the same bytes) — so its ratio is
/// the weight stream's own, measured on the decode's tensors rather than assumed
/// from one wide multiply.
fn weights_cost(pair: &Pair, rows: usize) -> Result<f64, String> {
    let built = graphs::body_new_rows_head_chain(pair.body.clone(), rows, 0, true)
        .map_err(|e| format!("build: {e:#}"))?;
    let mut compiled =
        graphs::Compiled::new(Device::Gpu, built).map_err(|e| format!("compile: {e:#}"))?;
    let cfg = pair.body.config();
    let (cos, sin) = host::rope_tables_at(0, rows, cfg.head_dim, cfg.rope_base as f64);
    let inputs = vec![
        ("x".to_string(), support::ramped_row(rows * cfg.d)),
        ("cos".to_string(), f32s(&cos)),
        ("sin".to_string(), f32s(&sin)),
    ];
    let _ = compiled.runner.run(&inputs);
    let mut warm = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t = Tick::now();
        compiled.runner.run(&inputs);
        warm.push(Tick::now().elapsed_ns(t) as f64 / 1e6);
    }
    Ok(support::median(&warm))
}

/// The pass's attention term, isolated at the decode's own geometry in the
/// head-canonical rank-3 layout: `layers` chained many-row attention ops over
/// `keys` keys against the decode's custom mask, warm median over [`ROUNDS`] runs.
/// This is the term that decides whether a many-row pass can pay at all, so it is
/// measured rather than inferred from the pass's residue.
fn attention_only(pair: &Pair, rows: usize, keys: usize) -> Result<(f64, String), String> {
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::Session;
    let cfg = pair.body.config();
    let f = DType::F32;
    let dim = cfg.heads * cfg.head_dim;
    let kv = cfg.kv_heads * cfg.head_dim;
    let mut g = Graph::new(format!("attention_{rows}_{keys}"));
    let mut q = g.input("q", Shape::new(&[1, rows, dim], f));
    let k = g.input("k", Shape::new(&[1, keys, kv], f));
    let v = g.input("v", Shape::new(&[1, keys, kv], f));
    let mask = g.input("mask", Shape::new(&[1, keys], f));
    for _ in 0..cfg.layers {
        q = g.attention(
            q,
            k,
            v,
            mask,
            cfg.heads,
            cfg.head_dim,
            Shape::new(&[1, rows, dim], f),
        );
    }
    g.set_outputs(vec![q]);
    let mut c = Session::new(Device::Gpu).compile(g);
    let (q, k, v, mask) = (
        support::ramped_row(rows * dim),
        support::ramped_row(keys * kv),
        support::ramped_row(keys * kv),
        vec![1f32; keys],
    );
    let inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v), ("mask", &mask)];
    let _ = c.run(&inputs);
    let mut warm = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t = Tick::now();
        let _ = c.run(&inputs);
        warm.push(Tick::now().elapsed_ns(t) as f64 / 1e6);
    }
    rlx_wgpu::kernels::reset_attention_route_counts();
    let _ = c.run(&inputs);
    let routes = rlx_wgpu::kernels::attention_route_counts();
    Ok((support::median(&warm), format!("{routes:?}")))
}

#[test]
#[ignore = "measurement, not a gate"]
fn draft_acceptance() {
    support::assert_run_report_off();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let Some(pair) = pair(Role::Recognition) else {
        return;
    };
    let Some(voice) = voice() else {
        return;
    };
    let started = report::Stopwatch::start();
    let Some(pinned) = clip(&pair, "pinned", &voice) else {
        eprintln!("skip: the recognition walk did not answer");
        return;
    };
    let curves = curves_of("pinned", &pinned);
    let mut other: Vec<Curve> = Vec::new();
    if let Some(alt) = alt_clip(&voice)
        && let Some(second) = clip(&pair, "alternate", &alt)
    {
        other = curves_of("alternate", &second);
        // The second recording joins the confidence and the decision — a curve
        // that more than one recording reaches is what says whether a verdict is
        // the drafter's or the clip's.
        for (i, (a, b)) in curves[0].depths.iter().zip(&other[0].depths).enumerate() {
            println!(
                "census draft joined depth={} pinned={}/{} other={}/{}",
                i + 1,
                a.matched,
                a.proposed,
                b.matched,
                b.proposed,
            );
        }
        if let (Some(x), Some(y)) = (curves[0].depths[0].rate(), other[0].depths[0].rate()) {
            println!(
                "census draft joined depth=1 rate={:.4} other={:.4} delta={:+.4}",
                x,
                y,
                y - x
            );
        }
    }
    decide(
        &pair,
        &pinned,
        &[("pinned", &curves), ("alternate", &other)],
    );
    println!(
        "census draft done build_ms={:.1} process_peak_bytes={} wall_ms={:.1}",
        pinned.build_ms,
        report::peak_text(),
        started.ms(),
    );
}

/// The drafter and the round simulation on a stream worked out by hand: the
/// longest suffix with an earlier occurrence wins, the latest occurrence breaks
/// the tie, a draft never reads past the context, and a round runs one row for the
/// token it emitted last plus one per candidate, emitting the accepted prefix plus
/// the model's own next token.
#[test]
fn the_round_rule_is_the_hand_computed_one() {
    // A prompt of two ids, then six generated; the model's choices run
    // [5, 6, 5, 6, 5, 6] and stop on 7.
    let text = vec![9usize, 9];
    let generated = vec![5usize, 6, 5, 6, 5, 6];
    let mut choices = generated.clone();
    choices.push(7);
    let mut ids = text.clone();
    ids.extend_from_slice(&generated);
    let drafter = Drafter {
        ids: ids.clone(),
        prefix: text.len(),
        min_match: 1,
    };
    // One id of context has nothing before it; two ids repeat the last one.
    assert_eq!(drafter.at(1), NOTHING);
    assert_eq!(drafter.at(2), Proposal { start: 1, len: 1 });
    // Context [9, 9, 5, 6, 5]: the latest single id that recurs is the 5 at 2, so
    // the draft copies from 3 to the context's own end.
    assert_eq!(drafter.at(5), Proposal { start: 3, len: 2 });
    // Context [9, 9, 5, 6, 5, 6, 5, 6]: [5, 6, 5, 6] recurs at 2, and the latest
    // occurrence (at 6) breaks the tie, so the draft copies the two ids after it.
    assert_eq!(drafter.at(8), Proposal { start: 6, len: 2 });
    // With a three-id minimum the short match at 5 is not a draft.
    let strict = Drafter {
        ids,
        prefix: text.len(),
        min_match: 3,
    };
    assert_eq!(strict.at(5), NOTHING);
    assert_eq!(strict.at(8), Proposal { start: 6, len: 2 });

    let curve = measure(&drafter, &choices);
    // The prefill produced the first choice, so the walk emits six over four
    // rounds, and each round's pass runs one row for the token it emitted last
    // plus one per candidate it drafted.
    assert_eq!(curve.rounds(), 4);
    assert_eq!(curve.emitted, 6);
    assert_eq!(curve.lengths, vec![0, 0, 2, 0]);
    assert_eq!(curve.accepts, vec![0, 0, 2, 0]);
    assert_eq!(curve.passes, vec![1, 1, 3, 1]);
    assert_eq!(curve.coverage(), 1.0 / 4.0);
    // Depth tallies: one draft at each of the first two depths, both of them the
    // model's own choice; the last round's two-id draft is capped away at the
    // stream's end, so it checks nothing.
    assert_eq!(
        curve.depths[0],
        Tally {
            proposed: 1,
            matched: 1
        }
    );
    assert_eq!(
        curve.depths[1],
        Tally {
            proposed: 1,
            matched: 1
        }
    );
    assert_eq!(curve.depths[2].proposed, 0);
    assert_eq!(
        curve.conditional(),
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(Curve::yield_at_full_depth(&curve.conditional()), 3.0);
    assert_eq!(curve.tokens_per_round(), 6.0 / 4.0);
    // The yield at a constant acceptance is the geometric series: eight
    // candidates, nine rows, up to nine tokens.
    assert!((yield_at(0.0, 8) - 1.0).abs() < 1e-12);
    assert!((yield_at(1.0, 8) - 9.0).abs() < 1e-12);
    assert!((yield_at(0.5, 8) - 1.996_093_75).abs() < 1e-9);
    assert!((yield_at(0.5, 7) - 1.992_187_5).abs() < 1e-9);
    // Break-even at a 1.75x pass needs an acceptance just under 0.43 at eight
    // candidates and 0.429 at seven; a 1.8x pass moves it to 0.445, and a 1.3x win
    // on top of 1.75 needs 0.563.
    let be = acceptance_for(1.75, 8).expect("break-even is reachable");
    assert!((be - 0.4289).abs() < 0.001, "break-even {be}");
    let seven = acceptance_for(1.75, 7).expect("break-even at seven is reachable");
    assert!(
        (seven - 0.4292).abs() < 0.001,
        "break-even at seven {seven}"
    );
    let wide = acceptance_for(1.8, 8).expect("a 1.8x pass is reachable");
    assert!((wide - 0.4448).abs() < 0.001, "break-even at 1.8 {wide}");
    let win = acceptance_for(1.3 * 1.75, 8).expect("the win target is reachable");
    assert!((win - 0.5629).abs() < 0.001, "win acceptance {win}");
    assert_eq!(acceptance_for(10.0, 8), None);
}

/// The byte figure's own rule: a round pays one weight-stream read, a round past
/// [`DEPTH`] rows pays two — the grouped multiply takes eight rows a group, and the
/// pinned clip's walked stream never reaches a full-depth round.
#[test]
fn a_full_depth_round_reads_the_weight_stream_twice() {
    let curve = Curve {
        min_match: 1,
        depths: [Tally::default(); DEPTH],
        lengths: vec![0, 1, DEPTH],
        accepts: vec![0, 1, DEPTH],
        passes: vec![1, 2, DEPTH + 1],
        emitted: 3,
        walk: (f64::NAN, f64::NAN, f64::NAN),
    };
    // One read for the one-row and two-row rounds, two for the one past eight rows.
    assert_eq!(curve.stream_reads_per_token(), 4.0 / 3.0);
    let deep = Curve {
        lengths: vec![DEPTH],
        accepts: vec![DEPTH],
        passes: vec![DEPTH + 1],
        emitted: 1,
        ..curve
    };
    assert_eq!(deep.stream_reads_per_token(), 2.0);
}
