// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake decision: enrolment prototype, negative calibration and the
//! rolling/adaptive score.
//!
//! Enrolment builds a prototype from the owner's windows, a negative pool fixes
//! the cosine floor, and the rolling decision scores each window against that
//! prototype with this crate's own encoder ([`crate::listen`]). The cosine
//! scale is absolute for a given model, but the floor is derived from the
//! negative pool measured against *that* prototype — thresholds do not transfer
//! between models, which is why the bench reports the ones it derived.

/// Maximum cosine floor — the upper clamp bound for the derived floor.
///
/// Two reasons this is 0.75: it keeps the anchored fire factor at
/// (POS_MATCH_COSINE_ANCHOR − 0.75)/(1 − 0.75) = 0.352, so the static rolling
/// threshold is 3 × 0.352 = 1.056 — above one stride's maximum by the margin
/// [`MIN_STATIC_BAR`] asks for; and it leaves a soft band for reset/adaptive
/// resolution. The floor below this bound is what the negative pool derives; the
/// bound is what the benchmark's pool saturates, so it is the operating floor
/// there, and lowering it is what lets the anchor sit this low.
pub const MAX_SOFT_FLOOR: f32 = 0.75;

/// Fallback cosine floor when the enrolment collected no usable negatives, and
/// the lower clamp bound of the derived floor.
pub const DEFAULT_SOFT_FLOOR: f32 = 0.55;

/// Standard-normal 99th-percentile z-score.
///
/// The floor is the parametric p99 estimate `neg_mean + NEG_TAIL_Z × neg_std`.
/// Parametric on purpose: the empirical nearest-rank p99 equals the single
/// maximum for pools of n ≤ 51 (live enrolments have n ≈ 11), which is
/// outlier-driven and would compress the positive soft range; mean/std are
/// stable sufficient statistics at any pool size.
pub const NEG_TAIL_Z: f32 = 2.33;

/// The static firing point in RAW cosine space: with the anchored factor binding,
/// a rolling window fires when the mean cosine over its window reaches this.
///
/// The calibrated point sits at the lowest value the static bar's margin admits
/// (see [`MIN_STATIC_BAR`]). The wake bench measures recall over the enrolled
/// voice and ten cloned verification voices and the false accepts over the pinned
/// real-audio corpus; the record carries the figures.
pub const POS_MATCH_COSINE_ANCHOR: f32 = 0.838;

/// Cap on the per-frame soft-score factor; the effective static fire factor is
/// its min with the anchor-derived value ([`Calibration::fire_soft`]). At the
/// calibrated clamp the anchored factor is below it, so it binds only for an
/// enrolment whose derived floor sits much lower than the clamp's upper bound.
pub const MATCH_THRESHOLD_FACTOR: f32 = 0.55;

/// The no-match reset threshold is this ratio × the fire factor. The calibrated
/// value narrows the frame-accumulation band to just under the firing line, which
/// is what keeps false accepts down while the bar sits this low.
pub const RESET_FIRE_RATIO: f32 = 0.95;

/// The lowest static bar the module accepts, in rolling-sum space. One stride's
/// maximum soft score is 1.0, so a bar at or near it is a defect rather than an
/// invariant: the margin is what makes "a single stride can never fire" hold for
/// every floor an enrolment can derive, not only for the fixture's.
pub const MIN_STATIC_BAR: f32 = 1.05;

/// The numbers that set the firing bar: the floor clamp, the negative-tail
/// multiplier, the raw-cosine anchor, the fire-factor cap, the reset ratio, and the
/// adaptive arm's k and ceiling.
///
/// The decision's shape is not addressable here and cannot be reached through this
/// set: the rolling window and its length ([`ROLLING_WINDOW_N`]), the adaptive arm's
/// window and bootstrap count ([`ADAPTIVE_WINDOW_N`],
/// [`ADAPTIVE_BOOTSTRAP_FRAMES`]), the anti-prototype veto, the six-stride segment
/// limit ([`WAKE_WORD_MAX_SEGMENT_STEPS`]) and the per-fire cooldown stay exactly
/// as this module and the bench's stream declare them. An operating point is this
/// set; [`Thresholds::DEFAULT`] is the shipped one, built from the named constants
/// of this module, and the bench may price another by overriding the set without
/// changing any rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Lower clamp bound of the derived floor, and the fallback floor of an
    /// enrolment without usable negatives.
    pub floor_min: f32,
    /// Upper clamp bound of the derived floor.
    pub floor_max: f32,
    /// Multiplier on the negative pool's standard deviation.
    pub tail_z: f32,
    /// Cap on the per-frame soft-score factor.
    pub fire_factor_cap: f32,
    /// Raw-cosine firing anchor: with the anchor binding, a rolling window fires
    /// when its mean cosine reaches this.
    pub anchor: f32,
    /// No-match reset line, as a fraction of the fire factor.
    pub reset_ratio: f32,
    /// Adaptive arm: multiplier on the score window's standard deviation.
    pub adaptive_k: f32,
    /// Adaptive arm: ceiling in rolling-sum space.
    pub adaptive_ceiling: f32,
}

impl Thresholds {
    /// The set's numbers in the vocabulary `--wake-thresholds` reads: the name, a
    /// reader and a writer. [`Display`](core::fmt::Display) and the bench's parser
    /// both walk this list, so a number cannot be printed without being readable
    /// back. [`Thresholds::check`] lists the same numbers because a `const fn`
    /// cannot call through these functions.
    pub(crate) const FIELDS: &'static [(&'static str, fn(&Self) -> f32, fn(&mut Self, f32))] = &[
        ("floor_min", |s| s.floor_min, |s, v| s.floor_min = v),
        ("floor_max", |s| s.floor_max, |s, v| s.floor_max = v),
        ("tail_z", |s| s.tail_z, |s, v| s.tail_z = v),
        (
            "fire_factor_cap",
            |s| s.fire_factor_cap,
            |s, v| s.fire_factor_cap = v,
        ),
        ("anchor", |s| s.anchor, |s, v| s.anchor = v),
        ("reset_ratio", |s| s.reset_ratio, |s, v| s.reset_ratio = v),
        ("adaptive_k", |s| s.adaptive_k, |s, v| s.adaptive_k = v),
        (
            "adaptive_ceiling",
            |s| s.adaptive_ceiling,
            |s, v| s.adaptive_ceiling = v,
        ),
    ];

    /// The shipped point: this module's named constants and nothing else.
    pub const DEFAULT: Self = Self {
        floor_min: DEFAULT_SOFT_FLOOR,
        floor_max: MAX_SOFT_FLOOR,
        tail_z: NEG_TAIL_Z,
        fire_factor_cap: MATCH_THRESHOLD_FACTOR,
        anchor: POS_MATCH_COSINE_ANCHOR,
        reset_ratio: RESET_FIRE_RATIO,
        adaptive_k: ADAPTIVE_K_DEFAULT,
        adaptive_ceiling: ADAPTIVE_CEILING,
    };

    /// The per-frame static fire factor at a given floor — the const form of
    /// [`Calibration::fire_soft`], so the invariant guard and the scoring path
    /// read one derivation rather than two.
    const fn fire_factor_at(&self, floor: f32) -> f32 {
        let anchored = (self.anchor - floor) / (1.0 - floor);
        if anchored < self.fire_factor_cap {
            anchored
        } else {
            self.fire_factor_cap
        }
    }

    /// The lowest static bar this set can derive. The fire factor falls as the
    /// floor rises ([`Calibration::fire_soft`]), so the highest floor is the
    /// worst case; this is the number the single-stride invariant is checked
    /// against.
    #[must_use]
    pub const fn lowest_static_bar(&self) -> f32 {
        self.fire_factor_at(self.floor_max) * ROLLING_WINDOW_N as f32
    }

    /// The first invariant this set breaks, in const form: the build-time guard,
    /// the bench's override parser and [`validate`](Self::validate) read these
    /// same rules, so a point cannot pass one and fail another.
    const fn check(&self) -> Result<(), &'static str> {
        let all = [
            self.floor_min,
            self.floor_max,
            self.tail_z,
            self.fire_factor_cap,
            self.anchor,
            self.reset_ratio,
            self.adaptive_k,
            self.adaptive_ceiling,
        ];
        let mut i = 0;
        while i < all.len() {
            if !all[i].is_finite() {
                return Err("a number in the set is not finite");
            }
            i += 1;
        }
        if self.floor_min >= self.floor_max {
            return Err("floor_min must be below floor_max");
        }
        if self.floor_min <= -1.0 || self.floor_max >= 1.0 {
            return Err("the floor clamp must lie inside (−1, 1)");
        }
        if self.anchor <= self.floor_max || self.anchor > 1.0 {
            return Err("anchor must lie in (floor_max, 1]");
        }
        if self.fire_factor_cap <= 0.0 || self.fire_factor_cap > 1.0 {
            return Err("fire_factor_cap must lie in (0, 1]");
        }
        if self.reset_ratio <= 0.0 || self.reset_ratio > 1.0 {
            return Err("reset_ratio must lie in (0, 1]");
        }
        if self.adaptive_k < 0.0 || self.adaptive_ceiling <= 0.0 {
            return Err("the adaptive arm needs k ≥ 0 and a positive ceiling");
        }
        // The arm clamps to `[static bar, ceiling]`, so a ceiling below the
        // highest bar the cap can derive panics inside that clamp. The cap bounds
        // every enrolment's bar, which is why the check needs no enrolment.
        if self.adaptive_ceiling < ROLLING_WINDOW_N as f32 * self.fire_factor_cap {
            return Err("adaptive_ceiling is below the highest static bar the cap can derive");
        }
        if self.lowest_static_bar() < MIN_STATIC_BAR {
            return Err(
                "the lowest static bar is below MIN_STATIC_BAR — a single stride could fire",
            );
        }
        Ok(())
    }

    /// Refuse a set the module's invariants cannot hold — a number that is not
    /// finite, a floor clamp whose bounds cross, an anchor the clamp leaves no
    /// room for, an adaptive ceiling below the bar it has to clamp, or a static
    /// bar a single scoring stride could clear — so an unusable point is refused
    /// at start-up rather than panicking inside `f32::clamp` mid-run.
    ///
    /// # Errors
    /// The first invariant the set breaks, followed by the set itself.
    pub fn validate(&self) -> Result<(), String> {
        self.check()
            .map_err(|rule| format!("{rule} — the set was {self}"))
    }
}

impl core::fmt::Display for Thresholds {
    /// The set as the comma-separated `name=value` pairs the bench's
    /// `--wake-thresholds` reads, at the shortest precision that parses back to
    /// the same numbers.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (index, (name, read, _)) in Self::FIELDS.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{name}={}", read(self))?;
        }
        Ok(())
    }
}

// The shipped constants are held to the same rules the override parser applies,
// where the mistake is cheapest to see and impossible to ship: at build time.
const _: () = match (Thresholds::DEFAULT).check() {
    Ok(()) => (),
    Err(why) => panic!("{}", why),
};

/// Enrolment consistency: minimum cosine between an utterance embedding and
/// the centroid, and the fraction of utterances that must pass.
pub const ENROLLMENT_CONSISTENCY_MIN_SIMILARITY: f32 = 0.70;
pub const ENROLLMENT_CONSISTENCY_MIN_FRACTION: f32 = 0.7;

/// Minimum number of qualified utterances for a valid enrolment.
pub const MIN_ENROLLMENT_UTTERANCES: usize = 5;

/// Frames in the rolling score window; one frame is one scoring stride
/// (≈ 160 ms), so three cover ≈ 0.48 s.
pub const ROLLING_WINDOW_N: usize = 3;

/// Adaptive-threshold window length (~2.4 s at one score per stride).
pub const ADAPTIVE_WINDOW_N: usize = 15;
/// Multiplier on the window's standard deviation.
pub const ADAPTIVE_K_DEFAULT: f32 = 2.5;
/// The adaptive threshold never exceeds this rolling-sum value.
pub const ADAPTIVE_CEILING: f32 = 2.60;
/// Frames scored with the static threshold while the window fills.
pub const ADAPTIVE_BOOTSTRAP_FRAMES: usize = 5;

/// Only the first six scoring strides of a speech segment may fire; past that
/// position a match is inert until the segment ends.
pub const WAKE_WORD_MAX_SEGMENT_STEPS: usize = 6;

/// L2-normalize a vector in place (no-op for zero/empty vectors).
pub fn l2_normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// L2-normalized copy.
#[must_use]
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    l2_normalize_in_place(&mut out);
    out
}

/// Cosine similarity of two same-length vectors.
///
/// The embeddings this module compares are already L2-normalized, so the dot
/// product is the cosine; the norm is recomputed anyway so a caller that hands
/// in an unnormalized vector gets a cosine rather than a scaled dot.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / d
}

/// L2-normalized centroid (mean, then normalize) of the utterance embeddings.
///
/// Float operation order is load-bearing for calibration reproducibility:
/// `+=` accumulate, scale by `1.0 / len`, then normalize.
///
/// # Errors
/// Returns the offending embedding's length when an utterance's width differs
/// from the first one's.
pub fn normalized_centroid(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, usize> {
    let Some(dim) = embeddings.first().map(Vec::len) else {
        return Err(0);
    };
    let mut centroid = vec![0.0f32; dim];
    for emb in embeddings {
        if emb.len() != dim {
            return Err(emb.len());
        }
        for (c, e) in centroid.iter_mut().zip(emb) {
            *c += e;
        }
    }
    let inv = 1.0 / embeddings.len() as f32;
    for c in &mut centroid {
        *c *= inv;
    }
    l2_normalize_in_place(&mut centroid);
    Ok(centroid)
}

/// Negative-sample calibration: the cosine distribution of negative material
/// against the prototype, measured at enrolment time.
///
/// `Calibration::default()` is an enrolment with no usable negatives:
/// [`soft_floor`](Self::soft_floor) = [`DEFAULT_SOFT_FLOOR`] and
/// [`fire_soft`](Self::fire_soft) = [`MATCH_THRESHOLD_FACTOR`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Calibration {
    /// Mean cosine of negative samples vs the prototype.
    pub neg_mean: f32,
    /// Standard deviation of the negative cosine distribution.
    pub neg_std: f32,
    /// Raw (unclamped) 99th percentile — diagnostic only; scoring derives the
    /// floor parametrically from mean/std.
    pub neg_p99: f32,
    /// Number of negative windows scored.
    pub n_negatives: usize,
}

impl Calibration {
    /// Cosine floor: the parametric p99 estimate `neg_mean + tail_z × neg_std`,
    /// clamped to the set's floor bounds. Scores below the floor map to ~0.
    #[must_use]
    pub fn soft_floor(&self, thresholds: &Thresholds) -> f32 {
        if self.n_negatives == 0 || !self.neg_mean.is_finite() || !self.neg_std.is_finite() {
            return thresholds.floor_min;
        }
        let parametric = self.neg_mean + thresholds.tail_z * self.neg_std;
        if !parametric.is_finite() {
            return thresholds.floor_min;
        }
        parametric.clamp(thresholds.floor_min, thresholds.floor_max)
    }

    /// Per-frame soft-score factor for the static detection threshold — the min
    /// of the set's cap and the anchored cap.
    #[must_use]
    pub fn fire_soft(&self, thresholds: &Thresholds) -> f32 {
        thresholds.fire_factor_at(self.soft_floor(thresholds))
    }

    /// Per-frame reset threshold in soft space.
    #[must_use]
    pub fn reset_soft(&self, thresholds: &Thresholds) -> f32 {
        thresholds.reset_ratio * self.fire_soft(thresholds)
    }

    /// Map a raw cosine into soft-score space: linear from `soft_floor()` → 0
    /// to 1.0 → 1.
    #[must_use]
    pub fn soft_score(&self, cosine: f32, thresholds: &Thresholds) -> f32 {
        let floor = self.soft_floor(thresholds);
        ((cosine - floor) / (1.0 - floor)).clamp(0.0, 1.0)
    }
}

/// The enrolled wake word: prototype, calibration and anti-prototypes.
///
/// The prototype is the L2-normalized mean of the accepted window embeddings.
/// Scoring is cosine similarity against it, mapped through
/// [`Calibration::soft_score`], with the anti-prototype veto below.
#[derive(Debug, Clone)]
pub struct Enrollment {
    /// Normalized wake phrase (lowercased, trimmed).
    pub phrase: String,
    /// L2-normalized prototype embedding.
    pub prototype: Vec<f32>,
    /// Number of utterances that went into the prototype.
    pub utterance_count: usize,
    /// Negative-sample calibration stats.
    pub calibration: Calibration,
    /// Up to [`MAX_NEGATIVE_PROTOTYPES`] L2-normalized cluster centroids of the
    /// negative-sample embeddings.
    pub negative_prototypes: Vec<Vec<f32>>,
    /// The threshold set this enrolment is scored with.
    pub thresholds: Thresholds,
}

/// Maximum number of anti-prototype centroids distilled from the negative pool.
pub const MAX_NEGATIVE_PROTOTYPES: usize = 8;

impl Enrollment {
    /// Build an enrolment from per-utterance window embeddings (already
    /// L2-normalized) and the negative pool, or `None` when fewer than
    /// [`MIN_ENROLLMENT_UTTERANCES`] utterances are given.
    #[must_use]
    pub fn build(
        phrase: String,
        utterance_embeddings: &[Vec<f32>],
        calibration: Calibration,
        negative_embeddings: &[Vec<f32>],
    ) -> Option<Self> {
        if utterance_embeddings.len() < MIN_ENROLLMENT_UTTERANCES {
            return None;
        }
        let prototype = normalized_centroid(utterance_embeddings).ok()?;
        Some(Self {
            phrase,
            prototype,
            utterance_count: utterance_embeddings.len(),
            calibration,
            negative_prototypes: distill_negative_prototypes(negative_embeddings),
            thresholds: Thresholds::DEFAULT,
        })
    }

    /// Score this enrolment with another threshold set — the bench's measurement
    /// aid; the product reads [`Thresholds::DEFAULT`].
    ///
    /// # Errors
    /// The rule the set breaks (see [`Thresholds::validate`]): a point that could
    /// panic inside `f32::clamp` is refused before a stream scores with it.
    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Result<Self, String> {
        thresholds.validate()?;
        self.thresholds = thresholds;
        Ok(self)
    }

    /// Cosine of a window embedding against the prototype.
    #[must_use]
    pub fn positive_cosine(&self, embedding: &[f32]) -> f32 {
        cosine(embedding, &self.prototype)
    }

    /// Maximum cosine against the anti-prototypes (0.0 when there are none).
    #[must_use]
    pub fn max_negative_cosine(&self, embedding: &[f32]) -> f32 {
        self.negative_prototypes
            .iter()
            .map(|a| cosine(embedding, a))
            .fold(0.0_f32, f32::max)
    }

    /// Soft score in `[0, 1]` for a window embedding.
    ///
    /// A window closer to any anti-prototype than to the wake-word prototype is
    /// rejected outright (score 0): the confusable near-miss resembles both, and
    /// the discriminative margin lives in which prototype is closer. Otherwise
    /// the positive cosine goes through the calibration floor.
    #[must_use]
    pub fn soft_score(&self, embedding: &[f32]) -> f32 {
        let pos = self.positive_cosine(embedding);
        let neg = self.max_negative_cosine(embedding);
        if !self.negative_prototypes.is_empty() && neg > pos {
            return 0.0;
        }
        self.calibration.soft_score(pos, &self.thresholds)
    }

    /// The static rolling-sum firing threshold and the per-frame reset line,
    /// both derived from this enrolment's calibration.
    #[must_use]
    pub fn geometry(&self) -> DetectionGeometry {
        DetectionGeometry::for_calibration(&self.calibration, &self.thresholds)
    }
}

/// Distill the negative pool into up to [`MAX_NEGATIVE_PROTOTYPES`]
/// L2-normalized anti-prototypes via farthest-point (max-min) sampling, so the
/// small set covers the full negative manifold rather than its densest region.
#[must_use]
pub fn distill_negative_prototypes(negatives: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if negatives.is_empty() {
        return Vec::new();
    }
    let k = negatives.len().min(MAX_NEGATIVE_PROTOTYPES);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    chosen.push(0);
    while chosen.len() < k {
        let mut best_idx = None;
        let mut best_sim = f32::MAX;
        for (i, n) in negatives.iter().enumerate() {
            if chosen.contains(&i) {
                continue;
            }
            let min_sim = chosen
                .iter()
                .map(|&c| cosine(n, &negatives[c]))
                .fold(f32::MAX, f32::min);
            if min_sim < best_sim {
                best_sim = min_sim;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => chosen.push(i),
            None => break,
        }
    }
    chosen
        .into_iter()
        .map(|i| l2_normalize(&negatives[i]))
        .collect()
}

/// Compute calibration statistics from negative embeddings (already
/// L2-normalized) against the prototype.
///
/// The scoring floor is derived parametrically from mean/std at scoring time;
/// the raw 99th percentile is stored for diagnostics only.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn calibrate_negatives(prototype: &[f32], negatives: &[Vec<f32>]) -> Calibration {
    if negatives.is_empty() {
        return Calibration::default();
    }
    let mut cosines: Vec<f32> = negatives.iter().map(|n| cosine(n, prototype)).collect();
    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = cosines.len();
    let mean = cosines.iter().sum::<f32>() / n as f32;
    let var = cosines.iter().map(|c| (c - mean) * (c - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    let p99 = if n >= 4 {
        let idx = ((n - 1) as f32 * 0.99).round() as usize;
        cosines[idx.min(n - 1)]
    } else {
        (mean + 2.0 * std).min(cosines[n - 1])
    };
    Calibration {
        neg_mean: mean,
        neg_std: std,
        neg_p99: p99,
        n_negatives: n,
    }
}

/// Static detection geometry co-derived from a calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionGeometry {
    /// Rolling-sum firing threshold.
    pub static_threshold: f32,
    /// Per-frame reset line.
    pub reset: f32,
}

impl DetectionGeometry {
    #[must_use]
    pub fn for_calibration(calibration: &Calibration, thresholds: &Thresholds) -> Self {
        Self {
            static_threshold: (ROLLING_WINDOW_N as f32) * calibration.fire_soft(thresholds),
            reset: calibration.reset_soft(thresholds),
        }
    }
}

/// Process one per-frame soft score through the rolling window.
///
/// Returns `(fired, rolling_sum)`. A score below `geometry.reset` clears the
/// window entirely (no slow accumulation from noise); otherwise it is appended
/// and the rolling sum is compared with `threshold` (the adaptive one when it
/// is available, the static one otherwise). The caller owns the window.
pub fn process_score(
    total_score: f32,
    score_window: &mut Vec<f32>,
    geometry: DetectionGeometry,
    threshold: f32,
) -> (bool, f32) {
    if total_score < geometry.reset {
        score_window.clear();
        return (false, 0.0);
    }
    score_window.push(total_score);
    while score_window.len() > ROLLING_WINDOW_N {
        score_window.remove(0);
    }
    let rolling_sum: f32 = score_window.iter().sum();
    (rolling_sum >= threshold, rolling_sum)
}

/// Rolling mean/std of recent per-frame soft scores, feeding the adaptive
/// threshold `(mean + k × std) × ROLLING_WINDOW_N`, clamped to
/// `[harbor, adaptive_ceiling]` in rolling-sum space.
///
/// During the first [`ADAPTIVE_BOOTSTRAP_FRAMES`] frames the callers use the
/// static threshold: [`feed`](Self::feed) returns `None` and
/// [`peek`](Self::peek) returns `None` until the counter is exhausted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdaptiveThreshold {
    scores: Vec<f32>,
    sum: f32,
    sum_sq: f32,
    bootstrap_count: usize,
}

impl AdaptiveThreshold {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A state that has already exited bootstrap, seeded with near-silence
    /// scores so the adaptive path is active from the first real frame.
    #[must_use]
    pub fn warmed(thresholds: &Thresholds, harbor: f32) -> Self {
        let mut state = Self::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            state.feed(0.033, thresholds, harbor);
        }
        state
    }

    /// Feed a below-reset score and return the adaptive threshold, or `None`
    /// during bootstrap.
    pub fn feed(&mut self, score: f32, thresholds: &Thresholds, harbor: f32) -> Option<f32> {
        if self.scores.len() >= ADAPTIVE_WINDOW_N {
            let oldest = self.scores.remove(0);
            self.sum -= oldest;
            self.sum_sq -= oldest * oldest;
        }
        self.scores.push(score);
        self.sum += score;
        self.sum_sq += score * score;
        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES {
            self.bootstrap_count += 1;
            return None;
        }
        Some(self.compute_threshold(thresholds, harbor))
    }

    /// The current threshold without updating the statistics, or `None` during
    /// bootstrap / when the window is empty.
    #[must_use]
    pub fn peek(&self, thresholds: &Thresholds, harbor: f32) -> Option<f32> {
        if self.bootstrap_count < ADAPTIVE_BOOTSTRAP_FRAMES || self.scores.is_empty() {
            return None;
        }
        Some(self.compute_threshold(thresholds, harbor))
    }

    /// Drop all statistics (segment boundary / re-enrolment).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn compute_threshold(&self, thresholds: &Thresholds, harbor: f32) -> f32 {
        let n = self.scores.len() as f32;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n) - (mean * mean);
        let std = variance.max(0.0).sqrt();
        let adaptive = (mean + thresholds.adaptive_k * std) * ROLLING_WINDOW_N as f32;
        adaptive.clamp(harbor, thresholds.adaptive_ceiling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic unit vector whose cosine against `reference` is `c`.
    fn unit_with_cosine(reference: &[f32], c: f32, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut r: Vec<f32> = (0..reference.len())
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let proj: f32 = reference.iter().zip(&r).map(|(a, b)| a * b).sum();
        for (ri, &ri_ref) in r.iter_mut().zip(reference) {
            *ri -= proj * ri_ref;
        }
        let ortho = l2_normalize(&r);
        let s = (1.0 - c * c).max(0.0).sqrt();
        let mut v: Vec<f32> = reference
            .iter()
            .zip(&ortho)
            .map(|(&a, &b)| c * a + s * b)
            .collect();
        l2_normalize_in_place(&mut v);
        v
    }

    /// The shipped point: what the constants derive, in one name.
    fn t() -> Thresholds {
        Thresholds::DEFAULT
    }

    #[test]
    fn calibration_floor_maps_negatives_to_zero() {
        let proto = l2_normalize(&(0..1024).map(|i| i as f32).collect::<Vec<_>>());
        let negatives: Vec<Vec<f32>> = (0..20)
            .map(|i| unit_with_cosine(&proto, 0.20 + i as f32 * 0.0125, 1000 + i as u64))
            .collect();
        let calibration = calibrate_negatives(&proto, &negatives);
        assert_eq!(calibration.n_negatives, 20);
        // The parametric estimate sits below the floor clamp's lower bound, which
        // clamps it; every negative therefore maps to soft score 0.
        assert!((calibration.soft_floor(&t()) - DEFAULT_SOFT_FLOOR).abs() < 1e-6);
        for (i, neg) in negatives.iter().enumerate() {
            let c = cosine(neg, &proto);
            assert!(c < DEFAULT_SOFT_FLOOR, "negative {i} at cosine {c}");
            assert_eq!(calibration.soft_score(c, &t()), 0.0);
        }
        // The anchor keeps the static threshold above one stride's maximum.
        let geometry = DetectionGeometry::for_calibration(&calibration, &t());
        assert!(geometry.static_threshold >= MIN_STATIC_BAR);
        assert!(geometry.reset < geometry.static_threshold);
    }

    #[test]
    fn the_shipped_point_validates_and_moves_with_its_anchor() {
        t().validate().expect("the shipped point is valid");
        // The worst case is the clamp's upper bound: there the anchored factor is
        // smallest, so it is below the cap and the bar is the anchor's. The guard
        // and the scoring path must be one derivation, so the checked number is
        // asserted against the geometry a stream is actually built from.
        let at_upper_bound = Calibration {
            neg_mean: MAX_SOFT_FLOOR,
            neg_std: 0.0,
            neg_p99: MAX_SOFT_FLOOR,
            n_negatives: 1,
        };
        let worst = DetectionGeometry::for_calibration(&at_upper_bound, &t()).static_threshold;
        assert_eq!(t().lowest_static_bar(), worst);
        assert!(
            worst >= MIN_STATIC_BAR,
            "the shipped bar {worst} is inside the one-stride margin"
        );
        // Raising the anchor raises the bar in step.
        let mut raised = t();
        raised.anchor += 0.04;
        let raised_bar =
            DetectionGeometry::for_calibration(&at_upper_bound, &raised).static_threshold;
        assert!(raised_bar > worst, "the anchor is the bar at the clamp");
        // A bar one stride could clear is refused rather than measured, and it is
        // the margin rule that refuses it.
        let mut short = t();
        short.anchor = MAX_SOFT_FLOOR + 0.01;
        let err = short.validate().expect_err("a one-stride bar");
        assert!(err.contains("single stride"), "{err}");
    }

    #[test]
    fn a_crossed_floor_clamp_is_refused() {
        let mut set = t();
        set.floor_min = set.floor_max;
        let err = set.validate().expect_err("crossed bounds");
        assert!(err.contains("floor_min must be below floor_max"), "{err}");
        // A clamp reaching outside (−1, 1) cannot derive a soft score at all. The
        // rule is strict, so a bound exactly on the edge is outside it too.
        let mut set = t();
        set.floor_min = -1.5;
        let err = set.validate().expect_err("an out-of-range floor");
        assert!(err.contains("inside (−1, 1)"), "{err}");
        let mut set = t();
        set.floor_min = -1.0;
        let err = set.validate().expect_err("a bound on the interval's edge");
        assert!(err.contains("inside (−1, 1)"), "{err}");
    }

    #[test]
    fn a_non_finite_number_is_refused() {
        let mut set = t();
        set.anchor = f32::NAN;
        let err = set.validate().expect_err("NaN");
        assert!(err.contains("not finite"), "{err}");
        let mut set = t();
        set.tail_z = f32::INFINITY;
        assert!(set.validate().is_err(), "an infinite tail");
    }

    /// One vocabulary: every number the override parser can write is printed by
    /// `Display` and refused by the guard when it is not finite, so a number added
    /// to the set cannot be silently unprintable or unchecked.
    #[test]
    fn every_number_in_the_vocabulary_is_guarded_and_printed() {
        let printed = t().to_string();
        for (name, _, write) in Thresholds::FIELDS {
            assert!(
                printed.contains(&format!("{name}=")),
                "{name} is not printed: {printed}"
            );
            let mut set = t();
            write(&mut set, f32::NAN);
            let err = set.validate().expect_err(name);
            assert!(err.contains("not finite"), "{name} is not guarded: {err}");
        }
    }

    #[test]
    fn a_ceiling_below_the_bar_it_clamps_is_refused() {
        // The adaptive arm clamps to [static bar, ceiling]; a ceiling under the
        // bar would panic there rather than fail as a measurement.
        let mut set = t();
        set.adaptive_ceiling = set.fire_factor_cap * ROLLING_WINDOW_N as f32 - 1e-3;
        let err = set.validate().expect_err("ceiling under the cap's bar");
        assert!(err.contains("adaptive_ceiling is below"), "{err}");
    }

    #[test]
    fn distillation_spreads_selection() {
        let proto = l2_normalize(&[1.0f32; 32]);
        let negatives: Vec<Vec<f32>> = (0..20)
            .map(|i| unit_with_cosine(&proto, -0.2 + i as f32 * 0.05, 200 + i as u64))
            .collect();
        let distilled = distill_negative_prototypes(&negatives);
        assert_eq!(distilled.len(), MAX_NEGATIVE_PROTOTYPES);
        // Farthest-point sampling starts at index 0 and never repeats.
        assert_eq!(distilled[0], negatives[0]);
        for pair in distilled.windows(2) {
            assert!(cosine(&pair[0], &pair[1]) < 0.999);
        }
    }

    #[test]
    fn enrollment_requires_minimum_utterances() {
        let embedding = l2_normalize(&[0.5f32; 16]);
        assert!(
            Enrollment::build(
                "hey".into(),
                &vec![embedding.clone(); 4],
                Calibration::default(),
                &[]
            )
            .is_none()
        );
        let enrollment = Enrollment::build(
            "hey".into(),
            &vec![embedding; MIN_ENROLLMENT_UTTERANCES],
            Calibration::default(),
            &[],
        )
        .expect("five identical utterances enrol");
        assert!((cosine(&enrollment.prototype, &l2_normalize(&[1.0f32; 16])) - 1.0).abs() < 1e-6);
        assert!((enrollment.soft_score(&enrollment.prototype) - 1.0).abs() < 1e-6);
    }

    /// The set an enrolment scores with is refused where it is installed, in every
    /// build: a crossed clamp would otherwise panic inside `f32::clamp` mid-run.
    #[test]
    fn an_unusable_set_cannot_be_installed_on_an_enrolment() {
        let embedding = l2_normalize(&[0.5f32; 16]);
        let enrollment = Enrollment::build(
            "hey".into(),
            &vec![embedding; MIN_ENROLLMENT_UTTERANCES],
            Calibration::default(),
            &[],
        )
        .expect("five identical utterances enrol");
        assert_eq!(enrollment.thresholds, t());
        let mut crossed = t();
        crossed.floor_min = crossed.floor_max;
        let err = enrollment
            .with_thresholds(crossed)
            .expect_err("a crossed clamp");
        assert!(err.contains("floor_min must be below floor_max"), "{err}");
    }

    #[test]
    fn anti_prototype_veto_rejects_a_closer_negative() {
        let mut a = vec![0.0f32; 8];
        a[0] = 1.0;
        let mut b = vec![0.0f32; 8];
        b[1] = 1.0;
        let probe = b.clone();
        let enrollment = Enrollment {
            phrase: "hey".into(),
            prototype: a,
            utterance_count: 5,
            calibration: Calibration {
                neg_mean: 0.2,
                neg_std: 0.05,
                neg_p99: 0.4,
                n_negatives: 9,
            },
            negative_prototypes: vec![b],
            thresholds: Thresholds::DEFAULT,
        };
        assert_eq!(enrollment.soft_score(&probe), 0.0);
        assert!(enrollment.soft_score(&enrollment.prototype) > 0.9);
    }

    #[test]
    fn rolling_window_resets_below_the_reset_line() {
        let calibration = Calibration {
            neg_mean: 0.4,
            neg_std: 0.05,
            neg_p99: 0.5,
            n_negatives: 12,
        };
        let geometry = DetectionGeometry::for_calibration(&calibration, &t());
        let mut window = Vec::new();
        // Between the reset line and the fire factor: kept, but short of the bar.
        let high = calibration.fire_soft(&t()) * 0.975;
        for _ in 0..ROLLING_WINDOW_N {
            let (fired, _) = process_score(high, &mut window, geometry, geometry.static_threshold);
            assert!(
                !fired,
                "three frames below the static threshold must not fire"
            );
        }
        assert_eq!(window.len(), ROLLING_WINDOW_N);
        let (_, sum) = process_score(
            geometry.reset / 2.0,
            &mut window,
            geometry,
            geometry.static_threshold,
        );
        assert_eq!(sum, 0.0);
        assert!(window.is_empty());
    }

    #[test]
    fn adaptive_threshold_starts_at_the_harbor_and_never_exceeds_the_ceiling() {
        // The harbour is the static bar an enrolment derives, not a fixture.
        let harbor =
            DetectionGeometry::for_calibration(&Calibration::default(), &t()).static_threshold;
        let mut state = AdaptiveThreshold::new();
        for _ in 0..ADAPTIVE_BOOTSTRAP_FRAMES {
            assert_eq!(state.feed(0.01, &t(), harbor), None);
        }
        let warm = state.feed(0.01, &t(), harbor).expect("post-bootstrap");
        assert!((warm - harbor).abs() < 1e-6);
        for _ in 0..ADAPTIVE_WINDOW_N * 3 {
            state.feed(1.0, &t(), harbor);
        }
        assert!(state.peek(&t(), harbor).unwrap() <= ADAPTIVE_CEILING);
        state.reset();
        assert_eq!(state.peek(&t(), harbor), None);
    }

    #[test]
    fn degenerate_calibration_uses_the_documented_fallbacks() {
        let calibration = Calibration::default();
        assert_eq!(calibration.soft_floor(&t()), DEFAULT_SOFT_FLOOR);
        assert_eq!(calibration.fire_soft(&t()), MATCH_THRESHOLD_FACTOR);
        let geometry = DetectionGeometry::for_calibration(&calibration, &t());
        assert!(
            (geometry.static_threshold - ROLLING_WINDOW_N as f32 * MATCH_THRESHOLD_FACTOR).abs()
                < 1e-6
        );
    }
}
