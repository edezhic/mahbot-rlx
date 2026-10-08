// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Auto-tuner — searches for a RuntimeConfig for a model on current hardware.
//!
//! Runs the model with different parameter combinations, measures each, picks the
//! fastest. The swept thresholds and the returned [`TuneResult::config`] are
//! values to install in a *later* process: the kernels read the frozen
//! [`RuntimeConfig::global`], so a config selected here cannot take effect once
//! this process's configuration is frozen. Nothing printed claims to be in
//! effect — only what was measured and the pool width already in effect.
//!
//! ```rust,ignore
//! let best = autotune(&mut || model.run(inputs), &SearchSpace::default(), 5, 20);
//! eprintln!("Config for the next process: {:?}", best.config);
//! ```

use crate::config::RuntimeConfig;
use rlx_ir::Tick;

/// Result of one tuning trial.
#[derive(Debug, Clone)]
pub struct TuneResult {
    pub config: RuntimeConfig,
    pub p50_ms: f64,
    pub min_ms: f64,
}

/// Search space for auto-tuning.
///
/// The pool worker count is not an axis: the process-wide Rayon pool is sized
/// once, before the first CPU dispatch, so a trial cannot change it — sweeping
/// it would report a configuration the engine never applied. The threshold
/// values swept here are the same: they are a candidate for a later process,
/// since the kernels read the frozen [`RuntimeConfig::global`].
pub struct SearchSpace {
    pub par_thresholds: Vec<usize>,
    pub sdpa_thresholds: Vec<usize>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            par_thresholds: vec![10_000, 20_000, 30_000, 50_000],
            sdpa_thresholds: vec![16, 32, 48],
        }
    }
}

/// Auto-tune by running the model with different configs.
///
/// `run_fn` is called for each trial — it should execute one forward pass.
/// `warmup` iterations are run before timing. `trials` are timed.
///
/// The returned config is a candidate for a later process, not a live change:
/// the kernels read the frozen global config, so the thresholds that were swept
/// cannot take effect here.
pub fn autotune<F>(mut run_fn: F, search: &SearchSpace, warmup: usize, trials: usize) -> TuneResult
where
    F: FnMut(),
{
    let mut results: Vec<TuneResult> = Vec::new();
    let base = RuntimeConfig::auto_detect();
    let total = search.par_thresholds.len() * search.sdpa_thresholds.len();
    let mut n = 0;

    // Generate all combinations
    for &par in &search.par_thresholds {
        for &sdpa in &search.sdpa_thresholds {
            n += 1;
            let cfg = RuntimeConfig {
                par_threshold: par,
                sdpa_seq_threshold: sdpa,
                ..base.clone()
            };

            // Record this candidate for the caller. It does *not* reach the
            // kernels: they read the frozen `RuntimeConfig::global`, and nothing
            // in the tree reads `TUNE_CONFIG` (`active_config` has no callers).
            set_global_config(cfg.clone());

            // Warmup
            for _ in 0..warmup {
                run_fn();
            }

            // Measure — direct CNTVCT_EL0 read on Apple Silicon (#66).
            // Sub-microsecond resolution lets the search distinguish
            // configs whose wall-clock times differ by a few ticks.
            let mut times = Vec::with_capacity(trials);
            for _ in 0..trials {
                let t = Tick::now();
                run_fn();
                times.push(Tick::now().elapsed_ms(t));
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = times[trials / 2];
            let min = times[0];

            eprintln!("  candidate {n}/{total}: p50={p50:.2}ms min={min:.2}ms");
            results.push(TuneResult {
                config: cfg,
                p50_ms: p50,
                min_ms: min,
            });
        }
    }

    // Find best by p50
    results.sort_by(|a, b| a.p50_ms.partial_cmp(&b.p50_ms).unwrap());
    let best = results[0].clone();

    // Record the winner for the caller, for the same reason as above.
    set_global_config(best.config.clone());

    eprintln!(
        "[rlx] cpu: best of {total} candidate configs ({trials} runs each): {:.2}ms p50 (pool: {} workers in effect)",
        best.p50_ms,
        crate::pool::num_threads()
    );

    best
}

/// Record `cfg` as the tuned configuration.
///
/// Not the configuration the kernels read: [`RuntimeConfig::global`]'s
/// `OnceLock` cannot be re-set, so this writes `TUNE_CONFIG`, which only
/// [`active_config`] reads — and no in-tree caller does.
fn set_global_config(cfg: RuntimeConfig) {
    TUNE_CONFIG.lock().unwrap().replace(cfg);
}

/// Get the active tuning config (if set), otherwise fall back to global.
pub fn active_config() -> RuntimeConfig {
    if let Some(cfg) = TUNE_CONFIG.lock().unwrap().as_ref() {
        cfg.clone()
    } else {
        RuntimeConfig::global().clone()
    }
}

static TUNE_CONFIG: std::sync::Mutex<Option<RuntimeConfig>> = std::sync::Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_space_default() {
        let ss = SearchSpace::default();
        assert_eq!(ss.par_thresholds.len(), 4);
        assert_eq!(ss.sdpa_thresholds.len(), 3);
    }
}
