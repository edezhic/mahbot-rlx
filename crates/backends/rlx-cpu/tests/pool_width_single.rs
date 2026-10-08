// RLX — versatile ML compiler and runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A one-worker pool runs RLX's parallel entry points inline — the path a
//! process with no usable pool (Rayon could not build one) also takes.
//!
//! Only two tests in this binary: the pool is process-wide and created once, so
//! the width must be requested before anything else creates it — both ask for 1
//! before touching the pool.

#[test]
fn a_single_worker_runs_the_parallel_entry_points_inline() {
    rlx_ir::env::set("RLX_WORKERS", "1");
    assert_eq!(rlx_cpu::pool::num_threads(), 1);
    assert!(!rlx_cpu::pool::should_parallelize(1 << 20));

    // `par_for`'s single-chunk path is covered by `pool.rs`'s unit tests; this
    // is `par_range`'s own `num_threads() <= 1` guard.
    let seen = std::sync::atomic::AtomicUsize::new(0);
    rlx_cpu::pool::par_range(4, |_| {
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(seen.load(std::sync::atomic::Ordering::Relaxed), 4);
}

/// The two half-weight GEMMs dispatch straight from the thunk hot path, and at
/// width 1 they run their rows inline — the arm no other test reaches, since
/// the library's own GEMM tests run at whatever width the machine has. Both
/// shapes are covered (`m == 1` is the split-K shape a single worker folds into
/// one row) and checked against the f32 GEMM over the same widened weights.
#[test]
fn a_single_worker_runs_the_half_gemms_inline() {
    use rlx_cpu::blas::{sgemm, sgemm_bf16_rhs, sgemm_f16_rhs};
    fn worst(want: &[f32], got: &[f32]) -> f32 {
        want.iter()
            .zip(got)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
    rlx_ir::env::set("RLX_WORKERS", "1");
    assert_eq!(rlx_cpu::pool::num_threads(), 1);
    // `m == 1` is the split-K shape, which a single worker must fold into one
    // row; `m > 1` is the row-per-chunk shape.
    for &(m, k, n) in &[(1usize, 128usize, 64usize), (3, 96, 40)] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let bf16: Vec<u16> = w.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
        let f16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let mut want = vec![0f32; m * n];
        let mut got = vec![0f32; m * n];

        let widened: Vec<f32> = bf16
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        sgemm(&a, &widened, &mut want, m, k, n);
        sgemm_bf16_rhs(&a, &bf16, &mut got, m, k, n);
        assert!(
            worst(&want, &got) < 1e-4,
            "bf16 gemm {m}x{k}x{n} at width 1: {:?} vs {:?}",
            got,
            want
        );

        let widened: Vec<f32> = f16
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        sgemm(&a, &widened, &mut want, m, k, n);
        sgemm_f16_rhs(&a, &f16, &mut got, m, k, n);
        assert!(
            worst(&want, &got) < 1e-4,
            "f16 gemm {m}x{k}x{n} at width 1: {:?} vs {:?}",
            got,
            want
        );
    }
}
