// RLX — versatile ML compiler and runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A width requested after the pool exists cannot be applied — and `rlx-cpu`
//! reports it instead of pretending it did; a width installed while the pool is
//! still unclaimed takes effect.
//!
//! Rayon's pool is process-wide and created once, so the width assertions that
//! need an unclaimed pool re-run the test in a child process.

#[test]
fn a_later_width_request_cannot_be_applied() {
    // Stand in for any other Rayon user that creates the shared pool first (the
    // refusal itself is read back from a child process below). The width must
    // stay what it was.
    let pre_existing = rayon::current_num_threads();
    rlx_ir::env::set("RLX_WORKERS", (pre_existing + 1).to_string());
    assert_eq!(rlx_cpu::pool::num_threads(), pre_existing);

    // A width asked for after that is refused too — `ensure_pool` is what the
    // CPU backend calls on every compile, so the request is reported rather
    // than sitting there looking honoured.
    rlx_ir::env::set("RLX_WORKERS", (pre_existing + 2).to_string());
    rlx_cpu::pool::ensure_pool();
    assert_eq!(rlx_cpu::pool::num_threads(), pre_existing);

    // `RuntimeConfig::install()` is the other channel a caller can ask through;
    // it goes through the same report.
    let mut cfg = rlx_cpu::config::RuntimeConfig::auto_detect();
    cfg.pool_workers = pre_existing + 3;
    cfg.install();
    assert_eq!(rlx_cpu::pool::num_threads(), pre_existing);

    // A value that is not a usable width is not a width request.
    rlx_ir::env::set("RLX_WORKERS", "-5");
    assert_eq!(rlx_cpu::pool::num_threads(), pre_existing);
}

/// The refusal has to be *audible*, not merely non-effective: this re-runs itself
/// in its own process (the pool can be claimed only once per process), claims the
/// pool as a foreign Rayon user, asks RLX for another width, and pins the exact
/// stderr line — said once, not once per ask.
#[test]
fn the_refusal_is_reported_on_stderr() {
    const MARKER: &str = "POOL_WIDTH_REFUSAL_CHILD";
    if std::env::var(MARKER).is_ok() {
        let pre_existing = rayon::current_num_threads();
        rlx_ir::env::set("RLX_WORKERS", (pre_existing + 1).to_string());
        rlx_cpu::pool::ensure_pool();
        rlx_cpu::pool::ensure_pool();
        assert_eq!(rlx_cpu::pool::num_threads(), pre_existing);
        // `RuntimeConfig::install()` supersedes that request while the pool sits
        // at a third width: the line it prints must name the width really in
        // effect, not the installed one, which cannot take effect.
        let mut cfg = rlx_cpu::config::RuntimeConfig::auto_detect();
        cfg.pool_workers = pre_existing + 2;
        cfg.install();
        return;
    }

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "the_refusal_is_reported_on_stderr",
            "--nocapture",
        ])
        .env(MARKER, "1")
        .output()
        .expect("re-run this test in a child process");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stderr}");

    let pre_existing = rayon::current_num_threads();
    let expected = format!(
        "[rlx] cpu pool: {pre_existing} workers in effect, but {} (requested via RLX_WORKERS) could not be applied — the process-wide Rayon pool was already created",
        pre_existing + 1
    );
    assert_eq!(
        stderr.matches(&expected).count(),
        1,
        "stderr was:\n{stderr}"
    );

    let replaced = format!(
        "[rlx] cpu pool: {pre_existing} workers in effect — RuntimeConfig::install() set {}, replacing the RLX_WORKERS={} request",
        pre_existing + 2,
        pre_existing + 1
    );
    assert_eq!(
        stderr.matches(&replaced).count(),
        1,
        "stderr was:\n{stderr}"
    );
}

/// The other half of the contract: while the pool is still unclaimed, the width
/// an installed config asks for *is* the width the pool is created at, even when
/// it supersedes a different `RLX_WORKERS` request — the pool must therefore not
/// be resolved on the way, and nothing may be reported as refused. Runs in a
/// child process because the parent's pool is already sized.
#[test]
fn install_takes_effect_before_the_pool_exists() {
    const MARKER: &str = "POOL_INSTALL_CHILD";
    if std::env::var(MARKER).is_ok() {
        let want = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(4);
        // The width installed, superseding a different environment request —
        // both widths this host must actually be able to run.
        let request = want.saturating_sub(1).max(1);
        rlx_ir::env::set("RLX_WORKERS", request.to_string());
        let mut cfg = rlx_cpu::config::RuntimeConfig::auto_detect();
        cfg.pool_workers = want;
        cfg.install();
        assert_eq!(rlx_cpu::pool::num_threads(), want);
        assert_eq!(rayon::current_num_threads(), want);
        return;
    }

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "install_takes_effect_before_the_pool_exists",
            "--nocapture",
        ])
        .env(MARKER, "1")
        .output()
        .expect("re-run this test in a child process");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stderr}");
    assert!(
        !stderr.contains("[rlx] cpu pool:"),
        "an install that takes effect reported a refusal:\n{stderr}"
    );
}
