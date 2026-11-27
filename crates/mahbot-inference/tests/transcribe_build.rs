// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a build that cannot finish must do: fail naming the weight it could not
//! resolve — never run the model on the zeros a placeholder carried — and leave the
//! thread exactly as it found it, so the next build is not shaped by the failed one.
//!
//! Both checks are model-dependent (they build and compile a real recognition graph), so
//! without the installed model directory or without a wgpu adapter the test says why it
//! skipped. They are one test on purpose: the state a failed compile leaves is thread-local,
//! and cargo gives each `#[test]` its own thread, so a second test could not see the first
//! one's leak.

use std::sync::Arc;

use mahbot_inference::emit;
use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Pair, Role};

/// The engine's view of a failed build, on the thread that ran it.
#[test]
fn an_unresolvable_weight_fails_and_a_failed_build_leaves_nothing_behind() {
    // The adapter first: a host that has the models but no wgpu device skips here rather
    // than failing inside the compile this test is about.
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    mahbot_inference::device::pin_gpu();
    let Ok(dir) = loader::models_dir() else {
        eprintln!(
            "skipped: the model directory is not installed, so the build's failure paths were \
             not checked"
        );
        return;
    };
    let Some((body, sound)) = loader::pair_files(Role::Recognition) else {
        eprintln!("skipped: the model directory names no ASR pair");
        return;
    };
    if !dir.join(body).exists() || !dir.join(sound).exists() {
        eprintln!("skipped: the ASR pair is not under {}", dir.display());
        return;
    }
    let pair = Arc::new(Pair::open(dir.join(body), dir.join(sound)).expect("the ASR pair opens"));
    // A body graph, not the sound encoder: the encoder's weights are half-precision tensors
    // the build hands over at the file's width, while the body's packed weights are
    // placeholders the compile binds from the file the build's record names — the case this
    // test is about.
    let built =
        graphs::body_new_rows_head_chain(pair.body.clone(), 8, 0, false).expect("the body builds");
    // What a placeholder stands in for: the file the build's record names — the same file's
    // bytes, whether it serves them packed, as a fused plan of its own tensors, or as host
    // f32 this process kept. Released between the build and its compile, exactly as a caller
    // that dropped the model would leave them.
    emit::clear_weight_caches();
    // The build's own placeholder record, read before the compile clears it: which parameters
    // the refusal must name is the file's answer (the joined q+k on the pinned files, another
    // join or projection on a file whose scheme layout differs), not a spelling here.
    let placeholders = emit::placeholders();
    assert!(
        !placeholders.is_empty(),
        "the body's packed weights are placeholders the compile has to resolve"
    );
    let error = match graphs::Compiled::new(built) {
        Ok(_) => panic!("a graph whose weights are gone compiled anyway"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("placeholder"),
        "the refusal says the bytes were a placeholder: {error}"
    );
    assert!(
        placeholders
            .iter()
            .any(|name| name.starts_with("blk.0.") && error.contains(name)),
        "the refusal names the body's first parameter, not just the graph: {error}"
    );
    // The failed compile left nothing behind: a name still marked a placeholder would refuse a
    // later build that carries its own bytes.
    for name in &placeholders {
        assert!(
            !emit::is_placeholder(name),
            "a failed compile left `{name}` in the build's placeholder record"
        );
    }
    // Nothing of the failed build survives it: the same graph, built again, compiles.
    let again = graphs::body_new_rows_head_chain(pair.body.clone(), 8, 0, false)
        .expect("the body builds again");
    let compiled = graphs::Compiled::new(again).expect("a build after a failed build compiles");
    assert!(
        emit::scope_of(placeholders.iter().map(String::as_str)).is_none(),
        "a successful compile left the build's origin record behind"
    );
    drop(compiled);
}
