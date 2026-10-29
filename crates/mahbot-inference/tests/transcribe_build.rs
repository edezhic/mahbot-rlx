// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a build that cannot finish must do: fail naming the weight it could not
//! resolve — never run the model on the zeros a placeholder carried — and leave the
//! thread exactly as it found it, so the next build is not shaped by the failed one.
//!
//! Both checks are model-dependent (they build and compile a real recognition graph), so
//! without `MAHBOT_MODELS_DIR` or without a wgpu adapter the test says why it skipped. They
//! are one test on purpose: the state a failed compile leaves is thread-local, and cargo
//! gives each `#[test]` its own thread, so a second test could not see the first one's leak.

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
    let Ok(dir) = loader::models_dir() else {
        eprintln!(
            "skipped: {} is not set, so the build's failure paths were not checked",
            loader::MODELS_DIR_ENV
        );
        return;
    };
    let Some((body, sound)) = loader::pair_files(Role::Recognition) else {
        eprintln!("skipped: {} names no ASR pair", loader::MODELS_DIR_ENV);
        return;
    };
    if !dir.join(body).exists() || !dir.join(sound).exists() {
        eprintln!("skipped: the ASR pair is not under {}", dir.display());
        return;
    }
    let pair = Arc::new(Pair::open(dir.join(body), dir.join(sound)).expect("the ASR pair opens"));
    // A body graph, not the sound encoder: the encoder's weights are host f16 the build
    // carries itself, while the body's packed weights are placeholders the compile binds
    // from the process's caches — the case this test is about.
    let built =
        graphs::body_new_rows_head_chain(pair.body.clone(), 8, 0, false).expect("the body builds");
    // What a placeholder stands in for: the model file's mapping, the host copies the
    // process kept and the fused blob. Released between the build and its compile,
    // exactly as a caller that dropped the model would leave them.
    emit::clear_weight_caches();
    let error = match graphs::Compiled::new(rlx_runtime::Device::Gpu, built) {
        Ok(_) => panic!("a graph whose weights are gone compiled anyway"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("placeholder"),
        "the refusal says the bytes were a placeholder: {error}"
    );
    assert!(
        error.contains("weight `") && error.contains(".weight"),
        "the refusal names the operand, not just the graph: {error}"
    );
    // The failed compile left nothing behind: a name still marked a placeholder would refuse a
    // later build that carries its own bytes.
    assert!(
        !emit::is_placeholder("blk.0.attn_q.weight"),
        "a failed compile left the build's placeholder record behind"
    );
    // Nothing of the failed build survives it: the same graph, built again, compiles.
    let again = graphs::body_new_rows_head_chain(pair.body.clone(), 8, 0, false)
        .expect("the body builds again");
    let compiled = graphs::Compiled::new(rlx_runtime::Device::Gpu, again)
        .expect("a build after a failed build compiles");
    assert!(
        emit::scope_of(std::iter::once("blk.0.attn_q.weight")).is_none(),
        "a successful compile left the build's origin record behind"
    );
    drop(compiled);
}
