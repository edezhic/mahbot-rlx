// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The host copy of a parameter exists only for the reader that needs it.
//!
//! `WgpuExecutable` keeps a parameter's f32 payload on the host for exactly one
//! consumer: the cooperative host-activation mirror, which the CoopF16Vk route builds
//! and which needs the payload of an activation's *source* when that source is a
//! parameter. On an adapter without that route the mirror list is empty, so the copy
//! must not be made — held for the life of the process and never read, it is a
//! standing cost of a model's small parameters.
//!
//! What has to stay true: the stash is empty where the mirror is
//! (`a_stash_exists_only_for_its_reader`); a clone still carries every parameter off the
//! device, because the stash is no longer its record
//! (`a_clone_carries_the_parameters_off_the_device`, and
//! `a_clone_carries_a_packed_parameter` for the storage-width arm); and the stash is
//! *there* where the route exists (`the_mirror_keeps_its_parameter`), which needs a
//! Vulkan adapter with cooperative matrices and skips everywhere else.
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use rlx_wgpu::buffer::set_param_sharing_scope;
use std::sync::Mutex;

mod common;

/// The wgpu device and the `RLX_*` probes are process-global; serialize these tests.
static STASH_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct StashTestGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
impl StashTestGuard {
    fn new() -> Self {
        Self(STASH_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

const M: usize = 64;
const K: usize = 64;
const N: usize = 64;

fn values(n: usize, scale: f32, step: f32) -> Vec<f32> {
    (0..n).map(|i| scale * (i as f32 * step).sin()).collect()
}

/// `c = a · w`, the plainest parameter-carrying graph there is: no activation feeds the
/// multiply, so no host-activation mirror is built for it on any adapter.
fn plain_matmul() -> (Graph, Vec<f32>) {
    let mut g = Graph::new("stash_plain_matmul");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let w = g.param("w", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, w, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);
    (g, values(K * N, 0.02, 0.017))
}

/// The gate's own half: a graph no mirror is built for keeps nothing on the host.
#[test]
fn a_stash_exists_only_for_its_reader() {
    let _guard = StashTestGuard::new();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (g, wv) = plain_matmul();
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &wv);
    assert_eq!(
        exe.host_param_stash_bytes(),
        0,
        "a parameter no host-activation mirror reads is kept on the host anyway"
    );
}

/// A clone recompiles its own arena and has no host record to replay from, so it takes
/// every parameter's bytes off the parent's device storage. The answer must not move:
/// zeroed weights would show up here as a different (and not merely rounded) result.
#[test]
fn a_clone_carries_the_parameters_off_the_device() {
    let _guard = StashTestGuard::new();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let (g, wv) = plain_matmul();
    let av = values(M * K, 0.3, 0.3);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &wv);
    let want = exe.run(&[("a", av.as_slice())]).remove(0);
    let mut clone = exe.clone_for_cache();
    let got = clone.run(&[("a", av.as_slice())]).remove(0);
    assert_eq!(got.len(), want.len());
    let moved = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
    assert_eq!(
        moved, 0.0,
        "a clone computed with parameters its parent's storage holds but it did not take"
    );
}

/// Q8_0 packs 32 elements into 34 bytes, so the weight below is `PACKED_BYTES` in the
/// weight buffer: one byte per element where the plan's f32 slot counts four.
const PK: usize = 256;
const PN: usize = 2;
const PACKED_BYTES: usize = (PK / 32) * 34 * PN;

/// One `DequantMatMul` over a packed parameter — the geometry a fused projection has, and
/// the one a length taken from the plan's f32 slot measures wrong.
fn packed_graph() -> Graph {
    let mut g = Graph::new("stash_packed_clone");
    let x = g.param("x", Shape::new(&[1, PK], DType::F32));
    let w = g.param("w", Shape::new(&[PACKED_BYTES], DType::U8));
    let y = g.add_node(
        rlx_ir::op::Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![x, w],
        Shape::new(&[1, PN], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

/// A packed parameter moves by its storage bytes, which is the one place the weight
/// buffer's layout and the plan's slot disagree: a copy of the slot would write three
/// bytes per element into the buffer beside the parameter.
#[test]
fn a_clone_carries_a_packed_parameter() {
    let _guard = StashTestGuard::new();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // Parking is what puts the parameter in the weight buffer, where its storage is sized
    // by `weight_param_bytes` rather than by the plan.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    set_param_sharing_scope("host-param-stash/packed");
    let bytes = common::packed_gguf_bytes(rlx_gguf::GgmlType::Q8_0, PK, PN);
    assert_eq!(
        bytes.len(),
        PACKED_BYTES,
        "the fixture is the parameter's size"
    );
    let mut exe = WgpuExecutable::compile(packed_graph());
    exe.set_param("x", &values(PK, 0.3, 0.017));
    exe.set_param_bytes("w", &bytes);
    let want = exe.run(&[]);
    // A clone in the parent's own scope finds the parameter resident and moves nothing —
    // the pool's bytes are the answer either way.
    let mut shared = exe.clone_for_cache();
    assert_eq!(
        shared.run(&[]),
        want,
        "a clone that shares the pool answered with something else"
    );
    // A scope of its own, so the clone's compile cannot count the parameter resident in
    // the parent's entry: what the answer comes from is the copy below.
    set_param_sharing_scope("host-param-stash/packed-clone");
    let mut clone = exe.clone_for_cache();
    assert_eq!(
        clone.run(&[]),
        want,
        "a clone computed with a packed parameter it did not take"
    );
    set_param_sharing_scope("");
    rlx_ir::env::unset("RLX_WGPU_ALL_PARAMS_WEIGHT");
}

/// The reader's half of the contract: where the cooperative host-activation mirror
/// exists, the payload it mirrors is in the stash — in the executable that compiled it
/// and in a clone of that executable, which rebuilds the mirror list and has no stash of
/// its own. `run` is the assertion: the mirror panics on a tensor it cannot find, so a
/// stash gated away from a route that reads it fails here rather than returning a wrong
/// answer.
#[test]
fn the_mirror_keeps_its_parameter() {
    let _guard = StashTestGuard::new();
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let dev = rlx_wgpu::device::wgpu_device().expect("rlx-wgpu reports an adapter");
    rlx_ir::env::set("RLX_WGPU_COOP_F16_VK_ENABLE", "1");
    let ready = rlx_wgpu::device::coop_discrete_backend()
        && !rlx_ir::env::flag("RLX_WGPU_NO_COOP_F16_VK")
        && rlx_wgpu::device::coop_f16_16x16_supported()
        && rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&dev.device).is_some();
    if !ready {
        rlx_ir::env::unset("RLX_WGPU_COOP_F16_VK_ENABLE");
        eprintln!("no cooperative f16 Vulkan adapter, skipping");
        return;
    }
    // `relu(w) · b` with both operands parameters: the activation is what the mirror
    // holds, and `w` is the parameter it reads it from. `x` is here so the runs below
    // feed an input like any real call.
    let mut g = Graph::new("stash_mirrored_activation");
    let x = g.input("x", Shape::new(&[M, N], DType::F32));
    let w = g.param("w", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let act = g.relu(w);
    let c = g.matmul(act, b, Shape::new(&[M, N], DType::F32));
    let out = g.add(c, x);
    g.set_outputs(vec![out]);
    let (wv, bv) = (values(M * K, 0.1, 0.017), values(K * N, 0.02, 0.011));
    let xv = values(M * N, 0.01, 0.005);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &wv);
    exe.set_param("b", &bv);
    assert!(
        exe.host_param_stash_bytes() >= wv.len() * 4,
        "the mirror's own parameter is not in the stash it reads"
    );
    let out = exe.run(&[("x", xv.as_slice())]).remove(0);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the mirrored activation produced a non-finite answer"
    );
    // The clone's own compile rebuilds the mirror list, so the payload has to arrive
    // with the parameters it takes off the parent's device.
    let mut clone = exe.clone_for_cache();
    assert!(
        clone.host_param_stash_bytes() >= wv.len() * 4,
        "a clone kept the mirror and not the parameter it reads"
    );
    let cloned = clone.run(&[("x", xv.as_slice())]).remove(0);
    assert_eq!(
        cloned, out,
        "a clone answered with a parameter its mirror did not have"
    );
    // Left set until here because the clone's compile reads it too; a panic between the
    // set and this line leaves it set for the rest of this test binary, which is the
    // four tests of this file.
    rlx_ir::env::unset("RLX_WGPU_COOP_F16_VK_ENABLE");
}
