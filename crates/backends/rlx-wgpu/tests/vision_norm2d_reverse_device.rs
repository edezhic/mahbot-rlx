// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device execution for the two operations the synthesis path used to leave:
//! the channels-first normalisation (`Op::LayerNorm2d`) and the flip that
//! reflect-style padding lowers to (`Op::Reverse`).
//!
//! Both were whole-arena host steps. The normalisation copied the entire
//! activation arena off the device and back per occurrence; the flip staged its
//! operands but still ended the device batch and waited on the device. Every
//! check here pins the *route* through this backend's own schedule output
//! (machine-independent, unlike timing), the parity of each route at the shapes
//! those graphs present, and the active-extent contract of the new steps.
//!
//! This file is its own test process: it neutralises the process-global routing
//! switches before the first compile, so an operator's environment cannot flip
//! the pins. Neutralising writes one fixed value that every check here wants,
//! so the checks may share it.

mod common;
use common::max_abs_diff;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

/// Routing switches that would move either operation off the device, or move
/// the whole schedule to the host route, if the ambient environment asked. The
/// shard-cap lever is neutralised to a value the adapter clamps back to its own
/// limit, so a lowered ambient one cannot stripe these graphs' arenas.
fn pin_default_routing() {
    for key in [
        "RLX_WGPU_FORCE_HOST",
        "RLX_WGPU_DISCRETE_HOST",
        "RLX_WGPU_HOST_NORM",
        "RLX_WGPU_GPU_NORM",
        "RLX_WGPU_SHARD_GPU",
        "RLX_WGPU_ALL_PARAMS_WEIGHT",
        "RLX_WGPU_ALLOW_SHARD",
    ] {
        rlx_ir::env::set(key, "0");
    }
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "1048576");
}

fn device_missing() -> bool {
    rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available())
}

/// NCHW LayerNorm2d with the model file's rank-1 weight/bias as `Param`s.
fn ln2d_graph(c: usize, h: usize, w: usize) -> Graph {
    ln2d_graph_dt(c, h, w, DType::F32)
}

/// The same graph with the weight/bias declared at another width.
fn ln2d_graph_dt(c: usize, h: usize, w: usize, dt: DType) -> Graph {
    let mut g = Graph::new("ln2d_device");
    let x = g.input("x", Shape::new(&[1, c, h, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], dt));
    let beta = g.param("beta", Shape::new(&[c], dt));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    g
}

/// The 1024-channel input that a drifting accumulator cannot survive: a large
/// exact DC offset (`2^10`, so every partial sum stays an integer below `2^24`
/// and is exact in f32) plus small integer deviations. A one-pass
/// `E[x²] − E[x]²` variance cancels ~1.05e6 against a ~22 variance, and an f16
/// or short accumulator drifts much further — either way the result misses
/// 1e-4, which the tree-reduced two-pass device kernel does not.
fn dc_offset_input(len: usize) -> Vec<f32> {
    (0..len).map(|i| 1024.0 + ((i % 17) as f32 - 8.0)).collect()
}

/// The same normalisation with the one-pass `E[x²] − E[x]²` variance, so the
/// parity check below can show that its own input discriminates: if this agreed
/// with the reference inside 1e-4, a drifting accumulator would pass too.
fn naive_variance_output(
    xs: &[f32],
    c: usize,
    hw: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; xs.len()];
    for i in 0..hw {
        let mut sum = 0f32;
        let mut sq = 0f32;
        for ch in 0..c {
            let v = xs[ch * hw + i];
            sum += v;
            sq += v * v;
        }
        let mean = sum / c as f32;
        let var = sq / c as f32 - mean * mean;
        let inv = 1.0 / (var + eps).sqrt();
        for ch in 0..c {
            let v = xs[ch * hw + i];
            out[ch * hw + i] = (v - mean) * inv * gamma[ch] + beta[ch];
        }
    }
    out
}

/// Independent two-pass NCHW normalisation: the mean and the variance of a
/// channel axis are each a full pass, so nothing cancels. Used where the CPU
/// cannot be the reference.
fn two_pass_output(
    xs: &[f32],
    c: usize,
    hw: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; xs.len()];
    for i in 0..hw {
        let mean = (0..c).map(|ch| xs[ch * hw + i]).sum::<f32>() / c as f32;
        let var = (0..c)
            .map(|ch| (xs[ch * hw + i] - mean) * (xs[ch * hw + i] - mean))
            .sum::<f32>()
            / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ch in 0..c {
            out[ch * hw + i] = (xs[ch * hw + i] - mean) * inv * gamma[ch] + beta[ch];
        }
    }
    out
}

/// Reference flip: element `[…, i, …]` on a reversed axis of size `d` reads
/// input `[…, d-1-i, …]`.
fn flip_reference(dims: &[usize], axes: &[usize], x: &[f32]) -> Vec<f32> {
    let rank = dims.len();
    let total: usize = dims.iter().product();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let rev: Vec<bool> = (0..rank).map(|a| axes.contains(&a)).collect();
    (0..total)
        .map(|o| {
            let mut rem = o;
            let mut src = 0usize;
            for ax in 0..rank {
                let i = rem / strides[ax];
                rem %= strides[ax];
                src += if rev[ax] { dims[ax] - 1 - i } else { i } * strides[ax];
            }
            x[src]
        })
        .collect()
}

/// Builds the routing graph: the codec normalisation at `[1, 1024, 1, 144]` plus
/// the two strips the reflect padding flips (rank 4 axis 3, rank 1 axis 0).
fn routing_graph() -> Graph {
    let mut g = Graph::new("routes");
    let x = g.input("x", Shape::new(&[1, 1024, 1, 144], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[1024], DType::F32));
    let beta = g.param("beta", Shape::new(&[1024], DType::F32));
    let n = g.layer_norm2d(x, gamma, beta, 1e-6);
    let s4 = g.input("s4", Shape::new(&[1, 128, 1, 2], DType::F32));
    let r4 = g.reverse(s4, vec![3]);
    let s1 = g.input("s1", Shape::new(&[200], DType::F32));
    let r1 = g.reverse(s1, vec![0]);
    g.set_outputs(vec![n, r4, r1]);
    g
}

/// Both operations lower to device steps in the default configuration, and the
/// whole-arena host steps are absent from the schedule. Nothing else here would
/// notice a silent return to the host route.
///
/// The adapter-dependent half of this — the same pin with the discrete-adapter
/// host policy forced on — is `vision_norm2d_adapter_agnostic.rs`, in its own
/// process because it flips a process-global switch.
#[test]
fn both_ops_are_device_steps_by_default() {
    pin_default_routing();
    if device_missing() {
        return;
    }

    let exe = WgpuExecutable::compile(routing_graph());
    let kinds = exe.step_kind_counts();
    eprintln!("default routing: {kinds:?}");

    assert_eq!(
        kinds.get("layer_norm2d").copied().unwrap_or(0),
        1,
        "Op::LayerNorm2d must be a device step; schedule = {kinds:?}"
    );
    assert_eq!(
        kinds.get("reverse").copied().unwrap_or(0),
        2,
        "Op::Reverse must be a device step; schedule = {kinds:?}"
    );
}

/// The normalisation matches the CPU reference within this backend's 1e-4 norm
/// tolerance at 1024 channels — wider than any existing coverage — on the two
/// shapes the codec blocks use.
#[test]
fn layer_norm2d_matches_cpu_at_1024_channels() {
    pin_default_routing();
    if device_missing() {
        return;
    }

    let c = 1024usize;
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let beta: Vec<f32> = (0..c).map(|i| (i % 5) as f32 * 0.02 - 0.04).collect();

    for &(h, w) in &[(1usize, 144usize), (1, 288)] {
        let xs = dc_offset_input(c * h * w);

        let mut gpu = WgpuExecutable::compile(ln2d_graph(c, h, w));
        assert_eq!(
            gpu.step_kind_counts()
                .get("layer_norm2d")
                .copied()
                .unwrap_or(0),
            1,
            "the parity arm must exercise the device step"
        );
        gpu.set_param("gamma", &gamma);
        gpu.set_param("beta", &beta);
        let got = gpu.run(&[("x", &xs)]).remove(0);

        let mut cpu = Session::new(Device::Cpu).compile(ln2d_graph(c, h, w));
        cpu.set_param("gamma", &gamma);
        cpu.set_param("beta", &beta);
        let want = cpu.run(&[("x", &xs)]).remove(0);

        assert_eq!(got.len(), want.len(), "[1,{c},{h},{w}]: length");
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff <= 1e-4,
            "layer_norm2d wgpu [1,{c},{h},{w}] vs cpu: max abs diff {diff:e} > 1e-4"
        );
        // Anti-vacuity for the bar itself: on this input the one-pass form the
        // device kernel must not use is outside it.
        let naive = naive_variance_output(&xs, c, h * w, &gamma, &beta, 1e-6);
        let naive_diff = max_abs_diff(&naive, &want);
        assert!(
            naive_diff > 1e-4,
            "the [1,{c},{h},{w}] input must expose an unstable accumulation form; \
             a one-pass variance landed {naive_diff:e} from the reference"
        );
        eprintln!(
            "ln2d wgpu [1,{c},{h},{w}]: device max abs diff {diff:.2e}, \
             one-pass form {naive_diff:.2e}"
        );
    }
}

/// The eligibility boundary around a non-F32 weight/bias.
///
/// A parameter is readable as one f32 lane per element when its storage really
/// is that wide: `F64` widens into an eight-byte slot that has room for it, so
/// the device step reads it. A parameter the *flat* plan sized at its native
/// width — a one-byte `Bool`, say — has a slot its own widened upload does not
/// fit, so neither route can read it: that one is refused loudly rather than
/// computed from whatever the upload left behind. (`F16` was this case until the
/// flat plan started reserving the four bytes per element its upload writes; see
/// `tests/host_route_operand_fit.rs`.)
#[test]
fn a_non_f32_weight_and_bias_takes_the_device_route_or_is_refused() {
    pin_default_routing();
    if device_missing() {
        return;
    }
    let (c, h, w) = (16usize, 1usize, 4usize);

    let f64_graph = ln2d_graph_dt(c, h, w, DType::F64);
    let mut gpu = WgpuExecutable::compile(f64_graph);
    let kinds = gpu.step_kind_counts();
    eprintln!("F64 weight/bias route: {kinds:?}");
    assert_eq!(
        kinds.get("layer_norm2d").copied().unwrap_or(0),
        1,
        "an eight-byte parameter is one f32 lane per element and must not be \
         refused or hosted; schedule = {kinds:?}"
    );

    // …and that claim is numerical, so the route pin is not enough on its own:
    // a device step that read the wrong lanes would keep the route and still be
    // accepted. Upload the model's f32 weight/bias the way the typed upload's
    // widening arm does, and compare.
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i % 3) as f32 * 0.25).collect();
    let beta: Vec<f32> = (0..c).map(|i| (i % 4) as f32 * 0.125 - 0.2).collect();
    let xs = dc_offset_input(c * h * w);
    gpu.set_param("gamma", &gamma);
    gpu.set_param("beta", &beta);
    let got = gpu.run(&[("x", &xs)]).remove(0);
    // An independent reference rather than the CPU: the CPU does not read a
    // parameter it is told is F64 out of the f32 lanes this upload writes (the
    // same graph normalises to ~0 there), so the CPU arm would compare against
    // nothing.
    let want = two_pass_output(&xs, c, h * w, &gamma, &beta, 1e-6);
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= 1e-4,
        "F64 weight/bias device route: max abs diff {diff:e} > 1e-4"
    );
    eprintln!("F64 weight/bias device parity: max abs diff {diff:.2e}");

    let msg = compile_refusal(ln2d_graph_dt(c, h, w, DType::Bool));
    assert!(
        msg.contains("REFUSED (parameter not readable as f32 lanes)")
            && msg.contains("LayerNorm2d"),
        "a one-byte parameter whose slot cannot hold its widened upload must be \
         refused loudly; got {msg:?}"
    );
}

/// The other two edges of the envelope: the rank and the encoding. Both routes
/// read `h`/`w` from axes 2 and 3 and both read one f32 lane per element, so
/// anything else has to be refused with its geometry rather than computed from
/// the wrong axes or the wrong lanes.
#[test]
fn a_rank_or_an_encoding_outside_the_envelope_is_refused() {
    pin_default_routing();
    if device_missing() {
        return;
    }

    // Rank 5 is what the whole-arena host step used to compute silently: `h`/`w`
    // from axes 2 and 3, the trailing axis ignored. `rlx-cpu` refuses this rank
    // too.
    let mut g = Graph::new("ln2d_rank5");
    let x = g.input("x", Shape::new(&[1, 8, 1, 4, 2], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[8], DType::F32));
    let beta = g.param("beta", Shape::new(&[8], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    let msg = compile_refusal(g);
    eprintln!("rank 5: {}", msg.lines().next().unwrap_or(""));
    assert!(
        msg.contains("REFUSED (LayerNorm2d rank)"),
        "a rank-5 normalisation must be refused loudly; got {msg:?}"
    );

    // A complex operand: its slot packs several lanes per element, which neither
    // route reads that way.
    let mut g = Graph::new("ln2d_complex");
    let x = g.input("x", Shape::new(&[1, 8, 1, 4], DType::C128));
    let gamma = g.param("gamma", Shape::new(&[8], DType::C128));
    let beta = g.param("beta", Shape::new(&[8], DType::C128));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    let msg = compile_refusal(g);
    eprintln!("complex: {}", msg.lines().next().unwrap_or(""));
    assert!(
        msg.contains("REFUSED (LayerNorm2d encoding)"),
        "a complex normalisation must be refused rather than read lane-wise; got {msg:?}"
    );
}

/// Compile in-process and return the message the refusal panicked with.
fn compile_refusal(graph: Graph) -> String {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuExecutable::compile(graph)
    }));
    let Err(payload) = caught else {
        panic!("the graph was accepted; it must be refused loudly");
    };
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// The flip is a pure permutation, so it must match the CPU reference exactly —
/// at the strip shapes the padding produces, and at the ranks the existing
/// coverage pins (1, 3 and 4), including the empty-axes identity and a
/// duplicated axis.
#[test]
fn reverse_matches_cpu_exactly_at_the_padding_shapes() {
    pin_default_routing();
    if device_missing() {
        return;
    }

    let cases: [(Vec<usize>, Vec<usize>); 10] = [
        // Speech encoder: rank 4, axis 3, the later blocks' strips.
        (vec![1, 128, 1, 2], vec![3]),
        (vec![1, 64, 1, 2], vec![3]),
        (vec![1, 64, 1, 3], vec![3]),
        (vec![1, 64, 1, 4], vec![3]),
        // Mel front-end: rank 1, axis 0, 200 elements a side.
        (vec![200], vec![0]),
        // Ranks 3 and 4 with a sequence axis.
        (vec![3, 4, 2], vec![1]),
        (vec![2, 3, 4, 5], vec![3]),
        // Empty axes is the identity, and a duplicated axis is idempotent.
        (vec![2, 3, 4, 5], vec![]),
        (vec![2, 3, 4, 5], vec![1, 1, 3]),
        (vec![4, 3, 2], vec![2]),
    ];

    for (dims, axes) in cases {
        let xs: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|i| (i as f32) * 0.25 - 3.0)
            .collect();
        let mut g = Graph::new("reverse_device");
        let x = g.input("x", Shape::new(&dims, DType::F32));
        let y = g.reverse(x, axes.clone());
        g.set_outputs(vec![y]);
        let mut exe = WgpuExecutable::compile(g);
        assert_eq!(
            exe.step_kind_counts().get("reverse").copied().unwrap_or(0),
            1,
            "{dims:?} axes {axes:?}: parity arm must exercise the device step"
        );
        let got = exe.run(&[("x", &xs)]).remove(0);
        let want = flip_reference(&dims, &axes, &xs);
        assert_eq!(
            got, want,
            "reverse {dims:?} axes {axes:?}: must be bit-exact"
        );
    }
}

/// The new steps must be marked safe for the active extent, or they silently
/// disable the bucketed/padded-decode path for any graph containing them — and
/// the flip must not scale an axis it reverses.
#[test]
fn device_steps_keep_the_active_extent_path() {
    pin_default_routing();
    if device_missing() {
        return;
    }

    // A flip away from the leading (bucket) axis may be scaled; a flip ON it
    // must stay at full extent, so the whole output has to be right there.
    for (axes, whole) in [(vec![2usize], false), (vec![0usize], true)] {
        let dims = [4usize, 3, 2];
        let mut g = Graph::new("reverse_active_extent");
        let x = g.input("x", Shape::new(&dims, DType::F32));
        let y = g.reverse(x, axes.clone());
        g.set_outputs(vec![y]);
        let mut exe = WgpuExecutable::compile(g);
        assert!(
            exe.test_all_steps_safe_for_active_extent(),
            "axes {axes:?}: the device flip must not veto the active-extent path"
        );
        assert_eq!(
            exe.step_kind_counts().get("reverse").copied().unwrap_or(0),
            1,
            "axes {axes:?}: the check must exercise the device step"
        );

        exe.set_active_extent(Some((2, 4)));
        let xs: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let got = exe.run(&[("x", &xs)]).remove(0);
        let want = flip_reference(&dims, &axes, &xs);
        let checked = if whole { 24 } else { 12 };
        assert!(got.len() >= checked, "axes {axes:?}: short output");
        assert_eq!(
            &got[..checked],
            &want[..checked],
            "axes {axes:?}: active extent must keep the flipped axis at full extent"
        );
    }

    // The normalisation never scales, so it stays correct under the extent and
    // still must not veto it.
    let c = 64usize;
    let mut g = Graph::new("ln2d_active_extent");
    let x = g.input("x", Shape::new(&[2, c, 2, 3], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    assert!(
        exe.test_all_steps_safe_for_active_extent(),
        "LayerNorm2d must not veto the active-extent path"
    );
    assert_eq!(
        exe.step_kind_counts()
            .get("layer_norm2d")
            .copied()
            .unwrap_or(0),
        1,
        "the check must exercise the device step"
    );
    let gamma_v: Vec<f32> = (0..c).map(|i| 1.0 + (i % 3) as f32 * 0.05).collect();
    let beta_v: Vec<f32> = (0..c).map(|i| (i % 4) as f32 * 0.02).collect();
    exe.set_param("gamma", &gamma_v);
    exe.set_param("beta", &beta_v);
    exe.set_active_extent(Some((1, 2)));
    let xs = dc_offset_input(2 * c * 2 * 3);
    let got = exe.run(&[("x", &xs)]).remove(0);
    let want = {
        let mut cpu = Session::new(Device::Cpu).compile(ln2d_graph_active(c));
        cpu.set_param("gamma", &gamma_v);
        cpu.set_param("beta", &beta_v);
        cpu.run(&[("x", &xs)]).remove(0)
    };
    assert_eq!(got.len(), want.len());
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= 1e-4,
        "LayerNorm2d under an active extent: max abs diff {diff:e} > 1e-4"
    );
}

/// The CPU twin of [`ln2d_graph`]-with-a-batch: `[2, c, 2, 3]`.
fn ln2d_graph_active(c: usize) -> Graph {
    let mut g = Graph::new("ln2d_device_active");
    let x = g.input("x", Shape::new(&[2, c, 2, 3], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    g
}
