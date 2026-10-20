// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared HIR → MIR → LIR compile stages for runtime backends.

use rlx_driver::Device;
use rlx_ir::Graph;
use rlx_ir::GraphModule;
use rlx_ir::OpKind;
use rlx_ir::hir::HirModule;
use rlx_ir::lir::LirModule;
use rlx_opt::{CompilePipeline, CompileResult, FusionOptions, FusionReport, FusionTarget};

use crate::CompileOptions;

/// Map a runtime [`Device`] to the fusion pass pipeline target.
pub fn fusion_target_for(device: Device) -> FusionTarget {
    match device {
        Device::Cpu => FusionTarget::Cpu,
        Device::Metal => FusionTarget::Metal,
        Device::Mlx => FusionTarget::Mlx,
        Device::Cuda => FusionTarget::Cuda,
        Device::Rocm => FusionTarget::Rocm,
        Device::Gpu | Device::Vulkan | Device::OneApi => FusionTarget::Wgpu,
        Device::Tpu => FusionTarget::Tpu,
        // CoreML runs its own graph optimizer, so we want minimal RLX-side
        // fusion: the CPU fusion target only synthesizes Fused* ops the
        // backend *claims*. Claimed fused forms without a MIL arm are
        // host-segmented or expanded in `CoremlExecutable::compile_with_options`.
        Device::Ane => FusionTarget::Cpu,
        // Other devices without a dedicated fusion pipeline: CPU patterns
        // (safe superset for graph-level rewrites).
        _ => FusionTarget::Cpu,
    }
}

/// The activation codes the region kernel that runs on `device` implements, or `None`
/// when this tree does not declare that device's kernel.
///
/// Keyed to the *kernel*, not to the fusion bucket: `Device::Vulkan` and `Device::OneApi`
/// share [`FusionTarget::Wgpu`] but run their own region engines, which decompose every
/// region, and keep their present fusion decisions. What the set means is documented at
/// [`rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS`]; the judgement the
/// fusion passes apply is `rlx_fusion::region_kernel`.
///
/// The device is the only thing that knows which kernel will execute a region, so
/// [`pipeline_for`] takes this function's answer unconditionally: a set a caller put in
/// `CompileOptions::fusion_opts` cannot narrow another device's fusion.
pub fn region_kernel_activation_subs_for(device: Device) -> Option<&'static [u32]> {
    match device {
        Device::Gpu => Some(rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS),
        _ => None,
    }
}

/// Build a [`CompilePipeline`] from session/device options.
pub fn pipeline_for(device: Device, options: &CompileOptions) -> CompilePipeline {
    let target = options
        .fusion_target
        .unwrap_or_else(|| fusion_target_for(device));
    let mut opts = options.fusion_opts.merge_env();
    if !opts.keep_elementwise_regions {
        if matches!(target, FusionTarget::Cpu | FusionTarget::Mlx)
            && !opts.unfuse_elementwise_regions
        {
            // MLX ElementwiseRegion modulus tiling broadcasts `[1,C,1]` onto
            // `[1,C,T]` as flat `gid % C` (varies across time) instead of
            // channel-wise `bias[c]`. That corrupts AdaIN / Conv bias on
            // Kokoro/StyleTTS2 and drops MLX↔CPU cosine to ~0.06. Unfuse like
            // Metal/WGPU/CUDA so Binary+Expand use normal MLX broadcast.
            opts.unfuse_elementwise_regions = true;
        }
        if matches!(target, FusionTarget::Metal) {
            let metal_env = FusionOptions::from_metal_env();
            if metal_env.skip_fusion {
                opts.skip_fusion = true;
            }
            if !opts.unfuse_elementwise_regions {
                opts.unfuse_elementwise_regions = true;
            }
        }
        if matches!(target, FusionTarget::Wgpu) && !opts.unfuse_elementwise_regions {
            opts.unfuse_elementwise_regions = true;
        }
        if matches!(target, FusionTarget::Cuda | FusionTarget::Rocm)
            && !opts.unfuse_elementwise_regions
        {
            opts.unfuse_elementwise_regions = true;
        }
    }
    let mut pipe = CompilePipeline::new(target);
    pipe.opts = opts;
    pipe.opts = pipe.opts.apply_native_fk_defaults(target);
    // The device decides, not a caller-supplied set: only the device says which kernel
    // will execute a region (see `region_kernel_activation_subs_for`).
    pipe.opts.region_kernel_activation_subs = region_kernel_activation_subs_for(device);
    // One shared resolver: this used to be its own copy of the target-default
    // line, and a clamp added to the other copy simply never ran here.
    pipe.opts.fusion_limits =
        rlx_opt::rlx_compile::fusion_pipeline::resolve_fusion_limits(target, &pipe.opts);
    pipe.arena_alignment = options.arena_alignment;
    pipe.assert_fusion_clean = options.assert_fusion_clean;
    // Device token for legalize errors — Vulkan / OneAPI share Wgpu fusion
    // patterns but must not report themselves as `"wgpu"`.
    pipe.backend_label = Some(device.as_arg());
    if let Some(ops) = options.supported_ops {
        pipe.supported_ops = Some(ops);
    } else if let Some(backend) = crate::registry::backend_for(device) {
        let ops = backend.supported_ops();
        if !ops.is_empty() {
            pipe.supported_ops = Some(ops);
        }
    }
    pipe.kernel_dispatch = options.kernel_dispatch;
    pipe.lint_numerics = options.lint_numerics;
    pipe.fusion_report = options.fusion_report;
    pipe.no_io_peaks_output = options.no_io_peaks_output;
    pipe.opts.disable_conv_bias_act_fusion = options.disable_conv_bias_act_fusion;
    pipe.opts.no_io_peaks_output = options.no_io_peaks_output;
    pipe
}

/// Attach a backend op claim set for backend-aware fusion.
pub fn options_with_supported_ops(
    options: &CompileOptions,
    supported_ops: &'static [OpKind],
) -> CompileOptions {
    let mut opts = options.clone();
    opts.supported_ops = Some(supported_ops);
    opts
}

/// Run the MIR fusion pipeline on a graph and return optimized LIR +
/// fusion diagnostics.
pub fn compile_graph_stages(
    device: Device,
    graph: Graph,
    options: &CompileOptions,
) -> CompileResult {
    let pipe = pipeline_for(device, options);
    maybe_specialize(pipe.compile_graph(graph), &pipe, options)
}

fn maybe_specialize(
    result: CompileResult,
    pipe: &CompilePipeline,
    options: &CompileOptions,
) -> CompileResult {
    match &options.dim_binding {
        Some(binding) => result.specialize(pipe, binding),
        None => result,
    }
}

/// Same as [`compile_graph_stages`] with an explicit backend op claim.
pub fn compile_graph_stages_for_backend(
    device: Device,
    graph: Graph,
    options: &CompileOptions,
    supported_ops: &'static [OpKind],
) -> CompileResult {
    let opts = options_with_supported_ops(options, supported_ops);
    compile_graph_stages(device, graph, &opts)
}

/// HIR → LIR with fusion diagnostics.
pub fn compile_hir_stages(
    device: Device,
    hir: HirModule,
    options: &CompileOptions,
) -> Result<CompileResult, rlx_ir::hir::LowerError> {
    let pipe = pipeline_for(device, options);
    // Mirror backend `compile(Graph)`: bake `param_bindings` + DCE/fold
    // *before* fusion so FuseAdaLayerNorm sees Constants (ONNX affine-free
    // LN γ/β and scalar 1) instead of Params.
    let mir = CompilePipeline::lower_hir(hir)?;
    let graph = crate::precompile::precompile_cleanup(mir.into_graph(), options);
    Ok(maybe_specialize(pipe.compile_graph(graph), &pipe, options))
}

/// Compile a [`GraphModule`] (HIR, MIR, or LIR stage) through the pipeline.
pub fn compile_module_stages(
    device: Device,
    module: GraphModule,
    options: &CompileOptions,
) -> Result<CompileResult, rlx_ir::hir::LowerError> {
    match module.stage() {
        rlx_ir::GraphStage::Hir => {
            let hir = module
                .into_hir()
                .expect("GraphModule stage() / into_hir mismatch");
            compile_hir_stages(device, hir, options)
        }
        _ => {
            let pipe = pipeline_for(device, options);
            pipe.compile_module(module)
                .map(|r| maybe_specialize(r, &pipe, options))
        }
    }
}

/// Print fusion diagnostics when `RLX_FUSION_REPORT=1`.
pub fn maybe_log_fusion(report: &FusionReport) {
    if rlx_ir::env::flag("RLX_FUSION_REPORT") {
        eprintln!("{report}");
    }
    // Structured mirror for the observability/audit spine (no-op unless the
    // `tracing` feature is on). Metadata only — fusion stats, never tensor data.
    crate::obs::event("compile.fusion", format_args!("{report}"));
}

/// Extract the optimized graph from LIR (convenience for backends that
/// replan memory after precision rewrites).
pub fn graph_from_lir(lir: LirModule) -> Graph {
    lir.into_graph()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device→kernel keying, pinned: this is the glue that decides which labels narrow
    /// their fusion, so deleting it must not leave the suite green. `None` where the kernel
    /// is not the wgsl one is the "no other device's fusion decisions change" half, and the
    /// third assertion is the other half of `region_kernel_activation_subs_for`'s claim —
    /// the device outranks a set a caller put in `CompileOptions::fusion_opts`.
    #[test]
    fn only_the_labels_that_run_the_wgsl_region_kernel_get_its_activation_set() {
        use rlx_ir::region_encode::WGSL_REGION_KERNEL_ACTIVATION_SUBS;

        let wgsl = Some(WGSL_REGION_KERNEL_ACTIVATION_SUBS);
        let caller = CompileOptions::new().fusion_opts(FusionOptions {
            region_kernel_activation_subs: Some(&[3]),
            ..Default::default()
        });
        for (device, expected) in [
            (Device::Gpu, wgsl),
            (Device::Cpu, None),
            (Device::Metal, None),
            // Same `FusionTarget` as `Gpu`, different region engine.
            (Device::Vulkan, None),
            (Device::OneApi, None),
            (Device::Cuda, None),
            (Device::Rocm, None),
        ] {
            let installed = |opts: &CompileOptions| {
                pipeline_for(device, opts)
                    .opts
                    .region_kernel_activation_subs
            };
            assert_eq!(
                region_kernel_activation_subs_for(device),
                expected,
                "{device:?} either runs kernels/elementwise_region.wgsl or a kernel that \
                 keeps its present fusion decisions, and this mapping says which"
            );
            assert_eq!(
                installed(&CompileOptions::new()),
                expected,
                "the judgement a {device:?} compile runs under"
            );
            assert_eq!(
                installed(&caller),
                expected,
                "a caller-supplied set survived into a {device:?} compile, whose kernel is \
                 not the one it describes"
            );
        }
    }
}
